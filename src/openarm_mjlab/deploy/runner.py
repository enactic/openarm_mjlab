# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run an exported policy in PLAIN MuJoCo -- no mjlab, no mujoco-warp.

This is the deployment-shaped path: load an MJCF, load an ONNX policy, rebuild
the observation, step the CPU physics. The success criterion is replicated from
the task's own termination term, including its stateful bookkeeping:

  swing_gained  >= 1.0 rad, measured from the episode's start angle
  gained-under-contact >= 0.85 * target  (accrued only while a pad touches)
  |hinge rate|  <  1.0 rad/s
  a fingertip is touching the handle at that instant
"""

from __future__ import annotations

import mujoco
import numpy as np
import onnxruntime as ort

TARGET_SWING = 1.0
MAX_SWING_RATE = 1.0
HONEST_FRAC = 0.85


class PlainRunner:
    """Drives an exported ONNX policy against a plain-MuJoCo model."""

    @staticmethod
    def apply_sim_options(m: mujoco.MjModel, sim_cfg) -> None:
        """Copy mjlab's solver configuration onto a plain MuJoCo model.

        The exported MJCF carries NO <option> block, so a plain load silently
        uses MuJoCo's defaults -- Euler instead of implicitfast, a pyramidal
        friction cone instead of elliptic, impratio 1 instead of 10. Measured:
        leaving them at defaults produced a 20.3-point "sim-to-sim gap" that
        was really just two different physics configurations.
        """
        m.opt.timestep = sim_cfg.timestep
        m.opt.impratio = sim_cfg.impratio
        m.opt.iterations = sim_cfg.iterations
        m.opt.tolerance = sim_cfg.tolerance
        m.opt.ls_iterations = sim_cfg.ls_iterations
        m.opt.ls_tolerance = sim_cfg.ls_tolerance
        m.opt.gravity[:] = sim_cfg.gravity
        m.opt.integrator = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }[str(sim_cfg.integrator)]
        m.opt.cone = {
            "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
            "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
        }[str(sim_cfg.cone)]
        m.opt.solver = {
            "pgs": mujoco.mjtSolver.mjSOL_PGS,
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[str(sim_cfg.solver)]

    def __init__(
        self, xml: str, onnx_path: str, builder, decimation: int = 4, sim_cfg=None
    ):
        """Load the model and the ONNX policy, and resolve the actuator order."""
        self.m = mujoco.MjModel.from_xml_path(xml)
        if sim_cfg is not None:
            self.apply_sim_options(self.m, sim_cfg)
        self.d = mujoco.MjData(self.m)
        self.b = builder
        self.decimation = decimation
        self.dt = self.m.opt.timestep * decimation
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

        def name(obj_type, index):
            return mujoco.mj_id2name(self.m, obj_type, index) or ""

        # the 8 actuators the policy drives, in the action term's order
        order = [f"robot/openarm_right_joint{i}" for i in range(1, 8)]
        order.append("robot/openarm_right_finger_joint1")
        self.act_ids = []
        for want in order:
            hits = [
                a
                for a in range(self.m.nu)
                if name(mujoco.mjtObj.mjOBJ_ACTUATOR, a) == want
            ]
            if not hits:
                raise KeyError(f"actuator not found in the exported scene: {want}")
            self.act_ids.append(hits[0])

    def reset(
        self, rng: np.random.Generator, joint_jitter: float = 0.05, soft_limits=None
    ):
        """Match the task's reset events: jitter, CLAMPED to soft limits.

        The clamp is not cosmetic. mjlab's reset_joints_by_offset clamps the
        jittered position to the soft joint limits; a replica without it starts
        episodes mjlab never generates. Measured: omitting the clamp produced a
        9.4-point apparent sim-to-sim gap, while a paired test from identical
        initial states showed 100% agreement between the two backends.
        """
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        for k, adr in enumerate(self.b.robot_qpos_adr):
            self.d.qpos[adr] += rng.uniform(-joint_jitter, joint_jitter)
            if soft_limits is not None:
                lo, hi = soft_limits[k]
                self.d.qpos[adr] = min(max(self.d.qpos[adr], lo), hi)
        # reset_door: offset in [0, 0.05], matching the task's event
        self.d.qpos[self.b.door_qpos_adr] = self.b.default_qpos[
            self.b.door_qpos_adr
        ] + rng.uniform(0.0, 0.05)
        self.d.qvel[:] = 0.0
        self.d.ctrl[:] = self.b.default_ctrl
        mujoco.mj_forward(self.m, self.d)
        self.last_action = np.zeros(8, dtype=np.float32)
        self.start_angle = float(self.d.qpos[self.b.door_qpos_adr])
        self.prev_angle = self.start_angle
        self.gained_contact = 0.0

    def _contact(self) -> float:
        return 1.0 if any(self.d.sensordata[a] > 0 for a in self.b.sensor_adr) else 0.0

    def step(self, scale: np.ndarray, offset: np.ndarray):
        """Advance one control step and return (observation, angle, contact)."""
        obs = self.b.build(self.d, self.last_action)
        action = self.sess.run(None, {self.in_name: obs[None, :].astype(np.float32)})[
            0
        ][0]
        action = np.clip(action, -100.0, 100.0)
        self.last_action = action.astype(np.float32)
        target = offset + scale * action
        for k, a in enumerate(self.act_ids):
            self.d.ctrl[a] = target[k]
        for _ in range(self.decimation):
            mujoco.mj_step(self.m, self.d)

        angle = float(self.d.qpos[self.b.door_qpos_adr])
        contact = self._contact()
        self.gained_contact += max(angle - self.prev_angle, 0.0) * contact
        self.prev_angle = angle
        return obs, angle, contact

    def succeeded(self) -> bool:
        """Return True when the task's own success criterion is satisfied."""
        angle = float(self.d.qpos[self.b.door_qpos_adr])
        gained = max(angle - self.start_angle, 0.0)
        rate = abs(float(self.d.qvel[self.m.jnt_dofadr[self.b.door_joint]]))
        return bool(
            gained >= TARGET_SWING
            and self.gained_contact >= HONEST_FRAC * TARGET_SWING
            and rate < MAX_SWING_RATE
            and self._contact() > 0
        )
