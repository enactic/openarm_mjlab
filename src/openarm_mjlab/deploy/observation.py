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

"""Rebuild the door task's observation vector using ONLY plain MuJoCo.

No mjlab, no mujoco-warp. This is what a deployment target has to do, and it is
the part most likely to be subtly wrong -- so verify_obs_parity.py checks it
element-by-element against mjlab at identical states before any transfer number
is trusted.

Layout (49 dims), from the door task's actor observation group:
  joint_pos(18)  robot qpos - default qpos
  joint_vel(18)  robot qvel  (default qvel is zero)
  door_angle(1)  hinge qpos - default
  ee_to_handle(3) base-frame vector, cage centre -> handle site
  handle_contact(1) 1.0 if either fingertip touches the handle geom
  actions(8)     previous raw action
"""

from __future__ import annotations

import mujoco
import numpy as np

GRASP_LOCAL_OFFSET = np.array([0.0, 0.0, -0.135])


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by quaternion q (w, x, y, z), matching mjlab's convention."""
    w, x, y, z = q
    u = np.array([x, y, z])
    return (
        2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


class PlainObsBuilder:
    """Resolves every index once, then rebuilds the observation each step."""

    def __init__(self, model: mujoco.MjModel):
        """Resolve joint, site, body and sensor indices once from the model."""
        self.m = model

        def name(obj_type, index):
            return mujoco.mj_id2name(model, obj_type, index) or ""

        # Robot joints, in model order, excluding the door hinge.
        self.robot_qpos_adr, self.robot_dof_adr, self.robot_joints = [], [], []
        for j in range(model.njnt):
            n = name(mujoco.mjtObj.mjOBJ_JOINT, j)
            if n.startswith("robot/"):
                self.robot_joints.append(n)
                self.robot_qpos_adr.append(model.jnt_qposadr[j])
                self.robot_dof_adr.append(model.jnt_dofadr[j])
        self.door_joint = [
            j
            for j in range(model.njnt)
            if name(mujoco.mjtObj.mjOBJ_JOINT, j).endswith("door_hinge")
        ][0]
        self.door_qpos_adr = model.jnt_qposadr[self.door_joint]

        self.ee_site = [
            s
            for s in range(model.nsite)
            if name(mujoco.mjtObj.mjOBJ_SITE, s).endswith("right_ee_control_point")
        ][0]
        self.handle_site = [
            s
            for s in range(model.nsite)
            if name(mujoco.mjtObj.mjOBJ_SITE, s).endswith("handle_site")
        ][0]
        self.base_body = [
            b
            for b in range(model.nbody)
            if name(mujoco.mjtObj.mjOBJ_BODY, b).endswith("robot/openarm_body")
        ]
        if not self.base_body:
            # fall back to the robot's root body
            self.base_body = [
                b
                for b in range(model.nbody)
                if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith("robot/")
            ][:1]
        self.base_body = self.base_body[0]

        self.contact_sensors = [
            s
            for s in range(model.nsensor)
            if "finger_grip_contact" in (name(mujoco.mjtObj.mjOBJ_SENSOR, s))
        ]
        self.sensor_adr = [model.sensor_adr[s] for s in self.contact_sensors]

        # Default pose comes from the exported keyframe.
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
        self.default_qpos = np.array(model.key_qpos[key])
        self.default_ctrl = np.array(model.key_ctrl[key])

    def build(self, data: mujoco.MjData, last_action: np.ndarray) -> np.ndarray:
        """Return the 49-dim observation for the current state."""
        jp = np.array([data.qpos[a] for a in self.robot_qpos_adr])
        jp0 = np.array([self.default_qpos[a] for a in self.robot_qpos_adr])
        jv = np.array([data.qvel[a] for a in self.robot_dof_adr])
        door = np.array(
            [data.qpos[self.door_qpos_adr] - self.default_qpos[self.door_qpos_adr]]
        )

        ee_pos = np.array(data.site_xpos[self.ee_site])
        ee_mat = np.array(data.site_xmat[self.ee_site]).reshape(3, 3)
        ee_quat = np.empty(4)
        mujoco.mju_mat2Quat(ee_quat, ee_mat.flatten())
        tool = ee_pos + _quat_apply(ee_quat, GRASP_LOCAL_OFFSET)
        handle = np.array(data.site_xpos[self.handle_site])
        vec_w = handle - tool
        base_quat = np.array(data.xquat[self.base_body])
        ee_to_handle = _quat_apply(_quat_inv(base_quat), vec_w)

        found = 0.0
        for adr in self.sensor_adr:
            if data.sensordata[adr] > 0:
                found = 1.0
                break
        return np.concatenate(
            [jp - jp0, jv, door, ee_to_handle, [found], last_action]
        ).astype(np.float32)
