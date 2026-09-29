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

"""Compiled-scene checks for the held-high curriculum grasp."""

import mujoco
import numpy as np
import pytest

import openarm_mjlab.tasks  # noqa: F401
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from openarm_mjlab.openarm_bimanual import GRASP_LOCAL_OFFSET
from openarm_mjlab.tasks.bimanual_lift import mdp


@pytest.fixture
def held_scene():
    cfg = load_env_cfg("OpenArm-BimanualLift")
    cfg.scene.num_envs = 1
    model = Scene(cfg.scene, device="cpu").compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    adr = model.joint("bar/bar_free").qposadr[0]
    data.qpos[adr + 2] += mdp.HELD_HIGH_RAISE
    mujoco.mj_forward(model, data)
    return model, data


def bar_contacts(model, data):
    return [
        (tuple(model.geom(g).name for g in c.geom), c.dist)
        for c in data.contact
        if any(model.geom(g).name.startswith("bar/") for g in c.geom)
    ]


def test_held_high_pose_matches_bar_and_only_contacts_pads(held_scene):
    model, data = held_scene
    contacts = bar_contacts(model, data)
    assert contacts
    for side, sign in (("right", -1), ("left", 1)):
        site = data.site(f"robot/{side}_ee_control_point")
        rotation = site.xmat.reshape(3, 3)
        # Allow intentional palm clearance, not an arbitrary IK target offset.
        grip_offset = np.array(GRASP_LOCAL_OFFSET)
        grip_offset[2] -= mdp.HELD_HIGH_TOOL_CLEARANCE
        np.testing.assert_allclose(
            site.xpos + rotation @ grip_offset,
            data.geom(f"bar/bar_end_{side}_geom").xpos,
            atol=1e-6,
        )
        np.testing.assert_allclose(rotation[:, 2], (0, sign, 0), atol=1e-6)
        quat = getattr(mdp, f"DEFAULT_EE_QUAT_{side.upper()}")
        expected_rotation = np.empty(9)
        mujoco.mju_quat2Mat(expected_rotation, np.array(quat))
        np.testing.assert_allclose(rotation.ravel(), expected_rotation, atol=1e-6)
        for pad in ("inner", "outer"):
            pair = {
                f"bar/bar_end_{side}_geom",
                f"robot/finger_{pad}_{side}_collision_00",
            }
            assert any(set(names) == pair for names, _ in contacts)
    for names, distance in contacts:
        assert any("finger_" in n and n.endswith("_collision_00") for n in names)
        assert not any("bar_rod" in n for n in names)
        assert distance > -0.001  # At most 1 mm soft-contact overlap.


def test_held_high_grasp_survives_initial_settling(held_scene):
    model, data = held_scene
    model.opt.timestep = 0.002
    finger_actuators = []
    for a in range(model.nu):
        joint_id = model.actuator_trnid[a, 0]
        if "finger" in model.actuator(a).name:
            finger_actuators.append((a, model.jnt_dofadr[joint_id]))
        else:
            data.ctrl[a] = data.qpos[model.jnt_qposadr[joint_id]]
    for _ in range(250):
        # Reproduce IdealPd finger control: 2 Nm closing effort and damping 2.
        for a, dof in finger_actuators:
            effort = 2.0 if "right" in model.actuator(a).name else -2.0
            data.ctrl[a] = np.clip(effort - 2.0 * data.qvel[dof], -7.0, 7.0)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qpos).all()
    contacts = bar_contacts(model, data)
    for side in ("right", "left"):
        height = data.geom(f"bar/bar_end_{side}_geom").xpos[2]
        assert height > mdp.BAR_START[2] + mdp.HELD_HIGH_RAISE - 0.03
        for pad in ("inner", "outer"):
            assert any(
                f"bar/bar_end_{side}_geom" in names
                and any(f"finger_{pad}_{side}" in n for n in names)
                for names, _ in contacts
            )
