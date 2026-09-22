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

"""Integration tests for the OpenArm-PickPlace environment.

Note: building the env compiles mujoco-warp CPU kernels; the first run can
take a few minutes.
"""

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg

OBS_DIM = 9 + 9 + 3 + 3 + 8  # joint_pos, joint_vel, ee_to_cube, cube_to_tray, actions


def test_task_is_registered():
    assert "OpenArm-PickPlace" in list_tasks()


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg("OpenArm-PickPlace")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_scene_entities_placed(env):
    tray_pos = env.scene["tray"].data.root_link_pos_w[0]
    torch.testing.assert_close(
        tray_pos, torch.tensor([0.47, 0.15, 1.005]), atol=1e-5, rtol=0.0
    )


def test_action_and_observation_dims(env):
    assert env.action_manager.total_action_dim == 8
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, OBS_DIM)
    assert obs["critic"].shape == (2, OBS_DIM)


def test_env_steps_with_finite_signals(env):
    env.reset()
    for _ in range(10):
        action = torch.zeros(2, env.action_manager.total_action_dim)
        obs, rew, terminated, truncated, _ = env.step(action)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(rew).all()
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)


def test_gripper_action_reaches_the_closed_jaw(env):
    """Action -1 must shut the jaw, or the policy can never grasp the cube.

    `use_default_offset` anchors the finger offset to the home pose, which the
    asset defines as the fully OPEN jaw, so the scale has to span the whole
    finger range for the closed jaw to be reachable inside the nominal action
    band. Regression test for openarm-mujoco 2.3.0, which moved home from the
    closed jaw to the open one and left the old scale unable to reach a grasp.
    """
    from openarm_mjlab.robot import LEFT_FINGER_HOME

    term = env.action_manager.get_term("joint_pos")
    i = term.target_names.index("openarm_left_finger_joint1")
    assert term.offset[0, i].item() == pytest.approx(LEFT_FINGER_HOME)
    # action -1 lands on the closed jaw (qpos 0), not short of it.
    assert (term.offset[0, i] - term.scale[0, i]).item() == pytest.approx(0.0)


def test_cube_spawns_on_table_after_reset(env):
    env.reset()
    cube_pos = env.scene["cube"].data.root_link_pos_w
    assert (cube_pos[:, 0] > 0.30).all() and (cube_pos[:, 0] < 0.60).all()
    assert (cube_pos[:, 1] > -0.20).all() and (cube_pos[:, 1] < 0.10).all()
    assert (cube_pos[:, 2] > 0.95).all() and (cube_pos[:, 2] < 1.15).all()


def test_nan_env_recovers_via_termination(env):
    """A rare mjwarp solver NaN must terminate+reset the env, not poison the batch.

    Regression test for a GPU training crash: rsl_rl's check_nan raised on NaN
    observations produced by one env out of 4096.
    """
    env.reset()
    env.sim.data.qvel[0, :] = float("nan")
    action = torch.zeros(2, env.action_manager.total_action_dim)
    obs, rew, terminated, truncated, _ = env.step(action)
    assert terminated[0], "NaN env must terminate"
    assert torch.isfinite(obs["actor"]).all(), "post-reset observations must be finite"
    assert torch.isfinite(rew).all(), "rewards must be sanitized"
    # The env must be fully recovered on the next step.
    obs, rew, terminated, truncated, _ = env.step(action)
    assert torch.isfinite(obs["actor"]).all()
    assert not terminated[0]
