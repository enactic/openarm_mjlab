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

OBS_DIM = 9 + 9 + 3 + 3 + 1 + 8  # joint_pos, joint_vel, ee_to_cube, cube_to_tray,
# pinch, actions


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


def _squeeze_action(env, value: float) -> torch.Tensor:
    """Zero arm action (hold home) with the squeeze term set to ``value``."""
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    start = 0
    for name in env.action_manager.active_terms:
        dim = env.action_manager.get_term(name).action_dim
        if name == "squeeze":
            action[:, start : start + dim] = value
        start += dim
    return action


def _jaw(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.joint_pos[
        :, robot.joint_names.index("openarm_left_finger_joint1")
    ]


def test_finger_is_driven_by_the_squeeze_effort_alone(env):
    """The finger's XML position servo must be gone, or it fights the effort.

    A leftover position actuator would keep pulling the jaw back to its
    target, so the policy's squeeze torque would never be the only drive.
    """
    model = env.sim.mj_model
    import mujoco

    finger = model.joint("robot/openarm_left_finger_joint1").id
    on_finger = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        for a in range(model.nu)
        if model.actuator_trnid[a, 0] == finger
    ]
    assert len(on_finger) == 1, on_finger
    assert "squeeze" in env.action_manager.active_terms
    assert env.action_manager.total_action_dim == 8


def test_squeeze_action_shuts_and_opens_the_jaw_within_its_range(env):
    """Sustained -1 must shut the jaw and +1 reopen it, without breaching the limits.

    Effort control drives the jaw to an end stop instead of to a target, so a
    soft joint limit would let it overshoot past 0 (closed) or 0.785 (open).
    """
    from openarm_mjlab.openarm_cell import LEFT_FINGER_HOME

    # The asset's finger damping caps a 2 N*m squeeze at ~0.5 rad/s, so a
    # full stroke takes ~1.6 s (80 steps).
    env.reset()
    for _ in range(50):
        env.step(_squeeze_action(env, -1.0))
    assert (_jaw(env) < 0.05).all()
    assert (_jaw(env) > -0.02).all()
    for _ in range(100):
        env.step(_squeeze_action(env, 1.0))
    assert (_jaw(env) > LEFT_FINGER_HOME - 0.05).all()
    assert (_jaw(env) < LEFT_FINGER_HOME + 0.02).all()


@pytest.mark.parametrize("yaw_deg", [0.0, 22.5, 45.0])
def test_start_jaw_straddles_the_cube_and_a_short_squeeze_pinches_it(env, yaw_deg):
    """Episodes start with the jaw just wider than the cube, as the Lift task does.

    From fully open, zero-mean exploration never drove the damped jaw the
    ~0.6 rad down to the cube; starting near grasp width, the open jaw still
    clears the cube at any yaw, and a brief squeeze closes it on the pads.
    """
    import math

    from openarm_mjlab.common_mdp import both_pads_on_block
    from openarm_mjlab.tasks.pick_place.pick_place_env_cfg import FINGER_CUBE_SENSOR

    yaw = math.radians(yaw_deg)
    quat = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    env.reset()
    for _ in range(5):  # Zero squeeze: the start jaw must not touch the cube.
        _place_cube_at_grasp_site(env, quat)
        env.step(_squeeze_action(env, 0.0))
        assert not both_pads_on_block(env, FINGER_CUBE_SENSOR.name).any()
    # 0.5 s of full squeeze: ~0.15 rad of travel at ~0.5 rad/s, with margin
    # for the +-0.05 rad reset jitter on the start jaw.
    for _ in range(25):
        _place_cube_at_grasp_site(env, quat)
        env.step(_squeeze_action(env, -1.0))
    assert both_pads_on_block(env, FINGER_CUBE_SENSOR.name).all()


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


def _place_cube_at_grasp_site(env, quat=None):
    """Teleport the cube (at rest) to the grasp-center site between the open jaws."""
    from openarm_mjlab.openarm_cell import LEFT_GRASP_SITE

    robot = env.scene["robot"]
    cube = env.scene["cube"]
    site = robot.site_names.index(LEFT_GRASP_SITE)
    state = cube.data.default_root_state.clone()
    state[:, :3] = robot.data.site_pos_w[:, site]
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0]) if quat is None else quat
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state)


def test_closing_the_jaw_on_the_cube_registers_a_pinch_and_holds_it(env):
    """The finger-cube sensor must see a two-pad pinch once the jaw shuts on the cube.

    The pinch-gated rewards are only a learning signal if a genuine grasp
    actually trips them, and the grasp must hold the cube up against gravity.
    """
    from openarm_mjlab.common_mdp import both_pads_on_block

    from openarm_mjlab.tasks.pick_place.pick_place_env_cfg import (
        FINGER_CUBE_SENSOR,
        LIFT_MIN_Z,
    )

    env.reset()
    action = _squeeze_action(env, -1.0)  # Shut the jaw; hold the arm at home.
    # Pin the cube at the grasp center while the jaw closes, so it is caught
    # between the pads rather than dropping into the finger cage first.
    for _ in range(25):
        _place_cube_at_grasp_site(env)
        env.step(action)
    for _ in range(25):  # Released: the pinch alone must hold it for 0.5 s.
        env.step(action)
    assert both_pads_on_block(env, FINGER_CUBE_SENSOR.name).all()
    assert (env.scene["cube"].data.root_link_pos_w[:, 2] > LIFT_MIN_Z).all()


def test_lift_does_not_pay_for_an_unpinched_airborne_cube(env):
    """A cube knocked into the air (no grasp) must not earn the lift reward."""
    from openarm_mjlab.tasks.pick_place import mdp as pick_mdp
    from openarm_mjlab.tasks.pick_place.pick_place_env_cfg import (
        FINGER_CUBE_SENSOR,
        LIFT_MIN_Z,
    )

    env.reset()
    cube = env.scene["cube"]
    state = cube.data.default_root_state.clone()
    state[:, 2] = LIFT_MIN_Z + 0.05
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state)
    env.sim.forward()
    assert (cube.data.root_link_pos_w[:, 2] > LIFT_MIN_Z).all()
    lifted = pick_mdp.object_lifted(
        env, "cube", LIFT_MIN_Z, sensor_name=FINGER_CUBE_SENSOR.name
    )
    assert (lifted == 0.0).all()


def test_staged_reward_is_reaching_times_one_plus_bringing(env):
    """Staged reward follows mjlab's lift_cube: reaching * (1 + bringing).

    Moving the cube toward the hover target must raise it while the gripper
    stays on the cube, so height and transport always carry a gradient.
    """
    from openarm_mjlab.tasks.pick_place import mdp as pick_mdp
    from openarm_mjlab.tasks.pick_place.pick_place_env_cfg import (
        EE_SITE,
        TRAY_TARGET_OFFSET,
    )
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    env.reset()
    robot_cfg = SceneEntityCfg("robot", site_names=(EE_SITE,))
    robot_cfg.resolve(env.scene)
    params = dict(
        object_name="cube",
        target_name="tray",
        reaching_std=0.2,
        bringing_std=0.3,
        asset_cfg=robot_cfg,
        target_offset=TRAY_TARGET_OFFSET,
    )
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    tray = env.scene["tray"]

    def expected():
        ee = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
        obj = cube.data.root_link_pos_w
        target = tray.data.root_link_pos_w + torch.tensor(TRAY_TARGET_OFFSET)
        reaching = torch.exp(-((ee - obj) ** 2).sum(-1) / 0.2**2)
        bringing = torch.exp(-((target - obj) ** 2).sum(-1) / 0.3**2)
        return reaching * (1.0 + bringing)

    on_table = pick_mdp.staged_reach_bring_reward(env, **params)
    torch.testing.assert_close(on_table, expected())

    # Raise the cube halfway toward the hover target, keeping it on the gripper
    # side: the reward must grow.
    state = cube.data.default_root_state.clone()
    obj = cube.data.root_link_pos_w.clone()
    target = tray.data.root_link_pos_w + torch.tensor(TRAY_TARGET_OFFSET)
    state[:, :3] = obj + 0.5 * (target - obj)
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state)
    env.sim.forward()
    raised = pick_mdp.staged_reach_bring_reward(env, **params)
    torch.testing.assert_close(raised, expected())
    assert (raised > on_table).all()


def test_no_curriculum_ramps_the_velocity_penalty():
    """The hinge penalty must stay fixed: ramping it 100x mid-run froze exploration."""
    cfg = load_env_cfg("OpenArm-PickPlace")
    assert not cfg.curriculum
    assert cfg.rewards["joint_vel_hinge"].weight == -0.01


def _put_cube_in_tray(env):
    """Teleport the cube (at rest) onto the tray floor, as a push-in would leave it."""
    from openarm_mjlab.tasks.pick_place.pick_place_env_cfg import CUBE_TRAY_REST_DZ

    cube = env.scene["cube"]
    state = cube.data.default_root_state.clone()
    state[:, :3] = env.scene["tray"].data.root_link_pos_w
    state[:, 2] += CUBE_TRAY_REST_DZ
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state)
    env.sim.forward()


def _tray_signals(env):
    from openarm_mjlab.tasks.pick_place import mdp as pick_mdp

    place = pick_mdp.object_in_tray(env, **env.cfg.rewards["place"].params)
    success = pick_mdp.object_settled_in_tray(
        env, **env.cfg.terminations["success"].params
    )
    return place, success


def test_cube_pushed_into_tray_earns_no_place_or_success(env):
    """Place and success must require carrying: pushing the cube in pays nothing.

    Regression test for a trained policy that shoved the cube into the tray
    without ever grasping it and still collected place and the success bonus.
    """
    env.reset()
    _put_cube_in_tray(env)
    place, success = _tray_signals(env)
    assert not place.any()
    assert not success.any()


def test_carried_cube_set_in_tray_earns_place_and_success(env, monkeypatch):
    """A cube pinched above the tray walls this episode counts once it is in the tray."""
    from openarm_mjlab.tasks.pick_place import mdp as pick_mdp
    from openarm_mjlab.tasks.pick_place.pick_place_env_cfg import TRANSPORT_MIN_Z

    env.reset()
    cube = env.scene["cube"]
    state = cube.data.default_root_state.clone()
    state[:, 2] = TRANSPORT_MIN_Z + 0.01
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state)
    env.sim.forward()
    params = env.cfg.rewards["place"].params
    # Unpinched above the walls (a flick) does not count as carrying.
    carried = pick_mdp.object_carried(
        env, "cube", params["carry_min_height"], params["sensor_name"]
    )
    assert not carried.any()
    monkeypatch.setattr(
        pick_mdp,
        "both_pads_on_block",
        lambda env, name: torch.ones(env.num_envs, dtype=torch.bool),
    )
    carried = pick_mdp.object_carried(
        env, "cube", params["carry_min_height"], params["sensor_name"]
    )
    assert carried.all()
    monkeypatch.undo()

    _put_cube_in_tray(env)
    place, success = _tray_signals(env)
    assert place.all()
    assert success.all()


def test_reset_clears_the_carried_flag(env):
    from openarm_mjlab.tasks.pick_place import mdp as pick_mdp

    env.reset()
    pick_mdp._carried(env)[:] = True
    env.reset()
    assert not pick_mdp._carried(env).any()
