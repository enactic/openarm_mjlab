"""Integration tests for the OpenArm-Handover environment.

Note: building the env compiles mujoco-warp CPU kernels; the first run can
take a few minutes.
"""

from types import SimpleNamespace

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg

# joint_pos, joint_vel, tool_to_block_giver, tool_to_block_recv,
# block_to_target, pinch_giver, pinch_recv, transferred, actions
OBS_DIM = 18 + 18 + 3 + 3 + 3 + 1 + 1 + 1 + 16
ACTION_DIM = 7 + 7 + 1 + 1


def test_task_is_registered():
    assert "OpenArm-Handover" in list_tasks()


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg("OpenArm-Handover")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_action_and_observation_dims(env):
    assert env.action_manager.total_action_dim == ACTION_DIM
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


def test_both_arms_are_actuated(env):
    names = list(env.action_manager._terms.keys())
    for n in ("right_arm", "left_arm", "right_squeeze", "left_squeeze"):
        assert n in names


def test_latch_term_precedes_dependents():
    order = list(load_env_cfg("OpenArm-Handover").rewards)
    i = order.index("overlap_streak")
    assert i < order.index("giver_release")
    assert i < order.index("carry_to_target")


def test_block_starts_on_the_table():
    from mjlab.envs import ManagerBasedRlEnv

    from openarm_mjlab.tasks.handover import mdp as ho

    cfg = load_env_cfg("OpenArm-Handover")
    cfg.scene.num_envs = 2
    cfg.events["reset_giver_held_high"].params["probability"] = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        z = env.scene["block"].data.root_link_pos_w[:, 2]
        assert torch.allclose(z, torch.tensor(ho.BLOCK_START[2]), atol=0.02)
    finally:
        env.close()


def test_transferred_latch_starts_clear(env):
    from openarm_mjlab.tasks.handover import mdp as ho

    env.reset()
    assert (ho._transferred(env) == 0).all()


def test_geometry_constants_are_consistent():
    from openarm_mjlab.tasks.handover import mdp as ho

    assert ho.TARGET_POS[1] < 0.0 < ho.BLOCK_START[1]  # giver left, receiver right
    assert ho.TARGET_TOL < ho.CARRY_STD
    assert ho.HELD_HIGH_RAISE < ho.TARGET_LIFT


def test_held_high_reset_only_touches_left_arm_in_selected_envs(monkeypatch):
    from mjlab.managers import SceneEntityCfg

    from openarm_mjlab.tasks.handover import mdp as ho

    names = ["unrelated_joint", "openarm_right_joint1"]
    names += [f"openarm_left_joint{i}" for i in range(1, 8)]
    names.append("openarm_left_finger_joint1")
    held = [0.4, 0.0, *ho.HELD_HIGH_LEFT_Q, ho.LEFT_FINGER_SQUEEZE]
    positions = torch.tensor(held).repeat(3, 1) + 0.025
    velocities = torch.full_like(positions, 0.1)
    before_p, before_v = positions.clone(), velocities.clone()
    block_states = torch.zeros(3, 13)
    block_states[:, 3] = 1.0
    before_b = block_states.clone()

    def write_joints(joint_pos, joint_vel, *, env_ids):
        positions[env_ids] = joint_pos
        velocities[env_ids] = joint_vel

    def write_block(state, *, env_ids):
        block_states[env_ids] = state

    robot = SimpleNamespace(
        joint_names=names,
        data=SimpleNamespace(joint_pos=positions, joint_vel=velocities),
        write_joint_state_to_sim=write_joints,
    )
    block = SimpleNamespace(
        data=SimpleNamespace(default_root_state=before_b),
        write_root_state_to_sim=write_block,
    )
    buffers = {}
    monkeypatch.setattr(
        ho, "env_buffer", lambda env, name: buffers.setdefault(name, torch.zeros(3))
    )
    env = SimpleNamespace(scene={"robot": robot, "block": block}, num_envs=3, device="cpu")
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.1, 0.9]))
    ho.reset_giver_held_high(
        env,
        torch.tensor([2, 0]),
        asset_cfg=SceneEntityCfg("block"),
        robot_joints_cfg=SceneEntityCfg("robot"),
        probability=0.5,
    )
    torch.testing.assert_close(positions[2, 2:], torch.tensor(held[2:]))
    torch.testing.assert_close(positions[2, :2], before_p[2, :2])  # right untouched
    torch.testing.assert_close(velocities[2], torch.zeros_like(velocities[2]))
    torch.testing.assert_close(positions[:2], before_p[:2])
    torch.testing.assert_close(velocities[:2], before_v[:2])
    torch.testing.assert_close(block_states[:2], before_b[:2])
    expected = torch.tensor(ho.BLOCK_START)
    expected[2] += ho.HELD_HIGH_RAISE
    torch.testing.assert_close(block_states[2, :3], expected)


def test_overlap_latches_transferred_only_when_both_grip(monkeypatch):
    from openarm_mjlab.tasks.handover import mdp as ho

    buffers = {}
    monkeypatch.setattr(
        ho, "env_buffer", lambda env, name: buffers.setdefault(name, torch.zeros(2))
    )
    grip = {"g": torch.tensor([True, True]), "r": torch.tensor([True, False])}
    monkeypatch.setattr(
        ho, "both_pads_on_block", lambda env, s: grip["g"] if s == "g" else grip["r"]
    )
    env = SimpleNamespace(num_envs=2, device="cpu")
    for _ in range(int(ho.OVERLAP_STEPS) + 2):
        ho.handover_overlap_streak(env, "g", "r")
    assert ho._transferred(env).tolist() == [1.0, 0.0]