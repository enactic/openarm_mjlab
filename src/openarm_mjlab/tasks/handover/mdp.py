from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common_mdp import (
    both_pads_on_block,
    descent_penalty,
    env_buffer,
    new_progress_rate,
    object_pos_w,
    object_speed,
    partial_pinch_reward,
    pinch_obs,
    pinch_reward,
    reset_object_xy_uniform,
    terminated_by,
    tool_to_point,
)
from ..bimanual_lift.mdp import (
    HELD_HIGH_LEFT_Q,
    HELD_HIGH_RAISE,
    LEFT_FINGER_SQUEEZE,
    MAX_LIFT_RATE,
    PINCH_STREAK_CAP,
    SETTLED_SPEED,
    TARGET_LIFT,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "descent_penalty",
    "partial_pinch_reward",
    "pinch_obs",
    "pinch_reward",
    "terminated_by",
]

TABLE_TOP_Z = 0.40
BLOCK_START = (0.30, 0.16, 0.43)  # giver side (matches bimanual left bar end)
TARGET_POS = (0.30, -0.16, 0.43 + TARGET_LIFT)  # receiver side, raised
# TARGET_POS = (0.42, -0.26, 0.45)
GIVER_SIDE = "left"

OVERLAP_STEPS = 10.0  # simultaneous-grip steps required to latch `transferred`
TARGET_TOL = 0.03  # m, success radius
CARRY_STD = 0.10  # m, carry shaping width
FALL_Z = 0.30


def _target_w(env) -> torch.Tensor:
    return torch.tensor(TARGET_POS, device=env.device).expand(env.num_envs, 3)


def _transferred(env) -> torch.Tensor:
    return env_buffer(env, "_handover_transferred")


def _streak(env, side: str) -> torch.Tensor:
    return env_buffer(env, f"_handover_streak_{side}")


def _max_h(env) -> torch.Tensor:
    return env_buffer(env, "_handover_max_h")


def block_height(env, asset_cfg) -> torch.Tensor:
    z = object_pos_w(env, asset_cfg)[:, 2]
    return torch.clamp(z - BLOCK_START[2], min=0.0)


def tool_to_block_obs(env, robot_cfg, asset_cfg) -> torch.Tensor:
    return tool_to_point(env, robot_cfg, object_pos_w(env, asset_cfg))


def block_to_target_obs(env, asset_cfg) -> torch.Tensor:
    return _target_w(env) - object_pos_w(env, asset_cfg)


def transferred_obs(env) -> torch.Tensor:
    return _transferred(env).unsqueeze(-1).float()


def reach_block_reward(env, std: float, robot_cfg, asset_cfg) -> torch.Tensor:
    vec = tool_to_block_obs(env, robot_cfg, asset_cfg)
    return torch.exp(-torch.sum(torch.square(vec), dim=-1) / std**2)


def giver_lift_reward(env, sensor_giver: str, asset_cfg) -> torch.Tensor:
    """New-progress-only lift of the block while ONLY the giver need hold it.

    Pays only before transfer so it cannot compete with the carry phase.
    """
    rate = new_progress_rate(
        env, block_height(env, asset_cfg), _max_h(env), MAX_LIFT_RATE, cap=TARGET_LIFT
    )
    gate = both_pads_on_block(env, sensor_giver) & ~_transferred(env).bool()
    return rate * gate.float()


def handover_overlap_streak(env, sensor_giver: str, sensor_recv: str) -> torch.Tensor:
    """min(streak_giver, streak_receiver); latches `transferred` once sustained."""
    sg, sr = _streak(env, "giver"), _streak(env, "recv")
    g = both_pads_on_block(env, sensor_giver)
    r = both_pads_on_block(env, sensor_recv)
    sg[g] += 1.0
    sg[~g] = 0.0
    sr[r] += 1.0
    sr[~r] = 0.0
    paired = torch.minimum(sg, sr)
    t = _transferred(env)
    t[paired >= OVERLAP_STEPS] = 1.0
    return torch.clamp(paired / PINCH_STREAK_CAP, 0.0, 1.0)


def giver_release_reward(env, sensor_giver: str, sensor_recv: str) -> torch.Tensor:
    """Receiver holds, giver has let go -- only after the overlap latch."""
    g = both_pads_on_block(env, sensor_giver)
    r = both_pads_on_block(env, sensor_recv)
    return (_transferred(env).bool() & r & ~g).float()


def carry_to_target_reward(env, sensor_recv: str, asset_cfg) -> torch.Tensor:
    """Dense Gaussian toward the target, gated on receiver grip AND transfer."""
    d2 = torch.sum(torch.square(block_to_target_obs(env, asset_cfg)), dim=-1)
    gate = both_pads_on_block(env, sensor_recv) & _transferred(env).bool()
    return torch.exp(-d2 / CARRY_STD**2) * gate.float()


def handed_over(env, sensor_giver: str, sensor_recv: str, asset_cfg) -> torch.Tensor:
    """Success: receiver holds, giver has no contact, block at target, settled."""
    d = torch.norm(block_to_target_obs(env, asset_cfg), dim=-1)
    return (
        (d < TARGET_TOL)
        & both_pads_on_block(env, sensor_recv)
        & ~both_pads_on_block(env, sensor_giver)
        & _transferred(env).bool()
        & (object_speed(env, asset_cfg) < SETTLED_SPEED)
    )


def block_fell(env, asset_cfg) -> torch.Tensor:
    return object_pos_w(env, asset_cfg)[:, 2] < FALL_Z


def _zero_buffers(env, ids) -> None:
    _transferred(env)[ids] = 0.0
    _max_h(env)[ids] = 0.0
    _streak(env, "giver")[ids] = 0.0
    _streak(env, "recv")[ids] = 0.0


def reset_block_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_range: float = 0.03,
) -> None:
    reset_object_xy_uniform(env, env_ids, asset_cfg, xy_range)
    _zero_buffers(env, env_ids)


def reset_giver_held_high(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    robot_joints_cfg: SceneEntityCfg,
    probability: float = 0.5,
) -> None:
    """Start some episodes with the giver already holding the raised block.

    Reuses bimanual_lift's solved LEFT held-high pose (its grip point is exactly
    BLOCK_START's y). Run after reset_block_uniform.
    """
    if probability <= 0.0:
        return
    robot: Entity = env.scene[robot_joints_cfg.name]
    block: Entity = env.scene[asset_cfg.name]
    pick = torch.rand(len(env_ids), device=env.device) < probability
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    jp = robot.data.joint_pos[ids].clone()
    jv = torch.zeros_like(robot.data.joint_vel[ids])
    arm = {f"openarm_left_joint{i}": q for i, q in enumerate(HELD_HIGH_LEFT_Q, start=1)}
    for j, name in enumerate(robot.joint_names):
        if name in arm:
            jp[:, j] = arm[name]
        if "left_finger" in name:
            jp[:, j] = LEFT_FINGER_SQUEEZE
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    state = block.data.default_root_state[ids].clone()
    state[:, 0], state[:, 1] = BLOCK_START[0], BLOCK_START[1]
    state[:, 2] = BLOCK_START[2] + HELD_HIGH_RAISE
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:] = 0.0
    block.write_root_state_to_sim(state, env_ids=ids)
    _max_h(env)[ids] = HELD_HIGH_RAISE