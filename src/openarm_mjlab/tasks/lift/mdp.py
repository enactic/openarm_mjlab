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

"""Lift task MDP: squeeze-grip the 50mm block and raise it.

Contact-only feasibility: the block (50mm) is WIDER than the closed cage
gap (~30mm), so the pads genuinely squeeze it -- unlike a slim handle bar,
a real friction grip exists here. Lift income requires BOTH pads on the
block; height rate is capped; success requires the block held high and
settled.
"""

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
    pinch_streak_reward,
    reach_object_reward,
    reset_object_xy_uniform,
    reset_pinch_streak,
    terminated_by,
    tool_to_object_obs,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "both_pads_on_block",
    "descent_penalty",
    "partial_pinch_reward",
    "pinch_obs",
    "pinch_reward",
    "pinch_streak_reward",
    "reach_object_reward",
    "terminated_by",
    "tool_to_object_obs",
]

BLOCK_START = (0.30, -0.20, 0.43)
TABLE_TOP_Z = 0.40
TARGET_LIFT = 0.12  # m above start.
MAX_LIFT_RATE = 0.3  # m/s
SETTLED_SPEED = 0.10  # m/s, block speed for "held".
HEIGHT_TOLERANCE = 0.03  # m; success window is TARGET_LIFT..+30mm.


def lift_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the block's height gained above its start, clamped to non-negative."""
    return torch.clamp(object_pos_w(env, asset_cfg)[:, 2] - BLOCK_START[2], min=0.0)


def _max_height(env) -> torch.Tensor:
    """Return the per-env running-max lift height reached this episode."""
    return env_buffer(env, "_lift_max_height")


def lift_rate_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a pinch-gated rate of new height above the episode's running max.

    New-progress-only, so a plain rise-rate reward cannot be farmed by
    bouncing. Credited progress is capped at ``TARGET_LIFT``: the buffer
    itself still tracks the TRUE max height (for the ``block_fell`` termination and other
    bookkeeping), but the reward stops paying once the intended height is
    reached, removing any incentive to keep climbing past the target.
    """
    rate = new_progress_rate(
        env,
        lift_height(env, asset_cfg),
        _max_height(env),
        MAX_LIFT_RATE,
        cap=TARGET_LIFT,
    )
    return rate * both_pads_on_block(env, sensor_name).float()


def held_high_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return graded height-holding income.

    A binary height gate leaves a gradient desert between a shallow lift
    and the full target; dense height pay teaches squeeze-and-raise
    incrementally.
    """
    frac = torch.clamp(lift_height(env, asset_cfg) / TARGET_LIFT, 0.0, 1.0)
    return frac * both_pads_on_block(env, sensor_name).float()


def lifted_target(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return success: block height in the target window, settled, and pinched.

    The window is bounded (``TARGET_LIFT`` to ``TARGET_LIFT +
    HEIGHT_TOLERANCE``), not a bare lower bound: an unbounded check lets a
    policy that keeps climbing past the target still count as success,
    which does not measure the intended ~120mm lift.
    """
    h = lift_height(env, asset_cfg)
    high = (h >= TARGET_LIFT) & (h <= TARGET_LIFT + HEIGHT_TOLERANCE)
    slow = object_speed(env, asset_cfg) < SETTLED_SPEED
    return high & slow & both_pads_on_block(env, sensor_name)


def reset_block_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_range: float = 0.03,
) -> None:
    """Reset the block to its default pose plus xy jitter (raw coordinates)."""
    reset_object_xy_uniform(env, env_ids, asset_cfg, xy_range)
    # Reset the new-height buffer (written state has the block at rest on
    # the table: height gained = 0).
    _max_height(env)[env_ids] = 0.0
    reset_pinch_streak(env, env_ids)


# Held-at-height reference-state init: a DLS IK pose holding the tool
# point at block-start +80mm (residual 0.08mm).
HELD_HIGH_POSE = {
    "openarm_right_joint1": -0.2181,
    "openarm_right_joint2": 0.0461,
    "openarm_right_joint3": 0.1049,
    "openarm_right_joint4": 1.8194,
    "openarm_right_joint5": 0.0007,
    "openarm_right_joint6": -0.1009,
    "openarm_right_joint7": -0.0426,
}
HELD_HIGH_TOOL_Z = 0.51


def reset_held_high(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    robot_joints_cfg: SceneEntityCfg,
    probability: float = 0.5,
) -> None:
    """With probability ``probability``, start the episode already holding the block.

    Arm at the IK hold pose, fingers pressed to block width, block at the
    tool point, +80mm above the table. The policy must clamp quickly or
    the block slips out: the held-high income stream it forfeits is the
    real teacher, and the ``block_fell`` termination never fires on a table-height drop.
    Runs after ``reset_block`` (overrides the subset it picks).
    """
    robot: Entity = env.scene[robot_joints_cfg.name]
    block: Entity = env.scene[asset_cfg.name]
    pick = torch.rand(len(env_ids), device=env.device) < probability
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    # Arm is already at the held-high DEFAULT (reset_robot_joints jitters
    # around it); only pin the fingers to block width and place the block.
    jp = robot.data.joint_pos[ids].clone()
    jv = torch.zeros_like(robot.data.joint_vel[ids])
    for j, name in enumerate(robot.joint_names):
        if "right_finger" in name:
            jp[:, j] = -0.22
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    state = block.data.default_root_state[ids].clone()
    state[:, 0] = 0.30
    state[:, 1] = -0.20
    state[:, 2] = HELD_HIGH_TOOL_Z
    state[:, 3:7] = torch.tensor([1.0, 0, 0, 0], device=env.device)
    state[:, 7:] = 0.0
    block.write_root_state_to_sim(state, env_ids=ids)
    # Height buffer: credit only NEW height above the spawn hold.
    _max_height(env)[ids] = HELD_HIGH_TOOL_Z - BLOCK_START[2]
