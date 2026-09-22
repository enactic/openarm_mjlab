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
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from ...common_mdp import both_pads_on_block, pinch_obs
from ...robot_bimanual import GRASP_LOCAL_OFFSET

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["both_pads_on_block", "pinch_obs"]

BLOCK_START = (0.30, -0.20, 0.43)
TABLE_TOP_Z = 0.40
TARGET_LIFT = 0.12  # m above start.
MAX_LIFT_RATE = 0.3  # m/s
SETTLED_SPEED = 0.10  # m/s, block speed for "held".
HEIGHT_TOLERANCE = 0.03  # m; success window is TARGET_LIFT..+30mm.


def block_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the block's world position."""
    block: Entity = env.scene[asset_cfg.name]
    return block.data.root_link_pos_w


def block_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the block's 3D speed."""
    block: Entity = env.scene[asset_cfg.name]
    return torch.linalg.norm(block.data.root_link_vel_w[:, :3], dim=-1)


def pinch_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return a dense reward for holding a genuine bilateral pinch."""
    return both_pads_on_block(env, sensor_name).float()


def _pinch_streak(env) -> torch.Tensor:
    """Return the per-env count of consecutive steps holding a pinch."""
    if not hasattr(env, "_lift_pinch_streak"):
        env._lift_pinch_streak = torch.zeros(env.num_envs, device=env.device)
    return env._lift_pinch_streak


# ~0.5s (25 steps), roughly the time a real lift to TARGET_LIFT at
# MAX_LIFT_RATE would take (0.12 / 0.3 = 0.4s), so full streak credit
# lines up with "held about as long as lifting takes".
PINCH_STREAK_CAP = 25.0


def pinch_streak_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return a graded bonus for SUSTAINED bilateral contact, not just instantaneous contact.

    :func:`pinch_reward` and :func:`partial_pinch_reward` pay identically
    for a 2-step graze or a 100-step hold, so repeated brief pecks
    accumulate reward comparable to a genuine sustained grip without ever
    committing to the harder, more precise control a real lift needs.
    This term makes duration itself pay: the streak resets to 0 the
    instant contact breaks, so a peck barely registers, while a genuine
    hold ramps up to full credit over roughly the time an actual lift
    takes.
    """
    pinched = both_pads_on_block(env, sensor_name)
    streak = _pinch_streak(env)
    streak[pinched] += 1.0
    streak[~pinched] = 0.0
    return torch.clamp(streak / PINCH_STREAK_CAP, 0.0, 1.0)


def partial_pinch_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return a graded predecessor to the binary both-pads pinch.

    0, 0.5, or 1.0 for how many of the two pads currently touch the
    block. The binary AND in :func:`both_pads_on_block` is an all-or-
    nothing gate with no signal for "one pad landed, still working on
    the other"; this term is purely additive and does not touch
    ``pinch_reward``/``both_pads_on_block``, so success and termination
    semantics (which require the real two-pad pinch) are unchanged.
    """
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    per_pad = (found.view(env.num_envs, 2, -1).amax(dim=-1) > 0).float()
    return per_pad.mean(dim=-1)


def lift_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the block's height gained above its start, clamped to non-negative."""
    return torch.clamp(block_pos_w(env, asset_cfg)[:, 2] - BLOCK_START[2], min=0.0)


def tool_to_block_obs(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the vector from the finger-cage center to the block, base frame."""
    robot: Entity = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, robot_cfg.site_ids].squeeze(1)
    offset = torch.tensor(GRASP_LOCAL_OFFSET, device=ee_pos_w.device).expand_as(
        ee_pos_w
    )
    tool_w = ee_pos_w + quat_apply(ee_quat_w, offset)
    vec_w = block_pos_w(env, asset_cfg) - tool_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def reach_block_reward(
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a Gaussian-kernel reward on tool-to-block distance."""
    d2 = torch.sum(torch.square(tool_to_block_obs(env, robot_cfg, asset_cfg)), dim=-1)
    return torch.exp(-d2 / std**2)


def _max_height(env) -> torch.Tensor:
    """Return the per-env running-max lift height reached this episode."""
    if not hasattr(env, "_lift_max_height"):
        env._lift_max_height = torch.zeros(env.num_envs, device=env.device)
    return env._lift_max_height


def lift_rate_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a pinch-gated rate of new height above the episode's running max.

    New-progress-only, so a plain rise-rate reward cannot be farmed by
    bouncing. Credited progress is capped at ``TARGET_LIFT``: the buffer
    itself still tracks the TRUE max height (for ``block_fell`` and other
    bookkeeping), but the reward stops paying once the intended height is
    reached, removing any incentive to keep climbing past the target.
    """
    h = lift_height(env, asset_cfg)
    maxh = _max_height(env)
    h_capped = torch.clamp(h, max=TARGET_LIFT)
    maxh_capped = torch.clamp(maxh, max=TARGET_LIFT)
    new = torch.clamp(h_capped - maxh_capped, min=0.0)
    maxh.copy_(torch.maximum(maxh, h))
    rate = torch.clamp(new / env.step_dt, 0.0, MAX_LIFT_RATE) / MAX_LIFT_RATE
    return rate * both_pads_on_block(env, sensor_name).float()


def block_descent_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return an anti-pump penalty: lowering the block is never free."""
    block: Entity = env.scene[asset_cfg.name]
    return torch.clamp(-block.data.root_link_vel_w[:, 2], min=0.0)


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


def lift_success_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Fire exactly once, on the ``lifted_target`` termination step."""
    return env.termination_manager.get_term("lifted_target").float()


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
    slow = block_speed(env, asset_cfg) < SETTLED_SPEED
    return high & slow & both_pads_on_block(env, sensor_name)


def block_fell(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return early termination: the block fell off the table (top z=0.40)."""
    return block_pos_w(env, asset_cfg)[:, 2] < 0.30


def reset_block_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_range: float = 0.03,
) -> None:
    """Reset the block to its default pose plus xy jitter (raw coordinates)."""
    block: Entity = env.scene[asset_cfg.name]
    default = block.data.default_root_state
    assert default is not None
    state = default[env_ids].clone()
    n = len(env_ids)
    state[:, 0] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 1] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 7:] = 0.0
    block.write_root_state_to_sim(state, env_ids=env_ids)
    # Reset the new-height buffer (written state has the block at rest on
    # the table: height gained = 0).
    _max_height(env)[env_ids] = 0.0
    _pinch_streak(env)[env_ids] = 0.0


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
    real teacher, and ``block_fell`` never fires on a table-height drop.
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
