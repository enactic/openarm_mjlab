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

"""Move-puck (pushing) MDP terms.

Pushing needs no grasp, so the anti-cheat surface is smaller than the
drawer/valve tasks: the one exploit to close is FLICKING (smack the puck
and let it coast to the goal). The approach-rate reward is contact-gated
and capped (coasting earns nothing), success requires the puck SETTLED at
the goal, and puck overspeed is penalized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from ...common_mdp import (
    fingers_on_handle,
    fingers_on_handle_obs,
    terminated_by,
)
from ...robot_bimanual import GRASP_LOCAL_OFFSET

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "fingers_on_handle",
    "fingers_on_handle_obs",
    "terminated_by",
]

# Goal disc, local to the env origin.
GOAL_LOCAL = (0.33, -0.30, 0.422)
SUCCESS_DIST = 0.025  # m, planar.
SETTLED_SPEED = 0.05  # m/s
MAX_PUSH_SPEED = 0.25  # m/s, puck speed cap for the rate reward.


# Named targets for the language-conditioned variant. The default is the one
# this task has always used, so nothing here changes an existing run; the
# alternative exists so an instruction can choose between them. "left" mirrors
# the default across the puck's own start line, giving the same push distance
# in the opposite lateral direction.
NAMED_GOALS = {
    "right": (0.33, -0.30, 0.422),
    "left": (0.33, 0.02, 0.422),
}
assert NAMED_GOALS["right"] == GOAL_LOCAL, "the default target must stay the default"


def goal_buf(env) -> torch.Tensor:
    """Per-env target, defaulting to the one the task has always used."""
    if not hasattr(env, "_puck_goal"):
        env._puck_goal = (
            torch.tensor(GOAL_LOCAL, device=env.device).expand(env.num_envs, 3).clone()
        )
    return env._puck_goal


def _named_goal(env, name: str) -> torch.Tensor:
    """Return a cached device tensor for a named target.

    These are constants, and both callers run every step at up to a few
    thousand envs; rebuilding them from a Python tuple is a host-to-device
    copy each time.
    """
    cache = getattr(env, "_named_goal_cache", None)
    if cache is None:
        cache = env._named_goal_cache = {}
    if name not in cache:
        cache[name] = torch.tensor(
            NAMED_GOALS[name], device=env.device, dtype=torch.float32
        )
    return cache[name]


def set_puck_goal(env, env_ids: torch.Tensor, name: str) -> None:
    """Point the given envs at a named target."""
    goal_buf(env)[env_ids] = _named_goal(env, name)


def puck_to_midpoint_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Vector from the puck to the MIDPOINT of the named targets.

    The language variant substitutes this for ``puck_to_goal_obs``, which
    points straight at the active target and would make the instruction
    redundant.

    It points at the midpoint rather than, say, the puck's own position
    because the substitute has to keep the same MEANING as well as the same
    width. Replacing "vector from the puck to somewhere worth pushing" with
    "position of the puck" preserves the shape and changes what the numbers
    are, and a policy carried over from a checkpoint trained on the old
    meaning reads the new ones as the old quantity: measured, success went to
    0.000 for both goals and stayed there. The midpoint is the same kind of
    vector at the same scale, and identical for both goals, so it says
    nothing about which one was asked for.
    """
    robot: Entity = env.scene[robot_cfg.name]
    midpoint = ((_named_goal(env, "right") + _named_goal(env, "left")) / 2).expand(
        env.num_envs, 3
    )
    vec_w = midpoint - puck_pos_w(env, asset_cfg)
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def _goal_w(env) -> torch.Tensor:
    # mjlab batches each env as its OWN world: physics coordinates are
    # raw; env_origins is only a viewer layout grid. Do NOT add origins.
    return goal_buf(env)


def puck_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the puck's world position."""
    puck: Entity = env.scene[asset_cfg.name]
    return puck.data.root_link_pos_w


def puck_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the puck's planar speed."""
    puck: Entity = env.scene[asset_cfg.name]
    return torch.linalg.norm(puck.data.root_link_vel_w[:, :2], dim=-1)


def puck_goal_dist(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the planar distance from the puck to the goal center."""
    d = puck_pos_w(env, asset_cfg)[:, :2] - _goal_w(env)[:, :2]
    return torch.linalg.norm(d, dim=-1)


def puck_to_goal_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the vector from the puck to the goal, base frame."""
    robot: Entity = env.scene[robot_cfg.name]
    vec_w = _goal_w(env) - puck_pos_w(env, asset_cfg)
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def tool_to_puck_obs(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the vector from the finger-cage center to the puck, base frame."""
    robot: Entity = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, robot_cfg.site_ids].squeeze(1)
    offset = torch.tensor(GRASP_LOCAL_OFFSET, device=ee_pos_w.device).expand_as(
        ee_pos_w
    )
    tool_w = ee_pos_w + quat_apply(ee_quat_w, offset)
    vec_w = puck_pos_w(env, asset_cfg) - tool_w
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def reach_puck_reward(
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a Gaussian-kernel reward on tool-to-puck distance."""
    d2 = torch.sum(torch.square(tool_to_puck_obs(env, robot_cfg, asset_cfg)), dim=-1)
    return torch.exp(-d2 / std**2)


def push_rate_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a contact-gated, capped rate of approach to the goal (0..1).

    Coasting after a flick earns nothing (contact gate); speed above the
    cap earns no more than the cap. New-progress-only against the
    episode's best distance so push-pull cycling cannot farm income.
    """
    dist = puck_goal_dist(env, asset_cfg)
    if not hasattr(env, "_puck_min_dist"):
        env._puck_min_dist = dist.clone()
    new = torch.clamp(env._puck_min_dist - dist, min=0.0)
    env._puck_min_dist.copy_(torch.minimum(env._puck_min_dist, dist))
    capped = torch.clamp(new / env.step_dt, 0.0, MAX_PUSH_SPEED) / MAX_PUSH_SPEED
    return capped * fingers_on_handle(env, sensor_name).float()


def at_goal_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a reward for settling at the goal: close AND slow.

    A 1.2x band, not the exact success band: outside it the hold pays
    nothing, so parking just inside the success band without truly
    settling has no income.
    """
    close = puck_goal_dist(env, asset_cfg) < 1.2 * SUCCESS_DIST
    slow = puck_speed(env, asset_cfg) < SETTLED_SPEED
    return (close & slow).float()


def goal_fine_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a dense centering gradient in the 0-10cm band.

    The rate reward saturates near the goal disc, so nothing else pulls
    the puck the last few centimeters without this term.
    """
    d = puck_goal_dist(env, asset_cfg)
    return torch.exp(-((d / 0.05) ** 2))


def puck_overspeed_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return a penalty for puck speed beyond ``MAX_PUSH_SPEED``."""
    return torch.clamp(puck_speed(env, asset_cfg) - MAX_PUSH_SPEED, min=0.0)


def puck_at_goal(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return success: the puck settled inside the goal disc."""
    close = puck_goal_dist(env, asset_cfg) < SUCCESS_DIST
    slow = puck_speed(env, asset_cfg) < SETTLED_SPEED
    return close & slow


def puck_fell(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return early termination: the puck was knocked off the table (top z=0.40)."""
    return puck_pos_w(env, asset_cfg)[:, 2] < 0.30


def reset_puck_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_range: float = 0.03,
) -> None:
    """Reset the puck to its default pose plus xy jitter.

    Free-body resets must start from the default state (which already
    contains the env origin for attached free bodies) and add jitter on
    top, rather than adding origins a second time.
    """
    puck: Entity = env.scene[asset_cfg.name]
    default = puck.data.default_root_state
    assert default is not None
    state = default[env_ids].clone()
    n = len(env_ids)
    state[:, 0] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 1] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 7:] = 0.0
    puck.write_root_state_to_sim(state, env_ids=env_ids)
    # Seed the rate buffer from the WRITTEN positions: derived kinematics
    # are stale inside reset events.
    #
    # Against the PER-ENV goal, not the module constant. With a single fixed
    # target the two were the same thing; once an instruction can name a
    # different target they are not, and seeding from the constant gives the
    # progress-shaping term a baseline measured to the wrong place. The puck
    # spawns with +-xy_range jitter, so the error is up to a few centimetres
    # and `clamp(min_dist - dist, min=0)` pays it out as free progress on the
    # first contact step -- asymmetrically, for whichever goal is not the
    # constant. The instruction sampler runs first among the reset events, so
    # goal_buf is already correct here.
    goal = goal_buf(env)[env_ids]
    d = torch.linalg.norm(state[:, :2] - goal[:, :2], dim=-1)
    if not hasattr(env, "_puck_min_dist"):
        env._puck_min_dist = puck_goal_dist(env, asset_cfg).clone()
    env._puck_min_dist[env_ids] = d
