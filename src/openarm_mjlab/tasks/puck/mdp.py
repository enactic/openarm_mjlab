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

from ...common_mdp import (
    env_buffer,
    fingers_on_handle,
    fingers_on_handle_obs,
    object_pos_w,
    reach_object_reward,
    reset_object_xy_uniform,
    terminated_by,
    to_base_frame,
    tool_to_object_obs,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "fingers_on_handle",
    "fingers_on_handle_obs",
    "reach_object_reward",
    "terminated_by",
    "tool_to_object_obs",
]

# Goal disc, local to the env origin.
GOAL_LOCAL = (0.33, -0.30, 0.422)
SUCCESS_DIST = 0.025  # m, planar.
SETTLED_SPEED = 0.05  # m/s
MAX_PUSH_SPEED = 0.25  # m/s, puck speed cap for the rate reward.


def _goal_w(env) -> torch.Tensor:
    # mjlab batches each env as its OWN world: physics coordinates are
    # raw; env_origins is only a viewer layout grid. Do NOT add origins.
    goal = torch.tensor(GOAL_LOCAL, device=env.device)
    return goal.expand(env.num_envs, 3)


def puck_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the puck's planar speed."""
    puck: Entity = env.scene[asset_cfg.name]
    return torch.linalg.norm(puck.data.root_link_vel_w[:, :2], dim=-1)


def puck_goal_dist(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the planar distance from the puck to the goal center."""
    d = object_pos_w(env, asset_cfg)[:, :2] - _goal_w(env)[:, :2]
    return torch.linalg.norm(d, dim=-1)


def puck_to_goal_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the vector from the puck to the goal, base frame."""
    vec_w = _goal_w(env) - object_pos_w(env, asset_cfg)
    return to_base_frame(env, robot_cfg, vec_w)


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
    min_dist = env_buffer(env, "_puck_min_dist", dist.clone)
    new = torch.clamp(min_dist - dist, min=0.0)
    min_dist.copy_(torch.minimum(min_dist, dist))
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


def reset_puck_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_range: float = 0.03,
) -> None:
    """Reset the puck to its default pose plus xy jitter."""
    state = reset_object_xy_uniform(env, env_ids, asset_cfg, xy_range)
    # Seed the rate buffer from the WRITTEN positions: derived kinematics
    # are stale inside reset events.
    goal = torch.tensor(GOAL_LOCAL, device=env.device)
    d = torch.linalg.norm(state[:, :2] - goal[:2], dim=-1)
    env_buffer(env, "_puck_min_dist", lambda: puck_goal_dist(env, asset_cfg).clone())[
        env_ids
    ] = d
