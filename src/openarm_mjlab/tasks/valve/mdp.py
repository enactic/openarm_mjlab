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

"""Valve-turning MDP terms.

Contact-gated RATE reward (state rewards pay retroactively), gained progress
vs. episode start (a randomized start would otherwise pay free income),
anti-reverse penalty (no pumping), and a terminal success bonus (a success
termination without one makes the optimum avoid success).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common_mdp import (
    env_buffer,
    fixture_joint_pos,
    fixture_joint_vel,
    new_progress_rate,
    potential_shaping,
    contact_reward,
    ee_to_target,
    fingers_on_handle,
    reach_target_reward,
    terminated_by,
)


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "contact_reward",
    "ee_to_target",
    "fingers_on_handle",
    "reach_target_reward",
    "terminated_by",
]

TARGET_TURN = 1.35  # rad (77.4 deg): the single-grasp kinematic ceiling.
MAX_TURN_RATE = 1.0  # rad/s


def valve_angle(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the valve hinge angle, radians."""
    return fixture_joint_pos(env, asset_cfg)


def valve_rate(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the valve hinge angular velocity, rad/s."""
    return fixture_joint_vel(env, asset_cfg)


def _start_angle(env) -> torch.Tensor:
    """Return the per-env angle recorded at episode start."""
    return env_buffer(env, "_valve_start_angle")


def _gained_contact(env) -> torch.Tensor:
    """Return rotation accumulated only while fingers touch the grip."""
    return env_buffer(env, "_valve_gained_contact")


def _max_angle(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the per-env running-max angle reached this episode."""
    return env_buffer(
        env, "_valve_max_angle", lambda: valve_angle(env, asset_cfg).clone()
    )


def _prev_progress(env) -> torch.Tensor:
    """Return the previous step's clamped progress fraction, for shaping."""
    return env_buffer(env, "_valve_prev_progress")


def record_valve_start(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset all per-episode valve bookkeeping for the given envs."""
    a = valve_angle(env, asset_cfg)
    _start_angle(env)[env_ids] = a[env_ids]
    _gained_contact(env)[env_ids] = 0.0
    _max_angle(env, asset_cfg)[env_ids] = a[env_ids]
    _prev_progress(env)[env_ids] = 0.0


def turn_gained(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return rotation gained past the episode start (positive direction), rad."""
    return torch.clamp(valve_angle(env, asset_cfg) - _start_angle(env), min=0.0)


def turn_rate_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a contact-gated reward for new angle above the running max.

    Also accumulates gained-under-contact for the honesty check in
    :func:`turned_target`. Credited progress is capped at ``TARGET_TURN`` so
    continuing to turn past the target under contact is never separately
    incentivized once the target is already met.
    """
    contact = fingers_on_handle(env, sensor_name).float()
    a = valve_angle(env, asset_cfg)
    maxa = _max_angle(env, asset_cfg)
    _gained_contact(env).add_(torch.clamp(a - maxa, min=0.0) * contact)
    rate = new_progress_rate(env, a, maxa, MAX_TURN_RATE, cap=TARGET_TURN)
    return rate * contact


def turn_progress_shaping_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Return potential-based shaping on the gained-turn fraction.

    See :func:`potential_shaping`: holding still under contact is never free.
    """
    gate = fingers_on_handle(env, sensor_name).float()
    cur = torch.clamp(turn_gained(env, asset_cfg) / TARGET_TURN, 0.0, 1.0)
    return potential_shaping(cur, _prev_progress(env), gamma) * gate


def reverse_rate_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an anti-pump penalty: backing the valve up is never free."""
    return torch.clamp(-valve_rate(env, asset_cfg), min=0.0)


def overspeed_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a penalty for turning faster than ``MAX_TURN_RATE``."""
    return torch.clamp(valve_rate(env, asset_cfg).abs() - MAX_TURN_RATE, min=0.0)


def turned_target(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return success: target rotation gained, quasi-static, still in contact.

    Requires at least 85% of the gained turn to have been produced under
    contact, so a policy that flings the valve open without a real grip does
    not count as success.
    """
    done = turn_gained(env, asset_cfg) >= TARGET_TURN
    honest = _gained_contact(env) >= 0.85 * TARGET_TURN
    slow = valve_rate(env, asset_cfg).abs() < MAX_TURN_RATE
    return done & honest & slow & fingers_on_handle(env, sensor_name)
