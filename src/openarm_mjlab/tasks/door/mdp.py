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

"""Door-swing MDP terms: the same contact-gated hinge recipe used for valve.

Applies the same playbook (contact-gated rate reward, gained progress vs.
episode start, anti-reverse penalty, a terminal success bonus) to the door
hinge instead of the valve stem.
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
    contact_reward,
    ee_to_target,
    fingers_on_handle,
    fingers_on_handle_obs,
    reach_target_reward,
    terminated_by,
)


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "contact_reward",
    "ee_to_target",
    "fingers_on_handle",
    "fingers_on_handle_obs",
    "reach_target_reward",
    "terminated_by",
]

TARGET_SWING = 1.0  # rad (57.3 deg; classical weld-assisted: 53.7 deg)
MAX_SWING_RATE = 1.0  # rad/s


def door_angle(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the door hinge angle, radians."""
    return fixture_joint_pos(env, asset_cfg)


def door_rate(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the door hinge angular velocity, rad/s."""
    return fixture_joint_vel(env, asset_cfg)


def _start_angle(env) -> torch.Tensor:
    """Return the per-env angle recorded at episode start."""
    return env_buffer(env, "_door_start_angle")


def _gained_contact(env) -> torch.Tensor:
    """Return swing accumulated only while fingers touch the handle."""
    return env_buffer(env, "_door_gained_contact")


def _prev_angle(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the previous step's door angle, for rate bookkeeping."""
    return env_buffer(
        env, "_door_prev_angle", lambda: door_angle(env, asset_cfg).clone()
    )


def _max_angle(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the per-env running-max angle reached this episode."""
    return env_buffer(
        env, "_door_max_angle", lambda: door_angle(env, asset_cfg).clone()
    )


def record_door_start(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Reset all per-episode door bookkeeping for the given envs."""
    a = door_angle(env, asset_cfg)  # qpos read: safe inside reset events.
    _start_angle(env)[env_ids] = a[env_ids]
    _gained_contact(env)[env_ids] = 0.0
    _prev_angle(env, asset_cfg)[env_ids] = a[env_ids]
    _max_angle(env, asset_cfg)[env_ids] = a[env_ids]


def swing_gained(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return rotation gained past the episode start (positive direction), rad."""
    return torch.clamp(door_angle(env, asset_cfg) - _start_angle(env), min=0.0)


def swing_rate_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a contact-gated, capped reward for new angle above the running max.

    Also accumulates gained-under-contact for the honesty check in
    :func:`swung_target`. Runs exactly once per step, after physics.
    """
    contact = fingers_on_handle(env, sensor_name).float()
    a = door_angle(env, asset_cfg)
    prev = _prev_angle(env, asset_cfg)
    _gained_contact(env).add_(torch.clamp(a - prev, min=0.0) * contact)
    prev.copy_(a)
    rate = new_progress_rate(env, a, _max_angle(env, asset_cfg), MAX_SWING_RATE)
    return rate * contact


def uncontrolled_motion_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a penalty for any door motion while NOT holding the handle.

    The panel sits in the approach corridor, so crashing through it opens
    the door as a free collateral of fast reaching; this makes that path
    never free.
    """
    contact = fingers_on_handle(env, sensor_name).float()
    return door_rate(env, asset_cfg).abs() * (1.0 - contact)


def reverse_rate_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an anti-pump penalty: swinging the door back is never free."""
    return torch.clamp(-door_rate(env, asset_cfg), min=0.0)


def overspeed_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a penalty for swinging faster than ``MAX_SWING_RATE``."""
    return torch.clamp(door_rate(env, asset_cfg).abs() - MAX_SWING_RATE, min=0.0)


def swung_target(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return success: target swing gained, quasi-static, still in contact.

    Requires at least 85% of the gained swing to have been produced under
    contact, so a policy that smacks the panel open then taps the handle
    does not count as success.
    """
    done = swing_gained(env, asset_cfg) >= TARGET_SWING
    honest = _gained_contact(env) >= 0.85 * TARGET_SWING
    slow = door_rate(env, asset_cfg).abs() < MAX_SWING_RATE
    return done & honest & slow & fingers_on_handle(env, sensor_name)
