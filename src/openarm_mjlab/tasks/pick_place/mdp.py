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

"""MDP terms for the OpenArm pick & place task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common_mdp import (
    both_pads_on_block,
    env_buffer,
    partial_pinch_reward,
    pinch_obs,
    pinch_reward,
    pinch_streak_reward,
    reset_pinch_streak,
    terminated_by,
)


if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "partial_pinch_reward",
    "pinch_obs",
    "pinch_reward",
    "pinch_streak_reward",
    "reset_pinch_streak",
    "terminated_by",
]

# _target_pos_w is on the hot path (2 observation terms + the transport
# reward each step); cache the constant offset tensor per (value, device)
# instead of re-allocating and re-uploading it from a Python tuple.
_OFFSET_CACHE: dict[tuple[tuple[float, float, float], torch.device], torch.Tensor] = {}


def _target_pos_w(
    env: ManagerBasedRlEnv,
    target_name: str,
    target_offset: tuple[float, float, float],
) -> torch.Tensor:
    target: Entity = env.scene[target_name]
    key = (target_offset, target.data.root_link_pos_w.device)
    offset = _OFFSET_CACHE.get(key)
    if offset is None:
        offset = torch.tensor(target_offset, device=key[1])
        _OFFSET_CACHE[key] = offset
    return target.data.root_link_pos_w + offset


##
# Observations.
##


def object_to_target_offset(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Vector from object to an offset point above the target entity (world frame)."""
    obj: Entity = env.scene[object_name]
    return _target_pos_w(env, target_name, target_offset) - obj.data.root_link_pos_w


##
# Rewards.
##


def object_reach_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Gaussian kernel over EE-to-object distance."""
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    err = torch.sum(torch.square(obj.data.root_link_pos_w - ee_pos_w), dim=-1)
    return torch.exp(-err / std**2)


def staged_reach_bring_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    reaching_std: float,
    bringing_std: float,
    asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return ``reaching * (1 + bringing)``, mjlab's lift_cube staged reward.

    Both factors are Gaussian kernels: EE-to-object and object-to-target.
    Bringing pays only through reaching, so moving the object is credited
    while the gripper stays on it, and unlike a binary height gate it
    carries a gradient toward the target from the very first centimeter.
    """
    reaching = object_reach_reward(env, object_name, reaching_std, asset_cfg)
    obj: Entity = env.scene[object_name]
    err = torch.sum(
        torch.square(
            _target_pos_w(env, target_name, target_offset) - obj.data.root_link_pos_w
        ),
        dim=-1,
    )
    bringing = torch.exp(-err / bringing_std**2)
    return reaching * (1.0 + bringing)


def object_lifted(
    env: ManagerBasedRlEnv,
    object_name: str,
    minimum_height: float,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """1.0 while the object's center is above minimum_height (world z).

    With ``sensor_name``, pays only while both finger pads pinch the object,
    so a cube knocked into the air earns nothing.
    """
    obj: Entity = env.scene[object_name]
    lifted = obj.data.root_link_pos_w[:, 2] > minimum_height
    if sensor_name is not None:
        lifted = lifted & both_pads_on_block(env, sensor_name)
    return lifted.float()


def object_transport_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    target_name: str,
    std: float,
    minimum_height: float,
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Gaussian kernel over object-to-target distance, gated on the object being lifted."""
    obj: Entity = env.scene[object_name]
    obj_pos_w = obj.data.root_link_pos_w
    lifted = (obj_pos_w[:, 2] > minimum_height).float()
    err = torch.sum(
        torch.square(_target_pos_w(env, target_name, target_offset) - obj_pos_w), dim=-1
    )
    return lifted * torch.exp(-err / std**2)


def _carried(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the per-env flag: object carried at least once this episode."""
    return env_buffer(
        env,
        "_pick_place_carried",
        lambda: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )


def object_carried(
    env: ManagerBasedRlEnv,
    object_name: str,
    minimum_height: float,
    sensor_name: str,
) -> torch.Tensor:
    """Return whether the object has been carried this episode, updating the flag.

    Carried means pinched by both finger pads while above ``minimum_height``
    at least once since reset. Neither pushing the object along the table
    nor flicking it into the air counts.
    """
    obj: Entity = env.scene[object_name]
    above = obj.data.root_link_pos_w[:, 2] > minimum_height
    carried = _carried(env)
    carried |= above & both_pads_on_block(env, sensor_name)
    return carried


def reset_carried(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Reset event: clear the carried flag for the given envs."""
    _carried(env)[env_ids] = False


def object_in_tray(
    env: ManagerBasedRlEnv,
    object_name: str,
    tray_name: str,
    xy_tolerance: float,
    max_height_above_tray: float,
    carry_min_height: float | None = None,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Boolean mask: object inside the tray footprint and near its bottom.

    With ``carry_min_height``, it also requires :func:`object_carried`, so an
    object pushed into the tray without being carried over the walls does
    not count.
    """
    obj: Entity = env.scene[object_name]
    tray: Entity = env.scene[tray_name]
    delta = obj.data.root_link_pos_w - tray.data.root_link_pos_w
    in_xy = (delta[:, :2].abs() < xy_tolerance).all(dim=-1)
    low = delta[:, 2] < max_height_above_tray
    in_tray = in_xy & low
    if carry_min_height is not None:
        assert sensor_name is not None, "carry gating needs the finger sensor"
        in_tray = in_tray & object_carried(
            env, object_name, carry_min_height, sensor_name
        )
    return in_tray


##
# Terminations.
##


def object_settled_in_tray(
    env: ManagerBasedRlEnv,
    object_name: str,
    tray_name: str,
    xy_tolerance: float,
    max_height_above_tray: float,
    max_speed: float,
    max_ang_speed: float,
    carry_min_height: float | None = None,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Success: object resting inside the tray with near-zero velocity.

    ``carry_min_height``/``sensor_name`` gate it on carrying, as in
    :func:`object_in_tray`.
    """
    in_tray = object_in_tray(
        env,
        object_name,
        tray_name,
        xy_tolerance,
        max_height_above_tray,
        carry_min_height,
        sensor_name,
    )
    obj: Entity = env.scene[object_name]
    slow = torch.norm(obj.data.root_link_lin_vel_w, dim=-1) < max_speed
    calm = torch.norm(obj.data.root_link_ang_vel_w, dim=-1) < max_ang_speed
    return in_tray & slow & calm
