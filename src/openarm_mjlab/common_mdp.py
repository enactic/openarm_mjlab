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

"""MDP helpers shared across the pedestal-mounted OpenArm tasks.

Contact gates require genuine finger-pad contact rather than a proxy. The
remaining helpers cover what several tasks compute identically: the tool
point, per-env episode buffers, new-progress rates, and free-body state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .openarm_bimanual import GRASP_LOCAL_OFFSET

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


##
# Per-env bookkeeping.
##


def env_buffer(
    env: ManagerBasedRlEnv,
    attr: str,
    init: Callable[[], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return a per-env buffer stored on ``env`` as ``attr``, allocating it lazily.

    ``init`` builds the first value; by default a zero tensor of shape
    ``(num_envs,)``.
    """
    if not hasattr(env, attr):
        value = (
            init() if init is not None else torch.zeros(env.num_envs, device=env.device)
        )
        setattr(env, attr, value)
    return getattr(env, attr)


def new_progress_rate(
    env: ManagerBasedRlEnv,
    value: torch.Tensor,
    running_max: torch.Tensor,
    max_rate: float,
    cap: float | None = None,
) -> torch.Tensor:
    """Return the capped rate of new progress above the episode's running max (0..1).

    New-progress-only, so a plain rate reward cannot be farmed by bouncing
    or pumping. With ``cap``, progress is credited only up to ``cap``, so
    nothing pays for pushing past the target. ``running_max`` is updated in
    place and always tracks the TRUE (uncapped) maximum.
    """
    if cap is None:
        new = torch.clamp(value - running_max, min=0.0)
    else:
        new = torch.clamp(
            torch.clamp(value, max=cap) - torch.clamp(running_max, max=cap), min=0.0
        )
    running_max.copy_(torch.maximum(running_max, value))
    return torch.clamp(new / env.step_dt, 0.0, max_rate) / max_rate


def potential_shaping(
    cur: torch.Tensor, prev: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Return potential-based shaping (Ng, Harada & Russell 1999).

    ``reward = gamma * Phi(s') - Phi(s)``. Camping-negative by construction:
    a fixed state pays ``(gamma - 1) * Phi < 0`` every step instead of
    paying nothing-or-positive. ``prev`` is updated in place to ``cur``.
    """
    shaping = gamma * cur - prev
    prev.copy_(cur)
    return shaping


def terminated_by(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
    """Return 1.0 on the step the named termination term fires.

    Terminations are computed before rewards each step, so the manager's
    cached result is current; referencing it keeps the success bonus and the
    success condition structurally identical, instead of restating the
    predicate in two places that can drift apart.
    """
    return env.termination_manager.get_term(term_name).float()


##
# Finger contact.
##


def fingers_on_handle(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return True where a finger pad touches the handle/grip.

    Any-pad contact, not a two-pad pinch: this gripper's closed cage leaves
    a real gap for slim handles, so even a scripted approach caging the
    handle presses only one pad against it. Progress rewards are gated on
    this (sustained contact required, a flick loses contact and forfeits
    every later step) rather than on the pinch itself.
    """
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    return (found.view(env.num_envs, -1) > 0).any(dim=1)


def fingers_on_handle_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Observation wrapper for :func:`fingers_on_handle`."""
    return fingers_on_handle(env, sensor_name).float().unsqueeze(-1)


def contact_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return a dense reward for holding the handle/grip in contact."""
    return fingers_on_handle(env, sensor_name).float()


def _pads_touching(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return per-pad contact, shape ``(num_envs, 2)``."""
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    return found.view(env.num_envs, 2, -1).amax(dim=-1) > 0


def both_pads_on_block(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return True where BOTH finger pads touch the object.

    A genuine squeeze, not a poke. Used by tasks whose object is wide enough
    for a real friction pinch (unlike a slim handle bar).
    """
    return _pads_touching(env, sensor_name).all(dim=1)


def pinch_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Observation wrapper for :func:`both_pads_on_block`."""
    return both_pads_on_block(env, sensor_name).float().unsqueeze(-1)


def pinch_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return a dense reward for holding a genuine bilateral pinch."""
    return both_pads_on_block(env, sensor_name).float()


def partial_pinch_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Return a graded predecessor to the binary both-pads pinch.

    0, 0.5, or 1.0 for how many of the two pads currently touch the
    object. The binary AND in :func:`both_pads_on_block` is an all-or-
    nothing gate with no signal for "one pad landed, still working on
    the other"; this term is purely additive, so success and termination
    semantics (which require the real two-pad pinch) are unchanged.
    """
    return _pads_touching(env, sensor_name).float().mean(dim=-1)


def _pinch_streak(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the per-env count of consecutive steps holding a pinch."""
    return env_buffer(env, "_pinch_streak")


def pinch_streak_reward(
    env: ManagerBasedRlEnv, sensor_name: str, cap: float = 25.0
) -> torch.Tensor:
    """Return a graded bonus for SUSTAINED bilateral contact, not just instantaneous contact.

    :func:`pinch_reward` and :func:`partial_pinch_reward` pay identically
    for a 2-step graze or a 100-step hold, so repeated brief pecks
    accumulate reward comparable to a genuine sustained grip without ever
    committing to the harder, more precise control a real lift needs.
    This term makes duration itself pay: the streak resets to 0 the
    instant contact breaks, so a peck barely registers, while a genuine
    hold ramps up to full credit over ``cap`` steps. The default ~0.5 s
    (25 steps at 50 Hz) is roughly the time a lift itself takes.
    """
    pinched = both_pads_on_block(env, sensor_name)
    streak = _pinch_streak(env)
    streak[pinched] += 1.0
    streak[~pinched] = 0.0
    return torch.clamp(streak / cap, 0.0, 1.0)


def reset_pinch_streak(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Reset event: clear the pinch streak for the given envs."""
    _pinch_streak(env)[env_ids] = 0.0


##
# Tool point.
##


def to_base_frame(
    env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, vec_w: torch.Tensor
) -> torch.Tensor:
    """Rotate a world-frame vector into the robot base frame.

    Observations expressed in the base frame do not move with the world
    origin.
    """
    robot: Entity = env.scene[robot_cfg.name]
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def tool_pos_w(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the world position of the finger-cage center (the tool point).

    ``robot_cfg`` selects the end-effector site; the tool point is that site
    displaced by ``GRASP_LOCAL_OFFSET``, so reach targets land in the middle
    of the open fingers rather than at the wrist.
    """
    robot: Entity = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, robot_cfg.site_ids].squeeze(1)
    offset = torch.tensor(GRASP_LOCAL_OFFSET, device=ee_pos_w.device).expand_as(
        ee_pos_w
    )
    return ee_pos_w + quat_apply(ee_quat_w, offset)


def tool_to_point(
    env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, point_w: torch.Tensor
) -> torch.Tensor:
    """Return the vector from the tool point to ``point_w``, base frame."""
    return to_base_frame(env, robot_cfg, point_w - tool_pos_w(env, robot_cfg))


def gaussian_reach_reward(vec: torch.Tensor, std: float) -> torch.Tensor:
    """Return a Gaussian kernel on the length of ``vec``."""
    d2 = torch.sum(torch.square(vec), dim=-1)
    return torch.exp(-d2 / std**2)


def ee_to_target(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the vector from the tool point to a target site, base frame.

    ``target_cfg`` selects the site to reach for -- a drawer handle, a door
    handle, a valve grip.
    """
    target: Entity = env.scene[target_cfg.name]
    target_pos_w = target.data.site_pos_w[:, target_cfg.site_ids].squeeze(1)
    return tool_to_point(env, robot_cfg, target_pos_w)


def reach_target_reward(
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a Gaussian-kernel reward on the tool-to-target-site distance."""
    return gaussian_reach_reward(ee_to_target(env, robot_cfg, target_cfg), std)


def tool_to_object_obs(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the vector from the tool point to a free object's root, base frame."""
    return tool_to_point(env, robot_cfg, object_pos_w(env, asset_cfg))


def reach_object_reward(
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return a Gaussian-kernel reward on the tool-to-object distance."""
    return gaussian_reach_reward(tool_to_object_obs(env, robot_cfg, asset_cfg), std)


##
# Articulated fixtures (single-joint valve/door/drawer).
##


def fixture_joint_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return the position of a fixture's single joint, shape ``(num_envs,)``."""
    fixture: Entity = env.scene[asset_cfg.name]
    return fixture.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)


def fixture_joint_vel(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return the velocity of a fixture's single joint, shape ``(num_envs,)``."""
    fixture: Entity = env.scene[asset_cfg.name]
    return fixture.data.joint_vel[:, asset_cfg.joint_ids].squeeze(-1)


##
# Free objects (block/puck/bar).
##


def object_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a free object's root world position."""
    obj: Entity = env.scene[asset_cfg.name]
    return obj.data.root_link_pos_w


def object_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a free object's 3D linear speed."""
    obj: Entity = env.scene[asset_cfg.name]
    return torch.linalg.norm(obj.data.root_link_vel_w[:, :3], dim=-1)


def descent_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return an anti-pump penalty: lowering the object is never free."""
    obj: Entity = env.scene[asset_cfg.name]
    return torch.clamp(-obj.data.root_link_vel_w[:, 2], min=0.0)


def reset_object_xy_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_range: float,
) -> torch.Tensor:
    """Reset a free object to its default pose plus xy jitter, at rest.

    Free-body resets start from the default state (raw coordinates: mjlab
    batches each env as its own world) and add jitter on top, rather than
    adding env origins a second time. Returns the written root state, since
    derived kinematics are stale inside reset events.
    """
    obj: Entity = env.scene[asset_cfg.name]
    default = obj.data.default_root_state
    assert default is not None
    state = default[env_ids].clone()
    n = len(env_ids)
    state[:, 0] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 1] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 7:] = 0.0
    obj.write_root_state_to_sim(state, env_ids=env_ids)
    return state
