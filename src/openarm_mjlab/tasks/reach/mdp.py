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

"""Reach task MDP: drive the tool point to a random workspace target."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common_mdp import env_buffer, terminated_by, tool_pos_w, tool_to_point


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["terminated_by"]

# Right-arm workspace box for targets (raw world coordinates; every env is
# its own batched world, not a viewer-layout offset).
TARGET_LO = (0.25, -0.35, 0.45)
TARGET_HI = (0.45, -0.05, 0.65)
SUCCESS_DIST = 0.02
SETTLED_QVEL = 0.3  # rad/s max arm joint speed counted as "held".


def _sample_targets(env: ManagerBasedRlEnv, n: int) -> torch.Tensor:
    """Return ``n`` targets drawn uniformly from the workspace box."""
    lo = torch.tensor(TARGET_LO, device=env.device)
    hi = torch.tensor(TARGET_HI, device=env.device)
    return lo + (hi - lo) * torch.rand(n, 3, device=env.device)


def _targets(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Lazily allocate and return the per-env target buffer."""
    return env_buffer(env, "_reach_targets", lambda: _sample_targets(env, env.num_envs))


def resample_targets(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Reset event: draw a fresh random target for the given envs."""
    _targets(env)[env_ids] = _sample_targets(env, len(env_ids))


def target_dist(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Euclidean distance from the tool point to its current target."""
    return torch.linalg.norm(tool_pos_w(env, robot_cfg) - _targets(env), dim=-1)


def tool_to_target_obs(
    env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Target-minus-tool vector, in the robot's own base frame."""
    return tool_to_point(env, robot_cfg, _targets(env))


def reach_reward(
    env: ManagerBasedRlEnv, std: float, robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Gaussian-kernel dense reward on distance-to-target."""
    d = target_dist(env, robot_cfg)
    return torch.exp(-((d / std) ** 2))


def arm_settled(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return True where every joint's speed is below the settled threshold."""
    robot: Entity = env.scene[robot_cfg.name]
    return robot.data.joint_vel.abs().max(dim=-1).values < SETTLED_QVEL


def hold_at_target_reward(
    env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return a dense reward for staying near the target.

    Uses a looser band than success, so it does not collapse into a
    one-step reward paid only on termination.
    """
    close = target_dist(env, robot_cfg) < 2 * SUCCESS_DIST
    return close.float()


def reached(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Success termination: within SUCCESS_DIST of the target and settled."""
    close = target_dist(env, robot_cfg) < SUCCESS_DIST
    return close & arm_settled(env, robot_cfg)
