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

"""Observation terms that make five separate tasks share one policy input.

Reach, valve, door, drawer and puck already share an identical
``joint_pos(18)`` / ``joint_vel(18)`` / ``actions(8)`` structure; only each
task's own object-relative block differs in width (reach 3, valve/door/drawer
5, puck 7). Padding those blocks into disjoint per-task slots and appending a
task-identity one-hot gives every task the same observation shape, which is
the one thing a single shared network genuinely requires. Rewards,
terminations and actions are left untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

TASK_NAMES = ("reach", "valve", "door", "drawer", "puck")


def task_id_onehot(env: ManagerBasedRlEnv, index: int) -> torch.Tensor:
    """Return a constant one-hot task identity, the goal-conditioning signal."""
    onehot = torch.zeros(env.num_envs, len(TASK_NAMES), device=env.device)
    onehot[:, index] = 1.0
    return onehot


def zero_pad(env: ManagerBasedRlEnv, width: int) -> torch.Tensor:
    """Return constant zero padding that reserves one task's observation slot."""
    return torch.zeros(env.num_envs, width, device=env.device)
