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

"""PPO with per-task advantage normalization, for mixed multi-task batches."""

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from openarm_mjlab.tasks.multitask.multihead_model import NUM_TASKS


class MultiTaskPPO(PPO):
    """PPO that standardizes advantages within each task rather than globally.

    Stock PPO standardizes advantages with one mean and std over the whole
    batch. With five tasks of different reward scales in every batch, the task
    with the largest advantages dominates and the rest are scaled into noise.
    That was measured on the first concurrent run: reach alone learned while
    the other four sat at exactly 0.000 for 1000 iterations, and the batch mean
    reward, 17.05, sat right at reach's standalone value of 19.65.

    Tasks are identified from the one-hot that ``env_cfgs._unify`` appends to
    the observation, so this stays correct no matter how the mixture allocates
    environments -- including the deliberately unequal splits that give a
    harder task more of them.
    """

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute returns, then standardize advantages within each task."""
        # Suppress the base class's single global pass; the per-task pass below
        # replaces it. Restored immediately so minibatch behaviour is unchanged.
        previous = self.normalize_advantage_per_mini_batch
        self.normalize_advantage_per_mini_batch = True
        try:
            super().compute_returns(obs)
        finally:
            self.normalize_advantage_per_mini_batch = previous

        task_index = self._task_index(obs)
        if task_index is None:
            # No task one-hot present: fall back to the standard global pass so
            # this class stays usable on a single-task environment.
            advantages = self.storage.advantages
            self.storage.advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )
            return

        advantages = self.storage.advantages
        for task in range(NUM_TASKS):
            rows = task_index == task
            if not bool(rows.any()):
                continue
            block = advantages[:, rows]
            advantages[:, rows] = (block - block.mean()) / (block.std() + 1e-8)

    @staticmethod
    def _task_index(obs: TensorDict):
        """Return the per-environment task index, or None if unavailable."""
        actor_obs = obs.get("actor", None)
        if actor_obs is None or actor_obs.shape[-1] < NUM_TASKS:
            return None
        return actor_obs[..., -NUM_TASKS:].argmax(dim=-1)
