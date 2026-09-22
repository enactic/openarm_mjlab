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

"""A vectorised environment whose every batch spans all five tasks at once.

Training the tasks one at a time and varying the schedule -- visit length,
ordering, weighting -- cannot remove catastrophic forgetting, because
sequential fine-tuning is the structure that produces it; in every such run
whatever trained last was the only thing left. Putting rollouts from all five
tasks in every PPO batch removes the mechanism instead of mitigating it: the
gradient always carries all five, so there is nothing to forget between them.

This owns one ``ManagerBasedRlEnv`` per task and presents them to rsl_rl as a
single vectorised environment. The model needs no change, because
``MultiHeadMLPModel`` already selects its head per sample from the task
one-hot that ``env_cfgs._unify`` puts in the observation.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

MULTITASK_IDS = (
    "OpenArm-MultiTask-Reach",
    "OpenArm-MultiTask-Valve",
    "OpenArm-MultiTask-Door",
    "OpenArm-MultiTask-Drawer",
    "OpenArm-MultiTask-Puck",
)


class MultiTaskMixtureVecEnv:
    """One vectorised environment whose batch spans every task at once."""

    def __init__(
        self,
        num_envs_total: int,
        device: str = "cuda:0",
        task_ids: tuple[str, ...] = MULTITASK_IDS,
        disable_corruption: bool = False,
        weights: tuple[float, ...] | None = None,
    ) -> None:
        """Allocate environments across tasks, optionally unequally.

        An equal split starves the hardest task. Standalone drawer needed 1024
        envs for 3000 iterations to solve; an even five-way split of 1000 envs
        gives it 200. ``weights`` is the mixture's equivalent of task
        weighting, except that here every task still appears in every batch, so
        buying one task more data costs the others throughput rather than
        retention.
        """
        count = len(task_ids)
        if weights is None:
            counts = [num_envs_total // count] * count
        else:
            if len(weights) != count:
                raise ValueError(
                    f"expected one weight per task, got {len(weights)} for {count}"
                )
            total = sum(weights)
            counts = [max(1, int(num_envs_total * w / total)) for w in weights]
        if min(counts) <= 0:
            raise ValueError(f"{num_envs_total} envs cannot cover {count} tasks")

        self.task_ids = task_ids
        self.counts = counts
        self.envs: list[ManagerBasedRlEnv] = []
        for task, n in zip(task_ids, counts):
            cfg = load_env_cfg(task)
            cfg.scene.num_envs = n
            if disable_corruption and "actor" in cfg.observations:
                cfg.observations["actor"].enable_corruption = False
            self.envs.append(ManagerBasedRlEnv(cfg=cfg, device=device))

        self.num_envs = sum(counts)
        # Row ranges per task, since the split need not be uniform.
        self.bounds: list[tuple[int, int]] = []
        offset = 0
        for n in counts:
            self.bounds.append((offset, offset + n))
            offset += n
        self.device = torch.device(device)
        # Tasks differ in episode length; each sub-env still enforces its own,
        # this is only what the runner reports.
        self.max_episode_length = max(e.max_episode_length for e in self.envs)
        self.num_actions = self.envs[0].action_manager.total_action_dim
        self.cfg = self.envs[0].cfg
        self.unwrapped = self.envs[0]

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Return the concatenated per-environment episode step counters."""
        return torch.cat([e.episode_length_buf for e in self.envs])

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        for i, env in enumerate(self.envs):
            low, high = self.bounds[i]
            env.episode_length_buf = value[low:high]

    def _pack(self, per_env_obs: list[dict]) -> TensorDict:
        groups = per_env_obs[0].keys()
        merged = {
            group: torch.cat([obs[group] for obs in per_env_obs], dim=0)
            for group in groups
        }
        return TensorDict(merged, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        """Return the current observation for every task's environments."""
        return self._pack([e.observation_manager.compute() for e in self.envs])

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset every task's environments and return the packed observation."""
        observations: list[dict] = []
        extras: dict = {}
        for env in self.envs:
            obs, extra = env.reset()
            observations.append(obs)
            extras = self._merge_extras(extras, extra)
        return self._pack(observations), extras

    @staticmethod
    def _merge_extras(acc: dict, new: dict) -> dict:
        """Concatenate per-environment tensors and average scalar statistics."""
        for key, value in new.items():
            if isinstance(value, dict):
                acc[key] = MultiTaskMixtureVecEnv._merge_extras(acc.get(key, {}), value)
            elif torch.is_tensor(value) and value.dim() >= 1:
                acc[key] = torch.cat([acc[key], value]) if key in acc else value
            elif torch.is_tensor(value):
                acc[key] = (acc[key] + value) / 2 if key in acc else value
            else:
                acc[key] = value
        return acc

    def step(self, actions: torch.Tensor):
        """Step every task's environments with its slice of the action batch."""
        observations, rewards, dones = [], [], []
        extras: dict = {}
        for i, env in enumerate(self.envs):
            low, high = self.bounds[i]
            obs, reward, terminated, truncated, extra = env.step(actions[low:high])
            observations.append(obs)
            rewards.append(reward)
            dones.append((terminated | truncated).to(dtype=torch.long))
            if not env.cfg.is_finite_horizon:
                extra["time_outs"] = truncated
            extras = self._merge_extras(extras, extra)
        return (
            self._pack(observations),
            torch.cat(rewards),
            torch.cat(dones),
            extras,
        )

    def task_index_of_batch(self) -> torch.Tensor:
        """Return which task each row of the batch belongs to."""
        return torch.cat(
            [
                torch.full((n,), i, device=self.device, dtype=torch.long)
                for i, n in enumerate(self.counts)
            ]
        )

    def close(self) -> None:
        """Close every task's environments."""
        for env in self.envs:
            env.close()
