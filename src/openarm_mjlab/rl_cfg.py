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

"""PPO runner config shared by every OpenArm task.

All tasks train with the same PPO hyperparameters and actor/critic MLP; a
task passes only what it genuinely changes (experiment name, iteration
budget, and, for the vision variant, the model architecture).
"""

from __future__ import annotations

from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def ppo_runner_cfg(
    experiment_name: str,
    *,
    max_iterations: int = 3_000,
    entropy_coef: float = 0.01,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    model_kwargs: dict[str, Any] | None = None,
    obs_groups: dict[str, tuple[str, ...]] | None = None,
) -> RslRlOnPolicyRunnerCfg:
    """Return the shared PPO runner config.

    ``model_kwargs`` is forwarded to both the actor and the critic
    :class:`RslRlModelCfg` (e.g. ``cnn_cfg``/``class_name`` for a vision
    encoder). ``obs_groups`` overrides the runner's default actor/critic
    observation groups.
    """
    model_kwargs = model_kwargs or {}
    runner_kwargs: dict[str, Any] = {}
    if obs_groups is not None:
        runner_kwargs["obs_groups"] = obs_groups
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            **model_kwargs,
        ),
        critic=RslRlModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=True,
            **model_kwargs,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=entropy_coef,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=max_iterations,
        **runner_kwargs,
    )
