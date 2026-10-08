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

"""PPO runner configuration shared by all five MultiTask registrations."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

MULTITASK_EXPERIMENT_NAME = "openarm_multitask"


def multitask_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return the runner config for the shared multi-task policy.

    Architecture and hyperparameters match the single-task configurations;
    what makes this one shared policy is that all five registrations use the
    same experiment name, the actor carries one output head per task, and the
    algorithm standardizes advantages per task.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=(
                "openarm_mjlab.tasks.multitask.multihead_model.MultiHeadMLPModel"
            ),
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        # The critic stays single-headed: it is never deployed, and a value
        # function shared across tasks acts as a regulariser.
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name="openarm_mjlab.tasks.multitask.ppo.MultiTaskPPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=MULTITASK_EXPERIMENT_NAME,
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=3000,
    )
