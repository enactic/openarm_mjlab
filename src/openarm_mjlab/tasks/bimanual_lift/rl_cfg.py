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

"""PPO runner config for the OpenArm bimanual lift task.

Hyperparameters are the same ones the single-arm lift task uses; only the
reward and environment design differ between the two.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def openarm_bimanual_lift_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Build the PPO runner config for the bimanual lift task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # Raising this was tried (0.03) to break the plateau where reach
            # and grip saturate while the lift signal stays flat. It neither
            # unlocked lifting nor cost grip precision, so the proven value
            # is kept.
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
        # The two finger actuators are genuinely identical in the asset
        # (both capped at 7 N*m, verified against the compiled model), so
        # their action scales match; an earlier asymmetric scale gave the
        # left gripper ~43% more effective torque per unit of policy output
        # and produced a lopsided grasp. See LEFT/RIGHT_SQUEEZE_SCALE.
        experiment_name="openarm_bimanual_lift",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=3_000,
    )
