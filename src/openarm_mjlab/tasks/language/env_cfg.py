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

"""The puck task with its target named in language instead of observed."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from openarm_mjlab.tasks.language import mdp as language_mdp
from openarm_mjlab.tasks.language import model
from openarm_mjlab.tasks.puck import mdp as puck_mdp
from openarm_mjlab.tasks.puck.puck_env_cfg import openarm_puck_env_cfg

_MODEL = "openarm_mjlab.tasks.language.model"


def openarm_language_puck_env_cfg(
    play: bool = False, table_path: Path | None = None
) -> ManagerBasedRlEnvCfg:
    """Two named targets in one env, distinguishable only by the instruction.

    ``table_path`` selects the instruction embeddings. It belongs in the
    config rather than on the built env because the width of the observation
    depends on it, so it has to be known before the env is constructed.
    """
    cfg = openarm_puck_env_cfg(play=play)

    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        if "puck_to_goal" not in terms:
            raise ValueError(
                f"puck observation layout changed: no 'puck_to_goal' in {group}"
            )
        old = terms["puck_to_goal"]
        # The goal vector points at the active target, which would make the
        # instruction redundant and any claim about following it vacuous. The
        # midpoint of the two candidates has the same width, the same kind of
        # quantity and the same scale, and is identical for both goals.
        # `replace` rather than a hand-built term: ObservationTermCfg also
        # carries clip, scale, delay and history fields, and rebuilding by
        # hand silently drops any that the puck task later sets.
        terms["puck_to_midpoint"] = replace(old, func=puck_mdp.puck_to_midpoint_obs)
        del terms["puck_to_goal"]
        # No noise on the instruction: it is a symbol an operator supplies,
        # not a sensor reading, and perturbing it would change what the
        # embedding geometry means between training and deployment.
        terms["instruction"] = ObservationTermCfg(
            func=language_mdp.instruction, params={"table_path": table_path}
        )

    events = dict(cfg.events)
    cfg.events.clear()
    # First, so the target is set before any event records against it.
    cfg.events["sample_instruction"] = EventTermCfg(
        func=language_mdp.sample_instruction,
        mode="reset",
        params={"table_path": table_path},
    )
    cfg.events.update(events)

    # The model must read the same table the observation was built from.
    model.LanguageConditionedMLP.table_path = table_path
    return cfg


def openarm_language_puck_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config for the language-conditioned puck task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=f"{_MODEL}.LanguageConditionedMLP",
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
            class_name=f"{_MODEL}.LanguageConditionedCritic",
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
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
        experiment_name="openarm_language_puck",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=1800,
    )
