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

"""Sampling an instruction, and the observation term that carries it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from openarm_mjlab.tasks.language import embeddings, instructions
from openarm_mjlab.tasks.puck import mdp as puck_mdp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _instruction_buf(env, table_path: Path | None) -> torch.Tensor:
    if not hasattr(env, "_instruction"):
        width = embeddings.dim(table_path)
        env._instruction = torch.zeros(env.num_envs, width, device=env.device)
    return env._instruction


def _goal_buf(env) -> torch.Tensor:
    if not hasattr(env, "_instruction_goal"):
        env._instruction_goal = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    return env._instruction_goal


def _generator(env) -> torch.Generator:
    """Return a dedicated RNG stream for instruction sampling.

    Drawing instructions from the global RNG couples them to the physics.
    Evaluation conditions draw different numbers of values, so every
    condition would get different initial states as well as a different
    instruction, and the comparison would confound the two.
    """
    if not hasattr(env, "_instruction_gen"):
        gen = torch.Generator(device=env.device)
        gen.manual_seed(
            int(getattr(env, "instruction_seed", torch.initial_seed() % (2**31)))
        )
        env._instruction_gen = gen
    return env._instruction_gen


def instruction(env: ManagerBasedRlEnv, table_path: Path | None = None) -> torch.Tensor:
    """Observation term: this episode's instruction, as an embedding."""
    return _instruction_buf(env, table_path)


def goal_index(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Which goal each env is scored on. For metrics only, never observed."""
    return _goal_buf(env)


def sample_instruction(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor, table_path: Path | None = None
) -> None:
    """Reset event: choose a goal, then a way of saying it.

    Three attributes on the env steer this, and exist so evaluation can vary
    one thing at a time without touching the policy:

    instruction_goal
        force every env onto one goal, so a condition scores just that goal.
    instruction_shown
        show a DIFFERENT goal's phrasing while the reward still scores the
        real one. This is the control that decides whether any of it means
        anything: unchanged success here means the policy is not reading the
        instruction.
    instruction_split
        draw phrasings from "train" or from "held_out".
    """
    if len(env_ids) == 0:
        return

    goals = instructions.GOALS
    # `or` rather than a getattr default: the attribute may be present and
    # None -- that is what `measure()` restores when the caller never set it,
    # and a plain default would not fall back, leaving `pool(goal, None)` to
    # raise on the next reset.
    split = getattr(env, "instruction_split", None) or "train"
    gen = _generator(env)

    forced = getattr(env, "instruction_goal", None)
    if forced is not None:
        picked = torch.full(
            (len(env_ids),), goals.index(forced), device=env.device, dtype=torch.long
        )
    else:
        picked = torch.randint(
            len(goals), (len(env_ids),), device=env.device, generator=gen
        )
    _goal_buf(env)[env_ids] = picked

    shown_name = getattr(env, "instruction_shown", None)
    if shown_name is not None:
        shown = torch.full(
            (len(env_ids),),
            goals.index(shown_name),
            device=env.device,
            dtype=torch.long,
        )
    else:
        shown = picked

    buf = _instruction_buf(env, table_path)
    for gi, goal in enumerate(goals):
        mask = shown == gi
        if not bool(mask.any()):
            continue
        candidates = embeddings.pool(goal, split, env.device, table_path)
        forced_index = getattr(env, "instruction_phrase_index", None)
        if forced_index is not None:
            # Per-env choice of WHICH phrasing, so one env can evaluate a
            # whole pool at once instead of rebuilding the simulator per
            # sentence.
            choice = forced_index.to(env.device)[env_ids[mask]] % len(candidates)
        else:
            choice = torch.randint(
                len(candidates), (int(mask.sum()),), device=env.device, generator=gen
            )
        buf[env_ids[mask]] = candidates[choice]

    # The reward scores the goal that was PICKED, not the one that was shown.
    for gi, goal in enumerate(goals):
        mask = picked == gi
        if bool(mask.any()):
            puck_mdp.set_puck_goal(env, env_ids[mask], goal)
