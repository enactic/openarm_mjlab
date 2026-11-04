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

"""Measuring whether a policy follows its instruction, and which part fails.

Success rate cannot answer the question. If the observation determines the
goal, a policy that never reads a word still scores full marks. So every goal
is scored twice: once shown its own instruction, once shown the other goal's,
with the reward scoring the real goal either way. The observation is identical
in both, so only the sentence differs.

That gives a square:

                    shown "right"   shown "left"
      scored right       A               B
      scored left        C               D

and two abilities that the usual scalar (mean(A,D) - mean(B,C)) conflates:

    execution   = mean(A, D)        does it DO the goal it was told?
    suppression = 1 - mean(B, C)    does it WITHHOLD the other one?

Keeping them apart matters because they fail for opposite reasons. Measured
across five training conditions in this repository's history:

    condition                          execution   suppression
    both goals reachable and rewarded      0.958         0.958
    second goal's reward reduced to 0.3    0.958         0.958
    second goal's reward reduced to 0.0    0.459         0.959
    second goal made unreachable           0.456         0.998
    weak text encoder (all-MiniLM-L6-v2)   0.450         0.535

Every reward-side failure keeps suppression near perfect and collapses
execution: the policy knows which goal was named, withholds the other
correctly, and cannot or will not perform it. The encoder failure is the
mirror image -- it withholds nothing, because it never knew which goal was
named. So the pair says which precondition broke, from one evaluation.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

from openarm_mjlab.tasks.language import instructions

# Below this, a goal is treated as one the policy cannot reach. Its
# off-diagonal cell is then zero for a trivial reason and would flatter
# suppression, so it is excluded from that average.
REACHABLE_THRESHOLD = 0.05


@dataclass
class DiscriminationResult:
    """Per-sentence outcomes of the square, and what they say."""

    cells: dict[tuple[str, str], list[float]]
    goals: tuple[str, ...]
    execution: float = 0.0
    suppression: float = 0.0
    separation: float = 0.0
    sign_wins: int = 0
    sign_n: int = 0
    sign_p: float = 1.0
    permutation_p: float = 1.0
    reachable: tuple[str, ...] = ()
    verdict: str = ""
    per_sentence: list[tuple[str, float, float]] = field(default_factory=list)

    def report(self) -> str:
        """Render the result as the lines a human should read."""
        lines = [
            f"execution   {self.execution:.3f}   (does the goal it was told)",
            f"suppression {self.suppression:.3f}   (withholds the other,"
            f" over {len(self.reachable)} reachable goal(s))",
            f"separation  {self.separation:+.3f}",
            f"paired sign test {self.sign_wins}/{self.sign_n} favour the true"
            f" instruction, two-sided p = {self.sign_p:.2e}"
            + ("" if self.sign_wins * 2 >= self.sign_n else "  <-- INVERTED"),
            f"permutation p = {self.permutation_p:.2e}",
            f"-> {self.verdict}",
        ]
        return "\n".join(lines)


def _sign_test(truth: list[float], lie: list[float]) -> tuple[int, int, float]:
    """Paired sign test. Ties dropped, which is the conservative choice."""
    wins = sum(1 for t, other in zip(truth, lie) if t > other)
    losses = sum(1 for t, other in zip(truth, lie) if t < other)
    n = wins + losses
    if n == 0:
        return 0, 0, 1.0
    k = max(wins, losses)
    one_sided = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return wins, n, min(1.0, 2 * one_sided)


def _permutation_p(truth: list[float], lie: list[float], trials: int = 20_000) -> float:
    """Two-sided PAIRED permutation test: exact for small n, sampled otherwise.

    Paired, matching the sign test and the per-sentence design: the null is
    that within each sentence the two labels are exchangeable, so the
    permutation flips signs of the per-sentence differences rather than
    reshuffling the two groups. An unpaired two-sample shuffle throws away the
    pairing the whole measurement is built on.

    `sum()/len()` rather than `statistics.mean`, which is ~25x slower here and
    dominated the test suite.
    """
    diffs = [t - other for t, other in zip(truth, lie)]
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(sum(diffs) / n)
    if n <= 20:  # 2^20 = 1,048,576 sign patterns, still cheap
        hits = 0
        for pattern in range(1 << n):
            total = 0.0
            for i, diff in enumerate(diffs):
                total += -diff if (pattern >> i) & 1 else diff
            hits += abs(total / n) >= observed - 1e-12
        return hits / (1 << n)
    rng = random.Random(0)
    hits = 0
    for _ in range(trials):
        total = 0.0
        for diff in diffs:
            total += diff if rng.random() < 0.5 else -diff
        hits += abs(total / n) >= observed - 1e-12
    return (hits + 1) / (trials + 1)


def summarise(
    cells: dict[tuple[str, str], list[float]],
    goals: tuple[str, ...] = instructions.GOALS,
    split: str = "held_out",
) -> DiscriminationResult:
    """Turn the square into the two abilities and a verdict.

    cells[(scored, shown)] holds one success rate per held-out phrasing.
    Pure, so the statistics can be tested without a simulator.
    """
    if len(goals) != 2:
        raise ValueError(
            f"summarise pairs each goal with one other; got {len(goals)} goals. "
            "Generalise the pairing before adding a third."
        )
    result = DiscriminationResult(cells=cells, goals=goals)

    diagonals = [statistics.mean(cells[(g, g)]) for g in goals]
    result.execution = statistics.mean(diagonals)
    result.reachable = tuple(
        g for g, d in zip(goals, diagonals) if d > REACHABLE_THRESHOLD
    )

    off = [
        statistics.mean(cells[(g, other)])
        for g in result.reachable
        for other in goals
        if other != g
    ]
    result.suppression = 1.0 - statistics.mean(off) if off else float("nan")

    # Pair by SENTENCE: for a phrasing naming goal g, "truth" is success at g
    # while shown it, "lie" is success at the other goal shown that same
    # phrasing. Pairing this way controls for how hard each phrasing is.
    truth, lie = [], []
    for gi, g in enumerate(goals):
        other = goals[1 - gi]
        pool = instructions.TRAIN[g] if split == "train" else instructions.HELD_OUT[g]
        for i, sentence in enumerate(pool):
            hit = cells[(g, g)][i]
            miss = cells[(other, g)][i]
            truth.append(hit)
            lie.append(miss)
            result.per_sentence.append((sentence, hit, miss))
    result.separation = statistics.mean(truth) - statistics.mean(lie)
    result.sign_wins, result.sign_n, result.sign_p = _sign_test(truth, lie)
    result.permutation_p = _permutation_p(truth, lie)

    if result.suppression != result.suppression:  # NaN: nothing reachable
        result.verdict = (
            "no goal is reachable; the task cannot measure instruction following"
        )
    elif result.suppression < 0.7:
        result.verdict = (
            "the instruction is not being read: suppression is near chance. "
            "Check the encoder separates these phrasings "
            "(diagnostics.encoder_separability)"
        )
    elif result.execution < 0.7:
        result.verdict = (
            "the instruction is read but the goal is not performed: suppression "
            "is high, execution is not. Check the goal is reachable and rewarded"
        )
    else:
        result.verdict = "follows the instruction"
    return result


def measure(
    policy,
    env,
    steps: int,
    split: str = "held_out",
    goals: tuple[str, ...] = instructions.GOALS,
    seed: int = 0,
) -> DiscriminationResult:
    """Run the square on a built env and summarise it.

    ``env`` is reconfigured between cells through ``instruction_goal`` and
    ``instruction_shown``, so the reward always scores the real goal while the
    policy is shown whichever phrasing the cell calls for.

    Success is read from the task's own termination term, and from the STORED
    flag rather than live state: mjlab resets a terminated env inside
    ``step()``, so anything read from physics afterwards is the post-reset
    value. Reading the puck's position that way once made every condition
    report the puck at its start position, which looked like a clean result.
    """
    import torch

    num_envs = env.num_envs
    cells: dict[tuple[str, str], list[float]] = {}
    # Restore whatever the caller had, rather than blanking it. Leaving
    # `instruction_split` set to "held_out" would silently contaminate any
    # training done on this env afterwards.
    previous = {
        name: getattr(env, name, None)
        for name in (
            "instruction_goal",
            "instruction_shown",
            "instruction_split",
            "instruction_phrase_index",
        )
    }

    for scored in goals:
        for shown in goals:
            pool = (
                instructions.TRAIN[shown]
                if split == "train"
                else instructions.HELD_OUT[shown]
            )
            if num_envs < len(pool):
                # Otherwise `arange(num_envs) % len(pool)` leaves the tail of
                # the pool with no envs at all, their cell means come out NaN,
                # and the measurement silently omits those phrasings.
                raise ValueError(
                    f"{num_envs} envs cannot cover {len(pool)} phrasings of "
                    f"{shown!r}; use at least {len(pool)}"
                )
            env.instruction_goal = scored
            env.instruction_shown = shown
            env.instruction_split = split
            index = torch.arange(num_envs, device=env.device) % len(pool)
            env.instruction_phrase_index = index

            # Every cell starts from the SAME physics. Without this, cells run
            # sequentially off a global RNG that each one advances, so by the
            # last cell the puck starts up to 0.055 m from where it started in
            # the first -- nearly twice the reset jitter. The per-sentence
            # pairing below assumes a cell and its counterpart differ only in
            # the sentence shown, and that is only true if the states match.
            obs, _ = env.reset(seed=seed)
            finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
            success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
            with torch.no_grad():
                for _ in range(steps):
                    obs, _, terminated, truncated, _ = env.step(policy(obs))
                    hit = env.termination_manager.get_term("puck_at_goal")
                    success |= ~finished & hit
                    finished |= (terminated | truncated).to(torch.bool)
                    if bool(finished.all()):
                        break
            cells[(scored, shown)] = [
                float(success[index == i].float().mean()) for i in range(len(pool))
            ]

    for name, value in previous.items():
        setattr(env, name, value)
    return summarise(cells, goals, split)
