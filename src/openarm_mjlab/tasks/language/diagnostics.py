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

"""Cheap checks to run BEFORE training a language-conditioned policy.

Three things have to be true before instruction following can emerge at all.
Each is answerable without a GPU-hour, and each has a measured failure behind
it in this repository's history.

**1. Can the instruction be decoded, by the readout the policy actually has?**
encoder_separability fits a probe on the training phrasings and scores it
on held-out ones. Use nearest_prototype=True, the default: it scores
similarity to a goal prototype, which is what a conditioning path can extract
without supervision. An unconstrained classifier is too generous -- measured,
it scored two encoders at 0.917 and 0.875 while the policies they produced
separated at +0.917 and +0.006. The constrained score put them at 0.917 and
0.708, which tracks.

**2. Can the goal be decoded WITHOUT the instruction?** If the observation
already determines what to do, the policy has no reason to read the sentence
and will not. observation_ambiguity fits a classifier from the first
observation of each episode to the goal; at chance means language is the only
route, near 1.0 means any result from the task is vacuous.

**3. Is the alternative goal REACHABLE?** Not "is it rewarded enough" --
reachable. If the policy can never reach one goal, routing correctly and
routing wrongly both earn nothing and instruction following becomes a coin
flip. There is no cheap offline form of this one: train the single goal on its
own briefly and see whether it succeeds. Measured on a rotational task whose
second direction turned out kinematically blocked, following emerged in one
run of three; on a task where both goals are reachable it replicated in every
seed tried. A goal paying 30% of full reward behaved the same as one paying
100%, so the quantity that matters is reachability, not reward size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from openarm_mjlab.tasks.language import embeddings, instructions


def encoder_separability(
    nearest_prototype: bool = True,
    split: str = "held_out",
    table_path=None,
    seeds: int = 5,
) -> dict[str, float]:
    """Score how well held-out phrasings of each goal can be told apart.

    Returns per-goal accuracy plus "overall". Chance is 1/len(GOALS).
    """
    goals = instructions.GOALS
    train = [embeddings.pool(g, "train", path=table_path) for g in goals]
    test = [embeddings.pool(g, split, path=table_path) for g in goals]

    if nearest_prototype:
        # The MEAN of the training pool, not its first sentence. Using one
        # arbitrary phrasing makes the score depend on list order: measured on
        # this pool, MiniLM scores 0.708 with the first sentence and 0.833 with
        # the mean, and single sentences span 0.583-0.917. Match this to
        # whatever prototype your gate actually compares against, and if it
        # compares against one sentence, know that the number moves.
        prototypes = F.normalize(torch.stack([t.mean(0) for t in train]), dim=-1)
        preds = [F.normalize(t, dim=-1) @ prototypes.t() for t in test]
        per_goal = {
            g: float((p.argmax(-1) == gi).float().mean())
            for gi, (g, p) in enumerate(zip(goals, preds))
        }
    else:
        x_tr = torch.cat(train)
        y_tr = torch.cat([torch.full((len(t),), i) for i, t in enumerate(train)])
        accs: dict[str, list[float]] = {g: [] for g in goals}
        for seed in range(seeds):
            torch.manual_seed(seed)
            net = torch.nn.Linear(x_tr.shape[1], len(goals))
            torch.nn.init.zeros_(net.weight)
            torch.nn.init.zeros_(net.bias)
            opt = torch.optim.Adam(net.parameters(), lr=0.05)
            for _ in range(2000):
                opt.zero_grad()
                (
                    F.cross_entropy(net(x_tr), y_tr) + 1e-3 * net.weight.pow(2).sum()
                ).backward()
                opt.step()
            with torch.no_grad():
                for gi, (g, t) in enumerate(zip(goals, test)):
                    accs[g].append(float((net(t).argmax(-1) == gi).float().mean()))
        per_goal = {g: sum(v) / len(v) for g, v in accs.items()}

    counts = [len(t) for t in test]
    per_goal["overall"] = sum(per_goal[g] * n for g, n in zip(goals, counts)) / sum(
        counts
    )
    return per_goal


def observation_ambiguity(
    observations: torch.Tensor,
    goals: torch.Tensor,
    hidden: int = 64,
    seeds: int = 3,
) -> float:
    """Held-out accuracy of a classifier from observation to goal.

    observations must EXCLUDE the instruction and should come from the
    first step of each episode. Later in an episode the state reflects the
    policy's own goal-driven choices, so the goal becomes inferable from
    consequences rather than from anything the policy needed language for.
    """
    x = (observations - observations.mean(0)) / (observations.std(0) + 1e-6)
    n_classes = int(goals.max()) + 1
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(len(x), generator=generator)
    cut = int(0.7 * len(x))
    train_idx, test_idx = order[:cut], order[cut:]

    scores = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden),
            torch.nn.ELU(),
            torch.nn.Linear(hidden, n_classes),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(400):
            opt.zero_grad()
            F.cross_entropy(net(x[train_idx]), goals[train_idx]).backward()
            opt.step()
        with torch.no_grad():
            scores.append(
                float((net(x[test_idx]).argmax(-1) == goals[test_idx]).float().mean())
            )
    return sum(scores) / len(scores)
