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

"""Measure the success rate of a trained OpenArm policy.

Every task in this repository reports a success rate, and until now there
was no shared way to reproduce one: ``train`` and ``play`` exist, but
nothing answers "what fraction of episodes actually succeed". This does,
for any registered task, without task-specific code.

Four things make the number trustworthy, and all four are easy to get
wrong by hand:

* **Success is read from a named termination term**, not from
  ``done & !time_out``. Any task with a failure termination -- a dropped
  object, a puck knocked off the table -- counts those failures as
  successes under the coarser test.
* **An environment is scored once.** After it terminates it is excluded
  from further updates, so an env that succeeds, resets and succeeds again
  inside the same rollout cannot be counted twice.
* **The episode budget comes from the config**, as
  ``episode_length_s / (decimation * timestep)``. Hardcoding a step count
  silently truncates or over-runs whenever a task's episode length moves.
* **Observation corruption is disabled.** Training noise is a
  regularizer; leaving it on measures the policy plus the noise.

Usage::

    uv run openarm-mjlab-eval OpenArm-PickPlace
    uv run openarm-mjlab-eval OpenArm-Door --checkpoint path/to/model.pt
    uv run openarm-mjlab-eval OpenArm-PickPlace --success-term success
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from . import tasks  # noqa: F401  # Registers OpenArm tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import get_checkpoint_path

# Terms that describe how an episode ended badly or ran out of clock. They
# are never the success condition, so they are excluded when inferring one.
NON_SUCCESS_TERMS = frozenset(
    {
        "time_out",
        "nan_detection",
        "cube_dropped",
        "puck_fell",
        "bar_fell",
        "object_dropped",
    }
)


def resolve_success_term(active_terms: list[str], requested: str | None) -> str:
    """Return the termination term that counts as success.

    Prefers an explicit request, then a term literally named ``success``,
    then the single remaining term once clock and failure terms are set
    aside. Raises when the choice is genuinely ambiguous rather than
    guessing, because a wrong guess here silently changes the headline
    number.
    """
    if requested is not None:
        if requested not in active_terms:
            raise SystemExit(
                f"--success-term {requested!r} is not a termination term of this task.\n"
                f"Available: {', '.join(active_terms)}"
            )
        return requested
    if "success" in active_terms:
        return "success"
    candidates = [t for t in active_terms if t not in NON_SUCCESS_TERMS]
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(
        "Could not infer which termination term means success "
        f"(candidates: {', '.join(candidates) or 'none'}).\n"
        f"Pass --success-term explicitly. Available: {', '.join(active_terms)}"
    )


def _flatten(x: torch.Tensor) -> torch.Tensor:
    """Return a 1-D bool tensor from a (num_envs,) or (num_envs, 1) input."""
    x = x.to(torch.bool)
    return x.squeeze(-1) if x.dim() > 1 else x


def evaluate(
    task: str,
    checkpoint: Path | None = None,
    num_envs: int = 256,
    success_term: str | None = None,
    device: str = "cuda:0",
    seed: int = 0,
) -> dict[str, object]:
    """Roll out one episode per environment and return the measured rates."""
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # Training-time observation noise is a regularizer, not part of the task.
    if "actor" in cfg.observations:
        cfg.observations["actor"].enable_corruption = False

    step_dt = cfg.decimation * cfg.sim.mujoco.timestep
    episode_steps = int(cfg.episode_length_s / step_dt)

    agent_cfg = load_rl_cfg(task)
    if checkpoint is None:
        checkpoint = get_checkpoint_path(
            Path("logs/rsl_rl") / agent_cfg.experiment_name, checkpoint="model_.*.pt"
        )

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    terms = list(env.termination_manager.active_terms)
    chosen = resolve_success_term(terms, success_term)

    print(f"task            {task}")
    print(f"checkpoint      {checkpoint}")
    print(f"episodes        {num_envs}")
    print(
        f"episode budget  {episode_steps} steps "
        f"({cfg.episode_length_s}s at dt={step_dt})"
    )
    print(f"success term    {chosen}   (of: {', '.join(terms)})")

    fired = {t: torch.zeros(num_envs, dtype=torch.bool, device=device) for t in terms}
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
    length = torch.zeros(num_envs, dtype=torch.long, device=device)

    obs, _ = wrapped.reset()
    with torch.no_grad():
        for step in range(episode_steps):
            obs, _, dones, _ = wrapped.step(policy(obs))
            live = ~finished
            # Read each term BEFORE marking envs finished, so the step that
            # ends an episode is the step whose terms are attributed to it.
            for t in terms:
                fired[t] |= live & _flatten(env.termination_manager.get_term(t))
            done = _flatten(dones)
            length[live & done] = step + 1
            finished |= done
            if bool(finished.all()):
                break
    length[~finished] = episode_steps
    env.close()

    # Nested rather than flat: a task may have a termination term literally
    # named "success" (pick_place does), so a top-level "success" key would
    # collide with it and silently report the wrong number whenever
    # --success-term names a different term.
    rates = {t: float(fired[t].float().mean()) for t in terms}
    results = {"success_term": chosen, "success_rate": rates[chosen], "terms": rates}
    lengths = length.tolist()

    print("\n--- results " + "-" * 48)
    print(f"success rate ({chosen}): {rates[chosen]:.3f}")
    for t in terms:
        if t != chosen:
            print(f"  {t}: {rates[t]:.3f}")
    print(
        f"episode length steps: mean {statistics.mean(lengths):.1f}, "
        f"median {statistics.median(lengths):.1f}, budget {episode_steps}"
    )
    unfinished = int((~finished).sum())
    if unfinished:
        print(f"  {unfinished} episode(s) never terminated within the budget")
    return results


def main() -> None:
    """Parse arguments and run one evaluation."""
    p = argparse.ArgumentParser(
        prog="openarm-mjlab-eval", description=__doc__.split("\n")[0]
    )
    p.add_argument("task", help="registered task id, e.g. OpenArm-PickPlace")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="policy to load (default: latest run for this task)",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="episodes to evaluate, one per environment (default: 256)",
    )
    p.add_argument(
        "--success-term",
        default=None,
        help="termination term counted as success (default: inferred)",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    evaluate(a.task, a.checkpoint, a.num_envs, a.success_term, a.device, a.seed)


if __name__ == "__main__":
    main()
