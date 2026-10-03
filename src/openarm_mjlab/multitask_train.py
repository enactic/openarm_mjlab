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

"""Train one policy on all five MultiTask environments concurrently.

Unlike ``openarm-mjlab-train``, which trains a single registered task, this
builds a mixture environment holding every task at once so each PPO batch
spans all five. See ``openarm_mjlab.tasks.multitask`` for why.
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import asdict
from pathlib import Path

import torch

import openarm_mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg
from openarm_mjlab.tasks.multitask.mixture_env import (
    MULTITASK_IDS,
    MultiTaskMixtureVecEnv,
)


def train(
    num_envs: int,
    iterations: int,
    experiment: str | None,
    seed: int,
    weights: tuple[float, ...] | None,
    device: str,
    logger: str | None = None,
) -> Path:
    """Train the shared policy and return the run's log directory."""
    torch.manual_seed(seed)
    env = MultiTaskMixtureVecEnv(
        num_envs_total=num_envs, device=device, weights=weights
    )
    allocation = {task.split("-")[-1]: n for task, n in zip(MULTITASK_IDS, env.counts)}
    print(f"per-task environments: {allocation}", flush=True)

    agent_cfg = load_rl_cfg(MULTITASK_IDS[0])
    agent_cfg.max_iterations = iterations
    agent_cfg.seed = seed
    if experiment:
        agent_cfg.experiment_name = experiment
    if logger:
        agent_cfg.logger = logger

    log_dir = (
        Path("logs/rsl_rl")
        / agent_cfg.experiment_name
        / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logging to {log_dir}", flush=True)

    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device=device)
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.close()
    return log_dir


def main() -> None:
    """Parse arguments and run one concurrent multi-task training job."""
    parser = argparse.ArgumentParser(
        prog="openarm-mjlab-multitask-train",
        description=__doc__.split("\n")[0],
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1000,
        help="total environments, split across the five tasks (default: 1000)",
    )
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument(
        "--experiment",
        default=None,
        help="experiment name (default: the one in the task's rl_cfg)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "comma-separated allocation weights in task order "
            f"({', '.join(t.split('-')[-1] for t in MULTITASK_IDS)}); "
            "default is an even split"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--logger",
        choices=("wandb", "tensorboard"),
        default=None,
        help="override the run's logger (default: whatever the rl_cfg sets)",
    )
    args = parser.parse_args()

    weights = tuple(float(w) for w in args.weights.split(",")) if args.weights else None
    train(
        args.num_envs,
        args.iterations,
        args.experiment,
        args.seed,
        weights,
        args.device,
        args.logger,
    )


if __name__ == "__main__":
    main()
