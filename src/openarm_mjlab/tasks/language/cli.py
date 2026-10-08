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

"""Command-line entry points for the language-conditioned task.

Two commands, because there are two questions and they cost very different
amounts to answer:

openarm-mjlab-lang-check
    The offline checks. Seconds, CPU, no policy and no simulator. Run this
    BEFORE training: it says whether the instructions are separable by the
    readout a policy would have, and whether the observation gives the goal
    away anyway. Either failing means training cannot demonstrate instruction
    following no matter how long it runs.

openarm-mjlab-lang-eval
    The behavioural measurement on a trained checkpoint. Scores each goal
    twice -- once shown its own instruction, once shown the other's -- and
    reports execution and suppression separately, because they fail for
    opposite reasons.

openarm-mjlab-lang-play
    Watch it. Opens the viewer, alternates the instruction, and prints each
    sentence as it changes, so the claim can be checked by eye rather than
    taken from a table.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

TASK_ID = "OpenArm-Puck-Language"

# Typed into the terminal, NOT pressed in the viewer window. MuJoCo binds
# every letter A-Z to a visualization toggle and every digit to a geom group,
# and mjlab binds more on top, so a viewer keybinding here would silently do
# two things at once -- pressing "h" would draw convex hulls, "n" would
# recolour the limbs by constraint island, "2" would hide the robot's visual
# meshes entirely. All of those look like bugs. stdin has no such conflict.
PLAY_HELP = """
  type into this terminal and press Enter:
    l = push left    t = train phrasings     a = toggle auto-switching
    r = push right   h = held-out phrasings  q = quit
    n = same goal, new phrasing
"""


def _build_env(num_envs: int, table_path: Path | None, device: str, seed: int):
    from mjlab.envs import ManagerBasedRlEnv

    from openarm_mjlab.tasks.language.env_cfg import openarm_language_puck_env_cfg

    cfg = openarm_language_puck_env_cfg(table_path=table_path)
    cfg.scene.num_envs = num_envs
    cfg.observations["actor"].enable_corruption = False
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.instruction_seed = seed
    return cfg, env


def check() -> None:
    """Run the offline pre-training checks and print what they say."""
    import torch

    from openarm_mjlab.tasks.language import diagnostics, instructions

    parser = argparse.ArgumentParser(
        prog="openarm-mjlab-lang-check",
        description="Offline checks to run before training. No GPU needed.",
    )
    parser.add_argument("--table", default=None, help="instruction embedding table")
    # 64 x 6 = 384 samples is too few to answer this question: the standard
    # error on a proportion near 0.5 is then ~0.026, and a measured run gave
    # 0.598 -- which cleared the 0.10 margin below by 0.002 while printing
    # "at chance". The same environment measures 0.503 at 3072 samples. The
    # default is therefore sized so the estimate is stable, not so it is fast.
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--resets", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip-observation",
        action="store_true",
        help="skip the observation check, which needs to build the simulator",
    )
    args = parser.parse_args()
    table = Path(args.table) if args.table else None

    chance = 1.0 / len(instructions.GOALS)
    scores = diagnostics.encoder_separability(table_path=table)
    print("1. can a probe decode the instruction, using the readout a policy has?")
    for goal in instructions.GOALS:
        print(f"     {goal:8s} {scores[goal]:.3f}")
    print(f"     overall  {scores['overall']:.3f}   (chance {chance:.3f})")
    if scores["overall"] < 0.75:
        print("     -> too close to chance: a policy cannot recover what the")
        print("        encoder has already lost. Try a different encoder.")

    if args.skip_observation:
        return

    print("\n2. can the goal be decoded WITHOUT the instruction?")
    _, env = _build_env(args.num_envs, table, args.device, seed=0)
    try:
        from openarm_mjlab.tasks.language import embeddings
        from openarm_mjlab.tasks.language import mdp as language_mdp

        width = embeddings.dim(table)
        observations, goals = [], []
        for _ in range(args.resets):
            obs, _ = env.reset()
            observations.append(obs["actor"][:, :-width].detach().cpu())
            goals.append(language_mdp.goal_index(env).detach().cpu().clone())
        accuracy = diagnostics.observation_ambiguity(
            torch.cat(observations), torch.cat(goals)
        )
    finally:
        env.close()
    margin = accuracy - chance
    n = args.num_envs * args.resets
    stderr = (0.25 / n) ** 0.5
    print(
        f"     observation -> goal  {accuracy:.3f}   (chance {chance:.3f}, "
        f"{n} samples, standard error {stderr:.3f})"
    )
    if margin > 0.10:
        print("     -> the observation gives the goal away, so a policy that")
        print("        ignores language still scores full marks. Any")
        print("        instruction-following result from this task is vacuous.")
    elif margin > 3 * stderr:
        # Don't call this "at chance" -- it is above chance by more than
        # sampling noise explains, even though it clears the margin above.
        print(f"     -> {margin:+.3f} above chance, more than sampling noise")
        print("        explains. Under the 0.10 bar, but re-run with more")
        print("        samples before relying on it.")
    else:
        print("     -> at chance: language is the only route. Good.")


def evaluate() -> None:
    """Measure instruction following on a trained checkpoint."""
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg

    from openarm_mjlab.tasks.language import evaluation, instructions

    parser = argparse.ArgumentParser(
        prog="openarm-mjlab-lang-eval",
        description="Does the policy do what the instruction says?",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--table", default=None, help="instruction embedding table")
    parser.add_argument("--split", default="held_out", choices=["held_out", "train"])
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-sentence", action="store_true")
    args = parser.parse_args()
    table = Path(args.table) if args.table else None

    pool = instructions.TRAIN if args.split == "train" else instructions.HELD_OUT
    needed = max(len(p) for p in pool.values())
    num_envs = args.num_envs or needed
    if num_envs < needed:
        raise SystemExit(
            f"--num-envs {num_envs} cannot cover {needed} phrasings; "
            f"use at least {needed}"
        )

    cfg, env = _build_env(num_envs, table, args.device, args.seed)
    try:
        step_dt = cfg.decimation * cfg.sim.mujoco.timestep
        steps = int(cfg.episode_length_s / step_dt) + 5
        agent_cfg = load_rl_cfg(TASK_ID)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(
            args.checkpoint,
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        result = evaluation.measure(
            policy, env, steps, split=args.split, seed=args.seed
        )
    finally:
        env.close()

    print(f"checkpoint: {args.checkpoint}")
    print(f"phrasings:  {args.split}\n")
    header = f"{'':14s}" + "".join(f"shown {g:<10s}" for g in instructions.GOALS)
    print(header)
    for scored in instructions.GOALS:
        row = f"scored {scored:<7s}"
        for shown in instructions.GOALS:
            cell = result.cells[(scored, shown)]
            row += f"{sum(cell) / len(cell):<16.3f}"
        print(row)
    print()
    print(result.report())

    if args.per_sentence:
        print("\nper sentence: at the goal it names, vs at the other goal")
        for sentence, hit, miss in result.per_sentence:
            mark = "ok  " if hit > miss else ("tie " if hit == miss else "MISS")
            print(f"  [{mark}] {hit:.2f} vs {miss:.2f}  {sentence!r}")


def play() -> None:
    """Watch a trained policy follow instructions, and change them live."""
    import threading

    import torch
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.viewer import NativeMujocoViewer

    from openarm_mjlab.tasks.language import instructions

    parser = argparse.ArgumentParser(
        prog="openarm-mjlab-lang-play",
        description="Watch the policy follow instructions.",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--table", default=None, help="instruction embedding table")
    parser.add_argument("--split", default="held_out", choices=["held_out", "train"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--switch-every",
        type=int,
        default=300,
        help="policy steps between automatic goal switches; 0 disables",
    )
    args = parser.parse_args()

    _, env = _build_env(
        args.num_envs, Path(args.table) if args.table else None, args.device, args.seed
    )
    env.instruction_split = args.split
    env.instruction_shown = None  # show each goal its OWN sentence

    agent_cfg = load_rl_cfg(TASK_ID)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
        args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device
    )
    inner = runner.get_inference_policy(device=args.device)

    rng = random.Random(args.seed)
    lock = threading.Lock()
    state = {"pending": {}, "auto": args.switch_every > 0, "calls": 0}

    def apply(goal: str | None, split: str | None) -> None:
        if goal is not None:
            env.instruction_goal = goal
        if split is not None:
            env.instruction_split = split
        goal_now = env.instruction_goal
        split_now = env.instruction_split
        pool = (
            instructions.TRAIN[goal_now]
            if split_now == "train"
            else instructions.HELD_OUT[goal_now]
        )
        start = rng.randrange(len(pool))
        env.instruction_phrase_index = (
            torch.arange(args.num_envs, device=args.device) + start
        ) % len(pool)
        print(
            f'  {goal_now.upper():<5s} [{split_now:8s}] "{pool[start]}"',
            flush=True,
        )
        # Resample every env now rather than waiting for each to time out:
        # the point is to watch the arm change its mind.
        env.reset()

    def policy(obs):
        with lock:
            pending = dict(state["pending"])
            state["pending"].clear()
            auto = state["auto"]
        if pending:
            apply(pending.get("goal"), pending.get("split"))
            state["calls"] = 0
        elif auto:
            state["calls"] += 1
            if state["calls"] >= args.switch_every:
                state["calls"] = 0
                here = env.instruction_goal
                apply(next(g for g in instructions.GOALS if g != here), None)
        return inner(obs)

    def reader() -> None:
        for line in sys.stdin:
            c = line.strip().lower()[:1]
            with lock:
                if c == "l":
                    state["pending"]["goal"] = "left"
                elif c == "r":
                    state["pending"]["goal"] = "right"
                elif c == "n":
                    state["pending"]["goal"] = env.instruction_goal
                elif c == "t":
                    state["pending"]["split"] = "train"
                elif c == "h":
                    state["pending"]["split"] = "held_out"
                elif c == "a":
                    state["auto"] = not state["auto"]
                    print(f"  [auto {'on' if state['auto'] else 'off'}]", flush=True)
                elif c == "q":
                    os._exit(0)

    threading.Thread(target=reader, daemon=True).start()
    apply(instructions.GOALS[0], args.split)
    print(PLAY_HELP, flush=True)
    try:
        NativeMujocoViewer(env, policy).run()
    finally:
        env.close()
