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

"""openarm-mjlab-audit: check whether a task's success can be reached without doing the task.

Static checks need no policy and run in seconds on CPU:
  spawn    do resets start with the robot inside an object?
  gates    are the task's contact sensors already true before the policy acts?
  closure  from the spawn, does closing the jaws put both pads on the target first?
  probes   does the success check fire for zero or random actions?
With --checkpoint, a trained policy's successes are also scored by a panel of checks
the policy never saw (penetration, start state, the task's declared speed limit),
and the fraction of successes the panel rejects is reported.
"""

import argparse
import json
from dataclasses import asdict

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from openarm_mjlab.audit import checks
from openarm_mjlab.audit.specs import SPECS


def _print(title, report):
    print(f"\n[{title}]")
    for k, v in asdict(report).items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in sorted(
                v.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0
            ):
                print(
                    f"    {kk}: {vv:.3f}"
                    if isinstance(vv, float)
                    else f"    {kk}: {vv}"
                )
        else:
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


def main() -> None:
    """Run the audit for one task and print the report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", choices=sorted(SPECS))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--probe-steps", type=int, default=None, help="default: one episode"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="score a trained policy with the panel"
    )
    parser.add_argument("--panel-steps", type=int, default=400)
    parser.add_argument("--json", default=None, help="also write the report here")
    args = parser.parse_args()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    spec = SPECS[args.task]
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs = args.num_envs
    cfg.observations["actor"].enable_corruption = False
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    steps = args.probe_steps or int(env.max_episode_length)

    out = {"task": args.task, "spec": asdict(spec)}
    out["spawn"] = checks.spawn_clearance(env, spec)
    out["gates"] = checks.gate_vacuity(env)
    if spec.grasp:
        out["closure"] = checks.closure_from_spawn(env, spec)
    out["probe_zero"] = checks.probe(env, spec, "zero", steps)
    out["probe_random"] = checks.probe(env, spec, "random", steps)
    if args.checkpoint:
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

        agent = load_rl_cfg(args.task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(
            wrapped, asdict(agent), device=args.device
        )
        runner.load(
            args.checkpoint,
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        out["panel"] = checks.rollout_panel(
            env, wrapped, policy, spec, args.panel_steps
        )

    print(f"=== {args.task} ===")
    for k, v in out.items():
        if k not in ("task", "spec"):
            _print(k, v)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                    for k, v in out.items()
                },
                f,
                indent=2,
            )
    env.close()


if __name__ == "__main__":
    main()
