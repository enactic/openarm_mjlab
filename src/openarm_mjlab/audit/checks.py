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

"""Checks that score a task's success against physics its success check doesn't read.

All contact and depth numbers come from MuJoCo's own collision pass (``d.contact``),
run on a CPU ``MjData`` rebuilt from each env's ``qpos``. ``mj_geomDistance`` is
avoided because it returns spurious exact zeros on some geom pairs.
"""

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np
import torch

from openarm_mjlab.audit.specs import TaskIntegritySpec

PENETRATION_TOL = 0.001  # m. Deeper than this counts as being inside a geom.
# Penetration depth is REPORTED, never counted as a violation: honest load-bearing
# grips in this simulator sink millimetres into what they hold (OpenArm-Lift, whose
# successes are verified load-bearing two-pad grips: median 3.7 mm per grip sample,
# per-episode max median 5.8 mm, peaks to 22 mm), so no fixed depth separates an
# honest grip from a dishonest one.
# Penetration and peak object speed count only from this episode step on. A spawn
# overlap clears within 1-2 steps without the policy's help (measured on the door and
# drawer), and can kick the object while it does: with zero action the door, whose
# start pose overlaps the panel, swings faster than its limit in step 1 in most
# episodes. The policy does not choose its spawn; spawn_clearance reports that
# separately.
SETTLE_STEPS = 3
INNER_PADS = "finger_inner_right_collision"
OUTER_PADS = "finger_outer_right_collision"


class _Geometry:
    """Name lookups and a CPU MjData for one env's model."""

    def __init__(self, env, spec: TaskIntegritySpec):
        self.env = env
        self.m = env.sim.mj_model
        self.d = mujoco.MjData(self.m)
        self.names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for g in range(self.m.ngeom)
        ]
        self.robot = {g for g, n in enumerate(self.names) if n.startswith("robot/")}
        self.inner = {g for g, n in enumerate(self.names) if INNER_PADS in n}
        self.outer = {g for g, n in enumerate(self.names) if OUTER_PADS in n}
        self.target = {g for g, n in enumerate(self.names) if n in spec.target_geoms}
        self.spec_objects = {
            g for g, n in enumerate(self.names) if n in spec.object_geoms
        }
        missing = set(spec.target_geoms + spec.object_geoms) - set(self.names)
        if missing:
            raise ValueError(
                f"{spec.task_id}: spec names geoms the model lacks: {sorted(missing)}"
            )
        self.objects = {
            g
            for g, n in enumerate(self.names)
            if n and not n.startswith("robot/") and n != "terrain"
        }
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, spec.object_joint)
        if j < 0:
            raise ValueError(f"{spec.task_id}: no joint {spec.object_joint}")
        self.obj_dof = int(self.m.jnt_dofadr[j])
        self.obj_free = int(self.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)

    def load(self, qpos: np.ndarray) -> None:
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def robot_object_contacts(self):
        """(robot geom, object geom, signed distance) for every robot-object contact."""
        out = []
        for c in self.d.contact[: self.d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if g2 in self.robot and g1 in self.objects:
                g1, g2 = g2, g1
            if g1 in self.robot and g2 in self.objects:
                out.append((g1, g2, float(c.dist)))
        return out


def object_speed(env, geo: _Geometry) -> np.ndarray:
    """Per-env speed of the manipulated object's joint (linear speed for a free joint)."""
    qvel = env.sim.data.qvel.detach().cpu().numpy()
    if geo.obj_free:
        return np.linalg.norm(qvel[:, geo.obj_dof : geo.obj_dof + 3], axis=1)
    return np.abs(qvel[:, geo.obj_dof])


@dataclass
class SpawnReport:
    """Whether resets start with the robot inside an object."""

    resets: int
    penetrating_fraction: float
    deepest_mm: float
    pairs: dict[str, int] = field(
        default_factory=dict
    )  # pair -> resets it penetrates in


def spawn_clearance(env, spec: TaskIntegritySpec, rounds: int = 4) -> SpawnReport:
    """Reset every env ``rounds`` times and look for robot geoms inside object geoms."""
    geo = _Geometry(env, spec)
    n = bad = 0
    deepest = 0.0
    pairs: dict[str, int] = {}
    for _ in range(rounds):
        env.reset()
        for q in env.sim.data.qpos.detach().cpu().numpy():
            geo.load(q)
            n += 1
            hits = [c for c in geo.robot_object_contacts() if c[2] < -PENETRATION_TOL]
            bad += bool(hits)
            for key in {f"{geo.names[g1]} -> {geo.names[g2]}" for g1, g2, _ in hits}:
                pairs[key] = (
                    pairs.get(key, 0) + 1
                )  # resets in which this pair penetrates
            deepest = min([deepest] + [dist for _, _, dist in hits])
    return SpawnReport(n, bad / max(n, 1), 1000.0 * deepest, pairs)


@dataclass
class GateReport:
    """Whether the task's contact sensors are true before the policy does anything."""

    sensors: dict[str, float]


def gate_vacuity(env) -> GateReport:
    """Fraction of envs in which each contact sensor fires after one zero-action step."""
    env.reset()
    env.step(
        torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        )
    )
    out = {}
    for name, sensor in env.scene.sensors.items():
        found = getattr(getattr(sensor, "data", None), "found", None)
        if found is not None:
            out[name] = float(
                (found.view(env.num_envs, -1) > 0).any(dim=1).float().mean()
            )
    return GateReport(out)


@dataclass
class ClosureReport:
    """Whether closing the jaws from the spawn puts both pads on the target first."""

    spawns: int
    both_pads_on_target_first: float
    first_contact: dict[str, int] = field(default_factory=dict)


def closure_from_spawn(
    env, spec: TaskIntegritySpec, closed: float = 0.0, steps: int = 101
) -> ClosureReport:
    """Hold the arm at each spawn and close the right jaw kinematically.

    Contacts present before closing starts are ignored (spawn_clearance reports them),
    so this asks only what closing adds.
    """
    geo = _Geometry(env, spec)
    adr = [
        int(
            geo.m.jnt_qposadr[
                mujoco.mj_name2id(
                    geo.m,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"robot/openarm_right_finger_joint{k}",
                )
            ]
        )
        for k in (1, 2)
    ]
    env.reset()
    ok = 0
    firsts: dict[str, int] = {}
    Q = env.sim.data.qpos.detach().cpu().numpy()
    for q in Q:
        geo.load(q)
        start = {(g1, g2) for g1, g2, _ in geo.robot_object_contacts()}
        verdict = "never reaches anything"
        for f in np.linspace(q[adr[0]], closed, steps):
            qq = q.copy()
            qq[adr] = f
            geo.load(qq)
            new = [
                (g1, g2)
                for g1, g2, _ in geo.robot_object_contacts()
                if (g1, g2) not in start
            ]
            if not new:
                continue
            on_target = {g1 for g1, g2 in new if g2 in geo.target}
            others = sorted({geo.names[g2] for g1, g2 in new if g2 not in geo.target})
            if others:
                verdict = "first hits " + ", ".join(others)
                break
            if on_target & geo.inner and on_target & geo.outer:
                verdict = "both pads on target"
                ok += 1
                break
        firsts[verdict] = firsts.get(verdict, 0) + 1
    return ClosureReport(len(Q), ok / max(len(Q), 1), firsts)


@dataclass
class ProbeReport:
    """Whether the success check fires for policies that are not trying."""

    probe: str
    episodes: int
    success_rate: float


def probe(
    env, spec: TaskIntegritySpec, kind: str, steps: int, seed: int = 0
) -> ProbeReport:
    """Run a trivial policy ('zero' or 'random' actions) and count successes."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    env.reset()
    dim = env.action_manager.total_action_dim
    episodes = successes = 0
    for _ in range(steps):
        if kind == "zero":
            a = torch.zeros(env.num_envs, dim)
        elif kind == "random":
            a = torch.rand(env.num_envs, dim, generator=gen) * 2.0 - 1.0
        else:
            raise ValueError(kind)
        _, _, terminated, truncated, _ = env.step(a.to(env.device))
        done = (terminated | truncated).cpu()
        episodes += int(done.sum())
        successes += int(
            env.termination_manager.get_term(spec.success_term).cpu().sum()
        )
    return ProbeReport(kind, episodes, successes / max(episodes, 1))


@dataclass
class PanelReport:
    """A policy's successes, scored by checks its success check doesn't make."""

    episodes: int
    successes: int
    success_rate: float  # among episodes that finished within the rollout
    rejected_fraction_of_successes: float
    rejected_excluding_spawn: float  # the same, counting only what the policy does
    violations: dict[str, float]
    two_pads_on_target_at_success: float | None
    any_target_contact_at_success: float
    peak_speed_median: float
    speed_limit: float | None
    episode_max_depth_mm_median: float  # reported only, see module note
    episode_max_depth_mm_p90: float


def rollout_panel(
    env, wrapped, policy, spec: TaskIntegritySpec, steps: int
) -> PanelReport:
    """Roll a policy out and judge every success against the spec.

    ``env`` is the ManagerBasedRlEnv and ``wrapped`` the RslRlVecEnvWrapper around it
    that the policy reads observations from. The state at a success is read BEFORE
    the step that terminated the episode, because mjlab resets a terminated env
    inside the same step.
    """
    geo = _Geometry(env, spec)
    n = env.num_envs
    peak = np.zeros(n)
    deepest = np.zeros(n)
    spawn_bad = np.zeros(n, dtype=bool)

    def spawn_check(ids):
        Q = env.sim.data.qpos.detach().cpu().numpy()
        for e in ids:
            geo.load(Q[e])
            spawn_bad[e] = any(
                dist < -PENETRATION_TOL for _, _, dist in geo.robot_object_contacts()
            )

    obs, _ = wrapped.reset()
    spawn_check(range(n))
    age = np.zeros(n, dtype=int)
    episodes = 0
    rows = []
    for _ in range(steps):
        before = env.sim.data.qpos.detach().cpu().numpy().copy()
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        done = dones.cpu().numpy().astype(bool)
        won = (
            env.termination_manager.get_term(spec.success_term)
            .cpu()
            .numpy()
            .astype(bool)
        )
        speed = object_speed(env, geo)
        age += 1
        settled = ~done & (age >= SETTLE_STEPS)
        peak[settled] = np.maximum(peak[settled], speed[settled])
        Q = env.sim.data.qpos.detach().cpu().numpy()
        for e in np.where(settled)[0]:
            geo.load(Q[e])
            for _, g2, dist in geo.robot_object_contacts():
                if g2 in geo.spec_objects:
                    deepest[e] = min(deepest[e], dist)
        for e in np.where(done)[0]:
            episodes += 1
            if won[e]:
                geo.load(before[e])
                contacts = geo.robot_object_contacts()
                on_t = {g1 for g1, g2, _ in contacts if g2 in geo.target}
                rows.append(
                    dict(
                        two_pads=bool(on_t & geo.inner and on_t & geo.outer),
                        any_target=bool(on_t),
                        depth_mm=float(-1000.0 * deepest[e]),
                        overspeed=bool(
                            spec.speed_limit is not None and peak[e] > spec.speed_limit
                        ),
                        spawn_inside=bool(spawn_bad[e]),
                        peak=float(peak[e]),
                    )
                )
            peak[e] = 0.0
            deepest[e] = 0.0
            age[e] = 0
        if done.any():
            spawn_check(np.where(done)[0])
    succ = len(rows)

    def frac(key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    violations = {
        "episode_started_inside_an_object": frac("spawn_inside"),
    }
    if spec.contact_required_at_success:
        violations["no_target_contact_at_success"] = 1.0 - frac("any_target")
    if spec.speed_limit is not None:
        key = (
            "peak_speed_over_declared_limit"
            if spec.speed_limit_is_rule
            else "peak_speed_over_reward_cap_(not_counted)"
        )
        violations[key] = frac("overspeed")

    def rejected_row(r):
        return (
            r["spawn_inside"]
            or (spec.speed_limit_is_rule and r["overspeed"])
            or (spec.contact_required_at_success and not r["any_target"])
        )

    rejected = float(np.mean([rejected_row(r) for r in rows])) if rows else 0.0
    rejected_policy = (
        float(np.mean([rejected_row({**r, "spawn_inside": False}) for r in rows]))
        if rows
        else 0.0
    )
    return PanelReport(
        episodes=episodes,
        successes=succ,
        success_rate=succ / max(episodes, 1),
        rejected_fraction_of_successes=rejected,
        rejected_excluding_spawn=rejected_policy,
        violations=violations,
        two_pads_on_target_at_success=frac("two_pads") if spec.grasp else None,
        any_target_contact_at_success=frac("any_target"),
        peak_speed_median=float(np.median([r["peak"] for r in rows])) if rows else 0.0,
        speed_limit=spec.speed_limit,
        episode_max_depth_mm_median=float(np.median([r["depth_mm"] for r in rows]))
        if rows
        else 0.0,
        episode_max_depth_mm_p90=float(np.percentile([r["depth_mm"] for r in rows], 90))
        if rows
        else 0.0,
    )


def as_dict(report) -> dict:
    """Return a report as a plain dict."""
    return asdict(report)
