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

"""Valve family: the lever valve with parameterised geometry.

Varied per instance:

* ``lever_reach`` -- distance from the hinge axis to the grip. This is the
  substantive one. It moves where the policy must put its hand AND changes
  the moment arm, so a short lever needs more force for the same torque. It
  is observable through ``ee_to_grip``, which is what makes generalising to
  an unseen reach possible at all.
* ``x``, ``y``, ``height`` -- where the fixture sits.
* ``damping`` -- hinge damping. Deliberately NOT observable; this is a
  robustness axis, like domain randomisation, not something to infer.

NOMINAL reproduces the hand-written task's GEOMETRY exactly -- every geom size
and position is bit-identical, verified by diag_family_nominal.py -- with one
deliberate difference: the grip carries the contact-priority fix (priority=2),
without which the finger geoms outrank it and MuJoCo never reads its friction,
so the task's own dr_grip_friction would randomise a value with no effect. The
merged upstream valve has that fix; this repo's copy predates it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco

from mjlab.entity import EntityCfg
from openarm_mjlab.tasks.valve.valve_env_cfg import openarm_valve_env_cfg

# The hand-written task puts the lever box half-length and centre at 0.035
# with the grip at 0.062, i.e. both at 0.5645 * reach. Keeping that ratio
# means NOMINAL rebuilds the original geometry bit for bit.
_LEVER_RATIO = 0.035 / 0.062


@dataclass(frozen=True)
class ValveParams:
    """One instance of the valve family."""

    lever_reach: float = 0.062
    x: float = 0.25
    y: float = 0.0
    height: float = 0.40
    damping: float = 0.5

    def name(self) -> str:
        """Return a short stable id, used in the registered task id."""
        return (
            f"r{round(self.lever_reach * 1000):03d}"
            f"x{round(self.x * 1000):03d}"
            f"y{round(self.y * 1000):+04d}"
            f"h{round(self.height * 1000):03d}"
            f"d{round(self.damping * 100):03d}"
        )


NOMINAL = ValveParams()


def make_valve_spec(params: ValveParams):
    """Return a zero-arg spec builder for one instance's geometry."""

    def _spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.modelname = "valve_fixture"
        # Programmatic MjSpec defaults to DEGREES; the hinge range is radians.
        spec.compiler.degree = False

        base = spec.worldbody.add_body(name="valve_base")
        base.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.41, 0.55, 0.04),
            pos=(0.47 - params.x, -params.y, 0.36 - params.height),
            rgba=(0.82, 0.71, 0.55, 1.0),
            friction=(1.0, 0.005, 0.0001),
        )
        base.add_geom(
            name="valve_pipe",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(0.015, 0.0275, 0),
            pos=(0, 0, 0.0275),
            rgba=(0.4, 0.4, 0.45, 1.0),
            friction=(1.0, 0.01, 0.01),
        )

        valve = base.add_body(name="valve", pos=(0, 0, 0.065))
        valve.add_joint(
            name="valve_turn",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=(0, 0, 1),
            range=(-3.0, 3.0),
            damping=params.damping,
            frictionloss=0.05,
        )
        valve.add_geom(
            name="valve_hub",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=(0, 0, -0.006, 0, 0, 0.006),
            size=(0.014, 0, 0),
            mass=0.05,
            rgba=(0.3, 0.3, 0.35, 1.0),
        )
        lever = _LEVER_RATIO * params.lever_reach
        valve.add_geom(
            name="valve_lever",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(lever, 0.008, 0.006),
            pos=(lever, 0, 0),
            mass=0.05,
            rgba=(0.7, 0.2, 0.2, 1.0),
        )
        valve.add_geom(
            name="valve_grip",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=(params.lever_reach, 0, -0.02, params.lever_reach, 0, 0.02),
            size=(0.007, 0, 0),
            mass=0.02,
            rgba=(0.2, 0.2, 0.22, 1.0),
            # Same contact-priority fix the hand-written tasks needed: the finger
            # geoms are priority=1, so a default-priority grip is outranked and its
            # friction is never read.
            priority=2,
            condim=3,
            friction=(1.0, 0.01, 0.01),
            solref=(0.005, 1.0),
        )
        valve.add_site(
            name="grip_site", pos=(params.lever_reach, 0, 0), size=(0.005, 0, 0)
        )
        return spec

    return _spec


def valve_family_env_cfg(params: ValveParams, play: bool = False):
    """Return the valve task rebuilt with one instance's geometry.

    Everything except the scene -- rewards, terminations, observations, events,
    actions -- is taken unchanged from the hand-written task, so an instance
    differs from it only in geometry.
    """
    cfg = openarm_valve_env_cfg(play=play)
    valve_init = EntityCfg.InitialStateCfg(
        pos=(params.x, params.y, params.height),
        joint_pos={"valve_turn": 0.0},
        joint_vel={"valve_turn": 0.0},
    )
    cfg.scene.entities["valve"] = EntityCfg(
        spec_fn=make_valve_spec(params), init_state=valve_init
    )
    return cfg


def perturbed(base: ValveParams, **kwargs) -> ValveParams:
    """Return a copy of ``base`` with the named fields replaced."""
    return replace(base, **kwargs)


__all__ = [
    "NOMINAL",
    "ValveParams",
    "make_valve_spec",
    "perturbed",
    "valve_family_env_cfg",
]
