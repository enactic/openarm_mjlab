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

"""Door family: the hinged door with parameterised geometry.

Structurally different from the valve family -- a hinged panel with a
stand-off handle, rather than a lever rotating on its own axis -- which is
what makes it a test of the GENERATOR rather than a second valve.

Varied per instance:

* ``handle_span`` -- distance from the hinge to the handle along the panel.
  The substantive axis, matching ``lever_reach``'s role in the valve family:
  it moves where the hand must go AND sets the moment arm about the hinge.
  The panel is resized with it so the handle stays at the panel's outer edge.
* ``standoff`` -- how far the handle stands proud of the panel face. Changes
  how much room the gripper has to close around it.
* ``x``, ``y``, ``height`` -- where the fixture sits.
* ``damping`` -- hinge damping; deliberately unobservable, a robustness axis.

NOMINAL reproduces the MERGED UPSTREAM door geometry, not this repo's copy.
This repo still has the pre-review handle: a cylinder of radius 7mm centred in
a panel of half-thickness 6mm, so it protruded 1mm from each face and a
parallel gripper could not close around it. Building a family on that would
inherit the defect.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco

from mjlab.entity import EntityCfg
from openarm_mjlab.tasks.door.door_env_cfg import openarm_door_env_cfg

_PANEL_HALF_THICKNESS = 0.006
_PANEL_INSET = 0.015  # gap between hinge and panel's inner edge
_PANEL_OVERHANG = -0.005  # panel ends 5mm PAST the handle


@dataclass(frozen=True)
class DoorParams:
    """One instance of the door family."""

    handle_span: float = 0.13
    standoff: float = 0.030
    x: float = 0.29
    y: float = -0.10
    height: float = 0.40
    damping: float = 0.2

    def name(self) -> str:
        """Return a short stable id, used in the registered task id."""
        return (
            f"s{round(self.handle_span * 1000):03d}"
            f"o{round(self.standoff * 1000):02d}"
            f"x{round(self.x * 1000):03d}"
            f"y{round(self.y * 1000):+04d}"
            f"h{round(self.height * 1000):03d}"
            f"d{round(self.damping * 100):03d}"
        )


NOMINAL = DoorParams()


def make_door_spec(params: DoorParams):
    """Return a zero-arg spec builder for one instance's geometry."""

    def _spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.modelname = "door_fixture"
        spec.compiler.degree = False

        base = spec.worldbody.add_body(name="door_base")
        base.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.41, 0.55, 0.04),
            pos=(0.47 - params.x, -params.y, 0.36 - params.height),
            rgba=(0.82, 0.71, 0.55, 1.0),
            friction=(1.0, 0.005, 0.0001),
        )
        base.add_geom(
            name="door_post",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.007, 0.007, 0.06),
            pos=(0, -0.07, 0.06),
            rgba=(0.4, 0.4, 0.45, 1.0),
            friction=(1.0, 0.01, 0.01),
        )

        door = base.add_body(name="door", pos=(0, -0.07, 0.06))
        door.add_joint(
            name="door_hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=(0, 0, 1),
            range=(0.0, 1.4),
            damping=params.damping,
            frictionloss=0.02,
        )
        # The panel runs from the hinge out to just past the handle, so widening
        # the span widens the door rather than leaving the handle floating past
        # its edge. Upstream's panel starts 15mm from the hinge and ends 5mm past
        # the handle; keeping both offsets reproduces it exactly at the nominal
        # 130mm span (half 0.060, centre 0.075).
        panel_half = (params.handle_span - _PANEL_INSET - _PANEL_OVERHANG) / 2.0
        door.add_geom(
            name="door_panel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(_PANEL_HALF_THICKNESS, panel_half, 0.05),
            pos=(0, _PANEL_INSET + panel_half, 0),
            mass=0.08,
            rgba=(0.55, 0.45, 0.3, 1.0),
        )
        handle_y = params.handle_span
        handle_x = -(_PANEL_HALF_THICKNESS + params.standoff)
        door.add_geom(
            name="door_handle",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=(handle_x, handle_y, -0.025, handle_x, handle_y, 0.025),
            size=(0.007, 0, 0),
            mass=0.02,
            rgba=(0.2, 0.2, 0.22, 1.0),
            priority=2,
            condim=3,
            friction=(1.0, 0.01, 0.01),
            solref=(0.005, 1.0),
        )
        for z in (-0.02, 0.02):
            door.add_geom(
                name=f"door_handle_post_{'hi' if z > 0 else 'lo'}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=(-_PANEL_HALF_THICKNESS, handle_y, z, handle_x, handle_y, z),
                size=(0.004, 0, 0),
                mass=0.005,
                rgba=(0.2, 0.2, 0.22, 1.0),
            )
        door.add_site(
            name="handle_site", pos=(handle_x, handle_y, 0), size=(0.005, 0, 0)
        )
        return spec

    return _spec


def door_family_env_cfg(params: DoorParams, play: bool = False):
    """Return the door task rebuilt with one instance's geometry."""
    cfg = openarm_door_env_cfg(play=play)
    door_init = EntityCfg.InitialStateCfg(
        pos=(params.x, params.y, params.height),
        joint_pos={"door_hinge": 0.0},
        joint_vel={"door_hinge": 0.0},
    )
    cfg.scene.entities["door"] = EntityCfg(
        spec_fn=make_door_spec(params), init_state=door_init
    )
    return cfg


def perturbed(base: DoorParams, **kwargs) -> DoorParams:
    """Return a copy of ``base`` with the named fields replaced."""
    return replace(base, **kwargs)
