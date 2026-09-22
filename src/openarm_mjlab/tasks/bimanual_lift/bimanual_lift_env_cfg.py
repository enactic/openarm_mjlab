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

"""Environment configuration for the OpenArm bimanual lift task.

Bimanual lift task: BOTH arms grip their own end of a 370 mm bar and

raise it together.

Every prior task in this suite (reach/valve/puck/move_puck/door/drawer/
lift) only ever actively controls the RIGHT arm; the left arm is parked
via HoldDefaultPositionActionCfg. Genuine bimanual coordination is the one
untapped capability of this platform (the classical sibling project has a
full bimanual controller, openarm_control/bimanual.py; the RL side never
touched it). This task requires both arms because the bar is deliberately
too long for either ~30 mm gripper cage to lift alone without the far,
ungripped end simply tipping/dragging: unless both ends are genuinely
gripped and raised in sync, a single-arm attempt rotates the bar about its
one grip point rather than lifting it.

SCENE GEOMETRY -- avoiding a real, documented collision mode:

The two arms mount only 62 mm apart at the base (openarm_v20_bimanual.xml:
174/259, `openarm_left_base_link pos="0 0.031 0"` / `openarm_right_base_
link pos="0 -0.031 0"`), and the classical sibling project's README
(Limitations & scope) documents a real failure mode already hit and
designed around there: "Two close-mounted 7-DOF arms collide when *both*
reach over one centred object (their upper arms cross)" -- which is why
its own bottle-unscrew and cloth-fold tasks are single-arm, while its
working simultaneous-bimanual demos (ParallelSort, BimanualStack) keep
both arms' targets well off the centerline.

This task's two grip points are placed at that SAME proven-safe
separation, not merely "some wide number": bar end offset
END_Y_OFFSET=0.16 m (see mdp.py) is copied directly from
openarm_control/bimanual.py's ParallelSort, whose right_jobs pick at
(0.18, -0.16) SIMULTANEOUSLY with left_jobs at (0.18, +0.16) -- an
already-working, non-colliding, both-arms-at-once configuration in the
same robot model. It is independently corroborated from the mjlab side
too: reach/mdp.py's own validated right-arm workspace box has y in
[-0.35, -0.05], and -0.16 sits comfortably mid-box, nowhere near either
the centerline (y=0, the risky zone) or the box edge. Both arms' resting
default pose (HOME_KEYFRAME, elbow bent up at 1.5708 rad, everything else
at 0 -- unchanged here, see get_bimanual_lift_robot_cfg below) also never
requires either arm's IK solution to sweep across the other's side, since
each end sits within that arm's OWN existing validated workspace box, not
centered between the two.

Total bar length (outer face to outer face) works out to END_Y_OFFSET*2 +
2*block_half_width = 0.32 + 0.05 = 0.37 m -- a bit past the 250-350mm
originally sketched, a deliberate trade: given the documented collision
risk, erring toward MORE separation than the minimum proven-safe value
(0.16) rather than trimming back toward it purely to hit a round number.

OPEN UNCERTAINTY (see final report): this reasoning transfers the
CLASSICAL project's proven y-separation number into the MJLAB scene by
axis-and-magnitude analogy (same robot mesh, same base spacing, same
general x-depth-and-y-separation geometry), not by re-running IK in this
scene -- the two projects use different absolute coordinate origins (the
classical pick_xy values don't share mjlab's x convention, only the
y-axis separation logic was carried over). It is corroborated by mjlab's
OWN reach-task workspace box, which is a stronger same-scene signal than
the cross-project number alone. Still, no IK or physics check was run in
this scene (code-only task, no GPU) -- a human should watch specifically
for self-collision or upper-arm crossing during early exploration at the
GPU smoke test, per the instructions this task was built under.
"""

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.velocity import mdp as base_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg

from ...robot_bimanual import (
    BIMANUAL_ACTION_SCALE,
    EE_SITE_LEFT,
    EE_SITE_RIGHT,
    get_bimanual_robot_cfg,
    get_bimanual_spec,
)
from . import mdp as bl_mdp


# 2026-08-23 fix: the
# module docstring below in get_bimanual_lift_robot_cfg used to explain,
# honestly, why init_state stayed the plain HOME_KEYFRAME -- no verified
# two-arm IK pose existed yet. an earlier attempt later solved and verified
# one (bl_mdp.HELD_HIGH_RIGHT_Q/LEFT_Q, error 0.046mm), but only ever
# used it inside the reset_bar_held_high EVENT, never as the entity's
# actual default pose -- so JointPositionAction's use_default_offset
# (snapshotted ONCE at build time from entity.data.default_joint_pos,
# see mjlab/envs/mdp/actions/actions.py) stayed anchored to
# HOME_KEYFRAME regardless of what any reset event wrote into the sim. A
# direct zero-action probe confirmed this: a held-high reset gets yanked
# back toward HOME_KEYFRAME and the bar drops from 100mm to 30mm within
# 9 steps (~180ms) -- the same "zero action yanks the arm home" bug
# lift's own review #3 found and fixed via LIFT_HOME. This mirrors that
# exact fix: BIMANUAL_LIFT_HOME uses the already-solved, already-
# verified held-high joint values as the DEFAULT pose for both arms (and
# starts both fingers near their squeeze targets, matching LIFT_HOME's
# own "-0.25, already near-closed" convention), so zero action holds the
# arm right where a genuine grip-and-lift needs it, exactly like lift's
# LIFT_HOME == HELD_HIGH_POSE trick.
BIMANUAL_LIFT_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "openarm_right_joint1": bl_mdp.HELD_HIGH_RIGHT_Q[0],
        "openarm_right_joint2": bl_mdp.HELD_HIGH_RIGHT_Q[1],
        "openarm_right_joint3": bl_mdp.HELD_HIGH_RIGHT_Q[2],
        "openarm_right_joint4": bl_mdp.HELD_HIGH_RIGHT_Q[3],
        "openarm_right_joint5": bl_mdp.HELD_HIGH_RIGHT_Q[4],
        "openarm_right_joint6": bl_mdp.HELD_HIGH_RIGHT_Q[5],
        "openarm_right_joint7": bl_mdp.HELD_HIGH_RIGHT_Q[6],
        "openarm_right_finger_joint1": bl_mdp.RIGHT_FINGER_SQUEEZE,
        "openarm_left_joint1": bl_mdp.HELD_HIGH_LEFT_Q[0],
        "openarm_left_joint2": bl_mdp.HELD_HIGH_LEFT_Q[1],
        "openarm_left_joint3": bl_mdp.HELD_HIGH_LEFT_Q[2],
        "openarm_left_joint4": bl_mdp.HELD_HIGH_LEFT_Q[3],
        "openarm_left_joint5": bl_mdp.HELD_HIGH_LEFT_Q[4],
        "openarm_left_joint6": bl_mdp.HELD_HIGH_LEFT_Q[5],
        "openarm_left_joint7": bl_mdp.HELD_HIGH_LEFT_Q[6],
        "openarm_left_finger_joint1": bl_mdp.LEFT_FINGER_SQUEEZE,
    },
    joint_vel={".*": 0.0},
)


def get_bimanual_lift_spec() -> mujoco.MjSpec:
    """Build the robot spec with both fingers effort-controlled.

    Both fingers are raw-effort-controlled in this task (see the

    actuator cfg below), so BOTH need lift's get_lift_spec() joint-limit
    fix, not just the right one. the "Fourth attempt"
    (an earlier attempt) found MuJoCo's default solref_limit/solimp_limit lets a
    SUSTAINED raw-effort command drive a finger joint roughly double past
    its own hard stop -- a compliance level that's fine for every
    position-controlled joint (which self-limits via ctrlrange long before
    pressing hard against its stop) but not for direct torque control. That
    only ever mattered for the one raw-effort joint in the codebase before
    this task; now there are two, so both get the identical, probe-verified
    stiffening an earlier attempt landed on. Scoped to this task's spec only, same as
    lift's get_lift_spec.
    """
    spec = get_bimanual_spec()
    for j in spec.joints:
        if j.name in ("openarm_right_finger_joint1", "openarm_left_finger_joint1"):
            j.solref_limit = [0.004, 1.0]
            j.solimp_limit = [0.98, 0.999, 0.0002, 0.5, 2.0]
    return spec


def get_bimanual_lift_robot_cfg():
    """Entity config for the bimanual arms in this task."""
    cfg = get_bimanual_robot_cfg()
    # 2026-08-23: init_state now BIMANUAL_LIFT_HOME (see its docstring
    # above for the full root-cause diagnosis and fix rationale) -- was
    # the plain HOME_KEYFRAME through an earlier attempt, which is what
    # caused the held-high curriculum's own yank-back bug. Both arms'
    # "reach" is correspondingly easier now (hovering near the grip point,
    # not the far neutral pose), same trade lift's own LIFT_HOME already
    # made and shipped successfully -- reach_right/reach_left still have a
    # real, non-trivial gap to close (the bar sits below/at table height,
    # the default pose hovers HELD_HIGH_RAISE above that), so this is not
    # a "reach already solved for free" shortcut, just the same default-
    # offset anchor lift itself uses.
    cfg.init_state = BIMANUAL_LIFT_HOME
    cfg.spec_fn = get_bimanual_lift_spec
    actuators = (
        BuiltinPositionActuatorCfg(
            target_names_expr=("openarm_(left|right)_joint[12]",),
            stiffness=230.0,
            damping=2.7,
            effort_limit=40.0,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("openarm_(left|right)_joint[34]",),
            stiffness=190.0,
            damping=2.2,
            effort_limit=27.0,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("openarm_(left|right)_joint[567]",),
            stiffness=30.0,
            damping=1.5,
            effort_limit=7.0,
        ),
        # Right finger: effort_limit=7 matches ITS OWN XML actuatorfrcrange
        # ("-7 7", openarm_v20_bimanual.xml:323) -- the an earlier attempt fix. A
        # borrowed 30 Nm figure from a different actuator was harmless for
        # position control and directly became the max commandable torque
        # once switched to raw effort.
        IdealPdActuatorCfg(
            target_names_expr=("openarm_right_finger_joint1",),
            stiffness=0.0,
            damping=2.0,
            effort_limit=7.0,
        ),
        # Left finger's OWN XML actuatorfrcrange is "-10 10"
        # (openarm_v20_bimanual.xml:238) -- NOT symmetric with the right
        # finger's 7 Nm (a quirk of the upstream model, confirmed by reading
        # the XML directly, not assumed). Reading each joint's own designed
        # capacity rather than assuming left/right symmetry is exactly the
        # an earlier attempt lesson ("30 Nm was never this joint's real designed
        # capacity, just a borrowed number") applied to a joint that has
        # never been under effort control before this task.
        IdealPdActuatorCfg(
            target_names_expr=("openarm_left_finger_joint1",),
            stiffness=0.0,
            damping=2.0,
            effort_limit=10.0,
        ),
    )
    cfg.articulation = EntityArticulationInfoCfg(
        actuators=actuators, soft_joint_pos_limit_factor=0.9
    )
    return cfg


# Arm-only scale (fingers excluded, matching lift's LIFT_ARM_SCALE
# derivation exactly): both fingers are effort-controlled now, so the
# position action for either arm must not carry a finger scale key.
BIMANUAL_ARM_SCALE = {
    k: v for k, v in BIMANUAL_ACTION_SCALE.items() if "finger" not in k
}

# Right squeeze scale=2.0 is an earlier attempt/12's own probe-verified value (a
# 1-sigma exploration action lands at ~2 Nm against the right finger's 7
# Nm cap -- the "settles at the true joint limit" regime, not the
# saturating one). The left finger was ASSUMED to have a 10 Nm cap and
# scaled proportionally (2.0 * (10.0/7.0) ~= 2.857) -- this was flagged
# in this same comment as an unverified extrapolation needing a direct
# saturation-probe recheck, which was never actually done until the
# 2026-08-25 bimanual-lift asymmetry investigation ran it: a direct
# compiled-model check (`left_finger1_ctrl`/`right_finger1_ctrl`
# actuator forcerange) shows BOTH fingers are genuinely capped at 7 Nm,
# not 10 -- there is no 10 Nm actuator anywhere in the asset. The old
# scale therefore gave the left squeeze action ~43% more effective
# torque per unit of policy output than the right (2.857/7=0.408 of its
# real cap vs. right's proven 2.0/7=0.286), a real, mechanistic
# candidate for the left/right grip asymmetry seen in both trained seeds
# (left grips reliably, right barely does): larger effective torque per
# unit of exploration noise makes the left actuator far more likely to
# reach full-squeeze during early random exploration. Corrected to match
# right's proven value now that both arms are confirmed to share the
# same real 7 Nm cap.
RIGHT_SQUEEZE_SCALE = 2.0
LEFT_SQUEEZE_SCALE = 2.0

ROBOT_EE_RIGHT_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_RIGHT,))
ROBOT_EE_LEFT_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_LEFT,))
BAR_CFG = SceneEntityCfg("bar")

FINGER_BAR_SENSOR_RIGHT = ContactSensorCfg(
    name="finger_bar_contact_right",
    primary=ContactMatch(
        mode="body",
        pattern=r"openarm_right_ee_(inner|outer)_finger",
        entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="bar_end_right_geom", entity="bar"),
    # "force" added 2026-08-26 for the grasp-quality reward: the
    # binary "found" alone cannot distinguish a feather touch from a
    # secure clamp, which is the diagnosed root cause of the
    # never-lifts plateau (see the project notes).
    fields=("found", "force", "dist", "pos"),
    reduce="none",
    num_slots=1,
)
FINGER_BAR_SENSOR_LEFT = ContactSensorCfg(
    name="finger_bar_contact_left",
    primary=ContactMatch(
        mode="body",
        pattern=r"openarm_left_ee_(inner|outer)_finger",
        entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="bar_end_left_geom", entity="bar"),
    # "force" added 2026-08-26 for the grasp-quality reward: the
    # binary "found" alone cannot distinguish a feather touch from a
    # secure clamp, which is the diagnosed root cause of the
    # never-lifts plateau (see the project notes).
    fields=("found", "force", "dist", "pos"),
    reduce="none",
    num_slots=1,
)


def get_bimanual_table_spec() -> mujoco.MjSpec:
    """Build the static table slab the bar rests on."""
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_lift_table"
    spec.compiler.degree = False
    body = spec.worldbody.add_body(name="table")
    body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.41, 0.55, 0.04),
        pos=(0.47, 0.0, 0.36),
        rgba=(0.82, 0.71, 0.55, 1.0),
        friction=(1.0, 0.005, 0.0001),
    )
    return spec


def get_bar_spec() -> mujoco.MjSpec:
    """Build the two-ended bar the arms lift together.

    Bimanual bar: two lift-block-identical grip ends (mass/friction

    copied verbatim from lift's proven get_block_spec() -- the exact
    physical regime that finally made a friction-only pinch work, per
    the project notes) joined by a thin connecting rod, all one rigid
    body/freejoint. Grip-point placement (RIGHT_END_OFFSET/LEFT_END_OFFSET)
    and the reasoning behind END_Y_OFFSET live in mdp.py / this module's
    docstring.

    Being one rigid body is deliberate: if only one end is genuinely
    gripped, the ungripped end doesn't stay magically fixed -- it simply
    goes wherever the bar's actual rigid-body physics takes it (drags,
    tips, or stays resting on the table while the gripped end rises),
    which is exactly what produces the height-mismatch signal the
    together_* rewards in mdp.py gate on. No separate interlock or
    scripted logic is needed to enforce "together" -- it falls out of
    rigid-body dynamics plus the reward shaping.
    """
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_bar"
    spec.compiler.degree = False
    body = spec.worldbody.add_body(name="bar")
    body.add_freejoint(name="bar_free")
    end_half = (0.025, 0.025, 0.03)
    end_mass = 0.05
    end_friction = (2.5, 0.1, 0.01)
    # Both arms' finger collision geoms are priority=1, so a
    # default-priority bar end is outranked and the fingers' parameters
    # govern every grasp contact: the 2.5 grip friction authored here was
    # never read (measured: effective mu 1.0 on both ends). priority=2
    # puts these geoms above the fingers so their own friction applies.
    # condim=4 is required as well -- at the default condim=3 MuJoCo
    # ignores the torsional component, which is what resists the bar
    # twisting out of a two-point grasp.
    body.add_geom(
        name="bar_end_right_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=end_half,
        pos=bl_mdp.RIGHT_END_OFFSET,
        mass=end_mass,
        friction=end_friction,
        priority=2,
        condim=4,
        solref=(0.005, 1.0),
        rgba=(0.9, 0.2, 0.2, 1.0),
    )
    # Both arms' finger collision geoms are priority=1, so a
    # default-priority bar end is outranked and the fingers' parameters
    # govern every grasp contact: the 2.5 grip friction authored here was
    # never read (measured: effective mu 1.0 on both ends). priority=2
    # puts these geoms above the fingers so their own friction applies.
    # condim=4 is required as well -- at the default condim=3 MuJoCo
    # ignores the torsional component, which is what resists the bar
    # twisting out of a two-point grasp.
    body.add_geom(
        name="bar_end_left_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=end_half,
        pos=bl_mdp.LEFT_END_OFFSET,
        mass=end_mass,
        friction=end_friction,
        priority=2,
        condim=4,
        solref=(0.005, 1.0),
        rgba=(0.2, 0.3, 0.9, 1.0),
    )
    # Rod spans between the two end blocks' inner faces (offset ∓ half
    # width in y), connecting them into one visible bar. Not a grip
    # surface (the sensors above target the end geoms specifically), so
    # its own friction is unused by the reward logic; given a modest
    # generic value only for contact stability if the arm brushes it.
    rod_half_y = bl_mdp.END_Y_OFFSET - end_half[1]
    body.add_geom(
        name="bar_rod_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=(0.0, -rod_half_y, 0.0, 0.0, rod_half_y, 0.0),
        size=(0.008, 0, 0),
        mass=0.02,
        friction=(0.5, 0.005, 0.0001),
        rgba=(0.5, 0.5, 0.55, 1.0),
    )
    # 2026-08-20 NaN-bug fix, root-caused via mjlab's own NaN-guard dump
    # (a real crash from the first full 1024-env training run, not a
    # hypothetical): auto-computed inertia from these geoms alone gives
    # principal moments [2.733e-3, 2.723e-3, 5.15e-5] kg*m^2 -- the third
    # axis (the bar's own long/roll axis, through both grip points) is
    # ~53x LOWER than the other two, because both end blocks' mass sits
    # far from the bar's center (large tip-over resistance) but close to
    # the roll axis itself (~zero roll resistance contributed by the
    # separation). Confirmed via mj_setState replay of the crash dump:
    # the freejoint's ANGULAR velocity (not position, not any robot
    # joint) grew smoothly by orders of magnitude over ~14 steps
    # (2.8e2 -> 3.0e16) right up to the NaN -- the signature of a stiff
    # contact torque hitting a near-degenerate rotational inertia, not a
    # single-step singularity. Explicit fullinertia override pads ONLY
    # that roll axis (5.15e-5 -> 1.0e-3, ratio 53:1 -> ~2.7:1), same
    # total mass, same collision/visual geometry, same grip surfaces --
    # this changes nothing about what the task requires, it only removes
    # a numerical-modeling artifact of representing a real object with
    # two thin-ish box+cylinder primitives. Standard MuJoCo practice for
    # thin/light rotating bodies (see MuJoCo's own inertia-fitting
    # guidance), not a task simplification.
    body.mass = end_mass * 2 + 0.02
    body.fullinertia = [2.733e-3, 1.0e-3, 2.723e-3, 0.0, 0.0, 0.0]
    body.explicitinertial = True
    return spec


def openarm_bimanual_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm bimanual lift environment config."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "tool_to_right_end": ObservationTermCfg(
            func=bl_mdp.tool_to_end_obs,
            params={
                "robot_cfg": ROBOT_EE_RIGHT_CFG,
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.RIGHT_END_OFFSET,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "tool_to_left_end": ObservationTermCfg(
            func=bl_mdp.tool_to_end_obs,
            params={
                "robot_cfg": ROBOT_EE_LEFT_CFG,
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.LEFT_END_OFFSET,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "pinch_right": ObservationTermCfg(
            func=bl_mdp.pinch_obs, params={"sensor_name": "finger_bar_contact_right"}
        ),
        "pinch_left": ObservationTermCfg(
            func=bl_mdp.pinch_obs, params={"sensor_name": "finger_bar_contact_left"}
        ),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
    }

    # BOTH arms actively controlled -- no HoldDefaultPositionActionCfg in
    # this task. Four independent action terms (two 7-dim arm position
    # terms, two 1-dim finger effort terms) so the policy can genuinely
    # coordinate rather than being forced to mirror one arm's command onto
    # the other.
    right_arm = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=("openarm_right_joint[1-7]",),
        scale=BIMANUAL_ARM_SCALE,
        use_default_offset=True,
    )
    left_arm = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=("openarm_left_joint[1-7]",),
        scale=BIMANUAL_ARM_SCALE,
        use_default_offset=True,
    )
    actions: dict[str, ActionTermCfg] = {
        "right_arm": right_arm,
        "left_arm": left_arm,
        "right_squeeze": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=("openarm_right_finger_joint1",),
            scale=RIGHT_SQUEEZE_SCALE,
        ),
        "left_squeeze": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=("openarm_left_finger_joint1",),
            scale=LEFT_SQUEEZE_SCALE,
        ),
    }

    events = {
        "reset_robot_joints": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_bar": EventTermCfg(
            func=bl_mdp.reset_bar_uniform,
            mode="reset",
            params={"asset_cfg": BAR_CFG, "xy_range": 0.03},
        ),
        # Must run after reset_bar (overrides the subset it picks) --
        # An earlier attempt showed that neither individual nor together lift
        # rewards ever fired even once in 3000 iterations from a cold
        # start. Rewritten 2026-08-23 to drop the arm-joint teleport/anneal
        # entirely now that BIMANUAL_LIFT_HOME (above) IS the held-high
        # pose -- see mdp.py's HELD_HIGH_PROBABILITY docstring for the
        # yank-back bug this fixes and reset_bar_held_high's own docstring
        # for why only the bar needs moving now, exactly mirroring lift's
        # own reset_held_high. probability left at its default (0.15,
        # lift's own proven ratio) rather than passed explicitly here.
        "reset_bar_held_high": EventTermCfg(
            func=bl_mdp.reset_bar_held_high,
            mode="reset",
            params={
                "asset_cfg": BAR_CFG,
                "robot_joints_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Domain randomization, same pattern and magnitudes as lift's.
        # mode="startup" so each parallel env draws one fixed value for its
        # whole training life, matching mjlab's own reference convention.
        #
        # Randomizing the bar's friction only became meaningful once the end
        # geoms were given contact priority: at default priority the fingers
        # outranked them and MuJoCo never read the bar's friction at all, so
        # this term would have randomized a value with no effect, which is
        # exactly the dead-randomization defect the other tasks had.
        "dr_bar_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "bar", geom_names=("bar_end_right_geom", "bar_end_left_geom")
                ),
                "operation": "scale",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.4),
            },
        ),
        "dr_bar_mass": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("bar", body_names=("bar",)),
                "alpha_range": (-0.1, 0.1),
            },
        ),
        # No actuator_names filter: dr.pd_gains indexes the grouped actuator
        # configs rather than the individually-named joints every other dr
        # function indexes, so a regex here silently produces out-of-range
        # indices. Both arms are actuated in this task anyway.
        "dr_arm_gains": EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.85, 1.15),
                "kd_range": (0.85, 1.15),
                "operation": "scale",
            },
        ),
    }

    rewards = {
        # Split coarse/fine at UNCHANGED total weight (0.7+0.3 = the old 1.0),
        # so camping income and the success-bonus margin are untouched. The
        # coarse std of 0.2m is saturated (0.992) at the 18mm positioning error
        # that the measured grasp actually has, so on its own it supplies no
        # gradient for the last centimetre -- which is why the grasp forms high
        # on the block and peels off under load. The fine term restores that
        # gradient exactly where it was missing. Same coarse/fine pattern the
        # drawer task already uses.
        "reach_right": RewardTermCfg(
            func=bl_mdp.reach_bar_end_reward,
            weight=0.7,
            params={
                "std": 0.2,
                "robot_cfg": ROBOT_EE_RIGHT_CFG,
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.RIGHT_END_OFFSET,
            },
        ),
        "reach_fine_right": RewardTermCfg(
            func=bl_mdp.reach_bar_end_reward,
            weight=0.3,
            params={
                "std": 0.02,
                "robot_cfg": ROBOT_EE_RIGHT_CFG,
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.RIGHT_END_OFFSET,
            },
        ),
        # Split coarse/fine at UNCHANGED total weight (0.7+0.3 = the old 1.0),
        # so camping income and the success-bonus margin are untouched. The
        # coarse std of 0.2m is saturated (0.992) at the 18mm positioning error
        # that the measured grasp actually has, so on its own it supplies no
        # gradient for the last centimetre -- which is why the grasp forms high
        # on the block and peels off under load. The fine term restores that
        # gradient exactly where it was missing. Same coarse/fine pattern the
        # drawer task already uses.
        "reach_left": RewardTermCfg(
            func=bl_mdp.reach_bar_end_reward,
            weight=0.7,
            params={
                "std": 0.2,
                "robot_cfg": ROBOT_EE_LEFT_CFG,
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.LEFT_END_OFFSET,
            },
        ),
        "reach_fine_left": RewardTermCfg(
            func=bl_mdp.reach_bar_end_reward,
            weight=0.3,
            params={
                "std": 0.02,
                "robot_cfg": ROBOT_EE_LEFT_CFG,
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.LEFT_END_OFFSET,
            },
        ),
        "pinch_right": RewardTermCfg(
            func=bl_mdp.pinch_reward,
            weight=0.5,
            params={"sensor_name": "finger_bar_contact_right"},
        ),
        "pinch_left": RewardTermCfg(
            func=bl_mdp.pinch_reward,
            weight=0.5,
            params={"sensor_name": "finger_bar_contact_left"},
        ),
        "partial_pinch_right": RewardTermCfg(
            func=bl_mdp.partial_pinch_reward,
            weight=0.3,
            params={"sensor_name": "finger_bar_contact_right"},
        ),
        "partial_pinch_left": RewardTermCfg(
            func=bl_mdp.partial_pinch_reward,
            weight=0.3,
            params={"sensor_name": "finger_bar_contact_left"},
        ),
        "pinch_streak_together": RewardTermCfg(
            func=bl_mdp.together_pinch_streak_reward,
            weight=1.5,
            params={
                "sensor_right": "finger_bar_contact_right",
                "sensor_left": "finger_bar_contact_left",
            },
        ),
        "lift_rate_together": RewardTermCfg(
            func=bl_mdp.together_lift_rate_reward,
            weight=3.0,
            params={
                "sensor_right": "finger_bar_contact_right",
                "sensor_left": "finger_bar_contact_left",
                "asset_cfg": BAR_CFG,
            },
        ),
        # An earlier attempt showed that lift_rate_together alone stayed at exact
        # zero for 2500+ iterations after reach/grip/level all saturated --
        # min()-gating means both arms' exploration noise must coincide
        # before any lift reward appears from a cold start. These two
        # bootstrapping companions (see individual_lift_rate_reward's
        # docstring) let each arm discover "lifting my own end is
        # possible" independently. Weighted well below lift_rate_together
        # (0.5+0.5 vs 3.0) so together_* stays the dominant channel --
        # this can only help escape the plateau, not substitute for it.
        "individual_lift_right": RewardTermCfg(
            func=bl_mdp.individual_lift_rate_reward,
            weight=0.5,
            params={
                "sensor_name": "finger_bar_contact_right",
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.RIGHT_END_OFFSET,
                "side": "right",
            },
        ),
        "individual_lift_left": RewardTermCfg(
            func=bl_mdp.individual_lift_rate_reward,
            weight=0.5,
            params={
                "sensor_name": "finger_bar_contact_left",
                "asset_cfg": BAR_CFG,
                "local_offset": bl_mdp.LEFT_END_OFFSET,
                "side": "left",
            },
        ),
        "held_high_together": RewardTermCfg(
            func=bl_mdp.together_held_high_reward,
            weight=2.0,
            params={
                "sensor_right": "finger_bar_contact_right",
                "sensor_left": "finger_bar_contact_left",
                "asset_cfg": BAR_CFG,
            },
        ),
        "level": RewardTermCfg(
            func=bl_mdp.level_reward,
            weight=1.0,
            params={
                "sensor_right": "finger_bar_contact_right",
                "sensor_left": "finger_bar_contact_left",
                "asset_cfg": BAR_CFG,
                "std": 0.03,
            },
        ),
        # An earlier attempt showed that the ORIGINAL comment here used a gamma-
        # discounted-infinite-horizon estimate ("810") to justify weight
        # 1000 -- that framework was superseded by an earlier attempt/an earlier attempt's actual
        # postmortem this same session, which established the real
        # comparison is bonus_reward (weight*dt, one-shot) vs. the FULL
        # EPISODE-LENGTH camping income (steady-state weight-sum *
        # episode_length_s), since a policy that never triggers success
        # collects dense income for the WHOLE episode, not a
        # discounted-forever approximation. Re-run with the corrected
        # framework: steady-state per-step income = reach_right+reach_left
        # (2.0) + pinch_right+pinch_left(1.0) + partial_pinch_right+
        # partial_pinch_left(0.6) + pinch_streak_together capped(1.5) +
        # held_high_together(2.0) + level(1.0) = 8.1/step -> full-episode
        # (8.0s) camping ceiling = 64.8. The old weight=1000 (20.0 real
        # reward) was NEVER sufficient -- 20.0 << 64.8, the exact same
        # "camping pays more than succeeding" bug an earlier attempt had, just never
        # observed here because the policy never got tall enough to
        # exploit it (it plateaued at reach+grip+level, height ~0, before
        # ever reaching the point where this would matter). Corrected to
        # weight=3660 (73.2 real reward, 1.13x the 3240 minimum) --
        # matching an earlier attempt's own validated ratio (which worked) rather than
        # an earlier attempt's 1.42x (which destabilized training), since this is now
        # the only empirical data point available for what ratio is safe.
        "success": RewardTermCfg(
            func=bl_mdp.together_success_bonus, weight=3660.0, params={}
        ),
        "descent": RewardTermCfg(
            func=bl_mdp.bar_descent_penalty,
            weight=-2.0,
            params={"asset_cfg": BAR_CFG},
        ),
        "overshoot": RewardTermCfg(
            func=bl_mdp.together_overshoot_penalty,
            weight=-10.0,
            params={"asset_cfg": BAR_CFG},
        ),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.05,
            params={
                "max_vel": 1.0,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("openarm_(left|right)_.*",)
                ),
            },
        ),
        "joint_pos_limits": RewardTermCfg(
            func=base_mdp.joint_pos_limits,
            weight=-10.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("openarm_(left|right)_joint[1-7]",)
                )
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "lifted_together": TerminationTermCfg(
            func=bl_mdp.lifted_together,
            params={
                "sensor_right": "finger_bar_contact_right",
                "sensor_left": "finger_bar_contact_left",
                "asset_cfg": BAR_CFG,
            },
        ),
        "bar_fell": TerminationTermCfg(
            func=bl_mdp.bar_fell,
            params={"asset_cfg": BAR_CFG},
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_bimanual_lift_robot_cfg(),
                # Reused unmodified from lift (same table, same world placement)
                # -- its footprint (half-extents x=0.41, y=0.55) already spans
                # well past this bar's y=±0.16-ish grip points, so no new table
                # geometry is needed.
                "table": EntityCfg(spec_fn=get_bimanual_table_spec),
                "bar": EntityCfg(
                    spec_fn=get_bar_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=bl_mdp.BAR_START),
                ),
            },
            num_envs=1,
            env_spacing=2.5,
            sensors=(FINGER_BAR_SENSOR_RIGHT, FINGER_BAR_SENSOR_LEFT),
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="openarm_right_base_link",
            distance=2.0,
            elevation=-20.0,
            azimuth=220.0,
        ),
        sim=SimulationCfg(
            # Bumped from lift's 150/1000: this scene has twice the manipulated
            # contact pairs (two grippers x their own bar end, plus the shared
            # table) that lift's single-block scene has. A conservative,
            # untested increase -- purely a static buffer size, so higher is
            # safe; flagged for the human to watch for MuJoCo buffer-overflow
            # warnings at the GPU smoke test and raise further if needed.
            nconmax=200,
            njmax=1400,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        # Matches lift's proven episode budget exactly -- same actuator/
        # timestep regime, no evidence yet to retune for the bimanual case.
        episode_length_s=8.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
