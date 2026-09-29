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

"""Environment configuration for the OpenArm drawer-pulling task.

The cabinet sits offset from the arm base column, validated against the
home pose so the drawer's slide sweep clears the parked arms. The
table slab is part of the cabinet entity so the workspace matches the
robot's own scenes.
"""

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
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
from mjlab.tasks.velocity import mdp as base_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from ...actions import HoldDefaultPositionActionCfg
from ...openarm_bimanual import (
    BIMANUAL_ACTION_SCALE,
    EE_SITE_RIGHT,
    get_bimanual_robot_cfg,
)
from . import mdp as drawer_mdp

# Caged start = PULL_POSE_CLOSED (see mdp.py): the jaws close ACROSS the
# handle bar with the bar between the pads, and the gripper touches nothing
# at any spawn opening.
DRAWER_CAGED_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "openarm_right_joint1": 0.268,
        "openarm_right_joint2": 0.6291,
        "openarm_right_joint3": -0.4159,
        "openarm_right_joint4": 1.4393,
        "openarm_right_joint5": -0.9643,
        "openarm_right_joint6": 0.4809,
        "openarm_right_joint7": -0.0296,
        "openarm_right_finger_joint[12]": -0.25,
        "openarm_left_joint4": 1.5708,
        "openarm_left_joint[12356]": 0.0,
        "openarm_left_joint7": 0.0,
        "openarm_left_finger_joint[12]": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_drawer_robot_cfg() -> EntityCfg:
    """Return the bimanual robot config, homed to the drawer's caged grasp.

    Every joint keeps the shared actuator defaults except the right
    finger's grip: it is softened (stiffness 150 -> 60) so the cage
    passively conforms to the handle bar via contact force across a long
    90mm pull, instead of rigidly fighting a possible small tracking
    mismatch (impedance control, the same principle used for the lift
    task's squeeze-and-hold).
    """
    cfg = get_bimanual_robot_cfg()
    cfg.init_state = DRAWER_CAGED_HOME
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
        BuiltinPositionActuatorCfg(
            target_names_expr=("openarm_left_finger_joint1",),
            stiffness=150.0,
            damping=2.0,
            effort_limit=30.0,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("openarm_right_finger_joint1",),
            stiffness=60.0,
            damping=2.0,
            effort_limit=30.0,
        ),
    )
    cfg.articulation = EntityArticulationInfoCfg(
        actuators=actuators, soft_joint_pos_limit_factor=0.9
    )
    return cfg


# Right-finger scale recomputed for the softened stiffness (same
# 0.25*effort/stiffness convention as BIMANUAL_ACTION_SCALE, but the
# lower kp means the same raw action now needs a larger scale to reach a
# comparable target-angle range). The general "(left|right)_finger_joint1"
# key is dropped and replaced with an explicit "right_finger_joint1" key:
# this action only ever targets right-side actuators anyway, but keeping
# two regex keys that could both match the same actuator name risks
# ambiguous resolution.
DRAWER_ARM_SCALE = {k: v for k, v in BIMANUAL_ACTION_SCALE.items() if "finger" not in k}
DRAWER_ARM_SCALE["openarm_right_finger_joint1"] = 0.25 * 30.0 / 60.0

# World-frame layout: table top z=0.40, cabinet origin z=0.46 (handle bar
# ends up ~0.505). Entity origin = cabinet origin.
CABINET_POS = (0.50, -0.26, 0.46)
_TABLE_REL_POS = (0.47 - CABINET_POS[0], 0.0 - CABINET_POS[1], 0.36 - CABINET_POS[2])

FULL_OPENING = 0.09
MAX_PULL_SPEED = 0.15  # A grasp-pull moves the slide slowly.
# Peak slide speed allowed at ANY point of a successful episode. Well above a
# grasped pull, well below a yank (a policy that yanks the drawer open peaks
# around 1.2 m/s).
PEAK_PULL_SPEED = 0.5


def get_cabinet_spec() -> mujoco.MjSpec:
    """Build the cabinet + sliding-drawer fixture spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "drawer_cabinet"
    cab = spec.worldbody.add_body(name="cabinet")

    def fixture(name, size, pos):
        cab.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            rgba=(0.55, 0.45, 0.32, 1.0),
            friction=(1.0, 0.01, 0.01),
        )

    # Table slab (cell footprint), so the workspace matches the scenes.
    cab.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.41, 0.55, 0.04),
        pos=_TABLE_REL_POS,
        rgba=(0.82, 0.71, 0.55, 1.0),
        friction=(1.0, 0.005, 0.0001),
    )

    fixture("drawer_base", (0.05, 0.055, 0.03), (0, 0, -0.03))
    fixture("drawer_bottom", (0.05, 0.055, 0.005), (0, 0, 0.005))
    fixture("drawer_top", (0.05, 0.055, 0.005), (0, 0, 0.105))
    fixture("drawer_back", (0.005, 0.055, 0.05), (0.05, 0, 0.055))
    fixture("drawer_sideL", (0.055, 0.005, 0.05), (0, 0.05, 0.055))
    fixture("drawer_sideR", (0.055, 0.005, 0.05), (0, -0.05, 0.055))

    drawer = cab.add_body(name="drawer", pos=(0, 0, 0.045))
    drawer.add_joint(
        name="drawer_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(1, 0, 0),
        range=(-0.10, 0.0),
        damping=1.0,
        frictionloss=0.05,
    )
    drawer.add_geom(
        name="drawer_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.038, 0.038, 0.03),
        mass=0.20,
        friction=(0.4, 0.01, 0.01),
        rgba=(0.30, 0.45, 0.62, 1.0),
    )
    drawer.add_geom(
        name="drawer_front",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.006, 0.04, 0.028),
        pos=(-0.046, 0, 0),
        mass=0.05,
        friction=(0.4, 0.01, 0.01),
        rgba=(0.22, 0.36, 0.52, 1.0),
    )
    # Stand-off handle bar on two stems (gripper closes on the bar). The stems
    # are 74 mm apart (66 mm clear): the fingertip pads are ~55 mm plates, and
    # with the jaws across the bar their long side lies along it, so stems
    # closer together (these were 28 mm clear) block every grip on the bar.
    drawer.add_geom(
        name="drawer_handle",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=(-0.092, -0.04, 0, -0.092, 0.04, 0),
        size=(0.007, 0, 0),
        mass=0.02,
        rgba=(0.2, 0.2, 0.22, 1.0),
        # The right-arm finger collision geoms are priority=1, so a
        # default-priority handle is outranked and their parameters govern
        # every fingertip contact -- which left dr_handle_friction below
        # randomizing a value MuJoCo never read (measured: effective mu
        # stayed 1.0 across the whole 0.6-1.4 range). priority=2 puts this
        # geom above them, so its own friction, and the randomization
        # applied to it, actually reach the contact.
        # (robot.FINGERTIP_COLLISION's priority=2 override is not in play:
        # its regex matches only the *_left_* geoms, not this arm.)
        priority=2,
        condim=3,
        friction=(1.0, 0.01, 0.01),
        solref=(0.005, 1.0),
    )
    drawer.add_geom(
        name="drawer_handle_stem1",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=(-0.052, -0.037, 0, -0.092, -0.037, 0),
        size=(0.004, 0, 0),
        mass=0.005,
        rgba=(0.2, 0.2, 0.22, 1.0),
    )
    drawer.add_geom(
        name="drawer_handle_stem2",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=(-0.052, 0.037, 0, -0.092, 0.037, 0),
        size=(0.004, 0, 0),
        mass=0.005,
        rgba=(0.2, 0.2, 0.22, 1.0),
    )
    drawer.add_site(name="handle_site", pos=(-0.092, 0, 0), size=(0.005, 0, 0))
    return spec


ROBOT_EE_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_RIGHT,))
CABINET_JOINT_CFG = SceneEntityCfg("cabinet", joint_names=("drawer_slide",))
CABINET_HANDLE_CFG = SceneEntityCfg("cabinet", site_names=("handle_site",))

# Hand-subtree vs whole drawer body, not finger-pads vs the handle bar: a
# caged pull's drag force flows through palm/knuckle contact, not the two
# finger pads alone, so finger-only sensing would be structurally blind.
# Pulling via any hand-drawer contact is honest for a drawer; the cabinet
# shell stays excluded.
FINGER_HANDLE_SENSOR = ContactSensorCfg(
    name="finger_handle_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern="openarm_right_ee_base_link",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="drawer", entity="cabinet"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)


def openarm_drawer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm drawer-pulling environment config."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "drawer_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": CABINET_JOINT_CFG},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        ),
        "ee_to_handle": ObservationTermCfg(
            func=drawer_mdp.ee_to_target,
            params={"robot_cfg": ROBOT_EE_CFG, "target_cfg": CABINET_HANDLE_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "handle_contact": ObservationTermCfg(
            func=drawer_mdp.fingers_on_handle_obs,
            params={"sensor_name": "finger_handle_contact"},
        ),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
    }
    # Right arm + right gripper only; the left arm's actuators servo-hold
    # home via a zero-dim action term (mjlab leaves unactioned ctrl at 0,
    # which would let the parked arm sag).
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("openarm_right_.*",),
            scale=DRAWER_ARM_SCALE,
            use_default_offset=True,
        ),
        "left_hold": HoldDefaultPositionActionCfg(
            entity_name="robot",
            actuator_names=("openarm_left_.*",),
            scale=1.0,
            use_default_offset=True,
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
        # Narrow start jitter (0-10mm) keeps the handle at the caged
        # start; a depth curriculum is superseded by reset_along_pull.
        "reset_drawer": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.01, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("cabinet", joint_names=(".*",)),
            },
        ),
        # Must run after reset_drawer.
        "reset_along_pull": EventTermCfg(
            func=drawer_mdp.reset_along_pull,
            mode="reset",
            params={
                "asset_cfg": CABINET_JOINT_CFG,
                "robot_joints_cfg": SceneEntityCfg("robot"),
                "max_opening": 0.08,
            },
        ),
        # Records each episode's start opening so progress rewards pay
        # only for opening the policy produced.
        "record_drawer_start": EventTermCfg(
            func=drawer_mdp.record_drawer_start,
            mode="reset",
            params={"asset_cfg": CABINET_JOINT_CFG},
        ),
        # Domain randomization, same verified pattern as valve/door:
        # mode="startup", proportional ranges, modest magnitude.
        "dr_handle_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("cabinet", geom_names=("drawer_handle",)),
                "operation": "scale",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.4),
            },
        ),
        "dr_slide_friction": EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": CABINET_JOINT_CFG,
                "operation": "scale",
                "distribution": "uniform",
                "ranges": (0.5, 2.0),
            },
        ),
        "dr_slide_damping": EventTermCfg(
            mode="startup",
            func=dr.joint_damping,
            params={
                "asset_cfg": CABINET_JOINT_CFG,
                "operation": "scale",
                "distribution": "uniform",
                "ranges": (0.5, 2.0),
            },
        ),
        "dr_drawer_mass": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("cabinet", body_names=("drawer",)),
                "alpha_range": (-0.1, 0.1),
            },
        ),
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
        "reach_handle": RewardTermCfg(
            func=drawer_mdp.reach_target_reward,
            weight=1.0,
            params={
                "std": 0.2,
                "robot_cfg": ROBOT_EE_CFG,
                "target_cfg": CABINET_HANDLE_CFG,
            },
        ),
        # handle_contact runs first: it's what freezes the engagement
        # depth that reach_fine/frontal_grasp below read, so this
        # ordering keeps it same-step-fresh instead of one step stale.
        "handle_contact": RewardTermCfg(
            func=drawer_mdp.handle_contact_reward,
            weight=0.5,
            params={
                "sensor_name": "finger_handle_contact",
                "asset_cfg": CABINET_JOINT_CFG,
            },
        ),
        # Tight cage-on-bar kernel, approach phase only (latched off
        # after first genuine contact, not continuously depth-faded).
        "reach_fine": RewardTermCfg(
            func=drawer_mdp.approach_precision_reward,
            weight=1.5,
            params={
                "std": 0.05,
                "robot_cfg": ROBOT_EE_CFG,
                "cabinet_cfg": CABINET_HANDLE_CFG,
            },
        ),
        "open_shaping": RewardTermCfg(
            func=drawer_mdp.open_progress_shaping_reward,
            weight=2.0,
            params={
                "sensor_name": "finger_handle_contact",
                "asset_cfg": CABINET_JOINT_CFG,
                "gamma": 0.99,
            },
        ),
        "frontal_grasp": RewardTermCfg(
            func=drawer_mdp.frontal_grasp_reward,
            weight=2.0,
            params={"robot_cfg": ROBOT_EE_CFG, "pitch": 0.2618, "fade": 0.7},
        ),
        "pull_rate": RewardTermCfg(
            func=drawer_mdp.pull_rate_reward,
            weight=3.0,
            params={
                "sensor_name": "finger_handle_contact",
                "max_speed": MAX_PULL_SPEED,
                "asset_cfg": CABINET_JOINT_CFG,
            },
        ),
        "closing": RewardTermCfg(
            func=drawer_mdp.closing_speed_penalty,
            weight=-2.0,
            params={"asset_cfg": CABINET_JOINT_CFG},
        ),
        "success": RewardTermCfg(
            func=drawer_mdp.terminated_by,
            weight=900.0,
            params={"term_name": "held_fully_open"},
        ),
        "drawer_speed": RewardTermCfg(
            func=drawer_mdp.drawer_speed_penalty,
            weight=-5.0,
            params={"max_speed": MAX_PULL_SPEED, "asset_cfg": CABINET_JOINT_CFG},
        ),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
        # Fingers excluded: their default (closed, qpos 0) lies outside
        # the 0.9 soft limits, which would make this a constant bias the
        # policy cannot influence (left fingers are servo-held at 0).
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
        "held_fully_open": TerminationTermCfg(
            func=drawer_mdp.drawer_held_fully_open,
            params={
                "sensor_name": "finger_handle_contact",
                "threshold": FULL_OPENING,
                "max_speed": MAX_PULL_SPEED,
                "peak_speed": PEAK_PULL_SPEED,
                "asset_cfg": CABINET_JOINT_CFG,
            },
        ),
    }
    cabinet_init = EntityCfg.InitialStateCfg(
        pos=CABINET_POS,
        joint_pos={"drawer_slide": 0.0},
        joint_vel={"drawer_slide": 0.0},
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_drawer_robot_cfg(),
                "cabinet": EntityCfg(spec_fn=get_cabinet_spec, init_state=cabinet_init),
            },
            num_envs=1,
            env_spacing=2.5,
            sensors=(FINGER_HANDLE_SENSOR,),
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        # A fixed WORLD camera rather than an ASSET_BODY tracking one:
        # MuJoCo tracking cameras follow the tracked body's subtree COM
        # (here the whole right arm), so offscreen-rendered videos sway
        # with every arm motion. mjlab's interactive viewer works around
        # that, so the artifact only shows up in recorded video. Viewed from
        # the robot's right side: the drawer slides out along -x toward the
        # robot, so a camera behind the cabinet hides the drawer front.
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.38, -0.26, 0.52),
            distance=0.9,
            elevation=-20.0,
            azimuth=80.0,
        ),
        sim=SimulationCfg(
            nconmax=150,
            njmax=1000,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=8.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
