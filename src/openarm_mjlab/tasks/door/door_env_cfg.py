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

"""Environment configuration for the OpenArm door-swing task.

Cage the vertical handle bar and swing the door. Classical baseline: 53.7
deg (weld-assisted). Target: 1.0 rad (57.3 deg), contact-only.
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

from ...actions import HoldDefaultPositionActionCfg
from ...openarm_bimanual import (
    BIMANUAL_ACTION_SCALE,
    EE_SITE_RIGHT,
    get_bimanual_robot_cfg,
)
from . import mdp as door_mdp

DOOR_POS = (0.29, -0.10, 0.40)

# Rounds 1-3 lesson: the approach phase is an exploration cliff (the light
# door flees from any touch; the 7 mm bar is never caged by accident in 768
# episodes). Episodes therefore START CAGED: the task's home pose is a
# DLS-IK solution (residual 0.07 mm) placing the finger-cage point at the
# resting handle, fingers open around the bar. Re-solved 2026-08-28
# when the handle was moved 30mm proud of the panel face: this pose is
# IK'd to the handle, so it had to move with it (it was ~40mm off
# otherwise). Position AND orientation solved (0.002mm / 0.000deg),
# keeping the previous wrist orientation so only the standoff changed. Comparable to the classical
# skill, which is also staged to its grasp by a scripted approach.
# Zero-action HOLDS this pose (position actions offset from the default).
CAGED_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "openarm_right_joint1": -0.4154,
        "openarm_right_joint2": -0.0994,
        "openarm_right_joint3": -0.2940,
        "openarm_right_joint4": 1.8242,
        "openarm_right_joint5": -0.1300,
        "openarm_right_joint6": -0.0327,
        "openarm_right_joint7": 0.0081,
        "openarm_right_finger_joint[12]": -0.25,
        "openarm_left_joint4": 1.5708,
        "openarm_left_joint[12356]": 0.0,
        "openarm_left_joint7": 0.0,
        "openarm_left_finger_joint[12]": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_door_robot_cfg() -> EntityCfg:
    """Return the bimanual robot config, homed to the door's caged grasp."""
    cfg = get_bimanual_robot_cfg()
    cfg.init_state = CAGED_HOME
    return cfg


def get_door_spec() -> mujoco.MjSpec:
    """Build the post + hinged-panel door fixture spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "door_fixture"
    spec.compiler.degree = False
    base = spec.worldbody.add_body(name="door_base")
    base.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.41, 0.55, 0.04),
        pos=(0.47 - DOOR_POS[0], -DOOR_POS[1], 0.36 - DOOR_POS[2]),
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
        damping=0.2,
        frictionloss=0.02,
    )
    door.add_geom(
        name="door_panel",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.006, 0.06, 0.05),
        pos=(0, 0.075, 0),
        mass=0.08,
        rgba=(0.55, 0.45, 0.3, 1.0),
    )
    # The handle stands 30 mm PROUD of the panel face on two short posts,
    # the same arrangement the drawer fixture uses. It was previously a
    # cylinder centred inside the panel: with panel half-thickness 0.006 and
    # handle radius 0.007 it protruded just 1.0 mm from each face and its
    # z-range lay entirely within the panel, so a parallel gripper could not
    # close around it and the contact gate could be satisfied by pressing
    # the panel rather than caging the bar. The robot base is at x=0 and the
    # door at x=0.29, so the near face is door-local x=-0.006 and the handle
    # stands off toward the robot. y is left at 0.13 so the hinge moment arm
    # -- and therefore the task's difficulty -- is unchanged.
    door.add_geom(
        name="door_handle",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=(-0.036, 0.13, -0.025, -0.036, 0.13, 0.025),
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
    for z in (-0.02, 0.02):
        door.add_geom(
            name=f"door_handle_post_{'hi' if z > 0 else 'lo'}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=(-0.006, 0.13, z, -0.036, 0.13, z),
            size=(0.004, 0, 0),
            mass=0.005,
            rgba=(0.2, 0.2, 0.22, 1.0),
        )
    door.add_site(name="handle_site", pos=(-0.036, 0.13, 0), size=(0.005, 0, 0))
    return spec


ROBOT_EE_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_RIGHT,))
DOOR_JOINT_CFG = SceneEntityCfg("door", joint_names=("door_hinge",))
DOOR_HANDLE_CFG = SceneEntityCfg("door", site_names=("handle_site",))

FINGER_DOOR_SENSOR = ContactSensorCfg(
    name="finger_grip_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"openarm_right_ee_(inner|outer)_finger",
        entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="door_handle", entity="door"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)


def openarm_door_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm door-swing environment config."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "door_angle": ObservationTermCfg(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": DOOR_JOINT_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ee_to_handle": ObservationTermCfg(
            func=door_mdp.ee_to_target,
            params={"robot_cfg": ROBOT_EE_CFG, "target_cfg": DOOR_HANDLE_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "handle_contact": ObservationTermCfg(
            func=door_mdp.fingers_on_handle_obs,
            params={"sensor_name": "finger_grip_contact"},
        ),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
    }
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("openarm_right_.*",),
            scale=BIMANUAL_ACTION_SCALE,
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
        "reset_door": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("door", joint_names=(".*",)),
            },
        ),
        "record_door_start": EventTermCfg(
            func=door_mdp.record_door_start,
            mode="reset",
            params={"asset_cfg": DOOR_JOINT_CFG},
        ),
        # Domain randomization, same pattern verified on valve: mode="startup"
        # (each of the N parallel envs draws ONE fixed value for its whole
        # training life), proportional ranges (operation="scale"), modest
        # magnitude (roughly +-20-40%) to certify robustness rather than
        # make the task harder.
        "dr_handle_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("door", geom_names=("door_handle",)),
                "operation": "scale",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.4),
            },
        ),
        "dr_door_joint_friction": EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": DOOR_JOINT_CFG,
                "operation": "scale",
                "distribution": "uniform",
                "ranges": (0.5, 2.0),
            },
        ),
        "dr_door_joint_damping": EventTermCfg(
            mode="startup",
            func=dr.joint_damping,
            params={
                "asset_cfg": DOOR_JOINT_CFG,
                "operation": "scale",
                "distribution": "uniform",
                "ranges": (0.5, 2.0),
            },
        ),
        "dr_door_mass": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("door", body_names=("door",)),
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
            func=door_mdp.reach_target_reward,
            weight=1.0,
            params={
                "std": 0.2,
                "robot_cfg": ROBOT_EE_CFG,
                "target_cfg": DOOR_HANDLE_CFG,
            },
        ),
        "handle_contact": RewardTermCfg(
            func=door_mdp.contact_reward,
            weight=0.5,
            params={"sensor_name": "finger_grip_contact"},
        ),
        "swing_rate": RewardTermCfg(
            func=door_mdp.swing_rate_reward,
            weight=4.0,
            params={"sensor_name": "finger_grip_contact", "asset_cfg": DOOR_JOINT_CFG},
        ),
        "success": RewardTermCfg(
            func=door_mdp.terminated_by,
            weight=400.0,
            params={"term_name": "swung_target"},
        ),
        "uncontrolled": RewardTermCfg(
            func=door_mdp.uncontrolled_motion_penalty,
            weight=-2.0,
            params={"sensor_name": "finger_grip_contact", "asset_cfg": DOOR_JOINT_CFG},
        ),
        "reverse": RewardTermCfg(
            func=door_mdp.reverse_rate_penalty,
            weight=-1.0,
            params={"asset_cfg": DOOR_JOINT_CFG},
        ),
        "overspeed": RewardTermCfg(
            func=door_mdp.overspeed_penalty,
            weight=-2.0,
            params={"asset_cfg": DOOR_JOINT_CFG},
        ),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.05,
            params={
                "max_vel": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=("openarm_right_.*",)),
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
        "swung_target": TerminationTermCfg(
            func=door_mdp.swung_target,
            params={"sensor_name": "finger_grip_contact", "asset_cfg": DOOR_JOINT_CFG},
        ),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_door_robot_cfg(),
                "door": EntityCfg(
                    spec_fn=get_door_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=DOOR_POS),
                ),
            },
            num_envs=1,
            env_spacing=2.5,
            sensors=(FINGER_DOOR_SENSOR,),
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
        # that, so the artifact only shows up in recorded video.
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.33, -0.10, 0.52),
            distance=1.6,
            elevation=-20.0,
            azimuth=220.0,
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
