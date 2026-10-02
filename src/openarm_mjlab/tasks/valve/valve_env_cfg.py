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

"""Environment configuration for the OpenArm valve-turning task.

Fixture geometry: pipe base, hinged hub + lever, vertical grip cylinder at
the lever tip. A rotary task, so the arm-branch depth wall that constrains
the drawer task does not apply here.
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
from ...common_mdp import fingers_on_handle_obs
from . import mdp as valve_mdp

VALVE_POS = (0.25, 0.0, 0.40)
_TABLE_REL_POS = (0.47 - VALVE_POS[0], 0.0, 0.36 - VALVE_POS[2])


def get_valve_spec() -> mujoco.MjSpec:
    """Build the pipe + hinged-lever valve fixture spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "valve_fixture"
    # Programmatic MjSpec defaults to degrees: without this, the hinge
    # range (-3, 3) compiles to +-3 degrees, not +-3 rad.
    spec.compiler.degree = False

    base = spec.worldbody.add_body(name="valve_base")
    base.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.41, 0.55, 0.04),
        pos=_TABLE_REL_POS,
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
        damping=0.5,
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
    valve.add_geom(
        name="valve_lever",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.035, 0.008, 0.006),
        pos=(0.035, 0, 0),
        mass=0.05,
        rgba=(0.7, 0.2, 0.2, 1.0),
    )
    valve.add_geom(
        name="valve_grip",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=(0.062, 0, -0.02, 0.062, 0, 0.02),
        size=(0.007, 0, 0),
        mass=0.02,
        rgba=(0.2, 0.2, 0.22, 1.0),
        # The right-arm finger collision geoms are priority=1, so a
        # default-priority grip is outranked and their parameters govern
        # every fingertip contact -- which left dr_grip_friction below
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
    valve.add_site(name="grip_site", pos=(0.062, 0, 0), size=(0.005, 0, 0))
    return spec


ROBOT_EE_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_RIGHT,))
VALVE_JOINT_CFG = SceneEntityCfg("valve", joint_names=("valve_turn",))
VALVE_GRIP_CFG = SceneEntityCfg("valve", site_names=("grip_site",))

FINGER_GRIP_SENSOR = ContactSensorCfg(
    name="finger_grip_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"openarm_right_ee_(inner|outer)_finger",
        entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="valve_grip", entity="valve"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)


def openarm_valve_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm valve-turning environment config."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "valve_angle": ObservationTermCfg(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": VALVE_JOINT_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "ee_to_grip": ObservationTermCfg(
            func=valve_mdp.ee_to_target,
            params={"robot_cfg": ROBOT_EE_CFG, "target_cfg": VALVE_GRIP_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "grip_contact": ObservationTermCfg(
            func=fingers_on_handle_obs,
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
        "reset_valve": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.3, 0.3),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("valve", joint_names=(".*",)),
            },
        ),
        "record_valve_start": EventTermCfg(
            func=valve_mdp.record_valve_start,
            mode="reset",
            params={"asset_cfg": VALVE_JOINT_CFG},
        ),
        # Domain randomization: each of the N parallel envs draws ONE fixed
        # value at startup that persists its whole training life, matching
        # mjlab's own reference convention (tasks/manipulation/
        # lift_cube_env_cfg.py's fingertip_friction_* events). Ranges are
        # proportional (operation="scale") and deliberately modest
        # (roughly +-20-40%): the goal is to certify robustness, not make
        # the task harder.
        "dr_grip_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("valve", geom_names=("valve_grip",)),
                "operation": "scale",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.4),
            },
        ),
        "dr_valve_joint_dynamics": EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": VALVE_JOINT_CFG,
                "operation": "scale",
                "distribution": "uniform",
                "ranges": (0.5, 2.0),
            },
        ),
        "dr_valve_joint_damping": EventTermCfg(
            mode="startup",
            func=dr.joint_damping,
            params={
                "asset_cfg": VALVE_JOINT_CFG,
                "operation": "scale",
                "distribution": "uniform",
                "ranges": (0.5, 2.0),
            },
        ),
        "dr_valve_mass": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("valve", body_names=("valve",)),
                # alpha_range=(-0.1, 0.1): mass/inertia scale by e^(2*alpha),
                # roughly 0.82x-1.22x, a physically-consistent density
                # change (mass and inertia scaled together, per Rucker &
                # Wensing 2022) rather than the bare body_mass shortcut.
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
        "reach_grip": RewardTermCfg(
            func=valve_mdp.reach_target_reward,
            weight=1.0,
            params={
                "std": 0.2,
                "robot_cfg": ROBOT_EE_CFG,
                "target_cfg": VALVE_GRIP_CFG,
            },
        ),
        "grip_contact": RewardTermCfg(
            func=valve_mdp.contact_reward,
            weight=0.5,
            params={"sensor_name": "finger_grip_contact"},
        ),
        "turn_rate": RewardTermCfg(
            func=valve_mdp.turn_rate_reward,
            weight=3.0,
            params={"sensor_name": "finger_grip_contact", "asset_cfg": VALVE_JOINT_CFG},
        ),
        "turn_progress_shaping": RewardTermCfg(
            func=valve_mdp.turn_progress_shaping_reward,
            weight=2.0,
            params={"sensor_name": "finger_grip_contact", "asset_cfg": VALVE_JOINT_CFG},
        ),
        "success": RewardTermCfg(
            func=valve_mdp.terminated_by,
            weight=500.0,
            params={"term_name": "turned_target"},
        ),
        "reverse": RewardTermCfg(
            func=valve_mdp.reverse_rate_penalty,
            weight=-1.0,
            params={"asset_cfg": VALVE_JOINT_CFG},
        ),
        "overspeed": RewardTermCfg(
            func=valve_mdp.overspeed_penalty,
            weight=-2.0,
            params={"asset_cfg": VALVE_JOINT_CFG},
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
        "turned_target": TerminationTermCfg(
            func=valve_mdp.turned_target,
            params={"sensor_name": "finger_grip_contact", "asset_cfg": VALVE_JOINT_CFG},
        ),
    }
    valve_init = EntityCfg.InitialStateCfg(
        pos=VALVE_POS,
        joint_pos={"valve_turn": 0.0},
        joint_vel={"valve_turn": 0.0},
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_bimanual_robot_cfg(),
                "valve": EntityCfg(spec_fn=get_valve_spec, init_state=valve_init),
            },
            num_envs=1,
            env_spacing=2.5,
            sensors=(FINGER_GRIP_SENSOR,),
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
            # Framed between the valve and the shoulders: the lever is only
            # ~6 cm, so the wide table shot used by the other tasks hides
            # what the gripper is doing, but the whole arm should stay in
            # frame.
            lookat=(0.16, -0.05, 0.55),
            distance=0.85,
            elevation=-20.0,
            azimuth=212.0,
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
