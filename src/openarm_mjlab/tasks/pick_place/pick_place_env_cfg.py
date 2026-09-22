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

"""Environment configuration for the OpenArm pick & place task."""

import math

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from ...robot import (
    LEFT_FINGER_HOME,
    LEFT_FINGERTIP_GEOMS,
    LEFT_GRASP_SITE,
    TABLE_TOP_Z,
    get_openarm_robot_cfg,
)
from . import mdp as pick_mdp

##
# Scene geometry. Heights derive from TABLE_TOP_Z (read from the Cell asset)
# and the cube/tray dimensions below, so only the margins are tuned numbers.
##

CUBE_HALF_SIZE = 0.02  # get_cube_spec's box geom uses this.
TRAY_FLOOR_TOP_DZ = 0.01  # Top of the tray bottom plate, relative to tray root.
TRAY_WALL_TOP_DZ = 0.04  # Top of the tray walls, relative to tray root.

CUBE_TABLE_REST_Z = TABLE_TOP_Z + CUBE_HALF_SIZE
CUBE_TRAY_REST_DZ = TRAY_FLOOR_TOP_DZ + CUBE_HALF_SIZE
# Spawn just above the resting height (5 mm drop). Must stay below LIFT_MIN_Z
# so an un-grasped cube never earns lift reward at reset.
CUBE_SPAWN_POS = (0.45, 0.0, CUBE_TABLE_REST_Z + 0.005)
TRAY_POS = (0.47, 0.15, TABLE_TOP_Z)
# Cube counts as lifted 2 cm above its table resting height.
LIFT_MIN_Z = CUBE_TABLE_REST_Z + 0.02
# Transport pays out only when the cube bottom clears the tray walls with
# 3.5 cm margin.
TRANSPORT_MIN_Z = TABLE_TOP_Z + TRAY_WALL_TOP_DZ + CUBE_HALF_SIZE + 0.035
# Terminate when the cube falls well below the table top.
CUBE_DROPPED_Z = TABLE_TOP_Z - 0.105
# Transport target is a hover point above the tray, not the resting spot: a
# Euclidean-distance kernel toward a point behind the wall would drag the cube
# straight into it. The descent into the tray is paid by place/success.
TRAY_TARGET_OFFSET = (0.0, 0.0, 0.10)
TRAY_XY_TOL = 0.05
# Cube center within 2.5 cm above its in-tray resting height = "in tray".
TRAY_MAX_HEIGHT = CUBE_TRAY_REST_DZ + 0.025
# Success requires the cube at (or within 1.5 cm of) its in-tray resting
# height, so holding it higher inside the tray does not count.
SUCCESS_MAX_HEIGHT = CUBE_TRAY_REST_DZ + 0.015
SETTLE_MAX_SPEED = 0.05
# The linear cap misses free spin about the contact normal (zero COM motion;
# torsional friction is randomized down to 1e-4, so a pinched cube can spin
# indefinitely). 1.0 rad/s is below the ~1.8 rad/s that edge-rocking already
# admits under the linear cap (0.05 m/s over the 28 mm corner radius), so it
# binds only for spin.
SETTLE_MAX_ANG_SPEED = 1.0

# Grasp-center site between the fingertips (added in the robot module), so
# reach/observation terms target where a grasped cube sits, not the wrist.
EE_SITE = LEFT_GRASP_SITE


def get_cube_spec() -> mujoco.MjSpec:
    """Orange cube identical to demo.xml."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint()
    geom = body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE),
        mass=0.05,
        rgba=(1.0, 0.4, 0.0, 1.0),
    )
    geom.friction = (1.0, 0.5, 0.01)
    return spec


_TRAY_FLOOR_HALF = TRAY_FLOOR_TOP_DZ / 2
_TRAY_WALL_HALF = TRAY_WALL_TOP_DZ / 2
_TRAY_GEOMS = (
    ("tray_bottom", (0.0, 0.0, _TRAY_FLOOR_HALF), (0.07, 0.07, _TRAY_FLOOR_HALF)),
    ("tray_wall_px", (0.065, 0.0, _TRAY_WALL_HALF), (0.005, 0.07, _TRAY_WALL_HALF)),
    ("tray_wall_nx", (-0.065, 0.0, _TRAY_WALL_HALF), (0.005, 0.07, _TRAY_WALL_HALF)),
    ("tray_wall_py", (0.0, 0.065, _TRAY_WALL_HALF), (0.06, 0.005, _TRAY_WALL_HALF)),
    ("tray_wall_ny", (0.0, -0.065, _TRAY_WALL_HALF), (0.06, 0.005, _TRAY_WALL_HALF)),
)


def get_tray_spec() -> mujoco.MjSpec:
    """Black tray identical to demo.xml's black_frame."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="tray")
    for name, pos, size in _TRAY_GEOMS:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=(0.1, 0.1, 0.1, 1.0),
        )
    return spec


def openarm_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm pick & place environment config."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "ee_to_cube": ObservationTermCfg(
            func=manipulation_mdp.ee_to_object_distance,
            params={
                "object_name": "cube",
                "asset_cfg": SceneEntityCfg("robot", site_names=(EE_SITE,)),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "cube_to_tray": ObservationTermCfg(
            func=pick_mdp.object_to_target_offset,
            params={
                "object_name": "cube",
                "target_name": "tray",
                "target_offset": TRAY_TARGET_OFFSET,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = {**actor_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(
                "openarm_left_joint[1-7]",
                "openarm_left_finger_joint1",
            ),
            # use_default_offset puts each offset at the home pose, and the
            # asset's home jaw is fully OPEN at the top of the finger range.
            # A finger action therefore only has room to close, so its scale
            # spans the whole range: action -1 shuts the jaw, 0 holds it open.
            # A smaller scale cannot reach a grasp at all — the 4 cm cube needs
            # the jaw below ~0.16 rad, which 0.4 never reaches from 0.7854.
            scale={
                "openarm_left_joint[1-7]": 0.5,
                "openarm_left_finger_joint1": LEFT_FINGER_HOME,
            },
            use_default_offset=True,
        )
    }

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=("openarm_left_.*",)),
            },
        ),
        "reset_cube": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                # Offsets relative to CUBE_SPAWN_POS. The y upper bound keeps the
                # cube clear of the tray wall at y=0.08.
                "pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.075, 0.04),
                    "yaw": (-math.pi, math.pi),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "fingertip_friction_slide": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=LEFT_FINGERTIP_GEOMS),
                "operation": "abs",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.3, 1.5),
            },
        ),
        "fingertip_friction_spin": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=LEFT_FINGERTIP_GEOMS),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [1],
                "ranges": (1e-4, 2e-2),
            },
        ),
        "fingertip_friction_roll": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=LEFT_FINGERTIP_GEOMS),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [2],
                "ranges": (1e-5, 5e-3),
            },
        ),
    }

    rewards = {
        "reach": RewardTermCfg(
            func=pick_mdp.object_reach_reward,
            weight=1.0,
            params={
                "object_name": "cube",
                "std": 0.2,
                "asset_cfg": SceneEntityCfg("robot", site_names=(EE_SITE,)),
            },
        ),
        "lift": RewardTermCfg(
            func=pick_mdp.object_lifted,
            weight=1.0,
            params={"object_name": "cube", "minimum_height": LIFT_MIN_Z},
        ),
        # Weight/std chosen so carrying spawn->tray gains ~+0.9/step over holding
        # in place (the reach+lift baseline is 2.0/step); at 1.0/0.3 the gain was
        # a negligible +0.2/step.
        "transport": RewardTermCfg(
            func=pick_mdp.object_transport_reward,
            weight=2.0,
            params={
                "object_name": "cube",
                "target_name": "tray",
                "std": 0.2,
                "minimum_height": TRANSPORT_MIN_Z,
                "target_offset": TRAY_TARGET_OFFSET,
            },
        ),
        "place": RewardTermCfg(
            func=pick_mdp.object_in_tray,
            weight=2.0,
            params={
                "object_name": "cube",
                "tray_name": "tray",
                "xy_tolerance": TRAY_XY_TOL,
                "max_height_above_tray": TRAY_MAX_HEIGHT,
            },
        ),
        # Terminal bonus (eff. 20 after dt scaling: 1000 * step_dt 0.02) must
        # exceed the discounted return of hovering in-tray without settling
        # (~5.9 raw/step * 0.02 * gamma/(1-gamma) ~= 11.7 at gamma=0.99), or the
        # policy learns to stall just above the success thresholds forever.
        "success_bonus": RewardTermCfg(
            func=pick_mdp.terminated_by,
            weight=1000.0,
            params={"term_name": "success"},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            # Arm joints only: the fingers' whole range [0, 0.7854] is
            # functional (qpos 0.7854 = fully open is the home posture the
            # episode starts from, qpos 0 = fully closed), so a soft-limit
            # band would tax normal open/close postures.
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("openarm_left_joint[1-7]",)
                )
            },
        ),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.01,
            params={
                "max_vel": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # mjwarp's GPU solver very rarely produces NaN in one step (observed ~1
        # env per ~4e7 world-steps at a stiff multi-contact jam); terminate and
        # reset that env instead of crashing the whole run in rsl_rl's check_nan.
        "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
        "cube_dropped": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": CUBE_DROPPED_Z,
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "success": TerminationTermCfg(
            func=pick_mdp.object_settled_in_tray,
            params={
                "object_name": "cube",
                "tray_name": "tray",
                "xy_tolerance": TRAY_XY_TOL,
                "max_height_above_tray": SUCCESS_MAX_HEIGHT,
                "max_speed": SETTLE_MAX_SPEED,
                "max_ang_speed": SETTLE_MAX_ANG_SPEED,
            },
        ),
    }

    curriculum = {
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=manipulation_mdp.reward_curriculum,
            params={
                "reward_name": "joint_vel_hinge",
                "stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 500 * 24, "weight": -0.1},
                    {"step": 1000 * 24, "weight": -1.0},
                ],
            },
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1,
            env_spacing=1.0,
            entities={
                "robot": get_openarm_robot_cfg(),
                "cube": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(pos=CUBE_SPAWN_POS),
                    spec_fn=get_cube_spec,
                ),
                "tray": EntityCfg(
                    init_state=EntityCfg.InitialStateCfg(pos=TRAY_POS),
                    spec_fn=get_tray_spec,
                ),
            },
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            # Fixed view matching cell.xml's camera_head_right: above the arm
            # bases, looking forward-down (+x) at the table workspace.
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.46, 0.03, 1.01),
            distance=0.7,
            elevation=-62.0,
            azimuth=0.0,
            fovy=52.0,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=800,
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=10,  # 50 Hz control.
        episode_length_s=10.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg
