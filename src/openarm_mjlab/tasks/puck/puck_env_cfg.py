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

"""Environment configuration for the OpenArm move-puck task.

A 70x70x44mm puck is pushed into a goal disc and left there. Pushing
needs no grasp: fully feasible contact-only.
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
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
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
from . import mdp as puck_mdp

PUCK_START = (0.20, -0.14, 0.422)


def get_table_spec() -> mujoco.MjSpec:
    """Build the static table slab + goal disc marker spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "puck_table"
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
    body.add_site(
        name="goal_site",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(0.05, 0.001, 0),
        pos=(puck_mdp.GOAL_LOCAL[0], puck_mdp.GOAL_LOCAL[1], 0.401),
        rgba=(0.2, 0.8, 0.3, 0.4),
    )
    return spec


def get_puck_spec() -> mujoco.MjSpec:
    """Build the free-body puck fixture spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "puck"
    spec.compiler.degree = False
    body = spec.worldbody.add_body(name="puck")
    body.add_freejoint(name="puck_free")
    body.add_geom(
        name="puck_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.035, 0.035, 0.022),
        mass=0.2,
        friction=(0.4, 0.01, 0.01),
        rgba=(0.85, 0.4, 0.1, 1.0),
        # Without a priority this friction never applied anywhere, so the
        # puck was declared slippery and never was. MuJoCo takes the
        # higher-priority geom's contact parameters and mixes only when
        # priorities are equal, and that mix is an elementwise MAX. The
        # right-arm finger geoms are priority=1, so they governed the push
        # contact at mu=1.0; the table is priority=0 like the puck was, so
        # that pair mixed to max(0.4, 1.0) = 1.0 as well. dr_puck_friction
        # was randomizing a number MuJoCo read on neither pair. Outranking
        # both makes the declared 0.4 govern (measured: 0.4 at the finger
        # and 0.4 at the table). condim=3 matches the class the rest of the
        # robot's collision geoms use, so priority hands the puck only the
        # friction, never a wider contact model: the torsional axis is not
        # solved at all, which is why dr_puck_friction randomizes axes=[0]
        # alone. A flat puck resting on the table is unaffected either way
        # (its four corner contacts resist twist through sliding friction);
        # what condim=3 gives up is twist resistance under a single-point
        # fingertip press, which a pushing task does not rely on.
        priority=2,
        condim=3,
        solref=(0.005, 1.0),
    )
    return spec


ROBOT_EE_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_RIGHT,))
PUCK_CFG = SceneEntityCfg("puck")

FINGER_PUCK_SENSOR = ContactSensorCfg(
    name="finger_puck_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern="openarm_right_ee_base_link",
        entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="puck_geom", entity="puck"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)

# World-fixed overhead depth camera. A MuJoCo camera looks down its own
# -Z by convention, so the identity quaternion here means straight down.
PUCK_TABLECAM_POS = (0.26, -0.18, 1.15)
PUCK_TABLECAM_QUAT = (1.0, 0.0, 0.0, 0.0)
PUCK_TABLECAM_FOVY = 60.0


def openarm_puck_env_cfg(
    play: bool = False, vision: bool = False
) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm move-puck environment config.

    With ``vision=True``, the actor loses the privileged puck-position
    and contact observation terms and instead relies on a fixed overhead
    depth camera; the critic (discarded at deployment) keeps full
    privileged state and also gets the camera, the same asymmetric
    actor-critic pattern mjlab's own vision reference task uses.
    """
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "tool_to_puck": ObservationTermCfg(
            func=puck_mdp.tool_to_object_obs,
            params={"robot_cfg": ROBOT_EE_CFG, "asset_cfg": PUCK_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "puck_to_goal": ObservationTermCfg(
            func=puck_mdp.puck_to_goal_obs,
            params={"asset_cfg": PUCK_CFG, "robot_cfg": ROBOT_EE_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "push_contact": ObservationTermCfg(
            func=puck_mdp.fingers_on_handle_obs,
            params={"sensor_name": "finger_puck_contact"},
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
        "reset_puck": EventTermCfg(
            func=puck_mdp.reset_puck_uniform,
            mode="reset",
            params={"asset_cfg": PUCK_CFG, "xy_range": 0.03},
        ),
        # Domain randomization, same verified pattern as the other tasks.
        # Puck is a free body (no joint dynamics axis, like reach) --
        # friction, mass, and arm gains only.
        "dr_puck_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("puck", geom_names=("puck_geom",)),
                "operation": "scale",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.4),
            },
        ),
        "dr_puck_mass": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("puck", body_names=("puck",)),
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
        "reach_puck": RewardTermCfg(
            func=puck_mdp.reach_object_reward,
            weight=1.0,
            params={"std": 0.2, "robot_cfg": ROBOT_EE_CFG, "asset_cfg": PUCK_CFG},
        ),
        "push_rate": RewardTermCfg(
            func=puck_mdp.push_rate_reward,
            weight=3.0,
            params={"sensor_name": "finger_puck_contact", "asset_cfg": PUCK_CFG},
        ),
        "goal_fine": RewardTermCfg(
            func=puck_mdp.goal_fine_reward, weight=2.0, params={"asset_cfg": PUCK_CFG}
        ),
        "at_goal": RewardTermCfg(
            func=puck_mdp.at_goal_reward, weight=2.0, params={"asset_cfg": PUCK_CFG}
        ),
        "success": RewardTermCfg(
            func=puck_mdp.terminated_by,
            weight=800.0,
            params={"term_name": "puck_at_goal"},
        ),
        "puck_overspeed": RewardTermCfg(
            func=puck_mdp.puck_overspeed_penalty,
            weight=-2.0,
            params={"asset_cfg": PUCK_CFG},
        ),
        # Direct negative counterpart to "success" (half its magnitude):
        # without this, a fall was only punished implicitly by losing
        # future dense income, too weak a deterrent once training drifts
        # toward a faster, more aggressive push.
        "puck_fell": RewardTermCfg(
            func=puck_mdp.terminated_by,
            weight=-400.0,
            params={"term_name": "puck_fell"},
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
        "puck_at_goal": TerminationTermCfg(
            func=puck_mdp.puck_at_goal, params={"asset_cfg": PUCK_CFG}
        ),
        "puck_fell": TerminationTermCfg(
            func=base_mdp.root_height_below_minimum,
            # Table top is z=0.40; anything this low has left the table.
            params={"minimum_height": 0.30, "asset_cfg": PUCK_CFG},
        ),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_bimanual_robot_cfg(),
                "table": EntityCfg(spec_fn=get_table_spec),
                "puck": EntityCfg(
                    spec_fn=get_puck_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=PUCK_START),
                ),
            },
            num_envs=1,
            env_spacing=2.5,
            sensors=(FINGER_PUCK_SENSOR,),
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
            lookat=(0.28, -0.22, 0.50),
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
    if vision:
        # Depth-only (matches mjlab's own vision reference task's
        # simplest variant): avoids RGB/lighting domain-randomization
        # complexity for a first vision attempt. The camera group is
        # separate from "actor"/"critic" so rl_cfg's obs_groups controls
        # which policy sees it.
        cam_cfg = CameraSensorCfg(
            name="tablecam",
            pos=PUCK_TABLECAM_POS,
            quat=PUCK_TABLECAM_QUAT,
            fovy=PUCK_TABLECAM_FOVY,
            width=64,
            height=64,
            data_types=("depth",),
            use_shadows=False,
            use_textures=True,
        )
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
        cfg.observations["camera"] = ObservationGroupCfg(
            terms={
                "tablecam_depth": ObservationTermCfg(
                    func=manipulation_mdp.camera_depth,
                    params={"sensor_name": "tablecam", "cutoff_distance": 1.0},
                ),
            },
            enable_corruption=False,
            concatenate_terms=True,
        )
        # The puck's goal is a fixed constant (not resampled per
        # episode), so no separate goal-conditioning term is needed --
        # the network can learn it directly.
        actor_obs = cfg.observations["actor"]
        actor_obs.terms.pop("tool_to_puck")
        actor_obs.terms.pop("puck_to_goal")
        # The contact flag comes from a simulator contact sensor that the
        # real gripper cannot produce, so it is privileged too: the vision
        # actor has to infer contact from the depth image. The critic keeps
        # it.
        actor_obs.terms.pop("push_contact")
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
