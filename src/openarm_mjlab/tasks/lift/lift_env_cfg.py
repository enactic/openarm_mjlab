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

"""Environment configuration for the OpenArm lift task.

Squeeze-grip a 50mm block and raise it 120mm, then hold. The block is
wider than the closed cage gap, so a genuine friction pinch exists
(contact-only feasibility, no special-cased grasp).
"""

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
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
from ...robot_bimanual import (
    BIMANUAL_ACTION_SCALE,
    EE_SITE_RIGHT,
    get_bimanual_robot_cfg,
    get_bimanual_spec,
)
from . import mdp as lift_mdp

# Force-controlled RIGHT finger: position-servo fingers make squeeze-and-
# raise a coordination cliff, but with direct effort control, holding a
# friction grip is a constant command. The left finger stays a position
# servo for the hold action. Damping-only PD stabilizes; the policy's
# effort action rides on top as feedforward.
LIFT_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # Held-high IK pose as the DEFAULT: as a reset offset (not a
        # separate action-offset target), zero-action HOLDS this pose,
        # and table episodes start with the arm above the block.
        "openarm_right_joint1": -0.2181,
        "openarm_right_joint2": 0.0461,
        "openarm_right_joint3": 0.1049,
        "openarm_right_joint4": 1.8194,
        "openarm_right_joint5": 0.0007,
        "openarm_right_joint6": -0.1009,
        "openarm_right_joint7": -0.0426,
        "openarm_right_finger_joint[12]": -0.25,
        "openarm_left_joint4": 1.5708,
        "openarm_left_joint[12356]": 0.0,
        "openarm_left_joint7": 0.0,
        "openarm_left_finger_joint[12]": 0.0,
    },
    joint_vel={".*": 0.0},
)


def get_lift_spec() -> mujoco.MjSpec:
    """Build the bimanual asset spec with the right finger's joint limit stiffened.

    A raw-effort-controlled joint can apply sustained torques the default
    MuJoCo joint-limit solref/solimp (a 0.02s time constant, fine for the
    position-controlled joints everywhere else, which self-limit via
    ctrlrange) is too soft to resist -- this is the only joint in the
    whole task family under raw effort control, so it's the only one
    where this default ever matters. Under a moderate, sub-saturating
    torque (what a well-controlled grip on a 50g block actually needs)
    the stiffened limit settles the joint essentially at its true range;
    only a sustained maximum-force command still overshoots.
    """
    spec = get_bimanual_spec()
    for j in spec.joints:
        if j.name == "openarm_right_finger_joint1":
            j.solref_limit = [0.004, 1.0]
            j.solimp_limit = [0.98, 0.999, 0.0002, 0.5, 2.0]
    return spec


def get_lift_robot_cfg() -> EntityCfg:
    """Return the bimanual robot config, homed and actuated for the lift task.

    The right finger is switched from a position actuator to a raw
    effort actuator (``IdealPdActuatorCfg`` with ``stiffness=0.0``) so
    the policy directly commands squeeze force; its effort limit (7 Nm)
    matches the joint's own designed capacity, not a borrowed number
    from a different actuator's position-servo forcerange.
    """
    cfg = get_bimanual_robot_cfg()
    cfg.init_state = LIFT_HOME
    cfg.spec_fn = get_lift_spec
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
        IdealPdActuatorCfg(
            target_names_expr=("openarm_right_finger_joint1",),
            stiffness=0.0,
            damping=2.0,
            effort_limit=7.0,
        ),
    )
    cfg.articulation = EntityArticulationInfoCfg(
        actuators=actuators, soft_joint_pos_limit_factor=0.9
    )
    return cfg


# The right finger is effort-controlled, so the position action's scale
# excludes it (and its scale key must too).
LIFT_ARM_SCALE = {k: v for k, v in BIMANUAL_ACTION_SCALE.items() if "finger" not in k}


def get_lift_table_spec() -> mujoco.MjSpec:
    """Build the static table slab spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "lift_table"
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


def get_block_spec() -> mujoco.MjSpec:
    """Build the free-body block fixture spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "block"
    spec.compiler.degree = False
    body = spec.worldbody.add_body(name="block")
    body.add_freejoint(name="block_free")
    body.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.025, 0.025, 0.03),
        mass=0.05,
        friction=(2.5, 0.1, 0.01),
        rgba=(0.9, 0.2, 0.2, 1.0),
        # This is a GRIP friction -- it exists so the fingers can hold the
        # block -- but without a priority it was inert at exactly the
        # contact it was written for. The right-arm finger geoms are
        # priority=1 and a default-priority geom is outranked, so the
        # fingers' mu=1.0 governed the grasp and the 2.5 applied only
        # against the table, where a high value makes the block harder to
        # slide: the opposite of the intent. priority=2 puts the block
        # above the fingers so 2.5 reaches the grasp (measured), leaving
        # the table pair as it already was. condim=4 and the fingers' own
        # solref keep the contact model as it was, so only friction changes.
        priority=2,
        condim=4,
        solref=(0.005, 1.0),
    )
    return spec


ROBOT_EE_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_RIGHT,))
BLOCK_CFG = SceneEntityCfg("block")

FINGER_BLOCK_SENSOR = ContactSensorCfg(
    name="finger_block_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"openarm_right_ee_(inner|outer)_finger",
        entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="block_geom", entity="block"),
    fields=("found",),
    reduce="none",
    num_slots=1,
)


def openarm_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm lift environment config."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "tool_to_block": ObservationTermCfg(
            func=lift_mdp.tool_to_block_obs,
            params={"robot_cfg": ROBOT_EE_CFG, "asset_cfg": BLOCK_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "pinch": ObservationTermCfg(
            func=lift_mdp.pinch_obs,
            params={"sensor_name": "finger_block_contact"},
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
            actuator_names=("openarm_right_joint[1-7]",),
            scale=LIFT_ARM_SCALE,
            use_default_offset=True,
        ),
        # A single-sigma exploration action (raw~1.0) at this scale
        # produces a well-behaved 2 Nm -- the regime where the joint
        # settles essentially at its true limit; only a rare 3.5+ sigma
        # excursion saturates the actuator at all.
        "squeeze": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=("openarm_right_finger_joint1",),
            scale=2.0,
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
        "reset_block": EventTermCfg(
            func=lift_mdp.reset_block_uniform,
            mode="reset",
            params={"asset_cfg": BLOCK_CFG, "xy_range": 0.03},
        ),
        # A held-high reference-state init, run AFTER reset_block
        # (overrides the subset it picks): with probability 0.15, the
        # episode starts with the block already gripped near the target
        # height. Table starts already get a dense reach_block gradient
        # on their own; keeping this probability low means the training
        # batch is overwhelmingly the hard grasp-from-table skill that
        # actually needs to be learned, while still bootstrapping what
        # "near target height" looks like for the terminal bonus.
        "reset_held_high": EventTermCfg(
            func=lift_mdp.reset_held_high,
            mode="reset",
            params={
                "asset_cfg": BLOCK_CFG,
                "robot_joints_cfg": SceneEntityCfg("robot"),
                "probability": 0.15,
            },
        ),
        # Domain randomization, same verified pattern as the other
        # tasks. Lift has no revolute/slide joint on the object (it's a
        # free block), so there's no joint-dynamics axis to randomize
        # here -- friction, mass, and arm gains only.
        "dr_block_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("block", geom_names=("block_geom",)),
                "operation": "scale",
                "distribution": "uniform",
                "axes": [0],
                "ranges": (0.6, 1.4),
            },
        ),
        "dr_block_mass": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("block", body_names=("block",)),
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
        "reach_block": RewardTermCfg(
            func=lift_mdp.reach_block_reward,
            weight=1.0,
            params={"std": 0.2, "robot_cfg": ROBOT_EE_CFG, "asset_cfg": BLOCK_CFG},
        ),
        "pinch": RewardTermCfg(
            func=lift_mdp.pinch_reward,
            weight=0.5,
            params={"sensor_name": "finger_block_contact"},
        ),
        "partial_pinch": RewardTermCfg(
            func=lift_mdp.partial_pinch_reward,
            weight=0.3,
            params={"sensor_name": "finger_block_contact"},
        ),
        "pinch_streak": RewardTermCfg(
            func=lift_mdp.pinch_streak_reward,
            weight=1.5,
            params={"sensor_name": "finger_block_contact"},
        ),
        "lift_rate": RewardTermCfg(
            func=lift_mdp.lift_rate_reward,
            weight=3.0,
            params={"sensor_name": "finger_block_contact", "asset_cfg": BLOCK_CFG},
        ),
        "held_high": RewardTermCfg(
            func=lift_mdp.held_high_reward,
            weight=2.0,
            params={"sensor_name": "finger_block_contact", "asset_cfg": BLOCK_CFG},
        ),
        # mjlab/RSL-RL's RewardManager scales every term by weight*dt
        # (dt=0.02s here). This one-shot bonus must outweigh the full
        # standing income from reach/pinch/partial_pinch/pinch_streak/
        # held_high while merely camping pinched near the target --
        # succeeding also ends the episode, forfeiting whatever income
        # remained, so the bonus must beat camping outright under any
        # strategy, not just on average. 2400 (48.0 real reward) is
        # comfortably above the measured ~42.4 camping-income ceiling.
        "success": RewardTermCfg(
            func=lift_mdp.lift_success_bonus, weight=2400.0, params={}
        ),
        "descent": RewardTermCfg(
            func=lift_mdp.block_descent_penalty,
            weight=-2.0,
            params={"asset_cfg": BLOCK_CFG},
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
        "lifted_target": TerminationTermCfg(
            func=lift_mdp.lifted_target,
            params={"sensor_name": "finger_block_contact", "asset_cfg": BLOCK_CFG},
        ),
        "block_fell": TerminationTermCfg(
            func=lift_mdp.block_fell, params={"asset_cfg": BLOCK_CFG}
        ),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_lift_robot_cfg(),
                "table": EntityCfg(spec_fn=get_lift_table_spec),
                "block": EntityCfg(
                    spec_fn=get_block_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=lift_mdp.BLOCK_START),
                ),
            },
            num_envs=1,
            env_spacing=2.5,
            sensors=(FINGER_BLOCK_SENSOR,),
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
            lookat=(0.32, -0.20, 0.55),
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
