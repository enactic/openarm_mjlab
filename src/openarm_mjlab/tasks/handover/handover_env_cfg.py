import mujoco

from mjlab.entity import EntityCfg
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

from ..bimanual_lift.bimanual_lift_env_cfg import (
    BIMANUAL_ARM_SCALE,
    LEFT_SQUEEZE_SCALE,
    RIGHT_SQUEEZE_SCALE,
    ROBOT_EE_LEFT_CFG,
    ROBOT_EE_RIGHT_CFG,
    get_bimanual_lift_robot_cfg,
    get_bimanual_table_spec,
)
from . import mdp as ho

BLOCK_CFG = SceneEntityCfg("block")
SENSOR_GIVER = "finger_block_contact_giver"  # left arm
SENSOR_RECV = "finger_block_contact_recv"  # right arm

_FIELDS = ("found", "force", "dist", "pos")
FINGER_BLOCK_SENSOR_GIVER = ContactSensorCfg(
    name=SENSOR_GIVER,
    primary=ContactMatch(
        mode="body", pattern=r"openarm_left_ee_(inner|outer)_finger", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="block_geom", entity="block"),
    fields=_FIELDS,
    reduce="none",
    num_slots=1,
)
FINGER_BLOCK_SENSOR_RECV = ContactSensorCfg(
    name=SENSOR_RECV,
    primary=ContactMatch(
        mode="body", pattern=r"openarm_right_ee_(inner|outer)_finger", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="block_geom", entity="block"),
    fields=_FIELDS,
    reduce="none",
    num_slots=1,
)


def get_block_spec() -> mujoco.MjSpec:
    """Single free 50 mm block, lift's proven grip physics (priority 2, condim 4)."""
    spec = mujoco.MjSpec()
    spec.modelname = "handover_block"
    spec.compiler.degree = False
    body = spec.worldbody.add_body(name="block")
    body.add_freejoint(name="block_free")
    body.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.025, 0.025, 0.03),
        mass=0.05,
        friction=(2.5, 0.1, 0.01),
        priority=2,
        condim=4,
        solref=(0.005, 1.0),
        rgba=(0.9, 0.5, 0.1, 1.0),
    )
    return spec


def openarm_handover_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm handover environment config."""
    noise01 = Unoise(n_min=-0.01, n_max=0.01)
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, noise=noise01),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "tool_to_block_giver": ObservationTermCfg(
            func=ho.tool_to_block_obs,
            params={"robot_cfg": ROBOT_EE_LEFT_CFG, "asset_cfg": BLOCK_CFG},
            noise=noise01,
        ),
        "tool_to_block_recv": ObservationTermCfg(
            func=ho.tool_to_block_obs,
            params={"robot_cfg": ROBOT_EE_RIGHT_CFG, "asset_cfg": BLOCK_CFG},
            noise=noise01,
        ),
        "block_to_target": ObservationTermCfg(
            func=ho.block_to_target_obs, params={"asset_cfg": BLOCK_CFG}, noise=noise01
        ),
        "pinch_giver": ObservationTermCfg(
            func=ho.pinch_obs, params={"sensor_name": SENSOR_GIVER}
        ),
        "pinch_recv": ObservationTermCfg(
            func=ho.pinch_obs, params={"sensor_name": SENSOR_RECV}
        ),
        "transferred": ObservationTermCfg(func=ho.transferred_obs),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(dict(actor_terms), enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "right_arm": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("openarm_right_joint[1-7]",),
            scale=BIMANUAL_ARM_SCALE,
            use_default_offset=True,
        ),
        "left_arm": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("openarm_left_joint[1-7]",),
            scale=BIMANUAL_ARM_SCALE,
            use_default_offset=True,
        ),
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
        

        "reset_block": EventTermCfg(
            func=ho.reset_block_uniform,
            mode="reset",
            params={"asset_cfg": BLOCK_CFG, "xy_range": 0.03},
        ),
        "reset_giver_held_high": EventTermCfg(
            func=ho.reset_giver_held_high,
            mode="reset",
            params={
                "asset_cfg": BLOCK_CFG,
                "robot_joints_cfg": SceneEntityCfg("robot"),
                "probability": 0.5,
            },
        ),
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

    # NOTE: dict order matters -- overlap_streak updates the `transferred` latch
    # and must precede release/carry.
    rewards = {
        "reach_giver": RewardTermCfg(
            func=ho.reach_block_reward, weight=0.5,
            params={"std": 0.2, "robot_cfg": ROBOT_EE_LEFT_CFG, "asset_cfg": BLOCK_CFG},
        ),
        "reach_fine_giver": RewardTermCfg(
            func=ho.reach_block_reward, weight=0.2,
            params={"std": 0.02, "robot_cfg": ROBOT_EE_LEFT_CFG, "asset_cfg": BLOCK_CFG},
        ),
        "reach_recv": RewardTermCfg(
            func=ho.reach_block_reward, weight=0.7,
            params={"std": 0.2, "robot_cfg": ROBOT_EE_RIGHT_CFG, "asset_cfg": BLOCK_CFG},
        ),
        "reach_fine_recv": RewardTermCfg(
            func=ho.reach_block_reward, weight=0.3,
            params={"std": 0.02, "robot_cfg": ROBOT_EE_RIGHT_CFG, "asset_cfg": BLOCK_CFG},
        ),
        "pinch_giver": RewardTermCfg(
            func=ho.pinch_reward, weight=0.5, params={"sensor_name": SENSOR_GIVER}
        ),
        "pinch_recv": RewardTermCfg(
            func=ho.pinch_reward, weight=0.5, params={"sensor_name": SENSOR_RECV}
        ),
        "partial_pinch_giver": RewardTermCfg(
            func=ho.partial_pinch_reward, weight=0.3, params={"sensor_name": SENSOR_GIVER}
        ),
        "partial_pinch_recv": RewardTermCfg(
            func=ho.partial_pinch_reward, weight=0.3, params={"sensor_name": SENSOR_RECV}
        ),
        "giver_lift": RewardTermCfg(
            func=ho.giver_lift_reward, weight=2.0,
            params={"sensor_giver": SENSOR_GIVER, "asset_cfg": BLOCK_CFG},
        ),
        "overlap_streak": RewardTermCfg(
            func=ho.handover_overlap_streak, weight=1.5,
            params={"sensor_giver": SENSOR_GIVER, "sensor_recv": SENSOR_RECV},
        ),
        "giver_release": RewardTermCfg(
            func=ho.giver_release_reward, weight=1.0,
            params={"sensor_giver": SENSOR_GIVER, "sensor_recv": SENSOR_RECV},
        ),
        "carry_to_target": RewardTermCfg(
            func=ho.carry_to_target_reward, weight=3.0,
            params={"sensor_recv": SENSOR_RECV, "asset_cfg": BLOCK_CFG},
        ),
        "success": RewardTermCfg(
            func=ho.terminated_by, weight=6200.0, params={"term_name": "handed_over"}
        ),
        "descent": RewardTermCfg(
            func=ho.descent_penalty, weight=-2.0, params={"asset_cfg": BLOCK_CFG}
        ),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
        "joint_vel_hinge": RewardTermCfg(
            func=manipulation_mdp.joint_velocity_hinge_penalty,
            weight=-0.05,
            params={
                "max_vel": 1.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=("openarm_(left|right)_.*",)),
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
        "handed_over": TerminationTermCfg(
            func=ho.handed_over,
            params={
                "sensor_giver": SENSOR_GIVER,
                "sensor_recv": SENSOR_RECV,
                "asset_cfg": BLOCK_CFG,
            },
        ),
        "block_fell": TerminationTermCfg(
            func=ho.block_fell, params={"asset_cfg": BLOCK_CFG}
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                "robot": get_bimanual_lift_robot_cfg(),
                "table": EntityCfg(spec_fn=get_bimanual_table_spec),
                "block": EntityCfg(
                    spec_fn=get_block_spec,
                    init_state=EntityCfg.InitialStateCfg(pos=ho.BLOCK_START),
                ),
            },
            num_envs=1,
            env_spacing=2.5,
            sensors=(FINGER_BLOCK_SENSOR_GIVER, FINGER_BLOCK_SENSOR_RECV),
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
            nconmax=200,
            njmax=1400,
            mujoco=MujocoCfg(
                timestep=0.005, iterations=10, ls_iterations=20,
                impratio=10, cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=10.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg