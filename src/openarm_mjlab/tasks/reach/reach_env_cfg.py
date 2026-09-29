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

"""Environment configuration for the OpenArm reach task."""

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
from . import mdp as reach_mdp

ROBOT_EE_CFG = SceneEntityCfg("robot", site_names=(EE_SITE_RIGHT,))


def openarm_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the OpenArm reach environment config."""
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "tool_to_target": ObservationTermCfg(
            func=reach_mdp.tool_to_target_obs,
            params={"robot_cfg": ROBOT_EE_CFG},
            noise=Unoise(n_min=-0.005, n_max=0.005),
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
        "resample_target": EventTermCfg(
            func=reach_mdp.resample_targets,
            mode="reset",
            params={},
        ),
        # Domain-randomization pass (arm actuator gains -- reach has no
        # manipulated object, so this is the only physically-meaningful
        # randomization axis): held up 100% clean success under +-15%
        # kp/kd, essentially unchanged from the nominal task.
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
        "reach_coarse": RewardTermCfg(
            func=reach_mdp.reach_reward,
            weight=1.0,
            params={"std": 0.2, "robot_cfg": ROBOT_EE_CFG},
        ),
        "reach_fine": RewardTermCfg(
            func=reach_mdp.reach_reward,
            weight=2.0,
            params={"std": 0.05, "robot_cfg": ROBOT_EE_CFG},
        ),
        "hold": RewardTermCfg(
            func=reach_mdp.hold_at_target_reward,
            weight=2.0,
            params={"robot_cfg": ROBOT_EE_CFG},
        ),
        # Success is a one-shot bonus (weight * dt) that ENDS the episode,
        # forfeiting all remaining dense reward. Hovering at the target
        # while avoiding the settle condition earns ~0.10/step (coarse +
        # fine + hold, dt-scaled), and because timeouts are value-
        # bootstrapped that stream is worth up to 0.10 / (1 - 0.99) = 10 in
        # discounted return. Terminating is therefore only optimal when
        # weight * 0.02 > 0.99 * 10, i.e. weight > ~500 -- so the earlier
        # 300 (= 6.0) sat below the bound and made hovering forever the
        # reward-optimal behaviour. 750 (= 15.0) clears it with 50% margin
        # while staying the same order as the per-episode dense return
        # (~30), so it does not drown the shaping gradient.
        "success": RewardTermCfg(
            func=reach_mdp.terminated_by,
            weight=750.0,
            params={"term_name": "reached"},
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
        "reached": TerminationTermCfg(
            func=reach_mdp.reached,
            params={"robot_cfg": ROBOT_EE_CFG},
        ),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_bimanual_robot_cfg()},
            num_envs=1,
            env_spacing=2.5,
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
            lookat=(0.35, -0.2, 0.55),
            distance=1.6,
            elevation=-20.0,
            azimuth=220.0,
        ),
        sim=SimulationCfg(
            nconmax=100,
            njmax=700,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=6.0,
    )
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    return cfg
