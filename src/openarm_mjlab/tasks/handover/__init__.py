"""OpenArm bimanual handover task (left arm gives, right arm receives)."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .handover_env_cfg import openarm_handover_env_cfg
from .rl_cfg import openarm_handover_ppo_runner_cfg

register_mjlab_task(
    task_id="OpenArm-Handover",
    env_cfg=openarm_handover_env_cfg(),
    play_env_cfg=openarm_handover_env_cfg(play=True),
    rl_cfg=openarm_handover_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)