
from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...rl_cfg import ppo_runner_cfg


def openarm_handover_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Build the PPO runner config for the handover task."""
    return ppo_runner_cfg("openarm_handover")
