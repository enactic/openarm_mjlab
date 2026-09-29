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

"""PPO runner configuration for the OpenArm pick & place task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...rl_cfg import ppo_runner_cfg


def openarm_pick_place_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO runner config for the OpenArm pick & place task."""
    return ppo_runner_cfg(
        "openarm_pick_place", max_iterations=10_000, entropy_coef=0.005
    )
