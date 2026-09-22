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

"""OpenArm bimanual bar-lifting task."""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .bimanual_lift_env_cfg import openarm_bimanual_lift_env_cfg
from .rl_cfg import openarm_bimanual_lift_ppo_runner_cfg

register_mjlab_task(
    task_id="OpenArm-BimanualLift",
    env_cfg=openarm_bimanual_lift_env_cfg(),
    play_env_cfg=openarm_bimanual_lift_env_cfg(play=True),
    rl_cfg=openarm_bimanual_lift_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
)
