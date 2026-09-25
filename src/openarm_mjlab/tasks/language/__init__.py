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

"""Language-conditioned manipulation: the target is named, not observed."""

from mjlab.tasks.registry import register_mjlab_task

from openarm_mjlab.tasks.language.env_cfg import (
    openarm_language_puck_env_cfg,
    openarm_language_puck_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="OpenArm-Puck-Language",
    env_cfg=openarm_language_puck_env_cfg(),
    play_env_cfg=openarm_language_puck_env_cfg(play=True),
    rl_cfg=openarm_language_puck_ppo_runner_cfg(),
)
