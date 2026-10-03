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

"""One shared policy across the reach, valve, door, drawer and puck tasks.

Each task is registered as its own ``OpenArm-MultiTask-*`` id so it can be
played and evaluated individually, but all five share one experiment name, one
observation layout and one set of weights. The lift task is excluded: its
effort-controlled finger gives it a genuinely different action space from the
8-dimensional joint-position action the other five share.
"""

from mjlab.tasks.registry import register_mjlab_task

from openarm_mjlab.tasks.multitask.env_cfgs import (
    multitask_door_env_cfg,
    multitask_drawer_env_cfg,
    multitask_puck_env_cfg,
    multitask_reach_env_cfg,
    multitask_valve_env_cfg,
)
from openarm_mjlab.tasks.multitask.rl_cfg import multitask_ppo_runner_cfg

for _task_id, _builder in (
    ("OpenArm-MultiTask-Reach", multitask_reach_env_cfg),
    ("OpenArm-MultiTask-Valve", multitask_valve_env_cfg),
    ("OpenArm-MultiTask-Door", multitask_door_env_cfg),
    ("OpenArm-MultiTask-Drawer", multitask_drawer_env_cfg),
    ("OpenArm-MultiTask-Puck", multitask_puck_env_cfg),
):
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_builder(),
        play_env_cfg=_builder(play=True),
        rl_cfg=multitask_ppo_runner_cfg(),
    )
