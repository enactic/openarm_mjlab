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

"""Procedurally generated task families.

A *family* is one task type whose scene geometry is parameterised, so a single
hand-written task yields many instances. Instances of a family share one
observation layout, one action space and one policy head, and carry NO
per-instance identity signal: the policy has to read the geometry out of the
observations it already receives. That is what makes a held-out instance a real
generalisation test rather than a lookup -- an identity one-hot would leave a
held-out instance with no head to route to, and memorising N instances would
score as well as understanding them.
"""

from mjlab.tasks.registry import register_mjlab_task
from openarm_mjlab.tasks.door.rl_cfg import openarm_door_ppo_runner_cfg
from openarm_mjlab.tasks.families.door_family import door_family_env_cfg
from openarm_mjlab.tasks.families.door_split import (
    assert_split_is_honest as assert_door_split_is_honest,
)
from openarm_mjlab.tasks.families.door_split import build_split as build_door_split
from openarm_mjlab.tasks.families.valve_family import valve_family_env_cfg
from openarm_mjlab.tasks.families.valve_split import (
    assert_split_is_honest as assert_valve_split_is_honest,
)
from openarm_mjlab.tasks.families.valve_split import build_split as build_valve_split
from openarm_mjlab.tasks.valve.rl_cfg import openarm_valve_ppo_runner_cfg

N_TRAIN = 8
N_HELD_OUT = 8

VALVE_SPLIT = build_valve_split(
    n_train=N_TRAIN, n_interp=N_HELD_OUT, n_extrap=N_HELD_OUT, seed=0
)
assert_valve_split_is_honest(VALVE_SPLIT)
DOOR_SPLIT = build_door_split(
    n_train=N_TRAIN, n_interp=N_HELD_OUT, n_extrap=N_HELD_OUT, seed=0
)
assert_door_split_is_honest(DOOR_SPLIT)

VALVE_INSTANCE_IDS: dict[str, list[str]] = {"train": [], "interp": [], "extrap": []}
DOOR_INSTANCE_IDS: dict[str, list[str]] = {"train": [], "interp": [], "extrap": []}


def _register(prefix, split, ids, builder, rl_cfg):
    for group, instances in split.items():
        for params in instances:
            task_id = f"{prefix}-{group}-{params.name()}"
            ids[group].append(task_id)
            register_mjlab_task(
                task_id=task_id,
                env_cfg=builder(params),
                play_env_cfg=builder(params, play=True),
                rl_cfg=rl_cfg(),
            )


_register(
    "OpenArm-ValveFamily",
    VALVE_SPLIT,
    VALVE_INSTANCE_IDS,
    valve_family_env_cfg,
    openarm_valve_ppo_runner_cfg,
)
_register(
    "OpenArm-DoorFamily",
    DOOR_SPLIT,
    DOOR_INSTANCE_IDS,
    door_family_env_cfg,
    openarm_door_ppo_runner_cfg,
)
