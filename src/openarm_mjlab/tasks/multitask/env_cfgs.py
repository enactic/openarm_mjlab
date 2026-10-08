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

"""MultiTask variants of the five tasks that share one policy.

Each variant takes its task's own, unmodified ``*_env_cfg`` builder and
rewrites only the observation groups: the task's object block is placed in a
reserved slot, the remaining slots are zero-filled, and a task-identity
one-hot is appended. Rewards, terminations and actions are untouched, so each
task keeps exactly the behaviour it was solved with standalone.

Per-task object-block widths were read off each built observation manager
rather than assumed: reach 3 (tool_to_target), valve/door/drawer 5
(angle + ee_to_X + contact), puck 7 (tool_to_puck + puck_to_goal +
push_contact).
"""

from itertools import accumulate

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from openarm_mjlab.tasks.door.door_env_cfg import openarm_door_env_cfg
from openarm_mjlab.tasks.drawer.drawer_env_cfg import openarm_drawer_env_cfg
from openarm_mjlab.tasks.multitask import mdp
from openarm_mjlab.tasks.puck.puck_env_cfg import openarm_puck_env_cfg
from openarm_mjlab.tasks.reach.reach_env_cfg import openarm_reach_env_cfg
from openarm_mjlab.tasks.valve.valve_env_cfg import openarm_valve_env_cfg

_OBJECT_WIDTH = {
    "reach": 3,
    "valve": 5,
    "door": 5,
    "drawer": 5,
    "puck": 7,
}

# Disjoint per-task slots. Sharing one padded block across tasks put unrelated
# quantities on the same index -- valve_angle for valve, door_angle for door,
# drawer_pos for drawer, a Cartesian offset for reach and puck. rsl_rl keeps
# ONE running mean/std per index pooled over every task's samples, so those
# statistics described a mixture of unrelated signals, and an index that was a
# structural zero for most tasks had its std collapse toward zero and amplify
# whichever task did use it (measured: index 41 std 0.006, a 4.3 sigma
# distortion for reach and 7.2 sigma for puck against inputs of order 1).
#
# Reserving a range per task makes every index carry exactly one quantity, for
# 18 extra observation dimensions.
_SLOT_OFFSET = dict(
    zip(
        mdp.TASK_NAMES,
        accumulate((_OBJECT_WIDTH[name] for name in mdp.TASK_NAMES), initial=0),
    )
)
TOTAL_OBJECT_WIDTH = sum(_OBJECT_WIDTH.values())


def _unify(cfg: ManagerBasedRlEnvCfg, task_name: str) -> ManagerBasedRlEnvCfg:
    """Rewrite one task's observation groups into the shared layout.

    The padding is inserted immediately after the variable-width object block
    and BEFORE ``actions``, so ``actions`` lands at the same column for every
    task. Appending it at the end instead put ``actions`` at a different raw
    index per task (reach 39:47, valve/door/drawer 41:49, puck 43:51), which
    left a single normalizer statistic fitted against up to three different
    action components; the std in that range had ballooned to 5-15 against
    0.1-0.5 in the correctly aligned joint region, crushing the network's own
    last-action feedback after normalization. Only the task-id one-hot goes at
    the very end.
    """
    index = mdp.TASK_NAMES.index(task_name)
    before = _SLOT_OFFSET[task_name]
    after = TOTAL_OBJECT_WIDTH - before - _OBJECT_WIDTH[task_name]
    for group_name in ("actor", "critic"):
        terms = cfg.observations[group_name].terms
        items = list(terms.items())
        last_key, last_term = items[-1]
        if last_key != "actions":
            raise ValueError(
                f"expected 'actions' to be the last pre-unification term for "
                f"{task_name}/{group_name}, got {last_key!r}: the fixed-offset "
                f"layout below no longer holds"
            )
        leading = [key for key, _ in items[:2]]
        if leading != ["joint_pos", "joint_vel"]:
            raise ValueError(
                f"expected joint_pos/joint_vel to lead {task_name}/{group_name}, "
                f"got {leading}: the object block is items[2:-1], which the "
                f"disjoint-slot layout depends on"
            )
        terms.clear()
        for key, term in items[:2]:
            terms[key] = term
        if before > 0:
            terms["mt_slot_before"] = ObservationTermCfg(
                func=mdp.zero_pad, params={"width": before}
            )
        for key, term in items[2:-1]:
            terms[key] = term
        if after > 0:
            terms["mt_slot_after"] = ObservationTermCfg(
                func=mdp.zero_pad, params={"width": after}
            )
        terms[last_key] = last_term
        terms["mt_task_id"] = ObservationTermCfg(
            func=mdp.task_id_onehot, params={"index": index}
        )
    return cfg


def multitask_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Return the reach task in the shared multi-task observation layout."""
    return _unify(openarm_reach_env_cfg(play=play), "reach")


def multitask_valve_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Return the valve task in the shared multi-task observation layout."""
    return _unify(openarm_valve_env_cfg(play=play), "valve")


def multitask_door_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Return the door task in the shared multi-task observation layout."""
    return _unify(openarm_door_env_cfg(play=play), "door")


def multitask_drawer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Return the drawer task in the shared multi-task observation layout."""
    return _unify(openarm_drawer_env_cfg(play=play), "drawer")


def multitask_puck_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Return the puck task in the shared multi-task observation layout."""
    return _unify(openarm_puck_env_cfg(play=play), "puck")
