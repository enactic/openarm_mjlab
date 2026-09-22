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

"""Tests for the procedurally generated task families.

Note: building an env compiles mujoco-warp CPU kernels; the first run can take
a few minutes.
"""

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg
from openarm_mjlab.tasks.families import (
    DOOR_INSTANCE_IDS,
    DOOR_SPLIT,
    VALVE_INSTANCE_IDS,
    VALVE_SPLIT,
)
from openarm_mjlab.tasks.families import door_split, valve_split
from openarm_mjlab.tasks.families.door_family import (
    NOMINAL as DOOR_NOMINAL,
)
from openarm_mjlab.tasks.families.door_family import (
    door_family_env_cfg,
)
from openarm_mjlab.tasks.families.valve_family import (
    NOMINAL as VALVE_NOMINAL,
)
from openarm_mjlab.tasks.families.valve_family import (
    valve_family_env_cfg,
)

FAMILIES = (
    ("valve", VALVE_SPLIT, VALVE_INSTANCE_IDS, valve_split, "lever_reach"),
    ("door", DOOR_SPLIT, DOOR_INSTANCE_IDS, door_split, "handle_span"),
)


@pytest.mark.parametrize("name,split,ids,module,axis", FAMILIES)
def test_every_instance_is_registered(name, split, ids, module, axis):
    registered = set(list_tasks())
    for group, task_ids in ids.items():
        assert task_ids, f"{name}/{group} registered nothing"
        for task_id in task_ids:
            assert task_id in registered, task_id


@pytest.mark.parametrize("name,split,ids,module,axis", FAMILIES)
def test_held_out_really_is_held_out(name, split, ids, module, axis):
    """Interp sits inside the trained span but on no trained value; extrap outside."""
    trained = [getattr(p, axis) for p in split["train"]]
    low, high = min(trained), max(trained)
    for params in split["interp"]:
        value = getattr(params, axis)
        assert low < value < high
        assert all(abs(value - t) > 1e-6 for t in trained)
    for params in split["extrap"]:
        assert getattr(params, axis) > high


@pytest.mark.parametrize("name,split,ids,module,axis", FAMILIES)
def test_honesty_check_rejects_a_broken_split(name, split, ids, module, axis):
    """The guard must actually fire; a check that cannot fail proves nothing."""
    broken = {k: list(v) for k, v in split.items()}
    broken["extrap"] = list(split["interp"])  # not beyond the trained span
    with pytest.raises(ValueError):
        module.assert_split_is_honest(broken)


def test_valve_nuisance_axes_are_stratified():
    """Valve stratifies its nuisance axes, so no axis may differ much by group.

    This guards a real defect: on an earlier, independently sampled valve split
    the lateral position y correlated -0.481 with success -- more strongly than
    lever_reach, the axis the split is built on -- and landed 12% of the
    interpolation group on the arm's own side against 62% of the extrapolation
    group, making the supposedly harder group easier on what mattered most.

    The door family does NOT stratify; see the note in door_split.
    """
    from openarm_mjlab.tasks.families import valve_split as module

    ranges = {
        "x": module.X_RANGE,
        "y": module.Y_RANGE,
        "height": module.HEIGHT_RANGE,
        "damping": module.DAMPING_RANGE,
    }
    for field, (low, high) in ranges.items():
        means = [
            sum(getattr(p, field) for p in VALVE_SPLIT[g]) / len(VALVE_SPLIT[g])
            for g in ("train", "interp", "extrap")
        ]
        spread = max(means) - min(means)
        # Compare against the SAMPLING RANGE, not the mean: y is centred on zero,
        # which makes a mean-relative tolerance meaningless.
        assert spread < 0.05 * (high - low), f"valve.{field} differs by group: {means}"


def test_valve_nominal_reproduces_the_hand_written_geometry():
    _assert_same_geometry(
        load_env_cfg("OpenArm-Valve"),
        valve_family_env_cfg(VALVE_NOMINAL),
        ("valve_hub", "valve_lever", "valve_grip"),
    )


def test_door_nominal_reproduces_the_hand_written_geometry():
    _assert_same_geometry(
        load_env_cfg("OpenArm-Door"),
        door_family_env_cfg(DOOR_NOMINAL),
        ("door_panel", "door_handle"),
    )


def _assert_same_geometry(hand_cfg, family_cfg, geom_suffixes):
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv

    def sizes(cfg):
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        model = env.sim.mj_model
        out = {}
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            for suffix in geom_suffixes:
                if name.endswith(suffix):
                    out[suffix] = (
                        tuple(round(float(v), 6) for v in model.geom_size[i]),
                        tuple(round(float(v), 6) for v in model.geom_pos[i]),
                    )
        env.close()
        return out

    hand, family = sizes(hand_cfg), sizes(family_cfg)
    for suffix in geom_suffixes:
        assert hand[suffix] == family[suffix], (
            f"{suffix}: hand-written {hand[suffix]} vs family {family[suffix]}"
        )


@pytest.fixture(scope="module")
def door_instance_env():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg(DOOR_INSTANCE_IDS["interp"][0])
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield env
    env.close()


def test_instances_step_with_finite_signals(door_instance_env):
    env = door_instance_env
    env.reset()
    for _ in range(5):
        action = torch.zeros(2, env.action_manager.total_action_dim)
        obs, rew, _, _, _ = env.step(action)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(rew).all()
