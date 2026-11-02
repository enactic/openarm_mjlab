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

"""Tests for the task integrity audit."""

from types import SimpleNamespace

import mujoco
import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg
from openarm_mjlab.audit import checks
from openarm_mjlab.audit.specs import SPECS, TaskIntegritySpec


@pytest.mark.parametrize("task_id", sorted(SPECS))
def test_spec_names_exist_in_the_built_task(task_id):
    """Every geom, joint and termination a spec names must exist, or the audit checks nothing."""
    from mjlab.envs import ManagerBasedRlEnv

    assert task_id in list_tasks()
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        geo = checks._Geometry(
            env, SPECS[task_id]
        )  # raises on any missing geom or joint
        assert SPECS[task_id].success_term in env.termination_manager.active_terms
        # object_geoms must list EVERY object collision geom, or penetration of an
        # unlisted one would go unseen.
        model_objects = {
            geo.names[g]
            for g in geo.objects
            if geo.m.geom_contype[g] or geo.m.geom_conaffinity[g]
        }
        assert set(SPECS[task_id].object_geoms) == model_objects
    finally:
        env.close()


_XML = """
<mujoco>
  <worldbody>
    <body name="robot/hand" pos="0 0 {z}"><freejoint name="robot/free"/><geom name="robot/pad" type="box" size="0.02 0.02 0.02"/></body>
    <body name="obj/block"><joint name="obj/slide" type="slide"/><geom name="obj/geom" type="box" size="0.02 0.02 0.02"/></body>
  </worldbody>
</mujoco>
"""


def _fake_env(z):
    model = mujoco.MjModel.from_xml_string(_XML.format(z=z))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos = torch.tensor(data.qpos).unsqueeze(0)
    return SimpleNamespace(
        sim=SimpleNamespace(mj_model=model, data=SimpleNamespace(qpos=qpos)),
        reset=lambda: None,
    )


_SPEC = TaskIntegritySpec(
    task_id="synthetic",
    success_term="none",
    object_joint="obj/slide",
    object_geoms=("obj/geom",),
    target_geoms=("obj/geom",),
    contact_required_at_success=False,
    grasp=False,
)


def test_spawn_clearance_flags_an_overlap_with_its_depth():
    """Boxes overlapping by 10 mm must be reported as penetrating, about 10 mm deep."""
    report = checks.spawn_clearance(_fake_env(z=0.03), _SPEC, rounds=1)
    assert report.penetrating_fraction == 1.0
    assert report.deepest_mm == pytest.approx(-10.0, abs=1.0)
    assert report.pairs == {"robot/pad -> obj/geom": 1}


def test_spawn_clearance_passes_separated_geoms():
    """Boxes 10 mm apart must not be reported."""
    report = checks.spawn_clearance(_fake_env(z=0.05), _SPEC, rounds=1)
    assert report.penetrating_fraction == 0.0
    assert report.pairs == {}
