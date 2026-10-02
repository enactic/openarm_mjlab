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

"""Tests for exporting a policy and its scene for use outside mjlab."""

import numpy as np
import pytest

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from openarm_mjlab.deploy.export import _merge_duplicate_default_classes


def test_merge_collapses_a_default_class_nested_inside_itself():
    """mjlab's scene export nests <default class="robot/main"> inside itself.

    MuJoCo rejects a repeated class name, so without this the exported scene
    does not load at all. Every child of the inner block must survive.
    """
    xml = (
        "<mujoco><default>"
        '<default class="a"><default class="x"/>'
        '<default class="a"><default class="y"/><default class="z"/></default>'
        "</default></default></mujoco>"
    )
    merged_xml, count = _merge_duplicate_default_classes(xml)
    assert count == 1
    import xml.etree.ElementTree as ET

    root = ET.fromstring(merged_xml)
    classes = [d.get("class") for d in root.iter("default")]
    assert classes.count("a") == 1, "the repeat should be gone"
    for kept in ("x", "y", "z"):
        assert kept in classes, f"child {kept} was dropped by the merge"


def test_merge_is_a_no_op_when_there_is_nothing_to_merge():
    xml = '<mujoco><default><default class="a"/><default class="b"/></default></mujoco>'
    merged_xml, count = _merge_duplicate_default_classes(xml)
    assert count == 0
    assert merged_xml == xml


@pytest.mark.parametrize("group", ["train", "interp"])
def test_exported_scene_matches_the_mjlab_model(tmp_path, group):
    """A rewritten XML is only useful if it compiles to the same physics."""
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from openarm_mjlab.deploy.export import export_scene_self_contained
    from openarm_mjlab.tasks.families import DOOR_INSTANCE_IDS

    task_id = DOOR_INSTANCE_IDS[group][0]
    out = tmp_path / group
    export_scene_self_contained(task_id, out)

    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    reference = env.sim.mj_model
    exported = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))

    for field in ("nq", "nv", "njnt", "ngeom", "nbody", "nu", "nsensor"):
        assert getattr(reference, field) == getattr(exported, field), field
    for field in (
        "geom_friction",
        "geom_priority",
        "geom_size",
        "body_mass",
        "jnt_range",
        "dof_damping",
        "geom_solref",
    ):
        assert np.allclose(
            np.asarray(getattr(reference, field)),
            np.asarray(getattr(exported, field)),
            atol=1e-9,
        ), field
    env.close()


def _tiny_policy(obs_dim: int = 12, action_dim: int = 4):
    """A small MLPModel, exported the way the runner exports one."""
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    obs = TensorDict(
        {"actor": torch.zeros(1, obs_dim), "critic": torch.zeros(1, obs_dim)},
        batch_size=[1],
    )
    groups = {"actor": ["actor"], "critic": ["critic"]}
    model = MLPModel(
        obs,
        groups,
        "actor",
        action_dim,
        (16, 16),
        "elu",
        True,
        {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    model.eval()
    return model


def _write_exports(model, out_dir, obs_dim):
    import torch

    torch.jit.script(model.as_jit()).save(str(out_dir / "policy.pt"))
    onnx_model = model.as_onnx(verbose=False)
    onnx_model.eval()
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        str(out_dir / "policy.onnx"),
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
    )


def test_verify_accepts_an_export_that_matches(tmp_path):
    """A faithful export round-trips through both files."""
    import torch
    from tensordict import TensorDict

    from openarm_mjlab.deploy.export import verify_exported_policy

    obs_dim = 12
    model = _tiny_policy(obs_dim)
    _write_exports(model, tmp_path, obs_dim)

    torch.manual_seed(0)
    observations = torch.randn(6, obs_dim)
    with torch.no_grad():
        expected = model(
            TensorDict({"actor": observations, "critic": observations}, batch_size=[6])
        )
    deltas = verify_exported_policy(
        expected.numpy().astype(np.float32),
        observations.numpy().astype(np.float32),
        tmp_path,
    )
    assert deltas["policy.pt"] <= 1e-5
    assert deltas["policy.onnx"] <= 1e-3


def test_verify_rejects_an_export_that_does_not_match(tmp_path):
    """The guard has to fire when the file is not the policy it came from.

    This is the failure it exists for: an export can be written successfully
    and still compute something else, in which case nothing else in the
    pipeline notices until the policy is driving hardware.
    """
    import torch
    from tensordict import TensorDict

    from openarm_mjlab.deploy.export import verify_exported_policy

    obs_dim = 12
    model = _tiny_policy(obs_dim)
    _write_exports(model, tmp_path, obs_dim)

    # Export one policy, then ask the check to reproduce a different one.
    other = _tiny_policy(obs_dim)
    torch.manual_seed(0)
    observations = torch.randn(6, obs_dim)
    with torch.no_grad():
        wrong = other(
            TensorDict({"actor": observations, "critic": observations}, batch_size=[6])
        )

    with pytest.raises(RuntimeError, match="does not reproduce the trained policy"):
        verify_exported_policy(
            wrong.numpy().astype(np.float32),
            observations.numpy().astype(np.float32),
            tmp_path,
        )
