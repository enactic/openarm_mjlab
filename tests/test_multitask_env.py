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

"""Integration tests for the shared multi-task observation layout.

Note: building an env compiles mujoco-warp CPU kernels; the first run can take
a few minutes.
"""

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks, load_env_cfg
from openarm_mjlab.tasks.multitask import mdp
from openarm_mjlab.tasks.multitask.env_cfgs import (
    TOTAL_OBJECT_WIDTH,
    _SLOT_OFFSET,
)
from openarm_mjlab.tasks.multitask.multihead_model import NUM_TASKS

TASK_IDS = {
    "reach": "OpenArm-MultiTask-Reach",
    "valve": "OpenArm-MultiTask-Valve",
    "door": "OpenArm-MultiTask-Door",
    "drawer": "OpenArm-MultiTask-Drawer",
    "puck": "OpenArm-MultiTask-Puck",
}

# joint_pos(18) + joint_vel(18) + the shared object region + actions(8) + the
# task one-hot. Every task pads to the same total, which is the point.
OBS_DIM = 18 + 18 + TOTAL_OBJECT_WIDTH + 8 + NUM_TASKS


def test_every_multitask_id_is_registered():
    registered = list_tasks()
    for task_id in TASK_IDS.values():
        assert task_id in registered


def test_slots_are_disjoint_and_cover_the_region():
    """Each task owns a distinct range, and together they tile the region."""
    covered: set[int] = set()
    for name in mdp.TASK_NAMES:
        start = _SLOT_OFFSET[name]
        width = {"reach": 3, "valve": 5, "door": 5, "drawer": 5, "puck": 7}[name]
        span = set(range(start, start + width))
        assert not (span & covered), f"{name}'s slot overlaps another task's"
        covered |= span
    assert covered == set(range(TOTAL_OBJECT_WIDTH))


@pytest.fixture(scope="module")
def envs():
    from mjlab.envs import ManagerBasedRlEnv

    built = {}
    for name, task_id in TASK_IDS.items():
        cfg = load_env_cfg(task_id)
        cfg.scene.num_envs = 2
        built[name] = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    yield built
    for env in built.values():
        env.close()


def test_all_tasks_share_one_observation_shape(envs):
    """A single network is only usable if every task presents one shape."""
    for name, env in envs.items():
        obs, _ = env.reset()
        assert obs["actor"].shape == (2, OBS_DIM), name
        assert obs["critic"].shape == (2, OBS_DIM), name
        assert env.action_manager.total_action_dim == 8, name


def test_task_one_hot_is_last_and_identifies_the_task(envs):
    """The final dims must be a clean one-hot; the model reads them raw."""
    for name, env in envs.items():
        obs, _ = env.reset()
        onehot = obs["actor"][:, -NUM_TASKS:]
        assert torch.equal(onehot.sum(dim=-1), torch.ones(2)), name
        expected = mdp.TASK_NAMES.index(name)
        assert torch.equal(
            onehot.argmax(dim=-1), torch.full((2,), expected, dtype=torch.long)
        ), name


def test_unused_slots_are_zero(envs):
    """A task must not write outside the slot it owns."""
    start_of_region = 18 + 18
    for name, env in envs.items():
        obs, _ = env.reset()
        region = obs["actor"][:, start_of_region : start_of_region + TOTAL_OBJECT_WIDTH]
        own_start = _SLOT_OFFSET[name]
        own_width = {"reach": 3, "valve": 5, "door": 5, "drawer": 5, "puck": 7}[name]
        before = region[:, :own_start]
        after = region[:, own_start + own_width :]
        assert torch.count_nonzero(before) == 0, f"{name} wrote before its slot"
        assert torch.count_nonzero(after) == 0, f"{name} wrote after its slot"


def test_actions_land_at_the_same_column_for_every_task(envs):
    """The layout bug this guards against put 'actions' at a different index."""
    actions_start = 18 + 18 + TOTAL_OBJECT_WIDTH
    for name, env in envs.items():
        obs, _ = env.reset()
        terms = env.observation_manager.cfg["actor"].terms
        keys = list(terms.keys())
        assert keys[-1] == "mt_task_id", name
        assert keys[-2] == "actions", name
        assert obs["actor"].shape[1] == actions_start + 8 + NUM_TASKS, name


def test_env_steps_with_finite_signals(envs):
    for name, env in envs.items():
        env.reset()
        for _ in range(5):
            action = torch.zeros(2, env.action_manager.total_action_dim)
            obs, rew, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(obs["actor"]).all(), name
            assert torch.isfinite(rew).all(), name


def test_config_points_at_the_multitask_classes():
    """Guard the silent-fallback risk: a bad path would quietly use stock PPO.

    ``class_name`` is resolved by string at runtime, so a typo or a moved class
    degrades to the default single-head model and global advantage
    normalization without raising anything.
    """
    import importlib

    from openarm_mjlab.tasks.multitask.multihead_model import MultiHeadMLPModel
    from openarm_mjlab.tasks.multitask.ppo import MultiTaskPPO
    from openarm_mjlab.tasks.multitask.rl_cfg import multitask_ppo_runner_cfg

    cfg = multitask_ppo_runner_cfg()
    for path, expected in (
        (cfg.actor.class_name, MultiHeadMLPModel),
        (cfg.algorithm.class_name, MultiTaskPPO),
    ):
        module_path, _, attr = path.rpartition(".")
        resolved = getattr(importlib.import_module(module_path), attr)
        assert resolved is expected, path


def _build_model(head_hidden: int = 0):
    """A MultiHeadMLPModel with deliberately distinct heads.

    The heads a fresh model gets are near-identical, so a wrong head would
    still produce nearly the right numbers and the test would pass on broken
    code. Offsetting each head makes picking the wrong one unmistakable.
    """
    from tensordict import TensorDict

    from openarm_mjlab.tasks.multitask.multihead_model import MultiHeadMLPModel

    obs_dim = OBS_DIM
    obs = TensorDict(
        {"actor": torch.zeros(1, obs_dim), "critic": torch.zeros(1, obs_dim)},
        batch_size=[1],
    )
    groups = {"actor": ["actor"], "critic": ["critic"]}
    model = MultiHeadMLPModel(
        obs,
        groups,
        "actor",
        8,
        (64, 64),
        "elu",
        True,
        {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    model.eval()
    for offset, head in enumerate(model.mlp.heads):
        for param in head.parameters():
            param.data.add_(float(offset))
    return model


def _one_hot_obs(index: int, core: torch.Tensor):
    """A raw observation batch tagged for one task."""
    from tensordict import TensorDict

    raw = torch.cat([core, torch.zeros(len(core), NUM_TASKS)], dim=-1)
    raw[:, OBS_DIM - NUM_TASKS + index] = 1.0
    return raw, TensorDict({"actor": raw, "critic": raw}, batch_size=[len(core)])


def test_jit_export_selects_the_head_the_one_hot_asks_for():
    """The head must be chosen inside the exported graph.

    rsl_rl's export wrappers skip ``get_latent``, which is the only place the
    task one-hot is read, so a model that leaves head selection there cannot
    be exported correctly. This uses ``torch.jit.script``, which is what
    ``export_policy_to_jit`` calls -- on the unfixed model it fails outright
    with ``Module '_MultiHead' has no attribute '_owner'``.
    """
    model = _build_model()
    exported = torch.jit.script(model.as_jit())

    torch.manual_seed(0)
    core = torch.randn(4, OBS_DIM - NUM_TASKS)
    for index, name in enumerate(mdp.TASK_NAMES):
        raw, obs = _one_hot_obs(index, core)
        with torch.no_grad():
            eager = model(obs)
            scripted = exported(raw)
        assert torch.allclose(eager, scripted, atol=1e-5), (
            f"{name}: exported policy disagrees with the eager model by "
            f"{(eager - scripted).abs().max().item():.4f} -- it is running a "
            "different task's head"
        )


def test_onnx_export_wrapper_selects_the_head_too():
    """The ONNX path needs the same treatment, and fails more quietly.

    ``export_policy_to_onnx`` traces with an all-zero dummy observation. On
    the unfixed model the head index is a Python value read at trace time, so
    ``argmax`` of an all-zero one-hot bakes in head 0 and every exported
    policy runs reach's head whatever task it is given -- with nothing raised.

    Checked at the wrapper level rather than through a real ONNX round trip
    because that would need onnxruntime, which this package does not depend
    on. The wrapper is where the selection lives, so it is what regresses.
    """
    model = _build_model()
    onnx_model = model.as_onnx(verbose=False)
    (dummy,) = onnx_model.get_dummy_inputs()
    assert dummy.shape == (1, OBS_DIM)
    assert onnx_model.input_names == ["obs"]
    assert onnx_model.output_names == ["actions"]

    torch.manual_seed(0)
    core = torch.randn(4, OBS_DIM - NUM_TASKS)
    for index, name in enumerate(mdp.TASK_NAMES):
        raw, obs = _one_hot_obs(index, core)
        with torch.no_grad():
            assert torch.allclose(model(obs), onnx_model(raw), atol=1e-5), name
