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

"""Tests for the language-conditioned puck task.

None of these need the real embedding table or sentence-transformers: a
synthetic table is built in a tmp_path, which is also a check that the table
format is simple enough to construct without the encoder.

Note: building an env compiles mujoco-warp CPU kernels; the first run can take
a few minutes.
"""

import pytest
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.tasks.registry import list_tasks
from openarm_mjlab.tasks.language import diagnostics, embeddings, instructions
from openarm_mjlab.tasks.language import mdp as language_mdp
from openarm_mjlab.tasks.puck import mdp as puck_mdp

TASK_ID = "OpenArm-Puck-Language"
WIDTH = 16
TABLE_PATH: dict[str, object] = {}


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    """A synthetic table whose two goals are linearly separable by design."""
    path = tmp_path_factory.mktemp("instructions") / "table.pt"
    torch.manual_seed(0)
    rows, index, sentences = [], {}, []
    for gi, goal in enumerate(instructions.GOALS):
        axis = torch.zeros(WIDTH)
        axis[gi] = 1.0
        for split in ("train", "held_out"):
            pool = (
                instructions.TRAIN[goal]
                if split == "train"
                else instructions.HELD_OUT[goal]
            )
            index[f"{goal}|{split}"] = (len(sentences), len(sentences) + len(pool))
            sentences.extend(pool)
            for _ in pool:
                rows.append(axis + 0.05 * torch.randn(WIDTH))
    emb = torch.nn.functional.normalize(torch.stack(rows), dim=-1)
    embeddings.save(path, emb, index, sentences, "synthetic")
    TABLE_PATH["p"] = path
    return path


def test_task_is_registered():
    assert TASK_ID in list_tasks()


def test_pools_do_not_overlap():
    for goal in instructions.GOALS:
        assert not set(instructions.TRAIN[goal]) & set(instructions.HELD_OUT[goal])
    everything = [
        s
        for g in instructions.GOALS
        for s in instructions.TRAIN[g] + instructions.HELD_OUT[g]
    ]
    assert len(everything) == len(set(everything))


def test_missing_table_says_how_to_build_it():
    with pytest.raises(FileNotFoundError, match="openarm-mjlab-build-instructions"):
        embeddings.load(
            tmp_path_missing := __import__("pathlib").Path("/nonexistent/table.pt")
        )
    assert not tmp_path_missing.exists()


def test_table_built_for_other_goals_is_rejected(tmp_path):
    """A stale table conditions the policy on vectors that mean something else."""
    path = tmp_path / "stale.pt"
    torch.save(
        {
            "embeddings": torch.zeros(2, WIDTH),
            "index": {},
            "sentences": [],
            "model": "synthetic",
            "goals": ["up", "down"],
        },
        path,
    )
    with pytest.raises(ValueError, match="rebuild"):
        embeddings.load(path)


def test_separability_diagnostic_scores_a_clean_table(table):
    scores = diagnostics.encoder_separability(table_path=table)
    assert scores["overall"] == pytest.approx(1.0)


def test_separability_diagnostic_reports_chance_for_meaningless_embeddings(tmp_path):
    """The floor: random vectors carry nothing, so held-out phrasings are chance.

    Guards against a diagnostic that always looks encouraging.
    """
    path = tmp_path / "random.pt"
    torch.manual_seed(1)
    rows, index, sentences = [], {}, []
    for goal in instructions.GOALS:
        for split in ("train", "held_out"):
            pool = (
                instructions.TRAIN[goal]
                if split == "train"
                else instructions.HELD_OUT[goal]
            )
            index[f"{goal}|{split}"] = (len(sentences), len(sentences) + len(pool))
            sentences.extend(pool)
            rows.extend(torch.randn(WIDTH) for _ in pool)
    emb = torch.nn.functional.normalize(torch.stack(rows), dim=-1)
    embeddings.save(path, emb, index, sentences, "random")
    scores = diagnostics.encoder_separability(table_path=path)
    assert scores["overall"] < 0.75


def test_observation_ambiguity_detects_a_leak():
    """A goal-revealing observation must score high, a blind one at chance."""
    torch.manual_seed(0)
    goals = torch.randint(2, (400,))
    leaky = torch.randn(400, 4)
    leaky[:, 0] += goals.float() * 8.0
    assert diagnostics.observation_ambiguity(leaky, goals) > 0.9
    blind = torch.randn(400, 4)
    assert diagnostics.observation_ambiguity(blind, goals) < 0.7


@pytest.fixture(scope="module")
def env(table):
    from mjlab.envs import ManagerBasedRlEnv

    from openarm_mjlab.tasks.language.env_cfg import openarm_language_puck_env_cfg

    cfg = openarm_language_puck_env_cfg(table_path=table)
    # At least one env per held-out phrasing, so no cell comes out empty.
    cfg.scene.num_envs = max(len(p) for p in instructions.HELD_OUT.values())
    built = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    built.instruction_seed = 0
    yield built
    built.close()


@pytest.fixture(autouse=True)
def _clean_instruction_state(request):
    """Clear the steering attributes after every test that uses the env.

    The env fixture is module-scoped, so anything a test leaves set on it
    leaks into the next one. These four are all read with `getattr(..., None)`,
    so a leaked value is silent -- a later test just quietly evaluates a
    different goal, or draws held-out phrasings while claiming to draw
    training ones.
    """
    yield
    built = request.node.funcargs.get("env")
    if built is not None:
        for name in (
            "instruction_goal",
            "instruction_shown",
            "instruction_split",
            "instruction_phrase_index",
        ):
            setattr(built, name, None)


def test_observation_ends_in_a_unit_norm_instruction(env):
    obs, _ = env.reset()
    instruction = obs["actor"][:, -WIDTH:]
    norms = instruction.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_named_goals_are_distinct_targets(env):
    """The two goals must be different places, or nothing below means anything."""
    seen = {}
    for goal in instructions.GOALS:
        env.instruction_goal = goal
        env.reset()
        seen[goal] = puck_mdp.goal_buf(env)[0].clone()
    assert not torch.allclose(seen["right"], seen["left"])


def test_the_instruction_shown_can_differ_from_the_goal_scored(env):
    """The control the whole measurement rests on."""
    env.instruction_goal = "right"
    env.instruction_shown = "left"
    env.reset()
    assert torch.all(language_mdp.goal_index(env) == instructions.GOALS.index("right"))
    shown = embeddings.pool("left", "train", path=TABLE_PATH["p"])
    got = language_mdp.instruction(env)[0]
    assert torch.cdist(got[None], shown).min() < 1e-5


def test_observation_does_not_reveal_the_goal(env):
    """The goal vector is gone; what remains is identical for both goals.

    Both resets are given the SAME seed, so the only difference between them
    is which goal was named. Without that the comparison is dominated by
    joint-velocity observation noise (Unoise +-1.5, worth ~0.83) while the
    signal being looked for lives in a 3-dim slot worth ~0.32 -- so a loose
    threshold passed on a deliberately LEAKY observation (0.632) and failed
    about a third of the time on the correct one, depending only on where the
    global RNG happened to be. Seeded, the tolerance can be tight enough to
    mean something.
    """
    cores = {}
    for goal in instructions.GOALS:
        env.instruction_goal = goal
        obs, _ = env.reset(seed=0)
        cores[goal] = obs["actor"][:, :-WIDTH]
    spread = (cores["right"] - cores["left"]).abs().max()
    assert spread < 1e-4, (
        f"the observation encodes which target was named: max difference "
        f"{spread:.4f} between identical initial states"
    )


# --- evaluation: the square, and what it says about which precondition broke ---


def _square(rr: float, rl: float, lr: float, ll: float):
    """Build a 2x2 of per-sentence rates from four cell means."""
    n_right = len(instructions.HELD_OUT["right"])
    n_left = len(instructions.HELD_OUT["left"])
    return {
        ("right", "right"): [rr] * n_right,
        ("right", "left"): [rl] * n_left,
        ("left", "right"): [lr] * n_right,
        ("left", "left"): [ll] * n_left,
    }


def test_a_policy_that_follows_instructions_scores_high():
    from openarm_mjlab.tasks.language import evaluation

    out = evaluation.summarise(_square(0.958, 0.042, 0.042, 0.958))
    assert out.execution > 0.9
    assert out.suppression > 0.9
    assert out.separation > 0.9
    assert out.verdict == "follows the instruction"


def test_a_policy_that_ignores_instructions_scores_zero():
    """The null the whole measurement rests on."""
    from openarm_mjlab.tasks.language import evaluation

    # Always pushes right, whatever it is told.
    out = evaluation.summarise(_square(0.9, 0.9, 0.0, 0.0))
    assert abs(out.separation) < 0.1
    assert out.suppression < 0.7
    assert "not being read" in out.verdict


def test_suppression_without_execution_is_named_as_such():
    """Reward-side failure: knows which goal was named, cannot perform it.

    Measured shape of the unreachable and unrewarded conditions, where the
    scalar separation alone (+0.45) looks like partial success.
    """
    from openarm_mjlab.tasks.language import evaluation

    out = evaluation.summarise(_square(0.912, 0.004, 0.000, 0.000))
    assert out.suppression > 0.9
    assert out.execution < 0.7
    assert "not performed" in out.verdict


def test_suppression_ignores_goals_the_policy_cannot_reach():
    """An unreachable goal's off-diagonal is zero for a trivial reason.

    Counting it would flatter suppression, so only reachable goals count.
    """
    from openarm_mjlab.tasks.language import evaluation

    out = evaluation.summarise(_square(0.9, 0.5, 0.0, 0.0))
    assert out.reachable == ("right",)
    # Only the right row's leak (0.5) counts, not the trivially-zero left row.
    assert out.suppression == pytest.approx(0.5)


def test_the_two_failure_modes_are_distinguished():
    """Both score about +0.45, and they are opposite problems."""
    from openarm_mjlab.tasks.language import evaluation

    reward_side = evaluation.summarise(_square(0.912, 0.004, 0.000, 0.000))
    encoder_side = evaluation.summarise(_square(0.900, 0.929, 0.000, 0.000))
    assert reward_side.suppression > 0.9 and encoder_side.suppression < 0.7
    assert reward_side.verdict != encoder_side.verdict


def test_sign_test_and_permutation_agree_on_a_clean_case():
    from openarm_mjlab.tasks.language import evaluation

    out = evaluation.summarise(_square(1.0, 0.0, 0.0, 1.0))
    assert out.sign_wins == out.sign_n == 24
    assert out.sign_p < 1e-5
    assert out.permutation_p < 1e-3


def test_report_mentions_both_abilities():
    from openarm_mjlab.tasks.language import evaluation

    text = evaluation.summarise(_square(0.958, 0.042, 0.042, 0.958)).report()
    assert "execution" in text and "suppression" in text


def test_measure_runs_every_cell_against_the_right_goal(env):
    """The env plumbing: reward scores the goal PICKED, not the one shown."""
    import torch

    from openarm_mjlab.tasks.language import evaluation

    seen = []

    def scripted(obs):
        seen.append(
            (
                int(language_mdp.goal_index(env)[0]),
                round(puck_mdp.goal_buf(env)[0, 1].item(), 4),
            )
        )
        return torch.zeros(env.num_envs, env.action_manager.total_action_dim)

    out = evaluation.measure(scripted, env, steps=3)
    assert set(out.cells) == {
        (a, b) for a in instructions.GOALS for b in instructions.GOALS
    }
    # A do-nothing policy reaches neither target.
    assert out.execution == pytest.approx(0.0)
    assert "cannot measure" in out.verdict
    # Every scored goal was paired with its own target position.
    positions = {gi: pos for gi, pos in seen}
    assert len(positions) == 2
    assert (
        positions[instructions.GOALS.index("right")]
        != positions[instructions.GOALS.index("left")]
    )


def test_progress_baseline_uses_the_episode_goal_not_the_constant(env):
    """Guards a shaping bug the two-goal change introduced.

    `reset_puck_uniform` seeds the progress-shaping buffer with the puck's
    distance to the goal. With one fixed target that was the module constant;
    with two it must be the goal this episode actually names, or the baseline
    is measured to the wrong place and `clamp(min_dist - dist, min=0)` pays
    free progress on the first contact step, for one goal and not the other.
    """
    import torch

    from openarm_mjlab.tasks.puck.puck_env_cfg import PUCK_CFG

    for goal in instructions.GOALS:
        env.instruction_goal = goal
        env.reset()
        seeded = env._puck_min_dist
        actual = puck_mdp.puck_goal_dist(env, PUCK_CFG)
        assert torch.allclose(seeded, actual, atol=1e-4), (
            f"{goal}: progress baseline {seeded[:3]} does not match the distance "
            f"to this episode's goal {actual[:3]}"
        )
    env.instruction_goal = None


def test_export_keeps_the_instruction_conditioning(table):
    """rsl_rl's own export wrapper drops the conditioning; ours must not.

    `_ExportModel` exists because the stock wrappers call obs_normalizer then
    mlp, skipping get_latent -- which is where the instruction is read. Nothing
    else in the suite would notice a refactor reintroducing exactly that.
    """
    import torch
    from tensordict import TensorDict

    from openarm_mjlab.tasks.language import model as lang_model

    lang_model.LanguageConditionedMLP.table_path = table
    obs_dim = 24 + WIDTH
    obs = TensorDict(
        {"actor": torch.zeros(1, obs_dim), "critic": torch.zeros(1, obs_dim)},
        batch_size=[1],
    )
    groups = {"actor": ["actor"], "critic": ["critic"]}
    net = lang_model.LanguageConditionedMLP(
        obs,
        groups,
        "actor",
        4,
        (32, 32),
        "elu",
        True,
        {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    net.eval()
    exported = torch.jit.script(net.as_jit())

    torch.manual_seed(0)
    state = torch.randn(4, 24)
    outputs = []
    for goal in instructions.GOALS:
        instruction = embeddings.pool(goal, "train", path=table)[0].expand(4, WIDTH)
        raw = torch.cat([state, instruction], dim=-1)
        td = TensorDict({"actor": raw, "critic": raw}, batch_size=[4])
        with torch.no_grad():
            live, traced = net(td), exported(raw)
        assert torch.allclose(live, traced, atol=1e-5), (
            f"{goal}: exported policy differs from the live one by "
            f"{(live - traced).abs().max().item():.4f}"
        )
        outputs.append(traced)
    # And the instruction must actually change the exported output, or the
    # agreement above would be satisfied by a model that ignores it.
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-6)


# --- previously untested paths ------------------------------------------------


class _FakeTerminationManager:
    """Reports success for a fixed set of envs on a fixed step."""

    def __init__(self, hit_envs, hit_step):
        self.hit_envs, self.hit_step, self.step = hit_envs, hit_step, 0

    def get_term(self, name):
        import torch

        flags = torch.zeros(12, dtype=torch.bool)
        if self.step == self.hit_step:
            flags[list(self.hit_envs)] = True
        return flags


class _FakeEnv:
    """Duck-typed env, enough for measure()'s accounting and nothing else."""

    def __init__(self, hit_envs):
        import torch

        self.num_envs = 12
        self.device = torch.device("cpu")
        self.termination_manager = _FakeTerminationManager(hit_envs, hit_step=1)
        self.instruction_goal = self.instruction_shown = None
        self.instruction_split = self.instruction_phrase_index = None

    def reset(self, seed=None):
        self.termination_manager.step = 0
        return {}, {}

    def step(self, action):
        import torch

        self.termination_manager.step += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.termination_manager.step >= 2:
            done[:] = True
        return {}, torch.zeros(self.num_envs), done, done, {}


def test_measure_attributes_success_to_the_right_sentences():
    """measure()'s success accounting, which no simulator test exercises.

    The existing integration test uses a do-nothing policy, so every cell is
    0.0 and the `success |= ~finished & hit` path never sees a True.
    """
    import torch

    from openarm_mjlab.tasks.language import evaluation

    # Envs 0 and 1 succeed; with index = arange(12) % 12 those are phrasings 0,1.
    env = _FakeEnv(hit_envs={0, 1})
    out = evaluation.measure(lambda obs: torch.zeros(12, 8), env, steps=4)
    for cell in out.cells.values():
        assert cell[0] == 1.0 and cell[1] == 1.0, "hits not credited"
        assert all(v == 0.0 for v in cell[2:]), "success credited to the wrong sentence"


def test_measure_restores_the_callers_state():
    """Leaving instruction_split set would silently contaminate later training."""
    import torch

    from openarm_mjlab.tasks.language import evaluation

    env = _FakeEnv(hit_envs=set())
    env.instruction_split = "train"
    env.instruction_goal = "left"
    evaluation.measure(lambda obs: torch.zeros(12, 8), env, steps=2)
    assert env.instruction_split == "train"
    assert env.instruction_goal == "left"


def test_a_half_built_table_is_rejected(tmp_path):
    """A table missing entries otherwise fails later as a bare KeyError."""
    import torch

    path = tmp_path / "partial.pt"
    torch.save(
        {
            "embeddings": torch.nn.functional.normalize(torch.randn(4, WIDTH), dim=-1),
            "index": {"right|train": (0, 4)},
            "sentences": ["a", "b", "c", "d"],
            "model": "synthetic",
            "goals": list(instructions.GOALS),
        },
        path,
    )
    with pytest.raises(ValueError, match="missing entries"):
        embeddings.load(path)


def test_a_table_with_unnormalised_vectors_is_rejected(tmp_path):
    """The gate compares by cosine; un-normalised vectors break that silently."""
    import torch

    path = tmp_path / "raw.pt"
    index, sentences, rows = {}, [], []
    for goal in instructions.GOALS:
        for split in ("train", "held_out"):
            pool = (
                instructions.TRAIN[goal]
                if split == "train"
                else instructions.HELD_OUT[goal]
            )
            index[f"{goal}|{split}"] = (len(sentences), len(sentences) + len(pool))
            sentences.extend(pool)
            rows.extend(torch.randn(WIDTH) * 5.0 for _ in pool)
    embeddings.save(path, torch.stack(rows), index, sentences, "unnormalised")
    with pytest.raises(ValueError, match="unit-norm"):
        embeddings.load(path)


def test_build_says_what_to_install_when_the_encoder_is_absent(monkeypatch):
    """sentence-transformers is optional; the error must name the extra."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *rest, **kw):
        if name.startswith("sentence_transformers"):
            raise ImportError("not installed")
        return real_import(name, *rest, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"openarm-mjlab\[language\]"):
        embeddings.build()


def test_every_declared_console_script_resolves():
    """A typo in pyproject.toml only shows up when a user runs the command.

    The entry points are the documented way in, so a bad module path or a
    renamed function is a broken front door that no other test touches.
    """
    import importlib
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as fh:
        scripts = tomllib.load(fh)["project"]["scripts"]

    language = {n: t for n, t in scripts.items() if ".tasks.language." in t}
    assert language, "no language console scripts declared"
    for name, target in language.items():
        module_path, _, func = target.partition(":")
        module = importlib.import_module(module_path)
        assert callable(getattr(module, func, None)), (
            f"{name} -> {target} is not callable"
        )
