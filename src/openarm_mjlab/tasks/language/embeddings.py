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

"""Build and load the frozen sentence embeddings the policy is conditioned on.

The encoder is frozen, so running it every reset would be pure cost and would
put a transformer next to the simulator on the training device. The whole pool
is 44 sentences, so it is embedded ONCE into a lookup table.

The table is a build artefact and is deliberately not committed: it depends on
which encoder you choose, and a stale one silently conditions a policy on
vectors from a different model. Build it with::

    openarm-mjlab-build-instructions            # default encoder
    openarm-mjlab-build-instructions MODEL      # any sentence-transformers id

sentence-transformers is an optional dependency (pip install
openarm-mjlab[language]) and is imported only by the build step, never at
training or evaluation time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openarm_mjlab.tasks.language import instructions

DEFAULT_TABLE = Path(__file__).with_name("instruction_embeddings.pt")
_CACHE: dict[Path, dict] = {}


def build(
    model_name: str = instructions.DEFAULT_ENCODER, out: Path = DEFAULT_TABLE
) -> Path:
    """Embed every instruction once and write the lookup table."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "building the instruction table needs sentence-transformers; "
            "install it with: pip install 'openarm-mjlab[language]'"
        ) from exc

    model = SentenceTransformer(model_name)
    sentences: list[str] = []
    index: dict[str, tuple[int, int]] = {}
    for goal in instructions.GOALS:
        for split, pool in (
            ("train", instructions.TRAIN[goal]),
            ("held_out", instructions.HELD_OUT[goal]),
        ):
            index[f"{goal}|{split}"] = (len(sentences), len(sentences) + len(pool))
            sentences.extend(pool)

    raw = torch.tensor(
        model.encode(sentences, convert_to_numpy=True), dtype=torch.float32
    )
    save(out, torch.nn.functional.normalize(raw, dim=-1), index, sentences, model_name)
    return out


def save(
    path: Path,
    embeddings: torch.Tensor,
    index: dict[str, tuple[int, int]],
    sentences: list[str],
    model_name: str,
) -> None:
    """Write a table. Separated out so tests can build a synthetic one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # A previous load of this path is now stale.
    _CACHE.pop(path, None)
    torch.save(
        {
            "embeddings": embeddings,
            "index": index,
            "sentences": sentences,
            "model": model_name,
            "goals": list(instructions.GOALS),
        },
        path,
    )


def load(path: Path | None = None) -> dict:
    """Load a table, with a message that says what to do if it is missing."""
    path = Path(path) if path is not None else DEFAULT_TABLE
    if path in _CACHE:
        return _CACHE[path]
    if not path.exists():
        raise FileNotFoundError(
            f"no instruction embeddings at {path}. Build them first:\n"
            f"    openarm-mjlab-build-instructions"
        )
    table = torch.load(path, weights_only=False)
    if table.get("goals") != list(instructions.GOALS):
        raise ValueError(
            f"{path} was built for goals {table.get('goals')} but instructions.py "
            f"now declares {list(instructions.GOALS)} -- rebuild it, or the policy "
            "is conditioned on vectors that mean something else"
        )
    # A half-built or hand-edited table otherwise fails later, inside a reset,
    # as a bare KeyError or a silent shape mismatch.
    embeddings = table.get("embeddings")
    index = table.get("index", {})
    sentences = table.get("sentences", [])
    missing = [
        f"{goal}|{split}"
        for goal in instructions.GOALS
        for split in ("train", "held_out")
        if f"{goal}|{split}" not in index
    ]
    if missing:
        raise ValueError(f"{path} is missing entries for {missing} -- rebuild it")
    if embeddings is None or embeddings.ndim != 2:
        raise ValueError(f"{path} has no 2-D embeddings tensor -- rebuild it")
    if len(sentences) != embeddings.shape[0]:
        raise ValueError(
            f"{path} has {len(sentences)} sentences but {embeddings.shape[0]} "
            "embeddings -- rebuild it"
        )
    norms = embeddings.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise ValueError(
            f"{path} embeddings are not unit-norm (min {norms.min():.3f}, max "
            f"{norms.max():.3f}); the gate compares them by cosine -- rebuild it"
        )
    _CACHE[path] = table
    return table


def dim(path: Path | None = None) -> int:
    """Width of the embedding the policy is conditioned on."""
    return int(load(path)["embeddings"].shape[-1])


def pool(goal: str, split: str, device=None, path: Path | None = None) -> torch.Tensor:
    """Every embedded phrasing of one goal, for one split."""
    table = load(path)
    lo, hi = table["index"][f"{goal}|{split}"]
    out = table["embeddings"][lo:hi]
    return out.to(device) if device is not None else out


def main() -> None:
    """Console entry point for building the table."""
    parser = argparse.ArgumentParser(
        prog="openarm-mjlab-build-instructions",
        description="Embed the instruction pool into a lookup table.",
    )
    parser.add_argument("model", nargs="?", default=instructions.DEFAULT_ENCODER)
    parser.add_argument("--out", default=str(DEFAULT_TABLE))
    args = parser.parse_args()
    out = build(args.model, Path(args.out))
    table = load(out)
    print(
        f"wrote {out} ({len(table['sentences'])} sentences x "
        f"{table['embeddings'].shape[1]}, {args.model})"
    )


if __name__ == "__main__":
    main()
