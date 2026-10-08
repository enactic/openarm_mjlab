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

"""A policy conditioned on an instruction, with no per-goal output heads.

The obvious design gives each goal its own output head and mixes them with a
router over the embedding. It works, and it quietly reinstates the limitation
the embedding was meant to remove: the head set is fixed, so an instruction
naming a goal the network has no head for can only ever be a blend of goals it
does have. It also creates an expert symmetry that has to be broken by hand --
identical heads receive identical gradients and never diverge, which is the
Net2Net (arXiv:1511.05641) observation applied to a router.

Neither problem exists here. The instruction is projected down and
concatenated to what the trunk reads, a single head produces the action, and
nothing in the architecture enumerates goals.

The projection is not cosmetic: concatenating 768 raw embedding dimensions to
a 51-dimensional observation would let the instruction dominate the first
layer by width alone. 32 keeps it comparable to the state.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from torch import nn

from openarm_mjlab.tasks.language import embeddings

# Width of the projected instruction. Concatenating 768 raw embedding dims to
# a ~51-dim observation would let the instruction dominate the first layer by
# width alone; 32 keeps it comparable to the state.
PROJECTION_DIM = 32


class LanguageConditionedMLP(MLPModel):
    """MLPModel whose observation ends in a frozen instruction embedding."""

    # Set by the env cfg so the model reads the SAME table the observation was
    # built from. Without it the model takes the default table's width while
    # the env emits another, and nothing crashes: `_get_latent_dim` and
    # `get_latent` use the same wrong number, so the shapes line up and part of
    # the state is fed to the projection as if it were the instruction.
    table_path = None

    def __init__(self, *args, **kwargs) -> None:
        """Read the instruction width first, then build the base model.

        Order matters: ``MLPModel.__init__`` calls ``_get_latent_dim()``, which
        needs ``embedding_dim`` already set.
        """
        self.embedding_dim = embeddings.dim(type(self).table_path)
        super().__init__(*args, **kwargs)
        if self.embedding_dim >= self.obs_dim:
            raise ValueError(
                f"instruction width {self.embedding_dim} does not fit inside an "
                f"observation of {self.obs_dim}; wrong embedding table?"
            )
        if len(self.obs_groups) != 1:
            raise ValueError(
                f"expected one observation group, got {self.obs_groups}: the "
                "instruction is located as the final columns of the "
                "concatenation, which is only well defined for one group"
            )
        self.instruction_proj = nn.Linear(self.embedding_dim, PROJECTION_DIM)

    def _get_latent_dim(self) -> int:
        return self.obs_dim - self.embedding_dim + PROJECTION_DIM

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        """Normalize the state, project the instruction, concatenate.

        The embedding is read RAW. It is unit-norm by construction and its
        geometry is the whole signal; the observation normalizer rescales each
        dimension independently and would destroy exactly that.
        """
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        instruction = raw[..., -self.embedding_dim :]
        state = self.obs_normalizer(raw)[..., : -self.embedding_dim]
        return torch.cat([state, self.instruction_proj(instruction)], dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a copy that projects the instruction inside the graph."""
        return _ExportModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-exportable copy."""
        model = _ExportModel(self)
        model.verbose = verbose
        return model


class _ExportModel(nn.Module):
    """Deployment copy. Does the projection in-graph.

    rsl_rl's own export wrappers call obs_normalizer then mlp, skipping
    get_latent entirely, so a model that conditions inside get_latent
    exports to something that ignores its conditioning.
    """

    is_recurrent: bool = False

    def __init__(self, model: LanguageConditionedMLP) -> None:
        super().__init__()
        self.embedding_dim = model.embedding_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.instruction_proj = copy.deepcopy(model.instruction_proj)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        width = x.shape[-1] - self.embedding_dim
        instruction = x[..., width:]
        state = self.obs_normalizer(x)[..., :width]
        latent = torch.cat([state, self.instruction_proj(instruction)], dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]


class LanguageConditionedCritic(LanguageConditionedMLP):
    """Critic: identical conditioning, scalar output."""
