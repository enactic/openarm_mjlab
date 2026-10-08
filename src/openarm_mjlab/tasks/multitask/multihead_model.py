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

"""A shared trunk with one output head per task, selected by the task one-hot.

Sequential fine-tuning across the five tasks always ended with only the
last-trained task working. Two measurements said that was not classic
catastrophic forgetting. The shared observation normalizer was not
responsible: its stored sample count reaches ~2e8, so one learning iteration
of ~24.5k samples moves the running statistics by about 0.01%, far too little
to explain reach dropping from 100% to 0.000 within a single iteration. And
the loss was masking rather than destruction: a 100-iteration reach revisit on
a finished checkpoint restored reach from 0.074 to 0.953, against the ~2000
iterations it needs from scratch, which a network that had truly destroyed the
skill could not do.

Together those point at the shared trunk still encoding all five tasks while
the comparatively small mapping on top gets overwritten. Per-task output heads
stop that overwriting directly, and cost far less than the Fisher-information
machinery of EWC, for which rsl_rl exposes no hooks.

Only two things are overridden on ``MLPModel``: ``get_latent`` reads the task
one-hot from the RAW observation before normalization, and ``mlp`` is wrapped
so the final layer becomes one head per task.

Export needs a third. ``get_latent`` is where the head is chosen, and rsl_rl's
JIT and ONNX wrappers do not call it -- they run ``obs_normalizer`` and then
``mlp`` directly, so neither can read the one-hot at all. The two paths then
fail differently. ``export_policy_to_jit`` runs ``torch.jit.script``, which
cannot compile the selection and raises. ``export_policy_to_onnx`` traces
against an all-zero dummy observation, which makes the head index a constant:
argmax over an all-zero one-hot is head 0, so the exported policy runs reach's
head for every task without raising anything.

``as_jit`` and ``as_onnx`` below return wrappers that select the head from the
observation inside the graph instead.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from torch import nn

# The task one-hot occupies the final observation dims; see env_cfgs._unify,
# which appends ``mt_task_id`` last.
NUM_TASKS = 5

# Width of each task's private hidden layer. A single Linear(128 -> 8) per task
# is very little private capacity on top of a trunk serving five manipulation
# behaviours, and concurrent training made the shortfall visible: with every
# batch containing all five tasks, so interference between visits is
# structurally impossible, skills still appeared and then vanished. That is an
# optimisation failure in the shared trunk rather than forgetting. At 128 each
# task gets 17,544 private parameters instead of 1,032.
HEAD_HIDDEN = 128


class _MultiHead(nn.Module):
    """A shared trunk followed by one output head per task."""

    def __init__(
        self, mlp: nn.Module, num_tasks: int, owner: MultiHeadMLPModel
    ) -> None:
        super().__init__()
        layers = list(mlp)
        if not isinstance(layers[-1], nn.Linear):
            raise TypeError(
                f"expected the MLP to end in nn.Linear, got {type(layers[-1]).__name__}"
            )
        self.trunk = nn.Sequential(*layers[:-1])
        last = layers[-1]
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(last.in_features, HEAD_HIDDEN),
                    nn.ELU(),
                    nn.Linear(HEAD_HIDDEN, last.out_features),
                )
                for _ in range(num_tasks)
            ]
        )
        # Plain attribute rather than a submodule, so it never enters the
        # state dict.
        object.__setattr__(self, "_owner", owner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the trunk, then the head belonging to each sample's task."""
        latent = self.trunk(x)
        stacked = torch.stack([head(latent) for head in self.heads], dim=1)
        index = self._owner.task_index
        if index is None:
            # No task one-hot seen yet, e.g. a shape-probing call: fall back to
            # head 0 rather than guessing.
            return stacked[:, 0]
        index = index.to(stacked.device).clamp(0, stacked.shape[1] - 1)
        gathered = stacked.gather(
            1, index.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        )
        return gathered.squeeze(1)


class MultiHeadMLPModel(MLPModel):
    """An ``MLPModel`` whose output head is chosen by the task one-hot."""

    def __init__(self, *args, **kwargs) -> None:
        """Build the base model, then replace its head with one head per task."""
        super().__init__(*args, **kwargs)
        self.task_index: torch.Tensor | None = None
        self.mlp = _MultiHead(self.mlp, NUM_TASKS, self)

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        """Normalize the observation, capturing the task index beforehand.

        The index is read from the RAW one-hot. After normalization the one-hot
        is no longer 0/1, and since each dimension gets its own mean and std,
        an argmax over normalized values is not guaranteed to agree with the
        argmax of the raw one-hot.
        """
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        if raw.shape[-1] >= NUM_TASKS:
            self.task_index = raw[..., -NUM_TASKS:].argmax(dim=-1)
        return self.obs_normalizer(raw)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible copy that selects its own head."""
        return _TorchMultiHeadModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible copy that selects its own head."""
        return _OnnxMultiHeadModel(self, verbose)


class _ExportMultiHead(nn.Module):
    """Deployment copy of the model, with head selection inside the graph.

    The base wrappers in rsl_rl call ``obs_normalizer`` and then ``mlp``,
    bypassing ``get_latent`` and so bypassing the only place the task one-hot
    is read. This does the same work on the raw observation tensor, so the
    argmax becomes a node in the exported graph rather than a Python value
    fixed when that graph was built.

    Everything here has to stay compatible with ``torch.jit.script`` as well
    as with tracing, since the JIT and ONNX exporters use one each.
    """

    def __init__(self, model: MultiHeadMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        multi_head = model.mlp
        self.trunk = copy.deepcopy(multi_head.trunk)
        self.heads = copy.deepcopy(multi_head.heads)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim
        # An instance attribute, not the module-level constant: TorchScript
        # cannot close over a global int, and ``export_policy_to_jit`` runs
        # ``torch.jit.script``, not ``torch.jit.trace``.
        self.num_tasks = int(NUM_TASKS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        index = x[..., x.shape[-1] - self.num_tasks :].argmax(dim=-1)
        latent = self.trunk(self.obs_normalizer(x))
        # A plain loop rather than a comprehension: TorchScript can iterate an
        # nn.ModuleList, but not comprehend over one.
        outputs: list[torch.Tensor] = []
        for head in self.heads:
            outputs.append(head(latent))
        stacked = torch.stack(outputs, dim=1)
        gathered = stacked.gather(
            1, index.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        )
        return self.deterministic_output(gathered.squeeze(1))


class _TorchMultiHeadModel(_ExportMultiHead):
    """Exportable multi-head model for JIT."""

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxMultiHeadModel(_ExportMultiHead):
    """Exportable multi-head model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MultiHeadMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

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
