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

"""PPO runner config for the OpenArm move-puck task, privileged and vision variants."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...rl_cfg import ppo_runner_cfg


def openarm_puck_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return the privileged-state puck task's PPO runner config."""
    return ppo_runner_cfg("openarm_puck")


# Same small CNN + spatial-softmax architecture mjlab's own vision
# reference task uses: proven, not reinvented.
_VISION_CNN_CFG = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


def openarm_puck_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return the depth-camera-only puck task's PPO runner config."""
    return ppo_runner_cfg(
        "openarm_puck_vision",
        hidden_dims=(256, 256, 128),
        model_kwargs={"cnn_cfg": _VISION_CNN_CFG, "class_name": _VISION_MODEL_CLS},
        obs_groups={
            "actor": ("actor", "camera"),
            "critic": ("critic", "camera"),
        },
    )
