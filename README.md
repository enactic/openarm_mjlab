# OpenArm mjlab

PPO manipulation tasks for the [OpenArm](https://github.com/enactic/openarm) bimanual robot,
built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp + RSL-RL).

## Tasks

- `OpenArm-PickPlace` — the left arm picks the orange cube from the table and
  places it in the black tray, in the full OpenArm Cell scene (right arm and lifter frozen at home).

## Setup

```bash
git clone git@github.com:enactic/openarm_mjlab.git
cd openarm_mjlab
uv sync
```

## Train

Real training needs a Linux machine with a CUDA GPU:

```bash
uv run openarm-mjlab-train OpenArm-PickPlace \
  --env.scene.num-envs 4096 \
  --env.viewer.height 480 \
  --env.viewer.max-extra-envs 0 \
  --env.viewer.width 640 \
  --video True
```

CPU smoke test (e.g. on macOS):

```bash
CUDA_VISIBLE_DEVICES= uv run openarm-mjlab-train OpenArm-PickPlace \
  --env.scene.num-envs 4 \
  --agent.max-iterations 2
```

## Play a checkpoint

```bash
uv run openarm-mjlab-play OpenArm-PickPlace --checkpoint-file <path/to/checkpoint>
```

## Measure a success rate

```bash
uv run openarm-mjlab-eval OpenArm-PickPlace
```

Rolls out one episode per environment and reports the fraction that reach
the task's success termination, alongside the rate of every other
termination term. Defaults to the latest checkpoint for the task and 256
episodes; pass `--checkpoint` and `--num-envs` to change either. When a
task has more than one candidate success condition, name it explicitly
with `--success-term`.

## Tests

```bash
uv run pytest
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
