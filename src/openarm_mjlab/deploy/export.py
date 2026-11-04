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

"""Export a trained policy and its scene for use outside mjlab.

Writes three things next to each other:

* ``policy.onnx`` / ``policy.pt`` -- the actor, with the observation
  normaliser folded in.
* ``policy_meta.json`` -- the contract needed to drive it: action scale and
  offset, actuator order, decimation, solver settings, default joint positions
  and soft joint limits. A deployment target cannot import mjlab to discover
  these, and hardcoding them in a runner is an easy way to be quietly wrong.
* ``scene/`` -- the compiled MJCF together with every mesh it references, so it
  loads with plain ``mujoco.MjModel.from_xml_path`` on a machine that has
  neither mjlab nor the original asset tree.

Both policy files are read back and checked against the policy still in
memory before the export is reported as done. An exporter that can write a
file which does not match what was trained is the worst kind to have, because
the next place that surfaces is on hardware.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

import openarm_mjlab.tasks  # noqa: F401  # Registers tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.export_scene import ExportSceneCfg, export_scene
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def _to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def openarm_mujoco_cell_xml(openarm_v2) -> str:
    """Return the packaged cell XML path, whatever the package version calls it."""
    for attr in ("openarm_cell_xml", "cell_xml", "CELL_XML"):
        value = getattr(openarm_v2, attr, None)
        if value is None:
            continue
        return value() if callable(value) else value
    raise AttributeError("openarm_mujoco.v2 exposes no cell XML accessor")


def _merge_duplicate_default_classes(xml_text: str) -> tuple[str, int]:
    """Merge sibling ``<default>`` blocks that share a class name.

    mjlab's scene exporter can emit two ``<default class="robot/main">``
    siblings for this robot -- one carrying the pedestal classes, one the motor
    classes. MuJoCo rejects a repeated class name, so the exported scene will
    not load at all without this. Merging keeps every child of both blocks,
    which is what a single correctly-emitted block would have contained.
    Verified equivalent against the in-mjlab compiled model.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    merged = 0

    def collapse(node: ET.Element) -> None:
        """Splice a <default> child that repeats its parent's class name."""
        nonlocal merged
        changed = True
        while changed:
            changed = False
            for child in list(node):
                if child.tag != "default":
                    continue
                same_as_parent = (
                    node.tag == "default"
                    and child.get("class") is not None
                    and child.get("class") == node.get("class")
                )
                if same_as_parent:
                    index = list(node).index(child)
                    node.remove(child)
                    for offset, grandchild in enumerate(list(child)):
                        node.insert(index + offset, grandchild)
                    for key, value in child.attrib.items():
                        node.attrib.setdefault(key, value)
                    merged += 1
                    changed = True
                    break
        for child in node:
            collapse(child)

    collapse(root)
    # a sibling repeat would also be rejected; fold those too
    for parent in root.iter():
        seen: dict[str, ET.Element] = {}
        for child in list(parent):
            if child.tag != "default" or child.get("class") is None:
                continue
            name = child.get("class")
            if name in seen:
                for grandchild in list(child):
                    seen[name].append(grandchild)
                parent.remove(child)
                merged += 1
            else:
                seen[name] = child

    if merged == 0:
        return xml_text, 0
    return ET.tostring(root, encoding="unicode"), merged


def export_scene_self_contained(task: str, out_dir: Path) -> int:
    """Export the compiled scene and copy every mesh it references.

    mjlab's exporter writes the XML but not its meshes, and the mesh root is
    wherever the task's robot config points -- an absolute path on the machine
    that trained the policy. Copying them makes the directory portable.
    """
    export_scene(ExportSceneCfg(target=task, output_dir=str(out_dir), zip=False))
    xml_path = out_dir / "scene.xml"
    text = xml_path.read_text()
    text, merged = _merge_duplicate_default_classes(text)
    if merged:
        xml_path.write_text(text)
        print(f"merged {merged} duplicate <default> class block(s) so the scene loads")
    meshdir_match = re.search(r'meshdir="([^"]*)"', text)
    meshdir = meshdir_match.group(1) if meshdir_match else "assets"
    refs = sorted(set(re.findall(r'file="([^"]+)"', text)))
    if not refs:
        return 0

    # Resolve the mesh root through the openarm-mujoco package rather than any
    # absolute path, so the export works on a machine that only has the
    # declared dependencies.
    import openarm_mujoco.v2 as openarm_v2

    cell_xml = Path(openarm_mujoco_cell_xml(openarm_v2))
    candidates = [
        cell_xml.parent / "assets",
        cell_xml.parent.parent / "assets",
        Path(str(meshdir)),
    ]
    root = next((c for c in candidates if (c / refs[0]).exists()), None)
    if root is None:
        raise FileNotFoundError(
            f"could not locate the mesh root for {refs[0]!r}; tried "
            f"{[str(c) for c in candidates]}"
        )

    copied = 0
    for rel in refs:
        dst = out_dir / meshdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(root / rel, dst)
            copied += 1
        material = (root / rel).with_suffix(".mtl")
        if material.exists() and not dst.with_suffix(".mtl").exists():
            shutil.copy2(material, dst.with_suffix(".mtl"))
            copied += 1
    return copied


def verify_exported_policy(
    expected: np.ndarray,
    observations: np.ndarray,
    out_dir: Path,
    jit_atol: float = 1e-5,
    onnx_atol: float = 1e-3,
) -> dict[str, float]:
    """Check the two written files reproduce the policy they came from.

    ``expected`` holds the actions the in-memory policy produced for
    ``observations``, one row each. Both files are loaded from disk exactly as
    a deployment would load them and run on the same inputs.

    The observations must come from a rollout rather than from zeros. A
    zero observation is what the ONNX exporter traces with, so it is the one
    input a mis-exported policy is most likely to get right, and checking only
    that would pass on a policy that is wrong everywhere else.

    ONNX gets a looser tolerance than TorchScript because it re-associates
    float arithmetic; TorchScript runs the same kernels and should be exact.

    Returns the worst difference seen for each file. Raises if either is over
    tolerance -- an artifact that does not match is not a usable one.
    """
    import onnxruntime as ort

    jit_policy = torch.jit.load(str(out_dir / "policy.pt"))
    jit_policy.eval()
    session = ort.InferenceSession(
        str(out_dir / "policy.onnx"), providers=["CPUExecutionProvider"]
    )

    with torch.no_grad():
        jit_out = jit_policy(torch.from_numpy(observations)).numpy()
    # The graph is exported with a fixed batch dimension of one, so ONNX has
    # to be fed a row at a time.
    onnx_out = np.concatenate(
        [session.run(None, {"obs": row[None]})[0] for row in observations]
    )

    deltas = {
        "policy.pt": float(np.abs(expected - jit_out).max()),
        "policy.onnx": float(np.abs(expected - onnx_out).max()),
    }
    for name, atol in (("policy.pt", jit_atol), ("policy.onnx", onnx_atol)):
        if not deltas[name] <= atol:
            raise RuntimeError(
                f"{name} does not reproduce the trained policy: actions differ "
                f"by up to {deltas[name]:.4g} over {len(observations)} "
                f"observations, tolerance {atol:g}. The exported file is not "
                f"the policy that was loaded; do not deploy it."
            )
    return deltas


def _rollout_for_verification(runner, wrapped, device: str, steps: int = 8):
    """Actions and observations from a short rollout of the loaded policy."""
    policy = runner.get_inference_policy(device=device)
    obs, _ = wrapped.reset()
    observations, actions = [], []
    with torch.no_grad():
        for _ in range(steps):
            observations.append(
                torch.cat([obs[group] for group in policy.obs_groups], dim=-1)
                .detach()
                .cpu()
            )
            action = policy(obs)
            actions.append(action.detach().cpu())
            obs, _, _, _ = wrapped.step(action)
    return (
        torch.cat(actions).numpy().astype(np.float32),
        torch.cat(observations).numpy().astype(np.float32),
    )


def export(task: str, checkpoint: str, out: str, device: str = "cpu") -> None:
    """Export the policy, its contract, and a self-contained scene."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    runner.export_policy_to_onnx(str(out_dir), "policy.onnx")
    runner.export_policy_to_jit(str(out_dir), "policy.pt")

    obs, _ = wrapped.reset()
    action_term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    meta = {
        "task": task,
        "obs_dim": int(obs["actor"].shape[-1]),
        "action_dim": int(env.action_manager.total_action_dim),
        "actuator_order": [f"robot/{n}" for n in action_term.target_names],
        "action_scale": _to_numpy(action_term._scale).ravel().tolist(),
        "action_offset": _to_numpy(action_term._offset)[0].tolist(),
        "decimation": int(cfg.decimation),
        "timestep": float(cfg.sim.mujoco.timestep),
        "episode_steps": int(
            cfg.episode_length_s / (cfg.decimation * cfg.sim.mujoco.timestep)
        ),
        "default_joint_pos": _to_numpy(robot.data.default_joint_pos)[0].tolist(),
        "soft_joint_pos_limits": _to_numpy(robot.data.soft_joint_pos_limits)[
            0
        ].tolist(),
        "robot_joint_names": list(robot.joint_names),
        "sim": {
            "integrator": str(cfg.sim.mujoco.integrator),
            "cone": str(cfg.sim.mujoco.cone),
            "solver": str(cfg.sim.mujoco.solver),
            "impratio": float(cfg.sim.mujoco.impratio),
            "iterations": int(cfg.sim.mujoco.iterations),
            "tolerance": float(cfg.sim.mujoco.tolerance),
            "ls_iterations": int(cfg.sim.mujoco.ls_iterations),
            "ls_tolerance": float(cfg.sim.mujoco.ls_tolerance),
            "gravity": list(cfg.sim.mujoco.gravity),
        },
    }
    (out_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2))

    expected, observations = _rollout_for_verification(runner, wrapped, device)
    deltas = verify_exported_policy(expected, observations, out_dir)
    env.close()

    copied = export_scene_self_contained(task, out_dir / "scene")
    model = mujoco.MjModel.from_xml_path(str(out_dir / "scene" / "scene.xml"))
    print(
        f"policy      {out_dir}/policy.onnx  ({meta['obs_dim']} obs, {meta['action_dim']} act)"
    )
    print(
        f"verified    matches the loaded policy over {len(observations)} "
        f"observations (policy.pt {deltas['policy.pt']:.2e}, "
        f"policy.onnx {deltas['policy.onnx']:.2e})"
    )
    print(f"contract    {out_dir}/policy_meta.json")
    print(f"scene       {out_dir}/scene/scene.xml  ({copied} mesh files copied)")
    print(f"loads standalone: nq={model.nq} nv={model.nv} ngeom={model.ngeom}")


def main() -> None:
    """Parse arguments and export one policy."""
    parser = argparse.ArgumentParser(
        prog="openarm-mjlab-export-policy", description=__doc__.split("\n")[0]
    )
    parser.add_argument("task", help="registered task id")
    parser.add_argument("checkpoint", help="path to a trained model_*.pt")
    parser.add_argument("out", help="output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    export(args.task, args.checkpoint, args.out, args.device)


if __name__ == "__main__":
    main()
