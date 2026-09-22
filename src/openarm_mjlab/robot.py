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

"""OpenArm Cell constants and entity config.

The robot entity is the full OpenArm Cell from third_party (table, walls, lifter,
both arms) loaded via MjSpec. The right arm and the lifter are frozen at the
demo home pose by deleting their joints/actuators, because mjlab zeroes joint
position targets on episode reset and any actuated-but-uncontrolled joint
would be pulled to 0 instead of holding home.
"""

from pathlib import Path

import mujoco
import openarm_mujoco.v2
import numpy as np

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

OPENARM_CELL_XML: Path = Path(openarm_mujoco.v2.openarm_cell_xml())


def _load_asset_constants() -> tuple[dict[str, float], float]:
    """Home keyframe angles and the table-top height, read from the asset.

    Read from the asset instead of hand-copied so an openarm_mujoco update
    cannot silently diverge from the pose baked and frozen below or from the
    scene heights the task derives its thresholds from.
    """
    model = mujoco.MjSpec.from_file(str(OPENARM_CELL_XML)).compile()
    key = model.key("home")
    home = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): float(
            key.qpos[model.jnt_qposadr[j]]
        )
        for j in range(model.njnt)
    }
    table = model.geom("cell_table_col")
    return home, float(table.pos[2] + table.size[2])


_HOME_ANGLES, TABLE_TOP_Z = _load_asset_constants()
LEFT_JOINT4_HOME = _HOME_ANGLES["openarm_left_joint4"]
# Home is the fully OPEN jaw and the finger range runs down to 0 (fully
# closed), so this doubles as the span a finger action must cover to close
# the gripper. Read from the asset so an openarm_mujoco jaw-range change
# cannot leave the action unable to reach the closed jaw.
LEFT_FINGER_HOME = _HOME_ANGLES["openarm_left_finger_joint1"]

_FROZEN_JOINTS = (
    "openarm_lifter_joint",
    *(f"openarm_right_joint{i}" for i in range(1, 8)),
    "openarm_right_finger_joint1",
    "openarm_right_finger_joint2",
)
_FROZEN_ACTUATORS = (
    "lifter_ctrl",
    *(f"right_joint{i}_ctrl" for i in range(1, 8)),
    "right_finger1_ctrl",
)

# Center of the left gripper opening (centroid of the 8 fingertip collision
# geoms with the jaw closed, i.e. finger qpos 0), in the
# openarm_left_ee_base_link frame. This is where a grasped object's center
# sits, so reach rewards target it instead of the wrist-mounted
# left_ee_control_point site. Anchored to the closed jaw rather than to the
# home pose, because home is the *open* jaw the episode starts from.
LEFT_GRASP_SITE = "left_grasp_site"
_LEFT_GRASP_SITE_POS = (-0.0039, 0.0, -0.1349)

LEFT_FINGERTIP_GEOMS = r"finger_(inner|outer)_left_collision_.*"

# The XML's collision class gives the fingertips condim=3 (no torsional
# friction), so a pinched cube spins freely about the contact normal and slips
# during transport. Mirror mjlab's YAM gripper: condim=6 with small torsional/
# rolling friction and a softer solref. priority=2 outranks the Cell geoms
# (priority=1) so these parameters govern fingertip contact pairs.
FINGERTIP_COLLISION = CollisionCfg(
    geom_names_expr=(LEFT_FINGERTIP_GEOMS,),
    contype=1,
    conaffinity=1,
    condim=6,
    priority=2,
    friction=(1.0, 5e-3, 5e-4),
    solref=(0.01, 1.0),
    disable_other_geoms=False,
)


def _bake_frozen_home_pose(spec: mujoco.MjSpec) -> None:
    """Fold every frozen joint's home angle into the body frame it drives.

    Freezing deletes the joint, which pins it at qpos 0, so a joint with a
    non-zero home angle would freeze in the wrong posture. Pre-rotating the
    owning body by that angle makes the deleted joint hold its home offset
    instead. Applies to the bent right elbow and to the open right jaw.

    Only a hinge anchored at its body origin can be baked as a pure body
    rotation — fail loudly if an openarm_mujoco update introduces anything
    else, rather than freezing the arm in a silently wrong pose.
    """
    for name in _FROZEN_JOINTS:
        angle = _HOME_ANGLES[name]
        if angle == 0.0:
            continue
        joint = spec.joint(name)
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(
                f"frozen joint {name} has home angle {angle} but is "
                f"{joint.type}, and only hinges can be baked into a body frame"
            )
        if not np.allclose(joint.pos, 0.0):
            raise ValueError(
                f"bake assumes the {name} anchor at the body origin, got {joint.pos}"
            )
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.asarray(joint.axis), angle)
        body = joint.parent
        baked = np.zeros(4)
        mujoco.mju_mulQuat(baked, body.quat, quat)
        body.quat = baked


def get_openarm_cell_spec() -> mujoco.MjSpec:
    """Load the OpenArm Cell spec, freezing the right arm for pick & place."""
    spec = mujoco.MjSpec.from_file(str(OPENARM_CELL_XML))

    # The XML's standalone timestep (0.001) is discarded on attach anyway;
    # the scene's SimulationCfg owns the timestep. Reset it to the MuJoCo
    # default to avoid the attach-conflict warning.
    spec.option.timestep = 0.002

    _bake_frozen_home_pose(spec)

    for name in _FROZEN_JOINTS:
        spec.delete(spec.joint(name))
    for name in _FROZEN_ACTUATORS:
        spec.delete(spec.actuator(name))
    # The right finger mimic constraint references deleted joints.
    spec.delete(spec.equality("openarm_right_ee_finger_joint_mimic"))
    # The 19-dim home keyframe no longer matches the reduced model.
    spec.delete(spec.key("home"))

    # Grasp-center site between the left fingertips (see _LEFT_GRASP_SITE_POS).
    spec.body("openarm_left_ee_base_link").add_site(
        name=LEFT_GRASP_SITE,
        pos=_LEFT_GRASP_SITE_POS,
        size=(0.005, 0.005, 0.005),
        rgba=(0.0, 1.0, 1.0, 1.0),
    )
    return spec


INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={
        name: angle
        for name, angle in _HOME_ANGLES.items()
        if name.startswith("openarm_left_") and angle != 0.0
    },
    joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=(
                "openarm_left_joint[1-7]",
                "openarm_left_finger_joint1",
            ),
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_openarm_robot_cfg() -> EntityCfg:
    """Entity config for the reduced OpenArm Cell robot."""
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_openarm_cell_spec,
        articulation=ARTICULATION,
        collisions=(FINGERTIP_COLLISION,),
    )
