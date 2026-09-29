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

"""Pedestal-mounted bimanual OpenArm entity config, shared by the reach/valve/door/drawer/puck/lift tasks.

Unlike the OpenArm Cell used by pick_place (a fixed table+walls+lifter scene
with one arm frozen), this family mounts BOTH arms on a stand-alone pedestal
and actively controls the right arm per task (the left is servo-held at its
default pose via ``mjlab.envs.mdp.actions``' hold pattern in each task's own
``actions`` dict), giving each task room to add its own fixture (a valve,
door, drawer, or free object) on a plain table without the Cell's lifter/
wall geometry in the way.

mjlab owns actuation, so the spec strips the asset's own 16 position
actuators; the entity config below rebuilds them with the same gains the
asset's own actuator classes declare (DM8009/DM4340/DM4310 per the OpenArm
hardware spec), verified directly against the compiled asset rather than
hand-copied.
"""

import mujoco
import openarm_mujoco.v2

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

OPENARM_PEDESTAL_XML = openarm_mujoco.v2.asset_path("pedestal/openarm_pedestal.xml")

EE_SITE_RIGHT = "right_ee_control_point"
EE_SITE_LEFT = "left_ee_control_point"

# The finger pads close ~0.135 m along each EE site's local -z axis; every
# task in this family that reaches for or grasps something targets this
# point (the tool point), not the wrist-mounted site itself, so the gripper
# closes AROUND an object instead of stopping short of it.
GRASP_LOCAL_OFFSET = (0.0, 0.0, -0.135)


def get_bimanual_spec() -> mujoco.MjSpec:
    """Load the pedestal-mounted asset with its own position actuators removed."""
    spec = mujoco.MjSpec.from_file(str(OPENARM_PEDESTAL_XML))
    for act in list(spec.actuators):
        spec.delete(act)
    return spec


# Gains/effort limits copied from the asset's own actuator classes: joints
# 1-2 DM8009 kp=230 kv=2.7 |40 Nm|, joints 3-4 DM4340 kp=190 kv=2.2 |27 Nm|,
# joints 5-7 DM4310/3507 kp=30 kv=1.5 |7 Nm|, finger kp=150 kv=2 |30 N|. Only
# finger_joint1 is actuated per side; finger_joint2 mimics it via equality.
_ACTUATOR_GROUPS = (
    ("openarm_(left|right)_joint[12]", 230.0, 2.7, 40.0),
    ("openarm_(left|right)_joint[34]", 190.0, 2.2, 27.0),
    ("openarm_(left|right)_joint[567]", 30.0, 1.5, 7.0),
    ("openarm_(left|right)_finger_joint1", 150.0, 2.0, 30.0),
)

BIMANUAL_ACTUATORS = tuple(
    BuiltinPositionActuatorCfg(
        target_names_expr=(expr,),
        stiffness=kp,
        damping=kv,
        effort_limit=effort,
    )
    for expr, kp, kv, effort in _ACTUATOR_GROUPS
)

# Home pose: elbows bent at 1.5708 rad, everything else at 0.
BIMANUAL_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "openarm_(left|right)_joint4": 1.5708,
        "openarm_(left|right)_joint[12356]": 0.0,
        "openarm_(left|right)_joint7": 0.0,
        "openarm_(left|right)_finger_joint[12]": 0.0,
    },
    joint_vel={".*": 0.0},
)

BIMANUAL_ARTICULATION = EntityArticulationInfoCfg(
    actuators=BIMANUAL_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

# Same convention mjlab's own yam config uses: scale = 0.25 * effort / stiffness.
BIMANUAL_ACTION_SCALE: dict[str, float] = {
    expr: 0.25 * effort / kp for expr, kp, _kv, effort in _ACTUATOR_GROUPS
}


def get_bimanual_robot_cfg() -> EntityCfg:
    """Entity config for the pedestal-mounted bimanual OpenArm, both arms actuated."""
    return EntityCfg(
        init_state=BIMANUAL_HOME,
        spec_fn=get_bimanual_spec,
        articulation=BIMANUAL_ARTICULATION,
    )
