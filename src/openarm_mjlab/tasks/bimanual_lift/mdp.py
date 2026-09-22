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

"""MDP terms for the OpenArm bimanual lift task.

Bimanual lift task MDP: BOTH arms must squeeze-grip their own end of a

long bar and raise it TOGETHER.

This is the platform's first task where the LEFT arm is not parked via
HoldDefaultPositionActionCfg -- see
bimanual_lift_env_cfg.py's module docstring for the scene geometry and the
collision-avoidance reasoning (both grip points sit well outside the
centerline zone where the classical sibling project documented the two
close-mounted arms' upper links colliding).

Grip mechanics are copied verbatim from lift's hard-won, proven mechanism: both bar ends are lift's exact block geometry
(50 mm box, mass 0.05, friction (2.5, 0.1, 0.01)), both fingers are raw
effort-controlled with the same stiffened joint-limit spec, and the reward
skeleton (reach / pinch / partial_pinch / pinch_streak / lift_rate capped at
target / held_high / success bonus / descent+overshoot penalties) is the
same shape lift shipped as `the single-arm lift task`.

THE ONE GENUINELY NEW DESIGN QUESTION: how to define "together" so the task
can't be solved by two independent single-arm lifts that happen to both
succeed. Every reward/termination term below that credits HEIGHT or PINCH
DURATION composes the two arms' signals with min(), not sum() or average():

  - together_lift_rate_reward tracks progress of min(right_height,
    left_height) -- a virtual "height of the lift" pinned to whichever end
    is behind. If the right arm surges ahead while the left arm lags, the
    surge earns nothing further until the left end catches up to that
    level. sum()/average() would let a strong single-arm lift compensate
    for a barely-lifted other end, which is exactly the "two independent
    lifts" failure mode this task is supposed to rule out.
  - together_pinch_streak_reward takes min(streak_right, streak_left):
    either arm losing contact resets ITS streak to 0, which immediately
    drags the paired reward back to 0 too -- streaks can only grow while
    BOTH grips are simultaneously held, not merely each held at some point.
  - level_reward is a dense, continuously-available companion (gated on
    both grippers actually holding their end, per the note below) that
    additionally discourages the bar tipping between the two height
    checks above land on it.
  - lifted_together (the success termination) requires both ends
    independently inside the bounded height window (an earlier attempt/17's overshoot
    lesson -- the project notes, "lifted_target ... has no upper bound")
    AND a tighter LEVEL_TOLERANCE on their height difference than the
    window width alone implies, so "both ends happened to be somewhere in
    the 30 mm window at the same instant, 25 mm apart" cannot pass.

reach_bar_end_reward and partial_pinch_reward stay UNGATED per arm (dense,
summed): gating early shaping on the OTHER arm's progress would recreate
the "gradient desert" an earlier attempt/9 already diagnosed (reach sitting near 1.0
while pinch stays at 0.000) -- exploration needs to be able to discover
each arm's own reach/one-pad-contact independently before the coordination
requirement can mean anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

from ...robot_bimanual import GRASP_LOCAL_OFFSET
from ...common_mdp import both_pads_on_block

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# Table/height convention matches lift's TABLE_TOP_Z/BLOCK_START exactly
# (same table entity, reused unmodified -- see get_lift_table_spec import
# in bimanual_lift_env_cfg.py), declared independently here since the bar
# is a different object, not lift's block.
BAR_START = (0.30, 0.0, 0.43)
TABLE_TOP_Z = 0.40

# Local (bar-body-frame) offset of each grippable end from the bar's
# center. Magnitude 0.16 m is not an arbitrary "order of 250-350mm bar"
# guess -- it is copied directly from openarm_control/bimanual.py's
# ParallelSort task (`right_jobs` pick_xy=(0.18, -0.16), simultaneously
# with `left_jobs` at (0.18, +0.16)), the classical sibling project's own
# PROVEN-safe simultaneous-bilateral-reach coordinate: both arms
# genuinely operate at once at this separation without the upper-arm
# collision the project's README documents for centered/shared targets
# ("Two close-mounted 7-DOF arms collide when *both* reach over one
# centred object"). It is independently corroborated by mjlab's own
# right-arm workspace box (reach/mdp.py TARGET_LO/TARGET_HI: y in
# [-0.35, -0.05]) -- -0.16 sits comfortably mid-range, nowhere near
# either the centerline or the box edge. See the env cfg module
# docstring for the full reasoning and the numbers checked.
END_Y_OFFSET = 0.16
RIGHT_END_OFFSET = (0.0, -END_Y_OFFSET, 0.0)
LEFT_END_OFFSET = (0.0, END_Y_OFFSET, 0.0)

TARGET_LIFT = 0.12  # m above start -- unchanged from lift: same actuators,
# same per-arm load order of magnitude (each end's local grip only has to
# resist roughly its own share of the bar's weight), no evidence to retune.
MAX_LIFT_RATE = 0.3  # m/s
SETTLED_SPEED = 0.10  # m/s bar speed for "held"
HEIGHT_TOLERANCE = 0.03  # m -- success window is TARGET_LIFT..+30mm, per end
# Tighter than HEIGHT_TOLERANCE on purpose: if it merely matched, it would
# be a no-op (both ends already being inside the same 30mm window bounds
# their difference to <=30mm anyway). At 15mm it is a genuine, independent
# honesty gate -- e.g. right at +0mm-over-target and left at +30mm-over-
# target both individually pass the window but are 30mm apart, which this
# catches and the window check alone would not.
LEVEL_TOLERANCE = 0.015  # m
PINCH_STREAK_CAP = 25.0  # steps, same derivation as lift's (~0.5s, roughly
# the time a real lift to TARGET_LIFT at MAX_LIFT_RATE would take).

# An earlier attempt showed that individual_lift_rate_reward (an UNGATED per-arm bootstrap)
# still never fired either -- neither arm ever discovers ANY upward
# motion, ruling out coordination-gating as the specific bottleneck.
# Mirrors lift's own history: lift needed reset_held_high (a curriculum
# spawning some episodes already partway lifted) before genuine lifting
# emerged, because pure from-scratch exploration of "squeeze AND raise"
# is a narrow needle regardless of reward shape. Unlike lift, this task
# had no IK-derived reference pose to build that curriculum on (both
# arms use the generic HOME_KEYFRAME) -- solved for one here, using the
# same OpenArmKinematics solver openarm_control/kinematics.py already
# uses for the classical stack's own bimanual work, seeded at
# HOME_KEYFRAME, targeting each end raised by HELD_HIGH_RAISE (0.08m,
# matching lift's own choice -- below TARGET_LIFT so the policy still
# has to finish the climb, not just hold at the goal). Converged
# cleanly (error 0.046mm, first seed). Verified the classical stack's
# own documented y=0-mirror identity (config.py's MIRROR_R2L, "verified
# 0.000mm/0.000deg") holds for this pose too: solving the LEFT arm
# independently and via q_left=MIRROR_R2L*q_right agreed to 1.4e-17
# (floating-point exact) -- both are recorded below rather than only
# trusting the mirror shortcut.
HELD_HIGH_RAISE = 0.08  # m above BAR_START, matches lift's own reset_held_high
# 2026-08-28 RE-SOLVED WITH THE ORIENTATION CONSTRAINED. The previous
# values came from a POSITION-ONLY IK solve (see the "error 0.046mm" note
# above -- millimetres, no angular term), so their wrist orientation was an
# arbitrary artifact of the solver's null-space. Measured consequence
# (probe_grasp_twist.py): the gripper rotates 57.6 deg mean along the path
# from the policy's natural table grasp to this default pose. Since the
# default pose IS the zero-action pose, that made a lift -- which "should"
# be a pure relaxation toward default -- into a motion that twists the bar
# out of the fingers, and it is the measured root cause of the never-lifts
# plateau. Rewarding alignment instead was tried and regressed
# (tested and reverted), because it fights kinematics rather than
# fixing the reference.
#
# These are re-solved by solve_held_high_oriented.py: the orientation the
# policy ACTUALLY adopts when grasping at the table, held fixed, at the
# grasp position raised by HELD_HIGH_RAISE. Converged to 0.007mm position
# AND 0.000 deg orientation error for both arms, seeded at the measured
# grasp configuration so the solution stays in the same kinematic branch.
# Default and grasp now differ by HEIGHT ALONE, so relaxing toward default
# is a genuine vertical lift.
#
# NOTE: unlike the old values these are NOT an exact y-mirror pair. They are
# derived from the policy's own grasp, which is not perfectly symmetric, so
# the classical stack's MIRROR_R2L identity no longer applies here -- each
# side is solved independently and that is deliberate.
HELD_HIGH_RIGHT_Q = (
    0.1060,
    0.5567,
    -0.0246,
    1.4836,
    0.5684,
    0.0565,
    1.0371,
)
HELD_HIGH_LEFT_Q = (
    -0.2007,
    -0.4367,
    0.4452,
    1.7822,
    -0.5292,
    -0.1500,
    0.6359,
)
# An earlier attempt showed that a FIXED 50/50 mix of
# "cold table start" and "already 67% up" never transferred cold-start
# competence -- checked every checkpoint (500-2999), all 0%. The
# curriculum episodes DID succeed (that's what drove the misleadingly
# strong aggregate training-log numbers), just never generalized.
# an earlier attempt then tried an ANNEALED version (spawn raise/probability
# decaying linearly to 0 by HELD_HIGH_ANNEAL_STEPS) reasoning from
# reverse-curriculum literature (Florensa et al. 2017) -- also 0% at
# every checkpoint.
#
# 2026-08-23 root cause found (see the project notes, dated entry):
# BOTH prior attempts moved the ROBOT'S JOINTS via write_joint_state_to_
# sim while leaving the arm action terms' use_default_offset anchored to
# the plain HOME_KEYFRAME (elbow-only neutral pose) -- because
# JointPositionAction snapshots its _offset ONCE, at env-build time, from
# entity.data.default_joint_pos, a per-episode write_joint_state_to_sim
# call can move the PHYSICAL joint but never touches that offset. Direct
# probe (scratchpad probe_yankback.py, zero action from a forced
# held-high reset): right/left joint4 collapsed from 1.83 back toward
# HOME_KEYFRAME's 1.57 within 3 steps, and h_together crashed from
# 100mm to 30mm in 9 steps (~180ms) -- numerically the SAME "zero action
# yanks the arm home, block drops in 160ms" signature lift's own review
# #3 already diagnosed and fixed for the single-arm task (LIFT_HOME).
# Control (same probe, offset monkey-patched to the held-high pose):
# height held near 90-100mm the entire 40 steps, confirming the
# actuator/gain setup itself is fine -- the bug is specifically the
# offset/reset mismatch. This also explains an earlier attempt's own
# unexplained late-training collapse: with maxh pre-set to the reset
# height, a fall-then-partial-recover cycle earns ZERO new lift_rate
# reward (new-progress-only, gated on the OLD high-water mark), so the
# curriculum-assisted episodes were never actually rewarding the
# intended "finish the climb" behavior, just the yank-and-recover
# transient.
#
# Fix: mirror lift's OWN proven recipe exactly rather than re-inventing
# a bimanual-specific curriculum shape. lift's LIFT_HOME IS its
# HELD_HIGH_POSE (byte-identical joint values, per lift/mdp.py's own
# comment: "Arm already at the held-high DEFAULT... only pin the
# fingers... and place the block") -- the robot's DEFAULT pose is the
# grip-ready pose, so zero action holds it, cold-table-start episodes
# already hover right above the object, and reset_held_high only ever
# needs to move the OBJECT, never the arm. Applied the same trick here:
# get_bimanual_lift_robot_cfg's init_state now uses HELD_HIGH_RIGHT_Q/
# LEFT_Q directly (already IK-solved, already verified <0.05mm error --
# see below), and reset_bar_held_high (further down) no longer touches
# the robot's joints at all. The old TABLE_RIGHT_Q/LEFT_Q (arm-at-table-
# height IK solve, used only for the now-removed anneal's interpolation)
# and the anneal schedule are removed as unneeded complexity fixing a
# problem (permanent easy-mode exploitation) that was itself confounded
# by this bug -- lift's own shipped config uses a FIXED, low probability
# (0.15, see lift_env_cfg.py's reset_held_high wiring), which is the
# only actually-proven ratio available, so that's what's reused here
# rather than guessing a new one.
HELD_HIGH_PROBABILITY = 0.15
# Same block geometry as lift (verbatim), so lift's own proven squeeze
# value transfers directly for the right finger; left is sign-mirrored
# to match its mirrored joint range (0..0.7854 vs right's -0.7854..0,
# confirmed in the XML -- see bimanual_lift_env_cfg.py's own actuator
# comments).
RIGHT_FINGER_SQUEEZE = -0.22
LEFT_FINGER_SQUEEZE = 0.22


def bar_pos_w(env, asset_cfg) -> torch.Tensor:
    """Return the bar's world position."""
    bar: Entity = env.scene[asset_cfg.name]
    return bar.data.root_link_pos_w


def bar_speed(env, asset_cfg) -> torch.Tensor:
    """Return the bar's linear speed."""
    bar: Entity = env.scene[asset_cfg.name]
    return torch.linalg.norm(bar.data.root_link_vel_w[:, :3], dim=-1)


def bar_end_pos_w(env, asset_cfg, local_offset) -> torch.Tensor:
    """Return the world position of one bar end.

    World position of one bar end: root pose + the end's LOCAL (body-

    frame) offset, rotated by the bar's current orientation. Rotating the
    offset (not just adding it in world frame) is what makes end_height
    correctly reflect a TIPPED bar -- if only one end is genuinely held,
    the free end drifts/tips rather than translating in lock-step, and this
    keeps tracking its true world height through that rotation.
    """
    bar: Entity = env.scene[asset_cfg.name]
    pos_w = bar.data.root_link_pos_w
    quat_w = bar.data.root_link_quat_w
    offset = torch.tensor(local_offset, device=pos_w.device).expand_as(pos_w)
    return pos_w + quat_apply(quat_w, offset)


def end_height(env, asset_cfg, local_offset) -> torch.Tensor:
    """Return one bar end's height above its start."""
    z = bar_end_pos_w(env, asset_cfg, local_offset)[:, 2]
    return torch.clamp(z - BAR_START[2], min=0.0)


def tool_pos_w(env, robot_cfg) -> torch.Tensor:
    """Return the gripper tool point in world coordinates."""
    robot: Entity = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
    ee_quat_w = robot.data.site_quat_w[:, robot_cfg.site_ids].squeeze(1)
    offset = torch.tensor(GRASP_LOCAL_OFFSET, device=ee_pos_w.device).expand_as(
        ee_pos_w
    )
    return ee_pos_w + quat_apply(ee_quat_w, offset)


def tool_to_end_obs(env, robot_cfg, asset_cfg, local_offset) -> torch.Tensor:
    """Return the tool-to-bar-end vector observation."""
    robot: Entity = env.scene[robot_cfg.name]
    vec_w = bar_end_pos_w(env, asset_cfg, local_offset) - tool_pos_w(env, robot_cfg)
    return quat_apply(quat_inv(robot.data.root_link_quat_w), vec_w)


def reach_bar_end_reward(
    env, std: float, robot_cfg, asset_cfg, local_offset
) -> torch.Tensor:
    """Return dense per-arm reach shaping toward one bar end.

    Dense per-arm reach shaping, deliberately NOT gated on the other

    arm's progress -- see module docstring.
    """
    vec = tool_to_end_obs(env, robot_cfg, asset_cfg, local_offset)
    d2 = torch.sum(torch.square(vec), dim=-1)
    return torch.exp(-d2 / std**2)


def pinch_reward(env, sensor_name: str) -> torch.Tensor:
    """Return 1.0 where both finger pads touch the bar end."""
    return both_pads_on_block(env, sensor_name).float()


def partial_pinch_reward(env, sensor_name: str) -> torch.Tensor:
    """Return 0, 0.5 or 1.0 for how many of the two pads are touching.

    A graded predecessor to the binary two-pad gate: the binary AND gives no
    signal for "one pad landed, still working on the other", a gradient
    desert that stalls discovery of the squeeze. Purely additive -- success
    and termination still require a genuine two-pad pinch.
    """
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    per_pad = (found.view(env.num_envs, 2, -1).amax(dim=-1) > 0).float()
    return per_pad.mean(dim=-1)


def pinch_obs(env, sensor_name: str) -> torch.Tensor:
    """Return the observation wrapper for the two-pad pinch gate."""
    return both_pads_on_block(env, sensor_name).float().unsqueeze(-1)


def both_ends_gripped(env, sensor_right: str, sensor_left: str) -> torch.Tensor:
    """Return True where both grippers hold their own bar end.

    Both grippers independently satisfy lift's proven two-pad pinch gate

    on their own end. Reused directly (not reimplemented) from lift.mdp:
    both_pads_on_block is fully generic given a sensor_name -- it carries no
    hidden per-task state, so calling it twice (once per side) is safe.
    """
    return both_pads_on_block(env, sensor_right) & both_pads_on_block(env, sensor_left)


def _streak_buffers(env) -> dict[str, torch.Tensor]:
    if not hasattr(env, "_bimanual_lift_streaks"):
        env._bimanual_lift_streaks = {
            "right": torch.zeros(env.num_envs, device=env.device),
            "left": torch.zeros(env.num_envs, device=env.device),
        }
    return env._bimanual_lift_streaks


# 2026-08-26 grasp-geometry fix. Measured root cause of the never-lifts
# plateau (probe_grasp_geometry.py, full trail in the project notes): at
# the instant a two-ended grasp forms, the TABLE grasp contacts the end
# block at a mean vertical offset of +18.4mm, while the held-high (pinned)
# grasp -- the one that demonstrably survives a lift -- sits at -2.4mm,
# essentially centred. The block's half-height is 30mm, so the table grasp
# clamps ~61% of the way up toward the TOP edge, and only about half as
# deep (0.9mm vs 1.9mm penetration). That is a peel-off geometry: under
# lift acceleration the block pivots about the shallow high contact and
# escapes, which is exactly what the scripted-lift probe measured (force
# decaying to 0.00N by 40mm of lift, versus 30.6N retained from the
# held-high grasp under an identical command).
#
# It is NOT a force problem -- static holding force is essentially equal
# in both cases (29.4N vs 32.8N) against a ~1.2N bar, a hypothesis that
# was measured and refuted before this one. It is purely WHERE the fingers
# land, and nothing in the reward ever distinguished that: `pinch_reward`
# is `both_pads_on_block(...)`, binary presence of any contact, so a
# shallow grab at the top edge scored exactly what a centred, deep grasp
# scored. The arm's default pose is above the bar, so descending and
# catching the top is the first thing that satisfies the sensor, and the
# policy had no reason to look for anything better.
# Sized so the CURRENT (failing) grasp sits mid-curve, not in the flat
# tail. Measured table-grasp offset is +18.4mm; at std=0.012 that scores
# 0.005, i.e. the policy would start in a gradient desert -- the exact
# failure mode lift's own r3b binary-gate fix had to remove. At 0.022 the
# same grasp scores ~0.50, a centred grasp 1.0, and the held-high
# reference ~0.78, so there is real gradient across the whole range the
# policy actually operates in.
GRASP_CENTRE_STD = 0.022  # m
GRASP_QUALITY_FLOOR = 0.4  # keep some income for ANY grasp (see below)


# 2026-08-26, measured (probe_grasp_twist.py): along the joint-space path
# from the grasp pose toward the default pose, the gripper ROTATES by 57.6
# deg on average (max 74.9). That is enough to wrench the bar out of a
# parallel-jaw grasp, and it explains the two failure modes of every
# scripted lift attempted: a slow rise gives the twist time to work the bar
# loose (it slips away by 25mm), while a fast rise outruns the twist but
# arrives ballistically (whenever both ends are inside the success window,
# grip has already been lost -- win_grip measured at exactly 0.000).
#
# The cause is that the policy's self-established table grasp adopts a wrist
# orientation ~58 deg away from the one the default/held-high pose uses. So
# "lift" is NOT the simple relaxation toward default that the action space
# would otherwise make it -- the policy would have to learn a coordinated
# joint motion that raises the bar WHILE actively counter-rotating the
# wrist, which is far harder than relaxing toward zero action.
#
# These are the EE orientations at the default (held-high) pose, measured
# directly from the compiled scene. Rewarding the grasp to align with them
# makes lifting a relaxation again.
DEFAULT_EE_QUAT_RIGHT = (-0.6449, 0.0080, 0.7638, -0.0255)
DEFAULT_EE_QUAT_LEFT = (-0.6488, 0.0007, 0.7587, 0.0579)
# Sized so the CURRENT ~58 deg (1.01 rad) misalignment scores ~0.5 rather
# than sitting in the flat tail of the Gaussian. An earlier grasp-centring
# term had exactly that gradient-desert bug at too tight a width, so the
# width here is set from the measurement rather than guessed.
GRASP_ORIENT_STD = 1.2  # rad


def grasp_orientation_reward(env, robot_cfg, target_quat) -> torch.Tensor:
    """Reward the gripper for holding the default pose's orientation.

    Aligning the grasp with the held-high orientation is what makes a lift a
    pure relaxation toward the default pose, instead of a motion that twists
    the bar out of the fingers on the way up.
    """
    robot: Entity = env.scene[robot_cfg.name]
    q = robot.data.site_quat_w[:, robot_cfg.site_ids].squeeze(1)
    tq = torch.tensor(target_quat, device=q.device, dtype=q.dtype).expand_as(q)
    dq = quat_mul(q, quat_inv(tq))
    ang = 2.0 * torch.acos(dq[:, 0].abs().clamp(max=1.0))
    return torch.exp(-((ang / GRASP_ORIENT_STD) ** 2))


def grasp_centring(env, sensor_name: str, asset_cfg, local_offset) -> torch.Tensor:
    """0..1 measure of how vertically centred the grasp is on its end block.

    Contact points are expressed in the BAR's own frame (not world) so this
    stays correct when the bar tips, matching bar_end_pos_w's reasoning.
    """
    bar: Entity = env.scene[asset_cfg.name]
    sensor_data = env.scene[sensor_name].data
    pos = sensor_data.pos
    if pos is None:
        return torch.ones(env.num_envs, device=env.device)
    end_w = bar_end_pos_w(env, asset_cfg, local_offset)
    rel_w = pos.mean(dim=1) - end_w
    rel_b = quat_apply(quat_inv(bar.data.root_link_quat_w), rel_w)
    return torch.exp(-((rel_b[:, 2] / GRASP_CENTRE_STD) ** 2))


def quality_pinch_reward(
    env, sensor_name: str, asset_cfg, local_offset
) -> torch.Tensor:
    """Binary pinch, scaled by how centred the grasp is.

    TESTED AND REGRESSED, kept for the record and for diagnostics -- NOT
    wired into the reward dict. `an earlier attempt` used this (plus
    the same weighting on the streak term) and both-ends-gripped collapsed
    from 0.93 to 0.000: multiplying the two terms that actually taught
    gripping cut up to 1.5/step of income, and the policy abandoned the
    behaviour rather than improving it. The floor of 0.4 was not enough.
    The working approach instead leaves grip income alone and fixes the
    saturated REACH term (see reach_fine_* in the env cfg), which is what
    actually determines where the fingers land before the grasp forms.

    Deliberately a MULTIPLIER on the existing pinch income rather than a new
    additive term: this task's whole difficulty is that grip-and-sit income
    already outbids lifting (see the success-bonus arithmetic in
    bimanual_lift_env_cfg.py), so adding another term payable while camping
    would make that worse. Redistributing the SAME budget toward good grasps
    adds the missing gradient without raising the camping ceiling at all --
    a bad grasp now earns strictly less than it used to, a centred one earns
    what it always did.

    The floor keeps a poor grasp worth something, so the already-solid
    gripping behaviour (0.85-0.97 both-ends-gripped, hard-won via the
    squeeze-scale fix) is shaped rather than destabilised.
    """
    gripped = both_pads_on_block(env, sensor_name).float()
    q = grasp_centring(env, sensor_name, asset_cfg, local_offset)
    return gripped * (GRASP_QUALITY_FLOOR + (1.0 - GRASP_QUALITY_FLOOR) * q)


def together_pinch_streak_reward(
    env, sensor_right: str, sensor_left: str
) -> torch.Tensor:
    r"""Return a reward for holding BOTH grips simultaneously over time.

    Paired version of the single-arm pinch-streak reward. An earlier attempt showed that a policy that pecked -- 2-step grab/release cycles --
    scored identically to a genuine hold under the old binary pinch/
    partial_pinch rewards; sustaining duration had to be made to pay
    directly). Two independent per-arm streaks are tracked (each resets to
    0 the instant THAT arm's own contact breaks), and the credited reward is
    min(streak_right, streak_left) -- so a policy that holds the right grip
    for 20 steps while pecking the left grip for 2 gets credit for only a
    2-step streak, not 20. This is what actually enforces "held together,"
    not just "each held for a while, not necessarily overlapping.\"
    """
    streaks = _streak_buffers(env)
    gripped_r = both_pads_on_block(env, sensor_right)
    gripped_l = both_pads_on_block(env, sensor_left)
    sr, sl = streaks["right"], streaks["left"]
    sr[gripped_r] += 1.0
    sr[~gripped_r] = 0.0
    sl[gripped_l] += 1.0
    sl[~gripped_l] = 0.0
    paired = torch.minimum(sr, sl)
    return torch.clamp(paired / PINCH_STREAK_CAP, 0.0, 1.0)


def _max_together_height(env) -> torch.Tensor:
    if not hasattr(env, "_bimanual_lift_max_h"):
        env._bimanual_lift_max_h = torch.zeros(env.num_envs, device=env.device)
    return env._bimanual_lift_max_h


def together_lift_rate_reward(
    env, sensor_right: str, sensor_left: str, asset_cfg
) -> torch.Tensor:
    """Return a reward for new upward progress of the lower bar end.

    Paired version of the single-arm lift-rate reward. a plain

    rise-rate reward is bounce-farmable; new-progress-only pays each mm
    once, and per an earlier attempt's fix, credited progress is capped at TARGET_LIFT
    so nothing rewards climbing past the intended window).

    The coordination step: progress is measured on h_together =
    min(right_height, left_height), a single virtual "height of the lift"
    pinned to whichever end is behind, THEN the same new-progress-only/
    capped-at-target logic lift already proved runs on that one scalar. If
    the right end is 80mm up and the left end is 20mm up, h_together is
    20mm -- the right arm's additional height earns nothing further until
    the left arm brings its end up to match. This is what forces the
    policy to bring the lagging arm along rather than banking progress on
    whichever arm happens to be easier.
    """
    h_r = end_height(env, asset_cfg, RIGHT_END_OFFSET)
    h_l = end_height(env, asset_cfg, LEFT_END_OFFSET)
    h_together = torch.minimum(h_r, h_l)
    maxh = _max_together_height(env)
    h_capped = torch.clamp(h_together, max=TARGET_LIFT)
    maxh_capped = torch.clamp(maxh, max=TARGET_LIFT)
    new = torch.clamp(h_capped - maxh_capped, min=0.0)
    maxh.copy_(torch.maximum(maxh, h_together))
    rate = torch.clamp(new / env.step_dt, 0.0, MAX_LIFT_RATE) / MAX_LIFT_RATE
    gate = both_ends_gripped(env, sensor_right, sensor_left).float()
    return rate * gate


def _max_individual_height(env, side: str) -> torch.Tensor:
    attr = f"_bimanual_lift_max_h_{side}"
    if not hasattr(env, attr):
        setattr(env, attr, torch.zeros(env.num_envs, device=env.device))
    return getattr(env, attr)


def individual_lift_rate_reward(
    env, sensor_name: str, asset_cfg, local_offset, side: str
) -> torch.Tensor:
    """Bootstrapping companion to together_lift_rate_reward.

    2026-08-20, first real training run's postmortem: by iteration ~200-500 the policy
    fully solved reach/grip/streak/level, then went completely flat for
    2500+ more iterations -- together_lift_rate_reward and
    together_held_high_reward stayed at exact zero the whole time. Root
    cause: both are gated on h_together = min(right,left), so from a
    cold start at height 0, BOTH arms' exploration noise has to
    coincidentally push upward in the SAME step before any lift reward
    appears at all -- a much narrower needle than lift's own single-arm
    problem ever was.

    This mirrors the reasoning already used for reach_bar_end_reward and
    partial_pinch_reward (module docstring: deliberately UNGATED per arm
    so "exploration needs to be able to discover each arm's own reach/
    one-pad-contact independently before the coordination requirement
    can mean anything") -- extended one step further, to the lift motion
    itself, which previously had no individual/ungated version to
    bootstrap from. Gated only on THIS arm's own grip (not both), so
    each arm can discover "lifting my own end is possible" on its own.

    Weighted much lower than together_lift_rate_reward in the env cfg
    (kept as the dominant channel) specifically so this can only ever
    help escape the flat-zero plateau, not replace the coordination
    requirement -- genuine success still requires together_* to fire.
    """
    h = end_height(env, asset_cfg, local_offset)
    maxh = _max_individual_height(env, side)
    h_capped = torch.clamp(h, max=TARGET_LIFT)
    maxh_capped = torch.clamp(maxh, max=TARGET_LIFT)
    new = torch.clamp(h_capped - maxh_capped, min=0.0)
    maxh.copy_(torch.maximum(maxh, h))
    rate = torch.clamp(new / env.step_dt, 0.0, MAX_LIFT_RATE) / MAX_LIFT_RATE
    gate = both_pads_on_block(env, sensor_name).float()
    return rate * gate


def together_held_high_reward(
    env, sensor_right: str, sensor_left: str, asset_cfg
) -> torch.Tensor:
    """Return a graded reward for holding both ends near the target height.

    Paired version of the single-arm held-high reward: graded height-holding income, avoiding the binary-gate gradient desert lift's r3b fix
    addressed). Uses the same h_together = min(...) composition as
    together_lift_rate_reward, for the same reason.
    """
    h_r = end_height(env, asset_cfg, RIGHT_END_OFFSET)
    h_l = end_height(env, asset_cfg, LEFT_END_OFFSET)
    h_together = torch.minimum(h_r, h_l)
    frac = torch.clamp(h_together / TARGET_LIFT, 0.0, 1.0)
    return frac * both_ends_gripped(env, sensor_right, sensor_left).float()


def level_reward(
    env, sensor_right: str, sensor_left: str, asset_cfg, std: float = 0.03
) -> torch.Tensor:
    """Return a reward for keeping the bar horizontal while gripped.

    Dense Gaussian shaping on the height DIFFERENCE between the two

    ends, encouraging the bar to stay roughly horizontal throughout the
    climb (not just at the final success check, where lifted_together's
    LEVEL_TOLERANCE already gates it).

    Gated on both_ends_gripped for the same reason review finding 5 (drawer:
    "with randomized initial opening, absolute-opening rewards pay free
    income at spawn") flags absolute/state-based terms with no engagement
    requirement: at rest, both ends sit at height 0 and are trivially
    "level" (kernel = 1.0) before any grasping has happened at all.

    2026-08-26: that grip gate turned out NOT to be sufficient, and this
    term was a real camping-income bug -- the exact class it was written
    to avoid. A bar resting flat ON THE TABLE has h_r == h_l == 0, so the
    kernel is a perfect 1.0, and the grip gate is trivially satisfied by
    closing both grippers on it without lifting at all. Measured directly
    in `an earlier attempt`'s training log: `Episode_Reward/level`
    climbed to 0.847 while the hardened eval of that same policy showed a
    max bar height of 0.3mm -- i.e. ~0.85/step of steady income, for the
    whole episode, for gripping a bar and leaving it exactly where it
    was. That is a meaningful share of the ~6/step camping income this
    task's success bonus has to outbid, and unlike reach/pinch (which at
    least pay for genuinely necessary sub-skills) this one paid for a
    literal non-behavior.

    Fix: scale by the achieved height fraction, the same shape
    together_held_high_reward already uses. At table height this is 0 (no
    free income); at TARGET_LIFT it is full -- preserving the intended
    "stay horizontal DURING the climb" shaping while removing the
    stay-put income. Deliberately a smooth scale rather than a hard
    height threshold, which would reintroduce the binary-gate gradient
    desert lift's own r3b fix had to remove.
    """
    h_r = end_height(env, asset_cfg, RIGHT_END_OFFSET)
    h_l = end_height(env, asset_cfg, LEFT_END_OFFSET)
    kernel = torch.exp(-(((h_r - h_l) / std) ** 2))
    frac = torch.clamp(torch.minimum(h_r, h_l) / TARGET_LIFT, 0.0, 1.0)
    return kernel * frac * both_ends_gripped(env, sensor_right, sensor_left).float()


def bar_descent_penalty(env, asset_cfg) -> torch.Tensor:
    """Return a penalty for lowering the bar.

    Anti-pump, matching lift's block_descent_penalty exactly: lowering

    the bar is never free.
    """
    bar: Entity = env.scene[asset_cfg.name]
    return torch.clamp(-bar.data.root_link_vel_w[:, 2], min=0.0)


def together_overshoot_penalty(env, asset_cfg) -> torch.Tensor:
    r"""Return a penalty for raising the bar past the target window.

    Paired version of the single-arm overshoot penalty. capping lift_rate's income stopped REWARDING overshoot but left it
    reward-neutral, and median max-height stayed ~658mm anyway; this makes
    exceeding the window actively costly). Fires on h_together = min(...),
    so a bar tipped up on only one side (the other end still near the
    table) never counts as "the coordinated lift overshooting.\"
    """
    h_r = end_height(env, asset_cfg, RIGHT_END_OFFSET)
    h_l = end_height(env, asset_cfg, LEFT_END_OFFSET)
    h_together = torch.minimum(h_r, h_l)
    return torch.clamp(h_together - (TARGET_LIFT + HEIGHT_TOLERANCE), min=0.0)


def together_success_bonus(env) -> torch.Tensor:
    """Fire once on the successful-lift termination step."""
    return env.termination_manager.get_term("lifted_together").float()


def lifted_together(
    env, sensor_right: str, sensor_left: str, asset_cfg
) -> torch.Tensor:
    """Success: BOTH ends independently inside the bounded TARGET_LIFT..

    +HEIGHT_TOLERANCE window (an earlier attempt/17's overshoot-termination fix,
    applied per end rather than lift's single scalar -- the project notes:
    an unbounded `height >= TARGET_LIFT` check let a genuinely-lifting
    policy sail hundreds of mm past the intended target since nothing
    penalized going higher and the one-time success bonus dwarfed the
    forgone shaping reward), settled, both grips genuinely holding, AND
    level (see LEVEL_TOLERANCE's docstring above for why this is a real,
    non-redundant additional check at these parameter values, not implied
    by the window bound alone).
    """
    h_r = end_height(env, asset_cfg, RIGHT_END_OFFSET)
    h_l = end_height(env, asset_cfg, LEFT_END_OFFSET)
    in_window_r = (h_r >= TARGET_LIFT) & (h_r <= TARGET_LIFT + HEIGHT_TOLERANCE)
    in_window_l = (h_l >= TARGET_LIFT) & (h_l <= TARGET_LIFT + HEIGHT_TOLERANCE)
    level = (h_r - h_l).abs() < LEVEL_TOLERANCE
    slow = bar_speed(env, asset_cfg) < SETTLED_SPEED
    gripped = both_ends_gripped(env, sensor_right, sensor_left)
    return in_window_r & in_window_l & level & slow & gripped


def bar_fell(env, asset_cfg) -> torch.Tensor:
    """Return True where the bar has dropped below the floor threshold.

    Fell if the bar's center OR either end drops below the floor

    threshold (lift's block_fell used a single point since its block has
    no meaningful extent; the bar can tip, so checking only the center
    could miss an end that slid off the table edge while the center
    stayed higher).
    """
    center_z = bar_pos_w(env, asset_cfg)[:, 2]
    right_z = bar_end_pos_w(env, asset_cfg, RIGHT_END_OFFSET)[:, 2]
    left_z = bar_end_pos_w(env, asset_cfg, LEFT_END_OFFSET)[:, 2]
    lowest = torch.minimum(torch.minimum(center_z, right_z), left_z)
    return lowest < 0.30


def reset_bar_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_range: float = 0.03,
) -> None:
    """Raw-coords free-body reset, matching lift's reset_block_uniform.

    xy_range=0.03 (lift's proven value) reused as-is: worst case it shifts
    an end's y from -0.16 to -0.13, still well clear of the centerline
    collision zone the env cfg docstring documents avoiding (a few mm of
    jitter is not the same regime as reaching toward y=0).
    """
    bar: Entity = env.scene[asset_cfg.name]
    default = bar.data.default_root_state
    assert default is not None
    state = default[env_ids].clone()
    n = len(env_ids)
    state[:, 0] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 1] += (torch.rand(n, device=env.device) * 2 - 1) * xy_range
    state[:, 7:] = 0.0
    bar.write_root_state_to_sim(state, env_ids=env_ids)
    _max_together_height(env)[env_ids] = 0.0
    _max_individual_height(env, "right")[env_ids] = 0.0
    _max_individual_height(env, "left")[env_ids] = 0.0
    streaks = _streak_buffers(env)
    streaks["right"][env_ids] = 0.0
    streaks["left"][env_ids] = 0.0


def reset_bar_held_high(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    robot_joints_cfg: SceneEntityCfg,
    probability: float = HELD_HIGH_PROBABILITY,
) -> None:
    """Reset a fraction of episodes with the bar already raised.

    Bimanual version of lift's reset_held_high -- rewritten 2026-08-23

    to fix the yank-back bug (see HELD_HIGH_PROBABILITY's docstring above
    for the full diagnosis and the probe that confirmed it).

    Now mirrors lift's actual mechanism exactly, not just its intent: the
    robot's default pose (get_bimanual_lift_robot_cfg's init_state, fixed
    in this same commit) IS the held-high pose, so reset_robot_joints'
    own jitter already puts the arm there for every episode, every time,
    with zero action holding it -- this function only needs to pin the
    fingers to their squeeze width and place the BAR at the held height,
    exactly like lift's reset_held_high does for the block. No arm-joint
    writes, no interpolation, no annealing: with the yank-back gone, a
    fixed low probability (lift's own proven 0.15) is the only tested
    ratio, so that's what's used rather than re-guessing a schedule.

    Runs AFTER reset_bar_uniform (overrides the subset for the picked
    envs; streak buffers are left as reset_bar_uniform set them, matching
    lift's own precedent of not separately touching them here).
    """
    if probability <= 0.0:
        return
    robot: Entity = env.scene[robot_joints_cfg.name]
    bar: Entity = env.scene[asset_cfg.name]
    pick = torch.rand(len(env_ids), device=env.device) < probability
    ids = env_ids[pick]
    if len(ids) == 0:
        return
    jp = robot.data.joint_pos[ids].clone()
    jv = torch.zeros_like(robot.data.joint_vel[ids])
    for j, jname in enumerate(robot.joint_names):
        if "right_finger" in jname:
            jp[:, j] = RIGHT_FINGER_SQUEEZE
        if "left_finger" in jname:
            jp[:, j] = LEFT_FINGER_SQUEEZE
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    state = bar.data.default_root_state[ids].clone()
    state[:, 0] = BAR_START[0]
    state[:, 1] = BAR_START[1]
    state[:, 2] = BAR_START[2] + HELD_HIGH_RAISE
    state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    state[:, 7:] = 0.0
    bar.write_root_state_to_sim(state, env_ids=ids)
    # Height buffers: credit only NEW height above the spawn hold (exact
    # match to lift's own reset_held_high comment: "credit only NEW
    # height above the spawn hold").
    _max_together_height(env)[ids] = HELD_HIGH_RAISE
    _max_individual_height(env, "right")[ids] = HELD_HIGH_RAISE
    _max_individual_height(env, "left")[ids] = HELD_HIGH_RAISE
