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

from ...common_mdp import (
    both_pads_on_block,
    descent_penalty,
    env_buffer,
    new_progress_rate,
    object_pos_w,
    object_speed,
    partial_pinch_reward,
    pinch_obs,
    pinch_reward,
    reset_object_xy_uniform,
    terminated_by,
    tool_to_point,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "descent_penalty",
    "partial_pinch_reward",
    "pinch_obs",
    "pinch_reward",
    "terminated_by",
]

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

# Start some episodes already grasping, below the success height. These
# joint poses were solved against the compiled task scene, with the palms
# outside the bar ends and the jaws closing along world X. The EE local X
# axis points up; local Z points outward along the bar (right: -Y, left: +Y).
HELD_HIGH_RAISE = 0.08
# The shared tool point is 135 mm from the wrist. Keep it 8 mm outward
# from each end center so the block contacts the pads, clearing the palm
# and proximal finger geometry. The actual grip center is thus 143 mm
# along local -Z. Centering the shared tool exactly causes base contact.
HELD_HIGH_TOOL_CLEARANCE = 0.008
HELD_HIGH_RIGHT_Q = (
    0.60144666,
    0.74906976,
    -0.66989157,
    0.98381081,
    0.95896037,
    -0.25823608,
    1.56930847,
)
HELD_HIGH_LEFT_Q = (
    -0.61281477,
    -0.75918873,
    0.70237430,
    0.98381081,
    -0.98149732,
    0.26922589,
    -1.56151126,
)
# Also used by BIMANUAL_LIFT_HOME: zero arm action holds the reset pose.
# Selected curriculum resets restore these exact angles after joint jitter;
# unassisted episodes retain the normal randomized joint initialization.
HELD_HIGH_PROBABILITY = 0.15
# Compiled-scene contacts at this width overlap only the intended pads by
# about 0.51 mm, with no palm, proximal-finger, or connecting-rod contact.
RIGHT_FINGER_SQUEEZE = -0.265
LEFT_FINGER_SQUEEZE = 0.265


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


def tool_to_end_obs(env, robot_cfg, asset_cfg, local_offset) -> torch.Tensor:
    """Return the tool-to-bar-end vector observation."""
    return tool_to_point(env, robot_cfg, bar_end_pos_w(env, asset_cfg, local_offset))


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


def both_ends_gripped(env, sensor_right: str, sensor_left: str) -> torch.Tensor:
    """Return True where both grippers hold their own bar end.

    Both grippers independently satisfy lift's proven two-pad pinch gate

    on their own end. Reused directly (not reimplemented) from lift.mdp:
    both_pads_on_block is fully generic given a sensor_name -- it carries no
    hidden per-task state, so calling it twice (once per side) is safe.
    """
    return both_pads_on_block(env, sensor_right) & both_pads_on_block(env, sensor_left)


def _streak_buffers(env) -> dict[str, torch.Tensor]:
    return {
        side: env_buffer(env, f"_bimanual_lift_streak_{side}")
        for side in ("right", "left")
    }


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
DEFAULT_EE_QUAT_RIGHT = (0.5, 0.5, -0.5, 0.5)
DEFAULT_EE_QUAT_LEFT = (0.5, -0.5, -0.5, -0.5)
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
    return env_buffer(env, "_bimanual_lift_max_h")


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
    rate = new_progress_rate(
        env, h_together, _max_together_height(env), MAX_LIFT_RATE, cap=TARGET_LIFT
    )
    gate = both_ends_gripped(env, sensor_right, sensor_left).float()
    return rate * gate


def _max_individual_height(env, side: str) -> torch.Tensor:
    return env_buffer(env, f"_bimanual_lift_max_h_{side}")


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
    rate = new_progress_rate(
        env, h, _max_individual_height(env, side), MAX_LIFT_RATE, cap=TARGET_LIFT
    )
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
    slow = object_speed(env, asset_cfg) < SETTLED_SPEED
    gripped = both_ends_gripped(env, sensor_right, sensor_left)
    return in_window_r & in_window_l & level & slow & gripped


def bar_fell(env, asset_cfg) -> torch.Tensor:
    """Return True where the bar has dropped below the floor threshold.

    Fell if the bar's center OR either end drops below the floor

    threshold (lift's block_fell termination used a single point since its block has
    no meaningful extent; the bar can tip, so checking only the center
    could miss an end that slid off the table edge while the center
    stayed higher).
    """
    center_z = object_pos_w(env, asset_cfg)[:, 2]
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
    reset_object_xy_uniform(env, env_ids, asset_cfg, xy_range)
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
    """Reset selected episodes to a grasp aligned with the raised bar.

    Run after the normal joint and bar resets. Restore the reference arm
    pose to remove joint jitter only for selected episodes, set the finger
    width, and place the bar. The reference is also the action default, so
    zero arm action holds this pose instead of pulling away from the bar.
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
    arm_positions = {
        f"openarm_{side}_joint{i}": q
        for side, angles in (("right", HELD_HIGH_RIGHT_Q), ("left", HELD_HIGH_LEFT_Q))
        for i, q in enumerate(angles, start=1)
    }
    for j, jname in enumerate(robot.joint_names):
        if jname in arm_positions:
            jp[:, j] = arm_positions[jname]
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
