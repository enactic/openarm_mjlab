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

"""Instructions for the language-conditioned puck task, and how they split.

Each goal gets a POOL of phrasings rather than one sentence. With a single
sentence per goal its embedding is a fixed constant -- a one-hot in another
basis -- and "the policy follows instructions" would mean it had memorised
two vectors. The pool is split into TRAIN and HELD_OUT, the policy only ever
sees TRAIN during training, and every number worth reporting is measured on
HELD_OUT.

Both goals live in ONE environment and share an observation distribution, so
the only thing that differs between them is the sentence.
"""

from __future__ import annotations

GOALS = ("right", "left")

TRAIN: dict[str, tuple[str, ...]] = {
    "right": (
        "push the puck to the right target",
        "slide the puck onto the right marker",
        "move the puck to the right-hand spot",
        "shove the puck over to the right target",
        "push the disc to the marker on the right",
        "nudge the puck onto the right goal",
        "send the puck to the right-hand marker",
        "push the puck across to the right spot",
        "get the puck to the target on the right",
        "drive the puck rightwards to its marker",
    ),
    "left": (
        "push the puck to the left target",
        "slide the puck onto the left marker",
        "move the puck to the left-hand spot",
        "shove the puck over to the left target",
        "push the disc to the marker on the left",
        "nudge the puck onto the left goal",
        "send the puck to the left-hand marker",
        "push the puck across to the left spot",
        "get the puck to the target on the left",
        "drive the puck leftwards to its marker",
    ),
}

HELD_OUT: dict[str, tuple[str, ...]] = {
    "right": (
        "the puck should end up on the right target",
        "get that disc to the right-hand marker",
        "work the puck over to the right",
        "leave the puck sitting on the right goal",
        "put the puck on the marker to the right",
        "the right-hand target is where the puck goes",
        "take the puck round to the right spot",
        "aim the puck at the right marker",
        "not the left one -- the other target",
        "push it right, onto the far marker",
        "settle the disc on the right-hand goal",
        "the puck belongs on the right target",
    ),
    "left": (
        "the puck should end up on the left target",
        "get that disc to the left-hand marker",
        "work the puck over to the left",
        "leave the puck sitting on the left goal",
        "put the puck on the marker to the left",
        "the left-hand target is where the puck goes",
        "take the puck round to the left spot",
        "aim the puck at the left marker",
        "not the right one -- the other target",
        "push it left, onto the near marker",
        "settle the disc on the left-hand goal",
        "the puck belongs on the left target",
    ),
}

# Measured, not assumed. diagnostics.encoder_separability scores candidate
# encoders on this exact pool; bge-base separates held-out phrasings of the
# two goals where all-MiniLM-L6-v2 does not, and the policies trained on them
# differ accordingly. See the module docstring in diagnostics.py.
DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"

for _g in GOALS:
    assert _g in TRAIN and _g in HELD_OUT, _g
    assert not (set(TRAIN[_g]) & set(HELD_OUT[_g])), (
        f"{_g}: held-out phrasing leaked into train"
    )
_all = [s for g in GOALS for s in TRAIN[g] + HELD_OUT[g]]
assert len(_all) == len(set(_all)), "the same sentence appears under two goals"
