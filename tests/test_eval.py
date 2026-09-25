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

"""Success-term resolution: the one place a wrong answer is silent.

Picking the wrong termination term does not raise -- it prints a plausible
number that means something else -- so the selection rules are pinned here.
"""

import pytest

from openarm_mjlab.eval import resolve_success_term


def test_explicit_request_wins():
    assert resolve_success_term(["time_out", "reached"], "reached") == "reached"


def test_explicit_request_must_exist():
    with pytest.raises(SystemExit, match="not a termination term"):
        resolve_success_term(["time_out", "reached"], "nope")


def test_prefers_term_named_success():
    terms = ["time_out", "nan_detection", "cube_dropped", "success"]
    assert resolve_success_term(terms, None) == "success"


def test_infers_sole_non_failure_term():
    assert resolve_success_term(["time_out", "swung_target"], None) == "swung_target"
    assert resolve_success_term(["time_out", "turned_target"], None) == "turned_target"


def test_failure_terms_are_never_inferred_as_success():
    # puck_fell is a failure; puck_at_goal is the real success condition.
    terms = ["time_out", "puck_fell", "puck_at_goal"]
    assert resolve_success_term(terms, None) == "puck_at_goal"


def test_ambiguous_selection_raises_rather_than_guessing():
    with pytest.raises(SystemExit, match="Could not infer"):
        resolve_success_term(["time_out", "lifted", "placed"], None)


def test_no_candidate_raises():
    with pytest.raises(SystemExit, match="Could not infer"):
        resolve_success_term(["time_out", "nan_detection"], None)
