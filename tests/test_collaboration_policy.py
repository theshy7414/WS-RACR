"""Layer 2 (feasibility gate) and Layer 3 (risk rules) of the collaborative decision policy."""
from types import SimpleNamespace

import pytest

from ws_racr.decision.collaboration_policy import (
    CollaborationPolicyModule,
    _detect_action_type,
    decide_risk,
)


def make_policy(visible=("cylinder", "screw")):
    return CollaborationPolicyModule(
        infer_progress_fn=lambda ws: {"step_index": 0},
        has_object_available_fn=lambda ws, obj: obj in visible,
    )


def world_state(holding=None, human_intent=""):
    return SimpleNamespace(
        robot=SimpleNamespace(holding_object_id=holding),
        human=SimpleNamespace(intent_stable=human_intent),
    )


# --- Layer 3: risk rules ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "sim, action_match, object_match, cur_type, ret_type, expected",
    [
        (0.90, True, True, "pickup", "pickup", "REUSE"),        # rule 1
        (0.85, True, True, "pickup", "pickup", "REUSE"),        # boundary
        (0.849, True, True, "pickup", "pickup", "ADAPT"),       # just below rule 1 -> rule 2
        (0.95, True, False, "pickup", "pickup", "ADAPT"),       # same action, other object
        (0.50, True, False, "place", "place", "ADAPT"),         # rule 2 lower boundary
        (0.49, True, True, "pickup", "pickup", "CLARIFY"),      # too dissimilar
        (0.94, False, True, "pickup", "handover", "ADAPT"),     # rule 3: compatible pair, high sim
        (0.74, False, True, "pickup", "handover", "CLARIFY"),   # rule 3 needs sim >= 0.75
        (0.95, False, True, "gripper", "pickup", "CLARIFY"),    # incompatible actions, any sim
        (0.95, False, True, "place", "move", "CLARIFY"),
    ],
)
def test_decide_risk_rules(sim, action_match, object_match, cur_type, ret_type, expected):
    assert decide_risk(sim, action_match, object_match, cur_type, ret_type) == expected


def test_high_similarity_alone_never_reuses_a_different_action():
    """Retrieval similarity only recalls candidates: pickup-vs-handover at sim 0.94 must not REUSE."""
    result = make_policy().score_and_select_policy(
        sim=0.94,
        world_state=world_state(),
        retrieved_task_type="handover",
        current_command="幫我拿圓柱體",
        retrieved_command="把圓柱體遞給我",
    )
    assert result["policy"] == "ADAPT"
    assert result["action_match"] is False


def test_identical_command_is_reused():
    result = make_policy().score_and_select_policy(
        sim=1.0,
        world_state=world_state(),
        retrieved_task_type="pickup",
        current_command="幫我拿圓柱體",
        retrieved_command="幫我拿圓柱體",
    )
    assert result["policy"] == "REUSE"


def test_naive_mode_reuses_regardless():
    result = make_policy().score_and_select_policy(
        sim=0.1, world_state=world_state(holding="x", human_intent="stop"), naive_mode=True
    )
    assert result["policy"] == "REUSE"


# --- Layer 2: feasibility gate ---------------------------------------------------------------

def test_human_stop_signal_rejects_any_candidate():
    result = make_policy().score_and_select_policy(
        sim=1.0,
        world_state=world_state(human_intent="stop"),
        retrieved_task_type="pickup",
        current_command="幫我拿圓柱體",
        retrieved_command="幫我拿圓柱體",
    )
    assert result["policy"] == "REJECT"
    assert result["feasibility_reason"] == "human_stop_signal"


def test_gripper_already_holding_blocks_pickup():
    result = make_policy().score_and_select_policy(
        sim=1.0,
        world_state=world_state(holding="cylinder_1"),
        retrieved_task_type="pickup",
        current_command="幫我拿圓柱體",
        retrieved_command="幫我拿圓柱體",
    )
    assert result["policy"] == "REJECT"
    assert result["feasibility_reason"].startswith("gripper_holding_object")


def test_holding_object_does_not_block_non_pickup_plans():
    result = make_policy().score_and_select_policy(
        sim=1.0,
        world_state=world_state(holding="cylinder_1"),
        retrieved_task_type="place",
        current_command="放到平台",
        retrieved_command="放到平台",
    )
    assert result["policy"] != "REJECT"


def test_invisible_target_rejects_even_at_similarity_one():
    result = make_policy(visible=("screw",)).score_and_select_policy(
        sim=1.0,
        world_state=world_state(),
        retrieved_task_type="pickup",
        current_command="幫我拿圓柱體",
        retrieved_command="幫我拿圓柱體",
        current_target_object="cylinder",
    )
    assert result["policy"] == "REJECT"
    assert result["feasibility_reason"] == "current_target_unavailable:cylinder"


def test_visible_target_passes_to_layer_3():
    result = make_policy(visible=("cylinder",)).score_and_select_policy(
        sim=1.0,
        world_state=world_state(),
        retrieved_task_type="pickup",
        current_command="幫我拿圓柱體",
        retrieved_command="幫我拿圓柱體",
        current_target_object="cylinder",
    )
    assert result["policy"] == "REUSE"


def test_global_feasibility_checks_stop_before_everything():
    ok, reason = make_policy().check_global_feasibility(
        world_state=world_state(holding="x", human_intent="stop"),
        current_action_type="pickup",
        current_target_object="cylinder",
    )
    assert (ok, reason) == (False, "human_stop_signal")


# --- keyword action detector -----------------------------------------------------------------

@pytest.mark.parametrize(
    "command, expected",
    [
        ("把螺絲遞給我", "handover"),
        ("打開夾爪", "gripper"),
        ("移到組裝區", "move"),
        ("把圓柱放到平台上", "place"),
        ("拿螺絲", "pickup"),
        ("你好", "unknown"),
    ],
)
def test_detect_action_type(command, expected):
    assert _detect_action_type(command) == expected
