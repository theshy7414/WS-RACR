"""
pytest 場景路由驗證：三個 canonical scenario 的 routing 檢查

驗證：
- execute_cylinder_handover → _is_query_intent()=False (進 execute_task)
- ask_ambiguous_assist → _is_query_intent()=False (進 execute_task，LLM 決定 ask_human)
- answer_tool_location → _is_query_intent()=True (早期路由 _handle_query_via_voice_query)
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

from ws_racr.execution.controller.robot_controller import RobotController

# Scenario 檔案路徑
_SCENARIOS_DIR = Path(__file__).parent.parent / 'data' / 'scenarios'


def _load_scenario(filename: str) -> dict:
    with open(_SCENARIOS_DIR / filename, encoding='utf-8') as f:
        return json.load(f)


def _make_controller() -> RobotController:
    ctrl = RobotController.__new__(RobotController)
    ctrl.memory_system = MagicMock()
    ctrl.memory_system.interaction_log = MagicMock()
    ctrl.task_manager = MagicMock()
    ctrl.voice_query_handler = None
    return ctrl


# ── Branch 1: EXECUTE ────────────────────────────────────────────────

def test_execute_scenario_not_query_intent():
    """
    Scenario: execute_cylinder_handover
    '幫我拿圓柱' = explicit object + explicit verb → NOT a query intent.
    Must enter execute_task(), not early-routed to voice query handler.
    """
    scenario = _load_scenario('execute_cylinder_handover.json')
    ctrl = _make_controller()

    utterance = scenario['human_utterance']          # "幫我拿圓柱"
    expected_routing = scenario['expected']['routing']  # "execute_directly"

    # action='pickup' is what parser would produce for "幫我拿圓柱"
    result = ctrl._is_query_intent(utterance, action='pickup')

    assert result is False, (
        f"[{scenario['scenario_id']}] '{utterance}' should NOT be a query intent "
        f"(expected routing: {expected_routing})"
    )


# ── Branch 2: ASK ────────────────────────────────────────────────────

def test_ask_scenario_not_query_intent():
    """
    Scenario: ask_ambiguous_assist
    '幫我一下' = ambiguous help request → NOT a direct query intent.
    Should enter execute_task() so LLMDecisionEngine can decide ask_human.
    assert: _is_query_intent() returns False
    """
    scenario = _load_scenario('ask_ambiguous_assist.json')
    ctrl = _make_controller()

    utterance = scenario['human_utterance']    # "幫我一下"

    # parser action is likely unknown/assist for this utterance
    result = ctrl._is_query_intent(utterance, action='assist')

    assert result is False, (
        f"[{scenario['scenario_id']}] '{utterance}' should NOT be early-routed as query; "
        f"LLMDecisionEngine must handle ambiguity"
    )


def test_ask_scenario_no_question_markers():
    """
    '幫我一下' must not contain any _QUESTION_MARKERS characters
    (ensuring the action-based fallback also won't misfire).
    """
    from ws_racr.execution.controller.robot_controller import _QUESTION_MARKERS
    utterance = "幫我一下"
    markers_found = [m for m in _QUESTION_MARKERS if m in utterance]
    assert not markers_found, (
        f"Unexpected question markers {markers_found} found in '{utterance}'"
    )


# ── Branch 3: ANSWER ─────────────────────────────────────────────────

def test_answer_scenario_is_query_intent():
    """
    Scenario: answer_tool_location
    '六角扳手在哪裡？' contains '在哪裡' → matches query pattern → IS a query intent.
    Must be early-routed to _handle_query_via_voice_query().
    """
    scenario = _load_scenario('answer_tool_location.json')
    ctrl = _make_controller()

    utterance = scenario['human_utterance']          # "六角扳手在哪裡？"
    expected_routing = scenario['expected']['routing']  # "_handle_query_via_voice_query()"

    # action='stop' as a conservative default (parser won't know what to do with spatial query)
    result = ctrl._is_query_intent(utterance, action='stop')

    assert result is True, (
        f"[{scenario['scenario_id']}] '{utterance}' SHOULD be a query intent "
        f"(expected routing: {expected_routing})"
    )


def test_answer_scenario_在哪裡_pattern_match():
    """
    Verify the specific regex pattern r'(.+)在哪裡' matches the ANSWER scenario utterance.
    This is the KG-grounded spatial query test.
    """
    import re
    utterance = "六角扳手在哪裡？"
    pattern = r'(.+)在哪裡'
    match = re.search(pattern, utterance)
    assert match is not None, f"Pattern '{pattern}' must match '{utterance}'"
    assert match.group(1) == '六角扳手', f"Captured subject should be '六角扳手', got '{match.group(1)}'"


# ── Schema validation: all scenario JSON files are well-formed ───────

def test_all_scenario_files_have_required_keys():
    """All scenario JSON files must contain required top-level keys."""
    required_keys = {
        'scenario_id', 'branch', 'world_state_summary',
        'human_utterance', 'expected', 'evaluation_criteria', 'thesis_claim'
    }
    for json_file in _SCENARIOS_DIR.glob('*.json'):
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        missing = required_keys - set(data.keys())
        assert not missing, f"{json_file.name}: missing keys {missing}"


def test_all_scenarios_cover_three_branches():
    """The scenarios directory must contain at least one file per branch."""
    branches_found = set()
    for json_file in _SCENARIOS_DIR.glob('*.json'):
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        branches_found.add(data.get('branch'))
    required_branches = {'execute', 'ask_human', 'answer_question'}
    missing = required_branches - branches_found
    assert not missing, f"Missing branch coverage: {missing}"
