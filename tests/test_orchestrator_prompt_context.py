"""Planning-stage prompt context (``HRCOrchestrator._think``).

Checks that the user message sent to the planner LLM contains

  A. the pointing target (deictic context from H_t),
  B. recent interaction history (interaction log),
  C. learned semantic rules - only those above the confidence threshold.

The LLM client is mocked, so no API key, ROS, camera or database is needed.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

from ws_racr.memory import InteractionLog, InteractionRecord, SemanticRule
from ws_racr.orchestrator import HRCOrchestrator


# ---------------------------------------------------------------------------
# Minimal WorldState stand-in
# ---------------------------------------------------------------------------

@dataclass
class _HumanState:
    intent: str = "idle"
    gesture_raw: Optional[str] = None
    pointing_target: Optional[str] = None
    state: Optional[str] = None
    gesture: Optional[str] = None


@dataclass
class _AssemblyProgress:
    current_step: str = "idle"


@dataclass
class _WorldState:
    human: _HumanState = field(default_factory=_HumanState)
    objects: Dict = field(default_factory=dict)
    step_class: str = "idle"
    assembly_progress: _AssemblyProgress = field(default_factory=_AssemblyProgress)
    robot: MagicMock = field(default_factory=lambda: MagicMock(gripper_state="open"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_planner_prompt(ws, command, memory_system=None) -> str:
    """Run ``_think`` with a fake LLM client and return the user message it received."""
    controller = MagicMock()
    controller.memory_system = memory_system

    orch = HRCOrchestrator(controller)
    captured = {}

    def fake_create(**kwargs):
        captured["user_content"] = kwargs["messages"][1]["content"]
        raise RuntimeError("stop_after_capture")

    with patch.object(orch, "_get_llm_client") as mock_client:
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = fake_create
        mock_client.return_value = fake_client
        try:
            orch._think(command, ws)
        except RuntimeError:
            pass

    return captured.get("user_content", "")


def _make_interaction_log(entries) -> InteractionLog:
    """entries = [(user_utterance, response_type, robot_response), ...]"""
    log = InteractionLog(maxlen=20)
    for user_utt, response_type, robot_resp in entries:
        log.record(InteractionRecord(
            timestamp=time.time(),
            user_utterance=user_utt,
            response_type=response_type,
            robot_response=robot_resp,
            step_context="idle",
        ))
    return log


def _memory_system(log: InteractionLog, rules: Optional[dict] = None) -> MagicMock:
    ms = MagicMock()
    ms.interaction_log = log
    ms.semantic_rules = rules or {}
    return ms


def _semantic_rule(text: str, confidence: float) -> SemanticRule:
    return SemanticRule(
        rule_id="test_rule_1",
        rule_text=text,
        confidence=confidence,
        supporting_episodes=["ep_001", "ep_002"],
    )


# ---------------------------------------------------------------------------
# A. pointing target
# ---------------------------------------------------------------------------

def test_pointing_target_injected():
    ws = _WorldState()
    ws.human.pointing_target = "screw_big_1"
    msg = _capture_planner_prompt(ws, "把那個拿來")
    assert "pointing_target" in msg
    assert "screw_big_1" in msg


def test_no_pointing_target_omitted():
    ws = _WorldState()
    msg = _capture_planner_prompt(ws, "幫我拿螺絲")
    assert "pointing_target" not in msg


# ---------------------------------------------------------------------------
# B. interaction log
# ---------------------------------------------------------------------------

def test_history_injected_when_log_has_entries():
    ms = _memory_system(_make_interaction_log([
        ("夾取大螺絲", "execute", "正在夾取大螺絲"),
        ("往右移5公分", "execute", "已往右移動 50mm"),
    ]))
    msg = _capture_planner_prompt(_WorldState(), "再拿一個", memory_system=ms)
    assert "Recent Interaction History" in msg
    assert "夾取大螺絲" in msg


def test_history_omitted_when_log_empty():
    ms = _memory_system(InteractionLog(maxlen=20))
    msg = _capture_planner_prompt(_WorldState(), "拿螺絲", memory_system=ms)
    assert "Recent Interaction History" not in msg


# ---------------------------------------------------------------------------
# C. semantic rules
# ---------------------------------------------------------------------------

def test_high_confidence_rule_injected():
    rule = _semantic_rule("在 step 5 時 pickup pi 應從左側接近", confidence=0.85)
    ms = _memory_system(InteractionLog(maxlen=20), {"r1": rule})
    msg = _capture_planner_prompt(_WorldState(), "幫我放好 pi", memory_system=ms)
    assert "Learned SOP Rules" in msg
    assert "從左側接近" in msg


def test_low_confidence_rule_filtered():
    rule = _semantic_rule("某個不確定的規則", confidence=0.45)
    ms = _memory_system(InteractionLog(maxlen=20), {"r_low": rule})
    msg = _capture_planner_prompt(_WorldState(), "拿螺絲", memory_system=ms)
    assert "Learned SOP Rules" not in msg
