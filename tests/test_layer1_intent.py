"""Layer 1: rule-based segmentation and classification of spoken commands (no LLM)."""
import pytest

from ws_racr.orchestrator.layer1_intent import IntentMixin


@pytest.fixture(scope="module")
def layer1():
    return IntentMixin()


def types(layer1, command):
    return [intent["type"] for intent in layer1._parse_intents(command)]


def test_single_execute(layer1):
    intents = layer1._parse_intents("把圓柱夾起來")
    assert [i["type"] for i in intents] == ["EXECUTE"]
    assert intents[0]["target_object"] is not None


def test_question_is_answer_not_execute(layer1):
    assert types(layer1, "桌上有哪些零件？") == ["ANSWER"]


def test_multi_intent_is_split_in_order(layer1):
    assert layer1._segment_command("幫我拿圓柱，然後告訴我下一步") == ["幫我拿圓柱", "告訴我下一步"]
    assert types(layer1, "幫我拿圓柱，然後告訴我下一步") == ["EXECUTE", "ANSWER"]


def test_execute_then_execute(layer1):
    assert types(layer1, "拿螺絲然後鎖上去") == ["EXECUTE", "EXECUTE"]


def test_vague_reference_without_object_asks(layer1):
    assert types(layer1, "把那個拿過來") == ["ASK"]


def test_action_prefix_overrides_question_marker(layer1):
    """'請告訴我…' style prefixes: an action verb prefix must never be routed as ANSWER by _is_strong_question."""
    assert layer1._is_strong_question("幫我拿螺絲嗎") is False
    assert layer1._is_strong_question("現在是第幾步") is True


def test_occluded_target_demotes_execute_to_ask(layer1):
    class WS:
        occluded_objects = ["cylinder"]

    intents = layer1._parse_intents("把圓柱夾起來", world_state=WS())
    assert intents[0]["type"] == "ASK"
