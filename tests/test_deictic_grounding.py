"""Deictic grounding FSM: pointing / speaking order, follow-up merging, explicit-object pass-through."""
from ws_racr.decision.deictic_grounding import DeicticGroundingModule, DeicticSessionState

OBJECTS = ["cylinder", "screw_big", "screw_small", "hex_wrench", "bracket"]


def make_module(wait=0.0):
    # zero wait windows: resolve() never sleeps, tests stay instant
    module = DeicticGroundingModule(known_objects=OBJECTS, wait_point_sec=wait, poll_sec=0.01)
    module.post_ask_wait_sec = 0.0
    return module


def never_pointing(_ws):
    return None


def ask_none(_text):
    return None


def test_predicates():
    m = make_module()
    assert m.is_deictic_command("把那個拿給我")
    assert not m.is_deictic_command("關閉夾爪")                 # 夾爪 must not trigger the 夾 hint
    assert m.contains_explicit_object("拿大螺絲")
    assert m.extract_object_from_text("拿大螺絲") == "screw_big"
    assert m.is_unresolved_deictic_command("拿那個給我")
    assert not m.is_unresolved_deictic_command("拿圓柱給我")


def test_point_then_speak_injects_target():
    m = make_module()
    out = m.resolve("把那個拿給我", world_state=None, get_live_pointing_target_fn=lambda ws: "screw_big_1", ask_fn=ask_none)
    assert out == "把那個拿給我（指的是screw_big_1）"
    assert m.session_state == DeicticSessionState.IDLE


def test_explicit_object_passes_through():
    m = make_module()
    out = m.resolve("拿圓柱給我", None, never_pointing, ask_none)
    assert out == "拿圓柱給我"
    assert m.session_state == DeicticSessionState.IDLE


def test_non_deictic_passes_through():
    m = make_module()
    assert m.resolve("目前在第幾步", None, never_pointing, ask_none) == "目前在第幾步"


def test_unresolved_command_waits_for_follow_up_and_merges():
    m = make_module()
    command = "把那個拿給我"
    assert m.resolve(command, None, never_pointing, ask_none) == command
    assert m.session_state == DeicticSessionState.AWAIT_FOLLOWUP
    assert m.pending_base_command == command

    merged = m.consume_pending_if_clarification("大螺絲", is_question_fn=lambda t: False)
    assert merged == "把那個拿給我（指的是大螺絲）"
    assert m.session_state == DeicticSessionState.IDLE
    record = m.take_last_merge_record()
    assert record["follow_up"] == "大螺絲"
    assert m.take_last_merge_record() is None


def test_new_command_drops_pending():
    m = make_module()
    m.resolve("把那個拿給我", None, never_pointing, ask_none)
    out = m.consume_pending_if_clarification("幫我拿圓柱", is_question_fn=lambda t: False)
    assert out == "幫我拿圓柱"
    assert m.session_state == DeicticSessionState.IDLE
    assert m.pending_base_command is None


def test_already_injected_target_is_untouched():
    m = make_module()
    cmd = "把那個拿給我（指的是cylinder_1）"
    assert m.resolve(cmd, None, never_pointing, ask_none) == cmd
