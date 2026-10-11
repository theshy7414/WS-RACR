"""World state W_t = {S_t, O_t, H_t, R_t} and the step definitions behind S_t."""
from ws_racr.world_state import ObjectState, WorldState, create_empty_world_state
from ws_racr.world_state.step_definitions import (
    ALLOWED_TRANSITIONS,
    CLASS_TO_INDEX,
    STEP_CLASSES,
    get_step_class,
    get_step_index,
    is_valid_transition,
    step_index_to_user_display,
    user_input_to_step_index,
)


def test_world_state_has_all_four_components():
    ws = create_empty_world_state()
    assert ws.step_class == "idle"          # S_t
    assert ws.objects == {}                 # O_t
    assert ws.human is not None             # H_t
    assert ws.robot is not None             # R_t
    assert isinstance(ws, WorldState)


def test_object_queries():
    ws = create_empty_world_state()
    ws.update_object(ObjectState(object_type="cylinder", object_id="cylinder_1", location="in_tray"))
    ws.update_object(ObjectState(object_type="cylinder", object_id="cylinder_2", location="on_platform"))
    ws.update_object(ObjectState(object_type="nut", object_id="nut_1", location="in_tray"))
    assert len(ws.get_objects_by_type("cylinder")) == 2
    assert {o.object_id for o in ws.get_objects_at_location("in_tray")} == {"cylinder_1", "nut_1"}
    assert ws.get_object("nut_1").object_type == "nut"


def test_interaction_state_follows_gesture_and_gripper():
    ws = create_empty_world_state()
    ws.human.intent = "receive"
    ws.robot.gripper_state = "closed"
    closed = ws.interaction_state
    ws.robot.gripper_state = "open"
    assert ws.interaction_state != closed
    ws.human.intent = "stop"
    assert ws.interaction_state == "human_requesting_stop" or "stop" in ws.interaction_state.lower()


def test_safety_flags():
    ws = create_empty_world_state()
    assert ws.is_safe()
    assert not ws.requires_emergency_stop()


def test_nine_step_classes_and_user_numbering():
    assert len(STEP_CLASSES) == 9
    assert get_step_class(8) == "idle"
    assert user_input_to_step_index(1) == 0      # "step 1" -> TSM index 0
    assert user_input_to_step_index(3) == 2
    assert user_input_to_step_index(0) == 8      # idle
    assert step_index_to_user_display(7) == 8
    assert step_index_to_user_display(8) == 0
    for index, name in STEP_CLASSES.items():
        assert get_step_index(name) == index == CLASS_TO_INDEX[name]


def test_step_transitions_are_forward_only():
    assert is_valid_transition("place_platform", "insert_cylinder_under_platform")
    assert not is_valid_transition("insert_cylinder_under_platform", "place_platform")
    assert is_valid_transition("insert_lower_big_screw", "idle")
    for source, targets in ALLOWED_TRANSITIONS.items():
        for target in targets:
            assert target is None or target in CLASS_TO_INDEX
