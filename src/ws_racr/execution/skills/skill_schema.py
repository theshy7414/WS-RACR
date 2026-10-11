"""
Skill Schema — single source of truth for robot action ontology.

Rules:
  - CANONICAL_ACTIONS  : the exact set LLM must output and executor dispatches on.
  - SKILL_DEFINITIONS  : machine-readable schema used to generate LLM prompts and
                         drive ActionValidator parameter checks.
  - ACTION_ALIASES     : many-to-one rewrite map (LLM hallucinations → canonical).

All other modules (action_validator, hrc_orchestrator) import from here.
Never hard-code action names elsewhere.
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical action names
# Each entry is the single authoritative string for that skill.
# ---------------------------------------------------------------------------

CANONICAL_ACTIONS: FrozenSet[str] = frozenset({
    # Object manipulation
    "pickup",
    "place",
    "handover",
    # Navigation
    "move",
    # Relative movement
    "shift",
    # Gripper
    "gripper_open",
    "gripper_close",
    # Rotation
    "rotate_gripper",
    # Return
    "return_to_origin",
    # Utility
    "stop",
    "emergency_stop",
    "unknown",
    # Gesture entry-points (kept for backward compat)
    "receive手勢處理",
    "pointing手勢處理",
})

# ---------------------------------------------------------------------------
# Skill definitions
#
# Each entry is a dict with:
#   description : human-readable label used in LLM prompt table
#   required    : list of required parameter keys
#   optional    : list of optional parameter keys (with default in parens)
#   notes       : extra hint for LLM prompt
# ---------------------------------------------------------------------------

SKILL_DEFINITIONS: List[Dict] = [
    {
        "action":      "pickup",
        "description": "Pick up an object from the table",
        "required":    ["target_object"],
        "optional":    [],
        "notes":       "",
    },
    {
        "action":      "place",
        "description": "Place held object at a named location",
        "required":    ["target_object", "target_location"],
        "optional":    [],
        "notes":       "",
    },
    {
        "action":      "handover",
        "description": "Hand an object to the human",
        "required":    ["target_object"],
        "optional":    [],
        "notes":       "",
    },
    {
        "action":      "move",
        "description": "Move to a named position",
        "required":    ["target_location"],
        "optional":    [],
        "notes":       "",
    },
    {
        "action":      "shift",
        "description": "Relative move from current position",
        "required":    ["direction", "distance"],
        "optional":    [],
        "notes":       "direction: left|right|forward|backward|up|down  |  distance unit: mm  |  1公分=10mm",
    },
    {
        "action":      "rotate_gripper",
        "description": "Rotate the gripper",
        "required":    ["rotation_direction"],
        "optional":    ["rotation_angle"],
        "notes":       "rotation_direction: clockwise|counterclockwise|left|right  |  rotation_angle default 45°",
    },
    {
        "action":      "gripper_open",
        "description": "Open the gripper",
        "required":    [],
        "optional":    [],
        "notes":       "",
    },
    {
        "action":      "gripper_close",
        "description": "Close the gripper",
        "required":    [],
        "optional":    [],
        "notes":       "",
    },
    {
        "action":      "return_to_origin",
        "description": "Return robot to home/origin position",
        "required":    [],
        "optional":    [],
        "notes":       "",
    },
    {
        "action":      "stop",
        "description": "Stop current motion immediately",
        "required":    [],
        "optional":    [],
        "notes":       "",
    },
]

# ---------------------------------------------------------------------------
# Known parameter values
# ---------------------------------------------------------------------------

KNOWN_OBJECTS: Tuple[str, ...] = (
    "bracket", "cylinder", "hex_wrench", "nut",
    "pi", "platform", "screw_big", "screw_small",
)

KNOWN_LOCATIONS: Tuple[str, ...] = (
    "origin", "storage", "standby",
    "left_area", "right_area", "center_area",
)

KNOWN_DIRECTIONS: Tuple[str, ...] = (
    "left", "right", "forward", "backward", "up", "down",
)

KNOWN_ROTATION_DIRECTIONS: Tuple[str, ...] = (
    "clockwise", "counterclockwise", "left", "right",
)

# ---------------------------------------------------------------------------
# Alias / synonym rewrite map
#
# Many-to-one: all keys rewrite to their canonical action.
# Applied BEFORE fuzzy matching in ActionValidator.
# ---------------------------------------------------------------------------

ACTION_ALIASES: Dict[str, str] = {
    # move variants
    "move to pos":       "move",
    "move to position":  "move",
    "move to":           "move",
    "go to":             "move",
    "go_to":             "move",
    "navigate to":       "move",
    "navigate":          "move",
    "go":                "move",
    # shift / relative movement variants
    "move_relative":     "shift",
    "adjust_position":   "shift",
    "nudge":             "shift",
    # pickup variants
    "grab":              "pickup",
    "grasp":             "pickup",
    # place variants
    "put":               "place",
    "release":           "place",
    # handover variants
    "give":              "handover",
    "pass":              "handover",
    # gripper
    "open_gripper":      "gripper_open",
    "open gripper":      "gripper_open",
    "close_gripper":     "gripper_close",
    "close gripper":     "gripper_close",
    # rotate
    "twist":             "rotate_gripper",
    "turn":              "rotate_gripper",
    "rotate_relative":   "rotate_gripper",
    # return
    "return_origin":     "return_to_origin",
    "return origin":     "return_to_origin",
    "go_home":           "return_to_origin",
    "go home":           "return_to_origin",
    "return_to_home":    "return_to_origin",
    "return_to_last":    "return_to_origin",
    "go_back":           "return_to_origin",
}

# ---------------------------------------------------------------------------
# Prompt generation helpers  (used by hrc_orchestrator)
# ---------------------------------------------------------------------------

def build_skill_table() -> str:
    """Return a formatted skill table for insertion into LLM system prompts."""
    lines = [
        "  action            | required fields      | notes",
        "  ------------------|----------------------|-------------------------------",
    ]
    for sd in SKILL_DEFINITIONS:
        req  = ", ".join(sd["required"]) or "(none)"
        note = sd["notes"]
        lines.append(f"  {sd['action']:<18}| {req:<20} | {note}")
    return "\n".join(lines)


def build_object_list() -> str:
    return ", ".join(KNOWN_OBJECTS)


def build_location_list() -> str:
    return ", ".join(KNOWN_LOCATIONS)
