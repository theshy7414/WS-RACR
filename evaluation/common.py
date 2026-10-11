"""Helpers shared by the offline evaluation scripts.

``production_risk_decision`` / ``production_object_gate`` run the *production* Layer 3 / Layer 2 policy
(``ws_racr.decision.collaboration_policy``) on a retrieved-plan candidate, so EXP3 and the
baseline comparison measure the deployed rules and not a copy of them.
"""
from __future__ import annotations

from types import SimpleNamespace

from ws_racr.decision.collaboration_policy import CollaborationPolicyModule

# World state with no stop gesture and an empty gripper: Layer 2 passes, only Layer 3 decides.
_NEUTRAL_WORLD_STATE = SimpleNamespace(
    robot=SimpleNamespace(holding_object_id=None),
    human=SimpleNamespace(intent_stable=""),
)
_POLICY = CollaborationPolicyModule(
    infer_progress_fn=lambda ws: {"step_index": 0},
    has_object_available_fn=lambda ws, obj: True,
)


def production_risk_decision(current_cmd: str, retrieved_cmd: str, ret_type: str, sim: float) -> str:
    """Layer 3 decision (REUSE / ADAPT / CLARIFY) of the production policy for one retrieved plan."""
    result = _POLICY.score_and_select_policy(
        sim=sim,
        world_state=_NEUTRAL_WORLD_STATE,
        retrieved_task_type=ret_type,
        current_command=current_cmd,
        retrieved_command=retrieved_cmd,
    )
    return result["policy"]


def production_object_gate(target_object, visible_objects) -> bool:
    """Layer 2 hard gate of the production policy: is the commanded target visible (``O_t``)?

    Returns True when the plan may proceed to Layer 3. ``target_object=None`` means the plan
    needs no object.
    """
    policy = CollaborationPolicyModule(
        infer_progress_fn=lambda ws: {"step_index": 0},
        has_object_available_fn=lambda ws, obj: obj in ws.objects,
    )
    ws = SimpleNamespace(
        objects=list(visible_objects),
        robot=SimpleNamespace(holding_object_id=None),
        human=SimpleNamespace(intent_stable=""),
    )
    feasible, _reason = policy.check_global_feasibility(
        world_state=ws, current_target_object=target_object
    )
    return feasible


def pct(value: float) -> str:
    """Format a ratio in [0, 1] as a fixed-width percentage."""
    return f"{value * 100:5.1f}%"


def bar(value: float, width: int = 20) -> str:
    """Unicode progress bar for a ratio in [0, 1]."""
    filled = round(value * width)
    return "\u2588" * filled + "\u2591" * (width - filled)
