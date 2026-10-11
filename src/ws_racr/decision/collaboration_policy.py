"""Layer 2 (execution feasibility) and Layer 3 (execution risk) of the collaborative decision policy.

Plan RAG only recalls candidate plans. This module decides what to do with a candidate:

* Layer 2 - hard feasibility gate on the world state ``W_t``: human stop signal (``H_t``), gripper
  already holding an object (``R_t``) and visibility of the commanded target (``O_t``) -> ``REJECT``.
* Layer 3 - four explicit risk rules over (similarity, action match, object match) ->
  ``REUSE`` / ``ADAPT`` / ``CLARIFY``.

The module is stateless; world-state access goes through injected callbacks.
Thesis: Ch. 4 (collaborative decision). Offline evaluation: ``evaluation/exp2_feasibility_gate.py``,
``exp3_execution_risk.py``, ``exp4_e2e_ablation.py``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Layer 3 lightweight action-type detector (keyword-based, no LLM) ──────────────
# Only used for the current command; retrieved plan's type comes from DB directly.

_HANDOVER_KW = ('交給', '遞給', '給我', '給操作員')
_GRIPPER_KW  = ('夾爪', '夾頭')
_MOVE_KW     = ('移動到', '移到', '前往', '回到', '走到', '去')
_PLACE_KW    = ('放到', '放在', '放置', '搬到', '插到', '插入')
_PICKUP_KW   = ('拿', '夾', '取', '抓', '拾')

# Object aliases for comparison (keyword → canonical form)
_OBJ_ALIASES: Dict[str, str] = {
    # Longer Chinese phrases first — substring matching order in _detect_target_object
    '圓柱體': 'cylinder',
    '圓柱': 'cylinder',   'cylinder': 'cylinder',
    '螺絲': 'screw',      'screw': 'screw',
    '大螺絲': 'screw',    '小螺絲': 'screw',
    '支架': 'bracket',    'bracket': 'bracket',
    '六角扳手': 'hex_wrench', '扳手': 'hex_wrench', 'hex_wrench': 'hex_wrench',
    '平台': 'platform',   'PI': 'PI',
    '螺帽': 'nut',
}
_KNOWN_OBJECTS = tuple(_OBJ_ALIASES.keys())

# Location aliases for move command target comparison
# Different canonical locations → move plan cannot be directly reused
_LOCATION_ALIASES: Dict[str, str] = {
    '組裝台': 'assembly',  '組裝區': 'assembly',
    '裝配位置': 'assembly', '組裝位置': 'assembly', '裝配區': 'assembly',
    '待機區': 'home',       '待機位': 'home',       '原點': 'home',
    '工作台': 'workbench',  '工作區': 'workbench',
    '左側': 'left',         '右側': 'right',
    '正前方': 'front',      '旁邊': 'side',         '旁': 'side',
}
_KNOWN_LOCATIONS = tuple(_LOCATION_ALIASES.keys())

# Action pairs where LLM can meaningfully bridge despite type mismatch
# (both sides involve gripping + positioning of an object)
_COMPATIBLE_PAIRS = {
    ('pickup', 'handover'), ('handover', 'pickup'),
    ('place',  'handover'), ('handover', 'place'),
}


def _detect_action_type(cmd: str) -> str:
    """Lightweight keyword-based action-type detector for the current command."""
    for kw in _HANDOVER_KW:
        if kw in cmd: return 'handover'
    for kw in _GRIPPER_KW:
        if kw in cmd: return 'gripper'
    for kw in _MOVE_KW:             # check move before place ('移到' in both)
        if kw in cmd: return 'move'
    for kw in _PLACE_KW:
        if kw in cmd: return 'place'
    for kw in _PICKUP_KW:
        if kw in cmd: return 'pickup'
    return 'unknown'


def _detect_target_object(cmd: str) -> Optional[str]:
    """Return canonical object name if found in command, else None."""
    for obj in _KNOWN_OBJECTS:
        if obj in cmd:
            return _OBJ_ALIASES[obj]
    return None


def _detect_target_location(cmd: str) -> Optional[str]:
    """Return canonical location name if found in command, else None."""
    for loc in _KNOWN_LOCATIONS:
        if loc in cmd:
            return _LOCATION_ALIASES[loc]
    return None


def _objects_match(cur_cmd: str, ret_cmd: str, action_type: str = "unknown") -> bool:
    """True if both commands target the same canonical object/location (or either unknown).

    For move commands: compares target locations instead of physical objects.
    For manipulation commands: compares physical object names.
    """
    if action_type == 'move':
        cur = _detect_target_location(cur_cmd)
        ret = _detect_target_location(ret_cmd)
    else:
        cur = _detect_target_object(cur_cmd)
        ret = _detect_target_object(ret_cmd)
    if cur is None or ret is None:
        return True       # unknown → benefit of doubt
    return cur == ret


def decide_risk(
    sim: float,
    action_match: bool,
    object_match: bool,
    cur_type: str = "unknown",
    ret_type: str = "unknown",
) -> str:
    """Layer 3 risk rules (pure function): REUSE / ADAPT / CLARIFY for a feasible candidate plan.

    1. REUSE   - action match and object match and sim >= 0.85
    2. ADAPT   - action match and sim >= 0.50 (same action, different parameters)
    3. ADAPT   - compatible action pair (pickup<->handover, place<->handover) and sim >= 0.75
    4. CLARIFY - otherwise (ask the human)
    """
    compatible = action_match or (cur_type, ret_type) in _COMPATIBLE_PAIRS
    if action_match and object_match and sim >= 0.85:
        return "REUSE"
    if action_match and sim >= 0.50:
        return "ADAPT"
    if compatible and sim >= 0.75:
        return "ADAPT"
    return "CLARIFY"


_ACTION_TYPE_ZH = {
    "pickup": "夾取／取物",
    "place": "放置",
    "move": "移動",
    "handover": "交接",
    "gripper": "夾爪操作",
    "unknown": "未明確辨識的動作",
}


def explain_rag_clarify_zh(score: Dict[str, Any]) -> str:
    """One short TTS-friendly sentence for Plan RAG CLARIFY (Layer 3).

    Long rationale belongs in logs, not in the human-facing ask string.
    """
    sim = float(score.get("sim") or 0.0)
    am = bool(score.get("action_match"))
    cur = str(score.get("cur_type") or "unknown")
    ret = str(score.get("ret_type") or "unknown")
    cz = _ACTION_TYPE_ZH.get(cur, cur)
    rz = _ACTION_TYPE_ZH.get(ret, ret)

    logger.debug(
        "RAG CLARIFY: sim=%.3f action_match=%s cur=%s ret=%s",
        sim,
        am,
        cur,
        ret,
    )

    if am:
        return f"與舊流程語意約 {sim:.0%}，不確定是否同一任務。"
    if cur == "unknown" and ret == "unknown":
        return f"動作類型對不起來，語意約 {sim:.0%}。"
    return f"你說的像「{cz}」，舊流程像「{rz}」，語意約 {sim:.0%}。"


class CollaborationPolicyModule:
    """
    Execution-feasibility-aware experience reuse policy.

    Decision pipeline — two explicit stages:

      Stage 1 — Feasibility Gate (hard, world-state-grounded)
        Checks **object availability** only.  Assembly step index from the
        retrieved plan is **not** a gate: SOP progress is treated as a soft
        reference; humans may request the same motion from any phase.
        A plan that fails object feasibility is REJECTED regardless of
        how semantically similar it is to the current command.

      Stage 2 — Risk Decision (explicit rule-based, NOT a weighted formula)

        Old approach (abandoned):
            score = 0.4·sim + 0.3·align + 0.2·consistency + 0.1·confidence
            Problem: align/consistency compensate for low sim → unsafe REUSE
            Problem: different steps legitimately reuse the same plan type,
                     so step-index proximity is a poor risk signal.

        Current approach — 4-rule explicit constraint cascade:
            Rule 1 REUSE   am ∧ om ∧ sim ≥ 0.85   exact match → reuse directly
            Rule 2 ADAPT   am ∧ sim ≥ 0.50          same action, moderate sim
            Rule 3 ADAPT   compatible(cur,ret) ∧ sim ≥ 0.75   compatible pair
            Rule 4 CLARIFY otherwise                low confidence → ask human

        Rationale:
          • action_match (am): detected via keyword rules (pickup/handover/
            place/move/gripper). Prevents same-object-different-action misuse
            (pickup screw ≠ handover screw — sim can reach 0.94).
          • object_match (om): direct string comparison of target object.
            High sim + am + om = semantically identical → safe REUSE at 0.85.
          • compatible(cur, ret): pickup ↔ handover, place ↔ handover pairs
            share enough structure to warrant ADAPT rather than CLARIFY.
          • Thresholds (0.85 / 0.50 / 0.75) derived from distribution scan
            over 80 real embedding pairs (see experiments chapter).

      REJECT  feasibility failed — plan contextually infeasible; plan fresh

    This module is intentionally stateless: all world/progress access is
    delegated through callbacks to keep integration flexible.
    """

    def __init__(
        self,
        infer_progress_fn: Callable[[object], Dict[str, Any]],
        has_object_available_fn: Callable[[object, str], bool],
    ) -> None:
        self._infer_progress = infer_progress_fn
        self._has_object_available = has_object_available_fn

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def required_objects_from_subcommands(self, sub_commands: List[Dict]) -> List[str]:
        """Extract unique required objects from a plan's sub-command list."""
        req: List[str] = []
        for sc in sub_commands or []:
            if not isinstance(sc, dict):
                continue
            action = sc.get("action")
            if action in ("pickup", "handover", "place"):
                obj = sc.get("target_object")
                if isinstance(obj, str) and obj and obj not in req:
                    req.append(obj)
        return req

    # ------------------------------------------------------------------
    # Stage 1: Feasibility Gate
    # ------------------------------------------------------------------

    def check_feasibility(
        self,
        *,
        retrieved_step_index: Optional[int],
        world_state,
        required_objects: Optional[List[str]] = None,
        retrieved_task_type: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Hard feasibility gate: can this retrieved plan execute right now?

        Evaluates two hard constraints from W_t (global, applied to all candidates):
          1. R_t gripper state       — pickup blocked when gripper already holding
          2. H_t stop signal         — any action blocked on explicit human stop

        Per-candidate O_t object availability (required_objects) is intentionally
        NOT checked here — mismatched objects between the retrieved plan and the
        current request are handled by Layer 3's om (object match) signal, which
        can produce ADAPT rather than an outright REJECT.

        ``retrieved_step_index`` and ``required_objects`` are accepted for API
        compatibility but **not** used to reject plans.

        Returns:
            (feasible, reason)
            feasible = True  → plan may proceed to reliability estimation
            feasible = False → plan must be REJECTED; reason is machine-readable
        """
        _ = retrieved_step_index  # API compatibility; step is not a feasibility signal
        _ = required_objects       # per-candidate O_t check removed; handled by Layer 3

        self._infer_progress(world_state)  # still run callback for side effects / logging

        # ── R_t: Gripper state ───────────────────────────────────────────────
        # Block pickup only when gripper is confirmed holding an object (DI truth).
        # We do NOT block handover here: "give me X" may need ADAPT (pickup→handover),
        # so a retrieved handover plan with gripper=open should reach Layer 3, not
        # be rejected here.
        robot = getattr(world_state, "robot", None)
        if robot and retrieved_task_type:
            act = (retrieved_task_type or "").lower()
            holding = getattr(robot, "holding_object_id", None)
            if act == "pickup" and holding is not None:
                return False, f"gripper_holding_object:{holding}"

        # ── H_t: Human stop signal ───────────────────────────────────────────
        # Explicit stop gesture is a safety interrupt — block any retrieved plan.
        human = getattr(world_state, "human", None)
        if human:
            intent = getattr(human, "intent_stable", "") or ""
            if "stop" in intent.lower():
                return False, "human_stop_signal"

        return True, ""

    def check_global_feasibility(
        self,
        *,
        world_state,
        current_action_type: Optional[str] = None,
        current_target_object: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Global Layer 2 gate (H_t, R_t, current target O_t).

        Applies uniformly to all top-k candidates; a failure here rejects all.
        H_t (stop gesture) and R_t (gripper state) are safety hard-stops.
        The current command's explicit target object is also checked: if the
        user asks for an invisible object, no retrieved plan can fix that.

        Returns (feasible, reason).
        """
        # H_t: safety interrupt — highest priority, block everything
        human = getattr(world_state, "human", None)
        if human:
            intent = getattr(human, "intent_stable", "") or ""
            if "stop" in intent.lower():
                return False, "human_stop_signal"

        # R_t: gripper already holding → block pickup regardless of candidate
        if current_action_type == "pickup":
            robot = getattr(world_state, "robot", None)
            if robot:
                holding = getattr(robot, "holding_object_id", None)
                if holding is not None:
                    return False, f"gripper_holding_object:{holding}"

        # O_t extension: user's explicit target must be visible now
        if current_target_object and not self._has_object_available(world_state, current_target_object):
            return False, f"current_target_unavailable:{current_target_object}"

        return True, ""

    def check_object_feasibility(
        self,
        *,
        world_state,
        required_objects: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Retained for API compatibility; per-candidate O_t check is no longer used.

        Object mismatches between retrieved and current plan are delegated to
        Layer 3's om (object match) signal (ADAPT / PLAN), not rejected here.
        Always returns (True, "").
        """
        _ = world_state
        _ = required_objects
        return True, ""

    # ------------------------------------------------------------------
    # Stage 2 + 3: Full decision pipeline
    # ------------------------------------------------------------------

    def score_and_select_policy(
        self,
        *,
        sim: float,
        retrieved_step_index: Optional[int] = None,
        world_state,
        required_objects: Optional[List[str]] = None,
        # Layer 3 inputs — used for action/object matching
        retrieved_task_type: Optional[str] = None,   # from DB task_type column
        current_command: Optional[str] = None,        # raw NL command
        retrieved_command: Optional[str] = None,      # stored NL command in DB
        naive_mode: bool = False,
        use_feasibility_gate: bool = True,
        # Legacy keywords kept for backward-compat; ignored.
        sim_top2: Optional[float] = None,
        retrieved_action_type: Optional[str] = None,
        current_action_type: Optional[str] = None,
        current_target_object: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Two-stage experience reuse decision.

        Stage 1 — Feasibility Gate  (Layer 2, world-state-grounded, hard constraints)
          Checks object availability only (not SOP step index).
          Failure → REJECT regardless of similarity.

        Stage 2 — Risk Decision  (Layer 3, three explicit rules)

          Evaluated offline on annotated embedding pairs; see ``evaluation/exp3_execution_risk.py``.

          Inputs:
            sim              cosine similarity between current command and retrieved
                             plan's stored command embedding
            retrieved_task_type  task_type stored in DB for the retrieved plan
            current_command  raw NL command (for lightweight keyword detection)
            retrieved_command stored NL command in DB (for object comparison)

          Decision:
            Rule 1 — REUSE
              sim ≥ 0.85  AND  action_match  AND  object_match
              → embedding is very close AND same action type AND same target object
              → plan parameters transfer safely without LLM modification

            Rule 2 — ADAPT  (same-action, different params)
              action_match  AND  sim ≥ 0.50
              → same action type but different object/target/recipient
              → LLM adjusts parameters (coordinates, target, orientation)

            Rule 3 — ADAPT  (related-action mismatch, high sim)
              compatible(cur_type, ret_type)  AND  sim ≥ 0.75
              Compatible pairs: pickup↔handover, place↔handover
              → related actions (both involve grasping + object positioning)
              → LLM can bridge the gap

            Rule 4 — CLARIFY  (everything else)
              Incompatible action types (e.g. pickup↔place, gripper↔handover),
              or insufficient semantic similarity for LLM to safely adapt

          Why keyword-based action detection is valid here:
            The action vocabulary in assembly HRC is small and well-defined
            (pickup / handover / place / move / gripper — ~5 types).
            Keyword rules achieve reliable detection without LLM overhead.
            Retrieved plan's type comes directly from DB (100% reliable).
            Known limitation: recipient differences ("給我" vs "給操作員") and
            object-size variants ("大螺絲" vs "小螺絲") may not be distinguished.

        Params:
          sim                  cosine similarity of retrieved plan [0, 1]
          required_objects     objects the retrieved plan needs (for Stage 1)
          retrieved_step_index step index from DB (ignored by feasibility gate;
                             may still be logged / shown in UI)
          naive_mode           ablation: always REUSE when sim > 0
          use_feasibility_gate False = skip Stage 1 (ablation only)

        Returns dict:
          sim, policy, feasible, feasibility_reason, current_step_index
        """
        sim = max(0.0, min(1.0, float(sim or 0.0)))
        prog = self._infer_progress(world_state)

        # ── Naive-mode ablation baseline ─────────────────────────────────────
        if naive_mode:
            return {
                "sim": sim,
                "policy": "REUSE" if sim > 0.0 else "CLARIFY",
                "feasible": True,
                "feasibility_reason": "",
                "current_step_index": None,
            }

        # ── Stage 1: Feasibility Gate (Layer 2) ────────────────────────────────────
        if use_feasibility_gate:
            feasible, feasibility_reason = self.check_feasibility(
                retrieved_step_index=retrieved_step_index,
                world_state=world_state,
                required_objects=required_objects,
                retrieved_task_type=retrieved_task_type,
            )
            if not feasible:
                return {
                    "sim": sim,
                    "policy": "REJECT",
                    "feasible": False,
                    "feasibility_reason": feasibility_reason,
                    "current_step_index": prog.get("step_index"),
                }

            # O_t extension: also check the current command's explicit target object.
            # Required objects above come from the *retrieved* plan; this catches the
            # case where the user asks for an object not in the retrieved plan but
            # also not visible in the current world state.
            # Use REJECT (not CLARIFY) so the orchestrator falls through to LLM-A,
            # which will see the enriched W_t prompt (objects list) and can respond
            # with a natural "I cannot see <object>" message instead of a confusing
            # "should I reuse this plan?" question.
            if current_target_object and not self._has_object_available(world_state, current_target_object):
                return {
                    "sim": sim,
                    "policy": "REJECT",
                    "feasible": False,
                    "feasibility_reason": f"current_target_unavailable:{current_target_object}",
                    "current_step_index": prog.get("step_index"),
                }
        else:
            feasible, feasibility_reason = True, ""

        # ── Stage 2: Risk Decision (Layer 3) ──────────────────────────────────────
        cur_cmd = current_command or ""
        ret_cmd = retrieved_command or ""
        cur_type = _detect_action_type(cur_cmd) if cur_cmd else "unknown"
        ret_type = (retrieved_task_type or "unknown").lower()

        am = (cur_type == ret_type) and cur_type != "unknown"
        om = _objects_match(cur_cmd, ret_cmd, action_type=cur_type if am else "unknown")
        policy = decide_risk(sim, am, om, cur_type, ret_type)

        return {
            "sim": sim,
            "cur_type": cur_type,
            "ret_type": ret_type,
            "action_match": am,
            "object_match": om,
            "policy": policy,
            "feasible": True,
            "feasibility_reason": "",
            "current_step_index": prog.get("step_index"),
        }
