"""Plan RAG glue (thesis Ch. 4, plan retrieval and write-back).

Retrieval only *recalls candidates* from the plan memory; the REUSE / ADAPT / CLARIFY / REJECT
decision is made by :class:`ws_racr.decision.collaboration_policy.CollaborationPolicyModule`
(Layer 2 feasibility gate and Layer 3 risk rules).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ws_racr.orchestrator.constants import PLAN_RAG_HINT_THRESHOLD


class PlanMemoryMixin:
    """Plan-memory helpers of :class:`HRCOrchestrator` (mixed in)."""

    def _infer_retrieved_action_type(self, rag_top: Dict, sub_cmds: List) -> Optional[str]:
        """Derive an action label for the retrieved plan (column may be multi_plan)."""
        _SKIP = frozenset({"", "unknown", "multi_plan", "multi"})
        for step in sub_cmds or []:
            if not isinstance(step, dict):
                continue
            a = (step.get("action") or "").strip().lower()
            if a and a not in _SKIP:
                return a
        raw = (rag_top.get("task_type") or "").strip().lower()
        if raw and raw not in _SKIP:
            return raw
        return None

    def _ws_compat_info(self, result: Dict, world_state) -> Dict:
        """Return W_t compatibility info for one Plan RAG candidate.

        Returns dict with:
          penalty       float  (<=0) — -0.20 per missing required object, capped at -0.30
          all_visible   bool   — True when all required objects are currently visible
          missing       list   — names of missing required objects
        """
        info = {"penalty": 0.0, "all_visible": True, "missing": []}
        if world_state is None:
            return info
        try:
            sub_cmds = (result.get("plan_json") or {}).get("sub_commands", [])
            required = self.policy.required_objects_from_subcommands(sub_cmds)
            for obj in required:
                if not self._has_object_available(world_state, obj):
                    info["missing"].append(obj)
                    info["penalty"] -= 0.20
            info["penalty"] = max(info["penalty"], -0.30)
            info["all_visible"] = len(info["missing"]) == 0
        except Exception as e:
            logger.debug("_ws_compat_info error: %s", e)
        return info

    def _retrieve_multi_plan(self, command: str, world_state=None) -> Optional[List[Dict]]:
        """Query the vector store for previously completed MULTI plans.

        Returns a list of candidate dicts sorted by descending similarity,
        filtered to those above PLAN_RAG_HINT_THRESHOLD with valid sub_commands.
        Returns None (or empty) when no candidates qualify.

        Candidates are NOT reranked by W_t compatibility here.  The top-1
        (highest-similarity) candidate is forwarded to Layer 3, which decides
        REUSE / ADAPT / PLAN / REJECT based on action and object match signals.
        """
        try:
            tm = getattr(self.controller, "task_manager", None)
            if tm is None:
                return None
            results = tm.search_similar_plans(
                query=command,
                filters={"action": "multi_plan"},
                k=3,
            )
            if not results:
                return None

            candidates: List[Dict] = []
            for r in results:
                sim = float(r["similarity"])
                if sim < PLAN_RAG_HINT_THRESHOLD:
                    continue
                plan_data = r.get("plan_json") or {}
                sub_cmds  = plan_data.get("sub_commands", [])
                if not sub_cmds:
                    continue
                src_cmd      = (r.get("task_json") or {}).get("user_command", "")
                step_idx     = (r.get("task_json") or {}).get("step_index")
                task_type_row = r.get("task_type")
                inferred_tt  = self._infer_retrieved_action_type(r, sub_cmds)
                candidates.append({
                    "sub_commands":   sub_cmds,
                    "similarity":     sim,
                    "source_command": src_cmd,
                    "step_index":     step_idx,
                    "task_type":      inferred_tt or task_type_row,
                })

            if not candidates:
                return None

            top      = candidates[0]
            sim_top2 = candidates[1]["similarity"] if len(candidates) > 1 else None
            logger.info(f"   Plan RAG: 找到 {len(candidates)} 個候選計畫 "
                f"(top sim={top['similarity']:.2f}"
                + (f", top2={sim_top2:.2f}" if sim_top2 is not None else "")
                + f", 步數={len(top['sub_commands'])}"
                f", step_index={top['step_index']}"
                f", 原始指令: 「{top['source_command']}」)")
            return candidates

        except Exception as e:
            logger.debug("_retrieve_multi_plan failed (non-critical): %s", e)
        return None

    def _store_multi_plan(
        self,
        command: str,
        sub_commands: List[Dict],
        session_id: Optional[str],
        world_state=None,
    ) -> None:
        """
        Persist a successfully completed MULTI skill sequence to the RAG store
        so that future similar commands can retrieve and reuse it.

        Stored with action='multi_plan' to allow precise filtering in
        search_similar_plans().  The command text is used as the embedding
        source so semantic search finds it via natural-language similarity.
        """
        try:
            tm = getattr(self.controller, "task_manager", None)
            if tm is None:
                return
            # Attach progress snapshot for progress_alignment scoring later.
            prog = self._infer_progress(world_state)

            task_json = {
                "action":       "multi_plan",
                "user_command": command,
                "sub_commands": sub_commands,
                "step_index":   prog.get("step_index"),
                "step_class":   prog.get("step_class"),
            }
            plan_json = {"sub_commands": sub_commands}
            task_id = tm.save_task(
                user_input=command,
                task_json=task_json,
                plan_json=plan_json,
                source="orchestrator",
                session_id=session_id,
            )
            tm.update_task_status(task_id, "completed")
            logger.info(f"   Plan RAG: 已記錄 MULTI 計畫 {task_id} "
                f"({len(sub_commands)} 步)")
        except Exception as e:
            logger.warning("_store_multi_plan failed (non-critical): %s", e)

    def _required_objects_from_subcommands(self, sub_commands: List[Dict]) -> List[str]:
        return self.policy.required_objects_from_subcommands(sub_commands)

    def _score_and_select_policy(
        self,
        *,
        sim: float,
        sim_top2: Optional[float],
        retrieved_step_index: Optional[int],
        world_state,
        required_objects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self.policy.score_and_select_policy(
            sim=sim,
            sim_top2=sim_top2,
            retrieved_step_index=retrieved_step_index,
            world_state=world_state,
            required_objects=required_objects,
        )

    def _infer_progress(self, world_state) -> Dict[str, Any]:
        """
        Progress inferencer (rule-based, thesis-friendly).

        Primary signal: world_state.step_class + step_confidence.
        Output fields:
          - step_class / step_index
          - next_step_class / next_step_index (or None if finished)
          - confidence (0-1)
        """
        try:
            from ws_racr.world_state.step_definitions import get_step_index, get_step_class
        except Exception:
            # Fallback (should not happen in normal runtime)
            def get_step_index(_c: str) -> int:  # type: ignore
                return 8
            def get_step_class(_i: int) -> str:  # type: ignore
                return "idle"

        step_class = getattr(world_state, "step_class", "idle") if world_state else "idle"
        step_conf  = float(getattr(world_state, "step_confidence", 0.0) or 0.0) if world_state else 0.0
        step_index = get_step_index(step_class)

        # Next step logic follows TSM ordering (0..7 are steps, 8 is idle).
        if step_index == 8:
            next_index = 0
        elif step_index >= 7:
            next_index = None
        else:
            next_index = step_index + 1

        next_class = get_step_class(next_index) if next_index is not None else None

        # Confidence: use step_confidence but clamp; idle with low conf is still OK.
        conf = max(0.0, min(1.0, step_conf))
        if step_class == "idle" and conf < 0.2:
            conf = 0.6

        return {
            "step_class": step_class,
            "step_index": step_index,
            "next_step_class": next_class,
            "next_step_index": next_index,
            "confidence": conf,
        }

    def _has_object_available(self, world_state, object_type: str) -> bool:
        """Return True if object_type is visible or occluded (not missing)."""
        if not world_state or not getattr(world_state, "objects", None):
            return False
        for _oid, obj in world_state.objects.items():
            if getattr(obj, "object_type", None) != object_type:
                continue
            vis = getattr(obj, "visibility", "visible")
            if vis != "missing":
                return True
        return False

    def _rag_clarify_inventory_hint(self, command: str, world_state) -> str:
        """
        If the utterance names a known canonical object that is absent from the
        current world snapshot, add inventory context (orthogonal to Layer 3 CLARIFY).
        """
        from ws_racr.decision.collaboration_policy import _detect_target_object

        canon = _detect_target_object(command or "")
        if not canon or not world_state:
            return ""

        label_map = {
            "cylinder": "圓柱體",
            "screw": "螺絲",
            "bracket": "支架",
            "hex_wrench": "六角扳手",
            "platform": "平台",
            "nut": "螺帽",
        }
        label = label_map.get(canon, canon)
        if self._has_object_available(world_state, canon):
            return ""
        return f"目前沒偵測到「{label}」。"
