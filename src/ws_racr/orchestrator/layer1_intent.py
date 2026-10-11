"""Layer 1 - instruction understanding (thesis Ch. 4, Layer 1).

Rule-based segmentation of a spoken command into sub-intents (EXECUTE / ANSWER / ASK) and
slot extraction (action, target object, location, spatial hints). No LLM call is made here.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntentMixin:
    """Layer 1 methods of :class:`HRCOrchestrator` (mixed in; relies on ``self`` state of the orchestrator)."""

    # question-marker detection (shared with database_controller heuristic)
    _Q_MARKERS = (
        "哪些", "什麼", "哪裡", "在哪", "幾個", "幾步", "多少",
        "嗎", "呢", "哪邊", "哪方向", "需要", "還缺", "還在",
        "怎麼", "如何", "告訴我", "說明", "是什麼",
    )

    # Strong question signals: unambiguous regardless of surrounding words.
    # Used by Layer 0 pre-dispatch gate to fast-path ANSWER without calling LLM-A.
    _Q_STRONG = (
        "嗎", "呢", "怎麼", "如何", "告訴我", "說明",
        "幾步", "幾個", "多少", "是什麼", "有哪些", "哪些",
    )

    # Command-verb prefixes that override any question marker.
    # If the utterance starts with one of these it is an action, not a query.
    _ACTION_PREFIX = (
        "幫我", "請", "把", "將", "拿", "放", "移", "鎖", "取",
        "夾", "遞", "轉", "鬆", "開始", "停止", "繼續",
    )

    # Connective words used to segment multi-intent commands.
    _CONNECTIVES = ("然後", "接著", "再來", "之後", "並且", "順便", "再(?!說|次|三)")

    # Vague pronouns that indicate missing grounding (ASK trigger).
    _VAGUE_REFS = ("那個", "這個", "它", "那邊", "這邊", "那", "那些", "這些")

    # Action verb prefixes for EXECUTE classification.
    _EXECUTE_VERBS = (
        "幫我", "把", "將", "拿", "放", "夾", "移", "鎖", "取",
        "插", "對齊", "遞", "固定", "套", "穩", "推", "夾起",
    )

    _STRUCTURED_ROBOT_ACTIONS = frozenset(
        {
            "pickup",
            "place",
            "handover",
            "move",
            "gripper_open",
            "gripper_close",
            "shift",
            "rotate_gripper",
            "return_to_origin",
            "stop",
            "emergency_stop",
            "insert",
            "align",
            "tighten",
            "hold",
        }
    )

    def _is_question(self, text: str) -> bool:
        return any(m in text for m in self._Q_MARKERS) or text.rstrip().endswith("？")

    def _is_strong_question(self, text: str) -> bool:
        """Layer 0 fast-path: only fires on unambiguous question signals.

        Returns True when the utterance is clearly a query (no LLM-A needed).
        Deliberately conservative: ambiguous cases return False so they fall
        through to _think() and let LLM-A decide.
        """
        t = text.strip()
        if not t:
            return False
        # Explicit action verb prefix → always a command, never a query
        if any(t.startswith(p) for p in self._ACTION_PREFIX):
            return False
        # Explicit question mark → query
        if t.endswith("？") or t.endswith("?"):
            return True
        # Strong question words → query
        return any(m in t for m in self._Q_STRONG)

    def _segment_command(self, command: str) -> List[str]:
        """Split a command string into sub-intent spans on coordination connectives.

        Examples
        --------
        "幫我拿圓柱，然後告訴我下一步" → ["幫我拿圓柱", "告訴我下一步"]
        "先把支架放好再對齊平台"        → ["先把支架放好", "對齊平台"]
        "把圓柱夾起來"                  → ["把圓柱夾起來"]
        """
        import re
        # Split on connective words (with optional leading comma/pause)
        pattern = r'[，,]?\s*(?:然後|接著|再來|之後|並且|順便|再(?!說|次|三))'
        parts = re.split(pattern, command)
        # Remove "先" standalone prefix (e.g. "先把" → keep as is)
        cleaned = [p.strip().lstrip("先") if p.strip().startswith("先把") or
                   p.strip().startswith("先幫") or p.strip().startswith("先拿")
                   else p.strip() for p in parts]
        return [p for p in cleaned if p]

    def _classify_span(self, span: str) -> str:
        """Classify a single span as EXECUTE, ANSWER, or ASK.

        Priority order:
          1. ANSWER  — contains question markers or ends with ？
          2. EXECUTE — starts with a known action verb
          3. ASK     — contains vague/unresolved references
          4. EXECUTE — default (assume action intent)
        """
        # ANSWER: question signal dominates (unless action prefix overrides)
        if self._is_question(span):
            # Even question-like spans starting with action verbs are EXECUTE
            if any(span.startswith(p) for p in self._ACTION_PREFIX):
                pass  # fall through to EXECUTE check
            else:
                return "ANSWER"
        # Strong question markers: always ANSWER even without ？
        if self._is_strong_question(span):
            return "ANSWER"
        # EXECUTE: starts with known action verb
        if any(span.startswith(v) for v in self._EXECUTE_VERBS):
            # But if the target is vague → ASK
            if any(ref in span for ref in self._VAGUE_REFS):
                if not any(obj in span for obj in (
                    "圓柱體", "圓柱", "螺絲", "扳手", "平台", "支架", "PI", "螺帽", "大螺絲", "小螺絲"
                )):
                    return "ASK"
            return "EXECUTE"
        # ASK: vague reference with no clear target
        if any(ref in span for ref in self._VAGUE_REFS):
            return "ASK"
        # Default: treat as EXECUTE
        return "EXECUTE"

    def _extract_execute_slots(self, span: str) -> Dict[str, Optional[str]]:
        """Extract action, target_object, location, spatial_modifier, object_size.

        Uses keyword matching against known assembly objects and action verbs.
        Falls back to None when information cannot be determined from text alone.
        """
        # Object keyword mapping (longest match priority)
        _OBJ_MAP = [
            ("小螺絲", "screw_small"), ("大螺絲", "screw_big"),
            ("螺絲", "screw_small"),   ("螺帽", "nut"),
            ("扳手", "hex_wrench"),    ("圓柱體", "cylinder"), ("圓柱", "cylinder"),
            ("平台", "platform"),      ("支架", "bracket"),
            ("PI", "pi"),
        ]
        # Action keyword mapping
        _ACT_MAP = [
            ("插入", "insert"), ("插進", "insert"), ("插好", "insert"),
            ("插", "insert"),   ("對齊", "align"),  ("鎖緊", "tighten"),
            ("鎖好", "tighten"),("鎖上", "tighten"),("鎖", "tighten"),
            ("遞給", "handover"),("遞過", "handover"),("遞", "handover"),
            ("夾起", "pickup"), ("夾", "pickup"),   ("拿過", "pickup"),
            ("拿來", "pickup"), ("拿", "pickup"),   ("放好", "place"),
            ("放到", "place"),  ("放上", "place"),  ("放", "place"),
            ("固定", "place"),  ("移到", "place"),  ("移", "place"),
            ("穩住", "hold"),   ("推到", "place"),
        ]
        # Location keyword mapping
        _LOC_MAP = [
            ("上方孔位", "upper_hole"), ("下方孔位", "lower_hole"),
            ("平台上", "platform"),     ("支架上", "bracket"),
            ("桌上", "table"),          ("定位點", "fixture"),
            ("孔位", "hole"),           ("旁邊", "aside"),
        ]

        target_object = None
        for kw, obj in _OBJ_MAP:
            if kw in span:
                target_object = obj
                break

        action = None
        for kw, act in _ACT_MAP:
            if kw in span:
                action = act
                break

        target_location = None
        for kw, loc in _LOC_MAP:
            if kw in span:
                target_location = loc
                break

        # Spatial / size hints — must survive structured execute → GraspPlanner / WorldState
        spatial_modifier = None
        _SPATIAL_PATTERNS = (
            ("最右邊", "最右邊"), ("最右側", "最右邊"), ("最右边", "最右邊"),
            ("最左邊", "最左邊"), ("最左側", "最左邊"), ("最左边", "最左邊"),
            ("最前面", "最前面"), ("最後面", "最後面"),
            ("最近", "最近"), ("最遠", "最遠"), ("中間", "中間"),
            ("rightmost", "rightmost"), ("leftmost", "leftmost"),
            ("nearest", "nearest"), ("farthest", "farthest"),
        )
        for kw, canon in _SPATIAL_PATTERNS:
            if kw in span:
                spatial_modifier = canon
                break

        object_size = None
        if any(x in span for x in ("大螺絲", "大號螺絲", "大的螺絲")):
            object_size = "大"
        elif any(x in span for x in ("小螺絲", "小號螺絲", "小的螺絲")):
            object_size = "小"
        elif "螺絲" in span and "大" in span:
            object_size = "大"
        elif "螺絲" in span and "小" in span:
            object_size = "小"

        return {
            "action": action,
            "target_object": target_object,
            "target_location": target_location,
            "spatial_modifier": spatial_modifier,
            "object_size": object_size,
        }

    def _sanitize_llm_single_execute(
        self,
        command: str,
        world_state,
        sub_commands: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """If LLM-A emits a non-executable placeholder (e.g. action=ask) but Layer 1
        keyword slots yield a concrete EXECUTE, prefer the rule-based step so
        execute_task_from_structured does not fail validation.
        """
        if not sub_commands or not isinstance(sub_commands[0], dict):
            return sub_commands
        llm_step = sub_commands[0]
        act = (llm_step.get("action") or "").strip().lower()
        if act in self._STRUCTURED_ROBOT_ACTIONS:
            return sub_commands

        intents = self._parse_intents(command, world_state)
        if len(intents) != 1 or intents[0].get("type") != "EXECUTE":
            return sub_commands
        it = intents[0]
        slot_action = it.get("action")
        slot_obj = it.get("target_object")
        if not slot_action or not slot_obj:
            return sub_commands

        fixed: Dict[str, Any] = {
            "action": slot_action,
            "target_object": slot_obj,
        }
        if it.get("target_location"):
            fixed["target_location"] = it["target_location"]
        if it.get("spatial_modifier"):
            fixed["spatial_modifier"] = it["spatial_modifier"]
        if it.get("object_size"):
            fixed["object_size"] = it["object_size"]

        logger.info(f"   LLM 單步 action={act!r} 無法送 execute_task，"
            f"改採關鍵字槽位: {fixed.get('action')} {fixed.get('target_object')}")
        return [fixed]

    def _inject_spatial_hints_from_intents(
        self,
        intents: List[Dict[str, Any]],
        sub_commands: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge rule-based spatial_modifier / object_size from intents into LLM/RAG steps.

        LLM-A or cached RAG plans often omit 最右邊 / 大號等 hints; perception and
        GraspPlanner already honor these fields when present on the structured step.
        """
        if not sub_commands:
            return sub_commands
        spatial_modifier: Optional[str] = None
        object_size: Optional[str] = None
        for it in intents:
            if it.get("type") != "EXECUTE":
                continue
            if spatial_modifier is None and it.get("spatial_modifier"):
                spatial_modifier = str(it["spatial_modifier"])
            if object_size is None and it.get("object_size"):
                object_size = str(it["object_size"])
        if spatial_modifier is None and object_size is None:
            return sub_commands

        out: List[Dict[str, Any]] = []
        for sc in sub_commands:
            if not isinstance(sc, dict):
                out.append(sc)
                continue
            act = sc.get("action") or ""
            merged = dict(sc)
            if spatial_modifier and not merged.get("spatial_modifier"):
                if act in ("pickup", "handover"):
                    merged["spatial_modifier"] = spatial_modifier
            if object_size and not merged.get("object_size"):
                if act in ("pickup", "handover", "insert"):
                    merged["object_size"] = object_size
            out.append(merged)
        return out

    def _parse_intents(
        self, command: str, world_state=None
    ) -> List[Dict[str, Any]]:
        """Parse a command string into a list of sub-intent dicts.

        Each sub-intent has:
          span_text    : original text fragment
          type         : "EXECUTE" | "ANSWER" | "ASK"
          action       : (EXECUTE) robot action verb
          target_object: (EXECUTE) grounded object name
          target_location: (EXECUTE) optional destination
          spatial_modifier / object_size: (EXECUTE) optional disambiguation hints
          query        : (ANSWER) the question text

        Design notes
        ------------
        - Rule-based segmentation on connective words (zero LLM cost).
        - Single-segment commands return a list with one element.
        - LLM-A downstream receives the parsed intents as structured context,
          which replaces the previous whole-utterance retrieval query.
        """
        segments = self._segment_command(command)
        intents: List[Dict[str, Any]] = []

        for seg in segments:
            itype = self._classify_span(seg)
            entry: Dict[str, Any] = {
                "span_text": seg,
                "type": itype,
                "action": None,
                "target_object": None,
                "target_location": None,
                "spatial_modifier": None,
                "object_size": None,
                "query": None,
            }
            if itype == "EXECUTE":
                slots = self._extract_execute_slots(seg)
                entry.update(slots)
                # If object occluded in world state, demote to ASK
                if world_state is not None and entry["target_object"]:
                    try:
                        occ = getattr(world_state, "occluded_objects", []) or []
                        if entry["target_object"] in occ:
                            entry["type"] = "ASK"
                    except Exception:
                        pass
            elif itype == "ANSWER":
                entry["query"] = seg
            # ASK: no slots needed; the whole span is the clarification trigger

            intents.append(entry)

        logger.debug("_parse_intents(%r) → %s", command,
                     [(i["type"], i["span_text"]) for i in intents])
        return intents
