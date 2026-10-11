"""Multimodal deictic grounding glue for the orchestrator.

Thin delegation to :class:`ws_racr.decision.deictic_grounding.DeicticGroundingModule` plus the
keyword lists used to recognise pronoun-only commands ("give me that one").
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class DeicticMixin:
    """Deictic-reference helpers of :class:`HRCOrchestrator` (mixed in)."""

    _DEICTIC_PRONOUNS = (
        "那個", "這個", "那邊", "這邊", "那裡", "這裡", "它", "他", "她", "那一個",
        "這一個", "那個給我", "拿給我", "給我", "幫我拿", "幫我夾", "放過去", "移過去",
    )

    _DEICTIC_ACTION_HINTS = (
        "拿", "夾", "取", "給我", "遞給", "放到", "放過去", "移到", "移過去", "交給我",
    )

    _NEW_COMMAND_HINTS = (
        "請", "幫我", "拿", "夾", "取", "放", "移", "轉", "鎖", "鬆", "開始", "停止",
        "再", "先", "現在", "可以", "要不要", "有沒有", "在哪", "有哪些", "什麼", "如何",
    )

    def _get_known_objects(self) -> List[str]:
        try:
            from ws_racr.execution.skills.skill_schema import build_object_list
            return [o.strip() for o in build_object_list().split(",") if o.strip()]
        except Exception:
            return ["bracket", "cylinder", "hex_wrench", "nut", "pi", "platform", "screw_big", "screw_small"]

    def _contains_explicit_object(self, text: str) -> bool:
        return self.grounding.contains_explicit_object(text)

    def _extract_object_from_text(self, text: str) -> Optional[str]:
        return self.grounding.extract_object_from_text(text)

    def _looks_like_new_top_level_command(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if self._is_question(t):
            return True
        return any(h in t for h in self._NEW_COMMAND_HINTS)

    def _has_injected_deictic_target(self, text: str) -> bool:
        return self.grounding.has_injected_deictic_target(text)

    def _is_deictic_command(self, text: str) -> bool:
        return self.grounding.is_deictic_command(text)

    def _is_unresolved_deictic_command(self, text: str) -> bool:
        return self.grounding.is_unresolved_deictic_command(text)

    def _get_live_pointing_target(self, world_state) -> Optional[str]:
        """Try snapshot first, then poll live world state if available."""
        try:
            pt = getattr(getattr(world_state, "human", None), "pointing_target", None)
            if pt:
                return pt
        except Exception:
            pass

        try:
            wsb = getattr(self.controller, "world_state_builder", None)
            if wsb is not None:
                ws_now = wsb.get_current_state()
                return getattr(getattr(ws_now, "human", None), "pointing_target", None)
        except Exception:
            pass
        return None

    def _resolve_multimodal_deictic(self, command: str, world_state) -> str:
        return self.grounding.resolve(
            command,
            world_state,
            get_live_pointing_target_fn=self._get_live_pointing_target,
            ask_fn=self._tool_ask,
        )
