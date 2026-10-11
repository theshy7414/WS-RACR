"""ANSWER path: SOP retrieval and question answering (thesis Ch. 4, interaction log + SOP knowledge).

Routes a question to world-state evidence, SOP knowledge or both, and composes the spoken answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

from ws_racr.config import get_model_name
from ws_racr.memory.sop_step_knowledge import STEP_KNOWLEDGE as _STEP_KNOWLEDGE
from ws_racr.memory.sop_rich_content import RICH_SOP_ENTRIES as _RICH_SOP_ENTRIES, seed as _seed_rich_sop
from ws_racr.orchestrator.text_utils import normalize_stt_assembly_query


class AnswerMixin:
    """ANSWER-path methods of :class:`HRCOrchestrator` (mixed in)."""

    def _ensure_sop_seeded(self) -> None:
        """
        On first start, populate the `sops` table from the hardcoded
        _STEP_KNOWLEDGE dict so that semantic SOP search works immediately
        without manual data entry.  Idempotent: skips if the table is
        already non-empty.
        """
        try:
            tm = getattr(self.controller, "task_manager", None)
            if tm is None:
                return
            # Check if table already has rows
            from sqlalchemy import text as _text
            with tm.engine.connect() as conn:
                count = conn.execute(_text("SELECT COUNT(*) FROM sops")).scalar() or 0
            if count > 0:
                return  # already seeded

            logger.info("SOP 知識庫空白，正在從 _STEP_KNOWLEDGE 自動建立...")
            for step_class, info in _STEP_KNOWLEDGE.items():
                if step_class == "idle":
                    continue
                step_no  = info.get("step", "?")
                zh_desc  = info.get("zh", "")
                objs     = ", ".join(info.get("required_objects", []))
                assist   = info.get("robot_assistance", "")
                title    = f"步驟 {step_no}：{zh_desc}"
                content  = (
                    f"步驟 {step_no}：{zh_desc}。"
                    f"所需物件：{objs}。"
                    f"機器人協助方式：{assist}。"
                )
                tm.add_sop(
                    title=title,
                    content=content,
                    category="assembly_sop",
                    tags=[step_class, f"step_{step_no}"],
                    metadata={"step_class": step_class, "step": step_no},
                )
            logger.info(f"SOP 知識庫建立完成（{len(_STEP_KNOWLEDGE) - 1} 筆）")
        except Exception as e:
            logger.warning("_ensure_sop_seeded failed (non-critical): %s", e)

    def _ensure_rich_sop_seeded(self) -> None:
        """Seed detailed chunked SOP entries from sop_rich_content.py.

        Only inserts entries whose `content` field is non-empty.
        Uses a metadata marker row to detect whether this seeder has
        already run, so it is safe to call on every startup.
        """
        if not _RICH_SOP_ENTRIES or _seed_rich_sop is None:
            return
        try:
            tm = getattr(self.controller, "task_manager", None)
            if tm is None:
                return
            from sqlalchemy import text as _text
            with tm.engine.connect() as conn:
                already = conn.execute(_text(
                    "SELECT COUNT(*) FROM sops WHERE category LIKE 'assembly_sop_%'"
                )).scalar() or 0
            if already > 0:
                return  # rich content already seeded
            filled = [e for e in _RICH_SOP_ENTRIES if (e.get("content") or "").strip()]
            if not filled:
                return  # nothing to seed yet
            n = _seed_rich_sop(tm)
            if n:
                logger.info(f"Rich SOP 知識庫建立完成（{n} 筆詳細內容）")
        except Exception as e:
            logger.warning("_ensure_rich_sop_seeded failed (non-critical): %s", e)

    def _search_sop(self, query: str, k: int = 2) -> str:
        """
        Semantic search over the `sops` table.

        Returns a formatted string for injection into LLM prompts, or an
        empty string when no relevant SOP is found or the table is unavailable.
        """
        try:
            tm = getattr(self.controller, "task_manager", None)
            if tm is None:
                return ""
            embedding = tm.model.encode(query).astype("float32").tolist()
            emb_str   = str(embedding)
            from sqlalchemy import text as _text
            with tm.engine.connect() as conn:
                rows = conn.execute(_text("""
                    SELECT title, content,
                           (content_embedding <=> :emb) AS dist
                    FROM sops
                    ORDER BY content_embedding <=> :emb
                    LIMIT :k
                """), {"emb": emb_str, "k": k}).fetchall()
            if not rows:
                return ""
            # Only keep rows whose cosine distance is below 0.55 (similarity > 0.45)
            relevant = [(r.title, r.content) for r in rows if r.dist < 0.55]
            if not relevant:
                return ""
            lines = ["=== 相關 SOP 知識 ==="]
            for title, content in relevant:
                lines.append(f"• {title}\n  {content}")
            return "\n".join(lines) + "\n"
        except Exception as e:
            logger.debug("_search_sop failed: %s", e)
            return ""

    _STATE_QUERY_MARKERS = (
        "桌上", "有哪些", "有什麼", "幾個", "在哪", "哪裡", "位置", "現在", "目前",
    )

    _SOP_QUERY_MARKERS = (
        "下一步", "接下來", "步驟", "流程", "先做", "需要什麼", "要準備什麼",
    )

    def _route_query_type(self, query: str) -> str:
        """
        Stage 1 of the ANSWER path - query router.

        Returns one of: state | sop | hybrid
        """
        q = (query or "").strip()
        has_state = any(m in q for m in self._STATE_QUERY_MARKERS)
        has_sop = any(m in q for m in self._SOP_QUERY_MARKERS)
        if has_state and has_sop:
            return "hybrid"
        if has_state:
            return "state"
        return "sop"

    def _collect_query_evidence(self, query: str, qtype: str, world_state) -> Dict[str, Any]:
        """
        Stage 2 of the ANSWER path - evidence collector.
        """
        evidence: Dict[str, Any] = {"qtype": qtype, "state_text": "", "sop_text": ""}
        q = (query or "").strip()

        # Collect state evidence via existing query handler
        if qtype in ("state", "hybrid"):
            try:
                vqh = getattr(self.controller, "voice_query_handler", None)
                if vqh is not None:
                    qr = vqh.parse_and_execute(q)
                    if isinstance(qr, dict) and qr.get("success"):
                        txt = qr.get("response_text")
                        if txt:
                            evidence["state_text"] = txt
            except Exception as e:
                logger.debug("state evidence collection failed: %s", e)

        # Collect SOP evidence by retrieval
        if qtype in ("sop", "hybrid"):
            evidence["sop_text"] = self._search_sop(q)

        # Add progress snapshot for hybrid/sop answering quality
        evidence["progress"] = self._infer_progress(world_state)
        return evidence

    def _compose_query_answer(self, query: str, qtype: str, evidence: Dict[str, Any], world_state) -> str:
        """
        Stage 3 of the ANSWER path - answer composer.
        """
        q = (query or "").strip()
        state_text = (evidence.get("state_text") or "").strip()
        sop_text = (evidence.get("sop_text") or "").strip()

        # Fast path: state query should return deterministic answer when possible.
        if qtype == "state" and state_text:
            return state_text

        # Keep deterministic handling for key progress/feasibility questions.
        # Current step query
        if any(k in q for k in ("現在", "這一步", "這步", "目前", "當前")) and any(k in q for k in ("是什麼", "要做", "做什麼", "幹什麼", "幹嘛", "哪一步", "第幾步", "呢", "嗎", "?", "？")):
            prog = self._infer_progress(world_state)
            try:
                from ws_racr.world_state.step_definitions import get_step_name_zh, step_index_to_user_display
            except Exception:
                get_step_name_zh = None  # type: ignore
                step_index_to_user_display = None  # type: ignore
            if prog["step_class"] == "idle":
                return "目前系統處於待機狀態，尚未開始任何步驟。"
            step_name = get_step_name_zh(prog["step_index"]) if get_step_name_zh else prog["step_class"]
            user_no = step_index_to_user_display(prog["step_index"]) if step_index_to_user_display else prog["step_index"] + 1
            req = (_STEP_KNOWLEDGE.get(prog["step_class"], {}) or {}).get("required_objects", [])
            return f"現在是第{user_no}步：{step_name}，需要：{', '.join(req)}。" if req else f"現在是第{user_no}步：{step_name}。"

        if any(k in q for k in ("下一步", "接下來")) and any(k in q for k in ("是什麼", "要做", "做什麼", "呢", "嗎", "?", "？")):
            prog = self._infer_progress(world_state)
            try:
                from ws_racr.world_state.step_definitions import get_step_name_zh
            except Exception:
                get_step_name_zh = None  # type: ignore
            if prog["next_step_index"] is None:
                return "恭喜！所有步驟已完成。"
            next_name = get_step_name_zh(prog["next_step_index"]) if get_step_name_zh else str(prog["next_step_class"])
            req = (_STEP_KNOWLEDGE.get(prog["next_step_class"], {}) or {}).get("required_objects", [])
            return f"下一步是{next_name}，需要：{', '.join(req)}。" if req else f"下一步是{next_name}。"

        if ("鎖" in q and "螺絲" in q) and any(k in q for k in ("可以", "能", "現在")):
            prog = self._infer_progress(world_state)
            near_ok = prog["step_index"] in (1, 2, 3)
            has_wrench = self._has_object_available(world_state, "hex_wrench")
            has_screw = self._has_object_available(world_state, "screw_small")
            if near_ok and has_wrench and has_screw:
                return "可以，扳手與小螺絲都在，目前步驟也符合。需要我先把扳手拿給你嗎？"
            missing = []
            if not has_wrench:
                missing.append("六角扳手")
            if not has_screw:
                missing.append("小螺絲")
            if not near_ok:
                missing.append("目前步驟尚未到鎖小螺絲階段")
            tip = "、".join(missing) if missing else "條件不足"
            return f"可能還不行：{tip}。需要我幫你先準備缺的工具/零件嗎？"

        # Hybrid/SOP composition through LLM with grounded evidence.
        if qtype == "hybrid" and state_text and sop_text:
            sys_prompt = (
                "你是人機協作組裝助理。請基於提供的現場狀態與 SOP 證據，"
                "用 1-2 句繁體中文回答。不可編造未出現的事實。"
            )
            user_prompt = f"=== 現場狀態 ===\n{state_text}\n\n=== SOP 證據 ===\n{sop_text}\n\n問題：{q}"
        elif sop_text:
            sys_prompt = (
                "你是一個人機協作組裝任務的助理機器人。"
                "根據提供的 SOP 知識回答使用者問題，用 1-2 句繁體中文回答。"
                "若證據不足，明確說明不知道。"
            )
            user_prompt = f"{sop_text}\n問題：{q}"
        elif state_text:
            return state_text
        else:
            return "我目前沒有這方面的資訊。"

        try:
            client = self._get_llm_client()
            resp = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("_compose_query_answer LLM call failed: %s", e)
            return state_text or "抱歉，我現在無法回答這個問題。"

    def _tool_sop_answer(self, query: str, world_state=None) -> str:
        """
        Three-layer answer pipeline (after Orchestrator chooses ANSWER):
          1) Query Router
          2) Evidence Collector
          3) Answer Composer
        """
        q = normalize_stt_assembly_query((query or "").strip())
        # 步驟 N 所需材料：優先走 YAML SOP（VoiceQueryHandler），避免僅靠向量片段漏答
        if re.search(r"第\s*[\d一二三四五六七八九十百零〇两兩]+\s*步", q) and (
            "需要" in q or "材料" in q or "零件" in q or "工具" in q
        ):
            vqh = getattr(self.controller, "voice_query_handler", None)
            if vqh is not None:
                try:
                    qr = vqh.parse_and_execute(q)
                    if isinstance(qr, dict) and qr.get("success") and (qr.get("response_text") or "").strip():
                        answer = (qr.get("response_text") or "").strip()
                        # priority 須 <「執行第N步」前綴（見 run 多步迴圈）；TTS 越小越先播
                        self._tool_answer(answer, blocking=True, priority=5)
                        return answer
                except Exception as e:
                    logger.debug("step_materials fast-path (VQH) failed: %s", e)

        qtype = self._route_query_type(q)
        evidence = self._collect_query_evidence(q, qtype, world_state)
        answer = self._compose_query_answer(q, qtype, evidence, world_state)

        self._tool_answer(answer, blocking=True, priority=5)
        return answer
