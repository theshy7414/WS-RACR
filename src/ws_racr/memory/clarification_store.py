"""
澄清／語意不明對話的 DB 儲存 + 階層式 LLM 壓縮。

規則（論文／產品友善的簡化版）：
  • 每個 session 最多保留 10 筆「原始」對話列（user + robot 各一行合併成一筆）。
  • 當第 11 筆要寫入前，先將當前 10 筆用 LLM 壓成 1 段摘要，寫入 summary 表 tier=0，再刪除那 10 筆 raw。
  • tier=k 的摘要列若超過 10 筆，再將最舊 10 筆壓成 1 筆 tier=k+1，刪除那 10 筆。
  • 供 Orchestrator LLM 讀取：較高 tier 摘要 + 較低 tier + 最近 raw。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

RECENT_CAP = 10
SUMMARY_ROLLUP_CAP = 10


def _default_llm_compress(lines: List[str]) -> str:
    """將多行對話壓成一段中文摘要；失敗時退回字串拼接。"""
    if not lines:
        return ""
    blob = "\n".join(lines)
    try:
        from ws_racr.config import get_model_name, get_openai_client

        client = get_openai_client()
        model = get_model_name()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": "你是對話壓縮器。只輸出一段簡短中文摘要（≤200字），保留使用者意圖、"
                    "機器人回應重點、是否已澄清物件／任務。不要列表、不要前言。",
                },
                {
                    "role": "user",
                    "content": "以下為連續的人機對話片段，請壓縮：\n\n" + blob,
                },
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or blob[:500]
    except Exception as e:
        logger.warning("ClarificationDialogueStore LLM compress failed: %s", e)
        return blob[:800]


class ClarificationDialogueStore:
    """PostgreSQL 後端；與 TaskManager 共用 engine。"""

    def __init__(self, engine: Engine, compress_fn: Optional[Callable[[List[str]], str]] = None):
        self.engine = engine
        self._compress = compress_fn or _default_llm_compress
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS clarification_dialogue_recent (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL,
                    user_text TEXT NOT NULL,
                    robot_text TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
                )
            )
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS clarification_dialogue_summary (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL,
                    tier INT NOT NULL DEFAULT 0,
                    summary_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
                )
            )
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_clar_recent_sess
                ON clarification_dialogue_recent(session_id, id)
                """
                )
            )
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_clar_sum_sess_tier
                ON clarification_dialogue_summary(session_id, tier, id)
                """
                )
            )

    def append_exchange(
        self,
        session_id: str,
        user_text: str,
        robot_text: Optional[str] = None,
    ) -> None:
        """寫入一筆澄清對話；觸發 10→1 壓縮與 summary 層級 rollup。"""
        sid = (session_id or "default").strip() or "default"
        ut = (user_text or "").strip()
        if not ut:
            return
        rt = (robot_text or "").strip() or ""

        with self.engine.begin() as conn:
            n = conn.execute(
                text(
                    "SELECT COUNT(*) FROM clarification_dialogue_recent "
                    "WHERE session_id = :sid"
                ),
                {"sid": sid},
            ).scalar()
            n = int(n or 0)

            if n >= RECENT_CAP:
                rows = conn.execute(
                    text(
                        "SELECT id, user_text, robot_text FROM clarification_dialogue_recent "
                        "WHERE session_id = :sid ORDER BY id ASC LIMIT :cap"
                    ),
                    {"sid": sid, "cap": RECENT_CAP},
                ).fetchall()
                lines = []
                ids: List[int] = []
                for r in rows:
                    ids.append(int(r[0]))
                    u, rob = r[1] or "", r[2] or ""
                    lines.append(f"使用者：{u}")
                    lines.append(f"機器人：{rob}" if rob else "機器人：（無文字回覆）")

                summary = self._compress(lines)
                conn.execute(
                    text(
                        "INSERT INTO clarification_dialogue_summary "
                        "(session_id, tier, summary_text) VALUES (:sid, 0, :txt)"
                    ),
                    {"sid": sid, "txt": summary},
                )
                if ids:
                    ph = ",".join(str(int(i)) for i in ids)
                    conn.execute(
                        text(
                            f"DELETE FROM clarification_dialogue_recent "
                            f"WHERE session_id = :sid AND id IN ({ph})"
                        ),
                        {"sid": sid},
                    )

            conn.execute(
                text(
                    "INSERT INTO clarification_dialogue_recent "
                    "(session_id, user_text, robot_text) VALUES (:sid, :u, :r)"
                ),
                {"sid": sid, "u": ut, "r": rt or None},
            )

        self._rollup_summaries(sid, start_tier=0)

    def _rollup_summaries(self, session_id: str, start_tier: int) -> None:
        """某 tier 摘要超過 SUMMARY_ROLLUP_CAP 時，壓成下一 tier。"""
        tier = start_tier
        while tier < 32:
            with self.engine.begin() as conn:
                cnt = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM clarification_dialogue_summary "
                        "WHERE session_id = :sid AND tier = :t"
                    ),
                    {"sid": session_id, "t": tier},
                ).scalar()
                cnt = int(cnt or 0)
                if cnt <= SUMMARY_ROLLUP_CAP:
                    break

                rows = conn.execute(
                    text(
                        "SELECT id, summary_text FROM clarification_dialogue_summary "
                        "WHERE session_id = :sid AND tier = :t ORDER BY id ASC LIMIT :cap"
                    ),
                    {"sid": session_id, "t": tier, "cap": SUMMARY_ROLLUP_CAP},
                ).fetchall()
                if len(rows) < SUMMARY_ROLLUP_CAP:
                    break

                ids = [int(r[0]) for r in rows]
                lines = [f"[摘要{tier}] {r[1]}" for r in rows]
                mega = self._compress(lines)
                conn.execute(
                    text(
                        "INSERT INTO clarification_dialogue_summary "
                        "(session_id, tier, summary_text) VALUES (:sid, :nt, :txt)"
                    ),
                    {"sid": session_id, "nt": tier + 1, "txt": mega},
                )
                ph = ",".join(str(int(i)) for i in ids)
                conn.execute(
                    text(
                        f"DELETE FROM clarification_dialogue_summary "
                        f"WHERE session_id = :sid AND id IN ({ph})"
                    ),
                    {"sid": session_id},
                )
            tier += 1

    def get_context_block(self, session_id: Optional[str], max_chars: int = 2000) -> str:
        """組合成給 LLM 的純文字區塊（高 tier → 低 tier → 最近 raw）。"""
        sid = (session_id or "default").strip() or "default"
        parts: List[str] = []

        with self.engine.connect() as conn:
            tiers = conn.execute(
                text(
                    "SELECT DISTINCT tier FROM clarification_dialogue_summary "
                    "WHERE session_id = :sid ORDER BY tier DESC"
                ),
                {"sid": sid},
            ).fetchall()
            for (t,) in tiers:
                t = int(t)
                sums = conn.execute(
                    text(
                        "SELECT summary_text FROM clarification_dialogue_summary "
                        "WHERE session_id = :sid AND tier = :t ORDER BY id ASC"
                    ),
                    {"sid": sid, "t": t},
                ).fetchall()
                for (st,) in sums:
                    if st:
                        parts.append(f"[澄清壓縮 層級{t}] {st}")

            rrows = conn.execute(
                text(
                    "SELECT user_text, robot_text FROM clarification_dialogue_recent "
                    "WHERE session_id = :sid ORDER BY id DESC LIMIT :lim"
                ),
                {"sid": sid, "lim": RECENT_CAP},
            ).fetchall()
            for u, r in reversed(rrows):
                parts.append(f"使用者：{u}")
                if r:
                    parts.append(f"機器人：{r}")

        if not parts:
            return ""
        block = "\n".join(parts)
        if len(block) > max_chars:
            block = block[-max_chars:]
        return "\n=== 過去澄清／語意不明對話（壓縮＋最近）===\n" + block + "\n"
