"""
Multimodal deictic grounding — explicit finite-state machine (FSM).

================================================================================
A. 兩層狀態（為什麼要分兩層）
================================================================================

(1) **Session state（跨輪、存在記憶體裡）**
    只描述「上一句含糊指令是否還在等人補資訊」。

    - IDLE
        沒有未完成的指代補全；或上一輪已解消 / 已放棄 pending。

    - AWAIT_FOLLOWUP
        上一輪 deictic 在「澄清」後仍沒有目標，已寫入 pending_base_command；
        等使用者**下一次按 V 說話**（或之後某次 utterance 被判定為補充語）。

(2) **Ephemeral phase（單次 resolve() 內、不另存狀態變數）**
    同一輪語音進來後，在函式內依序走的小步驟，語意上等價於子狀態機：

    E0  ENTRY          — 本輪 utterance 進入
    E1  PTS            — point-then-speak（說話當下已有 pointing）
    E2  PASS           — 非指代或已有明確物件名 → 直通
    E3  STP_WAIT       — speak-then-point（短時間輪詢 pointing）
    E4  STC_ASK        — speak-then-clarify（TTS 問一次；不開麥）
    E5  LATE_POINT     — 問完後再讀一次 pointing

================================================================================
B. 跨輪轉移表（Session: IDLE / AWAIT_FOLLOWUP）
================================================================================

  事件: 新的一條語音文字 command 進入 run() 最前面（在 grounding 之前）

  [AWAIT_FOLLOWUP] + utterance U
      若 U 被判為「補充描述」（不是新頂層指令、不是問句）:
          → IDLE, 輸出 merged = pending_base + 「（指的是U）」, 清除 pending
      若 U 被判為「新指令 / 問句」:
          → IDLE, 丟棄 pending, 輸出原 U（當成全新一輪）

  [IDLE] + utterance U
      不經 consume，維持 IDLE，進入 resolve(U)

  resolve 結尾若仍無法注入「（指的是…）」且 U 仍為含糊 deictic:
      → AWAIT_FOLLOWUP, pending_base = U

  resolve 結尾若已成功注入「（指的是…）」或 U 不需 grounding:
      → IDLE, 清除 pending（若有）

================================================================================
C. 單輪內轉移表（Ephemeral，依序執行）
================================================================================

  E0  ENTRY
      若 command 已含 「（指的是」 / 「(指的是」 → **RESOLVED**，結束（不再走後續）

  E1  PTS — point-then-speak
      條件: 當下 get_live_pointing_target 非空 AND 句中無 ontology 明確物件
      → 輸出 command + 「（指的是{pt}）」, **RESOLVED**, session → IDLE

  E2  PASS
      條件: 句中已有明確物件 OR 非 deictic 句
      → 原樣輸出, **RESOLVED**, session → IDLE

  E3  STP_WAIT — speak-then-point
      條件: deictic 且無明確物件
      在 wait_point_sec 內每 poll_sec 輪詢 pointing
      若期間出現 pt → 輸出 +「（指的是pt）」, **RESOLVED**, session → IDLE
      若逾時 → 進 E4

  E4  STC_ASK — speak-then-clarify
      設 asked_object_clarification_this_run = True
      呼叫 ask_fn（僅 TTS；不回傳語音時固定 None）
      若同一呼叫內 clar 非空（例如未來接上 push-to-talk callback）:
          ontology 命中 → 「（指的是obj）」
          否則自由文字 → 「（指的是clar）」
          → **RESOLVED**, session → IDLE

    E5  LATE_POINT_WAIT
      問完後再開一段短窗口持續輪詢 pointing；
      若有 → **RESOLVED**, session → IDLE
      否則 → **UNRESOLVED**, pending_base = command, session → AWAIT_FOLLOWUP

================================================================================
D. 與 Planner / RAG 的邊界（合約）
================================================================================

  - 本模組**只負責**把自然語言補成「帶（指的是…）」或維持含糊並設 pending。
  - 若輸出仍為 UNRESOLVED deictic，Orchestrator 應**不進** RAG、不執行 skill
    （由外層 guard 實作）。
  - 一旦含「（指的是…）」，視為已 grounding，外層可選擇跳過 Plan RAG 以免覆寫目標。
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeicticSessionState(Enum):
    """跨輪狀態：是否在等人補完指代。"""

    IDLE = auto()
    AWAIT_FOLLOWUP = auto()


class DeicticGroundingModule:
    """
    FSM-backed deictic grounding. Public API 與重構前相同，方便 Orchestrator 沿用。
    """

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
    # Chinese aliases for ontology object names. Keep this small and explicit:
    # deictic grounding only needs high-frequency disambiguation terms.
    _OBJECT_ALIASES: Dict[str, tuple] = {
        "screw_big": ("大螺絲", "大顆螺絲", "大螺栓"),
        "screw_small": ("小螺絲", "小顆螺絲", "小螺栓"),
        "hex_wrench": ("內六角", "六角板手", "六角扳手", "板手", "扳手"),
        "cylinder": ("圓柱", "圓柱體"),
        "platform": ("平台", "底座"),
        "bracket": ("支架",),
        "nut": ("螺帽", "螺母"),
    }

    def __init__(
        self,
        known_objects: List[str],
        wait_point_sec: float = 4.0,
        poll_sec: float = 0.2,
    ) -> None:
        self.known_objects = [o.strip() for o in known_objects if o and o.strip()]
        self.wait_point_sec = wait_point_sec
        self.poll_sec = poll_sec
        # Extra window after STC ask so "先說再指" can still resolve
        # even if the pointing appears slightly after TTS prompt.
        self.post_ask_wait_sec = max(wait_point_sec, 8.0)
        self.pending_base_command: Optional[str] = None
        self.session_state: DeicticSessionState = DeicticSessionState.IDLE
        self.asked_object_clarification_this_run: bool = False
        # 供 ClarificationDialogueStore：跨輪「跟進語句」與前一輪 pending 合併時記錄
        self._last_merge_record: Optional[dict] = None

    def reset_turn(self) -> None:
        """每輪 utterance 開頭：只清「本輪是否已問過澄清」旗標。"""
        self.asked_object_clarification_this_run = False

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def contains_explicit_object(self, text: str) -> bool:
        t = text or ""
        if any(obj in t for obj in self.known_objects):
            return True
        for aliases in self._OBJECT_ALIASES.values():
            if any(alias in t for alias in aliases):
                return True
        return False

    def extract_object_from_text(self, text: str) -> Optional[str]:
        t = text or ""
        for obj in self.known_objects:
            if obj in t:
                return obj
        # Prefer longest alias first to avoid short-token shadowing.
        alias_pairs: List[tuple] = []
        for canonical, aliases in self._OBJECT_ALIASES.items():
            if canonical in self.known_objects:
                for alias in aliases:
                    alias_pairs.append((alias, canonical))
        alias_pairs.sort(key=lambda p: len(p[0]), reverse=True)
        for alias, canonical in alias_pairs:
            if alias in t:
                return canonical
        return None

    def has_injected_deictic_target(self, text: str) -> bool:
        t = (text or "").strip()
        return ("（指的是" in t) or ("(指的是" in t)

    def is_deictic_command(self, text: str) -> bool:
        t = (text or "").strip()
        if any(p in t for p in self._DEICTIC_PRONOUNS):
            return True
        # 「夾爪」「夾頭」內含子字串「夾」會誤觸 _DEICTIC_ACTION_HINTS 裡的「夾」
        # （例如「關閉夾爪」被當成需指物的 deictic → STP_WAIT / STC_ASK）。
        # 先把這兩個詞遮掉再做 action-hint 比對。
        if not self.contains_explicit_object(t):
            masked = t.replace("夾爪", "■爪").replace("夾頭", "■頭")
            if any(v in masked for v in self._DEICTIC_ACTION_HINTS):
                return True
        return False

    def is_unresolved_deictic_command(self, text: str) -> bool:
        t = (text or "").strip()
        if not self.is_deictic_command(t):
            return False
        if self.contains_explicit_object(t):
            return False
        if self.has_injected_deictic_target(t):
            return False
        return True

    # ------------------------------------------------------------------
    # Cross-turn: AWAIT_FOLLOWUP handling（對應轉移表 B）
    # ------------------------------------------------------------------

    def consume_pending_if_clarification(
        self,
        command: str,
        is_question_fn: Callable[[str], bool],
    ) -> str:
        if self.session_state != DeicticSessionState.AWAIT_FOLLOWUP or not self.pending_base_command:
            return command

        t = (command or "").strip()
        looks_new = is_question_fn(t) or any(h in t for h in self._NEW_COMMAND_HINTS)
        if not looks_new:
            merged = f"{self.pending_base_command}（指的是{command}）"
            self._last_merge_record = {
                "base": self.pending_base_command,
                "follow_up": t,
                "merged": merged,
            }
            logger.info(f"   [Deictic FSM] AWAIT_FOLLOWUP --(follow-up)--> IDLE | merged")
            self.pending_base_command = None
            self.session_state = DeicticSessionState.IDLE
            return merged

        logger.info(f"   [Deictic FSM] AWAIT_FOLLOWUP --(new top-level)--> IDLE | drop pending")
        self.pending_base_command = None
        self.session_state = DeicticSessionState.IDLE
        return command

    def take_last_merge_record(self) -> Optional[dict]:
        """讀取並清除上一輪 consume_pending 若發生的「合併澄清」紀錄。"""
        r = self._last_merge_record
        self._last_merge_record = None
        return r

    # ------------------------------------------------------------------
    # Single-run resolve（對應轉移表 C: E0→E5）
    # ------------------------------------------------------------------

    def resolve(
        self,
        command: str,
        world_state,
        get_live_pointing_target_fn: Callable[[object], Optional[str]],
        ask_fn: Callable[[str], Optional[str]],
    ) -> str:
        # E0: already grounded by text
        if self.has_injected_deictic_target(command):
            self._goto_idle_cleared()
            logger.info("   [Deictic FSM] E0 --(already injected)--> RESOLVED")
            return command

        # E1: point-then-speak
        pt = get_live_pointing_target_fn(world_state)
        if pt and not self.contains_explicit_object(command):
            out = f"{command}（指的是{pt}）"
            logger.info(f"   [Deictic FSM] E1 PTS --(point+speak)--> RESOLVED | {out}")
            self._goto_idle_cleared()
            return out

        # E2: explicit object or non-deictic
        if self.contains_explicit_object(command) or not self.is_deictic_command(command):
            logger.info("   [Deictic FSM] E2 PASS --(no deictic resolution needed)--> RESOLVED")
            self._goto_idle_cleared()
            return command

        # E3: speak-then-point
        logger.info("   [Deictic FSM] E3 STP_WAIT --enter--")
        deadline = time.time() + self.wait_point_sec
        while time.time() < deadline:
            pt = get_live_pointing_target_fn(world_state)
            if pt:
                out = f"{command}（指的是{pt}）"
                logger.info(f"   [Deictic FSM] E3 --(point during window)--> RESOLVED | {out}")
                self._goto_idle_cleared()
                return out
            time.sleep(self.poll_sec)

        # E4: speak-then-clarify (TTS only; ask_fn returns None in current UX)
        logger.info("   [Deictic FSM] E4 STC_ASK --enter--")
        self.asked_object_clarification_this_run = True
        clar = ask_fn("我還不知道目標物件。你可以指向目標，或直接說出物件名稱。")
        if clar:
            obj = self.extract_object_from_text(clar)
            if obj:
                out = f"{command}（指的是{obj}）"
                logger.info(f"   [Deictic FSM] E4 --(clarification utterance)--> RESOLVED | {out}")
                self._goto_idle_cleared()
                return out
            clar_text = clar.strip()
            if clar_text:
                out = f"{command}（指的是{clar_text}）"
                logger.info(f"   [Deictic FSM] E4 --(free-form clarification)--> RESOLVED | {out}")
                self._goto_idle_cleared()
                return out

        # E5: late point wait window (important for speak-then-point)
        logger.info("   [Deictic FSM] E5 LATE_POINT_WAIT --enter--")
        deadline = time.time() + self.post_ask_wait_sec
        while time.time() < deadline:
            pt = get_live_pointing_target_fn(world_state)
            if pt:
                out = f"{command}（指的是{pt}）"
                logger.info(f"   [Deictic FSM] E5 --(late point in window)--> RESOLVED | {out}")
                self._goto_idle_cleared()
                return out
            time.sleep(self.poll_sec)

        # UNRESOLVED → AWAIT_FOLLOWUP
        logger.info("   [Deictic FSM] E5 --(unresolved)--> AWAIT_FOLLOWUP | pending set")
        self.pending_base_command = command
        self.session_state = DeicticSessionState.AWAIT_FOLLOWUP
        return command

    def _goto_idle_cleared(self) -> None:
        self.pending_base_command = None
        self.session_state = DeicticSessionState.IDLE
