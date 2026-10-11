"""Interaction log: bounded record of recent user utterances and the robot's response type (EXECUTE / ASK / ANSWER). One of the three memory types (thesis Ch. 4)."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import logging
logger = logging.getLogger(__name__)
SPATIAL_CALC_AVAILABLE = True


@dataclass
class InteractionRecord:
    """
    互動記錄 - 一次 Execute / Ask / Answer 決策的結構化快照

    這是讓 collaboration policy 真正成為 multi-turn interaction-aware
    的核心資料結構：記錄每次互動的輸入、決策類型、與機器人回應，
    使後續輪次能在 context 中看到「我之前問了什麼、回答了什麼」。
    """
    timestamp: float                  # time.time()
    user_utterance: str               # 人類的原始輸入
    response_type: str                # 'execute' | 'ask' | 'answer'
    robot_response: str               # TTS 播報的文字或執行的動作描述
    step_context: str                 # 當時的 TSM step_class
    task_id: Optional[str] = None     # 關聯的 task_id（如有）
    success: Optional[bool] = None    # 執行結果（execute branch 用）

    def to_text(self) -> str:
        """給 LLM prompt 用的單行摘要。"""
        import time as _time
        from datetime import datetime as _dt
        ts = _dt.fromtimestamp(self.timestamp).strftime('%H:%M:%S')
        return (
            f"[{ts}] [{self.response_type.upper()}] "
            f"Human: \"{self.user_utterance}\" → "
            f"Robot: \"{self.robot_response}\" "
            f"(step: {self.step_context})"
        )

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'user_utterance': self.user_utterance,
            'response_type': self.response_type,
            'robot_response': self.robot_response,
            'step_context': self.step_context,
            'task_id': self.task_id,
            'success': self.success,
        }


class InteractionLog:
    """
    輕量互動日誌 - 保存最近 N 筆 InteractionRecord 的 FIFO 緩衝。

    與 TemporalMemory（感知導向）互補：
    - TemporalMemory  記錄「機器人看到了什麼」
    - InteractionLog  記錄「人機之間說了什麼、決策了什麼」
    """

    def __init__(self, maxlen: int = 20):
        self._buffer: deque = deque(maxlen=maxlen)

    def record(self, rec: InteractionRecord) -> None:
        self._buffer.append(rec)

    def get_recent(self, n: int = 5) -> List[InteractionRecord]:
        return list(self._buffer)[-n:]

    def to_context_text(self, n: int = 5) -> str:
        """返回可注入 LLM prompt 的文字片段。"""
        recent = self.get_recent(n)
        if not recent:
            return ""
        lines = ["=== Recent Interaction History ==="]
        for rec in recent:
            lines.append(rec.to_text())
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._buffer)
