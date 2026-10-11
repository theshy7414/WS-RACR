"""Temporal perception log: FIFO buffer of per-step scene summaries with optional LLM compression. One of the three memory types (thesis Ch. 4)."""

from typing import List
from collections import deque
import logging
logger = logging.getLogger(__name__)
SPATIAL_CALC_AVAILABLE = True
from ws_racr.memory.types import StepSummary


class TemporalMemory:
    """
    時序記憶 - FIFO 緩衝區
    
    根據 RoboMemory 設計:
    - 固定長度緩衝
    - 滿了之後用 LLM 壓縮總結
    - 保存近期互動流程
    """
    
    def __init__(self, max_length: int = 50, compression_threshold: int = 40):
        """
        Args:
            max_length: 最大長度
            compression_threshold: 達到這個長度時觸發壓縮（必須 < max_length）
        """
        self.buffer = deque(maxlen=max_length)
        self.max_length = max_length
        self.compression_threshold = compression_threshold
        self.compressed_summaries: List[str] = []
    
    def add(self, step_summary: StepSummary):
        """添加新的步驟摘要"""
        self.buffer.append(step_summary)
        
        # 檢查是否需要壓縮
        if len(self.buffer) >= self.compression_threshold:
            self._trigger_compression()
    
    def _trigger_compression(self):
        """觸發壓縮（需要 LLM）"""
        # 壓縮一半的緩衝區（例如 600 → 壓縮 300 筆）
        n_to_compress = min(self.compression_threshold // 2, len(self.buffer))
        
        items_to_compress = [self.buffer.popleft() for _ in range(n_to_compress)]
        
        # 使用 LLM 壓縮（如果提供）
        if hasattr(self, 'llm_compressor') and self.llm_compressor:
            try:
                # 構建 LLM prompt
                items_text = "\n".join([item.to_text() for item in items_to_compress])
                prompt = f"""Summarize the following {n_to_compress} sequential robot interaction steps into a concise summary (2-3 sentences):

{items_text}

Summary:"""
                
                # 調用 LLM
                summary_text = self.llm_compressor(prompt)

            except Exception as e:
                logger.warning(f"   LLM 壓縮失敗: {e}, 回退到文字拼接")
                summary_text = "\n".join([item.to_text() for item in items_to_compress])
        else:
            # 簡化版：直接文字連接
            summary_text = "\n".join([item.to_text() for item in items_to_compress])
        
        self.compressed_summaries.append(summary_text)
    
    def get_recent(self, n: int = 10) -> List[StepSummary]:
        """取得最近 N 筆"""
        return list(self.buffer)[-n:]
    
    def get_all(self) -> List[StepSummary]:
        """取得所有當前未壓縮的記錄"""
        return list(self.buffer)
    
    def to_text_context(self, n: int = 20) -> str:
        """
        轉換為文字上下文（用於 LLM prompt）
        
        Args:
            n: 最近 N 筆
        """
        recent_items = self.get_recent(n)
        
        text = "=== Recent Temporal Memory ===\n"
        
        # 添加壓縮的歷史（如果有）
        if self.compressed_summaries:
            text += "\n[Compressed History]\n"
            text += self.compressed_summaries[-1]  # 只取最近一次壓縮
            text += "\n\n"
        
        # 添加最近的詳細記錄
        text += "[Recent Events]\n"
        for item in recent_items:
            text += item.to_text() + "\n"
        
        return text
