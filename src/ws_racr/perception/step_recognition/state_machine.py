#!/usr/bin/env python3
"""
Step Recognition State Machine
步驟識別狀態機 - 防止錯誤回溯，使用 FSM + top-k rescue
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import deque
import time
import numpy as np

logger = logging.getLogger(__name__)


class StepStateMachine:
    """
    步驟狀態機 - 使用 FSM + top-k rescue 防止步驟跳躍錯誤
    
    核心功能：
    1. confirmed_step: 只有連續穩定才更新
    2. step_graph: 硬編碼的步驟轉移規則
    3. top-k rescue: 當 top-1 不合法時，嘗試 top-2, top-3
    4. reset event: 允許在特定條件下回溯
    """
    
    def __init__(
        self,
        num_steps: int = 9,
        confirm_frames: int = 5,  # 需要連續 N 幀才確認
        allow_retry: bool = False,  # 是否允許重試（step ±1）
        reset_on_idle: bool = True  # idle 超過閾值後自動 reset
    ):
        """
        Args:
            num_steps: 步驟總數（不含 idle）
            confirm_frames: 需要連續穩定的幀數
            allow_retry: 是否允許回退一步（重試）
            reset_on_idle: idle 狀態持續過久是否自動 reset
        """
        self.num_steps = num_steps
        self.confirm_frames = confirm_frames
        self.allow_retry = allow_retry
        self.reset_on_idle = reset_on_idle
        
        # 狀態變量
        self.confirmed_step = 0  # 已確認的步驟（0=未開始/idle）
        self.step_belief_history = deque(maxlen=confirm_frames)  # 最近 N 幀的 belief
        self.reset_requested = False
        
        # Step Graph: 定義允許的步驟轉移
        # 格式：allowed_next[current_step] = {allowed_next_steps}
        self.allowed_next = self._build_step_graph()
        
        # Idle 追蹤
        self.idle_counter = 0
        self.idle_reset_threshold = 30  # idle 持續 30 幀後允許 reset
        
        # 統計
        self.transition_history = []  # [(timestamp, from_step, to_step, reason), ...]
        
        logger.info("Step State Machine Initialized")
        logger.info(f"   - Confirm frames: {confirm_frames}")
        logger.info(f"   - Allow retry: {allow_retry}")
        logger.info(f"   - Step graph: {self.allowed_next}")
    
    def _build_step_graph(self) -> Dict[int, Set[int]]:
        """
        構建步驟轉移圖
        
        規則：
        - 每個步驟只能前進到下一步或保持不變
        - 如果 allow_retry=True，可以回退一步
        - idle (0) 可以轉移到任何步驟（初始化）
        - 🆕 一旦進入步驟圈（1-9），不能回到 idle，除非完成最後一步
        """
        graph = {}
        
        # idle 可以去任何步驟（初始化）
        graph[0] = set(range(self.num_steps + 1))
        
        # 正常步驟
        for step in range(1, self.num_steps + 1):
            allowed = {step, step + 1} if step < self.num_steps else {step}
            
            if self.allow_retry and step > 1:
                allowed.add(step - 1)  # 允許回退一步
            
            # 只有最後一步才能回到 idle
            if step == self.num_steps:
                allowed.add(0)  # 最後一步完成後可以回到 idle
            
            graph[step] = allowed
        
        return graph
    
    def update(
        self,
        vision_prediction: Dict,
        world_state: Optional[Dict] = None
    ) -> Dict:
        """
        更新狀態機
        
        Args:
            vision_prediction: TSM 預測結果（包含 top-k）
                {
                    'class_index': int,
                    'confidence': float,
                    'top_k_predictions': [(step_id, name, prob), ...]
                }
            world_state: 當前世界狀態（可選，用於 RAG 輔助）
        
        Returns:
            決策結果:
                {
                    'confirmed_step': int,  # 確認的步驟
                    'final_step': int,  # 最終選擇的步驟（可能與 vision top-1 不同）
                    'confidence': float,
                    'reasoning': str,  # 決策理由
                    'can_update': bool,  # 是否可以更新 confirmed
                    'vision_top1': int,  # vision 的 top-1
                    'alternatives': List[Tuple]  # top-k 候選
                }
        """
        # 1. 檢查 reset 請求
        if self.reset_requested:
            self._perform_reset()
            self.reset_requested = False
            return self._create_result(0, 0, 1.0, "System reset")
        
        # 2. 提取 vision top-k
        vision_top1 = vision_prediction['class_index']
        top_k = vision_prediction.get('top_k_predictions', [])
        
        # 3. 嘗試找到合法的步驟（從 top-k 中選擇）
        final_step, confidence, reason = self._select_valid_step(
            vision_top1,
            top_k,
            world_state
        )
        
        # 4. 更新 belief history
        self.step_belief_history.append(final_step)
        
        # 5. 檢查是否可以更新 confirmed_step
        can_update = self._check_can_update(final_step)
        
        if can_update:
            old_step = self.confirmed_step
            self.confirmed_step = final_step
            self.transition_history.append((
                time.time(),
                old_step,
                final_step,
                reason
            ))
            self.idle_counter = 0
        
        # 6. Idle 追蹤
        if final_step == 0:
            self.idle_counter += 1
            if self.reset_on_idle and self.idle_counter >= self.idle_reset_threshold:
                self._perform_reset()
        else:
            self.idle_counter = 0
        
        return self._create_result(
            self.confirmed_step,
            final_step,
            confidence,
            reason,
            vision_top1,
            top_k
        )
    
    def _select_valid_step(
        self,
        vision_top1: int,
        top_k: List[Tuple[int, str, float]],
        world_state: Optional[Dict]
    ) -> Tuple[int, float, str]:
        """
        從 top-k 中選擇合法的步驟
        
        優先級：
        1. 如果 top-1 合法 → 使用 top-1
        2. 如果 top-1 不合法，檢查 top-2, top-3
        3. 如果都不合法 → fallback 到 confirmed_step
        
        Returns:
            (selected_step, confidence, reason)
        """
        allowed = self.allowed_next.get(self.confirmed_step, set())
        
        # 策略 1: top-1 合法
        if vision_top1 in allowed:
            top1_conf = top_k[0][2] if top_k else 0.0
            return vision_top1, top1_conf, "vision_top1"
        
        # 策略 2: 嘗試 top-k
        for step_id, step_name, prob in top_k[1:]:  # 跳過 top-1（已檢查）
            if step_id in allowed:
                return step_id, prob, f"vision_top{top_k.index((step_id, step_name, prob)) + 1}"
        
        # 策略 3: 簡單物體狀態檢查（預留，可選）
        # 例如：如果平台不在桌上，排除某些步驟
        # if world_state is not None:
        #     if 'platform' not in world_state.get('detected_objects', []):
        #         # 排除需要平台的步驟
        #         pass
        
        # 策略 4: Fallback 保持 confirmed
        return self.confirmed_step, 0.0, "fallback_confirmed"
    
    def _check_can_update(self, final_step: int) -> bool:
        """
        檢查是否可以更新 confirmed_step
        
        條件：
        - belief history 中最近 N 幀都是同一個步驟
        - 且該步驟 != confirmed_step（有變化）
        """
        if len(self.step_belief_history) < self.confirm_frames:
            return False
        
        # 檢查最近 N 幀是否一致
        recent = list(self.step_belief_history)
        if len(set(recent)) == 1 and recent[0] == final_step:
            # 一致且與 confirmed 不同
            return final_step != self.confirmed_step
        
        return False
    
    def request_reset(self, reason: str = "manual"):
        """請求重置狀態機"""
        self.reset_requested = True
        logger.info(f"Reset requested: {reason}")
    
    def _perform_reset(self):
        """執行重置"""
        self.confirmed_step = 0
        self.step_belief_history.clear()
        self.idle_counter = 0
        logger.info("State machine reset to step 0")
    
    def _create_result(
        self,
        confirmed: int,
        final: int,
        conf: float,
        reason: str,
        vision_top1: Optional[int] = None,
        alternatives: Optional[List] = None
    ) -> Dict:
        """創建結果字典"""
        return {
            'confirmed_step': confirmed,
            'final_step': final,
            'confidence': conf,
            'reasoning': reason,
            'can_update': (confirmed == final and len(self.step_belief_history) >= self.confirm_frames),
            'vision_top1': vision_top1,
            'alternatives': alternatives or [],
            'belief_history': list(self.step_belief_history)
        }
    
    def get_state(self) -> Dict:
        """獲取當前狀態"""
        return {
            'confirmed_step': self.confirmed_step,
            'belief_history': list(self.step_belief_history),
            'idle_counter': self.idle_counter,
            'transition_count': len(self.transition_history)
        }
    
    def get_transition_history(self, last_n: int = 10) -> List[Tuple]:
        """獲取轉移歷史"""
        return self.transition_history[-last_n:]
