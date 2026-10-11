#!/usr/bin/env python3
"""
Failure Recovery Module
失敗恢復模組 - 人類在環

核心理念：
1. 失敗 → 通知人類「沒夾到」
2. 人類語音指導 → "往下5mm" / "往左3mm"
3. 執行修正 → 應用調整並重試
4. 記錄結果 → 更新重試計數

Recovery FSM: 
Detect → Notify Human → Wait Voice Guidance → Apply Adjustment → Execute → 
Record Result → (Success → Reset) / (Fail → Loop)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from ws_racr.execution.safety.failure_detector import FailureType
from ws_racr.execution.safety.voice_guided_correction import get_voice_guided_correction

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """恢復策略"""
    HUMAN_GUIDED_ADJUSTMENT = "human_guided"      # 人類語音指導修正（新增）
    RE_PERCEIVE_AND_GRASP = "re_perceive_grasp"  # 重新感知+重抓
    SCAN_AREA = "scan_area"                       # 掃描區域
    UPDATE_TARGET = "update_target"               # 更新目標
    RESET_GRIPPER = "reset_gripper"               # 重置夾爪
    ASK_HUMAN = "ask_human"                       # 請求人類協助
    ABORT = "abort"                               # 放棄


class RecoveryState(Enum):
    """恢復狀態"""
    IDLE = "idle"
    DETECTING = "detecting"
    NOTIFYING_HUMAN = "notifying_human"           # 通知人類（新增）
    WAITING_GUIDANCE = "waiting_guidance"         # 等待語音指導（新增）
    APPLYING_ADJUSTMENT = "applying_adjustment"   # 應用調整（新增）
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECORDING_RESULT = "recording_result"         # 記錄結果（新增）
    SUCCESS = "success"
    FAILED = "failed"


class RecoveryPolicy:
    """
    失敗恢復策略 - 人類在環
    
    流程:
    1. 檢測失敗 → 通知人類「沒夾到 cylinder_1」
    2. 請求語音指導 HUMAN_GUIDED_ADJUSTMENT
    3. 等待語音 → "往下5mm" or "往左3mm" or "再試一次"
    4. 執行修正 → 應用調整參數
    5. 驗證結果 → 成功/失敗
    """
    
    def __init__(self, detection_mode=None):
        """
        初始化恢復策略
        
        Args:
            detection_mode: 失敗檢測模式（預設 HYBRID）
                - AUTO_VISION: 自動視覺檢測
                - HUMAN_NOTIFY: 僅人類通知
                - HYBRID: 人類優先 + 視覺備援（推薦）
        """
        from ws_racr.execution.safety.failure_detector import FailureDetectionMode, get_failure_detector
        
        self.state = RecoveryState.IDLE
        self.max_retry = 5  # 增加重試次數（人類可以多次指導）
        self.retry_count = {}  # {object_id: count}
        
        # 設定檢測模式（預設 HYBRID）
        self.detection_mode = detection_mode or FailureDetectionMode.HYBRID
        self.failure_detector = get_failure_detector(detection_mode=self.detection_mode)
        
        # 整合語音解析
        self.voice_parser = get_voice_guided_correction()
        
        # 當前恢復上下文
        self.current_recovery = {
            'object_id': None,
            'failure_type': None,
            'adjustment': None,
            'waiting_for_voice': False
        }
    
    def get_recovery_action(self, 
                           failure: Dict,
                           world_state,
                           robot_status) -> Optional[Dict]:
        """
        根據失敗類型獲取恢復動作（人類在環版本）
        
        流程：
        1. 請求人類語音指導
        2. 應用調整並重試
        
        Args:
            failure: 失敗檢測結果
            world_state: 當前 WorldState
            robot_status: 當前 RobotStatus
            
        Returns:
            Dict: {
                'strategy': RecoveryStrategy,
                'action': str,
                'parameters': Dict,
                'explanation': str,
                'wait_for_voice': bool  # 是否需要等待語音指導
            } or None
        """
        failure_type = failure['failure_type']
        object_id = failure['object_id']
        
        # 更新當前恢復上下文
        self.current_recovery['object_id'] = object_id
        self.current_recovery['failure_type'] = failure_type
        
        # === Step 1: 檢查重試次數 ===
        retry_count = self.retry_count.get(object_id, 0)
        if retry_count >= self.max_retry:
            logger.error(f"{object_id} 已重試 {retry_count} 次，超過上限")
            return self._plan_ask_human(failure, reason=f"多次失敗（{retry_count}次）")
        
        # === Step 2: 針對抓取失敗，使用人類在環修正 ===
        if failure_type == FailureType.GRASP_MISS:
            return self._plan_human_guided_grasp(failure, world_state)
        
        # === Step 3: 其他失敗類型的處理 ===
        elif failure_type == FailureType.POSITION_DRIFT:
            return self._plan_update_target(failure, world_state)
        
        elif failure_type == FailureType.GRIPPER_OPEN_FAIL:
            return self._plan_reset_gripper(failure, robot_status)
        
        else:
            return self._plan_ask_human(failure)
    
    def _plan_human_guided_grasp(self, failure: Dict, world_state) -> Dict:
        """
        規劃人類指導的抓取修正
        
        流程：請求語音指導 (HUMAN_GUIDED)
        """
        object_id = failure['object_id']
        obj_state = world_state.objects.get(object_id)
        
        if not obj_state or not obj_state.position:
            return self._plan_ask_human(failure, reason="找不到物體位置")
        
        # 請求人類語音指導
        logger.info(f"請求人類語音指導: {object_id}")
        self.current_recovery['waiting_for_voice'] = True
        
        return {
            'strategy': RecoveryStrategy.HUMAN_GUIDED_ADJUSTMENT,
            'action': 'wait_for_voice_guidance',
            'parameters': {
                'target_object': object_id,
                'target_position': obj_state.position,
                'prompt': f'{obj_state.object_type} 夾取策略不確定，請提供方向性提示（例如：位置偏了、方向不對）',
                'examples': ['位置偏了', '方向不太對', '夾取狀態怪怪的']
            },
            'explanation': f'請求驗證 {obj_state.object_type} 的夾取狀態',
            'wait_for_voice': True
        }
    
    def apply_voice_guidance(self, voice_command: str) -> Optional[Dict]:
        """
        應用人類的語音指導
        
        Args:
            voice_command: 人類語音指令（如 "往下5mm" 或 "沒夾到"）
            
        Returns:
            Dict: 恢復動作 or None（如果是失敗通知）
        """
        # === 1. 檢查是否為失敗通知 ===
        if self.voice_parser.is_failure_notification(voice_command):
            object_id = self.voice_parser.extract_object_id_from_failure(voice_command)
            if object_id:
                logger.info(f"收到人類失敗通知: {object_id}")
                self.failure_detector.notify_failure_by_human(object_id, voice_command)
                return None
            else:
                logger.warning("失敗通知但未指定物體 ID")
                return None
        
        # === 2. 解析修正指令 ===
        if not self.current_recovery['waiting_for_voice']:
            logger.warning("當前不在等待語音指導狀態")
            return None
        
        object_id = self.current_recovery['object_id']
        if not object_id:
            logger.warning("沒有當前恢復上下文")
            return None
        
        # 解析語音指令
        adjustment = self.voice_parser.parse(voice_command)
        
        if adjustment:
            # 記錄調整（暫時未知結果）
            self.current_recovery['adjustment'] = adjustment
            self.current_recovery['waiting_for_voice'] = False
            
            logger.info(f"接收到人類驗證反饋，系統生成新的抓取假設")
            
            return {
                'strategy': RecoveryStrategy.HUMAN_GUIDED_ADJUSTMENT,
                'action': 'pickup_with_adjustment',
                'parameters': {
                    'target_object': object_id,
                    'adjustment': {
                        'dx': adjustment.dx,
                        'dy': adjustment.dy,
                        'dz': adjustment.dz,
                        'angle_offset': adjustment.angle_offset,
                        'approach_offset': adjustment.approach_offset
                    },
                    'source': 'human_verification_hint'
                },
                'explanation': f'已調整抓取策略，重新執行'
            }
        else:
            logger.warning("無法理解反饋，請用抽象描述（例如：位置偏了、方向不對）")
            return None
    
    def _plan_update_target(self, failure: Dict, world_state) -> Dict:
        """
        規劃更新目標位置（用於位置偏差）
        """
        object_id = failure['object_id']
        evidence = failure.get('evidence', {})
        new_position = evidence.get('actual_position')
        
        if not new_position:
            return self._plan_ask_human(failure, reason="無法獲取新位置")
        
        return {
            'strategy': RecoveryStrategy.UPDATE_TARGET,
            'action': 'update_and_retry',
            'parameters': {
                'target_object': object_id,
                'updated_position': new_position,
                'force_update': True
            },
            'explanation': f'更新{object_id}的目標位置並重試',
            'wait_for_voice': False
        }
    
    def _plan_reset_gripper(self, failure: Dict, robot_status) -> Dict:
        """
        規劃重置夾爪
        """
        return {
            'strategy': RecoveryStrategy.RESET_GRIPPER,
            'action': 'reset_gripper',
            'parameters': {
                'open_gripper': True,
                'close_gripper': False,
                'wait_time': 1.0
            },
            'explanation': '重置夾爪狀態',
            'wait_for_voice': False
        }
    
    def _plan_ask_human(self, failure: Dict, reason: str = None) -> Dict:
        """
        請求人類協助
        """
        object_id = failure['object_id']
        failure_type = failure['failure_type']
        
        explanation = reason or f'{failure_type.value}: 需要您的協助處理 {object_id}'
        
        return {
            'strategy': RecoveryStrategy.ASK_HUMAN,
            'action': 'request_help',
            'parameters': {
                'object_id': object_id,
                'failure_type': failure_type.value,
                'retry_count': self.retry_count.get(object_id, 0),
                'reason': explanation
            },
            'explanation': explanation,
            'wait_for_voice': False
        }
    
    def record_attempt_result(self, 
                              object_id: str,
                              success: bool):
        """
        記錄嘗試結果並更新重試計數
        
        Args:
            object_id: 物體ID
            success: 是否成功
        """
        if success:
            # === 成功 → 重置計數 ===
            if object_id in self.retry_count:
                del self.retry_count[object_id]
            
            logger.info(f"{object_id} 成功")
            
            # 清空當前恢復上下文
            self.current_recovery = {
                'object_id': None,
                'failure_type': None,
                'adjustment': None,
                'waiting_for_voice': False
            }
        else:
            # === 失敗 → 增加計數 ===
            self.retry_count[object_id] = self.retry_count.get(object_id, 0) + 1
            logger.error(f"{object_id} 失敗（第{self.retry_count[object_id]}次）")
    
    def get_failure_stats(self) -> Dict:
        """
        獲取失敗統計
        """
        return {
            'retry_counts': self.retry_count.copy(),
            'total_objects_with_retries': len(self.retry_count)
        }
    
    def reset_recovery_context(self):
        """
        重置當前恢復上下文（用於取消語音等待）
        """
        self.current_recovery = {
            'object_id': None,
            'failure_type': None,
            'adjustment': None,
            'waiting_for_voice': False
        }
        logger.info("恢復上下文已重置")
    
    def notify_failure_by_human(self, object_id: str, description: str = ""):
        """
        人類主動通知失敗
        
        Args:
            object_id: 失敗的物體 ID
            description: 失敗描述（如 "沒夾到"）
        """
        self.failure_detector.notify_failure_by_human(object_id, description)
    
    def set_detection_mode(self, mode):
        """
        切換失敗檢測模式
        
        Args:
            mode: FailureDetectionMode 枚舉值
        """
        from ws_racr.execution.safety.failure_detector import FailureDetectionMode
        self.detection_mode = mode
        self.failure_detector.set_detection_mode(mode)
        logger.info(f"失敗檢測模式已切換為: {mode.value}")


def get_recovery_policy(detection_mode=None) -> RecoveryPolicy:
    """
    全局單例
    
    Args:
        detection_mode: 失敗檢測模式（僅首次調用有效）
    """
    if not hasattr(get_recovery_policy, '_instance'):
        get_recovery_policy._instance = RecoveryPolicy(detection_mode=detection_mode)
    return get_recovery_policy._instance
