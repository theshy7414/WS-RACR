#!/usr/bin/env python3
"""
Failure Detection Module
失敗檢測模組

檢測三類失敗：
1. 抓取失敗 (Grasp Failure) - 夾爪閉合但無負載
2. 視覺失敗 (Vision Failure) - 物體仍在桌上
3. 位置偏差 (Position Drift) - 執行後位置偏差過大

使用 Robot IO Truth + Vision Verification
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """失敗類型"""
    GRASP_MISS = "grasp_miss"           # 抓取未成功（夾爪閉合但沒物體）
    VISION_VERIFY_FAIL = "vision_fail"  # 視覺驗證失敗（物體仍在原位）
    POSITION_DRIFT = "position_drift"   # 位置偏差過大
    GRIPPER_OPEN_FAIL = "gripper_fail"  # 夾爪未正確打開/閉合
    TIMEOUT = "timeout"                 # 動作超時
    UNKNOWN = "unknown"                 # 未知錯誤


class FailureDetectionMode(Enum):
    """失敗檢測模式"""
    AUTO_VISION = "auto_vision"        # 自動視覺檢測（預設）
    HUMAN_NOTIFY = "human_notify"      # 人類主動通知
    HYBRID = "hybrid"                  # 混合模式（人類優先，視覺備援）


class FailureDetector:
    """
    失敗檢測器
    
    檢測模式：
    1. AUTO_VISION: 自動視覺檢測（延遲0.5秒，無人值守）
    2. HUMAN_NOTIFY: 人類語音通知（即時，需人在場）
    3. HYBRID: 混合模式（人類優先，視覺備援）← 推薦
    
    ⚠️ 注意：TM機械手臂無負載感測器，無法直接知道是否夾到物體
    """
    
    def __init__(self, detection_mode: FailureDetectionMode = FailureDetectionMode.HYBRID):
        """初始化失敗檢測器
        
        Args:
            detection_mode: 檢測模式（預設：HYBRID）
        """
        self.detection_mode = detection_mode
        
        # 閾值設定
        self.position_tolerance = 50.0  # mm - 位置偏差容忍度（5cm）
        self.verification_delay = 2.0   # 秒 - 執行後等待視覺穩定（已在 main 中處理）
        self.confidence_threshold = 0.6  # 視覺檢測最低信心度
        
        # 人類通知追蹤
        self.human_notified_failures: List[str] = []  # 人類已通知的失敗（object_id列表）
        
        # 歷史記錄
        self.failure_history: List[Dict] = []
        self.max_history = 100
    
    def detect_pickup_failure(self, 
                              target_object_id: str,
                              robot_status,
                              world_state,
                              action_result: Optional[Dict] = None) -> Optional[Dict]:
        """
        檢測抓取（pickup）失敗
        
        檢測流程（根據 detection_mode）：
        
        AUTO_VISION 模式：
        1. 夾爪閉合（gripper_state == 'closed'）
        2. 等待 0.5 秒視覺穩定
        3. 物體仍在原位（visibility='visible'）→ 失敗
        
        HUMAN_NOTIFY 模式：
        1. 等待人類語音通知（"沒夾到"）
        2. 收到通知後立即標記失敗
        
        HYBRID 模式：
        1. 優先等待人類通知（更快）
        2. 如果無通知，執行視覺檢測（備援）
        
        ⚠️ 注意：無負載感測器，無法從 IO 直接判斷是否夾到物體
        
        Args:
            target_object_id: 目標物體 ID
            robot_status: RobotStatus 物件
            world_state: WorldState 物件
            action_result: 執行結果（可選）
            
        Returns:
            Dict: {
                'failure_type': FailureType,
                'object_id': str,
                'reason': str,
                'evidence': Dict,
                'timestamp': float,
                'detected_by': str  # 'vision' or 'human'
            } or None (無失敗)
        """
        # === Step 0: 檢查人類是否已通知失敗 ===
        if self.detection_mode in [FailureDetectionMode.HUMAN_NOTIFY, FailureDetectionMode.HYBRID]:
            if target_object_id in self.human_notified_failures:
                # 人類已通知失敗，立即返回
                self.human_notified_failures.remove(target_object_id)
                
                failure = {
                    'failure_type': FailureType.GRASP_MISS,
                    'object_id': target_object_id,
                    'reason': f'人類通知：未成功夾取 {target_object_id}',
                    'evidence': {
                        'detection_method': 'human_notification',
                        'gripper_state': robot_status.gripper_state
                    },
                    'timestamp': time.time(),
                    'detected_by': 'human'
                }
                self._record_failure(failure)
                return failure
        
        # === Step 1: 視覺自動檢測（僅 AUTO_VISION 或 HYBRID 模式）===
        if self.detection_mode in [FailureDetectionMode.AUTO_VISION, FailureDetectionMode.HYBRID]:
            # 不再依賴夾爪狀態，直接檢查物體
            # （因為 TM 機器人 IO 狀態不可靠，且可能有延遲）
            
            obj_state = world_state.objects.get(target_object_id)
            
            # 詳細調試輸出
            logger.info(f"\n失敗檢測詳細資訊:")
            logger.info(f"   target_object_id: {target_object_id}")
            logger.info(f"   物體在 WorldState: {target_object_id in world_state.objects}")
            
            if obj_state:
                logger.info(f"   visibility: {obj_state.visibility}")
                logger.info(f"   confidence: {obj_state.confidence:.3f}")
                logger.info(f"   position: {obj_state.position}")
                logger.info(f"   location: {obj_state.location}")
                logger.info(f"   semantic_location: {getattr(obj_state, 'semantic_location', 'N/A')}")
                logger.info(f"   forced_by_io: {getattr(obj_state, 'forced_by_io', False)}")
                
                # 判斷 1: 物體完全消失（不在 WorldState）→ 成功
                # （此情況已在外層 if 判斷）
                
                # 判斷 2: 物體被 Robot IO 標記為 in_gripper → 成功
                if getattr(obj_state, 'semantic_location', None) == 'in_gripper':
                    logger.info(f"{target_object_id} 被 Robot IO 標記為 in_gripper，視為夾取成功")
                    return None
                
                # 判斷 3: 物體仍然可見且在桌上 → 確認抓取失敗
                if obj_state.visibility == 'visible' and obj_state.confidence > self.confidence_threshold:
                    logger.error(f"視覺檢測：{target_object_id} 仍在原位 (visibility=visible, confidence={obj_state.confidence:.2f})")
                    failure = {
                        'failure_type': FailureType.GRASP_MISS,
                        'object_id': target_object_id,
                        'reason': f'{obj_state.object_type}仍在原位，未成功夾取',
                        'evidence': {
                            'detection_method': 'vision_verification',
                            'gripper_state': robot_status.gripper_state,
                            'object_visibility': 'visible',
                            'object_confidence': obj_state.confidence,
                            'object_position': obj_state.position,
                            'semantic_location': getattr(obj_state, 'semantic_location', 'unknown')
                        },
                        'timestamp': time.time(),
                        'detected_by': 'vision'
                    }
                    self._record_failure(failure)
                    return failure
                
                # 判斷 4: 物體 visibility 是 occluded 但 confidence 仍高 → 可能誤判，需檢查語義位置
                elif obj_state.visibility == 'occluded' and obj_state.confidence > 0.5:
                    semantic_loc = getattr(obj_state, 'semantic_location', 'unknown')
                    if semantic_loc == 'on_table':
                        logger.warning(f"{target_object_id} visibility=occluded 但 semantic_location=on_table，可能是誤判遮擋")
                        logger.info(f"   confidence={obj_state.confidence:.2f} > 0.5，物體可能仍在桌上")
                        # 降低容忍度：如果 confidence > 0.7，視為失敗
                        if obj_state.confidence > 0.7:
                            logger.error(f"Confidence 過高 ({obj_state.confidence:.2f})，判定為夾取失敗")
                            failure = {
                                'failure_type': FailureType.GRASP_MISS,
                                'object_id': target_object_id,
                                'reason': f'{obj_state.object_type}仍在桌上（visibility=occluded 但 confidence={obj_state.confidence:.2f} 過高）',
                                'evidence': {
                                    'detection_method': 'vision_verification_stricter',
                                    'gripper_state': robot_status.gripper_state,
                                    'object_visibility': 'occluded',
                                    'object_confidence': obj_state.confidence,
                                    'object_position': obj_state.position,
                                    'semantic_location': semantic_loc
                                },
                                'timestamp': time.time(),
                                'detected_by': 'vision'
                            }
                            self._record_failure(failure)
                            return failure
                        else:
                            logger.info(f"Confidence 不高 ({obj_state.confidence:.2f})，視為夾取成功（可能被夾爪遮擋）")
                            return None
                    else:
                        logger.info(f"{target_object_id} visibility=occluded, semantic_location={semantic_loc}，視為夾取成功")
                        return None
                
                # 判斷 5: 物體不可見 → 可能成功夾起（被夾爪遮擋）
                elif obj_state.visibility in ['occluded', 'missing']:
                    logger.info(f"{target_object_id} 已消失/被遮擋 (visibility={obj_state.visibility})，視為夾取成功")
                    return None
                
                # 判斷 6: 物體可見但信心度低 → 不確定，保守視為成功
                else:
                    logger.warning(f"{target_object_id} 信心度低 ({obj_state.confidence:.2f})，保守視為夾取成功")
                    return None
            else:
                # 物體完全不在 WorldState → 成功（已被夾起）
                logger.info(f"WorldState 中找不到 {target_object_id}，視為夾取成功（物體已消失）")
                return None
        
        # === Step 3: 動作結果檢查（如果有）===
        if action_result and not action_result.get('success', True):
            failure_code = action_result.get('failure_code', 'unknown')
            failure = {
                'failure_type': FailureType.UNKNOWN,
                'object_id': target_object_id,
                'reason': f'動作執行失敗: {failure_code}',
                'evidence': action_result,
                'timestamp': time.time()
            }
            self._record_failure(failure)
            return failure
        
        # 無失敗檢測到
        return None
    
    def detect_place_failure(self,
                            target_object_id: str,
                            expected_position: List[float],
                            robot_status,
                            world_state,
                            action_result: Optional[Dict] = None) -> Optional[Dict]:
        """
        檢測放置（place）失敗
        
        檢測流程：
        1. 檢查 Robot IO - gripper_state == 'open' 且 holding_object_id == None
        2. 視覺二次確認 - 物體是否在目標位置附近
        3. 位置偏差檢查
        
        Args:
            target_object_id: 目標物體 ID
            expected_position: 預期放置位置 [x, y, z]
            robot_status: RobotStatus 物件
            world_state: WorldState 物件
            action_result: 執行結果（可選）
            
        Returns:
            Dict or None
        """
        # === Step 1: Robot IO Truth 檢查 ===
        gripper_open = robot_status.gripper_state == 'open'
        released_object = robot_status.holding_object_id is None
        
        if not (gripper_open and released_object):
            # 夾爪未正確打開
            failure = {
                'failure_type': FailureType.GRIPPER_OPEN_FAIL,
                'object_id': target_object_id,
                'reason': '夾爪未正確打開或仍持有物體',
                'evidence': {
                    'gripper_state': robot_status.gripper_state,
                    'holding_object': robot_status.holding_object_id
                },
                'timestamp': time.time()
            }
            self._record_failure(failure)
            return failure
        
        # === Step 2: 視覺二次確認 + 位置檢查 ===
        time.sleep(self.verification_delay)  # 等待視覺穩定
        
        obj_state = world_state.objects.get(target_object_id)
        if obj_state:
            # 物體可見
            if obj_state.visibility == 'visible' and obj_state.position:
                # 計算位置偏差
                actual_pos = obj_state.position
                position_error = self._calculate_position_error(expected_position, actual_pos)
                
                # 失敗信號 2: 位置偏差過大
                if position_error > self.position_tolerance:
                    failure = {
                        'failure_type': FailureType.POSITION_DRIFT,
                        'object_id': target_object_id,
                        'reason': f'放置位置偏差{position_error:.1f}mm，超過容忍度{self.position_tolerance}mm',
                        'evidence': {
                            'expected_position': expected_position,
                            'actual_position': actual_pos,
                            'position_error': position_error,
                            'object_confidence': obj_state.confidence
                        },
                        'timestamp': time.time()
                    }
                    self._record_failure(failure)
                    return failure
                
                # 成功：位置在容忍範圍內
                logger.info(f"放置成功：{target_object_id}，位置偏差 {position_error:.1f}mm")
                return None
            
            # 物體不可見 → 可能被遮擋
            elif obj_state.visibility in ['occluded', 'missing']:
                # 暫時無法判斷，等待重新可見
                logger.warning(f"放置不確定：{target_object_id} 現在{obj_state.visibility}")
                return None
        
        # 找不到物體 → 可能是 ID tracking 問題
        else:
            failure = {
                'failure_type': FailureType.VISION_VERIFY_FAIL,
                'object_id': target_object_id,
                'reason': f'放置後找不到{target_object_id}在 World State 中',
                'evidence': {'world_state_objects': list(world_state.objects.keys())},
                'timestamp': time.time()
            }
            self._record_failure(failure)
            return failure
        
        return None
    
    def _calculate_position_error(self, expected: List[float], actual: List[float]) -> float:
        """計算 3D 位置誤差（歐氏距離）"""
        import numpy as np
        exp = np.array(expected)
        act = np.array(actual)
        return float(np.linalg.norm(exp - act))
    
    def notify_failure_by_human(self, object_id: str, failure_description: str = None):
        """
        人類主動通知失敗（語音觸發）
        
        使用場景：
        - 人類說："沒夾到 cylinder_1"
        - 人類說："夾爪空的"
        - 人類說："失敗了"
        
        Args:
            object_id: 目標物體 ID
            failure_description: 人類的描述（可選）
        """
        if object_id not in self.human_notified_failures:
            self.human_notified_failures.append(object_id)
            logger.info(f"人類通知失敗: {object_id}")
            if failure_description:
                logger.info(f"   描述: {failure_description}")
    
    def clear_human_notifications(self):
        """清除所有人類通知（用於重置）"""
        self.human_notified_failures.clear()
    
    def set_detection_mode(self, mode: FailureDetectionMode):
        """
        切換檢測模式
        
        Args:
            mode: AUTO_VISION / HUMAN_NOTIFY / HYBRID
        """
        self.detection_mode = mode
        logger.info(f"失敗檢測模式切換為: {mode.value}")
    
    def _record_failure(self, failure: Dict):
        """記錄失敗到歷史"""
        self.failure_history.append(failure)
        
        # 限制歷史大小
        if len(self.failure_history) > self.max_history:
            self.failure_history.pop(0)
        
        # 調試輸出
        logger.info(f"檢測到失敗: {failure['failure_type'].value}")
        logger.info(f"   物體: {failure['object_id']}")
        logger.info(f"   原因: {failure['reason']}")
    
    def get_failure_statistics(self) -> Dict:
        """獲取失敗統計"""
        if not self.failure_history:
            return {'total': 0, 'by_type': {}}
        
        stats = {'total': len(self.failure_history), 'by_type': {}}
        
        for failure in self.failure_history:
            failure_type = failure['failure_type'].value
            stats['by_type'][failure_type] = stats['by_type'].get(failure_type, 0) + 1
        
        return stats
    
    def clear_history(self):
        """清空失敗歷史"""
        self.failure_history.clear()


def get_failure_detector(detection_mode: FailureDetectionMode = FailureDetectionMode.HYBRID) -> FailureDetector:
    """
    全局單例
    
    Args:
        detection_mode: 失敗檢測模式（預設 HYBRID）
            - AUTO_VISION: 自動視覺檢測（0.5s 延遲）
            - HUMAN_NOTIFY: 僅人類主動通知
            - HYBRID: 人類優先 + 視覺備援（推薦）
    """
    if not hasattr(get_failure_detector, '_instance'):
        get_failure_detector._instance = FailureDetector(detection_mode=detection_mode)
    return get_failure_detector._instance
