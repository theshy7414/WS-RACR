#!/usr/bin/env python3
"""
Occlusion Handling Strategy (Smart 3-Tier Classification)
遮擋處理策略（智能三層分類）

核心理念：遮擋 ≠ 看不到 ≠ 不可處理

三類遮擋分類：
🟢 Type A: 可移動、可安全處理的遮擋（機器人自己移除）
🟡 Type B: 不確定是否可移動（先降低不確定性）
🔴 Type C: 不可移動/不應移動（人類介入）

決策流程：
1. 偵測遮擋 → 記錄 occluded_by
2. 判斷遮擋物類型 → Type A/B/C
3. Type A → 機器人移除遮擋物
4. Type B → 嘗試獲取更多資訊或盲抓
5. Type C → 請求人類協助（有語意的請求）
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class OcclusionHandlingStrategy:
    """
    遮擋處理策略（智能三層分類）
    
    區分三件事：
    1️⃣ 我「知不知道」下面有什麼？→ last_seen 記憶 + 任務 SOP
    2️⃣ 我「能不能安全地自己處理」？→ 遮擋物可移動性判斷
    3️⃣ 自己處理 vs 人類處理，哪個比較合理？→ 成功率 + 風險 + 任務中斷成本
    """
    
    def __init__(self):
        """初始化遮擋處理策略"""
        # 遮擋時間閾值
        self.blind_grasp_timeout = 3.0  # 3秒內可以盲抓
        self.ask_human_timeout = 10.0   # 10秒後詢問用戶
        
        # 空間鄰近閾值（mm）
        self.occlusion_distance_threshold = 100.0  # 10cm內可能遮擋
        
        # === 物體可移動性分類 ===
        # Type C: 不可移動（需人類協助）
        self.non_movable_objects = {'platform', 'bracket'}
        
        # Type B: 謹慎處理（高風險）
        self.high_risk_objects = {'pi'}  # Pi板需謹慎
        
        # Type A: 可安全移動（其他工具和零件）
        # 包括: cylinder, nut, screw_*, hex_wrench 等
        
        # 策略執行歷史
        self.strategy_history: Dict[str, Dict] = {}
    
    def get_best_strategy(self, 
                          occluded_obj, 
                          world_state,
                          time_occluded: float) -> Optional[Dict]:
        """
        為被遮擋物品選擇最佳策略（智能三層分類）
        
        Args:
            occluded_obj: 被遮擋的 ObjectState
            world_state: 當前 WorldState
            time_occluded: 被遮擋時長（秒）
            
        Returns:
            Dict: {
                'strategy': 'blind_grasp' | 'remove_occluder' | 'ask_human',
                'action': str,
                'parameters': Dict,
                'explanation': str,
                'occlusion_type': 'A' | 'B' | 'C'
            }
        """
        # === Step 1: 找出可能的遮擋物 ===
        occluder_result = self._find_likely_occluder(occluded_obj, world_state)
        
        # 短時間遮擋且沒找到明確遮擋物 → 請求協助
        if time_occluded < self.blind_grasp_timeout and not occluder_result:
            # 口語化時間描述
            if time_occluded < 1.0:
                time_desc = "剛剛"
            elif time_occluded < 3.0:
                time_desc = f"{int(time_occluded)}秒前"
            else:
                time_desc = f"約{int(time_occluded)}秒前"
            
            return {
                'strategy': 'ask_human',
                'action': 'request_help',
                'parameters': {
                    'target_object': occluded_obj.object_id,
                    'last_seen_position': occluded_obj.position,
                    'request_type': 'remove_occlusion',
                    'question': f'{occluded_obj.object_type}被遮擋了，請幫我移開遮擋物'
                },
                'explanation': f'{occluded_obj.object_type}{time_desc}被遮住了，請你幫我移開障礙物好嗎',
                'occlusion_type': 'B'
            }
        
        # === Step 2: 如果找到遮擋物，判斷類型 ===
        if occluder_result:
            occluder = occluder_result['obj']
            confidence = occluder_result.get('confidence', 0.7)
            
            # 更新 World State 中的 occluded_by
            occluded_obj.occluded_by = occluder.object_id
            occluded_obj.occluded_by_confidence = confidence
            
            # === 判斷遮擋物是否可移動 ===
            occlusion_type, movability = self._classify_occluder(occluder, occluded_obj, world_state)
            
            # Type C: 不可移動（bracket, platform）→ 請人類協助
            if occlusion_type == 'C':
                return {
                    'strategy': 'ask_human',
                    'action': 'request_help',
                    'parameters': {
                        'target_object': occluded_obj.object_id,
                        'occluder_object': occluder.object_id,
                        'last_seen_position': occluded_obj.position,
                        'request_type': 'move_occluder',
                        'question': f'我需要拿{occluded_obj.object_type}，但現在被{occluder.object_type}遮住了，您能幫我移開{occluder.object_type}或指示{occluded_obj.object_type}的位置嗎？'
                    },
                    'explanation': f'我需要拿{occluded_obj.object_type}，但被{occluder.object_type}遮住了，請你幫我移開它好嗎',
                    'occlusion_type': 'C'
                }
            
            # Type B: 高風險（Pi板等）→ 謹慎處理
            elif occlusion_type == 'B':
                if confidence < 0.5:  # 不確定遮擋關係
                    return {
                        'strategy': 'ask_human',
                        'action': 'request_help',
                        'parameters': {
                            'target_object': occluded_obj.object_id,
                            'last_seen_position': occluded_obj.position,
                            'request_type': 'uncertain_occlusion',
                            'question': f'{occluded_obj.object_type}被遮擋了，不確定遮擋關係，請協助'
                        },
                        'explanation': f'{occluded_obj.object_type}看不到了，但我不確定是什麼東西擋住的，需要你幫忙',
                        'occlusion_type': 'B'
                    }
                else:
                    # 高風險物體，謹慎詢問
                    return {
                        'strategy': 'ask_human',
                        'action': 'request_help',
                        'parameters': {
                            'target_object': occluded_obj.object_id,
                            'occluder_object': occluder.object_id,
                            'request_type': 'confirm_move',
                            'question': f'{occluded_obj.object_type}被{occluder.object_type}遮住了，{occluder.object_type}可以移動嗎？'
                        },
                        'explanation': f'{occluded_obj.object_type}被{occluder.object_type}遮住了，但我不確定能不能移它，需要你確認一下',
                        'occlusion_type': 'B'
                    }
            
            # Type A: 可安全移動 → 機器人自己移除
            else:  # occlusion_type == 'A'
                return {
                    'strategy': 'remove_occluder',
                    'action': 'move_occluder',
                    'parameters': {
                        'target_object': occluded_obj.object_id,
                        'occluder_object': occluder.object_id,
                        'occluder_position': occluder.position,
                        'move_to': self._get_safe_placement_position(occluder, world_state)
                    },
                    'explanation': f'{occluded_obj.object_type}被{occluder.object_type}遮住了，我先移開{occluder.object_type}',
                    'occlusion_type': 'A'
                }
        
        # === 沒找到遮擋物但遮擋很久 → 請求協助 ===
        if time_occluded > self.ask_human_timeout:
            return {
                'strategy': 'ask_human',
                'action': 'request_help',
                'parameters': {
                    'target_object': occluded_obj.object_id,
                    'last_seen_position': occluded_obj.position,
                    'request_type': 'locate',
                    'question': f'{occluded_obj.object_type}已經找不到{time_occluded:.0f}秒了，您能幫我找一下嗎？'
                },
                'explanation': f'{occluded_obj.object_type}已經看不到{int(time_occluded)}秒了，需要你幫我找一下',
                'occlusion_type': 'C'
            }
        
        # === 默認：請求協助 ===
        return {
            'strategy': 'ask_human',
            'action': 'request_help',
            'parameters': {
                'target_object': occluded_obj.object_id,
                'last_seen_position': occluded_obj.position,
                'request_type': 'locate',
                'question': f'{occluded_obj.object_type}被遮擋了，請協助移除遮擋物'
            },
            'explanation': f'{occluded_obj.object_type}被遮住了，請你幫我移開障礙物好嗎',
            'occlusion_type': 'B'
        }
    
    def _classify_occluder(self, occluder, occluded_obj, world_state) -> Tuple[str, str]:
        """
        判斷遮擋物類型（A/B/C）和可移動性
        
        Returns:
            Tuple[occlusion_type, movability]:
                occlusion_type: 'A' | 'B' | 'C'
                movability: 'robot_safe' | 'high_risk' | 'human_only'
        """
        occluder_type = occluder.object_type
        
        # Type C: 不可移動（bracket, platform）
        if occluder_type in self.non_movable_objects:
            return 'C', 'human_only'
        
        # Type B: 高風險（Pi板等）
        if occluder_type in self.high_risk_objects:
            return 'B', 'high_risk'
        
        # 檢查是否在任務關鍵步驟中
        if self._is_task_critical(occluder, world_state):
            return 'B', 'high_risk'
        
        # Type A: 可安全移動（工具、零件）
        return 'A', 'robot_safe'
    
    def _is_task_critical(self, obj, world_state) -> bool:
        """
        判斷物體是否在任務關鍵步驟中（不應亂動）
        
        例如：
        - 正在組裝的零件
        - 已固定的螺絲
        """
        # 簡化版本：檢查 location_stable
        if hasattr(obj, 'location_stable'):
            # 如果物體已經被固定（in_platform, in_bracket），不應移動
            if obj.location_stable in ['in_platform', 'in_bracket', 'screwed']:
                return True
        
        return False
    
    def _find_likely_occluder(self, occluded_obj, world_state) -> Optional[Dict]:
        """
        找出最可能的遮擋物（增強版：返回信心度）
        
        規則：
        1. 在被遮擋物上方（Z 高度更高）
        2. XY 平面距離 < threshold
        3. 當前可見（visibility='visible'）
        
        Returns:
            Dict: {'obj': ObjectState, 'score': float, 'confidence': float} or None
        """
        if occluded_obj.position is None:
            return None
        
        occluded_pos = np.array(occluded_obj.position)
        candidates = []
        
        for obj_id, obj_state in world_state.objects.items():
            # 跳過自己
            if obj_id == occluded_obj.object_id:
                continue
            
            # 只考慮可見物品
            if obj_state.visibility != 'visible':
                continue
            
            # 需要有位置
            if obj_state.position is None:
                continue
            
            obj_pos = np.array(obj_state.position)
            
            # 計算 XY 平面距離
            xy_dist = np.linalg.norm(obj_pos[:2] - occluded_pos[:2])
            
            # === Eye-to-hand 相機：Z 越小越靠近相機（越在上方）===
            # 遮擋物的 Z 應該 < 被遮擋物的 Z（更靠近相機）
            z_diff = occluded_pos[2] - obj_pos[2]  # 正值表示遮擋物在上方
            
            # 如果在附近且在上方
            if xy_dist < self.occlusion_distance_threshold:
                if z_diff > -20.0:  # 允許稍低一點（-2cm）
                    score = 1.0 / (xy_dist + 1.0)  # 距離越近分數越高
                    # Z diff 貢獻：越在上方分數越高
                    if z_diff > 0:
                        score += z_diff / 100.0  # Z 差異的獎勵
                    candidates.append({
                        'obj': obj_state,
                        'score': score,
                        'xy_dist': xy_dist,
                        'z_diff': z_diff
                    })
        
        if not candidates:
            return None
        
        # 按分數排序，返回最可能的遮擋物
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        
        # 計算信心度（基於距離和數量）
        confidence = min(0.95, best['score'] * 0.5)  # 越近信心度越高
        if len(candidates) == 1:
            confidence += 0.2  # 唯一候選者，信心度更高
        
        best['confidence'] = confidence
        
        logger.info(f"   找到可能的遮擋物: {best['obj'].object_id}")
        logger.info(f"      距離: XY={best['xy_dist']:.0f}mm, Z={best['z_diff']:.0f}mm")
        logger.info(f"      信心度: {confidence:.1%}")
        
        return best
    
    def _get_safe_placement_position(self, obj_state, world_state) -> List[float]:
        """
        為遮擋物找一個安全的放置位置
        
        策略：移到托盤邊緣或空曠區域
        """
        # 簡單策略：移到托盤左側或右側空曠處
        left_pos = [-150.0, 400.0, 50.0]
        right_pos = [150.0, 400.0, 50.0]
        
        # 檢查哪側更空曠
        left_count = self._count_nearby_objects(left_pos, world_state, radius=100.0)
        right_count = self._count_nearby_objects(right_pos, world_state, radius=100.0)
        
        return left_pos if left_count <= right_count else right_pos
    
    def _count_nearby_objects(self, position: List[float], world_state, radius: float) -> int:
        """計算位置附近的物品數量"""
        pos = np.array(position)
        count = 0
        
        for obj_state in world_state.objects.values():
            if obj_state.position is None:
                continue
            obj_pos = np.array(obj_state.position)
            dist = np.linalg.norm(obj_pos[:2] - pos[:2])
            if dist < radius:
                count += 1
        
        return count
    
    def check_and_suggest_strategy(self, world_state) -> Optional[Dict]:
        """
        檢查 World State 中所有被遮擋的物品，建議處理策略
        
        Returns:
            Optional[Dict]: 策略建議（如果有被遮擋物品）
        """
        current_time = time.time()
        
        for obj_id, obj_state in world_state.objects.items():
            if obj_state.visibility == 'occluded' and obj_state.last_seen:
                time_occluded = current_time - obj_state.last_seen
                
                # 獲取最佳策略
                strategy = self.get_best_strategy(obj_state, world_state, time_occluded)
                
                if strategy:
                    logger.info(f"被遮擋物品: {obj_id}")
                    logger.info(f"   類型: {strategy['occlusion_type']} - {strategy['strategy']}")
                    logger.info(f"   說明: {strategy['explanation']}")
                    return strategy
        
        return None
    
    def record_strategy_result(self, object_id: str, strategy: str, success: bool):
        """記錄策略執行結果（用於學習）"""
        self.strategy_history[object_id] = {
            'strategy': strategy,
            'time': time.time(),
            'success': success
        }


def get_occlusion_handler() -> OcclusionHandlingStrategy:
    """全局單例"""
    if not hasattr(get_occlusion_handler, '_instance'):
        get_occlusion_handler._instance = OcclusionHandlingStrategy()
    return get_occlusion_handler._instance
