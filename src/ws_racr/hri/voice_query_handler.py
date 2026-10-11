"""Voice query handler: rule-based answers to questions about the workbench and the assembly step.

Supported queries: inventory ("what is on the table"), location ("where is X", via spatial relations),
count ("how many X"), availability ("is X still there"), current / next step, and SOP look-ups.
These answers use the world state and object memory directly, without an LLM call (the ANSWER fast path).
"""

import logging
from typing import Dict, List, Optional, Tuple
import re

from ws_racr.config import get_sop_yaml_path
from ws_racr.world_state.step_definitions import (
    user_input_to_step_index,
    step_index_to_user_display,
    get_step_class,
    get_step_name_zh
)

logger = logging.getLogger(__name__)


class VoiceQueryHandler:
    """
    語音查詢處理器
    
    支持的查詢類型：
    1. 庫存查詢："桌上還有哪些物品"
    2. 定位查詢："X 在哪裡"
    3. 數量查詢："有幾個 X"
    4. 狀態查詢："X 還在嗎"
    """
    
    def __init__(self, object_memory_system, step_fsm=None, world_state_getter=None, sop_path=None, spatial_memory=None, world_state_updater=None):
        """
        Args:
            object_memory_system: ObjectMemorySystem 實例
            step_fsm: StepStateMachine 實例（可選）
            world_state_getter: 獲取當前 world state 的函數（可選）
            sop_path: SOP YAML 文件路徑（可選）
            spatial_memory: SpatialMemory 實例（可選，用於空間關係查詢）
            world_state_updater: callable(step_class, confidence) 用於同步 WorldState.step_class（可選）
        """
        self.memory = object_memory_system
        self.step_fsm = step_fsm
        self.get_world_state = world_state_getter
        self.update_world_state_step = world_state_updater
        self.sop_path = sop_path or str(get_sop_yaml_path())
        self.spatial_memory = spatial_memory
        
        # 載入 SOP（如果有）
        self.sop_data = None
        if sop_path:
            self._load_sop()
        
        # 查詢模式映射
        self.patterns = {
            'inventory': [
                r'桌上.*有.*哪些',
                r'桌面.*有.*哪些',
                r'桌上.*有.*什麼',
                r'桌面.*有.*什麼',
                r'桌上.*什麼.*物',
                r'桌面.*什麼.*物',
                r'還有.*哪些.*物',
                r'還有.*什麼.*東西',
                r'有.*什麼.*物品',
                r'有哪些.*物',
                r'剩下.*哪些',
                r'列出.*所有',
            ],
            'locate': [
                r'(.+)在哪',
                r'(.+)的位置',
                r'找(.+)',
                r'(.+)放在',
            ],
            'count': [
                r'有幾個(.+)',
                r'(.+)有幾個',
                r'還有多少(.+)',
                r'數量',
            ],
            'status': [
                r'(.+)還在嗎',
                r'(.+)不見了',
                r'看得到(.+)嗎',
            ],
            'progress': [
                r'現在.*第幾步',
                r'做到.*哪',
                r'進度.*如何',
                r'完成.*多少',
            ],
            'next_step': [
                r'下一步.*什麼',
                r'接下來.*做',
                r'下個步驟',
                r'然後.*要',
            ],
            'step_status': [
                r'(.+)完成了嗎',
                r'(.+)做好了嗎',
                r'我剛剛.*完成',
            ],
            'missing': [
                r'我漏了.*什麼',
                r'還缺.*什麼',
                r'遺漏.*什麼',
            ],
            'spatial_relation': [
                r'(.+)在(.+)的.*哪',
                r'(.+)在(.+).*方向',
                r'(.+)相對(.+)',
            ],
            'nearby': [
                r'(.+)附近有.*什麼',
                r'(.+)旁邊有.*什麼',
                r'(.+)周圍',
            ],
            'deliver_materials': [
                r'給我.*第(.+)步',
                r'遞.*第(.+)步',
                r'拿.*第(.+)步',
                r'請給我.*第(.+)步',
                r'幫我拿.*第(.+)步',
                r'準備.*第(.+)步',
            ],
            'step_materials': [
                r'第(.+)步.*需要',
                r'第(.+)部.*需要',  # STT 常把「步」辨成「部」
                r'(步驟|step)\s*(.+).*需要',
                r'(.+)步.*要.*零件',
                r'(.+)步.*要.*什麼',
                r'(.+)步.*要.*哪些',
                r'(.+)部.*要.*什麼',
                r'(.+)部.*要.*哪些',
                r'下一步.*需要',
                r'接下來.*需要',
            ],
            'update_step': [
                r'更新.*步驟.*到.*第(.+)步',
                r'跳到.*第(.+)步',
                r'切換.*到.*第(.+)步',
                r'現在.*在.*第([一二三四五六七八1-8])步',  # 更精確：需要有「在」字且步驟編號明確
                r'現在.*是.*第([一二三四五六七八1-8])步',
                r'改成.*第(.+)步',
                r'我們.*目前.*在.*步驟([一二三四五六七八1-8])',
                r'我們.*現在.*在.*步驟([一二三四五六七八1-8])',
                r'我們.*在.*第([一二三四五六七八1-8])步',
                r'目前.*在.*步驟([一二三四五六七八1-8])',
                r'目前.*第([一二三四五六七八1-8])步',
            ]
        }
        
        # 物體類別中英文映射
        self.object_names_zh = {
            '螺絲起子': 'screwdriver',
            '起子': 'screwdriver',
            '六角扳手': 'hex_wrench',
            '扳手': 'hex_wrench',
            '六角': 'hex_wrench',
            '平台': 'platform',
            '底座': 'platform',
            '支架': 'bracket',
            'PI字型工件': 'pi_shaped_part',
            'PI工件': 'pi_shaped_part',
            'pi': 'pi_shaped_part',
            '圓柱': 'cylinder',
            '螺絲': 'screw',
            '大螺絲': 'screw_big',
            '大號螺絲': 'screw_big',
            '小螺絲': 'screw_small',
            '小號螺絲': 'screw_small',
            '螺帽': 'nut',
            '螺母': 'nut',
        }
        
        logger.info("Voice Query Handler Initialized")
        if self.sop_data:
            logger.info(f"   SOP loaded: {len(self.sop_data.get('phase_1', {}).get('steps', []))} steps in phase 1")
    
    def _load_sop(self):
        """載入 SOP YAML 文件"""
        try:
            import yaml
            from pathlib import Path
            
            sop_file = Path(self.sop_path)
            if sop_file.exists():
                with open(sop_file, 'r', encoding='utf-8') as f:
                    self.sop_data = yaml.safe_load(f)
                return True
            else:
                logger.warning(f" SOP file not found: {self.sop_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load SOP: {e}")
            return False
    
    def parse_and_execute(self, query_text: str) -> Dict:
        """
        解析並執行查詢
        
        Args:
            query_text: 查詢文本（中文）
        
        Returns:
            查詢結果:
                {
                    'success': bool,
                    'query_type': str,
                    'result': Any,
                    'response_text': str  # 給用戶的回覆
                }
        """
        query_text = query_text.strip()
        
        # 1. 識別查詢類型
        query_type = self._identify_query_type(query_text)
        
        if query_type is None:
            return {
                'success': False,
                'query_type': 'unknown',
                'result': None,
                'response_text': '抱歉，我不理解這個查詢。請嘗試問："桌上還有哪些物品" 或 "螺絲起子在哪"'
            }
        
        # 2. 執行查詢
        if query_type == 'inventory':
            return self._handle_inventory_query(query_text)
        elif query_type == 'locate':
            return self._handle_locate_query(query_text)
        elif query_type == 'count':
            return self._handle_count_query(query_text)
        elif query_type == 'status':
            return self._handle_status_query(query_text)
        elif query_type == 'progress':
            return self._handle_progress_query(query_text)
        elif query_type == 'next_step':
            return self._handle_next_step_query(query_text)
        elif query_type == 'step_status':
            return self._handle_step_status_query(query_text)
        elif query_type == 'missing':
            return self._handle_missing_query(query_text)
        elif query_type == 'spatial_relation':
            return self._handle_spatial_relation_query(query_text)
        elif query_type == 'nearby':
            return self._handle_nearby_query(query_text)
        elif query_type == 'step_materials':
            return self._handle_step_materials_query(query_text)
        elif query_type == 'update_step':
            return self._handle_update_step_query(query_text)
        elif query_type == 'deliver_materials':
            return self._handle_deliver_materials_query(query_text)
        else:
            return {
                'success': False,
                'query_type': query_type,
                'result': None,
                'response_text': f'不支持的查詢類型: {query_type}'
            }
    
    def _identify_query_type(self, text: str) -> Optional[str]:
        """
        識別查詢類型
        
        優先順序：
        1. update_step - 最具體（「我們目前在步驟X」）
        2. 其他類型
        """

        
        # 優先檢查 update_step（避免被 progress 誤匹配）
        priority_types = ['update_step']
        for qtype in priority_types:
            if qtype in self.patterns:
                for pattern in self.patterns[qtype]:
                    if re.search(pattern, text):
                        return qtype
        
        # 然後檢查其他類型
        for qtype, patterns in self.patterns.items():
            if qtype in priority_types:
                continue  # 已經檢查過了
            for pattern in patterns:
                if re.search(pattern, text):
                    return qtype
        
        return None
    
    def _handle_inventory_query(self, text: str) -> Dict:
        """處理庫存查詢"""
        inventory = self.memory.query_inventory(visible_only=True)
        
        if not inventory:
            return {
                'success': True,
                'query_type': 'inventory',
                'result': [],
                'response_text': '桌上目前沒有檢測到任何物品。'
            }
        
        # 按類別分組
        by_class = {}
        for obj in inventory:
            cls = obj['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(obj)
        
        # 生成回覆文本
        items_zh = []
        for cls, objs in by_class.items():
            cls_zh = self._translate_class_to_zh(cls)
            count = len(objs)
            items_zh.append(f"{cls_zh} {count}個")
        
        response = f"桌上目前有：{', '.join(items_zh)}"
        
        return {
            'success': True,
            'query_type': 'inventory',
            'result': inventory,
            'response_text': response
        }
    
    def _handle_locate_query(self, text: str) -> Dict:
        """處理定位查詢"""
        # 提取物體名稱
        obj_name_zh = None
        for pattern in self.patterns['locate']:
            match = re.search(pattern, text)
            if match:
                obj_name_zh = match.group(1).strip()
                break
        
        if not obj_name_zh:
            return {
                'success': False,
                'query_type': 'locate',
                'result': None,
                'response_text': '無法識別要查找的物品。'
            }
        
        # 轉換為英文類別
        obj_class = self.object_names_zh.get(obj_name_zh)
        if not obj_class:
            return {
                'success': False,
                'query_type': 'locate',
                'result': None,
                'response_text': f'不認識 "{obj_name_zh}"。請使用已知的物品名稱。'
            }
        
        # 查詢位置（先查可見物品）
        result = self.memory.locate_object(obj_class, visible_only=True)
        
        if not result:
            # 如果看不到，檢查是否被遮擋（從 World State 查詢）
            if self.get_world_state:
                world_state = self.get_world_state()
                occluded_obj = self._find_occluded_object(world_state, obj_class)
                
                if occluded_obj:
                    # 找到被遮擋的物品
                    pos = occluded_obj.position
                    relative_desc = self._get_relative_position_description(pos)
                    
                    # 智能建議：根據遮擋時間決定策略
                    import time
                    time_occluded = time.time() - occluded_obj.last_seen
                    
                    if time_occluded < 3.0:
                        # 短時間遮擋 → 建議盲抓
                        suggestion = "我可以去最後位置試試看"
                    else:
                        # 長時間遮擋 → 建議移除遮擋物或詢問
                        suggestion = "可能被其他東西擋住了，需要我移開遮擋物嗎？"
                    
                    return {
                        'success': True,
                        'query_type': 'locate',
                        'result': {
                            'object_id': occluded_obj.object_id,
                            'class': occluded_obj.object_type,
                            'position': pos,
                            'visibility': 'occluded',
                            'time_occluded': time_occluded
                        },
                        'response_text': f'{obj_name_zh}最後在{relative_desc}看到（{time_occluded:.0f}秒前），現在被遮住了。{suggestion}',
                        'suggest_action': 'blind_grasp' if time_occluded < 3.0 else 'remove_occluder'
                    }
            
            return {
                'success': True,
                'query_type': 'locate',
                'result': None,
                'response_text': f'{obj_name_zh}目前不在視野內或已被移走。'
            }
        
        # 生成相對位置描述
        pos = result['position']
        relative_desc = self._get_relative_position_description(pos)
        
        response = f'{obj_name_zh}{relative_desc}'
        
        # 添加精確座標（可選）
        # response += f'（座標: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] 米）'
        
        return {
            'success': True,
            'query_type': 'locate',
            'result': result,
            'response_text': response
        }
    
    def _handle_count_query(self, text: str) -> Dict:
        """處理數量查詢"""
        logger.info(f"[Count Query] 處理數量查詢: '{text}'")
        # 提取物體名稱
        obj_name_zh = None
        for pattern in self.patterns['count']:
            match = re.search(pattern, text)
            if match:
                obj_name_zh = match.group(1).strip()
                logger.info(f"   提取到物品名稱: '{obj_name_zh}'")
                break
        
        if not obj_name_zh:
            # 如果沒有指定物品，返回總數
            inventory = self.memory.query_inventory(visible_only=True)
            return {
                'success': True,
                'query_type': 'count',
                'result': len(inventory),
                'response_text': f'桌上目前有 {len(inventory)} 個物品。'
            }
        
        # 轉換為英文類別
        obj_class = self.object_names_zh.get(obj_name_zh)
        logger.info(f"   查找類別映射: '{obj_name_zh}' → '{obj_class}'")
        if not obj_class:
            logger.error(f"[Count Query] 不認識物品: '{obj_name_zh}'")
            logger.info(f"   可用映射: {list(self.object_names_zh.keys())}")
            return {
                'success': False,
                'query_type': 'count',
                'result': None,
                'response_text': f'不認識 "{obj_name_zh}"。'
            }
        
        # 計數
        inventory = self.memory.query_inventory(visible_only=True)
        count = sum(1 for obj in inventory if obj['class'] == obj_class)
        
        return {
            'success': True,
            'query_type': 'count',
            'result': count,
            'response_text': f'{obj_name_zh}有 {count} 個。'
        }
    
    def _handle_status_query(self, text: str) -> Dict:
        """處理狀態查詢"""
        # 提取物體名稱
        obj_name_zh = None
        for pattern in self.patterns['status']:
            match = re.search(pattern, text)
            if match:
                obj_name_zh = match.group(1).strip()
                break
        
        if not obj_name_zh:
            return {
                'success': False,
                'query_type': 'status',
                'result': None,
                'response_text': '無法識別要查詢的物品。'
            }
        
        # 轉換為英文類別
        obj_class = self.object_names_zh.get(obj_name_zh)
        if not obj_class:
            return {
                'success': False,
                'query_type': 'status',
                'result': None,
                'response_text': f'不認識 "{obj_name_zh}"。'
            }
        
        # 查詢狀態
        result = self.memory.locate_object(obj_class, visible_only=False)
        
        if not result:
            return {
                'success': True,
                'query_type': 'status',
                'result': {'visible': False, 'exists': False},
                'response_text': f'{obj_name_zh}從未出現過。'
            }
        
        if result['visible']:
            response = f'是的，{obj_name_zh}還在桌上，信心度 {result["confidence"]:.0%}'
        else:
            response = f'{obj_name_zh}目前不在視野內（可能被遮擋或已移走）'
        
        return {
            'success': True,
            'query_type': 'status',
            'result': result,
            'response_text': response
        }
    
    def _translate_class_to_zh(self, obj_class: str) -> str:
        """將物體類別翻譯為中文"""
        for zh, en in self.object_names_zh.items():
            if en == obj_class:
                return zh
        return obj_class  # 如果找不到，返回原文
    
    def _get_relative_position_description(self, position: List[float]) -> str:
        """生成相對位置的自然語言描述"""
        import numpy as np
        
        x, y, z = position
        
        # 計算距離
        dist_xy = np.sqrt(x**2 + y**2)
        
        # 方向描述
        direction = ""
        if abs(x) > 0.1 or abs(y) > 0.1:  # 有明顯偏移
            # 前後
            if x > 0.2:
                direction += "前"
            elif x < -0.2:
                direction += "後"
            
            # 左右
            if y > 0.2:
                direction += "左"
            elif y < -0.2:
                direction += "右"
            
            if direction:
                direction += "方"
        
        # 組合描述
        if not direction:
            desc = "在工作區中央"
        else:
            desc = f"在工作區{direction}"
        
        # 添加距離信息
        if dist_xy > 0.1:
            desc += f"，距離約 {dist_xy*100:.0f} 公分"
        
        return desc
    
    def _find_occluded_object(self, world_state, obj_class: str):
        """
        從 World State 中找被遮擋的物品
        
        Args:
            world_state: WorldState 物件
            obj_class: 物品類別（英文）
            
        Returns:
            ObjectState or None
        """
        for obj_id, obj_state in world_state.objects.items():
            if obj_state.object_type == obj_class and obj_state.visibility == 'occluded':
                return obj_state
        return None
    
    def _handle_progress_query(self, text: str) -> Dict:
        """處理進度查詢：我們現在第幾步？"""
        if self.step_fsm is None:
            return {
                'success': False,
                'query_type': 'progress',
                'result': None,
                'response_text': '步驟追蹤系統未啟用。'
            }
        
        # 獲取當前步驟
        from ws_racr.world_state.step_definitions import get_step_name_zh, step_index_to_user_display
        
        fsm_state = self.step_fsm.get_state()
        step_index = fsm_state['step_index']  # TSM index (0-8)
        user_display = step_index_to_user_display(step_index)  # 用戶看到的步驟編號
        step_name = get_step_name_zh(step_index)
        logger.info(f"[DEBUG] 進度查詢: step_index={step_index}, user_display={user_display}, step_name={step_name}")
        
        # 獲取世界狀態（如果有）
        assembly_progress = None
        if self.get_world_state:
            try:
                world_state = self.get_world_state()
                assembly_progress = world_state.get('assembly_progress', {})
            except:
                pass
        
        # 生成回覆
        if step_index == 8:  # idle
            response = "目前處於待機狀態，尚未開始組裝。"
        else:
            response = f"目前在{step_name}"
            
            # 添加進度信息
            if assembly_progress:
                cylinders = assembly_progress.get('cylinders_inserted', 0)
                screws_small = assembly_progress.get('screws_small_used', 0)
                if cylinders > 0 or screws_small > 0:
                    response += f"\n進度：已插入 {cylinders}/4 個圓柱，已鎖 {screws_small}/4 個小螺絲。"
        
        return {
            'success': True,
            'query_type': 'progress',
            'result': {
                'step_index': step_index,
                'user_display': user_display,
                'assembly_progress': assembly_progress
            },
            'response_text': response
        }
    
    def _handle_next_step_query(self, text: str) -> Dict:
        """處理下一步查詢：下一步要做什麼？"""
        if self.step_fsm is None:
            return {
                'success': False,
                'query_type': 'next_step',
                'result': None,
                'response_text': '步驟追蹤系統未啟用。'
            }
        
        # 獲取當前步驟
        fsm_state = self.step_fsm.get_state()
        current_index = fsm_state['step_index']  # TSM index (0-8)
        current_display = step_index_to_user_display(current_index)  # 顯示給用戶的步驟編號
        logger.info(f"[DEBUG] 下一步查詢 - 當前: step_index={current_index}, user_display={current_display}")
        
        # 計算下一步 (TSM index)
        if current_index == 8:
            # 從 idle 開始 → 第一步
            next_index = 0
        elif current_index >= 7:
            # 最後一步 → 完成
            next_index = None
        else:
            # 正常流程：下一步
            next_index = current_index + 1
        
        # 生成回應
        if next_index is None:
            response = "恭喜！所有步驟已完成。"
        elif next_index is not None:
            next_name = get_step_name_zh(next_index)
            next_display = step_index_to_user_display(next_index)
            logger.info(f"[DEBUG] 下一步計算: next_index={next_index}, next_display={next_display}, next_name={next_name}")
            response = f"下一步是{next_name}"
            
            # 如果有 SOP，嘗試添加更詳細的信息
            if self.sop_data:
                next_class = get_step_class(next_index)
                
                # 收集所有 SOP 步驟
                all_steps = []
                for phase_name in ['phase_1', 'phase_2', 'phase_3']:
                    if phase_name in self.sop_data:
                        phase_steps = self.sop_data[phase_name].get('steps', [])
                        all_steps.extend(phase_steps)
                
                # 嘗試找到匹配的 SOP 步驟
                for sop_step in all_steps:
                    step_name = sop_step.get('name', '').lower()
                    keywords = next_class.replace('_', ' ').split()
                    if any(keyword in step_name for keyword in keywords):
                        if 'object' in sop_step:
                            obj_name = self._translate_class_to_zh(sop_step['object'])
                            response += f"\n需要的物品：{obj_name}"
                        elif 'objects' in sop_step:
                            obj_names = [self._translate_class_to_zh(obj) for obj in sop_step['objects']]
                            response += f"\n需要的物品：{', '.join(obj_names)}"
                        break
        else:
            response = "無法確定下一步。"
        
        return {
            'success': True,
            'query_type': 'next_step',
            'result': {
                'current_index': current_index,
                'current_display': current_display,
                'next_index': next_index
            },
            'response_text': response
        }
    
    def _handle_step_status_query(self, text: str) -> Dict:
        """處理步驟狀態查詢：我剛剛完成了嗎？"""
        if self.step_fsm is None:
            return {
                'success': False,
                'query_type': 'step_status',
                'result': None,
                'response_text': '步驟追蹤系統未啟用。'
            }
        
        # 獲取轉移歷史
        history = self.step_fsm.get_transition_history(last_n=3)
        
        if not history:
            return {
                'success': True,
                'query_type': 'step_status',
                'result': None,
                'response_text': '尚未開始任何步驟。'
            }
        
        # 最近的轉移
        last_transition = history[-1]
        timestamp, from_step, to_step, reason = last_transition
        
        import time
        time_ago = time.time() - timestamp
        
        if to_step > from_step:
            response = f"是的，您剛剛完成了步驟 {from_step}，{time_ago:.0f} 秒前進入步驟 {to_step}。"
        else:
            response = f"目前仍在步驟 {to_step}，尚未完成。"
        
        return {
            'success': True,
            'query_type': 'step_status',
            'result': {'last_transition': last_transition},
            'response_text': response
        }
    
    def _handle_missing_query(self, text: str) -> Dict:
        """處理遺漏查詢：我漏了什麼？"""
        if not self.get_world_state:
            return {
                'success': False,
                'query_type': 'missing',
                'result': None,
                'response_text': '世界狀態系統未啟用。'
            }
        
        try:
            world_state = self.get_world_state()
            assembly_progress = world_state.get('assembly_progress', {})
            
            # 檢查各項進度
            missing_items = []
            
            cylinders = assembly_progress.get('cylinders_inserted', 0)
            if cylinders < 4:
                missing_items.append(f"還有 {4-cylinders} 個圓柱未插入")
            
            screws_small = assembly_progress.get('screws_small_used', 0)
            if screws_small < 4:
                missing_items.append(f"還有 {4-screws_small} 個小螺絲未鎖")
            
            if not assembly_progress.get('bracket_ready', False):
                missing_items.append("支架尚未準備好")
            
            if not assembly_progress.get('pi_ready', False):
                missing_items.append("PI 工件尚未準備好")
            
            if not missing_items:
                response = "目前沒有遺漏的步驟，進度正常！"
            else:
                response = "遺漏項目：\n" + "\n".join([f"- {item}" for item in missing_items])
            
            return {
                'success': True,
                'query_type': 'missing',
                'result': {'missing_items': missing_items},
                'response_text': response
            }
        except Exception as e:
            return {
                'success': False,
                'query_type': 'missing',
                'result': None,
                'response_text': f'無法檢查遺漏項目: {str(e)}'
            }
    
    def _handle_spatial_relation_query(self, text: str) -> Dict:
        """處理物體間空間關係查詢：「螺絲在扳手的哪邊？」"""
        if not self.spatial_memory:
            return {
                'success': False,
                'query_type': 'spatial_relation',
                'result': None,
                'response_text': '空間關係查詢功能未啟用'
            }
        
        # 解析兩個物體名稱
        match = None
        for pattern in self.patterns['spatial_relation']:
            match = re.search(pattern, text)
            if match:
                break
        
        if not match or len(match.groups()) < 2:
            return {
                'success': False,
                'query_type': 'spatial_relation',
                'result': None,
                'response_text': '無法識別查詢的物體。請嘗試：「螺絲在扳手的哪邊？」'
            }
        
        # 轉換物體名稱
        obj_a_zh = match.group(1).strip()
        obj_b_zh = match.group(2).strip()
        
        obj_a = self._translate_object_name(obj_a_zh)
        obj_b = self._translate_object_name(obj_b_zh)
        
        # 查詢空間關係
        result = self.spatial_memory.query_spatial_relation(obj_a, obj_b)
        
        if not result or not result.get('success'):
            return {
                'success': False,
                'query_type': 'spatial_relation',
                'result': None,
                'response_text': f'找不到 {obj_a_zh} 或 {obj_b_zh} 的位置資訊'
            }
        
        return {
            'success': True,
            'query_type': 'spatial_relation',
            'result': result,
            'response_text': result.get('description', f'{obj_a_zh} 和 {obj_b_zh} 的空間關係')
        }
    
    def _handle_nearby_query(self, text: str) -> Dict:
        """處理鄰近物體查詢：「扳手附近有什麼？」"""
        if not self.spatial_memory:
            return {
                'success': False,
                'query_type': 'nearby',
                'result': None,
                'response_text': '空間關係查詢功能未啟用'
            }
        
        # 解析中心物體名稱
        match = None
        for pattern in self.patterns['nearby']:
            match = re.search(pattern, text)
            if match:
                break
        
        if not match:
            return {
                'success': False,
                'query_type': 'nearby',
                'result': None,
                'response_text': '無法識別查詢的物體。請嘗試：「扳手附近有什麼？」'
            }
        
        # 轉換物體名稱
        obj_zh = match.group(1).strip()
        obj_id = self._translate_object_name(obj_zh)
        
        # 查詢鄰近物體（200mm 範圍）
        nearby = self.spatial_memory.get_objects_near(obj_id, radius=200)
        
        if not nearby:
            return {
                'success': True,
                'query_type': 'nearby',
                'result': [],
                'response_text': f'{obj_zh} 附近沒有其他物體（200mm 範圍內）'
            }
        
        # 生成回應文本
        items = []
        for name, distance in nearby:
            name_zh = self._translate_object_name_to_zh(name)
            items.append(f'{name_zh}（距離 {distance:.0f} mm）')
        
        response = f'{obj_zh} 附近有：{", ".join(items)}'
        
        return {
            'success': True,
            'query_type': 'nearby',
            'result': nearby,
            'response_text': response
        }
    
    def _handle_step_materials_query(self, text: str) -> Dict:
        """處理步驟所需零件查詢：「第三步需要哪些零件？」或「下一步需要什麼？」"""
        if not self.sop_data:
            return {
                'success': False,
                'query_type': 'step_materials',
                'result': None,
                'response_text': 'SOP 資料未載入，無法查詢步驟材料。'
            }
        
        # 解析步驟編號
        step_num = None
        
        # 檢查是否為「下一步」查詢
        if re.search(r'下一步|接下來', text):
            # 獲取當前步驟
            if self.step_fsm:
                fsm_state = self.step_fsm.get_state()
                current_step = fsm_state['confirmed_step']
                step_num = current_step + 1  # 下一步
            else:
                return {
                    'success': False,
                    'query_type': 'step_materials',
                    'result': None,
                    'response_text': '無法確定當前步驟，請明確指定步驟編號。'
                }
        else:
            # 解析指定的步驟編號
            # 支援格式：「第X步」、「第X部」(語音誤辨)、「步驟X」、「X步」
            match = re.search(
                r'(?:第|step)?\s*([\d一二三四五六七八九十百]+)\s*(?:步|部)', text
            )
            if not match:
                # 再試「步驟X」
                match = re.search(r'步驟\s*([\d一二三四五六七八九十百]+)', text)
            
            if match:
                num_str = match.group(1).strip()
                # 使用統一的中文數字轉換函數
                step_num = self._parse_chinese_number(num_str)
        
        if step_num is None:
            return {
                'success': False,
                'query_type': 'step_materials',
                'result': None,
                'response_text': '無法識別步驟編號。請嘗試：「第三步需要哪些零件？」'
            }
        
        # 從 SOP 獲取步驟信息（跨所有階段查找）
        target_step = None
        for phase_name in ['phase_1', 'phase_2', 'phase_3']:
            phase_data = self.sop_data.get(phase_name, {})
            steps = phase_data.get('steps', [])
            
            for step in steps:
                # 直接使用整數 id (1-9)
                step_id = step.get('id')
                if isinstance(step_id, int) and step_id == step_num:
                    target_step = step
                    break
            
            if target_step:
                break
        
        if not target_step:
            return {
                'success': False,
                'query_type': 'step_materials',
                'result': None,
                'response_text': f'找不到步驟 {step_num} 的資訊。'
            }
        
        # 提取該步驟需要的物件
        step_description = target_step.get('description', '無描述')
        step_object = target_step.get('object', None)
        
        # 構建回應
        materials = []
        if step_object:
            obj_zh = self._translate_object_name_to_zh(step_object)
            materials.append(obj_zh)
        
        # 檢查是否需要工具
        tools = target_step.get('tools', [])
        for tool in tools:
            tool_zh = self._translate_object_name_to_zh(tool)
            materials.append(f'{tool_zh}（工具）')
        
        if materials:
            materials_str = '、'.join(materials)
            response = f'步驟 {step_num}（{step_description}）需要：{materials_str}'
        else:
            response = f'步驟 {step_num}（{step_description}）不需要額外材料。'
        
        return {
            'success': True,
            'query_type': 'step_materials',
            'result': {
                'step_num': step_num,
                'description': step_description,
                'materials': materials
            },
            'response_text': response
        }
    
    def _translate_object_name(self, zh_name: str) -> str:
        """將中文物體名稱轉換為英文 ID"""
        return self.object_names_zh.get(zh_name, zh_name)
    
    def _translate_object_name_to_zh(self, en_name: str) -> str:
        """將英文 ID 轉換為中文名稱（使用標準全名）"""
        # 使用標準的完整名稱映射，避免字典反向映射時的覆蓋問題
        standard_names = {
            'hex_wrench': '六角扳手',
            'screwdriver': '螺絲起子',
            'platform': '平台',
            'bracket': '支架',
            'pi_shaped_part': 'PI工件',
            'cylinder': '圓柱',
            'screw': '螺絲',
            'screw_big': '大螺絲',
            'screw_small': '小螺絲',
            'nut': '螺帽',
        }
        return standard_names.get(en_name, en_name)
    
    def _parse_chinese_number(self, num_str: str) -> Optional[int]:
        """將中文數字或阿拉伯數字字串轉換為整數
        
        Args:
            num_str: 數字字串，例如 "三", "5", "九"
            
        Returns:
            轉換後的整數，失敗則返回 None
        """
        num_str = num_str.strip()
        
        # 中文數字映射表
        chinese_nums = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '十一': 11
        }
        
        if num_str in chinese_nums:
            return chinese_nums[num_str]
        
        try:
            return int(num_str)
        except:
            return None
    
    def _handle_update_step_query(self, text: str) -> Dict:
        """處理語音更新步驟查詢
        
        範例："更新步驟到第三步", "跳到第五步", "我們目前在步驟三"
        """
        if not self.step_fsm:
            return {
                'success': False,
                'query_type': 'update_step',
                'result': None,
                'response_text': '步驟管理系統未啟用。'
            }
        
        # 提取步驟編號
        step_num = None
        
        # 匹配「第X步」或「步驟X」
        match = re.search(r'第(.+)步', text)
        if not match:
            match = re.search(r'步驟(.+)', text)
        
        if match:
            num_text = match.group(1)
            step_num = self._parse_chinese_number(num_text)
        
        if step_num is None:
            return {
                'success': False,
                'query_type': 'update_step',
                'result': None,
                'response_text': '無法識別步驟編號。請嘗試：「更新步驟到第三步」'
            }
        
        # 驗證步驟範圍 (1-8 為組裝步驟，0 或 9 為 idle)
        if step_num < 0 or step_num > 9:
            return {
                'success': False,
                'query_type': 'update_step',
                'result': None,
                'response_text': f'步驟編號 {step_num} 超出範圍（1-8為組裝步驟，0或9為待機）。'
            }
        
        # 將用戶步驟編號轉換為 TSM index
        step_index = user_input_to_step_index(step_num)
        logger.info(f"[DEBUG] 步驟更新: user_input={step_num} → step_index={step_index}")
        
        # 更新步驟（手動覆蓋FSM，繞過轉換檢查）
        try:
            # 獲取舊步驟信息
            old_state = self.step_fsm.get_state()
            old_step_index = old_state.get('step_index', 8)  # TSM index
            logger.info(f"[DEBUG] 舊狀態: step_index={old_step_index}, step_class={old_state.get('step_class')}")
            
            # 執行更新
            self.step_fsm.manual_override_step(step_index)

            # 同步 WorldState.step_class，確保 world_state_builder 也反映新步驟
            # （progress_tracker 和 WorldStateBuilder.current_state 是獨立物件）
            _synced_state = self.step_fsm.get_state()
            _new_class = _synced_state.get("step_class", "idle")
            if callable(self.update_world_state_step):
                try:
                    self.update_world_state_step(_new_class, 1.0)
                except Exception:
                    pass

            # 驗證更新後的狀態
            new_state = self.step_fsm.get_state()
            logger.info(f"[DEBUG] 新狀態: step_index={new_state.get('step_index')}, user_display={new_state.get('user_display')}")
            
            # 獲取步驟名稱
            new_name = get_step_name_zh(step_index)
            old_name = get_step_name_zh(old_step_index)
            
            return {
                'success': True,
                'query_type': 'update_step',
                'result': {
                    'old_step_index': old_step_index,
                    'new_step_index': step_index
                },
                'response_text': f'已從{old_name}更新到{new_name}。'
            }
        except Exception as e:
            return {
                'success': False,
                'query_type': 'update_step',
                'result': None,
                'response_text': f'更新步驟時出錯：{str(e)}'
            }
    
    def _handle_deliver_materials_query(self, text: str) -> Dict:
        """處理主動零件遞送查詢
        
        範例："請給我第三步的零件", "幫我拿第五步需要的材料"
        
        這個函數會：
        1. 識別目標步驟
        2. 查詢該步驟需要的材料
        3. 生成 pickup + handover 組合動作
        """
        # 提取步驟編號
        step_num = None
        
        # 匹配「第X步」
        match = re.search(r'第(.+)步', text)
        if match:
            num_text = match.group(1)
            step_num = self._parse_chinese_number(num_text)
        
        if step_num is None:
            return {
                'success': False,
                'query_type': 'deliver_materials',
                'result': None,
                'response_text': '無法識別步驟編號。請嘗試：「請給我第三步的零件」'
            }
        
        # 從 SOP 獲取步驟材料信息
        if not self.sop_data:
            return {
                'success': False,
                'query_type': 'deliver_materials',
                'result': None,
                'response_text': 'SOP 資料未載入。'
            }
        
        # 跨所有階段查找目標步驟
        target_step = None
        for phase_name in ['phase_1', 'phase_2', 'phase_3']:
            phase_data = self.sop_data.get(phase_name, {})
            steps = phase_data.get('steps', [])
            
            for step in steps:
                step_id = step.get('id')
                if isinstance(step_id, int) and step_id == step_num:
                    target_step = step
                    break
            
            if target_step:
                break
        
        if not target_step:
            return {
                'success': False,
                'query_type': 'deliver_materials',
                'result': None,
                'response_text': f'找不到步驟 {step_num} 的資訊。'
            }
        
        # 提取材料清單
        materials = []
        step_object = target_step.get('object', None)
        if step_object:
            materials.append(step_object)
        
        # 檢查是否有多個物件（objects 陣列）
        step_objects = target_step.get('objects', [])
        materials.extend(step_objects)
        
        # 檢查工具
        tools = target_step.get('tools', [])
        materials.extend(tools)
        
        if not materials:
            return {
                'success': False,
                'query_type': 'deliver_materials',
                'result': None,
                'response_text': f'步驟 {step_num} 不需要額外材料。'
            }
        
        # 生成動作序列：pickup + handover 組合
        actions = []
        for material in materials:
            material_zh = self._translate_object_name_to_zh(material)
            actions.append({
                'type': 'pickup_and_handover',
                'object': material,
                'object_zh': material_zh
            })
        
        materials_zh = [self._translate_object_name_to_zh(m) for m in materials]
        materials_str = '、'.join(materials_zh)
        
        return {
            'success': True,
            'query_type': 'deliver_materials',
            'result': {
                'step_num': step_num,
                'materials': materials,
                'actions': actions
            },
            'response_text': f'好的，我會為您準備步驟 {step_num} 需要的材料：{materials_str}。'
        }
