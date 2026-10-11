"""Information preprocessor: turns a world-state dict into a StepSummary and a MemoryQuery for the memory stores."""

from typing import Dict, Tuple
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
SPATIAL_CALC_AVAILABLE = True
from ws_racr.memory.types import (
    HumanStateEnum,
    MemoryQuery,
    OutcomeEnum,
    SafetyStateEnum,
    StepSummary,
)


class InformationPreprocessor:
    """
    資訊預處理器
    
    將原始多相機感知轉換為:
    1. StepSummary (寫入導向)
    2. MemoryQuery (檢索導向)
    """
    
    def __init__(self):
        pass
    
    def process(self, world_state: Dict) -> Tuple[StepSummary, MemoryQuery]:
        """
        處理 WorldState 並產生 StepSummary 和 MemoryQuery
        
        Args:
            world_state: 來自 WorldStateBuilder 的狀態字典
        
        Returns:
            (step_summary, memory_query)
        """
        # 提取狀態資訊 (keys 匹配 WorldState.to_dict() / asdict(WorldState()) 輸出)
        human_state_str = world_state.get('human', {}).get('intent', 'unknown')
        current_step = world_state.get('step_class', 'unknown')
        safety_state_str = world_state.get('human', {}).get('safety', 'OK')
        
        # 提取物件 (objects 是 Dict[str, ObjectState], 不是 list)
        objects_involved = []
        for obj_data in world_state.get('objects', {}).values():
            objects_involved.append(obj_data.get('object_type', 'unknown'))
        
        # 判斷 handover intent
        handover_intent = human_state_str in ['receive', 'handover_ready']
        
        # 機器人最後動作（如果有的話）
        robot_last_action = world_state.get('robot', {}).get('current_action')
        
        # 創建 StepSummary
        step_summary = StepSummary(
            time=datetime.now(),
            human_state=self._map_human_state(human_state_str),
            current_step=current_step,
            objects_involved=objects_involved,
            handover_intent=handover_intent,
            safety_state=self._map_safety_state(safety_state_str),
            robot_last_action=robot_last_action,
            outcome=OutcomeEnum.UNKNOWN,  # 初始為 unknown
            detected_gesture=human_state_str
        )
        
        # 創建 MemoryQuery
        query_text = self._generate_query_text(world_state)
        
        memory_query = MemoryQuery(
            query_text=query_text,
            query_type="episodic",  # 預設查詢情節記憶
            context={
                'step': current_step,
                'human_state': human_state_str,
                'objects': objects_involved
            }
        )
        
        return step_summary, memory_query
    
    def _map_human_state(self, gesture: str) -> HumanStateEnum:
        """映射手勢到人類狀態"""
        gesture_lower = gesture.lower()
        
        if 'stop' in gesture_lower or 'halt' in gesture_lower:
            return HumanStateEnum.STOP
        elif 'receive' in gesture_lower or 'handover' in gesture_lower:
            return HumanStateEnum.HANDOVER_READY
        elif 'idle' in gesture_lower or 'none' in gesture_lower:
            return HumanStateEnum.IDLE
        elif 'working' in gesture_lower or 'assembly' in gesture_lower:
            return HumanStateEnum.WORKING
        else:
            return HumanStateEnum.UNKNOWN
    
    def _map_safety_state(self, status: str) -> SafetyStateEnum:
        """映射安全狀態"""
        status_lower = status.lower()
        
        if 'stop' in status_lower or 'danger' in status_lower:
            return SafetyStateEnum.STOP
        elif 'warning' in status_lower or 'caution' in status_lower:
            return SafetyStateEnum.WARNING
        else:
            return SafetyStateEnum.SAFE
    
    def _generate_query_text(self, world_state: Dict) -> str:
        """
        根據當前狀態生成記憶查詢文字
        
        這是 RoboMemory 的關鍵：查詢不是摘要，而是問題
        """
        current_step = world_state.get('step_class', 'unknown')
        human_gesture = world_state.get('human', {}).get('intent', 'unknown')
        
        # 根據不同情境生成不同的查詢
        if 'receive' in human_gesture.lower() or 'handover' in human_gesture.lower():
            query = f"In similar handover situations during {current_step}, when was robot assistance accepted?"
        elif 'stop' in human_gesture.lower():
            query = f"How should robot respond to stop signals during {current_step}?"
        elif human_gesture.lower() == 'idle':
            query = f"What actions are appropriate when human is idle during {current_step}?"
        else:
            query = f"What is the best robot action during {current_step} when human is {human_gesture}?"
        
        return query
