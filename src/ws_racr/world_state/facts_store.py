"""
Facts Store
Deterministic 事實存儲，用於 Critic 快速一致性檢查
"""
from typing import Optional, Dict, Set, Any
from dataclasses import dataclass, field
import time


@dataclass
class FactsStore:
    """
    事實存儲（in-memory）
    
    存儲「可信」的事實，用於：
    - Consistency checks（避免矛盾）
    - Precondition checks（技能可行性）
    - Safety checks（緊急狀態）
    
    真值來源優先級：
    1. Robot IO (DI signals, action results)
    2. Stable state (hysteresis 後)
    3. Vision (raw detection)
    """
    
    # === Core Facts (high confidence) ===
    holding_object: Optional[str] = None  # robot 當前握持的物體 ID
    human_holding: Optional[str] = None   # human 握持的物體（如果可推估）
    
    gripper_state: str = "unknown"  # "open", "closed", "unknown"
    
    safety_state: str = "OK"  # "OK", "STOP", "WARNING"
    handover_active: bool = False
    
    # === Object Facts (stable locations) ===
    object_locations: Dict[str, str] = field(default_factory=dict)  # {obj_id: location_stable}
    object_visibility: Dict[str, str] = field(default_factory=dict)  # {obj_id: visibility}
    
    # === Execution History ===
    completed_steps: Set[str] = field(default_factory=set)  # 已完成的 step_class
    last_skill: Optional[str] = None
    last_skill_success: Optional[bool] = None
    last_skill_time: float = 0.0
    
    # === Temporal Context (recent N steps) ===
    recent_actions: list = field(default_factory=list)  # 最近 N 個 actions
    max_recent_actions: int = 10
    
    def update_from_world_state(self, world_state):
        """從 WorldState 更新 facts（使用 stable fields）"""
        from ws_racr.world_state.state_representation import VisibilityState
        
        # 更新 safety
        if world_state.human.safety_stable:
            self.safety_state = world_state.human.safety_stable
        
        # 更新 handover
        self.handover_active = world_state.human.handover_ready
        
        # 更新物體 stable locations
        for obj_id, obj in world_state.objects.items():
            if obj.location_stable:
                self.object_locations[obj_id] = obj.location_stable
            self.object_visibility[obj_id] = obj.visibility
        
        # 更新 robot holding（從 robot status）
        if world_state.robot.holding_object_id:
            self.holding_object = world_state.robot.holding_object_id
        
        # 更新 gripper
        if world_state.robot.gripper_state:
            self.gripper_state = world_state.robot.gripper_state
    
    def update_from_skill_result(self, skill: str, success: bool, **kwargs):
        """從技能執行結果更新 facts"""
        self.last_skill = skill
        self.last_skill_success = success
        self.last_skill_time = time.time()
        
        # 記錄到 recent actions
        self.recent_actions.append({
            'skill': skill,
            'success': success,
            'time': self.last_skill_time,
            **kwargs
        })
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions.pop(0)
        
        # 根據技能結果更新 facts
        if skill == "pickup" and success:
            target = kwargs.get('target_object')
            if target:
                self.holding_object = target
                self.object_locations[target] = "in_gripper"
                self.gripper_state = "closed"
        
        elif skill == "place" and success:
            if self.holding_object:
                target_location = kwargs.get('target_location', 'unknown')
                self.object_locations[self.holding_object] = target_location
                self.holding_object = None
                self.gripper_state = "open"
        
        elif skill == "handover" and success:
            if self.holding_object:
                # 交接後，物體到人類手上
                self.human_holding = self.holding_object
                self.holding_object = None
                self.gripper_state = "open"
    
    def mark_step_completed(self, step_class: str):
        """標記 step 已完成"""
        self.completed_steps.add(step_class)
    
    def is_step_completed(self, step_class: str) -> bool:
        """檢查 step 是否已完成"""
        return step_class in self.completed_steps
    
    def get_object_location(self, obj_id: str) -> Optional[str]:
        """取得物體的穩定位置"""
        return self.object_locations.get(obj_id)
    
    def is_object_visible(self, obj_id: str) -> bool:
        """檢查物體是否可見"""
        return self.object_visibility.get(obj_id) == "visible"
    
    def is_object_missing(self, obj_id: str) -> bool:
        """檢查物體是否消失"""
        return self.object_visibility.get(obj_id) == "missing"
    
    def get_recent_failures(self, skill: Optional[str] = None, limit: int = 3) -> list:
        """取得最近的失敗記錄"""
        failures = [a for a in self.recent_actions if not a['success']]
        if skill:
            failures = [a for a in failures if a['skill'] == skill]
        return failures[-limit:]
    
    def reset(self):
        """重置所有 facts（新任務開始時）"""
        self.holding_object = None
        self.human_holding = None
        self.gripper_state = "unknown"
        self.safety_state = "OK"
        self.handover_active = False
        self.object_locations.clear()
        self.object_visibility.clear()
        self.completed_steps.clear()
        self.recent_actions.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典（用於 logging/debugging）"""
        return {
            'holding_object': self.holding_object,
            'human_holding': self.human_holding,
            'gripper_state': self.gripper_state,
            'safety_state': self.safety_state,
            'handover_active': self.handover_active,
            'object_locations': dict(self.object_locations),
            'object_visibility': dict(self.object_visibility),
            'completed_steps': list(self.completed_steps),
            'last_skill': self.last_skill,
            'last_skill_success': self.last_skill_success,
            'recent_actions_count': len(self.recent_actions)
        }
