"""
Action Validator - 驗證 LLM 輸出的 action 是否合法

確保 LLM 生成的動作符合預定義的 action schema，
防止無效或危險的動作被執行。

Action ontology is defined in skill_schema.py — do not add new action names here.
"""
import logging
from typing import Dict, List, Optional, Tuple

from ws_racr.execution.skills.skill_schema import (
    ACTION_ALIASES,
    CANONICAL_ACTIONS,
    KNOWN_LOCATIONS,
    KNOWN_OBJECTS,
    SKILL_DEFINITIONS,
)

logger = logging.getLogger(__name__)


class ActionValidator:
    """動作驗證器「來源權」: skill_schema.py"""

    # Canonical set + legacy aliases are ALL accepted input forms.
    # Aliases will be rewritten to canonical before further checks.
    ALLOWED_ACTIONS = CANONICAL_ACTIONS | frozenset(ACTION_ALIASES.keys())

    # Synonym rewrite map (imported from skill_schema)
    ACTION_SYNONYMS: Dict[str, str] = ACTION_ALIASES

    # Parameter requirements (derived from SKILL_DEFINITIONS)
    REQUIRES_OBJECT:    frozenset = frozenset()
    REQUIRES_LOCATION:  frozenset = frozenset()
    REQUIRES_DIRECTION: frozenset = frozenset()
    REQUIRES_ROTATION:  frozenset = frozenset()

    def __init_subclass_requirements(self) -> None:
        pass  # populated at class body level below

    # Build requirement sets from SKILL_DEFINITIONS
    _req_obj  = set()
    _req_loc  = set()
    _req_dir  = set()
    _req_rot  = set()
    for _sd in SKILL_DEFINITIONS:
        _a = _sd["action"]
        if "target_object"       in _sd["required"]: _req_obj.add(_a)
        if "target_location"     in _sd["required"]: _req_loc.add(_a)
        if "direction"           in _sd["required"]: _req_dir.add(_a)
        if "rotation_direction"  in _sd["required"]: _req_rot.add(_a)
    REQUIRES_OBJECT    = frozenset(_req_obj)
    REQUIRES_LOCATION  = frozenset(_req_loc)
    REQUIRES_DIRECTION = frozenset(_req_dir)
    REQUIRES_ROTATION  = frozenset(_req_rot)
    del _sd, _a, _req_obj, _req_loc, _req_dir, _req_rot

    # 允許的位置名稱
    ALLOWED_LOCATIONS = set(KNOWN_LOCATIONS)

    # 允許的物體類型
    ALLOWED_OBJECTS = set(KNOWN_OBJECTS)
    
    def __init__(self):
        """初始化驗證器"""
        pass
    
    def validate(self, task_json: Dict) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        驗證任務 JSON
        
        Args:
            task_json: LLM 生成的任務 JSON
            
        Returns:
            (is_valid, error_message, corrected_json)
            - is_valid: 是否有效
            - error_message: 錯誤訊息（如果無效）
            - corrected_json: 修正後的 JSON（如果可修正）
        """
        action = task_json.get('action')

        # 1. 檢查 action 是否存在
        if not action:
            return False, "缺少 'action' 欄位", None

        # 2. Alias rewrite — always applied first, regardless of whether the alias
        #    itself is in ALLOWED_ACTIONS.  Ensures non-canonical names are
        #    normalised to canonical before parameter checks.
        rewritten = False
        canonical = self.ACTION_SYNONYMS.get(action) or self.ACTION_SYNONYMS.get(action.lower())
        if canonical:
            task_json = task_json.copy()
            task_json['action'] = canonical
            action = canonical
            rewritten = True

        # 3. 檢查 action 是否在允許列表中
        if action not in self.ALLOWED_ACTIONS:
            # fuzzy match as last resort
            corrected = self._fuzzy_match_action(action)
            if corrected:
                corrected_json = task_json.copy()
                corrected_json['action'] = corrected
                return True, None, corrected_json
            else:
                return False, f"不允許的動作: '{action}'。允許的動作: {sorted(CANONICAL_ACTIONS)}", None
        
        # 3. 檢查必要參數
        if action in self.REQUIRES_OBJECT:
            if 'target_object' not in task_json or not task_json['target_object']:
                return False, f"動作 '{action}' 需要 'target_object' 參數", None
        
        if action in self.REQUIRES_LOCATION:
            if 'target_location' not in task_json or not task_json['target_location']:
                return False, f"動作 '{action}' 需要 'target_location' 參數", None
        
        if action in self.REQUIRES_DIRECTION:
            # move_relative 需要 direction，distance 可選（預設 50mm）
            if 'direction' not in task_json or not task_json['direction']:
                return False, f"動作 '{action}' 需要 'direction' 參數（left/right/forward/backward/up/down）", None
            # 自動添加預設 distance
            if 'distance' not in task_json:
                corrected_json = task_json.copy()
                corrected_json['distance'] = 50  # 預設 50mm
                return True, None, corrected_json
        
        if action in self.REQUIRES_ROTATION:
            # rotate_gripper 需要 rotation_direction，rotation_angle 可選（預設 45 度）
            if 'rotation_direction' not in task_json or not task_json['rotation_direction']:
                return False, f"動作 '{action}' 需要 'rotation_direction' 參數（left/right/clockwise/counterclockwise）", None
            # 自動添加預設 rotation_angle
            if 'rotation_angle' not in task_json:
                corrected_json = task_json.copy()
                corrected_json['rotation_angle'] = 45  # 預設 45 度
                return True, None, corrected_json
        
        # 4. 驗證參數值（可選）
        target_location = task_json.get('target_location')
        if target_location and target_location not in self.ALLOWED_LOCATIONS:
            # 這裡只警告，不拒絕（因為可能是 LLM 語義理解的位置）
            logger.warning(f"未知位置: '{target_location}'，將嘗試語義解析")
        
        # 5. 檢查衝突參數
        if action in ['gripper_open', 'open_gripper'] and task_json.get('target_object'):
            # 打開夾爪不應該有 target_object
            corrected_json = task_json.copy()
            del corrected_json['target_object']
            return True, None, corrected_json
        
        # 全部通過 — return rewritten task_json if alias was applied
        return True, None, (task_json if rewritten else None)
    
    def _fuzzy_match_action(self, action: str) -> Optional[str]:
        """模糊匹配動作名稱（處理大小寫、空格等）"""
        action_lower = action.lower().replace(' ', '_').replace('-', '_')
        
        # 直接匹配
        for allowed in self.ALLOWED_ACTIONS:
            if allowed.lower() == action_lower:
                return allowed
        
        # 部分匹配 - 拾取動作
        if 'pick' in action_lower:
            return 'pickup'
        if 'grab' in action_lower or 'grasp' in action_lower:
            return 'pickup'
        
        # 放置動作
        if 'put' in action_lower or 'drop' in action_lower or 'release' in action_lower:
            return 'place'
        
        # 遞交動作
        if 'give' in action_lower or 'pass' in action_lower or 'hand' in action_lower:
            return 'handover'
        
        # 移動動作
        if 'go to' in action_lower or 'goto' in action_lower or 'navigate' in action_lower:
            return 'move'
        if action_lower.startswith('move to') or action_lower == 'move_to':
            return 'move'
        
        # 相對移動
        if 'adjust' in action_lower or 'nudge' in action_lower or 'shift' in action_lower:
            return 'shift'
        if 'move_relative' in action_lower or 'relative' in action_lower:
            return 'shift'
        
        # 旋轉動作
        if 'rotate' in action_lower or 'twist' in action_lower or 'turn' in action_lower:
            if 'grip' in action_lower:
                return 'rotate_gripper'
            return 'rotate_gripper'
        
        # 夾爪控制
        if 'open' in action_lower and 'grip' in action_lower:
            return 'gripper_open'
        if 'close' in action_lower and 'grip' in action_lower:
            return 'gripper_close'
        
        # 返回動作
        if 'back' in action_lower or 'return' in action_lower:
            if 'origin' in action_lower or 'home' in action_lower:
                return 'return_to_origin'
            return 'return_to_last'
        
        # 停止動作
        if 'emergency' in action_lower:
            return 'emergency_stop'
        if 'stop' in action_lower or 'halt' in action_lower:
            return 'stop'
        
        return None
    
    def validate_and_correct(self, task_json: Dict) -> Tuple[bool, Dict, Optional[str]]:
        """
        驗證並自動修正（如果可能）
        
        Args:
            task_json: 原始任務 JSON
            
        Returns:
            (is_valid, final_json, error_message)
        """
        is_valid, error_msg, corrected = self.validate(task_json)
        
        if is_valid and corrected:
            logger.info(f"動作已自動修正: {task_json.get('action')} → {corrected.get('action')}")
            return True, corrected, None
        elif is_valid:
            return True, task_json, None
        else:
            return False, task_json, error_msg


# 全域實例
_validator = None

def get_action_validator() -> ActionValidator:
    """獲取全域 ActionValidator 實例"""
    global _validator
    if _validator is None:
        _validator = ActionValidator()
    return _validator
