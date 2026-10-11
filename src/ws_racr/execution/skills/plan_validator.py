"""
計畫驗證器
在執行前檢查計畫的安全性和可行性
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PlanValidator:
    """
    計畫驗證器
    檢查速度、力矩、Z高度、禁入區、IO序列等
    """
    
    def __init__(self):
        # 安全參數
        self.MAX_VELOCITY = 200  # %
        self.MIN_VELOCITY = 1    # %
        self.MAX_ACCELERATION = 200  # %
        self.MIN_Z_HEIGHT = 0.0  # mm (工作台面)
        self.MAX_Z_HEIGHT = 1000.0  # mm (安全上限)
        self.SAFE_Z_THRESHOLD = 150.0  # mm (安全高度閾值)
        
        # 禁入區 (x_min, x_max, y_min, y_max, z_min, z_max)
        self.FORBIDDEN_ZONES = [
            # 例如：相機區域
            # (-100, 100, -100, 100, 500, 1000),
        ]
        
        # IO 序列規則
        self.IO_RULES = {
            'must_open_before_pickup': True,  # 抓取前必須先開夾爪
            'must_close_after_pickup': True,  # 抓取後必須關閉夾爪
            'must_open_before_place': False,  # 放置前不一定要開（可能已經開了）
        }
        
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate(self, plan_json: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        驗證計畫
        
        Args:
            plan_json: 計畫 JSON
                {
                    "waypoints": [...],
                    "metadata": {...},
                    "constraints": {...}
                }
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        waypoints = plan_json.get('waypoints', [])
        
        if not waypoints:
            self.validation_errors.append("計畫中沒有路徑點")
            return False, self.validation_errors, self.validation_warnings
        
        logger.info(f"驗證計畫: {len(waypoints)} 個路徑點")
        
        # 1. 檢查速度和加速度
        self._validate_velocity_acceleration(waypoints)
        
        # 2. 檢查 Z 高度
        self._validate_z_height(waypoints)
        
        # 3. 檢查禁入區
        self._validate_forbidden_zones(waypoints)
        
        # 4. 檢查 IO 序列
        self._validate_io_sequence(waypoints)
        
        # 5. 檢查路徑連續性
        self._validate_path_continuity(waypoints)
        
        # 判斷是否有錯誤
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info(f"計畫驗證通過")
            if self.validation_warnings:
                logger.warning(f" {len(self.validation_warnings)} 個警告:")
                for warn in self.validation_warnings:
                    logger.info(f"   - {warn}")
        else:
            logger.error(f"計畫驗證失敗: {len(self.validation_errors)} 個錯誤")
            for err in self.validation_errors:
                logger.info(f"   - {err}")
        
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_velocity_acceleration(self, waypoints: List[Dict]):
        """檢查速度和加速度"""
        for i, wp in enumerate(waypoints):
            if wp.get('action') not in ['move', 'move_to']:
                continue
            
            velocity = wp.get('velocity', 100)
            acceleration = wp.get('acceleration', 200)
            
            # 檢查速度
            if velocity < self.MIN_VELOCITY:
                self.validation_errors.append(
                    f"路徑點 {i+1}: 速度過低 ({velocity}% < {self.MIN_VELOCITY}%)"
                )
            elif velocity > self.MAX_VELOCITY:
                self.validation_errors.append(
                    f"路徑點 {i+1}: 速度過高 ({velocity}% > {self.MAX_VELOCITY}%)"
                )
            
            # 檢查加速度
            if acceleration > self.MAX_ACCELERATION:
                self.validation_errors.append(
                    f"路徑點 {i+1}: 加速度過高 ({acceleration}% > {self.MAX_ACCELERATION}%)"
                )
            
            # 警告：高速移動
            if velocity > 80:
                self.validation_warnings.append(
                    f"路徑點 {i+1}: 高速移動 ({velocity}%)，請確認安全"
                )
    
    def _validate_z_height(self, waypoints: List[Dict]):
        """檢查 Z 高度"""
        for i, wp in enumerate(waypoints):
            if wp.get('action') not in ['move', 'move_to']:
                continue
            
            position = wp.get('position')
            if not position or len(position) < 3:
                continue
            
            z = position[2]
            
            # 檢查最小高度
            if z < self.MIN_Z_HEIGHT:
                self.validation_errors.append(
                    f"路徑點 {i+1}: Z 高度過低 ({z:.1f} mm < {self.MIN_Z_HEIGHT} mm)"
                )
            
            # 檢查最大高度
            if z > self.MAX_Z_HEIGHT:
                self.validation_errors.append(
                    f"路徑點 {i+1}: Z 高度過高 ({z:.1f} mm > {self.MAX_Z_HEIGHT} mm)"
                )
            
            # 警告：低於安全高度
            if z < self.SAFE_Z_THRESHOLD:
                self.validation_warnings.append(
                    f"路徑點 {i+1}: Z 高度較低 ({z:.1f} mm)，請確認工作台無障礙物"
                )
    
    def _validate_forbidden_zones(self, waypoints: List[Dict]):
        """檢查禁入區"""
        if not self.FORBIDDEN_ZONES:
            return
        
        for i, wp in enumerate(waypoints):
            if wp.get('action') not in ['move', 'move_to']:
                continue
            
            position = wp.get('position')
            if not position or len(position) < 3:
                continue
            
            x, y, z = position[0], position[1], position[2]
            
            for zone_idx, zone in enumerate(self.FORBIDDEN_ZONES):
                x_min, x_max, y_min, y_max, z_min, z_max = zone
                
                if (x_min <= x <= x_max and 
                    y_min <= y <= y_max and 
                    z_min <= z <= z_max):
                    self.validation_errors.append(
                        f"路徑點 {i+1}: 位於禁入區 #{zone_idx+1} "
                        f"({x:.1f}, {y:.1f}, {z:.1f})"
                    )
    
    def _validate_io_sequence(self, waypoints: List[Dict]):
        """檢查 IO 序列（夾爪開關順序）"""
        gripper_states = []  # 記錄夾爪狀態序列
        move_after_gripper = []  # 記錄夾爪動作後是否有移動
        
        for i, wp in enumerate(waypoints):
            action = wp.get('action')
            
            if action == 'gripper_open':
                gripper_states.append(('open', i))
            elif action == 'gripper_close':
                gripper_states.append(('close', i))
            elif action in ['move', 'move_to']:
                if gripper_states:
                    move_after_gripper.append(gripper_states[-1])
        
        # 檢查規則 1: 抓取前必須先開夾爪
        if self.IO_RULES['must_open_before_pickup']:
            _PICKUP_ACTIONS = {
                'grasp', 'pickup', 'move_to_pickup', 'move_to_grasp_position',
                'descend_to_pickup', 'approach'
            }
            has_pickup = any(wp.get('action', '') in _PICKUP_ACTIONS for wp in waypoints)
            
            if has_pickup and not any(state == 'open' for state, _ in gripper_states):
                self.validation_errors.append(
                    "抓取任務缺少開啟夾爪動作"
                )
        
        # 檢查規則 2: 連續相同動作
        for i in range(len(gripper_states) - 1):
            if gripper_states[i][0] == gripper_states[i+1][0]:
                self.validation_warnings.append(
                    f"連續執行相同夾爪動作: {gripper_states[i][0]} "
                    f"(路徑點 {gripper_states[i][1]+1} 和 {gripper_states[i+1][1]+1})"
                )
        
        # 檢查規則 3: 夾爪動作後應該有移動
        for state, idx in gripper_states:
            # 檢查此夾爪動作後是否有移動
            has_move_after = False
            for j in range(idx + 1, len(waypoints)):
                if waypoints[j].get('action') in ['move', 'move_to']:
                    has_move_after = True
                    break
                elif waypoints[j].get('action') in ['gripper_open', 'gripper_close']:
                    break
            
            if not has_move_after and idx < len(waypoints) - 1:
                self.validation_warnings.append(
                    f"路徑點 {idx+1}: {state} 夾爪後沒有移動動作"
                )
    
    def _validate_path_continuity(self, waypoints: List[Dict]):
        """檢查路徑連續性（避免大幅跳躍）"""
        MAX_POSITION_JUMP = 500.0  # mm
        
        prev_position = None
        
        for i, wp in enumerate(waypoints):
            if wp.get('action') not in ['move', 'move_to']:
                continue
            
            position = wp.get('position')
            if not position or len(position) < 3:
                continue
            
            if prev_position is not None:
                # 計算距離
                dx = position[0] - prev_position[0]
                dy = position[1] - prev_position[1]
                dz = position[2] - prev_position[2]
                distance = (dx**2 + dy**2 + dz**2) ** 0.5
                
                if distance > MAX_POSITION_JUMP:
                    self.validation_warnings.append(
                        f"路徑點 {i} 到 {i+1}: 位移過大 ({distance:.1f} mm > {MAX_POSITION_JUMP} mm)"
                    )
            
            prev_position = position


# 測試範例
if __name__ == "__main__":
    validator = PlanValidator()
    
    # 測試計畫
    test_plan = {
        "waypoints": [
            {"action": "gripper_open", "description": "Open gripper"},
            {"action": "move", "position": [400, 200, 500, 180, 0, 90], "velocity": 100},
            {"action": "move", "position": [400, 200, 300, 180, 0, 90], "velocity": 50},
            {"action": "move", "position": [400, 200, 100, 180, 0, 90], "velocity": 30},
            {"action": "gripper_close", "description": "Grasp object"},
            {"action": "move", "position": [400, 200, 500, 180, 0, 90], "velocity": 50},
        ],
        "metadata": {
            "planner": "moveit",
            "task_type": "pickup"
        }
    }
    
    is_valid, errors, warnings = validator.validate(test_plan)
    
    print(f"\n驗證結果: {'✅ 通過' if is_valid else '❌ 失敗'}")
    print(f"錯誤: {len(errors)}, 警告: {len(warnings)}")
