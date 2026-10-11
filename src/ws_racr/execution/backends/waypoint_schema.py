"""
統一的 Waypoint Schema
支援 MoveIt2、TMScript 和 SetPositions 三種執行後端
"""
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, field, asdict
import json

# 動作類型定義
ActionType = Literal[
    "move",                  # 移動到位置
    "move_to_safe_position", # 移動到安全位置
    "move_to_pickup",        # 移動到抓取準備位置
    "lift_object",           # 舉起物件
    "gripper_open",          # 開夾爪
    "gripper_close",         # 關夾爪
    "grasp",                 # 抓取
    "release",               # 釋放
    "wait",                  # 等待
    "wait_for_human"         # 等待人類
]

@dataclass
class WaypointConstraints:
    """路徑點約束條件 (用於 MoveIt2)"""
    position_tolerance: float = 0.01  # 位置容差 (m)
    orientation_tolerance: float = 0.01  # 姿態容差 (rad)
    joint_tolerance: float = 0.01  # 關節容差 (rad)
    avoid_collisions: bool = True  # 避障
    
    # 工作空間限制
    workspace_min: Optional[List[float]] = None  # [x, y, z] (m)
    workspace_max: Optional[List[float]] = None  # [x, y, z] (m)
    
    # 速度/加速度限制 (MoveIt2 用)
    max_velocity_scaling: float = 1.0  # 最大速度縮放 0.0-1.0
    max_acceleration_scaling: float = 1.0  # 最大加速度縮放 0.0-1.0
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return asdict(self)

@dataclass
class Waypoint:
    """
    統一的路徑點格式
    支援三種執行後端：MoveIt2、TMScript、SetPositions
    """
    # 基本資訊
    action: ActionType  # 動作類型
    position: List[float]  # [x, y, z, rx, ry, rz] (mm, deg)
    
    # 運動參數 (三種後端都支援)
    velocity: int = 100  # 速度百分比 1-100% (TM) 或轉換為 m/s (MoveIt2)
    acc_time: int = 200  # 加速時間 ms (TM) 或轉換為 acc (MoveIt2)
    blend: int = 0  # 路徑混合 0-100% (TMScript 專用，MoveIt2 忽略)
    
    # 時間參數
    wait_time: float = 0.0  # 等待時間 (秒)
    timeout: float = 30.0  # 超時時間 (秒)
    
    # MoveIt2 專用參數
    constraints: Optional[WaypointConstraints] = None  # 約束條件
    planner_id: str = "RRTConnect"  # 規劃器 ID
    planning_time: float = 5.0  # 規劃時間限制 (秒)
    
    # IO 同步 (TMScript/SetIO 用)
    io_before: Optional[Dict] = None  # 動作前的 IO 設定
    io_after: Optional[Dict] = None   # 動作後的 IO 設定
    
    # 其他屬性
    description: str = ""  # 描述
    metadata: Dict = field(default_factory=dict)  # 額外資訊
    
    def __post_init__(self):
        """初始化後處理"""
        if self.constraints is None:
            self.constraints = WaypointConstraints()
    
    def to_dict(self) -> dict:
        """轉換為字典 (用於序列化)"""
        data = asdict(self)
        # 處理 Optional 欄位
        if self.constraints:
            data['constraints'] = self.constraints.to_dict()
        return data
    
    def to_json(self) -> str:
        """轉換為 JSON 字串"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Waypoint':
        """從字典建立 Waypoint"""
        # 處理 constraints
        if 'constraints' in data and data['constraints']:
            data['constraints'] = WaypointConstraints(**data['constraints'])
        return cls(**data)
    
    @classmethod
    def from_legacy_waypoint(cls, legacy: dict) -> 'Waypoint':
        """
        從舊格式路徑點轉換
        
        Args:
            legacy: 舊格式 {"action": "move", "position": [...], ...}
        """
        return cls(
            action=legacy.get("action", "move"),
            position=legacy.get("position", [0, 0, 0, 0, 0, 0]),
            velocity=legacy.get("velocity", 100),
            acc_time=legacy.get("acc_time", 200),
            blend=legacy.get("blend", 0),
            wait_time=legacy.get("wait_time", 0.0),
            timeout=legacy.get("timeout", 30.0),
            description=legacy.get("description", "")
        )
    
    def get_tm_velocity_percent(self) -> int:
        """獲取 TM Robot 速度百分比 (1-100)"""
        return max(1, min(100, self.velocity))
    
    def get_tm_acc_time_ms(self) -> int:
        """獲取 TM Robot 加速時間 (1-2000 ms)"""
        return max(1, min(2000, self.acc_time))
    
    def get_tm_blend_percent(self) -> int:
        """獲取 TM Robot 混合百分比 (0-100)"""
        return max(0, min(100, self.blend))
    
    def get_moveit_velocity_scaling(self) -> float:
        """獲取 MoveIt2 速度縮放 (0.0-1.0)"""
        return self.velocity / 100.0
    
    def get_moveit_acceleration_scaling(self) -> float:
        """
        獲取 MoveIt2 加速度縮放 (0.0-1.0)
        根據 acc_time 轉換：較短的 acc_time = 較高的加速度
        """
        # acc_time: 1-2000ms -> scaling: 1.0-0.1
        # 使用反比關係：短時間 = 高加速度
        normalized = (2000 - self.acc_time) / 2000.0  # 0.0-1.0
        return max(0.1, min(1.0, normalized))
    
    def get_position_meters(self) -> List[float]:
        """獲取位置 (轉換為 meters for MoveIt2)"""
        pos_mm = self.position[:3]
        return [p / 1000.0 for p in pos_mm]  # mm -> m
    
    def get_orientation_radians(self) -> List[float]:
        """獲取姿態 (轉換為 radians for MoveIt2)"""
        import math
        ori_deg = self.position[3:6]
        return [math.radians(o) for o in ori_deg]  # deg -> rad
    
    def copy(self) -> 'Waypoint':
        """深拷貝"""
        import copy
        return copy.deepcopy(self)


class WaypointBuilder:
    """Waypoint 建構器 - 簡化建立過程"""
    
    @staticmethod
    def move_to(
        position: List[float],
        velocity: int = 100,
        acc_time: int = 200,
        blend: int = 0,
        description: str = ""
    ) -> Waypoint:
        """建立移動路徑點"""
        return Waypoint(
            action="move",
            position=position,
            velocity=velocity,
            acc_time=acc_time,
            blend=blend,
            description=description
        )
    
    @staticmethod
    def gripper_open(wait_time: float = 1.0) -> Waypoint:
        """建立開夾爪路徑點"""
        return Waypoint(
            action="gripper_open",
            position=[0, 0, 0, 0, 0, 0],  # 不需要位置
            wait_time=wait_time,
            description="Open gripper"
        )
    
    @staticmethod
    def gripper_close(wait_time: float = 1.5) -> Waypoint:
        """建立關夾爪路徑點"""
        return Waypoint(
            action="gripper_close",
            position=[0, 0, 0, 0, 0, 0],  # 不需要位置
            wait_time=wait_time,
            description="Close gripper"
        )
    
    @staticmethod
    def wait(duration: float, description: str = "") -> Waypoint:
        """建立等待路徑點"""
        return Waypoint(
            action="wait",
            position=[0, 0, 0, 0, 0, 0],
            wait_time=duration,
            description=description or f"Wait {duration}s"
        )
    
    @staticmethod
    def from_legacy_list(legacy_waypoints: List[dict]) -> List[Waypoint]:
        """批次轉換舊格式路徑點"""
        return [Waypoint.from_legacy_waypoint(wp) for wp in legacy_waypoints]


# 測試範例
if __name__ == "__main__":
    # 建立路徑點
    wp1 = WaypointBuilder.move_to(
        position=[291, -225, 613, 180, 0, -90],
        velocity=100,
        acc_time=200,
        description="Move to safe position"
    )
    
    wp2 = WaypointBuilder.gripper_close(wait_time=1.5)
    
    print("=== Waypoint 1 ===")
    print(wp1.to_json())
    print(f"\nTM Velocity: {wp1.get_tm_velocity_percent()}%")
    print(f"MoveIt2 Velocity Scaling: {wp1.get_moveit_velocity_scaling()}")
    print(f"MoveIt2 Acceleration Scaling: {wp1.get_moveit_acceleration_scaling()}")
    
    print("\n=== Waypoint 2 ===")
    print(wp2.to_json())
