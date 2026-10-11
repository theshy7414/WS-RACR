"""
MoveIt2 規劃器整合
使用 MoveIt Task Constructor 生成抓取/放置語意路徑
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple
import math

from ws_racr.execution.backends.waypoint_schema import Waypoint, WaypointBuilder, WaypointConstraints

logger = logging.getLogger(__name__)

# 嘗試導入 MoveIt2
try:
    import rclpy
    from rclpy.node import Node
    from moveit_msgs.srv import GetPositionIK, GetCartesianPath
    from geometry_msgs.msg import Pose, PoseStamped
    from moveit_msgs.msg import RobotTrajectory, Constraints
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False
    logger.warning("MoveIt2 未安裝，MoveItPlanner 不可用")


class MoveItPlanner:
    """
    MoveIt2 規劃器
    使用 MoveIt2 進行路徑規劃，支援避障和約束
    """
    
    def __init__(self, group_name: str = "tm_arm", base_frame: str = "base_link"):
        self.group_name = group_name
        self.base_frame = base_frame
        self.node = None
        self.ik_client = None
        self.cartesian_path_client = None
        
        # 預設參數
        self.SAFE_HEIGHT = 0.2  # 安全高度 (m)
        self.APPROACH_HEIGHT = 0.05  # 接近高度 (m)
        
        # Origin 位置 [x, y, z, rx, ry, rz] (mm, deg)
        self.origin_position = [291, -225, 613, 180, 0, -90]
        
        if not MOVEIT_AVAILABLE:
            logger.error("MoveIt2 不可用，規劃功能將受限")
    
    def initialize(self) -> bool:
        """初始化 MoveIt2 規劃器"""
        if not MOVEIT_AVAILABLE:
            return False
        
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.node = Node('moveit_planner')
            
            # 創建 IK 客戶端
            self.ik_client = self.node.create_client(
                GetPositionIK,
                f'/compute_ik'
            )
            
            # 創建笛卡爾路徑客戶端
            self.cartesian_path_client = self.node.create_client(
                GetCartesianPath,
                f'/compute_cartesian_path'
            )
            
            # 等待服務
            logger.info("等待 MoveIt2 規劃服務...")
            if not self.ik_client.wait_for_service(timeout_sec=5.0):
                logger.error("IK 服務超時")
                return False
            
            if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
                logger.error("Cartesian Path 服務超時")
                return False
            
            logger.info("MoveIt2 Planner 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"MoveIt2 Planner 初始化失敗: {e}")
            return False
        
    def plan_pickup_task(self, 
                        object_position: List[float],
                        velocity: int = 100,
                        avoid_collisions: bool = True) -> List[Waypoint]:
        """
        規劃抓取任務 (使用 MoveIt2)
        
        Args:
            object_position: [x, y, z, rx, ry, rz] (mm, deg)
            velocity: 速度百分比
            avoid_collisions: 是否避障
        
        Returns:
            路徑點序列
        """
        logger.info(f"MoveIt2 規劃抓取任務...")
        
        # 轉換單位 mm -> m, deg -> rad
        pos_m = [object_position[i] / 1000.0 for i in range(3)]
        ori_rad = [math.radians(object_position[i]) for i in range(3, 6)]
        
        waypoints = []
        
        # 1. 打開夾爪（延長等待時間確保狀態重置）
        waypoints.append(WaypointBuilder.gripper_open(wait_time=2.0))  # 延長到 2 秒
        
        # 2. 移動到安全高度上方
        safe_pose = pos_m.copy()
        safe_pose[2] += self.SAFE_HEIGHT  # 向上 200mm
        
        wp_safe = self._create_waypoint_from_pose(
            safe_pose, ori_rad, velocity,
            description="Move to safe height above object",
            avoid_collisions=avoid_collisions
        )
        waypoints.append(wp_safe)
        
        # 3. 使用笛卡爾路徑下降到接近高度
        approach_pose = pos_m.copy()
        approach_pose[2] += self.APPROACH_HEIGHT  # 向上 50mm
        
        cartesian_waypoints = self._plan_cartesian_path(
            [safe_pose, approach_pose],
            ori_rad,
            velocity=velocity
        )
        
        if cartesian_waypoints:
            waypoints.extend(cartesian_waypoints)
        else:
            # 如果笛卡爾路徑失敗，使用單點移動
            wp_approach = self._create_waypoint_from_pose(
                approach_pose, ori_rad, velocity,
                description="Approach object",
                avoid_collisions=avoid_collisions
            )
            waypoints.append(wp_approach)
        
        # 4. 最後下降到抓取位置
        wp_grasp = self._create_waypoint_from_pose(
            pos_m, ori_rad, 50,  # 下降速度 50%
            description="Grasp position",
            avoid_collisions=False  # 最後一段不避障
        )
        waypoints.append(wp_grasp)
        
        # 5. 關閉夾爪
        waypoints.append(WaypointBuilder.gripper_close())
        
        # 6. 上升到安全高度
        wp_lift = self._create_waypoint_from_pose(
            safe_pose, ori_rad, velocity,
            description="Lift object",
            avoid_collisions=avoid_collisions
        )
        waypoints.append(wp_lift)
        
        # 7. 回到 origin 位置
        origin_pose = [self.origin_position[i] / 1000.0 for i in range(3)]  # mm -> m
        origin_ori = [math.radians(self.origin_position[i]) for i in range(3, 6)]  # deg -> rad
        
        wp_origin = self._create_waypoint_from_pose(
            origin_pose, origin_ori, velocity,
            description="Return to origin",
            avoid_collisions=avoid_collisions
        )
        waypoints.append(wp_origin)
        
        logger.info(f"MoveIt2 規劃完成: {len(waypoints)} 個路徑點")
        return waypoints
    
    def plan_place_task(self,
                       target_position: List[float],
                       velocity: int = 100,
                       avoid_collisions: bool = True) -> List[Waypoint]:
        """
        規劃放置任務 (使用 MoveIt2)
        
        Args:
            target_position: [x, y, z, rx, ry, rz] (mm, deg)
            velocity: 速度百分比
            avoid_collisions: 是否避障
        
        Returns:
            路徑點序列
        """
        logger.info(f"MoveIt2 規劃放置任務...")
        
        # 轉換單位
        pos_m = [target_position[i] / 1000.0 for i in range(3)]
        ori_rad = [math.radians(target_position[i]) for i in range(3, 6)]
        
        waypoints = []
        
        # 1. 移動到目標上方安全高度
        safe_pose = pos_m.copy()
        safe_pose[2] += self.SAFE_HEIGHT
        
        wp_safe = self._create_waypoint_from_pose(
            safe_pose, ori_rad, velocity,
            description="Move to safe height above target",
            avoid_collisions=avoid_collisions
        )
        waypoints.append(wp_safe)
        
        # 2. 下降到放置位置
        wp_place = self._create_waypoint_from_pose(
            pos_m, ori_rad, 50,
            description="Place position",
            avoid_collisions=False
        )
        waypoints.append(wp_place)
        
        # 3. 打開夾爪
        waypoints.append(WaypointBuilder.gripper_open())
        
        # 4. 上升到安全高度
        wp_retreat = self._create_waypoint_from_pose(
            safe_pose, ori_rad, velocity,
            description="Retreat from placed object",
            avoid_collisions=avoid_collisions
        )
        waypoints.append(wp_retreat)
        
        logger.info(f"MoveIt2 規劃完成: {len(waypoints)} 個路徑點")
        return waypoints
    
    def _create_waypoint_from_pose(self,
                                   position_m: List[float],
                                   orientation_rad: List[float],
                                   velocity: int,
                                   description: str = "",
                                   avoid_collisions: bool = True,
                                   acc_time: int = 200,
                                   blend: int = 0) -> Waypoint:
        """從姿態建立路徑點 - 包含所有必要參數"""
        # 轉換回 mm 和 deg
        pos_mm = [position_m[i] * 1000.0 for i in range(3)]
        ori_deg = [math.degrees(orientation_rad[i]) for i in range(3)]
        
        position = pos_mm + ori_deg
        
        # 建立約束
        constraints = WaypointConstraints()
        constraints.avoid_collisions = avoid_collisions
        constraints.max_velocity_scaling = velocity / 100.0
        constraints.max_acceleration_scaling = (2000 - acc_time) / 2000.0  # 根據 acc_time 計算
        
        return Waypoint(
            action="move",
            position=position,
            velocity=velocity,
            acc_time=acc_time,
            blend=blend,
            constraints=constraints,
            description=description,
            io_before=None,  # 確保這些欄位存在
            io_after=None,
            wait_time=0.0
        )
    
    def _plan_cartesian_path(self,
                            positions_m: List[List[float]],
                            orientation_rad: List[float],
                            velocity: int,
                            max_step: float = 0.01) -> List[Waypoint]:
        """
        規劃笛卡爾路徑
        
        Args:
            positions_m: 位置序列 [[x,y,z], ...] (m)
            orientation_rad: 姿態 [rx, ry, rz] (rad)
            velocity: 速度百分比
            max_step: 最大步長 (m)
        
        Returns:
            路徑點序列
        """
        if not self.cartesian_path_client:
            logger.warning("Cartesian Path 服務不可用")
            return []
        
        try:
            # 建立請求
            request = GetCartesianPath.Request()
            request.header.frame_id = self.base_frame
            request.group_name = self.group_name
            request.max_step = max_step
            request.jump_threshold = 0.0
            request.avoid_collisions = True
            
            # 歐拉角轉四元數（使用數學公式，不依賴 tf_transformations）
            def euler_to_quaternion(roll, pitch, yaw):
                """歐拉角 (XYZ) 轉四元數"""
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                cp = math.cos(pitch * 0.5)
                sp = math.sin(pitch * 0.5)
                cr = math.cos(roll * 0.5)
                sr = math.sin(roll * 0.5)
                
                w = cr * cp * cy + sr * sp * sy
                x = sr * cp * cy - cr * sp * sy
                y = cr * sp * cy + sr * cp * sy
                z = cr * cp * sy - sr * sp * cy
                
                return [x, y, z, w]
            
            # 建立路徑點
            for pos in positions_m:
                pose = PoseStamped()
                pose.header.frame_id = self.base_frame
                pose.pose.position.x = pos[0]
                pose.pose.position.y = pos[1]
                pose.pose.position.z = pos[2]
                
                q = euler_to_quaternion(
                    orientation_rad[0],
                    orientation_rad[1],
                    orientation_rad[2]
                )
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                
                request.waypoints.append(pose.pose)
            
            # 發送請求
            future = self.cartesian_path_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
            
            response = future.result()
            
            if response and response.fraction > 0.9:  # 90% 以上成功率
                logger.info(f"笛卡爾路徑規劃成功: {response.fraction * 100:.1f}%")
                
                # 將軌跡轉換為 Waypoint
                waypoints = self._trajectory_to_waypoints(
                    response.solution,
                    velocity
                )
                return waypoints
            else:
                logger.warning(f"笛卡爾路徑規劃失敗: {response.fraction * 100:.1f}%")
                return []
            
        except Exception as e:
            logger.error(f"笛卡爾路徑規劃錯誤: {e}")
            return []
    
    def _trajectory_to_waypoints(self,
                                trajectory: RobotTrajectory,
                                velocity: int) -> List[Waypoint]:
        """將 RobotTrajectory 轉換為 Waypoint 列表"""
        waypoints = []
        
        # 這裡需要根據你的機器人模型進行 FK 轉換
        # 暫時返回空列表，實際使用時需要實作
        logger.warning("軌跡轉 Waypoint 功能需要實作 FK")
        
        return waypoints
    
    def plan_handover_task(self,
                          handover_position: List[float],
                          velocity: int = 100,
                          wait_time: float = 1.0) -> List[Waypoint]:
        """
        規劃交接任務（對齊 receive 手勢參數）
        
        Args:
            handover_position: 交接位置 [x, y, z, rx, ry, rz] (mm, deg)
            velocity: 速度百分比
            wait_time: 等待時間（秒，默認 1 秒）
        """
        waypoints = []
        
        # 1. 移動到交接位置
        waypoints.append(WaypointBuilder.move_to(
            handover_position, velocity, description="Handover position"
        ))
        
        # 2. 釋放物體（打開夾爪）+ 等待人類拿取
        waypoints.append(Waypoint(
            action="release",
            position=handover_position,
            velocity=0,
            acc_time=0,
            blend=0,
            constraints=WaypointConstraints(),
            io_before=None,
            io_after={"EndModule": {"DO": {0: False}}},  # 打開夾爪
            wait_time=wait_time,  # 等待 1 秒後返回（與 receive 一致）
            description="Release object and wait"
        ))
        
        # 3. 返回待機位置（與 receive 一致）
        standby_position = [402, -134, 554, 180, 0, 90]  # 待機位置
        waypoints.append(WaypointBuilder.move_to(
            standby_position, 100, description="Return to standby position"
        ))
        
        return waypoints
    
    def plan_observe_task(self,
                         observe_position: List[float],
                         velocity: int = 100,
                         observe_time: float = 2.0) -> List[Waypoint]:
        """
        規劃觀察任務
        
        Args:
            observe_position: 觀察位置 [x, y, z, rx, ry, rz] (mm, deg)
            velocity: 速度百分比
            observe_time: 觀察時間（秒）
        """
        waypoints = []
        
        # 移動到觀察位置並停留
        waypoints.append(Waypoint(
            action="move",
            position=observe_position,
            velocity=velocity,
            acc_time=200,
            blend=0,
            constraints=WaypointConstraints(),
            io_before=None,
            io_after=None,
            wait_time=observe_time,
            description="Observe position"
        ))
        
        return waypoints
    
    def plan_prepare_task(self,
                         object_position: List[float],
                         handover_position: List[float],
                         velocity: int = 100) -> List[Waypoint]:
        """
        規劃準備任務：pickup + handover 組合
        
        Args:
            object_position: 物件位置 [x, y, z, rx, ry, rz] (mm, deg)
            handover_position: 交接位置 [x, y, z, rx, ry, rz] (mm, deg)
            velocity: 速度百分比
        """
        waypoints = []
        
        # 1. 執行 pickup 任務
        pickup_waypoints = self.plan_pickup_task(object_position, velocity)
        waypoints.extend(pickup_waypoints)
        
        # 2. 移動到交接位置
        waypoints.append(WaypointBuilder.move_to(
            handover_position, 80, description="Move to handover position"
        ))
        
        # 3. 等待人類拿取
        waypoints.append(Waypoint(
            action="wait",
            position=handover_position,
            velocity=0,
            acc_time=0,
            blend=0,
            constraints=WaypointConstraints(),
            io_before=None,
            io_after=None,
            wait_time=2.0,
            description="Wait for human to take"
        ))
        
        return waypoints
    
    def plan_retreat_task(self,
                         safe_position: Optional[List[float]] = None,
                         velocity: int = 100) -> List[Waypoint]:
        """
        規劃安全後退任務
        
        Args:
            safe_position: 安全位置，None 則使用預設原點
            velocity: 速度百分比
        """
        waypoints = []
        
        # 使用預設安全位置或指定位置
        if safe_position is None:
            safe_position = [291, -225, 613, 180, 0, -90]  # 預設原點
        
        # 移動到安全位置（較慢速度）
        waypoints.append(WaypointBuilder.move_to(
            safe_position, velocity, description="Retreat to safe position"
        ))
        
        return waypoints
    
    def cleanup(self):
        """清理資源"""
        if self.node:
            self.node.destroy_node()
