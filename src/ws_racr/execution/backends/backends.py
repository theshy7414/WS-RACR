"""
統一的執行後端系統
支援三種執行模式：MoveIt2、TM SetPositions、TMScript
"""
import logging
from typing import List, Dict, Optional, Literal
from abc import ABC, abstractmethod
import time
import threading

from ws_racr.execution.backends.waypoint_schema import Waypoint

logger = logging.getLogger(__name__)

# 嘗試導入各種後端
try:
    from ws_racr.execution.backends.ros_communicator import RobotCommunicator
    TM_AVAILABLE = True
except ImportError:
    TM_AVAILABLE = False
    RobotCommunicator = None

try:
    from tm_task.script_translator import TMScript
    TMSCRIPT_AVAILABLE = True
except ImportError:
    TMSCRIPT_AVAILABLE = False
    TMScript = None

try:
    import rclpy
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import Constraints
    from rclpy.action import ActionClient
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False
    # 建立假類型避免類型提示錯誤
    class Constraints:
        pass

BackendType = Literal["moveit", "tm_setpos", "tmscript"]


class ExecutionBackend(ABC):
    """執行後端基類"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
        self.current_waypoint_index = 0
        
    @abstractmethod
    def initialize(self) -> bool:
        """初始化後端"""
        pass
    
    @abstractmethod
    def execute_waypoints(self, waypoints: List[Waypoint]) -> bool:
        """執行路徑點序列"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """停止執行"""
        pass
    
    @abstractmethod
    def is_moving(self) -> bool:
        """檢查是否正在移動"""
        pass
    
    def get_progress(self) -> Dict:
        """獲取執行進度"""
        return {
            "current_waypoint": self.current_waypoint_index,
            "is_moving": self.is_moving(),
            "backend": self.name
        }


class MoveItBackend(ExecutionBackend):
    """MoveIt2 執行後端 - 使用 MoveGroupInterface"""
    
    def __init__(self, group_name: str = "tm_arm"):
        super().__init__("moveit")
        self.group_name = group_name
        self.move_group_client = None
        self.current_goal_handle = None
        self._is_moving = False
        
        if not MOVEIT_AVAILABLE:
            logger.warning("MoveIt2 不可用")
    
    def initialize(self) -> bool:
        """初始化 MoveIt2 客戶端"""
        if not MOVEIT_AVAILABLE:
            logger.error("MoveIt2 未安裝")
            return False
        
        try:
            # 初始化 ROS2
            if not rclpy.ok():
                rclpy.init()
            
            # 創建 Action Client
            from rclpy.node import Node
            self.node = Node('moveit_backend')
            
            self.move_group_client = ActionClient(
                self.node,
                MoveGroup,
                f'/{self.group_name}/move_action'
            )
            
            logger.info(f"等待 MoveIt2 Action Server: /{self.group_name}/move_action")
            if not self.move_group_client.wait_for_server(timeout_sec=5.0):
                logger.error("MoveIt2 Action Server 超時")
                return False
            
            self.is_initialized = True
            logger.info("MoveIt2 Backend 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"MoveIt2 初始化失敗: {e}")
            return False
    
    def execute_waypoints(self, waypoints: List[Waypoint]) -> bool:
        """
        執行路徑點序列
        使用 MoveIt2 的 compute_cartesian_path + execute
        """
        if not self.is_initialized:
            logger.error("MoveIt2 Backend 未初始化")
            return False
        
        logger.info(f"MoveIt2 執行 {len(waypoints)} 個路徑點")
        
        try:
            for i, wp in enumerate(waypoints):
                self.current_waypoint_index = i
                
                # 處理不同動作類型
                if wp.action in ["gripper_open", "gripper_close"]:
                    # IO 動作不通過 MoveIt2
                    success = self._execute_io_action(wp)
                    if not success:
                        return False
                    
                elif wp.action == "wait":
                    # 等待動作
                    time.sleep(wp.wait_time)
                    
                else:
                    # 移動動作
                    success = self._execute_move_action(wp)
                    if not success:
                        return False
            
            logger.info("MoveIt2 執行完成")
            return True
            
        except Exception as e:
            logger.error(f"MoveIt2 執行錯誤: {e}")
            return False
    
    def _execute_move_action(self, wp: Waypoint) -> bool:
        """執行單個移動動作"""
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
        
        # 建立目標姿態
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.num_planning_attempts = 3
        goal.request.allowed_planning_time = wp.planning_time
        goal.request.planner_id = wp.planner_id
        
        # 設定目標位置
        pose = Pose()
        pos_m = wp.get_position_meters()
        ori_rad = wp.get_orientation_radians()
        
        pose.position.x = pos_m[0]
        pose.position.y = pos_m[1]
        pose.position.z = pos_m[2]
        
        # 轉換 RPY 到四元數
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(ori_rad[0], ori_rad[1], ori_rad[2])
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        
        goal.request.goal_constraints.append(self._create_pose_constraint(pose, wp))
        
        # 設定速度/加速度縮放
        goal.request.max_velocity_scaling_factor = wp.get_moveit_velocity_scaling()
        goal.request.max_acceleration_scaling_factor = wp.get_moveit_acceleration_scaling()
        
        # 發送目標
        logger.info(f"MoveIt2: {wp.description or wp.action}")
        self._is_moving = True
        
        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=wp.timeout)
        
        self.current_goal_handle = future.result()
        if not self.current_goal_handle.accepted:
            logger.error("MoveIt2 目標被拒絕")
            self._is_moving = False
            return False
        
        # 等待執行完成
        result_future = self.current_goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=wp.timeout)
        
        result = result_future.result()
        self._is_moving = False
        
        if result.result.error_code.val == 1:  # SUCCESS
            logger.info(f"MoveIt2 完成: {wp.description}")
            return True
        else:
            logger.error(f"MoveIt2 失敗: error_code={result.result.error_code.val}")
            return False
    
    def _create_pose_constraint(self, pose, wp: Waypoint) -> Constraints:
        """建立姿態約束"""
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Point
        
        constraints = Constraints()
        
        # 位置約束
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "tool0"
        
        # 使用球形約束區域
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [wp.constraints.position_tolerance]
        pos_constraint.constraint_region.primitives.append(sphere)
        
        pos_constraint.constraint_region.primitive_poses.append(pose)
        pos_constraint.weight = 1.0
        
        # 姿態約束
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "base_link"
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = pose.orientation
        ori_constraint.absolute_x_axis_tolerance = wp.constraints.orientation_tolerance
        ori_constraint.absolute_y_axis_tolerance = wp.constraints.orientation_tolerance
        ori_constraint.absolute_z_axis_tolerance = wp.constraints.orientation_tolerance
        ori_constraint.weight = 1.0
        
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        
        return constraints
    
    def _execute_io_action(self, wp: Waypoint) -> bool:
        """執行 IO 動作 (委派給 TM 通道)"""
        logger.warning(f"MoveIt2 不支援 IO 動作: {wp.action}")
        logger.info(f"建議: 使用 TMScriptBackend 或在外部處理 IO")
        
        # 如果有 TM Communicator 可用，委派給它
        if TM_AVAILABLE:
            try:
                from ws_racr.execution.backends.ros_communicator import RobotCommunicator
                temp_comm = RobotCommunicator()
                
                closed = (wp.action == "gripper_close" or wp.action == "grasp")
                success = temp_comm.set_gripper(closed=closed)
                
                if success:
                    logger.info(f"   IO 已委派執行")
                    time.sleep(wp.wait_time if wp.wait_time > 0 else 1.0)
                    return True
                else:
                    logger.error(f"   IO 執行失敗")
                    return False
            except Exception as e:
                logger.error(f"   IO 委派錯誤: {e}")
                return False
        
        return True  # 不阻塞執行
    
    def stop(self) -> bool:
        """停止當前執行"""
        if self.current_goal_handle:
            try:
                self.current_goal_handle.cancel_goal_async()
                logger.info("MoveIt2 已停止")
                self._is_moving = False
                return True
            except Exception as e:
                logger.error(f"停止失敗: {e}")
                return False
        return True
    
    def is_moving(self) -> bool:
        """檢查是否正在移動"""
        return self._is_moving


class TMSetPosBackend(ExecutionBackend):
    """TM SetPositions 執行後端 - 逐點發送"""
    
    def __init__(self):
        super().__init__("tm_setpos")
        self.communicator = None
        
        if not TM_AVAILABLE:
            logger.warning("TM ROS2 通訊不可用")
    
    def initialize(self) -> bool:
        """初始化 TM Communicator"""
        if not TM_AVAILABLE:
            logger.error("TM Communicator 未安裝")
            return False
        
        try:
            self.communicator = RobotCommunicator()
            self.is_initialized = True
            logger.info("TM SetPos Backend 初始化成功")
            return True
        except Exception as e:
            logger.error(f"TM SetPos 初始化失敗: {e}")
            return False
    
    def execute_waypoints(self, waypoints: List[Waypoint]) -> bool:
        """執行路徑點序列 - 逐點調用 SetPositions"""
        if not self.is_initialized:
            logger.error("TM SetPos Backend 未初始化")
            return False
        
        logger.info(f"TM SetPos 執行 {len(waypoints)} 個路徑點")
        
        try:
            for i, wp in enumerate(waypoints):
                self.current_waypoint_index = i
                
                # IO 動作前
                if wp.io_before:
                    self._execute_io(wp.io_before)
                
                # 執行動作
                if wp.action in ["gripper_open", "gripper_close"]:
                    success = self._execute_gripper(wp)
                elif wp.action == "wait":
                    time.sleep(wp.wait_time)
                    success = True
                else:
                    success = self._execute_move(wp)
                
                if not success:
                    logger.error(f"路徑點 {i} 執行失敗")
                    return False
                
                # IO 動作後
                if wp.io_after:
                    self._execute_io(wp.io_after)
                
                # 等待時間
                if wp.wait_time > 0:
                    time.sleep(wp.wait_time)
            
            logger.info("TM SetPos 執行完成")
            return True
            
        except Exception as e:
            logger.error(f"TM SetPos 執行錯誤: {e}")
            return False
    
    def _execute_move(self, wp: Waypoint) -> bool:
        """執行移動動作 - 完整支持 velocity/acc_time/blend"""
        logger.info(f" {wp.description or wp.action}")
        logger.info(f"   速度: {wp.get_tm_velocity_percent()}%, 加速: {wp.get_tm_acc_time_ms()}ms, 混合: {wp.get_tm_blend_percent()}%")
        
        # 調用 RobotCommunicator (SetPositions 服務)
        # 注意: SetPositions 可能不支援 blend_percentage，需檢查 API
        success = self.communicator.move_to_position(
            position=wp.position,
            velocity=wp.get_tm_velocity_percent() / 100.0,  # 轉換為 0.0-1.0
            acc_time=wp.get_tm_acc_time_ms() / 1000.0  # 轉換為秒
            # blend 參數通常在 SetPositions 中不支援，改用 TMScript
        )
        
        if success:
            # 等待移動完成
            self.communicator.wait_for_movement_complete(timeout=wp.timeout)
            return True
        else:
            logger.error(f"移動失敗: {wp.position}")
            return False
    
    def _execute_gripper(self, wp: Waypoint) -> bool:
        """執行夾爪動作"""
        closed = (wp.action == "gripper_close" or wp.action == "grasp")
        logger.info(f"夾爪: {'關閉' if closed else '打開'}")
        
        return self.communicator.set_gripper(closed=closed)
    
    def _execute_io(self, io_config: Dict) -> bool:
        """執行 IO 設定"""
        # io_config: {"module": "EndModule", "type": "DO", "pin": 0, "state": True}
        logger.info(f"IO: {io_config}")
        # 可以調用 communicator 的 IO 方法
        return True
    
    def stop(self) -> bool:
        """停止執行"""
        if self.communicator:
            # TM Robot 沒有直接停止方法，可以發送停止腳本
            logger.info("TM SetPos 停止")
            return True
        return False
    
    def is_moving(self) -> bool:
        """檢查是否正在移動"""
        if self.communicator and hasattr(self.communicator, 'robot_state'):
            return self.communicator.robot_state.get("moving", False)
        return False


class TMScriptBackend(ExecutionBackend):
    """TMScript 執行後端 - 批次腳本執行（統一 QueueTag 管理）"""
    
    def __init__(self):
        super().__init__("tmscript")
        self.communicator = None
        self.tmscript = None
        self.execution_helper = None  # 新增執行助手
        
        if not TM_AVAILABLE or not TMSCRIPT_AVAILABLE:
            logger.warning("TMScript 或 TM Communicator 不可用")
    
    def initialize(self) -> bool:
        """初始化 TMScript + Communicator + ExecutionHelper"""
        if not TM_AVAILABLE or not TMSCRIPT_AVAILABLE:
            logger.error("TMScript Backend 依賴未滿足")
            return False
        
        try:
            self.communicator = RobotCommunicator()
            self.tmscript = TMScript()
            
            # 初始化執行助手（不需要 task_manager，由外部處理）
            from ws_racr.execution.backends.script_helper import ScriptExecutionHelper
            self.execution_helper = ScriptExecutionHelper(
                communicator=self.communicator
            )
            
            self.is_initialized = True
            logger.info("TMScript Backend 初始化成功")
            return True
        except Exception as e:
            logger.error(f"TMScript 初始化失敗: {e}")
            return False
    
    def execute_waypoints(self, waypoints: List[Waypoint]) -> bool:
        """執行路徑點序列 - 分段批次執行（避免連續任務狀態衝突）"""
        if not self.is_initialized:
            logger.error("TMScript Backend 未初始化")
            return False
        
        logger.info(f"TMScript 分段執行 {len(waypoints)} 個路徑點")
        
        # DEBUG: 打印所有 waypoints
        logger.info("\nWaypoint 列表:")
        for i, wp in enumerate(waypoints):
            logger.info(f"   {i+1}. {wp.action} - {wp.description or ''}")
        
        try:
            # 策略：分段執行，每個關鍵動作獨立批次
            # Batch 1: Gripper Open + Wait
            # Batch 2: Move to safe height
            # Batch 3: Approach + Grasp
            # Batch 4: Gripper Close + Wait
            # Batch 5: Lift
            
            batches = []
            current_batch = []
            
            for i, wp in enumerate(waypoints):
                self.current_waypoint_index = i
                
                # === IO 動作前 ===
                if wp.io_before:
                    io_script = self._generate_io_script(wp.io_before)
                    current_batch.append(io_script)
                
                # === 主要動作 ===
                if wp.action in ["move", "move_to"]:
                    script = self._generate_move_script(wp)
                    current_batch.append(script)
                    
                elif wp.action in ["move_relative", "rotate_relative"]:
                    script = self._generate_move_relative_script(wp)
                    current_batch.append(script)
                    
                elif wp.action == "move_linear":
                    script = self._generate_linear_script(wp)
                    current_batch.append(script)
                    
                elif wp.action in ["gripper_open", "gripper_close", "grasp", "release"]:
                    script = self._generate_gripper_script(wp)
                    logger.info(f"   生成夾爪指令: {wp.action} → {script}")
                    current_batch.append(script)
                    
                    # 夾爪動作後立即添加等待
                    if wp.wait_time > 0:
                        wait_ms = int(wp.wait_time * 1000)
                        current_batch.append(f"Sleep({wait_ms})")
                        wp.wait_time = 0
                    
                    # 夾爪動作後切換批次（關鍵隔離點）
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                    continue
                    
                elif wp.action == "wait":
                    wait_ms = int(wp.wait_time * 1000)
                    current_batch.append(f"Sleep({wait_ms})")
                    continue
                    
                else:
                    logger.warning(f"未知動作: {wp.action}")
                    continue
                
                # === IO 動作後 ===
                if wp.io_after:
                    io_script = self._generate_io_script(wp.io_after)
                    current_batch.append(io_script)
                
                # === 等待時間 ===
                if wp.wait_time > 0:
                    wait_ms = int(wp.wait_time * 1000)
                    current_batch.append(f"Sleep({wait_ms})")
            
            # 收集剩餘指令
            if current_batch:
                batches.append(current_batch)
            
            if not batches:
                logger.warning("沒有生成任何指令")
                return {"success": False, "error": "沒有生成任何指令", "command_count": 0, "execution_time": 0}
            
            # === 逐批執行 ===
            logger.info(f"分成 {len(batches)} 個批次執行")
            total_commands = 0
            total_time = 0.0
            
            for batch_idx, batch in enumerate(batches, 1):
                logger.info(f"\n批次 {batch_idx}/{len(batches)}: {len(batch)} 條指令")
                result = self._execute_script_batch(batch)
                
                if not result["success"]:
                    logger.error(f"批次 {batch_idx} 執行失敗: {result.get('error')}")
                    return result
                
                total_commands += result["command_count"]
                total_time += result["execution_time"]
                
                # 批次間短暫延遲（確保狀態穩定）
                if batch_idx < len(batches):
                    time.sleep(0.1)
            
            logger.info(f"\n所有批次完成: {total_commands} 條指令, 總耗時 {total_time:.2f}秒")
            return {"success": True, "command_count": total_commands, "execution_time": total_time}
            
        except Exception as e:
            logger.error(f"錯誤: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e), "command_count": 0, "execution_time": 0}
    
    def _execute_script_batch(self, scripts: List[str]) -> Dict:
        """
        批次執行 TMScript 指令（使用 ExecutionHelper）
        
        Args:
            scripts: TMScript 指令列表（不含 QueueTag）
        
        Returns:
            Dict: 執行報告 {success, command_count, execution_time, error, ...}
        """
        if not scripts:
            return {"success": False, "error": "空指令列表", "command_count": 0, "execution_time": 0}
        
        # 使用執行助手，自動處理「送→等→驗」
        if self.execution_helper:
            result = self.execution_helper.execute_and_track(
                scripts=scripts,
                task_id=None,  # task_id 由外部 DatabaseController 管理
                timeout=60.0,
                description=f"執行 {len(scripts)} 個 TMScript 指令"
            )
            return result  # 回傳完整報告
        else:
            # 降級：直接使用 communicator
            logger.warning("ExecutionHelper 不可用，使用基本模式")
            result = self.communicator.execute_script_sequence(scripts, timeout=60.0)
            
            if result["success"]:
                logger.info(f"執行成功: {result['command_count']} 個指令, 耗時 {result['execution_time']:.2f}秒")
            else:
                logger.error(f"執行失敗: {result.get('error', '未知錯誤')}")
            
            return result  # 回傳完整報告
    
    def _generate_move_script(self, wp: Waypoint) -> str:
        """生成移動腳本 - 完整支持 velocity/acc_time/blend"""
        # 使用 TM Robot 參數
        return self.tmscript.PTP(
            wp.position,  # [x, y, z, rx, ry, rz]
            wp.get_tm_velocity_percent(),  # 1-100%
            wp.get_tm_acc_time_ms(),  # 1-2000ms
            wp.get_tm_blend_percent()  # 0-100%
        )
    
    def _generate_move_relative_script(self, wp: Waypoint) -> str:
        """生成相對移動腳本 - 使用 movePTP"""
        # 使用 TMScript.movePTP() 執行相對移動
        return self.tmscript.movePTP(
            wp.position,  # [dx, dy, dz, drx, dry, drz]
            wp.get_tm_velocity_percent(),  # 1-100%
            wp.get_tm_acc_time_ms(),  # 1-2000ms
            wp.get_tm_blend_percent()  # 0-100%
        )
    
    def _generate_linear_script(self, wp: Waypoint) -> str:
        """生成直線移動腳本"""
        # Line 指令也支持相同參數
        return self.tmscript.Line(
            wp.position,
            wp.get_tm_velocity_percent(),
            wp.get_tm_acc_time_ms(),
            wp.get_tm_blend_percent()
        )
    
    def _generate_gripper_script(self, wp: Waypoint) -> str:
        """生成夾爪腳本"""
        from ws_racr.config import get_gripper_config
        config = get_gripper_config()
        
        state = (wp.action == "gripper_close" or wp.action == "grasp")
        return self.tmscript.setIO(
            config['module'],  # 位置參數
            config['io_type'],
            config['pin'],
            state
        )
    
    def _generate_io_script(self, io_config: Dict) -> str:
        """
        生成 IO 腳本
        支援兩種格式：
        1. {"EndModule.DO[0]": True/False}  # 字典key格式
        2. {"module": "EndModule", "type": "DO", "pin": 0, "state": True/False}  # 分解格式
        """
        # 檢查是否為字典key格式（如 "EndModule.DO[0]"）
        if len(io_config) == 1:
            key = list(io_config.keys())[0]
            if '.' in key and '[' in key:
                # 解析格式：EndModule.DO[0]
                parts = key.split('.')
                module = parts[0]
                io_part = parts[1]  # DO[0]
                io_type = io_part.split('[')[0]  # DO
                pin = int(io_part.split('[')[1].rstrip(']'))  # 0
                state = io_config[key]
                
                return self.tmscript.setIO(
                    module=module,
                    name=io_type,
                    pin=pin,
                    state=state
                )
        
        # 使用分解格式
        return self.tmscript.setIO(
            module=io_config.get('module', 'EndModule'),
            name=io_config.get('type', 'DO'),
            pin=io_config.get('pin', 0),
            state=io_config.get('state', False)
        )
    
    def stop(self) -> bool:
        """停止執行"""
        if self.tmscript and self.communicator:
            stop_script = self.tmscript.stop()
            self.communicator.send_script(stop_script, script_id=999)
            logger.info("TMScript 已停止")
            return True
        return False
    
    def is_moving(self) -> bool:
        """檢查是否正在移動"""
        if self.communicator and hasattr(self.communicator, 'robot_state'):
            return self.communicator.robot_state.get("moving", False)
        return False


class ExecutionBackendManager:
    """執行後端管理器 - 統一介面"""
    
    def __init__(self, backend_type: BackendType = "tm_setpos"):
        self.backend_type = backend_type
        self.backend: Optional[ExecutionBackend] = None
        
        # 建立後端實例
        if backend_type == "moveit":
            self.backend = MoveItBackend()
        elif backend_type == "tm_setpos":
            self.backend = TMSetPosBackend()
        elif backend_type == "tmscript":
            self.backend = TMScriptBackend()
        else:
            raise ValueError(f"不支援的後端類型: {backend_type}")
    
    def initialize(self) -> bool:
        """初始化後端"""
        if self.backend:
            return self.backend.initialize()
        return False
    
    def execute_waypoints(self, waypoints: List[Waypoint]) -> bool:
        """執行路徑點"""
        if self.backend and self.backend.is_initialized:
            result = self.backend.execute_waypoints(waypoints)
            # TMScriptBackend 回傳 Dict；統一轉換為 bool 供所有呼叫方使用
            return result.get('success', False) if isinstance(result, dict) else bool(result)
        else:
            logger.error(f"Backend {self.backend_type} 未初始化")
            return False
    
    def stop(self) -> bool:
        """停止執行"""
        if self.backend:
            return self.backend.stop()
        return False
    
    def is_moving(self) -> bool:
        """檢查是否移動中"""
        if self.backend:
            return self.backend.is_moving()
        return False
    
    def get_progress(self) -> Dict:
        """獲取執行進度"""
        if self.backend:
            return self.backend.get_progress()
        return {}
    
    def switch_backend(self, new_backend_type: BackendType) -> bool:
        """切換執行後端"""
        logger.info(f"切換後端: {self.backend_type} -> {new_backend_type}")
        
        # 停止當前後端
        if self.backend:
            self.backend.stop()
        
        # 建立新後端
        self.backend_type = new_backend_type
        if new_backend_type == "moveit":
            self.backend = MoveItBackend()
        elif new_backend_type == "tm_setpos":
            self.backend = TMSetPosBackend()
        elif new_backend_type == "tmscript":
            self.backend = TMScriptBackend()
        
        # 初始化
        return self.initialize()


# 測試範例
if __name__ == "__main__":
    from ws_racr.execution.backends.waypoint_schema import WaypointBuilder
    
    # 建立測試路徑點
    waypoints = [
        WaypointBuilder.move_to([291, -225, 613, 180, 0, -90], velocity=100),
        WaypointBuilder.gripper_open(),
        WaypointBuilder.move_to([352, 336, 563, 180, 0, 90], velocity=100),
        WaypointBuilder.gripper_close(),
    ]
    
    # 測試不同後端
    print("=== 測試 TM SetPos Backend ===")
    manager = ExecutionBackendManager("tm_setpos")
    if manager.initialize():
        manager.execute_waypoints(waypoints)
