"""
LLM ROS2 通訊介面模組
整合 TM Robot 和視覺檢測的通訊功能
"""
import logging
import json
import time
import math
from typing import Dict, List, Callable, Optional
import threading

from ws_racr.config import get_gripper_config, get_movement_config

logger = logging.getLogger(__name__)

GRIPPER_CONFIG = get_gripper_config()
MOVEMENT_CONFIG = get_movement_config()

# 可選的 ROS2 導入
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    
    from std_msgs.msg import UInt16, String
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Point
    from tm_msgs.msg import HandPos, Objects, FeedbackState
    from tm_msgs.srv import SetPositions, SetIO, SendScript, PixelToWorld
    
    ROS_AVAILABLE = True
except ImportError:
    # 如果 ROS2 不可用，創建模擬類別
    ROS_AVAILABLE = False
    
    class MockClient:
        """Stand-in service client used when ROS 2 is unavailable."""

        def wait_for_service(self, *args, **kwargs):
            return False

    class Node:
        def __init__(self, name): 
            self.name = name
        def get_logger(self):
            return type('Logger', (), {'info': print, 'error': print, 'warn': print})()
        def create_service(self, *args, **kwargs): pass
        def create_client(self, *args, **kwargs): return MockClient()
        def create_subscription(self, *args, **kwargs): pass
        def create_publisher(self, *args, **kwargs): return None
        def destroy_node(self): pass
    
    # 模擬 ROS2 訊息類型
    class MockMessage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    UInt16 = String = Image = Point = MockMessage
    HandPos = Objects = FeedbackState = MockMessage
    SetPositions = SetIO = SendScript = PixelToWorld = MockMessage
    
    ReentrantCallbackGroup = None
    MultiThreadedExecutor = None

class RobotCommunicator(Node):
    """LLM 機器人通訊管理器 - 處理所有 ROS2 通訊"""
    
    def __init__(self, node_name: str = "llm_robot_communicator"):
        super().__init__(node_name)
        
        # 使用多線程回調組
        self.callback_group = ReentrantCallbackGroup()
        
        # 機器人狀態
        self.robot_state = {
            "position": [0.0] * 6,
            "moving": False,
            "gripper_closed": False,
            "project_running": False,
            "connected": False
        }
        
        # 檢測到的物件
        self.detected_objects = {
            "shapes": [],
            "colors": [],
            "world_coords": []
        }
        
        # 手部位置
        self.hand_position = {
            "image_coord": [0, 0],
            "world_coord": [0.0, 0.0, 0.0]
        }
        
        # 執行狀態
        self.execution_complete = threading.Event()
        self.current_script_id = None
        
        # 設置 ROS2 通訊
        self._setup_ros_communication()
        
        self.get_logger().info("LLM Robot Communicator initialized")
    
    def _setup_ros_communication(self):
        """設置 ROS2 訂閱和服務"""
        # 訂閱機器人反饋
        subscription_kwargs = {"callback_group": self.callback_group} if self.callback_group else {}
        self.create_subscription(
            FeedbackState,
            "feedback_states",
            self.feedback_callback,
            10,
            **subscription_kwargs
        )
        
        # 訂閱物件檢測結果
        self.create_subscription(
            Objects,
            "objects",
            self.objects_callback,
            10,
            callback_group=self.callback_group
        )
        
        # 訂閱手部位置
        self.create_subscription(
            HandPos,
            "hand_pos",
            self.hand_pos_callback,
            10,
            callback_group=self.callback_group
        )
        
        # 機器人控制服務
        self.set_positions_client = self.create_client(
            SetPositions, 
            'set_positions',
            callback_group=self.callback_group
        )
        
        self.set_io_client = self.create_client(
            SetIO,
            'set_io', 
            callback_group=self.callback_group
        )
        
        self.send_script_client = self.create_client(
            SendScript,
            'send_script',
            callback_group=self.callback_group
        )
        
        self.pixel_to_world_client = self.create_client(
            PixelToWorld,
            'pixel_to_world',
            callback_group=self.callback_group
        )
        
        # 等待服務可用
        self._wait_for_services()
    
    def _wait_for_services(self):
        """等待所有服務可用"""
        # 必要服務
        required_services = [
            (self.set_positions_client, 'set_positions'),
            (self.set_io_client, 'set_io'),
            (self.send_script_client, 'send_script')
        ]
        
        # 可選服務
        optional_services = [
            (self.pixel_to_world_client, 'pixel_to_world')
        ]
        
        # 等待必要服務
        for client, name in required_services:
            self.get_logger().info(f"Waiting for service: {name}")
            client.wait_for_service(timeout_sec=10.0)
            if client.service_is_ready():
                self.get_logger().info(f"Service ready: {name}")
            else:
                self.get_logger().error(f"Required service timeout: {name}")
                raise RuntimeError(f"Required service {name} not available")
        
        # 檢查可選服務
        for client, name in optional_services:
            self.get_logger().info(f"Waiting for service: {name}")
            if client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f"Service ready: {name}")
            else:
                self.get_logger().warn(f"Optional service timeout: {name} (will use local implementation)")
    
    def feedback_callback(self, msg):
        """機器人反饋回調"""
        # 更新位置 (轉換單位：m -> mm, rad -> deg)
        position = list(msg.tool_pose)
        self.robot_state["position"] = [
            position[0] * 1000,  # m -> mm
            position[1] * 1000,
            position[2] * 1000,
            position[3] * 180 / math.pi,  # rad -> deg
            position[4] * 180 / math.pi,
            position[5] * 180 / math.pi
        ]
        
        # 更新運動狀態
        tcp_speed = sum(abs(v) for v in msg.tcp_speed)
        self.robot_state["moving"] = tcp_speed > 0.001
        
        # 更新項目狀態
        self.robot_state["project_running"] = msg.project_run
        self.robot_state["connected"] = True
        
        # 更新夾爪狀態（基於 DO 推測）
        # 原因：機器人 Listen Node 未配置發送 EndModule DI 數據
        # 實測發現 ee_digital_input 為固定垃圾值，無法使用
        # 解決方案：從 DO（控制命令）推測夾爪狀態
        if msg.ee_digital_output and len(msg.ee_digital_output) >= 1:
            do_0 = bool(msg.ee_digital_output[0])
            
            # DO[0]=1 → 發送了關閉命令 → 推測夾爪關閉
            # DO[0]=0 → 發送了打開命令 → 推測夾爪打開
            self.robot_state["gripper_closed"] = do_0
    
    def objects_callback(self, msg):
        """物件檢測回調"""
        self.detected_objects = {
            "shapes": msg.shapes,
            "colors": msg.colors,
            "world_coords": [[p.x, p.y, p.z] for p in msg.world_coords]
        }
    
    def hand_pos_callback(self, msg):
        """手部位置回調"""
        self.hand_position = {
            "image_coord": [msg.image_x, msg.image_y],
            "world_coord": [msg.world_x, msg.world_y, msg.world_z]
        }
    
    def move_to_position(self, position: List[float], velocity: float = 1.0, acc_time: float = 0.2) -> bool:
        """
        移動到指定位置
        
        Args:
            position: [x, y, z, rx, ry, rz] (mm, deg)
            velocity: 移動速度 (0.0-1.0)
            acc_time: 加速時間 (秒)
            
        Returns:
            bool: 是否成功發送指令
        """
        if not self.set_positions_client.service_is_ready():
            self.get_logger().error("SetPositions service not ready")
            return False
        
        request = SetPositions.Request()
        
        # 轉換單位：mm -> m, deg -> rad
        request.positions = [
            position[0] / 1000,  # mm -> m
            position[1] / 1000,
            position[2] / 1000,
            position[3] * math.pi / 180,  # deg -> rad
            position[4] * math.pi / 180,
            position[5] * math.pi / 180
        ]
        
        request.motion_type = SetPositions.Request.PTP_T
        request.velocity = velocity
        request.acc_time = acc_time
        request.blend_percentage = 10
        request.fine_goal = False
        
        try:
            future = self.set_positions_client.call_async(request)
            # 不等待回應，異步執行
            self.get_logger().info(f"Moving to position: {position}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to move to position: {e}")
            return False
    
    def set_gripper(self, closed: bool) -> bool:
        """
        設置夾爪狀態
        
        Args:
            closed: True=關閉夾爪, False=打開夾爪
            
        Returns:
            bool: 是否成功發送指令
        """
        if not self.set_io_client.service_is_ready():
            self.get_logger().error("SetIO service not ready")
            return False
        
        request = SetIO.Request()
        request.module = SetIO.Request.MODULE_ENDEFFECTOR
        request.type = SetIO.Request.TYPE_DIGITAL_OUT
        request.pin = GRIPPER_CONFIG['pin']  # 使用配置中的腳位
        request.state = float(closed)
        
        try:
            future = self.set_io_client.call_async(request)
            action = "關閉" if closed else "打開"
            self.get_logger().info(f"{action}夾爪 (pin={GRIPPER_CONFIG['pin']})")
            
            # 更新本地狀態（因為 DO 會立即反映命令）
            self.robot_state["gripper_closed"] = closed
            
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to set gripper: {e}")
            return False
    
    def initialize_gripper_state(self) -> bool:
        """
        初始化夾爪狀態 - 確保 DO 與實際狀態同步
        
        流程：打開 → 等待 → 關閉 → 等待 → 回到預設狀態（打開）
        這樣可以確保系統知道夾爪的實際狀態
        
        Returns:
            bool: 是否成功初始化
        """
        self.get_logger().info("🔧 初始化夾爪狀態...")
        
        import time
        
        try:
            # 1. 打開夾爪
            if not self.set_gripper(False):
                return False
            time.sleep(GRIPPER_CONFIG['open_delay'])
            
            # 2. 關閉夾爪
            if not self.set_gripper(True):
                return False
            time.sleep(GRIPPER_CONFIG['close_delay'])
            
            # 3. 回到預設狀態（打開）
            if not self.set_gripper(False):
                return False
            time.sleep(GRIPPER_CONFIG['open_delay'])
            
            self.get_logger().info("✅ 夾爪初始化完成 (預設狀態: 張開)")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ 夾爪初始化失敗: {e}")
            return False
    
    def send_script(self, script: str, script_id: str = "llm_script") -> bool:
        """
        發送 TMScript 腳本
        
        Args:
            script: TMScript 腳本內容（單條指令，不含 QueueTag）
            script_id: 腳本 ID
            
        Returns:
            bool: 是否成功發送指令
        """
        if not self.send_script_client.service_is_ready():
            self.get_logger().error("SendScript service not ready")
            return False
        
        request = SendScript.Request()
        request.id = script_id
        request.script = script  # 直接使用腳本，不添加 QueueTag
        
        try:
            self.get_logger().info(f"Sending script {script_id}: {script}")
            future = self.send_script_client.call_async(request)

            # 等待 ROS service 回應（確認腳本已送達 TM Robot driver）
            # 使用 polling 而非 spin_until_future_complete，避免多執行緒衝突
            deadline = time.time() + 2.0
            while time.time() < deadline and not future.done():
                time.sleep(0.01)
                if ROS_AVAILABLE:
                    import rclpy
                    rclpy.spin_once(self, timeout_sec=0.005)

            if not future.done() or future.result() is None:
                self.get_logger().warning(f"Script {script_id} service ack timeout, proceeding anyway")
                # 不 return False：ack timeout 不代表腳本未送達，繼續執行
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to send script: {e}")
            return False
    
    def execute_script_sequence(self, commands: List[str], timeout: float = 30.0) -> Dict:
        """
        執行 TMScript 指令序列（逐條發送 + 統一 QueueTag）
        
        完整流程：
        1. 逐條發送每個指令（每條都有獨立的訊息格式）
        2. 在最後一條指令後添加 QueueTag(1,1) - 參數為 (tag_number, wait) 皆為 int
        3. 等待執行完成或超時
        4. 驗證執行結果
        5. 返回執行報告
        
        Args:
            commands: TMScript 指令列表（不含 QueueTag）
            timeout: 超時時間（秒）
            
        Returns:
            Dict: 執行報告
                {
                    "success": bool,
                    "command_count": int,
                    "execution_time": float,
                    "error": Optional[str]
                }
        """
        if not commands:
            return {
                "success": False,
                "command_count": 0,
                "execution_time": 0.0,
                "error": "沒有指令"
            }
        
        start_time = time.time()
        
        self.get_logger().info(f"📦 執行批次腳本: {len(commands)} 個指令")
        
        # 檢查機器人狀態
        if self.robot_state.get("moving", False):
            self.get_logger().warn("⚠️ 機器人仍在移動中，等待停止...")
            wait_time = 0
            while self.robot_state.get("moving", False) and wait_time < 5.0:
                time.sleep(0.2)
                wait_time += 0.2
                if ROS_AVAILABLE:
                    import rclpy
                    rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.robot_state.get("moving", False):
                return {
                    "success": False,
                    "command_count": 0,
                    "execution_time": time.time() - start_time,
                    "error": "機器人忙碌中，無法執行新指令"
                }
        
        # 1. 逐條發送每個指令（增加延遲，避免 QUEUE3205 錯誤）
        timestamp = int(time.time()) % 100000  # 只取後5位，避免太長
        for i, cmd in enumerate(commands, start=1):
            # 根據指令類型生成簡短 ID
            if "IO[" in cmd:
                script_id = f"IO{timestamp}{i}"
            elif "PTP(" in cmd:
                script_id = f"PTP{timestamp}{i}"
            elif "Wait(" in cmd or "Sleep(" in cmd:
                script_id = f"WAIT{timestamp}{i}"
            else:
                script_id = f"CMD{timestamp}{i}"
            
            success = self.send_script(cmd, script_id)
            
            if not success:
                return {
                    "success": False,
                    "command_count": i,
                    "execution_time": time.time() - start_time,
                    "error": f"發送第 {i} 條指令失敗"
                }
            
            # 增加延遲：50ms → 100ms，避免 QUEUE3205 錯誤
            time.sleep(0.1)
        
        # 2. 發送 QueueTag(1,1) 在最後（ID 不能有特殊符號）
        # 參數: (tag_number, wait) 都是 int，1 表示等待
        queue_script_id = f"QUEUE{timestamp}"
        success = self.send_script("QueueTag(1,1)", queue_script_id)
        
        if not success:
            return {
                "success": False,
                "command_count": len(commands),
                "execution_time": time.time() - start_time,
                "error": "發送 QueueTag 失敗"
            }
        
        # 3. 批次等待策略
        #
        # 核心問題：tcp_speed / robot_state["moving"] 不可靠：
        #   • 夾爪 IO 致動 → 機械振動 → tcp_speed 瞬間非零 → motion_detected 誤判
        #   • 斷線重連後 robot_state["moving"] 可能殘留 True（舊值）
        #
        # 解決方案：先根據 batch 內容分類，決定等待策略：
        #   IO-only batch（僅含 IO / Sleep，無 PTP/Line/Move）
        #     → 完全不依賴 tcp_speed，直接做時間等待
        #   Motion batch（含 PTP / Line / Move_PTP 等）
        #     → 用三段式 moving 偵測（原始邏輯）
        import re as _re
        if ROS_AVAILABLE:
            import rclpy

        # 解析 batch 類型
        _MOTION_KEYWORDS = ('PTP(', 'Line(', 'Move_PTP(', 'movePTP(', 'Move_Line(')
        _has_motion_cmd = any(
            any(kw in cmd for kw in _MOTION_KEYWORDS)
            for cmd in commands
        )
        _max_sleep_ms = 0
        for _cmd in commands:
            _m = _re.search(r'Sleep\((\d+)\)', _cmd)
            if _m:
                _max_sleep_ms = max(_max_sleep_ms, int(_m.group(1)))

        wait_start = time.time()

        if not _has_motion_cmd:
            # ── IO-only batch：純時間等待，完全不用 tcp_speed ──────────────
            # 等到 Sleep(N) 結束即可，無需額外 overhead。
            required_wait = _max_sleep_ms / 1000.0
            elapsed = time.time() - start_time
            extra = required_wait - elapsed
            if extra > 0:
                self.get_logger().info(
                    f"ℹ️ IO-only 批次含 Sleep({_max_sleep_ms}ms)，"
                    f"等待 {extra:.2f}s（elapsed={elapsed:.2f}s）"
                )
                time.sleep(extra)
            else:
                self.get_logger().info(
                    f"ℹ️ IO-only 批次，已等待足夠時間（elapsed={elapsed:.2f}s）"
                )
            motion_detected = False
        else:
            # ── Motion batch：三段式 tcp_speed 偵測 ────────────────────────
            # — 啟動延遲
            time.sleep(0.1)
            if ROS_AVAILABLE:
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.02)

            # — 等待 moving=True（機器人開始動）
            MOTION_START_TIMEOUT = 2.5
            MOTION_POLL_INTERVAL = 0.05
            motion_detected = False
            phase2_start = time.time()
            while time.time() - phase2_start < MOTION_START_TIMEOUT:
                if ROS_AVAILABLE:
                    rclpy.spin_once(self, timeout_sec=0.02)
                if self.robot_state.get("moving", False):
                    motion_detected = True
                    self.get_logger().info("✅ 偵測到機器人開始移動")
                    break
                time.sleep(MOTION_POLL_INTERVAL)

            # — 等待 moving=False（機器人停止）
            wait_start = time.time()
            if motion_detected:
                while time.time() - wait_start < timeout:
                    if ROS_AVAILABLE:
                        rclpy.spin_once(self, timeout_sec=0.01)
                    if not self.robot_state.get("moving", False):
                        time.sleep(0.3)
                        if ROS_AVAILABLE:
                            rclpy.spin_once(self, timeout_sec=0.01)
                        if not self.robot_state.get("moving", False):
                            break
                    time.sleep(0.1)
            else:
                # Motion batch 但未偵測到動作（罕見）
                self.get_logger().info("ℹ️ Motion batch: 未偵測到移動，可能是短路徑")
                wait_start = time.time()

        execution_time = time.time() - start_time

        # 4. 驗證執行結果
        if motion_detected and time.time() - wait_start >= timeout:
            self.get_logger().warn(f"⚠️ 執行超時 ({timeout}秒)")
            return {
                "success": False,
                "command_count": len(commands),
                "execution_time": execution_time,
                "error": f"執行超時 ({timeout}秒)"
            }

        # 5. 返回成功報告
        self.get_logger().info(f"✅ 批次執行完成: {len(commands)} 個指令, 耗時 {execution_time:.2f}秒")
        
        return {
            "success": True,
            "command_count": len(commands),
            "execution_time": execution_time,
            "error": None
        }
    
    def detect_object_position(self, object_name: str) -> Optional[List[float]]:
        """
        檢測指定物件的位置
        
        Args:
            object_name: 物件名稱
            
        Returns:
            Optional[List[float]]: 物件位置 [x, y, z, rx, ry, rz] 或 None
        """
        # 檢查是否有檢測到的物件
        if not self.detected_objects["world_coords"]:
            return None
        
        # 簡單實現：返回第一個檢測到的物件位置
        # 在實際應用中，這裡需要根據物件名稱進行匹配
        if self.detected_objects["world_coords"]:
            pos = self.detected_objects["world_coords"][0]
            # 轉換為完整的 6D 位置（假設固定姿態）
            return [pos[0] * 1000, pos[1] * 1000, pos[2] * 1000, 180, 0, 90]  # mm, deg
        
        return None
    
    def pixel_to_world_coordinate(self, pixel_x: int, pixel_y: int) -> Optional[List[float]]:
        """
        將像素座標轉換為世界座標
        
        Args:
            pixel_x: 像素 X 座標
            pixel_y: 像素 Y 座標
            
        Returns:
            Optional[List[float]]: 世界座標 [x, y, z] (mm) 或 None
        """
        if not self.pixel_to_world_client.service_is_ready():
            self.get_logger().debug("pixel_to_world service not available, using local implementation")
            return None
        
        request = PixelToWorld.Request()
        request.image_x = pixel_x
        request.image_y = pixel_y
        
        try:
            future = self.pixel_to_world_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if future.result():
                result = future.result()
                return [result.world_x * 1000, result.world_y * 1000, result.world_z * 1000]
        except Exception as e:
            self.get_logger().error(f"Pixel to world conversion failed: {e}")
        
        return None
    
    def stop_robot(self) -> bool:
        """停止機器人運動"""
        stop_script = "StopAndClearBuffer(2)"
        return self.send_script(stop_script, "stop")
    
    def get_robot_status(self) -> Dict:
        """獲取機器人狀態"""
        return self.robot_state.copy()
    
    def get_detected_objects(self) -> Dict:
        """獲取檢測到的物件"""
        return self.detected_objects.copy()
    
    def get_hand_position(self) -> Dict:
        """獲取手部位置"""
        return self.hand_position.copy()
    
    def wait_for_movement_complete(self, timeout: float = 30.0) -> bool:
        """
        等待機器人移動完成
        
        Args:
            timeout: 超時時間 (秒)
            
        Returns:
            bool: 是否在超時前完成移動
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.robot_state["moving"]:
                return True
            time.sleep(0.1)
        
        return False

def main():
    """測試 ROS2 通訊模組"""
    rclpy.init()
    
    communicator = RobotCommunicator()
    
    # 使用多線程執行器
    executor = MultiThreadedExecutor()
    executor.add_node(communicator)
    
    try:
        logger.info("LLM Robot Communicator 啟動")
        logger.info("監聽機器人狀態和物件檢測...")
        
        # 測試移動指令
        test_position = [280, 18, 462, 180, 0, 90]  # mm, deg
        communicator.move_to_position(test_position)
        
        executor.spin()
        
    except KeyboardInterrupt:
        logger.info("停止 LLM Robot Communicator")
    finally:
        communicator.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()