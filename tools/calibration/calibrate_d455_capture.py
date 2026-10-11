#!/usr/bin/env python3
"""
相機校正數據採集程式 - 使用 D455 相機
按空白鍵: 同時記錄機器人座標和拍照
按 'd': 刪除最後一筆記錄
按 's': 保存所有數據
按 ESC: 退出
"""
import cv2
import json
import rclpy
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
from pathlib import Path
from ws_racr.config import get_data_dir
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetIO

# 數據保存路徑（明確標注 D455）
CAL_DATA_DIR = str(get_data_dir() / "calibration" / "calibration_data")
IMAGES_DIR = f"{CAL_DATA_DIR}/images_D455"

class CalibrationCapture:
    def __init__(self):
        # 創建保存目錄
        Path(CAL_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        
        # 初始化 ROS2
        rclpy.init()
        self.node = rclpy.create_node('calibration_capture_d455')
        
        # 訂閱機器人狀態
        self.robot_pose = None
        self.node.create_subscription(
            FeedbackState,
            'feedback_states',
            self.feedback_callback,
            10
        )
        
        # 創建 IO 控制客戶端
        self.io_client = self.node.create_client(SetIO, 'set_io')
        
        # 初始化 D455 相機
        self.init_camera()
        
        # 數據記錄
        self.capture_data = []
        self.capture_count = 0
        
        # 校正板檢測參數
        self.chessboard_size = (8, 5)  # 內角點數量 (列, 行)
        self.chessboard_detected = False
        self.chessboard_corners = None
        
        self.node.get_logger().info("📷 D455 相機校正數據採集程式已啟動")
        self.node.get_logger().info("🤖 等待機器人連接...")
        self.node.get_logger().info(f"🎯 校正板尺寸: {self.chessboard_size[0]}x{self.chessboard_size[1]} 內角點")
        
        # 初始化夾爪狀態 - 關閉夾爪
        self.init_gripper()
        
    def feedback_callback(self, msg):
        """接收機器人位姿回調"""
        self.robot_pose = {
            'position': list(msg.tool_pose[:3]),  # [X, Y, Z] in meters
            'orientation': list(msg.tool_pose[3:6])  # [RX, RY, RZ] in radians
        }
    
    def set_gripper(self, close: bool):
        """
        控制夾爪
        Args:
            close: True=關閉夾爪, False=打開夾爪
        """
        # 等待服務可用
        if not self.io_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().warn("⚠️  SetIO 服務不可用，無法控制夾爪")
            return False
        
        # 創建請求 (使用正確的常數)
        request = SetIO.Request()
        request.module = SetIO.Request.MODULE_ENDEFFECTOR  # EndModule
        request.type = SetIO.Request.TYPE_DIGITAL_OUT      # Digital Output
        request.pin = 0     # Pin 0
        request.state = float(close)  # True/False 會自動轉換為 1.0/0.0
        
        # 發送請求
        try:
            future = self.io_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.0)
            
            if future.result() is not None and future.result().ok:
                status = "關閉" if close else "打開"
                self.node.get_logger().info(f"✅ 夾爪已{status}")
                return True
            else:
                self.node.get_logger().error("❌ 夾爪控制失敗")
                return False
        except Exception as e:
            self.node.get_logger().error(f"❌ 夾爪控制異常: {e}")
            return False
    
    def init_gripper(self):
        """初始化夾爪 - 關閉夾爪"""
        self.node.get_logger().info("🤏 初始化夾爪狀態...")
        self.set_gripper(close=True)
    
    def init_camera(self):
        """初始化 D455 相機"""
        try:
            # 創建 context 並查找 D455
            ctx = rs.context()
            devices = ctx.query_devices()
            
            d455_serial = None
            for device in devices:
                name = device.get_info(rs.camera_info.name)
                if "D455" in name:
                    d455_serial = device.get_info(rs.camera_info.serial_number)
                    print(f"✅ 找到 D455 相機 (序列號: {d455_serial})")
                    break
            
            if d455_serial is None:
                raise Exception("❌ 未找到 D455 相機")
            
            # 配置 pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_device(d455_serial)
            
            # 配置串流 - 使用高解析度
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            
            # 啟動 pipeline
            self.profile = self.pipeline.start(self.config)
            
            # 獲取相機內參
            color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # 創建對齊對象
            self.align = rs.align(rs.stream.color)
            
            print("✅ D455 相機初始化完成")
            print(f"   解析度: 1280x720")
            print(f"   內參 fx: {self.intrinsics.fx:.2f}, fy: {self.intrinsics.fy:.2f}")
            print(f"   內參 ppx: {self.intrinsics.ppx:.2f}, ppy: {self.intrinsics.ppy:.2f}")
            
        except Exception as e:
            print(f"❌ 相機初始化失敗: {e}")
            raise
    
    def get_frames(self):
        """獲取相機幀"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    
    def detect_chessboard(self, color_image):
        """
        檢測校正板
        
        Args:
            color_image: 彩色影像
            
        Returns:
            (detected, corners, viz_image): 是否檢測到、角點、可視化影像
        """
        # 轉換為灰度圖
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # 檢測棋盤格角點
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # 創建可視化影像
        viz_image = color_image.copy()
        
        if ret:
            # 精細化角點位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 繪製角點
            cv2.drawChessboardCorners(viz_image, self.chessboard_size, corners_refined, ret)
            
            # 添加狀態文字
            cv2.putText(viz_image, "Chessboard Detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(viz_image, "Chessboard Detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            return True, corners_refined, viz_image
        else:
            # 添加未檢測到的提示
            cv2.putText(viz_image, "Chessboard NOT Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            cv2.putText(viz_image, "Chessboard NOT Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            return False, None, viz_image
    
    def save_capture(self, color_image, depth_image):
        """保存一筆校正數據"""
        if self.robot_pose is None:
            self.node.get_logger().warn("⚠️  機器人位姿未接收，無法保存")
            return False
        
        # 檢查是否檢測到校正板
        if not self.chessboard_detected:
            self.node.get_logger().warn("⚠️  未檢測到校正板，無法保存")
            return False
        
        # 生成時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # 保存彩色影像
        color_filename = f"color_{self.capture_count:03d}_{timestamp}.png"
        color_path = f"{IMAGES_DIR}/{color_filename}"
        cv2.imwrite(color_path, color_image)
        
        # 保存深度影像
        depth_filename = f"depth_{self.capture_count:03d}_{timestamp}.png"
        depth_path = f"{IMAGES_DIR}/{depth_filename}"
        cv2.imwrite(depth_path, depth_image)
        
        # 記錄數據
        capture_info = {
            'index': self.capture_count,
            'timestamp': timestamp,
            'color_image': color_filename,
            'depth_image': depth_filename,
            'robot_pose': self.robot_pose.copy(),
            'chessboard_detected': True,
            'chessboard_corners': self.chessboard_corners.tolist() if self.chessboard_corners is not None else None,
            'camera_intrinsics': {
                'fx': self.intrinsics.fx,
                'fy': self.intrinsics.fy,
                'ppx': self.intrinsics.ppx,
                'ppy': self.intrinsics.ppy,
                'width': self.intrinsics.width,
                'height': self.intrinsics.height,
                'model': str(self.intrinsics.model)
            }
        }
        
        self.capture_data.append(capture_info)
        self.capture_count += 1
        
        # 顯示資訊
        pos = self.robot_pose['position']
        ori = self.robot_pose['orientation']
        self.node.get_logger().info(
            f"✅ 記錄 #{self.capture_count-1}: "
            f"位置=[{pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}] mm, "
            f"姿態=[{np.rad2deg(ori[0]):.1f}°, {np.rad2deg(ori[1]):.1f}°, {np.rad2deg(ori[2]):.1f}°]"
        )
        
        return True
    
    def delete_last_capture(self):
        """刪除最後一筆記錄"""
        if not self.capture_data:
            self.node.get_logger().warn("⚠️  沒有可刪除的記錄")
            return
        
        # 獲取最後一筆記錄
        last_capture = self.capture_data.pop()
        self.capture_count -= 1
        
        # 刪除圖片文件
        try:
            color_path = f"{IMAGES_DIR}/{last_capture['color_image']}"
            depth_path = f"{IMAGES_DIR}/{last_capture['depth_image']}"
            Path(color_path).unlink(missing_ok=True)
            Path(depth_path).unlink(missing_ok=True)
            self.node.get_logger().info(f"🗑️  已刪除記錄 #{last_capture['index']}")
        except Exception as e:
            self.node.get_logger().error(f"❌ 刪除文件失敗: {e}")
    
    def save_all_data(self):
        """保存所有校正數據"""
        if not self.capture_data:
            self.node.get_logger().warn("⚠️  沒有可保存的數據")
            return False
        
        # 生成保存文件名（明確標注 D455）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        camera_id = "D455"
        json_filename = f"{CAL_DATA_DIR}/calibration_data_{camera_id}_{timestamp}.json"
        
        # 準備保存數據
        save_data = {
            'metadata': {
                'camera_model': 'Intel RealSense D455',
                'total_captures': len(self.capture_data),
                'timestamp': timestamp,
                'resolution': '1280x720'
            },
            'captures': self.capture_data
        }
        
        # 保存 JSON
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        self.node.get_logger().info(f"💾 已保存 {len(self.capture_data)} 筆校正數據到: {json_filename}")
        return True
    
    def run(self):
        """主循環"""
        cv2.namedWindow("D455 Calibration Capture", cv2.WINDOW_NORMAL)
        
        info_text = [
            "D455 Camera Calibration Capture",
            "==================================",
            f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]} corners",
            "[Space] - Capture (need board detected)",
            "[D]     - Delete Last Capture",
            "[S]     - Save All Data",
            "[ESC]   - Exit",
            ""
        ]
        
        try:
            while rclpy.ok():
                # 處理 ROS 消息
                rclpy.spin_once(self.node, timeout_sec=0.001)
                
                # 獲取相機幀
                color_image, depth_image = self.get_frames()
                if color_image is None:
                    continue
                
                # 檢測校正板
                self.chessboard_detected, self.chessboard_corners, chessboard_viz = self.detect_chessboard(color_image)
                
                # 創建深度著色圖
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # 使用校正板檢測的可視化影像
                display_image = chessboard_viz.copy()
                
                # 繪製資訊文字
                y_offset = 30
                for line in info_text:
                    cv2.putText(display_image, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(display_image, line, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 30
                
                # 顯示當前狀態
                status_y = y_offset + 10
                
                # 顯示拍攝數量
                cv2.putText(display_image, f"Captures: {self.capture_count}", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(display_image, f"Captures: {self.capture_count}", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 顯示校正板狀態
                board_status = "Board: Detected" if self.chessboard_detected else "Board: NOT Detected"
                board_color = (0, 255, 0) if self.chessboard_detected else (0, 0, 255)
                cv2.putText(display_image, board_status, (10, status_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(display_image, board_status, (10, status_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, board_color, 2)
                
                # 顯示機器人狀態
                if self.robot_pose is not None:
                    pos = self.robot_pose['position']
                    robot_text = f"Robot: [{pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}] mm"
                    cv2.putText(display_image, robot_text, (10, status_y + 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(display_image, robot_text, (10, status_y + 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.circle(display_image, (display_image.shape[1] - 30, 30), 10, (0, 255, 0), -1)
                else:
                    robot_text = "Robot: Not Connected"
                    cv2.putText(display_image, robot_text, (10, status_y + 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(display_image, robot_text, (10, status_y + 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.circle(display_image, (display_image.shape[1] - 30, 30), 10, (0, 0, 255), -1)
                
                # 水平拼接彩色和深度圖
                display_combined = np.hstack((display_image, depth_colormap))
                cv2.imshow("D455 Calibration Capture", display_combined)
                
                # 鍵盤事件
                key = cv2.waitKey(1) & 0xFF
                
                # ESC - 退出
                if key == 27:
                    break
                
                # Space - 拍照並記錄機器人座標
                elif key == 32:
                    self.save_capture(color_image, depth_image)
                
                # D - 刪除最後一筆
                elif key == ord('d') or key == ord('D'):
                    self.delete_last_capture()
                
                # S - 保存所有數據
                elif key == ord('s') or key == ord('S'):
                    if self.save_all_data():
                        self.node.get_logger().info("✅ 數據保存完成，可以退出程式")
        
        except KeyboardInterrupt:
            self.node.get_logger().info("⚠️  程式被中斷")
        
        finally:
            # 清理資源
            cv2.destroyAllWindows()
            self.pipeline.stop()
            self.node.destroy_node()
            rclpy.shutdown()
            
            self.node.get_logger().info("👋 程式已退出")

def main():
    try:
        capture = CalibrationCapture()
        capture.run()
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
