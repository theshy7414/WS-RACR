#!/usr/bin/env python3
"""
測試 Hand-Eye 校正結果
讓機器人夾著校正板，相機拍攝後計算校正板在基座座標系的位置
與機器人實際位置比較，驗證校正精度
"""
import os
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from ws_racr.config import get_data_dir
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetIO
import time
import argparse

class CalibrationTester(Node):
    def __init__(self, camera_type='D455'):
        super().__init__(f'calibration_tester_{camera_type.lower()}')
        
        # 儲存相機類型
        self.camera_type = camera_type.upper()
        
        # === TCP 到校正板的 offset ===
        # Eye-to-Hand: 棋盤夾在夾爪上，有固定的 target→gripper offset
        # 以下數值通過多次測量估計（4 次測量平均，標準差 < 2mm）
        # 在 gripper 局部座標系中的偏移
        if self.camera_type == 'D455':
            self.tcp_to_board_offset = np.array([
                0.003581,  # X
                -0.036440,  # Y
                0.324022   # Z
            ])
        else:  # D435i
            self.tcp_to_board_offset = np.array([
                -0.006140,  # X
                0.041179,   # Y
                0.326179    # Z
            ])
        
        # 用於估計 offset 的測量記錄
        self.offset_measurements = []
        
        print("=" * 60)
        print(f"🔍 載入校正結果 ({self.camera_type})...")
        print("=" * 60)
        print(f"\n⚙️  TCP→校正板 offset: [{self.tcp_to_board_offset[0]*1000:.1f}, {self.tcp_to_board_offset[1]*1000:.1f}, {self.tcp_to_board_offset[2]*1000:.1f}] mm")
        print(f"   ⚠️  如果誤差很大，請修改程式中的 tcp_to_board_offset 參數！")
        
        # 載入校正文件
        CAL_DIR = str(get_data_dir() / "calibration" / "cal_info")
        
        if self.camera_type == 'D455':
            # D455 校正文件
            self.T_cam2base = np.load(f"{CAL_DIR}/extrinsics_D455.npy")
            intrinsics_data = np.load(f"{CAL_DIR}/intrinsics_D455.npz")
            self.K = intrinsics_data['K']
            self.dist = intrinsics_data['dist'].reshape(1, -1)
        else:  # D435i
            # D435i 校正文件
            try:
                self.T_cam2base = np.load(f"{CAL_DIR}/extrinsics_D435i.npy")
            except FileNotFoundError:
                # 如果沒有 D435i 後綴，嘗試舊命名
                self.T_cam2base = np.load(f"{CAL_DIR}/extrinsics.npy")
            
            try:
                intrinsics_data = np.load(f"{CAL_DIR}/intrinsics_D435i.npz")
                self.K = intrinsics_data['K']
                self.dist = intrinsics_data['dist'].reshape(1, -1)
            except:
                # 嘗試舊格式
                self.K = np.load(f"{CAL_DIR}/intrinsics.npy", allow_pickle=True)
                self.dist = np.zeros((1, 5), dtype=np.float64)
        
        # 轉換為 4x4 矩陣
        if self.T_cam2base.shape == (3, 4):
            T_full = np.eye(4)
            T_full[:3, :] = self.T_cam2base
            self.T_cam2base = T_full
        
        print(f"\n📐 相機→基座變換矩陣 ({self.camera_type}):")
        print(self.T_cam2base)
        
        print(f"\n📷 相機內參:")
        print(f"   fx: {self.K[0,0]:.2f}, fy: {self.K[1,1]:.2f}")
        print(f"   ppx: {self.K[0,2]:.2f}, ppy: {self.K[1,2]:.2f}")
        
        # 棋盤格設定（與校正時相同）
        self.CHECKERBOARD = (8, 5)
        self.SQUARE = 0.025  # 25mm
        
        print(f"\n🎯 棋盤格設定:")
        print(f"   尺寸: {self.CHECKERBOARD[0]}x{self.CHECKERBOARD[1]} 內角點")
        print(f"   方格邊長: {self.SQUARE*1000:.1f} mm")
        
        # 建立世界座標
        self.objp = np.zeros((self.CHECKERBOARD[0]*self.CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= self.SQUARE
        
        # 初始化 D455 相機
        self.init_camera()
        
        # 訂閱機器人狀態
        self.robot_pose = None
        self.robot_connected = False
        self.subscription = self.create_subscription(
            FeedbackState,
            'feedback_states',
            self.robot_callback,
            10
        )
        
        # 建立 SetIO 服務客戶端（用於控制夾爪）
        self.set_io_client = self.create_client(SetIO, 'set_io')
        
        print(f"\n⏳ 等待機器人連接...")
        
        # 等待服務可用並初始化夾爪
        self.init_gripper()
        
    def init_camera(self):
        """初始化相機"""
        print(f"\n📷 初始化 {self.camera_type} 相機...")
        
        try:
            # 尋找指定相機
            ctx = rs.context()
            devices = ctx.query_devices()
            camera_serial = None
            
            for dev in devices:
                name = dev.get_info(rs.camera_info.name)
                if self.camera_type == 'D455' and "D455" in name:
                    camera_serial = dev.get_info(rs.camera_info.serial_number)
                    print(f"   找到 D455: {camera_serial}")
                    break
                elif self.camera_type == 'D435I' and ("D435I" in name or "D435i" in name):
                    camera_serial = dev.get_info(rs.camera_info.serial_number)
                    print(f"   找到 D435i: {camera_serial}")
                    break
            
            if camera_serial is None:
                raise Exception(f"找不到 {self.camera_type} 相機")
            
            # 配置 pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_device(camera_serial)
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            
            # 啟動
            self.pipeline.start(self.config)
            
            print(f"   ✅ {self.camera_type} 相機初始化完成")
            
        except Exception as e:
            print(f"   ❌ 相機初始化失敗: {e}")
            raise
    
    def robot_callback(self, msg):
        """接收機器人位姿"""
        if not self.robot_connected:
            self.robot_connected = True
            print(f"   ✅ 機器人已連接")
        
        self.robot_pose = {
            'position': np.array([
                msg.tool_pose[0],  # X (m)
                msg.tool_pose[1],  # Y (m)
                msg.tool_pose[2]   # Z (m)
            ]),
            'orientation': np.array([
                msg.tool_pose[3],  # RX (rad)
                msg.tool_pose[4],  # RY (rad)
                msg.tool_pose[5]   # RZ (rad)
            ])
        }
    
    def init_gripper(self):
        """初始化夾爪 - 關閉夾爪"""
        print(f"\n🤖 初始化夾爪...")
        
        # 等待服務可用
        timeout_sec = 5.0
        if not self.set_io_client.wait_for_service(timeout_sec=timeout_sec):
            print(f"   ⚠️  SetIO 服務未就緒 (等待 {timeout_sec} 秒)")
            print(f"   夾爪控制可能不可用")
            return
        
        # 關閉夾爪
        self.set_gripper(True)
    
    def set_gripper(self, close=True):
        """
        控制夾爪開關
        Args:
            close: True=關閉夾爪, False=打開夾爪
        """
        try:
            request = SetIO.Request()
            request.module = SetIO.Request.MODULE_ENDEFFECTOR
            request.type = SetIO.Request.TYPE_DIGITAL_OUT
            request.pin = 0
            request.state = 1.0 if close else 0.0
            
            future = self.set_io_client.call_async(request)
            
            # 等待服務回應
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.result() is not None:
                action = "關閉" if close else "打開"
                print(f"   ✅ 夾爪已{action}")
                return True
            else:
                print(f"   ⚠️  SetIO 服務無回應")
                return False
                
        except Exception as e:
            print(f"   ❌ 夾爪控制失敗: {e}")
            return False
    
    def capture_and_test(self):
        """拍攝影像並測試校正精度"""
        if not self.robot_connected:
            print("❌ 機器人未連接！")
            return False
        
        print(f"\n" + "=" * 60)
        print("📸 開始測試...")
        print("=" * 60)
        
        # 拍攝影像
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            print("❌ 無法獲取影像")
            return False
        
        color_image = np.asanyarray(color_frame.get_data())
        
        # 檢測棋盤格
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
        
        ret, corners = cv2.findChessboardCorners(
            gray, self.CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not ret:
            print("❌ 未檢測到棋盤格！")
            print("   請確保:")
            print("   1. 棋盤格在相機視野內")
            print("   2. 光線充足")
            print("   3. 棋盤格平整且清晰")
            return False
        
        # 精細化角點
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print(f"✅ 檢測到棋盤格 ({len(corners)} 個角點)")
        
        # === 1) 使用 solvePnP 計算棋盤→相機的變換 ===
        ret, rvec, tvec = cv2.solvePnP(
            self.objp, corners, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not ret:
            print("❌ solvePnP 失敗")
            return False
        
        R_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = tvec.flatten()
        
        # 組裝 T_target2cam
        T_target2cam = np.eye(4)
        T_target2cam[:3, :3] = R_target2cam
        T_target2cam[:3, 3] = t_target2cam
        
        print(f"\n📐 棋盤→相機變換:")
        print(f"   位置: [{t_target2cam[0]*1000:.1f}, {t_target2cam[1]*1000:.1f}, {t_target2cam[2]*1000:.1f}] mm")
        
        # === 2) 使用校正結果計算棋盤→基座的變換 ===
        # Eye-to-Hand: T_target2base = T_cam2base @ T_target2cam
        T_target2base_from_camera = self.T_cam2base @ T_target2cam
        
        position_computed = T_target2base_from_camera[:3, 3]
        
        print(f"\n🔄 計算的棋盤位置 (從相機觀測):")
        print(f"   X: {position_computed[0]*1000:.2f} mm")
        print(f"   Y: {position_computed[1]*1000:.2f} mm")
        print(f"   Z: {position_computed[2]*1000:.2f} mm")
        
        # === 3) 從機器人位姿計算棋盤位置 ===
        robot_tcp_position = self.robot_pose['position']
        robot_orientation = self.robot_pose['orientation']
        
        print(f"\n🤖 機器人 TCP (gripper) 位置:")
        print(f"   X: {robot_tcp_position[0]*1000:.2f} mm")
        print(f"   Y: {robot_tcp_position[1]*1000:.2f} mm")
        print(f"   Z: {robot_tcp_position[2]*1000:.2f} mm")
        print(f"   RX: {np.rad2deg(robot_orientation[0]):.2f}°")
        print(f"   RY: {np.rad2deg(robot_orientation[1]):.2f}°")
        print(f"   RZ: {np.rad2deg(robot_orientation[2]):.2f}°")
        
        # 建立 gripper→base 的變換矩陣
        rx, ry, rz = robot_orientation[0], robot_orientation[1], robot_orientation[2]
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        R_gripper2base = Rz @ Ry @ Rx
        
        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_gripper2base
        T_gripper2base[:3, 3] = robot_tcp_position
        
        # Eye-to-Hand: 棋盤夾在夾爪上
        # T_target2base = T_target2gripper @ T_gripper2base
        # 其中 T_target2gripper 是固定的 offset（棋盤在夾爪中的位置）
        
        # 簡化假設：棋盤中心在 TCP 中心（offset = 0）
        # 如果有實際 offset，需要在這裡加上
        T_target2gripper = np.eye(4)
        T_target2gripper[:3, 3] = self.tcp_to_board_offset  # 使用設定的 offset
        
        T_target2base_from_robot = T_gripper2base @ T_target2gripper
        position_from_robot = T_target2base_from_robot[:3, 3]
        
        print(f"\n📋 棋盤位置 (從機器人位姿計算):")
        print(f"   X: {position_from_robot[0]*1000:.2f} mm")
        print(f"   Y: {position_from_robot[1]*1000:.2f} mm")
        print(f"   Z: {position_from_robot[2]*1000:.2f} mm")
        print(f"   (假設 target→gripper offset = [{self.tcp_to_board_offset[0]*1000:.1f}, {self.tcp_to_board_offset[1]*1000:.1f}, {self.tcp_to_board_offset[2]*1000:.1f}] mm)")
        
        # === 4) 計算誤差 ===
        position_error = position_computed - position_from_robot
        distance_error = np.linalg.norm(position_error)
        
        print(f"\n📊 誤差分析:")
        print(f"   ΔX: {position_error[0]*1000:+.2f} mm")
        print(f"   ΔY: {position_error[1]*1000:+.2f} mm")
        print(f"   ΔZ: {position_error[2]*1000:+.2f} mm")
        print(f"   總誤差: {distance_error*1000:.2f} mm")
        
        # 評估精度
        # === 5) 估計 target→gripper offset ===
        # 從誤差反推 offset（在 gripper 局部座標系）
        # position_computed = T_target2gripper @ T_gripper2base
        # position_from_robot = T_gripper2base
        # 所以 T_target2gripper ≈ position_computed @ inv(T_gripper2base)
        
        T_gripper2base_inv = np.linalg.inv(T_gripper2base)
        estimated_offset_in_gripper = T_gripper2base_inv[:3, :3] @ position_error
        
        self.offset_measurements.append({
            'offset_in_base': position_error,
            'offset_in_gripper': estimated_offset_in_gripper
        })
        
        print(f"\n📏 估計的 target→gripper offset (gripper 局部座標系):")
        print(f"   X: {estimated_offset_in_gripper[0]*1000:+.2f} mm")
        print(f"   Y: {estimated_offset_in_gripper[1]*1000:+.2f} mm")
        print(f"   Z: {estimated_offset_in_gripper[2]*1000:+.2f} mm")
        
        if len(self.offset_measurements) >= 3:
            # 計算平均 offset
            offsets = np.array([m['offset_in_gripper'] for m in self.offset_measurements])
            mean_offset = np.mean(offsets, axis=0)
            std_offset = np.std(offsets, axis=0)
            
            print(f"\n� 多次測量統計 ({len(self.offset_measurements)} 次):")
            print(f"   平均 offset: [{mean_offset[0]*1000:+.1f}, {mean_offset[1]*1000:+.1f}, {mean_offset[2]*1000:+.1f}] mm")
            print(f"   標準差: [{std_offset[0]*1000:.1f}, {std_offset[1]*1000:.1f}, {std_offset[2]*1000:.1f}] mm")
            print(f"\n💡 建議：將以下代碼加入 test_calibration.py 第 17 行:")
            print(f"   self.tcp_to_board_offset = np.array([")
            print(f"       {mean_offset[0]:.6f},  # X")
            print(f"       {mean_offset[1]:.6f},  # Y")
            print(f"       {mean_offset[2]:.6f}   # Z")
            print(f"   ])")
        
        print(f"\n�📈 精度評估:")
        if distance_error < 0.005:  # < 5mm
            print(f"   ✅ 優秀 (< 5mm)")
        elif distance_error < 0.010:  # < 10mm
            print(f"   ✅ 良好 (5-10mm)")
        elif distance_error < 0.020:  # < 20mm
            print(f"   ⚠️  可接受 (10-20mm)")
        else:
            print(f"   ⚠️  誤差較大 (> 20mm)")
            print(f"   💡 這個誤差可能是 target→gripper 的固定 offset")
            print(f"   📝 請在不同位置多測試幾次（至少 3 次）")
            print(f"   📝 然後使用上面建議的平均 offset 值")
        
        # === 5) 可視化 ===
        img_viz = color_image.copy()
        
        # 畫檢測到的角點
        cv2.drawChessboardCorners(img_viz, self.CHECKERBOARD, corners, ret)
        
        # 畫座標軸
        axis = np.float32([
            [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05], [0, 0, 0]
        ]).reshape(-1, 3)
        
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.K, self.dist)
        imgpts = imgpts.astype(int)
        
        origin = tuple(imgpts[3].ravel())
        img_viz = cv2.line(img_viz, origin, tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # X軸-紅
        img_viz = cv2.line(img_viz, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # Y軸-綠
        img_viz = cv2.line(img_viz, origin, tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # Z軸-藍
        
        # 顯示資訊
        cv2.putText(img_viz, f"Error: {distance_error*1000:.2f} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img_viz, f"From Camera: [{position_computed[0]*1000:.0f}, {position_computed[1]*1000:.0f}, {position_computed[2]*1000:.0f}] mm", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_viz, f"From Robot: [{position_from_robot[0]*1000:.0f}, {position_from_robot[1]*1000:.0f}, {position_from_robot[2]*1000:.0f}] mm", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img_viz, f"Target->Gripper offset: [{self.tcp_to_board_offset[0]*1000:.0f}, {self.tcp_to_board_offset[1]*1000:.0f}, {self.tcp_to_board_offset[2]*1000:.0f}] mm", 
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 顯示和保存
        cv2.namedWindow("Calibration Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Test", 1280, 720)
        cv2.imshow("Calibration Test", img_viz)
        
        output_path = "./calibration_test_result.png"
        cv2.imwrite(output_path, img_viz)
        print(f"\n💾 測試結果已保存: {output_path}")
        
        return True
    
    def run(self):
        """主循環"""
        print(f"2. 確保校正板在 {self.camera_type} 相機視野內")
        print("3. 按 Space 鍵進行測試")
        print("4. 按 'O' 鍵打開夾爪，按 'C' 鍵關閉夾爪")
        print("5. 按 ESC 鍵退出")
        print("=" * 60)
        
        window_name = f"{self.camera_type} Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        
        try:
            while rclpy.ok():
                # 處理 ROS 訊息
                rclpy.spin_once(self, timeout_sec=0.01)
                
                # 獲取影像（增加超時時間並加入錯誤處理）
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        continue
                    
                    color_image = np.asanyarray(color_frame.get_data())
                    
                except RuntimeError as e:
                    # 如果相機超時，跳過這一幀
                    continue
                except Exception as e:
                    # 處理其他可能的錯誤
                    continue
                
                # 顯示資訊
                img_display = color_image.copy()
                
                # 機器人連接狀態
                status_color = (0, 255, 0) if self.robot_connected else (0, 0, 255)
                status_text = "Robot Connected" if self.robot_connected else "Robot NOT Connected"
                cv2.circle(img_display, (1250, 30), 10, status_color, -1)
                cv2.putText(img_display, status_text, (1050, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                window_name
                # 操作提示
                cv2.putText(img_display, "SPACE:Test | O:Open | C:Close | ESC:Exit", (10, 690),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("D455 Camera", img_display)
                
                # 處理按鍵
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\n👋 退出測試")
                    break
                elif key == ord(' '):  # Space
                    self.capture_and_test()
                    print("\n按任意鍵繼續...")
                    cv2.waitKey(0)
                elif key == ord('o') or key == ord('O'):  # O - 打開夾爪
                    print("\n🔓 打開夾爪...")
                    self.set_gripper(False)
                elif key == ord('c') or key == ord('C'):  # C - 關閉夾爪
                    print("\n🔒 關閉夾爪...")
                    self.set_gripper(True)
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():# 解析命令行參數
    parser = argparse.ArgumentParser(description='Hand-Eye 校正測試程式')
    parser.add_argument('--camera', type=str, default='D455', 
                       choices=['D455', 'D435i', 'D435I'],
                       help='選擇相機類型: D455 或 D435i (預設: D455)')
    
    args = parser.parse_args()
    
    # 統一相機名稱格式
    camera_type = 'D435I' if args.camera.upper() in ['D435I', 'D435I'] else args.camera.upper()
    
    print("\n" + "=" * 60)
    print(f"🎥 相機模式: {camera_type}")
    print("=" * 60)
    
    rclpy.init()
    
    try:
        tester = CalibrationTester(camera_type=camera_type)
        tester.run()
    except KeyboardInterrupt:
        print("\n👋 程式中斷")
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()