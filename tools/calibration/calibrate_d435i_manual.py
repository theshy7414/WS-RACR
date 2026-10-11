#!/usr/bin/env python3
import os
import cv2
import math
import rclpy
import numpy as np
import argparse
import sys
from pathlib import Path

from ws_racr.config import get_data_dir
from ws_racr.perception.realsense_camera import RealSenseCamera
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SetPositions

CAL_DIR = str(get_data_dir() / "calibration" / "cal_info")
EXTRINSICS_PATH = f"{CAL_DIR}/extrinsics_D435i.npy"
INTRINSICS_PATH = f"{CAL_DIR}/intrinsics_D435i.npz"
REF_POINTS_PATH = f"{CAL_DIR}/ref_points_D435i.npy"
IMAGE_PATH = f"{CAL_DIR}/ref_points_D435i.png"

PI = math.pi
ORIGIN = [0.291, -0.225, 0.613]  # (m) - 按r鍵時機器人要移動到的位置 [X, Y, Z]
ANGLE = [PI, 0.0, PI/2] # (rad) - 按r鍵時機器人要移動到的角度 [RX, RY, RZ]

# Step 1
def get_reference_points():
    Path(CAL_DIR).mkdir(parents = True, exist_ok = True)

    point_text1 = "Image uv:"
    point_text2 = "Camera xyz:"
    click_point = (0, 0)
    click_point_cam = (0, 0, 0)
    ref_points = []
    ref_points_3d = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x, y)

    rs_cam = RealSenseCamera()
    cv2.namedWindow("Window")
    cv2.setMouseCallback("Window", mouse_callback)
    try:
        while True:
            rs_cam.wait_for_frames()
            viz_img = rs_cam.color_image.copy()
            
            # 影像座標 (x, y, z)
            u, v = click_point
            click_point_cam = rs_cam.deproject_image_to_camera(u, v)
            
            # 轉換成相機座標 (X, Y, Z)
            if click_point_cam is not None:
                x, y, z = click_point_cam
                x *= 100; y *= 100; z *= 100
                point_text1 = f"Image uv: ({u}, {v}) px"
                point_text2 = f"Camera xyz: ({x:.2f}, {y:.2f}, {z:.2f}) cm"
            else:
                point_text1 = "Image uv: N/A"
                point_text2 = "Camera xyz: N/A"
            
            # 顯示滑鼠點擊位置
            cv2.putText(viz_img, point_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(viz_img, point_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(viz_img, point_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(viz_img, point_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.drawMarker(viz_img, click_point, (0, 255, 0), markerType = cv2.MARKER_CROSS, markerSize = 10)
            # 畫出參考點
            for i, point in enumerate(ref_points):
                x, y = point
                cv2.putText(viz_img, str(i), (x+7, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(viz_img, point, 3, (0, 255, 0), 2)

            cv2.imshow("Window", viz_img)
            key = cv2.waitKey(1)
            # ESC
            if key == 27:
                break
            # Space: Append 相機座標
            elif key == 32:
                if click_point_cam is not None and click_point_cam[2] > 0:  # 確保有效的深度值
                    ref_points.append(click_point)
                    ref_points_3d.append(click_point_cam)

                    # 畫出參考點
                    x, y = click_point
                    cv2.putText(viz_img, str(len(ref_points)-1), (x+7, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(viz_img, (x, y), 3, (0, 255, 0), 2)
                    
                    print(f"Point #{len(ref_points)-1} - Camera xyz: {x:.2f}, {y:.2f}, {z:.2f}")
                else:
                    print("Warning: Invalid depth value, point not added")
            # D key: Delete 最後一筆資料
            elif key == ord('d') and ref_points:
                ref_points.pop()
                ref_points_3d.pop()
            elif key == ord('s') and ref_points:
                if len(ref_points) < 4:
                    print(f"Warning: Need at least 4 points for calibration, current: {len(ref_points)}")
                    continue
                print("cal_points:")
                for i, point in enumerate(ref_points_3d):
                    print(f"Point #{i}: {[round(j,2) for j in point]}")
                print(f"Save image to {IMAGE_PATH}")
                print(f"Save reference points to {REF_POINTS_PATH}")
                cv2.imwrite(IMAGE_PATH, viz_img)
                np.save(REF_POINTS_PATH, ref_points_3d)
                break
                
    finally:
        rs_cam.release()
        cv2.destroyAllWindows()

# Step 2
def calculate_extrinsics():
    
    def feedback_states_callback(msg):
        nonlocal object_point
        object_point = [i for i in msg.tool_pose[:3]]

    camera_points = np.load(REF_POINTS_PATH)
    object_point = None
    object_points = []

    img = cv2.imread(IMAGE_PATH)

    try:
        # ROS
        rclpy.init()
        # Node
        node = rclpy.create_node('feedback_subscriber')
        node.create_subscription(
            FeedbackState,
            'feedback_states',
            feedback_states_callback,
            10
        )
        i = 0
        while rclpy.ok():
            node.get_logger().info('Press space to save object point...')

            # 顯示翻轉的圖片（水平翻轉）
            img_display = cv2.flip(img, -1)
            cv2.imshow('Reference points', img_display)
            key = cv2.waitKey(0)

            if key == 32:
                rclpy.spin_once(node)
                object_points.append(object_point)
                node.get_logger().info(f"Object Point #{i}: {[round(i*1000,2) for i in object_point]} (mm)")
                i += 1
            elif key == ord('d') and object_point:
                i -= 1
                object_points.pop()
                node.get_logger().info(f'Delete object point #{i}')
            elif key == ord('s'):
                if len(object_points) != len(camera_points):
                    node.get_logger().info(f'Object Points Length: {len(object_points)}')
                    node.get_logger().info(f'Camera Point Length: {len(camera_points)}')
                    continue
                node.get_logger().info('Camera Points:')
                for i, pt in enumerate(camera_points):
                    node.get_logger().info(f'#{i}:{[round(i*1000,2) for i in pt]}')
                node.get_logger().info('Object Points:')
                for i, pt in enumerate(object_points):
                    node.get_logger().info(f'#{i}:{[round(i*1000,2) for i in pt]}')
                
                object_points = np.array(object_points)
                ret, extrinsics, inliers = cv2.estimateAffine3D(camera_points, object_points)

                if ret:
                    # 計算重投影誤差
                    camera_points_hom = np.hstack([camera_points, np.ones((len(camera_points), 1))])
                    projected_points = camera_points_hom @ extrinsics.T
                    errors = np.linalg.norm(projected_points - object_points, axis=1)
                    mean_error = np.mean(errors)
                    max_error = np.max(errors)
                    
                    node.get_logger().info(f"Calibration successful!")
                    node.get_logger().info(f"Mean reprojection error: {mean_error*1000:.2f} mm")
                    node.get_logger().info(f"Max reprojection error: {max_error*1000:.2f} mm")
                    node.get_logger().info(f"Inliers: {np.sum(inliers)}/{len(inliers)}")
                    
                    print("Extrinsics matrix:")
                    print(extrinsics)
                    print(f"Save to {EXTRINSICS_PATH}")
                    np.save(EXTRINSICS_PATH, extrinsics)
                else:
                    node.get_logger().error("Failed to estimate extrinsics!")
                    continue 
                break
            elif key == 27:
                node.get_logger().info('Cancel calculating extr!')
                break
            
    finally:
        node.destroy_node()
        rclpy.shutdown()

# Step 3
def test():
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = [x, y]
    
    click_point = None
    click_point_world = None

    # Extrinsics
    extrinsics = np.load(EXTRINSICS_PATH)

    # Realsense
    rs_cam = RealSenseCamera(extrinsics=extrinsics)
    cv2.namedWindow("Click2move")
    cv2.setMouseCallback("Click2move", mouse_callback)

    try:
        # ROS
        rclpy.init()
        # Node
        node = rclpy.create_node('sendscript_client')
        client = node.create_client(SetPositions, 'set_positions')
        request = SetPositions.Request()
        request.motion_type = request.PTP_T
        request.velocity = 1.0
        request.acc_time = 0.2
        request.blend_percentage = 10
        request.fine_goal = False

        while not client.wait_for_service(1.0):
            node.get_logger().info('SendScript service not available, waiting...')

        while rclpy.ok():
            rs_cam.wait_for_frames()
            viz_img = rs_cam.color_image.copy()

            if click_point is not None:
                u, v = click_point
                click_point_world = rs_cam.deproject_image_to_world(u, v)
                x, y, z = click_point_world
                point_text1 = f"Image uv: ({u}, {v}) px"
                point_text2 = f"World XYZ: ({x*1000:.2f}, {y*1000:.2f}, {z*1000:.2f}) mm"
            else:
                point_text1 = "Image uv: N/A"
                point_text2 = "World XYZ: N/A"
            
            cv2.putText(viz_img, point_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(viz_img, point_text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(viz_img, point_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(viz_img, point_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.drawMarker(viz_img, click_point, (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

            cv2.imshow("Click2move", viz_img)
            key = cv2.waitKey(1)

            if key == 27:
                node.get_logger().info('Cancel testing!')
                break
            elif key == 32 and click_point_world is not None:
                node.get_logger().info('Move to click point!')
                positions = click_point_world.copy() + ANGLE
                request.positions = positions
                client.call_async(request)
                rclpy.spin_once(node)
            elif key == ord('r'):
                node.get_logger().info('Reset robot arm...')
                positions = ORIGIN + ANGLE
                request.positions = positions
                client.call_async(request)
                rclpy.spin_once(node)
    finally:
        rs_cam.release()
        node.destroy_node()
        rclpy.shutdown()

def main():
    """主函數：解析命令行參數並執行對應的步驟"""
    parser = argparse.ArgumentParser(description='D435i Hand-Eye 校正程式')
    parser.add_argument('--step', type=int, required=True, choices=[1, 2, 3],
                       help='選擇執行步驟: 1=獲取參考點, 2=計算外參, 3=測試校正')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(f"📸 D435i Hand-Eye 校正")
    print("=" * 60)
    
    if args.step == 1:
        print("\n📝 步驟 1: 獲取參考點")
        print("操作說明:")
        print("- 點擊影像中的特定點")
        print("- 按 Space: 保存當前點")
        print("- 按 D: 刪除最後一個點")
        print("- 按 S: 保存所有點並退出（至少需要 4 個點）")
        print("- 按 ESC: 取消並退出")
        print("=" * 60 + "\n")
        get_reference_points()
        
    elif args.step == 2:
        print("\n🔧 步驟 2: 計算外參矩陣")
        print("操作說明:")
        print("- 將機器人 TCP 移動到參考點位置")
        print("- 按 Space: 記錄當前機器人位置")
        print("- 按 D: 刪除最後一個位置")
        print("- 按 S: 計算並保存外參矩陣")
        print("- 按 ESC: 取消並退出")
        print("=" * 60 + "\n")
        calculate_extrinsics()
        
    elif args.step == 3:
        print("\n✅ 步驟 3: 測試校正結果")
        print("操作說明:")
        print("- 點擊影像中的位置")
        print("- 按 Space: 移動機器人到點擊位置")
        print("- 按 R: 重置機器人到原點")
        print("- 按 ESC: 退出")
        print("=" * 60 + "\n")
        test()
    
    print("\n✅ 完成！\n")

if __name__ == '__main__':
    main()