#!/usr/bin/env python3
"""
桌面平面校準工具

使用 RealSense depth + RANSAC 擬合桌面平面
比基於物體 Z 值的方法更穩健

使用方式：
1. 運行此腳本
2. 在畫面上框選桌面區域（排除機器人、物體）
3. 按 's' 保存 ROI 配置
4. world_state_builder 會自動載入並使用
"""

import logging
import cv2
import numpy as np
import pyrealsense2 as rs
import json
from pathlib import Path
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class TablePlaneCalibrator:
    """桌面平面校準器"""
    
    def __init__(self, config_path: str = "data/calibration/table_roi.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # RealSense pipeline
        self.pipeline = None
        self.align = None
        
        # ROI selection state
        self.roi_points = []  # 多邊形頂點列表 [(x1,y1), (x2,y2), ...]
        self.roi_mask = None  # 二值化 mask
        self.roi = None  # (x1, y1, x2, y2) - bounding box (相容性)
        
        # Plane parameters (ax + by + cz + d = 0)
        self.plane_normal = None  # [a, b, c]
        self.plane_d = None
        self.plane_reference_z = None  # Z at origin (for simpler checking)
        
    def start_realsense(self):
        """啟動 RealSense D435i"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams (與系統其他模組一致)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # Start pipeline
        profile = self.pipeline.start(config)
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Warm up (丟棄前幾幀)
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        logger.info("RealSense D435i 已啟動")
    
    def stop_realsense(self):
        """停止 RealSense"""
        if self.pipeline:
            self.pipeline.stop()
    
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠回調：點擊選擇多邊形 ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加頂點
            self.roi_points.append((x, y))
            logger.info(f"點 {len(self.roi_points)}: ({x}, {y})")
    
    def select_roi_interactive(self):
        """互動式選擇桌面 ROI"""
        logger.info("\n=== 桌面 ROI 選擇（多邊形模式）===")
        logger.info("1. 點擊桌面的多個角點（至少 3 點，按順序點擊）")
        logger.info("2. 按 空白鍵 完成選擇")
        logger.info("3. 按 'r' 重新選擇")
        logger.info("4. 按 's' 保存 ROI")
        logger.info("5. 按 'q' 退出\n")
        
        cv2.namedWindow("Select Table ROI")
        callback_param = {'current_point': None}
        cv2.setMouseCallback("Select Table ROI", self.mouse_callback, callback_param)
        
        while True:
            # 獲取幀
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # 轉換為 numpy
            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()
            
            # 繪製已選擇的點
            if len(self.roi_points) > 0:
                for i, pt in enumerate(self.roi_points):
                    cv2.circle(display, pt, 5, (0, 255, 0), -1)
                    cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 繪製連線
                if len(self.roi_points) > 1:
                    pts = np.array(self.roi_points, dtype=np.int32)
                    cv2.polylines(display, [pts], False, (0, 255, 0), 2)
            
            # 繪製完成的多邊形
            if self.roi_mask is not None:
                # 半透明顯示 mask
                mask_colored = cv2.cvtColor(self.roi_mask, cv2.COLOR_GRAY2BGR)
                mask_colored[:, :, 1] = self.roi_mask  # 綠色通道
                display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)
                cv2.putText(display, "ROI confirmed (press 's' to save)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Select Table ROI", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and len(self.roi_points) >= 3:
                # 空白鍵：完成選擇
                self._finalize_roi(color_image.shape[:2])
            elif key == ord('s') and self.roi_mask is not None:
                # 's' 鍵：保存 ROI
                self.save_roi()
                logger.info("ROI 已保存")
                break
            elif key == ord('r'):
                # 'r' 鍵：重新選擇
                self.roi_points = []
                self.roi_mask = None
                self.roi = None
                logger.info("重新選擇 ROI")
            elif key == ord('q'):
                logger.info("取消選擇")
                break
        
        cv2.destroyAllWindows()
    
    def _finalize_roi(self, image_shape: Tuple[int, int]):
        """完成 ROI 選擇，建立 mask"""
        if len(self.roi_points) < 3:
            logger.warning("至少需要 3 個點")
            return
        
        # 建立二值化 mask
        h, w = image_shape
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.roi_points, dtype=np.int32)
        cv2.fillPoly(self.roi_mask, [pts], 255)
        
        # 計算 bounding box (相容性)
        x_coords = [pt[0] for pt in self.roi_points]
        y_coords = [pt[1] for pt in self.roi_points]
        self.roi = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        logger.info(f"ROI 已完成：{len(self.roi_points)} 個點，面積 = {np.sum(self.roi_mask > 0)} pixels")
    
    def fit_plane_ransac(self, depth_frame, color_frame, n_samples: int = 1000) -> bool:
        """
        用 RANSAC 擬合桌面平面
        
        Args:
            depth_frame: Depth frame
            color_frame: Color frame (用於獲取 intrinsics)
            n_samples: 採樣點數
        
        Returns:
            是否成功擬合
        """
        if self.roi_mask is None:
            logger.warning("請先選擇 ROI")
            return False
        
        # 獲取 depth intrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # 從 ROI mask 中採樣深度點
        mask_coords = np.argwhere(self.roi_mask > 0)  # Shape: (N, 2) - (y, x)
        if len(mask_coords) < 100:
            logger.warning(f"ROI 太小：{len(mask_coords)} pixels")
            return False
        
        # 隨機採樣
        sample_indices = np.random.choice(len(mask_coords), min(n_samples, len(mask_coords)), replace=False)
        sampled_coords = mask_coords[sample_indices]
        
        points_3d = []
        for y, x in sampled_coords:
            depth = depth_frame.get_distance(int(x), int(y))
            if depth > 0.1 and depth < 2.0:  # 有效深度範圍 10cm ~ 2m
                # 反投影到 3D
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics, [int(x), int(y)], depth
                )
                points_3d.append(point_3d)
        
        if len(points_3d) < 100:
            logger.warning(f"有效點太少: {len(points_3d)}")
            return False
        
        points_3d = np.array(points_3d)  # Shape: (N, 3)
        
        # RANSAC 擬合平面
        best_inliers = 0
        best_plane = None
        n_iterations = 200
        threshold = 0.01  # 1cm inlier threshold
        
        for _ in range(n_iterations):
            # 隨機選 3 個點
            idx = np.random.choice(len(points_3d), 3, replace=False)
            p1, p2, p3 = points_3d[idx]
            
            # 計算平面法向量
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            if norm < 1e-6:
                continue
            
            normal = normal / norm  # 單位化
            
            # 平面方程: ax + by + cz + d = 0
            # normal = [a, b, c], d = -dot(normal, p1)
            d = -np.dot(normal, p1)
            
            # 計算所有點到平面的距離
            distances = np.abs(np.dot(points_3d, normal) + d)
            
            # 統計 inliers
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (normal, d)
        
        if best_plane is None:
            logger.warning("RANSAC 擬合失敗")
            return False
        
        self.plane_normal, self.plane_d = best_plane
        
        # 計算參考點的 Z（相機原點在平面上的投影）
        # 對於點 (0, 0, z)，代入平面方程：c*z + d = 0 → z = -d/c
        if abs(self.plane_normal[2]) > 1e-6:
            self.plane_reference_z = -self.plane_d / self.plane_normal[2]
        else:
            # 平面幾乎垂直（不太可能）
            self.plane_reference_z = 0.6  # 默認 60cm
        
        # 轉換為 mm
        self.plane_reference_z *= 1000
        
        inlier_ratio = best_inliers / len(points_3d)
        logger.info(f"平面擬合成功:")
        logger.info(f"   法向量: {self.plane_normal}")
        logger.info(f"   d: {self.plane_d:.4f}")
        logger.info(f"   參考 Z: {self.plane_reference_z:.1f} mm")
        logger.info(f"   Inlier 比例: {inlier_ratio:.1%}")
        
        return True
    
    def calibrate_live(self, duration: float = 5.0):
        """
        即時校準（連續擬合多次取平均）
        
        Args:
            duration: 校準持續時間（秒）
        """
        if not self.roi:
            logger.warning("請先選擇 ROI")
            return
        
        logger.info(f"\n=== 開始校準（持續 {duration} 秒）===")
        logger.info("請確保桌面清空，無遮擋物...\n")
        
        start_time = cv2.getTickCount()
        reference_z_list = []
        
        while True:
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed > duration:
                break
            
            # 獲取幀
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 擬合平面
            if self.fit_plane_ransac(depth_frame, color_frame, n_samples=500):
                reference_z_list.append(self.plane_reference_z)
                logger.info(f"  [{elapsed:.1f}s] Z = {self.plane_reference_z:.1f} mm")
        
        # 取中位數（更穩健）
        if reference_z_list:
            reference_z_list.sort()
            median_idx = len(reference_z_list) // 2
            self.plane_reference_z = reference_z_list[median_idx]
            
            logger.info(f"\n校準完成（{len(reference_z_list)} 次測量）")
            logger.info(f"   最終桌面高度: {self.plane_reference_z:.1f} mm")
        else:
            logger.warning("校準失敗，無有效測量")
    
    def save_roi(self):
        """保存 ROI 配置"""
        if not self.roi_points or self.roi_mask is None:
            logger.warning("沒有 ROI 可保存")
            return
        
        config = {
            'roi_points': [[int(x), int(y)] for x, y in self.roi_points],
            'roi': {
                'x1': int(self.roi[0]),
                'y1': int(self.roi[1]),
                'x2': int(self.roi[2]),
                'y2': int(self.roi[3])
            } if self.roi else None,
            'plane_reference_z': float(self.plane_reference_z) if self.plane_reference_z else None,
            'description': '桌面 ROI 配置（用於 RANSAC 平面擬合）'
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"配置已保存: {self.config_path}")
    
    @staticmethod
    def load_roi(config_path: str = "data/calibration/table_roi.json") -> Optional[Tuple]:
        """
        載入 ROI 配置
        
        Returns:
            (x1, y1, x2, y2, reference_z) 或 None
        """
        config_path = Path(config_path)
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            roi_config = config['roi']
            roi = (roi_config['x1'], roi_config['y1'], roi_config['x2'], roi_config['y2'])
            reference_z = config.get('plane_reference_z')
            
            return (*roi, reference_z)
        except Exception as e:
            logger.warning(f"載入配置失敗: {e}")
            return None


def main():
    """主程式：互動式校準"""
    calibrator = TablePlaneCalibrator()
    
    try:
        # 啟動相機
        calibrator.start_realsense()
        
        # 選擇 ROI
        calibrator.select_roi_interactive()
        
        if calibrator.roi:
            # 校準桌面平面
            calibrator.calibrate_live(duration=5.0)
            
            # 保存配置
            calibrator.save_roi()
        
    except Exception as e:
        logger.error(f"錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 停止相機
        calibrator.stop_realsense()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
