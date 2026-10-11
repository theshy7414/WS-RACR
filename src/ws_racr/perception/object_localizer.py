"""
LLM 視覺檢測模組
整合 YOLO 物件檢測、座標轉換和機器人控制
"""
import logging
from ws_racr.config import get_data_dir, get_yolo_weights_path
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class ObjectLocalizer:
    """LLM 系統的視覺檢測模組"""
    
    def __init__(self, model_path=None, 
                 external_pipeline=None, external_intrinsics=None,
                 camera_name="D435i",
                 external_yolo_detector=None,
                 external_top_camera_pipeline=None,
                 external_top_camera_intrinsics=None):
        """
        初始化視覺模組
        
        Args:
            model_path: YOLO 模型路徑（None 則使用預設路徑）
            external_pipeline: 外部相機 pipeline（如果提供，則不初始化新相機）
            external_intrinsics: 外部相機內參
            camera_name: 相機名稱 ("D435i" 或 "D455")，用於載入對應的外參
            external_yolo_detector: 外部 YOLO 檢測器（如果提供，則不載入新模型）
            external_top_camera_pipeline: 外部俯視相機 pipeline
            external_top_camera_intrinsics: 外部俯視相機內參
        """
        self.pipeline = external_pipeline
        self.model = external_yolo_detector  # 使用外部 YOLO 檢測器（如果有）
        self.align = None
        self.has_depth = False
        self.intrinsics = external_intrinsics
        self.extrinsics = None
        self.ref_points = None
        self.use_external_camera = (external_pipeline is not None)
        self.camera_name = camera_name  # 記錄相機名稱
        
        # 俯視相機（用於雙相機系統）
        self.top_pipeline = external_top_camera_pipeline
        self.top_intrinsics = external_top_camera_intrinsics
        
        # 物件類別映射 (根據你的模型)
        self.class_mapping = {
            0: 'bracket',
            1: 'cylinder',
            2: 'hex_wrench',
            3: 'nut',
            4: 'pi',
            5: 'platform',
            6: 'screw_big',
            7: 'screw_small'
        }
        
        # 如果沒有提供模型路徑，使用預設路徑
        if model_path is None:
            # 獲取專案根目錄
            model_path = get_yolo_weights_path()
        
        # 只在沒有外部相機時才初始化
        if not self.use_external_camera:
            self.init_camera()
        else:
            logger.info(f"使用外部相機實例（共用主系統 {camera_name} 相機）")
            self.has_depth = True  # 假設外部相機有深度
        
        # 只在沒有外部YOLO檢測器時才初始化模型
        if self.model is None:
            self.init_model(model_path)
        else:
            logger.info(f"使用外部YOLO檢測器（共用主系統模型）")
            
        self.load_calibration()
        
    def init_camera(self, width=1280, height=720, fps=30, retry=3):
        """初始化相機"""
        for attempt in range(retry):
            try:
                logger.info(f"正在初始化相機... (嘗試 {attempt + 1}/{retry})")
                
                self.pipeline = rs.pipeline()
                config = rs.config()
                
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                try:
                    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                    self.has_depth = True
                except:
                    self.has_depth = False
                    
                self.profile = self.pipeline.start(config)
                
                # 等待相機穩定 (丟棄前幾幀)
                logger.info("等待相機穩定...")
                for _ in range(5):
                    self.pipeline.wait_for_frames(timeout_ms=2000)
                    time.sleep(0.1)
                
                if self.has_depth:
                    self.align = rs.align(rs.stream.color)
                    
                # 獲取相機內參
                self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                logger.info(f"相機初始化成功: {width}x{height} @ {fps}fps")
                logger.info(f"相機內參: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
                return True
                
            except Exception as e:
                logger.warning(f"相機初始化失敗 (嘗試 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    logger.info("等待 2 秒後重試...")
                    time.sleep(2)
                else:
                    logger.error("相機初始化失敗，已達最大重試次數")
                    return False
        
    def check_camera_health(self) -> bool:
        """檢查相機是否健康"""
        if not self.pipeline:
            logger.error("Pipeline 未初始化")
            return False
        
        try:
            # 嘗試獲取一幀畫面
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if color_frame:
                logger.info("相機健康檢查通過")
                return True
            else:
                logger.warning("相機健康檢查失敗：無彩色畫面")
                return False
        except Exception as e:
            logger.error(f"相機健康檢查失敗: {e}")
            return False
    
    def init_model(self, model_path):
        """初始化 YOLO 模型（必須成功載入）"""
        try:
            # 確保路徑是 Path 對象
            if not isinstance(model_path, Path):
                model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO 模型不存在: {model_path}")
            
            self.model = YOLO(str(model_path))
            logger.info(f"YOLO 模型已載入: {model_path}")
            
        except Exception as e:
            logger.error(f"YOLO 模型載入失敗: {e}")
            raise RuntimeError(f"工件檢測必須載入 YOLO 模型，但初始化失敗: {e}")
            
    def load_calibration(self):
        """載入相機校正檔案"""
        try:
            cal_dir = str(get_data_dir() / "calibration" / "cal_info")
            
            # 根據相機名稱載入對應的外參矩陣
            if self.camera_name == "D455":
                extrinsics_file = "extrinsics_D455.npy"
                ref_points_file = "ref_points_D455.npy" if os.path.exists(os.path.join(cal_dir, "ref_points_D455.npy")) else None
            else:
                extrinsics_file = "extrinsics_D435i.npy"
                ref_points_file = "ref_points_D435i.npy" if os.path.exists(os.path.join(cal_dir, "ref_points_D435i.npy")) else None
            
            self.extrinsics = np.load(os.path.join(cal_dir, extrinsics_file))
            
            # 載入參考點（可選）
            if ref_points_file:
                self.ref_points = np.load(os.path.join(cal_dir, ref_points_file))
            else:
                self.ref_points = None
            
            logger.info(f"Calibration files loaded ({extrinsics_file})")
            logger.info(f"Extrinsics shape: {self.extrinsics.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            self.extrinsics = None
            self.ref_points = None
    
    def get_frame(self, timeout_ms=5000):
        """獲取相機畫面"""
        if not self.pipeline:
            logger.error("Pipeline 未初始化")
            return None, None
        
        try:
            # 添加超時設定
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            
            if self.has_depth and self.align:
                frames = self.align.process(frames)
                
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.has_depth else None
            
            if not color_frame:
                logger.warning("未獲取到彩色畫面")
                return None, None
                
            color_image = np.asanyarray(color_frame.get_data())
            
        except RuntimeError as e:
            error_msg = str(e)
            if "Frame didn't arrive" in error_msg:
                logger.error(f"相機連接問題: 在 {timeout_ms}ms 內未收到畫面")
                logger.info("請檢查:")
                logger.info("   1. 相機是否正確連接 (USB 3.0 接口)")
                logger.info("   2. 相機電源是否正常")
                logger.info("   3. 是否有其他程序占用相機")
                logger.info("   4. 嘗試重新插拔相機")
            else:
                logger.error(f"相機錯誤: {error_msg}")
            return None, None
        except Exception as e:
            logger.error(f"獲取畫面時發生未預期錯誤: {e}")
            return None, None
            
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_frame
    
    def detect_objects(self, image, target_classes=None, confidence=0.3):
        """檢測物件 - 使用 mask 版本"""
        if not self.model:
            return []
            
        results = self.model.predict(source=image, conf=confidence, verbose=False)
        detections = []
        
        if len(results) > 0:
            r = results[0]
            
            if not (hasattr(r, "boxes") and r.boxes is not None):
                return detections
                
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            # 檢查是否有 masks
            masks = None
            if hasattr(r, "masks") and r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                
            # 處理每個檢測結果
            for i, (bbox, confidence_score, class_id) in enumerate(zip(xyxy, conf, cls)):
                class_name = self.class_mapping.get(int(class_id), "unknown")
                
                # 如果指定了目標類別，進行過濾
                if target_classes and class_name.lower() not in [t.lower() for t in target_classes]:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                
                # 提取對應的 mask
                mask = None
                if masks is not None and i < len(masks):
                    mask = masks[i]
                    # 調整 mask 大小到原圖尺寸
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                detection = {
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': confidence_score,
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'size': (width, height),
                    'mask': mask
                }
                
                detections.append(detection)
        
        return detections
    
    def find_object_by_name(self, object_name: str, image=None, spatial_modifier: Optional[str] = None, object_size: Optional[str] = None, position_hint: Optional[list] = None) -> Optional[Dict]:
        """根據物件名稱尋找物件，支持空間篩選和尺寸篩選

        Args:
            object_name: 物件名稱
            image: 圖像（可選）
            spatial_modifier: 空間描述詞（leftmost/rightmost/nearest等）
            object_size: 尺寸描述（big/small/大/小等）
            position_hint: [x, y, z] metres — if provided and multiple detections
                exist, return the one whose 3D position is closest to this hint.
                Used when pointing_target resolves to a specific instance ID.

        Returns:
            檢測結果，如果有多個物體會根據 spatial_modifier 和 object_size 篩選
        """
        if image is None:
            # 先檢查相機健康狀態
            if not self.check_camera_health():
                logger.error("相機不可用，無法檢測物件")
                return None
            
            image, _ = self.get_frame()
            if image is None:
                logger.error("無法獲取影像畫面")
                return None
        
        # 將物件名稱映射到類別
        target_classes = []
        object_lower = object_name.lower()
        
        # 支架
        if any(word in object_lower for word in ['支架', 'bracket', '托架']):
            target_classes.append('bracket')
        # 圓柱
        elif any(word in object_lower for word in ['圓柱', 'cylinder', '柱子']):
            target_classes.append('cylinder')
        # 六角扳手
        elif any(word in object_lower for word in ['扳手', 'hex_wrench', '六角扳手', '內六角']):
            target_classes.append('hex_wrench')
        # 螺帽
        elif any(word in object_lower for word in ['螺帽', 'nut', '螺母']):
            target_classes.append('nut')
        # 樹莓派
        elif any(word in object_lower for word in ['pi', 'raspberry', '樹莓派', 'rpi']):
            target_classes.append('pi')
        # 平台
        elif any(word in object_lower for word in ['平台', 'platform', '底座']):
            target_classes.append('platform')
        # 大螺絲
        elif any(word in object_lower for word in ['大螺絲', 'screw_big', '大螺釘']):
            target_classes.append('screw_big')
        # 小螺絲
        elif any(word in object_lower for word in ['小螺絲', 'screw_small', '小螺釘']):
            target_classes.append('screw_small')
        # 螺絲（不分大小）- 根據 object_size 參數決定
        elif any(word in object_lower for word in ['螺絲', 'screw', '螺釘']):
            if object_size and any(s in object_size.lower() for s in ['big', 'large', '大', '大號']):
                target_classes.append('screw_big')
                logger.info(f"   根據尺寸描述 '{object_size}' 搜尋大螺絲")
            elif object_size and any(s in object_size.lower() for s in ['small', 'tiny', '小', '小號']):
                target_classes.append('screw_small')
                logger.info(f"   根據尺寸描述 '{object_size}' 搜尋小螺絲")
            else:
                # 沒有指定尺寸，搜尋兩種
                target_classes.extend(['screw_big', 'screw_small'])
        elif any(word in object_lower for word in ['紅色', '藍色', '綠色']):
            # 顏色描述，檢測所有類別
            target_classes = list(self.class_mapping.values())
        else:
            # 默認檢測所有類別
            target_classes = list(self.class_mapping.values())
        
        detections = self.detect_objects(image, target_classes)

        if detections:
            # Priority 1: position_hint — pick the YOLO detection closest in 2D
            # image-space to the projected hint position when multiple instances exist.
            # Requires depth camera to compute 3D; we approximate using pixel position
            # from the intrinsics stored on this camera object.
            if position_hint is not None and len(detections) > 1:
                try:
                    import numpy as np
                    hint = np.array(position_hint[:3], dtype=float)
                    best_det = None
                    best_d = float('inf')
                    for det in detections:
                        pos = det.get('position_3d') or det.get('position')
                        if pos is None:
                            continue
                        d = float(np.linalg.norm(np.array(pos[:3], dtype=float) - hint))
                        if d < best_d:
                            best_d = d
                            best_det = det
                    if best_det is not None and best_d < 0.20:  # 20 cm threshold
                        logger.info(f"   position_hint 匹配: {best_det.get('class_name')} "
                              f"dist={best_d*100:.1f}cm")
                        return best_det
                except Exception:
                    pass

            # Priority 2: spatial_modifier
            if spatial_modifier:
                filtered = self._filter_by_spatial_modifier(detections, spatial_modifier)
                if filtered:
                    return filtered
                else:
                    logger.warning(f"   無法根據 '{spatial_modifier}' 篩選，返回信心度最高的物體")

            # Fallback: highest confidence
            best_detection = max(detections, key=lambda x: x['confidence'])
            return best_detection
        
        return None
    
    def _filter_by_spatial_modifier(self, detections: list, spatial_modifier: str) -> Optional[Dict]:
        """根據空間描述詞篩選物體
        
        機器人基座標系（Base Frame）：
        - X 軸：大=離人類遠（後） | 小=離人類近（前）
        - Y 軸：大=左邊 | 小=右邊
        - Z 軸：大=上面 | 小=下面
        
        俯視相機圖像座標（近似映射）：
        - 圖像 x（像素左右）≈ 機器人 Y 軸（左右）
        - 圖像 y（像素上下）≈ 機器人 X 軸（前後）
        - 相機朝向：從上往下看工作區
        
        Args:
            detections: 檢測結果列表
            spatial_modifier: 空間描述詞
        
        Returns:
            篩選後的物體
        """
        if len(detections) == 1:
            return detections[0]
        
        # 標準化描述詞（支持中英文）
        modifier_map = {
            'leftmost': 'leftmost', '最左邊': 'leftmost',
            'rightmost': 'rightmost', '最右邊': 'rightmost',
            'frontmost': 'frontmost', '最前面': 'frontmost',
            'backmost': 'backmost', '最後面': 'backmost',
            'nearest': 'nearest', '最近': 'nearest',
            'farthest': 'farthest', '最遠': 'farthest',
            'center': 'center', '中間': 'center'
        }
        
        modifier = modifier_map.get(spatial_modifier, spatial_modifier)
        
        logger.info(f"   從 {len(detections)} 個物體中篩選: {modifier}")
        
        # 根據 bbox 中心點排序（圖像座標近似）
        # 注意：這裡使用圖像座標作為近似，實際3D位置在後續轉換
        if modifier == 'leftmost':
            # 最左邊 = 機器人 Y 最大 ≈ 圖像 x 最大
            selected = max(detections, key=lambda d: d['center'][0])
            logger.info(f"      選擇最左邊: center_x={selected['center'][0]}")
        
        elif modifier == 'rightmost':
            # 最右邊 = 機器人 Y 最小 ≈ 圖像 x 最小
            selected = min(detections, key=lambda d: d['center'][0])
            logger.info(f"      選擇最右邊: center_x={selected['center'][0]}")
        
        elif modifier == 'frontmost':
            # 最前面（離人類近）= 機器人 X 最小 ≈ 圖像 y 最大
            selected = max(detections, key=lambda d: d['center'][1])
            logger.info(f"      選擇最前面: center_y={selected['center'][1]}")
        
        elif modifier == 'backmost':
            # 最後面（離人類遠）= 機器人 X 最大 ≈ 圖像 y 最小
            selected = min(detections, key=lambda d: d['center'][1])
            logger.info(f"      選擇最後面: center_y={selected['center'][1]}")
        
        elif modifier == 'nearest':
            # 最近的（圖像中最大的物體，假設距離與大小成反比）
            selected = max(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])  # width * height
            logger.info(f"      選擇最近的: bbox_area={selected['bbox'][2] * selected['bbox'][3]}")
        
        elif modifier == 'farthest':
            # 最遠的（圖像中最小的物體）
            selected = min(detections, key=lambda d: d['bbox'][2] * d['bbox'][3])
            logger.info(f"      選擇最遠的: bbox_area={selected['bbox'][2] * selected['bbox'][3]}")
        
        elif modifier == 'center':
            # 中間的（最接近圖像中心的物體）
            img_center_x, img_center_y = 320, 240  # 假設 640x480 圖像
            selected = min(detections, key=lambda d: 
                ((d['center'][0] - img_center_x)**2 + (d['center'][1] - img_center_y)**2)**0.5)
            logger.info(f"      選擇中間的: center={selected['center']}")
        
        else:
            logger.warning(f"      未知的空間描述詞: {modifier}")
            return None
        
        return selected
    
    def calculate_gripper_angle_and_position(self, detection, image):
        """計算夾爪角度和精確位置 (從 simple_gripper.py 移植)"""
        mask = detection.get('mask')
        
        if mask is None:
            return self._calculate_bbox_method(detection)
        
        try:
            # 將 mask 轉換為二值圖像
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # 找到 mask 的輪廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return self._calculate_bbox_method(detection)
            
            # 找最大輪廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) < 500:
                return self._calculate_bbox_method(detection)
            
            # 使用最小面積矩形
            rect = cv2.minAreaRect(largest_contour)
            center, size, angle = rect
            width, height = size
            
            # 簡單直接：確保夾爪沿著短邊
            if width > height:
                # 橫向長方形，旋轉90度
                angle += 90
            
            # 標準化角度到 [-90, 90) 範圍
            while angle >= 90:
                angle -= 180
            while angle < -90:
                angle += 180
            
            # 計算精確的中心點
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                precise_cx = int(M["m10"] / M["m00"])
                precise_cy = int(M["m01"] / M["m00"])
            else:
                precise_cx, precise_cy = detection['center']
            
            # 轉換角度到機械手臂座標系
            robot_angle = self._transform_angle_to_robot_frame(angle, precise_cx, precise_cy)
            
            return {
                'angle': robot_angle,
                'image_angle': angle,
                'center': (precise_cx, precise_cy),
                'contour': largest_contour,
                'rect': rect,
                'method': 'mask_based'
            }
            
        except Exception as e:
            logger.error(f"Mask processing failed: {e}")
            return self._calculate_bbox_method(detection)
    
    def _calculate_bbox_method(self, detection):
        """後備方法：使用邊界框計算"""
        width, height = detection['size']
        angle = 0 if width > height * 1.2 else 90
        
        return {
            'angle': angle,
            'center': detection['center'],
            'contour': None,
            'rect': None,
            'method': 'bbox_fallback'
        }
    
    def _transform_angle_to_robot_frame(self, image_angle, pixel_x, pixel_y):
        """將圖像角度轉換到機械手臂座標系 (從 simple_gripper.py 移植)"""
        if self.extrinsics is None:
            return image_angle
            
        try:
            # 獲取旋轉矩陣
            R_cam_to_robot = self.extrinsics[:3, :3]
            
            # 將圖像角度轉換為方向向量
            angle_rad = np.radians(image_angle)
            direction_cam = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
            
            # 轉換到機器人座標系
            direction_robot = R_cam_to_robot @ direction_cam
            
            # 計算機器人座標系中的角度
            robot_angle = np.arctan2(direction_robot[1], direction_robot[0]) * 180 / np.pi
            
            # 正規化角度
            while robot_angle > 180:
                robot_angle -= 360
            while robot_angle < -180:
                robot_angle += 360
                
            return robot_angle
            
        except Exception as e:
            return image_angle
    
    def pixel_to_world_coordinates(self, px, py, depth_frame=None, distance=None):
        """像素座標轉世界座標 (從 simple_gripper.py 移植)"""
        if self.extrinsics is None:
            return None
            
        # 獲取深度值
        if distance is None and depth_frame:
            distance = depth_frame.get_distance(px, py)
            
        if not distance or distance <= 0:
            return None
            
        try:
            # 使用 RealSense 內參進行轉換
            camera_point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [px, py], distance)
            
            # 相機座標轉機器人座標
            x_c, y_c, z_c = camera_point
            camera_coords = np.array([x_c, y_c, z_c, 1.0])
            
            # 構建變換矩陣
            transform_matrix = np.eye(4)
            transform_matrix[:3, :4] = self.extrinsics
            
            # 轉換到機器人座標系
            robot_coords = transform_matrix @ camera_coords
            
            # 轉換為毫米
            world_x = robot_coords[0] * 1000
            world_y = robot_coords[1] * 1000
            world_z = robot_coords[2] * 1000
            
            return [world_x, world_y, world_z]
            
        except Exception as e:
            logger.error(f"Coordinate transformation failed: {e}")
            return None
    
    def get_object_position_and_orientation(self, object_name: str) -> Optional[Dict]:
        """獲取物件的完整位置和方向信息"""
        # 獲取當前畫面
        image, depth_frame = self.get_frame()
        if image is None:
            return None
        
        # 檢測物件
        detection = self.find_object_by_name(object_name, image)
        if not detection:
            return None
        
        # 計算夾爪角度和精確位置
        grip_info = self.calculate_gripper_angle_and_position(detection, image)
        
        # 轉換座標
        world_coords = self.pixel_to_world_coordinates(
            grip_info['center'][0], 
            grip_info['center'][1], 
            depth_frame
        )
        
        if not world_coords:
            return None
        
        # 構建完整的位置信息
        result = {
            'object_name': object_name,
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'position': [
                world_coords[0],  # x (mm)
                world_coords[1],  # y (mm) 
                world_coords[2],  # z (mm)
                180.0,            # rx (deg) - 固定朝下
                0.0,              # ry (deg)
                grip_info['angle'] # rz (deg) - 夾爪角度
            ],
            'pixel_center': grip_info['center'],
            'detection_method': grip_info['method'],
            'timestamp': time.time()
        }
        
        return result
    
    def restart_camera(self):
        """重啟相機"""
        logger.info("正在重啟相機...")
        try:
            if self.pipeline:
                self.pipeline.stop()
                time.sleep(1)
            
            return self.init_camera()
        except Exception as e:
            logger.error(f"相機重啟失敗: {e}")
            return False
    
    def cleanup(self):
        """清理資源"""
        try:
            if self.pipeline:
                logger.info("正在停止相機...")
                self.pipeline.stop()
                self.pipeline = None
            logger.info("視覺模組已清理")
        except Exception as e:
            logger.warning(f"清理資源時發生錯誤: {e}")
    
    def get_hand_position_mediapipe(self) -> Optional[List[float]]:
        """
        使用 MediaPipe 檢測手部位置並轉換為3D座標
        
        Returns:
            [x, y, z] 手部中心的3D座標 (mm)，如果未檢測到則返回 None
        """
        try:
            # 導入 MediaPipe
            import mediapipe as mp
            
            # 初始化 MediaPipe Hands（如果還沒有）
            if not hasattr(self, 'mp_hands'):
                mp_module = mp.solutions.hands
                self.mp_hands = mp_module.Hands(
                    static_image_mode=False,
                    max_num_hands=1,  # 只檢測一隻手
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            
            # 獲取當前幀
            image, depth_frame = self.get_frame()
            if image is None:
                return None
            
            # 轉換為 RGB (MediaPipe 需要)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 檢測
            results = self.mp_hands.process(image_rgb)
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                # 取第一隻手的 landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # 使用手掌中心 (landmark 0 = 手腕，或計算手掌中心)
                # 計算手掌中心：取手腕(0)、食指根部(5)、小指根部(17)的平均
                wrist = hand_landmarks.landmark[0]
                index_mcp = hand_landmarks.landmark[5]
                pinky_mcp = hand_landmarks.landmark[17]
                
                palm_center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0
                palm_center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3.0
                
                # 轉換為像素座標
                h, w = image.shape[:2]
                pixel_x = int(palm_center_x * w)
                pixel_y = int(palm_center_y * h)
                
                # 邊界檢查
                pixel_x = max(0, min(w-1, pixel_x))
                pixel_y = max(0, min(h-1, pixel_y))
                
                # 如果有深度資訊，獲取3D座標
                if self.has_depth and depth_frame is not None:
                    depth_value = depth_frame.get_distance(pixel_x, pixel_y)
                    
                    if depth_value > 0:
                        # 使用相機內參進行反投影
                        point_3d = rs.rs2_deproject_pixel_to_point(
                            self.intrinsics,
                            [pixel_x, pixel_y],
                            depth_value
                        )
                        
                        # 轉換為 mm 並應用外參轉換（如果有）
                        x_mm = point_3d[0] * 1000
                        y_mm = point_3d[1] * 1000
                        z_mm = point_3d[2] * 1000
                        
                        # 應用外參轉換到機器人座標系
                        if self.extrinsics is not None:
                            camera_point = np.array([x_mm, y_mm, z_mm, 1.0])
                            robot_point = self.extrinsics @ camera_point
                            x_mm, y_mm, z_mm = robot_point[:3]
                        
                        logger.info(f"   MediaPipe 檢測到手部: [{x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}] mm")
                        return [x_mm, y_mm, z_mm]
                    else:
                        logger.warning(f"   無效的深度值: {depth_value}")
                else:
                    logger.warning(f"   沒有深度資訊")
            
            return None
            
        except Exception as e:
            logger.warning(f"   MediaPipe 手部檢測失敗: {e}")
            return None


def main():
    """測試視覺模組"""
    vision = ObjectLocalizer()
    
    try:
        logger.info("測試物件檢測...")
        
        # 測試檢測沙包
        result = vision.get_object_position_and_orientation("藍色沙包")
        if result:
            logger.info(f"找到物件: {result['class_name']}")
            logger.info(f"位置: {result['position'][:3]} mm")
            logger.info(f"角度: {result['position'][5]:.1f}°")
        else:
            logger.error("未找到目標物件")
            
    except KeyboardInterrupt:
        logger.info("測試中斷")
    finally:
        vision.cleanup()


if __name__ == "__main__":
    main()
