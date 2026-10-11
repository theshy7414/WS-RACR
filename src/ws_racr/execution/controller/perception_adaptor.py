"""
PerceptionAdaptorMixin — Camera, depth, and object detection helpers.

Extracted from RobotController to reduce per-query context size.
All methods rely on self.* attributes initialized by RobotController.__init__().
Do NOT instantiate this class directly.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Dict, Optional, List, Any, Tuple

import numpy as np

from ws_racr.config import get_data_dir, get_yolo_weights_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass  # avoid circular imports


class PerceptionAdaptorMixin:
    """
    Mixin providing camera/perception helper methods for RobotController.

    Included methods:
      - _init_dual_camera_pointing
      - _deproject_and_transform
      - _validate_depth
      - _resolve_object_with_size
      - _detect_object_with_d435i
      - _handle_occluded_object
      - _detect_hand_with_d435i
      - _create_simple_waypoint
      - _create_plan_json
      - _calculate_ray_score
      - _detect_pointed_object_dual_camera

    All methods access shared state via self.* (RobotController instance).
    """
    def _init_dual_camera_pointing(self):
        """初始化雙相機手部指向檢測系統（從 hand_pointing_dual_camera.py 整合）"""
        try:
            import pyrealsense2 as rs
            import mediapipe as mp
            from ultralytics import YOLO
            import numpy as np
            
            logger.info("      初始化雙相機手部指向檢測系統...")
            
            # 優先使用外部傳入的相機（避免重複初始化導致資源衝突）
            use_external_cameras = False
            
            if hasattr(self, 'external_camera_pipeline') and self.external_camera_pipeline is not None:
                # 使用主系統已初始化的 D455 (前視)
                self.front_pipeline = self.external_camera_pipeline
                self.front_intrinsics = self.external_camera_intrinsics
                self.front_align = rs.align(rs.stream.color)
                logger.info("         使用外部 D455 前視相機")
                use_external_cameras = True
            
            if hasattr(self, 'external_top_camera_pipeline') and self.external_top_camera_pipeline is not None:
                # 使用主系統已初始化的 D435i (俯視)
                self.top_pipeline = self.external_top_camera_pipeline
                self.top_intrinsics = self.external_top_camera_intrinsics
                self.top_align = rs.align(rs.stream.color)
                logger.info("         使用外部 D435i 俯視相機")
                use_external_cameras = True
            
            # 如果沒有外部相機，才嘗試自行初始化（會導致資源衝突）
            if not use_external_cameras:
                # 1. 初始化相機
                ctx = rs.context()
                devices = ctx.query_devices()
                
                d455_serial = None
                d435i_serial = None
                
                for dev in devices:
                    name = dev.get_info(rs.camera_info.name)
                    serial = dev.get_info(rs.camera_info.serial_number)
                    if "D455" in name:
                        d455_serial = serial
                        logger.info(f"         找到 D455 (前視): {serial}")
                    elif "D435I" in name:
                        d435i_serial = serial
                        logger.info(f"         找到 D435i (俯視): {serial}")
                
                if not d455_serial or not d435i_serial:
                    logger.warning("      需要同時連接 D455 和 D435i")
                    return False
                
                # D455 (前視 - 手部)
                self.front_pipeline = rs.pipeline()
                front_config = rs.config()
                front_config.enable_device(d455_serial)
                front_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                front_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                front_profile = self.front_pipeline.start(front_config)
                self.front_intrinsics = front_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.front_align = rs.align(rs.stream.color)
                
                # D435i (俯視 - 物體)
                self.top_pipeline = rs.pipeline()
                top_config = rs.config()
                top_config.enable_device(d435i_serial)
                top_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                top_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                top_profile = self.top_pipeline.start(top_config)
                self.top_intrinsics = top_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.top_align = rs.align(rs.stream.color)
                
                logger.info("         相機初始化完成")
            
            # 2. 載入 YOLO 模型（優先使用外部檢測器）
            if hasattr(self, 'external_yolo_detector') and self.external_yolo_detector is not None:
                self.yolo_model = self.external_yolo_detector
                logger.info("         使用外部 YOLO 檢測器")
            else:
                model_path = str(get_yolo_weights_path())
                self.yolo_model = YOLO(model_path)
                logger.info("         YOLO 載入完成")
            
            # 3. 載入外參
            cal_dir = str(get_data_dir() / "calibration" / "cal_info")
            
            # D455 (前視): 3x4 -> 4x4
            extr_front = np.load(f"{cal_dir}/extrinsics_D455.npy")
            self.T_front_to_base = np.eye(4)
            self.T_front_to_base[:3, :] = extr_front
            
            # D435i (俯視) - 兩個版本的外參
            # 1. 原始外參 (夾爪參考點) - 用於 receive 任務
            extr_top = np.load(f"{cal_dir}/extrinsics_D435i.npy")
            self.T_top_to_base = np.eye(4)
            self.T_top_to_base[:3, :] = extr_top
            
            # 2. 補償後外參 (TCP參考點) - 用於 pointing 任務
            extr_top_pointing = np.load(f"{cal_dir}/extrinsics_D435i_withoutgripper.npy")
            self.T_top_to_base_for_pointing = np.eye(4)
            self.T_top_to_base_for_pointing[:3, :] = extr_top_pointing
            
            logger.info("         外參載入完成")
            logger.info("            - receive 任務: extrinsics_D435i.npy (夾爪參考點)")
            logger.info("            - pointing 任務: extrinsics_D435i_withoutgripper.npy (TCP參考點)")
            
            # 4. 初始化 MediaPipe
            # 針對手心向上的俯視角度優化：
            # - model_complexity=1: 提高困難角度的檢測能力
            # - min_detection_confidence=0.5: 降低閾值以適應手心向上的低特徵可見度
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,              # 0:輕量, 1:完整 (預設0)
                min_detection_confidence=0.5,    # 從0.7降至0.5
                min_tracking_confidence=0.4      # 從0.5降至0.4
            )
            logger.info("         MediaPipe 初始化完成 (model_complexity=1, detection_conf=0.5)")
            
            # 5. 初始化 ObjectLocalizer 用於俯視相機（用於物體檢測）
            from ws_racr.perception.object_localizer import ObjectLocalizer
            self.top_camera = ObjectLocalizer(
                external_pipeline=self.top_pipeline,
                external_intrinsics=self.top_intrinsics,
                camera_name="D435i",
                external_yolo_detector=self.yolo_model
            )
            logger.info("         俯視相機 ObjectLocalizer 初始化完成")
            
            self.dual_camera_initialized = True
            logger.info("      雙相機系統初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"      雙相機初始化失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _deproject_and_transform(self, u, v, depth, intrinsics, T_to_base):
        """像素座標 + 深度 -> 世界座標（用於俯視相機的物體）"""
        import pyrealsense2 as rs
        import numpy as np
        
        if depth <= 0:
            return None
        
        # 像素 -> 相機座標
        point_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
        p_cam = np.array(point_cam, dtype=np.float32)
        
        # 相機座標 -> 世界座標
        p_hom = np.append(p_cam, 1.0)
        p_world = T_to_base @ p_hom
        
        # 如果使用俯視相機外參（T_top_to_base），應用物體位置偏差修正
        if np.array_equal(T_to_base, self.T_top_to_base):
            bias_vector = np.array([self.TOP_CAMERA_X_BIAS, self.TOP_CAMERA_Y_BIAS, self.TOP_CAMERA_Z_BIAS])
            p_world[:3] = p_world[:3] + bias_vector
        
        return p_world[:3]
    
    # ==================== 公共方法：減少重複代碼 ====================
    
    def _validate_depth(self, depth_value: float, min_depth: float = 0.3, max_depth: float = 3.0) -> tuple:
        """
        驗證深度值是否有效（D435i 範圍：0.3m - 3.0m）
        
        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        if depth_value <= 0:
            return False, "深度值無效 (≤0)，可能原因：反光表面、黑色物體、深度相機未對準"
        
        if depth_value < min_depth:
            return False, f"物體太近 ({depth_value*1000:.0f}mm < {min_depth*1000:.0f}mm)，建議移遠至少 {min_depth*100:.0f}cm"
        
        if depth_value > max_depth:
            return False, f"物體太遠 ({depth_value*1000:.0f}mm > {max_depth*1000:.0f}mm)，建議移近至多 {max_depth:.0f}m"
        
        return True, None
    
    def _resolve_object_with_size(self, target_object: str, object_size: Optional[str] = None) -> str:
        """
        根據 object_size 解析完整的物件類別名稱
        
        Args:
            target_object: 目標物件名稱（可能是中文或英文）
            object_size: 尺寸描述（big/small/大/小等）
        
        Returns:
            完整的 YOLO 類別名稱（如 screw_big, screw_small）
        """
        # 小寫轉換
        obj_lower = target_object.lower()
        
        # 如果沒有指定尺寸，直接返回
        if not object_size:
            return target_object
        
        size_lower = object_size.lower()
        
        # 螺絲特殊處理：螺絲 + 大/小 → screw_big/screw_small
        if any(word in obj_lower for word in ['螺絲', 'screw', '螺釘']):
            if any(s in size_lower for s in ['big', 'large', '大', '大號']):
                logger.info(f"   尺寸解析: '{target_object}' + '{object_size}' → 'screw_big'")
                return 'screw_big'
            elif any(s in size_lower for s in ['small', 'tiny', '小', '小號']):
                logger.info(f"   尺寸解析: '{target_object}' + '{object_size}' → 'screw_small'")
                return 'screw_small'
        
        # 其他物件目前不需要尺寸區分，直接返回
        return target_object
    
    def _detect_object_with_d435i(self, object_name: str, spatial_modifier: Optional[str] = None, object_size: Optional[str] = None) -> Optional[Dict]:
        """
        使用 D435i 俯視相機檢測物體（統一方法，避免重複代碼）
        
        Args:
            object_name: 物體名稱
            spatial_modifier: 空間描述詞（leftmost/rightmost/nearest等）
            object_size: 尺寸描述（big/small/大/小等）
        
        Returns:
            {
                'position': [x, y, z, rx, ry, rz],  # mm, degree
                'grip_angle': float,
                'detection': {...}
            } 或 None
        """
        if not self.dual_camera_initialized:
            if not self._init_dual_camera_pointing():
                logger.error("   雙相機系統初始化失敗")
                return None
        
        try:
            image_top, depth_frame_top = self.top_camera.get_frame()
            if image_top is None:
                logger.error("   無法獲取相機畫面")
                return None
            
            # 如果有空間修飾詞，先從 WorldState 找到目標物體
            target_obj_from_ws = None
            if spatial_modifier and hasattr(self, 'world_state_builder') and self.world_state_builder:
                world_state = self.world_state_builder.get_current_state()
                
                # 找出所有同類型物體（包括 missing/occluded）
                same_type_objects = []
                for obj_id, obj in world_state.objects.items():
                    if obj.object_type == object_name:
                        same_type_objects.append((obj_id, obj))
                
                if same_type_objects:
                    logger.info(f"   WorldState 中找到 {len(same_type_objects)} 個 {object_name}")
                    for obj_id, obj in same_type_objects:
                        pos_str = f"[{obj.position[0]:.1f}, {obj.position[1]:.1f}]" if obj.position else "無位置"
                        logger.info(f"      - {obj_id}: {pos_str} (狀態: {obj.visibility})")
                    
                    # 根據空間修飾詞找出目標物體
                    if spatial_modifier == 'rightmost' or spatial_modifier == '最右邊':
                        # 找 X 座標最大的
                        target_obj_from_ws = max(same_type_objects, key=lambda x: x[1].position[0] if x[1].position else -float('inf'))
                        logger.info(f"   根據 WorldState 記憶，最右邊的是: {target_obj_from_ws[0]} (X={target_obj_from_ws[1].position[0]:.1f})")
                    elif spatial_modifier == 'leftmost' or spatial_modifier == '最左邊':
                        # 找 X 座標最小的
                        target_obj_from_ws = min(same_type_objects, key=lambda x: x[1].position[0] if x[1].position else float('inf'))
                        logger.info(f"   根據 WorldState 記憶，最左邊的是: {target_obj_from_ws[0]} (X={target_obj_from_ws[1].position[0]:.1f})")
                    elif spatial_modifier == 'nearest' or spatial_modifier == '最近':
                        # 找 Y 座標最小的（最靠近機器人）
                        target_obj_from_ws = min(same_type_objects, key=lambda x: x[1].position[1] if x[1].position else float('inf'))
                        logger.info(f"   根據 WorldState 記憶，最近的是: {target_obj_from_ws[0]} (Y={target_obj_from_ws[1].position[1]:.1f})")
                    
                    if target_obj_from_ws:
                        target_obj_id, target_obj = target_obj_from_ws

                        # 檢查該物體當前狀態
                        if target_obj.visibility in ['missing', 'occluded']:
                            logger.warning(f"   目標物體 {target_obj_id} 當前狀態: {target_obj.visibility}")
                            return self._handle_occluded_object(target_obj, object_name, world_state)
                        else:
                            # WorldState 僅用來決定「哪一個」是目標（occluded check）；
                            # 實際座標 + 夾取角度必須從即時 YOLO 重新偵測，
                            # 因為 WorldState 位置可能過時，且不包含 grip_angle。
                            # → fall through 到下方的 find_object_by_name，使用即時偵測。
                            ws_hint_position = list(target_obj.position[:3]) if target_obj.position else None
                            logger.info(f"   WorldState 確認目標: {target_obj_id}，"
                                  f"以記憶位置 hint 做即時 YOLO 偵測")
                            # 嘗試用 hint 找最近的 YOLO 偵測結果
                            _hint_det = self._find_object_nearest_to_hint(
                                class_name=object_name,
                                image=image_top,
                                depth_frame=depth_frame_top,
                                hint_pos=ws_hint_position,
                                spatial_modifier=spatial_modifier,
                                object_size=object_size,
                            )
                            if _hint_det:
                                # 有 hint 結果 → 把它當作 detection 直接進入共用處理
                                logger.info(f"   YOLO hint 偵測成功 (信心度: {_hint_det.get('confidence', 0):.2f})")
                                detection_for_ws = _hint_det
                            else:
                                # hint 失敗 → 用 spatial_modifier 再試一次
                                logger.warning(f"   YOLO hint 偵測失敗，改用 spatial_modifier 篩選")
                                detection_for_ws = self.top_camera.find_object_by_name(
                                    object_name, image_top, spatial_modifier, object_size
                                )
                            if detection_for_ws:
                                # 共用的 grip_angle 計算 + 深度轉換
                                return self._process_yolo_detection(
                                    detection_for_ws, image_top, depth_frame_top
                                )
                            # YOLO 完全找不到 → fall through 到下方做最後一次嘗試

            # 即時視覺偵測（無 WorldState hint，或 WorldState 路徑找不到物體時）
            detection = self.top_camera.find_object_by_name(object_name, image_top, spatial_modifier, object_size)
            if not detection:
                logger.error(f"   當前視覺檢測找不到物體: {object_name}")
                
                # 檢查 WorldState 記憶，判斷是遮擋還是真的不存在
                if hasattr(self, 'world_state_builder') and self.world_state_builder:
                    world_state = self.world_state_builder.get_current_state()
                    
                    # 查找該物件在 WorldState 中的記錄（不考慮空間修飾詞的通用查找）
                    target_obj = None
                    for obj_id, obj in world_state.objects.items():
                        if obj.object_type == object_name or object_name in obj.object_type:
                            target_obj = obj
                            break
                    
                    if target_obj:
                        logger.info(f"   WorldState 記憶: {target_obj.object_type} ({target_obj.visibility})")
                        
                        # 如果物體曾經被看到過，但現在看不見 → 可能是遮擋
                        if target_obj.visibility in ['missing', 'occluded']:
                            logger.warning(f"   物體狀態: {target_obj.visibility} - 觸發遮擋處理策略")
                            
                            # 計算遮擋時長
                            current_time = time.time()
                            time_occluded = current_time - target_obj.last_seen if target_obj.last_seen else 0
                            
                            # 使用 occlusion_handler 處理（正確的方法名稱）
                            action = self.occlusion_handler.get_best_strategy(
                                occluded_obj=target_obj,
                                world_state=world_state,
                                time_occluded=time_occluded
                            )
                            
                            logger.info(f"   遮擋處理決策: {action['strategy']}")
                            logger.info(f"      說明: {action.get('explanation', '')}")
                            
                            # 根據遮擋物類型決定處理策略
                            occluder_id = action.get('occluder_id', '')
                            occluder_name = action.get('occluder_name', '障礙物')
                            
                            # 物體名稱中英映射（用於語音提示）
                            name_mapping = {
                                'screw_big': '大螺絲',
                                'screw_small': '小螺絲',
                                'cylinder': '圓柱',
                                'platform': '平台',
                                'bracket': '支架',
                                'screw': '螺絲'
                            }
                            target_name_zh = name_mapping.get(target_obj.object_type, target_obj.object_type)
                            
                            # 判斷遮擋物類型（檢查物體類型前綴）
                            # platform, bracket, 或類型不明 → 請人類移開
                            # 其他工件（cylinder, screw_big, screw_small 等）→ 機器人直接移除
                            needs_human_removal = False
                            occluder_type = occluder_id.split('_')[0].lower() if occluder_id and '_' in occluder_id else (occluder_id.lower() if occluder_id else 'unknown')
                            
                            if occluder_type in ['platform', 'bracket']:
                                needs_human_removal = True
                                logger.info(f"   遮擋物為 {occluder_name}（類型: {occluder_type}），需要人類移開")
                            elif not occluder_id or occluder_type == 'unknown':
                                needs_human_removal = True
                                logger.info(f"   遮擋物類型不明，需要人類移開")
                            else:
                                needs_human_removal = False
                                logger.info(f"   遮擋物為 {occluder_name}（類型: {occluder_type}），機器人可直接移除")
                            
                            if needs_human_removal:
                                # 請人類移開：語音詢問
                                question = f"{target_name_zh}被{occluder_name}遮住了，需要我幫你移開{occluder_name}嗎？還是你自己移開？"
                                logger.info(f"   語音詢問: {question}")
                                
                                return {
                                    'occlusion_detected': True,
                                    'needs_voice_confirmation': True,
                                    'action': action,
                                    'object': target_obj,
                                    'question': question,
                                    'target_object': object_name,
                                    'target_object_zh': target_name_zh,
                                    'occluder_object': occluder_id,
                                    'occluder_name': occluder_name
                                }
                            else:
                                # 機器人直接移除：返回自動執行標記
                                notification = f"偵測到{target_name_zh}被{occluder_name}遮住，先將{occluder_name}移至放置區"
                                logger.info(f"   語音通知: {notification}")
                                
                                return {
                                    'occlusion_detected': True,
                                    'auto_remove_occluder': True,
                                    'action': action,
                                    'object': target_obj,
                                    'notification': notification,
                                    'target_object': object_name,
                                    'target_object_zh': target_name_zh,
                                    'occluder_object': occluder_id,
                                    'occluder_name': occluder_name
                                }
                
                logger.info("      建議：1) 確認物體在視野內  2) 檢查物體名稱  3) 增強光照")
                return None
            
            logger.info(f"   YOLO 檢測成功: {detection['class_name']} (信心度: {detection.get('confidence', 0):.2f})")
            return self._process_yolo_detection(detection, image_top, depth_frame_top)
            
        except Exception as e:
            logger.error(f"   物體檢測異常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_yolo_detection(self, detection: dict, image, depth_frame) -> Optional[dict]:
        """YOLO 偵測結果 → grip_angle + 深度轉換 → 標準回傳格式。

        抽取自 _detect_object_with_d435i，讓 WorldState hint 路徑可重用相同邏輯。
        """
        try:
            grip_info = self.top_camera.calculate_gripper_angle_and_position(detection, image)
            center_u, center_v = grip_info['center']
            depth_value = depth_frame.get_distance(center_u, center_v)

            is_valid, error_msg = self._validate_depth(depth_value)
            if not is_valid:
                logger.error(f"   {error_msg}")
                return None

            grasp_pos_3d = self._deproject_and_transform(
                center_u, center_v,
                depth_value,
                self.top_intrinsics, self.T_top_to_base
            )
            if grasp_pos_3d is None:
                logger.error(f"   座標轉換失敗")
                return None

            object_position = [
                grasp_pos_3d[0] * 1000, grasp_pos_3d[1] * 1000, grasp_pos_3d[2] * 1000,
                180.0, 0.0, grip_info['angle']
            ]
            logger.info(f"   找到物體: [{object_position[0]:.1f}, {object_position[1]:.1f}, {object_position[2]:.1f}] mm")
            logger.info(f"      夾取角度: {grip_info['angle']:.1f}°")
            return {
                'position': object_position,
                'grip_angle': grip_info['angle'],
                'detection': detection
            }
        except Exception as e:
            logger.error(f"   _process_yolo_detection 異常: {e}")
            return None

    def _handle_occluded_object(self, target_obj, object_name: str, world_state) -> Dict:
        """處理被遮擋的物體
        
        Args:
            target_obj: WorldState 中的物體
            object_name: 物體名稱（用於返回值）
            world_state: 當前 WorldState
            
        Returns:
            遮擋處理指示字典
        """
        logger.warning(f"   物體狀態: {target_obj.visibility} - 觸發遮擋處理策略")
        
        # 計算遮擋時長
        current_time = time.time()
        time_occluded = current_time - target_obj.last_seen if target_obj.last_seen else 0
        
        # 使用 occlusion_handler 處理
        action = self.occlusion_handler.get_best_strategy(
            occluded_obj=target_obj,
            world_state=world_state,
            time_occluded=time_occluded
        )
        
        logger.info(f"   遮擋處理決策: {action['strategy']}")
        logger.info(f"      說明: {action.get('explanation', '')}")
        
        # 根據遮擋物類型決定處理策略
        occluder_id = action.get('occluder_id', '')
        occluder_name = action.get('occluder_name', '障礙物')
        
        # 物體名稱中英映射
        name_mapping = {
            'screw_big': '大螺絲',
            'screw_small': '小螺絲',
            'cylinder': '圓柱',
            'platform': '平台',
            'bracket': '支架',
            'screw': '螺絲'
        }
        target_name_zh = name_mapping.get(target_obj.object_type, target_obj.object_type)
        
        # 判斷遮擋物類型
        needs_human_removal = False
        occluder_type = occluder_id.split('_')[0].lower() if occluder_id and '_' in occluder_id else (occluder_id.lower() if occluder_id else 'unknown')
        
        if occluder_type in ['platform', 'bracket']:
            needs_human_removal = True
            logger.info(f"   遮擋物為 {occluder_name}（類型: {occluder_type}），需要人類移開")
        elif not occluder_id or occluder_type == 'unknown':
            needs_human_removal = True
            logger.info(f"   遮擋物類型不明，需要人類移開")
        else:
            needs_human_removal = False
            logger.info(f"   遮擋物為 {occluder_name}（類型: {occluder_type}），機器人可直接移除")
        
        if needs_human_removal:
            # 請人類移開：語音詢問
            question = f"{target_name_zh}被{occluder_name}遮住了，需要我幫你移開{occluder_name}嗎？還是你自己移開？"
            logger.info(f"   語音詢問: {question}")
            
            return {
                'occlusion_detected': True,
                'needs_voice_confirmation': True,
                'action': action,
                'object': target_obj,
                'question': question,
                'target_object': object_name,
                'target_object_zh': target_name_zh,
                'occluder_object': occluder_id,
                'occluder_name': occluder_name
            }
        else:
            # 機器人直接移除：返回自動執行標記
            notification = f"偵測到{target_name_zh}被{occluder_name}遮住，先將{occluder_name}移至放置區"
            logger.info(f"   語音通知: {notification}")
            
            return {
                'occlusion_detected': True,
                'auto_remove_occluder': True,
                'action': action,
                'object': target_obj,
                'notification': notification,
                'target_object': object_name,
                'target_object_zh': target_name_zh,
                'occluder_object': occluder_id,
                'occluder_name': occluder_name
            }
    
    def _detect_hand_with_d435i(self, max_attempts: int = 30) -> Optional[np.ndarray]:
        """
        使用 D435i 俯視相機 + MediaPipe 檢測手部位置（統一方法）
        
        Args:
            max_attempts: 最大嘗試幀數（預設 30 幀 ≈ 1 秒）
        
        Returns:
            hand_pos_3d: [x, y, z] in meters, 或 None
        """
        if not self.dual_camera_initialized:
            logger.warning("   雙相機系統未初始化")
            return None
        
        import cv2
        
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                top_frames = self.top_pipeline.wait_for_frames(timeout_ms=1000)
                top_aligned = self.top_align.process(top_frames)
                top_depth_frame = top_aligned.get_depth_frame()
                top_color_frame = top_aligned.get_color_frame()
                
                if not top_depth_frame or not top_color_frame:
                    continue
                
                top_color_image = np.asanyarray(top_color_frame.get_data())
                top_rgb = cv2.cvtColor(top_color_image, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(top_rgb)
                
                if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) > 0:
                    hand_landmarks = hand_results.multi_hand_landmarks[0]
                    wrist = hand_landmarks.landmark[0]
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]
                    
                    # 手掌中心：手腕加權 2.5 倍
                    palm_x = (wrist.x * 2.5 + index_tip.x + middle_tip.x) / 4.5
                    palm_y = (wrist.y * 2.5 + index_tip.y + middle_tip.y) / 4.5
                    
                    h, w = top_color_image.shape[:2]
                    pixel_x = int(palm_x * w)
                    pixel_y = int(palm_y * h)
                    depth = top_depth_frame.get_distance(pixel_x, pixel_y)
                    
                    if depth > 0:
                        hand_pos_3d = self._deproject_and_transform(
                            pixel_x, pixel_y, depth,
                            self.top_intrinsics, self.T_top_to_base
                        )
                        if hand_pos_3d is not None:
                            logger.info(f"      檢測到人手（D435i 俯視，第 {attempt} 幀）")
                            return hand_pos_3d
            except Exception as e:
                if attempt == 1:
                    logger.warning(f"      手部檢測失敗: {e}")
        
        logger.warning(f"   未檢測到人手（嘗試 {max_attempts} 幀）")
        return None
    
    def _create_simple_waypoint(self, action: str, position: List[float],
                               velocity: int = 100, wait_time: float = 0.0):
        """創建簡單的 Waypoint 對象（避免重複代碼）"""
        from ws_racr.execution.backends.waypoint_schema import Waypoint
        return Waypoint(
            action=action,
            position=position,
            velocity=velocity,
            acc_time=200,
            blend=0,
            constraints={},
            io_before=None,
            io_after=None,
            wait_time=wait_time
        )
    
    def _create_plan_json(self, waypoints: List, task_type: str, **metadata) -> Dict:
        """創建標準的 plan_json 格式（避免重複代碼）"""
        return {
            "waypoints": [wp.to_dict() if hasattr(wp, 'to_dict') else wp.__dict__ for wp in waypoints],
            "metadata": {
                "planner": self.planner_type,
                "task_type": task_type,
                **metadata
            }
        }
    
    # ==================== 任務規劃方法 ====================
    
    def _calculate_ray_score(self, p_hand, d_hat, p_obj):
        """
        計算射線與物體的匹配分數（使用 soft score）
        
        判斷邏輯：
        1. 角度誤差 (theta): 射線方向 與 (物體-手) 方向的夾角
        2. 垂直距離 (miss): 物體到射線的垂直距離
        3. 前方檢查 (forward): 物體必須在射線前方
        
        使用 soft score 避免 hard gating 導致的抖動失敗
        
        Returns:
            score, theta, miss, ok, dist
        """
        v = p_obj - p_hand
        dist = np.linalg.norm(v)
        
        if dist < 1e-6:
            return 0.0, 180.0, 1e9, False, dist
        
        v_norm = v / dist
        
        # 角度誤差
        cos_theta = np.clip(np.dot(d_hat, v_norm), -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))
        
        # 垂直距離
        miss = np.linalg.norm(np.cross(d_hat, v))
        
        # 前方檢查（硬約束，物體必須在射線前方）
        forward = np.dot(d_hat, v) > 0
        
        if not forward:
            # 物體在後方，直接給 0 分
            return 0.0, theta, miss, False, dist
        
        # 使用 soft score（指數衰減）避免 hard gating
        # score 會隨著 theta 和 miss 平滑下降，而非突然變 0
        score = np.exp(-theta / self.THETA_MAX_DEG) * \
                np.exp(-miss / self.MISS_MAX_M) * \
                np.exp(-0.05 * dist)
        
        # 判斷是否在合理範圍內（用於 debug 輸出）
        ok = (theta < self.THETA_MAX_DEG) and (miss < self.MISS_MAX_M)
        
        return score, theta, miss, ok, dist
    
    def _detect_pointed_object_dual_camera(self) -> Optional[Dict]:
        """
        使用雙相機 + 手部射線檢測指向的物體（從 hand_pointing_dual_camera.py 整合）
        
        Returns:
            {
                'class_name': str,
                'position': [x, y, z] (公尺),
                'confidence': float,
                'ray_info': {
                    'theta': float,  # 角度誤差（度）
                    'miss': float,   # 距離誤差（公尺）
                    'distance': float  # 手到物體的距離（公尺）
                }
            } or None
        """
        try:
            import cv2
            import numpy as np
            
            # 延遲初始化
            if not self.dual_camera_initialized:
                if not self._init_dual_camera_pointing():
                    return None
            
            logger.info(f"      雙相機射線檢測...")
            
            # === 1. 前視相機 (D455) - 手部 ===
            front_frames = self.front_pipeline.wait_for_frames()
            front_frames = self.front_align.process(front_frames)
            front_color = front_frames.get_color_frame()
            front_depth = front_frames.get_depth_frame()
            
            if not front_color or not front_depth:
                logger.warning(f"      前視相機無影像")
                return None
            
            front_img = np.asanyarray(front_color.get_data())
            
            # === 2. 俯視相機 (D435i) - 物體 ===
            top_frames = self.top_pipeline.wait_for_frames()
            top_frames = self.top_align.process(top_frames)
            top_color = top_frames.get_color_frame()
            top_depth = top_frames.get_depth_frame()
            
            if not top_color or not top_depth:
                logger.warning(f"      俯視相機無影像")
                return None
            
            top_img = np.asanyarray(top_color.get_data())
            
            # === 3. YOLO 檢測俯視相機的物體 ===
            results = self.yolo_model.predict(source=top_img, conf=0.3, verbose=False)
            objects = []
            
            if len(results) > 0:
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None:
                    boxes = r.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                    
                    # 檢查是否有 masks
                    masks = None
                    if hasattr(r, 'masks') and r.masks is not None:
                        masks = r.masks.data.cpu().numpy()  # (N, H, W)
                    
                    for idx, (bbox, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                        x1, y1, x2, y2 = map(int, bbox)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # 優先使用 mask 內的深度（避免混入背景）
                        depths = []
                        if masks is not None and idx < len(masks):
                            # 有 mask：只取 mask 內的像素深度
                            mask_resized = cv2.resize(
                                masks[idx], (top_img.shape[1], top_img.shape[0]),
                                interpolation=cv2.INTER_LINEAR
                            )
                            mask_binary = (mask_resized > 0.5).astype(np.uint8)
                            
                            # 在 bbox 範圍內且 mask=1 的像素
                            for v in range(max(0, y1), min(top_img.shape[0], y2), 2):
                                for u in range(max(0, x1), min(top_img.shape[1], x2), 2):
                                    if mask_binary[v, u] > 0:
                                        d = top_depth.get_distance(u, v)
                                        if d > 0:
                                            depths.append(d)
                        else:
                            # 無 mask：fallback 到 bbox 內的深度中位數
                            for v in range(y1, y2, 2):
                                for u in range(x1, x2, 2):
                                    d = top_depth.get_distance(u, v)
                                    if d > 0:
                                        depths.append(d)
                        
                        if len(depths) == 0:
                            continue
                        
                        depth = np.median(depths)
                        
                        # 轉換到世界座標（使用 pointing 專用外參）
                        pos_world = self._deproject_and_transform(
                            cx, cy, depth, self.top_intrinsics, self.T_top_to_base_for_pointing
                        )
                        
                        if pos_world is not None:
                            obj_data = {
                                'name': self.CLASS_MAPPING.get(int(class_id), "unknown"),
                                'confidence': float(confidence),
                                'pos': pos_world,
                                'depth': depth,
                                'cx': cx,  # 像素中心 x
                                'cy': cy   # 像素中心 y
                            }
                            
                            # 加入 mask（如果有）
                            if masks is not None and idx < len(masks):
                                obj_data['mask'] = masks[idx]
                            
                            objects.append(obj_data)
            
            if not objects:
                logger.warning(f"      未檢測到物體")
                return None
            
            logger.info(f"         檢測到 {len(objects)} 個物體")
            
            # === 4. MediaPipe 手部偵測 (前視相機) ===
            front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(front_rgb)
            
            if not hand_results.multi_hand_landmarks:
                logger.warning(f"      未檢測到手部")
                return None
            
            # 處理第一隻手
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            h, w = front_img.shape[:2]
            
            # 使用食指骨架：index_mcp (5) → index_tip (8)
            # 比 wrist→tip 更穩定，因為沿著手指骨架方向
            index_mcp = hand_landmarks.landmark[5]   # 食指根部
            index_tip = hand_landmarks.landmark[8]   # 食指尖
            
            # 轉換為像素座標
            mcp_u = int(index_mcp.x * w)
            mcp_v = int(index_mcp.y * h)
            tip_u = int(index_tip.x * w)
            tip_v = int(index_tip.y * h)
            
            # === 容錯機制：持續檢測直到獲得有效深度 ===
            max_retries = 10
            retry_count = 0
            depth_mcp = 0.0
            depth_tip = 0.0
            
            while retry_count < max_retries:
                if retry_count > 0:
                    logger.info(f"      重試深度檢測 ({retry_count}/{max_retries})...")
                    # 重新獲取影像
                    front_frames = self.front_pipeline.wait_for_frames()
                    front_frames = self.front_align.process(front_frames)
                    front_color = front_frames.get_color_frame()
                    front_depth = front_frames.get_depth_frame()
                    
                    if not front_color or not front_depth:
                        retry_count += 1
                        continue
                    
                    front_img = np.asanyarray(front_color.get_data())
                    front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
                    
                    # 重新檢測手部
                    hand_results = self.hands.process(front_rgb)
                    if not hand_results.multi_hand_landmarks:
                        retry_count += 1
                        continue
                    
                    hand_landmarks = hand_results.multi_hand_landmarks[0]
                    h, w = front_img.shape[:2]
                    
                    index_mcp = hand_landmarks.landmark[5]
                    index_tip = hand_landmarks.landmark[8]
                    mcp_u = int(index_mcp.x * w)
                    mcp_v = int(index_mcp.y * h)
                    tip_u = int(index_tip.x * w)
                    tip_v = int(index_tip.y * h)
                
                # 讀取深度值
                depth_mcp = front_depth.get_distance(mcp_u, mcp_v)
                depth_tip = front_depth.get_distance(tip_u, tip_v)
                
                # 逐步擴大搜索範圍（3x3 → 5x5 → 7x7）
                search_radius = 3 + (retry_count // 3) * 2  # 3, 5, 7...
                
                # 如果深度無效，嘗試周圍像素（MCP）
                if depth_mcp <= 0:
                    for dy in range(-search_radius, search_radius + 1):
                        for dx in range(-search_radius, search_radius + 1):
                            u, v = mcp_u + dx, mcp_v + dy
                            if 0 <= u < w and 0 <= v < h:
                                d = front_depth.get_distance(u, v)
                                if d > 0:
                                    depth_mcp = d
                                    break
                        if depth_mcp > 0:
                            break
                
                # 如果深度無效，嘗試周圍像素（TIP）
                if depth_tip <= 0:
                    for dy in range(-search_radius, search_radius + 1):
                        for dx in range(-search_radius, search_radius + 1):
                            u, v = tip_u + dx, tip_v + dy
                            if 0 <= u < w and 0 <= v < h:
                                d = front_depth.get_distance(u, v)
                                if d > 0:
                                    depth_tip = d
                                    break
                        if depth_tip > 0:
                            break
                
                # 如果 tip 還是無效，用 mcp 深度估算（假設手指長約10cm）
                if depth_mcp > 0 and depth_tip <= 0:
                    # 計算手指在圖像中的方向
                    dx_pixel = tip_u - mcp_u
                    dy_pixel = tip_v - mcp_v
                    pixel_dist = np.sqrt(dx_pixel**2 + dy_pixel**2)
                    
                    if pixel_dist > 10:  # 手指至少10像素長
                        # 估算深度變化（假設手指指向相機方向時深度減少）
                        estimated_depth_change = 0.05  # 假設5cm
                        depth_tip = depth_mcp - estimated_depth_change
                        logger.info(f"      使用估算的 TIP 深度: {depth_tip:.3f}m")
                
                # 檢查是否成功
                if depth_mcp > 0 and depth_tip > 0:
                    if retry_count > 0:
                        logger.info(f"      深度檢測成功 (重試 {retry_count} 次)")
                    break
                
                retry_count += 1
            
            # 最終檢查
            if depth_mcp <= 0 or depth_tip <= 0:
                logger.error(f"      深度檢測失敗 (mcp={depth_mcp:.3f}, tip={depth_tip:.3f})")
                logger.info(f"         已重試 {retry_count} 次，仍無有效深度值")
                return None
            
            # 使用 RealSense SDK 反投影到 3D 相機座標
            import pyrealsense2 as rs
            p_mcp_cam = rs.rs2_deproject_pixel_to_point(
                self.front_intrinsics, [mcp_u, mcp_v], depth_mcp
            )
            p_tip_cam = rs.rs2_deproject_pixel_to_point(
                self.front_intrinsics, [tip_u, tip_v], depth_tip
            )
            
            p_mcp_cam = np.array(p_mcp_cam, dtype=np.float32)
            p_tip_cam = np.array(p_tip_cam, dtype=np.float32)
            
            # 轉換到世界座標系
            p_mcp_world = self.T_front_to_base[:3, :3] @ p_mcp_cam + self.T_front_to_base[:3, 3]
            p_tip_world = self.T_front_to_base[:3, :3] @ p_tip_cam + self.T_front_to_base[:3, 3]
            
            # 射線方向：index_mcp → index_tip
            d = p_tip_world - p_mcp_world
            dist_hand = np.linalg.norm(d)
            if dist_hand < 1e-6:
                logger.warning(f"      手部座標異常")
                return None
            
            d_hat = d / dist_hand
            
            # 射線起點使用食指尖（已含 bias）
            p_hand = p_tip_world
            
            logger.info(f"         MCP深度: {depth_mcp:.3f}m, TIP深度: {depth_tip:.3f}m")
            logger.info(f"         手指長度: {dist_hand*1000:.1f}mm")
            
            logger.info(f"         手部射線: 起點={p_hand}, 方向={d_hat}")
            
            # === 5. 射線與物體匹配（使用 soft score）===
            candidates = []
            for obj in objects:
                score, theta, miss, ok, dist = self._calculate_ray_score(
                    p_hand, d_hat, obj['pos']
                )
                
                candidates.append({
                    'obj': obj,
                    'score': score,
                    'theta': theta,
                    'miss': miss,
                    'dist': dist,
                    'ok': ok
                })
            
            # 按 score 排序（降序）
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 檢查 top-1 是否超過閾值（soft score 需要最小信心度）
            SCORE_THRESHOLD = 0.1  # exp(-1) ≈ 0.37, 這裡設較低閾值允許稍遠的物體
            
            if len(candidates) == 0 or candidates[0]['score'] < SCORE_THRESHOLD:
                logger.warning(f"      未找到匹配的物體")
                if len(candidates) > 0:
                    logger.info(f"         最佳候選 score={candidates[0]['score']:.3f} < {SCORE_THRESHOLD}")
                return None
            
            best = candidates[0]
            
            # 返回結果（包含 mask 和原始檢測資訊用於重新計算）
            result = {
                'class_name': best['obj']['name'],
                'position': best['obj']['pos'].tolist(),  # 用 withoutgripper 外參計算（僅用於射線匹配）
                'confidence': best['obj']['confidence'],
                'mask': best['obj'].get('mask', None),  # 包含 mask（如果有）
                'depth': best['obj']['depth'],  # 原始深度（公尺）
                'pixel_center': (best['obj']['cx'], best['obj']['cy']),  # 像素中心
                'ray_info': {
                    'theta': best['theta'],
                    'miss': best['miss'],
                    'distance': best['dist']
                }
            }
            
            logger.info(f"      指向物體: {result['class_name']}")
            logger.info(f"         位置: [{result['position'][0]:.3f}, {result['position'][1]:.3f}, {result['position'][2]:.3f}] m")
            logger.info(f"         角度誤差: {result['ray_info']['theta']:.1f}°")
            logger.info(f"         距離誤差: {result['ray_info']['miss']*100:.1f} cm")
            
            return result
            
        except Exception as e:
            logger.error(f"      雙相機檢測失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
