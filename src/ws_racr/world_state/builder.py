"""
World State Builder
Central integration layer that aggregates all perception modules
"""
import logging
from ws_racr.config import get_data_dir
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path

from ws_racr.world_state.state_representation import (
    WorldState, ObjectState, HumanState, RobotStatus, AssemblyProgress,
    SafetyState, IntentState, RobotState, InteractionState
)
from ws_racr.world_state.spatial_relations import SpatialRelationCalculator, SpatialRelation
from ws_racr.world_state.workspace_config import (
    is_valid_object_id, get_expected_count, validate_detections,
    EXPECTED_OBJECTS, TOTAL_EXPECTED_OBJECTS
)

logger = logging.getLogger(__name__)


class WorldStateBuilder:
    """
    Aggregates inputs from all perception modules and builds unified world state
    
    Input Sources:
    1. Step Recognition: step_class, confidence, phase
    2. Object Detection: object detections with positions
    3. Gesture Detection: intent, safety, gesture confidence
    4. Assembly Tracker: progress tracking from SOP
    5. Robot Controller: robot state and action status
    """
    
    def __init__(self):
        self.current_state = WorldState()
        self.previous_state = None
        self._state_lock = threading.RLock()  # [Fix-3] 防止 perception 寫、worker 讀同時發生 race condition

        # === Progress Tracker (authoritative step source) ===
        # When set, get_current_state() uses progress_tracker.confirmed_step
        # instead of raw vision output so manual overrides and smoothing persist.
        self.progress_tracker = None  # set via set_progress_tracker()

        # === Robot Controller (for IO Truth) ===
        self.robot_controller = None  # 由外部設定
        
        # === 單位統一：全部使用 mm ===
        self.UNIT = 'mm'  # 統一單位：毫米
        
        # Object tracking
        self.object_history: Dict[str, List[ObjectState]] = {}
        self.object_id_counter = 0
        
        # === 簡化的物體追蹤===
        # 不使用固定 ID 池，改用簡單的 nearest-neighbor
        self.tracked_objects: Dict[str, Dict] = {}  # {object_id: {'position': [x,y,z], 'last_seen': time, 'class_name': str}}
        self.next_id_counter: Dict[str, int] = {}  # {class_name: next_id}
        
        # 匹配閘值（根據物體類型調整）
        self.position_match_threshold = {
            'platform': 300.0,      # 30cm - 大型靜態物體（檢測位置變化大）
            'bracket': 200.0,       # 20cm - 中型物體
            'pi': 200.0,            # 20cm - 中型物體
            'cylinder': 100.0,      # 10cm - 小型物體
            'nut': 80.0,            # 8cm - 小型精細物體
            'screw_big': 80.0,
            'screw_small': 60.0,
            'hex_wrench': 150.0,
            'default': 150.0        # 15cm - 預設值
        }
        
        # Stability thresholds（只對幾何狀態生效）
        self.geom_stable_hits = 3  # 連續 3 次一致才更新 geom_stable
        
        # Class-specific occlusion/missing timeouts
        self.occlusion_timeout = {
            'platform': 5.0,    # 大型靜態物體，更寬容
            'bracket': 3.0,
            'pi': 3.0,
            'default': 2.0      # 其他物體 2 秒
        }
        self.missing_timeout = {
            'platform': 10.0,   # 大型靜態物體，更寬容
            'bracket': 7.0,
            'pi': 7.0,
            'default': 5.0      # 其他物體 5 秒
        }
        
        self.reappear_confirm_hits = 3  # missing 後需連續 3 次檢測才變 visible
        self.id_ttl = 10.0  # 10 秒不見就釋放 ID（簡化）
        
        # Debug log 上限（避免記憶體洩漏）
        self.max_log_frames = 300
        
        # Confidence thresholds（過濾低信心度檢測）
        self.min_confidence = {
            'cylinder': 0.75,    # 圓柱需要更高信心度（容易誤檢）
            'nut': 0.7,          # 螺帽也容易誤檢
            'screw_small': 0.65,
            'screw_big': 0.7,
            'platform': 0.5,     # 大型物體較穩定
            'bracket': 0.65,
            'pi': 0.65,
            'hex_wrench': 0.65,
            'default': 0.6       # 預設閘值
        }
        
        # 對於 missing 物體重新出現，要求更高的信心度
        self.reappear_min_confidence = {
            'cylinder': 0.85,    # 重現需要更高信心度
            'nut': 0.8,
            'screw_small': 0.75,
            'screw_big': 0.8,
            'default': 0.75
        }
        
        # === 桌面高度（優先載入 RANSAC 校準結果）===
        # 相機座標系：Z = 相機到物體的距離（mm）
        self.table_z_fixed = None
        self.table_z_calibrated = False
        self.calibration_frame_count = 0
        
        # 載入 RANSAC 校準結果
        self._load_table_calibration()
        
        # KG 輔助追蹤
        self.use_kg_for_tracking = False
        self.memory_system = None
        
        # Step tracking
        self.step_history: List[Tuple[str, float]] = []
        self.step_duration_threshold = 2.0
        
        # Safety tracking
        self.safety_violation_count = 0
        self.safety_violation_threshold = 3
        
        # Spatial relations calculator
        self.spatial_calc = SpatialRelationCalculator()
        
        # 清理超過數量限制的已追蹤物體
        self._cleanup_excess_tracked_objects()
    
    def _cleanup_excess_tracked_objects(self):
        """清理超過 expected_count 的已追蹤物體"""
        from collections import defaultdict
        
        # 統計每個類別的物體數
        class_counts = defaultdict(list)
        for obj_id, info in list(self.tracked_objects.items()):
            class_name = info['class_name']
            class_counts[class_name].append((obj_id, info['last_seen']))
        
        # 對每個類別檢查數量限制
        for class_name, obj_list in class_counts.items():
            expected_count = get_expected_count(class_name)
            if expected_count > 0 and len(obj_list) > expected_count:
                # 保留最近的 expected_count 個
                obj_list.sort(key=lambda x: x[1], reverse=True)
                for obj_id, _ in obj_list[expected_count:]:
                    del self.tracked_objects[obj_id]
                    if obj_id in self.current_state.objects:
                        del self.current_state.objects[obj_id]
                logger.info(f"已清理 {class_name}: {len(obj_list)} → {expected_count} 個")
    
    def _cleanup_excess_objects_per_frame(self):
        """每幀清理超過 expected_count 的物體"""
        from collections import defaultdict
        import time
        
        current_time = time.time()
        
        # 統計每個類別的物體
        class_objects = defaultdict(list)
        for obj_id, obj in list(self.current_state.objects.items()):
            class_name = obj.object_type
            class_objects[class_name].append((obj_id, obj))
        
        # 對每個類別檢查數量限制
        for class_name, obj_list in class_objects.items():
            expected_count = get_expected_count(class_name)
            if expected_count > 0 and len(obj_list) > expected_count:
                # 保留最近見到的 expected_count 個
                obj_list.sort(key=lambda x: x[1].last_seen, reverse=True)
                
                # 刪除多餘的
                for obj_id, _ in obj_list[expected_count:]:
                    if obj_id in self.current_state.objects:
                        del self.current_state.objects[obj_id]
                    if obj_id in self.tracked_objects:
                        del self.tracked_objects[obj_id]
                
                # 只在第一次清理時顯示警告
                if not hasattr(self, '_cleanup_warned'):
                    self._cleanup_warned = {}
                if class_name not in self._cleanup_warned:
                    logger.info(f"清理 {class_name}: {len(obj_list)} → {expected_count} 個")
                    self._cleanup_warned[class_name] = True
    
    def _load_table_calibration(self):
        """載入 RANSAC 桌面校準結果"""
        from pathlib import Path
        import json
        
        config_path = get_data_dir() / "calibration" / "table_roi.json"
        if not config_path.exists():
            logger.warning("未找到 RANSAC 校準配置，將使用自動校準")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            reference_z = config.get('plane_reference_z')
            if reference_z is not None:
                self.table_z_fixed = float(reference_z)
                self.table_z_calibrated = True
                logger.info(f"已載入 RANSAC 桌面校準：{self.table_z_fixed:.1f} mm")
            else:
                logger.warning("配置中無 plane_reference_z，將使用自動校準")
        except Exception as e:
            logger.warning(f"載入校準配置失敗：{e}")
    
    def set_memory_system(self, memory_system):
        """設置 memory_system"""
        self.memory_system = memory_system
        logger.info("WorldStateBuilder: memory_system 已設置")
    
    def _normalize_position(self, pos: List[float]) -> List[float]:
        """統一 position 單位為 mm"""
        if pos is None or len(pos) < 3:
            return None
        
        # 檢測單位：如果絕對值都 < 5，很可能是公尺（轉換為 mm）
        if abs(pos[0]) < 5 and abs(pos[1]) < 5 and abs(pos[2]) < 5:
            # 公尺 → 毫米
            return [pos[0] * 1000, pos[1] * 1000, pos[2] * 1000]
        else:
            # 已經是毫米
            return pos
        
    def update_step_recognition(self, step_class: str, confidence: float, phase: int = 0):
        """
        Update from step recognition module
        
        Args:
            step_class: Predicted step (e.g., "idle", "pick_cylinder", etc.)
            confidence: Prediction confidence [0-1]
            phase: Assembly phase [0-3]
        """
        self.current_state.step_class = step_class
        self.current_state.step_confidence = confidence
        self.current_state.phase = phase
        
        # Track step history
        current_time = time.time()
        self.step_history.append((step_class, current_time))
        
        # Keep only recent history (last 30 seconds)
        cutoff_time = current_time - 30.0
        self.step_history = [(s, t) for s, t in self.step_history if t > cutoff_time]
    
    def update_object_detection(self, detections: List[Dict]):
        """
        Update from object detection module
        
        Args:
            detections: List of detection dicts with keys:
                - class_name: str (e.g., "cylinder", "platform")
                - confidence: float
                - bbox: [x1, y1, x2, y2]
                - position: Optional[List[float]] - 3D position if available
                - mask: Optional[np.ndarray] - segmentation mask
        """
        current_time = time.time()
        
        # 修復 3: 桌面校準每幀只 +1（不是每個物體 +1）
        if not self.table_z_calibrated:
            self.calibration_frame_count += 1
        
        # === 最前面先 normalize position（統一單位）===
        for det in detections:
            if 'position' in det and det['position'] is not None:
                det['position'] = self._normalize_position(det['position'])
        
        # === 第一層過濾：class-level confidence threshold ===
        detections_by_type: Dict[str, List[Dict]] = {}
        filtered_count = 0
        
        for det in detections:
            class_name = det['class_name']
            confidence = det.get('confidence', 1.0)
            
            # 只用 class-level min_confidence（不在這裡處理 reappear）
            min_conf = self.min_confidence.get(class_name, self.min_confidence['default'])
            
            if confidence < min_conf:
                filtered_count += 1
                continue  # 過濾掉低信心度檢測
            
            if class_name not in detections_by_type:
                detections_by_type[class_name] = []
            detections_by_type[class_name].append(det)
        
        if not hasattr(self, '_debug_frame_counter'):
            self._debug_frame_counter = 0
        self._debug_frame_counter += 1
        
        # Update or create object states
        updated_object_ids = set()
        
        # 調試：記錄這一幀更新了哪些物體
        if not hasattr(self, '_frame_update_log'):
            self._frame_update_log = {}
        
        # 限制 log 大小（避免記憶體洩漏）
        if len(self._frame_update_log) > self.max_log_frames:
            oldest = min(self._frame_update_log.keys())
            del self._frame_update_log[oldest]
        
        self._frame_update_log[self._debug_frame_counter] = []
        
        for class_name, dets in detections_by_type.items():
            # === 驗證數量：嚴格不超過預期數量 ===
            expected_count = get_expected_count(class_name)
            if expected_count > 0:
                if len(dets) > expected_count:
                    # 降低警告頻率：每30幀顯示一次
                    if not hasattr(self, '_warning_counter'):
                        self._warning_counter = {}
                    if class_name not in self._warning_counter:
                        self._warning_counter[class_name] = 0
                    self._warning_counter[class_name] += 1
                    
                    pass  # truncation is normal; suppress per-frame warning
                    
                    # 保留置信度最高的 N 個
                    dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)[:expected_count]
            
            # === 簡化的物體追蹤：nearest-neighbor matching ===
            matched_ids = self._match_detections_simple(class_name, dets, current_time, expected_count)
            
            # 分配檢測到物體（過濾掉 None）
            for det, object_id in zip(dets, matched_ids):
                if object_id is None:  # 達到數量上限，跳過
                    continue
                
                # 獲取檢測信心度和位置
                confidence = det.get('confidence', 1.0)
                position = det.get('position')
                
                # Object-level reappear gate
                old_obj = self.current_state.objects.get(object_id)
                if old_obj and hasattr(old_obj, 'visibility') and old_obj.visibility in ['missing', 'occluded']:
                    reappear_conf = self.reappear_min_confidence.get(class_name, self.reappear_min_confidence['default'])
                    if confidence < reappear_conf:
                        continue  # 靜默過濾
                
                # Infer geometry state (only)
                geom_state = self._infer_geometry_state(det, class_name)
                
                # === Geometry Stability: N-hit confirm ===
                geom_stable = self._update_geom_stable(object_id, geom_state)
                
                # === Visibility 重現確認：防止誤檢就重置 ===
                # old_obj 已在上面檢查過
                should_be_visible = True
                
                if old_obj and old_obj.visibility in ['missing', 'occluded']:
                    # 物體之前是 missing/occluded，需要確認
                    if object_id not in self.tracked_objects:
                        self.tracked_objects[object_id] = {
                            'position': det.get('position'),
                            'last_seen': current_time,
                            'class_name': class_name,
                            'visibility': old_obj.visibility,
                            'geom_candidate': geom_state,
                            'geom_counter': 0,
                            'reappear_counter': 0
                        }
                    
                    track = self.tracked_objects[object_id]
                    reappear_counter = track.get('reappear_counter', 0) + 1
                    
                    if reappear_counter < self.reappear_confirm_hits:
                        # 還沒達到確認次數，保持 missing/occluded
                        should_be_visible = False
                        if object_id in self.tracked_objects:
                            self.tracked_objects[object_id]['reappear_counter'] = reappear_counter
                    else:
                        # 達到確認次數，重置計數器
                        if object_id in self.tracked_objects:
                            self.tracked_objects[object_id]['reappear_counter'] = 0
                        pass  # visibility restored
                
                # 決定最終的 visibility
                final_visibility = 'visible' if should_be_visible else old_obj.visibility if old_obj else 'visible'
                
                # Create or update object state
                obj_state = ObjectState(
                    object_type=class_name,
                    object_id=object_id,
                    location=geom_state,  # 幾何狀態（on_table/above_table/below_table）
                    location_stable=geom_stable,  # 穩定後的幾何狀態
                    position=det.get('position'),
                    confidence=det['confidence'],
                    last_seen=current_time,
                    visibility=final_visibility
                )
                
                self.current_state.update_object(obj_state)
                updated_object_ids.add(object_id)
                
                # 調試：記錄更新
                if hasattr(self, '_frame_update_log'):
                    self._frame_update_log[self._debug_frame_counter].append(object_id)
                
                # 修復 2: 不覆蓋 tracked_objects（避免 counter 被重置）
                if det.get('position') is not None:
                    track = self.tracked_objects.get(object_id)
                    if track is None:
                        # 新物體：建立完整記錄
                        self.tracked_objects[object_id] = {
                            'position': det['position'],
                            'last_seen': current_time,
                            'class_name': class_name,
                            'visibility': 'visible',
                            'geom_candidate': geom_state,
                            'geom_counter': 0,
                        }
                    else:
                        # 已追蹤物體：只更新部分欄位（geom_candidate/counter 由 _update_geom_stable 管理）
                        track['position'] = det['position']
                        track['last_seen'] = current_time
                        if final_visibility == 'visible':
                            track['visibility'] = 'visible'
                            # 重置 reappear_counter
                            track['reappear_counter'] = 0
        
        # === Visibility 狀態機：處理未檢測到的物體 ===
        for obj_id, obj in list(self.current_state.objects.items()):
            if obj_id not in updated_object_ids:
                # 計算自上次見到以來的時間
                if obj_id in self.tracked_objects:
                    time_since_seen = current_time - self.tracked_objects[obj_id]['last_seen']
                    old_visibility = obj.visibility
                    
                    # 獲取 class-specific timeout
                    class_name = obj_id.rsplit('_', 1)[0]  # cylinder_1 -> cylinder
                    occl_timeout = self.occlusion_timeout.get(class_name, self.occlusion_timeout['default'])
                    miss_timeout = self.missing_timeout.get(class_name, self.missing_timeout['default'])
                    
                    if time_since_seen < occl_timeout:
                        # 短暫不見 → occluded（可能遮擋）
                        obj.visibility = 'occluded'
                        self.tracked_objects[obj_id]['visibility'] = 'occluded'
                    elif time_since_seen < miss_timeout:
                        # 較長時間不見 → 仍是 occluded
                        obj.visibility = 'occluded'
                        self.tracked_objects[obj_id]['visibility'] = 'occluded'
                    else:
                        # 超過閾值 → missing
                        obj.visibility = 'missing'
                        self.tracked_objects[obj_id]['visibility'] = 'missing'
                    

                    
                    # 更新 current_state 中的物體
                    self.current_state.objects[obj_id] = obj
                else:
                    # 不在 tracked_objects 中：可能是舊殘留，直接刪除
                    if obj_id in self.current_state.objects:
                        del self.current_state.objects[obj_id]
        
        # === 每幀清理：移除超過 expected_count 的物體 ===
        self._cleanup_excess_objects_per_frame()
    
    def update_gesture_detection(self, intent: str, safety: str, 
                                confidence: float, raw_gesture: str = "",
                                pointing_target: Optional[str] = None):
        """
        Update from gesture detection module
        
        Args:
            intent: Human intent (idle, receive, pointing)
            safety: Safety state (OK, WARNING, EMERGENCY_STOP)
            confidence: Gesture confidence [0-1]
            raw_gesture: Raw gesture classification
            pointing_target: Object ID if pointing gesture
        """
        self.current_state.human.intent = intent
        self.current_state.human.safety = safety
        self.current_state.human.confidence = confidence
        self.current_state.human.gesture_raw = raw_gesture
        self.current_state.human.pointing_target = pointing_target
        
        # Track safety violations
        if safety == SafetyState.EMERGENCY_STOP.value:
            self.safety_violation_count += 1
        else:
            self.safety_violation_count = 0
    
    def update_robot_status(self, state: str, current_action: Optional[str] = None,
                           last_action: Optional[str] = None,
                           action_success: Optional[bool] = None,
                           gripper_state: Optional[str] = None,
                           holding_object_id: Optional[str] = None):
        """
        Update robot status（包含 Robot IO 真值）
        
        Args:
            state: Robot state (idle, moving, grasping, etc.)
            current_action: Description of current action
            last_action: Description of last completed action
            action_success: Whether last action succeeded
            gripper_state: Gripper state from IO (open/closed/unknown)
            holding_object_id: Which object is held (None if empty)
        """
        self.current_state.robot.state = state
        self.current_state.robot.current_action = current_action
        self.current_state.robot.last_action = last_action
        self.current_state.robot.action_success = action_success
        
        # === Robot IO 真值 ===
        if gripper_state is not None:
            self.current_state.robot.gripper_state = gripper_state
        if holding_object_id is not None:
            self.current_state.robot.holding_object_id = holding_object_id
            
            # 如果 action_success=True 且 holding_object，標記物體狀態
            # 不寫進 geom_stable（那是幾何層）
            if action_success and holding_object_id and holding_object_id in self.current_state.objects:
                obj = self.current_state.objects[holding_object_id]
                obj.visibility = 'visible'
                # interaction_state is a computed property on WorldState —
                # no explicit write needed; it derives from human.intent + robot.gripper_state.
    
    def update_assembly_progress(self, progress: AssemblyProgress):
        """Update assembly progress tracking"""
        self.current_state.assembly_progress = progress
    
    def _update_geom_stable(self, object_id: str, current_geom: str) -> str:
        """
        Geometry Stability: N-hit confirm（只對幾何狀態）
        
        Returns: 穩定後的幾何狀態
        """
        if object_id not in self.tracked_objects:
            return current_geom
        
        track = self.tracked_objects[object_id]
        candidate = track.get('geom_candidate', current_geom)
        counter = track.get('geom_counter', 0)
        
        if current_geom == candidate:
            counter += 1
            track['geom_counter'] = counter
            track['geom_candidate'] = candidate
            
            if counter >= self.geom_stable_hits:
                return current_geom
            else:
                # 返回上一幀的 stable（從 previous_state 讀取）
                if self.previous_state and object_id in self.previous_state.objects:
                    return self.previous_state.objects[object_id].location_stable
                return current_geom
        else:
            track['geom_candidate'] = current_geom
            track['geom_counter'] = 1
            
            if self.previous_state and object_id in self.previous_state.objects:
                return self.previous_state.objects[object_id].location_stable
            return current_geom
    
    def _infer_geometry_state(self, detection: Dict, class_name: str) -> str:
        """
        推理幾何狀態（只做幾何判斷，不做語義推理）
        
        Returns: "on_table" / "above_table" / "below_table" / "unknown"
        """
        pos = detection.get('position')
        
        if pos is None or len(pos) < 3:
            return "unknown"
        
        # === 桌面高度校準（前 30 幀）===
        # 收集 z 值（frame_count 在 update_object_detection 已 +1）
        if not self.table_z_calibrated:
            if not hasattr(self, '_calibration_z_buffer'):
                self._calibration_z_buffer = []
            
            # 收集當前檢測的 z 值
            obj_z = pos[2]
            if 300 < obj_z < 1000:  # 過濾異常值
                self._calibration_z_buffer.append(obj_z)
            
            if self.calibration_frame_count >= 30 and len(self._calibration_z_buffer) >= 10:
                # 用中位數（更穩健）
                self._calibration_z_buffer.sort()
                median_idx = len(self._calibration_z_buffer) // 2
                self.table_z_fixed = self._calibration_z_buffer[median_idx]
                self.table_z_calibrated = True
                logger.info(f"\n桌面高度自動校準完成：{self.table_z_fixed:.0f} mm（中位數）")
                logger.warning("建議使用 table_plane_calibration.py 進行 RANSAC 校準以獲得更穩健的結果\n")
                del self._calibration_z_buffer  # 釋放記憶體
        
        # === 幾何判斷：基於桌面高度 ===
        table_z = self.table_z_fixed if self.table_z_fixed else 600.0
        object_z = pos[2]
        z_diff = object_z - table_z
        
        # 只做幾何層判斷
        if abs(z_diff) < 100:  # ±10cm
            return "on_table"
        elif z_diff < -100:  # 近相機 >10cm
            return "above_table"
        elif z_diff > 100:  # 遠相機 >10cm
            return "below_table"
        else:
            return "on_table"
    
    def _match_detections_simple(self, class_name: str,
                                  detections: List[Dict],
                                  current_time: float,
                                  expected_count: int = 0) -> List[str]:
        """
        簡化的物體匹配：nearest-neighbor
        
        Args:
            expected_count: 預期數量（0 = 不限制）
        
        Returns: 每個檢測的 object_id
        """
        # 找出該類別的已追蹤物體
        tracked_candidates = {}
        for obj_id, track_info in list(self.tracked_objects.items()):
            if track_info['class_name'] == class_name:
                # 超時清理
                if current_time - track_info['last_seen'] > self.id_ttl:
                    del self.tracked_objects[obj_id]
                    continue
                tracked_candidates[obj_id] = track_info
        
        # 沒有歷史：直接分配新 ID
        if not tracked_candidates:
            result_ids = []
            for i in range(len(detections)):
                # 檢查 expected_count 限制
                if expected_count > 0 and i >= expected_count:
                    break
                
                new_id = self._allocate_available_id(class_name, expected_count)
                if new_id:
                    result_ids.append(new_id)
            return result_ids
        
        # 提取位置
        det_positions = []
        det_indices = []
        for i, det in enumerate(detections):
            pos = det.get('position')
            if pos and len(pos) >= 3:
                det_positions.append(np.array(pos[:3]))
                det_indices.append(i)
        
        # 沒有 3D 位置：直接分配
        if not det_positions:
            result_ids = []
            for _ in detections:
                new_id = self._allocate_available_id(class_name, expected_count)
                if new_id:
                    result_ids.append(new_id)
            return result_ids
        
        # Nearest-neighbor matching
        tracked_ids = list(tracked_candidates.keys())
        # 過濾掉 position 為 None 的追蹤物體（深度計算失敗或降採樣未檢測）
        tracked_positions = []
        valid_tracked_ids = []
        for obj_id in tracked_ids:
            pos = tracked_candidates[obj_id]['position']
            if pos is not None and len(pos) >= 3:
                tracked_positions.append(np.array(pos[:3]))
                valid_tracked_ids.append(obj_id)
        
        # 如果沒有有效的追蹤位置，分配新 ID
        if not tracked_positions:
            result_ids = []
            for _ in detections:
                new_id = self._allocate_available_id(class_name, expected_count)
                if new_id:
                    result_ids.append(new_id)
            return result_ids
        
        # 獲取該類別的匹配閘值
        threshold = self.position_match_threshold.get(class_name, 
                                                       self.position_match_threshold['default'])
        
        result_ids = [None] * len(detections)
        matched_tracks = set()
        
        # 為每個檢測找最近的追蹤物體
        for i, det_pos in enumerate(det_positions):
            best_dist = float('inf')
            best_track_idx = -1
            
            for j, track_pos in enumerate(tracked_positions):
                if j in matched_tracks:
                    continue
                dist = np.linalg.norm(det_pos - track_pos)
                if dist < best_dist and dist < threshold:
                    best_dist = dist
                    best_track_idx = j
            
            det_idx_full = det_indices[i]
            if best_track_idx >= 0:
                # 匹配成功：使用舊 ID
                result_ids[det_idx_full] = valid_tracked_ids[best_track_idx]
                matched_tracks.add(best_track_idx)
            else:
                # 無匹配：檢查是否能分配新 ID
                if expected_count > 0:
                    # 計算已分配的 ID 數
                    existing_count = sum(1 for oid in self.tracked_objects 
                                        if self.tracked_objects[oid]['class_name'] == class_name)
                    if existing_count >= expected_count:
                        # 已達上限，不分配新 ID（跳過此檢測）
                        if not hasattr(self, '_id_limit_warned'):
                            self._id_limit_warned = {}
                        if class_name not in self._id_limit_warned:
                            self._id_limit_warned[class_name] = True  # suppress repeat ID-limit prints
                        continue
                
                # 分配新 ID（重用範圍內的 ID）
                new_id = self._allocate_available_id(class_name, expected_count)
                if new_id:
                    result_ids[det_idx_full] = new_id
        
        # 處理沒有 3D 位置的檢測
        for i, result_id in enumerate(result_ids):
            if result_id is None:
                new_id = self._allocate_available_id(class_name, expected_count)
                if new_id:
                    result_ids[i] = new_id
        
        # 保持長度對齊：不過濾 None，讓 zip 正確對應
        return result_ids
    
    def _allocate_available_id(self, class_name: str, expected_count: int) -> Optional[str]:
        """
        分配可用的 ID（重用 1~expected_count 範圍內的 ID）
        
        Returns:
            可用的 ID，或 None（如果已達上限）
        """
        if expected_count <= 0:
            # 不限制：使用舊的 counter 邏輯
            if class_name not in self.next_id_counter:
                self.next_id_counter[class_name] = 1
            new_id = f"{class_name}_{self.next_id_counter[class_name]}"
            self.next_id_counter[class_name] += 1
            return new_id
        
        # 有限制：找出 1~expected_count 範圍內未使用的最小 ID
        used_ids = set()
        for obj_id in self.tracked_objects:
            if self.tracked_objects[obj_id]['class_name'] == class_name:
                # 提取 ID 號碼
                try:
                    id_num = int(obj_id.split('_')[-1])
                    used_ids.add(id_num)
                except ValueError:
                    pass
        
        # 找出最小未使用的 ID
        for id_num in range(1, expected_count + 1):
            if id_num not in used_ids:
                return f"{class_name}_{id_num}"
        
        # 已達上限
        return None
    
    def _compute_relation_similarity_matrix(self, class_name: str,
                                           det_positions: List[np.ndarray],
                                           det_indices: List[int],
                                           tracked_ids: List[str],
                                           tracked_candidates: Dict,
                                           detections: List[Dict]) -> np.ndarray:
        """
        計算檢測物體與已追蹤物體的關係相似度矩陣
        
        原理：比較物體的「空間特徵簽名」
        - 例如：cylinder_1 = "left_of:platform, above:surface, near:nut_1"
        - 如果新檢測的圓柱也是「左邊、平台上、靠近螺母」→ 很可能是同一個
        
        Args:
            class_name: 物體類別
            det_positions: 檢測的 3D 位置列表
            det_indices: 檢測在原始列表中的索引
            tracked_ids: 已追蹤物體的 ID 列表
            tracked_candidates: 已追蹤物體的信息
            detections: 原始檢測列表
        
        Returns:
            相似度矩陣 [len(det_positions) x len(tracked_ids)]，值域 [0, 1]
        """
        n_det = len(det_positions)
        n_track = len(tracked_ids)
        sim_matrix = np.zeros((n_det, n_track))
        
        if not self.memory_system or not hasattr(self.memory_system, 'spatial_memory'):
            return sim_matrix
        
        kg = self.memory_system.spatial_memory
        
        # 獲取所有穩定的參考物體（用於計算相對關係）
        reference_objects = {}
        for obj_id in self.current_state.objects.keys():
            if obj_id.split('_')[0] != class_name:  # 不同類別的物體
                obj = self.current_state.objects[obj_id]
                if obj.position is not None:
                    reference_objects[obj_id] = np.array(obj.position[:3])
        
        # 如果沒有參考物體，無法計算關係
        if not reference_objects:
            return sim_matrix
        
        # 為每個已追蹤物體提取空間簽名
        tracked_signatures = {}
        for obj_id in tracked_ids:
            # 從 KG 獲取該物體的關係
            relations = kg.get_relations_by_source(obj_id) + kg.get_relations_by_target(obj_id)
            signature = set()
            for rel in relations:
                # 提取關係類型和目標/來源
                other_id = rel.target_id if rel.source_id == obj_id else rel.source_id
                signature.add(f"{rel.relation_type}:{other_id}")
            tracked_signatures[obj_id] = signature
        
        # 為每個新檢測計算臨時空間簽名
        for i, det_pos in enumerate(det_positions):
            det_signature = set()
            
            # 計算與參考物體的關係
            for ref_id, ref_pos in reference_objects.items():
                # 使用 spatial_calc 計算相對位置
                rel_type, dist, _ = self.spatial_calc.compute_relative_position(
                    det_pos.tolist(), ref_pos.tolist()
                )
                if dist < 0.5:  # 50cm 內才記錄關係
                    det_signature.add(f"{rel_type}:{ref_id}")
            
            # 與每個已追蹤物體比較簽名相似度
            for j, tracked_id in enumerate(tracked_ids):
                tracked_sig = tracked_signatures.get(tracked_id, set())
                
                if not tracked_sig and not det_signature:
                    sim_matrix[i, j] = 0.5  # 都沒有關係，中等相似度
                elif not tracked_sig or not det_signature:
                    sim_matrix[i, j] = 0.0  # 一個有關係一個沒有，不相似
                else:
                    # Jaccard 相似度：交集 / 並集
                    intersection = len(tracked_sig & det_signature)
                    union = len(tracked_sig | det_signature)
                    sim_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        return sim_matrix
    
    def apply_robot_io_truth(self):
        """
        Apply Robot IO Truth to World State
        
        從 Robot Controller 讀取 holding_object，強制更新 semantic_location
        並實現 position freeze（夾取時位置不變）
        """
        if not self.robot_controller or not hasattr(self.robot_controller, 'facts'):
            return
        
        try:
            facts = self.robot_controller.facts
            holding_obj_id = facts.holding_object  # e.g., 'cylinder_1'
            
            # === 1. 更新所有物體的 semantic_location ===
            for obj_id, obj in self.current_state.objects.items():
                if obj_id == holding_obj_id:
                    # Robot 正在握持
                    obj.semantic_location = 'in_gripper'
                    obj.forced_by_io = True
                    obj.source = 'robot_io'
                    
                    # Position freeze: 夾取後保持最後一次視覺位置
                    if obj.last_vision_position is not None:
                        obj.position = obj.last_vision_position.copy()
                    
                else:
                    # 不在手上
                    if obj.forced_by_io:
                        # 剛放下：恢復視覺驅動
                        obj.forced_by_io = False
                        obj.source = 'top_view'
                    
                    # 根據 visibility 判斷 semantic_location
                    if obj.visibility == 'visible':
                        obj.semantic_location = 'on_table'
                        # 儲存最後一次視覺位置（供未來 freeze 使用）
                        if obj.position is not None:
                            obj.last_vision_position = obj.position.copy()
                    else:
                        obj.semantic_location = 'unknown'
            
            # === 2. 如果握持的物體不在 WorldState 中，強制加入 ===
            if holding_obj_id and holding_obj_id not in self.current_state.objects:
                # 從 tracked_objects 恢復
                if holding_obj_id in self.tracked_objects:
                    tracked_info = self.tracked_objects[holding_obj_id]
                    obj = ObjectState(
                        object_type=tracked_info['class_name'],
                        object_id=holding_obj_id,
                        location='in_hand',
                        position=tracked_info.get('position'),
                        confidence=0.9,  # 高信心度（來自 IO）
                        last_seen=time.time(),
                        visibility='visible',
                        semantic_location='in_gripper',
                        forced_by_io=True,
                        source='robot_io'
                    )
                    self.current_state.objects[holding_obj_id] = obj
                    logger.info(f"   Robot IO 強制加入物體: {holding_obj_id}")
        
        except Exception as e:
            logger.warning(f"   Robot IO Truth 更新失敗: {e}")
    
    def set_progress_tracker(self, tracker) -> None:
        """Register the authoritative progress tracker.

        When set, get_current_state() overrides current_state.step_class with
        tracker.confirmed_step so that manual overrides and smoothing are
        not erased by raw vision updates every frame.
        """
        self.progress_tracker = tracker

    def get_current_state(self) -> WorldState:
        """Return a snapshot of the current world state.

        Returns a shallow copy so that subsequent vision-loop writes to
        current_state do not mutate the snapshot held by callers (e.g. the
        orchestrator holds the snapshot across the LLM call ~500 ms while
        the vision loop keeps overwriting current_state.step_class).

        step_class is overridden with progress_tracker.confirmed_step when
        available so that manual overrides and smoothed values survive the
        per-frame raw vision updates.
        """
        import copy as _copy
        with self._state_lock:
            self.current_state.timestamp = time.time()
            # Apply authoritative step before snapshotting
            if self.progress_tracker is not None:
                try:
                    confirmed = getattr(self.progress_tracker, "confirmed_step", None)
                    if confirmed:
                        self.current_state.step_class = confirmed
                except Exception:
                    pass
            return _copy.copy(self.current_state)
    
    def finalize_frame(self):
        """Frame 結束：更新 previous_state（只在 main loop 最後呼叫一次）"""
        import copy
        self.previous_state = copy.deepcopy(self.current_state)
    
    def get_state_dict(self) -> Dict:
        """Get current state as dictionary"""
        return self.current_state.to_dict()
    
    def get_state_json_compatible(self) -> Dict:
        """Get JSON-compatible state dict"""
        return self.current_state.to_json_compatible()
    
    def reset(self):
        """Reset world state"""
        self.previous_state = self.current_state
        self.current_state = WorldState()
        self.safety_violation_count = 0
    
    def should_trigger_emergency_stop(self) -> bool:
        """Check if emergency stop should be triggered"""
        return (self.current_state.requires_emergency_stop() or 
                self.safety_violation_count >= self.safety_violation_threshold)
    
    def get_objects_by_type(self, object_type: str) -> List[ObjectState]:
        """Get all objects of specific type"""
        return self.current_state.get_objects_by_type(object_type)
    
    def get_objects_at_location(self, location: str) -> List[ObjectState]:
        """Get all objects at specific location"""
        return self.current_state.get_objects_at_location(location)
    
    def get_step_duration(self) -> float:
        """Get duration of current step in seconds"""
        if not self.step_history:
            return 0.0
        
        current_step = self.current_state.step_class
        current_time = time.time()
        
        # Find when current step started
        for step, timestamp in reversed(self.step_history):
            if step != current_step:
                # Found transition point
                return current_time - timestamp
        
        # Same step since beginning of history
        return current_time - self.step_history[0][1]
    
    def is_step_stable(self) -> bool:
        """Check if current step has been stable for threshold duration"""
        return self.get_step_duration() >= self.step_duration_threshold
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        return self.current_state.summary()


class AssemblyProgressTracker:
    """
    Event-driven Assembly Progress Tracker
    
    不再依賴 step classifier 的每幀輸出，改用事件觸發：
    1. Step stable (>2s) + Robot IO state change → 確認進度
    2. Latch 機制：一旦確認就不會因 classifier 誤判而回退
    3. 避免 FPS 波動影響（不用 time gate）
    """
    
    def __init__(self):
        self.progress = AssemblyProgress()
        
        # Event-driven State ===
        self.cylinders_inserted_set: set = set()  # Latch: {'cylinder_1', 'cylinder_2', ...}
        self.cylinders_screwed_set: set = set()   # Latch: {'cylinder_1', ...}
        self.bracket_placed: bool = False         # Latch
        self.pi_placed: bool = False              # Latch
        
        # 上一次狀態（用於檢測 state change）
        self.last_holding_object: Optional[str] = None
        self.last_step_class: Optional[str] = None
        self.step_stable_time: float = 0.0  # Step 穩定持續時間
        self.step_stable_start: Optional[float] = None
        
        # Step Smoothing: 連續幀一致性 (2026-01-14)
        self.step_consensus_buffer: list = []  # 滑動窗口，存最近 K 個 step
        self.consensus_k: int = 5  # 需要連續 5 幀一致才切換
        self.confirmed_step: Optional[str] = None  # 當前確認的 step（經過平滑）
        
        # Transition Constraints: 允許的步驟轉換 (2026-01-15 更新：單向前進FSM)
        # 
        # FSM設計原則（單向前進版本）：
        # 1. 單向約束：每步只能到「下一步」，禁止回退
        # 2. idle循環：idle → 第一步驟，最後步驟(步驟7) → idle
        # 3. 禁止中途idle：只有完成所有步驟（步驟7）才能到idle
        # 4. LLM手動控制：通過manual_override_step()可直接跳到任何步驟（繞過此檢查）
        # 5. 自循環：允許步驟重複（classifier可能連續輸出相同類別）
        #
        # ========================================
        # 統一步驟定義系統 (使用 step_definitions.py)
        # ========================================
        # 步驟編號：
        #   0: idle (待機)
        #   1: place_platform (放置平台)
        #   2: insert_cylinder_under_platform (插入圓柱)
        #   3: tighten_small_screw_under_platform (鎖小螺絲)
        #   4: place_bracket_on_table (放支架)
        #   5: align_pi_on_bracket (對齊PI)
        #   6: insert_upper_big_screw (插上方螺絲)
        #   7: align_bracket_to_platform (對齊支架到平台)
        #   8: insert_lower_big_screw (插下方螺絲)
        
        # 載入統一的 FSM 轉換規則
        from ws_racr.world_state.step_definitions import ALLOWED_TRANSITIONS
        self.allowed_transitions = ALLOWED_TRANSITIONS
        
        # Define step → progress mapping
        self.step_to_progress = {
            # Phase 1: Platform base assembly
            "pick_cylinder": ("phase_1", "picking"),
            "insert_cylinder": ("phase_1", "inserting"),
            "screw_cylinder": ("phase_1", "screwing"),
            
            # Phase 2: Bracket attachment
            "pick_bracket": ("phase_2", "picking"),
            "place_bracket": ("phase_2", "placing"),
            "screw_bracket": ("phase_2", "screwing"),
            
            # Phase 3: Pi component
            "pick_pi": ("phase_3", "picking"),
            "place_pi": ("phase_3", "placing"),
            "final_check": ("phase_3", "checking"),
        }
    
    def manual_override_step(self, step_index: int):
        """
        手動更新步驟狀態（用於語音指令 "我們目前在步驟X"）
        
        ⚠️ 重要：此方法繞過FSM轉換檢查，允許直接跳到任意步驟
        （用於LLM語音控制，支援手動同步實際工作進度）
        
        Args:
            step_index: TSM步驟索引 (0-8)
                0-7: 組裝步驟
                8: idle (待機)
        
        Raises:
            ValueError: 如果步驟索引超出範圍
        """
        if step_index < 0 or step_index > 8:
            raise ValueError(f"步驟索引必須在 0-8 之間，收到: {step_index}")
        
        # 使用統一步驟定義
        from ws_racr.world_state.step_definitions import get_step_class, step_index_to_user_display, get_step_name_zh
        
        step_class = get_step_class(step_index)
        user_display = step_index_to_user_display(step_index)
        step_name = get_step_name_zh(step_index)
        
        # 強制更新confirmed_step（繞過FSM轉換檢查和smoothing）
        old_step = self.confirmed_step
        self.confirmed_step = step_class
        self.last_step_class = step_class
        
        # 重置smoothing buffer（確保不會被平滑掉）
        self.step_consensus_buffer = [step_class] * self.consensus_k
        
        logger.info(f"[手動同步] 步驟已更新: {old_step} → {step_class} ({step_name})")
        
        return self.progress
    
    def update_from_step(self, step_class: str, objects: Dict[str, ObjectState], 
                        robot_controller=None) -> AssemblyProgress:
        """
        Event-driven update with smoothing
        
        Args:
            step_class: Current step classification (raw, 未平滑)
            objects: Current object states
            robot_controller: Robot controller (for IO state)
        
        Returns:
            Updated AssemblyProgress
        """
        import time
        
        # === 0. Step Smoothing: 連續幀一致性檢查 ===
        smoothed_step = self._apply_step_smoothing(step_class)
        
        # === 1. 更新 Step Stable Time (使用平滑後的 step) ===
        current_time = time.time()
        if smoothed_step != self.last_step_class:
            # Step 變化，重置計時
            self.last_step_class = smoothed_step
            self.step_stable_start = current_time
            self.step_stable_time = 0.0
        else:
            # Step 不變，續續計時
            if self.step_stable_start:
                self.step_stable_time = current_time - self.step_stable_start
        
        # === 2. 獲取 Robot IO State ===
        holding_object = None
        if robot_controller and hasattr(robot_controller, 'facts'):
            holding_object = robot_controller.facts.holding_object
        
        # 檢測 Robot IO state change
        io_state_changed = (holding_object != self.last_holding_object)
        
        # === 3. Event-driven Progress Update ===
        # 注意: 以下判斷依賴 Robot IO (holding_object)
        # 如果未啟用機器人，這些條件永遠不會觸發
        # TODO: 改用視覺檢測或移除
        
        step_stable = (self.step_stable_time > 1.0)  # Step 穩定 > 1s
        
        # 暫時停用所有依賴 Robot IO 的判斷（在沒有機器人時無法運作）
        # 保留架構以便未來整合真實機器人時重新啟用
        
        # === 4. 更新 Progress Object (僅基於 set 大小) ===
        self.progress.cylinders_inserted = len(self.cylinders_inserted_set)
        self.progress.cylinders_screwed = len(self.cylinders_screwed_set)
        self.progress.bracket_ready = self.bracket_placed
        self.progress.pi_ready = self.pi_placed
        
        # Determine phase from step (使用平滑後的 step)
        if smoothed_step and smoothed_step in self.step_to_progress:
            phase_name, _ = self.step_to_progress[smoothed_step]
            phase_num = int(phase_name.split('_')[1])
            self.progress.current_phase = phase_num
        
        # === 5. 更新 Last State ===
        self.last_holding_object = holding_object
        
        return self.progress
    
    def _apply_step_smoothing(self, raw_step: str) -> Optional[str]:
        """
        🆕 Step Smoothing: 連續 K 幀一致才切換 (2026-01-14)
        
        Args:
            raw_step: 原始 step classification 結果
        
        Returns:
            平滑後的 step（None 表示還未確認）
        """
        # 加入滑動窗口
        self.step_consensus_buffer.append(raw_step)
        if len(self.step_consensus_buffer) > self.consensus_k:
            self.step_consensus_buffer.pop(0)  # 移除最舊的
        
        # 檢查一致性：需要 K 個全部相同
        if len(self.step_consensus_buffer) >= self.consensus_k:
            if len(set(self.step_consensus_buffer)) == 1:  # 全部相同
                candidate_step = self.step_consensus_buffer[0]
                
                # Transition Constraints 檢查
                if self._is_valid_transition(self.confirmed_step, candidate_step):
                    # 合法轉換，確認切換
                    if self.confirmed_step != candidate_step:
                        logger.info(f"[Step Smoothing] {self.confirmed_step} → {candidate_step} (連續 {self.consensus_k} 幀一致)")
                    self.confirmed_step = candidate_step
                    return candidate_step
                else:
                    # 非法轉換，拒絕（可能是誤判）
                    logger.warning(f" [Step Smoothing] 非法轉換被阻止: {self.confirmed_step} → {candidate_step}")
                    # 清空 buffer，只保留當前 confirmed_step（重新開始累積）
                    self.step_consensus_buffer = [self.confirmed_step] if self.confirmed_step else []
                    return self.confirmed_step  # 保持舊 step
        
        # 未達到一致性門檻，保持舊 step
        return self.confirmed_step
    
    def _is_valid_transition(self, from_step: Optional[str], to_step: str) -> bool:
        """
        🆕 檢查步驟轉換是否合法 (2026-01-14)
        
        Args:
            from_step: 當前步驟（None 表示初始狀態）
            to_step: 目標步驟
        
        Returns:
            True 表示合法轉換
        """
        if from_step == to_step:
            return True  # 保持不變永遠合法
        
        allowed = self.allowed_transitions.get(from_step, [])
        return to_step in allowed
    
    def get_state(self) -> Dict:
        """獲取當前步驟狀態（VoiceQueryHandler接口兼容）
        
        Returns:
            包含 step_index 的字典：
            - step_index: TSM步驟索引 (0-8)
            - step_class: 步驟類別名稱
            - user_display: 用戶看到的步驟編號 (0-8，其中0表示待機)
        """
        from ws_racr.world_state.step_definitions import get_step_index, step_index_to_user_display
        
        # 獲取 TSM 步驟索引
        if self.confirmed_step:
            step_index = get_step_index(self.confirmed_step)
        else:
            step_index = 8  # 未初始化時視為 idle
        
        user_display = step_index_to_user_display(step_index)
        
        return {
            'step_index': step_index,
            'step_class': self.confirmed_step,
            'user_display': user_display,
            'progress': self.progress.__dict__ if self.progress else {}
        }
    
    def reset(self):
        """重置 progress"""
        self.progress = AssemblyProgress()
        self.cylinders_inserted_set.clear()
        self.cylinders_screwed_set.clear()
        self.bracket_placed = False
        self.pi_placed = False
        self.last_holding_object = None
        self.last_step_class = None
        self.step_stable_time = 0.0
        self.step_stable_start = None
        # 重置平滑狀態
        self.step_consensus_buffer.clear()
        self.confirmed_step = None


if __name__ == "__main__":
    # Test the WorldStateBuilder
    print("Testing WorldStateBuilder...")
    
    builder = WorldStateBuilder()
    tracker = AssemblyProgressTracker()
    
    # Simulate step recognition update
    builder.update_step_recognition("pick_cylinder", 0.95, phase=1)
    print(f"\n[Step Recognition] Updated: pick_cylinder (conf: 0.95, phase: 1)")
    
    # Simulate object detection
    detections = [
        {
            'class_name': 'platform',
            'confidence': 0.98,
            'bbox': [100, 100, 300, 300],
            'position': [0.5, 0.0, 0.0]
        },
        {
            'class_name': 'cylinder',
            'confidence': 0.92,
            'bbox': [150, 150, 200, 250],
            'position': [0.3, 0.2, 0.05]
        },
        {
            'class_name': 'cylinder',
            'confidence': 0.88,
            'bbox': [250, 150, 300, 250],
            'position': [0.3, -0.2, 0.05]
        }
    ]
    builder.update_object_detection(detections)
    print(f"[Object Detection] Updated: {len(detections)} objects")
    
    # Simulate gesture detection
    builder.update_gesture_detection(
        intent="pointing",
        safety="OK",
        confidence=0.87,
        raw_gesture="pointing",
        pointing_target="cylinder_1"
    )
    print(f"[Gesture Detection] Intent: pointing, Safety: OK")
    
    # Update assembly progress
    progress = tracker.update_from_step(
        builder.current_state.step_class,
        builder.current_state.objects
    )
    builder.update_assembly_progress(progress)
    builder.finalize_frame()
    
    # Print summary
    print("\n" + "="*60)
    print(builder.get_summary())
    print("="*60)
    
    # Print JSON-compatible state
    import json
    state_json = builder.get_state_json_compatible()
    print("\nJSON Output (first 500 chars):")
    print(json.dumps(state_json, indent=2)[:500] + "...")
