"""Perception side of the runtime: camera start-up, per-frame processing of the D435i (top-down: step recognition + object detection) and D455 (front: gesture) streams, world-state update, pointing detection and the on-screen visualisation.

Mixed into :class:`ws_racr.app.system.HRCSystem`; relies on its ``self`` state."""

from ws_racr.config import get_gesture_checkpoint_path, get_step_checkpoint_path, get_yolo_weights_path
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
import time
import torch
from typing import Optional, Dict, Any
import logging
from ws_racr.world_state import WorldStateBuilder
from ws_racr.execution.safety.workspace_monitor import WorkspaceClutterMonitor
logger = logging.getLogger(__name__)


class PerceptionMixin:
    """Perception side of the runtime: camera start-up, per-frame processing of the D435i (top-down: step recognition + object detection) and D455 (front: gesture) streams, world-state update, pointing detection and the on-screen visualisation."""

    def init_cameras(self):
        """初始化雙 RealSense 相機"""
        logger.info("\n[1/5] 初始化相機...")
        
        ctx = rs.context()
        devices = ctx.query_devices()
        
        d435i_serial = None
        d455_serial = None
        
        # 尋找相機
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            
            if "D435I" in name or "D435i" in name:
                d435i_serial = serial
            elif "D455" in name:
                d455_serial = serial
        
        if d435i_serial is None:
            raise RuntimeError("❌ 找不到 D435i 相機!")
        if d455_serial is None:
            raise RuntimeError("❌ 找不到 D455 相機!")
        
        # 配置 D435i (俯視) - 用於步驟識別和物件偵測
        self.d435i_pipeline = rs.pipeline()
        config_435i = rs.config()
        config_435i.enable_device(d435i_serial)
        config_435i.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config_435i.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        self.d435i_pipeline.start(config_435i)
        logger.info("D435i (俯視) 已啟動 (1280x720@30fps)")
        
        time.sleep(1.0)
        
        # 配置 D455 (正視) - 用於手勢識別
        self.d455_pipeline = rs.pipeline()
        config_455 = rs.config()
        config_455.enable_device(d455_serial)
        config_455.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config_455.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        self.d455_pipeline.start(config_455)
        logger.info("D455 (正視) 已啟動 (1280x720@30fps)")
        
        time.sleep(2.0)
        logger.info("雙相機就緒 (1280x720@30fps)")

    def init_perception_modules(self):
        """初始化感知模組"""
        logger.info("\n[2/5] 初始化感知模組...")
        
        # 1. Step Recognition
        try:
            import torch  # 先 import torch
            from ws_racr.perception.step_recognition.realtime import RealtimeStepRecognizer
            
            checkpoint_path = get_step_checkpoint_path()
            
            if checkpoint_path.exists():
                self.step_model = RealtimeStepRecognizer(
                    str(checkpoint_path),
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    clip_len=16,
                    confidence_threshold=0.5
                )
                logger.info("   Step Recognition (TSM+ResNet18)")
            else:
                logger.warning(f"   Step Recognition checkpoint not found")
        except Exception as e:
            logger.warning(f"   Step Recognition failed: {e}")
        
        # 2. Object Detection (YOLO)
        try:
            import torch
            from ultralytics import YOLO
            
            model_path = get_yolo_weights_path()
            if model_path.exists():
                self.object_detector = YOLO(str(model_path))
                
                # 強制使用 GPU
                if torch.cuda.is_available():
                    self.object_detector.to('cuda')
                    device = 'cuda:0'
                else:
                    device = 'cpu'
                
                logger.info(f"   Object Detection (YOLOv8-seg)")
                logger.info(f"      Device: {device}")
                logger.info(f"      CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"      GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning(f"   YOLO model not found")
        except Exception as e:
            logger.warning(f"   YOLO failed: {e}")
        
        # 3. Gesture Recognition (MediaPipe + MLP)
        try:
            import mediapipe as mp
            from ws_racr.perception.gesture.pose_dataset import HandFeatureExtractor
            from ws_racr.perception.gesture.pose_model import HandGestureModel
            from ws_racr.perception.gesture.state_machine import IntentSmoothingStateMachine
            
            gesture_ckpt = get_gesture_checkpoint_path()
            model_files = [gesture_ckpt] if gesture_ckpt.exists() else []
            
            if model_files:
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                feature_extractor = HandFeatureExtractor()
                self.gesture_model = HandGestureModel.load_from_checkpoint(
                    str(model_path),
                    feature_extractor,
                    device=device,
                    confidence_threshold=0.6
                )
                
                self.gesture_state_machine = IntentSmoothingStateMachine(
                    window_size=10,
                    stop_threshold=0.7,
                    stop_min_count=5,
                    receive_threshold=0.7,
                    receive_min_count=5,
                    pointing_threshold=0.7,
                    pointing_min_count=5
                )
                
                mp_module = mp.solutions.hands
                self.hands_detector = mp_module.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                logger.info("   Gesture Recognition (MediaPipe + MLP + FSM)")
            else:
                logger.warning("   Gesture model not found")
        except Exception as e:
            logger.warning(f"   Gesture Recognition failed: {e}")
        
        logger.info("感知模組就緒")

    def run_startup_clutter_check(self):
        """
        初始化最後一步的一次性雜物檢查。
        僅執行一次，不會進入每幀檢測。
        """
        logger.info("\n[Init-Final] Workspace clutter check...")
        try:
            if self.d435i_pipeline is None:
                logger.warning("跳過雜物檢查：D435i pipeline 未就緒")
                return

            frames = self.d435i_pipeline.wait_for_frames(timeout_ms=2500)
            color_frame = frames.get_color_frame()
            if not color_frame:
                logger.warning("跳過雜物檢查：無法取得 D435i 彩色影像")
                return
            image = np.asanyarray(color_frame.get_data())

            # 使用你指定的四點 polygon ROI
            startup_roi_polygon = [
                (0, 426),
                (1280, 369),
                (1280, 0),
                (0, 0),
            ]

            monitor = WorkspaceClutterMonitor(
                yolo_model=self.object_detector,   # 共用主系統 YOLO，避免重複載入
                cooldown_seconds=0.0,
                max_calls_per_task=1,
                roi_polygon=startup_roi_polygon,
                roi_polygon_is_ratio=False,
            )

            result = monitor.check(image, triggered_by="startup_init", force=True)
            if result is None:
                logger.warning("初始化雜物檢查未取得結果（API/回傳格式問題），將繼續啟動")
                return

            if result.has_clutter:
                clutter_list = ", ".join(result.detected_objects) if result.detected_objects else "foreign objects"
                logger.warning(f"偵測到非任務物品：{clutter_list}")
                logger.info("   請先移除後再開始操作。")
                self._speak_tts("Workspace has unrelated items. Please remove them before operation.")
            else:
                logger.info("初始化雜物檢查通過：工作區乾淨")

        except Exception as e:
            logger.warning(f"初始化雜物檢查失敗：{e}")

    def process_d435i_frame(self, frameset):
        """處理 D435i 俯視畫面 (步驟識別 + 物件偵測 + 深度)"""
        # 提取 color 和 depth frame
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return "idle", 0.0, 0, []
        
        # 轉換為 numpy array
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 獲取相機內參（用於 3D 座標轉換）
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        self.d435i_buffer.append(frame)
        
        step_class = "idle"
        step_conf = 0.0
        phase = 0
        detections = []
        
        # Step Recognition（buffer 每幀填入以維持時序連續性；[Fix-4] inference 每 tsm_skip_frames 幀執行一次）
        t_step_start = time.time()
        self.tsm_inference_counter += 1
        if self.step_model and len(self.d435i_buffer) == 16 and (self.tsm_inference_counter % self.tsm_skip_frames == 0):
            try:
                frames = list(self.d435i_buffer)
                result = self.step_model.predict(frames)
                step_class = result['class_name']
                step_conf = result['confidence']
                phase = self.step_to_phase.get(step_class, 0)
            except Exception as e:
                logger.info(f"Step recognition error: {e}")
        t_step = (time.time() - t_step_start) * 1000
        
        # Object Detection (with 3D position + confidence filtering + 降採樣)
        self.detect_frame_counter += 1
        should_run_detect = (self.detect_frame_counter % self.detect_skip_frames == 0)
        
        if should_run_detect and self.object_detector:
            t_detect_start = time.time()
            try:
                
                # === 設定檢測閾值（減少誤檢）===
                detection_confidence_thresholds = {
                    'cylinder': 0.6,      # 圓柱：容易誤檢，提高閾值
                    'screw_small': 0.65,  # 小螺絲：容易誤檢
                    'screw_big': 0.6,     # 大螺絲
                    'nut': 0.6,           # 螺母
                    'platform': 0.5,      # 平台：大物體，閾值較低
                    'bracket': 0.55,      # 支架
                    'pi': 0.55,           # Pi
                    'hex_wrench': 0.6,    # 扳手
                    'default': 0.5        # 默認閾值
                }
                
                results = self.object_detector(frame, verbose=False)
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            bbox = box.xyxy[0].cpu().numpy().tolist()
                            class_name = r.names[cls_id]
                            
                            # === Confidence 過濾 ===
                            threshold = detection_confidence_thresholds.get(
                                class_name, 
                                detection_confidence_thresholds['default']
                            )
                            if conf < threshold:
                                continue  # 跳過低置信度檢測
                            
                            # === 計算 3D position ===
                            position_3d = None
                            try:
                                # 取 bbox 中心點
                                center_x = int((bbox[0] + bbox[2]) / 2)
                                center_y = int((bbox[1] + bbox[3]) / 2)
                                
                                # 從深度圖取得深度值（mm）
                                depth_value = depth_frame.get_distance(center_x, center_y)
                                
                                if depth_value > 0:  # 有效深度
                                    # 使用相機內參轉換為 3D 座標（公尺）
                                    point_3d = rs.rs2_deproject_pixel_to_point(
                                        depth_intrinsics, [center_x, center_y], depth_value
                                    )
                                    # 轉換為 mm（WorldStateBuilder 統一使用 mm）
                                    position_3d = [
                                        point_3d[0] * 1000,  # X (mm)
                                        point_3d[1] * 1000,  # Y (mm)
                                        point_3d[2] * 1000   # Z (mm)
                                    ]
                            except Exception as e:
                                # 深度計算失敗，position 保持 None
                                pass
                            
                            detections.append({
                                'class_name': class_name,
                                'confidence': conf,
                                'bbox': bbox,
                                'position': position_3d  # 3D 座標 (mm)
                            })
                
                t_detect = (time.time() - t_detect_start) * 1000
                pass  # 效能資訊已移至 debug 模式
                    
            except Exception as e:
                logger.info(f"Object detection error: {e}")
                t_detect = (time.time() - t_detect_start) * 1000
            
            # 儲存此次 Detection 結果（供跳過幀使用）
            self.last_detections = detections
        else:
            # 跳過幀不執行 Detection，使用上一次結果（保持物體狀態）
            detections = self.last_detections
            t_detect = 0.0
        
        return step_class, step_conf, phase, detections

    def process_d455_frame(self, frame):
        """處理 D455 正視畫面 (手勢識別)"""
        # 降採样：每 N 幀才跑一次 Gesture Detection
        self.d455_frame_counter += 1
        should_run_gesture = (self.d455_frame_counter % self.gesture_skip_frames == 0)
        
        intent = "idle"
        safety = "OK"
        gesture_conf = 0.0
        gesture_data = {}
        
        t_gesture_start = time.time()  # 移到这里，在判断之后
        
        if should_run_gesture and hasattr(self, 'gesture_model') and self.gesture_model:
            try:
                # 直接使用記憶體中的 BGR 影像，避免寫檔 IO
                prediction = self.gesture_model.predict_from_bgr(frame)
                
                if prediction['confidence'] > 0.0:
                    if hasattr(self, 'gesture_state_machine') and self.gesture_state_machine:
                        state = self.gesture_state_machine.update(prediction)
                        intent = state['intent_state']
                        safety = state['safe_state']
                        gesture_conf = prediction['confidence']
                        
                        gesture_data = {
                            'class': intent,
                            'confidence': gesture_conf,
                            'raw_class': prediction['class_name'],
                            'safety': safety
                        }
                    else:
                        intent = prediction['class_name']
                        gesture_conf = prediction['confidence']
                        safety = "EMERGENCY_STOP" if intent == "stop" else "OK"
                        
                        gesture_data = {
                            'class': intent,
                            'confidence': gesture_conf,
                            'safety': safety
                        }
            except Exception as e:
                logger.info(f"Gesture error: {e}")
        
        # 缓存结果（下次跳过时使用）
        if should_run_gesture:
            self.last_gesture_result = {
                'intent': intent,
                'safety': safety,
                'confidence': gesture_conf,
                'data': gesture_data
            }
        else:
            # 使用缓存的结果
            intent = self.last_gesture_result['intent']
            safety = self.last_gesture_result['safety']
            gesture_conf = self.last_gesture_result['confidence']
            gesture_data = self.last_gesture_result['data']
        
        return intent, safety, gesture_conf, gesture_data

    def update_world_state(self, step_class, step_conf, phase, detections,
                          intent, safety, gesture_conf, gesture_data):
        """更新 WorldState（不記錄資料庫，讓機器人動作優先）"""
        
        # Update step recognition
        self.world_state_builder.update_step_recognition(step_class, step_conf, phase)
        
        # Update object detection
        self.world_state_builder.update_object_detection(detections)
        
        # Update gesture detection
        # pointing_target: 若 TTL 內有快取值則一併注入，讓 WorldState 持有有效指代目標
        _cached_pt = self._pointing_cache if time.time() < self._pointing_expiry else None
        self.world_state_builder.update_gesture_detection(
            intent=intent,
            safety=safety,
            confidence=gesture_conf,
            raw_gesture=gesture_data.get('class', 'unknown'),
            pointing_target=_cached_pt,
        )
        
        # Update assembly progress
        progress = self.progress_tracker.update_from_step(
            step_class,
            self.world_state_builder.current_state.objects,
            robot_controller=self.robot_controller  # 傳遞 robot_controller
        )
        self.world_state_builder.update_assembly_progress(progress)
        
        # ========== Apply Robot IO Truth ==========
        # 從 Robot Controller 讀取 holding_object，強制更新 semantic_location
        self.world_state_builder.apply_robot_io_truth()
        # ======================================================
        
        # ========== Update Memory System (RoboMemory) - 降採樣 ==========
        # 每 10 幀更新一次，減少計算負載（Memory/KG 是"越跑越卡"元凶）
        self.memory_frame_counter += 1
        should_update_memory = (self.memory_frame_counter % self.memory_skip_frames == 0)
        
        if should_update_memory:
            try:
                # 確保轉換為正確的字典格式
                world_state_obj = self.world_state_builder.current_state
                
                # 構建符合 memory_system 期望的字典格式
                world_state_dict = {
                    'human_state': {
                        'gesture': world_state_obj.human.intent,
                        'confidence': world_state_obj.human.confidence
                    },
                    'assembly_progress': {
                        'current_step': world_state_obj.step_class,
                        'phase': world_state_obj.phase
                    },
                    'safety_state': {
                        'status': world_state_obj.human.safety
                    },
                    'objects': {
                        obj.object_id: {
                            'object_type': obj.object_type,
                            'confidence': obj.confidence,
                            'position': obj.position
                        }
                        for obj in world_state_obj.objects.values()
                    },
                    'robot_status': {
                        'current_action': self.last_robot_action,
                        'busy': self.robot_busy
                    },
                    'timestamp': world_state_obj.timestamp
                }
                
                step_summary, memory_query = self.memory_system.update(world_state_dict)
                
                # 每 300 幀顯示一次記憶系統統計
                if self.frame_count % 300 == 0:
                    self.memory_system.print_stats()
            except Exception as e:
                logger.warning(f"Memory system update failed: {e}")
        # ==========================================================================
        
        # 暫存資料庫記錄的數據（延遲到主循環後執行）
        self._pending_db_log = {
            'detections': detections,
            'gesture_data': gesture_data
        }

    def _resolve_pointed_object_id(self, class_name: str, pointed_pos) -> str:
        """
        Resolve a detected class_name + 3D position to a specific tracked object_id.
        e.g. "cylinder" + [0.528, 0.220, 0.104] → "cylinder_2"

        Looks through world_state.objects for objects whose object_type matches
        class_name, then returns the ID of the nearest one (if within 15 cm).
        Falls back to class_name if world_state has no candidates or no position.
        """
        if pointed_pos is None:
            return class_name
        try:
            import numpy as np
            pt = np.array(pointed_pos[:3], dtype=float)
            ws = self.world_state_builder.current_state
            best_id: Optional[str] = None
            best_dist = float('inf')
            for oid, obj in ws.objects.items():
                # Match by class prefix (e.g. "cylinder_1" starts with "cylinder")
                if not oid.startswith(class_name):
                    continue
                if obj.position is None:
                    continue
                d = float(np.linalg.norm(np.array(obj.position[:3], dtype=float) - pt))
                if d < best_dist:
                    best_dist = d
                    best_id = oid
            if best_id and best_dist < 0.15:  # 15 cm threshold
                return best_id
        except Exception:
            pass
        return class_name  # fallback: class name without ID

    def _run_pointing_detection(self):
        """
        背景執行：用雙相機射線識別使用者正在指向的物體，結果寫入 _pointing_cache。
        每次 pointing 手勢的 rising edge 只執行一次（_pointing_detect_lock 保護）。
        """
        try:
            if not self.robot_controller:
                return
            # 初始化視覺系統（若尚未初始化）
            if not getattr(self.robot_controller, "vision", None):
                self.robot_controller.initialize_vision()

            result = self.robot_controller._detect_pointed_object_dual_camera()

            if result:
                obj_class = result.get("class_name", "unknown")
                pointed_pos = result.get("position")  # [x, y, z] metres

                # Resolve class_name → specific tracked object_id using 3D position.
                # e.g. "cylinder" → "cylinder_2" so the LLM knows WHICH cylinder.
                resolved_id = self._resolve_pointed_object_id(obj_class, pointed_pos)

                self._pointing_cache = resolved_id
                self._pointing_expiry = time.time() + 10.0
                # Eagerly push to world_state so on_result() can see it immediately,
                # without waiting for the next frame's update_world_state() call.
                try:
                    ws_human = self.world_state_builder.current_state.human
                    ws_human.pointing_target = resolved_id
                    ws_human.pointing_position = pointed_pos
                    # Store the top-camera pixel center from pointing detection —
                    # used by _find_object_nearest_to_hint to match the exact
                    # YOLO bbox in the same image space (avoids coord-frame issues).
                    px = result.get("pixel_center")
                    if px:
                        ws_human.pointing_pixel = [int(px[0]), int(px[1])]
                except Exception:
                    pass
                logger.info(f"   [Pointing] 目標：{resolved_id}（TTL 10s）")
                self._speak_tts(f"你在指{resolved_id}，請說要我怎麼幫你", priority=self._TTS_GESTURE)
            else:
                logger.warning("   [Pointing] 未偵測到指向目標")
                self._speak_tts("我看不清楚你在指哪裡", priority=self._TTS_GESTURE)
        except Exception as e:
            logger.error(f"   [Pointing] 偵測失敗：{e}")
        finally:
            # 釋放 lock，讓下次 rising edge 可再觸發
            try:
                self._pointing_detect_lock.release()
            except RuntimeError:
                pass  # lock 可能未被 acquire

    def _find_nearest_object(self, objects: Dict) -> Optional[Any]:
        """找到最近的物件（用於指向手勢）"""
        if not objects:
            return None
        
        # 簡單實現：返回第一個高信心度的物件
        for obj in objects.values():
            if obj.confidence > 0.7:
                return obj
        
        return None

    def create_visualization(self, frame_d435i, frame_d455):
        """創建視覺化畫面"""
        h, w = frame_d435i.shape[:2]
        
        scale = 0.6
        d435i_resized = cv2.resize(frame_d435i, (int(w * scale), int(h * scale)))
        d455_resized = cv2.resize(frame_d455, (int(w * scale), int(h * scale)))
        
        canvas_h = int(h * scale) + 300
        canvas_w = int(w * scale) * 2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # 放置相機畫面
        canvas[0:int(h * scale), 0:int(w * scale)] = d435i_resized
        canvas[0:int(h * scale), int(w * scale):] = d455_resized
        
        # 標籤
        cv2.putText(canvas, "D435i (Top-View)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, "D455 (Front-View)", (int(w * scale) + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 資訊面板
        info_y = int(h * scale) + 30
        state = self.world_state_builder.current_state
        tracker = self.progress_tracker
        
        # 左側 (步驟 + 物件 + 進度)
        # 1. 感測步驟 (即時 classifier 輸出，可能不穩定)
        detected_color = (180, 180, 180)  # 灰色 - 未確認
        cv2.putText(canvas, f"Detected: {state.step_class} ({state.step_confidence:.1%})",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detected_color, 2)
        
        info_y += 30
        # 2. 確認步驟 (FSM 穩定確認的步驟)
        confirmed_step = tracker.last_step_class if tracker.last_step_class else "idle"
        stable_time = tracker.step_stable_time
        confirmed_color = (0, 255, 0) if stable_time > 2.0 else (0, 165, 255)  # 綠色=穩定, 橘色=穩定中
        stability_indicator = "V" if stable_time > 2.0 else f"{stable_time:.1f}s"
        cv2.putText(canvas, f"Confirmed: {confirmed_step} {stability_indicator}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, confirmed_color, 2)
        
        info_y += 35
        cv2.putText(canvas, f"Phase {state.phase}: {state.get_phase_name()}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 35
        cv2.putText(canvas, f"Objects: {len(state.objects)} | Cylinders: {state.assembly_progress.cylinders_inserted}/4",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 右側 (手勢 + 安全 + 機器人)
        info_y = int(h * scale) + 30
        cv2.putText(canvas, f"Intent: {state.human.intent} ({state.human.confidence:.1%})",
                   (int(w * scale) + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 35
        safety_color = (0, 255, 0) if state.human.safety == "OK" else (0, 0, 255)
        cv2.putText(canvas, f"Safety: {state.human.safety}",
                   (int(w * scale) + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, safety_color, 2)
        
        # 安全指示燈
        cv2.circle(canvas, (canvas_w - 50, info_y - 10), 20, safety_color, -1)
        
        info_y += 35
        robot_status = "BUSY" if self.robot_busy else "IDLE"
        robot_color = (0, 165, 255) if self.robot_busy else (128, 128, 128)
        cv2.putText(canvas, f"Robot: {robot_status}",
                   (int(w * scale) + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, robot_color, 2)
        
        # 系統狀態
        info_y += 50
        cv2.putText(canvas, f"FPS: {self.fps:.1f} | Frame: {self.frame_count}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        info_y += 30
        cv2.putText(canvas, f"Episode: {self.current_episode_id[:25] if self.current_episode_id else 'N/A'}...",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 控制說明
        info_y += 40
        voice_status = "🔴REC" if self.voice_recording else "V-Voice"
        cv2.putText(canvas, f"Controls: Q-Quit | S-Summary | {voice_status} | R-Robot Status",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return canvas
