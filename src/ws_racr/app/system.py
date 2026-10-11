"""HRCSystem: the runtime that wires perception, world state, memory, orchestrator and robot together
(thesis Ch. 3, asynchronous pipeline). Perception runs continuously; commands are snapshotted and
queued for the orchestrator worker thread."""

import cv2
import numpy as np
from collections import deque
import time
from datetime import datetime
import threading
import queue
from typing import Optional, Dict
import logging
from ws_racr.world_state import WorldStateBuilder, AssemblyProgressTracker
logger = logging.getLogger(__name__)
from ws_racr.app.perception_runtime import PerceptionMixin
from ws_racr.app.robot_runtime import RobotRuntimeMixin
from ws_racr.app.triggers import TriggerMixin
from ws_racr.app.voice_runtime import VoiceMixin


class HRCSystem(PerceptionMixin, RobotRuntimeMixin, TriggerMixin, VoiceMixin):
    """
    人機協作系統 - 整合感知、推理、控制
    """
    
    def __init__(self, db_url=None, enable_robot=False):
        """
        初始化系統
        
        Args:
            db_url: 資料庫連線 URL
            enable_robot: 是否啟用機器人控制 (需要硬體連接)
        """
        logger.info("\n" + "="*70)
        logger.info("人機協作組裝系統 (HRC System)")
        logger.info("="*70)
        
        # 系統配置
        self.db_url = db_url
        self.enable_robot = enable_robot
        
        # Core components
        self.world_state_builder = WorldStateBuilder()
        self.progress_tracker = AssemblyProgressTracker()
        # Register progress_tracker as authoritative step source so that
        # smoothed / manually-overridden steps are not erased by raw vision.
        self.world_state_builder.set_progress_tracker(self.progress_tracker)
        
        # Memory System (延遲初始化，需要 logger 作為 backend)
        self.memory_system = None
        
        self.logger = None  # 延遲初始化
        self.robot_controller = None  # 延遲初始化
        
        # Camera pipelines
        self.d435i_pipeline = None  # 俯視 (步驟識別 + 物件偵測)
        self.d455_pipeline = None   # 正視 (手勢識別)
        
        # Perception modules
        self.step_model = None
        self.object_detector = None
        self.gesture_model = None
        self.gesture_state_machine = None
        self.hands_detector = None
        
        # Frame buffers
        self.d435i_buffer = deque(maxlen=16)
        self.d455_frame_counter = 0  # Gesture 降採样计数器
        self.gesture_skip_frames = 5  # 每 5 幀跑一次 Gesture（降低負載）
        # Step Recognition 不能降採樣（需要連續 16 幀維持時序）
        self.detect_frame_counter = 0  # Detection 降採樣計數器
        self.detect_skip_frames = 6    # 每 6 幀跑一次 Detection（降低負載）
        self.last_detections = []      # 快取上一次 Detection 結果（跳過幀使用）
        self.tsm_inference_counter = 0  # [Fix-4] TSM 推論降採樣計數器（buffer 付每幀填入）
        self.tsm_skip_frames = 2        # [Fix-4] 每 2 幀推論一次（15 Hz），釋放 GPU 給 YOLO
        self.memory_frame_counter = 0  # Memory System 降採樣計數器
        self.memory_skip_frames = 10   # 每 10 幀更新一次 Memory（降低負載）
        self.last_gesture_result = {  # 缓存上一次 Gesture 结果
            'intent': 'idle',
            'safety': 'OK',
            'confidence': 0.0,
            'data': {}
        }  # For step recognition
        
        # Tracking
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.current_episode_id = None
        self.current_session_id = None
        
        # Robot state
        self.robot_busy = False
        self._task_submit_lock = threading.Lock()  # 保護 robot_busy check-and-set
        self.last_robot_action = None
        self.robot_action_queue = deque(maxlen=10)
        self.pending_tasks = {}  # task_id -> task_info
        
        # Gesture state machine to prevent duplicate triggers
        self.last_triggered_gesture = None  # 上次觸發的手勢
        # 需要重置的手勢（機器人執行完後必須回到 idle 才能再次觸發動作）
        # pointing 已改為上下文模式，不再觸發機器人動作，故不在此列表中。
        self.gestures_need_reset = ["receive"]

        # Pointing gesture — context-only mode
        # pointing 手勢只做目標識別，不自動執行動作。
        # 識別結果以 pointing_target 注入 WorldState，供語音指令的 LLM-A 指代消解使用。
        self._pointing_cache: Optional[str] = None       # 最後一次識別到的指向目標 class_name
        self._pointing_expiry: float = 0.0               # TTL expiry（10 秒後清空）
        self._pointing_was_active: bool = False          # rising-edge 偵測：上幀是否有 pointing
        self._pointing_detect_lock = threading.Lock()    # 防止同一 pointing session 重複觸發偵測
        
        # Voice recognition
        self.voice_recognizer = None  # 延遲初始化
        self.voice_recording = False
        self.query_handler = None  # 語音查詢處理器 
        self.tts_engine = None  # 
        self.orchestrator = None  # HRC Orchestrator (ReAct agent)
        self._manual_command_text = None  # 語音/文字指令暫存
        
        # Failure Recovery System 
        self.failure_detector = None  # 延遲初始化
        self.recovery_policy = None   # 延遲初始化
        self.in_recovery_mode = False  # 是否正在恢復中
        self.pending_recovery = None   # 待執行的恢復動作
        
        # 互斥等待狀態機：三種「等待人類語音回應」的情境不可能同時發生，
        # 以單一 _wait_state + _wait_context 取代原本 3 個獨立 bool + 3 個 context dict，
        # 避免多個 flag 同時為 True 造成未定義行為。
        # 合法值: 'idle' | 'occlusion_choice' | 'occlusion_clear' | 'retry_confirm'
        self._wait_state: str = 'idle'
        self._wait_context: Optional[Dict] = None  # 對應的上下文（idle 時為 None）
        
        # 步驟變更監控（用於 TTS 提示）
        self.last_confirmed_step = 0
        self.last_step_tts_time = 0
        
        # 多層觸發控制（防止重複執行）
        self.trigger_control = {
            'gesture': {  # 手勢觸發：需要重置機制
                'last_trigger': None,
                'cooldown': 0,  # 手勢由 FSM 控制，無需額外 cooldown
                'needs_reset': True  # 必須回到 idle 才能再次觸發
            },
            'voice': {  # 語音觸發：允許連續觸發
                'last_trigger': None,
                'cooldown': 0,  # 無 cooldown，每次語音都應執行
                'needs_reset': False  # 不需要重置
            },
            'auto': {  # 自動推理觸發：需要冷卻時間
                'last_trigger': None,
                'last_reason': None,
                'cooldown': 5.0,  # 5秒冷卻時間（防止同一任務重複觸發）
                'needs_reset': False
            }
        }
        
        # Task execution thread
        self.task_queue = queue.Queue()
        self.task_thread = None
        self.task_thread_running = False
        
        # Phase mapping (步驟 → 階段)
        self.step_to_phase = {
            'idle': 0,
            'place_platform': 1,
            'insert_cylinder_under_platform': 1,
            'tighten_small_screw_under_platform': 1,
            'place_bracket_on_table': 2,
            'align_pi_on_bracket': 2,
            'insert_upper_big_screw': 2,
            'align_bracket_to_platform': 3,
            'insert_lower_big_screw': 3,
        }
        
        # 初始化語音識別
        self._init_voice_recognition()
        
        logger.info("系統初始化完成")
    
    
    
    
    
    
    
    
    
    def start_episode(self):
        """開始新 Episode"""
        logger.info("\n[5/5] 開始 Episode...")
        
        self.current_episode_id = self.logger.start_episode(
            session_id=f"hrc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            metadata={
                'cameras': ['D435i', 'D455'],
                'perception_modules': ['step_recognition', 'object_detection', 'gesture_recognition'],
                'robot_enabled': self.enable_robot,
                'operator': 'operator_01',
                'environment': 'lab'
            }
        )
        
        logger.info(f"Episode 已建立: {self.current_episode_id[:35]}...")
        
        # 相機喚醒
        logger.info("\n喚醒相機...")
        for i in range(5):
            try:
                self.d435i_pipeline.wait_for_frames()
                self.d455_pipeline.wait_for_frames()
            except:
                pass
        logger.info("系統完全就緒!\n")

    
    
    
    
    


    
    
    
    
    
    def run(self):
        """執行主循環"""
        logger.info("\n" + "="*70)
        logger.info("▶ 系統運行中...")
        logger.info("="*70 + "\n")
        
        try:
            while True:
                # 完整幀計時
                t_frame_start = time.time()
                
                self.frame_count += 1
                
                # 取得畫面
                t_camera_start = time.time()
                frames_d435i = self.d435i_pipeline.wait_for_frames()
                frames_d455 = self.d455_pipeline.wait_for_frames()
                
                color_frame_d435i = frames_d435i.get_color_frame()
                color_frame_d455 = frames_d455.get_color_frame()
                
                if not color_frame_d435i or not color_frame_d455:
                    continue
                
                frame_d435i = np.asanyarray(color_frame_d435i.get_data())
                frame_d455 = np.asanyarray(color_frame_d455.get_data())
                t_camera = (time.time() - t_camera_start) * 1000
                
                # 處理感知（傳入 frameset 以獲取深度數據）
                step_class, step_conf, phase, detections = self.process_d435i_frame(frames_d435i)
                intent, safety, gesture_conf, gesture_data = self.process_d455_frame(frame_d455)
                
                # 更新 WorldState（計時）
                t_ws_start = time.time()
                self.update_world_state(
                    step_class, step_conf, phase, detections,
                    intent, safety, gesture_conf, gesture_data
                )
                t_ws = (time.time() - t_ws_start) * 1000
                
                # 檢測步驟變更並播報
                # 使用 assembly_progress.current_phase 作為步驟指標
                current_confirmed_step = self.world_state_builder.current_state.assembly_progress.current_phase
                if current_confirmed_step != self.last_confirmed_step:
                    # 步驟變更，播報新步驟
                    current_time = time.time()
                    # 防止過於頻繁播報（至少間隔 3 秒）
                    if current_time - self.last_step_tts_time > 3.0:
                        self._announce_step_change(self.last_confirmed_step, current_confirmed_step)
                        self.last_confirmed_step = current_confirmed_step
                        self.last_step_tts_time = current_time
                
                # === 穩定化 WorldState（每幀都執行）===
                t_stab_start = time.time()
                if self.robot_controller and hasattr(self.robot_controller, 'stabilizer'):
                    current_state = self.world_state_builder.get_current_state()
                    stabilized_state = self.robot_controller.stabilizer.stabilize(current_state)
                    # 更新回 builder（保持狀態同步）
                    self.world_state_builder.current_state = stabilized_state
                    # 更新 Facts Store
                    self.robot_controller.facts.update_from_world_state(stabilized_state)
                t_stab = (time.time() - t_stab_start) * 1000
                
                # 完整幀時間
                t_frame_total = (time.time() - t_frame_start) * 1000
                actual_fps = 1000.0 / t_frame_total if t_frame_total > 0 else 0
                
                # 輕量狀態心跳（每 90 幀 / ~3秒）
                if self.frame_count % 90 == 0:
                    world_state = self.world_state_builder.get_current_state()
                    intent = world_state.human.intent
                    step   = world_state.step_class
                    busy   = "busy" if self.robot_busy else "idle"
                    holding = ""
                    if self.robot_controller and hasattr(self.robot_controller, 'facts'):
                        h = self.robot_controller.facts.holding_object
                        if h:
                            holding = f" | holding={h}"
                    logger.info(f"[{self.frame_count}] {actual_fps:.1f}fps  intent={intent}  step={step}  robot={busy}{holding}")
                    
                # LLM 推理 (每 30 幀執行一次) - 機器人動作優先
                if self.frame_count % 30 == 0:
                    self.llm_reasoning()
                
                # ========== 延遲資料庫記錄（機器人動作後執行） ==========
                # 只記錄不影響機器人動作的感知數據
                if self.current_episode_id and hasattr(self, '_pending_db_log'):
                    try:
                        self.logger.log_perception(
                            world_state=self.world_state_builder.current_state,
                            frame_number=self.frame_count,
                            detected_objects=self._pending_db_log['detections'],
                            gesture_data=self._pending_db_log['gesture_data']
                        )
                        
                        # Update object tracking
                        for obj in self.world_state_builder.current_state.objects.values():
                            self.logger.update_object_tracking(
                                object_id=obj.object_id,
                                object_type=obj.object_type,
                                location=obj.location,
                                position=obj.position
                            )
                        
                        # 清除已處理的記錄
                        del self._pending_db_log
                    except Exception as e:
                        logger.warning(f"Database logging failed: {e}")
                # ==========================================================
                
                # 視覺化
                vis = self.create_visualization(frame_d435i, frame_d455)
                cv2.imshow('HRC System', vis)
                
                # FPS
                current_time = time.time()
                self.fps = 1.0 / (current_time - self.last_time) if current_time > self.last_time else 0
                self.last_time = current_time
                
                # 鍵盤控制
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    logger.info("\n" + self.world_state_builder.get_summary())
                elif key == ord('v') or key == ord('V'):
                    self.manual_voice_command()
                elif key == ord('r') or key == ord('R'):
                    self.show_robot_status()
        
        except KeyboardInterrupt:
            logger.warning("\n使用者中斷")
        
        finally:
            self.shutdown()
    


    

    
    
    
    
    
    
    
    
    
    
    def show_robot_status(self):
        """顯示機器人狀態"""
        logger.info("\n" + "="*60)
        logger.info("機器人狀態")
        logger.info("="*60)
        logger.info(f"狀態: {'忙碌' if self.robot_busy else '閒置'}")
        logger.info(f"最後動作: {self.last_robot_action}")
        logger.info(f"動作佇列: {len(self.robot_action_queue)} 個")
        logger.info(f"待處理任務: {self.task_queue.qsize()} 個")
        logger.info(f"已完成任務: {len([t for t in self.pending_tasks.values() if 'completed_at' in t])} 個")
        
        if self.robot_action_queue:
            logger.info("\n最近動作:")
            for action in list(self.robot_action_queue)[-5:]:
                status = "✅" if action.get('success') else ("⏳" if 'success' not in action else "❌")
                logger.info(f"  {status} {action['command']} ({action['task_id'][:25]}...)")
        
        if self.pending_tasks:
            logger.info("\n待處理任務詳情:")
            for task_id, task in list(self.pending_tasks.items())[-3:]:
                status = "完成" if 'completed_at' in task else "執行中"
                logger.info(f"  - [{status}] {task['command']}")
                logger.info(f"    觸發: {task['reason']}, 優先級: {task.get('priority', 0)}")
        
        logger.info("="*60)
    
    def shutdown(self):
        """關閉系統"""
        logger.info("\n關閉系統...")
        
        # 停止任務執行線程
        if self.task_thread_running:
            logger.info("等待任務執行線程結束...")
            self.task_thread_running = False
            if self.task_thread and self.task_thread.is_alive():
                self.task_thread.join(timeout=5.0)
            logger.info("任務執行線程已停止")
        
        # 結束 episode
        if self.current_episode_id:
            self.logger.end_episode(
                success=True,
                assembly_phase=self.world_state_builder.current_state.phase
            )
            
            # 同步 Memory System 到持久化後端
            if self.memory_system:
                try:
                    self.memory_system.end_episode(self.current_episode_id)
                except Exception as e:
                    logger.warning(f"Memory system sync failed: {e}")
            
            summary = self.logger.get_episode_summary(self.current_episode_id)
            logger.info("\n" + "="*70)
            logger.info("Episode 摘要:")
            logger.info(f"  Episode ID: {summary['episode_id']}")
            logger.info(f"  狀態: {summary['status']}")
            logger.info(f"  階段: {summary['phase']}")
            logger.info(f"  感知記錄: {summary['perception_logs']} 筆")
            logger.info(f"  LLM 互動: {summary['llm_interactions']} 次")
            logger.info(f"  物件追蹤: {summary['objects_tracked']} 個")
            logger.info(f"  機器人任務: {len(self.pending_tasks)} 個")
            logger.info("="*70)
        
        # 停止相機
        if self.d435i_pipeline:
            self.d435i_pipeline.stop()
        if self.d455_pipeline:
            self.d455_pipeline.stop()
        
        cv2.destroyAllWindows()
        
        logger.info("\n系統已關閉")
