"""Robot / database side of the runtime: DB and robot-controller initialisation, the task-execution worker thread, failure checking and occlusion removal.

Mixed into :class:`ws_racr.app.system.HRCSystem`; relies on its ``self`` state."""

from ws_racr.execution.controller.robot_controller import RobotController
from ws_racr.config import get_sop_yaml_path
import sys
import pyrealsense2 as rs
import time
import threading
import queue
from typing import Optional, Dict
import logging
from ws_racr.world_state import WorldStateBuilder
from ws_racr.memory import HRCMemorySystem
from ws_racr.orchestrator import HRCOrchestrator
from ws_racr.execution.safety.failure_detector import (
    FailureDetector,
    FailureDetectionMode,
)
from ws_racr.execution.safety.recovery_policy import RecoveryPolicy
from ws_racr.hri.voice_query_handler import VoiceQueryHandler
logger = logging.getLogger(__name__)


class RobotRuntimeMixin:
    """Robot / database side of the runtime: DB and robot-controller initialisation, the task-execution worker thread, failure checking and occlusion removal."""

    def init_database(self):
        """初始化資料庫連線"""
        logger.info("\n[3/5] 初始化資料庫...")
        
        # 修正 import 路徑
        from ws_racr.database.perception_logger import PerceptionLogger
        self.logger = PerceptionLogger(self.db_url)
        
        logger.info("資料庫連線完成")
        
        # 初始化 Memory System（使用 logger 作為 persistence backend）
        logger.info("\n初始化記憶系統...")
        self.memory_system = HRCMemorySystem(
            temporal_buffer_size=50,
            persistence_backend=self.logger,
            embedding_model=self.logger.embedding_model
        )
        logger.info("資料庫 & 記憶系統就緒")
        
        # === 連接 Memory System 到 WorldStateBuilder ===
        self.world_state_builder.set_memory_system(self.memory_system)

    def init_robot_controller(self):
        """初始化機器人控制器"""
        logger.info("\n[4/5] 初始化機器人控制器...")
        
        # 初始化語音查詢處理器 (不依賴 robot_controller)
        self._init_voice_query_handler()
        
        # 初始化 TTS 語音合成 (不依賴 robot_controller)
        self._init_tts()
        
        logger.info("\n" + "="*70)
        logger.info("開始初始化失敗恢復系統...")
        logger.info("="*70)
        
        # 初始化失敗恢復系統 (不依賴 robot_controller，只需 WorldState)
        self._init_failure_recovery()
        
        logger.info(f"   檢查初始化結果:")
        logger.info(f"   failure_detector: {self.failure_detector is not None}")
        logger.info(f"   recovery_policy: {self.recovery_policy is not None}")
        logger.info("="*70 + "\n")
        
        # 初始化主動消遮擋策略 (不依賴 robot_controller)
        self._init_occlusion_strategy()
        
        if not self.enable_robot:
            logger.warning("機器人控制已停用 (僅感知模式)")
            # 即使不啟用 robot，也需要設定 robot_controller 為 None
            self.world_state_builder.robot_controller = None
            return
        
        try:
            self.robot_controller = RobotController()
            
            # 連接到 WorldStateBuilder（雙向連接）
            self.world_state_builder.robot_controller = self.robot_controller
            self.robot_controller.world_state_builder = self.world_state_builder  # 新增：讓 controller 能訪問 WorldState
            
            # 傳遞主系統的相機實例給 robot_controller（避免重複初始化）
            # 使用正視相機 D455 進行手部檢測（人在前方）
            if hasattr(self, 'd455_pipeline') and self.d455_pipeline:
                self.robot_controller.external_camera_pipeline = self.d455_pipeline
                self.robot_controller.external_camera_name = "D455"
                profile = self.d455_pipeline.get_active_profile()
                if profile:
                    stream_profile = profile.get_stream(rs.stream.color)
                    if stream_profile:
                        self.robot_controller.external_camera_intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
            
            # 傳遞俯視相機 D435i 給控制器（用於物體偵測）
            if hasattr(self, 'd435i_pipeline') and self.d435i_pipeline:
                self.robot_controller.external_top_camera_pipeline = self.d435i_pipeline
                profile = self.d435i_pipeline.get_active_profile()
                if profile:
                    stream_profile = profile.get_stream(rs.stream.color)
                    if stream_profile:
                        self.robot_controller.external_top_camera_intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
            
            # 傳遞 YOLO 檢測器給控制器（避免重複載入模型）
            if hasattr(self, 'object_detector') and self.object_detector:
                self.robot_controller.external_yolo_detector = self.object_detector
                logger.info("已傳遞 YOLO 檢測器給控制器（避免重複載入模型）")
            
            # 傳遞 VoiceQueryHandler 給控制器（用於步驟更新）
            if hasattr(self, 'voice_query_handler') and self.voice_query_handler:
                self.robot_controller.voice_query_handler = self.voice_query_handler
                logger.info("已傳遞 VoiceQueryHandler 給控制器（支援語音步驟更新）")

            # 傳遞 MemorySystem 給控制器（記錄 InteractionRecord）
            if hasattr(self, 'memory_system') and self.memory_system:
                self.robot_controller.memory_system = self.memory_system
                logger.info("已傳遞 MemorySystem 給控制器（InteractionLog 啟用）")
            
            # 設置 LLM compressor（用於 Temporal Memory 壓縮）
            # 注意：這是可選的，Memory System 有回退機制
            if hasattr(self, 'memory_system') and self.memory_system:
                try:
                    from ws_racr.config import get_openai_client, get_model_name
                    
                    def llm_compress_wrapper(prompt: str) -> str:
                        """直接調用 LLM 進行記憶壓縮（不需要結構化輸出）"""
                        try:
                            client = get_openai_client()
                            model = get_model_name()
                            
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a concise summarization assistant. Provide brief summaries without JSON formatting."
                                    },
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.3,
                                max_tokens=150
                            )
                            
                            summary = response.choices[0].message.content.strip()
                            return summary if summary else prompt
                        except Exception as e:
                            # 靜默失敗，回退到原始文字
                            return prompt
                    
                    self.memory_system.temporal_memory.llm_compressor = llm_compress_wrapper
                    logger.info("已設置 LLM compressor 用於記憶壓縮")
                except Exception as e:
                    logger.warning(f"LLM compressor 設置失敗（使用回退機制）: {e}")
            
            # 初始化雙相機系統（用於 pointing 和 receive 手勢）
            if self.robot_controller._init_dual_camera_pointing():
                logger.info("雙相機系統就緒")
            else:
                logger.warning("雙相機系統初始化失敗（手勢功能可能受限）")
            
            # 啟動新會話
            self.current_session_id = self.robot_controller.start_session()
            
            # 啟動任務執行線程
            self.task_thread_running = True
            self.task_thread = threading.Thread(target=self._task_execution_worker, daemon=True)
            self.task_thread.start()

            # 初始化 HRC Orchestrator（使用 TTS + 語音識別作為工具）
            def _listen_fn() -> Optional[str]:
                """Blocking STT: 4-second fixed-duration recording."""
                if self.voice_recognizer is not None:
                    return self.voice_recognizer.record_and_recognize(duration=4.0)
                return None

            self.orchestrator = HRCOrchestrator(
                controller=self.robot_controller,
                tts_handler=self.tts_engine,
                listen_fn=_listen_fn,
            )
            logger.info("HRC Orchestrator 已初始化")

            logger.info("機器人控制器就緒")
        except Exception as e:
            logger.warning(f"機器人控制器初始化失敗: {e}")
            self.enable_robot = False

    def _init_failure_recovery(self):
        """初始化失敗恢復系統 """
        logger.info("\n初始化失敗恢復系統...")
        try:
            # 使用 HYBRID 模式（人類優先 + 視覺備援）
            self.failure_detector = FailureDetector(detection_mode=FailureDetectionMode.HYBRID)
            self.recovery_policy = RecoveryPolicy(detection_mode=FailureDetectionMode.HYBRID)
            
            logger.info("失敗恢復系統就緒 (HYBRID 模式)")
            logger.info("   • 人類主動通知：沒夾到 <物體名稱>")
            logger.info("   • 視覺自動檢測：2.0s 延遲驗證")
            logger.info("   • 固定恢復策略：re_perceive + retry")
        except Exception as e:
            logger.error(f"失敗恢復系統初始化失敗: {e}")
            import traceback
            traceback.print_exc()
            self.failure_detector = None
            self.recovery_policy = None

    def _init_voice_query_handler(self):
        """初始化語音查詢處理器 """
        logger.info("\n初始化語音查詢處理器...")
        try:
            # 需要 memory_system 或 world_state_builder
            if not hasattr(self, 'world_state_builder') or not self.world_state_builder:
                logger.warning("WorldStateBuilder 未初始化，跳過 VoiceQueryHandler")
                self.voice_query_handler = None
                return
            
            # 創建適配器：將 WorldStateBuilder 包裝成 ObjectMemorySystem 接口
            class WorldStateAdapter:
                """適配器：將 WorldStateBuilder 適配為 ObjectMemorySystem 接口"""
                def __init__(self, world_state_builder):
                    self.world_state_builder = world_state_builder
                
                def query_inventory(self, visible_only=True):
                    """查詢物件庫存"""
                    world_state = self.world_state_builder.get_current_state()
                    inventory = []
                    for obj_id, obj in world_state.objects.items():
                        if visible_only and obj.visibility != 'visible':
                            continue
                        inventory.append({
                            'id': obj_id,
                            'class': obj.object_type,  # 修正：使用 object_type
                            'position': obj.position,
                            'confidence': obj.confidence
                        })
                    return inventory
                
                def locate_object(self, obj_class, visible_only=True):
                    """定位指定類別的物件"""
                    world_state = self.world_state_builder.get_current_state()
                    for obj_id, obj in world_state.objects.items():
                        if obj.object_type == obj_class:  # 修正：使用 object_type
                            if visible_only and obj.visibility != 'visible':
                                continue
                            return {
                                'id': obj_id,
                                'class': obj.object_type,  # 修正：使用 object_type
                                'position': obj.position,
                                'confidence': obj.confidence
                            }
                    return None
            
            # 創建 VoiceQueryHandler
            self.voice_query_handler = VoiceQueryHandler(
                object_memory_system=WorldStateAdapter(self.world_state_builder),
                step_fsm=self.progress_tracker,  # 使用 progress_tracker 作為 step_fsm
                world_state_getter=lambda: self.world_state_builder.get_current_state(),
                sop_path=str(get_sop_yaml_path()),
                spatial_memory=None,  # 可選
                world_state_updater=lambda sc, conf: self.world_state_builder.update_step_recognition(sc, conf),
            )
            
            logger.info("語音查詢處理器就緒")
            logger.info("   • 支援查詢: 進度、下一步、物件位置、庫存")
        except Exception as e:
            logger.error(f"語音查詢處理器初始化失敗: {e}")
            import traceback
            traceback.print_exc()
            self.voice_query_handler = None

    def _check_action_failure(self, action_type: str, task_json: Dict, execution_result: bool) -> bool:
        """
        在動作執行後檢查失敗並觸發恢復流程 
        
        Args:
            action_type: 動作類型 ('pickup', 'place', ...)
            task_json: 任務資訊
            execution_result: 執行是否成功
        
        Returns:
            bool: 如果需要進入恢復模式，返回 True
        """
        logger.info(f"\n   _check_action_failure() 被調用")
        logger.info(f"      failure_detector: {self.failure_detector is not None}")
        logger.info(f"      recovery_policy: {self.recovery_policy is not None}")
        
        if not self.failure_detector or not self.recovery_policy:
            logger.warning(f"   失敗檢測系統未初始化，跳過檢測")
            return False
        
        # 如果執行成功且沒有人類通知失敗，則不需檢查
        # (但留給視覺檢測的機會，可能執行者回傳成功但實際失敗)
        
        world_state = self.world_state_builder.get_current_state()
        
        # 優先使用具體的 object_id（如 cylinder_3），其次才是類型名稱
        object_id = task_json.get('target_object_id') or task_json.get('target_object', '')
        
        logger.info(f"   失敗檢測參數:")
        logger.info(f"      object_id: {object_id}")
        logger.info(f"      action_type: {action_type}")
        logger.info(f"      execution_result: {execution_result}")
        
        failure_result = None
        
        try:
            # 根據動作類型檢測失敗
            if action_type in ['pickup', 'grab', 'grasp']:
                # 建立簡單的 robot_status 物件（從 facts 取得）
                from types import SimpleNamespace
                robot_status = SimpleNamespace(
                    gripper_state=self.robot_controller.facts.gripper_state,
                    holding_object_id=self.robot_controller.facts.holding_object  # 使用 holding_object 而不是 holding
                )
                
                logger.info(f"      gripper_state: {robot_status.gripper_state}")
                logger.info(f"      holding_object_id: {robot_status.holding_object_id}")
                
                # 檢查物體是否在 world_state 中
                if object_id in world_state.objects:
                    obj = world_state.objects[object_id]
                    logger.info(f"      物體狀態: visibility={obj.visibility}, confidence={obj.confidence:.2f}")
                else:
                    logger.warning(f"      物體 {object_id} 不在 WorldState 中！")
                
                failure_result = self.failure_detector.detect_pickup_failure(
                    target_object_id=object_id,
                    robot_status=robot_status,
                    world_state=world_state,
                    action_result={'success': execution_result}
                )
            
            elif action_type in ['place', 'put', 'release']:
                target_zone = task_json.get('target_location', task_json.get('target_zone', ''))
                from types import SimpleNamespace
                robot_status = SimpleNamespace(
                    gripper_state=self.robot_controller.facts.gripper_state,
                    holding_object_id=self.robot_controller.facts.holding_object  # 使用 holding_object 而不是 holding
                )
                
                failure_result = self.failure_detector.detect_place_failure(
                    target_object_id=object_id,
                    target_zone=target_zone,
                    robot_status=robot_status,
                    world_state=world_state,
                    action_result={'success': execution_result}
                )
        
        except Exception as e:
            logger.warning(f"失敗檢測錯誤: {e}")
            return False
        
        # 如果檢測到失敗，觸發恢復流程
        if failure_result:
            logger.error(f"檢測到失敗: {failure_result['type']}")
            logger.info(f"   理由: {failure_result['reason']}")
            
            # 規劃恢復策略
            recovery_action = self.recovery_policy.plan_recovery(failure_result, world_state)
            
            if recovery_action:
                logger.info(f"恢復策略: {recovery_action['strategy']}")
                logger.info(f"   說明: {recovery_action.get('explanation', '')}")
                
                # TTS 語音提示（強調系統主導）
                if recovery_action.get('wait_for_voice'):
                    # 需要人類驗證的情況
                    self._speak_tts(f"檢測到{action_type}失敗，正在重新規劃", priority=self._TTS_TASK)
                    time.sleep(0.5)
                    self._speak_tts("Recovery 不確定，請提供驗證資訊", priority=self._TTS_TASK)
                else:
                    self._speak_tts(f"檢測到{action_type}失敗，正在自動重試", priority=self._TTS_TASK)
                
                # 記錄待執行的恢復動作
                self.pending_recovery = recovery_action
                self.in_recovery_mode = True
                
                return True
        
        return False

    def _init_occlusion_strategy(self):
        """初始化主動消遮擋策略 """
        try:
            from ws_racr.execution.safety.occlusion_strategy import OcclusionHandlingStrategy
            
            self.occlusion_strategy = OcclusionHandlingStrategy()
            logger.info("主動消遮擋策略已初始化")
            logger.info("   • Type A: 可移動物體（機器人自己處理）")
            logger.info("   • Type B: 盲抓/請求資訊")
            logger.info("   • Type C: 不可移動（人類介入）")
        except Exception as e:
            logger.warning(f"主動消遮擋策略初始化失敗: {e}")
            self.occlusion_strategy = None

    def _execute_deliver_materials(self, delivery_result: Dict):
        """執行零件遞送動作序列
        
        Args:
            delivery_result: {
                'step_num': int,
                'materials': ['platform', 'cylinder'],
                'actions': [
                    {'type': 'pickup_and_handover', 'object': 'platform', 'object_zh': '底座'},
                    {'type': 'pickup_and_handover', 'object': 'cylinder', 'object_zh': '圓柱'}
                ]
            }
        """
        try:
            actions = delivery_result.get('actions', [])
            
            if not actions:
                logger.warning("沒有需要遞送的零件")
                return
            
            logger.info(f"\n準備遞送 {len(actions)} 個零件...")
            self._speak_tts(f"我會為您遞送 {len(actions)} 個零件", priority=self._TTS_TASK)

            for i, action in enumerate(actions, 1):
                obj = action['object']
                obj_zh = action['object_zh']

                logger.info(f"   [{i}/{len(actions)}] 遞送 {obj_zh}...")
                self._speak_tts(f"正在遞送 {obj_zh}", priority=self._TTS_TASK)
                
                # 使用 LLM 執行組合動作：pickup + handover
                world_state = self.world_state_builder.get_current_state()
                task_id = self.robot_controller.execute_task_with_llm(
                    command=f"拿{obj_zh}遞給我",
                    world_state=world_state,
                    session_id=self.current_session_id,
                    memory_context=None
                )
                
                # 等待完成
                if task_id:
                    success = self._wait_for_task_completion(task_id, timeout=30.0)
                    if success:
                        logger.info(f"   {obj_zh} 遞送完成")
                    else:
                        logger.warning(f"   {obj_zh} 遞送超時")
                else:
                    logger.error(f"   {obj_zh} 任務創建失敗")
            
            logger.info(f"零件遞送完成！")
            self._speak_tts("所有零件已遞送完畢", priority=self._TTS_NORMAL)

        except Exception as e:
            logger.error(f"零件遞送失敗: {e}")
            self._speak_tts("零件遞送失敗", priority=self._TTS_NORMAL)

    def _wait_for_task_completion(self, task_id: str, timeout: float = 30.0) -> bool:
        """等待任務完成
        
        Args:
            task_id: 任務 ID
            timeout: 超時時間（秒）
        
        Returns:
            bool: 任務是否成功完成
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task_info = self.pending_tasks.get(task_id)
            
            # worker 完成後會設置 completed_at + success
            if task_info and 'completed_at' in task_info:
                return task_info.get('success', False)
            
            time.sleep(0.5)
        
        logger.warning(f"任務 {task_id} 超時")
        return False

    def _task_execution_worker(self):
        """任務執行工作線程 (背景執行機器人指令)"""
        logger.info("任務執行線程已啟動")
        
        while self.task_thread_running:
            try:
                # 從佇列取得任務 (阻塞,最多等待1秒)
                task_info = self.task_queue.get(timeout=1.0)

                # HRC Orchestrator 多步任務路徑
                if task_info.get('task_type') == 'orchestrator':
                    command = task_info.get('command', '')
                    deictic_snapshot = task_info.get('deictic_snapshot')
                    sid     = task_info.get('session_id')
                    # Always fetch a fresh world_state at execution time so that
                    # step updates made by previous tasks are visible.
                    # deictic_snapshot carries the pointing evidence captured at
                    # voice-input time and is injected below (separate concern).
                    ws = self.world_state_builder.get_current_state()
                    logger.info(f"\nOrchestrator 執行: {command}")
                    try:
                        self._inject_deictic_snapshot_into_world_state(ws, deictic_snapshot)
                        orch_result = self.orchestrator.run(command, world_state=ws, session_id=sid)
                        success = orch_result.success
                        if not success and orch_result.error:
                            logger.error(f"   Orchestrator 失敗: {orch_result.error}")
                    except Exception as _oe:
                        import traceback
                        logger.warning(f"Orchestrator 執行錯誤: {_oe}")
                        traceback.print_exc()
                        success = False
                    with self._task_submit_lock:
                        if self.task_queue.empty():
                            self.robot_busy = False
                    self.task_queue.task_done()
                    continue  # back to top of worker loop

                # [Compat] 舊 voice_command 路徑：
                # 優先收斂到 orchestrator，僅在 orchestrator 不可用時保留舊行為。
                if task_info.get('task_type') == 'voice_command':
                    _cmd = task_info['command']
                    _deictic_snapshot = task_info.get('deictic_snapshot')
                    _sid = task_info.get('session_id')
                    # Fresh world_state at execution time (same rationale as orchestrator path).
                    _ws = self.world_state_builder.get_current_state()

                    if self.orchestrator is not None:
                        logger.info(f"\n[Worker] voice_command → Orchestrator: {_cmd}")
                        try:
                            self._inject_deictic_snapshot_into_world_state(_ws, _deictic_snapshot)
                            orch_result = self.orchestrator.run(_cmd, world_state=_ws, session_id=_sid)
                            success = orch_result.success
                            if not success and orch_result.error:
                                logger.error(f"   Orchestrator 失敗: {orch_result.error}")
                        except Exception as _oe:
                            import traceback
                            logger.warning(f"Orchestrator 執行錯誤: {_oe}")
                            traceback.print_exc()
                            success = False
                    else:
                        logger.info(f"\n[Worker] 語音指令（LLM 解析中）: {_cmd}")
                        _task_id = self.robot_controller.process_user_command(
                            _cmd,
                            session_id=_sid,
                            memory_context=task_info.get('memory_context')
                        )
                        if _task_id:
                            logger.info(f"   任務已建立: {_task_id[:35]}...")
                            success = self._execute_task(_task_id)
                        else:
                            logger.error(f"   LLM 解析失敗，略過任務")
                            success = False
                    with self._task_submit_lock:
                        if self.task_queue.empty():
                            self.robot_busy = False
                    self.task_queue.task_done()
                    continue

                task_id = task_info['task_id']
                command = task_info['command']
                priority = task_info.get('priority', 0)
                
                logger.info(f"\n開始執行任務: {task_id[:35]}...")
                logger.info(f"   指令: {command}")
                logger.info(f"   優先級: {priority}")
                
                # 執行任務
                success = self._execute_task(task_id)
                
                # 更新狀態
                if task_id in self.pending_tasks:
                    self.pending_tasks[task_id]['success'] = success
                    self.pending_tasks[task_id]['completed_at'] = time.time()
                
                # 任務完成，佇列已空才釋放機器人（lock 保護）
                # 若佇列仍有項目（如 occlusion 子任務序列），保持 robot_busy=True
                # 直到最後一個子任務完成、佇列清空時才釋放
                with self._task_submit_lock:
                    if self.task_queue.empty():
                        self.robot_busy = False
                
                if success:
                    logger.info(f"   任務執行成功")
                    
                    # 手勢任務：執行完成後才記錄到資料庫
                    reason = task_info.get('reason', '')
                    is_gesture = reason in ["receive_gesture", "pointing_gesture", "stop_gesture"]
                    
                    if is_gesture:
                        logger.info(f"   記錄手勢任務執行結果到資料庫...")
                        self.logger.log_llm_interaction(
                            input_type="gesture",
                            world_state=task_info['world_state'],
                            voice_input=task_info['command'],
                            llm_response=f"手勢任務已完成: {task_info['command']}",
                            task_json={
                                'task_id': task_id,
                                'trigger': reason,
                                'priority': task_info.get('priority', 0),
                                'command_params': task_info.get('command_params', {}),
                                'execution_status': 'success',
                                'decision_source': 'gesture_direct_mapping'
                            },
                            robot_command=task_info['command']
                        )
                    
                    # 提取成功案例作為知識片段
                    self._extract_knowledge(task_info, success=True)
                else:
                    logger.error(f"   任務執行失敗")
                    
                    # 手勢任務失敗也要記錄
                    reason = task_info.get('reason', '')
                    is_gesture = reason in ["receive_gesture", "pointing_gesture", "stop_gesture"]
                    
                    if is_gesture:
                        logger.info(f"   記錄手勢任務失敗到資料庫...")
                        self.logger.log_llm_interaction(
                            input_type="gesture",
                            world_state=task_info['world_state'],
                            voice_input=task_info['command'],
                            llm_response=f"手勢任務失敗: {task_info['command']}",
                            task_json={
                                'task_id': task_id,
                                'trigger': reason,
                                'priority': task_info.get('priority', 0),
                                'command_params': task_info.get('command_params', {}),
                                'execution_status': 'failed',
                                'decision_source': 'gesture_direct_mapping'
                            },
                            robot_command=task_info['command']
                        )
                
                # 標記佇列任務完成
                self.task_queue.task_done()
                
            except queue.Empty:
                # 沒有任務,繼續等待
                continue
            except Exception as e:
                import traceback
                logger.warning(f"任務執行線程錯誤: {e}")
                traceback.print_exc()
                with self._task_submit_lock:
                    self.robot_busy = False
                # 確保 queue join() 不會永久阻塞
                try:
                    self.task_queue.task_done()
                except ValueError:
                    pass

    def _execute_task(self, task_id: str) -> bool:
        """執行單一任務"""
        try:
            # 從 TaskManager 取得任務資訊（用於失敗檢測）
            task_info = self.robot_controller.task_manager.get_task(task_id)
            task_json = task_info.get('task_json', {}) if task_info else {}
            action_type = task_json.get('action', 'unknown')
            
            logger.info(f"\n執行任務資訊:")
            logger.info(f"   task_id: {task_id[:40]}...")
            logger.info(f"   action: {action_type}")
            logger.info(f"   target_object: {task_json.get('target_object')}")
            logger.info(f"   target_object_id: {task_json.get('target_object_id', '未設定')}")
            
            # 使用 RobotController 執行任務
            success = self.robot_controller.execute_task(task_id)
            
            # Occlusion Handling
            if isinstance(success, dict):
                if success.get("feasibility_abort"):
                    msg = (success.get("speak_zh") or "目前無法執行。").strip()
                    self._speak_tts(msg, priority=self._TTS_NORMAL)
                    return False
                # 處理遮擋物移除序列（拆成獨立任務）
                if success.get('type') == 'occlusion_removal_sequence':
                    logger.info("\n收到遮擋物移除序列，拆分成獨立任務...")
                    
                    subtasks = success.get('subtasks', [])
                    occluder_name = success.get('occluder_info', {}).get('name', '遮擋物')
                    
                    logger.info(f"   共 {len(subtasks)} 個子任務:")
                    for i, st in enumerate(subtasks, 1):
                        logger.info(f"      {i}. {st['description']}")
                    
                    self._speak_tts(f"偵測到{occluder_name}遮擋，正在移除", priority=self._TTS_TASK)
                    
                    # 依序提交任務到 task queue
                    for subtask in subtasks:
                        # 構造 command
                        if subtask['action'] == 'pickup':
                            cmd = f"請幫我抓取{subtask['target_object_zh']}"
                        elif subtask['action'] == 'place':
                            cmd = f"請把它放到{subtask['target_location_zh']}"
                        else:
                            continue
                        
                        # 創建任務
                        world_state = self.world_state_builder.get_current_state()
                        sub_task_id = self.robot_controller.process_user_command(
                            cmd,
                            session_id=self.current_session_id
                        )
                        
                        if sub_task_id:
                            # 放入 task queue（不要立即執行）
                            self.task_queue.put({
                                'task_id': sub_task_id,
                                'command': cmd,
                                'world_state': world_state,
                                'reason': 'occlusion_removal_sequence',
                                'priority': 1  # 高優先級
                            })
                            logger.info(f"      子任務已加入佇列: {cmd}")
                    
                    logger.info("   所有子任務已加入執行佇列（序列化執行）")
                    return True  # 標記原始任務已處理
                
                # 檢查遮擋檢測
                if success.get('occlusion_detected'):
                    logger.warning("\n偵測到物體遮擋...")
                    
                    # 需要語音確認（platform/bracket/不明物體）
                    if success.get('needs_voice_confirmation'):
                        question = success.get('question', '物體被遮擋了，需要我幫你移開障礙物嗎？')

                        logger.info(f"   語音詢問: {question}")
                        self._speak_tts(question, priority=self._TTS_LLM)
                        
                        # 設置等待用戶語音回應的狀態
                        self._wait_state = 'occlusion_choice'
                        self._wait_context = {
                            'task_id': task_id,
                            'target_object': success.get('target_object'),
                            'target_object_zh': success.get('target_object_zh'),
                            'occluder_object': success.get('occluder_object'),
                            'occluder_name': success.get('occluder_name'),
                            'action': success.get('action'),
                            'question_asked_time': time.time()
                        }
                        
                        logger.info("   等待用戶語音回應: '請你幫我移開' 或 '我自己移'")
                        return False  # 任務暫停，等待用戶決定
                    
                    # 舊的遮擋等待邏輯（向後兼容）
                    elif success.get('occlusion_wait'):
                        explanation = success.get('explanation', '物體被遮擋，請移除障礙物')
                        self._speak_tts(explanation, priority=self._TTS_LLM)
                        self._wait_state = 'occlusion_clear'
                        self._wait_context = {'task_id': task_id}
                        return False

                # LLM 協作政策: wait（機器人主動等待）
                # NOTE: independent of occlusion_detected — must be top-level checks
                if success.get('policy_wait'):
                    reason = success.get('reason', '等待人類完成當前動作')
                    logger.info(f"   機器人待機: {reason}")
                    return False

                # LLM 協作政策: ask_human（請求人類澄清）
                if success.get('policy_ask_human'):
                    question = success.get('question', '請問您需要什麼協助？')
                    self._speak_tts(question, priority=self._TTS_LLM)
                    logger.info(f"   請求澄清: {question}")
                    return False

                if success.get('policy_answer'):
                    answer = success.get('answer_text', '')
                    self._speak_tts(answer, priority=self._TTS_LLM)
                    logger.info(f"   答覆: {answer}")
                    return False
            
            # 執行後失敗檢測
            if action_type in ['pickup', 'grab', 'grasp', 'place', 'put', 'release']:
                logger.info(f"檢查 {action_type} 執行結果...")
                
                # 重新讀取 task_json（execute_task 可能已更新 target_object_id）
                task_info_updated = self.robot_controller.task_manager.get_task(task_id)
                if task_info_updated:
                    task_json = task_info_updated.get('task_json', task_json)
                    logger.info(f"   重新讀取 task_json:")
                    logger.info(f"      target_object: {task_json.get('target_object')}")
                    logger.info(f"      target_object_id: {task_json.get('target_object_id', '未設定')}")
                
                # execute_task 是阻塞的，返回時動作應該已完成
                # 額外等待視覺系統更新（讓相機捕捉新畫面）
                logger.info("   等待視覺系統更新...")
                
                # 輪詢 WorldState 時間戳，確保獲取動作後的新畫面
                pre_action_timestamp = self.world_state_builder.current_state.timestamp
                max_wait = 5.0
                start_wait = time.time()
                
                while time.time() - start_wait < max_wait:
                    current_timestamp = self.world_state_builder.current_state.timestamp
                    if current_timestamp > pre_action_timestamp:
                        elapsed = time.time() - start_wait
                        logger.info(f"   視覺已更新 (耗時 {elapsed:.2f}s)")
                        break
                    time.sleep(0.2)  # 每 200ms 檢查一次
                else:
                    logger.warning(f"   視覺更新超時 ({max_wait}s)")
                
                # 重新取得最新的 world_state
                world_state = self.world_state_builder.get_current_state()
                
                # 檢查失敗並觸發恢復流程
                needs_recovery = self._check_action_failure(action_type, task_json, success)
                
                if needs_recovery and self.pending_recovery:
                    logger.info("檢測到可能失敗，詢問人類...")
                    
                    # TTS 詢問人類是否要重試
                    target_obj_zh = task_json.get('target_object_zh', task_json.get('target_object', '物體'))
                    question = f"{target_obj_zh}可能沒有夾起來，需要我再試一次嗎？"
                    self._speak_tts(question, priority=self._TTS_LLM)
                    
                    # 設置等待人類回應的狀態
                    self._wait_state = 'retry_confirm'
                    self._wait_context = {
                        'task_id': task_id,
                        'action_type': action_type,
                        'target_object': task_json.get('target_object'),
                        'target_object_zh': target_obj_zh,
                        'question_asked_time': time.time()
                    }
                    
                    logger.info("   等待用戶語音回應: '再試一次' 或 '不用了'")
                    return False  # 任務暫停，等待人類決定
            
            return success
        except Exception as e:
            logger.error(f"   任務執行異常: {e}")
            return False

    def _extract_knowledge(self, task_info: Dict, success: bool):
        """從成功任務中提取知識片段"""
        try:
            if not success:
                return
            
            # 構建知識內容
            content = f"成功執行 '{task_info['command']}' (觸發: {task_info['reason']})。" \
                     f"步驟: {task_info['world_state'].step_class}, " \
                     f"意圖: {task_info['world_state'].human.intent}"
            
            # 添加到知識庫
            self.logger.add_knowledge_fragment(
                fragment_type='robot_execution',
                title=f"執行 {task_info['command']}",
                content=content,
                source_type='robot_execution',
                source_data={
                    'task_id': task_info['task_id'],
                    'command': task_info['command'],
                    'reason': task_info['reason'],
                    'step_class': task_info['world_state'].step_class
                },
                confidence=1.0
            )
            
            logger.info(f"   知識片段已記錄")
        except Exception as e:
            logger.warning(f"   知識提取失敗: {e}")

    def _execute_robot_remove_occluder(self):
        """執行機器人移除障礙物的動作序列
        
        動作流程：
        1. 抓取障礙物
        2. 移動到放置區域
        3. 釋放障礙物
        4. 重試原始任務
        """
        if not self._wait_context:
            logger.warning("   無遮擋上下文，無法執行移除動作")
            return
        
        occluder_id = self._wait_context.get('occluder_object')
        original_task_id = self._wait_context.get('task_id')
        
        if not occluder_id:
            logger.warning("   無障礙物 ID，無法執行移除動作")
            self._speak_tts("抱歉，我找不到障礙物", priority=self._TTS_NORMAL)
            self._wait_state = 'idle'
            self._wait_context = None
            return
        
        logger.info(f"\n開始執行移除障礙物序列:")
        logger.info(f"   障礙物: {occluder_id}")
        logger.info(f"   原始任務: {original_task_id}")
        
        try:
            # Step 0: 暫停原始任務狀態（避免狀態衝突）
            if original_task_id:
                logger.info("\n   Step 0: 暫停原始任務狀態")
                self.robot_controller.task_manager.update_task_status(
                    original_task_id, "waiting_obstacle_removal",
                    error_message="等待移除遮擋物"
                )
                logger.info(f"      原始任務狀態已更新為 waiting_obstacle_removal")
            
            # DEBUG: 檢查當前 Facts Store 狀態
            logger.info("\n   DEBUG: 當前狀態檢查:")
            logger.info(f"      Facts.gripper_state: {self.robot_controller.facts.gripper_state}")
            logger.info(f"      Facts.holding_object: {self.robot_controller.facts.holding_object}")
            
            # 檢查 Robot State
            if hasattr(self.robot_controller.backend_manager, 'backend'):
                backend = self.robot_controller.backend_manager.backend
                if hasattr(backend, 'communicator') and hasattr(backend.communicator, 'robot_state'):
                    robot_state = backend.communicator.robot_state
                    logger.info(f"      RobotState.gripper_closed: {robot_state.get('gripper_closed', 'N/A')}")
                    logger.info(f"      RobotState.gripper_status: {robot_state.get('gripper_status', 'N/A')}")
            
            # Step 1: 抓取障礙物
            logger.info("\n   Step 1: 抓取障礙物")
            
            # 強制重置夾爪狀態（確保 IO 模組正常）
            logger.info("      重置夾爪狀態...")
            reset_command = "請打開夾爪"
            reset_task_id = self.robot_controller.process_user_command(
                reset_command,
                session_id=self.current_session_id
            )
            if reset_task_id:
                self.robot_controller.execute_task(reset_task_id)
                import time
                time.sleep(1.0)  # 等待 IO 模組穩定
                logger.info("      夾爪已重置")
            
            pickup_command = f"請幫我抓取{occluder_id}"
            pickup_task_id = self.robot_controller.process_user_command(
                pickup_command,
                session_id=self.current_session_id
            )
            
            if pickup_task_id:
                logger.info(f"      抓取任務已建立: {pickup_task_id[:35]}...")
                pickup_success = self.robot_controller.execute_task(pickup_task_id)
                
                if not pickup_success:
                    logger.error("      抓取失敗")
                    self._speak_tts("抱歉，我無法抓取障礙物", priority=self._TTS_NORMAL)
                    self._wait_state = 'idle'
                    self._wait_context = None
                    return
                
                logger.info("      抓取成功")
            
            # Step 2: 放置到放置區域
            logger.info("\n   Step 2: 放置障礙物到放置區")
            place_command = f"請把{occluder_id}放到放置區"
            place_task_id = self.robot_controller.process_user_command(
                place_command,
                session_id=self.current_session_id
            )
            
            if place_task_id:
                logger.info(f"      放置任務已建立: {place_task_id[:35]}...")
                place_success = self.robot_controller.execute_task(place_task_id)
                
                if not place_success:
                    logger.error("      放置失敗")
                    self._speak_tts("抱歉，我無法放置障礙物", priority=self._TTS_NORMAL)
                    self._wait_state = 'idle'
                    self._wait_context = None
                    return
                
                logger.info("      放置成功")
            
            # Step 3: 語音確認並重試原始任務
            logger.info("\n   Step 3: 重試原始任務")
            self._speak_tts("障礙物已移除，現在重新執行原來的任務", priority=self._TTS_TASK)
            
            # 重新檢測物體是否可見
            time.sleep(0.5)  # 等待 WorldState 更新
            
            # 重新執行原始任務
            if original_task_id:
                logger.info(f"      重試任務: {original_task_id[:35]}...")
                retry_success = self.robot_controller.execute_task(original_task_id)
                
                if retry_success:
                    logger.info("      原始任務執行成功")
                    self._speak_tts("完成了", priority=self._TTS_NORMAL)
                else:
                    logger.error("      原始任務仍然失敗")
                    self._speak_tts("抱歉，任務還是無法完成", priority=self._TTS_NORMAL)

        except Exception as e:
            logger.error(f"   移除障礙物過程異常: {e}")
            import traceback
            traceback.print_exc()
            self._speak_tts("抱歉，移除障礙物時發生錯誤", priority=self._TTS_NORMAL)
        
        finally:
            # 清除等待狀態
            self._wait_state = 'idle'
            self._wait_context = None
            logger.info("   遮擋處理流程結束\n")
