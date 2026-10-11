"""Voice side of the runtime: speech-to-text / text-to-speech set-up and routing of recognised utterances (reply to a pending question, recovery command, spoken query, new command).

Mixed into :class:`ws_racr.app.system.HRCSystem`; relies on its ``self`` state."""

import time
import threading
import queue
from typing import Optional, Dict
import logging
from ws_racr.hri.voice_query_handler import VoiceQueryHandler
logger = logging.getLogger(__name__)


class VoiceMixin:
    """Voice side of the runtime: speech-to-text / text-to-speech set-up and routing of recognised utterances (reply to a pending question, recovery command, spoken query, new command)."""

    def _handle_waiting_voice_reply(self, text: str) -> bool:
        """處理等待中的對話回覆（retry/occlusion）。回傳 True 表示已處理。"""
        # === 優先級 0-1: 檢查是否為重試確認回應 ===
        if self._wait_state == 'retry_confirm':
            text_lower = text.lower()

            if any(keyword in text_lower for keyword in ['再試', '重試', '再來', '再夾', '好', '是', 'yes']):
                # 用戶確認重試
                logger.info("   用戶確認：重試夾取")
                self._speak_tts("好的，我再試一次", priority=self._TTS_NORMAL)

                # 重新提交任務到 queue
                target_obj_zh = self._wait_context['target_object_zh']
                cmd = f"請幫我抓取{target_obj_zh}"
                world_state = self.world_state_builder.get_current_state()

                # 清除等待狀態（先清除，防止後續語音誤觸再次進入）
                self._wait_state = 'idle'
                self._wait_context = None

                # 立即鎖住機器人，防止其他任務插隊
                with self._task_submit_lock:
                    self.robot_busy = True

                # 在背景執行 LLM 呼叫，避免阻塞語音識別線程
                def _create_and_submit_retry(cmd=cmd, world_state=world_state):
                    new_task_id = self.robot_controller.process_user_command(
                        cmd,
                        session_id=self.current_session_id
                    )
                    if new_task_id:
                        self.task_queue.put({
                            'task_id': new_task_id,
                            'command': cmd,
                            'world_state': world_state,
                            'reason': 'retry_after_confirmation',
                            'priority': 2
                        })
                        logger.info(f"   重試任務已加入佇列: {cmd}")
                    else:
                        logger.error("   重試任務創建失敗，釋放機器人")
                        with self._task_submit_lock:
                            self.robot_busy = False

                threading.Thread(target=_create_and_submit_retry, daemon=True).start()

            elif any(keyword in text_lower for keyword in ['不用', '取消', '算了', '沒關係', 'no']):
                # 用戶拒絕重試
                logger.info("   用戶取消重試")
                self._speak_tts("好的，取消重試", priority=self._TTS_NORMAL)

                # 清除等待狀態
                self._wait_state = 'idle'
                self._wait_context = None

            else:
                # 無法識別的回應，重新詢問
                logger.warning(f"   無法識別回應: {text}")
                retry_question = "我沒聽清楚，需要我再試一次嗎？"
                self._speak_tts(retry_question, priority=self._TTS_LLM)

            return True

        # === 優先級 0-2: 檢查是否為遮擋處理選擇回應 ===
        if self._wait_state == 'occlusion_choice' and self._wait_context:
            logger.info(f"   處理遮擋選擇回應...")

            # 解析用戶選擇
            text_lower = text.lower().replace(' ', '')

            if any(keyword in text_lower for keyword in ['請你幫我', '你幫我', '機器人', '幫我移', '幫忙移']):
                # 用戶選擇：讓機器人移除障礙物
                logger.info("   用戶選擇：機器人移除障礙物")
                self._speak_tts("好的，我來幫你移開障礙物", priority=self._TTS_NORMAL)

                # 執行移除障礙物的動作序列
                self._execute_robot_remove_occluder()

            elif any(keyword in text_lower for keyword in ['我自己', '我來', '人工', '手動', '等一下']):
                # 用戶選擇：自己移除障礙物
                logger.info("   用戶選擇：人類移除障礙物")
                self._speak_tts("好的，請你移開障礙物，移開後告訴我繼續", priority=self._TTS_NORMAL)

                # 轉換到等待人類移除狀態（保留 task_id）
                self._wait_state = 'occlusion_clear'
                self._wait_context = {'task_id': self._wait_context['task_id']}

            else:
                # 無法識別的回應，重新詢問
                logger.warning(f"   無法識別回應: {text}")
                retry_question = "我沒聽清楚，請問要我幫你移開障礙物嗎？還是你自己移開？"
                self._speak_tts(retry_question, priority=self._TTS_LLM)

            return True

        return False

    def _handle_new_voice_command(self, text: str) -> None:
        """處理新語音指令（query/recovery/orchestrator）。"""
        if self._try_handle_voice_query(text):
            return
        if self._try_handle_recovery_command(text):
            return
        if self._try_handle_failure_notification(text):
            return

        # === 優先級 4: 正常指令處理（機器人動作）===
        if self.enable_robot:
            world_state = self.world_state_builder.get_current_state()
            # 透過 Orchestrator 路由（支援多步任務分解）
            if self.orchestrator is not None:
                with self._task_submit_lock:
                    if self.robot_busy:
                        logger.warning("   機器人忙碌中，略過指令")
                        return
                    self.robot_busy = True
                # Capture deictic evidence now, but do NOT rewrite command here.
                # Actual grounding remains single-source in orchestrator FSM.
                deictic_snapshot = self._capture_deictic_snapshot(world_state)
                task_info = {
                    'task_type': 'orchestrator',
                    'command': text,
                    'world_state': world_state,
                    'deictic_snapshot': deictic_snapshot,
                    'session_id': self.current_session_id,
                    'priority': 1,
                }
                self.task_queue.put(task_info)
                logger.info(f"   Orchestrator 任務已加入佇列: {text}")
            else:
                # fallback: 原有單步路徑
                self._manual_command_text = text
                self.request_robot_assistance(world_state, "manual_command", priority=1)

    def _capture_deictic_snapshot(self, world_state) -> Dict:
        """Capture pointing evidence at enqueue time (TTL-safe)."""
        human = getattr(world_state, "human", None)
        return {
            "pointing_target": getattr(human, "pointing_target", None),
            "pointing_pixel": getattr(human, "pointing_pixel", None),
            "pointing_position": getattr(human, "pointing_position", None),
            "captured_at": time.time(),
        }

    def _inject_deictic_snapshot_into_world_state(self, world_state, snapshot: Optional[Dict]) -> None:
        """Inject snapshot back before orchestrator.run so FSM sees stable evidence."""
        if not snapshot:
            return
        human = getattr(world_state, "human", None)
        if human is None:
            return
        try:
            if snapshot.get("pointing_target"):
                human.pointing_target = snapshot.get("pointing_target")
            if snapshot.get("pointing_pixel") is not None:
                human.pointing_pixel = snapshot.get("pointing_pixel")
            if snapshot.get("pointing_position") is not None:
                human.pointing_position = snapshot.get("pointing_position")
        except Exception:
            return

    def _try_handle_voice_query(self, text: str) -> bool:
        """嘗試以舊 query handler 處理；已處理回傳 True。"""
        # 若 Orchestrator 可用，跳過此層：Orchestrator 透過 LLM 分解指令，
        # 能同時處理純查詢、純執行及混合指令（如「先往下再告訴我桌上有什麼」），
        # 由 answer/ask/execute 步驟統一路由，不需 pre-filter。
        if self.voice_query_handler and self.orchestrator is None:
            try:
                result = self.voice_query_handler.parse_and_execute(text)
                if result['success']:
                    logger.info(f"[查詢成功] 類型: {result['query_type']}")
                    logger.info(f"回答: {result['response_text']}")
                    self._speak_tts(result['response_text'], priority=self._TTS_LLM)
                    return True
            except Exception as e:
                logger.warning(f"[查詢錯誤] VoiceQueryHandler 異常: {e}")
        return False

    def _try_handle_recovery_command(self, text: str) -> bool:
        """嘗試處理 recovery mode 指令；已處理回傳 True。"""
        if self.recovery_policy and self.in_recovery_mode:
            recovery_action = self.recovery_policy.apply_voice_guidance(text)
            if recovery_action:
                logger.info(f"應用恢復動作: {recovery_action['strategy'].value}")
                # TODO: 執行恢復動作（需要整合到 robot_controller）
                self.in_recovery_mode = False
                return True
        return False

    def _try_handle_failure_notification(self, text: str) -> bool:
        """嘗試處理失敗通知語句；已處理回傳 True。"""
        if self.recovery_policy and self.recovery_policy.voice_parser.is_failure_notification(text):
            logger.info("收到人類失敗通知")
            self.recovery_policy.apply_voice_guidance(text)
            return True
        return False

    def _init_voice_recognition(self):
        """初始化語音識別器"""
        try:
            from ws_racr.hri.voice_recognizer import VoiceRecognizerSD
            self.voice_recognizer = VoiceRecognizerSD(language='zh-TW')
            logger.info("STT 後端: Google Web Speech API")

            # 設定回調函數
            def on_result(text):
                if self._handle_waiting_voice_reply(text):
                    return
                self._handle_new_voice_command(text)
            
            def on_error(error):
                logger.error(f"\n語音識別錯誤: {error}")
            
            def on_start():
                self.voice_recording = True
                logger.info("\n開始錄音... (再按 V 停止)")
            
            def on_stop():
                self.voice_recording = False
                logger.info("\n停止錄音，識別中...")
            
            self.voice_recognizer.set_callbacks(
                on_result=on_result,
                on_error=on_error,
                on_start_recording=on_start,
                on_stop_recording=on_stop
            )
            
            logger.info("語音識別器已初始化")
            
        except Exception as e:
            logger.warning(f"語音識別初始化失敗: {e}")
            logger.info("   將使用文字輸入模式")
            self.voice_recognizer = None

    def _init_tts(self):
        """初始化 TTS 語音合成"""
        try:
            from ws_racr.hri.tts_handler import (
                TTSHandler,
                TTS_PRIORITY_SAFETY, TTS_PRIORITY_GESTURE,
                TTS_PRIORITY_TASK, TTS_PRIORITY_NORMAL, TTS_PRIORITY_LLM,
            )
            self.tts_engine = TTSHandler(
                voice='zh-TW-HsiaoChenNeural',
                rate='+10%',
            )
            # 掛載 priority 常數，讓 _speak_tts 可以取用
            self._TTS_SAFETY  = TTS_PRIORITY_SAFETY
            self._TTS_GESTURE = TTS_PRIORITY_GESTURE
            self._TTS_TASK    = TTS_PRIORITY_TASK
            self._TTS_NORMAL  = TTS_PRIORITY_NORMAL
            self._TTS_LLM     = TTS_PRIORITY_LLM
            logger.info("TTS 語音合成已初始化（queue + cache）")
        except Exception as e:
            logger.warning(f"TTS 初始化失敗: {e}")
            self.tts_engine = None
            self._TTS_SAFETY = self._TTS_GESTURE = self._TTS_TASK = \
                self._TTS_NORMAL = self._TTS_LLM = 5

    def _speak_tts(self, text: str, priority: int = 5, interrupt: bool = False):
        """TTS 播報文字。

        Args:
            text:      播報內容
            priority:  0=安全/最高… 7=LLM 長回答（預設 5）
            interrupt: True 時清空佇列並立即插隊（緊急訊息用）
        """
        if not text or not hasattr(self, 'tts_engine') or self.tts_engine is None:
            return
        try:
            self.tts_engine.speak(text, priority=priority, interrupt=interrupt)
        except Exception as e:
            logger.warning(f"TTS 播報失敗: {e}")

    def manual_voice_command(self):
        """手動語音指令輸入（支援真實語音或文字輸入）"""
        if self.voice_recognizer is not None:
            # 使用真實語音識別
            if self.voice_recording:
                # 已在錄音，停止錄音
                self.voice_recognizer.stop_recording()
            else:
                # 開始錄音
                self.voice_recognizer.start_recording()
        else:
            # 回退到文字輸入
            print("\n⌨️ 請輸入指令 (語音識別未啟用): ", end='')
            command = input().strip()
            
            if command:
                # === 優先級 1: 檢查是否為查詢（Query）===
                if self.voice_query_handler:
                    result = self.voice_query_handler.parse_and_execute(command)
                    
                    if result['success']:
                        logger.info(f"查詢類型: {result['query_type']}")
                        logger.info(f"回答: {result['response_text']}")
                        self._speak_tts(result['response_text'], priority=self._TTS_LLM)
                        return
                
                # === 優先級 2: 機器人動作指令 ===
                if self.enable_robot:
                    # 儲存指令供後續使用
                    self._manual_command_text = command
                    world_state = self.world_state_builder.get_current_state()
                    self.request_robot_assistance(world_state, "manual_command", priority=1)
