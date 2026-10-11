"""Event triggers that start collaboration: gesture-triggered assistance, trigger de-duplication and step-change announcements.

Mixed into :class:`ws_racr.app.system.HRCSystem`; relies on its ``self`` state."""

import time
import threading
import copy
from typing import Dict, List
import logging
from ws_racr.memory import MemoryQuery
from ws_racr.decision.language_mapper import language_mapper
logger = logging.getLogger(__name__)


class TriggerMixin:
    """Event triggers that start collaboration: gesture-triggered assistance, trigger de-duplication and step-change announcements."""

    def llm_reasoning(self):
        """
        🧠 決策引擎 - 根據 WorldState 決定機器人動作
        
        架構: 感知 → WorldState → 決策樹 → RAG檢索 → 機器人執行
        
        決策層級:
        1. 安全規則（硬規則，最高優先）
        2. 任務協助（基於步驟+手勢）
        3. 主動預測（基於進度+物件狀態）
        4. 學習優化（RAG 歷史案例）
        
        觸發機制（三層FSM）：
        - **手勢觸發** (gesture): 專用FSM控制，需回到idle才能再觸發
        - **語音觸發** (voice): 無限制，每次都執行（明確用戶意圖）
        - **自動觸發** (auto): 5秒冷卻，避免重複推理
        """
        if not self.enable_robot or not self.robot_controller:
            return
        
        if self.robot_busy:
            return  # 機器人忙碌中,跳過推理
        
        world_state = self.world_state_builder.get_current_state()
        # stabilizer 已在主循環調用，這裡不需要重複
        
        # 觸發條件檢查
        should_assist = False
        assist_reason = None
        assist_priority = 0  # 優先級: 0=低, 1=中, 2=高, 3=緊急
        
        # 手勢狀態機檢查（防止重複觸發）
        current_gesture = world_state.human.intent
        
        # 如果偵測到 idle 或其他不需重置的手勢，清空狀態允許下次觸發
        if current_gesture == "idle" or current_gesture not in self.gestures_need_reset:
            if self.last_triggered_gesture in self.gestures_need_reset:
                # print(f"   手勢重置: {self.last_triggered_gesture} → {current_gesture}")
                self.last_triggered_gesture = None
        
        # 如果當前手勢需要重置（receive, pointing），且上次觸發的就是這個手勢
        # 必須先檢測到 idle 或其他手勢才能再次觸發
        if current_gesture in self.gestures_need_reset:
            if self.last_triggered_gesture == current_gesture:
                # print(f"   手勢 {current_gesture} 需要重置（等待 idle 或其他手勢）")
                return  # 跳過，等待手勢變化
        
        # ==================== 第 1 層：安全規則（硬規則） ====================
        
        # 1.1 緊急停止（最高優先級）
        if world_state.human.safety == "EMERGENCY_STOP":
            should_assist = True
            assist_reason = "emergency_stop"
            assist_priority = 3
        
        # 1.2 人類入侵工作區（安全後退）
        elif world_state.human.safety == "COLLISION_RISK":
            should_assist = True
            assist_reason = "safety_retreat"
            assist_priority = 3
        
        # ==================== 第 2 層：任務協助（手勢 + 步驟） ====================
        
        # 2.1 手勢請求：receive
        #   語意：「我準備好接收了」或「遞給我」
        #   只有當機器人確實持有物體時才觸發動作（完成 handover）。
        #   機器人未持物時：僅播 TTS 詢問，不觸發機器人動作，避免手臂誤往人手移動。
        elif world_state.human.intent == "receive":
            is_holding = False
            try:
                facts = self.robot_controller.facts
                is_holding = bool(facts.holding_object) or (
                    getattr(facts, 'gripper_state', 'open') == 'closed'
                )
            except Exception:
                pass

            if is_holding:
                # 機器人手上有物體 → 完成交接
                should_assist = True
                assist_reason = "receive_gesture"
                assist_priority = 2
            else:
                # 機器人未持物 → 召喚 + 詢問，不觸發機械動作
                logger.info("   [receive] 機器人未持物，轉為召喚模式")
                self._speak_tts("需要我幫什麼嗎？", priority=self._TTS_GESTURE)
        
        # 2.2 手勢請求：pointing — 上下文模式（不自動執行動作）
        #   pointing 手勢只做目標識別，結果寫入 world_state.human.pointing_target，
        #   供後續語音指令的 LLM-A 用來解析「那個/這個」等指代詞。
        #   使用者流程：先指（pointing_target 被快取 5 秒）→ 按 V 說「把那個拿來」。
        elif world_state.human.intent == "pointing":
            _now = time.time()
            if not self._pointing_was_active:
                # Rising edge：本 pointing session 首幀，觸發背景偵測
                self._pointing_was_active = True
                # acquire(blocking=False)：若上一次偵測尚未完成則跳過（不排隊）
                if self._pointing_detect_lock.acquire(blocking=False):
                    threading.Thread(
                        target=self._run_pointing_detection,
                        daemon=True,
                    ).start()
            # 持續 pointing 期間刷新 TTL
            if self._pointing_cache:
                self._pointing_expiry = _now + 10.0
                world_state.human.pointing_target = self._pointing_cache
            # 不設 should_assist，不觸發機器人動作
        
        # 2.3 手勢請求：呼叫機器人
        elif world_state.human.intent == "call":
            should_assist = True
            assist_reason = "robot_召喚"
            assist_priority = 1
        
        # Pointing TTL 管理：pointing 停止後快取保留 5 秒供語音指令使用
        if world_state.human.intent != "pointing":
            self._pointing_was_active = False
            _now = time.time()
            if self._pointing_cache and _now < self._pointing_expiry:
                # TTL 未過期：持續注入 pointing_target
                world_state.human.pointing_target = self._pointing_cache
            elif self._pointing_cache and _now >= self._pointing_expiry:
                # TTL 已過期：清空快取
                self._pointing_cache = None
                world_state.human.pointing_target = None

        # 更新狀態記錄
        self.last_step_class = world_state.step_class

        # 執行推理
        if should_assist:
            self.request_robot_assistance(world_state, assist_reason, assist_priority)

    def request_robot_assistance(self, world_state, reason, priority=0):
        """
        🤖 請求機器人協助 - 整合決策樹 + RAG檢索 + LLM規劃
        
        流程:
        1. 根據 reason 構建結構化指令
        2. RAG 檢索歷史相似案例
        3. LLM 生成可執行計畫
        4. 提交到執行佇列
        
        Args:
            world_state: 當前世界狀態
            reason: 觸發原因（決策樹輸出）
            priority: 優先級 (0-3)
        """
        logger.info(f"\n機器人協助請求: {reason} (優先級: {priority})")
        
        # ==================== 1. 構建結構化指令（決策樹） ====================
        
        command = ""
        command_params = {}
        
        # 安全相關
        if reason == "emergency_stop":
            command = "立即停止所有動作"
            command_params = {"immediate": True, "reason": "emergency"}
            self._speak_tts("緊急停止！", priority=self._TTS_SAFETY, interrupt=True)
        
        elif reason == "safety_retreat":
            command = "安全後退到觀察位置"
            command_params = {"safety_distance": 500, "velocity": 30}
            self._speak_tts("偵測到安全風險，後退中", priority=self._TTS_SAFETY, interrupt=True)
        
        # 手勢請求
        elif reason == "receive_gesture":
            # receive 手勢 + 機器人持物 → 執行 handover（遞給使用者）
            # 此時已確認 is_holding=True（check_robot_assistance_needed 的前置條件），
            # 取出目前持有物件名稱作為 handover 目標。
            held_obj = None
            try:
                held_obj = self.robot_controller.facts.holding_object
            except Exception:
                pass
            command = "handover"
            command_params = {
                "target_object": held_obj or "current_object",
                "trigger": "receive_gesture",
            }
            self._speak_tts("好的，遞給你", priority=self._TTS_GESTURE)

        elif reason == "pointing_gesture":
            logger.warning("   [pointing_gesture] 此分支不應被觸發（pointing 已改為上下文模式）")
            return

        elif reason == "robot_召喚":
            command = "移動到用戶視野範圍"
            command_params = {"position": "front_view", "gesture_acknowledge": True}
            self._speak_tts("我在這裡，需要我幫忙嗎？", priority=self._TTS_GESTURE)
        
        # 主動任務協助
        elif reason == "tool_preparation":
            if "screw" in world_state.step_class:
                command = "準備六角扳手"
                command_params = {"tool": "六角扳手", "destination": "handover_position"}
                self._speak_tts("這個步驟需要扳手，我來遞給你", priority=self._TTS_TASK)
            else:
                command = "移動到待機位置"
                command_params = {"destination": "standby"}
                self._speak_tts("準備協助下一步驟", priority=self._TTS_TASK)
        
        elif reason == "part_preparation":
            # 根據步驟推測需要的零件
            step_to_part = {
                'place_platform': '平台',
                'place_bracket_on_table': '支架',
                'align_pi_on_bracket': 'PI字型工件'
            }
            part = step_to_part.get(world_state.step_class, '零件')
            command = f"準備 {part}"
            command_params = {"part": part, "destination": "assembly_area"}
            self._speak_tts(f"這個步驟需要{part}，我來準備", priority=self._TTS_TASK)
        
        elif reason == "retrieve_fallen_object":
            # 找到掉落的物件
            fallen_obj = None
            for obj in world_state.objects.values():
                if obj.position and obj.position[2] < 100:  # 異常低的 Z 座標
                    fallen_obj = obj
                    break
            
            if fallen_obj:
                command = f"撿起掉落的 {fallen_obj.object_type}"
                command_params = {
                    "target_object": fallen_obj.object_type,
                    "position": fallen_obj.position,
                    "return_to": "assembly_area"
                }
                self._speak_tts(f"偵測到{fallen_obj.object_type}掉落，我去撿起來", priority=self._TTS_TASK)
            else:
                return  # 找不到物件，取消
        
        elif reason == "quality_check":
            command = "執行組裝品質檢查"
            command_params = {
                "check_type": "quality_check",
                "expected_cylinders": 4,
                "actual_cylinders": world_state.assembly_progress.cylinders_inserted
            }
            self._speak_tts("正在確認組裝品質", priority=self._TTS_TASK)
        
        elif reason == "manual_command":
            # 手動指令從用戶輸入取得
            raw_command = getattr(self, '_manual_command_text', "移動到觀察位置")
            
            # 使用語言映射器解析中文指令
            parsed = language_mapper.parse_command(raw_command)
            logger.info(f"   手動指令: {raw_command}")
            
            # 重要：使用原始指令傳給 LLM，保留完整的空間描述詞（如"最右邊"）
            # 不使用簡化的 action_mapping，避免丟失語義信息
            command = raw_command  # 直接使用原始語音識別結果
            command_params = {
                "source": "manual",
                "raw_command": raw_command,  # 保存原始指令
                "action": parsed['action']  # 作為提示，但 LLM 會重新解析
            }
            
            # 如果有解析出物體，也加入提示（但不覆蓋原始指令）
            if parsed['objects']:
                command_params["target_object"] = parsed['objects'][0]
                command_params["target_object_zh"] = parsed['objects_zh'][0] if parsed['objects_zh'] else '物體'
            # 不提前 echo 用戶說的話——讓 execute_task 結果（ask_human / 執行完成）決定 TTS
        
        else:
            command = "移動到觀察位置"
            command_params = {"default_action": True}
            self._speak_tts("好的，我來了", priority=self._TTS_NORMAL)

        # ==================== 2. RAG 檢索相似歷史案例 ====================
        # 觸發類型分類（決定是否需要防重複機制）
        trigger_type = self._classify_trigger_type(reason)
        is_gesture_trigger = trigger_type == 'gesture'
        is_voice_trigger = trigger_type == 'voice'
        is_auto_trigger = trigger_type == 'auto'
        
        # 防重複觸發檢查（根據觸發類型）
        if not self._should_allow_trigger(trigger_type, reason):
            return  # 靜默忽略
        
        # 只有非手勢觸發的任務才需要 RAG（手勢已經是明確的機器人任務）
        if not is_gesture_trigger:
            # 統一記憶檢索接口：Memory System 整合短期 + 長期記憶
            try:
                # 構建記憶查詢
                memory_query = MemoryQuery(
                    query_text=f"What robot action is best for {reason} during {world_state.step_class}?",
                    query_type="episodic",
                    context={
                        'reason': reason,
                        'step': world_state.step_class,
                        'human_state': world_state.human.intent
                    }
                )
                
                # 取得整合的決策上下文（短期 Temporal + KG + 長期 DB RAG）
                decision_context = self.memory_system.get_decision_context(memory_query)
                
            except Exception as e:
                logger.warning(f"   Memory retrieval failed: {e}")
                decision_context = None
        else:
            decision_context = None
        
        # ==================== 3. 執行任務 ====================
        try:
            task_id = None
            
            # 手勢任務：直接執行（繞過 LLM 解析）
            if is_gesture_trigger:
                task_id = self.robot_controller.execute_gesture_directly(
                    gesture_type=reason,
                    params=command_params,
                    session_id=self.current_session_id
                )
            # 其他任務：[Fix-1] 非阻塞提交至 worker thread，LLM 在背景執行
            else:
                # [Fix-1] 主執行緒不再呼叫 LLM（避免感知循環凍結 500-2000ms）
                # [Fix-2] deepcopy world_state 作為觸發瞬間的 snapshot，
                #         避免 worker 拿到推理完成時已被更新的狀態
                with self._task_submit_lock:
                    if self.robot_busy:
                        logger.warning(f"   任務提交時機器人已忙（race condition 防護），略過")
                        return
                    self.robot_busy = True
                    self.last_robot_action = command
                    voice_task = {
                        'task_type': 'voice_command',
                        'command': command,
                        'command_params': command_params,
                        'session_id': self.current_session_id,
                        'memory_context': decision_context,
                        'world_state': copy.deepcopy(world_state),
                        'reason': reason,
                        'priority': priority,
                        'timestamp': time.time(),
                    }
                    self.task_queue.put(voice_task)
                logger.info(f"   語音任務已加入執行佇列（Worker 將背景呼叫 LLM）")
                return  # 非手勢路徑提前 return，不走下方 if task_id: 區塊
            
            if task_id:
                logger.info(f"   任務已建立: {task_id[:35]}...")
                
                # 非手勢任務在執行前記錄（因為有LLM推理過程）
                # 手勢任務在執行後記錄（直接執行，等完成再記錄結果）
                if not is_gesture_trigger:
                    # ==================== 4a. 記錄LLM推理決策到資料庫 ====================
                    self.logger.log_llm_interaction(
                        input_type="auto_trigger" if reason != "manual_command" else "manual",
                        world_state=world_state,
                        voice_input=command,
                        llm_response=f"正在執行: {command}",
                        task_json={
                            'task_id': task_id,
                            'trigger': reason,
                            'priority': priority,
                            'command_params': command_params,
                            'decision_source': 'memory_guided'
                        },
                        robot_command=command
                    )
                
                # ==================== 5. 提交到執行佇列 ====================
                task_info = {
                    'task_id': task_id,
                    'command': command,
                    'command_params': command_params,
                    'reason': reason,
                    'priority': priority,
                    'timestamp': time.time(),
                    'world_state': world_state
                }
                
                # 原子性 check-and-set：防止 main thread 與 voice/worker thread 同時提交
                with self._task_submit_lock:
                    if self.robot_busy:
                        logger.warning(f"   任務提交時機器人已忙（race condition 防護），略過")
                        return
                    self.robot_busy = True
                    self.last_robot_action = command
                    self.pending_tasks[task_id] = task_info
                    self.robot_action_queue.append(task_info)
                    self.task_queue.put(task_info)
                
                # 更新手勢狀態（用於狀態機防止重複觸發，不需要在 lock 內）
                if is_gesture_trigger:
                    gesture_name = world_state.human.intent
                    self.last_triggered_gesture = gesture_name
                    logger.info(f"   觸發手勢: {gesture_name}（需等待 idle 或其他手勢才能再次觸發）")
                
                logger.info(f"   任務已加入執行佇列")
        
        except Exception as e:
            import traceback
            logger.error(f"   機器人協助請求失敗: {e}")
            traceback.print_exc()

    def _classify_trigger_type(self, reason: str) -> str:
        """
        分類觸發類型
        
        Returns:
            'gesture': 手勢觸發（由 FSM 控制）
            'voice': 語音/打字觸發（允許連續）
            'auto': 自動推理觸發（需要冷卻）
        """
        if reason in ["receive_gesture", "pointing_gesture", "stop_gesture"]:
            return 'gesture'
        elif reason in ["manual_command"]:
            return 'voice'
        else:
            return 'auto'

    def _should_allow_trigger(self, trigger_type: str, reason: str) -> bool:
        """
        檢查是否允許觸發（防重複機制）
        
        Args:
            trigger_type: 觸發類型 ('gesture', 'voice', 'auto')
            reason: 觸發原因
            
        Returns:
            True: 允許觸發
            False: 冷卻中，拒絕觸發
        """
        control = self.trigger_control.get(trigger_type)
        if not control:
            return True  # 未知類型，允許觸發
        
        current_time = time.time()
        
        # 語音觸發：始終允許（明確的用戶意圖）
        if trigger_type == 'voice':
            control['last_trigger'] = current_time
            return True
        
        # 手勢觸發：由專用 FSM 控制（last_triggered_gesture）
        # 這裡不額外限制，在 llm_reasoning() 中已經檢查過
        if trigger_type == 'gesture':
            control['last_trigger'] = current_time
            return True
        
        # 自動觸發：需要冷卻時間
        if trigger_type == 'auto':
            last_trigger = control.get('last_trigger', 0)
            last_reason = control.get('last_reason')
            cooldown = control.get('cooldown', 5.0)
            
            # 同一任務在冷卻時間內不重複觸發
            if last_reason == reason and (current_time - last_trigger) < cooldown:
                return False
            
            # 更新狀態
            control['last_trigger'] = current_time
            control['last_reason'] = reason
            return True
        
        return True

    def _search_similar_cases(self, world_state, reason) -> List[Dict]:
        """從資料庫搜尋相似案例 (RAG)"""
        try:
            if not self.logger:
                return []
            
            # 構建查詢文本
            query_text = f"步驟: {world_state.step_class}, 觸發原因: {reason}, 意圖: {world_state.human.intent}"
            
            # 使用 PerceptionLogger 的 RAG 檢索
            similar = self.logger.search_similar_interactions(query_text, top_k=3)
            
            return similar
        except Exception as e:
            logger.warning(f"   RAG 檢索失敗: {e}")
            return []

    def _announce_step_change(self, old_step: int, new_step: int):
        """步驟變更時的 TTS 播報 + StepTransitionMemory 記錄.

        Args:
            old_step: 舊步驟編號
            new_step: 新步驟編號
        """
        # === 記錄 StepTransitionMemory ===
        try:
            from ws_racr.memory import StepTransitionMemory
            ws = self.world_state_builder.current_state
            objects_present = [
                obj.object_type
                for obj in ws.objects.values()
                if obj.visibility == 'visible'
            ]
            transition = StepTransitionMemory(
                from_step=old_step,
                to_step=new_step,
                from_step_class=getattr(self, '_last_step_class', ws.step_class),
                to_step_class=ws.step_class,
                objects_present=objects_present,
                interaction_state=ws.interaction_state,
                robot_last_action=ws.robot.last_action,
            )
            if self.memory_system:
                self.memory_system.record_step_transition(transition)
                logger.info(f"   StepTransition recorded: {transition.to_text()}")
        except Exception as e:
            logger.warning(f"   StepTransition 記錄失敗: {e}")

        # 步驟名稱映射（從 SOP 取得）
        step_names = {
            0: "待機",
            1: "放置平台",
            2: "插入圓柱",
            3: "鎖緊螺絲",
            4: "放置支架",
            5: "扳手歸位",
            6: "組裝支架和PI工件",
            7: "鎖緊螺帽",
            8: "對齊並插入大螺絲",
            9: "鎖緊最終螺帽"
        }
        
        step_name = step_names.get(new_step, f"步驟{new_step}")
        
        logger.info(f"\n步驟變更: {old_step} → {new_step} ({step_name})")
        
        if new_step == 0:
            self._speak_tts("系統待機中", priority=self._TTS_NORMAL)
        elif new_step > old_step:
            message = f"現在進入步驟 {new_step}，{step_name}"
            self._speak_tts(message, priority=self._TTS_TASK)

            if self.voice_query_handler and hasattr(self.voice_query_handler, 'sop_data'):
                try:
                    for phase_name in ['phase_1', 'phase_2', 'phase_3']:
                        phase_data = self.voice_query_handler.sop_data.get(phase_name, {})
                        steps = phase_data.get('steps', [])
                        for step in steps:
                            if step.get('id') == new_step:
                                obj = step.get('object')
                                if obj:
                                    obj_zh = self.voice_query_handler._translate_object_name_to_zh(obj)
                                    material_msg = f"請準備 {obj_zh}"
                                    logger.info(f"   {material_msg}")
                                    # 延遲 1.5s 避免與步驟宣告重疊
                                    import threading
                                    threading.Timer(1.5, lambda m=material_msg: self._speak_tts(m, priority=self._TTS_TASK)).start()
                                break
                except Exception as e:
                    logger.warning(f"   材料提示失敗: {e}")
        elif new_step < old_step:
            self._speak_tts(f"返回步驟 {new_step}", priority=self._TTS_TASK)
