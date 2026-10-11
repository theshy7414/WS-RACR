"""RobotController: executes validated skill commands on the robot.

Pipeline: parse the command -> validate the action -> persist the task (PostgreSQL) -> generate
waypoints -> run through the selected execution backend (MoveIt2 / TM SetPositions / TMScript)
-> record the outcome. Waypoint generation, perception helpers and low-level execution live in the
mixins (``grasp_planner``, ``perception_adaptor``, ``execution_mixin``).
"""
import logging
import os
import time
from typing import Dict, Optional, List, Union, Any, Tuple
from datetime import datetime
import numpy as np

# 步驟 2.5 硬性中止時給 TTS 的短句（避免朗讀長串 error_message）
_FEASIBILITY_ABORT_OBJ_ZH: Dict[str, str] = {
    "cylinder": "圓柱",
    "screw_big": "大螺絲",
    "screw_small": "小螺絲",
    "screw": "螺絲",
    "nut": "螺帽",
    "hex_wrench": "扳手",
    "bracket": "支架",
    "platform": "平台",
    "pi": "PI",
}


def _feasibility_abort_speak_zh(task_json: Dict[str, Any]) -> Dict[str, Any]:
    obj = (task_json.get("target_object") or "").strip()
    label = _FEASIBILITY_ABORT_OBJ_ZH.get(obj, obj or "東西")
    return {
        "feasibility_abort": True,
        "speak_zh": f"目前看不到{label}，先不動。",
    }

from ws_racr.config import FROM_ROS, get_backend_config
from ws_racr.database.task_manager import TaskManager
from ws_racr.decision.command_parser import process_user_input
from ws_racr.execution.backends.backends import ExecutionBackendManager
from ws_racr.execution.backends.moveit_planner import MoveItPlanner
from ws_racr.execution.backends.waypoint_schema import Waypoint
from ws_racr.execution.controller.execution_mixin import ExecutionMixin
from ws_racr.execution.controller.grasp_planner import GraspPlannerMixin
from ws_racr.execution.controller.perception_adaptor import PerceptionAdaptorMixin
from ws_racr.execution.safety.occlusion_strategy import get_occlusion_handler
from ws_racr.execution.skills.action_validator import get_action_validator
from ws_racr.execution.skills.plan_validator import PlanValidator
from ws_racr.world_state.facts_store import FactsStore

logger = logging.getLogger(__name__)

# 早期查詢路由：已知機器人動作集合（不在此集合內 + 含問句標誌詞 → 判定為查詢）
_KNOWN_ROBOT_ACTIONS = {
    'pickup', 'place', 'handover', 'move', 'move_relative', 'return_to_origin',
    'return_to_last', 'rotate_gripper', 'gripper_open', 'gripper_close',
    'prepare', 'stop', 'emergency_stop', 'retreat',
}
_QUESTION_MARKERS = ('哪些', '什麼', '哪裡', '在哪', '幾個', '幾步', '多少', '嗎', '呢',
                     '哪邊', '哪方向', '需要', '還缺', '還在')

# ROS communication is optional (kept for backward compatibility; the backend manager is preferred).
try:
    from ws_racr.execution.backends.ros_communicator import RobotCommunicator
    ROS_AVAILABLE = True
except ImportError:
    RobotCommunicator = None
    ROS_AVAILABLE = False


class RobotController(GraspPlannerMixin, PerceptionAdaptorMixin, ExecutionMixin):
    """Database-backed robot controller: MoveIt2 planning plus TM SetPositions / TMScript execution."""
    
    def __init__(self):
        logger.info("Initialising RobotController...")
        
        # 初始化任務管理器
        self.task_manager = TaskManager()

        # 語意不明／澄清對話：PostgreSQL + 每 10 筆 LLM 壓縮
        self.clarification_dialogue_store = None
        try:
            from ws_racr.memory.clarification_store import ClarificationDialogueStore as _CDS
            self.clarification_dialogue_store = _CDS(self.task_manager.engine)
            logger.info("澄清對話儲存（DB + 階層壓縮）已啟用")
        except Exception as e:
            logger.warning(f"澄清對話儲存未啟用: {e}")
        
        # 會話管理
        self.session_id = None  # 當前對話會話 ID
        
        # WorldStateBuilder 引用（由 HRCSystem 設置，用於遮擋檢測）
        self.world_state_builder = None
        
        # VoiceQueryHandler 引用（由 HRCSystem 設置，用於語音查詢和步驟更新）
        self.voice_query_handler = None

        # MemorySystem 引用（由 HRCSystem 設置，用於寫入 InteractionRecord）
        self.memory_system = None
        
        # 初始化 MoveIt2 規劃器
        logger.info("使用 MoveIt2 規劃器")
        self.moveit_planner = MoveItPlanner(group_name='tmr_arm')
        if not self.moveit_planner.initialize():
            logger.warning("MoveIt2 規劃器初始化失敗")
        
        self.planner_type = 'moveit'
        
        # 初始化計畫驗證器
        self.plan_validator = PlanValidator()
        
        self.facts = FactsStore()
        
        # 初始化 Action Validator
        self.action_validator = get_action_validator()
        logger.info("Action Validator 已初始化")
        
        # 初始化 Occlusion Handler (智能三層分類)
        self.occlusion_handler = get_occlusion_handler()
        logger.info("Occlusion Handler 已初始化")

        # 初始化執行後端
        backend_config = get_backend_config()
        self.backend_type = backend_config['type']
        logger.info(f"執行後端: {self.backend_type}")
        
        self.backend_manager = ExecutionBackendManager(self.backend_type)
        if not self.backend_manager.initialize():
            logger.warning(f"{self.backend_type} 後端初始化失敗")
        
        # 視覺系統（延遲初始化）
        self.vision = None
        self.external_camera_pipeline = None  # 主系統的相機實例
        self.external_camera_intrinsics = None  # 主系統的相機內參
        self.external_camera_name = "D435i"  # 預設使用 D435i，主系統會覆寫
        self.external_top_camera_pipeline = None  # 主系統的俯視相機 D435i
        self.external_top_camera_intrinsics = None  # 俯視相機內參
        self.external_yolo_detector = None  # 主系統的 YOLO 檢測器（避免重複載入）
        
        # 雙相機手部指向檢測系統（延遲初始化）
        self.dual_camera_initialized = False
        self.front_pipeline = None  # D455 前視相機
        self.top_pipeline = None    # D435i 俯視相機
        self.front_intrinsics = None
        self.top_intrinsics = None
        self.front_align = None
        self.top_align = None
        self.T_front_to_base = None
        self.T_top_to_base = None
        self.mp_hands = None
        self.hands = None
        self.yolo_model = None
        
        # 手部指向檢測參數（從 hand_pointing_dual_camera.py）
        self.THETA_MAX_DEG = 30.0  # 最大角度誤差
        self.MISS_MAX_M = 0.20     # 最大距離誤差（公尺）
        self.HAND_DEPTH_SCALE = 0.5  # MediaPipe 手部深度估計
        
        # 俯視相機物體位置偏差修正（單位：公尺）
        # 用於修正夾取時物體位置的細微偏差
        self.TOP_CAMERA_X_BIAS = 0.0  # X 軸偏移修正
        self.TOP_CAMERA_Y_BIAS = 0.0  # Y 軸偏移修正
        self.TOP_CAMERA_Z_BIAS = 0.005  # Z 軸偏移修正
        
        # 物體類別映射
        self.CLASS_MAPPING = {
            0: 'bracket',
            1: 'cylinder',
            2: 'hex_wrench',
            3: 'nut',
            4: 'pi',
            5: 'platform',
            6: 'screw_big',
            7: 'screw_small'
        }
        
        # ROS 通訊（保留向後兼容，但優先使用 backend_manager）
        self.communicator = None
        
        # 已知位置（支援中英文鍵名，包含常見變體）
        self.known_locations = {
            # 中文鍵
            "初始位置": [291, -225, 613, 180, 0, -90],
            "收納位置": [402, -134, 354, 180, 0, 90],
            "觀察位置": [402, -134, 354, 180, 0, 90],
            "待機位置": [291, -225, 613, 180, 0, -90],
            # 英文鍵（基本形式）
            "origin": [291, -225, 613, 180, 0, -90],
            "initial": [291, -225, 613, 180, 0, -90],
            "home": [291, -225, 613, 180, 0, -90],
            "storage": [402, -134, 354, 180, 0, 90],
            "observation": [402, -134, 354, 180, 0, 90],
            "standby": [291, -225, 613, 180, 0, -90],
            # 英文變體（帶空格或其他常見形式）
            "initial position": [291, -225, 613, 180, 0, -90],
            "home position": [291, -225, 613, 180, 0, -90],
            "storage area": [402, -134, 354, 180, 0, 90],
            "storage position": [402, -134, 354, 180, 0, 90],
            "observation area": [402, -134, 354, 180, 0, 90],
            "observation position": [402, -134, 354, 180, 0, 90],
            "standby position": [291, -225, 613, 180, 0, -90]
        }
        
        logger.info("Database Controller 已初始化")
        logger.info(f"   規劃器: {self.planner_type}")
        logger.info(f"   執行後端: {self.backend_type}")
    
    def start_session(self) -> str:
        """
        開始新的對話會話
        
        Returns:
            session_id: 新會話的 ID
        """
        self.session_id = self.task_manager.create_session()
        logger.info(f"開始新會話: {self.session_id[:35]}...")
        return self.session_id
    
    def get_current_session(self) -> Optional[str]:
        """
        取得當前會話 ID
        
        Returns:
            session_id or None
        """
        return self.session_id
    
    def execute_gesture_directly(self, gesture_type: str, params: Optional[Dict] = None, session_id: Optional[str] = None) -> Optional[str]:
        """
        直接執行手勢任務（繞過 LLM 解析，提升效率）
        
        用於人類手勢觸發的任務：
        - receive_gesture: 根據夾爪狀態決定放置或拿取
        - pointing_gesture: 雙相機射線檢測 → 抓取 → 交接
        - stop_gesture: 緊急停止
        
        Args:
            gesture_type: 手勢類型（receive_gesture, pointing_gesture, stop_gesture）
            params: 額外參數（例如夾爪狀態、置信度等）
            session_id: 會話 ID（可選）
        
        Returns:
            task_id: 任務 ID，如果失敗則返回 None
        """
        if session_id is None:
            session_id = self.session_id
        
        logger.info(f"\n{'='*60}")
        logger.info(f"直接執行手勢: {gesture_type}")
        if params:
            logger.info(f"   參數: {params}")
        logger.info(f"{'='*60}")
        
        # 構建簡化的 task_json（不需要 LLM 解析）
        # 映射 gesture_type 到正確的 action 名稱
        gesture_action_map = {
            "receive_gesture": "receive手勢處理",
            "pointing_gesture": "pointing手勢處理",
            "stop_gesture": "stop"
        }
        
        action = gesture_action_map.get(gesture_type, f"{gesture_type}處理")
        
        task_json = {
            "action": action,  # receive手勢處理, pointing手勢處理
            "parameters": params or {},
            "source": "gesture",
            "confidence": params.get('confidence', 1.0) if params else 1.0
        }
        
        # 直接存入資料庫（使用正確的參數名稱）
        user_input = f"手勢觸發: {gesture_type}"
        task_id = self.task_manager.save_task(
            user_input=user_input,
            task_json=task_json,
            source="gesture",
            session_id=session_id
        )
        
        logger.info(f"手勢任務已建立: {task_id}")
        logger.info(f"   已跳過 LLM 解析（直接映射）")
        
        return task_id

    def execute_task_from_structured(
        self,
        step: Dict[str, Any],
        session_id: Optional[str] = None,
        source: str = "orchestrator",
    ) -> Union[bool, Dict[str, Any]]:
        """
        Bypass process_user_command / llm_parser entirely.

        Accepts a structured step dict from HRCOrchestrator (already planned by LLM):
            {"action": "pickup",  "target_object": "screw_big"}
            {"action": "place",   "target_object": "screw_big", "target_location": "right_area"}
            {"action": "gripper_open"}
            {"action": "move",    "target_location": "storage"}

        Builds task_json directly → save_task → execute_task.
        No alias matching, no fuzzy NLP.
        """
        if session_id is None:
            session_id = self.session_id

        action = step.get("action", "unknown")
        target_object = step.get("target_object")
        target_location = step.get("target_location")
        direction = step.get("direction")           # for shift / move_relative
        distance = step.get("distance")             # mm; None → validator defaults to 50
        rotation_direction = step.get("rotation_direction")  # for rotate_gripper
        rotation_angle = step.get("rotation_angle")
        parameters = step.get("parameters", {})
        spatial_modifier = step.get("spatial_modifier")
        object_size = step.get("object_size")

        task_json: Dict[str, Any] = {
            "action": action,
            "target_object": target_object,
            "target_location": target_location,
            "parameters": parameters,
            "source": source,
            "confidence": 1.0,
        }
        if direction is not None:
            task_json["direction"] = direction
        if distance is not None:
            task_json["distance"] = distance
        if rotation_direction is not None:
            task_json["rotation_direction"] = rotation_direction
        if rotation_angle is not None:
            task_json["rotation_angle"] = rotation_angle
        # Pickup / handover / place: spatial & size hints (must reach GraspPlannerMixin)
        if spatial_modifier:
            task_json["spatial_modifier"] = spatial_modifier
        if object_size:
            task_json["object_size"] = object_size

        # Build a readable user_input label for DB / telemetry
        parts = [action]
        if spatial_modifier:
            parts.append(str(spatial_modifier))
        if object_size:
            parts.append(str(object_size))
        if target_object:
            parts.append(target_object)
        if target_location:
            parts.append(f"to {target_location}")
        if direction:
            parts.append(f"{direction} {distance or 50}mm")
        user_input_label = " ".join(parts)

        logger.info(f"\nStructured execute: {user_input_label}")

        task_id = self.task_manager.save_task(
            user_input=user_input_label,
            task_json=task_json,
            source=source,
            session_id=session_id,
        )

        return self.execute_task(task_id)

    def _resolve_location(self, location_name: str) -> Optional[List[float]]:
        """
        解析位置名稱到座標。

        1. 精確匹配 known_locations key
        2. 不分大小寫匹配
        3. difflib 模糊匹配（取代原本的 LLM 語義匹配）

        Args:
            location_name: 使用者輸入的位置名稱

        Returns:
            座標列表 [x, y, z, rx, ry, rz] 或 None
        """
        # 1. 精確匹配
        if location_name in self.known_locations:
            return self.known_locations[location_name]

        # 2. 不區分大小寫
        location_lower = location_name.lower()
        for key in self.known_locations:
            if key.lower() == location_lower:
                return self.known_locations[key]

        # 3. difflib 模糊匹配（cutoff=0.5，對應 ~15 個已知位置綽綽有餘）
        import difflib
        matches = difflib.get_close_matches(
            location_lower,
            [k.lower() for k in self.known_locations],
            n=1,
            cutoff=0.5,
        )
        if matches:
            # 找回原始大小寫的 key
            matched_key = next(k for k in self.known_locations if k.lower() == matches[0])
            logger.info(f"   位置模糊匹配: '{location_name}' → '{matched_key}'")
            return self.known_locations[matched_key]

        return None
    
    def initialize_vision(self):
        """初始化視覺系統"""
        if self.vision is None:
            try:
                # imported lazily: needs OpenCV / RealSense / ultralytics (``perception`` extra)
                from ws_racr.perception.object_localizer import ObjectLocalizer
                self.vision = ObjectLocalizer(
                    external_pipeline=self.external_camera_pipeline,
                    external_intrinsics=self.external_camera_intrinsics,
                    camera_name=self.external_camera_name,
                    external_yolo_detector=self.external_yolo_detector,
                    external_top_camera_pipeline=self.external_top_camera_pipeline,
                    external_top_camera_intrinsics=self.external_top_camera_intrinsics
                )
                logger.info("視覺系統已初始化")
            except Exception as e:
                logger.warning(f"視覺系統初始化失敗: {e}")
    
    def initialize_ros(self):
        """初始化 ROS 通訊"""
        if not ROS_AVAILABLE:
            logger.warning("ROS 不可用")
            return False
        
        if self.communicator is None:
            try:
                import rclpy
                if not rclpy.ok():
                    rclpy.init()
                
                # 初始化通訊器
                self.communicator = RobotCommunicator()
                logger.info("ROS 通訊已初始化")
                
                return True
            except Exception as e:
                logger.warning(f"ROS 初始化失敗: {e}")
                return False
        return True
    
    def process_user_command(self, user_input: str, session_id: Optional[str] = None, 
                            memory_context: Optional[str] = None) -> str:
        """
        處理使用者指令的完整流程
        
        流程:
        1. LLM 解析指令 (可選：整合 memory_context)
        2. 儲存任務到 Database (附加 session_id)
        3. 返回 task_id
        
        Args:
            user_input: 使用者的自然語言指令
            session_id: 對話會話 ID (可選，若未提供則使用當前會話)
            memory_context: 記憶系統提供的決策上下文 (可選，用於引導 LLM 決策)
            
        Returns:
            task_id: 任務 ID
        """
        # 如果沒有傳入 session_id，使用當前會話
        if session_id is None:
            session_id = self.session_id
        
        logger.info(f"\n{'='*60}")
        logger.info(f"處理使用者指令: {user_input}")
        if session_id:
            logger.info(f"   會話 ID: {session_id[:35]}...")
        if memory_context:
            logger.info(f"   記憶上下文: {len(memory_context)} 字符")
        logger.info(f"{'='*60}")
        
        # 步驟 1: LLM 解析 (整合 memory_context)
        logger.info("\n步驟 1: LLM 解析指令...")

        # 取得 world context 供 Stage F LLM fallback 使用
        world_context = None
        wsb = getattr(self, 'world_state_builder', None)
        if wsb is not None:
            try:
                ws = wsb.current_state
                visible_objects = [
                    obj_id for obj_id, obj in ws.objects.items()
                    if getattr(obj, 'visibility', 'visible') == 'visible'
                ]
                current_step = getattr(
                    getattr(ws, 'assembly_progress', None), 'current_step', None
                ) or getattr(ws, 'step_class', None)
                robot_status = getattr(getattr(ws, 'robot', None), 'state', None)
                world_context = {
                    'visible_objects': visible_objects,
                    'current_step':    current_step,
                    'robot_status':    robot_status,
                }
            except Exception:
                pass

        # 如果提供了 memory_context，將其附加到指令前
        if memory_context:
            augmented_input = f"""[Previous Experience Context]
{memory_context}

[Current Task]
{user_input}"""
            llm_result = process_user_input(augmented_input, world_context=world_context)
        else:
            llm_result = process_user_input(user_input, world_context=world_context)
        
        if not llm_result.get("success"):
            logger.error(f"LLM 解析失敗: {llm_result.get('error')}")
            return None
        
        # 步驟 2: 儲存到 Database
        logger.info("\n步驟 2: 儲存任務到資料庫...")
        task_id = self.task_manager.save_task(user_input, llm_result, session_id=session_id)
        
        logger.info(f"\n任務已建立: {task_id}")
        logger.info(f"   動作類型: {llm_result.get('action')}")
        if llm_result.get('target_object'):
            logger.info(f"   目標物件: {llm_result.get('target_object')}")
        if llm_result.get('target_location'):
            logger.info(f"   目標位置: {llm_result.get('target_location')}")
        logger.info(f"   信心度: {llm_result.get('confidence', 0):.2f}")
        
        return task_id
    
    def execute_task(self, task_id: str) -> Union[bool, Dict[str, Any]]:
        """
        執行任務 - 新架構：檢索→規劃→驗證→執行
        
        流程:
        1. 從 Database 讀取任務 (task_json)
        2. 【檢索】從歷史任務檢索相似的成功計畫
        3. 【規劃】使用 MoveItPlanner 生成執行計畫 (plan_json)
        4. 【驗證】安全檢查 (速度/力矩/Z高度/禁入區/IO序)
        5. 【執行】交給 ExecutionBackendManager
        6. 更新任務狀態 + telemetry
        
        Args:
            task_id: 任務 ID
            
        Returns:
            Union[bool, Dict[str, Any]]:
                True/False — 執行成功或失敗
                {'policy_wait': True, 'reason': str} — LLM 判斷需等待
                {'policy_ask_human': True, 'question': str, 'reason': str} — LLM 需澄清
                {'policy_answer': True, 'answer_text': str, 'reason': str} — LLM 判斷為查詢
                {'occlusion_wait': True, 'explanation': str} — 偵測到遮擋
                {'type': 'occlusion_removal_sequence', ...} — 需要拆解為多任務
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"執行任務: {task_id}")
        logger.info(f"{'='*60}")
        
        # 步驟 1: 從 Database 讀取任務
        logger.info("\n步驟 1: 從資料庫讀取任務...")
        task = self.task_manager.get_task(task_id)
        
        if not task:
            logger.error(f"找不到任務: {task_id}")
            return False
        
        logger.info(f"   使用者指令: {task['user_input']}")
        logger.info(f"   任務類型: {task['task_type']}")
        
        # 更新狀態為執行中
        self.task_manager.update_task_status(task_id, "executing")
        
        task_json = task['task_json']
        action = task_json.get('action')
        
        # 特殊處理：步驟更新指令
        # 如果檢測到是步驟更新意圖（例如「我們目前在步驟三」），使用voice_query_handler處理
        user_input = task['user_input']
        if self._is_query_intent(user_input, action):
            logger.info(f"\n檢測到查詢意圖，使用語音查詢處理器...")
            return self._handle_query_via_voice_query(user_input, task_id)
        if self._is_step_update_intent(user_input, action, task_json):
            logger.info(f"\n檢測到步驟更新意圖，使用語音查詢處理器...")
            result = self._handle_step_update_via_voice_query(user_input, task_id)
            return result
        
        try:
            # === 步驟 1.5: 驗證 Action ===
            logger.info(f"\n驗證動作...")
            is_valid, validated_json, error_msg = self.action_validator.validate_and_correct(task_json)

            _NEEDS_TARGET = {'pickup', 'grab', 'grasp', 'handover', 'give', 'pass'}
            if not is_valid:
                # LLM-B removed: ask human immediately when target is missing.
                if action in _NEEDS_TARGET and not task_json.get('target_object'):
                    question = '請問您需要我拿哪個物件？'
                    logger.warning(f"   '{action}' 缺少目標物件 → ASK_HUMAN: {question}")
                    self.task_manager.update_task_status(
                        task_id, "waiting_human", error_message=question
                    )
                    self._record_interaction(user_input, 'ask', question, task_id=task_id)
                    return {'policy_ask_human': True, 'question': question, 'reason': 'missing target_object'}
                else:
                    logger.error(f"動作驗證失敗: {error_msg}")
                    self.task_manager.update_task_status(
                        task_id, "failed",
                        error_message=f"Invalid action: {error_msg}"
                    )
                    return False

            # 如果有修正，更新 task_json
            if is_valid and validated_json != task_json:
                task_json = validated_json
                action = task_json.get('action')
                logger.info(f"動作已驗證並修正")
            elif is_valid:
                logger.info(f"動作驗證通過: {action}")
            
            # === 步驟 2: Plan RAG 已移至 HRCOrchestrator（MULTI 計畫序列層級）===
            reference_plan = None

            # === 步驟 2.5: World-State 可行性檢查 ===
            logger.info(f"\n步驟 2.5: World-State 可行性檢查...")
            ws_feasible, ws_warning = self._check_world_state_feasibility(task_json)
            if ws_warning:
                logger.info(f"   {ws_warning}")
            if ws_feasible:
                logger.info(f"   World-State 可行性通過")
            elif (
                action in ("pickup", "grab", "grasp", "handover", "give", "pass")
                and task_json.get("target_object")
            ):
                # Aligns with thesis Layer 2 narrative on real hardware: do not enter motion
                # planning when the manipulation target is absent from WorldState
                # (RAG-layer REJECT only runs when a multi_plan is retrieved).
                logger.error("   World-State 硬性中止：目標物無可見實例，不進入路徑規劃。")
                self.task_manager.update_task_status(
                    task_id,
                    "failed",
                    error_message=(ws_warning or "World-State: target not available").strip(),
                )
                return _feasibility_abort_speak_zh(task_json)

            # === 步驟 3: 規劃執行計畫 ===

            logger.info(f"\n 步驟 3: 生成執行計畫...")
            plan_json = self._generate_plan(task_json)

            if not plan_json:
                logger.error(f"規劃失敗")
                self.task_manager.update_task_status(
                    task_id, "failed",
                    error_message="規劃失敗"
                )
                return False
            
            # 將 plan_json 中的 object_id 寫回 task_json（用於失敗檢測）
            if 'metadata' in plan_json and 'object_id' in plan_json['metadata']:
                specific_object_id = plan_json['metadata']['object_id']
                if specific_object_id:
                    task_json['target_object_id'] = specific_object_id
                    # 也更新到資料庫的 task
                    self.task_manager.update_task_field(task_id, 'task_json', task_json)
                    logger.info(f"   記錄夾取目標 ID: {specific_object_id}")
            
            # 檢查是否需要處理遮擋
            if plan_json.get('type') == 'occlusion_wait':
                logger.warning(f"\n偵測到遮擋，需要人類協助")
                logger.info(f"   說明: {plan_json.get('explanation')}")
                logger.info(f"   目標物體: {plan_json.get('target_object')}")
                
                # 更新任務狀態為等待中
                self.task_manager.update_task_status(
                    task_id, "waiting_human",
                    error_message=plan_json.get('explanation')
                )
                
                # 返回特殊標記，讓 HRCSystem 觸發 TTS 語音通知
                return {'occlusion_wait': True, 'explanation': plan_json.get('explanation')}
            
            # 檢查是否為遮擋物移除序列（需要拆成多個任務）
            if plan_json.get('type') == 'occlusion_removal_sequence':
                logger.info(f"\n檢測到遮擋物移除序列，返回給上層處理")
                # 更新任務狀態為等待中
                self.task_manager.update_task_status(
                    task_id, "waiting_occlusion_removal",
                    error_message="等待移除遮擋物"
                )
                # 返回序列給 main_hrc_system 用 task queue 執行
                return plan_json
            
            # 儲存 plan_json 到資料庫
            self.task_manager.update_task_plan(task_id, plan_json)
            logger.info(f"   計畫已儲存: {len(plan_json.get('waypoints', []))} 個路徑點")
            
            # === 步驟 3.5: 生成解釋 (Explanation Generation) ===
            explanation = self._generate_explanation(task_json, plan_json)
            if explanation:
                logger.info(f"\n執行說明: {explanation}")
                # 儲存解釋到任務中
                task['explanation'] = explanation
            
            # === 步驟 4: 驗證計畫 ===
            logger.info(f"\n 步驟 4: 驗證執行計畫...")
            is_valid, errors, warnings = self.plan_validator.validate(plan_json)
            
            if not is_valid:
                logger.error(f"計畫驗證失敗:")
                for err in errors:
                    logger.info(f"   - {err}")
                
                self.task_manager.update_task_status(
                    task_id, "failed",
                    error_message=f"計畫驗證失敗: {'; '.join(errors)}"
                )
                return False
            
            # === 步驟 5: 執行計畫 ===
            logger.info(f"\n 步驟 5: 執行計畫...")
            
            waypoints = plan_json.get('waypoints', [])
            
            # 在執行前記錄當前位置（用於 return_to_last 功能）
            # 這是機器人開始執行任務前的實際位置
            current_position_before_execution = None
            try:
                if hasattr(self.backend_manager, 'backend') and \
                   hasattr(self.backend_manager.backend, 'communicator'):
                    comm = self.backend_manager.backend.communicator
                    
                    # 直接讀取快取的 robot_state。
                    # rclpy.spin_once() 不得在非 ROS 主線程呼叫（worker thread 執行此處），
                    # 否則會造成 segfault / undefined behaviour。
                    # robot_state 由 ROS callback 自動更新，無需在此 spin。
                    if hasattr(comm, 'robot_state') and comm.robot_state.get('position'):
                        robot_pos = comm.robot_state['position']
                        # 檢查是否為有效位置（非全零）
                        if any(abs(x) > 0.1 for x in robot_pos[:3]):
                            current_position_before_execution = robot_pos.copy()
                            logger.info(f"   記錄執行前位置: [{current_position_before_execution[0]:.1f}, {current_position_before_execution[1]:.1f}, {current_position_before_execution[2]:.1f}]")
                        else:
                            logger.warning(f"   robot_state 位置全為零: {robot_pos}")
                    else:
                        logger.warning(f"   robot_state 不存在或沒有 position")
            except Exception as e:
                logger.warning(f"   無法記錄執行前位置: {e}")
            
            # 轉換為 Waypoint 物件（處理兩階段任務的 checkpoint）
            waypoint_objects = []
            
            for i, wp_dict in enumerate(waypoints):
                # 如果已經是 Waypoint 物件，直接使用
                if isinstance(wp_dict, Waypoint):
                    waypoint_objects.append(wp_dict)
                    continue
                
                # 檢查是否為 checkpoint（僅記錄，不在此執行）
                if wp_dict.get('action') == 'checkpoint':
                    # 保留 checkpoint 信息，執行時再處理
                    waypoint_objects.append(wp_dict)  # 保留原始 dict
                    continue
                
                # 過濾掉非 Waypoint 的額外欄位
                wp_clean = {}
                valid_keys = {
                    'action', 'position', 'velocity', 'acc_time', 'blend',
                    'wait_time', 'timeout', 'constraints', 'planner_id',
                    'planning_time', 'io_before', 'io_after', 'description', 'metadata'
                }
                
                for key, value in wp_dict.items():
                    if key in valid_keys:
                        wp_clean[key] = value
                
                # 創建 Waypoint 物件
                try:
                    waypoint_objects.append(Waypoint(**wp_clean))
                except Exception as e:
                    logger.warning(f"   路徑點 {i} 創建失敗: {e}")
                    logger.info(f"      原始數據: {wp_dict}")
                    # 使用舊格式轉換作為備用
                    waypoint_objects.append(Waypoint.from_legacy_waypoint(wp_dict))
            
            # 執行
            execution_start = datetime.now()
            # 檢查是否有 checkpoint（用於日誌輸出）
            has_checkpoint = any(
                isinstance(wp, dict) and wp.get('action') == 'checkpoint' 
                for wp in waypoint_objects
            )
            if has_checkpoint:
                logger.info(f"\n▶ 開始執行兩階段任務 (含實時人手檢測)")
            result = self._execute_waypoints_v2(waypoint_objects)
            execution_end = datetime.now()
            
            # 關鍵：確保機器人真正完成所有移動
            logger.info(f"\n等待機器人完成所有動作...")
            if hasattr(self.backend_manager, 'backend') and \
               hasattr(self.backend_manager.backend, 'communicator'):
                comm = self.backend_manager.backend.communicator
                
                # 等待機器人完成移動（使用 QueueTag 機制）
                if comm is not None and comm.wait_for_movement_complete(timeout=60.0):
                    logger.info(f"機器人所有動作已完成")
                else:
                    logger.warning(f"機器人動作等待超時（60秒）")
            
            # NOTE: spin_once() 不得在非 ROS 主線程呼叫（此處為 worker thread）。
            # feedback callback 由 ROS executor 在主線程負責驅動，此處不需要 spin。
            
            # === 步驟 6: 更新任務狀態 + telemetry ===
            execution_duration = (execution_end - execution_start).total_seconds()
            
            # 記錄位置（用於 return_to_last 功能）
            # 策略：記錄任務開始執行前機器人的實際位置
            final_position = None
            
            # 優先使用執行前記錄的當前位置（這是最準確的）
            if current_position_before_execution:
                final_position = current_position_before_execution
                logger.info(f"   使用執行前記錄的實際位置")
            
            # 備用方案：從第一個 move waypoint 獲取（不太準確）
            elif waypoint_objects:
                for wp in waypoint_objects:
                    if hasattr(wp, 'action') and wp.action == 'move' and hasattr(wp, 'position'):
                        final_position = wp.position
                        logger.info(f"   使用第一個移動位置 (waypoint，備用方案)")
                        break
            
            telemetry = {
                "execution_duration": execution_duration,
                "waypoint_count": len(waypoint_objects),
                "planner_type": self.planner_type,
                "backend_type": self.backend_type,
                "warnings": warnings,
                "completed_at": execution_end.isoformat(),
                "last_position": final_position  # 記錄最終位置（絕對座標）
            }
            
            # 如果 result 是 Dict（來自 TMScriptBackend 的執行報告）
            if isinstance(result, dict):
                telemetry["script_execution"] = {
                    "command_count": result.get("command_count", 0),
                    "execution_time": result.get("execution_time", 0),
                    "start_time": result.get("start_time"),
                    "end_time": result.get("end_time")
                }
                success = result.get("success", False)
                error_msg = result.get("error")
            else:
                # 舊格式：bool 回傳值
                success = result
                error_msg = None if result else "執行失敗"
            
            if success:
                self.task_manager.update_task_status(
                    task_id, "completed", 
                    result={"success": True},
                    telemetry=telemetry
                )
                logger.info(f"\n任務完成: {task_id}")
                logger.info(f"   執行時間: {execution_duration:.2f} 秒")
                logger.info(f"   路徑點數: {len(waypoint_objects)}")
                
                # === 更新 Robot IO 真值（優先級最高）===
                self._update_robot_io_truth()
                
                # === 更新 Facts Store（成功） ===
                self.facts.update_from_skill_result(
                    skill=action,
                    success=True,
                    target_object=task_json.get('target_object'),
                    target_location=task_json.get('target_zone') or task_json.get('target_location')
                )
                # 記錄互動（InteractionLog）
                self._record_interaction(user_input, 'execute', f'{action} → 成功',
                                         task_id=task_id, success=True)
            else:
                telemetry["error"] = error_msg or "執行失敗"
                self.task_manager.update_task_status(
                    task_id, "failed",
                    error_message=error_msg or "執行失敗",
                    telemetry=telemetry
                )
                logger.error(f"\n任務失敗: {task_id}")
                
                # === 更新 Facts Store（失敗） ===
                self.facts.update_from_skill_result(
                    skill=action,
                    success=False,
                    target_object=task_json.get('target_object'),
                    failure_reason=error_msg
                )
                # 記錄互動（InteractionLog）
                self._record_interaction(user_input, 'execute',
                                         f'{action} → 失敗: {error_msg or "unknown"}',
                                         task_id=task_id, success=False)
                if error_msg:
                    logger.info(f"   錯誤: {error_msg}")
            
            return success
            
        except Exception as e:
            logger.error(f"執行異常: {e}")
            import traceback
            traceback.print_exc()
            
            self.task_manager.update_task_status(
                task_id, "failed",
                error_message=str(e),
                telemetry={"error": str(e), "traceback": traceback.format_exc()}
            )
            return False
    
    def _is_step_update_intent(self, user_input: str, action: str, task_json: Dict) -> bool:
        """
        判斷是否為步驟更新意圖
        
        檢測條件：
        1. 包含「步驟」或「第X步」關鍵字
        2. 包含「目前」、「現在」、「在」等狀態詞
        3. 或者action是move且target_location包含「步驟」
        
        Args:
            user_input: 原始使用者輸入
            action: LLM解析的動作類型
            task_json: 任務JSON
            
        Returns:
            bool: 是否為步驟更新意圖
        """
        import re
        
        # 關鍵字模式
        step_update_patterns = [
            r'我們.*目前.*在.*步驟',
            r'我們.*現在.*在.*步驟',
            r'我們.*在.*第.+步',
            r'目前.*在.*步驟',
            r'現在.*在.*步驟',
            r'目前.*第.+步',
            r'現在.*第.+步',
            r'跳到.*第.+步',
            r'切換.*到.*第.+步',
            r'更新.*步驟',
        ]
        
        # 檢查pattern匹配
        for pattern in step_update_patterns:
            if re.search(pattern, user_input):
                return True
        
        # 如果action是move且target_location包含「步驟」
        if action == 'move':
            target_location = task_json.get('target_location', '')
            if '步驟' in target_location or '第' in target_location and '步' in target_location:
                return True
        
        return False
    
    def _record_interaction(
        self,
        user_utterance: str,
        response_type: str,
        robot_response: str,
        task_id: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> None:
        """將一次互動寫入 MemorySystem.interaction_log。安靖失敗。"""
        if not self.memory_system:
            return
        try:
            # 取得當前 step context
            step_context = 'unknown'
            if self.world_state_builder:
                ws = self.world_state_builder.get_current_state()
                step_context = getattr(ws, 'step_class', 'unknown') or 'unknown'
            self.memory_system.record_interaction(
                user_utterance=user_utterance,
                response_type=response_type,
                robot_response=robot_response,
                step_context=step_context,
                task_id=task_id,
                success=success,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).debug(f'_record_interaction failed: {e}')

    def _is_query_intent(self, user_input: str, action: str) -> bool:
        """判斷是否為查詢意圖（庫存/定位/步驟進度等）。

        兩層識別：
        1. 高精確問句型 regex 模式（覆蓋所有 VOICE_COMMANDS_EXAMPLES 場景）
        2. 備援：parser 未識別出已知機器人動作，且含問句標誌詞 → 視為查詢
        """
        import re
        query_patterns = [
            # 庫存查詢
            r'桌上.*有.*哪些',
            r'還有.*哪些.*物品',
            r'還有.*什麼.*東西',
            r'剩下.*哪些',
            r'列出.*物品',
            # 物體定位
            r'(.+)在哪裡',
            r'(.+)在哪[？?]?$',
            r'(.+)的位置在',
            r'找(.+)',
            # 數量查詢
            r'有幾個(.+)',
            r'(.+)還有幾個',
            r'還有多少(.+)',
            r'還剩多少(.+)',
            r'(.+)還剩幾個',
            # 存在性查詢
            r'(.+)還在嗎',
            r'(.+)不見了嗎',
            r'看得到(.+)嗎',
            # 步驟進度查詢
            r'現在.*第幾步',
            r'目前.*第幾步',
            r'做到.*哪了',
            r'進度.*如何',
            r'完成.*多少',
            r'下一步.*什麼',
            r'接下來.*做什麼',
            r'下個步驟',
            # 材料需求查詢
            r'第(.+)步.*需要',
            r'第(.+)步.*什麼',
            r'下一步.*需要',
            r'還缺.*什麼',
            r'我漏了.*什麼',
            r'還需要.*什麼',
            r'這步.*需要.*什麼',
            # 空間關係查詢
            r'(.+)附近有.*什麼',
            r'(.+)旁邊有',
            r'(.+)在.*哪個方向',
            r'(.+)在.*哪邊',
            r'(.+)相對.*在哪',
        ]
        for pattern in query_patterns:
            if re.search(pattern, user_input):
                return True
        # 備援：parser 未識別出有效機器人動作 + 含問句標誌詞 → 判定為查詢
        if action not in _KNOWN_ROBOT_ACTIONS:
            if any(marker in user_input for marker in _QUESTION_MARKERS):
                return True
        return False

    def _handle_query_via_voice_query(self, user_input: str, task_id: str) -> dict:
        """使用 voice_query_handler 回答查詢意圖。"""
        try:
            if not hasattr(self, 'voice_query_handler') or self.voice_query_handler is None:
                logger.warning(f"VoiceQueryHandler未初始化，無法回答查詢")
                self.task_manager.update_task_status(
                    task_id, "failed", error_message="VoiceQueryHandler未初始化"
                )
                return False
            result = self.voice_query_handler.parse_and_execute(user_input)
            answer_text = result.get('response_text', '抱歉，我無法回答這個問題')
            if result.get('success'):
                logger.info(f"   查詢成功 ({result.get('query_type')}): {answer_text}")
            else:
                logger.warning(f"   VoiceQueryHandler 無法識別查詢類型")
            self.task_manager.update_task_status(task_id, "completed")
            self._record_interaction(user_input, 'answer', answer_text, task_id=task_id)
            return {'policy_answer': True, 'answer_text': answer_text}
        except Exception as e:
            logger.warning(f"查詢處理異常: {e}")
            self.task_manager.update_task_status(
                task_id, "failed", error_message=str(e)
            )
            return False

    def _handle_step_update_via_voice_query(self, user_input: str, task_id: str) -> bool:
        """
        使用voice_query_handler處理步驟更新
        
        Args:
            user_input: 使用者原始輸入
            task_id: 任務ID
            
        Returns:
            bool: 執行是否成功
        """
        try:
            # 確保voice_query_handler已初始化
            if not hasattr(self, 'voice_query_handler') or self.voice_query_handler is None:
                logger.warning(f"VoiceQueryHandler未初始化")
                self.task_manager.update_task_status(
                    task_id, "failed",
                    error_message="語音查詢處理器未初始化"
                )
                return False
            
            # 調用voice_query_handler處理
            result = self.voice_query_handler.parse_and_execute(user_input)
            
            if result.get('success'):
                logger.info(f"步驟更新成功: {result.get('response_text')}")
                self.task_manager.update_task_status(task_id, "completed")
                return True
            else:
                logger.error(f"步驟更新失敗: {result.get('response_text')}")
                self.task_manager.update_task_status(
                    task_id, "failed",
                    error_message=result.get('response_text')
                )
                return False
                
        except Exception as e:
            logger.error(f"處理步驟更新時出錯: {e}")
            import traceback
            traceback.print_exc()
            self.task_manager.update_task_status(
                task_id, "failed",
                error_message=f"處理步驟更新時出錯: {str(e)}"
            )
            return False
    
    def _should_skip_rag(self, task_json: Dict) -> bool:
        """
        判斷是否跳過 RAG 檢索
        
        簡單指令滿足以下條件時直接執行，不走 RAG：
        1. 手勢處理（已是明確動作序列）
        2. 基礎動作 + 明確參數（pickup/place/handover/move）
        3. 夾爪控制（open_gripper/close_gripper）
        
        Args:
            task_json: 任務 JSON
            
        Returns:
            bool: True 跳過 RAG，False 使用 RAG
        """
        action = task_json.get('action', '')
        
        # 1. 手勢處理 - 直接執行
        if action in ['receive手勢處理', 'pointing手勢處理']:
            return True
        
        # 2. 夾爪控制 - 直接執行
        if action in ['open_gripper', 'close_gripper', 'gripper_open', 'gripper_close']:
            return True
        
        # 3. 基礎動作 + 明確參數
        simple_actions = ['pickup', 'place', 'handover', 'move']
        if action in simple_actions:
            # pickup: 需要 target_object
            if action == 'pickup':
                return bool(task_json.get('target_object'))
            
            # place: 需要 target_position 或 target_location
            elif action == 'place':
                return bool(task_json.get('target_position') or task_json.get('target_location'))
            
            # handover: 標準流程，直接執行
            elif action == 'handover':
                return True
            
            # move: 需要 target_position
            elif action == 'move':
                return bool(task_json.get('target_position'))
        
        # 其他情況：使用 RAG（複雜任務、模糊指令、需要推理）
        return False

    def _check_world_state_feasibility(self, task_json: Dict) -> Tuple[bool, str]:
        """
        執行前的 World-State 可行性檢查。

        規則（execute_task 對「取物／交接目標缺失」會硬性中止）：
          - pickup / grab / grasp / handover / give / pass（且有 target_object）:
              target_object 在 world_state 中須至少一個 **visible** 實例。
          - place / put / release:
              robot.gripper_state 應為 'closed'（有夾到東西才可放）。
          - handover 開始時夾爪常為 open（要先夾再交），不沿用 place 的「必須已 closed」判斷。
          - gripper_open  : gripper 若已 open，發出 soft 警告。
          - gripper_close : gripper 若已 close，發出 soft 警告。

        Returns:
            (feasible, warning_message)
            feasible=False 時，execute_task 對 pickup/grab/grasp/handover/give/pass
            在「目標物無可見實例」時會中止；place 系仍依夾爪狀態；gripper 重複指令另處理。
        """
        if not (hasattr(self, 'world_state_builder') and self.world_state_builder):
            return True, ""  # 無法取得 world state，跳過檢查

        try:
            ws = self.world_state_builder.get_current_state()
        except Exception:
            return True, ""

        action = task_json.get('action', '')

        def _visible_targets(obj_type: str) -> list:
            return [
                o for o in ws.get_objects_by_type(obj_type)
                if getattr(o, "visibility", "") == "visible"
            ]

        # ── 取物／交接：目標類型須在場景中有可見實例（與論文 Layer 2「缺件不可做」對齊）──
        if action in ('pickup', 'grab', 'grasp', 'handover', 'give', 'pass'):
            target = task_json.get('target_object')
            if target and not _visible_targets(target):
                return False, (
                    f"⚠️ World-State: \"{target}\" 目前無可見實例（未偵測或不在視野），"
                    f"不執行取物／交接。"
                )

        # ── place / release：夾爪應關閉（手上要有東西才能放）────────────────
        if action in ('place', 'put', 'release'):
            gripper = ws.robot.gripper_state
            if gripper == 'open':
                return False, (
                    "⚠️ World-State 警告: 夾爪目前開啟（可能未持有物件），"
                    "place/release 可能無效，仍嘗試執行。"
                )

        # ── gripper_open：若已 open 無需再開 ────────────────────────
        elif action in ('gripper_open', 'open_gripper'):
            if ws.robot.gripper_state == 'open':
                return False, "⚠️ World-State 警告: 夾爪已開啟，跳過重複 open。"

        # ── gripper_close：若已 close 無需再關 ──────────────────────
        elif action in ('gripper_close', 'close_gripper'):
            if ws.robot.gripper_state == 'closed':
                return False, "⚠️ World-State 警告: 夾爪已關閉，跳過重複 close。"

        return True, ""

    def _generate_explanation(self, task_json: Dict, plan_json: Dict, reference_plan: Optional[Dict] = None) -> str:
        """
        生成執行計畫的說明 (Explanation Generation)
        
        使用 LLM 將技術性的 plan_json 轉換為人類可讀的解釋
        
        Args:
            task_json: 任務語義描述
            plan_json: 執行計畫
            reference_plan: 參考計畫（如果有）
            
        Returns:
            str: 人類可讀的解釋
        """
        try:
            action = task_json.get('action')
            target_object = task_json.get('target_object', '')
            target_location = task_json.get('target_location', '')
            waypoints = plan_json.get('waypoints', [])
            
            # 簡單動作直接返回預設解釋
            if action in ['gripper_open', 'gripper_close', 'stop']:
                explanations = {
                    'gripper_open': "打開夾爪",
                    'gripper_close': "關閉夾爪",
                    'stop': "停止所有動作"
                }
                return explanations.get(action, f"執行 {action}")
            
            # 複雜動作生成詳細解釋
            explanation_parts = []
            
            # 1. 主要動作
            if action == 'pickup':
                explanation_parts.append(f"拿取 {target_object}")
            elif action == 'place':
                explanation_parts.append(f"將物品放置到 {target_location}")
            elif action == 'handover':
                explanation_parts.append(f"將 {target_object} 遞交給您")
            elif action == 'move':
                explanation_parts.append(f"移動到 {target_location}")
            else:
                explanation_parts.append(f"執行 {action}")
            
            # 2. 路徑點數量
            if len(waypoints) > 0:
                explanation_parts.append(f"，分 {len(waypoints)} 步")
            
            # 3. 參考來源（如果有）
            if reference_plan and 'metadata' in reference_plan:
                citations = reference_plan['metadata'].get('citations')
                if citations:
                    similarity = citations.get('similarity_score', 0)
                    if similarity > 0.8:
                        explanation_parts.append("（參考過去的成功經驗）")
            
            return ''.join(explanation_parts)
            
        except Exception as e:
            logger.warning(f"生成解釋失敗: {e}")
            return f"執行 {task_json.get('action', '未知動作')}"
    
    def _generate_plan(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """
        生成執行計畫（直接規劃方法）
        
        Args:
            task_json: 任務的語意描述
            reference_plan: 參考計畫 (來自歷史任務)
        
        Returns:
            plan_json: 執行計畫
        """
        action = task_json.get('action')
        
        # 根據動作類型生成計畫
        if action in ['pickup', 'grab', 'grasp']:
            return self._plan_pickup(task_json, reference_plan)
        elif action in ['place', 'put', 'release']:
            return self._plan_place(task_json, reference_plan)
        elif action in ['move', 'navigate', 'go']:
            return self._plan_move(task_json, reference_plan)
        elif action in ['move_relative', 'adjust_position', 'nudge', 'shift']:
            return self._plan_move_relative(task_json, reference_plan)
        elif action in ['return_to_last', 'go_back']:
            return self._plan_return_to_last(task_json, reference_plan)
        elif action in ['return_to_origin', 'go_home']:
            return self._plan_return_to_origin(task_json, reference_plan)
        elif action in ['rotate_gripper', 'twist', 'turn']:
            return self._plan_rotate_gripper(task_json, reference_plan)
        elif action in ['gripper_open', 'open_gripper']:
            return self._plan_gripper_open(task_json, reference_plan)
        elif action in ['gripper_close', 'close_gripper']:
            return self._plan_gripper_close(task_json, reference_plan)
        elif action in ['handover', 'hand_over', 'transfer']:
            return self._plan_handover(task_json, reference_plan)
        elif action in ['prepare', 'prepare_tool', 'get_ready']:
            return self._plan_prepare(task_json, reference_plan)
        elif action in ['stop', 'halt', 'emergency_stop']:
            return self._plan_emergency_stop(task_json, reference_plan)
        elif action in ['retreat', 'back_off', 'move_away']:
            return self._plan_retreat(task_json, reference_plan)
        elif action in ['move_occluder', 'remove_obstacle']:
            return self._plan_move_occluder(task_json, reference_plan)
        # === 手勢直接映射（不需要 RAG） ===
        elif action in ['receive手勢處理']:
            return self._plan_receive_gesture(task_json, reference_plan)
        elif action in ['pointing手勢處理']:
            return self._plan_pointing_gesture(task_json, reference_plan)
        else:
            logger.warning(f"不支援的動作類型: {action}")
            return None
    
    def _update_robot_io_truth(self):
        """
        從機器人 IO 更新真值到 Facts Store（Robot IO Truth Priority）
        
        讀取 TM5 DO 狀態推測 gripper 狀態，並更新 holding_object
        """
        # 檢查是否有 ROS backend
        if not hasattr(self.backend_manager, 'backend') or \
           not hasattr(self.backend_manager.backend, 'communicator'):
            return
        
        try:
            comm = self.backend_manager.backend.communicator
            robot_state = comm.robot_state
            
            # === 1. 更新 gripper 狀態（從 DO 推測）===
            gripper_closed = robot_state.get('gripper_closed', False)
            
            if gripper_closed:
                self.facts.gripper_state = "closed"
            else:
                self.facts.gripper_state = "open"
                # 夾爪打開 → 清空 holding
                if self.facts.holding_object:
                    logger.info(f"   夾爪打開，釋放物體: {self.facts.holding_object}")
                    self.facts.holding_object = None
            
            # === 2. 更新 holding_object（邏輯推理）===
            # 規則：
            # - gripper closed + 最近成功 pickup → holding
            # - gripper open → not holding
            
            if gripper_closed and self.facts.last_skill == "pickup" and self.facts.last_skill_success:
                # 檢查是否在 pickup 後 5 秒內
                import time
                if time.time() - self.facts.last_skill_time < 5.0:
                    # 從最近的 action 推斷 holding_object
                    if self.facts.recent_actions:
                        last_action = self.facts.recent_actions[-1]
                        target_obj = last_action.get('target_object')
                        if target_obj and not self.facts.holding_object:
                            self.facts.holding_object = target_obj
                            logger.info(f"   推測握持物體: {target_obj}")
            
        except Exception as e:
            logger.warning(f"   無法更新 Robot IO 真值: {e}")

