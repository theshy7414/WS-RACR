"""
GraspPlannerMixin — Waypoint generation for all supported robot skills.

Extracted from RobotController to reduce per-query context size.
All methods rely on self.* attributes initialized by RobotController.__init__().
Do NOT instantiate this class directly.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, List, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ws_racr.execution.backends.waypoint_schema import Waypoint


class GraspPlannerMixin:
    """
    Mixin providing _plan_* waypoint-generation methods for RobotController.

    Included methods:
      - _plan_pickup / _plan_place / _plan_move / _plan_move_relative
      - _plan_return_to_last / _plan_return_to_origin
      - _plan_rotate_gripper / _plan_gripper_open / _plan_gripper_close
      - _plan_handover / _plan_prepare / _plan_emergency_stop / _plan_retreat
      - _plan_move_occluder / _plan_receive_gesture / _plan_pointing_gesture
      - _calculate_grasp_angle_from_mask / _transform_angle_to_robot_frame
      - _detect_objects_in_hand
      - _create_standby_waypoint / _check_gripper_state
    """

    def _plan_pickup(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃抓取任務 - 重構版：使用公共方法減少重複代碼"""
        target_object = task_json.get('target_object')
        if not target_object:
            logger.error("缺少目標物件")
            return None
        
        # 提取空間描述詞和尺寸描述
        spatial_modifier = task_json.get('spatial_modifier')
        object_size = task_json.get('object_size')
        
        # 解析完整物件名稱（如 screw + 大 → screw_big）
        resolved_object = self._resolve_object_with_size(target_object, object_size)
        
        if spatial_modifier and object_size:
            logger.info(f"   拾取目標: {spatial_modifier} {object_size} {target_object}")
        elif spatial_modifier:
            logger.info(f"   拾取目標: {spatial_modifier} {target_object}")
        elif object_size:
            logger.info(f"   拾取目標: {object_size} {target_object}")
        else:
            logger.info(f"   拾取目標: {target_object}")
        
        # 使用公共方法檢測物體（傳遞尺寸資訊）
        object_info = self._detect_object_with_d435i(resolved_object, spatial_modifier, object_size)
        
        # 檢查是否偵測到遮擋 → 返回序列化計畫給上層
        if isinstance(object_info, dict) and object_info.get('occlusion_detected'):
            logger.info("   偵測到物體遮擋")
            
            occluder_id = object_info.get('occluder_object')
            occluder_name = object_info.get('occluder_name', occluder_id)
            
            # 判斷是否需要語音確認（platform/bracket/unknown）
            needs_confirmation = object_info.get('needs_voice_confirmation', False)
            
            if needs_confirmation:
                # 需要人工介入，返回給上層處理
                logger.warning("   需要人工協助移除遮擋物")
                return object_info
            
            # 自動移除：返回序列化任務計畫（讓 main 用 task queue 執行）
            logger.info(f"   生成遮擋物移除計畫: {occluder_name}")
            
            return {
                'type': 'occlusion_removal_sequence',
                'subtasks': [
                    {
                        'action': 'pickup',
                        'target_object': occluder_id,
                        'target_object_zh': occluder_name,
                        'description': f'移除遮擋物: {occluder_name}'
                    },
                    {
                        'action': 'place',
                        'target_location': '放置區',
                        'target_location_zh': '放置區',
                        'description': f'放置遮擋物到放置區'
                    },
                    {
                        'action': 'pickup',
                        'target_object': resolved_object,
                        'target_object_zh': task_json.get('target_object'),
                        'spatial_modifier': spatial_modifier,
                        'object_size': object_size,
                        'description': f'抓取原始目標: {task_json.get("target_object")}'
                    }
                ],
                'original_task': task_json,
                'occluder_info': {
                    'id': occluder_id,
                    'name': occluder_name
                }
            }
        
        if not object_info:
            return None
        
        object_position = object_info['position']
        
        # 提取具體的物體 ID（如果有）
        specific_object_id = object_info['detection'].get('object_id')  # e.g., 'cylinder_3'
        if specific_object_id:
            logger.info(f"   確定夾取目標: {specific_object_id}")
        
        # 使用 MoveIt2 規劃
        waypoints = self.moveit_planner.plan_pickup_task(
            object_position=object_position,
            velocity=100
        )
        
        # 記錄關鍵位置
        key_position = None
        for wp in waypoints:
            if hasattr(wp, 'action') and wp.action == 'move' and hasattr(wp, 'position'):
                key_position = wp.position
                break
        
        # 使用公共方法創建 plan_json（並傳遞具體的 object_id）
        plan_json = self._create_plan_json(
            waypoints=waypoints,
            task_type="pickup",
            object_position=object_position,
            object_name=object_info['detection']['class_name'],
            object_id=specific_object_id,  # 傳遞具體的 ID
            key_position=key_position,
            reference_task=reference_plan.get('metadata', {}).get('task_id') if reference_plan else None
        )
        
        return plan_json
    
    def _plan_place(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃放置任務"""
        target_location = task_json.get('target_location')
        if not target_location:
            logger.error("缺少目標位置")
            return None
        
        # 使用 LLM 語義匹配解析位置
        location_coords = self._resolve_location(target_location)
        
        if not location_coords:
            logger.error(f"未知位置: {target_location}")
            return None
        
        logger.info(f"   目標位置: {target_location}")
        logger.info(f"      座標: {location_coords[:3]} mm")
        
        # 使用 MoveIt2 或 SimpleMoveIt 規劃
        waypoints = self.moveit_planner.plan_place_task(
            target_position=location_coords,
            velocity=100
        )
        
        # 找出關鍵位置（用於 return_to_last）
        # 記錄動作開始前的位置（第一個 move waypoint，通常是 "approach safe height"）
        key_position = None
        for wp in waypoints:
            if hasattr(wp, 'action') and wp.action == 'move' and hasattr(wp, 'position'):
                key_position = wp.position
                logger.info(f"   記錄動作開始前位置: {key_position}")
                break
        
        # 如果找不到，使用第一個有位置的 waypoint
        if not key_position:
            for wp in waypoints:
                if hasattr(wp, 'position'):
                    key_position = wp.position
                    break
        
        # 轉換為 plan_json 格式
        plan_json = {
            "waypoints": [wp.to_dict() if hasattr(wp, 'to_dict') else wp.__dict__ for wp in waypoints],
            "metadata": {
                "planner": self.planner_type,
                "task_type": "place",
                "target_location": target_location,
                "target_coords": location_coords,
                "key_position": key_position,  # 記錄動作開始前的位置（第一個 move waypoint）
                "reference_task": reference_plan.get('metadata', {}).get('task_id') if reference_plan else None
            }
        }
        
        return plan_json
    
    def _plan_move(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃移動任務"""
        target_location = task_json.get('target_location')
        if not target_location:
            logger.error("缺少目標位置")
            return None
        
        # 使用 LLM 語義匹配解析位置
        location_coords = self._resolve_location(target_location)
        
        if not location_coords:
            logger.error(f"未知位置: {target_location}")
            return None
        
        logger.info(f"   移動到: {target_location}")
        
        # 使用舊的規劃器
        waypoints = self.planner.plan_navigation_task({
            'destination': location_coords
        })
        
        # 轉換為新的 Waypoint 格式，並修正 action 名稱
        from ws_racr.execution.backends.waypoint_schema import Waypoint
        waypoint_objects = []
        for wp in waypoints:
            # 修正不支援的 action 名稱
            if wp.get('action') == 'navigate_to':
                wp['action'] = 'move_to'
            waypoint_objects.append(Waypoint.from_legacy_waypoint(wp))
        
        # 找出關鍵位置（目標位置）
        key_position = location_coords
        logger.info(f"   記錄目標位置: {key_position}")
        
        plan_json = {
            "waypoints": [wp.to_dict() if hasattr(wp, 'to_dict') else wp.__dict__ for wp in waypoint_objects],
            "metadata": {
                "planner": "legacy",
                "task_type": "move",
                "target_location": target_location,
                "target_coords": location_coords,
                "key_position": key_position  # 記錄目標位置
            }
        }
        
        return plan_json
    
    def _plan_move_relative(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃相對移動任務（使用 movePTP）"""
        direction = task_json.get('direction')
        distance = task_json.get('distance', 50.0)  # 預設 50mm
        
        if not direction:
            logger.error("缺少移動方向")
            return None
        
        # 方向映射（TM Robot 座標系）
        # 標準座標：X軸 前(-)/後(+), Y軸 左(+)/右(-), Z軸 上(+)/下(-)
        direction_map = {
            'left': [0, distance, 0, 0, 0, 0],      # Y+
            'right': [0, -distance, 0, 0, 0, 0],    # Y-
            'forward': [-distance, 0, 0, 0, 0, 0],   # X-
            'backward': [distance, 0, 0, 0, 0, 0], # X+
            'up': [0, 0, distance, 0, 0, 0],         # Z+ 修正：往上是正值
            'down': [0, 0, -distance, 0, 0, 0],      # Z- 修正：往下是負值
            '左': [0, distance, 0, 0, 0, 0],        # Y+
            '右': [0, -distance, 0, 0, 0, 0],       # Y-
            '前': [-distance, 0, 0, 0, 0, 0],        # X-
            '後': [+distance, 0, 0, 0, 0, 0],       # X+
            '上': [0, 0, distance, 0, 0, 0],         # Z+ 修正：往上是正值
            '下': [0, 0, -distance, 0, 0, 0],        # Z- 修正：往下是負值
        }
        
        relative_displacement = direction_map.get(direction)
        
        if not relative_displacement:
            logger.error(f"不支援的移動方向: {direction}")
            return None
        
        logger.info(f"   相對移動: {direction} {distance}mm")
        from ws_racr.execution.backends.waypoint_schema import Waypoint
        # 建立相對移動的 Waypoint
        waypoint = Waypoint(
            action="move_relative",
            position=relative_displacement,  # [dx, dy, dz, drx, dry, drz]
            velocity=100,  # 相對移動速度較慢
            acc_time=200,
            blend=0,
            constraints={},
            io_before=None,
            io_after=None,
            wait_time=0.0
        )
        
        plan_json = {
            "waypoints": [waypoint.to_dict()],
            "metadata": {
                "planner": "relative_move",
                "task_type": "move_relative",
                "direction": direction,
                "distance": distance
            }
        }
        
        return plan_json
    
    def _plan_return_to_last(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """
        規劃返回上個點（從資料庫讀取最後成功的位置）
        
        策略：
        1. 排除不產生有意義位置的任務類型（gripper, return 等）
        2. 查找最近 N 個有絕對位置的已完成任務（支援 steps_back 參數）
        3. 優先使用 metadata.key_position，其次 telemetry.last_position
        
        Args:
            task_json: 包含 parameters.steps_back（預設為 1）
        """
        # 取得 steps_back 參數（預設為 1）
        steps_back = task_json.get('parameters', {}).get('steps_back', 1)
        logger.info(f"   查詢往前 {steps_back} 步的位置...")
        
        # 排除這些任務類型（它們本質上不會改變位置）
        skip_task_types = [
            'return_to_last',     # 返回類任務沒有新位置
            'return_to_origin',
            'gripper_open',       # 夾爪動作不移動機器人
            'gripper_close'
        ]
        
        # 從資料庫查詢最近的 N 個已完成任務（排除上述類型）
        # 使用新的 get_recent_tasks API
        recent_tasks = self.task_manager.get_recent_tasks(k=steps_back, exclude_task_types=skip_task_types)
        
        if not recent_tasks:
            logger.error(f"   找不到往前 {steps_back} 步的位置記錄")
            logger.info("   提示: 請先執行會移動機器人的任務")
            return None
        
        # 取得第 N 個任務（index = steps_back - 1）
        if len(recent_tasks) < steps_back:
            logger.warning(f"    只有 {len(recent_tasks)} 個歷史任務，少於要求的 {steps_back} 步")
            logger.info(f"   將返回最早的任務位置")
            last_task = recent_tasks[-1]  # 最早的任務
        else:
            last_task = recent_tasks[steps_back - 1]  # 第 N 個任務
        
        if not last_task:
            logger.error("   找不到有意義的上個位置記錄")
            logger.info("   提示: 請先執行會移動機器人的任務")
            return None
        
        task_type = last_task.get('task_type')
        logger.info(f"   找到任務: {task_type}")
        
        # 嘗試從多個來源獲取最後位置（優先級從高到低）
        last_position = None
        position_source = None
        
        # 1. 優先使用 telemetry 中記錄的位置（執行前的位置，用於返回）
        #    這對所有任務都適用，記錄的是任務開始前的穩定位置
        if last_task.get('telemetry') and last_task['telemetry'].get('last_position'):
            telemetry_pos = last_task['telemetry']['last_position']
            # 檢查是否為有效位置（非全零）
            if any(abs(x) > 0.1 for x in telemetry_pos[:3]):
                last_position = telemetry_pos
                position_source = "telemetry.last_position"
                logger.info(f"   使用記錄的位置（任務開始前）")
        
        # 2. 其次使用 metadata 中的 key_position（規劃時標記的關鍵位置）
        if not last_position and last_task.get('plan_json'):
            metadata = last_task['plan_json'].get('metadata', {})
            if metadata.get('key_position'):
                last_position = metadata['key_position']
                position_source = "metadata.key_position"
                logger.info(f"   使用規劃時標記的關鍵位置")
        
        # 3. 最後從 plan_json waypoints 獲取（僅適用於絕對移動任務）
        if not last_position and last_task.get('plan_json'):
            waypoints = last_task['plan_json'].get('waypoints', [])
            if waypoints:
                # 對於相對移動任務，waypoint 的 position 是相對位移，不能直接使用
                if task_type in ['move_relative', 'rotate_relative', 'rotate_gripper']:
                    logger.warning(f"    相對移動任務缺少 telemetry 記錄")
                    logger.info(f"   這可能是因為任務執行時未能記錄最終位置")
                    logger.info(f"   請執行其他任務後再試")
                    return None
                else:
                    # 對於絕對移動任務，尋找最後一個有效的 move waypoint
                    for wp in reversed(waypoints):
                        if isinstance(wp, dict) and 'position' in wp and wp.get('action') == 'move':
                            # 檢查是否為有效位置
                            pos = wp['position']
                            if any(abs(x) > 0.1 for x in pos[:3]):
                                last_position = pos
                                position_source = "plan_json.waypoints"
                                logger.info(f"   使用 plan_json 路徑點位置")
                                break
        
        if not last_position:
            logger.error("   無法取得上個位置資訊")
            return None
        
        logger.info(f"   上個位置: [{last_position[0]:.1f}, {last_position[1]:.1f}, {last_position[2]:.1f}]")
        logger.info(f"      來自任務: {last_task['task_id']} ({task_type})")
        from ws_racr.execution.backends.waypoint_schema import Waypoint
        # 建立移動到上個位置的 Waypoint
        waypoint = Waypoint(
            action="move",
            position=last_position,
            velocity=100,  # 安全速度
            acc_time=200,
            blend=0,
            constraints={},
            io_before=None,
            io_after=None,
            wait_time=0.0
        )
        
        plan_json = {
            "waypoints": [waypoint.to_dict()],
            "metadata": {
                "planner": "return_to_last",
                "task_type": "return_to_last",
                "target_position": last_position,
                "source_task": last_task['task_id']
            }
        }
        
        return plan_json
    
    def _plan_return_to_origin(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃返回原點 - 重構版"""
        origin_position = [291, -225, 613, 180, 0, -90]
        logger.info(f"   返回原點: [{origin_position[0]}, {origin_position[1]}, {origin_position[2]}]")
        
        waypoint = self._create_simple_waypoint(
            action="move",
            position=origin_position,
            velocity=100
        )
        
        return self._create_plan_json(
            waypoints=[waypoint],
            task_type="return_to_origin",
            origin_position=origin_position
        )
    
    def _plan_rotate_gripper(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃旋轉夾爪（調整 RZ 角度）"""
        rotation_direction = task_json.get('rotation_direction')
        rotation_angle = task_json.get('rotation_angle', 45.0)  # 預設 45 度
        
        if not rotation_direction:
            logger.error("   缺少旋轉方向")
            return None
        
        # 方向映射
        direction_map = {
            'left': rotation_angle,           # 逆時針
            'right': -rotation_angle,         # 順時針
            'counterclockwise': rotation_angle,
            'clockwise': -rotation_angle,
            '左': rotation_angle,
            '右': -rotation_angle,
            '逆時針': rotation_angle,
            '順時針': -rotation_angle,
        }
        
        angle_change = direction_map.get(rotation_direction)
        
        if angle_change is None:
            logger.error(f"   不支援的旋轉方向: {rotation_direction}")
            return None
        
        logger.info(f"   旋轉夾爪: {rotation_direction} {abs(angle_change)}°")
        from ws_racr.execution.backends.waypoint_schema import Waypoint
        # 建立相對旋轉的 Waypoint（只改變 RZ）
        waypoint = Waypoint(
            action="rotate_relative",
            position=[0, 0, 0, 0, 0, angle_change],  # [dx, dy, dz, drx, dry, drz]
            velocity=100,  # 旋轉速度較慢
            acc_time=300,
            blend=0,
            constraints={},
            io_before=None,
            io_after=None,
            wait_time=0.0
        )
        
        plan_json = {
            "waypoints": [waypoint.to_dict()],
            "metadata": {
                "planner": "rotate_gripper",
                "task_type": "rotate_gripper",
                "rotation_direction": rotation_direction,
                "rotation_angle": abs(angle_change)
            }
        }
        
        return plan_json
    
    def _plan_gripper_open(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃開啟夾爪 - 重構版"""
        logger.info(f"   開啟夾爪")
        
        waypoint = self._create_simple_waypoint(
            action="gripper_open",
            position=[0, 0, 0, 0, 0, 0],
            velocity=0,
            wait_time=1.0
        )
        
        return self._create_plan_json([waypoint], "gripper_open")
    
    def _plan_gripper_close(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃關閉夾爪 - 重構版"""
        logger.info(f"   關閉夾爪")
        
        waypoint = self._create_simple_waypoint(
            action="gripper_close",
            position=[0, 0, 0, 0, 0, 0],
            velocity=0,
            wait_time=1.0
        )
        
        return self._create_plan_json([waypoint], "gripper_close")
    
    def _find_object_nearest_to_hint(
        self, class_name: str, image, depth_frame, hint_pos: list,
        spatial_modifier=None, object_size=None, hint_pixel: list = None
    ):
        """
        Detect all instances of class_name and return the one closest to the
        pointing hint.

        Strategy: pixel-space 2D distance using the top-camera pixel center
        stored from the pointing detection (hint_pixel = [cx, cy]).
        All objects on the same table surface share similar depth, so using
        depth/3D distance would give almost equal scores and is unreliable.
        Pixel space in the top (overhead) camera directly reflects X/Y position.

        Falls back to find_object_by_name if no pixel hint or no match.
        """
        import numpy as np
        try:
            all_dets = self.top_camera.detect_objects(image, [class_name])
            if not all_dets:
                return None
            if len(all_dets) == 1:
                return all_dets[0]

            # ── Pixel-space match ──────────────────────────────────────────
            if hint_pixel is not None:
                hint_px = np.array(hint_pixel[:2], dtype=float)
                best_det = None
                best_dist = float('inf')
                for det in all_dets:
                    det_px = np.array(det['center'], dtype=float)
                    dist = float(np.linalg.norm(det_px - hint_px))
                    if dist < best_dist:
                        best_dist = dist
                        best_det = det
                if best_det is not None:
                    logger.info(f"   像素最近匹配 ({class_name}): "
                          f"pixel_dist={best_dist:.1f}px → 選定 {best_det['center']}")
                    return best_det

        except Exception as e:
            logger.warning(f"   _find_object_nearest_to_hint 失敗: {e}")

        # fallback
        return self.top_camera.find_object_by_name(
            class_name, image, spatial_modifier, object_size
        )

    def _plan_handover(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """
        規劃工具交接任務 - 兩階段執行（正確的時序邏輯）
        
        ✅ 正確流程:
        1. 階段 1: 檢測物體位置 → 移動並夾取
        2. 階段 2: (夾取完成後) 檢測人手位置 → 移動並釋放 → 返回起始點
        
        這樣設計確保人手位置檢測在最後一刻，避免位置過時。
        
        ⚠️ 注意：當前實現仍是一次性規劃所有路徑，但標記了需要在執行時更新人手位置。
        未來可改為真正的兩階段執行（pickup完成後暫停，檢測人手，再繼續）。
        """
        tool = task_json.get('target_object', '工具')
        spatial_modifier = task_json.get('spatial_modifier')  # 提取空間描述詞
        object_size = task_json.get('object_size')  # 提取尺寸描述

        # If tool is a specific tracked instance (e.g. "cylinder_2"), extract the
        # YOLO class name and pre-load its known position from world_state so YOLO
        # re-detection can be anchored to the correct physical object.
        ws_hint_position = task_json.get('hint_position')  # metres [x,y,z], optional
        ws_hint_pixel = task_json.get('hint_pixel')  # [cx, cy] top-camera pixels, optional
        parts = tool.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            # tool is an instance ID — strip numeric suffix to get YOLO class
            yolo_class = parts[0]          # e.g. "cylinder"
            if ws_hint_position is None or ws_hint_pixel is None:
                try:
                    ws = getattr(self, 'world_state_builder', None)
                    if ws is not None:
                        ws_human = ws.current_state.human
                        if ws_hint_position is None:
                            obj = ws.current_state.objects.get(tool)
                            if obj is not None and obj.position is not None:
                                ws_hint_position = list(obj.position[:3])
                        if ws_hint_pixel is None and getattr(ws_human, 'pointing_pixel', None):
                            ws_hint_pixel = ws_human.pointing_pixel
                except Exception:
                    pass
        else:
            yolo_class = tool
            # Class-level target (e.g. "cylinder") can still be disambiguated
            # by pointing hints from world_state.
            if ws_hint_position is None or ws_hint_pixel is None:
                try:
                    ws = getattr(self, 'world_state_builder', None)
                    if ws is not None:
                        ws_human = ws.current_state.human
                        if ws_hint_position is None and getattr(ws_human, 'pointing_position', None):
                            ws_hint_position = list(ws_human.pointing_position[:3])
                        if ws_hint_pixel is None and getattr(ws_human, 'pointing_pixel', None):
                            ws_hint_pixel = list(ws_human.pointing_pixel[:2])
                except Exception:
                    pass

        # 解析完整物件名稱（如 screw + 大 → screw_big）
        resolved_tool = self._resolve_object_with_size(yolo_class, object_size)
        
        if spatial_modifier and object_size:
            logger.info(f"   準備交接: {spatial_modifier} {object_size} {tool}")
        elif spatial_modifier:
            logger.info(f"   準備交接: {spatial_modifier} {tool}")
        elif object_size:
            logger.info(f"   準備交接: {object_size} {tool}")
        else:
            logger.info(f"   準備交接: {tool}")
        logger.info(f"   執行模式: 兩階段任務 (pickup → detect hand → handover)")
        
        # 初始化視覺系統
        self.initialize_vision()
        
        # ==================== 階段 1: 檢測並夾取物體 ====================
        object_position = None
        
        # 檢查是否指定了具體物體（排除通用名詞）
        generic_terms = ['工具', '物體', 'object', 'tool', 'item']
        if tool not in generic_terms:  # 如果指定了具體物體名稱
            # 初始化雙相機系統
            if not self.dual_camera_initialized:
                if not self._init_dual_camera_pointing():
                    logger.error("   雙相機系統初始化失敗")
                    return None
            
            try:
                label = f"{spatial_modifier} {tool}" if spatial_modifier else tool
                logger.info(f"\n[階段 1/2] 檢測物體: {label}")
                if ws_hint_position:
                    logger.info(f"   位置提示 (world_state): {[f'{v:.3f}' for v in ws_hint_position]}")
                # 使用俯視相機D435i檢測（傳遞空間描述詞和尺寸描述）
                image_top, depth_frame_top = self.top_camera.get_frame()
                if image_top is not None:
                    # When a specific instance is requested and a position hint is
                    # available, pick the detection whose 3D position is closest to
                    # the hint — prevents picking the wrong cylinder/bracket/etc.
                    if ws_hint_pixel is not None or ws_hint_position is not None:
                        detection = self._find_object_nearest_to_hint(
                            resolved_tool, image_top, depth_frame_top,
                            ws_hint_position, spatial_modifier, object_size,
                            hint_pixel=ws_hint_pixel,
                        )
                    else:
                        detection = self.top_camera.find_object_by_name(
                            resolved_tool, image_top, spatial_modifier, object_size,
                        )
                    if detection is not None:
                        logger.info(f"   YOLO 檢測成功")
                        logger.info(f"      類別: {detection.get('class_name')}")
                        logger.info(f"      信心度: {detection.get('confidence', 0):.2f}")
                        logger.info(f"      BBox: {detection.get('bbox')}")
                        
                        grip_info = self.top_camera.calculate_gripper_angle_and_position(detection, image_top)
                        center_u, center_v = grip_info['center']
                        logger.info(f"      中心像素: ({center_u}, {center_v})")
                        
                        # 獲取深度值
                        depth_value = depth_frame_top.get_distance(center_u, center_v)
                        logger.info(f"      深度值: {depth_value:.4f} m ({depth_value * 1000:.1f} mm)")
                        
                        # 檢查深度有效性
                        if depth_value <= 0:
                            logger.error(f"   深度值無效 (≤0)")
                            logger.info(f"      可能原因:")
                            logger.info(f"      1. 物體表面反光（玻璃、金屬）")
                            logger.info(f"      2. 物體太暗（黑色物體）")
                            logger.info(f"      3. 深度相機未對準物體")
                            logger.info(f"      建議: 調整物體位置或光照條件")
                            return None
                        elif depth_value < 0.3:  # D435i 最小範圍約 0.3m
                            logger.warning(f"   物體太近 ({depth_value * 1000:.0f}mm < 300mm)")
                            logger.info(f"      建議: 將物體移遠一些（至少 30cm）")
                            return None
                        elif depth_value > 3.0:  # D435i 最大範圍約 3m
                            logger.warning(f"   物體太遠 ({depth_value * 1000:.0f}mm > 3000mm)")
                            logger.info(f"      建議: 將物體移近一些（最多 3m）")
                            return None
                        
                        # 深度有效，進行座標轉換
                        grasp_pos_3d = self._deproject_and_transform(
                            center_u, center_v,
                            depth_value,
                            self.top_intrinsics, self.T_top_to_base
                        )
                        
                        if grasp_pos_3d is not None:
                            object_position = [
                                grasp_pos_3d[0] * 1000, grasp_pos_3d[1] * 1000, grasp_pos_3d[2] * 1000,
                                180.0, 0.0, grip_info['angle']
                            ]
                            logger.info(f"   找到物體: [{object_position[0]:.1f}, {object_position[1]:.1f}, {object_position[2]:.1f}] mm")
                            logger.info(f"      夾取角度: {grip_info['angle']:.1f}°")
                        else:
                            logger.error(f"   座標轉換失敗")
                            logger.info(f"      深度值: {depth_value:.4f} m")
                            logger.info(f"      可能原因: 外參矩陣錯誤或深度值異常")
                            return None
                    else:
                        logger.error(f"   找不到物體: {tool}")
                        logger.info(f"      請確認:")
                        logger.info(f"      1. 物體在相機視野內")
                        logger.info(f"      2. 物體名稱正確（支援: 圓柱/cylinder, 扳手/hex_wrench, 螺絲/screw 等）")
                        logger.info(f"      3. 光照充足")
                        return None
                else:
                    logger.error(f"   無法獲取相機畫面")
                    return None
            except Exception as e:
                logger.error(f"   物體檢測失敗: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # 使用通用名詞（如 "物體"），需要使用空間描述詞
            if not spatial_modifier:
                logger.error(f"   使用通用名詞 '{tool}' 時，必須指定空間描述詞（如最右邊、最近等）")
                return None
            
            logger.info(f"   通用物體檢測：將使用空間篩選 '{spatial_modifier}'")
            # 初始化雙相機系統
            if not self.dual_camera_initialized:
                if not self._init_dual_camera_pointing():
                    logger.error("   雙相機系統初始化失敗")
                    return None
            
            try:
                logger.info(f"\n[階段 1/2] 檢測物體: {spatial_modifier} {tool}")
                image_top, depth_frame_top = self.top_camera.get_frame()
                if image_top is not None:
                    # 檢測所有物體，然後用空間篩選
                    # 使用空字符串作為類別名，檢測所有類別
                    all_detections = self.top_camera.detect_objects(image_top, target_classes=None)
                    
                    if not all_detections:
                        logger.error(f"   場景中沒有檢測到任何物體")
                        return None
                    
                    # 使用空間篩選
                    from ws_racr.perception.object_localizer import ObjectLocalizer
                    temp_vision = ObjectLocalizer()
                    detection = temp_vision._filter_by_spatial_modifier(all_detections, spatial_modifier)
                    
                    if not detection:
                        logger.error(f"   無法使用 '{spatial_modifier}' 篩選物體")
                        return None
                    
                    # 繼續處理檢測結果（與具體物體相同的邏輯）
                    logger.info(f"   YOLO 檢測成功")
                    logger.info(f"      類別: {detection.get('class_name')}")
                    logger.info(f"      信心度: {detection.get('confidence', 0):.2f}")
                    logger.info(f"      BBox: {detection.get('bbox')}")
                    
                    grip_info = self.top_camera.calculate_gripper_angle_and_position(detection, image_top)
                    center_u, center_v = grip_info['center']
                    logger.info(f"      中心像素: ({center_u}, {center_v})")
                    
                    depth_value = depth_frame_top.get_distance(center_u, center_v)
                    logger.info(f"      深度值: {depth_value:.4f} m ({depth_value * 1000:.1f} mm)")
                    
                    if depth_value <= 0:
                        logger.error(f"   深度值無效")
                        return None
                    
                    grasp_pos_3d = self._deproject_and_transform(
                        center_u, center_v, depth_value,
                        self.top_intrinsics, self.T_top_to_base
                    )
                    
                    if grasp_pos_3d is None:
                        logger.error(f"   無法轉換物體座標")
                        return None
                    
                    object_position = [
                        grasp_pos_3d[0] * 1000,
                        grasp_pos_3d[1] * 1000,
                        grasp_pos_3d[2] * 1000,
                        180, 0, grip_info['gripper_angle']
                    ]
                    logger.info(f"   找到物體: [{object_position[0]:.1f}, {object_position[1]:.1f}, {object_position[2]:.1f}] mm")
                    logger.info(f"      夾取角度: {grip_info['gripper_angle']:.1f}°")
                else:
                    logger.error(f"   無法獲取相機畫面")
                    return None
            except Exception as e:
                logger.error(f"   物體檢測失敗: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # 規劃階段 1: Pickup 任務
        logger.info(f"\n[階段 1/2] 規劃 pickup 任務...")
        pickup_waypoints = self.moveit_planner.plan_pickup_task(
            object_position=object_position,
            velocity=100
        )
        logger.info(f"   Pickup 計畫完成: {len(pickup_waypoints)} 個路徑點")
        
        # ==================== 階段 2: 檢測人手並交接 ====================
        # 重要：這裡我們使用預設位置進行初始規劃
        # 實際執行時，會在 pickup 完成後重新檢測人手位置並更新
        logger.info(f"\n[階段 2/2] 規劃 handover 任務...")
        logger.info(f"   注意：人手位置將在 pickup 完成後實時檢測")
        
        # 使用預設交接位置作為初始計畫（會在執行時更新）
        handover_position_default = [350, -150, 500, 180, 0, 90]  # rz=90°（與 receive 一致）
        logger.info(f"   初始規劃位置: [{handover_position_default[0]}, {handover_position_default[1]}, {handover_position_default[2]}] mm")
        logger.info(f"      (此位置會在 pickup 完成後根據實際人手位置更新)")
        
        # 規劃 handover waypoints（使用默認 wait_time=1.0）
        handover_waypoints = self.moveit_planner.plan_handover_task(
            handover_position=handover_position_default,
            velocity=100
        )
        logger.info(f"   Handover 計畫完成: {len(handover_waypoints)} 個路徑點")
        
        # ==================== 合併並標記階段 ====================
        all_waypoints = []
        
        # 階段 1: Pickup waypoints
        for i, wp in enumerate(pickup_waypoints):
            wp_dict = wp.to_dict() if hasattr(wp, 'to_dict') else wp.__dict__
            wp_dict['stage'] = 'pickup'
            wp_dict['stage_index'] = i
            all_waypoints.append(wp_dict)
        
        # 分隔點：標記 pickup 完成，需要檢測人手
        checkpoint = {
            'action': 'checkpoint',
            'checkpoint_type': 'detect_hand_before_handover',
            'description': '✅ Pickup 完成，準備檢測人手位置',
            'stage': 'transition',
            'requires_hand_detection': True  # 關鍵標記
        }
        all_waypoints.append(checkpoint)
        
        # 階段 2: Handover waypoints（第一個路徑點的位置會在執行時更新）
        for i, wp in enumerate(handover_waypoints):
            wp_dict = wp.to_dict() if hasattr(wp, 'to_dict') else wp.__dict__
            wp_dict['stage'] = 'handover'
            wp_dict['stage_index'] = i
            if i == 0:  # 第一個 handover waypoint 需要用實時檢測的人手位置更新
                wp_dict['requires_realtime_update'] = True
                wp_dict['update_type'] = 'hand_position'
                # 同時存到 metadata 確保不會丟失
                if 'metadata' not in wp_dict:
                    wp_dict['metadata'] = {}
                wp_dict['metadata']['requires_realtime_update'] = True
                wp_dict['metadata']['update_type'] = 'hand_position'
            all_waypoints.append(wp_dict)
        
        plan_json = {
            "waypoints": all_waypoints,
            "metadata": {
                "planner": self.planner_type,
                "task_type": "handover",
                "execution_mode": "two_stage",  # 新增：兩階段執行
                "stage_1": "pickup_object",
                "stage_2": "detect_hand_and_handover",
                "tool": tool,
                "object_position": object_position,
                "handover_position_default": handover_position_default,  # 預設位置
                "hand_detection_timing": "after_pickup_before_handover",  # 關鍵：在 pickup 後才檢測
                "requires_realtime_hand_detection": True,  # 標記需要實時檢測
                "total_waypoints": len(all_waypoints),
                "pickup_waypoints": len(pickup_waypoints),
                "handover_waypoints": len(handover_waypoints)
            }
        }
        
        logger.info(f"\n兩階段任務規劃完成")
        logger.info(f"   階段 1 (Pickup): {len(pickup_waypoints)} 個路徑點")
        logger.info(f"   階段 2 (Handover): {len(handover_waypoints)} 個路徑點")
        logger.info(f"   人手檢測: 在 pickup 完成後實時執行")
        logger.info(f"   執行時會在 checkpoint 處暫停並更新人手位置")
        
        return plan_json
    
    def _plan_prepare(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃準備任務 - 使用 moveit_planner.plan_prepare_task"""
        target = task_json.get('target_object', '物品')
        
        logger.info(f"   準備: {target}")
        
        # 初始化雙相機系統
        if not self.dual_camera_initialized:
            if not self._init_dual_camera_pointing():
                logger.error("   雙相機系統初始化失敗")
                return None
        
        # 1. 檢測目標物件（使用俯視相機D435i）
        logger.info(f"   檢測物件: {target}")
        image_top, depth_frame_top = self.top_camera.get_frame()
        if image_top is None:
            logger.error("   無法取得俯視相機畫面")
            return None
        
        detection = self.top_camera.find_object_by_name(target, image_top)
        if not detection:
            logger.error(f"   找不到物件: {target}")
            return None
        
        grip_info = self.top_camera.calculate_gripper_angle_and_position(detection, image_top)
        grasp_pos_3d = self._deproject_and_transform(
            grip_info['center'][0], grip_info['center'][1],
            depth_frame_top.get_distance(grip_info['center'][0], grip_info['center'][1]),
            self.top_intrinsics, self.T_top_to_base
        )
        
        if grasp_pos_3d is None:
            logger.error(f"   無法轉換物體座標")
            return None
        
        object_info_position = [
            grasp_pos_3d[0] * 1000, grasp_pos_3d[1] * 1000, grasp_pos_3d[2] * 1000,
            180.0, 0.0, grip_info['angle']
        ]
        logger.info(f"   找到物件: {object_info_position[:3]} mm")
        
        # 2. 使用公共方法檢測人手
        logger.info(f"   檢測人手位置...")
        hand_pos_3d = self._detect_hand_with_d435i(max_attempts=30)
        
        # 設定交接位置
        if hand_pos_3d is not None:
            handover_position = [
                hand_pos_3d[0] * 1000,
                hand_pos_3d[1] * 1000,
                hand_pos_3d[2] * 1000 + 100,  # 上方 100mm
                180, 0, -90
            ]
            logger.info(f"   交接位置: [{handover_position[0]:.1f}, {handover_position[1]:.1f}, {handover_position[2]:.1f}] mm")
        else:
            handover_position = [400, 0, 400, 180, 0, -90]
            logger.info(f"   使用預設交接位置")
        
        # 3. 使用 MoveIt2 規劃完整任務
        waypoints = self.moveit_planner.plan_prepare_task(
            object_position=object_info_position,
            handover_position=handover_position,
            velocity=100
        )
        
        # 使用公共方法創建 plan_json
        return self._create_plan_json(
            waypoints=waypoints,
            task_type="prepare",
            target=target,
            object_position=object_info_position,
            handover_position=handover_position,
            combined_actions=["pickup", "handover"],
            hand_detected=hand_pos_3d is not None
        )
    
    def _plan_emergency_stop(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃緊急停止 - 重構版"""
        logger.info(f"   緊急停止")
        
        waypoint = self._create_simple_waypoint(
            action="stop",
            position=[0, 0, 0, 0, 0, 0],
            velocity=0
        )
        
        plan_json = self._create_plan_json(
            waypoints=[waypoint],
            task_type="emergency_stop",
            immediate=True
        )
        plan_json['metadata']['planner'] = 'emergency'  # 特殊視為緊急
        return plan_json
    
    def _plan_retreat(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """規劃安全後退 - 重構版"""
        safety_distance = task_json.get('parameters', {}).get('safety_distance', 500)
        logger.info(f"    安全後退 ({safety_distance}mm)")
        
        safe_position = [291, -225, 613, 180, 0, -90]
        
        waypoints = self.moveit_planner.plan_retreat_task(
            safe_position=safe_position,
            velocity=100
        )
        
        return self._create_plan_json(
            waypoints=waypoints,
            task_type="retreat",
            safety_distance=safety_distance
        )
    
    def _plan_move_occluder(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """
        規劃移除遮擋物 - 針對 eye-to-hand 相機的聰明策略
        
        流程：
        1. Pickup 遮擋物
        2. Place 到安全位置（托盤邊緣或空曠區域）
        
        Args:
            task_json: {
                'action': 'move_occluder',
                'occluder_object': str,  # 遮擋物 ID
                'move_to': [x, y, z]     # 目標位置
            }
        """
        logger.info(f"   規劃移除遮擋物...")
        
        occluder_id = task_json.get('occluder_object')
        target_pos = task_json.get('move_to')
        
        if not occluder_id or not target_pos:
            logger.error(f"   缺少遮擋物 ID 或目標位置")
            return None
        
        logger.info(f"   移除遮擋物: {occluder_id}")
        logger.info(f"   移動到: [{target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f}]")
        
        # 使用 MoveIt 規劃兩段動作：pickup + place
        waypoints = []
        
        # 階段 1: Pickup 遮擋物（簡化版：直接規劃到物品位置）
        pickup_task = {
            'action': 'pickup',
            'target_object': occluder_id
        }
        pickup_plan = self._plan_pickup(pickup_task, reference_plan)
        if pickup_plan:
            waypoints.extend(pickup_plan.get('waypoints', []))
        
        # 階段 2: Place 到目標位置
        place_task = {
            'action': 'place',
            'target_location': 'custom',
            'target_position': target_pos
        }
        place_plan = self._plan_place(place_task, reference_plan)
        if place_plan:
            waypoints.extend(place_plan.get('waypoints', []))
        
        if not waypoints:
            logger.error(f"   移除遮擋物路徑規劃失敗")
            return None
        
        return self._create_plan_json(
            waypoints=waypoints,
            task_type="move_occluder",
            occluder=occluder_id,
            target_position=target_pos
        )
    
    def _plan_receive_gesture(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """
        規劃 receive 手勢處理
        
        流程：
        1. 檢查夾爪狀態
        2. 如果夾爪關閉（有物體）→ 放到人手中
           - 偵測人手位置
           - 移動到人手上方
           - 打開夾爪
           - 移動回待機位
        3. 如果夾爪張開（無物體）→ 從人手中拿取
           - 偵測人手位置
           - 移動到人手上方
           - 關閉夾爪
           - 移動回待機位
        """
        logger.info(f"   receive 手勢處理...")
        
        # 初始化視覺系統
        self.initialize_vision()
        
        if not self.vision:
            logger.error(f"   視覺系統未初始化")
            return None
        
        # 初始化雙相機系統（如果尚未初始化）
        if not hasattr(self, 'dual_camera_initialized') or not self.dual_camera_initialized:
            logger.info(f"   初始化雙相機系統...")
            if not self._init_dual_camera_pointing():
                logger.error(f"   雙相機初始化失敗")
                return None
        
        # 1. 檢查夾爪狀態
        logger.info(f"   檢查夾爪狀態...")
        gripper_closed = self._check_gripper_state()
        
        # 2. 偵測人手位置 - 使用俯視相機檢測（逐幀嘗試）
        logger.info(f"   偵測人手位置...")
        hand_pos_3d = None
        max_attempts = 30  # 最多嘗試 30 幀 (約 1 秒 @ 30fps)
        attempt = 0
        
        # 使用雙相機系統中的俯視相機檢測人手
        if hasattr(self, 'dual_camera_initialized') and self.dual_camera_initialized:
            import cv2
            import time
            
            while hand_pos_3d is None and attempt < max_attempts:
                attempt += 1
                
                try:
                    # 從俯視相機獲取新幀（wait_for_frames 會阻塞直到有新幀）
                    top_frames = self.top_pipeline.wait_for_frames()
                    top_aligned_frames = self.top_align.process(top_frames)
                    top_color_frame = top_aligned_frames.get_color_frame()
                    top_depth_frame = top_aligned_frames.get_depth_frame()
                    
                    if not top_color_frame or not top_depth_frame:
                        continue
                    
                    top_color_image = np.asanyarray(top_color_frame.get_data())
                    
                    # 使用 MediaPipe 檢測手部
                    top_rgb = cv2.cvtColor(top_color_image, cv2.COLOR_BGR2RGB)
                    hand_results = self.hands.process(top_rgb)
                    
                    if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) > 0:
                        hand_landmarks = hand_results.multi_hand_landmarks[0]
                        
                        # 使用手上有物體時不易被遮擋的點：
                        # - landmark 0: 手腕（最穩定，不會被遮擋）
                        # - landmark 8: 食指尖（手心向上時可見）
                        # - landmark 12: 中指尖（手心向上時可見）
                        wrist = hand_landmarks.landmark[0]          # 手腕
                        index_tip = hand_landmarks.landmark[8]      # 食指尖
                        middle_tip = hand_landmarks.landmark[12]    # 中指尖
                        
                        logger.info(f"         第 {attempt} 幀檢測到手部")
                        logger.info(f"         關鍵點座標: Wrist({wrist.x:.3f},{wrist.y:.3f}) Index({index_tip.x:.3f},{index_tip.y:.3f}) Middle({middle_tip.x:.3f},{middle_tip.y:.3f})")
                        
                        # 計算手掌中心（手腕權重 2.5，兩指尖各 1）
                        palm_x = (wrist.x * 2.5 + index_tip.x + middle_tip.x) / 4.5
                        palm_y = (wrist.y * 2.5 + index_tip.y + middle_tip.y) / 4.5
                        
                        h, w = top_color_image.shape[:2]
                        pixel_x = int(palm_x * w)
                        pixel_y = int(palm_y * h)
                        
                        # 獲取深度並轉換到世界座標
                        depth = top_depth_frame.get_distance(pixel_x, pixel_y)
                        
                        logger.info(f"         圖像座標: ({pixel_x}, {pixel_y})")
                        logger.info(f"         深度: {depth:.3f} m")
                        
                        if depth > 0:
                            # receive 任務：使用原始外參 (夾爪參考點)
                            hand_pos_3d = self._deproject_and_transform(
                                pixel_x, pixel_y, depth,
                                self.top_intrinsics,
                                self.T_top_to_base  # 夾爪參考點外參
                            )
                            if hand_pos_3d is not None:
                                logger.info(f"      檢測到人手（俯視相機 D435i）")
                                logger.info(f"         世界座標: [{hand_pos_3d[0]:.3f}, {hand_pos_3d[1]:.3f}, {hand_pos_3d[2]:.3f}] m")
                                logger.info(f"         使用外參: T_top_to_base (夾爪參考點)")
                                break  # 成功，立即停止
                
                except Exception as e:
                    if attempt == 1:
                        logger.warning(f"      檢測人手失敗: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 所有幀都檢測失敗
            if hand_pos_3d is None:
                logger.warning(f"      檢測 {attempt} 幀後仍未找到手部")
                logger.info(f"         可能原因:")
                logger.info(f"            1. 手心向上時，手掌幾乎平行相機視線，特徵不明顯")
                logger.info(f"            2. 手部在畫面外或被遮擋")
                logger.info(f"            3. 光照不足或過曝")
                logger.info(f"            4. 手部太小或太遠 (深度 > 1.5m)")
                
                # 保存最後一幀的 debug 影像
                try:
                    import os
                    debug_dir = "/tmp/hand_detection_debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    if 'top_color_image' in locals():
                        debug_path = os.path.join(debug_dir, f"no_hand_{timestamp}.jpg")
                        cv2.imwrite(debug_path, top_color_image)
                        logger.info(f"         Debug 影像已保存: {debug_path}")
                except Exception as de:
                    logger.warning(f"         無法保存 debug 影像: {de}")
        else:
            logger.warning(f"      雙相機系統未初始化")
        
        if hand_pos_3d is None:
            logger.error(f"   未偵測到人手")
            return None
        
        # 轉換單位：公尺 -> 毫米
        hand_pos_mm = hand_pos_3d * 1000
        logger.info(f"   人手位置: [{hand_pos_mm[0]:.1f}, {hand_pos_mm[1]:.1f}, {hand_pos_mm[2]:.1f}] mm")
        
        # 3. 計算交接位置（手部上方 122mm = 夾爪高度 12.2cm）
        handover_position = [
            hand_pos_mm[0],
            hand_pos_mm[1],
            hand_pos_mm[2] + 50,  # 上方 122mm (夾爪高度)
            180, 0, 90
        ]
        
        # 4. 根據夾爪狀態決定動作
        from ws_racr.execution.backends.waypoint_schema import Waypoint, WaypointBuilder
        waypoints = []
        
        if gripper_closed:
            logger.info(f"   夾爪關閉（有物體）→ 放置物體到人手上方")
            # 移動到手上方（保持夾爪關閉）
            waypoints.append(WaypointBuilder.move_to(
                handover_position, 60, description="Move to hand position (gripper closed)"
            ))
            # 打開夾爪放置（原地，不移動）
            waypoints.append(Waypoint(
                action="gripper_open",
                position=None,  # 不移動，原地執行
                velocity=0,
                acc_time=0,
                blend=0,
                wait_time=1.0,  # 等待人拿取
                description="Release object to human hand"
            ))
            # 回到待機位置（保持夾爪打開，不要關閉）
            logger.info(f"   回到待機位置（保持夾爪打開）")
            waypoints.append(self._create_standby_waypoint(velocity=100))
        else:
            logger.info(f"   夾爪張開（無物體）→ 從人手夾取物體")
            
            # 使用俯視相機（D435i）檢測手中物體的精確位置和角度
            logger.info(f"   使用俯視相機（D435i）檢測手中物體...")
            object_in_hand = self._detect_objects_in_hand(
                hand_pos_3d,  # 傳遞公尺單位
                distance_threshold=0.2  # 20cm 闾值
            )
            
            if object_in_hand is None:
                logger.error(f"   未檢測到手中物體，無法執行夾取")
                return None
            
            # 使用檢測到的物體位置和角度
            logger.info(f"   使用物體精確位置和角度進行夾取")
            logger.info(f"      原始 grasp_angle (圖像座標系): {object_in_hand['grasp_angle']:.1f}°")
            # 轉換角度到機器人座標系
            robot_angle = self._transform_angle_to_robot_frame(
                object_in_hand['grasp_angle'], 
                camera_type='top'
            )
            
            logger.info(f"      轉換後 robot_angle (機器人座標系): {robot_angle:.1f}°")
            
            grasp_position = [
                object_in_hand['position'][0] * 1000,  # m -> mm
                object_in_hand['position'][1] * 1000,
                object_in_hand['position'][2] * 1000,
                180,          # rx: 從上方抓取
                0,            # ry
                robot_angle   # rz: 使用計算的角度
            ]
            
            logger.info(f"      抓取位置: [{grasp_position[0]:.1f}, {grasp_position[1]:.1f}, {grasp_position[2]:.1f}] mm")
            logger.info(f"      抓取姿態: rx={grasp_position[3]:.1f}°, ry={grasp_position[4]:.1f}°, rz={grasp_position[5]:.1f}°")
            
            # 規劃抓取動作（會關閉夾爪）
            pickup_waypoints = self.moveit_planner.plan_pickup_task(
                object_position=grasp_position,
                velocity=100
            )
            waypoints.extend(pickup_waypoints)
            
            # 回到待機位置（保持夾爪關閉，不要打開）
            logger.info(f"   回到待機位置（保持夾爪關閉）")
            waypoints.append(self._create_standby_waypoint(velocity=100))
        
        plan_json = {
            "waypoints": [wp.to_dict() if hasattr(wp, 'to_dict') else wp.__dict__ for wp in waypoints],
            "metadata": {
                "planner": self.planner_type,
                "task_type": "receive_gesture",
                "gripper_closed": gripper_closed,
                "action": "release" if gripper_closed else "grasp",
                "hand_position": hand_pos_3d.tolist() if isinstance(hand_pos_3d, np.ndarray) else hand_pos_3d,
                "handover_position": handover_position
            }
        }
        
        return plan_json
    
    def _plan_pointing_gesture(self, task_json: Dict, reference_plan: Optional[Dict] = None) -> Optional[Dict]:
        """
        規劃 pointing 手勢處理
        
        流程：
        1. 使用雙相機 + 手部射線找到物體位置
        2. 去拿這物體
        3. 偵測人手位置
        4. 移動到人手上方
        5. 打開夾爪
        6. 移動回待機位
        """
        logger.info(f"   pointing 手勢處理...")
        
        # 初始化視覺系統
        self.initialize_vision()
        
        if not self.vision:
            logger.error(f"   視覺系統未初始化")
            return None
        
        # 1. 使用雙相機手部射線找到指向的物體
        logger.info(f"   使用雙相機手部射線檢測...")
        pointed_object = self._detect_pointed_object_dual_camera()
        
        if not pointed_object:
            logger.error(f"   未檢測到指向的物體")
            return None
        
        logger.info(f"   檢測到物體: {pointed_object['class_name']}")
        logger.info(f"      射線匹配位置 (withoutgripper): [{pointed_object['position'][0]:.3f}, {pointed_object['position'][1]:.3f}, {pointed_object['position'][2]:.3f}] m")
        
        # 重要：使用夾爪外參重新計算物體位置用於實際夹取
        # 射線匹配時使用 withoutgripper 外參，但實際夹取必須使用夾爪外參
        logger.info(f"   使用夾爪外參重新計算物體位置...")
        grasp_pos_3d = self._deproject_and_transform(
            pointed_object['pixel_center'][0],
            pointed_object['pixel_center'][1],
            pointed_object['depth'],
            self.top_intrinsics,
            self.T_top_to_base  # 使用夾爪外參
        )
        
        if grasp_pos_3d is None:
            logger.error(f"   無法轉換到夾取位置")
            return None
        
        logger.info(f"      實際夹取位置 (gripper): [{grasp_pos_3d[0]:.3f}, {grasp_pos_3d[1]:.3f}, {grasp_pos_3d[2]:.3f}] m")
        
        # 2. 使用 mask 計算抓取角度（使用重構的輔助方法）
        grasp_angle = -90  # 預設角度
        if 'mask' in pointed_object and pointed_object['mask'] is not None:
            try:
                import cv2
                mask = pointed_object['mask']
                
                # 二值化 mask
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # 使用提取的方法計算角度
                image_angle = self._calculate_grasp_angle_from_mask(binary_mask)
                
                # 轉換到機器人座標系
                grasp_angle = self._transform_angle_to_robot_frame(image_angle, camera_type='top')
                
            except Exception as e:
                logger.warning(f"      Mask 處理失敗: {e}，使用預設角度")
                import traceback
                traceback.print_exc()
        
        # 轉換位置：公尺 -> 毫米，並加入計算出的抓取姿態
        object_position_mm = [
            grasp_pos_3d[0] * 1000,  # x (m -> mm) - 使用夾爪外參計算的位置
            grasp_pos_3d[1] * 1000,  # y (m -> mm)
            grasp_pos_3d[2] * 1000,  # z (m -> mm)
            180,         # rx (度) - 從上方抓取
            0,           # ry (度)
            grasp_angle  # rz (度) - 使用 mask 計算的角度
        ]
        
        # 3. 規劃抓取物體
        logger.info(f"   規劃抓取...")
        logger.info(f"      位置 (mm): [{object_position_mm[0]:.1f}, {object_position_mm[1]:.1f}, {object_position_mm[2]:.1f}]")
        logger.info(f"      姿態 (度): [rx={object_position_mm[3]:.1f}, ry={object_position_mm[4]:.1f}, rz={object_position_mm[5]:.1f}]")
        pickup_waypoints = self.moveit_planner.plan_pickup_task(
            object_position=object_position_mm,
            velocity=100
        )
        
        # 4. 移動到固定待機位置（保持夾爪關閉）- 使用輔助方法
        logger.info(f"   移動到待機位置...")
        standby_waypoint = self._create_standby_waypoint(velocity=100)
        
        # 5. 合併所有 waypoints（抓取 + 移到待機位置，夾爪保持關閉）
        all_waypoints = pickup_waypoints + [standby_waypoint]
        
        plan_json = {
            "waypoints": [wp.to_dict() if hasattr(wp, 'to_dict') else wp.__dict__ for wp in all_waypoints],
            "metadata": {
                "planner": self.planner_type,
                "task_type": "pointing_gesture",
                "pointed_object": pointed_object['class_name'],
                "object_position": pointed_object['position'],
                "standby_position": [402, -134, 554, 180, 0, 90],
                "total_steps": len(all_waypoints)
            }
        }
        
        return plan_json
    
    def _calculate_grasp_angle_from_mask(self, mask: np.ndarray) -> float:
        """
        從物體 mask 計算抓取角度（短邊夾取策略）
        
        Args:
            mask: 物體的分割 mask (H, W) 二值圖
            
        Returns:
            float: 抓取角度（圖像座標系，度）
        """
        import cv2
        
        # 找到 mask 的輪廓
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning(f"      無法找到 mask 輪廓")
            return -90.0  # 默認角度
        
        # 使用最大輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 計算最小外接矩形
        # OpenCV minAreaRect 返回：
        # - (center_x, center_y): 中心點
        # - (width, height): 寬度和高度（不保證哪個大）
        # - angle: [-90, 0) 範圍，是矩形相對於 X 軸的旋轉角度
        #   當 width < height 時，angle 是長邊與 X 軸的夾角
        #   當 width > height 時，angle 是短邊與 X 軸的夾角
        rect = cv2.minAreaRect(largest_contour)
        (cx, cy), (width, height), angle = rect
        
        # 獲取矩形的四個頂點
        box = cv2.boxPoints(rect)
        
        # 計算長邊向量（第一條邊）
        edge1 = box[1] - box[0]
        edge1_length = np.linalg.norm(edge1)
        edge1_angle = np.arctan2(edge1[1], edge1[0]) * 180 / np.pi
        
        # 計算第二條邊
        edge2 = box[2] - box[1]
        edge2_length = np.linalg.norm(edge2)
        edge2_angle = np.arctan2(edge2[1], edge2[0]) * 180 / np.pi
        
        # 短邊夾取：夾爪沿著短邊方向夾（夾爪開口垂直於長邊）
        if edge1_length > edge2_length:
            # edge1 是長邊，edge2 是短邊
            grasp_angle = edge2_angle
            short_edge = edge2_length
            long_edge = edge1_length
        else:
            # edge2 是長邊，edge1 是短邊
            grasp_angle = edge1_angle
            short_edge = edge1_length
            long_edge = edge2_length
        
        # 標準化角度到 [-90, 90) 範圍
        while grasp_angle >= 90:
            grasp_angle -= 180
        while grasp_angle < -90:
            grasp_angle += 180
        
        logger.info(f"      圖像角度: {grasp_angle:.1f}° (OpenCV angle={angle:.1f}°)")
        logger.info(f"         短邊: {short_edge:.1f} px, 長邊: {long_edge:.1f} px")
        
        return grasp_angle
    
    def _transform_angle_to_robot_frame(self, angle_deg: float, camera_type: str = 'top') -> float:
        """
        將圖像座標系角度轉換到機器人座標系
        
        Args:
            angle_deg: 圖像座標系中的角度（度）
            camera_type: 'top' 或 'front'
            
        Returns:
            float: 機器人座標系中的角度（度）
        """
        # 選擇對應的外參矩陣
        T_matrix = self.T_top_to_base if camera_type == 'top' else None
        
        if T_matrix is None:
            logger.warning(f"      缺少 {camera_type} 相機外參矩陣")
            return angle_deg
        
        # 獲取旋轉矩陣
        R_cam_to_robot = T_matrix[:3, :3]
        
        # 將角度轉換為方向向量
        angle_rad = np.radians(angle_deg)
        direction_cam = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
        
        # 轉換到機器人座標系
        direction_robot = R_cam_to_robot @ direction_cam
        
        # 計算機器人座標系中的角度
        robot_angle = np.arctan2(direction_robot[1], direction_robot[0]) * 180 / np.pi
        
        # 正規化到 [-180, 180)
        while robot_angle > 180:
            robot_angle -= 360
        while robot_angle < -180:
            robot_angle += 360
        
        logger.info(f"      機器人座標系角度: {robot_angle:.1f}°")
        return robot_angle
    
    def _detect_objects_in_hand(self, hand_pos_3d: np.ndarray, distance_threshold: float = 0.15) -> Optional[Dict]:
        """
        使用俯視相機檢測手部附近的物體
        
        Args:
            hand_pos_3d: 手部3D位置 [x, y, z] (公尺)
            distance_threshold: 距離閾值（公尺），物體必須在手附近
            
        Returns:
            Dict: {
                'position': [x, y, z] (公尺),
                'class_name': str,
                'confidence': float,
                'mask': np.ndarray,  # 分割 mask
                'grasp_angle': float  # 圖像座標系角度（度）
            } or None
        """
        import cv2
        
        if not self.dual_camera_initialized:
            logger.warning(f"      雙相機系統未初始化")
            return None
        
        logger.info(f"      使用俯視相機檢測手中物體...")
        logger.info(f"         手部位置: [{hand_pos_3d[0]:.3f}, {hand_pos_3d[1]:.3f}, {hand_pos_3d[2]:.3f}] m")
        logger.info(f"         距離閾值: {distance_threshold:.3f} m")
        
        # 獲取俯視相機影像
        top_frames = self.top_pipeline.wait_for_frames()
        top_frames = self.top_align.process(top_frames)
        top_color = top_frames.get_color_frame()
        top_depth = top_frames.get_depth_frame()
        
        if not top_color or not top_depth:
            logger.warning(f"      俯視相機無影像")
            return None
        
        top_img = np.asanyarray(top_color.get_data())
        
        # 使用 MediaPipe 檢測手部區域（用於排除）
        hand_regions = []
        try:
            top_rgb = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(top_rgb)
            
            if hand_results.multi_hand_landmarks:
                h, w = top_img.shape[:2]
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # 取手部所有關鍵點的 bounding box
                    xs = [lm.x * w for lm in hand_landmarks.landmark]
                    ys = [lm.y * h for lm in hand_landmarks.landmark]
                    
                    # 擴大手部區域 30% 以確保完全覆蓋
                    margin = 0.3
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    
                    hand_bbox = [
                        int(x_min - x_range * margin),
                        int(y_min - y_range * margin),
                        int(x_max + x_range * margin),
                        int(y_max + y_range * margin)
                    ]
                    hand_regions.append(hand_bbox)
                
                logger.info(f"         檢測到 {len(hand_regions)} 個手部區域（用於記錄）")
        except Exception as e:
            logger.warning(f"         手部檢測失敗: {e}")
        
        # YOLO 檢測物體
        results = self.yolo_model.predict(source=top_img, conf=0.3, verbose=False)
        
        if len(results) == 0:
            logger.warning(f"      未檢測到任何物體")
            return None
        
        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None:
            logger.warning(f"      無檢測結果")
            return None
        
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        
        # 獲取類別名稱映射（用於過濾 person）
        class_names_map = r.names if hasattr(r, 'names') else {}
        logger.info(f"         YOLO 類別定義: {class_names_map}")
        
        # 獲取 masks（如果有）
        masks = None
        if hasattr(r, 'masks') and r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            logger.info(f"         檢測到 {len(masks)} 個物體 masks")
        else:
            logger.warning(f"         未檢測到 masks (hasattr={hasattr(r, 'masks')}, is None={r.masks is None if hasattr(r, 'masks') else 'N/A'})")
        
        # 找到最接近手部的物體
        closest_object = None
        min_distance = float('inf')
        
        for idx, (bbox, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 檢查類別名稱，過濾 person
            class_name = class_names_map.get(int(class_id), f"class_{int(class_id)}")
            
            # 過濾掉 person 類別（COCO/預訓練模型可能有 person）
            # 邏輯：YOLO 可能把手臂/手部整體誤判為 "person"，但我們要的是手上拿的物體
            if class_name.lower() in ['person', 'hand', 'arm']:
                logger.warning(f"         跳過人體部位類別: {class_name} (class_id={int(class_id)})")
                continue
            
            logger.info(f"         檢測到物體: {class_name} (class_id={int(class_id)}, conf={confidence:.2f})")
            
            # 取深度
            depths = []
            for v in range(y1, y2, 2):
                for u in range(x1, x2, 2):
                    d = top_depth.get_distance(u, v)
                    if d > 0:
                        depths.append(d)
            
            if len(depths) == 0:
                continue
            
            depth = np.median(depths)
            
            # _detect_objects_in_hand (receive 任務)：使用原始外參 (夾爪參考點)
            pos_world = self._deproject_and_transform(
                cx, cy, depth, self.top_intrinsics, self.T_top_to_base  # 夾爪參考點外參
            )
            
            if pos_world is None:
                continue
            
            # 計算與手的距離（只考慮 XY 平面，忽略高度差異）
            distance_xy = np.sqrt(
                (pos_world[0] - hand_pos_3d[0])**2 + 
                (pos_world[1] - hand_pos_3d[1])**2
            )
            
            logger.info(f"         物體 {int(class_id)}: 距離手 {distance_xy:.3f} m")
            
            # 如果在閾值內且更近，更新最接近的物體
            if distance_xy < distance_threshold and distance_xy < min_distance:
                min_distance = distance_xy
                
                # 計算抓取角度
                grasp_angle = -90.0  # 默認
                if masks is not None and idx < len(masks):
                    logger.info(f"         使用 mask 計算物體 {int(class_id)} 的抓取角度")
                    mask = masks[idx]
                    # 將 mask resize 到圖像尺寸
                    mask_resized = cv2.resize(
                        mask, (top_img.shape[1], top_img.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    grasp_angle = self._calculate_grasp_angle_from_mask(mask_binary)
                    logger.info(f"            計算出的角度: {grasp_angle:.1f}°")
                else:
                    logger.warning(f"         物體 {int(class_id)} 無 mask，使用默認角度 -90°")
                
                closest_object = {
                    'position': pos_world,
                    'class_name': str(int(class_id)),
                    'confidence': float(confidence),
                    'grasp_angle': grasp_angle,
                    'distance_to_hand': distance_xy
                }
        
        if closest_object:
            logger.info(f"      找到手中物體: {closest_object['class_name']}")
            logger.info(f"         位置: [{closest_object['position'][0]:.3f}, "
                  f"{closest_object['position'][1]:.3f}, "
                  f"{closest_object['position'][2]:.3f}] m")
            logger.info(f"         距離手: {closest_object['distance_to_hand']:.3f} m")
            
            # 保存檢測圖片（用於調試）
            try:
                import os
                import time
                debug_dir = "/tmp/receive_grasp_debug"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # 保存原始圖片
                original_path = os.path.join(debug_dir, f"receive_grasp_original_{timestamp}.jpg")
                cv2.imwrite(original_path, top_img)
                
                # 繪製檢測結果和 mask
                debug_img = top_img.copy()
                
                # 找到對應的物體索引（用於繪製）
                for idx, (bbox, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                    x1, y1, x2, y2 = map(int, bbox)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # 計算物體位置並檢查是否為最接近的物體
                    depths = []
                    for v in range(y1, y2, 2):
                        for u in range(x1, x2, 2):
                            d = top_depth.get_distance(u, v)
                            if d > 0:
                                depths.append(d)
                    
                    if len(depths) == 0:
                        continue
                    
                    depth = np.median(depths)
                    pos_world = self._deproject_and_transform(
                        cx, cy, depth, self.top_intrinsics, self.T_top_to_base
                    )
                    
                    if pos_world is None:
                        continue
                    
                    distance_xy = np.sqrt(
                        (pos_world[0] - hand_pos_3d[0])**2 + 
                        (pos_world[1] - hand_pos_3d[1])**2
                    )
                    
                    # 判斷是否為目標物體
                    is_target = (distance_xy == min_distance)
                    color = (0, 255, 0) if is_target else (255, 0, 0)  # 綠色=目標，藍色=其他
                    thickness = 3 if is_target else 2
                    
                    # 繪製 bounding box
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
                    
                    # 繪製類別和距離
                    label = f"Class:{int(class_id)} Dist:{distance_xy:.2f}m"
                    cv2.putText(debug_img, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 繪製 mask（如果有）
                    if masks is not None and idx < len(masks) and is_target:
                        mask = masks[idx]
                        mask_resized = cv2.resize(
                            mask, (top_img.shape[1], top_img.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        # 創建半透明 mask overlay
                        mask_colored = np.zeros_like(debug_img)
                        mask_colored[:, :] = (0, 255, 255)  # 黃色
                        mask_overlay = cv2.addWeighted(
                            debug_img, 1.0,
                            cv2.bitwise_and(mask_colored, mask_colored, mask=mask_binary), 0.4,
                            0
                        )
                        debug_img = mask_overlay
                        
                        # 保存單獨的 mask 圖片
                        mask_path = os.path.join(debug_dir, f"receive_grasp_mask_{timestamp}.jpg")
                        mask_visual = (mask_binary * 255).astype(np.uint8)
                        cv2.imwrite(mask_path, mask_visual)
                
                # 保存標註後的圖片
                annotated_path = os.path.join(debug_dir, f"receive_grasp_annotated_{timestamp}.jpg")
                cv2.imwrite(annotated_path, debug_img)
                
                logger.info(f"         Debug 圖片已保存:")
                logger.info(f"            原始: {original_path}")
                logger.info(f"            標註: {annotated_path}")
                if masks is not None:
                    logger.info(f"            Mask: {mask_path}")
                    
            except Exception as e:
                logger.warning(f"         無法保存 debug 圖片: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.error(f"      未找到手部附近的物體（閾值 {distance_threshold:.3f} m）")
        
        return closest_object
    
    def _create_standby_waypoint(self, velocity: int = 100) -> 'Waypoint':
        """
        創建回待機位置的 waypoint
        
        Args:
            velocity: 移動速度百分比
            
        Returns:
            Waypoint: 待機位置 waypoint
        """
        from ws_racr.execution.backends.waypoint_schema import WaypointBuilder
        
        standby_position = [402, -134, 554, 180, 0, 90]  # 固定待機位置
        return WaypointBuilder.move_to(
            standby_position,
            velocity=velocity,
            description="Return to standby position"
        )
    
    def _check_gripper_state(self) -> bool:
        """
        檢查夾爪狀態（從 DI 讀取，已在 llm_communicator 中正確實現）
        
        Returns:
            True: 夾爪關閉（有物體）
            False: 夾爪張開（無物體）
        """
        try:
            logger.info(f"      檢查夾爪狀態...")
            
            # 從 communicator 讀取已處理好的 gripper_closed 狀態
            # 該狀態已根據 gripper_state.md 規範從 DI 正確計算
            if hasattr(self, 'backend_manager') and self.backend_manager is not None:
                if hasattr(self.backend_manager, 'backend') and self.backend_manager.backend is not None:
                    if hasattr(self.backend_manager.backend, 'communicator'):
                        comm = self.backend_manager.backend.communicator
                        
                        # 直接讀取快取的 robot_state（spin_once 不得在 worker thread 呼叫）
                        if hasattr(comm, 'robot_state') and comm.robot_state is not None:
                            logger.info(f"      robot_state 存在")
                            if 'gripper_closed' in comm.robot_state:
                                gripper_state = comm.robot_state['gripper_closed']
                                logger.info(f"      gripper_closed 原始值: {gripper_state} (type: {type(gripper_state).__name__})")
                                logger.info(f"      夾爪狀態判定: {'關閉 (有物體)' if gripper_state else '張開 (無物體)'}")
                                return gripper_state
                            else:
                                logger.warning(f"      robot_state 中沒有 'gripper_closed' 鍵")
                                logger.info(f"      可用的 keys: {list(comm.robot_state.keys())[:10]}")
                        else:
                            logger.warning(f"      robot_state 不存在或為 None")
                    else:
                        logger.warning(f"      communicator 不存在")
                else:
                    logger.warning(f"      backend 不存在")
            else:
                logger.warning(f"      backend_manager 不存在")
            
            # 無法讀取，假設張開
            logger.error(f"      無法讀取夾爪狀態，假設張開")
            return False
            
        except Exception as e:
            logger.error(f"      檢查夾爪狀態失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
