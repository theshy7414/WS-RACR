"""
ExecutionMixin — Low-level waypoint and skill execution helpers.

Extracted from RobotController to reduce per-query context size.
All methods rely on self.* attributes initialized by RobotController.__init__().
Do NOT instantiate this class directly.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, List, Any, Tuple

logger = logging.getLogger(__name__)


class ExecutionMixin:
    """
    Mixin providing low-level execution methods for RobotController.

    Included methods:
      - _execute_pickup / _execute_place / _execute_move
      - _execute_gripper_open / _execute_gripper_close
      - _execute_waypoints_v2 (primary TMScript-batched executor)
      - _execute_waypoints (legacy executor)
      - _execute_with_robot
      - _simulate_waypoints

    All methods access shared state via self.* (RobotController instance).
    """
    def _execute_pickup(self, task_json: Dict) -> bool:
        """執行抓取任務 - 使用 MoveIt2 規劃 + TMScript 執行"""
        logger.info("\n執行抓取任務...")
        
        target_object = task_json.get('target_object')
        if not target_object:
            logger.error("缺少目標物件")
            return False
        
        # 初始化雙相機系統（使用俯視相機D435i）
        if not self.dual_camera_initialized:
            if not self._init_dual_camera_pointing():
                logger.error("雙相機系統初始化失敗")
                return False
        
        # 檢測物件位置
        logger.info(f"檢測物件: {target_object}")
        image_top, depth_frame_top = self.top_camera.get_frame()
        if image_top is None:
            logger.error("無法取得俯視相機畫面")
            return False
        
        detection = self.top_camera.find_object_by_name(target_object, image_top)
        if not detection:
            logger.error(f"找不到物件: {target_object}")
            return False
        
        grip_info = self.top_camera.calculate_gripper_angle_and_position(detection, image_top)
        grasp_pos_3d = self._deproject_and_transform(
            grip_info['center'][0], grip_info['center'][1],
            depth_frame_top.get_distance(grip_info['center'][0], grip_info['center'][1]),
            self.top_intrinsics, self.T_top_to_base
        )
        
        if grasp_pos_3d is None:
            logger.error(f"無法轉換物體座標")
            return False
        
        logger.info(f"找到物件: {detection['class_name']}")
        logger.info(f"   位置: [{grasp_pos_3d[0]*1000:.1f}, {grasp_pos_3d[1]*1000:.1f}, {grasp_pos_3d[2]*1000:.1f}] mm")
        logger.info(f"   角度: {grip_info['angle']:.1f}°")
        
        # 取得物件位置
        object_position = [
            grasp_pos_3d[0] * 1000, grasp_pos_3d[1] * 1000, grasp_pos_3d[2] * 1000,
            180.0, 0.0, grip_info['angle']
        ]
        
        # 使用新的規劃器（MoveIt2 或 SimpleMoveIt）
        logger.info(f"\n規劃抓取路徑 ({self.planner_type})...")
        
        try:
            # 使用 MoveIt2 規劃器生成路徑點
            # 使用 MoveIt2 規劃
            waypoints = self.moveit_planner.plan_pickup_task(
                object_position=object_position,
                velocity=100
            )
            
            logger.info(f"規劃完成: {len(waypoints)} 個路徑點")
            
            # 顯示路徑點
            for i, wp in enumerate(waypoints):
                if wp.action in ["gripper_open", "gripper_close"]:
                    logger.info(f"   {i+1}. {wp.description or wp.action}")
                else:
                    logger.info(f"   {i+1}.  {wp.description or wp.action} (速度: {wp.velocity}%)")
            
        except Exception as e:
            logger.error("MoveIt2 pickup planning failed: %s", e)
            return False
        
        # 使用新的執行後端
        return self._execute_waypoints_v2(waypoints)
    
    def _execute_place(self, task_json: Dict) -> bool:
        """執行放置任務 - 使用 MoveIt2 規劃 + TMScript 執行"""
        logger.info("\n執行放置任務...")
        
        target_location = task_json.get('target_location')
        if not target_location:
            logger.error("缺少目標位置")
            return False
        
        # 使用 LLM 語義匹配解析位置
        location_coords = self._resolve_location(target_location)
        
        if not location_coords:
            logger.error(f"未知位置: {target_location}")
            return False
        
        logger.info(f"目標位置: {target_location}")
        logger.info(f"   座標: {location_coords[:3]} mm")
        
        # 使用新的規劃器（MoveIt2 或 SimpleMoveIt）
        logger.info(f"\n規劃放置路徑 ({self.planner_type})...")
        
        try:
            # 使用 MoveIt2 規劃器生成路徑點
            waypoints = self.moveit_planner.plan_place_task(
                target_position=location_coords,
                velocity=100
            )
            
            logger.info(f"規劃完成: {len(waypoints)} 個路徑點")
            
            # 顯示路徑點
            for i, wp in enumerate(waypoints):
                if wp.action in ["gripper_open", "gripper_close"]:
                    logger.info(f"   {i+1}. {wp.description or wp.action}")
                else:
                    logger.info(f"   {i+1}.  {wp.description or wp.action} (速度: {wp.velocity}%)")
            
        except Exception as e:
            logger.error("MoveIt2 place planning failed: %s", e)
            return False
        
        # 使用新的執行後端
        return self._execute_waypoints_v2(waypoints)
    
    def _execute_move(self, task_json: Dict) -> bool:
        """執行移動任務"""
        logger.info("\n執行移動任務...")
        
        target_location = task_json.get('target_location')
        if not target_location:
            logger.error("缺少目標位置")
            return False
        
        # 使用 LLM 語義匹配解析位置
        location_coords = self._resolve_location(target_location)
        
        if not location_coords:
            logger.error(f"未知位置: {target_location}")
            return False
        
        logger.info(f"移動到: {target_location}")
        
        # 規劃路徑
        waypoints = self.planner.plan_navigation_task({
            'destination': location_coords
        })
        
        # 初始化 ROS（如果需要）
        if self.communicator is None:
            self.initialize_ros()
        
        # 執行動作
        return self._execute_waypoints(waypoints)
    
    def _execute_gripper_open(self) -> bool:
        """開啟夾爪"""
        logger.info("\n開啟夾爪...")
        
        # 初始化 ROS（如果需要）
        if self.communicator is None:
            self.initialize_ros()
        
        if self.communicator:
            try:
                success = self.communicator.set_gripper(closed=False)
                if success:
                    logger.info("   指令已發送")
                    import time
                    time.sleep(1.0)
                    return True
                else:
                    logger.error("   發送失敗")
                    return False
            except Exception as e:
                logger.error(f"   執行失敗: {e}")
                return False
        else:
            logger.warning("模擬執行")
            return True
    
    def _execute_gripper_close(self) -> bool:
        """關閉夾爪"""
        logger.info("\n關閉夾爪...")
        
        # 初始化 ROS（如果需要）
        if self.communicator is None:
            self.initialize_ros()
        
        if self.communicator:
            try:
                success = self.communicator.set_gripper(closed=True)
                if success:
                    logger.info("   指令已發送")
                    import time
                    time.sleep(1.0)
                    return True
                else:
                    logger.error("   發送失敗")
                    return False
            except Exception as e:
                logger.error(f"   執行失敗: {e}")
                return False
        else:
            logger.warning("模擬執行")
            return True
    
    def _execute_waypoints_v2(self, waypoints: list) -> bool:
        """
        執行路徑點序列 - 支持 checkpoint 暫停與實時更新
        
        Args:
            waypoints: Waypoint 物件列表（可能包含 checkpoint dict）
            
        Returns:
            bool: 執行是否成功
        """
        if not waypoints:
            logger.warning("沒有路徑點可執行")
            return False
        
        logger.info(f"\n使用 {self.backend_type} 執行 {len(waypoints)} 個路徑點...")
        
        try:
            # 檢查是否有 checkpoint（需要分批執行）
            has_checkpoint = any(
                isinstance(wp, dict) and wp.get('action') == 'checkpoint' 
                for wp in waypoints
            )
            
            if not has_checkpoint:
                # 沒有 checkpoint，直接執行所有 waypoints
                success = self.backend_manager.execute_waypoints(waypoints)
                if success:
                    logger.info(f"所有路徑點執行完成")
                    return True
                else:
                    logger.error(f"路徑點執行失敗")
                    return False
            
            # 有 checkpoint，分批執行
            logger.info(f"檢測到 checkpoint，將分批執行")
            
            batch_start_idx = 0
            for i, wp in enumerate(waypoints):
                if isinstance(wp, dict) and wp.get('action') == 'checkpoint':
                    # 先執行 checkpoint 之前的所有 waypoints
                    if i > batch_start_idx:
                        batch = waypoints[batch_start_idx:i]
                        logger.info(f"\n執行批次 1/{2}: Waypoints {batch_start_idx+1}-{i} (Pickup 階段)")
                        success = self.backend_manager.execute_waypoints(batch)
                        if not success:
                            logger.error(f"Pickup 階段執行失敗")
                            return False
                        logger.info(f"Pickup 階段完成")
                    
                    # 處理 checkpoint：檢測人手並更新後續 waypoints
                    checkpoint_type = wp.get('checkpoint_type')
                    if checkpoint_type == 'detect_hand_before_handover':
                        logger.info(f"\n Checkpoint: {wp.get('description', '檢測人手位置')}")
                        
                        # 檢測人手（使用公共方法）
                        logger.info(f"   等待人手出現...（請將手放在 D435i 俯視相機下方）")
                        hand_pos_3d = self._detect_hand_with_d435i(max_attempts=60)  # 30秒
                        
                        if hand_pos_3d is None:
                            logger.error(f"   未檢測到人手，無法執行交接任務")
                            return False
                        
                        # 計算交接位置（對齊 receive 手勢參數）
                        hand_position_updated = [
                            hand_pos_3d[0] * 1000,  # 公尺 → 毫米
                            hand_pos_3d[1] * 1000,
                            hand_pos_3d[2] * 1000 + 50,  # Z軸上方 50mm（與 receive 一致）
                            180, 0, 90  # rz=90°（與 receive 一致）
                        ]
                        logger.info(f"   人手位置: [{hand_position_updated[0]:.1f}, {hand_position_updated[1]:.1f}, {hand_position_updated[2]:.1f}] mm")
                        
                        # 更新後續所有需要實時更新的 waypoints
                        updated_count = 0
                        for j in range(i + 1, len(waypoints)):
                            future_wp = waypoints[j]
                            
                            # 檢查是否需要更新（支持多種格式）
                            needs_update = False
                            if isinstance(future_wp, dict):
                                needs_update = future_wp.get('requires_realtime_update', False)
                                # 也檢查 metadata 欄位
                                if not needs_update and 'metadata' in future_wp:
                                    needs_update = future_wp['metadata'].get('requires_realtime_update', False)
                            elif hasattr(future_wp, 'metadata') and isinstance(future_wp.metadata, dict):
                                needs_update = future_wp.metadata.get('requires_realtime_update', False)
                            elif hasattr(future_wp, 'requires_realtime_update'):
                                needs_update = getattr(future_wp, 'requires_realtime_update', False)
                            
                            if needs_update:
                                # 更新位置
                                if isinstance(future_wp, dict):
                                    future_wp['position'] = hand_position_updated.copy()
                                    logger.info(f"      更新 Waypoint {j+1} (dict): {hand_position_updated[:3]}")
                                elif hasattr(future_wp, 'position'):
                                    future_wp.position = hand_position_updated.copy()
                                    logger.info(f"      更新 Waypoint {j+1} (object): {hand_position_updated[:3]}")
                                updated_count += 1
                        
                        logger.info(f"   已更新 {updated_count} 個後續路徑點")
                    
                    # 執行 checkpoint 之後的 waypoints
                    batch_start_idx = i + 1
            
            # 執行最後一批 waypoints（checkpoint 之後）
            if batch_start_idx < len(waypoints):
                batch = waypoints[batch_start_idx:]
                logger.info(f"\n執行批次 2/{2}: Waypoints {batch_start_idx+1}-{len(waypoints)} (Handover 階段)")
                success = self.backend_manager.execute_waypoints(batch)
                if not success:
                    logger.error(f"Handover 階段執行失敗")
                    return False
                logger.info(f"Handover 階段完成")
            
            logger.info(f"所有批次執行完成")
            return True
                
        except Exception as e:
            logger.error(f"執行異常: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_waypoints(self, waypoints: list) -> bool:
        """
        執行路徑點序列 - 生成並發送 TMScript
        
        Args:
            waypoints: 路徑點列表
            
        Returns:
            bool: 執行是否成功
        """
        if not waypoints:
            logger.warning("沒有路徑點可執行")
            return False
        
        logger.info(f"\n開始執行 {len(waypoints)} 個路徑點...")
        
        # 如果 ROS 可用，使用實際機器人
        if self.communicator:
            return self._execute_with_robot(waypoints)
        else:
            logger.warning("機器人通訊不可用，僅模擬執行")
            return self._simulate_waypoints(waypoints)
    
    def _execute_with_robot(self, waypoints: list) -> bool:
        """使用實際機器人執行路徑點"""
        try:
            import rclpy
            import time
            
            for i, wp in enumerate(waypoints, 1):
                action = wp.get('action', 'move')
                logger.info(f"\n   [{i}/{len(waypoints)}] {action}")
                
                # 移動類型的動作
                if action in ['move', 'move_to_safe_position', 'move_to_pickup', 'lift_object',
                             'move_to_target_safe', 'move_to_release', 'return_safe',
                             'navigate', 'move_to_handover']:
                    # 使用 RobotCommunicator 移動
                    position = wp.get('position', [])
                    velocity = wp.get('velocity', 100) / 100.0  # 轉換為 0.0-1.0
                    
                    success = self.communicator.move_to_position(
                        position=position,
                        velocity=velocity,
                        acc_time=0.2
                    )
                    
                    if success:
                        logger.info(f"      已發送移動指令: move_{i}")
                        logger.info(f"      位置: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}]")
                        
                        # 等待移動完成
                        if self.communicator.wait_for_movement_complete(timeout=15.0):
                            logger.info(f"      移動完成")
                        else:
                            logger.warning(f"      移動超時")
                    else:
                        logger.error(f"      發送移動指令失敗")
                
                # 開啟夾爪
                elif action in ['gripper_open', 'release', 'release_to_human']:
                    success = self.communicator.set_gripper(closed=False)
                    if success:
                        logger.info(f"      開啟夾爪")
                        time.sleep(1.0)
                    else:
                        logger.error(f"      開啟夾爪失敗")
                
                # 關閉夾爪（抓取）
                elif action in ['gripper_close', 'grasp']:
                    success = self.communicator.set_gripper(closed=True)
                    if success:
                        logger.info(f"      關閉夾爪")
                        time.sleep(1.0)
                    else:
                        logger.error(f"      關閉夾爪失敗")
                
                # 等待動作
                elif action == 'wait_for_human':
                    logger.info(f"      等待人類...")
                    time.sleep(3.0)
                
                else:
                    logger.warning(f"      未知動作: {action}")
                
                # Spin ROS (使用 communicator)
                rclpy.spin_once(self.communicator, timeout_sec=0.1)
            
            logger.info(f"\n所有路徑點執行完成")
            return True
            
        except Exception as e:
            logger.error(f"\n執行失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _simulate_waypoints(self, waypoints: list) -> bool:
        """模擬執行路徑點"""
        for i, wp in enumerate(waypoints, 1):
            action = wp.get('action', 'move')
            logger.info(f"   [{i}/{len(waypoints)}] {action}")
            
            if action == 'move':
                position = wp.get('position', [])
                logger.info(f"      移動到: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}]")
            elif action == 'gripper_open':
                logger.info(f"      開啟夾爪")
            elif action == 'gripper_close':
                logger.info(f"      關閉夾爪")
        
        return True
    
