#!/usr/bin/env python3
"""
Object Memory System - 物體長期記憶系統
維護桌面物體的持久化狀態，即使被遮擋也不會遺忘
"""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ObjectMemory:
    """單個物體的記憶"""
    object_id: str
    object_class: str  # hex_wrench, screw, cylinder, etc.
    
    # 空間信息
    position: List[float]  # [x, y, z] in robot base frame
    orientation: Optional[float] = None  # yaw angle (optional)
    
    # 狀態信息
    visible: bool = True
    confidence: float = 1.0
    last_seen_ts: float = 0.0
    first_seen_ts: float = 0.0
    
    # 歷史信息
    detection_count: int = 1
    occlusion_count: int = 0  # 被遮擋次數
    
    # 元數據
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_seen_ts == 0.0:
            self.last_seen_ts = time.time()
        if self.first_seen_ts == 0.0:
            self.first_seen_ts = time.time()


class ObjectMemorySystem:
    """
    物體記憶系統
    
    功能：
    1. 維護桌面物體的長期狀態
    2. 處理遮擋（confidence decay）
    3. 支持空間查詢（哪些物體在某個區域）
    4. 支持語義查詢（"桌上還有哪些物品"）
    """
    
    def __init__(
        self,
        confidence_decay_rate: float = 0.95,  # 每秒衰減率
        invisible_threshold: float = 0.1,  # 低於此值標記為 invisible
        persistence_path: Optional[str] = None  # 持久化路徑
    ):
        """
        Args:
            confidence_decay_rate: 信心度衰減率（每秒）
            invisible_threshold: 信心度低於此值標記為不可見
            persistence_path: JSON 持久化路徑（可選）
        """
        self.memories: Dict[str, ObjectMemory] = {}  # object_id -> ObjectMemory
        self.decay_rate = confidence_decay_rate
        self.invisible_threshold = invisible_threshold
        self.persistence_path = persistence_path
        
        # 統計
        self.total_updates = 0
        self.total_queries = 0
        
        # 載入持久化數據（如果存在）
        if persistence_path and Path(persistence_path).exists():
            self.load_from_file(persistence_path)
        
        logger.info("Object Memory System Initialized")
        logger.info(f"   - Decay rate: {confidence_decay_rate}/sec")
        logger.info(f"   - Invisible threshold: {invisible_threshold}")
    
    def update_from_detection(
        self,
        detections: List[Dict],
        timestamp: Optional[float] = None
    ):
        """
        從檢測結果更新記憶
        
        Args:
            detections: YOLO 檢測結果列表
                [
                    {
                        'class': str,
                        'confidence': float,
                        'bbox': [x, y, w, h],
                        'position': [x, y, z],  # 3D position (optional)
                        'object_id': str  # 唯一標識（可選）
                    },
                    ...
                ]
            timestamp: 時間戳（如果為 None 則使用當前時間）
        """
        if timestamp is None:
            timestamp = time.time()
        
        detected_ids = set()
        
        for det in detections:
            obj_class = det['class']
            position = det.get('position', [0, 0, 0])
            confidence = det['confidence']
            obj_id = det.get('object_id')
            
            # 如果沒有提供 object_id，根據位置匹配或創建新的
            if obj_id is None:
                obj_id = self._find_or_create_id(obj_class, position, timestamp)
            
            detected_ids.add(obj_id)
            
            # 更新或創建記憶
            if obj_id in self.memories:
                mem = self.memories[obj_id]
                mem.position = position
                mem.visible = True
                mem.confidence = min(1.0, mem.confidence + 0.1)  # 信心度增強
                mem.last_seen_ts = timestamp
                mem.detection_count += 1
            else:
                self.memories[obj_id] = ObjectMemory(
                    object_id=obj_id,
                    object_class=obj_class,
                    position=position,
                    visible=True,
                    confidence=confidence,
                    last_seen_ts=timestamp,
                    first_seen_ts=timestamp
                )
        
        # 衰減未檢測到的物體
        self._decay_undetected(detected_ids, timestamp)
        
        self.total_updates += 1
    
    def _find_or_create_id(
        self,
        obj_class: str,
        position: List[float],
        timestamp: float,
        position_threshold: float = 0.05  # 5cm 內視為同一物體
    ) -> str:
        """
        根據類別和位置找到匹配的物體 ID，或創建新的
        """
        # 搜索同類別的物體
        for obj_id, mem in self.memories.items():
            if mem.object_class == obj_class and mem.visible:
                # 計算距離
                dist = np.linalg.norm(
                    np.array(position) - np.array(mem.position)
                )
                if dist < position_threshold:
                    return obj_id
        
        # 創建新 ID
        count = sum(1 for m in self.memories.values() if m.object_class == obj_class)
        return f"{obj_class}_{count + 1}"
    
    def _decay_undetected(self, detected_ids: set, timestamp: float):
        """衰減未檢測到的物體的信心度"""
        for obj_id, mem in self.memories.items():
            if obj_id not in detected_ids and mem.visible:
                # 計算時間差
                dt = timestamp - mem.last_seen_ts
                
                # 指數衰減
                decay_factor = self.decay_rate ** dt
                mem.confidence *= decay_factor
                
                # 檢查是否低於閾值
                if mem.confidence < self.invisible_threshold:
                    mem.visible = False
                    mem.occlusion_count += 1
    
    def query_inventory(self, visible_only: bool = True) -> List[Dict]:
        """
        查詢桌面庫存
        
        Args:
            visible_only: 只返回可見物體
        
        Returns:
            物體列表，按信心度排序
        """
        self.total_queries += 1
        
        results = []
        for mem in self.memories.values():
            if visible_only and not mem.visible:
                continue
            
            results.append({
                'object_id': mem.object_id,
                'class': mem.object_class,
                'position': mem.position,
                'confidence': mem.confidence,
                'visible': mem.visible,
                'last_seen': mem.last_seen_ts
            })
        
        # 按信心度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results
    
    def locate_object(
        self,
        object_class: str,
        visible_only: bool = True
    ) -> Optional[Dict]:
        """
        定位特定類別的物體
        
        Args:
            object_class: 物體類別
            visible_only: 只查找可見物體
        
        Returns:
            物體信息或 None
        """
        candidates = [
            mem for mem in self.memories.values()
            if mem.object_class == object_class and (not visible_only or mem.visible)
        ]
        
        if not candidates:
            return None
        
        # 返回信心度最高的
        best = max(candidates, key=lambda m: m.confidence)
        return {
            'object_id': best.object_id,
            'class': best.object_class,
            'position': best.position,
            'confidence': best.confidence,
            'visible': best.visible
        }
    
    def query_region(
        self,
        center: List[float],
        radius: float,
        visible_only: bool = True
    ) -> List[Dict]:
        """
        查詢某個區域內的物體
        
        Args:
            center: 區域中心 [x, y, z]
            radius: 半徑（米）
            visible_only: 只返回可見物體
        
        Returns:
            區域內的物體列表
        """
        center_np = np.array(center)
        results = []
        
        for mem in self.memories.values():
            if visible_only and not mem.visible:
                continue
            
            dist = np.linalg.norm(np.array(mem.position) - center_np)
            if dist <= radius:
                results.append({
                    'object_id': mem.object_id,
                    'class': mem.object_class,
                    'position': mem.position,
                    'distance': float(dist),
                    'confidence': mem.confidence
                })
        
        # 按距離排序
        results.sort(key=lambda x: x['distance'])
        return results
    
    def reset(self, reason: str = "manual"):
        """重置記憶系統"""
        logger.info(f"Resetting object memory: {reason}")
        self.memories.clear()
        self.total_updates = 0
        self.total_queries = 0
    
    def save_to_file(self, path: Optional[str] = None):
        """保存到 JSON 文件"""
        save_path = path or self.persistence_path
        if not save_path:
            return
        
        data = {
            'memories': {
                obj_id: asdict(mem)
                for obj_id, mem in self.memories.items()
            },
            'stats': {
                'total_updates': self.total_updates,
                'total_queries': self.total_queries,
                'total_objects': len(self.memories)
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Object memory saved to {save_path}")
    
    def load_from_file(self, path: str):
        """從 JSON 文件載入"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.memories = {
            obj_id: ObjectMemory(**mem_data)
            for obj_id, mem_data in data['memories'].items()
        }
        
        stats = data.get('stats', {})
        self.total_updates = stats.get('total_updates', 0)
        self.total_queries = stats.get('total_queries', 0)
        
        logger.info(f"Loaded {len(self.memories)} objects from {path}")
    
    def get_statistics(self) -> Dict:
        """獲取統計信息"""
        visible_count = sum(1 for m in self.memories.values() if m.visible)
        
        return {
            'total_objects': len(self.memories),
            'visible_objects': visible_count,
            'invisible_objects': len(self.memories) - visible_count,
            'total_updates': self.total_updates,
            'total_queries': self.total_queries,
            'avg_confidence': np.mean([m.confidence for m in self.memories.values()]) if self.memories else 0.0
        }
