#!/usr/bin/env python3
"""
空間關係計算模組
自動計算物體間的相對位置關係並更新 Knowledge Graph
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from enum import Enum


class SpatialRelation(str, Enum):
    """空間關係類型"""
    # 方向關係
    FRONT_OF = "front_of"       # A 在 B 前方
    BACK_OF = "back_of"         # A 在 B 後方
    LEFT_OF = "left_of"         # A 在 B 左方
    RIGHT_OF = "right_of"       # A 在 B 右方
    ABOVE = "above"             # A 在 B 上方
    BELOW = "below"             # A 在 B 下方
    
    # 距離關係
    NEAR = "near"               # A 靠近 B（<200mm）
    FAR = "far"                 # A 遠離 B（>500mm）
    VERY_CLOSE = "very_close"   # A 非常靠近 B（<100mm）
    
    # 區域關係
    IN_REGION = "in_region"     # A 在某個工作區域
    ON_SURFACE = "on_surface"   # A 在 B 表面上
    
    # 複合關係
    FRONT_LEFT = "front_left"   # A 在 B 前左方
    FRONT_RIGHT = "front_right" # A 在 B 前右方
    BACK_LEFT = "back_left"     # A 在 B 後左方
    BACK_RIGHT = "back_right"   # A 在 B 後右方


class SpatialRelationCalculator:
    """空間關係計算器"""
    
    def __init__(self):
        # 閾值配置
        self.thresholds = {
            'very_close': 100,   # mm
            'near': 200,         # mm
            'far': 500,          # mm
            'directional': 150,  # mm，方向判斷的最小顯著距離
            'vertical': 50,      # mm，垂直方向的閾值
        }
        
        # 工作區域定義（機器人基座標系）
        self.workspace_regions = {
            'assembly_area': {'x': [200, 500], 'y': [-200, 200], 'z': [0, 300]},
            'tool_tray': {'x': [300, 600], 'y': [200, 400], 'z': [0, 100]},
            'human_zone': {'x': [-200, 200], 'y': [-400, -200], 'z': [0, 500]},
            'robot_safe_zone': {'x': [0, 400], 'y': [-100, 100], 'z': [100, 600]},
        }
    
    def compute_relative_position(
        self,
        pos_a: List[float],
        pos_b: List[float]
    ) -> Tuple[SpatialRelation, float, Dict]:
        """
        計算物體 A 相對於物體 B 的空間關係
        
        Args:
            pos_a: 物體 A 的 3D 位置 [x, y, z]（mm）
            pos_b: 物體 B 的 3D 位置 [x, y, z]（mm）
        
        Returns:
            (relation, distance, details)
            - relation: 主要空間關係
            - distance: 3D 歐式距離（mm）
            - details: 詳細信息字典
        """
        a = np.array(pos_a)
        b = np.array(pos_b)
        
        # 計算向量差
        delta = a - b
        dx, dy, dz = delta
        
        # 計算距離
        dist_3d = np.linalg.norm(delta)
        dist_xy = np.sqrt(dx**2 + dy**2)
        
        # 詳細信息
        details = {
            'delta_x': float(dx),
            'delta_y': float(dy),
            'delta_z': float(dz),
            'distance_3d': float(dist_3d),
            'distance_xy': float(dist_xy),
            'angle_deg': float(np.degrees(np.arctan2(dy, dx))),
        }
        
        # === 1. 距離關係判斷 ===
        if dist_3d < self.thresholds['very_close']:
            relation = SpatialRelation.VERY_CLOSE
        elif dist_3d < self.thresholds['near']:
            relation = SpatialRelation.NEAR
        elif dist_3d > self.thresholds['far']:
            relation = SpatialRelation.FAR
        else:
            # === 2. 方向關係判斷（中距離） ===
            relation = self._compute_directional_relation(dx, dy, dz)
        
        return relation, dist_3d, details
    
    def _compute_directional_relation(
        self,
        dx: float,
        dy: float,
        dz: float
    ) -> SpatialRelation:
        """計算方向關係"""
        # 水平面上的主要方向
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        abs_dz = abs(dz)
        
        # 垂直方向優先
        if abs_dz > self.thresholds['vertical'] and abs_dz > max(abs_dx, abs_dy):
            return SpatialRelation.ABOVE if dz > 0 else SpatialRelation.BELOW
        
        # 水平方向（判斷主要軸）
        if abs_dx > abs_dy:
            # X 軸為主
            if dx > self.thresholds['directional']:
                # 前方（可能帶左右）
                if abs_dy > 50:
                    return SpatialRelation.FRONT_LEFT if dy > 0 else SpatialRelation.FRONT_RIGHT
                return SpatialRelation.FRONT_OF
            elif dx < -self.thresholds['directional']:
                # 後方（可能帶左右）
                if abs_dy > 50:
                    return SpatialRelation.BACK_LEFT if dy > 0 else SpatialRelation.BACK_RIGHT
                return SpatialRelation.BACK_OF
        else:
            # Y 軸為主
            if dy > self.thresholds['directional']:
                return SpatialRelation.LEFT_OF
            elif dy < -self.thresholds['directional']:
                return SpatialRelation.RIGHT_OF
        
        # 預設：靠近
        return SpatialRelation.NEAR
    
    def compute_all_relations(
        self,
        objects: Dict[str, Dict]
    ) -> List[Dict]:
        """
        計算所有物體間的空間關係
        
        Args:
            objects: {obj_id: {'position': [x,y,z], 'type': str, ...}, ...}
        
        Returns:
            List of relations: [
                {
                    'source_id': str,
                    'target_id': str,
                    'relation': SpatialRelation,
                    'distance': float,
                    'details': Dict
                },
                ...
            ]
        """
        relations = []
        
        obj_ids = list(objects.keys())
        
        # 兩兩計算（排除自身）
        for i, id_a in enumerate(obj_ids):
            obj_a = objects[id_a]
            pos_a = obj_a.get('position')
            
            if pos_a is None or len(pos_a) < 3:
                continue
            
            for id_b in obj_ids[i+1:]:
                obj_b = objects[id_b]
                pos_b = obj_b.get('position')
                
                if pos_b is None or len(pos_b) < 3:
                    continue
                
                # 計算雙向關係
                rel_ab, dist, details = self.compute_relative_position(pos_a, pos_b)
                
                relations.append({
                    'source_id': id_a,
                    'target_id': id_b,
                    'relation': rel_ab,
                    'distance': dist,
                    'details': details
                })
                
                # 反向關係
                rel_ba = self._get_inverse_relation(rel_ab)
                relations.append({
                    'source_id': id_b,
                    'target_id': id_a,
                    'relation': rel_ba,
                    'distance': dist,
                    'details': {
                        'delta_x': -details['delta_x'],
                        'delta_y': -details['delta_y'],
                        'delta_z': -details['delta_z'],
                        'distance_3d': details['distance_3d'],
                        'distance_xy': details['distance_xy'],
                        'angle_deg': (details['angle_deg'] + 180) % 360,
                    }
                })
        
        return relations
    
    def _get_inverse_relation(self, relation: SpatialRelation) -> SpatialRelation:
        """取得反向關係"""
        inverse_map = {
            SpatialRelation.FRONT_OF: SpatialRelation.BACK_OF,
            SpatialRelation.BACK_OF: SpatialRelation.FRONT_OF,
            SpatialRelation.LEFT_OF: SpatialRelation.RIGHT_OF,
            SpatialRelation.RIGHT_OF: SpatialRelation.LEFT_OF,
            SpatialRelation.ABOVE: SpatialRelation.BELOW,
            SpatialRelation.BELOW: SpatialRelation.ABOVE,
            SpatialRelation.FRONT_LEFT: SpatialRelation.BACK_RIGHT,
            SpatialRelation.FRONT_RIGHT: SpatialRelation.BACK_LEFT,
            SpatialRelation.BACK_LEFT: SpatialRelation.FRONT_RIGHT,
            SpatialRelation.BACK_RIGHT: SpatialRelation.FRONT_LEFT,
        }
        
        # 對稱關係（沒有反向）
        if relation in [SpatialRelation.NEAR, SpatialRelation.FAR, SpatialRelation.VERY_CLOSE]:
            return relation
        
        return inverse_map.get(relation, relation)
    
    def check_in_region(
        self,
        position: List[float],
        region_name: str
    ) -> bool:
        """檢查物體是否在指定工作區域內"""
        if region_name not in self.workspace_regions:
            return False
        
        region = self.workspace_regions[region_name]
        x, y, z = position
        
        in_x = region['x'][0] <= x <= region['x'][1]
        in_y = region['y'][0] <= y <= region['y'][1]
        in_z = region['z'][0] <= z <= region['z'][1]
        
        return in_x and in_y and in_z
    
    def get_objects_in_region(
        self,
        objects: Dict[str, Dict],
        region_name: str
    ) -> List[str]:
        """取得在指定區域內的所有物體"""
        obj_ids = []
        
        for obj_id, obj_data in objects.items():
            pos = obj_data.get('position')
            if pos and self.check_in_region(pos, region_name):
                obj_ids.append(obj_id)
        
        return obj_ids
    
    def get_objects_near(
        self,
        objects: Dict[str, Dict],
        center_id: str,
        radius: float = 200
    ) -> List[Tuple[str, float]]:
        """
        取得在中心物體周圍的所有物體
        
        Returns:
            List of (obj_id, distance) sorted by distance
        """
        if center_id not in objects:
            return []
        
        center_pos = objects[center_id].get('position')
        if center_pos is None:
            return []
        
        nearby = []
        center = np.array(center_pos)
        
        for obj_id, obj_data in objects.items():
            if obj_id == center_id:
                continue
            
            pos = obj_data.get('position')
            if pos is None:
                continue
            
            dist = np.linalg.norm(np.array(pos) - center)
            
            if dist <= radius:
                nearby.append((obj_id, float(dist)))
        
        # 按距離排序
        nearby.sort(key=lambda x: x[1])
        
        return nearby
    
    def relation_to_chinese(self, relation: SpatialRelation) -> str:
        """將空間關係轉換為中文描述"""
        chinese_map = {
            SpatialRelation.FRONT_OF: "前方",
            SpatialRelation.BACK_OF: "後方",
            SpatialRelation.LEFT_OF: "左方",
            SpatialRelation.RIGHT_OF: "右方",
            SpatialRelation.ABOVE: "上方",
            SpatialRelation.BELOW: "下方",
            SpatialRelation.NEAR: "附近",
            SpatialRelation.FAR: "遠處",
            SpatialRelation.VERY_CLOSE: "非常靠近",
            SpatialRelation.FRONT_LEFT: "前左方",
            SpatialRelation.FRONT_RIGHT: "前右方",
            SpatialRelation.BACK_LEFT: "後左方",
            SpatialRelation.BACK_RIGHT: "後右方",
            SpatialRelation.IN_REGION: "在區域內",
            SpatialRelation.ON_SURFACE: "在表面上",
        }
        
        return chinese_map.get(relation, relation.value)
    
    def generate_natural_description(
        self,
        source_id: str,
        target_id: str,
        relation: SpatialRelation,
        distance: float
    ) -> str:
        """生成自然語言描述"""
        rel_zh = self.relation_to_chinese(relation)
        
        if relation in [SpatialRelation.VERY_CLOSE, SpatialRelation.NEAR, SpatialRelation.FAR]:
            return f"{source_id} 在 {target_id} {rel_zh}（距離 {distance:.0f} mm）"
        else:
            return f"{source_id} 在 {target_id} 的 {rel_zh}"


# ============================================================================
# 測試與示例
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Spatial Relations Calculator\n")
    
    calc = SpatialRelationCalculator()
    
    # 測試 1: 兩個物體的相對位置
    print("=" * 70)
    print("Test 1: Compute relative position")
    print("=" * 70)
    
    pos_wrench = [300, 100, 50]  # 六角扳手
    pos_screw = [500, 150, 50]   # 螺絲
    
    relation, dist, details = calc.compute_relative_position(pos_wrench, pos_screw)
    
    print(f"六角扳手位置: {pos_wrench}")
    print(f"螺絲位置: {pos_screw}")
    print(f"相對關係: {relation.value} ({calc.relation_to_chinese(relation)})")
    print(f"距離: {dist:.1f} mm")
    print(f"詳細: delta=({details['delta_x']:.1f}, {details['delta_y']:.1f}, {details['delta_z']:.1f})")
    
    desc = calc.generate_natural_description("六角扳手", "螺絲", relation, dist)
    print(f"自然語言: {desc}")
    
    # 測試 2: 多物體關係計算
    print("\n" + "=" * 70)
    print("Test 2: Compute all relations")
    print("=" * 70)
    
    objects = {
        'hex_wrench': {'position': [300, 100, 50], 'type': 'tool'},
        'screwdriver': {'position': [350, 80, 50], 'type': 'tool'},
        'platform': {'position': [400, 0, 20], 'type': 'part'},
        'bracket': {'position': [450, -50, 60], 'type': 'part'},
    }
    
    relations = calc.compute_all_relations(objects)
    
    print(f"找到 {len(relations)} 個空間關係:\n")
    for rel in relations[:10]:  # 只顯示前10個
        source = rel['source_id']
        target = rel['target_id']
        relation = rel['relation']
        dist = rel['distance']
        
        desc = calc.generate_natural_description(source, target, relation, dist)
        print(f"  • {desc}")
    
    # 測試 3: 區域檢查
    print("\n" + "=" * 70)
    print("Test 3: Region checks")
    print("=" * 70)
    
    for obj_id, obj_data in objects.items():
        pos = obj_data['position']
        print(f"\n{obj_id} @ {pos}")
        
        for region_name in calc.workspace_regions.keys():
            is_in = calc.check_in_region(pos, region_name)
            status = "✅" if is_in else "❌"
            print(f"  {status} {region_name}")
    
    # 測試 4: 鄰近物體查詢
    print("\n" + "=" * 70)
    print("Test 4: Find nearby objects")
    print("=" * 70)
    
    center_id = 'hex_wrench'
    nearby = calc.get_objects_near(objects, center_id, radius=200)
    
    print(f"在 {center_id} 200mm 範圍內的物體:")
    for obj_id, dist in nearby:
        print(f"  • {obj_id}: {dist:.1f} mm")
