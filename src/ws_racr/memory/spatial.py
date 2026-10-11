"""Spatial memory: knowledge-graph-style store of object/human/robot nodes and their spatial relations (direction, distance, near/on)."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
from ws_racr.world_state.spatial_relations import SpatialRelationCalculator, SpatialRelation
logger = logging.getLogger(__name__)
SPATIAL_CALC_AVAILABLE = True


class NodeType(str, Enum):
    """KG 節點類型"""
    HUMAN = "Human"
    ROBOT = "Robot"
    TOOL = "Tool"
    PART = "Part"
    WORKSPACE_ZONE = "WorkspaceZone"


class RelationType(str, Enum):
    """KG 關係類型"""
    HUMAN_HOLDING = "human_holding"
    TOOL_NEAR = "tool_near"
    ROBOT_GRIPPER_NEAR = "robot_gripper_near"
    HANDOVER_READY = "handover_ready"
    STOP_SIGNAL = "stop_signal"
    IN_ZONE = "in_zone"
    ASSEMBLED_WITH = "assembled_with"


@dataclass
class KGNode:
    """Knowledge Graph 節點"""
    node_id: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'properties': self.properties,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class KGRelation:
    """Knowledge Graph 關係"""
    relation_id: str
    relation_type: RelationType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type.value,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'properties': self.properties,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat()
        }


class SpatialMemory:
    """
    空間記憶 - Knowledge Graph
    
    表示人-機器人-物體的關係狀態
    不是幾何地圖，而是動態信念狀態
    """
    
    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.relations: Dict[str, KGRelation] = {}
        
        # 初始化空間關係計算器
        if SPATIAL_CALC_AVAILABLE:
            self.spatial_calc = SpatialRelationCalculator()
        else:
            self.spatial_calc = None
    
    def add_node(self, node: KGNode):
        """添加節點"""
        self.nodes[node.node_id] = node
    
    def add_relation(self, relation: KGRelation):
        """添加關係"""
        self.relations[relation.relation_id] = relation
    
    def get_node(self, node_id: str) -> Optional[KGNode]:
        """取得節點"""
        return self.nodes.get(node_id)
    
    def get_relations_by_source(self, source_id: str) -> List[KGRelation]:
        """取得從某節點出發的所有關係"""
        return [r for r in self.relations.values() if r.source_id == source_id]
    
    def get_relations_by_target(self, target_id: str) -> List[KGRelation]:
        """取得指向某節點的所有關係"""
        return [r for r in self.relations.values() if r.target_id == target_id]
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[KGRelation]:
        """取得特定類型的所有關係"""
        return [r for r in self.relations.values() if r.relation_type == relation_type]
    
    def remove_relation(self, relation_id: str):
        """移除關係"""
        if relation_id in self.relations:
            del self.relations[relation_id]
    
    def get_subgraph(self, center_node_id: str, k_hops: int = 2) -> Tuple[Dict[str, KGNode], Dict[str, KGRelation]]:
        """
        取得以某節點為中心的 K-hop 子圖
        
        Args:
            center_node_id: 中心節點 ID
            k_hops: 跳數
        
        Returns:
            (nodes_dict, relations_dict)
        """
        visited_nodes = {center_node_id}
        current_frontier = {center_node_id}
        
        subgraph_relations = {}
        
        for hop in range(k_hops):
            next_frontier = set()
            
            for node_id in current_frontier:
                # 找出所有相關關係
                out_relations = self.get_relations_by_source(node_id)
                in_relations = self.get_relations_by_target(node_id)
                
                for rel in out_relations + in_relations:
                    subgraph_relations[rel.relation_id] = rel
                    
                    # 添加鄰居到下一輪
                    if rel.source_id not in visited_nodes:
                        next_frontier.add(rel.source_id)
                        visited_nodes.add(rel.source_id)
                    if rel.target_id not in visited_nodes:
                        next_frontier.add(rel.target_id)
                        visited_nodes.add(rel.target_id)
            
            current_frontier = next_frontier
            
            if not current_frontier:
                break
        
        subgraph_nodes = {nid: self.nodes[nid] for nid in visited_nodes if nid in self.nodes}
        
        return subgraph_nodes, subgraph_relations
    
    def to_text_context(self) -> str:
        """轉換為文字上下文（用於 LLM）"""
        text = "=== Current Knowledge Graph ===\n"
        
        # 按類型分組節點
        text += "\n[Entities]\n"
        for node_type in NodeType:
            nodes_of_type = [n for n in self.nodes.values() if n.node_type == node_type]
            if nodes_of_type:
                text += f"{node_type.value}s: {', '.join([n.node_id for n in nodes_of_type])}\n"
        
        # 列出關係
        text += "\n[Relations]\n"
        for rel in self.relations.values():
            text += f"- {rel.source_id} --[{rel.relation_type.value}]--> {rel.target_id}\n"
        
        return text
    
    def prune_isolated_nodes(self):
        """移除孤立節點（沒有任何關係的節點）"""
        connected_nodes = set()
        
        for rel in self.relations.values():
            connected_nodes.add(rel.source_id)
            connected_nodes.add(rel.target_id)
        
        # 移除不在連接集合中的節點
        isolated = set(self.nodes.keys()) - connected_nodes
        
        for node_id in isolated:
            del self.nodes[node_id]
        
        if isolated:
            logger.info(f"KG: Pruned {len(isolated)} isolated nodes")
    
    def update_spatial_relations(self, auto_compute: bool = True, force: bool = False):
        """
        更新所有物體間的空間關係（自動計算，帶性能優化）
        
        Args:
            auto_compute: 是否自動計算所有物體間關係
            force: 強制更新（忽略節流）
        
        性能特性:
        - 5 個物體: ~0.05 ms (<1% @30FPS)
        - 10 個物體: ~0.2 ms (<1% @30FPS)
        - 20 個物體: ~0.8 ms (2.4% @30FPS)
        - 建議: >15 個物體時使用節流或事件驅動
        """
        if not self.spatial_calc:
            return
        
        # 性能優化：自適應節流（根據物體數量調整）
        if not force:
            if not hasattr(self, '_spatial_update_counter'):
                self._spatial_update_counter = 0
                self._last_object_count = 0
            
            self._spatial_update_counter += 1
            
            # 根據物體數量自動調整節流頻率
            current_object_count = len([n for n in self.nodes.values() 
                                       if n.node_type in [NodeType.TOOL, NodeType.PART]])
            
            if current_object_count != self._last_object_count:
                # 物體數量變化時強制更新
                self._last_object_count = current_object_count
            else:
                # 自適應節流
                if current_object_count <= 10:
                    throttle = 3  # 快速更新
                elif current_object_count <= 20:
                    throttle = 10  # 中速更新（你的場景）
                else:
                    throttle = 15  # 慢速更新
                
                if self._spatial_update_counter % throttle != 0:
                    return
        
        # 收集所有有位置信息的物體節點
        objects = {}
        for node_id, node in self.nodes.items():
            if node.node_type in [NodeType.TOOL, NodeType.PART]:
                pos = node.properties.get('position')
                if pos and len(pos) >= 3:
                    objects[node_id] = {
                        'position': pos,
                        'type': node.properties.get('type', 'unknown')
                    }
        
        if len(objects) < 2:
            return  # 至少需要兩個物體才能計算關係
        
        # 計算所有關係
        new_relations = self.spatial_calc.compute_all_relations(objects)
        
        # 移除舊的空間關係（只保留非空間類型的關係）
        spatial_relation_types = [
            RelationType.TOOL_NEAR,
            RelationType.IN_ZONE,
        ]
        
        old_spatial_rels = []
        for rel_id, rel in self.relations.items():
            if rel.relation_type in spatial_relation_types:
                old_spatial_rels.append(rel_id)
        
        for rel_id in old_spatial_rels:
            del self.relations[rel_id]
        
        # 添加新的空間關係
        for rel_data in new_relations:
            # 只添加有意義的關係（近距離或明確方向）
            relation = rel_data['relation']
            distance = rel_data['distance']
            
            if relation in [SpatialRelation.NEAR, SpatialRelation.VERY_CLOSE]:
                rel_id = f"spatial_{rel_data['source_id']}_{rel_data['target_id']}_{datetime.now().timestamp()}"
                
                self.add_relation(KGRelation(
                    relation_id=rel_id,
                    relation_type=RelationType.TOOL_NEAR,  # 統一使用 TOOL_NEAR
                    source_id=rel_data['source_id'],
                    target_id=rel_data['target_id'],
                    properties={
                        'spatial_relation': relation.value,
                        'distance': distance,
                        'details': rel_data['details']
                    },
                    confidence=1.0 if relation == SpatialRelation.VERY_CLOSE else 0.8
                ))
    
    def query_spatial_relation(self, obj_a: str, obj_b: str) -> Optional[Dict]:
        """
        查詢兩個物體的空間關係
        
        Returns:
            {
                'success': bool,
                'relation': SpatialRelation,
                'distance': float,
                'description': str (中文)
            }
        """
        if not self.spatial_calc:
            return {'success': False}
        
        node_a = self.get_node(obj_a)
        node_b = self.get_node(obj_b)
        
        if not node_a or not node_b:
            return {'success': False}
        
        pos_a = node_a.properties.get('position')
        pos_b = node_b.properties.get('position')
        
        if not pos_a or not pos_b:
            return {'success': False}
        
        relation, distance, details = self.spatial_calc.compute_relative_position(pos_a, pos_b)
        
        return {
            'success': True,
            'relation': relation,
            'distance': distance,
            'description': self.spatial_calc.generate_natural_description(obj_a, obj_b, relation, distance),
            'details': details
        }
    
    def get_objects_near(self, center_id: str, radius: float = 200) -> List[Tuple[str, float]]:
        """
        取得在中心物體周圍的所有物體
        
        Returns:
            List of (obj_id, distance) sorted by distance
        """
        if not self.spatial_calc:
            return []
        
        # 收集所有物體位置
        objects = {}
        for node_id, node in self.nodes.items():
            if node.node_type in [NodeType.TOOL, NodeType.PART]:
                pos = node.properties.get('position')
                if pos:
                    objects[node_id] = {'position': pos}
        
        return self.spatial_calc.get_objects_near(objects, center_id, radius)
