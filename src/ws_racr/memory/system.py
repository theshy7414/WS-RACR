"""HRCMemorySystem: facade that owns the temporal log, spatial memory, interaction log, episodic cache and semantic rules, and syncs episodes to PostgreSQL when a persistence backend is attached."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging
import numpy as np
logger = logging.getLogger(__name__)
SPATIAL_CALC_AVAILABLE = True
from ws_racr.memory.types import (
    EpisodicMemory,
    MemoryQuery,
    SemanticRule,
    StepSummary,
    StepTransitionMemory,
)
from ws_racr.memory.interaction_log import InteractionLog, InteractionRecord
from ws_racr.memory.temporal import TemporalMemory
from ws_racr.memory.spatial import KGNode, KGRelation, NodeType, RelationType, SpatialMemory
from ws_racr.memory.preprocessor import InformationPreprocessor


class HRCMemorySystem:
    """
    整合的人機協作記憶系統
    
    架構：單一記憶接口，雙層存儲後端
    - RAM Backend: Temporal Memory + KG + 近期 Episodes (運行時快速訪問)
    - DB Backend: PerceptionLogger (持久化 + 跨 session + 離線分析)
    
    整合所有記憶模組:
    - Information Preprocessor
    - Temporal Memory (RAM)
    - Spatial Memory (KG, RAM)
    - Episodic Memory (RAM cache + DB persistence)
    - Semantic Memory (RAM cache + DB persistence)
    """
    
    def __init__(self, temporal_buffer_size: int = 50, persistence_backend=None, 
                 llm_compressor=None, embedding_model=None):
        """
        初始化記憶系統
        
        Args:
            temporal_buffer_size: Temporal Memory 緩衝大小
            persistence_backend: 持久化後端（PerceptionLogger 實例），可選
            llm_compressor: LLM 壓縮器（用於 Temporal Memory 壓縮），可選
            embedding_model: 向量編碼模型（用於 Episodic Memory 相似度搜索），可選
        """
        logger.info("初始化 HRC Memory System...")
        
        # 核心組件 (RAM Backend)
        self.preprocessor = InformationPreprocessor()
        self.temporal_memory = TemporalMemory(max_length=temporal_buffer_size)
        self.spatial_memory = SpatialMemory()
        
        # 情節記憶與語義記憶（RAM cache）
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_rules: Dict[str, SemanticRule] = {}
        
        # LLM & Embedding Models
        self.llm_compressor = llm_compressor
        self.embedding_model = embedding_model
        
        # 將 LLM compressor 傳給 Temporal Memory
        if llm_compressor:
            self.temporal_memory.llm_compressor = llm_compressor
        
        # Episode embeddings cache
        self.episode_embeddings: Dict[str, np.ndarray] = {}
        
        # Persistence Backend (DB Backend)
        self.persistence_backend = persistence_backend
        if persistence_backend:
            logger.info("   持久化後端已連接")
            # 嘗試繼承 PerceptionLogger 的 embedding_model
            if embedding_model is None and hasattr(persistence_backend, 'embedding_model'):
                self.embedding_model = persistence_backend.embedding_model
                logger.info("   繼承 PerceptionLogger 的 embedding_model")
            self._load_from_persistence()
        else:
            logger.warning("   無持久化後端（純內存模式）")
        
        # Step transition memory (ordered list, newest last)
        self.step_transitions: List[StepTransitionMemory] = []

        # Interaction log — records every Execute / Ask / Answer decision
        self.interaction_log = InteractionLog(maxlen=20)

        # 統計
        self.update_count = 0
        self.last_sync_time = datetime.now()
    
    def update(self, world_state: Dict) -> Tuple[StepSummary, MemoryQuery]:
        """
        主更新循環 - 每次感知後呼叫
        
        Args:
            world_state: 來自 WorldStateBuilder 的當前狀態
        
        Returns:
            (step_summary, memory_query) - 可用於後續決策
        """
        # 1. Information Preprocessing
        step_summary, memory_query = self.preprocessor.process(world_state)
        
        # 2. Update Temporal Memory
        self.temporal_memory.add(step_summary)
        
        # 3. Update Spatial Memory (KG) - 在這裡實現動態更新
        self._update_spatial_memory(world_state, step_summary)
        
        self.update_count += 1
        
        return step_summary, memory_query
    
    def _update_spatial_memory(self, world_state: Dict, step_summary: StepSummary):
        """
        更新 Knowledge Graph
        
        根據 RoboMemory Algorithm 1:
        1. Retrieve relevant subgraph
        2. K-hop expansion
        3. Inject new relations
        4. Conflict resolution
        5. Merge back & prune
        """
        # 確保人類和機器人節點存在
        if 'human' not in self.spatial_memory.nodes:
            self.spatial_memory.add_node(KGNode(
                node_id='human',
                node_type=NodeType.HUMAN,
                properties={'state': step_summary.human_state.value}
            ))
        
        if 'robot' not in self.spatial_memory.nodes:
            self.spatial_memory.add_node(KGNode(
                node_id='robot',
                node_type=NodeType.ROBOT,
                properties={}
            ))
        
        # 更新人類狀態
        human_node = self.spatial_memory.get_node('human')
        if human_node:
            human_node.properties['state'] = step_summary.human_state.value
            human_node.last_updated = datetime.now()
        
        # 添加物件節點 (objects 是 Dict[str, ObjectState], 不是 list)
        for obj_id, obj in world_state.get('objects', {}).items():
            obj_type = obj.get('object_type', 'unknown')
            
            if obj_id not in self.spatial_memory.nodes:
                node_type = NodeType.TOOL if 'tool' in obj_type.lower() else NodeType.PART
                
                self.spatial_memory.add_node(KGNode(
                    node_id=obj_id,
                    node_type=node_type,
                    properties={
                        'type': obj_type,
                        'position': obj.get('position_3d', [0, 0, 0])
                    }
                ))
        
        # 根據手勢更新關係
        if step_summary.handover_intent:
            # 添加 handover_ready 關係
            rel_id = f"handover_{datetime.now().timestamp()}"
            self.spatial_memory.add_relation(KGRelation(
                relation_id=rel_id,
                relation_type=RelationType.HANDOVER_READY,
                source_id='human',
                target_id='robot',
                confidence=0.9
            ))
        
        # 移除過時的關係（例如超過 5 秒的 handover_ready）
        self._prune_old_relations(max_age_seconds=5.0)
        
        # 自動更新物體間空間關係（自適應節流）
        self.spatial_memory.update_spatial_relations(auto_compute=True, force=False)
    
    def _prune_old_relations(self, max_age_seconds: float):
        """移除過時的關係"""
        now = datetime.now()
        to_remove = []
        
        for rel_id, rel in self.spatial_memory.relations.items():
            age = (now - rel.last_updated).total_seconds()
            if age > max_age_seconds:
                to_remove.append(rel_id)
        
        for rel_id in to_remove:
            self.spatial_memory.remove_relation(rel_id)
    
    def add_episodic_memory(self, episode: EpisodicMemory):
        """
        添加情節記憶（雙層存儲）
        
        Args:
            episode: 情節記憶實例
        """
        # 1. 添加到 RAM cache
        self.episodic_memories[episode.event_id] = episode
        
        # 2. 生成 embedding（如果有 embedding_model）
        if self.embedding_model:
            try:
                episode_text = f"{episode.event_type.value} {episode.robot_action} {episode.human_response} {episode.outcome.value}"
                embedding = self.embedding_model.encode(episode_text)
                self.episode_embeddings[episode.event_id] = embedding
            except Exception as e:
                logger.warning(f"   Episode embedding 生成失敗: {e}")
        
        # 3. 持久化到 DB backend（如果有）
        if self.persistence_backend:
            try:
                self.persistence_backend.add_knowledge_fragment(
                    fragment_type='episodic_memory',
                    title=f"{episode.event_type.value}: {episode.robot_action}",
                    content=f"""
Event Type: {episode.event_type.value}
Step: {episode.step}
Tool: {episode.tool or 'N/A'}
Zone: {episode.zone or 'N/A'}
Human State: {episode.human_state.value}

Robot Action: {episode.robot_action}
Human Response: {episode.human_response}
Outcome: {episode.outcome.value}

Duration: {episode.duration_seconds}s
Timestamp: {episode.timestamp.isoformat()}
                    """.strip(),
                    source_type='memory_system',
                    source_data=episode.metadata,
                    confidence=0.9,
                    tags=[episode.event_type.value, episode.step, 'episodic']
                )
            except Exception as e:
                logger.warning(f"   Episodic memory 持久化失敗: {e}")
    
    def retrieve_similar_episodes(self, query: MemoryQuery, top_k: int = 5) -> List[EpisodicMemory]:
        """
        檢索相似的情節記憶
        
        使用 embedding + cosine similarity 或回退到時間排序
        """
        if not self.episodic_memories:
            return []
        
        query_text = query.query_text
        
        # 如果有 embedding_model，使用向量相似度搜索
        if self.embedding_model and self.episode_embeddings:
            try:
                # 1. 生成 query embedding
                query_embedding = self.embedding_model.encode(query_text)
                
                # 2. 計算與所有 episode 的相似度
                similarities = []
                for episode_id, episode in self.episodic_memories.items():
                    if episode_id in self.episode_embeddings:
                        episode_embedding = self.episode_embeddings[episode_id]
                        # Cosine similarity
                        similarity = np.dot(query_embedding, episode_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(episode_embedding)
                        )
                        similarities.append((similarity, episode))
                
                # 3. 排序並返回 top_k
                if similarities:
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    return [ep for _, ep in similarities[:top_k]]
            
            except Exception as e:
                logger.warning(f"   Vector search 失敗: {e}, 回退到時間排序")
        
        # 回退：按時間排序返回最近的 K 個
        sorted_episodes = sorted(
            self.episodic_memories.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )
        
        return sorted_episodes[:top_k]
    
    def record_step_transition(self, transition: 'StepTransitionMemory'):
        """Store a step transition record in RAM and optionally persist to DB.

        Args:
            transition: StepTransitionMemory instance built at the moment of
                        step change detection in the main perception loop.
        """
        self.step_transitions.append(transition)

        # Persist to DB backend when available
        if self.persistence_backend:
            try:
                self.persistence_backend.add_knowledge_fragment(
                    fragment_type='step_transition',
                    title=f"Step {transition.from_step}->{transition.to_step}: "
                          f"{transition.from_step_class}->{transition.to_step_class}",
                    content=transition.to_text(),
                    source_type='memory_system',
                    source_data=transition.to_dict(),
                    confidence=1.0,
                    tags=['step_transition', transition.to_step_class, f'step_{transition.to_step}']
                )
            except Exception as e:
                logger.warning(f"   StepTransition 持久化失敗: {e}")

    def get_transitions_to_step(self, to_step: int, limit: int = 5) -> List['StepTransitionMemory']:
        """Retrieve past transition records that led to a given step.

        Used by LLM reasoning: before deciding how to assist at step N,
        retrieve what happened in previous transitions into step N.

        Args:
            to_step: Target step index to filter by.
            limit: Maximum number of records to return (most recent first).

        Returns:
            List of matching StepTransitionMemory, newest first.
        """
        matches = [
            t for t in reversed(self.step_transitions)
            if t.to_step == to_step
        ]
        return matches[:limit]

    def get_recent_transitions(self, limit: int = 5) -> List['StepTransitionMemory']:
        """Return the most recent step transitions regardless of step index."""
        return list(reversed(self.step_transitions[-limit:]))

    def add_semantic_rule(self, rule: SemanticRule):
        """
        添加語義規則（雙層存儲）
        
        Args:
            rule: 語義規則實例
        """
        # 1. 添加到 RAM cache
        self.semantic_rules[rule.rule_id] = rule
        
        # 2. 持久化到 DB backend（如果有）
        if self.persistence_backend:
            try:
                self.persistence_backend.add_knowledge_fragment(
                    fragment_type='semantic_rule',
                    title=rule.rule_text[:100],
                    content=rule.rule_text,
                    source_type='memory_system_learning',
                    source_data={
                        'supporting_episodes': rule.supporting_episodes,
                        'counter_episodes': rule.counter_episodes,
                        'created_at': rule.created_at.isoformat(),
                        'last_updated': rule.last_updated.isoformat()
                    },
                    confidence=rule.confidence,
                    tags=['learned_rule', 'automated', 'semantic']
                )
            except Exception as e:
                logger.warning(f"   Semantic rule 持久化失敗: {e}")
    
    def _load_from_persistence(self):
        """啟動時從 DB backend 載入長期記憶"""
        if not self.persistence_backend:
            return
        
        logger.info("   從持久化後端載入記憶...")
        
        try:
            # 載入 Semantic Rules
            from sqlalchemy import text
            with self.persistence_backend.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT fragment_id, content, confidence, source_data, created_at
                    FROM knowledge_fragments
                    WHERE fragment_type = 'semantic_rule'
                      AND confidence >= 0.7
                    ORDER BY created_at DESC
                    LIMIT 20
                """))
                
                loaded_rules = 0
                for row in result:
                    try:
                        import json
                        source_data = json.loads(row.source_data) if row.source_data else {}
                        
                        rule = SemanticRule(
                            rule_id=row.fragment_id,
                            rule_text=row.content,
                            confidence=row.confidence,
                            supporting_episodes=source_data.get('supporting_episodes', []),
                            counter_episodes=source_data.get('counter_episodes', []),
                            created_at=row.created_at,
                            last_updated=row.created_at
                        )
                        self.semantic_rules[rule.rule_id] = rule
                        loaded_rules += 1
                    except Exception as e:
                        continue
                
                if loaded_rules > 0:
                    logger.info(f"      載入 {loaded_rules} 條語義規則")
        
        except Exception as e:
            logger.warning(f"      載入失敗: {e}")
    
    def sync_to_persistence(self, force: bool = False):
        """
        同步到持久化後端
        
        Args:
            force: 強制同步（忽略時間間隔）
        """
        if not self.persistence_backend:
            return
        
        # 檢查同步間隔（預設 5 分鐘）
        if not force:
            elapsed = (datetime.now() - self.last_sync_time).total_seconds()
            if elapsed < 300:  # 5 minutes
                return
        
        logger.info(f"\n同步記憶到持久化後端...")
        
        # 同步 Episodic Memories（新增的）
        synced_episodes = 0
        for episode in self.episodic_memories.values():
            # 檢查是否已經持久化過（簡單判斷：metadata 中有標記）
            if not episode.metadata.get('persisted'):
                try:
                    self.persistence_backend.add_knowledge_fragment(
                        fragment_type='episodic_memory',
                        title=f"{episode.event_type.value}: {episode.robot_action}",
                        content=f"""
Event: {episode.event_type.value}
Step: {episode.step}
Action: {episode.robot_action}
Response: {episode.human_response}
Outcome: {episode.outcome.value}
                        """.strip(),
                        source_type='memory_system_sync',
                        confidence=0.9
                    )
                    episode.metadata['persisted'] = True
                    synced_episodes += 1
                except:
                    continue
        
        # 同步 Semantic Rules（新增或更新的）
        synced_rules = 0
        for rule in self.semantic_rules.values():
            # 簡單判斷：last_updated > last_sync_time
            if rule.last_updated > self.last_sync_time:
                try:
                    self.add_semantic_rule(rule)  # 會自動持久化
                    synced_rules += 1
                except:
                    continue
        
        self.last_sync_time = datetime.now()
        
        if synced_episodes > 0 or synced_rules > 0:
            logger.info(f"   同步完成: {synced_episodes} episodes, {synced_rules} rules")
    
    def end_episode(self, episode_id: str):
        """
        Episode 結束時的處理
        
        Args:
            episode_id: Episode ID
        """
        logger.info(f"\nEpisode 結束: {episode_id}")
        
        # 1. 強制同步到持久化後端
        self.sync_to_persistence(force=True)
        
        # 2. 保存 KG Snapshot
        if self.spatial_memory.nodes or self.spatial_memory.relations:
            try:
                import json
                import os
                os.makedirs('data/kg_snapshots', exist_ok=True)
                
                kg_data = {
                    'episode_id': episode_id,
                    'timestamp': datetime.now().isoformat(),
                    'nodes': [n.to_dict() for n in self.spatial_memory.nodes.values()],
                    'relations': [r.to_dict() for r in self.spatial_memory.relations.values()]
                }
                
                filename = f'data/kg_snapshots/kg_{episode_id}.json'
                with open(filename, 'w') as f:
                    json.dump(kg_data, f, indent=2)
                
                logger.info(f"   KG Snapshot 已保存: {filename}")
            except Exception as e:
                logger.warning(f"   KG Snapshot 保存失敗: {e}")
        
        # 3. 顯示統計
        self.print_stats()
    
    def print_stats(self):
        """列印統計資訊"""
        stats = self.get_stats()
        logger.info("\n" + "="*50)
        logger.info("Memory System Statistics")
        logger.info("="*50)
        logger.info(f"Temporal Memory: {stats['temporal_buffer_size']} entries")
        logger.info(f"Compressed Summaries: {stats['compressed_summaries']}")
        logger.info(f"Knowledge Graph: {stats['kg_nodes']} nodes, {stats['kg_relations']} relations")
        logger.info(f"Episodic Memories (RAM): {stats['episodic_memories']}")
        logger.info(f"Semantic Rules (RAM): {stats['semantic_rules']}")
        logger.info(f"Total Updates: {stats['total_updates']}")
        if self.persistence_backend:
            logger.info(f"Persistence: Connected")
        else:
            logger.error(f"Persistence: Memory-only")
        
    def get_decision_context(self, query: str) -> str:
        """
        統一的決策上下文獲取接口
        Returns:
            Formatted context string combining all memory types
        """
        context = ""

        # 0. Interaction history (multi-turn context — most relevant for policy)
        interaction_ctx = self.interaction_log.to_context_text(n=5)
        if interaction_ctx:
            context += interaction_ctx + "\n\n"

        # 1. Temporal Context (recent 10 steps)
        recent_summaries = list(self.temporal_memory.buffer)[-10:]
        if recent_summaries:
            context += "=== Recent Steps ===\n"
            for summary in recent_summaries:
                context += f"Step {summary.current_step}: Human-{summary.human_state.value}, Robot-{summary.robot_last_action or 'idle'}, {summary.outcome.value}\n"
            context += "\n"

        # 2. Spatial Context (Knowledge Graph)
        if len(self.spatial_memory.nodes) > 0:
            context += f"=== Knowledge Graph ({len(self.spatial_memory.nodes)} nodes) ===\n"
            for node in list(self.spatial_memory.nodes.values())[:5]:
                context += f"- {node.node_id}: {node.node_type}\n"
            context += "\n"
        
        # 3. Episodic Context (similar past events)
        if self.episodic_memories:
            # 簡單使用時間排序，或向量搜索
            if isinstance(query, str):
                # 如果是字符串，創建臨時 MemoryQuery
                query_obj = MemoryQuery(query_text=query, query_type="episodic", context={})
                similar_episodes = self.retrieve_similar_episodes(query_obj, top_k=3)
            else:
                similar_episodes = self.retrieve_similar_episodes(query, top_k=3)
            
            if similar_episodes:
                context += "=== Similar Past Events ===\n"
                for ep in similar_episodes:
                    context += f"- {ep.event_type.value} in {ep.step}: {ep.outcome.value}\n"
                context += "\n"
        
        # 4. Semantic Rules
        if self.semantic_rules:
            context += "=== Learned Rules ===\n"
            for rule in list(self.semantic_rules.values())[:5]:
                context += f"- {rule.rule_text} (confidence: {rule.confidence:.2f})\n"
            context += "\n"
        
        return context

    def record_interaction(
        self,
        user_utterance: str,
        response_type: str,
        robot_response: str,
        step_context: str,
        task_id: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> InteractionRecord:
        """記錄一次 Execute / Ask / Answer 互動事件到 interaction_log。

        Args:
            user_utterance: 人類原始輸入文字
            response_type:  'execute' | 'ask' | 'answer'
            robot_response: 機器人的回應文字（TTS 播出的或動作描述）
            step_context:   當時的 TSM step_class
            task_id:        關聯任務 ID（可選）
            success:        執行分支的結果（可選）

        Returns:
            建立的 InteractionRecord（已寫入 interaction_log）
        """
        import time as _time
        rec = InteractionRecord(
            timestamp=_time.time(),
            user_utterance=user_utterance,
            response_type=response_type,
            robot_response=robot_response,
            step_context=step_context,
            task_id=task_id,
            success=success,
        )
        self.interaction_log.record(rec)
        return rec

    def get_stats(self) -> Dict:
        """取得統計資訊"""
        return {
            'temporal_buffer_size': len(self.temporal_memory.buffer),
            'compressed_summaries': len(self.temporal_memory.compressed_summaries),
            'kg_nodes': len(self.spatial_memory.nodes),
            'kg_relations': len(self.spatial_memory.relations),
            'episodic_memories': len(self.episodic_memories),
            'semantic_rules': len(self.semantic_rules),
            'total_updates': self.update_count
        }
    
    def print_stats(self):
        """列印統計資訊"""
        stats = self.get_stats()
        logger.info("\n" + "="*50)
        logger.info("Memory System Statistics")
        logger.info("="*50)
        logger.info(f"Temporal Memory: {stats['temporal_buffer_size']} entries")
        logger.info(f"Compressed Summaries: {stats['compressed_summaries']}")
        logger.info(f"Knowledge Graph: {stats['kg_nodes']} nodes, {stats['kg_relations']} relations")
        logger.info(f"Episodic Memories: {stats['episodic_memories']}")
        logger.info(f"Semantic Rules: {stats['semantic_rules']}")
        logger.info(f"Total Updates: {stats['total_updates']}")
        logger.info("="*50 + "\n")
