"""
Multi-Modal Perception Logger
統整所有感知模組(物件偵測、步驟偵測、手勢偵測)和 LLM 輸出到資料庫
用於更新 RAG 知識庫
"""
import logging
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import numpy as np

from ws_racr.config import get_database_config
from ws_racr.database.embedding_singleton import get_embedding_model
from ws_racr.world_state import WorldState

logger = logging.getLogger(__name__)

DEFAULT_DB_URL = get_database_config()['url']


class PerceptionLogger:
    """
    多模態感知記錄器
    
    功能:
    1. 記錄所有感知模組的輸出 (物件、步驟、手勢)
    2. 記錄 LLM 的指令和決策
    3. 記錄語音輸入的原始文本
    4. 建立 episode 級別的多模態資料集
    5. 自動更新 RAG 知識庫
    """
    
    def __init__(self, db_url=None):
        self.engine = create_engine(db_url or DEFAULT_DB_URL)
        self.embedding_model = get_embedding_model("all-MiniLM-L6-v2")  # 使用共享實例
        self._ensure_tables()
        
        # Current episode tracking
        self.current_episode_id = None
        self.episode_start_time = None
    
    def _ensure_tables(self):
        """確保所有資料表存在"""
        with self.engine.begin() as conn:
            # 確保 vector 擴展
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except:
                pass
            
            # ========== 1. Episodes 表 (組裝任務的一次完整執行) ==========
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id SERIAL PRIMARY KEY,
                    episode_id VARCHAR(100) UNIQUE NOT NULL,
                    session_id VARCHAR(100),  -- 關聯到 tasks 表的 session
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'in_progress',  -- in_progress, completed, failed, interrupted
                    assembly_phase INT,  -- 最終達到的階段
                    success BOOLEAN,
                    error_message TEXT,
                    metadata JSONB,  -- 額外資訊 (環境設置、操作員資訊等)
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            # ========== 2. Perception_Logs 表 (所有感知模組的時序記錄) ==========
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS perception_logs (
                    id SERIAL PRIMARY KEY,
                    episode_id VARCHAR(100) REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    timestamp TIMESTAMP NOT NULL,
                    frame_number INT,
                    
                    -- Step Recognition (步驟識別)
                    step_class VARCHAR(50),
                    step_confidence FLOAT,
                    step_phase INT,
                    
                    -- Object Detection (物件偵測)
                    detected_objects JSONB,  -- [{"class": "cylinder", "bbox": [...], "conf": 0.95}, ...]
                    object_count INT,
                    
                    -- Gesture Recognition (手勢識別)
                    gesture_class VARCHAR(50),
                    gesture_confidence FLOAT,
                    hand_keypoints JSONB,  -- MediaPipe 手部關鍵點
                    
                    -- Safety & Intent (從 FSM 輸出)
                    safety_state VARCHAR(20),  -- OK, WARNING, EMERGENCY_STOP
                    intent_state VARCHAR(20),  -- idle, receive, pointing
                    
                    -- World State (統整後的世界狀態)
                    world_state JSONB,  -- 完整的 WorldState JSON
                    
                    -- Assembly Progress
                    assembly_progress JSONB,
                    
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            # ========== 3. LLM_Interactions 表 (LLM 的輸入輸出記錄) ==========
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS llm_interactions (
                    id SERIAL PRIMARY KEY,
                    episode_id VARCHAR(100) REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    interaction_id VARCHAR(100) UNIQUE NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    
                    -- Input (輸入給 LLM)
                    input_type VARCHAR(20),  -- voice, text, auto_reasoning
                    user_voice_input TEXT,  -- 語音轉文字的原始輸入
                    user_text_input TEXT,   -- 文字輸入
                    world_state_input JSONB,  -- 當時的 world_state
                    sop_context TEXT,  -- RAG 檢索到的 SOP 片段
                    
                    -- LLM Processing
                    llm_model VARCHAR(50),
                    llm_prompt TEXT,  -- 完整的 prompt
                    llm_response TEXT,  -- LLM 的原始回應
                    
                    -- Output (LLM 產生的指令)
                    task_json JSONB,  -- 語意解析 (action, target_object, etc.)
                    plan_json JSONB,  -- 可執行計畫 (waypoints, IO, constraints)
                    robot_command VARCHAR(100),  -- 機器人指令類型
                    
                    -- Embedding (用於 RAG)
                    interaction_embedding vector(384),
                    
                    -- Execution
                    executed BOOLEAN DEFAULT FALSE,
                    execution_result JSONB,
                    execution_error TEXT,
                    
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            # ========== 4. Object_Tracking 表 (物件追蹤歷史) ==========
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS object_tracking (
                    id SERIAL PRIMARY KEY,
                    episode_id VARCHAR(100) REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    object_id VARCHAR(100) NOT NULL,
                    object_type VARCHAR(50) NOT NULL,
                    
                    -- Tracking over time
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    total_detections INT DEFAULT 1,
                    
                    -- State changes
                    location_history JSONB,  -- [{"time": ..., "location": "in_tray"}, ...]
                    position_history JSONB,  -- [{"time": ..., "pos": [x, y, z]}, ...]
                    
                    -- Final state
                    final_location VARCHAR(50),
                    final_position JSONB,
                    
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            # ========== 5. Knowledge_Fragments 表 (自動萃取的知識片段,用於更新 RAG) ==========
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS knowledge_fragments (
                    id SERIAL PRIMARY KEY,
                    fragment_id VARCHAR(100) UNIQUE NOT NULL,
                    episode_id VARCHAR(100) REFERENCES episodes(episode_id) ON DELETE CASCADE,
                    
                    -- Knowledge content
                    fragment_type VARCHAR(50),  -- step_pattern, error_recovery, user_preference, etc.
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_embedding vector(384),
                    
                    -- Source
                    source_type VARCHAR(50),  -- perception, llm_interaction, user_feedback
                    source_data JSONB,  -- 原始資料參考
                    
                    -- Metadata
                    confidence FLOAT,  -- 知識可信度
                    usefulness_score FLOAT,  -- 使用頻率/效果評分
                    tags TEXT[],
                    
                    -- RAG integration
                    added_to_rag BOOLEAN DEFAULT FALSE,
                    rag_sop_id VARCHAR(100),  -- 關聯到 sops 表
                    
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            # ========== 索引優化 ==========
            
            # Episodes 索引
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(start_time DESC)
            """))
            
            # Perception_Logs 索引
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_perception_episode ON perception_logs(episode_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_perception_timestamp ON perception_logs(timestamp)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_perception_step ON perception_logs(step_class)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_perception_safety ON perception_logs(safety_state)
            """))
            
            # LLM_Interactions 索引
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_llm_episode ON llm_interactions(episode_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_llm_timestamp ON llm_interactions(timestamp)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_llm_executed ON llm_interactions(executed)
            """))
            
            # Vector 索引
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_llm_embedding_hnsw 
                    ON llm_interactions USING hnsw(interaction_embedding vector_cosine_ops)
                """))
            except:
                pass
            
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_knowledge_embedding_hnsw 
                    ON knowledge_fragments USING hnsw(content_embedding vector_cosine_ops)
                """))
            except:
                pass
            
            # Object_Tracking 索引
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_object_tracking_episode ON object_tracking(episode_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_object_tracking_type ON object_tracking(object_type)
            """))
            
            # Knowledge_Fragments 索引
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_episode ON knowledge_fragments(episode_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_fragments(fragment_type)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_rag ON knowledge_fragments(added_to_rag)
            """))
            
        logger.info("多模態感知資料表已就緒")
    
    # ========== Episode Management ==========
    
    def start_episode(self, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """
        開始新的 episode (一次組裝任務執行)
        
        Args:
            session_id: 關聯的 session ID (與 tasks 表關聯)
            metadata: 額外資訊
        
        Returns:
            episode_id
        """
        episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.current_episode_id = episode_id
        self.episode_start_time = datetime.now()
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO episodes (episode_id, session_id, start_time, metadata)
                VALUES (:episode_id, :session_id, :start_time, :metadata)
            """), {
                'episode_id': episode_id,
                'session_id': session_id,
                'start_time': self.episode_start_time,
                'metadata': json.dumps(metadata) if metadata else None
            })
        
        logger.info(f"Episode started: {episode_id}")
        return episode_id
    
    def end_episode(self, success: bool, assembly_phase: int = 0, 
                   error_message: Optional[str] = None):
        """
        結束當前 episode
        
        Args:
            success: 是否成功完成
            assembly_phase: 最終達到的階段
            error_message: 錯誤訊息 (如果失敗)
        """
        if not self.current_episode_id:
            logger.warning("No active episode")
            return
        
        status = 'completed' if success else 'failed'
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE episodes
                SET end_time = :end_time,
                    status = :status,
                    success = :success,
                    assembly_phase = :phase,
                    error_message = :error
                WHERE episode_id = :episode_id
            """), {
                'end_time': datetime.now(),
                'status': status,
                'success': success,
                'phase': assembly_phase,
                'error': error_message,
                'episode_id': self.current_episode_id
            })
        
        logger.info(f"Episode ended: {self.current_episode_id} ({status})")
        self.current_episode_id = None
    
    # ========== Perception Logging ==========
    
    def log_perception(self, world_state: WorldState, frame_number: int = 0,
                      detected_objects: Optional[List[Dict]] = None,
                      gesture_data: Optional[Dict] = None):
        """
        記錄感知模組輸出
        
        Args:
            world_state: 統整後的 WorldState 物件
            frame_number: 影格編號
            detected_objects: 原始物件偵測結果 (YOLO 輸出)
            gesture_data: 原始手勢資料 (包含 keypoints)
        """
        if not self.current_episode_id:
            logger.warning("No active episode, call start_episode() first")
            return
        
        timestamp = datetime.fromtimestamp(world_state.timestamp)
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO perception_logs (
                    episode_id, timestamp, frame_number,
                    step_class, step_confidence, step_phase,
                    detected_objects, object_count,
                    gesture_class, gesture_confidence, hand_keypoints,
                    safety_state, intent_state,
                    world_state, assembly_progress
                )
                VALUES (
                    :episode_id, :timestamp, :frame_number,
                    :step_class, :step_conf, :step_phase,
                    :objects, :obj_count,
                    :gesture, :gesture_conf, :keypoints,
                    :safety, :intent,
                    :world_state, :progress
                )
            """), {
                'episode_id': self.current_episode_id,
                'timestamp': timestamp,
                'frame_number': frame_number,
                'step_class': world_state.step_class,
                'step_conf': float(world_state.step_confidence),  # 轉換 numpy.float32
                'step_phase': world_state.phase,
                'objects': json.dumps(detected_objects) if detected_objects else None,
                'obj_count': len(detected_objects) if detected_objects else 0,
                'gesture': gesture_data.get('class') if gesture_data else None,
                'gesture_conf': float(gesture_data.get('confidence')) if gesture_data and gesture_data.get('confidence') is not None else None,  # 轉換 numpy.float32
                'keypoints': json.dumps(gesture_data.get('keypoints')) if gesture_data and 'keypoints' in gesture_data else None,
                'safety': world_state.human.safety,
                'intent': world_state.human.intent,
                'world_state': json.dumps(world_state.to_json_compatible()),
                'progress': json.dumps(world_state.assembly_progress.__dict__)
            })
    
    # ========== LLM Interaction Logging ==========
    
    def log_llm_interaction(self, 
                           input_type: str,
                           world_state: WorldState,
                           user_input: Optional[str] = None,
                           voice_input: Optional[str] = None,
                           sop_context: Optional[str] = None,
                           llm_prompt: Optional[str] = None,
                           llm_response: Optional[str] = None,
                           task_json: Optional[Dict] = None,
                           plan_json: Optional[Dict] = None,
                           robot_command: Optional[str] = None,
                           llm_model: str = "gpt-4") -> str:
        """
        記錄 LLM 互動
        
        Args:
            input_type: 輸入類型 (voice, text, auto_reasoning)
            world_state: 當時的世界狀態
            user_input: 使用者文字輸入
            voice_input: 語音輸入 (轉文字後)
            sop_context: RAG 檢索到的 SOP 片段
            llm_prompt: 完整 prompt
            llm_response: LLM 回應
            task_json: 任務 JSON
            plan_json: 計畫 JSON
            robot_command: 機器人指令
            llm_model: 使用的模型
        
        Returns:
            interaction_id
        """
        if not self.current_episode_id:
            logger.warning("No active episode")
            return None
        
        interaction_id = f"llm_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 生成 embedding (用於 RAG)
        embedding_text = f"{user_input or ''} {voice_input or ''} {llm_response or ''}"
        embedding_list = self.embedding_model.encode(embedding_text).astype(np.float32).tolist()
        # Convert to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO llm_interactions (
                    episode_id, interaction_id, timestamp,
                    input_type, user_voice_input, user_text_input,
                    world_state_input, sop_context,
                    llm_model, llm_prompt, llm_response,
                    task_json, plan_json, robot_command,
                    interaction_embedding
                )
                VALUES (
                    :episode_id, :interaction_id, :timestamp,
                    :input_type, :voice_input, :text_input,
                    :world_state, :sop_context,
                    :llm_model, :llm_prompt, :llm_response,
                    :task_json, :plan_json, :robot_command,
                    :embedding
                )
            """), {
                'episode_id': self.current_episode_id,
                'interaction_id': interaction_id,
                'timestamp': datetime.now(),
                'input_type': input_type,
                'voice_input': voice_input,
                'text_input': user_input,
                'world_state': json.dumps(world_state.to_json_compatible()),
                'sop_context': sop_context,
                'llm_model': llm_model,
                'llm_prompt': llm_prompt,
                'llm_response': llm_response,
                'task_json': json.dumps(task_json) if task_json else None,
                'plan_json': json.dumps(plan_json) if plan_json else None,
                'robot_command': robot_command,
                'embedding': embedding_str
            })
        
        return interaction_id
    
    def update_llm_execution(self, interaction_id: str, 
                            executed: bool = True,
                            result: Optional[Dict] = None,
                            error: Optional[str] = None):
        """更新 LLM 指令的執行結果"""
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE llm_interactions
                SET executed = :executed,
                    execution_result = :result,
                    execution_error = :error
                WHERE interaction_id = :interaction_id
            """), {
                'executed': executed,
                'result': json.dumps(result) if result else None,
                'error': error,
                'interaction_id': interaction_id
            })
    
    # ========== Object Tracking ==========
    
    def update_object_tracking(self, object_id: str, object_type: str,
                             location: str, position: Optional[List[float]] = None):
        """
        更新物件追蹤記錄
        
        Args:
            object_id: 物件 ID
            object_type: 物件類型
            location: 當前位置
            position: 3D 座標
        """
        if not self.current_episode_id:
            return
        
        timestamp = datetime.now()
        
        with self.engine.begin() as conn:
            # Check if object exists in this episode
            result = conn.execute(text("""
                SELECT id, location_history, position_history, total_detections
                FROM object_tracking
                WHERE episode_id = :episode_id AND object_id = :object_id
            """), {
                'episode_id': self.current_episode_id,
                'object_id': object_id
            }).fetchone()
            
            if result:
                # Update existing
                # PostgreSQL JSONB is already parsed to dict/list in Python
                location_history = result[1] if result[1] else []
                position_history = result[2] if result[2] else []
                total_detections = result[3] + 1
                
                location_history.append({
                    'time': timestamp.isoformat(),
                    'location': location
                })
                
                if position:
                    position_history.append({
                        'time': timestamp.isoformat(),
                        'position': position
                    })
                
                conn.execute(text("""
                    UPDATE object_tracking
                    SET last_seen = :last_seen,
                        total_detections = :total,
                        location_history = :loc_hist,
                        position_history = :pos_hist,
                        final_location = :final_loc,
                        final_position = :final_pos
                    WHERE id = :id
                """), {
                    'last_seen': timestamp,
                    'total': total_detections,
                    'loc_hist': json.dumps(location_history),
                    'pos_hist': json.dumps(position_history),
                    'final_loc': location,
                    'final_pos': json.dumps(position) if position else None,
                    'id': result[0]
                })
            else:
                # Insert new
                location_history = [{'time': timestamp.isoformat(), 'location': location}]
                position_history = [{'time': timestamp.isoformat(), 'position': position}] if position else []
                
                conn.execute(text("""
                    INSERT INTO object_tracking (
                        episode_id, object_id, object_type,
                        first_seen, last_seen, total_detections,
                        location_history, position_history,
                        final_location, final_position
                    )
                    VALUES (
                        :episode_id, :object_id, :object_type,
                        :first_seen, :last_seen, :total,
                        :loc_hist, :pos_hist,
                        :final_loc, :final_pos
                    )
                """), {
                    'episode_id': self.current_episode_id,
                    'object_id': object_id,
                    'object_type': object_type,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'total': 1,
                    'loc_hist': json.dumps(location_history),
                    'pos_hist': json.dumps(position_history),
                    'final_loc': location,
                    'final_pos': json.dumps(position) if position else None
                })
    
    # ========== Knowledge Fragment Extraction ==========
    
    def add_knowledge_fragment(self, fragment_type: str, title: str, content: str,
                              source_type: str, source_data: Optional[Dict] = None,
                              tags: Optional[List[str]] = None,
                              confidence: float = 1.0) -> str:
        """
        新增知識片段 (將自動更新到 RAG)
        
        Args:
            fragment_type: 知識類型 (step_pattern, error_recovery, user_preference, etc.)
            title: 標題
            content: 內容
            source_type: 來源類型 (perception, llm_interaction, user_feedback)
            source_data: 原始資料
            tags: 標籤
            confidence: 可信度
        
        Returns:
            fragment_id
        """
        fragment_id = f"knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Generate embedding
        embedding_list = self.embedding_model.encode(content).astype(np.float32).tolist()
        # Convert to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO knowledge_fragments (
                    fragment_id, episode_id, fragment_type,
                    title, content, content_embedding,
                    source_type, source_data, confidence, tags
                )
                VALUES (
                    :fragment_id, :episode_id, :fragment_type,
                    :title, :content, :embedding,
                    :source_type, :source_data, :confidence, :tags
                )
            """), {
                'fragment_id': fragment_id,
                'episode_id': self.current_episode_id,
                'fragment_type': fragment_type,
                'title': title,
                'content': content,
                'embedding': embedding_str,
                'source_type': source_type,
                'source_data': json.dumps(source_data) if source_data else None,
                'confidence': confidence,
                'tags': tags
            })
        
        logger.info(f"Knowledge fragment added: {fragment_id}")
        return fragment_id
    
    def integrate_knowledge_to_rag(self, fragment_id: str, sop_id: str):
        """將知識片段整合到 RAG (sops 表)"""
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE knowledge_fragments
                SET added_to_rag = TRUE,
                    rag_sop_id = :sop_id,
                    updated_at = :updated_at
                WHERE fragment_id = :fragment_id
            """), {
                'sop_id': sop_id,
                'updated_at': datetime.now(),
                'fragment_id': fragment_id
            })
    
    # ========== Query Methods ==========
    
    def get_episode_summary(self, episode_id: str) -> Dict:
        """取得 episode 摘要"""
        with self.engine.begin() as conn:
            # Episode info
            episode = conn.execute(text("""
                SELECT * FROM episodes WHERE episode_id = :episode_id
            """), {'episode_id': episode_id}).fetchone()
            
            if not episode:
                return None
            
            # Perception logs count
            perception_count = conn.execute(text("""
                SELECT COUNT(*) FROM perception_logs WHERE episode_id = :episode_id
            """), {'episode_id': episode_id}).fetchone()[0]
            
            # LLM interactions count
            llm_count = conn.execute(text("""
                SELECT COUNT(*) FROM llm_interactions WHERE episode_id = :episode_id
            """), {'episode_id': episode_id}).fetchone()[0]
            
            # Objects tracked
            objects_count = conn.execute(text("""
                SELECT COUNT(*) FROM object_tracking WHERE episode_id = :episode_id
            """), {'episode_id': episode_id}).fetchone()[0]
            
            # Knowledge fragments
            knowledge_count = conn.execute(text("""
                SELECT COUNT(*) FROM knowledge_fragments WHERE episode_id = :episode_id
            """), {'episode_id': episode_id}).fetchone()[0]
            
            return {
                'episode_id': episode_id,
                'session_id': episode[2],
                'start_time': episode[3],
                'end_time': episode[4],
                'status': episode[5],
                'phase': episode[6],
                'success': episode[7],
                'perception_logs': perception_count,
                'llm_interactions': llm_count,
                'objects_tracked': objects_count,
                'knowledge_fragments': knowledge_count
            }
    
    def search_similar_interactions(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜尋相似的 LLM 互動 (用於 RAG)
        
        Args:
            query: 查詢文本
            top_k: 返回前 K 個結果
        
        Returns:
            相似互動列表
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).astype(np.float32).tolist()
        
        # Convert to PostgreSQL vector string format: '[1.0, 2.0, 3.0]'
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        with self.engine.begin() as conn:
            # Use raw SQL with format to avoid parameter issues
            sql = f"""
                SELECT interaction_id, user_voice_input, user_text_input,
                       llm_response, task_json, robot_command,
                       1 - (interaction_embedding <=> '{embedding_str}'::vector) as similarity
                FROM llm_interactions
                WHERE interaction_embedding IS NOT NULL
                ORDER BY interaction_embedding <=> '{embedding_str}'::vector
                LIMIT {top_k}
            """
            results = conn.execute(text(sql)).fetchall()
            
            return [
                {
                    'interaction_id': r[0],
                    'voice_input': r[1],
                    'text_input': r[2],
                    'llm_response': r[3],
                    'task_json': r[4],  # Already dict (JSONB)
                    'robot_command': r[5],
                    'similarity': float(r[6]) if r[6] is not None else 0.0
                }
                for r in results
            ]


# ========== Helper Function ==========

def create_perception_logger(db_url: Optional[str] = None) -> PerceptionLogger:
    """Factory function to create PerceptionLogger"""
    return PerceptionLogger(db_url)
