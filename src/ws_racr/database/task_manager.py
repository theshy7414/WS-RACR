"""TaskManager: PostgreSQL + pgvector persistence of tasks, plans, sessions and SOPs.

This is the storage side of the **plan RAG / episodic memory** (thesis Ch. 4): completed multi-step
plans are stored with a sentence embedding of the user command and retrieved by cosine similarity
(``search_similar_plans``). Retrieval only recalls candidates; the REUSE / ADAPT / CLARIFY / REJECT
decision is made by the decision layers, not here. The ``sops`` table backs the SOP knowledge base.
"""
import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, JSON, DateTime, Float
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
from typing import Dict, List, Optional
import json
import numpy as np

from ws_racr.config import get_database_config
from ws_racr.database.embedding_singleton import get_embedding_model

logger = logging.getLogger(__name__)

DEFAULT_DB_URL = get_database_config()['url']

class TaskManager:
    """任務資料庫管理器"""
    
    def __init__(self, db_url=None):
        self.engine = create_engine(db_url or DEFAULT_DB_URL)
        self.model = get_embedding_model("paraphrase-multilingual-MiniLM-L12-v2")  # 使用共享實例
        self._ensure_tables()
        
    def _ensure_tables(self):
        """確保資料表存在"""
        with self.engine.begin() as conn:
            # 確保 vector 擴展存在（必須在建表前）
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except:
                pass
            
            # 任務表
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    task_id VARCHAR(100) UNIQUE NOT NULL,
                    user_input TEXT NOT NULL,
                    task_type VARCHAR(50) NOT NULL,
                    target_object VARCHAR(100),
                    target_location VARCHAR(100),
                    task_json JSONB NOT NULL,
                    task_embedding vector(384),
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    executed_at TIMESTAMP,
                    result JSONB,
                    error_message TEXT,
                    -- 新增欄位：拆分「語意」與「可執行計畫」
                    plan_json JSONB,           -- 可執行計畫（waypoints + IO + 約束）
                    source VARCHAR(20),        -- 任務來源（voice/text/api）
                    revision_of VARCHAR(100),  -- 修訂自哪個任務
                    telemetry JSONB,           -- 執行過程摘要
                    session_id VARCHAR(100)    -- 任務會話 ID（同一輪互動的任務群組）
                )
            """))
            
            # SOP 文檔表（RAG 知識庫）
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sops (
                    id SERIAL PRIMARY KEY,
                    sop_id VARCHAR(100) UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_embedding vector(384),
                    category VARCHAR(50),
                    tags TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            # 為現有表新增欄位（向下兼容）
            try:
                conn.execute(text("""
                    ALTER TABLE tasks
                    ADD COLUMN IF NOT EXISTS plan_json JSONB,
                    ADD COLUMN IF NOT EXISTS source VARCHAR(20),
                    ADD COLUMN IF NOT EXISTS revision_of VARCHAR(100),
                    ADD COLUMN IF NOT EXISTS telemetry JSONB,
                    ADD COLUMN IF NOT EXISTS session_id VARCHAR(100)
                """))
            except:
                pass
            
            # ========== 索引優化 ==========
            
            # 基本索引
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_executed ON tasks(executed_at DESC)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_source ON tasks(source)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_revision ON tasks(revision_of)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)
            """))
            
            # GIN 索引（JSONB 條件查詢加速）- tasks 表
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_task_json_gin ON tasks USING GIN(task_json jsonb_path_ops)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tasks_plan_json_gin ON tasks USING GIN(plan_json jsonb_path_ops)
            """))
            
            # 向量索引（語義檢索加速）- HNSW 比 IVFFlat 更適合中小規模
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_embedding_hnsw 
                    ON tasks USING hnsw(task_embedding vector_cosine_ops)
                """))
            except Exception as e:
                # HNSW 可能需要較新版本的 pgvector，回退到 IVFFlat
                try:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_tasks_embedding_ivfflat 
                        ON tasks USING ivfflat(task_embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """))
                except:
                    logger.warning("無法建立向量索引，將使用序列掃描（性能較慢）")
            
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sops_embedding_hnsw 
                    ON sops USING hnsw(content_embedding vector_cosine_ops)
                """))
            except:
                try:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_sops_embedding_ivfflat 
                        ON sops USING ivfflat(content_embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """))
                except:
                    pass
            
            # GIN 索引（JSONB 條件查詢加速）- sops 表
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sops_metadata_gin ON sops USING GIN(metadata jsonb_path_ops)
                """))
            except:
                pass
            
        logger.info("任務資料表已就緒")
    
    @staticmethod
    def _build_embedding_text(task_json: dict, user_input: str) -> str:
        """Build clean structured text for embedding.

        Strips raw Python dict literals that sometimes appear in user_input
        (e.g. gesture trigger strings), then produces a short structured
        string that embedding models can parse reliably across languages.
        """
        import re
        action = task_json.get('action', '')
        obj = task_json.get('target_object') or ''
        loc = task_json.get('target_location') or ''
        # Remove Python dict literals: {...}
        clean = re.sub(r'\{[^}]{0,500}\}', '', user_input)
        # Remove gesture trigger prefix: "手勢觸發: pointing_gesture"
        clean = re.sub(r'^手勢觸發:\s*\S+', '', clean).strip()
        parts = [p for p in [action, obj, loc, clean] if p.strip()]
        return ' '.join(parts)[:512]

    def save_task(self, user_input: str, task_json: Dict,
                  plan_json: Optional[Dict] = None,
                  source: str = "text",
                  revision_of: Optional[str] = None,
                  session_id: Optional[str] = None) -> str:
        """
        儲存 LLM 產生的任務
        
        Args:
            user_input: 使用者原始輸入
            task_json: LLM 產生的任務 JSON（語意解析）
            plan_json: 可執行計畫（waypoints + IO + 約束），可選
            source: 任務來源（voice/text/api）
            revision_of: 修訂自哪個任務 ID
            session_id: 任務會話 ID（同一輪互動的任務群組）
            
        Returns:
            task_id: 任務 ID
        """
        # 生成任務 ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 提取關鍵資訊
        task_type = task_json.get('action', 'unknown')
        target_object = task_json.get('target_object')
        target_location = task_json.get('target_location')
        
        # 生成嵌入向量（用於語義搜尋）
        embedding_text = self._build_embedding_text(task_json, user_input)
        embedding = self.model.encode(embedding_text).astype(np.float32).tolist()
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO tasks (
                    task_id, user_input, task_type, target_object, 
                    target_location, task_json, task_embedding, status,
                    plan_json, source, revision_of, session_id
                ) VALUES (
                    :task_id, :user_input, :task_type, :target_object,
                    :target_location, :task_json, :embedding, 'pending',
                    :plan_json, :source, :revision_of, :session_id
                )
            """), {
                "task_id": task_id,
                "user_input": user_input,
                "task_type": task_type,
                "target_object": target_object,
                "target_location": target_location,
                "task_json": json.dumps(task_json),
                "embedding": embedding,
                "plan_json": json.dumps(plan_json) if plan_json else None,
                "source": source,
                "revision_of": revision_of,
                "session_id": session_id
            })
        
        logger.info(f"任務已儲存: {task_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """獲取任務資料"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT task_id, user_input, task_type, target_object,
                       target_location, task_json, status, created_at,
                       executed_at, result, error_message,
                       plan_json, source, revision_of, telemetry
                FROM tasks
                WHERE task_id = :task_id
            """), {"task_id": task_id}).fetchone()
            
            if result:
                return {
                    "task_id": result.task_id,
                    "user_input": result.user_input,
                    "task_type": result.task_type,
                    "target_object": result.target_object,
                    "target_location": result.target_location,
                    "task_json": json.loads(result.task_json) if isinstance(result.task_json, str) else result.task_json,
                    "status": result.status,
                    "created_at": result.created_at,
                    "executed_at": result.executed_at,
                    "result": json.loads(result.result) if result.result and isinstance(result.result, str) else result.result,
                    "error_message": result.error_message,
                    # 新增欄位
                    "plan_json": json.loads(result.plan_json) if result.plan_json and isinstance(result.plan_json, str) else result.plan_json,
                    "source": result.source,
                    "revision_of": result.revision_of,
                    "telemetry": json.loads(result.telemetry) if result.telemetry and isinstance(result.telemetry, str) else result.telemetry
                }
        return None
    
    def get_pending_tasks(self, limit: int = 10) -> List[Dict]:
        """獲取待執行的任務"""
        with self.engine.connect() as conn:
            results = conn.execute(text("""
                SELECT task_id, user_input, task_type, target_object,
                       target_location, task_json, created_at
                FROM tasks
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT :limit
            """), {"limit": limit}).fetchall()
            
            tasks = []
            for r in results:
                tasks.append({
                    "task_id": r.task_id,
                    "user_input": r.user_input,
                    "task_type": r.task_type,
                    "target_object": r.target_object,
                    "target_location": r.target_location,
                    "task_json": json.loads(r.task_json) if isinstance(r.task_json, str) else r.task_json,
                    "created_at": r.created_at
                })
            return tasks
    
    def update_task_status(self, task_id: str, status: str, 
                          result: Optional[Dict] = None, 
                          error_message: Optional[str] = None,
                          telemetry: Optional[Dict] = None):
        """
        更新任務狀態
        
        Args:
            task_id: 任務 ID
            status: 新狀態（pending/executing/completed/failed）
            result: 執行結果
            error_message: 錯誤訊息
            telemetry: 執行過程摘要（新增）
        """
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE tasks
                SET status = :status,
                    executed_at = CASE WHEN :status IN ('completed', 'failed') 
                                      THEN NOW() ELSE executed_at END,
                    result = :result,
                    error_message = :error_message,
                    telemetry = :telemetry
                WHERE task_id = :task_id
            """), {
                "task_id": task_id,
                "status": status,
                "result": json.dumps(result) if result else None,
                "error_message": error_message,
                "telemetry": json.dumps(telemetry) if telemetry else None
            })
        
        logger.info(f"任務狀態更新: {task_id} → {status}")
    
    def update_task_plan(self, task_id: str, plan_json: Dict):
        """
        更新任務的可執行計畫
        
        Args:
            task_id: 任務 ID
            plan_json: 可執行計畫（waypoints + IO + 約束）
        """
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE tasks
                SET plan_json = :plan_json
                WHERE task_id = :task_id
            """), {
                "task_id": task_id,
                "plan_json": json.dumps(plan_json)
            })
        
        logger.info(f"任務計畫已更新: {task_id}")
    
    def update_task_field(self, task_id: str, field_name: str, value):
        """
        更新任務的指定欄位（通用方法）
        
        Args:
            task_id: 任務 ID
            field_name: 欄位名稱（task_json, plan_json, metadata等）
            value: 新值（dict會自動轉JSON，其他類型直接使用）
        """
        # 僅允許更新特定欄位（安全性）
        allowed_fields = {'task_json', 'plan_json', 'metadata', 'target_object', 'target_location'}
        if field_name not in allowed_fields:
            raise ValueError(f"不允許更新欄位: {field_name}")
        
        # 處理 JSON 類型
        if isinstance(value, dict):
            value = json.dumps(value)
        
        with self.engine.begin() as conn:
            conn.execute(text(f"""
                UPDATE tasks
                SET {field_name} = :value
                WHERE task_id = :task_id
            """), {
                "task_id": task_id,
                "value": value
            })
        
        logger.info(f"任務欄位已更新: {task_id}.{field_name}")
    
    def search_similar_tasks(self, query: str, limit: int = 5) -> List[Dict]:
        """搜尋相似的歷史任務（用於學習）"""
        query_embedding = self.model.encode(query).astype(np.float32).tolist()
        
        with self.engine.connect() as conn:
            results = conn.execute(text("""
                SELECT task_id, user_input, task_type, target_object,
                       task_json, status, 
                       (task_embedding <=> :query_emb) AS similarity
                FROM tasks
                WHERE status = 'completed'
                ORDER BY task_embedding <=> :query_emb
                LIMIT :limit
            """), {
                "query_emb": query_embedding,
                "limit": limit
            }).fetchall()
            
            tasks = []
            for r in results:
                tasks.append({
                    "task_id": r.task_id,
                    "user_input": r.user_input,
                    "task_type": r.task_type,
                    "target_object": r.target_object,
                    "task_json": json.loads(r.task_json) if isinstance(r.task_json, str) else r.task_json,
                    "status": r.status,
                    "similarity": float(r.similarity)
                })
            return tasks
    
    def get_last_completed_task(self, exclude_task_types: Optional[List[str]] = None) -> Optional[Dict]:
        """
        獲取最後一個成功完成的任務
        
        Args:
            exclude_task_types: 要排除的任務類型列表（例如 ['return_to_last', 'gripper_open']）
        
        Returns:
            任務資料字典，如果找不到則返回 None
        """
        with self.engine.connect() as conn:
            # 建立 SQL 查詢
            sql = """
                SELECT task_id, user_input, task_type, task_json,
                       plan_json, telemetry, executed_at
                FROM tasks
                WHERE status = 'completed'
            """
            
            # 如果有排除類型，加入條件
            params = {}
            if exclude_task_types:
                placeholders = ','.join(f':type{i}' for i in range(len(exclude_task_types)))
                sql += f" AND task_type NOT IN ({placeholders})"
                params = {f'type{i}': t for i, t in enumerate(exclude_task_types)}
            
            sql += " ORDER BY executed_at DESC LIMIT 1"
            
            result = conn.execute(text(sql), params).fetchone()
            
            if result:
                return {
                    "task_id": result.task_id,
                    "user_input": result.user_input,
                    "task_type": result.task_type,
                    "task_json": json.loads(result.task_json) if isinstance(result.task_json, str) else result.task_json,
                    "plan_json": json.loads(result.plan_json) if result.plan_json and isinstance(result.plan_json, str) else result.plan_json,
                    "telemetry": json.loads(result.telemetry) if result.telemetry and isinstance(result.telemetry, str) else result.telemetry,
                    "executed_at": result.executed_at
                }
        return None
    
    # ========== 新增：升級版檢索 API ==========
    
    def get_recent_tasks(self, k: int = 10, exclude_task_types: Optional[List[str]] = None, 
                        status: str = 'completed') -> List[Dict]:
        """
        獲取最近 K 個完成的任務（升級版 get_last_completed_task）
        
        Args:
            k: 返回任務數量
            exclude_task_types: 要排除的任務類型列表
            status: 任務狀態過濾（默認 'completed'）
        
        Returns:
            任務列表，按執行時間降序
        """
        with self.engine.connect() as conn:
            sql = """
                SELECT task_id, user_input, task_type, task_json,
                       plan_json, telemetry, executed_at, created_at
                FROM tasks
                WHERE status = :status
            """
            
            params = {'status': status, 'k': k}
            
            # 安全處理排除類型（使用陣列參數）
            if exclude_task_types and len(exclude_task_types) > 0:
                sql += " AND task_type <> ALL(:exclude_types)"
                params['exclude_types'] = exclude_task_types
            
            sql += " ORDER BY COALESCE(executed_at, created_at) DESC LIMIT :k"
            
            results = conn.execute(text(sql), params).fetchall()
            
            tasks = []
            for r in results:
                tasks.append({
                    "task_id": r.task_id,
                    "user_input": r.user_input,
                    "task_type": r.task_type,
                    "task_json": json.loads(r.task_json) if isinstance(r.task_json, str) else r.task_json,
                    "plan_json": json.loads(r.plan_json) if r.plan_json and isinstance(r.plan_json, str) else r.plan_json,
                    "telemetry": json.loads(r.telemetry) if r.telemetry and isinstance(r.telemetry, str) else r.telemetry,
                    "executed_at": r.executed_at,
                    "created_at": r.created_at
                })
            return tasks
    
    def get_session_tasks(self, session_id: str) -> List[Dict]:
        """
        獲取某個 session 的所有任務序列（用於返回/Undo）
        
        Args:
            session_id: 任務會話 ID
        
        Returns:
            任務列表，按創建時間升序
        """
        with self.engine.connect() as conn:
            results = conn.execute(text("""
                SELECT task_id, status, user_input, task_type, task_json,
                       plan_json, telemetry, created_at, executed_at
                FROM tasks
                WHERE session_id = :sid
                ORDER BY created_at ASC
            """), {"sid": session_id}).fetchall()
            
            tasks = []
            for r in results:
                tasks.append({
                    "task_id": r.task_id,
                    "status": r.status,
                    "user_input": r.user_input,
                    "task_type": r.task_type,
                    "task_json": json.loads(r.task_json) if isinstance(r.task_json, str) else r.task_json,
                    "plan_json": json.loads(r.plan_json) if r.plan_json and isinstance(r.plan_json, str) else r.plan_json,
                    "telemetry": json.loads(r.telemetry) if r.telemetry and isinstance(r.telemetry, str) else r.telemetry,
                    "created_at": r.created_at,
                    "executed_at": r.executed_at
                })
            return tasks
    
    def search_similar_plans(self, query: str, filters: Optional[Dict] = None, k: int = 5) -> List[Dict]:
        """
        語義相似 + 條件過濾的混合檢索（升級版 search_similar_tasks）
        
        Args:
            query: 查詢文本（會被嵌入為向量）
            filters: 條件過濾 {'action': 'pickup', 'target_object': 'box', ...}
            k: 返回結果數量
        
        Returns:
            任務列表，包含相似度分數，按相似度升序
        """
        query_embedding = self.model.encode(query).astype(np.float32).tolist()
        
        with self.engine.connect() as conn:
            # 建立條件過濾的 CTE
            where_clauses = ["status = 'completed'"]
            # 將向量轉換為字符串格式 '[x,y,z,...]'
            params = {"q_emb": str(query_embedding), "k": k}
            
            if filters:
                if 'action' in filters:
                    where_clauses.append("task_json->>'action' = :action")
                    params['action'] = filters['action']
                if 'target_object' in filters:
                    where_clauses.append("(task_json->>'target_object' = :obj OR :obj IS NULL)")
                    params['obj'] = filters.get('target_object')
                if 'target_location' in filters:
                    where_clauses.append("(task_json->>'target_location' = :loc OR :loc IS NULL)")
                    params['loc'] = filters.get('target_location')
            
            where_sql = " AND ".join(where_clauses)
            
            sql = f"""
                WITH candidates AS (
                    SELECT task_id, task_type, task_json, plan_json, telemetry, executed_at, task_embedding
                    FROM tasks
                    WHERE {where_sql}
                )
                SELECT task_id, task_type, task_json, plan_json, telemetry, executed_at,
                       (task_embedding <=> :q_emb) AS distance
                FROM candidates
                ORDER BY task_embedding <=> :q_emb
                LIMIT :k
            """
            
            results = conn.execute(text(sql), params).fetchall()
            
            tasks = []
            for r in results:
                tasks.append({
                    "task_id": r.task_id,
                    "task_type": r.task_type,
                    "task_json": json.loads(r.task_json) if isinstance(r.task_json, str) else r.task_json,
                    "plan_json": json.loads(r.plan_json) if r.plan_json and isinstance(r.plan_json, str) else r.plan_json,
                    "telemetry": json.loads(r.telemetry) if r.telemetry and isinstance(r.telemetry, str) else r.telemetry,
                    "executed_at": r.executed_at,
                    "similarity": 1.0 - float(r.distance),  # 轉換距離為相似度
                    "source": "task"
                })
            return tasks
    
    def search_hybrid_rag(self, query: str, k: int = 5, task_weight: float = 0.7) -> List[Dict]:
        """
        混合檢索：Tasks + SOPs（RAG）
        
        Args:
            query: 查詢文本
            k: 每個來源返回數量
            task_weight: Tasks 的權重（SOPs 權重 = 1 - task_weight）
        
        Returns:
            混合結果列表，包含來源標記和加權分數
        """
        query_embedding = self.model.encode(query).astype(np.float32).tolist()
        query_emb_str = str(query_embedding)  # 轉換為字符串格式
        
        results = []
        
        with self.engine.connect() as conn:
            # 1. 從 Tasks 檢索
            task_results = conn.execute(text("""
                SELECT task_id, task_json, plan_json, executed_at,
                       (task_embedding <=> :q_emb) AS distance,
                       'task' as source
                FROM tasks
                WHERE status = 'completed'
                ORDER BY task_embedding <=> :q_emb
                LIMIT :k
            """), {"q_emb": query_emb_str, "k": k}).fetchall()
            
            for r in task_results:
                results.append({
                    "id": r.task_id,
                    "content": r.task_json,
                    "plan": json.loads(r.plan_json) if r.plan_json and isinstance(r.plan_json, str) else r.plan_json,
                    "source": "task",
                    "distance": float(r.distance),
                    "similarity": 1.0 - float(r.distance),
                    "weighted_score": (1.0 - float(r.distance)) * task_weight
                })
            
            # 2. 從 SOPs 檢索
            sop_results = conn.execute(text("""
                SELECT sop_id, title, content, metadata,
                       (content_embedding <=> :q_emb) AS distance,
                       'sop' as source
                FROM sops
                ORDER BY content_embedding <=> :q_emb
                LIMIT :k
            """), {"q_emb": query_emb_str, "k": k}).fetchall()
            
            for r in sop_results:
                results.append({
                    "id": r.sop_id,
                    "title": r.title,
                    "content": r.content,
                    "metadata": json.loads(r.metadata) if r.metadata and isinstance(r.metadata, str) else r.metadata,
                    "source": "sop",
                    "distance": float(r.distance),
                    "similarity": 1.0 - float(r.distance),
                    "weighted_score": (1.0 - float(r.distance)) * (1.0 - task_weight)
                })
        
        # 按加權分數排序
        results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return results[:k * 2]  # 返回最多 2k 個結果
    
    def get_task_statistics(self) -> Dict:
        """獲取任務統計資訊"""
        with self.engine.connect() as conn:
            stats = conn.execute(text("""
                SELECT 
                    status,
                    COUNT(*) as count,
                    COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as last_24h
                FROM tasks
                GROUP BY status
            """)).fetchall()
            
            return {
                r.status: {
                    "total": r.count,
                    "last_24h": r.last_24h
                }
                for r in stats
            }
    
    def cleanup_old_tasks(self, days: int = 30):
        """清理舊任務"""
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                DELETE FROM tasks
                WHERE created_at < NOW() - INTERVAL ':days days'
                AND status IN ('completed', 'failed')
            """), {"days": days})
            
            logger.info(f"已清理 {result.rowcount} 筆舊任務")
    
    # ========== SOP 管理方法 ==========
    
    def add_sop(self, title: str, content: str, category: Optional[str] = None, 
                tags: Optional[List[str]] = None, metadata: Optional[Dict] = None) -> str:
        """
        添加 SOP 文檔到知識庫
        
        Args:
            title: SOP 標題
            content: SOP 內容
            category: 分類
            tags: 標籤列表
            metadata: 元數據
        
        Returns:
            sop_id: SOP ID
        """
        sop_id = f"sop_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 生成嵌入向量
        embedding = self.model.encode(f"{title} {content}").astype(np.float32).tolist()
        
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO sops (
                    sop_id, title, content, content_embedding, 
                    category, tags, metadata
                ) VALUES (
                    :sop_id, :title, :content, :embedding,
                    :category, :tags, :metadata
                )
            """), {
                "sop_id": sop_id,
                "title": title,
                "content": content,
                "embedding": embedding,
                "category": category,
                "tags": tags,
                "metadata": json.dumps(metadata) if metadata else None
            })
        
        logger.info(f"SOP 已添加: {sop_id} - {title}")
        return sop_id
    
    def get_sop(self, sop_id: str) -> Optional[Dict]:
        """獲取 SOP 文檔"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT sop_id, title, content, category, tags, metadata, created_at, updated_at
                FROM sops
                WHERE sop_id = :sop_id
            """), {"sop_id": sop_id}).fetchone()
            
            if result:
                return {
                    "sop_id": result.sop_id,
                    "title": result.title,
                    "content": result.content,
                    "category": result.category,
                    "tags": result.tags,
                    "metadata": json.loads(result.metadata) if result.metadata and isinstance(result.metadata, str) else result.metadata,
                    "created_at": result.created_at,
                    "updated_at": result.updated_at
                }
        return None
    
    def list_sops(self, category: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """列出 SOPs"""
        with self.engine.connect() as conn:
            sql = "SELECT sop_id, title, category, tags, created_at FROM sops"
            params = {"limit": limit}
            
            if category:
                sql += " WHERE category = :category"
                params['category'] = category
            
            sql += " ORDER BY created_at DESC LIMIT :limit"
            
            results = conn.execute(text(sql), params).fetchall()
            return [
                {
                    "sop_id": r.sop_id,
                    "title": r.title,
                    "category": r.category,
                    "tags": r.tags,
                    "created_at": r.created_at
                }
                for r in results
            ]
    
    # ========== Session 管理方法 ==========
    
    def create_session(self) -> str:
        """創建新的任務會話 ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def update_task_session(self, task_id: str, session_id: str):
        """更新任務的 session_id"""
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE tasks
                SET session_id = :session_id
                WHERE task_id = :task_id
            """), {
                "task_id": task_id,
                "session_id": session_id
            })


def main():
    """測試任務管理器"""
    manager = TaskManager()
    
    # 測試儲存任務
    test_task = {
        "action": "pickup",
        "target_object": "紅色方塊",
        "target_location": "桌上",
        "confidence": 0.95,
        "parameters": {
            "speed": "normal",
            "precision": "high"
        }
    }
    
    task_id = manager.save_task("抓取桌上的紅色方塊", test_task)
    logger.info(f"\n任務已建立: {task_id}")
    
    # 測試查詢任務
    task = manager.get_task(task_id)
    logger.info(f"\n查詢任務:")
    logger.info(json.dumps(task, indent=2, ensure_ascii=False, default=str))
    
    # 測試獲取待執行任務
    pending = manager.get_pending_tasks()
    logger.info(f"\n待執行任務數量: {len(pending)}")
    
    # 測試更新狀態
    manager.update_task_status(task_id, "executing")
    manager.update_task_status(task_id, "completed", result={"success": True})
    
    # 測試統計
    stats = manager.get_task_statistics()
    logger.info(f"\n任務統計:")
    logger.info(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
