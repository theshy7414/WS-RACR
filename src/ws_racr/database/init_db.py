"""Database initialisation: creates the pgvector extension, the ``tasks`` table and its indexes.

Usage: ``python -m ws_racr.database.init_db`` (connection settings come from ``.env`` / environment).
"""
from sqlalchemy import create_engine, text

from ws_racr.config import get_database_config

db_config = get_database_config()
DB_URL = db_config['url']
VECTOR_INDEX_LISTS = db_config['vector_index_lists']

def init_database():
    """初始化資料庫"""
    print("🔧 初始化資料庫...")
    
    engine = create_engine(DB_URL)
    
    with engine.begin() as conn:
        # 建立 vector 擴展
        print("📦 建立 vector 擴展...")
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("✅ vector 擴展已就緒")
        except Exception as e:
            print(f"⚠️ vector 擴展警告: {e}")
        
        # 建立 tasks 表
        print("📋 建立 tasks 表...")
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
                telemetry JSONB            -- 執行過程摘要
            )
        """))
        print("✅ tasks 表已建立")
        
        # 為現有表新增欄位（向下兼容）
        print("🔄 檢查並新增欄位...")
        try:
            conn.execute(text("""
                ALTER TABLE tasks
                ADD COLUMN IF NOT EXISTS plan_json JSONB,
                ADD COLUMN IF NOT EXISTS source VARCHAR(20),
                ADD COLUMN IF NOT EXISTS revision_of VARCHAR(100),
                ADD COLUMN IF NOT EXISTS telemetry JSONB
            """))
            print("✅ 欄位已更新（向下兼容）")
        except Exception as e:
            print(f"⚠️ 欄位更新警告: {e}")
        
        # 建立索引
        print("🔍 建立索引...")
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status 
            ON tasks(status)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tasks_created 
            ON tasks(created_at DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tasks_type 
            ON tasks(task_type)
        """))
        
        # 新增 source 和 revision_of 索引
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tasks_source 
            ON tasks(source)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_tasks_revision 
            ON tasks(revision_of)
        """))
        
        # 建立向量索引（用於語義搜尋加速）
        print("🚀 建立向量索引（IVFFlat）...")
        try:
            # 注意：需要有足夠的資料（建議 >100 筆）才能建立 IVFFlat 索引
            # 先檢查資料量
            count_result = conn.execute(text("SELECT COUNT(*) as cnt FROM tasks")).fetchone()
            
            if count_result.cnt >= 100:
                # 資料量足夠，建立 IVFFlat 索引
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_tasks_embedding
                    ON tasks USING ivfflat (task_embedding vector_cosine_ops)
                    WITH (lists = {VECTOR_INDEX_LISTS})
                """))
                print(f"✅ 向量索引已建立（lists={VECTOR_INDEX_LISTS}）")
            else:
                # 資料量不足，使用基本索引
                print(f"⚠️ 資料量不足（{count_result.cnt} < 100），使用基本向量索引")
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_embedding
                    ON tasks USING ivfflat (task_embedding vector_cosine_ops)
                    WITH (lists = 10)
                """))
                print("✅ 基本向量索引已建立（lists=10）")
        except Exception as e:
            print(f"⚠️ 向量索引建立警告: {e}")
            print("   語義搜尋仍可運作，但效能可能較慢")
        
        print("✅ 索引已建立")
        
        # 顯示表資訊
        result = conn.execute(text("""
            SELECT COUNT(*) as count FROM tasks
        """)).fetchone()
        
        print(f"\n📊 現有任務數量: {result.count}")
    
    print("\n✅ 資料庫初始化完成！")


def reset_database():
    """重置資料庫（危險操作！）"""
    print("⚠️ 警告：這將刪除所有任務資料！")
    print("確定要繼續嗎? (yes/no): ", end='')
    
    if input().lower() != 'yes':
        print("❌ 已取消")
        return
    
    engine = create_engine(DB_URL)
    
    with engine.begin() as conn:
        print("🗑️ 刪除 tasks 表...")
        conn.execute(text("DROP TABLE IF EXISTS tasks CASCADE"))
        print("✅ 已刪除")
    
    print("\n🔄 重新初始化...")
    init_database()


def show_database_info():
    """顯示資料庫資訊"""
    engine = create_engine(DB_URL)
    
    with engine.connect() as conn:
        # 表資訊
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                COUNT(CASE WHEN status = 'executing' THEN 1 END) as executing,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
            FROM tasks
        """)).fetchone()
        
        print("📊 資料庫資訊:")
        print(f"   總任務數: {result.total}")
        print(f"   待執行: {result.pending}")
        print(f"   執行中: {result.executing}")
        print(f"   已完成: {result.completed}")
        print(f"   失敗: {result.failed}")
        
        # 最近的任務
        recent = conn.execute(text("""
            SELECT task_id, user_input, status, created_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT 5
        """)).fetchall()
        
        if recent:
            print("\n📝 最近的任務:")
            for r in recent:
                print(f"   [{r.status}] {r.user_input} ({r.created_at})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'reset':
            reset_database()
        elif sys.argv[1] == 'info':
            show_database_info()
        else:
            print("使用方式:")
            print("  python init_db.py        # 初始化資料庫")
            print("  python init_db.py reset  # 重置資料庫")
            print("  python init_db.py info   # 顯示資訊")
    else:
        init_database()
