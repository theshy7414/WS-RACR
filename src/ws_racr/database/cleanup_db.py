"""Database maintenance: delete old episodes / perception logs to keep the database small.

Usage: ``python -m ws_racr.database.cleanup_db --help``
"""
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import sys

from ws_racr.config import get_database_config

DB_URL = get_database_config()['url']


class DatabaseCleaner:
    """資料庫清理器"""
    
    def __init__(self, db_url=None):
        self.engine = create_engine(db_url or DB_URL)
    
    def cleanup_old_episodes(self, days=30, keep_latest=10):
        """
        清理舊的 episodes
        
        Args:
            days: 保留最近 N 天的資料（0 = 清除所有）
            keep_latest: 至少保留最新的 N 個 episode（0 = 不保留任何）
        """
        if days == 0 and keep_latest == 0:
            print(f"\n🧹 清除**所有** episodes（不保留任何）...")
        elif days == 0:
            print(f"\n🧹 清除所有舊 episodes（保留最新 {keep_latest} 個）...")
        else:
            print(f"\n🧹 清理 {days} 天前的舊 episodes（保留最新 {keep_latest} 個）...")
        
        with self.engine.begin() as conn:
            # 先查看有多少會被刪除
            if days == 0:
                # 清除所有，只保留最新 N 個
                result = conn.execute(text("""
                    WITH episodes_to_keep AS (
                        SELECT episode_id 
                        FROM episodes 
                        ORDER BY start_time DESC 
                        LIMIT :keep_latest
                    )
                    SELECT COUNT(*) 
                    FROM episodes 
                    WHERE episode_id NOT IN (SELECT episode_id FROM episodes_to_keep)
                """), {"keep_latest": keep_latest})
            else:
                cutoff_date = datetime.now() - timedelta(days=days)
                result = conn.execute(text("""
                    WITH episodes_to_keep AS (
                        SELECT episode_id 
                        FROM episodes 
                        ORDER BY start_time DESC 
                        LIMIT :keep_latest
                    )
                    SELECT COUNT(*) 
                    FROM episodes 
                    WHERE start_time < :cutoff_date 
                    AND episode_id NOT IN (SELECT episode_id FROM episodes_to_keep)
                """), {"cutoff_date": cutoff_date, "keep_latest": keep_latest})
            
            count = result.fetchone()[0]
            
            if count == 0:
                print("   ✅ 沒有需要清理的 episodes")
                return 0
            
            print(f"   發現 {count} 個舊 episodes")
            if days > 0:
                print(f"   刪除日期: {cutoff_date.strftime('%Y-%m-%d')} 之前")
            if keep_latest > 0:
                print(f"   保留最新: {keep_latest} 個")
            else:
                print(f"   ⚠️ 警告：將刪除所有 episodes（不保留任何）")
            
            response = input("   確定要刪除嗎? (y/N): ").strip().lower()
            if response != 'y':
                print("   ❌ 已取消")
                return 0
            
            # 執行刪除 (CASCADE 會自動刪除相關的 perception_logs 等)
            if days == 0:
                result = conn.execute(text("""
                    WITH episodes_to_keep AS (
                        SELECT episode_id 
                        FROM episodes 
                        ORDER BY start_time DESC 
                        LIMIT :keep_latest
                    )
                    DELETE FROM episodes 
                    WHERE episode_id NOT IN (SELECT episode_id FROM episodes_to_keep)
                """), {"keep_latest": keep_latest})
            else:
                result = conn.execute(text("""
                    WITH episodes_to_keep AS (
                        SELECT episode_id 
                        FROM episodes 
                        ORDER BY start_time DESC 
                        LIMIT :keep_latest
                    )
                    DELETE FROM episodes 
                    WHERE start_time < :cutoff_date 
                    AND episode_id NOT IN (SELECT episode_id FROM episodes_to_keep)
                """), {"cutoff_date": cutoff_date, "keep_latest": keep_latest})
            
            deleted = result.rowcount
            print(f"   ✅ 已刪除 {deleted} 個 episodes (及相關記錄)")
            return deleted
    
    def cleanup_old_tasks(self, days=60, status='completed', all_status=False):
        """
        清理舊的已完成任務
        
        Args:
            days: 保留最近 N 天的資料（0 = 清除所有指定狀態的任務）
            status: 只刪除此狀態的任務 (completed/failed)
            all_status: True = 刪除所有狀態的任務（忽略 status 參數）
        """
        if all_status:
            if days == 0:
                print(f"\n🧹 清除**所有狀態**的任務...")
            else:
                print(f"\n🧹 清理 {days} 天前的**所有狀態**任務...")
        else:
            if days == 0:
                print(f"\n🧹 清除所有 {status} 任務...")
            else:
                print(f"\n🧹 清理 {days} 天前的 {status} 任務...")
        
        with self.engine.begin() as conn:
            # 查看數量
            if all_status:
                if days == 0:
                    result = conn.execute(text("SELECT COUNT(*) FROM tasks"))
                else:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    result = conn.execute(text("""
                        SELECT COUNT(*) 
                        FROM tasks 
                        WHERE created_at < :cutoff_date
                    """), {"cutoff_date": cutoff_date})
            else:
                if days == 0:
                    result = conn.execute(text("""
                        SELECT COUNT(*) 
                        FROM tasks 
                        WHERE status = :status
                    """), {"status": status})
                else:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    result = conn.execute(text("""
                        SELECT COUNT(*) 
                        FROM tasks 
                        WHERE created_at < :cutoff_date 
                        AND status = :status
                    """), {"cutoff_date": cutoff_date, "status": status})
            
            count = result.fetchone()[0]
            
            if count == 0:
                status_desc = "所有狀態" if all_status else status
                print(f"   ✅ 沒有需要清理的 {status_desc} 任務")
                return 0
            
            status_desc = "所有狀態" if all_status else status
            print(f"   發現 {count} 個舊任務 ({status_desc})")
            if days > 0:
                print(f"   刪除日期: {cutoff_date.strftime('%Y-%m-%d')} 之前")
            if all_status:
                print(f"   ⚠️ 警告：將刪除所有狀態的任務（包含 failed, pending, executing）")
            
            response = input("   確定要刪除嗎? (y/N): ").strip().lower()
            if response != 'y':
                print("   ❌ 已取消")
                return 0
            
            # 執行刪除
            if all_status:
                if days == 0:
                    result = conn.execute(text("DELETE FROM tasks"))
                else:
                    result = conn.execute(text("""
                        DELETE FROM tasks 
                        WHERE created_at < :cutoff_date
                    """), {"cutoff_date": cutoff_date})
            else:
                if days == 0:
                    result = conn.execute(text("""
                        DELETE FROM tasks 
                        WHERE status = :status
                    """), {"status": status})
                else:
                    result = conn.execute(text("""
                        DELETE FROM tasks 
                        WHERE created_at < :cutoff_date 
                        AND status = :status
                    """), {"cutoff_date": cutoff_date, "status": status})
            
            deleted = result.rowcount
            print(f"   ✅ 已刪除 {deleted} 個任務")
            return deleted
    
    def cleanup_orphaned_records(self):
        """清理孤立的記錄 (沒有關聯 episode 的)"""
        print("\n🧹 清理孤立記錄...")
        
        with self.engine.begin() as conn:
            # 清理沒有對應 episode 的 perception_logs
            result = conn.execute(text("""
                DELETE FROM perception_logs 
                WHERE episode_id NOT IN (SELECT episode_id FROM episodes)
            """))
            
            perception_deleted = result.rowcount
            if perception_deleted > 0:
                print(f"   ✅ 清理了 {perception_deleted} 筆孤立的感知記錄")
            
            # 清理沒有對應 episode 的 object_tracking
            result = conn.execute(text("""
                DELETE FROM object_tracking 
                WHERE episode_id NOT IN (SELECT episode_id FROM episodes)
            """))
            
            tracking_deleted = result.rowcount
            if tracking_deleted > 0:
                print(f"   ✅ 清理了 {tracking_deleted} 筆孤立的物件追蹤記錄")
            
            # 清理沒有對應 episode 的 llm_interactions
            result = conn.execute(text("""
                DELETE FROM llm_interactions 
                WHERE episode_id NOT IN (SELECT episode_id FROM episodes)
            """))
            
            llm_deleted = result.rowcount
            if llm_deleted > 0:
                print(f"   ✅ 清理了 {llm_deleted} 筆孤立的 LLM 互動記錄")
            
            total = perception_deleted + tracking_deleted + llm_deleted
            if total == 0:
                print("   ✅ 沒有孤立記錄")
            
            return total
    
    def vacuum_database(self):
        """
        優化資料庫 (回收空間)
        注意: VACUUM 不能在事務中執行
        """
        print("\n🔧 優化資料庫...")
        
        # 需要使用 autocommit 模式
        conn = self.engine.raw_connection()
        conn.set_isolation_level(0)  # autocommit
        cursor = conn.cursor()
        
        try:
            cursor.execute("VACUUM ANALYZE")
            print("   ✅ 資料庫已優化")
        except Exception as e:
            print(f"   ⚠️ 優化失敗: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def show_statistics(self):
        """顯示資料庫統計"""
        print("\n📊 資料庫統計")
        print("="*60)
        
        with self.engine.connect() as conn:
            # 各表記錄數
            tables = [
                'episodes', 'perception_logs', 'llm_interactions',
                'object_tracking', 'tasks', 'knowledge_fragments', 'sops'
            ]
            
            print("記錄數:")
            total_records = 0
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.fetchone()[0]
                    total_records += count
                    print(f"  {table:25s}: {count:6,d} 筆")
                except Exception as e:
                    print(f"  {table:25s}: 錯誤")
            
            print(f"\n總記錄數: {total_records:,d} 筆")
            
            # 資料庫大小
            result = conn.execute(text("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """))
            size = result.fetchone()[0]
            print(f"資料庫大小: {size}")
            
            # 最舊和最新的 episode
            result = conn.execute(text("""
                SELECT 
                    MIN(start_time) as oldest,
                    MAX(start_time) as newest,
                    MAX(start_time) - MIN(start_time) as span
                FROM episodes
            """))
            row = result.fetchone()
            if row and row[0]:
                print(f"\nEpisode 時間範圍:")
                print(f"  最舊: {row[0]}")
                print(f"  最新: {row[1]}")
                print(f"  跨度: {row[2]}")
        
        print("="*60)


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='資料庫清理工具')
    parser.add_argument('--episodes', type=int, metavar='DAYS',
                       help='清理 N 天前的舊 episodes (0=清除所有，預設: 30)')
    parser.add_argument('--keep-latest', type=int, metavar='N', default=10,
                       help='保留最新 N 個 episodes (0=全部刪除，預設: 10)')
    parser.add_argument('--tasks', type=int, metavar='DAYS',
                       help='清理 N 天前的已完成任務 (0=清除所有，預設: 60)')
    parser.add_argument('--all-tasks', action='store_true',
                       help='刪除所有狀態的任務（包含 failed/pending/executing）')
    parser.add_argument('--orphaned', action='store_true',
                       help='清理孤立記錄')
    parser.add_argument('--vacuum', action='store_true',
                       help='優化資料庫 (回收空間)')
    parser.add_argument('--stats', action='store_true',
                       help='顯示統計資訊')
    parser.add_argument('--auto', action='store_true',
                       help='自動模式 (清理 30 天前 episodes + 60 天前任務)')
    
    args = parser.parse_args()
    
    cleaner = DatabaseCleaner()
    
    # 顯示統計
    if args.stats or not any([args.episodes is not None, args.tasks is not None, args.orphaned, args.vacuum, args.auto]):
        cleaner.show_statistics()
        return
    
    # 自動模式
    if args.auto:
        print("🤖 自動清理模式")
        cleaner.cleanup_old_episodes(days=30, keep_latest=10)
        cleaner.cleanup_old_tasks(days=60)
        cleaner.cleanup_orphaned_records()
        cleaner.vacuum_database()
        cleaner.show_statistics()
        return
    
    # 手動清理
    if args.episodes is not None:
        cleaner.cleanup_old_episodes(days=args.episodes, keep_latest=args.keep_latest)
    
    if args.tasks is not None:
        cleaner.cleanup_old_tasks(days=args.tasks, all_status=args.all_tasks)
    
    if args.orphaned:
        cleaner.cleanup_orphaned_records()
    
    if args.vacuum:
        cleaner.vacuum_database()
    
    # 顯示清理後統計
    cleaner.show_statistics()


if __name__ == "__main__":
    main()
