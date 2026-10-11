"""
Migration: re-embed all completed tasks with the new model and clean text format.

Run this once after switching the embedding model from
  all-MiniLM-L6-v2  →  paraphrase-multilingual-MiniLM-L12-v2
and after fixing _build_embedding_text() in TaskManager.

Usage:
    python tools/migrate_reembed_tasks.py
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

from ws_racr.database.task_manager import TaskManager
from sqlalchemy import text


def main():
    print("Re-embedding migration")
    print(f"  New model : paraphrase-multilingual-MiniLM-L12-v2")
    print(f"  Text format: action + object + location (clean, no raw dicts)")
    print()

    tm = TaskManager()

    # Fetch all tasks that have a task_embedding (regardless of status)
    with tm.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT task_id, user_input, task_json
            FROM tasks
            WHERE task_embedding IS NOT NULL
        """)).fetchall()

    total = len(rows)
    print(f"  Found {total} tasks to re-embed.\n")

    batch_size = 50
    updated = 0
    errors = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        try:
            task_json = json.loads(row.task_json) if isinstance(row.task_json, str) else row.task_json
            new_text = TaskManager._build_embedding_text(task_json, row.user_input)
            new_emb = tm.model.encode(new_text).astype(np.float32).tolist()

            with tm.engine.begin() as conn:
                conn.execute(text("""
                    UPDATE tasks
                    SET task_embedding = :emb
                    WHERE task_id = :tid
                """), {"emb": str(new_emb), "tid": row.task_id})

            updated += 1

            if (i + 1) % batch_size == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1:>4}/{total}]  re-embedded  ({rate:.1f} tasks/s)  "
                      f"last text: {repr(new_text[:60])}")
        except Exception as e:
            errors += 1
            print(f"  [ERROR] task_id={row.task_id}: {e}")

    elapsed = time.time() - t0
    print(f"\n  Done: {updated} updated, {errors} errors, {elapsed:.1f}s total")


if __name__ == "__main__":
    main()
