"""
Rich SOP content for RAG-based question answering.

Each assembly step is broken into focused chunks by question type.
Vector search scores are higher when each chunk covers a single topic,
so one step → multiple entries (procedure / materials / robot_help /
troubleshoot / safety) rather than one big blob.

HOW TO ADD CONTENT
──────────────────
Fill in the `content` field for each entry below.  Leave it as an
empty string ("") if you have nothing to say for that chunk yet —
the seeder skips empty entries automatically.

CHUNK TYPES and the questions they answer
──────────────────────────────────────────
  procedure   : 怎麼做？步驟是什麼？操作程序？
  materials   : 需要什麼工具/零件？用什麼？
  robot_help  : 機器人可以怎麼幫？什麼時候叫機器人？
  troubleshoot: 失敗了怎麼辦？為什麼對不上？常見錯誤？
  safety      : 要注意什麼？會不會危險？有什麼限制？

SEEDING
───────
Run once to push this content into the `sops` table:
    python -m robot_control.control_modules.sop_rich_content
Or it is called automatically by HRCOrchestrator on first start
(after the minimal _STEP_KNOWLEDGE seed, if table count < EXPECTED_ROWS).
"""

from __future__ import annotations
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────
# Rich SOP entries
# One dict per chunk.  Required keys: step_class, type, title, content
# Optional : tags (List[str]), extra_metadata (Dict)
# ──────────────────────────────────────────────────────────────────

RICH_SOP_ENTRIES = [

# ═════════════════════════════════════
# Step 1: place_platform
# ═════════════════════════════════════
{
    "step_class": "place_platform",
    "type": "procedure",
    "content": "將桌面雜物移至右側，將平台放置於左側操作區，並保持橫向擺放以利後續操作。",
},
{
    "step_class": "place_platform",
    "type": "materials",
    "content": "平台",
},
{
    "step_class": "place_platform",
    "type": "robot_action",
    "content": "機器人不直接夾取平台，而是透過語音或指示協助人類將平台放置於正確位置。",
},
{
    "step_class": "place_platform",
    "type": "troubleshoot",
    "content": "若平台方向錯誤（直立），需重新調整為橫放，避免影響後續孔位對齊。",
},
{
    "step_class": "place_platform",
    "type": "rationale",
    "content": "平台方向會影響後續圓柱與螺絲孔位的對齊，若放置錯誤將導致組裝失敗。",
},
{
    "step_class": "place_platform",
    "type": "safety",
    "content": "避免在放置平台時手部位於平台下方，以防壓傷。",
},

# ═════════════════════════════════════
# Step 2: insert_cylinder_under_platform
# ═════════════════════════════════════
{
    "step_class": "insert_cylinder_under_platform",
    "type": "procedure",
    "content": "將平台抬起，將四個圓柱分別放置於對應孔位下方並對齊，然後放下平台使圓柱對齊孔位。",
},
{
    "step_class": "insert_cylinder_under_platform",
    "type": "materials",
    "content": "4個圓柱 + 平台",
},
{
    "step_class": "insert_cylinder_under_platform",
    "type": "robot_action",
    "content": "機器人可拾取圓柱並遞交給人類，或引導圓柱至孔位上方協助對齊。",
},
{
    "step_class": "insert_cylinder_under_platform",
    "type": "troubleshoot",
    "content": "若圓柱傾斜或未對齊，需重新調整位置，確保與孔位垂直對齊。",
},
{
    "step_class": "insert_cylinder_under_platform",
    "type": "rationale",
    "content": "圓柱需先正確對齊，否則後續螺絲將無法鎖入，導致結構不穩定。",
},
{
    "step_class": "insert_cylinder_under_platform",
    "type": "safety",
    "content": "放下平台時避免夾到手指。",
},

# ═════════════════════════════════════
# Step 3: tighten_small_screw_under_platform
# ═════════════════════════════════════
{
    "step_class": "tighten_small_screw_under_platform",
    "type": "procedure",
    "content": "將四個小螺絲放入孔位後，使用六角板手逐一鎖緊。",
},
{
    "step_class": "tighten_small_screw_under_platform",
    "type": "materials",
    "content": "4個小螺絲 + 六角板手",
},
{
    "step_class": "tighten_small_screw_under_platform",
    "type": "robot_action",
    "content": "機器人可遞交六角板手或螺絲，並在需要時穩定結構避免旋轉。",
},
{
    "step_class": "tighten_small_screw_under_platform",
    "type": "troubleshoot",
    "content": "若螺絲無法旋入，通常是圓柱未對齊，需先調整圓柱位置。",
},
{
    "step_class": "tighten_small_screw_under_platform",
    "type": "rationale",
    "content": "鎖緊螺絲可固定圓柱位置，確保整體結構穩定。",
},
{
    "step_class": "tighten_small_screw_under_platform",
    "type": "safety",
    "content": "使用工具時避免滑脫造成手部受傷。",
},

# ═════════════════════════════════════
# Step 4: place_bracket_on_table
# ═════════════════════════════════════
{
    "step_class": "place_bracket_on_table",
    "type": "procedure",
    "content": "將支架放置於操作區並保持橫向擺放。",
},
{
    "step_class": "place_bracket_on_table",
    "type": "materials",
    "content": "支架",
},
{
    "step_class": "place_bracket_on_table",
    "type": "robot_action",
    "content": "機器人透過語音提示協助定位支架。",
},
{
    "step_class": "place_bracket_on_table",
    "type": "troubleshoot",
    "content": "若方向錯誤需重新擺放。",
},
{
    "step_class": "place_bracket_on_table",
    "type": "rationale",
    "content": "正確方向有助於後續PI工件對齊。",
},

# ═════════════════════════════════════
# Step 5: align_pi_on_bracket
# ═════════════════════════════════════
{
    "step_class": "align_pi_on_bracket",
    "type": "procedure",
    "content": "將PI工件與支架對齊並固定。",
},
{
    "step_class": "align_pi_on_bracket",
    "type": "robot_action",
    "content": "機器人可遞交PI工件或協助定位。",
},
{
    "step_class": "align_pi_on_bracket",
    "type": "troubleshoot",
    "content": "若孔位未對齊需重新調整位置。",
},
{
    "step_class": "align_pi_on_bracket",
    "type": "rationale",
    "content": "正確對齊可避免後續螺絲安裝困難。",
},

# ═════════════════════════════════════
# Step 6: insert_upper_big_screw
# ═════════════════════════════════════
{
    "step_class": "insert_upper_big_screw",
    "type": "procedure",
    "content": "將大螺絲插入並鎖緊螺帽固定結構。",
},
{
    "step_class": "insert_upper_big_screw",
    "type": "materials",
    "content": "大螺絲 + 螺帽",
},
{
    "step_class": "insert_upper_big_screw",
    "type": "robot_action",
    "content": "機器人可遞交螺絲或螺帽。",
},
{
    "step_class": "insert_upper_big_screw",
    "type": "troubleshoot",
    "content": "若無法插入，需確認對齊狀態。",
},
{
    "step_class": "insert_upper_big_screw",
    "type": "rationale",
    "content": "固定上方結構以確保穩定性。",
},

# ═════════════════════════════════════
# Step 7: align_bracket_to_platform
# ═════════════════════════════════════
{
    "step_class": "align_bracket_to_platform",
    "type": "procedure",
    "content": "將支架與平台對齊並固定。",
},
{
    "step_class": "align_bracket_to_platform",
    "type": "robot_action",
    "content": "機器人提供語音協助對齊。",
},
{
    "step_class": "align_bracket_to_platform",
    "type": "troubleshoot",
    "content": "若對齊失敗需重新定位。",
},
{
    "step_class": "align_bracket_to_platform",
    "type": "rationale",
    "content": "確保整體結構組裝一致。",
},

# ═════════════════════════════════════
# Step 8: insert_lower_big_screw
# ═════════════════════════════════════
{
    "step_class": "insert_lower_big_screw",
    "type": "procedure",
    "content": "插入下方大螺絲並鎖緊螺帽。",
},
{
    "step_class": "insert_lower_big_screw",
    "type": "materials",
    "content": "大螺絲 + 螺帽",
},
{
    "step_class": "insert_lower_big_screw",
    "type": "robot_action",
    "content": "機器人可遞交螺絲或穩定結構。",
},
{
    "step_class": "insert_lower_big_screw",
    "type": "troubleshoot",
    "content": "若插入困難需確認對齊。",
},
{
    "step_class": "insert_lower_big_screw",
    "type": "rationale",
    "content": "完成最終固定。",
},
{
    "step_class": "insert_lower_big_screw",
    "type": "safety",
    "content": "避免在立起結構時手部被夾住。",
},

]

# ──────────────────────────────────────────────────────────────────
# Standalone seeder  (python -m ws_racr.memory.sop_rich_content)
# ──────────────────────────────────────────────────────────────────

def seed(task_manager) -> int:
    """Push RICH_SOP_ENTRIES into the sops table.

    Skips entries where content is empty.
    Auto-generates a title from step_class + type when not provided.
    Returns the number of entries actually inserted.
    """
    from ws_racr.memory.sop_step_knowledge import STEP_KNOWLEDGE
    _TYPE_ZH = {
        "procedure":    "操作程序",
        "materials":    "所需材料",
        "robot_help":   "機器人協助",
        "robot_action": "機器人協助",
        "troubleshoot": "常見問題",
        "safety":       "注意事項",
        "rationale":    "設計原因",
    }
    inserted = 0
    for entry in RICH_SOP_ENTRIES:
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        step_class = entry["step_class"]
        chunk_type = entry["type"]
        # Auto-generate title if not provided
        if entry.get("title"):
            title = entry["title"]
        else:
            step_info = STEP_KNOWLEDGE.get(step_class, {})
            step_no   = step_info.get("step", "?")
            step_zh   = step_info.get("zh", step_class)
            type_zh   = _TYPE_ZH.get(chunk_type, chunk_type)
            title     = f"步驟{step_no} {step_zh} - {type_zh}"
        tags = entry.get("tags") or [step_class, chunk_type]
        metadata = {
            "step_class": step_class,
            "chunk_type": chunk_type,
            **(entry.get("extra_metadata") or {}),
        }
        task_manager.add_sop(
            title=title,
            content=content,
            category=f"assembly_sop_{chunk_type}",
            tags=tags,
            metadata=metadata,
        )
        inserted += 1
    return inserted


if __name__ == "__main__":
    try:
        from ws_racr.database.task_manager import TaskManager
        tm = TaskManager()
        n = seed(tm)
        print(f"✅ Rich SOP seeded: {n} entries inserted")
    except Exception as e:
        print(f"❌ Seeding failed: {e}")
        raise
