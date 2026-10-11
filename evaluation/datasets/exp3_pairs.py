"""
EXP3 command pairs — 80 組帶 embedding 相似度的指令配對（作者標註）
=====================================================================
資料集內容：
  共 80 組 (current_cmd, retrieved_cmd) 配對，每組模擬一次 RAG 查詢情境：
    current_cmd   = 使用者當前指令
    retrieved_cmd = 向量檢索回傳的最相似歷史計畫之原始指令
    ret_type      = 該歷史計畫的 task_type（pickup / handover / place / move / gripper）
    real_sim      = 兩句指令的 embedding cosine 相似度，由
                    paraphrase-multilingual-MiniLM-L12-v2 實際計算（非手填）
    gt            = 作者標註的期望決策：
                      REUSE   歷史計畫可直接使用（軌跡與參數皆適用）
                      ADAPT   動作類型相容，但需調整參數（物件／位置等）
                      CLARIFY 計畫對本次指令無幫助或資訊不足，應向使用者確認
    note          = 該組配對的簡短說明

分組（A–D 各 n=20，ID 對照見 exp3_ids_for_group）：
  Group A  同 task_type、同物件               → REUSE（P01–P10, P37–P46）
  Group B  同 task_type、物件／目標不同       → ADAPT（P11–P20, P47–P56）
  Group C  不同 task_type（高相似度仍動作不同）→ ADAPT / CLARIFY（P21–P28, P57–P64, P81–P84）
  Group D  低相似度 / 問句 / 模糊詞           → CLARIFY（P29–P36, P65–P74, P85–P86）

用法：
  python3 evaluation/datasets/exp3_pairs.py   # 依分組列出所有配對與 gt 分布

版本紀錄：
  v2: 由 36 組擴增為 80 組（P37–P80）
  v3: 移除 Group E（P75–P80）；Group C、D 各補至 20 組（P81–P86），總數維持 80 組
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Pair:
    id: str
    current_cmd: str        # 使用者當前指令
    retrieved_cmd: str      # 檢索到的歷史計畫之原始指令
    ret_type: str           # 歷史計畫的 task_type（pickup/handover/place/move/gripper）
    real_sim: float         # embedding cosine 相似度（實際計算，非手填）
    gt: str                 # 作者標註：REUSE / ADAPT / CLARIFY
    note: str


PAIRS: List[Pair] = [

    # ── Group A: 同 task_type + 高 sim（預期 REUSE）────────────────────────
    Pair("P01", "幫我拿圓柱體",       "幫我拿圓柱體",          "pickup",   1.000, "REUSE",   "完全相同"),
    Pair("P02", "拿起圓柱體",         "幫我拿圓柱體",          "pickup",   0.889, "REUSE",   "同語意，不同措辭"),
    Pair("P03", "把圓柱體夾起來",     "幫我拿圓柱體",          "pickup",   0.874, "REUSE",   "夾/拿 同義"),
    Pair("P04", "把螺絲遞給我",       "把螺絲交給我",          "handover", 0.995, "REUSE",   "遞/交 同義"),
    Pair("P05", "移動到組裝區",       "去組裝區",              "move",     0.910, "REUSE",   "move 同義"),
    Pair("P06", "打開夾爪",           "把夾爪打開",            "gripper",  0.993, "REUSE",   "gripper 同義"),
    Pair("P07", "拿起螺絲",           "拿螺絲",                "pickup",   0.974, "REUSE",   "省略助詞"),
    Pair("P08", "把支架交給操作員",   "把支架遞給我",          "handover", 0.587, "CLARIFY", "接收者不同（操作員 vs 我）→ 目標位置完全未知，需問人"),
    Pair("P09", "移到待機位置",       "移動到待機位",          "move",     0.965, "REUSE",   "待機 move 同義"),
    Pair("P10", "關閉夾爪",           "把夾爪關上",            "gripper",  0.976, "REUSE",   "gripper 關 同義"),

    # ── Group B: 同 task_type + 物件不同（預期 ADAPT）─────────────────────
    Pair("P11", "幫我拿螺絲",         "幫我拿圓柱體",          "pickup",   0.751, "ADAPT",   "pickup，物件不同（圓柱→螺絲）"),
    Pair("P12", "把支架夾起來",       "拿起圓柱體",            "pickup",   0.766, "ADAPT",   "pickup，物件不同"),
    Pair("P13", "把螺絲交給我",       "把圓柱體遞給操作員",    "handover", 0.535, "ADAPT",   "handover，物件不同，接收者不同"),
    Pair("P14", "把六角扳手遞給我",   "把支架交給我",          "handover", 0.875, "ADAPT",   "handover，物件不同（扳手 vs 支架），sim 高但物件不同"),
    Pair("P15", "移動到裝配位置",     "移動到組裝區",          "move",     0.897, "REUSE",   "裝配位置 = 組裝區，同個位置，REUSE"),
    Pair("P16", "移到右側工作區",     "移動到左側",            "move",     0.746, "ADAPT",   "move，方向相反"),
    Pair("P17", "把圓柱體放到右側",   "把螺絲放到組裝台",      "place",    0.533, "ADAPT",   "place，物件和目標都不同"),
    Pair("P18", "把支架放到平台上",   "把圓柱放到組裝台上",    "place",    0.801, "ADAPT",   "place，物件不同"),
    Pair("P19", "拿起大螺絲",         "拿起小螺絲",            "pickup",   0.914, "ADAPT",   "pickup，同物件族但尺寸不同，sim=0.914 但抓取參數不同"),
    Pair("P20", "把扳手移到工作台",   "把螺絲移到組裝台",      "place",    0.653, "ADAPT",   "place，物件不同"),

    # ── Group C: 不同 task_type + 同物件（高 sim 但動作不同，危險）─────────
    Pair("P21", "幫我拿圓柱體",       "把圓柱體交給我",        "handover", 0.939, "ADAPT",   "pickup query vs handover plan，sim=0.939 但軌跡不同"),
    Pair("P22", "把圓柱體交給我",     "幫我拿圓柱體",          "pickup",   0.939, "ADAPT",   "handover query vs pickup plan"),
    Pair("P23", "把螺絲放到組裝台",   "把螺絲交給我",          "handover", 0.781, "ADAPT",   "place query vs handover plan，同物件"),
    Pair("P24", "把螺絲交給我",       "把螺絲放到組裝台",      "place",    0.781, "ADAPT",   "handover query vs place plan，同物件"),
    Pair("P25", "拿起支架",           "把支架放到工作台",      "place",    0.693, "CLARIFY", "pickup vs place，計畫完全沒有幫助，問人"),
    Pair("P26", "把支架放到組裝台",   "拿起支架",              "pickup",   0.809, "CLARIFY", "place vs pickup，計畫完全沒有用，問人"),
    Pair("P27", "打開夾爪",           "移動到組裝區",          "move",     0.436, "CLARIFY", "gripper vs move，毫無關係"),
    Pair("P28", "移動到組裝區",       "打開夾爪",              "gripper",  0.436, "CLARIFY", "move vs gripper"),

    # ── Group D: 低 sim / 完全不同（預期 CLARIFY）────────────────────────
    Pair("P29", "下一步是什麼",       "幫我拿圓柱體",          "pickup",   0.280, "CLARIFY", "問句 vs 動作計畫"),
    Pair("P30", "告訴我SOP進度",      "把螺絲交給我",          "handover", 0.572, "CLARIFY", "問句 vs 動作計畫，完全不同域"),
    Pair("P31", "移動到待機位置",     "把螺絲交給我",          "handover", 0.499, "CLARIFY", "move vs handover，低 sim"),
    Pair("P32", "幫我拿圓柱體",       "移動到組裝區",          "move",     0.578, "CLARIFY", "pickup vs move，action 和 object 都不同"),
    Pair("P33", "繼續",               "把螺絲放到組裝台",      "place",    0.421, "CLARIFY", "模糊詞，無法對應任何計畫"),
    Pair("P34", "打開夾爪",           "把支架交給操作員",      "handover", 0.585, "CLARIFY", "gripper vs handover，都不符"),
    Pair("P35", "把螺絲鎖緊",         "移動到待機位",          "move",     0.509, "CLARIFY", "完全不相關"),
    Pair("P36", "把平台對齊",         "把螺絲遞給我",          "handover", 0.549, "CLARIFY", "不同動作不同物件"),

    # ════════════════════════════════════════════════════════════════════
    # v2 擴增（P37–P80）：新增 44 pairs，更廣泛的物件/動作覆蓋
    # 所有 sim 值由 paraphrase-multilingual-MiniLM-L12-v2 真實計算
    # ════════════════════════════════════════════════════════════════════

    # ── Group A-2: 同 task_type + 同物件 → REUSE ─────────────────────
    Pair("P37", "拿起螺帽",              "取得螺帽",               "pickup",   0.987, "REUSE",   "新物件螺帽，同動作同物件"),
    Pair("P38", "把支架交給我",           "遞支架給我",              "handover", 0.938, "REUSE",   "遞/交 同義，同物件"),
    Pair("P39", "移動到待機區",           "去待機區",               "move",     0.903, "REUSE",   "移動/去 同義"),
    Pair("P40", "把圓柱體放到工作台",     "將圓柱放置於工作台",      "place",    0.969, "REUSE",   "文言/口語表達同義"),
    Pair("P41", "夾取六角扳手",           "拿六角扳手",              "pickup",   0.985, "REUSE",   "夾取/拿 同義"),
    Pair("P42", "移動到原點",             "回到原點",               "move",     0.814, "REUSE",   "移動到/回到原點 語意相同（Layer 3 規則對此組輸出 ADAPT，屬邊界案例）"),
    Pair("P43", "把螺絲遞給操作員",       "把螺絲交給操作員",        "handover", 0.993, "REUSE",   "遞/交 同義，接收者相同"),
    Pair("P44", "拿圓柱",                 "幫我取圓柱體",            "pickup",   0.888, "REUSE",   "不同說法同語意"),
    Pair("P45", "把支架放到組裝台",       "把支架放置在組裝台上",    "place",    0.909, "REUSE",   "放到/放置在 同義"),
    Pair("P46", "移動到組裝台旁",         "前往組裝台",              "move",     0.935, "REUSE",   "移動到/前往 同義"),

    # ── Group B-2: 同 task_type + 物件/目標不同 → ADAPT ──────────────
    Pair("P47", "拿起螺帽",              "拿起支架",               "pickup",   0.851, "ADAPT",   "pickup，物件不同（螺帽 vs 支架），高 sim"),
    Pair("P48", "把平台交給我",           "把螺絲交給我",            "handover", 0.679, "ADAPT",   "handover，物件不同（平台 vs 螺絲）"),
    Pair("P49", "把螺帽放到工作台",       "把支架放到工作台",        "place",    0.876, "ADAPT",   "place，物件不同，目標相同"),
    Pair("P50", "拿六角扳手",             "拿螺帽",                 "pickup",   0.686, "ADAPT",   "pickup，物件不同"),
    Pair("P51", "移動到組裝台",           "移動到待機區",            "move",     0.851, "ADAPT",   "move，目標位置不同"),
    Pair("P52", "把圓柱遞給操作員",       "把支架遞給操作員",        "handover", 0.922, "ADAPT",   "handover，物件不同，接收者相同，高 sim"),
    Pair("P53", "把螺絲放在工作台",       "把圓柱放在組裝台",        "place",    0.646, "ADAPT",   "place，物件和目標都不同"),
    Pair("P54", "夾取螺帽",              "夾取圓柱",               "pickup",   0.761, "ADAPT",   "pickup，物件不同"),
    Pair("P55", "移到右側工作區",         "移到左側工作區",          "move",     0.932, "ADAPT",   "move，左右方向不同，高 sim"),
    Pair("P56", "把螺帽交給我",           "把六角扳手交給我",        "handover", 0.821, "ADAPT",   "handover，物件不同"),

    # ── Group C-2: 不同 task_type → ADAPT（相容對）or CLARIFY（不相容）
    Pair("P57", "拿起螺帽",              "把螺帽交給我",            "handover", 0.886, "ADAPT",   "pickup vs handover（相容），同物件，sim≥0.75 → ADAPT"),
    Pair("P58", "把圓柱放到台上",         "把圓柱交給操作員",        "handover", 0.715, "CLARIFY", "place vs handover（相容），但 sim=0.715<0.75 → CLARIFY"),
    Pair("P59", "把六角扳手交給我",       "把六角扳手放到工作台",    "place",    0.774, "ADAPT",   "handover vs place（相容），sim≥0.75 → ADAPT"),
    Pair("P60", "拿起支架",              "把支架放到組裝台",        "place",    0.809, "CLARIFY", "pickup vs place（不相容，方向相反）→ CLARIFY"),
    Pair("P61", "把螺絲放到工作台",       "拿起螺絲",               "pickup",   0.786, "CLARIFY", "place vs pickup（不相容，方向相反）→ CLARIFY"),
    Pair("P62", "打開夾爪",              "拿起螺絲",               "pickup",   0.746, "CLARIFY", "gripper vs pickup（不相容）→ CLARIFY"),
    Pair("P63", "移動到組裝台",           "拿起圓柱",               "pickup",   0.665, "CLARIFY", "move vs pickup（不相容）→ CLARIFY"),
    Pair("P64", "把圓柱遞給我",           "把支架放到組裝台",        "place",    0.682, "CLARIFY", "handover vs place（相容），但 sim=0.682<0.75 → CLARIFY"),

    # ── Group D-2: 低 sim / 問句 / 模糊詞 → CLARIFY ──────────────────
    Pair("P65", "告訴我第幾步",           "把螺絲交給我",            "handover", 0.649, "CLARIFY", "問句 vs 動作"),
    Pair("P66", "這個步驟怎麼做",         "移動到組裝區",            "move",     0.490, "CLARIFY", "問句 vs 動作"),
    Pair("P67", "好了",                  "拿起圓柱",               "pickup",   0.580, "CLARIFY", "模糊確認詞"),
    Pair("P68", "繼續下一步",             "把螺絲放到工作台",        "place",    0.429, "CLARIFY", "模糊進度詞"),
    Pair("P69", "停止",                  "把支架交給我",            "handover", 0.550, "CLARIFY", "停止指令 vs 動作計畫"),
    Pair("P70", "拿起螺帽",              "移動到組裝區",            "move",     0.465, "CLARIFY", "pickup vs move，完全不同域"),
    Pair("P71", "把支架鎖緊",             "把支架放到工作台",        "place",    0.572, "CLARIFY", "鎖緊 vs place，不相關"),
    Pair("P72", "夾爪全開",              "移動到左側",              "move",     0.516, "CLARIFY", "gripper vs move"),
    Pair("P73", "告訴我SOP",             "拿起六角扳手",            "pickup",   0.489, "CLARIFY", "問句 vs pickup"),
    Pair("P74", "可以了",                "把圓柱遞給我",            "handover", 0.599, "CLARIFY", "模糊確認詞"),

    # ── Group C 擴充（v3：+4，與 P21–P28、P57–P64 同類）──────────────────
    Pair("P81", "幫我拿螺絲",             "把螺絲遞給我",            "handover", 0.935, "ADAPT",   "pickup vs handover 相容、同物件、高 sim"),
    Pair("P82", "把支架放到組裝台",       "把支架交給我",            "handover", 0.651, "CLARIFY", "place vs handover 相容但 sim<0.75 → CLARIFY"),
    Pair("P83", "拿起圓柱體",           "把圓柱體鎖到支架上",      "place",    0.796, "CLARIFY", "pickup vs place 不相容，高 sim 仍不應 REUSE"),
    Pair("P84", "移動到工作台",         "打開夾爪",                "gripper",  0.444, "CLARIFY", "move vs gripper，低 sim"),

    # ── Group D 擴充（v3：+2，與 P29–P36、P65–P74 同類）──────────────────
    Pair("P85", "下一步要做什麼",       "拿起六角扳手",            "pickup",   0.337, "CLARIFY", "問句 vs 動作計畫，低 sim"),
    Pair("P86", "繼續執行任務",         "把螺絲放到組裝台",        "place",    0.496, "CLARIFY", "模糊進度詞 vs 具體 place 計畫"),
]


def exp3_ids_for_group(letter: str) -> List[str]:
    """Thesis stratification: Groups A–D, 20 pair IDs each (v3, no Group E)."""
    if letter == "A":
        return [f"P{i:02d}" for i in range(1, 11)] + [f"P{i:02d}" for i in range(37, 47)]
    if letter == "B":
        return [f"P{i:02d}" for i in range(11, 21)] + [f"P{i:02d}" for i in range(47, 57)]
    if letter == "C":
        return (
            [f"P{i:02d}" for i in range(21, 29)]
            + [f"P{i:02d}" for i in range(57, 65)]
            + ["P81", "P82", "P83", "P84"]
        )
    if letter == "D":
        return [f"P{i:02d}" for i in range(29, 37)] + [f"P{i:02d}" for i in range(65, 75)] + ["P85", "P86"]
    raise ValueError(f"unknown group letter: {letter!r} (expected A–D)")


def main():
    print("=" * 80)
    print("EXP3 Command Pairs — 80 pairs, author-annotated GT")
    print("=" * 80)
    print(f"  sim 由 paraphrase-multilingual-MiniLM-L12-v2 實際計算")
    print()

    for _g in "ABCD":
        assert len(exp3_ids_for_group(_g)) == 20

    groups = [
        ("A", "同 task_type + 同物件 → REUSE（P01–P10 + P37–P46）"),
        ("B", "同 task_type + 物件/目標不同 → ADAPT（P11–P20 + P47–P56）"),
        ("C", "不同 task_type → ADAPT/CLARIFY（P21–P28 + P57–P64 + P81–P84）"),
        ("D", "低 sim / 問句 / 模糊詞 → CLARIFY（P29–P36 + P65–P74 + P85–P86）"),
    ]

    for grp, desc in groups:
        want = set(exp3_ids_for_group(grp))
        grp_pairs = [p for p in PAIRS if p.id in want]
        grp_pairs.sort(key=lambda p: int(p.id[1:]))
        print(f"── Group {grp}: {desc}")
        print(f"  {'ID':>4}  {'sim':>5}  {'ret_type':>9}  {'GT':>8}  current → retrieved")
        for p in grp_pairs:
            print(f"  {p.id:>4}  {p.real_sim:.3f}  {p.ret_type:>9}  {p.gt:>8}  "
                  f"[{p.current_cmd}] ← [{p.retrieved_cmd}]")
        print()

    reuse_n   = sum(1 for p in PAIRS if p.gt == "REUSE")
    adapt_n   = sum(1 for p in PAIRS if p.gt == "ADAPT")
    clarify_n = sum(1 for p in PAIRS if p.gt == "CLARIFY")
    print(f"GT 分布：REUSE={reuse_n}  ADAPT={adapt_n}  CLARIFY={clarify_n}  Total={len(PAIRS)}")


if __name__ == "__main__":
    main()
