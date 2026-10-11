"""
Annotated datasets for the thesis experiments (EXP1, EXP2)
==========================================================
Author-annotated cases used by the offline evaluation scripts:

EXP1 - Intent understanding (Layer 1)       100 NL commands, groups A-F
EXP2 - Execution feasibility (Layer 2)       60 (command, world-state, plan) scenarios

The EXP3 pairs (80 command pairs with embedding similarities) live in
``evaluation/datasets/exp3_pairs.py``; EXP4 builds its scenarios in
``evaluation/exp4_e2e_ablation.py``.

EXP2 ground-truth rules (applied in order, first match wins)
------------------------------------------------------------
  REJECT  : the retrieved plan requires an object that is NOT in ws_objects
  REUSE   : the required object is present (the SOP step is not a hard
            constraint for Layer 2; current_step / retrieved_step remain in
            each case for logging only)
  CLARIFY : ambiguous / incomplete information (fallthrough; unused in the
            60-case grid)

The labels are author annotations derived from these rule definitions, which
are the same definitions Layer 2 implements.

Each case carries a 'rationale' field that states the rule applied.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentCase:
    """EXP1: Intent-classification case."""
    id: str
    command: str                        # NL command (Chinese)
    intents: List[str]                  # ordered GT intent list
    is_multi: bool                      # contains 2+ distinct intents
    catastrophic_if_wrong: bool         # ANSWER routed as EXECUTE → dangerous
    group: str                          # A/B/C/D/E/F
    rationale: str


@dataclass
class FeasibilityCase:
    """EXP2: Feasibility-gate evaluation case."""
    id: str
    command: str
    retrieved_plan_action: str          # e.g. "pickup"
    retrieved_plan_object: Optional[str]  # object the plan requires (None = no object)
    ws_objects: List[str]               # objects currently visible in workspace
    current_step: int                   # current assembly step index
    retrieved_step: int                 # step index the retrieved plan was designed for
    expected: str                       # REUSE | REJECT | ADAPT | CLARIFY
    rationale: str


# ─────────────────────────────────────────────────────────────────────────────
# EXP1 – Intent Understanding (100 cases)
# ─────────────────────────────────────────────────────────────────────────────
# Groups:
#   A – Pure EXECUTE  (30)
#   B – Pure ANSWER   (20)
#   C – CLARIFY needed (10) — too vague to execute or answer
#   D – EXECUTE + ANSWER  (20) — multi-intent, execute first
#   E – ANSWER + EXECUTE  (10) — multi-intent, question first
#   F – EXECUTE + EXECUTE (10) — two sequential physical actions

EXP1_CASES: List[IntentCase] = [

    # ── Group A: Pure EXECUTE (30) ───────────────────────────────────────────
    IntentCase("E1-A01", "幫我拿圓柱體",
               ["EXECUTE"], False, False, "A",
               "Clear pickup command, single physical action."),
    IntentCase("E1-A02", "拿起螺絲",
               ["EXECUTE"], False, False, "A",
               "Unambiguous pickup of screw."),
    IntentCase("E1-A03", "移動到組裝區",
               ["EXECUTE"], False, False, "A",
               "Single move command."),
    IntentCase("E1-A04", "打開夾爪",
               ["EXECUTE"], False, False, "A",
               "Gripper-open: single actuator command."),
    IntentCase("E1-A05", "關閉夾爪",
               ["EXECUTE"], False, False, "A",
               "Gripper-close: single actuator command."),
    IntentCase("E1-A06", "把圓柱體放到右邊",
               ["EXECUTE"], False, False, "A",
               "Place action with explicit target side."),
    IntentCase("E1-A07", "把螺絲交給我",
               ["EXECUTE"], False, False, "A",
               "Handover action to human."),
    IntentCase("E1-A08", "搬運零件到工作台",
               ["EXECUTE"], False, False, "A",
               "Move/place compound motion, single intent."),
    IntentCase("E1-A09", "移動到待機位置",
               ["EXECUTE"], False, False, "A",
               "Return-to-home move command."),
    IntentCase("E1-A10", "抓取工作台上最右邊的物品",
               ["EXECUTE"], False, False, "A",
               "Pickup with spatial qualifier, still single EXECUTE."),
    IntentCase("E1-A11", "按照第三步驟執行",
               ["EXECUTE"], False, False, "A",
               "SOP step reference → trigger execution."),
    IntentCase("E1-A12", "開始夾取動作",
               ["EXECUTE"], False, False, "A",
               "Generic grasp initiation."),
    IntentCase("E1-A13", "拿取 hex_wrench",
               ["EXECUTE"], False, False, "A",
               "Pickup with English object name still EXECUTE."),
    IntentCase("E1-A14", "放下手上的東西",
               ["EXECUTE"], False, False, "A",
               "Release / place-down command."),
    IntentCase("E1-A15", "移動到裝配位置",
               ["EXECUTE"], False, False, "A",
               "Move to assembly position."),
    IntentCase("E1-A16", "依照SOP執行第二步",
               ["EXECUTE"], False, False, "A",
               "SOP-driven execution, step 2."),
    IntentCase("E1-A17", "拿起支架",
               ["EXECUTE"], False, False, "A",
               "Pickup bracket."),
    IntentCase("E1-A18", "把六角扳手交還給我",
               ["EXECUTE"], False, False, "A",
               "Handover hex_wrench back to operator."),
    IntentCase("E1-A19", "去拿最左邊的圓柱",
               ["EXECUTE"], False, False, "A",
               "Pickup leftmost cylinder, spatial qualifier."),
    IntentCase("E1-A20", "抓取大螺絲",
               ["EXECUTE"], False, False, "A",
               "Pickup large screw."),
    IntentCase("E1-A21", "把物件移動到左側",
               ["EXECUTE"], False, False, "A",
               "Move/place to left side."),
    IntentCase("E1-A22", "回到原點",
               ["EXECUTE"], False, False, "A",
               "Return to home position."),
    IntentCase("E1-A23", "拿取傳送帶上的零件",
               ["EXECUTE"], False, False, "A",
               "Pickup from conveyor."),
    IntentCase("E1-A24", "將圓柱放到組裝台上",
               ["EXECUTE"], False, False, "A",
               "Place cylinder on assembly table."),
    IntentCase("E1-A25", "啟動夾取程序",
               ["EXECUTE"], False, False, "A",
               "Initiate grasp routine."),
    IntentCase("E1-A26", "拿著螺絲走到我面前",
               ["EXECUTE"], False, False, "A",
               "Pickup + move to operator (chained motion = one EXECUTE intent)."),
    IntentCase("E1-A27", "移到左邊的工作區域",
               ["EXECUTE"], False, False, "A",
               "Move to left work zone."),
    IntentCase("E1-A28", "拿起所有螺絲",
               ["EXECUTE"], False, False, "A",
               "Multi-object pickup but single intent."),
    IntentCase("E1-A29", "繼續執行上次的動作",
               ["EXECUTE"], False, False, "A",
               "Repeat last action — context-dependent but intent is EXECUTE."),
    IntentCase("E1-A30", "開始夾緊",
               ["EXECUTE"], False, False, "A",
               "Gripper tighten / close command."),

    # ── Group B: Pure ANSWER (20) ────────────────────────────────────────────
    IntentCase("E1-B01", "下一步是什麼？",
               ["ANSWER"], False, False, "B",
               "Question about next SOP step → information only."),
    IntentCase("E1-B02", "目前SOP進度在哪裡？",
               ["ANSWER"], False, False, "B",
               "Status query about SOP progress."),
    IntentCase("E1-B03", "這個零件叫什麼名字？",
               ["ANSWER"], False, False, "B",
               "Object identification question."),
    IntentCase("E1-B04", "組裝到第幾步了？",
               ["ANSWER"], False, False, "B",
               "Step-count query."),
    IntentCase("E1-B05", "怎麼判斷圓柱有沒有放好？",
               ["ANSWER"], False, False, "B",
               "Verification criterion question → no physical action."),
    IntentCase("E1-B06", "告訴我夾取螺絲的正確步驟",
               ["ANSWER"], False, False, "B",
               "Procedural explanation request."),
    IntentCase("E1-B07", "上一步做了什麼？",
               ["ANSWER"], False, False, "B",
               "History / log query."),
    IntentCase("E1-B08", "現在機器人在哪個位置？",
               ["ANSWER"], False, False, "B",
               "Robot position status query."),
    IntentCase("E1-B09", "任務完成了嗎？",
               ["ANSWER"], False, False, "B",
               "Completion check question."),
    IntentCase("E1-B10", "這個任務的SOP總共有幾步？",
               ["ANSWER"], False, False, "B",
               "SOP length query."),
    IntentCase("E1-B11", "右邊有什麼物件？",
               ["ANSWER"], False, False, "B",
               "Workspace observation query."),
    IntentCase("E1-B12", "夾爪目前是開的還是關的？",
               ["ANSWER"], False, False, "B",
               "Gripper state query."),
    IntentCase("E1-B13", "為什麼剛才要移動到那個位置？",
               ["ANSWER"], False, False, "B",
               "Explanation of past action → no new physical action."),
    IntentCase("E1-B14", "螺絲需要放到哪裡？",
               ["ANSWER"], False, False, "B",
               "Placement target query."),
    IntentCase("E1-B15", "這個動作完成的標準是什麼？",
               ["ANSWER"], False, False, "B",
               "Acceptance-criterion question."),
    IntentCase("E1-B16", "第五步需要用什麼工具？",
               ["ANSWER"], False, False, "B",
               "Tool-requirement query for a specific step."),
    IntentCase("E1-B17", "解釋一下整個流程",
               ["ANSWER"], False, False, "B",
               "Full procedure explanation request."),
    IntentCase("E1-B18", "現在做到第幾步了？",
               ["ANSWER"], False, False, "B",
               "Current step index query."),
    IntentCase("E1-B19", "上面的零件要怎麼安裝？",
               ["ANSWER"], False, False, "B",
               "Installation instruction query → ANSWER, not EXECUTE."),
    IntentCase("E1-B20", "操作完成後要確認什麼？",
               ["ANSWER"], False, False, "B",
               "Post-operation checklist query."),

    # ── Group C: CLARIFY needed (10) ─────────────────────────────────────────
    IntentCase("E1-C01", "繼續",
               ["CLARIFY"], False, False, "C",
               "Continue what? No context → must ask."),
    IntentCase("E1-C02", "執行",
               ["CLARIFY"], False, False, "C",
               "Execute what? Object and action both missing."),
    IntentCase("E1-C03", "幫我做一下",
               ["CLARIFY"], False, False, "C",
               "Vague reference 'do it' — unclear action and object."),
    IntentCase("E1-C04", "重複剛才的動作",
               ["CLARIFY"], False, False, "C",
               "Needs context of 'last action' — could be risky if misidentified."),
    IntentCase("E1-C05", "好了可以了",
               ["CLARIFY"], False, False, "C",
               "Ambiguous: task complete? Move to next? Stop?"),
    IntentCase("E1-C06", "把它放過去",
               ["CLARIFY"], False, False, "C",
               "'It' and 'there' both unresolved."),
    IntentCase("E1-C07", "用同樣的方式",
               ["CLARIFY"], False, False, "C",
               "Same as what? No referent."),
    IntentCase("E1-C08", "那個東西",
               ["CLARIFY"], False, False, "C",
               "Object noun only — no verb/action."),
    IntentCase("E1-C09", "做完之後呢",
               ["CLARIFY"], False, False, "C",
               "Post-condition query without specifying what was done."),
    IntentCase("E1-C10", "照剛才做",
               ["CLARIFY"], False, False, "C",
               "Anaphora 'as before' — must resolve referent."),

    # ── Group D: EXECUTE + ANSWER (20) — execute first, then information ─────
    # catastrophic_if_wrong=True because wrong routing means robot acts when
    # ANSWER portion expects only verbal response, or skips execution.
    IntentCase("E1-D01", "幫我拿圓柱體，然後告訴我下一步是什麼",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup first, then step-query. If ANSWER→EXECUTE: robot halts physical+asks back."),
    IntentCase("E1-D02", "先移動到組裝區，再解釋一下第三步",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Move then explain step 3. Both needed; wrong routing kills one branch."),
    IntentCase("E1-D03", "拿起螺絲，然後告訴我接下來要怎麼做",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup + next-step query."),
    IntentCase("E1-D04", "把夾爪打開，告訴我這樣是否正確",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Gripper open + verification question."),
    IntentCase("E1-D05", "夾取支架，同時告訴我SOP進度",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup + status query. 'Simultaneously' → multi-intent."),
    IntentCase("E1-D06", "把圓柱放到組裝台，解釋一下這個步驟",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Place + explanation."),
    IntentCase("E1-D07", "去拿六角扳手，再說一下接下來要做什麼",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup hex_wrench + future-step query."),
    IntentCase("E1-D08", "移動到待機位，然後確認一下任務完成了嗎",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Move to standby + completion check."),
    IntentCase("E1-D09", "拿大螺絲給我，告訴我下一個零件是什麼",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Handover screw + next-part query."),
    IntentCase("E1-D10", "把物件搬到工作台，告訴我預計完成時間",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Move object + completion-time query."),
    IntentCase("E1-D11", "先夾起來再解釋如何安裝",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Grasp + installation explanation."),
    IntentCase("E1-D12", "按第一步驟執行，並告知完成條件",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "SOP step 1 + acceptance criterion."),
    IntentCase("E1-D13", "將螺絲放好，然後說明下一個步驟",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Place screw + next-step explanation."),
    IntentCase("E1-D14", "抓取最左邊的零件，確認工作台狀態",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup + workspace-state confirmation."),
    IntentCase("E1-D15", "夾取圓柱體後說明裝配要求",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup + assembly requirement explanation."),
    IntentCase("E1-D16", "拿起支架並告訴我安裝方法",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup bracket + installation method."),
    IntentCase("E1-D17", "先搬到右側再說現在的狀態",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Move right + status report."),
    IntentCase("E1-D18", "執行夾取程序，之後解釋SOP第四步",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Grasp routine + step 4 explanation."),
    IntentCase("E1-D19", "去拿螺絲，完成後告訴我下一步",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup screw + next-step query after completion."),
    IntentCase("E1-D20", "抓起工具，說一下整個流程還剩幾步",
               ["EXECUTE", "ANSWER"], True, True, "D",
               "Pickup tool + remaining-steps count query."),

    # ── Group E: ANSWER + EXECUTE (10) — question first, then action ─────────
    IntentCase("E1-E01", "告訴我SOP第一步，然後開始執行",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Explain step 1, then execute it. Both intents required."),
    IntentCase("E1-E02", "解釋一下夾取螺絲的方法，然後幫我夾",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Explain grasp method then actually grasp."),
    IntentCase("E1-E03", "說明現在的步驟，再繼續動作",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Announce current step then resume motion."),
    IntentCase("E1-E04", "先告訴我目前進度，然後繼續下一步",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Progress report then proceed to next step."),
    IntentCase("E1-E05", "描述一下接下來要怎麼做，然後去做",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Describe next action then execute it."),
    IntentCase("E1-E06", "說一下圓柱要放哪裡，然後放過去",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Announce target location then place cylinder."),
    IntentCase("E1-E07", "告訴我夾爪狀態，如果開著幫我關上",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Conditional: state query + conditional gripper-close action."),
    IntentCase("E1-E08", "確認一下物件位置，然後去拿",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Confirm object location then pickup."),
    IntentCase("E1-E09", "解釋第二步驟，再按照第二步動作",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Explain step 2 then execute step 2."),
    IntentCase("E1-E10", "告訴我現在安全嗎，可以的話繼續",
               ["ANSWER", "EXECUTE"], True, True, "E",
               "Safety check + conditional continuation."),

    # ── Group F: EXECUTE + EXECUTE (10) — two sequential physical actions ─────
    IntentCase("E1-F01", "拿螺絲然後移動到組裝區",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Pickup then move — two physical actions in sequence."),
    IntentCase("E1-F02", "先打開夾爪再把物品放下",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Gripper open then release/place."),
    IntentCase("E1-F03", "移到左邊然後拿起圓柱",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Move left then pickup cylinder."),
    IntentCase("E1-F04", "夾起支架然後移到工作台",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Pickup bracket then move to worktable."),
    IntentCase("E1-F05", "先關夾爪再回到原點",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Gripper close then return home."),
    IntentCase("E1-F06", "把螺絲拿起來再放到右側托盤",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Pickup screw then place on right tray."),
    IntentCase("E1-F07", "移到組裝位置然後開始夾取",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Move then grasp — sequential execution."),
    IntentCase("E1-F08", "先拿六角扳手然後遞給操作員",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Pickup hex_wrench then handover."),
    IntentCase("E1-F09", "拿圓柱體然後直接交給我",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "Pickup cylinder then handover — two-step motion."),
    IntentCase("E1-F10", "依序執行第一步和第二步",
               ["EXECUTE", "EXECUTE"], True, False, "F",
               "SOP steps 1 and 2 chained — two EXECUTE intents."),
]

# ─────────────────────────────────────────────────────────────────────────────
# EXP2 – Execution Feasibility (60 cases)
# ─────────────────────────────────────────────────────────────────────────────
# Subsections:
#   R – REUSE (15):  object present + steps match (diff = 0)
#   J – REJECT/obj (15): required object NOT in workspace
#   S – was “step reject” (15): objects present; Layer 2 gate no longer uses step → REUSE
#   P – was “step ADAPT” (15): objects present; same → REUSE

EXP2_CASES: List[FeasibilityCase] = [

    # ── Subsection R: REUSE (15) ─────────────────────────────────────────────
    FeasibilityCase("E2-R01", "幫我拿圓柱體",
                    "pickup", "cylinder", ["cylinder", "screw"], 0, 0, "REUSE",
                    "cylinder in ws, step 0 == retrieved step 0 → REUSE."),
    FeasibilityCase("E2-R02", "移動到組裝區",
                    "move", None, ["cylinder"], 1, 1, "REUSE",
                    "No object required, step 1 == 1 → REUSE."),
    FeasibilityCase("E2-R03", "把螺絲交給我",
                    "handover", "screw", ["screw", "cylinder"], 2, 2, "REUSE",
                    "screw in ws, step 2 == 2 → REUSE."),
    FeasibilityCase("E2-R04", "打開夾爪",
                    "gripper_open", None, [], 0, 0, "REUSE",
                    "No object needed, step 0 == 0 → REUSE."),
    FeasibilityCase("E2-R05", "拿起支架",
                    "pickup", "bracket", ["bracket", "screw"], 1, 1, "REUSE",
                    "bracket in ws, step 1 == 1 → REUSE."),
    FeasibilityCase("E2-R06", "把圓柱放到組裝台",
                    "place", "cylinder", ["cylinder"], 3, 3, "REUSE",
                    "cylinder in ws, step 3 == 3 → REUSE."),
    FeasibilityCase("E2-R07", "移動到待機位",
                    "move", None, [], 5, 5, "REUSE",
                    "No object needed, step 5 == 5 → REUSE."),
    FeasibilityCase("E2-R08", "拿大螺絲",
                    "pickup", "screw", ["screw", "cylinder", "bracket"], 0, 0, "REUSE",
                    "screw in ws, step 0 == 0 → REUSE."),
    FeasibilityCase("E2-R09", "關閉夾爪",
                    "gripper_close", None, [], 2, 2, "REUSE",
                    "No object needed, step 2 == 2 → REUSE."),
    FeasibilityCase("E2-R10", "把六角扳手拿給我",
                    "handover", "hex_wrench", ["hex_wrench", "screw"], 4, 4, "REUSE",
                    "hex_wrench in ws, step 4 == 4 → REUSE."),
    FeasibilityCase("E2-R11", "拿起圓柱體（步驟1）",
                    "pickup", "cylinder", ["cylinder", "bracket"], 0, 0, "REUSE",
                    "cylinder in ws, step 0 == 0 → REUSE."),
    FeasibilityCase("E2-R12", "移動到右側工作區",
                    "move", None, [], 2, 2, "REUSE",
                    "No object needed, step 2 == 2 → REUSE."),
    FeasibilityCase("E2-R13", "把螺絲放到右側",
                    "place", "screw", ["screw"], 3, 3, "REUSE",
                    "screw in ws, step 3 == 3 → REUSE."),
    FeasibilityCase("E2-R14", "遞圓柱給操作員",
                    "handover", "cylinder", ["cylinder", "screw"], 2, 2, "REUSE",
                    "cylinder in ws, step 2 == 2 → REUSE."),
    FeasibilityCase("E2-R15", "按步驟拿取支架",
                    "pickup", "bracket", ["bracket", "hex_wrench"], 1, 1, "REUSE",
                    "bracket in ws, step 1 == 1 → REUSE."),

    # ── Subsection J: REJECT – required object missing (15) ──────────────────
    FeasibilityCase("E2-J01", "拿圓柱體",
                    "pickup", "cylinder", ["screw", "bracket"], 0, 0, "REJECT",
                    "cylinder NOT in ws → REJECT (object gate fails)."),
    FeasibilityCase("E2-J02", "把螺絲遞給我",
                    "handover", "screw", ["cylinder", "bracket"], 2, 2, "REJECT",
                    "screw NOT in ws → REJECT."),
    FeasibilityCase("E2-J03", "拿起支架",
                    "pickup", "bracket", ["screw", "cylinder"], 1, 1, "REJECT",
                    "bracket NOT in ws → REJECT."),
    FeasibilityCase("E2-J04", "夾取六角扳手",
                    "pickup", "hex_wrench", ["cylinder", "screw"], 4, 4, "REJECT",
                    "hex_wrench NOT in ws → REJECT."),
    FeasibilityCase("E2-J05", "放置圓柱",
                    "place", "cylinder", ["screw"], 3, 3, "REJECT",
                    "cylinder NOT in ws → REJECT."),
    FeasibilityCase("E2-J06", "拿取大螺絲",
                    "pickup", "screw", ["cylinder", "bracket"], 0, 0, "REJECT",
                    "screw NOT in ws → REJECT."),
    FeasibilityCase("E2-J07", "把支架交給操作員",
                    "handover", "bracket", ["screw", "cylinder", "hex_wrench"], 3, 3, "REJECT",
                    "bracket NOT in ws → REJECT."),
    FeasibilityCase("E2-J08", "把hex_wrench遞給我",
                    "handover", "hex_wrench", ["screw", "bracket"], 2, 2, "REJECT",
                    "hex_wrench NOT in ws → REJECT."),
    FeasibilityCase("E2-J09", "夾取圓柱",
                    "pickup", "cylinder", [], 0, 0, "REJECT",
                    "workspace empty → REJECT."),
    FeasibilityCase("E2-J10", "拿螺絲（步驟2）",
                    "pickup", "screw", ["cylinder", "bracket"], 2, 2, "REJECT",
                    "screw NOT in ws → REJECT."),
    FeasibilityCase("E2-J11", "移動並放下圓柱",
                    "place", "cylinder", ["screw"], 3, 3, "REJECT",
                    "cylinder NOT in ws → REJECT."),
    FeasibilityCase("E2-J12", "取bracket",
                    "pickup", "bracket", ["cylinder", "hex_wrench"], 0, 0, "REJECT",
                    "bracket NOT in ws → REJECT."),
    FeasibilityCase("E2-J13", "拿起所有螺絲",
                    "pickup", "screw", ["cylinder"], 1, 1, "REJECT",
                    "screw NOT in ws → REJECT."),
    FeasibilityCase("E2-J14", "遞支架",
                    "handover", "bracket", ["screw", "hex_wrench"], 4, 4, "REJECT",
                    "bracket NOT in ws → REJECT."),
    FeasibilityCase("E2-J15", "把圓柱放到組裝台",
                    "place", "cylinder", ["bracket", "screw"], 5, 5, "REJECT",
                    "cylinder NOT in ws → REJECT."),

    # ── Subsection S: large step delta but objects present → REUSE (no step gate)
    FeasibilityCase("E2-S01", "拿圓柱體",
                    "pickup", "cylinder", ["cylinder", "screw"], 3, 0, "REUSE",
                    "cylinder in ws; step indices differ but Layer 2 does not gate on step."),
    FeasibilityCase("E2-S02", "移動到組裝區",
                    "move", None, [], 0, 4, "REUSE",
                    "No object required; step delta ignored → REUSE."),
    FeasibilityCase("E2-S03", "螺絲交給我",
                    "handover", "screw", ["screw"], 5, 1, "REUSE",
                    "screw in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S04", "拿支架",
                    "pickup", "bracket", ["bracket"], 4, 0, "REUSE",
                    "bracket in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S05", "放圓柱",
                    "place", "cylinder", ["cylinder"], 0, 5, "REUSE",
                    "cylinder in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S06", "拿螺絲",
                    "pickup", "screw", ["screw"], 2, 6, "REUSE",
                    "screw in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S07", "移動",
                    "move", None, [], 1, 5, "REUSE",
                    "No object; step delta ignored → REUSE."),
    FeasibilityCase("E2-S08", "夾取hex_wrench",
                    "pickup", "hex_wrench", ["hex_wrench"], 3, 0, "REUSE",
                    "hex_wrench in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S09", "拿圓柱（第7步）",
                    "pickup", "cylinder", ["cylinder"], 6, 2, "REUSE",
                    "cylinder in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S10", "放螺絲",
                    "place", "screw", ["screw"], 0, 4, "REUSE",
                    "screw in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S11", "拿bracket",
                    "pickup", "bracket", ["bracket"], 5, 1, "REUSE",
                    "bracket in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S12", "遞圓柱",
                    "handover", "cylinder", ["cylinder"], 2, 6, "REUSE",
                    "cylinder in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S13", "移到待機位",
                    "move", None, [], 4, 0, "REUSE",
                    "No object; step delta ignored → REUSE."),
    FeasibilityCase("E2-S14", "夾取螺絲（第8步）",
                    "pickup", "screw", ["screw"], 3, 7, "REUSE",
                    "screw in ws; step delta ignored → REUSE."),
    FeasibilityCase("E2-S15", "執行放置動作",
                    "place", "cylinder", ["cylinder"], 1, 5, "REUSE",
                    "cylinder in ws; step delta ignored → REUSE."),

    # ── Subsection P: step diff == 1 historically; same treatment → REUSE ───
    FeasibilityCase("E2-P01", "拿圓柱體",
                    "pickup", "cylinder", ["cylinder", "screw"], 0, 1, "REUSE",
                    "cylinder in ws; step delta not used for gate → REUSE."),
    FeasibilityCase("E2-P02", "把螺絲拿給我",
                    "handover", "screw", ["screw"], 2, 3, "REUSE",
                    "screw in ws → REUSE."),
    FeasibilityCase("E2-P03", "放圓柱到左邊",
                    "place", "cylinder", ["cylinder"], 3, 4, "REUSE",
                    "cylinder in ws → REUSE."),
    FeasibilityCase("E2-P04", "夾取大螺絲",
                    "pickup", "screw", ["screw"], 1, 2, "REUSE",
                    "screw in ws → REUSE."),
    FeasibilityCase("E2-P05", "拿支架",
                    "pickup", "bracket", ["bracket", "cylinder"], 2, 3, "REUSE",
                    "bracket in ws → REUSE."),
    FeasibilityCase("E2-P06", "移動到工作台",
                    "move", None, [], 4, 3, "REUSE",
                    "No object → REUSE."),
    FeasibilityCase("E2-P07", "把hex_wrench遞給操作員",
                    "handover", "hex_wrench", ["hex_wrench"], 3, 2, "REUSE",
                    "hex_wrench in ws → REUSE."),
    FeasibilityCase("E2-P08", "放置圓柱",
                    "place", "cylinder", ["cylinder"], 5, 4, "REUSE",
                    "cylinder in ws → REUSE."),
    FeasibilityCase("E2-P09", "夾取螺絲（跟上次類似）",
                    "pickup", "screw", ["screw"], 0, 1, "REUSE",
                    "screw in ws → REUSE."),
    FeasibilityCase("E2-P10", "拿bracket然後放到右邊",
                    "pickup", "bracket", ["bracket", "screw"], 2, 3, "REUSE",
                    "bracket in ws → REUSE."),
    FeasibilityCase("E2-P11", "把圓柱放到組裝台",
                    "place", "cylinder", ["cylinder"], 4, 3, "REUSE",
                    "cylinder in ws → REUSE."),
    FeasibilityCase("E2-P12", "拿起螺絲",
                    "pickup", "screw", ["screw", "cylinder"], 3, 4, "REUSE",
                    "screw in ws → REUSE."),
    FeasibilityCase("E2-P13", "移到右側",
                    "move", None, [], 1, 0, "REUSE",
                    "No object → REUSE."),
    FeasibilityCase("E2-P14", "夾取圓柱",
                    "pickup", "cylinder", ["cylinder"], 2, 1, "REUSE",
                    "cylinder in ws → REUSE."),
    FeasibilityCase("E2-P15", "繼續拿螺絲",
                    "pickup", "screw", ["screw", "bracket"], 5, 4, "REUSE",
                    "screw in ws → REUSE."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity checks (run on import)
# ─────────────────────────────────────────────────────────────────────────────

def _check_counts():
    assert len(EXP1_CASES) == 100, f"EXP1 expects 100 cases, got {len(EXP1_CASES)}"
    assert len(EXP2_CASES) == 60,  f"EXP2 expects 60 cases, got {len(EXP2_CASES)}"

_check_counts()

if __name__ == "__main__":
    print(f"EXP1 cases : {len(EXP1_CASES)}")
    print(f"EXP2 cases : {len(EXP2_CASES)}")
    print(f"Total      : {len(EXP1_CASES) + len(EXP2_CASES)}")
    multi = [c for c in EXP1_CASES if c.is_multi]
    cat = {c.id.split('-')[1][0] for c in EXP1_CASES if c.is_multi}
    print(f"EXP1 multi-intent: {len(multi)} cases (groups {sorted(cat)})")
