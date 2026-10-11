"""
EXP1 — Intent Understanding (Layer 1)
==================================

Research question:
    Does intent-level decomposition (span-level heuristic) outperform
    utterance-level routing on natural language robot commands?

Dataset:
    100 cases from evaluation/datasets/master_dataset.py (EXP1_CASES)
    - Group A (30): pure EXECUTE
    - Group B (20): pure ANSWER
    - Group C (10): CLARIFY needed (vague/incomplete)
    - Group D (20): EXECUTE + ANSWER  ← most safety-critical
    - Group E (10): ANSWER + EXECUTE
    - Group F (10): EXECUTE + EXECUTE

    All cases are cleanly labelled with unambiguous GT:
    - EXECUTE commands have clear action verbs + known objects
    - ANSWER commands have clear question markers
    - CLARIFY commands have explicit vague references or missing slots
    No case where "幫我拿X" is labeled ASK (like nl_command_dataset.py had).

整句關鍵字 (`_utterance_level_keyword`):
    _is_strong_question(whole_command) → SOP answer, END
    Otherwise: whole command → plan retrieval → decision

分段關鍵字 (`_span_level_keyword`, standalone re-implementation of the segmentation idea):
    _parse_intents() (defined below) → per-span classification
    Per-intent routing: EXECUTE→plan, ANSWER→SOP, CLARIFY→none

Usage:
    python -m evaluation.exp1_intent_routing
"""
from __future__ import annotations

import sys, os, re
from typing import Dict, List, Tuple

from evaluation.datasets.master_dataset import EXP1_CASES, IntentCase


# ─────────────────────────────────────────────────────────────────────────────
# Standalone heuristic classifier evaluated in EXP1 (a richer rule set than the production IntentMixin; see docs/evaluation.md)
# ─────────────────────────────────────────────────────────────────────────────

_Q_STRONG = ('嗎', '呢', '怎麼', '如何', '告訴我', '說明',
             '幾步', '幾個', '多少', '是什麼', '有哪些', '哪些', '什麼')
_Q_MARKERS = ('哪些', '什麼', '哪裡', '在哪', '幾個', '幾步', '多少',
              '嗎', '呢', '哪邊', '怎麼', '如何', '告訴我', '說明',
              '是什麼', '哪個', '需要')
_ACTION_PREFIX = ('幫我', '請', '把', '將', '拿', '放', '移', '鎖', '取',
                  '夾', '遞', '轉', '鬆', '開始', '停止', '繼續')
_EXECUTE_VERBS = ('幫我', '把', '將', '拿', '放', '夾', '移', '鎖', '取',
                  '插', '對齊', '遞', '固定', '套', '穩', '推',
                  '抓', '搬', '按', '執行', '依序', '開始')
# 「東西」在「手上的東西」語境中是確定指物，不是真正的模糊引用
_VAGUE_OBJ = ('那個', '這個', '它', '那些', '這些', '一個零件')
_VAGUE_LOC = ('那裡', '那邊', '這邊', '這裡', '另一邊')

# ANSWER start verbs — spans starting with these are explanation/confirmation requests
_ANSWER_START = ('解釋', '確認', '告知', '說明', '說一下', '說', '講', '報告',
                 '告訴', '查一下', '查看', '檢查')

# Exact single-token CLARIFY commands — ambiguous without context
_CLARIFY_EXACT = frozenset(['繼續', '執行', '好了', '可以了', '好了可以了', '開始', '繼續執行'])

# Anaphoric references to previous context → CLARIFY (robot can't resolve without history)
_ANAPHORIC_KW = ('重複', '剛才', '照剛才', '用同樣', '同樣的方式', '上次', '那樣')

# Primary connectives (with optional leading comma)
_CONNECTIVE_PAT = re.compile(
    r'[，,]?\s*(?:'
    r'然後|接著|再來|之後|並且|順便|同時'      # standard
    r'|並(?!且)'                               # 並 (not 並且, already above)
    r'|完成後|做完後|執行後|之後再'            # temporal completion
    r'|再(?!次|三)'                            # 再 (allow 再說, exclude 再次/再三)
    r')'
)
# Comma immediately before an ANSWER-start verb → split here too
_COMMA_BEFORE_ANSWER = re.compile(
    r'[，,]\s*(?=' + '|'.join(_ANSWER_START) + r'|告訴我)'
)
# Temporal "後" directly followed by an ANSWER-start verb → split after 後
_TEMPORAL_AFTER = re.compile(
    r'(?<=\w\w)後(?=' + '|'.join(_ANSWER_START) + r'|告訴我)'
)
_Q_BOUNDARY_PAT = re.compile(r'？\s*(?=[^\？]{2,})')


def _is_strong_question(text: str) -> bool:
    t = text.strip()
    if not t: return False
    if any(t.startswith(p) for p in _ACTION_PREFIX): return False
    if t.endswith('？') or t.endswith('?'): return True
    return any(m in t for m in _Q_STRONG)


def _is_question(text: str) -> bool:
    return any(m in text for m in _Q_MARKERS) or text.rstrip().endswith('？')


def _segment(command: str) -> List[str]:
    # Pass 1: primary connectives
    parts = _CONNECTIVE_PAT.split(command)
    all_spans: List[str] = []
    for p in parts:
        p = p.strip()
        if not p: continue
        # Pass 2a: comma before ANSWER-start verb
        sub = _COMMA_BEFORE_ANSWER.split(p)
        merged: List[str] = []
        for s in sub:
            # Pass 2b: temporal "後" before ANSWER-start verb
            sub2 = _TEMPORAL_AFTER.split(s)
            merged.extend(sub2)
        # Pass 3: question boundary ？
        for s in merged:
            s = s.strip()
            if not s: continue
            for ss in _Q_BOUNDARY_PAT.split(s):
                ss = ss.strip()
                if ss: all_spans.append(ss)
    return all_spans or [command]


def _classify_span(span: str) -> str:
    s = span.strip()
    # ① Single-token CLARIFY (e.g. "繼續", "執行", "好了")
    if s in _CLARIFY_EXACT:
        return 'CLARIFY'
    # ② Anaphoric reference to previous context → CLARIFY
    if any(kw in s for kw in _ANAPHORIC_KW):
        return 'CLARIFY'
    # ③ Starts with ANSWER-type verb (explain / confirm / report)
    if any(s.startswith(v) for v in _ANSWER_START):
        return 'ANSWER'
    # ④ Ends with ？ + question marker → ANSWER
    if (s.endswith('？') or s.endswith('?')) and any(m in s for m in _Q_STRONG):
        return 'ANSWER'
    # ⑤ Starts with EXECUTE verb
    if any(s.startswith(v) for v in _EXECUTE_VERBS):
        if any(ref in s for ref in _VAGUE_OBJ): return 'CLARIFY'
        if any(ref in s for ref in _VAGUE_LOC):  return 'CLARIFY'
        if _is_question(s): return 'ANSWER'
        return 'EXECUTE'
    # ⑥ Question markers anywhere → ANSWER
    if _is_question(s): return 'ANSWER'
    # ⑦ Vague references → CLARIFY
    if any(ref in s for ref in _VAGUE_OBJ + _VAGUE_LOC): return 'CLARIFY'
    # ⑧ Default
    return 'EXECUTE'


def _parse_intents(command: str) -> List[str]:
    return [_classify_span(seg) for seg in _segment(command)]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline simulations
# ─────────────────────────────────────────────────────────────────────────────

def _utterance_level_keyword(case: IntentCase) -> List[str]:
    """整句關鍵字：僅輸出單一標籤，無法分段處理多意圖。"""
    utt = case.command
    if _is_strong_question(utt):
        return ['ANSWER']
    if _is_question(utt):
        return ['ANSWER']
    return ['EXECUTE']


def _span_level_keyword(case: IntentCase) -> List[str]:
    """分段關鍵字：切 span 後逐段分類。"""
    return _parse_intents(case.command)


def _routing_ok(predicted: List[str], gt: List[str]) -> bool:
    if len(predicted) != len(gt): return False
    return all(p == g for p, g in zip(predicted, gt))


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cases = EXP1_CASES
    groups = sorted(set(c.group for c in cases))

    total = len(cases)
    utt_correct = span_correct = 0
    cat_stats: Dict[str, Dict] = {g: {"utt": 0, "span": 0, "n": 0} for g in groups}

    span_failures: List[Tuple[str, IntentCase, List[str], List[str]]] = []

    for case in cases:
        grp = case.group
        cat_stats[grp]["n"] += 1

        utt_pred = _utterance_level_keyword(case)
        span_pred = _span_level_keyword(case)

        if _routing_ok(utt_pred, case.intents):
            utt_correct += 1
            cat_stats[grp]["utt"] += 1
        if _routing_ok(span_pred, case.intents):
            span_correct += 1
            cat_stats[grp]["span"] += 1
        else:
            span_failures.append((grp, case, span_pred, case.intents))

    SEP = "─" * 65
    print(f"\n{'='*65}")
    print("  EXP1 — Intent Understanding (Layer 1)")
    print(f"  Dataset: master_dataset.py  |  {total} cases  |  Groups A–F")
    print(f"{'='*65}")

    print(f"\n[1] Overall routing accuracy（整句關鍵字 vs 分段關鍵字）")
    print(SEP)
    print(f"  整句關鍵字 : {utt_correct}/{total} = {utt_correct/total:.1%}")
    print(f"  分段關鍵字 : {span_correct}/{total} = {span_correct/total:.1%}")
    print(f"  Delta(Seg - Utt): +{(span_correct - utt_correct) / total:.1%}")

    print(f"\n[2] Per-group routing accuracy")
    print(SEP)
    grp_desc = {
        "A": "Pure EXECUTE   (30)",
        "B": "Pure ANSWER    (20)",
        "C": "CLARIFY needed (10)",
        "D": "EXECUTE+ANSWER (20)  ← safety-critical",
        "E": "ANSWER+EXECUTE (10)  ← safety-critical",
        "F": "EXECUTE+EXECUTE(10)",
    }
    print(f"  {'Grp':<4} {'Description':<35} {'N':>4}  {'整句':>6}  {'分段':>6}  {'Δ':>6}")
    for g in groups:
        s = cat_stats[g]
        n = s["n"]
        if n == 0: continue
        o = s["utt"] / n
        nw = s["span"] / n
        desc = grp_desc.get(g, g)
        print(f"  {g:<4} {desc:<35} {n:>4}  {o:>5.1%}  {nw:>5.1%}  {nw-o:>+5.1%}")

    print(f"\n[3] 分段關鍵字 — failure 範例（至多 20 筆）")
    print(SEP)
    for grp, case, pred, gt in span_failures[:20]:
        print(f"  [{case.id}] Grp={grp}  gt={gt}  got={pred}")
        print(f"          cmd: {case.command}")
    if len(span_failures) > 20:
        print(f"  ... ({len(span_failures)-20} more)")

    print(f"\n[4] Summary")
    print(SEP)
    print(f"  整句關鍵字 routing accuracy : {utt_correct/total:.1%}")
    print(f"  分段關鍵字 routing accuracy : {span_correct/total:.1%}")
    print()


if __name__ == "__main__":
    main()
