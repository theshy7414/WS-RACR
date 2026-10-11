"""
EXP3 — Execution Risk Estimation (Layer 3)
======================================

Research question:
    Does the new Layer 3 rule (sim + action_match + object_match) outperform
    the old sim-only threshold on real embedding similarities?

Dataset:
    80 command pairs in evaluation/datasets/exp3_pairs.py
    - Real sim values computed by paraphrase-multilingual-MiniLM-L12-v2
    - Ground truth annotated by thesis author (verified pair-by-pair)

    Thesis groups A–D (20 each, balanced): see exp3_ids_for_group() in the
    dataset module (no separate Group E; former E pairs removed in v3).

Why the old sim-only threshold fails:
    Real embedding distributions show:
      pickup  vs handover (same object): sim ≈ 0.94  → sim-only says REUSE (wrong)
      handover vs handover (diff object): sim ≈ 0.87 → sim-only says REUSE (wrong)
    The embedding model encodes object presence strongly, causing high sim
    for same-object different-action pairs.  A single sim threshold cannot
    distinguish these from genuinely reusable pairs.

New Layer 3 decision (3 rules):
    Rule 1: sim ≥ 0.85  AND  action_match  AND  object_match  → REUSE
    Rule 2: action_match  AND  sim ≥ 0.50                     → ADAPT
    Rule 3: compatible(cur, ret) AND sim ≥ 0.75               → ADAPT
    Rule 4: otherwise                                          → CLARIFY

    compatible pairs: pickup↔handover, place↔handover (both involve
    object positioning; LLM can bridge the gap)

Usage:
    python -m evaluation.exp3_execution_risk
"""
from __future__ import annotations

import sys
import os
from typing import Dict, List


from evaluation.common import production_risk_decision
from evaluation.datasets.exp3_pairs import PAIRS, Pair, exp3_ids_for_group


# ─────────────────────────────────────────────────────────────────────────────
# Decision policies
# ─────────────────────────────────────────────────────────────────────────────

def _sim_only(p: Pair) -> str:
    """Old approach: pure cosine similarity threshold."""
    if p.real_sim >= 0.75: return "REUSE"
    if p.real_sim >= 0.50: return "ADAPT"
    return "CLARIFY"


def _new_rule(p: Pair) -> str:
    """Layer 3 decision of the production policy (``ws_racr.decision.collaboration_policy``)."""
    return production_risk_decision(p.current_cmd, p.retrieved_cmd, p.ret_type, p.real_sim)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _eval(pairs: List[Pair], fn) -> Dict:
    preds = [fn(p) for p in pairs]
    gt    = [p.gt for p in pairs]
    n     = len(pairs)
    correct       = sum(p == g for p, g in zip(preds, gt))
    false_reuse   = sum(p == "REUSE"   and g in ("ADAPT", "CLARIFY")
                        for p, g in zip(preds, gt))
    false_clarify = sum(p == "CLARIFY" and g == "REUSE"
                        for p, g in zip(preds, gt))
    adapt_gt = [(p, g) for p, g in zip(preds, gt) if g == "ADAPT"]
    adapt_acc = sum(p == "ADAPT" for p, _ in adapt_gt) / len(adapt_gt) if adapt_gt else 0.0
    return {
        "n": n, "correct": correct,
        "accuracy":          correct / n,
        "false_reuse":       false_reuse,
        "false_reuse_pct":   false_reuse / n,
        "false_clarify":     false_clarify,
        "false_clarify_pct": false_clarify / n,
        "adapt_accuracy":    adapt_acc,
        "predictions":       preds,
    }

def _pct(v: float) -> str: return f"{v*100:5.1f}%"
def _bar(v: float, w: int = 20) -> str:
    f = round(v * w); return "█" * f + "░" * (w - f)


def main():
    pairs = PAIRS
    n = len(pairs)

    print("=" * 70)
    print(f"EXP3 — Execution Risk Estimation (Layer 3)  |  {n} real-embedding cases")
    print("=" * 70)
    print("sim computed by: paraphrase-multilingual-MiniLM-L12-v2")
    print("GT annotated by: thesis author (pair-by-pair verification)")
    print()

    gt_dist = {k: sum(1 for p in pairs if p.gt == k) for k in ("REUSE","ADAPT","CLARIFY")}
    for k, v in gt_dist.items():
        print(f"  GT {k:>8}: {v:2d} cases")
    print()

    methods = {
        "Always-REUSE":   lambda p: "REUSE",
        "Always-CLARIFY": lambda p: "CLARIFY",
        "sim-only (old)": _sim_only,
        "New rules (ours)": _new_rule,
    }
    results = {name: _eval(pairs, fn) for name, fn in methods.items()}

    # Summary table
    print(f"{'Method':<20} │ {'Acc':>6} │ {'FalseReuse':>10} │ "
          f"{'FalseClarify':>12} │ {'ADAPT Acc':>9}")
    print("─"*20 + "─┼─" + "─"*6 + "─┼─" + "─"*10 + "─┼─" + "─"*12 + "─┼─" + "─"*9)
    for name, r in results.items():
        marker = "  ◄ ours" if name == "New rules (ours)" else ""
        print(f"{name:<20} │ {_pct(r['accuracy']):>6} │"
              f" {_pct(r['false_reuse_pct']):>10} │"
              f" {_pct(r['false_clarify_pct']):>12} │"
              f" {_pct(r['adapt_accuracy']):>9}{marker}")
    print()

    # Per-group breakdown
    print("─" * 70)
    print("Per-group accuracy (sim-only vs new rules)")
    print("─" * 70)
    group_defs = [
        ("A", "same type + same obj → REUSE (P01–P10 + P37–P46)"),
        ("B", "same type + diff obj → ADAPT (P11–P20 + P47–P56)"),
        ("C", "diff type / danger zone (P21–P28 + P57–P64 + P81–P84)"),
        ("D", "low sim / question / vague → CLARIFY (P29–P36 + P65–P74 + P85–P86)"),
    ]
    by_id = {p.id: p for p in pairs}
    for grp, desc in group_defs:
        want = exp3_ids_for_group(grp)
        grp_pairs = [by_id[i] for i in want]
        old_acc = sum(_sim_only(p) == p.gt for p in grp_pairs) / len(grp_pairs)
        new_acc = sum(_new_rule(p) == p.gt for p in grp_pairs) / len(grp_pairs)
        print(f"\nGroup {grp} ({len(grp_pairs)}): {desc}")
        print(f"  sim-only  : {_pct(old_acc)}  {_bar(old_acc)}")
        print(f"  new rules : {_pct(new_acc)}  {_bar(new_acc)}")
        # Show sim-only errors in this group
        wrong = [(p, _sim_only(p)) for p in grp_pairs if _sim_only(p) != p.gt]
        for p, pred in wrong:
            new_pred = _new_rule(p)
            fixed = "✓fixed" if new_pred == p.gt else f"✗still {new_pred}"
            print(f"    [{p.id}] sim={p.real_sim:.3f} ret={p.ret_type:>9}"
                  f" → {pred} (GT:{p.gt}) [{fixed}]")

    # Remaining errors of new rule
    nr = results["New rules (ours)"]
    new_errors = [(p, pred) for p, pred in zip(pairs, nr["predictions"]) if pred != p.gt]
    print()
    print("─" * 70)
    print(f"New rule remaining errors ({len(new_errors)}/{n}) — acknowledged limitations")
    print("─" * 70)
    for p, pred in new_errors:
        print(f"  [{p.id}] sim={p.real_sim:.3f}  pred={pred}  GT={p.gt}")
        print(f"         {p.note}")

    # Verdict
    so = results["sim-only (old)"]
    nr = results["New rules (ours)"]
    print()
    print("─" * 70)
    print("Thesis verdict")
    print("─" * 70)
    print(f"  Accuracy   : {_pct(so['accuracy'])} (sim-only)  →  {_pct(nr['accuracy'])} (new rules)")
    print(f"  FalseReuse : {_pct(so['false_reuse_pct'])} (sim-only)  →  {_pct(nr['false_reuse_pct'])} (new rules)")
    print(f"  ADAPT Acc  : {_pct(so['adapt_accuracy'])} (sim-only)  →  {_pct(nr['adapt_accuracy'])} (new rules)")
    print()
    print("  sim-only fails because the embedding model encodes object identity")
    print("  strongly: pickup-cylinder vs handover-cylinder gives sim=0.939,")
    print("  causing sim-only to say REUSE for fundamentally different operations.")
    print()
    print("  The new rule adds action_match (from DB task_type) and object_match")
    print("  (from keyword detection) to block false REUSE while preserving the")
    print("  correct ADAPT bridge for compatible action pairs (pickup↔handover,")
    print("  place↔handover).")
    print()


if __name__ == "__main__":
    main()
