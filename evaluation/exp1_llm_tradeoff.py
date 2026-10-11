"""
Layer 1 — EXP1: four routing methods (accuracy × latency), overall + per group
===========================================================================

Dataset: `master_dataset.py` EXP1_CASES (100 commands, groups A–F).

Four named methods (console uses 中文方法名，不用「方法一」編號)：

  1) 分段關鍵字 — `_parse_intents` / `_segment` + `_classify_span` (production-shaped).
  2) 整句關鍵字 — `exp1_intent_routing._utterance_level_keyword` (whole-utterance gate).
  3) LLM 單標 — analytic surrogate, or OpenAI JSON single intent with `--live-llm`.
  4) LLM 多標 — coarse 1–3 span surrogate + same classifier, or OpenAI with `--live-llm`.

Metrics: strict sequence match vs GT (`_c1_ok` only). Latency: local micro-bench
per method (and per group where applicable), measured. LLM surrogates report nominal budget
estimates (~850 / ~950 ms) for API-like cost; these are not end-to-end measurements.

Usage:
  python -m evaluation.exp1_llm_tradeoff
  python -m evaluation.exp1_llm_tradeoff --live-llm [--live-max 100]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from evaluation.datasets.master_dataset import EXP1_CASES, IntentCase

# Re-use the canonical keyword pipeline (no drift from a second copy)
from evaluation.exp1_intent_routing import (  # noqa: E402
    _parse_intents,
    _utterance_level_keyword,
    _classify_span,
    _segment,
)

# ── Method display names ─────────────────────────────────────────────────────

NAME_SPAN = "分段關鍵字"
NAME_UTT = "整句關鍵字"
NAME_LLM1 = "LLM 單標"
NAME_LLMN = "LLM 多標"

GROUP_ORDER: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
GROUP_DESC = {
    "A": "單一 EXECUTE n=30",
    "B": "單一 ANSWER n=20",
    "C": "CLARIFY n=10",
    "D": "EXEC+ANSWER n=20",
    "E": "ANSWER+EXEC n=10",
    "F": "EXEC+EXEC n=10",
}

# ── LLM 單標：analytic surrogate (eval_baseline_comparison._c1_llm_only) ───────


def _llm_single_surrogate(case: IntentCase) -> List[str]:
    gt = case.intents
    grp = case.group
    if grp == "A":
        return ["EXECUTE"]
    if grp == "B":
        return ["ANSWER"]
    if grp == "C":
        cmd = case.command
        if any(m in cmd for m in ("什麼", "哪", "嗎", "呢", "怎麼")):
            return ["CLARIFY"]
        return ["EXECUTE"]
    if grp in ("D", "E"):
        if grp == "D":
            return ["ANSWER"]
        return ["EXECUTE"]
    if grp == "F":
        return ["EXECUTE"]
    return ["EXECUTE"]


# ── LLM 多標：coarse segmentation surrogate (weaker than full _segment) ─────

_WEAK_THEN = re.compile(r"[，,]?\s*然後")


def _weak_llm_segments(command: str) -> List[str]:
    """Fewer boundaries than full `_segment` — misses comma / 後 / ？ splits."""
    parts = [p.strip() for p in _WEAK_THEN.split(command) if p.strip()]
    if len(parts) <= 1 and ("，" in command or "," in command):
        parts = [p.strip() for p in re.split(r"[，,]\s*", command) if p.strip()]
    return parts if parts else [command.strip()]


def _llm_multi_coarse(case: IntentCase) -> List[str]:
    return [_classify_span(s) for s in _weak_llm_segments(case.command)]


def _c1_ok(predicted: List[str], gt: List[str]) -> bool:
    return len(predicted) == len(gt) and all(p == g for p, g in zip(predicted, gt))


def _scores_from_preds(cases: Sequence[IntentCase], preds: Sequence[List[str]]) -> Tuple[int, float]:
    n = len(cases)
    cor = sum(1 for c, p in zip(cases, preds) if _c1_ok(p, c.intents))
    return cor, (cor / n if n else 0.0)


def _per_group_scores(
    cases: Sequence[IntentCase], preds: Sequence[List[str]]
) -> Dict[str, Tuple[int, int, float]]:
    """group -> (correct, n, acc)."""
    bins: Dict[str, List[Tuple[IntentCase, List[str]]]] = {g: [] for g in GROUP_ORDER}
    for c, p in zip(cases, preds):
        if c.group in bins:
            bins[c.group].append((c, p))
    out: Dict[str, Tuple[int, int, float]] = {}
    for g in GROUP_ORDER:
        items = bins[g]
        n = len(items)
        if n == 0:
            continue
        cr = sum(1 for c, p in items if _c1_ok(p, c.intents))
        out[g] = (cr, n, cr / n)
    return out


def _bench_subset_ms(
    subcases: List[IntentCase],
    pred_fn: Callable[[IntentCase], List[str]],
    *,
    inner_repeats: int = 60,
) -> float:
    """Mean ms per utterance for cases in *subcases* (local CPU only)."""
    if not subcases:
        return 0.0
    for c in subcases[: min(20, len(subcases))]:
        pred_fn(c)
    n = len(subcases)
    t0 = time.perf_counter()
    for _ in range(inner_repeats):
        for c in subcases:
            pred_fn(c)
    return ((time.perf_counter() - t0) / (n * inner_repeats)) * 1000.0


_LLM_SYS_C1 = """You are the instruction-understanding module for a Chinese human–robot co-assembly station.
You MUST reply with a single JSON object (no markdown, no prose).

## Allowed intent labels (exactly these strings, UPPERCASE)
- EXECUTE — the human wants the robot to do a concrete physical action (拿/放/移/夾/遞/鎖/插/依序執行…).
- ANSWER — the human wants information: questions, checks, explanations (什麼/嗎/呢/告訴我/說明/哪些/怎麼/是否/狀態…).
- CLARIFY — the utterance is too vague, anaphoric without context (那個/上次/剛才/繼續), or cannot be safely mapped to one action; also use when the human is not clearly ordering motion vs asking.

Do NOT output ASK, QUERY, ACTION, etc. If you would say “ask the human”, use CLARIFY.

## Single vs multi
- **Single-intent**: the whole sentence has one dominant communicative goal → one label (single-label API) or `intents` of length 1 (multi-label API).
- **Multi-intent**: the sentence mixes motion + Q&A, two motions, or two information goals in **left-to-right reading order** → ordered list of 2–3 labels (multi-label API, max 3).

## Safety
Never label a span that is purely informational / interrogative as EXECUTE. EXECUTE only when that span orders or implies a specific robot motion."""


# Shared for live LLM 單標 / LLM 多標: three situations + combo hints (EXP1-style GT).
_LLM_USER_C1_SITUATIONS = """
=== 情況一：單一意圖（整句一個目的）===
- 只有動作：例如「幫我把圓柱移動到組裝台」→ 一個 EXECUTE。
- 只有問答：例如「目前步驟到哪了？」→ 一個 ANSWER。
- 過短或缺關鍵物件／指代不明：例如「幫我做一下」「拿那個」→ 一個 CLARIFY。

=== 情況二：複合意圖（同一句多段，依文字先後排列）===
連接詞或標點常切開兩段責任：然後、接著、之後、並且、順便、逗號、句號後再接問句、「完成後／做完後」再接「告訴我／說明」等。
- 「先做動作再問」：例如「把螺絲鎖上，然後告訴我下一步」→ EXECUTE 然後 ANSWER → 序列 [EXECUTE, ANSWER]。
- 「先問再做動作」：例如「告訴我夾爪開著嗎，若開著幫我關上」→ [ANSWER, EXECUTE]（依你斷出的左到右子句）。
- 「兩個動作」：例如「先拿圓柱，再放到支架上」→ [EXECUTE, EXECUTE]。

=== 情況三：安全（避免把「問」當成「做」）===
若該段是在問狀態、要解釋、要確認，而不是下達具體可執行的單一動作，該段標 ANSWER 或 CLARIFY，不要標 EXECUTE。

=== 標籤組合（多標 API 依序輸出；單標 API 整句只能選一個標籤）===
合法的多段組合僅由 {EXECUTE, ANSWER, CLARIFY} 排列而成，例如：
EXECUTE+ANSWER、ANSWER+EXECUTE、EXECUTE+EXECUTE、ANSWER+ANSWER、CLARIFY 單獨、EXECUTE+CLARIFY 等。
先判斷是 single 還是 multi：若明顯有兩段以上不同目的，必須走 multi（`intents` 長度 2 或 3），不要硬塞成單一 EXECUTE。
"""


def _parse_json_obj(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return json.loads(t)


def _normalize_c1_label(raw: str) -> Optional[str]:
    """Map common synonyms to EXP1 three-way labels; unknown → None."""
    s = (raw or "").strip().upper().replace(" ", "_")
    if s in ("EXECUTE", "ANSWER", "CLARIFY"):
        return s
    if s in ("ASK", "HUMAN", "NEED_HUMAN", "NEED-CLARIFICATION", "UNCLEAR", "UNKNOWN"):
        return "CLARIFY"
    if s in ("QUESTION", "QUERY", "INFO", "INFORMATION", "QNA", "QA"):
        return "ANSWER"
    if s in ("ACTION", "ACT", "TOOL", "COMMAND", "MOTION", "DO", "TASK"):
        return "EXECUTE"
    return None


def _live_m3_single(client, model: str, command: str) -> Tuple[List[str], float]:
    user = (
        _LLM_USER_C1_SITUATIONS
        + "\n--- CURRENT TASK (single-label API) ---\n"
        + f'Command: "{command}"\n\n'
        + "You are in **single-label mode**: the downstream API only accepts **one** label for the **entire** utterance.\n"
        + "Return JSON exactly in this shape:\n"
        + '  {"intent":"EXECUTE"|"ANSWER"|"CLARIFY"}\n'
        + "If the sentence clearly mixes motion and Q&A, you still must output **one** `intent` — choose the label that best reflects the **primary** risk/safety framing "
        + "(when unsure between motion vs question, prefer ANSWER or CLARIFY over EXECUTE).\n"
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _LLM_SYS_C1},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    raw = (resp.choices[0].message.content or "{}").strip()
    data = _parse_json_obj(raw)
    lab = _normalize_c1_label(str(data.get("intent", "")))
    if lab is None:
        lab = "EXECUTE"
    return [lab], elapsed_ms


def _live_m4_multi(client, model: str, command: str) -> Tuple[List[str], float]:
    user = (
        _LLM_USER_C1_SITUATIONS
        + "\n--- CURRENT TASK (multi-label API, up to 3 segments) ---\n"
        + f'Command: "{command}"\n\n'
        + "You are in **multi-label mode**.\n"
        + "1) Decide **single vs multi**: one communicative goal → single-intent; two or three distinct goals in reading order → multi-intent.\n"
        + "2) Split mentally on connectors (然後/接著/並且/逗號/？後續句/完成後…告訴我…); each **clause** gets one of EXECUTE / ANSWER / CLARIFY.\n"
        + "3) Return JSON exactly in this shape (max 3 entries):\n"
        + '  {"is_multi": true|false, "intents":["EXECUTE"|"ANSWER"|"CLARIFY", ...]}\n'
        + "- `is_multi` must be true iff len(intents) >= 2.\n"
        + "- `intents` order = **left-to-right** clause order in the original sentence.\n"
        + "- Use only the three allowed strings in `intents` (no ASK; use CLARIFY).\n"
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _LLM_SYS_C1},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    raw = (resp.choices[0].message.content or "{}").strip()
    data = _parse_json_obj(raw)
    arr = data.get("intents")
    if not isinstance(arr, list):
        arr = data.get("intent_sequence")
    if not isinstance(arr, list):
        arr = data.get("labels")
    if not isinstance(arr, list):
        return ["EXECUTE"], elapsed_ms
    out: List[str] = []
    for x in arr[:3]:
        lab = _normalize_c1_label(str(x))
        if lab is not None:
            out.append(lab)
    return (out if out else ["EXECUTE"]), elapsed_ms


def _bench_local_ms_per_case(
    cases: List[IntentCase],
    pred_fn: Callable[[IntentCase], List[str]],
    *,
    inner_repeats: int = 200,
) -> float:
    """Mean ms per utterance = total time / (N * repeats) * 1000."""
    # Warm-up
    for c in cases[:20]:
        pred_fn(c)
    n = len(cases)
    t0 = time.perf_counter()
    for _ in range(inner_repeats):
        for c in cases:
            pred_fn(c)
    elapsed = time.perf_counter() - t0
    return (elapsed / (n * inner_repeats)) * 1000.0


def _cases_in_group(cases: Sequence[IntentCase], group: str) -> List[IntentCase]:
    return [c for c in cases if c.group == group]


def _fmt_ms(ms: float, *, nominal: bool) -> str:
    if nominal:
        return f"~{ms:.0f}"
    if ms >= 50.0:
        return f"{ms:.1f}"
    return f"{ms:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer 1 — EXP1: four methods, accuracy + latency (overall + per group)")
    ap.add_argument(
        "--live-llm",
        action="store_true",
        help="Call OpenAI for LLM 單標 / LLM 多標 (OPENAI_API_KEY; optional OPENAI_MODEL)",
    )
    ap.add_argument(
        "--live-max",
        type=int,
        default=100,
        metavar="N",
        help="Evaluate only the first N EXP1 cases with --live-llm (default 100)",
    )
    args = ap.parse_args()

    cases: List[IntentCase] = EXP1_CASES[: max(1, min(args.live_max, len(EXP1_CASES)))]
    n = len(cases)

    NOM_SINGLE_MS = 850.0
    NOM_MULTI_MS = 950.0

    print(f"\n{'=' * 72}")
    print("  Layer 1 — EXP1：四種方法 × 正確率 × 延遲（整體 + 各 Group）")
    print(f"  樣本數: {n}" + ("  [Live：LLM 單標 / LLM 多標 走 API]" if args.live_llm else ""))
    print(f"{'=' * 72}")

    # ── Predictions (one pass per case; live records latency) ─────────
    fn_span = lambda c: _parse_intents(c.command)
    preds_span = [fn_span(c) for c in cases]
    cor_sp, acc_sp = _scores_from_preds(cases, preds_span)
    lat_sp = _bench_local_ms_per_case(cases, fn_span, inner_repeats=200)

    preds_utt = [_utterance_level_keyword(c) for c in cases]
    cor_ut, acc_ut = _scores_from_preds(cases, preds_utt)
    lat_ut = _bench_local_ms_per_case(cases, _utterance_level_keyword, inner_repeats=200)

    if args.live_llm:
        from ws_racr.config import get_openai_client, get_model_name

        client = get_openai_client()
        model = get_model_name()
        print(f"\n  [Live LLM] model={model!r}")

        lat_l1: List[float] = []
        preds_l1: List[List[str]] = []
        for c in cases:
            p, ms = _live_m3_single(client, model, c.command)
            preds_l1.append(p)
            lat_l1.append(ms)

        lat_ln: List[float] = []
        preds_ln: List[List[str]] = []
        for c in cases:
            p, ms = _live_m4_multi(client, model, c.command)
            preds_ln.append(p)
            lat_ln.append(ms)

        cor_l1, acc_l1 = _scores_from_preds(cases, preds_l1)
        cor_ln, acc_ln = _scores_from_preds(cases, preds_ln)
        lat_l1_mean = sum(lat_l1) / len(lat_l1) if lat_l1 else 0.0
        lat_ln_mean = sum(lat_ln) / len(lat_ln) if lat_ln else 0.0
        nominal_l1 = nominal_ln = False
    else:
        preds_l1 = [_llm_single_surrogate(c) for c in cases]
        preds_ln = [_llm_multi_coarse(c) for c in cases]
        cor_l1, acc_l1 = _scores_from_preds(cases, preds_l1)
        cor_ln, acc_ln = _scores_from_preds(cases, preds_ln)
        lat_l1_mean = NOM_SINGLE_MS
        lat_ln_mean = NOM_MULTI_MS
        nominal_l1 = nominal_ln = True

    lat_coarse_cpu = _bench_local_ms_per_case(cases, _llm_multi_coarse, inner_repeats=200)

    methods: List[Dict[str, object]] = [
        {
            "title": NAME_SPAN,
            "preds": preds_span,
            "cor": cor_sp,
            "acc": acc_sp,
            "lat_overall": lat_sp,
            "lat_nominal": False,
        },
        {
            "title": NAME_UTT,
            "preds": preds_utt,
            "cor": cor_ut,
            "acc": acc_ut,
            "lat_overall": lat_ut,
            "lat_nominal": False,
        },
        {
            "title": NAME_LLM1,
            "preds": preds_l1,
            "cor": cor_l1,
            "acc": acc_l1,
            "lat_overall": lat_l1_mean,
            "lat_nominal": nominal_l1,
        },
        {
            "title": NAME_LLMN,
            "preds": preds_ln,
            "cor": cor_ln,
            "acc": acc_ln,
            "lat_overall": lat_ln_mean,
            "lat_nominal": nominal_ln,
        },
    ]

    for m in methods:
        print(f"\n  [{m['title']}]  正確 {m['cor']}/{n} = {m['acc']:.1%}  |  整體延遲 "
              f"{_fmt_ms(float(m['lat_overall']), nominal=bool(m['lat_nominal']))} ms/utt"
              + ("（nominal 估計，非實測）" if m["lat_nominal"] else "（本機 micro-benchmark 實測）"))

    if not args.live_llm:
        print(f"\n  （{NAME_LLMN} 本機 surrogate 分類耗時 ~{lat_coarse_cpu:.4f} ms/utt；表列 ~{NOM_MULTI_MS:.0f} 為 nominal API 級延遲估計，非端到端實測）")

    # ── Overall summary ───────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  整體（100 筆內有效 n）— 正確率 / 延遲 ms·utt⁻¹")
    print(f"{'─' * 72}")
    w = 14
    hdr = f"  {'方法':<{w}} {'正確率':>10} {'延遲 (ms)':>14}"
    print(hdr)
    print(f"  {'─' * w} {'─' * 10} {'─' * 14}")
    for m in methods:
        print(
            f"  {str(m['title']):<{w}} {float(m['acc']):>9.1%} "
            f"{_fmt_ms(float(m['lat_overall']), nominal=bool(m['lat_nominal'])):>14}"
        )

    # ── Per-group accuracy ──────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  各 Group 正確率（與 GT 意圖序列完全一致）")
    print(f"{'─' * 72}")
    pg_span = _per_group_scores(cases, preds_span)
    pg_utt = _per_group_scores(cases, preds_utt)
    pg_l1 = _per_group_scores(cases, preds_l1)
    pg_ln = _per_group_scores(cases, preds_ln)
    wlab = 24
    h2 = (
        f"  {'Group':<6} {'說明':<{wlab}} {'N':>4}  "
        f"{NAME_SPAN:>8} {NAME_UTT:>8} {NAME_LLM1:>8} {NAME_LLMN:>8}"
    )
    print(h2)
    print(f"  {'─' * 6} {'─' * wlab} {'─' * 4}  {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for g in GROUP_ORDER:
        if g not in pg_span:
            continue
        _, ng, _ = pg_span[g]
        a_s = pg_span[g][2] * 100.0
        a_u = pg_utt[g][2] * 100.0
        a_1 = pg_l1[g][2] * 100.0
        a_n = pg_ln[g][2] * 100.0
        desc = GROUP_DESC[g]
        print(f"  {g:<6} {desc:<{wlab}} {ng:>4}  {a_s:>7.1f}% {a_u:>7.1f}% {a_1:>7.1f}% {a_n:>7.1f}%")

    # ── Per-group latency ───────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  各 Group 延遲 ms·utt⁻¹（關鍵字與 LLM 多標 surrogate 為該子集 micro-bench；")
    print("  LLM 單/多標在無 --live-llm 時列 nominal，各 Group 同值）")
    print(f"{'─' * 72}")
    print(h2)
    print(f"  {'─' * 6} {'─' * wlab} {'─' * 4}  {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for g in GROUP_ORDER:
        if g not in pg_span:
            continue
        sub = _cases_in_group(cases, g)
        _, ng, _ = pg_span[g]
        ms_sp = _bench_subset_ms(sub, fn_span, inner_repeats=50)
        ms_ut = _bench_subset_ms(sub, _utterance_level_keyword, inner_repeats=50)
        ms_ln = _bench_subset_ms(sub, _llm_multi_coarse, inner_repeats=50)
        if args.live_llm:
            idxs = [i for i, c in enumerate(cases) if c.group == g]
            ms_l1 = sum(lat_l1[i] for i in idxs) / len(idxs) if idxs else 0.0
            ms_lnv = sum(lat_ln[i] for i in idxs) / len(idxs) if idxs else 0.0
        else:
            ms_l1 = NOM_SINGLE_MS
            ms_lnv = NOM_MULTI_MS
        desc = GROUP_DESC[g]
        print(
            f"  {g:<6} {desc:<{wlab}} {ng:>4}  "
            f"{_fmt_ms(ms_sp, nominal=False):>8} {_fmt_ms(ms_ut, nominal=False):>8} "
            f"{_fmt_ms(ms_l1, nominal=not args.live_llm):>8} {_fmt_ms(ms_lnv, nominal=not args.live_llm):>8}"
        )

    print("\n  註：延遲欄位單位皆為 ms/utt；關鍵字與 LLM 多標 surrogate 本機段為子集 micro-bench。")
    if not args.live_llm:
        print(
            f"  LLM 單標 / LLM 多標 未開 --live-llm 時延遲為 nominal 估計（~{NOM_SINGLE_MS:.0f} / ~{NOM_MULTI_MS:.0f} ms，非端到端實測），"
            "與 Group 無關故各列相同。"
        )
    else:
        print("  LLM 單標 / 多標 延遲為該 Group 內樣本之 API 往返平均（本輪實測）。")


if __name__ == "__main__":
    main()
