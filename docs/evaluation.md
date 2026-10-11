# Evaluation

The four experiments of the thesis (EXP1–EXP4) are in `evaluation/`. They run on a CPU in a few seconds and need only `numpy` (plus the package itself).

```bash
pip install -e ".[dev]"
python -m evaluation.run_all                 # run all experiments, check every headline number
python -m evaluation.run_all --only exp4     # one experiment
python -m evaluation.run_all --out results   # also write the full reports to results/<name>.txt
python -m evaluation.exp4_e2e_ablation       # a single script prints its full report
pytest tests/test_evaluation_regression.py   # the same headline numbers as a unit test
```

`run_all` exits non-zero if a headline number below changes.

## Setup

We conduct four experiments targeting each design decision: (EXP1) intent routing on 100 utterances, (EXP2) feasibility gating on 60 plan-reuse cases, (EXP3) action-type risk on 80 conflict pairs, and (EXP4) end-to-end ablation on 44 stratified scenarios.
Metrics: routing accuracy (EXECUTE / ASK / ANSWER), harmful reuse rate (HRR), false reuse rate (FRR), and decision latency.

| Experiment | Purpose | Type |
|---|---|---|
| EXP1 | Layer 1 intent segmentation and routing | module functional validation |
| EXP2 | necessity of Layer 2 (execution feasibility) | module functional validation |
| EXP3 | Layer 3 action-risk rules | in-sample rule-validation |
| EXP4 | end-to-end ablation of the three layers | end-to-end ablation |

## Headline numbers

| Experiment | Script | Cases | Result |
|---|---|---|---|
| EXP1 intent routing (Layer 1) | `exp1_intent_routing.py` | 100 annotated utterances | utterance-level keyword 49.0 % -> segment-level (Layer 1) 92.0 % |
| EXP2 feasibility gate (Layer 2) | `exp2_feasibility_gate.py` | 60 cases | harmful reuse on missing-object cases 100 % -> 0 % |
| EXP3 execution risk (Layer 3) | `exp3_execution_risk.py` | 80 retrieved pairs | accuracy 48.8 % (similarity only) -> 93.8 % (Layer 3 rules); FRR 28.7 % -> 2.5 % |
| EXP4 end-to-end ablation | `exp4_e2e_ablation.py` | 44 stratified scenarios | accuracy 31.8 % -> 56.8 % (B0 + Layer 1) -> 70.5 % (B0 + Layer 1 + Layer 2) -> 93.2 % (Full Layer 1–3) |
## EXP1: Intent Routing (100 utterances)

Layer 1 is evaluated on 100 annotated utterances spanning six groups: pure EXECUTE (A, n=30), pure ANSWER (B, n=20), clarification-needed (C, n=10), EXECUTE+ANSWER (D, n=20), ANSWER+EXECUTE (E, n=10), and EXECUTE+EXECUTE (F, n=10).
Groups D–F are *compound multi-intent*: utterance-level methods that return a single label fail entirely on all multi-intent groups (n=40).

Intent routing on 100 utterances. "Multi-intent" = groups D+E+F (n=40), where all intents must be returned in order.

| Method | Overall | Multi-intent (D+E+F, n=40) | Latency per utterance |
|---|---|---|---|
| Utterance-level keyword | 49.0% | 0% | ~0.002 ms (local micro-benchmark) |
| LLM single-prompt | 51.0% | 0% | ~850 ms (nominal estimate) |
| LLM multi-step | 85.0% | 77% | ~950 ms (nominal estimate) |
| **Segment-level (Layer 1)** | **92.0%** | **93%** | **~0.005 ms (local micro-benchmark)** |

Latency: rule-based parsing times are measured locally (CPU, micro-benchmark, no I/O). The LLM latencies are nominal API-level budget estimates, not end-to-end measurements; no live LLM calls are made in the offline evaluation.

The table shows that utterance-level and LLM single-prompt methods fail entirely on compound commands (0% multi-intent accuracy) because they produce at most one intent label.
LLM multi-step achieves 85% overall but still misses 23% of compound cases.
Layer 1 segment parsing reaches 92% overall and 93% on multi-intent groups, with ~0.005 ms rule-parsing time (measured micro-benchmark) vs. an estimated ~850–950 ms for LLM-based routing (nominal estimate).

## EXP2: Feasibility Gate (60 cases)

On 60 labeled plan-reuse scenarios (15 object-missing, 45 object-available; labels follow the same rule definitions as the system), B0 without Layer 2 exhibits 100% HRR on missing-object cases (all 15 incorrectly reused).
Adding Layer 2 reduces HRR to 0% with 0% false rejection on available cases, even when $s>0.9$.
EXP2 primarily validates the $O_t$ object-availability gate; global $H_t$/$R_t$ conditions follow the same deterministic rule structure.

| Setting | HRR (missing, n=15) | False rejection (available, n=45) |
|---|---|---|
| without Layer 2 (direct REUSE) | 100% | 0% |
| **with Layer 2 (object gate)** | **0%** | **0%** |

## EXP3: Reuse Rules (80 action-conflict pairs)

EXP3 uses 80 labeled embedding pairs (four groups of 20: A same-type/same-object → REUSE; B same-type/different-object → ADAPT; C incompatible-action high-similarity → ADAPT; D low-similarity or ambiguous → PLAN).
Similarity-only reuse yields 48.8% accuracy and 28.7% FRR; Layer 3 rules improve accuracy to 93.8% and FRR to 2.5%.
Group C (incompatible actions at high similarity) is the critical failure zone for similarity-only baselines (15.0% accuracy); action-type rules raise it to 100%.

| Group | Ground truth | Similarity threshold | Layer 3 rules |
|---|---|---|---|
| A (n=20) | same type, same object → REUSE | 95.0% | 90.0% |
| B (n=20) | same type, different object → ADAPT | 40.0% | 85.0% |
| C (n=20) | incompatible action type at high similarity → ADAPT | 15.0% | **100.0%** |
| D (n=20) | low similarity or ambiguous → PLAN | 45.0% | **100.0%** |
| **Overall** (n=80) | | 48.8% | **93.8%** |

## EXP4: End-to-End Ablation (44 scenarios)

The purpose of this evaluation is controlled failure-mode isolation rather than dataset-scale benchmarking.
The 44 cases are *stratified* to isolate each layer: 12 ambiguous commands (Cat-A, Layer 1 stress), 10 plan-reuse cases with feasibility variation (Cat-B: 6 object-missing REJECTs + 4 feasible controls, Layer 2 stress), 12 execution-risk pairs (Cat-C, Layer 3 stress), and 10 routine instructions (Cat-D).
Each case is annotated with expected routing, reject/reuse/adapt/plan decision, and arm motion class; accuracy is the fraction of fully correct end-to-end decisions.

**Baseline.** B0 is the single-stage EXECUTE baseline: the whole utterance is routed to plan retrieval without intent splitting; if a similar historical plan exists (s ≥ 0.50) it is reused directly, otherwise the LLM re-plans. B0 runs no feasibility gate or risk rule and takes neither the ASK nor the ANSWER path. Three progressive variants add Layer 1, Layer 2, and Layer 3 sequentially to B0.

**Layer ablation on 44 stratified scenarios.** Acc = end-to-end accuracy over all 44 cases; Rout.Err, HRR, and FRR are subset-specific rates computed on their corresponding stress categories (Cat-A, Cat-B, Cat-C respectively).

| Config. | Acc. | Rout.Err | HRR | FRR |
|---|---|---|---|---|
| B0 baseline | 31.8 | 92% | 60% | 58% |
| B0 + Layer 1 | 56.8 | 0% | 60% | 58% |
| B0 + Layer 1 + Layer 2 | 70.5 | 0% | 0% | 58% |
| **Full Layer 1–3** | **93.2** | **0%** | **0%** | **0%** |

Routing error is the proportion of Cat-A scenarios in which the system performs no intent splitting and treats the whole utterance as a single EXECUTE command (EXECUTE_ONLY).

**Per-category accuracy (%).** Each category is primarily designed to stress one failure mode; rows show progressive layer additions.

| Config. | Cat-A | Cat-B | Cat-C | Cat-D |
|---|---|---|---|---|
| B0 | 8.3 | 20.0 | 41.7 | 60.0 |
| B0 + Layer 1 | 83.3 | 20.0 | 41.7 | 80.0 |
| B0 + Layer 1 + Layer 2 | 83.3 | 80.0 | 41.7 | 80.0 |
| **Full Layer 1–3** | 83.3 | **100.0** | **100.0** | 90.0 |

The tables show a near block-diagonal improvement pattern: each layer primarily improves its intended failure category, with limited spillover on routine cases.
Layer 1 raises Cat-A from 8.3% to 83.3% (+25.0 pp overall) while Cat-B/C are unaffected.
Layer 2 raises Cat-B from 20% to 80% (+13.6 pp overall; HRR 60% → 0%) while Cat-C remains at 41.7%; the two remaining Cat-B errors are feasible cases with action-type conflicts resolved only by Layer 3.
Layer 3 raises Cat-C from 41.7% to 100% (+22.7 pp overall; FRR 58% → 0%).
The three remaining errors (93.2% ≠ 100%) comprise ≈2 semantically similar multi-intent utterances missed by keyword splitting (Cat-A) and one Cat-D case with STT-induced character errors; they are boundary cases in perception or Layer 1 parsing, not Layer 2/3 logic failures.

## Unit tests

`pytest` (from the repository root) runs the unit tests in `tests/`: Layer 1 segmentation, Layer 2 reject conditions and Layer 3 rule boundaries (`decide_risk`), the deictic grounding FSM, world-state types, the interaction log, early routing of questions, and the orchestrator prompt assembly.
