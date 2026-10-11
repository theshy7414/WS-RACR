# WS-RACR

**WS-RACR: World-State-Centered Retrieval-Augmented Collaboration Reasoning for Human–Robot Collaboration**

[![CI](https://github.com/theshy7414/WS-RACR/actions/workflows/ci.yml/badge.svg)](https://github.com/theshy7414/WS-RACR/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange.svg)](https://docs.ros.org/en/humble/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Master's thesis project, National Taiwan University, Department of Mechanical Engineering (2026).
Thesis: *A World-State-Centered Retrieval-Augmented Framework for Human–Robot Collaboration Applications*
(以世界狀態為核心進行檢索增強人機協作之應用).
Website: <https://theshy7414.github.io/WS-RACR/>

**TL;DR** An LLM-based collaboration system for human–robot assembly on a Techman TM5-900 arm. Plans retrieved
from memory are only *candidates*: they are verified against a live **world state** before the robot decides to
reuse, adapt, clarify or reject them.

---

## The problem

Natural-language HRC systems built on LLMs and RAG tend to fail in three ways:

1. **Ambiguous, multi-intent commands.** One utterance may contain an action, a question and an unclear request.
2. **Stale plans.** A retrieved plan may no longer match the scene (a part is missing, the gripper is holding
   something, the human said stop).
3. **Plans that look similar but behave differently.** Retrieval by semantic similarity alone can mark a plan with
   a different action type as directly reusable.

## Approach

### World state

All reasoning uses a shared world state that is updated continuously by asynchronous perception:

W<sub>t</sub> = { S<sub>t</sub>, O<sub>t</sub>, H<sub>t</sub>, R<sub>t</sub> }

| Component | Meaning | Main source |
|---|---|---|
| S<sub>t</sub> | Task step / assembly progress | TSM-ResNet18 step recognition + SOP state machine |
| O<sub>t</sub> | Object state | YOLOv8-seg on a top-down RGB-D camera (D435i) |
| H<sub>t</sub> | Human interaction state | MediaPipe gesture / pointing on a front RGB-D camera (D455), speech |
| R<sub>t</sub> | Robot execution state (gripper, held object, last skill result) | Robot controller feedback |

### Three-layer collaboration decision

| Layer | Role | Code |
|---|---|---|
| **Layer 1** – instruction understanding | Splits an utterance into sub-intents and routes each (execute / answer / ask). | `orchestrator/layer1_intent.py` |
| **Layer 2** – execution feasibility | Hard gate on W<sub>t</sub>: target visible (O<sub>t</sub>), gripper state (R<sub>t</sub>), human stop (H<sub>t</sub>). A failed gate means REJECT. | `decision/collaboration_policy.py` |
| **Layer 3** – execution risk | Four rules on similarity, action match and object match choose REUSE, ADAPT or CLARIFY. | `decision/collaboration_policy.py` |

Plan RAG (PostgreSQL + pgvector) **only recalls candidates**. REUSE / ADAPT / CLARIFY / REJECT is decided by
Layer 2 and Layer 3, never by a similarity threshold alone. Memory consists of an interaction log, the plan
store (episodic memory) and a temporal perception log. Deictic grounding combines speech and pointing to resolve
"that one". Details: [`docs/architecture.md`](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/theshy7414/WS-RACR.git && cd WS-RACR
pip install -e ".[dev]"            # Python >= 3.10; numpy, pyyaml, sqlalchemy, pytest

pytest                             # unit tests for the decision logic
python -m evaluation.run_all       # run EXP1–EXP4, check the headline numbers
```

The full system (cameras, GPU, ROS 2, PostgreSQL + pgvector, OpenAI key, TM5-900) is described in
[`docs/setup.md`](docs/setup.md). Entry point: `python -m ws_racr.app.main [--enable-robot | --perception-only] [--db-url URL]`.

## Evaluation

We conduct four experiments targeting each design decision: (EXP1) intent routing on 100 utterances, (EXP2) feasibility
gating on 60 plan-reuse cases, (EXP3) action-type risk on 80 conflict pairs, and (EXP4) end-to-end ablation on 44
stratified scenarios. Metrics: routing accuracy (EXECUTE / ASK / ANSWER), harmful reuse rate (HRR), false reuse rate
(FRR), and decision latency. Full description: [`docs/evaluation.md`](docs/evaluation.md).

| Experiment | Cases | Result |
|---|---|---|
| EXP1 intent routing (Layer 1) | 100 annotated utterances | utterance-level keyword 49.0 % → segment-level (Layer 1) 92.0 % |
| EXP2 feasibility gate (Layer 2) | 60 cases | HRR on missing-object cases 100 % → 0 % |
| EXP3 execution risk (Layer 3) | 80 conflict pairs | accuracy 48.8 % (similarity only) → 93.8 % (Layer 3 rules); FRR 28.7 % → 2.5 % |
| EXP4 end-to-end ablation | 44 stratified scenarios | see below |

**EXP1: intent routing on 100 utterances.** Layer 1 is evaluated on 100 annotated utterances spanning six groups: pure
EXECUTE (A, n=30), pure ANSWER (B, n=20), clarification-needed (C, n=10), EXECUTE+ANSWER (D, n=20), ANSWER+EXECUTE
(E, n=10), and EXECUTE+EXECUTE (F, n=10). Groups D–F are *compound multi-intent*: utterance-level methods that return a
single label fail entirely on all multi-intent groups (n=40).

Intent routing on 100 utterances. "Multi-intent" = groups D+E+F (n=40), where all intents must be returned in order.

| Method | Overall | Multi-intent |
|---|---|---|
| Utterance-level keyword | 49.0% | 0% |
| LLM single-prompt | 51.0% | 0% |
| LLM multi-step | 85.0% | 77% |
| **Segment-level (Layer 1)** | **92.0%** | **93%** |

The table shows that utterance-level and LLM single-prompt methods fail entirely on compound commands (0% multi-intent
accuracy) because they produce at most one intent label. LLM multi-step achieves 85% overall but still misses 23% of
compound cases. Layer 1 segment parsing reaches 92% overall and 93% on multi-intent groups, with ~0.005 ms
rule-parsing time (local micro-benchmark) vs. an estimated ~850–950 ms for LLM-based routing (nominal estimate, not an
end-to-end measurement).

**EXP4: layer ablation on 44 stratified scenarios.** The purpose of this evaluation is controlled failure-mode
isolation rather than dataset-scale benchmarking. The 44 cases are *stratified* to isolate each layer: 12 ambiguous
commands (Cat-A, Layer 1 stress), 10 plan-reuse cases with feasibility variation (Cat-B: 6 object-missing REJECTs + 4
feasible controls, Layer 2 stress), 12 execution-risk pairs (Cat-C, Layer 3 stress), and 10 routine instructions
(Cat-D). Acc = end-to-end accuracy over all 44 cases; Rout.Err, HRR, and FRR are subset-specific rates computed on
their corresponding stress categories (Cat-A, Cat-B, Cat-C respectively).

| Config. | Acc. | Rout.Err | HRR | FRR |
|---|---|---|---|---|
| B0 baseline | 31.8 | 92% | 60% | 58% |
| B0 + Layer 1 | 56.8 | 0% | 60% | 58% |
| B0 + Layer 1 + Layer 2 | 70.5 | 0% | 0% | 58% |
| **Full Layer 1–3** | **93.2** | **0%** | **0%** | **0%** |

Each layer primarily improves its intended failure category (near block-diagonal pattern): Layer 1 raises Cat-A from
8.3 % to 83.3 %, Layer 2 raises Cat-B from 20 % to 80 % (HRR 60 % → 0 %), and Layer 3 raises Cat-C from 41.7 % to
100 % (FRR 58 % → 0 %). Per-category tables and the remaining errors are in
[`docs/evaluation.md`](docs/evaluation.md).

## Repository layout

```
src/ws_racr/        installable package (src layout), mapped to the thesis chapters
  world_state/         W_t = {S_t, O_t, H_t, R_t}, builder, stabilisation          (Ch. 3)
  perception/          YOLOv8-seg, MediaPipe gesture, TSM step recognition, RealSense (Ch. 3)
  decision/            Layer 2/3 policy, deictic grounding, command parser          (Ch. 4)
  orchestrator/        HRCOrchestrator, Layer 1, plan memory, answering             (Ch. 4)
  memory/, database/   interaction / temporal memory, PostgreSQL + pgvector plan RAG (Ch. 4)
  execution/           skills, robot controller, backends (MoveIt2 / TM), safety   (Ch. 3-4)
  hri/, app/           speech I/O, runtime (HRCSystem) and command-line entry
evaluation/          EXP1–EXP4 scripts, shared runner (`run_all`), datasets
tests/               pytest unit tests
training/            gesture classifier training scripts
tools/               calibration and analysis helpers (need hardware)
data/                example scenarios, knowledge-graph analysis figures
docs/                architecture / setup / evaluation notes, project website (GitHub Pages)
```

ROS 2: `package.xml`, `setup.py` and `setup.cfg` make the repository an `ament_python` package
(`ros2 run ws_racr hrc_system`).

## 中文摘要

本專案為臺大機械所碩士論文《以世界狀態為核心進行檢索增強人機協作之應用》的系統實作。
系統以世界狀態 W<sub>t</sub> = {S<sub>t</sub>, O<sub>t</sub>, H<sub>t</sub>, R<sub>t</sub>} 為共享上下文，整合雙 RGB-D 相機、
物件偵測、手勢辨識、步驟辨識與語音互動，並以三層協作決策（指令理解層、執行可行性層、執行風險層）
決定歷史計畫應重用（REUSE）、調整（ADAPT）、澄清（CLARIFY）或拒絕（REJECT）。
向量檢索只負責召回候選計畫，不以相似度直接決定是否執行。

端對端消融（EXP4）設計 44 筆分層標注場景，依序累加 Layer 1、Layer 2、Layer 3，比較四種系統配置之決策正確率。
場景分為四類：Cat-A（指令歧義，n=12，測 Layer 1）、Cat-B（可行性衝突，n=10，測 Layer 2）、Cat-C（執行風險，n=12，測 Layer 3）、
Cat-D（常規指令，n=10，基準對照）。完整系統達 93.2%（基線 B0 為 31.8%）；各層效果整體呈近似「區塊對角線」模式：
Layer 1 主要提升 Cat-A，Layer 2 主要提升 Cat-B，Layer 3 主要提升 Cat-C。
重現方式：`python -m evaluation.run_all`，說明見 [`docs/evaluation.md`](docs/evaluation.md)。

## Author

Szu-Nien Chi · [szunienchi1@gmail.com](mailto:szunienchi1@gmail.com) · [@theshy7414](https://github.com/theshy7414)

## License

[Apache-2.0](LICENSE). The Techman driver, URDF and MoveIt configuration are third-party and not redistributed here.
