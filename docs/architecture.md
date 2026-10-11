# Architecture

This page maps the system described in the thesis onto the code. Paths are relative to
`src/ws_racr/`.

> Terminology: **Layer 1 / 2 / 3** always means the three collaborative *decision* layers
> (instruction understanding, execution feasibility, execution risk). The three *system* layers are
> **perception**, **reasoning and memory**, and **execution**; they are not numbered.

## 1. System layers and packages

| System layer | Package | Contents |
|---|---|---|
| Perception | `perception/`, `world_state/` | YOLOv8-seg object detection and localisation, MediaPipe hand features + gesture classifier, TSM-ResNet18 step recognition, RealSense wrappers; world-state representation, stabilisation (hysteresis), spatial relations, SOP step definitions |
| Reasoning and memory | `decision/`, `orchestrator/`, `memory/`, `database/` | Layer 1/2/3 policy, deictic grounding, command parser; `HRCOrchestrator`; three kinds of memory; PostgreSQL + pgvector plan store |
| Execution | `execution/` | skill ontology and validators, `RobotController`, backends (MoveIt2 / TM SetPositions / TMScript), workspace monitor, failure detection and recovery, occlusion handling |
| Human interface | `hri/` | speech recognition, text-to-speech, rule-based spoken queries |
| Runtime | `app/` | `HRCSystem` (perception, robot, trigger and voice mixins) and the command-line entry point |

```
src/ws_racr/
  config.py              environment / .env driven configuration (paths, DB, robot, backend)
  world_state/           W_t representation, builder, stabilisation, step definitions
  perception/            cameras, YOLO localisation, step recognition, gesture recognition
  decision/              collaboration_policy (Layer 2/3), deictic_grounding, command_parser
  orchestrator/          HRCOrchestrator + Layer 1 (layer1_intent), plan memory, answering
  memory/                interaction log, temporal log, SOP knowledge, spatial memory
  database/              TaskManager (plan RAG), perception logger, Alembic schema
  execution/             skills, controller, backends, safety
  hri/                   voice_recognizer, tts_handler, voice_query_handler
  app/                   HRCSystem runtime, main entry point
```

## 2. World state

`world_state/state_representation.py` defines the shared context that every later stage reads:

W<sub>t</sub> = { S<sub>t</sub>, O<sub>t</sub>, H<sub>t</sub>, R<sub>t</sub> }

| Component | Field in `WorldState` | Source |
|---|---|---|
| S<sub>t</sub> task step | `step_class`, `step_confidence`, `phase`, `assembly_progress` | TSM-ResNet18 (9 step classes, `world_state/step_definitions.py`) fused with an event-driven progress tracker |
| O<sub>t</sub> objects | `objects: Dict[str, ObjectState]` (raw and stable location, visibility, occlusion, source camera) | YOLOv8-seg on the top-down D435i, depth deprojection into the robot base frame |
| H<sub>t</sub> human | `human` (`intent`, `intent_stable`, `safety`, `pointing_target`, ...) | MediaPipe features + gesture classifier + intent smoothing state machine on the front D455 |
| R<sub>t</sub> robot | `robot` (`gripper_state`, `holding_object_id`, `last_skill_result`, ...) | robot digital-IO feedback and skill results (`WorldStateBuilder` applies "robot IO truth") |

`world_state/builder.py` (`WorldStateBuilder`) assembles the state from the perception outputs and
applies stabilisation (hysteresis on object location and visibility, smoothing of the step estimate).

**Asynchronous pipeline.** `app/system.py` and `app/perception_runtime.py` run camera capture and
inference continuously (1280x720 at 30 fps on both cameras). Heavier models run on a frame-skip
schedule: step recognition every 2nd frame on a 16-frame clip, object detection every 6th frame,
gesture every 5th frame. A spoken command is **snapshotted** together with the current `W_t` and
queued for the orchestrator worker thread, so the robot reasons over a consistent state while
perception keeps running.

## 3. Orchestration flow

`orchestrator/orchestrator.py` (`HRCOrchestrator`) is a ReAct-style loop. For one command it runs

1. **ANSWER fast path** - unambiguous questions are answered from `W_t` and the SOP knowledge base
   (`orchestrator/answering.py`, `hri/voice_query_handler.py`); no robot motion.
2. **Planning stage** (`_think`) - Layer 1 splits the command into sub-intents; for each EXECUTE
   sub-intent the plan store is queried; Layer 2 and Layer 3 decide what to do with the retrieved
   candidate; if nothing is reusable the LLM planner is called with the world state and the three
   memories as context (`orchestrator/prompts.py`).
3. **Execution stage** (`_tool_execute`) - the validated skill sequence is passed to
   `RobotController`, which parses, validates, plans waypoints and runs them through the selected
   backend; the outcome updates `R_t` and, for completed multi-step plans, is written back to the plan
   memory.

Pointing and pronouns ("that one") are handled before planning by the deictic FSM
(`decision/deictic_grounding.py`, `orchestrator/deictic_dialogue.py`).

## 4. Collaborative decision layers

| | Where | Input | Output |
|---|---|---|---|
| **Layer 1** instruction understanding | `orchestrator/layer1_intent.py` (`IntentMixin`) | utterance | ordered sub-intents: EXECUTE / ANSWER / ASK, with slots (action, object, location, spatial hints) |
| **Layer 2** execution feasibility | `decision/collaboration_policy.py` (`check_global_feasibility`, `check_feasibility`) | `W_t`, commanded target | pass, or REJECT with a reason |
| **Layer 3** execution risk | `decision/collaboration_policy.py` (`decide_risk`, `score_and_select_policy`) | similarity, action match, object match | REUSE / ADAPT / CLARIFY |

**Layer 2** is a hard gate on the world state, not on similarity:

- `H_t`: an explicit human stop signal rejects every candidate;
- `R_t`: a `pickup` plan is rejected while the gripper already holds an object;
- `O_t`: the commanded target must be visible in the current scene.

**Layer 3** applies four explicit rules (`decide_risk`):

| Rule | Condition | Outcome |
|---|---|---|
| 1 | action match and object match and sim >= 0.85 | REUSE |
| 2 | action match and sim >= 0.50 | ADAPT |
| 3 | compatible action pair (pickup/handover, place/handover) and sim >= 0.75 | ADAPT |
| 4 | otherwise | CLARIFY |

The action type of the current command comes from a keyword detector
(`_detect_action_type`); the action type of the retrieved plan is the `task_type` stored with it.

**Retrieval versus decision.** Plan RAG (`database/task_manager.py`, cosine similarity on
`paraphrase-multilingual-MiniLM-L12-v2` embeddings in pgvector) only **recalls candidates**. A high
similarity never executes a plan by itself: the outcome REUSE / ADAPT / CLARIFY / REJECT is produced
by Layer 2 and Layer 3.

## 5. Memory

Three kinds of memory are injected at different stages (`memory/__init__.py` has the details):

| Memory | Module | Used for |
|---|---|---|
| Interaction log | `memory/interaction_log.py` | recent utterances and response types, injected into the planner prompt for follow-ups such as "one more" |
| Plan RAG / episodic memory | `database/task_manager.py` (PostgreSQL + pgvector) | candidate recall of completed multi-step plans |
| Temporal perception log | `memory/temporal.py` | FIFO buffer of scene summaries for questions about recent progress |

`HRCMemorySystem` (`memory/system.py`) is the facade. The SOP knowledge used for procedural
questions is in `memory/sop_step_knowledge.py` and `memory/sop_rich_content.py` and is imported into the
database by `database/import_knowledge.py`.

## 6. Execution and safety

- `execution/skills/skill_schema.py` is the single source of truth for the action ontology; the
  planner prompt and the validators are generated from it.
- `execution/controller/` - `RobotController` with mixins for waypoint generation
  (`grasp_planner.py`), camera / depth helpers (`perception_adaptor.py`) and low-level execution
  (`execution_mixin.py`).
- `execution/backends/` - MoveIt2, TM SetPositions and TMScript behind one interface; selected with
  `EXECUTION_BACKEND` (default `tm_setpos`).
- `execution/safety/` - workspace clutter monitor, failure detection, recovery policy with voice
  guidance, occlusion handling.

## 7. Dependencies per part

| Part | Needs | Entry point |
|---|---|---|
| Decision layers, world-state types, deictic FSM, interaction log, orchestrator prompt assembly | Python + numpy + SQLAlchemy | `pytest` |
| EXP1–EXP4 | numpy | `python -m evaluation.run_all` |
| Perception, step recognition, gesture, robot execution, database, LLM planner | cameras, GPU, ROS 2 / MoveIt2 / TM driver, PostgreSQL + pgvector, OpenAI API key | `python -m ws_racr.app.main`; see [`setup.md`](setup.md) |
