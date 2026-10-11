"""Collaborative decision orchestrator (thesis Ch. 4, "Orchestrator implementation").

:class:`HRCOrchestrator` receives a spoken command together with a frozen world-state snapshot
and runs two stages:

* **planning stage** (``_think``): Layer 1 segments the command into sub-intents; EXECUTE
  intents trigger Plan RAG retrieval, then Layer 2 (feasibility gate) and Layer 3 (risk rules)
  choose REUSE / ADAPT / PLAN / REJECT / CLARIFY; ANSWER intents take the SOP/world-state
  question-answering path.
* **execution stage** (``_tool_execute`` etc.): the resulting skill sequence is handed to the
  robot controller; completed multi-step plans are written back to the plan memory.

Module layout:

``orchestrator.py``       main class and control flow
``layer1_intent.py``      Layer 1 segmentation / classification / slot extraction
``deictic_dialogue.py``   pronoun commands and pointing-target resolution
``plan_memory.py``        Plan RAG retrieval / write-back glue
``answering.py``          ANSWER path (SOP + world-state evidence)
``state.py``              AgentState / OrchestratorResult dataclasses
``prompts.py``            LLM prompt templates
"""
from ws_racr.orchestrator.orchestrator import HRCOrchestrator
from ws_racr.orchestrator.state import AgentState, OrchestratorAction, OrchestratorResult, TaskStep

__all__ = ["AgentState", "HRCOrchestrator", "OrchestratorAction", "OrchestratorResult", "TaskStep"]
