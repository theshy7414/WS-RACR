"""Memory and retrieval layer (thesis Ch. 4, "Memory system and retrieval").

The system keeps three kinds of memory, each injected into the LLM context at a
different stage of the pipeline:

* **Interaction log** (:mod:`ws_racr.memory.interaction_log`) - recent
  user utterances and the response type chosen for each (EXECUTE / ASK / ANSWER).
* **Plan RAG / episodic memory** - successfully completed multi-step plans stored
  in PostgreSQL + pgvector (see :mod:`ws_racr.database.task_manager`). Retrieval
  only *recalls candidates*; whether a candidate is reused is decided by the
  decision layers in :mod:`ws_racr.decision`.
* **Temporal perception log** (:mod:`ws_racr.memory.temporal`) - FIFO buffer of
  scene summaries used when answering questions about recent progress.

In addition: the SOP knowledge base (``sop_*``), the clarification-dialogue store,
a knowledge-graph-style spatial memory, and the :class:`HRCMemorySystem` facade.
"""
from ws_racr.memory.types import (
    EpisodicMemory,
    EventTypeEnum,
    HumanStateEnum,
    MemoryQuery,
    OutcomeEnum,
    SafetyStateEnum,
    SemanticRule,
    StepSummary,
    StepTransitionMemory,
)
from ws_racr.memory.interaction_log import InteractionLog, InteractionRecord
from ws_racr.memory.temporal import TemporalMemory
from ws_racr.memory.spatial import KGNode, KGRelation, NodeType, RelationType, SpatialMemory
from ws_racr.memory.preprocessor import InformationPreprocessor
from ws_racr.memory.system import HRCMemorySystem

__all__ = [
    "EpisodicMemory",
    "EventTypeEnum",
    "HRCMemorySystem",
    "HumanStateEnum",
    "InformationPreprocessor",
    "InteractionLog",
    "InteractionRecord",
    "KGNode",
    "KGRelation",
    "MemoryQuery",
    "NodeType",
    "OutcomeEnum",
    "RelationType",
    "SafetyStateEnum",
    "SemanticRule",
    "SpatialMemory",
    "StepSummary",
    "StepTransitionMemory",
    "TemporalMemory",
]
