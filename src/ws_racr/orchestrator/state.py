"""Data structures of the orchestrator: cross-turn agent state, actions, step and result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ws_racr.orchestrator.constants import DIALOGUE_MAX


@dataclass
class AgentState:
    """
    Persistent cross-turn agent state.

    Lives on HRCOrchestrator instance and survives across run() calls,
    enabling context-aware dialogue ("不是那個", "換小的", etc.).

    Fields:
        goal              : high-level intent from the most recent user request
        active_plan       : structured steps of the current multi-step plan
        current_plan_step : index into active_plan (0-based)
        waiting_for_human : True while ASK is pending
        last_failure      : error string from the most recent failed run()
        last_mentioned_object : most recent target_object seen (reference grounding)
        dialogue_context  : ring-buffer of recent (role, text) pairs
        status            : idle | planning | executing | waiting | failed | done
    """
    goal:                   Optional[str]        = None
    active_plan:            List[Dict]           = field(default_factory=list)
    current_plan_step:      int                  = 0
    waiting_for_human:      bool                 = False
    last_failure:           Optional[str]        = None
    last_mentioned_object:  Optional[str]        = None
    dialogue_context:       List[tuple]          = field(default_factory=list)  # (role, text)
    status:                 str                  = "idle"

    def push_dialogue(self, role: str, text: str) -> None:
        """Append a turn; keep at most DIALOGUE_MAX entries."""
        self.dialogue_context.append((role, text))
        if len(self.dialogue_context) > DIALOGUE_MAX:
            self.dialogue_context = self.dialogue_context[-DIALOGUE_MAX:]

    def reset_plan(self) -> None:
        """Clear active plan on new top-level goal."""
        self.active_plan       = []
        self.current_plan_step = 0

    def context_summary(self) -> str:
        """One-paragraph summary injected into LLM prompts."""
        lines: List[str] = []
        if self.last_mentioned_object:
            lines.append(f"Last mentioned object: {self.last_mentioned_object}")
        if self.last_failure:
            lines.append(f"Last failure: {self.last_failure}")
        if self.dialogue_context:
            lines.append("Recent dialogue:")
            for role, text in self.dialogue_context[-4:]:  # show last 4 turns
                lines.append(f"  {role}: {text}")
        return "\n".join(lines) if lines else "(no prior context)"


class OrchestratorAction(str, Enum):
    PLAN    = "PLAN"
    EXECUTE = "EXECUTE"
    ASK     = "ASK"       # 
    CHECK   = "CHECK"     # 
    ANSWER  = "ANSWER"
    DONE    = "DONE"


@dataclass
class TaskStep:
    """One step in an orchestrated plan."""
    index: int
    command: str                    # natural language sub-command
    status: str = "pending"         # pending | executing | success | failed | skipped
    result: Optional[Any] = None    # raw result from execute_task()
    error: Optional[str] = None


@dataclass
class OrchestratorResult:
    """Final result returned by HRCOrchestrator.run()."""
    success: bool
    steps: List[TaskStep] = field(default_factory=list)
    answer_text: Optional[str] = None   # non-empty if the command was a query
    error: Optional[str] = None
    aborted_at: Optional[int] = None    # step index that caused abort
