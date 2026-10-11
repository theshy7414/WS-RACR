"""Tunable constants of the orchestrator."""

DIALOGUE_MAX = 8   # max turns kept in dialogue_context

# Plan RAG decision thresholds — used inside CollaborationPolicyModule.
#
# The staged pipeline (feasibility gate → reliability score → decision) replaces
# the old single-threshold approach.  CollaborationPolicyModule.score_and_select_policy()
# returns one of: REUSE / ADAPT / CLARIFY / REJECT.
#
# REUSE   (reliability ≥ 0.75): plan is adopted verbatim; LLM-A is skipped.
# ADAPT   (reliability ≥ 0.50): plan injected as LLM-A prompt hint.
# CLARIFY (reliability < 0.50): human confirmation requested before execution.
# REJECT  (feasibility failed):  plan is infeasible; LLM-A plans from scratch.
#
# These constants are kept for reference / backward compatibility with tests.
PLAN_RAG_ADOPT_THRESHOLD = 0.75  # maps to REUSE
PLAN_RAG_HINT_THRESHOLD  = 0.50  # maps to ADAPT

# Deictic resolution waiting windows
DEICTIC_WAIT_POINT_SEC = 4.0
DEICTIC_POLL_SEC = 0.2
