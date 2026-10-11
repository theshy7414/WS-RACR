"""HRC orchestrator: the ReAct-style control loop of the collaboration system (thesis Ch. 4).

Flow::

    human command
          |
    HRCOrchestrator.run()
          |-- ANSWER fast path (questions answered from W_t / SOP, no robot motion)
          |-- planning stage  (_think):  Layer 1 segmentation -> Plan RAG recall ->
          |                              Layer 2 feasibility gate -> Layer 3 risk rules
          |                              (REUSE / ADAPT / CLARIFY / REJECT) -> LLM planner if needed
          +-- execution stage (_tool_execute): sub-command -> RobotController

Tools used by the loop: ``_tool_plan``, ``_tool_execute``, ``_tool_answer`` (TTS),
``_tool_ask`` (clarification question), ``_tool_check`` (read W_t).

The layer logic lives in the mixins (``layer1_intent``, ``plan_memory``, ``deictic_dialogue``,
``answering``) and in ``ws_racr.decision``; this class wires them together.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ws_racr.config import get_model_name, get_openai_client
from ws_racr.decision.collaboration_policy import CollaborationPolicyModule, explain_rag_clarify_zh
from ws_racr.decision.deictic_grounding import DeicticGroundingModule
from ws_racr.orchestrator.answering import AnswerMixin
from ws_racr.orchestrator.constants import DEICTIC_POLL_SEC, DEICTIC_WAIT_POINT_SEC
from ws_racr.orchestrator.deictic_dialogue import DeicticMixin
from ws_racr.orchestrator.layer1_intent import IntentMixin
from ws_racr.orchestrator.plan_memory import PlanMemoryMixin
from ws_racr.orchestrator.prompts import SYSTEM_PROMPT, USER_PROMPT_TMPL
from ws_racr.orchestrator.state import AgentState, OrchestratorResult, TaskStep
from ws_racr.orchestrator.text_utils import normalize_stt_assembly_query
from ws_racr.memory.sop_step_knowledge import STEP_KNOWLEDGE as _STEP_KNOWLEDGE

logger = logging.getLogger(__name__)


class HRCOrchestrator(IntentMixin, DeicticMixin, PlanMemoryMixin, AnswerMixin):
    """
    ReAct-style orchestrator.  Wire it up by passing a RobotController
    instance and (optionally) a TTSHandler instance.

    Usage::

        orch = HRCOrchestrator(controller, tts_handler)
        result = orch.run("先把大螺絲放到右邊，再把小螺絲給我")
    """

    MAX_REACT_STEPS = 15  # guard against infinite loops

    # Default ASK recording duration in seconds.
    ASK_LISTEN_DURATION = 5.0

    def __init__(self, controller, tts_handler=None, listen_fn=None):
        """
        Args:
            controller : RobotController instance
            tts_handler: TTSHandler instance (optional — prints if None)
            listen_fn  : Callable[[], Optional[str]] — blocking STT function.
                         Receives no arguments; returns recognised text or None.
                         If None, _tool_ask() falls back to VoiceRecognizerSD
                         record_and_recognize() if available, else input().
        """
        self.controller  = controller
        self.tts         = tts_handler
        self._listen_fn  = listen_fn
        self._llm_client = None   # lazy init
        self.agent_state = AgentState()  # persistent cross-turn state
        self._ensure_sop_seeded()           # populate sops table on first start
        self._ensure_rich_sop_seeded()      # populate rich chunked SOP entries
        self.grounding = DeicticGroundingModule(
            known_objects=self._get_known_objects(),
            wait_point_sec=DEICTIC_WAIT_POINT_SEC,
            poll_sec=DEICTIC_POLL_SEC,
        )
        self.policy = CollaborationPolicyModule(
            infer_progress_fn=self._infer_progress,
            has_object_available_fn=self._has_object_available,
        )

    # ------------------------------------------------------------------
    # Internal: small NLP helpers
    # ------------------------------------------------------------------

    _YES_MARKERS = ("好", "可以", "要", "是", "對", "照做", "直接", "採用", "沒問題", "恩", "嗯", "y", "yes")
    _NO_MARKERS  = ("不要", "不", "不用", "否", "不是", "改", "重新", "不要照", "不要用", "n", "no")

    def _is_yes(self, text: Optional[str]) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        if any(m in t for m in self._NO_MARKERS):
            return False
        return any(m in t for m in self._YES_MARKERS)

    def _is_no(self, text: Optional[str]) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        if any(m in t for m in self._YES_MARKERS):
            return False
        return any(m in t for m in self._NO_MARKERS)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        world_state=None,
        session_id: Optional[str] = None,
    ) -> OrchestratorResult:
        """
        Execute a (possibly multi-step) user command.

        Args:
            command    : raw natural language from voice / text input
            world_state: WorldState snapshot (from world_state_builder) or None
            session_id : optional session ID forwarded to controller

        Returns:
            OrchestratorResult
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"HRC Orchestrator — 收到指令: {command}")
        logger.info(f"{'='*60}")
        self.grounding.reset_turn()

        sid = session_id or getattr(self.controller, "session_id", None) or "default"
        _cds = getattr(self.controller, "clarification_dialogue_store", None)

        # Consume one pending deictic clarification from previous turn.
        command = self.grounding.consume_pending_if_clarification(
            command,
            is_question_fn=self._is_question,
        )
        _merge = self.grounding.take_last_merge_record()
        if _cds and _merge:
            _cds.append_exchange(
                sid,
                _merge["follow_up"],
                f"（系統合併）前次「{_merge['base']}」+ 本輪「{_merge['follow_up']}」→「{_merge['merged']}」",
            )

        # ── Update AgentState before reasoning ─────────────────────────
        self.agent_state.status = "planning"
        self.agent_state.goal   = command
        self.agent_state.push_dialogue("human", command)
        self.agent_state.waiting_for_human = False

        # ── Multimodal deictic grounding (3 modes) ──────────────────────
        # Supports:
        #   1) point-then-speak
        #   2) speak-then-point (short wait window)
        #   3) speak-then-clarify (second utterance names object)
        # Done BEFORE _think so BOTH planning and execution use the same
        # resolved command string.
        command = self.grounding.resolve(
            command,
            world_state,
            get_live_pointing_target_fn=self._get_live_pointing_target,
            ask_fn=self._tool_ask,
        )

        # Hard stop for unresolved deictic requests:
        # do not continue to RAG/planning/execution without a grounded target.
        if self.grounding.is_unresolved_deictic_command(command):
            if _cds:
                _cds.append_exchange(
                    sid,
                    command,
                    "我還不知道目標物件。你可以指向目標，或直接說出物件名稱。（進入等待澄清）",
                )
            step = TaskStep(
                index=0,
                command=command,
                status="success",
                result={"waiting_for_clarification": True},
            )
            orch_result = OrchestratorResult(success=True, steps=[step])
            self._update_agent_state_after_run(orch_result, sub_commands=[])
            return orch_result

        # ── Layer 0: Pre-dispatch gate ──────────────────────────────────
        # Fast-path for pure single-intent ANSWER commands: skip LLM-A entirely.
        # Only fires when the WHOLE command is a strong question (single segment).
        # Mixed commands (EX+AN, AN+EX, etc.) fall through to _think() so that
        # per-intent routing can handle each sub-intent correctly.
        _quick_intents = self._parse_intents(command)
        _is_pure_answer = (
            len(_quick_intents) == 1
            and _quick_intents[0]["type"] == "ANSWER"
            and self._is_strong_question(command)
        )
        if _is_pure_answer:
            logger.info("Layer 0: 強問句，直接 SOP RAG 回答（跳過 LLM-A）")
            answer_text = self._tool_sop_answer(command, world_state=world_state)
            step = TaskStep(index=0, command=command, status="success",
                            result={"answer_text": answer_text})
            orch_result = OrchestratorResult(
                success=True, steps=[step], answer_text=answer_text
            )
            self._update_agent_state_after_run(orch_result, sub_commands=[])
            return orch_result

        # ── Layer 1: Planning ───────────────────────────────────────────
        # RAG retrieval runs concurrently with context prep inside _think().
        # CONFIRM signals are returned here (not inside _think) so that
        # _think() remains a pure planning function with no interactive side effects.
        skip_rag = self.grounding.has_injected_deictic_target(command)
        plan_type, sub_commands, confirm_signal = self._think(
            command, world_state, skip_rag=skip_rag, session_id=sid
        )

        # ── Handle RAG CLARIFY: ask human before executing ──────────────
        # Triggered when Layer 3 policy is CLARIFY (semantic / action-type uncertainty),
        # not the Layer 2 feasibility gate (that path is REJECT, no confirm here).
        if confirm_signal is not None:
            rag_plan = confirm_signal["rag_plan"]
            n_steps  = confirm_signal["n_steps"]
            clarify_zh = (confirm_signal.get("clarify_explanation_zh") or "").strip()
            inv_hint = self._rag_clarify_inventory_hint(command, world_state)
            src = (rag_plan.get("source_command") or "").strip()
            if len(src) > 24:
                src = src[:24] + "…"
            intro = f"找到舊流程 {n_steps} 步（「{src}」）。"
            parts = [intro, clarify_zh, inv_hint]
            body = "".join(s if not s or s.endswith(("。", "？", "！")) else s + "。" for s in parts if s)
            response = self._tool_ask(f"{body}要沿用嗎？")
            if self._is_yes(response):
                logger.info("   Human confirmed: REUSE historical plan")
                self._tool_answer(
                    f"找到相似且狀態一致的歷史計畫，直接採用，共 {n_steps} 步。",
                    blocking=True,
                    priority=4,
                )
                plan_type = "multi"
                _intents_confirm = self._parse_intents(command, world_state)
                sub_commands = self._inject_spatial_hints_from_intents(
                    _intents_confirm,
                    [dict(s) for s in rag_plan["sub_commands"]],
                )
            else:
                logger.info("   Human rejected / unclear: plan with LLM-A normally")
                plan_type, sub_commands, _ = self._think(
                    command, world_state, skip_rag=True, session_id=sid
                )

        if plan_type == "single":
            sub_commands = self._sanitize_llm_single_execute(
                command, world_state, sub_commands
            )
            # If LLM-A identified this as a pure question, use SOP-grounded answer.
            first_action = (sub_commands[0].get("action") if sub_commands else None)
            if first_action == "answer" or (not sub_commands and self._is_question(command)):
                logger.info("Thought: 問答查詢，SOP RAG 回答")
                query = (sub_commands[0].get("query", command) if sub_commands else command)
                answer_text = self._tool_sop_answer(query, world_state=world_state)
                step = TaskStep(index=0, command=command, status="success",
                                result={"answer_text": answer_text})
                orch_result = OrchestratorResult(
                    success=True, steps=[step], answer_text=answer_text
                )
                self._update_agent_state_after_run(orch_result, sub_commands=[])
                return orch_result

            logger.info("Thought: 單一動作，直接執行")
            step = TaskStep(index=0, command=command)
            self.agent_state.status = "executing"
            # Use structured sub_command when available so spatial_modifier,
            # object_size and other parsed fields are preserved without
            # re-parsing the NL string through llm_parser again.
            single_step = sub_commands[0] if sub_commands else command
            result_data = self._tool_execute(single_step, session_id)
            orch_result = self._finalize_single(step, result_data)
            self._update_agent_state_after_run(orch_result, sub_commands=[])
            return orch_result

        # MULTI path — sub_commands is List[Dict] (structured steps)
        logger.info(f"Thought: 多步任務，共 {len(sub_commands)} 步")
        for i, sc in enumerate(sub_commands, 1):
            label = sc.get("action", "?") if isinstance(sc, dict) else sc
            obj = sc.get("target_object", "") if isinstance(sc, dict) else ""
            loc = sc.get("target_location", "") if isinstance(sc, dict) else ""
            detail = " ".join(filter(None, [obj, f"→{loc}" if loc else ""]))
            logger.info(f"   {i}. {label} {detail}".rstrip())

        steps: List[TaskStep] = [
            TaskStep(index=i, command=sc if isinstance(sc, str) else sc.get("action", str(sc)))
            for i, sc in enumerate(sub_commands)
        ]
        # Keep the original structured step alongside for execution
        structured_steps = sub_commands

        # ── Act: execute plan steps one by one ─────────────────────────
        for step, raw_step in zip(steps, structured_steps):
            action = raw_step.get("action", "") if isinstance(raw_step, dict) else ""
            obj = raw_step.get("target_object", "") if isinstance(raw_step, dict) else ""
            loc = raw_step.get("target_location", "") if isinstance(raw_step, dict) else ""
            detail = " ".join(filter(None, [obj, f"→{loc}" if loc else ""]))
            label = action or step.command
            display = f"{label} {detail}".rstrip()
            logger.info(f"\n▶ 步驟 {step.index + 1}/{len(steps)}: {display}")

            # ── Interaction step: answer ────────────────────────────────
            if action == "answer":
                query_text = raw_step.get("query", "") or step.command
                if query_text:
                    logger.info(f"   SOP 查詢: {query_text}")
                step.status = "executing"
                # Route through SOP-grounded answer generation (RAG ③)
                answer_text = self._tool_sop_answer(query_text, world_state=world_state)
                step.result = {"answer_text": answer_text}
                step.status = "success"
                logger.info(f"   查詢完成")
                continue

            # ── Interaction step: ask ───────────────────────────────────
            if action == "ask":
                question = raw_step.get("question", "請問需要什麼協助？")
                # Avoid repeating the same object-clarification prompt in one run.
                if self.grounding.asked_object_clarification_this_run and any(
                    k in question for k in ("哪個物品", "目標物件", "哪一個")
                ):
                    logger.info("   [ASK] object clarification already asked this run; skip duplicate prompt")
                    step.status = "success"
                    step.result = None
                    continue
                response = self._tool_ask(question)
                if response:
                    self.agent_state.push_dialogue("human", response)
                    logger.info(f"   取得回應: {response}")
                step.status = "success"
                step.result = response
                continue

            # ── Robot skill step ────────────────────────────────────────
            # TTS：priority 越小越先播。勿用 3（高於一般回答 5），否則會插隊蓋過上一步的 answer TTS
            self._tool_answer(
                f"執行第{step.index + 1}步：{display}",
                blocking=True,
                priority=6,
            )

            step.status = "executing"
            result_data = self._tool_execute(raw_step, session_id)
            step.result = result_data

            # ── Observe & decide ────────────────────────────────────────
            if result_data is True:
                step.status = "success"
                logger.info(f"   完成")

            elif result_data is False:
                step.status = "failed"
                step.error = "execute_task returned False"
                logger.error(f"   步驟失敗，中止序列")
                self._tool_answer(
                    f"第{step.index + 1}步執行失敗，任務中止",
                    blocking=True,
                    priority=5,
                )
                fail_result = OrchestratorResult(
                    success=False,
                    steps=steps,
                    error=step.error,
                    aborted_at=step.index,
                )
                self._update_agent_state_after_run(fail_result, sub_commands=structured_steps)
                return fail_result

            elif isinstance(result_data, dict) and result_data.get("feasibility_abort"):
                msg = (result_data.get("speak_zh") or "目前無法執行。").strip()
                step.status = "failed"
                step.error = msg
                logger.error(f"   步驟失敗（可行性），中止序列")
                self._tool_answer(msg, blocking=True, priority=5)
                fail_result = OrchestratorResult(
                    success=False,
                    steps=steps,
                    error=msg,
                    aborted_at=step.index,
                )
                self._update_agent_state_after_run(fail_result, sub_commands=structured_steps)
                return fail_result

            elif isinstance(result_data, dict):
                handled = self._handle_policy(step, result_data, steps)
                if not handled:
                    # unhandled policy → abort
                    return OrchestratorResult(
                        success=False,
                        steps=steps,
                        error=f"Unhandled policy at step {step.index}",
                        aborted_at=step.index,
                    )
                # if the policy produced an answer (query), speak and continue
                # (do NOT stop early — there may be more steps in the sequence)
                if result_data.get("policy_answer"):
                    step.status = "success"
                    continue

        # ── All steps done ──────────────────────────────────────────────
        completed = sum(1 for s in steps if s.status == "success")
        self._tool_answer(
            f"所有 {completed} 個步驟已完成",
            blocking=True,
            priority=5,
        )
        final_result = OrchestratorResult(success=True, steps=steps)
        self._update_agent_state_after_run(final_result, sub_commands=structured_steps)

        # Persist only meaningful multi-step robot skill plans.
        # Skip: unresolved deictic, pure questions, plans with no robot steps.
        robot_steps = [s for s in structured_steps
                       if isinstance(s, dict) and s.get("action") not in ("answer", "ask", None)]
        if self.grounding.is_unresolved_deictic_command(command):
            logger.info("   ℹPlan RAG: skip — unresolved deictic command")
        elif self._is_question(command):
            logger.info("   ℹPlan RAG: skip — pure question, not a robot skill plan")
        elif not robot_steps:
            logger.info("   ℹPlan RAG: skip — no robot skill steps to remember")
        else:
            self._store_multi_plan(command, structured_steps, session_id, world_state=world_state)

        return final_result

    # ------------------------------------------------------------------
    # Internal: Deictic grounding pre-processor
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Internal: Plan RAG helpers
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Internal: SOP knowledge helpers  (RAG ③)
    # ------------------------------------------------------------------


    # ── Intent parsing ────────────────────────────────────────────────────────


    # ------------------------------------------------------------------
    # Internal: Answer pipeline (3-layer)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Internal: Thought step
    # ------------------------------------------------------------------

    def _think(
        self,
        command: str,
        world_state,
        skip_rag: bool = False,
        session_id: Optional[str] = None,
    ) -> tuple[str, List[str], Optional[Dict]]:
        """
        Parallel pre-flight + LLM planning.  Pure function: no side effects.

        All independent context sources are fetched concurrently (4 threads):
          • RAG plan retrieval  (vector store, ~50-150 ms)
          • Interaction history (memory system)
          • Semantic rules      (memory system)
          • SOP context         (keyword search, only for questions)

        After the parallel phase, RAG scoring + REUSE/CONFIRM/REFINE decisions
        are made.  REUSE short-circuits the LLM call entirely.  REFINE injects
        the retrieved plan as a prompt hint.  Everything else calls LLM-A.

        CONFIRM is NOT handled here: a signal dict is returned so that run()
        can issue the _tool_ask() call (keeping _think side-effect-free).

        Returns:
            (plan_type, sub_commands, confirm_signal)
            plan_type      : "single" | "multi"
            sub_commands   : list[Dict] (structured steps, may be [] for single)
            confirm_signal : None, or for RAG CLARIFY:
                             {"rag_plan", "n_steps", "clarify_explanation_zh"} so
                             run() can ask the human with concrete reasons.
        """
        context_str = self._format_world_context(world_state)
        agent_ctx   = self.agent_state.context_summary()

        # Pointing target — already resolved by perception; zero extra cost
        pointing_context = ""
        try:
            pt = getattr(getattr(world_state, "human", None), "pointing_target", None)
            if pt:
                pointing_context = f"\n  pointing_target: {pt}  # human is pointing at this object"
        except Exception:
            pass

        # ── Step 0: Parse intents ─────────────────────────────────────
        # Replaces utterance-level classification with per-intent segmentation.
        # A single-segment command yields one intent; multi-segment yields N.
        intents = self._parse_intents(command, world_state)
        is_question = any(i["type"] == "ANSWER" for i in intents)
        has_execute = any(i["type"] == "EXECUTE" for i in intents)
        has_ask     = any(i["type"] == "ASK"     for i in intents)

        # Collect per-type query texts for targeted retrieval
        execute_spans = [i["span_text"] for i in intents if i["type"] == "EXECUTE"]
        answer_spans  = [i["span_text"] for i in intents if i["type"] == "ANSWER"]

        # Use the first EXECUTE span for plan retrieval (most representative)
        execute_query = execute_spans[0] if execute_spans else command
        # Concatenate ANSWER spans for SOP retrieval
        answer_query = (
            normalize_stt_assembly_query(" ".join(answer_spans))
            if answer_spans
            else normalize_stt_assembly_query(command)
        )

        logger.info(f"[Intent-Parse] segments={len(intents)} "
            f"EX={len(execute_spans)} AN={len(answer_spans)} "
            f"ASK={sum(1 for i in intents if i['type']=='ASK')}")

        # ── Step 1: Parallel per-intent retrieval ────────────────────
        def _fetch_plan_rag() -> Optional[List[Dict]]:
            """Retrieve plan candidates only for EXECUTE intents, using the execute span."""
            if skip_rag or not has_execute:
                return None
            return self._retrieve_multi_plan(execute_query, world_state=world_state)

        def _fetch_sop_rag() -> Optional[str]:
            """Retrieve SOP knowledge only for ANSWER intents."""
            if not is_question:
                return None
            try:
                return self._search_sop(answer_query)
            except Exception:
                return None

        def _build_interaction_history() -> str:
            try:
                ms = getattr(self.controller, "memory_system", None)
                if ms is not None:
                    log = getattr(ms, "interaction_log", None)
                    if log is not None:
                        hist = log.to_context_text(n=5)
                        if hist:
                            return f"\n{hist}\n"
            except Exception:
                pass
            return ""

        def _build_semantic_rules() -> str:
            try:
                ms = getattr(self.controller, "memory_system", None)
                if ms is not None:
                    rules = getattr(ms, "semantic_rules", {})
                    high_conf = [r for r in rules.values() if getattr(r, "confidence", 0) >= 0.6]
                    if high_conf:
                        lines = ["=== Learned SOP Rules ==="]
                        for r in high_conf[:5]:
                            lines.append(f"- {r.rule_text} (conf: {r.confidence:.2f})")
                        return "\n" + "\n".join(lines) + "\n"
            except Exception:
                pass
            return ""

        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_plan  = pool.submit(_fetch_plan_rag)
            fut_sop   = pool.submit(_fetch_sop_rag)
            fut_hist  = pool.submit(_build_interaction_history)
            fut_rules = pool.submit(_build_semantic_rules)
            rag_candidates      = fut_plan.result()   # List[Dict] | None
            sop_ctx_str         = fut_sop.result() or ""
            interaction_history = fut_hist.result()
            semantic_rules      = fut_rules.result()

        # ── Step 2: Per-intent decision ───────────────────────────────
        # Decision taxonomy:
        #   EXECUTE → DIRECT | REUSE | ADAPT | CLARIFY
        #   ANSWER  → DIRECT | SOP   | CLARIFY
        #   ASK     → CLARIFY (system must ask human)
        rag_hint: Optional[Dict] = None
        rag_plan: Optional[Dict] = None   # the selected viable candidate (set below)

        if rag_candidates and has_execute:
            # Extract current-command properties used for global Layer 2 checks
            cur_target_obj = next(
                (i.get("target_object") for i in intents if i["type"] == "EXECUTE" and i.get("target_object")),
                None,
            )
            from ws_racr.decision.collaboration_policy import _detect_action_type  # noqa: PLC0415
            cur_action_type = _detect_action_type(execute_query) if execute_query else "unknown"

            # ── Layer 2a: Global checks (H_t, R_t, current target O_t) ──────
            # These apply to every candidate; if they fail no retry helps.
            global_ok, global_reason = self.policy.check_global_feasibility(
                world_state=world_state,
                current_action_type=cur_action_type,
                current_target_object=cur_target_obj,
            )

            if not global_ok:
                rag_policy = "REJECT"
                score = {
                    "sim": rag_candidates[0]["similarity"],
                    "policy": "REJECT",
                    "feasible": False,
                    "feasibility_reason": global_reason,
                }
                logger.info(f"[RAG-Decision/EXECUTE] REJECT (global) — {global_reason} "
                    f"(sim={rag_candidates[0]['similarity']:.2f}) → LLM-A plans fresh")
            else:
                # ── Layer 3: Risk decision (top-1 candidate) ─────────────────
                # Per-candidate O_t filtering removed: object mismatches are
                # handled by Layer 3's om signal (ADAPT / PLAN), not rejected here.
                rag_plan = rag_candidates[0]
                score = self.policy.score_and_select_policy(
                    sim=rag_plan["similarity"],
                    retrieved_step_index=rag_plan.get("step_index"),
                    world_state=world_state,
                    current_command=execute_query,
                    retrieved_command=rag_plan.get("source_command") or "",
                    retrieved_task_type=rag_plan.get("task_type"),
                    use_feasibility_gate=False,  # already cleared by Layer 2a above
                )
                rag_policy = score["policy"]
                tail = (
                    f"am={score.get('action_match')} om={score.get('object_match')} "
                    f"{score.get('cur_type', '?')}→{score.get('ret_type', '?')} → {rag_policy}"
                )
                logger.info("[RAG-Decision/EXECUTE] "
                    f"query=「{execute_query[:20]}」 "
                    f"sim={score['sim']:.2f} {tail}")

            if rag_policy == "REJECT":
                # Layer 2 global gate blocked (H_t/R_t/target O_t), or
                # Layer 3 decided REJECT. Fall through to LLM-A.
                pass

            elif rag_policy == "REUSE" and not is_question and not has_ask:
                # Pure EXECUTE command: REUSE short-circuits LLM-A entirely.
                # Mixed commands (with ANSWER/ASK intents) still need LLM-A to
                # handle the non-EXECUTE parts.
                n_steps = len(rag_plan["sub_commands"])
                logger.info(f"Plan RAG: REUSE 歷史計畫 "
                    f"(query=「{execute_query[:20]}」, {n_steps} 步, "
                    f"原始指令: 「{rag_plan['source_command']}」)")
                self._tool_answer(
                    f"找到相似且狀態一致的歷史計畫，直接採用，共 {n_steps} 步。",
                    blocking=True,
                    priority=4,
                )
                reused = [dict(s) for s in rag_plan["sub_commands"]]
                return (
                    "multi",
                    self._inject_spatial_hints_from_intents(intents, reused),
                    None,
                )

            elif rag_policy == "CLARIFY" and not is_question and not has_ask:
                # Feasibility passed but Layer 3 risk rules → ask human before executing.
                n_steps = len(rag_plan["sub_commands"])
                logger.info("[RAG-Decision/EXECUTE] CLARIFY → 回傳 confirm_signal 給 run()")
                return "pending", [], {
                    "rag_plan": rag_plan,
                    "n_steps": n_steps,
                    "clarify_explanation_zh": explain_rag_clarify_zh(score),
                }

            elif rag_policy == "ADAPT":
                rag_hint = rag_plan  # inject as prompt hint for LLM-A

        # ── Build LLM-A prompt ────────────────────────────────────────
        rag_hint_str = ""
        if rag_hint and rag_hint.get("sub_commands"):
            steps_str = json.dumps(rag_hint["sub_commands"], ensure_ascii=False)
            rag_hint_str = (
                f"\n=== 歷史相似計畫（相似度: {rag_hint['similarity']:.2f}，"
                f"原始指令: 「{rag_hint['source_command']}」）===\n"
                f"建議步驟: {steps_str}\n"
                f"（可直接採用，或根據當前情境調整物件名稱）\n"
            )

        # Build intent-parse context for LLM-A (replaces utterance-level hint)
        intent_ctx_str = ""
        if len(intents) > 1:
            lines = ["=== 指令解析結果（Sub-intents）==="]
            for idx, it in enumerate(intents, 1):
                itype = it["type"]
                span  = it["span_text"]
                if itype == "EXECUTE":
                    act = it.get("action") or "?"
                    obj = it.get("target_object") or "?"
                    loc = it.get("target_location") or ""
                    sm = it.get("spatial_modifier") or ""
                    oz = it.get("object_size") or ""
                    extra = ""
                    if sm:
                        extra += f" spatial={sm}"
                    if oz:
                        extra += f" size={oz}"
                    lines.append(
                        f"{idx}. EXECUTE: {span} → action={act} obj={obj}"
                        + (f" loc={loc}" if loc else "")
                        + extra
                    )
                elif itype == "ANSWER":
                    lines.append(f"{idx}. ANSWER: {span}")
                else:
                    lines.append(f"{idx}. ASK: {span}")
            intent_ctx_str = "\n".join(lines) + "\n"

        clar_mem = ""
        _store = getattr(self.controller, "clarification_dialogue_store", None)
        if _store:
            _sid = session_id or getattr(self.controller, "session_id", None) or "default"
            clar_mem = _store.get_context_block(_sid)

        user_msg = USER_PROMPT_TMPL.format(
            step=context_str["step"],
            objects=context_str["objects"],
            pointing_context=pointing_context,
            gripper_state=context_str.get("gripper_state", "unknown"),
            human_state=context_str.get("human_state", ""),
            agent_context=agent_ctx,
            interaction_history=interaction_history,
            semantic_rules=semantic_rules,
            rag_hint=rag_hint_str + intent_ctx_str,
            clarification_memory=clar_mem,
            sop_context=sop_ctx_str,
            command=command,
        )

        try:
            client = self._get_llm_client()
            resp = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            data = json.loads(raw)
        except Exception as e:
            logger.warning("Orchestrator LLM call failed: %s — fallback to single", e)
            return "single", [], None

        plan_type = data.get("type", "single")
        sub_commands: List[Dict[str, Any]] = data.get("sub_commands", [])

        if plan_type == "multi" and not sub_commands:
            logger.warning("LLM returned multi but no sub_commands — fallback to single")
            return "single", [], None

        sub_commands = self._inject_spatial_hints_from_intents(intents, sub_commands)
        return plan_type, sub_commands, None

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _tool_execute(
        self,
        sub_command: Any,           # Dict (structured) from MULTI path
        session_id: Optional[str],  # or str (raw NL) from SINGLE/voice path
    ) -> Any:
        """
        Execute one step.

        MULTI path: sub_command is a structured Dict from _think() LLM output.
            → calls execute_task_from_structured(), bypassing parser entirely.

        SINGLE/voice path: sub_command is the original raw NL string from the user.
            → falls back to process_user_command → parser → execute_task.

        Returns the raw result from execute_task (True / False / Dict).
        """
        # ── Execution-time W_t re-check ──────────────────────────────────────
        # Re-query live world state (may differ from planning-time snapshot).
        if isinstance(sub_command, dict):
            action     = sub_command.get("action", "")
            target_obj = sub_command.get("target_object")
            wsb = getattr(self.controller, "world_state_builder", None)
            live_ws = wsb.get_current_state() if wsb else None

            if live_ws and action in ("pickup", "place") and target_obj:
                if not self._has_object_available(live_ws, target_obj):
                    logger.warning(
                        "Execution-time feasibility abort: %s unavailable for %s",
                        target_obj, action,
                    )
                    return {
                        "feasibility_abort": True,
                        "reason": f"object_unavailable_at_execution:{target_obj}",
                    }

            if live_ws and action == "handover":
                robot = getattr(live_ws, "robot", None)
                holding = getattr(robot, "holding_object_id", None) if robot else None
                gripper = getattr(robot, "gripper_state", None) if robot else None
                # Hard block only when confirmed empty (open + no holding record)
                if gripper == "open" and holding is None:
                    logger.warning("Execution-time feasibility abort: gripper empty for handover")
                    return {
                        "feasibility_abort": True,
                        "reason": "gripper_empty_cannot_handover_at_execution",
                    }

        try:
            if isinstance(sub_command, dict):
                # Structured path — bypass parser
                if hasattr(self.controller, "execute_task_from_structured"):
                    return self.controller.execute_task_from_structured(
                        sub_command, session_id=session_id
                    )
                # Fallback: controller doesn't have the new method yet (tests / mock)
                logger.warning("controller has no execute_task_from_structured — using mock")
                return True

            # Natural-language fallback (SINGLE path or voice raw input)
            task_id = self.controller.process_user_command(
                sub_command, session_id=session_id
            )
            if task_id is None:
                logger.error("process_user_command returned None for: %s", sub_command)
                return False
            return self.controller.execute_task(task_id)
        except Exception as e:
            logger.error("_tool_execute exception: %s", e)
            return False

    def _tool_answer(self, text: str, blocking: bool = False, priority: int = 7):
        """Speak text via TTS (or print if TTS unavailable).

        blocking=True: 等 TTSHandler 佇列與當前播放結束再返回（播完再動／問句播完）。
        """
        logger.info(f"{text}")
        if self.tts is not None:
            try:
                self.tts.speak(text, priority=priority)
                if blocking and hasattr(self.tts, "wait_until_idle"):
                    if not self.tts.wait_until_idle(timeout=120.0):
                        logger.warning("TTS wait_until_idle timed out (120s)")
            except Exception as e:
                logger.warning("TTS speak failed: %s", e)

    def _tool_ask(self, question: str) -> Optional[str]:
        """
        Ask human a clarifying question.

        IMPORTANT UX rule:
        - Never auto-open microphone here.
        - Voice input must be user-initiated (push-to-talk, e.g., press V).

        Therefore this method only speaks the question and returns None.
        The next user-triggered utterance will be processed by the normal
        command pipeline (and can be consumed by pending clarification logic).
        """
        self._tool_answer(question, blocking=True, priority=5)
        logger.info("   [ASK] 等待使用者手動按 V 後再說話（不自動開麥）")
        return None

    def _tool_check(self, attribute: str, world_state) -> str:
        """
        Query a world state attribute with structured interpretation.

        Supports dotted paths (e.g. 'assembly_progress.current_step')
        and a small set of semantic shortcuts:
          gripper  → robot gripper state (open/closed/holding)
          step     → current assembly step name + zh description
          objects  → list of visible objects on table
        """
        if world_state is None:
            return "world_state unavailable"

        # --- semantic shortcuts ---
        if attribute == "gripper":
            try:
                robot = getattr(world_state, "robot", None)
                gs = getattr(robot, "gripper_state", None) or getattr(world_state, "gripper_state", None)
                return str(gs) if gs else "gripper_state: unknown"
            except Exception:
                return "gripper_state: unknown"

        if attribute == "step":
            ctx = self._format_world_context(world_state)
            return ctx["step"]

        if attribute == "objects":
            ctx = self._format_world_context(world_state)
            return ctx["objects"]

        # --- dotted path traversal ---
        parts = attribute.split(".")
        obj = world_state
        for part in parts:
            try:
                obj = getattr(obj, part, None)
                if obj is None:
                    return f"{attribute}: None"
            except Exception:
                return f"{attribute}: error"
        return str(obj)

    # ------------------------------------------------------------------
    # Policy handlers
    # ------------------------------------------------------------------

    def _handle_policy(self, step: TaskStep, result: Dict, all_steps: List[TaskStep]) -> bool:
        """
        Handle dict result signals from execute_task.

        Returns True if handled (caller may continue), False if unrecoverable.
        """
        if result.get("policy_answer"):
            answer_text = result.get("answer_text", "")
            self._tool_answer(answer_text, blocking=True, priority=7)
            step.status = "success"
            return True

        if result.get("policy_wait"):
            reason = result.get("reason", "正在等待")
            logger.info(f"    WAIT: {reason}")
            self._tool_answer(f"稍等一下，{reason}", blocking=True, priority=5)
            step.status = "skipped"
            return True

        if result.get("policy_ask_human"):
            question = result.get("question", "請問需要什麼協助？")
            response = self._tool_ask(question)
            step.status = "skipped"
            step.error = "needs_clarification"
            return True

        if result.get("occlusion_wait"):
            explanation = result.get("explanation", "物件被遮擋")
            self._tool_answer(
                f"⚠️ {explanation}，請移開遮擋物後再試",
                blocking=True,
                priority=5,
            )
            step.status = "failed"
            step.error = explanation
            return False  # abort the sequence; the user must clear the occluder first

        if result.get("type") == "occlusion_removal_sequence":
            # multi-task occlusion removal is not handled here: ask the user to clear the occluder
            self._tool_answer(
                "偵測到遮擋，請協助移開遮擋物",
                blocking=True,
                priority=5,
            )
            step.status = "failed"
            step.error = "occlusion_removal_sequence"
            return False

        # Unknown dict signal — log and treat as failure
        logger.warning("Unknown result dict from execute_task: %s", result)
        step.status = "failed"
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finalize_single(self, step: TaskStep, result_data: Any) -> OrchestratorResult:
        """Wrap single-step execution into OrchestratorResult."""
        step.result = result_data

        if result_data is True:
            step.status = "success"
            return OrchestratorResult(success=True, steps=[step])

        if result_data is False:
            step.status = "failed"
            step.error = "execute_task returned False"
            return OrchestratorResult(success=False, steps=[step], error=step.error, aborted_at=0)

        if isinstance(result_data, dict) and result_data.get("feasibility_abort"):
            msg = (result_data.get("speak_zh") or "目前無法執行。").strip()
            step.status = "failed"
            step.error = msg
            self._tool_answer(msg, blocking=True, priority=5)
            return OrchestratorResult(success=False, steps=[step], error=msg, aborted_at=0)

        if isinstance(result_data, dict):
            self._handle_policy(step, result_data, [step])
            answer_text = result_data.get("answer_text") if result_data.get("policy_answer") else None
            return OrchestratorResult(
                success=(step.status in ("success", "skipped")),
                steps=[step],
                answer_text=answer_text,
            )

        # fallback
        step.status = "failed"
        return OrchestratorResult(success=False, steps=[step], error=f"unexpected result: {result_data!r}")

    def _format_world_context(self, world_state) -> Dict[str, str]:
        """Extract step, objects, robot state and human state from WorldState."""
        step_str = "unknown"
        objects_str = "unknown"
        gripper_str = "unknown"
        human_str = ""

        if world_state is None:
            return {"step": step_str, "objects": objects_str,
                    "gripper_state": gripper_str, "human_state": human_str}

        try:
            # step from assembly_progress
            progress = getattr(world_state, "assembly_progress", None)
            if progress:
                current_step = getattr(progress, "current_step", None)
                if current_step:
                    step_knowledge = _STEP_KNOWLEDGE.get(current_step, {})
                    step_str = f"{current_step} ({step_knowledge.get('zh', '')})"

            # visible objects — strip _N instance suffix and deduplicate
            # so LLM sees "screw_big, cylinder" not "screw_big_1, cylinder_2"
            objects_on_table = getattr(world_state, "objects", {})
            if objects_on_table:
                seen_classes: set = set()
                canonical_visible: list = []
                for obj_id, obj_state in objects_on_table.items():
                    if getattr(obj_state, "visibility", "visible") == "visible":
                        cls_name = re.sub(r'_\d+$', '', obj_id)
                        if cls_name not in seen_classes:
                            seen_classes.add(cls_name)
                            canonical_visible.append(cls_name)
                objects_str = ", ".join(canonical_visible) if canonical_visible else "none visible"

            # R_t: gripper + holding object
            robot = getattr(world_state, "robot", None)
            if robot:
                gs = getattr(robot, "gripper_state", None)
                holding = getattr(robot, "holding_object_id", None)
                if gs:
                    gripper_str = gs
                    if holding:
                        gripper_str += f" (holding: {holding})"
                elif holding:
                    gripper_str = f"holding: {holding}"

            # H_t: interaction state (beyond pointing_target, which is separate)
            human = getattr(world_state, "human", None)
            if human:
                parts = []
                interaction = getattr(human, "interaction_state", None)
                if interaction and interaction not in ("idle", "none", ""):
                    parts.append(f"interaction={interaction}")
                intent = getattr(human, "intent_stable", None)
                if intent and intent not in ("idle", "none", ""):
                    parts.append(f"intent={intent}")
                if parts:
                    human_str = "\n  human   : " + ", ".join(parts)

        except Exception as e:
            logger.debug("_format_world_context error: %s", e)

        return {
            "step": step_str,
            "objects": objects_str,
            "gripper_state": gripper_str,
            "human_state": human_str,
        }

    def _get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_openai_client()
        return self._llm_client

    # ------------------------------------------------------------------
    # AgentState lifecycle
    # ------------------------------------------------------------------

    def _update_agent_state_after_run(
        self,
        result: OrchestratorResult,
        sub_commands: List[Any],
    ) -> None:
        """
        Called at the end of every run() path to keep agent_state in sync.

        - Extracts the last target_object from executed steps for reference grounding.
        - Records robot reply in dialogue_context.
        - Sets status and last_failure.
        - Stores the structured plan for multi-step runs.
        """
        if result.success:
            self.agent_state.status       = "done"
            self.agent_state.last_failure = None
        else:
            self.agent_state.status       = "failed"
            self.agent_state.last_failure = result.error

        # Extract last mentioned object from structured steps (MULTI path)
        for sc in reversed(sub_commands):
            if isinstance(sc, dict):
                obj = sc.get("target_object")
                if obj:
                    self.agent_state.last_mentioned_object = obj
                    break

        # Also scan step results for object info (SINGLE path)
        if not self.agent_state.last_mentioned_object:
            for step in reversed(result.steps):
                if step.command and step.command not in ("unknown", ""):
                    # command is action string; try to extract from it
                    break  # heuristic only; structured path covers the main case

        # Record robot reply in dialogue context
        if result.success:
            reply = f"執行成功: {self.agent_state.goal or ''}"
        else:
            reply = f"執行失敗: {result.error or ''}"
        self.agent_state.push_dialogue("robot", reply)

        # Store plan for potential replan
        if sub_commands:
            self.agent_state.active_plan       = list(sub_commands)
            self.agent_state.current_plan_step = len(result.steps)
