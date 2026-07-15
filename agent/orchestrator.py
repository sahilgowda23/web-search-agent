"""Orchestrator: owns the loop for a production chat-search grounding step.

This runs inline in a chat request path (target: 5-20s), not as a
standalone research job. Hard limits (MAX_SEARCH_ITERATIONS,
MAX_TIME_BUDGET_SECONDS) are enforced here regardless of what the planner
recommends. Shape:

  1. Gate: one call decides if search is needed at all, and whether the
     query should short-circuit to a clarifying question instead (private
     individual / ungrounded proprietary entity) rather than searching blind.
  2. Round 1: bounded parallel search on the initial query plan.
  3. Round 2 (optional): only if genuinely unresolved subtopics remain and
     the time budget allows finishing it — never more than 2 rounds total.

A subtopic that was actually searched and still didn't resolve is reported
as "no public info found", not silently dropped or retried forever — that
was the earlier failure mode on proprietary/internal-system queries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config, extract, planner, search, synthesize
from .memory import IterationRecord, WorkingMemory


@dataclass
class ResearchResult:
    context: str
    memory: WorkingMemory
    iterations_run: int
    elapsed_seconds: float
    stop_reason: str
    needs_search: bool = True
    clarification_question: str = ""
    searched_but_unresolved: list[str] = field(default_factory=list)


def _empty_result(start: float, stop_reason: str, **kwargs) -> ResearchResult:
    return ResearchResult(
        context="",
        memory=WorkingMemory(),
        iterations_run=0,
        elapsed_seconds=time.monotonic() - start,
        stop_reason=stop_reason,
        **kwargs,
    )


def _run_round(user_query: str, memory: WorkingMemory, query_plan: list[dict], iteration: int) -> list[str]:
    results_by_query = search.run_searches(query_plan)

    # Only offer the extractor subtopics whose query actually came back with
    # something above the relevance threshold — a subtopic with zero
    # grounded results must never be presented as "targeted" or the
    # extraction model will write up whatever's in the pooled snippet block
    # as if it answered it (this is how NSF content became "facts about
    # NECF" before this filter existed).
    targeted_subtopics = [
        qp["subtopic"]
        for qp in query_plan
        if qp.get("subtopic") and results_by_query.get(qp["query"])
    ]
    extract.extract_findings(user_query, results_by_query, memory, targeted_subtopics)

    # Guardrail in extract.py means only targeted subtopics could have
    # moved to known this round, so this diff is exact, not a guess.
    resolved_this_round = [s for s in targeted_subtopics if s in memory.known_subtopics]
    memory.search_log.append(
        IterationRecord(
            iteration=iteration,
            targets=query_plan,
            resolved_subtopics=resolved_this_round,
            still_unknown_after=list(memory.unknown_subtopics),
        )
    )
    return targeted_subtopics


def run_research(user_query: str) -> ResearchResult:
    start = time.monotonic()

    gate = planner.assess_query(user_query)
    if not gate["needs_search"]:
        return _empty_result(start, "no_search_needed", needs_search=False)
    if gate["clarification_needed"]:
        return _empty_result(
            start,
            "clarification_needed",
            clarification_question=gate["clarification_question"] or "Could you clarify what you're asking about?",
        )

    memory = WorkingMemory()
    for subtopic in gate["subtopics"]:
        memory.add_unknown(subtopic)
    query_plan = gate["query_plan"] or [{"subtopic": "", "query": user_query}]

    ever_targeted: set[str] = set()
    iteration = 0
    stop_reason = "unknown"

    while True:
        iteration += 1
        ever_targeted.update(qp["subtopic"] for qp in query_plan if qp.get("subtopic"))
        _run_round(user_query, memory, query_plan, iteration)

        elapsed = time.monotonic() - start
        avg_iteration_cost = elapsed / iteration
        if elapsed >= config.MAX_TIME_BUDGET_SECONDS:
            stop_reason = "time_budget_exceeded"
            break
        if iteration >= config.MAX_SEARCH_ITERATIONS:
            stop_reason = "max_iterations_reached"
            break
        # Predictive check: don't greenlight a round we won't finish within
        # budget — the loop can't interrupt an in-flight search+extract pass,
        # so gating only on *current* elapsed time lets total runtime
        # overshoot. Reserves room for the assess_gaps call this branch is
        # about to make (up to LLM_TIMEOUT_SECONDS), the round itself
        # (avg_iteration_cost), AND the final context-build step — a chain
        # of several calls can blow the budget on latency variance alone
        # even with zero errors, so round 2 is opportunistic, not default.
        if (
            elapsed + config.LLM_TIMEOUT_SECONDS + avg_iteration_cost + config.SYNTHESIS_RESERVE_SECONDS
            > config.MAX_TIME_BUDGET_SECONDS
        ):
            stop_reason = "time_budget_would_be_exceeded"
            break
        if not memory.unknown_subtopics:
            stop_reason = "all_subtopics_resolved"
            break

        gap = planner.assess_gaps(user_query, memory, iteration)
        if not gap["should_continue"] or not gap["query_plan"]:
            stop_reason = "planner_satisfied"
            break

        query_plan = gap["query_plan"][: config.MAX_QUERIES_PER_ITERATION]

    # Subtopics we actually spent a search on but never resolved get reported
    # honestly as gaps, distinct from ones we simply never got to.
    searched_but_unresolved = [s for s in memory.unknown_subtopics if s in ever_targeted]

    time_remaining = config.MAX_TIME_BUDGET_SECONDS - (time.monotonic() - start)
    context = synthesize.build_context(user_query, memory, searched_but_unresolved, time_remaining)
    elapsed = time.monotonic() - start
    return ResearchResult(
        context=context,
        memory=memory,
        iterations_run=iteration,
        elapsed_seconds=elapsed,
        stop_reason=stop_reason,
        searched_but_unresolved=searched_but_unresolved,
    )
