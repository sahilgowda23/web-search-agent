"""Orchestrator: owns the loop. The LLM plans; it never controls execution.

Hard limits (MAX_SEARCH_ITERATIONS, MAX_TIME_BUDGET_SECONDS) are enforced here
regardless of what the planner recommends.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from . import config, extract, planner, search, synthesize
from .memory import IterationRecord, WorkingMemory


@dataclass
class ResearchResult:
    report_markdown: str
    memory: WorkingMemory
    iterations_run: int
    elapsed_seconds: float
    stop_reason: str


def run_research(user_query: str) -> ResearchResult:
    start = time.monotonic()
    memory = WorkingMemory()
    stop_reason = "unknown"

    plan = planner.plan_initial_queries(user_query)
    for subtopic in plan["subtopics"]:
        memory.add_unknown(subtopic)
    query_plan = plan["query_plan"]

    iteration = 0

    while True:
        iteration += 1
        queries = [qp["query"] for qp in query_plan]
        targeted_subtopics = [qp["subtopic"] for qp in query_plan if qp.get("subtopic")]

        results_by_query = search.run_searches(queries)
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
        # so gating only on *current* elapsed time lets total runtime overshoot.
        if elapsed + avg_iteration_cost > config.MAX_TIME_BUDGET_SECONDS:
            stop_reason = "time_budget_would_be_exceeded"
            break

        gap = planner.assess_gaps(user_query, memory, iteration)

        # Explicit unresolved subtopics (facets the user asked for, or facets
        # the planner itself flagged) outweigh the small model's own
        # should_continue/info_gain guess — that guess is unreliable on
        # multi-facet queries, which is exactly when it matters most.
        unresolved = bool(memory.unknown_subtopics)
        if not (gap["should_continue"] or unresolved):
            stop_reason = "planner_satisfied"
            break
        if gap["info_gain_estimate"] < config.MIN_INFO_GAIN_TO_CONTINUE and not unresolved:
            stop_reason = "low_info_gain"
            break

        query_plan = gap["query_plan"]
        if not query_plan and unresolved:
            query_plan = [
                {"subtopic": subtopic, "query": f"{user_query} {subtopic}"}
                for subtopic in memory.unknown_subtopics[: config.MAX_QUERIES_PER_ITERATION]
            ]
        if not query_plan:
            stop_reason = "no_further_queries"
            break

        query_plan = query_plan[: config.MAX_QUERIES_PER_ITERATION]

        elapsed = time.monotonic() - start
        if elapsed >= config.MAX_TIME_BUDGET_SECONDS:
            stop_reason = "time_budget_exceeded"
            break

    report = synthesize.synthesize(user_query, memory)
    elapsed = time.monotonic() - start
    return ResearchResult(
        report_markdown=report,
        memory=memory,
        iterations_run=iteration,
        elapsed_seconds=elapsed,
        stop_reason=stop_reason,
    )
