"""Planner: proposes searches and estimates information gain.

The planner never controls the loop — it only advises. The orchestrator
enforces MAX_SEARCH_ITERATIONS / MAX_TIME_BUDGET_SECONDS regardless of
what the planner returns, and forces continuation while unresolved
subtopics remain (see orchestrator.py) rather than trusting the small
model's should_continue verdict alone.

Each search query is paired with the single subtopic it targets. This is
a deterministic guardrail: extract.py is only allowed to mark a subtopic
"known" if this iteration actually ran a query for it — regardless of what
the (unreliable, small) extraction model claims was "covered". Without this,
a small model tends to mark most/all outstanding subtopics as resolved after
any search, even ones untouched by that round's queries.
"""
from __future__ import annotations

from . import config
from .llm import call_json
from .memory import WorkingMemory

_INITIAL_SYSTEM = """You plan web research for an agent that decomposes a topic
before searching. Given a user's research query:

1. Identify the distinct facets/subtopics the query asks about. If the user
   explicitly lists aspects (e.g. "architecture, adopters, SDK support,
   authentication, security concerns"), each one is its own subtopic — do not
   merge them. If the query is a simple topic with no listed facets, infer
   3-5 natural facets (e.g. definition, how it works, use cases).
2. Propose up to 3 focused search queries covering the highest-priority
   subtopics first (later subtopics get covered in later iterations). Each
   query targets exactly one subtopic.

Return strict JSON only, matching this shape exactly:
{
  "subtopics": ["...", "...", ...],
  "query_plan": [{"subtopic": "...", "query": "..."}, ...]
}
Every "subtopic" value in query_plan must be copied verbatim from "subtopics"."""

_GAP_SYSTEM = """You are the planning step of an iterative research agent.
Subtopic resolution already happened during extraction, so you're only told
which subtopics are still unknown_subtopics — decide what to search next.
Return strict JSON only, matching this shape exactly:
{
  "info_gain_estimate": 0.0-1.0,
  "should_continue": true/false,
  "query_plan": [{"subtopic": "...", "query": "..."}, ...],
  "reasoning": "one short sentence"
}
Guidance:
- If unknown_subtopics is non-empty, you should almost always continue — each
  one is a facet the user explicitly wants covered and hasn't been yet.
- Only set should_continue false when unknown_subtopics is empty, or fact_count
  is already large relative to remaining subtopics (diminishing returns).
- query_plan must have one entry per unknown_subtopic you're targeting this
  round (subtopic copied verbatim from unknown_subtopics), and must not repeat
  queries_already_executed."""


def _to_str_list(items) -> list[str]:
    """Coerce a list that may contain strings or dicts like {"query": "..."}."""
    out = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(
                item.get("query") or item.get("text") or item.get("topic")
                or item.get("name") or next(iter(item.values()), "")
            ).strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def _to_query_plan(items) -> list[dict]:
    """Coerce planner output into a list of {"subtopic", "query"} dicts."""
    out = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        subtopic = str(item.get("subtopic", "")).strip()
        if query:
            out.append({"subtopic": subtopic, "query": query})
    return out


def plan_initial_queries(user_query: str) -> dict:
    data = call_json(
        config.PLANNER_MODEL,
        _INITIAL_SYSTEM,
        f"Research query: {user_query}",
        max_tokens=500,
    )
    subtopics = _to_str_list(data.get("subtopics", []))
    query_plan = _to_query_plan(data.get("query_plan", []))[: config.MAX_QUERIES_PER_ITERATION]
    if not query_plan:
        query_plan = [{"subtopic": "", "query": user_query}]
    return {"subtopics": subtopics, "query_plan": query_plan}


def assess_gaps(user_query: str, memory: WorkingMemory, iteration: int) -> dict:
    view = memory.to_gap_view()
    user_prompt = (
        f"Research topic: {user_query}\n"
        f"Iteration completed: {iteration}\n"
        f"State:\n{view}\n\n"
        "Assess remaining gaps per the schema."
    )
    data = call_json(config.PLANNER_MODEL, _GAP_SYSTEM, user_prompt, max_tokens=350)
    return {
        "info_gain_estimate": float(data.get("info_gain_estimate", 0.0) or 0.0),
        "should_continue": bool(data.get("should_continue", False)),
        "query_plan": _to_query_plan(data.get("query_plan", [])),
        "reasoning": data.get("reasoning", ""),
    }
