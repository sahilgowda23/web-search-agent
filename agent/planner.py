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

_INITIAL_SYSTEM = f"""You plan web research for an agent that decomposes a topic
before searching. Given a user's research query:

1. Identify the distinct entities the query is about, comparing, or
   evaluating together — could be 0 (a process, event, or concept with no
   named entities, e.g. "why did X happen", "how do I migrate to Y"), 1
   ("explain gRPC"), 2 ("should X partner with Y"), or more ("compare AWS,
   GCP, and Azure for Z"). There is no upper bound — name every one the
   query mentions, not just the first two.
2. Identify the distinct facets/subtopics the query asks about. If the user
   explicitly lists aspects (e.g. "architecture, adopters, SDK support,
   authentication, security concerns"), each one is its own facet — do not
   merge them. If the query is a simple topic with no listed facets, infer
   3-5 natural facets (e.g. definition, how it works, use cases; or for a
   process/causal query: steps, causes, timeline, consequences — whatever
   fits the query's actual shape).
3. If there are 2+ entities, decide per facet:
   - Entity-specific facets (what each entity individually does/costs/
     offers — e.g. "strategy", "pricing", "performance") must be repeated
     as ONE subtopic PER entity, covering ALL of them, e.g. for entities
     "A", "B", "C" and facet "pricing" produce "A: pricing", "B: pricing",
     AND "C: pricing" — never leave one entity out, and never collapse them
     into one blended subtopic.
   - Never create pairwise subtopics like "A vs B" — with 3+ entities that
     explodes combinatorially and fragments the research. Comparison is
     handled later during reasoning, not as a search subtopic.
   - Facets that are inherently about the group as a whole (e.g.
     "compatibility between them", "market they compete in") stay as ONE
     single subtopic covering all entities together, not split or paired.
   If there are 0-1 entities, use plain facet names as subtopics, no
   entity prefix.
4. Propose up to {config.MAX_QUERIES_PER_ITERATION} focused search queries
   covering the highest-priority subtopics first (later subtopics get
   covered in later iterations). Each query targets exactly one subtopic and
   must name the specific entity/ies it targets, not just the abstract facet.

Return strict JSON only, matching this shape exactly:
{{
  "subtopics": ["...", "...", ...],
  "query_plan": [{{"subtopic": "...", "query": "..."}}, ...]
}}
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
