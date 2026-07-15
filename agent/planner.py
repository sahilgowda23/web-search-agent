"""Planner: gates whether search is needed, decomposes the query into a
bounded set of searches, and flags when a clarifying question should be
asked instead of searching.

This runs inline in a production chat request path, not as a standalone
research job — so it optimizes for a small, fixed number of LLM/search round
trips over exhaustive subtopic coverage. The orchestrator enforces a hard
cap of 2 search rounds and a wall-clock budget regardless of what the
planner recommends.

Each search query is paired with the single subtopic it targets. This is
a deterministic guardrail: extract.py is only allowed to mark a subtopic
"known" if this round actually ran a query for it — regardless of what
the (unreliable, small) extraction model claims was "covered".
"""
from __future__ import annotations

from . import config
from .llm import call_json
from .memory import WorkingMemory

_ASSESS_SYSTEM = f"""You are the first-pass triage step for a production
chatbot's web search. Given the user's message, decide in one pass:

1. needs_search: does answering this actually require fresh/external web
   information (facts, current events, named products/companies, prices,
   comparisons)? Ordinary conversation, opinions, math, or things answerable
   from general knowledge alone need NO search — set this false and skip
   the rest.
2. clarification_needed: set true ONLY if search would be useless or
   inappropriate as posed — e.g. the query centers on a named private
   individual you'd be building a profile of, or references a specific
   proprietary/internal system, product, or org name that has no public
   footprint to search for (you can't tell what it is from the query alone).
   Do NOT set this for ordinary ambiguity a reasonable search can route
   around — only when guessing would produce ungrounded, made-up-sounding
   output. If true, write one short, specific clarification_question and
   leave subtopics/query_plan empty.
3. If needs_search is true and clarification is not needed: identify at most
   {config.MAX_QUERIES_PER_ITERATION} distinct subtopics and one search
   query per subtopic — enough to get real coverage, not exhaustive. Entities
   the query names (companies, products, systems) each get their own
   entity-specific subtopics for facets that differ per entity (e.g. "A:
   pricing", "B: pricing"); never produce pairwise subtopics like "A vs B".
   Facets inherently about the group as a whole stay as one joint subtopic.
   With 0-1 entities, use plain facet names, no prefix. Prioritize the
   highest-value subtopics first — you only get one guaranteed round.

Return strict JSON only, matching this shape exactly:
{{
  "needs_search": true/false,
  "clarification_needed": true/false,
  "clarification_question": "...",
  "subtopics": ["...", "...", ...],
  "query_plan": [{{"subtopic": "...", "query": "..."}}, ...]
}}
Every "subtopic" value in query_plan must be copied verbatim from "subtopics"."""

_GAP_SYSTEM = f"""You are the single allowed follow-up planning step of a
production chatbot's web search (at most one more round after this).
Subtopic resolution already happened during extraction, so you're only told
which subtopics are still unknown_subtopics — decide what's worth one more
targeted search, if anything.
Return strict JSON only, matching this shape exactly:
{{
  "should_continue": true/false,
  "query_plan": [{{"subtopic": "...", "query": "..."}}, ...],
  "reasoning": "one short sentence"
}}
Guidance:
- Only propose queries for unknown_subtopics that are genuinely high-value
  for answering the user — skip minor/nice-to-have ones, this is the last
  round.
- query_plan must have at most {config.MAX_QUERIES_PER_ITERATION} entries,
  one per subtopic targeted (subtopic copied verbatim from
  unknown_subtopics), and must not repeat queries_already_executed."""


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


def assess_query(user_query: str) -> dict:
    """Single triage call: needs_search gate, clarification check, and (if
    clear to proceed) the bounded initial query plan — one round trip."""
    data = call_json(
        config.PLANNER_MODEL,
        _ASSESS_SYSTEM,
        f"User message: {user_query}",
        max_tokens=500,
    )
    if not data:
        # The call itself failed/timed out (call_json returns {} in that
        # case) — distinct from the model explicitly deciding no search is
        # needed. Silently answering with zero web context on a query that
        # may well have needed it is worse than a plain fallback search, so
        # fail toward attempting search rather than skipping it.
        return {
            "needs_search": True,
            "clarification_needed": False,
            "clarification_question": "",
            "subtopics": [],
            "query_plan": [{"subtopic": "", "query": user_query}],
        }
    needs_search = bool(data.get("needs_search", False))
    clarification_needed = bool(data.get("clarification_needed", False))
    query_plan = _to_query_plan(data.get("query_plan", []))[: config.MAX_QUERIES_PER_ITERATION]
    return {
        "needs_search": needs_search,
        "clarification_needed": clarification_needed,
        "clarification_question": str(data.get("clarification_question", "")).strip(),
        "subtopics": _to_str_list(data.get("subtopics", [])),
        "query_plan": query_plan,
    }


def assess_gaps(user_query: str, memory: WorkingMemory, iteration: int) -> dict:
    view = memory.to_gap_view()
    user_prompt = (
        f"Research topic: {user_query}\n"
        f"Iteration completed: {iteration}\n"
        f"State:\n{view}\n\n"
        "Assess remaining gaps per the schema."
    )
    data = call_json(config.PLANNER_MODEL, _GAP_SYSTEM, user_prompt, max_tokens=300)
    return {
        "should_continue": bool(data.get("should_continue", False)),
        "query_plan": _to_query_plan(data.get("query_plan", [])),
        "reasoning": data.get("reasoning", ""),
    }
