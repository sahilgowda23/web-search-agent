"""Tavily search wrapper: normalized-query cache + parallel execution +
relevance filtering.

Tavily always returns its top-N results even when nothing is actually
relevant to the query — for a made-up/internal term it just fuzzy-matches
on whatever generic words are left over. Its own relevance score is the
only signal that separates a real hit from noise, so results below
MIN_RELEVANCE_SCORE are dropped here, before anything downstream (the
extraction LLM) gets a chance to write them up as if they were relevant.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from tavily import TavilyClient

from . import config
from .memory import RawResult

_client = TavilyClient(api_key=config.TAVILY_API_KEY)

# Process-lifetime cache. Keyed by normalized query text so paraphrases
# ("What is Kubernetes" / "Explain Kubernetes") reuse the same results.
_cache: dict[str, list[RawResult]] = {}


def _normalize(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"^(what is|what are|explain|describe|tell me about|research)\s+", "", q)
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def _run_single_search(query: str) -> list[RawResult]:
    key = _normalize(query)
    if key in _cache:
        return _cache[key]

    response = _client.search(
        query=query,
        max_results=config.RESULTS_PER_QUERY,
        search_depth="basic",
    )
    results = [
        RawResult(
            query=query,
            url=r.get("url", ""),
            title=r.get("title", ""),
            snippet=r.get("content", ""),
        )
        for r in response.get("results", [])
        if r.get("score", 0.0) >= config.MIN_RELEVANCE_SCORE
    ]
    _cache[key] = results
    return results


def _entity_fallback_query(subtopic: str) -> str | None:
    """For an entity-scoped subtopic like "NECF: current offerings", the
    original query embeds extra descriptive words that dilute the relevance
    score even when the entity itself is real — retry with just the entity
    name alone, a cleaner signal for whether it has any public footprint at
    all before giving up on it."""
    if ":" not in subtopic:
        return None
    entity = subtopic.split(":", 1)[0].strip()
    return entity or None


def run_searches(query_plan: list[dict]) -> dict[str, list[RawResult]]:
    """Run each {"subtopic", "query"} pair in parallel. Any query that comes
    back with nothing above the relevance threshold gets one fallback
    attempt on just its entity name. Returns original query text -> results."""
    query_plan = query_plan[: config.MAX_QUERIES_PER_ITERATION]
    queries = [qp["query"] for qp in query_plan]

    with ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
        results_by_query = dict(zip(queries, pool.map(_run_single_search, queries)))

    fallback_map = {
        qp["query"]: _entity_fallback_query(qp.get("subtopic", ""))
        for qp in query_plan
        if not results_by_query.get(qp["query"])
    }
    fallback_map = {q: f for q, f in fallback_map.items() if f}
    if not fallback_map:
        return results_by_query

    fallback_queries = list(dict.fromkeys(fallback_map.values()))
    with ThreadPoolExecutor(max_workers=max(1, len(fallback_queries))) as pool:
        fallback_results = dict(zip(fallback_queries, pool.map(_run_single_search, fallback_queries)))

    for original_query, fallback_query in fallback_map.items():
        results_by_query[original_query] = fallback_results.get(fallback_query, [])

    return results_by_query
