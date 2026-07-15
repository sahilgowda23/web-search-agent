"""Tavily search wrapper: normalized-query cache + parallel execution."""
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
    ]
    _cache[key] = results
    return results


def run_searches(queries: list[str]) -> dict[str, list[RawResult]]:
    """Run multiple queries in parallel. Returns query -> raw results."""
    queries = queries[: config.MAX_QUERIES_PER_ITERATION]
    with ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
        results = list(pool.map(_run_single_search, queries))
    return dict(zip(queries, results))
