"""Turn raw search results into structured findings, merged into working memory.

Uses the small planner model. Compresses intelligently: keeps definitions,
concepts, facts, numbers, limitations; drops marketing filler and repetition.
Deduplication and source-attachment happen in WorkingMemory.add_fact/add_concept.
"""
from __future__ import annotations

from . import config
from .llm import call_json
from .memory import RawResult, WorkingMemory

_SYSTEM = """You extract research findings from raw web search snippets.
Return strict JSON only, matching this shape:
{
  "definition": "one sentence defining the topic, or empty string if not applicable",
  "concepts": ["short concept or term", ...],
  "facts": [{"text": "concise factual statement", "source_urls": ["url", ...]}, ...],
  "limitations": ["caveat or limitation mentioned in sources", ...],
  "subtopics_covered": ["...", ...]
}
Rules:
- Keep facts concise and technical: definitions, numbers, APIs, algorithms, benchmarks.
- Drop marketing language, anecdotes, and repeated restatements.
- Every fact must cite at least one source_url taken from the provided snippets.
- Do not invent facts not present in the snippets.
- subtopics_covered must ONLY contain strings copied verbatim from "Target
  subtopics this round" — include one only if these snippets contain enough
  concrete detail (not just a passing mention) to consider it addressed.
  If none qualify, return an empty list."""


def extract_findings(
    user_query: str,
    results_by_query: dict[str, list[RawResult]],
    memory: WorkingMemory,
    targeted_subtopics: list[str] | None = None,
) -> None:
    """Extract findings from this iteration's raw results and merge into memory.

    targeted_subtopics restricts which subtopics this round is even allowed to
    resolve: it's the deterministic guardrail against the extraction model
    over-claiming coverage of subtopics no query this round actually searched
    for (small models are unreliable at self-restraint here)."""
    all_results: list[RawResult] = [r for results in results_by_query.values() for r in results]
    if not all_results:
        return

    targeted_subtopics = targeted_subtopics or []
    snippet_block = "\n".join(
        f"[{i}] url={r.url}\ntitle={r.title}\nsnippet={r.snippet[:350]}"
        for i, r in enumerate(all_results)
    )
    targets = "\n".join(f"- {s}" for s in targeted_subtopics) or "(none)"
    user_prompt = (
        f"Research topic: {user_query}\n\n"
        f"Target subtopics this round:\n{targets}\n\n"
        f"Search snippets:\n{snippet_block}\n\n"
        "Extract findings as JSON per the schema."
    )

    data = call_json(config.PLANNER_MODEL, _SYSTEM, user_prompt, max_tokens=700)

    memory.raw_evidence.extend(all_results)
    for query in results_by_query:
        if query not in memory.queries_executed:
            memory.queries_executed.append(query)

    if data.get("definition") and not memory.definition:
        memory.definition = data["definition"]

    for concept in data.get("concepts", []):
        memory.add_concept(concept)

    for fact in data.get("facts", []):
        text = fact.get("text", "").strip()
        sources = [u for u in fact.get("source_urls", []) if u]
        if text:
            memory.add_fact(text, sources)
            for u in sources:
                memory.add_source(u)

    for limitation in data.get("limitations", []):
        limitation = limitation.strip()
        if limitation and limitation not in memory.limitations:
            memory.limitations.append(limitation)

    targeted_lower = {s.lower() for s in targeted_subtopics}
    for subtopic in data.get("subtopics_covered", []):
        # Hard guardrail: only allow marking a subtopic known if it was
        # actually targeted this round, no matter what the model claims.
        if subtopic.strip().lower() in targeted_lower:
            memory.mark_known(subtopic)

    # Fall back: attach any result URL we haven't captured via facts, so
    # sources are never lost even if the extractor missed one.
    for r in all_results:
        memory.add_source(r.url)
