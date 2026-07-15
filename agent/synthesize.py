"""Final synthesis (Layer 3): turn working memory into the curated report.

Search Queries, Remaining Unknowns, and Sources are built deterministically
from WorkingMemory/search_log — they're already accurate structured data, so
asking an LLM to reproduce them would just add latency/tokens and a chance of
it dropping or inventing entries. The LLM is only used for Key Findings,
where actual prose curation from facts/concepts adds value.
"""
from __future__ import annotations

from . import config
from .llm import call_text
from .memory import WorkingMemory

_SYSTEM = """You produce curated research summaries, not direct answers to the
user's question. Given structured findings gathered from the web, write ONLY
a markdown bullet list of Key Findings — concise, technical statements drawn
only from the provided facts/concepts/definition. Do not add outside
knowledge, do not add a heading, do not answer the user's question directly
or add opinions — just the bullet list of what was learned from the web."""


def _render_search_queries_section(memory: WorkingMemory) -> str:
    if not memory.search_log:
        queries = "\n".join(f"- {q}" for q in memory.queries_executed)
        return queries or "- (none)"

    lines = []
    for record in memory.search_log:
        lines.append(f"**Iteration {record.iteration}:**")
        for target in record.targets:
            subtopic = target.get("subtopic")
            label = f" _(targeting: {subtopic})_" if subtopic else ""
            lines.append(f"- {target['query']}{label}")
    return "\n".join(lines)


def _render_list_section(items: list[str]) -> str:
    if not items:
        return "None identified"
    return "\n".join(f"- {item}" for item in items)


def synthesize(user_query: str, memory: WorkingMemory) -> str:
    facts_block = "\n".join(f"- {f.text} (sources: {', '.join(f.sources) or 'n/a'})" for f in memory.facts)
    user_prompt = (
        f"Research topic: {user_query}\n\n"
        f"Definition: {memory.definition or 'n/a'}\n"
        f"Concepts: {', '.join(memory.concepts) or 'n/a'}\n"
        f"Facts:\n{facts_block or 'n/a'}\n"
        f"Limitations: {', '.join(memory.limitations) or 'n/a'}\n\n"
        "Write the Key Findings bullet list per the instructions."
    )
    key_findings = call_text(config.SYNTHESIS_MODEL, _SYSTEM, user_prompt, max_tokens=800)

    return (
        f"## Search Queries\n{_render_search_queries_section(memory)}\n\n"
        f"## Key Findings\n{key_findings}\n\n"
        f"## Remaining Unknowns\n{_render_list_section(memory.unknown_subtopics)}\n\n"
        f"## Sources\n{_render_list_section(memory.sources)}"
    )
