"""Context handoff (Layer 3): turn working memory into a compact context
block for the production chatbot's own LLM to use while answering — this
module does NOT answer the user's question or produce a standalone report.

One lean condensation call (fast model, short output) merges/dedupes the
raw extracted facts into a few grounded bullets. Gaps and sources are built
deterministically from WorkingMemory — they're already accurate structured
data, so asking an LLM to reproduce them would just add latency and a
chance of it dropping or inventing entries.
"""
from __future__ import annotations

from . import config
from .llm import call_text
from .memory import WorkingMemory

_CONDENSE_SYSTEM = """You compress grounded web-search findings into compact
context for another AI assistant to use while answering a user's question.
Write 3-8 short plain bullet points, facts only, drawn strictly from the
provided facts/concepts/definition. No heading, no meta-commentary, and do
not answer the user's question yourself — this is background context for
another model to reason over, not an answer."""


def _render_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_context(
    user_query: str,
    memory: WorkingMemory,
    searched_but_unresolved: list[str],
    time_remaining: float = config.LLM_TIMEOUT_SECONDS,
) -> str:
    if not memory.facts and not memory.concepts and not memory.definition:
        return ""

    facts_block = "\n".join(f"- {f.text} (sources: {', '.join(f.sources) or 'n/a'})" for f in memory.facts)

    # Not worth attempting a call that might eat the full LLM_TIMEOUT_SECONDS
    # if we don't have that much budget left — skip straight to the
    # deterministic fallback instead of finding out via timeout.
    if time_remaining < config.LLM_TIMEOUT_SECONDS:
        condensed = facts_block or memory.definition or "(no details extracted)"
    else:
        prompt = (
            f"User's question: {user_query}\n\n"
            f"Definition: {memory.definition or 'n/a'}\n"
            f"Concepts: {', '.join(memory.concepts) or 'n/a'}\n"
            f"Facts:\n{facts_block or 'n/a'}\n"
            f"Limitations: {', '.join(memory.limitations) or 'n/a'}\n\n"
            "Write the compact context bullets per the instructions."
        )
        condensed = call_text(config.PLANNER_MODEL, _CONDENSE_SYSTEM, prompt, max_tokens=400)
        if not condensed:
            # Condensation call timed out/failed — fall back to the raw
            # fact list rather than returning no context at all.
            condensed = facts_block or memory.definition or "(no details extracted)"

    never_reached = [s for s in memory.unknown_subtopics if s not in searched_but_unresolved]

    sections = [f"## Web search context\n{condensed}"]
    if searched_but_unresolved:
        sections.append(
            "## Searched but no public info found\n" + _render_list(searched_but_unresolved)
        )
    if never_reached:
        sections.append(
            "## Not covered (time budget)\n" + _render_list(never_reached)
        )
    if memory.sources:
        sections.append("## Sources\n" + _render_list(memory.sources))

    return "\n\n".join(sections)
