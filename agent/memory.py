"""Structured working memory (Layer 2) and raw evidence store (Layer 1).

The planner reasons only over WorkingMemory. Raw search results are kept
separately and never fed wholesale back into the LLM context.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fact:
    text: str
    sources: list[str] = field(default_factory=list)

    def add_source(self, url: str) -> None:
        if url not in self.sources:
            self.sources.append(url)


@dataclass
class RawResult:
    """One raw search hit (Layer 1). Kept for provenance, never re-summarized."""

    query: str
    url: str
    title: str
    snippet: str


@dataclass
class IterationRecord:
    """Observability trace of one loop iteration: what was targeted, what was
    searched, and which subtopics moved from unknown to known as a result."""

    iteration: int
    targets: list[dict]  # [{"subtopic": "...", "query": "..."}, ...]
    resolved_subtopics: list[str]
    still_unknown_after: list[str]


@dataclass
class WorkingMemory:
    definition: str = ""
    concepts: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    known_subtopics: list[str] = field(default_factory=list)
    unknown_subtopics: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    raw_evidence: list[RawResult] = field(default_factory=list, repr=False)
    queries_executed: list[str] = field(default_factory=list)
    search_log: list[IterationRecord] = field(default_factory=list)

    def add_fact(self, text: str, source_urls: list[str]) -> None:
        text_norm = text.strip().rstrip(".")
        for existing in self.facts:
            if existing.text.strip().rstrip(".").lower() == text_norm.lower():
                for u in source_urls:
                    existing.add_source(u)
                return
        fact = Fact(text=text.strip())
        for u in source_urls:
            fact.add_source(u)
        self.facts.append(fact)

    def add_concept(self, concept: str) -> None:
        concept = concept.strip()
        if concept and concept not in self.concepts:
            self.concepts.append(concept)

    def add_source(self, url: str) -> None:
        if url and url not in self.sources:
            self.sources.append(url)

    def mark_known(self, subtopic: str) -> None:
        subtopic = subtopic.strip()
        if not subtopic:
            return
        match = next(
            (u for u in self.unknown_subtopics if u.strip().lower() == subtopic.lower()),
            None,
        )
        if match:
            self.unknown_subtopics.remove(match)
            subtopic = match  # keep the original phrasing already tracked
        if subtopic not in self.known_subtopics:
            self.known_subtopics.append(subtopic)

    def add_unknown(self, subtopic: str) -> None:
        subtopic = subtopic.strip()
        if subtopic and subtopic not in self.known_subtopics and subtopic not in self.unknown_subtopics:
            self.unknown_subtopics.append(subtopic)

    def to_planner_view(self) -> dict:
        """Compact JSON-able snapshot for the planner LLM. No raw evidence included."""
        return {
            "definition": self.definition,
            "concepts": self.concepts,
            "facts": [f.text for f in self.facts],
            "limitations": self.limitations,
            "known_subtopics": self.known_subtopics,
            "unknown_subtopics": self.unknown_subtopics,
            "queries_already_executed": self.queries_executed,
        }

    def to_gap_view(self) -> dict:
        """Minimal snapshot for gap assessment: subtopic resolution already
        happens in extract.py per-iteration, so the gap step only needs to know
        what's left, not re-read every accumulated fact (keeps tokens low)."""
        return {
            "known_subtopics": self.known_subtopics,
            "unknown_subtopics": self.unknown_subtopics,
            "queries_already_executed": self.queries_executed,
            "fact_count": len(self.facts),
        }
