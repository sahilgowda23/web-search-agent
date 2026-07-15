#!/usr/bin/env python3
import argparse
import sys

from agent.orchestrator import run_research


def _print_search_trace(memory, file=sys.stderr) -> None:
    print("\n--- Search trace ---", file=file)
    for record in memory.search_log:
        print(f"\nIteration {record.iteration}:", file=file)
        for target in record.targets:
            subtopic = target.get("subtopic") or "(general)"
            print(f"  [{subtopic}] -> \"{target['query']}\"", file=file)
        if record.resolved_subtopics:
            print(f"  resolved: {record.resolved_subtopics}", file=file)
        else:
            print("  resolved: (none)", file=file)
        print(f"  still unknown after this round: {record.still_unknown_after or '(none)'}", file=file)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI web research agent")
    parser.add_argument("query", nargs="+", help="Research query, e.g. 'Explain Kubernetes'")
    parser.add_argument("--verbose", action="store_true", help="Print timing/debug info")
    args = parser.parse_args()

    user_query = " ".join(args.query)
    result = run_research(user_query)

    if not result.needs_search:
        print("(no search needed)")
    elif result.clarification_question:
        print(f"(clarification needed) {result.clarification_question}")
    elif result.context:
        print(result.context)
    else:
        print("(search ran but found nothing usable)")

    if args.verbose:
        print("\n---", file=sys.stderr)
        print(f"iterations: {result.iterations_run}", file=sys.stderr)
        print(f"elapsed: {result.elapsed_seconds:.2f}s", file=sys.stderr)
        print(f"stop_reason: {result.stop_reason}", file=sys.stderr)
        _print_search_trace(result.memory)


if __name__ == "__main__":
    main()
