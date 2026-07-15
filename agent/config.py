import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "llama-3.1-8b-instant")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "llama-3.3-70b-versatile")

# Orchestrator controls the loop, not the LLM.
# This runs inline in a production chat request path: bounded to at most 2
# search rounds (1 initial + 1 optional follow-up), targeting 5-20s total.
MAX_SEARCH_ITERATIONS = 2
MAX_QUERIES_PER_ITERATION = 5
MAX_TIME_BUDGET_SECONDS = 18.0
RESULTS_PER_QUERY = 3

# Hard per-call cap on Groq requests. Observed in practice: a single call
# occasionally takes 20s+ on the provider side with no error, no retry
# involved — that alone can blow the whole request budget. Every call site
# degrades gracefully (skip that step, proceed with less context) rather
# than block past this.
LLM_TIMEOUT_SECONDS = 8.0

# Reserved headroom for the final context-condensation call. Even with each
# call individually capped at LLM_TIMEOUT_SECONDS, a chain of several calls
# (gate -> extract -> gap-assess -> extract -> condense) can still overrun
# the total budget under real latency variance with zero errors involved —
# so a second search round is only started if there's still enough budget
# left to *also* afford the final step, not just the round itself.
SYNTHESIS_RESERVE_SECONDS = 6.0

# Tavily always returns its top-N results even when nothing is actually
# relevant (e.g. a made-up/internal term) — it fuzzy-matches on whatever
# generic words are left in the query instead of returning nothing. Its own
# relevance score is the only signal that catches this. Calibrated from
# real runs: well-known entities scored 0.8-0.9, a real-but-obscure product
# name scored 0.2-0.48, a fictional acronym scored 0.07-0.09.
MIN_RELEVANCE_SCORE = 0.25
