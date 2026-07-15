import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "llama-3.1-8b-instant")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "llama-3.3-70b-versatile")

# Orchestrator controls the loop, not the LLM.
MAX_SEARCH_ITERATIONS = 3
MAX_QUERIES_PER_ITERATION = 3
MAX_TIME_BUDGET_SECONDS = 18.0
RESULTS_PER_QUERY = 3
MIN_INFO_GAIN_TO_CONTINUE = 0.35
