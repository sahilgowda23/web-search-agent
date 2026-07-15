"""Thin Groq wrapper shared by planner and synthesizer."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time

from groq import APIConnectionError, APITimeoutError, Groq, RateLimitError

from . import config

_client = Groq(api_key=config.GROQ_API_KEY)

_MAX_RETRIES = 3
_DEBUG_TIMING = os.getenv("DEBUG_LLM_TIMING", "")


class LLMTimeout(Exception):
    """Raised when a call exceeds LLM_TIMEOUT_SECONDS wall-clock, even if
    the underlying request never itself raised an error."""


def _call_with_deadline(func, timeout, **kwargs):
    """Groq/httpx's own `timeout=` kwarg is a per-read-operation timeout, not
    a wall-clock deadline — a response trickling in slow-but-steady chunks
    can still take far longer than the configured timeout (observed: 16-19s
    against timeout=8.0). A daemon thread + join(timeout) enforces a true
    wall-clock cap regardless of pacing, and — unlike a shared
    ThreadPoolExecutor — an abandoned call doesn't block process exit or
    starve a fixed-size worker pool if several calls run long at once; each
    gets its own thread that the OS reclaims independently once it finishes."""
    box: dict = {}

    def _target():
        try:
            box["result"] = func(**kwargs)
        except BaseException as err:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = err

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise LLMTimeout(f"{kwargs.get('model')} exceeded {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def _retry_delay_seconds(err: RateLimitError, attempt: int) -> float:
    """Groq's 429 body names the exact wait ("try again in 12.16s"); prefer
    that over a blind exponential backoff since it's usually far shorter."""
    match = re.search(r"try again in ([\d.]+)s", str(err))
    if match:
        return float(match.group(1)) + 0.5
    return 2.0 * (attempt + 1)


def _create(**kwargs):
    """Rate limits (429) get a smart, bounded retry since Groq tells us
    exactly how long to wait. A timeout/connection failure does NOT retry —
    a slow call is likely to be slow again, and retrying it would just risk
    spending 2x LLM_TIMEOUT_SECONDS on a single step; callers degrade
    gracefully instead (see call_json/call_text)."""
    call_start = time.monotonic()
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _call_with_deadline(
                _client.chat.completions.create,
                config.LLM_TIMEOUT_SECONDS,
                **kwargs,
            )
            if _DEBUG_TIMING:
                print(
                    f"[llm] {kwargs.get('model')} ok in {time.monotonic() - call_start:.2f}s"
                    f" (attempt {attempt + 1})",
                    file=sys.stderr,
                )
            return resp
        except RateLimitError as err:
            delay = _retry_delay_seconds(err, attempt)
            if _DEBUG_TIMING:
                print(
                    f"[llm] {kwargs.get('model')} RATE LIMITED (attempt {attempt + 1}/{_MAX_RETRIES + 1}),"
                    f" sleeping {delay:.2f}s: {err}",
                    file=sys.stderr,
                )
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(delay)
        except (APITimeoutError, APIConnectionError) as err:
            if _DEBUG_TIMING:
                print(
                    f"[llm] {kwargs.get('model')} TIMED OUT after "
                    f"{time.monotonic() - call_start:.2f}s: {err}",
                    file=sys.stderr,
                )
            raise
        except LLMTimeout:
            # Wall-clock cap hit — the request thread is abandoned (still
            # running in the background, but nothing waits on it further,
            # and it can't block process exit or starve other calls since
            # it isn't holding a slot in a shared pool); this is the
            # tradeoff for a hard latency bound.
            if _DEBUG_TIMING:
                print(
                    f"[llm] {kwargs.get('model')} WALL-CLOCK TIMEOUT at "
                    f"{config.LLM_TIMEOUT_SECONDS:.1f}s (actual call still in flight)",
                    file=sys.stderr,
                )
            raise


def call_json(model: str, system: str, user: str, max_tokens: int = 800) -> dict:
    """Call Groq chat completion, forcing JSON object output. Returns {} on
    a timeout/connection failure or malformed JSON — callers already treat
    missing fields as safe defaults, so this degrades rather than crashes."""
    try:
        resp = _create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except (APITimeoutError, APIConnectionError, LLMTimeout):
        return {}
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def call_text(model: str, system: str, user: str, max_tokens: int = 1200) -> str:
    """Returns "" on a timeout/connection failure so callers can fall back
    to a deterministic (non-LLM) rendering instead of crashing."""
    try:
        resp = _create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
    except (APITimeoutError, APIConnectionError, LLMTimeout):
        return ""
    return resp.choices[0].message.content.strip()
