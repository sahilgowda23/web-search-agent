"""Thin Groq wrapper shared by planner and synthesizer."""
from __future__ import annotations

import json

from groq import Groq

from . import config

_client = Groq(api_key=config.GROQ_API_KEY)


def call_json(model: str, system: str, user: str, max_tokens: int = 800) -> dict:
    """Call Groq chat completion, forcing JSON object output."""
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def call_text(model: str, system: str, user: str, max_tokens: int = 1200) -> str:
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
