"""OpenAI-compatible FastAPI proxy in front of Gemini, backed by semcache.

Point any OpenAI client at this server (change its ``base_url`` to e.g.
``http://localhost:8000/v1``) and repeated/paraphrased prompts are served from
the cache instead of hitting the model.

``POST /v1/chat/completions`` accepts the standard chat-completions schema,
treats the last ``user`` message as the cache query, and returns an
OpenAI-shaped response with an ``x-semcache`` header set to one of
``hit-exact`` | ``hit-semantic`` | ``miss``. On a miss it calls Gemini (via
``langchain-google-genai``), stores the result, and returns it.

The metrics routes from ``server.dashboard`` are mounted over the same cache, so
this one server also serves ``GET /metrics`` and ``GET /recent``.

Run:  uvicorn server.proxy:app --reload
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from functools import lru_cache
from typing import Callable, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from semcache import CacheConfig, SemCache, estimate_cost_split, estimate_tokens
from server.dashboard import metrics_router

load_dotenv()  # pick up GOOGLE_API_KEY from a local .env if present

logger = logging.getLogger("semcache.proxy")

# Maps internal hit_type to the x-semcache header value.
_HIT_HEADER = {"exact": "hit-exact", "semantic": "hit-semantic", "miss": "miss"}

_warned_fake = False


class _RateLimiter:
    """Spaces calls so real Gemini requests stay under the free-tier RPM limit.

    Thread-safe: uvicorn serves the sync endpoint from a threadpool, so
    concurrent misses could hit Gemini at once; the lock serialises them and
    enforces a minimum gap between calls.
    """

    def __init__(self, max_per_minute: int, safety_sec: float = 0.5) -> None:
        # e.g. 10 RPM -> 6.0s + 0.5s cushion = 6.5s gap (~9.2 calls/min).
        self._interval = 60.0 / max(1, max_per_minute) + safety_sec
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval


# gemini-3.1-flash-lite free tier: 15 RPM. Cap just under it (override via env).
_GEMINI_RPM = int(os.getenv("SEMCACHE_GEMINI_RPM", "15"))
_gemini_limiter = _RateLimiter(_GEMINI_RPM)

# A completion fn takes (query, model) and returns (response, tokens, cost).
CompletionFn = Callable[[str, str], "tuple[str, int, float]"]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat-completions request we use; extra keys allowed."""

    model_config = ConfigDict(extra="allow")

    model: str = "gemini-3.1-flash-lite"
    messages: list[ChatMessage]
    temperature: Optional[float] = Field(default=None)


def _extract_user_query(messages: list[ChatMessage]) -> str:
    """Return the most recent user message content (the cache query)."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise HTTPException(status_code=400, detail="no 'user' message in request")


@lru_cache(maxsize=4)
def _get_gemini(model: str):
    """Lazily construct (and cache) a Gemini chat model.

    ``max_retries=0`` so a 429 (e.g. daily free-tier quota exhausted) fails fast
    with a clean error instead of langchain retrying with ~minute-long backoffs
    (which otherwise makes an exhausted-quota request hang for minutes). Our own
    ``_RateLimiter`` already handles per-minute spacing.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, max_retries=0)


def _tokens_and_cost(
    usage: dict, model: str, query: str, text: str
) -> tuple[int, float]:
    """Derive (total_tokens, cost) from LLM usage metadata.

    Uses the real input/output token split when present (and prices each with
    its own rate); falls back to char-based estimates only for missing fields.
    """
    in_tok = usage.get("input_tokens")
    out_tok = usage.get("output_tokens")
    if in_tok is None:
        in_tok = estimate_tokens(query)
    if out_tok is None:
        out_tok = estimate_tokens(text)
    in_tok, out_tok = int(in_tok), int(out_tok)
    total = int(usage.get("total_tokens") or (in_tok + out_tok))
    return total, estimate_cost_split(model, in_tok, out_tok)


def _content_to_text(content) -> str:
    """Flatten LLM message content (a str, or a list of content parts) to text.

    Newer Gemini models can return ``content`` as a list of blocks rather than a
    plain string; storing a non-string would fail CacheResult validation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return str(content) if content else ""


def _gemini_complete(query: str, model: str) -> tuple[str, int, float]:
    """Call Gemini and report (text, total_tokens, cost) — cost from the real
    input/output token split in the usage metadata."""
    _gemini_limiter.acquire()  # respect the free-tier requests-per-minute cap
    result = _get_gemini(model).invoke(query)
    text = _content_to_text(getattr(result, "content", ""))
    usage = getattr(result, "usage_metadata", None) or {}
    total, cost = _tokens_and_cost(usage, model, query, text)
    return text, total, cost


def _fake_complete(query: str, model: str) -> tuple[str, int, float]:
    """Keyless fallback so the proxy + dashboard are runnable without Gemini."""
    text = f"[semcache fake LLM - set GOOGLE_API_KEY for real Gemini] Re: {query}"
    in_tok, out_tok = estimate_tokens(query), estimate_tokens(text)
    return text, in_tok + out_tok, estimate_cost_split(model, in_tok, out_tok)


def _default_complete(query: str, model: str) -> tuple[str, int, float]:
    """Use Gemini if a key is configured, else a clearly-labelled fake LLM."""
    global _warned_fake
    if os.getenv("GOOGLE_API_KEY"):
        return _gemini_complete(query, model)
    if not _warned_fake:
        logger.warning(
            "GOOGLE_API_KEY not set - proxy is serving a FAKE local LLM on cache "
            "misses. Set GOOGLE_API_KEY to call real Gemini."
        )
        _warned_fake = True
    return _fake_complete(query, model)


def _to_openai_response(result, model: str) -> dict:
    """Shape a CacheResult into an OpenAI chat.completion object."""
    text = result.response or ""
    total_tokens = (
        result.tokens
        if result.tokens is not None
        else estimate_tokens(result.query) + estimate_tokens(text)
    )
    # Keep the split internally consistent (sum to the accurate total).
    prompt_tokens = min(estimate_tokens(result.query), total_tokens)
    completion_tokens = max(0, total_tokens - prompt_tokens)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        # Echo the cache outcome in the body too (the header is authoritative).
        "x_semcache": _HIT_HEADER[result.hit_type],
    }


def create_app(
    cache: Optional[SemCache] = None,
    complete: Optional[CompletionFn] = None,
) -> FastAPI:
    """Build the proxy app. ``complete`` is injectable (tests pass a fake)."""
    cache = cache or SemCache(CacheConfig())
    complete = complete or _default_complete

    app = FastAPI(title="semcache OpenAI-compatible proxy")

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, response: Response) -> dict:
        query = _extract_user_query(req.messages)

        def llm_fn(q: str):
            return complete(q, req.model)

        result = cache.call(query, llm_fn=llm_fn)
        response.headers["x-semcache"] = _HIT_HEADER[result.hit_type]
        return _to_openai_response(result, req.model)

    # Same-process metrics API over the same cache.
    app.include_router(metrics_router(cache))
    return app


# Default app for `uvicorn server.proxy:app`.
app = create_app()
