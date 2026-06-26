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
import time
import uuid
from functools import lru_cache
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from semcache import CacheConfig, SemCache, estimate_cost, estimate_tokens
from server.dashboard import metrics_router

logger = logging.getLogger("semcache.proxy")

# Maps internal hit_type to the x-semcache header value.
_HIT_HEADER = {"exact": "hit-exact", "semantic": "hit-semantic", "miss": "miss"}

_warned_fake = False

# A completion fn takes (query, model) and returns (response, tokens, cost).
CompletionFn = Callable[[str, str], "tuple[str, int, float]"]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat-completions request we use; extra keys allowed."""

    model_config = ConfigDict(extra="allow")

    model: str = "gemini-1.5-flash"
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
    """Lazily construct (and cache) a Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model)


def _gemini_complete(query: str, model: str) -> tuple[str, int, float]:
    """Call Gemini and report (text, tokens, cost)."""
    result = _get_gemini(model).invoke(query)
    text = getattr(result, "content", None) or str(result)
    usage = getattr(result, "usage_metadata", None) or {}
    tokens = int(usage.get("total_tokens") or (estimate_tokens(query) + estimate_tokens(text)))
    return text, tokens, estimate_cost(model, tokens)


def _fake_complete(query: str, model: str) -> tuple[str, int, float]:
    """Keyless fallback so the proxy + dashboard are runnable without Gemini."""
    text = f"[semcache fake LLM — set GOOGLE_API_KEY for real Gemini] Re: {query}"
    tokens = estimate_tokens(query) + estimate_tokens(text)
    return text, tokens, estimate_cost(model, tokens)


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
    prompt_tokens = estimate_tokens(result.query)
    completion_tokens = estimate_tokens(text)
    total_tokens = result.tokens if result.tokens is not None else prompt_tokens + completion_tokens
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
