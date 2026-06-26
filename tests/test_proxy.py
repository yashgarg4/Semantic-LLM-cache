"""Tests for the OpenAI-compatible proxy.

A fake completion function is injected so no real Gemini call (and no API key)
is needed; the tests verify cache behaviour, the x-semcache header, the
OpenAI-shaped response, and the mounted metrics routes.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from semcache import CacheConfig, SemCache
from server.proxy import create_app


def _make_fake_complete():
    """A completion fn that counts invocations (i.e. cache misses)."""
    state = {"calls": 0}

    def complete(query: str, model: str):
        state["calls"] += 1
        return f"answer #{state['calls']}", 50, 0.001

    return complete, state


def _client(complete):
    app = create_app(cache=SemCache(CacheConfig()), complete=complete)
    return TestClient(app)


def _chat(client, content: str):
    return client.post(
        "/v1/chat/completions",
        json={"model": "gemini-1.5-flash", "messages": [{"role": "user", "content": content}]},
    )


def test_exact_repeat_returns_hit_exact_header_and_cached_body() -> None:
    complete, state = _make_fake_complete()
    client = _client(complete)

    r1 = _chat(client, "What is your return policy?")
    assert r1.status_code == 200
    assert r1.headers["x-semcache"] == "miss"

    r2 = _chat(client, "What is your return policy?")
    assert r2.headers["x-semcache"] == "hit-exact"
    assert state["calls"] == 1  # second request served from cache

    # Same answer text, and a well-formed OpenAI response shape.
    content = r2.json()["choices"][0]["message"]["content"]
    assert content == r1.json()["choices"][0]["message"]["content"]
    assert r2.json()["object"] == "chat.completion"
    assert "total_tokens" in r2.json()["usage"]


def test_paraphrase_returns_hit_semantic_header() -> None:
    complete, state = _make_fake_complete()
    client = _client(complete)

    r1 = _chat(client, "How do I reset my password?")
    assert r1.headers["x-semcache"] == "miss"

    r2 = _chat(client, "I forgot my password, how do I reset it?")
    assert r2.headers["x-semcache"] == "hit-semantic"
    assert state["calls"] == 1  # paraphrase served from cache


def test_missing_user_message_is_400() -> None:
    complete, _ = _make_fake_complete()
    client = _client(complete)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "gemini-1.5-flash", "messages": [{"role": "system", "content": "hi"}]},
    )
    assert r.status_code == 400


def test_metrics_and_recent_routes() -> None:
    complete, _ = _make_fake_complete()
    client = _client(complete)
    _chat(client, "What is your return policy?")
    _chat(client, "What is your return policy?")  # exact hit

    metrics = client.get("/metrics").json()
    assert metrics["counts"]["total"] == 2
    assert metrics["counts"]["exact"] == 1
    assert "cost_saved_usd" in metrics["savings"]

    recent = client.get("/recent?n=5").json()["recent"]
    assert len(recent) == 2
    assert recent[-1]["hit_type"] == "exact"
