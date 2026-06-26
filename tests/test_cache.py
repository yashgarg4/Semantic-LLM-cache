"""End-to-end tests for the dual-layer SemCache (real embedder + paraphrases)."""
from __future__ import annotations

import pytest

from semcache import CacheConfig, SemCache


def make_counting_llm():
    """An llm_fn that counts how many times it was actually invoked."""
    state = {"calls": 0}

    def llm(query: str):
        state["calls"] += 1
        return f"answer-{state['calls']}", 100, 0.001

    return llm, state


@pytest.fixture(scope="module")
def config() -> CacheConfig:
    return CacheConfig(threshold=0.92)


def test_first_query_is_miss_and_invokes_llm(config: CacheConfig) -> None:
    llm, state = make_counting_llm()
    cache = SemCache(config=config, llm_fn=llm)
    result = cache.call("What is your return policy?")
    assert result.hit_type == "miss"
    assert result.response == "answer-1"
    assert state["calls"] == 1


def test_exact_repeat_returns_exact_hit(config: CacheConfig) -> None:
    llm, state = make_counting_llm()
    cache = SemCache(config=config, llm_fn=llm)
    cache.call("What is your return policy?")
    # Different case/whitespace, same normalised query -> exact hit.
    result = cache.call("what is   your RETURN policy?")
    assert result.hit_type == "exact"
    assert result.score == 1.0
    assert result.response == "answer-1"
    assert state["calls"] == 1          # LLM not called again


def test_paraphrase_returns_semantic_hit(config: CacheConfig) -> None:
    llm, state = make_counting_llm()
    cache = SemCache(config=config, llm_fn=llm)
    cache.call("How do I reset my password?")
    result = cache.call("I forgot my password, how do I reset it?")
    assert result.hit_type == "semantic"
    assert result.score >= 0.92
    assert result.response == "answer-1"   # served from the cached entry
    assert state["calls"] == 1


def test_dissimilar_query_is_a_miss(config: CacheConfig) -> None:
    llm, state = make_counting_llm()
    cache = SemCache(config=config, llm_fn=llm)
    cache.call("How do I reset my password?")
    result = cache.call("What is the capital of France?")
    assert result.hit_type == "miss"
    assert state["calls"] == 2          # genuinely new query -> LLM called


def test_get_never_invokes_llm(config: CacheConfig) -> None:
    llm, state = make_counting_llm()
    cache = SemCache(config=config, llm_fn=llm)
    result = cache.get("a query that was never cached")
    assert result.hit_type == "miss"
    assert result.response is None
    assert state["calls"] == 0


def test_call_records_metrics_with_savings(config: CacheConfig) -> None:
    # Each call returns 100 tokens / $0.01 so savings are easy to check.
    def llm(query: str):
        return "answer", 100, 0.01

    cache = SemCache(config=config, llm_fn=llm)
    cache.call("How do I reset my password?")               # miss (stored)
    cache.call("How do I reset my password?")               # exact hit
    cache.call("I forgot my password, how do I reset it?")  # semantic hit

    counts = cache.metrics.counts()
    assert counts == {"exact": 1, "semantic": 1, "miss": 1, "total": 3}

    savings = cache.metrics.savings()
    assert savings["calls_avoided"] == 2          # the two hits
    assert savings["tokens_saved"] == 200         # 2 hits * 100 tokens
    assert savings["cost_saved_usd"] == pytest.approx(0.02)
