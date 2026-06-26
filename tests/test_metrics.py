"""Tests for metrics math and cost estimation (no embedder needed)."""
from __future__ import annotations

import pytest

from semcache.metrics import GEMINI_PRICING_USD_PER_1M, Metrics, estimate_cost


def _rec(m: Metrics, hit_type: str, *, tokens=0, cost=0.0, score=None, query="q"):
    m.record(
        query=query,
        hit_type=hit_type,
        score=score,
        matched_query="base" if hit_type != "miss" else None,
        tokens_saved=tokens,
        cost_saved=cost,
        latency_ms=1.0,
    )


def test_hit_rate_and_counts() -> None:
    m = Metrics()
    _rec(m, "miss")
    _rec(m, "exact", tokens=100, cost=0.01, score=1.0)
    _rec(m, "semantic", tokens=100, cost=0.01, score=0.95)

    hr = m.hit_rate()
    assert hr["exact"] == pytest.approx(1 / 3)
    assert hr["semantic"] == pytest.approx(1 / 3)
    assert hr["miss"] == pytest.approx(1 / 3)
    assert hr["total"] == pytest.approx(2 / 3)  # overall hit rate

    assert m.counts() == {"exact": 1, "semantic": 1, "miss": 1, "total": 3}


def test_savings_accumulate() -> None:
    m = Metrics()
    _rec(m, "exact", tokens=100, cost=0.01)
    _rec(m, "semantic", tokens=150, cost=0.02)
    _rec(m, "miss")

    savings = m.savings()
    assert savings["tokens_saved"] == 250
    assert savings["cost_saved_usd"] == pytest.approx(0.03)
    assert savings["calls_avoided"] == 2  # only the two hits


def test_empty_metrics_have_no_division_by_zero() -> None:
    m = Metrics()
    assert m.hit_rate() == {"exact": 0.0, "semantic": 0.0, "total": 0.0, "miss": 0.0}
    assert m.savings() == {"tokens_saved": 0, "cost_saved_usd": 0, "calls_avoided": 0}


def test_recent_returns_last_n_in_order() -> None:
    m = Metrics()
    for i in range(5):
        _rec(m, "miss", query=f"q{i}")
    recent = m.recent(3)
    assert [r["query"] for r in recent] == ["q2", "q3", "q4"]
    assert "matched_query" in recent[0] and "score" in recent[0]


def test_estimate_cost_uses_table_and_default_fallback() -> None:
    # 1M tokens at the flash rate equals exactly the per-1M rate.
    assert estimate_cost("gemini-1.5-flash", 1_000_000) == pytest.approx(
        GEMINI_PRICING_USD_PER_1M["gemini-1.5-flash"]
    )
    # Cost scales linearly with tokens.
    assert estimate_cost("gemini-1.5-pro", 500_000) == pytest.approx(
        GEMINI_PRICING_USD_PER_1M["gemini-1.5-pro"] / 2
    )
    # Unknown model falls back to the default model's rate.
    assert estimate_cost("does-not-exist", 1_000_000) == pytest.approx(
        GEMINI_PRICING_USD_PER_1M["gemini-1.5-flash"]
    )
