"""Tests for metrics math and cost estimation (no embedder needed)."""
from __future__ import annotations

import pytest

from semcache.metrics import (
    GEMINI_PRICING_USD_PER_1M,
    Metrics,
    estimate_cost,
    estimate_cost_split,
)


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


def test_estimate_cost_split_prices_input_and_output_separately() -> None:
    in_rate, out_rate = GEMINI_PRICING_USD_PER_1M["gemini-2.5-flash-lite"]
    # 1M input + 1M output == input_rate + output_rate.
    assert estimate_cost_split("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(
        in_rate + out_rate
    )
    # Output tokens cost more than the same number of input tokens.
    assert estimate_cost_split("gemini-2.5-flash-lite", 0, 1_000_000) > estimate_cost_split(
        "gemini-2.5-flash-lite", 1_000_000, 0
    )
    # A concrete bill: 1000 in + 500 out at (0.10, 0.40) per 1M.
    assert estimate_cost_split("gemini-2.5-flash-lite", 1_000, 500) == pytest.approx(
        1_000 / 1e6 * in_rate + 500 / 1e6 * out_rate
    )


def test_estimate_cost_split_unknown_model_falls_back_to_default() -> None:
    default_in, default_out = GEMINI_PRICING_USD_PER_1M["gemini-1.5-flash"]
    assert estimate_cost_split("does-not-exist", 1_000_000, 1_000_000) == pytest.approx(
        default_in + default_out
    )


def test_estimate_cost_blended_is_between_input_and_output_rates() -> None:
    in_rate, out_rate = GEMINI_PRICING_USD_PER_1M["gemini-1.5-flash"]
    # Total-only fallback applies the average of the two rates.
    assert estimate_cost("gemini-1.5-flash", 1_000_000) == pytest.approx(
        (in_rate + out_rate) / 2
    )
