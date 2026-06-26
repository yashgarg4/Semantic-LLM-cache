"""Tests for the dashboard metrics: threshold-replay logic and best_score recording."""
from __future__ import annotations

from fastapi.testclient import TestClient

from semcache import CacheConfig, SemCache
from server.dashboard import (
    classify_at_threshold,
    replay_counts,
    replay_hit_rate,
    simulated_false_positive_risk,
)
from server.proxy import create_app


def _rec(hit_type: str, best_score):
    return {"hit_type": hit_type, "best_score": best_score}


def test_exact_is_threshold_independent() -> None:
    record = _rec("exact", None)
    assert classify_at_threshold(record, 0.80) == "exact"
    assert classify_at_threshold(record, 0.99) == "exact"


def test_semantic_classification_depends_on_threshold() -> None:
    record = _rec("miss", 0.93)  # near-miss at the live threshold
    assert classify_at_threshold(record, 0.92) == "semantic"  # would hit at 0.92
    assert classify_at_threshold(record, 0.95) == "miss"      # would miss at 0.95


def test_hit_rate_rises_as_threshold_drops() -> None:
    records = [_rec("exact", None), _rec("miss", 0.93), _rec("miss", 0.88), _rec("miss", 0.40)]
    # At 0.85: exact + 0.93 + 0.88 = 3/4 ; at 0.95: exact only = 1/4.
    assert replay_hit_rate(records, 0.85) > replay_hit_rate(records, 0.95)
    assert replay_counts(records, 0.85)["semantic"] == 2
    assert replay_counts(records, 0.95)["semantic"] == 0


def test_simulated_fp_risk_rises_as_threshold_drops() -> None:
    records = [_rec("miss", 0.90), _rec("miss", 0.85)]
    risky = simulated_false_positive_risk(records, 0.84)   # both count, low scores
    safe = simulated_false_positive_risk(records, 0.95)    # neither counts
    assert risky > safe


def test_empty_records_are_safe() -> None:
    assert replay_hit_rate([], 0.92) == 0.0
    assert simulated_false_positive_risk([], 0.92) == 0.0


def test_recent_records_carry_best_score_for_replay() -> None:
    # Integration: seed via the proxy (fake LLM, no key) and verify the recorded
    # history is replayable — misses carry a best_score, and a lower threshold
    # yields at least as many hits.
    def complete(query: str, model: str):
        return "answer", 50, 0.001

    client = TestClient(create_app(cache=SemCache(CacheConfig()), complete=complete))

    def chat(content: str):
        return client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": content}]},
        )

    chat("How do I reset my password?")               # miss (empty cache)
    chat("I forgot my password, how do I reset it?")  # semantic hit
    chat("What is the capital of France?")            # miss, low best_score

    recent = client.get("/recent", params={"n": 100}).json()["recent"]
    france = recent[-1]
    assert france["hit_type"] == "miss"
    assert france["best_score"] is not None and france["best_score"] < 0.92

    assert replay_hit_rate(recent, 0.30) >= replay_hit_rate(recent, 0.95)
