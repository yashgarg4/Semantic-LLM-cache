"""Metrics REST API consumed by the Streamlit dashboard (Phase 4).

Exposes read-only views over a :class:`~semcache.SemCache`'s metrics:

* ``GET /metrics`` — hit rate, counts, savings, latency, threshold, entries.
* ``GET /recent?n=20`` — the last N lookups.

The routes are provided as a router factory so the proxy can mount them over
*its* cache (single process, shared in-memory state), and a standalone app
factory is provided for running the dashboard API on its own.
"""
from __future__ import annotations

from typing import Iterable

from fastapi import APIRouter, FastAPI

from semcache import SemCache


# --- Threshold-explorer replay logic ------------------------------------------
# Pure functions over recorded lookups (dicts as returned by /recent). They let
# the dashboard re-classify history at an arbitrary threshold to visualise the
# precision/recall tradeoff, without re-running any embeddings.

def classify_at_threshold(record: dict, threshold: float) -> str:
    """Re-classify one recorded lookup at ``threshold``.

    Exact matches are threshold-independent (the exact layer runs first), so
    they stay ``"exact"``. Everything else is a semantic hit iff its recorded
    top-neighbour cosine (``best_score``) clears the threshold.
    """
    if record.get("hit_type") == "exact":
        return "exact"
    best = record.get("best_score")
    if best is not None and best >= threshold:
        return "semantic"
    return "miss"


def replay_counts(records: Iterable[dict], threshold: float) -> dict[str, int]:
    """Counts of {exact, semantic, miss} if history were replayed at ``threshold``."""
    counts = {"exact": 0, "semantic": 0, "miss": 0}
    for record in records:
        counts[classify_at_threshold(record, threshold)] += 1
    return counts


def replay_hit_rate(records, threshold: float) -> float:
    """Overall hit rate (exact + semantic) at ``threshold``; 0.0 if no records."""
    records = list(records)
    if not records:
        return 0.0
    counts = replay_counts(records, threshold)
    return (counts["exact"] + counts["semantic"]) / len(records)


def simulated_false_positive_risk(
    records: Iterable[dict],
    threshold: float,
    *,
    safe: float = 0.97,
    floor: float = 0.80,
) -> float:
    """Heuristic (NOT measured) false-positive risk for semantic hits at ``threshold``.

    A semantic match is treated as riskier the closer its score sits to the
    ``floor``; matches at/above ``safe`` carry ~no risk. Summed over the semantic
    hits at this threshold, it rises as the threshold drops — visualising why a
    lower threshold trades correctness for recall.
    """
    total = 0.0
    span = safe - floor
    for record in records:
        if classify_at_threshold(record, threshold) == "semantic":
            best = record.get("best_score") or 0.0
            risk = min(1.0, max(0.0, (safe - best) / span))
            total += risk
    return total


def metrics_router(cache: SemCache) -> APIRouter:
    """Build a router exposing read-only metrics for ``cache``."""
    router = APIRouter(tags=["metrics"])

    @router.get("/metrics")
    def get_metrics() -> dict:
        return {
            "hit_rate": cache.metrics.hit_rate(),
            "counts": cache.metrics.counts(),
            "savings": cache.metrics.savings(),
            "average_latency_ms": cache.metrics.average_latency_ms(),
            "entries": len(cache.store),
            "threshold": cache.config.threshold,
            "embedding_model": cache.config.embedding_model,
        }

    @router.get("/recent")
    def get_recent(n: int = 20) -> dict:
        return {"recent": cache.metrics.recent(n)}

    return router


def create_dashboard_app(cache: SemCache) -> FastAPI:
    """Standalone FastAPI app serving only the metrics routes for ``cache``."""
    app = FastAPI(title="semcache metrics API")
    app.include_router(metrics_router(cache))
    return app
