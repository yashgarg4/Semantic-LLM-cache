"""Metrics REST API consumed by the Streamlit dashboard (Phase 4).

Exposes read-only views over a :class:`~semcache.SemCache`'s metrics:

* ``GET /metrics`` — hit rate, counts, savings, latency, threshold, entries.
* ``GET /recent?n=20`` — the last N lookups.

The routes are provided as a router factory so the proxy can mount them over
*its* cache (single process, shared in-memory state), and a standalone app
factory is provided for running the dashboard API on its own.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI

from semcache import SemCache


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
