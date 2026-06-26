"""semcache — a dual-layer (exact + semantic) cache for LLM calls.

Public API (Phase 1):

    from semcache import SemCache, CacheConfig

    cache = SemCache(CacheConfig(threshold=0.92), llm_fn=my_llm)
    result = cache.call("What is your return policy?")
    print(result.hit_type, result.score)

The ``@cached`` decorator and the OpenAI-compatible proxy arrive in later
phases and will be exported from here too.
"""
from .cache import CacheResult, SemCache
from .config import CacheConfig
from .embedder import Embedder
from .metrics import Metrics, estimate_cost
from .store import CacheEntry, CacheStore, normalise_query

__version__ = "0.1.0"

__all__ = [
    "SemCache",
    "CacheResult",
    "CacheConfig",
    "Embedder",
    "CacheStore",
    "CacheEntry",
    "normalise_query",
    "Metrics",
    "estimate_cost",
]
