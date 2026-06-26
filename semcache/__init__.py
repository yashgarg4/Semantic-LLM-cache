"""semcache — a dual-layer (exact + semantic) cache for LLM calls.

Two drop-in integration interfaces:

1. The ``@cached`` decorator — wrap any function whose first argument is the
   query string; repeated/paraphrased queries are served from cache::

       from semcache import cached

       @cached                      # default threshold (0.92)
       def ask(query: str) -> str:
           return call_my_llm(query)

       @cached(threshold=0.95)      # stricter, dedicated cache
       def ask_strict(query: str) -> str:
           ...

2. The OpenAI-compatible proxy in ``server/proxy.py`` (point any OpenAI client
   at it by changing ``base_url``).

The lower-level API (``SemCache``, ``CacheConfig``, ...) is also exported.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional

from .cache import CacheResult, SemCache
from .config import CacheConfig
from .embedder import Embedder
from .metrics import Metrics, estimate_cost, estimate_cost_split, estimate_tokens
from .store import CacheEntry, CacheStore, normalise_query

__version__ = "0.1.0"

# Default model used to price avoided calls when a decorated function does not
# report token usage of its own.
_DECORATOR_PRICING_MODEL = "gemini-3.1-flash-lite"

# Module-level caches shared by the decorator. The default (no threshold
# override) cache is created lazily so importing semcache does not load the
# embedding model until something is actually cached.
_default_cache: Optional[SemCache] = None
_threshold_caches: dict[float, SemCache] = {}


def get_default_cache() -> SemCache:
    """Return the process-wide default :class:`SemCache` (lazily created)."""
    global _default_cache
    if _default_cache is None:
        _default_cache = SemCache(CacheConfig())
    return _default_cache


def _resolve_cache(threshold: Optional[float]) -> SemCache:
    """Pick the cache for a given threshold: default, or a dedicated instance."""
    if threshold is None:
        return get_default_cache()
    if threshold not in _threshold_caches:
        _threshold_caches[threshold] = SemCache(CacheConfig(threshold=threshold))
    return _threshold_caches[threshold]


def cached(
    _fn: Optional[Callable] = None,
    *,
    threshold: Optional[float] = None,
    model: str = _DECORATOR_PRICING_MODEL,
):
    """Cache a function whose first positional argument is the query string.

    Usable bare (``@cached``) or parametrised (``@cached(threshold=0.95)``).
    A bare decorator (or any decorator without a ``threshold``) shares the
    module-level default cache; a ``threshold`` override gets its own cache.

    On a cache hit the wrapped function is **not** called — keep it free of
    important side effects. The wrapper returns the same value the function
    would have returned. The resolved cache is exposed as ``wrapper.cache`` for
    metrics inspection.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(query: str, *args, **kwargs):
            cache = _resolve_cache(threshold)
            wrapper.cache = cache  # type: ignore[attr-defined]

            def llm_fn(q: str):
                response = func(q, *args, **kwargs)
                text = response if isinstance(response, str) else str(response)
                tokens = estimate_tokens(text)
                return response, tokens, estimate_cost(model, tokens)

            result = cache.call(query, llm_fn=llm_fn)
            return result.response

        wrapper.cache = None  # type: ignore[attr-defined]  # set on first call
        return wrapper

    # Support both @cached and @cached(...).
    if _fn is not None and callable(_fn):
        return decorator(_fn)
    return decorator


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
    "estimate_cost_split",
    "estimate_tokens",
    "cached",
    "get_default_cache",
]
