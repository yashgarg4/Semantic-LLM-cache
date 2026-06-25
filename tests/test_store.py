"""Tests for the dual-layer store: exact dict, FAISS index, LRU eviction.

Uses small synthetic unit vectors instead of the real embedder so the store's
mechanics are tested deterministically and fast, independent of the model.
"""
from __future__ import annotations

import numpy as np

from semcache.store import CacheStore, normalise_query


def unit(vec) -> np.ndarray:
    """Return ``vec`` as a unit-length float32 array (cosine == inner product)."""
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_normalise_query_collapses_case_and_whitespace() -> None:
    assert normalise_query("  Hello   WORLD  ") == "hello world"
    assert normalise_query("What\tIS\nthis?") == "what is this?"
    assert normalise_query("Already normal") == "already normal"


def test_exact_get_hit_and_miss() -> None:
    store = CacheStore(dim=4)
    store.add("Hello World", unit([1, 0, 0, 0]), "resp", 10, 0.1)
    assert store.exact_get("hello world") is not None        # case-normalised
    assert store.exact_get("HELLO   WORLD") is not None       # ws-normalised
    assert store.exact_get("something else entirely") is None


def test_semantic_get_on_empty_index_returns_none() -> None:
    # The first-ever query must not crash on an empty FAISS index.
    store = CacheStore(dim=4)
    assert store.semantic_get(unit([1, 0, 0, 0]), 0.92) is None


def test_semantic_get_above_and_below_threshold() -> None:
    store = CacheStore(dim=4)
    store.add("a", unit([1, 0, 0, 0]), "respA", 10, 0.1)

    near = unit([0.99, 0.1414, 0, 0])      # cosine with [1,0,0,0] ~= 0.990
    found = store.semantic_get(near, 0.92)
    assert found is not None
    entry, score = found
    assert entry.response == "respA"
    assert score >= 0.92

    orthogonal = unit([0, 1, 0, 0])        # cosine 0 -> well below threshold
    assert store.semantic_get(orthogonal, 0.92) is None


def test_lru_eviction_drops_least_recently_used() -> None:
    store = CacheStore(dim=4, max_entries=2)
    store.add("first", unit([1, 0, 0, 0]), "r1", 1, 0.0)
    store.add("second", unit([0, 1, 0, 0]), "r2", 1, 0.0)

    # Touch "first" so "second" becomes the least-recently-used entry.
    assert store.exact_get("first") is not None

    # Over capacity -> evict the LRU entry ("second"), not merely the oldest.
    store.add("third", unit([0, 0, 1, 0]), "r3", 1, 0.0)

    assert len(store) == 2
    assert store.exact_get("second") is None          # evicted
    assert store.exact_get("first") is not None        # recently used -> survived
    assert store.exact_get("third") is not None        # just added

    # The FAISS index must be rebuilt consistently after eviction.
    found = store.semantic_get(unit([0, 0, 1, 0]), 0.92)
    assert found is not None and found[0].response == "r3"
