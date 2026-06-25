"""Dual-layer cache storage: exact-match dict + FAISS semantic index.

* **Layer 1 — ``_exact``**: maps the *normalised* query string to its entry.
  O(1), free, zero false positives.
* **Layer 2 — FAISS ``IndexFlatIP``**: an inner-product index over
  unit-normalised embeddings, so inner product equals cosine similarity. A
  parallel ``_entries`` list keeps cache entries aligned to FAISS row indices.

Eviction is LRU: every access stamps a monotonically increasing sequence
number, and when the store exceeds ``max_entries`` the lowest-stamped entry is
dropped and the flat index rebuilt. Rebuilding a flat index is just a matrix
copy, and eviction is comparatively rare, so this stays cheap and keeps the
``_entries`` <-> FAISS alignment trivially correct.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

import faiss
import numpy as np

_WHITESPACE_RE = re.compile(r"\s+")


def normalise_query(query: str) -> str:
    """Normalise a query for exact matching: lowercase, strip, collapse ws."""
    return _WHITESPACE_RE.sub(" ", query.strip().lower())


@dataclass
class CacheEntry:
    """A single cached query/response pair plus bookkeeping metadata."""

    query: str
    normalized_query: str
    response: str
    tokens: int
    cost: float
    embedding: np.ndarray
    created_at: float = field(default_factory=time.time)
    last_access_seq: int = 0
    hits: int = 0


class CacheStore:
    """Holds the exact-match dict and the FAISS semantic index."""

    def __init__(
        self,
        dim: int,
        max_entries: int = 1000,
        eviction_policy: str = "lru",
    ) -> None:
        if eviction_policy != "lru":
            raise ValueError(f"unsupported eviction_policy: {eviction_policy!r}")
        self.dim = dim
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy
        self._exact: dict[str, CacheEntry] = {}
        self._entries: list[CacheEntry] = []
        self._index = faiss.IndexFlatIP(dim)
        self._seq = count(1)  # monotonic clock for LRU ordering

    def __len__(self) -> int:
        return len(self._entries)

    # -- internal helpers -------------------------------------------------

    def _touch(self, entry: CacheEntry) -> None:
        """Mark an entry as just-used (LRU) and count the hit."""
        entry.last_access_seq = next(self._seq)
        entry.hits += 1

    def _rebuild_index(self) -> None:
        """Rebuild the flat index from the current ``_entries`` order."""
        self._index = faiss.IndexFlatIP(self.dim)
        if self._entries:
            matrix = np.vstack([e.embedding for e in self._entries]).astype(np.float32)
            self._index.add(matrix)

    def _evict_lru(self) -> None:
        """Drop the least-recently-used entry and rebuild the index."""
        victim_pos = min(
            range(len(self._entries)),
            key=lambda i: self._entries[i].last_access_seq,
        )
        victim = self._entries.pop(victim_pos)
        self._exact.pop(victim.normalized_query, None)
        self._rebuild_index()

    # -- public API -------------------------------------------------------

    def add(
        self,
        query: str,
        embedding: np.ndarray,
        response: str,
        tokens: int,
        cost: float,
    ) -> CacheEntry:
        """Insert a new entry. Called only after a full cache miss."""
        norm = normalise_query(query)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape[0] != self.dim:
            raise ValueError(
                f"embedding dim {embedding.shape[0]} != index dim {self.dim}"
            )

        # add() runs after a miss, so ``norm`` should be new. If a duplicate
        # slips through (e.g. concurrent callers), refresh in place rather than
        # inserting a second FAISS row for the same normalised query.
        existing = self._exact.get(norm)
        if existing is not None:
            existing.response = response
            existing.tokens = tokens
            existing.cost = cost
            self._touch(existing)
            return existing

        entry = CacheEntry(
            query=query,
            normalized_query=norm,
            response=response,
            tokens=tokens,
            cost=cost,
            embedding=embedding,
        )
        entry.last_access_seq = next(self._seq)
        self._exact[norm] = entry
        self._entries.append(entry)
        self._index.add(embedding.reshape(1, -1))

        if len(self._entries) > self.max_entries:
            self._evict_lru()
        return entry

    def exact_get(self, query: str) -> Optional[CacheEntry]:
        """Layer 1: exact, normalised-string lookup. Free and instant."""
        entry = self._exact.get(normalise_query(query))
        if entry is not None:
            self._touch(entry)
        return entry

    def semantic_get(
        self, embedding: np.ndarray, threshold: float
    ) -> Optional[tuple[CacheEntry, float]]:
        """Layer 2: nearest-neighbour cosine search. Hit iff score >= threshold.

        Returns ``None`` on an empty index (the first-ever query is always a
        miss and must not crash).
        """
        if self._index.ntotal == 0:
            return None
        q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        scores, idxs = self._index.search(q, 1)
        idx = int(idxs[0][0])
        score = float(scores[0][0])
        if idx < 0:  # FAISS sentinel when no neighbour exists
            return None
        if score >= threshold:
            entry = self._entries[idx]
            self._touch(entry)
            return entry, score
        return None

    def entries(self) -> list[CacheEntry]:
        """Return a shallow copy of cached entries (for metrics/inspection)."""
        return list(self._entries)
