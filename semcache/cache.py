"""The dual-layer SemCache: exact match first, then semantic, then the LLM.

Lookup order is deliberate and mandatory:

1. **Exact** — hash/normalise the query and dict-lookup. Instant, free, zero
   false positives. Most semantic-cache implementations skip this and waste
   embedding compute on queries that are byte-for-byte repeats.
2. **Semantic** — only on an exact miss, embed the query once locally and run a
   cosine nearest-neighbour search; a hit requires ``score >= threshold``.
3. **Miss** — call the real ``llm_fn``, store query+embedding+response.
"""
from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict

from .config import CacheConfig
from .embedder import Embedder
from .store import CacheStore

# An llm_fn takes the query string and returns (response, tokens, cost).
LLMFn = Callable[[str], "tuple[str, int, float]"]


class CacheResult(BaseModel):
    """Outcome of a single cache lookup."""

    model_config = ConfigDict(frozen=True)

    hit_type: str  # "exact" | "semantic" | "miss"
    response: Optional[str]
    score: Optional[float]
    query: str
    matched_query: Optional[str] = None

    @property
    def is_hit(self) -> bool:
        return self.hit_type != "miss"


class SemCache:
    """Front a callable LLM with a dual-layer (exact + semantic) cache."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        llm_fn: Optional[LLMFn] = None,
    ) -> None:
        self.config = config or CacheConfig()
        self.embedder = Embedder(self.config.embedding_model)
        self.store = CacheStore(
            dim=self.embedder.dim,
            max_entries=self.config.max_entries,
            eviction_policy=self.config.eviction_policy,
        )
        self.llm_fn = llm_fn

    def _lookup(self, query: str) -> "tuple[CacheResult, Optional[object]]":
        """Run both layers. Returns (result, embedding).

        The embedding is returned so ``call()`` can reuse it to store a miss
        without embedding the same query twice. It is ``None`` only on an exact
        hit, where no embedding was computed.
        """
        # Layer 1: exact match — instant, free, zero false positives.
        entry = self.store.exact_get(query)
        if entry is not None:
            return (
                CacheResult(
                    hit_type="exact",
                    response=entry.response,
                    score=1.0,
                    query=query,
                    matched_query=entry.query,
                ),
                None,
            )

        # Layer 2: semantic match — embed once, cosine search via FAISS.
        embedding = self.embedder.embed(query)
        found = self.store.semantic_get(embedding, self.config.threshold)
        if found is not None:
            entry, score = found
            return (
                CacheResult(
                    hit_type="semantic",
                    response=entry.response,
                    score=score,
                    query=query,
                    matched_query=entry.query,
                ),
                embedding,
            )

        return (
            CacheResult(
                hit_type="miss",
                response=None,
                score=None,
                query=query,
                matched_query=None,
            ),
            embedding,
        )

    def get(self, query: str) -> CacheResult:
        """Look up a query against the cache without ever calling the LLM."""
        result, _ = self._lookup(query)
        return result

    def call(self, query: str) -> CacheResult:
        """Look up a query; on a full miss call ``llm_fn``, store, and return it."""
        result, embedding = self._lookup(query)
        if result.is_hit:
            return result
        if self.llm_fn is None:
            raise RuntimeError("cache miss and no llm_fn was configured")
        response, tokens, cost = self.llm_fn(query)
        if embedding is None:  # defensive; a miss always carries an embedding
            embedding = self.embedder.embed(query)
        self.store.add(
            query=query,
            embedding=embedding,
            response=response,
            tokens=tokens,
            cost=cost,
        )
        return CacheResult(
            hit_type="miss",
            response=response,
            score=None,
            query=query,
            matched_query=None,
        )
