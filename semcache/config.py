"""Configuration model for semcache.

``CacheConfig`` is a pydantic v2 model holding every tunable knob of the cache.
It is intentionally small and fully validated so that misconfiguration fails
fast and loudly rather than silently degrading recall/precision.
"""
from __future__ import annotations

import os
import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Below this cosine threshold, semantic matching starts returning wrong answers
# for factual workloads (per production research). We don't forbid it, but we
# warn — the threshold must stay configurable and visible.
MIN_SAFE_THRESHOLD = 0.90


class CacheConfig(BaseModel):
    """Tunable configuration for a :class:`~semcache.cache.SemCache` instance."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity at/above which a semantic match is a hit.",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="sentence-transformers model id used for local embeddings.",
    )
    max_entries: int = Field(
        default=1000,
        gt=0,
        description="Maximum cached entries kept before LRU eviction begins.",
    )
    eviction_policy: Literal["lru"] = Field(
        default="lru",
        description="Policy applied when max_entries is exceeded.",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Unit-normalise embeddings so FAISS inner product == cosine.",
    )

    @field_validator("threshold")
    @classmethod
    def _warn_if_threshold_low(cls, v: float) -> float:
        if v < MIN_SAFE_THRESHOLD:
            warnings.warn(
                f"threshold={v:.2f} is below the safe floor of "
                f"{MIN_SAFE_THRESHOLD:.2f}; semantic matches may return wrong "
                f"answers for factual workloads.",
                stacklevel=2,
            )
        return v

    @classmethod
    def from_env(cls, **overrides) -> "CacheConfig":
        """Build a config from ``SEMCACHE_*`` environment variables.

        Precedence: explicit ``overrides`` > environment > field defaults.
        Unset variables simply fall through to the defaults. pydantic coerces
        the string env values into the declared field types.
        """
        env_map = {
            "threshold": os.getenv("SEMCACHE_THRESHOLD"),
            "embedding_model": os.getenv("SEMCACHE_EMBEDDING_MODEL"),
            "max_entries": os.getenv("SEMCACHE_MAX_ENTRIES"),
        }
        values = {k: v for k, v in env_map.items() if v is not None}
        values.update(overrides)
        return cls(**values)
