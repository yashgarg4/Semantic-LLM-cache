"""Local embedding model wrapper.

The embedder is the only model on the cache's critical path, so it must be
small, local and fast — never a remote API (a remote embedding call would cost
as much latency as the LLM call we are trying to avoid). The model is loaded
once per process (cached) and always returns unit-normalised float32 vectors,
so a FAISS inner-product search is exactly cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and process-wide cache a sentence-transformers model.

    ``lru_cache`` ensures repeated ``Embedder(...)`` constructions with the
    same model id share one set of in-memory weights.
    """
    return SentenceTransformer(model_name)


class Embedder:
    """Wraps a sentence-transformers model and yields normalised vectors."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)
        # sentence-transformers 5.x renamed this; support both names.
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = int(self._model.get_embedding_dimension())
        else:  # pragma: no cover - older sentence-transformers
            self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        """Embedding dimensionality (384 for bge-small-en-v1.5)."""
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a 1-D, unit-normalised float32 vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings into a 2-D ``(n, dim)`` float32 matrix."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        # Cosine == inner product ONLY for unit vectors. Assert it here so the
        # FAISS layer can rely on it (coding standard).
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"embeddings are not unit-normalised "
            f"(norms range {norms.min():.4f}..{norms.max():.4f})"
        )
        return vectors
