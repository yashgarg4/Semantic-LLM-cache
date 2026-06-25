"""Tests for the local embedding wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from semcache import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embed_shape_and_dtype(embedder: Embedder) -> None:
    vec = embedder.embed("hello world")
    assert vec.shape == (embedder.dim,)
    assert vec.dtype == np.float32


def test_embed_is_unit_normalised(embedder: Embedder) -> None:
    # Cosine == inner product only holds for unit vectors; this is the
    # invariant the whole FAISS layer relies on.
    vec = embedder.embed("some arbitrary sentence to embed")
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)


def test_embed_batch_shape(embedder: Embedder) -> None:
    matrix = embedder.embed_batch(["a", "b", "c"])
    assert matrix.shape == (3, embedder.dim)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_embed_batch_empty_does_not_crash(embedder: Embedder) -> None:
    matrix = embedder.embed_batch([])
    assert matrix.shape == (0, embedder.dim)


def test_paraphrase_scores_higher_than_unrelated(embedder: Embedder) -> None:
    base = embedder.embed("How do I reset my password?")
    paraphrase = embedder.embed("How can I reset my password?")
    unrelated = embedder.embed("What is the capital of France?")
    assert float(base @ paraphrase) > float(base @ unrelated)
    # Near-identical paraphrase should clear the default factual threshold.
    assert float(base @ paraphrase) >= 0.92
