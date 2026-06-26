"""Phase 1 standalone demo of the dual-layer cache.

Runs six queries against a *fake* LLM (no API key needed) and prints the hit
type and similarity score for each:

* 2 novel queries  -> miss   (populate the cache)
* 2 identical repeats -> exact hit  (Layer 1, score 1.0)
* 2 paraphrases    -> semantic hit  (Layer 2, score >= threshold)

Run:  python examples/demo_basic.py
"""
from __future__ import annotations

from semcache import CacheConfig, SemCache, estimate_cost, normalise_query

# The "model" our fake LLM pretends to be, used to price avoided calls.
DEMO_MODEL = "gemini-3.1-flash-lite"

# Canned "knowledge base" so the fake LLM returns sensible text for the two
# base queries. Anything else gets a generic answer.
CANNED = {
    "what is your return policy?": (
        "You can return any item within 30 days of delivery for a full refund, "
        "provided it is unused and in its original packaging."
    ),
    "how do i reset my password?": (
        "Go to Settings > Security > Reset password, enter your email, and "
        "follow the link we send you."
    ),
}


def fake_llm(query: str) -> tuple[str, int, float]:
    """Stand-in for a real LLM call: returns (response, tokens, cost)."""
    response = CANNED.get(
        normalise_query(query), f"(generic answer to: {query})"
    )
    tokens = 200  # pretend prompt + completion came to ~200 tokens
    cost = estimate_cost(DEMO_MODEL, tokens)
    return response, tokens, cost


def fmt_score(result) -> str:
    # ASCII-only so it prints cleanly on any console (incl. Windows cp1252).
    return "  -   " if result.score is None else f"{result.score:.4f}"


def main() -> None:
    config = CacheConfig(threshold=0.92)
    cache = SemCache(config=config, llm_fn=fake_llm)

    print("=" * 78)
    print("semcache - Phase 1 dual-layer cache demo")
    print(f"embedding model : {config.embedding_model} (dim={cache.embedder.dim})")
    print(f"threshold       : {config.threshold}")
    print("=" * 78)

    # (query, expected hit type) — expectation is only for the printed label.
    queries = [
        ("What is your return policy?", "miss"),
        ("How do I reset my password?", "miss"),
        ("What is your return policy?", "exact"),
        ("How do I reset my password?", "exact"),
        ("Can you explain your return policy?", "semantic"),
        ("I forgot my password, how do I reset it?", "semantic"),
    ]

    print(f"{'#':<3}{'query':<46}{'hit_type':<10}{'score':<8}")
    print("-" * 78)
    tally = {"exact": 0, "semantic": 0, "miss": 0}
    for i, (q, _expected) in enumerate(queries, start=1):
        result = cache.call(q)
        tally[result.hit_type] += 1
        print(f"{i:<3}{q:<46}{result.hit_type:<10}{fmt_score(result):<8}")
        if result.hit_type == "semantic":
            print(f"     -> matched cached query: {result.matched_query!r}")

    print("-" * 78)

    # --- Phase 2: metrics summary -------------------------------------------
    hr = cache.metrics.hit_rate()
    counts = cache.metrics.counts()
    savings = cache.metrics.savings()
    print(
        f"Hit rate: {hr['total'] * 100:.0f}% "
        f"({counts['exact']} exact, {counts['semantic']} semantic, "
        f"{counts['miss']} miss) - {savings['calls_avoided']} calls avoided, "
        f"~{savings['tokens_saved']} tokens saved, "
        f"${savings['cost_saved_usd']:.6f} saved"
    )
    print(f"Cache holds {len(cache.store)} entries.")
    print("=" * 78)


if __name__ == "__main__":
    main()
