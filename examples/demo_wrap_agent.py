"""Phase 3 demo: wrap an LLM-calling function with the @cached decorator.

The decorated ``ask()`` calls Gemini when ``GOOGLE_API_KEY`` is set; otherwise
it falls back to a local fake (with a simulated delay) so the decorator's
caching behaviour is demonstrable without an API key. Either way, repeated and
paraphrased queries are served from cache — the underlying function is not
called again.

Run:  python examples/demo_wrap_agent.py
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from semcache import cached, get_default_cache, normalise_query

load_dotenv()  # pick up GOOGLE_API_KEY from a local .env if present
USE_GEMINI = bool(os.getenv("GOOGLE_API_KEY"))
DEMO_MODEL = os.getenv("SEMCACHE_DEMO_MODEL", "gemini-3.1-flash-lite")

_CANNED = {
    "what is your return policy?": (
        "Return any item within 30 days for a full refund, unused and in its "
        "original packaging."
    ),
    "how do i reset my password?": (
        "Open Settings > Security > Reset password and follow the emailed link."
    ),
}

# Counts how many times the real work actually ran (i.e. cache misses).
_invocations = {"count": 0}


def _gemini_answer(query: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=DEMO_MODEL, temperature=0.3)
    return llm.invoke(query).content


def _fake_answer(query: str) -> str:
    time.sleep(0.4)  # simulate LLM latency so cache hits are visibly faster
    return _CANNED.get(normalise_query(query), f"(generic answer to: {query})")


@cached  # default threshold 0.92, uses the module-level default SemCache
def ask(query: str) -> str:
    """Answer a query — via Gemini if configured, else a local fake."""
    _invocations["count"] += 1
    return _gemini_answer(query) if USE_GEMINI else _fake_answer(query)


def main() -> None:
    cache = get_default_cache()
    mode = "Gemini (live)" if USE_GEMINI else "local fake (no GOOGLE_API_KEY)"
    print("=" * 84)
    print(f"semcache - Phase 3 @cached decorator demo   [LLM backend: {mode}]")
    print("=" * 84)

    queries = [
        "What is your return policy?",               # miss
        "What is your return policy?",               # exact hit
        "Can you explain your return policy?",       # semantic hit
        "How do I reset my password?",               # miss
        "I forgot my password, how do I reset it?",  # semantic hit
    ]

    print(f"{'#':<3}{'query':<46}{'hit_type':<10}{'latency':<10}")
    print("-" * 84)
    for i, q in enumerate(queries, start=1):
        t0 = time.perf_counter()
        answer = ask(q)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # The decorator returns only the answer; read the hit_type the cache
        # just recorded to show what happened.
        hit_type = cache.metrics.recent(1)[0]["hit_type"]
        print(f"{i:<3}{q:<46}{hit_type:<10}{dt_ms:7.1f}ms")
        print(f"     -> {answer[:60]}")

    print("-" * 84)
    counts = cache.metrics.counts()
    savings = cache.metrics.savings()
    hr = cache.metrics.hit_rate()
    print(
        f"Underlying function called {_invocations['count']} times for "
        f"{counts['total']} queries."
    )
    print(
        f"Hit rate: {hr['total'] * 100:.0f}% "
        f"({counts['exact']} exact, {counts['semantic']} semantic, "
        f"{counts['miss']} miss) - {savings['calls_avoided']} calls avoided."
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
