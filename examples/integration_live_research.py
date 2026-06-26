"""Phase 5 integration measurement: semcache in front of live-research-intel.

This mirrors the cache layer wired into live-research-intel
(``backend/semcache_integration.py``): the same threshold (0.92) and the same
"cache the whole research report, keyed on the user's question" boundary.

It runs a batch of 20 *varied-but-overlapping* research questions — the kind a
research dashboard sees as users re-ask and rephrase — through the real cache
(real bge-small embeddings, real FAISS search). The **hit rate is genuinely
measured**. The expensive multi-agent pipeline is stubbed by a representative
per-run token figure, so the token/$ savings are an honest, clearly-labelled
*estimate* (we did not make 20 live Gemini/Tavily calls).

Run:  python examples/integration_live_research.py
"""
from __future__ import annotations

from semcache import CacheConfig, SemCache, estimate_cost
from server.dashboard import replay_hit_rate, simulated_false_positive_risk

# A full Searcher->Critic->Synthesizer run is input-heavy (search results in
# context). ~8k tokens/run is a conservative representative figure; the live
# integration uses the same SEMCACHE_TOKENS_PER_RUN default.
TOKENS_PER_RUN = 8_000
MODEL = "gemini-2.5-flash-lite"  # live-research-intel's model family

# 20 queries: clusters of repeats/paraphrases plus unique topics — a realistic
# overlapping research workload, NOT hand-tuned to inflate the hit rate.
WORKLOAD = [
    # -- LLM developments (asked several ways + an exact refresh) --
    "What are the latest developments in large language models?",
    "What are the most recent developments in large language models?",
    "What are the latest developments in large language models?",   # exact refresh
    "Summarize the latest developments in large language models.",
    # -- Generative AI in healthcare --
    "How is generative AI being used in healthcare?",
    "How is generative AI being used in the healthcare industry?",
    "What is the outlook for renewable energy investment in 2026?",
    # -- Quantum computing --
    "What is the current state of quantum computing?",
    "What is the current state of quantum computing today?",
    # -- Unique, one-off research topics (expected misses) --
    "What are the main risks of using AI in financial trading?",
    "How does CRISPR gene editing work?",
    "What caused the 2008 financial crisis?",
    # -- EV market (paraphrase pair) --
    "What are the latest trends in the electric vehicle market?",
    "What are the newest trends in the electric vehicle market?",
    # -- More repeats/refreshes of earlier topics --
    "How is generative AI being used in healthcare?",              # exact refresh
    "What is the current state of quantum computing?",             # exact refresh
    "Summarize recent advances in large language models.",
    # -- A few more unique topics --
    "What is the impact of remote work on commercial real estate?",
    "How are central banks approaching interest rates in 2026?",
    "What are the latest trends in the electric vehicle market?",  # exact refresh
]


def fake_research_run(question: str) -> tuple[str, int, float]:
    """Stand-in for the real multi-agent pipeline (no live API calls).

    Returns (report, tokens, cost) with a representative full-run token count so
    that a future cache hit credits the whole avoided pipeline, not just the
    answer text.
    """
    report = f"[synthesised research report for: {question}]"
    return report, TOKENS_PER_RUN, estimate_cost(MODEL, TOKENS_PER_RUN)


def main() -> None:
    cache = SemCache(CacheConfig(threshold=0.92), llm_fn=fake_research_run)

    print("=" * 92)
    print("semcache + live-research-intel: research-query cache (offline measurement)")
    print(f"threshold=0.92 | ~{TOKENS_PER_RUN:,} tokens/run | model={MODEL}")
    print("=" * 92)
    print(f"{'#':<3}{'research question':<62}{'outcome':<10}{'score'}")
    print("-" * 92)
    for i, question in enumerate(WORKLOAD, start=1):
        result = cache.call(question)
        score = f"{result.score:.4f}" if result.score is not None else "  -"
        print(f"{i:<3}{question[:60]:<62}{result.hit_type:<10}{score}")

    print("-" * 92)
    counts = cache.metrics.counts()
    hr = cache.metrics.hit_rate()
    savings = cache.metrics.savings()
    avoided = savings["calls_avoided"]
    print(
        f"MEASURED hit rate: {hr['total'] * 100:.0f}%  "
        f"({counts['exact']} exact, {counts['semantic']} semantic, {counts['miss']} miss "
        f"of {counts['total']})"
    )
    print(
        f"Research pipelines avoided: {avoided}/{counts['total']}  "
        f"-> ESTIMATED ~{savings['tokens_saved']:,} tokens and "
        f"${savings['cost_saved_usd']:.4f} saved on this batch."
    )
    # Projection makes the at-scale impact concrete without overclaiming.
    per_1k = (avoided / counts["total"]) * 1000 * TOKENS_PER_RUN
    print(
        f"At this hit rate, ~{per_1k / 1e6:.1f}M tokens "
        f"(${estimate_cost(MODEL, int(per_1k)):.2f}) saved per 1,000 research queries."
    )

    print("\nThreshold sensitivity (replaying this batch):")
    print(f"  {'threshold':<11}{'hit_rate':<10}{'fp_risk'}")
    for t in (0.85, 0.90, 0.92, 0.95, 0.98):
        recent = cache.metrics.recent(10_000)
        print(
            f"  {t:<11.2f}{replay_hit_rate(recent, t) * 100:<10.0f}"
            f"{simulated_false_positive_risk(recent, t):.2f}"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()
