"""Seed a running semcache proxy with overlapping queries.

Gives the dashboard rich data to show — including near-miss `best_score`s that
make the threshold explorer interesting. Uses the proxy's keyless fake LLM, so
no GOOGLE_API_KEY is needed.

Usage:
    1.  make proxy                    # terminal 1 — starts the proxy on :8000
    2.  python examples/seed_proxy.py # terminal 2 — populate it
    3.  make dashboard                # terminal 3 — view at http://localhost:8501
"""
from __future__ import annotations

import sys

import requests

API = "http://localhost:8000"

# Overlapping research-style workload: exact repeats, close paraphrases (clear
# semantic hits), loose rephrasings (near-miss low scores), and unique topics.
WORKLOAD = [
    "What are the latest developments in large language models?",
    "What are the most recent developments in large language models?",
    "What are the latest developments in large language models?",      # exact
    "Summarize recent advances in large language models.",
    "What's going on with AI these days?",                              # loose -> miss
    "How is generative AI being used in healthcare?",
    "How is generative AI being used in the healthcare industry?",
    "How is generative AI being used in healthcare?",                  # exact
    "What is the current state of quantum computing?",
    "What is the current state of quantum computing today?",
    "What are the main risks of using AI in financial trading?",
    "How does CRISPR gene editing work?",
    "What caused the 2008 financial crisis?",
    "What are the latest trends in the electric vehicle market?",
    "What are the newest trends in the electric vehicle market?",
    "What is the impact of remote work on commercial real estate?",
    "How are central banks approaching interest rates in 2026?",
    "What are the latest trends in the electric vehicle market?",      # exact
]


def main() -> None:
    try:
        requests.get(f"{API}/metrics", timeout=5)
    except Exception:  # noqa: BLE001
        print(f"Proxy not reachable at {API}. Start it first:  make proxy")
        sys.exit(1)

    print(f"Seeding {len(WORKLOAD)} queries -> {API}\n")
    for query in WORKLOAD:
        resp = requests.post(
            f"{API}/v1/chat/completions",
            json={"model": "gemini-3.1-flash-lite",
                  "messages": [{"role": "user", "content": query}]},
            timeout=180,  # real research answers are long generations
        )
        print(f"  {resp.headers.get('x-semcache', '?'):<13} {query}")

    metrics = requests.get(f"{API}/metrics").json()
    counts = metrics["counts"]
    print(
        f"\nDone. Hit rate {metrics['hit_rate']['total'] * 100:.0f}% "
        f"({counts['exact']} exact, {counts['semantic']} semantic, {counts['miss']} miss). "
        f"Now run:  make dashboard"
    )


if __name__ == "__main__":
    main()
