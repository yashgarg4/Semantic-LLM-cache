"""Metrics and cost tracking for the cache.

Every lookup is recorded so we can report:

* **hit rate** — exact / semantic / overall / miss, as fractions;
* **savings** — tokens, USD, and LLM calls avoided by serving from cache;
* **recent** — the last N lookups, for inspection and the dashboard.

Cost is estimated from a small per-model table of (input, output) USD rates
(approximate published Gemini rates). Prefer ``estimate_cost_split`` when you
have the input/output token split (providers bill them differently);
``estimate_cost`` is a rough total-only fallback.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

# Approximate published Gemini rates in USD per 1,000,000 tokens, as
# (input_rate, output_rate). Output tokens are billed several times higher than
# input. Deliberately approximate (pricing changes) and trivial to update.
GEMINI_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.5-flash": (1.50, 9.00),  # standard paid tier (free tier is $0)
    "gemini-3.1-flash-lite": (0.25, 1.50),  # standard paid tier, text (free tier $0)
}
DEFAULT_MODEL = "gemini-1.5-flash"


def _rates(model: str) -> tuple[float, float]:
    """(input, output) USD-per-1M rates for ``model`` (unknown -> default)."""
    return GEMINI_PRICING_USD_PER_1M.get(model, GEMINI_PRICING_USD_PER_1M[DEFAULT_MODEL])


def estimate_cost_split(model: str, input_tokens: int, output_tokens: int) -> float:
    """Accurate USD cost from separate input/output token counts.

    Preferred whenever the split is known (e.g. from an LLM's usage metadata),
    since providers bill input and output tokens at different rates.
    """
    in_rate, out_rate = _rates(model)
    return input_tokens / 1_000_000 * in_rate + output_tokens / 1_000_000 * out_rate


def estimate_cost(model: str, tokens: int) -> float:
    """Rough USD cost for a TOTAL token count when the input/output split is
    unknown — applies the average of the input and output rates. Prefer
    :func:`estimate_cost_split` whenever the split is available.
    """
    in_rate, out_rate = _rates(model)
    return tokens / 1_000_000 * (in_rate + out_rate) / 2


def estimate_tokens(text: str) -> int:
    """Roughly estimate token count from text (~4 chars/token for English).

    Used to value cache savings when the wrapped function or LLM response does
    not report exact token usage. Approximate by design.
    """
    return max(1, len(text) // 4)


@dataclass
class LookupRecord:
    """One recorded cache lookup."""

    query: str
    hit_type: str  # "exact" | "semantic" | "miss"
    score: Optional[float]  # the match score for hits (1.0 exact, cosine semantic)
    best_score: Optional[float]  # top neighbour cosine, recorded even on a miss
    matched_query: Optional[str]
    tokens_saved: int
    cost_saved: float
    latency_ms: float
    timestamp: float


class Metrics:
    """Accumulates lookup records and computes hit rate + savings."""

    def __init__(self) -> None:
        self._records: list[LookupRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def record(
        self,
        *,
        query: str,
        hit_type: str,
        score: Optional[float],
        matched_query: Optional[str],
        tokens_saved: int,
        cost_saved: float,
        latency_ms: float,
        best_score: Optional[float] = None,
    ) -> None:
        """Append one lookup to the log."""
        self._records.append(
            LookupRecord(
                query=query,
                hit_type=hit_type,
                score=score,
                best_score=best_score,
                matched_query=matched_query,
                tokens_saved=tokens_saved,
                cost_saved=cost_saved,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )
        )

    def counts(self) -> dict[str, int]:
        """Raw counts: ``{exact, semantic, miss, total}``."""
        exact = sum(1 for r in self._records if r.hit_type == "exact")
        semantic = sum(1 for r in self._records if r.hit_type == "semantic")
        miss = sum(1 for r in self._records if r.hit_type == "miss")
        return {
            "exact": exact,
            "semantic": semantic,
            "miss": miss,
            "total": len(self._records),
        }

    def hit_rate(self) -> dict[str, float]:
        """Fractions (0..1): ``{exact, semantic, total, miss}``.

        ``total`` is the overall hit rate (exact + semantic). An empty log
        returns all zeros rather than dividing by zero.
        """
        c = self.counts()
        total = c["total"]
        if total == 0:
            return {"exact": 0.0, "semantic": 0.0, "total": 0.0, "miss": 0.0}
        return {
            "exact": c["exact"] / total,
            "semantic": c["semantic"] / total,
            "total": (c["exact"] + c["semantic"]) / total,
            "miss": c["miss"] / total,
        }

    def savings(self) -> dict[str, float]:
        """Accumulated savings: ``{tokens_saved, cost_saved_usd, calls_avoided}``."""
        return {
            "tokens_saved": sum(r.tokens_saved for r in self._records),
            "cost_saved_usd": sum(r.cost_saved for r in self._records),
            "calls_avoided": sum(1 for r in self._records if r.hit_type != "miss"),
        }

    def average_latency_ms(self) -> dict[str, float]:
        """Average lookup latency split by hit vs miss (0.0 if none)."""
        hits = [r.latency_ms for r in self._records if r.hit_type != "miss"]
        misses = [r.latency_ms for r in self._records if r.hit_type == "miss"]
        return {
            "hit": sum(hits) / len(hits) if hits else 0.0,
            "miss": sum(misses) / len(misses) if misses else 0.0,
        }

    def recent(self, n: int = 10) -> list[dict]:
        """Return the last ``n`` lookups as dicts (most recent last)."""
        return [asdict(r) for r in self._records[-n:]]
