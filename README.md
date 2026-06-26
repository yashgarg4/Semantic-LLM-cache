# semcache

**A dual-layer (exact + semantic) cache for LLM calls.**

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![tests](https://img.shields.io/badge/tests-31%20passing-brightgreen)
![embeddings](https://img.shields.io/badge/embeddings-BGE--small%20(local)-orange)
![vector%20search](https://img.shields.io/badge/vector%20search-FAISS-informational)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

semcache sits in front of any LLM and avoids redundant calls by recognising when
a new query is *semantically* the same as one already answered — not just
byte-for-byte identical. It checks an instant exact-match layer first, then falls
back to a local-embedding cosine search over past queries (via FAISS), and only
calls the real model on a true miss. The embedding model runs locally and is
small by design, so the cache check is always cheaper than the call it avoids. It
ships with a `@cached` decorator, an OpenAI-compatible proxy, a metrics dashboard,
and an honest account of where semantic caching helps and where it does not.

---

## Features

| Feature | What it does |
|---|---|
| **Dual-layer lookup** | Exact (hash/dict) first — instant, free, zero false positives — then semantic (FAISS cosine) only on an exact miss. |
| **Local embeddings** | `BAAI/bge-small-en-v1.5` (384-dim) runs locally; the cache check never makes a remote embedding API call. |
| **Configurable threshold** | Default `0.92`; warns below the `0.90` safe floor for factual workloads. |
| **LRU eviction** | Bounded memory (`max_entries`), least-recently-used dropped first. |
| **Metrics + cost tracking** | Hit rate, calls avoided, tokens & USD saved, per-lookup latency. |
| **`@cached` decorator** | Wrap any `fn(query, ...)`; repeated/paraphrased calls served from cache. |
| **OpenAI-compatible proxy** | Point any OpenAI client at it via `base_url`; `x-semcache` header reports the outcome. |
| **Metrics dashboard** | Streamlit KPIs, outcome split, savings, and a threshold explorer that replays history. |

---

## Architecture

```
                    query
                      │
              ┌───────▼────────┐
              │ normalise(query)│  lowercase · trim · collapse whitespace
              └───────┬────────┘
                      │
        ┌─────────────▼──────────────┐   HIT
        │ Layer 1 — exact dict lookup │──────────► cached response  (score 1.0)
        └─────────────┬──────────────┘
                      │ miss
        ┌─────────────▼──────────────┐
        │ embed locally (bge-small)   │
        │ Layer 2 — FAISS cosine top-1│──────────► cached response  (score ≥ threshold)
        └─────────────┬──────────────┘   HIT
                      │ miss
        ┌─────────────▼──────────────┐
        │ call real LLM · store       │──────────► fresh response
        │ (query + embedding + answer)│            (next repeat/paraphrase hits)
        └─────────────────────────────┘

   Integration surfaces:  @cached decorator   ·   OpenAI-compatible proxy   ·   metrics dashboard
```

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) | Small, fast, local — must be cheaper than the avoided call. |
| Vector search | `faiss-cpu` (`IndexFlatIP`) | Exact inner-product = cosine on unit vectors; fast at cache scale. |
| Vector ops | `numpy` | Normalisation, matrix handling. |
| Config / models | `pydantic` v2 | Validation, fail-fast on misconfiguration. |
| Proxy | `fastapi` + `uvicorn` | OpenAI-compatible HTTP endpoint. |
| Dashboard | `streamlit` | Metrics + threshold explorer UI. |
| Demo LLM | `langchain-google-genai` | Gemini for the proxy / wrap demos. |
| Tests | `pytest` | 31 tests across cache, store, embedder, metrics, proxy, dashboard. |

---

## Quick start

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install (pinned deps + editable package)
pip install -r requirements.txt
pip install -e .

# 3. Run the core demo (no API key needed — uses a fake LLM)
python examples/demo_basic.py
```

Expected: 2 exact hits, 2 semantic hits (scores ≥ 0.92), 2 misses, plus a metrics
summary. The first run downloads the ~130 MB embedding model once.

---

## When to use it — and when not to

Semantic caching pays off **only when queries genuinely repeat in meaning**. Be
honest about workload fit:

| Workload | Expected hit rate | Fit |
|---|---|---|
| FAQ / customer support | 40–65% | ✅ Great |
| Agent tool-call prompts | 40–65% | ✅ Great |
| Repeated research / dashboards | moderate–high | ✅ Good |
| Creative generation ("write a poem about X") | ~0% | ❌ Don't |
| Multi-turn chat (context-dependent turns) | ~0% | ❌ Don't |

For creative or context-dependent work, near-identical prompts legitimately want
*different* answers — cache hits there would be wrong. A cached **wrong** answer
is worse than a missed cache hit, which is why the default threshold (`0.92`)
biases toward precision.

---

## Integration

### 1. The `@cached` decorator

```python
from semcache import cached

@cached                       # default threshold (0.92), shared default cache
def answer(query: str) -> str:
    return call_my_llm(query)

@cached(threshold=0.95)       # stricter, gets its own cache
def answer_strict(query: str) -> str:
    ...

answer("What is your return policy?")          # miss -> calls the LLM
answer("Can you explain your return policy?")  # semantic hit -> served from cache
```

On a hit the wrapped function is **not** called, so keep it free of important
side effects (it's for "answer this query" functions, not actions).

### 2. The OpenAI-compatible proxy

```bash
make proxy        # uvicorn server.proxy:app  (also serves /metrics and /recent)
```

Point any OpenAI client at it by changing `base_url` to `http://localhost:8000/v1`.
Each response carries `x-semcache: hit-exact | hit-semantic | miss`. On a miss it
calls Gemini (set `GOOGLE_API_KEY`); without a key it serves a clearly-labelled
fake LLM so the proxy and dashboard are runnable with zero setup.

### 3. Measured integration — `live-research-intel`

semcache was wired into a real LangGraph multi-agent research app at the
**research-question → final-report** boundary (`backend/semcache_integration.py`
+ a guarded hook in `main.py`): a repeated or paraphrased question is served from
the cached report, skipping the entire Searcher → Critic → Synthesizer pipeline.

On a 20-query overlapping research workload (`examples/integration_live_research.py`):

```
MEASURED hit rate: 50%  (4 exact, 6 semantic, 10 miss of 20)
Research pipelines avoided: 10/20
```

The hit rate is genuinely measured (real bge-small embeddings, 0.92 threshold);
at a representative ~8k tokens per research run that is an estimated **~4.0M
tokens (~$0.40) saved per 1,000 research queries**. Paraphrases such as *"What
are the most recent developments in LLMs?"* matched *"What are the latest
developments in large language models?"* at cosine 0.99.

---

## Project structure

```
semcache/
├── semcache/
│   ├── __init__.py      # SemCache, @cached decorator, default-cache helpers
│   ├── cache.py         # dual-layer lookup (get / call / put), CacheResult
│   ├── embedder.py      # local sentence-transformers wrapper (unit-normalised)
│   ├── store.py         # FAISS index + exact-match dict + LRU eviction
│   ├── metrics.py       # hit rate, token/cost savings, cost table
│   └── config.py        # CacheConfig (threshold, model, eviction)
├── server/
│   ├── proxy.py         # OpenAI-compatible FastAPI proxy
│   └── dashboard.py     # metrics REST API + threshold-replay functions
├── dashboard/
│   └── app.py           # Streamlit dashboard
├── examples/
│   ├── demo_basic.py              # core cache + metrics demo
│   ├── demo_wrap_agent.py         # @cached decorator demo
│   └── integration_live_research.py  # measured integration batch
├── tests/               # 31 tests (cache, store, embedder, metrics, proxy, dashboard)
├── requirements.txt · pyproject.toml · Makefile · INTERNAL_NOTES.md
```

---

## API reference

| API | Description |
|---|---|
| `SemCache(config=None, llm_fn=None)` | Dual-layer cache fronting `llm_fn(query) -> (response, tokens, cost)`. |
| `SemCache.get(query)` | Look up without ever calling the LLM → `CacheResult`. |
| `SemCache.call(query, llm_fn=None)` | Look up; on a miss call the LLM, store, record metrics. |
| `SemCache.put(query, response, *, tokens=None, cost=None, model=...)` | Insert a precomputed answer. |
| `SemCache.metrics` | `Metrics`: `hit_rate()`, `counts()`, `savings()`, `recent(n)`. |
| `@cached` / `@cached(threshold=...)` | Decorator over `fn(query, ...)`. |
| `CacheResult` | `.hit_type`, `.response`, `.score`, `.matched_query`, `.tokens`, `.cost`, `.is_hit`. |
| `POST /v1/chat/completions` | OpenAI-compatible proxy endpoint; sets `x-semcache`. |
| `GET /metrics`, `GET /recent?n=` | Metrics REST API (consumed by the dashboard). |

---

## Configuration

`CacheConfig` fields (and the env vars `CacheConfig.from_env()` reads):

| Field | Env var | Default | Notes |
|---|---|---|---|
| `threshold` | `SEMCACHE_THRESHOLD` | `0.92` | Cosine hit threshold; warns below 0.90. |
| `embedding_model` | `SEMCACHE_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers model. |
| `max_entries` | `SEMCACHE_MAX_ENTRIES` | `1000` | LRU eviction beyond this. |
| `eviction_policy` | — | `lru` | Currently LRU only. |

Other env vars:

| Env var | Used by | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | proxy / demos | Real Gemini calls (optional; fake fallback otherwise). |
| `SEMCACHE_API` | dashboard | Base URL of the metrics API (default `http://localhost:8000`). |
| `SEMCACHE_ENABLED` | live-research-intel hook | Turn the report cache on. |
| `SEMCACHE_TOKENS_PER_RUN` | live-research-intel hook | Representative tokens per full research run. |

---

## Running tests

```bash
make test        # or: pytest -q
```

31 tests cover exact/semantic/miss behaviour, LRU eviction, query normalisation,
empty-index safety, unit-norm embeddings, metrics math, the proxy's `x-semcache`
header, and the dashboard's threshold-replay logic.

---

## Make targets

| Target | Action |
|---|---|
| `make install` | Install pinned deps + editable package. |
| `make demo` | Core cache + metrics demo. |
| `make demo-wrap` | `@cached` decorator demo. |
| `make proxy` | Run the OpenAI-compatible proxy (also serves `/metrics`). |
| `make dashboard` | Run the Streamlit dashboard. |
| `make test` | Run the test suite. |

---

## Notes

The embedding model on the critical path is small and local on purpose: a remote
embedding API would reintroduce the very network latency the cache exists to
remove. Embeddings are unit-normalised so a FAISS inner-product search equals
cosine similarity (asserted in the embedder). See `INTERNAL_NOTES.md` for the full
design rationale, the per-phase build log, known limitations, and interview Q&A.
