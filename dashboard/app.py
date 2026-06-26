"""Streamlit metrics dashboard for semcache.

Reads the REST API exposed by ``server/dashboard.py`` (mounted on the proxy):
``GET /metrics`` and ``GET /recent``. Shows KPI cards, the exact/semantic/miss
split, cumulative savings, a recent-lookups table, and a **threshold explorer**
that replays recorded history at any threshold to visualise the precision/recall
tradeoff.

Run (with the proxy already running on :8000):
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from server.dashboard import (
    replay_counts,
    replay_hit_rate,
    simulated_false_positive_risk,
)

st.set_page_config(page_title="semcache dashboard", layout="wide")

API = st.sidebar.text_input(
    "Metrics API base URL",
    os.getenv("SEMCACHE_API", "http://localhost:8000"),
).rstrip("/")

st.title("semcache — semantic cache dashboard")

# --- Fetch data ---------------------------------------------------------------
try:
    metrics = requests.get(f"{API}/metrics", timeout=5).json()
    records = requests.get(f"{API}/recent", params={"n": 100000}, timeout=5).json()["recent"]
except Exception as exc:  # noqa: BLE001 - surface any connection problem to the user
    st.error(
        f"Could not reach the metrics API at {API}.\n\n"
        f"Start the proxy first (`make proxy`), send it some queries, then "
        f"reload.\n\nDetails: {exc}"
    )
    st.stop()

counts = metrics["counts"]
hit_rate = metrics["hit_rate"]
savings = metrics["savings"]
live_threshold = float(metrics.get("threshold", 0.92))

# --- KPI cards ----------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Hit rate", f"{hit_rate['total'] * 100:.0f}%")
k2.metric("Calls avoided", f"{savings['calls_avoided']:,}")
k3.metric("Tokens saved", f"{savings['tokens_saved']:,}")
k4.metric("$ saved", f"${savings['cost_saved_usd']:.4f}")
st.caption(
    f"{counts['total']} lookups · {metrics['entries']} cached entries · "
    f"threshold {live_threshold:.2f} · model {metrics.get('embedding_model', '?')}"
)

left, right = st.columns(2)

# --- Pie: exact vs semantic vs miss -------------------------------------------
with left:
    st.subheader("Outcome split")
    split_df = pd.DataFrame(
        {
            "outcome": ["exact", "semantic", "miss"],
            "count": [counts["exact"], counts["semantic"], counts["miss"]],
        }
    )
    pie = (
        alt.Chart(split_df)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("outcome:N", scale=alt.Scale(scheme="set2")),
            tooltip=["outcome", "count"],
        )
    )
    st.altair_chart(pie, use_container_width=True)

# --- Cumulative cost saved over lookups ---------------------------------------
with right:
    st.subheader("Cumulative $ saved")
    if records:
        running = 0.0
        rows = []
        for i, record in enumerate(records, start=1):
            running += record.get("cost_saved", 0.0) or 0.0
            rows.append({"lookup": i, "cumulative_usd_saved": running})
        st.line_chart(pd.DataFrame(rows).set_index("lookup"))
    else:
        st.info("No lookups recorded yet.")

# --- Threshold explorer -------------------------------------------------------
st.subheader("Threshold explorer — precision/recall tradeoff")
st.caption(
    "Replays the recorded lookups at a chosen threshold (no re-embedding). "
    "Lower threshold → more semantic hits (higher recall) but higher simulated "
    "false-positive risk (lower precision)."
)
threshold = st.slider("threshold", 0.80, 0.99, value=live_threshold, step=0.01)

replayed = replay_counts(records, threshold)
e1, e2, e3 = st.columns(3)
e1.metric(f"Hit rate @ {threshold:.2f}", f"{replay_hit_rate(records, threshold) * 100:.0f}%")
e2.metric("Semantic hits", replayed["semantic"])
e3.metric("Sim. false-positive risk", f"{simulated_false_positive_risk(records, threshold):.2f}")

# Sweep the whole range so the tradeoff curve is visible, marking the slider.
sweep = []
step = 0.80
while step <= 0.991:
    sweep.append(
        {
            "threshold": round(step, 2),
            "hit_rate_%": replay_hit_rate(records, step) * 100,
            "fp_risk": simulated_false_positive_risk(records, step),
        }
    )
    step += 0.01
sweep_df = pd.DataFrame(sweep)
st.line_chart(sweep_df.set_index("threshold"))

# --- Recent lookups table -----------------------------------------------------
st.subheader("Recent lookups")
if records:
    table = pd.DataFrame(records)
    cols = [c for c in ["query", "matched_query", "best_score", "score", "hit_type"] if c in table.columns]
    st.dataframe(table[cols].tail(50).iloc[::-1], use_container_width=True)
else:
    st.info("No lookups recorded yet.")
