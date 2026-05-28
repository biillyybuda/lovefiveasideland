import logging
import os
import time
from contextlib import contextmanager

import streamlit as st


def perf_enabled() -> bool:
    flag = str(os.getenv("LOVEFIVE_PERF_DEBUG", "")).strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True

    try:
        value = st.query_params.get("perf")
    except Exception:
        value = None
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def reset_perf_trace() -> None:
    st.session_state["_lf_perf_trace"] = []


def add_perf_trace(label: str, seconds: float) -> None:
    if not perf_enabled():
        return
    st.session_state.setdefault("_lf_perf_trace", []).append((label, float(seconds)))


@contextmanager
def timed(label: str, *, log_over: float = 1.0):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        add_perf_trace(label, elapsed)
        if elapsed >= log_over:
            logging.info("Perf %s %.3fs", label, elapsed)


def render_perf_sidebar(total_label: str = "Page") -> None:
    if not perf_enabled():
        return

    rows = st.session_state.get("_lf_perf_trace") or []
    if not rows:
        return

    with st.sidebar.expander("Performance", expanded=False):
        for label, seconds in rows:
            st.caption(f"{label}: {seconds:.2f}s")
        total = sum(float(s) for _, s in rows)
        st.caption(f"{total_label} total: {total:.2f}s")
