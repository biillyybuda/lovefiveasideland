"""Shared cache invalidation helpers for Streamlit + Love Five engine."""

import streamlit as st


def invalidate_app_caches() -> None:
    """Clear Streamlit caches and the in-memory AI engine cache after data changes."""
    try:
        st.cache_data.clear()
    except Exception:
        pass

    try:
        from utils.team_ai_engine import clear_engine_cache
        clear_engine_cache()
    except Exception:
        pass
