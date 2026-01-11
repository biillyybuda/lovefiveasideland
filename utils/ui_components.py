# utils/ui_components.py
import streamlit as st
from html import escape

def page_header(
    title: str,
    subtitle: str | None = None,
    *,
    center: bool = True,
    divider: bool = False,
    top_space: str = "0.25rem",
    bottom_space: str = "1.75rem",
):
    """
    Consistent page header used across the whole app.

    - title: main page heading (e.g. "Dashboard")
    - subtitle: optional secondary line (e.g. "Season overview & key stats")
    - center: center align header (recommended for your current style)
    - divider: optional subtle divider under header
    - spacing: tweak top/bottom margin in rem
    """
    align = "center" if center else "left"

    title_html = escape(title)
    subtitle_html = escape(subtitle) if subtitle else ""

    st.markdown(
        f"""
        <div style="text-align:{align}; margin-top:{top_space}; margin-bottom:{bottom_space};">
            <h2 style="margin: 0 0 0.25rem 0;">{title_html}</h2>
            {"<div style='color:#9aa0a6; font-size:0.95rem; margin:0;'>" + subtitle_html + "</div>" if subtitle else ""}
        </div>
        {"<div style='height:1px; background:rgba(255,255,255,0.08); margin: -0.75rem 0 1.25rem 0;'></div>" if divider else ""}
        """,
        unsafe_allow_html=True,
    )
