import streamlit as st

from utils.ui_components import page_header


def render_matchday_hub_page():
    page_header(
        "Matchday Hub",
        "Generate teams or use history to guide team selection",
        center=True,
        divider=True,
    )

    # --- Mode toggle ---
    # Using radio keeps it simple + reliable. Styled like buttons via horizontal=True.
    mode = st.radio(
        "Mode",
        ["AI Team Generator", "Matchday Memory"],
        horizontal=True,
        key="matchday_hub_mode",
    )

    st.divider()

    # Import lazily so Streamlit doesn't import both pages every rerun.
    if mode == "Matchday Memory":
        try:
            from pages_disabled.matchday_memory_page import render_matchday_memory_page
        except Exception:
            # Fallback for alternate module layouts
            from matchday_memory_page import render_matchday_memory_page  # type: ignore

        render_matchday_memory_page(show_header=False)
        return

    # Default
    try:
        from pages_disabled.team_generator_page import render_team_generator_page
    except Exception:
        from team_generator_page import render_team_generator_page  # type: ignore

    render_team_generator_page()
