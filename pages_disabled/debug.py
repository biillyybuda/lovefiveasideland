import streamlit as st
import pandas as pd
from utils.team_ai_engine import get_engine_state
from pages_disabled.PlayerReels import (
    goal_timeline_for_match,
    classify_goal_tag,
    find_linked_goal_for_assist,
    load_data,   # NEW
)

def render_debug():
    st.header("🔍 FULL GOAL DEBUGGER")

    # ===============================
    # 🔧 ADMIN TOOL — REBUILD ALL TAGS
    # ===============================
    st.markdown("### 🛠 Maintenance Tools")

    if st.button("Rebuild ALL goal/assist/save tags"):
        rebuild_all_tags()
        st.success("All tags rebuilt! 🎉")
        st.info("Reload the page to see updated tags in the engine and debug tables.")


    # Use the same loader as PlayerReels so the debugger sees
    # exactly the same data and preprocessing.
    moments, matches = load_data()

    # If you still want engine stuff later on this page you can
    # still call it separately:
    eng = get_engine_state()

    match_id = st.number_input("Match ID", min_value=1, step=1)

    if match_id:
        goals_df = moments[
            (moments["match_id"] == match_id) &
            (moments["type"].str.lower() == "goal")
        ]

        st.subheader("🔹 Raw Goal Rows from DB")
        st.dataframe(goals_df, use_container_width=True, hide_index=True)

        st.subheader("🔹 classify_goal_tag() Output")
        rows = []
        for _, r in goals_df.iterrows():
            tag = classify_goal_tag(r, matches, moments)
            rows.append({
                "moment_id": r["id"],
                "label": r["label"],
                "special_tag": r["special_tag"],
                "classify_goal_tag()": tag,
                "player_name": r["player_name"],
                "goalkeeper_name": r["goalkeeper_name"],
                "timestamp": r["timestamp_sec"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("🔹 goal_timeline_for_match() Output")
        try:
            g_df, timeline, score_labels, scorers_team, score_changes = goal_timeline_for_match(
                int(match_id), moments, matches
            )

            st.markdown("**Timeline DataFrame:**")
            st.dataframe(g_df, use_container_width=True, hide_index=True)

            st.markdown("**Score Labels:**")
            st.write(score_labels)

            st.markdown("**Score Changes:**")
            st.write(score_changes)

        except Exception as e:
            st.error(f"Timeline Error: {e}")

# ======================================================
# ONE-TIME TAG REBUILD SCRIPT
# Fills: goal_tag, assist_tag, save_tag for ALL moments
# ======================================================

import pandas as pd
from utils.calc_utils import get_conn

# Import PlayerReels logic
from pages_disabled.PlayerReels import (
    classify_goal_tag,
    classify_save_tag,
    find_linked_goal_for_assist,
    goal_timeline_for_match,
    parse_names_from_label,
    load_data
)

def rebuild_all_tags():
    print("🔄 Starting full highlight_moments tag rebuild...")

    # Load moments + matches using PlayerReels loader
    moments, matches = load_data()

    conn = get_conn()
    cur = conn.cursor()

    total = len(moments)
    processed = 0

    for _, row in moments.iterrows():
        mid = int(row["match_id"])
        moment_id = int(row["id"])
        moment_type = str(row["type"]).lower()

        goal_tag = None
        assist_tag = None
        save_tag = None
        save_state = None
        save_importance = 0

        # Preload goal timeline once per match
        g_df, timeline, score_labels, scorers_team, score_changes = goal_timeline_for_match(
            mid, moments, matches
        )

        # ---- GOALS ----
        if moment_type == "goal":
            tag = classify_goal_tag(row, matches, moments)
            goal_tag = tag

        # ---- ASSISTS ----
        elif moment_type == "assist":
            # Find the linked goal using PlayerReels logic
            linked_goal = find_linked_goal_for_assist(row, g_df) if g_df is not None else None
            if linked_goal is not None:
                tag = classify_goal_tag(linked_goal, matches, moments)
                assist_tag = tag

        # ---- SAVES ----
        elif moment_type == "save":
            info = classify_save_tag(row, matches, moments)

            if isinstance(info, dict):
                save_tag = info.get("tag")
                save_state = info.get("state")
                save_importance = info.get("importance")
            else:
                # fallback for older style (string-only)
                save_tag = info
                save_state = None
                save_importance = 0

        # Write results to DB
        cur.execute(
            """
            UPDATE highlight_moments
            SET goal_tag = COALESCE(?, goal_tag),
                assist_tag = COALESCE(?, assist_tag),
                save_tag = COALESCE(?, save_tag),
                save_state = COALESCE(?, save_state),
                save_importance = COALESCE(?, save_importance)
            WHERE id = %s
            """,
            (goal_tag, assist_tag, save_tag, save_state, save_importance, moment_id)
        )

        processed += 1
        if processed % 50 == 0:
            print(f"   → Processed {processed}/{total} moments...")

    conn.commit()
    conn.close()

    print("✅ Tag rebuild complete!")
    print("All goal_tag, assist_tag, save_tag fields are now populated.")



