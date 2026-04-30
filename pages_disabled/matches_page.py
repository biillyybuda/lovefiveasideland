import streamlit as st
from utils.cache_utils import invalidate_app_caches
import pandas as pd
from datetime import datetime
from utils.db_utils import load_players_df, load_matches_df, get_conn, backup_db_manual, STARTING_MMR
from utils.calc_utils import expected_score, process_unprocessed_matches, compute_streaks_from_matches, reset_and_reprocess_season
from utils.export_utils import df_to_png, fig_to_png_bytes
from utils.ui_components import page_header

players_df = load_players_df()

# Build canonical -> display map
name_map = dict(
    zip(
        players_df["name"],
        players_df.get("display_name", players_df["name"])
    )
)

# Fallback if display_name is blank
for k, v in name_map.items():
    if not v or str(v).strip() == "":
        name_map[k] = k.replace("_", " ").title()



def render_matches_page():
    page_header("Matches Management", "Add, edit and process matches", center=True, divider=True)
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)

    display_to_canonical = {v: k for k, v in name_map.items()}
    canonical_to_display = {k: v for k, v in name_map.items()}
    display_names = sorted(display_to_canonical.keys())


    dfm = load_matches_df()
    dfm_view = dfm.rename(columns={
        'date': 'Date',
        'team_a': 'Team A',
        'team_b': 'Team B',
        'score': 'Score',
        'result': 'Result',
        'processed': 'Processed',
        'team_a_avg': 'Team A Avg',
        'team_b_avg': 'Team B Avg',
        'id': 'ID'
    })

    def to_display_list(s):
        if not s:
            return s
        return ", ".join(name_map.get(p.strip(), p.strip()) for p in s.split(","))

    dfm_view["Team A"] = dfm_view["Team A"].apply(to_display_list)
    dfm_view["Team B"] = dfm_view["Team B"].apply(to_display_list)


    # 🔹 Drop legacy colour columns (UI only)
    dfm_view = dfm_view.drop(
        columns=[c for c in ["team_a_color", "team_b_color"] if c in dfm_view.columns],
        errors="ignore",
    )
    st.dataframe(dfm_view, use_container_width=True)

    # --- Add / Edit / Delete Matches ---
    st.markdown('### Add a new match')
    with st.form(key='add_match_form'):
        m_date = st.date_input('Date', value=datetime.today())
        display_to_canonical = {v: k for k, v in name_map.items()}
        canonical_to_display = {k: v for k, v in name_map.items()}
        display_names = sorted(display_to_canonical.keys())

        team_a_disp = st.multiselect(
            "Team A players (comma separate or multiselect)",
            display_names,
            key="add_team_a",
        )

        team_b_disp = st.multiselect(
            "Team B players (comma separate or multiselect)",
            display_names,
            key="add_team_b",
        )

        team_a_sel = [display_to_canonical[d] for d in team_a_disp]
        team_b_sel = [display_to_canonical[d] for d in team_b_disp]
        score_in = st.text_input('Score (e.g. 10-8)', value='', key='add_score')
        result_in = st.selectbox('Result', options=['', 'A', 'B', 'Draw'], index=0)
        submitted = st.form_submit_button('Save Match')

    if submitted:
        if not team_a_sel or not team_b_sel:
            st.warning('Select at least one player per team.')
        elif set(team_a_sel) & set(team_b_sel):
            st.error('A player appears in both teams.')
        else:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO matches (date, team_a, team_b, score, result, processed)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (m_date.isoformat(), ", ".join(team_a_sel), ", ".join(team_b_sel), score_in, result_in, 0)
            )
            conn.commit()
            conn.close()

            # 🔥 clear cached matches so UI refreshes
            try:
                load_matches_df.clear()
            except Exception:
                invalidate_app_caches()

            st.success('Match added (draft).')
            st.rerun()

    # Edit / Delete existing matches
    st.markdown('### Edit or Delete existing match (select ID below)')
    try:
        sel_id = st.number_input('Match ID to edit/delete (use ID column)', value=0, step=1)
        if sel_id:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT id, date, team_a, team_b, score, result, processed FROM matches WHERE id=%s",
                (int(sel_id),)
            )
            row = cur.fetchone()
            conn.close()
            if row:
                edit_id, ed_date, ed_ta, ed_tb, ed_score, ed_result, ed_proc = row
                with st.form(key='edit_match_form'):
                    ed_date_in = st.date_input(
                        'Date',
                        value=datetime.fromisoformat(ed_date).date() if ed_date else datetime.today()
                    )
                    # Convert stored canonical strings → list
                    ed_team_a_list = [p.strip() for p in (ed_ta or "").split(",") if p.strip()]
                    ed_team_b_list = [p.strip() for p in (ed_tb or "").split(",") if p.strip()]

                    # Convert to display names for UI
                    ed_team_a_display = [
                        canonical_to_display.get(p, p) for p in ed_team_a_list
                    ]
                    ed_team_b_display = [
                        canonical_to_display.get(p, p) for p in ed_team_b_list
                    ]

                    ed_team_a_disp = st.multiselect(
                        "Team A players",
                        options=display_names,
                        default=ed_team_a_display,
                        key=f"edit_team_a_{edit_id}"
                    )

                    ed_team_b_disp = st.multiselect(
                        "Team B players",
                        options=display_names,
                        default=ed_team_b_display,
                        key=f"edit_team_b_{edit_id}"
                    )
                    ed_score_in = st.text_input('Score', value=ed_score or '')
                    result_options = ['', 'A', 'B', 'Draw']
                    result_index = result_options.index(ed_result) if ed_result in result_options else 0

                    ed_result_in = st.selectbox(
                        "Result",
                        options=result_options,
                        index=result_index,
                        key=f"edit_result_{edit_id}",
                    )
                    save_edit = st.form_submit_button('Save Changes')
                    delete_it = st.form_submit_button('Delete Match')
                if save_edit:
                    # Convert display selections back to canonical names for DB
                    team_a_db = ", ".join(display_to_canonical[d] for d in ed_team_a_disp)
                    team_b_db = ", ".join(display_to_canonical[d] for d in ed_team_b_disp)

                    # Validation: stop duplicates across teams
                    if set(team_a_db.split(", ")) & set(team_b_db.split(", ")):
                        st.error("A player appears in both teams.")
                    elif not ed_team_a_disp or not ed_team_b_disp:
                        st.error("Select at least one player per team.")
                    else:
                        conn = get_conn()
                        cur = conn.cursor()
                        cur.execute(
                            """
                            UPDATE matches
                            SET date=%s, team_a=%s, team_b=%s, score=%s, result=%s, processed=%s
                            WHERE id=%s
                            """,
                            (
                                ed_date_in.isoformat(),
                                team_a_db,
                                team_b_db,
                                ed_score_in,
                                ed_result_in,
                                ed_proc,
                                int(edit_id),
                            )
                        )
                        conn.commit()
                        conn.close()

                        try:
                            load_matches_df.clear()
                        except Exception:
                            invalidate_app_caches()

                        st.success('Match updated')
                        st.rerun()
                if delete_it:
                    conn = get_conn()
                    cur = conn.cursor()
                    cur.execute("DELETE FROM matches WHERE id=%s", (int(edit_id),))
                    conn.commit()
                    conn.close()

                    try:
                        load_matches_df.clear()
                    except Exception:
                        invalidate_app_caches()

                    st.success('Match deleted')
                    st.rerun()
            else:
                st.info('No match found with that ID.')
    except Exception as e:
        st.error(f'Error editing matches: {e}')

    st.download_button(
        "Export Matches (PNG)",
        data=df_to_png(dfm_view.head(50), title="Matches (latest)"),
        file_name="matches.png",
        mime="image/png"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Process Matches Button ---

    st.markdown("### Process Games")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Process Unprocessed Matches"):
            try:
                cnt = process_unprocessed_matches()

                # 🔥 clear cached matches so UI refreshes
                try:
                    load_matches_df.clear()
                except Exception:
                    invalidate_app_caches()

                if cnt > 0:
                    st.success(f"Processed {cnt} match(es).")
                else:
                    st.info("No unprocessed matches found.")
                st.rerun()
            except Exception as e:
                st.error(f"Error while processing matches: {e}")

    with col2:
        if st.button("Reset to 1000 + Reprocess Season"):
            try:
                backup_db_manual()
                cnt = reset_and_reprocess_season()

                # 🔥 clear cached matches so UI refreshes
                try:
                    load_matches_df.clear()
                except Exception:
                    invalidate_app_caches()

                st.success(f"Full rebuild complete. Processed {cnt} match(es) from 1000.")
                st.rerun()

            except Exception as e:
                st.error(f"Error while rebuilding season: {e}")



