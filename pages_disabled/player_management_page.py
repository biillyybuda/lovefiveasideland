import streamlit as st
import pandas as pd
from datetime import datetime
from utils.db_utils import (
    load_players_df,
    get_conn,
    get_current_league_id,
    backup_db_manual,
    STARTING_MMR,
)

from utils.ui_components import page_header
from utils.calc_utils import expected_score, process_unprocessed_matches, compute_streaks_from_matches
from utils.export_utils import df_to_png, fig_to_png_bytes
from utils.names import canonical_name, display_name


@st.cache_resource
def ensure_player_schema():
    """Ensure required columns exist in DB (SQLite + Postgres safe)."""
    conn = get_conn()
    cur = conn.cursor()

    # --- Detect existing columns (Postgres vs SQLite) ---
    try:
        # Postgres / Supabase
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'players'
        """)
        cols = [r[0] for r in cur.fetchall()]
    except Exception:
        # SQLite fallback
        cur.execute("PRAGMA table_info(players)")
        cols = [c[1] for c in cur.fetchall()]

    # --- Schema updates ---
    if "strengths" not in cols:
        cur.execute("ALTER TABLE players ADD COLUMN strengths TEXT DEFAULT ''")

    if "fitness" not in cols:
        cur.execute("ALTER TABLE players ADD COLUMN fitness TEXT DEFAULT 'Medium'")

    if "display_name" not in cols:
        cur.execute("ALTER TABLE players ADD COLUMN display_name TEXT DEFAULT ''")

    if "is_active" not in cols:
        cur.execute("ALTER TABLE players ADD COLUMN is_active INTEGER DEFAULT 1")
        cur.execute("UPDATE players SET is_active=1 WHERE is_active IS NULL")

    if "archived_at" not in cols:
        cur.execute("ALTER TABLE players ADD COLUMN archived_at TEXT DEFAULT NULL")

    conn.commit()
    conn.close()


def render_player_management_page():
    page_header("Player Management", "Add, edit and archive players", center=True, divider=True)
    role = (st.session_state.get("league_role") or "").lower()
    if role not in ("admin", "owner"):
        st.warning("Only league admins can add, edit, archive, or manage players.")
        if st.button("Back to Home", use_container_width=True):
            st.session_state["_nav_target"] = "Home"
            st.rerun()
        st.stop()

    st.markdown("<div class='stCard'>", unsafe_allow_html=True)

    league_id = get_current_league_id()
    ensure_player_schema()
    df = load_players_df()

    # Show/hide archived players
    show_archived = st.toggle("Show archived players", value=False, key="show_archived_players")

    # Ensure column exists in DF (defensive)
    if "is_active" not in df.columns:
        df["is_active"] = 1

    if not show_archived:
        df = df[df["is_active"].fillna(1).astype(int) == 1]


    # Ensure display_name always has something (fallback to title-case of db name)
    if not df.empty:
        if "display_name" not in df.columns:
            df["display_name"] = ""
        df["display_name"] = df["display_name"].fillna("").astype(str)
        df.loc[df["display_name"].str.strip() == "", "display_name"] = df["name"].astype(str).apply(display_name)


    # Desired column order (only include columns that actually exist)
    preferred_order = [
        "id",
        "name",
        "display_name",
        "mmr",
        "matches_played",
        "last_match_date",
        "fitness",
    ]

    # Explicitly restrict table to only desired columns
    visible_cols = [c for c in preferred_order if c in df.columns]
    df = df[visible_cols]


    updated_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="player_editor",
        disabled=["id", "name"],  # lock DB key
        column_config={
            "name": st.column_config.TextColumn(
                "DB Name (lowercase)",
                help="Internal key used for all stats/joins. Do not edit.",
                width="medium",
            ),
            "display_name": st.column_config.TextColumn(
                "Displayed Name",
                help="What appears in the app UI. You can edit this freely.",
                width="medium",
            ),
            "strengths": st.column_config.TextColumn(
                "Strengths",
                help="Comma-separated list of up to 2 strengths (e.g. 'Finishing, Creating').",
                width="large",
            ),
            "fitness": st.column_config.SelectboxColumn(
                "Fitness Level",
                help="Player's running and stamina level.",
                options=["High", "Medium", "Low"],
                required=False,
                width="medium",
            ),
        }
    )

    # --- Add single player ---
    st.markdown('### ➕ Add Single Player')

    new_name = st.text_input('Name', key='new_name')
    new_mmr = st.number_input('Starting MMR', value=STARTING_MMR, step=1, key='new_mmr')
    new_fitness = st.selectbox(
        'Fitness Level',
        ['High', 'Medium', 'Low'],
        index=1,
        key='new_fitness'
    )

    if st.button('Add Player'):
        if new_name.strip() == '':
            st.warning('Enter a name')
        else:
            backup_db_manual()
            conn = get_conn()
            cur = conn.cursor()
            try:
                strengths_str = ""
                name_key = canonical_name(new_name)
                ui_name = new_name.strip()
                if ui_name == "":
                    ui_name = display_name(name_key)

                cur.execute(
                    """
                    INSERT INTO players
                        (league_id, name, display_name, mmr, strengths, fitness, is_active, archived_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 1, NULL)
                    ON CONFLICT (league_id, name) DO NOTHING
                    """,
                    (league_id, name_key, ui_name, float(new_mmr), strengths_str, new_fitness)
                )
                conn.commit()
                try:
                    load_players_df.clear()
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Error adding player: {e}")
            finally:
                conn.close()
            st.rerun()


    # --- Archive / Unarchive Player ---
    st.markdown("### 🗃️ Archive Player")

    # Build options from full players table (active + archived), so you can unarchive too
    all_players = load_players_df()
    if "display_name" not in all_players.columns:
        all_players["display_name"] = all_players["name"].astype(str).apply(display_name)
    all_players["display_name"] = all_players["display_name"].fillna("").astype(str)
    all_players.loc[all_players["display_name"].str.strip() == "", "display_name"] = all_players["name"].astype(str).apply(display_name)

    if "is_active" not in all_players.columns:
        all_players["is_active"] = 1

    # Nice label: "Billy (active)" / "Kelso (archived)"
    all_players["status"] = all_players["is_active"].fillna(1).astype(int).map({1: "active", 0: "archived"})
    all_players["label"] = all_players["display_name"].astype(str) + " (" + all_players["status"] + ")" # type: ignore

    label_to_id = dict(zip(all_players["label"], all_players["id"]))

    colA, colB = st.columns([3, 1])

    with colA:
        chosen_label = st.selectbox(
            "Select player",
            options=list(label_to_id.keys()),
            index=0 if len(label_to_id) else None,
            key="archive_select_player",
        )

    with colB:
        action = st.radio("Action", ["Archive", "Unarchive"], horizontal=False, key="archive_action")

    reason = st.text_input("Reason (optional)", key="archive_reason")

    if st.button("Apply", key="archive_apply"):
        if not chosen_label:
            st.warning("Select a player")
        else:
            backup_db_manual()
            pid = int(label_to_id[chosen_label])
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            conn = get_conn()
            cur = conn.cursor()
            if action == "Archive":
                cur.execute(
                    "UPDATE players SET is_active=0, archived_at=%s WHERE id=%s AND league_id=%s",
                    (now, pid, league_id),
                )
            else:
                cur.execute(
                    "UPDATE players SET is_active=1, archived_at=NULL WHERE id=%s AND league_id=%s",
                    (pid, league_id),
                )
            conn.commit()
            conn.close()
            try:
                load_players_df.clear()
            except Exception:
                pass

            st.success(f"✅ {action}d: {chosen_label}")
            st.rerun()

    # --- Save Edits ---
    if st.button('💾 Save Data (apply edits)'):
        try:
            if {'id', 'name', 'mmr'}.issubset(updated_df.columns):
                backup_db_manual()
                conn = get_conn()
                cur = conn.cursor()
                for _, row in updated_df.iterrows():
                    rid = row.get('id')
                    name_key = canonical_name(row.get("name", ""))  # DB key (locked anyway)
                    ui_name = str(row.get("display_name", "")).strip()
                    if ui_name == "":
                        ui_name = display_name(name_key)
                    mmr = float(row.get('mmr') if row.get('mmr') not in (None, "") else STARTING_MMR) # type: ignore
                    strengths_str = str(row.get("strengths", "")).strip()
                    fitness = str(row.get("fitness", "")).strip()

                    if rid and not pd.isna(rid):
                        cur.execute(
                            'UPDATE players SET display_name=%s, mmr=%s, fitness=%s WHERE id=%s AND league_id=%s',
                            (ui_name, mmr, fitness, int(rid), league_id)
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO players
                                (league_id, name, display_name, mmr, strengths, fitness)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (league_id, name) DO NOTHING
                            """,
                            (league_id, name_key, ui_name, mmr, strengths_str, fitness)
                        )
                conn.commit()
                conn.close()
                try:
                    load_players_df.clear()
                except Exception:
                    pass
                st.success('✅ Player edits saved successfully.')
                st.rerun()
            else:
                st.error("Table doesn't contain expected columns.")
        except Exception as e:
            st.error(f'Error saving players: {e}')
