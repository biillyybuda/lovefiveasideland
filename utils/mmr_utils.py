from utils.db_utils import get_conn, STARTING_MMR

def get_season_mmr(conn, player_id: int, season_start: str, rolling_mmr: float) -> float:
    """
    Backwards compatible: keeps the 'conn' parameter so existing calls don't break,
    but uses its own fresh connection to avoid 'connection already closed'.
    """
    _ = conn  # keep signature, but don't rely on caller's connection

    conn2 = get_conn()
    cur = conn2.cursor()
    try:
        cur.execute(
            """
            SELECT mmr_after
            FROM public.mmr_history
            WHERE player_id = %s AND date < %s
            ORDER BY date DESC, id DESC
            LIMIT 1
            """,
            (player_id, season_start),
        )
        row = cur.fetchone()
    finally:
        cur.close()
        conn2.close()

    baseline = float(row[0]) if row and row[0] is not None else float(STARTING_MMR)
    return float(STARTING_MMR) + (float(rolling_mmr) - baseline)

def get_current_season_start(today=None) -> str:
    today = today or date.today()
    return f"{today.year}-01-01"

def get_display_mmr(conn, player_row) -> float:
    """
    Returns season-reset MMR for UI display,
    while engine continues to use rolling MMR.
    """
    season_start = get_current_season_start()
    return get_season_mmr(
        conn,
        player_id=player_row["id"],
        season_start=season_start,
        rolling_mmr=player_row["mmr"],
    )
