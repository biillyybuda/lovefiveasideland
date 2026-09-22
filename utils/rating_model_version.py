"""Keep the archived Python writer away from leagues using the new atomic writer."""
from utils.db_utils import get_conn, get_current_league_id


def uses_atomic_rating_writer():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.balance_model_states')")
            if cur.fetchone()[0] is None:
                return False
            cur.execute("SELECT EXISTS (SELECT 1 FROM public.balance_model_states WHERE league_id=%s)",
                        (get_current_league_id(),))
            return bool(cur.fetchone()[0])
    finally:
        conn.close()


def require_legacy_rating_writer():
    if uses_atomic_rating_writer():
        raise RuntimeError("This league now saves ratings through the updated website. Use its result-management page.")
