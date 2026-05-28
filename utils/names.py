def display_name(name_key: str) -> str:
    """
    Convert canonical lowercase name to display format.
    Examples:
      'billy' -> 'Billy'
      'tom d' -> 'Tom D'
      'sam k jr' -> 'Sam K Jr'
    """
    if not name_key:
        return ""
    return " ".join(word.capitalize() for word in name_key.split())

def canonical_name(name: str) -> str:
    """Lowercase + trim + collapse spaces for storage/logic."""
    if not name:
        return ""
    return " ".join(str(name).strip().lower().split())

def pretty_title(name_key: str) -> str:
    """Fallback UI formatting if display_name is missing."""
    if not name_key:
        return ""
    return " ".join(w.capitalize() for w in str(name_key).split())


def player_display_name(name_key: str, stored_display_name: str | None = None) -> str:
    """Prefer the DB display name, then fall back to a readable stored key."""
    display = str(stored_display_name or "").strip()
    if display:
        return display
    return display_name(str(name_key or "").strip())


def display_name_map_from_players_df(players_df) -> dict[str, str]:
    """Build a canonical name -> display name map from a players dataframe."""
    if players_df is None or getattr(players_df, "empty", True) or "name" not in players_df.columns:
        return {}

    out = {}
    for _, row in players_df.iterrows():
        key = str(row.get("name") or "").strip()
        if not key:
            continue
        stored = row.get("display_name") if "display_name" in players_df.columns else ""
        out[key] = player_display_name(key, stored)
    return out
