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
