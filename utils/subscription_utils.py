from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from utils.db_utils import get_conn


@dataclass(frozen=True)
class Plan:
    key: str
    name: str
    monthly_price_gbp: str
    yearly_price_gbp: str
    league_limit: int | None
    player_limit: int | None
    admin_limit: int | None
    features: tuple[str, ...]


PLANS: dict[str, Plan] = {
    "free": Plan(
        key="free",
        name="Free",
        monthly_price_gbp="£0",
        yearly_price_gbp="£0",
        league_limit=1,
        player_limit=15,
        admin_limit=1,
        features=(
            "Core ratings",
            "Match history",
            "Basic dashboard",
            "Invite code",
        ),
    ),
    "pro": Plan(
        key="pro",
        name="Pro League",
        monthly_price_gbp="£7.99",
        yearly_price_gbp="£79",
        league_limit=1,
        player_limit=60,
        admin_limit=3,
        features=(
            "AI team generator",
            "Matchday Hub",
            "Charts and Season Review",
            "WhatsApp share tools",
            "Player profiles",
            "Exports and backups",
        ),
    ),
    "club": Plan(
        key="club",
        name="Club",
        monthly_price_gbp="£29",
        yearly_price_gbp="£290",
        league_limit=None,
        player_limit=None,
        admin_limit=None,
        features=(
            "Multiple leagues",
            "Multiple admins",
            "Club-wide player database",
            "Divisions and groups ready",
            "Priority support",
        ),
    ),
}

DEFAULT_PLAN_KEY = "free"


def normalise_plan_key(value: Any) -> str:
    key = str(value or DEFAULT_PLAN_KEY).strip().lower()
    return key if key in PLANS else DEFAULT_PLAN_KEY


def get_plan(plan_key: Any) -> Plan:
    return PLANS[normalise_plan_key(plan_key)]


@st.cache_data(ttl=300, show_spinner=False)
def load_league_subscription_cached(league_id: int) -> dict[str, Any]:
    """
    Load subscription metadata if the new columns exist.

    Older local/Supabase databases will not have the subscription columns until
    the SQL foundation is applied, so this helper deliberately falls back to a
    Free active league instead of breaking the app.
    """
    fallback = {
        "plan_key": DEFAULT_PLAN_KEY,
        "subscription_status": "active",
        "trial_ends_at": None,
        "current_period_ends_at": None,
    }
    try:
        conn = get_conn()
        df = pd.read_sql(
            """
            SELECT plan_key, subscription_status, trial_ends_at, current_period_ends_at
            FROM public.leagues
            WHERE id = %s
            LIMIT 1
            """,
            conn,
            params=(int(league_id),),
        )
        conn.close()
    except Exception:
        return fallback

    if df.empty:
        return fallback

    row = df.iloc[0].to_dict()
    return {
        "plan_key": normalise_plan_key(row.get("plan_key")),
        "subscription_status": row.get("subscription_status") or "active",
        "trial_ends_at": row.get("trial_ends_at"),
        "current_period_ends_at": row.get("current_period_ends_at"),
    }


def get_current_league_plan() -> Plan:
    league_id = st.session_state.get("league_id")
    if not league_id:
        return get_plan(DEFAULT_PLAN_KEY)
    subscription = load_league_subscription_cached(int(league_id))
    return get_plan(subscription.get("plan_key"))


def format_limit(value: int | None) -> str:
    return "Unlimited" if value is None else str(value)


def plan_rows() -> list[dict[str, str]]:
    rows = []
    for plan in PLANS.values():
        rows.append(
            {
                "Plan": plan.name,
                "Monthly": plan.monthly_price_gbp,
                "Yearly": plan.yearly_price_gbp,
                "Players": format_limit(plan.player_limit),
                "Admins": format_limit(plan.admin_limit),
                "Leagues": format_limit(plan.league_limit),
            }
        )
    return rows
