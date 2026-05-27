import streamlit as st

# If you already have a DB function for this, import that instead.
from utils.league_utils import update_league_name
from utils.subscription_utils import (
    get_plan,
    load_league_subscription_cached,
    plan_rows,
)


def render_league_admin_page():
    st.title("⚙️ League Admin")

    role = (st.session_state.get("league_role") or "").lower()
    if role not in ("admin", "owner"):
        st.error("You don’t have permission to manage this league.")
        return

    league_id = st.session_state.get("league_id")
    current_name = (st.session_state.get("league_name") or "").strip()

    # -----------------------------
    # League settings
    # -----------------------------
    st.subheader("League Settings")

    new_name = st.text_input(
        "League name",
        value=current_name,
        max_chars=40,
        help="This updates what everyone sees when they select this league.",
    )

    if st.button("💾 Save league name", use_container_width=True):
        if not league_id:
            st.error("No league selected.")
            return

        clean = new_name.strip()
        if len(clean) < 3:
            st.warning("League name must be at least 3 characters.")
            return

        update_league_name(league_id, clean)
        st.session_state["league_name"] = clean  # update UI immediately
        st.success("League name updated.")
        st.rerun()

    # -----------------------------
    # Subscription plan
    # -----------------------------
    st.markdown("<hr style='opacity:0.18;'>", unsafe_allow_html=True)
    st.subheader("Subscription")

    if league_id:
        subscription = load_league_subscription_cached(int(league_id))
        plan = get_plan(subscription.get("plan_key"))
        status = str(subscription.get("subscription_status") or "active").title()

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Plan", plan.name)
        c2.metric("Status", status)
        c3.metric("Yearly", plan.yearly_price_gbp)

        st.caption("Payments are not connected yet. This is the plan model we will wire into Stripe later.")
        st.dataframe(plan_rows(), use_container_width=True, hide_index=True)
    else:
        st.info("Select a league to see its subscription plan.")

    # -----------------------------
    # Admin shortcuts (moved from Home)
    # -----------------------------
    st.markdown("<hr style='opacity:0.18;'>", unsafe_allow_html=True)
    st.subheader("Admin Shortcuts")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🧾 Matches Management", use_container_width=True):
            st.session_state["_nav_target"] = "Matches Management"
            st.rerun()

    with b2:
        if st.button("👤 Player Management", use_container_width=True):
            st.session_state["_nav_target"] = "Player Management"
            st.rerun()

    from utils.player_link_utils import (
        fetch_players_with_links,
        fetch_league_members,
        admin_unlink_player,
        admin_assign_player,
    )

    st.markdown("<hr style='opacity:0.18;'>", unsafe_allow_html=True)
    st.subheader("🔗 Player Links")

    league_id_int = int(st.session_state.get("league_id"))  # type: ignore

    players = fetch_players_with_links(league_id_int)
    members = fetch_league_members(league_id_int)

    # -----------------------------
    # Members list (accounts)
    # -----------------------------
    # members: (user_id, email, profile_display_name, role, status)
    member_ids = [str(m[0]) for m in members]
    member_labels = []
    for user_id, email, prof_dn, role, status in members:
        dn = (prof_dn or "").strip()
        label = f"{dn} — {email}" if dn else (email or str(user_id))
        member_labels.append(f"{label}  ({role}/{status})")

    # -----------------------------
    # Players links table
    # -----------------------------
    # players: (pid, pname, uid, email, profile_display_name)
    st.caption("Admins can fix mistakes here. Users can only self-link to unlinked players.")

    link_rows = []
    for pid, pname, uid, email, prof_dn in players:
        link_rows.append(
            {
                "Player": pname,
                "Account display name": (prof_dn or "").strip(),
                "Account email": (email or "").strip(),
                "Linked": "✅" if uid else "",
            }
        )

    if link_rows:
        st.dataframe(link_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No players exist in this league yet. Add players in Player Management.")

    # -----------------------------
    # Unlink tool
    # -----------------------------
    st.markdown("### Unlink a player")

    linked = [(pid, pname, uid, email, prof_dn) for pid, pname, uid, email, prof_dn in players if uid]
    if not linked:
        st.info("No linked players to unlink.")
    else:
        linked_labels = []
        linked_ids = []
        for pid, pname, uid, email, prof_dn in linked:
            who = (prof_dn or "").strip() or (email or str(uid)[:8] + "…")
            linked_labels.append(f"{pname}  →  {who}")
            linked_ids.append(pid)

        pick_unlink = st.selectbox("Select linked player", linked_labels, key="unlink_pick")
        unlink_pid = int(linked_ids[linked_labels.index(pick_unlink)])

        if st.button("🧹 Unlink", use_container_width=True, key="do_unlink"):
            admin_unlink_player(league_id_int, unlink_pid)
            st.success("Unlinked.")
            st.rerun()

    # -----------------------------
    # Assign / Reassign tool
    # -----------------------------
    st.markdown("### Assign / Reassign")

    if not players:
        st.info("No players to assign yet. Go to Player Management and add players first.")
        return

    if not members:
        st.info("No league members/accounts found yet. Invite someone (or ensure you are a member).")
        return

    player_labels = []
    player_ids = []
    for pid, pname, uid, email, prof_dn in players:
        if uid:
            who = (prof_dn or "").strip() or (email or str(uid)[:8] + "…")
            player_labels.append(f"{pname}  (linked: {who})")
        else:
            player_labels.append(f"{pname}  (unlinked)")
        player_ids.append(pid)

    pick_player = st.selectbox("Player", player_labels, key="assign_player")
    pick_user = st.selectbox("Account", member_labels, key="assign_user")

    # Safety (shouldn't trigger now, but keeps it bulletproof)
    if pick_player is None or pick_user is None:
        st.warning("Select both a player and an account.")
        return

    assign_pid = int(player_ids[player_labels.index(pick_player)])
    assign_uid = str(member_ids[member_labels.index(pick_user)])

    if st.button("✅ Assign account to player", use_container_width=True, key="do_assign"):
        admin_assign_player(league_id_int, assign_pid, assign_uid)
        st.success("Assigned.")
        st.rerun()
