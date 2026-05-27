-- Love Five RLS/security hardening draft.
--
-- Review before running in Supabase. This is intended for a multi-league
-- product where users may belong to several leagues with different roles.
--
-- Important:
-- - The Streamlit server currently connects directly to Postgres, so app-level
--   checks still matter.
-- - These policies protect browser/Supabase API access and future clients.
-- - Run this after confirming existing app flows against a test branch.

-- Keep unauthenticated users away from app tables.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon;
REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM anon, authenticated;

-- The app still needs signed-in users to read through RLS.
GRANT SELECT ON public.leagues TO authenticated;
GRANT SELECT ON public.league_members TO authenticated;
GRANT SELECT ON public.players TO authenticated;
GRANT SELECT ON public.matches TO authenticated;
GRANT SELECT ON public.mmr_history TO authenticated;
GRANT SELECT ON public.match_player_stats TO authenticated;
GRANT SELECT ON public.match_stats TO authenticated;
GRANT SELECT ON public.highlight_moments TO authenticated;
GRANT SELECT ON public.fantasy_points TO authenticated;
GRANT SELECT ON public.seasons TO authenticated;
GRANT SELECT, UPDATE ON public.profiles TO authenticated;

-- Admin writes. RLS below decides which rows are actually allowed.
GRANT INSERT, UPDATE, DELETE ON public.players TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.matches TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.mmr_history TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.match_player_stats TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.match_stats TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.highlight_moments TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.fantasy_points TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.seasons TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.league_invites TO authenticated;
GRANT UPDATE ON public.leagues TO authenticated;
GRANT INSERT, UPDATE, DELETE ON public.league_members TO authenticated;

-- Owners are not currently allowed by the live check constraint; the Python
-- app already treats owner as an admin role, so the DB should support it too.
ALTER TABLE public.league_members
DROP CONSTRAINT IF EXISTS league_members_role_check;

ALTER TABLE public.league_members
ADD CONSTRAINT league_members_role_check
CHECK (role IN ('owner', 'admin', 'member'));

-- Remove risky legacy defaults once existing insert paths have been checked.
-- New multi-league rows should always set league_id explicitly.
ALTER TABLE public.players ALTER COLUMN league_id DROP DEFAULT;
ALTER TABLE public.matches ALTER COLUMN league_id DROP DEFAULT;
ALTER TABLE public.mmr_history ALTER COLUMN league_id DROP DEFAULT;
ALTER TABLE public.match_player_stats ALTER COLUMN league_id DROP DEFAULT;
ALTER TABLE public.match_stats ALTER COLUMN league_id DROP DEFAULT;
ALTER TABLE public.highlight_moments ALTER COLUMN league_id DROP DEFAULT;
ALTER TABLE public.fantasy_points ALTER COLUMN league_id DROP DEFAULT;
ALTER TABLE public.seasons ALTER COLUMN league_id DROP DEFAULT;

-- Profiles: own profile only.
DROP POLICY IF EXISTS "read own profile" ON public.profiles;
DROP POLICY IF EXISTS profiles_select_own ON public.profiles;
DROP POLICY IF EXISTS profiles_update_own ON public.profiles;

CREATE POLICY profiles_select_own
ON public.profiles
FOR SELECT TO authenticated
USING (id = auth.uid());

CREATE POLICY profiles_update_own
ON public.profiles
FOR UPDATE TO authenticated
USING (id = auth.uid())
WITH CHECK (id = auth.uid());

-- Leagues.
DROP POLICY IF EXISTS leagues_select_member ON public.leagues;
DROP POLICY IF EXISTS leagues_admin_update ON public.leagues;

CREATE POLICY leagues_select_member
ON public.leagues
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = leagues.id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
  )
);

CREATE POLICY leagues_admin_update
ON public.leagues
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = leagues.id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
      AND lm.role IN ('owner', 'admin')
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = leagues.id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
      AND lm.role IN ('owner', 'admin')
  )
);

-- League members.
DROP POLICY IF EXISTS league_members_select_self ON public.league_members;
DROP POLICY IF EXISTS league_members_select_same_league_admin ON public.league_members;
DROP POLICY IF EXISTS league_members_admin_manage ON public.league_members;

CREATE POLICY league_members_select_self
ON public.league_members
FOR SELECT TO authenticated
USING (user_id = auth.uid());

CREATE POLICY league_members_select_same_league_admin
ON public.league_members
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = league_members.league_id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
      AND lm.role IN ('owner', 'admin')
  )
);

CREATE POLICY league_members_admin_manage
ON public.league_members
FOR ALL TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = league_members.league_id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
      AND lm.role IN ('owner', 'admin')
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = league_members.league_id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
      AND lm.role IN ('owner', 'admin')
  )
);

-- Standard league-owned tables.
DROP POLICY IF EXISTS players_league_select ON public.players;
DROP POLICY IF EXISTS players_league_admin_write ON public.players;
DROP POLICY IF EXISTS matches_league_select ON public.matches;
DROP POLICY IF EXISTS matches_league_admin_write ON public.matches;
DROP POLICY IF EXISTS mmr_history_league_select ON public.mmr_history;
DROP POLICY IF EXISTS mmr_history_league_admin_write ON public.mmr_history;
DROP POLICY IF EXISTS match_player_stats_league_select ON public.match_player_stats;
DROP POLICY IF EXISTS match_player_stats_league_admin_write ON public.match_player_stats;
DROP POLICY IF EXISTS match_stats_league_select ON public.match_stats;
DROP POLICY IF EXISTS match_stats_league_admin_write ON public.match_stats;
DROP POLICY IF EXISTS highlight_moments_league_select ON public.highlight_moments;
DROP POLICY IF EXISTS highlight_moments_league_admin_write ON public.highlight_moments;
DROP POLICY IF EXISTS fantasy_points_league_select ON public.fantasy_points;
DROP POLICY IF EXISTS fantasy_points_league_admin_write ON public.fantasy_points;
DROP POLICY IF EXISTS seasons_league_select ON public.seasons;
DROP POLICY IF EXISTS seasons_league_admin_write ON public.seasons;

CREATE POLICY players_league_select ON public.players
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = players.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY players_league_admin_write ON public.players
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = players.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = players.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

CREATE POLICY matches_league_select ON public.matches
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = matches.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY matches_league_admin_write ON public.matches
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = matches.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = matches.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

CREATE POLICY mmr_history_league_select ON public.mmr_history
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = mmr_history.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY mmr_history_league_admin_write ON public.mmr_history
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = mmr_history.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = mmr_history.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

CREATE POLICY match_player_stats_league_select ON public.match_player_stats
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = match_player_stats.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY match_player_stats_league_admin_write ON public.match_player_stats
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = match_player_stats.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = match_player_stats.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

CREATE POLICY match_stats_league_select ON public.match_stats
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = match_stats.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY match_stats_league_admin_write ON public.match_stats
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = match_stats.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = match_stats.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

CREATE POLICY highlight_moments_league_select ON public.highlight_moments
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = highlight_moments.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY highlight_moments_league_admin_write ON public.highlight_moments
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = highlight_moments.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = highlight_moments.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

CREATE POLICY fantasy_points_league_select ON public.fantasy_points
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = fantasy_points.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY fantasy_points_league_admin_write ON public.fantasy_points
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = fantasy_points.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = fantasy_points.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

CREATE POLICY seasons_league_select ON public.seasons
FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = seasons.league_id AND lm.user_id = auth.uid() AND lm.status = 'active'));

CREATE POLICY seasons_league_admin_write ON public.seasons
FOR ALL TO authenticated
USING (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = seasons.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')))
WITH CHECK (EXISTS (SELECT 1 FROM public.league_members lm WHERE lm.league_id = seasons.league_id AND lm.user_id = auth.uid() AND lm.status = 'active' AND lm.role IN ('owner', 'admin')));

-- Invites.
DROP POLICY IF EXISTS league_invites_admin_manage ON public.league_invites;

CREATE POLICY league_invites_admin_manage
ON public.league_invites
FOR ALL TO authenticated
USING (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = league_invites.league_id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
      AND lm.role IN ('owner', 'admin')
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1
    FROM public.league_members lm
    WHERE lm.league_id = league_invites.league_id
      AND lm.user_id = auth.uid()
      AND lm.status = 'active'
      AND lm.role IN ('owner', 'admin')
  )
);
