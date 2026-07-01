-- Love Five web onboarding RPCs.
--
-- Apply this in Supabase before opening public web signup. These functions let
-- signed-in users create their first league or join by code without weakening
-- row-level policies on leagues and league_members.

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_index i
    JOIN pg_class t ON t.oid = i.indrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
      AND t.relname = 'leagues'
      AND i.indisunique
      AND pg_get_indexdef(i.indexrelid) ILIKE '%(join_code)%'
  ) THEN
    CREATE UNIQUE INDEX leagues_join_code_unique_idx
    ON public.leagues (join_code)
    WHERE join_code IS NOT NULL;
  END IF;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_index i
    JOIN pg_class t ON t.oid = i.indrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    WHERE n.nspname = 'public'
      AND t.relname = 'league_members'
      AND i.indisunique
      AND pg_get_indexdef(i.indexrelid) ILIKE '%(league_id, user_id)%'
  ) THEN
    CREATE UNIQUE INDEX league_members_league_user_unique_idx
    ON public.league_members (league_id, user_id);
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.lovefive_generate_join_code(league_name text)
RETURNS text
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = public, pg_temp
AS $$
DECLARE
  prefix text;
BEGIN
  prefix := upper(substr(regexp_replace(coalesce(league_name, ''), '[^a-zA-Z0-9]', '', 'g'), 1, 4));
  IF length(prefix) < 2 THEN
    prefix := 'LFIV';
  END IF;

  RETURN prefix || upper(substr(md5(random()::text || clock_timestamp()::text), 1, 6));
END;
$$;

CREATE OR REPLACE FUNCTION public.create_league_for_current_user(league_name text)
RETURNS TABLE (league_id bigint, name text, join_code text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  active_user uuid;
  clean_name text;
  generated_code text;
  created_league record;
  attempt integer := 0;
BEGIN
  active_user := auth.uid();
  IF active_user IS NULL THEN
    RAISE EXCEPTION 'You must be signed in to create a league.';
  END IF;

  clean_name := trim(regexp_replace(coalesce(league_name, ''), '\s+', ' ', 'g'));
  IF length(clean_name) < 3 THEN
    RAISE EXCEPTION 'League name must be at least 3 characters.';
  END IF;

  LOOP
    attempt := attempt + 1;
    generated_code := public.lovefive_generate_join_code(clean_name);

    BEGIN
      INSERT INTO public.leagues (name, join_code)
      VALUES (clean_name, generated_code)
      RETURNING id, leagues.name, leagues.join_code
      INTO created_league;
      EXIT;
    EXCEPTION WHEN unique_violation THEN
      IF attempt >= 10 THEN
        RAISE EXCEPTION 'Could not generate a unique league code. Try again.';
      END IF;
    END;
  END LOOP;

  INSERT INTO public.league_members (league_id, user_id, role, status)
  VALUES (created_league.id, active_user, 'admin', 'active')
  ON CONFLICT ON CONSTRAINT league_members_pkey
  DO UPDATE SET role = 'admin', status = 'active';

  league_id := created_league.id;
  name := created_league.name;
  join_code := created_league.join_code;
  RETURN NEXT;
END;
$$;

CREATE OR REPLACE FUNCTION public.join_league_by_code(invite_code text)
RETURNS TABLE (league_id bigint, name text, role text)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  active_user uuid;
  clean_code text;
  matched_league record;
BEGIN
  active_user := auth.uid();
  IF active_user IS NULL THEN
    RAISE EXCEPTION 'You must be signed in to join a league.';
  END IF;

  clean_code := upper(trim(coalesce(invite_code, '')));
  IF clean_code = '' THEN
    RAISE EXCEPTION 'Enter a league code.';
  END IF;

  SELECT id, leagues.name
  INTO matched_league
  FROM public.leagues
  WHERE upper(join_code) = clean_code
  LIMIT 1;

  IF matched_league.id IS NULL THEN
    RAISE EXCEPTION 'That code does not match a league.';
  END IF;

  INSERT INTO public.league_members (league_id, user_id, role, status)
  VALUES (matched_league.id, active_user, 'member', 'active')
  ON CONFLICT ON CONSTRAINT league_members_pkey
  DO UPDATE SET status = 'active';

  league_id := matched_league.id;
  name := matched_league.name;
  role := 'member';
  RETURN NEXT;
END;
$$;

REVOKE ALL ON FUNCTION public.lovefive_generate_join_code(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.create_league_for_current_user(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.join_league_by_code(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.lovefive_generate_join_code(text) FROM anon;
REVOKE ALL ON FUNCTION public.create_league_for_current_user(text) FROM anon;
REVOKE ALL ON FUNCTION public.join_league_by_code(text) FROM anon;

GRANT EXECUTE ON FUNCTION public.create_league_for_current_user(text) TO authenticated;
GRANT EXECUTE ON FUNCTION public.join_league_by_code(text) TO authenticated;
