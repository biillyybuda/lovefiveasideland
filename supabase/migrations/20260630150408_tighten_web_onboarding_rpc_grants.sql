REVOKE ALL ON FUNCTION public.lovefive_generate_join_code(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.create_league_for_current_user(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.join_league_by_code(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.lovefive_generate_join_code(text) FROM anon;
REVOKE ALL ON FUNCTION public.create_league_for_current_user(text) FROM anon;
REVOKE ALL ON FUNCTION public.join_league_by_code(text) FROM anon;
GRANT EXECUTE ON FUNCTION public.create_league_for_current_user(text) TO authenticated;
GRANT EXECUTE ON FUNCTION public.join_league_by_code(text) TO authenticated;;
