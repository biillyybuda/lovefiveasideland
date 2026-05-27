-- Optional Supabase speed indexes for Love Five.
-- Run once in Supabase SQL Editor. Safe to rerun.
CREATE INDEX IF NOT EXISTS idx_matches_league_processed_date
ON public.matches (league_id, processed, date DESC);

CREATE INDEX IF NOT EXISTS idx_matches_league_result_date
ON public.matches (league_id, result, date DESC);

CREATE INDEX IF NOT EXISTS idx_mmr_history_match_id
ON public.mmr_history (match_id);

CREATE INDEX IF NOT EXISTS idx_mmr_history_player_id
ON public.mmr_history (player_id);

CREATE INDEX IF NOT EXISTS idx_mmr_history_player_date_id
ON public.mmr_history (player_id, date DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_mmr_history_match_player
ON public.mmr_history (match_id, player_id);

CREATE INDEX IF NOT EXISTS idx_players_league_name
ON public.players (league_id, name);
