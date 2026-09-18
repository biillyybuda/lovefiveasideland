import type { SupabaseClient } from "@supabase/supabase-js";

export const MATCH_NOTE_LIMIT = 1500;

export async function saveMatchNote(supabase: SupabaseClient, leagueId: number, matchId: number, expected: string, note: string) {
  if (note.length > MATCH_NOTE_LIMIT) throw new Error(`Keep the note to ${MATCH_NOTE_LIMIT} characters.`);
  const { data, error } = await supabase.rpc("save_match_review_note", { p_league_id: leagueId, p_match_id: matchId,
    p_expected_note: expected, p_note: note.trim() });
  if (error) {
    if (error.code === "40001") throw new Error("Someone updated this note. Refresh the report before saving your changes.");
    if (error.code === "PGRST202") throw new Error("Match notes need the match_review_notes database migration. See Database setup in the website README.");
    throw new Error(error.message);
  }
  return data as string;
}
