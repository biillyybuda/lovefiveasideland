import type { SupabaseClient } from "@supabase/supabase-js";
import type { Match, Player } from "./demo-data";
import { loadAllPages } from "./pagination";

export type BalancePreviewWorkspace = { players: Player[]; matches: Match[]; token: string | null };

/** Preview can read the original schema; recording still requires the migration. */
export async function loadBalancePreview(supabase: SupabaseClient, leagueId: number): Promise<BalancePreviewWorkspace> {
  const { data, error } = await supabase.rpc("rating_workspace", { p_league_id: leagueId });
  if (!error) {
    if (!data?.token || !Array.isArray(data.players) || !Array.isArray(data.matches)) throw new Error("Incomplete rating workspace.");
    return { players: data.players, matches: data.matches, token: data.token };
  }
  // Do not disguise permission, network or calculation errors as a missing migration.
  if (!["PGRST202", "42883"].includes(error.code)) throw new Error(error.message);
  const { data: auth, error: authError } = await supabase.auth.getUser();
  if (authError || !auth.user) throw new Error("Sign in to preview the new model.");
  const membership = await supabase.from("league_members").select("role")
    .eq("league_id", leagueId).eq("user_id", auth.user.id).eq("status", "active").maybeSingle();
  if (membership.error) throw new Error(membership.error.message);
  if (!["owner", "admin"].includes(membership.data?.role)) throw new Error("League admin access is needed for this preview.");
  const [players, matches] = await Promise.all([
    loadAllPages<Player>((from, to) => supabase.from("players")
      .select("id,name,display_name,mmr,matches_played,wins,draws,losses,win_streak,lose_streak,fitness,is_active,position")
      .eq("league_id", leagueId).order("id").range(from, to)),
    loadAllPages<Match>((from, to) => supabase.from("matches")
      .select("id,date,team_a,team_b,score,result,processed")
      .eq("league_id", leagueId).eq("processed", 1).order("date").order("id").range(from, to))
  ]);
  return { players, matches, token: null };
}
