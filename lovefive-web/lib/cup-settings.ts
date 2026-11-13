import type { SupabaseClient } from "@supabase/supabase-js";
import { cupName, validateCupName } from "./competition";

export async function loadCupNames(supabase: SupabaseClient, leagueIds: number[]) {
  if (!leagueIds.length) return new Map<number, string>();
  const { data, error } = await supabase.from("league_cup_settings")
    .select("league_id,cup_name").in("league_id", leagueIds);
  if (error) throw new Error(`Could not load cup names: ${error.message}`);
  return new Map<number, string>((data || []).map(row => [row.league_id, cupName(row.cup_name)]));
}

export async function saveCupName(supabase: SupabaseClient, leagueId: number, value: string) {
  const name = validateCupName(value);
  // Database policies enforce active admin/owner membership for this league.
  const { data, error } = await supabase.from("league_cup_settings")
    .upsert({ league_id: leagueId, cup_name: name }, { onConflict: "league_id" })
    .select("cup_name").single();
  if (error) throw new Error(error.message);
  if (!data) throw new Error("The cup name was not saved. Check your league admin access.");
  return cupName(data.cup_name);
}
