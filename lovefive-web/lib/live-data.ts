import { loadCupNames } from "./cup-settings";
import { cupName } from "./competition";
import { loadAllPages } from "./pagination";
import { loadCompletedMatches } from "./match-loader";
import type { SupabaseClient, User } from "@supabase/supabase-js";
import type { League, Match, MmrHistory, Player } from "@/lib/demo-data";
import {
  attachStyleProfiles,
  type HighlightMomentRow,
  type MatchStatRow
} from "@/lib/player-style-profiles";

export type Membership = {
  league_id: number;
  role: string | null;
  status: string | null;
};

export type LeagueOption = League & {
  role: string | null;
};

export type LiveLeagueData = {
  players: Player[];
  modelPlayers: Player[];
  matches: Match[];
  mmrHistory: MmrHistory[];
};

export function isAdminRole(role: string | null | undefined) {
  return ["admin", "owner"].includes(String(role || "").toLowerCase());
}

export function displayPlayerName(player: Pick<Player, "name" | "display_name">) {
  return player.display_name?.trim() || player.name;
}

export function canonicalName(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

export function tidyName(value: string) {
  return value
    .trim()
    .replace(/\s+/g, " ")
    .split(" ")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

export async function loadMyLeagues(supabase: SupabaseClient, user: User): Promise<LeagueOption[]> {
  const { data: memberships, error: membershipError } = await supabase
    .from("league_members")
    .select("league_id,role,status")
    .eq("user_id", user.id)
    .eq("status", "active");

  if (membershipError) throw new Error(membershipError.message);

  const rows = (memberships || []) as Membership[];
  const leagueIds = rows.map((row) => row.league_id).filter(Boolean);
  if (!leagueIds.length) return [];

  const { data: leagueRows, error: leagueError } = await supabase
    .from("leagues")
    .select("id,name,join_code")
    .in("id", leagueIds)
    .order("name", { ascending: true });

  if (leagueError) throw new Error(leagueError.message);

  const cupNames = await loadCupNames(supabase, leagueIds);
  const roleMap = new Map(rows.map((row) => [row.league_id, row.role]));
  return ((leagueRows || []) as League[]).map((league) => ({
    ...league,
    cup_name: cupName(cupNames.get(league.id)),
    role: roleMap.get(league.id) || "member"
  }));
}

export async function loadLeagueData(supabase: SupabaseClient, leagueId: number, _matchLimit = 160): Promise<LiveLeagueData> {
  // Analytical inputs are complete; each screen applies its own visible-list limit.
  const [players, matches, mmrHistory, stats, highlights] = await Promise.all([
    loadAllPages<Player>((from, to) => supabase.from("players")
      .select("id,name,display_name,mmr,matches_played,wins,draws,losses,win_streak,lose_streak,fitness,position,is_active")
      .eq("league_id", leagueId).order("mmr", { ascending: false }).order("id").range(from, to)),
    loadCompletedMatches(supabase, leagueId),
    loadAllPages<MmrHistory>((from, to) => supabase.from("mmr_history")
      .select("id,player_id,match_id,date,mmr_before,mmr_after")
      .eq("league_id", leagueId).order("date").order("id").range(from, to)),
    loadAllPages<MatchStatRow>((from, to) => supabase.from("match_stats")
      .select("match_id,player_name,goals,assists").eq("league_id", leagueId).order("id").range(from, to))
      .catch(error => { console.warn("Player statistics unavailable", error.message); return []; }),
    loadAllPages<HighlightMomentRow>((from, to) => supabase.from("highlight_moments")
      .select("match_id,player_name,timestamp_sec,label,type,special_tag,goal_tag,assist_tag,save_tag,save_importance")
      .eq("league_id", leagueId).order("id").range(from, to))
      .catch(error => { console.warn("Player highlights unavailable", error.message); return []; })
  ]);
  const completed = new Set(matches.map(m => m.id));
  const enriched = attachStyleProfiles(players, stats.filter(r => completed.has(r.match_id!)), highlights.filter(r => completed.has(r.match_id!)));
  return { players: enriched.filter(p => p.is_active === 1), modelPlayers: enriched, matches, mmrHistory };
}

export function selectedLeagueFromStorage(leagues: LeagueOption[]) {
  if (!leagues.length) return null;
  const saved = Number(window.localStorage.getItem("lovefive-selected-league") || 0);
  if (saved && leagues.some((league) => league.id === saved)) return saved;
  return leagues.length === 1 ? leagues[0].id : null;
}

export function saveSelectedLeague(leagueId: number) {
  window.localStorage.setItem("lovefive-selected-league", String(leagueId));
}

export function clearSelectedLeague() {
  window.localStorage.removeItem("lovefive-selected-league");
}
