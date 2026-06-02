import type { SupabaseClient, User } from "@supabase/supabase-js";
import type { League, Match, MmrHistory, Player } from "@/lib/demo-data";

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

  const roleMap = new Map(rows.map((row) => [row.league_id, row.role]));
  return ((leagueRows || []) as League[]).map((league) => ({
    ...league,
    role: roleMap.get(league.id) || "member"
  }));
}

export async function loadLeagueData(supabase: SupabaseClient, leagueId: number, matchLimit = 160): Promise<LiveLeagueData> {
  const [playersResult, matchesResult, historyResult] = await Promise.all([
    supabase
      .from("players")
      .select("id,name,display_name,mmr,matches_played,wins,draws,losses,win_streak,lose_streak,fitness")
      .eq("league_id", leagueId)
      .eq("is_active", 1)
      .order("mmr", { ascending: false }),
    supabase
      .from("matches")
      .select("id,date,team_a,team_b,score,result,processed")
      .eq("league_id", leagueId)
      .order("date", { ascending: false })
      .limit(matchLimit),
    supabase
      .from("mmr_history")
      .select("id,player_id,match_id,date,mmr_before,mmr_after")
      .eq("league_id", leagueId)
      .order("date", { ascending: true })
  ]);

  if (playersResult.error) throw new Error(playersResult.error.message);
  if (matchesResult.error) throw new Error(matchesResult.error.message);
  if (historyResult.error) throw new Error(historyResult.error.message);

  return {
    players: (playersResult.data || []) as Player[],
    matches: (matchesResult.data || []) as Match[],
    mmrHistory: (historyResult.data || []) as MmrHistory[]
  };
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
