import type { SupabaseClient } from "@supabase/supabase-js";
import { displayName, splitTeam, type Match, type Player } from "./demo-data";

export const matchActivityFields = [
  { key: "distance_km", label: "Distance (km)", min: 0, max: 100, step: "any", placeholder: "e.g. 5.2" },
  { key: "duration_minutes", label: "Total / playing time (minutes or m:ss)", min: 0.01, max: 600, step: "any", placeholder: "e.g. 60" },
  { key: "average_heart_rate", label: "Match average HR (bpm)", min: 1, max: 300, step: "1", placeholder: "e.g. 145" },
  { key: "max_heart_rate", label: "Match peak HR (bpm)", min: 1, max: 300, step: "1", placeholder: "e.g. 175" }
] as const;
export const baselineActivityFields = [
  { key: "resting_heart_rate", label: "Resting heart rate (bpm)", min: 1, max: 300, step: "1", placeholder: "e.g. 60" },
  { key: "personal_max_heart_rate", label: "Personal maximum HR (bpm)", min: 1, max: 300, step: "1", placeholder: "If known" }
] as const;
export const watchActivityFields = [
  { key: "average_pace_seconds", label: "Average pace (m:ss/km)", min: 1, max: 7200, step: "1", placeholder: "e.g. 14:42" },
  { key: "moving_pace_seconds", label: "Average moving pace (m:ss/km)", min: 1, max: 7200, step: "1", placeholder: "e.g. 12:33" },
  { key: "best_pace_seconds", label: "Best pace (m:ss/km)", min: 1, max: 7200, step: "1", placeholder: "e.g. 4:04" },
  { key: "average_speed_kmh", label: "Average speed (km/h)", min: 0, max: 60, step: "any", placeholder: "e.g. 4.1" },
  { key: "moving_speed_kmh", label: "Average moving speed (km/h)", min: 0, max: 60, step: "any", placeholder: "e.g. 4.8" },
  { key: "max_speed_kmh", label: "Maximum speed (km/h)", min: 0, max: 60, step: "any", placeholder: "e.g. 14.7" },
  { key: "moving_minutes", label: "Moving time (minutes or m:ss)", min: 0, max: 600, step: "any", placeholder: "e.g. 47:14" },
  { key: "elapsed_minutes", label: "Elapsed time (minutes or m:ss)", min: 0.01, max: 600, step: "any", placeholder: "e.g. 58:15" },
  { key: "calories", label: "Total calories (kcal)", min: 0, max: 10000, step: "1", placeholder: "e.g. 807" },
  { key: "average_respiration", label: "Average breathing rate (breaths/min)", min: 1, max: 120, step: "1", placeholder: "e.g. 37" },
  { key: "min_respiration", label: "Minimum breathing rate (breaths/min)", min: 1, max: 120, step: "1", placeholder: "e.g. 21" },
  { key: "max_respiration", label: "Maximum breathing rate (breaths/min)", min: 1, max: 120, step: "1", placeholder: "e.g. 50" }
] as const;
export const activityFields = [...matchActivityFields, ...baselineActivityFields, ...watchActivityFields];
export type ActivityMetric = typeof activityFields[number]["key"];
export type ActivityField = typeof activityFields[number];
export type ActivityValues = Record<ActivityMetric, number | null> & { source: string };
export type MatchActivity = ActivityValues & { league_id: number; match_id: number; player_id: number; version: string; updated_at: string };
export type ActivityDraft = { [K in keyof ActivityValues]: string };
export type ActivityPlayer = { id: number; name: string; side: "A" | "B" };
export const activitySelect = "league_id,match_id,player_id,distance_km,duration_minutes,average_heart_rate,max_heart_rate,resting_heart_rate,personal_max_heart_rate,average_pace_seconds,moving_pace_seconds,best_pace_seconds,average_speed_kmh,moving_speed_kmh,max_speed_kmh,moving_minutes,elapsed_minutes,calories,average_respiration,min_respiration,max_respiration,source,version,updated_at";
export const isPace = (key: ActivityMetric) => key.endsWith("_pace_seconds");
export const isDuration = (key: ActivityMetric) => key === "duration_minutes" || key === "moving_minutes" || key === "elapsed_minutes";
export function clockValue(seconds: number): string {
  const rounded = Math.round(seconds);
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, "0")}`;
}
export function formatActivityValue(key: ActivityMetric, value: number | null | undefined): string {
  if (value == null) return "—";
  if (isPace(key)) return clockValue(value);
  if (isDuration(key)) return clockValue(value * 60);
  return value.toLocaleString("en-GB", { maximumFractionDigits: key === "distance_km" ? 2 : key.endsWith("_kmh") ? 1 : 0 });
}

export function activityDraft(row?: Partial<ActivityValues>): ActivityDraft {
  return Object.fromEntries([...activityFields.map(f => {
    const value = row?.[f.key];
    // Keep arbitrary decimal-minute input intact; show exact whole-second readings as m:ss.
    const text = value == null ? "" : isPace(f.key) ? clockValue(value) : isDuration(f.key) && Math.abs(value * 60 - Math.round(value * 60)) < 1e-7 ? clockValue(value * 60) : value.toString();
    return [f.key, text];
  }), ["source", row?.source || ""]]) as ActivityDraft;
}

export function parseActivity(draft: ActivityDraft): ActivityValues {
  const result = Object.fromEntries([...activityFields.map(f => [f.key, null]), ["source", draft.source.trim()]]) as ActivityValues;
  for (const field of activityFields) {
    const raw = (draft[field.key] || "").trim();
    if (!raw) continue;
    const timed = isPace(field.key) || isDuration(field.key);
    let value: number;
    if (timed && raw.includes(":")) {
      if (!/^\d{1,3}:[0-5]\d(?::[0-5]\d)?$/.test(raw) || (isPace(field.key) && raw.split(":").length !== 2))
        throw new Error(`${field.label}: use minutes:seconds${isDuration(field.key) ? " or hours:minutes:seconds" : ""}.`);
      const seconds = raw.split(":").reduce((total, part) => total * 60 + Number(part), 0);
      value = isDuration(field.key) ? seconds / 60 : seconds;
    } else {
      if (!/^\d+(?:\.\d+)?$/.test(raw)) throw new Error(`${field.label}: enter a valid number${timed ? " or minutes:seconds" : ""}.`);
      value = Number(raw);
    }
    if (!Number.isFinite(value) || value < field.min || value > field.max || (field.step === "1" && !Number.isInteger(value)))
      throw new Error(`${field.label}: enter ${field.step === "1" ? "a whole number" : "a number"} between ${field.min} and ${field.max}${isPace(field.key) ? " seconds per km" : ""}.`);
    result[field.key] = value;
  }
  if (!activityFields.some(f => result[f.key] !== null)) throw new Error("Enter at least one activity figure. Leave anything you do not have blank.");
  if (result.average_heart_rate !== null && result.max_heart_rate !== null && result.average_heart_rate > result.max_heart_rate)
    throw new Error("Match peak heart rate must be at least the match average heart rate.");
  if (result.resting_heart_rate !== null && result.personal_max_heart_rate !== null && result.personal_max_heart_rate <= result.resting_heart_rate)
    throw new Error("Personal maximum heart rate must be higher than resting heart rate.");
  for (const [low, high, message] of [
    ["average_speed_kmh", "max_speed_kmh", "Maximum speed must be at least average speed."],
    ["moving_speed_kmh", "max_speed_kmh", "Maximum speed must be at least average moving speed."],
    ["min_respiration", "average_respiration", "Average breathing rate must be at least the minimum."],
    ["average_respiration", "max_respiration", "Maximum breathing rate must be at least the average."],
    ["min_respiration", "max_respiration", "Maximum breathing rate must be at least the minimum."]
  ] as [ActivityMetric, ActivityMetric, string][]) {
    if (result[low] !== null && result[high] !== null && result[low]! > result[high]!) throw new Error(message);
  }
  if (result.source.length > 80) throw new Error("Keep the watch or app name to 80 characters.");
  return result;
}

export function activityPlayers(match: Match, players: Player[]): ActivityPlayer[] {
  const normalise = (value: string | null) => (value || "").trim().replace(/\s+/g, " ").toLowerCase();
  const lookup = new Map<string, Player>();
  for (const player of players) for (const name of [player.name, player.display_name]) if (normalise(name)) lookup.set(normalise(name), player);
  const seen = new Set<number>();
  return (["A", "B"] as const).flatMap(side => splitTeam(side === "A" ? match.team_a : match.team_b).flatMap(name => {
    const player = lookup.get(normalise(name));
    if (!player || seen.has(player.id)) return [];
    seen.add(player.id);
    return [{ id: player.id, name: displayName(player), side }];
  }));
}

function activityError(error: { code?: string; message: string }) {
  if (["PGRST202", "PGRST204", "PGRST205", "42P01", "42703"].includes(error.code || "")) return new Error("Player activity needs the latest activity database migrations. See the website README.");
  return new Error(error.code === "40001" ? "This activity was changed by someone else. Your unsaved figures are still here. Copy anything you need, cancel, then reload saved activity before editing again." : error.message);
}

export async function loadMatchActivity(client: SupabaseClient, leagueId: number, matchId: number): Promise<MatchActivity[]> {
  const { data, error } = await client.from("match_player_activity")
    .select(activitySelect)
    .eq("league_id", leagueId).eq("match_id", matchId).order("player_id");
  if (error) throw activityError(error);
  return (data || []) as MatchActivity[];
}

export async function savePlayerActivity(client: SupabaseClient, leagueId: number, matchId: number, playerId: number,
  original: MatchActivity | undefined, values: ActivityValues | null): Promise<MatchActivity | null> {
  // Validate in the shared write path as well as the form. Null means explicit removal.
  if (values) values = parseActivity(activityDraft(values));
  const { data, error } = await client.rpc("save_match_player_activity", { p_league_id: leagueId, p_match_id: matchId,
    p_player_id: playerId, p_expected_version: original?.version || null, p_values: values, p_remove: values === null });
  if (error) throw activityError(error);
  if (values && (!data?.version || data.player_id !== playerId || data.match_id !== matchId || data.league_id !== leagueId))
    throw new Error("Could not confirm the saved activity. Reload saved activity before trying again.");
  if (values && baselineActivityFields.some(field => (data[field.key] ?? null) !== values[field.key]))
    throw new Error("Could not confirm the saved heart-rate baselines. Your figures are still here. Reload saved activity and check the latest database migration before trying again.");
  if (values && watchActivityFields.some(field => (data[field.key] ?? null) !== values[field.key]))
    throw new Error("Could not confirm the saved watch readings. Your figures are still here. Reload saved activity and check the latest database migration before trying again.");
  return data as MatchActivity | null;
}
