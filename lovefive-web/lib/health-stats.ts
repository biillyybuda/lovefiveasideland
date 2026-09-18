import type { SupabaseClient } from "@supabase/supabase-js";
import { displayName, scoreParts, type Match, type Player } from "./demo-data";
import { activityFields, activityPlayers, activitySelect, watchActivityFields, type ActivityMetric, type MatchActivity } from "./match-activity";
import { loadAllPages } from "./pagination";

export type HealthEntry = { match: Match; reading: MatchActivity | null };
export type DatedReading = { value: number; date: string; matchId: number };
export type PlayerHealth = {
  player: Player; name: string; entries: HealthEntry[]; recorded: HealthEntry[];
  distanceCount: number; totalDistance: number | null; averageDistance: number | null;
  timeCount: number; totalMinutes: number | null; pairedCount: number; metresPerMinute: number | null;
  heartRateCount: number; averageHeartRate: number | null; peakCount: number; peakHeartRate: number | null;
  speedCount: number; topSpeed: number | null;
  movingShare: number | null; movingCount: number;
  workRateChange: { baseline: number; latest: number; percent: number | null } | null;
  firstWorkRate: DatedReading | null; latestWorkRate: DatedReading | null;
  latestResting: DatedReading | null; latestPersonalMax: DatedReading | null;
};
export type HealthStats = ReturnType<typeof buildHealthStats>;
export type HealthLoadState = { status: "loading" | "ready" | "error"; records: MatchActivity[]; message?: string };
export type HealthContext = { data: HealthStats; status: HealthLoadState["status"]; message?: string;
  reload: () => void; canEdit: boolean; seasonLabel: string; reportBasePath: string };
export type TrendMetric = ActivityMetric | "metres_per_minute";
export const trendMetrics: { key: TrendMetric; label: string; unit: string; decimals: number }[] = [
  { key: "distance_km", label: "Distance", unit: "km", decimals: 2 },
  { key: "metres_per_minute", label: "Distance per minute", unit: "m/min", decimals: 1 },
  { key: "duration_minutes", label: "Total / playing time", unit: "min", decimals: 1 },
  { key: "average_heart_rate", label: "Match average heart rate", unit: "bpm", decimals: 0 },
  { key: "max_heart_rate", label: "Match peak heart rate", unit: "bpm", decimals: 0 },
  { key: "resting_heart_rate", label: "Resting heart rate", unit: "bpm", decimals: 0 },
  { key: "personal_max_heart_rate", label: "Personal maximum heart rate", unit: "bpm", decimals: 0 },
  ...watchActivityFields.map(field => ({ key: field.key, label: field.label.replace(/ \(.*\)$/, ""),
    unit: field.key.endsWith("_pace_seconds") ? "m:ss/km" : field.key.endsWith("_kmh") ? "km/h" : field.key.endsWith("_minutes") ? "min" : field.key === "calories" ? "kcal" : "breaths/min",
    decimals: field.key.endsWith("_kmh") || field.key.endsWith("_minutes") ? 1 : 0 }))
];
export const boardMetrics = [
  { key: "metresPerMinute", label: "Distance per minute", unit: "m/min", decimals: 1 },
  { key: "topSpeed", label: "Top recorded speed", unit: "km/h", decimals: 1 }
] as const;
export type BoardMetric = typeof boardMetrics[number]["key"];
export type ProfileMetric = "name" | "metresPerMinute" | "topSpeed" | "movingShare" | "change";
export type ProfileSort = { metric: ProfileMetric; direction: "ascending" | "descending" };

export async function loadLeagueActivity(client: SupabaseClient, leagueId: number): Promise<MatchActivity[]> {
  if (!Number.isSafeInteger(leagueId) || leagueId <= 0) throw new Error("Choose a league to load activity.");
  return loadAllPages<MatchActivity>((from, to) => client.from("match_player_activity")
    .select(activitySelect)
    .eq("league_id", leagueId).order("match_id").order("player_id").range(from, to));
}

function validDate(date: string | null): date is string {
  if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return false;
  const time = Date.parse(`${date}T00:00:00Z`);
  return Number.isFinite(time) && new Date(time).toISOString().slice(0, 10) === date;
}

function cleanReading(row: MatchActivity): MatchActivity | null {
  const clean = { ...row };
  for (const field of activityFields) {
    const value = row[field.key];
    clean[field.key] = typeof value === "number" && Number.isFinite(value) && value >= field.min && value <= field.max &&
      (field.step !== "1" || Number.isInteger(value)) ? value : null;
  }
  return activityFields.some(field => clean[field.key] !== null) ? clean : null;
}

export function metricValue(entry: HealthEntry, key: TrendMetric): number | null {
  const row = entry.reading;
  if (!row) return null;
  if (key === "metres_per_minute") return row.distance_km !== null && row.duration_minutes !== null && row.duration_minutes > 0
    ? row.distance_km * 1000 / row.duration_minutes : null;
  return row[key] ?? null;
}

const sum = (values: number[]) => values.length ? values.reduce((a, b) => a + b, 0) : null;
const mean = (values: number[]) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
const median = (values: number[]) => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const workRate = (entries: HealthEntry[]) => entries.reduce((s, e) => s + e.reading!.distance_km! * 1000, 0) /
  entries.reduce((s, e) => s + e.reading!.duration_minutes!, 0);
function summarise(player: Player, entries: HealthEntry[]): PlayerHealth {
  const recorded = entries.filter(entry => entry.reading !== null);
  const values = (key: TrendMetric) => entries.map(e => metricValue(e, key)).filter((v): v is number => v !== null);
  const distances = values("distance_km"), minutes = values("duration_minutes"), rates = values("average_heart_rate"), peaks = values("max_heart_rate");
  const speeds = values("max_speed_kmh");
  const paired = recorded.filter(e => e.reading!.distance_km !== null && e.reading!.duration_minutes !== null);
  // Garmin moving time is compared with elapsed time from the same recording.
  const moving = recorded.filter(({ reading: r }) => r!.moving_minutes !== null && r!.elapsed_minutes !== null && r!.moving_minutes! <= r!.elapsed_minutes!);
  const lastPaired = paired.at(-1);
  const baseline = paired.length >= 2 ? metricValue(paired[0], "metres_per_minute")! : null;
  const latest = paired.length >= 2 ? metricValue(lastPaired!, "metres_per_minute")! : null;
  const latestBaseline = (key: "resting_heart_rate" | "personal_max_heart_rate"): DatedReading | null => {
    const entry = [...recorded].reverse().find(e => e.reading![key] !== null);
    return entry ? { value: entry.reading![key]!, date: entry.match.date!, matchId: entry.match.id } : null;
  };
  return { player, name: displayName(player), entries, recorded,
    distanceCount: distances.length, totalDistance: sum(distances), averageDistance: mean(distances),
    timeCount: minutes.length, totalMinutes: sum(minutes), pairedCount: paired.length,
    // Use exactly the same matches in the numerator and denominator. Weight by duration.
    metresPerMinute: paired.length ? workRate(paired) : null,
    firstWorkRate: paired.length ? { value: metricValue(paired[0], "metres_per_minute")!, date: paired[0].match.date!, matchId: paired[0].match.id } : null,
    latestWorkRate: lastPaired ? { value: metricValue(lastPaired, "metres_per_minute")!, date: lastPaired.match.date!, matchId: lastPaired.match.id } : null,
    movingCount: moving.length,
    movingShare: moving.length ? moving.reduce((s, e) => s + e.reading!.moving_minutes!, 0) / moving.reduce((s, e) => s + e.reading!.elapsed_minutes!, 0) * 100 : null,
    workRateChange: baseline !== null && latest !== null ? { baseline, latest, percent: baseline > 0 ? (latest - baseline) / baseline * 100 : null } : null,
    speedCount: speeds.length, topSpeed: speeds.length ? Math.max(...speeds) : null,
    heartRateCount: rates.length, averageHeartRate: mean(rates), peakCount: peaks.length,
    peakHeartRate: peaks.length ? Math.max(...peaks) : null, latestResting: latestBaseline("resting_heart_rate"), latestPersonalMax: latestBaseline("personal_max_heart_rate") };
}

export function buildHealthStats(leagueId: number, players: Player[], matches: Match[], activities: MatchActivity[]) {
  // The caller supplies season-scoped matches. Ignore orphaned, pending and duplicate results.
  const seenMatches = new Set<number>();
  const games = matches.filter(m => m.processed === 1 && validDate(m.date) && scoreParts(m.score) && !seenMatches.has(m.id) && seenMatches.add(m.id))
    .sort((a, b) => a.date!.localeCompare(b.date!) || a.id - b.id);
  const readings = new Map<string, MatchActivity>();
  for (const row of activities) {
    if (row.league_id !== leagueId) continue;
    const key = `${row.match_id}:${row.player_id}`, current = readings.get(key);
    if (!current || String(row.updated_at) > String(current.updated_at)) readings.set(key, row);
  }
  const entries = new Map<number, HealthEntry[]>();
  const coverage = games.map(match => {
    const participants = activityPlayers(match, players);
    let recorded = 0;
    for (const player of participants) {
      const raw = readings.get(`${match.id}:${player.id}`);
      const reading = raw ? cleanReading(raw) : null;
      if (reading) recorded++;
      const history = entries.get(player.id) || [];
      history.push({ match, reading }); entries.set(player.id, history);
    }
    return { match, participants: participants.length, recorded };
  });
  const uniquePlayers = [...new Map(players.map(p => [p.id, p])).values()];
  const people = uniquePlayers.map(p => summarise(p, entries.get(p.id) || [])).sort((a, b) => a.name.localeCompare(b.name) || a.player.id - b.player.id);
  return { players: people, coverage, matchCount: games.length,
    typicalWorkRate: median(people.flatMap(p => p.metresPerMinute === null ? [] : [p.metresPerMinute])),
    typicalMovingShare: median(people.flatMap(p => p.movingShare === null ? [] : [p.movingShare])),
    participantCount: people.filter(p => p.entries.length).length,
    playersWithRecords: people.filter(p => p.recorded.length).length,
    matchesWithRecords: coverage.filter(c => c.recorded).length,
    recordCount: coverage.reduce((s, c) => s + c.recorded, 0), possibleRecords: coverage.reduce((s, c) => s + c.participants, 0),
    distanceCount: people.reduce((s, p) => s + p.distanceCount, 0),
    totalDistance: sum(people.flatMap(p => p.totalDistance === null ? [] : [p.totalDistance])),
    totalMinutes: sum(people.flatMap(p => p.totalMinutes === null ? [] : [p.totalMinutes])) };
}

export function healthLeaderboard(players: PlayerHealth[], metric: BoardMetric, minGames: number) {
  const decimals = boardMetrics.find(m => m.key === metric)!.decimals;
  const rows = players.map(player => {
    const value = player[metric];
    const count = metric === "metresPerMinute" ? player.pairedCount : metric === "topSpeed" ? player.speedCount : player.distanceCount;
    return { player, value, count, rounded: value === null ? null : Number(value.toFixed(decimals)) };
  }).filter((row): row is typeof row & { value: number; rounded: number } => row.value !== null && row.count >= Math.max(1, minGames));
  rows.sort((a, b) => b.rounded - a.rounded || a.player.name.localeCompare(b.player.name) || a.player.player.id - b.player.player.id);
  let rank = 0;
  return rows.map((row, index) => { if (index === 0 || row.rounded !== rows[index - 1].rounded) rank = index + 1; return { ...row, rank }; });
}

export function runningProfiles(players: PlayerHealth[], sort: ProfileSort) {
  const value = (player: PlayerHealth) => sort.metric === "change" ? player.workRateChange?.percent ?? null : player[sort.metric];
  return players.filter(p => p.metresPerMinute !== null || p.topSpeed !== null || p.movingShare !== null).sort((a, b) => {
    const left = value(a), right = value(b);
    // Missing readings stay last in either direction. A recorded zero is still a value.
    if (left === null && right !== null) return 1;
    if (right === null && left !== null) return -1;
    const comparison = typeof left === "string" && typeof right === "string" ? left.localeCompare(right) :
      typeof left === "number" && typeof right === "number" ? left - right : 0;
    return comparison * (sort.direction === "ascending" ? 1 : -1) || a.name.localeCompare(b.name) || a.player.id - b.player.id;
  });
}

export function healthTrend(player: PlayerHealth, metric: TrendMetric, limit = 12, recordedOnly = false) {
  const available = recordedOnly ? player.entries.filter(entry => metricValue(entry, metric) !== null) : player.entries;
  const entries = limit > 0 ? available.slice(-limit) : available;
  return entries.map(entry => ({ ...entry, value: metricValue(entry, metric) }));
}
