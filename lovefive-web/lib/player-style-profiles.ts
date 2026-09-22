export type MatchStatRow = {
  match_id: number | null;
  player_name: string | null;
  goals: number | null;
  assists: number | null;
};

export type HighlightMomentRow = {
  match_id: number | null;
  player_name: string | null;
  timestamp_sec: number | null;
  label: string | null;
  type: string | null;
  special_tag: string | null;
  goal_tag: string | null;
  assist_tag: string | null;
  save_tag: string | null;
  save_importance: number | null;
};

export type PlayerStyleProfile = {
  finishing: number;
  creation: number;
  saveImpact: number;
  impact: number;
  clutch: number;
  finisherRatio: number;
  creatorRatio: number;
  historicGoals: number;
  historicAssists: number;
  weightedGoals: number;
  weightedAssists: number;
  goalMoments: number;
  assistMoments: number;
  saveMoments: number;
  specialMoments: number;
  historicMatches: number;
  goalsPerGame: number;
  assistsPerGame: number;
  saveMomentsPerGame: number;
  weightedGoalsPerGame: number;
  weightedAssistsPerGame: number;
  saveImpactPerGame: number;
  clutchPerGame: number;
};

const PROFILE_RATE_CONFIDENCE_GAMES = 8;

export function attachStyleProfiles<T extends { name: string }>(
  players: T[],
  matchStats: MatchStatRow[],
  highlights: HighlightMomentRow[]
): Array<T & { style_profile?: PlayerStyleProfile | null; style_source: { stats: MatchStatRow[]; highlights: HighlightMomentRow[] } }> {
  const profiles = buildStyleProfiles(matchStats, highlights);
  return players.map((player) => ({
    ...player,
    style_source: {
      stats: matchStats.filter(row => cleanName(row.player_name) === cleanName(player.name)),
      highlights: highlights.filter(row => cleanName(row.player_name) === cleanName(player.name))
    },
    style_profile: profiles.get(cleanName(player.name)) || null
  }));
}

function buildStyleProfiles(matchStats: MatchStatRow[], highlights: HighlightMomentRow[]) {
  const rawGoals = new Map<string, number>();
  const rawAssists = new Map<string, number>();
  const weightedGoals = new Map<string, number>();
  const weightedAssists = new Map<string, number>();
  const saveImpact = new Map<string, number>();
  const clutch = new Map<string, number>();
  const goalMoments = new Map<string, number>();
  const assistMoments = new Map<string, number>();
  const saveMoments = new Map<string, number>();
  const specialMoments = new Map<string, number>();
  const statMatches = new Map<string, Set<number>>();
  const profileMatches = new Map<string, Set<number>>();

  for (const row of matchStats) {
    const player = cleanName(row.player_name);
    if (!player) continue;
    add(rawGoals, player, numberOr(row.goals, 0));
    add(rawAssists, player, numberOr(row.assists, 0));
    if (Number.isFinite(Number(row.match_id))) {
      addMatch(statMatches, player, Number(row.match_id));
      addMatch(profileMatches, player, Number(row.match_id));
    }
  }

  for (const row of highlights) {
    const player = cleanName(row.player_name);
    if (!player) continue;
    if (Number.isFinite(Number(row.match_id))) {
      addMatch(profileMatches, player, Number(row.match_id));
    }
    const type = cleanName(row.type);
    const special = specialBonus(row.special_tag);
    if (special > 0) add(specialMoments, player, 1);

    if (type === "goal") {
      const weight = tagWeight(row.goal_tag, "goal") + special;
      add(weightedGoals, player, weight);
      add(goalMoments, player, 1);
      if (weight >= 1.55) add(clutch, player, weight);
    } else if (type === "assist") {
      const weight = tagWeight(row.assist_tag, "assist") + special;
      add(weightedAssists, player, weight);
      add(assistMoments, player, 1);
      if (weight >= 1.55) add(clutch, player, weight * 0.8);
    } else if (type === "save") {
      const weight = tagWeight(row.save_tag, "save")
        + special
        + Math.min(0.75, Math.max(0, numberOr(row.save_importance, 0) * 0.15));
      add(saveImpact, player, weight);
      add(saveMoments, player, 1);
    }
  }

  const players = new Set([
    ...rawGoals.keys(),
    ...rawAssists.keys(),
    ...weightedGoals.keys(),
    ...weightedAssists.keys(),
    ...saveImpact.keys(),
    ...profileMatches.keys()
  ]);
  const finishRaw = new Map<string, number>();
  const createRaw = new Map<string, number>();
  const impactRaw = new Map<string, number>();
  const rateStats = new Map<string, {
    matches: number;
    goalsPerGame: number;
    assistsPerGame: number;
    saveMomentsPerGame: number;
    weightedGoalsPerGame: number;
    weightedAssistsPerGame: number;
    saveImpactPerGame: number;
    clutchPerGame: number;
  }>();

  for (const player of players) {
    const goals = valueOf(rawGoals, player);
    const assists = valueOf(rawAssists, player);
    const wGoals = valueOf(weightedGoals, player);
    const wAssists = valueOf(weightedAssists, player);
    const saves = valueOf(saveImpact, player);
    const clutchValue = valueOf(clutch, player);
    const matches = profileGameCount(player, profileMatches, goals + assists + valueOf(saveMoments, player));
    const goalsPerGame = perGame(goals, matches);
    const assistsPerGame = perGame(assists, matches);
    const saveMomentsPerGame = perGame(valueOf(saveMoments, player), matches);
    const weightedGoalsPerGame = perGame(wGoals, matches);
    const weightedAssistsPerGame = perGame(wAssists, matches);
    const saveImpactPerGame = perGame(saves, matches);
    const clutchPerGame = perGame(clutchValue, matches);
    const finishing = reliableRate(goals, matches) * 0.55 + reliableRate(wGoals, matches) * 0.45;
    const creation = reliableRate(assists, matches) * 0.55 + reliableRate(wAssists, matches) * 0.45;
    const saveRate = reliableRate(saves, matches);
    const clutchRate = reliableRate(clutchValue, matches);
    finishRaw.set(player, finishing);
    createRaw.set(player, creation);
    impactRaw.set(player, finishing + creation * 0.85 + saveRate * 0.65 + clutchRate * 0.35);
    rateStats.set(player, {
      matches,
      goalsPerGame,
      assistsPerGame,
      saveMomentsPerGame,
      weightedGoalsPerGame,
      weightedAssistsPerGame,
      saveImpactPerGame,
      clutchPerGame
    });
  }

  const finishing = normalizeScores(finishRaw);
  const creation = normalizeScores(createRaw);
  const impact = normalizeScores(impactRaw);
  const saves = normalizeScores(new Map([...rateStats].map(([player, stats]) => [player, reliableRate(valueOf(saveImpact, player), stats.matches)])));
  const clutchScores = normalizeScores(new Map([...rateStats].map(([player, stats]) => [player, reliableRate(valueOf(clutch, player), stats.matches)])));
  const profiles = new Map<string, PlayerStyleProfile>();

  for (const player of players) {
    const finishValue = valueOf(finishRaw, player);
    const createValue = valueOf(createRaw, player);
    const total = finishValue + createValue;
    const rates = rateStats.get(player) || {
      matches: 0,
      goalsPerGame: 0,
      assistsPerGame: 0,
      saveMomentsPerGame: 0,
      weightedGoalsPerGame: 0,
      weightedAssistsPerGame: 0,
      saveImpactPerGame: 0,
      clutchPerGame: 0
    };
    profiles.set(player, {
      finishing: valueOf(finishing, player),
      creation: valueOf(creation, player),
      saveImpact: valueOf(saves, player),
      impact: valueOf(impact, player),
      clutch: valueOf(clutchScores, player),
      finisherRatio: total > 0 ? finishValue / total : 0.5,
      creatorRatio: total > 0 ? createValue / total : 0.5,
      historicGoals: valueOf(rawGoals, player),
      historicAssists: valueOf(rawAssists, player),
      weightedGoals: valueOf(weightedGoals, player),
      weightedAssists: valueOf(weightedAssists, player),
      goalMoments: valueOf(goalMoments, player),
      assistMoments: valueOf(assistMoments, player),
      saveMoments: valueOf(saveMoments, player),
      specialMoments: valueOf(specialMoments, player),
      historicMatches: rates.matches || statMatches.get(player)?.size || 0,
      goalsPerGame: rates.goalsPerGame,
      assistsPerGame: rates.assistsPerGame,
      saveMomentsPerGame: rates.saveMomentsPerGame,
      weightedGoalsPerGame: rates.weightedGoalsPerGame,
      weightedAssistsPerGame: rates.weightedAssistsPerGame,
      saveImpactPerGame: rates.saveImpactPerGame,
      clutchPerGame: rates.clutchPerGame
    });
  }

  return profiles;
}

function tagWeight(value: string | null, kind: "goal" | "assist" | "save") {
  const tag = cleanName(value);
  if (!tag) return 1;

  if (kind === "save") {
    if (tag.includes("unbelievable")) return 1.75;
    if (tag.includes("crucial") || tag.includes("clutch")) return 1.6;
    if (tag.includes("big save") || tag.includes("keeps them alive")) return 1.4;
    if (tag.includes("strong")) return 1.2;
    if (tag.includes("routine")) return 0.65;
    return 1;
  }

  if (tag.includes("game-winning")) return 2;
  if (tag.includes("game-saving")) return 1.9;
  if (tag.includes("equaliser")) return 1.7;
  if (tag.includes("go-ahead")) return 1.55;
  if (tag.includes("pulls one back")) return 1.25;
  if (tag.includes("doubling")) return 1.1;
  if (tag.includes("extending")) return 0.95;
  if (tag.includes("rout")) return 0.55;
  return 1;
}

function specialBonus(value: string | null) {
  const tag = cleanName(value);
  if (tag.includes("worldie") || tag.includes("brilliant assist") || tag.includes("unbelievable save")) {
    return 0.35;
  }
  return 0;
}

function normalizeScores(raw: Map<string, number>) {
  const values = [...raw.values()].filter((value) => value > 0).sort((a, b) => a - b);
  const out = new Map<string, number>();
  if (!values.length) return out;
  const index = Math.round((values.length - 1) * 0.85);
  const cap = Math.max(values[index] || 0, Math.max(...values) * 0.35, 0.05);
  for (const [key, value] of raw) {
    out.set(key, Math.max(0, Math.min(1, value / cap)));
  }
  return out;
}

function addMatch(map: Map<string, Set<number>>, key: string, matchId: number) {
  const rows = map.get(key) || new Set<number>();
  rows.add(matchId);
  map.set(key, rows);
}

function profileGameCount(player: string, matches: Map<string, Set<number>>, eventFallback: number) {
  const knownMatches = matches.get(player)?.size || 0;
  if (knownMatches > 0) return knownMatches;
  return eventFallback > 0 ? 1 : 0;
}

function perGame(total: number, matches: number) {
  return matches > 0 ? total / matches : 0;
}

function reliableRate(total: number, matches: number) {
  if (matches <= 0) return 0;
  const confidence = Math.min(1, matches / PROFILE_RATE_CONFIDENCE_GAMES);
  return perGame(total, matches) * confidence;
}

function add(map: Map<string, number>, key: string, value: number) {
  map.set(key, valueOf(map, key) + value);
}

function valueOf(map: Map<string, number>, key: string) {
  return Number(map.get(key) || 0);
}

function numberOr(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function cleanName(value: string | null | undefined) {
  return String(value || "").trim().toLowerCase();
}
