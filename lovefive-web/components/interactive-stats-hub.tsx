"use client";

import { useMemo, useState } from "react";
import { InteractiveCharts } from "@/components/interactive-charts";
import { InteractiveChemistry } from "@/components/interactive-chemistry";
import { InteractiveInsights } from "@/components/interactive-insights";
import { InteractiveMatches } from "@/components/interactive-matches";
import { buildPlayerSummaries, duoChemistry, type Match, type MmrHistory, type Player } from "@/lib/demo-data";
import { applyPeriodMmr } from "@/lib/mmr-engine";

type View = "overview" | "matches" | "player" | "relationships";
type RelationshipMode = "head" | "team" | "matchup";

export function InteractiveStatsHub({
  initialPlayer,
  initialSeason,
  initialView,
  reportBasePath = "/demo/matches",
  players,
  matches,
  mmrHistory
}: {
  initialPlayer?: string;
  initialSeason?: string;
  initialView?: string;
  reportBasePath?: string;
  players: Player[];
  matches: Match[];
  mmrHistory: MmrHistory[];
}) {
  const seasons = useMemo(() => ["all", ...Array.from(new Set(matches.map((match) => String(match.date || "").slice(0, 4))))], [matches]);
  const [season, setSeason] = useState(() => initialSeason && seasons.includes(initialSeason) ? initialSeason : "all");
  const [view, setView] = useState<View>(() => coerceView(initialView || null));
  const [relationshipMode] = useState<RelationshipMode>(() => coerceRelationshipMode(initialView || null));

  const scopedMatches = useMemo(
    () => matches.filter((match) => season === "all" || String(match.date || "").startsWith(season)),
    [matches, season]
  );
  const summaries = useMemo(
    () => applyPeriodMmr(buildPlayerSummaries(players, scopedMatches), mmrHistory, season),
    [mmrHistory, players, scopedMatches, season]
  );
  const teammateRows = useMemo(() => duoChemistry(players, scopedMatches, "team"), [players, scopedMatches]);
  const matchupRows = useMemo(() => duoChemistry(players, scopedMatches, "opponent"), [players, scopedMatches]);
  const scopedHistory = useMemo(
    () => mmrHistory.filter((row) => season === "all" || String(row.date || "").startsWith(season)),
    [mmrHistory, season]
  );

  return (
    <div className="interactive-stack">
      <div className="stats-toolbar">
        <div>
          <span>Stats Centre</span>
          <strong>{viewLabel(view)}</strong>
          <small>{scopedMatches.length} processed matches in view</small>
        </div>
        <label>
          <span>Season View</span>
          <select value={season} onChange={(event) => setSeason(event.target.value)}>
            {seasons.map((item) => (
              <option value={item} key={item}>{item === "all" ? "Rolling (all years)" : item}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="segmented">
        <button className={view === "overview" ? "active" : ""} onClick={() => setView("overview")}>Overview</button>
        <button className={view === "matches" ? "active" : ""} onClick={() => setView("matches")}>Match History</button>
        <button className={view === "player" ? "active" : ""} onClick={() => setView("player")}>Player Insights</button>
        <button className={view === "relationships" ? "active" : ""} onClick={() => setView("relationships")}>Relationships</button>
      </div>

      {view === "overview" ? (
        <InteractiveCharts
          enablePreviousGames={season === "all"}
          players={summaries}
          mmrHistory={scopedHistory}
          matchCount={scopedMatches.length}
        />
      ) : null}

      {view === "player" ? (
        <section id="player-insights">
          <InteractiveInsights
            initialPlayerId={initialPlayer}
            players={summaries}
            rawPlayers={players}
            matches={scopedMatches}
          />
        </section>
      ) : null}

      {view === "matches" ? (
        <section id="match-history">
          <InteractiveMatches matches={scopedMatches} players={players} reportBasePath={reportBasePath} />
        </section>
      ) : null}

      {view === "relationships" ? (
        <InteractiveChemistry
          teammateRows={teammateRows}
          matchupRows={matchupRows}
          players={players}
          matches={scopedMatches}
          initialMode={relationshipMode}
        />
      ) : null}
    </div>
  );
}

function viewLabel(view: View) {
  if (view === "overview") return "Overview";
  if (view === "matches") return "Match History";
  if (view === "player") return "Player Insights";
  return "Relationships";
}

function coerceView(value: string | null): View {
  if (value === "summary" || value === "global") return "overview";
  if (value === "head" || value === "team" || value === "matchup") return "relationships";
  if (value === "players") return "player";
  if (value === "matches" || value === "player" || value === "relationships") return value;
  return "overview";
}

function coerceRelationshipMode(value: string | null): RelationshipMode {
  if (value === "team") return "team";
  if (value === "matchup") return "matchup";
  return "head";
}
