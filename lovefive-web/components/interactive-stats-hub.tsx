"use client";

import { useCompetitionLabels } from "./cup-name-provider";
import { useMemo, useState } from "react";
import { InteractiveCharts } from "@/components/interactive-charts";
import { InteractiveChemistry } from "@/components/interactive-chemistry";
import { InteractiveInsights } from "@/components/interactive-insights";
import { InteractiveMatches } from "@/components/interactive-matches";
import { buildPlayerSummaries, duoChemistry, type Match, type MmrHistory, type Player } from "@/lib/demo-data";
import { applyPeriodMmr } from "@/lib/mmr-engine";
import { LeagueHealthActivity } from "@/components/health-activity";
import { buildHealthStats, type HealthContext, type HealthLoadState } from "@/lib/health-stats";
import { selectMatches } from "@/lib/competition";
import { CompetitionControl, useCompetition } from "@/components/competition-control";
import { ShieldRecords } from "@/components/shield-records";

type View = "overview" | "matches" | "player" | "relationships" | "health";
type RelationshipMode = "head" | "team" | "matchup";

export function InteractiveStatsHub({
  initialPlayer,
  initialCompetition,
  initialSeason,
  initialView,
  initialPlayerSection,
  reportBasePath = "/demo/matches",
  players,
  historicalPlayers,
  matches,
  mmrHistory,
  health
}: {
  initialPlayer?: string;
  initialCompetition?: string;
  initialSeason?: string;
  initialView?: string;
  initialPlayerSection?: string;
  reportBasePath?: string;
  players: Player[];
  historicalPlayers?: Player[];
  matches: Match[];
  mmrHistory: MmrHistory[];
  health?: { leagueId: number; players: Player[]; load: HealthLoadState; reload: () => void; canEdit: boolean };
}) {
  const competitionLabels = useCompetitionLabels();
  const [competition, setCompetition] = useCompetition(initialCompetition);
  const seasons = useMemo(() => ["all", ...Array.from(new Set(matches.map((match) => String(match.date || "").slice(0, 4))))], [matches]);
  const [season, setSeason] = useState(() => initialSeason && seasons.includes(initialSeason) ? initialSeason : "all");
  const [view, setView] = useState<View>(() => initialView === "health" && !health ? "overview" : coerceView(initialView || null));
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(() => Number(initialPlayer) || null);
  const [playerSection, setPlayerSection] = useState<"football" | "health">(() => health && initialPlayerSection === "health" ? "health" : "football");
  const [relationshipMode] = useState<RelationshipMode>(() => coerceRelationshipMode(initialView || null));

  const scopedMatches = useMemo(
    () => selectMatches(matches, competition, season),
    [matches, season, competition]
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
  const insightPlayers = health?.players || historicalPlayers || players;
  const insightSummaries = useMemo(() => applyPeriodMmr(buildPlayerSummaries(insightPlayers, scopedMatches), mmrHistory, season),
    [insightPlayers, scopedMatches, mmrHistory, season]);
  const healthStats = useMemo(() => health ? buildHealthStats(health.leagueId, health.players, scopedMatches, health.load.records) : null,
    [health?.leagueId, health?.players, health?.load.records, scopedMatches]);
  const healthContext: HealthContext | undefined = health && healthStats ? { data: healthStats, status: health.load.status,
    message: health.load.message, reload: health.reload, canEdit: health.canEdit, seasonLabel: `${competitionLabels[competition]} · ${season === "all" ? "All recorded seasons" : season}`, reportBasePath } : undefined;

  return (
    <div className="interactive-stack competition-stats">
      <div className="stats-toolbar">
        <div>
          <span>Stats Centre</span>
          <strong>{viewLabel(view)}</strong>
          <small>{view === "health" || (view === "player" && playerSection === "health") ? "Player fitness and running profiles" : `${scopedMatches.length} processed matches in view`}</small>
        </div>
        <CompetitionControl value={competition} onChange={setCompetition} />
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
        {health && <button className={view === "health" ? "active" : ""} aria-pressed={view === "health"} onClick={() => setView("health")}>Health & Activity</button>}
      </div>

      {view === "overview" && competition === "immunity_shield" ? <ShieldRecords players={insightSummaries} /> : null}
      {view === "overview" && competition !== "immunity_shield" ? (
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
            competition={competition}
            initialPlayerId={initialPlayer}
            selectedPlayerId={selectedPlayerId}
            onPlayerChange={setSelectedPlayerId}
            section={playerSection}
            onSectionChange={setPlayerSection}
            health={healthContext}
            players={insightSummaries}
            rawPlayers={insightPlayers}
            matches={scopedMatches}
          />
        </section>
      ) : null}

      {view === "health" && healthContext && <LeagueHealthActivity health={healthContext} onPlayer={id => {
        setSelectedPlayerId(id); setPlayerSection("health"); setView("player");
      }}/>}

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
  if (view === "health") return "Health & Activity";
  return "Relationships";
}

function coerceView(value: string | null): View {
  if (value === "summary" || value === "global") return "overview";
  if (value === "head" || value === "team" || value === "matchup") return "relationships";
  if (value === "players") return "player";
  if (value === "matches" || value === "player" || value === "relationships" || value === "health") return value;
  return "overview";
}

function coerceRelationshipMode(value: string | null): RelationshipMode {
  if (value === "team") return "team";
  if (value === "matchup") return "matchup";
  return "head";
}
