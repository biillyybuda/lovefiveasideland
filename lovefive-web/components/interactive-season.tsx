"use client";

import { useMemo, useState } from "react";
import {
  buildPlayerSummaries,
  duoChemistry,
  formatUkDate,
  normalizeName,
  resultFor,
  scoreParts,
  seasonBreakdown,
  splitTeam,
  type Match,
  type MmrHistory,
  type Player,
  type PlayerSummary
} from "@/lib/demo-data";
import { weightedFormScore } from "@/lib/form-engine";
import { applyPeriodMmr, findMostImprovedPlayers } from "@/lib/mmr-engine";

type Quarter = "full" | "q1" | "q2" | "q3" | "q4";
type SeasonSortKey = "mmr" | "played" | "record" | "goalDiff" | "mmrChange" | "player";
type ImprovementRow = {
  key: string;
  label: string;
  value: number;
  detail?: string;
};

const quarters: Array<{ value: Quarter; label: string; start: string; end: string }> = [
  { value: "full", label: "Full year", start: "01-01", end: "12-31" },
  { value: "q1", label: "Jan-Mar", start: "01-01", end: "03-31" },
  { value: "q2", label: "Apr-Jun", start: "04-01", end: "06-30" },
  { value: "q3", label: "Jul-Sep", start: "07-01", end: "09-30" },
  { value: "q4", label: "Oct-Dec", start: "10-01", end: "12-31" }
];

export function InteractiveSeason({
  players,
  matches,
  mmrHistory
}: {
  players: Player[];
  matches: Match[];
  mmrHistory: MmrHistory[];
}) {
  const seasons = useMemo(() => seasonBreakdown(matches), [matches]);
  const seasonYears = useMemo(() => seasons.map((item) => item.season), [seasons]);
  const seasonOptions = useMemo(() => ["all", ...seasons.map((item) => item.season)], [seasons]);
  const [season, setSeason] = useState(seasonOptions[0] || "all");
  const [quarter, setQuarter] = useState<Quarter>("full");
  const [sortKey, setSortKey] = useState<SeasonSortKey>("mmr");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const period = useMemo(() => getPeriod(season, quarter), [quarter, season]);
  const periodMatches = useMemo(
    () => matches.filter((match) => inPeriod(match.date, period.start, period.end)),
    [matches, period]
  );
  const periodSummaries = useMemo(() => {
    const base = buildPlayerSummaries(players, periodMatches);
    const withSeasonMmr = applyPeriodMmr(base, mmrHistory, season === "all" || quarter !== "full" ? "all" : season);
    return withSeasonMmr.map((player) => {
      const change = periodMmrChange(mmrHistory, player.id, period.start, period.end);
      return change
        ? { ...player, periodMmrStart: change.start, periodMmrEnd: change.end, periodMmrChange: Math.round(change.end - change.start) }
        : player;
    });
  }, [mmrHistory, period.end, period.start, periodMatches, players, quarter, season]);

  const activeSummaries = periodSummaries.filter((player) => Number(player.matches_played || 0) > 0);
  const ratingLeaders = [...activeSummaries].sort((a, b) => Number(b.mmr || 0) - Number(a.mmr || 0)).slice(0, 8);
  const bestFormRun = useMemo(() => findBestFormRun(activeSummaries), [activeSummaries]);
  const periodImproved: ImprovementRow[] = [...activeSummaries]
    .filter((player) => player.periodMmrChange !== null)
    .sort((a, b) => Number(b.periodMmrChange || 0) - Number(a.periodMmrChange || 0))
    .slice(0, 8)
    .map((player) => ({
      key: String(player.id),
      label: player.label,
      value: Number(player.periodMmrChange || 0)
    }));
  const seasonImproved = useMemo<ImprovementRow[]>(() => {
    const summaryScope = buildPlayerSummaries(players, matches);
    return findMostImprovedPlayers(summaryScope, mmrHistory, seasonYears, season)
      .slice(0, 8)
      .map((row) => ({
        key: String(row.player.id),
        label: row.player.label,
        value: row.improvementScore,
        detail: `${signed(row.currentGain)} this season`
      }));
  }, [matches, mmrHistory, players, season, seasonYears]);
  const improved = quarter === "full" ? seasonImproved : periodImproved;
  const improvedSubtitle = quarter === "full"
    ? "season gain compared with previous seasons"
    : "MMR movement inside the period";
  const attendance = [...activeSummaries]
    .map((player) => ({ player, pct: Math.round((Number(player.matches_played || 0) / Math.max(periodMatches.length, 1)) * 100) }))
    .sort((a, b) => b.pct - a.pct || Number(b.player.matches_played || 0) - Number(a.player.matches_played || 0))
    .slice(0, 5);
  const totalGoals = periodMatches.reduce((sum, match) => {
    const score = scoreParts(match.score);
    return sum + (score ? score[0] + score[1] : 0);
  }, 0);
  const drawCount = periodMatches.filter((match) => {
    const score = scoreParts(match.score);
    return score ? score[0] === score[1] : false;
  }).length;
  const heavyDefeats = useMemo(() => biggestDefeats(periodMatches), [periodMatches]);
  const teammateRows = useMemo(() => duoChemistry(players, periodMatches, "team").slice(0, 6), [periodMatches, players]);
  const rivalryRows = useMemo(() => duoChemistry(players, periodMatches, "opponent").slice(0, 6), [periodMatches, players]);
  const forgetRows = useMemo(() => periodToForget(activeSummaries).slice(0, 5), [activeSummaries]);

  const topRating = ratingLeaders[0];
  const topImproved = improved[0];
  const topDuo = teammateRows[0];
  const topRivalry = rivalryRows[0];
  const dateLabel = period.start && period.end
    ? `${formatUkDate(period.start)} to ${formatUkDate(period.end)}`
    : "All recorded matches";
  const seasonTableRows = useMemo(
    () => sortSeasonRows(activeSummaries, sortKey, sortDir),
    [activeSummaries, sortDir, sortKey]
  );

  function toggleSort(nextKey: SeasonSortKey) {
    if (nextKey === sortKey) {
      setSortDir((current) => current === "desc" ? "asc" : "desc");
      return;
    }
    setSortKey(nextKey);
    setSortDir(nextKey === "player" ? "asc" : "desc");
  }

  return (
    <div className="interactive-stack season-review-stack">
      <div className="stats-toolbar">
        <div>
          <span className="stat-label">Season review</span>
          <h2>{period.label}</h2>
          <small>{dateLabel}</small>
        </div>
        <div className="season-control-row">
          <label>
            <span>Season</span>
            <select value={season} onChange={(event) => setSeason(event.target.value)}>
              {seasonOptions.map((item) => (
                <option value={item} key={item}>{item === "all" ? "All seasons" : item}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Period</span>
            <select value={quarter} onChange={(event) => setQuarter(event.target.value as Quarter)} disabled={season === "all"}>
              {quarters.map((item) => (
                <option value={item.value} key={item.value}>{item.label}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="season-scoreboard">
        <div className="season-kpi-card"><span>Matches</span><strong>{periodMatches.length}</strong><small>games recorded</small></div>
        <div className="season-kpi-card"><span>Total goals</span><strong>{totalGoals}</strong><small>{(totalGoals / Math.max(periodMatches.length, 1)).toFixed(1)} per game</small></div>
        <div className="season-kpi-card"><span>Players used</span><strong>{activeSummaries.length}</strong><small>active in this view</small></div>
        <div className="season-kpi-card"><span>Draws</span><strong>{drawCount}</strong><small>shared points</small></div>
      </div>

      <section className="season-awards-grid">
        <AwardCard title="Rating leader" main={topRating?.label || "-"} detail={topRating ? `${Math.round(Number(topRating.mmr || 0))} MMR` : ""} />
        <AwardCard
          title="Best form in period"
          main={bestFormRun?.player.label || "-"}
          detail={bestFormRun ? `${bestFormRun.form.join(" ")} from ${formatUkDate(bestFormRun.startDate)} to ${formatUkDate(bestFormRun.endDate)}` : ""}
        />
        <AwardCard title="Most improved" main={topImproved?.label || "-"} detail={topImproved ? `${signed(topImproved.value)} MMR` : ""} />
        <AwardCard title="Attendance" main={attendance[0]?.player.label || "-"} detail={attendance[0] ? `${attendance[0].pct}% of games` : ""} />
      </section>

      <section className="season-spotlight-grid">
        <div className="season-spotlight-card">
          <span>Best duo</span>
          <strong>{topDuo ? `${topDuo.a} + ${topDuo.b}` : "-"}</strong>
          <small>{topDuo ? `${topDuo.matches} games, ${topDuo.wins}-${topDuo.draws}-${topDuo.losses}, Chem ${topDuo.score.toFixed(1)}` : "No duo data"}</small>
        </div>
        <div className="season-spotlight-card rivalry">
          <span>Big rivalry</span>
          <strong>{topRivalry ? `${topRivalry.a} vs ${topRivalry.b}` : "-"}</strong>
          <small>{topRivalry ? `${topRivalry.matches} meetings, ${topRivalry.wins}-${topRivalry.draws}-${topRivalry.losses}, Intensity ${topRivalry.score.toFixed(1)}` : "No rivalry data"}</small>
        </div>
      </section>

      <section className="two-col season-feature-grid">
        <div className="panel">
          <div className="section-subhead">
            <strong>Top ratings</strong>
            <span>MMR in this period</span>
          </div>
          <div className="rank-list compact-list">
            {ratingLeaders.map((player, index) => (
              <div className="rank-row" key={player.id}>
                <span>{index + 1}</span>
                <strong>{player.label}</strong>
                <em>{Math.round(Number(player.mmr || 0))}</em>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-subhead">
            <strong>Most improved</strong>
            <span>{improvedSubtitle}</span>
          </div>
          <div className="rank-list compact-list">
            {improved.map((player, index) => (
              <div className="rank-row" key={player.key}>
                <span>{index + 1}</span>
                <strong>{player.label}</strong>
                <em title={player.detail}>{signed(player.value)}</em>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="two-col season-feature-grid">
        <div className="panel">
          <div className="section-subhead">
            <strong>Biggest results</strong>
            <span>Heaviest score swings</span>
          </div>
          <div className="story-list">
            {heavyDefeats.map((row) => (
              <div key={`${row.date}-${row.score}-${row.loser}`}>
                <span>{formatUkDate(row.date)}</span>
                <strong>{row.score}</strong>
                <small>{row.loser} lost by {row.margin}</small>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-subhead">
            <strong>Period to forget</strong>
            <span>Low form, goal swing and rating movement</span>
          </div>
          <div className="story-list">
            {forgetRows.map((row) => (
              <div key={row.player.id}>
                <span>{row.player.matches_played} matches</span>
                <strong>{row.player.label}</strong>
                <small>{signed(row.player.goalDiff)} GD, {signed(Number(row.player.periodMmrChange || 0))} MMR</small>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="two-col season-feature-grid">
        <RelationshipPanel title="Best duos" subtitle="Same-side chemistry" rows={teammateRows} joiner="+" />
        <RelationshipPanel title="Big rivalries" subtitle="Repeated head-to-head matchups" rows={rivalryRows} joiner="vs" />
      </section>

      <section className="panel">
        <div className="section-subhead">
          <strong>Attendance board</strong>
          <span>Percentage of available matches played</span>
        </div>
        <div className="season-bars">
          {attendance.map(({ player, pct }) => (
            <div className="season-row" key={player.id}>
              <strong>{player.label}</strong>
              <span style={{ width: `${Math.max(8, pct)}%` }} />
              <em>{pct}%</em>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="section-subhead">
          <strong>Season table</strong>
          <span>Tap a column to sort for screenshots</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th><button className="table-sort-button" onClick={() => toggleSort("player")}>Player {sortMarker(sortKey, sortDir, "player")}</button></th>
                <th><button className="table-sort-button" onClick={() => toggleSort("mmr")}>MMR {sortMarker(sortKey, sortDir, "mmr")}</button></th>
                <th><button className="table-sort-button" onClick={() => toggleSort("played")}>Played {sortMarker(sortKey, sortDir, "played")}</button></th>
                <th><button className="table-sort-button" onClick={() => toggleSort("record")}>W-D-L {sortMarker(sortKey, sortDir, "record")}</button></th>
                <th><button className="table-sort-button" onClick={() => toggleSort("goalDiff")}>Goal Diff {sortMarker(sortKey, sortDir, "goalDiff")}</button></th>
                <th><button className="table-sort-button" onClick={() => toggleSort("mmrChange")}>MMR Change {sortMarker(sortKey, sortDir, "mmrChange")}</button></th>
              </tr>
            </thead>
            <tbody>
              {seasonTableRows
                .map((player) => (
                  <tr key={player.id}>
                    <td>{player.label}</td>
                    <td>{Math.round(Number(player.mmr || 0))}</td>
                    <td>{player.matches_played}</td>
                    <td>{player.wins}-{player.draws}-{player.losses}</td>
                    <td>{signed(player.goalDiff)}</td>
                    <td>{player.periodMmrChange === null ? "-" : signed(Number(player.periodMmrChange))}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function AwardCard({ title, main, detail }: { title: string; main: string; detail: string }) {
  return (
    <div className="stat-card award-card">
      <span>{title}</span>
      <strong>{main}</strong>
      <small>{detail}</small>
    </div>
  );
}

function RelationshipPanel({
  title,
  subtitle,
  rows,
  joiner
}: {
  title: string;
  subtitle: string;
  rows: ReturnType<typeof duoChemistry>;
  joiner: "+" | "vs";
}) {
  return (
    <div className="panel">
      <div className="section-subhead">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <div className="story-list relationship-awards">
        {rows.map((row) => (
          <div key={`${row.a}-${row.b}`}>
            <span>{joiner === "+" ? `Chem ${row.score.toFixed(1)}` : `Intensity ${row.score.toFixed(1)}`}</span>
            <strong>{row.a} {joiner} {row.b}</strong>
            <small>
              {row.matches} matches, {row.wins}-{row.draws}-{row.losses}
              {joiner === "+" ? `, ${Math.round(row.winPct)}% wins` : `, ${row.avgGoalDiff.toFixed(1)} avg gap`}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
}

function getPeriod(season: string, quarter: Quarter) {
  if (season === "all") {
    return { start: null, end: null, label: "All seasons" };
  }
  const row = quarters.find((item) => item.value === quarter) || quarters[0];
  return {
    start: `${season}-${row.start}`,
    end: `${season}-${row.end}`,
    label: `${season} - ${row.label}`
  };
}

function inPeriod(date: string | null | undefined, start: string | null, end: string | null) {
  const value = String(date || "").slice(0, 10);
  if (!value) return false;
  if (start && value < start) return false;
  if (end && value > end) return false;
  return true;
}

function periodMmrChange(history: MmrHistory[], playerId: number, start: string | null, end: string | null) {
  const rows = history
    .filter((row) => row.player_id === playerId && inPeriod(row.date, start, end))
    .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")) || Number(a.id || 0) - Number(b.id || 0));
  if (!rows.length) return null;
  return {
    start: Number(rows[0].mmr_before || 1000),
    end: Number(rows[rows.length - 1].mmr_after || rows[0].mmr_before || 1000)
  };
}

function biggestDefeats(matches: Match[]) {
  return matches
    .map((match) => {
      const score = scoreParts(match.score);
      if (!score) return null;
      const margin = Math.abs(score[0] - score[1]);
      const loser = score[0] === score[1] ? "Draw" : score[0] > score[1] ? "Team B" : "Team A";
      return { date: match.date, score: match.score || "-", margin, loser };
    })
    .filter((row): row is { date: string | null; score: string; margin: number; loser: string } => Boolean(row && row.margin > 0))
    .sort((a, b) => b.margin - a.margin)
    .slice(0, 5);
}

function periodToForget(players: PlayerSummary[]) {
  return players
    .filter((player) => Number(player.matches_played || 0) >= 3)
    .map((player) => ({
      player,
      score: (Number(player.wins || 0) * 3 + Number(player.draws || 0))
        + player.goalDiff
        + Number(player.periodMmrChange || 0) / 5
    }))
    .sort((a, b) => a.score - b.score);
}

function findBestFormRun(players: PlayerSummary[]) {
  return players
    .flatMap((player) => {
      const appearances = playerAppearances(player).sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
      const windowSize = Math.min(5, appearances.length);
      if (!windowSize) return [];

      return appearances.slice(0, Math.max(1, appearances.length - windowSize + 1)).map((_, index) => {
        const window = appearances.slice(index, index + windowSize);
        const form = window.map((item) => item.result);
        const score = weightedFormScore({ form: [...form].reverse() } as Pick<PlayerSummary, "form">);
        const goalDiff = window.reduce((sum, item) => sum + item.goalDiff, 0);
        return {
          player,
          form,
          score,
          goalDiff,
          startDate: window[0]?.date,
          endDate: window[window.length - 1]?.date
        };
      });
    })
    .sort((a, b) => b.score - a.score || b.goalDiff - a.goalDiff || Number(b.player.mmr || 0) - Number(a.player.mmr || 0))[0];
}

function playerAppearances(player: PlayerSummary) {
  const keys = [player.name, player.display_name, player.label].map(normalizeName).filter(Boolean);
  return player.allMatches.flatMap((match) => {
    const score = scoreParts(match.score);
    if (!score) return [];
    const teamA = splitTeam(match.team_a).map(normalizeName);
    const teamB = splitTeam(match.team_b).map(normalizeName);
    const side = keys.some((key) => teamA.includes(key)) ? "A" : keys.some((key) => teamB.includes(key)) ? "B" : null;
    if (!side) return [];
    return [{
      date: match.date,
      result: resultFor(match, side),
      goalDiff: side === "A" ? score[0] - score[1] : score[1] - score[0]
    }];
  });
}

function sortSeasonRows(players: PlayerSummary[], sortKey: SeasonSortKey, sortDir: "asc" | "desc") {
  const direction = sortDir === "asc" ? 1 : -1;
  return [...players].sort((a, b) => {
    const primary = compareSeasonValue(a, b, sortKey);
    if (primary !== 0) return primary * direction;
    return a.label.localeCompare(b.label);
  });
}

function compareSeasonValue(a: PlayerSummary, b: PlayerSummary, sortKey: SeasonSortKey) {
  if (sortKey === "player") return a.label.localeCompare(b.label);
  if (sortKey === "mmr") return Number(a.mmr || 0) - Number(b.mmr || 0);
  if (sortKey === "played") return Number(a.matches_played || 0) - Number(b.matches_played || 0);
  if (sortKey === "goalDiff") return a.goalDiff - b.goalDiff;
  if (sortKey === "mmrChange") return Number(a.periodMmrChange || 0) - Number(b.periodMmrChange || 0);
  return recordPoints(a) - recordPoints(b);
}

function recordPoints(player: PlayerSummary) {
  return Number(player.wins || 0) * 3 + Number(player.draws || 0);
}

function sortMarker(current: SeasonSortKey, direction: "asc" | "desc", key: SeasonSortKey) {
  if (current !== key) return "";
  return direction === "asc" ? "^" : "v";
}

function signed(value: number) {
  return `${value > 0 ? "+" : ""}${Math.round(value)}`;
}
