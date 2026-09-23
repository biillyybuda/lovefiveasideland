"use client";

import { useMemo } from "react";
import { Shirt } from "lucide-react";
import { buildPlayerSummaries, displayName, type Player } from "@/lib/demo-data";
import { compareMatches, resolveLineup, type BalanceOption } from "@/lib/balance-model";
import { competitionLabels, matchCompetition, type Competition } from "@/lib/competition";
import type { buildMatchdayBalance } from "@/lib/matchday-balance";
import { LineupImageButton } from "@/components/lineup-image-button";
import { MatchDatePicker } from "@/components/match-date-picker";
import { MatchdayPlayerCards } from "@/components/matchday-player-cards";
import type { PlayerCardData } from "@/lib/player-cards-image";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

export function BalanceMatchdayCard({ competition = "league", option, model, players, date, rank, onCopy, copyStatus, onEditLineup, onDateChange }: {
  competition?: Competition;
  option: BalanceOption;
  model: ReturnType<typeof buildMatchdayBalance>;
  players: Player[];
  date: string;
  rank: number;
  onCopy: () => void;
  copyStatus: string;
  onEditLineup?: () => void;
  onDateChange: (date: string) => void;
}) {
  const context = useMemo(() => {
    const byId = new Map(players.map(player => [player.id, player]));
    // Use the same resolved identities and date cutoff as the forecast. These
    // records are display context only and never feed back into team selection.
    const history = [...model.history].sort((a, b) => compareMatches(b, a)).map(match => {
      const { a, b } = resolveLineup(match, players);
      return { ...match, team_a: a.map(id => byId.get(id)!.name).join(","), team_b: b.map(id => byId.get(id)!.name).join(",") };
    });
    return new Map(buildPlayerSummaries(players, history).map(player => [player.id, player]));
  }, [model.history, players]);
  const ratings = new Map(model.ratings.map(row => [row.playerId, row]));
  const sides = (["A", "B"] as const).map(side => {
    const team = option[side === "A" ? "teamA" : "teamB"];
    const roster = team.map(player => ({ player, rating: ratings.get(player.id)!, record: context.get(player.id)! }));
    return {
      side, team, roster,
      strength: average(roster.map(row => row.rating.rating)),
      uncertainty: average(roster.map(row => row.rating.uncertainty)),
      appearances: average(roster.map(row => row.rating.appearances)),
      form: roster.reduce((total, row) => total + row.record.form.slice(0, 5).reduce((sum, result) => sum + (result === "W" ? 3 : result === "D" ? 1 : 0), 0), 0),
      anchor: [...roster].sort((a, b) => b.rating.rating - a.rating.rating || a.player.id - b.player.id)[0],
      established: [...roster].sort((a, b) => b.rating.appearances - a.rating.appearances || a.rating.uncertainty - b.rating.uncertainty)[0],
      uncertain: [...roster].sort((a, b) => b.rating.uncertainty - a.rating.uncertainty || a.player.id - b.player.id)[0]
    };
  });
  const [a, b] = sides;
  function playerCard({ player, rating, record }: typeof a.roster[number]): PlayerCardData {
    return { name: displayName(player), mmr: player.mmr ?? 1000,
      games: rating.appearances, wins: record.wins ?? 0, draws: record.draws ?? 0, losses: record.losses ?? 0,
      form: record.recentMatches.slice(0, 5).map((match, i) => ({
        result: record.form[i] === "W" ? "W" : record.form[i] === "L" ? "L" : "D",
        shield: matchCompetition(match) === "immunity_shield", date: match.date
      })) };
  }
  const matchDate = new Date(`${date}T12:00:00Z`);
  const weekday = matchDate.toLocaleDateString("en-GB", { weekday: "long", timeZone: "UTC" });
  const dateLabel = matchDate.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
  const prediction = option.prediction;
  const gap = Math.abs(prediction.meanMargin);
  const favourite = prediction.win >= prediction.loss ? "Blue team" : "Red team";
  const imageData = { competition, date, blue: a.team.map(displayName), red: b.team.map(displayName), forecast: { win: prediction.win, draw: prediction.draw, loss: prediction.loss } };
  const comparisons = [
    { label: "Team strength", a: a.strength, b: b.strength, left: Math.round(a.strength), right: Math.round(b.strength) },
    { label: "Avg uncertainty", a: a.uncertainty, b: b.uncertainty, left: `±${Math.round(a.uncertainty)}`, right: `±${Math.round(b.uncertainty)}` },
    { label: "All-competition form", a: a.form, b: b.form, left: `${a.form} pts`, right: `${b.form} pts` },
    { label: "Games / player", a: a.appearances, b: b.appearances, left: a.appearances.toFixed(1), right: b.appearances.toFixed(1) }
  ];

  return <section className="panel matchday-card-panel balance-matchday-card" aria-label="Selected matchup forecast">
    <div className="panel-title-row">
      <div><span className="stat-label">Selected matchup</span><h2>Matchday Card</h2><small>{competitionLabels[competition]}{competition === "immunity_shield" ? " - MMR protected" : ""}</small></div>
      <div className="balance-card-actions">
        <a className="ghost-button" href="#balance-recommendations">Change matchup</a>
        <button className="ghost-button" type="button" onClick={onCopy}>Copy teams</button>
        {onEditLineup ? <button className="ghost-button" type="button" onClick={onEditLineup}>Edit lineup</button> : null}
        <LineupImageButton key={JSON.stringify(imageData)} data={imageData} />
      </div>
    </div>

    <div className="matchday-date-controls">
      <MatchDatePicker date={date} onChange={onDateChange} />
      <p className="muted">Choose when you&apos;re playing. This date appears on the lineup, copied teams and saved image.</p>
    </div>

    <section className="matchday-broadcast" aria-label="Matchday lineup card">
      <header className="broadcast-heading">
        <div><span className="broadcast-eyebrow">Love Five</span><h3>{competitionLabels[competition]} lineups</h3></div>
        <time className="broadcast-date" dateTime={date}><span>{weekday}</span>{" "}<strong>{dateLabel}</strong></time>
      </header>
      <div className="broadcast-teams" role="group" aria-label="Matchday teams">
        {sides.map(side => <section className={`broadcast-team broadcast-team-${side.side.toLowerCase()}`} aria-label={`${side.side === "A" ? "Blue" : "Red"} team`} key={side.side}>
          <header className="broadcast-team-heading">
            <Shirt aria-hidden="true" strokeWidth={1.6} />
            <h4>{side.side === "A" ? "Blue" : "Red"} team</h4>
          </header>
          <ul className="broadcast-lineup" aria-label={`${side.side === "A" ? "Blue" : "Red"} team players`}>
            {side.team.map(player => <li key={player.id}><strong>{displayName(player)}</strong></li>)}
          </ul>
        </section>)}
      </div>
      <section className="broadcast-forecast" aria-label="Match forecast">
        <h4>Match forecast</h4>
        <div className="broadcast-outcomes">
          <div className="broadcast-outcome-a"><span>Blue win</span><strong>{percent(prediction.win)}</strong></div>
          <div><span>Draw</span><strong>{percent(prediction.draw)}</strong></div>
          <div className="broadcast-outcome-b"><span>Red win</span><strong>{percent(prediction.loss)}</strong></div>
        </div>
      </section>
    </section>

    <div className="match-report-board">
      <section className="analysis-centrepiece">
        <div className="section-subhead"><strong>Team comparison</strong><span>Blue: Team A · Red: Team B</span></div>
        <div className="comparison-bar-list">{comparisons.map(row => {
          const max = Math.max(row.a, row.b, 1);
          return <div className="comparison-bar-row" key={row.label}>
            <div className="comparison-value left">{row.left}</div>
            <div className="comparison-track">
              <span className="bar-a" style={{ width: `${Math.max(0, row.a / max * 92)}%` }} />
              <em>{row.label}</em>
              <span className="bar-b" style={{ width: `${Math.max(0, row.b / max * 92)}%` }} />
            </div>
            <div className="comparison-value right">{row.right}</div>
          </div>;
        })}</div>
        <p className="balance-comparison-note">Smaller ± means a narrower player estimate. Form and games played are context; form uses each player&apos;s last five results (3 for a win, 1 for a draw).</p>
        <div className="analysis-signal-row balance-finish-row" aria-label="How tight might the score be?">
          <div><span>Finish within 2 goals</span><strong>{percent(prediction.close)}</strong><small>Draw, or either team wins by 1–2</small></div>
          <div><span>Win by 3+ goals</span><strong>{percent(1 - prediction.close)}</strong><small>Either team wins by 3 or more</small></div>
        </div>
        <p className="balance-comparison-note">These describe the size of the result, not which team wins.</p>
      </section>
      <aside className="analyst-verdict">
        <div className="section-subhead"><strong>Analyst verdict</strong></div>
        <div className="verdict-copy">
          <p>{Math.abs(prediction.win - prediction.loss) < 0.1
            ? "The win chances are similar, so there is little between the teams on the model’s forecast."
            : `${favourite} is favoured, with a ${percent(Math.max(prediction.win, prediction.loss))} chance of winning.`}</p>
          <p>{percent(prediction.close)} chance the game ends in a draw or a win by one or two goals. That can happen with either team on top.</p>
          <p className="warning">{prediction.close < 0.5
            ? `A win by three or more goals is more likely (${percent(1 - prediction.close)}), even with uncertainty included.`
            : "A close finish is more likely than a win by three or more goals, but neither is guaranteed."}</p>
        </div>
        <div className="verdict-tweak">
          <span>Selection read</span>
          <strong>{rank === 1 ? "Top suggestion with your current locks" : `Alternative ${rank - 1} with your current locks`}</strong>
          <small>The best available suggestion can still favour one side, especially with side locks. Positions and variety help order options with similar chances of a tight finish.</small>
        </div>
      </aside>
    </div>

    <div className="team-report-grid">{sides.map(side => <section className={`team-report-card team-report-${side.side.toLowerCase()}`} key={side.side}>
      <div className="section-subhead"><strong>{side.side === "A" ? "Blue" : "Red"} team · Team {side.side}</strong><span>{side.form} recent form pts</span></div>
      <div className="pill-row">{side.team.map(player => <span className="pill" key={player.id}>{displayName(player)}</span>)}</div>
      <div className="team-report-stats">
        <div><span>Top strength</span><strong>{displayName(side.anchor.player)} · {Math.round(side.anchor.rating.rating)}</strong></div>
        <div><span>Most games</span><strong>{displayName(side.established.player)} · {side.established.rating.appearances} played</strong></div>
        <div><span>Widest estimate</span><strong>{displayName(side.uncertain.player)} · ±{Math.round(side.uncertain.rating.uncertainty)}</strong></div>
      </div>
    </section>)}</div>

    <div className="scouting-grid balance-uncertainty-grid">
      <section className="scouting-card">
        <div className="section-subhead"><strong>How certain is this matchup?</strong><span>Uncertainty included in the forecast</span></div>
        <div className="balance-spread-stats">
          <div><span>Strength difference</span><strong>±{Math.sqrt(prediction.skillVariance).toFixed(1)} <small>goals</small></strong></div>
          <div><span>Full match forecast</span><strong>±{prediction.predictiveSd.toFixed(1)} <small>goals</small></strong></div>
        </div>
        <p className="muted">Both spreads are one standard deviation around the expected margin. The full forecast includes normal match variation as well as uncertainty about the players.</p>
        <details className="balance-margin-explainer"><summary>What is the average expected goal difference?</summary>
          <p>{gap < 0.05 ? "The model’s average expected goal difference is approximately zero."
            : `The model’s average expected goal difference is ${gap.toFixed(1)} goals in the ${prediction.meanMargin > 0 ? "blue" : "red"} team’s favour.`} This is the average across possible results, not a prediction that one game will finish by a fractional number of goals.</p>
          <p>Some results fall close to that average; others are wider wins, draws or wins for the other side. That is why the expected goal difference and the chance of finishing within two goals can look quite different.</p>
        </details>
      </section>
      <section className="scouting-card">
        <div className="section-subhead"><strong>What the numbers mean</strong><span>{model.history.length} completed games in the model</span></div>
        <div className="verdict-copy">
          <p>Team strength is the average of the five players&apos; strength estimates, on a scale centred on 1,000. The ± figure shows how uncertain a player&apos;s estimate is.</p>
          <p>Shared match history links player estimates. The forecast uses those links, rather than simply adding the individual ± figures.</p>
        </div>
        <small className="muted">Public MMR and the season award races keep their existing rules.</small>
      </section>
    </div>

    <MatchdayPlayerCards data={{ date, competition, blue: a.roster.map(playerCard), red: b.roster.map(playerCard) }} />
    {copyStatus ? <p role="status">{copyStatus}</p> : null}
  </section>;
}
