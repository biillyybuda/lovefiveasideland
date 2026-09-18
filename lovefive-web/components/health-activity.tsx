"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Activity, ArrowDown, ArrowUp, ArrowUpDown, ChevronRight, Footprints, Heart, Info, RefreshCw, X } from "lucide-react";
import { watchActivityFields, formatActivityValue, clockValue, type MatchActivity } from "@/lib/match-activity";
import { formatUkDate } from "@/lib/demo-data";
import { boardMetrics, healthLeaderboard, healthTrend, runningProfiles, trendMetrics, type BoardMetric, type HealthContext, type PlayerHealth, type ProfileMetric, type ProfileSort, type TrendMetric } from "@/lib/health-stats";

const format = (value: number | null | undefined, decimals = 0) => value == null ? "—" : value.toLocaleString("en-GB", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
const games = (n: number) => `${n} ${n === 1 ? "game" : "games"}`;
const readings = (n: number) => `${n} ${n === 1 ? "reading" : "readings"}`;
const activityHref = (health: HealthContext, matchId: number) => `${health.reportBasePath}/${matchId}#match-activity`;

function HealthStatus({ health, children }: { health: HealthContext; children: ReactNode }) {
  if (health.status === "loading") return <div className="panel health-state" role="status"><Activity size={24}/><h2>Loading activity</h2><p>Gathering the league's recorded readings…</p></div>;
  if (health.status === "error") return <div className="panel health-state" role="alert"><h2>Activity unavailable</h2><p>{health.message}</p><button className="button" onClick={health.reload}>Try loading activity again</button></div>;
  return <>{children}</>;
}

function HealthHeading({ health, title, subtitle, icon }: { health: HealthContext; title: string; subtitle: string; icon: ReactNode }) {
  return <div className="health-heading"><div className="health-title"><span className="health-icon" aria-hidden="true">{icon}</span><div><span className="health-kicker">{health.seasonLabel} · Recorded activity</span><h2>{title}</h2><p>{subtitle}</p></div></div>
    <button className="mini-link health-refresh" onClick={health.reload} aria-label="Refresh activity"><RefreshCw size={14}/>Refresh</button></div>;
}

function Snapshot({ label, value, detail, primary = false }: { label: string; value: string; detail: string; primary?: boolean }) {
  return <div className={`snapshot-card${primary ? " primary" : ""}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

const profileColumns: { key: ProfileMetric; label: string; unit?: string; help?: string }[] = [
  { key: "name", label: "Player" },
  { key: "metresPerMinute", label: "Distance/min", unit: "m/min", help: "Total distance divided by playing time, using sessions with both readings. Longer sessions carry more weight. Playing role and time in goal can affect this figure." },
  { key: "topSpeed", label: "Top speed", unit: "km/h", help: "The highest maximum speed recorded by the watch in this view. An individual GPS spike can affect the peak." },
  { key: "movingShare", label: "Time moving", unit: "%", help: "Moving time divided by elapsed time from the same watch sessions, weighted by session length. Sessions missing either reading are excluded." },
  { key: "change", label: "Change", help: "Change in distance per minute from the first to the latest paired reading in this view. One recording sets the baseline. If the first value is zero, change is shown in m/min instead of a percentage. Changes describe running output; playing role and time in goal provide context." }
];

function WorkRateChange({ player }: { player: PlayerHealth }) {
  const change = player.workRateChange;
  if (!change) return <span className="health-baseline">{player.metresPerMinute === null ? "—" : "Baseline set"}</span>;
  const amount = Number((change.percent ?? change.latest - change.baseline).toFixed(1));
  const direction = amount > 0 ? "up" : amount < 0 ? "down" : "steady";
  const unit = change.percent === null ? " m/min" : "%";
  return <div className="health-change">
    <strong className={`health-change-${direction}`} aria-label={amount === 0 ? "No change since first recording" : `${direction === "up" ? "Up" : "Down"} ${format(Math.abs(amount), 1)}${unit} since first recording`}>
      {amount === 0 ? "No change" : <><span aria-hidden="true">{direction === "up" ? "↑" : "↓"}</span> {format(Math.abs(amount), 1)}{unit}</>}
    </strong>
    <small>since {formatUkDate(player.firstWorkRate?.date)}</small>
    <small>Updated {formatUkDate(player.latestWorkRate?.date)}</small>
  </div>;
}

export function LeagueHealthActivity({ health, onPlayer }: { health: HealthContext; onPlayer: (id: number) => void }) {
  const [metric, setMetric] = useState<BoardMetric>("metresPerMinute");
  const [sort, setSort] = useState<ProfileSort>({ metric: "metresPerMinute", direction: "descending" });
  const [help, setHelp] = useState<ProfileMetric | "reference" | null>(null);
  const helpTrigger = useRef<HTMLButtonElement | null>(null);
  const [boardHelp, setBoardHelp] = useState(false);
  const board = useMemo(() => healthLeaderboard(health.data.players, metric, 1), [health.data.players, metric]);
  const profiles = useMemo(() => runningProfiles(health.data.players, sort), [health.data.players, sort]);
  const definition = boardMetrics.find(item => item.key === metric)!;
  const explanation = profileColumns.find(item => item.key === help);
  const latestMatch = health.data.coverage.at(-1)?.match;
  const topValue = board[0]?.rounded || 1;
  const nextDirection = (key: ProfileMetric) => sort.metric === key ? (sort.direction === "ascending" ? "descending" : "ascending") : key === "name" ? "ascending" : "descending";
  const toggleHelp = (key: ProfileMetric | "reference", trigger: HTMLButtonElement) => {
    helpTrigger.current = trigger;
    setHelp(current => current === key ? null : key);
  };
  const closeHelp = () => { setHelp(null); helpTrigger.current?.focus(); };

  return <HealthStatus health={health}><section className="health-area interactive-stack" aria-label="League health and activity">
    <HealthHeading health={health} title="Health & Activity" subtitle="Compare player running output and track changes over time." icon={<Activity/>}/>
    <section className="panel health-profiles" aria-labelledby="health-profiles-title" onKeyDown={event => { if (event.key === "Escape" && help) { event.stopPropagation(); closeHelp(); } }}>
      <div className="health-section-head health-profiles-head">
        <div><span className="health-kicker">Player comparisons</span><h3 id="health-profiles-title">Running profiles</h3><p className="health-section-intro">Select a player to explore their trends.</p></div>
        {latestMatch && <a className="mini-link" href={activityHref(health, latestMatch.id)}>{health.canEdit ? "Add or edit readings" : "View match readings"} →</a>}
      </div>
      <div id="health-profile-help" className={help ? "health-help-panel" : ""} hidden={!help} role="status">
        <div><strong>{help === "reference" ? "League reference" : explanation?.label}</strong><p>{help === "reference" ? "The median of each player's distance per minute and time moving in this view. Each player has equal weight; missing readings are excluded. These are reference values for comparing running output." : explanation?.help}</p></div>
        <button type="button" className="health-info-button" onClick={closeHelp} aria-label="Close metric explanation"><X size={16} aria-hidden="true"/></button>
      </div>
      {profiles.length ? <>
        <p className="health-mobile-hint">Swipe the table to compare all measures.</p>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable player running profiles"><table>
          <thead><tr>{profileColumns.map(column => <th key={column.key} scope="col" aria-sort={sort.metric === column.key ? sort.direction : "none"}>
            <div className="health-column-head">
              <button type="button" className="health-sort-button" onClick={() => setSort({ metric: column.key, direction: nextDirection(column.key) })} aria-label={`Sort by ${column.label}, ${nextDirection(column.key)}`}>
                <span>{column.label}{column.unit && <small>{column.unit}</small>}</span>
                {sort.metric !== column.key ? <ArrowUpDown size={13} aria-hidden="true"/> : sort.direction === "ascending" ? <ArrowUp size={13} aria-hidden="true"/> : <ArrowDown size={13} aria-hidden="true"/>}
              </button>
              {column.help && <button type="button" className="health-info-button" aria-label={`About ${column.label}`} aria-expanded={help === column.key} aria-controls="health-profile-help" onClick={event => toggleHelp(column.key, event.currentTarget)}><Info size={14} aria-hidden="true"/></button>}
            </div>
          </th>)}<th scope="col" aria-label="Open profile"/></tr></thead>
          <tbody>{profiles.map(player => <tr key={player.player.id} className="health-profile-row" onClick={() => onPlayer(player.player.id)}>
            <td><button type="button" className="health-player-link" aria-label={`Open ${player.name}'s running profile`} onClick={event => { event.stopPropagation(); onPlayer(player.player.id); }}>{player.name}{player.player.is_active === 0 && <small> · Archived</small>}</button><small>Latest · {formatUkDate(player.recorded.at(-1)?.match.date)}</small></td>
            <td className="health-work-rate">{format(player.metresPerMinute, 1)}</td><td>{format(player.topSpeed, 1)}</td><td>{player.movingShare === null ? "—" : `${format(player.movingShare, 0)}%`}</td>
            <td><WorkRateChange player={player}/></td><td className="health-profile-arrow"><ChevronRight size={17} aria-hidden="true"/></td>
          </tr>)}</tbody>
        </table></div>
        <div className="health-reference" aria-label="League reference values">
          <span className="health-reference-label">League reference <button type="button" className="health-info-button" aria-label="About league reference" aria-expanded={help === "reference"} aria-controls="health-profile-help" onClick={event => toggleHelp("reference", event.currentTarget)}><Info size={14} aria-hidden="true"/></button></span>
          <span>Median distance/min <strong>{format(health.data.typicalWorkRate, 1)}{health.data.typicalWorkRate !== null && " m/min"}</strong></span>
          <span>Median time moving <strong>{format(health.data.typicalMovingShare, 0)}{health.data.typicalMovingShare !== null && "%"}</strong></span>
        </div>
      </> : <div className="empty-state health-profile-empty"><Footprints aria-hidden="true"/><strong>Start a player's running profile</strong><p>Add distance and playing time from one Garmin recording to set their baseline.</p></div>}
    </section>

    <section className="panel health-board" aria-labelledby="activity-board-title">
      <div className="health-section-head health-board-head">
        <div className="health-board-title"><h3 id="activity-board-title">Activity leaderboards</h3><button type="button" className="health-info-button" aria-label="About activity leaderboards" aria-expanded={boardHelp} aria-controls="health-board-help" onClick={() => setBoardHelp(value => !value)}><Info size={16} aria-hidden="true"/></button></div>
        <div className="health-metric-toggle" role="group" aria-label="Leaderboard measure">{boardMetrics.map(item => <button type="button" key={item.key} aria-pressed={metric === item.key} onClick={() => setMetric(item.key)}>{item.key === "metresPerMinute" ? "Distance/min" : "Top speed"}</button>)}</div>
      </div>
      <div id="health-board-help" className={boardHelp ? "health-help-panel" : ""} hidden={!boardHelp} role="status"><p>{profileColumns.find(column => column.key === metric)?.help} Ties share a rank at the precision shown. One recording is enough to appear.</p></div>
      {board.length ? <ol className="health-ranking" aria-label={`${definition.label} leaderboard`}>{board.map(row => <li key={row.player.player.id}>
        <span className={`health-rank${row.rank === 1 ? " leading" : ""}`}>{row.rank}</span>
        <button type="button" className="health-player-link" onClick={() => onPlayer(row.player.player.id)}>{row.player.name}{row.player.player.is_active === 0 && <small> · Archived</small>}</button>
        <div className="health-bar" aria-hidden="true"><i style={{ width: `${Math.max(0, row.rounded / topValue * 100)}%` }}/></div>
        <div className="health-ranking-value"><strong>{format(row.rounded, definition.decimals)}</strong><small>{definition.unit}</small></div>
      </li>)}</ol> : <div className="empty-state"><strong>No {metric === "metresPerMinute" ? "distance-and-time" : "maximum-speed"} readings yet</strong></div>}
    </section>
    <p className="health-page-note">For team comparisons and personal progress. Automatic matchmaking does not use these readings yet.</p>
  </section></HealthStatus>;
}
export function PlayerHealthActivity({ health, playerId }: { health: HealthContext; playerId: number | null }) {
  const player = health.data.players.find(p => p.player.id === playerId);
  return <HealthStatus health={health}>{player ? <PlayerHealthContent key={playerId} health={health} player={player}/> : <div className="panel empty-state"><strong>Select a player</strong><p>Choose someone above to see their recorded activity and heart-rate history.</p></div>}</HealthStatus>;
}

function PlayerHealthContent({ health, player }: { health: HealthContext; player: PlayerHealth }) {
  const [metric, setMetric] = useState<TrendMetric>("metres_per_minute");
  const [limit, setLimit] = useState(12);
  const points = useMemo(() => healthTrend(player, metric, limit, true), [player, metric, limit]);
  const definition = trendMetrics.find(m => m.key === metric)!;
  const latestMatch = player.entries.at(-1)?.match;
  const latestWatch = [...player.recorded].reverse().find(e => watchActivityFields.some(f => e.reading![f.key] != null));
  const bestDistance = [...player.recorded].filter(e => e.reading!.distance_km !== null).sort((a, b) => b.reading!.distance_km! - a.reading!.distance_km! || b.match.date!.localeCompare(a.match.date!) || b.match.id - a.match.id)[0];
  return <section className="health-area interactive-stack" aria-label={`${player.name} health and activity`}>
    <HealthHeading health={health} title={`${player.name} · Health & Activity`} subtitle="Running output, movement and personal trends from sessions with the Garmin." icon={<Heart/>}/>
    <div className="stats-snapshot-grid health-snapshots">
      <Snapshot primary label="Distance per minute" value={player.metresPerMinute === null ? "—" : `${format(player.metresPerMinute, 1)} m/min`} detail={`${readings(player.pairedCount)} with distance and playing time`}/>
      <Snapshot label="Top recorded speed" value={player.topSpeed === null ? "—" : `${format(player.topSpeed, 1)} km/h`} detail="Highest maximum speed from the watch"/>
      <Snapshot label="Time moving (%)" value={player.movingShare === null ? "—" : `${format(player.movingShare, 0)}%`} detail="Moving ÷ elapsed time from paired readings"/>
    </div>
    {!player.recorded.length && <div className="panel health-welcome"><Activity aria-hidden="true"/><div><h3>{player.entries.length ? "Ready for the first reading" : "No appearances in this view"}</h3><p>{player.entries.length ? `There are no activity readings for ${player.name} in this season yet. Add any figures you have to a previous game.` : "Choose another season to explore this player's activity."}</p>{latestMatch && <a className="button" href={activityHref(health, latestMatch.id)}>{health.canEdit ? "Add match readings" : "View match activity"}</a>}</div></div>}
    <div className="health-personal-grid">
      <section className="panel health-trend" aria-labelledby="health-trend-title"><div className="health-section-head"><div><span className="health-kicker">Your watch sessions</span><h3 id="health-trend-title">Activity trends</h3></div></div>
        <div className="control-bar health-controls"><label><span>Reading</span><select value={metric} onChange={e => setMetric(e.target.value as TrendMetric)}>{trendMetrics.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}</select></label><label><span>Readings</span><select value={limit} onChange={e => setLimit(Number(e.target.value))}><option value={12}>Latest 12 readings</option><option value={0}>All in this view</option></select></label></div>
        <ActivityTrend points={points} unit={definition.unit} label={`${player.name}: ${definition.label}`} decimals={definition.decimals}/>
        <p className="health-caption">Each point is a session with this reading, in date order; the gaps between dates can vary. Sessions when someone else wore the watch are skipped. Compare similar playing roles and time in goal.</p>
      </section>
      <section className="panel health-heart" aria-labelledby="health-heart-title"><div className="health-section-head"><div><span className="health-kicker">Personal context</span><h3 id="health-heart-title">Heart-rate readings</h3></div><Heart size={20} aria-hidden="true"/></div>
        <div className="health-heart-grid"><div><span>Average match HR</span><strong>{format(player.averageHeartRate)} <small>bpm</small></strong><small>Mean of {player.heartRateCount} recorded match averages</small></div><div><span>Highest match peak</span><strong>{format(player.peakHeartRate)} <small>bpm</small></strong><small>Across {games(player.peakCount)} with peak readings</small></div>
          <div><span>Latest resting HR</span><strong>{format(player.latestResting?.value)} <small>bpm</small></strong>{player.latestResting ? <a href={activityHref(health, player.latestResting.matchId)}>{formatUkDate(player.latestResting.date)} · Match {player.latestResting.matchId}</a> : <small>Not recorded in this view</small>}</div>
          <div><span>Personal maximum HR</span><strong>{format(player.latestPersonalMax?.value)} <small>bpm</small></strong>{player.latestPersonalMax ? <a href={activityHref(health, player.latestPersonalMax.matchId)}>{formatUkDate(player.latestPersonalMax.date)} · Match {player.latestPersonalMax.matchId}</a> : <small>Not recorded in this view</small>}</div></div>
        <p className="health-caption">Baselines are the latest entered for this period, dated to their match. A match peak and a personal maximum are separate readings.</p>
        {bestDistance && <div className="health-highlight"><Footprints size={18} aria-hidden="true"/><div><span>Longest recorded outing</span><strong>{format(bestDistance.reading!.distance_km, 2)} km</strong><a href={activityHref(health, bestDistance.match.id)}>{formatUkDate(bestDistance.match.date)} · Match {bestDistance.match.id}</a></div></div>}
      </section>
    </div>
    {latestWatch && <section className="panel health-watch" aria-label="Latest detailed watch recording">
      <div className="health-section-head"><div><span className="health-kicker">{latestWatch.reading!.source || "Watch readings"}</span><h3>Latest watch detail</h3></div><a className="mini-link" href={activityHref(health, latestWatch.match.id)}>{formatUkDate(latestWatch.match.date)} · Match {latestWatch.match.id}</a></div>
      <WatchReadings row={latestWatch.reading!}/>
      <p className="health-caption">Values from this recording. Calories are a watch estimate and can depend on the wearer's profile. Pace and speed are shown as supplied. Use the trend selector above to explore any reading across games.</p>
    </section>}
    <details className="panel health-history"><summary>Recorded sessions · {player.recorded.length}</summary><p className="health-caption">“—” means not recorded. Distance per minute needs both distance and playing time. Open a match to {health.canEdit ? "add or correct readings" : "view its activity"}.</p>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Scrollable player activity readings"><table><thead><tr><th>Match</th><th>Distance<br/>km</th><th>Total time<br/>m:ss</th><th>Distance<br/>m/min</th><th>Average HR<br/>bpm</th><th>Peak HR<br/>bpm</th><th>Resting HR<br/>bpm</th><th>Personal max<br/>bpm</th><th>Watch details</th><th>Source</th></tr></thead><tbody>{[...player.recorded].reverse().map(entry => <tr key={entry.match.id}>
        <td><a className="health-table-link" href={activityHref(health, entry.match.id)}>{formatUkDate(entry.match.date)}<small>Match {entry.match.id}</small></a></td>
        <td>{format(entry.reading?.distance_km, 2)}</td><td>{formatActivityValue("duration_minutes", entry.reading?.duration_minutes)}</td><td>{format(entry.reading?.distance_km != null && entry.reading.duration_minutes != null ? entry.reading.distance_km * 1000 / entry.reading.duration_minutes : null, 1)}</td>
        <td>{format(entry.reading?.average_heart_rate)}</td><td>{format(entry.reading?.max_heart_rate)}</td><td>{format(entry.reading?.resting_heart_rate)}</td><td>{format(entry.reading?.personal_max_heart_rate)}</td><td>{entry.reading && watchActivityFields.some(f => entry.reading![f.key] != null) ? <details className="health-watch-disclosure"><summary>Show readings</summary><WatchReadings row={entry.reading}/></details> : "—"}</td><td className="health-source">{entry.reading?.source || "—"}</td>
      </tr>)}</tbody></table></div></details>
    <p className="health-caption">Readings are visible to your league. Team selection, MMR and awards do not use this data.</p>
  </section>;
}

function WatchReadings({ row }: { row: MatchActivity }) {
  return <dl className="health-watch-grid">{watchActivityFields.filter(f => row[f.key] != null).map(f => <div key={f.key}><dt>{f.label.replace("minutes or m:ss", "m:ss")}</dt><dd>{formatActivityValue(f.key, row[f.key])}</dd></div>)}</dl>;
}

export function ActivityTrend({ points, unit, label, decimals }: { points: ReturnType<typeof healthTrend>; unit: string; label: string; decimals: number }) {
  const show = (value: number) => unit === "m:ss/km" ? clockValue(value) : format(value, decimals);
  const [focused, setFocused] = useState<number | null>(null);
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const update = () => setWidth(Math.max(260, Math.round(element.getBoundingClientRect().width)));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const available = points.map((point, index) => ({ ...point, index })).filter((p): p is typeof p & { value: number } => p.value !== null);
  if (!available.length) return <div ref={container} className="empty-state health-chart-empty"><Activity aria-hidden="true"/><strong>No {unit === "bpm" ? "heart-rate readings" : "readings"} for this selection</strong><p>Choose another measure or add figures to a match.</p></div>;
  const lo = Math.min(...available.map(p => p.value)), hi = Math.max(...available.map(p => p.value));
  const padding = Math.max((hi - lo) * .15, unit === "bpm" ? 5 : hi * .08, unit === "km" ? .2 : 1);
  const min = Math.max(0, lo - padding), max = hi + padding;
  const left = width < 400 ? 44 : 60, right = width - 16;
  const x = (index: number) => points.length === 1 ? (left + right) / 2 : left + index / (points.length - 1) * (right - left);
  const y = (value: number) => 214 - (value - min) / (max - min) * 178;
  const segments: string[][] = [[]];
  points.forEach((point, index) => { if (point.value === null) { if (segments.at(-1)!.length) segments.push([]); } else segments.at(-1)!.push(`${x(index)},${y(point.value)}`); });
  const active = available.find(p => p.index === focused) || available.at(-1)!;
  return <div ref={container} className="health-chart">
    <div className="health-chart-readout" aria-live="polite"><strong>{show(active.value)} <small>{unit}</small></strong><span>{formatUkDate(active.match.date)} · Match {active.match.id}</span><small>{readings(available.length)} across {games(points.length)}</small></div>
    <svg viewBox={`0 0 ${width} 250`} role="group" aria-label={`${label}. ${readings(available.length)}. Full figures are in the match-by-match table.`}>
      {[min, (min + max) / 2, max].map((value, i) => <g key={i}><line className="health-gridline" x1={left} x2={right} y1={y(value)} y2={y(value)}/><text x={left - 9} y={y(value) + 4} textAnchor="end">{value >= 10000 ? value.toLocaleString("en-GB", { notation: "compact", maximumFractionDigits: 1 }) : show(value)}</text></g>)}
      {segments.filter(s => s.length > 1).map((segment, i) => <polyline key={i} points={segment.join(" ")} className="health-trend-line"/>)}
      {available.map(p => <circle key={p.match.id} cx={x(p.index)} cy={y(p.value)} r={p.index === active.index ? 6 : 4} className="health-trend-dot" tabIndex={0} role="img"
        aria-label={`${formatUkDate(p.match.date)}, match ${p.match.id}: ${show(p.value)} ${unit}`} onMouseEnter={() => setFocused(p.index)} onMouseLeave={() => setFocused(null)} onFocus={() => setFocused(p.index)} onBlur={() => setFocused(null)}><title>{`${formatUkDate(p.match.date)} · ${show(p.value)} ${unit}`}</title></circle>)}
      <text x={left} y="242">{formatUkDate(points[0]?.match.date)}</text>{points.length > 1 && <text x={right} y="242" textAnchor="end">{formatUkDate(points.at(-1)?.match.date)}</text>}
    </svg>
  </div>;
}
