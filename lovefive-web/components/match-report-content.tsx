import Link from "next/link";
import type { ReactNode } from "react";
import type { MatchReport } from "@/lib/match-report";
import { signed } from "@/lib/season-awards";

export function MatchReportContent({ report: r, basePath = "/app", share = false, noteEditor, activitySection, activityEditable = false }: {
  report: MatchReport; basePath?: string; share?: boolean; noteEditor?: ReactNode; activitySection?: ReactNode; activityEditable?: boolean;
}) {
  const [a, b] = r.scoreLabel.split("–");
  const highlighted = new Set(r.highlights.map(s => s.key));
  const supporting = r.stories.filter(s => !highlighted.has(s.key));
  const score = r.scoreContext;
  const prediction = r.rating.prediction;
  const record = (value: { wins: number; draws: number; losses: number }) => `${value.wins}W ${value.draws}D ${value.losses}L`;
  const lineups = <div className="review-lineups">{(["A", "B"] as const).map(side => <div className={`review-lineup review-team-${side.toLowerCase()}`} key={side}>
    <span className="review-eyebrow">Team {side}{(side === "A" ? r.teamALabel : r.teamBLabel) !== `Team ${side}` ? ` · ${side === "A" ? r.teamALabel : r.teamBLabel}` : ""}</span>
    <div className="review-chips">{(side === "A" ? r.teamA : r.teamB).map(name => <span key={name}>{name}</span>)}</div>
  </div>)}</div>;

  return <article className={`review-page${share ? " review-share" : ""}`}>
    <section className="panel review-hero">
      <nav className="review-nav" aria-label="Match review links">
        <Link className="mini-link" href={share ? `${basePath}/matches/${r.match.id}` : basePath === "/app" ? "/app/charts?view=matches#match-history" : "/demo/matches"}>{share ? "Full report" : "Back to match history"}</Link>
        {!share && <Link className="mini-link" href={`${basePath}/matches/${r.match.id}?share=1`}>Share screenshot view</Link>}
        {!share && activitySection && <a className="mini-link" href="#match-activity">{activityEditable ? "Add / edit player activity" : "Player activity"}</a>}
      </nav>
      <div className="review-kicker"><span>{r.season} · Match {r.matchNumber}</span><span>{r.dateLabel}</span><span>Full time</span></div>
      <h1>{r.headline}</h1>
      <div className="review-scoreboard">
        <div><span>{r.teamALabel}</span><strong>{a}</strong><small>Team A</small></div>
        <span className="review-score-separator">–</span>
        <div><span>{r.teamBLabel}</span><strong>{b}</strong><small>Team B</small></div>
      </div>
      <div className="review-highlights">{r.highlights.map(story => <div key={story.key}>
        <span className="review-eyebrow">{story.label}</span><strong>{story.title}</strong><p>{story.detail}</p>
      </div>)}</div>
      {!share && <p className="review-summary">{r.summary}</p>}
      {share && lineups}
    </section>

    {share ? <>
      {r.changes.length > 0 && <section className="panel"><h2>{r.season} · What changed</h2><div className="review-share-changes">{r.changes.slice(0, 2).map(c => <p key={c.name + c.label}><strong>{c.name}</strong> · {c.label}<br/>{c.before} → <b>{c.after}</b></p>)}</div></section>}
      {r.note && <section className="panel review-human-note"><span className="review-eyebrow">From the match · League admin note</span><p>{r.note}</p></section>}
      <p className="review-footnote">LoveFive · {r.season} season through this match · First appearances and career milestones refer to recorded league history.</p>
    </> : <div className="review-layout">
      <div className="review-main">
        <section className="panel"><div className="review-section-heading"><h2>What changed in {r.season}</h2><p>Immediately before → after this result</p></div>
          {r.changes.length ? <div className="review-change-list">{r.changes.slice(0, 5).map(c => <div className="review-change" key={c.name + c.label}>
            <div><span className="review-eyebrow">{c.label}</span><strong>{c.name}</strong></div>
            <div className="review-movement"><span>{c.before}</span><b aria-label="to">→</b><strong>{c.after}</strong></div><p>{c.detail}</p>
          </div>)}</div> : <p className="muted">No qualifying award movement from this result.</p>}
          <details className="review-standings"><summary>Season leaders after this match</summary><div>{r.standings.map(s => <p key={s.label}><span className="review-eyebrow">{s.label}</span><strong>{s.name} · {s.value}</strong><small>{s.detail}</small></p>)}</div>
            <p className="review-footnote">Awards use the same rules and eligible roster as Season Review. Matchday estimates do not change award points.</p>
          </details>
        </section>

        {r.groups.map(g => <section className="panel review-group" key={g.name}>
          <span className="review-eyebrow">The {g.name} reunion · {r.season}</span><h2>{g.teamLabel}</h2><p>{g.members.join(", ")} together{g.guests.length ? `, joined by ${g.guests.join(" and ")}` : ""}.</p>
          <div className="review-group-stats"><div><span>{g.minimumTogether}+ together</span><strong>{record(g.seasonRecord)}</strong><small>{g.seasonRecord.played} matches · was {record(g.beforeRecord)}</small></div>
            <div><span>These {g.coreSize} together</span><strong>{record(g.currentCoreRecord)}</strong><small>{g.currentCoreRecord.played} matches · includes any guest{g.coreSize < 5 ? " or the fifth HLHS player" : ""}</small></div>
            {g.coreSize < 5 && <div><span>All five HLHS players</span><strong>{g.allFiveRecord.played ? record(g.allFiveRecord) : "No games yet"}</strong><small>{g.allFiveRecord.played} matches in {r.season}</small></div>}
          </div>
          {g.streak >= 3 && <p>{g.streak} consecutive {g.outcome === "W" ? "wins" : g.outcome === "D" ? "draws" : "defeats"} in qualifying {g.name} games this season.</p>}
          <p className="review-footnote">{g.description} Reunion records count matches with at least {g.minimumTogether} of the five on the same side. All figures stop at this match.</p>
        </section>)}

        {supporting.length > 0 && <section className="panel"><h2>More from this result</h2><div className="review-supporting">{supporting.slice(0, 4).map(s => <div key={s.key}><span className="review-eyebrow">{s.scope} · {s.label}</span><h3>{s.title}</h3><p>{s.detail}</p></div>)}</div></section>}

        {(r.note || noteEditor) && <section className="panel review-human-note"><h2>From the match</h2><span className="review-eyebrow">League admin note</span>
          {r.note ? <p>{r.note}</p> : <p className="muted">Add the moments the score cannot tell us.</p>}{noteEditor}</section>}

        <section className="panel"><h2>The score in context</h2><div className="review-context-grid"><div><span>Total goals</span><strong>{score.total}</strong>
          <p>{score.priorMatches >= 5 ? `${score.tied ? "Joint " : ""}#${score.rank} for total goals in ${r.season} through this match.` : "More season results are needed for a useful ranking."}</p></div>
          <div><span>Earlier {r.season} average</span><strong>{score.averageBefore === null ? "—" : score.averageBefore.toFixed(1)}</strong><p>{score.priorMatches} earlier matches; this result is excluded.</p></div></div>
          {prediction && <details className="review-estimate"><summary>Expected versus actual</summary><p>Reconstructed pre-match estimate from the Matchday model: {Math.abs(prediction.meanMargin) < 0.05 ? "an even game" : `Team ${prediction.meanMargin > 0 ? "A" : "B"} ahead by ${Math.abs(prediction.meanMargin).toFixed(1)} goals`}. Actual result: {r.scoreLabel}.</p>
            <p className="review-footnote">Replayed using only earlier results. This was not a forecast saved before the game. Team A win {Math.round(prediction.win * 100)}% · draw {Math.round(prediction.draw * 100)}% · Team B win {Math.round(prediction.loss * 100)}%.</p></details>}
        </section>
      </div>

      <aside className="review-side">
        <section className="panel"><h2>Lineups</h2>{lineups}<p className="review-footnote">Recorded pre-match public MMR: A {r.rating.avgA === null ? "unavailable" : Math.round(r.rating.avgA)} · B {r.rating.avgB === null ? "unavailable" : Math.round(r.rating.avgB)}</p></section>
        {activitySection}
        <section className="panel"><h2>Player details · {r.season}</h2><p className="review-footnote">Season records and last five appearances. Public-rating changes describe the result, not an individual performance award.</p>
          <div className="review-players">{r.playerNotes.map(p => <details key={p.name}><summary><span><b>{p.name}</b><small>Team {p.side} · {p.appearances} {p.appearances === 1 ? "app" : "apps"} · {p.record}</small></span><strong>{p.delta === null ? "—" : `${signed(p.delta)} MMR`}</strong></summary>
            <p>{p.note || `${p.appearances} appearances in ${r.season}.`}</p><p>Last five: {p.form}</p><p>Public MMR: {p.before ?? "unavailable"} → {p.after ?? "unavailable"}</p></details>)}</div>
        </section>
        {r.relationships.length > 0 && <section className="panel"><h2>Connections · {r.season}</h2>{r.relationships.map(pair => <div className="review-connection" key={pair.title}><h3>{pair.title}</h3><p>{pair.detail}</p></div>)}</section>}
        {r.previousMeeting && <section className="panel"><h2>Closest previous lineups</h2><span className="review-eyebrow">{r.season} · Similar teams</span>
          <p><Link href={`${basePath}/matches/${r.previousMeeting.id}`}>{r.previousMeeting.date}</Link> · {r.previousMeeting.score}</p><p>{r.previousMeeting.shared} shared players; {r.previousMeeting.sameSide} kept together on equivalent sides.</p>
          {r.previousMeeting.sides.map(s => <details key={s.label}><summary>{s.label} · players changed</summary><p>Stayed together: {s.kept.join(", ") || "None"}</p><p>Joined this side: {s.joined.join(", ") || "None"}</p><p>Left this side: {s.left.join(", ") || "None"}</p></details>)}
        </section>}
        <p className="review-footnote">Season statistics stop at this result, including games on the same date. First appearances and career milestones cover the recorded archive; they may not be a player’s first game outside these records.</p>
      </aside>
    </div>}
  </article>;
}
