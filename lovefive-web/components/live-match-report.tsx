"use client";

import Link from "next/link";
import { notFound, useParams, useSearchParams } from "next/navigation";
import { LiveLeagueLoader } from "@/components/live-league-loader";
import { buildMatchReport, type MatchReport } from "@/lib/match-report";

export function LiveMatchReport() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const id = params?.id;
  const shareMode = searchParams?.get("share") === "1";

  return (
    <LiveLeagueLoader active="matches">
      {({ players, matches, mmrHistory }) => {
        const match = matches.find((item) => String(item.id) === String(id));
        if (!match) notFound();

        const report = buildMatchReport({ match, matches, players, mmrHistory });
        const [scoreA, scoreB] = report.scoreLabel.split("-");
        const teamAPlayerNotes = report.playerNotes.filter((note) => note.team === "Team A");
        const teamBPlayerNotes = report.playerNotes.filter((note) => note.team === "Team B");
        const sharePlayerNotes = shareHighlights(report.playerNotes);

        if (shareMode) {
          return (
            <section className="match-share-page">
              <MatchShareCard
                report={report}
                scoreA={scoreA || "-"}
                scoreB={scoreB || "-"}
                playerNotes={sharePlayerNotes}
              />
              <Link className="mini-link match-share-full-link" href={`/app/matches/${id}`}>Full report</Link>
            </section>
          );
        }

        return (
          <>
            <section className="page-head match-report-page-head">
              <div className="match-report-title-block">
                <div className="match-report-link-row">
                  <Link className="mini-link" href="/app/charts?view=matches#match-history">Back to match history</Link>
                  <Link className="mini-link" href={`/app/matches/${id}?share=1`}>Share screenshot view</Link>
                </div>
                <h1>{report.headline}</h1>
                <div className="match-report-meta">
                  <span>{report.dateLabel}</span>
                  <span>{report.resultLabel}</span>
                </div>
              </div>

              <div className="weekly-report-score">
                <div>
                  <span>Team A</span>
                  <strong>{scoreA || "-"}</strong>
                  <small>{Math.round(report.rating.avgA)} avg MMR</small>
                </div>
                <em>FT</em>
                <div>
                  <span>Team B</span>
                  <strong>{scoreB || "-"}</strong>
                  <small>{Math.round(report.rating.avgB)} avg MMR</small>
                </div>
              </div>

              <div className="weekly-report-facts">
                {report.facts.map((fact) => (
                  <div key={fact.label}>
                    <span>{fact.label}</span>
                    <strong>{fact.value}</strong>
                    <small>{fact.detail}</small>
                  </div>
                ))}
              </div>
            </section>

            <section className="weekly-report-layout">
              <main className="weekly-report-main">
                <section className="panel weekly-story-panel">
                  <div className="section-subhead">
                    <strong>Storylines</strong>
                    <span>The shape of the match</span>
                  </div>
                  <div className="weekly-story-list">
                    {report.storylines.map((line) => (
                      <p key={line}>{line}</p>
                    ))}
                  </div>
                </section>

                {report.changes.length ? (
                  <section className="panel weekly-change-panel">
                    <div className="section-subhead">
                      <strong>What Changed</strong>
                      <span>Season movement from this result</span>
                    </div>
                    <div className="weekly-change-grid">
                      {report.changes.map((item) => (
                        <div key={`${item.label}-${item.value}`}>
                          <span>{item.label}</span>
                          <strong>{item.value}</strong>
                          <small>{item.detail}</small>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}

              </main>

              <aside className="weekly-report-side">
                <section className="panel report-lineups-panel">
                  <div className="section-subhead">
                    <strong>Lineups</strong>
                    <span>Recorded teams</span>
                  </div>
                  <div className="report-lineup-grid">
                    <div className="report-lineup-card a">
                      <span>Team A</span>
                      <div className="pill-row">
                        {report.teamA.map((player) => (
                          <b className="pill" key={`team-a-${player}`}>{player}</b>
                        ))}
                      </div>
                    </div>
                    <div className="report-lineup-card b">
                      <span>Team B</span>
                      <div className="pill-row">
                        {report.teamB.map((player) => (
                          <b className="pill" key={`team-b-${player}`}>{player}</b>
                        ))}
                      </div>
                    </div>
                  </div>
                  {report.playerNotes.length ? (
                    <details className="report-player-details">
                      <summary>
                        <span>Player details</span>
                        <strong>{report.playerNotes.length} notes</strong>
                      </summary>
                      <div className="report-player-note-groups">
                        {[
                          { label: "Team A", notes: teamAPlayerNotes, tone: "a" },
                          { label: "Team B", notes: teamBPlayerNotes, tone: "b" }
                        ].map((group) => (
                          <div className={`report-player-note-group ${group.tone}`} key={group.label}>
                            <span>{group.label}</span>
                            <div>
                              {group.notes.map((note) => (
                                <article className="report-player-note-row" key={`${note.team}-${note.name}`}>
                                  <div>
                                    <div className="report-player-note-title">
                                      <strong>{note.name}</strong>
                                      <b>{note.tag}</b>
                                    </div>
                                    <small>{note.detail}</small>
                                  </div>
                                  <em className={note.delta && note.delta > 0 ? "up" : note.delta && note.delta < 0 ? "down" : ""}>
                                    {note.delta === null ? "0" : `${note.delta > 0 ? "+" : ""}${note.delta}`}
                                  </em>
                                </article>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </details>
                  ) : null}
                </section>

                {report.previousMeeting ? (
                  <section className="panel previous-meeting-panel">
                    <div className="section-subhead">
                      <strong>Previous Meeting</strong>
                    </div>
                    <div className="previous-meeting-topline">
                      <div>
                        <span>Date</span>
                        <strong>{report.previousMeeting.dateLabel}</strong>
                      </div>
                      <div>
                        <span>Score</span>
                        <strong>{report.previousMeeting.scoreLabel}</strong>
                      </div>
                    </div>
                    <div className="previous-report-sides">
                      {report.previousMeeting.sides.map((side) => (
                        <div className={`similar-side-card ${side.tone}`} key={side.label}>
                          <div className="previous-side-head">
                            <strong>{side.label}</strong>
                            <span>{side.keptCount}/{side.totalCount} kept</span>
                          </div>
                          <div className="mini-change-block">
                            <div className="history-change-row previous">
                              <small>Old</small>
                              <div className="pill-row">
                                {side.historicTeam.map((player) => (
                                  <span
                                    className={`pill ${side.missing.includes(player) ? "out" : "kept"}`}
                                    key={`${side.label}-old-${player}`}
                                  >
                                    {player}
                                  </span>
                                ))}
                              </div>
                            </div>
                            <div className="history-change-row subs">
                              <small>Subs</small>
                              <div className="pill-row">
                                {side.subs.length ? side.subs.map((player) => (
                                  <span className="pill in" key={`${side.label}-sub-${player}`}>{player}</span>
                                )) : <small>Same five</small>}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}
              </aside>
            </section>
          </>
        );
      }}
    </LiveLeagueLoader>
  );
}

function MatchShareCard({
  report,
  scoreA,
  scoreB,
  playerNotes
}: {
  report: MatchReport;
  scoreA: string;
  scoreB: string;
  playerNotes: MatchReport["playerNotes"];
}) {
  return (
    <article className="match-share-card">
      <header className="match-share-head">
        <div>
          <span>Love 5 A Side</span>
          <h1>{report.headline}</h1>
          <small>{report.dateLabel} | {report.resultLabel}</small>
        </div>
        <div className="match-share-score">
          <strong>{scoreA}</strong>
          <em>FT</em>
          <strong>{scoreB}</strong>
        </div>
      </header>

      <div className="match-share-facts">
        {report.facts.map((fact) => (
          <div key={fact.label}>
            <span>{fact.label}</span>
            <strong>{fact.value}</strong>
            <small>{fact.detail}</small>
          </div>
        ))}
      </div>

      <section className="match-share-section">
        <strong>Storylines</strong>
        <div className="match-share-lines">
          {report.storylines.slice(0, 2).map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      </section>

      <section className="match-share-two-col">
        <div className="match-share-section">
          <strong>What Changed</strong>
          <div className="match-share-mini-list">
            {report.changes.slice(0, 4).map((item) => (
              <div key={`${item.label}-${item.value}`}>
                <span>{item.label}</span>
                <b>{item.value}</b>
              </div>
            ))}
          </div>
        </div>

        <div className="match-share-section">
          <strong>Player Angles</strong>
          <div className="match-share-mini-list">
            {playerNotes.map((note) => (
              <div key={`${note.team}-${note.name}-${note.tag}`}>
                <span>{note.tag}</span>
                <b>{note.name} {note.delta === null ? "" : `${note.delta > 0 ? "+" : ""}${note.delta}`}</b>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="match-share-lineups">
        <div>
          <span>Team A</span>
          <strong>{report.teamA.join(", ")}</strong>
        </div>
        <div>
          <span>Team B</span>
          <strong>{report.teamB.join(", ")}</strong>
        </div>
      </section>
    </article>
  );
}

function shareHighlights(notes: MatchReport["playerNotes"]) {
  const rank = new Map([
    ["MVP leader", 120],
    ["Most improved", 118],
    ["Bounce", 104],
    ["Streak", 100],
    ["Run stopped", 96],
    ["Pressure", 94],
    ["MVP chase", 92],
    ["Improver chase", 90],
    ["Rivalry", 88],
    ["Partnership", 84],
    ["Big lift", 72],
    ["Heavy hit", 70],
    ["Lift", 58],
    ["Hit", 54]
  ]);
  const usedTags = new Set<string>();
  const rows = [...notes]
    .sort((a, b) => (rank.get(b.tag) || 0) - (rank.get(a.tag) || 0) || Math.abs(b.delta || 0) - Math.abs(a.delta || 0))
    .filter((note) => {
      if (usedTags.has(note.tag) && usedTags.size < 4) return false;
      usedTags.add(note.tag);
      return true;
    });
  return rows.slice(0, 5);
}
