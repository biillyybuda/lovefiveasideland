"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { LiveLeagueLoader } from "@/components/live-league-loader";
import { buildMatchReport } from "@/lib/match-report";

export function LiveMatchReport() {
  const params = useParams<{ id: string }>();
  const id = params?.id;

  return (
    <LiveLeagueLoader active="matches">
      {({ players, matches, mmrHistory }) => {
        const match = matches.find((item) => String(item.id) === String(id));
        if (!match) notFound();

        const report = buildMatchReport({ match, matches, players, mmrHistory });
        const [scoreA, scoreB] = report.scoreLabel.split("-");

        return (
          <>
            <section className="page-head match-report-page-head">
              <div className="match-report-title-block">
                <Link className="mini-link" href="/app/charts?view=matches#match-history">Back to match history</Link>
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
                      <span>Consequences from this result</span>
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
