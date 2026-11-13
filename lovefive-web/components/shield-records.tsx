"use client";

import { useCompetitionLabels } from "./cup-name-provider";
import type { PlayerSummary } from "@/lib/demo-data";

export function ShieldRecords({ players }: { players: PlayerSummary[] }) {
  const labels = useCompetitionLabels();
  const rows = players.filter(p => p.matches_played).sort((a, b) => b.shieldWins - a.shieldWins || (b.wins || 0) - (a.wins || 0) || a.label.localeCompare(b.label));
  return <section className="panel shield-records">
    <div className="eyebrow">{labels.immunity_shield}</div><h2>Cup honours & records</h2>
    <p className="muted">Every win earns each winning player one cup win. A draw awards no trophy. League MMR and awards are protected; both competitions help balance teams.</p>
    {rows.length ? <div className="table-wrap"><table><thead><tr><th>Player</th><th>Cups won</th><th>Played</th><th>W–D–L</th><th>Goal diff</th><th>Recorded goals</th><th>Recorded assists</th></tr></thead>
      <tbody>{rows.map(p => <tr key={p.id}><td>{p.label}{p.is_active === 0 ? " · Archived" : ""}</td><td>{p.shieldWins}</td><td>{p.matches_played}</td><td>{p.wins}–{p.draws}–{p.losses}</td><td>{p.goalDiff}</td><td>{p.style_source?.stats.some(row => row.goals != null) ? p.style_profile?.historicGoals : "—"}</td><td>{p.style_source?.stats.some(row => row.assists != null) ? p.style_profile?.historicAssists : "—"}</td></tr>)}</tbody></table></div>
      : <p>No cup matches in this period yet.</p>}
  </section>;
}
