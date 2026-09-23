"use client";

import { Shield } from "lucide-react";
import { formatUkDate } from "@/lib/demo-data";
import { createPlayerCardsImage, SHIELD_FORM_LEGEND, type PlayerCardsImageData } from "@/lib/player-cards-image";
import { LineupImageButton } from "./lineup-image-button";

export function MatchdayPlayerCards({ data }: { data: PlayerCardsImageData }) {
  return <section className="player-card-section player-cards-refined" aria-label="Matchday player cards">
    <div className="player-cards-heading">
      <h3>Matchday player cards</h3>
      <LineupImageButton key={JSON.stringify(data)} renderImage={() => createPlayerCardsImage(data)}
        buttonLabel="Save player cards image" previewAlt={`Player cards for ${formatUkDate(data.date)}. Blue: ${data.blue.map(p => p.name).join(", ")}. Red: ${data.red.map(p => p.name).join(", ")}. ${SHIELD_FORM_LEGEND}.`} />
    </div>
    <p className="form-cup-legend"><span className="form-cup-key" aria-hidden="true"><Shield size={12} /></span>{SHIELD_FORM_LEGEND}</p>
    <div className="player-card-columns">{(["blue", "red"] as const).map(side => <div className={`player-card-team player-cards-${side}`} key={side}>
      <h4 className="player-cards-team-heading"><span aria-hidden="true" />{side === "blue" ? "Blue team" : "Red team"}</h4>
      {data[side].map((player, index) => <article className="matchday-player-card" key={index}>
        <h4 className="player-card-name">{player.name}</h4>
        <dl className="player-card-ratings">
          <div><dt>MMR</dt><dd>{Math.round(player.mmr)}</dd></div>
        </dl>
        <p className="player-card-record">{player.games} {player.games === 1 ? "game" : "games"}<span aria-hidden="true">·</span>{player.wins}–{player.draws}–{player.losses} W–D–L</p>
        <div className="player-card-form"><span>Form</span><div className="form-dots compact" role="group" aria-label={`${player.name} recent form, latest first`}>
          {Array.from({ length: 5 }, (_, i) => {
            const entry = player.form.at(i), result = entry?.result || "N";
            const outcome = result === "W" ? "Win" : result === "L" ? "Loss" : "Draw";
            const label = entry ? `${entry.shield ? "Immunity Shield Game" : "League"} · ${outcome} · ${formatUkDate(entry.date)}` : "No game";
            return <i className={`${result}${entry?.shield ? " form-cup" : ""}`} key={i} role="img" aria-label={label} title={label}>{result === "N" ? "–" : result}</i>;
          })}
        </div></div>
      </article>)}
    </div>)}</div>
  </section>;
}
