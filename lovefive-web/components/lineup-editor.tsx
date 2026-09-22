"use client";

import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { ArrowLeftRight, LockKeyhole, RotateCcw, Sparkles, Unlock, X } from "lucide-react";
import { displayName, type Match, type Player } from "@/lib/demo-data";
import { buildMatchdayBalance, matchdayOptions } from "@/lib/matchday-balance";
import { predictBalance } from "@/lib/balance-model";
import { FORMATIONS, LINK_COLOURS, PITCH, ROLE_NAMES, SIDES, buildLineupChemistry, initialLineup, isFormation, lineupLinks, lineupText, playerLink, savedRole, shieldName, suggestTeam, swapLineupColours, swapLineupPlayers, validateDraft, type Chemistry, type FormationId, type LineupDraft, type LineupSeed, type Role, type Side, type TeamLineup } from "@/lib/lineup-editor";
import { SHIELD_PATH, SHIRT_PATH, TEAM_COLOURS, createFormationPoster } from "@/lib/lineup-poster";
import { competitionLabels, matchCompetition, type Competition } from "@/lib/competition";
import { LineupImageButton } from "@/components/lineup-image-button";

type Target = { side: Side; index: number };
const pct = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const rosterKey = (ids: number[]) => [...ids].sort((a, b) => a - b).join(",");
const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };

function PlayerShield({ player, role, side, locked }: { player: Player; role: Role; side: Side; locked: boolean }) {
  const theme = TEAM_COLOURS[side];
  const gradient = `shield-${side}-${player.id}`;
  const name = shieldName(displayName(player));
  const start = 18 + (48 - name.lines.length * name.size * 1.08) / 2 + name.size * 0.8;
  return <svg viewBox="-77 -95 154 190" aria-hidden="true">
    <defs><linearGradient id={gradient} x1="0" y1="0" x2="1" y2="1"><stop stopColor={theme.banner} /><stop offset=".47" stopColor="#132236" /><stop offset="1" stopColor="#07111e" /></linearGradient></defs>
    <path d={SHIELD_PATH} fill={`url(#${gradient})`} stroke="#edcf80" strokeWidth="2.5" />
    <path d={SHIRT_PATH} transform="translate(-24 -69) scale(2)" fill={theme.banner} stroke="#eaf3ff" strokeWidth="1" />
    <rect x="-23" y="-13" width="46" height="22" fill={theme.banner} />
    <g textAnchor="middle" fill="#fff" fontWeight="600">
      <text y="4" fontSize="18">{role}</text>
      {name.lines.map((line, i) => <text key={i} y={start + i * name.size * 1.08} fontSize={name.size}
        {...(line.length * name.size * 0.64 > 126 ? { textLength: 126, lengthAdjust: "spacingAndGlyphs" } : {})}>{line}</text>)}
      <text y="72" fontSize="29">{Math.round(player.mmr ?? 1000)}</text>
      <text y="86" fontSize="10" fill="#b8c9df" fontWeight="500">MMR</text>
    </g>
    {locked ? <g transform="translate(43 -59)"><rect x="-12" y="-12" width="24" height="25" rx="5" fill="#091523" /><path d="M-5 -2V-6a5 5 0 0 1 10 0v4M-7 -2h14v10H-7z" fill="none" stroke="#fff" strokeWidth="2" /></g> : null}
  </svg>;
}

function Pitch({ side, team, chemistry, players, locks, selected, dragging, onChoose, onPointer, onLink }: {
  side: Side; team: TeamLineup; chemistry: Chemistry; players: Map<number, Player>; locks: LineupDraft["locks"];
  selected: Target | null; dragging: boolean; onChoose: (target: Target, keyboard: boolean) => void;
  onPointer: (event: PointerEvent<HTMLButtonElement>, phase: "start" | "move" | "end" | "cancel", target: Target) => void;
  onLink: (a: number, b: number) => void;
}) {
  const theme = TEAM_COLOURS[side], formation = FORMATIONS[team.formation];
  return <section className={`formation-pitch formation-pitch-${side}`} aria-label={`${side} team formation`}>
    <svg className="formation-field" viewBox={`0 0 ${PITCH.width} ${PITCH.height}`} aria-label={`${side} player links`}>
      <rect width="600" height="1120" fill={theme.dark} />
      {Array.from({ length: 8 }, (_, i) => i % 2 === 0 ? <rect key={i} y={84 + i * 129} width="600" height="129" fill="rgba(7,15,25,.14)" /> : null)}
      <g stroke="rgba(188,215,241,.2)" strokeWidth="2" fill="none" aria-hidden="true"><rect x="22" y="108" width="556" height="978" /><path d="M22 590H578" /><ellipse cx="300" cy="590" rx="110" ry="90" /><path d="M156 108v116h288V108M156 1086V970h288v116" /></g>
      <rect width="600" height="82" fill={theme.banner} /><path d="M460 0H600V82H414Z" fill="rgba(255,255,255,.08)" />
      <text x="26" y="53" fill="#fff" fontSize="34" fontWeight="600">{side.toUpperCase()} TEAM</text>
      <text x="577" y="51" textAnchor="end" fill="#d2e4f8" fontSize="26">{formation.label}</text>
      {lineupLinks(team, chemistry).map(link => {
        const a = formation.slots[link.from], b = formation.slots[link.to];
        const label = `${displayName(players.get(link.a)!)} and ${displayName(players.get(link.b)!)} chemistry`;
        return <g key={`${link.a}-${link.b}`} role="button" tabIndex={0} aria-label={label} className="formation-link" onClick={() => onLink(link.a, link.b)}
          onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onLink(link.a, link.b); } }}>
          <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth="28" />
          <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#091523" strokeWidth="12" pointerEvents="none" />
          <line className="formation-link-colour" x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={LINK_COLOURS[link.colour]} strokeWidth="7" pointerEvents="none" />
        </g>;
      })}
    </svg>
    {team.ids.map((id, index) => {
      const slot = formation.slots[index], player = players.get(id)!;
      const chosen = selected?.side === side && selected.index === index;
      return <button key={id} type="button" data-lineup-slot={`${side}:${index}`} className={`formation-token${chosen ? " is-selected" : ""}${dragging ? " drag-active" : ""}`}
        style={{ left: `${slot.x / PITCH.width * 100}%`, top: `${slot.y / PITCH.height * 100}%` }} aria-pressed={chosen}
        aria-label={`${displayName(player)}, ${ROLE_NAMES[slot.role]}, ${side} team${locks[id] ? ", position locked" : ""}`}
        onClick={event => onChoose({ side, index }, event.detail === 0)} onPointerDown={e => onPointer(e, "start", { side, index })}
        onPointerMove={e => onPointer(e, "move", { side, index })} onPointerUp={e => onPointer(e, "end", { side, index })}
        onPointerCancel={e => onPointer(e, "cancel", { side, index })}>
        <PlayerShield player={player} role={slot.role} side={side} locked={!!locks[id]} />
      </button>;
    })}
  </section>;
}

export function LineupEditor({ players, modelPlayers, matches, selected, seed, storageKey, onSelectionChange, sideLocks }: {
  players: Player[]; modelPlayers: Player[]; matches: Match[]; selected: Player[]; seed: LineupSeed | null;
  storageKey: string; onSelectionChange: (ids: number[]) => void; sideLocks: { blue: number[]; red: number[] };
}) {
  const [draft, setDraft] = useState<LineupDraft | null>(null);
  const [date, setDate] = useState(seed?.date || today);
  const [ready, setReady] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [storageStatus, setStorageStatus] = useState("");
  const [chosen, setChosen] = useState<Target | null>(null);
  const [pair, setPair] = useState<[number, number] | null>(null);
  const [drag, setDrag] = useState<{ name: string; x: number; y: number } | null>(null);
  const pointer = useRef<{ id: number; startX: number; startY: number; source: Target; moved: boolean } | null>(null);
  const ignoreClick = useRef(false);
  const history = useRef<LineupDraft[]>([]);
  const initialized = useRef(false);
  const linkDetails = useRef<HTMLElement>(null);
  const playerSettings = useRef<HTMLElement>(null);
  const selectedKey = rosterKey(selected.map(p => p.id));
  const byId = useMemo(() => new Map(players.map(p => [p.id, p])), [players]);
  const calculation = useMemo(() => {
    try { return { model: buildMatchdayBalance(modelPlayers, matches, date), chemistry: buildLineupChemistry(modelPlayers, matches, date), error: "" }; }
    catch (cause) { return { model: null, chemistry: null, error: cause instanceof Error ? cause.message : "Could not load the match records." }; }
  }, [modelPlayers, matches, date]);

  // Drafts are scoped to the league, validated against its current players and restored only once.
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    try {
      const seedMatches = seed && rosterKey([...seed.blue, ...seed.red]) === selectedKey;
      if (seedMatches) {
        const chemistry = buildLineupChemistry(modelPlayers, matches, seed.date);
        setDraft(initialLineup(seed.date, seed.blue, seed.red, players, chemistry, seed.competition)); setDate(seed.date);
      } else {
        const raw = localStorage.getItem(storageKey);
        if (raw) {
          const saved = validateDraft(JSON.parse(raw), players);
          const savedIds = [...saved.blue.ids, ...saved.red.ids];
          if (!selected.length || rosterKey(savedIds) === selectedKey) {
            setDraft(saved); setDate(saved.date); onSelectionChange(savedIds); setMessage("Your saved lineup is ready.");
          }
        }
      }
    } catch { setMessage("Choose your ten players to start a fresh lineup."); }
    setReady(true);
  }, [matches, modelPlayers, onSelectionChange, players, seed, selected.length, selectedKey, storageKey]);

  useEffect(() => {
    if (!ready || selected.length !== 10 || !calculation.model || !calculation.chemistry) return;
    if (draft && rosterKey([...draft.blue.ids, ...draft.red.ids]) === selectedKey) return;
    try {
      const option = matchdayOptions(calculation.model, selected, sideLocks.blue, sideLocks.red)[0];
      if (!option) throw new Error("The side locks cannot make two teams of five. Clear them in Team Generator first.");
      setDraft(initialLineup(date, option.teamA.map(p => p.id), option.teamB.map(p => p.id), players, calculation.chemistry));
      history.current = []; setChosen(null); setPair(null); setError("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not prepare the lineup."); }
  }, [calculation, date, draft, players, ready, selected, selectedKey, sideLocks]);

  useEffect(() => {
    if (!ready || !draft) return;
    try { localStorage.setItem(storageKey, JSON.stringify(draft)); setStorageStatus("Draft saved on this device"); }
    catch { setStorageStatus("This browser cannot save the draft. Keep this tab open or download the image."); }
  }, [draft, ready, storageKey]);

  const usable = !!draft && selected.length === 10 && rosterKey([...draft.blue.ids, ...draft.red.ids]) === selectedKey;
  const forecast = useMemo(() => {
    if (!usable || !draft || !calculation.model) return null;
    try { return predictBalance(calculation.model.state, [...draft.blue.ids].sort((a, b) => a - b), [...draft.red.ids].sort((a, b) => a - b)); }
    catch { return null; }
  }, [calculation.model, draft, usable]);

  function commit(next: LineupDraft, status: string) {
    validateDraft(next, players);
    if (draft) history.current = [...history.current.slice(-29), draft];
    setDraft(next); setDate(next.date); setChosen(null); setPair(null); setError(""); setMessage(status);
  }
  function act(action: () => void) { try { action(); } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not change the lineup."); } }
  function swap(from: Target, to: Target) {
    if (!draft) return;
    act(() => {
      const nameA = displayName(byId.get(draft[from.side].ids[from.index])!), nameB = displayName(byId.get(draft[to.side].ids[to.index])!);
      commit(swapLineupPlayers(draft, from, to), `${nameA} and ${nameB} swapped.${from.side !== to.side ? " Forecast updated." : ""}`);
    });
  }
  function choose(target: Target, keyboard: boolean) {
    if (keyboard) ignoreClick.current = false;
    if (ignoreClick.current) { ignoreClick.current = false; return; }
    if (chosen && (chosen.side !== target.side || chosen.index !== target.index)) { swap(chosen, target); return; }
    setChosen(chosen ? null : target); setPair(null); setError("");
  }
  function handlePointer(event: PointerEvent<HTMLButtonElement>, phase: "start" | "move" | "end" | "cancel", source: Target) {
    if (!draft) return;
    if (phase === "start") {
      ignoreClick.current = false;
      if (event.button !== 0 || !event.isPrimary || draft.locks[draft[source.side].ids[source.index]]) return;
      pointer.current = { id: event.pointerId, startX: event.clientX, startY: event.clientY, source, moved: false };
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    const p = pointer.current;
    if (!p || p.id !== event.pointerId) return;
    if (phase === "move") {
      if (Math.hypot(event.clientX - p.startX, event.clientY - p.startY) > 8) p.moved = true;
      if (p.moved) setDrag({ name: displayName(byId.get(draft[p.source.side].ids[p.source.index])!), x: event.clientX, y: event.clientY });
      return;
    }
    pointer.current = null; setDrag(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (p.moved) {
      ignoreClick.current = true;
      if (phase === "end") {
        const hit = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>("[data-lineup-slot]")?.dataset.lineupSlot;
        if (hit) { const [side, index] = hit.split(":"); if (SIDES.includes(side as Side)) swap(p.source, { side: side as Side, index: Number(index) }); }
      }
    }
  }
  function suggest() {
    if (!draft || !calculation.chemistry) return;
    act(() => {
      const next = structuredClone(draft);
      for (const side of SIDES) next[side] = suggestTeam(next, side, players, calculation.chemistry!, next.autoFormations[side] ? undefined : next[side].formation);
      commit(next, "Positions suggested using your locks, preferred roles, player links and recorded goals and assists.");
    });
  }
  const selectedPlayer = chosen && draft ? byId.get(draft[chosen.side].ids[chosen.index]) : null;
  const selectedLink = pair && calculation.chemistry ? playerLink(calculation.chemistry, pair[0], pair[1]) : null;
  const activeError = calculation.error || error;
  useEffect(() => {
    if (pair) { linkDetails.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }); linkDetails.current?.focus({ preventScroll: true }); }
  }, [pair]);

  return <div className="interactive-stack lineup-editor" onKeyDown={event => { if (event.key === "Escape") { setChosen(null); setPair(null); } }}>
    <section className="panel lineup-editor-controls">
      <div className="panel-title-row"><div><span className="stat-label">Your matchday, your setup</span><h2>Lineup editor</h2></div>
        <label><span>Match date</span><input type="date" aria-label="Lineup match date" value={date} onChange={event => {
          const value = event.target.value; setDate(value); setChosen(null); setPair(null);
          if (draft) act(() => commit({ ...draft, date: value }, "Match date updated."));
        }} /></label>
      </div>
      <p className="muted">Drag a card onto another to swap, or tap one player then another. Tap a link for their shared record.</p>
      {usable && draft ? <>
        <label><span>Fixture competition</span><select value={matchCompetition(draft)} onChange={event => act(() => commit({ ...draft, competition: event.target.value as Competition }, "Fixture competition updated."))}><option value="league">League</option><option value="immunity_shield">MMR Immunity Shield</option></select></label>
        {draft.competition === "immunity_shield" && <p className="competition-note">MMR protected. A draw awards no Shield.</p>}
        <div className="lineup-actions">
          <button type="button" className="primary-button" onClick={suggest} disabled={!calculation.chemistry}><Sparkles size={16} />Suggest positions</button>
          <button type="button" className="ghost-button" onClick={() => act(() => commit(swapLineupColours(draft), "Team colours swapped. Forecast updated."))}><ArrowLeftRight size={16} />Swap team colours</button>
          <button type="button" className="ghost-button" disabled={!history.current.length} onClick={() => {
            const previous = history.current.pop(); if (previous) { setDraft(previous); setDate(previous.date); setChosen(null); setPair(null); setError(""); setMessage("Last change undone."); }
          }}><RotateCcw size={16} />Undo</button>
          <button type="button" className="ghost-button" onClick={async () => {
            try { await navigator.clipboard.writeText(lineupText(draft, players)); setMessage("Lineup copied."); } catch { setError("Copy is unavailable. Use Save lineup image instead."); }
          }}>Copy lineup</button>
          {forecast && calculation.chemistry && !calculation.error ? <LineupImageButton
            key={JSON.stringify({ draft, forecast: [forecast.win, forecast.draw, forecast.loss], players: draft.blue.ids.concat(draft.red.ids).map(id => [id, byId.get(id)?.mmr, displayName(byId.get(id)!)]) })}
            data={{ date: draft.date, blue: draft.blue.ids.map(id => displayName(byId.get(id)!)), red: draft.red.ids.map(id => displayName(byId.get(id)!)), forecast }}
            renderImage={() => createFormationPoster({ draft, players, chemistry: calculation.chemistry!, forecast })} /> : null}
        </div>
        <div className="lineup-team-settings">
          {SIDES.map(side => {
            const team = draft[side], keeper = team.ids[0];
            return <div key={side} className={`lineup-side-settings ${side}`}>
              <strong>{side === "blue" ? "Blue" : "Red"} team</strong>
              <label><span>Formation · includes keeper</span><select aria-label={`${side} formation`} value={draft.autoFormations[side] ? "auto" : team.formation} disabled={!calculation.chemistry}
                onChange={event => act(() => {
                  const next = structuredClone(draft), value = event.target.value;
                  next.autoFormations[side] = value === "auto";
                  next[side] = suggestTeam(next, side, players, calculation.chemistry!, isFormation(value) ? value : undefined);
                  commit(next, `${side === "blue" ? "Blue" : "Red"} formation updated.`);
                })}><option value="auto">Suggest a formation</option>{Object.entries(FORMATIONS).map(([id, f]) => <option value={id} key={id}>{f.label}</option>)}</select></label>
              <label><span>Goalkeeper</span><select aria-label={`${side} goalkeeper`} value={keeper} onChange={event => act(() => {
                const id = Number(event.target.value), next = structuredClone(draft);
                delete next.locks[keeper];
                const swapped = swapLineupPlayers(next, { side, index: 0 }, { side, index: team.ids.indexOf(id) });
                swapped.locks[id] = { side, role: "GK" };
                commit(swapped, `${displayName(byId.get(id)!)} locked in goal for ${side}.`);
              })}>{team.ids.map(id => <option key={id} value={id}>{displayName(byId.get(id)!)}</option>)}</select></label>
              <button type="button" className="ghost-button" aria-pressed={!!draft.locks[keeper]} aria-label={`${draft.locks[keeper] ? "Unlock" : "Lock"} ${side} goalkeeper`} onClick={() => {
                const next = structuredClone(draft); if (next.locks[keeper]) delete next.locks[keeper]; else next.locks[keeper] = { side, role: "GK" };
                commit(next, next.locks[keeper] ? "Goalkeeper locked." : "Goalkeeper unlocked.");
              }}>{draft.locks[keeper] ? <LockKeyhole size={15} /> : <Unlock size={15} />}{draft.locks[keeper] ? "Keeper locked" : "Lock keeper"}</button>
            </div>;
          })}
        </div>
        <small className="muted">Every formation covers all four positions. Position moves keep the same forecast; swaps between teams recalculate it. {storageStatus}.</small>
      </> : <p className="muted">Choose exactly ten players above to start, or open a matchup from Team Generator.</p>}
      <p className="lineup-status" role="status" aria-live="polite">{message}</p>
      {activeError ? <p className="lineup-error" role="alert">{activeError}</p> : null}
    </section>

    {usable && draft && calculation.chemistry ? <>
      <section className="formation-board" aria-label="Editable matchday lineup">
        <header className="formation-board-heading"><span>LOVE FIVE</span><h2>{competitionLabels[matchCompetition(draft)]} Lineups</h2><time dateTime={date}>{new Date(`${date}T12:00:00Z`).toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" })}</time></header>
        <div className="formation-pitches">{SIDES.map(side => <Pitch key={side} side={side} team={draft[side]} chemistry={calculation.chemistry!} players={byId} locks={draft.locks} selected={chosen} dragging={!!drag} onChoose={choose} onPointer={handlePointer} onLink={(a, b) => { setPair([a, b]); setChosen(null); }} />)}</div>
        <footer className="formation-forecast"><h3>MATCH FORECAST</h3>{forecast ? <div><section className="blue"><span>BLUE WIN</span><strong>{pct(forecast.win)}</strong></section><section><span>DRAW</span><strong>{pct(forecast.draw)}</strong></section><section className="red"><span>RED WIN</span><strong>{pct(forecast.loss)}</strong></section></div> : <p>Forecast unavailable. Check the match date and records above.</p>}</footer>
      </section>
      {selectedPlayer && chosen ? <section ref={playerSettings} tabIndex={-1} className="panel lineup-player-settings" aria-label="Selected player settings">
        <div className="panel-title-row"><h3>{displayName(selectedPlayer)}</h3><button className="ghost-button" aria-label="Clear player selection" onClick={() => setChosen(null)}><X size={18} /></button></div>
        <p className="muted">Select another card to swap, or adjust this player below.</p>
        <div className="lineup-actions"><button type="button" className="ghost-button" aria-pressed={!!draft.locks[selectedPlayer.id]} onClick={() => {
          const next = structuredClone(draft); if (next.locks[selectedPlayer.id]) delete next.locks[selectedPlayer.id]; else next.locks[selectedPlayer.id] = { side: chosen.side, role: FORMATIONS[draft[chosen.side].formation].slots[chosen.index].role };
          commit(next, next.locks[selectedPlayer.id] ? `${displayName(selectedPlayer)} locked in position.` : `${displayName(selectedPlayer)} unlocked.`);
        }}>{draft.locks[selectedPlayer.id] ? <Unlock size={16} /> : <LockKeyhole size={16} />}{draft.locks[selectedPlayer.id] ? "Unlock position" : "Lock position"}</button>
        <label><span>Preferred position</span><select aria-label={`${displayName(selectedPlayer)} preferred position`} value={draft.preferences[selectedPlayer.id] || savedRole(selectedPlayer.position) || ""} onChange={event => {
          const next = structuredClone(draft); if (event.target.value) next.preferences[selectedPlayer.id] = event.target.value as Role; else delete next.preferences[selectedPlayer.id];
          commit(next, "Preference saved. Use Suggest positions to apply it.");
        }}><option value="">Use recorded profile</option>{Object.entries(ROLE_NAMES).map(([role, name]) => <option value={role} key={role}>{name}</option>)}</select></label></div>
        {selectedPlayer.style_profile?.historicMatches ? <p className="muted">{selectedPlayer.style_profile.historicGoals} goals · {selectedPlayer.style_profile.historicAssists} assists · {selectedPlayer.style_profile.historicMatches} tracked matches. Recorded stats help suggest roles; they do not establish defensive or keeper ability.</p> : <p className="muted">No recorded goals or assists yet. Set a preferred position to guide suggestions.</p>}
      </section> : null}
      {selectedLink ? <section ref={linkDetails} tabIndex={-1} className="panel lineup-link-details" aria-label="Player link details">
        <div className="panel-title-row"><h3>{displayName(byId.get(selectedLink.a)!)} + {displayName(byId.get(selectedLink.b)!)}</h3><button className="ghost-button" aria-label="Close link details" onClick={() => setPair(null)}><X size={18} /></button></div>
        <div className="lineup-link-stats"><div><span>Wins together</span><strong>{pct(selectedLink.winRate)}</strong><small>{selectedLink.wins} wins in {selectedLink.games} shared games</small></div><div><span>Individual average</span><strong>{pct(selectedLink.averageRate)}</strong><small>Mean of their individual win percentages</small></div>
          {[selectedLink.a, selectedLink.b].map(id => { const record = calculation.chemistry!.players.get(id); return <div key={id}><span>{displayName(byId.get(id)!)}</span><strong>{pct(record?.games ? record.wins / record.games : null)}</strong><small>{record?.games || 0} games</small></div>; })}</div>
        <p className="muted">{selectedLink.games ? `${selectedLink.difference! > 0 ? "+" : ""}${(selectedLink.difference! * 100).toFixed(1)} percentage points against their average. Green is more than 5 points above, amber within 5, red more than 5 below. Draws count as games, not wins.` : "No shared games yet. The link is red for zero recorded chemistry; there is no shared win percentage to calculate."}</p>
      </section> : null}
      <p className="muted lineup-suggestion-note">Set your keepers first, then use Suggest positions to arrange the outfield players. You can adjust any suggestion.</p>
    </> : null}
    {usable && selectedPlayer ? <div className="lineup-selection-bar" role="region" aria-label="Player selected for swap">
      <div><strong>{displayName(selectedPlayer)}</strong><span>Tap another card to swap</span></div>
      <button type="button" className="ghost-button" onClick={() => { playerSettings.current?.scrollIntoView({ block: "center", behavior: "smooth" }); playerSettings.current?.focus({ preventScroll: true }); }}>Player options</button>
      <button type="button" className="ghost-button" aria-label="Cancel player selection" onClick={() => setChosen(null)}><X size={18} /></button>
    </div> : null}
    {drag ? <div className="lineup-drag-preview" style={{ left: drag.x + 12, top: drag.y + 12 }}>{drag.name}</div> : null}
  </div>;
}
