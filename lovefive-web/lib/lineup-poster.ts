import { displayName, type Player } from "./demo-data";
import { competitionLabels, matchCompetition } from "./competition";
import { FORMATIONS, LINK_COLOURS, PITCH, SIDES, lineupLinks, shieldName, validateDraft, type Chemistry, type LineupDraft, type Side } from "./lineup-editor";

export const SHIELD_PATH = "M0 -94 L16 -84 L69 -65 L69 67 L0 94 L-69 67 L-69 -65 L-16 -84 Z";
export const SHIRT_PATH = "M8 2 Q12 8 16 2 L20 4 L24 9 L20 12 L18 9 L18 25 L6 25 L6 9 L4 12 L0 9 L4 4 Z";
export const TEAM_COLOURS = { blue: { accent: "#54a6ff", banner: "#145bbd", dark: "#102745" }, red: { accent: "#ff5475", banner: "#ad2145", dark: "#301c30" } };
export type FormationPosterData = { draft: LineupDraft; players: Player[]; chemistry: Chemistry; forecast: { win: number; draw: number; loss: number } };
const FONT = '"Lineup Inter", Arial, sans-serif';

export function drawFormationPoster(canvas: HTMLCanvasElement, data: FormationPosterData) {
  validateDraft(data.draft, data.players);
  const probabilities = [data.forecast.win, data.forecast.draw, data.forecast.loss];
  if (probabilities.some(value => !Number.isFinite(value) || value < 0 || value > 1) || Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) > 0.001) throw new Error("The match forecast is unavailable.");
  canvas.width = 1200; canvas.height = 1500;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Image saving is unavailable in this browser.");
  const byId = new Map(data.players.map(p => [p.id, p]));
  function rect(x: number, y: number, w: number, h: number, fill: string) { ctx!.fillStyle = fill; ctx!.fillRect(x, y, w, h); }
  function text(value: string, x: number, y: number, size: number, colour = "#f3f7ff", weight = 500, maxWidth?: number) {
    ctx!.font = `${weight} ${size}px ${FONT}`;
    while (maxWidth && ctx!.measureText(value).width > maxWidth && size > 7) { size--; ctx!.font = `${weight} ${size}px ${FONT}`; }
    ctx!.fillStyle = colour; ctx!.fillText(value, x, y);
  }
  function line(x1: number, y1: number, x2: number, y2: number, colour: string, width = 2) {
    ctx!.strokeStyle = colour; ctx!.lineWidth = width; ctx!.beginPath(); ctx!.moveTo(x1, y1); ctx!.lineTo(x2, y2); ctx!.stroke();
  }
  rect(0, 0, 1200, 1500, "#081323");
  const backdrop = ctx.createLinearGradient(0, 0, 1200, 1400);
  backdrop.addColorStop(0, "#163454"); backdrop.addColorStop(0.5, "#091523"); backdrop.addColorStop(1, "#251b30");
  ctx.fillStyle = backdrop; ctx.fillRect(20, 20, 1160, 1460);
  rect(32, 28, 568, 4, TEAM_COLOURS.blue.accent); rect(600, 28, 568, 4, TEAM_COLOURS.red.accent);
  text("LOVE FIVE", 42, 70, 23, "#aebfd6");
  text(competitionLabels[matchCompetition(data.draft)].toUpperCase(), 40, 132, 57, "#fff", 600, 1120);
  const date = new Date(`${data.draft.date}T12:00:00Z`).toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }).toUpperCase();
  text(date, 42, 176, 29, "#c6d5e8", 500, 1116);

  function pitch(side: Side, x: number) {
    const team = data.draft[side], formation = FORMATIONS[team.formation], theme = TEAM_COLOURS[side];
    ctx!.save(); ctx!.translate(x, 210); ctx!.scale(552 / PITCH.width, 552 / PITCH.width);
    rect(0, 0, 600, 1120, theme.dark);
    for (let i = 0; i < 8; i++) if (i % 2 === 0) rect(0, 84 + i * 129, 600, 129, "rgba(7,15,25,0.14)");
    ctx!.strokeStyle = theme.accent; ctx!.lineWidth = 2; ctx!.strokeRect(1, 1, 598, 1118);
    ctx!.strokeStyle = "rgba(188,215,241,0.20)"; ctx!.lineWidth = 2;
    ctx!.strokeRect(22, 108, 556, 978); line(22, 590, 578, 590, "rgba(188,215,241,0.20)");
    ctx!.beginPath(); ctx!.ellipse(300, 590, 110, 90, 0, 0, Math.PI * 2); ctx!.stroke();
    ctx!.strokeRect(156, 108, 288, 116); ctx!.strokeRect(156, 970, 288, 116);
    rect(0, 0, 600, 82, theme.banner);
    ctx!.fillStyle = "rgba(255,255,255,0.08)"; ctx!.beginPath(); ctx!.moveTo(460, 0); ctx!.lineTo(600, 0); ctx!.lineTo(600, 82); ctx!.lineTo(414, 82); ctx!.fill();
    text(`${side.toUpperCase()} TEAM`, 26, 53, 34, "#fff", 600);
    ctx!.textAlign = "right"; text(formation.label, 577, 51, 26, "#d2e4f8"); ctx!.textAlign = "left";
    for (const link of lineupLinks(team, data.chemistry)) {
      const a = formation.slots[link.from], b = formation.slots[link.to];
      line(a.x, a.y, b.x, b.y, "#091523", 12);
      line(a.x, a.y, b.x, b.y, LINK_COLOURS[link.colour], 7);
    }
    team.ids.forEach((id, index) => {
      const player = byId.get(id)!, slot = formation.slots[index];
      ctx!.save(); ctx!.translate(slot.x, slot.y);
      const path = new Path2D(SHIELD_PATH);
      const fill = ctx!.createLinearGradient(-65, -90, 60, 90);
      fill.addColorStop(0, theme.banner); fill.addColorStop(0.47, "#132236"); fill.addColorStop(1, "#07111e");
      ctx!.shadowColor = theme.accent; ctx!.shadowBlur = 12; ctx!.fillStyle = fill; ctx!.fill(path); ctx!.shadowBlur = 0;
      ctx!.strokeStyle = "#edcf80"; ctx!.lineWidth = 2.5; ctx!.stroke(path);
      ctx!.save(); ctx!.translate(-24, -69); ctx!.scale(2, 2); ctx!.fillStyle = theme.banner; ctx!.strokeStyle = "#eaf3ff"; ctx!.lineWidth = 1; const shirt = new Path2D(SHIRT_PATH); ctx!.fill(shirt); ctx!.stroke(shirt); ctx!.restore();
      rect(-23, -13, 46, 22, theme.banner);
      ctx!.textAlign = "center"; text(slot.role, 0, 4, 18, "#fff", 600);
      const name = shieldName(displayName(player));
      const start = 18 + (48 - name.lines.length * name.size * 1.08) / 2 + name.size * 0.8;
      name.lines.forEach((value, i) => text(value, 0, start + i * name.size * 1.08, name.size, "#fff", 600, 126));
      text(String(Math.round(player.mmr ?? 1000)), 0, 72, 29, "#fff", 600);
      text("MMR", 0, 86, 10, "#b8c9df"); ctx!.restore();
    });
    ctx!.restore();
  }
  pitch("blue", 32); pitch("red", 616);
  rect(32, 1262, 1136, 206, "#0b1728"); line(32, 1262, 1168, 1262, "#38516e");
  text("MATCH FORECAST", 54, 1302, 25, "#c5d5e8");
  const results = [{ label: "BLUE WIN", value: data.forecast.win, colour: TEAM_COLOURS.blue.accent }, { label: "DRAW", value: data.forecast.draw, colour: "#fff" }, { label: "RED WIN", value: data.forecast.loss, colour: TEAM_COLOURS.red.accent }];
  ctx.textAlign = "center";
  results.forEach((result, index) => {
    const x = 32 + (index + 0.5) * 1136 / 3;
    if (index) line(32 + index * 1136 / 3, 1320, 32 + index * 1136 / 3, 1448, "#32455e");
    text(result.label, x, 1347, 25, "#c5d5e8"); text(`${(result.value * 100).toFixed(1)}%`, x, 1426, 66, result.colour, 600);
  });
}

export async function createFormationPoster(data: FormationPosterData) {
  try { await Promise.all([document.fonts.load(`500 28px ${FONT}`), document.fonts.load(`600 57px ${FONT}`)]); }
  catch (cause) { console.warn("Lineup font unavailable; using Arial", cause); }
  const canvas = document.createElement("canvas");
  drawFormationPoster(canvas, data);
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(value => value ? resolve(value) : reject(new Error("Could not create lineup image.")), "image/png"));
  return { blob, width: canvas.width, height: canvas.height, filename: `love-five-chemistry-${data.draft.date}.png` };
}
