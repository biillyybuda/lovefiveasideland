import { getCompetitionLabels, type Competition } from "./competition";
import { layoutLineupName } from "./lineup-image";

export type PlayerCardData = {
  name: string;
  mmr: number;
  games: number;
  wins: number;
  draws: number;
  losses: number;
  form: { result: "W" | "D" | "L"; shield: boolean; date: string | null }[];
};
export type PlayerCardsImageData = { cupName?: string; date: string; competition: Competition; blue: PlayerCardData[]; red: PlayerCardData[] };
export function cupFormLegend(name?: string) { return `Gold outline = ${getCompetitionLabels(name).immunity_shield} game`; }
const FONT = '"Lineup Inter", Arial, sans-serif';

/** A portrait export uses the same card data as the page, at a fixed readable size. */
export function drawPlayerCardsImage(canvas: HTMLCanvasElement, data: PlayerCardsImageData) {
  if (data.blue.length !== 5 || data.red.length !== 5) throw new Error("Choose two teams of five first.");
  const date = new Date(`${data.date}T12:00:00Z`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(data.date) || !Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== data.date) throw new Error("Choose a valid match date first.");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Image saving is unavailable in this browser.");
  const names = [data.blue, data.red].map(team => team.map(player => layoutLineupName(player.name, (value, size) => {
    ctx.font = `600 ${size}px ${FONT}`;
    // Wrap conservatively in a 472px column before drawing the names at 38px.
    return ctx.measureText(value).width * 418 / 472;
  })));
  const rowHeights = data.blue.map((_, i) => 224 + (Math.max(names[0][i].lines.length, names[1][i].lines.length) - 1) * 42);
  const bottom = 264 + rowHeights.reduce((sum, height) => sum + height, 0) + 4 * 14;
  canvas.width = 1200;
  canvas.height = bottom + 100;

  function box(x: number, y: number, width: number, height: number, fill: string, radius = 0, stroke?: string) {
    ctx!.beginPath(); ctx!.roundRect(x, y, width, height, radius);
    ctx!.fillStyle = fill; ctx!.fill();
    if (stroke) { ctx!.strokeStyle = stroke; ctx!.lineWidth = 1; ctx!.stroke(); }
  }
  function text(value: string, x: number, y: number, size: number, colour = "#f3f7ff", weight = 500, maxWidth?: number) {
    ctx!.font = `${weight} ${size}px ${FONT}`;
    while (maxWidth && ctx!.measureText(value).width > maxWidth && size > 7) { size--; ctx!.font = `${weight} ${size}px ${FONT}`; }
    ctx!.fillStyle = colour; ctx!.fillText(value, x, y);
  }
  const muted = "#aebfd2", gold = "#edcf80";
  box(0, 0, canvas.width, canvas.height, "#0b1320");
  const header = ctx.createLinearGradient(0, 0, 1200, 240);
  header.addColorStop(0, "#152b45"); header.addColorStop(1, "#241b2e");
  ctx.fillStyle = header; ctx.fillRect(0, 0, 1200, 184);
  box(40, 24, 560, 4, "#66adff"); box(600, 24, 560, 4, "#ff7f96");
  text("LOVE FIVE", 48, 65, 23, muted, 600);
  ctx.textAlign = "right"; text(getCompetitionLabels(data.cupName)[data.competition].toUpperCase(), 1152, 65, 21, muted, 600, 850); ctx.textAlign = "left";
  text("Matchday player cards", 48, 119, 51, "#ffffff", 600);
  text(date.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" }), 48, 158, 25, muted);

  [data.blue, data.red].forEach((team, side) => {
    const x = side === 0 ? 40 : 612, accent = side === 0 ? "#66adff" : "#ff7f96";
    box(x, 200, 548, 50, side === 0 ? "#17385c" : "#462536", 10);
    text(side === 0 ? "BLUE TEAM" : "RED TEAM", x + 24, 234, 27, accent, 600);
    let y = 264;
    team.forEach((player, i) => {
      const height = rowHeights[i], wrapped = names[side][i];
      box(x, y, 548, height, side === 0 ? "#132236" : "#242133", 12, side === 0 ? "#29405a" : "#493340");
      box(x + 1, y + 18, 4, height - 36, accent, 2);
      wrapped.lines.forEach((line, j) => text(line, x + 26, y + 48 + j * 42, 38, "#ffffff", 600, 472));
      const offset = (wrapped.lines.length - 1) * 42;
      text("MMR", x + 26, y + 82 + offset, 17, muted);
      text(String(Math.round(player.mmr)), x + 26, y + 116 + offset, 29, "#e3edf8", 600, 492);
      text(`${player.games} ${player.games === 1 ? "game" : "games"}  ·  ${player.wins}–${player.draws}–${player.losses} W–D–L`, x + 26, y + 150 + offset, 21, muted, 500, 492);
      text("FORM", x + 26, y + height - 30, 17, muted);
      Array.from({ length: 5 }, (_, index) => {
        const entry = player.form[index], result = entry?.result || "–";
        const bx = x + 116 + index * 54, by = y + height - 57;
        if (entry?.shield) { ctx!.beginPath(); ctx!.roundRect(bx - 4, by - 4, 48, 48, 12); ctx!.strokeStyle = gold; ctx!.lineWidth = 3; ctx!.stroke(); }
        box(bx, by, 40, 40, result === "W" ? "#65dc93" : result === "L" ? "#ff8091" : result === "D" ? "#f4d35e" : "#334052", 8);
        ctx!.textAlign = "center"; text(result, bx + 20, by + 28, 24, entry ? "#101923" : "#aebfd2", 600); ctx!.textAlign = "left";
      });
      y += height + 14;
    });
  });
  box(42, bottom + 33, 26, 26, "#172337", 6, gold);
  ctx.strokeStyle = gold; ctx.lineWidth = 2; ctx.beginPath();
  ctx.moveTo(49, bottom + 39); ctx.lineTo(55, bottom + 37); ctx.lineTo(61, bottom + 39); ctx.lineTo(61, bottom + 46); ctx.quadraticCurveTo(59, bottom + 51, 55, bottom + 54); ctx.quadraticCurveTo(51, bottom + 51, 49, bottom + 46); ctx.closePath(); ctx.stroke();
  text(cupFormLegend(data.cupName), 82, bottom + 54, 23, muted, 500, 1076);
}

export async function createPlayerCardsImage(data: PlayerCardsImageData) {
  try { await Promise.all([document.fonts.load(`500 25px ${FONT}`), document.fonts.load(`600 38px ${FONT}`)]); }
  catch (cause) { console.warn("Player card font unavailable; using Arial", cause); }
  const canvas = document.createElement("canvas");
  drawPlayerCardsImage(canvas, data);
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(value => value ? resolve(value) : reject(new Error("Could not create the player cards image.")), "image/png"));
  return { blob, width: canvas.width, height: canvas.height, filename: `love-five-player-cards-${data.date}.png` };
}
