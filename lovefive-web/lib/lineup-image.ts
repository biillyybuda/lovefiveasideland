import { competitionLabels, type Competition } from "./competition";

export type LineupImageData = {
  competition?: Competition;
  date: string;
  blue: readonly string[];
  red: readonly string[];
  forecast: { win: number; draw: number; loss: number };
};

const WIDTH = 1080;
const TEAM_WIDTH = 482;
const NAME_WIDTH = TEAM_WIDTH - 64;
// Matches the two locally hosted faces used by .matchday-broadcast.
const FONT = '"Lineup Inter", Arial, sans-serif';
type FontWeight = 500 | 600;

/** Wrap full names without dropping characters; unusually long names can make the image taller. */
export function layoutLineupName(name: string, measure: (text: string, size: number) => number) {
  const words = name.trim().split(/\s+/);
  const graphemes = new Intl.Segmenter("en", { granularity: "grapheme" });
  for (let size = 52; size >= 40; size -= 2) {
    const lines: string[] = [];
    let line = "";
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (measure(candidate, size) <= NAME_WIDTH) { line = candidate; continue; }
      if (line) { lines.push(line); line = ""; }
      for (const { segment } of graphemes.segment(word)) {
        if (line && measure(line + segment, size) > NAME_WIDTH) { lines.push(line); line = ""; }
        line += segment;
      }
    }
    if (line) lines.push(line);
    if (lines.length <= 2 || size === 40) return { size, lines: lines.length ? lines : ["Player"] };
  }
  throw new Error("Could not lay out player name.");
}

/** Draw a dedicated portrait graphic, independent of the website's viewport and CSS. */
export function drawLineupImage(canvas: HTMLCanvasElement, data: LineupImageData) {
  if (data.blue.length !== 5 || data.red.length !== 5) throw new Error("Choose two teams of five first.");
  const date = new Date(`${data.date}T12:00:00Z`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(data.date) || !Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== data.date) {
    throw new Error("Choose a valid match date first.");
  }
  if (Object.values(data.forecast).some(value => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("The match forecast is unavailable.");
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Image saving is unavailable in this browser.");
  const measure = (text: string, size: number) => {
    ctx.font = `500 ${size}px ${FONT}`;
    return ctx.measureText(text).width;
  };
  const teams = [data.blue, data.red].map(names => names.map(name => layoutLineupName(name.toUpperCase(), measure)));
  const rowHeight = Math.ceil(Math.max(142, ...teams.flat().map(name => name.lines.length * name.size * 1.18 + 32)));
  const teamsBottom = 380 + 5 * rowHeight;
  canvas.width = WIDTH;
  canvas.height = teamsBottom + 260;

  function rectangle(x: number, y: number, width: number, height: number, colour: string) {
    ctx!.fillStyle = colour;
    ctx!.fillRect(x, y, width, height);
  }
  function text(value: string, x: number, y: number, size: number, colour = "#f4f7fc", weight: FontWeight = 500, tracking = 0) {
    ctx!.font = `${weight} ${size}px ${FONT}`;
    ctx!.fillStyle = colour;
    if (!tracking) { ctx!.fillText(value, x, y); return; }
    const letters = Array.from(value);
    const widths = letters.map(letter => ctx!.measureText(letter).width);
    const width = widths.reduce((sum, part) => sum + part, 0) + tracking * Math.max(0, letters.length - 1);
    let cursor = ctx!.textAlign === "center" ? x - width / 2 : x;
    ctx!.save();
    ctx!.textAlign = "left";
    letters.forEach((letter, index) => { ctx!.fillText(letter, cursor, y); cursor += widths[index] + tracking; });
    ctx!.restore();
  }
  function fitted(value: string, x: number, y: number, size: number, maxWidth: number, colour?: string, weight: FontWeight = 500) {
    while (size > 20) {
      ctx!.font = `${weight} ${size}px ${FONT}`;
      if (ctx!.measureText(value).width <= maxWidth) break;
      size -= 1;
    }
    text(value, x, y, size, colour, weight);
  }
  function shirt(x: number, y: number) {
    ctx!.save();
    ctx!.translate(x, y);
    ctx!.scale(2, 2);
    ctx!.beginPath();
    ctx!.moveTo(8, 1); ctx!.quadraticCurveTo(12, 7, 16, 1);
    ctx!.lineTo(20, 3); ctx!.lineTo(24, 8); ctx!.lineTo(20, 11);
    ctx!.lineTo(18, 8); ctx!.lineTo(18, 24); ctx!.lineTo(6, 24);
    ctx!.lineTo(6, 8); ctx!.lineTo(4, 11); ctx!.lineTo(0, 8);
    ctx!.lineTo(4, 3); ctx!.closePath();
    ctx!.fillStyle = "rgba(255,255,255,0.1)"; ctx!.fill();
    ctx!.strokeStyle = "#fff"; ctx!.lineWidth = 1.5; ctx!.lineJoin = "round"; ctx!.stroke();
    ctx!.restore();
  }

  rectangle(0, 0, WIDTH, canvas.height, "#101b2b");
  ctx.save();
  ctx.beginPath(); ctx.roundRect(28, 28, 1024, canvas.height - 56, 18); ctx.clip();
  const header = ctx.createLinearGradient(28, 28, 1052, 280);
  header.addColorStop(0, "#192f50"); header.addColorStop(0.64, "#121d30"); header.addColorStop(1, "#2b2033");
  ctx.fillStyle = header; ctx.fillRect(28, 28, 1024, 252);
  rectangle(28, 28, 512, 8, "#438fff"); rectangle(540, 28, 512, 8, "#ff5672");
  text("LOVE FIVE", 54, 83, 26, "#b7c7dd", 500, 4);
  fitted(data.competition ? competitionLabels[data.competition].toUpperCase() : "MATCHDAY LINEUPS", 54, 158, 68, 966, "#fff", 600);
  const dateLabel = date.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
  fitted(dateLabel.toUpperCase(), 54, 228, 40, 966, "#e0e8f4");

  teams.forEach((names, side) => {
    const x = side === 0 ? 48 : 550;
    const accent = side === 0 ? "#6db2ff" : "#ff8397";
    rectangle(x, 282, TEAM_WIDTH, 98, side === 0 ? "#1c55b1" : "#aa2544");
    ctx.beginPath(); ctx.moveTo(x + 360, 282); ctx.lineTo(x + TEAM_WIDTH, 282);
    ctx.lineTo(x + TEAM_WIDTH, 380); ctx.lineTo(x + 300, 380); ctx.closePath();
    ctx.fillStyle = "rgba(255,255,255,0.07)"; ctx.fill();
    shirt(x + 26, 306);
    text(side === 0 ? "BLUE TEAM" : "RED TEAM", x + 100, 348, 46, "#fff", 600);
    names.forEach((name, index) => {
      const y = 380 + index * rowHeight;
      rectangle(x, y, TEAM_WIDTH, rowHeight, side === 0 ? (index % 2 ? "#1b2b44" : "#16253d") : (index % 2 ? "#272638" : "#222133"));
      rectangle(x, y, 5, rowHeight, accent);
      rectangle(x + 5, y + rowHeight - 1, TEAM_WIDTH - 5, 1, "#304056");
      const lineHeight = name.size * 1.18;
      const firstBaseline = y + (rowHeight - name.lines.length * lineHeight) / 2 + name.size * 0.96;
      name.lines.forEach((line, lineIndex) => text(line, x + 32, firstBaseline + lineIndex * lineHeight, name.size));
    });
  });

  const forecastY = teamsBottom + 24;
  rectangle(28, forecastY, 1024, 208, "#0c1523");
  rectangle(28, forecastY, 1024, 2, "#304056");
  text("MATCH FORECAST", 54, forecastY + 46, 26, "#d5deed", 500, 2.6);
  const outcomes = [
    { label: "BLUE WIN", value: data.forecast.win, colour: "#6db2ff" },
    { label: "DRAW", value: data.forecast.draw, colour: "#f4f7fc" },
    { label: "RED WIN", value: data.forecast.loss, colour: "#ff8397" }
  ];
  ctx.textAlign = "center";
  outcomes.forEach((outcome, index) => {
    const x = 28 + (index + 0.5) * (1024 / 3);
    if (index > 0) rectangle(28 + index * (1024 / 3), forecastY + 72, 1, 105, "#304056");
    text(outcome.label, x, forecastY + 98, 26, "#bdc9da", 500, 1);
    text(`${(outcome.value * 100).toFixed(1)}%`, x, forecastY + 172, 62, outcome.colour, 600);
  });
  ctx.restore();
}

export async function createLineupImage(data: LineupImageData) {
  // Wait before measuring as well as drawing, so the PNG cannot capture a fallback font.
  // A missing optional font must not stop the user from saving their teams.
  try {
    await Promise.all([document.fonts.load(`500 52px ${FONT}`), document.fonts.load(`600 68px ${FONT}`)]);
  } catch (cause) {
    console.warn("Lineup font unavailable; using the shared Arial fallback", cause);
  }
  const canvas = document.createElement("canvas");
  drawLineupImage(canvas, data);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(result => result ? resolve(result) : reject(new Error("Could not create the lineup image.")), "image/png");
  });
  return { blob, width: canvas.width, height: canvas.height, filename: `love-five-lineups-${data.date}.png` };
}
