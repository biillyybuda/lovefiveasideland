// Fictional players, two full seasons and intentionally different starting ratings.
// Includes uneven attendance, draws, an archived participant and a new player.
function awardFixture() {
  const players = Array.from({length: 16}, (_, i) => ({id: i + 1, name: `Player ${i + 1}`, display_name: null,
    mmr: 840 + i * 23, initial_mmr: 840 + i * 23, is_active: i === 14 ? 0 : 1, fitness: 'Medium',
    matches_played: 0, wins: 0, draws: 0, losses: 0, win_streak: 0, lose_streak: 0}));
  const matches = [];
  let id = 1;
  for (const year of [2025, 2026]) for (let month = 1; month <= 12; month++) for (let week = 0; week < 2; week++) {
    const pool = Array.from({length: 15}, (_, i) => players[(i + month * 3 + week * 4 + (year === 2026 ? 2 : 0)) % 15]);
    const selected = pool.slice(0, 10);
    const teamA = selected.filter((_, i) => i % 2 === 0).map(p => p.name).join(',');
    const teamB = selected.filter((_, i) => i % 2 === 1).map(p => p.name).join(',');
    matches.push({id: id++, date: `${year}-${String(month).padStart(2, '0')}-${week ? '21' : '07'}`,
      team_a: teamA, team_b: teamB, score: `${(month + week + year) % 9}-${(month * 2 + week * 4) % 8}`, processed: 1});
  }
  return {players, matches, history: [], adjustments: [], token: 'fixture'};
}
module.exports = { awardFixture };
