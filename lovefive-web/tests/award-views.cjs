function awardViews(awards, data) {
  const players = data.players.filter(p => p.is_active === 1);
  const matches = data.matches.filter(m => m.processed === 1);
  const years = [...new Set(matches.map(m => m.date.slice(0,4)))].sort();
  const results = {};
  for (const year of ['all', ...years]) for (const quarter of year === 'all' ? ['full'] : ['full','ytd','q1','q2','q3','q4']) {
    const period = awards.getPeriod(year, quarter, matches);
    const games = matches.filter(m => awards.inPeriod(m.date, period.start, period.end));
    const summaries = awards.playerSummariesForPeriod(players, games, data.history, period.start, period.end, year === 'all' || quarter !== 'full' ? 'all' : year);
    const previous = awards.previousComparisonPeriod(year, quarter, matches, period);
    const priorGames = previous ? matches.filter(m => awards.inPeriod(m.date, previous.start, previous.end)) : [];
    const priorPlayers = previous ? awards.playerSummariesForPeriod(players, priorGames, data.history, previous.start, previous.end, quarter === 'full' ? previous.season : 'all') : [];
    const active = summaries.filter(p => p.matches_played > 0);
    results[`${year}/${quarter}`] = {
      mvp: awards.mvpRace(summaries, games).map(({key,value}) => ({key,value})),
      improved: compactImproved(awards.improvedRace(active, games, priorPlayers, priorGames, previous, quarter))
    };
  }
  return results;
}
function compactImproved({title,subtitle,rows}) {
  return {title,subtitle,rows:rows.map(({key,value,unit})=>({key,value,unit}))};
}
module.exports = { awardViews };
