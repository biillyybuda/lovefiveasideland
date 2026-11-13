require('./register-typescript.cjs');
const test = require('node:test'), assert = require('node:assert/strict');
const { buildLeagueRatings, applyLeagueChange } = require('../lib/result-rebuild.ts');
const { selectMatches, shieldWinner, shieldWins, matchCompetition } = require('../lib/competition.ts');
const { buildPlayerSummaries } = require('../lib/demo-data.ts');
const { attachStyleProfiles } = require('../lib/player-style-profiles.ts');
const { buildMatchReport } = require('../lib/match-report.ts');
const { buildMatchdayBalance } = require('../lib/matchday-balance.ts');
const { buildHealthStats } = require('../lib/health-stats.ts');
const awards = require('../lib/season-awards.ts');
const { awardFixture } = require('./award-fixture.cjs');
const { awardViews } = require('./award-views.cjs');

const fixture = awardFixture();
const cup = { ...fixture.matches[0], id: 999, date: '2026-03-10', competition: 'immunity_shield', score: '19-0' };
test('Shield add/edit/delete/backdating never changes any public rating, counter or league award', () => {
  const baseline = buildLeagueRatings(fixture);
  for (const extra of [[cup], [{...cup,score:'0-18'}], [{...cup,date:'2024-01-01'}], [{...cup,date:'2026-12-31'}], [{...cup,score:'0-0'}], []]) {
    const matches = [...fixture.matches,...extra];
    const replay = buildLeagueRatings(fixture,matches);
    assert.deepEqual(replay.players,baseline.players);
    assert.deepEqual(replay.history,baseline.history);
    assert.equal(replay.summary.ratedMatches,fixture.matches.length);
    assert.equal(replay.shadow.records.length,matches.length);
    if(extra.length) assert.notDeepEqual(replay.shadow.state,baseline.shadow.state);
    const withHistory = { ...fixture, players: replay.players, matches, history: replay.history.map((h,i)=>({...h,id:i+1})) };
    const actualViews=awardViews(awards,withHistory), expectedViews=awardViews(awards,{...withHistory,matches:fixture.matches});
    for(const [key,value] of Object.entries(expectedViews)) assert.deepEqual(actualViews[key],value,key);
    for(const key of Object.keys(actualViews).filter(key=>!expectedViews[key])) {
      assert.deepEqual(actualViews[key].mvp,[]);assert.deepEqual(actualViews[key].improved.rows,[]);
    }
    assert.deepEqual(awards.seasonAwardSnapshot(replay.players,matches,withHistory.history,'2026'),
      awards.seasonAwardSnapshot(replay.players,fixture.matches,withHistory.history,'2026'));
  }
});
test('competition is explicit and an unknown classification cannot silently become rated', () => {
  assert.equal(matchCompetition({...cup,competition:undefined}),'league');
  assert.throws(()=>buildLeagueRatings(fixture,[{...cup,competition:'unknown'}]),/Unknown match competition/);
  assert.throws(()=>applyLeagueChange([cup],{action:'edit',original:{...cup,competition:'league'},match:cup}),/changed/);
  assert.equal(selectMatches([{...cup,competition:'league'}]).length,1,'a Tuesday can be a league match');
  assert.equal(selectMatches([{...cup,date:'2026-03-12'}]).length,0,'a Thursday can be a Shield match');
});
test('draws award nobody, wins derive from saved participants, and duplicate reads cannot duplicate honours', () => {
  const winner=fixture.players.find(p=>cup.team_a.split(',').includes(p.name));
  const loser=fixture.players.find(p=>cup.team_b.split(',').includes(p.name));
  assert.equal(shieldWinner(cup),'A');
  assert.equal(shieldWins(winner,[cup,cup]),1);
  assert.equal(shieldWins(loser,[cup]),0);
  assert.equal(shieldWinner({...cup,score:'8:8',result:'A'}),null);
  assert.equal(shieldWins(winner,[{...cup,score:'8-8'}]),0);
  assert.equal(shieldWinner({...cup,processed:0}),null);
  assert.equal(shieldWins({...winner,is_active:0},[cup]),1);
});
test('season and competition scope include the same football, style, highlight and health matches', () => {
  const p={...fixture.players.find(p=>cup.team_a.split(',').includes(p.name))};
  const league={...cup,id:1,competition:'league',score:'5-4'};
  const data=attachStyleProfiles([p],[{match_id:1,player_name:p.name,goals:2,assists:1},{match_id:cup.id,player_name:p.name,goals:7,assists:3}],[]);
  const matches=[cup,league];
  const activities=matches.map(m=>({league_id:1,player_id:p.id,match_id:m.id,distance_km:m.id===1?2:5,duration_minutes:60}));
  for(const [scope,played,goals,distance] of [['league',1,2,2],['immunity_shield',1,7,5],['all',2,9,7]]) {
    const selected=selectMatches(matches,scope,'2026');
    const summary=buildPlayerSummaries(data,selected)[0];
    assert.equal(summary.matches_played,played);
    assert.equal(summary.style_profile.historicGoals,goals);
    const health=buildHealthStats(1,[p],selected,activities);
    assert.equal(health.players[0].entries.length,played);
    assert.equal(health.players[0].entries.reduce((sum,e)=>sum+(e.reading?.distance_km||0),0),distance);
  }
  const empty=buildPlayerSummaries(data,selectMatches(matches,'immunity_shield','2025'))[0];
  assert.equal(empty.matches_played,0); assert.equal(empty.shieldWins,0);
  assert.ok(!empty.style_profile || empty.style_profile.historicGoals===0);
});
test('reports keep league milestones separate, show Shield honours and use both competitions for forecasts', () => {
  const shield={...cup,date:'2026-12-22'}, later={...fixture.matches.at(-1),id:1000,date:'2026-12-24'};
  const all=[...fixture.matches,shield,later];
  const leagueReport=buildMatchReport({match:later,matches:all,players:fixture.players,mmrHistory:[]});
  const baseline=buildMatchReport({match:later,matches:all.filter(m=>m.id!==shield.id),players:fixture.players,mmrHistory:[]});
  assert.equal(leagueReport.matchNumber,baseline.matchNumber);
  assert.deepEqual(leagueReport.playerNotes,baseline.playerNotes);
  assert.deepEqual(leagueReport.changes,baseline.changes);
  assert.notDeepEqual(leagueReport.rating.prediction,baseline.rating.prediction);
  const report=buildMatchReport({match:shield,matches:all,players:fixture.players,mmrHistory:[]});
  assert.match(report.headline,/win the Slimfast Cup/);
  assert.equal(report.changes.length,5);
  assert.equal(report.matchNumber,1);
  const draw={...shield,score:'7-7'};
  const drawn=buildMatchReport({match:draw,matches:[...fixture.matches,draw],players:fixture.players,mmrHistory:[]});
  assert.equal(drawn.changes.length,0);assert.match(drawn.headline,/nobody wins/);
  const before=buildMatchdayBalance(fixture.players,all,'2026-12-21');
  assert.ok(before.history.every(m=>m.id!==shield.id && m.id!==later.id));
});
