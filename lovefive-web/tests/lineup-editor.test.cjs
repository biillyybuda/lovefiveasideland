require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { awardFixture } = require('./award-fixture.cjs');
const { FORMATIONS, buildLineupChemistry, playerLink, initialLineup, suggestTeam, validateDraft, swapLineupPlayers, swapLineupColours, shieldName } = require('../lib/lineup-editor.ts');
const { buildMatchdayBalance } = require('../lib/matchday-balance.ts');
const { predictBalance } = require('../lib/balance-model.ts');
const { drawFormationPoster } = require('../lib/lineup-poster.ts');
const empty = () => ({players: new Map(), pairs: new Map()});
const date = '2026-09-10';
function fixture() {
  const data = awardFixture();
  const chemistry = buildLineupChemistry(data.players, data.matches, date);
  const draft = initialLineup(date, [1,2,3,4,5], [6,7,8,9,10], data.players, chemistry);
  return {...data, chemistry, draft};
}

test('chemistry resolves aliases, counts draws as non-wins and excludes pending/future/duplicate games', () => {
  const players = awardFixture().players;
  players[0].display_name = 'Bill';
  const match = {id:1,date:'2026-09-01',processed:1,score:'5-2',team_a:'Bill,Player 2,Player 3,Player 4,Player 5',team_b:'Player 6,Player 7,Player 8,Player 9,Player 10'};
  const chemistry = buildLineupChemistry(players, [match, match, {...match,id:2,score:'2-2'}, {...match,id:3,processed:0}, {...match,id:4,date:'2026-09-11'}],date);
  const link = playerLink(chemistry,1,2);
  assert.equal(link.games,2); assert.equal(link.wins,1); assert.equal(link.winRate,.5);
  assert.equal(link.averageRate,.5); assert.equal(link.colour,'amber');
  assert.deepEqual(playerLink(chemistry,2,1),{...link,a:2,b:1});
  assert.throws(()=>buildLineupChemistry(players,[{...match,team_a:'Missing,Player 2,Player 3,Player 4,Player 5'}],date),/cannot be found/i);
});

test('links use the requested five-point band; missing history stays unknown and red', () => {
  const chemistry = {players:new Map([[1,{wins:60,games:100}],[2,{wins:40,games:100}]]),pairs:new Map()};
  for (const [wins,colour] of [[56,'green'],[55,'amber'],[50,'amber'],[45,'amber'],[44,'red']]) {
    chemistry.pairs.set('1:2',{wins,games:100});
    assert.equal(playerLink(chemistry,1,2).colour,colour);
  }
  const unseen = playerLink(chemistry,1,3);
  assert.equal(unseen.colour,'red'); assert.equal(unseen.winRate,null); assert.equal(unseen.difference,null);
});

test('all formations cover every position and suggestions preserve keepers, squads and preferences', () => {
  const {players,chemistry,draft} = fixture();
  const keeper = draft.blue.ids[0];
  draft.locks[keeper] = {side:'blue',role:'GK'};
  const striker = draft.blue.ids[4]; draft.preferences[striker] = 'ST';
  for (const formation of Object.keys(FORMATIONS)) {
    assert.deepEqual([...new Set(FORMATIONS[formation].slots.map(s=>s.role))].sort(), ['DF','GK','MF','ST']);
    const team = suggestTeam(draft,'blue',players,chemistry,formation);
    assert.equal(team.ids[0],keeper);
    assert.equal(FORMATIONS[formation].slots[team.ids.indexOf(striker)].role,'ST');
    assert.deepEqual([...team.ids].sort(),[...draft.blue.ids].sort());
    assert.deepEqual(suggestTeam(draft,'blue',players,chemistry,formation),team);
    validateDraft({...draft,blue:team},players);
  }
  draft.blue = suggestTeam(draft,'blue',players,chemistry,'1-2-1-1');
  draft.locks[draft.blue.ids[1]] = {side:'blue',role:'DF'};
  draft.locks[draft.blue.ids[2]] = {side:'blue',role:'DF'};
  assert.throws(()=>suggestTeam(draft,'blue',players,chemistry,'1-1-1-2'),/cannot keep every position lock/);
});

test('recorded scoring and assists guide roles when links are equal, without using saves as keeper ability', () => {
  const players = awardFixture().players;
  players[1].style_profile = {historicMatches:20,goalsPerGame:3,assistsPerGame:.1,saves:0};
  players[2].style_profile = {historicMatches:20,goalsPerGame:.1,assistsPerGame:2,saves:500};
  const draft = initialLineup(date,[1,2,3,4,5],[6,7,8,9,10],players,empty());
  draft.blue = {formation:'1-1-2-1',ids:[1,2,3,4,5]}; draft.locks[1] = {side:'blue',role:'GK'};
  const team = suggestTeam(draft,'blue',players,empty(),'1-1-2-1');
  assert.equal(FORMATIONS[team.formation].slots[team.ids.indexOf(2)].role,'ST');
  assert.equal(FORMATIONS[team.formation].slots[team.ids.indexOf(3)].role,'MF');
  players[2].style_profile.saves = 0;
  assert.deepEqual(suggestTeam(draft,'blue',players,empty(),'1-1-2-1'),team);
});

test('swaps are immutable, preserve ten unique players and refuse locked players', () => {
  const {players,draft} = fixture(), original = structuredClone(draft);
  const swapped = swapLineupPlayers(draft,{side:'blue',index:2},{side:'red',index:3});
  validateDraft(swapped,players);
  assert.equal(swapped.red.ids[3],draft.blue.ids[2]); assert.equal(swapped.blue.ids[2],draft.red.ids[3]);
  assert.deepEqual(draft,original);
  draft.locks[draft.blue.ids[0]] = {side:'blue',role:'GK'};
  assert.throws(()=>swapLineupPlayers(draft,{side:'blue',index:0},{side:'red',index:0}),/Unlock both/);
  const recoloured = swapLineupColours(draft);
  assert.deepEqual(recoloured.red,draft.blue); assert.equal(recoloured.locks[draft.blue.ids[0]].side,'red');
  validateDraft(recoloured,players);
});

test('restoration rejects corrupt dates, duplicate, archived or missing players and moved locks', () => {
  const {players,draft} = fixture();
  assert.deepEqual(validateDraft(JSON.parse(JSON.stringify(draft)),players),draft);
  assert.throws(()=>validateDraft({...draft,date:'2026-02-30'},players),/valid match date/);
  assert.throws(()=>validateDraft({...draft,blue:{...draft.blue,formation:'1-2-2'}},players),/every position/);
  for (const bad of [draft.red.ids[0],15,999]) {
    const copy = structuredClone(draft); copy.blue.ids[0] = bad;
    assert.throws(()=>validateDraft(copy,players));
  }
  const copy = structuredClone(draft); copy.locks[copy.blue.ids[0]] = {side:'red',role:'GK'};
  assert.throws(()=>validateDraft(copy,players),/locked player has moved/);
});

test('positions leave forecast unchanged; changing sides recalculates it and colours reverse odds', () => {
  const {players,matches,draft} = fixture();
  const state = buildMatchdayBalance(players,matches,date).state;
  const forecast = d => predictBalance(state,[...d.blue.ids].sort((a,b)=>a-b),[...d.red.ids].sort((a,b)=>a-b));
  const before = forecast(draft);
  assert.deepEqual(forecast(swapLineupPlayers(draft,{side:'blue',index:0},{side:'blue',index:4})),before);
  assert.notEqual(forecast(swapLineupPlayers(draft,{side:'blue',index:0},{side:'red',index:4})).win,before.win);
  const reversed = forecast(swapLineupColours(draft));
  assert.ok(Math.abs(reversed.win-before.loss)<1e-12); assert.ok(Math.abs(reversed.draw-before.draw)<1e-12);
});

test('poster rejects incomplete forecasts and name wrapping keeps every character', () => {
  const {players,chemistry,draft} = fixture();
  for (const forecast of [{win:NaN,draw:.1,loss:.4},{win:.5,draw:.2},{win:.5,draw:.2,loss:.4}]) {
    assert.throws(()=>drawFormationPoster({}, {players,chemistry,draft,forecast}),/forecast is unavailable/);
  }
  for (const name of ['Alexandertheverylongname Smith','Élodie João-Silva','Sam K']) {
    const fitted = shieldName(name);
    assert.equal(fitted.lines.join('').replaceAll(' ',''),name.toUpperCase().replaceAll(' ',''));
    assert.ok(fitted.size*fitted.lines.length<=48);
  }
});
