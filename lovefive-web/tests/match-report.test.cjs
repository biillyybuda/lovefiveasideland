require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { buildMatchReport, completedReportMatch } = require('../lib/match-report.ts');
const { seasonAwardSnapshot } = require('../lib/season-awards.ts');
const { groupsForLeague } = require('../lib/league-groups.ts');
const { saveMatchNote } = require('../lib/match-notes.ts');
const { applyLeagueChange } = require('../lib/result-rebuild.ts');
const players = Array.from({ length: 12 }, (_, i) => ({ id: i + 1, name: `player${i + 1}`, display_name: `Player ${i + 1}`,
  is_active: 1, mmr: 1000, initial_mmr: 1000, matches_played: 0, wins: 0, losses: 0, draws: 0, win_streak: 0, lose_streak: 0 }));
const fixture = (id, date = '2026-01-01', extra = {}) => ({ id, date, team_a: 'player1,player2,player3,player4,player5',
  team_b: 'player6,player7,player8,player9,player10', score: '9-4', result: 'A', processed: 1, ...extra });
const historyFor = matches => matches.flatMap(m => [...m.team_a.split(','), ...m.team_b.split(',')].map((name, index) => ({
  id: m.id * 100 + index, match_id: m.id, date: m.date, player_id: Number(name.replace('player', '')), mmr_before: 1000, mmr_after: 1010
})));
const build = (match, matches = [match], extra = {}) => buildMatchReport({ match, matches, players, mmrHistory: historyFor(matches), ...extra });
const group = { name: 'School', description: 'A league-provided group', playerIds: [1,2,3,4,5], minimumTogether: 4 };

test('season scope and historical cutoff exclude old years, later same-day IDs, pending and malformed results', () => {
  const prior = fixture(1, '2025-12-31', { score: '99-1' });
  const before = fixture(2, '2026-01-01', { score: '3-1' });
  const target = fixture(3, '2026-01-01', { score: '8-4' });
  const future = fixture(4, '2026-01-01', { score: '80-1' });
  const matches = [prior, before, target, future, fixture(5, '2026-01-01', { processed: 0 }), fixture(6, '2026-02-30')];
  const r = build(target, matches);
  assert.equal(r.matchNumber, 2);
  assert.equal(r.scoreContext.averageBefore, 4);
  assert.equal(r.playerNotes[0].appearances, 2);
  assert.deepEqual(r, build(target, [prior, before, target]));
  assert.equal(completedReportMatch(fixture(10, '2026-02-30')), false);
  assert.throws(() => build(fixture(10, '2026-01-01', { score: '10-' })), /completed result/);
});

test('first recorded appearance differs from first season appearance, with archived players and aliases resolved', () => {
  const old = fixture(1, '2025-12-31');
  const current = fixture(2, '2026-01-01', { team_a: '  Player 1,player2,player3,player4,player11' });
  const roster = players.map(p => p.id === 1 ? { ...p, is_active: 0 } : p);
  const r = build(current, [old, current], { players: roster });
  assert.ok(r.stories.some(s => s.key === 'season-first-player1'));
  assert.ok(!r.stories.some(s => s.key === 'first-player1'));
  assert.ok(r.stories.some(s => s.key === 'first-player11'));
  assert.ok(r.teamA.includes('Player 1'));
  assert.equal(r.playerNotes.find(p => p.name === 'Player 1').appearances, 1);
});

test('career milestones use the archive while season streaks reset at the year boundary', () => {
  const old = Array.from({length:24}, (_,i) => fixture(i + 1, `2025-12-${String(i + 1).padStart(2,'0')}`, {score:'0-1'}));
  const target = fixture(25, '2026-01-01');
  const r = build(target, [...old, target]);
  assert.ok(r.stories.some(s => s.key === 'career-player1' && s.scope === 'Recorded career'));
  assert.ok(!r.stories.some(s => s.key === 'run-player1'));
});

test('season partnership breakthrough requires prior games without a win', () => {
  const matches = [fixture(1,'2026-01-01',{score:'1-2'}),fixture(2,'2026-01-02',{score:'1-1'}),fixture(3,'2026-01-03',{score:'0-3'}),fixture(4,'2026-01-04')];
  const r = build(matches.at(-1), matches);
  assert.ok(r.stories.some(s => s.key === 'pair-player1-player2' && /4th attempt/.test(s.detail)));
  assert.ok(!build(fixture(1)).stories.some(s => s.key.startsWith('pair-')));
});

test('HLHS is scoped to its league and exactly the five user-specified stable identities', () => {
  assert.deepEqual(groupsForLeague(1)[0].playerIds, [12,2,11,10,6]);
  assert.equal(groupsForLeague(1)[0].minimumTogether, 4);
  assert.deepEqual(groupsForLeague(2), []);
});

test('four together triggers a group headline and guest label, while three together does not', () => {
  const four = fixture(1,'2026-01-01',{team_a:'player1,player2,player3,player4,player11',score:'0-2'});
  const r = build(four, [four], {groups:[group]});
  assert.match(r.headline,/School/);
  assert.equal(r.teamALabel,'School + Player 11');
  assert.equal(r.groups[0].seasonRecord.played,1);
  assert.equal(r.groups[0].allFiveRecord.played,0);
  const three = {...four,team_a:'player1,player2,player3,player11,player12'};
  assert.equal(build(three,[three],{groups:[group]}).groups.length,0);
  const five = fixture(2);
  assert.equal(build(five,[five],{groups:[group]}).teamALabel,'School');
});

test('group records follow the group across sides, distinguish the current core and all five, and stop at the match', () => {
  const old = fixture(1,'2025-12-31',{score:'50-0'});
  const five = fixture(2,'2026-01-01',{score:'1-3'});
  const otherFour = fixture(3,'2026-01-02',{team_a:'player1,player2,player3,player5,player11',score:'1-1'});
  const target = fixture(4,'2026-01-03',{team_a:'player6,player7,player8,player9,player10',team_b:'player1,player2,player3,player4,player11',score:'2-0'});
  const future = fixture(5,'2026-01-03');
  const r = build(target,[old,five,otherFour,target,future],{groups:[group]});
  assert.deepEqual(r.groups[0].seasonRecord,{played:3,wins:0,draws:1,losses:2});
  assert.deepEqual(r.groups[0].currentCoreRecord,{played:2,wins:0,draws:0,losses:2});
  assert.equal(r.groups[0].allFiveRecord.played,1);
  assert.equal(r.teamBLabel,'School + Player 11');
});

test('group headlines celebrate a drought-breaking win and never call a draw a defeat', () => {
  const losses = Array.from({length:3},(_,i)=>fixture(i+1,`2026-01-0${i+1}`,{score:'1-3'}));
  const win = fixture(4,'2026-01-04');
  assert.match(build(win,[...losses,win],{groups:[group]}).headline,/break the losing spell/);
  const draw = {...win,score:'1-1'};
  assert.match(build(draw,[...losses,draw],{groups:[group]}).headline,/all square/);
});

test('historical ratings remain unknown without recorded history and prose does not invent match flow', () => {
  const target = fixture(1);
  const r = build(target,[target],{mmrHistory:[]});
  assert.equal(r.rating.avgA,null);
  assert.equal(r.playerNotes[0].delta,null);
  assert.doesNotMatch(JSON.stringify(r),/late on|last stretch|high tempo|controlled the game/i);
});

test('snapshot leaders and report standings use the identical shared season award calculation', () => {
  const matches = Array.from({length:8},(_,i)=>fixture(i+1,`2026-01-0${i+1}`));
  const target=matches.at(-1), mmrHistory=historyFor(matches);
  const expected=seasonAwardSnapshot(players,matches,mmrHistory,'2026');
  const r=build(target,matches);
  assert.equal(r.standings[0].name,expected.mvp[0].label);
  assert.equal(r.standings[0].value,`${expected.mvp[0].value} pts`);
});

test('note updates use only the note RPC, preserve text, and reject stale and oversized submissions', async () => {
  const calls=[];
  const client={rpc:async (name,args)=>{calls.push({name,args});return {data:args.p_note,error:null};}};
  assert.equal(await saveMatchNote(client,1,2,'old','  New <b>note</b>  '),'New <b>note</b>');
  assert.equal(calls[0].name,'save_match_review_note');
  assert.equal(calls[0].args.p_expected_note,'old');
  await assert.rejects(saveMatchNote(client,1,2,'','x'.repeat(1501)),/1500/);
  assert.equal(calls.length,1);
  await assert.rejects(saveMatchNote({rpc:async()=>({error:{code:'40001'}})},1,2,'old','new'),/Someone updated/);
  const target=fixture(1,'2026-01-01',{review_note:'edited elsewhere'});
  assert.throws(()=>applyLeagueChange([target],{action:'edit',original:{...target,review_note:'old'},match:target}),/changed since/);
});

if(process.env.LOVEFIVE_REPORT_SNAPSHOT) test('real result 129: season award parity, Cam, HLHS, partnership, score rank and career milestone',()=>{
  const source=JSON.parse(fs.readFileSync(process.env.LOVEFIVE_REPORT_SNAPSHOT,'utf8'));
  const target=source.matches.find(m=>m.id===129);
  const r=buildMatchReport({...source,match:target,awardPlayers:source.players.filter(p=>p.is_active===1),leagueId:1});
  assert.equal(r.headline,'HLHS + Jos, same old story');
  assert.deepEqual(r.groups[0].seasonRecord,{played:9,wins:0,draws:1,losses:8});
  assert.deepEqual(r.groups[0].currentCoreRecord,{played:7,wins:0,draws:0,losses:7});
  assert.deepEqual(r.groups[0].allFiveRecord,{played:4,wins:0,draws:0,losses:4});
  assert.equal(r.matchNumber,33);
  assert.equal(r.standings[0].value,'98 pts');
  assert.equal(r.standings[1].name,'Kris');
  assert.equal(r.standings[1].value,'+37 pts');
  assert.deepEqual(r.highlights.map(s=>s.key),['group-HLHS','first-cam','mvp-3']);
  assert.match(r.changes.find(c=>c.name==='Jacob'&&c.label==='2026 MVP race').after,/2nd · 78 pts/);
  assert.match(r.stories.find(s=>s.key==='pair-sug-sam k').detail,/6th attempt/);
  assert.match(r.playerNotes.find(p=>p.name==='Tom D').note,/50th recorded career/);
  assert.equal(r.scoreContext.rank,5);
  assert.equal(r.scoreContext.tied,true);
  assert.equal(r.scoreContext.averageBefore.toFixed(1),'20.8');
  assert.equal(r.previousMeeting.shared,8);
});
