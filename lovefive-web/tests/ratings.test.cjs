require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const model = require('../lib/balance-model.ts');
const rebuild = require('../lib/result-rebuild.ts');
const { scoreParts } = require('../lib/demo-data.ts');
const { buildEngineState } = require('../lib/team-ai-engine.ts');
const { loadAllPages } = require('../lib/pagination.ts');
const players = Array.from({ length: 12 }, (_, i) => ({ id: i+1, name: `p${i+1}`, display_name: null,
  mmr: 1000, initial_mmr: 1000, is_active: 1, fitness: 'Medium' }));
const a = [1,2,3,4,5], b = [6,7,8,9,10];
const match = { id: 1, date: '2026-01-01', team_a: 'p1,p2,p3,p4,p5', team_b: 'p6,p7,p8,p9,p10', score: '5-3', result: 'A', processed: 1 };
const workspace = { players, matches: [match], history: [], adjustments: [], token: 'test' };
const close = (a, b, eps=1e-6) => assert.ok(Math.abs(a-b)<eps, `${a} != ${b}`);

test('scores require two whole nonnegative numbers', () => {
  for (const input of ['', '-', ':', '10-', '-8', '1.5-2', 'Infinity-2', '1-2-3', '1e2-4', '1000-2']) assert.equal(scoreParts(input), null, input);
  assert.deepEqual(scoreParts(' 10 : 8 '), [10,8]);
  assert.deepEqual(scoreParts('0-0'), [0,0]);
});
test('pagination continues below requested server cap and propagates failures', async () => {
  const source = Array.from({length:1103}, (_,i)=>i);
  const rows = await loadAllPages((from,to) => Promise.resolve({ data:source.slice(from, Math.min(to+1,from+137)), error:null }));
  assert.deepEqual(rows,source);
  await assert.rejects(loadAllPages(()=>Promise.resolve({ data:null,error:{message:'offline'} })), /offline/);
});
test('labels and goal signs give symmetric predictions and posterior', () => {
  const left = model.createBalanceState(players), right = model.createBalanceState(players);
  model.updateBalance(left,a,b,4); model.updateBalance(right,b,a,-4);
  assert.deepEqual(left,right);
  const p = model.predictBalance(left,a,b), q = model.predictBalance(left,b,a);
  close(p.expected+q.expected,1); close(p.close,q.close); close(p.win,q.loss);
  close(p.win+p.draw+p.loss,1);
});
test('a balanced draw does not transfer strength between teammates', () => {
  const s = model.createBalanceState(players); s.mu[0]=2; for(let i=1;i<5;i++)s.mu[i]=-.5;
  const before=s.mu.slice(); model.updateBalance(s,a,b,0); assert.deepEqual(s.mu,before);
});
test('learning reduces uncertainty; time increases it without reducing ability', () => {
  const s = model.createBalanceState(players); model.advanceBalance(s,'2026-01-01');
  const prior=model.predictBalance(s,a,b).skillVariance;
  model.updateBalance(s,a,b,1); const posterior=model.predictBalance(s,a,b).skillVariance;
  assert.ok(posterior<prior); const means=s.mu.slice();
  model.advanceBalance(s,'2027-01-01'); assert.ok(model.predictBalance(s,a,b).skillVariance>posterior); assert.deepEqual(means,s.mu);
  assert.throws(()=>model.advanceBalance(s,'2026-01-01'),/date order/);
});
test('covariance stays positive for repeated extreme results', () => {
  const s=model.createBalanceState(players);
  for(let k=0;k<300;k++)model.updateBalance(s,a,b,k%3===0?50:-1);
  const n=s.mu.length, L=Array.from({length:n},()=>Array(n).fill(0));
  for(let i=0;i<n;i++) for(let j=0;j<=i;j++) {
    let v=s.covariance[i][j]; for(let k=0;k<j;k++)v-=L[i][k]*L[j][k];
    if(i===j){ assert.ok(v>0); L[i][j]=Math.sqrt(v); } else L[i][j]=v/L[j][j];
  }
});
test('future outcomes never alter earlier forecasts', () => {
  const games=[match,{...match,id:2,date:'2026-01-08',score:'8-4'}];
  const before=model.replayBalance(players,games).records;
  const after=model.replayBalance(players,[games[0],{...games[1],score:'0-20'}]).records;
  assert.deepEqual(before,after);
});
test('all 126 splits are distinct; locks, missing players and impossible locks are respected', () => {
  const s=model.createBalanceState(players), selected=players.slice(0,10);
  const all=model.selectBalanceTeams(s,selected); assert.equal(all.length,126); assert.equal(new Set(all.map(o=>o.key)).size,126);
  const locked=model.selectBalanceTeams(s,selected,[1,2],[6]);
  assert.ok(locked.length>0); for(const o of locked){ assert.ok(o.teamA.some(p=>p.id===1)&&o.teamA.some(p=>p.id===2)&&o.teamB.some(p=>p.id===6)); }
  assert.equal(model.selectBalanceTeams(s,selected,[1],[1]).length,0);
  assert.throws(()=>model.predictBalance(s,[1,1,2,3,4],b),/different/);
});
test('baselines and dated adjustments survive rebuilding and backdated games', () => {
  const custom=players.map(p=>({...p,initial_mmr:p.id===12?1200:1000}));
  const adjusted={...workspace,players:custom,adjustments:[{player_id:1,delta:35,after_date:match.date,after_match_id:match.id}]};
  const base=rebuild.buildLeagueRatings({...workspace,players:custom});
  const actual=rebuild.buildLeagueRatings(adjusted);
  assert.equal(actual.players.find(p=>p.id===12).mmr,1200);
  close(actual.players[0].mmr,base.players[0].mmr+35);
  assert.deepEqual(actual.shadow.state,base.shadow.state);
  const next={...match,id:2,date:'2026-01-08'};
  const extended=rebuild.buildLeagueRatings(adjusted,[match,next]);
  assert.equal(extended.history.find(h=>h.match_id===2&&h.player_id===1).mmr_before,actual.players[0].mmr);
});
test('invalid lineups and stale edits stop before saving', () => {
  assert.throws(()=>rebuild.buildLeagueRatings(workspace,[{...match,team_b:match.team_a}]),/distinct/);
  assert.throws(()=>rebuild.applyLeagueChange([match],{action:'edit',original:{...match,score:'2-2'},match}),/changed/);
});
test('pending matches never count as played draws', () => {
  const s=buildEngineState(players,[{...match,processed:0,score:null}]);
  assert.equal(s.totalMatches.get('p1'),0);
  assert.equal(model.replayBalance(players,[{...match,processed:0}]).records.length,0);
});
test('winning margins cannot increase the heavy-loss penalty', () => {
  const games=[]; let id=0;
  for(let i=0;i<12;i++)games.push({...match,id:++id,score:i<6?'1-0':'0-1'});
  // p1 and p2 win separately enough for the shared pair to underperform.
  for(let i=0;i<20;i++) {
    games.push({...match,id:++id,team_a:'p1,p3,p4,p5,p11',team_b:'p6,p7,p8,p9,p10',score:'1-0'});
    games.push({...match,id:++id,team_a:'p2,p3,p4,p5,p12',team_b:'p6,p7,p8,p9,p10',score:'1-0'});
  }
  const first=buildEngineState(players,games).badPairPenalty.get('p1|p2');
  const second=buildEngineState(players,games.map(g=>g.id<=6?{...g,score:'10-0'}:g)).badPairPenalty.get('p1|p2');
  assert.ok(first>0); close(first,second);
});
if(process.env.LOVEFIVE_TEST_SNAPSHOT) test('real history: all current ratings are preserved, margin prototype reproduced', () => {
  const data=JSON.parse(fs.readFileSync(process.env.LOVEFIVE_TEST_SNAPSHOT,'utf8').replace(/^\uFEFF/,''));
  const w={players:data.players,matches:data.matches,history:data.mmr_history||[],adjustments:[],token:'offline'};
  const result=rebuild.buildLeagueRatings(w);
  for(const p of result.players)close(p.mmr,data.players.find(q=>q.id===p.id).mmr);
  const forecast=result.shadow.records.slice(46), matches=data.matches.filter(m=>m.processed===1).sort(model.compareMatches).slice(46);
  const mse=forecast.reduce((sum,row,i)=>{const score=scoreParts(matches[i].score); const y=score[0]===score[1]?.5:score[0]>score[1]?1:0;return sum+(row.prediction.expected-y)**2;},0)/forecast.length;
  close(mse,0.1783639133,1e-6);
  console.log('Real-history validation:',result.summary,'prediction error',mse.toFixed(6));
});
