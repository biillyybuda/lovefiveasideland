require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { buildMatchdayBalance, matchdayOptions } = require('../lib/matchday-balance.ts');
const { createBalanceState, selectBalanceTeams } = require('../lib/balance-model.ts');
const { awardFixture } = require('./award-fixture.cjs');
const signature = options => options.map(({key,prediction,roleGap,repeatCost})=>({key,prediction,roleGap,repeatCost}));

test('the default pipeline includes archived history, ignores future games and never mutates public data', () => {
  const data = awardFixture(), original = structuredClone(data);
  const model = buildMatchdayBalance(data.players,data.matches,'2026-04-01');
  assert.ok(model.ratings.find(row=>row.playerId===15).appearances > 0);
  const changed = data.matches.map(match=>match.date > '2026-04-01' ? {...match,score:'999-0'} : match);
  assert.deepEqual(buildMatchdayBalance(data.players,changed,'2026-04-01'),model);
  const selected = data.players.filter(player=>player.is_active === 1).slice(0,10);
  const options = matchdayOptions(model,selected,[],[]);
  assert.equal(options.length,126);
  assert.deepEqual(signature(matchdayOptions(model,[...selected].reverse(),[],[])),signature(options));
  const alteredMmr = data.players.map(player=>({...player,mmr:3000-player.id*200}));
  const alteredModel = buildMatchdayBalance(alteredMmr,data.matches,'2026-04-01');
  assert.deepEqual(alteredModel,model);
  assert.deepEqual(signature(matchdayOptions(alteredModel,alteredMmr.filter(p=>selected.some(s=>s.id===p.id)),[],[])),signature(options));
  assert.deepEqual(data,original);
});

test('uncertainty in linked players actually affects equal-strength recommendations', () => {
  const selected = awardFixture().players.slice(0,10);
  const state = createBalanceState(selected);
  state.covariance[0][1] = state.covariance[1][0] = -0.8;
  const options = selectBalanceTeams(state,selected);
  assert.ok(options[0].prediction.close > options.at(-1).prediction.close);
  const top = options[0];
  assert.equal(top.teamA.some(p=>p.id===1),top.teamA.some(p=>p.id===2));
  assert.equal(top.prediction.meanMargin,0);
});

test('date, empty history, pending games and impossible locks have explicit outcomes', () => {
  const data = awardFixture();
  assert.throws(()=>buildMatchdayBalance(data.players,data.matches,''),/valid match date/);
  const empty = buildMatchdayBalance(data.players,data.matches.map(m=>({...m,processed:0})),'2026-01-01');
  assert.equal(empty.history.length,0);
  assert.ok(empty.ratings.every(r=>r.rating===1000&&r.appearances===0));
  const selected = data.players.slice(0,10);
  assert.equal(matchdayOptions(empty,selected.slice(0,9),[],[]).length,0);
  assert.equal(matchdayOptions(empty,selected,[1,2,3,4,5,6],[]).length,0);
  const fixed = matchdayOptions(empty,selected,[1,2,3,4,5],[6,7,8,9,10]);
  assert.equal(fixed.length,1);
  assert.deepEqual(fixed[0].teamA.map(p=>p.id),[1,2,3,4,5]);
});
