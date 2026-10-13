require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { buildMatchdayBalance, drawMatchdayOptions, matchdayOptions } = require('../lib/matchday-balance.ts');
const { createBalanceState, predictBalance, selectBalanceTeams } = require('../lib/balance-model.ts');
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

test('colour draws remove the first-player bias and shuffle both rosters without changing the ranked splits', () => {
  const selected = awardFixture().players.slice(0,10);
  const options = selectBalanceTeams(createBalanceState(selected),selected);
  const original = structuredClone(options);
  const positions = { A: new Set(), B: new Set() };
  let blueCount = 0;
  const ids = team => team.map(p=>p.id).sort((a,b)=>a-b).join(',');
  for (let seed = 0; seed < 256; seed++) {
    const drawn = drawMatchdayOptions(options,[],[],seed);
    assert.deepEqual(drawn.map(o=>o.key),options.map(o=>o.key));
    drawn.forEach((option,index)=>{
      assert.deepEqual([ids(option.teamA),ids(option.teamB)].sort(),[ids(options[index].teamA),ids(options[index].teamB)].sort());
      assert.equal(option.prediction.close,options[index].prediction.close);
      assert.equal(option.roleGap,options[index].roleGap);
      assert.equal(option.repeatCost,options[index].repeatCost);
    });
    const top = drawn[0], firstId = selected[0].id;
    const side = top.teamA.some(p=>p.id===firstId) ? 'A' : 'B';
    if (side === 'A') blueCount++;
    positions[side].add(top[`team${side}`].findIndex(p=>p.id===firstId));
  }
  assert.ok(blueCount > 90 && blueCount < 166,`First player was blue in ${blueCount}/256 draws`);
  assert.equal(positions.A.size,5);
  assert.equal(positions.B.size,5);
  assert.deepEqual(options,original);
});

test('a drawn forecast follows its colours and a split keeps its draw when ranking or input order changes', () => {
  const data = awardFixture();
  const selected = data.players.filter(p=>p.is_active===1).slice(0,10);
  const model = buildMatchdayBalance(data.players,data.matches,'2026-04-01');
  const options = matchdayOptions(model,selected,[],[]);
  const drawn = drawMatchdayOptions(options,[],[],42);
  let swapped = 0;
  drawn.forEach((option,index)=>{
    const expected = predictBalance(model.state,option.teamA.map(p=>p.id),option.teamB.map(p=>p.id));
    for (const key of Object.keys(expected)) {
      assert.ok(Math.abs(option.prediction[key]-expected[key]) < 1e-12,`${key} must follow the displayed sides`);
    }
    if (!option.teamA.some(p=>p.id===options[index].teamA[0].id)) swapped++;
  });
  assert.ok(swapped > 0 && swapped < options.length);
  assert.deepEqual(drawMatchdayOptions([...options].reverse(),[],[],42).reverse(),drawn);
  assert.deepEqual(drawMatchdayOptions(matchdayOptions(model,[...selected].reverse(),[],[]),[],[],42),drawn);
  assert.deepEqual(drawMatchdayOptions(options,[],[],42),drawn);
  assert.notDeepEqual(drawMatchdayOptions(options,[],[],43),drawn);
});

test('draws always honour one-sided, two-sided and fully fixed side locks', () => {
  const selected = awardFixture().players.slice(0,10);
  const state = createBalanceState(selected);
  for (const [a,b] of [[[1],[]],[[],[1]],[[1,2],[6]],[[1,2,3,4,5],[6,7,8,9,10]]]) {
    const options = selectBalanceTeams(state,selected,a,b);
    for (let seed = 0; seed < 20; seed++) {
      for (const option of drawMatchdayOptions(options,a,b,seed)) {
        assert.ok(a.every(id=>option.teamA.some(p=>p.id===id)));
        assert.ok(b.every(id=>option.teamB.some(p=>p.id===id)));
      }
    }
  }
  assert.deepEqual(drawMatchdayOptions([],[],[],42),[]);
});
