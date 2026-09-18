require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { loadLeagueData } = require('../lib/live-data.ts');

function client(fail = false) {
  const filters = [];
  const players = [{id:1,name:'Active',is_active:1},{id:2,name:'Archived',is_active:0}];
  return {filters, from(table) {
    const query = {select(){return query},eq(key,value){filters.push({table,key,value});return query},order(){return query},
      range: async from => fail && table === 'players' ? {data:null,error:{message:'Access denied'}} : {data:table === 'players' ? players.slice(from,from+1) : [], error:null}};
    return query;
  }};
}
test('league data keeps archived players for the model but out of active tables and the player picker',async()=>{
  const supabase=client(), data=await loadLeagueData(supabase,42);
  assert.deepEqual(data.players.map(p=>p.id),[1]);
  assert.deepEqual(data.modelPlayers.map(p=>p.id),[1,2]);
  assert.ok(supabase.filters.some(f=>f.table==='players'&&f.key==='league_id'&&f.value===42));
  assert.ok(!supabase.filters.some(f=>f.table==='players'&&f.key==='is_active'));
});
test('failed model inputs fail visibly instead of silently generating from an empty league',async()=>{
  await assert.rejects(loadLeagueData(client(true),42),/Access denied/);
});
