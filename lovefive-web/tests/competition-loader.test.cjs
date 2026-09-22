require('./register-typescript.cjs');
const test=require('node:test'),assert=require('node:assert/strict');
const {loadCompletedMatches}=require('../lib/match-loader.ts');

function client(error) {
  const queries=[];
  return {queries,from(table){
    const entry={table,fields:'',filters:[]};queries.push(entry);
    const q={select(fields){entry.fields=fields;return q},eq(key,value){entry.filters.push([key,value]);return q},order(){return q},
      async range(from){return error && entry.fields.includes('competition') ? {data:null,error} :
        {data:from===0?[{id:1,processed:1,...(!error?{competition:'immunity_shield'}:{})}]:[],error:null}}};
    return q;
  }};
}
test('match reads retain competition, pagination and league boundaries',async()=>{
  const db=client();
  assert.deepEqual(await loadCompletedMatches(db,42),[{id:1,processed:1,competition:'immunity_shield'}]);
  assert.equal(db.queries.length,2);
  for(const q of db.queries){assert.equal(q.table,'matches');assert.deepEqual(q.filters,[['league_id',42],['processed',1]]);}
});
test('only missing competition schema permits legacy reads during rollout',async()=>{
  for(const code of ['42703','PGRST204']) {
    const db=client({code,message:'matches.competition does not exist'});
    assert.deepEqual(await loadCompletedMatches(db,42),[{id:1,processed:1}]);
    assert.equal(db.queries.length,3);
  }
  for(const error of [{code:'42501',message:'Permission denied: competition'},
    {code:'42703',message:'matches.review_note does not exist'},
    {code:'NETWORK',message:'competition request failed'}]) {
    const db=client(error);
    await assert.rejects(loadCompletedMatches(db,42),{message:error.message});
    assert.equal(db.queries.length,1);
  }
});
