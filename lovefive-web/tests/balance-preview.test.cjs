require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { loadBalancePreview } = require('../lib/balance-preview.ts');

function client(role='admin', code='PGRST202') {
  const reads=[];
  const players=[{id:1,is_active:1},{id:2,is_active:0}];
  return { reads, players,
    rpc:async()=>({data:null,error:{code,message:'RPC unavailable'}}),
    auth:{getUser:async()=>({data:{user:{id:'signed-in-user'}},error:null})},
    from(table){
      reads.push(table);
      const query={select(){return query},eq(){return query},order(){return query},
        maybeSingle:async()=>({data:{role},error:null}),
        range:async(from)=>({data:(table==='players'?players:[{id:8,processed:1}]).slice(from,from+1),error:null})};
      return query;
    }
  };
}
test('missing migration permits only a read-only admin preview, including archived players',async()=>{
  const supabase=client();
  const workspace=await loadBalancePreview(supabase,1);
  assert.equal(workspace.token,null);
  assert.deepEqual(workspace.players,supabase.players);
  assert.equal(workspace.matches.length,1);
});
test('ordinary members cannot use the preview fallback',async()=>{
  const supabase=client('member');
  await assert.rejects(loadBalancePreview(supabase,1),/admin access/);
  assert.deepEqual(supabase.reads,['league_members']);
});
test('permission and connection errors never trigger fallback reads',async()=>{
  const supabase=client('admin','42501');
  await assert.rejects(loadBalancePreview(supabase,1),/RPC unavailable/);
  assert.deepEqual(supabase.reads,[]);
});
