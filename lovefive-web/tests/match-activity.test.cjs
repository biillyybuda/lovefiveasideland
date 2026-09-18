require('./register-typescript.cjs');
const test=require('node:test'),assert=require('node:assert/strict');
const {activityDraft,parseActivity,activityPlayers,loadMatchActivity,savePlayerActivity,activityFields,watchActivityFields,activitySelect,formatActivityValue}=require('../lib/match-activity.ts');
const emptyWatch=Object.fromEntries(watchActivityFields.map(f=>[f.key,null]));
const values={...emptyWatch,distance_km:5.2,duration_minutes:60,average_heart_rate:145,max_heart_rate:180,resting_heart_rate:60,personal_max_heart_rate:190,source:'Garmin'};
const saved={...values,league_id:1,match_id:129,player_id:2,version:'version-1'};

test('activity accepts partial readings, keeps blank fields unknown, and preserves a recorded zero distance',()=>{
  assert.deepEqual(parseActivity({...activityDraft(),distance_km:'0'}),{...emptyWatch,distance_km:0,duration_minutes:null,average_heart_rate:null,max_heart_rate:null,resting_heart_rate:null,personal_max_heart_rate:null,source:''});
  assert.deepEqual(parseActivity(activityDraft(values)),values);
  assert.equal(parseActivity({...activityDraft(),max_heart_rate:'180',source:'  Apple Watch  '}).source,'Apple Watch');
  assert.throws(()=>parseActivity({...activityDraft(),source:'Garmin'}),/at least one/);
});
test('activity validation rejects impossible input and reversed heart-rate figures before a write',()=>{
  for(const extra of [{distance_km:'-1'},{distance_km:'NaN'},{distance_km:'Infinity'},{distance_km:'101'},{duration_minutes:'0'},
    {duration_minutes:'601'},{average_heart_rate:'1.5'},{max_heart_rate:'301'},{average_heart_rate:'180',max_heart_rate:'140'},
    {resting_heart_rate:'0'},{resting_heart_rate:'61.5'},{personal_max_heart_rate:'301'},
    {resting_heart_rate:'190'},{resting_heart_rate:'195'},{source:'x'.repeat(81)}]) assert.throws(()=>parseActivity({...activityDraft(values),...extra}));
});
test('baselines can be gathered alone, explicitly cleared, and never inferred from a match peak or another player',()=>{
  const restingOnly=parseActivity({...activityDraft(),resting_heart_rate:'60'});
  assert.equal(restingOnly.resting_heart_rate,60);
  assert.equal(restingOnly.personal_max_heart_rate,null);
  assert.equal(restingOnly.max_heart_rate,null);
  const peakOnly=parseActivity({...activityDraft(),max_heart_rate:'180'});
  assert.equal(peakOnly.resting_heart_rate,null);
  assert.equal(peakOnly.personal_max_heart_rate,null);
  const cleared=parseActivity({...activityDraft(values),resting_heart_rate:'',personal_max_heart_rate:''});
  assert.equal(cleared.resting_heart_rate,null);
  assert.equal(cleared.personal_max_heart_rate,null);
  assert.equal(cleared.max_heart_rate,180);
  const legacy={distance_km:5.2,duration_minutes:60,average_heart_rate:145,max_heart_rate:180,source:'Garmin'};
  assert.equal(activityDraft(legacy).resting_heart_rate,'');
  assert.equal(activityDraft(legacy).personal_max_heart_rate,'');
  assert.equal(activityDraft().resting_heart_rate,'');
});
test('past-match choices use the recorded lineup, support aliases and archived players, and exclude non-participants',()=>{
  const players=[{id:1,name:'tom d',display_name:'Tom',is_active:0},{id:2,name:'glee',display_name:'Glee',is_active:1},{id:3,name:'absent',display_name:'Absent',is_active:1}];
  assert.deepEqual(activityPlayers({team_a:' TOM  D, Tom',team_b:'Glee, unknown'},players),[{id:1,name:'Tom',side:'A'},{id:2,name:'Glee',side:'B'}]);
});
test('activity loads are restricted to the chosen league and previous match',async()=>{
  const calls=[]; const query={select(fields){calls.push(['select',fields]);return query;},eq(k,v){calls.push([k,v]);return query;},order(){return Promise.resolve({data:[saved],error:null});}};
  const result=await loadMatchActivity({from(table){assert.equal(table,'match_player_activity');return query;}},1,129);
  assert.deepEqual(result,[saved]);assert.deepEqual(calls.slice(-2),[['league_id',1],['match_id',129]]);
  assert.ok(calls[0][1].includes('resting_heart_rate,personal_max_heart_rate'));
});
test('add, edit and explicit removal use only the activity RPC with the original version',async()=>{
  const calls=[];const client={rpc:async(name,args)=>{calls.push({name,args});return {data:args.p_remove?null:saved,error:null};}};
  assert.deepEqual(await savePlayerActivity(client,1,129,2,undefined,values),saved);
  await savePlayerActivity(client,1,129,2,saved,values);
  assert.equal(await savePlayerActivity(client,1,129,2,saved,null),null);
  assert.ok(calls.every(c=>c.name==='save_match_player_activity'));
  assert.equal(calls[0].args.p_expected_version,null);assert.equal(calls[1].args.p_expected_version,'version-1');
  assert.deepEqual(calls[0].args.p_values,values);
  assert.equal(calls[2].args.p_remove,true);assert.equal(calls[2].args.p_values,null);
  await assert.rejects(savePlayerActivity(client,1,129,2,saved,{...values,average_heart_rate:200}),/Match peak heart rate/);
  assert.equal(calls.length,3);
});
test('a server that ignores baseline fields cannot silently report a successful save',async()=>{
  const legacy={...saved};delete legacy.resting_heart_rate;delete legacy.personal_max_heart_rate;
  await assert.rejects(savePlayerActivity({rpc:async()=>({data:legacy,error:null})},1,129,2,saved,values),/Could not confirm the saved heart-rate baselines/);
  await assert.rejects(savePlayerActivity({rpc:async()=>({data:{...saved,resting_heart_rate:61},error:null})},1,129,2,saved,values),/Could not confirm the saved heart-rate baselines/);
});
test('failed or conflicting saves retain an actionable error and do not claim success',async()=>{
  await assert.rejects(savePlayerActivity({rpc:async()=>({error:{code:'40001',message:'Conflict'}})},1,129,2,saved,values),/unsaved figures are still here/);
  await assert.rejects(savePlayerActivity({rpc:async()=>({data:null,error:null})},1,129,2,saved,values),/Could not confirm/);
  await assert.rejects(savePlayerActivity({rpc:async()=>({data:{...saved,league_id:2},error:null})},1,129,2,saved,values),/Could not confirm/);
  await assert.rejects(savePlayerActivity({rpc:async()=>({error:{code:'42501',message:'Admin access required'}})},1,129,2,saved,values),/Admin access required/);
});

test('watch screenshots preserve seconds, distinct timing, pace and raw readings without inferring distance',()=>{
  const draft={...activityDraft(),duration_minutes:'58:14',moving_minutes:'47:14',elapsed_minutes:'58:15',
    average_pace_seconds:'14:42',moving_pace_seconds:'12:33',best_pace_seconds:'4:04',
    average_speed_kmh:'4.1',moving_speed_kmh:'4.8',max_speed_kmh:'14.7',calories:'807',
    average_respiration:'37',min_respiration:'21',max_respiration:'50',average_heart_rate:'163',max_heart_rate:'186',source:'Garmin'};
  const result=parseActivity(draft);
  assert.equal(result.duration_minutes,3494/60);assert.equal(result.moving_minutes,2834/60);assert.equal(result.elapsed_minutes,3495/60);
  assert.equal(result.average_pace_seconds,882);assert.equal(result.best_pace_seconds,244);assert.equal(result.max_speed_kmh,14.7);
  assert.equal(result.distance_km,null);assert.equal(result.resting_heart_rate,null);assert.equal(result.personal_max_heart_rate,null);
  assert.equal(result.calories,807);assert.deepEqual(activityDraft(result),draft);
  assert.equal(parseActivity({...activityDraft(),duration_minutes:'1:02:03'}).duration_minutes,3723/60);
  assert.equal(parseActivity({...activityDraft(),duration_minutes:'60.25'}).duration_minutes,60.25);
  assert.equal(formatActivityValue('moving_minutes',result.moving_minutes),'47:14');
  assert.equal(formatActivityValue('average_pace_seconds',result.average_pace_seconds),'14:42');
  assert.equal(formatActivityValue('max_speed_kmh',null),'—');
  assert.deepEqual(new Set(activityFields.map(f=>f.key)),new Set(activitySelect.split(',').filter(k=>!['league_id','match_id','player_id','source','version','updated_at'].includes(k))));
});

test('additional readings validate clock syntax, ranges, ordering, empty values and zeros',()=>{
  for(const extra of [{duration_minutes:'58:74'},{duration_minutes:'1:99:00'},{average_pace_seconds:'1:02:03'},
    {average_pace_seconds:'0:00'},{moving_minutes:'-1'},{elapsed_minutes:'601:00'},{max_speed_kmh:'61'},
    {average_speed_kmh:'15',max_speed_kmh:'14'},{moving_speed_kmh:'15',max_speed_kmh:'14'},
    {calories:'1.5'},{calories:'-1'},{average_respiration:'0'},{min_respiration:'30',max_respiration:'20'},
    {average_respiration:'20',min_respiration:'30'},{average_respiration:'20',max_respiration:'10'}])
    assert.throws(()=>parseActivity({...activityDraft(),...extra}));
  const zero=parseActivity({...activityDraft(),max_speed_kmh:'0',moving_minutes:'0:00',calories:'0'});
  assert.equal(zero.max_speed_kmh,0);assert.equal(zero.moving_minutes,0);assert.equal(zero.calories,0);
  assert.equal(zero.distance_km,null);assert.equal(zero.average_speed_kmh,null);
});

test('a server that drops extra watch data cannot report a successful save',async()=>{
  const watchValues={...values,max_speed_kmh:14.7};
  await assert.rejects(savePlayerActivity({rpc:async()=>({data:saved,error:null})},1,129,2,saved,watchValues),/Could not confirm the saved watch readings/);
  const response={...saved,...watchValues};
  assert.deepEqual(await savePlayerActivity({rpc:async()=>({data:response,error:null})},1,129,2,saved,watchValues),response);
});
