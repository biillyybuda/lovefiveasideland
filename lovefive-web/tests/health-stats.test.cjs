require('./register-typescript.cjs');
const test=require('node:test'),assert=require('node:assert/strict');
const {buildHealthStats,healthLeaderboard,healthTrend,loadLeagueActivity,boardMetrics,runningProfiles}=require('../lib/health-stats.ts');
const players=[{id:1,name:'alex',display_name:'Alex',is_active:1},{id:2,name:'ben',display_name:'Ben',is_active:0},{id:3,name:'casey',display_name:'Casey',is_active:1}];
const match=(id,date='2026-01-01',extra={})=>({id,date,team_a:'Alex, Ben',team_b:'Casey',score:'5-3',processed:1,...extra});
const reading=(match_id,player_id=1,values={})=>({league_id:1,match_id,player_id,distance_km:null,duration_minutes:null,average_heart_rate:null,max_heart_rate:null,resting_heart_rate:null,personal_max_heart_rate:null,source:'Test watch',version:'v1',updated_at:'2026-09-11T10:00:00Z',...values});

test('rotating watch coverage and speed rankings use only games with the relevant recorded speed',()=>{
  const data=buildHealthStats(1,players,[match(1),match(2),match(3)],[reading(1,1,{max_speed_kmh:14.7,average_pace_seconds:882,calories:807,average_respiration:37}),
    reading(2,1,{max_speed_kmh:13}),reading(3,1,{resting_heart_rate:60}),reading(1,2,{max_speed_kmh:14.74}),reading(1,3,{max_speed_kmh:0})]);
  assert.equal(data.totalDistance,null);assert.equal(data.players[0].topSpeed,14.7);assert.equal(data.players[0].speedCount,2);
  assert.deepEqual(healthLeaderboard(data.players,'topSpeed',1).map(p=>[p.player.name,p.rank,p.count]),[['Alex',1,2],['Ben',1,1],['Casey',3,1]]);
  assert.deepEqual(healthLeaderboard(data.players,'topSpeed',2).map(p=>p.player.name),['Alex']);
  assert.deepEqual(healthTrend(data.players[0],'calories',0).map(p=>p.value),[807,null,null]);
  assert.deepEqual(healthTrend(data.players[0],'average_pace_seconds',0).map(p=>p.value),[882,null,null]);
  assert.deepEqual(healthTrend(data.players[0],'moving_speed_kmh',0).map(p=>p.value),[null,null,null]);
});

test('health totals distinguish unknown and zero, use metric-specific counts, and weight only paired distance/time readings',()=>{
  const matches=Array.from({length:6},(_,i)=>match(i+1,`2026-01-0${i+1}`));
  const data=buildHealthStats(1,players,matches,[reading(1,1,{distance_km:4,duration_minutes:40,average_heart_rate:140}),
    reading(2,1,{distance_km:2,duration_minutes:10,average_heart_rate:160}),reading(3,1,{distance_km:6}),
    reading(4,1,{duration_minutes:60}),reading(5,1,{distance_km:0}),reading(6,1,{resting_heart_rate:60})]);
  const p=data.players[0];
  assert.equal(p.totalDistance,12);assert.equal(p.distanceCount,4);assert.equal(p.averageDistance,3);
  assert.equal(p.totalMinutes,110);assert.equal(p.pairedCount,2);assert.equal(p.metresPerMinute,120);
  assert.equal(p.averageHeartRate,150);assert.equal(p.heartRateCount,2);
  assert.equal(data.recordCount,6);assert.equal(data.possibleRecords,18);assert.equal(data.playersWithRecords,1);
  assert.equal(data.players[1].totalDistance,null);assert.equal(data.players[1].metresPerMinute,null);
  assert.equal(data.players[0].latestPersonalMax,null);
});

test('health scope follows match dates and IDs, keeps archived aliases, and excludes pending, invalid, duplicate and non-lineup records',()=>{
  const matches=[match(4,'2026-01-04'),match(2,'2026-01-02'),match(3,'2026-01-02'),match(1,'2025-12-31'),match(5,'2026-01-05',{processed:0}),match(6,'2026-02-30'),match(7,'2026-01-07',{score:'unknown'}),match(4,'2026-01-04')];
  const rows=[reading(1,2,{resting_heart_rate:40}),reading(2,2,{resting_heart_rate:62}),reading(3,2,{resting_heart_rate:61,personal_max_heart_rate:190}),
    reading(4,2,{distance_km:3}),reading(5,2,{distance_km:100}),reading(6,2,{distance_km:100}),reading(7,2,{distance_km:100}),
    reading(4,999,{distance_km:100}),reading(4,1,{distance_km:100,league_id:2}),reading(999,1,{distance_km:100})];
  const data=buildHealthStats(1,players,matches.filter(m=>m.date.startsWith('2026')),rows),p=data.players.find(p=>p.player.id===2);
  assert.deepEqual(p.entries.map(e=>e.match.id),[2,3,4]);assert.equal(p.player.is_active,0);assert.equal(data.totalDistance,3);
  assert.deepEqual(p.latestResting,{value:61,date:'2026-01-02',matchId:3});assert.equal(p.latestPersonalMax.value,190);
  assert.equal(data.playersWithRecords,1);assert.equal(data.possibleRecords,9);
});

test('leaderboards use qualifying games for each metric, handle ties at displayed precision, and never rank unknowns as zero',()=>{
  const data=buildHealthStats(1,players,[match(1),match(2),match(3)],[reading(1,1,{distance_km:4.001,duration_minutes:60}),reading(2,1,{resting_heart_rate:60}),reading(3,1,{resting_heart_rate:61}),reading(1,2,{distance_km:4.002,duration_minutes:60}),reading(1,3,{distance_km:0,duration_minutes:60})]);
  assert.deepEqual(healthLeaderboard(data.players,'metresPerMinute',1).map(r=>[r.player.name,r.rank]),[['Alex',1],['Ben',1],['Casey',3]]);
  assert.equal(healthLeaderboard(data.players,'metresPerMinute',3).length,0);
  assert.equal(healthLeaderboard(data.players,'topSpeed',1).length,0);
  assert.deepEqual(boardMetrics.map(m=>m.key),['metresPerMinute','topSpeed']);
  assert.equal(data.players[0].averageDistance,4.001);
});

test('personal trends keep gaps for unrecorded appearances and blank metrics; a baseline never comes from a match peak',()=>{
  const data=buildHealthStats(1,players,[match(1),match(2),match(3)],[reading(1,1,{distance_km:0,max_heart_rate:180}),reading(3,1,{distance_km:4})]);
  const p=data.players[0];
  assert.deepEqual(healthTrend(p,'distance_km',0).map(p=>p.value),[0,null,4]);
  assert.deepEqual(healthTrend(p,'personal_max_heart_rate',0).map(p=>p.value),[null,null,null]);
  assert.deepEqual(healthTrend(p,'distance_km',2).map(p=>p.match.id),[2,3]);
  assert.equal(p.peakHeartRate,180);assert.equal(p.latestPersonalMax,null);
});

test('duplicate and malformed readings cannot inflate health totals or mutate the supplied league data',()=>{
  const rows=[reading(1,1,{distance_km:2,updated_at:'2026-09-10'}),reading(1,1,{distance_km:3}),reading(1,2,{distance_km:NaN,average_heart_rate:0}),reading(1,3,{distance_km:-4,duration_minutes:0})];
  const before=structuredClone(rows),data=buildHealthStats(1,players,[match(1)],rows);
  assert.equal(data.recordCount,1);assert.equal(data.totalDistance,3);assert.deepEqual(rows,before);
});

test('empty periods have unknown distance/time totals, zero coverage, and no leaderboard',()=>{
  const data=buildHealthStats(1,players,[],[reading(1,1,{distance_km:5})]);
  assert.equal(data.totalDistance,null);assert.equal(data.totalMinutes,null);assert.equal(data.recordCount,0);
  assert.equal(data.possibleRecords,0);assert.equal(data.participantCount,0);assert.equal(healthLeaderboard(data.players,'metresPerMinute',1).length,0);
  assert.equal(data.typicalWorkRate,null);assert.equal(data.typicalMovingShare,null);
});

test('moving share uses matching elapsed readings, excludes inconsistent times and keeps real zeroes',()=>{
  const data=buildHealthStats(1,players,[match(1),match(2),match(3),match(4)],[
    reading(1,1,{moving_minutes:30,elapsed_minutes:60,duration_minutes:40}),
    reading(2,1,{moving_minutes:30,elapsed_minutes:30}),
    reading(3,1,{moving_minutes:70,elapsed_minutes:60}),
    reading(4,1,{moving_minutes:10,duration_minutes:20}),
    reading(1,3,{moving_minutes:0,elapsed_minutes:60})]);
  assert.equal(data.players[0].movingCount,2);
  assert.ok(Math.abs(data.players[0].movingShare-200/3)<1e-10);
  assert.equal(data.players[1].movingShare,null);
  assert.equal(data.players[2].movingShare,0);
  assert.ok(Math.abs(data.typicalMovingShare-100/3)<1e-10);
});

test('typical work rate gives each player equal weight despite a different number of watch sessions',()=>{
  const data=buildHealthStats(1,players,[match(1),match(2),match(3)],[
    reading(1,1,{distance_km:3,duration_minutes:60}),reading(2,1,{distance_km:3,duration_minutes:60}),
    reading(3,1,{distance_km:3,duration_minutes:60}),reading(1,2,{distance_km:6,duration_minutes:60}),reading(1,3,{distance_km:9})]);
  assert.equal(data.typicalWorkRate,75);
});

test('one session establishes a baseline and the second tracks change; zero baselines never produce infinity',()=>{
  const matches=[match(1,'2026-01-01'),match(2,'2026-01-02'),match(3,'2026-01-03'),match(4,'2026-01-04')];
  const rows=[reading(1,1,{distance_km:3,duration_minutes:60}),reading(2,1,{distance_km:3.6,duration_minutes:60}),
    reading(3,1,{distance_km:12}),reading(4,1,{distance_km:2.4,duration_minutes:60})];
  const initial=buildHealthStats(1,players,matches,rows.slice(0,1)).players[0];
  assert.equal(initial.metresPerMinute,50);assert.equal(initial.workRateChange,null);
  assert.deepEqual(initial.firstWorkRate,{value:50,date:'2026-01-01',matchId:1});
  assert.deepEqual(buildHealthStats(1,players,matches,rows.slice(0,2)).players[0].workRateChange,{baseline:50,latest:60,percent:20});
  const final=buildHealthStats(1,players,matches,rows).players[0];
  assert.deepEqual(final.workRateChange,{baseline:50,latest:40,percent:-20});
  assert.deepEqual(final.latestWorkRate,{value:40,date:'2026-01-04',matchId:4});
  const zero=buildHealthStats(1,players,matches,[reading(1,1,{distance_km:0,duration_minutes:60}),reading(2,1,{distance_km:3,duration_minutes:60})]);
  assert.deepEqual(zero.players[0].workRateChange,{baseline:0,latest:50,percent:null});
});

test('running profiles sort every measure without mutating players or treating missing readings as zero',()=>{
  const data=buildHealthStats(1,players,[match(1),match(2)],[
    reading(1,1,{distance_km:3,duration_minutes:60,max_speed_kmh:14,moving_minutes:30,elapsed_minutes:60}),
    reading(2,1,{distance_km:3.6,duration_minutes:60}),
    reading(1,2,{max_speed_kmh:16,moving_minutes:45,elapsed_minutes:60}),
    reading(1,3,{distance_km:3,duration_minutes:60,max_speed_kmh:0}),
    reading(2,3,{distance_km:0,duration_minutes:60})]);
  const before=structuredClone(data.players);
  const names=(metric,direction='descending')=>runningProfiles(data.players,{metric,direction}).map(p=>p.name);
  assert.deepEqual(names('name','ascending'),['Alex','Ben','Casey']);
  assert.deepEqual(names('name'),['Casey','Ben','Alex']);
  assert.deepEqual(names('metresPerMinute'),['Alex','Casey','Ben']);
  assert.deepEqual(names('metresPerMinute','ascending'),['Casey','Alex','Ben']);
  assert.deepEqual(names('topSpeed'),['Ben','Alex','Casey']);
  assert.deepEqual(names('topSpeed','ascending'),['Casey','Alex','Ben']);
  assert.deepEqual(names('movingShare'),['Ben','Alex','Casey']);
  assert.deepEqual(names('movingShare','ascending'),['Alex','Ben','Casey']);
  assert.deepEqual(names('change'),['Alex','Casey','Ben']);
  assert.deepEqual(names('change','ascending'),['Casey','Alex','Ben']);
  assert.deepEqual(data.players,before);
});

test('recorded-only trends skip other wearers and apply the reading limit after selecting the metric',()=>{
  const matches=Array.from({length:20},(_,i)=>match(i+1));
  const data=buildHealthStats(1,players,matches,[reading(1,1,{distance_km:3,duration_minutes:60}),reading(8,1,{distance_km:4,duration_minutes:60}),reading(19,1,{resting_heart_rate:60})]);
  assert.deepEqual(healthTrend(data.players[0],'metres_per_minute',12,true).map(p=>p.match.id),[1,8]);
  assert.deepEqual(healthTrend(data.players[0],'metres_per_minute',1,true).map(p=>p.match.id),[8]);
});

test('league activity reads are scoped and paginated even below the server cap, and errors never become empty success',async()=>{
  const rows=[reading(1),reading(2),reading(3)],calls=[];
  const client={from(table){assert.equal(table,'match_player_activity');const q={select(){return q},eq(k,v){assert.equal(k,'league_id');assert.equal(v,1);return q},order(k){calls.push(['order',k]);return q},range(from,to){calls.push(['range',from,to]);return Promise.resolve({data:rows.slice(from,from+2),error:null})}};return q}};
  assert.deepEqual(await loadLeagueActivity(client,1),rows);
  assert.deepEqual(calls.filter(c=>c[0]==='range').map(c=>c[1]),[0,2,3]);
  assert.deepEqual(calls.slice(0,2),[['order','match_id'],['order','player_id']]);
  await assert.rejects(loadLeagueActivity(client,0),/Choose a league/);
  const failing={from(){const q={select(){return q},eq(){return q},order(){return q},range(){return Promise.resolve({data:null,error:{message:'Access denied'}})}};return q}};
  await assert.rejects(loadLeagueActivity(failing,1),/Access denied/);
});
