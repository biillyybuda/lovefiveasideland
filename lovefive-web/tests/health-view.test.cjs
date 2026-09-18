require('./register-typescript.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),ts=require('typescript');
require.extensions['.tsx']=(mod,file)=>mod._compile(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText,file);
const React=require('react'),{renderToStaticMarkup:render}=require('react-dom/server');
const {buildHealthStats,healthTrend}=require('../lib/health-stats.ts');
const {LeagueHealthActivity,PlayerHealthActivity,ActivityTrend}=require('../components/health-activity.tsx');
const {InteractiveStatsHub}=require('../components/interactive-stats-hub.tsx');
const player={id:1,name:'Alex',display_name:'Alex',is_active:1,mmr:1000};
const match={id:1,date:'2026-01-01',team_a:'Alex',team_b:'',score:'5-3',processed:1};
const row={league_id:1,match_id:1,player_id:1,distance_km:0,duration_minutes:60,average_heart_rate:145,max_heart_rate:180,resting_heart_rate:60,personal_max_heart_rate:190,source:'<script>example</script>',updated_at:'2026-01-02'};
const health={data:buildHealthStats(1,[player],[match],[row]),status:'ready',reload(){},canEdit:true,seasonLabel:'2026',reportBasePath:'/app/matches'};

test('health group accepts one baseline without recording targets; personal view keeps safe match links and distinct heart readings',()=>{
  const group=render(React.createElement(LeagueHealthActivity,{health,onPlayer(){}}));
  assert.match(group,/Activity leaderboards/);assert.match(group,/Baseline set/);assert.match(group,/0.0/);assert.match(group,/\/app\/matches\/1#match-activity/);
  assert.match(group,/Median distance\/min/);assert.doesNotMatch(group,/Next watch wearer/);assert.match(group,/Running profiles/);
  assert.match(group,/<button type="button" aria-pressed="true">Distance\/min<\/button>/);
  assert.match(group,/<button type="button" aria-pressed="false">Top speed<\/button>/);
  assert.ok(group.indexOf('id="health-profiles-title"') < group.indexOf('id="activity-board-title"'));
  assert.match(group,/aria-sort="descending"/);assert.match(group,/Sort by Distance\/min, ascending/);
  assert.match(group,/About Time moving/);assert.match(group,/Open Alex&#x27;s running profile/);
  assert.doesNotMatch(group,/<select|snapshot-card|Moving share|Changes tracked from the next recording/);
  assert.doesNotMatch(group,/Early sample|Minimum readings|Minimum recorded games|recorded of|Recording coverage|Games covered|Total distance recorded|Distance per recorded game|Total \/ playing time logged|Still to record/);
  assert.doesNotMatch(group,/145|190|VO₂|VO2/);
  const personal=render(React.createElement(PlayerHealthActivity,{health,playerId:1}));
  assert.match(personal,/Latest resting HR/);assert.match(personal,/Personal maximum HR/);assert.match(personal,/Highest match peak/);
  assert.match(personal,/01\/01\/2026/);assert.match(personal,/&lt;script&gt;example&lt;\/script&gt;/);assert.doesNotMatch(personal,/<script>/);
  assert.match(personal,/0.00/);assert.match(personal,/190/);
  assert.doesNotMatch(personal,/Distance per game|Distance recorded|Total \/ playing time logged|with readings out of/);
});

test('the second recording shows the change from baseline and personal history excludes other watch wearers',()=>{
  const matches=[match,{...match,id:2,date:'2026-01-02'},{...match,id:3,date:'2026-01-03'}];
  const data=buildHealthStats(1,[player],matches,[{...row,distance_km:3},{...row,match_id:3,distance_km:3.6}]);
  const context={...health,data};
  const group=render(React.createElement(LeagueHealthActivity,{health:context,onPlayer(){}}));
  assert.match(group,/Up 20.0% since first recording/);assert.match(group,/since 01\/01\/2026/);assert.match(group,/Updated 03\/01\/2026/);assert.doesNotMatch(group,/Baseline set|Early sample/);
  const personal=render(React.createElement(PlayerHealthActivity,{health:context,playerId:1}));
  assert.equal((personal.match(/<circle/g)||[]).length,2);
  assert.doesNotMatch(personal,/\/app\/matches\/2#match-activity/);
});

test('profile change labels distinguish a decrease, no change and growth from a zero baseline',()=>{
  const matches=[match,{...match,id:2,date:'2026-01-02'}];
  for (const [first,latest,label] of [[3,2.4,'Down 20.0% since first recording'],[3,3,'No change since first recording'],[0,3,'Up 50.0 m/min since first recording']]) {
    const data=buildHealthStats(1,[player],matches,[{...row,distance_km:first},{...row,match_id:2,distance_km:latest}]);
    const group=render(React.createElement(LeagueHealthActivity,{health:{...health,data},onPlayer(){}}));
    assert.ok(group.includes(label),label);assert.doesNotMatch(group,/NaN|Infinity/);
  }
});

test('loading and failures do not display partial leaderboards or misleading zero statistics',()=>{
  for(const Component of [LeagueHealthActivity,PlayerHealthActivity]) {
    const props={health:{...health,status:'error',message:'Please retry'},playerId:1,onPlayer(){}};
    const error=render(React.createElement(Component,props));assert.match(error,/Activity unavailable/);assert.match(error,/Try loading activity again/);assert.doesNotMatch(error,/0.00|Activity leaderboards|190/);
    const loading=render(React.createElement(Component,{...props,health:{...health,status:'loading'}}));assert.match(loading,/Loading activity/);assert.doesNotMatch(loading,/Activity leaderboards|190/);
  }
});

test('single chart readings are visible points; missing games break the line and remain available in the data',()=>{
  const entries=[{match,reading:row},{match:{...match,id:2},reading:null},{match:{...match,id:3},reading:{...row,distance_km:4}}];
  const points=healthTrend({...health.data.players[0],entries},'distance_km',0);
  const chart=render(React.createElement(ActivityTrend,{points,unit:'km',label:'Distance',decimals:2}));
  assert.equal((chart.match(/<circle/g)||[]).length,2);assert.doesNotMatch(chart,/<polyline/);
  const single=render(React.createElement(ActivityTrend,{points:points.slice(0,1),unit:'km',label:'Distance',decimals:2}));
  assert.match(single,/<circle/);assert.doesNotMatch(single,/NaN|Infinity/);
});

test('stats hub exposes health only with member data and supports a direct personal health view',()=>{
  const props={players:[player],matches:[match],mmrHistory:[],initialView:'health'};
  const demo=render(React.createElement(InteractiveStatsHub,props));assert.doesNotMatch(demo,/Health &amp; Activity|Activity leaderboards|Latest resting HR/);
  const live=render(React.createElement(InteractiveStatsHub,{...props,initialView:'player',initialPlayer:'1',initialPlayerSection:'health',initialSeason:'2026',reportBasePath:'/app/matches',health:{leagueId:1,players:[player],load:{status:'ready',records:[row]},reload(){},canEdit:true}}));
  assert.match(live,/Latest resting HR/);assert.match(live,/Alex · Health &amp; Activity/);assert.match(live,/2026 · Recorded activity/);
});

test('watch details and pace charts show supplied readings with exact time formatting and source context',()=>{
  const recorded={...row,average_pace_seconds:882,best_pace_seconds:244,moving_minutes:2834/60,elapsed_minutes:3495/60,max_speed_kmh:14.7,calories:807,average_respiration:37};
  const data=buildHealthStats(1,[player],[match],[recorded]);
  const personal=render(React.createElement(PlayerHealthActivity,{health:{...health,data},playerId:1}));
  for(const text of ['Latest watch detail','14:42','4:04','47:14','58:15','14.7','807','Average breathing rate','watch estimate']) assert.ok(personal.includes(text),text);
  const points=healthTrend(data.players[0],'average_pace_seconds');
  const chart=render(React.createElement(ActivityTrend,{points,unit:'m:ss/km',label:'Pace',decimals:0}));
  assert.match(chart,/14:42/);assert.match(chart,/m:ss\/km/);assert.doesNotMatch(chart,/882 m:ss/);
});
