require('./register-typescript.cjs');
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),ts=require('typescript');
require.extensions['.tsx']=(mod,file)=>mod._compile(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText,file);
const React=require('react'),{renderToStaticMarkup}=require('react-dom/server');
const {buildMatchReport}=require('../lib/match-report.ts');
const {MatchReportContent}=require('../components/match-report-content.tsx');

test('full and shared reviews preserve the same three stories, all ten players, scope and safely escaped human notes',()=>{
  const players=Array.from({length:10},(_,i)=>({id:i+1,name:`p${i+1}`,display_name:`Name ${i+1}`,mmr:1000,is_active:1}));
  const match={id:1,date:'2026-01-01',team_a:'p1,p2,p3,p4,p5',team_b:'p6,p7,p8,p9,p10',score:'10-8',processed:1,review_note:'A save <script>alert(1)</script> & a win'};
  const report=buildMatchReport({match,matches:[match],players,mmrHistory:[]});
  const activitySection=React.createElement('section',{id:'match-activity'},'Recorded activity: 145 bpm');
  const full=renderToStaticMarkup(React.createElement(MatchReportContent,{report,activitySection,activityEditable:true}));
  const shared=renderToStaticMarkup(React.createElement(MatchReportContent,{report,activitySection,share:true}));
  for(const html of [full,shared]) {
    for(const story of report.highlights) assert.ok(html.includes(story.title));
    for(const player of players) assert.ok(html.includes(player.display_name));
    assert.ok(html.includes('2026 · Match 1'));
    assert.ok(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));
    assert.ok(!html.includes('<script>'));
    assert.ok(html.includes('League admin note'));
  }
  assert.ok(shared.includes('/app/matches/1'));
  assert.ok(!shared.includes('Player details'));
  assert.ok(full.includes('Add / edit player activity'));
  assert.ok(full.includes('Recorded activity: 145 bpm'));
  assert.ok(!shared.includes('Recorded activity: 145 bpm'));
  assert.ok(!shared.includes('#match-activity'));
});

test('drawn Shield reports and share views award nobody and link back to Shield history',()=>{
  const players=Array.from({length:10},(_,i)=>({id:i+1,name:`p${i+1}`,display_name:`Name ${i+1}`,mmr:1000,is_active:1}));
  const match={id:1,date:'2026-09-22',team_a:'p1,p2,p3,p4,p5',team_b:'p6,p7,p8,p9,p10',score:'7-7',processed:1,competition:'immunity_shield'};
  const report=buildMatchReport({match,matches:[match],players,mmrHistory:[]});
  for(const share of [false,true]) {
    const html=renderToStaticMarkup(React.createElement(MatchReportContent,{report,share}));
    assert.ok(html.includes('nobody wins the cup'));
    assert.ok(html.includes('Draw — no trophy awarded'));
    assert.ok(html.includes('MMR protected'));
    assert.ok(html.includes('competition=immunity_shield'));
    assert.ok(!html.includes('Recorded pre-match public MMR'));
    assert.ok(!html.includes('0 → 1'));
  }
});

test('Shield honours retain archived winners and distinguish missing goals from recorded zero assists',()=>{
  const {ShieldRecords}=require('../components/shield-records.tsx');
  const {buildPlayerSummaries}=require('../lib/demo-data.ts');
  const {attachStyleProfiles}=require('../lib/player-style-profiles.ts');
  const player={id:1,name:'Winner',display_name:null,is_active:0};
  const match={id:1,date:'2026-09-22',team_a:'Winner',team_b:'Opponent',score:'5-4',processed:1,competition:'immunity_shield'};
  const players=attachStyleProfiles([player],[{match_id:1,player_name:'Winner',goals:null,assists:0}],[]);
  const html=renderToStaticMarkup(React.createElement(ShieldRecords,{players:buildPlayerSummaries(players,[match])}));
  assert.ok(html.includes('Winner · Archived'));
  assert.ok(html.includes('<td>—</td><td>0</td>'));
  assert.ok(html.includes('<td>1</td><td>1</td><td>1–0–0</td>'));
});
