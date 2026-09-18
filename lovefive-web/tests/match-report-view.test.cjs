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
