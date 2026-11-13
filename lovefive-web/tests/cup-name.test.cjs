require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { PGlite } = require('@electric-sql/pglite');
const { getCompetitionLabels, validateCupName } = require('../lib/competition.ts');
const { loadCupNames, saveCupName } = require('../lib/cup-settings.ts');
const { buildMatchReport } = require('../lib/match-report.ts');
const { lineupText } = require('../lib/lineup-editor.ts');
const { drawLineupImage } = require('../lib/lineup-image.ts');
const { drawFormationPoster } = require('../lib/lineup-poster.ts');
const { drawPlayerCardsImage, cupFormLegend } = require('../lib/player-cards-image.ts');
const { awardFixture } = require('./award-fixture.cjs');

test('default, custom names, whitespace and validation', () => {
  assert.equal(getCompetitionLabels().immunity_shield, 'Slimfast Cup');
  assert.equal(getCompetitionLabels('   ').immunity_shield, 'Slimfast Cup');
  assert.equal(validateCupName('  Bill’s  Cup 🏆  '), 'Bill’s Cup 🏆');
  assert.equal(getCompetitionLabels('Tuesday Trophy').league, 'League');
  assert.throws(() => validateCupName(' \t '), /Enter a cup name/);
  assert.throws(() => validateCupName('X'.repeat(81)), /80 characters/);
  assert.throws(() => validateCupName('Cup\u0000'), /control characters/);
  assert.equal(validateCupName('X'.repeat(80)).length, 80);
});

test('saving uses the selected league, server response, and propagates failures', async () => {
  let payload;
  const client = { from(table) { assert.equal(table, 'league_cup_settings'); return {
    upsert(value, options) { payload = value; assert.equal(options.onConflict, 'league_id'); return this; },
    select() { return this; }, single: async () => ({ data: { cup_name: payload.cup_name }, error: null })
  }; } };
  assert.equal(await saveCupName(client, 7, '  Tuesday  Trophy '), 'Tuesday Trophy');
  assert.deepEqual(payload, { league_id: 7, cup_name: 'Tuesday Trophy' });
  await assert.rejects(saveCupName({ from() { throw new Error('must not write'); } }, 7, ' '), /Enter a cup name/);
  const denied = { from() { return { upsert() { return this; }, select() { return this; }, single: async () => ({ error: { message: 'row-level security' } }) }; } };
  await assert.rejects(saveCupName(denied, 7, 'New Cup'), /row-level security/);
  const reader = { from() { return { select() { return this; }, in(_column, ids) { assert.deepEqual(ids, [7,8]); return { data: [{ league_id: 7, cup_name: 'Tuesday Trophy' }, { league_id: 8, cup_name: 'Another Cup' }] }; } }; } };
  assert.deepEqual([...await loadCupNames(reader, [7,8])], [[7,'Tuesday Trophy'],[8,'Another Cup']]);
});

test('all three image renderers and lineup copy use the supplied cup name', () => {
  const { players } = awardFixture();
  const roster = players.slice(0,10);
  const draft = { version: 1, locks: {}, preferences: {}, autoFormations: { blue: true, red: true }, competition: 'immunity_shield', date: '2026-11-13', blue: { formation: '1-1-2-1', ids: roster.slice(0,5).map(p => p.id) }, red: { formation: '1-1-2-1', ids: roster.slice(5).map(p => p.id) } };
  const forecast = { win: 0.45, draw: 0.1, loss: 0.45 };
  const originalPath = global.Path2D;
  global.Path2D = class {};
  try {
    for (const name of ['Slimfast Cup', 'Bill’s Trophy 🏆', 'W'.repeat(80), '<Cup & Friends>']) {
      const text = [];
      const drawn = [];
      const context = new Proxy({ font: '20px Arial', fillText(value, x, y) { text.push(value); drawn.push({value, width: this.measureText(value).width, x, y}); },
        measureText(value) { return { width: Array.from(value).length * Number(/([\d.]+)px/.exec(this.font)?.[1] || 20) * 0.55 }; },
        createLinearGradient() { return { addColorStop() {} }; }
      }, { get(target, key) { return key in target ? target[key] : () => {}; } });
      const canvas = { getContext() { return context; } };
      drawLineupImage(canvas, { competition: 'immunity_shield', cupName: name, date: draft.date, blue: roster.slice(0,5).map(p=>p.name), red: roster.slice(5).map(p=>p.name), forecast });
      assert.ok(text.includes(name.toUpperCase()), 'lineup PNG title');
      assert.ok(drawn.find(d=>d.value===name.toUpperCase()).width <= 966, 'long lineup title fits');
      drawn.length = 0;
      text.length = 0;
      drawFormationPoster(canvas, { cupName: name, draft, players: roster, chemistry: { players: new Map(), pairs: new Map() }, forecast });
      assert.ok(text.includes(name.toUpperCase()), 'formation PNG title');
      assert.ok(drawn.find(d=>d.value===name.toUpperCase()).width <= 1120, 'long formation title fits');
      drawn.length = 0;
      text.length = 0;
      const card = p => ({ name: p.name, mmr: 1000, games: 1, wins: 1, draws: 0, losses: 0, form: [{ result: 'W', shield: true, date: draft.date }] });
      drawPlayerCardsImage(canvas, { cupName: name, competition: 'immunity_shield', date: draft.date, blue: roster.slice(0,5).map(card), red: roster.slice(5).map(card) });
      assert.ok(text.includes(name.toUpperCase()), 'player-card PNG title');
      assert.ok(text.includes(cupFormLegend(name)), 'player-card PNG legend');
      assert.ok(drawn.find(d=>d.value===name.toUpperCase()).width <= 850, 'player-card title fits');
      assert.ok(drawn.find(d=>d.value===cupFormLegend(name)).width <= 1076, 'player-card legend fits');
      assert.ok(lineupText(draft, roster, name).includes(name), 'copied lineup');
      assert.ok(text.every(value => !value.includes('Immunity Shield')));
    }
  } finally { global.Path2D = originalPath; }
});

test('renaming historical cup reports preserves scores, honours, and rating calculations', () => {
  const fixture = awardFixture();
  const match = { ...fixture.matches[0], competition: 'immunity_shield', score: '5-3' };
  const args = { match, matches: [match], players: fixture.players, mmrHistory: [] };
  const before = buildMatchReport(args), after = buildMatchReport({ ...args, cupName: 'Tuesday Trophy' });
  assert.match(after.headline, /win the Tuesday Trophy/);
  assert.equal(after.highlights[0].label, 'Tuesday Trophy');
  assert.equal(after.scoreLabel, before.scoreLabel);
  assert.deepEqual(after.playerNotes.map(p=>[p.name,p.before,p.after,p.delta]), before.playerNotes.map(p=>[p.name,p.before,p.after,p.delta]));
  assert.deepEqual(after.changes.map(c=>[c.name,c.before,c.after]), before.changes.map(c=>[c.name,c.before,c.after]));
  assert.deepEqual(after.rating, before.rating);
  const drawn = { ...match, score: '4-4' };
  const report = buildMatchReport({ ...args, match: drawn, matches: [drawn], cupName: 'Tuesday Trophy' });
  assert.match(report.headline, /Tuesday Trophy draw/);
  assert.equal(report.changes.length, 0);
});

test('real database policies restrict cup settings to active league admins and isolate leagues', async t => {
  const db = await PGlite.create(); t.after(() => db.close());
  const owner='00000000-0000-4000-8000-000000000001', admin='00000000-0000-4000-8000-000000000002', member='00000000-0000-4000-8000-000000000003', inactive='00000000-0000-4000-8000-000000000004';
  await db.exec(`create role anon; create role authenticated; create schema auth;
    create function auth.uid() returns uuid language sql as $$ select nullif(current_setting('request.jwt.claim.sub',true),'')::uuid $$;
    grant usage on schema auth to authenticated,anon;
    create table leagues(id bigint primary key);
    create table league_members(league_id bigint,user_id uuid,role text,status text);
    alter table league_members enable row level security;
    create policy members_self on league_members for select to authenticated using(user_id=auth.uid());
    grant select on league_members to authenticated;
    insert into leagues values(1),(2);
    insert into league_members values (1,'${owner}','owner','active'),(1,'${admin}','admin','active'),(1,'${member}','member','active'),(1,'${inactive}','admin','inactive');`);
  await db.exec(fs.readFileSync(path.resolve(__dirname,'../../supabase/migrations/20261113102203_league_cup_settings.sql'),'utf8'));
  const as = async id => db.exec(`reset role; set role authenticated; select set_config('request.jwt.claim.sub','${id}',false);`);
  const save = (id, name) => db.query('insert into league_cup_settings(league_id,cup_name) values($1,$2) on conflict(league_id) do update set league_id=excluded.league_id,cup_name=excluded.cup_name returning cup_name',[id,name]);
  await as(owner);
  assert.equal((await save(1,'Slimfast Cup')).rows[0].cup_name, 'Slimfast Cup');
  await as(admin);
  assert.equal((await save(1,'Tuesday Trophy')).rows[0].cup_name, 'Tuesday Trophy');
  await assert.rejects(save(2,'Wrong League'), /row-level security/);
  await assert.rejects(db.query('update league_cup_settings set league_id=2 where league_id=1'), /row-level security/);
  for (const name of ['', ' ', 'X'.repeat(81), '\nCup']) await assert.rejects(save(1,name), /cup_name_valid/);
  await as(member);
  assert.equal((await db.query('select cup_name from league_cup_settings')).rows[0].cup_name, 'Tuesday Trophy');
  await assert.rejects(save(1,'Not allowed'), /row-level security/);
  assert.equal((await db.query("update league_cup_settings set cup_name='Not allowed' returning *")).rows.length, 0);
  await as(inactive);
  assert.equal((await db.query('select * from league_cup_settings')).rows.length, 0);
  await assert.rejects(save(1,'Not allowed'), /row-level security/);
  await db.exec('reset role; set role anon;');
  await assert.rejects(db.query('select * from league_cup_settings'), /permission denied/);
  await assert.rejects(save(1,'Not allowed'), /permission denied/);
  await as(owner);
  assert.equal((await db.query('select cup_name from league_cup_settings where league_id=1')).rows[0].cup_name, 'Tuesday Trophy');
});
