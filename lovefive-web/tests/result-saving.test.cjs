require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { saveLeagueChange } = require('../lib/result-rebuild.ts');

const requestId = '90000000-0000-4000-8000-000000000001';
const change = { action: 'add', match: { id: Number.MAX_SAFE_INTEGER, date: '2026-09-11',
  team_a: 'p1,p2,p3,p4,p5', team_b: 'p6,p7,p8,p9,p10', score: '16-11', result: 'A', processed: 1, competition: 'league' } };
const workspace = { token: 'source-token', matches: [], history: [], adjustments: [],
  players: Array.from({ length: 10 }, (_, i) => ({ id: i + 1, name: `p${i + 1}`, mmr: 1000, initial_mmr: 1000 })) };
const summary = { processedMatches: 1, playersUpdated: 10, historyRows: 10 };

function client({ priorError = null, priorData = null, workspaceError = null, commitError = null } = {}) {
  const calls = [];
  return { calls,
    from(table) {
      calls.push({ table });
      assert.equal(table, 'rating_runs', 'never fall back to a legacy table');
      const query = { select() { return query; }, eq() { return query; },
        maybeSingle: async () => ({ data: priorData, error: priorError }) };
      return query;
    },
    async rpc(name, payload) {
      calls.push({ name, payload });
      if (name === 'rating_workspace') return { data: workspaceError ? null : workspace, error: workspaceError };
      assert.equal(name, 'commit_competition_rating_run');
      return { data: commitError ? null : summary, error: commitError };
    }
  };
}

const migrationMessage = /Apply supabase\/migrations\/20260908080902_mmr_v2_foundation\.sql|cup competition database migration/;

for (const code of ['PGRST205', '42P01']) {
  test(`missing rating_runs (${code}) explains setup and stops before a write`, async () => {
    const supabase = client({ priorError: { code, message: "Could not find the table 'public.rating_runs' in the schema cache" } });
    await assert.rejects(saveLeagueChange(supabase, 1, change, requestId), migrationMessage);
    assert.deepEqual(supabase.calls, [{ table: 'rating_runs' }]);
  });
}

test('partially installed schema stops before committing when the workspace is missing', async () => {
  const supabase = client({ workspaceError: { code: 'PGRST202', message: 'Could not find the function public.rating_workspace(p_league_id) in the schema cache' } });
  await assert.rejects(saveLeagueChange(supabase, 1, change, requestId), migrationMessage);
  assert.deepEqual(supabase.calls.map(c => c.name || c.table), ['rating_runs', 'rating_workspace']);
});

for (const [code, message] of [
  ['PGRST202', 'Could not find the function public.commit_competition_rating_run in the schema cache'],
  ['42703', 'column "selection_source" of relation "matches" does not exist']
]) {
  test(`missing save dependency (${code}) explains the complete migration`, async () => {
    const supabase = client({ commitError: { code, message } });
    await assert.rejects(saveLeagueChange(supabase, 1, change, requestId), migrationMessage);
    assert.deepEqual(supabase.calls.map(c => c.name || c.table), ['rating_runs', 'rating_workspace', 'commit_competition_rating_run']);
  });
}

test('permission, connection and unrelated missing-object errors retain their cause', async () => {
  for (const error of [
    { code: '42501', message: 'permission denied for table rating_runs' },
    { code: '', message: 'Failed to fetch' },
    { code: 'PGRST205', message: "Could not find the table 'public.unrelated_table' in the schema cache" }
  ]) {
    for (const stage of ['priorError', 'workspaceError', 'commitError']) {
      const supabase = client({ [stage]: error });
      await assert.rejects(saveLeagueChange(supabase, 1, change, requestId), e =>
        e.message.includes(error.message) && !migrationMessage.test(e.message));
      assert.ok(supabase.calls.filter(c => c.name === 'commit_competition_rating_run').length <= 1);
    }
  }
});

test('a migrated project saves the match and calculated history with one atomic commit', async () => {
  const supabase = client();
  assert.deepEqual(await saveLeagueChange(supabase, 1, change, requestId), summary);
  const commits = supabase.calls.filter(c => c.name === 'commit_competition_rating_run');
  assert.equal(commits.length, 1);
  const payload = commits[0].payload;
  assert.equal(payload.p_request_id, requestId);
  assert.equal(payload.p_expected_token, workspace.token);
  assert.deepEqual(payload.p_change, change);
  assert.equal(payload.p_calculated.history.length, 10);
  assert.equal(payload.p_calculated.players.length, 10);
});

test('retrying a saved request verifies it through the RPC without replaying it', async () => {
  const supabase = client({ priorData: { id: requestId } });
  assert.deepEqual(await saveLeagueChange(supabase, 1, change, requestId), summary);
  assert.deepEqual(supabase.calls.map(c => c.name || c.table), ['rating_runs', 'commit_competition_rating_run']);
  assert.equal(supabase.calls[1].payload.p_request_id, requestId);
  assert.deepEqual(supabase.calls[1].payload.p_change, change);
  assert.deepEqual(supabase.calls[1].payload.p_calculated, {});
});

test('a missing commit function on a retry gets the same setup guidance', async () => {
  const supabase = client({ priorData: { id: requestId }, commitError: {
    code: 'PGRST202', message: 'Could not find the function public.commit_competition_rating_run in the schema cache' } });
  await assert.rejects(saveLeagueChange(supabase, 1, change, requestId), migrationMessage);
  assert.deepEqual(supabase.calls.map(c => c.name || c.table), ['rating_runs', 'commit_competition_rating_run']);
});

test('stale calculations still ask for a refresh', async () => {
  const supabase = client({ commitError: { code: '40001', message: 'League changed during calculation' } });
  await assert.rejects(saveLeagueChange(supabase, 1, change, requestId), /The league changed while saving\. Refresh and try again\./);
});
