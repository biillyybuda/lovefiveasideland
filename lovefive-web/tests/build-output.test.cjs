require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { PHASE_DEVELOPMENT_SERVER, PHASE_PRODUCTION_BUILD, PHASE_PRODUCTION_SERVER } = require('next/constants');
const config = require('../next.config.ts').default;

test('production checks cannot overwrite assets served to open development pages', () => {
  const development = config(PHASE_DEVELOPMENT_SERVER);
  const build = config(PHASE_PRODUCTION_BUILD);
  const production = config(PHASE_PRODUCTION_SERVER);
  assert.notEqual(development.distDir, build.distDir);
  assert.equal(build.distDir, production.distDir);
});
