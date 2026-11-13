require('./register-typescript.cjs');
const test = require('node:test');
const assert = require('node:assert/strict');
const { requestPasswordReset, verifyPasswordRecovery, saveRecoveredPassword, safeNextPath, RECOVERY_STORAGE_KEY, RESET_EMAIL_MESSAGE } = require('../lib/password-recovery.ts');

const user = { id: 'test-user', email: 'player@example.com' };
const recoveryUrl = 'https://lovefive.co.uk/reset-password#type=recovery&access_token=test-access&refresh_token=test-refresh';
const storage = () => {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key,value) => values.set(key,value), removeItem: key => values.delete(key) };
};
function client(overrides = {}) {
  return { auth: {
    initialize: async () => ({ error: null }),
    getSession: async () => ({ data: { session: { user } }, error: null }),
    getUser: async () => ({ data: { user }, error: null }),
    updateUser: async () => ({ data: { user }, error: null }),
    ...overrides
  } };
}

test('forgot-password sends a recovery request to the dedicated same-origin page without registering a user', async () => {
  let request;
  const fake = client({ resetPasswordForEmail: async (...args) => { request=args; return {error:null}; } });
  assert.equal(await requestPasswordReset(fake, ' player@example.com ', 'https://lovefive.co.uk'), RESET_EMAIL_MESSAGE);
  assert.deepEqual(request, ['player@example.com', {redirectTo:'https://lovefive.co.uk/reset-password'}]);
  await requestPasswordReset(fake, 'player@example.com', 'http://localhost:3000');
  assert.equal(request[1].redirectTo, 'http://localhost:3000/reset-password');
  await assert.rejects(requestPasswordReset(fake, 'not-an-email', 'https://lovefive.co.uk'), /valid email/);
});

test('reset requests distinguish rate limits and failed delivery from accepted requests', async () => {
  await assert.rejects(requestPasswordReset(client({resetPasswordForEmail:async()=>({error:{status:429}})}),user.email,'https://lovefive.co.uk'), /Wait a few minutes/);
  await assert.rejects(requestPasswordReset(client({resetPasswordForEmail:async()=>({error:{status:500,message:'SMTP rejected'}})}),user.email,'https://lovefive.co.uk'), /Could not send/);
});

test('verified recovery session survives refresh in the same tab and remains tied to its user', async () => {
  const tab = storage();
  assert.equal((await verifyPasswordRecovery(client(),recoveryUrl,tab)).id,user.id);
  assert.equal(tab.getItem(RECOVERY_STORAGE_KEY),user.id);
  assert.equal((await verifyPasswordRecovery(client(),'https://lovefive.co.uk/reset-password',tab)).id,user.id);
  await assert.rejects(verifyPasswordRecovery(client({getSession:async()=>({data:{session:{user:{id:'another-user'}}}})}),'https://lovefive.co.uk/reset-password',tab), /invalid or has expired/);
  assert.equal(tab.getItem(RECOVERY_STORAGE_KEY),null);
});

test('direct visits and expired, reused, incomplete or wrong-type links cannot enable password changes', async () => {
  for (const href of [
    'https://lovefive.co.uk/reset-password',
    'https://lovefive.co.uk/reset-password#error=access_denied&error_code=otp_expired',
    'https://lovefive.co.uk/reset-password?error_code=otp_expired',
    recoveryUrl.replace('&refresh_token=test-refresh',''),
    recoveryUrl.replace('type=recovery','type=signup'),
    'https://lovefive.co.uk/reset-password?code=unsupported-flow'
  ]) await assert.rejects(verifyPasswordRecovery(client(),href,storage()), /invalid or has expired/);
  const tab = storage(); tab.setItem(RECOVERY_STORAGE_KEY,user.id);
  await assert.rejects(verifyPasswordRecovery(client(),'https://lovefive.co.uk/reset-password#error=access_denied',tab), /invalid or has expired/);
  assert.equal(tab.getItem(RECOVERY_STORAGE_KEY),null);
  await assert.rejects(verifyPasswordRecovery(client({initialize:async()=>({error:{message:'expired'}})}),recoveryUrl,storage()), /invalid or has expired/);
  await assert.rejects(verifyPasswordRecovery(client({getSession:async()=>({data:{session:null},error:null})}),recoveryUrl,storage()), /invalid or has expired/);
});

test('browser session data alone cannot validate recovery', async () => {
  await assert.rejects(verifyPasswordRecovery(client({getUser:async()=>({data:{user:null},error:{message:'invalid JWT'}})}),recoveryUrl,storage()), /Could not verify/);
  await assert.rejects(verifyPasswordRecovery(client({getUser:async()=>({data:{user:{id:'other'}}})}),recoveryUrl,storage()), /Could not verify/);
});

test('password update validates confirmation and identity, and sends only the new password', async () => {
  let writes = [];
  const fake = client({updateUser:async value=>{writes.push(value);return {error:null};}});
  await assert.rejects(saveRecoveredPassword(fake,'short','short',user.id), /at least 8/);
  await assert.rejects(saveRecoveredPassword(fake,'new-password','different',user.id), /do not match/);
  assert.equal(writes.length,0);
  await assert.rejects(saveRecoveredPassword(fake,'new-password','new-password','other'), /invalid or has expired/);
  assert.equal(writes.length,0);
  await saveRecoveredPassword(fake,'new-password','new-password',user.id);
  assert.deepEqual(writes,[{password:'new-password'}]);
  await assert.rejects(saveRecoveredPassword(client({updateUser:async()=>({error:{message:'Choose a different password.'}})}),'new-password','new-password',user.id), /different password/);
});

test('login next paths reject external URLs and backslash/control-character redirects', () => {
  for(const value of [null,'https://example.com','//example.com','/\\example.com','/\n/example.com','javascript:alert(1)']) assert.equal(safeNextPath(value),'/app');
  assert.equal(safeNextPath('/app/charts?view=matches#history'),'/app/charts?view=matches#history');
});

test('login exposes forgot-password, reset mode requests email only, and signup remains available', () => {
  const fs=require('node:fs'),ts=require('typescript');
  require.extensions['.tsx']=(mod,file)=>mod._compile(ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText,file);
  const React=require('react'),{renderToStaticMarkup}=require('react-dom/server');
  const {AppRouterContext}=require('next/dist/shared/lib/app-router-context.shared-runtime');
  const {SearchParamsContext}=require('next/dist/shared/lib/hooks-client-context.shared-runtime');
  const {LoginPanel}=require('../components/login-panel.tsx');
  const render=mode=>renderToStaticMarkup(React.createElement(AppRouterContext.Provider,{value:{push(){},refresh(){}}},React.createElement(SearchParamsContext.Provider,{value:new URLSearchParams({mode})},React.createElement(LoginPanel))));
  const login=render('login'),reset=render('reset'),signup=render('signup');
  assert.match(login,/Forgot password\?/);
  assert.match(reset,/Send reset link/);assert.match(reset,/type="email"/);assert.doesNotMatch(reset,/type="password"/);
  assert.match(signup,/Confirm password/);assert.match(signup,/Create account/);
});
