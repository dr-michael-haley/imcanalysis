import test from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import worker from '../worker.mjs';

function setup() {
  const db = new DatabaseSync(':memory:');
  for (const migration of ['0001_usage.sql', '0002_actions.sql']) {
    db.exec(readFileSync(new URL('../migrations/' + migration, import.meta.url), 'utf8'));
  }
  const env = { TOOL_ORIGINS: JSON.stringify({ 'https://imc.spatialbiology.app': 'imc', 'https://mcd.spatialbiology.app': 'mcd' }),
    DB: { prepare(sql) {
      const statement = db.prepare(sql);
      return { all: () => statement.all(), bind: (...args) => ({ run: () => statement.run(...args), all: () => statement.all(...args) }) };
    }, batch: statements => statements.map(statement => ({ results: statement.all() })) } };
  const send = (data = {}, options = {}) => worker.fetch(new Request('https://usage.spatialbiology.app/record', {
    method: 'POST', headers: { Origin: 'https://imc.spatialbiology.app', 'Content-Type': 'application/json', ...options.headers },
    body: JSON.stringify({ tool: 'imc', affiliation: 'manchester', name: 'Jane Smith', eventId: crypto.randomUUID(), ...data }),
  }), env);
  return { db, env, send };
}
test('names deduplicate by day/tool/affiliation; SQL remains data', async () => {
  const { db, send } = setup();
  assert.equal((await send()).status, 204);
  assert.equal((await send({ name: '  JANE   SMITH ' })).status, 204);
  assert.equal(db.prepare('SELECT count(*) n FROM usage_events').get().n, 1);
  assert.equal((await send({ name: "O'Brien; DROP TABLE usage_events;" })).status, 204);
  assert.equal(db.prepare('SELECT count(*) n FROM usage_events').get().n, 2);
});
test('anonymous retries deduplicate but different anonymous events remain distinct', async () => {
  const { db, send } = setup();
  const eventId = crypto.randomUUID();
  await send({ name: null, eventId }); await send({ name: null, eventId }); await send({ name: null });
  const rows = db.prepare('SELECT * FROM usage_events').all();
  assert.equal(rows.length, 2);
  assert(rows.every(row => row.name === null && row.name_key === null));
  assert.equal(db.prepare('SELECT named_users FROM monthly_usage').get().named_users, 0);
});
test('rejects unsupported origin, tool spoofing, unexpected fields, bad names and large bodies', async () => {
  const { db, send } = setup();
  assert.equal((await send({}, { headers: { Origin: 'https://evil.test' } })).status, 403);
  for (const data of [{ tool: 'mcd' }, { affiliation: 'unknown' }, { name: '' }, { name: 4 }, { name: 'x'.repeat(121) }, { name: 'a\u0000b' }, { eventId: 'not-a-uuid' }, { filename: 'private.h5ad' }]) assert.equal((await send(data)).status, 400);
  assert.equal((await send({ name: 'x'.repeat(3000) })).status, 413);
  assert.equal(db.prepare('SELECT count(*) n FROM usage_events').get().n, 0);
});
test('CORS preflight, read denial and database failure', async () => {
  const { env, send } = setup();
  for (const [method, status] of [['OPTIONS', 204], ['GET', 405]]) {
    const response = await worker.fetch(new Request('https://usage.spatialbiology.app/record', { method, headers: { Origin: 'https://imc.spatialbiology.app' } }), env);
    assert.equal(response.status, status);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), 'https://imc.spatialbiology.app');
  }
  env.DB.prepare = () => { throw new Error('offline'); };
  assert.equal((await send()).status, 503);
});

test('action receipts accept only fixed app actions, exclude identities and deduplicate retries', async () => {
  const { db, env } = setup();
  const payload = { tool: 'imc', action: 'dataset_opened', affiliation: 'external', eventId: crypto.randomUUID() };
  const post = (data, origin = 'https://imc.spatialbiology.app') => worker.fetch(new Request('https://usage.spatialbiology.app/actions', {
    method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  }), env);
  assert.equal((await post(payload)).status, 204);
  assert.equal((await post({ ...payload, eventId: payload.eventId.toUpperCase() })).status, 204);
  assert.equal((await post({ ...payload, eventId: crypto.randomUUID() })).status, 204);
  for (const bad of [{ name: null }, { name: 'Private' }, { filename: 'sample.mcd' }, { action: 'demo_opened' }, { tool: 'mcd' }, { action: 'other' }]) {
    assert.equal((await post({ ...payload, ...bad })).status, 400);
  }
  assert.equal((await post({ ...payload, eventId: crypto.randomUUID(), action: 'demo_opened', tool: 'mcd' }, 'https://mcd.spatialbiology.app')).status, 204);
  assert.equal(db.prepare('SELECT count(*) n FROM usage_actions').get().n, 3);
});

test('aggregate API reconciles filters, history and unavailable metrics without returning identifying fields', async () => {
  const { db, env } = setup();
  db.exec(`UPDATE usage_coverage SET start_day='2022-05-02';
    INSERT INTO usage_events VALUES (1,'2022-05-01','imc','external','Private Name','private name','private-receipt');
    INSERT INTO usage_actions VALUES ('receipt-a','2022-05-02','mcd','manchester','demo_opened'),
      ('receipt-b','2022-05-02','imc','external','dataset_opened'),('receipt-c','2022-05-02','imc','external','figure_exported');
    PRAGMA query_only=ON;`);
  const get = (query, method) => worker.fetch(new Request('https://usage.spatialbiology.app/api/v1/usage?' + query, { method }), env);
  const response = await get('from=2022-05-01&to=2022-05-03');
  assert.equal(response.status, 200);
  const text = await response.text();
  assert.doesNotMatch(text, /private|receipt|name_key|event_id/i);
  const report = JSON.parse(text);
  assert.deepEqual(report.participation, { distinct_names: 1, anonymous_activity_days: 0 });
  assert.deepEqual(report.totals, { activity_days: 1, dataset_opened: 1, figure_exported: 1, demo_opened: 1 });
  assert.equal(report.daily.length, 3);
  assert.equal(report.daily[0].dataset_opened, null);
  assert.equal(report.daily[2].dataset_opened, 0);
  assert.equal(report.apps[0].demo_opened, null);
  const filtered = await (await get('from=2022-05-02&to=2022-05-02&tool=mcd&affiliation=external')).json();
  assert.equal(filtered.apps.length, 1);
  assert.deepEqual(filtered.participation, { distinct_names: 0, anonymous_activity_days: 0 });
  assert.equal(filtered.totals.demo_opened, 0);
  const historical = await (await get('from=2022-05-01&to=2022-05-01&tool=imc')).json();
  assert.equal(historical.totals.activity_days, 1);
  assert.equal(historical.totals.figure_exported, null);
  for (const query of ['tool=x', 'name=someone', 'to=2022-02-30', 'from=2022-01-01&to=2024-01-01', 'tool=imc&tool=mcd']) {
    assert.equal((await get(query)).status, 400);
  }
  assert.equal((await get('', 'POST')).status, 405);
  env.DB.batch = () => { throw new Error('private diagnostic'); };
  const unavailable = await get('');
  assert.equal(unavailable.status, 503);
  assert.doesNotMatch(await unavailable.text(), /private diagnostic/);
});
