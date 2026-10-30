import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker/legacy-redirect.mjs';

test('legacy requests redirect once to the fixed IMC origin without losing path or query', () => {
  for (const method of ['GET', 'HEAD']) {
    for (const suffix of ['/', '/assets/example.js', '/a%20b/?next=https%3A%2F%2Fexample.com%2F&view=2']) {
      const response = worker.fetch(new Request(`https://explorer.spatialbiology.app${suffix}`, { method }));
      assert.equal(response.status, 308);
      assert.equal(response.headers.get('Location'), `https://imc.spatialbiology.app${suffix}`);
      assert.equal(response.body, null);
    }
  }
});

test('legacy worker rejects writes and drops a development port from its destination', () => {
  const response = worker.fetch(new Request('https://explorer.spatialbiology.app/', { method: 'POST', body: 'data' }));
  assert.equal(response.status, 405);
  assert.equal(response.headers.get('Allow'), 'GET, HEAD');
  assert.equal(worker.fetch(new Request('http://localhost:8787/?check=1')).headers.get('Location'), 'https://imc.spatialbiology.app/?check=1');
});
