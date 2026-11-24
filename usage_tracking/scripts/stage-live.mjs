// Stage only the usage addition over verified deployed assets, preserving other local work.
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const root = fileURLToPath(new URL('../../', import.meta.url));
const stage = resolve(root, 'usage_tracking/deployment');
const apps = [['imc', 'web_viewer', 'SpatialBiologyToolkit/browser_viewer/static'], ['mcd', 'mcd_viewer', 'mcd_viewer/dist'], ['merscope', 'spatialdata_viewer', 'spatialdata_viewer/dist']];
async function files(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) out.push(...await files(join(dir, entry.name)));
    else out.push(join(dir, entry.name));
  }
  return out;
}
for (const [tool, source, built] of apps) {
  const base = `https://${tool}.spatialbiology.app/`;
  const response = await fetch(base);
  assert(response.ok);
  const html = await response.text();
  assert(!html.includes('data-usage-tool'), 'Already integrated; do not re-stage from modified assets.');
  const localHtml = await readFile(join(root, built, 'index.html'), 'utf8');
  assert.equal(localHtml.replace(/\r\n/g, '\n'), html.replace(/\r\n/g, '\n'), `${tool}: live/local build mismatch`);
  for (const file of await files(join(root, built))) {
    const path = relative(join(root, built), file).replaceAll('\\', '/');
    const local = await readFile(file);
    if (path !== '_headers') {
      const remote = await fetch(new URL(path, base));
      assert(remote.ok, `${tool}/${path}: ${remote.status}`);
      assert.deepEqual(Buffer.from(await remote.arrayBuffer()), local, `${tool}/${path}: changed asset`);
    }
    for (const variant of ['original', 'updated']) {
      const dest = join(stage, tool, variant, path);
      await mkdir(join(dest, '..'), { recursive: true });
      await writeFile(dest, local);
    }
  }
  const originalHeaders = await readFile(join(stage, tool, 'original/_headers'), 'utf8');
  const csp = originalHeaders.match(/Content-Security-Policy: (.+)/)?.[1]?.trim();
  assert.equal(response.headers.get('content-security-policy'), csp, `${tool}: live CSP mismatch`);
  const sourceHtml = await readFile(join(root, source, 'index.html'), 'utf8');
  const tag = sourceHtml.match(/  <script defer src="\.\/usage.js"[^\n]+/)[0];
  await writeFile(join(stage, tool, 'updated/index.html'), html.replace('</head>', `${tag}\n</head>`)
    .replace("connect-src 'none'", 'connect-src https://usage.spatialbiology.app'));
  await writeFile(join(stage, tool, 'updated/usage.js'), await readFile(join(root, 'usage_tracking/client/usage.js')));
  await writeFile(join(stage, tool, 'updated/_headers'), originalHeaders
    .replace("connect-src 'none'", 'connect-src https://usage.spatialbiology.app')
    .replace("connect-src 'self';", "connect-src 'self' https://usage.spatialbiology.app;"));
  const config = JSON.parse(await readFile(join(root, source, 'wrangler.jsonc'), 'utf8'));
  for (const variant of ['original', 'updated']) {
    await writeFile(join(stage, tool, `${variant}.json`), JSON.stringify({ ...config, assets: { ...config.assets, directory: `./${variant}` } }, null, 2));
  }
  console.log(`${tool}: all live assets verified; staged usage-only deployment and rollback.`);
}
