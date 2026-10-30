// Static assets only. MCD files never enter this process.
import { createServer } from 'node:http';
import { readFile, stat, realpath } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, relative, extname, isAbsolute } from 'node:path';

const root = await realpath(fileURLToPath(new URL('.', import.meta.url)));
const port = Number(process.env.MCD_VIEWER_PORT || process.argv[2] || 8878);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.txt': 'text/plain; charset=utf-8', '.svg': 'image/svg+xml' };
const allowed = new Set(['/index.html', '/THIRD-PARTY-NOTICES.txt']);
const server = createServer(async (req, res) => {
  try {
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405); res.end(); return; }
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    const targetPath = pathname === '/' ? '/index.html' : pathname;
    if (!allowed.has(targetPath) && !targetPath.startsWith('/src/')) throw new Error('Not found');
    const target = await realpath(resolve(root, `.${targetPath}`));
    const rel = relative(root, target);
    if (rel.startsWith('..') || isAbsolute(rel) || !(await stat(target)).isFile()) throw new Error('Not found');
    const content = await readFile(target);
    res.writeHead(200, { 'Content-Type': types[extname(target)] || 'application/octet-stream', 'Content-Length': content.length, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; worker-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'" });
    res.end(req.method === 'HEAD' ? undefined : content);
  } catch { res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('Not found'); }
});
server.on('error', error => { console.error(`MCD Explorer: ${error.message}`); process.exitCode = 1; });
server.listen(port, '127.0.0.1', () => console.log(`MCD Explorer: http://127.0.0.1:${server.address().port}/\nAll file reading and image processing happen in your browser. Ctrl+C to stop.`));
