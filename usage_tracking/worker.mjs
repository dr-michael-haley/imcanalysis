import { usageReport } from './report.mjs';

const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (path === '/health' && request.method === 'GET') return new Response('ok');
    if (path === '/api/v1/usage') return usageReport(request, env.DB);
    const isAction = path === '/actions';
    if (!isAction && path !== '/record') {
      if (['GET', 'HEAD'].includes(request.method) && env.ASSETS) return env.ASSETS.fetch(request);
      return new Response('Not found', { status: 404 });
    }
    const origin = request.headers.get('Origin');
    const origins = JSON.parse(env.TOOL_ORIGINS);
    if (!origin || !Object.hasOwn(origins, origin)) return new Response('Forbidden', { status: 403 });
    const headers = {
      'Access-Control-Allow-Origin': origin, 'Vary': 'Origin', 'Cache-Control': 'no-store',
      'Access-Control-Allow-Methods': 'POST, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type',
      'X-Content-Type-Options': 'nosniff',
    };
    const reply = (status, body = null) => new Response(body, { status, headers });
    if (request.method === 'OPTIONS') return reply(204);
    if (request.method !== 'POST') return reply(405);
    if (request.headers.get('Content-Type')?.split(';')[0].trim() !== 'application/json') return reply(415);
    if (Number(request.headers.get('Content-Length')) > 2048) return reply(413);
    // Bound actual streamed bytes too; Content-Length is not trusted.
    let data;
    try {
      const reader = request.body?.getReader();
      if (!reader) return reply(400);
      let length = 0;
      const chunks = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > 2048) { await reader.cancel(); return reply(413); }
        chunks.push(value);
      }
      const buffer = new Uint8Array(length);
      let offset = 0;
      for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
      data = JSON.parse(new TextDecoder().decode(buffer));
    } catch { return reply(400); }
    if (!data || typeof data !== 'object' || Array.isArray(data)
      || Object.keys(data).some(key => !['tool', isAction ? 'action' : 'name', 'affiliation', 'eventId'].includes(key))
      || data.tool !== origins[origin]
      || !['manchester', 'external'].includes(data.affiliation)
      || typeof data.eventId !== 'string' || !uuid.test(data.eventId)) return reply(400);
    const day = new Date().toISOString().slice(0, 10);
    if (isAction) {
      if (!['dataset_opened', 'figure_exported'].includes(data.action)
        && !(data.tool === 'mcd' && data.action === 'demo_opened')) return reply(400);
      try {
        await env.DB.prepare('INSERT INTO usage_actions (day, tool, affiliation, action, event_id) VALUES (?, ?, ?, ?, ?) ON CONFLICT(event_id) DO NOTHING')
          .bind(day, data.tool, data.affiliation, data.action, data.eventId.toLowerCase()).run();
        return reply(204);
      } catch { return reply(503); }
    }
    if (!(data.name === null || typeof data.name === 'string')) return reply(400);
    const name = data.name === null ? null : data.name.normalize('NFKC').trim().replace(/\s+/g, ' ');
    if (name !== null && (!name || name.length > 120 || /[\u0000-\u001f\u007f]/.test(name))) return reply(400);
    try {
      await env.DB.prepare(`INSERT INTO usage_events (day, tool, affiliation, name, name_key, event_id)
        VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING`)
        .bind(day, data.tool, data.affiliation, name, name?.toLocaleLowerCase('en-GB') ?? null, data.eventId).run();
      return reply(204);
    } catch { return reply(503); } // Never log a request body or a person's name.
  },
};
