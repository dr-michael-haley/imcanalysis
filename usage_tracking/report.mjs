const dayMs = 86400000;
const tools = ['imc', 'mcd', 'merscope'];
const dateAt = time => new Date(time).toISOString().slice(0, 10);

function readFilters(params, today) {
  for (const key of params.keys()) {
    if (!['from', 'to', 'tool', 'affiliation'].includes(key) || params.getAll(key).length !== 1) return null;
  }
  const to = params.get('to') ?? today;
  const from = params.get('from') ?? dateAt(Date.parse(today) - 29 * dayMs);
  const tool = params.get('tool') ?? 'all';
  const affiliation = params.get('affiliation') ?? 'all';
  for (const date of [from, to]) {
    if (!/^\d{4}-\d\d-\d\d$/.test(date) || !Number.isFinite(Date.parse(date)) || dateAt(Date.parse(date)) !== date) return null;
  }
  if (from > to || to > today || Date.parse(to) - Date.parse(from) >= 366 * dayMs
    || !['all', ...tools].includes(tool) || !['all', 'manchester', 'external'].includes(affiliation)) return null;
  return { from, to, tool, affiliation };
}

const emptyCounts = () => ({ activity_days: 0, dataset_opened: 0, figure_exported: 0, demo_opened: 0 });

export async function usageReport(request, db) {
  const json = (data, status = 200) => Response.json(data, {
    status, headers: { 'Cache-Control': status === 200 ? 'public, max-age=60' : 'no-store', 'X-Content-Type-Options': 'nosniff' },
  });
  if (request.method !== 'GET') return new Response(null, { status: 405, headers: { Allow: 'GET' } });
  const today = dateAt(Date.now());
  const filters = readFilters(new URL(request.url).searchParams, today);
  if (!filters) return json({ error: 'Use valid UTC dates covering up to 366 days, an app and an affiliation.' }, 400);
  const { from, to, tool, affiliation } = filters;
  const where = ['day BETWEEN ? AND ?'];
  const values = [from, to];
  if (tool !== 'all') { where.push('tool = ?'); values.push(tool); }
  if (affiliation !== 'all') { where.push('affiliation = ?'); values.push(affiliation); }
  const predicate = where.join(' AND ');
  try {
    // Only grouped counts leave SQLite. Never select names or receipt identifiers.
    const [coverage, visits, actions, participation] = await db.batch([
      db.prepare("SELECT start_day FROM usage_coverage WHERE metric = 'actions'"),
      db.prepare(`SELECT day, tool, count(*) AS n FROM usage_events WHERE ${predicate} GROUP BY day, tool`).bind(...values),
      db.prepare(`SELECT day, tool, action, count(*) AS n FROM usage_actions WHERE ${predicate} GROUP BY day, tool, action`).bind(...values),
      db.prepare(`SELECT count(DISTINCT name_key) AS distinct_names,
        count(*) FILTER (WHERE name_key IS NULL) AS anonymous_activity_days FROM usage_events WHERE ${predicate}`).bind(...values),
    ]);
    const actionsFrom = coverage.results[0].start_day;
    const byTool = new Map((tool === 'all' ? tools : [tool]).map(id => [id, emptyCounts()]));
    const daily = new Map();
    for (let time = Date.parse(from); time <= Date.parse(to); time += dayMs) daily.set(dateAt(time), emptyCounts());
    const totals = emptyCounts();
    function add(row, metric) {
      totals[metric] += row.n;
      byTool.get(row.tool)[metric] += row.n;
      daily.get(row.day)[metric] += row.n;
    }
    for (const row of visits.results) add(row, 'activity_days');
    for (const row of actions.results) add(row, row.action);
    function covered(counts, through, demoAvailable) {
      return { ...counts, dataset_opened: through >= actionsFrom ? counts.dataset_opened : null,
        figure_exported: through >= actionsFrom ? counts.figure_exported : null,
        demo_opened: demoAvailable && through >= actionsFrom ? counts.demo_opened : null };
    }
    return json({ filters, actions_from: actionsFrom, participation: participation.results[0],
      totals: covered(totals, to, tool === 'all' || tool === 'mcd'),
      apps: Array.from(byTool, ([tool, counts]) => ({ tool, ...covered(counts, to, tool === 'mcd') })),
      daily: Array.from(daily, ([day, counts]) => ({ day, ...covered(counts, day, tool === 'all' || tool === 'mcd') })) });
  } catch { return json({ error: 'Usage reports are temporarily unavailable.' }, 503); }
}
