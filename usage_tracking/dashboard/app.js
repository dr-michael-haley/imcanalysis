const $ = id => document.getElementById(id);
const apps = { imc: 'IMC Explorer', mcd: 'MCD Explorer', merscope: 'MERSCOPE Explorer' };
const columns = {
  activity_days: 'Activity days', dataset_opened: 'Dataset opens',
  figure_exported: 'Figure exports', demo_opened: 'Demo opens',
};
const count = value => value == null ? '—' : value.toLocaleString('en-GB');
const dayMs = 86400000;
const isoDay = value => new Date(value).toISOString().slice(0, 10);
const displayDay = day => new Date(day).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
let report;
let activeRequest;

function setDates() {
  const custom = $('period').value === 'custom';
  document.querySelectorAll('.date').forEach(label => { label.hidden = !custom; });
  const today = isoDay(Date.now());
  $('from').max = $('to').max = today;
  if (!custom) {
    $('to').value = today;
    $('from').value = isoDay(Date.parse(today) - (Number($('period').value) - 1) * dayMs);
  }
  $('to').setCustomValidity('');
}

function fillTable(id, rows, label) {
  $(id).replaceChildren(...rows.map(row => {
    const tr = document.createElement('tr');
    const heading = document.createElement('th');
    heading.scope = 'row';
    heading.textContent = label(row);
    tr.append(heading);
    for (const [metric, name] of Object.entries(columns)) {
      const cell = tr.insertCell();
      cell.dataset.label = name;
      cell.textContent = count(row[metric]);
    }
    return tr;
  }));
}

function svgNode(type, attrs, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', type);
  for (const [name, value] of Object.entries(attrs)) el.setAttribute(name, value);
  if (text !== undefined) el.textContent = text;
  return el;
}

function plot() {
  if (!report || $('report').hidden) return;
  const container = $('chart');
  const metric = $('measure').value;
  const days = report.daily.filter(row => row[metric] != null);
  if (!days.length) {
    const p = document.createElement('p');
    p.className = 'chart-note';
    p.textContent = 'This measure was unavailable during the selected period.';
    container.replaceChildren(p);
    return;
  }
  const width = Math.max(220, container.clientWidth);
  const peak = Math.max(1, ...days.map(row => row[metric]));
  const mid = Math.ceil(peak / 2);
  const max = peak === 1 ? 1 : mid * 2;
  const left = Math.max(28, count(max).length * 8 + 10);
  const right = width - 6;
  const step = (right - left) / days.length;
  const y = value => 128 - 112 * value / max;
  const svg = svgNode('svg', { viewBox: `0 0 ${width} 160`, role: 'img',
    'aria-label': `${columns[metric]} by UTC day, ${displayDay(days[0].day)} through ${displayDay(days.at(-1).day)}. Values are in Daily totals.` });
  for (const tick of new Set([0, mid, max])) {
    svg.append(svgNode('line', { x1: left, x2: right, y1: y(tick), y2: y(tick) }));
    svg.append(svgNode('text', { x: left - 6, y: y(tick) + 4, 'text-anchor': 'end' }, count(tick)));
  }
  days.forEach((row, i) => {
    const bar = svgNode('rect', { x: left + i * step + step * .1, y: y(row[metric]), width: step * .8, height: 128 - y(row[metric]) });
    bar.append(svgNode('title', {}, `${displayDay(row.day)} · ${columns[metric]} ${count(row[metric])}`));
    svg.append(bar);
  });
  svg.append(svgNode('text', { x: left, y: 150 }, displayDay(days[0].day)));
  if (days.length > 1) svg.append(svgNode('text', { x: right, y: 150, 'text-anchor': 'end' }, displayDay(days.at(-1).day)));
  container.replaceChildren(svg);
}

function render(data, apiUrl) {
  report = data;
  const { filters, totals } = report;
  for (const key of ['activity_days', 'dataset_opened', 'figure_exported']) $(key).textContent = count(totals[key]);
  const { distinct_names: names, anonymous_activity_days: anonymousDays } = data.participation;
  $('participation').textContent = `${count(names)} distinct ${names === 1 ? 'name' : 'names'} · ${count(anonymousDays)} anonymous activity ${anonymousDays === 1 ? 'day' : 'days'}`;
  fillTable('apps', report.apps, row => apps[row.tool]);
  fillTable('days', report.daily, row => row.day);
  const scope = [displayDay(filters.from) + ' – ' + displayDay(filters.to) + ' UTC',
    apps[filters.tool] || 'All apps',
    { manchester: 'University of Manchester', external: 'External', all: 'All affiliations' }[filters.affiliation]];
  if (!Object.values(totals).some(value => value > 0)) scope.push('No reported activity in this selection');
  $('status').textContent = scope.join(' · ');
  const actionStart = filters.from > report.actions_from ? filters.from : report.actions_from;
  if (filters.to < actionStart) {
    $('coverage').textContent = `Action reporting began on ${displayDay(report.actions_from)}, after this period.`;
  } else {
    const dates = actionStart === filters.to ? displayDay(actionStart) : `${displayDay(actionStart)} – ${displayDay(filters.to)}`;
    $('coverage').textContent = `Dataset opens and figure exports cover ${dates} UTC.`;
  }
  $('api').href = apiUrl;
  $('report').hidden = false;
  plot();
}

async function loadReport() {
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  const params = new URLSearchParams(['from', 'to', 'tool', 'affiliation'].map(key => [key, $(key).value]));
  const url = '/api/v1/usage?' + params;
  $('report').hidden = $('error').hidden = true;
  $('status').textContent = 'Loading usage totals…';
  try {
    const response = await fetch(url, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(12000)]) });
    if (!response.ok) throw new Error('Report unavailable');
    const data = await response.json();
    if (!controller.signal.aborted) render(data, url);
  } catch {
    if (controller.signal.aborted) return;
    $('status').textContent = '';
    $('error').querySelector('span').textContent = 'Could not load the usage report.';
    $('error').hidden = false;
  }
}

function download(group) {
  const perApp = group === 'apps';
  const rows = [[perApp ? 'App' : 'UTC day', ...Object.values(columns)]];
  for (const row of report[group]) rows.push([perApp ? apps[row.tool] : row.day, ...Object.keys(columns).map(key => row[key] ?? '')]);
  const url = URL.createObjectURL(new Blob([rows.map(row => row.join(',')).join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  const { from, to, tool, affiliation } = report.filters;
  link.download = `explorer-${group}-${tool}-${affiliation}-${from}-${to}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

$('filters').addEventListener('submit', event => {
  event.preventDefault();
  const span = (Date.parse($('to').value) - Date.parse($('from').value)) / dayMs;
  if (span < 0 || span >= 366) {
    $('to').setCustomValidity('Choose an end date within 366 days of the start date.');
    $('to').reportValidity();
    return;
  }
  void loadReport();
});
$('period').addEventListener('change', setDates);
$('filters').addEventListener('input', () => $('to').setCustomValidity(''));
$('retry').addEventListener('click', loadReport);
$('measure').addEventListener('change', plot);
$('app-csv').addEventListener('click', () => download('apps'));
$('daily-csv').addEventListener('click', () => download('daily'));
new ResizeObserver(plot).observe($('chart'));
setDates();
void loadReport();
