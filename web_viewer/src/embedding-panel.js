import { ReaderClient } from './rpc.js';
import { colourScale, extent } from './core.js';
import { COLOURMAPS } from './colourmaps.js';
import { pointInPolygon } from './flags.js';
import { createColourLimitControls } from './colour-limit-controls.js';

export function createEmbeddingPanel(state, $, actions) {
  let points = [], drag = null, currentData = null, currentKey = '', context = null, contextKey = '', job = 0, worker = null, pendingKey = '', errorKey = '';
  const chart = $('embedding-canvas'), background = document.createElement('canvas');
  const limits = createColourLimitControls($('embedding-colour-limits'), 'embedding', value => { Object.assign(state.embeddingOptions, value); draw(); actions.changed(); });
  const option = (value, label) => { const o = document.createElement('option'); o.value = value; o.textContent = label; return o; };
  function sync() {
    const spec = state.embeddingOptions, header = state.header;
    limits.sync(spec);
    $('embedding-colour').replaceChildren(option('view', 'Selected view'), ...(header?.obsKeys ?? []).map(k => option(`obs:${k}`, `Observation · ${k}`)), ...(header?.markers ?? []).map(k => option(`var:${k}`, `Marker · ${k}`)));
    $('embedding-colour').value = spec.colourBy;
    $('embedding-source').replaceChildren(...(header?.sources ?? ['X']).map(k => option(k, k))); $('embedding-source').value = spec.source;
    $('embedding-colourmap').replaceChildren(...Object.entries(COLOURMAPS).map(([k, v]) => option(k, v[0]))); $('embedding-colourmap').value = spec.colourmap;
    $('embedding-square').checked = spec.square; $('embedding-context').checked = spec.context; $('embedding-limit').value = String(spec.contextLimit);
    $('embedding-source').disabled = !spec.colourBy.startsWith('var:'); $('embedding-limit').disabled = !spec.context;
    $('embedding-update').disabled = !state.ready || !$('embedding').value;
  }
  function keys() { const s = state.embeddingOptions; return [JSON.stringify([state.dataset, $('roi').value, $('embedding').value, s.colourBy, s.source]), JSON.stringify([state.dataset, state.mapping, $('embedding').value, s.colourBy === 'view' ? '' : s.colourBy, s.source, s.contextLimit])]; }
  function stop() { job++; worker?.stop(); worker = null; pendingKey = ''; $('embedding-cancel').hidden = true; }
  async function load(force = false) {
    if (!state.ready || !$('embedding').value || state.review.tab !== 'embedding' || !$('review-panels').open) return;
    const spec = { ...state.embeddingOptions }, [key, globalKey] = keys(), combined = key + (spec.context ? globalKey : '');
    const needCurrent = spec.colourBy !== 'view' && currentKey !== key, needContext = spec.context && contextKey !== globalKey;
    if (!needCurrent && !needContext || !force && (pendingKey === combined || errorKey === combined)) return;
    stop(); const mine = job, dataset = state.dataset; pendingKey = combined; errorKey = ''; worker = new ReaderClient(); const client = worker;
    const rows = state.region.rows, embedding = $('embedding').value, mapping = { ...state.mapping };
    $('embedding-cancel').hidden = false; $('embedding-status').textContent = 'Reading embedding colours and context…';
    try {
      await client.call('open', state.anndata);
      if (needCurrent) {
        // Explicit rows avoid rebuilding the full ROI index in a second worker.
        const result = await client.call('embeddingCurrent', { rows, colourBy: spec.colourBy, source: spec.source });
        if (mine !== job || dataset !== state.dataset) return; currentData = result; currentKey = key;
      }
      if (needContext) {
        const result = await client.callProgress('embeddingContext', p => { if (mine === job) $('embedding-status').textContent = p.detail; }, { embedding, roiKey: mapping.roi, idKey: mapping.id, colourBy: spec.colourBy === 'view' ? '' : spec.colourBy, source: spec.source, limit: spec.contextLimit });
        if (mine !== job || dataset !== state.dataset) return; context = result; contextKey = globalKey;
      }
    } catch (error) { if (mine === job) { errorKey = combined; $('embedding-status').textContent = error.message; } }
    finally { client.stop(); if (mine === job) { worker = null; pendingKey = ''; $('embedding-cancel').hidden = true; draw(); } }
  }
  function draw() {
    const rect = chart.getBoundingClientRect(), ctx = chart.getContext('2d'); if (!rect.width || !rect.height) return;
    chart.width = Math.round(rect.width * devicePixelRatio); chart.height = Math.round(rect.height * devicePixelRatio); ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); points = [];
    if (!state.ready) { $('embedding-colour-limits').hidden = true; drag = null; return; }
    const xy = state.region?.coordinates, s = state.embeddingOptions, [key, globalKey] = keys(), roi = $('roi').value;
    if (!xy?.length) { $('embedding-status').textContent = 'Choose an available embedding.'; return; }
    const extra = s.context && contextKey === globalKey ? context : null, data = s.colourBy === 'view' ? state.region : currentKey === key ? currentData : null;
    const xs = [], ys = [];
    for (const coords of [xy, extra?.coordinates ?? []]) for (let i = 0; i < coords.length; i += 2) { if (Number.isFinite(coords[i]) && Number.isFinite(coords[i + 1])) { xs.push(coords[i]); ys.push(coords[i + 1]); } }
    const [xmin, xmax] = extent(xs), [ymin, ymax] = extent(ys);
    const width = s.square ? Math.min(rect.width, rect.height) : rect.width, height = s.square ? width : rect.height, left = (rect.width - width) / 2, top = (rect.height - height) / 2;
    if (s.square) { ctx.strokeStyle = '#344752'; ctx.lineWidth = .7; ctx.strokeRect(left + .5, top + .5, width - 1, height - 1); }
    const categories = data?.categories || extra?.categories ? [...new Set([...(extra?.categories ?? []), ...(data?.categories ?? [])])] : null;
    const overlay = state.views[state.selectedView].overlay;
    const map = s.colourBy === 'view' ? overlay.colourmap : s.colourmap, reverse = s.colourBy === 'view' && overlay.reverse;
    const scale = data ? s.colourBy === 'view' && (categories || s.vmin == null && s.vmax == null) ? state.colours : colourScale(s.colourBy === 'view' ? state.colours.range ?? [] : [...(data.values ?? []), ...(extra?.values ?? [])], categories, extra?.palette?.length ? extra.palette : data.palette, map, reverse, s) : null;
    $('embedding-colour-limits').hidden = !scale?.range;
    $('embedding-colour-limits').querySelector('.colour-limits-help').textContent = s.colourBy === 'view' ? 'Blank = selected view limit. Overrides affect only this embedding.' : 'Blank = automatic. Values outside the range use the end colours.';
    function point(coords, i, id, region, value, isContext) {
      const xx = coords[2 * i], yy = coords[2 * i + 1]; if (!Number.isFinite(xx) || !Number.isFinite(yy)) return;
      const x = left + 14 + (xx - xmin) / (xmax - xmin) * (width - 28), y = top + height - 14 - (yy - ymin) / (ymax - ymin) * (height - 28);
      const selected = state.selection.has(region, id) || region === roi && state.selected === id;
      ctx.globalAlpha = selected ? 1 : isContext ? .28 : 1;
      ctx.fillStyle = selected ? '#ffffff' : isContext && s.colourBy === 'view' ? '#8ca4b3' : scale ? `rgb(${scale.colour(value)})` : '#8ca4b3';
      ctx.beginPath(); ctx.arc(x, y, selected ? 4 : isContext ? 1.6 : 2.5, 0, 2 * Math.PI); ctx.fill();
      if (selected) { ctx.strokeStyle = '#38d3c7'; ctx.lineWidth = 1; ctx.stroke(); }
      points.push({ x, y, id, roi: region });
    }
    if (extra) extra.ids.forEach((id, i) => { if (extra.rois[i] !== roi) point(extra.coordinates, i, id, extra.rois[i], extra.values[i], true); });
    let currentCount = 0;
    state.region.ids.forEach((id, i) => { if (actions.included(i)) { point(xy, i, id, roi, data?.values[i], false); currentCount++; } });
    ctx.globalAlpha = 1;
    background.width = chart.width; background.height = chart.height; background.getContext('2d').drawImage(chart, 0, 0);
    drawLasso();
    if (!pendingKey && !errorKey) $('embedding-status').textContent = `${currentCount.toLocaleString()} current ROI cells · ${extra ? `${extra.sampled ? 'context sampled at ' : 'context includes all '}${extra.ids.length.toLocaleString()} / ${extra.total.toLocaleString()} dataset cells · faint = other ROIs` : 'current ROI only'} · ${state.selection.size.toLocaleString()} selected across ROIs. Click/lasso uses the quick toolbar; context selection includes plotted points only.`;
    $('embedding-legend').replaceChildren();
    if (scale && (s.colourBy !== 'view' || scale.range)) {
      const legend = $('embedding-legend');
      if (scale.entries.length) for (const [value, colour] of scale.entries.slice(0, 40)) { const chip = document.createElement('span'), swatch = document.createElement('i'); swatch.style.background = `rgb(${colour})`; chip.append(swatch, document.createTextNode(value == null ? 'Missing' : String(value))); legend.append(chip); }
      else { const label = document.createElement('span'), bar = document.createElement('i'); bar.className = 'continuous-legend'; bar.style.background = `linear-gradient(90deg, ${(reverse ? [...COLOURMAPS[map][1]].reverse() : COLOURMAPS[map][1]).join(',')})`; label.append(bar, document.createTextNode(`${(s.colourBy === 'view' ? overlay.colourBy : s.colourBy).slice(4)} · ${scale.range?.map(v => Number(v.toPrecision(4))).join(' – ')}`)); legend.append(label); }
    }
    load();
  }
  function drawLasso() {
    if (drag?.mode !== 'lasso') return;
    const ctx = chart.getContext('2d'); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, chart.width, chart.height); ctx.drawImage(background, 0, 0);
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath();
    drag.points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); ctx.setLineDash([]);
  }
  const position = event => { const r = chart.getBoundingClientRect(); return [event.clientX - r.left, event.clientY - r.top]; };
  chart.onpointerdown = event => { if (event.button !== 0 || !state.ready) return; actions.space('embedding'); drag = { mode: actions.mode(), points: [position(event)] }; chart.setPointerCapture(event.pointerId); };
  chart.onpointermove = event => { if (drag?.mode === 'lasso') { const p = position(event), last = drag.points.at(-1); if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 2 && drag.points.length < 5000) drag.points.push(p); drawLasso(); } };
  chart.onpointerup = event => {
    if (!drag) return; const gesture = drag; drag = null;
    if (gesture.mode === 'lasso' && gesture.points.length > 2) { for (const p of points) if (pointInPolygon(p.x, p.y, gesture.points)) state.selection.add(p.roi, p.id); actions.selected(); }
    else { const [x, y] = position(event); let nearest, distance = 12; for (const p of points) { const d = Math.hypot(p.x - x, p.y - y); if (d <= distance) { nearest = p; distance = d; } }
      if (nearest) { if (gesture.mode === 'click') { state.selection.toggle(nearest.roi, nearest.id); actions.selected(); } else actions.inspect(nearest.roi, nearest.id); }
    } draw();
  };
  chart.onpointercancel = () => { drag = null; draw(); };
  for (const [id, field] of [['embedding-colour', 'colourBy'], ['embedding-source', 'source'], ['embedding-colourmap', 'colourmap'], ['embedding-square', 'square'], ['embedding-context', 'context'], ['embedding-limit', 'contextLimit']]) $(id).onchange = () => {
    state.embeddingOptions[field] = $(id).type === 'checkbox' ? $(id).checked : field === 'contextLimit' ? Number($(id).value) : $(id).value;
    errorKey = ''; stop(); sync(); draw(); actions.changed();
  };
  $('embedding-update').onclick = () => { currentKey = contextKey = ''; errorKey = ''; load(true); };
  $('embedding-cancel').onclick = () => { const [a, b] = keys(); stop(); errorKey = a + (state.embeddingOptions.context ? b : ''); $('embedding-status').textContent = 'Embedding loading cancelled. Press Update to retry.'; };
  return { draw, sync, cancelGesture() { if (drag) { drag = null; draw(); } }, reset() { stop(); drag = null; currentData = context = null; currentKey = contextKey = errorKey = ''; sync(); }, selectVisible() { for (const p of points) state.selection.add(p.roi, p.id); actions.selected(); } };
}
