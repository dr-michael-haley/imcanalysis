import { COLOURMAPS } from './colourmaps.js';
import { qcDisplay, qcPalette } from './analysis-tissue.js';

export function createQCReviewPanel($, { changeDisplay, openROI }) {
  const classes = new Map();
  const element = (tag, text) => {
    const node = document.createElement(tag);
    node.textContent = text;
    return node;
  };
  for (const [key, { name }] of Object.entries(qcPalette)) {
    const label = document.createElement('label'), visible = document.createElement('input');
    visible.type = 'checkbox';
    label.append(visible, name);
    const colour = document.createElement('input');
    colour.type = 'color';
    colour.setAttribute('aria-label', name + ' colour');
    const style = document.createElement('select');
    style.setAttribute('aria-label', name + ' style');
    style.append(new Option('Filled', 'fill'), new Option('Outline', 'outline'));
    classes.set(key, { visible, style, colour });
    $('qc-class-display').append(label, colour, style);
    for (const control of [visible, colour, style]) control.onchange = save;
  }
  $('qc-metric-cmap').replaceChildren(...Object.entries(COLOURMAPS).map(([key, [name]]) => new Option(name, key)));
  for (const id of ['qc-display-mode', 'qc-metric-cmap', 'qc-metric-style', 'qc-metric-min', 'qc-metric-max'])
    $(id).onchange = save;
  function save() {
    const settings = { mode: $('qc-display-mode').value, cmap: $('qc-metric-cmap').value,
      style: $('qc-metric-style').value, min: $('qc-metric-min').valueAsNumber, max: $('qc-metric-max').valueAsNumber,
      classes: Object.fromEntries([...classes].map(([key, c]) =>
        [key, { visible: c.visible.checked, colour: c.colour.value, style: c.style.value }])) };
    changeDisplay(settings);
  }
  function display(s) {
    const settings = qcDisplay(s.run.display);
    const swatches = [];
    for (const [key, controls] of classes) {
      const item = settings.classes[key];
      controls.visible.checked = item.visible;
      controls.colour.value = item.colour;
      controls.style.value = item.style;
      if (!item.visible || settings.mode !== 'labels') continue;
      const swatch = document.createElement('i'), label = document.createElement('span');
      swatch.style.backgroundColor = item.colour;
      label.append(swatch, qcPalette[key].name); swatches.push(label);
    }
    $('qc-colour-key').replaceChildren(...swatches);
    $('qc-display-mode').value = settings.mode;
    $('qc-class-display').hidden = settings.mode !== 'labels';
    $('qc-metric-display').hidden = settings.mode === 'labels';
    $('qc-metric-cmap').value = settings.cmap;
    $('qc-metric-style').value = settings.style;
    $('qc-metric-min').value = settings.min;
    $('qc-metric-max').value = settings.max;
    const description = settings.mode === 'score' ? 'Higher scores favour artefact.' :
      'Uncertainty peaks at an artefact score of 0.5 and falls to zero at 0 and 1.';
    $('qc-metric-help').textContent = s.model ? description + ' Only scored cells are drawn.' : 'Train a classifier to see scores in this overlay.';
  }

  let page = 0, previousRun, session;
  const pageSize = 12;
  const byName = new Intl.Collator(undefined, { numeric: true });
  function list(s) {
    session = s;
    if (s.descriptor.id !== previousRun) {
      previousRun = s.descriptor.id; page = 0; $('qc-roi-search').value = '';
    }
    const rows = s.data.source.rois.map((roi, index) => {
      let good = 0, artefact = 0;
      for (const label of Object.values(s.run.labels[index] ?? {})) {
        if (label === 'good') good++;
        else if (label === 'artefact') artefact++;
      }
      return { roi, index, good, artefact, labels: good + artefact, stats: s.review[index] };
    });
    const labelled = rows.filter(r => r.labels).length;
    const eligible = rows.filter(r => r.roi.dna);
    const scored = eligible.filter(r => r.stats).length;
    $('qc-roi-coverage').textContent = labelled + ' of ' + rows.length + ' ROIs labelled' +
      (s.model ? ' · ' + scored + ' of ' + eligible.length + ' ROIs with DNA scored' : '');
    $('qc-score-rois').disabled = Boolean(s.job) || !s.model;
    $('qc-roi-note').textContent = !s.model ? 'Train a classifier to compare predictions across ROIs.' :
      (s.run.generation !== s.run.model.generation ? 'Labels changed since training. ' : '') +
      'Predictions and mean uncertainty cover unlabelled, scored cells. Higher uncertainty means scores closer to 0.5.';
    const order = $('qc-roi-sort').value, filter = $('qc-roi-filter').value, query = $('qc-roi-search').value.trim().toLowerCase();
    const filtered = rows.filter(r => r.roi.name.toLowerCase().includes(query) &&
      (filter === 'all' || (filter === 'labelled' ? r.labels > 0 : r.labels === 0)));
    filtered.sort((a, b) => {
      if (order === 'name') return byName.compare(a.roi.name, b.roi.name);
      const missing = Number(!a.roi.dna) - Number(!b.roi.dna);
      if (missing) return missing;
      if (order === 'uncertainty' || order === 'artefact') {
        const difference = (b.stats?.[order] ?? -1) - (a.stats?.[order] ?? -1);
        if (difference) return difference;
      }
      return a.labels - b.labels || byName.compare(a.roi.name, b.roi.name);
    });
    const candidates = filtered.filter(r => r.roi.dna);
    const current = s.view ? candidates.findIndex(r => r.index === s.roi) : -1;
    const next = candidates.length ? candidates[(current + 1) % candidates.length] : null;
    $('qc-next-review-roi').disabled = Boolean(s.job) || !next || (s.view && next.index === s.roi);
    $('qc-next-review-roi').title = next ? 'Open ' + next.roi.name : '';
    $('qc-next-review-roi').onclick = () => { if (next) openROI(next.roi.name); };
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    page = Math.min(page, pages - 1);
    const body = document.createDocumentFragment();
    for (const r of filtered.slice(page * pageSize, (page + 1) * pageSize)) {
      const row = document.createElement('tr'), th = document.createElement('th'), link = element('button', r.roi.name);
      th.scope = 'row'; link.disabled = Boolean(s.job);
      row.setAttribute('aria-current', String(s.roi === r.index && Boolean(s.view)));
      link.onclick = () => openROI(r.roi.name); th.append(link);
      row.append(th, element('td', r.good), element('td', r.artefact));
      if (!r.roi.dna || !r.stats) {
        const cell = element('td', r.roi.dna ? 'Not scored' : 'DNA missing');
        cell.colSpan = 3; row.append(cell);
      } else row.append(element('td', r.stats.good), element('td', r.stats.artefact),
        element('td', r.stats.uncertainty == null ? '—' : r.stats.uncertainty.toFixed(3)));
      body.append(row);
    }
    if (!filtered.length) {
      const row = document.createElement('tr'), cell = element('td', 'No matching ROIs');
      cell.colSpan = 6; row.append(cell); body.append(row);
    }
    $('qc-roi-rows').replaceChildren(body);
    $('qc-roi-pages').hidden = pages === 1;
    $('qc-roi-page').textContent = (page + 1) + ' / ' + pages;
    $('qc-roi-prev').disabled = Boolean(s.job) || page === 0;
    $('qc-roi-next').disabled = Boolean(s.job) || page === pages - 1;
  }
  for (const id of ['qc-roi-sort', 'qc-roi-filter', 'qc-roi-search'])
    $(id).oninput = () => { page = 0; if (session) list(session); };
  $('qc-roi-prev').onclick = () => { page--; list(session); };
  $('qc-roi-next').onclick = () => { page++; list(session); };
  return { display, list };
}
