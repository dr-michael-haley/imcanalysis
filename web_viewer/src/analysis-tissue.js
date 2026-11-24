import { deferredAssetFile } from './asset-files.js';
import { asset } from './qc-data.js';
import { relativeInput } from './analysis-project.js';
import { decision, scoreUncertainty } from './mask-qc.js';
import { COLOURMAPS, colourmap, colourmapGradient } from './colourmaps.js';
import { rgb } from './core.js';

// Manifests written before the shared viewer recorded only the DNA channel.
export function sourceFiles(data, kind) {
  const masks = kind === 'masks';
  const folder = masks ? data.source.maskFolder : data.source.imageFolder;
  const paths = new Set(
    data.source.rois.flatMap(r => (masks ? [r.mask] : (r.images ?? (r.dna ? [r.dna] : []))))
  );
  const files = [...paths].map(path =>
    deferredAssetFile(
      {
        name: path.split('/').at(-1),
        webkitRelativePath: folder + '/' + path
      },
      () => asset(data, kind, path)
    )
  );
  files.workspaceHandle = data.handles[kind];
  return files;
}

export async function matchInputs(data, kind, files) {
  if (!files.length) return;
  const saved = data.handles[kind],
    current = files.workspaceHandle;
  if (saved && current && (await saved.isSameEntry(current))) return;
  const local = data.local[kind];
  if (local && files.every(file => local.get(relativeInput(file)) === file)) return;
  // Compatibility folder inputs have no directory handle. When that is the
  // case, use the saved root name and relative paths as in workspace reconnect.
  const expected = kind === 'masks' ? data.source.maskFolder : data.source.imageFolder;
  if (!(saved && current) && files[0].webkitRelativePath?.split('/')[0] === expected) {
    const paths = new Set(files.map(relativeInput));
    if (data.source.rois.every(r => (kind === 'masks' ? paths.has(r.mask) : !r.dna || paths.has(r.dna))))
      return;
  }
  throw Error(
    'The viewer uses a different ' + kind + ' folder. Choose ' + expected + ' in Setup to use this analysis.'
  );
}

export const qcPalette = {
  manualGood: { name: 'Manual good', colour: '#00ff00' },
  manualArtefact: { name: 'Manual artefact', colour: '#ff0000' },
  predictedGood: { name: 'Predicted good', colour: '#478963' },
  predictedArtefact: { name: 'Predicted artefact', colour: '#aa585f' }
};
export function qcDisplay(settings = {}) {
  const classes = {};
  for (const [key, { colour }] of Object.entries(qcPalette)) {
    const item = settings.classes?.[key];
    classes[key] = { colour: /^#[\da-f]{6}$/i.test(item?.colour) ? item.colour : colour,
      visible: item?.visible !== false, style: item?.style === 'outline' ? 'outline' : 'fill' };
  }
  const { min, max } = settings;
  const valid = Number.isFinite(min) && Number.isFinite(max) && min >= 0 && min < max && max <= 1;
  return { classes, mode: ['uncertainty', 'score'].includes(settings.mode) ? settings.mode : 'labels',
    cmap: Object.hasOwn(COLOURMAPS, settings.cmap) ? settings.cmap : 'viridis',
    style: settings.style === 'outline' ? 'outline' : 'fill', min: valid ? min : 0, max: valid ? max : 1 };
}

export function decisionOverlay(name, rows, labels, scores, threshold, settings) {
  const display = qcDisplay(settings), colours = new Map(), outlines = new Set();
  const objects = new Set(rows.map(row => row.id));
  if (display.mode !== 'labels') {
    const lut = Array.from({ length: 256 }, (_, i) => colourmap(i / 255, display.cmap));
    for (const id of objects) {
      const score = scores?.get(id);
      if (!Number.isFinite(score)) { colours.set(id, null); continue; }
      const value = display.mode === 'uncertainty' ? scoreUncertainty(score) : score;
      const t = Math.max(0, Math.min(1, (value - display.min) / (display.max - display.min)));
      colours.set(id, lut[Math.round(t * 255)]);
      if (display.style === 'outline') outlines.add(id);
    }
    const metric = display.mode === 'uncertainty' ? 'Uncertainty' : 'Artefact score';
    return { name, objects, colours, outlines, legend: [], scale: {
      title: name + ' · ' + metric, range: [display.min, display.max], gradient: colourmapGradient(display.cmap)
    } };
  }
  const tints = Object.fromEntries(Object.entries(display.classes).map(([key, item]) => [key, rgb(item.colour)]));
  for (const id of objects) {
    const { keep, reason } = decision(labels[id], scores?.get(id), threshold);
    if (reason === 'unscored') continue;
    const key = reason === 'confirmed' ? (keep ? 'manualGood' : 'manualArtefact') : (keep ? 'predictedGood' : 'predictedArtefact');
    const item = display.classes[key];
    colours.set(id, item.visible ? tints[key] : null);
    if (item.style === 'outline') outlines.add(id);
  }
  const legend = Object.entries(qcPalette)
    .filter(([key]) => display.classes[key].visible && (scores || key.startsWith('manual')))
    .map(([key, item]) => ({ name: item.name, colour: display.classes[key].colour }));
  return { name, colours, objects, outlines, legend };
}
