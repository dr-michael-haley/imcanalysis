import { validateColourLimits } from './colour-limits.js';
import { COLOURMAPS } from './colourmaps.js';
import { validateLayerSettings } from './aligned-layers.js';
import { parseNormalization } from './normalization.js';
import { validateFigure, validateScaleBar } from './figure.js';
import { validatePopulations } from './populations.js';

export const PRESET_KEYS = ['F1', 'F2', 'F3', 'F4', 'F6', 'F7', 'F8', 'F9', 'F10'];
const invalid = field => { throw new Error(`Invalid saved ${field}.`); };
const text = (v, field, max = 16384, required = false) => typeof v === 'string' && v.length <= max && (!required || v.trim()) ? v : invalid(field);
const num = (v, field, min = -Infinity, max = Infinity) => Number.isFinite(v) && v >= min && v <= max ? v : invalid(field);
const bool = (v, field) => typeof v === 'boolean' ? v : invalid(field);
export function validatePane(v) {
  if (!v || !Array.isArray(v.channels) || v.channels.length !== 6 || !v.overlay) invalid('view preset');
  const o = v.overlay;
  if (!['outline', 'fill', 'off'].includes(o.style) || !Object.hasOwn(COLOURMAPS, o.colourmap ?? 'legacy')) invalid('overlay');
  if (o.population) { let p; try { p = JSON.parse(o.population); } catch { invalid('population'); } if (p !== null && !['string', 'number', 'boolean'].includes(typeof p)) invalid('population'); }
  return {
    name: text(v.name, 'view name', 80, true), imagesVisible: bool(v.imagesVisible, 'image visibility'),
    legendVisible: bool(v.legendVisible ?? true, 'legend visibility'), legendScale: num(v.legendScale ?? 1, 'legend size', .7, 2),
    scaleBar: validateScaleBar(v.scaleBar),
    layers: validateLayerSettings(v.layers),
    channels: v.channels.map(c => {
      if (!c || !/^#[0-9a-f]{6}$/i.test(c.colour) || !Number.isSafeInteger(c.plane) || c.plane < 0 || c.high <= c.low) invalid('channel');
      return { name: text(c.name, 'channel'), sourceId: text(c.sourceId ?? '', 'channel source'), path: '', showInLegend: bool(c.showInLegend ?? true, 'channel legend visibility'), active: bool(c.active, 'channel active'), enabled: bool(c.enabled, 'channel enabled'), auto: bool(c.auto, 'auto contrast'), colour: c.colour, plane: c.plane, low: num(c.low, 'contrast'), high: num(c.high, 'contrast') };
    }),
    overlay: { ...validateColourLimits(o), showInLegend: bool(o.showInLegend ?? true, 'overlay legend visibility'), visible: bool(o.visible, 'overlay visibility'), colourBy: text(o.colourBy, 'overlay feature'), source: text(o.source, 'expression source'), populations: validatePopulations(o), population: validatePopulations(o).length === 1 ? validatePopulations(o)[0] : '', style: o.style, opacity: num(o.opacity, 'opacity', 0, 1), colourmap: o.colourmap ?? 'legacy', reverse: bool(o.reverse ?? false, 'reverse colourmap') },
  };
}
export const defaultEmbedding = () => ({ colourBy: 'view', source: 'X', colourmap: 'viridis', square: false, context: false, contextLimit: 50000, vmin: null, vmax: null });
export function validateEmbedding(v = defaultEmbedding()) {
  if (!v || typeof v !== 'object' || Array.isArray(v)) invalid('embedding settings');
  const s = { ...defaultEmbedding(), ...v };
  if (!Object.hasOwn(COLOURMAPS, s.colourmap) || !Number.isSafeInteger(s.contextLimit)) invalid('embedding settings');
  if (s.colourBy !== 'view' && !/^(obs|var):.+/s.test(s.colourBy)) invalid('embedding feature');
  if (![10000, 50000, 100000, 200000].includes(s.contextLimit)) invalid('context sample size');
  return { ...validateColourLimits(s), colourBy: text(s.colourBy, 'embedding feature'), source: text(s.source, 'embedding expression source'), colourmap: s.colourmap, square: bool(s.square, 'square plot'), context: bool(s.context, 'embedding context'), contextLimit: num(s.contextLimit, 'context sample size', 1000, 200000) };
}
export function validateSnapshot(v) {
  if (!v || !Array.isArray(v.panes) || !v.panes.length || v.panes.length > 4 || !Number.isInteger(v.selectedView) || v.selectedView < 0 || v.selectedView >= v.panes.length || !v.camera || !v.normalization) invalid('bookmark viewer state');
  const c = v.camera, n = v.normalization;
  if (v.selected !== null && (!Number.isInteger(v.selected) || v.selected < 1 || v.selected > 0xffffffff)) invalid('bookmarked cell');
  return { roi: text(v.roi, 'bookmark ROI', 16384, true), panes: v.panes.map(validatePane), selectedView: v.selectedView, selected: v.selected,
    ...(v.mapping ? { mapping: { roi: text(v.mapping.roi, 'bookmark ROI column', 16384, true), id: text(v.mapping.id, 'bookmark ID column', 16384, true) } } : {}),
    embedding: text(v.embedding, 'embedding'), embeddingOptions: validateEmbedding(v.embeddingOptions), figure: validateFigure(v.figure),
    camera: { fitted: bool(c.fitted, 'camera fit'), scale: num(c.scale, 'camera scale', Number.MIN_VALUE), centreX: num(c.centreX, 'camera centre'), centreY: num(c.centreY, 'camera centre') },
    normalization: { dictionary: n.dictionary === null ? null : Object.fromEntries(parseNormalization(JSON.stringify(n.dictionary), 'bookmark.json')), name: text(n.name, 'normalization name'), quantile: num(n.quantile, 'normalization quantile', Number.MIN_VALUE, 1), minimum: num(n.minimum, 'normalization minimum', 0) },
  };
}
export function validateNavigation(bookmarks = [], presets = []) {
  const list = (items, max, label, parse) => {
    if (!Array.isArray(items) || items.length > max) invalid(label);
    const ids = new Set(); return items.map(item => {
      if (!item) invalid(label); const id = text(item.id, `${label} ID`, 100, true);
      if (ids.has(id)) invalid(`duplicate ${label} ID`); ids.add(id);
      return { id, name: text(item.name, `${label} name`, 120, true), ...parse(item) };
    });
  };
  const keys = new Set();
  return {
    bookmarks: list(bookmarks, 500, 'bookmark', item => ({ comment: text(item.comment, 'bookmark comment', 10000), createdAt: text(item.createdAt, 'bookmark date', 80), viewer: validateSnapshot(item.viewer) })),
    savedViews: list(presets, 100, 'view preset', item => {
      const hotkey = text(item.hotkey, 'hotkey', 3);
      if (hotkey && (!PRESET_KEYS.includes(hotkey) || keys.has(hotkey))) invalid('duplicate or unsupported preset hotkey');
      if (hotkey) keys.add(hotkey);
      return { hotkey, settings: validatePane(item.settings) };
    }),
  };
}
export function presetForKey(presets, event) {
  if (event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.repeat || event.target?.closest?.('input, textarea, select, [contenteditable="true"]')) return null;
  return presets.find(p => p.hotkey && p.hotkey === event.key) ?? null;
}
