import { rgb, isBoundary } from './core.js';
import { yieldToBrowser } from './loading.js';
import { layerSettings, prepareLayer, blendLayer } from './aligned-layers.js';
import { populationKeys } from './populations.js';
import { defaultScaleBar } from './figure.js';

export function newView(name, preset = null) {
  return {
    name, channels: preset ? preset.channels.map(channel => ({ ...channel, raster: null })) : [],
    layers: (preset?.layers ?? []).map(layer => ({ ...layer, raster: null })),
    legendVisible: preset?.legendVisible ?? true,
    legendScale: preset?.legendScale ?? 1,
    scaleBar: { ...defaultScaleBar(), ...preset?.scaleBar },
    imagesVisible: preset?.imagesVisible ?? true, overlayVisible: preset?.overlay.visible ?? true,
    overlay: { showInLegend: true, colourBy: '', source: 'X', population: '', style: 'outline', opacity: 0.8, colourmap: 'viridis', reverse: false, vmin: null, vmax: null, ...preset?.overlay },
    scene: document.createElement('canvas'), region: null, colours: null, fingerprint: '',
  };
}
export function viewSettings(view) {
  return { name: view.name, scaleBar: { ...view.scaleBar }, legendVisible: view.legendVisible ?? true, legendScale: view.legendScale ?? 1, layers: (view.layers ?? []).map(layerSettings), channels: view.channels.map(({ active, enabled, showInLegend = true, name, plane, low, high, colour, auto, sourceId }) => ({ active, enabled, showInLegend, name, plane, low, high, colour, auto, ...(sourceId ? { sourceId } : {}) })), imagesVisible: view.imagesVisible, overlay: { ...view.overlay, visible: view.overlayVisible } };
}
export function includedInView(view, index) { const keys = populationKeys(view.overlay); return !keys.length || keys.includes(JSON.stringify(view.region?.values[index])); }
function* renderView(view, { width, height, mask, selected, flagSelection, layers = [], component = null }) {
  const { region, scene, overlay } = view;
  if (!width || !region) { scene.width = 0; scene.height = 0; return; }
  const ctx = scene.getContext('2d'), image = ctx.createImageData(width, height), pixels = image.data;
  const channels = view.imagesVisible && (!component || component === 'images') ? view.channels.filter(channel => channel.active && channel.enabled && channel.raster).map(channel => ({ ...channel, tint: rgb(channel.colour) })) : [];
  const extras = layers.filter(source => !component || component === `layer:${source.id}`).map(source => { const layer = view.layers?.find(item => item.id === source.id); return layer ? prepareLayer(layer, source) : null; }).filter(Boolean);
  const colours = new Map(region.ids.map((id, i) => [id, includedInView(view, i) ? view.colours.colour(region.values[i]) : null]));
  for (let i = 0; i < width * height; i++) {
    if (i % 65536 === 0) yield;
    const offset = i * 4;
    for (const channel of channels) {
      const value = channel.raster.data[i];
      const level = Number.isFinite(value) ? Math.max(0, Math.min(1, (value - channel.low) / (channel.high - channel.low))) : 0;
      for (let c = 0; c < 3; c++) pixels[offset + c] = Math.min(255, pixels[offset + c] + level * channel.tint[c]);
    }
    for (const layer of extras) blendLayer(pixels, offset, layer, i, width, height, Boolean(component));
    if (!component || component === 'images') pixels[offset + 3] = 255;
    if (component && component !== 'cells') continue;
    if (!mask || !mask.data[i]) continue;
    const id = mask.data[i], isSelected = id === selected || flagSelection?.has(id), edge = isBoundary(mask.data, i, width, height);
    if (!view.overlayVisible && !(isSelected && edge)) continue;
    if (!(isSelected && edge) && (overlay.style === 'off' || (overlay.style === 'outline' && !edge))) continue;
    const colour = isSelected && edge ? [255, 255, 255] : colours.has(id) ? colours.get(id) : [105, 115, 125];
    if (!colour) continue;
    const alpha = isSelected && edge ? 1 : colours.has(id) ? overlay.opacity : overlay.opacity * 0.35;
    for (let c = 0; c < 3; c++) pixels[offset + c] = component ? colour[c] : pixels[offset + c] * (1 - alpha) + colour[c] * alpha;
    if (component) pixels[offset + 3] = alpha * 255;
  }
  // Commit only a complete frame; cancelled renders leave the previous scene.
  scene.width = width; scene.height = height; ctx.putImageData(image, 0, 0);
}
export function composeView(view, dimensions) {
  for (const _ of renderView(view, dimensions)) { /* Small interactive redraw. */ }
}
export async function composeViewAsync(view, dimensions, current, pause = yieldToBrowser) {
  const rendering = renderView(view, dimensions);
  let started = performance.now();
  await pause();
  while (current()) {
    if (rendering.next().done) return true;
    if (performance.now() - started >= 12) { await pause(); started = performance.now(); }
  }
  rendering.return();
  return false;
}

export async function recipeFingerprint(view, normalization) {
  const settings = viewSettings(view); delete settings.name;
  delete settings.legendVisible;
  delete settings.legendScale;
  delete settings.scaleBar;
  for (const channel of settings.channels) delete channel.showInLegend;
  delete settings.overlay.showInLegend;
  for (const layer of settings.layers) {
    delete layer.showInLegend; delete layer.flatLabel;
    if (!layer.flatten) { delete layer.flatten; delete layer.flatColour; }
  }
  // Preserve the original fingerprint for all/single-population workspaces.
  if (populationKeys(settings.overlay).length < 2) delete settings.overlay.populations;
  for (const bound of ['vmin', 'vmax']) if (settings.overlay[bound] == null) delete settings.overlay[bound];
  for (const layer of settings.layers) if (layer.outlineWidth === 1) delete layer.outlineWidth;
  if (!settings.layers.length) delete settings.layers; // Preserve visit signatures from older workspaces.
  // Paths, ROI, selection, zoom and pane names do not change the review recipe.
  const recipe = { ...settings, normalization };
  if (view.remapRecipe) recipe.remap = view.remapRecipe;
  if (view.layerRecipe?.length) recipe.alignedLayers = view.layerRecipe;
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(recipe)));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}
