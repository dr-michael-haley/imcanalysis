import { composeViewAsync } from './views.js';
import { gridShape, paneFurniture, xml } from './figure.js';
import { legendGroups } from './view-legend.js';
import { yieldToBrowser } from './loading.js';

export async function createFigureExport(state, rect, progress = () => {}) {
  const settings = state.figure, { columns, rows } = gridShape(state.views.length);
  const width = settings.paneWidth, height = Math.round(width * rect.height / rect.width), gap = settings.gap;
  const totalWidth = columns * width + (columns - 1) * gap, totalHeight = rows * height + (rows - 1) * gap;
  if (!height || totalWidth > 16384 || totalHeight > 16384 || totalWidth * totalHeight > 32_000_000) throw new Error('This figure is too large. Reduce pane width or choose a less extreme aspect ratio (maximum 32 million pixels and 16,384 px per side).');
  const ratio = width / rect.width, camera = { x: state.x * ratio, y: state.y * ratio, zoom: state.scale * ratio };
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d'), scratch = document.createElement('canvas'), groups = [];
  const dimensions = { ...state, selected: null, flagSelection: null };
  async function raster(view, component = null) {
    await composeViewAsync({ ...view, scene: scratch }, { ...dimensions, component }, () => true);
    ctx.clearRect(0, 0, width, height); ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scratch, camera.x, camera.y, state.width * camera.zoom, state.height * camera.zoom);
    const result = canvas.toDataURL('image/png'); await yieldToBrowser(); return result;
  }
  const image = (id, label, href, blend = 'over') => `<g id="${id}" inkscape:groupmode="layer" inkscape:label="${xml(label)}"${blend === 'add' ? ' style="mix-blend-mode:plus-lighter"' : ''}><image width="${width}" height="${height}" href="${href}"/></g>`;
  try {
    for (const [index, view] of state.views.entries()) {
      progress(`Rendering ${view.name} · ${index + 1} / ${state.views.length}`);
      const prefix = `pane-${index + 1}`, content = [];
      content.push(`<rect width="${width}" height="${height}" fill="${settings.background}"/>`);
      if (settings.format === 'svg') {
        content.push(image(`${prefix}-images`, 'Image channels', await raster(view, 'images')));
        for (const source of state.layers) {
          const layer = view.layers.find(l => l.id === source.id);
          if (source.kind !== 'imc' && layer?.visible && layer.raster) content.push(image(`${prefix}-layer-${source.id}`, source.name, await raster(view, `layer:${source.id}`), layer.blend));
        }
        if (view.overlayVisible && view.overlay.style !== 'off') content.push(image(`${prefix}-cells`, 'Cell overlay', await raster(view, 'cells')));
      } else content.push(image(`${prefix}-tissue`, 'Tissue', await raster(view)));
      content.push(paneFurniture(view, legendGroups(view, state.layers, state), settings, width, height, camera.zoom, prefix));
      groups.push(`<g id="${prefix}" inkscape:groupmode="layer" inkscape:label="${xml(view.name)}" transform="translate(${index % columns * (width + gap)} ${Math.floor(index / columns) * (height + gap)})"><clipPath id="${prefix}-clip"><rect width="${width}" height="${height}"/></clipPath><g clip-path="url(#${prefix}-clip)" style="isolation:isolate">${content.join('')}</g></g>`);
    }
    const metadata = { roi: state.exportROI, umPerPixel: settings.umPerPixel, paneWidthUm: rect.width / state.scale * settings.umPerPixel, paneHeightUm: rect.height / state.scale * settings.umPerPixel, centreX: (rect.width / 2 - state.x) / state.scale, centreY: (rect.height / 2 - state.y) / state.scale };
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="${totalWidth}" height="${totalHeight}" viewBox="0 0 ${totalWidth} ${totalHeight}"><title>${xml(state.exportROI)} · IMC Explorer</title><metadata>${xml(JSON.stringify(metadata))}</metadata><rect width="100%" height="100%" fill="${settings.background}"/>${groups.join('')}</svg>`;
    if (svg.length > 128 * 1024 * 1024) throw new Error('This SVG exceeds 128 MB. Reduce the export dimensions or use PNG.');
    if (settings.format === 'svg') return new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    progress('Encoding image…');
    const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
    try {
      const picture = new Image(); picture.src = url; await picture.decode();
      canvas.width = totalWidth; canvas.height = totalHeight; ctx.drawImage(picture, 0, 0);
      return await new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('The browser could not encode this image. Reduce its dimensions.')), `image/${settings.format}`, .95));
    } finally { URL.revokeObjectURL(url); }
  } finally { canvas.width = scratch.width = 0; canvas.height = scratch.height = 0; }
}
