import { paneFurniture, xml } from './figure.js';

export function exportGeometry(width, rect, camera) {
  const height = Math.round(width * rect.height / rect.width);
  if (!Number.isInteger(width) || width < 128 || width > 8192 || !Number.isFinite(height) || height < 1 || height > 16384 || width * height > 32_000_000 || !(camera.scale > 0)) {
    throw new Error('Export is too large or has invalid dimensions. Use up to 32 million pixels and 16,384 pixels per side.');
  }
  const ratio = width / rect.width;
  return { width, height, x: camera.x * ratio, y: camera.y * ratio, zoom: camera.scale * ratio };
}

export function figureSVG(image, furniture, geometry, settings, metadata) {
  const { width, height } = geometry;
  // Image, title, legend and scale bar remain separately editable SVG layers.
  const layers = furniture.replace(/<g id="(export-[^"]+)">/g, (_, id) => `<g id="${id}" inkscape:groupmode="layer" inkscape:label="${xml(id.replace('export-', '').replaceAll('-', ' '))}">`);
  return `<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><title>${xml(metadata.title)} · MCD Explorer</title><metadata>${xml(JSON.stringify(metadata))}</metadata><g id="background" inkscape:groupmode="layer" inkscape:label="Background"><rect width="100%" height="100%" fill="${settings.background}"/></g><g id="image" inkscape:groupmode="layer" inkscape:label="Image"><image width="${width}" height="${height}" href="${image}"/></g>${layers}</svg>`;
}

export async function createImageExport({ source, rect, camera, settings, view, groups, metadata, calibrated }, current = () => true) {
  const geometry = exportGeometry(settings.paneWidth, rect, camera), { width, height } = geometry;
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  const check = () => { if (!current()) throw new DOMException('Export cancelled because the image changed.', 'AbortError'); };
  check(); ctx.imageSmoothingEnabled = false;
  ctx.drawImage(source, geometry.x, geometry.y, source.width * geometry.zoom, source.height * geometry.zoom);
  const furniture = paneFurniture(view, groups, { ...settings, mode: 'figure' }, width, height, geometry.zoom, 'export');
  const information = { ...metadata, umPerPixel: calibrated ? 1 : null, fieldWidthPixels: width / geometry.zoom, fieldHeightPixels: height / geometry.zoom,
    fieldWidthUm: calibrated ? width / geometry.zoom : null, fieldHeightUm: calibrated ? height / geometry.zoom : null,
    centreX: (width / 2 - geometry.x) / geometry.zoom, centreY: (height / 2 - geometry.y) / geometry.zoom };
  try {
    if (settings.format === 'svg') {
      const svg = figureSVG(canvas.toDataURL('image/png'), furniture, geometry, settings, information);
      if (svg.length > 128 * 1024 * 1024) throw new Error('SVG exceeds 128 MB. Reduce the export width.');
      check(); return new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    }
    // The same vector overlay is rasterised for PNG/JPEG, so its geometry matches SVG and preview.
    const overlay = new Blob([`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${furniture}</svg>`], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(overlay);
    try {
      const image = new Image(); image.src = url; await image.decode(); check();
      ctx.globalCompositeOperation = 'destination-over'; ctx.fillStyle = settings.background; ctx.fillRect(0, 0, width, height);
      ctx.globalCompositeOperation = 'source-over'; ctx.drawImage(image, 0, 0);
      const blob = await new Promise((resolve, reject) => canvas.toBlob(value => value ? resolve(value) : reject(new Error('Image encoding failed. Try a smaller export width.')), `image/${settings.format}`, .95));
      check(); return blob;
    } finally { URL.revokeObjectURL(url); }
  } finally { canvas.width = canvas.height = 0; }
}
