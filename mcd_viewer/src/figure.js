import { defaultFigureFilename, validateFigureFilename } from './figure-filename.js';

export const defaultScaleBar = () => ({ visible: true, automatic: true, length: 100, thickness: 3, fontSize: 12, colour: '#ffffff', position: 'bottom-left' });
export const defaultFigure = () => ({ mode: 'viewer', umPerPixel: 1, aspect: 'free', physicalLock: false, physicalWidth: 500, physicalHeight: 500, paneWidth: 1024, gap: 24, background: '#000000', foreground: '#ffffff', fontSize: 28, scaleFontSize: 28, legendSize: 24, legendRows: 24, showNames: true, showLegends: true, showScaleBars: true, overlayBackground: true, autoBookmark: true, format: 'png', filename: defaultFigureFilename() });
const finite = (value, min, max) => Number.isFinite(value) && value >= min && value <= max;
const colour = value => typeof value === 'string' && /^#[a-f0-9]{6}$/i.test(value);
export function validateScaleBar(value) {
  const s = { ...defaultScaleBar(), ...value };
  if (!['visible', 'automatic'].every(k => typeof s[k] === 'boolean') || !finite(s.length, .001, 1e7) || !finite(s.thickness, 1, 20) || !finite(s.fontSize, 8, 48) || !colour(s.colour) || !['bottom-left', 'bottom-right'].includes(s.position)) throw new Error('Invalid scale bar settings.');
  return Object.fromEntries(Object.keys(defaultScaleBar()).map(k => [k, s[k]]));
}
export function validateFigure(value) {
  const s = { ...defaultFigure(), ...value };
  s.scaleFontSize = value?.scaleFontSize ?? value?.fontSize ?? s.scaleFontSize;
  s.filename = validateFigureFilename(value?.filename);
  if (!['viewer', 'figure'].includes(s.mode) || !['free', '1:1', '4:3', '3:2', '16:9'].includes(s.aspect) || !['png', 'jpeg', 'svg'].includes(s.format) || !colour(s.background) || !colour(s.foreground)) throw new Error('Invalid figure settings.');
  for (const key of ['physicalLock', 'showNames', 'showLegends', 'showScaleBars', 'overlayBackground', 'autoBookmark']) if (typeof s[key] !== 'boolean') throw new Error('Invalid figure settings.');
  for (const [key, min, max] of [['umPerPixel', .000001, 1e6], ['physicalWidth', .001, 1e7], ['physicalHeight', .001, 1e7], ['paneWidth', 128, 8192], ['gap', 0, 500], ['fontSize', 8, 96], ['scaleFontSize', 8, 96], ['legendSize', 8, 72], ['legendRows', 1, 100]]) if (!finite(s[key], min, max)) throw new Error(`Invalid figure ${key}.`);
  for (const key of ['paneWidth', 'gap', 'legendRows']) if (!Number.isInteger(s[key])) throw new Error(`Figure ${key} must be a whole number.`);
  return Object.fromEntries(Object.keys(defaultFigure()).map(k => [k, s[k]]));
}
export const gridShape = count => ({ columns: count === 4 ? 2 : count, rows: count === 4 ? 2 : 1 });
export function figureLayout(count, width, height, settings) {
  const { columns, rows } = gridShape(count);
  const ratio = settings.physicalLock ? settings.physicalWidth / settings.physicalHeight : settings.aspect === 'free' ? (width / columns) / (height / rows) : settings.aspect.split(':').reduce((a, b) => Number(a) / Number(b));
  const paneWidth = settings.paneWidth, paneHeight = paneWidth / ratio;
  const outputWidth = columns * paneWidth + (columns - 1) * settings.gap, outputHeight = rows * paneHeight + (rows - 1) * settings.gap;
  const scale = Math.min(width / outputWidth, height / outputHeight);
  return { columns, rows, width: outputWidth * scale, height: outputHeight * scale, paneWidth: paneWidth * scale, paneHeight: paneHeight * scale, gap: settings.gap * scale, outputWidth, outputHeight, scale };
}
export function scaleBarGeometry(bar, umPerPixel, zoom, width) {
  const maximum = width * .35 / zoom * umPerPixel;
  const power = 10 ** Math.floor(Math.log10(maximum));
  const nice = [1, 2, 5, 10].map(n => n * power).filter(n => n <= maximum).at(-1) ?? power / 2;
  const length = bar.automatic ? nice : bar.length;
  return { length, pixels: length / umPerPixel * zoom, fits: length / umPerPixel * zoom <= width - 24 };
}
export const xml = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));
const text = (x, y, value, size, fill, extra = '') => `<text x="${x}" y="${y}" font-family="Arial, sans-serif" font-size="${size}" fill="${fill}" ${extra}>${xml(value)}</text>`;
const cssColour = c => Array.isArray(c) ? `rgb(${c.join(',')})` : c;

// The same SVG furniture is used for the live preview and all export formats.
export function paneFurniture(view, groups, settings, width, height, zoom, prefix = 'pane') {
  const figure = settings.mode === 'figure', unit = figure ? width / settings.paneWidth : 1;
  const fg = figure ? settings.foreground : view.scaleBar.colour, pad = 14 * unit, font = settings.fontSize * unit;
  const result = [];
  const backdrop = (x, y, w, h) => figure && settings.overlayBackground ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${settings.background}"/>` : '';
  if (figure && settings.showNames) result.push(`<g id="${prefix}-name">${backdrop(0, 0, Math.min(width, view.name.length * font * .65 + pad * 2), font + pad * 2)}${text(pad, pad + font, view.name, font, fg, 'font-weight="bold"')}</g>`);
  const bar = view.scaleBar;
  if (figure ? settings.showScaleBars && bar.visible : bar.visible) {
    const measured = scaleBarGeometry(bar, settings.umPerPixel, zoom, width), size = (figure ? settings.scaleFontSize : bar.fontSize) * unit;
    const x = bar.position === 'bottom-right' ? width - pad - measured.pixels : pad, y = height - pad;
    result.push(`<g id="${prefix}-scale-bar">${measured.fits ? `${backdrop(x - pad / 2, y - size - 12 * unit, measured.pixels + pad, size + 20 * unit)}<path d="M${x} ${y}h${measured.pixels}" stroke="${fg}" stroke-width="${bar.thickness * unit}"/>${text(x + measured.pixels / 2, y - 7 * unit, `${Number(measured.length.toPrecision(4))} µm`, size, fg, 'text-anchor="middle"')}` : text(pad, y, 'Scale bar too long', size, fg)}</g>`);
  }
  if (figure && settings.showLegends && view.legendVisible) {
    const size = settings.legendSize * unit, line = size * 1.5, legendWidth = Math.min(width * .48, size * 15), x = width - pad - legendWidth;
    let y = pad + size, shown = 0, omitted = 0; const content = [];
    for (let g = 0; g < groups.length; g++) {
      const group = groups[g];
      if (y + line * 3 > height - 55 * unit) { omitted += group.rows?.length ?? 1; continue; }
      content.push(text(x, y, group.title, size, fg, 'font-weight="bold"')); y += line;
      for (const row of group.rows ?? []) {
        if (shown >= settings.legendRows || y + line > height - 55 * unit) { omitted++; continue; }
        if (row.colour) content.push(`<rect x="${x}" y="${y - size * .75}" width="${size * .75}" height="${size * .75}" fill="${cssColour(row.colour)}"/>`);
        content.push(text(x + size * 1.2, y, row.name, size, fg)); y += line; shown++;
      }
      if (group.range) {
        const stops = group.gradient.match(/#[a-f0-9]{6}/gi) ?? ['#000000', '#ffffff'], id = `${prefix}-gradient-${g}`;
        content.push(`<defs><linearGradient id="${id}">${stops.map((c, i) => `<stop offset="${i / (stops.length - 1)}" stop-color="${c}"/>`).join('')}</linearGradient></defs><rect x="${x}" y="${y - size}" width="${legendWidth}" height="${size * .6}" fill="url(#${id})"/>`);
        y += line; content.push(text(x, y, group.range.map(v => Number(v.toPrecision(4))).join(' – '), size, fg)); y += line;
      }
      y += size * .5;
    }
    if (omitted) content.push(text(x, Math.min(y, height - 40 * unit), `+ ${omitted} more`, size, fg));
    result.push(`<g id="${prefix}-legend">${backdrop(x - pad / 2, pad / 2, legendWidth + pad, Math.min(y + line, height - 30 * unit) - pad / 2)}${content.join('')}</g>`);
  }
  return result.join('');
}
