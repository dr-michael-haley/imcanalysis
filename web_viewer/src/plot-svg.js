import { alluvialSVG } from './alluvial.js';
import { colourRange } from './colour-limits.js';
const escape = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&apos;');
const number = v => v == null ? 'No finite values' : Number.isInteger(v) ? v.toLocaleString('en-GB') : Number(v.toPrecision(4)).toString();
const transpose = matrix => matrix[0].map((_, i) => matrix.map(row => row[i]));
const short = (text, limit) => text.length > limit ? text.slice(0, limit - 1) + '…' : text;
const mix = (a, b, t) => '#' + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, '0')).join('');
function colour(t, palette) {
  t = Math.max(0, Math.min(1, t));
  const stops = palette === 'diverging' ? [[43, 91, 156], [247, 247, 247], [181, 37, 54]] : palette === 'viridis' ? [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]] : [[239, 246, 251], [24, 87, 136]];
  const at = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(at)); return mix(stops[i], stops[i + 1], at - i);
}

// A self-contained SVG also serves as the downloadable, resolution-independent figure.
export function plotSVG(result) {
  if (result.type === 'alluvial') return alluvialSVG(result);
  let rows = result.rows, columns = result.columns, values = result.values, rowTitle = result.rowTitle, columnTitle = result.columnTitle;
  const bars = result.type === 'bars';
  if (result.settings.swapAxes && !bars) { [rows, columns] = [columns, rows]; values = transpose(values); [rowTitle, columnTitle] = [columnTitle, rowTitle]; }
  const left = Math.min(410, Math.max(130, ...rows.map(row => row.label.length * 7 + 34)));
  const cellWidth = Math.max(50, Math.min(85, 620 / columns.length)), cellHeight = 27;
  const plotWidth = bars ? Math.max(480, columns.length * 44) : Math.max(280, columns.length * cellWidth);
  const plotHeight = bars ? 290 : rows.length * cellHeight;
  const bottom = Math.min(300, Math.max(100, ...columns.map(col => col.label.length * 6.2 + 40)));
  const width = Math.ceil(left + plotWidth + (bars ? 250 : 170));
  const titleLimit = Math.max(30, Math.floor((width - 48) / 11)), titleLines = result.title.match(new RegExp(`.{1,${titleLimit}}`, 'g')) || [''];
  const top = 92 + (titleLines.length - 1) * 22, height = Math.ceil(top + plotHeight + bottom);
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="plot-title plot-description"><title id="plot-title">${escape(result.title)}</title><desc id="plot-description">${escape(result.subtitle + '. ' + result.valueLabel)}</desc><rect width="100%" height="100%" fill="white"/>`];
  const text = (x, y, label, attrs = '', limit = Infinity) => parts.push(`<text x="${x}" y="${y}" font-family="Arial, sans-serif" font-size="12" fill="#243341" ${attrs}><title>${escape(label)}</title>${escape(short(label, limit))}</text>`);
  const rect = (x, y, w, h, fill, title = '') => parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="white" stroke-width="0.4">${title ? `<title>${escape(title)}</title>` : ''}</rect>`);
  titleLines.forEach((line, i) => text(24, 30 + i * 22, line, 'font-weight="bold" style="font-size:19px"'));
  text(24, top - 40, result.subtitle, '', Math.floor((width - 48) / 6.5));
  text(24, top - 21, `${result.settings.filterBy ? `Filter: ${result.settings.filterBy} · ` : ''}${result.settings.populations.length ? `${result.settings.populations.length} population(s) selected · ` : ''}${result.valueLabel}`, '', Math.floor((width - 48) / 6.5));
  if (bars) {
    const totals = columns.map((_, col) => values.reduce((sum, row) => sum + row[col], 0)), maximum = result.settings.measure === 'percent' ? 100 : Math.max(1, ...totals);
    for (let tick = 0; tick <= 4; tick++) {
      const y = top + plotHeight - tick / 4 * plotHeight;
      parts.push(`<path d="M${left},${y}H${left + plotWidth}" stroke="#dce3e8"/>`); text(left - 12, y + 4, number(maximum * tick / 4), 'text-anchor="end"');
    }
    const slot = plotWidth / columns.length;
    columns.forEach((column, col) => {
      let base = 0;
      rows.forEach((row, r) => { const value = values[r][col], h = value / maximum * plotHeight; rect(left + col * slot + slot * .13, top + plotHeight - base - h, slot * .74, h, row.colour || '#89949e', `${column.label} · ${row.label}: ${number(value)}${result.settings.measure === 'percent' ? '%' : ''} (${result.counts[r][col]} cells)`); base += h; });
    });
    rows.forEach((row, r) => { rect(left + plotWidth + 20, top + r * 19, 11, 11, row.colour || '#89949e'); text(left + plotWidth + 38, top + r * 19 + 10, row.label, '', 32); });
    text(22, top + plotHeight / 2, result.valueLabel, `transform="rotate(-90 22 ${top + plotHeight / 2})" text-anchor="middle"`);
  } else {
    const finite = values.flat().filter(Number.isFinite);
    const palette = result.settings.palette === 'automatic' ? result.type === 'matrix' && result.settings.scale === 'zscore' ? 'diverging' : 'blue' : result.settings.palette;
    let low = finite.length ? Math.min(...finite) : 0, high = finite.length ? Math.max(...finite) : 1;
    if (palette === 'diverging') { high = Math.max(Math.abs(low), Math.abs(high), 1e-12); low = -high; }
    else if (result.type !== 'matrix' || result.settings.scale === 'minmax') { low = 0; high = result.type !== 'matrix' && result.valueLabel.includes('%') ? 100 : result.settings.scale === 'minmax' && result.type === 'matrix' ? 1 : high; }
    [low, high] = colourRange([low, high], result.settings);
    const w = plotWidth / columns.length;
    rows.forEach((row, r) => {
      if (row.colour) rect(left - 18, top + r * cellHeight + 5, 10, 17, row.colour);
      text(left - 24, top + r * cellHeight + 18, row.label, 'text-anchor="end"', 52);
      columns.forEach((col, c) => {
        const value = values[r][c], fill = value == null ? '#dce1e5' : colour((value - low) / (high - low || 1), palette);
        rect(left + c * w, top + r * cellHeight, w, cellHeight, fill, `${row.label} · ${col.label}: ${number(value)}`);
        if (rows.length * columns.length <= 225) text(left + c * w + w / 2, top + r * cellHeight + 18, value == null ? '—' : number(value), `text-anchor="middle" style="font-size:10px;fill:${value != null && ((value - low) / (high - low || 1) > .65 || palette === 'diverging' && (value - low) / (high - low || 1) < .25) ? 'white' : '#172b3a'}"`);
      });
    });
    columns.forEach((col, c) => { if (col.colour) rect(left + c * w, top - 11, w, 7, col.colour); });
    const keyX = left + plotWidth + 25, keyHeight = Math.min(140, Math.max(60, plotHeight));
    for (let i = 0; i < 60; i++) rect(keyX, top + i * keyHeight / 60, 15, keyHeight / 60 + .5, colour(1 - i / 59, palette));
    text(keyX + 21, top + 10, number(high)); text(keyX + 21, top + keyHeight, number(low));
    text(22, top + plotHeight / 2, rowTitle, `transform="rotate(-90 22 ${top + plotHeight / 2})" text-anchor="middle"`, Math.max(10, Math.floor((plotHeight + 70) / 7)));
  }
  columns.forEach((column, col) => { const x = left + (col + .5) * plotWidth / columns.length, y = top + plotHeight + 16; text(x, y, column.label, `transform="rotate(55 ${x} ${y})" text-anchor="start"`, 40); });
  text(left + plotWidth / 2, height - 15, columnTitle, 'text-anchor="middle" font-weight="bold"', 70);
  // Long categorical legends may extend below the bars.
  const finalHeight = bars ? Math.max(height, top + rows.length * 19 + 25) : height;
  parts[0] = parts[0].replace(`height="${height}" viewBox="0 0 ${width} ${height}"`, `height="${finalHeight}" viewBox="0 0 ${width} ${finalHeight}"`);
  parts.push('</svg>'); return parts.join('');
}
