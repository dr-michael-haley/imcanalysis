// Matrix orientation matches Compare labels: rows are targets, columns sources.
// One common count-to-height scale keeps every ribbon's width conserved.
export function alluvialLayout(counts, height = 420, gap = 10) {
  if (!Array.isArray(counts) || !counts.length || !counts[0].length || counts.some(row => row.length !== counts[0].length || row.some(v => !Number.isFinite(v) || v < 0))) throw new Error('Invalid alluvial counts.');
  const leftTotals = counts[0].map((_, c) => counts.reduce((sum, row) => sum + row[c], 0)), rightTotals = counts.map(row => row.reduce((a, b) => a + b, 0));
  const total = leftTotals.reduce((a, b) => a + b, 0), available = height - gap * (Math.max(leftTotals.length, rightTotals.length) - 1);
  if (!total || available <= 0) throw new Error('Alluvial plot needs positive counts and enough space for its groups.');
  const scale = available / total;
  const nodes = totals => { let y = (height - (total * scale + gap * (totals.length - 1))) / 2; return totals.map(count => { const node = { y, height: count * scale, count }; y += node.height + gap; return node; }); };
  const left = nodes(leftTotals), right = nodes(rightTotals), offsetsLeft = left.map(n => n.y), offsetsRight = right.map(n => n.y), flows = [];
  for (let c = 0; c < left.length; c++) for (let r = 0; r < right.length; r++) if (counts[r][c]) {
    const count = counts[r][c], h = count * scale;
    flows.push({ source: c, target: r, count, height: h, y1: offsetsLeft[c], y2: offsetsRight[r] }); offsetsLeft[c] += h; offsetsRight[r] += h;
  }
  return { left, right, flows, total, height, scale };
}
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));
export function alluvialSVG(result) {
  const height = Math.max(360, Math.max(result.rows.length, result.columns.length) * 30), chart = alluvialLayout(result.counts, height);
  const longest = Math.max(...[...result.rows, ...result.columns].map(c => c.label.length)), margin = Math.min(380, Math.max(180, longest * 7 + 75)), width = margin * 2 + 480;
  const top = 100, x1 = margin, x2 = width - margin - 16;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height + 160}" viewBox="0 0 ${width} ${height + 160}" role="img" aria-labelledby="alluvial-title alluvial-desc"><title id="alluvial-title">${esc(result.title)}</title><desc id="alluvial-desc">Categorical alluvial plot. Ribbon widths represent cell counts, from ${esc(result.columnTitle)} to ${esc(result.rowTitle)}.</desc><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" fill="#18232f"><text x="24" y="30" font-size="20" font-weight="bold">${esc(result.title)}</text><text x="24" y="53" font-size="12">${esc(result.subtitle)}</text><text x="${x1}" y="82" font-size="13" font-weight="bold">${esc(result.columnTitle)}</text><text x="${x2 + 16}" y="82" text-anchor="end" font-size="13" font-weight="bold">${esc(result.rowTitle)}</text>`;
  for (const f of chart.flows) {
    const a = x1 + 16, b = x2, mid = (a + b) / 2, y1 = top + f.y1, y2 = top + f.y2;
    const colour = /^#[0-9a-f]{6}$/i.test(result.columns[f.source].colour) ? result.columns[f.source].colour : '#5b8db8';
    svg += `<path class="alluvial-flow" d="M${a},${y1} C${mid},${y1} ${mid},${y2} ${b},${y2} L${b},${y2 + f.height} C${mid},${y2 + f.height} ${mid},${y1 + f.height} ${a},${y1 + f.height} Z" fill="${colour}" fill-opacity="0.45"><title>${esc(result.columns[f.source].label)} → ${esc(result.rows[f.target].label)}: ${f.count.toLocaleString()} cells (${(f.count / chart.total * 100).toFixed(1)}%)</title></path>`;
  }
  for (const [nodes, labels, x, side] of [[chart.left, result.columns, x1, 'left'], [chart.right, result.rows, x2, 'right']]) nodes.forEach((n, i) => {
    const colour = /^#[0-9a-f]{6}$/i.test(labels[i].colour) ? labels[i].colour : '#5b8db8';
    svg += `<rect x="${x}" y="${top + n.y}" width="16" height="${n.height}" fill="${colour}"><title>${esc(labels[i].label)}: ${n.count.toLocaleString()} cells</title></rect><text x="${side === 'left' ? x - 8 : x + 24}" y="${top + n.y + n.height / 2 + 4}" text-anchor="${side === 'left' ? 'end' : 'start'}" font-size="12">${esc(labels[i].label)} · ${n.count.toLocaleString()}</text>`;
  });
  return svg + `<text x="24" y="${height + 138}" font-size="11" fill="#506070">Ribbon widths show cell counts. Hover over a ribbon for its count and percentage of selected cells.</text></g></svg>`;
}
