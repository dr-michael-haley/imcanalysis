import { colourTable } from './colormaps.js';

const $ = id => document.getElementById(id);
const svgNode = (tag, attributes = {}, text) => {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value);
  if (text !== undefined) el.textContent = text; return el;
};
const format = value => Math.abs(value) >= 10000 ? value.toExponential(1) : Number(value.toPrecision(3)).toLocaleString('en');

export class HistogramPanel {
  constructor(getState, task) {
    this.getState = getState; this.task = task; this.key = ''; this.selected = new Set(); this.epoch = 0;
    $('generate-histogram').onclick = () => this.generate();
    $('cancel-histogram').onclick = () => { this.invalidate('Generation cancelled. Click Generate to try again.'); this.sync(); };
    $('histogram-log').onchange = () => this.draw();
  }
  available() {
    const s = this.getState();
    return [...new Set(s.slots.filter(slot => slot.enabled).map(slot => slot.index))].map(index => ({
      index, channel: s.acquisition.channels.find(c => c.index === index), slot: s.slots.find(slot => slot.enabled && slot.index === index),
    }));
  }
  invalidate(text = 'Click Generate to calculate curves.') {
    this.epoch++; this.controller?.abort(); this.controller = null; this.result = null;
    $('histogram-result').hidden = true; $('histogram-plot').replaceChildren(); $('histogram-legend').replaceChildren();
    $('cancel-histogram').hidden = true; $('histogram-status').textContent = text;
  }
  sync() {
    const state = this.getState(), available = state.acquisition ? this.available() : [];
    const key = `${state.fileEpoch}/${state.acquisition?.id}/${available.map(c => c.index).join(',')}`;
    if (key !== this.key) {
      const sameAcquisition = key.split('/').slice(0, 2).join('/') === this.key.split('/').slice(0, 2).join('/');
      const oldIndices = this.indices || [];
      this.selected = new Set(available.filter(c => !sameAcquisition || this.selected.has(c.index) || !oldIndices.includes(c.index)).map(c => c.index));
      this.key = key; this.indices = available.map(c => c.index);
      this.invalidate(available.length ? 'Click Generate to calculate curves.' : state.acquisition ? 'Show a channel to generate its histogram.' : 'Open an acquisition to begin.');
      $('histogram-channels').replaceChildren(...available.map(({ index, channel }) => {
        const label = document.createElement('label'); label.className = 'checkbox-label';
        const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = this.selected.has(index); checkbox.ariaLabel = `Histogram for ${channel.label}`;
        checkbox.onchange = () => { checkbox.checked ? this.selected.add(index) : this.selected.delete(index); this.invalidate(); this.sync(); };
        label.append(checkbox, document.createTextNode(channel.label)); return label;
      }));
    }
    $('generate-histogram').disabled = !state.raster || !this.selected.size || !!this.controller;
    if (this.result) this.draw();
  }
  async generate() {
    const state = this.getState(); if (!state.raster || !this.selected.size) return;
    this.invalidate('Generating histogram…');
    const epoch = this.epoch, controller = new AbortController(); this.controller = controller;
    $('cancel-histogram').hidden = false; this.sync();
    try {
      // Structured cloning keeps the displayed raster intact; only requested channels are copied.
      const planes = state.raster.planes.filter(p => this.selected.has(p.index)).map(({ index, data }) => ({ index, data }));
      const result = await this.task({ type: 'histogram', planes }, controller.signal, false);
      if (epoch !== this.epoch) return;
      this.result = result;
      $('histogram-status').textContent = result.curves.length ? `${state.acquisition.name} · ${result.bins} bins · raw counts` : 'No finite measured pixels in the selected channels.';
    } catch (error) {
      if (epoch === this.epoch && error.name !== 'AbortError') $('histogram-status').textContent = `Could not generate histogram: ${error.message}`;
    } finally {
      if (epoch === this.epoch) { this.controller = null; $('cancel-histogram').hidden = true; this.sync(); }
    }
  }
  draw() {
    const result = this.result; if (!result?.curves.length) return;
    $('histogram-result').hidden = false;
    const svg = $('histogram-plot'), width = 270, height = 210, left = 47, right = 10, top = 20, bottom = 40;
    const plotW = width - left - right, plotH = height - top - bottom, log = $('histogram-log').checked;
    const transform = v => log ? Math.log1p(v) : v;
    const max = Math.max(1, ...result.curves.map(c => Math.max(...c.counts)));
    const x = t => left + t * plotW, y = count => top + plotH * (1 - transform(count) / transform(max));
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.replaceChildren();
    svg.append(svgNode('text', { x: left, y: 10, class: 'axis-title' }, log ? 'Pixels / bin (log)' : 'Pixels / bin'));
    for (let i = 0; i <= 4; i++) {
      const fraction = i / 4, value = log ? Math.expm1(fraction * Math.log1p(max)) : max * fraction;
      svg.append(svgNode('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), class: 'histogram-grid' }), svgNode('text', { x: left - 5, y: y(value) + 3, 'text-anchor': 'end' }, format(value)));
    }
    for (let i = 0; i <= 2; i++) svg.append(svgNode('text', { x: x(i / 2), y: height - 23, 'text-anchor': i === 0 ? 'start' : i === 2 ? 'end' : 'middle' }, format(result.low + (result.high - result.low) * i / 2)));
    svg.append(svgNode('text', { x: left + plotW / 2, y: height - 5, 'text-anchor': 'middle', class: 'axis-title' }, 'Intensity (counts)'));
    const available = this.available(); $('histogram-legend').replaceChildren();
    for (const curve of result.curves) {
      const item = available.find(c => c.index === curve.index); if (!item) continue;
      const colour = item.slot.map && item.slot.map !== 'pure' ? `rgb(${colourTable(item.slot)[170].join(',')})` : item.slot.colour;
      const path = svgNode('path', { d: [...curve.counts].map((count, i) => `${i ? 'L' : 'M'}${x((i + .5) / result.bins)},${y(count)}`).join(' '), fill: 'none', stroke: colour, 'stroke-width': 1.6 });
      path.append(svgNode('title', {}, `${item.channel.label}: ${curve.samples.toLocaleString()} measured pixels`)); svg.append(path);
      const label = document.createElement('div'), line = document.createElement('i'); line.style.background = colour;
      label.append(line, document.createTextNode(`${item.channel.label} · ${curve.samples.toLocaleString()} pixels`));
      label.title = `${curve.missing.toLocaleString()} missing or non-finite pixels excluded`; $('histogram-legend').append(label);
    }
  }
}
