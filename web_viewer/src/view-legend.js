import { labelColour } from './aligned-layers.js';
import { colourmapGradient } from './colourmaps.js';
import { populationKeys } from './populations.js';

export const resizedLegend = (scale, step) => Math.max(0.7, Math.min(2, Math.round(((scale ?? 1) + step * 0.1) * 10) / 10));

export function legendGroups(view, sources, state = {}) {
  const groups = [], rows = [];
  if (view.imagesVisible) for (const channel of view.channels) if (channel.active && channel.enabled && channel.raster && channel.showInLegend !== false) {
    const source = sources.find(source => source.id === channel.sourceId);
    rows.push({ name: source ? `${source.name} · ${channel.name}` : channel.name, colour: channel.colour });
  }
  if (rows.length) groups.push({ title: 'Image channels', rows });
  for (const source of sources) {
    const layer = view.layers?.find(layer => layer.id === source.id); if (!layer?.visible || !layer.raster || layer.showInLegend === false) continue;
    if (source.kind === 'labels') {
      if (layer.flatten) { groups.push({ title: source.name, rows: [{ name: layer.flatLabel ?? 'Foreground', colour: layer.flatColour ?? '#81d8c1' }] }); continue; }
      const labels = source.mapping.length ? source.mapping : (layer.raster.labels ?? []).map(value => ({ value, label: `Label ${value}` }));
      groups.push({ title: source.name, rows: labels.filter(item => item.value).map(item => ({ name: item.label, colour: labelColour(item.value, source.mapping) })).filter(item => item.colour) });
    } else if (source.kind === 'scalar') groups.push({ title: source.name, range: [layer.low, layer.high], gradient: colourmapGradient(layer.colourmap) });
    else groups.push({ title: source.name, rows: [{ name: 'Colour reference image' }] });
  }
  if (view.overlayVisible && view.overlay.style !== 'off' && view.region && view.overlay.showInLegend !== false) {
    const name = (state.remaps ?? []).find(plan => `remap:${plan.id}` === view.overlay.colourBy)?.name ?? (state.flags ?? []).find(field => `flag:${field.id}` === view.overlay.colourBy)?.name ?? view.overlay.colourBy.replace(/^(obs|var|remap|flag):/, '');
    const title = view.overlay.colourBy ? `Cells · ${name}` : 'Cell outlines';
    if (view.colours?.range) groups.push({ title, range: view.colours.range, gradient: colourmapGradient(view.overlay.colourmap, view.overlay.reverse) });
    else {
      const keys = populationKeys(view.overlay);
      const entries = (view.colours?.entries ?? []).filter(([category]) => !keys.length || keys.includes(JSON.stringify(category)));
      groups.push({ title, rows: entries.length ? entries.map(([category, colour]) => ({ name: String(category ?? (view.overlay.colourBy.startsWith('flag:') ? 'Unassigned' : 'Missing')), colour })) : [{ name: 'Cells', colour: '#64d8bc' }] });
    }
  }
  return groups;
}
export function renderViewLegends(state) {
  for (const view of state.views) {
    if (!view.legend) continue;
    const groups = legendGroups(view, state.layers, state);
    view.legend.hidden = !view.legendVisible || !view.region || !groups.length;
    view.legend.setAttribute('aria-label', `${view.name} legend`); view.legend.replaceChildren();
    const tools = document.createElement('header'); tools.className = 'legend-size-controls';
    const buttons = [-1, 1].map(step => {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = step < 0 ? '−' : '+';
      button.setAttribute('aria-label', `${step < 0 ? 'Decrease' : 'Increase'} legend size · ${view.name}`);
      button.onclick = event => {
        event.stopPropagation(); view.legendScale = resizedLegend(view.legendScale, step); size();
        document.dispatchEvent(new Event('workspace-changed'));
      };
      tools.append(button); return button;
    });
    function size() {
      const scale = view.legendScale ?? 1;
      view.legend.style.fontSize = `${10 * scale}px`;
      view.legend.style.width = `min(${185 * scale}px, calc(100% - 20px))`;
      buttons[0].disabled = scale <= 0.7; buttons[1].disabled = scale >= 2;
      buttons.forEach(button => { button.title = `${button.getAttribute('aria-label')} (${Math.round(scale * 100)}%)`; });
    }
    size(); view.legend.append(tools);
    for (const group of groups) {
      const section = document.createElement('div'), title = document.createElement('strong'); title.textContent = group.title; section.append(title);
      for (const row of (group.rows ?? []).slice(0, 24)) {
        const line = document.createElement('div'); line.className = 'floating-legend-row';
        if (row.colour) { const swatch = document.createElement('i'); swatch.style.background = Array.isArray(row.colour) ? `rgb(${row.colour})` : row.colour; line.append(swatch); }
        const name = document.createElement('span'); name.textContent = row.name; name.title = row.name; line.append(name); section.append(line);
      }
      if (group.rows?.length > 24) { const more = document.createElement('small'); more.textContent = `+ ${group.rows.length - 24} more · see sidebar`; section.append(more); }
      if (group.range) {
        const gradient = document.createElement('div'); gradient.className = 'floating-legend-gradient'; gradient.style.background = group.gradient;
        const range = document.createElement('div'); range.className = 'floating-legend-range';
        for (const value of group.range) { const text = document.createElement('span'); text.textContent = value.toPrecision(3); range.append(text); }
        section.append(gradient, range);
      }
      view.legend.append(section);
    }
  }
}
