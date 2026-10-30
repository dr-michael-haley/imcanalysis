import { COLORMAP_DATA } from './colormap-data.js';

export const COLOUR_MAPS = ['viridis', 'plasma', 'magma', 'inferno'];
export const mapName = id => id.charAt(0).toUpperCase() + id.slice(1);
export function colourTable(slot) {
  if (COLORMAP_DATA[slot.map]) return COLORMAP_DATA[slot.map];
  const rgb = [1, 3, 5].map(i => parseInt(slot.colour.slice(i, i + 2), 16));
  return Array.from({ length: 256 }, (_, i) => rgb.map(c => c * i / 255));
}
export function colourGradient(slot) {
  const table = colourTable(slot);
  return `linear-gradient(to right, ${[0, 32, 64, 96, 128, 160, 192, 224, 255].map(i => `rgb(${table[i].join(',')})`).join(', ')})`;
}
