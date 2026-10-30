export const CORE_COLOURS = Object.freeze([
  { name: 'Red', hex: '#ff0000' },
  { name: 'Green', hex: '#00ff00' },
  { name: 'Blue', hex: '#0000ff' },
  { name: 'Cyan', hex: '#00ffff' },
  { name: 'Yellow', hex: '#ffff00' },
  { name: 'Magenta', hex: '#ff00ff' },
  { name: 'White', hex: '#ffffff' },
]);

// Stable slot colours when markers or channel ordering change.
export const DEFAULT_COLOURS = ['#0000ff', '#ff0000', '#00ff00', '#00ffff', '#ffff00', '#ff00ff'];

export function adjacentChannel(channels, currentIndex, direction) {
  if (!channels.length) return null;
  const position = channels.findIndex(c => c.index === currentIndex);
  return channels[((position < 0 ? 0 : position) + direction + channels.length) % channels.length];
}

export function orderedChannels(channels, order = 'alphabetical') {
  return order === 'original' ? [...channels] : [...channels].sort((a, b) =>
    a.label.localeCompare(b.label, 'en', { sensitivity: 'base', numeric: true }) || a.index - b.index);
}
