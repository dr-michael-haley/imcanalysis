// Sampled colour stops shared by cell rendering and the continuous legend.
export const COLOURMAPS = {
  viridis: ['Viridis', ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']],
  plasma: ['Plasma', ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921']],
  magma: ['Magma', ['#000004', '#51127c', '#b73779', '#fc8961', '#fcfdbf']],
  inferno: ['Inferno', ['#000004', '#57106e', '#bc3754', '#f98e09', '#fcffa4']],
  cividis: ['Cividis', ['#00224e', '#434e6c', '#7d7c78', '#bcae6c', '#fee838']],
  grey: ['Greyscale', ['#000000', '#ffffff']],
  red: ['Black–red', ['#000000', '#ff0000']],
  green: ['Black–green', ['#000000', '#00ff00']],
  blue: ['Black–blue', ['#000000', '#0000ff']],
  coolwarm: ['Blue–white–red', ['#2166ac', '#f7f7f7', '#b2182b']],
  legacy: ['Original', ['#303469', '#ffe835']],
};
export function colourmap(t, name = 'viridis', reverse = false) {
  const stops = (COLOURMAPS[name] ?? COLOURMAPS.viridis)[1];
  const value = Math.max(0, Math.min(1, reverse ? 1 - t : t)) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(value)), fraction = value - index;
  const parse = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  const a = parse(stops[index]), b = parse(stops[index + 1]);
  return a.map((v, i) => Math.round(v + fraction * (b[i] - v)));
}
export function colourmapGradient(name, reverse = false) {
  const stops = [...(COLOURMAPS[name] ?? COLOURMAPS.viridis)[1]];
  return `linear-gradient(to right, ${(reverse ? stops.reverse() : stops).join(', ')})`;
}
