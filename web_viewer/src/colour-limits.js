// Null bounds retain automatic scaling. Limits change colours, never source values.
export function validateColourLimits(value = {}) {
  const result = { vmin: value.vmin ?? null, vmax: value.vmax ?? null };
  for (const bound of Object.values(result)) if (bound !== null && !Number.isFinite(bound)) throw new Error('Colour limits must be finite numbers or blank for automatic.');
  if (result.vmin !== null && result.vmax !== null && result.vmin >= result.vmax) throw new Error('Lower colour limit must be smaller than upper colour limit.');
  return result;
}

export function colourRange(automatic, limits = {}) {
  const { vmin, vmax } = validateColourLimits(limits);
  let low = vmin ?? automatic[0], high = vmax ?? automatic[1];
  // A one-sided bound can lie beyond every value in a new ROI. Keep the explicit
  // bound and extend only the automatic end, so the colours still clip correctly.
  if (high <= low) {
    if (vmax === null) high = low + Math.max(1, Math.abs(low) * .01);
    else low = high - Math.max(1, Math.abs(high) * .01);
  }
  return [low, high];
}
