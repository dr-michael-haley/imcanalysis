// Keep typed category identities (including null, booleans and numeric labels).
export const populationKeys = overlay => overlay.populations ?? (overlay.population ? [overlay.population] : []);
export function validatePopulations(overlay) {
  const values = populationKeys(overlay);
  if (!Array.isArray(values) || values.length > 10000 || new Set(values).size !== values.length) throw new Error('Invalid population selection.');
  return values.map(key => {
    if (typeof key !== 'string' || !key || key.length > 16384) throw new Error('Invalid population selection.');
    let value; try { value = JSON.parse(key); } catch { throw new Error('Invalid population selection.'); }
    if (value !== null && !['string', 'number', 'boolean'].includes(typeof value)) throw new Error('Invalid population selection.');
    return key;
  });
}
export function populationFilter(overlay, keys) {
  overlay.populations = keys; overlay.population = keys.length === 1 ? keys[0] : '';
}
