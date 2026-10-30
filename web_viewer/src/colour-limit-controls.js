import { validateColourLimits } from './colour-limits.js';

export function createColourLimitControls(root, prefix, changed) {
  root.classList.add('colour-limits');
  root.innerHTML = `<div class="colour-limits-heading"><span>Colour limits</span><button type="button" class="quiet" id="${prefix}-auto">Auto</button></div><div class="colour-limits-inputs"><label>Lower (vmin)<input id="${prefix}-vmin" type="number" step="any" placeholder="Auto"></label><label>Upper (vmax)<input id="${prefix}-vmax" type="number" step="any" placeholder="Auto"></label></div><p class="muted colour-limits-help">Blank = automatic. Values outside the range use the end colours.</p><p class="input-error" role="alert" hidden></p>`;
  const lower = root.querySelector(`#${prefix}-vmin`), upper = root.querySelector(`#${prefix}-vmax`), error = root.querySelector('[role=alert]');
  let last = { vmin: null, vmax: null };
  function clearError() { error.hidden = true; lower.setCustomValidity(''); upper.setCustomValidity(''); }
  function read() {
    if (lower.validity.badInput || upper.validity.badInput) throw new Error('Enter a valid number or leave the colour limit blank.');
    return validateColourLimits({ vmin: lower.value === '' ? null : Number(lower.value), vmax: upper.value === '' ? null : Number(upper.value) });
  }
  function sync(value) { last = validateColourLimits(value); lower.value = last.vmin ?? ''; upper.value = last.vmax ?? ''; clearError(); }
  function commit() {
    clearError();
    try {
      const next = read();
      if (next.vmin !== last.vmin || next.vmax !== last.vmax) { last = next; changed(next); }
    } catch (err) { error.textContent = err.message; error.hidden = false; upper.setCustomValidity(err.message); }
  }
  for (const input of [lower, upper]) { input.addEventListener('input', clearError); input.addEventListener('change', commit); input.addEventListener('blur', commit); }
  root.querySelector('button').onclick = () => { sync({}); changed({ ...last }); };
  return { sync, read };
}
