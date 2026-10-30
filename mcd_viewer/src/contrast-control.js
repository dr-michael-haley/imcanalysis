export const countStep = value => Math.round(value * 10) / 10;
export function autoBounds(limits) {
  const low = Math.floor(limits.low * 10) / 10;
  return { low, high: Math.max(countStep(low + .1), Math.ceil(limits.high * 10) / 10) };
}
export function moveBound(low, high, key, value) {
  const rounded = countStep(value);
  return key === 'low' ? Math.min(rounded, countStep(high - .1)) : Math.max(rounded, countStep(low + .1));
}

// Two independently focusable handles share one track. Values always stay in raw counts.
export function contrastControl(slot, label, ready, onChange, onAuto) {
  const root = document.createElement('div'); root.className = 'contrast-control';
  const limits = document.createElement('div'); limits.className = 'limits';
  const track = document.createElement('div'); track.className = 'dual-range';
  const fill = document.createElement('div'); fill.className = 'range-fill'; track.append(fill);
  const inputs = {}, handles = {};
  let floor = Math.min(0, Math.floor((slot.low ?? 0) * 10) / 10);
  let ceiling = Math.max(.1, Math.ceil((slot.max ?? 1) * 10) / 10);
  function sync(editing = null) {
    floor = Math.min(floor, slot.low ?? 0); ceiling = Math.max(ceiling, slot.high ?? 1);
    const position = value => (value - floor) / (ceiling - floor) * 100;
    for (const key of ['low', 'high']) {
      const value = slot[key] ?? (key === 'low' ? 0 : 1), handle = handles[key];
      if (inputs[key] !== editing) inputs[key].value = slot[key] === null ? '' : value.toFixed(1);
      handle.style.left = `${position(value)}%`;
      handle.setAttribute('aria-valuemin', key === 'low' ? floor : countStep((slot.low ?? 0) + .1));
      handle.setAttribute('aria-valuemax', key === 'high' ? ceiling : countStep((slot.high ?? 1) - .1));
      handle.setAttribute('aria-valuenow', value);
      handle.setAttribute('aria-valuetext', `${value.toFixed(1)} counts`);
      handle.title = `${key === 'low' ? 'Min' : 'Max'}: ${value.toFixed(1)} counts`;
    }
    fill.style.left = `${position(slot.low ?? 0)}%`; fill.style.right = `${100 - position(slot.high ?? 1)}%`;
  }
  function update(key, value, editing = null) {
    if (!ready || !Number.isFinite(value)) { sync(); return; }
    slot[key] = moveBound(slot.low, slot.high, key, value);
    slot.max = Math.max(slot.max, slot.high); sync(editing); onChange();
  }
  for (const [key, name] of [['low', 'Min'], ['high', 'Max']]) {
    const field = document.createElement('label'); field.className = 'limit-row'; field.textContent = name;
    const input = document.createElement('input'); input.type = 'number'; input.step = '0.1'; input.disabled = !ready;
    input.ariaLabel = `${label} ${name === 'Min' ? 'minimum' : 'maximum'}`;
    input.oninput = () => { if (input.value !== '' && input.validity.valid) update(key, input.valueAsNumber, input); };
    input.onchange = () => update(key, input.valueAsNumber);
    input.onblur = () => update(key, input.valueAsNumber);
    inputs[key] = input; field.append(input); limits.append(field);
    const handle = document.createElement('button'); handle.type = 'button'; handle.className = `range-handle ${key}`;
    handle.setAttribute('role', 'slider'); handle.ariaLabel = `${label} ${name.toLowerCase()} slider`;
    handle.setAttribute('aria-orientation', 'horizontal'); handle.disabled = !ready; handles[key] = handle; track.append(handle);
    handle.onkeydown = event => {
      const delta = { ArrowLeft: -.1, ArrowDown: -.1, ArrowRight: .1, ArrowUp: .1, PageDown: -1, PageUp: 1 }[event.key];
      if (delta !== undefined || event.key === 'Home' || event.key === 'End') {
        event.preventDefault(); update(key, event.key === 'Home' ? floor : event.key === 'End' ? ceiling : slot[key] + delta);
      }
    };
  }
  let drag = null;
  const valueAt = event => { const r = track.getBoundingClientRect(); return floor + Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)) * (ceiling - floor); };
  track.onpointerdown = event => {
    if (!ready || event.button !== 0) return;
    const value = valueAt(event);
    const key = event.target === handles.low ? 'low' : event.target === handles.high ? 'high' : Math.abs(value - slot.low) < Math.abs(value - slot.high) ? 'low' : 'high';
    drag = key; handles[key].focus(); track.setPointerCapture(event.pointerId); event.preventDefault(); update(key, value);
  };
  track.onpointermove = event => { if (drag) update(drag, valueAt(event)); };
  track.onpointerup = track.onpointercancel = () => { drag = null; };
  const bottom = document.createElement('div'); bottom.className = 'channel-bottom';
  const auto = document.createElement('button'); auto.textContent = 'Auto'; auto.className = 'quiet'; auto.disabled = !ready;
  auto.title = 'Use zero / minimum and the sampled 99.5th percentile'; auto.onclick = onAuto;
  bottom.append(track, auto); root.append(limits, bottom); sync(); return root;
}
