/* Reusable, dependency-free usage prompt. No dataset or page-content access. */
(() => {
  const script = document.currentScript;
  const tool = script?.dataset.usageTool;
  const endpoint = script?.dataset.usageEndpoint;
  const hosts = script?.dataset.usageHosts?.split(',');
  if (!tool || !endpoint || (hosts && !hosts.includes(location.hostname))) return;
  if (document.querySelector('sbt-usage-preferences')) return;

  const preferenceKey = 'sbt-usage-preferences-v1';
  const receiptKey = `sbt-usage-receipt-v1:${tool}`;
  const actionChoiceKey = 'sbt-usage-action-counts-v1';
  const read = key => { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } };
  const write = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Private/restricted storage: session only. */ } };
  const valid = p => p?.version === 1 && typeof p.anonymous === 'boolean'
    && ['manchester', 'external'].includes(p.affiliation)
    && (p.anonymous || (typeof p.name === 'string' && p.name.trim().length > 0 && p.name.length <= 120));
  let preference = read(preferenceKey);
  if (!valid(preference)) preference = null;
  let actionChoice = read(actionChoiceKey);
  let receipt = read(receiptKey);
  let sending = false;
  let lastAttempt = 0;

  // The viewer supplies only a fixed action name, after successful completion.
  window.sbtRecordAction = async action => {
    if (!preference || actionChoice !== true || !['dataset_opened', 'figure_exported', 'demo_opened'].includes(action)) return;
    try {
      await fetch(new URL('/actions', endpoint), {
        method: 'POST', credentials: 'omit', referrerPolicy: 'no-referrer', keepalive: true,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool, affiliation: preference.affiliation, action, eventId: crypto.randomUUID() }),
        signal: AbortSignal.timeout(5000),
      });
    } catch { /* Best-effort reporting must not interrupt the viewer. */ }
  };

  // An anonymous event has a fresh key each UTC day/tool, never a persistent user ID.
  async function record() {
    if (!preference || sending || Date.now() - lastAttempt < 30000) return;
    const day = new Date().toISOString().slice(0, 10);
    const identity = JSON.stringify(preference);
    if (receipt?.day !== day || receipt?.identity !== identity) {
      receipt = { day, identity, eventId: crypto.randomUUID(), sent: false };
    }
    if (receipt.sent) return;
    write(receiptKey, receipt);
    sending = true;
    lastAttempt = Date.now();
    const pending = receipt;
    try {
      const response = await fetch(endpoint, {
        method: 'POST', mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool, name: preference.anonymous ? null : preference.name,
          affiliation: preference.affiliation, eventId: pending.eventId }),
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok && receipt === pending) { receipt.sent = true; write(receiptKey, receipt); }
    } catch { /* Usage reporting must never interrupt science or offline use. */ }
    finally { sending = false; if (receipt !== pending) { lastAttempt = 0; void record(); } }
  }

  const host = document.createElement('sbt-usage-preferences');
  const root = host.attachShadow({ mode: 'open' });
  root.innerHTML = `
    <style>
      :host { font: 15px/1.5 Inter, ui-sans-serif, system-ui, sans-serif; color: #dfe8eb; color-scheme: dark; }
      * { box-sizing: border-box; }
      button, input { font: inherit; }
      button { cursor: pointer; }
      :focus-visible { outline: 2px solid #81d8c1; outline-offset: 3px; }
      ::selection { background: #81d8c1; color: #10171d; }
      .launcher { position: fixed; bottom: 12px; right: 12px; z-index: 1000; min-height: 36px; padding: 6px 12px; border: 1px solid #526570; border-radius: 6px; background: #18242d; color: #dfe8eb; font-size: 12px; }
      .launcher:hover, .secondary:hover { background: #2b3943; }
      dialog { width: min(480px, calc(100vw - 32px)); max-height: calc(100dvh - 32px); padding: 28px; border: 1px solid #526570; border-radius: 14px; background: #18242d; color: #dfe8eb; overflow-y: auto; scrollbar-color: #526570 #18242d; }
      dialog::backdrop { background: rgb(6 12 17 / 78%); }
      h2 { margin: 0 0 12px; font-size: 24px; line-height: 1.2; font-weight: 650; text-wrap: balance; }
      p { margin: 0 0 20px; color: #b5c5cd; }
      label, legend { color: #dfe8eb; }
      .name-label { display: block; margin-bottom: 6px; font-weight: 600; }
      input[type=text] { width: 100%; min-height: 44px; padding: 10px 12px; border: 1px solid #667c89; border-radius: 6px; background: #10171d; color: #dfe8eb; caret-color: #81d8c1; }
      input::placeholder { color: #9aaeb9; }
      input:disabled { opacity: .5; cursor: not-allowed; }
      .choice { display: flex; align-items: center; gap: 10px; min-height: 44px; cursor: pointer; }
      input[type=checkbox], input[type=radio] { width: 18px; height: 18px; margin: 0; accent-color: #81d8c1; flex-shrink: 0; }
      fieldset { border: 0; padding: 0; margin: 18px 0; min-width: 0; }
      legend { font-weight: 600; padding: 0; margin-bottom: 4px; }
      .note { font-size: 13px; margin-bottom: 16px; }
      .actions { display: flex; gap: 12px; flex-wrap: wrap; }
      .primary, .secondary { min-height: 44px; padding: 10px 16px; border-radius: 6px; font-weight: 600; }
      .primary { border: 0; background: #81d8c1; color: #10171d; flex: 1; }
      .primary:hover { background: #a2e5d4; }
      .secondary { border: 1px solid #667c89; background: transparent; color: #dfe8eb; }
      .skip { display: block; margin: 14px auto 0; padding: 8px; border: 0; background: transparent; color: #b5c5cd; text-decoration: underline; text-underline-offset: 3px; }
      .error { color: #ffb9aa; margin: 0 0 12px; }
      [hidden] { display: none !important; }
      @media (max-width: 420px) { dialog { padding: 20px; } h2 { font-size: 22px; } }
    </style>
    <button class="launcher" type="button" aria-haspopup="dialog">Usage preferences</button>
    <dialog aria-labelledby="usage-title" aria-describedby="usage-intro">
      <form novalidate>
        <h2 id="usage-title">Help us keep these tools free</h2>
        <p id="usage-intro">Basic usage records help us demonstrate the value of these tools and support their continued maintenance.</p>
        <label class="name-label" for="usage-name">Your name</label>
        <input id="usage-name" type="text" autocomplete="name" maxlength="120" placeholder="Enter your name" autofocus>
        <label class="choice"><input id="usage-anonymous" type="checkbox">Remain anonymous</label>
        <fieldset><legend>Where are you using this tool?</legend>
          <label class="choice"><input type="radio" name="affiliation" value="manchester">University of Manchester</label>
          <label class="choice"><input type="radio" name="affiliation" value="external">External</label>
        </fieldset>
        <p class="note">We record the tool, date, affiliation and, if supplied, your name. Your files and analysis data are never included in usage records.</p>
        <label class="choice"><input id="usage-actions" type="checkbox">Share anonymous counts of dataset opens${tool === 'mcd' ? ', demo opens' : ''} and figure exports</label>
        <p class="note">Action counts include only the app, day and affiliation. The public dashboard shows totals.</p>
        <p class="note">Your choice is remembered in this browser for this tool. Change it anytime in Usage preferences. Changes apply to future records.</p>
        <p class="error" role="alert" hidden></p>
        <div class="actions"><button class="secondary" type="button" hidden>Cancel</button><button class="primary" type="submit">Continue to tool</button></div>
        <button class="skip" type="button">Not now</button>
      </form>
    </dialog>`;
  document.body.append(host);
  const dialog = root.querySelector('dialog');
  const name = root.querySelector('#usage-name');
  const anonymous = root.querySelector('#usage-anonymous');
  const error = root.querySelector('.error');
  const cancel = root.querySelector('.secondary');
  root.querySelector('form').addEventListener('input', () => { error.hidden = true; });
  const open = () => {
    name.value = preference?.name || '';
    anonymous.checked = preference?.anonymous || false;
    root.querySelector('#usage-actions').checked = actionChoice === true;
    name.disabled = anonymous.checked;
    for (const radio of root.querySelectorAll('[name=affiliation]')) radio.checked = radio.value === preference?.affiliation;
    error.hidden = true;
    cancel.hidden = !preference;
    root.querySelector('.skip').hidden = Boolean(preference);
    root.querySelector('.primary').textContent = preference ? 'Save preferences' : 'Continue to tool';
    dialog.showModal();
  };
  root.querySelector('.launcher').addEventListener('click', open);
  anonymous.addEventListener('change', () => { name.disabled = anonymous.checked; if (!anonymous.checked) name.focus(); });
  cancel.addEventListener('click', () => dialog.close());
  root.querySelector('.skip').addEventListener('click', () => dialog.close());
  // Keep viewer-wide shortcuts from consuming input while the dialog is open.
  for (const event of ['keydown', 'keyup', 'keypress']) dialog.addEventListener(event, e => e.stopPropagation());
  root.querySelector('form').addEventListener('submit', event => {
    event.preventDefault();
    const affiliation = root.querySelector('[name=affiliation]:checked')?.value;
    const cleanName = name.value.normalize('NFKC').trim().replace(/\s+/g, ' ');
    if ((!anonymous.checked && (!cleanName || cleanName.length > 120 || /[\u0000-\u001f\u007f]/.test(cleanName))) || !affiliation) {
      error.textContent = !anonymous.checked && !cleanName ? 'Enter your name or choose Remain anonymous.' : !affiliation ? 'Choose University of Manchester or External.' : 'Please enter a name of up to 120 characters.';
      error.hidden = false;
      (!anonymous.checked && !cleanName ? name : root.querySelector('[name=affiliation]')).focus();
      return;
    }
    const next = { version: 1, anonymous: anonymous.checked, name: anonymous.checked ? null : cleanName, affiliation };
    const changed = JSON.stringify(next) !== JSON.stringify(preference);
    preference = next;
    actionChoice = root.querySelector('#usage-actions').checked;
    write(actionChoiceKey, actionChoice);
    write(preferenceKey, preference);
    // Discard the old local receipt immediately when changing to anonymous.
    if (changed) { receipt = null; write(receiptKey, null); }
    lastAttempt = 0;
    dialog.close();
    void record();
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) void record(); });
  window.addEventListener('online', () => { lastAttempt = 0; void record(); });
  if (preference) {
    void record();
    // Give returning visitors one chance to enable the newly introduced counts.
    if (actionChoice === null) { actionChoice = false; write(actionChoiceKey, false); open(); }
  } else open();
})();
