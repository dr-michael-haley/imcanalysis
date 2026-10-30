// Disclosures and contextual feedback only; scientific and selection state stay
// with their existing controllers so the same tools work in both spaces.
export function createViewerToolbar(state, $, actions) {
  const modes = { inspect: ['Inspect', '↖'], click: ['Select cells', '⊕'], lasso: ['Lasso', '◌'] };
  const disclosures = [
    [$('selection-mode-trigger'), $('selection-mode-menu')],
    [$('view-menu-trigger'), $('view-menu')],
  ];
  function close(restoreFocus = false) {
    for (const [trigger, panel] of disclosures) {
      if (panel.hidden) continue;
      panel.hidden = true; trigger.setAttribute('aria-expanded', 'false');
      if (restoreFocus) trigger.focus();
    }
  }
  function open(trigger, panel, focus = false) {
    close(); panel.hidden = false; trigger.setAttribute('aria-expanded', 'true');
    if (focus) panel.querySelector('button:not(:disabled), input:not(:disabled)')?.focus();
  }
  for (const [trigger, panel] of disclosures) {
    trigger.addEventListener('click', () => panel.hidden ? open(trigger, panel) : close());
    trigger.addEventListener('keydown', event => {
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
      event.preventDefault(); open(trigger, panel, true);
    });
    panel.addEventListener('keydown', event => {
      if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
      if (event.target.matches('select, input:not([type=checkbox])')) return;
      const items = [...panel.querySelectorAll('button:not(:disabled), input:not(:disabled)')];
      const index = items.indexOf(document.activeElement); if (index < 0) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
      items[next]?.focus();
    });
  }
  for (const mode of Object.keys(modes)) $(`selection-${mode}`).addEventListener('click', () => close(true));
  for (const event of ['pointerdown', 'focusin']) document.addEventListener(event, e => {
    if (!disclosures.some(([trigger, panel]) => trigger.contains(e.target) || panel.contains(e.target))) close();
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    const menuOpen = disclosures.some(([, panel]) => !panel.hidden);
    if (menuOpen) { event.preventDefault(); event.stopPropagation(); close(true); }
    if (!$('viewer-toolbar').inert) actions.setMode('inspect');
  });
  $('selection-done').addEventListener('click', () => { actions.setMode('inspect'); $('selection-mode-trigger').focus(); });
  $('review-panels').addEventListener('toggle', sync);
  function sync() {
    const mode = actions.mode(), view = state.views[state.selectedView];
    $('selection-mode-name').textContent = modes[mode][0];
    $('selection-mode-icon').textContent = modes[mode][1];
    $('selection-mode-trigger').dataset.active = String(mode !== 'inspect');
    $('viewport').dataset.mouseMode = mode; $('embedding-canvas').dataset.mouseMode = mode;
    const hints = { inspect: 'Drag to pan, scroll to zoom, click a cell to inspect.', click: 'Click a cell to toggle its selection. Drag to pan. Escape returns to Inspect.', lasso: 'Draw around cell centres to select them. Escape returns to Inspect.' };
    $('viewport').setAttribute('aria-label', `Tissue image. ${hints[mode]}`);
    $('selection-mode-trigger').title = `${modes[mode][0]} · ${hints[mode]}`;
    $('toolbar-roi').textContent = $('roi').value || 'No ROI loaded';
    $('toolbar-roi').title = $('roi').value;
    $('toolbar-roi-position').textContent = $('roi').value ? `${$('roi').selectedIndex + 1} / ${$('roi').options.length}` : '';
    $('view-menu-heading').textContent = `Display · ${view.name}`;
    $('view-menu-trigger').title = `Display options for ${view.name}`;
    for (const [id, value] of [['quick-images', view.imagesVisible], ['quick-overlay', view.overlayVisible], ['quick-legend', view.legendVisible]]) {
      $(id).checked = value; $(id).disabled = !state.ready;
    }
    for (const id of ['selection-click', 'selection-lasso', 'zoom-in', 'zoom-out', 'fit']) $(id).disabled = !state.ready;
    if (state.figure?.mode === 'figure') for (const id of ['selection-click', 'selection-lasso']) $(id).disabled = true;
    if (state.figure?.mode === 'figure' && state.figure.physicalLock) for (const id of ['zoom-in', 'zoom-out', 'fit']) $(id).disabled = true;
    $('selection-strip').hidden = mode === 'inspect' && !state.selection.size;
    $('selection-done').hidden = mode === 'inspect';
    const roiCount = [...state.selection.rois.values()].filter(ids => ids.size).length;
    $('selection-rois').textContent = roiCount ? ` · ${roiCount} ROI${roiCount === 1 ? '' : 's'}` : '';
    const embedding = actions.scope() === 'embedding';
    $('selection-space').textContent = embedding ? 'Embedding · plotted cells' : 'Tissue · current ROI';
    $('selection-visible').title = embedding ? 'Select plotted embedding cells, including any displayed context points' : 'Select all cells matching the selected pane’s population filter in this ROI';
    $('selection-space').title = 'Select displayed uses the last space you clicked. Hidden embedding falls back to tissue.';
  }
  sync();
  return { sync, close };
}
