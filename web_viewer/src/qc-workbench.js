import {
  blankCatalog,
  blankRun,
  buildSource,
  commit,
  mapROIs,
  readJson,
  rememberInputs,
  savedDraft,
  sourceInventory,
  writeFile,
  writeJson
} from './analysis-project.js';
import { asset, computeJob, localInputMap, openSource, roiFeatures } from './qc-data.js';
import { readMetrics, decision, roiScoreSummary } from './mask-qc.js';
import { decisionOverlay } from './analysis-tissue.js';
import { createQCReviewPanel } from './qc-review-panel.js';

const option = (value, label) => new Option(label, value);
function options(select, pairs) {
  select.replaceChildren(...pairs.map(([value, label]) => option(value, label)));
}
const countLabels = groups => Object.values(groups).reduce((sum, group) => sum + Object.keys(group).length, 0);

export function createQCWorkbench($, setupFolders, selectAnalyses, viewer) {
  const space = document.querySelector('.workspace'),
    tabs = [...document.querySelectorAll('[data-qc-tab]')];
  let session = null,
    active = false,
    tab = 'inputs';
  const reviewPanel = createQCReviewPanel($, {
    openROI: roi => action((s, signal) => openViews(s, signal, roi)),
    changeDisplay: display => action(async s => {
      if (!(0 <= display.min && display.min < display.max && display.max <= 1))
        throw Error('Choose a lower and upper colour limit between 0 and 1, with the lower limit first.');
      await saveRun(s, { display });
      paint(s);
      message(s, 'Display settings saved for this analysis.');
    })
  });
  const clearAfterMark = $('qc-clear-after-mark');
  try {
    clearAfterMark.checked = localStorage.getItem('imc-qc-clear-after-mark') !== 'false';
  } catch { /* The checkbox still works for this session. */ }
  clearAfterMark.onchange = () => {
    try { localStorage.setItem('imc-qc-clear-after-mark', String(clearAfterMark.checked)); }
    catch { /* The checkbox still works for this session. */ }
  };
  const hardware = Math.min(8, navigator.hardwareConcurrency || 2);
  $('qc-workers').max = hardware;
  for (const [key, initial] of [
    ['workers', Math.min(2, hardware)],
    ['memory', 512]
  ]) {
    const input = $('qc-' + key);
    const limit = value => Math.max(+input.min, Math.min(+input.max, Math.floor(Number(value) || initial)));
    input.value = initial;
    try {
      input.value = limit(localStorage.getItem('imc-analysis-' + key) ?? initial);
    } catch {
      /* Browser storage can be disabled. */
    }
    input.onchange = () => {
      input.value = limit(input.value);
      try {
        localStorage.setItem('imc-analysis-' + key, input.value);
      } catch {
        /* Keep the session setting. */
      }
    };
  }
  function resources() {
    const workers = Math.max(1, Math.min(hardware, +$('qc-workers').value || 1));
    return { workers, maxPixels: Math.floor(((+$('qc-memory').value || 512) * 1024 ** 2) / (workers * 32)) };
  }
  function message(s, text, error = false) {
    if (session !== s) return;
    $('qc-status').textContent = text;
    $('qc-status').dataset.error = String(error);
  }
  function paint(s) {
    if (session !== s) return;
    const tissue = viewer.tissue(s.data);
    if (!tissue || tissue.mask !== s.view?.mask || tissue.index !== s.view.index) {
      viewer.overlay(null);
      return;
    }
    viewer.overlay(
      decisionOverlay(
        s.descriptor.name,
        s.view.table.rows,
        s.run.labels[tissue.index] ?? {},
        s.scores,
        s.run.threshold,
        s.run.display
      )
    );
  }
  function syncLabelActions() {
    const s = session;
    const tissue = s?.data && viewer.tissue(s.data);
    const byObject = $('qc-target').value === 'object';
    const object = $('qc-object');
    $('qc-object-field').hidden = !byObject;
    const selectionSize = s?.run ? viewer.selection().size : 0;
    $('qc-target').options[0].textContent = selectionSize ? 'Selected cells (' + selectionSize + ')' : 'Selected cells';
    const hasTarget = byObject ? object.value !== '' && object.validity.valid : selectionSize > 0;
    const ready = tissue && s.view && tissue.index === s.view.index && tissue.mask === s.view.mask;
    for (const label of ['good', 'artefact', 'remove'])
      $('qc-label-' + label).disabled = !ready || Boolean(s?.job) || !hasTarget;
  }
  $('qc-target').onchange = syncLabelActions;
  $('qc-object').oninput = syncLabelActions;
  function panelTitle() {
    const title = active && tab !== 'inputs' && session?.descriptor?.name;
    $('review-panels').querySelector('.review-heading > span').textContent =
      active ? (title ? 'Mask QC · ' + title : 'Mask QC') : 'Review tools';
  }
  function sync() {
    const s = session,
      run = s?.run,
      busy = Boolean(s?.job);
    panelTitle();
    $('qc-panel').setAttribute('aria-busy', String(busy));
    for (const t of tabs) {
      t.disabled = busy || (!run && t.dataset.qcTab !== 'inputs');
      t.setAttribute('aria-pressed', String(t.dataset.qcTab === tab));
    }
    for (const page of document.querySelectorAll('[data-qc-page]')) page.hidden = page.dataset.qcPage !== tab;
    $('qc-navigation').hidden = !run || tab === 'inputs' || tab === 'display';
    for (const node of document.querySelectorAll(
      '#qc-panel input,#qc-panel select,#qc-panel button:not([data-qc-tab]),#qc-new,#qc-run,#qc-reload'
    ))
      node.disabled = busy;
    $('qc-cancel').hidden = !s?.cancellable;
    $('qc-cancel').disabled = false;
    $('qc-new-source').hidden = Boolean($('qc-source').value);
    $('qc-create').disabled = busy || (!s?.inventory && !$('qc-source').value);
    const tissue = s?.data && viewer.tissue(s.data);
    const linked = tissue && tissue.mask === s.view?.mask && tissue.index === s.view.index;
    $('qc-view-roi').textContent = tissue?.roi ?? '';
    syncLabelActions();
    $('qc-export').disabled = busy || !run || ($('qc-export-scope').value === 'current' && !linked);
    $('qc-previous-edits').hidden = !s?.previousEdits;
    if (!run) return;
    if (!busy) reviewPanel.display(s);
    if (tab === 'label') reviewPanel.list(s);
    $('qc-threshold').value = run.threshold;
    const counts = { good: 0, artefact: 0 };
    for (const group of Object.values(run.labels)) for (const v of Object.values(group)) counts[v]++;
    $('qc-counts').textContent = counts.good + ' good · ' + counts.artefact + ' artefact';
    $('qc-train').disabled = busy || !counts.good || !counts.artefact;
    let info = run.model
      ? 'Model trained on ' + run.model.examples + ' labelled objects.'
      : 'No model trained.';
    const state = !run.model ? 'untrained' : run.model.generation === run.generation ? 'current' : 'stale';
    const stateLabel = $('qc-model-state');
    stateLabel.dataset.state = state;
    stateLabel.textContent = { untrained: 'No trained classifier', current: 'Classifier up to date', stale: 'Retraining needed' }[state];
    $('qc-model-guidance').textContent = state === 'stale'
      ? 'Manual labels changed. Predictions still use the last trained classifier. Retrain to include your edits.'
      : 'Training is manual. Adding, changing or removing a label requires retraining to update predictions.';
    $('qc-train').textContent = run.model ? 'Retrain classifier' : 'Train classifier';
    if (state === 'untrained')
      $('qc-model-guidance').textContent = 'Mark at least one good cell and one artefact, then choose Train classifier.';
    if (state === 'stale' && (!counts.good || !counts.artefact))
      $('qc-model-guidance').textContent += ' Label at least one good cell and one artefact to retrain.';
    if (tab === 'model' && s.view) {
      let removed = 0;
      for (const row of s.view.table.rows)
        if (!decision(run.labels[s.roi]?.[row.id], s.scores?.get(row.id), run.threshold).keep) removed++;
      info += ' ' + removed + ' of ' + s.view.table.rows.length + ' objects marked for removal in this ROI.';
    }
    $('qc-model-info').textContent = info;
  }
  function switchTab(name) {
    tab = name;
    sync();
  }
  for (const t of tabs) t.onclick = () => switchTab(t.dataset.qcTab);
  function mode(on) {
    active = Boolean(on && session);
    space.classList.toggle('qc-open', active);
    $('qc-panel').hidden = !active;
    $('mode-review').setAttribute('aria-pressed', String(!active));
    $('mode-analysis').value = active ? 'mask-qc' : '';
    panelTitle();
    if (active) $('review-panels').open = true;
  }
  $('mode-review').onclick = () => mode(false);
  $('mode-analysis').onchange = () => {
    if ($('mode-analysis').value !== 'mask-qc') return;
    if (session?.run) show();
    else selectAnalyses();
  };
  async function action(fn, cancellable = false) {
    const s = session;
    if (!s || s.job) return;
    const controller = new AbortController();
    s.job = controller;
    s.cancellable = cancellable;
    sync();
    try {
      await fn(s, controller.signal);
    } catch (e) {
      message(s, e.name === 'AbortError' ? 'Job cancelled.' : e.message, true);
    } finally {
      s.job = null;
      s.cancellable = false;
      if (session === s) sync();
    }
  }
  $('qc-cancel').onclick = () => session?.job?.abort();
  function menus(s) {
    options($('qc-run'), [['', 'Choose an analysis'], ...s.catalog.analyses.map(a => [a.id, a.name])]);
    options($('qc-source'), [
      ['', 'New feature set from Setup'],
      ...s.catalog.sources.map(source => [
        source.id,
        source.channel +
          ' · ' +
          source.rois.length +
          ' ROIs · ' +
          source.maskFolder +
          (source.metrics ? ' + CSV' : '') +
          ' · ' +
          source.id.slice(0, 6)
      ])
    ]);
    $('qc-source').value = s.descriptor?.source ?? s.catalog.sources.at(-1)?.id ?? '';
    $('qc-run').value = s.descriptor?.id ?? '';
  }
  function connect(on, folder, project) {
    if ((on ? project.id : null) === (session?.id ?? null)) return;
    session?.job?.abort();
    mode(false);
    viewer.detach();
    session = on ? { id: project.id, folder, locals: new Map(), roi: 0 } : null;
    $('analysis-modes').hidden = $('section-analysis-compute').hidden = !session;
    $('qc-name').value = $('qc-metrics').value = '';
    options($('qc-run'), []);
    options($('qc-source'), []);
    options($('qc-dna'), []);
    $('qc-status').textContent = '';
    tab = 'inputs';
    sync();
  }
  function show() {
    if (!session) return;
    mode(true);
    if (session.catalog) return;
    action(async (s, signal) => {
      message(s, 'Opening the analysis list…');
      const catalog = await readJson(s.folder, 'analyses.json', blankCatalog());
      if (catalog.version !== 1) throw Error('Unsupported analysis catalog version.');
      signal.throwIfAborted();
      s.catalog = catalog;
      menus(s);
      if (catalog.analyses.length) await loadRun(s, s.descriptor?.id ?? catalog.analyses.at(-1).id, signal);
      else message(s, 'Choose masks and images in Setup, then use those folders here.');
    });
  }
  $('qc-reload').onclick = () => {
    if (session && !session.job) {
      session.catalog = null;
      show();
    }
  };
  async function loadRun(s, id, signal) {
    const descriptor = s.catalog.analyses.find(a => a.id === id);
    if (!descriptor) {
      menus(s);
      return;
    }
    const source = s.catalog.sources.find(item => item.id === descriptor.source);
    if (descriptor.type !== 'mask-qc' || source?.recipe !== 'mask-dna-v1')
      throw Error('Unsupported analysis or feature recipe.');
    const same = s.data?.source.id === source.id;
    const folder = await s.folder.getDirectoryHandle(descriptor.directory ?? 'run-' + id),
      run = await readJson(folder, 'run.json');
    const [data, model, draft] = await Promise.all([
      same ? s.data : openSource(s.folder, source, s.locals.get(source.id)),
      run.model ? readJson(folder, 'model-' + run.model.id + '.json') : null,
      Object.hasOwn(run, 'staged') ? savedDraft(id) : null
    ]);
    const earlier = draft?.generation === run.generation ? draft.staged : run.staged;
    const previousEdits = earlier && countLabels(earlier) ? earlier : null;
    signal.throwIfAborted();
    // An older viewer can change a run while carrying its unknown cache fields forward.
    const key = run.roiReviewFor;
    const validReview = model && key?.modelId === run.model.id &&
      key.threshold === run.threshold && key.labelGeneration === run.generation;
    Object.assign(s, { descriptor, folderRun: folder, run, data, model, previousEdits, scores: null,
      review: validReview ? { ...run.roiReview } : {} });
    if (!same) {
      viewer.detach();
      s.roi = null;
      s.view = null;
    }
    menus(s);

    switchTab('label');
    if (s.view && model) await score(s, signal);
    paint(s);
    message(
      s,
      'Opened ' +
        descriptor.name +
        '. ' +
        (s.view ? 'Current ROI retained.' : 'Choose Open in Views to label cells.')
    );
  }
  $('qc-run').onchange = () => action((s, signal) => loadRun(s, $('qc-run').value, signal));
  $('qc-new').onclick = () => {
    mode(true);
    switchTab('inputs');
    $('qc-name').focus();
  };
  $('qc-source').onchange = sync;
  $('qc-read-setup').onclick = () =>
    action(async s => {
      const folders = setupFolders(),
        inventory = sourceInventory(folders.masks, folders.images);
      s.inventory = inventory;
      s.folders = folders;
      options(
        $('qc-dna'),
        inventory.channels.map(c => [c, c])
      );
      $('qc-dna').value = inventory.channels.find(c => /dna|dapi|hoechst/i.test(c)) ?? inventory.channels[0];
      message(
        s,
        'Matched ' + inventory.map.size + ' mask ROIs. Choose the DNA channel used for this feature set.'
      );
    });
  $('qc-create').onclick = () =>
    action(async (s, signal) => {
      const name = $('qc-name').value.trim();
      if (!name) throw Error('Enter an analysis name.');
      if (s.catalog.analyses.some(a => a.name === name)) throw Error('Choose a distinct analysis name.');
      let source = s.catalog.sources.find(item => item.id === $('qc-source').value);
      const sources = [...s.catalog.sources];
      if (!source) {
        if (!s.inventory) throw Error('Select the source folders first.');
        const file = $('qc-metrics').files[0],
          metrics = file ? readMetrics(await file.text()) : null;
        source = buildSource(
          s.inventory,
          $('qc-dna').value,
          s.folders,
          metrics ? { name: file.name, columns: metrics.columns } : null
        );
        if (metrics && !source.rois.some(r => metrics.groups.has(r.name)))
          throw Error('No ROI in this metrics CSV matches the selected masks.');
        const folder = await s.folder.getDirectoryHandle('features-' + source.id, { create: true });
        if (metrics)
          for (const [index, roi] of source.rois.entries()) {
            signal.throwIfAborted();
            if (metrics.groups.has(roi.name))
              await writeJson(folder, 'metrics-' + index + '.json', metrics.groups.get(roi.name));
          }
        const { masks, images } = s.folders;
        await rememberInputs(source.id, { masks: masks.workspaceHandle, images: images.workspaceHandle });
        s.locals.set(source.id, { masks: localInputMap(masks), images: localInputMap(images) });
        sources.push(source);
      }
      signal.throwIfAborted();
      const descriptor = { id: crypto.randomUUID(), name, type: 'mask-qc', source: source.id };
      const safeName =
        name
          .replace(/[^\p{L}\p{N}]+/gu, '-')
          .slice(0, 48)
          .replace(/^-|-$/g, '') || 'mask-qc';
      descriptor.directory = 'run-' + safeName + '-' + descriptor.id;
      const folder = await s.folder.getDirectoryHandle(descriptor.directory, { create: true });
      await writeJson(folder, 'run.json', blankRun());
      s.catalog = await commit(
        s.folder,
        'analyses.json',
        { ...s.catalog, sources, analyses: [...s.catalog.analyses, descriptor] },
        s.id
      );
      signal.throwIfAborted();
      $('qc-name').value = $('qc-metrics').value = '';
      await loadRun(s, descriptor.id, signal);
    });
  async function saveRun(s, changes) {
    const next = { ...s.run, roiReview: { ...s.review }, ...changes };
    next.roiReviewFor = { modelId: next.model?.id ?? null, threshold: next.threshold, labelGeneration: next.generation };
    if (Object.hasOwn(changes, 'labels')) delete next.staged;
    s.run = await commit(s.folderRun, 'run.json', next, s.id);
    if (Object.hasOwn(changes, 'labels')) s.previousEdits = null;
  }
  for (const kind of ['masks', 'images'])
    $('qc-reconnect-' + kind).onclick = () =>
      action(async (s, signal) => {
        const expected = kind === 'masks' ? s.data.source.maskFolder : s.data.source.imageFolder;
        message(s, 'Choose ' + expected + ' for this saved feature set.');
        const handle = await window.showDirectoryPicker({ id: 'qc-' + kind, mode: 'read' });
        if (handle.name !== expected) throw Error('Choose the saved folder “' + expected + '”.');
        signal.throwIfAborted();
        const handles = { ...s.data.handles, [kind]: handle };
        await rememberInputs(s.data.source.id, handles);
        s.data.handles = handles;
        delete s.data.local[kind];
        viewer.detach();
        message(s, 'Reconnected ' + handle.name + '. Choose Open in Views to use this folder.');
      });
  async function score(s, signal) {
    s.scores = null;
    if (!s.model || !s.view) return;
    const view = s.view,
      model = s.model;
    const values = await computeJob({ op: 'score', model, table: view.table }, signal);
    if (s.view === view && s.model === model) {
      s.scores = new Map(view.table.rows.map((r, i) => [r.id, values[i]]));
      s.review[view.index] = roiScoreSummary(s.scores, s.run.labels[view.index] ?? {}, s.run.threshold);
    }
  }
  function unlinked() {
    if (!session) return;
    session.following?.controller.abort();
    Object.assign(session, { following: null, view: null, scores: null, roi: null });
    sync();
  }
  async function follow() {
    const s = session;
    if (!s?.data) return;
    const tissue = viewer.tissue(s.data);
    if (!tissue) {
      if (s.view || s.following) unlinked();
      if (viewer.linked(s.data)) message(s,viewer.roi()+' has no matching mask in this analysis. Choose another ROI.');
      return;
    }
    if (s.following?.mask === tissue.mask && s.following.index === tissue.index) return s.following.promise;
    if (s.view?.mask === tissue.mask && s.view.index === tissue.index) {
      paint(s);
      sync();
      return;
    }
    unlinked();
    const { index, mask, roi } = tissue,
      controller = new AbortController(),
      signal = controller.signal;
    const task = { index, mask, controller };
    s.following = task;
    const current = () => session === s && s.following === task && viewer.tissue(s.data)?.mask === mask;
    task.promise = (async () => {
      message(s, 'Preparing Mask QC for ' + roi + '…');
      const table = await roiFeatures(s.data, index, { ...resources(), tissue, signal });
      if (!current()) return;
      s.view = { index, mask, table };
      s.roi = index;
      await score(s, signal);
      if (!current()) return;
      paint(s);
      message(
        s,
        roi +
          ' · ' +
          table.rows.length +
          ' objects.' +
          (s.data.source.rois[index].dna ? '' : ' DNA missing. Unscored objects are kept.')
      );
    })()
      .catch(error => {
        if (current()) {
          s.view = null;
          s.roi = null;
          message(s, error.message, true);
        }
      })
      .finally(() => {
        if (s.following === task) {
          s.following = null;
          sync();
        }
      });
    return task.promise;
  }
  async function openViews(s, signal, roi) {
    await viewer.attach(s.data, signal, roi);
    signal.throwIfAborted();
    viewer.selectTool();
    await follow();
  }
  $('qc-open-roi').onclick = () => action(openViews);
  $('qc-export-scope').onchange = sync;
  async function applyEdits(s, edits, importing = false) {
    const labels = { ...s.run.labels };
    let changed = 0;
    const affected = [];
    for (const [index, editsForRoi] of Object.entries(edits)) {
      const before = changed;
      const group = { ...labels[index] };
      for (const [id, value] of Object.entries(editsForRoi)) {
        if (value === 'clear') {
          if (!Object.hasOwn(group, id)) continue;
          delete group[id];
        } else {
          if (group[id] === value) continue;
          group[id] = value;
        }
        changed++;
      }
      if (changed !== before) affected.push(index);
      if (Object.keys(group).length) labels[index] = group;
      else delete labels[index];
    }
    if (changed || importing) {
      const roiReview = { ...s.review };
      for (const index of affected) {
        delete roiReview[index];
        if (+index === s.view?.index && s.scores)
          roiReview[index] = roiScoreSummary(s.scores, labels[index] ?? {}, s.run.threshold);
      }
      await saveRun(s, { labels, roiReview, generation: s.run.generation + (changed ? 1 : 0) });
      for (const index of affected) {
        if (roiReview[index]) s.review[index] = roiReview[index];
        else delete s.review[index];
      }
    }
    return changed;
  }
  async function labelCells(s, signal, label) {
    const byObject = $('qc-target').value === 'object';
    const selections = byObject
      ? [[s.data.source.rois[s.roi].name, [Number($('qc-object').value)]]]
      : [...viewer.selection().rois].filter(([, ids]) => ids.size).map(([roi, ids]) => [roi, [...ids]]);
    const clearMarked = !byObject && clearAfterMark.checked;
    const edits = {};
    message(s, 'Saving labels…');
    for (const [roi, ids] of selections) {
      const index = s.data.roiIndex.get(roi);
      if (index === undefined) throw Error('Selection includes an ROI outside this analysis: ' + roi + '.');
      // Removing a manual label only needs its saved ROI and object number.
      if (label !== 'clear') {
        const features = await table(s, index, { ...resources(), signal });
        const missing = new Set(ids);
        for (const row of features.rows) {
          missing.delete(row.id);
          if (!missing.size) break;
        }
        if (missing.size) {
          const [id] = missing;
          throw Error('Object ' + id + ' is absent from the analysis mask for ' + roi + '.');
        }
      }
      edits[index] = Object.fromEntries(ids.map(id => [id, label]));
    }
    signal.throwIfAborted();
    const changed = await applyEdits(s, edits);
    if (clearMarked && session === s) viewer.deselect(selections);
    if (changed || clearMarked) paint(s);
    const cells = changed === 1 ? ' cell' : ' cells';
    if (!changed) message(s, 'No labels changed.');
    else if (label === 'clear') message(s, 'Removed ' + changed + ' manual label' + (changed === 1 ? '.' : 's.'));
    else message(s, 'Marked ' + changed + cells + ' as ' + label + '.');
  }
  for (const [button, label] of [['good', 'good'], ['artefact', 'artefact'], ['remove', 'clear']])
    $('qc-label-' + button).onclick = () => action((s, signal) => labelCells(s, signal, label));
  $('qc-import-edits').onclick = () =>
    action(async s => {
      if (!s.previousEdits) return;
      await applyEdits(s, s.previousEdits, true);
      paint(s);
      message(s, 'Earlier label edits applied.');
    });
  $('qc-threshold').onchange = event => {
    const value = event.target.valueAsNumber;
    return action(async s => {
      if (!Number.isFinite(value) || value < 0 || value > 1) throw Error('Enter a threshold between 0 and 1.');
      if (value !== s.run.threshold) {
        const roiReview = {};
        if (s.view && s.scores) roiReview[s.view.index] = roiScoreSummary(s.scores, s.run.labels[s.view.index] ?? {}, value);
        await saveRun(s, { threshold: value, roiReview });
        s.review = roiReview;
      }
      paint(s);
      message(s, 'Preview updated.');
    });
  };
  async function table(s, index, settings) {
    return s.view?.index === index ? s.view.table : roiFeatures(s.data, index, settings);
  }
  $('qc-score-rois').onclick = () =>
    action(async (s, signal) => {
      const settings = { ...resources(), signal };
      const indices = s.data.source.rois.flatMap((r, i) => r.dna && !s.review[i] ? [i] : []);
      let done = 0;
      message(s, 'Scoring ROIs…');
      await mapROIs(indices, settings.workers, async index => {
        signal.throwIfAborted();
        const measured = await table(s, index, settings);
        s.review[index] = await computeJob({ op: 'roi-summary', model: s.model, table: measured,
          labels: s.run.labels[index] ?? {}, threshold: s.run.threshold }, signal);
        message(s, 'Scored ' + ++done + ' / ' + indices.length + ' ROIs…');
      });
      signal.throwIfAborted();
      await saveRun(s, {});
      message(s, 'ROI scores saved for the current classifier and threshold.');
    }, true);
  $('qc-extract').onclick = () =>
    action(async (s, signal) => {
      const settings = { ...resources(), signal, ensure: true },
        rois = s.data.source.rois;
      let done = 0;
      message(s, 'Preparing features for ' + rois.length + ' ROIs…');
      await mapROIs(rois, settings.workers, async (_, index) => {
        signal.throwIfAborted();
        await table(s, index, settings);
        message(s, 'Features ready · ' + ++done + ' / ' + rois.length + ' ROIs');
      });
    }, true);
  $('qc-train').onclick = () =>
    action(async (s, signal) => {
      const features = [],
        labels = [];
      let columns;
      message(s, 'Reading labelled examples…');
      for (const [index, group] of Object.entries(s.run.labels)) {
        if (!Object.keys(group).length) continue;
        const data = await table(s, Number(index), { ...resources(), signal });
        columns = data.columns;
        for (const row of data.rows)
          if (group[row.id] && row.values.some(Number.isFinite)) {
            features.push(row.values);
            labels.push(Number(group[row.id] === 'artefact'));
          }
      }
      message(s, 'Training histogram gradient boosting on ' + labels.length + ' cells…');
      const model = await computeJob({ op: 'train', features, labels, columns }, signal);
      const id = crypto.randomUUID();
      await writeJson(s.folderRun, 'model-' + id + '.json', model);
      signal.throwIfAborted();
      await saveRun(s, { model: { id, generation: s.run.generation, examples: model.examples }, roiReview: {} });
      s.review = {};
      s.model = model;
      await score(s, signal);
      paint(s);
      message(s, 'Trained model saved. Predictions now use the updated classifier.');
    }, true);
  $('qc-export').onclick = () =>
    action(async (s, signal) => {
      const settings = { ...resources(), signal },
        source = s.data.source;
      const indices = $('qc-export-scope').value === 'all' ? source.rois.map((_, i) => i) : [s.roi];
      const name =
        'export-' + new Date().toISOString().replace(/[:.]/g, '-') + '-' + crypto.randomUUID().slice(0, 8);
      const folder = await s.folderRun.getDirectoryHandle(name, { create: true });
      const manifest = {
        version: 1,
        analysis: s.descriptor,
        recipe: source.recipe,
        model: s.run.model,
        threshold: s.run.threshold,
        labelRevision: s.run.generation,
        rois: []
      };
      await writeJson(folder, 'export.json', { ...manifest, status: 'incomplete' });
      message(s, 'Writing cleaned masks and decisions…');
      await mapROIs(indices, settings.workers, async index => {
        signal.throwIfAborted();
        const roi = source.rois[index];
        const measured =
          s.view?.index === index
            ? s.view
            : s.model
              ? await roiFeatures(s.data, index, { ...settings, retainMask: true })
              : {};
        const decodedMask = measured.mask;
        const result = await computeJob(
          {
            op: 'export',
            roi: roi.name,
            mask: decodedMask ? null : await asset(s.data, 'masks', roi.mask),
            decodedMask,
            table: measured.table,
            model: s.model,
            threshold: s.run.threshold,
            labels: s.run.labels[index] ?? {},
            maxPixels: settings.maxPixels
          },
          signal
        );
        const filename = roi.mask.split('/').at(-1);
        await writeFile(folder, filename, result.bytes);
        await writeFile(folder, roi.name + '.qc.csv', result.csv);
        manifest.rois.push({
          name: roi.name,
          file: filename,
          objects: result.objects,
          removed: result.removed
        });
        message(s, 'Exported ' + manifest.rois.length + ' / ' + indices.length + ' ROIs.');
      });
      signal.throwIfAborted();
      await writeJson(folder, 'export.json', { ...manifest, status: 'complete' });
      message(
        s,
        'Saved ' + indices.length + ' ROI(s) in ' + s.folder.name + '/' + s.folderRun.name + '/' + name + '.'
      );
    }, true);
  return {
    connect,
    show,
    follow,
    unlinked,
    review() {
      if (active) mode(false);
    },
    loading() {
      if (session?.run) sync();
    },
    selectionChanged: syncLabelActions
  };
}
