const markerName = 'imc-explorer.analysis.json';
const format = 'imc-explorer-analysis-project';
const writeAccess = { mode: 'readwrite' };

export function validateAnalysisProject(value) {
  if (!value || typeof value.id !== 'string' || !/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(value.id)
      || typeof value.folderName !== 'string' || !value.folderName.trim() || value.folderName.length > 255) {
    throw new Error('Invalid workspace analysis folder reference.');
  }
  return { id: value.id, folderName: value.folderName };
}

// Only the explicitly chosen folder is writable. Read one known file on reconnect.
export async function openAnalysisFolder(directory, projectId, { create = false, adopt = false } = {}) {
  let handle;
  try {
    handle = await directory.getFileHandle(markerName);
  } catch (error) {
    if (error.name !== 'NotFoundError') throw error;
    if (!create) throw new Error('Choose the analysis folder containing ' + markerName + ' for this workspace.');
    handle = await directory.getFileHandle(markerName, { create: true });
  }
  // A first-time connection interrupted before close can leave an empty marker.
  let file = await handle.getFile();
  if (create && file.size === 0) {
    const writer = await handle.createWritable();
    try {
      await writer.write(JSON.stringify({ format, version: 1, id: projectId }, null, 2) + '\n');
      await writer.close();
    } catch (failure) {
      await writer.abort().catch(() => {});
      throw failure;
    }
    file = await handle.getFile();
  }
  let marker;
  if (file.size <= 4096) {
    try { marker = JSON.parse(await file.text()); } catch (error) { if (!(error instanceof SyntaxError)) throw error; }
  }
  if (marker?.format !== format || marker.version !== 1) throw new Error('This folder contains an unreadable or unsupported IMC Explorer analysis project. Choose another folder.');
  if (!adopt && marker.id !== projectId) throw new Error('This analysis folder belongs to another project. Choose the folder for this workspace.');
  return validateAnalysisProject({ id: marker.id, folderName: directory.name });
}

export function createAnalysisFolderPanel($, { current, save, run, connected }) {
  const choose = $('choose-analysis-folder'), reconnect = $('reconnect-analysis-folder'), disconnect = $('disconnect-analysis-folder');
  const status = $('analysis-folder-status');
  let ready = false, locked = true, generation = 0, pending = false;
  const supported = typeof window.showDirectoryPicker === 'function';

  function render(message, error = false) {
    const record = current(), project = record?.analysisProject;
    choose.textContent = project ? 'Locate folder' : 'Choose folder';
    choose.hidden = !supported || ready;
    reconnect.hidden = !supported || !record?.analysisHandle || ready;
    disconnect.hidden = !project;
    const existing = $('open-analysis-project');
    existing.hidden = !supported || Boolean(project);
    for (const button of [choose, reconnect, disconnect, existing]) button.disabled = !record || locked || pending;
    if (message !== undefined) {
      status.textContent = message;
      status.dataset.state = error ? 'error' : '';
    }
    $('analysis-project-name').textContent = ready ? record?.analysisHandle?.name ?? '' : '';
    connected(ready && Boolean(project), record?.analysisHandle, project);
  }

  async function restore() {
    const ticket = ++generation, record = current();
    ready = false;
    if (!supported) { render('Use desktop Chrome or Edge to connect a writable analysis folder.'); return; }
    if (!record?.analysisProject) { render('No analysis folder connected.'); return; }
    const name = record.analysisProject.folderName;
    if (!record.analysisHandle) { render('Locate “' + name + '” to open this project.'); return; }
    render('Checking “' + name + '”…');
    try {
      const handle = record.analysisHandle;
      if (await handle.queryPermission(writeAccess) !== 'granted') {
        if (ticket === generation) render('Reconnect “' + name + '” to allow read and write access.');
        return;
      }
      await openAnalysisFolder(handle, record.analysisProject.id);
      if (ticket === generation) { ready = true; render('Connected · ' + handle.name); }
    } catch (error) { if (ticket === generation) render(error.message, true); }
  }

  function change(action) {
    if (locked || pending || !current()) return;
    pending = true;
    ++generation;
    render();
    run(async () => {
      try { await action(current()); }
      catch (error) {
        if (error.name !== 'AbortError') render(error.name === 'NotAllowedError' ? 'Write access was not granted. Choose or reconnect the folder to try again.' : error.message, true);
      } finally { pending = false; render(); }
    });
  }
  async function connect(record, handle, adopt = false) {
    if (await handle.queryPermission(writeAccess) !== 'granted' && await handle.requestPermission(writeAccess) !== 'granted') {
      throw new Error('Write access was not granted. Reconnect the folder to try again.');
    }
    const id = record.analysisProject?.id ?? record.analysisOwnerId ?? record.id;
    const open = () => openAnalysisFolder(handle, id, { create: !record.analysisProject && !adopt, adopt });
    const project = navigator.locks ? await navigator.locks.request('imc-explorer-analysis-folder', open) : await open();
    await save(record.id, { analysisOwnerId: project.id, analysisProject: project, analysisHandle: handle });
    ready = true;
    render('Connected · ' + handle.name);
  }
  choose.onclick = () => change(async record => {
    const handle = await window.showDirectoryPicker({ id: 'imc-analysis-folder', mode: 'readwrite' });
    await connect(record, handle);
  });
  $('open-analysis-project').onclick = () => change(async record => {
    const handle = await window.showDirectoryPicker({ id: 'imc-analysis-folder', mode: 'readwrite' });
    await connect(record, handle, true);
  });
  reconnect.onclick = () => change(async record => {
    if (await record.analysisHandle.requestPermission(writeAccess) !== 'granted') throw new Error('Write access was not granted. Reconnect the folder to try again.');
    await connect(record, record.analysisHandle);
  });
  disconnect.onclick = () => change(async record => {
    await save(record.id, { analysisProject: null, analysisHandle: null });
    ready = false;
    render('Folder disconnected. Saved files are still in the folder.');
  });
  async function checkPermission() {
    if (!ready || pending) return;
    const record = current(), ticket = generation;
    try {
      if (await record.analysisHandle.queryPermission(writeAccess) === 'granted') return;
    } catch { /* The saved directory may have been removed or access revoked. */ }
    if (!record || ticket !== generation || record.id !== current()?.id) return;
    ready = false;
    render('Reconnect “' + record.analysisProject.folderName + '” to allow read and write access.');
  }
  window.addEventListener('focus', checkPermission);
  $('tab-analyses').addEventListener('click', checkPermission);
  return {
    restore,
    controls(value) { locked = value; render(); },
  };
}
