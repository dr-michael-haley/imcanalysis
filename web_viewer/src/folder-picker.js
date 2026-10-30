// Retain handles so root.resolve() can discover ancestors without crawling them.
export async function readPickedFolder(handle, accepts, progress = () => {}) {
  const files = []; let visited = 0;
  async function walk(directory, path) {
    for await (const [name, child] of directory.entries()) {
      if (child.kind === 'directory') await walk(child, `${path}/${name}`);
      else if (accepts({ name })) {
        const file = await child.getFile();
        Object.defineProperty(file, 'webkitRelativePath', { value: `${path}/${name}` }); files.push(file);
      }
      if (++visited % 100 === 0) { progress(`Reading selected folder · ${files.length.toLocaleString()} files…`); await new Promise(resolve => setTimeout(resolve, 0)); }
    }
  }
  await walk(handle, handle.name); files.workspaceHandle = handle; return files;
}
export function bindFolderPicker(input, accepts, receive, error) {
  input.addEventListener('change', () => receive(Array.from(input.files)));
  if (!window.showDirectoryPicker) return;
  const label = input.closest('label'), button = document.createElement('button'), fallback = document.createElement('details'), summary = document.createElement('summary');
  const name = input.id === 'mask-files' ? 'Masks' : 'Images', status = label.querySelector('span');
  const caption = () => input.id === 'aligned-files' ? '+ Add layer folder' : `${name} · ${status?.textContent || 'Choose folder'}`;
  button.className = 'folder-handle-picker'; button.textContent = caption();
  button.setAttribute('aria-label', input.getAttribute('aria-label') || label.textContent.trim());
  button.title = 'Remember the location relative to your connected workspace folder';
  summary.textContent = 'Compatibility folder chooser'; fallback.className = 'folder-picker-fallback';
  label.before(button, fallback); fallback.append(summary, label);
  if (status) new MutationObserver(() => { if (!button.disabled) button.textContent = caption(); }).observe(status, { childList: true, characterData: true, subtree: true });
  button.onclick = async () => {
    try {
      const handle = await window.showDirectoryPicker({ mode: 'read', id: input.id });
      button.disabled = true; button.textContent = 'Reading selected folder…';
      const files = await readPickedFolder(handle, accepts, message => { button.textContent = message; });
      await receive(files);
    } catch (failure) { if (failure.name !== 'AbortError') error(failure); }
    finally { button.disabled = false; button.textContent = caption(); }
  };
}

export function bindFilePicker(input, extension, title, receive, error) {
  input.addEventListener('change', () => { if (input.files[0]) receive(input.files[0]); });
  if (!window.showOpenFilePicker) return;
  const label = input.closest('label'), button = document.createElement('button'), fallback = document.createElement('details'), summary = document.createElement('summary'), status = label.querySelector('span');
  const caption = () => status ? `${title} · ${status.textContent}` : title;
  button.className = 'folder-handle-picker'; button.textContent = caption(); button.setAttribute('aria-label', title);
  summary.textContent = 'Compatibility file chooser'; fallback.className = 'folder-picker-fallback'; label.before(button, fallback); fallback.append(summary, label);
  if (status) new MutationObserver(() => { if (!button.disabled) button.textContent = caption(); }).observe(status, { childList: true, characterData: true, subtree: true });
  button.onclick = async () => {
    try {
      const [handle] = await window.showOpenFilePicker({ multiple: false, types: [{ description: title, accept: { 'application/octet-stream': [extension] } }] });
      button.disabled = true; const file = await handle.getFile(); file.workspaceHandle = handle; await receive(file);
    } catch (failure) { if (failure.name !== 'AbortError') error(failure); }
    finally { button.disabled = false; button.textContent = caption(); }
  };
}
