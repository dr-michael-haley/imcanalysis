import test from 'node:test';
import assert from 'node:assert/strict';
import { folderReference, sameFolder, sameFile, relativeFilePath, resolveWorkspaceRoot } from '../src/workspace.js';

test('folder references tolerate relocation and ordering, but reject missing, renamed or resized TIFFs', async () => {
  const files = [
    { name: 'CD3.tiff', size: 100, webkitRelativePath: 'images/ROI_A/CD3.tiff' },
    { name: 'DNA.tiff', size: 200, webkitRelativePath: 'images/ROI_A/DNA.tiff' },
  ];
  const expected = await folderReference(files);
  const relocated = files.map(file => ({ ...file, webkitRelativePath: file.webkitRelativePath.replace('images/', 'renamed/') })).reverse();
  assert.equal(sameFolder(expected, await folderReference(relocated)), true);
  assert.equal(sameFolder(expected, await folderReference(files.slice(1))), false);
  assert.equal(sameFolder(expected, await folderReference([{ ...files[0], size: 101 }, files[1]])), false);
  assert.equal(sameFolder(expected, await folderReference([{ ...files[0], webkitRelativePath: 'images/ROI_B/CD3.tiff' }, files[1]])), false);
  assert.equal(await folderReference([]), null);
  assert.equal(relativeFilePath(files[0]), 'ROI_A/CD3.tiff');
  assert.equal(sameFile({ name: 'data.h5ad', size: 100 }, { name: 'other.h5ad', size: 100 }), false);
});

test('one root matches named folders only and retains exact paths for future reconnects', async () => {
  const local = (path, size) => {
    const file = new File([new Uint8Array(size)], path.split('/').at(-1));
    Object.defineProperty(file, 'webkitRelativePath', { value: path }); return file;
  };
  const images = [local('images/ROI_A/CD3.tiff', 100)], masks = [local('masks/ROI_A.tiff', 30)];
  const sources = { anndata: { name: 'data.h5ad', size: 20 }, images: await folderReference(images), masks: await folderReference(masks) };
  const files = [local('root/analysis/data.h5ad', 20), local('root/nested/images/ROI_A/CD3.tiff', 100), local('root/masks/ROI_A.tiff', 30)];
  const resolved = await resolveWorkspaceRoot(files, sources);
  assert.equal(resolved.anndata, files[0]);
  assert.equal(relativeFilePath(resolved.images[0]), 'ROI_A/CD3.tiff');
  assert.equal(sameFolder(sources.masks, await folderReference(resolved.masks)), true);
  assert.equal((await folderReference(resolved.images)).path, 'nested/images');
  await assert.rejects(resolveWorkspaceRoot(files.slice(1), sources), /Could not find data.h5ad/);
  await assert.rejects(resolveWorkspaceRoot([...files, local('root/copy/data.h5ad', 20)], sources), /Multiple copies/);
  await assert.rejects(resolveWorkspaceRoot([...files, local('root/other/images/ROI_A/CD3.tiff', 100)], sources), /Multiple matching Images/);
  await assert.rejects(resolveWorkspaceRoot([files[0], files[2], local('root/renamed/ROI_A/CD3.tiff', 100)], sources), /Could not find the saved Images/);
  await assert.rejects(resolveWorkspaceRoot([files[0], files[2], local('root/nested/images/ROI_A/CD3.tiff', 101)], sources), /Could not find the saved Images/);
  sources.images.path = 'nested/images'; sources.anndata.path = 'analysis/data.h5ad';
  assert.equal((await resolveWorkspaceRoot([...files, local('root/other/images/ROI_A/CD3.tiff', 100), local('root/copy/data.h5ad', 20)], sources)).anndata, files[0]);
});
