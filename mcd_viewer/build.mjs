// Package only browser assets. Never publish the repository or a data directory.
import { copyFile, mkdir, readdir, realpath, rm, lstat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, relative, isAbsolute, extname } from 'node:path';

const root = await realpath(fileURLToPath(new URL('.', import.meta.url)));
const output = join(root, 'dist');
const assets = ['index.html', 'THIRD-PARTY-NOTICES.txt', '_headers'];
async function collect(directory) {
  for (const entry of await readdir(join(root, directory), { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Symlinks are not deployable: ${path}`);
    if (entry.isDirectory()) await collect(path);
    else if (['.js', '.css', '.svg'].includes(extname(path))) assets.push(path);
    else throw new Error(`Unexpected browser asset: ${path}`);
  }
}
await collect('src');
// Resolve and check the exact generated directory before removing an older build.
try {
  if ((await lstat(output)).isSymbolicLink()) throw new Error('The build directory must not be a symlink.');
  const target = await realpath(output), rel = relative(root, target);
  if (rel !== 'dist' || isAbsolute(rel)) throw new Error('Build directory is outside the expected location.');
  await rm(target, { recursive: true });
} catch (error) { if (error.code !== 'ENOENT') throw error; }
for (const asset of assets) {
  const destination = join(output, asset);
  await mkdir(join(destination, '..'), { recursive: true });
  await copyFile(join(root, asset), destination);
}
console.log(`Packaged ${assets.length} app assets in ${output}. No datasets or server code included.`);
