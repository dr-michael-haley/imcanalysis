// Include worker dependencies too: Vite's main-bundle license collector does
// not see libraries that are only imported by a separately bundled worker.
import { existsSync, readFileSync, readdirSync, realpathSync } from 'node:fs';
import { dirname, join, parse } from 'node:path';
import { fileURLToPath } from 'node:url';

export function runtimeLicensePlugin() {
  return {
    name: 'sbt-runtime-licenses',
    generateBundle() {
      const project = dirname(fileURLToPath(import.meta.url));
      const visited = new Set();
      const sections = ['# Third-party software\n\nThis viewer uses h5wasm from the National Institute of Standards and Technology, the HDF5 library, GeoTIFF.js, and the runtime dependencies listed below.'];
      const locate = (name, from) => {
        for (let path = from; path !== parse(path).root; path = dirname(path)) {
          const candidate = join(path, 'node_modules', name);
          if (existsSync(join(candidate, 'package.json'))) return realpathSync(candidate);
        }
        throw new Error(`Cannot locate license dependency ${name}`);
      };
      const visit = (name, from) => {
        const root = locate(name, from);
        if (visited.has(root)) return;
        visited.add(root);
        const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
        const licenses = readdirSync(root).filter(file => /^(licen[sc]e|copying|notice)(\.|$)/i.test(file));
        let notices = licenses.map(file => readFileSync(join(root, file), 'utf8')).join('\n\n');
        // lerc's npm archive includes the notice in its source, but omits LICENSE.
        if (!notices && pkg.name === 'lerc' && pkg.license === 'Apache-2.0') {
          const source = readFileSync(join(root, pkg.main), 'utf8');
          const notice = source.match(/\/\*\s*Copyright[\s\S]*?\*\//)?.[0];
          if (!notice) throw new Error('LERC copyright notice is missing.');
          notices = `${notice}\n\n${readFileSync(join(project, 'licenses/Apache-2.0.txt'), 'utf8')}`;
        }
        if (!notices) throw new Error(`Missing license text for ${pkg.name}`);
        sections.push(`## ${pkg.name} ${pkg.version}\n\n${notices}`);
        for (const dependency of Object.keys(pkg.dependencies ?? {})) visit(dependency, root);
      };
      for (const name of ['h5wasm', 'geotiff']) visit(name, project);
      this.emitFile({ type: 'asset', fileName: 'THIRD_PARTY_LICENSES.md', source: sections.join('\n\n---\n\n') });
    },
  };
}
