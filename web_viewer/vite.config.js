import { defineConfig } from 'vite';
import { runtimeLicensePlugin } from './licenses.js';
import { readFileSync } from 'node:fs';

const { version } = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8'));

export default defineConfig({
  base: './',
  define: { __VIEWER_VERSION__: JSON.stringify(version) },
  plugins: [runtimeLicensePlugin()],
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    outDir: '../SpatialBiologyToolkit/browser_viewer/static',
    emptyOutDir: true,
  },
});
