import { readFileSync } from 'node:fs';

// Ship the same classic script with every Vite explorer; no runtime CDN dependency.
export function usageComponentPlugin() {
  const source = () => readFileSync(new URL('./client/usage.js', import.meta.url), 'utf8');
  return {
    name: 'sbt-usage-component',
    configureServer(server) {
      server.middlewares.use('/usage.js', (_req, res) => {
        res.setHeader('Content-Type', 'text/javascript');
        res.end(source());
      });
    },
    generateBundle() { this.emitFile({ type: 'asset', fileName: 'usage.js', source: source() }); },
  };
}
