import h5wasm from 'h5wasm';
import { AnnDataReader } from './anndata.js';
import { readPlot } from './plot-reader.js';
import { representativeMarkers, roiMetadata } from './review-data.js';
import { readEmbeddingContext, readEmbeddingCurrent } from './embedding-reader.js';

let file;
let reader;
let mounted = false;
const ready = h5wasm.ready.then(module => {
  const { FS } = module;
  FS.mkdir('/input');
  return module;
});

let queue = Promise.resolve();
self.onmessage = ({ data: { id, method, args } }) => {
  queue = queue.then(async () => {
    try {
      const { FS } = await ready;
      let result;
      if (method === 'open') {
        file?.close(); file = null; reader = null;
        if (mounted) { FS.unmount('/input'); mounted = false; }
        FS.mount(FS.filesystems.WORKERFS, { files: [args[0]] }, '/input');
        mounted = true;
        file = new h5wasm.File(`/input/${args[0].name}`, 'r');
        reader = new AnnDataReader(file);
        result = reader.header();
      } else {
        if (!reader) throw new Error('Open an AnnData file first.');
        if (method === 'plotSummary') result = readPlot(reader, args[0], progress => self.postMessage({ id, progress }));
        else if (method === 'embeddingContext') result = readEmbeddingContext(reader, args[0], progress => self.postMessage({ id, progress }));
        else if (method === 'embeddingCurrent') result = readEmbeddingCurrent(reader, args[0]);
        else if (method === 'representativeMarkers') result = representativeMarkers(reader, args[0], progress => self.postMessage({ id, progress }));
        else if (method === 'roiMetadata') result = roiMetadata(reader, args[0], progress => self.postMessage({ id, progress }));
        else if (method === 'configure') result = reader.configure(...args, progress => self.postMessage({ id, progress }));
        else {
          if (!['validateFlagCells', 'region', 'cell', 'categories', 'rank', 'populationSummary'].includes(method)) throw new Error('Unknown read operation.');
          result = reader[method](...args);
        }
      }
      self.postMessage({ id, result });
    } catch (error) {
      self.postMessage({ id, error: error.message ?? String(error) });
    }
  });
};
