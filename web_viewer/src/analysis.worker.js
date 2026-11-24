import { openTiffPlane, readScalarPlane } from './tiff-plane.js';
import { measureObjects, cleanedTiff, decision, roiScoreSummary } from './mask-qc.js';
import { trainBinary, predictBinary } from './analysis-model.js';

async function raster(file, maxPixels, mask) {
  const { image, width, height } = await openTiffPlane(
    file,
    0,
    file.name + ' must be a single-plane TIFF.',
    (w, h) => {
      if (w * h > maxPixels) throw Error(file.name + ' exceeds the pixel buffer budget. Adjust Setup 05.');
    }
  );
  const data = await readScalarPlane(image);
  if (mask)
    for (const v of data)
      if (!Number.isInteger(v) || v < 0 || v > 0xffffffff)
        throw Error(file.name + ' contains invalid object IDs.');
  return { width, height, data };
}

async function execute(job) {
  if (job.op === 'train') return trainBinary(job.features, job.labels, job.columns);
  if (job.op === 'roi-summary') {
    const scores = job.table.rows.map(row => [row.id, predictBinary(job.model, row.values)]);
    return roiScoreSummary(scores, job.labels, job.threshold);
  }
  if (job.op === 'score') return job.table.rows.map(row => predictBinary(job.model, row.values));
  const mask = job.decodedMask ?? (await raster(job.mask, job.maxPixels, true));
  if (mask.width * mask.height > job.maxPixels)
    throw Error('This ROI exceeds the analysis pixel buffer budget in Setup 05.');
  if (job.op === 'export') {
    const byId = new Map(job.table?.rows.map(r => [r.id, r.values]));
    const ids = new Set(mask.data);
    ids.delete(0);
    const remove = new Set(),
      lines = ['ROI,ObjectNumber,confirmed_label,artefact_score,threshold,keep,decision_source'];
    const roi = '"' + job.roi.replaceAll('"', '""') + '"';
    for (const id of ids) {
      const label = job.labels[id] || '',
        score = byId.has(id) ? predictBinary(job.model, byId.get(id)) : null;
      const { keep, reason } = decision(label, score, job.threshold);
      if (!keep) remove.add(id);
      lines.push([roi, id, label, score ?? '', job.threshold, Number(keep), reason].join(','));
    }
    return {
      bytes: cleanedTiff(mask, remove),
      csv: lines.join('\r\n') + '\r\n',
      objects: ids.size,
      removed: remove.size
    };
  }
  if (!job.dna && !job.decodedDNA) {
    const ids = new Set(mask.data);
    ids.delete(0);
    const result = {
      table: {
        columns: job.columns,
        rows: Array.from(ids, id => ({ id, values: job.columns.map(() => null) }))
      }
    };
    if (job.retainMask) result.mask = mask;
    return result;
  }
  const dna = job.decodedDNA ?? (await raster(job.dna, job.maxPixels, false));
  if (mask.width !== dna.width || mask.height !== dna.height)
    throw Error('The DNA and mask dimensions differ.');
  const result = {};
  if (job.extract) result.table = measureObjects(mask, dna, job.imported);
  if (job.retainMask) result.mask = mask;
  return result;
}
self.onmessage = async ({ data }) => {
  try {
    const result = await execute(data),
      transfers = [];
    if (result.bytes) transfers.push(result.bytes);
    if (result.mask) transfers.push(result.mask.data.buffer);
    self.postMessage({ result }, transfers);
  } catch (error) {
    self.postMessage({ error: error.message ?? String(error) });
  }
};
