import { readSchema, readAcquisition, contrast, readOpticalImage } from './mcd.js';
import { histograms } from './histogram.js';
self.onmessage = async ({ data: { type, file, acquisition, selected, range, planes } }) => {
  try {
    const progress = value => self.postMessage({ progress: value });
    let result;
    if (type === 'schema') result = await readSchema(file, progress);
    else if (type === 'image') result = await readOpticalImage(file, range);
    else if (type === 'histogram') result = histograms(planes);
    else {
      result = await readAcquisition(file, acquisition, selected, progress);
      for (const plane of result.planes) plane.limits = contrast(plane.data);
    }
    const transfer = result?.planes ? [...result.planes.map(p => p.data.buffer), result.occupied.buffer] : [];
    self.postMessage({ result }, transfer);
  } catch (error) { self.postMessage({ error: error.message || String(error) }); }
};
