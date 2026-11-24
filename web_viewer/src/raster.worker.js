import { openTiffPlane, readScalarPlane } from './tiff-plane.js';
import { normalizeImage } from './normalization.js';

function checkSize(width, height) {
  if (width * height > 16_777_216 || width > 16384 || height > 16384) throw new Error('ROI images must be at most 16 megapixels and 16384 pixels per side.');
}
self.onmessage = async ({ data: { file, plane, mask, normalization, kind } }) => {
  try {
    if (kind === 'rgb' && !/\.tiff?$/i.test(file.name)) {
      const bitmap = await createImageBitmap(file);
      try {
        const { width, height } = bitmap; checkSize(width, height);
        const canvas = new OffscreenCanvas(width, height), ctx = canvas.getContext('2d');
        ctx.drawImage(bitmap, 0, 0);
        const pixels = ctx.getImageData(0, 0, width, height).data;
        self.postMessage({ result: { width, height, data: pixels, planes: 1, kind } }, [pixels.buffer]);
      } finally { bitmap.close(); }
      return;
    }
    const singlePlane = kind ? 'Aligned layers require one 2-D image per ROI, not a TIFF stack.'
      : mask && 'Choose a single-plane segmentation mask; this file contains multiple planes.';
    const { image, width, height, count } = await openTiffPlane(file, plane, singlePlane, checkSize);
    if (kind === 'rgb') {
      const samples = image.getSamplesPerPixel(), directory = image.fileDirectory;
      if (![3, 4].includes(samples) || directory.getValue('PhotometricInterpretation') !== 2) throw new Error('Choose an RGB/RGBA TIFF, or use Scalar image for a single-channel TIFF.');
      const bits = Array.from(directory.getValue('BitsPerSample') ?? []), formats = Array.from(directory.getValue('SampleFormat') ?? [1]);
      if (bits.some(bit => ![8, 16].includes(bit)) || formats.some(format => format !== 1)) throw new Error('Colour TIFFs must use unsigned 8-bit or 16-bit RGB/RGBA samples.');
      const values = await image.readRasters({ interleave: true });
      const pixels = new Uint8ClampedArray(width * height * 4), associated = directory.getValue('ExtraSamples')?.[0] === 1;
      for (let i = 0; i < width * height; i++) {
        const alpha = samples === 4 ? values[i * samples + 3] / (2 ** (bits[3] ?? bits[0]) - 1) : 1;
        for (let c = 0; c < 3; c++) pixels[i * 4 + c] = values[i * samples + c] / (2 ** (bits[c] ?? bits[0]) - 1) * 255 / (associated && alpha ? alpha : 1);
        pixels[i * 4 + 3] = alpha * 255;
      }
      self.postMessage({ result: { width, height, data: pixels, planes: count, kind } }, [pixels.buffer]); return;
    }
    const values = await readScalarPlane(image);
    const labels = new Set();
    if (kind === 'labels') for (const value of values) {
      if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) throw new Error('Categorical TIFFs require non-negative integer label IDs.');
      if (value && labels.size < 100) labels.add(value);
    }
    const scaled = normalization && !mask ? normalizeImage(values, normalization) : { data: values, normalization: null };
    self.postMessage({ result: { width, height, ...scaled, planes: count, kind, labels: [...labels] } }, [scaled.data.buffer]);
  } catch (error) { self.postMessage({ error: error.message ?? String(error) }); }
};
