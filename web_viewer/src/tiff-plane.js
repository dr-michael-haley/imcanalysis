import { fromBlob } from 'geotiff';

export async function openTiffPlane(file, plane, singlePlaneError, checkSize) {
  const tiff = await fromBlob(file);
  const count = await tiff.getImageCount();
  if (singlePlaneError && count !== 1) throw new Error(singlePlaneError);
  if (!Number.isInteger(plane) || plane < 0 || plane >= count)
    throw new Error(`This TIFF contains ${count} plane(s).`);
  const image = await tiff.getImage(plane);
  const width = image.getWidth(),
    height = image.getHeight();
  checkSize(width, height);
  return { image, width, height, count };
}

export async function readScalarPlane(image) {
  if (image.getSamplesPerPixel() !== 1) throw new Error('Choose a scalar channel TIFF.');
  const [data] = await image.readRasters({ samples: [0] });
  if (typeof data[0] === 'bigint') throw new Error('64-bit integer TIFFs are not supported.');
  return data;
}
