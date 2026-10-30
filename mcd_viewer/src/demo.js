// Deterministic synthetic data; this exercises the same MCD reader as real files.
export function createDemoFile() {
  const names = [['Ir(191)', 'DNA'], ['Ho(165)', 'PanCK'], ['Er(170)', 'CD3'], ['Sm(147)', 'CD20'], ['Tb(159)', 'CD68'], ['Pr(141)', 'SMA']];
  const parts = [new Uint8Array(512)], rows = [], rois = [], channels = [], points = [];
  let offset = 512;
  for (let roi = 1; roi <= 2; roi++) {
    const width = roi === 1 ? 448 : 352, height = 352;
    const data = new Float32Array(width * height * 9);
    const cells = [];
    for (let j = 0; j < 16; j++) for (let i = 0; i < 21; i++) {
      const seed = Math.sin((i + 1) * 31.7 + j * 73.1 + roi * 4) * 43758.5453;
      const r = seed - Math.floor(seed);
      const x = 16 + i * 21 + Math.sin(j * 4 + i) * 6, y = 14 + j * 21 + Math.cos(i * 3 + j) * 6;
      if (((x - width / 2) / (width * .48)) ** 2 + ((y - height / 2) / (height * .46)) ** 2 < 1) cells.push({ x, y, type: Math.floor(r * 5), radius: 4 + r * 3 });
    }
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const base = (y * width + x) * 9;
      data[base] = x; data[base + 1] = y;
    }
    for (const cell of cells) {
      for (let y = Math.max(0, Math.floor(cell.y - 13)); y < Math.min(height, cell.y + 13); y++) for (let x = Math.max(0, Math.floor(cell.x - 13)); x < Math.min(width, cell.x + 13); x++) {
        const d = Math.hypot(x - cell.x, y - cell.y), base = (y * width + x) * 9;
        const noise = .9 + .1 * Math.sin(x * 77 + y * 13);
        data[base + 3] += 180 * Math.exp(-d * d / (cell.radius * cell.radius * .7)) * noise;
        data[base + 4 + cell.type] += 125 * Math.exp(-((d - cell.radius) ** 2) / 5) * noise;
      }
    }
    const end = offset + data.byteLength;
    rows.push(`<Acquisition><ID>${roi}</ID><Description>${roi === 1 ? 'Synthetic tissue · central region' : 'Synthetic tissue · adjacent region'}</Description><AcquisitionROIID>${roi}</AcquisitionROIID><MaxX>${width}</MaxX><MaxY>${height}</MaxY><ValueBytes>4</ValueBytes><SegmentDataFormat>Float</SegmentDataFormat><DataStartOffset>${offset}</DataStartOffset><DataEndOffset>${end}</DataEndOffset></Acquisition>`);
    rois.push(`<AcquisitionROI><ID>${roi}</ID><PanoramaID>1</PanoramaID></AcquisitionROI>`);
    [[0, 0], [width, 0], [width, height], [0, height]].forEach(([x, y], i) => points.push(`<ROIPoint><AcquisitionROIID>${roi}</AcquisitionROIID><OrderNumber>${i}</OrderNumber><SlideXPosUm>${x + 60 + (roi - 1) * 510}</SlideXPosUm><SlideYPosUm>${y + 40}</SlideYPosUm></ROIPoint>`));
    [['X', 'X'], ['Y', 'Y'], ['Z', 'Z'], ...names].forEach(([name, label], i) => channels.push(`<AcquisitionChannel><AcquisitionID>${roi}</AcquisitionID><OrderNumber>${i}</OrderNumber><ChannelName>${name}</ChannelName><ChannelLabel>${label}</ChannelLabel></AcquisitionChannel>`));
    parts.push(data); offset = end;
  }
  const photo = syntheticSlideImage();
  const imageEnd = offset + 161 + photo.byteLength + 1;
  parts.push(new Uint8Array(161), photo, new Uint8Array(1));
  const xml = `<MCDSchema xmlns="urn:synthetic-mcd"><Slide><ID>1</ID><Description>Synthetic demonstration</Description><WidthUm>1200</WidthUm><HeightUm>450</HeightUm><ImageStartOffset>${offset}</ImageStartOffset><ImageEndOffset>${imageEnd}</ImageEndOffset><ImageFile>synthetic-slide.bmp</ImageFile></Slide><Panorama><ID>1</ID><SlideID>1</SlideID><Description>Acquisition map</Description><Type>Default</Type></Panorama>${rows.join('')}${rois.join('')}${channels.join('')}${points.join('')}</MCDSchema>`;
  const encoded = new Uint16Array(xml.length);
  for (let i = 0; i < xml.length; i++) encoded[i] = xml.charCodeAt(i);
  parts.push(encoded);
  return new File(parts, 'Synthetic demonstration.mcd', { type: 'application/octet-stream' });
}

function syntheticSlideImage() {
  // Tiny uncompressed BMP fixture with two tissue patches on a whole slide.
  const width = 600, height = 225, stride = width * 3, bytes = new Uint8Array(54 + stride * height);
  const header = new DataView(bytes.buffer);
  bytes[0] = 66; bytes[1] = 77; header.setUint32(2, bytes.length, true); header.setUint32(10, 54, true); header.setUint32(14, 40, true);
  header.setInt32(18, width, true); header.setInt32(22, height, true); header.setUint16(26, 1, true); header.setUint16(28, 24, true); header.setUint32(34, stride * height, true);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const d1 = ((x - 142) / 105) ** 2 + ((y - 117) / 82) ** 2;
    const d2 = ((x - 373) / 82) ** 2 + ((y - 117) / 82) ** 2;
    const tissue = Math.min(d1, d2) < 1 + .04 * Math.sin(x * .2 + y * .16);
    const texture = Math.sin(x * .27) * Math.cos(y * .31) * 12;
    const [r, g, b] = x > 510 ? [207, 217, 220] : tissue ? [207 + texture, 151 + texture, 166 + texture] : [232, 233, 226];
    const at = 54 + (height - 1 - y) * stride + x * 3;
    bytes[at] = b; bytes[at + 1] = g; bytes[at + 2] = r;
  }
  return bytes;
}
