import { imageRange } from './mcd.js';

function panoramaArea(metadata) {
  const x1 = Number(metadata.SlideX1PosUm), y1 = Number(metadata.SlideY1PosUm);
  const x2 = Number(metadata.SlideX2PosUm), y2 = Number(metadata.SlideY2PosUm);
  const x4 = Number(metadata.SlideX4PosUm), y4 = Number(metadata.SlideY4PosUm);
  const area = Math.abs((x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1));
  return Number.isFinite(area) ? area : 0;
}

// Imported phone/camera photographs live in Panorama records, even when their
// filenames end in .jpg and their stored bytes are PNG. They need not own ROIs.
export function slideImageSources(model, slideId) {
  const imported = model.panoramas.filter(p => p.slideId === slideId && String(p.metadata.Type).trim().toLowerCase() === 'imported' && imageRange(p.metadata));
  // Prefer the broadest photograph for the existing "Whole-slide photo" option;
  // retain every other imported image as a separately named choice.
  imported.sort((a, b) => panoramaArea(b.metadata) - panoramaArea(a.metadata) || String(a.id).localeCompare(String(b.id), undefined, { numeric: true }));
  const sources = imported.map((p, i) => ({
    id: i === 0 ? 'slide' : `slide-photo:${p.id}`,
    label: `Whole-slide photo · ${p.name}`,
    name: p.name,
    range: imageRange(p.metadata), metadata: p.metadata, projection: 'panorama',
  }));
  const slide = model.slides.find(s => s.id === slideId);
  const range = slide && imageRange(slide.metadata);
  if (range) sources.push({
    id: sources.length ? 'slide-overview' : 'slide',
    label: sources.length ? 'Slide overview' : 'Whole-slide photo',
    name: 'Slide overview', range, metadata: slide.metadata, projection: 'slide',
  });
  return sources;
}
