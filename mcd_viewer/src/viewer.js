import { colourTable } from './colormaps.js';

export class ImageViewer {
  constructor(canvas, { onZoom, onPixel, onDraw = () => {} }) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.onZoom = onZoom; this.onPixel = onPixel; this.onDraw = onDraw;
    this.source = null; this.scale = 1; this.x = 0; this.y = 0; this.fitted = true;
    new ResizeObserver(() => { this.resize(); }).observe(canvas.parentElement);
    canvas.addEventListener('wheel', e => { if (!this.source) return; e.preventDefault(); const p = this.position(e); this.zoom(Math.exp(-e.deltaY * .0015), p.x, p.y); }, { passive: false });
    canvas.addEventListener('pointerdown', e => { if (!this.source || e.button !== 0) return; canvas.setPointerCapture(e.pointerId); this.drag = { x: e.clientX, y: e.clientY, sx: this.x, sy: this.y }; });
    canvas.addEventListener('pointermove', e => { if (!this.drag) return; this.fitted = false; this.x = this.drag.sx + e.clientX - this.drag.x; this.y = this.drag.sy + e.clientY - this.drag.y; this.draw(); });
    canvas.addEventListener('pointerup', e => {
      if (!this.drag) return;
      if (!this.figure && Math.hypot(e.clientX - this.drag.x, e.clientY - this.drag.y) < 4) { const p = this.position(e); this.onPixel(Math.floor((p.x - this.x) / this.scale), Math.floor((p.y - this.y) / this.scale)); }
      this.drag = null;
    });
    canvas.addEventListener('pointercancel', () => { this.drag = null; });
    canvas.addEventListener('dblclick', () => this.fit());
    canvas.addEventListener('keydown', e => {
      if (e.key.toLowerCase() === 'f') this.fit();
      else if (e.key === '+' || e.key === '=') this.zoom(1.3);
      else if (e.key === '-') this.zoom(1 / 1.3);
    });
  }
  position(e) { const r = this.canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  camera() { return { centreX: (this.width / 2 - this.x) / this.scale, centreY: (this.height / 2 - this.y) / this.scale, scale: this.scale, fitted: this.fitted }; }
  restoreCamera(camera) { this.scale = camera.scale; this.x = this.width / 2 - camera.centreX * this.scale; this.y = this.height / 2 - camera.centreY * this.scale; this.fitted = camera.fitted; this.draw(); }
  resize() {
    const camera = this.camera(), oldWidth = this.width, r = this.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (r.width <= 0 || r.height <= 0) return;
    this.width = r.width; this.height = r.height; this.canvas.width = Math.round(r.width * dpr); this.canvas.height = Math.round(r.height * dpr);
    if (this.figure && this.source && oldWidth > 0) {
      this.restoreCamera({ ...camera, scale: this.physicalWidth ? this.width / this.physicalWidth : camera.scale * this.width / oldWidth, fitted: false });
    } else if (this.fitted) this.fit(); else if (oldWidth > 0) this.restoreCamera(camera); else this.draw();
  }
  setSource(source, fit = false) { this.source = source; if (fit || this.fitted) this.fit(); else this.draw(); }
  clear() { this.source = null; this.draw(); }
  fit() { if (!this.source || this.physicalWidth) return this.draw(); const pad = this.figure ? 0 : 44; this.scale = Math.min((this.width - pad) / this.source.width, (this.height - pad) / this.source.height); this.scale = Math.max(.001, this.scale); this.x = (this.width - this.source.width * this.scale) / 2; this.y = (this.height - this.source.height * this.scale) / 2; this.fitted = true; this.draw(); }
  zoom(factor, cx = this.width / 2, cy = this.height / 2) { if (!this.source || this.physicalWidth) return; const scale = Math.min(64, Math.max(.005, this.scale * factor)); const ratio = scale / this.scale; this.x = cx - (cx - this.x) * ratio; this.y = cy - (cy - this.y) * ratio; this.scale = scale; this.fitted = false; this.draw(); }
  oneToOne() { this.zoom(1 / this.scale); }
  draw() {
    const ctx = this.ctx, dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, this.width || 0, this.height || 0);
    if (!this.source) { this.onDraw(); return; }
    if (this.figure) { ctx.fillStyle = this.background || '#000000'; ctx.fillRect(0, 0, this.width, this.height); }
    ctx.imageSmoothingEnabled = !this.figure && this.scale < 1;
    ctx.drawImage(this.source, this.x, this.y, this.source.width * this.scale, this.source.height * this.scale);
    this.onZoom(this.scale);
    this.onDraw();
  }
}

const transformValue = (v, mode, cofactor) => mode === 'arcsinh' ? Math.asinh(v / cofactor) : mode === 'log' ? Math.log1p(Math.max(0, v)) : mode === 'sqrt' ? Math.sqrt(Math.max(0, v)) : v;
export async function composite(raster, slots, mode, cofactor, cancelled) {
  const image = new ImageData(raster.width, raster.height);
  const active = slots.filter(s => s.enabled).map(s => {
    const plane = raster.planes.find(p => p.index === s.index);
    const low = transformValue(s.low, mode, cofactor), high = transformValue(s.high, mode, cofactor);
    return { data: plane?.data, low, range: Math.max(1e-20, high - low), table: colourTable(s) };
  }).filter(s => s.data);
  for (let begin = 0; begin < raster.occupied.length; begin += 65536) {
    if (cancelled()) return null;
    const end = Math.min(begin + 65536, raster.occupied.length);
    for (let i = begin; i < end; i++) {
      const at = i * 4;
      if (!raster.occupied[i]) {
        const check = ((Math.floor((i % raster.width) / 8) + Math.floor(i / raster.width / 8)) % 2) ? 28 : 17;
        image.data[at] = check; image.data[at + 1] = check + 3; image.data[at + 2] = check + 6;
      } else {
        let r = 0, g = 0, b = 0;
        for (const s of active) {
          const value = s.data[i]; if (!Number.isFinite(value)) continue;
          const t = Math.max(0, Math.min(1, (transformValue(value, mode, cofactor) - s.low) / s.range));
          const rgb = s.table[Math.min(255, Math.floor(t * 256))];
          r += rgb[0]; g += rgb[1]; b += rgb[2];
        }
        image.data[at] = r; image.data[at + 1] = g; image.data[at + 2] = b;
      }
      image.data[at + 3] = 255;
    }
    if (end < raster.occupied.length) await new Promise(requestAnimationFrame);
  }
  if (cancelled()) return null;
  const surface = document.createElement('canvas'); surface.width = raster.width; surface.height = raster.height;
  surface.getContext('2d').putImageData(image, 0, 0);
  return surface;
}
