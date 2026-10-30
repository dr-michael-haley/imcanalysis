export function zoomBox(box, factor, point, fitBox) {
  const width = Math.min(fitBox.width * 2, Math.max(fitBox.width / 64, box.width / factor));
  const ratio = width / box.width;
  return { x: point.x - (point.x - box.x) * ratio, y: point.y - (point.y - box.y) * ratio, width, height: box.height * ratio };
}

export class OverviewNavigation {
  constructor(svg, onZoom) {
    this.svg = svg; this.onZoom = onZoom; this.box = null; this.key = null;
    svg.addEventListener('wheel', event => {
      if (!this.box) return;
      event.preventDefault(); this.zoom(Math.exp(-event.deltaY * .002), this.point(event));
    }, { passive: false });
    svg.addEventListener('pointerdown', event => {
      if (!this.box || event.button !== 0) return;
      this.drag = { x: event.clientX, y: event.clientY, box: { ...this.box }, matrix: svg.getScreenCTM().inverse(), moved: false };
    });
    svg.addEventListener('pointermove', event => {
      const d = this.drag; if (!d) return;
      if (!event.buttons) { this.drag = null; svg.classList.remove('panning'); return; }
      if (!d.moved && Math.hypot(event.clientX - d.x, event.clientY - d.y) < 4) return;
      d.moved = true; svg.setPointerCapture(event.pointerId); svg.classList.add('panning');
      const dx = event.clientX - d.x, dy = event.clientY - d.y;
      this.box = { ...d.box, x: d.box.x - dx * d.matrix.a - dy * d.matrix.c, y: d.box.y - dx * d.matrix.b - dy * d.matrix.d }; this.apply();
    });
    const release = () => { this.suppressClick = !!this.drag?.moved; this.drag = null; svg.classList.remove('panning'); };
    svg.addEventListener('pointerup', release); svg.addEventListener('pointercancel', release);
    svg.addEventListener('click', event => { if (this.suppressClick) { event.stopImmediatePropagation(); event.preventDefault(); this.suppressClick = false; } }, true);
    svg.addEventListener('dblclick', event => { event.preventDefault(); this.fit(); });
    svg.addEventListener('keydown', event => {
      if (!this.box) return;
      if (event.key === '+' || event.key === '=') this.zoom(1.4);
      else if (event.key === '-') this.zoom(1 / 1.4);
      else if (event.key.toLowerCase() === 'f' || event.key === 'Home') this.fit();
      else if (event.key.startsWith('Arrow')) {
        const dx = { ArrowLeft: -.1, ArrowRight: .1 }[event.key] || 0, dy = { ArrowUp: -.1, ArrowDown: .1 }[event.key] || 0;
        this.box.x += this.box.width * dx; this.box.y += this.box.height * dy; this.apply();
      } else return;
      event.preventDefault(); event.stopPropagation();
    });
  }
  point(event) { return new DOMPoint(event.clientX, event.clientY).matrixTransform(this.svg.getScreenCTM().inverse()); }
  setBounds(key, box) { this.fitBox = box; if (key !== this.key || !this.box) { this.key = key; this.box = { ...box }; } this.apply(); }
  fit() { if (this.fitBox) { this.box = { ...this.fitBox }; this.apply(); } }
  zoom(factor, point) { if (!this.box) return; this.box = zoomBox(this.box, factor, point || { x: this.box.x + this.box.width / 2, y: this.box.y + this.box.height / 2 }, this.fitBox); this.apply(); }
  apply() {
    const b = this.box; this.svg.setAttribute('viewBox', `${b.x} ${b.y} ${b.width} ${b.height}`);
    // Labels stay readable while zooming; outlines use non-scaling strokes.
    for (const label of this.svg.querySelectorAll('text')) label.setAttribute('font-size', b.width / 28);
    this.onZoom(this.fitBox.width / b.width);
  }
}
