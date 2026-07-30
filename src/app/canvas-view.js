// The workflow canvas's drawing and editing: blocks as round widgets with their typed ports, wires as curves from
// an output on a block's right to an input on another's left, named by what they carry. Drag a block to move it,
// drag from an output port to an input port to wire them (a port only takes a wire of its own type), click to
// select, Delete to remove, double-click a block for its parameters.
(function (S) {
  'use strict';
  const W = S.Flow, NS = 'http://www.w3.org/2000/svg';
  const R = 32, PORT = 6.5;
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs || {}) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const r2 = (v) => Math.round(v * 100) / 100;
  // the icons inside the blocks, on a 24 × 24 grid
  const ICONS = {
    table: 'M5 6h14v12H5z M5 10h14 M5 14h14 M10 6v12',
    split: 'M4 12h6 M10 12l6-5 M10 12l6 5 M16 7h4 M16 17h4',
    model: 'M5 17c3 0 4-10 7-10s4 10 7 10 M5 12h14',
    grid: 'M5 5h14v14H5z M12 5v14 M5 12h14',
  };

  class Canvas {
    /** host: an element; opts: {doc, onChange(doc, why), onSelect(sel), onEdit(blockId)} */
    constructor(host, opts) {
      this.host = host;
      this.opts = opts || {};
      this.doc = opts.doc;
      this.sel = null;
      this.state = { out: {}, err: {}, missing: {} };
      this.svg = el('svg', { class: 'cv-svg', role: 'img', 'aria-label': 'Workflow canvas: blocks joined by wires' });
      host.appendChild(this.svg);
      this.fit();
      this.svg.addEventListener('pointerdown', (e) => this.down(e));
      this.svg.addEventListener('pointermove', (e) => this.move(e));
      this.svg.addEventListener('pointerup', (e) => this.up(e));
      this.svg.addEventListener('keydown', (e) => this.key(e));
      this.svg.setAttribute('tabindex', '0');
    }
    /** the camera: every block in view, with room for names and ports */
    fit() {
      const bs = this.doc.blocks;
      if (!bs.length) { this.vb = { x: 0, y: 0, w: 800, h: 300 }; this.applyVb(); return; }
      const x0 = Math.min(...bs.map((b) => b.x)) - R - 70, x1 = Math.max(...bs.map((b) => b.x)) + R + 70;
      const y0 = Math.min(...bs.map((b) => b.y)) - R - 26, y1 = Math.max(...bs.map((b) => b.y)) + R + 74;
      this.vb = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
      // on a wide screen the drawing keeps its own scale, centred, rather than growing to fill the canvas
      const W = this.svg.clientWidth, H = this.svg.clientHeight;
      if (W >= 600 && H > 0) {
        const k = Math.max(this.vb.w / W, this.vb.h / H, 1 / 1.2), w = W * k, hh = H * k;
        this.vb = { x: this.vb.x - (w - this.vb.w) / 2, y: this.vb.y - (hh - this.vb.h) / 2, w, h: hh };
      }
      this.applyVb();
    }
    applyVb() { this.svg.setAttribute('viewBox', [this.vb.x, this.vb.y, this.vb.w, this.vb.h].map(r2).join(' ')); }
    setState(st) { this.state = st; this.render(); }
    point(e) { const m = this.svg.getScreenCTM(); if (!m) return { x: 0, y: 0 }; const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse()); return { x: p.x, y: p.y }; }

    /** where a block's port is: inputs spread down its left side, outputs down its right */
    port(b, side, id) {
      const K = W.KINDS[b.kind], list = side === 'in' ? K.ins : K.outs, k = list.findIndex((p) => p.id === id), n = list.length;
      const dy = (k - (n - 1) / 2) * 24, dx = Math.sqrt(Math.max(0, R * R - dy * dy));
      return { x: b.x + (side === 'in' ? -dx : dx), y: b.y + dy };
    }
    render() {
      const svg = this.svg, doc = this.doc, st = this.state;
      for (const c of [...svg.childNodes]) c.remove();
      const gw = el('g', { class: 'cv-wires' }, svg), gb = el('g', { class: 'cv-blocks' }, svg), gl = el('g', { class: 'cv-labels' }, svg);
      this.tmp = el('path', { class: 'cv-tmp', 'data-audit': 'ignore', d: '' }, svg);
      for (const w of doc.wires) {
        const a = doc.blocks.find((b) => b.id === w.from[0]), b = doc.blocks.find((x) => x.id === w.to[0]);
        if (!a || !b) continue;
        const pa = this.port(a, 'out', w.from[1]), pb = this.port(b, 'in', w.to[1]), type = W.portOf(doc, w.from, 'out').type;
        const c = S.Kit.flow.curve({ x: pa.x + PORT, y: pa.y }, { x: pb.x - PORT, y: pb.y });
        const g = el('g', { class: 'cv-wire ' + W.TYPES[type].cls + (this.sel && this.sel.wire === w.id ? ' sel' : ''), 'data-wire': w.id }, gw);
        el('path', { class: 'cv-hit', d: c.d, 'data-audit': 'ignore' }, g);
        el('path', { class: 'cv-line', d: c.d, 'data-audit': 'edge' }, g);
        // what the wire carries, on a pill at its middle
        const name = W.portOf(doc, w.from, 'out').name, tw = S.geom.textWidth(name, 12) + 14;
        el('rect', { class: 'cv-pill', x: r2(c.mid.x - tw / 2), y: r2(c.mid.y - 10), width: r2(tw), height: 20, rx: 10, 'data-audit': 'overlap-ok' }, gl);
        const t = el('text', { class: 'cv-wname', x: r2(c.mid.x), y: r2(c.mid.y + 4), 'text-anchor': 'middle', 'data-audit': 'on-wire' }, gl);
        t.textContent = name;
      }
      for (const b of doc.blocks) {
        const K = W.KINDS[b.kind], status = st.err[b.id] ? 'bad' : st.missing[b.id] ? 'warn' : st.out[b.id] ? 'ok' : '';
        const g = el('g', { class: 'cv-block ' + status + (this.sel && this.sel.block === b.id ? ' sel' : ''), 'data-block': b.id }, gb);
        el('circle', { class: 'cv-body', cx: b.x, cy: b.y, r: R, 'data-audit': 'node' }, g);
        el('path', { class: 'cv-icon', d: ICONS[K.icon] || '', transform: `translate(${r2(b.x - 15)} ${r2(b.y - 15)}) scale(1.25)`, 'data-audit': 'ignore' }, g);
        for (const side of ['in', 'out']) for (const p of side === 'in' ? K.ins : K.outs) {
          const q = this.port(b, side, p.id);
          const pc = el('circle', { class: 'cv-port ' + side + ' ' + W.TYPES[p.type].cls, cx: r2(q.x), cy: r2(q.y), r: PORT, 'data-port': b.id + ':' + p.id, 'data-side': side, 'data-audit': 'overlap-ok' }, g);
          const tt = el('title', {}, pc); tt.textContent = (side === 'in' ? 'input: ' : 'output: ') + p.name + ' (' + W.TYPES[p.type].name + ')';
        }
        const name = el('text', { class: 'cv-name', x: b.x, y: b.y + R + 19, 'text-anchor': 'middle' }, gl);
        name.textContent = K.name;
        const sum = st.err[b.id] ? 'error: ' + st.err[b.id] : st.missing[b.id] ? 'needs ' + st.missing[b.id].join(' and ') : st.out[b.id] && K.sum ? K.sum(st.out[b.id]) : '';
        if (sum) { const s = el('text', { class: 'cv-sum ' + status, x: b.x, y: b.y + R + 36, 'text-anchor': 'middle' }, gl); s.textContent = sum.length > 34 ? sum.slice(0, 33) + '…' : sum; if (sum.length > 34) { const t2 = el('title', {}, s); t2.textContent = sum; } }
      }
    }

    // ---- editing -------------------------------------------------------------------------------------------
    down(e) {
      const port = e.target.closest('[data-port]'), blk = e.target.closest('[data-block]'), wire = e.target.closest('[data-wire]');
      const p = this.point(e);
      if (port && port.dataset.side === 'out') {
        const [bid, pid] = port.dataset.port.split(':'), b = this.doc.blocks.find((x) => x.id === bid), q = this.port(b, 'out', pid);
        this.drag = { kind: 'wire', from: [bid, pid], start: q };
        this.svg.setPointerCapture(e.pointerId);
        e.preventDefault();
        return;
      }
      if (blk) {
        const b = this.doc.blocks.find((x) => x.id === blk.dataset.block);
        this.drag = { kind: 'block', b, dx: p.x - b.x, dy: p.y - b.y, moved: false, x0: e.clientX, y0: e.clientY };
        this.svg.setPointerCapture(e.pointerId);
        return;
      }
      if (wire) { this.select({ wire: wire.dataset.wire }); return; }
      this.select(null);
    }
    move(e) {
      const d = this.drag;
      if (!d) return;
      const p = this.point(e);
      if (d.kind === 'block') {
        if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < 4) return;
        d.moved = true;
        d.b.x = Math.round(p.x - d.dx); d.b.y = Math.round(p.y - d.dy);
        this.render();
      } else {
        const c = S.Kit.flow.curve({ x: d.start.x + PORT, y: d.start.y }, p);
        this.tmp.setAttribute('d', c.d);
        // light the input ports that would take this wire
        for (const pe of this.svg.querySelectorAll('[data-side="in"]')) { const [bid, pid] = pe.dataset.port.split(':'); pe.classList.toggle('can', W.canWire(this.doc, d.from, [bid, pid])); }
      }
    }
    up(e) {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      try { this.svg.releasePointerCapture(e.pointerId); } catch (err) { /* gone */ }
      if (d.kind === 'block') {
        if (!d.moved) {
          // a second click on the same block soon after is a double-click (pointer capture hides the browser's own)
          const now = performance.now(), dbl = this.lastClick && this.lastClick.id === d.b.id && now - this.lastClick.t < 450;
          this.lastClick = dbl ? null : { id: d.b.id, t: now };
          this.select({ block: d.b.id });
          if (dbl && this.opts.onEdit) this.opts.onEdit(d.b.id);
          return;
        }
        this.fit(); this.render();
        this.changed('move');
        return;
      }
      this.tmp.setAttribute('d', '');
      const hit = document.elementFromPoint(e.clientX, e.clientY), pe = hit && hit.closest && hit.closest('[data-side="in"]');
      for (const x of this.svg.querySelectorAll('.can')) x.classList.remove('can');
      if (!pe) return;
      const to = pe.dataset.port.split(':');
      if (!W.canWire(this.doc, d.from, to)) { if (S.UI.toast) S.UI.toast('That input takes ' + W.TYPES[W.portOf(this.doc, to, 'in').type].name + ', not ' + W.TYPES[W.portOf(this.doc, d.from, 'out').type].name + '.', 'err'); return; }
      this.connect(d.from, to);
    }
    /** wire an output to an input, replacing what that input had */
    connect(from, to) {
      this.doc.wires = this.doc.wires.filter((w) => !(w.to[0] === to[0] && w.to[1] === to[1]));
      let n = 1; while (this.doc.wires.some((w) => w.id === 'w' + n)) n++;
      this.doc.wires.push({ id: 'w' + n, from, to });
      this.changed('wire');
    }
    /** select a block or a wire (classes only: the elements stay, so a double-click still lands) */
    select(sel) {
      this.sel = sel;
      for (const g of this.svg.querySelectorAll('[data-block]')) g.classList.toggle('sel', !!(sel && sel.block === g.dataset.block));
      for (const g of this.svg.querySelectorAll('[data-wire]')) g.classList.toggle('sel', !!(sel && sel.wire === g.dataset.wire));
      if (this.opts.onSelect) this.opts.onSelect(sel);
    }
    key(e) {
      if ((e.key === 'Delete' || e.key === 'Backspace') && this.sel) {
        e.preventDefault();
        if (this.sel.wire) this.doc.wires = this.doc.wires.filter((w) => w.id !== this.sel.wire);
        else if (this.sel.block) { const id = this.sel.block; this.doc.blocks = this.doc.blocks.filter((b) => b.id !== id); this.doc.wires = this.doc.wires.filter((w) => w.from[0] !== id && w.to[0] !== id); }
        this.sel = null;
        this.fit();
        this.changed('delete');
        if (this.opts.onSelect) this.opts.onSelect(null);
      }
    }
    /** add a block of a kind where there is room: to the right of everything */
    add(kind) {
      const K = W.KINDS[kind], bs = this.doc.blocks;
      let n = 1; while (bs.some((b) => b.id === 'b' + n)) n++;
      const x = bs.length ? Math.max(...bs.map((b) => b.x)) + 200 : 90, y = bs.length ? Math.round(bs.reduce((a, b) => a + b.y, 0) / bs.length) : 150;
      const b = { id: 'b' + n, kind, x, y, params: Object.assign({}, K.defaults) };
      bs.push(b);
      this.fit();
      this.changed('add');
      this.render();
      this.select({ block: b.id });
      return b;
    }
    changed(why) { if (this.opts.onChange) this.opts.onChange(this.doc, why); }
  }
  S.FlowCanvas = Canvas;
})(globalThis.Nooral = globalThis.Nooral || {});
