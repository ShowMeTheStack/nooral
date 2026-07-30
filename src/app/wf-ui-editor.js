// The canvas editor: blocks dragged and snapped to a grid, wires drawn from port to port (compatible ports light up
// while a wire is carried, the others say why not), selection by click or box, copy and paste, undo and redo of
// everything, pan and zoom (wheel, pinch, keys), fit and tidy-up. It draws with S.WF.Draw and lays out with S.WF.tidy.
//   new S.WF.Editor(host, {doc, readOnly, allowed, onChange(doc, why), onSelect(ids), onOpen(id), onEdit(id)})
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const { h, icon } = S.UI;
  const NS = 'http://www.w3.org/2000/svg';
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const svgEl = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); return e; };

  /** a document read with WF.fromJSON, keeping the lanes the tidy-up left for its long wires */
  WF.adopt = function (doc) {
    const d = WF.fromJSON(doc);
    const via = new Map(((doc && doc.wires) || []).filter((w) => w && w.via).map((w) => [w.from.join('.') + '>' + w.to.join('.'), w.via]));
    for (const w of d.wires) { const v = via.get(w.from.join('.') + '>' + w.to.join('.')); if (v) w.via = clone(v); }
    return d;
  };

  class Editor {
    constructor(host, opts) {
      this.opts = opts || {};
      this.host = host;
      this.doc = WF.setCompact(WF.adopt(this.opts.doc || WF.newDoc()), !!this.opts.compact);
      this.sel = new Set();
      this.selWire = null;
      this.states = {};
      this.view = { x: 40, y: 40, z: 1 };
      this.hist = [JSON.stringify(this.doc)]; this.hi = 0;
      this.wrap = h('div.wf-wrap', { tabindex: '0', 'aria-label': 'Workflow canvas: drag blocks here from the palette, then drag from an output port to an input port to wire them' });
      this.svg = svgEl('svg', { class: 'wf-svg', role: 'img', 'aria-label': 'workflow' });
      this.root = svgEl('g', { class: 'wf-root' });
      this.wireLayer = svgEl('g', { class: 'wf-wires' });
      this.blockLayer = svgEl('g', { class: 'wf-blocks' });
      this.ui = svgEl('g', { class: 'wf-ui', 'data-audit': 'ignore' });
      this.root.append(this.wireLayer, this.blockLayer, this.ui);
      this.svg.appendChild(this.root);
      this.tip = h('div.wf-tip', { role: 'status', 'aria-live': 'polite' });
      const zb = (ic, title, fn, id) => { const b = h('button.icon-btn.zbtn', { type: 'button', title, 'aria-label': title, id }, icon(ic)); b.addEventListener('click', fn); return b; };
      this.zoomBox = h('div.wf-zoom', zb('zoomIn', 'Zoom in (+)', () => this.zoomBy(1.2), 'wfZoomIn'), zb('zoomOut', 'Zoom out (−)', () => this.zoomBy(1 / 1.2), 'wfZoomOut'), zb('fit', 'Fit the workflow (F)', () => this.fit(), 'wfFit'));
      this.empty = h('div.wf-empty', icon('layout'), h('b', 'An empty canvas'), h('span', this.opts.readOnly ? 'Nothing to show.' : 'Drag a block here from the palette, or open a sample.'));
      this.wrap.append(this.svg, this.zoomBox, this.tip, this.empty);
      host.appendChild(this.wrap);
      this.bind();
      // the view follows the size of the canvas until the reader pans or zooms it
      this.ro = new ResizeObserver(() => { const w = this.wrap.clientWidth, hh = this.wrap.clientHeight; if (!w || !hh) return; if (w !== this._w || hh !== this._h) {
        this._w = w; this._h = hh;
        // a canvas laid out to fit its width (in a lesson) lays out afresh when its width changes, until it is edited
        if (this.opts.tidy && this.hi === 0 && !this.userView && (!this._tidyW || Math.abs(this._tidyW - w) > 40)) { this.tidyToFit(); this.hist[0] = JSON.stringify(this.doc); this.render(); }
        if (!this.userView) this.fit(); else this.applyView();
      } });
      this.ro.observe(this.wrap);
      if (this.opts.tidy) this.tidyToFit(); else if (WF.unplaced(this.doc)) WF.tidy(this.doc);
      this.render();
    }
    dispose() { this.ro.disconnect(); this.disposed = true; window.removeEventListener('pointermove', this.onMove); window.removeEventListener('pointerup', this.onUp); }

    // ---- the document ------------------------------------------------------------------------------------------
    /** replace the whole document (open, a sample, undo) */
    setDoc(doc, keepView) {
      this.doc = WF.setCompact(WF.adopt(doc), !!this.opts.compact);
      if (this.opts.tidy) this.tidyToFit(); else if (WF.unplaced(this.doc)) WF.tidy(this.doc);
      this.sel.clear(); this.selWire = null;
      this.render();
      if (!keepView) this.fit();
    }
    /** a change worth an undo step: remember it and tell the page */
    commit(why) {
      const j = JSON.stringify(this.doc);
      if (j === this.hist[this.hi]) { this.render(); return; }
      this.hist = this.hist.slice(0, this.hi + 1); this.hist.push(j); this.hi = this.hist.length - 1;
      if (this.hist.length > 200) { this.hist.shift(); this.hi--; }
      this.render();
      if (this.opts.onChange) this.opts.onChange(this.doc, why);
    }
    undo() { if (this.hi <= 0) return false; this.hi--; this.restore('undo'); return true; }
    redo() { if (this.hi >= this.hist.length - 1) return false; this.hi++; this.restore('redo'); return true; }
    restore(why) {
      this.doc = WF.setCompact(JSON.parse(this.hist[this.hi]), !!this.opts.compact);
      for (const id of [...this.sel]) if (!WF.byId(this.doc, id)) this.sel.delete(id);
      this.selWire = null;
      this.render();
      if (this.opts.onChange) this.opts.onChange(this.doc, why);
    }
    allowed(type) { return !this.opts.allowed || this.opts.allowed.includes(type); }
    /** add a block of a type where there is room near the middle of the view (or at a page point) */
    addBlock(type, params, at) {
      if (this.opts.readOnly || !this.allowed(type)) return null;
      const g0 = WF.dims(this.doc);
      let p = at;
      if (!p) {
        const r = this.wrap.getBoundingClientRect();
        p = this.toWorld(r.left + r.width * 0.42, r.top + r.height * 0.4);
        // step down and right until it overlaps nothing
        for (let k = 0; k < 40 && this.doc.blocks.some((b) => Math.abs(b.x - p.x) < g0.W + 20 && Math.abs(b.y - p.y) < 90); k++) { p = { x: p.x + 24, y: p.y + 60 }; }
      }
      const b = WF.add(this.doc, type, params, this.snap(p.x), this.snap(p.y));
      this.sel = new Set([b.id]); this.selWire = null;
      this.commit('add ' + type);
      this.ensureVisible(b);
      if (this.opts.onSelect) this.opts.onSelect([...this.sel]);
      return b;
    }
    /** a palette item dropped at a point of the page */
    dropAt(type, cx, cy) {
      const r = this.wrap.getBoundingClientRect();
      if (cx < r.left || cx > r.right || cy < r.top || cy > r.bottom) return null;
      const p = this.toWorld(cx, cy);
      return this.addBlock(type, null, { x: p.x - WF.dims(this.doc).W / 2, y: p.y - 30 });
    }
    deleteSelection() {
      if (this.opts.readOnly) return;
      if (this.selWire) { this.doc.wires = this.doc.wires.filter((w) => w.id !== this.selWire); this.selWire = null; this.commit('delete wire'); return; }
      if (!this.sel.size) return;
      for (const id of this.sel) WF.remove(this.doc, id);
      this.sel.clear();
      this.commit('delete');
      if (this.opts.onSelect) this.opts.onSelect([]);
    }
    copy() {
      if (!this.sel.size) return false;
      const blocks = this.doc.blocks.filter((b) => this.sel.has(b.id));
      Editor.clip = { blocks: clone(blocks), wires: clone(this.doc.wires.filter((w) => this.sel.has(w.from[0]) && this.sel.has(w.to[0]))) };
      return true;
    }
    paste() {
      const c = Editor.clip;
      if (!c || this.opts.readOnly) return false;
      const map = {};
      this.sel.clear();
      for (const b of c.blocks) {
        if (!this.allowed(b.type)) continue;
        const nb = { id: WF.newId(this.doc, b.type), type: b.type, params: clone(b.params), x: b.x + 36, y: b.y + 36 };
        map[b.id] = nb.id;
        this.doc.blocks.push(nb);
        this.sel.add(nb.id);
      }
      for (const w of c.wires) if (map[w.from[0]] && map[w.to[0]]) { try { WF.connect(this.doc, [map[w.from[0]], w.from[1]], [map[w.to[0]], w.to[1]]); } catch (e) { /* skip */ } }
      Editor.clip = { blocks: c.blocks.map((b) => Object.assign({}, b, { x: b.x + 36, y: b.y + 36 })), wires: c.wires };
      this.commit('paste');
      return true;
    }
    tidy() { if (this.opts.tidy) this.tidyToFit(); else WF.tidy(this.doc); this.commit('tidy'); this.fit(); }
    /** lay out folded into as many columns as make the whole workflow largest in this canvas (lessons, challenges) */
    tidyToFit() {
      const c = this.opts.compact ? 1 : 0, W = (this.wrap.clientWidth || 0) - 40 - 44 * c, H = this.wrap.clientHeight - 40 - 40 * c;
      if (W > 0) this._tidyW = this.wrap.clientWidth;
      const top = Math.max(2, (this.opts.maxLayers && this.opts.maxLayers()) || WF.dims(this.doc).MAXL);
      if (W <= 0 || H <= 0) { WF.tidy(this.doc, { maxLayers: top }); return; }
      let best = null;
      for (let m = top; m >= 2; m--) {
        const d = WF.setCompact(clone(this.doc), !!this.opts.compact);
        WF.tidy(d, { maxLayers: m });
        const b = this.boundsOf(d), z = Math.min(W / b.w, H / b.h);
        if (!best || z > best.z + 0.02) best = { m, z };
      }
      WF.tidy(this.doc, { maxLayers: best.m });
    }
    setParams(id, patch) {
      const b = WF.byId(this.doc, id);
      if (!b) return;
      Object.assign(b.params, patch);
      // a change of ports may leave wires on ports that are gone
      const ports = WF.ports(b);
      this.doc.wires = this.doc.wires.filter((w) => (w.to[0] !== id || ports.inputs.some((p) => p.id === w.to[1])) && (w.from[0] !== id || ports.outputs.some((p) => p.id === w.from[1])));
      this.doc.wires = this.doc.wires.filter((w) => w.from[0] !== id || WF.canWire(Object.assign({}, this.doc, { wires: this.doc.wires.filter((x) => x !== w) }), w.from, w.to) === true);
      this.commit('edit ' + id);
    }
    rename(name) { this.doc.name = name; this.commit('rename'); }
    select(ids) { this.sel = new Set(ids); this.selWire = null; this.render(); if (this.opts.onSelect) this.opts.onSelect([...this.sel]); }

    // ---- drawing --------------------------------------------------------------------------------------------------
    render() {
      if (this.disposed) return;
      const D = WF.Draw;
      this.crossing = D.crossingWires(this.doc);
      this.wireLayer.innerHTML = D.wires(this.doc, this.selWire, this.crossing);
      this.renderBlocks();
      this.empty.style.display = this.doc.blocks.length ? 'none' : '';
    }
    /** draw the blocks; a block whose drawing has not changed keeps its element (a double-click needs the same one) */
    renderBlocks() {
      const D = WF.Draw, old = new Map([...this.blockLayer.children].map((g) => [g.dataset.id, g]));
      const tmp = svgEl('g');
      const frag = [];
      for (const b of this.doc.blocks) {
        const html = D.block(b, this.doc, this.states[b.id], { sel: this.sel.has(b.id) });
        const g0 = old.get(b.id);
        if (g0 && g0._html === html) { frag.push(g0); continue; }
        tmp.innerHTML = html;
        const g = tmp.firstChild;
        g._html = html;
        g._fresh = true;
        frag.push(g);
      }
      const same = frag.length === this.blockLayer.children.length && frag.every((g, i) => this.blockLayer.children[i] === g);
      if (!same) this.blockLayer.replaceChildren(...frag);
      this.markClipped();
      for (const g of frag) if (g._fresh) { g._fresh = false; this.fitTexts(g); }
    }
    /** words that would run past their room in a block are cut, with an ellipsis (the whole text is its tooltip) */
    fitTexts(g) {
      const W = WF.dims(this.doc).W;
      const room = (t) => (t.classList.contains('wf-btitle') ? W - 40 : t.classList.contains('wf-bsum') ? W - 26 : W / 2 - 18);
      for (const t of g.querySelectorAll('text')) {
        let len = 0;
        try { len = t.getComputedTextLength(); } catch (e) { return; }
        const max = room(t);
        if (!len || len <= max) continue;
        const full = t.textContent;
        let n = full.length;
        while (n > 3 && t.getComputedTextLength() > max) { n--; t.textContent = full.slice(0, n).trimEnd() + '…'; }
        const tt = document.createElementNS(NS, 'title'); tt.textContent = full; t.appendChild(tt);
      }
    }
    /** the run's states for the blocks (from the Remote): redrawn at most once a frame */
    setStates(states) {
      this.states = states || {};
      if (this.statePending) return;
      this.statePending = true;
      requestAnimationFrame(() => { this.statePending = false; if (!this.drag) this.renderBlocks(); });
    }
    applyView() { const v = this.view; this.root.setAttribute('transform', 'translate(' + v.x.toFixed(1) + ',' + v.y.toFixed(1) + ') scale(' + v.z.toFixed(4) + ')'); this.wrap.style.setProperty('--wf-z', v.z.toFixed(3)); this.markClipped(); }
    /**
     * the canvas is a window on a larger plane: a block that is not wholly in view (panned past an edge, or under the
     * viewer panel) is marked as out of the picture, for the reader's tooltip and for the drawing audit alike
     */
    markClipped() {
      const W = this.wrap.clientWidth, H = this.wrap.clientHeight, v = this.view;
      if (!W || !H) return;
      const ins = (this.opts.inset && this.opts.inset()) || {};
      const x0 = -v.x / v.z, y0 = -v.y / v.z, x1 = (W - (ins.right || 0) - v.x) / v.z, y1 = (H - (ins.bottom || 0) - v.y) / v.z;
      for (const g of this.blockLayer.children) {
        const b = WF.byId(this.doc, g.dataset.id);
        if (!b) continue;
        const gg = WF.geom(b, this.doc), out = b.x - 8 < x0 || b.y - 8 < y0 || b.x + gg.w + 8 > x1 || b.y + gg.h + 8 > y1;
        if (out) g.setAttribute('data-audit', 'ignore'); else g.removeAttribute('data-audit');
        g.classList.toggle('wf-clip', out);
      }
      // a wire with an end out of view leaves the picture with it
      const off = new Set([...this.blockLayer.children].filter((g) => g.classList.contains('wf-clip')).map((g) => g.dataset.id));
      for (const g of this.wireLayer.children) {
        const w = this.doc.wires.find((x) => x.id === g.dataset.wire);
        const out = !!w && (off.has(w.from[0]) || off.has(w.to[0]));
        if (out) g.setAttribute('data-audit', 'ignore'); else g.removeAttribute('data-audit');
      }
    }
    bounds() { return this.boundsOf(this.doc); }
    boundsOf(doc) {
      if (!doc.blocks.length) return null;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const b of doc.blocks) { const g = WF.geom(b, doc); x0 = Math.min(x0, b.x - 10); y0 = Math.min(y0, b.y - 10); x1 = Math.max(x1, b.x + g.w + 10); y1 = Math.max(y1, b.y + g.h + 10); }
      // the wires' lanes and the routes around blocks, as drawn
      for (const g of WF.wireGeom(doc)) for (const p of g.via || []) { x0 = Math.min(x0, p.x - 10); x1 = Math.max(x1, p.x + 10); y0 = Math.min(y0, p.y - 10); y1 = Math.max(y1, p.y + 10); }
      return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    }
    /** fit the whole workflow; never so small that its words fall under the size a reader needs (pan for the rest) */
    fit() {
      const b = this.bounds(), ins = Object.assign({}, (this.opts.inset && this.opts.inset()) || {});
      // a compact canvas keeps its buttons over the drawing (the blocks at the top left, zoom at the top right): the
      // workflow fits below and beside them
      const top = this.opts.compact ? 40 : 0;
      if (this.opts.compact) ins.right = Math.max(ins.right || 0, 44);
      const W = (this.wrap.clientWidth || 800) - (ins.right || 0), H = (this.wrap.clientHeight || 500) - (ins.bottom || 0) - top;
      if (!b) { this.view = { x: 40, y: 40, z: 1 }; this.applyView(); return; }
      // on a phone the whole workflow shows, however small (pinch to read it); elsewhere its words stay readable
      const phone = innerWidth < 600;
      const fitZ = Math.min((W - 40) / b.w, (H - 40) / b.h);
      // a compact canvas has no port names, so its smallest words (the summaries, 12.5 px there) allow a little less
      const z = phone ? Math.max(0.2, Math.min(1, fitZ)) : Math.max(this.opts.compact ? 0.88 : 0.96, Math.min(1.15, fitZ));
      this.userView = false;
      const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
      this.view = { z, x: W / 2 - cx * z, y: top + H / 2 - cy * z };
      // too big to fit: start at the top left
      if (b.w * z > W - 20) this.view.x = 20 - b.x * z;
      if (b.h * z > H - 20) this.view.y = top + 16 - b.y * z;
      this.applyView();
    }
    zoomBy(f, cx, cy) {
      this.userView = true;
      const r = this.wrap.getBoundingClientRect();
      if (cx === undefined) { cx = r.left + r.width / 2; cy = r.top + r.height / 2; }
      const z = Math.max(0.3, Math.min(2.5, this.view.z * f)), px = cx - r.left, py = cy - r.top;
      this.view.x = px - ((px - this.view.x) * z) / this.view.z; this.view.y = py - ((py - this.view.y) * z) / this.view.z; this.view.z = z;
      this.applyView();
    }
    ensureVisible(b) {
      const ins = (this.opts.inset && this.opts.inset()) || {};
      const W = this.wrap.clientWidth - (ins.right || 0), H = this.wrap.clientHeight - (ins.bottom || 0), z = this.view.z, g = WF.geom(b, this.doc);
      const sx = b.x * z + this.view.x, sy = b.y * z + this.view.y;
      if (sx < 10) this.view.x += 20 - sx; else if (sx + g.w * z > W - 10) this.view.x -= sx + g.w * z - W + 20;
      if (sy < 10) this.view.y += 20 - sy; else if (sy + g.h * z > H - 10) this.view.y -= sy + g.h * z - H + 20;
      this.applyView();
    }
    toWorld(cx, cy) { const r = this.wrap.getBoundingClientRect(); return { x: (cx - r.left - this.view.x) / this.view.z, y: (cy - r.top - this.view.y) / this.view.z }; }
    snap(v) { const g = WF.dims(this.doc).GRID; return Math.round(v / g) * g; }
    say(text, x, y) {
      if (!text) { this.tip.classList.remove('on'); return; }
      const r = this.wrap.getBoundingClientRect();
      this.tip.textContent = text;
      this.tip.classList.add('on');
      this.tip.style.left = Math.min(r.width - 240, Math.max(8, x - r.left + 14)) + 'px';
      this.tip.style.top = Math.max(8, y - r.top + 16) + 'px';
    }

    // ---- pointer and keys -----------------------------------------------------------------------------------------
    bind() {
      const pointers = new Map();
      this.onMove = (e) => this.move(e, pointers);
      this.onUp = (e) => this.up(e, pointers);
      this.svg.addEventListener('pointerdown', (e) => this.down(e, pointers));
      window.addEventListener('pointermove', this.onMove);
      window.addEventListener('pointerup', this.onUp);
      window.addEventListener('pointercancel', this.onUp);
      this.svg.addEventListener('dblclick', (e) => { const bl = e.target.closest('.wf-block'); if (bl && !e.target.closest('.wf-port')) { e.preventDefault(); if (this.opts.onEdit && !this.opts.readOnly) this.opts.onEdit(bl.dataset.id); else if (this.opts.onOpen) this.opts.onOpen(bl.dataset.id); } });
      this.wrap.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) this.zoomBy(Math.exp(-e.deltaY * 0.0022), e.clientX, e.clientY);
        else { this.userView = true; this.view.x -= e.deltaX; this.view.y -= e.deltaY; this.applyView(); }
      }, { passive: false });
      this.wrap.addEventListener('keydown', (e) => { if (this.key(e)) { e.preventDefault(); e.__wf = true; } });
    }
    down(e, pointers) {
      if (e.button === 2) return;
      this.wrap.focus({ preventScroll: true });
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) { const [a, b] = [...pointers.values()]; this.drag = { kind: 'pinch', d0: Math.hypot(a.x - b.x, a.y - b.y), z0: this.view.z, v0: Object.assign({}, this.view), c: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } }; return; }
      const port = e.target.closest('.wf-port'), bl = e.target.closest('.wf-block'), wire = e.target.closest('.wf-wire-g');
      const p = this.toWorld(e.clientX, e.clientY);
      if (port && bl && !this.opts.readOnly) {
        e.preventDefault();
        const id = bl.dataset.id, side = port.dataset.side, name = port.dataset.port;
        if (side === 'out') this.startWire([id, name], null, e);
        else {
          const w = this.doc.wires.find((x) => x.to[0] === id && x.to[1] === name);
          if (w) { this.doc.wires = this.doc.wires.filter((x) => x !== w); this.render(); this.startWire(w.from.slice(), null, e, true); }
          else this.startWire(null, [id, name], e);
        }
        return;
      }
      if (bl) {
        const id = bl.dataset.id;
        if (e.shiftKey || e.metaKey || e.ctrlKey) { if (this.sel.has(id)) this.sel.delete(id); else this.sel.add(id); }
        else if (!this.sel.has(id)) this.sel = new Set([id]);
        this.selWire = null;
        this.renderBlocks(); this.wireLayer.innerHTML = WF.Draw.wires(this.doc, null, this.crossing);
        if (this.opts.onSelect) this.opts.onSelect([...this.sel]);
        if (!this.opts.readOnly) this.drag = { kind: 'move', start: p, orig: [...this.sel].map((x) => { const b = WF.byId(this.doc, x); return [b, b.x, b.y]; }), moved: false, id };
        else this.drag = { kind: 'click', id };
        e.preventDefault();
        return;
      }
      if (wire) {
        this.selWire = wire.dataset.wire; this.sel.clear();
        this.render();
        if (this.opts.onSelect) this.opts.onSelect([]);
        return;
      }
      // the background: a touch pans, a mouse draws a selection box (with the middle button or Space it pans)
      if (e.pointerType === 'touch' || e.button === 1 || this.space) this.drag = { kind: 'pan', sx: e.clientX, sy: e.clientY, v0: Object.assign({}, this.view) };
      else this.drag = { kind: 'box', start: p, add: e.shiftKey ? new Set(this.sel) : new Set() };
      if (!e.shiftKey) { this.sel.clear(); this.selWire = null; this.render(); if (this.opts.onSelect) this.opts.onSelect([]); }
    }
    startWire(from, to, e, rewire) {
      const doc = this.doc;
      // which ports may take this wire, and why the others may not
      const verdict = new Map();
      for (const b of doc.blocks) {
        const g = WF.geom(b, doc);
        if (from) for (const q of g.ins) verdict.set(b.id + '|in|' + q.id, WF.canWire(doc, from, [b.id, q.id]));
        else for (const q of g.outs) verdict.set(b.id + '|out|' + q.id, WF.canWire(doc, [b.id, q.id], to));
      }
      for (const el of this.blockLayer.querySelectorAll('.wf-port')) {
        const k = el.closest('.wf-block').dataset.id + '|' + el.dataset.side + '|' + el.dataset.port;
        if (verdict.has(k)) el.classList.add(verdict.get(k) === true ? 'can' : 'cannot');
      }
      this.wrap.classList.add('wiring');
      const ghost = svgEl('path', { class: 'wf-ghost' });
      this.ui.appendChild(ghost);
      this.drag = { kind: 'wire', from, to, verdict, ghost, rewire };
      this.move(e, new Map());
    }
    portUnder(cx, cy) {
      const el = document.elementFromPoint(cx, cy);
      const port = el && el.closest && el.closest('.wf-port');
      if (port) return { id: port.closest('.wf-block').dataset.id, side: port.dataset.side, port: port.dataset.port };
      const bl = el && el.closest && el.closest('.wf-block');
      if (bl && this.drag && this.drag.kind === 'wire') {
        // over a block but not a port: its first port that can take the wire (a free one first)
        const id = bl.dataset.id, g = WF.geom(WF.byId(this.doc, id), this.doc), side = this.drag.from ? 'in' : 'out';
        const list = (side === 'in' ? g.ins : g.outs).filter((q) => this.drag.verdict.get(id + '|' + side + '|' + q.id) === true);
        const free = list.find((q) => side === 'out' || !this.doc.wires.some((w) => w.to[0] === id && w.to[1] === q.id));
        const q = free || list[0];
        if (q) return { id, side, port: q.id, guessed: true };
      }
      return null;
    }
    move(e, pointers) {
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const d = this.drag;
      if (!d) return;
      const p = this.toWorld(e.clientX, e.clientY);
      if (d.kind === 'pinch') {
        const [a, b] = [...pointers.values()];
        if (!a || !b) return;
        const f = Math.hypot(a.x - b.x, a.y - b.y) / Math.max(1, d.d0);
        this.view = Object.assign({}, d.v0); this.zoomBy((d.z0 * f) / this.view.z, d.c.x, d.c.y);
        return;
      }
      if (d.kind === 'pan') { this.userView = true; this.view.x = d.v0.x + e.clientX - d.sx; this.view.y = d.v0.y + e.clientY - d.sy; this.applyView(); return; }
      if (d.kind === 'move') {
        const dx = p.x - d.start.x, dy = p.y - d.start.y;
        if (!d.moved && Math.hypot(dx, dy) * this.view.z < 4) return;
        d.moved = true;
        for (const [b, x, y] of d.orig) { b.x = x + dx; b.y = y + dy; }
        const moving = new Set(d.orig.map(([b]) => b.id));
        for (const w of this.doc.wires) if (w.via && (moving.has(w.from[0]) || moving.has(w.to[0]))) delete w.via;
        this.render();
        return;
      }
      if (d.kind === 'box') {
        const x0 = Math.min(d.start.x, p.x), y0 = Math.min(d.start.y, p.y), x1 = Math.max(d.start.x, p.x), y1 = Math.max(d.start.y, p.y);
        if (!d.rect) { d.rect = svgEl('rect', { class: 'wf-boxsel' }); this.ui.appendChild(d.rect); }
        d.rect.setAttribute('x', x0); d.rect.setAttribute('y', y0); d.rect.setAttribute('width', x1 - x0); d.rect.setAttribute('height', y1 - y0);
        this.sel = new Set(d.add);
        for (const b of this.doc.blocks) { const g = WF.geom(b, this.doc); if (b.x < x1 && b.x + g.w > x0 && b.y < y1 && b.y + g.h > y0) this.sel.add(b.id); }
        this.renderBlocks();
        return;
      }
      if (d.kind === 'wire') {
        const end = d.from ? WF.portAt(this.doc, d.from[0], d.from[1], 'out') : WF.portAt(this.doc, d.to[0], d.to[1], 'in');
        const u = this.portUnder(e.clientX, e.clientY);
        let tgt = p;
        if (u && u.side === (d.from ? 'in' : 'out')) { const v = d.verdict.get(u.id + '|' + u.side + '|' + u.port); if (v === true) { tgt = WF.portAt(this.doc, u.id, u.port, u.side); this.say(''); } else this.say(v ? 'Cannot wire: ' + v : '', e.clientX, e.clientY); }
        else this.say('');
        d.ghost.setAttribute('d', d.from ? WF.wirePath(end, tgt) : WF.wirePath(tgt, end));
      }
    }
    up(e, pointers) {
      pointers.delete(e.pointerId);
      const d = this.drag;
      if (!d) return;
      if (d.kind === 'pinch') { if (pointers.size < 2) this.drag = null; return; }
      this.drag = null;
      if (d.kind === 'move') {
        if (d.moved) { for (const [b] of d.orig) { b.x = this.snap(b.x); b.y = this.snap(b.y); } this.commit('move'); }
        else if (this.opts.onOpen && !e.shiftKey && !e.metaKey && !e.ctrlKey) this.opts.onOpen(d.id);
        return;
      }
      if (d.kind === 'click') { if (this.opts.onOpen) this.opts.onOpen(d.id); return; }
      if (d.kind === 'box') { if (d.rect) d.rect.remove(); if (this.opts.onSelect) this.opts.onSelect([...this.sel]); return; }
      if (d.kind === 'wire') {
        d.ghost.remove();
        this.wrap.classList.remove('wiring');
        this.say('');
        const u = this.portUnder(e.clientX, e.clientY);
        let made = false;
        if (u && u.side === (d.from ? 'in' : 'out')) {
          const from = d.from || [u.id, u.port], to = d.to || [u.id, u.port];
          const v = WF.canWire(this.doc, from, to);
          if (v === true) { WF.connect(this.doc, from, to); made = true; }
          else S.UI.toast('Cannot wire: ' + v, 'err', 3500);
        }
        if (made || d.rewire) this.commit(made ? 'wire' : 'unwire');
        else this.render();
      }
    }
    /** the editor's keys; true when the key was used */
    key(e, force) {
      const k = e.key, mod = e.ctrlKey || e.metaKey;
      if (e.target !== this.wrap && !force) return false;
      if (k === ' ') { this.space = true; setTimeout(() => { this.space = false; }, 1500); return true; }
      if (mod && (k === 'z' || k === 'Z')) { if (e.shiftKey) this.redo(); else this.undo(); return true; }
      if (mod && (k === 'y' || k === 'Y')) { this.redo(); return true; }
      if (mod && (k === 'c' || k === 'C')) return this.copy();
      if (mod && (k === 'x' || k === 'X')) { if (this.copy()) this.deleteSelection(); return true; }
      if (mod && (k === 'v' || k === 'V')) return this.paste();
      if (mod && (k === 'a' || k === 'A')) { this.select(this.doc.blocks.map((b) => b.id)); return true; }
      if (mod || e.altKey) return false;
      if (k === 'Delete' || k === 'Backspace') { this.deleteSelection(); return true; }
      if (k === 'f' || k === 'F') { this.fit(); return true; }
      if ((k === 'l' || k === 'L') && !this.opts.readOnly) { this.tidy(); return true; }
      if (k === '+' || k === '=') { this.zoomBy(1.2); return true; }
      if (k === '-') { this.zoomBy(1 / 1.2); return true; }
      if (k === 'Escape' && this.opts.onEscape && this.opts.onEscape()) return true;
      if (k === 'Escape' && (this.sel.size || this.selWire)) { this.select([]); return true; }
      const one = this.sel.size === 1 ? [...this.sel][0] : null;
      if (k === 'Enter' && one) { if (this.opts.onEdit && !this.opts.readOnly) this.opts.onEdit(one); return true; }
      if ((k === 'o' || k === 'O') && one) { if (this.opts.onOpen) this.opts.onOpen(one); return true; }
      if (k.startsWith('Arrow') && this.sel.size && !this.opts.readOnly) {
        const step = WF.dims(this.doc).GRID * (e.shiftKey ? 4 : 1), dx = k === 'ArrowLeft' ? -step : k === 'ArrowRight' ? step : 0, dy = k === 'ArrowUp' ? -step : k === 'ArrowDown' ? step : 0;
        for (const id of this.sel) { const b = WF.byId(this.doc, id); b.x += dx; b.y += dy; }
        for (const w of this.doc.wires) if (w.via && (this.sel.has(w.from[0]) || this.sel.has(w.to[0]))) delete w.via;
        this.commit('nudge');
        return true;
      }
      return false;
    }
  }
  Editor.clip = null;
  WF.Editor = Editor;
})(globalThis.Nooral = globalThis.Nooral || {});
