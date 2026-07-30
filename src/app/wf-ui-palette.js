// The palette of the canvas: every block in its category, searchable, each category opened or closed (remembered).
// Drag a block onto the canvas, or click it to drop it in the middle of the view; a block's blurb is its tooltip.
//   new S.WF.Palette(host, {editor: () => editor, allowed: [types] | null, onPlace(block)})
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const { h } = S.UI;
  const store = S.UI.storage;

  class Palette {
    constructor(host, opts) {
      this.opts = opts || {};
      this.host = host;
      this.search = h('input.wf-pal-search', { type: 'search', placeholder: 'Find a block', 'aria-label': 'Find a block', id: this.opts.idp ? this.opts.idp + 'palSearch' : 'wfPalSearch' });
      this.search.addEventListener('input', S.UI.debounce(() => this.render(), 60));
      this.search.addEventListener('keydown', (e) => { if (e.key === 'Escape') { this.search.value = ''; this.render(); } if (e.key === 'Enter') { const b = this.list.querySelector('.wf-pal-item'); if (b) b.click(); } });
      this.list = h('div.wf-pal-list');
      this.open = new Set(store.get('wfPalOpen', ['Data', 'Prepare', 'Split', 'Models', 'Train', 'Evaluate']));
      host.append(this.search, this.list);
      this.render();
    }
    groups() {
      const allowed = this.opts.allowed;
      return WF.CATS.map((cat) => ({ cat, items: WF.list(cat).filter((d) => !d.hidden && (!allowed || allowed.includes(d.type))) })).filter((g) => g.items.length);
    }
    render() {
      const q = this.search.value.trim().toLowerCase();
      this.list.innerHTML = '';
      let any = false;
      for (const g of this.groups()) {
        const items = g.items.filter((d) => !q || (d.label + ' ' + d.type + ' ' + g.cat + ' ' + (d.blurb || '')).toLowerCase().includes(q));
        if (q && !items.length) continue;
        any = true;
        const isOpen = this.open.has(g.cat) || !!q || !!this.opts.allowed;
        const sec = h('section.wf-pal-cat.' + WF.Draw.catCls(g.cat) + (isOpen ? '.open' : ''), { dataset: { cat: g.cat } });
        const head = h('button.wf-pal-head', { type: 'button', 'aria-expanded': String(isOpen) }, h('span.wf-pal-dot'), h('span', g.cat), h('span.n', String(g.items.length)));
        head.addEventListener('click', () => {
          if (this.open.has(g.cat)) this.open.delete(g.cat); else this.open.add(g.cat);
          store.set('wfPalOpen', [...this.open]);
          sec.classList.toggle('open');
          head.setAttribute('aria-expanded', String(sec.classList.contains('open')));
        });
        const grid = h('div.wf-pal-grid');
        for (const d of items) grid.appendChild(this.item(d));
        sec.append(head, grid);
        this.list.appendChild(sec);
      }
      if (!any) this.list.appendChild(h('p.wf-pal-none', 'No block matches "' + q + '". Try a shorter word: split, tree, scale, roc.'));
    }
    item(d) {
      const b = h('button.wf-pal-item', { type: 'button', title: d.label + ': ' + (d.blurb || '') + ' Drag it onto the canvas, or click to add it.', dataset: { type: d.type } }, h('span.wf-pal-sym', { html: WF.Draw.glyph(d) }), h('span.wf-pal-name', d.label));
      let drag = null;
      b.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        drag = { x: e.clientX, y: e.clientY, moved: false };
        try { b.setPointerCapture(e.pointerId); } catch (x) { /* synthetic */ }
      });
      b.addEventListener('pointermove', (e) => {
        if (!drag) return;
        if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 6) return;
        if (!drag.moved) { drag.moved = true; drag.ghost = h('div.wf-pal-ghost', h('span', { html: WF.Draw.glyph(d) }), h('span', d.label)); document.body.appendChild(drag.ghost); document.body.classList.add('wf-dragging'); }
        drag.ghost.style.left = e.clientX + 'px'; drag.ghost.style.top = e.clientY + 'px';
      });
      const end = (e, cancel) => {
        if (!drag) return;
        const dd = drag; drag = null;
        document.body.classList.remove('wf-dragging');
        if (dd.ghost) dd.ghost.remove();
        if (cancel || !dd.moved) return;
        b._suppress = true; setTimeout(() => { b._suppress = false; }, 0);
        const ed = this.opts.editor && this.opts.editor();
        if (!ed) return;
        const placed = ed.dropAt(d.type, e.clientX, e.clientY);
        if (placed && this.opts.onPlace) this.opts.onPlace(placed);
      };
      b.addEventListener('pointerup', (e) => end(e));
      b.addEventListener('pointercancel', (e) => end(e, true));
      b.addEventListener('click', () => {
        if (b._suppress) return;
        const ed = this.opts.editor && this.opts.editor();
        if (!ed) return;
        const placed = ed.addBlock(d.type);
        if (placed && this.opts.onPlace) this.opts.onPlace(placed, true);
      });
      return b;
    }
  }
  WF.Palette = Palette;
})(globalThis.Nooral = globalThis.Nooral || {});
