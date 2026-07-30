// A workspace: the palette, the canvas and the viewer of the open block, joined to a Remote that runs the workflow in
// its worker. The canvas page (wf-ui-page.js) is one, with a toolbar around it; S.WF.embed puts the same thing inside a
// lesson, an exercise or a challenge.
//   new S.WF.Workspace(host, {doc, readOnly, allowed, onChange(doc), idp, page})
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const { h, icon } = S.UI;
  const LEGEND = [['table', 'table'], ['images', 'images'], ['text', 'text'], ['model', 'model'], ['predictions', 'predictions']];

  class Workspace {
    constructor(host, opts) {
      this.opts = opts || {};
      this.readOnly = !!this.opts.readOnly;
      this.el = h('div.wf-ws' + (this.readOnly ? '.ro' : '') + (this.opts.page ? '.wf-ws-page' : '.wf-ws-embed'));
      host.appendChild(this.el);
      this.palHost = h('aside.wf-pal', { 'aria-label': 'Blocks' });
      this.canvasHost = h('div.wf-canvas');
      this.dock = h('div.wf-dock');
      const scrim = h('div.wf-scrim', { 'aria-hidden': 'true' });
      scrim.addEventListener('click', () => { this.el.classList.remove('pal-open'); if (this.narrow()) this.closeViewer(); });
      if (!this.readOnly) this.el.appendChild(this.palHost);
      this.el.append(this.canvasHost, this.dock, scrim);
      this.legend = h('div.wf-legend', { 'aria-label': 'What the wires carry' }, ...LEGEND.map(([t, l]) => h('span', h('i.wf-lgsw.t-' + t), l)));
      this.compact = !this.opts.page;
      this.fitUI();
      if (this.compact && !this.readOnly) {
        const pb = h('button.btn.small.wf-pal-in', { type: 'button', title: 'The blocks', 'aria-label': 'The blocks' }, icon('plus'), h('span', 'Blocks'));
        pb.addEventListener('click', () => this.el.classList.toggle('pal-open'));
        this.palBtn = pb;
      }
      this.remote = new WF.Remote({ onEvent: (e) => this.onEvent(e) });
      this.editor = new WF.Editor(this.canvasHost, {
        doc: this.opts.doc, readOnly: this.readOnly, allowed: this.opts.allowed,
        // inside a lesson or a challenge: compact blocks, laid out afresh to fit the width it has
        compact: this.compact, tidy: this.compact && this.opts.tidy !== false,
        maxLayers: () => { const g = WF.dims({ _compact: this.compact }), w = Math.max(300, (this.canvasHost.clientWidth || this.el.clientWidth || 800) - 40); return Math.max(2, Math.floor((w / (this.compact ? 0.88 : 0.96) + g.GAPX) / (g.W + g.GAPX))); },
        onChange: (doc, why) => this.changed(doc, why),
        onOpen: (id) => this.open(id),
        onEdit: (id) => this.editParams(id),
        onSelect: () => {},
        onEscape: () => { if (!this.viewer.open) return false; this.closeViewer(); return true; },
        inset: () => (!this.el.classList.contains('view-open') ? {} : this.narrow() ? { bottom: this.dock.offsetHeight } : { right: this.dock.offsetWidth + 10 }),
      });
      this.editor.wrap.appendChild(this.legend);
      if (this.palBtn) this.editor.wrap.appendChild(this.palBtn);
      this.ro = new ResizeObserver(() => this.fitUI());
      this.ro.observe(this.el);
      if (!this.readOnly) this.palette = new WF.Palette(this.palHost, { editor: () => this.editor, allowed: this.opts.allowed, idp: this.opts.idp, onPlace: () => { if (this.narrow() || this.el.classList.contains('compact-ui')) this.el.classList.remove('pal-open'); } });
      this.viewer = new WF.ViewerPanel(this.dock, this);
      this.remote.load(this.editor.doc);
    }
    get doc() { return this.editor.doc; }
    /** a workspace narrower than a desktop canvas keeps its palette in a drawer */
    fitUI() { const w = this.el.clientWidth; if (!w) return; this.el.classList.toggle('compact-ui', this.compact && w < 1000); this.el.classList.toggle('narrow-ui', w < 640); }
    narrow() { return (this.el.clientWidth || innerWidth) < 640 || innerWidth < 760; }
    onEvent(e) {
      if (e.type === 'state') this.editor.setStates(this.remote.states);
      this.viewer.event(e);
      if (this.opts.onEvent) this.opts.onEvent(e);
    }
    changed(doc, why) {
      this.remote.load(doc);
      this.viewer.refreshHead();
      if (this.opts.onChange) this.opts.onChange(doc, why);
    }
    /** a whole new document (open, a sample, a share link) */
    loadDoc(doc) {
      this.closeViewer();
      this.editor.setDoc(doc);
      this.editor.hist = [JSON.stringify(this.editor.doc)]; this.editor.hi = 0;
      this.remote.load(this.editor.doc);
      if (this.opts.onChange) this.opts.onChange(this.editor.doc, 'load');
      return this.whenIdle();
    }
    loadSample(id) {
      const doc = WF.sample(id);
      if (!doc) return Promise.reject(new Error('there is no sample called ' + id));
      // every sample opens laid out by the tidy-up, whatever positions it came with
      WF.tidy(doc);
      return this.loadDoc(doc);
    }
    whenIdle() { return this.remote.idle(); }
    open(id) {
      if (!WF.byId(this.doc, id)) return;
      this.viewer.show(id);
      this.el.classList.add('view-open');
      this.editor.select([id]);
      requestAnimationFrame(() => { const b = WF.byId(this.doc, id); if (!b) return; this.editor.fit(); this.editor.ensureVisible(b); });
    }
    closeViewer() { const was = this.viewer.open; this.viewer.close(); this.el.classList.remove('view-open'); if (was) requestAnimationFrame(() => this.editor.fit()); }
    /** what flows into a block, for its dialog: the feature names of the table upstream */
    upstream(id) {
      const f = WF.feeds(this.doc, id), src = f.in || f.data || Object.values(f)[0];
      if (!src) return {};
      const v = this.remote.views[src[0]] || {};
      const feats = v.features || (v.head && v.head.features) || v.columns || [];
      return { features: feats.map((x) => (typeof x === 'string' ? x : x.name)), view: v };
    }
    async editParams(id) {
      if (this.readOnly) return;
      const b = WF.byId(this.doc, id);
      if (!b) return;
      const patch = await WF.paramDialog(b, { doc: this.doc, upstream: this.upstream(id) });
      if (patch) this.setParams(id, patch);
    }
    setParams(id, patch) { this.editor.setParams(id, patch); }
    redraw() { this.editor.render(); this.viewer.redraw(); }
    dispose() { if (this.ro) this.ro.disconnect(); this.remote.dispose(); this.viewer.close(); this.editor.dispose(); }
  }
  WF.Workspace = Workspace;

  /** the canvas inside a lesson, an exercise or a challenge */
  WF.embed = function (host, doc, opts) {
    opts = opts || {};
    // the height given, else the height the host was given, else 460 px
    const given = opts.height || (host.style && parseFloat(host.style.height)) || 0;
    const box = h('div.wf-embed', { style: { height: (given || 460) + 'px' } });
    host.appendChild(box);
    const ws = new Workspace(box, { doc, readOnly: !!opts.readOnly, allowed: opts.allowed || null, onChange: opts.onChange, onEvent: opts.onEvent, idp: 'emb' });
    return { doc: () => ws.doc, remote: ws.remote, editor: ws.editor, workspace: ws, open: (id) => ws.open(id), whenIdle: () => ws.whenIdle(), loadDoc: (d) => ws.loadDoc(d), dispose: () => ws.dispose(), redraw: () => ws.redraw() };
  };
  void icon;
})(globalThis.Nooral = globalThis.Nooral || {});
