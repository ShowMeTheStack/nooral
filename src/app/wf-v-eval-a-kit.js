// What the evaluation and explanation viewers share: a figure (a stage laid out in its own pixels, so text keeps its
// size at any width), small HTML pieces (a row of choices, a table of scores, probability bars), image thumbnails, and
// a safe wrapper round the viewer API the canvas hands over.
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const VK = (WF.VK = WF.VK || {});
  const h = (...a) => S.UI.h(...a);
  VK.fmt = (x, d) => (x === null || x === undefined || !Number.isFinite(+x) ? '–' : String(+(+x).toFixed(d === undefined ? 3 : d)).replace(/^-/, '−'));
  VK.pct = (x) => (Number.isFinite(x) ? (100 * x).toFixed(1).replace(/\.0$/, '') + '%' : '–');
  /** the width a viewer lays its figures out in: the host's, within reason */
  VK.width = (host) => { let w = host && host.clientWidth; for (let e = host && host.parentElement; !w && e; e = e.parentElement) w = e.clientWidth; if (!w && typeof innerWidth !== 'undefined') w = innerWidth - 32; return Math.max(280, Math.min(980, Math.floor(w || 640))); };
  VK.narrow = (host, api) => (api && api.narrow ? api.narrow() : VK.width(host) < 560);
  /** the class names, shortened for small cells */
  VK.short = (s, n) => { s = String(s); return s.length > (n || 12) ? s.slice(0, (n || 12) - 1) + '…' : s; };

  /** the API with every piece optional: a viewer may run in a lesson or a test without the canvas */
  VK.api = (api) => Object.assign({ call: () => Promise.reject(new Error('this viewer cannot ask the model here')), setParams: null, control() {}, onCurve() {}, narrow: null }, api || {});

  /**
   * a figure: a stage whose camera shows exactly the W × H box it was laid out in, sized to keep that ratio.
   * fig.show(items, W, H) draws; fig.onPick(key) is called on a click of a pickable item
   */
  VK.fig = function (label, cls) {
    const el = h('div.wf-fig' + (cls ? '.' + cls : ''));
    const box = h('div.wf-fig-stage');
    el.appendChild(box);
    const stage = new S.Stage(box, { label });
    const fig = { el, stage, W: 0, H: 0, onPick: null };
    stage.onPick = (k, e) => { if (fig.onPick) fig.onPick(k, e); };
    fig.show = function (items, W, H, instant) {
      fig.W = W; fig.H = H;
      box.style.aspectRatio = W + ' / ' + H;
      box.style.width = W + 'px'; box.style.maxWidth = '100%';
      stage.show({ items, view: { x: 0, y: 0, w: W, h: H } }, { instant: instant !== false, duration: 260 });
    };
    return fig;
  };
  /** a row of choices (segmented buttons): opts [[value, label]], on(value) */
  VK.seg = function (label, opts, value, on) {
    const box = h('div.seg.wf-seg', { role: 'group', 'aria-label': label });
    for (const [v, l] of opts) {
      const b = h('button', { type: 'button', 'aria-pressed': String(v === value), class: v === value ? 'on' : '' }, l);
      b.addEventListener('click', () => { for (const c of box.children) { c.classList.remove('on'); c.setAttribute('aria-pressed', 'false'); } b.classList.add('on'); b.setAttribute('aria-pressed', 'true'); on(v); });
      box.appendChild(b);
    }
    return box;
  };
  /** a labelled select: opts [[value, label]] */
  VK.select = function (label, opts, value, on, id) {
    const s = h('select', { 'aria-label': label, id }, ...opts.map(([v, l]) => h('option', { value: String(v), selected: String(v) === String(value) }, l)));
    s.addEventListener('change', () => on(s.value));
    return h('label.wf-pick', h('span', label), s);
  };
  /** a small table: head [..], rows [[..]], with cells that may be elements */
  VK.table = function (head, rows, cls) {
    return h('div.fit-table', h('table.wf-tab' + (cls ? '.' + cls : ''), head ? h('thead', h('tr', ...head.map((x) => h('th', x)))) : null, h('tbody', ...rows.map((r) => h('tr', ...r.map((x) => (x && x.nodeType ? h('td', x) : h('td', x === undefined ? '' : String(x)))))))));
  };
  /** probability bars, one per class, the largest marked */
  VK.bars = function (names, probs, pick) {
    const top = probs.indexOf(Math.max(...probs));
    const order = names.map((_, i) => i);
    if (names.length > 6) order.sort((a, b) => probs[b] - probs[a]);
    return h('div.wf-bars', ...order.slice(0, 10).map((i) => h('div.wf-bar' + (i === top ? '.top' : '') + (i < 4 ? '.c' + i : ''), { title: names[i] + ': ' + VK.pct(probs[i]) },
      h('span.wf-bar-name', VK.short(names[i], 16)), h('span.wf-bar-track', h('i', { style: { width: (100 * probs[i]).toFixed(1) + '%' } })), h('span.wf-bar-v', VK.pct(probs[i])), pick === i ? h('span.wf-bar-note', 'truth') : null)));
  };
  /** an image (grey levels 0..15, or 0..1 with max 1) as a small canvas, scaled up crisply */
  VK.thumb = function (px, w, hgt, scale, max, cls) {
    const c = document.createElement('canvas');
    c.width = w; c.height = hgt;
    c.className = 'wf-thumb' + (cls ? ' ' + cls : '');
    c.style.width = w * (scale || 3) + 'px'; c.style.height = hgt * (scale || 3) + 'px';
    const ctx = c.getContext && c.getContext('2d');
    if (ctx) {
      const im = ctx.createImageData(w, hgt), ink = S.themeRgb ? S.themeRgb('--text') : [20, 20, 20], paper = S.themeRgb ? S.themeRgb('--node') : [255, 255, 255];
      for (let i = 0; i < w * hgt; i++) { const t = Math.max(0, Math.min(1, (px[i] || 0) / (max || 15))); for (let k = 0; k < 3; k++) im.data[i * 4 + k] = paper[k] + (ink[k] - paper[k]) * t; im.data[i * 4 + 3] = 255; }
      ctx.putImageData(im, 0, 0);
    }
    c.setAttribute('role', 'img');
    return c;
  };
  /** a note under a figure, with `values` set in the monospace face */
  VK.note = (text, cls) => h('p.wf-note' + (cls ? '.' + cls : ''), ...String(text).split('`').map((p, i) => (i % 2 ? h('span.v', p) : p)));
  /** a viewer's frame: the host is cleared and given the class every viewer shares */
  VK.frame = function (host, cls) { host.innerHTML = ''; const box = h('div.wf-ev' + (cls ? '.wfv-' + cls : '')); host.appendChild(box); return box; };
  /** an error or an empty view, said plainly */
  VK.empty = (host, title, text) => { host.innerHTML = ''; host.appendChild(S.UI.empty('info', title, text)); };
  /** a viewer made from a draw(box, view, ctx) function: update and redraw draw it again, keeping its local state */
  VK.viewer = function (draw, cls) {
    return function (host, view, api) {
      api = VK.api(api);
      const state = {};
      let cur = view;
      const paint = () => {
        if (!cur) return VK.empty(host, 'Nothing to show yet', 'This block shows its picture once it has run.');
        if (cur.error) return VK.empty(host, 'This view could not be drawn', cur.error);
        const box = VK.frame(host, cls);
        try { draw(box, cur, { api, state, host, redraw: paint, W: VK.width(host), narrow: VK.narrow(host, api) }); S.UI.fitTables(box); requestAnimationFrame(() => S.UI.fitTables(box)); }
        catch (e) { VK.empty(host, 'This view could not be drawn', String(e.message || e)); if (typeof console !== 'undefined') console.error(e); }
      };
      paint();
      // laid out in pixels, so a viewer draws itself again when its box changes width (a panel opening, a phone turning)
      let lastW = host.clientWidth, ro = null;
      if (typeof ResizeObserver !== 'undefined') { ro = new ResizeObserver(() => { const w = host.clientWidth; if (w && Math.abs(w - lastW) > 16) { lastW = w; paint(); } }); ro.observe(host); }
      return { update(v) { cur = v; paint(); }, redraw: paint, dispose() { if (ro) ro.disconnect(); host.innerHTML = ''; } };
    };
  };
  /** ask the block to change a parameter (it runs again); without the canvas, change it here only */
  VK.set = (ctx, patch) => { if (ctx.api.setParams) ctx.api.setParams(patch); else { Object.assign(ctx.state, patch); ctx.redraw(); } };
})(globalThis.Nooral = globalThis.Nooral || {});
