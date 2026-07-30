// The Compare block's viewer: the models' scores side by side with the best of each marked, and for each model a
// panel: its boundary over the test points (on 2-D data) and its confusion matrix, or for numbers its predictions
// against the truth.
(function (S) {
  'use strict';
  const WF = S.WF, VK = WF.VK, h = (...a) => S.UI.h(...a);
  const C = () => S.Kit.curve;

  function compare(box, v, ctx) {
    const n = v.items.length, W = ctx.W, narrow = ctx.narrow;
    const head = ['model'].concat(v.metrics.map((m) => WF.METRICS[m].label + (WF.METRICS[m].higher ? ' ↑' : ' ↓')));
    const rows = v.items.map((it, i) => [h('span.wf-cmp-name', it.model)].concat(v.metrics.map((m) => (v.best[m] === i && n > 1 ? h('span.best', { title: 'the best of the ' + n }, VK.fmt(it.scores[m]) + ' ★') : VK.fmt(it.scores[m])))));
    box.append(h('div.wf-ev-head', h('div.wf-ev-title', h('b', n + ' model' + (n === 1 ? '' : 's')), ' on the same ' + v.n + ' test rows')), VK.table(head, rows, 'compare'));
    // the panels
    const cols = narrow ? 1 : Math.min(n, W >= 900 ? 4 : 2), gap = 26, pw = Math.min(narrow ? W - 20 : 300, Math.floor((W - gap * (cols - 1) - 10) / cols));
    v.items.forEach((it, i) => { it._h = panel(it, i, v, 0, 0, pw).h; });
    // rows of panels: lay them out again now that each panel's height is known
    const rowsH = [];
    v.items.forEach((it, i) => { const r = Math.floor(i / cols); rowsH[r] = Math.max(rowsH[r] || 0, it._h); });
    const out = [];
    let y0 = 0;
    for (let r = 0; r < rowsH.length; r++) {
      for (let i = r * cols; i < Math.min(n, (r + 1) * cols); i++) out.push(...panel(v.items[i], i, v, (i % cols) * (pw + gap) + 4, y0, pw).items);
      y0 += rowsH[r] + 24;
    }
    const H = y0;
    const fig = VK.fig('The models side by side');
    fig.show(out, Math.min(W, cols * (pw + gap) - gap + 10), H);
    box.appendChild(fig.el);
    box.appendChild(VK.note(v.items[0].boundary ? 'The shading is what each model says at every place in the plane (paler where it is less sure); the dots are the test rows in their true colours. Below each, its confusion matrix on those rows.' : v.task === 'regression' ? 'Each panel sets a model’s predictions against the truth; on the dashed line they would be exact.' : 'Each panel is a model’s confusion matrix on the same test rows.', 'muted'));
  }
  function panel(it, i, v, x0, y0, pw) {
    const items = [], P = 'p' + i + ':';
    const name = VK.short(it.model, Math.max(10, Math.floor(pw / 8)));
    items.push({ key: P + 'title', kind: 'text', x: x0, y: y0 + 16, text: name, size: 14, cls: 'cap' });
    let y = y0 + 28;
    if (it.boundary && v.points) {
      const b = it.boundary, side = pw - 46, ax = C().axes({ prefix: P + 'ax', x: x0 + 40, y: y + 8, w: side, h: side, xr: b.xr, yr: b.yr, xticks: undefined, yticks: undefined, grid: false });
      items.push(...ax.items);
      items.push({ key: P + 'field', kind: 'raster', x: ax.box.x, y: ax.box.y, w: side, h: side, nx: b.nx, ny: b.ny, vals: b.vals, vals2: b.conf, scheme: v.task === 'classification' ? 'classes' : 'seq', max: v.task === 'classification' ? 1 : Math.max(...b.vals.map(Math.abs)), alpha: 0.42, smooth: false, layer: 'back', z: 0 });
      v.points.X.forEach((p, k) => { if (ax.inside(p[0], p[1])) items.push({ key: P + 'pt' + k, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 3.4, cls: 'sp c' + Math.min(3, v.points.y[k]), audit: 'overlap-ok' }); });
      y = ax.box.y + side + 28;
    }
    if (it.counts) {
      const K = v.classes.length, cell = Math.max(30, Math.min(48, Math.floor((pw - 70) / K)));
      const cm = WF.VK.confusion({ prefix: P + 'cm', classes: v.classes, counts: it.counts, W: pw, cell });
      for (const x of cm.items) shiftItem(x, x0, y);
      items.push(...cm.items);
      y += cm.H;
    }
    return { items, h: y - y0 };
  }
  function shiftItem(it, dx, dy) {
    for (const k of ['x', 'x1', 'x2']) if (it[k] !== undefined) it[k] = Math.round((it[k] + dx) * 100) / 100;
    for (const k of ['y', 'y1', 'y2']) if (it[k] !== undefined) it[k] = Math.round((it[k] + dy) * 100) / 100;
  }
  WF.viewers.compare = VK.viewer(compare, 'compare');
})(globalThis.Nooral = globalThis.Nooral || {});
