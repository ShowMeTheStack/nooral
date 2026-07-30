// What each block of the workflow canvas shows when selected, drawn with the stage kits: the data as a scatter,
// the split as filled and hollow points, the model's boundary, loss and weights, and the confusion matrix with
// clickable cells beside the test points (the examples of a picked cell ringed, and listed under the picture).
(function (S) {
  'use strict';
  const V = (S.FlowViews = {});
  const C = () => S.Kit.curve, SC = () => S.Kit.scatter, HM = () => S.Kit.heat;
  const NAMES = ['blue', 'orange'];
  const range = (X) => {
    const xs = X.map((p) => p[0]), ys = X.map((p) => p[1]), pad = 0.25;
    return { xr: [Math.floor((Math.min(...xs) - pad) * 2) / 2, Math.ceil((Math.max(...xs) + pad) * 2) / 2], yr: [Math.floor((Math.min(...ys) - pad) * 2) / 2, Math.ceil((Math.max(...ys) + pad) * 2) / 2] };
  };
  /** axes sized to the data, keeping the units square, inside a box */
  const fitAxes = (prefix, X, bx) => {
    const r = range(X), du = r.xr[1] - r.xr[0], dv = r.yr[1] - r.yr[0], k = Math.min(bx.w / du, bx.h / dv);
    const w = du * k, hh = dv * k;
    return C().axes({ prefix, x: bx.x, y: bx.y, w, h: hh, xr: r.xr, yr: r.yr, xticks: 1, yticks: 1, xname: 'x', yname: 'y' });
  };
  const legend = (key, x, y, entries) => C().legend(key, x, y, entries, { dir: 'col', size: 13 });

  V.data = function (out, o) {
    const d = out.data, ax = fitAxes('vd', d.X, o.narrow ? { x: 34, y: 24, w: 320, h: 260 } : { x: 50, y: 24, w: 560, h: 330 });
    const items = [...ax.items, ...SC().points(ax, 'vd:p', d.X, d.y, { r: 3.8 })];
    const n1 = d.y.filter((v) => v).length;
    items.push(...legend('vd:lg', o.narrow ? 34 : ax.box.x + ax.box.w + 40, o.narrow ? ax.box.y + ax.box.h + 64 : 60, [{ text: `blue: ${d.y.length - n1} rows`, cls: 'c0', dot: true }, { text: `orange: ${n1} rows`, cls: 'c1', dot: true }]));
    return { items, view: o.narrow ? { x: -4, y: -14, w: 392, h: ax.box.h + 150 } : { x: -10, y: -14, w: 1000, h: ax.box.h + 80 } };
  };
  V.split = function (out, o) {
    const all = out.all, ax = fitAxes('vs', all.X, o.narrow ? { x: 34, y: 24, w: 320, h: 260 } : { x: 50, y: 24, w: 560, h: 330 });
    const test = new Set(out.test.idx);
    const items = [...ax.items, ...SC().points(ax, 'vs:p', all.X, all.y, { r: 3.8, cls: (i) => (test.has(i) ? 'test' : '') })];
    items.push(...legend('vs:lg', o.narrow ? 34 : ax.box.x + ax.box.w + 40, o.narrow ? ax.box.y + ax.box.h + 64 : 60, [{ text: `train: ${out.train.X.length} rows (filled)`, cls: 'c0', dot: true }, { text: `test: ${out.test.X.length} rows (hollow)`, cls: 'c0 hollow', dot: true }]));
    return { items, view: o.narrow ? { x: -4, y: -14, w: 392, h: ax.box.h + 150 } : { x: -10, y: -14, w: 1000, h: ax.box.h + 80 } };
  };
  V.logreg = function (out, o) {
    const m = out.model, tr = m.train;
    const ax = fitAxes('vl', tr.X, o.narrow ? { x: 34, y: 24, w: 320, h: 240 } : { x: 50, y: 24, w: 480, h: 330 });
    const items = [...ax.items, SC().field(ax, 'vl:f', (u, v) => m.prob([u, v]), { n: 80 }), SC().level(ax, 'vl:b', SC().grid(ax, (u, v) => m.prob([u, v]), 70), 0.5, ''), ...SC().points(ax, 'vl:p', tr.X, tr.y, { r: 3.6 })];
    const L = o.narrow ? { lc: { x: 50, y: ax.box.y + ax.box.h + 90, w: 300, h: 120 } } : { lc: { x: 640, y: 40, w: 330, h: 140 } };
    const ep = m.curve.length - 1, lmax = Math.max(0.75, ...m.curve.map((c) => c[1]));
    const lax = C().axes(Object.assign({ prefix: 'vc', xr: [0, ep], yr: [0, Math.ceil(lmax * 4) / 4], xticks: C().ticks(0, ep, 3), yticks: 0.25, xname: 'epoch', yname: 'training loss' }, L.lc));
    items.push(...lax.items, C().line(lax, 'vc:l', m.curve.map((c) => [c[0], c[1]]), 'act'));
    const wb = o.narrow ? { x: 20, y: L.lc.y + L.lc.h + 92, w: 350 } : { x: 620, y: 270, w: 360 };
    items.push({ key: 'vw:t', kind: 'text', x: wb.x, y: wb.y - 14, text: 'the weights (on standardised features)', size: 13, cls: 'strong' });
    items.push(...S.Demo.bars('vw', wb.x, wb.y, wb.w, 80, m.w, m.names, { size: 12 }));
    const bottom = o.narrow ? wb.y + 120 : Math.max(ax.box.y + ax.box.h + 50, wb.y + 116);
    return { items, view: o.narrow ? { x: -4, y: -14, w: 392, h: bottom + 14 } : { x: -10, y: -14, w: 1000, h: bottom + 14 } };
  };
  V.confusion = function (out, o) {
    const items = [], cm = out.matrix, picked = o.cell;
    const cw = o.narrow ? 84 : 96;
    const g = HM().grid({ prefix: 'cm', x: o.narrow ? 120 : 170, y: 56, cw, ch: o.narrow ? 64 : 72, gap: 4, vals: cm.map((r) => r.map((c) => c.length)), scheme: 'seq', max: Math.max(...cm.flat().map((c) => c.length)) * 1.15, fmt: (v) => String(v), size: 22, rows: ['true blue', 'true orange'], cols: ['said blue', 'said orange'], labelSize: 13, pick: true, cls: (i, j) => (picked && picked[0] === i && picked[1] === j ? 'sel' : i === j ? '' : '') });
    items.push(...g.items);
    items.push({ key: 'cm:t', kind: 'text', x: o.narrow ? 6 : 40, y: 18, text: o.narrow ? 'test rows: said against true' : 'the test rows: what the model said, against the truth (click a cell)', size: 13, cls: 'strong' });
    const my = 56 + 2 * ((o.narrow ? 64 : 72) + 4) + 30;
    const pct = (v) => Math.round(v * 100) + '%';
    [['accuracy', pct(out.acc), 'right of all'], ['precision', pct(out.prec), 'of “orange” said, truly orange'], ['recall', pct(out.rec), 'of truly orange, found'], ['F1', out.f1.toFixed(2), 'balance of the two']].forEach(([a, b, c], k) => {
      items.push({ key: 'cm:m' + k, kind: 'text', x: o.narrow ? 6 : 40, y: my + k * 22, text: `${a} ${b}`, size: 13, cls: 'strong' });
      items.push({ key: 'cm:n' + k, kind: 'text', x: o.narrow ? 130 : 170, y: my + k * 22, text: c, size: 13, cls: 'axname' });
    });
    items.push({ key: 'cm:th', kind: 'text', x: o.narrow ? 6 : 40, y: my + 4 * 22 + 6, text: `threshold: orange when p(orange) ≥ ${out.threshold}`, size: 13, cls: 'axname' });
    // the test points over the model's boundary at the threshold
    const d = out.data, bx = o.narrow ? { x: 34, y: my + 150, w: 320, h: 240 } : { x: 520, y: 30, w: 440, h: 300 };
    const ax = fitAxes('cs', d.X, bx);
    const sel = new Set(picked ? cm[picked[0]][picked[1]] : []);
    items.push(...ax.items, SC().field(ax, 'cs:f', (u, v) => out.model.prob([u, v]), { n: 70 }), SC().level(ax, 'cs:b', SC().grid(ax, (u, v) => out.model.prob([u, v]), 60), out.threshold, ''));
    const wrong = new Set(out.rows.filter((r) => r.t !== r.pred).map((r) => r.k));
    items.push(...SC().points(ax, 'cs:p', d.X, d.y, { r: 4, cls: (i) => (wrong.has(i) ? 'wrong' : '') + (picked && !sel.has(i) ? ' dim' : '') }));
    for (const i of sel) items.push(SC().ring(ax, 'cs:r' + i, d.X[i][0], d.X[i][1], 'act', 8.5));
    const bottom = o.narrow ? ax.box.y + ax.box.h + 46 : Math.max(my + 4 * 22 + 20, ax.box.y + ax.box.h + 46);
    return { items, view: o.narrow ? { x: -4, y: -10, w: 392, h: bottom + 10 } : { x: -10, y: -10, w: 1000, h: bottom + 10 } };
  };
  /** the examples of a picked cell, as a table under the picture */
  V.examples = function (out, cell) {
    const { h } = S.UI;
    if (!cell) return h('p.cv-hint', 'Click a cell of the matrix to list its examples and ring them on the plot.');
    const ks = out.matrix[cell[0]][cell[1]], rows = out.rows.filter((r) => ks.includes(r.k));
    const title = `${rows.length} test row${rows.length === 1 ? '' : 's'} truly ${NAMES[cell[0]]}, said ${NAMES[cell[1]]}` + (cell[0] === cell[1] ? ' (right)' : ' (wrong)');
    const tbl = h('table.cv-table', h('thead', h('tr', h('th', 'row'), h('th', 'x'), h('th', 'y'), h('th', 'truth'), h('th', 'p(orange)'), h('th', 'said'))),
      h('tbody', ...rows.slice(0, 12).map((r) => h('tr', h('td', String(r.k + 1)), h('td', r.x[0].toFixed(2)), h('td', r.x[1].toFixed(2)), h('td', h('span.cv-c.c' + r.t, NAMES[r.t])), h('td', r.p.toFixed(3)), h('td', h('span.cv-c.c' + r.pred, NAMES[r.pred]))))));
    return h('div.cv-ex', h('h3', title), rows.length ? tbl : h('p.cv-hint', 'This cell is empty.'), rows.length > 12 ? h('p.cv-hint', 'and ' + (rows.length - 12) + ' more') : null);
  };
})(globalThis.Nooral = globalThis.Nooral || {});
