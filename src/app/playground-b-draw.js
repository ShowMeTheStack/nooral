// The playground's pictures: the network drawn left to right, every neuron a small map of what it says over the
// plane (blue where it is positive, vermilion where negative), every weight a line whose colour is its sign and whose
// thickness is its size; the output over the plane with the training and test points and the boundary p = 0.5; and
// the training and test loss against epochs.
(function (S) {
  'use strict';
  const PG = (S.PG = S.PG || {});
  const C = () => S.Kit.curve, SC = () => S.Kit.scatter, NK = () => S.Kit.net;
  const r2 = (v) => Math.round(v * 100) / 100;
  const D3 = 3;
  const RES = 22;
  /** a neuron's values over the plane, on an RES × RES grid (rows from the top) */
  const mapOf = (lab, wk, l, i) => {
    const vals = new Array(RES * RES);
    for (let r = 0; r < RES; r++) for (let c = 0; c < RES; c++) {
      const u = -D3 + ((c + 0.5) / RES) * 2 * D3, v = D3 - ((r + 0.5) / RES) * 2 * D3;
      vals[r * RES + c] = lab.forwardWith(wk, u, v)[l][i];
    }
    return vals;
  };
  const featMap = (f) => { const vals = new Array(RES * RES); for (let r = 0; r < RES; r++) for (let c = 0; c < RES; c++) vals[r * RES + c] = f.f(-D3 + ((c + 0.5) / RES) * 2 * D3, D3 - ((r + 0.5) / RES) * 2 * D3); return vals; };
  const maxAbs = (vals) => Math.max(1e-6, ...vals.map(Math.abs));
  PG.fmtW = (w) => (Math.abs(w) >= 10 ? w.toFixed(1) : w.toFixed(3)).replace(/^-/, '−');

  /** the network: {items, view, pos} (pos: the centre of every neuron, for the page's hit areas) */
  PG.netFrame = function (lab, wk, o) {
    o = o || {};
    const items = [], all = PG.FEATURES, on = new Set(lab.cfg.features);
    const sizes = [all.length].concat(lab.sizes.slice(1));
    const cols = sizes.length, maxN = Math.max(...sizes);
    // the picture keeps about the same size however many layers and neurons, so its text keeps its size
    const BOX = maxN > 7 ? 42 : 48, GAPY = maxN > 7 ? 12 : 20, COLW = Math.max(o.narrow ? 84 : 104, Math.min(168, ((o.narrow ? 520 : 660) - BOX) / (cols - 1)));
    const H = maxN * BOX + (maxN - 1) * GAPY;
    const colX = (l) => l * COLW;
    const pos = sizes.map((n, l) => Array.from({ length: n }, (_, i) => ({ x: colX(l), y: r2((H - (n * BOX + (n - 1) * GAPY)) / 2 + i * (BOX + GAPY)) })));
    // input l = 0 is every possible feature; the network's own input index is the order of the ones switched on
    const inIdx = []; all.forEach((f, j) => { if (on.has(f.id)) inIdx.push(j); });
    const wmax = Math.max(1, ...wk.W.flat(2).map(Math.abs));
    // weights
    wk.W.forEach((W, l) => W.forEach((row, i) => row.forEach((w, j) => {
      const a = l === 0 ? pos[0][inIdx[j]] : pos[l][j], b = pos[l + 1][i];
      const p = NK().weightPaint(w, wmax, 7);
      const from = l === 0 ? lab.feats[j].label : 'h' + l + '.' + (j + 1), to = l + 1 === wk.W.length ? 'the output' : 'h' + (l + 1) + '.' + (i + 1);
      const sel = o.weight === l + '_' + i + '_' + j;
      items.push({ key: 'pg:w' + l + '_' + i + '_' + j, kind: 'path', d: `M${r2(a.x + BOX + 2)},${r2(a.y + BOX / 2)} L${r2(b.x - 2)},${r2(b.y + BOX / 2)}`, cls: 'weight pg-w' + (sel ? ' sel' : ''), paint: sel ? { stroke: 'var(--accent)', sw: Math.max(3, p.sw) } : p, pathAudit: 'edge-x', layer: 'lines', z: Math.round(Math.abs(w) * 100), title: 'weight ' + PG.fmtW(w) + ' from ' + from + ' to ' + to });
    })));
    // inputs: every feature, the ones switched off dimmed (a click switches)
    all.forEach((f, j) => {
      const p = pos[0][j], vals = featMap(f), isOn = on.has(f.id);
      items.push({ key: 'pg:f:' + f.id, kind: 'raster', x: p.x, y: p.y, w: BOX, h: BOX, nx: RES, ny: RES, vals, scheme: 'div', max: maxAbs(vals), smooth: true, cls: 'feature pg-node pg-in' + (isOn ? '' : ' off'), layer: 'items', z: 1, audit: 'node', title: f.label + (isOn ? ': on (click to switch off)' : ': off (click to switch on)') });
      items.push({ key: 'pg:fl:' + f.id, kind: 'text', x: r2(p.x - 10), y: r2(p.y + BOX / 2 + (o.narrow ? 7 : 5)), text: f.label, anchor: 'end', size: o.narrow ? 20 : 14, cls: 'pg-flabel' + (isOn ? '' : ' off') });
    });
    // hidden neurons and the output
    for (let l = 1; l < cols; l++) {
      const last = l === cols - 1;
      pos[l].forEach((p, i) => {
        const vals = mapOf(lab, wk, l, i), key = last ? 'pg:out' : 'pg:n' + l + '_' + i;
        items.push({ key, kind: 'raster', x: p.x, y: p.y, w: BOX, h: BOX, nx: RES, ny: RES, vals, scheme: last ? 'prob' : 'div', max: last ? 1 : lab.cfg.act === 'sigmoid' ? 1 : maxAbs(vals), smooth: true, cls: 'feature pg-node' + (o.neuron === key ? ' act' : ''), layer: 'items', z: 1, audit: 'node', title: last ? 'the output: p(orange) over the plane' : 'neuron ' + l + '.' + (i + 1) + ': bias ' + PG.fmtW(wk.b[l - 1][i]) });
      });
    }
    // column names
    sizes.forEach((n, l) => {
      const name = l === 0 ? 'inputs' : l === cols - 1 ? 'output' : 'layer ' + l;
      const sub = l === 0 ? lab.cfg.features.length + ' on' : l === cols - 1 ? 'p(orange)' : n + ' neuron' + (n === 1 ? '' : 's');
      items.push({ key: 'pg:ct' + l, kind: 'text', x: r2(colX(l) + BOX / 2), y: o.narrow ? -14 : -34, text: name, anchor: 'middle', size: o.narrow ? 19 : 14, cls: 'axname' });
      if (!o.narrow) items.push({ key: 'pg:cs' + l, kind: 'text', x: r2(colX(l) + BOX / 2), y: -14, text: sub, anchor: 'middle', size: 13, cls: 'tick' });
    });
    return { items, view: { x: -72, y: o.narrow ? -36 : -58, w: (cols - 1) * COLW + BOX + 96, h: H + (o.narrow ? 54 : 78) }, pos };
  };

  /** the output (or one neuron) over the plane, with the points */
  PG.plotFrame = function (lab, wk, o) {
    o = o || {};
    const W = o.w || 330, items = [];
    const ax = C().axes({ prefix: 'po', x: 0, y: 0, w: W, h: W, xr: [-D3, D3], yr: [-D3, D3], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3], xname: 'x₁', yname: 'x₂', frame: 'box' });
    items.push(...ax.items);
    let f, scheme = 'prob', max = 1, what;
    if (o.neuron && o.neuron !== 'pg:out') {
      const [l, i] = o.neuron.slice(4).split('_').map(Number);
      f = (u, v) => lab.forwardWith(wk, u, v)[l][i];
      scheme = 'div'; max = lab.cfg.act === 'sigmoid' ? 1 : maxAbs(mapOf(lab, wk, l, i));
      what = 'neuron ' + l + '.' + (i + 1);
    } else { f = (u, v) => lab.forwardWith(wk, u, v).pop()[0]; what = 'output'; }
    items.push(SC().field(ax, 'po:f', f, { n: o.n || 64, scheme, max }));
    items.push(SC().level(ax, 'po:b', SC().grid(ax, f, 56), what === 'output' ? 0.5 : 0, what === 'output' ? '' : 'soft', { minPts: 2 }));
    const d = lab.data;
    if (o.showTest !== false) items.push(...SC().points(ax, 'po:te', d.X, d.y, { only: d.test, r: 3.6, cls: () => 'test' }));
    items.push(...SC().points(ax, 'po:tr', d.X, d.y, { only: d.train, r: 3.8 }));
    return { items, view: { x: -40, y: -32, w: W + 56, h: W + 76 }, what };
  };

  /** train and test loss against epochs */
  PG.curveFrame = function (lab, o) {
    o = o || {};
    const W = o.w || 330, Hh = o.h || 130, hs = lab.history, n = Math.max(10, hs[hs.length - 1].epoch);
    const top = Math.max(0.1, ...hs.map((x) => Math.max(x.train, x.test)));
    const yt = C().ticks(0, top, 3);
    const ax = C().axes({ prefix: 'pc', x: 0, y: 0, w: W, h: Hh, xr: [0, n], yr: [0, Math.max(top, yt[yt.length - 1])], xticks: C().ticks(0, n, 4), yticks: yt, xname: 'epoch', yname: 'log loss', tickSize: 13 });
    const items = ax.items.slice();
    items.push(C().line(ax, 'pc:tr', hs.map((x) => [x.epoch, x.train]), 'c1'));
    items.push(C().line(ax, 'pc:te', hs.map((x) => [x.epoch, x.test]), 'faint'));
    items.push(...C().legend('pc:lg', W - 150, -14, [{ text: 'train', cls: 'c1' }, { text: 'test', cls: 'faint' }], { size: 13 }));
    return { items, view: { x: -44, y: -36, w: W + 60, h: Hh + 82 } };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
