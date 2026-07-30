// The insides of the networks that see and read, page side: a CNN's first filters beside the feature map each one
// finds in a probe image, the maps after pooling and after the second convolution, and its probabilities; a recurrent
// network's hidden state token by token (with an LSTM's forget and input gates); a transformer's attention, head by
// head, between the words of a probe text. Also the diagram checker's list of every model viewer (S.Checks).
(function (S) {
  'use strict';
  const { h } = S.UI;
  const WF = S.WF, M = S.WMV, P = M.pics;
  const r2 = (v) => Math.round(v * 100) / 100;
  const K = () => S.Kit;
  const tw = (t, fs) => S.geom.textWidth(String(t), fs);

  /** probabilities of the classes as bars, the true class named */
  const probs = (prefix, view, ps, x, y, w, truth) => {
    const names = view.classes.map((c, i) => String(c) + (i === truth ? ' (true)' : ''));
    const b = M.bars(prefix, names, ps, { x, y: y + 22, w, size: 12, rowH: 20, max: 1, fmt: (v) => M.num(v, 2), cls: (i) => (ps[i] === Math.max(...ps) ? 'act' : '') });
    b.items.push({ key: prefix + ':t', kind: 'text', x, y: y + 10, text: 'what it says for this example', size: 13, cls: 'axname' });
    return { items: b.items, h: b.h + 30 };
  };
  const ras = (key, x, y, s, vals, nx, ny, scheme, max) => ({ key, kind: 'raster', x: r2(x), y: r2(y), w: s, h: s, nx, ny, vals, scheme, max, smooth: false, audit: 'ignore' });

  // ---- the CNN ----------------------------------------------------------------------------------------------------
  P.cnn = function (view, ctx) {
    const I = view.internals;
    if (!I || !I.filters) return;
    const items = [], W = ctx.W, gap = 12;
    let y = 24;
    const title = (key, text) => { items.push({ key, kind: 'text', x: 0, y: y - 8, text, size: 13, cls: 'axname' }); };
    // the input and the answer
    const img = ctx.narrow ? 96 : 120;
    title('cn:t0', 'a probe image');
    items.push(ras('cn:img', 0, y, img, I.image, I.w, I.h, 'ink', 1));
    const pb = probs('cn:pr', view, I.probs, img + 40, y - 18, Math.min(360, W - img - 50), I.probeY);
    items.push(...pb.items);
    y += Math.max(img, pb.h - 18) + 44;
    // the first filters, each above the map it finds
    title('cn:t1', ctx.narrow ? 'filters, each above its map' : 'first-layer filters (' + I.k + ' × ' + I.k + '), each above the feature map it finds');
    const fcell = ctx.narrow ? 34 : 40, mcell = ctx.narrow ? 64 : 84, per = Math.max(1, Math.floor((W + gap) / (mcell + gap)));
    const fmax = Math.max(1e-9, ...I.filters.flat().map(Math.abs));
    I.filters.forEach((f, c) => {
      const x = (c % per) * (mcell + gap), yy = y + Math.floor(c / per) * (fcell + mcell + 28);
      items.push(ras('cn:f' + c, x + (mcell - fcell) / 2, yy, fcell, f, I.k, I.k, 'div', fmax));
      if (I.relu1[c]) items.push(ras('cn:m' + c, x, yy + fcell + 8, mcell, I.relu1[c], I.relu1Shape[2], I.relu1Shape[1], 'seq', Math.max(1e-9, ...I.relu1[c])));
    });
    y += Math.ceil(I.filters.length / per) * (fcell + mcell + 28) + 30;
    // after pooling, and after the second convolution
    const row = (key, text, maps, shape, cell) => {
      if (!maps || !maps.length) return;
      title(key + ':t', text);
      const n = Math.max(1, Math.floor((W + gap) / (cell + gap)));
      maps.forEach((m, c) => items.push(ras(key + c, (c % n) * (cell + gap), y + Math.floor(c / n) * (cell + gap), cell, m, shape[2], shape[1], 'seq', Math.max(1e-9, ...m))));
      y += Math.ceil(maps.length / n) * (cell + gap) + 36;
    };
    row('cn:p', ctx.narrow ? 'after 2 × 2 max pooling' : 'after 2 × 2 max pooling: half the size, the strongest answer of each patch kept', I.pool1, I.pool1Shape, ctx.narrow ? 48 : 60);
    row('cn:r', ctx.narrow ? 'the second convolution' : 'the second convolution: patterns of patterns (' + I.relu2.length + ' maps)', I.relu2, I.relu2Shape, ctx.narrow ? 40 : 50);
    M.still(ctx.el, items, { x: -4, y: 0, w: W, h: y }, 'Filters and feature maps of the CNN');
    ctx.el.appendChild(M.sub('Filters: blue weights look for ink, vermilion for paper. Maps: the brighter, the more strongly the filter’s pattern is there. The dense layers read the last maps.'));
  };
  M.architecture.cnn = function (view, ctx) {
    const p = view.params, stages = [['image', ''], ['conv ' + p.kernel + '×' + p.kernel, p.f1 + ' maps'], ['max pool', '2 × 2'], ['conv ' + p.kernel + '×' + p.kernel, p.f2 + ' maps'], ['max pool', '2 × 2'], ['dense', p.hidden + ' neurons'], ['classes', 'softmax']];
    const N = S.Kit.flow.net('ca'), bw = 92, bh = 50, gapX = ctx.narrow ? 14 : 22, per = Math.max(2, Math.floor((ctx.W - 10 + gapX) / (bw + gapX)));
    stages.forEach(([a, b], i) => N.box('s' + i, (i % per) * (bw + gapX) + bw / 2 + 4, Math.floor(i / per) * (bh + 40) + bh / 2 + 10, bw, bh, b ? [a, b] : [a]));
    for (let i = 0; i + 1 < stages.length; i++) if (Math.floor(i / per) === Math.floor((i + 1) / per)) N.wire('w' + i, [N.port('s' + i, 'e'), N.port('s' + (i + 1), 'w')], { head: true });
    const rows = Math.ceil(stages.length / per);
    M.still(ctx.el, N.items, { x: 0, y: 0, w: ctx.W, h: rows * (bh + 40) + 10 }, 'The layers of the CNN');
  };

  // ---- the recurrent network ------------------------------------------------------------------------------------
  /** the probe text's tokens that fit across the width, taken from the end (the ones nearest the decision) */
  const fitTokens = (toks, W, left, fs, minCell, maxCell) => {
    const cw = Math.max(minCell, Math.min(maxCell, Math.max(...toks.map((t) => tw(t, fs))) + 8));
    const n = Math.max(2, Math.min(toks.length, Math.floor((W - left - 10) / (cw + 2))));
    return { cw, from: toks.length - n, toks: toks.slice(toks.length - n) };
  };
  P.rnn = function (view, ctx) {
    const I = view.internals;
    if (!I || !I.h) return;
    const fs = 12, left = 56, F = fitTokens(I.toks, ctx.W, left, fs, 24, 64), items = [];
    const H = Math.min(I.H, 16), rows = I.h.slice(F.from).map((v) => v.slice(0, H));
    const T = rows.length, vals = Array.from({ length: H }, (_, k) => rows.map((v) => v[k]));
    let y = 40;
    items.push({ key: 'rn:t', kind: 'text', x: 0, y: 14, text: ctx.narrow ? 'the hidden state, step by step' : 'the hidden state after each ' + (I.level === 'char' ? 'character' : 'word') + ' of the probe text (one row per unit)', size: 13, cls: 'axname' });
    const g = K().heat.grid({ prefix: 'rn:h', x: left, y: y + 18, cw: F.cw, ch: 14, gap: 2, vals, scheme: 'div', max: 1, fmt: null, cols: F.toks.map((t) => (t === ' ' ? '·' : t)), labelSize: fs });
    items.push(...g.items);
    items.push({ key: 'rn:u', kind: 'text', x: left - 8, y: y + 30, text: 'h', anchor: 'end', size: 12, cls: 'legend' });
    y += 18 + g.h + 30;
    if (I.gates) {
      const names = I.cell === 'LSTM' ? [['f', 'forget'], ['i', 'input'], ['o', 'output']] : [['z', 'update'], ['r', 'reset']];
      items.push({ key: 'rn:gt', kind: 'text', x: 0, y: y - 4, text: ctx.narrow ? 'the gates (0 shut, 1 open)' : 'the gates, averaged over the units (0 shut, 1 open)', size: 13, cls: 'axname' });
      const gv = names.map(([k]) => I.gates[k].slice(F.from).map((v) => v.reduce((a, b) => a + b, 0) / v.length));
      const g2 = K().heat.grid({ prefix: 'rn:g', x: left, y: y + 8, cw: F.cw, ch: 22, gap: 2, vals: gv, scheme: 'seq', max: 1, fmt: (v) => M.num(v, 2).replace(/^0\./, '.'), rows: names.map((n) => n[1]), labelSize: fs, size: 11 });
      items.push(...g2.items);
      y += 8 + g2.h + 34;
    }
    const pb = probs('rn:pr', view, I.probs, left, y, Math.min(360, ctx.W - left - 10), I.probeY);
    items.push(...pb.items);
    y += pb.h + 10;
    ctx.el.appendChild(h('p.wfm-probe', h('b', 'Probe text: '), I.text));
    M.still(ctx.el, items, { x: 0, y: 0, w: ctx.W, h: y }, 'The hidden state of the recurrent network');
    ctx.el.appendChild(M.sub('Each column is the state after reading one more ' + (I.level === 'char' ? 'character' : 'word') + (F.from > 0 ? ' (the last ' + T + ' shown)' : '') + ': blue units are positive, vermilion negative. The last column is what decides.'));
  };

  // ---- the transformer --------------------------------------------------------------------------------------------------
  P.transformer = function (view, ctx) {
    const I = view.internals;
    if (!I || !I.heads) return;
    const fs = 12, labelW = Math.min(110, Math.max(...I.toks.map((t) => tw(t, fs))) + 12);
    const heads = I.heads.length, wide = !ctx.narrow && ctx.W >= 700;
    const perRow = wide ? Math.min(heads, 2) : 1, avail = (ctx.W - 10) / perRow - labelW - 20;
    const F = fitTokens(I.toks, avail + labelW, labelW, fs, 18, 46), n = F.toks.length, cell = Math.min(F.cw, Math.floor(avail / n) - 2);
    const items = [];
    let y = 0, maxY = 0;
    I.heads.forEach((A, hd) => {
      const sub = A.slice(A.length - n).map((r) => r.slice(r.length - n));
      const col = hd % perRow, x0 = col * ((ctx.W - 10) / perRow) + labelW;
      if (col === 0 && hd) y = maxY + 20;
      const top = y + 34 + 20;
      const g = K().heat.grid({ prefix: 'tf' + hd, x: x0, y: top, cw: cell, ch: Math.min(cell, 22), gap: 1, vals: sub, scheme: 'seq', max: Math.max(0.05, ...sub.flat()), fmt: null, rows: F.toks, cols: cell >= 26 ? F.toks.map((t) => (tw(t, 11) > cell ? t.slice(0, 3) : t)) : null, labelSize: cell >= 26 ? 11 : fs });
      items.push(...g.items, { key: 'tf' + hd + ':t', kind: 'text', x: x0 - labelW, y: y + 18, text: ctx.narrow ? 'head ' + (hd + 1) + ': row attends to column' : 'head ' + (hd + 1) + ': how much each word (row) attends to each word (column)', size: 13, cls: 'axname' });
      maxY = Math.max(maxY, top + g.h + 20);
    });
    let yy = maxY + 10;
    const pb = probs('tf:pr', view, I.probs, labelW, yy, Math.min(360, ctx.W - labelW - 10), I.probeY);
    items.push(...pb.items);
    yy += pb.h + 10;
    ctx.el.appendChild(h('p.wfm-probe', h('b', 'Probe text: '), I.text));
    M.still(ctx.el, items, { x: 0, y: 0, w: ctx.W, h: yy }, 'Attention of the transformer, head by head');
    ctx.el.appendChild(M.sub('Every row of a head sums to one: it is how the word on the left shares out its attention (the darkest cell of a head is its largest weight)' + (F.from > 0 ? ' (the last ' + n + ' words shown)' : '') + '. Different heads learn to look for different things.'));
  };

  // ---- the diagram checker's list --------------------------------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.group === 'canvas-models')) {
    const open = (sample, id, extra) => ({ group: 'canvas-models', name: sample + ' / ' + id + (extra ? ' ' + extra.name : ''), hash: 'canvas', fresh: true,
      setup: "await Nooral.app.page.loadDoc(Nooral.WF.newDoc()); await Nooral.app.page.whenIdle(); await Nooral.app.page.loadSample('" + sample + "'); await Nooral.app.page.whenIdle(); Nooral.app.page.open('" + id + "');" + (extra ? extra.js : ''),
      ready: "!!document.querySelector('.wf-v-host .wfm')", sel: '.wf-v-host' });
    for (const s of ['tree-moons', 'svm-circles', 'forest-moons', 'boost-moons', 'ada-xor', 'knn-blobs', 'nb-moons', 'qda-blobs', 'perceptron-line', 'mlp-spirals', 'cnn-digits', 'lstm-spam', 'transformer-spam', 'regression-rents', 'lasso-rents', 'svr-curve', 'boost-curve', 'forest-loans']) S.Checks.push(open(s, 'train1'));
    S.Checks.push(open('mlp-spirals', 'mlp1'), open('cnn-digits', 'cnn1'), open('tree-moons', 'tree1'));
  }
})(globalThis.Nooral = globalThis.Nooral || {});
