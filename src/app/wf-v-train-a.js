// Training a network on the canvas, page side: the controls (play, pause, one epoch, one batch, back to the start),
// where the run stands (epoch and batch, a progress bar), the loss and accuracy curves for the training and the
// validation rows, growing as the epochs end, and the multilayer perceptron drawn inside: every weight as a line whose
// colour is its sign and whose thickness is its size, a probe row's activations filling the neurons, what each hidden
// neuron says over the plane, and for images the picture each first-layer neuron looks for.
(function (S) {
  'use strict';
  const { h, icon } = S.UI;
  const WF = S.WF, M = S.WMV, P = M.pics;
  const r2 = (v) => Math.round(v * 100) / 100;
  const K = () => S.Kit;

  // ---- the controls and the curves -----------------------------------------------------------------------------
  M.trainPanel = function (view, ctx) {
    const api = ctx.api || {}, mem = ctx.mem;
    const st = () => { try { return api.state ? api.state() : null; } catch (e) { return null; } };
    const state = st(), running = (state && (state.state || state) === 'training') || !!view.live;
    const finished = view.epoch >= view.epochs && !view.live;
    const btn = (id, ic, label, action, primary) => { const b = h('button.btn.small' + (primary ? '.primary' : ''), { type: 'button', id, title: label }, icon(ic), h('span', label)); b.addEventListener('click', () => { if (api.control) api.control(action); mem.playing = action === 'play'; }); return b; };
    const s0 = state ? (typeof state === 'string' ? state : state.state) : null;
    const playing = !finished && (!!view.live || s0 === 'training' || (s0 !== 'paused' && !!mem.playing));
    let play;
    // play and pause trade places at once, before the run answers
    const swap = (b, other) => b.addEventListener('click', () => { const o = other(); b.replaceWith(o); });
    const mkPause = () => { const b = btn('wfmPause', 'pause', 'Pause', 'pause', true); swap(b, mkPlay); return b; };
    const mkPlay = () => { const b = btn('wfmPlay', 'play', 'Play', 'play', true); swap(b, mkPause); return b; };
    if (playing) play = mkPause();
    else if (!finished) play = mkPlay();
    else {
      play = h('button.btn.small.primary', { type: 'button', id: 'wfmPlay', title: 'Train again from the start' }, icon('play'), h('span', 'Train again'));
      play.addEventListener('click', () => { if (api.control) { api.control('reset'); setTimeout(() => api.control('play'), 80); } mem.playing = true; });
    }
    const bar = h('div.wfm-progress', h('i', { style: { width: Math.round(100 * Math.min(1, (view.epoch * (view.batches || 1) + (view.batch || 0)) / Math.max(1, view.epochs * (view.batches || 1)))) + '%' } }));
    const where = h('span.wfm-where', { id: 'wfmWhere', 'aria-live': 'off' }, 'epoch ' + view.epoch + ' of ' + view.epochs + (view.batches ? ' · batch ' + (view.batch || 0) + ' of ' + view.batches : ''));
    ctx.el.appendChild(h('div.wfm-train', play, btn('wfmEpoch', 'stepOver', 'One epoch', 'epoch'), btn('wfmBatch', 'stepInto', 'One batch', 'batch'), btn('wfmReset', 'restart', 'Reset', 'reset'), where, bar));
    if (view.nTrain && view.batches) ctx.el.appendChild(M.sub('Learning from ' + view.nTrain + ' training rows in ' + view.batches + ' batches an epoch' + (view.nVal ? '; the validation curves use ' + view.nVal + ' rows it never learns from' : '') + '.'));
    // the curves, drawn again as each epoch ends (points arrive between the views)
    const host = h('div.wfm-curves');
    ctx.el.appendChild(host);
    const pts = () => { const base = (view.curve || []).slice(), last = base.length ? base[base.length - 1].epoch : 0; return base.concat((mem.pts || []).filter((p) => p.epoch > last)); };
    let stage = null;
    const draw = () => { const items = curveItems(pts(), view, ctx); if (!stage) stage = M.still(host, items.items, items.view, 'The loss and accuracy as training goes on'); else stage.stage.show({ items: items.items, view: items.view }, { instant: true }); };
    draw();
    if (!mem.off && api.onCurve) {
      mem.off = api.onCurve((p) => {
        const w = document.getElementById('wfmWhere');
        if (w && p) w.textContent = 'epoch ' + p.epoch + ' of ' + view.epochs + (view.batches ? ' · batch ' + (p.batch || 0) + ' of ' + view.batches : '');
        if (p && p.epochEnd) { (mem.pts = mem.pts || []).push(p); if (mem.redrawCurves) mem.redrawCurves(); }
      });
    }
    mem.redrawCurves = draw;
  };
  /** loss on one plot, accuracy on another (side by side when there is room) */
  const curveItems = function (pts, view, ctx) {
    const clf = view.task === 'classification', two = clf, wide = !ctx.narrow && ctx.W >= 640;
    const pw = two && wide ? Math.floor((ctx.W - 20) / 2) : ctx.W - 8, ph = 170, items = [];
    const xr = [0, Math.max(1, view.epochs)];
    const plot = (prefix, x, y, key, vkey, name, yr) => {
      const box = { x: x + 50, y: y + 26, w: pw - 70, h: ph };
      const ax = M.axes(prefix, box, xr, yr, ['epoch', name]);
      items.push(...ax.items);
      const tr = pts.map((p) => [p.epoch, p[key]]).filter((p) => p[1] !== null && p[1] !== undefined), va = pts.map((p) => [p.epoch, p[vkey]]).filter((p) => p[1] !== null && p[1] !== undefined);
      if (tr.length) items.push(K().curve.line(ax, prefix + ':tr', tr, 'act'));
      if (va.length) items.push(K().curve.line(ax, prefix + ':va', va, 'c0 val'));
      if (tr.length) items.push(K().curve.dot(ax, prefix + ':trd', tr[tr.length - 1][0], tr[tr.length - 1][1], 'act', 3.5));
      if (va.length) items.push(K().curve.dot(ax, prefix + ':vad', va[va.length - 1][0], va[va.length - 1][1], 'c0', 3.5));
      return box;
    };
    const losses = pts.flatMap((p) => [p.loss, p.valLoss]).filter((v) => v !== null && v !== undefined && isFinite(v));
    const lmax = losses.length ? Math.max(...losses) * 1.08 : 1;
    const b1 = plot('cl', 0, 0, 'loss', 'valLoss', 'loss', [0, lmax || 1]);
    let bottom = b1.y + b1.h + 52;
    if (two) { const b2 = plot('ca', wide ? pw : 0, wide ? 0 : bottom, 'acc', 'valAcc', 'accuracy', [0, 1]); bottom = Math.max(bottom, b2.y + b2.h + 52); }
    items.push(...K().curve.legend('cleg', 50, bottom + 6, [{ text: 'training rows', cls: 'act' }, { text: 'validation rows', cls: 'c0 val' }]));
    if (!pts.length) items.push({ key: 'cl:none', kind: 'text', x: b1.x + 12, y: b1.y + 24, text: 'no epoch has ended yet', size: 13, cls: 'legend' });
    return { items, view: { x: 0, y: 0, w: ctx.W, h: bottom + 16 } };
  };

  // ---- the multilayer perceptron -----------------------------------------------------------------------------------
  const short = (t, n) => (String(t).length > n ? String(t).slice(0, n - 1) + '…' : String(t));
  /** the network drawn: neurons in layers, weights as lines. -> {items, view} */
  M.netItems = function (I, view, box, o) {
    o = o || {};
    const sizes = I.sizes.slice(), cap = o.cap || 10;
    const shown = sizes.map((n) => Math.min(n, cap));
    const L = K().net.layout({ x: box.x, y: box.y, w: box.w, h: box.h, sizes: shown, r: o.r || 15 });
    const pick = (W, l) => (W ? W.slice(0, shown[l + 1]).map((row) => row.slice(0, shown[l])) : null);
    const weights = (I.W || []).map((W, l) => pick(W, l));
    const wmax = Math.max(1e-9, ...weights.flat(2).filter((v) => typeof v === 'number').map(Math.abs));
    const acts = I.acts ? I.acts.map((a, l) => a.slice(0, shown[l])) : null;
    const amax = acts ? Math.max(1e-9, ...acts.slice(1, -1).flat().map(Math.abs)) : 1;
    const items = K().net.draw(L, { prefix: 'nn', weights, acts: acts ? acts.map((a, l) => (l === 0 ? null : a)) : null, wmax, amax, wide: 5, nodeCls: (l) => (l === 0 ? 'input' : l === sizes.length - 1 ? 'out' : '') });
    const fs = 12;
    if (o.names !== false) {
      const inNames = view.task && I.features && I.features.length === sizes[0] ? I.features : null;
      L.pos[0].forEach((p, i) => { if (inNames) items.push({ key: 'nn:in' + i, kind: 'text', x: r2(p.x - L.r - 8), y: r2(p.y + 4), text: short(inNames[i], 14), anchor: 'end', size: fs, cls: 'legend' }); });
      const outs = view.task === 'classification' ? view.classes : ['prediction'];
      L.pos[L.pos.length - 1].forEach((p, i) => { if (outs && outs[i] !== undefined) items.push({ key: 'nn:out' + i, kind: 'text', x: r2(p.x + L.r + 8), y: r2(p.y + 4), text: short(outs[i], 14), size: fs, cls: 'legend' }); });
    }
    sizes.forEach((n, l) => { if (n > shown[l]) items.push({ key: 'nn:more' + l, kind: 'text', x: L.pos[l][0].x, y: r2(box.y + box.h + 22), text: '+' + (n - shown[l]) + ' more', anchor: 'middle', size: 12, cls: 'legend' }); });
    return { items, L };
  };
  P.mlp = function (view, ctx) {
    const I = view.internals;
    if (!I || !I.sizes) return P.generic(view, ctx);
    const wide = !ctx.narrow && ctx.W >= 760;
    // the network, and beside it what the network says over the plane
    const nameW = I.features && I.features.length === I.sizes[0] && !I.inputWeights ? Math.min(110, Math.max(...I.features.map((f) => S.geom.textWidth(short(f, 14), 12))) + 12) : 20;
    const outW = 100;
    const netW = wide && view.plane ? Math.min(460, ctx.W * 0.5) : ctx.W - 8;
    const bh = Math.min(300, Math.max(160, Math.min(10, Math.max(...I.sizes)) * 30));
    const box = { x: nameW + 22, y: 40, w: Math.max(120, netW - nameW - outW - 30), h: bh };
    const net = M.netItems(I, view, box, { names: !I.inputWeights });
    const items = net.items.slice();
    items.push({ key: 'nn:t', kind: 'text', x: 8, y: 16, text: I.inputWeights ? 'the layers (the input is every pixel)' : 'the network, with row ' + (I.probe + 1) + ' going through it', size: 13, cls: 'axname' });
    let H = box.y + box.h + 40;
    if (view.plane) {
      const pb = wide ? { x: netW + 50, y: 58, w: Math.min(340, ctx.W - netW - 80), h: 0 } : { x: 50, y: H + 50, w: Math.min(360, ctx.W - 80), h: 0 };
      pb.h = Math.round(pb.w * 0.8);
      const pic = M.plane('np', view.plane, pb, { r: 2.8 });
      items.push(...pic.items, { key: 'np:t', kind: 'text', x: pb.x, y: pb.y - 34, text: 'what it says over the plane', size: 13, cls: 'axname' });
      H = Math.max(H, pb.y + pb.h + 46);
    }
    ctx.el.appendChild(M.legend(view) || h('span'));
    M.still(ctx.el, items, { x: 0, y: 0, w: ctx.W, h: H }, 'The network and its weights');
    ctx.el.appendChild(M.sub('Blue weights are positive and vermilion negative; a thicker line is a larger weight. A neuron is filled by how strongly it fires for the row going through.'));
    // every hidden neuron's own map: the pieces the output is built from
    if (I.maps) {
      const cell = ctx.narrow ? 64 : 76, gap = 12, items2 = [];
      let y = 30;
      I.maps.layers.forEach((layer, l) => {
        items2.push({ key: 'hm:l' + l, kind: 'text', x: 0, y: y - 10, text: 'layer ' + (l + 1), size: 13, cls: 'axname' });
        const per = Math.max(1, Math.floor((ctx.W - 10) / (cell + gap)));
        layer.forEach((vals, j) => {
          const x = (j % per) * (cell + gap), yy = y + Math.floor(j / per) * (cell + gap), mx = Math.max(1e-9, ...vals.map(Math.abs));
          items2.push({ key: 'hm:r' + l + '_' + j, kind: 'raster', x, y: yy, w: cell, h: cell, nx: I.maps.n + 1, ny: I.maps.n + 1, vals, scheme: 'div', max: mx, smooth: true, audit: 'ignore' });
        });
        y += Math.ceil(layer.length / per) * (cell + gap) + 30;
      });
      ctx.el.appendChild(M.sub('What each hidden neuron says at every place of the plane (blue where it fires, vermilion where it is negative). The next layer adds these up.'));
      M.still(ctx.el, items2, { x: -4, y: 0, w: ctx.W, h: y }, 'Every hidden neuron over the plane');
    }
    if (I.inputWeights) {
      const cell = ctx.narrow ? 54 : 64, gap = 10, per = Math.max(1, Math.floor((ctx.W - 10) / (cell + gap))), items3 = [];
      I.inputWeights.forEach((w, j) => { const mx = Math.max(1e-9, ...w.map(Math.abs)); items3.push({ key: 'iw:' + j, kind: 'raster', x: (j % per) * (cell + gap), y: 10 + Math.floor(j / per) * (cell + gap), w: cell, h: cell, nx: I.w, ny: I.h, vals: w, scheme: 'div', max: mx, smooth: false, audit: 'ignore' }); });
      ctx.el.appendChild(M.sub('What the first ' + I.inputWeights.length + ' neurons of the first layer look for: each pixel’s weight, blue where ink raises the neuron and vermilion where it lowers it.'));
      M.still(ctx.el, items3, { x: -4, y: 0, w: ctx.W, h: 20 + Math.ceil(I.inputWeights.length / per) * (cell + gap) }, 'First-layer weights as pictures');
    }
  };
  // before training: the layers, drawn without weights
  M.architecture.mlp = function (view, ctx) {
    const hidden = [].concat(view.params.hidden || []);
    const sizes = ['in'].concat(hidden, ['out']);
    const box = { x: 30, y: 30, w: Math.min(ctx.W - 60, 120 * (sizes.length - 1)), h: 230 };
    const L = K().net.layout({ x: box.x, y: box.y, w: box.w, h: box.h, sizes: sizes.map((s) => (typeof s === 'number' ? Math.min(s, 8) : 3)), r: 12 });
    const items = K().net.draw(L, { prefix: 'ar', weights: [], wmax: 1 });
    sizes.forEach((s, l) => items.push({ key: 'ar:n' + l, kind: 'text', x: L.pos[l][0].x, y: box.y + box.h + 26, text: l === 0 ? 'inputs' : l === sizes.length - 1 ? 'outputs' : s + ' neurons', anchor: 'middle', size: 12, cls: 'legend' }));
    M.still(ctx.el, items, { x: 0, y: 0, w: Math.max(ctx.W, box.x + box.w + 40), h: box.y + box.h + 44 }, 'The layers of the network');
  };
})(globalThis.Nooral = globalThis.Nooral || {});
