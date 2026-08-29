// Part 9, neural networks: the families, and the pieces the topics share: small networks trained here on toy sets
// (full batch, seeded, with snapshots of the losses and of what the network says over the plane), a decision
// boundary drawn from one batched forward pass, network diagrams whose neurons are their own region maps, and
// formatting for vectors in the narration.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'networks', order, title, blurb });
  fam('p09-neuron', 10, 'Neurons and activations', 'One neuron: a weighted sum of its inputs, a bias and a squashing function; then the activation functions side by side with their slopes, which decide how well a signal and its gradient pass through.');
  fam('p09-mlp', 20, 'The multilayer perceptron', 'Neurons in layers, every weight a wire: the forward pass with the value on every wire, backpropagation on the computational graph with every gradient shown and checked, and the loss surface of a network small enough to draw.');
  fam('p09-init', 30, 'Initialisation', 'Why the first weights matter: too small and the signal dies, too large and it saturates or explodes; Xavier and He scale them so each layer passes on as much as it receives.');
  fam('p09-reg', 40, 'Overfitting and regularisation', 'A big network memorising a small training set, and the cures: dropout, weight decay, early stopping; and batch and layer normalisation, which keep the numbers inside a network in a healthy range.');
  fam('p09-play', 50, 'The playground', 'Build a small network on a 2-D set and watch every hidden neuron’s region as it trains.');

  const P = (S.P09 = {}), H = () => S.Kit.ml;
  const memo = new Map();
  /** a value computed once per page (or per test run) and kept */
  P.once = (key, f) => { if (!memo.has(key)) memo.set(key, f()); return memo.get(key); };
  P.r2 = (v) => Math.round(v * 100) / 100;
  /** a vector as text: [0.25, −0.10] */
  P.vec = (v, d) => '[' + Array.from(v).map((x) => H().fx(x, d === undefined ? 2 : d)).join(', ') + ']';
  /** the same, every number quoted (so the claims check reads them back) */
  P.qvec = (q, v, d) => '[' + Array.from(v).map((x) => q(x, d === undefined ? 2 : d)).join(', ') + ']';
  P.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');
  P.sub = (k) => String(k).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
  P.sigmoid = (z) => 1 / (1 + Math.exp(-z));
  /** a closer camera for each frame on a phone: box(meta, frame) -> {x, y, w, h} or null */
  P.narrow = (frames, box) => {
    for (const f of frames) {
      const b = box(f.meta || {}, f);
      if (b) f.narrowView = b;
      // formulas are set in the monospaced face: measure them so (the camera and the checks count their true width)
      for (const it of f.items) if (it && it.kind === 'text' && /(^|\s)math(\s|$)/.test(it.cls || '')) it.mono = true;
    }
    return frames;
  };
  /** plot axes with ticks and names large enough for the scaled-down stage */
  P.ax = (o) => S.Kit.curve.axes(Object.assign({ tickSize: 14, nameSize: 14 }, o));
  /** a marked point that may sit on another (two series meeting) */
  P.dot = (ax, key, u, v, cls, r) => Object.assign(S.Kit.curve.dot(ax, key, u, v, cls, r), { audit: 'overlap-ok', point: false });

  // ---- toy sets -----------------------------------------------------------------------------------------------------
  P.SETS = {
    moons: { label: 'two moons', make: (n, noise, seed) => S.Data.moons({ n, noise, seed }), xr: [-1.7, 2.7], yr: [-1.2, 1.7], noise: 0.2 },
    circles: { label: 'a ring round a disc', make: (n, noise, seed) => S.Data.circles({ n, noise: noise * 0.5, seed }), xr: [-1.5, 1.5], yr: [-1.5, 1.5], noise: 0.12 },
    xor: { label: 'XOR corners', make: (n, noise, seed) => S.Data.xor({ n, noise: noise * 0.25, seed }), xr: [-1.15, 1.15], yr: [-1.15, 1.15], noise: 0 },
    spirals: { label: 'two spirals', make: (n, noise, seed) => S.Data.spirals({ n, noise: noise * 0.25, seed }), xr: [-1.2, 1.2], yr: [-1.2, 1.2], noise: 0.08 },
  };
  P.setOptions = (ids) => ids.map((id) => ({ value: id, label: P.SETS[id].label }));
  /** {X, y} rows of a toy set (copied, so pictures never change the engine's data) */
  P.data = (set, n, noise, seed) => P.once(['d', set, n, noise, seed].join(':'), () => { const d = P.SETS[set].make(n, noise, seed); return { X: d.X.map((r) => r.slice()), y: d.y.slice() }; });

  // ---- training ------------------------------------------------------------------------------------------------------
  /**
   * Train an MLP on a 2-D set, full batch, seeded. o: {train {X, y}, val {X, y}, hidden, act, dropout, lr,
   * optimizer, weightDecay, steps, evalEvery, snaps: [steps], seed, grid: {xr, yr, n}}
   * -> {net, evals: [{step, trainLoss, valLoss, trainAcc, valAcc}], snaps: [{step, ..., state, field}], best}
   */
  P.train = function (o) {
    const T = S.T, net = S.DL.build('mlp', { inputs: 2, hidden: o.hidden, outputs: 2, act: o.act || 'relu', dropout: o.dropout || 0, seed: o.seed || 1 });
    const opt = T.optim.make(o.optimizer || 'adam', net.parameters(), Object.assign({ lr: o.lr || 0.01, weightDecay: o.weightDecay || 0 }, o.optOpts || {}));
    const xt = T.tensor(o.train.X), yt = o.train.y, xv = o.val ? T.tensor(o.val.X) : null, yv = o.val ? o.val.y : null;
    const evals = [], snaps = [], snapAt = new Set(o.snaps || []), every = o.evalEvery || 10;
    const measure = (step) => {
      net.eval();
      const tr = T.noGrad(() => net.lossOn({ x: xt, y: yt })), va = xv ? T.noGrad(() => net.lossOn({ x: xv, y: yv })) : null;
      net.train();
      return { step, trainLoss: tr.loss.item(), trainAcc: tr.correct / tr.count, valLoss: va ? va.loss.item() : null, valAcc: va ? va.correct / va.count : null };
    };
    const snap = (step, e) => {
      const s = Object.assign({}, e || measure(step), { state: net.stateDict() });
      if (o.grid) s.field = P.fieldOf(net, o.grid);
      if (o.snapExtra) s.extra = o.snapExtra(net);
      snaps.push(s);
    };
    evals.push(measure(0));
    if (o.onEval) o.onEval(evals[0], net);
    if (snapAt.has(0)) snap(0, evals[0]);
    for (let t = 1; t <= o.steps; t++) {
      net.train();
      opt.zeroGrad();
      const r = net.lossOn({ x: xt, y: yt });
      r.loss.backward();
      opt.step();
      if (t % every === 0 || t === o.steps) { evals.push(measure(t)); if (o.onEval) o.onEval(evals[evals.length - 1], net); }
      if (snapAt.has(t)) snap(t, evals[evals.length - 1].step === t ? evals[evals.length - 1] : null);
      if (o.onStep) o.onStep(t, net, opt);
    }
    net.eval();
    return { net, evals, snaps, opt };
  };
  /** what the net says over a grid of the plane: p(class 1) at the corners of an n-wide grid (rows from the top) */
  P.fieldOf = function (net, g) {
    const n = g.n || 48, ny = g.ny || Math.max(2, Math.round((n * (g.h || 1)) / (g.w || 1))), nx = n + 1, nyy = ny + 1, pts = [];
    for (let j = 0; j < nyy; j++) { const v = g.yr[1] - ((g.yr[1] - g.yr[0]) * j) / ny; for (let i = 0; i < nx; i++) pts.push([g.xr[0] + ((g.xr[1] - g.xr[0]) * i) / n, v]); }
    const out = net.predict(S.T.tensor(pts)), L = out.values(), k = out.shape[1], p = new Float64Array(pts.length);
    for (let i = 0; i < pts.length; i++) p[i] = k === 1 ? P.sigmoid(L[i]) : P.sigmoid(L[i * k + 1] - L[i * k]);
    return { nx, ny: nyy, vals: p };
  };
  /** the field as a shaded raster (cell centres from the corners) and its p = 0.5 line, over axes ax */
  P.boundary = function (ax, key, field, o) {
    o = o || {};
    const { nx, ny, vals } = field, cx = nx - 1, cy = ny - 1, cells = new Array(cx * cy);
    for (let j = 0; j < cy; j++) for (let i = 0; i < cx; i++) cells[j * cx + i] = (vals[j * nx + i] + vals[j * nx + i + 1] + vals[(j + 1) * nx + i] + vals[(j + 1) * nx + i + 1]) / 4;
    const items = [{ key: key + ':f', kind: 'raster', x: ax.o.x, y: ax.o.y, w: ax.o.w, h: ax.o.h, nx: cx, ny: cy, vals: cells, scheme: o.scheme || 'prob', max: o.max, alpha: o.alpha, smooth: true, layer: 'back', z: 0 }];
    if (o.line !== false) items.push(S.Kit.scatter.level(ax, key + ':b', { nx, ny, vals, x: ax.o.x, y: ax.o.y, w: ax.o.w, h: ax.o.h }, o.level === undefined ? 0.5 : o.level, o.cls || '', { minPts: 1 }));
    return items;
  };
  /** the parameters of a state dict as flat arrays: {path: [..]} */
  P.flatState = (state) => Object.fromEntries(Object.entries(state).map(([k, v]) => [k, Array.from(v.data)]));
  /** the dense weights of an MLP as kit-net matrices (out × in) and its biases */
  P.netWeights = function (net) {
    return net.layers.map((l) => { const [i, o] = l.W.shape, w = l.W.values(); return Array.from({ length: o }, (_, a) => Array.from({ length: i }, (_, b) => w[b * o + a])); });
  };
  P.netBiases = (net) => net.layers.map((l) => Array.from(l.b.values()));
  // ---- activations ----
  P.ACTS = {
    sigmoid: { label: 'sigmoid', range: [-0.1, 1.1], mid: 0.5 },
    tanh: { label: 'tanh', range: [-1.15, 1.15], mid: 0 },
    relu: { label: 'ReLU', range: [-0.5, 5.2], mid: 0 },
    gelu: { label: 'GELU', range: [-0.5, 5.2], mid: 0 },
  };
  P.actOptions = (ids) => (ids || Object.keys(P.ACTS)).map((id) => ({ value: id, label: P.ACTS[id].label }));
  /** f(z) and f′(z) of an activation at the given points, from the engine (the slope by backpropagation) */
  P.actCurve = function (act, zs) {
    const T = S.T, z = T.param(T.tensor(Array.from(zs)));
    const a = T[act](z);
    T.sum(a).backward();
    return { f: Array.from(a.values()), d: Array.from(z.grad.values()) };
  };
  /** the accuracy as a whole-number percentage, quoted */
  P.qpct = (q, v) => q.pct(v);
})(globalThis.Nooral = globalThis.Nooral || {});
// Network diagrams whose neurons are small region maps: each hidden neuron is drawn as a square showing its own
// activation over the input plane (blue where positive, vermilion where negative), and the weights join the squares.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml;
  /**
   * P.mapNet({prefix, x, y, w, h, sizes, half, weights (out × in per layer), maps: [[vals per neuron]..] per hidden
   * layer, n (map width in cells), wmax, names: [[..]..] for plain layers, lit(l, i)}) -> items
   */
  P.mapNet = function (o) {
    const L = S.Kit.net.layout({ x: o.x, y: o.y, w: o.w, h: o.h, sizes: o.sizes, r: o.half, gapY: o.gapY }), items = [], half = o.half, r2 = P.r2;
    const isMap = (l) => l > 0 && l < o.sizes.length - 1 && o.maps && o.maps[l - 1];
    const edge = (p, ux, uy, l) => { const t = isMap(l) ? half / Math.max(Math.abs(ux), Math.abs(uy)) + 2 : (o.r || half) + 1; return [p.x + ux * t, p.y + uy * t]; };
    (o.weights || []).forEach((W, l) => W.forEach((row, i) => row.forEach((w, j) => {
      const a = L.pos[l][j], b = L.pos[l + 1][i], dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
      const s = edge(a, ux, uy, l), e = edge(b, -ux, -uy, l + 1), pt = S.Kit.net.weightPaint(w, o.wmax || 1, o.wide || 5);
      items.push({ key: `${o.prefix}:w${l}_${i}_${j}`, kind: 'path', d: `M${r2(s[0])},${r2(s[1])} L${r2(e[0])},${r2(e[1])}`, cls: 'weight', pathAudit: 'edge-x', paint: pt, layer: 'lines', z: Math.round(Math.abs(w) * 100) });
    })));
    L.pos.forEach((col, l) => col.forEach((p, i) => {
      const lit = o.lit ? o.lit(l, i) : '';
      if (isMap(l)) {
        items.push({ key: `${o.prefix}:m${l}_${i}`, kind: 'raster', x: r2(p.x - half), y: r2(p.y - half), w: 2 * half, h: 2 * half, nx: o.n, ny: o.n, vals: o.maps[l - 1][i], scheme: 'div', max: o.mapMax || 1, smooth: true, cls: 'p09map ' + (lit || ''), layer: 'items' });
      } else {
        items.push({ key: `${o.prefix}:n${l}_${i}`, kind: 'state', x: p.x, y: p.y, r: o.r || half, text: o.names && o.names[l] ? o.names[l][i] || '' : '', size: 13, cls: 'neuron ' + (l === 0 ? 'input ' : '') + (lit || '') });
      }
    }));
    return { items, L };
  };
  /** each hidden neuron's activation over the plane, n × n cells, for every hidden layer of an MLP */
  P.neuronMaps = function (net, xr, yr, n) {
    const pts = [];
    for (let j = 0; j < n; j++) { const v = yr[1] - ((yr[1] - yr[0]) * (j + 0.5)) / n; for (let i = 0; i < n; i++) pts.push([xr[0] + ((xr[1] - xr[0]) * (i + 0.5)) / n, v]); }
    net.predict(S.T.tensor(pts));
    const out = [];
    for (let l = 0; l + 1 < net.layers.length; l++) {
      const a = net.outputs['act' + l], v = a.values(), k = a.shape[1];
      out.push(Array.from({ length: k }, (_, c) => Array.from({ length: pts.length }, (_, p) => v[p * k + c])));
    }
    return out;
  };
  /** a short heading over a picture */
  P.head = (key, x, y, text) => H().text(key, x, y, text, 'title', 13);
})(globalThis.Nooral = globalThis.Nooral || {});
