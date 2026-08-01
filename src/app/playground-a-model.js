// The playground's model (S.PG): a small network on 2-D data, built from the tensor engine (S.T: Dense layers,
// autodiff, SGD or Adam), trained a batch at a time, with plain-number copies of its weights for drawing every
// neuron's region over the plane. Nothing here touches the page, so node tests drive it the same way.
//   PG.make(cfg) -> lab   cfg {data, n, noise, testShare, seed, features: [ids], layers: [sizes], act, lr, reg,
//                              regRate, batch, optimizer}
//   lab.step() one batch      lab.epochStep() the rest of the epoch     lab.epoch, lab.steps
//   lab.losses() -> {train, test}   lab.weights() -> {W: [out × in per layer], b}   lab.forward(x1, x2) -> [acts per layer]
(function (S) {
  'use strict';
  const PG = (S.PG = S.PG || {});
  PG.DATASETS = [
    ['circles', 'Circles'], ['xor', 'XOR'], ['moons', 'Two moons'], ['spirals', 'Spirals'], ['blobs', 'Two blobs'], ['checkerboard', 'Checkerboard'], ['linear', 'Noisy line'],
  ];
  /** the inputs a reader may switch on; each is a function of the point (x1, x2) on the plane [−3, 3]² */
  PG.FEATURES = [
    { id: 'x1', label: 'x₁', f: (a) => a },
    { id: 'x2', label: 'x₂', f: (a, b) => b },
    { id: 'x1sq', label: 'x₁²', f: (a) => a * a },
    { id: 'x2sq', label: 'x₂²', f: (a, b) => b * b },
    { id: 'x1x2', label: 'x₁x₂', f: (a, b) => a * b },
    { id: 'sin1', label: 'sin x₁', f: (a) => Math.sin(a) },
    { id: 'sin2', label: 'sin x₂', f: (a, b) => Math.sin(b) },
  ];
  PG.ACTS = { relu: (z) => (z > 0 ? z : 0), tanh: Math.tanh, sigmoid: (z) => 1 / (1 + Math.exp(-z)), linear: (z) => z };
  PG.LRS = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];
  PG.RATES = [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
  PG.DOMAIN = 3;
  PG.defaults = () => ({ data: 'circles', n: 300, noise: 0.1, testShare: 0.5, seed: 1, features: ['x1', 'x2'], layers: [4, 2], act: 'tanh', lr: 0.1, reg: 'none', regRate: 0, batch: 10, optimizer: 'sgd' });

  /** the points of a dataset, scaled into [−2.7, 2.7]², split into train and test */
  PG.data = function (cfg) {
    const D = S.Data, n = Math.max(20, Math.min(1000, cfg.n | 0));
    const opts = { n, seed: cfg.seed };
    if (cfg.data !== 'checkerboard' && cfg.data !== 'blobs') opts.noise = cfg.noise;
    if (cfg.data === 'blobs') { opts.k = 2; opts.std = 0.8 + cfg.noise * 6; }
    if (cfg.data === 'checkerboard') opts.cells = 4;
    const ds = D.load(cfg.data, opts);
    // scale: the centre to 0, the farthest coordinate to 2.7
    const xs = ds.X.map((r) => r[0]), ys = ds.X.map((r) => r[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const m = Math.max(...ds.X.map((r) => Math.max(Math.abs(r[0] - cx), Math.abs(r[1] - cy)))) || 1, k = 2.7 / m;
    const X = ds.X.map((r) => [(r[0] - cx) * k, (r[1] - cy) * k]);
    const sp = D.split(ds.y, cfg.testShare, cfg.seed + 7, true);
    return { X, y: ds.y.slice(), train: sp.train, test: sp.test, name: (ds.doc && ds.doc.title) || cfg.data };
  };

  PG.make = function (cfg0) {
    const cfg = Object.assign(PG.defaults(), cfg0 || {});
    const T = S.T, nn = T.nn;
    const feats = PG.FEATURES.filter((f) => cfg.features.includes(f.id));
    if (!feats.length) throw new Error('switch on at least one input');
    const data = PG.data(cfg);
    const sizes = [feats.length].concat(cfg.layers.filter((s) => s > 0), [1]);
    const g = T.rng(cfg.seed);
    const layers = [];
    for (let i = 0; i + 1 < sizes.length; i++) layers.push(new nn.Dense(sizes[i], sizes[i + 1], { rng: g, init: cfg.act === 'relu' ? 'he' : 'xavier' }));
    const params = layers.flatMap((l) => [l.W, l.b]);
    const opt = T.optim.make(cfg.optimizer || 'sgd', params, { lr: cfg.lr });
    const featOf = (p) => feats.map((f) => f.f(p[0], p[1]));
    const Xtr = data.train.map((i) => featOf(data.X[i])), ytr = data.train.map((i) => data.y[i]);
    const Xte = data.test.map((i) => featOf(data.X[i])), yte = data.test.map((i) => data.y[i]);
    const order = S.ML.rng(cfg.seed + 101);
    const bs = Math.max(1, Math.min(cfg.batch | 0, Xtr.length));
    let perm = order.perm(Xtr.length), at = 0;
    const lab = { cfg, data, feats, sizes, layers, epoch: 0, steps: 0, history: [] };
    const actT = (x) => (cfg.act === 'linear' ? x : T[cfg.act](x));
    lab.step = function () {
      const idx = perm.slice(at, at + bs);
      at += bs;
      const xb = T.tensor(idx.map((i) => Xtr[i])), yb = T.tensor(idx.map((i) => [ytr[i]]));
      opt.zeroGrad();
      let h = xb;
      layers.forEach((l, i) => { h = l.forward(h); if (i + 1 < layers.length) h = actT(h); });
      let loss = T.bceWithLogits(h, yb);
      if (cfg.reg !== 'none' && cfg.regRate > 0) {
        let pen = null;
        for (const l of layers) { const t = T.sum(cfg.reg === 'l1' ? T.abs(l.W) : T.mul(l.W, l.W)); pen = pen ? T.add(pen, t) : t; }
        loss = T.add(loss, T.mul(pen, cfg.regRate * (cfg.reg === 'l2' ? 0.5 : 1)));
      }
      loss.backward();
      opt.step();
      lab.steps++;
      if (at >= Xtr.length) { at = 0; perm = order.perm(Xtr.length); lab.epoch++; const L = lab.losses(); lab.history.push({ epoch: lab.epoch, train: L.train, test: L.test }); }
      return loss.item();
    };
    /** the rest of the current epoch */
    lab.epochStep = function () { const e = lab.epoch; let guard = 0; while (lab.epoch === e && guard++ < 1e5) lab.step(); };
    /** the weights as plain numbers: W[l][out][in], b[l][out] */
    lab.weights = function () {
      return {
        W: layers.map((l) => { const v = l.W.values(), [i, o] = l.W.shape; return Array.from({ length: o }, (_, r) => Array.from({ length: i }, (_, c) => v[c * o + r])); }),
        b: layers.map((l) => Array.from(l.b.values())),
      };
    };
    /** every layer's values at a point of the plane: [inputs, hidden..., [p(class 1)]] (cached weights wk) */
    lab.forwardWith = function (wk, x1, x2) {
      const act = PG.ACTS[cfg.act], out = [featOf([x1, x2])];
      let h = out[0];
      wk.W.forEach((W, l) => {
        const last = l + 1 === wk.W.length;
        h = W.map((row, r) => { let z = wk.b[l][r]; for (let c = 0; c < row.length; c++) z += row[c] * h[c]; return last ? 1 / (1 + Math.exp(-z)) : act(z); });
        out.push(h);
      });
      return out;
    };
    lab.forward = (x1, x2) => lab.forwardWith(lab.weights(), x1, x2);
    /** mean log loss on the training and the test points */
    lab.losses = function (wk) {
      wk = wk || lab.weights();
      const ll = (idx) => { let s = 0; for (const i of idx) { const p = lab.forwardWith(wk, data.X[i][0], data.X[i][1]).pop()[0], q = Math.min(1 - 1e-7, Math.max(1e-7, p)); s -= data.y[i] ? Math.log(q) : Math.log(1 - q); } return idx.length ? s / idx.length : 0; };
      const acc = (idx) => idx.length ? idx.filter((i) => (lab.forwardWith(wk, data.X[i][0], data.X[i][1]).pop()[0] >= 0.5 ? 1 : 0) === data.y[i]).length / idx.length : 0;
      return { train: ll(data.train), test: ll(data.test), trainAcc: acc(data.train), testAcc: acc(data.test) };
    };
    lab.history.push(Object.assign({ epoch: 0 }, (() => { const L = lab.losses(); return { train: L.train, test: L.test }; })()));
    lab.setLr = (v) => { opt.lr = v; cfg.lr = v; };
    lab.numParams = () => layers.reduce((s, l) => s + l.W.size + l.b.size, 0);
    return lab;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
