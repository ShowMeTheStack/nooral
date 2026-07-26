// Networks on the workflow canvas, trained batch by batch: the trainer every network shares (a validation share held
// back, an epoch counted as a pass over the training rows, train and validation loss and accuracy after every
// epoch), and the multilayer perceptron and the convolutional network, built with S.DL on the tensor engine. Their
// insides are kept for the pictures: the weights of every layer, what each neuron says over the plane, a probe
// example's activations, the filters and feature maps. DOM-free: this runs in the workflow's worker and in node.
(function (S) {
  'use strict';
  const WF = S.WF;
  const { r4, arr } = WF.mr;
  const T = () => S.T, DL = () => S.DL;
  const NETS = (WF.nets = WF.nets || {});

  /**
   * the shared trainer. o: {net, batchOf(list of row indexes, rng, augment) -> {x, y}, forward(row indexes) -> logits
   * (no gradient), regress, yScale: {mean, std}, augment, internals(model, net) -> JSON}
   */
  NETS.trainer = function (ctx, spec, data, o) {
    const p = ctx.params, Tn = T(), dl = DL();
    const n = data.y.length, g = Tn.rng(ctx.seed || 1);
    const order = g.permutation(n), nVal = data.y.length >= 10 ? Math.round(n * Math.max(0, Math.min(0.5, +p.valShare || 0))) : 0;
    const valIdx = order.slice(0, nVal), trIdx = order.slice(nVal);
    if (!trIdx.length) throw new Error('there is nothing left to train on after holding back the validation share');
    const bs = Math.max(1, Math.min(+p.batch || 32, trIdx.length)), bpe = Math.max(1, Math.floor(trIdx.length / bs));
    const split = (idx) => dl.split(idx, (list, rng, aug) => o.batchOf(list, rng, aug));
    const task = { train: split(trIdx), val: nVal ? split(valIdx) : null };
    const trainer = new dl.Trainer(o.net, task, { batch: bs, lr: +p.lr || 0.01, optimizer: p.optimizer || 'adam', weightDecay: +spec.params.weightDecay || 0, seed: ctx.seed || 1, augment: o.augment || null, clip: o.clip || 0 });
    const curve = [];
    let steps = 0;
    const epochs = Math.max(1, +p.epochs || 1);
    const evalOn = (split) => { const r = trainer.evaluate(split, 400); return r ? { loss: r4(r.loss), acc: r.accuracy === undefined ? null : r4(r.accuracy) } : null; };
    const model = () => {
      const net = o.net;
      const predictRaw = (X) => {
        const out = [];
        for (let a = 0; a < X.length; a += 256) {
          const lg = o.forward(X.slice(a, a + 256));
          const v = lg.values(), k = lg.shape[lg.shape.length - 1];
          for (let i = 0; i < lg.shape[0]; i++) out.push(Array.from(v.subarray(i * k, i * k + k)));
        }
        return out;
      };
      const m = WF.fitted({ family: spec.family, label: spec.label, params: spec.params, net, steps, epoch: Math.floor(steps / bpe),
        predict: o.regress ? (X) => predictRaw(X).map((r) => r[0] * o.yScale.std + o.yScale.mean) : (X) => predictRaw(X).map((r) => r.indexOf(Math.max(...r))),
        proba: o.regress ? null : (X) => predictRaw(X).map((r) => { const mx = Math.max(...r), e = r.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }) }, data);
      try { m.internals = o.internals ? o.internals(m, net) : null; } catch (e) { m.internals = { error: String(e.message || e) }; }
      return m;
    };
    const tr = {
      epochs, epoch: 0, batch: 0, batches: bpe, nTrain: trIdx.length, nVal,
      step() {
        const r = trainer.step();
        steps++;
        this.epoch = Math.floor(steps / bpe); this.batch = steps % bpe;
        const out = { loss: r4(r.loss), metrics: r.metrics && r.metrics.accuracy !== undefined ? { accuracy: r4(r.metrics.accuracy) } : null };
        if (this.batch === 0) {
          // the end of an epoch: the curves get their point
          const t = evalOn('train'), v = nVal ? evalOn('val') : null;
          const pt = { epoch: this.epoch, loss: t.loss, acc: t.acc, valLoss: v ? v.loss : null, valAcc: v ? v.acc : null };
          curve.push(pt);
          Object.assign(out, { epochEnd: true }, pt);
        }
        return out;
      },
      done() { return this.epoch >= epochs; },
      progress() { return Math.min(1, steps / (epochs * bpe)); },
      current() { const m = model(); return { out: { model: m }, curve: curve.slice(), epoch: this.epoch, epochs, batch: this.batch, batches: bpe, steps, nTrain: trIdx.length, nVal }; },
      snapshot() {
        const m = model(), fam = WF.families[spec.family];
        const v = { live: true, family: spec.family, label: spec.label, task: data.task, classes: data.classes, features: data.features.map((f) => f.name), curve: curve.slice(), epoch: this.epoch, epochs, batch: this.batch, batches: bpe, steps, internals: m.internals };
        if (fam && fam.view) Object.assign(v, fam.view(m, data, ctx));
        return v;
      },
    };
    return tr;
  };

  // ---- the multilayer perceptron --------------------------------------------------------------------------------
  const flatOf = (data) => (data.kind === 'images' ? data.w * data.h : data.features.length);
  const colStats = (X) => {
    const d = X[0].length, mean = new Array(d).fill(0), sd = new Array(d).fill(0);
    for (const r of X) for (let j = 0; j < d; j++) mean[j] += r[j] / X.length;
    for (const r of X) for (let j = 0; j < d; j++) sd[j] += (r[j] - mean[j]) ** 2 / X.length;
    return { mean, sd: sd.map((v) => Math.sqrt(v) || 1) };
  };
  WF.model('mlp', {
    label: 'Neural network (MLP)', task: 'both', order: 60, icon: 'gw', iterative: true, takes: ['table', 'images'],
    blurb: 'Layers of neurons, each a weighted sum passed through a bend (an activation). Trained by backpropagation: the error flows back through every weight, batch by batch.',
    params: [
      { id: 'hidden', label: 'Hidden layers (neurons in each)', kind: 'layers', def: [8, 8], min: 1, max: 256 },
      { id: 'act', label: 'Activation', kind: 'choice', def: 'relu', options: [['relu', 'ReLU'], ['tanh', 'tanh'], ['sigmoid', 'sigmoid'], ['gelu', 'GELU']] },
      { id: 'dropout', label: 'Dropout', kind: 'number', def: 0, min: 0, max: 0.8, step: 0.05 },
      { id: 'weightDecay', label: 'Weight decay', kind: 'number', def: 0, min: 0, max: 0.1, step: 0.0001 },
      { id: 'scale', label: 'Standardise the inputs', kind: 'bool', def: true, help: 'networks learn best from features on similar scales' },
    ],
  });
  WF.trainers.mlp = function (ctx, spec, data) {
    const Tn = T(), p = spec.params, regress = data.task === 'regression', k = regress ? 1 : data.classes.length;
    const hidden = [].concat(p.hidden || [8]).map((v) => Math.max(1, Math.round(+v) || 1)).slice(0, 6);
    const net = DL().build('mlp', { inputs: flatOf(data), hidden, outputs: k, regress, act: p.act || 'relu', dropout: +p.dropout || 0, seed: ctx.seed || 1 });
    const st = data.kind === 'table' && p.scale !== false ? colStats(data.X) : null;
    const norm = (r) => (st ? Array.from(r, (v, j) => (v - st.mean[j]) / st.sd[j]) : Array.from(r));
    let yScale = { mean: 0, std: 1 };
    if (regress) { const m = data.y.reduce((a, b) => a + b, 0) / data.y.length; yScale = { mean: m, std: Math.sqrt(data.y.reduce((a, b) => a + (b - m) ** 2, 0) / data.y.length) || 1 }; }
    const xs = data.X.map(norm);
    const batchOf = (list) => ({ x: Tn.tensor(list.map((i) => xs[i])), y: regress ? Tn.tensor(list.map((i) => [(data.y[i] - yScale.mean) / yScale.std])) : list.map((i) => data.y[i]) });
    const forward = (X) => net.predict(Tn.tensor(X.map(norm)));
    return NETS.trainer(ctx, spec, data, { net, batchOf, forward, regress, yScale, internals: (m) => NETS.mlpInside(m, net, data, xs, norm, hidden) });
  };
  /** an MLP's insides: weights (out × in) of every layer, a probe row's activations, and on 2-D data each hidden neuron's map */
  NETS.mlpInside = function (m, net, data, xs, norm, hidden) {
    const sizes = [flatOf(data)].concat(hidden, [m.task === 'regression' ? 1 : data.classes.length]);
    const W = net.layers.map((l) => { const w = l.W.toArray(); return w[0].map((_, o) => w.map((row) => r4(row[o]))); });
    const b = net.layers.map((l) => arr(l.b.values()));
    const probeAt = 0;
    net.predict(T().tensor([xs[probeAt]]));
    const acts = [Array.from(xs[probeAt], r4)].concat(hidden.map((_, i) => arr(net.outputs['act' + i].values())), [arr(net.outputs.out.values())]);
    const out = { sizes, W: data.kind === 'images' ? [null].concat(W.slice(1)) : W, b, acts, probe: probeAt, probeY: data.y[probeAt], act: net.config ? net.config.act : null, nParams: net.numParams ? net.numParams() : null, features: data.features.map((f) => f.name) };
    // images: what each first-layer neuron looks for, as a picture of its input weights
    if (data.kind === 'images') out.inputWeights = W[0].slice(0, 16).map((row) => row.map(r4));
    if (data.kind === 'images') out.w = data.w, out.h = data.h;
    // two features: what every hidden neuron says at every place of the plane
    if (data.kind === 'table' && data.features.length === 2) {
      const xr = WF.mv.range(data.X, 0), yr = WF.mv.range(data.X, 1), n = 20, pts = WF.mv.corners(xr, yr, n);
      net.predict(T().tensor(pts.map(norm)));
      out.maps = { n, xr, yr, layers: hidden.map((h, l) => { const v = net.outputs['act' + l].values(); return Array.from({ length: Math.min(h, 10) }, (_, j) => pts.map((_, i) => r4(v[i * h + j]))); }) };
    }
    return out;
  };

  // ---- the convolutional network ---------------------------------------------------------------------------------
  WF.model('cnn', {
    label: 'Convolutional network (CNN)', task: 'classification', order: 62, icon: 'image', iterative: true, takes: ['images'],
    blurb: 'Small filters slide over the image and answer “is my pattern here?” at every place; pooling keeps the strongest answers; dense layers read the result. LeNet-style: two convolutions, then a classifier.',
    params: [
      { id: 'f1', label: 'Filters in the first convolution', kind: 'int', def: 6, min: 1, max: 32 },
      { id: 'f2', label: 'Filters in the second convolution', kind: 'int', def: 12, min: 1, max: 64 },
      { id: 'kernel', label: 'Filter size', kind: 'choice', def: 5, options: [[3, '3 × 3'], [5, '5 × 5']] },
      { id: 'hidden', label: 'Dense neurons', kind: 'int', def: 32, min: 4, max: 256 },
      { id: 'dropout', label: 'Dropout', kind: 'number', def: 0, min: 0, max: 0.8, step: 0.05 },
      { id: 'augment', label: 'Augment (shift and turn the images a little)', kind: 'bool', def: false },
      { id: 'weightDecay', label: 'Weight decay', kind: 'number', def: 0, min: 0, max: 0.1, step: 0.0001 },
    ],
  });
  WF.trainers.cnn = function (ctx, spec, data) {
    const Tn = T(), p = spec.params, w = data.w, h = data.h;
    if (w !== h) throw new Error('the convolutional network takes square images');
    const net = DL().build('cnn', { size: w, channels: 1, classes: data.classes.length, filters: [+p.f1 || 6, +p.f2 || 12], kernel: +p.kernel || 5, hidden: [+p.hidden || 32], dropout: +p.dropout || 0, seed: ctx.seed || 1 });
    const batchOf = (list, rng, aug) => {
      const x = Tn.zeros([list.length, 1, h, w]);
      list.forEach((i, k) => x.data.set(aug ? DL().augmentImage(data.X[i], w, h, rng, aug) : data.X[i], k * w * h));
      return { x, y: list.map((i) => data.y[i]) };
    };
    const forward = (X) => { const x = Tn.zeros([X.length, 1, h, w]); X.forEach((px, k) => x.data.set(px, k * w * h)); return net.predict(x); };
    return NETS.trainer(ctx, spec, data, { net, batchOf, forward, augment: p.augment ? { shift: Math.max(1, Math.round(w / 14)), rotate: 0.15 } : null, internals: (m) => NETS.cnnInside(m, net, data, forward) });
  };
  const mapsOf = (t, max) => { const [, C, H, W] = t.shape, v = t.values(); return Array.from({ length: Math.min(C, max) }, (_, c) => Array.from(v.subarray(c * H * W, (c + 1) * H * W), r4)); };
  /** a CNN's insides: the first filters, and one probe image's feature maps and probabilities */
  NETS.cnnInside = function (m, net, data, forward) {
    const probe = 0, lg = forward([data.X[probe]]), o = net.outputs;
    const W1 = net.conv1.W, [f1, , k] = W1.shape, wv = W1.values();
    const shape = (t) => t.shape.slice(1);
    const lv = Array.from(lg.values()), mx = Math.max(...lv), e = lv.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0);
    return { k, filters: Array.from({ length: f1 }, (_, c) => Array.from(wv.subarray(c * k * k, (c + 1) * k * k), r4)), probe, probeY: data.y[probe], image: Array.from(data.X[probe], r4), w: data.w, h: data.h,
      relu1: mapsOf(o.relu1, 8), relu1Shape: shape(o.relu1), pool1: mapsOf(o.pool1, 8), pool1Shape: shape(o.pool1), relu2: mapsOf(o.relu2, 16), relu2Shape: shape(o.relu2), probs: e.map((v) => r4(v / s)), nParams: net.numParams() };
  };
  // the Train view says where a network's run stands: the batch, the batches an epoch, the rows held back
  const train = WF.def('train');
  if (train && !train.view.__counts) {
    const tv = train.view;
    train.view = function (r, ctx) { return Object.assign(tv.call(this, r, ctx), r.batches ? { batch: r.batch, batches: r.batches, nTrain: r.nTrain, nVal: r.nVal, steps: r.steps } : {}); };
    train.view.__counts = true;
  }
  for (const id of ['mlp', 'cnn']) WF.families[id].view = (m, data) => (id === 'mlp' ? WF.mv.basic(m, data) : {});
})(globalThis.Nooral = globalThis.Nooral || {});
