// Multilayer perceptrons and convolutional networks, with image augmentation and their demo tasks.
//   DL.build('mlp', {inputs, hidden: [16, 16], outputs, regress: false, act: 'relu', dropout, seed})
//      named outputs dense0, act0, dense1, act1, ..., out
//   DL.build('cnn', {size: 16 | 28, channels: 1, classes, filters: [6, 16], kernel: 5, hidden, batchNorm, dropout, seed})
//      LeNet-style: conv (same padding) → ReLU → 2×2 max pool → conv (valid) → ReLU → pool → dense layers;
//      named outputs conv1 relu1 pool1 conv2 relu2 pool2 flat fc0 fcAct0 ... logits; Grad-CAM uses relu2
//   DL.augmentImage(px, w, h, rng, {shift: pixels, rotate: radians, scale, noise}) -> a new Float32Array
//   tasks: 'digits' and 'shapes' {size, n, val, seed, augment: {shift, rotate, noise}} (generated images; augmentation
//          applies to training batches only); 'moons' {n, noise}; 'curve' {shape, n} for the MLP
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});

  DL.register('mlp', (c) => {
    const T = S.T, nn = T.nn, Net = DL.makeNet(), g = T.rng(c.seed || 1);
    const sizes = [c.inputs || 2].concat(c.hidden || [16, 16], [c.outputs || c.classes || 2]);
    class MLP extends Net {
      constructor() {
        super();
        this.layers = [];
        for (let i = 0; i + 1 < sizes.length; i++) this.layers.push(new nn.Dense(sizes[i], sizes[i + 1], { rng: g, init: (c.act || 'relu') === 'relu' ? 'he' : 'xavier' }));
        this.drop = c.dropout ? new nn.Dropout(c.dropout, { rng: g.fork() }) : null;
        this.isRegression = !!c.regress;
        this.snapshotNames = ['layers.0.W', 'act0', 'out'];
      }
      forward(x) {
        this.begin();
        let h = x;
        this.layers.forEach((l, i) => {
          h = this.tap('dense' + i, l.forward(h));
          if (i + 1 < this.layers.length) {
            h = this.tap('act' + i, S.T[c.act || 'relu'](h));
            if (this.drop) h = this.drop.forward(h);
          }
        });
        return this.tap('out', h);
      }
      lossOn(b) {
        if (!this.isRegression) return super.lossOn(b);
        const out = this.forward(b.x);
        return { loss: S.T.mseLoss(out, b.y), logits: out, correct: 0, count: 0 };
      }
    }
    return new MLP();
  });

  DL.register('cnn', (c) => {
    const T = S.T, nn = T.nn, Net = DL.makeNet(), g = T.rng(c.seed || 1);
    const size = c.size || 16, ch = c.channels || 1, k = c.kernel || 5, [f1, f2] = c.filters || [6, 16];
    const s1 = Math.floor(size / 2), s2 = Math.floor((s1 - k + 1) / 2);
    if (s2 < 1) throw new Error('images of ' + size + ' pixels are too small for ' + k + '×' + k + ' filters');
    const hidden = c.hidden || (size >= 28 ? [120, 84] : [64]), flat = f2 * s2 * s2;
    class CNN extends Net {
      constructor() {
        super();
        this.conv1 = new nn.Conv2d(ch, f1, k, { rng: g, padding: (k - 1) >> 1 });
        this.conv2 = new nn.Conv2d(f1, f2, k, { rng: g });
        if (c.batchNorm) { this.bn1 = new nn.BatchNorm(f1); this.bn2 = new nn.BatchNorm(f2); }
        const sizes = [flat].concat(hidden, [c.classes || 10]);
        this.fc = sizes.slice(1).map((n, i) => new nn.Dense(sizes[i], n, { rng: g, init: i + 2 < sizes.length ? 'he' : 'xavier' }));
        this.drop = c.dropout ? new nn.Dropout(c.dropout, { rng: g.fork() }) : null;
        this.camLayer = 'relu2';
        this.shapes = { input: [ch, size, size], conv1: [f1, size, size], pool1: [f1, s1, s1], conv2: [f2, s1 - k + 1, s1 - k + 1], pool2: [f2, s2, s2], flat };
        this.snapshotNames = ['conv1.W', 'conv2.W', 'relu1', 'relu2', 'pool2', 'logits'];
      }
      forward(x) {
        this.begin();
        const t = (n, v) => this.tap(n, v);
        let h = t('conv1', this.conv1.forward(x));
        if (this.bn1) h = t('bn1', this.bn1.forward(h));
        h = t('pool1', T.maxPool2d(t('relu1', T.relu(h)), 2));
        h = t('conv2', this.conv2.forward(h));
        if (this.bn2) h = t('bn2', this.bn2.forward(h));
        h = t('flat', T.flatten(t('pool2', T.maxPool2d(t('relu2', T.relu(h)), 2)), 1));
        this.fc.forEach((l, i) => {
          if (i + 1 < this.fc.length) {
            h = t('fcAct' + i, T.relu(t('fc' + i, l.forward(h))));
            if (this.drop) h = this.drop.forward(h);
          } else h = t('logits', l.forward(h));
        });
        return h;
      }
    }
    return new CNN();
  });

  /** a shifted, rotated, rescaled and noisy copy of an image (bilinear sampling about the centre) */
  DL.augmentImage = (px, w, h, rng, o = {}) => {
    const out = new Float32Array(w * h), sh = o.shift || 0, rot = o.rotate || 0, sc = o.scale || 0;
    const dx = sh ? (rng.next() * 2 - 1) * sh : 0, dy = sh ? (rng.next() * 2 - 1) * sh : 0;
    const a = rot ? (rng.next() * 2 - 1) * rot : 0, s = sc ? 1 + (rng.next() * 2 - 1) * sc : 1;
    const cos = Math.cos(a) / s, sin = Math.sin(a) / s, cx = (w - 1) / 2, cy = (h - 1) / 2;
    const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : px[y * w + x]);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const u = x - cx - dx, v = y - cy - dy, sx = cos * u + sin * v + cx, sy = -sin * u + cos * v + cy;
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
      let val = at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy) + at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy;
      if (o.noise) val += rng.normal(0, o.noise);
      out[y * w + x] = val < 0 ? 0 : val > 1 ? 1 : val;
    }
    return out;
  };

  /** a task from an image set: the first n images train, the next val images validate */
  const imageTask = (set, n, classes, o) => {
    const w = set.w, h = set.h, ex = Array.from({ length: set.n }, (_, i) => ({ px: set.get(i), y: set.y[i] }));
    const batchOf = (list, rng, aug) => {
      const x = S.T.zeros([list.length, 1, h, w]);
      list.forEach((e, i) => x.data.set(aug ? DL.augmentImage(e.px, w, h, rng, aug) : e.px, i * w * h));
      return { x, y: list.map((e) => e.y) };
    };
    const train = DL.split(ex.slice(0, n), batchOf), val = DL.split(ex.slice(n), batchOf);
    return { kind: 'cnn', train, val, probe: val.get([0]), classes, model: { size: w, channels: 1, classes: classes.length },
      fit: { batch: 32, lr: 2e-3, evalEvery: o.evalEvery || 0, augment: o.augment === undefined ? null : o.augment },
      info: { w, h, classes } };
  };
  DL.tasks.register('digits', (o) => {
    const n = o.n || 3000, v = o.val || 600, set = S.Data.digits({ n: n + v, size: o.size || 16, seed: o.seed || 1 });
    return imageTask(set, n, set.classes.map(String), o);
  });
  DL.tasks.register('shapes', (o) => {
    const n = o.n || 2400, v = o.val || 400, set = S.Data.shapes({ n: n + v, size: o.size || 16, seed: o.seed || 1 });
    return imageTask(set, n, set.classes.slice(), o);
  });

  /** tabular tasks for the MLP: rows of numbers, with class ids or a number to predict */
  const tableTask = (tr, va, o) => {
    const batchOf = (list) => ({ x: S.T.tensor(list.map((e) => e.x)), y: o.regress ? S.T.tensor(list.map((e) => [e.y])) : list.map((e) => e.y) });
    const ex = (d) => d.X.map((x, i) => ({ x, y: d.y[i] }));
    const val = DL.split(ex(va), batchOf);
    return { kind: 'mlp', train: DL.split(ex(tr), batchOf), val, probe: val.get([0, 1, 2, 3]), model: { inputs: tr.X[0].length, outputs: o.regress ? 1 : 2, regress: !!o.regress, hidden: [16, 16], act: o.regress ? 'tanh' : 'relu' }, fit: { batch: 32, lr: 0.01 } };
  };
  DL.tasks.register('moons', (o) => tableTask(S.Data.moons({ n: o.n || 400, noise: o.noise || 0.15, seed: o.seed || 1 }), S.Data.moons({ n: 400, noise: o.noise || 0.15, seed: (o.seed || 1) + 100 }), {}));
  DL.tasks.register('curve', (o) => tableTask(S.Data.curve({ n: o.n || 200, shape: o.shape || 'sine', seed: o.seed || 1 }), S.Data.curve({ n: 200, shape: o.shape || 'sine', seed: (o.seed || 1) + 100 }), { regress: true }));
})(globalThis.Nooral = globalThis.Nooral || {});
