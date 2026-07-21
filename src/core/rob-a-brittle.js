// Robustness (S.Rob), part 1: why models are brittle, and defences. Taught defensively: a toy classifier on 2-D
// points or on our generated digits, the smallest step along the loss gradient's sign that flips its answer
// (FGSM-style), then adversarial training and input smoothing as the remedies.
//   S.Rob.data('moons' | 'digits', {n, seed}) -> {Xtr, ytr, Xte, yte, dim, classes, image}
//   S.Rob.classifier({data, hidden, steps, lr, batch, seed, adv: {eps, mix: 0.5}}) -> trainer with
//      predict(X) -> labels, probs(X), accuracy(X?, y?), model
//   S.Rob.inputGradient(clf, x, y) -> d loss / d x for one input (an array)
//   S.Rob.fgsm(clf, x, y, eps) -> {x: perturbed, delta, before, after, trace}  (pixels stay in [0, 1] for images)
//   S.Rob.fgsmBatch(clf, X, y, eps) -> perturbed rows
//   S.Rob.smallestFlip(clf, x, y, {maxEps}) -> {eps, x, trace}: the least eps along the sign direction that changes the
//      prediction (bisection), or null when none up to maxEps does
//   S.Rob.robustCurve(clf, epsList, {n}) -> [{eps, accuracy}] on test points perturbed with FGSM at each eps
//   S.Rob.smoothPredict(clf, x, {sigma, samples, seed}) -> {label, votes}: the majority over noisy copies
//   S.Rob.defences(clf, eps, {sigma, samples, n, bits}) -> accuracy on clean and perturbed inputs, plain and defended
//   S.Rob.logreg(X, y, {weights, l2, iters, lr}) -> {w, b, prob(x), predict(x)}: weighted logistic regression
(function (S) {
  'use strict';
  const R = (S.Rob = S.Rob || {});
  const fmt = (x) => S.T.fmt(x);
  R.fmt = fmt;

  R.data = (kind = 'moons', o = {}) => {
    const D = S.Data;
    if (kind === 'digits') {
      const tr = S.Gen.images('digits', { n: o.n || 800, seed: o.seed || 1, size: 16 }), te = S.Gen.images('digits', { n: o.nTest || 300, seed: (o.seed || 1) + 500, size: 16 });
      const rows = (s) => Array.from({ length: s.n }, (_, i) => Array.from(s.X.subarray(i * s.P, (i + 1) * s.P)));
      return { Xtr: rows(tr), ytr: tr.y, Xte: rows(te), yte: te.y, dim: tr.P, classes: tr.classes, image: true, size: 16 };
    }
    const tr = D.moons({ n: o.n || 400, noise: o.noise || 0.15, seed: o.seed || 1 }), te = D.moons({ n: o.nTest || 400, noise: o.noise || 0.15, seed: (o.seed || 1) + 500 });
    return { Xtr: tr.X, ytr: tr.y, Xte: te.X, yte: te.y, dim: 2, classes: [0, 1], image: false };
  };

  R.classifier = (o0 = {}) => {
    const T = S.T, nn = T.nn, data = o0.dataset || R.data(o0.data || 'moons', o0);
    const o = Object.assign({ hidden: data.image ? 64 : 32, steps: data.image ? 600 : 800, lr: data.image ? 3e-3 : 1e-2, batch: 64 }, o0);
    const C = data.classes.length, g = T.rng(o.seed || 1), r = T.rng((o.seed || 1) * 11 + 2);
    const model = new nn.MLP(data.image ? [data.dim, o.hidden, C] : [data.dim, o.hidden, o.hidden, C], { rng: g, act: data.image ? 'relu' : 'tanh' });
    const opt = new T.optim.Adam(model.parameters(), { lr: o.lr });
    const clf = { model, data, image: data.image, C };
    clf.logits = (X) => T.noGrad(() => model.forward(T.tensor(X)));
    clf.probs = (X) => { const z = T.softmax(clf.logits(X), -1).values(), out = []; for (let i = 0; i < X.length; i++) out.push(Array.from(z.subarray(i * C, (i + 1) * C))); return out; };
    clf.predict = (X) => clf.probs(X).map((p) => p.indexOf(Math.max(...p)));
    clf.accuracy = (X = data.Xte, y = data.yte) => { const p = clf.predict(X); return p.filter((v, i) => v === y[i]).length / y.length; };
    const tr = S.Gen.trainer({
      snapEvery: o.snapEvery || 100,
      step() {
        let idx = Array.from({ length: o.batch }, () => r.int(data.Xtr.length)), X = idx.map((i) => data.Xtr[i]);
        const y = idx.map((i) => data.ytr[i]);
        // adversarial training: replace a share of the batch by its own FGSM perturbations against the current model
        if (o.adv && o.adv.eps) { const k = Math.round(o.batch * (o.adv.mix == null ? 0.5 : o.adv.mix)); X = R.fgsmBatch(clf, X.slice(0, k), y.slice(0, k), o.adv.eps).concat(X.slice(k)); }
        opt.zeroGrad();
        const loss = T.crossEntropy(model.forward(T.tensor(X)), y);
        loss.backward(); opt.step();
        return { loss: loss.item() };
      },
      snapshot: () => ({ accuracy: clf.accuracy() }),
    });
    return Object.assign(tr, clf, { adv: o.adv || null });
  };

  R.inputGradient = (clf, x, y) => {
    const T = S.T, xt = T.param(T.tensor([x]));
    const loss = T.crossEntropy(clf.model.forward(xt), [y]);
    loss.backward();
    for (const p of clf.model.parameters()) p.grad = null;
    return { grad: Array.from(xt.grad.values()), loss: loss.item() };
  };

  const clip01 = (clf, v) => (clf.image ? Math.max(0, Math.min(1, v)) : v);
  R.fgsm = (clf, x, y, eps, o = {}) => {
    const T = S.T, { grad, loss } = R.inputGradient(clf, x, y), delta = grad.map((g) => eps * Math.sign(g)), xa = x.map((v, i) => clip01(clf, v + delta[i]));
    if (o.quiet) return { x: xa };
    const pb = clf.probs([x])[0], pa = clf.probs([xa])[0], b = pb.indexOf(Math.max(...pb)), a = pa.indexOf(Math.max(...pa));
    const gn = Math.hypot(...grad), what = clf.image ? 'every pixel moves by at most ' + fmt(eps) : 'the point moves by ' + fmt(eps) + ' along each axis';
    const trace = [
      T.step('predict', 'The model says class ' + b + ' with probability ' + fmt(pb[b]) + ' (the true class is ' + y + '; its loss is ' + fmt(loss) + ').', { probs: pb }),
      T.step('gradient', 'The gradient of the loss with respect to the input has length ' + fmt(gn) + (clf.image ? '' : ': ' + G2(grad)) + '. It points to where the loss rises fastest.', { grad }),
      T.step('sign', 'Taking only its sign and scaling by ε = ' + fmt(eps) + ': ' + what + '.', { delta }),
      T.step('after', 'Now the model says class ' + a + ' with probability ' + fmt(pa[a]) + (a !== b ? ': a tiny, deliberate change flipped the answer. Models that are this brittle need defending.' : ': this step was not enough to flip it.'), { probs: pa }),
    ];
    return { x: xa, delta, before: b, after: a, probsBefore: pb, probsAfter: pa, grad, trace };
  };
  /** FGSM for many inputs at once: the loss is summed, so each row of the input gradient is that example's own */
  R.fgsmBatch = (clf, X, y, eps) => {
    const T = S.T;
    if (!X.length) return [];
    const xt = T.param(T.tensor(X));
    T.crossEntropy(clf.model.forward(xt), y, { reduction: 'sum' }).backward();
    for (const p of clf.model.parameters()) p.grad = null;
    const g = xt.grad.values(), d = X[0].length;
    return X.map((x, i) => x.map((v, j) => clip01(clf, v + eps * Math.sign(g[i * d + j]))));
  };
  const G2 = (v) => '(' + v.map(fmt).join(', ') + ')';

  R.smallestFlip = (clf, x, y, o = {}) => {
    const T = S.T, maxEps = o.maxEps || (clf.image ? 0.5 : 2), b0 = clf.predict([x])[0];
    const { grad } = R.inputGradient(clf, x, y), at = (e) => x.map((v, i) => clip01(clf, v + e * Math.sign(grad[i])));
    if (clf.predict([at(maxEps)])[0] === b0) return null;
    let lo = 0, hi = maxEps;
    for (let k = 0; k < 30; k++) { const m = (lo + hi) / 2; if (clf.predict([at(m)])[0] === b0) lo = m; else hi = m; }
    const xa = at(hi), a = clf.predict([xa])[0];
    return { eps: hi, x: xa, before: b0, after: a, trace: [T.step('flip', 'Moving ' + (clf.image ? 'every pixel' : 'each coordinate') + ' by ' + fmt(hi) + ' in the direction that raises the loss changes the prediction from ' + b0 + ' to ' + a + '.', { eps: hi })] };
  };

  R.robustCurve = (clf, epsList, o = {}) => {
    const n = Math.min(o.n || 200, clf.data.Xte.length), X = clf.data.Xte.slice(0, n), y = clf.data.yte.slice(0, n);
    return epsList.map((eps) => ({ eps, accuracy: clf.accuracy(eps ? R.fgsmBatch(clf, X, y, eps) : X, y) }));
  };

  R.smoothPredict = (clf, x, o = {}) => {
    const g = S.T.rng(o.seed || 3), k = o.samples || 32, sigma = o.sigma == null ? 0.3 : o.sigma;
    const copies = Array.from({ length: k }, () => x.map((v) => clip01(clf, v + g.normal(0, sigma))));
    const votes = new Array(clf.C).fill(0); clf.predict(copies).forEach((p) => votes[p]++);
    return { label: votes.indexOf(Math.max(...votes)), votes };
  };
  /** feature squeezing for images: fewer grey levels, so small pixel nudges are rounded away */
  R.quantize = (x, bits = 1) => { const L = 2 ** bits - 1; return x.map((v) => Math.round(Math.max(0, Math.min(1, v)) * L) / L); };

  R.defences = (clf, eps, o = {}) => {
    const T = S.T, n = Math.min(o.n || 150, clf.data.Xte.length), X = clf.data.Xte.slice(0, n), y = clf.data.yte.slice(0, n);
    const Xa = R.fgsmBatch(clf, X, y, eps), acc = (pred) => pred.filter((p, i) => p === y[i]).length / n;
    const smooth = (Z) => Z.map((x, i) => R.smoothPredict(clf, x, { sigma: o.sigma, samples: o.samples, seed: i + 1 }).label);
    const rows = [
      { defence: 'none', clean: acc(clf.predict(X)), attacked: acc(clf.predict(Xa)) },
      { defence: 'smoothing (σ = ' + fmt(o.sigma == null ? 0.3 : o.sigma) + ')', clean: acc(smooth(X)), attacked: acc(smooth(Xa)) },
    ];
    if (clf.image) rows.push({ defence: 'squeezing (' + (o.bits || 1) + ' bit)', clean: acc(clf.predict(X.map((x) => R.quantize(x, o.bits || 1)))), attacked: acc(clf.predict(Xa.map((x) => R.quantize(x, o.bits || 1)))) });
    return { eps, rows, trace: rows.map((r) => T.step('defence', 'With ' + r.defence + ': accuracy ' + fmt(r.clean) + ' on clean inputs and ' + fmt(r.attacked) + ' on inputs perturbed by ε = ' + fmt(eps) + '.', r)) };
  };

  R.logreg = (X, y, o = {}) => {
    const n = X.length, d = X[0].length, w = new Float64Array(d), wt = o.weights || new Array(n).fill(1), l2 = o.l2 == null ? 1e-3 : o.l2, lr = o.lr || 0.5, W = wt.reduce((s, v) => s + v, 0);
    let b = 0;
    const sig = (z) => 1 / (1 + Math.exp(-z)), z = (x) => { let s = b; for (let j = 0; j < d; j++) s += w[j] * x[j]; return s; };
    for (let it = 0; it < (o.iters || 400); it++) {
      const gw = new Float64Array(d); let gb = 0;
      for (let i = 0; i < n; i++) { const e = wt[i] * (sig(z(X[i])) - y[i]); gb += e; for (let j = 0; j < d; j++) gw[j] += e * X[i][j]; }
      for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / W + l2 * w[j]);
      b -= (lr * gb) / W;
    }
    return { w: Array.from(w), get b() { return b; }, prob: (x) => sig(z(x)), predict: (x, t = 0.5) => (sig(z(x)) >= t ? 1 : 0) };
  };
  /** standardise columns with the training means and spreads */
  R.standardise = (Xtr, ...others) => {
    const d = Xtr[0].length, mu = new Array(d).fill(0), sd = new Array(d).fill(0);
    Xtr.forEach((r) => r.forEach((v, j) => { mu[j] += v / Xtr.length; }));
    Xtr.forEach((r) => r.forEach((v, j) => { sd[j] += (v - mu[j]) ** 2 / Xtr.length; }));
    const f = (X) => X.map((r) => r.map((v, j) => (v - mu[j]) / (Math.sqrt(sd[j]) || 1)));
    return [f(Xtr), ...others.map(f)];
  };

  S.lateJob('robustClassifier', S.Gen.asJob((p) => R.classifier(p)));
})(globalThis.Nooral = globalThis.Nooral || {});
