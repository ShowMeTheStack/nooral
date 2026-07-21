// Privacy, part 3: the intuition of DP-SGD on a logistic regression over the small-loans table. Each step clips every
// example's own gradient to length C (so no single person can pull the model far) and adds Gaussian noise of
// standard deviation σC to the sum (so no single person's presence can be told apart). The per-example gradients
// of logistic regression have a closed form, (p − y)·[x, 1], so every one of them can be shown.
//   S.Rob.perExampleGrads(w, b, X, y) -> rows [∂/∂w..., ∂/∂b]
//   S.Rob.dpEpsilon({q, sigma, steps, delta}) -> {eps, order, approximate: true, say}
//      an APPROXIMATE accountant in the moments style: per step, order α costs about
//      min(α / (2σ²), 2q²α / σ²); over T steps the total is converted to (ε, δ) by ε = min_α T·cost + log(1/δ) / (α − 1).
//      It is a teaching estimate, not a certified bound.
//   S.Rob.dpsgd({sigma: 1, clip: 1, lr: 0.5, epochs: 3, batch: 16, seed, delta: 1e-5}) ->
//      {accuracy, eps, history [{step, loss}], trace (the first step spelled out), w, b}
//   S.Rob.dpTradeoff({sigmas: [0, 1, 2, 4, 8, 16], seeds: [1, 2, 3]}) -> rows [{sigma, eps, accuracy}]
(function (S) {
  'use strict';
  const R = S.Rob, fmt = (x) => R.fmt(x);
  const sig = (z) => 1 / (1 + Math.exp(-z));

  /** the loans table without the group column, standardised; a fixed train / test split */
  R.loansData = (seed = 1) => {
    const ds = S.Data.load('loans', { seed, n: 600 }), Xall = ds.X.map((r) => r.slice(0, 5)), sp = S.Data.split(ds.y, 0.3, seed);
    const [Xtr, Xte] = R.standardise(sp.train.map((i) => Xall[i]), sp.test.map((i) => Xall[i]));
    return { Xtr, ytr: sp.train.map((i) => ds.y[i]), Xte, yte: sp.test.map((i) => ds.y[i]), gtr: sp.train.map((i) => ds.X[i][5]), gte: sp.test.map((i) => ds.X[i][5]), raw: ds };
  };

  R.perExampleGrads = (w, b, X, y) => X.map((x, i) => { let z = b; for (let j = 0; j < w.length; j++) z += w[j] * x[j]; const e = sig(z) - y[i]; return x.map((v) => e * v).concat([e]); });

  R.dpEpsilon = (o = {}) => {
    const q = o.q, s = o.sigma, steps = o.steps, delta = o.delta || 1e-5;
    if (!s) return { eps: Infinity, order: null, approximate: true, say: 'With no noise there is no privacy guarantee: ε is infinite.' };
    let best = Infinity, order = null;
    for (const a of [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256]) {
      const cost = Math.min(a / (2 * s * s), (2 * q * q * a) / (s * s)), e = steps * cost + Math.log(1 / delta) / (a - 1);
      if (e < best) { best = e; order = a; }
    }
    return { eps: best, order, approximate: true, say: 'Approximately ε ≈ ' + fmt(best) + ' at δ = ' + fmt(delta) + ' after ' + steps + ' steps with sampling rate ' + fmt(q) + ' and noise multiplier ' + fmt(s) + ' (a moments-style estimate, best at order ' + order + '; not a certified bound).' };
  };

  R.dpsgd = (o0 = {}) => {
    const T = S.T, o = Object.assign({ sigma: 1, clip: 1, lr: 0.5, epochs: 3, batch: 16, delta: 1e-5 }, o0), data = o.data || R.loansData(o.dataSeed || 1);
    const X = data.Xtr, y = data.ytr, n = X.length, d = X[0].length, g = T.rng(o.seed || 1), w = new Array(d).fill(0);
    let b = 0;
    const steps = Math.round((o.epochs * n) / o.batch), history = [], trace = [];
    for (let t = 0; t < steps; t++) {
      const idx = Array.from({ length: o.batch }, () => g.int(n)), G = R.perExampleGrads(w, b, idx.map((i) => X[i]), idx.map((i) => y[i]));
      const norms = G.map((r) => Math.hypot(...r)), sum = new Array(d + 1).fill(0);
      let clipped = 0;
      G.forEach((r, k) => { const f = Math.min(1, o.clip / (norms[k] || 1e-12)); if (f < 1) clipped++; r.forEach((v, j) => { sum[j] += v * f; }); });
      const noise = sum.map(() => g.normal(0, o.sigma * o.clip)), upd = sum.map((v, j) => (v + noise[j]) / o.batch);
      if (t === 0) trace.push(
        T.step('grads', 'Step 1 draws ' + o.batch + ' applicants; their own gradients have lengths from ' + fmt(Math.min(...norms)) + ' to ' + fmt(Math.max(...norms)) + '.', { norms }),
        T.step('clip', clipped + ' of them are longer than C = ' + fmt(o.clip) + ' and are scaled down to exactly C, so nobody can pull the model further than C.', { clipped }),
        T.step('noise', 'Gaussian noise with standard deviation σC = ' + fmt(o.sigma * o.clip) + ' is added to the sum of the clipped gradients (length ' + fmt(Math.hypot(...sum)) + ', noise length ' + fmt(Math.hypot(...noise)) + ').', { sum, noise }),
        T.step('update', 'The noisy average is used as the gradient: the weights move by learning rate ' + fmt(o.lr) + ' times a vector of length ' + fmt(Math.hypot(...upd)) + '.', { update: upd }));
      for (let j = 0; j < d; j++) w[j] -= o.lr * upd[j];
      b -= o.lr * upd[d];
      if (t % 10 === 0 || t === steps - 1) { let L = 0; for (let i = 0; i < n; i++) { let z = b; for (let j = 0; j < d; j++) z += w[j] * X[i][j]; const p = Math.min(1 - 1e-9, Math.max(1e-9, sig(z))); L -= y[i] ? Math.log(p) : Math.log(1 - p); } history.push({ step: t + 1, loss: L / n }); }
    }
    const pred = (x) => { let z = b; for (let j = 0; j < d; j++) z += w[j] * x[j]; return sig(z) >= 0.5 ? 1 : 0; };
    const accuracy = data.Xte.filter((x, i) => pred(x) === data.yte[i]).length / data.Xte.length;
    const acct = R.dpEpsilon({ q: o.batch / n, sigma: o.sigma, steps, delta: o.delta });
    trace.push(T.step('result', 'After ' + steps + ' steps the test accuracy is ' + fmt(accuracy) + '. ' + acct.say, { accuracy, eps: acct.eps }));
    return { accuracy, eps: acct.eps, accountant: acct, history, trace, w: w.slice(), b, steps };
  };

  R.dpTradeoff = (o = {}) => {
    const T = S.T, sigmas = o.sigmas || [0, 1, 2, 4, 8, 16], seeds = o.seeds || [1, 2, 3], data = R.loansData(o.dataSeed || 1);
    const rows = sigmas.map((sigma) => {
      const runs = seeds.map((seed) => R.dpsgd(Object.assign({}, o, { sigma, seed, data })));
      const accuracy = runs.reduce((s, r) => s + r.accuracy, 0) / runs.length;
      return { sigma, eps: runs[0].eps, accuracy, say: 'σ = ' + fmt(sigma) + ': ε ≈ ' + (isFinite(runs[0].eps) ? fmt(runs[0].eps) : '∞') + ', mean test accuracy ' + fmt(accuracy) + '.' };
    });
    const base = rows[0].accuracy;
    return { rows, approximate: true, trace: rows.map((r) => T.step('point', r.say, r)).concat([T.step('tradeoff', 'More noise buys a smaller ε (stronger privacy) at the cost of accuracy: from ' + fmt(base) + ' without noise to ' + fmt(rows[rows.length - 1].accuracy) + ' at σ = ' + fmt(rows[rows.length - 1].sigma) + '.', {})]) };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
