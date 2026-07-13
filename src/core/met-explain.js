// Model explanations that work for any model: permutation importance, partial dependence with ICE curves, a
// LIME-style local linear surrogate, and gradient pictures (saliency, Grad-CAM) for networks built on S.T.
// A model is a function from a list of rows to a list of numbers, or an object with predict(rows).
(function (S) {
  'use strict';
  const E = (S.Explain = S.Explain || {});
  const M = () => S.Metrics;
  const f = (x) => M().f(x, 4), st = (k, s, d) => M().step(k, s, d);
  const asModel = (m) => (typeof m === 'function' ? m : (X) => m.predict(X));
  const nameOf = (opt, j) => (opt.names ? opt.names[j] : 'x' + j);

  /** the default score: accuracy when the predictions are labels, R² when they are numbers with fractions */
  const defaultScore = (y, p) => (p.every((v) => typeof v !== 'number' || Number.isInteger(v)) && y.every((v) => typeof v !== 'number' || Number.isInteger(v)) ? M().accuracy(y, p).value : M().r2(y, p).value);

  /**
   * permutationImportance(model, X, y, {nRepeats: 5, seed: 1, score(y, pred), names})
   * → { baseline, mean, std, drops, trace }: how much the score falls when one column is shuffled
   */
  E.permutationImportance = (model, X, y, opt = {}) => {
    model = asModel(model);
    const score = opt.score || defaultScore, reps = opt.nRepeats || 5, rnd = S.Data.rng(opt.seed || 1), d = X[0].length;
    const baseline = score(y, model(X)), drops = [];
    for (let j = 0; j < d; j++) {
      const dj = [];
      for (let r = 0; r < reps; r++) {
        const col = X.map((x) => x[j]);
        for (let i = col.length - 1; i > 0; i--) { const k = Math.floor(rnd() * (i + 1)); [col[i], col[k]] = [col[k], col[i]]; }
        dj.push(baseline - score(y, model(X.map((x, i) => { const z = x.slice(); z[j] = col[i]; return z; }))));
      }
      drops.push(dj);
    }
    const mean = drops.map((a) => a.reduce((s, v) => s + v, 0) / a.length);
    const std = drops.map((a, j) => Math.sqrt(a.reduce((s, v) => s + (v - mean[j]) ** 2, 0) / a.length));
    const order = mean.map((_, j) => j).sort((a, b) => mean[b] - mean[a]);
    return { baseline, mean, std, drops, order, trace: [
      st('baseline', 'With the data intact the model scores ' + f(baseline) + '.', { baseline }),
      ...order.map((j) => st('feature', 'Shuffling ' + nameOf(opt, j) + ' ' + reps + ' times drops the score by ' + f(mean[j]) + ' on average (spread ' + f(std[j]) + ')' + (Math.abs(mean[j]) < 1e-12 ? ': the model does not use it.' : '.'), { feature: j, mean: mean[j], std: std[j] }))] };
  };

  // the default grid: the distinct values when there are few, else an even grid between the 5th and 95th
  // percentiles (the plotting-position quantile with α = β = 0.4)
  const quantile = (sorted, p) => {
    const n = sorted.length, m = 0.4 + p * 0.2, a = n * p + m, k = Math.floor(Math.min(Math.max(a, 1), n - 1)), g = Math.min(Math.max(a - k, 0), 1);
    return (1 - g) * sorted[k - 1] + g * sorted[k];
  };
  E.grid = (values, size = 20, pct = [0.05, 0.95]) => {
    const sorted = values.slice().sort((a, b) => a - b), uniq = sorted.filter((v, i) => i === 0 || v !== sorted[i - 1]);
    if (uniq.length < size) return uniq;
    const lo = quantile(sorted, pct[0]), hi = quantile(sorted, pct[1]), step = (hi - lo) / (size - 1);
    return Array.from({ length: size }, (_, i) => (i === size - 1 ? hi : lo + i * step));
  };

  /**
   * partialDependence(model, X, feature, {grid, gridSize: 20, percentiles, names})
   * → { grid, average, ice, trace }: ice[i][g] is example i's prediction with the feature set to grid[g]
   */
  E.partialDependence = (model, X, feature, opt = {}) => {
    model = asModel(model);
    const grid = opt.grid || E.grid(X.map((x) => x[feature]), opt.gridSize || 20, opt.percentiles);
    const rows = []; for (const x of X) for (const g of grid) { const z = x.slice(); z[feature] = g; rows.push(z); }
    const out = model(rows), ice = X.map((_, i) => grid.map((_, g) => out[i * grid.length + g]));
    const average = grid.map((_, g) => ice.reduce((s, r) => s + r[g], 0) / X.length);
    const lo = average.indexOf(Math.min(...average)), hi = average.indexOf(Math.max(...average));
    const spread = Math.max(...grid.map((_, g) => Math.max(...ice.map((r) => r[g])) - Math.min(...ice.map((r) => r[g]))));
    return { grid, average, ice, feature, trace: [
      st('grid', 'Set ' + nameOf(opt, feature) + ' to each of ' + grid.length + ' values from ' + f(grid[0]) + ' to ' + f(grid[grid.length - 1]) + ' in every one of the ' + X.length + ' examples.', { grid }),
      st('ice', 'Each example traces its own curve (ICE); at one grid value the curves are up to ' + f(spread) + ' apart, a sign of interactions when large.', { spread }),
      st('average', 'Averaging the curves gives partial dependence: lowest ' + f(average[lo]) + ' at ' + f(grid[lo]) + ', highest ' + f(average[hi]) + ' at ' + f(grid[hi]) + '.', { average })] };
  };

  /**
   * lime(model, x, X, {nSamples: 500, seed: 1, kernelWidth, ridge: 1e-3, names})
   * → { coef, intercept, score, samples, weights, trace }: a weighted linear fit around x. Samples are drawn
   * from a normal around x with each feature's spread in X; weights fall off with the scaled distance to x.
   */
  E.lime = (model, x, X, opt = {}) => {
    model = asModel(model);
    const d = x.length, n = opt.nSamples || 500, rnd = S.Data.rng(opt.seed || 1), ridge = opt.ridge == null ? 1e-3 : opt.ridge;
    const sd = Array.from({ length: d }, (_, j) => { const c = X.map((r) => r[j]), m = c.reduce((a, b) => a + b, 0) / c.length; return Math.sqrt(c.reduce((a, b) => a + (b - m) ** 2, 0) / c.length) || 1; });
    const width = opt.kernelWidth || 0.75 * Math.sqrt(d);
    const gauss = () => { let u = 0; while (!u) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); };
    const Z = [x.slice()]; for (let i = 1; i < n; i++) Z.push(x.map((v, j) => v + sd[j] * gauss()));
    const y = model(Z);
    const w = Z.map((z) => Math.exp(-z.reduce((s, v, j) => s + ((v - x[j]) / sd[j]) ** 2, 0) / (width * width)));
    // weighted ridge on the scaled offsets (intercept not penalised)
    const p = d + 1, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
    Z.forEach((z, k) => { const r = [1, ...z.map((v, j) => (v - x[j]) / sd[j])]; for (let i = 0; i < p; i++) { b[i] += w[k] * r[i] * y[k]; for (let j = 0; j < p; j++) A[i][j] += w[k] * r[i] * r[j]; } });
    for (let i = 1; i < p; i++) A[i][i] += ridge;
    const beta = solve(A, b), coef = beta.slice(1).map((c, j) => c / sd[j]), intercept = beta[0];
    const fit = Z.map((z) => intercept + z.reduce((s, v, j) => s + coef[j] * (v - x[j]), 0));
    const W = w.reduce((a, c) => a + c, 0), ym = y.reduce((s, v, k) => s + w[k] * v, 0) / W;
    const ssr = y.reduce((s, v, k) => s + w[k] * (v - fit[k]) ** 2, 0), sst = y.reduce((s, v, k) => s + w[k] * (v - ym) ** 2, 0);
    const score = sst === 0 ? 1 : 1 - ssr / sst;
    const order = coef.map((_, j) => j).sort((a, c) => Math.abs(coef[c] * sd[c]) - Math.abs(coef[a] * sd[a]));
    return { coef, intercept, score, samples: Z, weights: w, predictions: y, trace: [
      st('sample', 'Drew ' + n + ' points around the example and asked the model about each; nearer points weigh more (kernel width ' + f(width) + ').', { n, width }),
      st('fit', 'A weighted straight-line fit matches the model near the example with R² = ' + f(score) + '; the model gives ' + f(y[0]) + ' here and the line gives ' + f(intercept) + '.', { score }),
      ...order.map((j) => st('feature', 'Near this example, raising ' + nameOf(opt, j) + ' by one unit changes the output by ' + f(coef[j]) + '.', { feature: j, coef: coef[j] }))] };
  };
  function solve(A, b) {
    const n = b.length; A = A.map((r) => r.slice()); b = b.slice();
    for (let c = 0; c < n; c++) {
      let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
      [A[c], A[piv]] = [A[piv], A[c]]; [b[c], b[piv]] = [b[piv], b[c]];
      if (Math.abs(A[c][c]) < 1e-300) continue;
      for (let r = c + 1; r < n; r++) { const q = A[r][c] / A[c][c]; for (let k = c; k < n; k++) A[r][k] -= q * A[c][k]; b[r] -= q * b[c]; }
    }
    const out = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= A[i][k] * out[k]; out[i] = Math.abs(A[i][i]) < 1e-300 ? 0 : s / A[i][i]; }
    return out;
  }
  E.solve = solve;

  /**
   * saliency(fn, x, {eps}) → { grad, magnitude, trace }: the gradient of a scalar function of a flat input.
   * fn: plain function of a number list, differentiated by central differences. For a network on S.T use
   * saliencyT, which takes the exact gradient by backpropagation.
   */
  E.saliency = (fn, x, opt = {}) => {
    const h = opt.eps || 1e-4, grad = Array.from(x, (_, i) => { const a = Array.from(x), b = Array.from(x); a[i] += h; b[i] -= h; return (fn(a) - fn(b)) / (2 * h); });
    return finishSaliency(grad, 'central differences');
  };
  const finishSaliency = (grad, how) => {
    const mag = grad.map(Math.abs), max = Math.max(...mag) || 1, top = mag.indexOf(Math.max(...mag));
    return { grad, magnitude: mag.map((v) => v / max), trace: [st('saliency', 'The gradient of the score with respect to each input (by ' + how + ') is largest at input ' + top + ', ' + f(grad[top]) + ': changing that input moves the score most.', { top })] };
  };

  /**
   * saliencyT(net, x, shape, cls): the exact input gradient through S.T autodiff. net maps an input tensor to an
   * output tensor of scores; cls picks the score to explain (the output is summed when cls is omitted).
   */
  E.saliencyT = (net, x, shape, cls) => {
    const T = S.T;
    if (!T || !T.param) throw new Error('saliencyT needs the tensor engine S.T');
    const input = T.param(T.tensor(Array.from(x), shape));
    let out = net(input);
    if (cls != null) { const mask = new Array(out.size).fill(0); mask[cls] = 1; out = T.mul(out, T.tensor(mask, out.shape)); }
    T.sum(out).backward();
    return finishSaliency(Array.from(input.grad.values()), 'backpropagation');
  };

  /**
   * gradCam(A, G) for one image: A[c][h][w] are a convolution layer's activations, G the gradients of the class
   * score with respect to them. Channel weight αc = mean of Gc; map = ReLU(Σ αc Ac), also given scaled to [0, 1].
   */
  E.gradCam = (A, G) => {
    const C = A.length, H = A[0].length, W = A[0][0].length;
    const weights = G.map((g) => g.flat().reduce((s, v) => s + v, 0) / (H * W));
    const raw = Array.from({ length: H }, (_, i) => Array.from({ length: W }, (_, j) => Math.max(0, weights.reduce((s, a, c) => s + a * A[c][i][j], 0))));
    const max = Math.max(...raw.flat()) || 1, map = raw.map((r) => r.map((v) => v / max));
    const top = weights.map((_, c) => c).sort((a, b) => weights[b] - weights[a]);
    return { weights, raw, map, trace: [
      st('weights', 'Each of the ' + C + ' channels is weighted by its average gradient; channel ' + top[0] + ' matters most (' + f(weights[top[0]]) + ').', { weights }),
      st('map', 'The weighted sum of the channel maps, with negatives cut to 0, marks where the evidence for the class lies; it peaks at ' + f(max) + '.', { max })] };
  };

  /**
   * gradCamT(net, x, shape, cls, layer): Grad-CAM through S.T. net(input, keep) must call keep(t) on the chosen
   * convolution output (shape [1, C, H, W]) and return the class scores.
   * TODO: once S.T modules expose named intermediate outputs, take the layer by name instead of the keep callback.
   */
  E.gradCamT = (net, x, shape, cls) => {
    const T = S.T;
    if (!T || !T.param) throw new Error('gradCamT needs the tensor engine S.T');
    let act = null;
    const input = T.tensor(Array.from(x), shape);
    const out = net(input, (t) => { t.requiresGrad = true; act = t; return t; });
    if (!act) throw new Error('the network did not hand over a layer through keep()');
    const mask = new Array(out.size).fill(0); mask[cls == null ? 0 : cls] = 1;
    T.sum(T.mul(out, T.tensor(mask, out.shape))).backward();
    const [, C, H, W] = act.shape, a = act.values(), g = act.grad.values();
    const grid = (v) => Array.from({ length: C }, (_, c) => Array.from({ length: H }, (_, i) => Array.from({ length: W }, (_, j) => v[(c * H + i) * W + j])));
    return E.gradCam(grid(a), grid(g));
  };
})(globalThis.Nooral = globalThis.Nooral || {});
