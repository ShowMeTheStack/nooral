// Small models for the sample scenes (S.Toy): logistic regression on polynomial features trained by gradient
// descent, a dense network with backpropagation and Adam, and a kernel SVM trained by SMO. Each keeps snapshots
// a scene can draw.
(function (S) {
  'use strict';
  const T = (S.Toy = S.Toy || {});
  const sig = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  T.sigmoid = sig;

  // ---- logistic regression ------------------------------------------------------------------------------------
  /** the features of a point: every monomial x^a y^b with a + b ≤ degree (the constant is the bias) */
  T.poly = (p, deg) => { const out = []; for (let d = 1; d <= deg; d++) for (let a = d; a >= 0; a--) out.push(Math.pow(p[0], a) * Math.pow(p[1], d - a)); return out; };
  T.polyNames = (deg) => { const out = [], m = (v, k) => (k === 0 ? '' : k === 1 ? v : v + (k === 2 ? '²' : '³')); for (let d = 1; d <= deg; d++) for (let a = d; a >= 0; a--) out.push(m('x', a) + m('y', d - a)); return out; };
  /**
   * Full-batch gradient descent on the mean log-loss (with an L2 penalty lam): returns {snaps: [{epoch, w, b,
   * loss, acc}], predict(w, b, p)}. Features are standardised with the training set's means and spreads.
   */
  T.logreg = function (X, y, o) {
    o = o || {};
    const deg = o.degree || 1, lr = o.lr || 0.5, epochs = o.epochs || 300, lam = o.lam || 0, keep = o.keep || [0, 1, 2, 5, 10, 20, 40, 80, 150, 300];
    const F = X.map((p) => T.poly(p, deg)), m = F[0].length, n = F.length;
    const mu = new Array(m).fill(0), sd = new Array(m).fill(0);
    for (const f of F) f.forEach((v, k) => { mu[k] += v / n; });
    for (const f of F) f.forEach((v, k) => { sd[k] += (v - mu[k]) ** 2 / n; });
    for (let k = 0; k < m; k++) sd[k] = Math.sqrt(sd[k]) || 1;
    const Z = F.map((f) => f.map((v, k) => (v - mu[k]) / sd[k]));
    const w = new Array(m).fill(0);
    let b = 0;
    const feat = (p) => T.poly(p, deg).map((v, k) => (v - mu[k]) / sd[k]);
    const prob = (ww, bb, z) => sig(z.reduce((a, v, k) => a + v * ww[k], bb));
    const predict = (ww, bb, p) => prob(ww, bb, feat(p));
    const snaps = [], lossOf = () => { let L = 0, ok = 0; Z.forEach((z, i) => { const p = prob(w, b, z); L -= y[i] ? Math.log(Math.max(1e-12, p)) : Math.log(Math.max(1e-12, 1 - p)); if ((p >= 0.5 ? 1 : 0) === y[i]) ok++; }); return { loss: L / n + (lam / 2) * w.reduce((a, v) => a + v * v, 0), acc: ok / n }; };
    for (let e = 0; e <= epochs; e++) {
      if (keep.includes(e)) snaps.push(Object.assign({ epoch: e, w: w.slice(), b }, lossOf()));
      if (e === epochs) break;
      const gw = new Array(m).fill(0);
      let gb = 0;
      Z.forEach((z, i) => { const d = prob(w, b, z) - y[i]; gb += d / n; z.forEach((v, k) => { gw[k] += (d * v) / n; }); });
      for (let k = 0; k < m; k++) w[k] -= lr * (gw[k] + lam * w[k]);
      b -= lr * gb;
    }
    const curve = [];
    return { snaps, predict, feat, names: T.polyNames(deg), m, curve };
  };
  /** the loss after every epoch, for a curve (runs the same descent again, cheaply) */
  T.logregCurve = function (X, y, o) {
    const all = [];
    for (let e = 0; e <= (o.epochs || 300); e++) all.push(e);
    return T.logreg(X, y, Object.assign({}, o, { keep: all })).snaps.map((s) => [s.epoch, s.loss, s.acc]);
  };

  // ---- a dense network ------------------------------------------------------------------------------------------
  /** sizes [in, h1, .., out]; hidden layers use tanh (or relu), the output is left linear for the loss to read */
  T.dense = function (sizes, seed, act) {
    const r = T.rng(seed || 1), W = [], b = [];
    for (let l = 0; l + 1 < sizes.length; l++) {
      const s = Math.sqrt(1 / sizes[l]);
      W.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => r.normal() * s * 1.4)));
      b.push(new Array(sizes[l + 1]).fill(0));
    }
    const f = act === 'relu' ? (z) => Math.max(0, z) : Math.tanh, df = act === 'relu' ? (a) => (a > 0 ? 1 : 0) : (a) => 1 - a * a;
    const net = { sizes, W, b, act: act || 'tanh' };
    /** every layer's activations for one input (the last is the linear output) */
    net.forward = (x) => {
      const A = [x];
      for (let l = 0; l < W.length; l++) {
        const a = A[l], last = l === W.length - 1;
        const o = new Array(W[l].length);
        for (let i = 0; i < o.length; i++) { const row = W[l][i]; let z = b[l][i]; for (let j = 0; j < row.length; j++) z += row[j] * a[j]; o[i] = last ? z : f(z); }
        A.push(o);
      }
      return A;
    };
    /** add the gradients of one example (dOut: the loss's gradient at the linear output) into G */
    net.backward = (A, dOut, G) => {
      let d = dOut;
      for (let l = W.length - 1; l >= 0; l--) {
        const a = A[l];
        for (let i = 0; i < d.length; i++) { const di = d[i]; if (!di) continue; G.b[l][i] += di; const gr = G.W[l][i]; for (let j = 0; j < a.length; j++) gr[j] += di * a[j]; }
        const nd = new Array(a.length).fill(0);
        for (let j = 0; j < a.length; j++) { let s = 0; for (let i = 0; i < d.length; i++) s += W[l][i][j] * d[i]; nd[j] = l ? s * df(a[j]) : s; }
        d = nd;
      }
      // the gradient at the input
      return d;
    };
    net.zeros = () => ({ W: W.map((M) => M.map((row) => row.map(() => 0))), b: b.map((v) => v.map(() => 0)) });
    // Adam, in place
    let m1 = null, m2 = null, t = 0;
    net.step = (G, lr, scale) => {
      if (!m1) { m1 = net.zeros(); m2 = net.zeros(); }
      t++;
      const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t), k = scale || 1;
      for (let l = 0; l < W.length; l++) for (let i = 0; i < W[l].length; i++) {
        const row = W[l][i], g = G.W[l][i], M = m1.W[l][i], V = m2.W[l][i];
        for (let j = 0; j < row.length; j++) { const gg = g[j] * k; M[j] = b1 * M[j] + (1 - b1) * gg; V[j] = b2 * V[j] + (1 - b2) * gg * gg; row[j] -= (lr * (M[j] / c1)) / (Math.sqrt(V[j] / c2) + 1e-8); }
        const gg = G.b[l][i] * k, Mb = m1.b[l], Vb = m2.b[l];
        Mb[i] = b1 * Mb[i] + (1 - b1) * gg; Vb[i] = b2 * Vb[i] + (1 - b2) * gg * gg; b[l][i] -= (lr * (Mb[i] / c1)) / (Math.sqrt(Vb[i] / c2) + 1e-8);
      }
    };
    net.copy = () => ({ W: W.map((M) => M.map((row) => row.slice())), b: b.map((v) => v.slice()) });
    return net;
  };
  /** a binary classifier: [2, .., 1] trained on log-loss by full-batch Adam; snapshots at the epochs asked */
  T.trainMlp = function (X, y, o) {
    o = o || {};
    const net = T.dense(o.sizes || [2, 6, 1], o.seed || 4, o.act), snaps = [], keep = o.keep || [0, 10, 30, 80, 200, 500];
    const epochs = Math.max(...keep);
    for (let e = 0; e <= epochs; e++) {
      const G = net.zeros();
      let L = 0, ok = 0;
      X.forEach((x, i) => { const A = net.forward(x), p = sig(A[A.length - 1][0]); L -= y[i] ? Math.log(Math.max(1e-12, p)) : Math.log(Math.max(1e-12, 1 - p)); if ((p >= 0.5 ? 1 : 0) === y[i]) ok++; net.backward(A, [p - y[i]], G); });
      if (keep.includes(e)) snaps.push(Object.assign({ epoch: e, loss: L / X.length, acc: ok / X.length }, net.copy()));
      net.step(G, o.lr || 0.03, 1 / X.length);
    }
    const run = (P, x) => { const A = [x]; for (let l = 0; l < P.W.length; l++) { const a = A[l], last = l === P.W.length - 1; A.push(P.W[l].map((row, i) => { let z = P.b[l][i]; row.forEach((w, j) => { z += w * a[j]; }); return last ? z : (o.act === 'relu' ? Math.max(0, z) : Math.tanh(z)); })); } return A; };
    return { snaps, run };
  };

  // ---- a kernel SVM by SMO --------------------------------------------------------------------------------------
  T.kernel = (kind, gamma) => (kind === 'linear' ? (a, b) => a[0] * b[0] + a[1] * b[1] : (a, b) => Math.exp(-gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)));
  /**
   * The simplified SMO: pairs of multipliers updated until none moves for `passes` sweeps. Labels are 0/1 and
   * become −1/+1. Returns {alpha, b, f(p) (the decision value), sv: [i..], margin: indices with 0 < α < C}.
   */
  T.svm = function (X, y01, o) {
    o = o || {};
    const C = o.C || 1, tol = 1e-3, K = T.kernel(o.kernel || 'rbf', o.gamma || 1), n = X.length, y = y01.map((v) => (v ? 1 : -1)), r = T.rng(o.seed || 9);
    const G = X.map((a) => X.map((b) => K(a, b))), alpha = new Array(n).fill(0);
    let b = 0, passes = 0, iter = 0;
    const fx = (i) => { let s = b; for (let j = 0; j < n; j++) if (alpha[j]) s += alpha[j] * y[j] * G[j][i]; return s; };
    while (passes < (o.passes || 8) && iter < 400) {
      iter++;
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const Ei = fx(i) - y[i];
        if (!((y[i] * Ei < -tol && alpha[i] < C) || (y[i] * Ei > tol && alpha[i] > 0))) continue;
        let j = r.int(n - 1); if (j >= i) j++;
        const Ej = fx(j) - y[j], ai = alpha[i], aj = alpha[j];
        const L = y[i] !== y[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - C), H = y[i] !== y[j] ? Math.min(C, C + aj - ai) : Math.min(C, ai + aj);
        if (H - L < 1e-12) continue;
        const eta = 2 * G[i][j] - G[i][i] - G[j][j];
        if (eta >= 0) continue;
        let nj = aj - (y[j] * (Ei - Ej)) / eta;
        nj = Math.min(H, Math.max(L, nj));
        if (Math.abs(nj - aj) < 1e-6) continue;
        const ni = ai + y[i] * y[j] * (aj - nj);
        const b1 = b - Ei - y[i] * (ni - ai) * G[i][i] - y[j] * (nj - aj) * G[i][j];
        const b2 = b - Ej - y[i] * (ni - ai) * G[i][j] - y[j] * (nj - aj) * G[j][j];
        alpha[i] = ni; alpha[j] = nj;
        b = ni > 0 && ni < C ? b1 : nj > 0 && nj < C ? b2 : (b1 + b2) / 2;
        changed++;
      }
      passes = changed ? 0 : passes + 1;
    }
    const sv = alpha.map((a, i) => (a > 1e-6 ? i : -1)).filter((i) => i >= 0);
    const f = (p) => { let s = b; for (const j of sv) s += alpha[j] * y[j] * K(X[j], p); return s; };
    let ok = 0;
    X.forEach((p, i) => { if ((f(p) >= 0 ? 1 : 0) === y01[i]) ok++; });
    return { alpha, b, f, sv, margin: sv.filter((i) => alpha[i] < C - 1e-6), acc: ok / n, C, gamma: o.gamma, kernel: o.kernel || 'rbf' };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
