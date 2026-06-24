// Normalisation and dropout, with hand-derived backward passes.
//   layerNorm(x, gamma?, beta?, eps = 1e-5)      over the last axis; info keeps each row's mean and 1/std
//   batchNorm(x, gamma?, beta?, opts)            x [N, C] or [N, C, H, W]; statistics per channel C
//      opts: {training: true, running: {mean, var} (tensors, updated in place in training), momentum: 0.1, eps: 1e-5}
//   dropout(x, p, rng, training = true)          inverted dropout: kept units are scaled by 1 / (1 − p); info.mask
(function (S) {
  'use strict';
  const T = S.T;
  const F = (t) => t.data.constructor;

  T.def('layerNorm', (x, gamma, beta, eps = 1e-5) => {
    const D = x.shape[x.ndim - 1], rows = x.size / D, v = x.values(), C = F(x);
    const y = new C(x.size), xhat = new C(x.size), mean = new C(rows), rstd = new C(rows);
    const gv = gamma ? gamma.values() : null, bv = beta ? beta.values() : null;
    for (let r = 0; r < rows; r++) {
      const o = r * D; let m = 0;
      for (let i = 0; i < D; i++) m += v[o + i];
      m /= D;
      let s = 0;
      for (let i = 0; i < D; i++) { const d = v[o + i] - m; s += d * d; }
      const rs = 1 / Math.sqrt(s / D + eps);
      mean[r] = m; rstd[r] = rs;
      for (let i = 0; i < D; i++) { const h = (v[o + i] - m) * rs; xhat[o + i] = h; y[o + i] = gv ? h * gv[i] + (bv ? bv[i] : 0) : h + (bv ? bv[i] : 0); }
    }
    return T.record(T.make(y, x.shape), 'layerNorm', [x, gamma || null, beta || null], (g) => {
      const G = g.values(), dx = new C(x.size), dg = new C(D), db = new C(D);
      for (let r = 0; r < rows; r++) {
        const o = r * D; let s1 = 0, s2 = 0;
        for (let i = 0; i < D; i++) {
          const dh = G[o + i] * (gv ? gv[i] : 1);
          s1 += dh; s2 += dh * xhat[o + i];
          dg[i] += G[o + i] * xhat[o + i]; db[i] += G[o + i];
        }
        for (let i = 0; i < D; i++) { const dh = G[o + i] * (gv ? gv[i] : 1); dx[o + i] = (rstd[r] / D) * (D * dh - s1 - xhat[o + i] * s2); }
      }
      return [T.make(dx, x.shape), gamma ? T.make(dg, gamma.shape) : null, beta ? T.make(db, beta.shape) : null];
    }, { mean, rstd, eps });
  });

  T.def('batchNorm', (x, gamma, beta, opts = {}) => {
    const training = opts.training !== false, eps = opts.eps === undefined ? 1e-5 : opts.eps, mom = opts.momentum === undefined ? 0.1 : opts.momentum;
    const N = x.shape[0], Ch = x.shape[1], inner = T.prod(x.shape.slice(2)), m = N * inner, v = x.values(), C = F(x);
    const mean = new C(Ch), varr = new C(Ch);
    if (training) {
      for (let n = 0; n < N; n++) for (let c = 0; c < Ch; c++) { const o = (n * Ch + c) * inner; for (let i = 0; i < inner; i++) mean[c] += v[o + i]; }
      for (let c = 0; c < Ch; c++) mean[c] /= m;
      for (let n = 0; n < N; n++) for (let c = 0; c < Ch; c++) { const o = (n * Ch + c) * inner; for (let i = 0; i < inner; i++) { const d = v[o + i] - mean[c]; varr[c] += d * d; } }
      for (let c = 0; c < Ch; c++) varr[c] /= m;
      if (opts.running) {
        const rm = opts.running.mean.data, rv = opts.running.var.data;
        for (let c = 0; c < Ch; c++) { rm[c] = (1 - mom) * rm[c] + mom * mean[c]; rv[c] = (1 - mom) * rv[c] + mom * (m > 1 ? (varr[c] * m) / (m - 1) : varr[c]); }
      }
    } else {
      if (!opts.running) throw new Error('batchNorm in eval mode needs running statistics');
      mean.set(opts.running.mean.values()); varr.set(opts.running.var.values());
    }
    const rstd = new C(Ch);
    for (let c = 0; c < Ch; c++) rstd[c] = 1 / Math.sqrt(varr[c] + eps);
    const gv = gamma ? gamma.values() : null, bv = beta ? beta.values() : null, xhat = new C(x.size), y = new C(x.size);
    for (let n = 0; n < N; n++) for (let c = 0; c < Ch; c++) {
      const o = (n * Ch + c) * inner, ga = gv ? gv[c] : 1, be = bv ? bv[c] : 0;
      for (let i = 0; i < inner; i++) { const h = (v[o + i] - mean[c]) * rstd[c]; xhat[o + i] = h; y[o + i] = h * ga + be; }
    }
    return T.record(T.make(y, x.shape), 'batchNorm', [x, gamma || null, beta || null], (g) => {
      const G = g.values(), dx = new C(x.size), dg = new C(Ch), db = new C(Ch), s1 = new C(Ch), s2 = new C(Ch);
      for (let n = 0; n < N; n++) for (let c = 0; c < Ch; c++) {
        const o = (n * Ch + c) * inner, ga = gv ? gv[c] : 1;
        for (let i = 0; i < inner; i++) { const gg = G[o + i]; dg[c] += gg * xhat[o + i]; db[c] += gg; s1[c] += gg * ga; s2[c] += gg * ga * xhat[o + i]; }
      }
      for (let n = 0; n < N; n++) for (let c = 0; c < Ch; c++) {
        const o = (n * Ch + c) * inner, ga = gv ? gv[c] : 1;
        for (let i = 0; i < inner; i++) {
          const dh = G[o + i] * ga;
          dx[o + i] = training ? (rstd[c] / m) * (m * dh - s1[c] - xhat[o + i] * s2[c]) : dh * rstd[c];
        }
      }
      return [T.make(dx, x.shape), gamma ? T.make(dg, gamma.shape) : null, beta ? T.make(db, beta.shape) : null];
    }, { mean, var: varr, training });
  });

  T.def('dropout', (x, p = 0.5, rng, training = true) => {
    if (!training || p <= 0) return x;
    const g = T.asRng(rng), n = x.size, mask = new (F(x))(n), v = x.values(), out = new (F(x))(n), k = 1 / (1 - p);
    for (let i = 0; i < n; i++) { mask[i] = g.next() >= p ? k : 0; out[i] = v[i] * mask[i]; }
    return T.record(T.make(out, x.shape), 'dropout', [x], (gr) => {
      const gv = gr.values(), r = new (F(gr))(n);
      for (let i = 0; i < n; i++) r[i] = gv[i] * mask[i];
      return [T.make(r, x.shape)];
    }, { p, mask });
  });
})(globalThis.Nooral = globalThis.Nooral || {});
