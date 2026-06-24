// Convolution and pooling on [N, C, H, W] images. conv2d unrolls each image's receptive fields into columns
// (im2col) and multiplies by the filter matrix; its backward pass folds the column gradients back (col2im).
//   conv2d(x, w, b?, {stride, padding, dilation, groups})   w: [O, C / groups, kh, kw]; numbers or [h, w] pairs
//   im2col(x, kh, kw, opts) -> [N, C·kh·kw, Ho·Wo]           convOut(H, W, kh, kw, opts) -> [Ho, Wo]
//   maxPool2d(x, k, stride = k, padding = 0)   avgPool2d(x, k, stride = k, padding = 0) (padding counts as zeros)
(function (S) {
  'use strict';
  const T = S.T;
  const F = (t) => t.data.constructor;
  const pair = (v, d) => (v === undefined ? [d, d] : Array.isArray(v) ? v : [v, v]);
  const geom = (opts = {}) => ({ s: pair(opts.stride, 1), p: pair(opts.padding, 0), d: pair(opts.dilation, 1), groups: opts.groups || 1 });
  T.convOut = (H, W, kh, kw, opts) => {
    const g = geom(opts);
    return [Math.floor((H + 2 * g.p[0] - g.d[0] * (kh - 1) - 1) / g.s[0]) + 1, Math.floor((W + 2 * g.p[1] - g.d[1] * (kw - 1) - 1) / g.s[1]) + 1];
  };

  /** columns for channels [c0, c0 + Cg) of image n: rows (c, i, j), columns output positions */
  function im2colInto(v, n, C, H, W, c0, Cg, kh, kw, g, Ho, Wo, cols) {
    const L = Ho * Wo;
    for (let c = 0; c < Cg; c++) {
      const base = (n * C + c0 + c) * H * W;
      for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) {
        const row = ((c * kh + i) * kw + j) * L;
        for (let oy = 0; oy < Ho; oy++) {
          const y = oy * g.s[0] - g.p[0] + i * g.d[0];
          const r = row + oy * Wo;
          if (y < 0 || y >= H) { for (let ox = 0; ox < Wo; ox++) cols[r + ox] = 0; continue; }
          for (let ox = 0; ox < Wo; ox++) {
            const xx = ox * g.s[1] - g.p[1] + j * g.d[1];
            cols[r + ox] = xx < 0 || xx >= W ? 0 : v[base + y * W + xx];
          }
        }
      }
    }
    return cols;
  }
  function col2imAdd(cols, dx, n, C, H, W, c0, Cg, kh, kw, g, Ho, Wo) {
    const L = Ho * Wo;
    for (let c = 0; c < Cg; c++) {
      const base = (n * C + c0 + c) * H * W;
      for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) {
        const row = ((c * kh + i) * kw + j) * L;
        for (let oy = 0; oy < Ho; oy++) {
          const y = oy * g.s[0] - g.p[0] + i * g.d[0];
          if (y < 0 || y >= H) continue;
          for (let ox = 0; ox < Wo; ox++) {
            const xx = ox * g.s[1] - g.p[1] + j * g.d[1];
            if (xx >= 0 && xx < W) dx[base + y * W + xx] += cols[row + oy * Wo + ox];
          }
        }
      }
    }
  }
  T.kernels = T.kernels || {};
  T.kernels.im2col = im2colInto;

  T.def('im2col', (x, kh, kw, opts) => {
    const g = geom(opts), [N, C, H, W] = x.shape, [Ho, Wo] = T.convOut(H, W, kh, kw, opts), K = C * kh * kw, L = Ho * Wo, v = x.values();
    const out = new (F(x))(N * K * L);
    for (let n = 0; n < N; n++) im2colInto(v, n, C, H, W, 0, C, kh, kw, g, Ho, Wo, out.subarray(n * K * L, (n + 1) * K * L));
    return T.record(T.make(out, [N, K, L]), 'im2col', [x], (gr) => {
      const gv = gr.values(), dx = new (F(x))(x.size);
      for (let n = 0; n < N; n++) col2imAdd(gv.subarray(n * K * L, (n + 1) * K * L), dx, n, C, H, W, 0, C, kh, kw, g, Ho, Wo);
      return [T.make(dx, x.shape)];
    }, { kh, kw, Ho, Wo });
  });

  T.def('conv2d', (x, w, b, opts = {}) => {
    const g = geom(opts), [N, C, H, W] = x.shape, [O, Cg, kh, kw] = w.shape, G = g.groups;
    if (Cg * G !== C) throw new Error('conv2d: ' + C + ' input channels, but the filters expect ' + Cg + ' × ' + G + ' groups');
    if (O % G) throw new Error('conv2d: ' + O + ' filters do not split into ' + G + ' groups');
    const [Ho, Wo] = T.convOut(H, W, kh, kw, opts), L = Ho * Wo, K = Cg * kh * kw, Og = O / G;
    if (Ho <= 0 || Wo <= 0) throw new Error('conv2d: the output would be empty');
    const Cf = x.data instanceof Float64Array || w.data instanceof Float64Array ? Float64Array : F(x);
    const v = x.values(), wv = w.values(), bv = b ? b.values() : null, out = new Cf(N * O * L), cols = new Cf(K * L);
    for (let n = 0; n < N; n++) for (let gi = 0; gi < G; gi++) {
      im2colInto(v, n, C, H, W, gi * Cg, Cg, kh, kw, g, Ho, Wo, cols);
      T.kernels.gemm(wv, cols, Og, K, L, out, gi * Og * K, 0, (n * O + gi * Og) * L);
    }
    if (bv) for (let n = 0; n < N; n++) for (let o = 0; o < O; o++) { const base = (n * O + o) * L; for (let l = 0; l < L; l++) out[base + l] += bv[o]; }
    return T.record(T.make(out, [N, O, Ho, Wo]), 'conv2d', [x, w, b || null], (gr) => {
      const gv = gr.values(), dx = x.requiresGrad ? new Cf(x.size) : null, dw = w.requiresGrad ? new Cf(w.size) : null;
      const db = b && b.requiresGrad ? new Cf(O) : null;
      const gemm = T.kernels.gemm, part = dw ? new Cf(Og * K) : null, dcols = dx ? new Cf(K * L) : null;
      for (let n = 0; n < N; n++) for (let gi = 0; gi < G; gi++) {
        const go = gv.subarray((n * O + gi * Og) * L, (n * O + (gi + 1) * Og) * L);
        if (dw) {
          im2colInto(v, n, C, H, W, gi * Cg, Cg, kh, kw, g, Ho, Wo, cols);
          gemm(go, cols, Og, L, K, part, 0, 0, 0, false, true);
          const base = gi * Og * K;
          for (let i = 0; i < part.length; i++) dw[base + i] += part[i];
        }
        if (dx) {
          gemm(wv, go, K, Og, L, dcols, gi * Og * K, 0, 0, true, false);
          col2imAdd(dcols, dx, n, C, H, W, gi * Cg, Cg, kh, kw, g, Ho, Wo);
        }
      }
      if (db) for (let n = 0; n < N; n++) for (let o = 0; o < O; o++) { const base = (n * O + o) * L; for (let l = 0; l < L; l++) db[o] += gv[base + l]; }
      return [dx && T.make(dx, x.shape), dw && T.make(dw, w.shape), db && T.make(db, b.shape)];
    }, { stride: g.s, padding: g.p, dilation: g.d, groups: G, Ho, Wo });
  });

  function pool(kind) {
    return (x, k, stride, padding = 0) => {
      const kk = pair(k), s = pair(stride === undefined ? k : stride), p = pair(padding), [N, C, H, W] = x.shape;
      const Ho = Math.floor((H + 2 * p[0] - kk[0]) / s[0]) + 1, Wo = Math.floor((W + 2 * p[1] - kk[1]) / s[1]) + 1;
      const v = x.values(), out = new (F(x))(N * C * Ho * Wo), arg = kind === 'max' ? new Int32Array(out.length) : null, area = kk[0] * kk[1];
      for (let nc = 0; nc < N * C; nc++) {
        const base = nc * H * W;
        for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
          let best = -Infinity, bi = -1, sum = 0;
          for (let i = 0; i < kk[0]; i++) {
            const y = oy * s[0] - p[0] + i;
            if (y < 0 || y >= H) continue;
            for (let j = 0; j < kk[1]; j++) {
              const xx = ox * s[1] - p[1] + j;
              if (xx < 0 || xx >= W) continue;
              const z = v[base + y * W + xx];
              sum += z;
              if (z > best) { best = z; bi = base + y * W + xx; }
            }
          }
          const o = (nc * Ho + oy) * Wo + ox;
          if (arg) { out[o] = best; arg[o] = bi; } else out[o] = sum / area;
        }
      }
      return T.record(T.make(out, [N, C, Ho, Wo]), kind + 'Pool2d', [x], (gr) => {
        const gv = gr.values(), dx = new (F(x))(x.size);
        if (arg) { for (let o = 0; o < gv.length; o++) if (arg[o] >= 0) dx[arg[o]] += gv[o]; }
        else for (let nc = 0; nc < N * C; nc++) for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
          const gg = gv[(nc * Ho + oy) * Wo + ox] / area;
          for (let i = 0; i < kk[0]; i++) {
            const y = oy * s[0] - p[0] + i;
            if (y < 0 || y >= H) continue;
            for (let j = 0; j < kk[1]; j++) { const xx = ox * s[1] - p[1] + j; if (xx >= 0 && xx < W) dx[nc * H * W + y * W + xx] += gg; }
          }
        }
        return [T.make(dx, x.shape)];
      }, { k: kk, stride: s, padding: p, argmax: arg });
    };
  }
  T.def('maxPool2d', pool('max'));
  T.def('avgPool2d', pool('avg'));
})(globalThis.Nooral = globalThis.Nooral || {});
