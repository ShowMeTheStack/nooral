// Reductions over axes, and softmax / log-softmax (numerically stable: the row maximum is subtracted first).
//   sum(x, axes?, keepdims?)  mean  max  min  variance(x, axes, keepdims, unbiased?)  logsumexp(x, axis, keepdims)
//   argmax(x, axis, keepdims?) / argmin   -> indices as a float tensor (no gradient)
//   softmax(x, axis = -1)  logSoftmax(x, axis = -1)
// axes: a number, an array, or null/undefined for all of them.
(function (S) {
  'use strict';
  const T = S.T;
  const F = (t) => t.data.constructor;
  /** outer × n × inner around one axis */
  const split = (shape, axis) => ({ outer: T.prod(shape.slice(0, axis)), n: shape[axis], inner: T.prod(shape.slice(axis + 1)) });
  T.splitAxis = split;
  const normAxes = (axes, nd) => (axes === null || axes === undefined ? [...Array(nd).keys()] : (Array.isArray(axes) ? axes : [axes]).map((a) => T.normAxis(a, nd))).sort((a, b) => a - b);

  /** reduce one axis (kept as size 1): kind 'sum' | 'max' | 'min' */
  function reduceAxis(v, shape, axis, kind, C) {
    const { outer, n, inner } = split(shape, axis), out = new C(outer * inner);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < inner; i++) {
        let base = o * n * inner + i, acc = v[base];
        if (kind === 'sum') { for (let k = 1; k < n; k++) acc += v[base + k * inner]; }
        else if (kind === 'max') { for (let k = 1; k < n; k++) { const z = v[base + k * inner]; if (z > acc) acc = z; } }
        else { for (let k = 1; k < n; k++) { const z = v[base + k * inner]; if (z < acc) acc = z; } }
        out[o * inner + i] = n ? acc : 0;
      }
    }
    const s = shape.slice(); s[axis] = 1;
    return { v: out, shape: s };
  }
  function reduce(x, axes, kind) {
    let v = x.values(), shape = x.shape.slice();
    for (const a of axes) { const r = reduceAxis(v, shape, a, kind, F(x)); v = r.v; shape = r.shape; }
    return { v, shape };
  }
  const finalShape = (shape, axes, keepdims) => (keepdims ? shape : shape.filter((_, i) => !axes.includes(i)));

  T.def('sum', (x, axes, keepdims = false) => {
    x = T.as(x);
    const ax = normAxes(axes, x.ndim), r = reduce(x, ax, 'sum');
    const out = T.make(r.v, finalShape(r.shape, ax, keepdims));
    return T.record(out, 'sum', [x], (g) => [T.make(T.expandValues(T.make(g.values(), r.shape), x.shape), x.shape)], { axes: ax });
  });
  T.def('mean', (x, axes, keepdims = false) => {
    x = T.as(x);
    const ax = normAxes(axes, x.ndim), cnt = ax.reduce((p, a) => p * x.shape[a], 1);
    const r = reduce(x, ax, 'sum');
    for (let i = 0; i < r.v.length; i++) r.v[i] /= cnt;
    const out = T.make(r.v, finalShape(r.shape, ax, keepdims));
    return T.record(out, 'mean', [x], (g) => {
      const e = T.expandValues(T.make(g.values(), r.shape), x.shape), o = new (F(g))(e.length);
      for (let i = 0; i < e.length; i++) o[i] = e[i] / cnt;
      return [T.make(o, x.shape)];
    }, { axes: ax });
  });
  const extreme = (kind) => (x, axes, keepdims = false) => {
    x = T.as(x);
    const ax = normAxes(axes, x.ndim), r = reduce(x, ax, kind);
    const out = T.make(r.v, finalShape(r.shape, ax, keepdims));
    return T.record(out, kind, [x], (g) => {
      // the gradient goes to the extreme element; ties share it equally
      const v = x.values(), m = T.expandValues(T.make(r.v, r.shape), x.shape), mask = new (F(x))(v.length);
      for (let i = 0; i < v.length; i++) mask[i] = v[i] === m[i] ? 1 : 0;
      const cnt = reduce(T.make(mask, x.shape), ax, 'sum'), ce = T.expandValues(T.make(cnt.v, cnt.shape), x.shape);
      const ge = T.expandValues(T.make(g.values(), r.shape), x.shape), o = new (F(x))(v.length);
      for (let i = 0; i < v.length; i++) o[i] = mask[i] ? ge[i] / ce[i] : 0;
      return [T.make(o, x.shape)];
    }, { axes: ax });
  };
  T.def('max', extreme('max'));
  T.def('min', extreme('min'));
  const argext = (better) => (x, axis = -1, keepdims = false) => {
    x = T.as(x);
    const a = T.normAxis(axis, x.ndim), { outer, n, inner } = split(x.shape, a), v = x.values(), out = new (F(x))(outer * inner);
    for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
      const base = o * n * inner + i; let bk = 0, bv = v[base];
      for (let k = 1; k < n; k++) { const z = v[base + k * inner]; if (better(z, bv)) { bv = z; bk = k; } }
      out[o * inner + i] = bk;
    }
    const s = x.shape.slice(); if (keepdims) s[a] = 1; else s.splice(a, 1);
    return T.make(out, s);
  };
  T.def('argmax', argext((z, b) => z > b));
  T.def('argmin', argext((z, b) => z < b));
  T.def('variance', (x, axes, keepdims = false, unbiased = false) => {
    const ax = normAxes(axes, x.ndim), cnt = ax.reduce((p, a) => p * x.shape[a], 1);
    const d = T.sub(x, T.mean(x, ax, true)), s = T.sum(T.mul(d, d), ax, keepdims);
    return T.scale(s, 1 / (unbiased ? cnt - 1 : cnt));
  });

  /** softmax along one axis; info keeps the axis for the pictures */
  let row = new Float64Array(64);
  function softmaxKernel(v, shape, axis, C, log) {
    const { outer, n, inner } = split(shape, axis), out = new C(v.length);
    if (inner === 1 && !log) {
      // rows along the last axis: one exp per element, kept in float64 (the same numbers as computing it twice)
      if (row.length < n) row = new Float64Array(n);
      for (let o = 0, base = 0; o < outer; o++, base += n) {
        let m = -Infinity;
        for (let k = 0; k < n; k++) if (v[base + k] > m) m = v[base + k];
        if (m === -Infinity) m = 0;
        let s = 0;
        for (let k = 0; k < n; k++) { const e = Math.exp(v[base + k] - m); row[k] = e; s += e; }
        for (let k = 0; k < n; k++) out[base + k] = row[k] / s;
      }
      return out;
    }
    for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
      const base = o * n * inner + i;
      let m = -Infinity;
      for (let k = 0; k < n; k++) { const z = v[base + k * inner]; if (z > m) m = z; }
      if (m === -Infinity) m = 0;
      let s = 0;
      for (let k = 0; k < n; k++) s += Math.exp(v[base + k * inner] - m);
      if (log) { const l = m + Math.log(s); for (let k = 0; k < n; k++) out[base + k * inner] = v[base + k * inner] - l; }
      else for (let k = 0; k < n; k++) out[base + k * inner] = Math.exp(v[base + k * inner] - m) / s;
    }
    return out;
  }
  T.softmaxKernel = softmaxKernel;
  T.def('softmax', (x, axis = -1) => {
    x = T.as(x);
    const a = T.normAxis(axis, x.ndim), y = softmaxKernel(x.values(), x.shape, a, F(x), false);
    return T.record(T.make(y, x.shape), 'softmax', [x], (g) => {
      // dx = y ⊙ (g − Σ g ⊙ y)
      const { outer, n, inner } = split(x.shape, a), gv = g.values(), r = new (F(g))(y.length);
      for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
        const base = o * n * inner + i; let s = 0;
        for (let k = 0; k < n; k++) s += gv[base + k * inner] * y[base + k * inner];
        for (let k = 0; k < n; k++) { const j = base + k * inner; r[j] = y[j] * (gv[j] - s); }
      }
      return [T.make(r, x.shape)];
    }, { axis: a });
  });
  T.def('logSoftmax', (x, axis = -1) => {
    x = T.as(x);
    const a = T.normAxis(axis, x.ndim), y = softmaxKernel(x.values(), x.shape, a, F(x), true);
    return T.record(T.make(y, x.shape), 'logSoftmax', [x], (g) => {
      // dx = g − softmax ⊙ Σ g
      const { outer, n, inner } = split(x.shape, a), gv = g.values(), r = new (F(g))(y.length);
      for (let o = 0; o < outer; o++) for (let i = 0; i < inner; i++) {
        const base = o * n * inner + i; let s = 0;
        for (let k = 0; k < n; k++) s += gv[base + k * inner];
        for (let k = 0; k < n; k++) { const j = base + k * inner; r[j] = gv[j] - Math.exp(y[j]) * s; }
      }
      return [T.make(r, x.shape)];
    }, { axis: a });
  });
  T.def('logsumexp', (x, axis = -1, keepdims = false) => {
    const a = T.normAxis(axis, x.ndim), m = T.noGrad(() => T.max(x, a, true)).detach();
    const r = T.add(T.log(T.sum(T.exp(T.sub(x, m)), a, true)), m);
    return keepdims ? r : T.squeeze(r, a);
  });
})(globalThis.Nooral = globalThis.Nooral || {});
