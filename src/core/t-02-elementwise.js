// Elementwise ops with broadcasting, and the activations. Each records its backward: for a broadcast input the
// gradient is summed back down to the input's shape (sumTo).
//   add sub mul div pow maximum minimum   (a, b)   numbers are fine for either side
//   neg exp log sqrt abs square sin cos   (x)
//   relu sigmoid tanh gelu (tanh form) leakyRelu(x, slope) elu(x, a) softplus clamp(x, lo, hi)
//   where(cond, a, b), maskedFill(x, mask, value), sumTo(g, shape), broadcastTo(x, shape)
(function (S) {
  'use strict';
  const T = S.T;
  const F = (t) => t.data.constructor;
  const outF = (a, b) => (a.data instanceof Float64Array || (b && b.data instanceof Float64Array) ? Float64Array : T.F === Float64Array ? Float64Array : Float32Array);

  // monomorphic loops for the four arithmetic ops (a shared loop calling f would be slow): s same shape,
  // r b repeats every m elements (a row or a scalar), l a repeats every m elements
  const LOOPS = {
    add: { s: (x, y, o, n) => { for (let i = 0; i < n; i++) o[i] = x[i] + y[i]; }, r: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[i] + y[j]; }, l: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[j] + y[i]; } },
    sub: { s: (x, y, o, n) => { for (let i = 0; i < n; i++) o[i] = x[i] - y[i]; }, r: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[i] - y[j]; }, l: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[j] - y[i]; } },
    mul: { s: (x, y, o, n) => { for (let i = 0; i < n; i++) o[i] = x[i] * y[i]; }, r: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[i] * y[j]; }, l: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[j] * y[i]; } },
    div: { s: (x, y, o, n) => { for (let i = 0; i < n; i++) o[i] = x[i] / y[i]; }, r: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[i] / y[j]; }, l: (x, y, o, n, m) => { for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) o[i] = x[j] / y[i]; } },
  };
  /** apply f(x, y) with broadcasting; returns a contiguous tensor (op names one of LOOPS for a faster loop) */
  function binaryKernel(a, b, f, op) {
    const shape = T.sameShape(a.shape, b.shape) ? a.shape.slice() : T.broadcastShapes(a.shape, b.shape);
    const n = T.prod(shape), C = outF(a, b), out = new C(n), L = op && LOOPS[op];
    if (a.size === n && b.size === n) { const x = a.values(), y = b.values(); if (L) L.s(x, y, out, n); else for (let i = 0; i < n; i++) out[i] = f(x[i], y[i]); }
    else if (b.size === 1) { const x = a.size === n ? a.values() : T.expandValues(a, shape), y = b.values(); if (L) L.r(x, y, out, n, 1); else { const z = y[0]; for (let i = 0; i < n; i++) out[i] = f(x[i], z); } }
    else if (a.size === 1) { const x = a.values(), y = b.size === n ? b.values() : T.expandValues(b, shape); if (L) L.l(x, y, out, n, 1); else { const z = x[0]; for (let i = 0; i < n; i++) out[i] = f(z, y[i]); } }
    else if (a.size === n && isSuffix(b.shape, shape)) { const x = a.values(), y = b.values(), m = y.length; if (L) L.r(x, y, out, n, m); else for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) out[i] = f(x[i], y[j]); }
    else if (b.size === n && isSuffix(a.shape, shape)) { const x = a.values(), y = b.values(), m = x.length; if (L) L.l(x, y, out, n, m); else for (let i = 0, j = 0; i < n; i++, j = j + 1 === m ? 0 : j + 1) out[i] = f(x[j], y[i]); }
    else { const x = T.expandValues(a, shape), y = T.expandValues(b, shape); if (L) L.s(x, y, out, n); else for (let i = 0; i < n; i++) out[i] = f(x[i], y[i]); }
    return T.make(out, shape);
  }
  // b's shape is the trailing part of `shape` (ignoring leading ones), so b repeats every b.size elements
  function isSuffix(s, shape) {
    let i = 0; while (i < s.length && s[i] === 1) i++;
    const r = s.slice(i), k = shape.length - r.length;
    return k >= 0 && r.every((d, j) => d === shape[k + j]);
  }
  T.binaryKernel = binaryKernel;

  /** sum a gradient over the axes that broadcasting added or stretched, back to `shape` */
  T.sumTo = (g, shape) => {
    if (T.sameShape(g.shape, shape)) return g;
    const n = g.shape.length, k = n - shape.length, gs = g.shape;
    const C = F(g), out = new C(T.prod(shape)), v = g.values();
    if (T.prod(shape) === 1) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]; out[0] = s; return T.make(out, shape); }
    if (isSuffix(shape, gs)) { const m = out.length; for (let i = 0, j = 0; i < v.length; i++, j = j + 1 === m ? 0 : j + 1) out[j] += v[i]; return T.make(out, shape); }
    // general: walk g with an index into out whose strides are 0 along summed axes
    const os = T.stridesOf(shape), st = new Array(n).fill(0);
    for (let d = 0; d < shape.length; d++) st[k + d] = shape[d] === 1 ? 0 : os[d];
    const idx = new Array(n).fill(0);
    let o = 0;
    for (let i = 0; i < v.length; i++) {
      out[o] += v[i];
      for (let d = n - 1; d >= 0; d--) { idx[d]++; o += st[d]; if (idx[d] < gs[d]) break; o -= st[d] * gs[d]; idx[d] = 0; }
    }
    return T.make(out, shape);
  };

  const binary = (name, f, grads) => T.def(name, (a, b) => {
    a = T.as(a, b instanceof T.Tensor ? b : null); b = T.as(b, a);
    const out = binaryKernel(a, b, f, name);
    return T.record(out, name, [a, b], (g) => {
      const [ga, gb] = grads(g, a, b, out);
      return [a.requiresGrad && ga ? T.sumTo(ga, a.shape) : null, b.requiresGrad && gb ? T.sumTo(gb, b.shape) : null];
    });
  });
  const MUL = (u, v) => u * v, DIV = (u, v) => u / v;
  const k2 = (g, x, f) => binaryKernel(g, x, f, f === MUL ? 'mul' : f === DIV ? 'div' : null);
  binary('add', (x, y) => x + y, (g) => [g, g]);
  binary('sub', (x, y) => x - y, (g) => [g, k2(g, T.scalar(-1), MUL)]);
  binary('mul', (x, y) => x * y, (g, a, b) => [a.requiresGrad ? k2(g, b, MUL) : null, b.requiresGrad ? k2(g, a, MUL) : null]);
  binary('div', (x, y) => x / y, (g, a, b) => [a.requiresGrad ? k2(g, b, DIV) : null, b.requiresGrad ? binaryKernel(k2(g, a, (u, v) => -u * v), k2(b, b, (u, v) => u * v), (u, v) => u / v) : null]);
  binary('pow', (x, y) => Math.pow(x, y), (g, a, b, out) => [
    a.requiresGrad ? binaryKernel(k2(g, b, (u, v) => u * v), binaryKernel(a, b, (x, y) => (y === 0 ? 0 : Math.pow(x, y - 1))), (u, v) => u * v) : null,
    b.requiresGrad ? binaryKernel(k2(g, out, (u, v) => u * v), a, (u, x) => (x > 0 ? u * Math.log(x) : 0)) : null,
  ]);
  binary('maximum', (x, y) => (x >= y ? x : y), (g, a, b) => [k2(g, binaryKernel(a, b, (x, y) => (x >= y ? 1 : 0)), (u, v) => u * v), k2(g, binaryKernel(a, b, (x, y) => (x >= y ? 0 : 1)), (u, v) => u * v)]);
  binary('minimum', (x, y) => (x <= y ? x : y), (g, a, b) => [k2(g, binaryKernel(a, b, (x, y) => (x <= y ? 1 : 0)), (u, v) => u * v), k2(g, binaryKernel(a, b, (x, y) => (x <= y ? 0 : 1)), (u, v) => u * v)]);

  /** an elementwise function f with derivative df(x, y) (y = f(x)) */
  const unary = (name, f, df, info) => T.def(name, (x, ...args) => {
    x = T.as(x);
    const v = x.values(), n = v.length, out = new (F(x))(n), fast = FAST[name];
    const aux = fast && fast[2] ? new Float64Array(n) : null;
    if (fast) fast[0](v, out, n, args[0], aux);
    else { const fa = args.length ? (z) => f(z, ...args) : f; for (let i = 0; i < n; i++) out[i] = fa(v[i]); }
    const y = T.make(out, x.shape);
    return T.record(y, name, [x], (g) => {
      const gv = g.values(), r = new (F(g))(n);
      if (fast) fast[1](v, out, gv, r, n, args[0], aux);
      else for (let i = 0; i < n; i++) r[i] = gv[i] * df(v[i], out[i], ...args);
      return [T.make(r, x.shape)];
    }, info ? info(...args) : null);
  });
  // the hot activations as their own loops, with the same arithmetic as f and df: [forward (v, out, n, c), backward (v, y, g, r, n, c)]
  const GC = Math.sqrt(2 / Math.PI);
  const FAST = {
    relu: [(v, o, n) => { for (let i = 0; i < n; i++) o[i] = v[i] > 0 ? v[i] : 0; }, (v, y, g, r, n) => { for (let i = 0; i < n; i++) r[i] = g[i] * (v[i] > 0 ? 1 : 0); }],
    tanh: [(v, o, n) => { for (let i = 0; i < n; i++) o[i] = Math.tanh(v[i]); }, (v, y, g, r, n) => { for (let i = 0; i < n; i++) r[i] = g[i] * (1 - y[i] * y[i]); }],
    exp: [(v, o, n) => { for (let i = 0; i < n; i++) o[i] = Math.exp(v[i]); }, (v, y, g, r, n) => { for (let i = 0; i < n; i++) r[i] = g[i] * y[i]; }],
    neg: [(v, o, n) => { for (let i = 0; i < n; i++) o[i] = -v[i]; }, (v, y, g, r, n) => { for (let i = 0; i < n; i++) r[i] = -g[i]; }],
    square: [(v, o, n) => { for (let i = 0; i < n; i++) o[i] = v[i] * v[i]; }, (v, y, g, r, n) => { for (let i = 0; i < n; i++) r[i] = 2 * v[i] * g[i]; }],
    scale: [(v, o, n, c) => { for (let i = 0; i < n; i++) o[i] = v[i] * c; }, (v, y, g, r, n, c) => { for (let i = 0; i < n; i++) r[i] = g[i] * c; }],
    addScalar: [(v, o, n, c) => { for (let i = 0; i < n; i++) o[i] = v[i] + c; }, (v, y, g, r, n) => { for (let i = 0; i < n; i++) r[i] = g[i]; }],
    sigmoid: [(v, o, n) => { for (let i = 0; i < n; i++) { const x = v[i]; o[i] = x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); } }, (v, y, g, r, n) => { for (let i = 0; i < n; i++) r[i] = g[i] * (y[i] * (1 - y[i])); }],
    // GELU keeps tanh(u) from the forward pass for its backward
    gelu: [(v, o, n, c, th) => { for (let i = 0; i < n; i++) { const x = v[i], t = Math.tanh(GC * (x + 0.044715 * x * x * x)); th[i] = t; o[i] = 0.5 * x * (1 + t); } },
      (v, y, g, r, n, c, th) => { for (let i = 0; i < n; i++) { const x = v[i], t = th[i]; r[i] = g[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GC * (1 + 3 * 0.044715 * x * x)); } }, true],
  };
  T.unary = unary;
  unary('neg', (x) => -x, () => -1);
  unary('exp', Math.exp, (x, y) => y);
  unary('log', Math.log, (x) => 1 / x);
  unary('sqrt', Math.sqrt, (x, y) => 0.5 / y);
  unary('abs', Math.abs, (x) => (x > 0 ? 1 : x < 0 ? -1 : 0));
  unary('square', (x) => x * x, (x) => 2 * x);
  unary('sin', Math.sin, (x) => Math.cos(x));
  unary('cos', Math.cos, (x) => -Math.sin(x));
  unary('relu', (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0));
  unary('sigmoid', (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))), (x, y) => y * (1 - y));
  unary('tanh', Math.tanh, (x, y) => 1 - y * y);
  unary('leakyRelu', (x, s = 0.01) => (x > 0 ? x : s * x), (x, y, s = 0.01) => (x > 0 ? 1 : s), (s = 0.01) => ({ slope: s }));
  unary('elu', (x, a = 1) => (x > 0 ? x : a * (Math.exp(x) - 1)), (x, y, a = 1) => (x > 0 ? 1 : y + a), (a = 1) => ({ alpha: a }));
  unary('softplus', (x) => (x > 20 ? x : Math.log1p(Math.exp(x))), (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))));
  unary('clamp', (x, lo = -Infinity, hi = Infinity) => (x < lo ? lo : x > hi ? hi : x), (x, y, lo = -Infinity, hi = Infinity) => (x < lo || x > hi ? 0 : 1));
  unary('scale', (x, c) => x * c, (x, y, c) => c, (c) => ({ c }));
  unary('addScalar', (x, c) => x + c, () => 1, (c) => ({ c }));
  // GELU in its tanh form: 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
  unary('gelu', (x) => 0.5 * x * (1 + Math.tanh(GC * (x + 0.044715 * x * x * x))), (x) => {
    const u = GC * (x + 0.044715 * x * x * x), th = Math.tanh(u);
    return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * GC * (1 + 3 * 0.044715 * x * x);
  });

  /** a copy of x laid out in a bigger, broadcast-compatible shape */
  T.def('broadcastTo', (x, shape) => {
    x = T.as(x);
    if (T.sameShape(x.shape, shape)) return x;
    const out = T.make(new (F(x))(T.expandValues(x, shape)), shape);
    return T.record(out, 'broadcastTo', [x], (g) => [T.sumTo(g, x.shape)]);
  });
  /** where(cond, a, b): a where cond is non-zero, else b (cond needs no gradient) */
  T.def('where', (cond, a, b) => {
    cond = T.as(cond); a = T.as(a); b = T.as(b);
    const shape = T.broadcastShapes(cond.shape, a.shape, b.shape), c = T.expandValues(cond, shape), x = T.expandValues(a, shape), y = T.expandValues(b, shape);
    const n = T.prod(shape), out = new (outF(a, b))(n);
    for (let i = 0; i < n; i++) out[i] = c[i] ? x[i] : y[i];
    return T.record(T.make(out, shape), 'where', [a, b], (g) => {
      const gv = g.values(), ga = new (F(g))(n), gb = new (F(g))(n);
      for (let i = 0; i < n; i++) { if (c[i]) ga[i] = gv[i]; else gb[i] = gv[i]; }
      return [T.sumTo(T.make(ga, shape), a.shape), T.sumTo(T.make(gb, shape), b.shape)];
    });
  });
  /** x with `value` wherever mask is non-zero (mask broadcasts to x) */
  T.def('maskedFill', (x, mask, value) => {
    mask = T.as(mask);
    // a mask over the trailing axes (such as one [T, T] mask for every batch and head) repeats without a copy
    const per = T.sameShape(mask.shape, x.shape) || !isSuffix(mask.shape, x.shape) ? x.size : mask.size;
    const m = per === x.size ? T.expandValues(mask, x.shape) : mask.values(), v = x.values(), n = v.length, out = new (F(x))(n);
    for (let i = 0, j = 0; i < n; i++, j = j + 1 === per ? 0 : j + 1) out[i] = m[j] ? value : v[i];
    return T.record(T.make(out, x.shape), 'maskedFill', [x], (g) => {
      const gv = g.values(), r = new (F(g))(n);
      for (let i = 0, j = 0; i < n; i++, j = j + 1 === per ? 0 : j + 1) r[i] = m[j] ? 0 : gv[i];
      return [T.make(r, x.shape)];
    });
  });
  // comparisons give 0/1 tensors with no gradient
  const cmp = (name, f) => T.def(name, (a, b) => binaryKernel(T.as(a), T.as(b, a), (x, y) => (f(x, y) ? 1 : 0)));
  cmp('gt', (x, y) => x > y); cmp('lt', (x, y) => x < y); cmp('ge', (x, y) => x >= y); cmp('le', (x, y) => x <= y); cmp('eq', (x, y) => x === y);
})(globalThis.Nooral = globalThis.Nooral || {});
