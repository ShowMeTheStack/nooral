// Tensors: n-dimensional arrays of floats with a shape, strides and an offset, so transposes and slices are views
// that share their data. Float32 by default; S.T.float64(fn) runs fn with Float64 storage (for gradient checks).
//   S.T.tensor(values, shape?)      nested arrays, a typed array or a number
//   S.T.zeros / ones / full / arange / eye / linspace
//   S.T.rng(seed)                   a seeded generator: next(), normal(), int(n), shuffle(a), fork()
//   S.T.uniform / normal / xavier / he (shape, rng, ...)  initialisers
//   t.shape, t.strides, t.size, t.values() (contiguous data), t.toArray() (nested), t.item(), t.get(...idx)
// The autodiff fields (requiresGrad, grad, ctx, name) live on the same object: see t-01-autodiff.js.
(function (S) {
  'use strict';
  const T = (S.T = S.T || {});
  T.F = Float32Array;
  let ID = 0;

  const prod = (s) => { let p = 1; for (const d of s) p *= d; return p; };
  const stridesOf = (s) => { const r = new Array(s.length); let p = 1; for (let i = s.length - 1; i >= 0; i--) { r[i] = p; p *= s[i]; } return r; };
  T.prod = prod;
  T.stridesOf = stridesOf;
  T.step = (kind, say, data) => Object.assign({ kind, say }, data || {});
  /** a number for a sentence: up to 4 significant digits */
  T.fmt = (x) => (Number.isInteger(x) ? String(x) : Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : String(+x.toPrecision(4)));

  class Tensor {
    constructor(data, shape, strides, offset) {
      this.data = data;
      this.shape = shape;
      this.strides = strides || stridesOf(shape);
      this.offset = offset || 0;
      this.size = prod(shape);
      this.id = ++ID;
      this.name = null;
      this.requiresGrad = false;
      this.grad = null;
      this.ctx = null;
    }
    get ndim() { return this.shape.length; }
    get dtype() { return this.data instanceof Float64Array ? 'float64' : 'float32'; }
    isContiguous() {
      if (this.offset !== 0 || this.data.length !== this.size) return false;
      const s = stridesOf(this.shape);
      for (let i = 0; i < s.length; i++) if (this.shape[i] > 1 && s[i] !== this.strides[i]) return false;
      return true;
    }
    /** the elements in row-major order, as a typed array (the storage itself when it already is) */
    values() {
      if (this.isContiguous()) return this.data;
      const out = new this.data.constructor(this.size), sh = this.shape, st = this.strides, n = sh.length;
      if (!this.size) return out;
      const idx = new Array(n).fill(0);
      let off = this.offset;
      for (let i = 0; i < this.size; i++) {
        out[i] = this.data[off];
        for (let d = n - 1; d >= 0; d--) {
          idx[d]++; off += st[d];
          if (idx[d] < sh[d]) break;
          off -= st[d] * sh[d]; idx[d] = 0;
        }
      }
      return out;
    }
    /** nested plain arrays */
    toArray() {
      const v = this.values(), sh = this.shape;
      if (!sh.length) return v[0];
      const rec = (d, base) => {
        const n = sh[d], step = prod(sh.slice(d + 1)), r = new Array(n);
        for (let i = 0; i < n; i++) r[i] = d === sh.length - 1 ? v[base + i] : rec(d + 1, base + i * step);
        return r;
      };
      return rec(0, 0);
    }
    item() { if (this.size !== 1) throw new Error('item() needs a tensor of one element, not shape [' + this.shape + ']'); return this.data[this.offset]; }
    get(...idx) { let o = this.offset; idx.forEach((k, d) => { o += (k < 0 ? k + this.shape[d] : k) * this.strides[d]; }); return this.data[o]; }
    toString() { return 'Tensor' + (this.name ? ' ' + this.name : '') + ' [' + this.shape.join(', ') + ']'; }
  }
  T.Tensor = Tensor;
  T.isTensor = (x) => x instanceof Tensor;

  /** run fn with float64 storage for every tensor made inside it */
  T.float64 = (fn) => { const old = T.F; T.F = Float64Array; try { return fn(); } finally { T.F = old; } };
  T.make = (data, shape, strides, offset) => new Tensor(data, shape.slice(), strides && strides.slice(), offset);
  T.empty = (shape, F) => new Tensor(new (F || T.F)(prod(shape)), shape.slice());

  function flatten(a, out, shape, d) {
    if (Array.isArray(a)) {
      if (shape.length <= d) shape.push(a.length);
      else if (shape[d] !== a.length) throw new Error('ragged nested array');
      for (const x of a) flatten(x, out, shape, d + 1);
    } else out.push(+a);
  }
  /** a tensor from nested arrays, a typed or plain array (with a shape), or a number */
  T.tensor = (v, shape, opts) => {
    let t;
    if (v instanceof Tensor) t = T.make(new T.F(v.values()), v.shape);
    else if (typeof v === 'number') t = T.make(new T.F([v]), shape || []);
    else if (ArrayBuffer.isView(v)) t = T.make(new T.F(v), shape || [v.length]);
    else {
      const flat = [], sh = [];
      flatten(v, flat, sh, 0);
      t = T.make(new T.F(flat), shape || sh);
    }
    if (prod(t.shape) !== t.data.length) throw new Error('shape [' + t.shape + '] does not hold ' + t.data.length + ' values');
    if (opts && opts.requiresGrad) t.requiresGrad = true;
    if (opts && opts.name) t.name = opts.name;
    return t;
  };
  T.scalar = (x) => T.make(new T.F([x]), []);
  T.full = (shape, x) => { const t = T.empty(shape); t.data.fill(x); return t; };
  T.zeros = (shape) => T.empty(shape);
  T.ones = (shape) => T.full(shape, 1);
  T.zerosLike = (t) => T.empty(t.shape, t.data.constructor);
  T.onesLike = (t) => { const z = T.empty(t.shape, t.data.constructor); z.data.fill(1); return z; };
  T.arange = (a, b, step = 1) => { if (b === undefined) { b = a; a = 0; } const n = Math.max(0, Math.ceil((b - a) / step)), t = T.empty([n]); for (let i = 0; i < n; i++) t.data[i] = a + i * step; return t; };
  T.linspace = (a, b, n) => { const t = T.empty([n]); for (let i = 0; i < n; i++) t.data[i] = n === 1 ? a : a + ((b - a) * i) / (n - 1); return t; };
  T.eye = (n) => { const t = T.empty([n, n]); for (let i = 0; i < n; i++) t.data[i * n + i] = 1; return t; };

  // ---- the seeded generator: mulberry32, with Box-Muller normals ----
  T.rng = (seed = 1) => {
    let s = seed >>> 0, spare = null;
    const r = {
      seed,
      next() { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; },
      normal(mean = 0, std = 1) {
        if (spare !== null) { const z = spare; spare = null; return mean + std * z; }
        let u = 0; while (u === 0) u = r.next();
        const v = r.next(), m = Math.sqrt(-2 * Math.log(u));
        spare = m * Math.sin(2 * Math.PI * v);
        return mean + std * m * Math.cos(2 * Math.PI * v);
      },
      uniform(a = 0, b = 1) { return a + (b - a) * r.next(); },
      int(n) { return Math.floor(r.next() * n); },
      shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = r.int(i + 1); const x = a[i]; a[i] = a[j]; a[j] = x; } return a; },
      permutation(n) { return r.shuffle(Array.from({ length: n }, (_, i) => i)); },
      /** a new generator whose stream depends on this one's state, for a sub-task */
      fork() { return T.rng((r.next() * 4294967296) >>> 0); },
      state() { return { s, spare }; },
      setState(st) { s = st.s >>> 0; spare = st.spare; },
    };
    return r;
  };
  const asRng = (r) => (r && typeof r.next === 'function' ? r : T.rng(r === undefined ? 1 : r));
  T.asRng = asRng;

  // ---- initialisers ----
  T.uniform = (shape, a = 0, b = 1, rng) => { const g = asRng(rng), t = T.empty(shape); for (let i = 0; i < t.size; i++) t.data[i] = a + (b - a) * g.next(); return t; };
  T.rand = (shape, rng) => T.uniform(shape, 0, 1, rng);
  T.normal = (shape, mean = 0, std = 1, rng) => { const g = asRng(rng), t = T.empty(shape); for (let i = 0; i < t.size; i++) t.data[i] = g.normal(mean, std); return t; };
  T.randn = (shape, rng) => T.normal(shape, 0, 1, rng);
  /** fan in and fan out of a weight: [in, out] for a dense layer, [out, in/groups, kh, kw] for a convolution */
  T.fans = (shape) => {
    if (shape.length === 2) return { fanIn: shape[0], fanOut: shape[1] };
    if (shape.length > 2) { const rf = prod(shape.slice(2)); return { fanIn: shape[1] * rf, fanOut: shape[0] * rf }; }
    return { fanIn: shape[0] || 1, fanOut: shape[0] || 1 };
  };
  /** Xavier (Glorot): variance 2 / (fanIn + fanOut); dist 'uniform' (default) or 'normal' */
  T.xavier = (shape, rng, opts = {}) => {
    const f = Object.assign(T.fans(shape), opts), gain = opts.gain || 1;
    const std = gain * Math.sqrt(2 / (f.fanIn + f.fanOut));
    return opts.dist === 'normal' ? T.normal(shape, 0, std, rng) : T.uniform(shape, -std * Math.sqrt(3), std * Math.sqrt(3), rng);
  };
  /** He (Kaiming): variance 2 / fanIn, for ReLU layers; dist 'normal' (default) or 'uniform' */
  T.he = (shape, rng, opts = {}) => {
    const f = Object.assign(T.fans(shape), opts), std = Math.sqrt(2 / f.fanIn);
    return opts.dist === 'uniform' ? T.uniform(shape, -std * Math.sqrt(3), std * Math.sqrt(3), rng) : T.normal(shape, 0, std, rng);
  };
  T.init = (kind, shape, rng, opts) => {
    if (kind === 'zeros') return T.zeros(shape);
    if (kind === 'ones') return T.ones(shape);
    if (kind === 'he') return T.he(shape, rng, opts);
    if (kind === 'xavier') return T.xavier(shape, rng, opts);
    if (kind === 'normal') return T.normal(shape, 0, (opts && opts.std) || 0.02, rng);
    if (kind === 'uniform') { const a = (opts && opts.a) || 0.1; return T.uniform(shape, -a, a, rng); }
    throw new Error('unknown initialiser ' + kind);
  };

  // ---- broadcasting ----
  T.broadcastShapes = (...shapes) => {
    const n = Math.max(...shapes.map((s) => s.length)), out = new Array(n).fill(1);
    for (const s of shapes) {
      for (let i = 0; i < s.length; i++) {
        const d = n - s.length + i, a = out[d], b = s[i];
        if (a === b || b === 1) continue;
        if (a === 1) out[d] = b;
        else throw new Error('shapes [' + shapes.map((x) => x.join(',')).join('] and [') + '] do not broadcast');
      }
    }
    return out;
  };
  /** strides that read a tensor of shape `shape` as if it had shape `to` (0 along broadcast axes) */
  T.broadcastStrides = (shape, strides, to) => {
    const r = new Array(to.length).fill(0), k = to.length - shape.length;
    for (let i = 0; i < shape.length; i++) r[k + i] = shape[i] === 1 ? 0 : strides[i];
    return r;
  };
  /** the values of t laid out in shape `to` (a copy) */
  T.expandValues = (t, to) => {
    if (t.shape.length === to.length && t.shape.every((d, i) => d === to[i])) return t.values();
    const v = T.make(t.data, to, T.broadcastStrides(t.shape, t.strides, to), t.offset);
    return v.values();
  };
  T.sameShape = (a, b) => a.length === b.length && a.every((d, i) => d === b[i]);
  T.normAxis = (axis, n) => { const a = axis < 0 ? axis + n : axis; if (a < 0 || a >= n) throw new Error('axis ' + axis + ' out of range for ' + n + ' dimensions'); return a; };
})(globalThis.Nooral = globalThis.Nooral || {});
