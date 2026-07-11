// S.ML base: seeded random numbers, traces, the estimator pattern and small helpers shared by every
// classical algorithm. Data are plain arrays: X is an array of rows, y an array of labels or numbers.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});

  // mulberry32: small, fast and the same on every platform
  ML.rng = function (seed) {
    let s = (seed == null ? 1 : seed) >>> 0;
    const r = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    r.int = (n) => Math.floor(r() * n);
    let spare = null;
    r.normal = () => {
      if (spare != null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0;
      while (u <= 1e-300) u = r();
      v = r();
      const m = Math.sqrt(-2 * Math.log(u));
      spare = m * Math.sin(2 * Math.PI * v);
      return m * Math.cos(2 * Math.PI * v);
    };
    r.shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = r.int(i + 1); const t = a[i]; a[i] = a[j]; a[j] = t; } return a; };
    r.perm = (n) => r.shuffle([...Array(n).keys()]);
    // k distinct indices out of n
    r.sample = (n, k) => r.perm(n).slice(0, k);
    return r;
  };
  ML.asRng = (seed) => (typeof seed === 'function' ? seed : ML.rng(seed == null ? 0 : seed));

  // numbers in the "say" sentences: short, honest about precision
  ML.fmt = function (x, d = 3) {
    if (typeof x !== 'number') return String(x);
    if (!isFinite(x)) return String(x);
    if (x !== 0 && (Math.abs(x) >= 1e5 || Math.abs(x) < 1e-3)) return x.toExponential(2);
    return String(+x.toFixed(d));
  };
  ML.fmtv = (v, d = 3) => '[' + Array.from(v, (x) => ML.fmt(x, d)).join(', ') + ']';

  // A tracer collects steps {kind, say, ...data}. Off when opts.trace === false; capped by opts.maxSteps.
  ML.tracer = function (opts) {
    opts = opts || {};
    const on = opts.trace !== false, cap = opts.maxSteps || 20000;
    const t = { on, steps: [], dropped: 0 };
    t.add = function (kind, say, data) {
      if (!on) return;
      if (t.steps.length >= cap) { t.dropped++; return; }
      t.steps.push(Object.assign({ kind, say }, data || {}));
    };
    return t;
  };

  /**
   * class weights as one weight per class index, or null: spec null | 'none' | 'balanced' (n / (K count_k), as
   * scikit-learn) | {label: weight} | [weight per class index]; classes the sorted labels, idx each row's class index
   */
  ML.classWeights = function (spec, classes, idx) {
    if (spec == null || spec === 'none' || spec === '') return null;
    const K = classes.length;
    if (spec === 'balanced') {
      const cnt = new Array(K).fill(0);
      for (const k of idx) cnt[k]++;
      return cnt.map((c) => (c ? idx.length / (K * c) : 1));
    }
    if (Array.isArray(spec)) return classes.map((_, k) => (spec[k] == null ? 1 : +spec[k]));
    if (typeof spec === 'object') return classes.map((c) => (spec[c] == null ? 1 : +spec[c]));
    throw new Error('class weights: none, balanced, or a weight for each class');
  };

  // ---- the estimator pattern ----
  // ML.model(name, {kind, defaults, fit(X, y, tr), ...methods}) defines new ML[name](params).
  // fit() returns this and leaves this.trace; clone(over) gives an unfitted copy with changed params.
  ML.registry = ML.registry || {};
  const Base = {
    fit(X, y) {
      const tr = ML.tracer(this.params);
      this._fit(X, y, tr);
      this.trace = tr.steps;
      this.fitted = true;
      return this;
    },
    clone(over) { return new this.constructor(Object.assign({}, this.params, over || {})); },
    score(X, y) {
      if (this.kind === 'regressor') return ML.r2(y, this.predict(X));
      return ML.accuracy(y, this.predict(X));
    },
    // classes for probabilistic classifiers: argmax of predictProba
    _argmaxPredict(X) { return this.predictProba(X).map((p) => this.classes[ML.argmax(p)]); },
  };
  ML.model = function (name, spec) {
    function C(params) {
      if (!(this instanceof C)) return new C(params);
      this.params = Object.assign({}, spec.defaults || {}, params || {});
    }
    const proto = Object.assign(Object.create(Base), spec);
    proto._fit = spec.fit;
    delete proto.fit;
    proto.constructor = C;
    proto.name = name;
    C.prototype = proto;
    C.kind = spec.kind;
    C.defaults = spec.defaults || {};
    ML.registry[name] = C;
    ML[name] = C;
    return C;
  };
  ML.create = (name, params) => new ML.registry[name](params);

  // ---- small helpers ----
  ML.argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  ML.argmin = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[b]) b = i; return b; };
  ML.sum = (a) => { let s = 0; for (const v of a) s += v; return s; };
  ML.mean = (a) => ML.sum(a) / a.length;
  ML.range = (n) => [...Array(n).keys()];
  ML.take = (a, idx) => idx.map((i) => a[i]);
  ML.cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
  // sorted unique labels and each sample's index into them
  ML.encode = function (y) {
    const classes = [...new Set(y)].sort(ML.cmp);
    const pos = new Map(classes.map((c, i) => [c, i]));
    return { classes, idx: y.map((v) => pos.get(v)) };
  };
  ML.sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  ML.logsumexp = (a) => { let m = -Infinity; for (const v of a) if (v > m) m = v; if (m === -Infinity) return m; let s = 0; for (const v of a) s += Math.exp(v - m); return m + Math.log(s); };
  ML.softmax = (a) => { const l = ML.logsumexp(a); return a.map((v) => Math.exp(v - l)); };
  ML.accuracy = (y, p) => { let c = 0; for (let i = 0; i < y.length; i++) if (y[i] === p[i]) c++; return c / y.length; };
  ML.r2 = (y, p) => {
    const m = ML.mean(y); let ss = 0, st = 0;
    for (let i = 0; i < y.length; i++) { ss += (y[i] - p[i]) ** 2; st += (y[i] - m) ** 2; }
    return st === 0 ? (ss === 0 ? 1 : 0) : 1 - ss / st;
  };
  ML.mse = (y, p) => { let s = 0; for (let i = 0; i < y.length; i++) s += (y[i] - p[i]) ** 2; return s / y.length; };
  ML.dist = function (a, b, p = 2) {
    let s = 0;
    if (p === 2) { for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
    if (p === 1) { for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s; }
    if (p === Infinity) { for (let i = 0; i < a.length; i++) s = Math.max(s, Math.abs(a[i] - b[i])); return s; }
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) ** p;
    return s ** (1 / p);
  };
  ML.sqdist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s; };
  ML.colMeans = (X) => { const d = X[0].length, m = new Array(d).fill(0); for (const r of X) for (let j = 0; j < d; j++) m[j] += r[j]; return m.map((v) => v / X.length); };
  ML.colVars = (X, ddof = 0) => {
    const m = ML.colMeans(X), d = m.length, v = new Array(d).fill(0);
    for (const r of X) for (let j = 0; j < d; j++) v[j] += (r[j] - m[j]) ** 2;
    return v.map((s) => s / (X.length - ddof));
  };
  ML.center = (X, m) => X.map((r) => r.map((v, j) => v - m[j]));
})();
