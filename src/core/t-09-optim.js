// Optimisers, learning-rate schedules and gradient clipping. An optimiser updates its parameters in place from
// their .grad; its per-parameter state (velocity, moments) is kept in opt.state, keyed by parameter, for pictures.
//   new S.T.optim.SGD(params, {lr, momentum, nesterov, weightDecay, dampening})
//   Momentum / Nesterov (SGD with momentum 0.9 by default), AdaGrad {lr, eps}, RMSProp {lr, alpha, eps, momentum},
//   Adam {lr, betas, eps, weightDecay (L2)}, AdamW {lr, betas, eps, weightDecay (decoupled)}
//   opt.step() -> {lr, t}   opt.zeroGrad()   opt.schedule = S.T.sched.cosine({...})  (a function of the step t)
//   S.T.sched.constant / step({lr, every, gamma}) / exp({lr, gamma}) / cosine({lr, min, total}) / warmup({steps, then})
//   S.T.clipGradNorm(params, maxNorm) -> the norm before clipping;  clipGradValue(params, v)
(function (S) {
  'use strict';
  const T = S.T;
  const O = (T.optim = {});

  class Optimizer {
    constructor(params, opts, defaults) {
      this.params = (Array.isArray(params) ? params : params.parameters ? params.parameters() : [params]).filter((p) => p);
      this.opts = Object.assign({}, defaults, opts);
      this.lr = this.opts.lr;
      this.t = 0;
      this.state = new Map();
      this.schedule = this.opts.schedule || null;
    }
    zeroGrad() { for (const p of this.params) p.grad = null; }
    st(p) { let s = this.state.get(p); if (!s) { s = {}; this.state.set(p, s); } return s; }
    buf(s, key, p) { if (!s[key]) s[key] = new (p.data.constructor)(p.size); return s[key]; }
    step() {
      this.t++;
      if (this.schedule) this.lr = this.schedule(this.t - 1);
      for (const p of this.params) {
        if (!p.grad) continue;
        if (!p.isContiguous()) throw new Error('optimiser parameters must be contiguous tensors');
        this.update(p, p.data, p.grad.values(), this.st(p));
      }
      return { lr: this.lr, t: this.t };
    }
    /** a summary of the state for a picture: per parameter, the norms of each buffer */
    inspect() {
      return this.params.map((p) => {
        const s = this.state.get(p) || {}, r = { name: T.label(p) };
        for (const k of Object.keys(s)) if (ArrayBuffer.isView(s[k])) { let q = 0; for (const x of s[k]) q += x * x; r[k] = Math.sqrt(q); }
        return r;
      });
    }
  }
  O.Optimizer = Optimizer;

  O.SGD = class SGD extends Optimizer {
    constructor(params, opts) { super(params, opts, { lr: 0.01, momentum: 0, nesterov: false, weightDecay: 0, dampening: 0 }); this.name = this.opts.nesterov ? 'Nesterov' : this.opts.momentum ? 'Momentum' : 'SGD'; }
    update(p, w, g, s) {
      const { momentum: mu, nesterov, weightDecay: wd, dampening } = this.opts, lr = this.lr;
      if (!mu) { for (let i = 0; i < w.length; i++) w[i] -= lr * (g[i] + wd * w[i]); return; }
      const fresh = !s.v, v = this.buf(s, 'v', p);
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] + wd * w[i];
        v[i] = fresh ? gi : mu * v[i] + (1 - dampening) * gi;
        w[i] -= lr * (nesterov ? gi + mu * v[i] : v[i]);
      }
    }
  };
  O.Momentum = class Momentum extends O.SGD { constructor(params, opts) { super(params, Object.assign({ momentum: 0.9 }, opts)); this.name = 'Momentum'; } };
  O.Nesterov = class Nesterov extends O.SGD { constructor(params, opts) { super(params, Object.assign({ momentum: 0.9 }, opts, { nesterov: true })); this.name = 'Nesterov'; } };
  O.AdaGrad = class AdaGrad extends Optimizer {
    constructor(params, opts) { super(params, opts, { lr: 0.01, eps: 1e-10, weightDecay: 0 }); this.name = 'AdaGrad'; }
    update(p, w, g, s) {
      const h = this.buf(s, 'sumSq', p), { eps, weightDecay: wd } = this.opts;
      for (let i = 0; i < w.length; i++) { const gi = g[i] + wd * w[i]; h[i] += gi * gi; w[i] -= (this.lr * gi) / (Math.sqrt(h[i]) + eps); }
    }
  };
  O.RMSProp = class RMSProp extends Optimizer {
    constructor(params, opts) { super(params, opts, { lr: 0.01, alpha: 0.99, eps: 1e-8, momentum: 0, weightDecay: 0 }); this.name = 'RMSProp'; }
    update(p, w, g, s) {
      const sq = this.buf(s, 'sqAvg', p), { alpha, eps, momentum: mu, weightDecay: wd } = this.opts, b = mu ? this.buf(s, 'v', p) : null;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] + wd * w[i];
        sq[i] = alpha * sq[i] + (1 - alpha) * gi * gi;
        const u = gi / (Math.sqrt(sq[i]) + eps);
        if (b) { b[i] = mu * b[i] + u; w[i] -= this.lr * b[i]; } else w[i] -= this.lr * u;
      }
    }
  };
  O.Adam = class Adam extends Optimizer {
    constructor(params, opts) { super(params, opts, { lr: 0.001, betas: [0.9, 0.999], eps: 1e-8, weightDecay: 0, decoupled: false }); this.name = this.opts.decoupled ? 'AdamW' : 'Adam'; }
    update(p, w, g, s) {
      const m = this.buf(s, 'm', p), v = this.buf(s, 'v', p), [b1, b2] = this.opts.betas, { eps, weightDecay: wd, decoupled } = this.opts;
      s.t = (s.t || 0) + 1;
      const c1 = 1 - Math.pow(b1, s.t), c2 = 1 - Math.pow(b2, s.t), lr = this.lr;
      for (let i = 0; i < w.length; i++) {
        if (decoupled && wd) w[i] -= lr * wd * w[i];
        const gi = decoupled ? g[i] : g[i] + wd * w[i];
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        w[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
  };
  O.AdamW = class AdamW extends O.Adam { constructor(params, opts) { super(params, Object.assign({ weightDecay: 0.01 }, opts, { decoupled: true })); this.name = 'AdamW'; } };
  /** make an optimiser by name: 'sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam', 'adamw' */
  O.make = (name, params, opts) => {
    const C = { sgd: O.SGD, momentum: O.Momentum, nesterov: O.Nesterov, adagrad: O.AdaGrad, rmsprop: O.RMSProp, adam: O.Adam, adamw: O.AdamW }[String(name).toLowerCase()];
    if (!C) throw new Error('unknown optimiser ' + name);
    return new C(params, opts);
  };

  // ---- schedules: functions of the step number t (from 0) ----
  const Sc = (T.sched = {});
  Sc.constant = ({ lr }) => () => lr;
  Sc.step = ({ lr, every, gamma = 0.1 }) => (t) => lr * Math.pow(gamma, Math.floor(t / every));
  Sc.exp = ({ lr, gamma }) => (t) => lr * Math.pow(gamma, t);
  Sc.cosine = ({ lr, min = 0, total }) => (t) => min + 0.5 * (lr - min) * (1 + Math.cos(Math.PI * Math.min(t, total) / total));
  /** linear warm-up from 0 over `steps`, then `then` (a schedule, counted from the end of warm-up) or a constant lr */
  Sc.warmup = ({ steps, lr, then }) => (t) => {
    const after = then || (() => lr);
    return t < steps ? (after(0) * (t + 1)) / steps : after(t - steps);
  };
  Sc.curve = (fn, n) => Array.from({ length: n }, (_, t) => fn(t));

  const plist = (params) => (Array.isArray(params) ? params : params.parameters ? params.parameters() : [params]);
  T.gradNorm = (params) => { let s = 0; for (const p of plist(params)) if (p.grad) { const v = p.grad.values(); for (let i = 0; i < v.length; i++) s += v[i] * v[i]; } return Math.sqrt(s); };
  T.clipGradNorm = (params, maxNorm) => {
    const n = T.gradNorm(params);
    if (n > maxNorm && n > 0) { const k = maxNorm / n; for (const p of plist(params)) if (p.grad) { const d = p.grad.data; for (let i = 0; i < d.length; i++) d[i] *= k; } }
    return n;
  };
  T.clipGradValue = (params, v) => { for (const p of plist(params)) if (p.grad) { const d = p.grad.data; for (let i = 0; i < d.length; i++) d[i] = d[i] > v ? v : d[i] < -v ? -v : d[i]; } };
})(globalThis.Nooral = globalThis.Nooral || {});
