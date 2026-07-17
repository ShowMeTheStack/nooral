// Shared pieces for the generative models (S.Gen), and a small adapter that lets the generative, RL, GNN and
// robustness engines register training jobs with S.T.jobs although their files load before t-15-worker.js.
//   S.lateJob(name, factory)       register now if S.T.jobs exists, else as soon as it is created
//   S.Gen.dist2d(name, {n, seed})  2-D target distributions: 'ring' (8 Gaussians), 'grid' (5×5), 'circle', 'moons',
//                                  'spiral', 'two' -> {X: [[x, y]], modes: [[x, y]] | null, sd, name}
//   S.Gen.images(kind, {n, seed, size: 16, classes})  generated digits, shapes or icons -> {X: Float32Array(n·size²), y, n, size}
//   S.Gen.swd(A, B, {dirs})        sliced Wasserstein-1 distance between two 2-D point sets (fixed directions)
//   S.Gen.chamfer(A, B)            mean distance from each point of A to its nearest point of B
//   S.Gen.coverage(P, modes, r)    how many modes have samples within r (covered), the effective number of modes
//                                  the samples spread over (e^entropy), and the share of samples near a mode (quality)
//   S.Gen.trainer(job)             wraps {step, snapshot} into a stepping trainer with history and snapshots
(function (S) {
  'use strict';
  const G = (S.Gen = S.Gen || {});

  // ---- late registration of training jobs ----
  S.lateJobs = S.lateJobs || [];
  S.lateJob = (name, factory) => {
    const T = S.T;
    if (T && T.jobs && typeof T.jobs.register === 'function') { T.jobs.register(name, factory); return; }
    S.lateJobs.push([name, factory]);
    const T0 = (S.T = S.T || {});
    const d = Object.getOwnPropertyDescriptor(T0, 'jobs');
    if (d && d.set) return;
    let v;
    Object.defineProperty(T0, 'jobs', {
      configurable: true, enumerable: true, get: () => v,
      set(x) {
        v = x;
        Object.defineProperty(T0, 'jobs', { value: x, writable: true, configurable: true, enumerable: true });
        for (const [n, f] of S.lateJobs.splice(0)) x.register(n, f);
      },
    });
  };

  const fmt = (x) => S.T.fmt(x);
  G.fmt = fmt;
  const pt = (p) => '(' + fmt(p[0]) + ', ' + fmt(p[1]) + ')';
  G.pt = pt;

  // ---- 2-D target distributions ----
  G.distNames = ['ring', 'grid', 'circle', 'moons', 'spiral', 'two'];
  G.dist2d = (name = 'ring', o = {}) => {
    const n = o.n || 512, r = S.T.rng(o.seed || 1), X = [];
    let modes = null, sd = o.sd;
    if (name === 'ring') {
      const k = o.k || 8, R = o.radius || 2; sd = sd || 0.08;
      modes = Array.from({ length: k }, (_, i) => [R * Math.cos((2 * Math.PI * i) / k), R * Math.sin((2 * Math.PI * i) / k)]);
      for (let i = 0; i < n; i++) { const m = modes[i % k]; X.push([m[0] + r.normal(0, sd), m[1] + r.normal(0, sd)]); }
    } else if (name === 'grid') {
      const k = o.k || 5; sd = sd || 0.06; modes = [];
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) modes.push([-2 + (4 * a) / (k - 1), -2 + (4 * b) / (k - 1)]);
      for (let i = 0; i < n; i++) { const m = modes[i % modes.length]; X.push([m[0] + r.normal(0, sd), m[1] + r.normal(0, sd)]); }
    } else if (name === 'circle') {
      sd = sd || 0.05;
      for (let i = 0; i < n; i++) { const t = 2 * Math.PI * r.next(); X.push([1.5 * Math.cos(t) + r.normal(0, sd), 1.5 * Math.sin(t) + r.normal(0, sd)]); }
    } else if (name === 'moons') {
      sd = sd || 0.06;
      for (let i = 0; i < n; i++) { const c = i % 2, t = Math.PI * r.next(); X.push([(c ? 1 - Math.cos(t) : Math.cos(t)) * 1.4 - 0.7 + r.normal(0, sd), (c ? 0.5 - Math.sin(t) : Math.sin(t)) * 1.4 - 0.35 + r.normal(0, sd)]); }
    } else if (name === 'spiral') {
      sd = sd || 0.05;
      for (let i = 0; i < n; i++) { const t = 0.5 + 3.5 * Math.PI * Math.sqrt(r.next()), s = t / (4 * Math.PI) * 2.2; X.push([s * Math.cos(t) + r.normal(0, sd), s * Math.sin(t) + r.normal(0, sd)]); }
    } else if (name === 'two') {
      sd = sd || 0.25; modes = [[-1.2, -0.6], [1.2, 0.6]];
      for (let i = 0; i < n; i++) { const m = modes[i % 2]; X.push([m[0] + r.normal(0, sd), m[1] + r.normal(0, sd)]); }
    } else throw new Error('no 2-D distribution called ' + name);
    // rows in a seeded order, so a prefix is a fair sample
    r.shuffle(X);
    return { name, X, modes, sd, n };
  };

  // ---- small generated images ----
  G.images = (kind = 'digits', o = {}) => {
    const D = S.Data, size = o.size || 16, seed = o.seed || 1, want = o.classes || null;
    const n0 = o.n || 512, make = { digits: D.digits, shapes: D.shapes, icons: D.icons }[kind];
    if (!make) throw new Error('no image set called ' + kind);
    // draw more than needed when some classes are left out, then keep the wanted ones
    const k = kind === 'digits' ? 10 : kind === 'shapes' ? 4 : 6, over = want ? Math.ceil((n0 * k) / want.length) + k : n0;
    const ds = make({ n: over, size, seed });
    const keep = []; for (let i = 0; i < ds.n && keep.length < n0; i++) if (!want || want.includes(ds.y[i])) keep.push(i);
    const P = size * size, X = new Float32Array(keep.length * P);
    keep.forEach((i, j) => X.set(ds.get(i), j * P));
    return { X, y: keep.map((i) => ds.y[i]), n: keep.length, size, P, classes: ds.classes, kind };
  };
  /** rows idx of a flat image block as a tensor [b, P] (or [b, 1, s, s] when shape4) */
  G.batchOf = (set, idx, shape4) => {
    const T = S.T, P = set.P, t = T.zeros(shape4 ? [idx.length, 1, set.size, set.size] : [idx.length, P]);
    idx.forEach((i, j) => t.data.set(set.X.subarray(i * P, (i + 1) * P), j * P));
    return t;
  };
  G.rows = (X, idx) => S.T.tensor(idx.map((i) => X[i]));

  // ---- distances between point sets ----
  G.swd = (A, B, o = {}) => {
    const k = o.dirs || 32, proj = (P, c, s) => Float64Array.from(P, (p) => p[0] * c + p[1] * s).sort();
    let tot = 0;
    for (let d = 0; d < k; d++) {
      const th = (Math.PI * d) / k, c = Math.cos(th), s = Math.sin(th), a = proj(A, c, s), b = proj(B, c, s), m = Math.max(a.length, b.length);
      let sum = 0; for (let i = 0; i < m; i++) sum += Math.abs(a[Math.floor((i * a.length) / m)] - b[Math.floor((i * b.length) / m)]);
      tot += sum / m;
    }
    return tot / k;
  };
  G.chamfer = (A, B) => {
    let s = 0;
    for (const a of A) { let best = Infinity; for (const b of B) { const d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2; if (d < best) best = d; } s += Math.sqrt(best); }
    return s / A.length;
  };
  G.coverage = (P, modes, r) => {
    const counts = new Array(modes.length).fill(0), nearest = new Array(modes.length).fill(0);
    let near = 0;
    for (const p of P) {
      let b = -1, bd = Infinity;
      modes.forEach((m, i) => { const d = Math.hypot(p[0] - m[0], p[1] - m[1]); if (d < bd) { bd = d; b = i; } });
      nearest[b]++;
      if (bd <= r) { counts[b]++; near++; }
    }
    const fair = Math.max(1, 0.2 * (P.length / modes.length));
    // the effective number of modes: e to the entropy of the shares of samples nearest to each mode
    const covered = counts.filter((c) => c >= fair).length, effective = Math.exp(-nearest.reduce((h, c) => (c ? h + (c / P.length) * Math.log(c / P.length) : h), 0));
    return { covered, effective, of: modes.length, counts, nearest, quality: near / P.length,
      say: covered + ' of ' + modes.length + ' modes have a fair share of samples (at least a fifth of an equal share within ' + fmt(r) + '); ' + Math.round((100 * near) / P.length) + '% of samples lie near some mode, and the samples spread over the equivalent of ' + fmt(effective) + ' modes.' };
  };
  /** a sinusoidal embedding of the time step t (of Tmax), dim features: sin and cos at geometric frequencies */
  G.timeEmbed = (ts, dim, Tmax) => {
    const T = S.T, out = T.zeros([ts.length, dim]), half = dim / 2;
    ts.forEach((t, i) => { for (let j = 0; j < half; j++) { const f = Math.pow(1000, -j / Math.max(1, half - 1)), a = (t / Tmax) * 1000 * f; out.data[i * dim + j] = Math.sin(a); out.data[i * dim + half + j] = Math.cos(a); } });
    return out;
  };

  /** copy of a module's parameters (for snapshots and for target networks) */
  G.copyParams = (m) => m.parameters().map((p) => Float32Array.from(p.values()));
  G.setParams = (m, vals) => m.parameters().forEach((p, i) => p.data.set(vals[i]));

  /** a stepping trainer: core = {step(i) -> {loss, ...}, snapshot(i) -> object, snapEvery}; run(n) or step() */
  G.trainer = (core, o = {}) => {
    const tr = { i: 0, history: [], snapshots: [], core };
    const every = o.snapEvery || core.snapEvery || 50;
    tr.step = () => {
      if (tr.i === 0 && core.snapshot) tr.snapshots.push(Object.assign({ step: 0 }, core.snapshot(0)));
      const r = core.step(tr.i); tr.i++;
      tr.history.push(Object.assign({ step: tr.i }, r));
      if (core.snapshot && tr.i % every === 0) tr.snapshots.push(Object.assign({ step: tr.i }, core.snapshot(tr.i)));
      return r;
    };
    tr.run = (n) => { let r; for (let k = 0; k < n; k++) r = tr.step(); return r; };
    return tr;
  };
  /** a job for S.T.jobs from a trainer factory: {step -> {loss, metrics}, tensors} */
  G.asJob = (make) => (p) => { const tr = make(p); return { model: tr.model, step: () => { const r = tr.step(); return { loss: r.loss, metrics: r }; }, tensors: () => (tr.tensors ? tr.tensors() : {}), trainer: tr }; };
})(globalThis.Nooral = globalThis.Nooral || {});
