// Part 6, support vector machines: the families, the datasets the topics use, and the pieces of picture they
// share (a fitted SVC, its decision field, its boundary and margins, and the support vectors ringed).
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'svm', order, title, blurb });
  fam('p06-margin', 10, 'The margin', 'The widest street between two classes, the support vectors that hold it up, the soft margin that lets some points in, and the hinge loss that prices them.');
  fam('p06-kernels', 20, 'Kernels', 'A curved boundary from a straight one: lift the points into more dimensions, cut there with a plane, and let the kernel do the lifting without ever building the new features.');
  fam('p06-solver', 30, 'The dual and SMO', 'The weights as a sum over support vectors, the dual problem in the multipliers α, and the SMO solver improving two multipliers at a time.');
  fam('p06-more', 40, 'Beyond two classes', 'Many classes from binary machines (one-vs-rest and one-vs-one), regression with an ε-tube, and the one-class machine that draws a border round normal data.');

  const ML = () => S.ML, H = () => S.Kit.ml;
  const P = (S.P06 = {});

  // ---- datasets: every set is generated from its seed by S.Data, or is a classic open set -----------------------
  P.SETS = {
    apart: { label: 'two groups, well apart', make: (o) => S.Data.blobs({ n: o.n || 24, centers: [[-1.3, -0.9], [1.3, 0.9]], std: 0.55, seed: o.seed }) },
    close: { label: 'two groups, close', make: (o) => S.Data.blobs({ n: o.n || 24, centers: [[-0.9, -0.6], [0.9, 0.6]], std: 0.55, seed: o.seed }) },
    overlap: { label: 'two groups that overlap', make: (o) => S.Data.blobs({ n: o.n || 40, centers: [[-0.7, -0.5], [0.7, 0.5]], std: 0.75, seed: o.seed }) },
    line: { label: 'a noisy line', make: (o) => S.Data.linear({ n: o.n || 40, noise: 0.08, seed: o.seed }) },
    moons: { label: 'two moons', make: (o) => S.Data.moons({ n: o.n || 60, noise: 0.15, seed: o.seed }) },
    circles: { label: 'a ring round a disc', make: (o) => S.Data.circles({ n: o.n || 60, noise: 0.08, factor: 0.45, seed: o.seed }) },
    xor: { label: 'XOR corners', make: (o) => S.Data.xor({ n: o.n || 48, noise: 0, seed: o.seed }) },
    iris: { label: 'iris petals: versicolor and virginica', make: () => P.iris2() },
  };
  /** two iris species (versicolor 0, virginica 1) by petal length and width, standardised */
  P.iris2 = function () {
    const d = S.Data.load('iris'), rows = [];
    d.X.forEach((x, i) => { if (d.y[i] > 0) rows.push([x[2], x[3], d.y[i] - 1]); });
    const X = rows.map((r) => [r[0], r[1]]), m = ML().colMeans(X), sd = ML().colVars(X).map(Math.sqrt);
    return { X: X.map((r) => r.map((v, j) => +((v - m[j]) / sd[j]).toFixed(4))), y: rows.map((r) => r[2]), names: ['petal length (standardised)', 'petal width (standardised)'] };
  };
  P.data = (id, o) => { const s = P.SETS[id]; return s.make(o || {}); };
  /** every machine a run fits, with its data and parameters, kept for the claims check (it refits them elsewhere) */
  P.log = null;
  P.record = (kind, X, y, params, m, Xe) => {
    if (!P.log) return;
    const e = { kind, X, y, params: Object.assign({}, params), support: (m.support || []).slice().sort((a, b) => a - b) };
    if (Xe) { e.Xe = Xe; e.decE = m.classes.length === 2 ? m.decisionFunction(Xe) : m.machineValues(Xe); }
    if (kind === 'svc') { e.pred = m.predict(X); e.dec = m.classes.length === 2 ? m.decisionFunction(X) : m.machineValues(X); }
    else if (kind === 'svr') e.pred = m.predict(X);
    else e.dec = m.decisionFunction(X);
    P.log.push(e);
  };
  P.setOptions = (ids) => ids.map((id) => ({ value: id, label: P.SETS[id].label }));
  /** the range of a dataset's points, padded, for the axes */
  P.range = function (X, pad) {
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const x of X) for (let j = 0; j < 2; j++) { lo[j] = Math.min(lo[j], x[j]); hi[j] = Math.max(hi[j], x[j]); }
    const p = pad === undefined ? 0.35 : pad;
    return { xr: [lo[0] - p, hi[0] + p], yr: [lo[1] - p, hi[1] + p] };
  };

  /**
   * A binary SVC fitted by the engine, read the way the pictures need it: f(x) > 0 means class 1; alpha[i] is
   * point i's multiplier (0 for the rest); for a linear kernel, w and b with f(x) = w·x + b.
   */
  P.fit = function (X, y, params, Xe) {
    const m = new (ML().SVC)(Object.assign({ kernel: 'linear', C: 1, tol: 1e-4 }, params)).fit(X, y);
    P.record('svc', X, y, m.params, m, Xe);
    const mc = m.machines[0], alpha = new Array(X.length).fill(0), sign = new Array(X.length).fill(0);
    mc.sv.forEach((i, t) => { alpha[i] = Math.abs(mc.coef[t]); sign[i] = Math.sign(mc.coef[t]); });
    // f(x) straight from the machine's support vectors (the same sum the engine's decision function makes)
    const svX = mc.sv.map((i) => X[i]), kf = m.kf, rho = mc.rho, cf = mc.coef;
    const f = (x) => { let s = -rho; for (let t = 0; t < svX.length; t++) s += cf[t] * kf(svX[t], x); return -s; };
    const out = { m, f, alpha, sv: mc.sv.slice().sort((a, b) => a - b), C: m.params.C, iters: mc.iters, gamma: m.gammaValue, kernel: m.params.kernel };
    if (m.params.kernel === 'linear') {
      // the engine's machine says +1 for the first class; the picture wants class 1 positive
      const w = new Array(X[0].length).fill(0);
      mc.sv.forEach((i, t) => { for (let j = 0; j < w.length; j++) w[j] -= mc.coef[t] * X[i][j]; });
      out.w = w; out.b = mc.rho; out.norm = Math.hypot(...w); out.width = 2 / out.norm;
    }
    const F = X.map(f);
    out.F = F;
    out.yf = F.map((v, i) => (y[i] === 1 ? 1 : -1) * v);
    out.acc = ML().accuracy(y, m.predict(X));
    // support vectors on the margin (0 < α < C) and at the cap (α = C)
    out.free = out.sv.filter((i) => alpha[i] < out.C * (1 - 1e-6));
    out.bound = out.sv.filter((i) => alpha[i] >= out.C * (1 - 1e-6));
    out.slack = out.yf.map((v) => Math.max(0, 1 - v));
    return out;
  };

  /** the decision field, the boundary f = 0 and the margins f = ±1 of a fitted machine */
  P.boundary = function (ax, key, fit, o) {
    o = o || {};
    const SC = S.Kit.scatter, items = [];
    // f once, at the corners of a grid: the level lines use the corners, the shading the mean of each cell's four
    const g = SC.grid(ax, (u, v) => fit.f([u, v]), o.n || 90);
    if (o.field !== false) {
      const nx = g.nx - 1, ny = g.ny - 1, vals = new Array(nx * ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const a = g.vals, w = g.nx; vals[j * nx + i] = ML().sigmoid(1.6 * (a[j * w + i] + a[j * w + i + 1] + a[(j + 1) * w + i] + a[(j + 1) * w + i + 1]) / 4); }
      items.push({ key: key + ':f', kind: 'raster', x: ax.o.x, y: ax.o.y, w: ax.o.w, h: ax.o.h, nx, ny, vals, scheme: 'prob', smooth: true, layer: 'back', z: 0 });
    }
    if (o.margins !== false) items.push(SC.level(ax, key + ':m0', g, -1, 'margin'), SC.level(ax, key + ':m1', g, 1, 'margin'));
    items.push(SC.level(ax, key + ':b', g, 0, ''));
    return items;
  };
  /** points by class, and a ring round each support vector (accent on the margin, plain at the cap) */
  P.points = function (ax, key, X, y, fit, o) {
    o = o || {};
    const SC = S.Kit.scatter, items = SC.points(ax, key + ':p', X, y, { r: o.r || 5, cls: o.cls });
    if (fit && o.rings !== false) for (const i of fit.sv) items.push(SC.ring(ax, key + ':r' + i, X[i][0], X[i][1], fit.alpha[i] < fit.C * (1 - 1e-6) ? 'act' : '', (o.r || 5) + 4));
    return items;
  };
  /** a legend of the two rings */
  P.ringLegend = function (key, x, y, freeOnly) {
    const SC = S.Kit.scatter, fake = (yy) => ({ X: () => x + 8, Y: () => yy });
    const out = [SC.ring(fake(y - 4.5), key + ':r1', 0, 0, 'act', 7), H().text(key + ':t1', x + 24, y, freeOnly ? 'support vector: on the margin' : 'on the margin: 0 < α < C', 'legend', 13)];
    if (!freeOnly) out.push(SC.ring(fake(y + 19.5), key + ':r2', 0, 0, '', 7), H().text(key + ':t2', x + 24, y + 24, 'at the cap: α = C', 'legend', 13));
    return out;
  };
  /** standard axes for a 2-D plot of the data */
  P.axes = (prefix, X, box, o) => {
    o = o || {};
    const r = o.xr ? { xr: o.xr.slice(), yr: o.yr.slice() } : P.range(X, o.pad);
    // the same scale on both axes, so a perpendicular looks perpendicular and a street's width is true
    if (o.equal !== false) {
      const sx = (r.xr[1] - r.xr[0]) / box.w, sy = (r.yr[1] - r.yr[0]) / box.h, s = Math.max(sx, sy);
      const grow = (rg, len) => { const m = (rg[0] + rg[1]) / 2; return [m - (s * len) / 2, m + (s * len) / 2]; };
      r.xr = grow(r.xr, box.w); r.yr = grow(r.yr, box.h);
    }
    return S.Kit.curve.axes(Object.assign({ prefix, xr: r.xr, yr: r.yr, xname: o.xname || 'x₁', yname: o.yname || 'x₂' }, box));
  };
})(globalThis.Nooral = globalThis.Nooral || {});
// the straight line n·x = c across the axes' box, as a path ('level' look, cls 'margin' for a dashed one)
(function (S) {
  'use strict';
  const P = S.P06;
  P.clipLine = function (ax, n, c) {
    const [x0, x1] = ax.o.xr, [y0, y1] = ax.o.yr, pts = [];
    if (Math.abs(n[1]) > 1e-12) for (const x of [x0, x1]) { const y = (c - n[0] * x) / n[1]; if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]); }
    if (Math.abs(n[0]) > 1e-12) for (const y of [y0, y1]) { const x = (c - n[1] * y) / n[0]; if (x >= x0 - 1e-9 && x <= x1 + 1e-9) pts.push([x, y]); }
    pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    return pts.length >= 2 ? [pts[0], pts[pts.length - 1]] : null;
  };
  P.line = function (ax, key, n, c, cls) {
    const s = P.clipLine(ax, n, c);
    const d = s ? `M${ax.X(s[0][0])},${ax.Y(s[0][1])} L${ax.X(s[1][0])},${ax.Y(s[1][1])}` : 'M0,0';
    return { key, kind: 'path', d, cls: 'level ' + (cls || ''), pathAudit: 'edge-x', layer: 'lines' };
  };
  /** the widest street with normal direction angle th: {n, c, h (half-width), lo, hi (the limiting points)} or null */
  P.streetAt = function (X, y, th) {
    const n = [Math.cos(th), Math.sin(th)];
    let max0 = -Infinity, min1 = Infinity, lo = -1, hi = -1;
    X.forEach((x, i) => { const s = n[0] * x[0] + n[1] * x[1]; if (y[i] === 0 && s > max0) { max0 = s; lo = i; } if (y[i] === 1 && s < min1) { min1 = s; hi = i; } });
    if (!(min1 > max0)) return null;
    return { th, n, c: (min1 + max0) / 2, h: (min1 - max0) / 2, lo, hi };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
