// Part 8, optimisation: the families, the loss surfaces (each with its exact gradient and Hessian), a runner that
// drives the engine's optimisers (S.T.optim) across a surface step by step, and the pieces of picture the topics
// share: the shaded contour map, a path clipped to the plot, a loss curve on a log scale.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'optimisation', order, title, blurb });
  fam('p08-descent', 10, 'Gradient descent', 'Loss surfaces seen from above and from the side, the gradient as the steepest way up, one step at a time downhill, how big a step can be, and the noisy steps of stochastic and mini-batch descent.');
  fam('p08-optimisers', 20, 'Better optimisers', 'Momentum and Nesterov, which remember where they were going; AdaGrad, RMSProp and Adam, which scale each coordinate’s step by its own gradients; and all six raced on the same ravines, saddles and bananas.');
  fam('p08-schedules', 30, 'Learning-rate schedules', 'Changing the learning rate as training goes: steps, exponential decay, the cosine curve, and a warm-up at the start.');
  fam('p08-landscape', 40, 'The shape of the landscape', 'Convex bowls where every way down leads to the bottom, bumpy surfaces with local minima that trap descent, and plateaus where the gradient all but vanishes.');

  const P = (S.P08 = {}), H = () => S.Kit.ml;
  const ex = Math.exp;

  // ---- surfaces: f(u, v), its gradient g and Hessian h, the box to draw, a start and the minima -----------------
  // a Gaussian well: depth a, centre (cu, cv), width s (for the bumpy surface)
  const well = (a, cu, cv, s) => ({
    f: (u, v) => -a * ex(-((u - cu) ** 2 + (v - cv) ** 2) / s),
    g: (u, v) => { const e = a * ex(-((u - cu) ** 2 + (v - cv) ** 2) / s); return [(2 * (u - cu) * e) / s, (2 * (v - cv) * e) / s]; },
    h: (u, v) => { const e = a * ex(-((u - cu) ** 2 + (v - cv) ** 2) / s), du = u - cu, dv = v - cv; return [[(2 * e) / s - (4 * du * du * e) / (s * s), (-4 * du * dv * e) / (s * s)], [(-4 * du * dv * e) / (s * s), (2 * e) / s - (4 * dv * dv * e) / (s * s)]]; },
  });
  const WELLS = [well(1.6, -1.3, -1.2, 1.2), well(1.0, 1.6, 1.1, 0.7), well(0.8, -1.4, 1.7, 0.6), well(0.55, 1.5, -1.6, 0.5)];
  P.SURF = {
    bowl: { label: 'a round bowl', formula: 'L(u, v) = ½(u² + v²)', f: (u, v) => 0.5 * (u * u + v * v), g: (u, v) => [u, v], h: () => [[1, 0], [0, 1]], xr: [-3, 3], yr: [-3, 3], start: [-2.4, 1.8], min: [[0, 0]] },
    ravine: { label: 'a long ravine', formula: 'L(u, v) = 0.1u² + 2v²', f: (u, v) => 0.1 * u * u + 2 * v * v, g: (u, v) => [0.2 * u, 4 * v], h: () => [[0.2, 0], [0, 4]], xr: [-4.5, 4.5], yr: [-3, 3], start: [-4, 1.6], min: [[0, 0]] },
    tilted: { label: 'a tilted valley', formula: 'L(u, v) = ½(1.2u² + 1.6uv + 1.2v²)', f: (u, v) => 0.5 * (1.2 * u * u + 1.6 * u * v + 1.2 * v * v), g: (u, v) => [1.2 * u + 0.8 * v, 0.8 * u + 1.2 * v], h: () => [[1.2, 0.8], [0.8, 1.2]], xr: [-3, 3], yr: [-3, 3], start: [-2.5, 0.6], min: [[0, 0]] },
    saddle: { label: 'a saddle', formula: 'L(u, v) = ½u² + ¼v⁴ − ½v²', f: (u, v) => 0.5 * u * u + 0.25 * v ** 4 - 0.5 * v * v, g: (u, v) => [u, v ** 3 - v], h: (u, v) => [[1, 0], [0, 3 * v * v - 1]], xr: [-2.2, 2.2], yr: [-1.8, 1.8], start: [-2, 0.02], min: [[0, 1], [0, -1]], saddle: [0, 0] },
    rosen: { label: 'a banana valley (Rosenbrock)', formula: 'L(u, v) = (1 − u)² + 20(v − u²)²', f: (u, v) => (1 - u) ** 2 + 20 * (v - u * u) ** 2, g: (u, v) => [-2 * (1 - u) - 80 * u * (v - u * u), 40 * (v - u * u)], h: (u, v) => [[2 - 80 * v + 240 * u * u, -80 * u], [-80 * u, 40]], xr: [-2, 2], yr: [-1, 3], start: [-1.5, 2.5], min: [[1, 1]] },
    bumpy: {
      label: 'a bumpy field', formula: 'L(u, v) = 0.06(u² + v²) − four Gaussian wells',
      f: (u, v) => 0.06 * (u * u + v * v) + WELLS.reduce((s, w) => s + w.f(u, v), 0),
      g: (u, v) => WELLS.reduce((s, w) => { const g = w.g(u, v); return [s[0] + g[0], s[1] + g[1]]; }, [0.12 * u, 0.12 * v]),
      h: (u, v) => WELLS.reduce((s, w) => { const h = w.h(u, v); return [[s[0][0] + h[0][0], s[0][1] + h[0][1]], [s[1][0] + h[1][0], s[1][1] + h[1][1]]]; }, [[0.12, 0], [0, 0.12]]),
      xr: [-3, 3], yr: [-3, 3], start: [0.2, 0.3], min: [],
    },
    plateau: { label: 'a plateau round a pit', formula: 'L(u, v) = 1 − exp(−(u² + 2v²)/2)', f: (u, v) => 1 - ex(-(u * u + 2 * v * v) / 2), g: (u, v) => { const e = ex(-(u * u + 2 * v * v) / 2); return [u * e, 2 * v * e]; }, h: (u, v) => { const e = ex(-(u * u + 2 * v * v) / 2); return [[e * (1 - u * u), -2 * u * v * e], [-2 * u * v * e, e * (2 - 4 * v * v)]]; }, xr: [-4, 4], yr: [-2.6, 2.6], start: [3.6, 0.9], min: [[0, 0]] },
  };
  P.surfOptions = (ids) => ids.map((id) => ({ value: id, label: P.SURF[id].label }));
  /** eigenvalues of a symmetric 2 × 2 matrix, largest first */
  P.eig2 = (m) => { const t = (m[0][0] + m[1][1]) / 2, d = Math.sqrt(((m[0][0] - m[1][1]) / 2) ** 2 + m[0][1] * m[1][0]); return [t + d, t - d]; };

  // ---- the optimisers ---------------------------------------------------------------------------------------------
  P.OPT = [
    { id: 'sgd', name: 'gradient descent', short: 'GD', cls: 'o-gd' },
    { id: 'momentum', name: 'momentum', short: 'momentum', cls: 'o-mom' },
    { id: 'nesterov', name: 'Nesterov', short: 'Nesterov', cls: 'o-nest' },
    { id: 'adagrad', name: 'AdaGrad', short: 'AdaGrad', cls: 'o-ada' },
    { id: 'rmsprop', name: 'RMSProp', short: 'RMSProp', cls: 'o-rms' },
    { id: 'adam', name: 'Adam', short: 'Adam', cls: 'o-adam' },
  ];
  P.opt = (id) => P.OPT.find((o) => o.id === id);
  /**
   * Run an engine optimiser on a surface: name ('sgd', 'momentum', ...), opts for its constructor ({lr, momentum,
   * betas, ...}), steps. grad(p, k) gives the gradient used at step k (default: the surface's own), schedule a
   * function of the step. Returns {pts, loss, grads, lrs, state: [buffers after each step], diverged}.
   */
  P.run = function (surf, name, opts, steps, o) {
    o = o || {};
    const T = S.T, start = o.start || surf.start;
    return T.float64(() => {
      const p = T.tensor(start.slice()), opt = T.optim.make(name, [p], opts);
      if (o.schedule) opt.schedule = o.schedule;
      const pts = [start.slice()], loss = [surf.f(start[0], start[1])], grads = [], lrs = [], state = [];
      let diverged = false;
      for (let k = 0; k < steps; k++) {
        const [u, v] = p.values(), g = o.grad ? o.grad([u, v], k) : surf.g(u, v);
        grads.push(g.slice());
        p.grad = T.tensor(g);
        const r = opt.step();
        lrs.push(r.lr);
        const s = opt.state.get(p) || {}, snap = {};
        for (const key of Object.keys(s)) if (ArrayBuffer.isView(s[key])) snap[key] = Array.from(s[key]);
        state.push(snap);
        const q = Array.from(p.values());
        pts.push(q);
        loss.push(surf.f(q[0], q[1]));
        if (!q.every(Number.isFinite) || Math.abs(q[0]) > 1e6 || Math.abs(q[1]) > 1e6) { diverged = true; break; }
      }
      return { pts, loss, grads, lrs, state, diverged };
    });
  };

  // ---- pictures ---------------------------------------------------------------------------------------------------
  const cache = new Map();
  /** the shaded contour map of a surface on axes ax (cached: the same surface and box give the same items) */
  P.contour = function (ax, key, surf, o) {
    o = o || {};
    const k = key + '|' + JSON.stringify([ax.o.xr, ax.o.yr, ax.o.x, ax.o.y, ax.o.w, ax.o.h, surf.formula, o.levels]);
    if (!cache.has(k)) cache.set(k, S.Kit.contour.surface(ax, key, surf.f, { levels: o.levels || 10, n: o.n || 90, depth: 2.2, min: o.min === undefined ? P.gridMin(surf) : o.min }));
    return cache.get(k);
  };
  /** the lowest value of a surface over its box (on a fine grid) */
  P.gridMin = (surf) => { let m = Infinity; for (let i = 0; i <= 120; i++) for (let j = 0; j <= 120; j++) m = Math.min(m, surf.f(surf.xr[0] + ((surf.xr[1] - surf.xr[0]) * i) / 120, surf.yr[0] + ((surf.yr[1] - surf.yr[0]) * j) / 120)); return m; };
  P.axes = (prefix, surf, box, o) => S.Kit.curve.axes(Object.assign({ prefix, xr: surf.xr, yr: surf.yr, xname: 'u', yname: 'v' }, box, o || {}));
  /** a path over the axes, clipped to the plot (a run that leaves it and comes back shows as two pieces), with a
   *  dot at its head when the head is on the plot */
  P.path = function (ax, key, pts, cls, upto, o) {
    o = o || {};
    const n = Math.min(pts.length - 1, upto === undefined ? pts.length - 1 : upto);
    const [x0, x1] = ax.o.xr, [y0, y1] = ax.o.yr;
    const inside = (p) => p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1;
    // Liang–Barsky: the part of segment a–b inside the box, or null
    const clip = (a, b) => {
      let t0 = 0, t1 = 1;
      const dx = b[0] - a[0], dy = b[1] - a[1];
      for (const [pp, qq] of [[-dx, a[0] - x0], [dx, x1 - a[0]], [-dy, a[1] - y0], [dy, y1 - a[1]]]) {
        if (pp === 0) { if (qq < 0) return null; continue; }
        const r = qq / pp;
        if (pp < 0) { if (r > t1) return null; if (r > t0) t0 = r; } else { if (r < t0) return null; if (r < t1) t1 = r; }
      }
      return [[a[0] + t0 * dx, a[1] + t0 * dy], [a[0] + t1 * dx, a[1] + t1 * dy]];
    };
    let d = '', pen = null;
    const all = [];
    for (let k = 1; k <= n; k++) {
      const a = pts[k - 1], b = pts[k];
      if (!b.every(Number.isFinite)) break;
      const c = clip(a, b);
      if (!c) { pen = null; continue; }
      const A = [ax.X(c[0][0]), ax.Y(c[0][1])], B = [ax.X(c[1][0]), ax.Y(c[1][1])];
      if (!pen || Math.abs(pen[0] - A[0]) > 0.01 || Math.abs(pen[1] - A[1]) > 0.01) d += 'M' + A[0] + ',' + A[1] + ' ';
      d += 'L' + B[0] + ',' + B[1] + ' ';
      pen = B; all.push(A, B);
    }
    const items = [];
    if (d) items.push({ key: key + ':p', kind: 'path', d: d.trim(), cls: 'optpath ' + cls, pathAudit: 'edge-x', layer: 'lines', box: H().boxOf(all) });
    if (o.dots) for (let k = 1; k < n; k++) if (inside(pts[k])) items.push({ key: key + ':d' + k, kind: 'dot', x: ax.X(pts[k][0]), y: ax.Y(pts[k][1]), r: 2.6, cls: 'optstep ' + cls, audit: 'ignore' });
    const h = pts[n];
    if (inside(h)) items.push({ key: key + ':h', kind: 'dot', x: ax.X(h[0]), y: ax.Y(h[1]), r: o.r || 6, cls: 'opthead ' + cls, audit: 'overlap-ok', layer: 'items' });
    return items;
  };
  /** whether the head of the path is off the plot at step k */
  P.off = (ax, pts, k) => { const p = pts[Math.min(k, pts.length - 1)]; return !(p[0] >= ax.o.xr[0] && p[0] <= ax.o.xr[1] && p[1] >= ax.o.yr[0] && p[1] <= ax.o.yr[1]); };
  /** whether the path has left the plot by step k */
  P.left = (ax, pts, k) => pts.slice(0, k + 1).some((p) => p[0] < ax.o.xr[0] || p[0] > ax.o.xr[1] || p[1] < ax.o.yr[0] || p[1] > ax.o.yr[1]);
  /** a loss curve on log axes: lax from P.logAxes, values clipped to its range */
  P.logAxes = function (prefix, box, steps, lo, hi, o) {
    const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi));
    const sup = (e) => (e === 0 ? '1' : e === 1 ? '10' : '10' + [...String(e)].map((c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]).join(''));
    const every = Math.max(1, Math.ceil((e1 - e0) / 5)), ticks = [];
    for (let e = e0; e <= e1; e += every) ticks.push(e);
    return S.Kit.curve.axes(Object.assign({ prefix, xr: [0, steps], yr: [e0, e1], yticks: ticks, xticks: S.Kit.curve.ticks(0, steps, 4), fmtY: sup, xname: 'step', yname: 'loss (log scale)' }, box, o || {}));
  };
  P.logLine = (lax, key, loss, cls, upto) => {
    const [lo, hi] = lax.o.yr;
    const pts = [];
    for (let k = 0; k < loss.length && (upto === undefined || k <= upto); k++) { const v = loss[k]; if (!Number.isFinite(v)) break; pts.push([k, Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-300))))]); }
    return S.Kit.curve.line(lax, key, pts, cls);
  };
  /** a legend row: a swatch line and a name for each run */
  P.legend = (key, x, y, entries, o) => S.Kit.curve.legend(key, x, y, entries, Object.assign({ dir: 'col', size: 13 }, o || {}));
  P.start = (ax, key, p) => ({ key, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 5, cls: 'goal', point: true, audit: 'overlap-ok' });
  P.minDot = (ax, key, p) => ({ key, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 6, cls: 'goal', point: true, audit: 'overlap-ok' });
  /** an arrow from stage point a to b as one filled path (shaft and head); null when it is too short to draw */
  P.arrow = function (key, a, b, cls, o) {
    o = o || {};
    const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
    if (L < (o.min || 4)) return null;
    const hl = Math.min(o.head || 10, L * 0.6), hw = (o.head || 10) * 0.45, ux = dx / L, uy = dy / L, r = (v) => Math.round(v * 100) / 100;
    const bx = b[0] - ux * hl, by = b[1] - uy * hl;
    const d = `M${r(a[0])},${r(a[1])} L${r(bx)},${r(by)} M${r(b[0])},${r(b[1])} L${r(bx - uy * hw)},${r(by + ux * hw)} L${r(bx + uy * hw)},${r(by - ux * hw)} Z`;
    return { key, kind: 'path', d, cls: 'p8-arrow ' + (cls || ''), audit: 'overlap-ok', layer: 'items', z: 3, box: H().boxOf([a, b]) };
  };
  /** the floor of a 3-D box (z = 0) with u and v named along its front edges, and the loss axis up its back corner */
  P.box3 = function (key, V, top) {
    const C = [[-1, -1], [1, -1], [1, 1], [-1, 1]], items = [], pr = C.map((c) => V([c[0], c[1], 0]));
    let back = 0;
    pr.forEach((p, k) => { if (p.d < pr[back].d) back = k; });
    items.push({ key: key + ':floor', kind: 'path', d: pr.map((p, k) => (k ? 'L' : 'M') + p.x + ',' + p.y).join(' ') + ' Z', cls: 'axis', audit: 'ignore', layer: 'back', z: 1, box: H().boxOf(pr) });
    const b = C[back], t = V([b[0], b[1], top || 1.1]);
    items.push({ key: key + ':up', kind: 'path', d: `M${pr[back].x},${pr[back].y} L${t.x},${t.y}`, cls: 'axis', audit: 'ignore', layer: 'back', z: 1, box: H().boxOf([pr[back], t]) });
    items.push(H().text(key + ':nl', t.x, t.y - 10, 'loss', 'axname', 13, 'middle'));
    // the front edges: the two that do not touch the back corner, named at their middles, pushed outwards
    const cx = pr.reduce((s, p) => s + p.x, 0) / 4, cy = pr.reduce((s, p) => s + p.y, 0) / 4;
    for (const [i, j, name] of [[0, 1, 'u'], [1, 2, 'v'], [2, 3, 'u'], [3, 0, 'v']]) {
      if (i === back || j === back) continue;
      const mx = (pr[i].x + pr[j].x) / 2, my = (pr[i].y + pr[j].y) / 2, dx = mx - cx, dy = my - cy, L = Math.hypot(dx, dy) || 1;
      items.push(H().text(key + ':n' + name, mx + (dx / L) * 18, my + (dy / L) * 18 + 4, name, 'axname', 13, 'middle'));
    }
    return items;
  };
  P.dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
})(globalThis.Nooral = globalThis.Nooral || {});
