// Scatter plots and decision boundaries on the stage: points coloured by class, a raster of what a model says
// at every place in the plane (its probability, its class, any score), and level lines traced by marching squares
// (the boundary p = 0.5, an SVM's margins f = ±1). Works on axes from S.Kit.curve.axes.
//   S.Kit.scatter.points(ax, key, X, y, opts {r, cls(i) -> extra classes, only: [i..]}) -> dots: class k is 'c<k>'
//     (points may overlap one another: they are marked overlap-ok; labels are placed clear of them by the scene)
//   S.Kit.scatter.field(ax, key, f(u, v), opts {n, scheme, max, alpha}) -> a raster over the plot, f at cell centres
//   S.Kit.scatter.grid(ax, f, n) -> {nx, ny, vals} f sampled at the cell corners (for level lines)
//   S.Kit.scatter.level(ax, key, g, level, cls, opts) -> a path along f = level (g from grid)
//   S.Kit.scatter.ring(ax, key, u, v, cls, r) -> a ring round a point (a support vector, a mistake, the one picked)
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const SC = (K.scatter = {});
  const r2 = (v) => Math.round(v * 100) / 100;

  SC.points = function (ax, key, X, y, opts) {
    opts = opts || {};
    const out = [], idx = opts.only || X.map((_, i) => i);
    for (const i of idx) {
      const p = X[i];
      out.push({ key: key + ':' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: opts.r || 4.2, cls: 'sp c' + (y ? y[i] : 0) + ' ' + (opts.cls ? opts.cls(i) || '' : ''), audit: 'overlap-ok', layer: 'items' });
    }
    return out;
  };
  SC.ring = (ax, key, u, v, cls, r) => ({ key, kind: 'dot', x: ax.X(u), y: ax.Y(v), r: r || 8, cls: 'ring ' + (cls || ''), audit: 'overlap-ok', layer: 'items', z: 2 });

  SC.field = function (ax, key, f, opts) {
    opts = opts || {};
    const n = opts.n || 80, ny = Math.max(2, Math.round((n * ax.o.h) / ax.o.w)), vals = new Array(n * ny), vals2 = opts.scheme === 'classes' ? new Array(n * ny) : null;
    for (let j = 0; j < ny; j++) {
      const v = ax.V(ax.o.y + ((j + 0.5) / ny) * ax.o.h);
      for (let i = 0; i < n; i++) {
        const u = ax.U(ax.o.x + ((i + 0.5) / n) * ax.o.w), r = f(u, v);
        if (vals2) { vals[j * n + i] = r[0]; vals2[j * n + i] = r[1]; } else vals[j * n + i] = r;
      }
    }
    const it = { key, kind: 'raster', x: ax.o.x, y: ax.o.y, w: ax.o.w, h: ax.o.h, nx: n, ny, vals, scheme: opts.scheme || 'prob', max: opts.max, alpha: opts.alpha, smooth: opts.smooth !== false, layer: 'back', z: 0 };
    if (vals2) it.vals2 = vals2;
    return it;
  };

  /** f at the corners of an n-wide grid over the plot (rows from the top) */
  SC.grid = function (ax, f, n) {
    n = n || 70;
    const nx = n + 1, ny = Math.max(2, Math.round((n * ax.o.h) / ax.o.w)) + 1, vals = new Float64Array(nx * ny);
    for (let j = 0; j < ny; j++) { const v = ax.V(ax.o.y + (j / (ny - 1)) * ax.o.h); for (let i = 0; i < nx; i++) vals[j * nx + i] = f(ax.U(ax.o.x + (i / (nx - 1)) * ax.o.w), v); }
    return { nx, ny, vals, x: ax.o.x, y: ax.o.y, w: ax.o.w, h: ax.o.h };
  };
  /** the segments of f = level through a grid of corner values (marching squares), joined into polylines */
  SC.trace = function (g, level) {
    const { nx, ny, vals } = g, dx = g.w / (nx - 1), dy = g.h / (ny - 1);
    const at = (i, j) => vals[j * nx + i] - level;
    const segs = [];
    const cut = (i1, j1, i2, j2) => { const a = at(i1, j1), b = at(i2, j2), t = a / (a - b); return [g.x + (i1 + (i2 - i1) * t) * dx, g.y + (j1 + (j2 - j1) * t) * dy]; };
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const c = (at(i, j) > 0 ? 8 : 0) | (at(i + 1, j) > 0 ? 4 : 0) | (at(i + 1, j + 1) > 0 ? 2 : 0) | (at(i, j + 1) > 0 ? 1 : 0);
      if (c === 0 || c === 15) continue;
      const T = () => cut(i, j, i + 1, j), R = () => cut(i + 1, j, i + 1, j + 1), B = () => cut(i, j + 1, i + 1, j + 1), L = () => cut(i, j, i, j + 1);
      const mid = (at(i, j) + at(i + 1, j) + at(i + 1, j + 1) + at(i, j + 1)) / 4 > 0;
      switch (c) {
        case 1: case 14: segs.push([L(), B()]); break;
        case 2: case 13: segs.push([B(), R()]); break;
        case 3: case 12: segs.push([L(), R()]); break;
        case 4: case 11: segs.push([T(), R()]); break;
        case 6: case 9: segs.push([T(), B()]); break;
        case 7: case 8: segs.push([L(), T()]); break;
        case 5: if (mid) { segs.push([L(), T()], [B(), R()]); } else { segs.push([L(), B()], [T(), R()]); } break;
        case 10: if (mid) { segs.push([L(), B()], [T(), R()]); } else { segs.push([L(), T()], [B(), R()]); } break;
      }
    }
    // join segments that share ends into polylines
    const k = (p) => Math.round(p[0] * 20) + ',' + Math.round(p[1] * 20);
    const ends = new Map();
    segs.forEach((s, i) => { for (const e of [0, 1]) { const kk = k(s[e]); if (!ends.has(kk)) ends.set(kk, []); ends.get(kk).push(i); } });
    const used = new Uint8Array(segs.length), lines = [];
    for (let i = 0; i < segs.length; i++) {
      if (used[i]) continue;
      used[i] = 1;
      const line = [segs[i][0], segs[i][1]];
      for (const dir of [1, 0]) {
        for (;;) {
          const end = dir ? line[line.length - 1] : line[0];
          const next = (ends.get(k(end)) || []).find((m) => !used[m]);
          if (next === undefined) break;
          used[next] = 1;
          const s = segs[next], far = k(s[0]) === k(end) ? s[1] : s[0];
          if (dir) line.push(far); else line.unshift(far);
        }
      }
      lines.push(line);
    }
    return lines;
  };
  SC.level = function (ax, key, g, level, cls, opts) {
    opts = opts || {};
    const lines = SC.trace(g, level).filter((l) => l.length > (opts.minPts || 3));
    const d = lines.map((l) => l.map((p, i) => (i ? 'L' : 'M') + r2(p[0]) + ',' + r2(p[1])).join(' ')).join(' ');
    return { key, kind: 'path', d: d || 'M0,0', cls: 'level ' + (cls || ''), pathAudit: opts.audit || 'edge-x', layer: opts.layer || 'lines', samples: lines.flat() };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
