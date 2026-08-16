// Part 3, linear models: the families, and the helpers that parts 3, 4 and 5 share (numbers for narration,
// plot panels, a 3-D projection, matrices drawn as cells, arrows and small data sets of our own).
// Every number a frame quotes is put in its meta too, so tests/claims-p03.mjs (and p04, p05) can check it.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'linear', order, title, blurb });
  fam('p03-ls', 10, 'Least squares', 'The line that makes the squared residuals smallest, found by solving the normal equations, and seen as the projection of the targets onto the column space.');
  fam('p03-gd', 20, 'Gradient descent on the loss', 'The same least-squares loss as a surface over the weight and the intercept: the closed form jumps to its floor, gradient descent walks there.');
  fam('p03-poly', 30, 'Polynomial regression', 'Powers of x as extra features: a straight-line method that bends, and overfits when the degree grows past what the data support.');
  fam('p03-reg', 40, 'Regularisation', 'Ridge, lasso and elastic net: a penalty on the size of the weights, their coefficient paths as the penalty slides, and the geometry that makes lasso give exact zeros.');
  fam('p03-logit', 50, 'Logistic and softmax regression', 'A linear score squashed into a probability, trained on log loss by following its gradient; and softmax for many classes.');
  fam('p03-perc', 60, 'The perceptron', 'The oldest learning rule: on each mistake, add the example to the weights.');
  fam('p03-da', 70, 'Discriminant analysis', 'A Gaussian for each class: one shared covariance gives a straight boundary (LDA), one each gives a curved one (QDA).');

  const H = (S.PL = S.PL || {});
  const ML = () => S.ML;

  // ---- numbers in sentences -------------------------------------------------------------------------------
  /** n significant digits (3 by default), a true minus sign, whole numbers of 1000 and more rounded */
  H.n = (v, sig) => {
    if (!isFinite(v)) return String(v);
    if (Math.abs(v) < 5e-13) return '0';
    const a = Math.abs(v);
    let s = a >= 1000 ? String(Math.round(v)) : String(+(+v).toPrecision(sig || 3));
    if (/e/.test(s)) s = (+v).toFixed(Math.min(8, Math.max(0, 1 - Math.floor(Math.log10(a)) + (sig || 3) - 2)));
    return s.replace(/^-/, '−');
  };
  /** d decimals, a true minus */
  H.fix = (v, d) => { const s = (+v).toFixed(d === undefined ? 2 : d); return (/^-0\.?0*$/.test(s) ? s.slice(1) : s).replace(/^-/, '−'); };
  H.pct = (v) => Math.round(v * 100) + '%';
  H.plural = (k, word, many) => k + ' ' + (Number(k) === 1 ? word : many || word + 's');
  H.list = (xs) => { xs = xs.map(String); return xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]; };
  H.vec = (v, sig) => '(' + Array.from(v, (x) => H.n(x, sig)).join(', ') + ')';
  H.sup = (k) => String(k).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d] || d).join('');
  H.sub = (k) => String(k).split('').map((d) => '₀₁₂₃₄₅₆₇₈₉'[d] || d).join('');
  /** a signed term for a formula: " + 1.2x" or " − 0.4x" */
  H.term = (c, x, first, sig) => (first ? (c < 0 ? '−' : '') : c < 0 ? ' − ' : ' + ') + H.n(Math.abs(c), sig) + (x || '');
  /** a number written in a/an position: "an 8", "a 5" */
  H.an = (word) => (/^(8|11|18|8\d|8\.|[aeiou])/i.test(String(word)) ? 'an ' : 'a ') + word;

  // ---- stage items ----------------------------------------------------------------------------------------
  H.text = (key, x, y, text, cls, size, anchor) => { const it = { key, kind: 'text', x, y, text: String(text), cls: cls || 'strong' }; if (size) it.size = size; if (anchor) it.anchor = anchor; return it; };
  H.title = (key, x, y, text, anchor) => { const it = { key, kind: 'text', x, y, text, cls: 'title' }; if (anchor) it.anchor = anchor; return it; };
  H.seg = (key, x1, y1, x2, y2, cls, audit) => ({ key, kind: 'path', d: `M${r2(x1)},${r2(y1)} L${r2(x2)},${r2(y2)}`, cls: cls || 'pl-seg', pathAudit: audit || 'edge-x', layer: 'lines' });
  const r2 = (v) => Math.round(v * 100) / 100;
  H.r2 = r2;
  /** an arrow from (x1, y1) to (x2, y2) drawn as a path with its own head (so it may start anywhere) */
  H.arrow = (key, x1, y1, x2, y2, cls, o) => {
    o = o || {};
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, hl = Math.min(o.head || 10, L * 0.45), hw = hl * 0.5;
    const bx = x2 - ux * hl, by = y2 - uy * hl;
    const d = `M${r2(x1)},${r2(y1)} L${r2(bx)},${r2(by)} M${r2(bx + uy * hw)},${r2(by - ux * hw)} L${r2(x2)},${r2(y2)} L${r2(bx - uy * hw)},${r2(by + ux * hw)} Z`;
    return { key, kind: 'path', d, cls: 'pl-arrow ' + (cls || ''), pathAudit: o.audit || 'edge-x', layer: o.layer || 'lines', samples: [[x1, y1], [x2, y2]] };
  };
  /** a matrix of numbers in cells with brackets, top-left at (x, y): {items, w, h} */
  H.matrix = (key, x, y, rows, o) => {
    o = o || {};
    const cw = o.cw || 62, ch = o.ch || 28, fs = o.size || 14, items = [], nr = rows.length, nc = rows[0].length;
    rows.forEach((row, i) => row.forEach((v, j) => items.push({ key: key + ':' + i + '_' + j, kind: 'cell', x: r2(x + 8 + j * cw), y: r2(y + i * ch), w: cw - 4, h: ch - 4, rx: 4, text: typeof v === 'string' ? v : H.n(v, o.sig), size: fs, cls: 'pl-mcell ' + (o.cls ? o.cls(i, j) || '' : '') })));
    const W = nc * cw + 12, Ht = nr * ch - 4;
    items.push({ key: key + ':br', kind: 'path', d: `M${r2(x + 6)},${r2(y - 2)} h-5 v${r2(Ht + 4)} h5 M${r2(x + W - 6)},${r2(y - 2)} h5 v${r2(Ht + 4)} h-5`, cls: 'pl-bracket', audit: 'ignore', layer: 'lines' });
    if (o.name) items.push(Object.assign(H.text(key + ':nm', x - 8, y + Ht / 2 + 5, o.name, 'pl-mname', 15, 'end'), { mono: true }));
    return { items, w: W, h: Ht };
  };
  /** lines of formulas, each [text, cls], one under the other */
  H.eqs = (key, x, y, lines, size, gap) => lines.map((l, i) => Object.assign(H.text(key + ':' + i, x, y + i * (gap || 26), l[0], 'pl-eq ' + (l[1] || ''), size || 15), { mono: true }));
  /** a horizontal bar chart of signed values: bars from a zero line, names to the left, values to the right */
  H.hbars = (key, x, y, w, rowH, vals, names, o) => {
    o = o || {};
    const max = o.max || Math.max(1e-12, ...vals.map(Math.abs)), mid = x + w / 2, items = [];
    items.push({ key: key + ':zero', kind: 'path', d: `M${r2(mid)},${r2(y - 4)} v${r2(vals.length * rowH + 2)}`, cls: 'axis', audit: 'ignore', layer: 'back' });
    vals.forEach((v, i) => {
      const bw = Math.max(v === 0 ? 0 : 1, (Math.abs(v) / max) * (w / 2 - 2)), cy = y + i * rowH;
      if (bw > 0) items.push({ key: key + ':b' + i, kind: 'cell', x: r2(v >= 0 ? mid : mid - bw), y: r2(cy), w: r2(bw), h: rowH - 8, rx: 2, text: '', cls: 'bar ' + (o.cls ? o.cls(i, v) : v >= 0 ? 'pos' : 'neg'), audit: 'ignore' });
      if (names) items.push(H.text(key + ':n' + i, x - 8, cy + rowH / 2, names[i], 'tick ' + (o.nameCls ? o.nameCls(i) : ''), 12, 'end'));
      if (o.values !== false) items.push(H.text(key + ':v' + i, x + w + 8, cy + rowH / 2, v === 0 ? '0' : H.n(v), v === 0 ? 'pl-zero' : 'tick', 12));
    });
    return items;
  };
  /** the axes of a plot (the curve kit) */
  H.axes = (o) => S.Kit.curve.axes(o);
  /** a box-shaped legend of class swatches */
  H.classLegend = (key, x, y, names, o) => S.Kit.curve.legend(key, x, y, names.map((t, k) => ({ text: t, cls: 'c' + k, dot: true })), o);

  /**
   * Place labels clear of each other, of the boxes given and of the polylines given: each label is
   * {key, text, cls, size, at: [[x, y, anchor]..] candidate spots, need} and takes its first free spot; a label
   * with no free spot is dropped unless need is set (then it takes its first spot).
   */
  H.place = (labels, boxes, lines) => {
    const taken = (boxes || []).slice(), out = [];
    for (const L of labels) {
      let chosen = null;
      for (const [x, y, anchor] of L.at) {
        const it = H.text(L.key, x, y, L.text, L.cls, L.size, anchor || 'middle'), b = S.itemBox(it);
        const pad = { x: b.x - 3, y: b.y - 3, w: b.w + 6, h: b.h + 6 };
        if (taken.some((t) => S.geom.boxHit(t, pad)) || (lines || []).some((pl) => S.geom.polyBox(pl.map((p) => ({ x: p[0], y: p[1] })), pad))) continue;
        chosen = it; taken.push(pad); break;
      }
      if (!chosen && L.need) { const [x, y, anchor] = L.at[0]; chosen = H.text(L.key, x, y, L.text, L.cls, L.size, anchor || 'middle'); taken.push(S.itemBox(chosen)); }
      if (chosen) out.push(chosen);
    }
    return out;
  };

  // ---- a 3-D view -----------------------------------------------------------------------------------------
  /** a camera: azimuth az and elevation el in degrees, a scale and a centre; P([x, y, z]) -> [sx, sy] (z up) */
  H.view3 = (az, el, scale, cx, cy) => {
    const a = (az * Math.PI) / 180, e = (el * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a), ce = Math.cos(e), se = Math.sin(e);
    const P = (p) => { const x = p[0] * ca - p[1] * sa, yd = p[0] * sa + p[1] * ca; return [r2(cx + scale * x), r2(cy - scale * (p[2] * ce - yd * se))]; };
    P.depth = (p) => (p[0] * sa + p[1] * ca) * ce + p[2] * se;
    return P;
  };

  // ---- linear algebra checks used by several topics ---------------------------------------------------------
  /** solve the 2 × 2 or general normal equations X^T X w = X^T y with an intercept column, by Gauss-Jordan */
  H.normalEq = (X, y) => {
    const A = X.map((r) => [1, ...r]), d = A[0].length, G = ML().la.zeros(d, d), g = new Array(d).fill(0);
    A.forEach((r, i) => { for (let a = 0; a < d; a++) { g[a] += r[a] * y[i]; for (let b = 0; b < d; b++) G[a][b] += r[a] * r[b]; } });
    return { G, g, w: ML().la.solve(G, g) };
  };

  // ---- data -------------------------------------------------------------------------------------------------
  const cache = new Map();
  H.once = (key, fn) => { if (!cache.has(key)) { if (cache.size > 400) cache.clear(); cache.set(key, fn()); } return cache.get(key); };
  /** points of one input and a target along a line (or a curve), from a seed */
  H.lineData = (n, seed, slope, icpt, noise, outlier) => {
    const r = S.Data.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const x = S.Data.round(r.range(0.5, 9.5), 2); X.push([x]); y.push(S.Data.round(icpt + slope * x + r.normal(0, noise), 2)); }
    if (outlier) { X.push([8.8]); y.push(S.Data.round(icpt + slope * 8.8 - 3.2 * Math.max(1.4, noise * 3), 2)); }
    return { X, y };
  };
  /** two Gaussian classes in the plane: means m0, m1, covariances as [sx, sy, rho] */
  H.gauss2 = (n, seed, m0, m1, c0, c1) => {
    const r = S.Data.rng(seed), X = [], y = [];
    const draw = (m, c) => { const a = r.normal(0, 1), b = r.normal(0, 1); return [S.Data.round(m[0] + c[0] * a, 3), S.Data.round(m[1] + c[1] * (c[2] * a + Math.sqrt(1 - c[2] * c[2]) * b), 3)]; };
    for (let i = 0; i < n; i++) { const k = i % 2; X.push(draw(k ? m1 : m0, k ? c1 : c0)); y.push(k); }
    return { X, y };
  };
  /** a range around values with a margin */
  H.span = (vals, pad) => { let lo = Math.min(...vals), hi = Math.max(...vals); const p = (hi - lo || 1) * (pad === undefined ? 0.08 : pad); return [lo - p, hi + p]; };
  /** a nice range: the span rounded out to ticks */
  H.nice = (vals, pad) => { const [a, b] = H.span(vals, pad), t = S.Kit.curve.ticks(a, b, 5), st = t.length > 1 ? t[1] - t[0] : 1; return [Math.floor(a / st) * st, Math.ceil(b / st) * st]; };
  /** the part of the line a·u + b·v + c = 0 inside the box xr × yr: [[u, v], [u, v]] or null */
  H.lineInBox = (a, b, c, xr, yr) => {
    const pts = [];
    const add = (u, v) => { if (u >= xr[0] - 1e-9 && u <= xr[1] + 1e-9 && v >= yr[0] - 1e-9 && v <= yr[1] + 1e-9 && !pts.some((p) => Math.abs(p[0] - u) < 1e-9 && Math.abs(p[1] - v) < 1e-9)) pts.push([u, v]); };
    if (Math.abs(b) > 1e-12) { add(xr[0], -(a * xr[0] + c) / b); add(xr[1], -(a * xr[1] + c) / b); }
    if (Math.abs(a) > 1e-12) { add(-(b * yr[0] + c) / a, yr[0]); add(-(b * yr[1] + c) / a, yr[1]); }
    return pts.length >= 2 ? [pts[0], pts[1]] : null;
  };
  /** ticks every step from lo to hi (inclusive, rounded) */
  H.tickList = (lo, hi, step) => { const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out; };
  /** standardise columns: {X, mean, sd} */
  H.standardise = (X) => { const m = ML().colMeans(X), v = ML().colVars(X), sd = v.map((s) => Math.sqrt(s) || 1); return { X: X.map((r) => r.map((x, j) => (x - m[j]) / sd[j])), mean: m, sd }; };
  /** the ellipse of a 2 × 2 covariance at k standard deviations: {rx, ry, angle in degrees} */
  H.ellipse = (C, k) => {
    const a = C[0][0], b = C[0][1], d = C[1][1], tr = (a + d) / 2, det = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b));
    const l1 = tr + det, l2 = Math.max(1e-12, tr - det), ang = Math.abs(b) < 1e-15 ? (a >= d ? 0 : 90) : (Math.atan2(l1 - a, b) * 180) / Math.PI;
    return { rx: k * Math.sqrt(l1), ry: k * Math.sqrt(l2), angle: ang };
  };
  /** an ellipse in data coordinates drawn on axes as a path (axes may scale x and y differently) */
  H.ellipsePath = (ax, key, mean, C, k, cls, clip) => {
    const e = H.ellipse(C, k), t = (e.angle * Math.PI) / 180, pts = [], N = clip ? 200 : 64;
    for (let i = 0; i <= N; i++) { const u = (2 * Math.PI * i) / N, px = e.rx * Math.cos(u), py = e.ry * Math.sin(u); pts.push([ax.X(mean[0] + px * Math.cos(t) - py * Math.sin(t)), ax.Y(mean[1] + px * Math.sin(t) + py * Math.cos(t))]); }
    let d = '', pen = false;
    const o = ax.o, inBox = (p) => p[0] >= o.x - 0.5 && p[0] <= o.x + o.w + 0.5 && p[1] >= o.y - 0.5 && p[1] <= o.y + o.h + 0.5, kept = [];
    pts.forEach((p) => { if (clip && !inBox(p)) { pen = false; return; } d += (pen ? 'L' : 'M') + r2(p[0]) + ',' + r2(p[1]) + ' '; pen = true; kept.push(p); });
    if (!clip) d = d.trim() + ' Z';
    return { key, kind: 'path', d: d.trim() || 'M0,0', cls: 'pl-ell ' + (cls || ''), pathAudit: 'edge-x', layer: 'lines', samples: kept };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
