// Helpers for the topics of parts 6 to 8 (support vector machines, unsupervised learning, optimisation): numbers
// for narration that remember the engine value they show (the claims check reads them back), a frame builder
// that keeps them with the frame, small plot pieces (a titled readout, a bar row, a covariance ellipse, an
// arrow) and a 3-D view (points and a shaded surface seen from an angle, painted back to front).
//   const H = S.Kit.ml
//   H.fx(v, d) fixed decimals with a true minus;  H.sg(v, n) n significant digits;  H.pct(v) a percentage
//   const F = H.frames(counters)   F.q(v, d) / F.q.sg(v, n) / F.q.n(k) / F.q.pct(v) / F.q.t(v, d, first) (a signed term): a value as text for the
//     narration, remembered;  F.push(items, say, meta, extra) adds a frame whose meta.quotes are those values
//   H.text(key, x, y, text, cls, size, anchor)  H.lines(prefix, x, y, [[text, cls]..], size, gap)
//   H.ellipse(ax, key, mean, cov, k, cls) the k-standard-deviation ellipse of a 2-D Gaussian, as a path
//   H.arrow(key, [x1, y1], [x2, y2], cls) an arrow between two stage points
//   H.view3({cx, cy, s, az, el}) -> P([x, y, z]) -> {x, y, d} (d: depth, larger is nearer)
//   H.surface3(key, V, f, {xr, yr, n, zmin, zmax, zs}) -> shaded quads, far ones first
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const H = (K.ml = {});
  const r2 = (v) => Math.round(v * 100) / 100;
  const neg0 = (s) => (/^-0(\.0*)?$/.test(s) ? s.slice(1) : s);
  H.r2 = r2;
  /** the box round a list of stage points (a path's box, so the camera counts it) */
  H.boxOf = (pts) => { const xs = pts.map((p) => p.x === undefined ? p[0] : p.x), ys = pts.map((p) => p.y === undefined ? p[1] : p.y), x = Math.min(...xs), y = Math.min(...ys); return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y }; };
  H.fx = (v, d) => neg0((+v).toFixed(d === undefined ? 2 : d)).replace(/^-/, '−');
  H.sg = (v, n) => { if (!isFinite(v)) return String(v); const a = Math.abs(v); if (a >= 1e5 || (a < 1e-3 && a > 0)) { const [m, e] = (+v).toExponential((n || 3) - 1).split('e'); return m.replace('-', '−') + '×10' + [...String(+e)].map((c) => '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'['-0123456789'.indexOf(c)]).join(''); } return neg0(String(+(+v).toPrecision(n || 3))).replace(/^-/, '−'); };
  H.pct = (v) => Math.round(v * 100) + '%';
  /** "1 point" / "3 points" */
  H.pl = (k, w, many) => k + ' ' + (k === 1 ? w : many || w + 's');
  /** "a, b and c" */
  H.list = (xs) => (xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]);

  /** a quoter: each call gives the value as text and remembers {v, s} until take() (the claims check reads them) */
  H.quoter = function () {
    let list = [];
    const keep = (v, s) => { list.push({ v: +v, s }); return s; };
    const q = (v, d) => keep(v, H.fx(v, d));
    q.sg = (v, n) => keep(v, H.sg(v, n));
    q.n = (k) => keep(k, String(k));
    q.pct = (v) => keep(v, H.pct(v));
    /** a signed term of a sum: " + 0.31" / " − 0.03" (first: "0.31" / "−0.03"), its sign checked by the claims */
    q.t = (v, d, first) => { const s = H.fx(Math.abs(v), d); list.push({ v: +v, s, signed: true }); return (v < 0 && Math.abs(+H.fx(Math.abs(v), d)) > 0 ? (first ? '−' : ' − ') : first ? '' : ' + ') + s; };
    q.take = () => { const out = list; list = []; return out; };
    return q;
  };
  H.frames = function (counters) {
    const F = new S.Frames(counters || []);
    F.q = H.quoter();
    F.push = (items, say, meta, extra) => F.add(Object.assign({ items, say, meta: Object.assign({}, meta || {}, { quotes: F.q.take() }) }, extra || {}));
    return F;
  };

  H.text = (key, x, y, text, cls, size, anchor) => { const it = { key, kind: 'text', x: r2(x), y: r2(y), text, cls: cls || 'legend' }; if (size) it.size = size; if (anchor) it.anchor = anchor; return it; };
  H.lines = (prefix, x, y, rows, size, gap) => rows.map((r, i) => H.text(prefix + ':' + i, x, y + i * (gap || 22), r[0], r[1] || 'legend', size || 13));
  H.head = (key, x, y, text) => H.text(key, x, y, text, 'strong', 15);

  /** the k-sigma ellipse of a 2-D Gaussian with covariance [[a, b], [b, c]] around mean, on axes ax */
  H.ellipse = function (ax, key, mean, cov, k, cls) {
    const a = cov[0][0], b = cov[0][1], c = cov[1][1], tr = (a + c) / 2, det = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b));
    const l1 = tr + det, l2 = Math.max(0, tr - det), th = Math.abs(b) < 1e-15 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
    const pts = [];
    for (let i = 0; i <= 64; i++) {
      const t = (2 * Math.PI * i) / 64, u = k * Math.sqrt(l1) * Math.cos(t), v = k * Math.sqrt(l2) * Math.sin(t);
      pts.push([mean[0] + u * Math.cos(th) - v * Math.sin(th), mean[1] + u * Math.sin(th) + v * Math.cos(th)]);
    }
    const sp = pts.map((p) => [ax.X(p[0]), ax.Y(p[1])]);
    return { key, kind: 'path', d: sp.map((p, i) => (i ? 'L' : 'M') + p[0] + ',' + p[1]).join(' ') + ' Z', cls: 'mlell ' + (cls || ''), pathAudit: 'edge-x', layer: 'lines', axes: [l1, l2], angle: th, box: H.boxOf(sp) };
  };
  H.arrow = (key, a, b, cls) => ({ key, kind: 'edge', from: { x: r2(a[0]), y: r2(a[1]) }, to: { x: r2(b[0]), y: r2(b[1]) }, cls: 'mlarrow ' + (cls || ''), head: true, audit: 'edge-x' });
  /** a level line of f = level over the axes (S.Kit.scatter.level on a fresh grid) */
  H.level = (ax, key, f, level, cls, n) => S.Kit.scatter.level(ax, key, S.Kit.scatter.grid(ax, f, n || 80), level, cls);

  // ---- 3-D ----------------------------------------------------------------------------------------------------
  /** a view from azimuth az and elevation el (radians): x to the right, z up */
  H.view3 = function (o) {
    const ca = Math.cos(o.az), sa = Math.sin(o.az), ce = Math.cos(o.el), se = Math.sin(o.el), s = o.s || 100;
    const P = (p) => {
      const x1 = p[0] * ca - p[1] * sa, y1 = p[0] * sa + p[1] * ca; // turn about z
      return { x: r2(o.cx + s * x1), y: r2(o.cy - s * (p[2] * ce + y1 * se)), d: -y1 * ce + p[2] * se };
    };
    P.o = o;
    return P;
  };
  /** a surface z = f(x, y) as quads, coloured by height (zs scales the height drawn), far ones painted first */
  H.surface3 = function (key, P, f, o) {
    const n = o.n || 18, [x0, x1] = o.xr, [y0, y1] = o.yr, zs = o.zs || 1, quads = [];
    const z = (x, y) => Math.max(o.zmin === undefined ? -Infinity : o.zmin, Math.min(o.zmax === undefined ? Infinity : o.zmax, f(x, y)));
    let lo = Infinity, hi = -Infinity;
    const G = [];
    for (let i = 0; i <= n; i++) { G.push([]); for (let j = 0; j <= n; j++) { const x = x0 + ((x1 - x0) * i) / n, y = y0 + ((y1 - y0) * j) / n, v = z(x, y); G[i].push([x, y, v]); lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const c = [G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1]], pp = c.map((p) => P([p[0], p[1], p[2] * zs + (o.z0 || 0)]));
      const mz = (c[0][2] + c[1][2] + c[2][2] + c[3][2]) / 4, t = (mz - lo) / (hi - lo || 1);
      quads.push({ d: pp.reduce((s, p) => s + p.d, 0) / 4, it: { key: key + ':' + i + '_' + j, kind: 'path', d: pp.map((p, k) => (k ? 'L' : 'M') + p.x + ',' + p.y).join(' ') + ' Z', cls: 'mlquad', paint: { fill: `color-mix(in srgb, var(--heat1) ${Math.round(12 + 70 * (1 - t))}%, var(--heat0))`, stroke: 'color-mix(in srgb, var(--node-line) 55%, transparent)', sw: 0.6 }, audit: 'ignore', layer: 'back', z: 2, box: H.boxOf(pp) } });
    }
    quads.sort((a, b) => a.d - b.d);
    return quads.map((q, k) => Object.assign(q.it, { z: 2 + k / 10000 }));
  };
  /** the three axes of a 3-D box from lo to hi, with their names at the far ends */
  H.axes3 = function (key, P, lo, hi, names) {
    const o = [lo[0], lo[1], lo[2]], ends = [[hi[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]]], items = [];
    ends.forEach((e, k) => {
      const a = P(o), b = P(e);
      items.push({ key: key + ':a' + k, kind: 'path', d: `M${a.x},${a.y} L${b.x},${b.y}`, cls: 'axis', audit: 'ignore', layer: 'back', z: 1, box: H.boxOf([a, b]) });
      if (names && names[k]) { const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1; items.push(H.text(key + ':n' + k, b.x + (dx / L) * 16, b.y + (dy / L) * 16 + 4, names[k], 'axname', 13, 'middle')); }
    });
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
