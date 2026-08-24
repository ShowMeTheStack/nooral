// Part 7, unsupervised learning: the families, the datasets the topics use, and the pieces of picture they share
// (points coloured by cluster, centre markers, equal-scale axes, a small table of numbers).
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'unsupervised', order, title, blurb });
  fam('p07-kmeans', 10, 'k-means', 'Centres that move to the mean of their points: Lloyd’s steps, k-means++ seeding, and the elbow and silhouette for choosing k.');
  fam('p07-hier', 20, 'Hierarchical clustering', 'Merge the two closest clusters, again and again: the dendrogram that records it, the cut that turns it into clusters, and the four ways to measure “closest”.');
  fam('p07-dbscan', 30, 'Density: DBSCAN', 'Clusters as crowded regions: core points with enough neighbours within ε, the border points they reach, and the noise left over.');
  fam('p07-gmm', 40, 'Gaussian mixtures', 'Soft clusters as overlapping Gaussians, fitted by expectation and maximisation, and the number of them chosen by BIC.');
  fam('p07-pca', 50, 'Principal components', 'The directions of greatest spread: projection onto them, the variance each explains, and what is lost when the rest are dropped.');
  fam('p07-embed', 60, 'Maps of high-dimensional data', 't-SNE and a UMAP-style layout: neighbours kept together on a flat map, drawn while the optimiser runs.');
  fam('p07-anomaly', 70, 'Anomaly detection', 'Isolation forests: an odd point is cut off from the rest in a few random cuts, so a short path means a high anomaly score.');

  const ML = () => S.ML, H = () => S.Kit.ml;
  const P = (S.P07 = {});
  P.NAMES = ['blue', 'orange', 'green', 'violet', 'pink', 'teal'];

  const blobs = (centers, std) => (o) => S.Data.blobs({ n: o.n || 90, centers, std: o.std || std, seed: o.seed || 1 });
  const std2 = (X) => { const m = ML().colMeans(X), sd = ML().colVars(X).map(Math.sqrt); return X.map((r) => r.map((v, j) => +((v - m[j]) / sd[j]).toFixed(4))); };
  P.SETS = {
    three: { label: 'three round groups', make: blobs([[-2, 1.2], [2, 1.5], [0, -2]], 0.7) },
    four: { label: 'four groups', make: blobs([[-2.2, 1.8], [2, 2], [-1.8, -1.8], [2.2, -1.6]], 0.65) },
    touching: { label: 'three groups that touch', make: blobs([[-1.3, 0.6], [1.3, 0.8], [0, -1.3]], 0.75) },
    uneven: { label: 'a big group and two small ones', make: (o) => { const d = S.Data.blobs({ n: o.n || 90, centers: [[-1.2, 0], [2.2, 1.8], [2.4, -1.6]], std: 0.5, seed: o.seed || 1 }); return { X: d.X.map((x, i) => (d.y[i] === 0 ? [(x[0] + 1.2) * 2.6 - 1.2, (x[1]) * 2.6] : x)), y: d.y }; } },
    stretched: { label: 'three stretched groups', make: (o) => { const d = S.Data.blobs({ n: o.n || 90, centers: [[-2, 1], [0.5, 0], [3, -1]], std: 0.45, seed: o.seed || 1 }); return { X: d.X.map((x) => [+(0.5 * x[0] + 1.2 * x[1]).toFixed(4), +(1.3 * x[1] - 0.3 * x[0]).toFixed(4)]), y: d.y }; } },
    moons: { label: 'two moons', make: (o) => { const d = S.Data.moons({ n: o.n || 100, noise: o.noise === undefined ? 0.07 : o.noise, seed: o.seed || 1 }); return { X: d.X.map((r) => r.map((v) => +(2 * v).toFixed(4))), y: d.y }; } },
    circles: { label: 'a ring round a disc', make: (o) => { const d = S.Data.circles({ n: o.n || 100, noise: 0.06, factor: 0.4, seed: o.seed || 1 }); return { X: d.X.map((r) => r.map((v) => +(2.5 * v).toFixed(4))), y: d.y }; } },
    bridge: { label: 'two groups joined by a thin bridge', make: (o) => { const n = o.n || 30, m = Math.max(3, Math.round(n / 3.5)), d = S.Data.blobs({ n: n - m, centers: [[-2.5, 0], [2.5, 0.4]], std: 0.42, seed: o.seed || 1 }), r = S.Data.rng((o.seed || 1) + 7), X = d.X.slice(), y = d.y.slice(); for (let i = 0; i < m; i++) { const t = (i + 1) / (m + 1); X.push([+(-1.9 + 3.8 * t).toFixed(4), +(0.06 + 0.3 * t + r.normal(0, 0.06)).toFixed(4)]); y.push(2); } return { X, y }; } },
    noisy: { label: 'three groups in scattered noise', make: (o) => { const n = o.n || 110, m = Math.round(n / 7), d = S.Data.blobs({ n: n - m, centers: [[-2, 1.2], [2, 1.4], [0, -1.8]], std: 0.55, seed: o.seed || 1 }), r = S.Data.rng((o.seed || 1) + 11), X = d.X.slice(), y = d.y.slice(); for (let i = 0; i < m; i++) { X.push([+r.range(-4, 4).toFixed(4), +r.range(-3.5, 3.5).toFixed(4)]); y.push(3); } return { X, y }; } },
    iris: { label: 'iris: petal length and width (cm)', make: () => { const d = S.Data.load('iris'); return { X: d.X.map((x) => [x[2], x[3]]), y: d.y, names: ['petal length (cm)', 'petal width (cm)'], real: true }; } },
    penguins: { label: 'penguins: bill length and depth (standardised)', make: () => { const d = S.Data.load('penguins'), rows = d.X.map((x, i) => [x[1], x[2], d.y[i]]).filter((r) => isFinite(r[0]) && isFinite(r[1]) && r[0] !== null && r[1] !== null); return { X: std2(rows.map((r) => [r[0], r[1]])), y: rows.map((r) => r[2]), names: ['bill length (standardised)', 'bill depth (standardised)'], real: true }; } },
  };
  P.data = (id, o) => { const d = P.SETS[id].make(o || {}); return Object.assign({ names: ['x₁', 'x₂'] }, d); };
  P.setOptions = (ids) => ids.map((id) => ({ value: id, label: P.SETS[id].label }));
  P.std = std2;

  /** equal-scale axes round a set of points (and anything else to keep in view) */
  P.axes = function (prefix, X, box, o) {
    o = o || {};
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const x of X.concat(o.extra || [])) for (let j = 0; j < 2; j++) { lo[j] = Math.min(lo[j], x[j]); hi[j] = Math.max(hi[j], x[j]); }
    const pad = o.pad === undefined ? 0.08 : o.pad, dx = (hi[0] - lo[0]) || 1, dy = (hi[1] - lo[1]) || 1;
    let xr = [lo[0] - pad * dx, hi[0] + pad * dx], yr = [lo[1] - pad * dy, hi[1] + pad * dy];
    // fit: a wide cloud gets a shorter box (never under 0.55 of its width), so less of the stage is empty
    if (o.fit) { box = Object.assign({}, box); const want = (box.w * (yr[1] - yr[0])) / (xr[1] - xr[0]); if (want < box.h) { const h = Math.max(box.w * 0.55, want); box.y += (box.h - h) / 2; box.h = h; } }
    if (o.equal !== false) {
      const s = Math.max((xr[1] - xr[0]) / box.w, (yr[1] - yr[0]) / box.h);
      const grow = (rg, len) => { const m = (rg[0] + rg[1]) / 2; return [m - (s * len) / 2, m + (s * len) / 2]; };
      xr = grow(xr, box.w); yr = grow(yr, box.h);
    }
    return S.Kit.curve.axes(Object.assign({ prefix, xr, yr, xname: o.xname || 'x₁', yname: o.yname || 'x₂', tickSize: 12 }, box));
  };
  /** points coloured by cluster (−1 or null: grey, unassigned or noise) */
  P.points = function (ax, key, X, labels, o) {
    o = o || {};
    return X.map((p, i) => {
      const l = labels ? labels[i] : -1, it = { key: key + ':' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: o.r ? (typeof o.r === 'function' ? o.r(i) : o.r) : 4.6, cls: 'sp ' + (l === null || l === undefined || l < 0 ? 'cn' : 'c' + l) + ' ' + (o.cls ? o.cls(i) || '' : ''), audit: 'overlap-ok', layer: 'items' };
      if (o.opacity) { const op = o.opacity(i); if (op !== undefined && op !== 1) it.opacity = op; }
      return it;
    });
  };
  /** a cluster centre: a square in its cluster's colour */
  P.centre = (ax, key, c, k, size) => ({ key, kind: 'state', x: ax.X(c[0]), y: ax.Y(c[1]), w: size || 20, h: size || 20, rx: 4, text: '', cls: 'centre c' + k, audit: 'overlap-ok', layer: 'items', z: 5 });
  /** a swatch and a line of text: a legend row */
  P.row = (key, x, y, k, text, cls) => [{ key: key + ':s', kind: 'dot', x: x + 6, y: y - 4.5, r: 6, cls: 'swatch c' + k }, H().text(key + ':t', x + 20, y, text, cls || 'legend', 13)];
  /** cluster sizes */
  P.sizes = (labels, k) => { const c = new Array(k).fill(0); for (const l of labels) if (l >= 0) c[l]++; return c; };
  /** a quoted count with its noun: "1 point", "3 points" */
  P.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');
  /** the bend of a rising sorted curve: the index furthest below the chord from its first to its last value */
  P.elbowKnee = (v) => { const n = v.length, lo = v[0], hi = v[n - 1]; let b = 0, bd = -Infinity; v.forEach((y, i) => { const d = i / (n - 1) - (y - lo) / (hi - lo || 1); if (d > bd) { bd = d; b = i; } }); return b; };
  P.ord = (k) => (k % 100 >= 11 && k % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][k % 10] || 'th');
  P.sq = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
})(globalThis.Nooral = globalThis.Nooral || {});
