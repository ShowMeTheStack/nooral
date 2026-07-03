// A decision tree (CART, by Gini impurity) and k-means for the sample scenes (S.Toy), each with a trace of every
// step: the candidate thresholds a node weighed and the one it took; each assignment and each move of the centres.
(function (S) {
  'use strict';
  const T = (S.Toy = S.Toy || {});

  T.gini = (counts) => { const n = counts.reduce((a, b) => a + b, 0); return n ? 1 - counts.reduce((a, c) => a + (c / n) ** 2, 0) : 0; };
  const countsOf = (idx, y, k) => { const c = new Array(k).fill(0); for (const i of idx) c[y[i]]++; return c; };

  /**
   * Grow a tree breadth first: {root, steps: [{node, cands: [{f, t, g}..], best}]}; a node is {id, idx, counts,
   * gini, depth, box: [[x0, x1], [y0, y1]], split: {f, t, g}, kids: [left (≤ t), right]}.
   */
  T.cart = function (X, y, o) {
    o = o || {};
    const k = Math.max(...y) + 1, maxDepth = o.maxDepth || 3, minLeaf = o.minLeaf || 2;
    let nid = 0;
    const node = (idx, depth, box) => { const counts = countsOf(idx, y, k); return { id: nid++, idx, counts, gini: T.gini(counts), depth, box }; };
    const root = node(X.map((_, i) => i), 0, o.box || [[-Infinity, Infinity], [-Infinity, Infinity]]);
    const steps = [], queue = [root];
    while (queue.length) {
      const nd = queue.shift();
      if (nd.depth >= maxDepth || nd.gini < 1e-9 || nd.idx.length < 2 * minLeaf) continue;
      const cands = [];
      let best = null;
      for (let f = 0; f < 2; f++) {
        const vals = [...new Set(nd.idx.map((i) => X[i][f]))].sort((a, b) => a - b);
        for (let m = 1; m < vals.length; m++) {
          const t = (vals[m - 1] + vals[m]) / 2, L = [], R = [];
          for (const i of nd.idx) (X[i][f] <= t ? L : R).push(i);
          if (L.length < minLeaf || R.length < minLeaf) continue;
          const g = (L.length * T.gini(countsOf(L, y, k)) + R.length * T.gini(countsOf(R, y, k))) / nd.idx.length;
          cands.push({ f, t, g });
          if (!best || g < best.g - 1e-12) best = { f, t, g, L, R };
        }
      }
      if (!best || best.g >= nd.gini - 1e-9) continue;
      nd.split = { f: best.f, t: best.t, g: best.g };
      const bl = nd.box.map((r) => r.slice()), br = nd.box.map((r) => r.slice());
      bl[best.f][1] = best.t; br[best.f][0] = best.t;
      nd.kids = [node(best.L, nd.depth + 1, bl), node(best.R, nd.depth + 1, br)];
      steps.push({ node: nd, cands, best: nd.split });
      queue.push(...nd.kids);
    }
    return { root, steps, k };
  };
  /** the leaf a point falls in, given how far the tree has grown (nodes whose split is not yet made are leaves) */
  T.cartLeaf = (root, p, made) => { let n = root; while (n.kids && made.has(n.id)) n = p[n.split.f] <= n.split.t ? n.kids[0] : n.kids[1]; return n; };

  /**
   * k-means (Lloyd's steps) from given starting centres: {steps: [{kind: 'init' | 'assign' | 'move', C, lab,
   * inertia, moved}], converged}
   */
  T.kmeans = function (X, C0, o) {
    o = o || {};
    let C = C0.map((c) => c.slice()), lab = X.map(() => -1);
    const steps = [{ kind: 'init', C: C.map((c) => c.slice()), lab: lab.slice(), inertia: null }];
    const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
    for (let it = 0; it < (o.maxIter || 20); it++) {
      let changed = 0, inertia = 0;
      lab = X.map((p, i) => { let best = 0; C.forEach((c, j) => { if (d2(p, c) < d2(p, C[best])) best = j; }); if (best !== lab[i]) changed++; inertia += d2(p, C[best]); return best; });
      steps.push({ kind: 'assign', C: C.map((c) => c.slice()), lab: lab.slice(), inertia, changed });
      if (!changed && it) break;
      const sum = C.map(() => [0, 0, 0]);
      X.forEach((p, i) => { sum[lab[i]][0] += p[0]; sum[lab[i]][1] += p[1]; sum[lab[i]][2]++; });
      const nc = C.map((c, j) => (sum[j][2] ? [sum[j][0] / sum[j][2], sum[j][1] / sum[j][2]] : c.slice()));
      const moved = nc.map((c, j) => Math.hypot(c[0] - C[j][0], c[1] - C[j][1]));
      C = nc;
      let inert2 = 0;
      X.forEach((p, i) => { inert2 += d2(p, C[lab[i]]); });
      steps.push({ kind: 'move', C: C.map((c) => c.slice()), lab: lab.slice(), inertia: inert2, moved });
    }
    return { steps };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
