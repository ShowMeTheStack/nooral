// k nearest neighbours: brute force or a k-d tree; uniform or distance weights; Minkowski metrics
// (p = 1 Manhattan, p = 2 Euclidean, any p >= 1, or Infinity for Chebyshev).
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);

  const metricP = (m, p) => (m === 'manhattan' ? 1 : m === 'euclidean' ? 2 : m === 'chebyshev' ? Infinity : p);

  // ---- k-d tree ----
  ML.KDTree = function (X, opts) {
    const leafSize = (opts && opts.leafSize) || 10, d = X[0].length;
    const build = (idx, depth) => {
      const lo = new Array(d).fill(Infinity), hi = new Array(d).fill(-Infinity);
      for (const i of idx) for (let j = 0; j < d; j++) { lo[j] = Math.min(lo[j], X[i][j]); hi[j] = Math.max(hi[j], X[i][j]); }
      const node = { lo, hi, n: idx.length, depth };
      if (idx.length <= leafSize) { node.idx = idx; return node; }
      let dim = 0;
      for (let j = 1; j < d; j++) if (hi[j] - lo[j] > hi[dim] - lo[dim]) dim = j;
      if (hi[dim] === lo[dim]) { node.idx = idx; return node; }
      const s = idx.slice().sort((a, b) => X[a][dim] - X[b][dim] || a - b), m = s.length >> 1;
      node.dim = dim; node.split = X[s[m]][dim];
      node.left = build(s.slice(0, m), depth + 1);
      node.right = build(s.slice(m), depth + 1);
      return node;
    };
    const root = build(ML.range(X.length), 0);
    const boxDist = (node, x, p) => {
      let s = 0;
      for (let j = 0; j < d; j++) {
        const g = x[j] < node.lo[j] ? node.lo[j] - x[j] : x[j] > node.hi[j] ? x[j] - node.hi[j] : 0;
        if (p === Infinity) s = Math.max(s, g); else s += g ** p;
      }
      return p === Infinity ? s : s ** (1 / p);
    };
    // k nearest of x: returns {idx, dist, steps}; steps show visited and pruned boxes
    const query = (x, k, p, tr) => {
      const best = []; // sorted by (dist, index)
      const worst = () => (best.length < k ? Infinity : best[best.length - 1].d);
      const offer = (i) => {
        const dd = ML.dist(X[i], x, p);
        if (best.length === k && (dd > worst() || (dd === worst() && i > best[best.length - 1].i))) return;
        let at = best.length;
        while (at > 0 && (best[at - 1].d > dd || (best[at - 1].d === dd && best[at - 1].i > i))) at--;
        best.splice(at, 0, { d: dd, i });
        if (best.length > k) best.pop();
      };
      const visit = (node) => {
        const bd = boxDist(node, x, p);
        if (bd > worst()) { if (tr) tr.push({ kind: 'prune', say: 'Box at depth ' + node.depth + ' is ' + f(bd) + ' away, farther than the current k-th neighbour ' + f(worst()) + ': skipped.', lo: node.lo, hi: node.hi, depth: node.depth }); return; }
        if (node.idx) {
          node.idx.forEach(offer);
          if (tr) tr.push({ kind: 'leaf', say: 'Leaf at depth ' + node.depth + ' with ' + node.n + ' points checked; k-th distance is now ' + f(worst()) + '.', lo: node.lo, hi: node.hi, best: best.map((b) => b.i) });
          return;
        }
        const first = x[node.dim] < node.split ? node.left : node.right, second = first === node.left ? node.right : node.left;
        if (tr) tr.push({ kind: 'descend', say: 'At depth ' + node.depth + ' the query has x' + node.dim + ' = ' + f(x[node.dim]) + ' against the split ' + f(node.split) + ', so the near side is searched first.', dim: node.dim, split: node.split, lo: node.lo, hi: node.hi });
        visit(first); visit(second);
      };
      visit(root);
      return { idx: best.map((b) => b.i), dist: best.map((b) => b.d) };
    };
    return { root, query };
  };

  function knnFit(X, y, tr) {
    const p = this.params;
    this.X = X; this.y = y; this.p = metricP(p.metric, p.p);
    if (this.kind === 'classifier') { const e = ML.encode(y); this.classes = e.classes; this.yi = e.idx; }
    if (p.algorithm === 'kd_tree') {
      this.tree = ML.KDTree(X, { leafSize: p.leafSize });
      tr.add('tree', 'Built a k-d tree over ' + X.length + ' points (leaf size ' + p.leafSize + ').', {});
    } else tr.add('store', 'k-NN just stores the ' + X.length + ' training points; the work happens at prediction time.', {});
  }
  // indices and distances of the k nearest training points, nearest first (ties by index)
  function kneighbors(Q, k, withTrace) {
    k = k || this.params.k;
    return Q.map((x) => {
      if (this.tree) { const steps = withTrace ? [] : null, r = this.tree.query(x, k, this.p, steps); if (steps) r.steps = steps; return r; }
      const ds = this.X.map((r, i) => ({ d: ML.dist(r, x, this.p), i })).sort((a, b) => a.d - b.d || a.i - b.i).slice(0, k);
      const r = { idx: ds.map((o) => o.i), dist: ds.map((o) => o.d) };
      if (withTrace) r.steps = [{ kind: 'brute', say: 'Measured the distance to all ' + this.X.length + ' training points; the ' + k + ' nearest are at ' + ML.fmtv(r.dist) + '.', idx: r.idx, dist: r.dist }];
      return r;
    });
  }
  function weightsOf(dist, mode) {
    if (mode !== 'distance') return dist.map(() => 1);
    if (dist.some((v) => v === 0)) return dist.map((v) => (v === 0 ? 1 : 0));
    return dist.map((v) => 1 / v);
  }

  ML.model('KNeighborsClassifier', {
    kind: 'classifier',
    defaults: { k: 5, weights: 'uniform', metric: 'minkowski', p: 2, algorithm: 'brute', leafSize: 10 },
    fit: knnFit,
    kneighbors,
    predictProba(X) {
      return this.kneighbors(X).map(({ idx, dist }) => {
        const w = weightsOf(dist, this.params.weights), v = new Array(this.classes.length).fill(0);
        idx.forEach((i, t) => (v[this.yi[i]] += w[t]));
        const s = ML.sum(v);
        return v.map((a) => a / s);
      });
    },
    predict(X) { return this._argmaxPredict(X); },
    // one query with its neighbours and the search steps, for the picture
    explain(x) {
      const r = this.kneighbors([x], this.params.k, true)[0], w = weightsOf(r.dist, this.params.weights);
      return Object.assign(r, { weights: w, labels: r.idx.map((i) => this.y[i]) });
    },
  });

  ML.model('KNeighborsRegressor', {
    kind: 'regressor',
    defaults: { k: 5, weights: 'uniform', metric: 'minkowski', p: 2, algorithm: 'brute', leafSize: 10 },
    fit: knnFit,
    kneighbors,
    predict(X) {
      return this.kneighbors(X).map(({ idx, dist }) => {
        const w = weightsOf(dist, this.params.weights);
        let s = 0, t = 0;
        idx.forEach((i, j) => { s += w[j] * this.y[i]; t += w[j]; });
        return s / t;
      });
    },
  });
})();
