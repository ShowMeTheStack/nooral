// Isolation forest: random trees that cut a random feature at a random value until each point is alone.
// Anomalies are isolated in few cuts, so a short average path means a high anomaly score.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);
  const EULER = 0.5772156649015329;

  // average path length of an unsuccessful search in a binary search tree of n points
  ML.isoC = (n) => (n <= 1 ? 0 : n === 2 ? 1 : 2 * (Math.log(n - 1) + EULER) - (2 * (n - 1)) / n);

  function isoTree(X, idx, rng, maxDepth) {
    const d = X[0].length;
    const grow = (rows, depth) => {
      if (depth >= maxDepth || rows.length <= 1) return { size: rows.length };
      // features with some spread at this node
      const live = [];
      for (let j = 0; j < d; j++) { let lo = Infinity, hi = -Infinity; for (const i of rows) { lo = Math.min(lo, X[i][j]); hi = Math.max(hi, X[i][j]); } if (hi > lo) live.push([j, lo, hi]); }
      if (!live.length) return { size: rows.length };
      const [j, lo, hi] = live[rng.int(live.length)], thr = lo + rng() * (hi - lo);
      const L = rows.filter((i) => X[i][j] < thr), R = rows.filter((i) => X[i][j] >= thr);
      return { feature: j, threshold: thr, left: grow(L, depth + 1), right: grow(R, depth + 1) };
    };
    return grow(idx, 0);
  }
  function pathLength(node, x) {
    let depth = 0;
    while (node.left) { node = x[node.feature] < node.threshold ? node.left : node.right; depth++; }
    return depth + ML.isoC(node.size);
  }

  ML.model('IsolationForest', {
    kind: 'outlier',
    defaults: { nEstimators: 100, maxSamples: 'auto', contamination: 'auto', seed: 0 },
    fit(X, y, tr) {
      const p = this.params, rng = ML.rng(p.seed), n = X.length;
      const m = (this.maxSamples = p.maxSamples === 'auto' ? Math.min(256, n) : Math.min(p.maxSamples, n));
      const depth = Math.ceil(Math.log2(Math.max(m, 2)));
      this.trees = [];
      for (let b = 0; b < p.nEstimators; b++) {
        const T = isoTree(X, rng.sample(n, m), rng, depth);
        this.trees.push(T);
        if (b < 5) tr.add('tree', 'Tree ' + (b + 1) + ' cuts a sample of ' + m + ' points with random features and thresholds, down to depth ' + depth + '.', { tree: b, root: { feature: T.feature, threshold: T.threshold } });
      }
      this.offset = -0.5;
      if (p.contamination !== 'auto') {
        const s = this.scoreSamples(X).slice().sort((a, b) => a - b);
        this.offset = s[Math.min(n - 1, Math.max(0, Math.floor(p.contamination * n)))];
      }
      tr.add('done', 'Scores are -2^(-mean path / ' + f(ML.isoC(m)) + '); points scoring below ' + f(this.offset) + ' are flagged.', { offset: this.offset });
    },
    // the mean path length of each point over the trees
    pathLengths(X) { return X.map((x) => ML.mean(this.trees.map((T) => pathLength(T, x)))); },
    // higher is more normal (near -0.5 normal, near -1 anomalous)
    scoreSamples(X) { const c = ML.isoC(this.maxSamples); return this.pathLengths(X).map((h) => -Math.pow(2, -h / c)); },
    decisionFunction(X) { return this.scoreSamples(X).map((s) => s - this.offset); },
    predict(X) { return this.decisionFunction(X).map((v) => (v < 0 ? -1 : 1)); },
  });
})();
