// Clustering: k-means (k-means++ seeding, Lloyd iterations), silhouette, DBSCAN and agglomerative
// clustering (single, complete, average, Ward) with the full merge list for the dendrogram.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);

  // greedy k-means++: each new centre is the best of a few candidates drawn in proportion to D(x)^2
  ML.kmeansPlusPlus = function (X, k, rng, tr) {
    const n = X.length, trials = 2 + Math.floor(Math.log(k)), centers = [X[rng.int(n)].slice()];
    let closest = X.map((x) => ML.sqdist(x, centers[0])), pot = ML.sum(closest);
    tr && tr.add('seed', 'The first centre is a random point, ' + ML.fmtv(centers[0]) + '.', { centers: centers.map((c) => c.slice()) });
    for (let c = 1; c < k; c++) {
      let bestPot = Infinity, best = -1, bestClosest = null;
      for (let t = 0; t < trials; t++) {
        let r = rng() * pot, i = 0;
        while (i < n - 1 && r >= closest[i]) { r -= closest[i]; i++; }
        const cl = closest.map((d, q) => Math.min(d, ML.sqdist(X[q], X[i]))), p = ML.sum(cl);
        if (p < bestPot) { bestPot = p; best = i; bestClosest = cl; }
      }
      centers.push(X[best].slice()); closest = bestClosest; pot = bestPot;
      tr && tr.add('seed', 'Centre ' + (c + 1) + ' is point ' + best + ', drawn with probability proportional to its squared distance; total D^2 is now ' + f(pot) + '.', { centers: centers.map((q) => q.slice()), d2: closest.slice() });
    }
    return centers;
  };
  const nearest = (x, C) => { let b = 0, bd = Infinity; for (let k = 0; k < C.length; k++) { const d = ML.sqdist(x, C[k]); if (d < bd) { bd = d; b = k; } } return [b, bd]; };

  ML.model('KMeans', {
    kind: 'cluster',
    defaults: { k: 3, init: 'k-means++', nInit: 1, maxIter: 300, tol: 1e-4, seed: 0 },
    fit(X, y, tr) {
      const p = this.params, rng = ML.rng(p.seed), d = X[0].length;
      const tolAbs = p.tol * ML.mean(ML.colVars(X));
      let best = null;
      for (let run = 0; run < (Array.isArray(p.init) ? 1 : p.nInit); run++) {
        const t = run === 0 ? tr : null;
        let C = Array.isArray(p.init) ? p.init.map((c) => c.slice()) : p.init === 'random' ? rng.sample(X.length, p.k).map((i) => X[i].slice()) : ML.kmeansPlusPlus(X, p.k, rng, t);
        let labels = null, it = 0;
        for (; it < p.maxIter; it++) {
          const nl = X.map((x) => nearest(x, C)[0]);
          const sums = C.map(() => new Array(d).fill(0)), cnt = new Array(C.length).fill(0);
          X.forEach((x, i) => { cnt[nl[i]]++; for (let j = 0; j < d; j++) sums[nl[i]][j] += x[j]; });
          const NC = C.map((c, k) => (cnt[k] ? sums[k].map((s) => s / cnt[k]) : c.slice()));
          let shift = 0;
          NC.forEach((c, k) => (shift += ML.sqdist(c, C[k])));
          const inertia = X.reduce((s, x, i) => s + ML.sqdist(x, C[nl[i]]), 0);
          t && t.add('assign', 'Iteration ' + (it + 1) + ': each point joins its nearest centre (inertia ' + f(inertia) + ').', { iter: it, labels: nl, centers: C.map((c) => c.slice()), inertia });
          t && t.add('update', 'Each centre moves to the mean of its points; the centres moved ' + f(Math.sqrt(shift)) + ' in total.', { iter: it, centers: NC.map((c) => c.slice()), shift });
          const same = labels && nl.every((v, i) => v === labels[i]);
          labels = nl; C = NC;
          if (same) { it++; break; }
          if (shift <= tolAbs) { it++; break; }
        }
        // labels and inertia for the final centres
        const fin = X.map((x) => nearest(x, C));
        const inertia = ML.sum(fin.map((v) => v[1]));
        if (!best || inertia < best.inertia) best = { centers: C, labels: fin.map((v) => v[0]), inertia, nIter: it };
      }
      Object.assign(this, best);
      this.centers = best.centers;
      tr.add('done', 'Converged after ' + best.nIter + ' iterations with inertia ' + f(best.inertia) + '.', { centers: best.centers, labels: best.labels, inertia: best.inertia });
    },
    predict(X) { return X.map((x) => nearest(x, this.centers)[0]); },
    transform(X) { return X.map((x) => this.centers.map((c) => Math.sqrt(ML.sqdist(x, c)))); },
  });

  // inertia for k = 1..kMax, for the elbow picture
  ML.elbow = (X, kMax, seed = 0) => ML.range(kMax).map((i) => new ML.KMeans({ k: i + 1, seed, nInit: 4, trace: false }).fit(X).inertia);

  // per-sample silhouette (b - a) / max(a, b) and its mean
  ML.silhouette = function (X, labels) {
    const n = X.length, ids = [...new Set(labels)], s = new Array(n).fill(0);
    const D = X.map((a) => X.map((b) => Math.sqrt(ML.sqdist(a, b))));
    for (let i = 0; i < n; i++) {
      const sums = new Map(ids.map((c) => [c, [0, 0]]));
      for (let j = 0; j < n; j++) if (j !== i) { const e = sums.get(labels[j]); e[0] += D[i][j]; e[1]++; }
      const own = sums.get(labels[i]);
      if (own[1] === 0) { s[i] = 0; continue; }
      const a = own[0] / own[1];
      let b = Infinity;
      for (const [c, e] of sums) if (c !== labels[i] && e[1]) b = Math.min(b, e[0] / e[1]);
      s[i] = (b - a) / Math.max(a, b);
    }
    return { samples: s, mean: ML.mean(s) };
  };

  // DBSCAN: core points have at least minSamples neighbours within eps (counting themselves)
  ML.model('DBSCAN', {
    kind: 'cluster',
    defaults: { eps: 0.5, minSamples: 5, p: 2 },
    fit(X, y, tr) {
      const p = this.params, n = X.length;
      const nb = X.map((a) => ML.range(n).filter((j) => ML.dist(a, X[j], p.p) <= p.eps));
      const core = nb.map((l) => l.length >= p.minSamples), labels = new Array(n).fill(-1);
      tr.add('core', ML.sum(core.map(Number)) + ' of ' + n + ' points have at least ' + p.minSamples + ' neighbours within ' + f(p.eps) + ' and are core points.', { core: core.slice() });
      let c = 0;
      for (let i0 = 0; i0 < n; i0++) {
        if (labels[i0] !== -1 || !core[i0]) continue;
        const stack = [];
        let i = i0, size = 0;
        for (;;) {
          if (labels[i] === -1) { labels[i] = c; size++; if (core[i]) for (const v of nb[i]) if (labels[v] === -1) stack.push(v); }
          if (!stack.length) break;
          i = stack.pop();
        }
        tr.add('cluster', 'Cluster ' + c + ' grows from core point ' + i0 + ' through its core neighbours to ' + size + ' points.', { cluster: c, seed: i0, labels: labels.slice() });
        c++;
      }
      this.labels = labels; this.core = core; this.coreIndices = ML.range(n).filter((i) => core[i]); this.nClusters = c;
      this.kinds = labels.map((l, i) => (core[i] ? 'core' : l >= 0 ? 'border' : 'noise'));
      tr.add('done', c + ' clusters; ' + labels.filter((l) => l < 0).length + ' points are noise.', { labels });
    },
  });

  // agglomerative clustering by Lance-Williams updates. merges[t] = {a, b, dist, size}; new cluster id n + t.
  ML.linkage = function (X, linkage, tr) {
    const n = X.length, D = X.map((a) => X.map((b) => Math.sqrt(ML.sqdist(a, b))));
    const id = ML.range(n), size = new Array(n).fill(1), alive = new Array(n).fill(true), merges = [];
    for (let t = 0; t < n - 1; t++) {
      let bi = -1, bj = -1, bd = Infinity;
      for (let i = 0; i < n; i++) if (alive[i]) for (let j = i + 1; j < n; j++) if (alive[j] && D[i][j] < bd) { bd = D[i][j]; bi = i; bj = j; }
      const ni = size[bi], nj = size[bj];
      for (let k = 0; k < n; k++) {
        if (!alive[k] || k === bi || k === bj) continue;
        const dik = D[bi][k], djk = D[bj][k], nk = size[k];
        let v;
        if (linkage === 'single') v = Math.min(dik, djk);
        else if (linkage === 'complete') v = Math.max(dik, djk);
        else if (linkage === 'average') v = (ni * dik + nj * djk) / (ni + nj);
        else v = Math.sqrt(Math.max(0, ((ni + nk) * dik * dik + (nj + nk) * djk * djk - nk * bd * bd) / (ni + nj + nk)));
        D[bi][k] = D[k][bi] = v;
      }
      const a = Math.min(id[bi], id[bj]), b = Math.max(id[bi], id[bj]);
      merges.push({ a, b, dist: bd, size: ni + nj });
      tr && tr.add('merge', 'Merge ' + (t + 1) + ': clusters ' + a + ' and ' + b + ' at ' + linkage + ' distance ' + f(bd) + ', giving cluster ' + (n + t) + ' of ' + (ni + nj) + ' points.', { step: t, a, b, dist: bd, size: ni + nj });
      alive[bj] = false; size[bi] = ni + nj; id[bi] = n + t;
    }
    return merges;
  };
  // flat labels from a merge list: stop with nClusters clusters, or before the first merge above a distance
  ML.cutTree = function (merges, n, opts) {
    const parent = ML.range(2 * n - 1), find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    const stop = opts.nClusters != null ? n - opts.nClusters : merges.filter((m) => m.dist <= opts.distance).length;
    for (let t = 0; t < stop; t++) { parent[find(merges[t].a)] = n + t; parent[find(merges[t].b)] = n + t; }
    const roots = new Map(), labels = [];
    for (let i = 0; i < n; i++) { const r = find(i); if (!roots.has(r)) roots.set(r, roots.size); labels.push(roots.get(r)); }
    return labels;
  };
  ML.model('AgglomerativeClustering', {
    kind: 'cluster',
    defaults: { nClusters: 2, linkage: 'ward', distanceThreshold: null },
    fit(X, y, tr) {
      const p = this.params;
      this.merges = ML.linkage(X, p.linkage, tr);
      this.labels = ML.cutTree(this.merges, X.length, p.distanceThreshold != null ? { distance: p.distanceThreshold } : { nClusters: p.nClusters });
      this.nClusters = new Set(this.labels).size;
    },
  });
})();
