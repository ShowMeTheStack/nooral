// The Unsupervised blocks, part two: Gaussian mixtures by EM (soft groups, each an ellipse), and the two maps that
// squeeze a wide table onto a page while keeping neighbours together, t-SNE and a UMAP-style embedding. A map replaces
// the features with its two coordinates; it cannot place rows it was not built on, and says so if asked to.
(function (S) {
  'use strict';
  const WF = S.WF, U = () => WF.dutil;

  /** a covariance (full, diagonal or one number) seen through a 2-D projection: P C Pᵀ, as [[a, b], [b, c]] */
  const project2 = (C, type, d, P) => {
    const full = type === 'full' ? C : type === 'diag' ? C.map((v, i) => C.map((_, j) => (i === j ? v : 0))) : Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? C : 0)));
    if (!P) return [[full[0][0], full[0][1] || 0], [full[1] ? full[1][0] : 0, full[1] ? full[1][1] : 1]];
    const M = P.map((p) => full[0].map((_, j) => p.reduce((s, pi, i) => s + pi * full[i][j], 0)));
    return P.map((_, a) => P.map((q, b) => M[a].reduce((s, v, j) => s + v * q[j], 0)));
  };
  /** an ellipse for a 2×2 covariance: the axes at 2 standard deviations and the angle in degrees */
  const ellipse = (m, C) => {
    const a = C[0][0], b = C[0][1], c = C[1][1], tr = (a + c) / 2, det = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b));
    const l1 = tr + det, l2 = Math.max(1e-12, tr - det), ang = Math.atan2(l1 - a, b || 1e-12) * (180 / Math.PI);
    return { x: m[0], y: m[1], rx: 2 * Math.sqrt(l1), ry: 2 * Math.sqrt(l2), angle: Math.abs(b) < 1e-12 ? (a >= c ? 0 : 90) : ang };
  };

  WF.block('gmm', {
    cat: 'Unsupervised', label: 'Gaussian mixture', icon: 'target', order: 40,
    blurb: 'Explains the rows as a mix of k bell-shaped clouds. EM alternates: the E-step says how much each cloud owns each row, the M-step refits each cloud to the rows it owns. Groups are soft, and clouds may be stretched and tilted.',
    inputs: [{ id: 'in', label: 'table', type: 'table' }],
    outputs: [{ id: 'out', label: 'table + cluster', type: 'table' }],
    params: [
      { id: 'k', label: 'Components (k)', kind: 'int', def: 3, min: 1, max: 10 },
      { id: 'cov', label: 'Shape of each cloud', kind: 'choice', def: 'full', options: [['full', 'any ellipse (full)'], ['diag', 'axis-aligned (diagonal)'], ['spherical', 'round (spherical)']] },
      { id: 'maxIter', label: 'At most this many EM rounds', kind: 'int', def: 100, min: 1, max: 1000 },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const v = ctx.in.in, p = ctx.params, X = WF.clusterInput(v, 'a Gaussian mixture');
      const E = WF.needEngine('GaussianMixture', 'Gaussian mixture');
      const g = new E({ k: Math.min(p.k, X.length), covarianceType: p.cov, maxIter: p.maxIter, seed: p.seed }).fit(X);
      const names = g.means.map((_, i) => String(i));
      const out = WF.withClusters(v, g.labels, names, (Y) => g.predict(Y), 'component');
      return { out: { out }, g };
    },
    view(r, ctx) {
      const u = U(), v = ctx.in.in, g = r.g, pl = WF.clusterPlane(v), d = v.features.length, type = ctx.params.cov;
      const P = pl.pca ? pl.pca.components.slice(0, 2) : d === 1 ? null : null;
      const shape = (means, covs, weights) => means.map((m, k) => { const pm = pl.proj([m])[0], C = d === 1 ? [[type === 'spherical' ? covs[k] : type === 'diag' ? covs[k][0] : covs[k][0][0], 0], [0, 1e-4]] : project2(covs[k], type, d, P); const e = ellipse(pm, C); return { x: u.r(e.x), y: u.r(e.y), rx: u.r(e.rx), ry: u.r(e.ry), angle: u.r(e.angle, 4), w: u.r(weights[k], 3) }; });
      const em = (g.trace || []).filter((s) => s.kind === 'em').slice(0, 40).map((s) => ({ iter: s.iter, ll: u.r(s.ll, 5), clouds: shape(s.means, s.covariances, s.weights) }));
      const R = g.predictProba(pl.keep.map((i) => v.X[i]));
      return Object.assign({ kind: 'gmm', twoD: d === 2, n: v.X.length, P: pl.P, names: pl.names, how: pl.how, y: pl.y, classes: v.classes, k: g.means.length, cov: type,
        labels: pl.keep.map((i) => g.labels[i]), certainty: R.map((row) => u.r(Math.max(...row), 3)), clouds: shape(g.means, g.covariances, g.weights), em,
        iterations: g.nIter, ll: u.r(g.score(v.X), 5), bic: u.r(g.bic(v.X), 6), sizes: g.means.map((_, c) => g.labels.filter((l) => l === c).length) }, WF.clusterQuality(v, g.labels));
    },
    summary(r) { return r.g.means.length + ' components · ' + r.g.nIter + ' EM rounds'; },
  });

  // ---- the maps -------------------------------------------------------------------------------------------------
  const mapBlock = (type, label, engine, blurb, params, make, order) => WF.block(type, {
    cat: 'Unsupervised', label, icon: 'compare', order, blurb,
    inputs: [{ id: 'in', label: 'table', type: 'table' }],
    outputs: [{ id: 'out', label: '2-D map', type: 'table' }],
    params,
    run(ctx) {
      const v = ctx.in.in, X = WF.clusterInput(v, label);
      if (X.length > 2500) throw new Error(label + ' is slow on many rows: keep it to 2500 here (this table has ' + X.length + ')');
      WF.needEngine(engine, label);
      const m = make(ctx.params, X.length);
      const Y = m.fitTransform(X);
      const step = { name: type, say: 'a ' + label + ' map', apply: () => { throw new Error(label + ' cannot place new rows on its map: it only arranges the rows it was built on. Use it to look at data, not inside a model you will test'); } };
      const out = WF.transformed(v, Y.map((r) => r.slice()), [{ name: label + ' 1', type: 'number' }, { name: label + ' 2', type: 'number' }], step);
      return { out: { out }, m, Y };
    },
    view(r, ctx) {
      const u = U(), v = ctx.in.in, keep = u.sample(v.X.length, 700), snaps = r.m.snapshots || [];
      const pickS = u.sample(snaps.length, Math.min(snaps.length, 24)).map((i) => snaps[i]);
      return { kind: type, label, n: v.X.length, d: v.features.length, y: v.task === 'classification' ? keep.map((i) => v.y[i]) : null, target: v.task === 'regression' ? keep.map((i) => u.r(v.y[i], 4)) : null, classes: v.classes,
        snaps: pickS.map((s) => ({ say: s.say, iter: s.iter !== undefined ? s.iter : s.epoch, kl: s.kl !== undefined ? u.r(s.kl, 4) : null, Y: keep.map((i) => [u.r(s.Y[i][0], 4), u.r(s.Y[i][1], 4)]) })),
        Y: keep.map((i) => [u.r(r.Y[i][0], 4), u.r(r.Y[i][1], 4)]), kl: r.m.kl !== undefined ? u.r(r.m.kl, 4) : null, params: ctx.params,
        trust: (() => { try { const k = u.sample(v.X.length, 300); return u.r(S.ML.trustworthiness(k.map((i) => v.X[i]), k.map((i) => r.Y[i]), 5), 4); } catch (e) { return null; } })() };
    },
    summary(r) { return r.Y.length + ' rows on a map'; },
  });
  mapBlock('tsne', 't-SNE', 'TSNE', 'Places the rows on a plane so that rows that are near in the table stay near on the map. Perplexity is roughly how many neighbours each row cares about. Distances between far groups mean little.',
    [{ id: 'perplexity', label: 'Perplexity', kind: 'number', def: 30, min: 2, max: 100 }, { id: 'iters', label: 'Iterations', kind: 'int', def: 500, min: 250, max: 2000 }, { id: 'seed', label: 'Seed', kind: 'seed', def: 1 }],
    (p, n) => S.ML.TSNE({ perplexity: Math.min(p.perplexity, Math.max(2, (n - 1) / 3)), maxIter: p.iters, seed: p.seed, method: n > 400 ? 'barnes-hut' : 'exact', snapshotEvery: 20 }), 60);
  mapBlock('umap', 'UMAP', 'UMAP', 'Builds a graph of each row’s nearest neighbours, then lays it out on a plane, pulling neighbours together and pushing random pairs apart. It keeps more of the large-scale shape than t-SNE and is faster.',
    [{ id: 'neighbours', label: 'Neighbours', kind: 'int', def: 15, min: 2, max: 100 }, { id: 'minDist', label: 'Minimum distance', kind: 'number', def: 0.1, min: 0, max: 1, step: 0.05 }, { id: 'epochs', label: 'Epochs', kind: 'int', def: 200, min: 20, max: 1000 }, { id: 'seed', label: 'Seed', kind: 'seed', def: 1 }],
    (p) => S.ML.UMAP({ nNeighbors: p.neighbours, minDist: p.minDist, nEpochs: p.epochs, seed: p.seed, snapshotEvery: 10 }), 70);
})(globalThis.Nooral = globalThis.Nooral || {});
