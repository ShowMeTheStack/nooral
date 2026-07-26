// The Unsupervised blocks, part one: k-means, DBSCAN and hierarchical clustering. Each finds groups without looking
// at the target and adds the group of every row as a new category column, so a later model may use it; the target
// stays, so a view can compare the groups with the classes. Their views keep every step for the pictures (on the two
// features, or on the first two principal components of a wider table).
(function (S) {
  'use strict';
  const WF = S.WF, U = () => WF.dutil;
  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  const MAXV = 700; // points in a view

  WF.needEngine = (name, label) => { const ML = S.ML || {}; const E = ML[name] || (ML.registry || {})[name]; if (!E) throw new Error('the ' + label + ' engine is not in this build yet'); return E; };
  /** the numbers a clustering block works on: every cell filled */
  WF.clusterInput = function (v, label) {
    if (!v || v.kind !== 'table') throw new Error(label + ' works on tables' + (v && v.kind === 'text' ? ': vectorise the texts first' : ''));
    if (v.X.some((r) => !r.every(isNum))) throw new Error('some cells are missing: clean or impute them before ' + label);
    if (v.X.length < 3) throw new Error(label + ' needs at least three rows');
    return v.X;
  };
  /** the table with the group of each row added as a column (and a pipe step that places new rows) */
  WF.withClusters = function (v, labels, names, place, colName) {
    const cats = names, idx = labels.map((l) => cats.indexOf(String(l)));
    const step = { name: colName, say: 'add the ' + colName + ' of each row', apply: (X) => { const g = place(X); return X.map((r, i) => r.concat([cats.indexOf(String(g[i]))])); } };
    return WF.transformed(v, v.X.map((r, i) => r.concat([idx[i]])), v.features.concat([{ name: colName, type: 'category', categories: cats }]), step);
  };
  /** the 2-D picture of a table's rows, a sample of them, for a clustering view */
  WF.clusterPlane = function (v) {
    const u = U(), keep = u.sample(v.X.length, MAXV), pl = u.plane(v.X, v.features.map((f) => f.name));
    const proj = pl.pca ? (Y) => pl.pca.transform(Y) : v.features.length === 1 ? (Y) => Y.map((r) => [r[0], 0]) : (Y) => Y.map((r) => [r[0], r[1]]);
    return { keep, P: u.round2(keep.map((i) => pl.P[i])), names: pl.names, how: pl.how, proj, pca: pl.pca || null, y: v.task === 'classification' ? keep.map((i) => v.y[i]) : null };
  };
  /** how well groups match the classes, when there are classes: the adjusted Rand index */
  const agreement = (v, labels) => { try { return v.task === 'classification' ? U().r(S.Metrics.adjustedRand(v.y, labels).value, 4) : null; } catch (e) { return null; } };
  const silhouette = (X, labels) => { try { const ok = labels.map((l, i) => (l >= 0 ? i : -1)).filter((i) => i >= 0); if (new Set(ok.map((i) => labels[i])).size < 2) return null; const k = U().sample(ok.length, 1200).map((t) => ok[t]); return U().r(S.Metrics.silhouette(k.map((i) => X[i]), k.map((i) => labels[i])).value, 4); } catch (e) { return null; } };
  WF.clusterQuality = (v, labels) => ({ silhouette: silhouette(v.X, labels), ari: agreement(v, labels) });
  /** the distance from each of up to 600 rows to its minPts-th nearest neighbour, sorted: the knee suggests an ε */
  const kDistance = (X, k) => {
    const u = U(), idx = u.sample(X.length, 600);
    return idx.map((i) => { const d = X.map((y) => Math.sqrt(y.reduce((a, v, j) => a + (v - X[i][j]) ** 2, 0))).sort((a, b) => a - b); return u.r(d[Math.min(k - 1, d.length - 1)], 4); }).sort((a, b) => a - b);
  };
  const nearest = (x, C) => { let b = 0, bd = Infinity; C.forEach((c, k) => { let d = 0; for (let j = 0; j < c.length; j++) d += (x[j] - c[j]) ** 2; if (d < bd) { bd = d; b = k; } }); return b; };
  const centroids = (X, labels) => { const m = new Map(); X.forEach((x, i) => { if (labels[i] < 0) return; const s = m.get(labels[i]) || [new Array(x.length).fill(0), 0]; x.forEach((v, j) => (s[0][j] += v)); s[1]++; m.set(labels[i], s); }); return [...m.entries()].sort((a, b) => a[0] - b[0]).map(([l, [s, c]]) => ({ l, c: s.map((v) => v / c) })); };

  // ---- k-means ----------------------------------------------------------------------------------------------------
  WF.block('kmeans', {
    cat: 'Unsupervised', label: 'k-means', icon: 'target', order: 10,
    blurb: 'Finds k groups: each row joins its nearest centre, each centre moves to the mean of its rows, and that repeats until nothing changes. k-means++ spreads the first centres apart.',
    inputs: [{ id: 'in', label: 'table', type: 'table' }],
    outputs: [{ id: 'out', label: 'table + cluster', type: 'table' }],
    params: [
      { id: 'k', label: 'Clusters (k)', kind: 'int', def: 3, min: 1, max: 12 },
      { id: 'init', label: 'First centres', kind: 'choice', def: 'k-means++', options: [['k-means++', 'k-means++ (spread out)'], ['random', 'random rows']] },
      { id: 'maxIter', label: 'At most this many rounds', kind: 'int', def: 100, min: 1, max: 1000 },
      { id: 'elbow', label: 'Draw the elbow (k = 1 to 8)', kind: 'bool', def: true },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const v = ctx.in.in, p = ctx.params, X = WF.clusterInput(v, 'k-means');
      const E = WF.needEngine('KMeans', 'k-means');
      const km = new E({ k: Math.min(p.k, X.length), init: p.init, maxIter: p.maxIter, seed: p.seed, nInit: 1 }).fit(X);
      const names = km.centers.map((_, i) => String(i));
      const out = WF.withClusters(v, km.labels, names, (Y) => km.predict(Y), 'cluster');
      let elbow = null;
      if (p.elbow) { try { elbow = S.ML.elbow(X, Math.min(8, X.length), p.seed); } catch (e) { elbow = null; } }
      return { out: { out }, km, elbow };
    },
    view(r, ctx) {
      const u = U(), v = ctx.in.in, km = r.km, pl = WF.clusterPlane(v);
      // an update keeps the assignment before it; its inertia is measured with the moved centres
      let lastLab = null;
      const steps = (km.trace || []).filter((s) => s.kind === 'assign' || s.kind === 'update').slice(0, 40).map((s) => {
        if (s.labels) lastLab = s.labels;
        const inertia = s.inertia !== undefined ? s.inertia : lastLab ? v.X.reduce((a, x, i) => a + s.centers[lastLab[i]].reduce((t, c, j) => t + (x[j] - c) ** 2, 0), 0) : null;
        return { kind: s.kind, iter: s.iter, inertia: u.r(inertia, 5), shift: u.r(s.shift, 4), centers: u.round2(pl.proj(s.centers)), labels: lastLab ? pl.keep.map((i) => lastLab[i]) : null };
      });
      const init = (km.trace || []).find((s) => s.kind === 'assign');
      return { kind: 'kmeans', twoD: v.features.length === 2, k: km.centers.length, n: v.X.length, P: pl.P, names: pl.names, how: pl.how, y: pl.y, classes: v.classes,
        labels: pl.keep.map((i) => km.labels[i]), centers: u.round2(pl.proj(km.centers)), start: init ? u.round2(pl.proj(init.centers)) : null, steps,
        inertia: u.r(km.inertia, 5), iterations: km.nIter, sizes: km.centers.map((_, c) => km.labels.filter((l) => l === c).length),
        silhouette: silhouette(v.X, km.labels), ari: agreement(v, km.labels), elbow: r.elbow ? r.elbow.map((x) => u.r(x, 5)) : null, chosen: km.centers.length };
    },
    summary(r) { return r.km.centers.length + ' clusters · inertia ' + (+r.km.inertia.toPrecision(3)); },
  });

  // ---- DBSCAN -----------------------------------------------------------------------------------------------------
  WF.block('dbscan', {
    cat: 'Unsupervised', label: 'DBSCAN', icon: 'target', order: 20,
    blurb: 'Groups by density: a core point has at least minPts neighbours within ε, a cluster is core points that reach each other, border points hang on its edge, and the rest is noise. It finds odd shapes and needs no k.',
    inputs: [{ id: 'in', label: 'table', type: 'table' }],
    outputs: [{ id: 'out', label: 'table + cluster', type: 'table' }],
    params: [
      { id: 'eps', label: 'ε (neighbourhood radius)', kind: 'number', def: 0.3, min: 0.001, max: 100, log: true },
      { id: 'minPts', label: 'minPts (neighbours to be a core point)', kind: 'int', def: 5, min: 1, max: 100 },
    ],
    run(ctx) {
      const v = ctx.in.in, p = ctx.params, X = WF.clusterInput(v, 'DBSCAN');
      if (X.length > 3000) throw new Error('DBSCAN compares every pair of rows: keep it to 3000 rows here');
      const E = WF.needEngine('DBSCAN', 'DBSCAN');
      const db = new E({ eps: p.eps, minSamples: p.minPts }).fit(X);
      const names = ['noise'].concat(Array.from({ length: db.nClusters }, (_, i) => String(i)));
      // a new row joins the cluster of the nearest core point within ε, else it is noise
      const cores = db.coreIndices.map((i) => X[i]), coreLab = db.coreIndices.map((i) => db.labels[i]);
      const place = (Y) => Y.map((y) => { let b = -1, bd = p.eps * p.eps; cores.forEach((c, k) => { let d = 0; for (let j = 0; j < c.length; j++) d += (y[j] - c[j]) ** 2; if (d <= bd) { bd = d; b = coreLab[k]; } }); return b < 0 ? 'noise' : String(b); });
      const out = WF.withClusters(v, db.labels.map((l) => (l < 0 ? 'noise' : String(l))), names, place, 'cluster');
      return { out: { out }, db };
    },
    view(r, ctx) {
      const u = U(), v = ctx.in.in, db = r.db, pl = WF.clusterPlane(v);
      const grow = (db.trace || []).filter((s) => s.kind === 'cluster').slice(0, 30).map((s) => ({ cluster: s.cluster, labels: pl.keep.map((i) => s.labels[i]) }));
      return { kind: 'dbscan', n: v.X.length, P: pl.P, names: pl.names, how: pl.how, y: pl.y, classes: v.classes, eps: ctx.params.eps, minPts: ctx.params.minPts, twoD: v.features.length === 2,
        labels: pl.keep.map((i) => db.labels[i]), kinds: pl.keep.map((i) => db.kinds[i]), clusters: db.nClusters, noise: db.labels.filter((l) => l < 0).length, core: db.coreIndices.length,
        border: db.kinds.filter((k) => k === 'border').length, grow, silhouette: silhouette(v.X, db.labels), ari: agreement(v, db.labels), kdist: kDistance(v.X, ctx.params.minPts) };
    },
    summary(r) { return r.db.nClusters + ' clusters · ' + r.db.labels.filter((l) => l < 0).length + ' noise'; },
  });

  // ---- hierarchical -------------------------------------------------------------------------------------------------
  WF.block('hierarchical', {
    cat: 'Unsupervised', label: 'Hierarchical clustering', icon: 'layers', order: 30,
    blurb: 'Starts with every row on its own and merges the two closest groups, again and again, until one is left. The dendrogram records every merge at its height; cutting it at a height gives the clusters.',
    inputs: [{ id: 'in', label: 'table', type: 'table' }],
    outputs: [{ id: 'out', label: 'table + cluster', type: 'table' }],
    params: [
      { id: 'linkage', label: 'Distance between groups', kind: 'choice', def: 'ward', options: [['ward', 'Ward (least growth in variance)'], ['average', 'average'], ['complete', 'complete (farthest pair)'], ['single', 'single (closest pair)']] },
      { id: 'cut', label: 'Cut by', kind: 'choice', def: 'k', options: [['k', 'the number of clusters'], ['height', 'a height']] },
      { id: 'k', label: 'Clusters', kind: 'int', def: 3, min: 1, max: 30, show: (p) => p.cut === 'k' },
      { id: 'height', label: 'Cut height', kind: 'number', def: 1, min: 0, max: 1e6, show: (p) => p.cut === 'height' },
    ],
    run(ctx) {
      const v = ctx.in.in, p = ctx.params, X = WF.clusterInput(v, 'hierarchical clustering');
      if (X.length > 400) throw new Error('hierarchical clustering works on up to 400 rows here (it has ' + X.length + '): split off fewer rows first');
      const E = WF.needEngine('AgglomerativeClustering', 'hierarchical clustering');
      const hc = new E(p.cut === 'height' ? { linkage: p.linkage, distanceThreshold: p.height } : { linkage: p.linkage, nClusters: Math.min(p.k, X.length) }).fit(X);
      const cs = centroids(X, hc.labels);
      const names = cs.map((c) => String(c.l));
      const out = WF.withClusters(v, hc.labels, names, (Y) => Y.map((y) => cs[nearest(y, cs.map((c) => c.c))].l), 'cluster');
      return { out: { out }, hc };
    },
    view(r, ctx) {
      const u = U(), v = ctx.in.in, hc = r.hc, pl = WF.clusterPlane(v), n = v.X.length;
      const cutAt = ctx.params.cut === 'height' ? ctx.params.height : (() => { const m = hc.merges, s = n - hc.nClusters; return s <= 0 ? 0 : s >= m.length ? m[m.length - 1].dist * 1.05 : (m[s - 1].dist + m[s].dist) / 2; })();
      return { kind: 'hierarchical', n, linkage: ctx.params.linkage, merges: hc.merges.map((m) => [m.a, m.b, u.r(m.dist, 5), m.size]), labels: hc.labels, cut: u.r(cutAt, 5), clusters: hc.nClusters,
        y: v.task === 'classification' ? v.y.slice() : null, classes: v.classes, P: pl.P, keep: pl.keep, names: pl.names, how: pl.how, silhouette: silhouette(v.X, hc.labels), ari: agreement(v, hc.labels) };
    },
    summary(r) { return r.hc.nClusters + ' clusters · ' + r.hc.merges.length + ' merges'; },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
