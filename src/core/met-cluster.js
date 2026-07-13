// Clustering metrics: silhouette (per example and mean), Davies–Bouldin, the adjusted Rand index and normalised
// mutual information. Distances are Euclidean.
(function (S) {
  'use strict';
  const M = (S.Metrics = S.Metrics || {});
  const f = (x) => M.f(x), st = (k, s, d) => M.step(k, s, d);
  const dist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s); };

  /** silhouette(X, labels) → { value, perExample, a, b, trace }; s = (b − a) / max(a, b), 0 for a lone member */
  M.silhouette = (X, labels) => {
    const ls = M.labels(labels), ix = M.index(ls), K = ls.length, n = X.length;
    if (K < 2 || K > n - 1) throw new Error('silhouette needs between 2 and n − 1 clusters; there are ' + K);
    const lab = labels.map(ix), size = new Array(K).fill(0); lab.forEach((c) => size[c]++);
    const A = new Array(n), B = new Array(n), s = new Array(n);
    for (let i = 0; i < n; i++) {
      const tot = new Array(K).fill(0);
      for (let j = 0; j < n; j++) if (j !== i) tot[lab[j]] += dist(X[i], X[j]);
      const own = lab[i];
      A[i] = size[own] > 1 ? tot[own] / (size[own] - 1) : 0;
      B[i] = Math.min(...tot.map((t, c) => (c === own ? Infinity : t / size[c])));
      s[i] = size[own] > 1 ? M.div(B[i] - A[i], Math.max(A[i], B[i])) : 0;
    }
    const value = M.mean(s);
    let worst = 0; s.forEach((v, i) => { if (v < s[worst]) worst = i; });
    return { value, perExample: s, a: A, b: B, trace: [
      st('ab', 'For each example, a = mean distance to its own cluster and b = mean distance to the nearest other cluster; example 0 has a = ' + f(A[0]) + ', b = ' + f(B[0]) + ', s = ' + f(s[0]) + '.', { a: A, b: B }),
      st('worst', 'Example ' + worst + ' fits worst, with s = ' + f(s[worst]) + (s[worst] < 0 ? ': it is closer to another cluster than to its own.' : '.'), { worst }),
      st('mean', 'The silhouette score is the mean over ' + n + ' examples = ' + f(value) + ' (1 is tight and separate, near 0 is overlapping).', { value })] };
  };

  /** Davies–Bouldin: for each cluster the worst (spread_i + spread_j) / centre distance, averaged; lower is better */
  M.daviesBouldin = (X, labels) => {
    const ls = M.labels(labels), ix = M.index(ls), K = ls.length, d = X[0].length;
    const lab = labels.map(ix), cent = ls.map(() => new Array(d).fill(0)), size = new Array(K).fill(0);
    X.forEach((x, i) => { size[lab[i]]++; x.forEach((v, k) => (cent[lab[i]][k] += v)); });
    cent.forEach((c, k) => c.forEach((_, j) => (c[j] /= size[k])));
    const spread = new Array(K).fill(0); X.forEach((x, i) => (spread[lab[i]] += dist(x, cent[lab[i]])));
    spread.forEach((_, k) => (spread[k] /= size[k]));
    const cd = cent.map((a) => cent.map((b) => dist(a, b)));
    const tiny = (v) => Math.abs(v) < 1e-15;
    if (spread.every(tiny) || cd.every((r) => r.every(tiny))) return { value: 0, spread, centroids: cent, trace: [st('degenerate', 'All clusters are points or share a centre, so the index is 0.')] };
    const worst = spread.map((si, i) => Math.max(...spread.map((sj, j) => (i === j ? 0 : (si + sj) / (cd[i][j] === 0 ? Infinity : cd[i][j])))));
    const value = M.mean(worst);
    return { value, spread, centroids: cent, worst, trace: [
      st('spread', 'Average distance to the centre in each cluster: ' + spread.map((v, k) => ls[k] + ' → ' + f(v)).join(', ') + '.', { spread }),
      ...worst.map((w, i) => st('ratio', 'Cluster ' + ls[i] + ': its worst ratio (spread + spread) / centre distance is ' + f(w) + '.', { cluster: ls[i] })),
      st('mean', 'Davies–Bouldin = mean of the worst ratios = ' + f(value) + ' (lower means tighter, better separated clusters).', { value })] };
  };

  /** the contingency table between two labellings */
  M.contingency = (a, b) => {
    const la = M.labels(a), lb = M.labels(b), ia = M.index(la), ib = M.index(lb);
    const C = la.map(() => new Array(lb.length).fill(0)); a.forEach((v, i) => C[ia(v)][ib(b[i])]++);
    return { rows: la, cols: lb, table: C };
  };

  /** adjusted Rand index from the pair counts: 1 is identical up to renaming, about 0 is chance */
  M.adjustedRand = (a, b) => {
    M.check(a, b, 'adjusted Rand');
    const C = M.contingency(a, b).table, n = a.length;
    const nc = C.map((r) => M.sum(r)), nk = C[0] ? C[0].map((_, j) => M.sum(C.map((r) => r[j]))) : [];
    const sq = M.sum(C.map((r) => M.sum(r.map((v) => v * v))));
    const tp = sq - n, fp = M.sum(C.map((r) => M.sum(r.map((v, j) => v * nk[j])))) - sq, fn = M.sum(C.map((r, i) => M.sum(r.map((v) => v * nc[i])))) - sq;
    const tn = n * n - fp - fn - sq;
    const value = fn === 0 && fp === 0 ? 1 : (2 * (tp * tn - fn * fp)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
    return { value, pairs: { tp: tp / 2, fp: fp / 2, fn: fn / 2, tn: tn / 2 }, trace: [
      st('pairs', 'Of the ' + (n * (n - 1)) / 2 + ' pairs of examples, ' + tp / 2 + ' are together in both labellings, ' + tn / 2 + ' apart in both, and ' + (fp + fn) / 2 + ' disagree.', { tp: tp / 2, tn: tn / 2 }),
      st('ari', 'Adjusting the agreement for chance gives ARI = ' + f(value) + '.', { value })] };
  };

  const entropy = (counts, n) => -M.sum(counts.filter((c) => c > 0).map((c) => (c / n) * Math.log(c / n)));
  /** normalised mutual information with the arithmetic mean of the two entropies */
  M.nmi = (a, b) => {
    M.check(a, b, 'NMI');
    const { table: C, rows, cols } = M.contingency(a, b), n = a.length;
    if (rows.length === cols.length && rows.length <= 1) return { value: 1, trace: [st('trivial', 'Both labellings put everything in one group, which counts as a perfect match.')] };
    const pi = C.map((r) => M.sum(r)), pj = cols.map((_, j) => M.sum(C.map((r) => r[j])));
    let mi = 0;
    if (pi.length > 1 && pj.length > 1) for (let i = 0; i < C.length; i++) for (let j = 0; j < pj.length; j++) if (C[i][j]) mi += (C[i][j] / n) * Math.log((n * C[i][j]) / (pi[i] * pj[j]));
    mi = Math.max(0, mi);
    const ha = entropy(pi, n), hb = entropy(pj, n), value = mi === 0 ? 0 : mi / ((ha + hb) / 2);
    return { value, mi, ha, hb, trace: [
      st('entropy', 'Entropies: H(true) = ' + f(ha) + ', H(clusters) = ' + f(hb) + ' nats.', { ha, hb }),
      st('mi', 'Mutual information = ' + f(mi) + ' nats: how much knowing the cluster tells about the true group.', { mi }),
      st('nmi', 'NMI = MI / mean entropy = ' + f(value) + '.', { value })] };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
