// A UMAP-style embedding: a fuzzy k-nearest-neighbour graph (each point's own distance scale), then a
// layout by stochastic gradient descent that pulls graph neighbours together and pushes random pairs
// apart. It follows the published recipe closely but is not meant to reproduce any library bit for bit.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);

  // fit 1 / (1 + a d^(2b)) to the target curve set by minDist and spread (Gauss-Newton on a grid)
  ML.umapAB = function (minDist = 0.1, spread = 1) {
    const xs = ML.range(300).map((i) => ((i + 1) * 3 * spread) / 300), ys = xs.map((x) => (x < minDist ? 1 : Math.exp(-(x - minDist) / spread)));
    let a = 1, b = 1;
    for (let it = 0; it < 200; it++) {
      let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
      xs.forEach((x, i) => {
        const x2b = Math.pow(x, 2 * b), den = 1 + a * x2b, r = 1 / den - ys[i];
        const da = -x2b / (den * den), db = (-a * x2b * 2 * Math.log(x)) / (den * den);
        g0 += r * da; g1 += r * db; h00 += da * da; h01 += da * db; h11 += db * db;
      });
      const det = h00 * h11 - h01 * h01;
      if (Math.abs(det) < 1e-300) break;
      const sa = (h11 * g0 - h01 * g1) / det, sb = (h00 * g1 - h01 * g0) / det;
      a -= sa; b -= sb;
      if (Math.abs(sa) + Math.abs(sb) < 1e-12) break;
    }
    return { a, b };
  };

  // the fuzzy graph: for each point its k neighbours, rho (nearest distance) and sigma (scale)
  ML.fuzzyGraph = function (X, k) {
    const n = X.length, target = Math.log2(k), edges = new Map(), rhos = [], sigmas = [];
    X.forEach((a, i) => {
      const nb = ML.range(n).filter((j) => j !== i).map((j) => ({ j, d: Math.sqrt(ML.sqdist(a, X[j])) })).sort((u, v) => u.d - v.d || u.j - v.j).slice(0, k);
      const rho = nb[0].d;
      let lo = 0, hi = Infinity, sigma = 1;
      for (let it = 0; it < 64; it++) {
        const s = ML.sum(nb.map((e) => Math.exp(-Math.max(0, e.d - rho) / sigma)));
        if (Math.abs(s - target) < 1e-5) break;
        if (s > target) { hi = sigma; sigma = (lo + hi) / 2; } else { lo = sigma; sigma = hi === Infinity ? sigma * 2 : (lo + hi) / 2; }
      }
      rhos.push(rho); sigmas.push(sigma);
      nb.forEach((e) => edges.set(i * n + e.j, Math.exp(-Math.max(0, e.d - rho) / sigma)));
    });
    // symmetrise with the fuzzy union a + b - ab
    const out = [];
    for (const [key, w] of edges) {
      const i = Math.floor(key / n), j = key % n, back = edges.get(j * n + i) || 0;
      if (back && j < i) continue;
      out.push({ i, j, w: w + back - w * back });
    }
    return { edges: out, rhos, sigmas };
  };

  ML.UMAP = function (params) {
    const p = Object.assign({ nNeighbors: 15, minDist: 0.1, spread: 1, nEpochs: 200, learningRate: 1, negativeSamples: 5, seed: 0, init: 'pca', snapshotEvery: 10 }, params || {});
    const o = { params: p };
    o.fitTransform = function (X) {
      const n = X.length, rng = ML.rng(p.seed), { a, b } = ML.umapAB(p.minDist, p.spread);
      const G = ML.fuzzyGraph(X, Math.min(p.nNeighbors, n - 1));
      o.graph = G; o.a = a; o.b = b;
      let Y;
      if (Array.isArray(p.init)) Y = p.init.map((r) => r.slice());
      else if (p.init === 'pca') {
        const Z = ML.PCA({ nComponents: 2, trace: false }).fitTransform(X), m = Math.max(...Z.flat().map(Math.abs)) || 1;
        Y = Z.map((r) => r.map((v) => (10 * v) / m + 1e-4 * rng.normal()));
      } else Y = X.map(() => [rng() * 20 - 10, rng() * 20 - 10]);
      const wmax = Math.max(...G.edges.map((e) => e.w)), every = G.edges.map((e) => wmax / e.w), next = every.slice();
      const clip = (v) => Math.max(-4, Math.min(4, v));
      o.snapshots = [{ kind: 'graph', say: 'The fuzzy neighbour graph has ' + G.edges.length + ' weighted edges; the layout starts from a scaled PCA projection.', epoch: 0, Y: Y.map((r) => r.slice()) }];
      for (let ep = 1; ep <= p.nEpochs; ep++) {
        const alpha = p.learningRate * (1 - (ep - 1) / p.nEpochs);
        G.edges.forEach((e, t) => {
          if (next[t] > ep) return;
          next[t] += every[t];
          const yi = Y[e.i], yj = Y[e.j];
          let d2 = ML.sqdist(yi, yj);
          if (d2 > 0) {
            const c = (-2 * a * b * Math.pow(d2, b - 1)) / (1 + a * Math.pow(d2, b));
            for (let k = 0; k < 2; k++) { const g = clip(c * (yi[k] - yj[k])); yi[k] += g * alpha; yj[k] -= g * alpha; }
          }
          for (let s = 0; s < p.negativeSamples; s++) {
            const m = rng.int(n);
            if (m === e.i) continue;
            const ym = Y[m];
            d2 = ML.sqdist(yi, ym);
            const c = d2 > 0 ? (2 * b) / ((0.001 + d2) * (1 + a * Math.pow(d2, b))) : 0;
            for (let k = 0; k < 2; k++) yi[k] += (c > 0 ? clip(c * (yi[k] - ym[k])) : 4) * alpha;
          }
        });
        if (ep % p.snapshotEvery === 0 || ep === p.nEpochs) o.snapshots.push({ kind: 'epoch', say: 'Epoch ' + ep + ': neighbours pulled together, random pairs pushed apart, step size ' + f(alpha) + '.', epoch: ep, Y: Y.map((r) => r.slice()) });
      }
      o.embedding = Y; o.trace = o.snapshots;
      return Y;
    };
    return o;
  };
})();
