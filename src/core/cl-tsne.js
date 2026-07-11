// t-SNE: exact for small n, Barnes-Hut (a quadtree summarises far points) for medium n in 2-D.
// The optimiser is gradient descent with momentum and per-coordinate gains, early exaggeration for the
// first 250 iterations, and a snapshot of the embedding every few iterations for the animation.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);
  const MEPS = 2.220446049250313e-16;

  // conditional probabilities p_j|i from squared distances, with each row's bandwidth found by bisection
  // so that the row's perplexity matches. rows[i] = [{j, d2}], returns per-row arrays of p.
  function conditional(rows, perplexity) {
    const desired = Math.log(perplexity), betas = [];
    const P = rows.map((row) => {
      let beta = 1, lo = -Infinity, hi = Infinity, p = null;
      for (let step = 0; step < 100; step++) {
        p = row.map((e) => Math.exp(-e.d2 * beta));
        let sum = ML.sum(p);
        if (sum === 0) sum = 1e-8;
        let sdp = 0;
        for (let t = 0; t < row.length; t++) { p[t] /= sum; sdp += row[t].d2 * p[t]; }
        const H = Math.log(sum) + beta * sdp, diff = H - desired;
        if (Math.abs(diff) <= 1e-5) break;
        if (diff > 0) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; }
        else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
      }
      betas.push(beta);
      return p;
    });
    return { P, betas };
  }

  // joint probabilities over all pairs (exact) as a full matrix
  ML.tsneP = function (X, perplexity) {
    const n = X.length, rows = X.map((a, i) => ML.range(n).filter((j) => j !== i).map((j) => ({ j, d2: Math.fround(ML.sqdist(a, X[j])) })));
    const { P: C, betas } = conditional(rows, perplexity), P = ML.la.zeros(n, n);
    rows.forEach((row, i) => row.forEach((e, t) => { P[i][e.j] += C[i][t]; P[e.j][i] += C[i][t]; }));
    let s = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s += P[i][j];
    s = Math.max(s, MEPS);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i][j] = i === j ? 0 : Math.max(P[i][j] / s, MEPS);
    return { P, sigmas: betas.map((b) => Math.sqrt(1 / (2 * b))) };
  };
  // sparse joint probabilities from each point's k = 3 * perplexity nearest neighbours
  ML.tsnePsparse = function (X, perplexity) {
    const n = X.length, k = Math.min(n - 1, Math.floor(3 * perplexity + 1));
    const rows = X.map((a, i) => ML.range(n).filter((j) => j !== i).map((j) => ({ j, d2: ML.sqdist(a, X[j]) })).sort((u, v) => u.d2 - v.d2).slice(0, k));
    const { P: C } = conditional(rows, perplexity), M = X.map(() => new Map());
    rows.forEach((row, i) => row.forEach((e, t) => { M[i].set(e.j, (M[i].get(e.j) || 0) + C[i][t]); M[e.j].set(i, (M[e.j].get(i) || 0) + C[i][t]); }));
    let s = 0;
    M.forEach((m) => m.forEach((v) => (s += v)));
    s = Math.max(s, MEPS);
    return M.map((m) => [...m].map(([j, v]) => ({ j, p: v / s })));
  };

  // exact KL divergence and gradient (2-D or any dimension)
  function exactGrad(Y, P, exag) {
    const n = Y.length, dim = Y[0].length, W = ML.la.zeros(n, n);
    let Z = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const w = 1 / (1 + ML.sqdist(Y[i], Y[j])); W[i][j] = W[j][i] = w; Z += 2 * w; }
    const G = Y.map(() => new Array(dim).fill(0));
    let kl = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const q = Math.max(W[i][j] / Z, MEPS), pij = P[i][j] * exag, m = (pij - q) * W[i][j];
      for (let a = 0; a < dim; a++) G[i][a] += 4 * m * (Y[i][a] - Y[j][a]);
      if (j > i) kl += 2 * pij * Math.log(Math.max(pij, MEPS) / q);
    }
    return { kl, G };
  }

  // Barnes-Hut in 2-D: a quadtree with centres of mass; a cell counts as one point when size / dist < angle
  function quadtree(Y) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of Y) { x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
    const size = Math.max(x1 - x0, y1 - y0) * (1 + 1e-6) + 1e-12;
    const root = { x: x0, y: y0, s: size, n: 0, cx: 0, cy: 0, pt: -1, kids: null };
    const insert = (node, i, depth) => {
      const p = Y[i];
      node.cx = (node.cx * node.n + p[0]) / (node.n + 1); node.cy = (node.cy * node.n + p[1]) / (node.n + 1); node.n++;
      if (node.n === 1) { node.pt = i; return; }
      if (depth > 40) return; // coincident points stay lumped together
      if (!node.kids) {
        const h = node.s / 2;
        node.kids = [0, 1, 2, 3].map((q) => ({ x: node.x + (q & 1 ? h : 0), y: node.y + (q & 2 ? h : 0), s: h, n: 0, cx: 0, cy: 0, pt: -1, kids: null }));
        if (node.pt >= 0) { const o = node.pt; node.pt = -1; insert(child(node, Y[o]), o, depth + 1); }
      }
      insert(child(node, p), i, depth + 1);
    };
    const child = (node, p) => node.kids[(p[0] >= node.x + node.s / 2 ? 1 : 0) + (p[1] >= node.y + node.s / 2 ? 2 : 0)];
    Y.forEach((_, i) => insert(root, i, 0));
    return root;
  }
  function bhGrad(Y, Ps, exag, angle) {
    const n = Y.length, root = quadtree(Y), neg = Y.map(() => [0, 0]);
    let Z = 0;
    for (let i = 0; i < n; i++) {
      const yi = Y[i], stack = [root];
      while (stack.length) {
        const c = stack.pop();
        if (c.n === 0 || (c.n === 1 && c.pt === i)) continue;
        const dx = yi[0] - c.cx, dy = yi[1] - c.cy, d2 = dx * dx + dy * dy;
        if (!c.kids || c.s * c.s < angle * angle * d2) {
          if (d2 === 0 && !c.kids) continue;
          const w = 1 / (1 + d2), cnt = c.pt === i ? c.n - 1 : c.n;
          Z += cnt * w; neg[i][0] += cnt * w * w * dx; neg[i][1] += cnt * w * w * dy;
        } else for (const k of c.kids) stack.push(k);
      }
    }
    const G = Y.map(() => [0, 0]);
    let kl = 0;
    for (let i = 0; i < n; i++) {
      for (const { j, p } of Ps[i]) {
        const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1], w = 1 / (1 + dx * dx + dy * dy), pe = p * exag;
        G[i][0] += 4 * pe * w * dx; G[i][1] += 4 * pe * w * dy;
        kl += pe * Math.log(Math.max(pe, MEPS) / Math.max(w / Z, MEPS));
      }
      G[i][0] -= (4 * neg[i][0]) / Z; G[i][1] -= (4 * neg[i][1]) / Z;
    }
    return { kl, G };
  }

  // t-SNE. params: perplexity, method ('exact' | 'barnes_hut'), init ('pca' | 'random' | array), maxIter...
  ML.TSNE = function (params) {
    const p = Object.assign({ nComponents: 2, perplexity: 30, earlyExaggeration: 12, learningRate: 'auto', maxIter: 1000, method: 'exact', angle: 0.5, init: 'pca', seed: 0, snapshotEvery: 10, minGradNorm: 1e-7, nIterWithoutProgress: 300 }, params || {});
    const o = { params: p };
    o.fitTransform = function (X) {
      const n = X.length, rng = ML.rng(p.seed), exact = p.method === 'exact';
      const lr = p.learningRate === 'auto' ? Math.max(n / p.earlyExaggeration / 4, 50) : p.learningRate;
      let Y;
      if (Array.isArray(p.init)) Y = p.init.map((r) => r.slice());
      else if (p.init === 'pca') { const Z = ML.PCA({ nComponents: p.nComponents, trace: false }).fitTransform(X), sd = Math.sqrt(ML.colVars(Z)[0]); Y = Z.map((r) => r.map((v) => (v / sd) * 1e-4)); }
      else Y = X.map(() => ML.range(p.nComponents).map(() => 1e-4 * rng.normal()));
      const P = exact ? ML.tsneP(X, p.perplexity).P : ML.tsnePsparse(X, p.perplexity);
      const grad = (exag) => (exact ? exactGrad(Y, P, exag) : bhGrad(Y, P, exag, p.angle));
      o.snapshots = [];
      o.history = [];
      let it = 0, kl = NaN;
      const phase = (maxIter, momentum, exag, noProgress) => {
        let update = Y.map((r) => r.map(() => 0)), gains = Y.map((r) => r.map(() => 1)), best = Infinity, bestIter = it;
        for (; it < maxIter; it++) {
          const g = grad(exag);
          kl = g.kl;
          let gn = 0;
          for (let i = 0; i < n; i++) for (let a = 0; a < Y[i].length; a++) {
            const gr = g.G[i][a];
            gains[i][a] = update[i][a] * gr < 0 ? gains[i][a] + 0.2 : Math.max(gains[i][a] * 0.8, 0.01);
            const gg = gr * gains[i][a];
            gn += gg * gg;
            update[i][a] = momentum * update[i][a] - lr * gg;
            Y[i][a] += update[i][a];
          }
          if (it % p.snapshotEvery === 0 || it === maxIter - 1) o.snapshots.push({ kind: 'iter', say: 'Iteration ' + it + (exag > 1 ? ' (early exaggeration x' + exag + ')' : '') + ': KL divergence ' + f(kl) + '.', iter: it, kl, Y: Y.map((r) => r.slice()) });
          if ((it + 1) % 50 === 0) {
            o.history.push({ iter: it, kl });
            if (kl < best) { best = kl; bestIter = it; } else if (it - bestIter > noProgress) { it++; return; }
            if (Math.sqrt(gn) <= p.minGradNorm) { it++; return; }
          }
        }
      };
      phase(Math.min(250, p.maxIter), 0.5, p.earlyExaggeration, 250);
      if (p.maxIter > 250) phase(p.maxIter, 0.8, 1, p.nIterWithoutProgress);
      o.embedding = Y; o.kl = kl; o.nIter = it; o.P = P; o.trace = o.snapshots;
      return Y;
    };
    return o;
  };

  // how well an embedding keeps each point's k nearest neighbours (1 is perfect)
  ML.trustworthiness = function (X, Y, k = 5) {
    const n = X.length, rankIn = X.map((a, i) => { const order = ML.range(n).filter((j) => j !== i).sort((u, v) => ML.sqdist(a, X[u]) - ML.sqdist(a, X[v]) || u - v); const r = new Array(n).fill(0); order.forEach((j, t) => (r[j] = t + 1)); return r; });
    let t = 0;
    for (let i = 0; i < n; i++) {
      const nnY = ML.range(n).filter((j) => j !== i).sort((u, v) => ML.sqdist(Y[i], Y[u]) - ML.sqdist(Y[i], Y[v]) || u - v).slice(0, k);
      for (const j of nnY) t += Math.max(0, rankIn[i][j] - k);
    }
    return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * t;
  };
})();
