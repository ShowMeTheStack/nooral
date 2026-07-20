// Graph neural networks (S.GNN), part 1: adjacency, normalisation, GCN and GraphSAGE-mean layers, message passing
// traced node by node, readouts, and a small graph-level dataset of our own.
//   S.GNN.adjacency(g, {selfLoops}) -> {A: rows of 0/1, deg}       (directed graphs are made symmetric)
//   S.GNN.normAdj(g, {kind: 'sym' | 'mean', selfLoops: true}) -> {M: tensor [n, n], deg, trace}
//      'sym'  Â = D^−½ (A + I) D^−½ (GCN);   'mean'  each row averages the neighbours (GraphSAGE-mean, no self loop)
//   S.GNN.messagePassing(g, H, {kind: 'gcn' | 'sage' | 'sum', nodes}) -> {out, trace}: per node, who sends what and
//      with which weight, and the aggregate (H: rows of numbers)
//   S.GNN.net({kind: 'gcn' | 'sage' | 'mlp', sizes: [in, h, ..., out], seed, dropout}) -> {forward(X, adj) -> logits,
//      hidden (the outputs of every layer), parameters()}
//   S.GNN.readout(H, 'mean' | 'sum' | 'max') -> [1, d]
//   S.GNN.graphSet({n: 90, seed}) -> graphs [{n, edges, X, y}] in three families: ring, star and two-cluster
//   S.GNN.pca2(rows) -> 2-D coordinates along the two main directions (for embedding maps)
(function (S) {
  'use strict';
  const N = (S.GNN = S.GNN || {});
  const fmt = (x) => S.T.fmt(x);

  N.adjacency = (g, o = {}) => {
    const A = Array.from({ length: g.n }, () => new Array(g.n).fill(0));
    for (const [a, b] of g.edges) { if (a === b) continue; A[a][b] = 1; A[b][a] = 1; }
    if (o.selfLoops) for (let i = 0; i < g.n; i++) A[i][i] = 1;
    return { A, deg: A.map((r) => r.reduce((s, v) => s + v, 0)) };
  };

  N.normAdj = (g, o = {}) => {
    const T = S.T, kind = o.kind || 'sym', self = o.selfLoops == null ? kind === 'sym' : o.selfLoops, { A, deg } = N.adjacency(g, { selfLoops: self }), n = g.n;
    const M = T.zeros([n, n]);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (A[i][j]) M.data[i * n + j] = kind === 'sym' ? 1 / Math.sqrt(deg[i] * deg[j]) : 1 / Math.max(1, deg[i]);
    const i0 = deg.indexOf(Math.max(...deg)), j0 = A[i0].findIndex((v, j) => v && j !== i0);
    const trace = kind === 'sym'
      ? [T.step('loops', (self ? 'Add a self loop to every node, so a node keeps some of its own features. ' : '') + 'Degrees then run from ' + Math.min(...deg) + ' to ' + Math.max(...deg) + '.', { deg }),
        T.step('norm', 'Each edge (i, j) gets weight 1 / √(d_i d_j): node ' + i0 + ' (degree ' + deg[i0] + ') and node ' + j0 + ' (degree ' + deg[j0] + ') are joined with weight ' + fmt(M.data[i0 * n + j0]) + '. Busy nodes are turned down so they do not dominate.', { M: Array.from(M.data) })]
      : [T.step('mean', 'Each node averages its ' + (self ? 'neighbours and itself' : 'neighbours') + ': row i has 1 / d_i in each neighbour\'s column, so node ' + i0 + ' gives each of its ' + deg[i0] + ' neighbours weight ' + fmt(1 / Math.max(1, deg[i0])) + '.', { deg })];
    return { M, deg, A, kind, trace };
  };

  N.messagePassing = (g, H, o = {}) => {
    const T = S.T, kind = o.kind || 'gcn', n = g.n, d = H[0].length;
    const { A, deg } = N.adjacency(g, { selfLoops: kind === 'gcn' }), out = [], trace = [];
    const nodes = o.nodes || [...Array(n).keys()];
    for (let v = 0; v < n; v++) {
      const nb = []; for (let u = 0; u < n; u++) if (A[v][u]) nb.push(u);
      const w = nb.map((u) => (kind === 'gcn' ? 1 / Math.sqrt(deg[v] * deg[u]) : kind === 'sage' ? 1 / Math.max(1, nb.length) : 1));
      const agg = new Array(d).fill(0);
      nb.forEach((u, k) => { for (let j = 0; j < d; j++) agg[j] += w[k] * H[u][j]; });
      const row = kind === 'sage' ? H[v].concat(agg) : agg;
      out.push(row);
      if (nodes.includes(v)) {
        const parts = nb.map((u, k) => fmt(w[k]) + '·h' + u);
        trace.push(T.step('node', 'Node ' + v + ' hears from ' + (kind === 'gcn' ? 'itself and ' : '') + (nb.filter((u) => u !== v).length) + ' neighbour' + (nb.length === 1 ? '' : 's') + ': ' + (parts.length ? parts.join(' + ') : 'nobody') + ' = [' + agg.map(fmt).join(', ') + ']' + (kind === 'sage' ? ', placed beside its own [' + H[v].map(fmt).join(', ') + ']' : '') + '.', { node: v, from: nb, weights: w, messages: nb.map((u) => H[u].slice()), aggregate: agg.slice() }));
      }
    }
    return { out, trace };
  };

  /** a GCN layer H' = Â H W + b, and a GraphSAGE-mean layer H' = H W_self + mean(neighbours) W_neigh + b */
  const layer = (kind, nIn, nOut, g) => {
    const T = S.T, nn = T.nn;
    if (kind === 'sage') {
      const self = new nn.Dense(nIn, nOut, { rng: g }), nb = new nn.Dense(nIn, nOut, { rng: g, bias: false });
      return { kind, mods: [self, nb], forward: (H, adj) => T.add(self.forward(H), nb.forward(T.matmul(adj.mean, H))) };
    }
    const d = new nn.Dense(nIn, nOut, { rng: g });
    return { kind, mods: [d], forward: kind === 'mlp' ? (H) => d.forward(H) : (H, adj) => T.matmul(adj.sym, d.forward(H)) };
  };
  N.net = (o = {}) => {
    const T = S.T, g = T.rng(o.seed || 1), kind = o.kind || 'gcn', sizes = o.sizes, drop = o.dropout || 0, dr = T.rng((o.seed || 1) * 3 + 1);
    const layers = []; for (let i = 0; i + 1 < sizes.length; i++) layers.push(layer(kind, sizes[i], sizes[i + 1], g));
    const net = {
      kind, sizes, layers, training: true, hidden: [],
      parameters: () => layers.flatMap((l) => l.mods.flatMap((m) => m.parameters())),
      forward(X, adj) {
        let H = X; net.hidden = [];
        layers.forEach((l, i) => {
          if (drop && net.training) H = T.dropout(H, drop, dr, true);
          H = l.forward(H, adj);
          if (i + 1 < layers.length) H = T.relu(H);
          net.hidden.push(H);
        });
        return H;
      },
    };
    return net;
  };
  /** both normalised adjacencies of a graph as tensors, ready for N.net */
  N.adjOf = (g) => ({ sym: N.normAdj(g, { kind: 'sym' }).M, mean: N.normAdj(g, { kind: 'mean' }).M });

  N.readout = (H, kind = 'mean') => { const T = S.T; return kind === 'sum' ? T.sum(H, 0, true) : kind === 'max' ? T.max(H, 0, true) : T.mean(H, 0, true); };

  N.graphFamilies = ['ring', 'star', 'two clusters'];
  N.graphSet = (o = {}) => {
    const r = S.Data.rng(o.seed || 1), n = o.n || 90, graphs = [];
    for (let i = 0; i < n; i++) {
      const y = i % 3, m = 8 + r.int(9), E = new Set(), add = (a, b) => { if (a !== b) E.add(Math.min(a, b) + ',' + Math.max(a, b)); };
      if (y === 0) { for (let k = 0; k < m; k++) add(k, (k + 1) % m); if (r() < 0.5) add(0, Math.floor(m / 2)); }
      else if (y === 1) { for (let k = 1; k < m; k++) add(0, k); for (let t = 0; t < 1 + r.int(2); t++) add(1 + r.int(m - 1), 1 + r.int(m - 1)); }
      else { const h = Math.floor(m / 2); for (let a = 0; a < m; a++) for (let b = a + 1; b < m; b++) if ((a < h) === (b < h) && r() < 0.6) add(a, b); add(0, h); for (let a = 1; a < m; a++) add(a, a < h ? 0 : h); }
      const edges = [...E].map((s) => s.split(',').map(Number)), deg = new Array(m).fill(0);
      edges.forEach(([a, b]) => { deg[a]++; deg[b]++; });
      graphs.push({ n: m, edges, y, X: deg.map((d) => [1, d / 4]), family: N.graphFamilies[y] });
    }
    // a seeded order, so a prefix is a fair sample of the three families
    r.shuffle(graphs);
    return { graphs, classes: N.graphFamilies.slice(), doc: 'Three families of small graphs (8 to 16 nodes): rings (sometimes with one chord), stars (a hub with a few extra leaf links) and two dense clusters joined at a bridge. Node features are [1, degree / 4].' };
  };

  N.pca2 = (rows) => {
    const n = rows.length, d = rows[0].length, mu = new Array(d).fill(0);
    rows.forEach((r) => r.forEach((v, j) => { mu[j] += v / n; }));
    const Xc = rows.map((r) => r.map((v, j) => v - mu[j])), C = Array.from({ length: d }, () => new Array(d).fill(0));
    Xc.forEach((r) => { for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += (r[a] * r[b]) / n; });
    const dirs = [];
    for (let k = 0; k < Math.min(2, d); k++) {
      let v = Array.from({ length: d }, (_, j) => Math.cos(j + 1 + k));
      for (let it = 0; it < 100; it++) {
        let w = C.map((row) => row.reduce((s, c, j) => s + c * v[j], 0));
        for (const u of dirs) { const p = w.reduce((s, x, j) => s + x * u[j], 0); w = w.map((x, j) => x - p * u[j]); }
        const nm = Math.hypot(...w) || 1; v = w.map((x) => x / nm);
      }
      dirs.push(v);
    }
    return Xc.map((r) => dirs.map((u) => r.reduce((s, x, j) => s + x * u[j], 0)).concat(d < 2 ? [0] : []));
  };
})(globalThis.Nooral = globalThis.Nooral || {});
