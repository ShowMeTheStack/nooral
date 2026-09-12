// Part 16, graph neural networks: the narration of every topic read back against an independent working-out. Rows
// and degrees from the edge lists, two-step walks by counting shared neighbours, every message weight from its
// formula, the GCN layer's products by hand-written loops, repeated propagation and cosine similarities from the
// edges, the readout from the node rows, and the trained runs replayed (the same seed must give the same numbers).
// Usage: node tests/claims-p16.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';

const P = S.P16;
// formula constants the narration may write as they are
const ALLOWED = [/1 \/ √\(/g, /1 \/ /g, /is 1 when/g, /else 0\b/g];
const near = (c, a, b, msg, tol) => c.ok(Math.abs(a - b) <= (tol || 1e-6) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const vnear = (c, a, b, msg, tol) => c.ok(a && b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= (tol || 1e-6) * Math.max(1, Math.abs(b[i]))), msg + ': got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b));
/** neighbour sets, both ways, from the edge list */
const nbrs = (g) => { const nb = Array.from({ length: g.n }, () => new Set()); for (const [a, b] of g.edges) if (a !== b) { nb[a].add(b); nb[b].add(a); } return nb; };
/** the GCN propagation matrix written out: (A + I) scaled by 1 / √(d̂ᵢ d̂ⱼ), or by rows, or not at all */
const ahat = (g, kind) => {
  const nb = nbrs(g), d = nb.map((s) => s.size + 1);
  return Array.from({ length: g.n }, (_, i) => Array.from({ length: g.n }, (_, j) => (i === j || nb[i].has(j) ? (kind === 'sym' ? 1 / Math.sqrt(d[i] * d[j]) : kind === 'mean' ? 1 / d[i] : 1) : 0)));
};
const mul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
const f32 = 1e-5;

const checks = {
  'graph-adjacency': {
    extra: [{ graph: 'ring' }, { graph: 'clusters', loops: 'yes' }, { graph: 'friends', loops: 'yes' }],
    check(c) {
      const g = P.graph(c.input.graph), nb = nbrs(g), dir = !!g.directed;
      const out = Array.from({ length: g.n }, (_, i) => [...new Set(g.edges.filter(([a]) => a === i).map(([, b]) => b))].sort((a, b) => a - b));
      phase(c.frames, 'row').forEach((f) => {
        const want = dir ? out[f.meta.node] : [...nb[f.meta.node]].sort((a, b) => a - b);
        c.eq(f.meta.nb, want, 'row ' + f.meta.node + ' neighbours');
        c.eq(f.meta.deg, want.length, 'row ' + f.meta.node + ' degree');
      });
      const s = phase(c.frames, 'sym')[0].meta, links = new Set(g.edges.map(([a, b]) => Math.min(a, b) + ',' + Math.max(a, b))).size;
      if (dir) { c.eq(s.total, g.edges.length, 'one-way ones'); c.eq(s.totU, 2 * links, 'two-way ones'); }
      else { c.eq(s.total, 2 * links, 'ones are twice the links'); c.eq(s.sumDeg, 2 * links, 'degrees add to twice the links'); }
      const q = phase(c.frames, 'square')[0].meta, shared = (i, j) => [...nb[i]].filter((k) => nb[j].has(k));
      c.eq(q.common, shared(q.i, q.j), 'shared neighbours of the pair');
      let best = -1; for (let i = 0; i < g.n; i++) for (let j = i + 1; j < g.n; j++) best = Math.max(best, shared(i, j).length);
      c.eq(q.count, best, 'the most two-step walks between two nodes');
      const L = phase(c.frames, 'loops')[0];
      if (c.input.loops === 'yes') { const d = nb.map((x) => x.size + 1); c.eq([L.meta.dmin, L.meta.dmax], [Math.min(...d), Math.max(...d)], 'degrees with loops'); c.eq(L.meta.total, 2 * links + g.n, 'ones with loops'); }
      else c.ok(!L, 'no loop frame without loops');
    },
  },
  'message-passing': {
    random: (r) => { const graph = ['friends', 'star', 'clusters', 'communities'][Math.floor(r() * 4)]; return { graph, node: Math.floor(r() * P.graph(graph).n), rule: ['gcn', 'sage', 'sum'][Math.floor(r() * 3)], rounds: r() < 0.5 ? '1' : '2' }; },
    randomRuns: 10,
    check(c) {
      const g = P.graph(c.input.graph), nb = nbrs(g), v = c.input.node, kind = c.input.rule, X = g.X.map((r) => r.slice(0, 2));
      const senders = (u) => (kind === 'gcn' ? [...nb[u], u] : [...nb[u]]);
      const weight = (u, w) => (kind === 'gcn' ? 1 / Math.sqrt((nb[u].size + 1) * (nb[w].size + 1)) : kind === 'sage' ? 1 / nb[u].size : 1);
      const gather = (Hm, u) => senders(u).reduce((s, w) => [s[0] + weight(u, w) * Hm[w][0], s[1] + weight(u, w) * Hm[w][1]], [0, 0]);
      const msgs = phase(c.frames, 'msg');
      c.eq(msgs.map((f) => f.meta.u).sort((a, b) => a - b), senders(v).sort((a, b) => a - b), 'who sends to the node');
      msgs.forEach((f) => { near(c, f.meta.w, weight(v, f.meta.u), 'weight from ' + f.meta.u); vnear(c, f.meta.msg, X[f.meta.u].map((x) => weight(v, f.meta.u) * x), 'message from ' + f.meta.u); });
      vnear(c, phase(c.frames, 'sum')[0].meta.agg, gather(X, v), 'gathered numbers');
      const m = X.map((_, u) => gather(X, u)), size = m.map((r) => Math.hypot(...r)), a = phase(c.frames, 'all')[0].meta;
      c.eq(a.big, size.indexOf(Math.max(...size)), 'largest result');
      vnear(c, a.mBig, m[a.big], 'largest result value');
      const r2 = phase(c.frames, 'round2')[0];
      if (c.input.rounds === '2') {
        vnear(c, r2.meta.m2, gather(m, v), 'second round');
        const reach = new Set([v]); for (const u of nb[v]) { reach.add(u); for (const w of nb[u]) reach.add(w); }
        c.eq(r2.meta.reach, reach.size, 'nodes within two links');
        c.eq(r2.meta.far, [...reach].filter((u) => u !== v && !nb[v].has(u)).sort((x, y) => x - y), 'nodes exactly two links away');
      } else c.ok(!r2, 'one round only');
    },
  },
  'gcn-layer': {
    random: (r) => { const graph = ['friends', 'star', 'ring', 'clusters'][Math.floor(r() * 4)]; return { graph, norm: ['sym', 'mean', 'none'][Math.floor(r() * 3)], node: Math.floor(r() * P.graph(graph).n), seed: 1 + Math.floor(r() * 500) }; },
    randomRuns: 10,
    check(c) {
      const g = P.graph(c.input.graph), nb = nbrs(g), v = c.input.node, A = ahat(g, c.input.norm), d = nb.map((s) => s.size + 1);
      const L = phase(c.frames, 'loops')[0].meta;
      c.eq([L.dmin, L.dmax, L.dv], [Math.min(...d), Math.max(...d), d[v]], 'degrees with self loops');
      const N = phase(c.frames, 'norm')[0].meta, rs = A.map((r) => r.reduce((s, x) => s + x, 0));
      near(c, N.w, A[v][N.j0], 'Â entry', f32); near(c, N.rsMin, Math.min(...rs), 'smallest row sum', f32); near(c, N.rsMax, Math.max(...rs), 'largest row sum', f32);
      c.eq(N.dj, d[N.j0], 'the neighbour’s degree');
      if (N.j0 !== v) c.ok([...nb[v]].every((u) => d[u] <= d[N.j0]), 'the busiest neighbour is shown');
      const hw = phase(c.frames, 'hw')[0].meta, Z = mul(g.X, hw.W);
      vnear(c, hw.zv, Z[v], 'z = h W', f32);
      const M = mul(A, Z), mix = phase(c.frames, 'mix')[0].meta;
      vnear(c, mix.mv, M[v], 'row of Â Z', f32);
      c.eq(mix.terms, nb[v].size + 1, 'rows mixed');
      near(c, mix.maxNorm, Math.max(...M.map((r) => Math.hypot(...r))), 'largest output', f32);
      const re = phase(c.frames, 'relu')[0].meta;
      c.eq(re.zeroed, M.flat().filter((x) => x < 0).length, 'numbers set to zero');
      vnear(c, re.rv, M[v].map((x) => Math.max(0, x)), 'ReLU row', f32);
    },
  },
  'node-classify': {
    extra: [{ data: 'citations', kind: 'sage', perClass: 4, seed: 5 }],
    check(c) {
      const g = P.nodeSet(c.input.data), s = phase(c.frames, 'start')[0].meta, k = g.classes.length;
      c.eq([s.nTrain, s.nTest], [c.input.perClass * k, g.n - c.input.perClass * k], 'labelled and test nodes');
      // the same training, run again from scratch with the engine, must give the same numbers
      const t = S.GNN.trainNodes({ data: g, kind: c.input.kind, perClass: c.input.perClass, seed: c.input.seed, snapEvery: 5 });
      t.run(200);
      c.eq(t.masks.train.filter((i) => true).length, s.nTrain, 'labelled count of the replay');
      c.ok(c.input.perClass * k === new Set(t.masks.train).size && [...new Set(t.masks.train.map((i) => g.y[i]))].length === k, 'labels spread evenly over the classes');
      phase(c.frames, 'epoch').forEach((f) => {
        if (!f.meta.E) return;
        const h = t.history[f.meta.E - 1];
        near(c, f.meta.loss, h.loss, 'loss at epoch ' + f.meta.E, 1e-9); near(c, f.meta.testAcc, h.testAcc, 'test accuracy at epoch ' + f.meta.E, 1e-12);
        c.eq(f.meta.wrong, Math.round((1 - f.meta.testAcc) * s.nTest), 'wrong test nodes at epoch ' + f.meta.E);
      });
      const cmp = phase(c.frames, 'compare')[0].meta;
      near(c, cmp.mine, t.accuracy(), 'final accuracy', 1e-12);
      const o = S.GNN.trainNodes({ data: g, kind: c.input.kind === 'mlp' ? 'gcn' : 'mlp', perClass: c.input.perClass, seed: c.input.seed, snapEvery: 5 });
      o.run(200);
      near(c, cmp.other, o.accuracy(), 'the other model', 1e-12);
    },
  },
  'over-smoothing': {
    check(c) {
      const g = P.nodeSet(c.input.data), A = ahat(g, 'sym');
      let Hm = g.X.map((r) => r.slice()), k = 0;
      const meanCos = (M) => { const nr = M.map((r) => Math.hypot(...r) || 1); let s = 0, t = 0; for (let i = 0; i < M.length; i++) for (let j = i + 1; j < M.length; j++) { s += M[i].reduce((a, v, q) => a + v * M[j][q], 0) / (nr[i] * nr[j]); t++; } return s / t; };
      phase(c.frames, 'depth').forEach((f) => {
        while (k < f.meta.d) { Hm = mul(A, Hm); k++; }
        near(c, f.meta.sim, meanCos(Hm), 'mean cosine after ' + f.meta.d, 1e-4);
      });
      const e = phase(c.frames, 'end')[0].meta, accs = phase(c.frames, 'depth').filter((f) => f.meta.d > 0).map((f) => f.meta.acc);
      near(c, e.bestAcc, Math.max(...accs), 'best accuracy', 1e-12);
      const t = S.GNN.trainNodes({ data: g, layers: 16, epochs: 100, seed: 1, perClass: 3, snapEvery: 1e9, dropout: 0 });
      t.run(100);
      near(c, e.lastAcc, t.accuracy(), 'the 16-layer GCN replayed', 1e-12);
    },
  },
  'graph-readout': {
    check(c) {
      const p = phase(c.frames, 'pool')[0].meta, cols = p.nodes[0].length;
      const pool = Array.from({ length: cols }, (_, j) => { const col = p.nodes.map((r) => r[j]); return p.readout === 'sum' ? col.reduce((a, b) => a + b, 0) : p.readout === 'max' ? Math.max(...col) : col.reduce((a, b) => a + b, 0) / col.length; });
      vnear(c, p.pooled, pool, 'pooled row from the node rows', 1e-5);
      near(c, p.probs.reduce((a, b) => a + b, 0), 1, 'chances add to one', 1e-5);
      c.eq(p.best, p.probs.indexOf(Math.max(...p.probs)), 'the predicted kind');
      const pm = phase(c.frames, 'perm')[0].meta;
      near(c, pm.diff, Math.max(...pm.pooled.map((v, j) => Math.abs(v - pm.pooled2[j]))), 'difference after renumbering', 1e-12);
      c.ok(pm.diff < 1e-5, 'renumbering changes nothing but rounding');
      const s = phase(c.frames, 'start')[0].meta;
      c.eq(s.nTrain + s.nTest, s.count, 'train and test make the set');
      const t = S.GNN.trainGraphs({ readout: c.input.readout, seed: c.input.seed, epochs: 20, lr: 0.03, n: 60, nTrain: 40, snapEvery: 20 });
      t.run(20);
      const tr = phase(c.frames, 'trained')[0].meta, h = t.history[19];
      near(c, tr.trainAcc, h.trainAcc, 'training accuracy replayed', 1e-12); near(c, tr.testAcc, h.testAcc, 'test accuracy replayed', 1e-12);
    },
  },
};

run('claims-p16', /^p16-/, checks, ALLOWED);
