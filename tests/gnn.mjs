// Graph neural networks: the normalised adjacencies, message passing traced per node against the matrix form,
// gradient checks of the GCN and GraphSAGE layers and the readouts, node classification on communities and
// citations, over-smoothing with depth, graph classification with a readout, and determinism.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T, N = S.GNN;
const t = harness('gnn');
const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
const timed = (name, budgetMs, fn) => {
  const t0 = cpu(), r = fn(), ms = cpu() - t0;
  console.log(`  ${name.padEnd(34)} ${r}  in ${(ms / 1000).toFixed(2)} s`);
  t.ok(ms < budgetMs, name + ' took ' + ms.toFixed(0) + ' ms (budget ' + budgetMs + ')');
};
const g = S.Data.communities(), n = g.n;

// ---- normalised adjacency ----
{
  const sym = N.normAdj(g, { kind: 'sym' }), M = sym.M.values(), mean = N.normAdj(g, { kind: 'mean' }).M.values();
  let symOk = true; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (Math.abs(M[i * n + j] - M[j * n + i]) > 1e-7) symOk = false;
  t.ok(symOk, 'D^−½ (A + I) D^−½ is symmetric');
  t.ok(sym.deg.every((d, i) => Math.abs(M[i * n + i] - 1 / d) < 1e-7), 'the diagonal holds 1 / degree (with the self loop)');
  let rows = true; for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += mean[i * n + j]; if (Math.abs(s - 1) > 1e-6) rows = false; }
  t.ok(rows, 'the mean aggregator\'s rows sum to 1');
  // the largest eigenvalue of the symmetric normalisation is 1 (power iteration)
  let v = new Array(n).fill(1), lam = 0;
  for (let k = 0; k < 300; k++) { const w = v.map((_, i) => v.reduce((s, x, j) => s + M[i * n + j] * x, 0)); lam = Math.hypot(...w) / Math.hypot(...v); v = w.map((x) => x / Math.hypot(...w)); }
  t.ok(Math.abs(lam - 1) < 1e-4, 'the largest eigenvalue of Â is ' + lam);
  t.ok(sym.trace.length === 2 && /√/.test(sym.trace[1].say), 'normalisation trace');
  const dg = N.adjacency(S.Data.citations()), A = dg.A;
  t.ok(A.every((r, i) => r.every((x, j) => x === A[j][i])), 'a directed graph is made symmetric');
}

// ---- message passing per node equals the matrix product ----
for (const kind of ['gcn', 'sage']) {
  const H = g.X.map((r) => r.slice(0, 3)), mp = N.messagePassing(g, H, { kind, nodes: [0, 5] });
  const M = N.normAdj(g, { kind: kind === 'gcn' ? 'sym' : 'mean' }).M, AH = T.matmul(M, T.tensor(H)).values();
  let ok = true; for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) if (Math.abs(mp.out[i][kind === 'sage' ? 3 + j : j] - AH[i * 3 + j]) > 1e-5) ok = false;
  t.ok(ok, kind + ': per-node message passing matches Â H');
  t.ok(mp.trace.length === 2 && mp.trace[0].node === 0 && /hears from/.test(mp.trace[0].say) && mp.trace[0].weights.length === mp.trace[0].from.length, kind + ': traced nodes with senders and weights');
}

// ---- gradient checks ----
T.float64(() => {
  const check = (name, params, lossOf) => {
    for (const p of params) p.grad = null;
    lossOf().backward();
    let bad = null;
    params.forEach((p, pi) => { for (let i = 0; i < Math.min(p.size, 12); i++) { const o = p.data[i]; p.data[i] = o + 1e-6; const a = T.noGrad(lossOf).item(); p.data[i] = o - 1e-6; const b = T.noGrad(lossOf).item(); p.data[i] = o; const num = (a - b) / 2e-6, ana = p.grad ? p.grad.data[i] : 0; if (Math.abs(num - ana) / Math.max(1, Math.abs(num)) > 1e-6 && !bad) bad = 'param ' + pi + ' element ' + i + ': ' + ana + ' vs ' + num; } });
    t.ok(!bad, name + ': ' + bad);
  };
  const small = S.Data.communities({ n: 9, k: 3, dim: 4, seed: 2 }), adj = N.adjOf(small), X = T.param(T.tensor(small.X)), R = T.normal([9, 3], 0, 1, 4);
  for (const kind of ['gcn', 'sage']) {
    const net = N.net({ kind, sizes: [4, 5, 3], seed: 3 });
    check(kind + ' layers (weights)', net.parameters(), () => T.sum(T.mul(net.forward(X, adj), R)));
    check(kind + ' layers (node features)', [X], () => T.sum(T.mul(net.forward(X, adj), R)));
    check(kind + ' with a cross-entropy on some nodes', net.parameters(), () => T.crossEntropy(T.take(net.forward(X, adj), [0, 4, 8]), [0, 1, 2]));
  }
  const H = T.param(T.normal([6, 4], 0, 1, 7)), r = T.normal([1, 4], 0, 1, 8);
  for (const k of ['mean', 'sum', 'max']) check(k + ' readout', [H], () => T.sum(T.mul(N.readout(H, k), r)));
});

// ---- node classification ----
timed('GCN on communities', 8000, () => {
  const acc = (kind) => [1, 2, 3, 4, 5].map((seed) => { const tr = N.trainNodes({ kind, seed, snapEvery: 1000 }); tr.run(200); return tr.accuracy(); });
  const gcn = acc('gcn'), mlp = acc('mlp'), sage = acc('sage'), m = (a) => a.reduce((x, y) => x + y) / a.length;
  t.ok(gcn[0] >= 0.9 && m(gcn) >= 0.9, 'GCN test accuracy on communities ' + gcn.map((a) => a.toFixed(2)));
  t.ok(m(gcn) > m(mlp) + 0.3, 'the graph helps: GCN ' + m(gcn).toFixed(2) + ' against an MLP on features alone ' + m(mlp).toFixed(2));
  t.ok(m(sage) > m(mlp) + 0.15, 'GraphSAGE ' + m(sage).toFixed(2) + ' also beats the MLP');
  return 'GCN ' + m(gcn).toFixed(2) + ', SAGE ' + m(sage).toFixed(2) + ', MLP ' + m(mlp).toFixed(2) + ' (4 labels per class)';
});
timed('GCN on citations', 5000, () => {
  const tr = N.trainNodes({ data: 'citations', seed: 1, snapEvery: 50 }); tr.run(200);
  const a = tr.accuracy();
  t.ok(a >= 0.9, 'GCN on citations ' + a);
  const s = tr.snapshots[tr.snapshots.length - 1];
  t.ok(s.embedding.length === 60 && s.embedding[0].length === 2 && s.pred.length === 60, 'snapshots carry a 2-D embedding and predictions');
  t.ok(tr.history.every((h) => Number.isFinite(h.loss)) && tr.history[199].loss < tr.history[0].loss, 'training loss falls');
  t.ok(tr.explain().length === 3, 'node explanation');
  return 'test accuracy ' + a.toFixed(3);
});
timed('over-smoothing with depth', 15000, () => {
  const rows = N.overSmoothing({ depths: [1, 2, 4, 8, 16], seed: 1 });
  t.ok(rows.every((r, i) => i === 0 || r.similarity > rows[i - 1].similarity), 'similarity rises with depth: ' + rows.map((r) => r.similarity.toFixed(2)));
  t.ok(rows[4].similarity > 0.95 && rows[4].spread < 0.3 * rows[0].spread, 'after 16 steps the nodes are nearly alike');
  t.ok(rows[4].accuracy < rows[1].accuracy - 0.2, 'a 16-layer GCN does worse than a 2-layer one: ' + rows.map((r) => r.accuracy.toFixed(2)));
  return 'similarity ' + rows.map((r) => r.similarity.toFixed(2)).join(' → ') + '; accuracy ' + rows.map((r) => r.accuracy.toFixed(2)).join(' ');
});
timed('graph classification with a readout', 12000, () => {
  const set = N.graphSet();
  t.ok(set.graphs.length === 90 && set.graphs.every((gr) => gr.n >= 8 && gr.n <= 16 && gr.X.length === gr.n), 'the graph set');
  const out = [];
  for (const readout of ['mean', 'sum', 'max']) {
    const tr = N.trainGraphs({ readout, seed: 1, set }); tr.run(60);
    t.ok(tr.accuracy() >= 0.85, readout + ' readout: test accuracy ' + tr.accuracy());
    out.push(readout + ' ' + tr.accuracy().toFixed(2));
  }
  return out.join(', ');
});

// ---- determinism ----
const run = (seed) => { const tr = N.trainNodes({ seed, snapEvery: 1000 }); tr.run(20); return tr.history.map((h) => h.loss); };
t.eq(run(3), run(3), 'node classification is deterministic');
t.ok(JSON.stringify(run(3)) !== JSON.stringify(run(4)), 'another seed gives another run');
t.ok(T.jobs.has('gnnNodes'), 'GNN job is registered');
t.done();
