// Part 16, a GCN layer written as matrices: A + I, its symmetric normalisation Â = D̂^−½ (A + I) D̂^−½ (or a row
// mean, or none), the features H times the weights W, the mix Â (H W), and the ReLU. One node's row is followed
// through every product.
(function (S) {
  'use strict';
  const P = S.P16, H = () => S.Kit.ml;
  const BOX = { x: 0, y: 60, w: 200, h: 220 };
  const NORMS = {
    sym: 'symmetric: 1 / √(d̂ᵢ d̂ⱼ) (GCN)',
    mean: 'row mean: 1 / d̂ᵢ',
    none: 'none: the raw A + I',
  };
  const mat = (t) => { const v = t.values(), [r, c] = t.shape, out = []; for (let i = 0; i < r; i++) out.push(Array.from(v.subarray(i * c, (i + 1) * c))); return out; };

  S.topic({
    id: 'gcn-layer', family: 'p16-gcn', order: 10, title: 'A GCN layer as matrices',
    blurb: 'A graph convolutional layer is message passing written with matrices. Add self loops to the adjacency matrix and scale each entry by one over the square root of both degrees, giving Â. Multiply the node features H by a weight matrix W, as a dense layer does for every node, then multiply by Â so each node mixes its neighbours’ results, and apply a ReLU. The scaling keeps busy nodes from swamping the rest: compare it with a plain row mean and with no scaling at all.',
    inputs: [
      { id: 'graph', kind: 'choice', label: 'Graph', def: 'friends', options: P.setOptions(['friends', 'star', 'ring', 'clusters']) },
      { id: 'norm', kind: 'choice', label: 'Normalisation', def: 'sym', options: Object.keys(NORMS).map((k) => ({ value: k, label: NORMS[k] })) },
      { id: 'node', kind: 'number', label: 'Row to follow', def: 2, min: 0, max: 15, check: (v, inp) => (v >= P.graph(inp.graph).n ? 'Row to follow: this graph has nodes 0 to ' + (P.graph(inp.graph).n - 1) : null) },
      { id: 'seed', kind: 'number', label: 'Seed (the weights W)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Friends, GCN scaling', values: { graph: 'friends', norm: 'sym', node: 2, seed: 1 } },
      { label: 'A star, GCN scaling at the hub', values: { graph: 'star', norm: 'sym', node: 0, seed: 1 } },
      { label: 'A star, no scaling at the hub', values: { graph: 'star', norm: 'none', node: 0, seed: 1 } },
      { label: 'Row mean', values: { graph: 'clusters', norm: 'mean', node: 0, seed: 3 } },
    ],
    counters: [{ id: 'step', label: 'products done' }],
    run(inp) {
      const T = S.T, g = P.graph(inp.graph), n = g.n, v = inp.node, F = H().frames(['step']), q = F.q;
      const L = S.GNN.adjacency(g, { selfLoops: true }), Araw = L.A, dh = L.deg;
      const Ahat = inp.norm === 'none' ? T.tensor(Araw) : S.GNN.normAdj(g, { kind: inp.norm, selfLoops: true }).M, Am = mat(Ahat);
      const net = S.GNN.net({ kind: 'gcn', sizes: [2, 2], seed: inp.seed }), dense = net.layers[0].mods[0], W = mat(dense.W);
      const Ht = T.tensor(g.X), Z = mat(dense.forward(Ht)), M = mat(T.matmul(Ahat, dense.forward(Ht))), R = mat(T.relu(T.tensor(M)));
      const pos = P.layout(g, BOX, 13), cw = n > 8 ? 26 : 30, ax = 240, ay = 84, zx = ax + n * (cw + 2) + 30, mx = zx + 2 * 54 + 34;
      const graph = (o) => P.draw(g, pos, Object.assign({ r: 13 }, o || {}));
      const nb = P.nbrs(g)[v], rowLit = (i) => i === v;
      const big = (vals, d, fmtv) => (vals.flat().some((x) => Math.abs(x) >= 10) ? 1 : d);
      const A = (vals, o) => P.matrix('gl:a', ax, ay, vals, Object.assign({ cw, fmt: (x) => (x === 0 ? '' : String(+x.toFixed(2)).replace(/^0\./, '.')), max: Math.max(1e-9, ...vals.flat().map(Math.abs)), size: 12, rowCls: (i) => (i === v ? 'hot' : ''), cls: (i, j) => (i === v && vals[i][j] ? 'lit' : '') }, o || {})).items;
      const narrow = (prefix, x, vals, title, o) => {
        const mxv = Math.max(1e-9, ...vals.flat().map(Math.abs));
        return P.matrix(prefix, x, ay, vals, Object.assign({ cw: 52, ch: cw, d: big(vals, 2), scheme: 'div', max: mxv, size: 12, rows: false, cols: ['1', '2'], title, titleCls: 'strong', cls: (i) => (i === v ? 'lit' : '') }, o || {})).items;
      };
      const label = (key, x, t) => H().text(key, x, ay - 30, t, 'strong', 14);
      const panel = (rows) => H().lines('gl:p', 0, ay + n * (cw + 2) + 40, rows, 13, 22);
      // 1. A + I
      F.push([...graph({ act: v, lit: new Set(nb) }), ...A(Araw, { fmt: (x) => String(x) }), label('gl:la', ax, 'A + I'), ...panel([['d̂ = row sums of A + I: ' + dh.join(', '), 'math']])],
        `A GCN starts from the links with self loops, A + I, so every node also feeds itself. The row sums are the degrees counting the loop, d̂, from ${q.n(Math.min(...dh))} to ${q.n(Math.max(...dh))}; node ${q.n(v)} has d̂ = ${q.n(dh[v])}.`, { phase: 'loops', dmin: Math.min(...dh), dmax: Math.max(...dh), dv: dh[v] });
      // 2. normalisation
      F.count('step', 0);
      const rs = Am.map((r) => r.reduce((s, x) => s + x, 0)), j0 = nb.length ? nb.reduce((b, u) => (dh[u] > dh[b] ? u : b), nb[0]) : v;
      const normSay = inp.norm === 'sym' ? `Each entry is divided by √(d̂ᵢ d̂ⱼ): the link from node ${q.n(j0)} (d̂ = ${q.n(dh[j0])}) to node ${q.n(v)} (d̂ = ${q.n(dh[v])}) gets ${q(Am[v][j0], 3)}. Row sums are no longer whole numbers; here they run from ${q(Math.min(...rs), 2)} to ${q(Math.max(...rs), 2)}, so no node’s total runs away.`
        : inp.norm === 'mean' ? `Each row is divided by its own d̂, so it sums to one: node ${q.n(v)} gives each of its ${q.n(dh[v])} entries ${q(Am[v][v], 3)}, a plain average of itself and its neighbours.`
        : `With no scaling Â is just A + I: node ${q.n(v)} will add up ${P.qpl(q, dh[v], 'row')} of numbers at full weight, while the quietest node adds only ${q.n(Math.min(...dh))}. Row sums run from ${q(Math.min(...rs), 0)} to ${q(Math.max(...rs), 0)}.`;
      F.push([...graph({ act: v, lit: new Set(nb) }), ...A(Am), label('gl:la', ax, 'Â: ' + NORMS[inp.norm].split(':')[0]), ...panel([['Â = ' + (inp.norm === 'sym' ? 'D̂^−½ (A + I) D̂^−½' : inp.norm === 'mean' ? 'D̂⁻¹ (A + I)' : 'A + I'), 'math'], ['row ' + v + ' of Â sums to ' + H().fx(rs[v], 2), 'legend']])],
        normSay, { phase: 'norm', w: Am[v][j0], j0, dj: dh[j0], rsMin: Math.min(...rs), rsMax: Math.max(...rs), rsV: rs[v], self: Am[v][v], dv: dh[v], dmin: Math.min(...dh) });
      // 3. H W
      F.count('step');
      const hv = g.X[v], zv = Z[v];
      F.push([...graph({ act: v }), ...narrow('gl:h', ax, g.X, 'H', { scheme: 'seq', cols: ['h₁', 'h₂'] }), ...P.matrix('gl:w', ax + 52 * 2 + 40, ay, W, { cw: 52, ch: cw, d: 2, scheme: 'div', max: Math.max(...W.flat().map(Math.abs)), size: 12, rows: ['h₁', 'h₂'], cols: ['1', '2'], title: 'W', titleCls: 'strong' }).items, ...narrow('gl:z', zx, Z, 'Z = H W'),
        ...panel([['z' + v + ' = ' + H().fx(hv[0], 2) + ' × ' + P.vec(W[0]) + ' + ' + H().fx(hv[1], 2) + ' × ' + P.vec(W[1]) + ' = ' + P.vec(zv, 3), 'math']])],
        `First every node’s features go through the same weights, as in a dense layer: Z = H W. For node ${q.n(v)}, ${q(hv[0], 2)} × ${P.qvec(q, W[0])} + ${q(hv[1], 2)} × ${P.qvec(q, W[1])} = ${P.qvec(q, zv, 3)}. W is shared by all nodes, so the layer has only ${q.n(W.flat().length)} weights whatever the size of the graph.`, { phase: 'hw', hv, W, zv, nw: W.flat().length });
      // 4. Â Z
      F.count('step');
      const terms = Am[v].map((a, j) => [a, j]).filter(([a]) => a !== 0);
      const termTxt = terms.map(([a, j]) => H().fx(a, 2) + '·z' + j).join(' + ');
      F.push([...graph({ act: v, lit: new Set(nb), litEdge: (a, b) => (a === v && nb.includes(b)) || (b === v && nb.includes(a)) }), ...A(Am), label('gl:la', ax, 'Â'), ...narrow('gl:z', zx, Z, 'Z', { cls: (i) => (Am[v][i] ? 'lit' : '') }), ...narrow('gl:m', mx, M, 'Â Z'),
        ...panel([['row ' + v + ' of Â Z: ' + (termTxt.length > 60 ? termTxt.slice(0, 58) + '…' : termTxt), 'math'], ['         = ' + P.vec(M[v], 3), 'math']])],
        `Then each node mixes: row ${q.n(v)} of Â picks out ${P.qpl(q, terms.length, 'row')} of Z (itself and its neighbours) with their weights, giving ${P.qvec(q, M[v], 3)}.` + (inp.norm === 'none' ? ` Without scaling the biggest output in the graph has size ${q(Math.max(...M.map((r) => Math.hypot(...r))), 2)}, at a busy node.` : ''), { phase: 'mix', terms: terms.length, mv: M[v], maxNorm: Math.max(...M.map((r) => Math.hypot(...r))) });
      // 5. ReLU
      F.count('step');
      const zeroed = M.flat().filter((x) => x < 0).length;
      F.push([...graph({ act: v }), ...narrow('gl:m', ax, M, 'Â Z'), ...narrow('gl:r', ax + 2 * 54 + 50, R, 'ReLU', { scheme: 'seq' }), H().text('gl:arrow', ax + 2 * 54 + 12, ay + (n * (cw + 2)) / 2, '→', 'strong', 18),
        ...panel([['H′ = ReLU(Â H W)', 'math'], [zeroed + ' of ' + 2 * n + ' numbers were negative and became 0', 'legend']])],
        `Last, the ReLU sets every negative number to zero: ${q.n(zeroed)} of the ${q.n(2 * n)} numbers here. Node ${q.n(v)} leaves the layer with ${P.qvec(q, R[v], 3)}. Stacking a second layer repeats all of this on these new features, reaching two links out.`, { phase: 'relu', zeroed, total: 2 * n, rv: R[v] });
      for (const f of F.list) f.narrowView = null;
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A GCN layer', text: ['Â = D̂^−½ (A + I) D̂^−½,  d̂ᵢ = Σⱼ (A + I)ᵢⱼ', 'Z = H W        (the same W for every node)', 'H′ = ReLU(Â Z) = ReLU(Â H W)', 'row v of Â H W: Σ_u Âᵥᵤ (h_u W)', 'weights: in × out, independent of the graph size'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'hw', pick: 'option', once: true,
      ask: 'After the ReLU, how many of the numbers will be zero?',
      options: (fr) => { const t = fr.find((f) => f.meta.phase === 'relu').meta; const a = t.zeroed; return [...new Set([a, Math.max(0, a - 2), a + 2, Math.min(t.total, a + 4)])].sort((x, y) => x - y).map(String); },
      answer: (fr) => String(fr.find((f) => f.meta.phase === 'relu').meta.zeroed),
      explain: (fr) => fr.find((f) => f.meta.phase === 'relu').say,
    }],
    quiz: [
      { q: 'Node i has 3 neighbours and node j has 1; they are linked. With self loops, the entry Âᵢⱼ of the GCN normalisation is…', options: ['1 / √(4 × 2) ≈ 0.354', '1 / 3', '1 / 4', '1'], answer: 0, why: 'd̂ᵢ = 4 and d̂ⱼ = 2.' },
      { q: 'A GCN layer maps 16 features to 8. How many weights does W have, for a graph of 1,000 nodes?', options: ['128 (plus 8 biases)', '8,000', '16,000', '1,000 × 128'], answer: 0, why: 'W is shared by every node: 16 × 8.' },
      { q: 'Why does Â (H W) equal (Â H) W?', options: ['matrix products can be grouped either way', 'because Â is symmetric', 'because of the ReLU', 'it does not'], answer: 0, why: 'Matrix multiplication is associative.' },
      { q: 'Without any normalisation, what happens at a hub with many links?', options: ['its outputs grow with its degree', 'it is ignored', 'its outputs shrink to zero', 'nothing different'], answer: 0, why: 'It adds up many full-weight rows.' },
      { q: 'With the row-mean normalisation, each row of Â sums to…', options: ['1', 'the degree', '0', '√d̂'], answer: 0, why: 'Each entry of row i is 1 / d̂ᵢ, and there are d̂ᵢ of them.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
