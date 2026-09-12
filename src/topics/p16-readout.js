// Part 16, graph-level readout: classifying whole graphs (rings, stars, two clusters) by pooling the node vectors a
// GCN makes into one vector per graph. One test graph is followed through the pooling; renumbering its nodes leaves
// the pooled vector unchanged; then the training curve and a map of every graph's pooled vector.
(function (S) {
  'use strict';
  const P = S.P16, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const READOUTS = { mean: 'mean of the node vectors', sum: 'sum of the node vectors', max: 'largest value in each column' };
  const EPOCHS = 20;
  P.graphRun = (readout, seed) => P.once(['gr', readout, seed].join(':'), () => {
    const T = S.T, t = S.GNN.trainGraphs({ readout, seed, epochs: EPOCHS, lr: 0.03, n: 60, nTrain: 40, snapEvery: EPOCHS });
    t.run(EPOCHS);
    const pick = t.test[0], Hn = T.noGrad(() => T.relu(t.model.forward(pick.Xt, pick.adj))), d = Hn.shape[1], hv = Hn.values();
    const nodes = []; for (let i = 0; i < pick.n; i++) nodes.push(Array.from(hv.subarray(i * d, (i + 1) * d)));
    const pooled = Array.from(T.noGrad(() => S.GNN.readout(Hn, readout)).values());
    const probs = t.explain(0).find((s) => s.kind === 'classify').probs;
    // the same graph with its nodes renumbered (reversed), through the same network
    const perm = [...Array(pick.n).keys()].reverse(), inv = []; perm.forEach((p, i) => { inv[p] = i; });
    const g2 = { n: pick.n, edges: pick.edges.map(([a, b]) => [inv[a], inv[b]]), X: perm.map((p) => pick.X[p]) };
    const pooled2 = Array.from(T.noGrad(() => S.GNN.readout(T.relu(t.model.forward(T.tensor(g2.X), S.GNN.adjOf(g2))), readout)).values());
    const emb = S.GNN.pca2(t.snapshots[t.snapshots.length - 1].embeddings);
    return { history: t.history.map((h) => Object.assign({}, h)), acc: t.accuracy(), pick, nodes, pooled, pooled2, probs, emb, ys: t.set.graphs.map((g) => g.y), sizes: t.set.graphs.map((g) => g.n), nTrain: t.train.length, nTest: t.test.length, examples: [0, 1, 2].map((k) => t.set.graphs.find((g) => g.y === k)) };
  });

  S.topic({
    id: 'graph-readout', family: 'p16-readout', order: 10, title: 'Graph-level readout: one vector per graph',
    blurb: 'To label a whole graph (is this molecule toxic? is this network a ring or a star?) the node vectors a GCN makes have to be pooled into one vector of fixed length, whatever the number of nodes. The readout does it with a mean, a sum or a maximum over the nodes; because none of these cares about the order of the nodes, renumbering a graph cannot change its prediction. Then a dense layer classifies the pooled vector.',
    inputs: [
      { id: 'readout', kind: 'choice', label: 'Readout', def: 'mean', options: Object.keys(READOUTS).map((k) => ({ value: k, label: k + ': ' + READOUTS[k] })) },
      { id: 'seed', kind: 'number', label: 'Seed (the weights)', def: 1, min: 1, max: 999 },
    ],
    presets: [{ label: 'Mean readout', values: { readout: 'mean', seed: 1 } }, { label: 'Sum readout', values: { readout: 'sum', seed: 1 } }, { label: 'Max readout', values: { readout: 'max', seed: 2 } }],
    counters: [{ id: 'epoch', label: 'epochs' }],
    run(inp) {
      const F = H().frames(['epoch']), q = F.q, R = P.graphRun(inp.readout, inp.seed), fams = S.GNN.graphFamilies, gr = R.pick;
      // 1. the three families
      const gal = [];
      R.examples.forEach((ex, k) => {
        const g = Object.assign({ id: 'ro-ex' + k }, ex), pos = P.layout(g, { x: k * 260, y: 60, w: 220, h: 220 }, 11);
        gal.push(...P.draw(g, pos, { prefix: 'ro:g' + k, r: 11, size: 11, text: () => '', cls: () => 'c' + k }), H().text('ro:gt' + k, k * 260 + 110, 310, fams[k] + ' (' + ex.n + ' nodes)', 'c' + k, 14, 'middle'));
      });
      F.push([...gal, H().head('ro:h', 0, 26, 'Three kinds of graph')],
        `${P.qpl(q, R.ys.length, 'small graph')} of three kinds: rings, stars and two clusters joined by a bridge, with ${q.n(Math.min(...R.sizes))} to ${q.n(Math.max(...R.sizes))} nodes each. Every node starts with two numbers, one and its degree over four. The task is to name the kind of a whole graph: ${q.n(R.nTrain)} graphs to train on, ${q.n(R.nTest)} to test.`, { phase: 'start', nmin: Math.min(...R.sizes), nmax: Math.max(...R.sizes), count: R.ys.length, nTrain: R.nTrain, nTest: R.nTest });
      // 2. node vectors of one test graph
      const g = Object.assign({ id: 'ro-pick' }, gr), pos = P.layout(g, { x: 0, y: 60, w: 280, h: 280 }, 12), d = R.nodes[0].length;
      const cw = 15, top = 74, sx = 330, mx = Math.max(1e-9, ...R.nodes.flat(), ...R.pooled);
      const strip = (rows, prefix, y, cls) => rows.map((r, i) => ({ key: prefix + i, kind: 'raster', x: sx, y: y + i * (cw + 2), w: d * cw, h: cw, nx: d, ny: 1, vals: r, scheme: 'seq', max: mx, smooth: false, cls }));
      const rowNames = (rows, prefix, y, name) => rows.map((_, i) => H().text(prefix + i, sx - 8, y + i * (cw + 2) + 12, name(i), 'tick', 12, 'end'));
      const pooledY = top + gr.n * (cw + 2) + 30;
      const nodeItems = (order) => [...strip(order.map((i) => R.nodes[i]), 'ro:s', top, 'p16sim'), ...rowNames(order, 'ro:sn', top, (i) => 'node ' + i)];
      const bar = S.Kit.heat.bar({ prefix: 'ro:bar', x: sx + d * cw + 14, y: top, w: 10, h: 120, scheme: 'seq', max: mx, min: 0, fmt: (v) => H().fx(v, 2) });
      const ident = [...Array(gr.n).keys()];
      F.push([...P.draw(g, pos, { r: 12, size: 11 }), ...nodeItems(ident), H().text('ro:st', sx, top - 12, 'after two GCN layers: ' + d + ' numbers per node', 'axname', 13), ...bar, H().head('ro:h', 0, 26, 'A test graph: a ' + fams[gr.y])],
        `Follow one test graph, a ${fams[gr.y]} with ${P.qpl(q, gr.n, 'node')}. Two GCN layers turn each node’s two numbers into ${q.n(d)}, one row per node in the strip. A graph with more nodes would have more rows, but the classifier needs an input of fixed size.`, { phase: 'nodes', n: gr.n, d, y: gr.y });
      // 3. pooling
      const pooledRow = { key: 'ro:pool', kind: 'raster', x: sx, y: pooledY, w: d * cw, h: cw + 4, nx: d, ny: 1, vals: R.pooled, scheme: 'seq', max: mx, smooth: false, cls: 'p16sim' };
      const best = R.probs.indexOf(Math.max(...R.probs));
      const probBars = S.Kit.dt.hbars({ prefix: 'ro:pb', x: sx - 100, y: pooledY + 54, w: 340, rowH: 28, vals: R.probs, names: fams, max: 1, fmt: (v) => H().fx(v, 2), nameW: 100, cls: (k) => 'c' + k });
      F.push([...P.draw(g, pos, { r: 12, size: 11 }), ...nodeItems(ident), pooledRow, H().text('ro:pt', sx, pooledY - 8, inp.readout + ' readout: one row for the graph', 'strong', 13), H().text('ro:st', sx, top - 12, 'after two GCN layers: ' + d + ' numbers per node', 'axname', 13), ...bar, ...probBars, H().text('ro:pbt', sx, pooledY + 46, 'dense layer + softmax: the chance of each kind', 'axname', 13), H().head('ro:h', 0, 26, 'A test graph: a ' + fams[gr.y])],
        `The ${inp.readout} readout pools ${P.qpl(q, gr.n, 'row')} into one row of ${q.n(d)}, column by column (the ${READOUTS[inp.readout]}). A dense layer and a softmax turn it into chances: ${fams.map((f, k) => f + ' ' + q(R.probs[k], 2)).join(', ')}. The model says ${fams[best]}` + (best === gr.y ? ', which is right.' : `, but it is a ${fams[gr.y]}.`), { phase: 'pool', probs: R.probs, best, y: gr.y, n: gr.n, d, nodes: R.nodes, pooled: R.pooled, readout: inp.readout });
      // 4. renumbering
      const perm = ident.slice().reverse(), diff = Math.max(...R.pooled.map((v, j) => Math.abs(v - R.pooled2[j])));
      F.push([...P.draw(g, pos, { r: 12, size: 11, text: (i) => String(gr.n - 1 - i) }), ...nodeItems(perm), Object.assign({}, pooledRow, { vals: R.pooled2 }), H().text('ro:pt', sx, pooledY - 8, 'the same graph renumbered: the same row', 'strong', 13), H().text('ro:st', sx, top - 12, 'rows in the new order', 'axname', 13), ...bar, H().head('ro:h', 0, 26, 'Renumbered nodes')],
        `Renumber the nodes (last becomes first) and run the network again: the rows of the strip come out in a new order, but a ${inp.readout} over the rows does not care about order. ${diff === 0 ? 'The pooled rows are identical' : `The pooled rows differ by at most ${q.sg(diff, 2)}, rounding error`}, so the prediction cannot depend on how the nodes happened to be numbered.`, { phase: 'perm', diff, pooled: R.pooled, pooled2: R.pooled2 });
      // 5. training
      const ax = C().axes({ prefix: 'ro:ax', x: 40, y: 70, w: 300, h: 200, xr: [0, EPOCHS], yr: [0, 1], xticks: 5, yticks: 0.25, xname: 'epoch', yname: 'accuracy', tickSize: 12 });
      const hist = R.history;
      F.set('epoch', EPOCHS);
      const e = R.emb, ex = e.map((p) => p[0]), ey = e.map((p) => p[1]);
      const pad = (lo, hi) => [lo - 0.08 * (hi - lo || 1), hi + 0.08 * (hi - lo || 1)];
      const ax2 = C().axes({ prefix: 'ro:ax2', x: 450, y: 70, w: 280, h: 200, xr: pad(Math.min(...ex), Math.max(...ex)), yr: pad(Math.min(...ey), Math.max(...ey)), xticks: 0, yticks: 0, tickSize: 12, frame: 'box', grid: false });
      const dots = e.map((p, i) => ({ key: 'ro:e' + i, kind: 'dot', x: ax2.X(p[0]), y: ax2.Y(p[1]), r: 4.5, cls: 'sp c' + R.ys[i], audit: 'overlap-ok', layer: 'items' }));
      const last = hist[hist.length - 1];
      F.push([...ax.items, C().line(ax, 'ro:te', hist.map((h) => [h.step, h.testAcc]), 'p16test'), C().line(ax, 'ro:tr', hist.map((h) => [h.step, h.trainAcc]), 'p16train'), ...C().legend('ro:lg', 40, 330, [{ text: 'test accuracy', cls: 'p16test' }, { text: 'training accuracy', cls: 'p16train' }], { dir: 'col' }),
        ...ax2.items, ...dots, ...fams.map((f, k) => [{ key: 'ro:k' + k, kind: 'dot', x: 456 + k * 110, y: 325.5, r: 6, cls: 'swatch c' + k }, H().text('ro:kt' + k, 468 + k * 110, 330, f, 'legend', 13)]).flat(), H().head('ro:h', 0, 26, 'Trained for ' + EPOCHS + ' epochs'), H().text('ro:h2', 450, 52, 'every graph’s pooled vector, flattened to 2-D', 'axname', 13)],
        `After ${q.n(EPOCHS)} epochs the training graphs are ${q.pct(last.trainAcc)} right and the test graphs ${q.pct(last.testAcc)}. On the right, every graph’s pooled vector is flattened to two directions: the three kinds sit in their own regions, which is what lets one dense layer tell them apart.` + (inp.readout === 'mean' ? ' A mean cannot see how big a graph is; a sum can, which helps when size matters.' : inp.readout === 'sum' ? ' A sum also grows with the size of the graph, which is extra information but makes the numbers vary more.' : ' A maximum keeps the strongest signal in each column, such as the busiest hub of a star.'), { phase: 'trained', trainAcc: last.trainAcc, testAcc: last.testAcc, epochs: EPOCHS });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Readout', text: ['node vectors: H = GCN(X, Â), one row per node', 'mean: h_G = (1/n) Σᵢ hᵢ', 'sum:  h_G = Σᵢ hᵢ', 'max:  h_G[j] = maxᵢ hᵢ[j]', 'class = softmax(h_G W + b)', 'any order of the nodes gives the same h_G'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'pool', pick: 'option', once: true,
      ask: 'Renumber the nodes of this graph. The pooled row will…',
      options: ['stay the same', 'change a little', 'change completely'],
      answer: 'stay the same',
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'Why does a graph classifier need a readout?', options: ['graphs have different numbers of nodes, but the classifier needs a fixed-size input', 'to add self loops', 'to normalise the adjacency', 'to train faster'], answer: 0, why: 'Pooling over the nodes always gives one vector of the same length.' },
      { q: 'Which readout can tell a graph of 8 nodes from one of 16 with identical node vectors?', options: ['sum', 'mean', 'max', 'none of them'], answer: 0, why: 'The sum doubles; the mean and the maximum stay the same.' },
      { q: 'Why must a readout not depend on the order of the nodes?', options: ['node numbering is arbitrary, so the same graph would get different answers', 'it makes training faster', 'the order is always random', 'it must depend on it'], answer: 0, why: 'A graph has no natural first node.' },
      { q: 'Max readout over node vectors [1, 4], [3, 2], [2, 5] gives…', options: ['[3, 5]', '[2, 3.67]', '[6, 11]', '[2, 5]'], answer: 0, why: 'The largest value in each column.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
