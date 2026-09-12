// Part 16, training graph networks on nodes: classification from a few labelled nodes (the colours spreading as the
// GCN learns, against a network that ignores the links), and over-smoothing (node features growing alike as the
// propagations pile up, and the accuracy of deeper and deeper GCNs).
(function (S) {
  'use strict';
  const P = S.P16, H = () => S.Kit.ml, C = () => S.Kit.curve;
  P.NODESETS = {
    communities: { label: 'three clubs (30 people, noisy profiles)', make: () => S.Data.communities({ n: 30, k: 3, seed: 2, noise: 1 }) },
    citations: { label: 'papers in three fields (36 papers, keyword flags)', make: () => S.Data.citations({ n: 36, seed: 1 }) },
  };
  P.nodeSet = (id) => P.once('ns:' + id, () => Object.assign({ id: 'ns-' + id }, P.NODESETS[id].make()));
  const KINDS = { gcn: 'GCN (links used)', sage: 'GraphSAGE-mean (links used)', mlp: 'MLP (links ignored)' };
  const BOX = { x: 0, y: 40, w: 400, h: 400 };
  /** a trained run, kept for the same settings */
  P.trainRun = (id, kind, perClass, seed, epochs) => P.once(['tr', id, kind, perClass, seed, epochs].join(':'), () => {
    const t = S.GNN.trainNodes({ data: P.nodeSet(id), kind, perClass, seed, snapEvery: 5 });
    t.run(epochs);
    return { history: t.history.map((h) => Object.assign({}, h)), snapshots: t.snapshots, masks: t.masks, acc: t.accuracy() };
  });

  S.topic({
    id: 'node-classify', family: 'p16-gcn', order: 20, title: 'Node classification from a few labels',
    blurb: 'Only a handful of nodes come with labels; the task is to label the rest. A GCN trained on those few still reads the whole graph: every layer mixes each node with its neighbours, so what it learns about the labelled nodes spreads along the links. Watch the predicted colours settle as training goes on, then compare with a network that sees each node’s own features only.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Graph', def: 'communities', options: Object.keys(P.NODESETS).map((k) => ({ value: k, label: P.NODESETS[k].label })) },
      { id: 'kind', kind: 'choice', label: 'Model', def: 'gcn', options: Object.keys(KINDS).map((k) => ({ value: k, label: KINDS[k] })) },
      { id: 'perClass', kind: 'number', label: 'Labelled nodes per class', def: 3, min: 1, max: 6 },
      { id: 'seed', kind: 'number', label: 'Seed (which nodes, the weights)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'GCN, 3 labels per club', values: { data: 'communities', kind: 'gcn', perClass: 3, seed: 1 } },
      { label: 'GCN, one label per club', values: { data: 'communities', kind: 'gcn', perClass: 1, seed: 1 } },
      { label: 'GraphSAGE-mean', values: { data: 'communities', kind: 'sage', perClass: 3, seed: 1 } },
      { label: 'Links ignored (MLP)', values: { data: 'communities', kind: 'mlp', perClass: 3, seed: 1 } },
      { label: 'Papers', values: { data: 'citations', kind: 'gcn', perClass: 2, seed: 2 } },
    ],
    counters: [{ id: 'epoch', label: 'epochs' }],
    run(inp) {
      const g = P.nodeSet(inp.data), n = g.n, F = H().frames(['epoch']), q = F.q, epochs = 200;
      const run = P.trainRun(inp.data, inp.kind, inp.perClass, inp.seed, epochs), train = new Set(run.masks.train), nTest = run.masks.test.length;
      const pos = P.layout(g, BOX, 11);
      const ax = C().axes({ prefix: 'nc:ax', x: 490, y: 76, w: 280, h: 170, xr: [0, epochs], yr: [0, 1], xticks: 50, yticks: 0.25, xname: 'epoch', yname: 'accuracy', tickSize: 12 });
      const draw = (pred) => P.draw(g, pos, { r: 11, size: 11, accept: (i) => train.has(i), cls: (i) => (pred ? 'c' + pred[i] + (pred[i] !== g.y[i] && !train.has(i) ? ' wrong' : '') : train.has(i) ? 'c' + g.y[i] : 'unl'), text: () => '' });
      const legend = [...C().legend('nc:lg', 490, 300, [{ text: 'test accuracy', cls: 'p16test' }, { text: 'training accuracy', cls: 'p16train' }], { dir: 'col' })];
      const classes = g.classes.map((c, k) => [{ key: 'nc:k' + k, kind: 'dot', x: 496, y: 368 + k * 22 - 4.5, r: 6, cls: 'swatch c' + k }, H().text('nc:kt' + k, 510, 368 + k * 22, c, 'legend', 13)]).flat();
      const keyRing = [{ key: 'nc:kr', kind: 'state', x: 496, y: 368 + g.classes.length * 22 - 4.5, r: 6, text: '', accept: true, cls: 'gnode' }, H().text('nc:krt', 510, 368 + g.classes.length * 22, 'labelled: used in training', 'legend', 13)];
      const at = (E) => run.history[E - 1];
      F.push([...draw(null), ...ax.items, ...legend, ...classes, ...keyRing, H().head('nc:h', 490, 30, KINDS[inp.kind])],
        `${P.qpl(q, n, 'node')} in ${P.qpl(q, g.classes.length, 'class', 'classes')}, but only ${P.qpl(q, train.size, 'node')} (${q.n(inp.perClass)} per class, double rings) have labels to train on. The other ${q.n(nTest)} are the test: the model must label them.` + (inp.kind === 'mlp' ? ' This model looks at each node’s own features only; the links are ignored.' : ' The links are used: every layer mixes each node with its neighbours.'), { phase: 'start', n, k: g.classes.length, nTrain: train.size, nTest, perClass: inp.perClass });
      for (const E of [0, 5, 20, 50, 100, 200]) {
        const snap = run.snapshots.find((s) => s.step === E), hist = run.history.slice(0, E);
        F.set('epoch', E);
        const items = [...draw(snap.pred), ...ax.items, ...legend, ...classes, ...keyRing, H().head('nc:h', 490, 30, KINDS[inp.kind] + ', epoch ' + E)];
        if (E) {
          items.push(C().line(ax, 'nc:te', [[0, run.snapshots[0].testAcc]].concat(hist.map((h) => [h.step, h.testAcc])), 'p16test'), C().line(ax, 'nc:tr', [[0, run.snapshots[0].trainAcc]].concat(hist.map((h) => [h.step, h.trainAcc])), 'p16train'));
          items.push(C().dot(ax, 'nc:d', E, snap.testAcc, 'p16test'));
        }
        const wrong = snap.pred.filter((p, i) => p !== g.y[i] && !train.has(i)).length;
        F.push(items, E === 0
          ? `Before training the weights are random: ${q.pct(snap.testAcc)} of the test nodes happen to get the right colour.`
          : `Epoch ${q.n(E)}: the loss on the labelled nodes is ${q(at(E).loss, 3)}. They are ${q.pct(at(E).trainAcc)} right, and the test nodes ${q.pct(at(E).testAcc)}: ${wrong ? P.qpl(q, wrong, 'test node') + ' still wrong (dark outline)' : 'every test node is right'}.`, { phase: 'epoch', E, loss: E ? at(E).loss : null, trainAcc: E ? at(E).trainAcc : snap.trainAcc, testAcc: E ? at(E).testAcc : snap.testAcc, wrong });
      }
      // the other side of the comparison
      const other = inp.kind === 'mlp' ? 'gcn' : 'mlp', o = P.trainRun(inp.data, other, inp.perClass, inp.seed, epochs), mine = run.acc;
      const last = run.snapshots.find((s) => s.step === epochs);
      const bars = S.Kit.dt.hbars({ prefix: 'nc:b', x: 490, y: 300, w: 300, rowH: 30, vals: [mine, o.acc], names: [inp.kind === 'mlp' ? 'MLP' : inp.kind === 'gcn' ? 'GCN' : 'SAGE', other === 'mlp' ? 'MLP' : 'GCN'], max: 1, fmt: (v) => H().pct(v), nameW: 60, cls: (i) => (i ? 'p16other' : 'p16mine') });
      const diff = (inp.kind === 'mlp' ? o.acc - mine : mine - o.acc);
      F.push([...draw(last.pred), ...ax.items, C().line(ax, 'nc:te', [[0, run.snapshots[0].testAcc]].concat(run.history.map((h) => [h.step, h.testAcc])), 'p16test'), C().line(ax, 'nc:tr', [[0, run.snapshots[0].trainAcc]].concat(run.history.map((h) => [h.step, h.trainAcc])), 'p16train'), ...bars, H().text('nc:bt', 490, 290, 'test accuracy after ' + epochs + ' epochs', 'axname', 13), H().head('nc:h', 490, 30, KINDS[inp.kind] + ', epoch ' + epochs)],
        `Trained the same way on the same ${P.qpl(q, train.size, 'labelled node')}, ${inp.kind === 'mlp' ? 'a GCN, which uses the links,' : 'a plain MLP, which ignores the links,'} reaches ${q.pct(o.acc)} on the test nodes against ${q.pct(mine)} here.` + (diff > 0.02 ? ' Neighbours tend to share a class, so mixing in theirs is worth a lot when each node’s own features are noisy.' : diff < -0.02 ? ' This time the node’s own features carry the class well, and mixing in neighbours costs a little.' : ' Here the two come out about level.'), { phase: 'compare', other: o.acc, mine, nTrain: train.size });
      for (const f of F.list) f.narrowView = { x: -10, y: 20, w: 420, h: 440 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Semi-supervised node classification', text: ['H¹ = ReLU(Â X W¹),  Z = Â H¹ W²', 'loss: cross-entropy on the labelled nodes only', 'but every node’s features flow through Â,', 'so unlabelled nodes shape the predictions too', 'test: accuracy on the unlabelled nodes'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr, inp) => f.meta.phase === 'start' && inp.kind !== 'mlp', pick: 'option', once: true,
      ask: 'After training, will this model beat one that ignores the links?',
      options: ['yes, clearly', 'about the same', 'no, it will do worse'],
      answer: (fr) => { const m = fr[fr.length - 1].meta, d = m.mine - m.other; return d > 0.02 ? 'yes, clearly' : d < -0.02 ? 'no, it will do worse' : 'about the same'; },
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'In this kind of training, the loss is computed on…', options: ['the labelled nodes only', 'every node', 'the test nodes', 'the links'], answer: 0, why: 'Only labelled nodes have a target; the others still feed the forward pass.' },
      { q: 'How can an unlabelled node help, if it has no target?', options: ['its features reach labelled neighbours through Â', 'it cannot', 'it is given a random label', 'it is removed'], answer: 0, why: 'Message passing mixes features across links, labelled or not.' },
      { q: 'A two-layer GCN predicts a node’s class from…', options: ['nodes up to two links away', 'only the node itself', 'only its labelled neighbours', 'the whole graph equally'], answer: 0, why: 'Each layer reaches one hop further.' },
      { q: 'When does using the links help most?', options: ['when linked nodes tend to share a class and own features are noisy', 'when links join nodes of different classes at random', 'when there are no links', 'never'], answer: 0, why: 'Neighbours then carry extra evidence for the class.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const DEPTHS = [0, 1, 2, 4, 8, 16];
  P.smooth = (id) => P.once('sm:' + id, () => {
    const g = P.nodeSet(id), zero = S.GNN.overSmoothing({ data: g, depths: [0], train: false })[0];
    const rest = S.GNN.overSmoothing({ data: g, depths: DEPTHS.slice(1), epochs: 100, seed: 1, perClass: 3 });
    return [zero].concat(rest);
  });
  /** cosine similarity between every pair of rows, the rows taken in class order */
  P.cosines = (Hm, order) => {
    const nr = Hm.map((r) => Math.hypot(...r) || 1);
    return order.map((i) => order.map((j) => Hm[i].reduce((s, v, k) => s + v * Hm[j][k], 0) / (nr[i] * nr[j])));
  };
  S.topic({
    id: 'over-smoothing', family: 'p16-gcn', order: 30, title: 'Over-smoothing: too many layers make every node alike',
    blurb: 'Each layer averages a node with its neighbours. A few rounds blur away noise within a community, which helps; many rounds keep averaging until every node holds nearly the same numbers, and the communities can no longer be told apart. The heat map shows how alike every pair of nodes is, grouped by class, after more and more propagations; the curve shows the similarity rising and the accuracy of deeper GCNs falling.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Graph', def: 'communities', options: Object.keys(P.NODESETS).map((k) => ({ value: k, label: P.NODESETS[k].label })) },
    ],
    presets: [{ label: 'Three clubs', values: { data: 'communities' } }, { label: 'Papers', values: { data: 'citations' } }],
    counters: [{ id: 'depth', label: 'propagations' }],
    run(inp) {
      const g = P.nodeSet(inp.data), n = g.n, F = H().frames(['depth']), q = F.q, rows = P.smooth(inp.data);
      const order = [...Array(n).keys()].sort((a, b) => g.y[a] - g.y[b] || a - b), side = 330, cell = side / n;
      const ax = C().axes({ prefix: 'os:ax', x: 470, y: 70, w: 290, h: 190, xr: [0, 4], yr: [0, 1], xticks: [0, 1, 2, 3, 4], yticks: 0.25, fmtX: (u) => String(2 ** u), xname: 'propagations (layers)', yname: 'value', tickSize: 12 });
      const lx = (d) => (d === 0 ? null : Math.log2(d));
      const counts = g.classes.map((_, c) => g.y.filter((y) => y === c).length);
      const bands = [];
      let acc0 = 0;
      let cut = '';
      counts.slice(0, -1).reduce((a, m) => { const at = P.r2(10 + (a + m) * cell); cut += `M${at},64 V${64 + side} M10,${P.r2(64 + (a + m) * cell)} H${10 + side} `; return a + m; }, 0);
      bands.push({ key: 'os:cut', kind: 'path', d: cut.trim(), cls: 'p16cut', audit: 'ignore', layer: 'labels' });
      counts.forEach((m, c) => { bands.push(H().text('os:b' + c, 10 + (acc0 + m / 2) * cell, 54, g.classes[c].length > 11 ? 'class ' + c : g.classes[c], 'c' + c, 12, 'middle')); acc0 += m; });
      DEPTHS.forEach((d, k) => {
        const r = rows[k], M = P.cosines(r.features, order);
        F.set('depth', d);
        const ras = { key: 'os:heat', kind: 'raster', x: 10, y: 64, w: side, h: side, nx: n, ny: n, vals: M.flat().map((v) => Math.max(0, v)), scheme: 'seq', max: 1, smooth: false, cls: 'p16sim' };
        const done = rows.slice(1, k + 1);
        const items = [ras, ...bands, H().head('os:h', 10, 26, d === 0 ? 'Raw features: how alike is each pair?' : 'After ' + H().pl(d, 'propagation')), ...S.Kit.heat.bar({ prefix: 'os:bar', x: 10, y: 410, w: 220, h: 12, scheme: 'seq', max: 1, min: 0, fmt: (v) => H().fx(v, 0) }), H().text('os:bl', 240, 421, 'cosine similarity (negatives as 0)', 'tick', 12), ...ax.items,
          ...C().legend('os:lg', 470, 312, [{ text: 'mean similarity of all pairs', cls: 'p16sim' }, { text: 'test accuracy of a GCN this deep', cls: 'p16test' }], { dir: 'col' })];
        if (done.length) {
          items.push(C().line(ax, 'os:s', done.map((x) => [lx(x.depth), x.similarity]), 'p16sim'), C().line(ax, 'os:a', done.map((x) => [lx(x.depth), x.accuracy]), 'p16test'));
          done.forEach((x, j) => items.push(C().dot(ax, 'os:sd' + j, lx(x.depth), x.similarity, 'p16sim', 4), C().dot(ax, 'os:ad' + j, lx(x.depth), x.accuracy, 'p16test', 4)));
        }
        F.push(items, d === 0
          ? `Each cell is the cosine similarity of two nodes’ features, with the nodes grouped by class. Before any propagation the features are noisy: the mean similarity over all pairs is ${q(r.similarity, 2)}, and only faint blocks show along the diagonal.`
          : `After ${P.qpl(q, d, 'propagation')} by Â the mean similarity is ${q(r.similarity, 2)} and a GCN with ${P.qpl(q, d, 'layer')} reaches ${q.pct(r.accuracy)} test accuracy.` + (d <= 2 ? ' Averaging within each class sharpens the blocks: neighbours mostly share a class.' : r.similarity > 0.9 ? ' The map is nearly one colour: every node points the same way, and the classes have melted together.' : ' The blocks start to merge as information leaks across classes.'), { phase: 'depth', d, sim: r.similarity, acc: r.accuracy, spread: r.spread });
      });
      const best = rows.slice(1).reduce((b, r) => (r.accuracy > b.accuracy ? r : b));
      const last = rows[rows.length - 1];
      F.push(F.last().items.filter((it) => it.key !== 'os:h').concat([H().head('os:h', 10, 26, 'Deeper is not better here')]),
        `The best depth here is ${q.n(best.depth)}, at ${q.pct(best.accuracy)}; at ${q.n(last.depth)} layers the accuracy is ${q.pct(last.accuracy)}. Common remedies keep each node’s own signal alive: few layers, skip (residual) connections that add a node’s earlier features back, and normalisations that push nodes apart again.`, { phase: 'end', best: best.depth, bestAcc: best.accuracy, last: last.depth, lastAcc: last.accuracy });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 432, h: 440 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Repeated propagation', text: ['H⁽ᵏ⁾ = Âᵏ X', 'as k grows, every column of Âᵏ X tends to', '  the same direction (Â’s top eigenvector),', 'so cos(hᵢ, hⱼ) → 1 for nodes in one component', 'remedies: fewer layers, residual links, PairNorm'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'depth' && f.meta.d === 2, pick: 'option', once: true,
      ask: 'At 16 propagations, will the GCN be more or less accurate than at 2?',
      options: ['less accurate', 'more accurate', 'the same'],
      answer: (fr) => { const a = fr.find((f) => f.meta.d === 2).meta.acc, b = fr.find((f) => f.meta.d === 16).meta.acc; return b < a - 1e-9 ? 'less accurate' : b > a + 1e-9 ? 'more accurate' : 'the same'; },
      explain: (fr) => fr.find((f) => f.meta.d === 16).say,
    }],
    quiz: [
      { q: 'What does over-smoothing mean?', options: ['after many layers, node features become nearly identical', 'the loss curve is smooth', 'the weights become small', 'the graph loses links'], answer: 0, why: 'Repeated averaging washes out the differences between nodes.' },
      { q: 'Why can 2 layers beat 0 layers on a community graph?', options: ['averaging with same-class neighbours cancels feature noise', 'more weights always help', 'it uses the test labels', 'it cannot'], answer: 0, why: 'Neighbours mostly share a class, so their mean is a cleaner signal.' },
      { q: 'Which change helps a deep GCN resist over-smoothing?', options: ['residual connections that add the earlier features back', 'more propagations', 'removing self loops', 'a larger learning rate'], answer: 0, why: 'The node keeps a share of its own, unsmoothed signal.' },
      { q: 'After 16 propagations, the mean cosine similarity of the nodes is close to 1. Their features…', options: ['all point in nearly the same direction', 'are all zero', 'are random', 'are one-hot'], answer: 0, why: 'Cosine 1 means the same direction.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
