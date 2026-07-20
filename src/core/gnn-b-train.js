// Graph neural networks, part 2: training.
//   S.GNN.trainNodes({data: 'communities' | 'citations' | graph, kind: 'gcn' | 'sage' | 'mlp', hidden: 16, layers: 2,
//                     epochs: 200, lr: 0.01, weightDecay: 5e-4, dropout: 0.3, perClass: 4, seed, snapEvery: 10})
//      semi-supervised node classification: only perClass labelled nodes per class are trained on.
//      -> trainer: step() is one full-graph epoch; history [{loss, trainAcc, testAcc}];
//         snapshots [{embedding (2-D, from the last hidden layer), pred, probs}]; accuracy(); explain(node)
//   S.GNN.overSmoothing({data, depths: [1, 2, 4, 8, 16], train: true, epochs}) ->
//      [{depth, similarity (mean cosine between node features after depth propagations), spread, accuracy}]
//   S.GNN.trainGraphs({readout: 'mean' | 'sum' | 'max', kind, hidden, epochs: 60, seed})
//      graph classification with a readout -> trainer; history rows {loss, trainAcc, testAcc}
(function (S) {
  'use strict';
  const N = S.GNN;
  const fmt = (x) => S.T.fmt(x);

  const graphOf = (o) => {
    if (o.data && typeof o.data === 'object') return o.data;
    const D = S.Data, id = o.data || 'communities';
    const g = id === 'citations' ? D.citations({ n: o.n, seed: o.dataSeed || 1 }) : D.communities({ n: o.n, k: o.k, seed: o.dataSeed || 1 });
    return g;
  };
  /** perClass labelled nodes per class (chosen with the seed); the rest are for testing */
  const masks = (y, perClass, seed) => {
    const r = S.Data.rng(seed), train = [];
    const classes = [...new Set(y)].sort((a, b) => a - b);
    for (const c of classes) { const idx = r.shuffle(y.map((v, i) => (v === c ? i : -1)).filter((i) => i >= 0)); train.push(...idx.slice(0, perClass)); }
    train.sort((a, b) => a - b);
    const inTrain = new Set(train);
    return { train, test: y.map((_, i) => i).filter((i) => !inTrain.has(i)) };
  };
  const accOf = (logits, idx, y) => { if (!idx.length) return 0; const v = logits.values(), C = logits.shape[1]; let c = 0; for (const i of idx) { let b = 0; for (let k = 1; k < C; k++) if (v[i * C + k] > v[i * C + b]) b = k; if (b === y[i]) c++; } return c / idx.length; };

  N.trainNodes = (o0 = {}) => {
    const T = S.T, o = Object.assign({ kind: 'gcn', hidden: 16, layers: 2, epochs: 200, lr: 0.01, weightDecay: 5e-4, dropout: 0.3, perClass: 4, snapEvery: 10 }, o0);
    const g = graphOf(o), X = T.tensor(g.X), y = g.y, C = Math.max(...y) + 1, adj = N.adjOf(g), m = o.masks || masks(y, o.perClass, (o.seed || 1) + 100);
    const sizes = [g.X[0].length]; for (let i = 0; i + 1 < o.layers; i++) sizes.push(o.hidden); sizes.push(C);
    const net = N.net({ kind: o.kind, sizes, seed: o.seed || 1, dropout: o.dropout });
    const opt = new T.optim.Adam(net.parameters(), { lr: o.lr, weightDecay: o.weightDecay });
    const yTrain = m.train.map((i) => y[i]);
    const evalAll = () => T.noGrad(() => { net.training = false; const z = net.forward(X, adj); net.training = true; return z; });
    const snapshot = () => {
      const z = evalAll(), hid = net.hidden.length > 1 ? net.hidden[net.hidden.length - 2] : z, d = hid.shape[1], hv = hid.values(), rows = [];
      for (let i = 0; i < g.n; i++) rows.push(Array.from(hv.subarray(i * d, (i + 1) * d)));
      const p = T.softmax(z, -1).values(), probs = [], pred = [];
      for (let i = 0; i < g.n; i++) { const row = Array.from(p.subarray(i * C, (i + 1) * C)); probs.push(row); pred.push(row.indexOf(Math.max(...row))); }
      return { embedding: N.pca2(rows), pred, probs, testAcc: accOf(z, m.test, y), trainAcc: accOf(z, m.train, y) };
    };
    const tr = S.Gen.trainer({
      snapEvery: o.snapEvery, snapshot,
      step() {
        opt.zeroGrad();
        const z = net.forward(X, adj), loss = T.crossEntropy(T.take(z, m.train), yTrain);
        loss.backward(); opt.step();
        const ze = evalAll();
        return { loss: loss.item(), trainAcc: accOf(ze, m.train, y), testAcc: accOf(ze, m.test, y) };
      },
    });
    return Object.assign(tr, {
      model: net, net, graph: g, masks: m, adj,
      accuracy: () => accOf(evalAll(), m.test, y),
      explain(v = m.test[0]) {
        const z = evalAll(), nb = []; adj.sym.values().forEach((w, k) => { if (Math.floor(k / g.n) === v && w > 0) nb.push(k % g.n); });
        const p = Array.from(T.softmax(z, -1).values().subarray(v * C, (v + 1) * C)), pred = p.indexOf(Math.max(...p));
        const labelled = nb.filter((u) => m.train.includes(u));
        return [
          T.step('neighbours', 'Node ' + v + ' is linked to ' + nb.filter((u) => u !== v).length + ' nodes (' + nb.filter((u) => u !== v).join(', ') + '); ' + labelled.length + ' of them have labels.', { node: v, neighbours: nb }),
          T.step('layers', 'Each of the ' + o.layers + ' layers mixes in the neighbours\' features, so after ' + o.layers + ' layers node ' + v + ' has heard from everything up to ' + o.layers + ' hops away.', {}),
          T.step('predict', 'The softmax gives [' + p.map(fmt).join(', ') + ']: predicted ' + (g.classes ? g.classes[pred] : pred) + (pred === y[v] ? ', which is right.' : ', but the true class is ' + (g.classes ? g.classes[y[v]] : y[v]) + '.'), { probs: p, pred }),
        ];
      },
    });
  };

  N.overSmoothing = (o = {}) => {
    const T = S.T, g = graphOf(o), depths = o.depths || [1, 2, 4, 8, 16], adj = N.adjOf(g), n = g.n;
    const sim = (H) => {
      const d = H.length ? H[0].length : 0, nr = H.map((r) => Math.hypot(...r) || 1);
      let s = 0, c = 0, spread = 0; const mu = new Array(d).fill(0); H.forEach((r) => r.forEach((v, j) => { mu[j] += v / n; }));
      for (let i = 0; i < n; i++) { spread += Math.hypot(...H[i].map((v, j) => v - mu[j])) / n; for (let j = i + 1; j < n; j++) { s += H[i].reduce((t, v, k) => t + v * H[j][k], 0) / (nr[i] * nr[j]); c++; } }
      return { similarity: s / c, spread };
    };
    const M = adj.sym.values();
    let H = g.X.map((r) => r.slice()), k = 0;
    const out = [];
    for (const depth of depths) {
      while (k < depth) { H = H.map((_, i) => { const row = new Array(H[0].length).fill(0); for (let j = 0; j < n; j++) { const w = M[i * n + j]; if (w) for (let q = 0; q < row.length; q++) row[q] += w * H[j][q]; } return row; }); k++; }
      const s = sim(H);
      let acc = null;
      if (o.train !== false) { const t = N.trainNodes(Object.assign({}, o, { layers: depth, data: g, snapEvery: 1e9, dropout: 0 })); t.run(o.epochs || 150); acc = t.accuracy(); }
      out.push({ depth, similarity: s.similarity, spread: s.spread, accuracy: acc, features: H.map((r) => r.slice()),
        say: 'After ' + depth + ' propagation' + (depth === 1 ? '' : 's') + ' the average cosine similarity between nodes is ' + fmt(s.similarity) + ' and their spread ' + fmt(s.spread) + (acc == null ? '' : '; a ' + depth + '-layer GCN reaches test accuracy ' + fmt(acc)) + '.' });
    }
    return out;
  };

  N.trainGraphs = (o0 = {}) => {
    const T = S.T, o = Object.assign({ readout: 'mean', kind: 'gcn', hidden: 16, epochs: 60, lr: 0.01, nTrain: 60 }, o0);
    const set = o.set || N.graphSet({ n: o.n || 90, seed: o.dataSeed || 1 }), gs = set.graphs.map((gr) => Object.assign({ adj: N.adjOf(gr), Xt: T.tensor(gr.X) }, gr));
    const train = gs.slice(0, o.nTrain), test = gs.slice(o.nTrain);
    const g = T.rng(o.seed || 1), body = N.net({ kind: o.kind, sizes: [2, o.hidden, o.hidden], seed: o.seed || 1 }), head = new T.nn.Dense(o.hidden, 3, { rng: g });
    const params = [...body.parameters(), ...head.parameters()], opt = new T.optim.Adam(params, { lr: o.lr });
    const logits = (gr) => head.forward(N.readout(T.relu(body.forward(gr.Xt, gr.adj)), o.readout));
    const acc = (list) => T.noGrad(() => list.filter((gr) => { const v = logits(gr).values(); return v.indexOf(Math.max(...v)) === gr.y; }).length / Math.max(1, list.length));
    const order = T.rng((o.seed || 1) + 9);
    const tr = S.Gen.trainer({
      snapEvery: o.snapEvery || 10,
      step() {
        let tot = 0;
        const perm = order.permutation(train.length);
        for (let b = 0; b < perm.length; b += 10) {
          opt.zeroGrad();
          const batch = perm.slice(b, b + 10).map((i) => train[i]);
          const loss = T.scale(batch.map((gr) => T.crossEntropy(logits(gr), [gr.y])).reduce((a, c) => T.add(a, c)), 1 / batch.length);
          loss.backward(); opt.step(); tot += loss.item() * batch.length;
        }
        return { loss: tot / train.length, trainAcc: acc(train), testAcc: acc(test) };
      },
      snapshot: () => ({ embeddings: T.noGrad(() => gs.map((gr) => Array.from(N.readout(T.relu(body.forward(gr.Xt, gr.adj)), o.readout).values()))) }),
    });
    return Object.assign(tr, {
      model: body, set, train, test, accuracy: () => acc(test),
      explain(i = 0) {
        const gr = test[i] || gs[0], H = T.noGrad(() => T.relu(body.forward(gr.Xt, gr.adj))), r = T.noGrad(() => N.readout(H, o.readout)), v = T.noGrad(() => T.softmax(head.forward(r), -1).values());
        return [
          T.step('nodes', 'A ' + gr.family + ' graph with ' + gr.n + ' nodes: two GCN layers give each node a vector of ' + o.hidden + ' numbers.', { n: gr.n }),
          T.step('readout', 'The ' + o.readout + ' readout pools the ' + gr.n + ' node vectors into one graph vector, whatever the number of nodes.', { vector: Array.from(r.values()) }),
          T.step('classify', 'A dense layer and a softmax give ' + N.graphFamilies.map((f, k) => f + ' ' + fmt(v[k])).join(', ') + '.', { probs: Array.from(v) }),
        ];
      },
    });
  };

  S.lateJob('gnnNodes', S.Gen.asJob((p) => N.trainNodes(p)));
})(globalThis.Nooral = globalThis.Nooral || {});
