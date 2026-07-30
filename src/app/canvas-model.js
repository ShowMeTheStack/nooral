// The workflow canvas's document and its runner: block kinds with typed ports and parameters, wires from an output
// port to an input port of the same type, and a run that computes every block in order of its inputs. Until the
// full engines land, the blocks use the small engines of S.Toy.
(function (S) {
  'use strict';
  const W = (S.Flow = S.Flow || {});
  /** port types: a table of rows, and a trained model */
  W.TYPES = { table: { name: 'Data', cls: 't-table' }, model: { name: 'Model', cls: 't-model' } };
  /**
   * Block kinds: {name, cat, icon, ins: [{id, type, name}], outs: [...], params: [{id, name, kind: 'num' | 'choice',
   * min, max, step, choices, hint}], defaults, run(params, inputs) -> outputs, sum(out) -> one line}
   */
  W.KINDS = {
    data: {
      name: 'Data', cat: 'Data', icon: 'table', ins: [], outs: [{ id: 'data', type: 'table', name: 'Data' }],
      params: [
        { id: 'set', name: 'Dataset', kind: 'choice', choices: [['moons', 'Two moons'], ['circles', 'Two circles'], ['blobs', 'Two blobs']] },
        { id: 'n', name: 'Points', kind: 'num', min: 40, max: 400, step: 10 },
        { id: 'noise', name: 'Noise', kind: 'num', min: 0, max: 0.6, step: 0.02 },
        { id: 'seed', name: 'Seed', kind: 'num', min: 1, max: 999, step: 1, hint: 'the same seed gives the same points' },
      ],
      defaults: { set: 'moons', n: 200, noise: 0.22, seed: 7 },
      run(p) { return { data: W.dataset(p) }; },
      sum: (o) => o.data.X.length + ' rows · 2 features · 2 classes',
    },
    split: {
      name: 'Split', cat: 'Transform', icon: 'split', ins: [{ id: 'data', type: 'table', name: 'Data' }], outs: [{ id: 'train', type: 'table', name: 'Train' }, { id: 'test', type: 'table', name: 'Test' }],
      params: [
        { id: 'test', name: 'Test share', kind: 'num', min: 0.1, max: 0.6, step: 0.05, hint: 'the share of rows held back for testing' },
        { id: 'seed', name: 'Seed', kind: 'num', min: 1, max: 999, step: 1 },
      ],
      defaults: { test: 0.3, seed: 3 },
      run(p, i) { const s = S.Toy.split(i.data, p.test, p.seed); return { train: Object.assign(s.train, { set: i.data.set }), test: Object.assign(s.test, { set: i.data.set }), all: i.data }; },
      sum: (o) => o.train.X.length + ' to train · ' + o.test.X.length + ' to test',
    },
    logreg: {
      name: 'Logistic regression', cat: 'Model', icon: 'model', ins: [{ id: 'train', type: 'table', name: 'Train' }], outs: [{ id: 'model', type: 'model', name: 'Model' }],
      params: [
        { id: 'degree', name: 'Features', kind: 'choice', choices: [[1, 'x, y (a straight boundary)'], [2, 'up to squares'], [3, 'up to cubes']] },
        { id: 'lr', name: 'Learning rate', kind: 'num', min: 0.01, max: 3, step: 0.01 },
        { id: 'epochs', name: 'Epochs', kind: 'num', min: 1, max: 1000, step: 1 },
        { id: 'lam', name: 'L2 penalty λ', kind: 'num', min: 0, max: 1, step: 0.001 },
      ],
      defaults: { degree: 3, lr: 1, epochs: 300, lam: 0.001 },
      run(p, i) {
        const ep = Math.round(p.epochs), fit = S.Toy.logreg(i.train.X, i.train.y, { degree: +p.degree, lr: p.lr, epochs: ep, lam: p.lam, keep: [ep] });
        const sn = fit.snaps[0], curve = S.Toy.logregCurve(i.train.X, i.train.y, { degree: +p.degree, lr: p.lr, epochs: ep, lam: p.lam });
        return { model: { kind: 'logreg', prob: (q) => fit.predict(sn.w, sn.b, q), w: sn.w, b: sn.b, names: fit.names, loss: sn.loss, acc: sn.acc, curve, train: i.train } };
      },
      sum: (o) => 'train loss ' + o.model.loss.toFixed(3) + ' · ' + Math.round(o.model.acc * 100) + '% right',
    },
    confusion: {
      name: 'Confusion matrix', cat: 'Evaluate', icon: 'grid', ins: [{ id: 'model', type: 'model', name: 'Model' }, { id: 'data', type: 'table', name: 'Data' }], outs: [],
      params: [{ id: 'threshold', name: 'Threshold', kind: 'num', min: 0.05, max: 0.95, step: 0.05, hint: 'predict orange when p(orange) is at least this' }],
      defaults: { threshold: 0.5 },
      run(p, i) {
        const m = [[[], []], [[], []]], rows = [];
        i.data.X.forEach((x, k) => { const pr = i.model.prob(x), pred = pr >= p.threshold ? 1 : 0, t = i.data.y[k]; m[t][pred].push(k); rows.push({ k, x, t, pred, p: pr }); });
        const tp = m[1][1].length, tn = m[0][0].length, fp = m[0][1].length, fn = m[1][0].length, n = rows.length;
        const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0;
        return { matrix: m, rows, data: i.data, model: i.model, threshold: p.threshold, acc: (tp + tn) / n, prec, rec, f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0 };
      },
      sum: (o) => 'accuracy ' + Math.round(o.acc * 100) + '% · F1 ' + o.f1.toFixed(2),
    },
  };
  /** our 2-D datasets, by name */
  W.dataset = function (p) {
    const T = S.Toy, n = Math.round(p.n);
    let d;
    if (p.set === 'circles') {
      const r = T.rng(p.seed), X = [], y = [];
      for (let i = 0; i < n; i++) { const c = i % 2, a = r() * 2 * Math.PI, rad = c ? 0.5 : 1; X.push([rad * Math.cos(a) + p.noise * 0.5 * r.normal(), rad * Math.sin(a) + p.noise * 0.5 * r.normal()]); y.push(c); }
      d = { X, y };
    } else if (p.set === 'blobs') d = T.blobs(n, [[-0.2, 1], [1.3, -0.2]], 0.25 + p.noise, p.seed);
    else d = T.moons(n, p.noise, p.seed);
    return Object.assign(d, { set: p.set });
  };

  /** a new document: the pipeline Data → Split → Logistic regression → Confusion matrix */
  W.starter = () => ({
    blocks: [
      { id: 'b1', kind: 'data', x: 90, y: 150, params: Object.assign({}, W.KINDS.data.defaults) },
      { id: 'b2', kind: 'split', x: 290, y: 150, params: Object.assign({}, W.KINDS.split.defaults) },
      { id: 'b3', kind: 'logreg', x: 500, y: 80, params: Object.assign({}, W.KINDS.logreg.defaults) },
      { id: 'b4', kind: 'confusion', x: 710, y: 150, params: Object.assign({}, W.KINDS.confusion.defaults) },
    ],
    wires: [
      { id: 'w1', from: ['b1', 'data'], to: ['b2', 'data'] },
      { id: 'w2', from: ['b2', 'train'], to: ['b3', 'train'] },
      { id: 'w3', from: ['b3', 'model'], to: ['b4', 'model'] },
      { id: 'w4', from: ['b2', 'test'], to: ['b4', 'data'] },
    ],
  });
  W.portOf = (doc, end, side) => { const b = doc.blocks.find((x) => x.id === end[0]); if (!b) return null; const K = W.KINDS[b.kind]; return (side === 'out' ? K.outs : K.ins).find((p) => p.id === end[1]) || null; };
  /** can a wire go from this output to that input? */
  W.canWire = (doc, from, to) => { const a = W.portOf(doc, from, 'out'), b = W.portOf(doc, to, 'in'); return !!(a && b && a.type === b.type && from[0] !== to[0]); };
  /**
   * Run every block whose inputs are all wired and computed, in order: {out: {blockId: outputs}, err: {blockId:
   * message}, missing: {blockId: [input names]}}
   */
  W.run = function (doc) {
    const out = {}, err = {}, missing = {}, done = new Set();
    let progress = true;
    while (progress) {
      progress = false;
      for (const b of doc.blocks) {
        if (done.has(b.id)) continue;
        const K = W.KINDS[b.kind], ins = {}, need = [];
        let wait = false;
        for (const p of K.ins) {
          const w = doc.wires.find((x) => x.to[0] === b.id && x.to[1] === p.id);
          if (!w) { need.push(p.name); continue; }
          if (err[w.from[0]] || missing[w.from[0]]) { need.push(p.name); continue; }
          if (!done.has(w.from[0])) { wait = true; break; }
          ins[p.id] = out[w.from[0]][w.from[1]];
        }
        if (wait) continue;
        done.add(b.id); progress = true;
        if (need.length) { missing[b.id] = need; continue; }
        try { out[b.id] = K.run(b.params, ins); } catch (e) { err[b.id] = String(e.message || e); }
      }
    }
    return { out, err, missing };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
