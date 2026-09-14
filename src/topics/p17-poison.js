// Part 17, data poisoning as a concept: a share of one region's training labels is flipped, the boundary a
// logistic regression learns tilts towards them and test accuracy falls; then the defence: drop every training
// point whose label disagrees with most of its nearest neighbours, and retrain.
(function (S) {
  'use strict';
  const P = S.P17, H = () => S.Kit.ml, C = () => S.Kit.curve;
  /** the noisy-line data split into training and test rows */
  P.lineData = (seed) => P.once('ln:' + seed, () => {
    const ds = S.Data.load('linear', { n: 300, seed, noise: 0.02 }), sp = S.Data.split(ds.y, 0.4, seed);
    return { Xtr: sp.train.map((i) => ds.X[i]), ytr: sp.train.map((i) => ds.y[i]), Xte: sp.test.map((i) => ds.X[i]), yte: sp.test.map((i) => ds.y[i]) };
  });
  /** flip the labels of a share of the training rows: class 0 rows in the upper half (x₂ > 0), in a seeded order */
  P.poison = (D, share, seed) => {
    const pool = S.Data.rng(seed + 4).shuffle(D.Xtr.map((x, i) => i).filter((i) => D.ytr[i] === 0 && D.Xtr[i][1] > 0));
    const k = Math.min(pool.length, Math.round(share * D.Xtr.length)), flip = new Set(pool.slice(0, k));
    return { flip, y: D.ytr.map((v, i) => (flip.has(i) ? 1 : v)), pool: pool.length };
  };
  const accOf = (m, X, y) => X.filter((x, i) => m.predict(x) === y[i]).length / y.length;

  S.topic({
    id: 'data-poisoning', family: 'p17-privacy', order: 30, title: 'Poisoned training labels, and cleaning them out',
    blurb: 'If some training labels are wrong on purpose, the model learns the wrong lesson. Here a share of one class’s labels in one region is flipped: the boundary tilts and test accuracy falls. A simple defence asks each training point’s nearest neighbours whether its label fits; points that disagree with most of them are set aside before retraining. It works while the bad labels are scattered among good ones, and fails when they outnumber them.',
    inputs: [
      { id: 'share', kind: 'slider', label: 'Share of training labels flipped', def: 0.1, min: 0, max: 0.3, step: 0.05, fmt: (v) => Math.round(v * 100) + '%' },
      { id: 'k', kind: 'number', label: 'Neighbours asked (k)', def: 7, min: 3, max: 15 },
      { id: 'seed', kind: 'number', label: 'Seed (the data)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: '10% flipped, k 7', values: { share: 0.1, k: 7, seed: 1 } },
      { label: '5% flipped', values: { share: 0.05, k: 7, seed: 1 } },
      { label: 'Too much poison: 25%', values: { share: 0.25, k: 7, seed: 1 } },
      { label: 'Nothing flipped', values: { share: 0, k: 5, seed: 2 } },
    ],
    counters: [{ id: 'flipped', label: 'labels flipped' }, { id: 'dropped', label: 'points set aside' }],
    run(inp) {
      const F = H().frames(['flipped', 'dropped']), q = F.q, D = P.lineData(inp.seed), n = D.Xtr.length;
      const ax = C().axes({ prefix: 'dp2:ax', x: 30, y: 50, w: 340, h: 340, xr: [-2.1, 2.1], yr: [-2.1, 2.1], xticks: 1, yticks: 1, xname: 'x₁', yname: 'x₂', tickSize: 12 });
      const m0 = S.Rob.logreg(D.Xtr, D.ytr, {}), acc0 = accOf(m0, D.Xte, D.yte);
      const line = (m, key, cls) => H().level(ax, key, (u, v) => m.prob([u, v]), 0.5, cls, 50);
      const pz = P.poison(D, inp.share, inp.seed), yp = pz.y, flip = pz.flip;
      const m1 = S.Rob.logreg(D.Xtr, yp, {}), acc1 = accOf(m1, D.Xte, D.yte);
      const knn = new S.ML.KNeighborsClassifier({ k: inp.k }).fit(D.Xtr, yp), pr = knn.predictProba(D.Xtr);
      const keep = yp.map((v, i) => pr[i][knn.classes.indexOf(v)] >= 0.5), dropped = keep.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
      const caught = dropped.filter((i) => flip.has(i)).length;
      const m2 = S.Rob.logreg(D.Xtr.filter((_, i) => keep[i]), yp.filter((_, i) => keep[i]), {}), acc2 = accOf(m2, D.Xte, D.yte);
      const panel = (rows) => H().lines('dp2:t', 420, 80, rows, 13, 24);
      const keys = (extra) => [...P.key('dp2:k0', 420, 330, 'c0', 'class 0'), ...P.key('dp2:k1', 520, 330, 'c1', 'class 1'), ...(extra || [])];
      const pts = (y, cls) => P.dots(ax, 'dp2:p', D.Xtr, y, { cls, r: (i) => (flip.has(i) ? 4.8 : 3.6) });
      F.push([...ax.items, line(m0, 'dp2:l0', 'p17edge'), ...pts(D.ytr), ...keys(), H().head('dp2:h', 30, 26, 'Clean training labels'), ...panel([['test accuracy ' + H().pct(acc0), 'mono']])],
        `${P.qpl(q, n, 'training point')} on either side of a straight rule. A logistic regression trained on the true labels gets ${q.pct(acc0)} of the ${q.n(D.yte.length)} test points right.`, { phase: 'clean', n, acc0, nTest: D.yte.length });
      F.set('flipped', flip.size);
      F.push([...ax.items, line(m0, 'dp2:l0', 'p17other'), line(m1, 'dp2:l1', 'p17edge'), ...pts(yp, (i) => (flip.has(i) ? 'p17flip' : '')), ...keys([...P.key('dp2:k2', 420, 354, 'p17flip', 'label flipped')]), H().head('dp2:h', 30, 26, 'Some labels flipped'), ...panel([['test accuracy ' + H().pct(acc1), 'mono'], ['dashed: the clean model', 'axname']])],
        flip.size ? `Flip ${P.qpl(q, flip.size, 'label')} (${q.pct(inp.share)} of the training set), all blue points in the upper half, to orange (red rings). Retrained on these labels, the boundary tilts towards them and test accuracy falls to ${q.pct(acc1)}.`
          : `Nothing is flipped this time, so the retrained model is the same and test accuracy stays at ${q.pct(acc1)}.`, { phase: 'poison', flipped: flip.size, share: inp.share, acc1, pool: pz.pool });
      F.set('dropped', dropped.length);
      const clean = dropped.length - caught;
      F.push([...ax.items, line(m1, 'dp2:l1', 'p17edge'), ...pts(yp, (i) => (flip.has(i) ? 'p17flip ' : '') + (!keep[i] ? 'p17drop' : '')), ...keys([...P.key('dp2:k2', 420, 354, 'p17flip', 'label flipped'), ...P.key('dp2:k3', 420, 378, 'p17drop', 'set aside')]), H().head('dp2:h', 30, 26, 'Ask the ' + inp.k + ' nearest neighbours'), ...panel([['set aside: ' + dropped.length, 'mono'], ['flipped among them: ' + caught, 'mono']])],
        `For every training point, ask its ${P.qpl(q, inp.k, 'nearest neighbour')} (itself included) for a vote. ${P.qpl(q, dropped.length, 'point')} ${dropped.length === 1 ? 'has' : 'have'} a label most of them disagree with and ${dropped.length === 1 ? 'is' : 'are'} set aside: ${q.n(caught)} of the ${q.n(flip.size)} flipped labels and ${q.n(clean)} honest points near the boundary.`, { phase: 'vote', dropped: dropped.length, caught, flipped: flip.size, clean, k: inp.k });
      F.push([...ax.items, line(m1, 'dp2:l1', 'p17other'), line(m2, 'dp2:l2', 'p17edge'), ...P.dots(ax, 'dp2:p', D.Xtr, yp, { cls: (i) => (flip.has(i) ? 'p17flip ' : '') + (!keep[i] ? 'p17drop' : ''), r: (i) => (flip.has(i) ? 4.8 : 3.6) }), ...keys([...P.key('dp2:k2', 420, 354, 'p17flip', 'label flipped'), ...P.key('dp2:k3', 420, 378, 'p17drop', 'set aside')]), H().head('dp2:h', 30, 26, 'Retrained on what is left'), ...panel([['clean labels:  ' + H().pct(acc0), 'mono'], ['poisoned:      ' + H().pct(acc1), 'mono'], ['after cleaning: ' + H().pct(acc2), 'mono'], ['dashed: the poisoned model', 'axname']])],
        `Retrained without the ${P.qpl(q, dropped.length, 'point')} set aside, the model gets ${q.pct(acc2)} of the test points right, against ${q.pct(acc1)} poisoned and ${q.pct(acc0)} clean.` + (flip.size && caught < flip.size / 2 ? ' Here the flipped labels crowd their region, so their neighbours vote with them and most survive: no single check is enough. Knowing where data came from, and watching models trained on it, matter as much.' : flip.size ? ' The flipped labels sat among honest neighbours, so the vote exposed them.' : ''), { phase: 'retrain', acc0, acc1, acc2, dropped: dropped.length, caught, flipped: flip.size });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 400, h: 420 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A neighbour vote', text: ['for each training point i:', '  N_k(i) = its k nearest training points', '  (i itself included)', '  keep i if at least half of N_k(i)', '  carry the label of i', 'retrain on the points kept'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'poison' && f.meta.flipped > 0, pick: 'option', once: true,
      ask: 'Will cleaning by neighbour vote win back most of the lost accuracy?',
      options: ['yes', 'no'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'retrain').meta; return m.acc0 - m.acc1 < 1e-9 ? null : (m.acc2 - m.acc1) / (m.acc0 - m.acc1) > 0.5 ? 'yes' : 'no'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'retrain').say,
    }],
    quiz: [
      { q: 'Label poisoning changes…', options: ['the labels of some training rows', 'the test set', 'the model’s code', 'the learning rate'], answer: 0, why: 'The model faithfully learns from labels that lie.' },
      { q: 'With k = 7, a point is set aside when…', options: ['fewer than half of its 7 nearest points share its label', 'it is far from the boundary', 'it is in class 1', 'its label is 0'], answer: 0, why: 'Its neighbours outvote its label.' },
      { q: 'Why does the vote fail when most labels in a region are flipped?', options: ['the neighbours then agree with the wrong labels', 'k is too small', 'logistic regression cannot be retrained', 'it never fails'], answer: 0, why: 'A majority of bad labels looks normal locally.' },
      { q: 'A cost of the neighbour vote on clean data is…', options: ['it also sets aside honest points near the boundary', 'it flips labels', 'it makes the model deeper', 'none'], answer: 0, why: 'Near a boundary neighbours of both classes mix.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
