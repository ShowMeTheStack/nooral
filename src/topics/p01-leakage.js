// Part 1, leakage shown happening: the same data through two pipelines drawn one above the other. The leaky one lets
// the test folds influence training (columns chosen on all rows, a column that is a consequence of the label, copies
// of a row on both sides of the split, a scaler fitted on all rows); the honest one keeps every learned step inside
// the training folds. Both are cross-validated fold by fold, and the gap between their scores is the leak.
(function (S) {
  'use strict';
  const H = S.P01, DT = () => S.Kit.dt, FL = () => S.Kit.flow, ML = () => S.ML;
  const LEAKS = {
    select: { label: 'Choosing columns before splitting', leakStep: ['pick 10 columns', 'on all rows'], okStep: ['pick 10 columns', 'per training fold'], okSay: 'chooses its 10 columns afresh inside each training fold' },
    target: { label: 'A column that comes after the label', leakStep: ['keep the column', '"letters sent"'], okStep: ['drop the column', '"letters sent"'], okSay: 'leaves out "letters sent", which does not exist yet when the loan is decided' },
    dupes: { label: 'Copies of a row on both sides', leakStep: ['shuffle rows', 'into folds'], okStep: ['group the rows', 'by student'], okSay: 'keeps all 3 rows of a student in the same fold' },
    scale: { label: 'Scaling on all rows', leakStep: ['fit the scaler', 'on all rows'], okStep: ['fit the scaler', 'per training fold'], okSay: 'fits the scaler on each training fold' },
  };

  /** the data of a scenario: X, y, groups (or null), and how a fold is fitted (leaky or honest) */
  function scenario(key) {
    const ml = ML();
    if (key === 'select') {
      const r = H.rng(5), n = 100, d = 400, X = [], y = [];
      for (let i = 0; i < n; i++) { X.push(Array.from({ length: d }, () => +r.normal(0, 1).toFixed(3))); y.push(i % 2); }
      r.shuffle(y);
      const top = (rows) => { const ys = rows.map((i) => y[i]); return Array.from({ length: d }, (_, j) => [Math.abs(H.pearson(rows.map((i) => X[i][j]), ys).r), j]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, 10); };
      const allRows = X.map((_, i) => i), leakTop = top(allRows);
      const fresh = (() => { const q = H.rng(6), Xf = [], yf = []; for (let i = 0; i < 100; i++) { Xf.push(Array.from({ length: d }, () => +q.normal(0, 1).toFixed(3))); yf.push(q() < 0.5 ? 1 : 0); } return { Xf, yf }; })();
      const knn = (cols, tr, te, A, ya) => { const m = ml.KNeighborsClassifier({ k: 5 }).fit(tr.map((i) => cols.map((j) => X[i][j])), tr.map((i) => y[i])); return ml.accuracy(ya, m.predict(A.map((row) => cols.map((j) => row[j])))); };
      const leakCols = leakTop.map(([, j]) => j);
      return {
        X, y, n, groups: null, what: 'accuracy', chance: 0.5, model: '5-NN',
        intro: `${n} rows of ${d} columns of pure noise, and labels shuffled at random (${y.filter((v) => v === 1).length} of each class, unrelated to every column). No model can beat 50% on new rows.`,
        leakNote: `The 10 columns most correlated with the label over all ${n} rows reach |r| = ${H.f(leakTop[9][0], 3)} to ${H.f(leakTop[0][0], 3)}, purely by chance.`,
        leakFold: (tr, te) => knn(leakCols, tr, te, te.map((i) => X[i]), te.map((i) => y[i])),
        okFold: (tr, te) => { const c = top(tr).map(([, j]) => j); return knn(c, tr, te, te.map((i) => X[i]), te.map((i) => y[i])); },
        freshScore: () => knn(leakCols, allRows, null, fresh.Xf, fresh.yf),
        meta: { leakTop: leakTop.map((p) => p[0]) },
      };
    }
    if (key === 'target') {
      const ds = H.data('loans', { n: 400 }), r = H.rng(8);
      const letters = ds.y.map((v) => (v === 1 ? (r() < 0.93 ? 1 + Math.floor(r() * 4) : 0) : r() < 0.04 ? 1 : 0));
      const X = ds.X.map((q, i) => q.slice(0, 5).concat([letters[i]])), y = ds.y, n = X.length;
      const fit = (cols, tr, te) => { const m = ml.LogisticRegression({ C: 1 }).fit(tr.map((i) => cols.map((j) => X[i][j])), tr.map((i) => y[i])); return ml.accuracy(te.map((i) => y[i]), m.predict(te.map((i) => cols.map((j) => X[i][j])))); };
      const withL = [0, 1, 2, 3, 4, 5], without = [0, 1, 2, 3, 4];
      const share = y.filter((v) => v === 0).length / n;
      return {
        X, y, n, groups: null, what: 'accuracy', chance: share, model: ['logistic', 'regression'],
        intro: `${n} loan applications: income, debt ratio, years employed, missed payments, amount, and a column "letters sent": reminder letters the lender sent later, almost only to people who defaulted. ${H.pct(share)} repaid, so always saying "repaid" scores ${H.pct(share)}.`,
        leakNote: `"Letters sent" is 0 for ${H.pct(letters.filter((v, i) => v === 0 && y[i] === 0).length / y.filter((v) => v === 0).length)} of the people who repaid and at least 1 for ${H.pct(letters.filter((v, i) => v > 0 && y[i] === 1).length / y.filter((v) => v === 1).length)} of those who defaulted: it is written after the outcome.`,
        leakFold: (tr, te) => fit(withL, tr, te), okFold: (tr, te) => fit(without, tr, te), meta: { zeroRepaid: letters.filter((v, i) => v === 0 && y[i] === 0).length / y.filter((v) => v === 0).length, someDefault: letters.filter((v, i) => v > 0 && y[i] === 1).length / y.filter((v) => v === 1).length },
      };
    }
    if (key === 'dupes') {
      const ds = H.data('exam', { n: 80 }), r = H.rng(9), X = [], y = [], g = [];
      ds.X.forEach((q, i) => { for (let c = 0; c < 3; c++) { X.push([q[0] + +(r.normal(0, 0.05)).toFixed(2), q[1] + +(r.normal(0, 0.05)).toFixed(2), q[2]]); y.push(ds.y[i]); g.push(i); } });
      const fit = (tr, te) => { const m = ml.KNeighborsClassifier({ k: 1 }).fit(tr.map((i) => X[i]), tr.map((i) => y[i])); return ml.accuracy(te.map((i) => y[i]), m.predict(te.map((i) => X[i]))); };
      return {
        X, y, n: X.length, groups: g, what: 'accuracy', chance: Math.max(H.mean(y), 1 - H.mean(y)), model: '1-NN',
        intro: `${ds.X.length} students, but each was recorded 3 times with tiny differences: ${X.length} rows. A 1-nearest-neighbour model predicts the class of the most similar training row.`,
        leakNote: 'Shuffled into folds, nearly every test row has a near copy of itself in the training folds, and 1-NN simply finds it.',
        leakFold: fit, okFold: fit, meta: { students: ds.X.length },
      };
    }
    const ds = H.data('wine'), X = ds.X.map((q) => [q[12], q[10], q[6]]), y = ds.y, n = X.length;
    const fit = (scFrom, tr, te) => { const sc = H.scaler(scFrom.map((i) => X[i]), 'standard'), Z = sc.apply(X); const m = ml.KNeighborsClassifier({ k: 5 }).fit(tr.map((i) => Z[i]), tr.map((i) => y[i])); return ml.accuracy(te.map((i) => y[i]), m.predict(te.map((i) => Z[i]))); };
    return {
      X, y, n, groups: null, what: 'accuracy', chance: Math.max(...[0, 1, 2].map((c) => y.filter((v) => v === c).length)) / n, model: '5-NN',
      intro: `${n} wines and three columns (proline, hue, flavanoids), scaled before 5-NN. The scaler's means and sds are learned numbers too.`,
      leakNote: 'Fitted on all rows, the scaler has seen the test fold\'s values: its mean and sd include them.',
      leakFold: (tr, te) => fit(X.map((_, i) => i), tr, te), okFold: (tr, te) => fit(tr, tr, te), meta: {},
    };
  }
  /** five folds: stratified, or (with groups) every group's rows in one fold */
  function foldsOf(sc, honest) {
    const ml = ML();
    if (sc.groups && honest) {
      const ids = [...new Set(sc.groups)], r = ml.rng(3), order = r.shuffle(ids.slice()), foldOf = new Map(order.map((g, k) => [g, k % 5]));
      return [0, 1, 2, 3, 4].map((f) => ({ train: sc.groups.map((g, i) => (foldOf.get(g) !== f ? i : -1)).filter((i) => i >= 0), test: sc.groups.map((g, i) => (foldOf.get(g) === f ? i : -1)).filter((i) => i >= 0) }));
    }
    return ml.stratifiedKFold(sc.y, 5, { shuffle: true, seed: 1 });
  }

  S.topic({
    id: 'data-leakage', family: 'p01-leakage', order: 10, title: 'Leakage, shown happening',
    blurb: 'A test score means something only if nothing about the test rows reached training. Leakage breaks that, often quietly: columns chosen by looking at all the labels, a column recorded after the outcome, copies of one example on both sides of the split, or preparation fitted on all rows. Here the same data runs through a leaky pipeline and an honest one, side by side; every step that learns anything sits inside the training folds in the honest one.',
    inputs: [{ id: 'leak', kind: 'choice', label: 'Leak', def: 'select', options: Object.entries(LEAKS).map(([value, l]) => ({ value, label: l.label })) }],
    presets: Object.entries(LEAKS).map(([value, l]) => ({ label: l.label, values: { leak: value } })),
    counters: [{ id: 'folds', label: 'folds scored' }],
    run(inp) {
      const L = LEAKS[inp.leak], sc = scenario(inp.leak);
      const fLeak = foldsOf(sc, false), fOk = foldsOf(sc, true);
      const leak = fLeak.map((f) => sc.leakFold(f.train, f.test)), ok = fOk.map((f) => sc.okFold(f.train, f.test));
      const mLeak = H.mean(leak), mOk = H.mean(ok);
      const F = new S.Frames(['folds']), view = H.box(-14, -8, 790, 470);
      const pipe = (prefix, y, steps, cls, lit) => {
        const N = FL().net(prefix), xs = [62, 196, 330, 464], w = 124, h = 50;
        steps.forEach((t, k) => N.box('b' + k, xs[k], y, w, h, t, 'pipe ' + (lit && lit.includes(k) ? cls : '')));
        for (let k = 0; k < steps.length - 1; k++) N.wire('w' + k, [N.port('b' + k, 'e'), N.port('b' + (k + 1), 'w')], { head: true, cls: lit && lit.includes(k + 1) ? (cls === 'leak' ? 'leak' : 'ok') : '' });
        return N.items;
      };
      const leakOrder = inp.leak === 'dupes' ? [['all ' + sc.n, 'rows'], L.leakStep, ['5 folds'], [].concat(sc.model)] : [['all ' + sc.n, 'rows'], L.leakStep, ['5-fold split'], [].concat(sc.model)];
      const okOrder = inp.leak === 'dupes' ? [['all ' + sc.n, 'rows'], L.okStep, ['5 folds'], [].concat(sc.model)] : [['all ' + sc.n, 'rows'], ['5-fold split'], L.okStep, [].concat(sc.model)];
      const zone = (key, y, cls, text) => ({ key, kind: 'region', x: -6, y: y - 52, w: 776, h: 104, rx: 8, cls, audit: 'group' });
      const label = (key, y, text, cls) => H.text(key, 0, y - 60, text, cls, 14);
      const scores = (prefix, y, vals, cls) => [...DT().vbars({ prefix, x: 548, y: y - 44, w: 150, h: 86, vals, max: 1, fmt: (v) => H.f(v * 100, 0), cls: () => cls, size: 12 }), H.text(prefix + 'm', 708, y + 6, H.pct(H.mean(vals)), cls === 'bad' ? 'bad' : 'good', 16)];
      const chance = (prefix, y) => [{ key: prefix + 'ch', kind: 'path', d: `M548,${(y + 42 - (86 - 20) * sc.chance).toFixed(2)} h150`, cls: 'guide', audit: 'ignore', layer: 'top' }];
      const yL = 130, yO = 340;
      const both = (litL, litO, sL, sO) => [H.title('tt', 0, 6, L.label), zone('zl', yL, 'leakzone'), label('ll', yL, 'leaky pipeline', 'bad'), ...pipe('pl', yL, leakOrder, 'leak', litL), zone('zo', yO, 'okzone'), label('lo', yO, 'honest pipeline', 'good'), ...pipe('po', yO, okOrder, 'ok', litO)].concat(sL ? [...scores('sl', yL, leak, 'bad'), ...chance('sl', yL)] : [], sO ? [...scores('so', yO, ok, 'good'), ...chance('so', yO)] : []);
      F.add({ items: both([], [], false, false), say: sc.intro, meta: { phase: 'setup', n: sc.n, chance: sc.chance }, view });
      F.add({ items: both([1], [], false, false), say: sc.leakNote, meta: Object.assign({ phase: 'leakstep' }, sc.meta), view });
      F.set('folds', 5);
      F.add({ items: both([1, 2, 3], [], true, false), say: `The leaky pipeline's five folds score ${leak.map((v) => H.pct(v, 0)).join(', ')}: a mean ${sc.what} of ${H.pct(mLeak)}, against ${H.pct(sc.chance)} for always guessing the commonest class.`, meta: { phase: 'leak', folds: leak, mean: mLeak }, view });
      F.set('folds', 10);
      F.add({ items: both([1, 2, 3], [1, 2, 3], true, true), say: `The honest pipeline ${L.okSay}. Its folds score ${ok.map((v) => H.pct(v, 0)).join(', ')}: a mean of ${H.pct(mOk)}.`, meta: { phase: 'honest', folds: ok, mean: mOk }, view });
      const fresh = sc.freshScore ? sc.freshScore() : null;
      const gap = mLeak - mOk;
      F.add({ items: both([1, 2, 3], [1, 2, 3], true, true), say: `The leak is worth ${H.f(gap * 100, 1)} points of ${sc.what} (${H.pct(mLeak)} against ${H.pct(mOk)}).` + (fresh !== null ? ` On 100 brand-new noise rows, the leaky pipeline's model scores ${H.pct(fresh)}: its cross-validated score was an illusion.` : inp.leak === 'scale' ? ' Small here, but the rule is the same: the honest score is the one to report.' : ' The honest number is what to expect on new data.'), meta: { phase: 'gap', gap, fresh }, view });
      const ctx = Object.assign({ chance: sc.chance, mLeak, mOk, n: sc.n }, sc.meta);
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'The rule', text: ['everything learned is learned on training rows:', '  column choice, scaling, filling, encoding,', '  bin edges, the model', 'no column that is known only after the outcome', 'related rows (one person, one day) stay together', 'a pipeline inside each fold keeps it honest'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'leakstep', pick: 'option', ask: 'What mean score will the leaky pipeline report?', options: ['far above chance', 'about chance', 'below chance'], answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'leak').meta.mean, c = fr[0].meta.chance; return m > c + 0.08 ? 'far above chance' : m < c - 0.08 ? 'below chance' : 'about chance'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'leak').say },
      { at: (f) => f.meta.phase === 'leak', pick: 'option', ask: 'Will the honest pipeline score higher or lower than the leaky one?', options: ['lower', 'higher', 'the same'], answer: (fr) => { const g = fr.find((f) => f.meta.phase === 'gap').meta.gap; return g > 1e-9 ? 'lower' : g < -1e-9 ? 'higher' : 'the same'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'gap').say },
    ],
    quiz: [
      { q: 'You pick the 20 columns most correlated with the label using all rows, then cross-validate. The score is…', options: ['optimistic: the choice already saw the test folds', 'honest', 'pessimistic', 'unaffected by the choice'], answer: 0, why: 'Selection is training; it must be redone inside each training fold.' },
      { q: 'Which column leaks in a model that predicts hospital readmission at discharge?', options: ['"readmission paperwork filed"', 'age', 'length of stay', 'number of medicines'], answer: 0, why: 'It exists only because the outcome happened.' },
      { q: 'Each patient has 5 scans. How should they be split?', options: ['all of a patient\'s scans on the same side', 'scans at random', 'the first scan to test', 'it does not matter'], answer: 0, why: 'Otherwise the model can recognise the patient instead of the disease.' },
      { q: 'Fitting a scaler on all rows before splitting usually…', options: ['leaks a little: small effect, same mistake', 'leaks hugely', 'is required', 'removes leakage'], answer: 0, why: 'The test rows nudge the mean and sd; selection on labels leaks far more.' },
      { q: 'The best protection against leakage is…', options: ['a pipeline whose every learned step is fitted inside each training fold', 'a bigger test set', 'more features', 'a higher learning rate'], answer: 0, why: 'Then nothing from the held-out fold can reach training.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
