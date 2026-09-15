// Part 18, practice and interpretability: the families, the data sets the topics share (a 2-D set with a rare class,
// the toy sets for comparing models, real tables made ready for models), the model line-up, and small helpers.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'practice', order, title, blurb });
  fam('p18-practice', 10, 'Choosing and judging models', 'Rare classes and the metrics that notice them, choosing a model by cross-validation, and why no model wins on every data set.');
  fam('p18-explain', 20, 'Explaining a model', 'Which features a model leans on (permutation importance), how its output moves with one feature (partial dependence and ICE), and why it made one prediction (SHAP-style contributions and LIME-style local fits).');
  fam('p18-project', 30, 'A real project', 'The steps of a machine-learning project from question to monitoring, each run on one real table.');

  const P = (S.P18 = {});
  const H = () => S.Kit.ml;
  const cache = new Map();
  P.once = (k, f) => { if (!cache.has(k)) cache.set(k, f()); return cache.get(k); };
  P.r2 = (v) => Math.round(v * 100) / 100;
  P.mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  P.sd = (a) => { const m = P.mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
  P.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');

  /** columns scaled to mean 0 and spread 1 with the statistics of the rows given (train), applied to others */
  P.scaler = (X) => {
    const d = X[0].length, mu = new Array(d).fill(0), sd = new Array(d).fill(0);
    X.forEach((r) => r.forEach((v, j) => { mu[j] += v / X.length; }));
    X.forEach((r) => r.forEach((v, j) => { sd[j] += (v - mu[j]) ** 2 / X.length; }));
    const s = sd.map((v) => Math.sqrt(v) || 1);
    return { mu, sd: s, apply: (Z) => Z.map((r) => r.map((v, j) => (v - mu[j]) / s[j])) };
  };

  /** a 2-D set with a rare class: many points round the middle, few off to one side, overlapping a little */
  P.rareToy = (seed) => {
    const r = S.Data.rng(seed || 3), X = [], y = [];
    for (let i = 0; i < 285; i++) { X.push([+r.normal(0, 1).toFixed(3), +r.normal(0, 1).toFixed(3)]); y.push(0); }
    for (let i = 0; i < 15; i++) { X.push([+r.normal(1.4, 0.7).toFixed(3), +r.normal(1, 0.7).toFixed(3)]); y.push(1); }
    return { X, y, names: ['x₁', 'x₂'], classes: ['common', 'rare'] };
  };
  /** pump readings: temperature and vibration (complete rows), faults as the rare class */
  P.pumps2 = () => {
    const d = S.Data.load('pumps'), X = [], y = [];
    d.X.forEach((x, i) => { if (X.length < 400 && Number.isFinite(x[0]) && Number.isFinite(x[1])) { X.push([x[0], x[1]]); y.push(d.y[i]); } });
    return { X, y, names: ['temperature (°C)', 'vibration (mm/s)'], classes: ['normal', 'fault'], real: true };
  };

  /** the toy sets models are compared on, and one real table (loans: will the applicant default?) */
  P.TOYS = {
    linear: { label: 'a noisy line', make: (n, seed) => S.Data.linear({ n, seed, noise: 0.08 }) },
    moons: { label: 'two moons', make: (n, seed) => S.Data.moons({ n, seed, noise: 0.25 }) },
    circles: { label: 'a ring round a disc', make: (n, seed) => S.Data.circles({ n, seed, noise: 0.12 }) },
    xor: { label: 'XOR corners', make: (n, seed) => S.Data.xor({ n, seed, noise: 0.03 }) },
    checker: { label: 'a checkerboard', make: (n, seed) => S.Data.checkerboard({ n, seed, cells: 3 }) },
    loans: { label: 'loans (real table, 5 columns)', make: (n, seed) => { const d = S.Data.load('loans', { n, seed }); return { X: d.X.map((r) => r.slice(0, 5)), y: d.y, real: true }; } },
  };
  P.toy = (id, n, seed) => P.once(['toy', id, n, seed].join(':'), () => P.TOYS[id].make(n, seed));

  /** the model line-up, simplest first (the order the one-standard-error rule uses) */
  P.MODELS = [
    { id: 'nb', name: 'naive Bayes', say: 'naive Bayes', make: () => new S.ML.GaussianNB({}) },
    { id: 'logistic', name: 'logistic', say: 'logistic regression', make: () => new S.ML.LogisticRegression({}) },
    { id: 'tree', name: 'tree (depth 5)', say: 'the tree', make: () => new S.ML.DecisionTreeClassifier({ maxDepth: 5 }) },
    { id: 'knn', name: 'k-NN (k = 7)', say: 'k-NN', make: () => new S.ML.KNeighborsClassifier({ k: 7 }) },
    { id: 'svm', name: 'RBF SVM', say: 'the RBF SVM', make: () => new S.ML.SVC({ C: 1 }) },
    { id: 'forest', name: 'random forest', say: 'the random forest', make: () => new S.ML.RandomForestClassifier({ nEstimators: 30, maxDepth: 6, seed: 1 }) },
  ];
  /** cross-validated accuracy of one model: the fold scores (features scaled inside each fold) */
  P.cvScores = (m, X, y, k) => S.ML.stratifiedKFold(y, k).map((fd) => {
    const sc = P.scaler(fd.train.map((i) => X[i])), mod = m.make().fit(sc.apply(fd.train.map((i) => X[i])), fd.train.map((i) => y[i]));
    const p = mod.predict(sc.apply(fd.test.map((i) => X[i])));
    return S.Metrics.accuracy(fd.test.map((i) => y[i]), p).value;
  });

  /** a 2-D picture of a set: equal-scale axes round the points */
  P.axes = function (prefix, X, box, o) {
    o = o || {};
    const lo = [0, 1].map((j) => Math.min(...X.map((x) => x[j]))), hi = [0, 1].map((j) => Math.max(...X.map((x) => x[j])));
    const pad = o.pad === undefined ? 0.06 : o.pad;
    let xr = [lo[0] - pad * (hi[0] - lo[0]), hi[0] + pad * (hi[0] - lo[0])], yr = [lo[1] - pad * (hi[1] - lo[1]), hi[1] + pad * (hi[1] - lo[1])];
    if (o.equal !== false) {
      const s = Math.max((xr[1] - xr[0]) / box.w, (yr[1] - yr[0]) / box.h), g = (rg, len) => { const m = (rg[0] + rg[1]) / 2; return [m - (s * len) / 2, m + (s * len) / 2]; };
      xr = g(xr, box.w); yr = g(yr, box.h);
    }
    return S.Kit.curve.axes(Object.assign({ prefix, xr, yr, xname: o.xname || 'x₁', yname: o.yname || 'x₂', tickSize: 12, xticks: o.xticks, yticks: o.yticks }, box));
  };
  /** a swatch and a name: one legend row */
  P.key = (key, x, y, cls, text) => [{ key: key + ':s', kind: 'dot', x: x + 6, y: y - 4.5, r: 6, cls: 'swatch ' + cls }, H().text(key + ':t', x + 20, y, text, 'legend', 13)];
  /** a confusion matrix of a binary problem, rows true and columns predicted */
  P.confusion = (prefix, x, y, cm, names, title) => {
    const g = S.Kit.heat.grid({ prefix, x, y, cw: 66, ch: 40, vals: cm, scheme: 'seq', max: Math.max(1, ...cm.flat()), fmt: (v) => String(v), rows: names.map((n) => 'true ' + n), cols: names, size: 14, labelSize: 13 });
    const out = g.items.concat([H().text(prefix + ':pr', x + g.w / 2, y - 26, 'predicted', 'axname', 13, 'middle')]);
    if (title) out.push(H().text(prefix + ':ti', x + g.w / 2, y - 48, title, 'strong', 13, 'middle'));
    return out;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
