// Model selection: train/test split (optionally stratified), k-fold and stratified k-fold, cross-validated
// scores, grid and random search, learning and validation curves, and permutation importance.
// Every splitter returns index arrays so the folds can be drawn.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);

  // allocate `total` draws over classes in proportion to their counts (largest remainders first)
  function apportion(counts, total, rng) {
    const all = ML.sum(counts), exact = counts.map((c) => (c * total) / all), out = exact.map(Math.floor);
    let left = total - ML.sum(out);
    const order = ML.range(counts.length).sort((a, b) => (exact[b] - out[b]) - (exact[a] - out[a]) || (rng ? rng() - 0.5 : a - b));
    for (const k of order) { if (left <= 0) break; if (out[k] < counts[k]) { out[k]++; left--; } }
    return out;
  }

  ML.trainTestSplit = function (n, opts) {
    const p = Object.assign({ testSize: 0.25, shuffle: true, stratify: null, seed: 0 }, opts || {}), rng = ML.rng(p.seed);
    const nTest = p.testSize < 1 ? Math.ceil(p.testSize * n) : p.testSize, nTrain = n - nTest;
    if (!p.stratify) {
      const idx = p.shuffle ? rng.perm(n) : ML.range(n);
      return { train: idx.slice(nTest, nTest + nTrain), test: idx.slice(0, nTest) };
    }
    const { classes, idx: yi } = ML.encode(p.stratify), counts = classes.map((_, k) => yi.filter((v) => v === k).length);
    const nTr = apportion(counts, nTrain), nTe = apportion(counts.map((c, k) => c - nTr[k]), nTest);
    const train = [], test = [];
    classes.forEach((_, k) => {
      const members = rng.shuffle(ML.range(n).filter((i) => yi[i] === k));
      train.push(...members.slice(0, nTr[k])); test.push(...members.slice(nTr[k], nTr[k] + nTe[k]));
    });
    return { train: rng.shuffle(train), test: rng.shuffle(test), trainCounts: nTr, testCounts: nTe };
  };

  // k folds of consecutive indices (the first n % k folds are one larger)
  ML.kFold = function (n, k = 5, opts) {
    const p = Object.assign({ shuffle: false, seed: 0 }, opts || {}), idx = p.shuffle ? ML.rng(p.seed).perm(n) : ML.range(n);
    const folds = [];
    let start = 0;
    for (let i = 0; i < k; i++) {
      const size = Math.floor(n / k) + (i < n % k ? 1 : 0), test = idx.slice(start, start + size);
      const inTest = new Set(test);
      folds.push({ train: ML.range(n).filter((j) => !inTest.has(idx[j])).map((j) => idx[j]).sort((a, b) => a - b), test: test.slice().sort((a, b) => a - b) });
      start += size;
    }
    return folds;
  };

  // stratified folds: each class is dealt out over the folds so every fold keeps the class proportions
  ML.stratifiedKFold = function (y, k = 5, opts) {
    const p = Object.assign({ shuffle: false, seed: 0 }, opts || {}), n = y.length, rng = ML.rng(p.seed);
    // classes numbered by first appearance
    const first = new Map();
    y.forEach((v) => { if (!first.has(v)) first.set(v, first.size); });
    const enc = y.map((v) => first.get(v)), K = first.size;
    const sorted = enc.slice().sort((a, b) => a - b);
    const alloc = ML.range(k).map((i) => { const c = new Array(K).fill(0); for (let t = i; t < n; t += k) c[sorted[t]]++; return c; });
    const testFold = new Array(n).fill(0);
    for (let c = 0; c < K; c++) {
      let folds = [];
      for (let i = 0; i < k; i++) for (let r = 0; r < alloc[i][c]; r++) folds.push(i);
      if (p.shuffle) folds = rng.shuffle(folds);
      let t = 0;
      for (let i = 0; i < n; i++) if (enc[i] === c) testFold[i] = folds[t++];
    }
    return ML.range(k).map((i) => ({ train: ML.range(n).filter((j) => testFold[j] !== i), test: ML.range(n).filter((j) => testFold[j] === i) }));
  };

  // scorer(model, X, y): 'accuracy' / 'r2' / the model's own score / 'neg_mse' / a function
  ML.scorer = function (s) {
    if (typeof s === 'function') return s;
    if (s === 'accuracy') return (m, X, y) => ML.accuracy(y, m.predict(X));
    if (s === 'r2') return (m, X, y) => ML.r2(y, m.predict(X));
    if (s === 'neg_mse') return (m, X, y) => -ML.mse(y, m.predict(X));
    return (m, X, y) => m.score(X, y);
  };
  const folds = (est, X, y, cv) => (Array.isArray(cv) ? cv : est.kind === 'classifier' ? ML.stratifiedKFold(y, cv || 5) : ML.kFold(X.length, cv || 5));
  const quiet = (est, over) => est.clone(Object.assign({ trace: false }, over || {}));

  // fit a fresh copy on each training fold and score it on the held-out fold
  ML.crossValScore = function (est, X, y, opts) {
    const p = Object.assign({ cv: 5, scoring: null, returnTrain: false }, opts || {}), sc = ML.scorer(p.scoring), F = folds(est, X, y, p.cv);
    const test = [], train = [];
    F.forEach((fd) => {
      const m = quiet(est).fit(ML.take(X, fd.train), ML.take(y, fd.train));
      test.push(sc(m, ML.take(X, fd.test), ML.take(y, fd.test)));
      if (p.returnTrain) train.push(sc(m, ML.take(X, fd.train), ML.take(y, fd.train)));
    });
    return p.returnTrain ? { test, train, folds: F } : test;
  };
  const sd = (a) => { const m = ML.mean(a); return Math.sqrt(ML.mean(a.map((v) => (v - m) ** 2))); };

  // all combinations of a grid {name: [values]}: names in sorted order, the last one changing fastest
  ML.paramGrid = function (grid) {
    const keys = Object.keys(grid).sort();
    let out = [{}];
    for (const k of keys) out = out.flatMap((o) => grid[k].map((v) => Object.assign({}, o, { [k]: v })));
    return out;
  };
  function search(est, cands, X, y, p) {
    const steps = [], results = cands.map((params, i) => {
      const s = ML.crossValScore(quiet(est, params), X, y, { cv: p.cv, scoring: p.scoring });
      const r = { params, scores: s, mean: ML.mean(s), std: sd(s) };
      steps.push({ kind: 'candidate', say: 'Candidate ' + (i + 1) + ' ' + JSON.stringify(params) + ': mean cross-validated score ' + f(r.mean) + ' (+/- ' + f(r.std) + ').', index: i, params, mean: r.mean, std: r.std });
      return r;
    });
    let best = 0;
    results.forEach((r, i) => { if (r.mean > results[best].mean) best = i; });
    steps.push({ kind: 'best', say: 'The best candidate is ' + JSON.stringify(results[best].params) + ' with ' + f(results[best].mean) + '.', index: best });
    const bestModel = p.refit === false ? null : quiet(est, results[best].params).fit(X, y);
    return { results, bestIndex: best, bestParams: results[best].params, bestScore: results[best].mean, bestModel, trace: steps };
  }
  ML.gridSearch = (est, grid, X, y, opts) => search(est, ML.paramGrid(grid), X, y, Object.assign({ cv: 5 }, opts || {}));
  // distributions: arrays (sampled uniformly) or functions rng -> value
  ML.randomSearch = function (est, dists, X, y, opts) {
    const p = Object.assign({ cv: 5, nIter: 10, seed: 0 }, opts || {}), rng = ML.rng(p.seed), keys = Object.keys(dists).sort();
    const cands = ML.range(p.nIter).map(() => Object.fromEntries(keys.map((k) => [k, typeof dists[k] === 'function' ? dists[k](rng) : dists[k][rng.int(dists[k].length)]])));
    return search(est, cands, X, y, p);
  };
  ML.logUniform = (lo, hi) => (rng) => Math.exp(Math.log(lo) + rng() * (Math.log(hi) - Math.log(lo)));

  // scores against training-set size: each fold trains on the first m of its training indices
  ML.learningCurve = function (est, X, y, opts) {
    const p = Object.assign({ cv: 5, trainSizes: [0.1, 0.325, 0.55, 0.775, 1], scoring: null }, opts || {}), sc = ML.scorer(p.scoring), F = folds(est, X, y, p.cv);
    const nMax = F[0].train.length;
    const sizes = [...new Set(p.trainSizes.map((s) => Math.max(1, Math.min(nMax, s <= 1 ? Math.floor(s * nMax) : s))))];
    const train = sizes.map(() => []), test = sizes.map(() => []);
    F.forEach((fd) => sizes.forEach((m, si) => {
      const tr = fd.train.slice(0, m), mod = quiet(est).fit(ML.take(X, tr), ML.take(y, tr));
      train[si].push(sc(mod, ML.take(X, tr), ML.take(y, tr))); test[si].push(sc(mod, ML.take(X, fd.test), ML.take(y, fd.test)));
    }));
    return { sizes, train, test, trace: sizes.map((m, i) => ({ kind: 'size', say: 'With ' + m + ' training rows: training score ' + f(ML.mean(train[i])) + ', validation score ' + f(ML.mean(test[i])) + '.', size: m, train: ML.mean(train[i]), test: ML.mean(test[i]) })) };
  };
  ML.validationCurve = function (est, X, y, param, values, opts) {
    const p = Object.assign({ cv: 5, scoring: null }, opts || {});
    const r = values.map((v) => ML.crossValScore(quiet(est, { [param]: v }), X, y, { cv: p.cv, scoring: p.scoring, returnTrain: true }));
    return { values, train: r.map((q) => q.train), test: r.map((q) => q.test), trace: values.map((v, i) => ({ kind: 'value', say: param + ' = ' + v + ': training ' + f(ML.mean(r[i].train)) + ', validation ' + f(ML.mean(r[i].test)) + '.', value: v })) };
  };

  // how much the score drops when one feature's column is shuffled
  ML.permutationImportance = function (model, X, y, opts) {
    const p = Object.assign({ nRepeats: 5, seed: 0, scoring: null }, opts || {}), sc = ML.scorer(p.scoring), rng = ML.rng(p.seed), base = sc(model, X, y);
    const imp = X[0].map((_, j) => ML.range(p.nRepeats).map(() => { const perm = rng.perm(X.length), Xp = X.map((r, i) => { const c = r.slice(); c[j] = X[perm[i]][j]; return c; }); return base - sc(model, Xp, y); }));
    return { baseline: base, importances: imp, mean: imp.map(ML.mean), std: imp.map(sd) };
  };
})();
