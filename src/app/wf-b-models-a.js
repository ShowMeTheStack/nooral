// The classical model families of the workflow's Models category: linear, ridge and lasso regression, k nearest
// neighbours, naive Bayes, linear and quadratic discriminant analysis, the decision tree, random forest, AdaBoost,
// gradient boosting, the support vector machine (and SVR) and the perceptron. Each says how to build its estimator
// from S.ML and what of its insides the viewers draw (internals; the pictures' extra numbers are in
// wf-b-models-view.js). DOM-free: this runs in the workflow's worker and in node.
(function (S) {
  'use strict';
  const WF = S.WF, ML = () => S.ML;
  const r4 = (v) => (typeof v === 'number' && isFinite(v) ? +v.toPrecision(4) : v);
  const arr = (a) => Array.from(a || [], r4);
  const mat = (m) => Array.from(m || [], (r) => arr(r));
  const names = (data) => data.features.map((f) => f.name);
  const isClf = (data) => data.task === 'classification';
  const depth = (v) => (v ? +v : null);
  WF.mr = { r4, arr, mat };
  // an image set keeps its pixels in one flat array (images) with get(i) for one image: hand the rows on as rows
  if (WF.fromDataset && !WF.fromDataset.__rows) {
    const orig = WF.fromDataset;
    WF.fromDataset = function (ds, source) { if (ds && ds.images && ds.get && !Array.isArray(ds.images)) ds = Object.assign({}, ds, { images: Array.from({ length: ds.y.length }, (_, i) => ds.get(i)) }); return orig(ds, source); };
    WF.fromDataset.__rows = true;
  }

  // ---- linear models -----------------------------------------------------------------------------------------
  const coefOf = (est, data) => {
    const W = est.coef, b = est.intercept;
    const rows = W && (Array.isArray(W[0]) || ArrayBuffer.isView(W[0])) ? mat(W) : [arr(W)];
    return { coef: rows, intercept: [].concat(Array.isArray(b) || ArrayBuffer.isView(b) ? arr(b) : r4(b)), features: names(data) };
  };
  WF.model('linreg', {
    label: 'Linear regression', task: 'regression', order: 5, icon: 'curve',
    blurb: 'A weighted sum of the features plus a constant, with the weights that make the squared errors smallest: the projection of the target onto the space the columns span.',
    params: [{ id: 'solver', label: 'Solver', kind: 'choice', def: 'qr', options: [['qr', 'Exact (least squares)'], ['gd', 'Gradient descent']] }],
    make(p, data, rng, ctx) {
      // more columns than rows: least squares has many exact answers, so take the one with the smallest weights (the
      // limit of ridge as its penalty goes to zero)
      if (p.solver !== 'gd' && data.X.length && data.X[0].length >= data.X.length) { const e = ML().create('Ridge', { alpha: 1e-8 }); e.leastNorm = { rows: data.X.length, cols: data.X[0].length }; return e; }
      return ML().create('LinearRegression', { solver: p.solver, epochs: 500, lr: 0.05, seed: ctx ? ctx.seed : 0 });
    },
    internals: (est, data) => Object.assign(coefOf(est, data), est.leastNorm ? { leastNorm: est.leastNorm } : {}),
  });
  WF.model('ridge', {
    label: 'Ridge regression', task: 'regression', order: 6, icon: 'curve',
    blurb: 'Least squares with a penalty on the squared size of the weights: α pulls every weight towards zero, smoothly, and steadies them when features are alike.',
    params: [{ id: 'alpha', label: 'α (penalty)', kind: 'number', def: 1, min: 0.0001, max: 10000, log: true }],
    make: (p) => ML().create('Ridge', { alpha: p.alpha }),
    internals: coefOf,
  });
  WF.model('lasso', {
    label: 'Lasso regression', task: 'regression', order: 7, icon: 'curve',
    blurb: 'Least squares with a penalty on the absolute size of the weights: as α grows, weights reach exactly zero one by one, so the lasso also chooses features.',
    params: [{ id: 'alpha', label: 'α (penalty)', kind: 'number', def: 0.1, min: 0.0001, max: 10000, log: true }],
    make: (p) => ML().create('Lasso', { alpha: p.alpha, maxIter: 20000, tol: 1e-8 }),
    internals: coefOf,
  });
  // logistic regression: the same as the core's, with the same coefficient picture
  WF.families.logistic.internals = (est, data) => Object.assign(coefOf(est, data), { classes: data.classes, nIter: est.nIter });
  WF.model('perceptron', {
    label: 'Perceptron', task: 'classification', order: 12, icon: 'curve',
    blurb: 'The oldest learning rule: go through the examples, and on every mistake move the line towards the example. It settles only if a straight line can separate the classes.',
    params: [{ id: 'eta', label: 'Step size η', kind: 'number', def: 1, min: 0.001, max: 10, log: true }, { id: 'epochs', label: 'Passes over the data', kind: 'int', def: 5, min: 1, max: 100 }, { id: 'shuffle', label: 'Shuffle each pass', kind: 'bool', def: false }],
    make: (p, data, rng, ctx) => ML().create('Perceptron', { eta: p.eta, epochs: p.epochs, shuffle: !!p.shuffle, seed: ctx ? ctx.seed : 0 }),
    internals(est, data) {
      // the line after each mistake of the first passes (a binary perceptron), for the picture of it moving
      const ups = (est.trace || []).filter((s) => s.kind === 'update' && s.w).slice(0, 80).map((s) => ({ i: s.i, epoch: s.epoch, w: arr(s.w), b: r4(s.b), head: s.head }));
      const ep = (est.trace || []).filter((s) => s.kind === 'epoch').map((s) => ({ epoch: s.epoch, mistakes: s.mistakes, head: s.head }));
      return Object.assign(coefOf(est, data), { updates: ups, epochs: ep });
    },
  });

  // ---- neighbours and probabilistic models ------------------------------------------------------------------------
  WF.model('knn', {
    label: 'k nearest neighbours', task: 'both', order: 20, icon: 'target',
    blurb: 'No training at all: to predict, find the k training rows nearest the new one and let them vote (or average). Small k follows every point; large k smooths.',
    params: [
      { id: 'k', label: 'k (neighbours)', kind: 'int', def: 5, min: 1, max: 100 },
      { id: 'weights', label: 'Votes', kind: 'choice', def: 'uniform', options: [['uniform', 'equal'], ['distance', 'nearer counts more']] },
      { id: 'p', label: 'Distance', kind: 'choice', def: 2, options: [[2, 'Euclidean'], [1, 'Manhattan']] },
    ],
    make: (p, data) => ML().create(isClf(data) ? 'KNeighborsClassifier' : 'KNeighborsRegressor', { k: +p.k, weights: p.weights, p: +p.p, algorithm: data.X.length > 400 ? 'kd' : 'brute' }),
    internals: (est, data) => ({ k: est.params.k, n: data.X.length, features: names(data) }),
  });
  WF.model('naivebayes', {
    label: 'Naive Bayes', task: 'classification', order: 22, icon: 'chip',
    blurb: 'Bayes’ rule with the naive guess that features are independent within a class: Gaussian for measurements (an axis-aligned ellipse per class), multinomial for counts such as words.',
    params: [{ id: 'variant', label: 'Kind', kind: 'choice', def: 'gaussian', options: [['gaussian', 'Gaussian (measurements)'], ['multinomial', 'Multinomial (counts)']] }, { id: 'alpha', label: 'Smoothing α (multinomial)', kind: 'number', def: 1, min: 0, max: 10, step: 0.1, show: (p) => p.variant === 'multinomial' }],
    make(p, data) {
      if (p.variant === 'multinomial' && data.X.some((r) => r.some((v) => v < 0))) throw new Error('multinomial naive Bayes counts things, so it needs features of zero or more; these have negative values (scale with min-max, or use the Gaussian kind)');
      return p.variant === 'multinomial' ? ML().create('MultinomialNB', { alpha: p.alpha }) : ML().create('GaussianNB', {});
    },
    internals(est, data) {
      if (est.theta) return { variant: 'gaussian', classes: data.classes, priors: arr(est.priors || est.classPrior), means: mat(est.theta), vars: mat(est.var), features: names(data) };
      const lp = est.featureLogProb || est.logProb || null;
      const top = lp ? lp.map((row) => Array.from(row).map((v, j) => [j, v]).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([j, v]) => [names(data)[j] || 'f' + j, r4(Math.exp(v))])) : null;
      return { variant: 'multinomial', classes: data.classes, top, features: names(data) };
    },
  });
  WF.model('lda', {
    label: 'Linear discriminant analysis', task: 'classification', order: 24, icon: 'chip',
    blurb: 'Each class a Gaussian cloud with one covariance they all share: the boundaries between classes come out straight.',
    params: [{ id: 'shrinkage', label: 'Shrinkage', kind: 'number', def: 0, min: 0, max: 1, step: 0.05 }],
    make: (p) => ML().create('LDA', { shrinkage: p.shrinkage }),
    internals: (est, data) => ({ classes: data.classes, priors: arr(est.priors), means: mat(est.means), shared: mat(est.covariance), features: names(data) }),
  });
  WF.model('qda', {
    label: 'Quadratic discriminant analysis', task: 'classification', order: 25, icon: 'chip',
    blurb: 'Each class a Gaussian cloud with a covariance of its own: the boundaries curve (they are conic sections).',
    params: [{ id: 'regParam', label: 'Regularisation', kind: 'number', def: 0, min: 0, max: 1, step: 0.05 }],
    make(p, data) {
      const e = ML().create('QDA', { regParam: p.regParam }), fit = e.fit.bind(e), d = data.X.length ? data.X[0].length : 0;
      e.fit = (X, y) => {
        try { return fit(X, y); } catch (err) {
          const counts = new Map(); for (const c of y) counts.set(c, (counts.get(c) || 0) + 1);
          const small = Math.min(...counts.values());
          throw new Error(small <= d ? 'a class has only ' + small + ' training rows for ' + d + ' features, too few to estimate its own covariance: raise the regularisation, use fewer features, or try LDA' : 'QDA could not be fitted (' + err.message + '): raise the regularisation');
        }
      };
      return e;
    },
    internals: (est, data) => ({ classes: data.classes, priors: arr(est.priors), means: mat(est.means), covs: (est.covariances || []).map(mat), features: names(data) }),
  });

  // ---- trees and ensembles -----------------------------------------------------------------------------------
  /** a tree's nodes as plain JSON, at most maxNodes of them (a breadth-first cut) */
  const treeJSON = (T, maxNodes) => {
    const nodes = T.nodes || [];
    const keep = new Set(), q = [0];
    while (q.length && keep.size < (maxNodes || 127)) { const i = q.shift(); if (!nodes[i]) continue; keep.add(i); if (nodes[i].left >= 0) q.push(nodes[i].left, nodes[i].right); }
    return nodes.filter((n) => keep.has(n.id)).map((n) => ({ id: n.id, d: n.depth, n: n.n, imp: r4(n.impurity), l: keep.has(n.left) ? n.left : -1, r: keep.has(n.right) ? n.right : -1, cut: n.left >= 0 && !keep.has(n.left), f: n.feature, t: r4(n.threshold), v: Array.isArray(n.value) || ArrayBuffer.isView(n.value) ? arr(n.value) : r4(n.value) }));
  };
  WF.treeJSON = treeJSON;
  const treeParams = (clf) => [
    { id: 'criterion', label: 'Impurity', kind: 'choice', def: 'gini', options: [['gini', 'Gini'], ['entropy', 'Entropy']], help: 'for classification; regression trees use the squared error' },
    { id: 'maxDepth', label: 'Maximum depth (0: none)', kind: 'int', def: clf ? 4 : 4, min: 0, max: 30 },
    { id: 'minSamplesLeaf', label: 'Fewest rows in a leaf', kind: 'int', def: 1, min: 1, max: 200 },
  ];
  WF.model('tree', {
    label: 'Decision tree', task: 'both', order: 30, icon: 'tidy',
    blurb: 'Asks one yes-or-no question about one feature at a time, choosing each question so the two groups it makes are as pure as possible. Its boundary is made of straight cuts.',
    params: treeParams(true).concat([{ id: 'ccpAlpha', label: 'Pruning α', kind: 'number', def: 0, min: 0, max: 0.5, step: 0.001 }]),
    make: (p, data, rng, ctx) => ML().create(isClf(data) ? 'DecisionTreeClassifier' : 'DecisionTreeRegressor', Object.assign({ maxDepth: depth(p.maxDepth), minSamplesLeaf: +p.minSamplesLeaf, ccpAlpha: +p.ccpAlpha || 0, seed: ctx ? ctx.seed : 0 }, isClf(data) ? { criterion: p.criterion } : {})),
    internals(est, data) {
      const tr = est.trace || [];
      // the order the splits were made in, with the best threshold each feature offered there
      const grow = [];
      for (let i = 0; i < tr.length; i++) {
        const s = tr[i];
        if (s.kind !== 'split') continue;
        const cand = tr.slice(0, i).reverse().find((x) => x.kind === 'candidates' && x.node === s.node);
        grow.push({ node: s.node, f: s.feature, t: r4(s.threshold), il: r4(s.il), ir: r4(s.ir), say: s.say, tried: cand ? (cand.candidates || []).reduce((a, c) => a + (c.thresholds ? c.thresholds.length : 0), 0) : 0,
          // the impurity after each threshold of the chosen feature: the curve the split was read from
          curve: cand ? (() => { const c = (cand.candidates || []).find((x) => x.feature === s.feature); if (!c || !c.scores) return null; const k = Math.max(1, Math.ceil(c.thresholds.length / 60)); return c.thresholds.filter((_, j) => j % k === 0).map((t, j) => [r4(t), r4(c.scores[j * k])]); })() : null });
      }
      return { nodes: treeJSON(est.tree, 63), depth: est.depth, leaves: est.nLeaves, nodeCount: est.nodeCount, features: names(data), classes: data.classes, criterion: est.params.criterion, grow: grow.slice(0, 40), importances: arr(est.featureImportances) };
    },
  });
  WF.model('forest', {
    label: 'Random forest', task: 'both', order: 32, icon: 'tidy',
    blurb: 'Many deep trees, each grown on a bootstrap sample of the rows and choosing among a random few features at every split; they vote, and their mistakes cancel.',
    params: [
      { id: 'nEstimators', label: 'Trees', kind: 'int', def: 50, min: 1, max: 300 },
      { id: 'maxDepth', label: 'Maximum depth (0: none)', kind: 'int', def: 0, min: 0, max: 30 },
      { id: 'maxFeatures', label: 'Features tried at a split', kind: 'choice', def: 'sqrt', options: [['sqrt', '√ of the features'], ['all', 'all of them']] },
    ],
    make: (p, data, rng, ctx) => ML().create(isClf(data) ? 'RandomForestClassifier' : 'RandomForestRegressor', { nEstimators: +p.nEstimators, maxDepth: depth(p.maxDepth), maxFeatures: p.maxFeatures === 'all' || !isClf(data) ? null : 'sqrt', seed: ctx ? ctx.seed : 0, oobScore: true }),
    internals: (est, data) => ({ n: est.trees.length, trees: est.trees.slice(0, 3).map((t) => treeJSON(t, 31)), oob: r4(est.oobScore), importances: arr(est.featureImportances), features: names(data), classes: data.classes }),
  });
  WF.model('adaboost', {
    label: 'AdaBoost', task: 'classification', order: 34, icon: 'tidy',
    blurb: 'A row of tiny trees, each trained on the rows the ones before got wrong, weighted up: the final say is a weighted vote of them all.',
    params: [{ id: 'nEstimators', label: 'Rounds', kind: 'int', def: 30, min: 1, max: 300 }, { id: 'learningRate', label: 'Learning rate', kind: 'number', def: 1, min: 0.01, max: 2, step: 0.01 }, { id: 'maxDepth', label: 'Depth of each tree', kind: 'int', def: 1, min: 1, max: 4 }],
    make: (p, data, rng, ctx) => ML().create('AdaBoostClassifier', { nEstimators: +p.nEstimators, learningRate: +p.learningRate, maxDepth: +p.maxDepth, seed: ctx ? ctx.seed : 0 }),
    internals(est, data) {
      const rounds = (est.trace || []).filter((s) => s.kind === 'stage').slice(0, 12);
      // the weight of every row before each of the first rounds (scaled so the mean weight is 1)
      const n = data.X.length, take = Math.min(n, 400);
      return { n: est.trees.length, alphas: arr(est.alphas), errors: arr(est.errors), stumps: est.trees.slice(0, 12).map((t) => { const r = t.nodes[0]; return { f: r.feature, t: r4(r.threshold) }; }),
        weights: rounds.map((s) => Array.from((s.weights || []).slice(0, take), (w) => r4(w * n))), says: rounds.map((s) => s.say), features: names(data), classes: data.classes };
    },
  });
  WF.model('gboost', {
    label: 'Gradient boosting', task: 'both', order: 36, icon: 'tidy',
    blurb: 'Trees added one at a time, each fitted to what the model so far still gets wrong (the residuals), and shrunk by the learning rate before it joins.',
    params: [{ id: 'nEstimators', label: 'Stages', kind: 'int', def: 60, min: 1, max: 500 }, { id: 'learningRate', label: 'Learning rate', kind: 'number', def: 0.1, min: 0.001, max: 1, log: true }, { id: 'maxDepth', label: 'Depth of each tree', kind: 'int', def: 3, min: 1, max: 8 }, { id: 'subsample', label: 'Share of rows per stage', kind: 'number', def: 1, min: 0.1, max: 1, step: 0.05 }],
    make: (p, data, rng, ctx) => ML().create(isClf(data) ? 'GradientBoostingClassifier' : 'GradientBoostingRegressor', { nEstimators: +p.nEstimators, learningRate: +p.learningRate, maxDepth: +p.maxDepth, subsample: +p.subsample, seed: ctx ? ctx.seed : 0 }),
    internals: (est, data) => ({ n: est.stages.length, lr: est.params.learningRate, trainLoss: arr(est.trainLoss), K: est.K, features: names(data), classes: data.classes, first: treeJSON(est.stages[0][0], 15) }),
  });

  // ---- support vector machines ---------------------------------------------------------------------------------
  const gammaOf = (p) => (+p.gamma > 0 ? +p.gamma : 'scale');
  WF.model('svm', {
    label: 'Support vector machine', task: 'classification', order: 40, icon: 'target',
    blurb: 'The boundary with the widest margin to the nearest examples, the support vectors. C sets the price of a point inside the margin; a kernel lets the boundary bend, and γ how sharply.',
    params: [
      { id: 'kernel', label: 'Kernel', kind: 'choice', def: 'rbf', options: [['linear', 'Linear'], ['poly', 'Polynomial'], ['rbf', 'RBF (Gaussian)']] },
      { id: 'C', label: 'C (cost of a margin violation)', kind: 'number', def: 1, min: 0.001, max: 10000, log: true },
      { id: 'gamma', label: 'γ (0: from the data’s spread)', kind: 'number', def: 0, min: 0, max: 100, step: 0.01, show: (p) => p.kernel !== 'linear' },
      { id: 'degree', label: 'Degree', kind: 'int', def: 3, min: 2, max: 6, show: (p) => p.kernel === 'poly' },
      { id: 'probability', label: 'Probabilities (Platt scaling: five more fits)', kind: 'bool', def: false, help: 'needed for ROC curves, log loss and a decision threshold' },
    ],
    make: (p, data, rng, ctx) => ML().create('SVC', { kernel: p.kernel, C: +p.C, gamma: gammaOf(p), degree: +p.degree, coef0: p.kernel === 'poly' ? 1 : 0, probability: !!p.probability, seed: ctx ? ctx.seed : 0 }),
    internals(est, data) {
      const m = est.machines || [], C = est.params.C;
      return { kernel: est.params.kernel, C: r4(C), gamma: r4(est.gammaValue), support: Array.from(est.support || []).slice(0, 600), nSupport: arr(est.nSupport), n: data.X.length,
        // binary: each support vector's multiplier α (|coefficient|), at the cap C or below it
        alpha: m.length === 1 ? m[0].sv.map((i, k) => [i, r4(Math.abs(m[0].coef[k]))]).slice(0, 600) : null, smoSteps: (est.trace || []).filter((s) => s.kind === 'pair').length, classes: data.classes, features: names(data) };
    },
  });
  WF.model('svr', {
    label: 'Support vector regression', task: 'regression', order: 42, icon: 'target',
    blurb: 'A tube of half-width ε round the fitted curve: points inside it cost nothing, and only the points on or outside it, the support vectors, shape the fit.',
    params: [
      { id: 'kernel', label: 'Kernel', kind: 'choice', def: 'rbf', options: [['linear', 'Linear'], ['poly', 'Polynomial'], ['rbf', 'RBF (Gaussian)']] },
      { id: 'C', label: 'C', kind: 'number', def: 10, min: 0.001, max: 10000, log: true },
      { id: 'epsilon', label: 'ε (half-width of the tube)', kind: 'number', def: 0.1, min: 0, max: 100, step: 0.01 },
      { id: 'gamma', label: 'γ (0: from the data’s spread)', kind: 'number', def: 0, min: 0, max: 100, step: 0.01, show: (p) => p.kernel !== 'linear' },
    ],
    make: (p) => ML().create('SVR', { kernel: p.kernel, C: +p.C, epsilon: +p.epsilon, gamma: gammaOf(p), coef0: p.kernel === 'poly' ? 1 : 0 }),
    internals: (est, data) => ({ kernel: est.params.kernel, C: r4(est.params.C), epsilon: r4(est.params.epsilon), support: Array.from(est.support || []).slice(0, 600), n: data.X.length, features: names(data) }),
  });
})(globalThis.Nooral = globalThis.Nooral || {});
