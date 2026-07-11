// Ensembles of trees: random forests (bootstrap, feature subsampling, out-of-bag score, impurity
// importance), AdaBoost (SAMME) and gradient boosting for regression and classification.
// The randomness can be supplied from outside (bootstrap draws, tree seeds, subsample masks) so a
// run can be replayed exactly; otherwise it comes from the seeded generator.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x), fv = (v) => ML.fmtv(v);

  // max features, by scikit-learn's rule: a whole number is a count of features (1 means one), a number below 1 a
  // share of them (at least one); JS has one kind of number, so a share of all of them is written as the string
  // '1.0' (or any string with a decimal point), and null or 'all' means every feature
  ML.resolveMaxFeatures = (mf, d) => {
    if (mf == null || mf === 'all') return null;
    if (mf === 'sqrt' || mf === 'auto') return Math.max(1, Math.floor(Math.sqrt(d)));
    if (mf === 'log2') return Math.max(1, Math.floor(Math.log2(d)));
    if (typeof mf === 'string') {
      const v = +mf;
      if (!Number.isFinite(v) || v <= 0) throw new Error('maxFeatures ' + mf + ' is not a count or a share');
      if (!mf.includes('.')) return v >= d ? null : v;
      if (v > 1) throw new Error('maxFeatures ' + mf + ': a share is at most 1');
      const k = Math.max(1, Math.floor(v * d));
      return k >= d ? null : k;
    }
    if (!(mf > 0)) throw new Error('maxFeatures must be above 0');
    if (Number.isInteger(mf)) return mf >= d ? null : mf;
    if (mf < 1) return Math.max(1, Math.floor(mf * d));
    throw new Error('maxFeatures ' + mf + ': a count must be a whole number, a share at most 1');
  };
  const resolveMaxFeatures = ML.resolveMaxFeatures;
  const treeSeed = (rng) => 1 + rng.int(0x7ffffffe);
  const leafValue = (T, x) => T.nodes[ML.treeApply(T, x)].value;

  function forestFit(X, y, tr) {
    const p = this.params, n = X.length, d = X[0].length, cls = this.kind === 'classifier', rng = ML.rng(p.seed);
    let yy = y;
    if (cls) { const e = ML.encode(y); this.classes = e.classes; yy = e.idx; }
    const K = cls ? this.classes.length : 0, mf = resolveMaxFeatures(p.maxFeatures, d);
    const CW = cls ? ML.classWeights(p.classWeight, this.classes, yy) : null;
    this.trees = []; this.inBag = [];
    const oobSum = X.map(() => (cls ? new Array(K).fill(0) : 0)), oobCount = new Array(n).fill(0);
    for (let b = 0; b < p.nEstimators; b++) {
      const draws = p.bootstrapIndices ? p.bootstrapIndices[b] : p.bootstrap ? ML.range(n).map(() => rng.int(n)) : null;
      const w = new Array(n).fill(draws ? 0 : 1);
      if (draws) for (const i of draws) w[i]++;
      const wt = CW ? w.map((v, i) => v * CW[yy[i]]) : w;
      const seed = p.treeSeeds ? p.treeSeeds[b] : treeSeed(rng);
      const T = ML.growTree(X, yy, { criterion: cls ? p.criterion : 'mse', maxDepth: p.maxDepth == null ? Infinity : p.maxDepth, minSamplesSplit: p.minSamplesSplit, minSamplesLeaf: p.minSamplesLeaf, maxFeatures: mf, seed, nClasses: K, w: wt, float32: p.float32 });
      this.trees.push(T); this.inBag.push(w);
      let oobN = 0;
      if (draws) for (let i = 0; i < n; i++) if (w[i] === 0) {
        oobN++; oobCount[i]++;
        const v = leafValue(T, X[i]);
        if (cls) v.forEach((q, k) => (oobSum[i][k] += q)); else oobSum[i] += v;
      }
      tr.add('tree', 'Tree ' + (b + 1) + ': ' + (draws ? 'a bootstrap sample leaves ' + oobN + ' rows out of the bag; ' : '') + 'it has ' + ML.treeLeaves(T) + ' leaves and depth ' + ML.treeDepth(T) + '.', { tree: b, inBag: w, oob: oobN, depth: ML.treeDepth(T), leaves: ML.treeLeaves(T) });
    }
    // impurity importance: the trees' normalised importances, averaged and normalised again
    const imp = new Array(d).fill(0);
    this.trees.forEach((T) => T.importances.forEach((v, j) => (imp[j] += v / this.trees.length)));
    const s = ML.sum(imp);
    this.featureImportances = s > 0 ? imp.map((v) => v / s) : imp;
    if (p.bootstrap && p.oobScore) {
      if (cls) {
        this.oobDecision = oobSum.map((v, i) => v.map((q) => q / Math.max(oobCount[i], 1)));
        this.oobScore = ML.accuracy(yy, this.oobDecision.map(ML.argmax));
      } else {
        this.oobPrediction = oobSum.map((v, i) => v / Math.max(oobCount[i], 1));
        this.oobScore = ML.r2(y, this.oobPrediction);
      }
      tr.add('oob', 'Each row is predicted only by the trees that did not see it: out-of-bag score ' + f(this.oobScore) + '.', { oobScore: this.oobScore });
    }
  }
  const forestDefaults = { classWeight: null, nEstimators: 100, criterion: 'gini', maxDepth: null, minSamplesSplit: 2, minSamplesLeaf: 1, maxFeatures: 'sqrt', bootstrap: true, oobScore: false, seed: 0, float32: false, bootstrapIndices: null, treeSeeds: null };
  ML.model('RandomForestClassifier', {
    kind: 'classifier', defaults: forestDefaults, fit: forestFit,
    predictProba(X) { return X.map((x) => { const s = new Array(this.classes.length).fill(0); this.trees.forEach((T) => leafValue(T, x).forEach((q, k) => (s[k] += q / this.trees.length))); return s; }); },
    predict(X) { return this._argmaxPredict(X); },
  });
  ML.model('RandomForestRegressor', {
    kind: 'regressor', defaults: Object.assign({}, forestDefaults, { criterion: 'mse', maxFeatures: null }), fit: forestFit,
    predict(X) { return X.map((x) => ML.mean(this.trees.map((T) => leafValue(T, x)))); },
  });

  // ---- AdaBoost, multi-class SAMME, on weighted trees (stumps by default) ----
  ML.model('AdaBoostClassifier', {
    kind: 'classifier',
    defaults: { nEstimators: 50, learningRate: 1, maxDepth: 1, seed: 0, treeSeeds: null, float32: false },
    fit(X, y, tr) {
      const p = this.params, n = X.length, rng = ML.rng(p.seed), { classes, idx } = ML.encode(y), K = classes.length;
      this.classes = classes; this.trees = []; this.alphas = []; this.errors = [];
      let w = new Array(n).fill(1 / n);
      for (let m = 0; m < p.nEstimators; m++) {
        const seed = p.treeSeeds ? p.treeSeeds[m] : treeSeed(rng);
        const T = ML.growTree(X, idx, { criterion: 'gini', maxDepth: p.maxDepth, seed, nClasses: K, w, float32: p.float32 });
        const pred = X.map((x) => ML.argmax(leafValue(T, x))), wrong = pred.map((c, i) => c !== idx[i]);
        let err = 0, W = 0;
        for (let i = 0; i < n; i++) { W += w[i]; if (wrong[i]) err += w[i]; }
        err /= W;
        if (err <= 0) { this.trees.push(T); this.alphas.push(1); this.errors.push(0); tr.add('stage', 'Round ' + (m + 1) + ' classifies every weighted sample correctly; boosting stops.', { round: m, error: 0 }); break; }
        if (err >= 1 - 1 / K) { tr.add('stop', 'Round ' + (m + 1) + ' is no better than guessing (error ' + f(err) + '); it is discarded and boosting stops.', { round: m, error: err }); if (!this.trees.length) throw new Error('first weak learner is worse than random'); break; }
        const alpha = p.learningRate * (Math.log((1 - err) / err) + Math.log(K - 1));
        this.trees.push(T); this.alphas.push(alpha); this.errors.push(err);
        tr.add('stage', 'Round ' + (m + 1) + ': weighted error ' + f(err) + ', so this learner gets the say alpha = ' + f(alpha) + '; the ' + wrong.filter(Boolean).length + ' misclassified samples have their weights multiplied by ' + f(Math.exp(alpha)) + '.', { round: m, error: err, alpha, weights: w.slice(), wrong });
        if (m === p.nEstimators - 1) break;
        w = w.map((v, i) => (wrong[i] && v > 0 ? Math.exp(Math.log(v) + alpha) : v));
        const s = ML.sum(w);
        w = w.map((v) => v / s);
      }
      this.sampleWeight = w;
    },
    // the weighted vote of the rounds: +alpha for the predicted class, -alpha/(K-1) for the others
    decisionFunction(X, upTo) {
      const K = this.classes.length, M = upTo || this.trees.length, aSum = ML.sum(this.alphas.slice(0, M));
      const out = X.map((x) => {
        const s = new Array(K).fill(0);
        for (let m = 0; m < M; m++) { const c = ML.argmax(leafValue(this.trees[m], x)); for (let k = 0; k < K; k++) s[k] += k === c ? this.alphas[m] : -this.alphas[m] / (K - 1); }
        return s.map((v) => v / aSum);
      });
      return K === 2 ? out.map((s) => s[1] - s[0]) : out;
    },
    predict(X) {
      const D = this.decisionFunction(X);
      return this.classes.length === 2 ? D.map((v) => this.classes[v > 0 ? 1 : 0]) : D.map((v) => this.classes[ML.argmax(v)]);
    },
  });

  // ---- gradient boosting ----
  // Each stage fits a small regression tree to the negative gradient (the residuals), then sets the leaf
  // values by a Newton step for log loss, and adds learningRate times the tree to the raw scores.
  const safeDiv = (a, b) => (Math.abs(b) < 1e-150 ? 0 : a / b);
  function gbFit(X, y, tr) {
    const p = this.params, n = X.length, cls = this.kind === 'classifier', rng = ML.rng(p.seed);
    let K = 1, yi = y, init;
    if (cls) {
      const e = ML.encode(y); this.classes = e.classes; yi = e.idx;
      const prior = this.classes.map((_, k) => yi.filter((v) => v === k).length / n).map((q) => Math.min(Math.max(q, 2.220446049250313e-16), 1 - 2.220446049250313e-16));
      if (this.classes.length === 2) init = [Math.log(prior[1] / (1 - prior[1]))];
      else { K = this.classes.length; const lp = prior.map(Math.log), m = ML.mean(lp); init = lp.map((v) => v - m); }
    } else init = [ML.mean(y)];
    this.init = init; this.K = K; this.stages = [];
    const raw = X.map(() => init.slice()), nIn = Math.floor(p.subsample * n);
    this.trainLoss = [];
    let sIdx = 0;
    for (let m = 0; m < p.nEstimators; m++) {
      // sample mask: subsample < 1 keeps nIn rows chosen at random for this stage
      let mask = null;
      if (p.subsample < 1) {
        if (p.masks) mask = p.masks[m];
        else { const pick = rng.perm(n).slice(0, nIn), mk = new Array(n).fill(false); pick.forEach((i) => (mk[i] = true)); mask = mk; }
      }
      const prob = cls ? raw.map((r) => (K === 1 ? [ML.sigmoid(r[0])] : ML.softmax(r))) : null;
      const stage = [];
      for (let k = 0; k < K; k++) {
        const target = yi.map((v, i) => (cls ? (K === 1 ? v : v === k ? 1 : 0) : v));
        const neg = cls ? target.map((t, i) => t - prob[i][k]) : y.map((v, i) => v - raw[i][0]);
        const w = mask ? mask.map((b) => (b ? 1 : 0)) : null;
        const seed = p.treeSeeds ? p.treeSeeds[sIdx++] : treeSeed(rng);
        const T = ML.growTree(X, neg, { criterion: 'friedman_mse', maxDepth: p.maxDepth, minSamplesSplit: p.minSamplesSplit, minSamplesLeaf: p.minSamplesLeaf, seed, w, float32: p.float32 });
        const leafOf = X.map((x) => ML.treeApply(T, x));
        if (cls) {
          T.nodes.forEach((nd) => {
            if (nd.left >= 0) return;
            let num = 0, den = 0, cnt = 0;
            for (let i = 0; i < n; i++) if (leafOf[i] === nd.id && (!mask || mask[i])) { const q = target[i] - neg[i]; num += neg[i]; den += q * (1 - q); cnt++; }
            if (!cnt) { nd.value = 0; return; }
            nd.value = safeDiv((num / cnt) * (K > 1 ? (K - 1) / K : 1), den / cnt);
          });
        }
        for (let i = 0; i < n; i++) raw[i][k] += p.learningRate * T.nodes[leafOf[i]].value;
        stage.push(T);
      }
      this.stages.push(stage);
      const loss = cls
        ? ML.mean(raw.map((r, i) => (K === 1 ? Math.max(r[0], 0) - yi[i] * r[0] + Math.log1p(Math.exp(-Math.abs(r[0]))) : ML.logsumexp(r) - r[yi[i]])))
        : ML.mean(raw.map((r, i) => (y[i] - r[0]) ** 2));
      this.trainLoss.push(loss);
      tr.add('stage', 'Stage ' + (m + 1) + ': ' + K + (K > 1 ? ' trees fit' : ' tree fits') + ' the residuals' + (mask ? ' of a ' + nIn + '-row subsample' : '') + ', shrunk by ' + f(p.learningRate) + '; training ' + (cls ? 'log loss' : 'squared error') + ' is now ' + f(loss) + '.', { stage: m, loss, raw: raw.map((r) => r.slice()), leaves: stage.map((T) => T.nodes.filter((nd) => nd.left < 0).map((nd) => nd.value)) });
    }
  }
  function gbRaw(X, upTo) {
    const M = upTo == null ? this.stages.length : upTo, lr = this.params.learningRate;
    return X.map((x) => { const r = this.init.slice(); for (let m = 0; m < M; m++) this.stages[m].forEach((T, k) => (r[k] += lr * leafValue(T, x))); return r; });
  }
  const gbDefaults = { nEstimators: 100, learningRate: 0.1, maxDepth: 3, minSamplesSplit: 2, minSamplesLeaf: 1, subsample: 1, seed: 0, treeSeeds: null, masks: null, float32: false };
  ML.model('GradientBoostingRegressor', {
    kind: 'regressor', defaults: gbDefaults, fit: gbFit, raw: gbRaw,
    predict(X) { return this.raw(X).map((r) => r[0]); },
    stagedPredict(X) { return this.stages.map((_, m) => this.raw(X, m + 1).map((r) => r[0])); },
  });
  ML.model('GradientBoostingClassifier', {
    kind: 'classifier', defaults: gbDefaults, fit: gbFit, raw: gbRaw,
    decisionFunction(X) { const R = this.raw(X); return this.K === 1 ? R.map((r) => r[0]) : R; },
    predictProba(X) { return this.raw(X).map((r) => (this.K === 1 ? [1 - ML.sigmoid(r[0]), ML.sigmoid(r[0])] : ML.softmax(r))); },
    predict(X) { return this._argmaxPredict(X); },
  });
})();
