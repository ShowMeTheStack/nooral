// Generative classifiers: linear and quadratic discriminant analysis, Gaussian and multinomial naive Bayes.
// Each keeps its per-class statistics so the class ellipses and word counts can be drawn.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x), fv = (v) => ML.fmtv(v);
  const LOG2PI = Math.log(2 * Math.PI);

  function byClass(X, y) {
    const { classes, idx } = ML.encode(y), groups = classes.map(() => []);
    idx.forEach((k, i) => groups[k].push(X[i]));
    return { classes, idx, groups };
  }
  const probaFromJoint = (J) => J.map((r) => { const l = ML.logsumexp(r); return r.map((v) => Math.exp(v - l)); });

  // LDA: shared covariance. cov 'empirical' divides the within-class scatter by n; 'pooled' by n - K.
  ML.model('LDA', {
    kind: 'classifier',
    defaults: { cov: 'empirical', priors: null, shrinkage: 0 },
    fit(X, y, tr) {
      const p = this.params, { classes, groups } = byClass(X, y), n = X.length, d = X[0].length, K = classes.length;
      this.classes = classes;
      this.priors = p.priors || groups.map((g) => g.length / n);
      this.means = groups.map((g) => ML.colMeans(g));
      const Sw = ML.la.zeros(d, d);
      groups.forEach((g, k) => g.forEach((r) => { for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) Sw[a][b] += (r[a] - this.means[k][a]) * (r[b] - this.means[k][b]); }));
      const den = p.cov === 'pooled' ? n - K : n;
      let Sig = Sw.map((r) => r.map((v) => v / den));
      if (p.shrinkage) { const mu = ML.mean(Sig.map((r, i) => r[i])); Sig = Sig.map((r, i) => r.map((v, j) => (1 - p.shrinkage) * v + (i === j ? p.shrinkage * mu : 0))); }
      this.covariance = Sig;
      tr.add('means', 'Class means ' + this.means.map(fv).join(', ') + ' with priors ' + fv(this.priors) + '.', { means: this.means, priors: this.priors });
      tr.add('cov', 'One covariance matrix is shared by all classes (within-class scatter divided by ' + den + ').', { covariance: Sig });
      const W = ML.la.solve(Sig, ML.la.T(this.means));
      this.coef = this.means.map((m, k) => W.map((r) => r[k]));
      this.intercept = this.means.map((m, k) => -0.5 * ML.la.dot(m, this.coef[k]) + Math.log(this.priors[k]));
      tr.add('linear', 'Each class score is a linear function: w_k = Sigma^-1 mu_k, b_k = -mu_k w_k / 2 + log prior.', { coef: this.coef, intercept: this.intercept });
      // discriminant directions: eigenvectors of Sw^-1 Sb, for the projection picture
      const gm = ML.colMeans(X), Sb = ML.la.zeros(d, d);
      groups.forEach((g, k) => { const m = this.means[k]; for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) Sb[a][b] += g.length * (m[a] - gm[a]) * (m[b] - gm[b]); });
      const L = ML.la.chol(Sig);
      if (L) {
        const Li = ML.la.triInv(L), M = ML.la.matmul(ML.la.matmul(Li, Sb), ML.la.T(Li)), e = ML.la.symEig(M);
        const keep = Math.min(K - 1, d), V = ML.la.matmul(ML.la.T(Li), e.vectors);
        this.scalings = V.map((r) => r.slice(0, keep));
        const tot = ML.sum(e.values.slice(0, keep).map((v) => Math.max(v, 0)));
        this.explainedVarianceRatio = e.values.slice(0, keep).map((v) => Math.max(v, 0) / tot);
        this.xbar = gm;
      }
    },
    decisionFunction(X) { return X.map((r) => this.coef.map((w, k) => ML.la.dot(w, r) + this.intercept[k])); },
    predictProba(X) { return probaFromJoint(this.decisionFunction(X)); },
    predict(X) { return this._argmaxPredict(X); },
    transform(X) { return X.map((r) => { const c = r.map((v, j) => v - this.xbar[j]); return this.scalings[0].map((_, t) => this.scalings.reduce((s, row, j) => s + row[t] * c[j], 0)); }); },
  });

  // QDA: a covariance for each class (divided by n_k - ddof), optionally shrunk toward the identity by regParam
  ML.model('QDA', {
    kind: 'classifier',
    defaults: { regParam: 0, priors: null, ddof: 0 },
    fit(X, y, tr) {
      const p = this.params, { classes, groups } = byClass(X, y), n = X.length, d = X[0].length;
      this.classes = classes;
      this.priors = p.priors || groups.map((g) => g.length / n);
      this.means = groups.map((g) => ML.colMeans(g));
      this.covariances = groups.map((g, k) => {
        const C = ML.la.cov(g, p.ddof, this.means[k]);
        return C.map((r, a) => r.map((v, b) => (1 - p.regParam) * v + (a === b ? p.regParam : 0)));
      });
      this.chols = this.covariances.map((C) => ML.la.chol(C));
      this.covariances.forEach((C, k) => tr.add('class', 'Class ' + classes[k] + ': mean ' + fv(this.means[k]) + ', its own covariance with log-determinant ' + f(ML.la.logdetChol(this.chols[k])) + '.', { k, mean: this.means[k], cov: C }));
    },
    decisionFunction(X) {
      return X.map((r) => this.means.map((m, k) => {
        const L = this.chols[k], z = ML.la.cholSolve(L, r.map((v, j) => v - m[j]));
        return -0.5 * (ML.la.logdetChol(L) + ML.la.dot(z, r.map((v, j) => v - m[j]))) + Math.log(this.priors[k]);
      }));
    },
    predictProba(X) { return probaFromJoint(this.decisionFunction(X)); },
    predict(X) { return this._argmaxPredict(X); },
  });

  // Gaussian naive Bayes: independent normal features per class; varSmoothing adds a fraction of the largest variance
  ML.model('GaussianNB', {
    kind: 'classifier',
    defaults: { varSmoothing: 1e-9, priors: null },
    fit(X, y, tr) {
      const p = this.params, { classes, groups } = byClass(X, y), n = X.length;
      this.classes = classes;
      const eps = p.varSmoothing * Math.max(...ML.colVars(X));
      this.epsilon = eps;
      this.priors = p.priors || groups.map((g) => g.length / n);
      this.theta = groups.map((g) => ML.colMeans(g));
      this.var = groups.map((g) => ML.colVars(g).map((v) => v + eps));
      classes.forEach((c, k) => tr.add('class', 'Class ' + c + ': prior ' + f(this.priors[k]) + ', means ' + fv(this.theta[k]) + ', variances ' + fv(this.var[k]) + '.', { k, mean: this.theta[k], var: this.var[k], prior: this.priors[k] }));
    },
    jointLogLikelihood(X) {
      return X.map((r) => this.theta.map((m, k) => {
        let s = Math.log(this.priors[k]);
        for (let j = 0; j < r.length; j++) s -= 0.5 * (LOG2PI + Math.log(this.var[k][j])) + (0.5 * (r[j] - m[j]) ** 2) / this.var[k][j];
        return s;
      }));
    },
    // per-feature log-likelihood terms for one sample, for the "why" picture
    explain(x) {
      return this.classes.map((c, k) => ({ cls: c, prior: Math.log(this.priors[k]), terms: x.map((v, j) => -0.5 * (LOG2PI + Math.log(this.var[k][j])) - (0.5 * (v - this.theta[k][j]) ** 2) / this.var[k][j]) }));
    },
    predictProba(X) { return probaFromJoint(this.jointLogLikelihood(X)); },
    predict(X) { return this._argmaxPredict(X); },
  });

  // multinomial naive Bayes on counts, with additive (Laplace / Lidstone) smoothing alpha
  ML.model('MultinomialNB', {
    kind: 'classifier',
    defaults: { alpha: 1, fitPrior: true },
    fit(X, y, tr) {
      const p = this.params, { classes, groups } = byClass(X, y), n = X.length, d = X[0].length;
      this.classes = classes;
      this.featureCount = groups.map((g) => { const c = new Array(d).fill(0); g.forEach((r) => r.forEach((v, j) => (c[j] += v))); return c; });
      this.classLogPrior = groups.map((g) => (p.fitPrior ? Math.log(g.length / n) : -Math.log(classes.length)));
      // as scikit-learn does, an alpha below 1e-10 is raised to 1e-10 (alpha = 0 would put log 0 into the sums)
      const a = Math.max(+p.alpha || 0, 1e-10);
      if (a !== p.alpha) tr.add('alpha', 'alpha = ' + f(p.alpha) + ' is too small to divide safely: it is raised to 1e-10, so unseen words get a tiny probability rather than zero.', { alpha: a });
      this.alpha = a;
      this.featureLogProb = this.featureCount.map((c) => { const tot = ML.sum(c) + a * d; return c.map((v) => Math.log((v + a) / tot)); });
      classes.forEach((c, k) => tr.add('class', 'Class ' + c + ': ' + ML.sum(this.featureCount[k]) + ' counted tokens; with alpha = ' + f(p.alpha) + ' every feature gets probability (count + alpha) / (total + alpha d).', { k, counts: this.featureCount[k], logProb: this.featureLogProb[k] }));
    },
    jointLogLikelihood(X) { return X.map((r) => this.featureLogProb.map((lp, k) => this.classLogPrior[k] + ML.la.dot(r, lp))); },
    explain(x) { return this.classes.map((c, k) => ({ cls: c, prior: this.classLogPrior[k], terms: x.map((v, j) => v * this.featureLogProb[k][j]) })); },
    predictProba(X) { return probaFromJoint(this.jointLogLikelihood(X)); },
    predict(X) { return this._argmaxPredict(X); },
  });
})();
