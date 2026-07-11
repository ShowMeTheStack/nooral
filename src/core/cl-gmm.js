// Gaussian mixtures fitted by expectation-maximisation, with full, diagonal or spherical covariances.
// Each iteration's means, covariances, weights and log-likelihood are traced, so the ellipses can breathe.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);
  const LOG2PI = Math.log(2 * Math.PI), TINY = 10 * 2.220446049250313e-16;

  // a covariance of any type as a full matrix
  const asFull = (c, type, d) => (type === 'full' ? c : type === 'diag' ? ML.la.zeros(d, d).map((r, i) => r.map((_, j) => (i === j ? c[i] : 0))) : ML.la.zeros(d, d).map((r, i) => r.map((_, j) => (i === j ? c : 0))));

  // log N(x | mean, cov) for every row and component
  function logGauss(X, means, covs, type) {
    const d = X[0].length;
    return means.map((m, k) => {
      if (type === 'full') {
        const L = ML.la.chol(covs[k]);
        if (!L) throw new Error('covariance ' + k + ' is not positive definite; increase regCovar');
        const ld = ML.la.logdetChol(L), Li = ML.la.triInv(L);
        return X.map((x) => { const z = Li.map((r) => r.reduce((s, v, j) => s + v * (x[j] - m[j]), 0)); return -0.5 * (d * LOG2PI + ld + ML.la.dot(z, z)); });
      }
      const v = type === 'diag' ? covs[k] : new Array(d).fill(covs[k]);
      const ld = ML.sum(v.map(Math.log));
      return X.map((x) => { let q = 0; for (let j = 0; j < d; j++) q += (x[j] - m[j]) ** 2 / v[j]; return -0.5 * (d * LOG2PI + ld + q); });
    });
  }

  ML.model('GaussianMixture', {
    kind: 'cluster',
    defaults: { k: 3, covarianceType: 'full', tol: 1e-3, regCovar: 1e-6, maxIter: 100, seed: 0, meansInit: null, weightsInit: null, covariancesInit: null },
    fit(X, y, tr) {
      const p = this.params, n = X.length, d = X[0].length, K = p.k, type = p.covarianceType;
      let means, weights, covs;
      if (p.meansInit) {
        means = p.meansInit.map((m) => m.slice());
        weights = p.weightsInit ? p.weightsInit.slice() : new Array(K).fill(1 / K);
        covs = p.covariancesInit ? JSON.parse(JSON.stringify(p.covariancesInit)) : null;
      } else {
        // start from k-means responsibilities
        const km = new ML.KMeans({ k: K, seed: p.seed, trace: false }).fit(X);
        const R = X.map((_, i) => ML.range(K).map((k) => (km.labels[i] === k ? 1 : 0)));
        ({ means, weights, covs } = mStep(X, R, type, p.regCovar));
      }
      if (!covs) covs = mStep(X, eStep(X, means, weights, means.map(() => (type === 'full' ? ML.la.eye(d) : type === 'diag' ? new Array(d).fill(1) : 1)), type).R, type, p.regCovar).covs;
      let lower = -Infinity, it = 0, converged = false;
      this.history = [];
      for (it = 1; it <= p.maxIter; it++) {
        const prev = lower, e = eStep(X, means, weights, covs, type);
        ({ means, weights, covs } = mStep(X, e.R, type, p.regCovar));
        lower = e.ll;
        this.history.push(lower);
        tr.add('em', 'Iteration ' + it + ': the E-step gives responsibilities (mean log-likelihood ' + f(lower) + '); the M-step moves the means to ' + means.map((m) => ML.fmtv(m, 2)).join(', ') + ' with weights ' + ML.fmtv(weights) + '.', { iter: it, ll: lower, means: means.map((m) => m.slice()), weights: weights.slice(), covariances: JSON.parse(JSON.stringify(covs)), resp: e.R });
        if (Math.abs(lower - prev) < p.tol) { converged = true; break; }
      }
      Object.assign(this, { means, weights, covariances: covs, lowerBound: lower, nIter: Math.min(it, p.maxIter), converged });
      this.labels = this.predict(X);
      tr.add('done', (converged ? 'Converged' : 'Stopped') + ' after ' + this.nIter + ' iterations; BIC = ' + f(this.bic(X)) + '.', { means, weights });
    },
    predictProba(X) { return eStep(X, this.means, this.weights, this.covariances, this.params.covarianceType).R; },
    predict(X) { return this.predictProba(X).map(ML.argmax); },
    scoreSamples(X) { const L = logGauss(X, this.means, this.covariances, this.params.covarianceType); return X.map((_, i) => ML.logsumexp(this.means.map((_, k) => L[k][i] + Math.log(this.weights[k])))); },
    score(X) { return ML.mean(this.scoreSamples(X)); },
    nParameters() {
      const d = this.means[0].length, K = this.means.length, t = this.params.covarianceType;
      const cov = t === 'full' ? (K * d * (d + 1)) / 2 : t === 'diag' ? K * d : t === 'tied' ? (d * (d + 1)) / 2 : K;
      return cov + K * d + K - 1;
    },
    bic(X) { return -2 * this.score(X) * X.length + this.nParameters() * Math.log(X.length); },
    aic(X) { return -2 * this.score(X) * X.length + 2 * this.nParameters(); },
    fullCovariances() { const d = this.means[0].length; return this.covariances.map((c) => asFull(c, this.params.covarianceType, d)); },
  });

  function eStep(X, means, weights, covs, type) {
    const L = logGauss(X, means, covs, type), K = means.length;
    let ll = 0;
    const R = X.map((_, i) => {
      const r = ML.range(K).map((k) => L[k][i] + Math.log(weights[k])), z = ML.logsumexp(r);
      ll += z;
      return r.map((v) => Math.exp(v - z));
    });
    return { R, ll: ll / X.length };
  }
  function mStep(X, R, type, reg) {
    const n = X.length, d = X[0].length, K = R[0].length;
    const nk = ML.range(K).map((k) => R.reduce((s, r) => s + r[k], 0) + TINY);
    const means = ML.range(K).map((k) => { const m = new Array(d).fill(0); X.forEach((x, i) => { for (let j = 0; j < d; j++) m[j] += R[i][k] * x[j]; }); return m.map((v) => v / nk[k]); });
    const covs = ML.range(K).map((k) => {
      const m = means[k];
      if (type === 'full') {
        const C = ML.la.zeros(d, d);
        X.forEach((x, i) => { const r = R[i][k]; for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += r * (x[a] - m[a]) * (x[b] - m[b]); });
        return C.map((row, a) => row.map((v, b) => v / nk[k] + (a === b ? reg : 0)));
      }
      // diagonal: E[x^2] - 2 mean E[x] + mean^2, as the reference computes it
      const v = new Array(d).fill(0);
      X.forEach((x, i) => { for (let j = 0; j < d; j++) v[j] += R[i][k] * x[j] * x[j]; });
      const dv = v.map((s, j) => s / nk[k] - 2 * m[j] * m[j] + m[j] * m[j] + reg);
      return type === 'diag' ? dv : ML.mean(dv);
    });
    return { means, weights: nk.map((v) => v / n), covs };
  }
})();
