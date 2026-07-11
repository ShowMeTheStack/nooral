// Logistic regression, binary and multinomial (softmax). The objective is
//   C * sum_i logloss_i + 0.5 * ||W||^2   (the intercept is not penalised)
// solved exactly by Newton's method (IRLS) with a line search, or by plain gradient descent with traces.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x), fv = (v) => ML.fmtv(v);

  // parameters are a flat vector: for each output k, d weights then one intercept
  function scores(theta, X, K, d, fitB) {
    const st = d + 1;
    return X.map((r) => {
      const z = new Array(K);
      for (let k = 0; k < K; k++) { let s = fitB ? theta[k * st + d] : 0; for (let j = 0; j < d; j++) s += theta[k * st + j] * r[j]; z[k] = s; }
      return z;
    });
  }
  // value, gradient and (optionally) Hessian of the objective
  function objective(theta, X, T, K, d, C0, lam, fitB, wantH, binary, sw) {
    const n = X.length, st = d + 1, P = theta.length, Z = scores(theta, X, K, d, fitB);
    let L = 0;
    const g = new Array(P).fill(0), H = wantH ? ML.la.zeros(P, P) : null;
    const probs = [];
    for (let i = 0; i < n; i++) {
      const xi = X[i], C = sw ? C0 * sw[i] : C0;
      let p;
      if (binary) {
        const z = Z[i][0], y = T[i];
        L += C * (Math.max(z, 0) - y * z + Math.log1p(Math.exp(-Math.abs(z))));
        p = [ML.sigmoid(z)];
        const e = p[0] - y;
        for (let j = 0; j < d; j++) g[j] += C * e * xi[j];
        if (fitB) g[d] += C * e;
        if (wantH) {
          const h = C * p[0] * (1 - p[0]);
          for (let a = 0; a <= d; a++) { const xa = a < d ? xi[a] : 1; if (a === d && !fitB) continue; for (let b = 0; b <= d; b++) { if (b === d && !fitB) continue; H[a][b] += h * xa * (b < d ? xi[b] : 1); } }
        }
      } else {
        const lse = ML.logsumexp(Z[i]);
        p = Z[i].map((v) => Math.exp(v - lse));
        L += C * (lse - Z[i][T[i]]);
        for (let k = 0; k < K; k++) {
          const e = C * (p[k] - (T[i] === k ? 1 : 0));
          for (let j = 0; j < d; j++) g[k * st + j] += e * xi[j];
          if (fitB) g[k * st + d] += e;
        }
        if (wantH) {
          for (let k = 0; k < K; k++) for (let l = 0; l < K; l++) {
            const h = C * p[k] * ((k === l ? 1 : 0) - p[l]);
            if (h === 0) continue;
            for (let a = 0; a <= d; a++) {
              if (a === d && !fitB) continue;
              const xa = a < d ? xi[a] : 1, row = H[k * st + a];
              for (let b = 0; b <= d; b++) { if (b === d && !fitB) continue; row[l * st + b] += h * xa * (b < d ? xi[b] : 1); }
            }
          }
        }
      }
      probs.push(p);
    }
    for (let k = 0; k < K; k++) for (let j = 0; j < d; j++) {
      const t = theta[k * st + j];
      L += 0.5 * lam * t * t; g[k * st + j] += lam * t;
      if (wantH) H[k * st + j][k * st + j] += lam;
    }
    if (wantH) {
      // fixed directions (unused intercepts, and the common shift of all softmax intercepts) get a unit
      // curvature so the Newton step is defined and leaves them alone
      for (let k = 0; k < K; k++) if (!fitB) H[k * st + d][k * st + d] = 1;
      if (!binary && fitB) for (let k = 0; k < K; k++) for (let l = 0; l < K; l++) H[k * st + d][l * st + d] += 1 / K;
    }
    return { L, g, H, probs };
  }

  ML.model('LogisticRegression', {
    kind: 'classifier',
    defaults: { C: 1, classWeight: null, sampleWeight: null, penalty: 'l2', solver: 'newton', fitIntercept: true, tol: 1e-10, maxIter: 100, lr: 0.1, epochs: 1000, traceEvery: 10 },
    fit(X, y, tr) {
      const p = this.params, { classes, idx } = ML.encode(y), d = X[0].length;
      const binary = classes.length === 2, K = binary ? 1 : classes.length, st = d + 1;
      const lam = p.penalty === 'none' || p.penalty == null ? 0 : 1, fitB = p.fitIntercept;
      this.classes = classes;
      // class weights and row weights scale each row's share of the loss (scikit-learn's class_weight, sample_weight)
      const CW = ML.classWeights(p.classWeight, classes, idx);
      const sw = CW || p.sampleWeight ? idx.map((k, i) => (CW ? CW[k] : 1) * (p.sampleWeight ? p.sampleWeight[i] : 1)) : null;
      if (CW) tr.add('weights', 'Class weights ' + classes.map((c, k) => c + ': ' + f(CW[k])).join(', ') + ' scale each row\'s share of the loss.', { classWeights: CW });
      let theta = new Array(K * st).fill(0);
      const obj = (t, h) => objective(t, X, idx, K, d, p.C, lam, fitB, h, binary, sw);
      this.history = [];
      if (p.solver === 'gd') {
        // gradient descent on the mean loss (the same minimiser: divide the objective by C n)
        const n = X.length, scale = 1 / (p.C * n);
        for (let e = 0; e <= p.epochs; e++) {
          const o = obj(theta, false);
          this.history.push(o.L * scale);
          if (e % p.traceEvery === 0 || e === p.epochs)
            tr.add('gd', 'Step ' + e + ': mean loss ' + f(o.L * scale) + ', gradient norm ' + f(ML.la.norm(o.g) * scale) + '.', { iter: e, loss: o.L * scale, theta: theta.slice(), gnorm: ML.la.norm(o.g) * scale });
          if (e === p.epochs) break;
          theta = theta.map((t, i) => t - p.lr * o.g[i] * scale);
        }
      } else {
        let o = obj(theta, true);
        for (let it = 0; it < p.maxIter; it++) {
          const gn = ML.la.norm(o.g);
          this.history.push(o.L);
          tr.add('newton', 'Newton step ' + it + ': objective ' + f(o.L) + ', gradient norm ' + f(gn) + '.', { iter: it, loss: o.L, gnorm: gn, theta: theta.slice() });
          if (gn < p.tol * Math.max(1, X.length)) break;
          const step = ML.la.solve(o.H, o.g.map((v) => -v));
          let t = 1, next, cand;
          for (let ls = 0; ls < 40; ls++) {
            cand = theta.map((v, i) => v + t * step[i]);
            next = obj(cand, true);
            if (next.L <= o.L + 1e-4 * t * ML.la.dot(o.g, step) || Math.abs(next.L - o.L) < 1e-14 * Math.abs(o.L)) break;
            t /= 2;
          }
          if (next.L > o.L) break;
          theta = cand; o = next;
          this.nIter = it + 1;
        }
      }
      if (!binary && fitB) { const m = ML.mean(ML.range(K).map((k) => theta[k * st + d])); for (let k = 0; k < K; k++) theta[k * st + d] -= m; }
      this.coef = ML.range(K).map((k) => theta.slice(k * st, k * st + d));
      this.intercept = ML.range(K).map((k) => (fitB ? theta[k * st + d] : 0));
      tr.add('done', 'Weights ' + this.coef.map(fv).join(' ; ') + ', intercepts ' + fv(this.intercept) + '.', { coef: this.coef, intercept: this.intercept });
    },
    decisionFunction(X) {
      const Z = X.map((r) => this.coef.map((w, k) => ML.la.dot(w, r) + this.intercept[k]));
      return this.classes.length === 2 ? Z.map((z) => z[0]) : Z;
    },
    predictProba(X) {
      const Z = this.decisionFunction(X);
      return this.classes.length === 2 ? Z.map((z) => { const q = ML.sigmoid(z); return [1 - q, q]; }) : Z.map(ML.softmax);
    },
    predict(X) { return this._argmaxPredict(X); },
  });
})();
