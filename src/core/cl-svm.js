// Support vector machines solved by SMO with second-order working-set selection: C-SVC (linear,
// polynomial, RBF and sigmoid kernels; one-vs-one or one-vs-rest for many classes), epsilon-SVR and the
// one-class SVM. Every step of the solver (the chosen pair and how their alphas moved) is traced.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);
  const TAU = 1e-12;

  // kernel functions; gamma 'scale' = 1 / (d * variance of all entries), 'auto' = 1 / d
  ML.kernelGamma = function (gamma, X) {
    if (typeof gamma === 'number') return gamma;
    const d = X[0].length;
    if (gamma === 'auto') return 1 / d;
    let s = 0, ss = 0, n = 0;
    for (const r of X) for (const v of r) { s += v; ss += v * v; n++; }
    const v = ss / n - (s / n) ** 2;
    return v > 0 ? 1 / (d * v) : 1;
  };
  ML.kernel = function (p, gamma) {
    const k = p.kernel || 'rbf';
    if (k === 'linear') return (a, b) => ML.la.dot(a, b);
    if (k === 'poly') return (a, b) => Math.pow(gamma * ML.la.dot(a, b) + p.coef0, p.degree);
    if (k === 'sigmoid') return (a, b) => Math.tanh(gamma * ML.la.dot(a, b) + p.coef0);
    return (a, b) => Math.exp(-gamma * ML.sqdist(a, b));
  };

  // min 0.5 a'Qa + p'a  s.t. y'a = const, 0 <= a_i <= C_i, with Q_ij = y_i y_j K_ij.
  // K(i, j) gives kernel values of the (possibly doubled) variables. Returns alpha, rho and the steps.
  ML.smo = function (n, K, y, p, C, opts) {
    const eps = opts.tol, maxIter = opts.maxIter || 10000000, tr = opts.tr;
    const a = opts.alpha0 ? opts.alpha0.slice() : new Array(n).fill(0);
    const QD = ML.range(n).map((i) => K(i, i));
    const Q = (i, j) => y[i] * y[j] * K(i, j);
    const rowCache = new Map();
    const row = (i) => { let r = rowCache.get(i); if (!r) { r = new Float64Array(n); for (let t = 0; t < n; t++) r[t] = Q(i, t); rowCache.set(i, r); } return r; };
    const G = p.slice();
    for (let i = 0; i < n; i++) if (a[i] !== 0) { const Qi = row(i); for (let t = 0; t < n; t++) G[t] += a[i] * Qi[t]; }
    const up = (t) => (y[t] === 1 ? a[t] < C[t] : a[t] > 0), low = (t) => (y[t] === 1 ? a[t] > 0 : a[t] < C[t]);
    let iter = 0;
    for (; iter < maxIter; iter++) {
      let Gmax = -Infinity, i = -1;
      // ties go to the later index, as in the reference solver
      for (let t = 0; t < n; t++) if (up(t) && -y[t] * G[t] >= Gmax) { Gmax = -y[t] * G[t]; i = t; }
      if (i < 0) break;
      const Qi = row(i);
      let Gmax2 = -Infinity, j = -1, best = Infinity;
      for (let t = 0; t < n; t++) {
        if (!low(t)) continue;
        if (y[t] * G[t] > Gmax2) Gmax2 = y[t] * G[t];
        const b = Gmax + y[t] * G[t];
        if (b > 0) {
          let q = QD[i] + QD[t] - 2 * y[i] * y[t] * Qi[t];
          if (q <= 0) q = TAU;
          const ob = -(b * b) / q;
          if (ob <= best) { best = ob; j = t; }
        }
      }
      if (Gmax + Gmax2 < eps || j < 0) break;
      const Qj = row(j), Ci = C[i], Cj = C[j], oi = a[i], oj = a[j];
      if (y[i] !== y[j]) {
        let q = QD[i] + QD[j] + 2 * Qi[j]; if (q <= 0) q = TAU;
        const delta = (-G[i] - G[j]) / q, diff = a[i] - a[j];
        a[i] += delta; a[j] += delta;
        if (diff > 0) { if (a[j] < 0) { a[j] = 0; a[i] = diff; } } else if (a[i] < 0) { a[i] = 0; a[j] = -diff; }
        if (diff > Ci - Cj) { if (a[i] > Ci) { a[i] = Ci; a[j] = Ci - diff; } } else if (a[j] > Cj) { a[j] = Cj; a[i] = Cj + diff; }
      } else {
        let q = QD[i] + QD[j] - 2 * Qi[j]; if (q <= 0) q = TAU;
        const delta = (G[i] - G[j]) / q, sum = a[i] + a[j];
        a[i] -= delta; a[j] += delta;
        if (sum > Ci) { if (a[i] > Ci) { a[i] = Ci; a[j] = sum - Ci; } } else if (a[j] < 0) { a[j] = 0; a[i] = sum; }
        if (sum > Cj) { if (a[j] > Cj) { a[j] = Cj; a[i] = sum - Cj; } } else if (a[i] < 0) { a[i] = 0; a[j] = sum; }
      }
      const di = a[i] - oi, dj = a[j] - oj;
      for (let t = 0; t < n; t++) G[t] += Qi[t] * di + Qj[t] * dj;
      if (tr) tr.add('pair', 'Step ' + (iter + 1) + ': the pair (' + i + ', ' + j + ') violates the optimality conditions most (gap ' + f(Gmax + Gmax2) + '); alpha_' + i + ' moves ' + f(oi) + ' -> ' + f(a[i]) + ', alpha_' + j + ' ' + f(oj) + ' -> ' + f(a[j]) + '.', { iter, i, j, ai: a[i], aj: a[j], gap: Gmax + Gmax2 });
    }
    // rho from the free variables, or the middle of the feasible interval
    let ub = Infinity, lb = -Infinity, nFree = 0, sFree = 0;
    for (let t = 0; t < n; t++) {
      const yG = y[t] * G[t];
      if (a[t] >= C[t]) { if (y[t] === -1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else if (a[t] <= 0) { if (y[t] === 1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else { nFree++; sFree += yG; }
    }
    const rho = nFree > 0 ? sFree / nFree : (ub + lb) / 2;
    let obj = 0;
    for (let t = 0; t < n; t++) obj += a[t] * (G[t] + p[t]);
    return { alpha: a, rho, iters: iter, objective: obj / 2 };
  };

  // one binary C-SVC between rows idx with labels +1 / -1
  function binarySVC(X, idx, yy, p, gamma, kf, tr, Cpos, Cneg) {
    const n = idx.length, cache = new Map();
    const K = (i, j) => { const key = i < j ? i * n + j : j * n + i; let v = cache.get(key); if (v === undefined) { v = kf(X[idx[i]], X[idx[j]]); cache.set(key, v); } return v; };
    const r = ML.smo(n, K, yy, new Array(n).fill(-1), yy.map((v) => (v === 1 ? Cpos : Cneg)), { tol: p.tol, maxIter: p.maxIter, tr });
    const sv = [], coef = [];
    r.alpha.forEach((al, t) => { if (al > 0) { sv.push(idx[t]); coef.push(al * yy[t]); } });
    return { sv, coef, rho: r.rho, iters: r.iters, objective: r.objective };
  }
  const machineValue = (m, X, kf, x) => { let s = -m.rho; for (let t = 0; t < m.sv.length; t++) s += m.coef[t] * kf(X[m.sv[t]], x); return s; };

  const svmDefaults = { C: 1, kernel: 'rbf', gamma: 'scale', degree: 3, coef0: 0, tol: 1e-3, maxIter: 10000000, multiClass: 'ovo', decisionShape: 'ovr', classWeight: null, probability: false, seed: 0 };
  ML.model('SVC', {
    kind: 'classifier', defaults: svmDefaults,
    fit(X, y, tr) {
      const p = this.params, { classes, idx } = ML.encode(y), K = classes.length;
      this.classes = classes; this.X = X; this.gammaValue = ML.kernelGamma(p.gamma, X);
      const kf = (this.kf = ML.kernel(p, this.gammaValue)), CW = ML.classWeights(p.classWeight, classes, idx), cw = (k) => p.C * (CW ? CW[k] : 1);
      this.machines = [];
      if (K === 2 || p.multiClass === 'ovo') {
        // pairs (a, b), a < b: class a is +1
        for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) {
          const rows = ML.range(X.length).filter((i) => idx[i] === a || idx[i] === b);
          const m = binarySVC(X, rows, rows.map((i) => (idx[i] === a ? 1 : -1)), p, this.gammaValue, kf, this.machines.length === 0 ? tr : null, cw(a), cw(b));
          m.pair = [a, b]; this.machines.push(m);
          tr.add('machine', 'Classes ' + classes[a] + ' vs ' + classes[b] + ': ' + m.sv.length + ' support vectors after ' + m.iters + ' SMO steps, rho = ' + f(m.rho) + '.', { pair: [a, b], sv: m.sv, coef: m.coef, rho: m.rho });
        }
      } else {
        for (let a = 0; a < K; a++) {
          const rows = ML.range(X.length);
          const m = binarySVC(X, rows, rows.map((i) => (idx[i] === a ? 1 : -1)), p, this.gammaValue, kf, a === 0 ? tr : null, cw(a), p.C);
          m.cls = a; this.machines.push(m);
          tr.add('machine', 'Class ' + classes[a] + ' against the rest: ' + m.sv.length + ' support vectors, rho = ' + f(m.rho) + '.', { cls: a, sv: m.sv, coef: m.coef, rho: m.rho });
        }
      }
      this.support = [...new Set(this.machines.flatMap((m) => m.sv))].sort((a, b) => idx[a] - idx[b] || a - b);
      this.nSupport = classes.map((_, k) => this.support.filter((i) => idx[i] === k).length);
      // probability: true adds Platt-scaled probabilities, fitted on held-out decision values (cl-svm-platt.js)
      if (p.probability) { ML.svcPlatt(this, X, y); tr.add('platt', 'Probabilities: a sigmoid per pair of classes, fitted on the decision values of rows held out in five folds.', { platt: this.platt }); }
      else delete this.predictProba;
    },
    // raw machine outputs: one per pair (positive favours the first class) or per class for one-vs-rest
    machineValues(X) { return X.map((x) => this.machines.map((m) => machineValue(m, this.X, this.kf, x))); },
    decisionFunction(X, shape) {
      const V = this.machineValues(X), K = this.classes.length;
      if (K === 2) return V.map((v) => -v[0]);
      if (this.params.multiClass === 'ovr') return V;
      if ((shape || this.params.decisionShape) === 'ovo') return V;
      return V.map((v) => {
        const votes = new Array(K).fill(0), conf = new Array(K).fill(0);
        this.machines.forEach((m, t) => { const [a, b] = m.pair; conf[a] += v[t]; conf[b] -= v[t]; votes[v[t] > 0 ? a : b]++; });
        return votes.map((vt, k) => vt + conf[k] / (3 * (Math.abs(conf[k]) + 1)));
      });
    },
    predict(X) {
      const V = this.machineValues(X), K = this.classes.length;
      if (K === 2) return V.map((v) => this.classes[v[0] > 0 ? 0 : 1]);
      if (this.params.multiClass === 'ovr') return V.map((v) => this.classes[ML.argmax(v)]);
      return V.map((v) => { const votes = new Array(K).fill(0); this.machines.forEach((m, t) => votes[v[t] > 0 ? m.pair[0] : m.pair[1]]++); return this.classes[ML.argmax(votes)]; });
    },
  });

  // epsilon-SVR: 2n variables (alpha, alpha*) and an epsilon-insensitive tube around the fit
  ML.model('SVR', {
    kind: 'regressor', defaults: Object.assign({}, svmDefaults, { epsilon: 0.1 }),
    fit(X, y, tr) {
      const p = this.params, n = X.length;
      this.X = X; this.gammaValue = ML.kernelGamma(p.gamma, X);
      const kf = (this.kf = ML.kernel(p, this.gammaValue)), Km = X.map((a) => X.map((b) => kf(a, b)));
      const yy = ML.range(2 * n).map((t) => (t < n ? 1 : -1)), lin = ML.range(2 * n).map((t) => (t < n ? p.epsilon - y[t] : p.epsilon + y[t - n]));
      const r = ML.smo(2 * n, (i, j) => Km[i % n][j % n], yy, lin, new Array(2 * n).fill(p.C), { tol: p.tol, maxIter: p.maxIter, tr });
      this.support = []; this.coef = [];
      for (let i = 0; i < n; i++) { const b = r.alpha[i] - r.alpha[i + n]; if (b !== 0) { this.support.push(i); this.coef.push(b); } }
      this.rho = r.rho; this.intercept = -r.rho;
      tr.add('done', this.support.length + ' of ' + n + ' points lie on or outside the tube of half-width ' + f(p.epsilon) + ' and become support vectors.', { support: this.support, coef: this.coef, rho: r.rho });
    },
    predict(X) { return X.map((x) => machineValue({ sv: this.support, coef: this.coef, rho: this.rho }, this.X, this.kf, x)); },
  });

  // one-class SVM: separates the data from the origin in feature space; nu bounds the outlier fraction
  ML.model('OneClassSVM', {
    kind: 'outlier', defaults: Object.assign({}, svmDefaults, { nu: 0.5 }),
    fit(X, y, tr) {
      const p = this.params, n = X.length;
      this.X = X; this.gammaValue = ML.kernelGamma(p.gamma, X);
      const kf = (this.kf = ML.kernel(p, this.gammaValue)), Km = X.map((a) => X.map((b) => kf(a, b)));
      const nn = Math.floor(p.nu * n), a0 = new Array(n).fill(0);
      for (let i = 0; i < nn; i++) a0[i] = 1;
      if (nn < n) a0[nn] = p.nu * n - nn;
      const r = ML.smo(n, (i, j) => Km[i][j], new Array(n).fill(1), new Array(n).fill(0), new Array(n).fill(1), { tol: p.tol, maxIter: p.maxIter, tr, alpha0: a0 });
      this.support = []; this.coef = [];
      r.alpha.forEach((al, i) => { if (al > 0) { this.support.push(i); this.coef.push(al); } });
      this.rho = r.rho;
      tr.add('done', this.support.length + ' support vectors; the boundary is where the kernel sum equals rho = ' + f(r.rho) + '.', { support: this.support, coef: this.coef, rho: r.rho });
    },
    decisionFunction(X) { return X.map((x) => machineValue({ sv: this.support, coef: this.coef, rho: this.rho }, this.X, this.kf, x)); },
    predict(X) { return this.decisionFunction(X).map((v) => (v >= 0 ? 1 : -1)); },
  });
})();
