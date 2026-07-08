// Linear regression (normal equations through QR, or gradient descent), polynomial features, ridge,
// lasso and elastic net by coordinate descent with the regularisation path, and the perceptron.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x), fv = (v) => ML.fmtv(v);

  // centre X and y so the intercept can be recovered afterwards
  function centred(X, y, fitIntercept) {
    const d = X[0].length;
    if (!fitIntercept) return { Xc: X, yc: y, xm: new Array(d).fill(0), ym: 0 };
    const xm = ML.colMeans(X), ym = ML.mean(y);
    return { Xc: ML.center(X, xm), yc: y.map((v) => v - ym), xm, ym };
  }
  const linPredict = function (X) { return X.map((r) => ML.la.dot(r, this.coef) + this.intercept); };

  ML.model('LinearRegression', {
    kind: 'regressor',
    defaults: { solver: 'qr', fitIntercept: true, lr: 0.05, epochs: 500, traceEvery: 1, seed: 0 },
    fit(X, y, tr) {
      const p = this.params;
      if (p.solver === 'gd') return gdFit.call(this, X, y, tr);
      const { Xc, yc, xm, ym } = centred(X, y, p.fitIntercept);
      tr.add('design', 'The design matrix has ' + X.length + ' rows and ' + X[0].length + ' columns; we centre each column and the target.', { means: xm, ymean: ym });
      if (!X.length) throw new Error('LinearRegression needs at least one row');
      delete this.rank;
      let x;
      const n = Xc.length, d = Xc[0].length;
      const qr = n >= d ? ML.la.lstsq(Xc, yc) : null;
      const deficient = !qr || qr.R.some((r, i) => Math.abs(r[i]) < 1e-12 * Math.abs(qr.R[0][0] || 1));
      if (!deficient) {
        x = qr.x;
        tr.add('qr', 'Householder QR turns X into Q R; R is upper-triangular with diagonal ' + fv(qr.R.map((r, i) => r[i])) + '.', { R: qr.R });
      } else {
        // more columns than rows (or columns that depend on each other): many weights fit equally well; as
        // scikit-learn does, take the shortest of them, from the SVD
        const m = ML.la.lstsqMinNorm(Xc, yc);
        x = m.x;
        this.rank = m.rank;
        tr.add('svd', (n < d ? 'There are ' + n + ' rows but ' + d + ' columns, so' : 'Some columns depend on others, so') + ' many weight vectors fit equally well (rank ' + m.rank + '); the SVD gives the one of least length.', { s: m.s, rank: m.rank });
      }
      this.coef = x;
      this.intercept = p.fitIntercept ? ym - ML.la.dot(xm, x) : 0;
      const pred = linPredict.call(this, X);
      this.sse = y.reduce((s, v, i) => s + (v - pred[i]) ** 2, 0);
      tr.add('solve', (this.rank !== undefined ? 'The least-norm solution is w = ' : 'Back-substitution in R w = Q^T y gives w = ') + fv(x) + ' and intercept ' + f(this.intercept) + '; squared error ' + f(this.sse) + '.', { coef: x.slice(), intercept: this.intercept, sse: this.sse });
    },
    predict: linPredict,
  });

  // full-batch gradient descent on half the mean squared error
  function gdFit(X, y, tr) {
    const p = this.params, n = X.length, d = X[0].length;
    let w = new Array(d).fill(0), b = 0;
    this.history = [];
    for (let e = 0; e <= p.epochs; e++) {
      const g = new Array(d).fill(0); let gb = 0, loss = 0;
      for (let i = 0; i < n; i++) {
        const r = ML.la.dot(X[i], w) + (p.fitIntercept ? b : 0) - y[i];
        loss += r * r; gb += r;
        for (let j = 0; j < d; j++) g[j] += r * X[i][j];
      }
      loss /= 2 * n;
      this.history.push(loss);
      if (e % p.traceEvery === 0 || e === p.epochs)
        tr.add('gd', 'Epoch ' + e + ': loss ' + f(loss) + ', w = ' + fv(w) + ', b = ' + f(b) + '.', { epoch: e, loss, w: w.slice(), b, grad: g.map((v) => v / n) });
      if (e === p.epochs) break;
      for (let j = 0; j < d; j++) w[j] -= (p.lr * g[j]) / n;
      if (p.fitIntercept) b -= (p.lr * gb) / n;
    }
    this.coef = w; this.intercept = b;
  }

  // polynomial features in the usual order: 1, x0, x1, x0^2, x0 x1, x1^2, ...
  ML.PolynomialFeatures = function (params) {
    const p = Object.assign({ degree: 2, includeBias: true, interactionOnly: false }, params || {});
    const obj = {
      params: p,
      fit(X) {
        const d = X[0].length, combos = p.includeBias ? [[]] : [];
        const rec = (start, deg, cur) => {
          if (cur.length === deg) { combos.push(cur.slice()); return; }
          for (let j = start; j < d; j++) { cur.push(j); rec(p.interactionOnly ? j + 1 : j, deg, cur); cur.pop(); }
        };
        for (let k = 1; k <= p.degree; k++) rec(0, k, []);
        obj.combos = combos;
        obj.names = combos.map((c) => (c.length ? c.map((j) => 'x' + j).join(' ') : '1').replace(/(x\d+)(?: \1)+/g, (m, a) => a + '^' + m.split(' ').length));
        return obj;
      },
      transform(X) { return X.map((r) => obj.combos.map((c) => c.reduce((s, j) => s * r[j], 1))); },
      fitTransform(X) { return obj.fit(X).transform(X); },
    };
    return obj;
  };

  ML.model('Ridge', {
    kind: 'regressor',
    defaults: { alpha: 1, fitIntercept: true },
    fit(X, y, tr) {
      const p = this.params, { Xc, yc, xm, ym } = centred(X, y, p.fitIntercept);
      const G = ML.la.gram(Xc), d = G.length;
      for (let j = 0; j < d; j++) G[j][j] += p.alpha;
      const Xty = new Array(d).fill(0);
      Xc.forEach((r, i) => { for (let j = 0; j < d; j++) Xty[j] += r[j] * yc[i]; });
      tr.add('normal', 'Adding alpha = ' + f(p.alpha) + ' to the diagonal of X^T X makes it better conditioned.', { gram: G });
      this.coef = ML.la.solve(G, Xty);
      this.intercept = p.fitIntercept ? ym - ML.la.dot(xm, this.coef) : 0;
      tr.add('solve', 'Solving (X^T X + alpha I) w = X^T y gives w = ' + fv(this.coef) + ', intercept ' + f(this.intercept) + '.', { coef: this.coef.slice(), intercept: this.intercept });
    },
    predict: linPredict,
  });

  const soft = (z, g) => (z > g ? z - g : z < -g ? z + g : 0);

  // coordinate descent for (1/2n)||y - Xw||^2 + a r ||w||_1 + (a (1 - r) / 2) ||w||^2
  ML.cdSolve = function (Xc, yc, alpha, l1Ratio, opts) {
    opts = opts || {};
    const n = Xc.length, d = Xc[0].length, tol = opts.tol == null ? 1e-10 : opts.tol, maxIter = opts.maxIter || 100000;
    const w = opts.w0 ? opts.w0.slice() : new Array(d).fill(0), tr = opts.tr;
    const cols = ML.range(d).map((j) => Xc.map((r) => r[j])), sq = cols.map((c) => ML.la.dot(c, c));
    const r = yc.map((v, i) => v - ML.la.dot(Xc[i], w));
    const l1 = alpha * l1Ratio * n, l2 = alpha * (1 - l1Ratio) * n;
    let it = 0;
    for (; it < maxIter; it++) {
      let maxDelta = 0, maxW = 0;
      for (let j = 0; j < d; j++) {
        if (sq[j] === 0) continue;
        const c = cols[j], old = w[j];
        let rho = 0;
        for (let i = 0; i < n; i++) rho += c[i] * r[i];
        rho += sq[j] * old;
        const nw = soft(rho, l1) / (sq[j] + l2);
        if (nw !== old) { const dw = nw - old; for (let i = 0; i < n; i++) r[i] -= dw * c[i]; w[j] = nw; }
        if (tr && it < (opts.traceSweeps || 20))
          tr.add('coord', 'Sweep ' + (it + 1) + ', w' + j + ': rho = ' + f(rho) + ', soft-threshold at ' + f(l1) + ' gives ' + f(nw) + '.', { sweep: it + 1, j, rho, w: w.slice(), zeroed: nw === 0 });
        maxDelta = Math.max(maxDelta, Math.abs(nw - old)); maxW = Math.max(maxW, Math.abs(nw));
      }
      if (maxDelta <= tol * Math.max(maxW, 1e-300) || maxW === 0) { it++; break; }
    }
    return { w, iters: it };
  };

  function enetModel(name, l1Default) {
    ML.model(name, {
      kind: 'regressor',
      defaults: { alpha: 1, l1Ratio: l1Default, fitIntercept: true, tol: 1e-10, maxIter: 100000 },
      fit(X, y, tr) {
        const p = this.params, { Xc, yc, xm, ym } = centred(X, y, p.fitIntercept);
        const res = ML.cdSolve(Xc, yc, p.alpha, p.l1Ratio, { tol: p.tol, maxIter: p.maxIter, tr });
        this.coef = res.w; this.nIter = res.iters;
        this.intercept = p.fitIntercept ? ym - ML.la.dot(xm, res.w) : 0;
        const zeros = res.w.filter((v) => v === 0).length;
        tr.add('done', 'After ' + res.iters + ' sweeps w = ' + fv(res.w) + '; ' + zeros + ' of ' + res.w.length + ' coefficients are exactly zero.', { coef: res.w.slice(), intercept: this.intercept });
      },
      predict: linPredict,
    });
  }
  enetModel('Lasso', 1);
  enetModel('ElasticNet', 0.5);

  // the regularisation path, strongest alpha first, warm-started from the previous solution
  ML.enetPath = function (X, y, opts) {
    const p = Object.assign({ l1Ratio: 1, nAlphas: 100, eps: 1e-3, fitIntercept: true, tol: 1e-10 }, opts || {});
    const { Xc, yc } = centred(X, y, p.fitIntercept), n = X.length, d = X[0].length;
    let alphas = p.alphas;
    if (!alphas) {
      let m = 0;
      for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += Xc[i][j] * yc[i]; m = Math.max(m, Math.abs(s)); }
      const amax = m / (n * p.l1Ratio);
      alphas = ML.range(p.nAlphas).map((k) => amax * Math.pow(p.eps, k / (p.nAlphas - 1)));
    }
    let w = new Array(d).fill(0);
    const coefs = alphas.map((a) => { w = ML.cdSolve(Xc, yc, a, p.l1Ratio, { tol: p.tol, w0: w }).w; return w.slice(); });
    const steps = alphas.map((a, k) => ({ kind: 'path', say: 'At alpha = ' + f(a) + ', ' + coefs[k].filter((v) => v !== 0).length + ' coefficients are non-zero.', alpha: a, coef: coefs[k] }));
    return { alphas, coefs, trace: steps };
  };

  // the perceptron: one pass over the samples in order per epoch, updating on every mistake
  ML.model('Perceptron', {
    kind: 'classifier',
    defaults: { eta: 1, epochs: 5, shuffle: false, seed: 0, fitIntercept: true },
    fit(X, y, tr) {
      const p = this.params, { classes, idx } = ML.encode(y), d = X[0].length, rng = ML.rng(p.seed);
      this.classes = classes;
      const heads = classes.length === 2 ? [1] : ML.range(classes.length);
      this.W = []; this.b = [];
      for (const k of heads) {
        const t = idx.map((c) => (c === k ? 1 : -1)), w = new Array(d).fill(0);
        let b = 0;
        for (let e = 0; e < p.epochs; e++) {
          const order = p.shuffle ? rng.perm(X.length) : ML.range(X.length);
          let mistakes = 0;
          for (const i of order) {
            const z = ML.la.dot(w, X[i]) + b;
            if (t[i] * z <= 0) {
              mistakes++;
              for (let j = 0; j < d; j++) w[j] += p.eta * t[i] * X[i][j];
              if (p.fitIntercept) b += p.eta * t[i];
              tr.add('update', 'Sample ' + i + ' (label ' + t[i] + ') scored ' + f(z) + ', a mistake: w becomes ' + fv(w) + ', b = ' + f(b) + '.', { head: k, epoch: e, i, z, w: w.slice(), b });
            }
          }
          tr.add('epoch', 'Epoch ' + (e + 1) + ' made ' + mistakes + ' mistakes.', { head: k, epoch: e, mistakes });
          if (!mistakes) break;
        }
        this.W.push(w); this.b.push(b);
      }
      this.coef = this.W; this.intercept = this.b;
    },
    decisionFunction(X) {
      const s = X.map((r) => this.W.map((w, k) => ML.la.dot(w, r) + this.b[k]));
      return this.classes.length === 2 ? s.map((v) => v[0]) : s;
    },
    predict(X) {
      const s = this.decisionFunction(X);
      return this.classes.length === 2 ? s.map((v) => this.classes[v > 0 ? 1 : 0]) : s.map((v) => this.classes[ML.argmax(v)]);
    },
  });
})();
