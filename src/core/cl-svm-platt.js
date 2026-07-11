// Probabilities for the support vector classifier (scikit-learn's probability=True, after libsvm): for each pair of
// classes, decision values of rows the machine did not train on (five folds) are fitted with a sigmoid
// P = 1 / (1 + exp(A·f + B)) (Platt scaling, with the smoothed targets and the guarded Newton steps libsvm uses); with
// more than two classes the pairwise probabilities are joined into one distribution by libsvm's pairwise coupling. The
// folds are drawn from a seed, so the numbers are close to scikit-learn's, not equal to them.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});

  /** fit A, B to decision values f with labels t (true for the positive side) */
  ML.plattFit = function (f, t) {
    const n = f.length, p1 = t.filter(Boolean).length, p0 = n - p1;
    const hi = (p1 + 1) / (p1 + 2), lo = 1 / (p0 + 2), T = t.map((v) => (v ? hi : lo));
    let A = 0, B = Math.log((p0 + 1) / (p1 + 1));
    const obj = (a, b) => { let s = 0; for (let i = 0; i < n; i++) { const z = f[i] * a + b; s += z >= 0 ? T[i] * z + Math.log1p(Math.exp(-z)) : (T[i] - 1) * z + Math.log1p(Math.exp(z)); } return s; };
    let fv = obj(A, B);
    for (let it = 0; it < 100; it++) {
      let h11 = 1e-12, h22 = 1e-12, h21 = 0, g1 = 0, g2 = 0;
      for (let i = 0; i < n; i++) {
        const z = f[i] * A + B, p = z >= 0 ? Math.exp(-z) / (1 + Math.exp(-z)) : 1 / (1 + Math.exp(z)), q = 1 - p, d2 = p * q, d1 = T[i] - p;
        h11 += f[i] * f[i] * d2; h22 += d2; h21 += f[i] * d2; g1 += f[i] * d1; g2 += d1;
      }
      if (Math.abs(g1) < 1e-5 && Math.abs(g2) < 1e-5) break;
      const det = h11 * h22 - h21 * h21, dA = -(h22 * g1 - h21 * g2) / det, dB = -(-h21 * g1 + h11 * g2) / det, gd = g1 * dA + g2 * dB;
      let step = 1, ok = false;
      while (step >= 1e-10) {
        const nA = A + step * dA, nB = B + step * dB, nf = obj(nA, nB);
        if (nf < fv + 1e-4 * step * gd) { A = nA; B = nB; fv = nf; ok = true; break; }
        step /= 2;
      }
      if (!ok) break;
    }
    return { A, B };
  };
  ML.plattProb = (f, ab) => { const z = f * ab.A + ab.B; return z >= 0 ? Math.exp(-z) / (1 + Math.exp(-z)) : 1 / (1 + Math.exp(z)); };

  /** pairwise probabilities r[i][j] = P(i | i or j) joined into one distribution over k classes */
  ML.coupleProbabilities = function (r) {
    const k = r.length, Q = Array.from({ length: k }, () => new Array(k).fill(0)), p = new Array(k).fill(1 / k), Qp = new Array(k).fill(0);
    for (let t = 0; t < k; t++) for (let j = 0; j < k; j++) if (j !== t) { Q[t][t] += r[j][t] * r[j][t]; Q[t][j] = -r[j][t] * r[t][j]; }
    const eps = 0.005 / k;
    for (let it = 0; it < Math.max(100, k); it++) {
      let pQp = 0;
      for (let t = 0; t < k; t++) { Qp[t] = 0; for (let j = 0; j < k; j++) Qp[t] += Q[t][j] * p[j]; pQp += p[t] * Qp[t]; }
      if (Math.max(...Qp.map((v) => Math.abs(v - pQp))) < eps) break;
      for (let t = 0; t < k; t++) {
        const diff = (-Qp[t] + pQp) / Q[t][t];
        p[t] += diff;
        pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
        for (let j = 0; j < k; j++) { Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff); p[j] /= 1 + diff; }
      }
    }
    return p;
  };

  /**
   * give a fitted SVC its probabilities: for every machine (pair of classes) a sigmoid fitted on decision values of
   * held-out rows, five folds; sets svc.platt and svc.predictProba
   */
  ML.svcPlatt = function (svc, X, y, folds) {
    if (svc.machines.some((m) => !m.pair)) throw new Error('probabilities need the one-against-one machines (multiClass: ovo)');
    const nf = folds || 5, p = svc.params, rng = ML.rng(p.seed == null ? 0 : p.seed), classes = svc.classes;
    const idx = y.map((v) => classes.indexOf(v));
    svc.platt = svc.machines.map((m) => {
      const [a, b] = m.pair, rows = ML.range(X.length).filter((i) => idx[i] === a || idx[i] === b);
      const perm = rng.shuffle(rows.slice()), dec = new Array(perm.length).fill(0);
      for (let k = 0; k < nf; k++) {
        const lo = Math.floor((k * perm.length) / nf), hi = Math.floor(((k + 1) * perm.length) / nf);
        const test = perm.slice(lo, hi), train = perm.slice(0, lo).concat(perm.slice(hi));
        const pa = train.filter((i) => idx[i] === a).length, pb = train.length - pa;
        if (!pa || !pb) { for (let t = lo; t < hi; t++) dec[t] = pa ? 1 : pb ? -1 : 0; continue; }
        const sub = svc.clone({ probability: false, trace: false }).fit(train.map((i) => X[i]), train.map((i) => y[i]));
        // the sub-machine's first class is a (classes keep their order), so a positive value favours a, as in m
        const v = sub.machineValues(test.map((i) => X[i]));
        for (let t = lo; t < hi; t++) dec[t] = v[t - lo][0];
      }
      return ML.plattFit(dec, perm.map((i) => idx[i] === a));
    });
    svc.predictProba = function (Xn) {
      const V = this.machineValues(Xn), K = this.classes.length;
      return V.map((v) => {
        if (K === 2) { const q = Math.min(Math.max(ML.plattProb(v[0], this.platt[0]), 1e-7), 1 - 1e-7); return [q, 1 - q]; }
        const r = Array.from({ length: K }, () => new Array(K).fill(0));
        this.machines.forEach((m, t) => { const q = Math.min(Math.max(ML.plattProb(v[t], this.platt[t]), 1e-7), 1 - 1e-7); r[m.pair[0]][m.pair[1]] = q; r[m.pair[1]][m.pair[0]] = 1 - q; });
        return ML.coupleProbabilities(r);
      });
    };
    return svc;
  };
})();
