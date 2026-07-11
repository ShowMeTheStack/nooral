// Principal component analysis by the SVD of the centred data (or the eigen-decomposition of the
// covariance), with explained variance, projection, reconstruction and optional whitening.
(function () {
  const S = (globalThis.Nooral = globalThis.Nooral || {});
  const ML = (S.ML = S.ML || {});
  const f = (x) => ML.fmt(x);

  // make each component's largest-magnitude entry positive, so signs are reproducible
  ML.signNormalise = (rows) => rows.map((r) => { let b = 0; r.forEach((v, j) => { if (Math.abs(v) > Math.abs(r[b])) b = j; }); return r[b] < 0 ? r.map((v) => -v) : r.slice(); });

  ML.PCA = function (params) {
    const p = Object.assign({ nComponents: null, solver: 'svd', whiten: false, trace: true }, params || {});
    const tr = ML.tracer(p);
    const o = {
      params: p,
      fit(X) {
        const n = X.length, d = X[0].length;
        o.mean = ML.colMeans(X);
        const Xc = ML.center(X, o.mean);
        tr.add('centre', 'Subtract the column means ' + ML.fmtv(o.mean) + ' so every feature is centred at zero.', { mean: o.mean });
        let comps, vars;
        if (p.solver === 'eigen') {
          const C = ML.la.cov(X, 1), e = ML.la.symEig(C);
          tr.add('cov', 'The ' + d + ' x ' + d + ' covariance matrix is decomposed into eigenvectors and eigenvalues.', { cov: C });
          comps = ML.la.T(e.vectors); vars = e.values.map((v) => Math.max(v, 0));
        } else {
          const r = ML.la.svd(Xc);
          comps = ML.la.T(r.V); vars = r.s.map((s) => (s * s) / (n - 1));
          o.singularValues = r.s;
          tr.add('svd', 'The SVD of the centred data gives singular values ' + ML.fmtv(r.s) + '.', { s: r.s });
        }
        comps = ML.signNormalise(comps);
        const total = ML.sum(vars);
        let k = p.nComponents == null ? Math.min(n, d) : p.nComponents;
        if (k > 0 && k < 1) { let c = 0, i = 0; while (i < vars.length && c / total < k) c += vars[i++]; k = i; }
        o.components = comps.slice(0, k);
        o.explainedVariance = vars.slice(0, k);
        o.explainedVarianceRatio = o.explainedVariance.map((v) => v / total);
        if (o.singularValues) o.singularValues = o.singularValues.slice(0, k);
        o.nComponents = k;
        o.noiseVariance = k < Math.min(n, d) ? ML.mean(vars.slice(k)) : 0;
        tr.add('keep', 'Keeping ' + k + ' components explains ' + f(100 * ML.sum(o.explainedVarianceRatio)) + '% of the variance.', { ratio: o.explainedVarianceRatio, components: o.components });
        o.trace = tr.steps;
        return o;
      },
      transform(X) {
        return X.map((x) => o.components.map((c, k) => { const v = c.reduce((s, w, j) => s + w * (x[j] - o.mean[j]), 0); return p.whiten ? v / Math.sqrt(o.explainedVariance[k]) : v; }));
      },
      fitTransform(X) { return o.fit(X).transform(X); },
      inverseTransform(Z) {
        return Z.map((z) => o.mean.map((m, j) => m + o.components.reduce((s, c, k) => s + (p.whiten ? z[k] * Math.sqrt(o.explainedVariance[k]) : z[k]) * c[j], 0)));
      },
      // squared reconstruction error per row, for the "what is lost" picture
      reconstructionError(X) { const R = o.inverseTransform(o.transform(X)); return X.map((x, i) => ML.sqdist(x, R[i])); },
    };
    return o;
  };
})();
