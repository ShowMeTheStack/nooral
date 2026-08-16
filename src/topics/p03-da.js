// Part 3, linear and quadratic discriminant analysis side by side. Both fit a Gaussian to each class and classify by
// the larger posterior. LDA makes every class share one covariance (the same ellipse at every mean), so the log-ratio
// of two classes is linear: a straight boundary. QDA gives each class its own ellipse, and the boundary curves.
// Frames follow the engines' traces (ML.LDA: means, shared covariance, linear scores; ML.QDA: one step per class).
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n, SC = () => S.Kit.scatter;

  const DATA = [{ value: 'same', label: 'two classes, the same shape' }, { value: 'diff', label: 'two classes, different shapes' }, { value: 'inside', label: 'a narrow class inside a wide one' }, { value: 'flowers', label: 'meadow wildflowers (3 species)' }];
  H.daData = (kind, n, seed) => {
    if (kind === 'flowers') { const t = S.Data.load('wildflowers', { seed, n }); return { X: t.X.map((r) => [r[0], r[2]]), y: t.y.slice(), names: ['petal length (cm)', 'stem height (cm)'], classes: t.classes }; }
    const g = kind === 'same' ? H.gauss2(n, seed, [-1, -0.6], [1.3, 0.9], [1, 0.7, 0.5], [1, 0.7, 0.5])
      : kind === 'diff' ? H.gauss2(n, seed, [-1.2, 0], [1.2, 0.3], [0.5, 1.4, 0.2], [1.3, 0.5, -0.4])
        : H.gauss2(n, seed, [0, 0], [0.3, 0.2], [1.7, 1.4, 0.1], [0.55, 0.5, 0.3]);
    return { X: g.X, y: g.y, names: ['x₁', 'x₂'], classes: ['class 0', 'class 1'] };
  };

  S.topic({
    id: 'lda-qda', family: 'p03-da', order: 10, title: 'LDA and QDA: Gaussian classes',
    blurb: 'Model each class as a Gaussian cloud: a mean, a covariance (an ellipse) and a prior share. A new point goes to the class with the highest posterior, prior times density. If all classes share one covariance (LDA), the quadratic parts cancel between classes and the boundary is straight. If each class has its own (QDA), they do not cancel and the boundary is a conic: an ellipse, a parabola or a hyperbola. QDA is more flexible but has many more numbers to estimate.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'data', def: 'diff', options: DATA },
      { id: 'n', kind: 'number', label: 'points', def: 100, min: 12, max: 300 },
      { id: 'seed', kind: 'number', label: 'seed', def: 3, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Different shapes: QDA bends', values: { data: 'diff', n: 100, seed: 3 } },
      { label: 'The same shape: LDA is enough', values: { data: 'same', n: 100, seed: 1 } },
      { label: 'A narrow class inside a wide one', values: { data: 'inside', n: 120, seed: 2 } },
      { label: 'Meadow wildflowers', values: { data: 'flowers', n: 150, seed: 1 } },
      { label: 'Few points: QDA guesses its shapes', values: { data: 'diff', n: 14, seed: 5 } },
    ],
    counters: [{ id: 'step', label: 'step' }],
    run(inp) {
      const D = H.daData(inp.data, inp.n, inp.seed), X = D.X, y = D.y, K = D.classes.length, n = X.length;
      const lda = ML().LDA().fit(X, y), qda = ML().QDA().fit(X, y);
      const te = H.daData(inp.data, 400, inp.seed + 7777);
      const accOf = (m, A, b) => m.predict(A).filter((v, i) => v === b[i]).length / b.length;
      const accL = accOf(lda, X, y), accQ = accOf(qda, X, y), tL = accOf(lda, te.X, te.y), tQ = accOf(qda, te.X, te.y);
      const xr = H.nice(X.map((r) => r[0]), 0.08), yr = H.nice(X.map((r) => r[1]), 0.08);
      const panel = (P, x0) => H.axes({ prefix: P, x: x0, y: 50, w: 330, h: 330, xr, yr, xname: D.names[0], yname: D.names[1], grid: false });
      const aL = panel('dl', 46), aQ = panel('dq', 430);
      const F = new S.Frames(['step']);
      const heads = [H.title('dl:t', 46, 18, 'LDA: one shared ellipse'), H.title('dq:t', 430, 18, 'QDA: an ellipse for each class')];
      const pts = (ax, P) => SC().points(ax, P + ':p', X, y, { r: 3.6, cls: () => 'faded' });
      const meanDots = (ax, P, means) => means.map((m, k) => ({ key: P + ':m' + k, kind: 'dot', x: ax.X(m[0]), y: ax.Y(m[1]), r: 7, cls: 'mark c' + k, point: true, z: 5 }));
      const ells = (ax, P, means, covs) => means.flatMap((m, k) => [1, 2].map((s) => H.ellipsePath(ax, P + ':e' + k + '_' + s, m, covs[k], s, 'c' + k + (s === 2 ? ' soft' : ''), true)));
      const region = (ax, P, m) => SC().field(ax, P + ':f', (u, v) => { const p = m.predictProba([[u, v]])[0], k = ML().argmax(p); return [k, Math.min(1, (p[k] - 1 / K) / (1 - 1 / K) * 1.15)]; }, { n: 66, scheme: 'classes', smooth: false });
      const bound = (ax, P, m) => (K === 2 ? [SC().level(ax, P + ':b', SC().grid(ax, (u, v) => m.predictProba([[u, v]])[0][1], 70), 0.5, '')] : []);
      const meta0 = { n, K, priors: lda.priors };
      const base = () => [...heads, ...aL.items, ...aQ.items];
      // 1: the data
      F.add({ items: [...base(), ...SC().points(aL, 'dl:p', X, y, { r: 3.6 }), ...SC().points(aQ, 'dq:p', X, y, { r: 3.6 })], say: n + ' points of ' + K + ' classes. Both models will fit a Gaussian to each class; they differ only in whether the classes must share a covariance.', meta: Object.assign({ phase: 'data' }, meta0) });
      // 2: the means and priors (the LDA trace's first step; QDA estimates the same ones)
      F.set('step', 1);
      F.add({ items: [...base(), ...pts(aL, 'dl'), ...pts(aQ, 'dq'), ...meanDots(aL, 'dl', lda.means), ...meanDots(aQ, 'dq', qda.means)], say: 'Each class gets its mean, ' + H.list(lda.means.map((m) => H.vec(m))) + ', and its prior, the share of the data it holds: ' + H.list(lda.priors.map((p) => n$(p))) + '. Both models use the same ones.', meta: Object.assign({ phase: 'means', means: lda.means }, meta0) });
      // 3: LDA's shared covariance
      const sharedC = lda.means.map(() => lda.covariance), Sg = lda.covariance;
      F.set('step', 2);
      F.add({ items: [...base(), ...pts(aL, 'dl'), ...pts(aQ, 'dq'), ...ells(aL, 'dl', lda.means, sharedC), ...meanDots(aL, 'dl', lda.means), ...meanDots(aQ, 'dq', qda.means)], say: 'LDA pools the spread of every class around its own mean into one covariance: variances ' + n$(Sg[0][0]) + ' and ' + n$(Sg[1][1]) + ', covariance ' + n$(Sg[0][1]) + '. The same ellipse (1 and 2 standard deviations) is drawn at every mean.', meta: Object.assign({ phase: 'shared', cov: Sg }, meta0) });
      // 4: QDA's covariance per class, one frame per trace step
      const qsteps = qda.trace.filter((s) => s.kind === 'class');
      qsteps.forEach((s, j) => {
        F.set('step', 3 + j);
        const C = s.cov, corr = C[0][1] / Math.sqrt(C[0][0] * C[1][1]);
        F.add({ items: [...base(), ...pts(aL, 'dl'), ...pts(aQ, 'dq'), ...ells(aL, 'dl', lda.means, sharedC), ...ells(aQ, 'dq', qda.means.slice(0, j + 1), qda.covariances.slice(0, j + 1)), ...meanDots(aL, 'dl', lda.means), ...meanDots(aQ, 'dq', qda.means)], say: 'QDA gives ' + D.classes[s.k] + ' its own covariance: variances ' + n$(C[0][0]) + ' and ' + n$(C[1][1]) + ', correlation ' + n$(corr) + '.' + (j === 0 ? '' : ' Compare its ellipse with the ones before: the shapes differ, and LDA had to average them.'), meta: Object.assign({ phase: 'qcov', k: s.k, cov: C, corr }, meta0) });
      });
      // 5: the boundaries
      F.set('step', 3 + qsteps.length);
      const bitems = [...base(), region(aL, 'dl', lda), region(aQ, 'dq', qda), ...bound(aL, 'dl', lda), ...bound(aQ, 'dq', qda), ...SC().points(aL, 'dl:p', X, y, { r: 3.6 }), ...SC().points(aQ, 'dq:p', X, y, { r: 3.6 }), ...ells(aL, 'dl', lda.means, sharedC), ...ells(aQ, 'dq', qda.means, qda.covariances)];
      const wd = K === 2 ? [lda.coef[1][0] - lda.coef[0][0], lda.coef[1][1] - lda.coef[0][1]] : null, bd = K === 2 ? lda.intercept[1] - lda.intercept[0] : null;
      F.add({ items: bitems, say: (K === 2 ? 'LDA\'s log-odds are linear, ' + n$(wd[0]) + '·x₁' + H.term(wd[1], '·x₂', false) + H.term(bd, '', false) + ', so its boundary is the straight line where that is 0. ' : 'LDA\'s class scores are linear, so its regions meet along straight lines. ') + 'QDA keeps the quadratic terms, and its boundary curves around the narrower class. On the training points: LDA ' + H.pct(accL) + ', QDA ' + H.pct(accQ) + ' right.', meta: Object.assign({ phase: 'bound', accL, accQ, wd, bd }, meta0) });
      // 6: fresh points
      const tItems = [...base(), region(aL, 'dl', lda), region(aQ, 'dq', qda), ...bound(aL, 'dl', lda), ...bound(aQ, 'dq', qda), ...SC().points(aL, 'dl:t', te.X.slice(0, 160), te.y, { r: 3, cls: () => 'hollow' }), ...SC().points(aQ, 'dq:t', te.X.slice(0, 160), te.y, { r: 3, cls: () => 'hollow' })];
      const better = tQ > tL + 0.01 ? 'QDA wins: the classes really do have different shapes.' : tL > tQ + 0.01 ? 'LDA wins: with ' + (n < 40 ? 'so few points' : 'shapes this alike') + ', QDA\'s extra numbers fit noise.' : 'They tie: the extra flexibility of QDA buys nothing here.';
      F.add({ items: tItems, say: 'On 400 fresh points from the same classes (hollow, 160 drawn): LDA gets ' + H.pct(tL) + ' and QDA ' + H.pct(tQ) + ' right. ' + better + ' QDA estimates ' + K * 5 + ' numbers for its means and covariances, LDA ' + (K * 2 + 3) + '.', meta: Object.assign({ phase: 'test', tL, tQ, nTest: 400, nDrawn: 160, paramsQ: K * 5, paramsL: K * 2 + 3 }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Discriminant analysis', text: ['class k: prior πₖ, mean μₖ, covariance Σₖ', 'score: ln πₖ − ½ ln|Σₖ| − ½ (x − μₖ)ᵀ Σₖ⁻¹ (x − μₖ)', 'QDA: each class its own Σₖ, a curved boundary', 'LDA: Σₖ = Σ for all, so xᵀΣ⁻¹x cancels:', '  score = xᵀΣ⁻¹μₖ − ½ μₖᵀΣ⁻¹μₖ + ln πₖ, linear'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'shared', pick: 'option', ask: 'What shape will LDA\'s boundary be?', options: ['straight', 'curved', 'a circle', 'it depends on the data'], answer: 'straight', explain: 'With one shared covariance, the quadratic terms cancel between classes.' },
      { at: (f, i, fr) => f.meta.phase === 'bound', pick: 'option', ask: 'On fresh points, which will do better?', options: ['QDA', 'LDA', 'about the same'], answer: (fr) => { const t = fr.find((f) => f.meta.phase === 'test').meta; return t.tQ > t.tL + 0.01 ? 'QDA' : t.tL > t.tQ + 0.01 ? 'LDA' : 'about the same'; }, explain: (fr) => { const t = fr.find((f) => f.meta.phase === 'test').meta; return 'LDA ' + H.pct(t.tL) + ', QDA ' + H.pct(t.tQ) + '.'; } },
    ],
    quiz: [
      { q: 'Why is LDA\'s boundary straight?', options: ['all classes share one covariance, so the quadratic term is the same for every class and cancels', 'it uses a sigmoid', 'it fits a line by least squares', 'the means are equal'], answer: 0, why: 'Only the terms linear in x differ between classes.' },
      { q: 'With 2 features and 3 classes, how many numbers do QDA\'s covariances hold?', options: ['9 (3 per class)', '3', '4', '12'], answer: 0, why: 'A 2 × 2 symmetric matrix has 3 free numbers.' },
      { q: 'When is LDA likely to beat QDA on new data?', options: ['few points, or classes of similar shape', 'many points and very different shapes', 'always', 'never'], answer: 0, why: 'QDA\'s extra numbers are noisy when data are scarce.' },
      { q: 'What do LDA and QDA assume about each class?', options: ['a Gaussian distribution', 'a uniform distribution', 'a linear boundary', 'nothing'], answer: 0, why: 'They are generative models: they model p(x | class).' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
