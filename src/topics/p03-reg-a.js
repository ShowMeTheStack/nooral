// Part 3, ridge and lasso coefficient paths. Every feature is standardised, then the penalty λ slides from strong
// (every weight squeezed to zero) to weak (the least-squares weights). Ridge (ML.Ridge, one solve per λ) shrinks all
// weights smoothly and never to exactly zero; lasso (ML.enetPath, coordinate descent warm-started down the path)
// lets weights in one at a time and holds the rest at exactly zero.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n, C = () => S.Kit.curve;

  /** our own data: n rows, features named, y from the given weights plus noise; groups of correlated features */
  H.regData = (kind, seed) => {
    const r = S.Data.rng(seed), rows = [], y = [];
    if (kind === 'rents') {
      const t = S.Data.load('rents', { seed, n: 120 });
      return { X: t.X.map((q) => q.slice()), y: t.y.slice(), names: ['area', 'rooms', 'floor', 'distance', 'balcony'], truth: null };
    }
    if (kind === 'pair') {
      // two near-copies that both matter, and four that do not
      // y depends on the shared signal z, which both twins measure with a little error
      for (let i = 0; i < 80; i++) { const z = r.normal(0, 1), x = [z + r.normal(0, 0.2), z + r.normal(0, 0.45), r.normal(0, 1), r.normal(0, 1), r.normal(0, 1), r.normal(0, 1)]; rows.push(x.map((v) => S.Data.round(v, 4))); y.push(S.Data.round(4 * z + r.normal(0, 1.5), 4)); }
      return { X: rows, y, names: ['twin 1', 'twin 2', 'x3', 'x4', 'x5', 'x6'], truth: null };
    }
    // three of eight matter, the features mildly correlated
    const w = [3, -2, 1.5, 0, 0, 0, 0, 0];
    for (let i = 0; i < 60; i++) { const z = r.normal(0, 1), x = w.map(() => 0.5 * z + 0.87 * r.normal(0, 1)); rows.push(x.map((v) => S.Data.round(v, 4))); y.push(S.Data.round(x.reduce((s, v, j) => s + v * w[j], 0) + r.normal(0, 2), 4)); }
    return { X: rows, y, names: ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'], truth: w };
  };
  const DATA = [{ value: 'sparse', label: '3 of 8 features matter' }, { value: 'rents', label: 'flat rents (5 features)' }, { value: 'pair', label: 'twin features' }];

  S.topic({
    id: 'reg-paths', family: 'p03-reg', order: 10, title: 'Ridge and lasso coefficient paths',
    blurb: 'Add a penalty on the size of the weights to the squared error and slide its strength λ. Ridge penalises Σ w², and as λ grows every weight shrinks smoothly toward zero without ever reaching it. Lasso penalises Σ |w|, and its weights hit zero one after another: it selects features. The features are standardised first, so the penalty treats them fairly.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'data', def: 'sparse', options: DATA },
      { id: 'method', kind: 'choice', label: 'penalty', def: 'lasso', options: [{ value: 'lasso', label: 'lasso: λ Σ |w|' }, { value: 'ridge', label: 'ridge: λ Σ w²' }] },
      { id: 'steps', kind: 'number', label: 'values of λ', def: 18, min: 6, max: 40 },
      { id: 'seed', kind: 'number', label: 'seed', def: 5, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Lasso: 3 of 8 features matter', values: { data: 'sparse', method: 'lasso', steps: 18, seed: 5 } },
      { label: 'Ridge on the same data', values: { data: 'sparse', method: 'ridge', steps: 18, seed: 5 } },
      { label: 'Lasso on flat rents', values: { data: 'rents', method: 'lasso', steps: 18, seed: 2 } },
      { label: 'Ridge on flat rents', values: { data: 'rents', method: 'ridge', steps: 18, seed: 2 } },
      { label: 'Lasso and twin features', values: { data: 'pair', method: 'lasso', steps: 20, seed: 3 } },
      { label: 'Ridge and twin features', values: { data: 'pair', method: 'ridge', steps: 20, seed: 3 } },
    ],
    counters: [{ id: 'nz', label: 'non-zero weights' }],
    run(inp) {
      const raw = H.regData(inp.data, inp.seed), st = H.standardise(raw.X), X = st.X, y = raw.y, n = X.length, d = X[0].length, names = raw.names;
      const ym = ML().mean(y), lasso = inp.method === 'lasso', K = inp.steps;
      const ols = ML().LinearRegression().fit(X, y).coef;
      // the λ values, strongest first; the path of weights at each
      let alphas, coefs, label;
      if (lasso) {
        const p = ML().enetPath(X, y, { l1Ratio: 1, nAlphas: K, eps: 1e-3 });
        alphas = p.alphas; coefs = p.coefs; label = 'λ (lasso)';
      } else {
        const amax = 1e4 * n / 60, amin = 1e-2 * n / 60;
        alphas = ML().range(K).map((k) => amax * Math.pow(amin / amax, k / (K - 1)));
        coefs = alphas.map((a) => ML().Ridge({ alpha: a }).fit(X, y).coef);
        label = 'λ (ridge)';
      }
      const mse = (w) => { let s = 0; for (let i = 0; i < n; i++) { let p = ym; for (let j = 0; j < d; j++) p += w[j] * X[i][j]; s += (y[i] - p) ** 2; } return s / n; };
      // the intercept of a centred fit is the mean of y: predictions above use it directly
      const lg = (v) => Math.log10(v), xr = [Math.floor(lg(Math.min(...alphas)) * 2) / 2, Math.ceil(lg(Math.max(...alphas)) * 2) / 2];
      const wmax = Math.max(...ols.map(Math.abs), ...coefs.flat().map(Math.abs)) * 1.08, yr = H.nice([-wmax, wmax], 0);
      const ax = H.axes({ prefix: 'rp', x: 46, y: 40, w: 400, h: 380, xr, yr, xticks: 1, fmtX: (v) => (Number.isInteger(v) ? '10' + (v < 0 ? '⁻' : '') + H.sup(Math.abs(v)) : ''), xname: label + ', stronger →', yname: 'weight (standardised x)' });
      const cls = (j) => 'f' + (j % 8);
      const F = new S.Frames(['nz']);
      const series = (upto) => names.map((nm, j) => C().line(ax, 'rp:s' + j, alphas.slice(0, upto + 1).map((a, k) => [lg(a), coefs[k][j]]), cls(j) + ' thin'));
      const zero = C().hline(ax, 'rp:zero', 0, '');
      const barX = 560, barY = 66, rowH = Math.min(40, 300 / d);
      const bars = (w) => H.hbars('rp:b', barX, barY, 180, rowH, w, names, { max: wmax, cls: (j) => cls(j), nameCls: cls, values: true });
      const head = () => [...ax.items, zero, H.title('rp:t', 46, 14, 'The weights as λ slides'), H.title('rp:t2', 500, 30, 'At this λ')];
      const nz = (w) => w.filter((v) => Math.abs(v) > 0).length;
      const meta0 = { n, d, olsMax: Math.max(...ols.map(Math.abs)) };
      F.add({ items: [...head(), ...bars(ols), ...H.eqs('rp:eq', 500, 400, [['no penalty: least squares', 'muted'], ['training MSE ' + n$(mse(ols)), 'muted']])], say: n + ' rows and ' + d + ' standardised features. With no penalty the least-squares weights are the bars at the right; their training error is ' + n$(mse(ols)) + '. Now we add the penalty at full strength and weaken it step by step.', meta: Object.assign({ phase: 'ols', ols, mse: mse(ols) }, meta0) });
      let prevNz = -1, entered = [];
      alphas.forEach((a, k) => {
        const w = coefs[k], m = mse(w), z = nz(w);
        F.set('nz', z);
        const items = [...head(), ...series(k), C().vline(ax, 'rp:at', lg(a), 'act'), ...bars(w)];
        items.push(...H.eqs('rp:eq', 500, 400, [['λ = ' + n$(a), 'act'], [z + ' of ' + d + ' weights non-zero', lasso ? 'good' : 'muted'], ['training MSE ' + n$(m), 'muted']]));
        const newOnes = lasso && prevNz >= 0 ? names.filter((nm, j) => w[j] !== 0 && coefs[k - 1][j] === 0) : [];
        entered = entered.concat(newOnes);
        const big = names[ML().argmax(w.map(Math.abs))];
        let say;
        if (k === 0) say = lasso ? 'λ = ' + n$(a) + ' is just strong enough to hold every weight at exactly 0: the model predicts the mean of y for everyone (training MSE ' + n$(m) + ').' : 'λ = ' + n$(a) + ': ridge squeezes every weight close to 0 (the largest is ' + n$(Math.max(...w.map(Math.abs))) + '), but none is exactly 0. Training MSE ' + n$(m) + '.';
        else if (lasso && newOnes.length) say = 'λ = ' + n$(a) + ': ' + H.list(newOnes) + (newOnes.length === 1 ? ' leaves' : ' leave') + ' zero and ' + (newOnes.length === 1 ? 'joins' : 'join') + ' the model; ' + z + ' of ' + d + ' weights are now non-zero. Training MSE ' + n$(m) + '.';
        else if (lasso) say = 'λ = ' + n$(a) + ': still ' + z + ' non-zero weights, growing toward least squares; the largest is on ' + big + '. Training MSE ' + n$(m) + '.';
        else say = 'λ = ' + n$(a) + ': all ' + d + ' weights are non-zero and growing together; the largest is on ' + big + ' (' + n$(w[ML().argmax(w.map(Math.abs))]) + '). Training MSE ' + n$(m) + '.';
        F.add({ items, say, meta: Object.assign({ phase: 'path', k, alpha: a, w, mse: m, nz: z }, meta0) });
        prevNz = z;
      });
      const last = coefs[coefs.length - 1], firstIn = lasso ? entered[0] : null;
      F.add({ items: [...head(), ...series(alphas.length - 1), ...bars(last), ...H.eqs('rp:eq', 500, 400, [['weak penalty: close to', 'muted'], ['least squares', 'muted']])], say: lasso ? 'Read right to left: the lasso path. ' + (firstIn ? firstIn + ' was the first feature in, the one that explains most.' : '') + ' At the weakest λ the weights are close to least squares. A λ chosen in between keeps only the features that earn their place.' : 'Read right to left: the ridge path. Every weight shrinks smoothly as λ grows, the larger ones fastest in absolute terms, and none reaches zero at any finite λ: ridge never removes a feature.', meta: Object.assign({ phase: 'end', firstIn }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Penalised least squares', text: ['ridge: ‖y − Xw‖² + λ Σ wⱼ²', '  w = (XᵀX + λI)⁻¹ Xᵀy, never exactly 0', 'lasso: (1/2n)‖y − Xw‖² + λ Σ |wⱼ|', '  coordinate descent, soft-thresholding:', '  wⱼ = S(ρⱼ, nλ) / ‖xⱼ‖², exactly 0 when |ρⱼ| ≤ nλ', 'features standardised; the intercept is not penalised'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr, inp) => f.meta.phase === 'ols', pick: 'option', ask: (fr, i, inp) => 'As λ grows very large, what happens to the ' + (inp.method === 'lasso' ? 'lasso' : 'ridge') + ' weights?', options: ['they all become exactly zero', 'they shrink toward zero but never reach it', 'they grow', 'they stay at least squares'], answer: (fr, i, inp) => (inp.method === 'lasso' ? 'they all become exactly zero' : 'they shrink toward zero but never reach it'), explain: (fr, i, inp) => (inp.method === 'lasso' ? 'Past λ = max |xⱼᵀy| / n every soft-threshold returns 0.' : '(XᵀX + λI)⁻¹Xᵀy only tends to 0 as λ → ∞.') },
    ],
    quiz: [
      { q: 'Why standardise features before ridge or lasso?', options: ['the penalty compares weight sizes, which depend on each feature\'s units', 'it makes the loss convex', 'it removes the intercept', 'it is not needed'], answer: 0, why: 'A feature in metres would need a 1000 times larger weight than the same feature in millimetres, so the penalty would treat them very differently.' },
      { q: 'Which penalty can set a weight to exactly zero?', options: ['lasso (Σ |w|)', 'ridge (Σ w²)', 'both', 'neither'], answer: 0, why: 'The soft-threshold step returns exactly 0 whenever the correlation with the residual is small enough.' },
      { q: 'At λ = 0 both methods give…', options: ['ordinary least squares', 'all zeros', 'the mean of y', 'different answers'], answer: 0, why: 'No penalty, no change.' },
      { q: 'Moving λ up from 0 usually makes the training error…', options: ['rise', 'fall', 'stay the same', 'become zero'], answer: 0, why: 'Least squares is the best possible fit to the training data; any penalty trades some of it for smaller weights.' },
      { q: 'Two nearly identical features both matter. Lasso tends to…', options: ['keep one and drop the other', 'keep both equally', 'drop both', 'double both'], answer: 0, why: 'The L1 corner prefers one; elastic net, with some L2, shares between them.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
