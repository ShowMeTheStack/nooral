// Part 2, complexity and tuning. Learning curves: training and validation error as the training set grows, for a
// model too simple (high bias), one too flexible (high variance) and one in between, with the fitted curve beside.
// Validation curves: the two errors against one setting. Grid and random search: cross-validated scores of an RBF
// SVM over C and γ as a heat map, with randomly sampled settings dropped on top.
(function (S) {
  'use strict';
  const H = S.P01, P = S.P02, C = () => S.Kit.curve, HM = () => S.Kit.heat, ML = () => S.ML;
  /** a noisy sine, rows shuffled so any prefix is a fair sample */
  P.sine = () => P.once('sine300', () => { const d = S.Data.curve({ n: 300, shape: 'sine', noise: 0.3, seed: 12 }), perm = ML().rng(5).perm(d.X.length); return { X: perm.map((i) => d.X[i]), y: perm.map((i) => d.y[i]), noise: 0.3 }; });
  const mseScore = (m, X, y) => ML().mse(y, m.predict(X));

  const LC = {
    line: { label: 'a straight line (too simple)', make: () => ML().LinearRegression() },
    deep: { label: 'a tree grown to the end (too flexible)', make: () => ML().DecisionTreeRegressor({ seed: 0 }) },
    tree4: { label: 'a depth-4 tree', make: () => ML().DecisionTreeRegressor({ maxDepth: 4, seed: 0 }) },
  };
  S.topic({
    id: 'eval-learning-curve', family: 'p02-tuning', order: 10, title: 'Bias, variance and learning curves',
    blurb: 'Train on more and more rows and measure the error twice: on the rows trained on, and on held-out rows. A model too simple for the pattern (high bias) has both errors high and close together, and more data will not help. A model too flexible (high variance) has a tiny training error and a much larger validation error; more data narrows the gap. The noise in the data is a floor no model can go below.',
    inputs: [{ id: 'model', kind: 'choice', label: 'Model', def: 'line', options: Object.entries(LC).map(([value, m]) => ({ value, label: m.label })) }],
    presets: Object.entries(LC).map(([value, m]) => ({ label: m.label, values: { model: value } })),
    counters: [{ id: 'm', label: 'training rows' }],
    run(inp) {
      const D = P.sine(), ml = ML(), folds = ml.kFold(D.X.length, 5, { shuffle: true, seed: 1 });
      const lc = P.once('lc:' + inp.model, () => ml.learningCurve(LC[inp.model].make(), D.X, D.y, { cv: folds, trainSizes: [0.04, 0.08, 0.15, 0.3, 0.5, 0.75, 1], scoring: mseScore }));
      const tr = lc.train.map(H.mean), va = lc.test.map(H.mean), sizes = lc.sizes, floor = D.noise ** 2;
      const top = Math.min(0.9, Math.max(...va, ...tr) * 1.1);
      const ax = C().axes({ prefix: 'ax', x: 56, y: 50, w: 360, h: 300, xr: [0, sizes[sizes.length - 1]], yr: [0, top], yticks: C().ticks(0, top, 4), xname: 'training rows', yname: 'mean squared error', tickSize: 12 });
      const fx = C().axes({ prefix: 'fx', x: 480, y: 50, w: 290, h: 200, xr: [-3.1, 3.1], yr: [-1.8, 1.8], xticks: [-3, 0, 3], yticks: [-1, 0, 1], xname: 'x', yname: 'y', tickSize: 12 });
      const fl = { key: 'floor', kind: 'path', d: `M${ax.o.x},${ax.Y(Math.min(top, floor))} H${ax.o.x + ax.o.w}`, cls: 'diag', audit: 'ignore', layer: 'back', z: 3 };
      const full = LC[inp.model].make().fit(folds[0].train.map((i) => D.X[i]), folds[0].train.map((i) => D.y[i]));
      const fit = C().line(fx, 'fit', Array.from({ length: 150 }, (_, k) => { const u = -3 + (6 * k) / 149; return [u, full.predict([[u]])[0]]; }), 'act');
      const dots = D.X.slice(0, 120).map((p, i) => ({ key: 'd' + i, kind: 'dot', x: fx.X(p[0]), y: fx.Y(Math.max(-1.8, Math.min(1.8, D.y[i]))), r: 2, cls: 'pt0', audit: 'overlap-ok' }));
      const legend = C().legend('lg', 60, 410, [{ text: 'training error', cls: 'c0' }, { text: 'validation error', cls: 'c1' }, { text: 'noise floor ' + H.f(floor, 2), cls: 'faint' }]);
      const F = new S.Frames(['m']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, 'A noisy sine, 300 rows, 5 folds: ' + LC[inp.model].label);
      const upto = (k) => [C().line(ax, 'tr', sizes.slice(0, k + 1).map((m, j) => [m, tr[j]]), 'c0'), C().line(ax, 'va', sizes.slice(0, k + 1).map((m, j) => [m, va[j]]), 'c1'), Object.assign(C().dot(ax, 'tdot', sizes[k], Math.min(top, tr[k]), 'c0'), { point: false, audit: 'overlap-ok' }), Object.assign(C().dot(ax, 'vdot', sizes[k], Math.min(top, va[k]), 'c1'), { point: false, audit: 'overlap-ok' })];
      const side = (ls) => ls.map((l, i) => H.text('sd' + i, 480, 300 + i * 24, l[0], l[1] || 'strong', 13));
      [0, 2, 4, 6].forEach((k) => {
        F.set('m', sizes[k]);
        F.add({ items: [title, ...ax.items, fl, ...upto(k), ...legend, ...fx.items, ...dots, fit, ...side([['the fit on 240 rows (right)', 'tick']])], say: `With ${sizes[k]} training rows (averaged over 5 folds): training error ${H.f(tr[k], 3)}, validation error ${H.f(va[k], 3)}.` + (k === 0 ? ` The dashed floor, ${H.f(floor, 2)}, is the variance of the noise in the data.` : ''), meta: { phase: 'size', k, m: sizes[k], train: tr[k], val: va[k] }, view });
      });
      const last = sizes.length - 1, gap = va[last] - tr[last];
      const diag = gap > 0.05 && tr[last] < floor * 0.7 ? 'variance' : va[last] > floor * 1.5 && gap < 0.05 ? 'bias' : 'balanced';
      const words = { variance: 'High variance: the training error is far below the noise floor (it memorised the noise) and the gap to the validation error is large. More rows, or a simpler model, would help.', bias: 'High bias: both errors are high and close together, well above the noise floor. More rows will not help; the model needs more flexibility.', balanced: 'Both errors end near the noise floor with a small gap: about as good as this data allows.' };
      F.add({ items: [title, ...ax.items, fl, ...upto(last), ...legend, ...fx.items, ...dots, fit, ...side([['gap at the end: ' + H.f(gap, 3), 'act'], ['validation ' + H.f(va[last], 3) + ' vs floor ' + H.f(floor, 2), 'strong']])], say: `At ${sizes[last]} rows: training ${H.f(tr[last], 3)}, validation ${H.f(va[last], 3)}, a gap of ${H.f(gap, 3)}. ${words[diag]}`, meta: { phase: 'end', train: tr[last], val: va[last], gap, diag, floor }, view });
      F.list.forEach((f) => { f.meta.ctx = { floor, folds: 5, max: sizes[last] }; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Reading learning curves', text: ['error = bias² + variance + noise', 'high bias: both errors high, close together', 'high variance: low training, high validation', 'more data shrinks variance, not bias', 'the noise is a floor for every model'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'size' && f.meta.k === 0, pick: 'option', ask: 'At the full training size, what will the diagnosis be?', options: ['high bias', 'high variance', 'about right'], answer: (fr) => ({ bias: 'high bias', variance: 'high variance', balanced: 'about right' })[fr[fr.length - 1].meta.diag], explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Training error 0.30, validation error 0.31, noise floor 0.05. The model has…', options: ['high bias', 'high variance', 'no problem', 'leakage'], answer: 0, why: 'Both errors are high and close: it cannot represent the pattern.' },
      { q: 'Training error 0.01, validation error 0.40. The model has…', options: ['high variance', 'high bias', 'no problem', 'too little flexibility'], answer: 0, why: 'It fits the training rows far better than new ones.' },
      { q: 'Which fix helps high bias?', options: ['a more flexible model or better features', 'more training rows', 'stronger regularisation', 'fewer features'], answer: 0, why: 'More of the same data cannot add the missing flexibility.' },
      { q: 'Why does the training error usually rise as the training set grows?', options: ['more rows are harder to fit all at once', 'the model gets worse', 'the noise grows', 'it does not'], answer: 0, why: 'A few rows can be fitted almost exactly; many cannot.' },
      { q: 'Can validation error go below the noise floor on average?', options: ['no', 'yes, with enough data', 'yes, with a deep tree', 'only for classification'], answer: 0, why: 'The noise is unpredictable by definition.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Validation curves
  // ---------------------------------------------------------------------------------------------------------------
  const VC = {
    depth: { label: 'tree depth on the noisy sine', param: 'maxDepth', values: [1, 2, 3, 4, 5, 6, 8, 10, 14], make: () => ML().DecisionTreeRegressor({ seed: 0 }), reg: true, simple: 'shallow', complex: 'deep' },
    leaf: { label: 'minimum rows per leaf on the sine', param: 'minSamplesLeaf', values: [1, 2, 4, 8, 16, 32, 64, 128], make: () => ML().DecisionTreeRegressor({ seed: 0 }), reg: true, simple: 'large leaves', complex: 'small leaves', reversed: true },
    k: { label: 'k of k-NN on two moons', param: 'k', values: [1, 3, 5, 9, 15, 25, 45, 75, 119], make: () => ML().KNeighborsClassifier(), reg: false, simple: 'large k', complex: 'small k', reversed: true },
  };
  P.moonsVC = () => P.once('moonsvc', () => { const d = S.Data.moons({ n: 200, noise: 0.35, seed: 8 }); return { X: d.X, y: d.y }; });
  S.topic({
    id: 'eval-validation-curve', family: 'p02-tuning', order: 20, title: 'Validation curves',
    blurb: 'Most models have a knob that sets how flexible they are: a tree\'s depth, how many rows a leaf must hold, the k of k-NN. Sweep it and cross-validate at each value. On the simple side both scores are poor (underfitting); on the flexible side the training score keeps improving while the validation score turns down (overfitting). Choose the value where the validation score is best, never where the training score is.',
    inputs: [{ id: 'knob', kind: 'choice', label: 'Knob', def: 'depth', options: Object.entries(VC).map(([value, v]) => ({ value, label: v.label })) }],
    presets: Object.entries(VC).map(([value, v]) => ({ label: v.label, values: { knob: value } })),
    counters: [{ id: 'vals', label: 'values tried' }],
    run(inp) {
      const V = VC[inp.knob], ml = ML(), D = V.reg ? P.sine() : P.moonsVC();
      const folds = V.reg ? ml.kFold(D.X.length, 5, { shuffle: true, seed: 1 }) : ml.stratifiedKFold(D.y, 5, { shuffle: true, seed: 1 });
      const vc = P.once('vc:' + inp.knob, () => ml.validationCurve(V.make(), D.X, D.y, V.param, V.values, { cv: folds, scoring: V.reg ? mseScore : 'accuracy' }));
      const tr = vc.train.map(H.mean), va = vc.test.map(H.mean), n = V.values.length;
      const better = (a, b) => (V.reg ? a < b : a > b), best = va.reduce((b, v, k) => (better(v, va[b]) ? k : b), 0);
      const lo = V.reg ? 0 : Math.min(...va, ...tr) - 0.05, hi = V.reg ? Math.min(0.9, Math.max(...va, ...tr) * 1.1) : 1.0;
      const ax = C().axes({ prefix: 'ax', x: 60, y: 50, w: 460, h: 300, xr: [-0.5, n - 0.5], yr: [lo, hi], xticks: V.values.map((_, k) => k), fmtX: (k) => String(V.values[Math.round(k)]), yticks: C().ticks(lo, hi, 4), xname: V.param, yname: V.reg ? 'mean squared error' : 'accuracy', tickSize: 12 });
      const F = new S.Frames(['vals']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, 'Validation curve: ' + V.label + ' (5-fold)');
      const legend = C().legend('lg', 60, 420, [{ text: 'training', cls: 'c0' }, { text: 'validation', cls: 'c1' }]);
      const upto = (k) => [C().line(ax, 'tr', tr.slice(0, k + 1).map((v, j) => [j, v]), 'c0'), C().line(ax, 'va', va.slice(0, k + 1).map((v, j) => [j, v]), 'c1'), Object.assign(C().dot(ax, 'td', k, tr[k], 'c0'), { point: false, audit: 'overlap-ok' }), Object.assign(C().dot(ax, 'vd', k, va[k], 'c1'), { point: false, audit: 'overlap-ok' })];
      const side = (ls) => ls.map((l, i) => H.text('sd' + i, 560, 80 + i * 26, l[0], l[1] || 'strong', 14));
      const marks = [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1];
      marks.forEach((k) => {
        F.set('vals', k + 1);
        F.add({ items: [title, ...ax.items, ...upto(k), ...legend, ...side([[V.param + ' = ' + V.values[k], 'act'], ['training ' + H.f(tr[k], 3), 'c0'], ['validation ' + H.f(va[k], 3), 'c1']])], say: `${V.param} = ${V.values[k]}: training ${V.reg ? 'error' : 'accuracy'} ${H.f(tr[k], 3)}, validation ${H.f(va[k], 3)}.` + (k === 0 ? ` The first value is the most ${V.reversed ? 'flexible' : 'restricted'} setting.` : ''), meta: { phase: 'value', k, v: V.values[k], train: tr[k], val: va[k] }, view });
      });
      const band = { key: 'best', kind: 'path', d: `M${ax.X(best)},${ax.o.y} V${ax.o.y + ax.o.h}`, cls: 'guide act', audit: 'ignore', layer: 'back', z: 3 };
      const flexEnd = V.reversed ? 0 : n - 1;
      F.add({ items: [title, ...ax.items, band, ...upto(n - 1), ...legend, ...side([['best ' + V.param + ' = ' + V.values[best], 'act'], ['validation ' + H.f(va[best], 3), 'c1'], [V.simple + ': underfits', 'tick'], [V.complex + ': overfits', 'tick']])], say: `The validation ${V.reg ? 'error is lowest' : 'accuracy is highest'} at ${V.param} = ${V.values[best]} (${H.f(va[best], 3)}). At the most flexible end, ${V.param} = ${V.values[flexEnd]}, training ${V.reg ? 'error' : 'accuracy'} is ${H.f(tr[flexEnd], 3)} but validation is ${H.f(va[flexEnd], 3)}: the model fits the noise.`, meta: { phase: 'best', best, bestValue: V.values[best], val: va[best], flex: flexEnd, trFlex: tr[flexEnd], vaFlex: va[flexEnd] }, view });
      F.list.forEach((f) => { f.meta.ctx = { values: V.values }; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Validation curves', text: ['sweep one setting, cross-validate each value', 'underfit side: both scores poor', 'overfit side: training good, validation poor', 'pick the best validation score', 'then check it once on the test set'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'value' && f.meta.k === 0, pick: 'option', ask: 'Will the best validation score come at one of the two end values?', options: ['yes, at an end', 'no, somewhere in between'], answer: (fr) => { const m = fr[fr.length - 1].meta, n = fr[fr.length - 2].meta.k + 1; return m.best === 0 || m.best === n - 1 ? 'yes, at an end' : 'no, somewhere in between'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'k-NN with k = 1 has training accuracy…', options: ['1 (each row is its own nearest neighbour)', '0.5', 'the same as validation', '0'], answer: 0, why: 'Every training row predicts its own label.' },
      { q: 'On a validation curve, overfitting shows as…', options: ['training score improving while validation gets worse', 'both scores improving', 'both scores worsening', 'the curves crossing at 0'], answer: 0, why: 'The model is learning the training rows\' noise.' },
      { q: 'Which value should you choose?', options: ['the best validation score', 'the best training score', 'the largest value', 'the smallest value'], answer: 0, why: 'Validation estimates performance on new rows.' },
      { q: 'A larger minimum leaf size makes a tree…', options: ['simpler', 'more flexible', 'unchanged', 'deeper'], answer: 0, why: 'Leaves must average over more rows.' },
      { q: 'After choosing the knob on validation scores, the honest estimate comes from…', options: ['the untouched test set', 'the best validation score', 'the training score', 'the noise floor'], answer: 0, why: 'Picking the best of many validation scores is itself optimistic.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Grid and random search
  // ---------------------------------------------------------------------------------------------------------------
  const CS = [0.01, 0.1, 1, 10, 100], GS = [0.01, 0.1, 1, 10, 100];
  P.searchData = () => P.once('moons120', () => { const d = S.Data.moons({ n: 120, noise: 0.3, seed: 2 }); return { X: d.X, y: d.y }; });
  S.topic({
    id: 'eval-search', family: 'p02-tuning', order: 30, title: 'Grid and random search',
    blurb: 'An RBF support vector machine has two knobs: C (how hard it punishes mistakes) and γ (how far one example\'s influence reaches). Grid search tries every pair from fixed lists and cross-validates each; drawn as a heat map, the good settings form a ridge. Random search samples pairs from the same ranges instead: for the same number of fits it tries many more distinct values of each knob, which pays off when only one of them matters much.',
    inputs: [
      { id: 'n', kind: 'slider', label: 'Random samples', def: 12, min: 5, max: 25, step: 1 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 3, min: 0, max: 99 },
    ],
    presets: [
      { label: '12 random samples', values: { n: 12, seed: 3 } },
      { label: '25 random samples', values: { n: 25, seed: 3 } },
      { label: '5 samples, another seed', values: { n: 5, seed: 8 } },
    ],
    counters: [{ id: 'fits', label: 'models fitted' }],
    run(inp) {
      const D = P.searchData(), ml = ML();
      const grid = P.once('grid', () => ml.gridSearch(ml.SVC({ kernel: 'rbf' }), { C: CS, gamma: GS }, D.X, D.y, { cv: 5, refit: false }));
      const score = (c, g) => grid.results.find((r) => r.params.C === c && r.params.gamma === g).mean;
      const rnd = P.once('rnd:' + inp.n + ':' + inp.seed, () => ml.randomSearch(ml.SVC({ kernel: 'rbf' }), { C: ml.logUniform(0.01, 100), gamma: ml.logUniform(0.01, 100) }, D.X, D.y, { cv: 5, nIter: inp.n, seed: inp.seed, refit: false }));
      const vals = GS.map((g) => CS.map((c) => score(c, g))).reverse();
      const lo = Math.min(...grid.results.map((r) => r.mean)), hi = Math.max(...grid.results.map((r) => r.mean));
      const cw = 66, gx = 110, gy = 70;
      const F = new S.Frames(['fits']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, 'RBF SVM on two moons (120 rows): 5-fold accuracy');
      const fmt = (v) => H.f(v, 3).replace(/^0\./, '.');
      const heat = (rows) => HM().grid({ prefix: 'hm', x: gx, y: gy, cw, ch: cw, vals: vals.map((r, i) => r.map((v) => (GS.length - 1 - i < rows ? (v - lo) / (hi - lo || 1) : 0))), scheme: 'seq', max: 1, fmt: null, rows: GS.slice().reverse().map(String), cols: CS.map(String), size: 13, cls: (i, j) => (GS.length - 1 - i === GS.indexOf(grid.bestParams.gamma) && j === CS.indexOf(grid.bestParams.C) && rows === GS.length ? 'sel' : '') }).items
        .map((it) => { const m = /^hm:c(\d+)_(\d+)$/.exec(it.key); if (!m) return it; const i = +m[1], j = +m[2]; return GS.length - 1 - i < rows ? Object.assign(it, { text: fmt(vals[i][j]) }) : it; });
      const axisNames = [H.text('cn', gx + (CS.length * (cw + 2)) / 2, gy + CS.length * (cw + 2) + 22, 'C', 'axname', 13, 'middle'), H.text('gn', gx - 44, gy + 10, 'γ', 'axname', 14, 'end')];
      // log positions on the heat map for random samples: cell centres sit at integer powers of ten
      const px = (c) => gx + (Math.log10(c) + 2) * (cw + 2) + cw / 2, py = (g) => gy + (2 - Math.log10(g)) * (cw + 2) + cw / 2;
      const side = (ls) => ls.map((l, i) => H.text('sd' + i, 480, 90 + i * 26, l[0], l[1] || 'strong', 14));
      [1, 3, 5].forEach((rows) => {
        F.set('fits', rows * CS.length * 5);
        const row = GS[rows - 1], rs = CS.map((c) => score(c, row));
        F.add({ items: [title, ...heat(rows), ...axisNames, ...side([[rows * CS.length + ' of 25 settings', 'act'], ['each: 5 folds, 5 fits', 'tick'], ['γ = ' + row + ': best C = ' + CS[rs.indexOf(Math.max(...rs))], 'strong']])], say: `Grid search, γ = ${GS.slice(0, rows).join(', ')}: every C from ${CS[0]} to ${CS[CS.length - 1]}, each cross-validated. For γ = ${row} the scores run ${rs.map((v) => H.f(v, 3)).join(', ')}.`, meta: { phase: 'grid', rows, scores: rs }, view });
      });
      F.add({ items: [title, ...heat(5), ...axisNames, ...side([['best on the grid:', 'act'], ['C = ' + grid.bestParams.C + ', γ = ' + grid.bestParams.gamma, 'strong'], ['accuracy ' + H.f(grid.bestScore, 3), 'good'], ['worst: ' + H.f(lo, 3), 'tick']])], say: `All 25 settings, 125 fits: the best is C = ${grid.bestParams.C}, γ = ${grid.bestParams.gamma} with ${H.f(grid.bestScore, 3)}. Very large γ does worst (${H.f(lo, 3)}): each example's influence is so local that the model memorises points.`, meta: { phase: 'gridbest', best: grid.bestParams, score: grid.bestScore, lo, cells: 25, fits: 125 }, view });
      const rb = rnd.results[rnd.bestIndex];
      const drops = (k) => rnd.results.slice(0, k).map((r, i) => ({ key: 'rs' + i, kind: 'dot', x: +px(r.params.C).toFixed(2), y: +py(r.params.gamma).toFixed(2), r: 7, cls: 'ring ' + (i === rnd.bestIndex && k === rnd.results.length ? 'hot' : 'act'), audit: 'overlap-ok', layer: 'top' }));
      const half = Math.ceil(inp.n / 2);
      F.set('fits', 125 + half * 5);
      F.add({ items: [title, ...heat(5), ...axisNames, ...drops(half), ...side([['random search: ' + half + ' of ' + inp.n, 'act'], ['C and γ drawn log-uniformly', 'tick'], ['from 0.01 to 100', 'tick']])], say: `Random search draws C and γ log-uniformly between 0.01 and 100 (rings, placed on the same scales). The first ${half} draws score ${rnd.results.slice(0, half).map((r) => H.f(r.mean, 3)).join(', ')}.`, meta: { phase: 'random', k: half, scores: rnd.results.slice(0, half).map((r) => r.mean) }, view });
      F.set('fits', 125 + inp.n * 5);
      const distinctC = new Set(rnd.results.map((r) => r.params.C.toFixed(6))).size;
      F.add({ items: [title, ...heat(5), ...axisNames, ...drops(inp.n), ...side([['best random: ' + H.f(rb.mean, 3), 'c1'], ['C = ' + H.g(rb.params.C, 3) + ', γ = ' + H.g(rb.params.gamma, 3), 'strong'], ['grid best: ' + H.f(grid.bestScore, 3) + ' in 25 tries', 'strong'], [distinctC + ' distinct C values vs 5', 'tick']])], say: `After ${inp.n} random draws (${inp.n * 5} fits) the best is C = ${H.g(rb.params.C, 3)}, γ = ${H.g(rb.params.gamma, 3)} with ${H.f(rb.mean, 3)}, against ${H.f(grid.bestScore, 3)} for the full grid of 25. Random search tried ${distinctC} different values of C where the grid tried 5.`, meta: { phase: 'randbest', best: rb.params, score: rb.mean, gridScore: grid.bestScore, distinctC, fits: inp.n * 5, cells: 25 }, view });
      F.list.forEach((f) => { f.meta.ctx = { C: CS, gamma: GS }; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Searching', text: ['grid: every combination of fixed lists', '  5 × 5 values × 5 folds = 125 fits', 'random: n draws from distributions', '  log-uniform for scale knobs like C, γ', 'score each by cross-validation on training rows', 'the test set is not used to choose'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'gridbest', pick: 'option', ask: 'Will the best random draw beat the best grid setting?', options: ['yes', 'no', 'tie'], answer: (fr) => { const m = fr[fr.length - 1].meta; return m.score > m.gridScore + 1e-12 ? 'yes' : m.score < m.gridScore - 1e-12 ? 'no' : 'tie'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'A grid of 4 values of C, 5 of γ, with 5-fold CV, fits how many models?', options: ['100', '20', '25', '9'], answer: 0, why: '4 × 5 settings × 5 folds.' },
      { q: 'Why sample C log-uniformly rather than uniformly from 0.01 to 100?', options: ['its effect changes by factors, so each power of ten deserves equal tries', 'it is faster', 'uniform is impossible', 'C must be an integer'], answer: 0, why: 'Uniform sampling would put almost every draw between 10 and 100.' },
      { q: 'When does random search beat grid search for the same budget?', options: ['when only a few knobs really matter', 'never', 'always', 'only with one knob'], answer: 0, why: 'It tries many distinct values of the important knob instead of repeating 5.' },
      { q: 'Very large γ in an RBF SVM tends to…', options: ['overfit: each point influences only its tiny neighbourhood', 'underfit', 'make a straight boundary', 'ignore the data'], answer: 0, why: 'The boundary wraps around individual training points.' },
      { q: 'You pick the best cell of the heat map. Its score is…', options: ['slightly optimistic: it was the best of 25', 'an unbiased test score', 'pessimistic', 'the training score'], answer: 0, why: 'Selection on the same folds favours lucky settings; confirm on held-out data.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
