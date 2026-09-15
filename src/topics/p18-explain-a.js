// Part 18, global explanations: permutation importance (shuffle one column of the test rows and watch the score fall,
// repeated, for every feature, and what a near-copy of a feature does to it), and partial dependence with ICE curves
// (one row's curve as one feature is swept, every row's, their average, and a case where the curves cross).
(function (S) {
  'use strict';
  const P = S.P18, H = () => S.Kit.ml, C = () => S.Kit.curve, E = () => S.Explain, M = () => S.Metrics;

  // ---- data: loans (will the applicant default?), rents, loans with a near-copy of income ----------------------------
  P.TABLES = {
    loans: { label: 'loans: default, a random forest (AUC)', cls: true },
    copy: { label: 'loans with a near-copy of income added', cls: true },
    rents: { label: 'rents: a random-forest regressor (R²)', cls: false },
  };
  P.tableRun = (set) => P.once('tab:' + set, () => {
    const cls = P.TABLES[set].cls, src = set === 'rents' ? S.Data.load('rents') : S.Data.load('loans', { n: 600 });
    let X = src.X.map((r) => r.slice(0, 5)), names = set === 'rents' ? ['area', 'rooms', 'floor', 'distance', 'balcony'] : ['income', 'debt ratio', 'years', 'missed', 'amount'];
    if (set === 'copy') { const r = S.Data.rng(17); X = X.map((x) => x.concat([+(x[0] + r.normal(0, 1.5)).toFixed(1)])); names = names.concat(['income copy']); }
    const sp = S.Data.split(src.y, 0.3, 1, cls), Xtr = sp.train.map((i) => X[i]), ytr = sp.train.map((i) => src.y[i]), Xte = sp.test.map((i) => X[i]), yte = sp.test.map((i) => src.y[i]);
    const model = cls ? new S.ML.RandomForestClassifier({ nEstimators: 40, maxDepth: 8, seed: 1 }).fit(Xtr, ytr) : new S.ML.RandomForestRegressor({ nEstimators: 30, maxDepth: 8, seed: 1 }).fit(Xtr, ytr);
    const score = cls ? (y, p) => M().rocAuc(y, p).value : (y, p) => M().r2(y, p).value;
    return { cls, names, Xtr, ytr, Xte, yte, model, predict: cls ? (Z) => model.predictProba(Z).map((r) => r[1]) : (Z) => model.predict(Z), score, scoreName: cls ? 'AUC' : 'R²' };
  });
  /** the shuffled column of one repeat, drawn from the same seeded stream the importance engine uses */
  P.shuffledColumn = (X, j, rep, reps, seed) => {
    const rnd = S.Data.rng(seed), n = X.length;
    for (let t = 0; t < j * reps + rep; t++) for (let i = n - 1; i > 0; i--) rnd();
    const col = X.map((x) => x[j]);
    for (let i = n - 1; i > 0; i--) { const k = Math.floor(rnd() * (i + 1)); [col[i], col[k]] = [col[k], col[i]]; }
    return col;
  };

  S.topic({
    id: 'permutation-importance', family: 'p18-explain', order: 10, title: 'Permutation importance: shuffle a column, watch the score',
    blurb: 'How much does a trained model rely on a feature? Take the test rows, shuffle that one column so it no longer matches its rows, and score the model again. If the score falls a lot, the model needed that feature; if it does not move, the model ignores it. Repeat the shuffle a few times to see the noise. It works for any model, but beware of near-copies: when two features carry the same information, shuffling one barely hurts, because the model can lean on the other.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data and model', def: 'loans', options: Object.keys(P.TABLES).map((k) => ({ value: k, label: P.TABLES[k].label })) },
      { id: 'reps', kind: 'choice', label: 'Shuffles per feature', def: '5', options: ['3', '5', '10'].map((v) => ({ value: v, label: v })) },
      { id: 'seed', kind: 'number', label: 'Seed (the shuffles)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Loans', values: { set: 'loans', reps: '5', seed: 1 } },
      { label: 'Rents', values: { set: 'rents', reps: '5', seed: 1 } },
      { label: 'A near-copy of income', values: { set: 'copy', reps: '5', seed: 1 } },
      { label: 'Ten shuffles', values: { set: 'loans', reps: '10', seed: 3 } },
    ],
    counters: [{ id: 'shuffles', label: 'shuffles scored' }],
    run(inp) {
      const R = P.tableRun(inp.set), reps = +inp.reps, F = H().frames(['shuffles']), q = F.q;
      const imp = P.once(['pi', inp.set, reps, inp.seed].join(':'), () => E().permutationImportance(R.predict, R.Xte, R.yte, { nRepeats: reps, seed: inp.seed, score: R.score, names: R.names }));
      const d = R.names.length, top = imp.order[0], show = 7;
      const target = R.cls ? 'default' : 'rent';
      const cols = [{ name: '', w: 34, cls: 'idx' }].concat(R.names.map((nm) => ({ name: nm.length > 10 ? nm.slice(0, 10) : nm, w: nm === 'income copy' ? 92 : 78 }))).concat([{ name: target, w: 64 }]);
      const fmtv = (v) => String(v);
      const table = (colJ, colVals, lit) => S.Kit.dt.table({ prefix: 'pi:t', x: 20, y: 60, cols, rows: R.Xte.slice(0, show).map((x, i) => [String(i)].concat(x.map((v, j) => fmtv(j === colJ ? colVals[i] : v))).concat([String(R.yte[i])])), rowH: 26, size: 12, cls: (i, j) => (lit && j === colJ + 1 ? 'p18shuf' : '') }).items;
      F.push([...table(-1), H().head('pi:h', 20, 34, 'The test rows (first ' + show + ' of ' + R.yte.length + ')')],
        `A random forest was trained on ${P.qpl(q, R.ytr.length, 'row')}. On the ${q.n(R.yte.length)} test rows, with every column intact, its ${R.scoreName} is ${q(imp.baseline, 3)}. That is the score to beat.`, { phase: 'base', nTr: R.ytr.length, nTe: R.yte.length, baseline: imp.baseline });
      // one shuffle of the top feature
      const col = P.shuffledColumn(R.Xte, top, 0, reps, inp.seed), after = R.score(R.yte, R.predict(R.Xte.map((x, i) => { const z = x.slice(); z[top] = col[i]; return z; })));
      F.count('shuffles');
      F.push([...table(top, col, true), H().head('pi:h', 20, 34, 'Column “' + R.names[top] + '” shuffled')],
        `Shuffle the ${R.names[top]} column: each row now carries some other row’s value (shaded), so the column says nothing about its own row while its values stay realistic. The ${R.scoreName} falls to ${q(after, 3)}, a drop of ${q(imp.drops[top][0], 3)}.`, { phase: 'shuffle', j: top, after, drop: imp.drops[top][0], col });
      // the repeats
      const all = imp.drops.flat(), lo = Math.min(0, ...all), hi = Math.max(...all, 1e-3);
      const ax = C().axes({ prefix: 'pi:ax', x: 150, y: 70, w: 520, h: d * 40, xr: [lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo)], yr: [0, d], yticks: 0, xname: 'drop in ' + R.scoreName + ' when shuffled', tickSize: 12 });
      const rowY = (k) => ax.Y(d - k - 0.5);
      const rowsOf = (upto, order) => {
        const it = [...ax.items, C().vline(ax, 'pi:z', 0, '')];
        order.forEach((j, k) => {
          it.push(H().text('pi:n' + j, 140, rowY(k) + 4.5, R.names[j], k === 0 ? 'strong' : 'legend', 13, 'end'));
          if (k > upto) return;
          const m = imp.mean[j], s = imp.std[j];
          if (Math.abs(ax.X(m) - ax.X(0)) > 0.5) it.push({ key: 'pi:b' + j, kind: 'cell', x: Math.min(ax.X(0), ax.X(m)), y: P.r2(rowY(k) - 9), w: P.r2(Math.abs(ax.X(m) - ax.X(0))), h: 18, rx: 2, text: '', cls: 'bar ' + (k === 0 ? 'p18top' : 'p18imp'), audit: 'ignore' });
          it.push({ key: 'pi:e' + j, kind: 'path', d: `M${ax.X(m - s)},${rowY(k)} H${ax.X(m + s)}`, cls: 'p18err', audit: 'ignore', layer: 'items' });
          imp.drops[j].forEach((v, r) => it.push({ key: 'pi:d' + j + '_' + r, kind: 'dot', x: ax.X(v), y: P.r2(rowY(k) + 13), r: 2.6, cls: 'sp cn', audit: 'overlap-ok', layer: 'items' }));
        });
        return it;
      };
      F.count('shuffles', reps - 1);
      F.push([...rowsOf(0, imp.order), H().head('pi:h', 20, 34, R.names[top] + ': ' + reps + ' shuffles')],
        `Each shuffle is a different random order, so the drop varies: over ${P.qpl(q, reps, 'shuffle')} of ${R.names[top]} the drops run from ${q(Math.min(...imp.drops[top]), 3)} to ${q(Math.max(...imp.drops[top]), 3)}. Their mean, ${q(imp.mean[top], 3)}, is the importance (the bar); the spread, ${q(imp.std[top], 3)}, says how sure it is.`, { phase: 'repeats', j: top, min: Math.min(...imp.drops[top]), max: Math.max(...imp.drops[top]), mean: imp.mean[top], std: imp.std[top] });
      F.count('shuffles', reps * (d - 1));
      const last = imp.order[d - 1];
      F.push([...rowsOf(d, imp.order), H().head('pi:h', 20, 34, 'Every feature, ' + reps + ' shuffles each')],
        `All ${P.qpl(q, d, 'feature')}: ${R.names[top]} matters most (${q(imp.mean[top], 3)}), then ${R.names[imp.order[1]]} (${q(imp.mean[imp.order[1]], 3)}); ${R.names[last]} barely matters (${q(imp.mean[last], 3)})` + (imp.mean[last] < 0 ? ', and a negative drop only means a shuffle happened to help by chance.' : '.') + ' This says what this model uses, not what causes the outcome.', { phase: 'all', mean: imp.mean, order: imp.order, d });
      if (inp.set === 'copy') {
        const plain = P.once(['pi', 'loans', reps, inp.seed].join(':'), () => E().permutationImportance(P.tableRun('loans').predict, P.tableRun('loans').Xte, P.tableRun('loans').yte, { nRepeats: reps, seed: inp.seed, score: P.tableRun('loans').score }));
        F.push([...rowsOf(d, imp.order), H().head('pi:h', 20, 34, 'The near-copy caveat')],
          `Without the copy, income alone scored ${q(plain.mean[0], 3)}. With a noisy copy beside it, income scores ${q(imp.mean[0], 3)} and the copy ${q(imp.mean[5], 3)}: the forest splits its trust between them, and shuffling either one leaves the other to fall back on. Correlated features each look less important than the information they share.`, { phase: 'copy', plain: plain.mean[0], income: imp.mean[0], copy: imp.mean[5] });
      }
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Permutation importance', text: ['s₀ = score(model, X_test, y_test)', 'for feature j, repeat r = 1..R:', '  X′ = X_test with column j shuffled', '  drop_jr = s₀ − score(model, X′, y_test)', 'importance_j = mean_r drop_jr  (± sd)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'base', pick: 'option', once: true,
      ask: 'Which feature will the model miss most when shuffled?',
      options: (fr, i, inp) => P.tableRun(inp.set).names,
      answer: (fr, i, inp) => P.tableRun(inp.set).names[fr.find((f) => f.meta.phase === 'all').meta.order[0]],
      explain: (fr) => fr.find((f) => f.meta.phase === 'all').say,
    }],
    quiz: [
      { q: 'Shuffling a column leaves the score unchanged. This means…', options: ['the model does not use that feature', 'the feature causes nothing', 'the feature is noise in the world', 'the model is wrong'], answer: 0, why: 'Importance is about this model’s use of the feature.' },
      { q: 'Why shuffle the column rather than set it to zero?', options: ['shuffled values stay realistic; zeros may be impossible inputs', 'zeros are slower', 'shuffling is exact', 'there is no difference'], answer: 0, why: 'The model is only scored on values that look like real data.' },
      { q: 'Two features are near-copies. Their permutation importances will tend to be…', options: ['both lower than the information they share', 'both doubled', 'exactly equal to one feature alone', 'negative'], answer: 0, why: 'Shuffling one leaves the other to carry the information.' },
      { q: 'Why compute importance on the test rows rather than the training rows?', options: ['so it reflects what generalises, not what was memorised', 'it is faster', 'the training rows have no labels', 'it must be the training rows'], answer: 0, why: 'An overfitted model can lean on noise that only helps on its own training rows.' },
    ],
  });

  // ---- partial dependence and ICE -------------------------------------------------------------------------------------
  const twist = (n, seed) => { const r = S.Data.rng(seed), X = [], y = []; for (let i = 0; i < n; i++) { const a = r.range(-1, 1), b = r.range(-1, 1); X.push([+a.toFixed(3), +b.toFixed(3)]); y.push(+(2 * a * b + r.normal(0, 0.1)).toFixed(3)); } return { X, y }; };
  P.PDCASES = {
    area: { label: 'rents: floor area', set: 'rents', j: 0, unit: 'm²' },
    distance: { label: 'rents: distance to the centre', set: 'rents', j: 3, unit: 'km' },
    rain: { label: 'harvest: rain', set: 'harvest', j: 0, unit: 'mm' },
    twist: { label: 'a toy with an interaction: y = 2 x₁ x₂', set: 'twist', j: 0, unit: '' },
  };
  P.pdModel = (set) => P.once('pdm:' + set, () => {
    const d = set === 'twist' ? Object.assign({ names: ['x₁', 'x₂'], target: 'y' }, twist(300, 5)) : (() => { const t = S.Data.load(set); return { X: t.X, y: t.y, names: t.features.map((f) => f.name), target: t.target.name }; })();
    const m = new S.ML.RandomForestRegressor({ nEstimators: 30, maxDepth: 8, seed: 1 }).fit(d.X, d.y);
    return Object.assign(d, { predict: (Z) => m.predict(Z) });
  });

  S.topic({
    id: 'partial-dependence', family: 'p18-explain', order: 20, title: 'Partial dependence and ICE curves',
    blurb: 'How does the model’s output change as one feature changes, everything else as it is? Take one row and sweep that feature across its range, asking the model each time: that is the row’s ICE curve (individual conditional expectation). Do it for many rows and average the curves: that is partial dependence. When the ICE curves run parallel the average tells the story; when they fan out or cross, the feature works together with others and the average can hide it completely.',
    inputs: [
      { id: 'case', kind: 'choice', label: 'Feature', def: 'area', options: Object.keys(P.PDCASES).map((k) => ({ value: k, label: P.PDCASES[k].label })) },
      { id: 'rows', kind: 'choice', label: 'Rows (ICE curves)', def: '30', options: ['15', '30', '50'].map((v) => ({ value: v, label: v })) },
    ],
    presets: [
      { label: 'Rents: area', values: { case: 'area', rows: '30' } },
      { label: 'Rents: distance', values: { case: 'distance', rows: '30' } },
      { label: 'Harvest: rain (a hump)', values: { case: 'rain', rows: '30' } },
      { label: 'An interaction hides', values: { case: 'twist', rows: '30' } },
    ],
    counters: [{ id: 'curves', label: 'ICE curves' }],
    run(inp) {
      const c = P.PDCASES[inp.case], D = P.pdModel(c.set), j = c.j, n = +inp.rows, F = H().frames(['curves']), q = F.q;
      const X = D.X.slice(0, n), pd = P.once(['pd', inp.case, n].join(':'), () => E().partialDependence(D.predict, X, j, { names: D.names }));
      const g = pd.grid, ice = pd.ice, av = pd.average, nm = D.names[j];
      const ys = ice.flat().concat(D.y.slice(0, 200)), ylo = Math.min(...ys), yhi = Math.max(...ys), pad = 0.05 * (yhi - ylo);
      const xs = D.X.slice(0, 200).map((x) => x[j]).concat(g), xlo = Math.min(...xs), xhi = Math.max(...xs);
      const ax = C().axes({ prefix: 'pd:ax', x: 70, y: 76, w: 560, h: 320, xr: [xlo, xhi], yr: [ylo - pad, yhi + pad], xname: nm + (c.unit ? ' (' + c.unit + ')' : ''), yname: 'predicted ' + D.target, tickSize: 12 });
      const dots = D.X.slice(0, 200).map((x, i) => ({ key: 'pd:p' + i, kind: 'dot', x: ax.X(x[j]), y: ax.Y(D.y[i]), r: 2.6, cls: 'sp cn', audit: 'overlap-ok', layer: 'items', opacity: 0.55 }));
      F.push([...ax.items, ...dots, H().head('pd:h', 70, 26, 'The data: ' + D.target + ' against ' + nm)],
        `${P.qpl(q, D.y.length, 'row')}; a random forest of ${q.n(30)} trees learns ${D.target} from all ${P.qpl(q, D.names.length, 'feature')}. The dots are the data against ${nm} alone; the other features scatter them.`, { phase: 'data', n: D.y.length, trees: 30, d: D.names.length });
      // one row
      const r0 = 0, cur = X[r0][j], curPred = D.predict([X[r0]])[0];
      const line = (i, cls) => C().line(ax, 'pd:i' + i, g.map((u, k) => [u, ice[i][k]]), cls);
      F.count('curves');
      F.push([...ax.items, line(r0, 'p18one'), { key: 'pd:me', kind: 'dot', x: ax.X(cur), y: ax.Y(curPred), r: 6, cls: 'mark p18pick', point: true }, H().head('pd:h', 70, 26, 'One row, ' + nm + ' swept')],
        `Take row ${q.n(r0)} (its ${nm} is ${q.sg(cur, 3)}, where the model says ${q.sg(curPred, 3)}, the green dot). Keep its other features and set ${nm} to each of ${P.qpl(q, g.length, 'value')} from ${q.sg(g[0], 3)} to ${q.sg(g[g.length - 1], 3)}: the model’s answers trace its ICE curve, from ${q.sg(ice[r0][0], 3)} to ${q.sg(ice[r0][g.length - 1], 3)}.`, { phase: 'one', r0, cur, curPred, g0: g[0], g1: g[g.length - 1], a: ice[r0][0], b: ice[r0][g.length - 1], ng: g.length });
      F.set('curves', n);
      const lines = ice.map((_, i) => line(i, 'p18ice'));
      F.push([...ax.items, ...lines, H().head('pd:h', 70, 26, n + ' ICE curves')],
        `Every one of the ${P.qpl(q, n, 'row')} gets its own curve. At one value of ${nm} the curves are up to ${q.sg(pd.trace[1].spread, 3)} apart: the rows differ in their other features, and ` + (inp.case === 'twist' ? 'here the curves even slope in opposite directions.' : 'mostly the curves are shifted copies of one shape.'), { phase: 'ice', spread: pd.trace[1].spread, n });
      const lo = av.indexOf(Math.min(...av)), hi = av.indexOf(Math.max(...av));
      // values in the hundreds and more as whole numbers, and the rise as the difference of the numbers printed
      const big = (v) => Math.abs(v) >= 100, fv = (v) => (big(v) ? q(v, 0) : q.sg(v, 3)), fd = (a, b) => (big(a) || big(b) ? q(Math.round(a) - Math.round(b), 0) : q.sg(a - b, 3));
      F.push([...ax.items, ...lines, C().line(ax, 'pd:avg', g.map((u, k) => [u, av[k]]), 'p18pdp'), H().head('pd:h', 70, 26, 'Partial dependence: the average curve')],
        `Averaging the curves gives the partial dependence (thick): lowest ${fv(av[lo])} at ${nm} = ${q.sg(g[lo], 3)}, highest ${fv(av[hi])} at ${q.sg(g[hi], 3)}.` + (inp.case === 'twist' ? ` Its whole rise is only ${fd(av[hi], av[lo])}, nearly flat, although each row’s curve moves a lot: x₁ matters, but up for rows with x₂ above zero and down for the others, and the average cancels the two. Look at the ICE curves before trusting a flat average.` : ` Over this range the model’s answer moves by ${fd(av[hi], av[lo])} on average.`), { phase: 'pdp', lo: av[lo], hi: av[hi], glo: g[lo], ghi: g[hi], rise: av[hi] - av[lo], risePrinted: Math.round(av[hi]) - Math.round(av[lo]), maxSpan: Math.max(...ice.map((r) => Math.max(...r) - Math.min(...r))) });
      for (const f of F.list) f.narrowView = null;
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'ICE and partial dependence', text: ['ICE of row i: v ↦ f(xᵢ with feature j set to v)', 'PD(v) = (1/n) Σᵢ f(xᵢ with feature j set to v)', 'parallel ICE curves: no interaction', 'crossing curves: feature j interacts', 'beware: v may be unrealistic for some rows'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'one', pick: 'option', once: true,
      ask: 'Averaged over many rows, will the curve go up, go down, or stay nearly flat?',
      options: ['up', 'down', 'nearly flat'],
      answer: (fr) => { const g = fr.find((f) => f.meta.phase === 'pdp').meta; return g.rise < 0.25 * g.maxSpan ? 'nearly flat' : g.ghi > g.glo ? 'up' : 'down'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'pdp').say,
    }],
    quiz: [
      { q: 'An ICE curve shows…', options: ['one row’s prediction as one feature is swept', 'the average over all rows', 'the data points', 'the importance of a feature'], answer: 0, why: 'Individual: one row, the others features held fixed.' },
      { q: 'Partial dependence is…', options: ['the average of the ICE curves', 'the steepest ICE curve', 'the correlation of feature and target', 'the model’s accuracy'], answer: 0, why: 'PD(v) averages f over the rows with the feature set to v.' },
      { q: 'ICE curves cross each other. This suggests…', options: ['the feature interacts with others', 'the model is linear', 'the feature is unused', 'there is too little data'], answer: 0, why: 'The effect of the feature depends on the other features.' },
      { q: 'Why can partial dependence mislead with strongly correlated features?', options: ['it sets the feature to values that rarely occur with the other features', 'it is too slow', 'it ignores the model', 'it needs labels'], answer: 0, why: 'A big flat with one room is an impossible row, yet it is averaged in.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
