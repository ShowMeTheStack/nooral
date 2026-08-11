// Part 1, feature selection: real columns and pure-noise columns side by side, scored three ways. A filter scores
// each column alone (its correlation with the label); forward selection adds, one step at a time, the column that
// raises the cross-validated score most; an embedded method reads the choice off a fitted model (lasso weights that
// reach exactly zero, or a tree's importances).
(function (S) {
  'use strict';
  const H = S.P01, DT = () => S.Kit.dt, ML = () => S.ML;
  const SETS = {
    rents: { label: 'Flat rents (regression)' },
    cancer: { label: 'Breast cancer, 10 mean measurements (two classes)' },
  };
  function selData(key, noise) {
    const r = H.rng(77);
    let X, y, names, task;
    if (key === 'rents') { const d = H.data('rents', { n: 200 }); X = d.X.map((q) => q.slice()); y = d.y; names = d.features.map(H.short); task = 'reg'; }
    else { const d = H.data('breastCancer'); X = d.X.map((q) => q.slice(0, 10)); y = d.y; names = d.features.slice(0, 10).map((f) => H.short(f).replace(/^mean /, '')); task = 'cls'; }
    for (let k = 0; k < noise; k++) { X.forEach((q) => q.push(+r.normal(0, 1).toFixed(3))); names.push('noise ' + (k + 1)); }
    return { X, y, names, task, real: names.length - noise };
  }

  S.topic({
    id: 'data-select', family: 'p01-engineer', order: 30, title: 'Feature selection',
    blurb: 'Not every column helps: some carry no information about the label, and each extra column gives a model one more way to fit noise. A filter scores each column on its own, quickly, but cannot see redundancy or teamwork. Forward selection builds a set greedily, adding whichever column most improves the cross-validated score, which is slower but judges columns together. Embedded methods let the model choose while it fits: lasso drives useless weights to exactly zero, and a tree only splits on columns that help.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'rents', options: Object.entries(SETS).map(([value, s]) => ({ value, label: s.label })) },
      { id: 'noise', kind: 'number', label: 'Noise columns added', def: 5, min: 0, max: 8 },
      { id: 'k', kind: 'number', label: 'Columns to keep', def: 3, min: 1, max: 5 },
    ],
    presets: [
      { label: 'Rents with 5 noise columns', values: { data: 'rents', noise: 5, k: 3 } },
      { label: 'Rents, keep 5', values: { data: 'rents', noise: 5, k: 5 } },
      { label: 'Cancer measurements', values: { data: 'cancer', noise: 3, k: 3 } },
      { label: 'Cancer, no noise, keep 2', values: { data: 'cancer', noise: 0, k: 2 } },
    ],
    counters: [{ id: 'kept', label: 'columns kept' }, { id: 'fits', label: 'models fitted' }],
    run(inp) {
      const D = selData(inp.data, inp.noise), X = D.X, y = D.y, d = D.names.length, ml = ML(), k = Math.min(inp.k, d);
      const isReg = D.task === 'reg', scoring = isReg ? 'r2' : 'accuracy', what = isReg ? 'R²' : 'accuracy';
      const est = isReg ? ml.LinearRegression() : ml.LogisticRegression({ C: 1 });
      const Z = H.scaler(X, 'standard').apply(X);
      const take = (cols) => Z.map((q) => cols.map((j) => q[j]));
      let fits = 0;
      const cv = (cols) => { fits += 5; return H.mean(ml.crossValScore(est, take(cols), y, { cv: 5, scoring })); };
      // filter: |r| of each column with the label
      const filt = D.names.map((_, j) => Math.abs(H.pearson(X.map((q) => q[j]), y).r));
      const filtPick = filt.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k).map(([, j]) => j);
      const F = new S.Frames(['kept', 'fits']), view = H.box(-14, -8, 790, 470);
      const rowH = Math.min(30, 380 / d);
      const cls0 = (j) => (j >= D.real ? 'muted' : '');
      const bars = (prefix, vals, lit, fmt, max, x, w, title, labels) => [H.title(prefix + 't', x, 34, title)].concat(DT().hbars({ prefix, x, y: 46, w, rowH, vals, names: D.names, max, labels, fmt: fmt || ((v) => H.f(v, 3)), nameW: 118, valW: 56, size: rowH < 22 ? 12 : 13, nameCls: (j) => (lit.includes(j) ? 'act' : cls0(j)), cls: (j) => (lit.includes(j) ? 'act' : j >= D.real ? 'soft' : '') }));
      const title = H.title('tt', 0, 6, SETS[inp.data].label + ': ' + D.real + ' real columns' + (inp.noise ? ' and ' + inp.noise + ' of pure noise' : ''));
      const side = (lines) => lines.map((l, i) => H.text('sd' + i, 470, 70 + i * 26, l[0], l[1] || 'strong', 14));
      const all = D.names.map((_, j) => j), cvAll = cv(all);
      F.set('kept', k).set('fits', fits);
      F.add({ items: [title, ...bars('fl', filt, filtPick, null, 1, 0, 440, 'filter: |r| with the label'), ...side([['filter keeps the top ' + k + ':', 'act'], [filtPick.map((j) => D.names[j]).join(', '), 'strong'], ['all ' + d + ' columns: CV ' + what + ' ' + H.f(cvAll, 3), 'tick']])], say: `A filter scores each column alone by its correlation with the label: ${H.list(filtPick.map((j) => D.names[j] + ' ' + H.f(filt[j], 3)))} come top.${inp.noise ? ' The noise columns score at most ' + H.f(Math.max(...filt.slice(D.real)), 3) + '.' : ''} With all ${d} columns the 5-fold cross-validated ${what} is ${H.f(cvAll, 3)}.`, meta: { phase: 'filter', filt, pick: filtPick, cvAll, names: D.names }, view });
      // wrapper: forward selection
      const chosen = [], path = [];
      for (let s = 0; s < k; s++) {
        const cand = all.filter((j) => !chosen.includes(j)).map((j) => [cv(chosen.concat([j])), j]);
        cand.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
        const [bs, bj] = cand[0];
        const scores = all.map((j) => { const c = cand.find((q) => q[1] === j); return c ? Math.max(0, c[0]) : 0; });
        const labels = all.map((j) => { const c = cand.find((q) => q[1] === j); return c ? H.f(c[0], 3) : 'chosen'; });
        chosen.push(bj); path.push(bs);
        F.set('fits', fits);
        F.add({ items: [title, ...bars('fw', scores, chosen, null, Math.max(...scores, 1e-9), 0, 440, 'forward step ' + (s + 1) + ': CV ' + what + ' with each added', labels), ...side([['chosen so far:', 'act'], [chosen.map((j) => D.names[j]).join(', '), 'strong'], ['CV ' + what + ' ' + H.f(bs, 3), 'good'], [cand.length + ' candidates × 5 folds', 'tick']])], say: `Forward step ${s + 1}: each of the ${cand.length} remaining columns is tried ${chosen.length > 1 ? 'beside ' + H.list(chosen.slice(0, -1).map((j) => D.names[j])) : 'on its own'}, each with 5-fold cross-validation. Adding ${D.names[bj]} gives the best ${what}, ${H.f(bs, 3)}` + (s ? `, up from ${H.f(path[s - 1], 3)}.` : '.'), meta: { phase: 'forward', step: s + 1, pick: bj, score: bs, scores: cand.map((q) => q.slice()) }, view });
      }
      // embedded
      let emb, embName, embPick;
      if (isReg) {
        const alpha = 8, las = ml.Lasso({ alpha }).fit(Z, y);
        emb = las.coef.map(Math.abs); embName = 'lasso |weights| (α = ' + alpha + ', scaled columns)';
        embPick = emb.map((v, j) => [v, j]).filter(([v]) => v > 0).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k).map(([, j]) => j);
        F.add({ items: [title, ...bars('em', emb, embPick, (v) => (v === 0 ? '0' : H.g(v, 3)), Math.max(...emb, 1e-9), 0, 440, embName), ...side([['lasso sets ' + emb.filter((v) => v === 0).length + ' of ' + d + ' weights to 0', 'act'], ['largest kept: ' + embPick.map((j) => D.names[j]).join(', '), 'strong']])], say: `Lasso adds α·Σ|w| to the squared error. With α = ${alpha} on the scaled columns it sets ${emb.filter((v) => v === 0).length} of the ${d} weights to exactly 0` + (inp.noise ? `, including ${emb.slice(D.real).filter((v) => v === 0).length} of the ${inp.noise} noise columns` : '') + `; the largest remaining are ${H.list(embPick.map((j) => D.names[j] + ' ' + H.g(emb[j], 3)))}.`, meta: { phase: 'embedded', emb, pick: embPick, zeros: emb.filter((v) => v === 0).length }, view });
      } else {
        const tree = ml.DecisionTreeClassifier({ maxDepth: 4, seed: 0 }).fit(X, y);
        emb = tree.featureImportances; embName = 'tree importances (depth 4)';
        embPick = emb.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k).map(([, j]) => j);
        F.add({ items: [title, ...bars('em', emb, embPick, (v) => H.f(v, 3), Math.max(...emb, 1e-9), 0, 440, embName), ...side([['the tree used ' + emb.filter((v) => v > 0).length + ' of ' + d + ' columns', 'act'], ['top: ' + embPick.map((j) => D.names[j]).join(', '), 'strong']])], say: `A depth-4 tree splits only on columns that reduce impurity; each column's share of the total reduction is its importance. It used ${emb.filter((v) => v > 0).length} of the ${d} columns, led by ${H.list(embPick.map((j) => D.names[j] + ' ' + H.f(emb[j], 3)))}.`, meta: { phase: 'embedded', emb, pick: embPick, used: emb.filter((v) => v > 0).length }, view });
      }
      const sets = [['all ' + d, all], ['filter', filtPick], ['forward', chosen], [isReg ? 'lasso' : 'tree', embPick]];
      const scores = sets.map(([, c]) => (c.length ? cv(c) : NaN));
      F.set('fits', fits);
      const rows = sets.map(([nm, c], i) => [nm, String(c.length), H.f(scores[i], 3)]);
      const tb = DT().table({ prefix: 'cmp', x: 0, y: 50, cols: [{ name: 'columns', w: 110 }, { name: 'kept', w: 60, cls: 'num' }, { name: 'CV ' + what, w: 110, cls: 'num' }], rows, rowH: 30, size: 14 });
      const lists = sets.slice(1).map(([nm, c], i) => H.text('ls' + i, 330, 98 + (i + 1) * 32, nm + ': ' + c.map((j) => D.names[j]).join(', '), 'strong', 13));
      F.add({ items: [title, ...tb.items, ...lists, H.title('cmpT', 330, 104, 'columns chosen')], say: `With ${k} columns: the filter's set scores ${H.f(scores[1], 3)}, forward selection's ${H.f(scores[2], 3)}, ${isReg ? 'lasso' : 'the tree'}'s ${H.f(scores[3], 3)}, against ${H.f(scores[0], 3)} with all ${d}. Note that the forward score was used to choose the columns, so it is a little optimistic; an honest number needs the selection inside each fold.`, meta: { phase: 'compare', scores }, view });
      const ctx = { d, path, cvAll };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Three ways', text: ['filter: score each column alone (|r|, F, MI)', '  fast; blind to redundancy and teamwork', 'wrapper: forward selection by CV score', '  slow: about k·d models, each cross-validated', 'embedded: lasso zeros, tree importances', 'select inside the training folds only'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'filter', pick: 'option', ask: 'Which column will forward selection add first?', options: (fr) => { const m = fr[0].meta, ids = [...new Set([fr[1].meta.pick].concat(m.pick, m.filt.map((v, j) => [v, j]).sort((p, q) => p[0] - q[0]).map(([, j]) => j)))].slice(0, 4); return ids.sort((p, q) => p - q).map((j) => m.names[j]); }, answer: (fr) => fr[0].meta.names[fr[1].meta.pick], explain: (fr) => fr[1].say },
    ],
    quiz: [
      { q: 'Two columns are near copies of each other and both correlate strongly with the label. A filter…', options: ['keeps both, though the second adds little', 'keeps neither', 'keeps exactly one', 'cannot score them'], answer: 0, why: 'It scores each column alone, so redundancy is invisible to it.' },
      { q: 'Forward selection on 20 columns, keeping 5, fits about how many cross-validated models?', options: ['20 + 19 + 18 + 17 + 16 = 90', '5', '20', '2²⁰'], answer: 0, why: 'Each step tries every remaining column.' },
      { q: 'Why does lasso select features while ridge does not?', options: ['its |w| penalty pushes small weights to exactly zero', 'it uses fewer rows', 'ridge has no penalty', 'lasso ignores the label'], answer: 0, why: 'The corner of the |w| penalty at zero catches weights; the w² penalty only shrinks them.' },
      { q: 'Choosing columns on all rows and then cross-validating the chosen set…', options: ['gives an optimistic score: the choice saw the test folds', 'is the honest way', 'makes no difference ever', 'is impossible'], answer: 0, why: 'Selection is part of training and must happen inside each training fold.' },
      { q: 'A pure-noise column gets a small but nonzero filter score. Why?', options: ['by chance it correlates a little with the label in a finite sample', 'it is secretly useful', 'the filter is broken', 'noise always has r = 0.5'], answer: 0, why: 'Sample correlations of unrelated columns scatter around 0, roughly ±1/√n.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
