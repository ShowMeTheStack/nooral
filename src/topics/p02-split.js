// Part 2, splitting: every row a small cell coloured by its class, shuffled, then cut into training, validation and
// test blocks, with each block's class shares (with and without stratification); and k-fold cross-validation drawn
// as k rows of the same cells, each fold's test cells lit and scored.
(function (S) {
  'use strict';
  const H = S.P01, DT = () => S.Kit.dt, ML = () => S.ML;
  const SETS = {
    pumps: { label: 'Pump sensors: 5% faults', load: () => { const d = H.data('pumps', { missing: 0, n: 200 }); return { y: d.y, classes: ['normal', 'fault'] }; } },
    loans: { label: 'Small loans: about a quarter default', load: () => { const d = H.data('loans', { n: 200 }); return { y: d.y, classes: ['repaid', 'defaulted'] }; } },
    flowers: { label: 'Wildflowers: three classes', load: () => { const d = H.data('wildflowers'); return { y: d.y, classes: d.classes }; } },
  };
  const CC = ['cA', 'cB', 'cC'];

  S.topic({
    id: 'eval-split', family: 'p02-split', order: 10, title: 'Training, validation and test sets',
    blurb: 'A model is judged on rows it never saw. Shuffle the rows, then cut them: a training set to fit on, a validation set to choose settings with, and a test set kept locked until the very end and used once. A plain random cut can leave a rare class badly under-represented in a small part; a stratified cut shuffles each class separately so every part keeps the class shares of the whole.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'pumps', options: Object.entries(SETS).map(([value, s]) => ({ value, label: s.label })) },
      { id: 'val', kind: 'slider', label: 'Validation share', def: 0.2, min: 0, max: 0.3, step: 0.05, fmt: (v) => Math.round(v * 100) + '%' },
      { id: 'test', kind: 'slider', label: 'Test share', def: 0.2, min: 0.1, max: 0.4, step: 0.05, fmt: (v) => Math.round(v * 100) + '%' },
      { id: 'strat', kind: 'choice', label: 'Stratify', def: 'no', options: [{ value: 'no', label: 'no: one shuffle' }, { value: 'yes', label: 'yes: by class' }] },
      { id: 'seed', kind: 'number', label: 'Seed', def: 2, min: 0, max: 99 },
    ],
    presets: [
      { label: 'Rare faults, plain shuffle', values: { data: 'pumps', strat: 'no', seed: 2 } },
      { label: 'Rare faults, stratified', values: { data: 'pumps', strat: 'yes', seed: 2 } },
      { label: 'Loans, 60 / 20 / 20', values: { data: 'loans', val: 0.2, test: 0.2, strat: 'yes' } },
      { label: 'Three classes, no validation set', values: { data: 'flowers', val: 0, test: 0.3, strat: 'no', seed: 5 } },
    ],
    counters: [{ id: 'tr', label: 'training' }, { id: 'va', label: 'validation' }, { id: 'te', label: 'test' }],
    run(inp) {
      const D = SETS[inp.data].load(), y = D.y, n = y.length, ml = ML(), strat = inp.strat === 'yes', K = D.classes.length;
      const split = (st) => {
        const a = ml.trainTestSplit(n, { testSize: inp.test, stratify: st ? y : null, seed: inp.seed });
        const rest = a.train.slice();
        let tr = rest, va = [];
        if (inp.val > 0) { const b = ml.trainTestSplit(rest.length, { testSize: inp.val / (1 - inp.test), stratify: st ? rest.map((i) => y[i]) : null, seed: inp.seed + 1 }); tr = b.train.map((k) => rest[k]); va = b.test.map((k) => rest[k]); }
        return { tr, va, te: a.test };
      };
      const sp = split(strat), other = split(!strat);
      const share = (idx) => D.classes.map((_, c) => (idx.length ? idx.filter((i) => y[i] === c).length / idx.length : 0));
      const counts = (idx) => D.classes.map((_, c) => idx.filter((i) => y[i] === c).length);
      const all = y.map((_, i) => i), cw = 30, ch = 16, per = 20, gap = 3;
      const at = (k, line0) => ({ x: (k % per) * (cw + 2), y: 40 + (line0 + Math.floor(k / per)) * (ch + gap) });
      const cells = (order, starts, cls) => order.map((i, k) => { const s = starts ? starts(i, k) : at(k, 0); return { key: 'r' + i, kind: 'cell', x: s.x, y: s.y, w: cw, h: ch, rx: 2, text: '', cls: 'split ' + (cls ? cls(i) : CC[y[i]]), audit: 'ignore' }; });
      const F = new S.Frames(['tr', 'va', 'te']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, SETS[inp.data].label + ': ' + n + ' rows');
      const legend = D.classes.map((c, k) => [{ key: 'lg' + k, kind: 'cell', x: 500 + k * 96, y: -2, w: 14, h: 14, rx: 2, text: '', cls: 'split ' + CC[k], audit: 'ignore' }, H.text('lgt' + k, 518 + k * 96, 10, c, 'legend', 13)]).flat();
      const byClass = all.slice().sort((a, b) => y[a] - y[b] || a - b);
      const c0 = counts(all);
      F.add({ items: [title, ...legend, ...cells(byClass)], say: `${n} rows, one cell each, grouped by class: ${H.list(D.classes.map((c, k) => c + ' ' + c0[k] + ' (' + H.pct(c0[k] / n) + ')'))}.`, meta: { phase: 'classes', counts: c0 }, view });
      const shuf = ml.rng(inp.seed + 7).perm(n);
      F.add({ items: [title, ...legend, ...cells(shuf)], say: 'Shuffle first: data often arrives in an order (by date, by class, by source), and a cut through ordered rows gives parts that differ for no good reason.', meta: { phase: 'shuffle' }, view });
      // the three blocks, each starting on its own line
      const parts = [['tr', 'training', sp.tr], ['va', 'validation', sp.va], ['te', 'test', sp.te]].filter((p) => p[2].length);
      const place = new Map();
      let line = 0;
      const heads = [];
      parts.forEach(([k, nm, idx]) => {
        idx.forEach((i, j) => place.set(i, at(j, line)));
        heads.push({ key: 'pb' + k, kind: 'text', x: per * (cw + 2) + 10, y: 40 + line * (ch + gap) + 12, text: nm + ' ' + idx.length, size: 14, cls: k === 'te' ? 'hot' : k === 'va' ? 'c2' : 'c0' });
        line += Math.ceil(idx.length / per) + 1;
      });
      F.set('tr', sp.tr.length).set('va', sp.va.length).set('te', sp.te.length);
      const cut = (cls) => cells(all, (i) => place.get(i), cls);
      F.add({ items: [title, ...legend, ...cut(), ...heads], say: `Cut ${strat ? 'class by class (stratified)' : 'the shuffled rows'} into ${H.list(parts.map(([, nm, idx]) => idx.length + ' ' + nm))} rows.` + (sp.va.length ? ' Settings are chosen on validation; the test rows wait, locked.' : ' With no validation set, settings must be chosen by cross-validation on the training rows.'), meta: { phase: 'cut', tr: sp.tr.length, va: sp.va.length, te: sp.te.length }, view });
      const rare = c0.indexOf(Math.min(...c0));
      const sh = parts.map(([k, nm, idx]) => [nm, share(idx), counts(idx)]);
      const bars = sh.map(([nm, s], r) => DT().hbars({ prefix: 'sb' + r, x: 0, y: 330 + r * 30, w: 420, rowH: 28, vals: [s[rare]], names: [nm], max: Math.max(0.01, ...sh.map((q) => q[1][rare]), c0[rare] / n) * 1.15, fmt: (v) => H.pct(v), nameW: 100, valW: 60, cls: () => CC[rare] })).flat();
      const whole = { key: 'wl', kind: 'path', d: `M${100 + (c0[rare] / n) / (Math.max(0.01, ...sh.map((q) => q[1][rare]), c0[rare] / n) * 1.15) * 260},322 v${30 * parts.length + 4}`, cls: 'guide act', audit: 'ignore', layer: 'top' };
      F.add({ items: [title, ...legend, ...cut(), ...heads, H.title('sht', 0, 314, 'share of ' + D.classes[rare] + ' in each part'), ...bars, whole, H.text('wlt', 440, 340, 'dashed: the whole set, ' + H.pct(c0[rare] / n), 'act', 13)], say: `The rarest class, ${D.classes[rare]}, is ${H.pct(c0[rare] / n)} of all rows. ${H.cap(H.list(sh.map(([nm, s, c]) => nm + ' has ' + c[rare] + ' (' + H.pct(s[rare]) + ')')))}.` + (strat ? ' Stratified: every part keeps the shares of the whole, up to rounding.' : ' A plain shuffle lets the shares drift, and in a small part a rare class can almost vanish.'), meta: { phase: 'shares', rare, shares: sh.map((q) => q[1][rare]), counts: sh.map((q) => q[2][rare]) }, view });
      const oParts = [['training', other.tr], ['validation', other.va], ['test', other.te]].filter((p) => p[1].length);
      F.add({ items: [title, ...legend, ...cut(), ...heads, H.title('sht', 0, 314, 'share of ' + D.classes[rare] + ' in each part'), ...bars, whole, H.text('wlt', 440, 340, 'dashed: the whole set, ' + H.pct(c0[rare] / n), 'act', 13), H.text('oth', 440, 372, (strat ? 'without' : 'with') + ' stratification:', 'strong', 13), ...oParts.map(([nm, idx], k) => H.text('oth' + k, 440, 396 + k * 22, nm + ': ' + counts(idx)[rare] + ' (' + H.pct(share(idx)[rare]) + ')', 'tick', 13))], say: `With the same seed ${strat ? 'but no stratification' : 'and stratification'}, the ${D.classes[rare]} rows would fall ${H.list(oParts.map(([nm, idx]) => counts(idx)[rare] + ' in ' + nm))}.`, meta: { phase: 'other', counts: oParts.map(([, idx]) => counts(idx)[rare]) }, view });
      const ctx = { n, shares: c0.map((c) => c / n) };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Three sets', text: ['training: fit the model', 'validation: choose settings, compare models', 'test: one final, honest estimate', 'shuffle before cutting', 'stratify: cut each class separately', 'never tune on the test set'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'cut', pick: 'option', ask: 'How many rows of the rarest class will the test set hold?', options: (fr) => { const c = fr.find((f) => f.meta.phase === 'shares').meta.counts, t = c[c.length - 1]; return [...new Set([t, t + 2, Math.max(0, t - 2), t + 4])].sort((a, b) => a - b).map(String); }, answer: (fr) => { const c = fr.find((f) => f.meta.phase === 'shares').meta.counts; return String(c[c.length - 1]); }, explain: (fr) => fr.find((f) => f.meta.phase === 'shares').say },
    ],
    quiz: [
      { q: 'What is the validation set for?', options: ['choosing settings and comparing models', 'the final score', 'fitting the weights', 'nothing: it is optional decoration'], answer: 0, why: 'The test set must stay untouched until the end, so choices are made on validation rows.' },
      { q: 'You tried 50 settings and reported the best test score. That score is…', options: ['optimistic: the test set was used to choose', 'honest', 'pessimistic', 'the validation score'], answer: 0, why: 'Picking the best of 50 on the test set fits the test set.' },
      { q: '2% of 500 rows are positive. A plain random 20% test split can hold…', options: ['anywhere from about 0 to 5 positives, by luck', 'exactly 2', 'exactly 10', 'none, always'], answer: 0, why: 'About 2 are expected, but a plain shuffle lets it vary; stratifying fixes it at 2.' },
      { q: 'Why shuffle before splitting?', options: ['rows often come in an order that would make the parts differ', 'to make training faster', 'to remove duplicates', 'it is not needed'], answer: 0, why: 'Sorted by class or date, a straight cut gives unrepresentative parts.' },
      { q: 'Rows are measurements over time and the model will predict the future. The test set should be…', options: ['the latest rows, not a random sample', 'random rows', 'the earliest rows', 'the training rows'], answer: 0, why: 'A random split would let the model peek at the future.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // k-fold cross-validation
  // ---------------------------------------------------------------------------------------------------------------
  const CVSETS = {
    flowers: { label: 'Wildflowers (shuffled)', load: () => { const d = H.data('wildflowers'); return { X: d.X, y: d.y, classes: d.classes }; } },
    iris: { label: 'Iris (rows sorted by species)', load: () => { const d = H.data('iris'); return { X: d.X, y: d.y, classes: d.classes }; } },
    moons: { label: 'Two moons, 100 points', load: () => { const d = S.Data.moons({ n: 100, noise: 0.25, seed: 7 }); return { X: d.X, y: d.y, classes: ['blue', 'orange'] }; } },
  };
  const MODELS = { knn: { label: '5-NN', make: () => S.ML.KNeighborsClassifier({ k: 5 }) }, tree: { label: 'a depth-3 tree', make: () => S.ML.DecisionTreeClassifier({ maxDepth: 3, seed: 0 }) }, nb: { label: 'Gaussian naive Bayes', make: () => S.ML.GaussianNB() } };
  S.topic({
    id: 'eval-kfold', family: 'p02-split', order: 20, title: 'k-fold cross-validation',
    blurb: 'One split gives one score, and a different split would give another. k-fold cross-validation cuts the rows into k folds and fits k models: each fold takes one turn as the test set while the other k − 1 train. Every row is tested exactly once, and the k scores give both an average and a spread. Shuffle (or stratify) first: if the rows are sorted by class, a fold can hold a class the training folds never saw.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'flowers', options: Object.entries(CVSETS).map(([value, s]) => ({ value, label: s.label })) },
      { id: 'model', kind: 'choice', label: 'Model', def: 'knn', options: Object.entries(MODELS).map(([value, m]) => ({ value, label: m.label })) },
      { id: 'k', kind: 'slider', label: 'Folds k', def: 5, min: 2, max: 10, step: 1 },
      { id: 'how', kind: 'choice', label: 'Folds', def: 'strat', options: [{ value: 'strat', label: 'stratified, shuffled' }, { value: 'shuffle', label: 'shuffled' }, { value: 'plain', label: 'in row order' }] },
    ],
    presets: [
      { label: '5 stratified folds', values: { data: 'flowers', k: 5, how: 'strat' } },
      { label: '10 folds', values: { data: 'flowers', k: 10, how: 'strat' } },
      { label: 'Sorted rows, no shuffle: disaster', values: { data: 'iris', k: 3, how: 'plain', model: 'knn' } },
      { label: 'Sorted rows, stratified', values: { data: 'iris', k: 3, how: 'strat', model: 'knn' } },
      { label: 'Moons with a tree', values: { data: 'moons', k: 5, how: 'shuffle', model: 'tree' } },
    ],
    counters: [{ id: 'fits', label: 'models fitted' }],
    run(inp) {
      const D = CVSETS[inp.data].load(), X = D.X, y = D.y, n = y.length, k = inp.k, ml = ML();
      const folds = inp.how === 'strat' ? ml.stratifiedKFold(y, k, { shuffle: true, seed: 1 }) : ml.kFold(n, k, { shuffle: inp.how === 'shuffle', seed: 1 });
      const scores = ml.crossValScore(MODELS[inp.model].make(), X, y, { cv: folds, scoring: 'accuracy' });
      const mean = H.mean(scores), sd = H.sd(scores);
      const left = 70, W = 560, cw = W / n, rh = Math.min(46, 330 / k), top = 60;
      const F = new S.Frames(['fits']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, CVSETS[inp.data].label + ': ' + n + ' rows, ' + k + ' folds, ' + MODELS[inp.model].label);
      const classRow = y.map((c, i) => ({ key: 'cr' + i, kind: 'cell', x: +(left + i * cw).toFixed(2), y: 34, w: +Math.max(0.6, cw - 0.4).toFixed(2), h: 14, rx: 0, text: '', cls: 'split ' + CC[c], audit: 'ignore' }));
      const rowName = H.text('crn', left - 8, 46, 'class', 'tick', 12, 'end');
      const foldRow = (f, lit) => {
        const test = new Set(folds[f].test), yy = top + f * rh;
        const it = [{ key: 'fn' + f, kind: 'text', x: left - 8, y: yy + rh / 2 + 4, text: 'fold ' + (f + 1), size: 12, cls: lit ? 'act' : 'tick', anchor: 'end' }];
        // runs of equal state drawn as one cell each, so a fold row is a handful of shapes
        let s = 0;
        for (let i = 1; i <= n; i++) if (i === n || test.has(i) !== test.has(s)) { it.push({ key: 'fc' + f + '_' + s, kind: 'cell', x: +(left + s * cw).toFixed(2), y: +(yy + 3).toFixed(2), w: +Math.max(0.6, (i - s) * cw - 0.4).toFixed(2), h: +(rh - 6).toFixed(2), rx: 1, text: '', cls: 'split ' + (test.has(s) ? 'te' : lit ? 'tr' : 'no'), audit: 'ignore' }); s = i; }
        return it;
      };
      const scoreLabel = (f) => H.text('fs' + f, left + W + 12, top + f * rh + rh / 2 + 5, H.pct(scores[f]), scores[f] < mean - sd ? 'bad' : 'strong', 13);
      const legend = [{ key: 'k1', kind: 'cell', x: left, y: top + k * rh + 14, w: 14, h: 14, rx: 2, text: '', cls: 'key tr' }, H.text('k1t', left + 20, top + k * rh + 26, 'training', 'legend', 13), { key: 'k2', kind: 'cell', x: left + 110, y: top + k * rh + 14, w: 14, h: 14, rx: 2, text: '', cls: 'key te' }, H.text('k2t', left + 130, top + k * rh + 26, 'test', 'legend', 13)];
      const rowsUpTo = (m, lit) => { const it = []; for (let f = 0; f < k; f++) it.push(...foldRow(f, f === lit)); for (let f = 0; f < m; f++) it.push(scoreLabel(f)); return it; };
      F.add({ items: [title, rowName, ...classRow, ...rowsUpTo(0, -1), ...legend], say: `${n} rows in their stored order (the top strip shows each row's class), and ${k} folds of about ${Math.round(n / k)} rows each, ${inp.how === 'plain' ? 'taken as consecutive blocks' : inp.how === 'shuffle' ? 'drawn after a shuffle' : 'drawn class by class after a shuffle, so each keeps the class shares'}.`, meta: { phase: 'folds', sizes: folds.map((f) => f.test.length) }, view });
      for (let f = 0; f < k; f++) {
        F.count('fits');
        const te = folds[f].test, cls = D.classes.map((_, c) => te.filter((i) => y[i] === c).length), trc = D.classes.map((_, c) => folds[f].train.filter((i) => y[i] === c).length);
        const missing = D.classes.filter((_, c) => trc[c] === 0);
        F.add({ items: [title, rowName, ...classRow, ...rowsUpTo(f + 1, f), ...legend], say: `Fold ${f + 1}: train on ${folds[f].train.length} rows, test on the other ${te.length} (${H.list(D.classes.map((c, q) => cls[q] + ' ' + c))}). Accuracy ${H.pct(scores[f])}.` + (missing.length ? ` The training rows hold no ${H.list(missing)} at all, so the model cannot predict that class.` : ''), meta: { phase: 'fold', f, score: scores[f], testCounts: cls, missing: missing.length }, view });
      }
      F.add({ items: [title, rowName, ...classRow, ...rowsUpTo(k, -1), ...legend, H.text('mean', left + W + 12, top + k * rh + 26, 'mean ' + H.pct(mean), 'act', 14), H.text('sd', left + W + 12, top + k * rh + 48, 'sd ' + H.f(sd * 100, 1) + ' points', 'tick', 13)], say: `Every row was tested exactly once. The ${k} scores average ${H.pct(mean)} with a standard deviation of ${H.f(sd * 100, 1)} points: ` + (sd < 0.005 ? 'every fold gave the same score.' : `they run from ${H.pct(Math.min(...scores))} to ${H.pct(Math.max(...scores))}, and one split alone could have reported any of them.`), meta: { phase: 'mean', mean, sd, scores }, view });
      const ctx = { n, k, trainSizes: folds.map((f) => f.train.length), testSizes: folds.map((f) => f.test.length), fold: folds.map((_, f) => f + 1) };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'k-fold', text: ['split rows into k folds', 'for each fold i: train on the rest, test on i', 'score = mean of the k test scores', 'spread = their standard deviation', 'stratified: each fold keeps the class shares', 'k models: k times the cost of one'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'folds', pick: 'option', ask: 'Will the lowest fold score be below 50%?', options: ['yes', 'no'], answer: (fr) => (Math.min(...fr[fr.length - 1].meta.scores) < 0.5 ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'With 5-fold cross-validation on 200 rows, each model trains on…', options: ['160 rows', '200 rows', '40 rows', '100 rows'], answer: 0, why: 'Four of the five folds of 40.' },
      { q: 'How many times is each row used for testing?', options: ['exactly once', 'k times', 'never', 'k − 1 times'], answer: 0, why: 'The folds partition the rows.' },
      { q: 'Rows are sorted by class and the folds are consecutive blocks. What can happen?', options: ['a training set lacks a whole class, and that fold scores terribly', 'nothing', 'the scores become identical', 'the model trains faster'], answer: 0, why: 'Shuffle or stratify first.' },
      { q: 'Why report the spread of the fold scores?', options: ['it shows how much the estimate depends on the particular split', 'it is the test error', 'it measures bias', 'it is always zero'], answer: 0, why: 'Two models whose means differ by less than the spread may not really differ.' },
      { q: 'Leave-one-out is k-fold with k equal to…', options: ['the number of rows', '2', '10', '1'], answer: 0, why: 'Each fold holds a single row.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
