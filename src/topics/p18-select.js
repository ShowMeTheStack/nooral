// Part 18, choosing a model: k-fold cross-validation of six models on one data set (every fold score, the mean and its
// spread), the choice by the best mean and by the one-standard-error rule, and the score on the untouched test rows;
// then no free lunch: the same six models on six data sets, where the winner keeps changing.
(function (S) {
  'use strict';
  const P = S.P18, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const SEL = ['moons', 'circles', 'xor', 'loans'];
  const SHORT = { nb: 'NB', logistic: 'logistic', tree: 'tree', knn: 'k-NN', svm: 'SVM', forest: 'forest' };

  P.selectRun = (set, k, seed) => P.once(['sel', set, k, seed].join(':'), () => {
    const d = P.toy(set, 240, 1), sp = S.Data.split(d.y, 0.25, seed);
    const Xd = sp.train.map((i) => d.X[i]), yd = sp.train.map((i) => d.y[i]), Xt = sp.test.map((i) => d.X[i]), yt = sp.test.map((i) => d.y[i]);
    const rows = P.MODELS.map((m) => { const s = P.cvScores(m, Xd, yd, k); return { id: m.id, name: m.name, say: m.say, scores: s, mean: P.mean(s), sd: P.sd(s) }; });
    const best = rows.reduce((b, r, i) => (r.mean > rows[b].mean ? i : b), 0), se = rows[best].sd / Math.sqrt(k), bar = rows[best].mean - se;
    const pick = rows.findIndex((r) => r.mean >= bar - 1e-12);
    const test = (i) => { const sc = P.scaler(Xd), m = P.MODELS[i].make().fit(sc.apply(Xd), yd); return S.Metrics.accuracy(yt, m.predict(sc.apply(Xt))).value; };
    return { d, nDev: yd.length, nTest: yt.length, foldSizes: S.ML.stratifiedKFold(yd, k).map((f) => f.test.length), rows, best, se, bar, pick, testBest: test(best), testPick: test(pick) };
  });

  S.topic({
    id: 'model-selection', family: 'p18-practice', order: 20, title: 'Choosing a model by cross-validation',
    blurb: 'Put a test set aside and do not touch it. Split the rest into k folds; train each candidate k times, each time holding one fold out to score it on. The k scores give a mean and a spread, and the spread matters: two models whose means differ by less than the noise are tied, and then the simpler one is the safer choice (the one-standard-error rule). Only at the very end is the chosen model scored once on the test set.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'moons', options: SEL.map((k) => ({ value: k, label: P.TOYS[k].label })) },
      { id: 'k', kind: 'choice', label: 'Folds', def: '5', options: ['3', '5', '10'].map((v) => ({ value: v, label: v + ' folds' })) },
      { id: 'seed', kind: 'number', label: 'Seed (the test split)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Moons, 5 folds', values: { set: 'moons', k: '5', seed: 1 } },
      { label: 'Loans, 10 folds', values: { set: 'loans', k: '10', seed: 1 } },
      { label: 'Circles, 3 folds', values: { set: 'circles', k: '3', seed: 2 } },
      { label: 'XOR corners', values: { set: 'xor', k: '5', seed: 1 } },
    ],
    counters: [{ id: 'fits', label: 'models trained' }],
    run(inp) {
      const k = +inp.k, R = P.selectRun(inp.set, k, inp.seed), F = H().frames(['fits']), q = F.q, rows = R.rows;
      // the folds
      const fx = 40, fw = 640, total = R.nDev + R.nTest, items0 = [];
      let at = fx;
      R.foldSizes.forEach((m, i) => { const w = (fw * m) / total; items0.push({ key: 'ms:f' + i, kind: 'cell', x: P.r2(at), y: 70, w: P.r2(w - 3), h: 34, rx: 3, text: 'fold ' + (i + 1), size: 12, cls: 'p18fold' }); at += w; });
      items0.push({ key: 'ms:test', kind: 'cell', x: P.r2(at + 8), y: 70, w: P.r2((fw * R.nTest) / total - 8), h: 34, rx: 3, text: 'test', size: 12, cls: 'p18test' });
      const rounds = Math.min(k, 5);
      for (let r = 0; r < rounds; r++) {
        let a = fx;
        R.foldSizes.forEach((m, i) => { const w = (fw * m) / total; items0.push({ key: 'ms:r' + r + '_' + i, kind: 'cell', x: P.r2(a), y: 140 + r * 30, w: P.r2(w - 3), h: 22, rx: 3, text: '', cls: i === r ? 'p18hold' : 'p18train' }); a += w; });
        items0.push(H().text('ms:rn' + r, fx - 8, 156 + r * 30, String(r + 1), 'tick', 12, 'end'));
      }
      items0.push(H().text('ms:rt', fx, 128, 'round: train on the pale folds, score on the dark one' + (k > rounds ? ' (first ' + rounds + ' of ' + k + ')' : ''), 'axname', 13), H().head('ms:h', fx, 40, 'Development rows and a locked test set'));
      F.push(items0, `${P.qpl(q, total, 'row')}: ${q.n(R.nTest)} are locked away as the test set, and the other ${q.n(R.nDev)} are dealt into ${q.n(k)} folds of about ${q.n(R.foldSizes[0])} rows, keeping the class shares. Each model is trained ${q.n(k)} times, each time scored on the fold it did not see.`, { phase: 'folds', total, nTest: R.nTest, nDev: R.nDev, k, fold: R.foldSizes[0] });
      // the dot plot
      const lo = Math.max(0, Math.floor(Math.min(...rows.flatMap((r) => r.scores)) * 10) / 10), ax = C().axes({ prefix: 'ms:ax', x: 150, y: 70, w: 520, h: rows.length * 46, xr: [lo, 1], yr: [0, rows.length], xticks: 0.1, yticks: 0, xname: 'accuracy on the held-out fold', tickSize: 12 });
      const rowY = (i) => ax.Y(rows.length - i - 0.5);
      const plot = (upto, extra) => {
        const it = [...ax.items];
        rows.forEach((r, i) => {
          it.push(H().text('ms:n' + i, 140, rowY(i) + 4.5, r.name, i === extra.pick ? 'strong' : 'legend', 13, 'end'));
          if (i > upto) return;
          r.scores.forEach((s, j) => it.push({ key: 'ms:d' + i + '_' + j, kind: 'dot', x: ax.X(s), y: P.r2(rowY(i) - 9), r: 3.2, cls: 'sp cn', audit: 'overlap-ok', layer: 'items' }));
          it.push({ key: 'ms:sd' + i, kind: 'path', d: `M${ax.X(Math.max(lo, r.mean - r.sd))},${rowY(i) + 6} H${ax.X(Math.min(1, r.mean + r.sd))}`, cls: 'p18spread' + (i === extra.best ? ' best' : ''), audit: 'ignore', layer: 'lines' });
          it.push({ key: 'ms:m' + i, kind: 'dot', x: ax.X(r.mean), y: P.r2(rowY(i) + 6), r: 5.5, cls: 'mark ' + (i === extra.pick ? 'p18pick' : i === extra.best ? 'p18best' : 'p18mean'), point: true });
        });
        return it;
      };
      rows.forEach((r, i) => {
        F.count('fits', k);
        F.push([...plot(i, {}), H().head('ms:h', 40, 40, 'Cross-validated accuracy: dots are folds, the bar is mean ± one spread')],
          `${r.say[0].toUpperCase() + r.say.slice(1)}: fold scores ${r.scores.map((s) => q(s, 2)).join(', ')}. Mean ${q(r.mean, 3)}, spread (standard deviation) ${q(r.sd, 3)}.`, { phase: 'model', i, scores: r.scores, mean: r.mean, sd: r.sd });
      });
      const b = rows[R.best], p = rows[R.pick];
      const guide = { key: 'ms:se', kind: 'path', d: `M${ax.X(Math.max(lo, R.bar))},${ax.o.y} V${ax.o.y + ax.o.h}`, cls: 'p18guide', audit: 'ignore', layer: 'back', z: 4 };
      F.push([...plot(rows.length, { best: R.best, pick: R.pick }), guide, H().head('ms:h', 40, 40, 'Best mean, and the simplest within one standard error')],
        `The best mean is that of ${b.say}, ${q(b.mean, 3)}. Its standard error, the spread over √k, is ${q(R.se, 3)}, so anything above ${q(R.bar, 3)} (dashed line) is within the noise of the best. ` + (R.pick === R.best ? `No simpler model gets there, so ${b.say} is chosen.` : `The simplest model that clears it is ${p.say}, at ${q(p.mean, 3)}: the one-standard-error rule picks it over ${b.say}.`), { phase: 'choose', bestMean: b.mean, se: R.se, bar: R.bar, pickMean: p.mean, best: R.best, pick: R.pick });
      F.push([...plot(rows.length, { best: R.best, pick: R.pick }), guide, H().head('ms:h', 40, 40, 'Once, at the end: the test set'), ...H().lines('ms:t', 150, ax.o.y + ax.o.h + 70, [[p.name + ': cross-validated ' + H().fx(p.mean, 3) + ', test ' + H().fx(R.testPick, 3), 'strong']].concat(R.pick !== R.best ? [[b.name + ': cross-validated ' + H().fx(b.mean, 3) + ', test ' + H().fx(R.testBest, 3), 'legend']] : []), 13, 22)],
        `Retrained on all ${q.n(R.nDev)} development rows, ${p.say} scores ${q(R.testPick, 3)} on the ${q.n(R.nTest)} test rows, against ${q(p.mean, 3)} in cross-validation.` + (R.pick !== R.best ? ` For comparison ${b.say} scores ${q(R.testBest, 3)}.` : '') + ' This number is reported, not used to choose again: choosing on the test set would make it optimistic too.', { phase: 'test', testPick: R.testPick, testBest: R.testBest, pickMean: p.mean, nDev: R.nDev, nTest: R.nTest });
      for (const f of F.list.slice(1)) f.narrowView = { x: 0, y: 20, w: 700, h: 400 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'k-fold cross-validation', text: ['for fold i = 1..k:', '  train on the other k − 1 folds, score on fold i', 'mean = (1/k) Σ sᵢ,  spread = sd(s)', 'standard error = sd / √k', 'one-SE rule: the simplest model with', '  mean ≥ best mean − its standard error'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'folds', pick: 'option', once: true,
      ask: 'Which model will have the best mean?',
      options: () => P.MODELS.map((m) => m.name),
      answer: (fr) => P.MODELS[fr.find((f) => f.meta.phase === 'choose').meta.best].name,
      explain: (fr) => fr.find((f) => f.meta.phase === 'choose').say,
    }],
    quiz: [
      { q: 'With 5 folds, how many times is each model trained?', options: ['5', '1', '4', '25'], answer: 0, why: 'Once per held-out fold.' },
      { q: 'Model A: mean 0.912, sd 0.04 over 5 folds; model B (simpler): mean 0.900. The one-SE rule picks…', options: ['B, since 0.900 ≥ 0.912 − 0.04/√5 ≈ 0.894', 'A, it has the higher mean', 'neither', 'whichever trains faster'], answer: 0, why: 'B is within one standard error of the best.' },
      { q: 'Why keep a test set apart from cross-validation?', options: ['the choice itself is tuned to the folds, so its CV score is a little optimistic', 'cross-validation cannot score models', 'to train faster', 'the test set has different classes'], answer: 0, why: 'Picking the best of many CV means favours lucky ones.' },
      { q: 'Why stratify the folds?', options: ['each fold keeps the class shares, so fold scores are comparable', 'to make folds of different sizes', 'to shuffle the test set', 'it removes the need for a test set'], answer: 0, why: 'A fold with few of one class would give a skewed score.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const NFL = ['linear', 'moons', 'circles', 'xor', 'checker', 'loans'];
  const NFLNAME = { linear: 'The noisy line', moons: 'Two moons', circles: 'The ring and disc', xor: 'XOR', checker: 'The checkerboard', loans: 'The loans table' };
  P.nflRun = (n, k, seed) => P.once(['nfl', n, k, seed].join(':'), () => NFL.map((id) => { const d = P.toy(id, id === 'loans' ? 2 * n : n, seed); return { id, d, means: P.MODELS.map((m) => P.mean(P.cvScores(m, d.X, d.y, k))) }; }));
  S.topic({
    id: 'no-free-lunch', family: 'p18-practice', order: 30, title: 'No free lunch: the winner depends on the data',
    blurb: 'No learning method is best on every problem: each one’s built-in assumptions (a straight boundary, independent features, nearby points alike, axis-aligned cuts) suit some data and fail on other data. Run the same six models on six data sets and fill a table of cross-validated accuracy: the best model in one row is ordinary or poor in another. That is why we try several models and let validation decide.',
    inputs: [
      { id: 'n', kind: 'choice', label: 'Points per toy set', def: '150', options: ['90', '150', '240'].map((v) => ({ value: v, label: v })) },
      { id: 'k', kind: 'choice', label: 'Folds', def: '3', options: ['3', '5'].map((v) => ({ value: v, label: v + ' folds' })) },
      { id: 'seed', kind: 'number', label: 'Seed (the data)', def: 1, min: 1, max: 999 },
    ],
    presets: [{ label: '150 points, 3 folds', values: { n: '150', k: '3', seed: 1 } }, { label: 'Few points (90)', values: { n: '90', k: '3', seed: 2 } }, { label: 'More points, 5 folds', values: { n: '240', k: '5', seed: 1 } }],
    counters: [{ id: 'sets', label: 'data sets tried' }],
    run(inp) {
      const R = P.nflRun(+inp.n, +inp.k, inp.seed), F = H().frames(['sets']), q = F.q;
      const thumbs = (cur) => R.flatMap((r, i) => {
        const x0 = 20 + i * 120, out = [{ key: 'nf:tb' + i, kind: 'region', x: x0, y: 30, w: 108, h: 108, cls: i === cur ? 'p18cur' : 'frame', text: '' }];
        if (r.d.real) out.push(H().text('nf:tr' + i, x0 + 54, 80, 'a table', 'legend', 12, 'middle'), H().text('nf:tr2' + i, x0 + 54, 98, '5 columns', 'tick', 12, 'middle'));
        else {
          const lo = [0, 1].map((j) => Math.min(...r.d.X.map((x) => x[j]))), hi = [0, 1].map((j) => Math.max(...r.d.X.map((x) => x[j])));
          r.d.X.forEach((x, j) => { if (j % 2) return; out.push({ key: 'nf:p' + i + '_' + j, kind: 'dot', x: P.r2(x0 + 8 + (92 * (x[0] - lo[0])) / (hi[0] - lo[0])), y: P.r2(38 + 92 - (92 * (x[1] - lo[1])) / (hi[1] - lo[1])), r: 2, cls: 'sp c' + r.d.y[j], audit: 'overlap-ok' }); });
        }
        out.push(H().text('nf:tn' + i, x0 + 54, 156, r.id === 'checker' ? 'checkerboard' : r.id, i === cur ? 'strong' : 'legend', 12, 'middle'));
        return out;
      });
      const table = (upto) => {
        const vals = R.map((r, i) => (i <= upto ? r.means.map((v) => Math.max(0, v - 0.4)) : r.means.map(() => 0)));
        const win = R.map((r) => r.means.indexOf(Math.max(...r.means)));
        return S.Kit.heat.grid({ prefix: 'nf:hm', x: 130, y: 210, cw: 92, ch: 32, vals, scheme: 'seq', max: 0.6, fmt: null, rows: R.map((r) => (r.id === 'checker' ? 'checkerboard' : r.id)), cols: P.MODELS.map((m) => SHORT[m.id]), size: 13, labelSize: 12, cls: (i, j) => (i <= upto && j === win[i] ? 'hot' : i > upto ? 'p18blank' : '') }).items
          .map((it) => { const m = /:c(\d+)_(\d+)$/.exec(it.key); if (m && +m[1] <= upto) it.text = H().fx(R[+m[1]].means[+m[2]], 2); return it; });
      };
      F.push([...thumbs(-1), ...table(-1), H().text('nf:ht', 130, 186, 'cross-validated accuracy (' + inp.k + ' folds); outlined: the best in its row', 'axname', 13)],
        `Six data sets: five pictures in the plane and one real table, the loans. Six models, from naive Bayes to a random forest, each scored by ${q.n(+inp.k)}-fold cross-validation on every set.`, { phase: 'start', k: +inp.k });
      R.forEach((r, i) => {
        F.set('sets', i + 1);
        const w = r.means.indexOf(Math.max(...r.means)), wo = r.means.indexOf(Math.min(...r.means));
        F.push([...thumbs(i), ...table(i), H().text('nf:ht', 130, 186, 'cross-validated accuracy (' + inp.k + ' folds); outlined: the best in its row', 'axname', 13)],
          `${NFLNAME[r.id]}: the best is ${P.MODELS[w].say} with ${q(r.means[w], 2)}, the worst ${P.MODELS[wo].say} with ${q(r.means[wo], 2)}.`, { phase: 'set', i, means: r.means, best: w, worst: wo });
      });
      const wins = R.map((r) => r.means.indexOf(Math.max(...r.means))), distinct = [...new Set(wins)];
      const ranks = P.MODELS.map((_, j) => P.mean(R.map((r) => 1 + r.means.filter((v) => v > r.means[j]).length)));
      const bestRank = ranks.indexOf(Math.min(...ranks));
      F.push([...thumbs(-1), ...table(R.length - 1), H().text('nf:ht', 130, 186, 'cross-validated accuracy (' + inp.k + ' folds); outlined: the best in its row', 'axname', 13)],
        `Over the six sets, ${P.qpl(q, distinct.length, 'different model')} ${distinct.length === 1 ? 'wins' : 'win'} at least once. The best average rank, ${q(ranks[bestRank], 2)}, belongs to ${P.MODELS[bestRank].say}, yet it is not first everywhere. Logistic regression, one straight boundary, scores ${q(R[0].means[1], 2)} on the noisy line and ${q(R[3].means[1], 2)} on XOR: the no-free-lunch idea in one table.`, { phase: 'end', distinct: distinct.length, rank: ranks[bestRank], bestRank, wins, linLine: R[0].means[1], linXor: R[3].means[1] });
      for (const f of F.list) f.narrowView = { x: 0, y: 170, w: 720, h: 270 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'No free lunch', text: ['averaged over all possible problems,', 'every learner does equally well;', 'on a real problem, the learner whose', 'assumptions fit the data wins', 'so: try several, validate, then choose'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'set' && f.meta.i === 2, pick: 'option', once: true,
      ask: 'On XOR corners, how will the logistic model do?',
      options: ['about as well as guessing', 'the best of the six', 'perfectly'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'set' && f.meta.i === 3).meta.means[1]; return m < 0.65 ? 'about as well as guessing' : m >= 0.99 ? 'perfectly' : null; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'set' && f.meta.i === 3).say,
    }],
    quiz: [
      { q: 'What does the no-free-lunch idea say?', options: ['no learner is best on every problem', 'more data is always free', 'simple models always win', 'deep networks always win'], answer: 0, why: 'Every method’s assumptions fit some problems and not others.' },
      { q: 'Why does logistic regression fail on XOR corners?', options: ['its boundary is a single straight line', 'it needs more data', 'XOR has missing values', 'it overfits'], answer: 0, why: 'No straight line separates opposite corners.' },
      { q: 'Why can naive Bayes do well on the ring round a disc?', options: ['each class is a Gaussian centred at the origin with a different spread', 'it draws straight lines', 'it memorises points', 'it uses many trees'], answer: 0, why: 'Different variances give a circular boundary.' },
      { q: 'The practical lesson is to…', options: ['try several models and choose by validation', 'always use a random forest', 'always use the simplest model', 'use the test set to choose'], answer: 0, why: 'The data decide, through an honest validation.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
