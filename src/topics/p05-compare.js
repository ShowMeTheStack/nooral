// Part 5, the ensembles side by side: one deep tree, bagged trees, a random forest, AdaBoost on stumps and gradient
// boosting on small trees, trained on the same points; each panel is a model's boundary (pale where it is unsure),
// and the bars compare their accuracy on the training points and on fresh test points.
(function (S) {
  'use strict';
  const H = S.PL, T5 = S.PL5, ML = () => S.ML;
  const n$ = H.n;

  S.topic({
    id: 'ensemble-compare', family: 'p05-compare', order: 10, title: 'Five tree models on the same points',
    blurb: 'A single deep tree draws a jagged boundary that follows every training point. Bagging and a random forest average many such trees into a smoother, steadier one. AdaBoost and gradient boosting build theirs from many small trees added in sequence. Same data, five boundaries; the accuracy on fresh points tells which smoothing paid off here.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'moons', options: T5.SETS },
      { id: 'n', kind: 'number', label: 'training points', def: 60, min: 30, max: 200 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.25, min: 0.05, max: 0.6, step: 0.05 },
      { id: 'trees', kind: 'number', label: 'trees per ensemble', def: 25, min: 5, max: 60 },
      { id: 'seed', kind: 'number', label: 'seed', def: 4, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Noisy moons', values: { set: 'moons', n: 60, noise: 0.25, trees: 25, seed: 4 } },
      { label: 'Moons where one tree does well', values: { set: 'moons', n: 80, noise: 0.2, trees: 25, seed: 7 } },
      { label: 'Circles', values: { set: 'circles', n: 100, noise: 0.12, trees: 25, seed: 2 } },
      { label: 'XOR', values: { set: 'xor', n: 80, noise: 0.1, trees: 25, seed: 3 } },
      { label: 'Wildflowers (three classes)', values: { set: 'flowers', n: 90, noise: 0.3, trees: 25, seed: 5 } },
      { label: 'Exam results', values: { set: 'exam', n: 120, noise: 0.3, trees: 25, seed: 6 } },
    ],
    counters: [{ id: 'models', label: 'models' }],
    run(inp) {
      const d = T5.data2(inp.set, inp.n, inp.seed, inp.noise), te = T5.data2(inp.set, 300, inp.seed + 1000, inp.noise), B = inp.trees;
      const specs = [
        { name: 'one deep tree', short: 'tree', m: () => ML().DecisionTreeClassifier({ seed: 0 }) },
        { name: 'bagging', short: 'bagging', m: () => ML().RandomForestClassifier({ nEstimators: B, maxFeatures: null, seed: inp.seed }) },
        { name: 'random forest', short: 'forest', m: () => ML().RandomForestClassifier({ nEstimators: B, maxFeatures: 'sqrt', seed: inp.seed }) },
        { name: 'AdaBoost (stumps)', short: 'AdaBoost', m: () => ML().AdaBoostClassifier({ nEstimators: B * 2, maxDepth: 1, seed: inp.seed }) },
        { name: 'gradient boosting', short: 'boosting', m: () => ML().GradientBoostingClassifier({ nEstimators: B * 2, learningRate: 0.1, maxDepth: 2, seed: inp.seed }) },
      ];
      const fits = specs.map((s) => { const m = s.m().fit(d.X, d.y); return { s, m, train: T5.acc(d.y, m.predict(d.X)), test: T5.acc(te.y, m.predict(te.X)) }; });
      const conf = (m, x) => {
        if (m.predictProba) { const p = m.predictProba([x])[0], k = ML().argmax(p); return [k, Math.max(0.15, (p[k] - 1 / d.K) / (1 - 1 / d.K))]; }
        const v = m.decisionFunction([x])[0];
        if (typeof v === 'number') return [v > 0 ? 1 : 0, Math.max(0.15, Math.min(1, Math.abs(v) * 1.5))];
        const k = ML().argmax(v); return [k, Math.max(0.15, Math.min(1, v[k]))];
      };
      const box = (k) => ({ x: 20 + (k % 3) * 256, y: 44 + Math.floor(k / 3) * 262, w: 220, h: 190 });
      const panels = fits.map((f, k) => T5.panel('ec' + k, box(k), d, (u, v) => conf(f.m, [u, v]), (k + 1) + '. ' + f.s.name, { r: 2.4, n: 48 }).items);
      const bars = (upto) => {
        const b = box(5), items = [H.title('eb:t', b.x, b.y - 8, 'accuracy: train, test')];
        const x0 = b.x + 70, w = b.w - 100, rh = 36;
        items.push({ key: 'eb:z', kind: 'path', d: `M${x0},${b.y} v${5 * rh}`, cls: 'axis', audit: 'ignore', layer: 'back' });
        fits.forEach((f, k) => {
          const y = b.y + k * rh;
          items.push(H.text('eb:n' + k, x0 - 8, y + 16, f.s.short, k < upto ? 'tick' : 'pl-lab muted', 12, 'end'));
          if (k >= upto) return;
          items.push({ key: 'eb:a' + k, kind: 'cell', x: x0, y: y + 2, w: H.r2(w * f.train), h: 13, rx: 2, text: '', cls: 'pl-pbar soft', audit: 'ignore' });
          items.push({ key: 'eb:b' + k, kind: 'cell', x: x0, y: y + 17, w: H.r2(w * f.test), h: 13, rx: 2, text: '', cls: 'pl-pbar' + (k === upto - 1 ? ' hot' : ''), audit: 'ignore' });
          items.push(H.text('eb:v' + k, x0 + w + 8, y + 22, H.pct(f.test), 'tick', 12));
        });
        return items;
      };
      const F = new S.Frames(['models']);
      F.add({ items: [...T5.panel('ec0', box(0), d, () => [0, 0], 'the training points', { r: 2.4, n: 4 }).items.filter((it) => !it.key.endsWith(':f')), ...bars(0)], say: d.X.length + ' training points and 300 fresh test points from the same source. Five models will be trained on the same points; each ensemble uses ' + B + ' trees (the boosters twice as many smaller ones).', meta: { phase: 'start', n: d.X.length, nte: 300, B, B2: 2 * B } });
      fits.forEach((f, k) => {
        F.set('models', k + 1);
        const items = [...panels.slice(0, k + 1).flat(), ...bars(k + 1)];
        const notes = ['it fits the training points perfectly or nearly, with a jagged boundary', 'averaging ' + B + ' bootstrap trees smooths the edges', 'drawing one feature per split makes the trees differ more', 'its boundary is a sum of axis cuts, one per stump', 'small trees added at rate 0.1 build a smooth boundary'];
        F.add({ items, say: (k + 1) + '. ' + f.s.name[0].toUpperCase() + f.s.name.slice(1) + ': ' + H.pct(f.train) + ' of the training points and ' + H.pct(f.test) + ' of the test points right; ' + notes[k] + '.', meta: { phase: 'model', k: k + 1, train: f.train, test: f.test, B, rate: 0.1 } });
      });
      const best = fits.reduce((a, b) => (b.test > a.test + 1e-12 ? b : a)), tree = fits[0];
      F.add({ items: [...panels.flat(), ...bars(5)], say: (best === tree ? 'Here the single tree is best on fresh points, with ' + H.pct(tree.test) + ': on this sample its jagged cuts happened to land well. ' : 'Best on fresh points: ' + best.s.name + ' with ' + H.pct(best.test) + ', against ' + H.pct(tree.test) + ' for the single tree. ') + 'The single tree had ' + H.pct(tree.train) + ' on its own training points; the gap between training and test accuracy is the variance the ensembles were built to remove.', meta: { phase: 'done', best: best.test, tree: tree.test, treeTrain: tree.train } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Five tree models', text: ['tree: one deep CART tree', 'bagging: vote of trees on bootstrap samples', 'forest: bagging + random features per split', 'AdaBoost: α-weighted vote of reweighted stumps', 'boosting: sum of small trees fitted to residuals'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which will score best on the fresh test points?', options: ['one deep tree', 'bagging', 'random forest', 'AdaBoost (stumps)', 'gradient boosting'], answer: (fr) => { const ms = fr.filter((x) => x.meta.phase === 'model'); const b = ms.reduce((a, x) => (x.meta.test > a.meta.test + 1e-12 ? x : a)); return ['one deep tree', 'bagging', 'random forest', 'AdaBoost (stumps)', 'gradient boosting'][b.meta.k - 1]; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Which two methods build their trees independently of one another?', options: ['bagging and random forests', 'AdaBoost and gradient boosting', 'all five', 'none'], answer: 0, why: 'Their trees can even be grown in parallel; boosting is sequential.' },
      { q: 'Which models mainly reduce variance, and which mainly reduce bias?', options: ['averaging (bagging, forests) reduces variance; boosting reduces bias', 'the reverse', 'all reduce bias only', 'none change either'], answer: 0, why: 'Averages steady deep trees; boosting adds capacity to weak learners step by step.' },
      { q: 'A single tree scores 100% on training and 81% on test. This gap shows…', options: ['high variance (overfitting)', 'high bias', 'a bug', 'label noise only'], answer: 0, why: 'It fits the training points too closely.' },
      { q: 'Why does no single ensemble always win?', options: ['which works best depends on the data and the settings', 'the seeds are random', 'accuracy is not a real measure', 'they are all the same'], answer: 0, why: 'Compare them on held-out data for the problem at hand.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
