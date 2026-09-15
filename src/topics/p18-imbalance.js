// Part 18, imbalanced data: a rare class, the always-majority baseline whose accuracy looks good, a plain logistic
// model that barely finds the rare class, then a fix (random oversampling, undersampling or class weights) moving the
// boundary, and the metrics that notice the difference (recall, precision, F1, balanced accuracy, average precision).
(function (S) {
  'use strict';
  const P = S.P18, H = () => S.Kit.ml, M = () => S.Metrics;
  const FIX = { none: 'no fix', over: 'random oversampling of the rare class', under: 'random undersampling of the common class', weights: 'class weights n / (2 nₖ)' };
  const SETS = { toy: { label: 'a toy set: 5% rare points', make: () => P.rareToy(3) }, pumps: { label: 'pump readings: faults are 5%', make: () => P.pumps2() } };

  /** the weights a fix gives the training rows (copies for oversampling, 0 for rows left out) */
  P.fixWeights = (fix, y, seed) => {
    const n = y.length, rare = y.map((v, i) => (v ? i : -1)).filter((i) => i >= 0), common = y.map((v, i) => (v ? -1 : i)).filter((i) => i >= 0);
    const w = new Array(n).fill(1), r = S.Data.rng(seed);
    if (fix === 'over') for (let k = rare.length; k < common.length; k++) w[rare[Math.floor(r() * rare.length)]]++;
    if (fix === 'under') { const keep = new Set(r.shuffle(common.slice()).slice(0, rare.length)); for (const i of common) if (!keep.has(i)) w[i] = 0; }
    if (fix === 'weights') for (let i = 0; i < n; i++) w[i] = n / (2 * (y[i] ? rare.length : common.length));
    return w;
  };
  /** every number the topic shows for one way of training */
  P.imbRun = (set, fix, seed) => P.once(['imb', set, fix, seed].join(':'), () => {
    const d = SETS[set].make(), sp = S.Data.split(d.y, 0.3, 1), Xtr = sp.train.map((i) => d.X[i]), ytr = sp.train.map((i) => d.y[i]), Xte = sp.test.map((i) => d.X[i]), yte = sp.test.map((i) => d.y[i]);
    const sc = P.scaler(Xtr), w = P.fixWeights(fix, ytr, seed), m = S.Rob.logreg(sc.apply(Xtr), ytr, { weights: w, iters: 400 });
    const prob = (x) => m.prob(sc.apply([x])[0]), score = Xte.map(prob), pred = score.map((p) => (p >= 0.5 ? 1 : 0));
    return { d, Xtr, ytr, Xte, yte, w, prob, score, pred, metrics: P.imbMetrics(yte, pred, score) };
  });
  P.imbMetrics = (y, pred, score) => {
    const cm = M().confusion(y, pred, { labels: [0, 1] }), pr = M().precisionRecallF1(y, pred, { average: 'binary', posLabel: 1 });
    return { cm: cm.counts, acc: M().accuracy(y, pred).value, bal: M().balancedAccuracy(y, pred).value, recall: pr.recall, precision: pr.precision, f1: pr.f1, ap: score ? M().averagePrecision(y, score).value : null };
  };

  S.topic({
    id: 'imbalanced-data', family: 'p18-practice', order: 10, title: 'Imbalanced data: when the rare class is the point',
    blurb: 'Faults, frauds and diseases are rare, and they are what we want to find. A model that always says “common” scores high accuracy and finds nothing. Accuracy hides this; recall, precision, F1, balanced accuracy and average precision do not. To make a model care about the rare class, show it more of it (oversampling), less of the rest (undersampling) or make its mistakes cost more (class weights): the boundary moves and the rare points start to be caught, at a price in false alarms.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'toy', options: Object.keys(SETS).map((k) => ({ value: k, label: SETS[k].label })) },
      { id: 'fix', kind: 'choice', label: 'Fix', def: 'weights', options: Object.keys(FIX).map((k) => ({ value: k, label: FIX[k] })) },
      { id: 'seed', kind: 'number', label: 'Seed (which rows are copied or dropped)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Class weights', values: { set: 'toy', fix: 'weights', seed: 1 } },
      { label: 'Oversampling', values: { set: 'toy', fix: 'over', seed: 1 } },
      { label: 'Undersampling', values: { set: 'toy', fix: 'under', seed: 1 } },
      { label: 'Pump faults, weights', values: { set: 'pumps', fix: 'weights', seed: 1 } },
      { label: 'No fix', values: { set: 'toy', fix: 'none', seed: 1 } },
    ],
    counters: [{ id: 'found', label: 'rare test points found' }],
    run(inp) {
      const F = H().frames(['found']), q = F.q, base = P.imbRun(inp.set, 'none', 1), d = base.d, real = !!d.real;
      const box = { x: 30, y: 40, w: 330, h: 330 }, ax = P.axes('im:ax', d.X, box, { equal: !real, xname: d.names[0], yname: d.names[1] });
      const pts = (Xs, ys, o) => Xs.map((x, i) => ({ key: 'im:p' + (o.key || '') + i, kind: 'dot', x: ax.X(x[0]), y: ax.Y(x[1]), r: o.r ? o.r(i) : ys[i] ? 5 : 3.2, cls: 'sp c' + ys[i] + (o.cls ? ' ' + (o.cls(i) || '') : ''), audit: 'overlap-ok', layer: 'items', z: ys[i] ? 2 : 1 }));
      const nR = d.y.filter((v) => v).length, n = d.y.length, nTeR = base.yte.filter((v) => v).length;
      const legend = [...P.key('im:k0', 400, 60, 'c0', d.classes[0] + ' (class 0)'), ...P.key('im:k1', 400, 84, 'c1', d.classes[1] + ' (class 1)')];
      const conf = (m, t) => P.confusion('im:cm', 500, 172, m.cm, ['0', '1'], t);
      const lines = (m) => H().lines('im:m', 400, 300, [['accuracy ' + H().fx(m.acc, 3), 'legend'], ['recall of the rare class ' + H().fx(m.recall, 2), 'strong'], ['precision ' + H().fx(m.precision, 2) + ', F1 ' + H().fx(m.f1, 2), 'legend']], 13, 22);
      F.push([...ax.items, ...pts(d.X, d.y, {}), ...legend], `${P.qpl(q, n, 'row')}, of which ${q.n(nR)} (${q.pct(nR / n)}) ${nR === 1 ? 'is' : 'are'} of class ‘${d.classes[1]}’. A stratified split keeps that share in both parts: ${q.n(base.ytr.length)} rows to train on and ${q.n(base.yte.length)} to test, with only ${q.n(nTeR)} ${nTeR === 1 ? 'test row' : 'test rows'} of class ‘${d.classes[1]}’.`, { phase: 'data', n, nR, share: nR / n, nTr: base.ytr.length, nTe: base.yte.length, nTeR });
      // the baseline
      const zero = base.yte.map(() => 0), mb = P.imbMetrics(base.yte, zero, null);
      F.push([...ax.items, S.Kit.scatter.field(ax, 'im:f', () => 0.08, { n: 8, scheme: 'prob' }), ...pts(base.Xte, base.yte, {}), ...legend, ...conf(mb, 'always “common”'), ...lines(mb)],
        `The laziest model says “${d.classes[0]}” every time. On the test rows its accuracy is ${q(mb.acc, 3)}, which sounds excellent, yet its recall of the rare class is ${q(mb.recall, 0)}: it finds none of the ${q.n(nTeR)}. Any real model must beat this baseline on the metrics that matter.`, { phase: 'baseline', acc: mb.acc, recall: mb.recall, nTeR });
      const plainItems = (r, key) => [S.Kit.scatter.field(ax, 'im:f', (u, v) => r.prob([u, v]), { n: 60, scheme: 'prob' }), H().level(ax, 'im:lv' + (key || ''), (u, v) => r.prob([u, v]), 0.5, key ? 'margin' : '')];
      const mp = base.metrics;
      F.set('found', mp.cm[1][1]);
      F.push([...ax.items, ...plainItems(base), ...pts(base.Xte, base.yte, { cls: (i) => (base.yte[i] && !base.pred[i] ? 'wrong' : '') }), ...legend, ...conf(mp, 'logistic, no fix'), ...lines(mp)],
        `A logistic model trained as usual: accuracy ${q(mp.acc, 3)}, against the baseline’s ${q(mb.acc, 3)}. The line where it says 50% sits close to the rare points, because missing a few of them costs little in the average loss: it catches ${P.qpl(q, mp.cm[1][1], 'rare test point')} of ${q.n(nTeR)} (recall ${q(mp.recall, 2)}).`, { phase: 'plain', accB: mb.acc, acc: mp.acc, recall: mp.recall, tp: mp.cm[1][1], nTeR });
      let fx = base;
      if (inp.fix !== 'none') {
        fx = P.imbRun(inp.set, inp.fix, inp.seed);
        const mf = fx.metrics, w = fx.w, kept = w.filter((v) => v > 0).length, copies = w.reduce((s, v) => s + v, 0);
        const rW = fx.ytr.map((v, i) => (v ? w[i] : 0)).reduce((a, b) => a + b, 0), cW = fx.ytr.map((v, i) => (v ? 0 : w[i])).reduce((a, b) => a + b, 0);
        F.set('found', mf.cm[1][1]);
        const trainPts = pts(fx.Xtr, fx.ytr, { key: 't', r: (i) => (w[i] === 0 ? 3 : Math.min(9, (fx.ytr[i] ? 4.5 : 3.2) * Math.sqrt(Math.max(1, inp.fix === 'over' ? w[i] : 1)))), cls: (i) => (w[i] === 0 ? 'faded' : '') });
        const how = inp.fix === 'over' ? `Oversampling copies rare training rows at random until both classes count ${q.n(cW)}: ${P.qpl(q, copies, 'row')} in all (bigger dots were copied more).`
          : inp.fix === 'under' ? `Undersampling keeps only ${q.n(rW)} common rows, as many as the rare ones, and drops the rest (faded): ${P.qpl(q, kept, 'row')} remain.`
          : `Class weights make each rare row count ${q(w[fx.ytr.indexOf(1)], 2)} and each common row ${q(w[fx.ytr.indexOf(0)], 2)}, so both classes weigh the same in the loss.`;
        F.push([...ax.items, ...plainItems(fx), H().level(ax, 'im:old', (u, v) => base.prob([u, v]), 0.5, 'margin'), ...trainPts, ...legend, ...conf(mf, FIX[inp.fix].split(' of')[0]), ...lines(mf)],
          how + ` The 50% line (solid) moves out past the old one (dashed): on the test rows recall rises to ${q(mf.recall, 2)}, while precision is ${q(mf.precision, 2)} and accuracy ${q(mf.acc, 3)}.`, { phase: 'fix', recall: mf.recall, precision: mf.precision, acc: mf.acc, kept, copies, rW, cW, wR: w[fx.ytr.indexOf(1)], wC: w[fx.ytr.indexOf(0)] });
      }
      // the comparison table
      const rows = [['always common', mb], ['logistic', mp]].concat(inp.fix !== 'none' ? [['+ ' + inp.fix, fx.metrics]] : []);
      const tab = S.Kit.dt.table({ prefix: 'im:tb', x: 400, y: 130, cols: [{ name: '', w: 104 }, { name: 'acc', w: 46 }, { name: 'bal', w: 46 }, { name: 'rec', w: 46 }, { name: 'prec', w: 46 }, { name: 'F1', w: 46 }, { name: 'AP', w: 46 }], rows: rows.map(([nm, m]) => [nm, H().fx(m.acc, 2), H().fx(m.bal, 2), H().fx(m.recall, 2), H().fx(m.precision, 2), H().fx(m.f1, 2), m.ap === null ? 'none' : H().fx(m.ap, 2)]), rowH: 28, size: 12 });
      const mL = fx.metrics;
      F.push([...ax.items, ...plainItems(fx), ...pts(fx.Xte, fx.yte, { cls: (i) => (fx.yte[i] && !fx.pred[i] ? 'wrong' : '') }), ...legend, ...tab.items, ...H().lines('im:tn', 400, 280, [['acc accuracy · bal balanced accuracy', 'legend'], ['rec recall · prec precision of the rare class', 'legend'], ['AP average precision (area under PR)', 'legend']], 13, 21)],
        `Side by side on the test rows: accuracy barely separates the models, from ${q(mb.acc, 2)} to ${q(Math.max(mp.acc, mL.acc), 2)}. Balanced accuracy, the mean of the two class recalls, goes from ${q(mb.bal, 2)} for the baseline to ${q(mL.bal, 2)}. Average precision, which scores the model’s ranking at every threshold, is ${q(mp.ap, 2)}` + (inp.fix !== 'none' ? ` without the fix and ${q(mL.ap, 2)} with it` + (Math.abs(mL.ap - mp.ap) < 0.05 ? ': the fix mostly moves the threshold, not the ranking.' : ': the fix changed the ranking too.') : '.'), { phase: 'compare', accB: mb.acc, accMax: Math.max(mp.acc, mL.acc), balB: mb.bal, balL: mL.bal, apP: mp.ap, apL: mL.ap });
      for (const f of F.list) f.narrowView = { x: 0, y: 20, w: 400, h: 420 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Metrics for a rare class', text: ['recall = TP / (TP + FN)   rare rows found', 'precision = TP / (TP + FP)   alarms that are real', 'F1 = 2 · precision · recall / (precision + recall)', 'balanced accuracy = (recall₀ + recall₁) / 2', 'class weight wₖ = n / (2 nₖ)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'plain', pick: 'option', once: true,
      ask: 'With the fix, the recall of the rare class will…',
      options: ['go up', 'go down', 'stay the same'],
      answer: (fr, i) => { const f = fr.find((x) => x.meta.phase === 'fix'); if (!f) return null; const d = f.meta.recall - fr[i].meta.recall; return d > 1e-9 ? 'go up' : d < -1e-9 ? 'go down' : 'stay the same'; },
      explain: (fr) => (fr.find((x) => x.meta.phase === 'fix') || { say: '' }).say,
    }, {
      at: (f) => f.meta.phase === 'data', pick: 'option', once: true,
      ask: 'A model that always says “common” will have an accuracy of about…',
      options: ['95%', '50%', '5%', '0%'],
      answer: (fr) => H().pct(Math.round((1 - fr[0].meta.share) * 20) / 20),
      explain: (fr) => fr[1].say,
    }],
    quiz: [
      { q: '1,000 transactions, 10 of them fraud. A model flags none. Its accuracy is…', options: ['99%', '1%', '50%', '90%'], answer: 0, why: '990 of 1,000 are right, but it finds no fraud: recall 0.' },
      { q: 'A model flags 40 rows; 8 are real faults out of 10 faults in all. Recall and precision are…', options: ['recall 0.8, precision 0.2', 'recall 0.2, precision 0.8', 'both 0.8', 'both 0.2'], answer: 0, why: 'Recall 8 / 10, precision 8 / 40.' },
      { q: 'Why must resampling happen after the train / test split, on the training rows only?', options: ['copies of a row in both parts would leak it into the test', 'it is faster', 'the test set has no rare rows', 'it does not matter'], answer: 0, why: 'A duplicated row in the test set is a row the model has already seen.' },
      { q: 'With 900 common and 100 rare rows, the class weight n / (2 nₖ) of a rare row is…', options: ['5', '0.1', '10', '0.56'], answer: 0, why: '1000 / (2 × 100).' },
      { q: 'Which metric ignores the threshold and judges the ranking of scores?', options: ['average precision', 'accuracy', 'recall', 'F1'], answer: 0, why: 'It sweeps every threshold along the precision–recall curve.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
