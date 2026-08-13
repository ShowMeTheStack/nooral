// Part 2, scores read from the confusion matrix: accuracy against balanced accuracy when one class is rare;
// precision, recall and F1 as the threshold moves (the first frame always shows the slider's threshold); and the
// averages over many classes (macro, weighted, micro), specificity and the Matthews correlation.
(function (S) {
  'use strict';
  const H = S.P01, P = S.P02, C = () => S.Kit.curve, DT = () => S.Kit.dt, M = () => S.Metrics, ML = () => S.ML;

  /** pumps with a chosen share of faults: every fault row, and enough normal rows to make the share */
  P.imbalanced = (share) => P.once('imb:' + share, () => {
    const d = H.data('pumps', { missing: 0 }), faults = [], normal = [];
    d.y.forEach((v, i) => (v ? faults : normal).push(i));
    const nNorm = Math.min(normal.length, Math.round((faults.length * (1 - share)) / share));
    const idx = faults.concat(normal.slice(0, nNorm)).sort((a, b) => a - b), X = idx.map((i) => d.X[i]), y = idx.map((i) => d.y[i]);
    const sp = ML().trainTestSplit(X.length, { testSize: 0.4, stratify: y, seed: 2 }), tr = sp.train.slice().sort((a, b) => a - b), te = sp.test.slice().sort((a, b) => a - b);
    const sc = H.scaler(tr.map((i) => X[i]), 'standard'), Z = sc.apply(X), yt = te.map((i) => y[i]);
    const fitP = (m) => m.fit(tr.map((i) => Z[i]), tr.map((i) => y[i])).predict(te.map((i) => Z[i]));
    const models = [['always "normal"', te.map(() => 0)], ['logistic regression', fitP(ML().LogisticRegression({ C: 1 }))], ['5-NN', fitP(ML().KNeighborsClassifier({ k: 5 }))]];
    return { n: X.length, faults: faults.length, yt, models };
  });

  S.topic({
    id: 'eval-balanced', family: 'p02-confusion', order: 30, title: 'Accuracy and balanced accuracy',
    blurb: 'When 95% of the rows are normal, a model that always says "normal" is 95% accurate and completely useless. Balanced accuracy averages the recall of each class instead, so every class counts equally however rare it is: the do-nothing model scores 50%. On imbalanced data, look at the per-class recalls, or a score that weighs the classes evenly, never at accuracy alone.',
    inputs: [{ id: 'share', kind: 'slider', label: 'Share of faults', def: 0.05, min: 0.05, max: 0.5, step: 0.05, fmt: (v) => Math.round(v * 100) + '%' }],
    presets: [
      { label: '5% faults', values: { share: 0.05 } },
      { label: '20% faults', values: { share: 0.2 } },
      { label: 'Balanced: 50%', values: { share: 0.5 } },
    ],
    counters: [{ id: 'n', label: 'test rows' }],
    run(inp) {
      const R = P.imbalanced(inp.share), yt = R.yt, n = yt.length;
      const res = R.models.map(([nm, yp]) => { const cm = M().confusion(yt, yp, { labels: [0, 1] }), [[tn, fp], [fn, tp]] = cm.counts; return { nm, yp, tp, fp, fn, tn, acc: M().accuracy(yt, yp).value, bal: M().balancedAccuracy(yt, yp).value }; });
      const F = new S.Frames(['n']), view = H.box(-14, -8, 790, 470);
      const pos = yt.filter((v) => v === 1).length;
      const title = H.title('tt', 0, 6, 'Pump faults: ' + pos + ' of ' + n + ' test rows are faults (' + H.pct(pos / n) + ')');
      const cm = (r) => P.cm2({ prefix: 'cm', x: 120, y: 90, cell: 92, c: { tp: r.tp, fn: r.fn, fp: r.fp, tn: r.tn }, posName: 'fault', negName: 'normal' });
      const lines = (r) => [['model: ' + r.nm, 'act'], ['accuracy = ' + (r.tp + r.tn) + ' / ' + n + ' = ' + H.f(r.acc, 3), 'strong'], ['recall of faults = ' + r.tp + ' / ' + (r.tp + r.fn) + ' = ' + H.f(M().div(r.tp, r.tp + r.fn), 3), 'strong'], ['recall of normal = ' + r.tn + ' / ' + (r.tn + r.fp) + ' = ' + H.f(M().div(r.tn, r.tn + r.fp), 3), 'strong'], ['balanced accuracy = their mean = ' + H.f(r.bal, 3), 'good']].map((l, i) => H.text('ln' + i, 360, 110 + i * 28, l[0], l[1], 14));
      F.set('n', n);
      res.forEach((r, k) => {
        const say = k === 0
          ? `The laziest model says "normal" every time. It is right on all ${r.tn} normal rows, so its accuracy is ${H.f(r.acc, 3)}, yet it finds none of the ${pos} faults. Its recalls are 1 and 0, and balanced accuracy, their mean, is ${H.f(r.bal, 3)}.`
          : `${H.cap(r.nm)}: accuracy ${H.f(r.acc, 3)}; it catches ${r.tp} of ${pos} faults (recall ${H.f(M().div(r.tp, pos), 3)}) and ${r.tn} of ${n - pos} normal rows, so balanced accuracy is ${H.f(r.bal, 3)}.`;
        F.add({ items: [title, ...cm(r), ...lines(r)], say, meta: { phase: 'model', k, acc: r.acc, bal: r.bal, tp: r.tp, fn: r.fn, fp: r.fp, tn: r.tn }, view });
      });
      const vals = res.flatMap((r) => [r.acc, r.bal]), names = res.flatMap((r) => [r.nm.replace('logistic regression', 'logistic').replace('always "normal"', 'always normal') + ': acc', 'balanced']);
      F.add({ items: [title, H.title('bt', 0, 40, 'accuracy and balanced accuracy'), ...DT().hbars({ prefix: 'cmp', x: 0, y: 56, w: 600, rowH: 30, vals, names, max: 1, fmt: (v) => H.f(v, 3), nameW: 200, valW: 60, cls: (i) => (i % 2 ? 'good' : 'soft') })], say: `Side by side: accuracy ranks the models ${H.list(res.slice().sort((a, b) => b.acc - a.acc).map((r) => r.nm + ' ' + H.f(r.acc, 3)))}; balanced accuracy ranks them ${H.list(res.slice().sort((a, b) => b.bal - a.bal).map((r) => r.nm + ' ' + H.f(r.bal, 3)))}.` + (inp.share >= 0.5 ? ' With balanced classes the two scores nearly agree.' : ''), meta: { phase: 'compare', acc: res.map((r) => r.acc), bal: res.map((r) => r.bal) }, view });
      const ctx = { n, pos, neg: n - pos, recallPos: res.map((r) => M().div(r.tp, pos)), names: res.map((r) => r.nm) };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Two accuracies', text: ['accuracy = (TP + TN) / n', 'balanced = (TP/(TP + FN) + TN/(TN + FP)) / 2', 'many classes: the mean of the class recalls', 'a constant guess scores 1/k balanced', 'imbalanced data: never trust accuracy alone'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'model' && f.meta.k === 0, pick: 'option', ask: 'Which model will have the highest balanced accuracy?', options: ['always "normal"', 'logistic regression', '5-NN'], answer: (fr) => { const b = fr.find((f) => f.meta.phase === 'compare').meta.bal; return ['always "normal"', 'logistic regression', '5-NN'][b.indexOf(Math.max(...b))]; }, explain: (fr) => fr.find((f) => f.meta.phase === 'compare').say },
    ],
    quiz: [
      { q: '99% of transactions are honest. A model that never flags fraud has accuracy…', options: ['99%', '50%', '1%', '0%'], answer: 0, why: 'It is right on every honest transaction.' },
      { q: '…and balanced accuracy…', options: ['50%', '99%', '1%', '0%'], answer: 0, why: 'Recall 1 on honest, 0 on fraud: mean 0.5.' },
      { q: 'Recall on class A is 0.9 and on class B 0.6. Balanced accuracy is…', options: ['0.75', '0.6', '0.9', 'it depends on the class sizes'], answer: 0, why: 'The plain mean of the recalls, whatever the class sizes.' },
      { q: 'On balanced classes, accuracy and balanced accuracy…', options: ['are nearly the same', 'always differ a lot', 'are both 50%', 'cannot be computed'], answer: 0, why: 'With equal class sizes accuracy is the mean of the recalls.' },
      { q: 'Three classes, a model that always predicts the same class. Balanced accuracy is…', options: ['1/3', '0', '1', 'the size of that class'], answer: 0, why: 'Recalls 1, 0 and 0.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Precision, recall and F1 against the threshold
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'eval-prf', family: 'p02-threshold', order: 10, title: 'Precision, recall and F1',
    blurb: 'A classifier\'s score becomes a decision at a threshold. Lower it and more examples are flagged: recall (the share of positives caught) rises, and precision (the share of flags that are right) usually falls. F1, the harmonic mean of the two, is high only when both are. Which balance is right depends on the cost of a miss against the cost of a false alarm. Move the slider: the first frame always shows your threshold.',
    inputs: [
      { id: 'src', kind: 'choice', label: 'Task', def: 'loans', options: P.sourceOptions(['loans', 'cancer', 'pumps', 'moons']) },
      { id: 't', kind: 'slider', label: 'Threshold', def: 0.5, min: 0.05, max: 0.95, step: 0.01, fmt: (v) => (+v).toFixed(2) },
    ],
    presets: [
      { label: 'Loans at 0.5', values: { src: 'loans', t: 0.5 } },
      { label: 'Loans at 0.2', values: { src: 'loans', t: 0.2 } },
      { label: 'Cancer at 0.5', values: { src: 'cancer', t: 0.5 } },
      { label: 'Rare pump faults at 0.3', values: { src: 'pumps', t: 0.3 } },
    ],
    counters: [{ id: 'flag', label: 'flagged' }],
    run(inp) {
      const B = P.binary(inp.src), n = B.y.length, pos = B.y.filter((v) => v === 1).length;
      const grid = Array.from({ length: 91 }, (_, k) => 0.05 + k * 0.01);
      const cur = grid.map((t) => P.at(B, t));
      const best = cur.reduce((b, c, k) => (c.f1 > cur[b].f1 ? k : b), 0);
      const ax = C().axes({ prefix: 'ax', x: 50, y: 50, w: 330, h: 300, xr: [0, 1], yr: [0, 1], xname: 'threshold', yname: 'score', tickSize: 12 });
      const sx = C().axes({ prefix: 'sx', x: 440, y: 290, w: 330, h: 90, xr: [0, 1], yr: [0, 1], yticks: 0, xname: 'P(' + B.posName + ')', grid: false, tickSize: 12 });
      const curves = [C().line(ax, 'cp', grid.map((t, k) => [t, cur[k].tp + cur[k].fp ? cur[k].precision : NaN]), 'c0'), C().line(ax, 'cr', grid.map((t, k) => [t, cur[k].recall]), 'c1'), C().line(ax, 'cf', grid.map((t, k) => [t, cur[k].f1]), 'c2')];
      const legend = C().legend('lg', 60, 420, [{ text: 'precision', cls: 'c0' }, { text: 'recall', cls: 'c1' }, { text: 'F1', cls: 'c2' }]);
      const F = new S.Frames(['flag']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, B.title + ': ' + n + ' test examples, ' + pos + ' ' + B.posName);
      const state = (t, c, lit) => [title, ...ax.items, ...curves, ...legend, C().vline(ax, 'tv', t, 'act'), ...[['dp', c.precision, 'c0'], ['dr', c.recall, 'c1'], ['df', c.f1, 'c2']].map(([k, v, cls]) => Object.assign(C().dot(ax, k, t, v, cls), { point: false, audit: 'overlap-ok' })),
        ...P.cm2({ prefix: 'cm', x: 560, y: 80, cell: 70, c, posName: B.posName, negName: B.negName }),
        ...sx.items, ...P.strip(sx, 's', B, { t, mark: true }).map((d) => Object.assign(d, { r: 2.4 })), { key: 'st', kind: 'path', d: `M${sx.X(t)},${sx.o.y} V${sx.o.y + sx.o.h}`, cls: 'thr', audit: 'ignore', layer: 'lines' },
        H.text('vals', 440, 268, 'P ' + H.f(c.precision, 3) + '   R ' + H.f(c.recall, 3) + '   F1 ' + H.f(c.f1, 3), lit ? 'act' : 'strong', 14)];
      const sayAt = (t, c) => `At threshold ${H.f(t, 2)}: ${c.tp + c.fp} flagged, TP = ${c.tp}, FP = ${c.fp}, FN = ${c.fn}. Precision ${c.tp} / ${c.tp + c.fp} = ${H.f(c.precision, 3)}, recall ${c.tp} / ${pos} = ${H.f(c.recall, 3)}, F1 = 2PR / (P + R) = ${H.f(c.f1, 3)}.`;
      const c0 = P.at(B, inp.t);
      F.set('flag', c0.tp + c0.fp);
      F.add({ items: state(inp.t, c0, true), say: 'Your threshold. ' + sayAt(inp.t, c0), meta: { phase: 'yours', t: inp.t, tp: c0.tp, fp: c0.fp, fn: c0.fn, p: c0.precision, r: c0.recall, f1: c0.f1 }, view });
      for (const t of [0.9, 0.7, 0.5, 0.3, 0.1]) {
        const c = P.at(B, t);
        F.set('flag', c.tp + c.fp);
        F.add({ items: state(t, c, false), say: sayAt(t, c), meta: { phase: 'sweep', t, tp: c.tp, fp: c.fp, fn: c.fn, p: c.precision, r: c.recall, f1: c.f1 }, view });
      }
      const cb = cur[best], tb = grid[best];
      F.set('flag', cb.tp + cb.fp);
      F.add({ items: state(tb, cb, true), say: `Over thresholds from 0.05 to 0.95 in steps of 0.01, F1 is highest at ${H.f(tb, 2)}: precision ${H.f(cb.precision, 3)}, recall ${H.f(cb.recall, 3)}, F1 ${H.f(cb.f1, 3)}. That balances the two; if a miss costs far more than a false alarm, choose a lower threshold.`, meta: { phase: 'best', t: tb, f1: cb.f1, p: cb.precision, r: cb.recall }, view });
      const ctx = { n, pos, grid: [0.05, 0.95, 0.01] };
      F.list.forEach((f) => { f.meta.ctx = ctx; f.meta.flagged = f.meta.tp + f.meta.fp; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Three scores', text: ['precision = TP / (TP + FP)', 'recall = TP / (TP + FN)', 'F1 = 2·P·R / (P + R)', 'lower threshold: recall up, precision usually down', 'F1 is high only when both are'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'sweep' && f.meta.t === 0.7, pick: 'option', ask: 'Lowering the threshold to 0.5, recall will…', options: ['rise or stay', 'fall'], answer: 'rise or stay', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'TP = 40, FP = 10, FN = 40. F1 is…', options: ['0.615', '0.8', '0.5', '0.65'], answer: 0, why: 'P = 0.8, R = 0.5, F1 = 2·0.4/1.3 ≈ 0.615.' },
      { q: 'Lowering the threshold can never…', options: ['lower the recall', 'lower the precision', 'add false positives', 'flag more examples'], answer: 0, why: 'Every example flagged before stays flagged, so TP can only grow.' },
      { q: 'For screening a serious disease you would favour…', options: ['high recall', 'high precision', 'neither', 'the threshold 0.5 always'], answer: 0, why: 'A miss is far worse than a follow-up test.' },
      { q: 'Why the harmonic mean in F1 rather than the plain mean?', options: ['it stays low when either score is low', 'it is easier to compute', 'it equals accuracy', 'it ignores recall'], answer: 0, why: 'P = 1 and R = 0.01 give a plain mean near 0.5 but F1 ≈ 0.02.' },
      { q: 'A spam filter where a lost real email is costly should favour…', options: ['high precision', 'high recall', 'a threshold of 0', 'flagging everything'], answer: 0, why: 'Of the flagged messages, nearly all should really be spam.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Averages over classes, specificity, MCC
  // ---------------------------------------------------------------------------------------------------------------
  const AVG = {
    flowers: { label: 'Wildflowers: balanced classes', make: () => { const R = P.multi('flowers', 'nb'); return { yt: R.yt, yp: R.yp, classes: R.classes }; } },
    pumps: { label: 'Pump fault kinds, no vibration sensor', make: () => P.once('pkinds', () => { const d = H.data('pumps', { missing: 0 }), y = d.kind.map((k) => d.kinds.indexOf(k)); const sp = ML().trainTestSplit(y.length, { testSize: 0.4, stratify: y, seed: 5 }); const tr = sp.train.slice().sort((a, b) => a - b), te = sp.test.slice().sort((a, b) => a - b); const X = d.X.map((r) => [r[0], r[2], r[3]]); const m = ML().DecisionTreeClassifier({ maxDepth: 3, seed: 0 }).fit(tr.map((i) => X[i]), tr.map((i) => y[i])); return { yt: te.map((i) => y[i]), yp: m.predict(te.map((i) => X[i])), classes: d.kinds }; }) },
    wine: { label: 'Wine: unscaled 5-NN', make: () => { const R = P.multi('wine', 'knn'); return { yt: R.yt, yp: R.yp, classes: R.classes }; } },
  };
  P.AVG = AVG;
  S.topic({
    id: 'eval-averages', family: 'p02-confusion', order: 40, title: 'Macro, micro and weighted averages; MCC',
    blurb: 'With many classes, each class has its own precision, recall and F1, from its own one-against-the-rest counts. Macro averaging takes their plain mean, so a tiny class counts as much as a huge one. Weighted averaging weights each class by its size. Micro averaging pools the counts of all classes first, which for single-label problems equals accuracy. The Matthews correlation uses the whole matrix at once and stays near 0 for a model that guesses, however imbalanced the classes.',
    inputs: [{ id: 'data', kind: 'choice', label: 'Data', def: 'pumps', options: Object.entries(AVG).map(([value, a]) => ({ value, label: a.label })) }],
    presets: Object.entries(AVG).map(([value, a]) => ({ label: a.label, values: { data: value } })),
    counters: [{ id: 'rows', label: 'classes scored' }],
    run(inp) {
      const R = AVG[inp.data].make(), labels = R.classes.map((_, k) => k), m = M();
      const per = m.precisionRecallF1(R.yt, R.yp, { average: null, labels }), cm = m.confusion(R.yt, R.yp, { labels }), pl = m.perLabel(cm);
      const mac = m.precisionRecallF1(R.yt, R.yp, { average: 'macro', labels }), wei = m.precisionRecallF1(R.yt, R.yp, { average: 'weighted', labels }), mic = m.precisionRecallF1(R.yt, R.yp, { average: 'micro', labels });
      const spec = m.specificity(R.yt, R.yp, { average: 'macro' }), mcc = m.mcc(R.yt, R.yp).value, acc = m.accuracy(R.yt, R.yp).value;
      const F = new S.Frames(['rows']), view = H.box(-14, -8, 790, 470), n = R.yt.length;
      const cols = [{ name: 'class', w: 128, cls: 'cat' }, { name: 'support', w: 72, cls: 'num' }, { name: 'TP', w: 50, cls: 'num' }, { name: 'FP', w: 50, cls: 'num' }, { name: 'FN', w: 50, cls: 'num' }, { name: 'precision', w: 86, cls: 'num' }, { name: 'recall', w: 72, cls: 'num' }, { name: 'F1', w: 64, cls: 'num' }, { name: 'specif.', w: 72, cls: 'num' }];
      const row = (k) => [R.classes[k], String(pl[k].support), String(pl[k].tp), String(pl[k].fp), String(pl[k].fn), H.f(per.precision[k], 3), H.f(per.recall[k], 3), H.f(per.f1[k], 3), H.f(spec.perClass[k].value, 3)];
      const title = H.title('tt', 0, 6, AVG[inp.data].label + ': ' + n + ' test rows, ' + labels.length + ' classes');
      const table = (upto, extra) => DT().table({ prefix: 'tb', x: 0, y: 30, cols, rows: labels.slice(0, upto).map(row).concat(extra || []), rowH: 30, size: 13, cls: (i) => (i >= upto ? 'act' : '') });
      for (let k = 0; k < labels.length; k++) {
        F.count('rows');
        F.add({ items: [title, ...table(k + 1).items], say: `Class ${R.classes[k]} against the rest: ${pl[k].support} true members, TP = ${pl[k].tp}, FP = ${pl[k].fp}, FN = ${pl[k].fn}. Precision ${H.f(per.precision[k], 3)}, recall ${H.f(per.recall[k], 3)}, F1 ${H.f(per.f1[k], 3)}, specificity ${H.f(spec.perClass[k].value, 3)}.`, meta: { phase: 'class', k, tp: pl[k].tp, fp: pl[k].fp, fn: pl[k].fn, p: per.precision[k], r: per.recall[k], f1: per.f1[k], spec: spec.perClass[k].value }, view });
      }
      const avgRows = [['macro', '', '', '', '', H.f(mac.precision, 3), H.f(mac.recall, 3), H.f(mac.f1, 3), H.f(spec.value, 3)], ['weighted', '', '', '', '', H.f(wei.precision, 3), H.f(wei.recall, 3), H.f(wei.f1, 3), '']];
      F.add({ items: [title, ...table(labels.length, avgRows).items], say: `Macro F1, the plain mean of the ${labels.length} class F1s, is ${H.f(mac.f1, 3)}; weighted by support it is ${H.f(wei.f1, 3)}.` + (Math.abs(mac.f1 - wei.f1) > 0.05 ? ' The gap shows that some small classes fare worse than the big ones.' : ''), meta: { phase: 'macro', macroF1: mac.f1, weightedF1: wei.f1, macroP: mac.precision, macroR: mac.recall }, view });
      const tpS = H.sum(pl.map((q) => q.tp)), fpS = H.sum(pl.map((q) => q.fp)), fnS = H.sum(pl.map((q) => q.fn));
      const all3 = avgRows.concat([['micro', String(n), String(tpS), String(fpS), String(fnS), H.f(mic.precision, 3), H.f(mic.recall, 3), H.f(mic.f1, 3), '']]);
      F.add({ items: [title, ...table(labels.length, all3).items], say: `Micro averaging pools the counts first: TP = ${tpS}, FP = ${fpS}, FN = ${fnS}, so precision = recall = F1 = ${H.f(mic.f1, 3)}, exactly the accuracy ${H.f(acc, 3)}: every mistake is one FP for some class and one FN for another.`, meta: { phase: 'micro', tp: tpS, fp: fpS, fn: fnS, micro: mic.f1, acc }, view });
      const mt = m.mcc(R.yt, R.yp).trace.find((s) => s.kind === 'mcc');
      F.add({ items: [title, ...table(labels.length, all3).items, H.text('mcc1', 0, 380, 'MCC = (c·n − Σ tₖpₖ) / √((n² − Σ pₖ²)(n² − Σ tₖ²))', 'math', 14), H.text('mcc2', 0, 408, 'c = ' + mt.c + ' correct, n = ' + n + ':  MCC = ' + H.f(mcc, 3), 'act', 15)], say: `The Matthews correlation uses the whole matrix: with c = ${mt.c} correct of n = ${n}, true class totals tₖ and predicted totals pₖ, MCC = ${H.f(mcc, 3)}. It is 1 for a perfect model, about 0 for guessing, and is not fooled by a big class.`, meta: { phase: 'mcc', mcc, c: mt.c }, view });
      const ctx = { n, K: labels.length, support: pl.map((q) => q.support), names: R.classes };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Averages', text: ['per class: one against the rest', 'macro = mean over classes', 'weighted = Σ supportₖ·scoreₖ / n', 'micro: pool TP, FP, FN, then score', '  single-label: micro P = R = F1 = accuracy', 'specificity = TN / (TN + FP)', 'MCC ∈ [−1, 1], 0 ≈ chance'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'class' && f.meta.k === 0, pick: 'option', ask: 'Which will be larger: macro F1 or weighted F1?', options: ['macro', 'weighted', 'equal'], answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'macro').meta; return m.macroF1 > m.weightedF1 + 1e-9 ? 'macro' : m.weightedF1 > m.macroF1 + 1e-9 ? 'weighted' : 'equal'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'macro').say },
    ],
    quiz: [
      { q: 'Class F1 scores 0.9 (900 rows) and 0.3 (100 rows). Macro F1 is…', options: ['0.6', '0.84', '0.9', '0.3'], answer: 0, why: 'The plain mean (0.9 + 0.3) / 2.' },
      { q: '…and weighted F1 is…', options: ['0.84', '0.6', '0.9', '0.3'], answer: 0, why: '(900·0.9 + 100·0.3) / 1000.' },
      { q: 'In a single-label multi-class problem, micro-averaged F1 equals…', options: ['accuracy', 'macro F1', 'balanced accuracy', 'MCC'], answer: 0, why: 'Pooled FP and FN are both the number of mistakes.' },
      { q: 'A model that always predicts the majority class has MCC…', options: ['0', '1', '−1', 'the majority share'], answer: 0, why: 'Its predictions carry no information about the truth.' },
      { q: 'Which average should you watch if every class matters equally, including rare ones?', options: ['macro', 'micro', 'weighted', 'accuracy'], answer: 0, why: 'It gives each class the same weight.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
