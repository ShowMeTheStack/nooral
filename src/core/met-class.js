// Classification metrics: the confusion matrix (with the examples in each cell), accuracy, balanced accuracy,
// precision, recall and F1 (binary, macro, micro, weighted), specificity, MCC and Cohen's kappa.
(function (S) {
  'use strict';
  const M = (S.Metrics = S.Metrics || {});
  const f = (x) => M.f(x), st = (k, s, d) => M.step(k, s, d);

  /**
   * confusion(yTrue, yPred, {labels, normalize}) → { labels, counts, matrix, cells, n, trace }
   * counts[i][j] = examples whose true label is labels[i] and predicted label is labels[j];
   * cells[i][j] = their indices; normalize 'true' divides each row by its total, 'pred' each column, 'all' by n.
   */
  M.confusion = (yTrue, yPred, opt = {}) => {
    M.check(yTrue, yPred, 'confusion');
    const labels = opt.labels ? opt.labels.slice() : M.labels(yTrue, yPred);
    const ix = M.index(labels), K = labels.length;
    const counts = labels.map(() => new Array(K).fill(0));
    const cells = labels.map(() => labels.map(() => []));
    let n = 0;
    for (let k = 0; k < yTrue.length; k++) {
      const i = ix(yTrue[k]), j = ix(yPred[k]);
      if (i < 0 || j < 0) continue;
      counts[i][j]++; cells[i][j].push(k); n++;
    }
    const rows = counts.map((r) => M.sum(r)), cols = labels.map((_, j) => M.sum(counts.map((r) => r[j])));
    const norm = opt.normalize || null;
    const matrix = counts.map((r, i) => r.map((c, j) => (norm === 'true' ? M.div(c, rows[i]) : norm === 'pred' ? M.div(c, cols[j]) : norm === 'all' ? M.div(c, n) : c)));
    const correct = M.sum(labels.map((_, i) => counts[i][i]));
    const trace = [st('count', 'Counted ' + n + ' examples into a ' + K + '×' + K + ' table: rows are the true labels, columns the predicted ones.', { counts }),
      st('diagonal', correct + ' of ' + n + ' examples lie on the diagonal (predicted correctly) and ' + (n - correct) + ' lie off it.', { correct })];
    if (norm) trace.push(st('normalise', norm === 'true' ? 'Each row was divided by its row total (' + rows.join(', ') + '), so a row shows where the examples of one true class went.' : norm === 'pred' ? 'Each column was divided by its column total (' + cols.join(', ') + '), so a column shows what a prediction really was.' : 'Every cell was divided by n = ' + n + '.', { matrix }));
    return { labels, counts, matrix, cells, n, rowTotals: rows, colTotals: cols, value: counts, trace };
  };

  /** the per-label counts tp, fp, fn, tn read from a confusion table */
  M.perLabel = (cm) => cm.labels.map((label, i) => {
    const tp = cm.counts[i][i], fn = cm.rowTotals[i] - tp, fp = cm.colTotals[i] - tp;
    return { label, tp, fp, fn, tn: cm.n - tp - fp - fn, support: cm.rowTotals[i] };
  });

  M.accuracy = (yTrue, yPred) => {
    M.check(yTrue, yPred, 'accuracy');
    let c = 0; for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) c++;
    const value = M.div(c, yTrue.length, NaN);
    return { value, correct: c, n: yTrue.length, trace: [st('accuracy', 'Accuracy = correct / all = ' + c + ' / ' + yTrue.length + ' = ' + f(value) + '.', { correct: c, n: yTrue.length })] };
  };

  /** the mean of the recalls of the classes that occur in yTrue */
  M.balancedAccuracy = (yTrue, yPred) => {
    const cm = M.confusion(yTrue, yPred), per = M.perLabel(cm).filter((p) => p.support > 0);
    const recalls = per.map((p) => p.tp / p.support), value = M.mean(recalls);
    return { value, recalls, labels: per.map((p) => p.label), trace: [
      ...per.map((p, i) => st('recall', 'Class ' + p.label + ': recall = ' + p.tp + ' / ' + p.support + ' = ' + f(recalls[i]) + '.', { label: p.label })),
      st('mean', 'Balanced accuracy is the mean of the ' + per.length + ' class recalls = ' + f(value) + '.', { value })] };
  };

  /**
   * precisionRecallF1(yTrue, yPred, {average: 'binary'|'macro'|'micro'|'weighted'|null, posLabel: 1, labels, beta: 1})
   * → { precision, recall, f1, support, perClass, trace }; a zero denominator gives 0.
   */
  M.precisionRecallF1 = (yTrue, yPred, opt = {}) => {
    const average = opt.average === undefined ? 'binary' : opt.average, beta = opt.beta == null ? 1 : opt.beta, b2 = beta * beta;
    const posLabel = opt.posLabel === undefined ? 1 : opt.posLabel;
    const all = M.labels(yTrue, yPred);
    let labels = opt.labels || all;
    if (average === 'binary') {
      if (all.length > 2) throw new Error('average "binary" needs two classes, but there are ' + all.length + '; choose macro, micro or weighted');
      if (all.length === 2 && M.index(all)(posLabel) < 0) throw new Error('the positive label ' + posLabel + ' is not one of ' + all.join(', '));
      labels = [posLabel];
    }
    const cm = M.confusion(yTrue, yPred, { labels: M.labels(all, labels) });
    const byLabel = new Map(M.perLabel(cm).map((p) => [typeof p.label + ':' + p.label, p]));
    const perClass = labels.map((l) => {
      const p = byLabel.get(typeof l + ':' + l) || { label: l, tp: 0, fp: 0, fn: 0, tn: 0, support: 0 };
      const precision = M.div(p.tp, p.tp + p.fp), recall = M.div(p.tp, p.tp + p.fn);
      const f1 = M.div((1 + b2) * p.tp, (1 + b2) * p.tp + b2 * p.fn + p.fp);
      return Object.assign({}, p, { precision, recall, f1 });
    });
    const trace = perClass.map((p) => st('class', 'Class ' + p.label + ': TP = ' + p.tp + ', FP = ' + p.fp + ', FN = ' + p.fn + ', so precision = ' + p.tp + ' / ' + (p.tp + p.fp) + ' = ' + f(p.precision) + ', recall = ' + p.tp + ' / ' + (p.tp + p.fn) + ' = ' + f(p.recall) + ', F' + beta + ' = ' + f(p.f1) + '.', { label: p.label, tp: p.tp, fp: p.fp, fn: p.fn }));
    const support = M.sum(perClass.map((p) => p.support));
    let precision, recall, f1;
    if (average === 'binary' || average === null) {
      if (average === null) return { precision: perClass.map((p) => p.precision), recall: perClass.map((p) => p.recall), f1: perClass.map((p) => p.f1), support: perClass.map((p) => p.support), labels, perClass, trace };
      ({ precision, recall, f1 } = perClass[0]);
    } else if (average === 'micro') {
      const tp = M.sum(perClass.map((p) => p.tp)), fp = M.sum(perClass.map((p) => p.fp)), fn = M.sum(perClass.map((p) => p.fn));
      precision = M.div(tp, tp + fp); recall = M.div(tp, tp + fn); f1 = M.div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp);
      trace.push(st('micro', 'Micro average pools the counts first: TP = ' + tp + ', FP = ' + fp + ', FN = ' + fn + ', giving precision ' + f(precision) + ', recall ' + f(recall) + ', F' + beta + ' ' + f(f1) + '.', { tp, fp, fn }));
    } else if (average === 'macro' || average === 'weighted') {
      const w = perClass.map((p) => (average === 'macro' ? 1 : p.support)), W = M.sum(w);
      const avg = (k) => M.div(M.sum(perClass.map((p, i) => w[i] * p[k])), W);
      precision = avg('precision'); recall = avg('recall'); f1 = avg('f1');
      trace.push(st(average, (average === 'macro' ? 'Macro average gives every class the same weight: ' : 'Weighted average weights each class by its support (' + w.join(', ') + '): ') + 'precision ' + f(precision) + ', recall ' + f(recall) + ', F' + beta + ' ' + f(f1) + '.', { weights: w }));
    } else throw new Error('unknown average ' + average);
    return { precision, recall, f1, value: f1, support, labels, perClass, trace };
  };
  M.precision = (t, p, o) => { const r = M.precisionRecallF1(t, p, o); return { value: r.precision, trace: r.trace }; };
  M.recall = (t, p, o) => { const r = M.precisionRecallF1(t, p, o); return { value: r.recall, trace: r.trace }; };
  M.f1 = (t, p, o) => { const r = M.precisionRecallF1(t, p, o); return { value: r.f1, trace: r.trace }; };

  /** specificity = TN / (TN + FP): for posLabel (binary), or averaged over classes one-against-rest (macro) */
  M.specificity = (yTrue, yPred, opt = {}) => {
    const cm = M.confusion(yTrue, yPred), per = M.perLabel(cm);
    const one = (p) => ({ label: p.label, tn: p.tn, fp: p.fp, value: M.div(p.tn, p.tn + p.fp) });
    const say = (s) => st('specificity', 'Class ' + s.label + ': specificity = TN / (TN + FP) = ' + s.tn + ' / ' + (s.tn + s.fp) + ' = ' + f(s.value) + '.', s);
    if ((opt.average || 'binary') === 'binary') {
      const pos = opt.posLabel === undefined ? 1 : opt.posLabel;
      const p = per.find((q) => q.label === pos) || { label: pos, tn: M.sum(cm.counts.map((r) => M.sum(r))), fp: 0 };
      const s = one(p); return { value: s.value, tn: s.tn, fp: s.fp, trace: [say(s)] };
    }
    const all = per.map(one), value = M.mean(all.map((s) => s.value));
    return { value, perClass: all, trace: [...all.map(say), st('mean', 'Macro specificity = mean of ' + all.length + ' values = ' + f(value) + '.')] };
  };

  /** Matthews correlation coefficient, the multi-class form computed from the confusion table */
  M.mcc = (yTrue, yPred) => {
    const cm = M.confusion(yTrue, yPred), C = cm.counts, n = cm.n;
    const t = cm.rowTotals, p = cm.colTotals, c = M.sum(C.map((r, i) => r[i]));
    const dot = (a, b) => M.sum(a.map((v, i) => v * b[i]));
    const cov = c * n - dot(t, p), vp = n * n - dot(p, p), vt = n * n - dot(t, t);
    const value = vp * vt === 0 ? 0 : cov / Math.sqrt(vp * vt);
    const trace = [st('mcc', 'MCC = (c·n − Σ tₖpₖ) / √((n² − Σ pₖ²)(n² − Σ tₖ²)) with c = ' + c + ' correct of n = ' + n + ': ' + cov + ' / √(' + vp + ' × ' + vt + ') = ' + f(value) + '.', { c, n, cov, vp, vt })];
    if (C.length === 2) { const [[tn, fp], [fn, tp]] = C; trace.unshift(st('counts', 'TP = ' + tp + ', TN = ' + tn + ', FP = ' + fp + ', FN = ' + fn + '.', { tp, tn, fp, fn })); }
    if (vp * vt === 0) trace.push(st('undefined', 'One side predicts or contains a single class, so the denominator is 0 and MCC is set to 0.'));
    return { value, trace };
  };

  /** Cohen's kappa: agreement beyond what the two label totals would give by chance (NaN when undefined) */
  M.kappa = (y1, y2, opt = {}) => {
    const cm = M.confusion(y1, y2, { labels: opt.labels }), C = cm.counts, n = cm.n, K = C.length;
    if (!n) return { value: NaN, trace: [st('undefined', 'No examples, so kappa is undefined.')] };
    const w = (i, j) => (opt.weights === 'linear' ? Math.abs(i - j) : opt.weights === 'quadratic' ? (i - j) ** 2 : i === j ? 0 : 1);
    let num = 0, den = 0;
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) { num += w(i, j) * C[i][j]; den += w(i, j) * cm.rowTotals[i] * cm.colTotals[j] / n; }
    const po = M.sum(C.map((r, i) => r[i])) / n, pe = M.sum(cm.rowTotals.map((r, i) => r * cm.colTotals[i])) / (n * n);
    const value = den === 0 ? NaN : 1 - num / den;
    return { value, po, pe, trace: [
      st('observed', 'Observed agreement pₒ = ' + Math.round(po * n) + ' / ' + n + ' = ' + f(po) + '.', { po }),
      st('chance', 'Chance agreement pₑ = Σ (row total × column total) / n² = ' + f(pe) + '.', { pe }),
      st('kappa', den === 0 ? 'Only one label is used, so chance agreement is already perfect and kappa is undefined.' : 'κ = (pₒ − pₑ) / (1 − pₑ) = ' + f(value) + '.', { value })] };
  };

  /** every classification score at once, for an Evaluate block */
  M.report = (yTrue, yPred, opt = {}) => {
    const labels = M.labels(yTrue, yPred), bin = labels.length <= 2;
    const avg = opt.average || (bin ? 'binary' : 'macro');
    const prf = M.precisionRecallF1(yTrue, yPred, { average: avg, posLabel: opt.posLabel !== undefined ? opt.posLabel : bin ? labels[labels.length - 1] : 1 });
    return {
      confusion: M.confusion(yTrue, yPred), accuracy: M.accuracy(yTrue, yPred).value, balancedAccuracy: M.balancedAccuracy(yTrue, yPred).value,
      precision: prf.precision, recall: prf.recall, f1: prf.f1, mcc: M.mcc(yTrue, yPred).value, kappa: M.kappa(yTrue, yPred).value, average: avg, perClass: prf.perClass,
    };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
