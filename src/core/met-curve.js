// Threshold curves and probability scores: ROC with thresholds and AUC, the precision–recall curve and average
// precision, log loss, the Brier score and the calibration (reliability) curve with uniform or quantile bins.
// Tied scores are grouped into one threshold, so a tie moves the curve diagonally, as the standard definition does.
(function (S) {
  'use strict';
  const M = (S.Metrics = S.Metrics || {});
  const f = (x) => M.f(x), st = (k, s, d) => M.step(k, s, d);

  /** the positive label: the one asked for, else 1 when present, else the largest label */
  const pickPos = (yTrue, pos) => {
    if (pos !== undefined) return pos;
    const ls = M.labels(yTrue);
    return ls.includes(1) || ls.length < 2 ? 1 : ls[ls.length - 1];
  };

  /**
   * the counts at each distinct threshold, from the highest score down:
   * { thresholds, tps, fps } where tps[k], fps[k] count examples with score ≥ thresholds[k]
   */
  M.thresholdCounts = (yTrue, score, posLabel) => {
    M.check(yTrue, score, 'curve');
    const pos = pickPos(yTrue, posLabel);
    const order = score.map((_, i) => i).sort((a, b) => score[b] - score[a]);
    const thresholds = [], tps = [], fps = [];
    let tp = 0;
    for (let k = 0; k < order.length; k++) {
      if (yTrue[order[k]] === pos) tp++;
      if (k === order.length - 1 || score[order[k + 1]] !== score[order[k]]) { thresholds.push(score[order[k]]); tps.push(tp); fps.push(k + 1 - tp); }
    }
    return { thresholds, tps, fps, order, pos };
  };

  /** the area under a curve by the trapezoid rule; x must be monotone */
  M.auc = (x, y) => {
    let a = 0;
    for (let i = 1; i < x.length; i++) a += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2;
    return x.length > 1 && x[x.length - 1] < x[0] ? -a : a;
  };

  /**
   * roc(yTrue, score, {posLabel, dropIntermediate: true}) → { fpr, tpr, thresholds, tps, fps, auc, trace }
   * thresholds[0] is +∞ (nothing predicted positive); a point is (fpr, tpr) when "positive" means score ≥ threshold.
   */
  M.roc = (yTrue, score, opt = {}) => {
    const c = M.thresholdCounts(yTrue, score, opt.posLabel);
    let { thresholds, tps, fps } = c;
    if (opt.dropIntermediate !== false && fps.length > 2) {
      const keep = [0];
      for (let i = 1; i < fps.length - 1; i++) if (fps[i - 1] - 2 * fps[i] + fps[i + 1] !== 0 || tps[i - 1] - 2 * tps[i] + tps[i + 1] !== 0) keep.push(i);
      keep.push(fps.length - 1);
      thresholds = keep.map((i) => thresholds[i]); tps = keep.map((i) => tps[i]); fps = keep.map((i) => fps[i]);
    }
    tps = [0, ...tps]; fps = [0, ...fps]; thresholds = [Infinity, ...thresholds];
    const P = tps[tps.length - 1], N = fps[fps.length - 1];
    const fpr = fps.map((v) => (N > 0 ? v / N : NaN)), tpr = tps.map((v) => (P > 0 ? v / P : NaN));
    const auc = P > 0 && N > 0 ? M.auc(fpr, tpr) : NaN;
    const ties = c.thresholds.length < score.length;
    const trace = [
      st('sort', 'Sorted ' + score.length + ' scores from high to low; there are ' + P + ' positives and ' + N + ' negatives' + (ties ? ', and ' + (score.length - c.thresholds.length) + ' tied scores are grouped with their equal neighbours.' : '.'), { P, N }),
      st('sweep', 'Lowering the threshold past each distinct score gives ' + (fpr.length) + ' points; at threshold ' + f(thresholds[1]) + ' the curve is at (FPR ' + f(fpr[1]) + ', TPR ' + f(tpr[1]) + ').', { fpr, tpr, thresholds }),
      st('auc', P > 0 && N > 0 ? 'The trapezoid rule gives AUC = ' + f(auc) + ': the chance that a random positive scores above a random negative (ties count half).' : 'Only one class is present, so the ROC curve and its AUC are undefined.', { auc }),
    ];
    return { fpr, tpr, thresholds, tps, fps, P, N, auc, value: auc, posLabel: c.pos, trace };
  };
  M.rocAuc = (yTrue, score, opt) => { const r = M.roc(yTrue, score, opt); return { value: r.auc, trace: r.trace }; };

  /** one-against-rest AUC for several classes: P has one probability column per sorted label; macro or weighted mean */
  M.rocAucOvr = (yTrue, P, opt = {}) => {
    const labels = opt.labels || M.labels(yTrue);
    const per = labels.map((l, k) => ({ label: l, auc: M.roc(yTrue.map((y) => (y === l ? 1 : 0)), P.map((r) => r[k]), { posLabel: 1 }).auc, support: yTrue.filter((y) => y === l).length }));
    const n = yTrue.length;
    const value = opt.average === 'weighted' ? M.sum(per.map((p) => p.auc * p.support)) / n : M.mean(per.map((p) => p.auc));
    return { value, perClass: per, trace: [...per.map((p) => st('class', 'Class ' + p.label + ' against the rest: AUC = ' + f(p.auc) + '.', p)), st('mean', (opt.average === 'weighted' ? 'Support-weighted' : 'Plain') + ' mean of the ' + per.length + ' AUCs = ' + f(value) + '.')] };
  };

  /** the confusion counts when "positive" means score ≥ t: used by a threshold slider */
  M.atThreshold = (yTrue, score, t, posLabel) => {
    const pos = pickPos(yTrue, posLabel);
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < score.length; i++) { const p = score[i] >= t, y = yTrue[i] === pos; if (p && y) tp++; else if (p) fp++; else if (y) fn++; else tn++; }
    const precision = M.div(tp, tp + fp), recall = M.div(tp, tp + fn);
    return { t, tp, fp, fn, tn, precision, recall, fpr: M.div(fp, fp + tn), f1: M.div(2 * tp, 2 * tp + fp + fn),
      say: 'At threshold ' + f(t) + ': TP = ' + tp + ', FP = ' + fp + ', FN = ' + fn + ', TN = ' + tn + ', so precision = ' + f(precision) + ' and recall = ' + f(recall) + '.' };
  };

  /**
   * prCurve(yTrue, score, {posLabel}) → { precision, recall, thresholds, ap, trace }
   * Points run from the lowest threshold to the highest, then end at (recall 0, precision 1).
   */
  M.prCurve = (yTrue, score, opt = {}) => {
    const c = M.thresholdCounts(yTrue, score, opt.posLabel);
    const { tps, fps } = c, P = tps[tps.length - 1];
    const prec = tps.map((t, i) => M.div(t, t + fps[i])), rec = tps.map((t) => (P === 0 ? 1 : t / P));
    const precision = [...prec.reverse(), 1], recall = [...rec.reverse(), 0], thresholds = c.thresholds.slice().reverse();
    let ap = 0;
    for (let i = 0; i < recall.length - 1; i++) ap -= (recall[i + 1] - recall[i]) * precision[i];
    ap = Math.max(0, ap);
    const trace = [
      st('sweep', 'Each of the ' + thresholds.length + ' distinct scores is tried as a threshold; with ' + P + ' positives, the lowest threshold gives precision ' + f(precision[0]) + ' at recall ' + f(recall[0]) + '.', { precision, recall, thresholds }),
      st('ap', 'Average precision = Σ (Rₙ − Rₙ₋₁) Pₙ, the precision at each threshold weighted by the recall it adds = ' + f(ap) + (P === 0 ? ' (no positives, so recall is set to 1 everywhere)' : '') + '.', { ap }),
    ];
    return { precision, recall, thresholds, ap, value: ap, posLabel: c.pos, trace };
  };
  M.averagePrecision = (yTrue, score, opt) => { const r = M.prCurve(yTrue, score, opt); return { value: r.ap, trace: r.trace }; };

  /** logLoss(yTrue, P, {labels}): P is a list of positive-class probabilities or rows of class probabilities */
  M.logLoss = (yTrue, P, opt = {}) => {
    const labels = opt.labels || M.labels(yTrue), ix = M.index(labels), rows = M.probRows(P), eps = 2.220446049250313e-16;
    const per = rows.map((r, i) => -Math.log(Math.min(1 - eps, Math.max(eps, r[ix(yTrue[i])]))));
    const value = M.mean(per);
    let worst = 0; per.forEach((v, i) => { if (v > per[worst]) worst = i; });
    return { value, perExample: per, trace: [
      st('each', 'Each example pays −ln(probability given to its true class); example ' + worst + ' pays the most, ' + f(per[worst]) + ', for giving its true class ' + f(rows[worst][ix(yTrue[worst])]) + '.', { per }),
      st('mean', 'Log loss = mean of ' + per.length + ' penalties = ' + f(value) + '.', { value })] };
  };

  /** Brier score: mean squared error of probabilities; binary uses the positive-class probability */
  M.brier = (yTrue, P, opt = {}) => {
    let value, per;
    if (P.length && typeof P[0] === 'number') {
      const pos = pickPos(yTrue, opt.posLabel);
      per = P.map((p, i) => ((yTrue[i] === pos ? 1 : 0) - p) ** 2);
    } else {
      const labels = opt.labels || M.labels(yTrue), ix = M.index(labels);
      per = P.map((r, i) => M.sum(Array.from(r, (p, k) => ((ix(yTrue[i]) === k ? 1 : 0) - p) ** 2)));
      if (labels.length < 3) per = per.map((v) => v / 2);
    }
    value = M.mean(per);
    return { value, perExample: per, trace: [st('brier', 'Brier score = mean of (outcome − probability)² over ' + per.length + ' examples = ' + f(value) + '; 0 is perfect, 0.25 is a constant 0.5 guess.', { value })] };
  };

  /** the linear-interpolation percentile, rounded the same way as the common numeric libraries */
  const percentile = (sorted, q) => {
    const h = (sorted.length - 1) * q, lo = Math.floor(h), hi = Math.min(lo + 1, sorted.length - 1), g = h - lo, a = sorted[lo], d = sorted[hi] - a;
    return g >= 0.5 ? sorted[hi] - d * (1 - g) : a + d * g;
  };
  /**
   * calibration(yTrue, prob, {nBins: 5, strategy: 'uniform'|'quantile', posLabel})
   * → { probTrue, probPred, counts, edges, bins } with empty bins left out of probTrue and probPred
   */
  M.calibration = (yTrue, prob, opt = {}) => {
    const n = opt.nBins || 5, pos = pickPos(yTrue, opt.posLabel);
    const sorted = prob.slice().sort((a, b) => a - b);
    const edges = Array.from({ length: n + 1 }, (_, i) => (i === n ? (opt.strategy === 'quantile' ? sorted[sorted.length - 1] : 1) : opt.strategy === 'quantile' ? percentile(sorted, (i * (1 / n) * 100) / 100) : i * (1 / n)));
    const inner = edges.slice(1, -1), sums = new Array(n).fill(0), hits = new Array(n).fill(0), counts = new Array(n).fill(0), members = edges.slice(1).map(() => []);
    prob.forEach((p, i) => {
      let b = 0; while (b < inner.length && inner[b] < p) b++;
      sums[b] += p; hits[b] += yTrue[i] === pos ? 1 : 0; counts[b]++; members[b].push(i);
    });
    const bins = counts.map((c, b) => ({ lo: edges[b], hi: edges[b + 1], count: c, positives: hits[b], meanProb: M.div(sums[b], c, NaN), fraction: M.div(hits[b], c, NaN), members: members[b] }));
    const used = bins.filter((b) => b.count > 0);
    const gap = used.reduce((a, b) => a + b.count * Math.abs(b.fraction - b.meanProb), 0) / prob.length;
    return { probTrue: used.map((b) => b.fraction), probPred: used.map((b) => b.meanProb), counts: used.map((b) => b.count), edges, bins, ece: gap, trace: [
      st('bins', (opt.strategy === 'quantile' ? 'Quantile' : 'Equal-width') + ' bins with edges ' + edges.map((e) => f(e)).join(', ') + ' hold ' + counts.join(', ') + ' examples.', { edges, counts }),
      ...used.map((b) => st('bin', 'Bin ' + f(b.lo) + '–' + f(b.hi) + ': mean predicted ' + f(b.meanProb) + ', actually positive ' + b.positives + ' / ' + b.count + ' = ' + f(b.fraction) + '.', { lo: b.lo, hi: b.hi })),
      st('gap', 'The count-weighted average gap between predicted and actual (expected calibration error) is ' + f(gap) + '.', { ece: gap })] };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
