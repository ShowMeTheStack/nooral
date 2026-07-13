// Fairness metrics across the groups of a sensitive attribute: selection rates and the demographic parity
// difference, true- and false-positive rates and the equalised-odds differences, and a confusion table per group.
(function (S) {
  'use strict';
  const M = (S.Metrics = S.Metrics || {});
  const f = (x) => M.f(x), st = (k, s, d) => M.step(k, s, d);

  /**
   * fairness(yTrue, yPred, group, {posLabel: 1})
   * → { groups, demographicParity, tprDifference, fprDifference, equalizedOdds, trace }
   * each group: { group, n, selected, selectionRate, tpr, fpr, confusion }
   */
  M.fairness = (yTrue, yPred, group, opt = {}) => {
    M.check(yTrue, yPred, 'fairness'); M.check(yTrue, group, 'fairness');
    const pos = opt.posLabel === undefined ? 1 : opt.posLabel;
    const gs = M.labels(group);
    const groups = gs.map((g) => {
      const idx = []; group.forEach((v, i) => { if (v === g) idx.push(i); });
      let tp = 0, fp = 0, fn = 0, tn = 0;
      for (const i of idx) { const p = yPred[i] === pos, y = yTrue[i] === pos; if (p && y) tp++; else if (p) fp++; else if (y) fn++; else tn++; }
      const yt = idx.map((i) => yTrue[i]), yp = idx.map((i) => yPred[i]);
      return { group: g, n: idx.length, indices: idx, tp, fp, fn, tn, selected: tp + fp, selectionRate: M.div(tp + fp, idx.length), tpr: M.div(tp, tp + fn), fpr: M.div(fp, fp + tn),
        confusion: M.confusion(yt, yp, { labels: M.labels(yTrue, yPred) }) };
    });
    const range = (k) => { const v = groups.map((g) => g[k]); return Math.max(...v) - Math.min(...v); };
    const dp = range('selectionRate'), dt = range('tpr'), df = range('fpr'), eo = Math.max(dt, df);
    const trace = [
      ...groups.map((g) => st('group', 'Group ' + g.group + ': ' + g.selected + ' of ' + g.n + ' predicted positive (rate ' + f(g.selectionRate) + '), TPR = ' + g.tp + ' / ' + (g.tp + g.fn) + ' = ' + f(g.tpr) + ', FPR = ' + g.fp + ' / ' + (g.fp + g.tn) + ' = ' + f(g.fpr) + '.', { group: g.group })),
      st('parity', 'Demographic parity difference = largest − smallest selection rate = ' + f(dp) + '.', { value: dp }),
      st('odds', 'Equalised odds: the TPR gap is ' + f(dt) + ' and the FPR gap is ' + f(df) + '; the larger, ' + f(eo) + ', is the equalised-odds difference.', { tpr: dt, fpr: df }),
    ];
    return { groups, demographicParity: dp, tprDifference: dt, fprDifference: df, equalizedOdds: eo, value: dp, trace };
  };
  M.demographicParity = (t, p, g, o) => { const r = M.fairness(t, p, g, o); return { value: r.demographicParity, trace: r.trace }; };
  M.equalizedOdds = (t, p, g, o) => { const r = M.fairness(t, p, g, o); return { value: r.equalizedOdds, tpr: r.tprDifference, fpr: r.fprDifference, trace: r.trace }; };
})(globalThis.Nooral = globalThis.Nooral || {});
