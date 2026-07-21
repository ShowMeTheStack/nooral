// Fairness, part 4: measuring a loan model's treatment of two groups with S.Metrics.fairness, and two mitigations.
// The favourable outcome is "approve" (predicted to repay). With useGroup: false the model never sees the group, yet
// it still treats the groups differently because income differs between them.
//   S.Rob.fairness({mitigation: 'none' | 'reweigh' | 'threshold', target: 'parity' | 'opportunity', useGroup: true, seed})
//      -> {accuracy, metrics (S.Metrics.fairness on the test set), thresholds, weights, trace}
//   reweighing: each (group, label) cell gets weight P(group)·P(label) / P(group, label), so that in the weighted
//      training data the label is independent of the group;
//   threshold: one cut-off per group on the predicted chance of repaying, chosen on the training set so that the
//      groups' approval rates (parity) or true-positive rates (opportunity) match the overall ones.
//   S.Rob.fairCompare({target, seed}) -> rows for the three settings side by side
(function (S) {
  'use strict';
  const R = S.Rob, fmt = (x) => R.fmt(x);

  R.reweigh = (y, grp) => {
    const n = y.length, cell = new Map(), pg = new Map(), py = new Map(), key = (g, v) => g + '|' + v;
    y.forEach((v, i) => { const g = grp[i]; cell.set(key(g, v), (cell.get(key(g, v)) || 0) + 1); pg.set(g, (pg.get(g) || 0) + 1); py.set(v, (py.get(v) || 0) + 1); });
    const table = [];
    for (const [k, c] of cell) { const [g, v] = k.split('|').map(Number); table.push({ group: g, label: v, count: c, weight: ((pg.get(g) / n) * (py.get(v) / n)) / (c / n) }); }
    table.sort((a, b) => a.group - b.group || a.label - b.label);
    const W = new Map(table.map((r) => [key(r.group, r.label), r.weight]));
    return { weights: y.map((v, i) => W.get(key(grp[i], v))), table };
  };

  /** the threshold for scores s (of the rows idx) that brings rate(t) closest to want; rate counts s ≥ t over base rows */
  const fit = (scores, want, rows) => {
    const cands = [...new Set(scores.map((s) => Math.round(s * 1000) / 1000))].sort((a, b) => a - b);
    let best = 0.5, bd = Infinity;
    for (const t of cands.concat([0.5])) { const rate = rows.filter((i) => scores[i] >= t).length / Math.max(1, rows.length), dd = Math.abs(rate - want); if (dd < bd - 1e-12) { bd = dd; best = t; } }
    return best;
  };

  R.fairness = (o = {}) => {
    const T = S.T, M = S.Metrics, data = o.data || R.loansData(o.seed || 1), mit = o.mitigation || 'none', target = o.target || 'parity';
    const ytr = data.ytr.map((v) => 1 - v), yte = data.yte.map((v) => 1 - v), gtr = data.gtr, gte = data.gte, trace = [];
    let weights = null, table = null;
    if (mit === 'reweigh') {
      ({ weights, table } = R.reweigh(ytr, gtr));
      trace.push(T.step('weights', 'Reweighing: ' + table.map((r) => 'group ' + 'AB'[r.group] + ' with label ' + (r.label ? 'repaid' : 'defaulted') + ' gets weight ' + fmt(r.weight)).join('; ') + '.', { table }));
    }
    // by default the model is shown the group too (as many real models effectively are, through proxies)
    const withG = (X, G) => (o.useGroup === false ? X : X.map((x, i) => x.concat([G[i] ? 1 : -1])));
    const Xtr = withG(data.Xtr, gtr), Xte = withG(data.Xte, gte);
    const model = R.logreg(Xtr, ytr, { weights, iters: 400 });
    const str = Xtr.map((x) => model.prob(x)), ste = Xte.map((x) => model.prob(x));
    const thresholds = [0.5, 0.5];
    if (mit === 'threshold') {
      const all = ytr.map((_, i) => i), overall = target === 'parity' ? all.filter((i) => str[i] >= 0.5).length / all.length : all.filter((i) => ytr[i] && str[i] >= 0.5).length / Math.max(1, ytr.filter((v) => v).length);
      for (const g of [0, 1]) {
        const rows = all.filter((i) => gtr[i] === g && (target === 'parity' || ytr[i] === 1));
        thresholds[g] = fit(str, overall, rows);
      }
      trace.push(T.step('thresholds', 'Per-group cut-offs chosen on the training set so that ' + (target === 'parity' ? 'approval rates' : 'true-positive rates') + ' match the overall ' + fmt(overall) + ': group A ' + fmt(thresholds[0]) + ', group B ' + fmt(thresholds[1]) + '.', { thresholds: thresholds.slice() }));
    }
    const pred = ste.map((s, i) => (s >= thresholds[gte[i]] ? 1 : 0)), accuracy = pred.filter((p, i) => p === yte[i]).length / yte.length;
    const metrics = M.fairness(yte, pred, gte.map((g) => 'AB'[g]), { posLabel: 1 });
    trace.push(...metrics.trace);
    trace.push(T.step('result', (mit === 'none' ? 'Without mitigation' : mit === 'reweigh' ? 'With reweighing' : 'With per-group thresholds') + ': accuracy ' + fmt(accuracy) + ', demographic parity difference ' + fmt(metrics.demographicParity) + ', equalised-odds difference ' + fmt(metrics.equalizedOdds) + '.', { accuracy }));
    return { mitigation: mit, target, accuracy, metrics, thresholds, weights: table, trace };
  };

  R.fairCompare = (o = {}) => {
    const data = R.loansData(o.seed || 1);
    return ['none', 'reweigh', 'threshold'].map((m) => { const r = R.fairness(Object.assign({}, o, { mitigation: m, data })); return { mitigation: m, accuracy: r.accuracy, parity: r.metrics.demographicParity, odds: r.metrics.equalizedOdds, groups: r.metrics.groups.map((g) => ({ group: g.group, selectionRate: g.selectionRate, tpr: g.tpr, fpr: g.fpr })), result: r }; });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
