// Part 17, fairness on the small-loans table: a logistic model approves applicants it expects to repay. Its
// treatment of groups A and B is measured (a confusion table per group, approval rates, true- and false-positive
// rates, the demographic parity and equalised-odds differences), then narrowed by reweighing the training data or
// by a cut-off for each group.
(function (S) {
  'use strict';
  const P = S.P17, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const GN = ['A', 'B'];
  /** bars side by side for groups A and B, one pair per measure */
  P.gbars = (ax, key, pairs) => {
    const it = [];
    pairs.forEach((pr, k) => pr.forEach((v, g) => { if (v === null || v === undefined) return; const x0 = ax.X(k + 0.5) - 26 + g * 27, top = ax.Y(v); it.push({ key: key + k + '_' + g, kind: 'cell', x: P.r2(x0), y: top, w: 25, h: P.r2(Math.max(0.5, ax.Y(0) - top)), rx: 2, text: '', cls: 'bar g' + GN[g], audit: 'ignore' }, H().text(key + 'v' + k + '_' + g, x0 + 12.5, top - 5, H().fx(v, 2).replace(/^0\./, '.'), 'tick', 11.5, 'middle')); }));
    return it;
  };
  const MEAS = ['repaid', 'approved', 'TPR', 'FPR'];
  const barAxes = (prefix, x, y, from) => C().axes({ prefix, x, y, w: from ? 260 : 330, h: 170, xr: [from || 0, 4], yr: [0, 1.1], xticks: 0, yticks: [0, 0.25, 0.5, 0.75, 1], tickSize: 12, yname: 'rate' });
  const names = (ax, key) => MEAS.map((m, k) => H().text(key + k, ax.X(k + 0.5), ax.Y(0) + 20, m, 'axname', 12.5, 'middle'));
  const gkeys = (x, y) => [...P.key('fm:ka', x, y, 'gA', 'group A'), ...P.key('fm:kb', x + 100, y, 'gB', 'group B')];
  /** a group's confusion table: rows truly repaid / defaulted, columns approved / denied */
  const conf = (prefix, x, y, g, cls) => S.Kit.heat.grid({ prefix, x, y, cw: 74, ch: 40, vals: [[g.tp, g.fn], [g.fp, g.tn]], scheme: cls === 'gA' ? 'c0' : 'c1', max: Math.max(g.tp, g.fn, g.fp, g.tn), fmt: (v) => String(v), rows: ['repaid', 'defaulted'], cols: ['approved', 'denied'], size: 14, labelSize: 12.5, title: 'group ' + g.group + ' (' + g.n + ' applicants)', titleCls: cls, titleSize: 13 }).items;
  P.loans = (seed) => P.once('ld:' + seed, () => S.Rob.loansData(seed));
  P.fair = (o) => P.once('fr:' + JSON.stringify(o), () => S.Rob.fairness(Object.assign({ data: P.loans(o.seed) }, o)));

  S.topic({
    id: 'fair-metrics', family: 'p17-fair', order: 10, title: 'Measuring fairness between groups',
    blurb: 'A lender’s model approves applicants it expects to repay. Is it fair to groups A and B? There are several honest answers. Demographic parity compares how often each group is approved; equalised odds compares the true-positive rate (repayers approved) and the false-positive rate (defaulters approved). The numbers come from one confusion table per group, and leaving the group out of the model does not by itself close the gaps.',
    inputs: [
      { id: 'useGroup', kind: 'choice', label: 'The model sees the group', def: 'yes', options: [{ value: 'yes', label: 'yes' }, { value: 'no', label: 'no: group column removed' }] },
      { id: 'seed', kind: 'number', label: 'Seed (the applicants)', def: 1, min: 1, max: 999 },
    ],
    presets: [{ label: 'Group visible', values: { useGroup: 'yes', seed: 1 } }, { label: 'Group removed', values: { useGroup: 'no', seed: 1 } }, { label: 'Other applicants', values: { useGroup: 'yes', seed: 4 } }],
    counters: [{ id: 'groups', label: 'groups measured' }],
    run(inp) {
      const F = H().frames(['groups']), q = F.q, L = P.loans(inp.seed), r = P.fair({ mitigation: 'none', useGroup: inp.useGroup === 'yes', seed: inp.seed }), M = r.metrics;
      const gs = [0, 1].map((g) => M.groups.find((x) => x.group === GN[g]));
      const yte = L.yte.map((v) => 1 - v), base = [0, 1].map((g) => { const idx = L.gte.map((x, i) => (x === g ? i : -1)).filter((i) => i >= 0); return idx.filter((i) => yte[i] === 1).length / idx.length; });
      const ax = barAxes('fm:ax', 420, 90);
      const pairs = (upto) => [base, gs.map((g) => g.selectionRate), gs.map((g) => g.tpr), gs.map((g) => g.fpr)].slice(0, upto);
      const right = (upto, extra) => [...ax.items, ...names(ax, 'fm:n'), ...P.gbars(ax, 'fm:b', pairs(upto)), ...gkeys(420, 320), ...(extra || [])];
      const left = (k) => [...(k >= 1 ? conf('fm:ca', 110, 90, gs[0], 'gA') : []), ...(k >= 2 ? conf('fm:cb', 110, 250, gs[1], 'gB') : [])];
      F.push([...right(1), H().head('fm:h', 20, 34, 'Who repaid, in each group')],
        `The test set holds ${P.qpl(q, gs[0].n, 'applicant')} of group A and ${q.n(gs[1].n)} of group B. In truth ${q.pct(base[0])} of group A repaid and ${q.pct(base[1])} of group B: group B has lower incomes on average. The model ${inp.useGroup === 'yes' ? 'sees the group' : 'does not see the group'} and approves when it expects repayment; it is ${q.pct(r.accuracy)} accurate overall.`, { phase: 'base', nA: gs[0].n, nB: gs[1].n, base, acc: r.accuracy });
      gs.forEach((g, k) => {
        F.set('groups', k + 1);
        F.push([...left(k + 1), ...right(4).filter((it) => !/fm:b\d_1|fm:bv\d_1/.test(it.key) || k === 1), H().head('fm:h', 20, 34, 'Group ' + g.group)],
          `Group ${g.group}: of ${q.n(g.tp + g.fn)} who repaid, ${q.n(g.tp)} were approved, a true-positive rate of ${q(g.tpr, 2)}; of ${q.n(g.fp + g.tn)} who defaulted, ${q.n(g.fp)} were approved anyway, a false-positive rate of ${q(g.fpr, 2)}. In all ${q.n(g.selected)} of ${q.n(g.n)} were approved: a rate of ${q(g.selectionRate, 2)}.`, { phase: 'group', g: k, pos: g.tp + g.fn, neg: g.fp + g.tn, tp: g.tp, fn: g.fn, fp: g.fp, tn: g.tn, tpr: g.tpr, fpr: g.fpr, sel: g.selectionRate, selected: g.selected, n: g.n });
      });
      F.push([...left(2), ...right(4), H().head('fm:h', 20, 34, 'Demographic parity')],
        `Demographic parity asks for equal approval rates: ${q(gs[0].selectionRate, 2)} for group A against ${q(gs[1].selectionRate, 2)} for group B, a difference of ${q(M.demographicParity, 3)}. It ignores who would really repay.`, { phase: 'parity', selA: gs[0].selectionRate, selB: gs[1].selectionRate, dp: M.demographicParity });
      F.push([...left(2), ...right(4), H().head('fm:h', 20, 34, 'Equalised odds')],
        `Equalised odds compares like with like: repayers should be approved equally often (the TPR gap is ${q(M.tprDifference, 3)}) and so should defaulters (the FPR gap is ${q(M.fprDifference, 3)}). The larger gap, ${q(M.equalizedOdds, 3)}, is the equalised-odds difference.`, { phase: 'odds', tprGap: M.tprDifference, fprGap: M.fprDifference, eo: M.equalizedOdds });
      const o = P.fair({ mitigation: 'none', useGroup: inp.useGroup !== 'yes', seed: inp.seed }).metrics;
      F.push([...left(2), ...right(4), H().head('fm:h', 20, 34, 'Hiding the group is not enough')],
        `${inp.useGroup === 'yes' ? 'Remove the group column and retrain' : 'Give the model the group column and retrain'}: the parity difference becomes ${q(o.demographicParity, 3)} and the equalised-odds difference ${q(o.equalizedOdds, 3)}, against ${q(M.demographicParity, 3)} and ${q(M.equalizedOdds, 3)} here. Income differs between the groups, so it carries the group’s signal anyway. And when the true repayment rates differ, no model can satisfy every fairness measure at once: choosing which one matters is a decision, not a calculation.`, { phase: 'hide', dpO: o.demographicParity, eoO: o.equalizedOdds, dp: M.demographicParity, eo: M.equalizedOdds });
      for (const f of F.list) f.narrowView = { x: 380, y: 0, w: 400, h: 330 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Group fairness measures', text: ['approval (selection) rate = approved / n', 'TPR = approved repayers / repayers', 'FPR = approved defaulters / defaulters', 'parity difference = |rate_A − rate_B|', 'equalised-odds difference =', '  max(|TPR_A − TPR_B|, |FPR_A − FPR_B|)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'odds', pick: 'option', once: true,
      ask: 'Without the group column, the parity difference would…',
      options: ['shrink a lot', 'stay about the same', 'grow'],
      answer: (fr, i, inp) => { const m = fr.find((f) => f.meta.phase === 'hide').meta, d = inp.useGroup === 'yes' ? m.dpO - m.dp : m.dp - m.dpO; return d < -0.03 ? 'shrink a lot' : d > 0.03 ? 'grow' : 'stay about the same'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'hide').say,
    }],
    quiz: [
      { q: 'Group A: 80 of 100 approved. Group B: 50 of 100. The demographic parity difference is…', options: ['0.30', '0.80', '0.50', '1.30'], answer: 0, why: '0.80 − 0.50.' },
      { q: 'The true-positive rate of a group is…', options: ['the share of its repayers who were approved', 'the share of the group approved', 'the share of defaulters denied', 'its accuracy'], answer: 0, why: 'TP / (TP + FN).' },
      { q: 'Why can removing the group column leave the gap unchanged?', options: ['other features, such as income, carry the group’s signal', 'the model memorises names', 'the metrics are wrong', 'it always closes the gap'], answer: 0, why: 'Proxies let the model treat the groups differently anyway.' },
      { q: 'When the groups’ true repayment rates differ, an accurate model…', options: ['cannot in general meet parity and equalised odds at once', 'meets every measure', 'is always unfair', 'cannot be trained'], answer: 0, why: 'Different base rates force a trade-off between the measures.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const MIT = { none: 'no mitigation', reweigh: 'reweighing', threshold: 'a cut-off per group' };
  S.topic({
    id: 'fair-mitigate', family: 'p17-fair', order: 20, title: 'Narrowing the gaps: reweighing and per-group cut-offs',
    blurb: 'Two common mitigations. Reweighing works before training: each (group, outcome) cell of the training data gets a weight that makes outcome and group independent, so the model no longer learns the link between them. Per-group cut-offs work after training: the approval threshold is chosen separately for each group so that approval rates (parity) or repayers’ approval rates (equal opportunity) match. Both close gaps; neither is free.',
    inputs: [
      { id: 'target', kind: 'choice', label: 'Cut-offs aim for', def: 'parity', options: [{ value: 'parity', label: 'equal approval rates (parity)' }, { value: 'opportunity', label: 'equal true-positive rates (opportunity)' }] },
      { id: 'seed', kind: 'number', label: 'Seed (the applicants)', def: 1, min: 1, max: 999 },
    ],
    presets: [{ label: 'Aim for parity', values: { target: 'parity', seed: 1 } }, { label: 'Aim for equal opportunity', values: { target: 'opportunity', seed: 1 } }, { label: 'Other applicants', values: { target: 'parity', seed: 3 } }],
    counters: [{ id: 'models', label: 'models compared' }],
    run(inp) {
      const F = H().frames(['models']), q = F.q, rows = P.once('fc:' + inp.target + ':' + inp.seed, () => S.Rob.fairCompare({ target: inp.target, seed: inp.seed }));
      const ax = barAxes('fg:ax', 30, 70, 1);
      const bars = (row) => [...ax.items, ...names(ax, 'fg:n').slice(1), ...P.gbars(ax, 'fg:b', [[null, null], row.groups.map((g) => g.selectionRate), row.groups.map((g) => g.tpr), row.groups.map((g) => g.fpr)]), ...gkeys(30, 300)];
      const gap = (row, x, y) => H().lines('fg:g', x, y, [['accuracy ' + H().pct(row.accuracy), 'legend'], ['parity difference ' + H().fx(row.parity, 3), 'mono'], ['equalised-odds difference ' + H().fx(row.odds, 3), 'mono']], 13, 22);
      const [none, rw, th] = rows;
      F.set('models', 1);
      F.push([...bars(none), ...gap(none, 430, 90), H().head('fg:h', 30, 30, 'No mitigation')],
        `Without mitigation the model approves ${q(none.groups[0].selectionRate, 2)} of group A and ${q(none.groups[1].selectionRate, 2)} of group B: a parity difference of ${q(none.parity, 3)} and an equalised-odds difference of ${q(none.odds, 3)}, at ${q.pct(none.accuracy)} accuracy.`, { phase: 'none', selA: none.groups[0].selectionRate, selB: none.groups[1].selectionRate, dp: none.parity, eo: none.odds, acc: none.accuracy });
      const tab = rw.result.weights;
      const t = S.Kit.dt.table({ prefix: 'fg:t', x: 430, y: 80, cols: [{ name: 'group', w: 70 }, { name: 'outcome', w: 100 }, { name: 'rows', w: 60 }, { name: 'weight', w: 80 }], rows: tab.map((w) => [GN[w.group], w.label ? 'repaid' : 'defaulted', String(w.count), H().fx(w.weight, 3)]), rowH: 28, size: 13, cls: (i, j) => (j === 3 ? (tab[i].weight > 1 ? 'lit' : '') : '') });
      const up = tab.filter((w) => w.weight > 1);
      F.push([...bars(none), ...t.items, H().head('fg:h', 30, 30, 'Reweighing the training rows'), H().text('fg:tt', 430, 64, 'P(group) · P(outcome) / P(group, outcome)', 'math', 12.5)],
        `Reweighing gives each training row a weight P(group) × P(outcome) / P(group, outcome). Cells rarer than independence would predict are weighted up: ${up.map((w) => 'group ' + GN[w.group] + (w.label ? ' repaid' : ' defaulted') + ' to ' + q(w.weight, 3)).join(' and ')}. In the weighted data, repaying no longer goes with group.`, { phase: 'weights', weights: tab.map((w) => w.weight), counts: tab.map((w) => w.count) });
      F.set('models', 2);
      F.push([...bars(rw), ...gap(rw, 430, 90), H().head('fg:h', 30, 30, 'With reweighing')],
        `Trained on the weighted rows, the model approves ${q(rw.groups[0].selectionRate, 2)} of group A and ${q(rw.groups[1].selectionRate, 2)} of group B: parity difference ${q(rw.parity, 3)} (was ${q(none.parity, 3)}), equalised-odds difference ${q(rw.odds, 3)}, accuracy ${q.pct(rw.accuracy)}.`, { phase: 'reweigh', selA: rw.groups[0].selectionRate, selB: rw.groups[1].selectionRate, dp: rw.parity, dp0: none.parity, eo: rw.odds, acc: rw.accuracy });
      const thr = th.result.thresholds, nl = C().axes({ prefix: 'fg:nl', x: 430, y: 250, w: 290, h: 30, xr: [0, 1], yr: [0, 1], xticks: 0.25, yticks: 0, grid: false, xname: 'chance of repaying', tickSize: 12 });
      const marks = [0, 1].map((g) => ({ key: 'fg:m' + g, kind: 'dot', x: nl.X(thr[g]), y: nl.Y(0.25 + 0.5 * g), r: 6.5, cls: 'swatch g' + GN[g], point: true }));
      F.set('models', 3);
      F.push([...bars(th), ...gap(th, 430, 90), ...nl.items, C().vline(nl, 'fg:half', 0.5, 'p17thr'), ...marks, H().text('fg:nlt', 430, 236, 'cut-offs: A (blue), B (orange); dashed 0.5', 'axname', 12.5), H().head('fg:h', 30, 30, 'A cut-off per group')],
        `Instead, keep the first model and approve group A above a chance of ${q(thr[0], 3)} and group B above ${q(thr[1], 3)}, chosen on the training set so that ${inp.target === 'parity' ? 'approval rates' : 'repayers’ approval rates'} match. On the test set: parity difference ${q(th.parity, 3)}, equalised-odds difference ${q(th.odds, 3)}, accuracy ${q.pct(th.accuracy)}.`, { phase: 'threshold', thr, dp: th.parity, eo: th.odds, acc: th.accuracy });
      const tt = S.Kit.dt.table({ prefix: 'fg:s', x: 60, y: 80, cols: [{ name: 'setting', w: 190 }, { name: 'accuracy', w: 100 }, { name: 'parity diff.', w: 110 }, { name: 'odds diff.', w: 110 }], rows: rows.map((r) => [MIT[r.mitigation], H().pct(r.accuracy), H().fx(r.parity, 3), H().fx(r.odds, 3)]), rowH: 32, size: 13 });
      const bestDp = rows.reduce((b, r) => (r.parity < b.parity ? r : b)), bestEo = rows.reduce((b, r) => (r.odds < b.odds ? r : b));
      F.push([...tt.items, H().head('fg:h', 60, 50, 'Side by side')],
        `Smallest parity difference: ${MIT[bestDp.mitigation]} (${q(bestDp.parity, 3)}); smallest equalised-odds difference: ${MIT[bestEo.mitigation]} (${q(bestEo.odds, 3)}). Accuracy runs from ${q.pct(Math.min(...rows.map((r) => r.accuracy)))} to ${q.pct(Math.max(...rows.map((r) => r.accuracy)))}. Closing one gap can widen another; which gap to close is a choice to make openly, with the people affected.`, { phase: 'compare', dps: rows.map((r) => r.parity), eos: rows.map((r) => r.odds), accs: rows.map((r) => r.accuracy), bestDp: bestDp.parity, bestEo: bestEo.odds });
      for (const f of F.list) f.narrowView = { x: -10, y: 0, w: 410, h: 330 };
      F.last().narrowView = { x: 40, y: 20, w: 560, h: 200 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Two mitigations', text: ['reweighing: w(g, y) = P(g) P(y) / P(g, y)', '  (before training; the weighted data has', '   outcome independent of group)', 'per-group cut-off t_g: approve if p ≥ t_g', '  chosen so rates (or TPRs) match overall'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'none', pick: 'option', once: true,
      ask: 'Which group’s rows will reweighing weigh up for “repaid”?',
      options: ['group A', 'group B'],
      answer: (fr) => { const t = fr.find((f) => f.meta.phase === 'weights').meta; return t.weights[1] > 1 ? 'group A' : t.weights[3] > 1 ? 'group B' : null; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'weights').say,
    }],
    quiz: [
      { q: 'P(group B) = 0.4, P(repaid) = 0.75, P(B and repaid) = 0.25. The reweighing weight of those rows is…', options: ['1.2', '0.83', '0.3', '1'], answer: 0, why: '0.4 × 0.75 / 0.25 = 1.2.' },
      { q: 'Per-group cut-offs are applied…', options: ['after training, to the model’s scores', 'before training, to the rows', 'to the labels', 'to the features'], answer: 0, why: 'The model is unchanged; only its decision rule differs by group.' },
      { q: 'Equal opportunity asks for equal…', options: ['true-positive rates', 'approval rates', 'accuracies', 'group sizes'], answer: 0, why: 'Those who would repay should be approved equally often.' },
      { q: 'A mitigation shrinks the parity gap but widens the false-positive gap. This shows…', options: ['fairness measures can pull against each other', 'the code is wrong', 'the model is perfect', 'parity implies equal odds'], answer: 0, why: 'Different measures encode different ideas of fair.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
