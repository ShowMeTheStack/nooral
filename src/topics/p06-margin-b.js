// Part 6, the soft margin: when the classes overlap no street is empty, so each point may pay to sit inside it or
// on the wrong side. The slack ξ of a point is how far short of its margin it falls (in units of f), C is the price
// of a unit of slack, and the hinge loss max(0, 1 − y·f) is that slack seen as a loss.
(function (S) {
  'use strict';
  const P = S.P06, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const SWEEP = [0.01, 0.1, 1, 10, 100];
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const cFmt = (c) => H().sg(c, 2);

  /** a dashed segment from each point with slack to its own margin, along the normal of the line */
  const slackLines = (ax, key, X, y, fit) => {
    const out = [], w = fit.w, nn = fit.norm * fit.norm;
    X.forEach((x, i) => {
      if (fit.slack[i] <= 1e-9) return;
      const yi = y[i] ? 1 : -1, t = (yi - fit.F[i]) / nn, foot = [x[0] + t * w[0], x[1] + t * w[1]];
      // a wide street puts the foot of a slack outside the plot: the dashed line stops at the plot's edge
      const p0 = [ax.X(x[0]), ax.Y(x[1])], p1 = [ax.X(foot[0]), ax.Y(foot[1])], b = ax.box;
      let t1 = 1;
      for (const [d, lo, hi, v] of [[p1[0] - p0[0], b.x, b.x + b.w, p0[0]], [p1[1] - p0[1], b.y, b.y + b.h, p0[1]]]) {
        if (d > 0 && v + d * t1 > hi) t1 = (hi - v) / d;
        if (d < 0 && v + d * t1 < lo) t1 = (lo - v) / d;
      }
      const e = [+(p0[0] + t1 * (p1[0] - p0[0])).toFixed(2), +(p0[1] + t1 * (p1[1] - p0[1])).toFixed(2)];
      out.push({ key: key + ':s' + i, kind: 'path', d: `M${p0[0]},${p0[1]} L${e[0]},${e[1]}`, cls: 'trail c' + y[i], pathAudit: 'edge-x', layer: 'lines' });
    });
    return out;
  };
  // a margin support vector has y·f = 1 only up to the solver's tolerance, so it counts as on the edge, not inside
  const counts = (fit) => ({ inside: fit.yf.filter((v) => v < 1 - 1e-3 && v >= 0).length, wrong: fit.yf.filter((v) => v < 0).length, xi: sum(fit.slack) });

  S.topic({
    id: 'svm-soft-c', family: 'p06-margin', order: 20, title: 'The soft margin and the price C',
    blurb: 'Real classes overlap, so no empty street exists. The soft margin lets point i fall short of its margin by a slack ξᵢ ≥ 0, and charges C for every unit: minimise ½‖w‖² + C·Σξᵢ. A small C buys a wide street with many points inside it; a large C pays almost anything to keep points out, and the street narrows round the few that are left.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'overlap', options: P.setOptions(['overlap', 'line', 'iris']) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
      { id: 'C', kind: 'slider', label: 'Your C', def: 3, min: 0.01, max: 100, log: true, fmt: (v) => H().sg(v, 2) },
    ],
    presets: [
      { label: 'Overlapping groups', values: { set: 'overlap', seed: 1, C: 3 } },
      { label: 'A noisy line', values: { set: 'line', seed: 2, C: 0.3 } },
      { label: 'Iris petals (real data)', values: { set: 'iris', seed: 1, C: 30 } },
      { label: 'Overlap, another draw', values: { set: 'overlap', seed: 5, C: 0.05 } },
    ],
    counters: [{ id: 'sv', label: 'support vectors' }, { id: 'inside', label: 'inside the street' }, { id: 'wrong', label: 'on the wrong side' }],
    run(inp) {
      const d = P.data(inp.set, { seed: inp.seed }), X = d.X, y = d.y, n = X.length;
      const F = H().frames(['sv', 'inside', 'wrong']), q = F.q;
      const Cu = +H().sg(inp.C, 2), list = SWEEP.concat(SWEEP.includes(Cu) ? [] : [Cu]);
      const fits = list.map((c) => P.fit(X, y, { C: c }));
      const box = { x: 40, y: 24, w: 400, h: 400 }, ax = P.axes('sc', X, box, { pad: 0.3, xname: d.names ? 'petal length' : 'x₁', yname: d.names ? 'petal width' : 'x₂' });
      // two small plots against log C: the width of the street and the total slack
      const lc = (v) => Math.log10(v), RX = 520;
      const wmax = Math.max(...fits.map((f) => f.width)), xmax = Math.max(...fits.map((f) => sum(f.slack)));
      const ticksC = [-2, -1, 0, 1, 2], fmtC = (v) => ({ '-2': '0.01', '-1': '0.1', 0: '1', 1: '10', 2: '100' }[v] || '');
      const ax1 = C().axes({ prefix: 'w1', x: RX + 20, y: 44, w: 250, h: 110, xr: [-2.2, 2.2], yr: [0, wmax * 1.1], xticks: ticksC, yticks: C().ticks(0, wmax, 3), fmtX: fmtC, yname: 'street width 2 / ‖w‖' });
      const ax2 = C().axes({ prefix: 'w2', x: RX + 20, y: 220, w: 250, h: 110, xr: [-2.2, 2.2], yr: [0, xmax * 1.1], xticks: ticksC, yticks: C().ticks(0, xmax, 3), fmtX: fmtC, yname: 'total slack Σξ', xname: 'C (log scale)' });
      const side = (upto, cur) => {
        const pts = SWEEP.map((c, k) => [lc(c), fits[k]]).slice(0, upto);
        const items = [...ax1.items, ...ax2.items];
        if (pts.length > 1) items.push(C().line(ax1, 'w1:l', pts.map(([u, f]) => [u, f.width]), 'act'), C().line(ax2, 'w2:l', pts.map(([u, f]) => [u, sum(f.slack)]), 'c1'));
        pts.forEach(([u, f], k) => items.push(C().dot(ax1, 'w1:d' + k, u, f.width, '', 3.5), C().dot(ax2, 'w2:d' + k, u, sum(f.slack), 'c1', 3.5)));
        if (cur) items.push(C().dot(ax1, 'w1:cur', lc(cur.C), cur.width, '', 6.5), C().dot(ax2, 'w2:cur', lc(cur.C), sum(cur.slack), 'c1', 6.5));
        return items;
      };
      const plot = (fit) => [...ax.items, ...P.boundary(ax, 'sc', fit), ...slackLines(ax, 'sc', X, y, fit), ...P.points(ax, 'sc', X, y, fit, { r: 4.6 })];
      const readout = (fit) => H().lines('rd', RX, 392, [['C = ' + cFmt(fit.C) + ':  ' + fit.sv.length + ' support vectors', 'strong'], ['½‖w‖² + C·Σξ = ' + H().fx(0.5 * fit.norm * fit.norm + fit.C * sum(fit.slack), 3), 'legend']], 13, 22);

      F.push([...ax.items, ...P.points(ax, 'sc', X, y, null, { r: 4.6 }), ...side(0)],
        `${q.n(n)} points whose classes overlap: no straight line puts every blue point on one side and every orange on the other, so the hard margin has no answer. The soft margin lets points into the street, at a price C for each unit of slack.`, { phase: 'data', n });
      SWEEP.forEach((c, k) => {
        const fit = fits[k], ct = counts(fit);
        F.set('sv', fit.sv.length).set('inside', ct.inside).set('wrong', ct.wrong);
        let say = `C = ${q.sg(c, 2)}: the street is ${q(fit.width, 3)} wide, ${q.n(ct.inside)} points sit inside it on their own side and ${q.n(ct.wrong)} on the wrong side; their slacks, the dashed lines, add up to Σξ = ${q(ct.xi)}. `;
        if (k === 0) say += `Slack is cheap, so the street spreads wide and every point inside it or on its edge is a support vector: ${q.n(fit.sv.length)} of them, most with α at the cap C.`;
        else if (k === SWEEP.length - 1) say += `Slack is dear now: the street has narrowed round the overlap and only ${q.n(fit.sv.length)} support vectors remain. Training accuracy ${q.pct(fit.acc)}.`;
        else say += `${q.n(fit.sv.length)} support vectors: ${q.n(fit.free.length)} exactly on a margin (0 < α < C) and ${q.n(fit.bound.length)} at the cap α = C, which is every point with slack. Training accuracy ${q.pct(fit.acc)}.`;
        F.push([...plot(fit), ...side(k + 1, fit), ...readout(fit)], say, { phase: 'sweep', C: c, width: fit.width, xi: ct.xi, inside: ct.inside, wrong: ct.wrong, nsv: fit.sv.length, free: fit.free.length, bound: fit.bound.length, acc: fit.acc, w: fit.w, b: fit.b, k });
      });
      const fu = fits[list.indexOf(Cu)], cu = counts(fu);
      F.set('sv', fu.sv.length).set('inside', cu.inside).set('wrong', cu.wrong);
      F.push([...plot(fu), ...side(SWEEP.length, fu), ...readout(fu)],
        `Your C = ${q.sg(Cu, 2)}: width ${q(fu.width, 3)}, Σξ = ${q(cu.xi)}, ${q.n(fu.sv.length)} support vectors and training accuracy ${q.pct(fu.acc)}. Raising C trades a narrower street for less slack; the best C for new data is found by validation, not by training accuracy.`, { phase: 'mine', C: Cu, width: fu.width, xi: cu.xi, nsv: fu.sv.length, acc: fu.acc, w: fu.w, b: fu.b });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 450, h: 470 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The soft margin', text: ['minimise ½‖w‖² + C·Σ ξᵢ', 'subject to yᵢ f(xᵢ) ≥ 1 − ξᵢ and ξᵢ ≥ 0', 'ξᵢ = max(0, 1 − yᵢ f(xᵢ)): the hinge loss', '0 < ξᵢ < 1: inside the street, right side', 'ξᵢ > 1: on the wrong side', 'in the dual: 0 ≤ αᵢ ≤ C'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'sweep' && f.meta.k === 1 && fr[i + 1] && fr[i + 1].meta.phase === 'sweep', pick: 'option', ask: (fr, i) => 'C grows tenfold, to ' + H().sg(fr[i + 1].meta.C, 2) + '. What happens to the street?', options: ['it gets narrower', 'it gets wider', 'it stays the same'], answer: (fr, i) => (fr[i + 1].meta.width < fr[i].meta.width ? 'it gets narrower' : fr[i + 1].meta.width > fr[i].meta.width ? 'it gets wider' : 'it stays the same'), explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => f.meta.phase === 'sweep' && f.meta.k === 3 && fr[i + 1] && fr[i + 1].meta.phase === 'sweep', pick: 'option', ask: 'At the next, larger C, will there be more support vectors or fewer?', options: ['fewer', 'more', 'the same number'], answer: (fr, i) => (fr[i + 1].meta.nsv < fr[i].meta.nsv ? 'fewer' : fr[i + 1].meta.nsv > fr[i].meta.nsv ? 'more' : 'the same number'), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'A point sits on the right side of the line but inside the street. Its slack ξ is…', options: ['between 0 and 1', 'exactly 0', 'greater than 1', 'negative'], answer: 0, why: 'ξ = 1 − y·f and 0 < y·f < 1 inside the street on the correct side.' },
      { q: 'What does a very large C approach?', options: ['the hard margin: slack is almost forbidden', 'a street of infinite width', 'a model with no support vectors', 'random guessing'], answer: 0, why: 'With slack priced very high, the optimiser keeps points out of the street whenever it can.' },
      { q: 'In the dual, every point with ξ > 0 has…', options: ['α = C', 'α = 0', '0 < α < C', 'α > C'], answer: 0, why: 'A point that falls short of its margin pushes its multiplier up to the cap.' },
      { q: 'Why not just pick the C with the best training accuracy?', options: ['it favours large C, which can overfit; choose C on validation data', 'training accuracy cannot be measured for an SVM', 'C does not affect accuracy', 'a small C always wins'], answer: 0, why: 'Training accuracy keeps rising as the street narrows round noisy points; held-out data tells whether that helps.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // The hinge loss
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'svm-hinge', family: 'p06-margin', order: 30, title: 'The hinge loss',
    blurb: 'Write m = y·f(x) for the signed margin of a point (y = ±1): it is positive when the point is on its own side and at least 1 when it is outside the street. The hinge loss max(0, 1 − m) is zero for points safely outside, grows straight for points in the street or beyond, and never cares how far a point is on the safe side. The SVM minimises ½‖w‖² + C·Σ hinge: the same problem as the soft margin, seen as a loss.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'overlap', options: P.setOptions(['overlap', 'line', 'iris']) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
      { id: 'C', kind: 'slider', label: 'C', def: 1, min: 0.01, max: 100, log: true, fmt: (v) => H().sg(v, 2) },
    ],
    presets: [
      { label: 'Overlap, C = 1', values: { set: 'overlap', seed: 1, C: 1 } },
      { label: 'A noisy line, C = 10', values: { set: 'line', seed: 3, C: 10 } },
      { label: 'Iris petals, C = 0.1', values: { set: 'iris', seed: 1, C: 0.1 } },
    ],
    counters: [{ id: 'zero', label: 'loss 0' }, { id: 'small', label: 'loss under 1' }, { id: 'big', label: 'loss over 1' }],
    run(inp) {
      const d = P.data(inp.set, { seed: inp.seed }), X = d.X, y = d.y, n = X.length, Cv = +H().sg(inp.C, 2);
      const fit = P.fit(X, y, { C: Cv }), F = H().frames(['zero', 'small', 'big']), q = F.q;
      const m = fit.yf, loss = m.map((v) => Math.max(0, 1 - v));
      const box = { x: 40, y: 24, w: 380, h: 380 }, ax = P.axes('hg', X, box, { pad: 0.3 });
      const mlo = Math.min(-2, Math.floor(Math.min(...m))), mhi = Math.max(3, Math.ceil(Math.max(...m)));
      const lmax = Math.max(3, Math.ceil(1 - mlo));
      const lx = C().axes({ prefix: 'lc', x: 500, y: 40, w: 290, h: 300, xr: [mlo, mhi], yr: [0, lmax], xticks: C().ticks(mlo, mhi, 5), yticks: C().ticks(0, lmax, 4), xname: 'm = y·f(x)', yname: 'loss' });
      const curve = (g) => { const pts = []; for (let k = 0; k <= 120; k++) { const u = mlo + ((mhi - mlo) * k) / 120; pts.push([u, g(u)]); } return pts; };
      const zone = (i) => (m[i] >= 1 - 1e-3 ? 'zero' : m[i] >= 0 ? 'small' : 'big');
      const groups = { zero: [], small: [], big: [] };
      for (let i = 0; i < n; i++) groups[zone(i)].push(i);
      const curves = (all) => {
        const items = [...lx.items, C().vline(lx, 'lc:m1', 1, ''), C().vline(lx, 'lc:m0', 0, '')];
        items.push(C().line(lx, 'lc:h', curve((u) => Math.max(0, 1 - u)), 'act'));
        if (all) items.push(C().line(lx, 'lc:z', curve((u) => (u < 0 ? 1 : 0)), 'faint'), C().line(lx, 'lc:g', curve((u) => Math.log2(1 + Math.exp(-u))), 'c0 thin'));
        items.push(...C().legend('lc:lg', 500, 392, [{ text: 'hinge', cls: 'act' }].concat(all ? [{ text: '0–1 (wrong or not)', cls: 'faint' }, { text: 'logistic', cls: 'c0 thin' }] : []), { size: 13 }));
        return items;
      };
      const onCurve = (show, lit) => show.map((i) => ({ key: 'lc:p' + i, kind: 'dot', x: lx.X(Math.max(mlo, Math.min(mhi, m[i]))), y: lx.Y(loss[i]), r: lit && lit.includes(i) ? 5.5 : 4, cls: 'sp c' + y[i], audit: 'overlap-ok', layer: 'items', opacity: lit && !lit.includes(i) ? 0.35 : 1 }));
      const plot = (lit) => [...ax.items, ...P.boundary(ax, 'hg', fit), ...S.Kit.scatter.points(ax, 'hg:p', X, y, { r: 4.6 }).map((it, i) => Object.assign(it, lit && !lit.includes(i) ? { opacity: 0.35 } : {}))];
      const all = [...Array(n).keys()];

      F.push([...plot(null), ...curves(false)], `The SVM with C = ${q.sg(Cv, 2)} on ${q.n(n)} points. On the right is the hinge loss against m = y·f(x), the point’s value of f with the sign of its class: m ≥ 1 is outside the street on its own side, 0 < m < 1 inside the street, m < 0 the wrong side.`, { phase: 'curve', C: Cv, n });
      F.set('zero', groups.zero.length);
      F.push([...plot(groups.zero), ...curves(false), ...onCurve(all, groups.zero)], `${q.n(groups.zero.length)} points have m ≥ 1 and lose nothing, however far out they lie: the hinge is flat there. Unlike the loss of logistic regression, a point that is already safe does not pull on the line.`, { phase: 'zero', count: groups.zero.length });
      F.set('small', groups.small.length);
      F.push([...plot(groups.small), ...curves(false), ...onCurve(all, groups.small)], `${q.n(groups.small.length)} points are inside the street but on their own side: 0 ≤ m < 1, so each loses 1 − m, less than 1. Together they lose ${q(groups.small.reduce((s, i) => s + loss[i], 0))}.`, { phase: 'small', count: groups.small.length, lossSum: groups.small.reduce((s, i) => s + loss[i], 0) });
      F.set('big', groups.big.length);
      const worst = groups.big.length ? groups.big.reduce((a, i) => (loss[i] > loss[a] ? i : a), groups.big[0]) : -1;
      F.push([...plot(groups.big), ...curves(false), ...onCurve(all, groups.big)], groups.big.length ? `${q.n(groups.big.length)} points are on the wrong side, m < 0, and each loses more than 1; the worst, at m = ${q(m[worst])}, loses ${q(loss[worst])}. The loss grows only in a straight line, so one far outlier cannot drag the line as hard as a squared loss would.` : `No point is on the wrong side at this C: every m is at least 0, so every loss is at most 1.`, { phase: 'big', count: groups.big.length, worst: worst >= 0 ? loss[worst] : 0, worstM: worst >= 0 ? m[worst] : 0 });
      const hingeSum = loss.reduce((a, b) => a + b, 0), primal = 0.5 * fit.norm * fit.norm + Cv * hingeSum, dual = -fit.m.machines[0].objective;
      F.push([...plot(null), ...curves(true), ...onCurve(all, null)], `Add them up: Σ hinge = ${q(hingeSum, 3)}, and ½‖w‖² = ${q(0.5 * fit.norm * fit.norm, 3)}, so the objective ½‖w‖² + C·Σ hinge is ${q(primal, 3)}. The dual problem the solver worked on reaches ${q(dual, 3)}: the same value, as it must be at the optimum. The hinge sits above the 0–1 loss everywhere, a convex stand-in for counting mistakes.`, { phase: 'total', hinge: hingeSum, half: 0.5 * fit.norm * fit.norm, primal, dual });
      for (const f of F.list) f.narrowView = { x: 460, y: 0, w: 350, h: 420 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The hinge loss', text: ['m = y·f(x), y = ±1', 'hinge(m) = max(0, 1 − m)', 'SVM: minimise ½‖w‖² + C·Σ hinge(mᵢ)', '0–1 loss: 1 if m < 0, else 0', 'logistic: log₂(1 + e^(−m))'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'curve', pick: 'option', ask: 'A point lies far outside the street on its own side, at m = 5. What is its hinge loss?', options: ['0', '−4', '5', '1'], answer: '0', explain: (fr, i) => fr[i + 1].say },
      { at: (f) => f.meta.phase === 'big', pick: 'option', ask: 'The primal objective ½‖w‖² + C·Σ hinge and the best value of the dual: how do they compare at the optimum?', options: ['they are equal', 'the primal is twice the dual', 'the dual is always 0', 'they have opposite signs'], answer: 'they are equal', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'What is the hinge loss of a point with m = y·f(x) = 0.25?', options: ['0.75', '0.25', '0', '1.25'], answer: 0, why: 'max(0, 1 − 0.25) = 0.75.' },
      { q: 'Which points have zero hinge loss?', options: ['those with m ≥ 1: on their own side, outside the street', 'only the support vectors', 'those exactly on the line', 'none: the loss is always positive'], answer: 0, why: 'The hinge is flat at 0 from m = 1 onwards.' },
      { q: 'Why use the hinge rather than the 0–1 loss when training?', options: ['it is convex, so the optimisation has one best answer', 'it counts mistakes exactly', 'it is zero for every point', 'it makes C unnecessary'], answer: 0, why: 'The 0–1 loss is flat almost everywhere, with no slope to follow; the hinge is a convex upper bound on it.' },
      { q: 'How does the hinge loss grow for a point far on the wrong side?', options: ['in a straight line', 'with the square of the distance', 'it stops growing at 1', 'exponentially'], answer: 0, why: '1 − m grows linearly as m becomes more negative, which limits an outlier’s pull.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
