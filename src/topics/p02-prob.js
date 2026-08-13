// Part 2, probabilities. The reliability diagram: predictions put in bins, each bin's mean prediction against how
// often its examples were really positive, the gaps to the diagonal and the calibration error, for models that are
// honest, over-confident or coarse. Log loss: every example placed on the curve −ln p at the probability it gave
// its true class, the worst offenders named, and the mean compared across models.
(function (S) {
  'use strict';
  const H = S.P01, P = S.P02, C = () => S.Kit.curve, DT = () => S.Kit.dt, M = () => S.Metrics, ML = () => S.ML;
  const MODELS = {
    logreg: { label: 'logistic regression', make: () => ML().LogisticRegression({ C: 1 }) },
    nb: { label: 'Gaussian naive Bayes', make: () => ML().GaussianNB() },
    tree: { label: 'a deep tree', make: () => ML().DecisionTreeClassifier({ seed: 0 }) },
    knn: { label: '15-NN', make: () => ML().KNeighborsClassifier({ k: 15 }) },
  };
  /** the scores of one model on a binary task's test rows (same split and scaling as P.binary) */
  P.scoresOf = (src, model) => P.once('sc:' + src + ':' + model, () => {
    const B = P.binary(src);
    if (model === 'logreg') return B.score;
    const Z = B.scaler.apply(B.Xall), m = MODELS[model].make().fit(B.train.map((i) => Z[i]), B.train.map((i) => B.yall[i]));
    return m.predictProba(B.test.map((i) => Z[i])).map((p) => p[m.classes.indexOf(1)]);
  });

  S.topic({
    id: 'eval-calibration', family: 'p02-prob', order: 10, title: 'Calibration and reliability diagrams',
    blurb: 'When a model says 0.8, is the answer yes about 80% of the time? Put the test examples in bins by their predicted probability; for each bin, plot the mean prediction across and the share that were really positive up. An honest (calibrated) model lands on the diagonal. Above it the model’s probabilities are too low, below it too high. The expected calibration error is the average distance from the diagonal, weighted by how many examples each bin holds.',
    inputs: [
      { id: 'src', kind: 'choice', label: 'Task', def: 'loans', options: P.sourceOptions(['loans', 'cancer', 'exam', 'moons']) },
      { id: 'model', kind: 'choice', label: 'Model', def: 'nb', options: Object.entries(MODELS).map(([value, m]) => ({ value, label: m.label })) },
      { id: 'bins', kind: 'slider', label: 'Bins', def: 5, min: 3, max: 10, step: 1 },
      { id: 'strategy', kind: 'choice', label: 'Bin edges', def: 'uniform', options: [{ value: 'uniform', label: 'equal widths' }, { value: 'quantile', label: 'equal counts' }] },
    ],
    presets: [
      { label: 'Logistic: close to honest', values: { src: 'loans', model: 'logreg' } },
      { label: 'Naive Bayes on cancer', values: { src: 'cancer', model: 'nb' } },
      { label: 'A deep tree: only 0 or 1', values: { src: 'loans', model: 'tree' } },
      { label: 'Equal-count bins', values: { src: 'loans', model: 'logreg', strategy: 'quantile', bins: 8 } },
    ],
    counters: [{ id: 'bins', label: 'bins used' }],
    run(inp) {
      const B = P.binary(inp.src), sc = P.scoresOf(inp.src, inp.model), y = B.y, n = y.length;
      const cal = M().calibration(y, sc, { nBins: inp.bins, strategy: inp.strategy, posLabel: 1 }), used = cal.bins.filter((b) => b.count > 0);
      const brier = M().brier(y, sc, { posLabel: 1 }).value;
      const ax = C().axes({ prefix: 'ax', x: 50, y: 50, w: 300, h: 300, xr: [0, 1], yr: [0, 1], xname: 'mean predicted probability', yname: 'share really ' + B.posName, tickSize: 12 });
      const diag = { key: 'diag', kind: 'path', d: `M${ax.X(0)},${ax.Y(0)} L${ax.X(1)},${ax.Y(1)}`, cls: 'diag', audit: 'ignore', layer: 'back', z: 3 };
      const hx = C().axes({ prefix: 'hx', x: 420, y: 70, w: 340, h: 120, xr: [0, 1], yr: [0, Math.max(...cal.bins.map((b) => b.count)) * 1.15], yticks: C().ticks(0, Math.max(...cal.bins.map((b) => b.count)), 3), xname: 'predicted probability', yname: 'examples', tickSize: 12 });
      const hist = DT().hist(hx, 'hb', cal.bins.map((b) => b.count), cal.edges.map((e, k) => (k === cal.edges.length - 1 && inp.strategy !== 'quantile' ? 1 : e)), 'soft');
      const edges = cal.edges.slice(1, -1).map((e, k) => ({ key: 'ed' + k, kind: 'path', d: `M${hx.X(e)},${hx.o.y} V${hx.o.y + hx.o.h}`, cls: 'guide act', audit: 'ignore', layer: 'back', z: 3 }));
      const pts = (upto) => used.slice(0, upto).map((b, k) => ({ key: 'bp' + k, kind: 'dot', x: ax.X(b.meanProb), y: ax.Y(b.fraction), r: 4 + Math.min(6, Math.sqrt(b.count) / 2), cls: 'mark c1', audit: 'overlap-ok' }));
      const gaps = used.map((b, k) => ({ key: 'gp' + k, kind: 'path', d: `M${ax.X(b.meanProb)},${ax.Y(b.fraction)} V${ax.Y(b.meanProb)}`, cls: 'resid', audit: 'ignore', layer: 'lines' }));
      const F = new S.Frames(['bins']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, B.title + ': ' + MODELS[inp.model].label + ', ' + n + ' test examples');
      const lines = (ls) => ls.map((l, i) => H.text('ln' + i, 420, 250 + i * 24, l[0], l[1] || 'strong', 13));
      const sharp = sc.filter((p) => p < 0.05 || p > 0.95).length;
      F.add({ items: [title, ...ax.items, diag, ...hx.items, ...hist, ...edges], say: `The ${n} test predictions, binned into ${inp.bins} ${inp.strategy === 'quantile' ? 'bins holding about equal numbers' : 'bins of equal width'}: ${cal.bins.map((b) => b.count).join(', ')} examples. ${sharp} of them (${H.pct(sharp / n)}) are below 0.05 or above 0.95.`, meta: { phase: 'bins', counts: cal.bins.map((b) => b.count), sharp }, view });
      const shown = Math.min(used.length, 3), b0 = used[0];
      F.set('bins', shown);
      F.add({ items: [title, ...ax.items, diag, ...hx.items, ...hist, ...edges, ...pts(shown), ...lines(used.slice(0, shown).map((b) => [H.f(b.lo, 2) + '–' + H.f(b.hi, 2) + ': predicted ' + H.f(b.meanProb, 3) + ', really ' + b.positives + '/' + b.count + ' = ' + H.f(b.fraction, 3), 'strong']))], say: `Each bin becomes one point. The lowest bin, ${H.f(b0.lo, 2)} to ${H.f(b0.hi, 2)}, predicts ${H.f(b0.meanProb, 3)} on average, and ${b0.positives} of its ${b0.count} examples (${H.f(b0.fraction, 3)}) were really ${B.posName}.`, meta: { phase: 'points', mean0: b0.meanProb, frac0: b0.fraction, pos0: b0.positives, n0: b0.count }, view });
      F.set('bins', used.length);
      const worst = used.reduce((w, b) => (Math.abs(b.fraction - b.meanProb) > Math.abs(w.fraction - w.meanProb) ? b : w), used[0]);
      F.add({ items: [title, ...ax.items, diag, ...gaps, ...hx.items, ...hist, ...edges, ...pts(used.length), ...lines([['expected calibration error = ' + H.f(cal.ece, 3), 'act'], ['largest gap: bin ' + H.f(worst.lo, 2) + '–' + H.f(worst.hi, 2), 'strong'], ['predicted ' + H.f(worst.meanProb, 3) + ', really ' + H.f(worst.fraction, 3), 'strong'], ['Brier score = ' + H.f(brier, 3), 'tick']])], say: `All ${used.length} bins, with their gaps to the diagonal. The largest is in the bin ${H.f(worst.lo, 2)} to ${H.f(worst.hi, 2)}: predicted ${H.f(worst.meanProb, 3)}, really ${H.f(worst.fraction, 3)}` + (worst.fraction < worst.meanProb ? ', so the model predicts too high a probability there.' : ', so the model predicts too low a probability there.') + ` Weighted by bin size, the gaps average ${H.f(cal.ece, 3)}, the expected calibration error.`, meta: { phase: 'gaps', ece: cal.ece, brier, worstMean: worst.meanProb, worstFrac: worst.fraction, used: used.length, worstLo: worst.lo, worstHi: worst.hi }, view });
      const cmp = Object.keys(MODELS).map((k) => { const s = P.scoresOf(inp.src, k); return [MODELS[k].label, M().calibration(y, s, { nBins: inp.bins, strategy: inp.strategy, posLabel: 1 }).ece, M().brier(y, s, { posLabel: 1 }).value]; });
      const tb = DT().table({ prefix: 'cmp', x: 420, y: 240, cols: [{ name: 'model', w: 170 }, { name: 'ECE', w: 70, cls: 'num' }, { name: 'Brier', w: 70, cls: 'num' }], rows: cmp.map(([nm, e, b]) => [nm, H.f(e, 3), H.f(b, 3)]), rowH: 26, size: 13, cls: (r) => (cmp[r][0] === MODELS[inp.model].label ? 'act' : '') });
      F.add({ items: [title, ...ax.items, diag, ...gaps, ...hx.items, ...hist, ...edges, ...pts(used.length), ...tb.items], say: `The four models on the same rows and bins: ${H.list(cmp.map(([nm, e]) => nm + ' ' + H.f(e, 3)))} expected calibration error. A model can rank well and still need recalibrating before its probabilities are used as probabilities.`, meta: { phase: 'compare', cmp }, view });
      const ctx = { n, sharpShare: sharp / n, edges: cal.edges };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Calibration', text: ['bin the predictions', 'x: mean prediction in the bin', 'y: share of positives in the bin', 'calibrated: points on the diagonal', 'ECE = Σ (nᵦ / n)·|yᵦ − xᵦ|', 'Brier = mean of (outcome − p)²'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'bins', pick: 'option', ask: 'Will the expected calibration error be below 0.05?', options: ['yes', 'no'], answer: (fr) => (fr.find((f) => f.meta.phase === 'gaps').meta.ece < 0.05 ? 'yes' : 'no'), explain: (fr) => fr.find((f) => f.meta.phase === 'gaps').say },
    ],
    quiz: [
      { q: 'Of 200 examples given about 0.7, 100 were positive. In that range the model is…', options: ['over-confident: it said 0.7, reality was 0.5', 'well calibrated', 'too timid', 'perfect'], answer: 0, why: 'The point sits below the diagonal.' },
      { q: 'A perfectly calibrated model\'s points lie…', options: ['on the diagonal', 'on the top edge', 'on the x axis', 'anywhere'], answer: 0, why: 'Predicted share equals actual share in every bin.' },
      { q: 'A deep tree that predicts only 0 or 1 puts its examples in…', options: ['the two end bins', 'every bin evenly', 'the middle bins', 'no bins'], answer: 0, why: 'Its probabilities are pure leaf votes.' },
      { q: 'Can a model have AUC 0.95 and be badly calibrated?', options: ['yes: ranking and calibration are different things', 'no', 'only for regression', 'only with 2 classes'], answer: 0, why: 'Squeeze every score towards 0.5 and the order, hence AUC, is unchanged.' },
      { q: 'Equal-count (quantile) bins help when…', options: ['most predictions crowd into a narrow range', 'there are no predictions', 'the classes are balanced', 'never'], answer: 0, why: 'Equal-width bins would leave some nearly empty and their points noisy.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Log loss
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'eval-logloss', family: 'p02-prob', order: 20, title: 'Log loss',
    blurb: 'Log loss scores the probabilities themselves. Each example pays −ln of the probability the model gave its true class: nothing for a confident right answer, ln 2 ≈ 0.69 for a shrug at 0.5, and without limit for a confident wrong one. The mean is the log loss. Accuracy cannot tell 0.51 from 0.99; log loss can, and one example given probability 0.001 for its true class costs as much as about ten given 0.5.',
    inputs: [
      { id: 'src', kind: 'choice', label: 'Task', def: 'loans', options: P.sourceOptions(['loans', 'cancer', 'exam', 'moons']) },
      { id: 'model', kind: 'choice', label: 'Model', def: 'logreg', options: Object.entries(MODELS).map(([value, m]) => ({ value, label: m.label })) },
    ],
    presets: [
      { label: 'Logistic regression', values: { src: 'loans', model: 'logreg' } },
      { label: 'A deep tree: certain and sometimes wrong', values: { src: 'loans', model: 'tree' } },
      { label: 'Naive Bayes on cancer', values: { src: 'cancer', model: 'nb' } },
    ],
    counters: [{ id: 'n', label: 'examples' }],
    run(inp) {
      const B = P.binary(inp.src), sc = P.scoresOf(inp.src, inp.model), y = B.y, n = y.length;
      const ll = M().logLoss(y, sc, { labels: [0, 1] }), per = ll.perExample, pt = sc.map((p, i) => (y[i] === 1 ? p : 1 - p));
      const acc = M().accuracy(y, sc.map((p) => (p >= 0.5 ? 1 : 0))).value;
      const cap = 5, ax = C().axes({ prefix: 'ax', x: 50, y: 50, w: 360, h: 300, xr: [0, 1], yr: [0, cap], xname: 'probability given to the true class', yname: '−ln p', tickSize: 12 });
      const curve = C().line(ax, 'cv', Array.from({ length: 200 }, (_, k) => { const p = 0.0067 + (k / 199) * (1 - 0.0067); return [p, -Math.log(p)]; }), 'act');
      const dots = pt.map((p, i) => ({ key: 'd' + i, kind: 'dot', x: ax.X(Math.max(0, p)), y: ax.Y(Math.min(cap, per[i])), r: 3, cls: 'sp c' + y[i] + (per[i] > cap ? ' wrong' : ''), audit: 'overlap-ok' }));
      const order = per.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
      const F = new S.Frames(['n']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, B.title + ': ' + MODELS[inp.model].label + ', ' + n + ' test examples');
      const side = (ls) => ls.map((l, i) => H.text('sd' + i, 450, 70 + i * 24, l[0], l[1] || 'strong', 13));
      F.add({ items: [title, ...ax.items, curve, ...side([['p = 1: pays 0', 'good'], ['p = 0.5: pays ln 2 = ' + H.f(Math.log(2), 3), 'strong'], ['p = 0.1: pays ' + H.f(-Math.log(0.1), 3), 'strong'], ['p = 0.01: pays ' + H.f(-Math.log(0.01), 3), 'bad']])], say: `The penalty for one example is −ln p, where p is the probability the model gave the class that turned out true: 0 at p = 1, ${H.f(Math.log(2), 3)} at 0.5, ${H.f(-Math.log(0.1), 3)} at 0.1, and it keeps growing as p goes to 0.`, meta: { phase: 'curve' }, view });
      const beyond = per.filter((v) => v > cap).length;
      F.set('n', n);
      F.add({ items: [title, ...ax.items, curve, ...dots, ...side([['each dot: one example', 'strong'], ['blue ' + B.negName + ', orange ' + B.posName, 'tick']].concat(beyond ? [[beyond + ' above ' + cap + ' (pinned to the top)', 'bad']] : []))], say: `Every test example sits on the curve at the probability it gave its true class.` + (beyond ? ` ${H.plural(beyond, 'example pays', 'examples pay')} more than ${cap} and ${beyond === 1 ? 'is' : 'are'} pinned to the top edge.` : ` None pays more than ${cap}.`), meta: { phase: 'dots', beyond }, view });
      const top5 = order.slice(0, 5), pf = (p) => (p < 0.001 ? 'under 0.001' : H.g(p, 3));
      F.add({ items: [title, ...ax.items, curve, ...dots, ...top5.map(([, i], r) => Object.assign({}, dots[i], { key: 'ring' + r, r: 7, cls: 'ring hot' })), H.title('wt', 450, 66, 'the five worst'), ...top5.map(([v, i], r) => H.text('w' + r, 450, 92 + r * 24, 'example ' + i + ': p ' + (pt[i] < 0.001 ? '< 0.001' : '= ' + H.g(pt[i], 3)) + ', pays ' + H.f(v, 3), 'mono', 13))], say: `The worst example, ${top5[0][1]}, gave its true class only ${pf(pt[top5[0][1]])} and pays ${H.f(top5[0][0], 3)}. The five worst together pay ${H.f(H.sum(top5.map((q) => q[0])), 3)} of the total ${H.f(H.sum(per), 3)}.`, meta: { phase: 'worst', worst: top5[0][1], worstPay: top5[0][0], top5: H.sum(top5.map((q) => q[0])), total: H.sum(per) }, view });
      F.add({ items: [title, ...ax.items, curve, ...dots, { key: 'mean', kind: 'path', d: `M${ax.o.x},${ax.Y(Math.min(cap, ll.value))} H${ax.o.x + ax.o.w}`, cls: 'guide act', audit: 'ignore', layer: 'top' }, ...side([['log loss = ' + H.f(H.sum(per), 3) + ' / ' + n + ' = ' + H.f(ll.value, 3), 'act'], ['accuracy at 0.5: ' + H.f(acc, 3), 'strong'], ['a constant guess of the class share: ' + H.f(-(H.mean(y) * Math.log(H.mean(y)) + (1 - H.mean(y)) * Math.log(1 - H.mean(y))), 3), 'tick']])], say: `The log loss is the mean penalty: ${H.f(H.sum(per), 3)} / ${n} = ${H.f(ll.value, 3)} (dashed). Always predicting the share of ${B.posName}, ${H.f(H.mean(y), 3)}, would score ${H.f(-(H.mean(y) * Math.log(H.mean(y)) + (1 - H.mean(y)) * Math.log(1 - H.mean(y))), 3)}.`, meta: { phase: 'mean', value: ll.value, acc, base: -(H.mean(y) * Math.log(H.mean(y)) + (1 - H.mean(y)) * Math.log(1 - H.mean(y))) }, view });
      const cmp = Object.keys(MODELS).map((k) => { const s = P.scoresOf(inp.src, k); return [MODELS[k].label, M().logLoss(y, s, { labels: [0, 1] }).value, M().accuracy(y, s.map((p) => (p >= 0.5 ? 1 : 0))).value]; });
      const tb = DT().table({ prefix: 'cmp', x: 450, y: 70, cols: [{ name: 'model', w: 162 }, { name: 'log loss', w: 74, cls: 'num' }, { name: 'accuracy', w: 76, cls: 'num' }], rows: cmp.map(([nm, l, a]) => [nm, H.f(l, 3), H.f(a, 3)]), rowH: 26, size: 13, cls: (r) => (cmp[r][0] === MODELS[inp.model].label ? 'act' : '') });
      F.add({ items: [title, ...ax.items, curve, ...dots, ...tb.items], say: `All four models on the same rows: log loss ${H.list(cmp.map(([nm, l]) => H.f(l, 3) + ' (' + nm + ')'))}. Accuracy ranks them ${H.list(cmp.slice().sort((a, b) => b[2] - a[2]).map((c) => c[0]))}; log loss also asks how much they believed their answers.`, meta: { phase: 'compare', cmp }, view });
      const ctx = { n, cap, total: H.sum(per), share: H.mean(y), worstP: pt[order[0][1]] };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Log loss', text: ['per example: −ln p(true class)', 'log loss = mean over examples', 'binary: −[y ln p + (1 − y) ln(1 − p)]', 'p = 0.5 → 0.693; p → 0 → ∞', 'probabilities are clipped just inside (0, 1)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'curve', pick: 'option', ask: 'Will this model\'s log loss be below ln 2 ≈ 0.693 (better than a shrug)?', options: ['yes', 'no'], answer: (fr) => (fr.find((f) => f.meta.phase === 'mean').meta.value < Math.log(2) ? 'yes' : 'no'), explain: (fr) => fr.find((f) => f.meta.phase === 'mean').say },
    ],
    quiz: [
      { q: 'An example gets probability 0.5 for its true class. It pays…', options: ['ln 2 ≈ 0.693', '0.5', '0', '1'], answer: 0, why: '−ln 0.5 = ln 2.' },
      { q: 'Which costs more: one example at p = 0.01, or five at p = 0.5?', options: ['the one at 0.01 (4.6 against 3.5)', 'the five at 0.5', 'the same', 'neither costs anything'], answer: 0, why: '−ln 0.01 ≈ 4.61; 5 × 0.693 ≈ 3.47.' },
      { q: 'Two models have the same accuracy. One is more confident when right. Its log loss is…', options: ['lower', 'higher', 'the same', 'undefined'], answer: 0, why: 'Higher p on correct answers means smaller penalties.' },
      { q: 'Why are probabilities clipped away from exactly 0 and 1?', options: ['−ln 0 is infinite', 'to speed things up', 'to round them', 'they are not'], answer: 0, why: 'One certain mistake would make the mean infinite.' },
      { q: 'Training a logistic regression minimises…', options: ['the log loss on the training rows', 'the accuracy', 'the AUC', 'the number of support vectors'], answer: 0, why: 'Its loss is exactly the mean −ln p of the true labels.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
