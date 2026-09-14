// Part 17, distribution shift: the three kinds (the inputs move, the class shares change, the rule itself moves)
// against a model trained before the shift; then watching a deployed model batch by batch, with a population
// stability index on each input and the accuracy once the true outcomes arrive.
(function (S) {
  'use strict';
  const P = S.P17, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const KINDS = { covariate: 'covariate: the inputs move', label: 'label: the class shares change', concept: 'concept: the rule itself moves' };
  /** the model of the shift topics: logistic regression on the unshifted training set (as S.Rob.shiftCurve fits it) */
  P.shiftModel = (kind, seed) => P.once('sm:' + kind + ':' + seed, () => { const base = S.Rob.shift({ kind, amount: 0, seed }); return { base, m: S.Rob.logreg(base.train.X, base.train.y, { iters: 300 }) }; });
  P.accOf = (m, s) => s.X.filter((x, i) => m.predict(x) === s.y[i]).length / s.y.length;
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  /** the labelling rule x₂ = 0.8 sin(1.2 x₁ + turn), as a path over the axes */
  const rule = (ax, key, turn, cls) => {
    const pts = []; for (let k = 0; k <= 80; k++) { const u = ax.o.xr[0] + ((ax.o.xr[1] - ax.o.xr[0]) * k) / 80; pts.push([u, 0.8 * Math.sin(1.2 * u + turn)]); }
    return C().line(ax, key, pts, cls);
  };

  S.topic({
    id: 'shift-kinds', family: 'p17-shift', order: 10, title: 'Three kinds of distribution shift',
    blurb: 'A model learns from the data it was given. After it is put to use, the data may change: the inputs may drift into regions it rarely saw (covariate shift), the share of each class may change while each class looks the same (label shift), or the rule linking inputs to labels may itself move (concept shift). Each hurts in its own way, and each has its own remedy.',
    inputs: [
      { id: 'kind', kind: 'choice', label: 'Kind of shift', def: 'covariate', options: Object.keys(KINDS).map((k) => ({ value: k, label: KINDS[k] })) },
      { id: 'amount', kind: 'slider', label: 'Amount of shift', def: 0.6, min: 0, max: 1, step: 0.05, fmt: (v) => v.toFixed(2) },
      { id: 'seed', kind: 'number', label: 'Seed (the data)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Covariate shift', values: { kind: 'covariate', amount: 0.6, seed: 1 } },
      { label: 'Label shift', values: { kind: 'label', amount: 1, seed: 1 } },
      { label: 'Concept shift', values: { kind: 'concept', amount: 0.5, seed: 1 } },
      { label: 'No shift at all', values: { kind: 'covariate', amount: 0, seed: 2 } },
    ],
    counters: [{ id: 'sets', label: 'data sets' }],
    run(inp) {
      const F = H().frames(['sets']), q = F.q, { base, m } = P.shiftModel(inp.kind, inp.seed), s = S.Rob.shift({ kind: inp.kind, amount: inp.amount, seed: inp.seed }), tr = base.train, te = s.test;
      const ax = C().axes({ prefix: 'sk:ax', x: 20, y: 50, w: 340, h: 340, xr: [-3, 2.8], yr: [-2.9, 2.9], xticks: 1, yticks: 1, xname: 'x₁', yname: 'x₂', tickSize: 12 });
      const line = H().level(ax, 'sk:m', (u, v) => m.prob([u, v]), 0.5, 'p17edge', 60);
      const accTr = P.accOf(m, tr), accTe = P.accOf(m, te);
      const curves = P.once('skc:' + inp.seed, () => S.Rob.shiftKinds.map((k) => S.Rob.shiftCurve({ kind: k, model: 'logistic', seed: inp.seed })));
      const cx = C().axes({ prefix: 'sk:cx', x: 440, y: 70, w: 280, h: 170, xr: [0, 1], yr: [0.4, 1], xticks: 0.25, yticks: 0.2, xname: 'amount of shift', yname: 'test accuracy', tickSize: 12 });
      const right = (all) => {
        const it = [...cx.items];
        S.Rob.shiftKinds.forEach((k, j) => { if (all || k === inp.kind) { it.push(C().line(cx, 'sk:c' + j, curves[j].rows.map((r) => [r.amount, r.accuracy]), k === inp.kind ? 'p17cur' : 'p17other')); } });
        it.push(C().dot(cx, 'sk:cd', inp.amount, accTe, 'p17cur', 5.5));
        it.push(...C().legend('sk:lg', 440, 296, [{ text: inp.kind + ' shift (this one)', cls: 'p17cur' }].concat(all ? [{ text: 'the other two kinds', cls: 'p17other' }] : []), { dir: 'col' }));
        return it;
      };
      const keys = [...P.key('sk:k0', 440, 380, 'c0', 'class 0'), ...P.key('sk:k1', 540, 380, 'c1', 'class 1')];
      F.set('sets', 1);
      F.push([...ax.items, rule(ax, 'sk:r', 0, 'p17other'), line, ...P.dots(ax, 'sk:p', tr.X, tr.y), ...keys, H().head('sk:h', 20, 24, 'Training data'), ...H().lines('sk:t', 440, 80, [['dashed: the true rule', 'legend'], ['black: what the model learned', 'legend']], 13, 22)],
        `The training inputs sit around x₁ = ${q(mean(tr.X.map((x) => x[0])), 2)}, where the true rule (dashed wave) is almost flat. A logistic regression learns a nearly straight boundary there and gets ${q.pct(accTr)} of its ${q.n(tr.y.length)} training points right.`, { phase: 'train', accTr, n: tr.y.length, mx0: mean(tr.X.map((x) => x[0])) });
      F.set('sets', 2);
      const mx = mean(te.X.map((x) => x[0])), mx0 = mean(tr.X.map((x) => x[0])), sh = mean(te.y), sh0 = mean(tr.y);
      const turn = inp.kind === 'concept' ? Math.PI * inp.amount : 0;
      const what = () => inp.kind === 'covariate' ? `The test inputs have moved: their x₁ averages ${q(mx, 2)}, against ${q(mx0, 2)} in training, onto the crest of the wave the model never saw. The rule has not changed.`
        : inp.kind === 'label' ? `Class one (orange) now makes up ${q.pct(sh)} of the test set, against ${q.pct(sh0)} in training, though each class looks just as it did.`
        : `The inputs look the same (x₁ averages ${q(mx, 2)}, against ${q(mx0, 2)}), but the rule itself has moved: the wave is shifted by a phase of ${q(turn, 2)} radians.`;
      F.push([...ax.items, rule(ax, 'sk:r', turn, 'p17other'), line, ...P.dots(ax, 'sk:p', te.X, te.y), ...keys, H().head('sk:h', 20, 24, 'After the shift: ' + inp.kind), ...H().lines('sk:t', 440, 80, [['amount ' + H().fx(inp.amount, 2), 'legend']], 13, 22)],
        inp.amount === 0 ? `With no shift the test set is drawn just like the training set: x₁ averages ${q(mx, 2)} and class one makes up ${q.pct(sh)}.` : what(), { phase: 'test', mx, mx0, share: sh, share0: sh0, turn, amount: inp.amount });
      const wrong = te.X.map((x, i) => m.predict(x) !== te.y[i]);
      const a0 = curves[S.Rob.shiftKinds.indexOf(inp.kind)].rows[0].accuracy;
      F.push([...ax.items, rule(ax, 'sk:r', turn, 'p17other'), line, ...P.dots(ax, 'sk:p', te.X, te.y, { cls: (i) => (wrong[i] ? 'p17wrong' : ''), r: (i) => (wrong[i] ? 4.6 : 3.4) }), ...right(false), H().head('sk:h', 20, 24, 'The old model on the new data')],
        `The model, unchanged, gets ${q.pct(accTe)} of the ${q.n(te.y.length)} shifted test points right (wrong ones outlined), against ${q.pct(a0)} with no shift.` + (inp.kind === 'label' ? ' Label shift hurts least here: the boundary is still right for each class, but its cut-off was tuned to the old class shares.' : inp.kind === 'covariate' && inp.amount > 0 ? ' The straight boundary was a fine guess only where the training data lay.' : inp.kind === 'concept' && inp.amount > 0 ? ' Points that look exactly like training points now carry other labels.' : ''), { phase: 'acc', accTe, a0, n: te.y.length });
      const drops = curves.map((c) => c.drop);
      F.push([...ax.items, rule(ax, 'sk:r', turn, 'p17other'), line, ...P.dots(ax, 'sk:p', te.X, te.y), ...right(true), H().head('sk:h', 20, 24, 'Remedies'), ...H().lines('sk:t', 440, 360, [[inp.kind === 'covariate' ? 'collect data where the inputs went' : inp.kind === 'label' ? 're-estimate the class shares' : 'retrain on fresh labels', 'strong']], 13, 22)],
        `Across amounts from none to full, the worst drop in accuracy is ${q(drops[0], 2)} for covariate shift, ${q(drops[1], 2)} for label shift and ${q(drops[2], 2)} for concept shift. ` + (inp.kind === 'covariate' ? 'Remedy: gather labelled data where the inputs now fall, or weight training points that resemble the new ones.' : inp.kind === 'label' ? 'Remedy: estimate the new class shares and move the cut-off; the rest of the model can stay.' : 'Remedy: only fresh labelled data reveals the new rule, so the model must be retrained on it.'), { phase: 'kinds', drops });
      for (const f of F.list) f.narrowView = { x: -22, y: 0, w: 402, h: 420 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Kinds of shift', text: ['covariate: p(x) changes, p(y | x) stays', 'label:     p(y) changes, p(x | y) stays', 'concept:   p(y | x) changes', 'here: y = [x₂ > 0.8 sin(1.2 x₁ + turn)]', 'train: x₁ ~ N(−1.3, 0.5²), x₂ ~ N(0, 0.9²)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'test', pick: 'option', once: true,
      ask: 'Will the old model’s accuracy on this test set be lower than with no shift?',
      options: ['clearly lower', 'about the same', 'higher'],
      answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'acc').meta, d = m.a0 - m.accTe; return d > 0.03 ? 'clearly lower' : d < -0.03 ? 'higher' : 'about the same'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'acc').say,
    }],
    quiz: [
      { q: 'A spam filter meets messages in a new language its training set lacked, with the same meaning of spam. This is…', options: ['covariate shift', 'label shift', 'concept shift', 'no shift'], answer: 0, why: 'The inputs moved; the rule did not.' },
      { q: 'During a flu season the share of sick patients doubles, but each illness looks the same. This is…', options: ['label shift', 'covariate shift', 'concept shift', 'overfitting'], answer: 0, why: 'Class shares changed; each class’s inputs did not.' },
      { q: 'Which shift cannot be seen from the inputs alone?', options: ['concept shift', 'covariate shift', 'none of them', 'all of them'], answer: 0, why: 'The inputs look the same; only the labels reveal it.' },
      { q: 'The usual remedy for concept shift is…', options: ['retrain on freshly labelled data', 'rescale the inputs', 'change the threshold only', 'nothing'], answer: 0, why: 'The old labels describe an old rule.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const B = 8;
  /** decile edges of a column (the interior cut points) */
  P.deciles = (v) => { const s = v.slice().sort((a, b) => a - b); return Array.from({ length: 9 }, (_, k) => s[Math.floor(((k + 1) * s.length) / 10)]); };
  /** the share of values in each of the ten bins cut at the edges, floored at 0.0001 */
  P.shares = (v, edges) => { const c = new Array(10).fill(0); for (const x of v) { let b = 0; while (b < 9 && x >= edges[b]) b++; c[b]++; } return c.map((k) => Math.max(1e-4, k / v.length)); };
  /** population stability index: Σ (new − ref) ln(new / ref) over the bins */
  P.psi = (ref, now) => ref.reduce((s, r, k) => s + (now[k] - r) * Math.log(now[k] / r), 0);
  const SCEN = { covariate: 'the inputs drift (covariate)', concept: 'the rule drifts (concept)', label: 'the class shares drift (label)', none: 'nothing drifts' };
  P.monitor = (scen, speed, seed) => P.once(['mon', scen, speed, seed].join(':'), () => {
    const kind = scen === 'none' ? 'covariate' : scen, { base, m } = P.shiftModel(kind, seed), tr = base.train;
    const edges = [0, 1].map((j) => P.deciles(tr.X.map((x) => x[j]))), ref = [0, 1].map((j) => P.shares(tr.X.map((x) => x[j]), edges[j]));
    const batches = [];
    for (let b = 1; b <= B; b++) {
      const amount = scen === 'none' ? 0 : Math.min(1, speed * b), s = S.Rob.shift({ kind, amount, n: 150, seed: seed * 100 + b }).test;
      const psi = [0, 1].map((j) => P.psi(ref[j], P.shares(s.X.map((x) => x[j]), edges[j])));
      batches.push({ b, amount, X: s.X, y: s.y, psi, stat: Math.max(...psi), acc: P.accOf(m, s) });
    }
    return { tr, m, edges, batches, accTr: P.accOf(m, tr) };
  });
  S.topic({
    id: 'shift-monitor', family: 'p17-shift', order: 20, title: 'Watching a deployed model for drift',
    blurb: 'Once a model is in use, new data arrives in batches. Comparing each batch’s inputs with the training inputs, bin by bin, gives a drift score (the population stability index); crossing a threshold raises an alarm before anyone knows whether the answers got worse. But a change in the rule itself leaves the inputs looking the same, so the true outcomes, when they arrive, must be watched too.',
    inputs: [
      { id: 'scen', kind: 'choice', label: 'What changes', def: 'covariate', options: Object.keys(SCEN).map((k) => ({ value: k, label: SCEN[k] })) },
      { id: 'speed', kind: 'slider', label: 'Drift per batch', def: 0.12, min: 0.04, max: 0.2, step: 0.02, fmt: (v) => v.toFixed(2) },
      { id: 'thr', kind: 'slider', label: 'Alarm threshold', def: 0.2, min: 0.1, max: 0.3, step: 0.05, fmt: (v) => v.toFixed(2) },
      { id: 'seed', kind: 'number', label: 'Seed (the data)', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'The inputs drift', values: { scen: 'covariate', speed: 0.12, thr: 0.2, seed: 1 } },
      { label: 'The rule drifts: a silent failure', values: { scen: 'concept', speed: 0.12, thr: 0.2, seed: 1 } },
      { label: 'The class shares drift', values: { scen: 'label', speed: 0.12, thr: 0.2, seed: 1 } },
      { label: 'Nothing drifts', values: { scen: 'none', speed: 0.12, thr: 0.2, seed: 3 } },
    ],
    counters: [{ id: 'batch', label: 'batches' }, { id: 'alarms', label: 'alarms' }],
    run(inp) {
      const F = H().frames(['batch', 'alarms']), q = F.q, R = P.monitor(inp.scen, inp.speed, inp.seed), tr = R.tr;
      const hx = [C().axes({ prefix: 'sm:h0', x: 30, y: 60, w: 320, h: 130, xr: [-3.5, 2.5], yr: [0, 0.5], xticks: 1, yticks: 0.25, xname: 'x₁', yname: 'share', tickSize: 12 }),
        C().axes({ prefix: 'sm:h1', x: 30, y: 270, w: 320, h: 130, xr: [-3, 3], yr: [0, 0.5], xticks: 1, yticks: 0.25, xname: 'x₂', yname: 'share', tickSize: 12 })];
      const lo = [-3.5, -3];
      const histos = (X, cls, key) => hx.flatMap((ax, j) => P.hist(ax, key + j, X.map((x) => x[j]), lo[j], 0.5, 12, cls, 1 / X.length).items);
      const px = C().axes({ prefix: 'sm:px', x: 430, y: 60, w: 290, h: 130, xr: [0.5, B + 0.5], yr: [0, 1.2], xticks: [1, 2, 3, 4, 5, 6, 7, 8], yticks: 0.4, xname: 'batch', yname: 'drift score (PSI, larger input)', tickSize: 12 });
      const acx = C().axes({ prefix: 'sm:ac', x: 430, y: 270, w: 290, h: 130, xr: [0.5, B + 0.5], yr: [0.4, 1], xticks: [1, 2, 3, 4, 5, 6, 7, 8], yticks: 0.2, xname: 'batch', yname: 'accuracy (once outcomes arrive)', tickSize: 12 });
      const clip = (v) => Math.min(1.2, v);
      const right = (k) => {
        const done = R.batches.slice(0, k), it = [...px.items, ...acx.items, C().hline(px, 'sm:thr', inp.thr, 'p17thr'), C().hline(acx, 'sm:base', R.accTr, 'p17thr')];
        done.forEach((b) => it.push({ key: 'sm:bar' + b.b, kind: 'cell', x: P.r2(px.X(b.b) - 10), y: px.Y(clip(b.stat)), w: 20, h: P.r2(px.Y(0) - px.Y(clip(b.stat))), rx: 2, text: '', cls: 'bar ' + (b.stat > inp.thr ? 'p17alarm' : 'p17calm'), audit: 'ignore' }));
        if (done.length) { it.push(C().line(acx, 'sm:acc', done.map((b) => [b.b, b.acc]), 'p17acc')); done.forEach((b) => it.push(C().dot(acx, 'sm:ad' + b.b, b.b, b.acc, 'p17acc', 4))); }
        return it;
      };
      const keys = [...P.key('sm:k0', 30, 440, 'p17ref', 'training inputs'), ...P.key('sm:k1', 180, 440, 'p17now', 'this batch')];
      F.push([...histos(tr.X, 'p17ref', 'sm:r'), ...hx.flatMap((a) => a.items), ...right(0), ...keys, H().head('sm:h', 30, 24, 'The reference: training inputs')],
        `The model is trained on ${P.qpl(q, tr.y.length, 'row')} and gets ${q.pct(R.accTr)} of them right (the dashed line on the lower right). Each input’s training values are cut into ten bins of equal share; every new batch will be compared bin by bin.`, { phase: 'ref', n: tr.y.length, accTr: R.accTr });
      let alarms = 0, first = null;
      for (const b of R.batches) {
        const alarm = b.stat > inp.thr;
        if (alarm) { alarms++; if (first === null) first = b.b; }
        F.set('batch', b.b).set('alarms', alarms);
        const j = b.psi[0] >= b.psi[1] ? 0 : 1;
        F.push([...histos(tr.X, 'p17ref', 'sm:r'), ...histos(b.X, 'p17now', 'sm:n'), ...hx.flatMap((a) => a.items), ...right(b.b), ...keys, H().head('sm:h', 30, 24, 'Batch ' + b.b + (alarm ? ': alarm' : '')), H().text('sm:al', 430, 34, alarm ? 'drift score over the threshold' : 'drift score under the threshold', alarm ? 'p17warn' : 'p17ok', 13)],
          `Batch ${q.n(b.b)}, ${P.qpl(q, b.y.length, 'row')}: the drift score is ${q(b.psi[0], 3)} on x₁ and ${q(b.psi[1], 3)} on x₂; the larger, ${q(b.stat, 3)}, is ${alarm ? 'over' : 'under'} the threshold ${q(inp.thr, 2)}${alarm ? ', so an alarm is raised' : ''}. When its outcomes arrive, the model turns out ${q.pct(b.acc)} right.`, { phase: 'batch', b: b.b, psi: b.psi, stat: b.stat, acc: b.acc, alarm, thr: inp.thr, n: b.y.length, j });
      }
      const worst = R.batches.reduce((w, b) => (b.acc < w.acc ? b : w));
      const lastB = R.batches[B - 1];
      F.push([...histos(tr.X, 'p17ref', 'sm:r'), ...histos(lastB.X, 'p17now', 'sm:n'), ...hx.flatMap((a) => a.items), ...right(B), ...keys, H().head('sm:h', 30, 24, 'What the monitor saw')],
        (alarms ? `The input monitor raised ${P.qpl(q, alarms, 'alarm')}, the first at batch ${q.n(first)}.` : 'The input monitor never raised an alarm.') + ` Accuracy was lowest at batch ${q.n(worst.b)}, ${q.pct(worst.acc)}, against ${q.pct(R.accTr)} in training. ` +
          (inp.scen === 'concept' ? 'The rule changed while the inputs stayed put, so only the outcomes could reveal it: watch accuracy on delayed labels, not just the inputs.' : inp.scen === 'none' ? 'Small scores like these are sampling noise; a threshold set too low would cry wolf.' : 'Input drift is a warning to check, not proof of harm: confirm it with outcomes before retraining.'), { phase: 'end', alarms, first, worst: worst.b, worstAcc: worst.acc, accTr: R.accTr });
      for (const f of F.list) f.narrowView = { x: -28, y: 0, w: 408, h: 460 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Population stability index', text: ['cut each input at its training deciles', 'r_k = training share in bin k (one tenth)', 'b_k = this batch’s share in bin k', 'PSI = Σ_k (b_k − r_k) · ln(b_k / r_k)', 'a rule of thumb: under 0.1 calm, over 0.2 drifted'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'ref', pick: 'option', once: true,
      ask: 'Will the input monitor raise an alarm in these eight batches?',
      options: ['yes', 'no'],
      answer: (fr) => (fr[fr.length - 1].meta.alarms ? 'yes' : 'no'),
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'A batch has exactly the training share (one tenth) in every bin. Its PSI is…', options: ['0', '1', '0.1', '10'], answer: 0, why: 'Every term (b − r) ln(b / r) is zero.' },
      { q: 'Why can concept shift slip past an input monitor?', options: ['the inputs keep their distribution; only the labels change', 'PSI ignores the inputs', 'it always raises alarms', 'concept shift is impossible'], answer: 0, why: 'Nothing in the inputs alone tells that the rule moved.' },
      { q: 'What should follow an input-drift alarm?', options: ['check outcomes and the data pipeline before acting', 'delete the model', 'ignore it', 'raise the threshold until it stops'], answer: 0, why: 'Drift may or may not hurt; it must be confirmed.' },
      { q: 'Setting the alarm threshold very low makes the monitor…', options: ['raise many false alarms from sampling noise', 'miss every drift', 'more accurate always', 'unable to compute PSI'], answer: 0, why: 'Small random differences between batches cross a low threshold.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
