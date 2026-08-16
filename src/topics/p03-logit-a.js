// Part 3, logistic regression in one dimension: hours of study against passing. The score z = w·x + b goes through
// the sigmoid to a probability; each point pays −ln of the probability given to its own class (its log loss, the
// bars underneath), and gradient descent (ML.LogisticRegression, solver 'gd', no penalty) moves w and b against the
// gradient, epoch by epoch. Training runs on standardised hours; the numbers are shown back in hours.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n, C = () => S.Kit.curve;

  /** one input and a pass/fail label: from the exam set (hours of study) or our own two overlapping groups */
  H.oneD = (kind, n, seed) => {
    if (kind === 'exam') { const t = S.Data.load('exam', { seed, n }); return { x: t.X.map((r) => r[0]), y: t.y.slice(), name: 'hours of study', lo: 0, hi: 20, classes: ['fail', 'pass'] }; }
    const r = S.Data.rng(seed), x = [], y = [], gap = kind === 'apart' ? 7 : 4.5, sd = kind === 'apart' ? 1.1 : 2.4;
    for (let i = 0; i < n; i++) { const k = i % 2; x.push(S.Data.round(Math.max(0.2, Math.min(19.8, 10 + (k ? 1 : -1) * gap / 2 + r.normal(0, sd))), 2)); y.push(k); }
    return { x, y, name: 'hours of study', lo: 0, hi: 20, classes: ['fail', 'pass'] };
  };
  const pickSteps = (E) => { const out = new Set([0, 1, 2, 3, 5]); let e = 5; while (e < E) { e = Math.ceil(e * 1.6); if (e < E) out.add(e); } out.add(E); return [...out].filter((v) => v <= E).sort((a, b) => a - b); };

  S.topic({
    id: 'logit-sigmoid', family: 'p03-logit', order: 10, title: 'Logistic regression: the sigmoid and log loss',
    blurb: 'A linear score z = w·x + b can be any number; the sigmoid σ(z) = 1 / (1 + e^(−z)) squeezes it into a probability between 0 and 1. Each training point is charged its log loss, −ln of the probability the model gives its true class: almost nothing when the model is confident and right, a lot when it is confident and wrong. The gradient of the mean log loss is the mean of (p − y)·x, and gradient descent follows it downhill.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'data', def: 'exam', options: [{ value: 'exam', label: 'exam results (study hours)' }, { value: 'overlap', label: 'two overlapping groups' }, { value: 'apart', label: 'two groups far apart' }] },
      { id: 'n', kind: 'number', label: 'students', def: 40, min: 8, max: 120 },
      { id: 'lr', kind: 'slider', label: 'learning rate η', def: 0.5, min: 0.05, max: 3, step: 0.05 },
      { id: 'epochs', kind: 'number', label: 'epochs', def: 120, min: 5, max: 1000 },
      { id: 'seed', kind: 'number', label: 'seed', def: 7, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Exam results, 40 students', values: { data: 'exam', n: 40, lr: 0.5, epochs: 120, seed: 7 } },
      { label: 'Two overlapping groups', values: { data: 'overlap', n: 40, lr: 0.5, epochs: 120, seed: 3 } },
      { label: 'Far apart: the weight keeps growing', values: { data: 'apart', n: 30, lr: 1, epochs: 600, seed: 2 } },
      { label: 'A tiny learning rate', values: { data: 'exam', n: 40, lr: 0.05, epochs: 120, seed: 7 } },
      { label: 'A large learning rate', values: { data: 'exam', n: 40, lr: 3, epochs: 60, seed: 7 } },
    ],
    counters: [{ id: 'epoch', label: 'epoch' }],
    run(inp) {
      const D = H.oneD(inp.data, inp.n, inp.seed), x = D.x, y = D.y, n = x.length;
      const mu = ML().mean(x), sd = Math.sqrt(ML().colVars(x.map((v) => [v]))[0]) || 1, U = x.map((v) => [(v - mu) / sd]);
      const m = ML().LogisticRegression({ solver: 'gd', penalty: 'none', lr: inp.lr, epochs: inp.epochs, traceEvery: 1 }).fit(U, y);
      const steps = m.trace.filter((s) => s.kind === 'gd');
      // back to hours: z = w·(x − μ)/σ + b = (w/σ)·x + (b − wμ/σ)
      const toH = (th) => ({ w: th[0] / sd, b: th[1] - (th[0] * mu) / sd });
      const sig = ML().sigmoid;
      const F = new S.Frames(['epoch']);
      const ax = H.axes({ prefix: 'ls', x: 46, y: 40, w: 460, h: 250, xr: [D.lo, D.hi], yr: [0, 1], xticks: 2, yticks: [0, 0.5, 1], xname: D.name, yname: 'P(pass)' });
      const lossMax = 4;
      const bx = H.axes({ prefix: 'lb', x: 46, y: 360, w: 460, h: 120, xr: [D.lo, D.hi], yr: [0, lossMax], xticks: 0, yticks: [0, 1, 2, 3, 4], yname: 'each point\'s log loss', grid: false });
      const pts = (p) => S.Kit.scatter.points(ax, 'ls:p', x.map((v, i) => [v, y[i] ? 1 : 0]), y, { r: 4.6, cls: (i) => (p && (p[i] >= 0.5 ? 1 : 0) !== y[i] ? 'wrong' : '') });
      const curve = (w, b) => { const g = []; for (let k = 0; k <= 160; k++) { const xx = D.lo + ((D.hi - D.lo) * k) / 160; g.push([xx, sig(w * xx + b)]); } return C().line(ax, 'ls:sig', g, 'act'); };
      const bars = (p) => x.map((v, i) => { const l = -Math.log(Math.max(1e-12, y[i] ? p[i] : 1 - p[i])), h = Math.min(lossMax, l) / lossMax * 120; return { key: 'lb:b' + i, kind: 'cell', x: H.r2(bx.X(v) - 2.2), y: H.r2(bx.Y(0) - h), w: 4.4, h: Math.max(0.6, H.r2(h)), rx: 1, text: '', cls: 'bar c' + y[i], audit: 'ignore' }; });
      const head = () => [...ax.items, ...bx.items, H.title('ls:t', 46, 14, 'The probability of passing'), C().hline(ax, 'ls:half', 0.5, '')];
      const eqX = 560;
      const meta0 = { n, mu, sd, lr: inp.lr, E: inp.epochs };
      // frame 0: the data alone
      F.add({ items: [...head(), ...pts()], say: n + ' students: ' + y.filter((v) => v).length + ' passed (top) and ' + y.filter((v) => !v).length + ' failed (bottom), by ' + D.name + '. Training works on standardised hours (mean ' + n$(mu) + ', spread ' + n$(sd) + ') and starts with w = 0 and b = 0.', meta: Object.assign({ phase: 'data', nPass: y.filter((v) => v).length, nFail: y.filter((v) => !v).length }, meta0) });
      let prevLoss = null;
      for (const e of pickSteps(inp.epochs)) {
        const st = steps[e];
        if (!st) continue;
        const hb = toH(st.theta), p = x.map((v) => sig(hb.w * v + hb.b)), loss = st.loss;
        // the gradient of the mean log loss in the standardised units the engine steps in
        let gw = 0, gb = 0;
        U.forEach((u, i) => { gw += (sig(st.theta[0] * u[0] + st.theta[1]) - y[i]) * u[0]; gb += sig(st.theta[0] * u[0] + st.theta[1]) - y[i]; });
        gw /= n; gb /= n;
        const acc = p.filter((q, i) => (q >= 0.5 ? 1 : 0) === y[i]).length / n;
        const worst = ML().argmax(p.map((q, i) => -Math.log(Math.max(1e-12, y[i] ? q : 1 - q))));
        const items = [...head(), curve(hb.w, hb.b), ...bars(p), ...pts(p)];
        const xb = Math.abs(hb.w) > 1e-9 ? -hb.b / hb.w : null;
        if (xb !== null && xb > D.lo && xb < D.hi) items.push(C().vline(ax, 'ls:cut', xb, 'act'), H.text('ls:cutl', ax.X(xb) + 6, ax.Y(0.16), 'p = 0.5 at ' + n$(xb) + ' h', 'pl-lab act', 12));
        items.push(S.Kit.scatter.ring(ax, 'ls:worst', x[worst], y[worst] ? 1 : 0, 'hot', 9));
        items.push(...H.eqs('ls:eq', eqX, 60, [['epoch ' + e, 'act'], ['z = ' + n$(hb.w) + '·x' + H.term(hb.b, '', false), ''], ['p = σ(z)', 'muted'], ['mean log loss ' + n$(loss), 'act'], ['∂L/∂w = ' + n$(gw), 'muted'], ['∂L/∂b = ' + n$(gb), 'muted'], [Math.round(acc * 100) + '% on the right side', 'good']], 15, 30));
        F.set('epoch', e);
        let say;
        if (e === 0) say = 'With w = 0 the score is 0 for everyone, so p = σ(0) = 0.5: every student pays −ln 0.5 = ' + n$(loss) + ', and that is the mean log loss. The gradient in standardised units is ∂L/∂w = ' + n$(gw) + ', ∂L/∂b = ' + n$(gb) + '.';
        else say = 'Epoch ' + e + ': z = ' + n$(hb.w) + '·x' + H.term(hb.b, '', false) + ' in hours, mean log loss ' + n$(loss) + (prevLoss !== null && loss > prevLoss + 1e-12 ? ' (up: the step overshot)' : '') + ', ' + Math.round(acc * 100) + '% on the right side of p = 0.5. The costliest point (ringed) is at ' + n$(x[worst]) + ' h, ' + (y[worst] ? 'a pass' : 'a fail') + ' given P(pass) = ' + n$(p[worst]) + '.';
        F.add({ items, say, meta: Object.assign({ phase: 'step', e, w: hb.w, b: hb.b, wStd: st.theta[0], bStd: st.theta[1], loss, gw, gb, acc, xb, worstX: x[worst], worstP: p[worst] }, meta0) });
        prevLoss = loss;
      }
      const last = steps[steps.length - 1], hb = toH(last.theta);
      const sepGap = (() => { const a = x.filter((v, i) => !y[i]), b = x.filter((v, i) => y[i]); return Math.min(...b) - Math.max(...a); })();
      const sayEnd = sepGap > 0 ? 'The two groups do not overlap, so the log loss can always be lowered by making the sigmoid steeper: w is ' + n$(hb.w) + ' after ' + inp.epochs + ' epochs and would keep growing for ever. Without a penalty, logistic regression on separable data has no finite best answer.' : 'After ' + inp.epochs + ' epochs the mean log loss is ' + n$(last.loss) + '. Points far on the wrong side still cost the most; the curve cannot move them without making others worse.';
      F.add({ items: F.last().items, say: sayEnd, meta: Object.assign({ phase: 'end', w: hb.w, loss: last.loss, sepGap }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Logistic regression', text: ['z = w·x + b,  p = σ(z) = 1 / (1 + e^(−z))', 'log loss of a point: −[y ln p + (1 − y) ln(1 − p)]', 'L = mean of the log losses', '∂L/∂w = mean of (p − y)·x', '∂L/∂b = mean of (p − y)', 'step: w ← w − η ∂L/∂w'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'Before any training (w = 0, b = 0), what probability does the model give every student?', options: ['0.5', '0', '1', 'the share who passed'], answer: '0.5', explain: 'σ(0) = 0.5 whatever x is.' },
      { at: (f, i, fr) => f.meta.phase === 'step' && f.meta.e === 0, pick: 'option', ask: 'Which points will cost the most once the curve has formed?', options: ['the ones on the wrong side, far from the crossing', 'the ones on the right side, far from the crossing', 'the ones right at the crossing', 'all the same'], answer: 'the ones on the wrong side, far from the crossing', explain: 'There the model is confidently wrong, and −ln of a small probability is large.' },
    ],
    quiz: [
      { q: 'What is σ(0)?', options: ['0.5', '0', '1', 'e'], answer: 0, why: '1 / (1 + e⁰) = 1/2.' },
      { q: 'A pass predicted with P(pass) = 0.9 costs…', options: ['−ln 0.9 ≈ 0.105', '−ln 0.1 ≈ 2.3', '0.9', '0.1'], answer: 0, why: 'The log loss is −ln of the probability given to the true class.' },
      { q: 'The gradient of the mean log loss with respect to w is…', options: ['the mean of (p − y)·x', 'the mean of y·x', 'p(1 − p)', 'the mean of (y − p)²'], answer: 0, why: 'The sigmoid\'s derivative cancels against the log, leaving the plain error p − y.' },
      { q: 'On data where the classes do not overlap at all, unpenalised logistic regression…', options: ['keeps growing w for ever', 'stops at w = 1', 'cannot start', 'gives p = 0.5'], answer: 0, why: 'A steeper sigmoid always lowers the loss; a penalty (C in most libraries) stops it.' },
      { q: 'Where is the decision boundary of z = 2x − 10?', options: ['x = 5', 'x = 2', 'x = 10', 'x = −5'], answer: 0, why: 'p = 0.5 where z = 0.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
