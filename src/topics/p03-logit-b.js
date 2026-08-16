// Part 3, logistic regression in the plane: the decision boundary forming while it trains. The shading is the
// model's probability of the orange class at every place, the line is p = 0.5, ringed points are on the wrong side.
// At the right: the mean log loss falling, and the weights' own path in weight space with the step about to be
// taken (minus the gradient times the learning rate). ML.LogisticRegression, solver 'gd', on standardised inputs.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n, C = () => S.Kit.curve, SC = () => S.Kit.scatter;

  /** a two-class set in the plane: {X, y, names, classes} */
  H.plane2 = (kind, n, seed) => {
    if (kind === 'exam') { const t = S.Data.load('exam', { seed, n }); return { X: t.X.map((r) => [r[0], r[1]]), y: t.y.slice(), names: ['study (h)', 'sleep (h)'], classes: ['fail', 'pass'] }; }
    if (kind === 'linear') { const t = S.Data.linear({ n, seed, noise: 0.06 }); return { X: t.X, y: t.y, names: ['x₁', 'x₂'], classes: ['class 0', 'class 1'] }; }
    if (kind === 'moons') { const t = S.Data.moons({ n, seed, noise: 0.15 }); return { X: t.X, y: t.y, names: ['x₁', 'x₂'], classes: ['class 0', 'class 1'] }; }
    if (kind === 'xor') { const t = S.Data.xor({ n, seed }); return { X: t.X, y: t.y, names: ['x₁', 'x₂'], classes: ['class 0', 'class 1'] }; }
    const g = H.gauss2(n, seed, [-1, -0.5], [1.2, 1], [0.9, 0.8, 0.3], [0.9, 0.8, 0.3]);
    return { X: g.X, y: g.y, names: ['x₁', 'x₂'], classes: ['class 0', 'class 1'] };
  };
  const pickSteps = (E) => { const out = new Set([0, 1, 2, 4]); let e = 4; while (e < E) { e = Math.ceil(e * 1.55); if (e < E) out.add(e); } out.add(E); return [...out].filter((v) => v <= E).sort((a, b) => a - b); };

  S.topic({
    id: 'logit-boundary', family: 'p03-logit', order: 20, title: 'A decision boundary forming',
    blurb: 'With two inputs the score z = w₁x₁ + w₂x₂ + b is a tilted plane, and p = σ(z) shades the picture from one class to the other. The boundary p = 0.5 is the line where z = 0: always straight, whatever the data. Each epoch of gradient descent moves the three numbers a little against the gradient of the mean log loss; the line turns and slides until the misfits cost as little as they can.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'data', def: 'blobs', options: [{ value: 'blobs', label: 'two clouds' }, { value: 'linear', label: 'a noisy line' }, { value: 'exam', label: 'exam: study and sleep' }, { value: 'moons', label: 'two moons' }, { value: 'xor', label: 'XOR' }] },
      { id: 'n', kind: 'number', label: 'points', def: 80, min: 10, max: 200 },
      { id: 'lr', kind: 'slider', label: 'learning rate η', def: 0.5, min: 0.05, max: 4, step: 0.05 },
      { id: 'epochs', kind: 'number', label: 'epochs', def: 150, min: 5, max: 800 },
      { id: 'seed', kind: 'number', label: 'seed', def: 4, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Two clouds', values: { data: 'blobs', n: 80, lr: 0.5, epochs: 150, seed: 4 } },
      { label: 'A noisy line', values: { data: 'linear', n: 120, lr: 1, epochs: 200, seed: 2 } },
      { label: 'Exam: study and sleep', values: { data: 'exam', n: 120, lr: 0.5, epochs: 200, seed: 7 } },
      { label: 'Two moons: a line is not enough', values: { data: 'moons', n: 120, lr: 1, epochs: 200, seed: 5 } },
      { label: 'XOR: no line works', values: { data: 'xor', n: 80, lr: 1, epochs: 100, seed: 3 } },
    ],
    counters: [{ id: 'epoch', label: 'epoch' }, { id: 'wrong', label: 'misclassified' }],
    run(inp) {
      const D = H.plane2(inp.data, inp.n, inp.seed), X = D.X, y = D.y, n = X.length;
      const st = H.standardise(X), mu = st.mean, sd = st.sd, U = st.X;
      const m = ML().LogisticRegression({ solver: 'gd', penalty: 'none', lr: inp.lr, epochs: inp.epochs, traceEvery: 1 }).fit(U, y);
      const steps = m.trace.filter((s) => s.kind === 'gd'), sig = ML().sigmoid;
      // in the original units: z = Σ wⱼ (xⱼ − μⱼ)/σⱼ + b
      const orig = (th) => ({ w: [th[0] / sd[0], th[1] / sd[1]], b: th[2] - (th[0] * mu[0]) / sd[0] - (th[1] * mu[1]) / sd[1] });
      const grad = (th) => { const g = [0, 0, 0]; U.forEach((u, i) => { const e = sig(th[0] * u[0] + th[1] * u[1] + th[2]) - y[i]; g[0] += e * u[0]; g[1] += e * u[1]; g[2] += e; }); return g.map((v) => v / n); };
      const xr = H.nice(X.map((r) => r[0]), 0.06), yr = H.nice(X.map((r) => r[1]), 0.06);
      const ax = H.axes({ prefix: 'lb', x: 46, y: 40, w: 400, h: 400, xr, yr, xname: D.names[0], yname: D.names[1], grid: false });
      // the loss and the weights' path
      const L0 = steps[0].loss, lossHi = Math.max(L0, ...steps.map((s) => s.loss).filter(isFinite).slice(0, 3)) * 1.05;
      const lax = H.axes({ prefix: 'll', x: 530, y: 40, w: 220, h: 130, xr: [0, inp.epochs], yr: [0, Math.max(0.8, Math.ceil(lossHi * 10) / 10)], xticks: C().ticks(0, inp.epochs, 3), yticks: Math.ceil(lossHi * 10) / 10 > 1 ? 0.5 : 0.25, xname: 'epoch', yname: 'mean log loss' });
      const ws = steps.map((s) => s.theta);
      const wlim = Math.max(1, ...ws.map((t) => Math.max(Math.abs(t[0]), Math.abs(t[1])))) * 1.15, wr = H.nice([-wlim, wlim], 0);
      const wax = H.axes({ prefix: 'lw', x: 560, y: 260, w: 190, h: 190, xr: wr, yr: wr, xticks: H.tickList(wr[0], wr[1], wr[1] > 6 ? 4 : wr[1] > 3 ? 2 : 1), yticks: H.tickList(wr[0], wr[1], wr[1] > 6 ? 4 : wr[1] > 3 ? 2 : 1), xname: 'w₁', yname: 'w₂', grid: true });
      const F = new S.Frames(['epoch', 'wrong']);
      const head = () => [...ax.items, H.title('lb:t', 46, 14, 'P(' + D.classes[1] + ') and the boundary'), ...lax.items, H.title('ll:t', 500, 14, 'Loss'), ...wax.items, H.title('lw:t', 500, 236, 'Weights (standardised inputs)')];
      const meta0 = { n, lr: inp.lr, E: inp.epochs };
      F.add({ items: [...head(), ...SC().points(ax, 'lb:p', X, y, { r: 4.2 })], say: n + ' points of two classes. Gradient descent starts from w₁ = w₂ = 0 and b = 0, where every point gets p = 0.5 and the mean log loss is ' + n$(L0) + '.', meta: Object.assign({ phase: 'data', L0 }, meta0) });
      let lastWrong = null;
      for (const e of pickSteps(inp.epochs)) {
        const s = steps[e];
        if (!s) continue;
        const th = s.theta, o = orig(th), g = grad(th), pf = (u, v) => sig(o.w[0] * u + o.w[1] * v + o.b);
        const wrong = X.map((p, i) => ((pf(p[0], p[1]) >= 0.5 ? 1 : 0) !== y[i]));
        const nw = wrong.filter(Boolean).length;
        const items = [...head(), SC().field(ax, 'lb:f', pf, { n: 70 })];
        if (Math.hypot(o.w[0], o.w[1]) > 1e-9) items.push(SC().level(ax, 'lb:b', SC().grid(ax, pf, 60), 0.5, ''));
        items.push(...SC().points(ax, 'lb:p', X, y, { r: 4.2, cls: (i) => (wrong[i] && e > 0 ? 'wrong' : '') }));
        items.push(C().line(lax, 'll:c', steps.slice(0, e + 1).map((q) => [q.iter, q.loss]), 'act'), C().dot(lax, 'll:d', e, s.loss, ''));
        const path = ws.slice(0, e + 1).map((t) => [t[0], t[1]]);
        if (path.length > 1) items.push({ key: 'lw:path', kind: 'path', d: path.map((p, i) => (i ? 'L' : 'M') + wax.X(p[0]) + ',' + wax.Y(p[1])).join(' '), cls: 'pl-trail act', pathAudit: 'edge-x', layer: 'lines' });
        items.push({ key: 'lw:now', kind: 'dot', x: wax.X(th[0]), y: wax.Y(th[1]), r: 5, cls: 'pl-pt act', audit: 'overlap-ok', layer: 'items' });
        const step = [-inp.lr * g[0], -inp.lr * g[1]], a0 = [wax.X(th[0]), wax.Y(th[1])], a1 = [wax.X(th[0] + step[0]), wax.Y(th[1] + step[1])];
        if (e < inp.epochs && Math.hypot(a1[0] - a0[0], a1[1] - a0[1]) > 7) items.push(H.arrow('lw:g', a0[0], a0[1], a1[0], a1[1], 'hot', { head: 8 }));
        F.set('epoch', e).set('wrong', nw);
        const gn = Math.hypot(...g);
        let say;
        if (e === 0) say = 'Epoch 0: the gradient is (' + n$(g[0]) + ', ' + n$(g[1]) + ') for the two weights and ' + n$(g[2]) + ' for b. The red arrow at the right is the first step, η = ' + n$(inp.lr) + ' times minus the gradient.';
        else say = 'Epoch ' + e + ': mean log loss ' + n$(s.loss) + ', ' + nw + ' of ' + n + ' points on the wrong side. Weights ' + n$(th[0]) + ' and ' + n$(th[1]) + ' (standardised), gradient size ' + n$(gn) + '.';
        if (e > 1 && lastWrong !== null && nw < lastWrong) say += ' The line has turned to take in ' + (lastWrong - nw) + ' more.';
        F.add({ items, say, meta: Object.assign({ phase: 'step', e, loss: s.loss, w: th.slice(), g, gn, wrong: nw, wo: o.w, bo: o.b, gained: lastWrong === null ? 0 : lastWrong - nw }, meta0) });
        lastWrong = nw;
      }
      const last = steps[steps.length - 1], acc = 1 - lastWrong / n;
      const verdict = inp.data === 'moons' || inp.data === 'xor' ? ' The boundary of logistic regression is a straight line, so ' + (inp.data === 'xor' ? (acc < 0.6 ? 'on XOR it stays near guessing: the four corners balance, so the log loss is lowest with the weights near 0. Even the best line gets only about 75%, by cutting off one corner.' : 'on XOR even the best line gets only about 75%, by cutting off one corner.') : 'it cannot follow the curve of the moons; curved features (x², x₁x₂, …) would let it.') : ' The remaining mistakes sit where the classes overlap.';
      F.add({ items: F.last().items.filter((it) => it.key !== 'lw:g'), say: 'After ' + inp.epochs + ' epochs: mean log loss ' + n$(last.loss) + ', ' + Math.round(acc * 100) + '% of points on their own side.' + verdict, meta: Object.assign({ phase: 'end', loss: last.loss, acc }, inp.data === 'xor' ? { bestLine: 0.75 } : {}, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Logistic regression in 2-D', text: ['z = w₁x₁ + w₂x₂ + b,  p = σ(z)', 'boundary: z = 0, a straight line', '∇L = mean of (p − y)·(x₁, x₂, 1)', 'step: θ ← θ − η ∇L', 'the weight vector (w₁, w₂) points', '  across the line, toward class 1'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'What shape will the final decision boundary be?', options: ['a straight line', 'a circle', 'whatever shape the data has', 'a parabola'], answer: 'a straight line', explain: 'z = 0 is linear in x₁ and x₂.' },
    ],
    quiz: [
      { q: 'Why is the boundary of logistic regression straight?', options: ['p = 0.5 exactly where w₁x₁ + w₂x₂ + b = 0, a linear equation', 'the sigmoid is straight', 'gradient descent forces it', 'it is not'], answer: 0, why: 'Curved boundaries need curved features.' },
      { q: 'Which direction does the vector (w₁, w₂) point?', options: ['across the boundary, toward the class labelled 1', 'along the boundary', 'toward class 0', 'anywhere'], answer: 0, why: 'z grows fastest along w.' },
      { q: 'A point far on the wrong side contributes to the gradient…', options: ['about its own x times 1 (p − y is near ±1)', 'nothing', 'a tiny amount', 'only to b'], answer: 0, why: 'Its error p − y is large, so it pulls hard.' },
      { q: 'Why standardise the inputs before gradient descent?', options: ['so one learning rate suits both weights', 'to change the boundary', 'to make the classes separable', 'it is required by the sigmoid'], answer: 0, why: 'Unequal scales stretch the loss surface into a ravine.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
