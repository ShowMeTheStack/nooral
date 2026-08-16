// Part 3, softmax regression for several classes. Each class k has its own score zₖ = wₖ·x + bₖ; softmax turns the
// scores into probabilities that add up to 1, and the predicted class is the largest. The regions where each class
// wins are convex and meet along straight lines. Trained by gradient descent on the mean cross-entropy
// (ML.LogisticRegression with three or four classes, solver 'gd'), on standardised inputs.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n, C = () => S.Kit.curve, SC = () => S.Kit.scatter;

  H.multi = (kind, n, seed) => {
    if (kind === 'flowers') { const t = S.Data.load('wildflowers', { seed, n }); return { X: t.X.map((r) => [r[0], r[1]]), y: t.y.slice(), names: ['petal length (cm)', 'petal width (cm)'], classes: t.classes }; }
    const four = kind === 'four', centers = four ? [[-2, -2], [2, -2], [2, 2], [-2, 2]] : [[-2.2, -1.2], [2.2, -1.2], [0, 2.4]];
    const t = S.Data.blobs({ n, centers, std: kind === 'tight' ? 0.7 : 1.15, seed });
    return { X: t.X, y: t.y, names: ['x₁', 'x₂'], classes: centers.map((_, k) => 'class ' + k) };
  };
  const pickSteps = (E) => { const out = new Set([0, 1, 2, 4]); let e = 4; while (e < E) { e = Math.ceil(e * 1.6); if (e < E) out.add(e); } out.add(E); return [...out].filter((v) => v <= E).sort((a, b) => a - b); };

  S.topic({
    id: 'softmax', family: 'p03-logit', order: 30, title: 'Softmax regression: many classes',
    blurb: 'For K classes keep K linear scores zₖ = wₖ·x + bₖ, one per class. Softmax turns them into probabilities, pₖ = e^(zₖ) / Σⱼ e^(zⱼ): all positive, adding up to 1, the largest score getting the largest share. The loss is the cross-entropy, −ln of the probability of the true class. Each region where one class wins is bounded by straight lines, where two scores tie.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'data', def: 'three', options: [{ value: 'three', label: 'three clouds' }, { value: 'four', label: 'four clouds' }, { value: 'flowers', label: 'meadow wildflowers' }] },
      { id: 'n', kind: 'number', label: 'points', def: 120, min: 15, max: 240 },
      { id: 'lr', kind: 'slider', label: 'learning rate η', def: 0.5, min: 0.05, max: 3, step: 0.05 },
      { id: 'epochs', kind: 'number', label: 'epochs', def: 120, min: 5, max: 600 },
      { id: 'seed', kind: 'number', label: 'seed', def: 2, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Three clouds', values: { data: 'three', n: 120, lr: 0.5, epochs: 120, seed: 2 } },
      { label: 'Four clouds in the corners', values: { data: 'four', n: 160, lr: 0.5, epochs: 150, seed: 3 } },
      { label: 'Meadow wildflowers', values: { data: 'flowers', n: 150, lr: 0.5, epochs: 200, seed: 1 } },
      { label: 'A slow learning rate', values: { data: 'three', n: 120, lr: 0.05, epochs: 120, seed: 2 } },
    ],
    counters: [{ id: 'epoch', label: 'epoch' }],
    run(inp) {
      const D = H.multi(inp.data, inp.n, inp.seed), X = D.X, y = D.y, n = X.length, K = D.classes.length;
      const st = H.standardise(X), mu = st.mean, sd = st.sd, U = st.X;
      const m = ML().LogisticRegression({ solver: 'gd', penalty: 'none', lr: inp.lr, epochs: inp.epochs, traceEvery: 1 }).fit(U, y);
      const steps = m.trace.filter((s) => s.kind === 'gd');
      const scores = (th, x) => { const u = [(x[0] - mu[0]) / sd[0], (x[1] - mu[1]) / sd[1]]; return ML().range(K).map((k) => th[k * 3] * u[0] + th[k * 3 + 1] * u[1] + th[k * 3 + 2]); };
      const probe = ML().colMeans(X).map((v) => +v.toFixed(2));
      const xr = H.nice(X.map((r) => r[0]), 0.06), yr = H.nice(X.map((r) => r[1]), 0.06);
      const ax = H.axes({ prefix: 'sm', x: 46, y: 40, w: 400, h: 400, xr, yr, xname: D.names[0], yname: D.names[1], grid: false });
      const L0 = steps[0].loss, lhi = Math.ceil(L0 * 5) / 5;
      const lax = H.axes({ prefix: 'sl', x: 530, y: 40, w: 220, h: 120, xr: [0, inp.epochs], yr: [0, lhi], xticks: C().ticks(0, inp.epochs, 3), yticks: lhi > 1 ? 0.5 : 0.25, xname: 'epoch', yname: 'cross-entropy' });
      const F = new S.Frames(['epoch']);
      const legend = H.classLegend('sm:lg', 46, 482, D.classes);
      const head = () => [...ax.items, H.title('sm:t', 46, 14, 'Where each class wins'), ...lax.items, H.title('sl:t', 500, 14, 'Loss'), H.title('sp:t', 500, 236, 'At the probe point'), ...legend];
      // the probe's scores and probabilities, as two rows of bars
      const probeBars = (z, p) => {
        const items = [], x0 = 560, w = 150, top = 262, rowH = 24, zmax = Math.max(1, ...z.map(Math.abs));
        items.push(H.text('sp:zl', x0 - 50, top + 10, 'score z', 'tick', 12, 'start'));
        z.forEach((v, k) => {
          const cy = top + 22 + k * rowH, mid = x0 + w / 2, bw = (Math.abs(v) / zmax) * (w / 2 - 4);
          if (k === 0) items.push({ key: 'sp:z0line', kind: 'path', d: `M${mid},${top + 18} v${K * rowH}`, cls: 'axis', audit: 'ignore', layer: 'back' });
          if (bw > 0.5) items.push({ key: 'sp:z' + k, kind: 'cell', x: H.r2(v >= 0 ? mid : mid - bw), y: cy, w: H.r2(bw), h: rowH - 8, rx: 2, text: '', cls: 'pl-pbar c' + k, audit: 'ignore' });
          items.push(H.text('sp:zv' + k, x0 + w + 6, cy + 12, n$(v), 'tick', 12));
        });
        const ptop = top + 34 + K * rowH;
        items.push(H.text('sp:pl', x0 - 50, ptop + 10, 'softmax p', 'tick', 12, 'start'));
        p.forEach((v, k) => {
          const cy = ptop + 22 + k * rowH, bw = v * w;
          items.push({ key: 'sp:pb' + k, kind: 'cell', x: x0, y: cy, w: w, h: rowH - 8, rx: 2, text: '', cls: 'pl-pbar soft', audit: 'ignore', layer: 'back' });
          if (bw > 0.5) items.push({ key: 'sp:p' + k, kind: 'cell', x: x0, y: cy, w: H.r2(bw), h: rowH - 8, rx: 2, text: '', cls: 'pl-pbar c' + k, audit: 'ignore' });
          items.push(H.text('sp:pv' + k, x0 + w + 6, cy + 12, n$(v), 'tick', 12));
        });
        return items;
      };
      const meta0 = { n, K, lr: inp.lr, E: inp.epochs, probe };
      F.add({ items: [...head(), ...SC().points(ax, 'sm:p', X, y, { r: 4 }), { key: 'sm:probe', kind: 'dot', x: ax.X(probe[0]), y: ax.Y(probe[1]), r: 7, cls: 'pl-probe', audit: 'overlap-ok', layer: 'items', z: 5 }, ...probeBars(new Array(K).fill(0), new Array(K).fill(1 / K))], say: n + ' points of ' + K + ' classes. All weights start at 0, so every score is 0 and softmax gives each class 1/' + K + ' = ' + n$(1 / K) + ' everywhere: the cross-entropy is ln ' + K + ' = ' + n$(L0) + '. The ringed probe point at (' + n$(probe[0]) + ', ' + n$(probe[1]) + ') is where we read the scores.', meta: Object.assign({ phase: 'data', L0, uniform: 1 / K }, meta0) });
      for (const e of pickSteps(inp.epochs)) {
        const s = steps[e];
        if (!s || e === 0) continue;
        const th = s.theta;
        const win = (u, v) => { const z = scores(th, [u, v]), p = ML().softmax(z), k = ML().argmax(p); return [k, Math.min(1, (p[k] - 1 / K) / (1 - 1 / K) * 1.1)]; };
        const pred = X.map((x) => ML().argmax(scores(th, x))), acc = pred.filter((k, i) => k === y[i]).length / n;
        const zP = scores(th, probe), pP = ML().softmax(zP), kP = ML().argmax(pP);
        const items = [...head(), SC().field(ax, 'sm:f', win, { n: 80, scheme: 'classes', smooth: false })];
        items.push(...SC().points(ax, 'sm:p', X, y, { r: 4, cls: (i) => (pred[i] !== y[i] ? 'wrong' : '') }));
        items.push({ key: 'sm:probe', kind: 'dot', x: ax.X(probe[0]), y: ax.Y(probe[1]), r: 7, cls: 'pl-probe', audit: 'overlap-ok', layer: 'items', z: 5 });
        items.push(C().line(lax, 'sl:c', steps.slice(0, e + 1).map((q) => [q.iter, q.loss]), 'act'), C().dot(lax, 'sl:d', e, s.loss, ''));
        items.push(...probeBars(zP, pP));
        F.set('epoch', e);
        const say = 'Epoch ' + e + ': cross-entropy ' + n$(s.loss) + ', ' + Math.round(acc * 100) + '% of points in their own class\'s region. At the probe the scores are ' + H.list(zP.map((v) => n$(v))) + ', and softmax gives ' + H.list(pP.map((v) => n$(v))) + ': ' + D.classes[kP] + ' wins there.';
        F.add({ items, say, meta: Object.assign({ phase: 'step', e, loss: s.loss, acc, zP, pP, kP, sumP: ML().sum(pP) }, meta0) });
      }
      const lastS = steps[steps.length - 1];
      F.add({ items: F.last().items, say: 'After ' + inp.epochs + ' epochs the cross-entropy is ' + n$(lastS.loss) + '. Every region is convex, and each border is a straight line where two scores are equal; the shading fades where the winning probability is small. Adding the same number to every score changes nothing, so only differences between the classes\' scores matter.', meta: Object.assign({ phase: 'end', loss: lastS.loss }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Softmax regression', text: ['zₖ = wₖ·x + bₖ  (k = 1 … K)', 'pₖ = e^(zₖ) / Σⱼ e^(zⱼ)', 'loss: −ln p_(true class)', '∂L/∂zₖ = pₖ − [k is the true class]', 'border of classes j and k: zⱼ = zₖ, a line', 'K = 2 is logistic regression again'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'What shape will each class\'s region have at the end?', options: ['convex, bounded by straight lines', 'any shape at all', 'a circle around its points', 'a strip'], answer: 'convex, bounded by straight lines', explain: 'Each region is where one linear score beats all the others: an intersection of half-planes.' },
    ],
    quiz: [
      { q: 'Softmax of the scores (2, 2, 2) is…', options: ['(1/3, 1/3, 1/3)', '(2, 2, 2)', '(1, 0, 0)', '(0.5, 0.5, 0.5)'], answer: 0, why: 'Equal scores, equal shares.' },
      { q: 'Adding 10 to every score…', options: ['changes no probability', 'raises every probability', 'favours the largest score', 'breaks softmax'], answer: 0, why: 'e^(z+10) / Σ e^(z+10) = e^z / Σ e^z.' },
      { q: 'With K = 2 classes, softmax regression is…', options: ['logistic regression', 'linear regression', 'k-means', 'a perceptron'], answer: 0, why: 'p₁ = σ(z₁ − z₀).' },
      { q: 'The gradient of the cross-entropy with respect to the score zₖ is…', options: ['pₖ minus 1 for the true class, pₖ for the others', 'pₖ(1 − pₖ)', 'always negative', 'zₖ'], answer: 0, why: 'The same p − y error as in the binary case, one entry per class.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
