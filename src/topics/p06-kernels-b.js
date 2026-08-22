// Part 6, the RBF kernel: K(x, x′) = exp(−γ‖x − x′‖²) gives every support vector a bump of influence whose reach
// is about 1/√γ. A small γ makes broad, smooth bumps; a large γ makes narrow ones that wrap each training point,
// and the accuracy on points held back falls while the training accuracy keeps climbing.
(function (S) {
  'use strict';
  const P = S.P06, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const GAMMAS = [0.1, 0.3, 1, 3, 10, 30, 100];

  S.topic({
    id: 'svm-rbf', family: 'p06-kernels', order: 30, title: 'The RBF kernel and γ against overfitting',
    blurb: 'The RBF (Gaussian) kernel K(x, x′) = exp(−γ‖x − x′‖²) measures closeness: 1 for the same point, falling to 1/e at a distance 1/√γ. The SVM’s decision value is a sum of these bumps, one per support vector. Small γ: wide bumps and a smooth boundary that may underfit. Large γ: narrow bumps round each training point, a boundary that memorises them, and worse accuracy on points it has not seen.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'moons', options: P.setOptions(['moons', 'circles', 'xor']) },
      { id: 'C', kind: 'slider', label: 'C', def: 1, min: 0.1, max: 100, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Moons, C = 1', values: { set: 'moons', C: 1, seed: 1 } },
      { label: 'Moons, C = 100', values: { set: 'moons', C: 100, seed: 2 } },
      { label: 'A ring round a disc', values: { set: 'circles', C: 1, seed: 3 } },
      { label: 'XOR corners', values: { set: 'xor', C: 10, seed: 1 } },
    ],
    counters: [{ id: 'sv', label: 'support vectors' }],
    run(inp) {
      const noise = { moons: 0.28, circles: 0.14, xor: 0 }[inp.set];
      const d = inp.set === 'moons' ? S.Data.moons({ n: 100, noise, seed: inp.seed }) : inp.set === 'circles' ? S.Data.circles({ n: 100, noise, factor: 0.5, seed: inp.seed }) : S.Data.xor({ n: 100, noise: 0.08, seed: inp.seed });
      const sp = S.Data.split(d.y, 0.3, inp.seed), Xtr = sp.train.map((i) => d.X[i]), ytr = sp.train.map((i) => d.y[i]), Xte = sp.test.map((i) => d.X[i]), yte = sp.test.map((i) => d.y[i]);
      const Cv = +H().sg(inp.C, 2), F = H().frames(['sv']), q = F.q;
      const fits = GAMMAS.map((g) => { const f = P.fit(Xtr, ytr, { kernel: 'rbf', gamma: g, C: Cv }, Xte); f.test = ML().accuracy(yte, Xte.map((x) => (f.f(x) > 0 ? 1 : 0))); return f; });
      const ax = P.axes('rb', d.X, { x: 40, y: 24, w: 400, h: 400 }, { pad: 0.25 });
      const lg = (g) => Math.log10(g);
      const ac = C().axes({ prefix: 'ra', x: 540, y: 50, w: 240, h: 170, xr: [-1.2, 2.2], yr: [0.4, 1], xticks: [-1, 0, 1, 2], fmtX: (v) => ({ '-1': '0.1', 0: '1', 1: '10', 2: '100' }[v]), yticks: [0.4, 0.6, 0.8, 1], fmtY: (v) => Math.round(v * 100) + '%', xname: 'γ (log scale)', yname: 'accuracy' });
      const side = (k) => {
        const items = [...ac.items];
        const tr = GAMMAS.slice(0, k + 1).map((g, j) => [lg(g), fits[j].acc]), te = GAMMAS.slice(0, k + 1).map((g, j) => [lg(g), fits[j].test]);
        if (k > 0) items.push(C().line(ac, 'ra:tr', tr, 'act'), C().line(ac, 'ra:te', te, 'c0'));
        tr.forEach((p, j) => items.push(C().dot(ac, 'ra:a' + j, p[0], p[1], '', j === k ? 5.5 : 3.5)));
        te.forEach((p, j) => items.push(C().dot(ac, 'ra:b' + j, p[0], p[1], 'c0', j === k ? 5.5 : 3.5)));
        items.push(...C().legend('ra:lg', 540, 285, [{ text: 'training points', cls: 'act' }, { text: 'held-back points', cls: 'c0' }], { dir: 'col', size: 13 }));
        return items;
      };
      // the data: training points filled, held-back points hollow
      const pts = (fit) => {
        const items = S.Kit.scatter.points(ax, 'rb:p', Xtr, ytr, { r: 4.4 });
        items.push(...S.Kit.scatter.points(ax, 'rb:t', Xte, yte, { r: 4.4, cls: () => 'test' }));
        if (fit) for (const i of fit.sv) items.push(S.Kit.scatter.ring(ax, 'rb:r' + i, Xtr[i][0], Xtr[i][1], fit.alpha[i] < fit.C * (1 - 1e-6) ? 'act' : '', 8));
        return items;
      };
      // the reach of one support vector: the circle where its kernel falls to 1/e
      const reach = (fit, g) => {
        const i = fit.free.length ? fit.free[0] : fit.sv[0], r = 1 / Math.sqrt(g), c = Xtr[i], sp2 = [];
        for (let k = 0; k <= 60; k++) { const t = (2 * Math.PI * k) / 60; sp2.push([ax.X(c[0] + r * Math.cos(t)), ax.Y(c[1] + r * Math.sin(t))]); }
        const inside = sp2.filter((p) => p[0] >= ax.o.x && p[0] <= ax.o.x + ax.o.w && p[1] >= ax.o.y && p[1] <= ax.o.y + ax.o.h);
        if (inside.length < sp2.length * 0.5) return [];
        return [{ key: 'rb:reach', kind: 'path', d: sp2.map((p, k) => (k ? 'L' : 'M') + H().r2(p[0]) + ',' + H().r2(p[1])).join(' ') + ' Z', cls: 'mlinfl', audit: 'ignore', layer: 'items', z: 5 }];
      };
      F.push([...ax.items, ...pts(null), ...side(-1)], `${q.n(Xtr.length)} training points (filled) and ${q.n(Xte.length)} held back (hollow) that the SVM never sees. We fit the RBF kernel with C = ${q.sg(Cv, 2)} for γ from ${q.sg(GAMMAS[0], 1)} to ${q.sg(GAMMAS[GAMMAS.length - 1], 1)}, and score both groups.`, { phase: 'data', ntr: Xtr.length, nte: Xte.length, C: Cv, gammas: GAMMAS });
      GAMMAS.forEach((g, k) => {
        const fit = fits[k];
        F.set('sv', fit.sv.length);
        let say = `γ = ${q.sg(g, 2)}: each support vector’s bump reaches about 1/√γ = ${q(1 / Math.sqrt(g))} (the yellow circle shows one). ${q.n(fit.sv.length)} support vectors; training accuracy ${q.pct(fit.acc)}, held-back accuracy ${q.pct(fit.test)}.`;
        if (k === 0) say += ' The bumps are so wide that the boundary is nearly straight.';
        else if (k === GAMMAS.length - 1) say += ' The bumps are tiny: the boundary wraps the training points one by one, and nearly every point has become a support vector.';
        F.push([...ax.items, ...P.boundary(ax, 'rb', fit, { n: 80 }), ...reach(fit, g), ...pts(fit), ...side(k)], say, { phase: 'gamma', gamma: g, reach: 1 / Math.sqrt(g), nsv: fit.sv.length, acc: fit.acc, test: fit.test, k });
      });
      const best = fits.reduce((b, f, k) => (f.test > fits[b].test ? k : b), 0), last = fits.length - 1;
      const tied = fits.map((f, k) => k).filter((k) => Math.abs(fits[k].test - fits[best].test) < 1e-9);
      F.set('sv', fits[best].sv.length);
      F.push([...ax.items, ...P.boundary(ax, 'rb', fits[best], { n: 80 }), ...pts(fits[best]), ...side(last), C().vline(ac, 'ra:best', lg(GAMMAS[best]), 'act')], (tied.length > 1 ? `On the held-back points ${tied.every((k, j) => j === 0 || k === tied[j - 1] + 1) ? 'γ = ' + q.sg(GAMMAS[tied[0]], 2) + ' to ' + q.sg(GAMMAS[tied[tied.length - 1]], 2) : 'γ = ' + tied.map((k) => q.sg(GAMMAS[k], 2)).join(', ')} tie for best, at ${q.pct(fits[best].test)}; the picture shows γ = ${q.sg(GAMMAS[best], 2)}.` : `The held-back points prefer γ = ${q.sg(GAMMAS[best], 2)}, at ${q.pct(fits[best].test)}.`) + ` At γ = ${q.sg(GAMMAS[last], 2)} the training accuracy is ${q.pct(fits[last].acc)} but the held-back accuracy only ${q.pct(fits[last].test)}${fits[last].test < fits[best].test ? ': the boundary learned the training points, not the shape behind them. That is overfitting, and γ (with C) is chosen by validation' : ''}.`, { phase: 'best', gamma: GAMMAS[best], test: fits[best].test, lastGamma: GAMMAS[last], lastAcc: fits[last].acc, lastTest: fits[last].test, tied: tied.map((k) => GAMMAS[k]) });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 450, h: 470 };
      return { frames: F.list, result: { test: yte }, def: { sections: [{ id: 'eq', kind: 'math', title: 'The RBF kernel', text: ['K(x, x′) = exp(−γ‖x − x′‖²)', 'K = 1/e when ‖x − x′‖ = 1/√γ', 'f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b, over the support vectors', 'small γ: smooth, may underfit', 'large γ: wiggly, overfits'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'gamma' && f.meta.k === 3, pick: 'option', ask: (fr, i) => 'Raise γ from ' + H().sg(fr[i].meta.gamma, 2) + ' to ' + H().sg(fr[i + 1].meta.gamma, 2) + '. Will the training accuracy go down?', options: ['no: it stays or rises', 'yes: it falls'], answer: (fr, i) => (fr[i + 1].meta.acc >= fr[i].meta.acc ? 'no: it stays or rises' : 'yes: it falls'), explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => f.meta.phase === 'gamma' && f.meta.k === 5, pick: 'option', ask: (fr, i) => 'At γ = ' + H().sg(fr[i + 1].meta.gamma, 2) + ', how will the number of support vectors compare with γ = ' + H().sg(fr[i].meta.gamma, 2) + '?', options: ['more or the same', 'fewer'], answer: (fr, i) => (fr[i + 1].meta.nsv >= fr[i].meta.nsv ? 'more or the same' : 'fewer'), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'With γ = 4, at what distance does the RBF kernel fall to 1/e?', options: ['0.5', '2', '0.25', '4'], answer: 0, why: '1/√4 = 0.5.' },
      { q: 'What is K(x, x) for the RBF kernel?', options: ['1', '0', 'γ', '‖x‖²'], answer: 0, why: 'exp(−γ·0) = 1.' },
      { q: 'A large γ with near-perfect training accuracy but poor held-back accuracy is…', options: ['overfitting', 'underfitting', 'a hard margin', 'the best setting'], answer: 0, why: 'The narrow bumps memorise the training points.' },
      { q: 'How should γ and C be chosen?', options: ['by validation, for example cross-validation over a grid', 'as large as possible', 'γ = 1 and C = 1 always', 'by training accuracy'], answer: 0, why: 'Their best values depend on the data and must be judged on points the model has not trained on.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
