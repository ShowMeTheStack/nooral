// Part 8, better optimisers (2): all six optimisers raced on the same surface from the same start, each with the
// learning rate that suits it best on that surface (picked from a grid by its mean log loss over the run), with their paths
// over the contour map and their distance above the lowest point on a log scale.
(function (S) {
  'use strict';
  const P = S.P08, H = () => S.Kit.ml;
  const boxFor = (surf, x, y, w, hmax) => { const r = (surf.yr[1] - surf.yr[0]) / (surf.xr[1] - surf.xr[0]); const h = Math.min(hmax, w * r); return { x, y, w: h / r, h }; };
  /** the grid of learning rates each optimiser is tried with */
  P.LR_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8];
  P.OPT_OPTS = { sgd: {}, momentum: { momentum: 0.9 }, nesterov: { momentum: 0.9 }, adagrad: {}, rmsprop: {}, adam: {} };
  /** a run's score: the mean over its steps of log10 of the loss above the lowest point (floored at 10⁻¹²) */
  P.score = (loss, Lstar) => loss.reduce((s, v) => s + Math.log10(Math.max(1e-12, v - Lstar)), 0) / loss.length;
  /** an optimiser's best rate on a surface: the grid value with the lowest score (diverging runs lose) */
  P.bestRate = function (surf, id, N, start, Lstar) {
    let best = null;
    for (const lr of P.LR_GRID) {
      const r = P.run(surf, id, Object.assign({ lr }, P.OPT_OPTS[id]), N, { start });
      const wild = r.pts.some((p) => p[0] < surf.xr[0] - 1 || p[0] > surf.xr[1] + 1 || p[1] < surf.yr[0] - 1 || p[1] > surf.yr[1] + 1);
      const L = r.diverged || wild || !Number.isFinite(r.loss[r.loss.length - 1]) ? Infinity : P.score(r.loss, Lstar);
      if (!best || L < best.L) best = { lr, L, r };
    }
    return best;
  };
  const STARTS = { ravine: [-4, 1.6], saddle: [-2, 0.02], rosen: [-1.5, 2.5], bumpy: [2.4, -0.4], plateau: [3.6, 0.9] };

  S.topic({
    id: 'opt-race', family: 'p08-optimisers', order: 30, title: 'Six optimisers raced on the same surface',
    blurb: 'Gradient descent, momentum, Nesterov, AdaGrad, RMSProp and Adam set off from the same point. Each runs with the learning rate that suits it best on this surface, chosen from the same grid of rates as the one that keeps its loss lowest on average over the run, so the race compares the methods and not a lucky setting. Ravines reward momentum and adaptive steps, a saddle tests who escapes a flat point, and the banana valley punishes everyone.',
    inputs: [
      { id: 'surf', kind: 'choice', label: 'Surface', def: 'ravine', options: P.surfOptions(['ravine', 'saddle', 'rosen', 'bumpy', 'plateau']) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 100, min: 20, max: 200 },
    ],
    presets: [
      { label: 'A ravine', values: { surf: 'ravine', steps: 100 } },
      { label: 'A saddle', values: { surf: 'saddle', steps: 80 } },
      { label: 'Rosenbrock’s banana', values: { surf: 'rosen', steps: 200 } },
      { label: 'Bumpy: which pit?', values: { surf: 'bumpy', steps: 100 } },
      { label: 'A plateau', values: { surf: 'plateau', steps: 100 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF[inp.surf], N = inp.steps, F = H().frames(['step']), q = F.q, start = STARTS[inp.surf];
      const m = P.globalMin(surf), Lstar = surf.f(m[0], m[1]);
      const runs = P.OPT.map((o) => { const b = P.bestRate(surf, o.id, N, start, Lstar); return Object.assign({}, o, { lr: b.lr, score: b.L }, b.r); });
      const gap = (v) => Math.max(1e-12, v - Lstar);
      const bx = boxFor(surf, 40, 30, 430, 430), ax = P.axes('rc', surf, bx, { xticks: 1, yticks: 1 });
      let hi = 0, lo = Infinity;
      for (const r of runs) for (const v of r.loss) if (Number.isFinite(v) && v < 1e6) { hi = Math.max(hi, gap(v)); lo = Math.min(lo, gap(v)); }
      const lax = P.logAxes('rl', { x: 545, y: 50, w: 225, h: 150 }, N, Math.max(1e-10, lo), hi, { yname: 'loss above the lowest' });
      const AT = [...new Set([0, 3, 6, 10, 15, 20, 30, 40, 50, 65, 80, 100, 125, 150, 175, 200].filter((k) => k <= N).concat([N]))];
      AT.forEach((k) => {
        F.set('step', k);
        const L = (r) => r.loss[Math.min(k, r.loss.length - 1)];
        const items = [...ax.items, ...P.contour(ax, 'rc:c', surf), P.minDot(ax, 'rc:m', m), P.start(ax, 'rc:s', start), ...lax.items];
        for (const r of runs) items.push(...P.path(ax, 'rc:' + r.id, r.pts, r.cls + ' thin', Math.min(k, r.pts.length - 1), { r: 5 }), P.logLine(lax, 'rl:' + r.id, r.loss.map(gap), r.cls, k));
        items.push(...P.legend('lg', 545, 262, runs.map((r) => ({ text: r.short + ' (η ' + H().sg(r.lr, 2) + '): ' + H().sg(L(r), 3), cls: r.cls })), { size: 13 }));
        const order = runs.slice().sort((a, b) => gap(L(a)) - gap(L(b)) || a.score - b.score);
        let say;
        if (k === 0) say = `All six start at (${q(start[0])}, ${q(start[1])}), loss ${q.sg(surf.f(start[0], start[1]), 3)}; the lowest point is at (${q(m[0])}, ${q(m[1])}), loss ${q.sg(Lstar, 3)}. Their rates, each the best of the grid over ${q.n(N)} steps here: ${runs.map((r) => r.short + ' ' + q.sg(r.lr, 2)).join(', ')}.`;
        else if (k < N) say = `Step ${q.n(k)}: ${order[0].name} leads with loss ${q.sg(L(order[0]), 3)}, then ${order[1].name} (${q.sg(L(order[1]), 3)}); ${order[5].name} trails at ${q.sg(L(order[5]), 3)}.`;
        else {
          const done = order.filter((r) => L(r) - Lstar < 1e-9), rest = order.filter((r) => !(L(r) - Lstar < 1e-9));
          if (done.length > 1) say = `After ${q.n(k)} steps, ${done.length === runs.length ? 'all six' : q.n(done.length)} have reached the bottom, within a billionth of it. Ranked by how low they kept the loss on the way (their mean log loss), the order is ${H().list(done.map((r) => r.name))}.` + (rest.length ? ` Still above it: ${H().list(rest.map((r) => r.name + ' by ' + q.sg(L(r) - Lstar, 2)))}.` : '');
          else say = `After ${q.n(k)} steps the order is ${H().list(order.map((r) => r.name + ' ' + q.sg(L(r), 3)))}.`;
          const far = runs.filter((r) => P.dist(r.pts[r.pts.length - 1], m) > 0.3);
          if (far.length && inp.surf === 'bumpy') say += ` ${far.length === 1 ? 'One run' : q.n(far.length) + ' runs'} settled in another pit, not the deepest: a better optimiser does not mean a better minimum.`;
        }
        F.push(items, say, { phase: k === 0 ? 'start' : k === N ? 'end' : 'step', k, start, m, Lstar, lrs: runs.map((r) => r.lr), losses: runs.map(L), gaps: runs.map((r) => L(r) - Lstar), ndone: runs.filter((r) => L(r) - Lstar < 1e-9).length, leader: order[0].id, nfar: runs.filter((r) => P.dist(r.pts[r.pts.length - 1], m) > 0.3).length });
      });
      // on a phone the whole picture, so the legend that names the colours stays in view
      return { frames: F.list, result: { runs }, def: { sections: [{ id: 'eq', kind: 'math', title: 'The race', text: [surf.formula, 'each optimiser tried with η in {0.001, 0.002, … 0.8}', 'its rate: the lowest mean of log₁₀(loss above the lowest point) over the run', 'a rate whose run strays well off the map is ruled out', 'momentum and Nesterov: β = 0.9', 'RMSProp: ρ = 0.99; Adam: β₁ = 0.9, β₂ = 0.999'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which optimiser will be lowest at the end?', options: P.OPT.map((o) => o.name), answer: (fr) => P.opt(fr[fr.length - 1].meta.leader).name, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Why give each optimiser its own best learning rate before racing them?', options: ['otherwise the race compares one lucky setting, not the methods', 'all optimisers need the same rate', 'the rate does not matter', 'to make the race longer'], answer: 0, why: 'A rate that suits one method can cripple another; tuning each is the fair comparison.' },
      { q: 'Which kind of surface most rewards momentum?', options: ['a long, gently sloping ravine', 'a perfectly round bowl', 'a flat plane', 'a single sharp spike'], answer: 0, why: 'Consistent gradients along the floor build up speed while the cross-ravine swings cancel.' },
      { q: 'On a bumpy surface, the fastest optimiser…', options: ['may still settle in a worse pit than a slower one', 'always finds the deepest pit', 'never stops', 'always ends where it started'], answer: 0, why: 'Speed is about the descent, not about which basin it falls into.' },
      { q: 'Why do adaptive methods escape a flat stretch quickly?', options: ['dividing by the small recent gradients keeps their steps near η', 'they ignore the gradient', 'they have no learning rate', 'they jump randomly'], answer: 0, why: 'A tiny gradient over its own tiny size is still about one.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
