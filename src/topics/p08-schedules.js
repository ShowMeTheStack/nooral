// Part 8, learning-rate schedules: the four common schedules drawn as curves and each used for the same
// stochastic descent on the same batches, so only the schedule differs; and a warm-up, which starts the rate small
// and raises it, shown saving Adam from a wild start in the banana valley.
(function (S) {
  'use strict';
  const P = S.P08, H = () => S.Kit.ml;
  const T = () => S.T;

  /** the four schedules of a run of N steps from rate lr (functions of the step t, counted from 0) */
  P.schedules = (lr, N) => [
    { id: 'constant', name: 'constant', cls: 'o-gd', fn: T().sched.constant({ lr }), rule: 'η' },
    { id: 'step', name: 'step decay', cls: 'o-rms', fn: T().sched.step({ lr, every: Math.ceil(N / 3), gamma: 0.3 }), rule: 'η × 0.3 every ' + Math.ceil(N / 3) + ' steps' },
    { id: 'exp', name: 'exponential', cls: 'o-nest', fn: T().sched.exp({ lr, gamma: Math.pow(0.02, 1 / N) }), rule: 'η × γᵗ, falling to 2% at the end' },
    { id: 'cosine', name: 'cosine', cls: 'o-adam', fn: T().sched.cosine({ lr, min: 0, total: N }), rule: 'η × ½(1 + cos(πt/N))' },
  ];

  S.topic({
    id: 'opt-schedules', family: 'p08-schedules', order: 10, title: 'Learning-rate schedules',
    blurb: 'A large learning rate makes fast progress early but, with noisy gradients, keeps the parameters jittering round the minimum for ever. A schedule lowers the rate as training goes: in steps, by a constant factor every step, or along half a cosine. Here four runs of one-point stochastic descent fit the same line on the same sequence of points; only the schedule differs.',
    inputs: [
      { id: 'focus', kind: 'choice', label: 'Schedule', def: 'cosine', options: P.schedules(1, 10).map((s) => ({ value: s.id, label: s.name })) },
      { id: 'lr', kind: 'slider', label: 'Starting rate η', def: 0.3, min: 0.05, max: 0.6, step: 0.01, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 60, min: 15, max: 120 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 4, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Cosine', values: { focus: 'cosine', lr: 0.3, steps: 60, seed: 4 } },
      { label: 'Step decay', values: { focus: 'step', lr: 0.3, steps: 60, seed: 4 } },
      { label: 'Constant: never settles', values: { focus: 'constant', lr: 0.45, steps: 90, seed: 9 } },
      { label: 'Exponential, a long run', values: { focus: 'exp', lr: 0.3, steps: 120, seed: 2 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const N = inp.steps, lr = inp.lr, D = P.lsq(40, inp.seed), surf = D.surf, F = H().frames(['step']), q = F.q;
      const order = P.batches(40, 1, N, inp.seed * 11 + 5), Ls = surf.f(D.best[0], D.best[1]);
      const runs = P.schedules(lr, N).map((s) => Object.assign(s, P.run(surf, 'sgd', { lr }, N, { schedule: s.fn, grad: (p, k) => D.gb(p, order[k]) })));
      const me = runs.find((r) => r.id === inp.focus);
      const sax = S.Kit.curve.axes({ prefix: 'sc', x: 40, y: 40, w: 230, h: 140, xr: [0, N], yr: [0, lr * 1.1], xticks: S.Kit.curve.ticks(0, N, 4), yticks: S.Kit.curve.ticks(0, lr, 3), xname: 'step', yname: 'learning rate' });
      const gap = (v) => Math.max(1e-8, v - Ls);
      let lo = Infinity, hi = 0;
      for (const r of runs) for (const v of r.loss) { lo = Math.min(lo, gap(v)); hi = Math.max(hi, gap(v)); }
      const lax = P.logAxes('sl', { x: 40, y: 290, w: 230, h: 140 }, N, lo, hi, { yname: 'loss above the best' });
      const bx = { x: 340, y: 30, w: 430, h: 430 * ((surf.yr[1] - surf.yr[0]) / (surf.xr[1] - surf.xr[0])) }, ax = P.axes('sp', surf, bx, { xticks: 1, yticks: 1, xname: 'slope a', yname: 'intercept b' });
      const AT = [...new Set([0, 5, 10, 20, 30, 40, 50, 60, 75, 90, 105, 120].filter((k) => k <= N).concat([N]))];
      // the jitter: how far the focused run strays from the best fit over the last fifth of the run
      const tail = (r, k) => { const a = Math.max(1, Math.floor(k * 0.8)); let m = 0; for (let t = a; t <= k; t++) m = Math.max(m, P.dist(r.pts[t], D.best)); return m; };
      AT.forEach((k) => {
        F.set('step', k);
        const items = [...sax.items, ...lax.items, ...ax.items, ...P.contour(ax, 'sp:c', surf), P.minDot(ax, 'sp:m', D.best), P.start(ax, 'sp:s', surf.start)];
        for (const r of runs) {
          const foc = r.id === inp.focus;
          items.push(S.Kit.curve.line(sax, 'sc:' + r.id, [...Array(N + 1).keys()].map((t) => [t, r.fn(t)]), r.cls + (foc ? '' : ' thin')));
          items.push(P.logLine(lax, 'sl:' + r.id, r.loss.map(gap), r.cls + (foc ? '' : ' thin'), k));
          if (foc) items.push(...P.path(ax, 'sp:' + r.id, r.pts, r.cls, k, { r: 6 }));
        }
        if (k) items.push(S.Kit.curve.vline(sax, 'sc:now', Math.min(k, N), 'act'));
        items.push(...P.legend('lg', 340, bx.y + bx.h + 64, runs.map((r) => ({ text: r.name, cls: r.cls })), { dir: 'row', size: 12 }));
        const L = (r) => r.loss[k];
        let say, meta = { phase: k === 0 ? 'start' : k === N ? 'end' : 'step', k, lr, lrs: runs.map((r) => r.fn(Math.max(0, k - 1))), losses: runs.map(L), Ls };
        if (k === 0) say = `Four runs of one-point stochastic descent, all starting at rate η = ${q.sg(lr, 3)} and fed the same points in the same order. Only the rate’s schedule differs (top left). The path on the right is the ${me.name} run; the best fit has loss ${q.sg(Ls, 4)}.`;
        else if (k < N) say = `Step ${q.n(k)}: the rates used are ${H().list(runs.map((r) => r.name + ' ' + q.sg(r.fn(k - 1), 3)))}; the losses ${H().list(runs.map((r) => q.sg(L(r), 4)))}.`;
        else {
          const tj = runs.map((r) => tail(r, k));
          Object.assign(meta, { tail: tj });
          say = `After ${q.n(k)} steps the losses are ${H().list(runs.map((r) => r.name + ' ' + q.sg(L(r), 4)))}. Over the last fifth of the run the constant rate still strays up to ${q.sg(tj[0], 3)} from the best fit, the cosine run ${q.sg(tj[3], 3)}` + (tj[3] < tj[0] ? ': a falling rate turns the jitter into a settle.' : '. This time the cosine run was unlucky with its last few points: with so few steps at a small rate, one bad point still moves it.');
        }
        F.push(items, say, meta);
      });
      for (const f of F.list) f.narrowView = { x: 300, y: 0, w: 490, h: bx.y + bx.h + 80 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Schedules (t = 0, 1, … N − 1)', text: runs.map((r) => r.name + ': ' + r.rule).concat(['each step: θ ← θ − η(t) · g, g from one point']).join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which run will still be jittering most at the end?', options: ['constant', 'step decay', 'exponential', 'cosine'], answer: (fr) => ['constant', 'step decay', 'exponential', 'cosine'][fr[fr.length - 1].meta.tail.indexOf(Math.max(...fr[fr.length - 1].meta.tail))], explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'A step schedule starts at 0.1 and multiplies by 0.5 every 10 steps. What is the rate at step 25?', options: ['0.025', '0.05', '0.0125', '0.1'], answer: 0, why: 'Two halvings have happened by step 25: 0.1 × 0.25.' },
      { q: 'What is the cosine schedule’s rate halfway through, from η = 0.4 down to 0?', options: ['0.2', '0.4', '0', '0.1'], answer: 0, why: '½(1 + cos(π/2)) = ½, so 0.2.' },
      { q: 'Why does a constant rate leave stochastic descent jittering?', options: ['each step still follows one noisy gradient at full size', 'the loss is not convex', 'the gradient is zero', 'the rate is too small'], answer: 0, why: 'Near the bottom the noise dominates the true gradient; only a smaller rate quietens it.' },
      { q: 'What is the price of decaying the rate too fast?', options: ['training stalls before reaching the bottom', 'it diverges', 'the loss rises', 'nothing'], answer: 0, why: 'Tiny steps early on leave the parameters far from the minimum when the rate runs out.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'opt-warmup', family: 'p08-schedules', order: 20, title: 'Warm-up: starting the learning rate small',
    blurb: 'In its first steps an optimiser knows little: Adam’s moving averages hold one or two gradients, and its first step is about η in every coordinate whatever the slope. With a large η that can fling the parameters far up a wall before the averages settle. A warm-up raises the rate in a straight line from almost nothing to η over the first few steps, then keeps it.',
    inputs: [
      { id: 'surf', kind: 'choice', label: 'Surface', def: 'rosen', options: P.surfOptions(['rosen', 'ravine', 'tilted']) },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 0.5, min: 0.05, max: 1.5, step: 0.01, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'warm', kind: 'number', label: 'Warm-up steps', def: 15, min: 2, max: 40 },
      { id: 'steps', kind: 'number', label: 'Steps', def: 60, min: 10, max: 120 },
    ],
    check: (inp) => (inp.warm >= inp.steps ? 'The warm-up must end before the run does' : null),
    presets: [
      { label: 'Banana valley, η = 0.5', values: { surf: 'rosen', lr: 0.5, warm: 15, steps: 60 } },
      { label: 'Banana valley, η = 1.2', values: { surf: 'rosen', lr: 1.2, warm: 20, steps: 80 } },
      { label: 'A ravine: it hardly matters', values: { surf: 'ravine', lr: 0.5, warm: 15, steps: 60 } },
      { label: 'Short warm-up', values: { surf: 'rosen', lr: 0.5, warm: 3, steps: 60 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF[inp.surf], N = inp.steps, W = inp.warm, lr = inp.lr, F = H().frames(['step']), q = F.q;
      const sched = T().sched.warmup({ steps: W, lr });
      const runs = [
        Object.assign({ id: 'cold', name: 'no warm-up', cls: 'o-rms' }, P.run(surf, 'adam', { lr }, N)),
        Object.assign({ id: 'warm', name: 'warm-up over ' + W + ' steps', cls: 'o-adam' }, P.run(surf, 'adam', { lr }, N, { schedule: sched })),
      ];
      runs[0].fn = () => lr; runs[1].fn = sched;
      for (const r of runs) r.len = r.pts.slice(1).map((p, k) => P.dist(p, r.pts[k]));
      const sax = S.Kit.curve.axes({ prefix: 'wa', x: 40, y: 40, w: 230, h: 130, xr: [0, N], yr: [0, lr * 1.1], xticks: S.Kit.curve.ticks(0, N, 4), yticks: S.Kit.curve.ticks(0, lr, 3), xname: 'step', yname: 'learning rate' });
      const lmax = Math.max(...runs.flatMap((r) => r.len));
      const lax = S.Kit.curve.axes({ prefix: 'wl', x: 40, y: 280, w: 230, h: 130, xr: [0, N], yr: [0, lmax * 1.1], xticks: S.Kit.curve.ticks(0, N, 4), yticks: S.Kit.curve.ticks(0, lmax, 3), xname: 'step', yname: 'length of each step' });
      const bx = { x: 340, y: 30, w: 430, h: Math.min(400, 430 * ((surf.yr[1] - surf.yr[0]) / (surf.xr[1] - surf.xr[0]))) }, ax = P.axes('wp', surf, bx, { xticks: 1, yticks: 1 });
      const AT = [...new Set([0, 1, 2, 3, 5, 8, Math.round(W / 2), W, W + 5, Math.round((W + N) / 2), N].filter((k) => k <= N))].sort((a, b) => a - b);
      AT.forEach((k) => {
        F.set('step', k);
        const items = [...sax.items, ...lax.items, ...ax.items, ...P.contour(ax, 'wp:c', surf), P.minDot(ax, 'wp:m', surf.min[0]), P.start(ax, 'wp:s', surf.start)];
        for (const r of runs) {
          items.push(S.Kit.curve.line(sax, 'wa:' + r.id, [...Array(N + 1).keys()].map((t) => [t, r.fn(t)]), r.cls));
          if (k) items.push(S.Kit.curve.line(lax, 'wl:' + r.id, r.len.slice(0, k).map((v, t) => [t + 1, v]), r.cls));
          items.push(...P.path(ax, 'wp:' + r.id, r.pts, r.cls, Math.min(k, r.pts.length - 1), { r: 6 }));
        }
        if (k) items.push(S.Kit.curve.vline(sax, 'wa:now', k, 'act'));
        items.push(...P.legend('lg', 340, bx.y + bx.h + 64, runs.map((r) => ({ text: r.name, cls: r.cls })), { dir: 'row', size: 13 }));
        const L = (r) => r.loss[k], mx = (r) => Math.max(...r.loss.slice(0, k + 1));
        let say, meta = { phase: k === 0 ? 'start' : k === N ? 'end' : 'step', k, lr, W, losses: runs.map(L), rates: runs.map((r) => r.fn(Math.max(0, k - 1))), lens: k ? runs.map((r) => r.len[k - 1]) : null, maxes: runs.map(mx) };
        if (k === 0) say = `Adam with η = ${q.sg(lr, 3)}, twice from the same start: once at full rate from the first step, once warming up in a straight line over the first ${q.n(W)} steps.`;
        else if (k === 1) say = `Step 1: without warm-up Adam’s step is ${q.sg(runs[0].len[0], 3)} long, about η in each coordinate, though it has seen only one gradient. With warm-up the rate is ${q.sg(runs[1].fn(0), 3)} and the step ${q.sg(runs[1].len[0], 3)}.`;
        else if (k < N) say = `Step ${q.n(k)}: the rates are ${q.sg(runs[0].fn(k - 1), 3)} and ${q.sg(runs[1].fn(k - 1), 3)}; the losses ${q.sg(L(runs[0]), 3)} without warm-up and ${q.sg(L(runs[1]), 3)} with it. The worst loss so far: ${q.sg(mx(runs[0]), 3)} against ${q.sg(mx(runs[1]), 3)}.`;
        else say = `After ${q.n(k)} steps: ${q.sg(L(runs[0]), 3)} without warm-up, ${q.sg(L(runs[1]), 3)} with it. The cold start climbed to a loss of ${q.sg(mx(runs[0]), 3)} on its way; the warm one never went above ${q.sg(mx(runs[1]), 3)}. ` + (mx(runs[0]) > 2 * mx(runs[1]) ? 'Small first steps let the averages learn the terrain before the rate is trusted.' : 'Here the start is gentle enough that warming up changes little.');
        F.push(items, say, meta);
      });
      for (const f of F.list) f.narrowView = { x: 300, y: 0, w: 490, h: bx.y + bx.h + 80 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Linear warm-up', text: [surf.formula, 'η(t) = η · (t + 1)/W for t < W, then η', 'Adam: θ ← θ − η(t) m̂/(√v̂ + ε)', 'Adam’s first step: m̂/√v̂ = ±1 in each coordinate'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which run will climb to the higher loss on its way down?', options: ['no warm-up', 'warm-up'], answer: (fr) => { const m = fr[fr.length - 1].meta.maxes; return m[0] >= m[1] ? 'no warm-up' : 'warm-up'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'A linear warm-up to η = 0.01 over 100 steps. What is the rate on the first step (t = 0)?', options: ['0.0001', '0.01', '0', '0.001'], answer: 0, why: '0.01 × 1/100.' },
      { q: 'Why is Adam’s very first step about η in every coordinate?', options: ['with one gradient, m̂/√v̂ is g/|g| = ±1', 'Adam ignores the gradient', 'η is divided by the step', 'the bias correction is zero'], answer: 0, why: 'After bias correction m̂ = g and v̂ = g², whatever the size of g.' },
      { q: 'What does a warm-up protect against?', options: ['huge early steps taken before the optimiser’s statistics mean anything', 'overfitting at the end of training', 'a learning rate that is too small', 'vanishing gradients'], answer: 0, why: 'Early updates are the least informed; keeping them small avoids a damaging first jump.' },
      { q: 'After the warm-up ends, the rate…', options: ['follows the main schedule: constant here, often a cosine decay', 'drops to zero', 'keeps rising', 'is chosen at random'], answer: 0, why: 'Warm-up is only the first stretch of the schedule.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
