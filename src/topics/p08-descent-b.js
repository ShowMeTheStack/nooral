// Part 8, gradient descent (2): the learning rate on a quadratic bowl, too small, just right and too large, with
// the limit 2/λmax worked out from the curvatures; and stochastic and mini-batch descent on a real least-squares
// loss, where each step sees only a few of the data points and the path turns noisy.
(function (S) {
  'use strict';
  const P = S.P08, H = () => S.Kit.ml;
  const boxFor = (surf, x, y, w, hmax) => { const r = (surf.yr[1] - surf.yr[0]) / (surf.xr[1] - surf.xr[0]); const h = Math.min(hmax, w * r); return { x, y, w: h / r, h }; };

  // ---------------------------------------------------------------------------------------------------------------
  const QUADS = { ravine: 'ravine', tilted: 'tilted', bowl: 'bowl' };
  S.topic({
    id: 'opt-lr', family: 'p08-descent', order: 30, title: 'The learning rate: too small, just right, too large',
    blurb: 'On a quadratic bowl, one step of gradient descent multiplies the distance to the bottom along each curvature direction by (1 − ηλ), where λ is the curvature that way. A small η shrinks it slowly; η = 2/(λmax + λmin) balances the two directions; past η = 2/λmax the factor’s size exceeds 1 and every step overshoots further than the last.',
    inputs: [
      { id: 'surf', kind: 'choice', label: 'Bowl', def: 'tilted', options: P.surfOptions(Object.keys(QUADS)) },
      { id: 'lr', kind: 'slider', label: 'Your η', def: 0.6, min: 0.01, max: 2.5, step: 0.01, log: true, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 30, min: 5, max: 60 },
    ],
    presets: [
      { label: 'Tilted valley', values: { surf: 'tilted', lr: 0.6, steps: 30 } },
      { label: 'Round bowl: one step', values: { surf: 'bowl', lr: 1, steps: 10 } },
      { label: 'Ravine: badly conditioned', values: { surf: 'ravine', lr: 0.3, steps: 60 } },
      { label: 'Yours just over the limit', values: { surf: 'tilted', lr: 1.02, steps: 30 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF[inp.surf], N = inp.steps, F = H().frames(['step']), q = F.q;
      const [lmax, lmin] = P.eig2(surf.h(0, 0)), limit = 2 / lmax;
      const runs = [
        { id: 'small', name: 'too small', lr: 0.15 / lmax, cls: 'o-mom' },
        { id: 'right', name: 'just right', lr: 2 / (lmax + lmin), cls: 'o-adam' },
        { id: 'large', name: 'too large', lr: 2.12 / lmax, cls: 'o-rms' },
        { id: 'yours', name: 'yours', lr: inp.lr, cls: 'o-gd' },
      ];
      for (const r of runs) { Object.assign(r, P.run(surf, 'sgd', { lr: r.lr }, N)); r.fac = [Math.abs(1 - r.lr * lmax), Math.abs(1 - r.lr * lmin)]; }
      const bx = boxFor(surf, 40, 30, 420, 420), ax = P.axes('lr', surf, bx, { xticks: 1, yticks: 1 });
      let lo = Infinity, hi = 0;
      for (const r of runs) for (const v of r.loss) if (Number.isFinite(v) && v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      const lax = P.logAxes('ll', { x: 540, y: 50, w: 230, h: 150 }, N, Math.max(1e-10, lo), Math.min(1e8, hi));
      const legend = (k) => P.legend('lg', 540, 262, runs.map((r) => ({ text: r.name + ', η = ' + H().sg(r.lr, 3) + ': L = ' + H().sg(r.loss[Math.min(k, r.loss.length - 1)], 3), cls: r.cls })), { size: 13 });
      const base = () => [...ax.items, ...P.contour(ax, 'lr:c', surf), P.start(ax, 'lr:s', surf.start), ...lax.items];
      const AT = [...new Set([0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 45, N].filter((k) => k <= N))];
      AT.forEach((k) => {
        F.set('step', k);
        const items = [...base()];
        for (const r of runs) items.push(...P.path(ax, 'lr:' + r.id, r.pts, r.cls + ' thin', Math.min(k, r.pts.length - 1), { r: 5 }), P.logLine(lax, 'll:' + r.id, r.loss, r.cls, k));
        items.push(...legend(k));
        let say;
        const L = (r) => r.loss[Math.min(k, r.loss.length - 1)];
        if (k === 0) say = `The curvatures of this bowl are λmax = ${q.sg(lmax, 3)} and λmin = ${q.sg(lmin, 3)}, so gradient descent is stable only below two over λmax, ${q.sg(limit, 3)}. Three rates set from them, ${q.sg(runs[0].lr, 3)}, ${q.sg(runs[1].lr, 3)} and ${q.sg(runs[2].lr, 3)}, and yours, ${q.sg(inp.lr, 3)}, all start from the same point.`;
        else if (k === AT[1]) say = `Each step multiplies the error along the steep direction by 1 − ηλmax: ${q.sg(1 - runs[0].lr * lmax, 3)} for the small rate, ${q.sg(1 - runs[1].lr * lmax, 3)} for the balanced one, ${q.sg(1 - runs[2].lr * lmax, 3)} for the large one. A negative factor means the step jumps over the floor to the other wall.`;
        else if (k === N) say = `After ${q.n(k)} steps the losses are ${q.sg(L(runs[0]), 3)} (too small: still creeping), ${q.sg(L(runs[1]), 3)} (just right) and ${q.sg(L(runs[2]), 3)} (too large: each overshoot is bigger than the last). Yours, η = ${q.sg(inp.lr, 3)}, ${inp.lr > limit ? 'is past the limit' : 'is inside the limit'} and ends at ${q.sg(L(runs[3]), 3)}.`;
        else say = `Step ${q.n(k)}: losses ${q.sg(L(runs[0]), 3)}, ${q.sg(L(runs[1]), 3)}, ${q.sg(L(runs[2]), 3)}, and yours ${q.sg(L(runs[3]), 3)}. The slowest factor decides the pace: for the balanced rate both directions shrink by ${q.sg(runs[1].fac[0], 3)} a step.`;
        F.push(items, say, { phase: k === 0 ? 'start' : k === N ? 'end' : 'step', k, lmax, lmin, limit, lrs: runs.map((r) => r.lr), losses: runs.map(L), f1: runs.map((r) => 1 - r.lr * lmax), fac: runs[1].fac });
      });
      // on a phone the whole picture, so the legend that names the colours stays in view
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The step size on a quadratic', text: [surf.formula, 'Hessian eigenvalues λmax, λmin: the curvatures', 'error along each eigenvector × (1 − ηλ) per step', 'converges only if η < 2/λmax', 'fastest: η = 2/(λmax + λmin)', 'condition number λmax/λmin: how slow the best is'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'The “too large” rate is just over 2/λmax. What will its loss do?', options: ['grow, step after step', 'fall quickly, then settle', 'fall slowly', 'stay where it is'], answer: 'grow, step after step', explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'A quadratic has curvatures 4 and 1. Above which learning rate does gradient descent diverge?', options: ['0.5', '2', '0.25', '1'], answer: 0, why: 'The limit is 2/λmax = 2/4.' },
      { q: 'With η = 0.1 and curvature λ = 5, the error in that direction is multiplied each step by…', options: ['0.5', '−0.5', '1.5', '0.1'], answer: 0, why: '1 − 0.1 × 5 = 0.5.' },
      { q: 'What does a negative factor 1 − ηλ mean?', options: ['each step jumps over the bottom to the other side', 'the loss becomes negative', 'the step goes uphill', 'the run has converged'], answer: 0, why: 'The sign of the error flips: the path bounces between the walls, shrinking only while the factor’s size is below 1.' },
      { q: 'Why is a ravine slow for gradient descent even at its best rate?', options: ['η must suit the steep direction, which leaves the flat one crawling', 'the gradient is zero in a ravine', 'there are many minima', 'the steps are random'], answer: 0, why: 'The best factor is (κ − 1)/(κ + 1) with κ = λmax/λmin: close to 1 when κ is large.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  /** points (x, y) round a line, from a seed, and the least-squares surface over (slope a, intercept b) */
  P.lsq = function (n, seed) {
    const r = S.ML.rng(seed), X = [], Y = [];
    for (let i = 0; i < n; i++) { const x = -2 + 4 * r(); X.push(+x.toFixed(3)); Y.push(+(1.4 * x - 0.6 + 0.5 * r.normal()).toFixed(3)); }
    let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < n; i++) { sx += X[i]; sy += Y[i]; sxx += X[i] * X[i]; sxy += X[i] * Y[i]; syy += Y[i] * Y[i]; }
    sx /= n; sy /= n; sxx /= n; sxy /= n; syy /= n;
    // L(a, b) = ½ mean (a x + b − y)²
    const f = (a, b) => 0.5 * (a * a * sxx + b * b + syy + 2 * a * b * sx - 2 * a * sxy - 2 * b * sy);
    const g = (a, b) => [a * sxx + b * sx - sxy, a * sx + b - sy];
    const det = sxx - sx * sx, best = [(sxy - sx * sy) / det, (sy * sxx - sx * sxy) / det];
    /** the gradient of one batch of indices */
    const gb = (p, idx) => { let ga = 0, gb2 = 0; for (const i of idx) { const res = p[0] * X[i] + p[1] - Y[i]; ga += res * X[i]; gb2 += res; } return [ga / idx.length, gb2 / idx.length]; };
    return { X, Y, best, surf: { label: 'least squares', formula: 'L(a, b) = ½ · mean of (a·x + b − y)²', f, g, h: () => [[sxx, sx], [sx, 1]], xr: [best[0] - 2.2, best[0] + 2.2], yr: [best[1] - 1.9, best[1] + 1.9], start: [best[0] - 1.7, best[1] + 1.5], min: [best] }, gb };
  };
  /** the batches of each step: the indices shuffled once per pass through the data, cut into pieces of size B */
  P.batches = function (n, B, steps, seed) {
    const r = S.ML.rng(seed), out = [];
    let order = [];
    while (out.length < steps) {
      if (order.length < B) order = order.concat(r.perm(n));
      out.push(order.splice(0, B));
    }
    return out;
  };
  S.topic({
    id: 'opt-sgd', family: 'p08-descent', order: 40, title: 'Stochastic and mini-batch gradient descent',
    blurb: 'The gradient of a loss averaged over all the data costs one pass through the data per step. Stochastic gradient descent uses one random example per step, mini-batch descent a few: each step is cheap and points roughly downhill, but not exactly, so the path wanders. Here a line y = a·x + b is fitted to points by least squares, and three runs share a learning rate.',
    inputs: [
      { id: 'B', kind: 'number', label: 'Mini-batch size', def: 5, min: 2, max: 20 },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 0.25, min: 0.02, max: 0.6, step: 0.01, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 40, min: 10, max: 80 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 3, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Batches of 5', values: { B: 5, lr: 0.25, steps: 40, seed: 3 } },
      { label: 'Batches of 15', values: { B: 15, lr: 0.25, steps: 40, seed: 3 } },
      { label: 'A large rate: noisy', values: { B: 4, lr: 0.5, steps: 60, seed: 8 } },
      { label: 'A small rate: smooth but slow', values: { B: 4, lr: 0.06, steps: 80, seed: 2 } },
    ],
    counters: [{ id: 'step', label: 'steps' }, { id: 'seen', label: 'examples seen (mini-batch)' }],
    run(inp) {
      const n = 40, D = P.lsq(n, inp.seed), surf = D.surf, N = inp.steps, B = inp.B, F = H().frames(['step', 'seen']), q = F.q;
      const bs = { full: null, mini: P.batches(n, B, N, inp.seed * 7 + 1), sgd: P.batches(n, 1, N, inp.seed * 7 + 2) };
      const runs = [
        { id: 'full', name: 'full batch', cls: 'o-gd' },
        { id: 'mini', name: 'mini-batch', cls: 'o-adam' },
        { id: 'sgd', name: 'one point', cls: 'o-rms' },
      ];
      for (const r of runs) Object.assign(r, P.run(surf, 'sgd', { lr: inp.lr }, N, { grad: (p, k) => (bs[r.id] ? D.gb(p, bs[r.id][k]) : surf.g(p[0], p[1])) }));
      // the data with the mini-batch run's current line
      const dax = S.Kit.curve.axes({ prefix: 'dt', x: 40, y: 40, w: 220, h: 190, xr: [-2.2, 2.2], yr: [Math.min(...D.Y) - 0.4, Math.max(...D.Y) + 0.4], xticks: [-2, 0, 2], xname: 'x', yname: 'y' });
      const lo = Math.min(...runs.map((r) => Math.min(...r.loss))), hi = Math.max(...runs.map((r) => r.loss[0]));
      const lax = P.logAxes('sl', { x: 40, y: 310, w: 220, h: 130 }, N, lo * 0.9, hi);
      const bx = { x: 340, y: 30, w: 430, h: 430 * ((surf.yr[1] - surf.yr[0]) / (surf.xr[1] - surf.xr[0])) }, ax = P.axes('sg', surf, bx, { xticks: 1, yticks: 1, xname: 'slope a', yname: 'intercept b' });
      const fmin = surf.f(D.best[0], D.best[1]);
      const AT = [...new Set([0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 26, 32, 40, 50, 60, 70, 80].filter((k) => k <= N).concat([N]))];
      AT.forEach((k) => {
        F.set('step', k).set('seen', k * B);
        const m = runs[1], p = m.pts[k], batch = k ? bs.mini[k - 1] : [];
        const items = [...dax.items, ...ax.items, ...P.contour(ax, 'sg:c', surf), P.minDot(ax, 'sg:m', D.best), P.start(ax, 'sg:s', surf.start), ...lax.items];
        D.X.forEach((x, i) => items.push({ key: 'dt:p' + i, kind: 'dot', x: dax.X(x), y: dax.Y(D.Y[i]), r: batch.includes(i) ? 5.5 : 4, cls: 'p8-pt' + (batch.includes(i) ? ' on' : ''), audit: 'overlap-ok' }));
        const ln = (a, b2) => S.Kit.curve.line(dax, 'dt:l', [[-2.2, a * -2.2 + b2], [2.2, a * 2.2 + b2]], 'o-adam');
        items.push(ln(p[0], p[1]));
        for (const r of runs) items.push(...P.path(ax, 'sg:' + r.id, r.pts, r.cls + ' thin', k, { r: 5 }), P.logLine(lax, 'sl:' + r.id, r.loss, r.cls, k));
        items.push(...P.legend('lg', 340, bx.y + bx.h + 64, runs.map((r) => ({ text: r.name, cls: r.cls })), { dir: 'row', size: 12 }));
        let say;
        const Ls = runs.map((r) => r.loss[k]);
        if (k === 0) say = `${q.n(n)} points lie round a line. The loss of a line y = a·x + b is half the mean squared error, a bowl over the plane of (a, b) whose bottom, the least-squares fit, is at a = ${q(D.best[0])}, b = ${q(D.best[1])}, loss ${q.sg(fmin, 3)}. All three runs start at the white dot with η = ${q.sg(inp.lr, 3)}.`;
        else {
          const gm = m.grads[k - 1], gf = surf.g(m.pts[k - 1][0], m.pts[k - 1][1]);
          const cosv = (gm[0] * gf[0] + gm[1] * gf[1]) / (Math.hypot(...gm) * Math.hypot(...gf));
          if (k <= 3) say = `Step ${q.n(k)}: the mini-batch (the ${q.n(B)} pink points) has gradient (${q.sg(gm[0], 3)}, ${q.sg(gm[1], 3)}); over all the data it would be (${q.sg(gf[0], 3)}, ${q.sg(gf[1], 3)}). They point within ${q(Math.acos(Math.max(-1, Math.min(1, cosv))) * 180 / Math.PI, 0)} degrees of each other: roughly downhill, but not exactly.`;
          else if (k === N) say = `After ${q.n(k)} steps the losses are ${q.sg(Ls[0], 4)} (full batch), ${q.sg(Ls[1], 4)} (mini-batch) and ${q.sg(Ls[2], 4)} (one point), against the best possible ${q.sg(fmin, 4)}. With η fixed, the noisy runs never settle: near the bottom each batch still pulls its own way, so they jitter round it.`;
          else say = `Step ${q.n(k)}: losses ${q.sg(Ls[0], 4)}, ${q.sg(Ls[1], 4)} and ${q.sg(Ls[2], 4)}. The full-batch path is smooth; a single point’s gradient is the noisiest. But a full step costs ${q.n(n)} gradients and a single point one, so for the same work SGD takes ${q.n(n)} steps.`;
          F.push(items, say, { phase: k === N ? 'end' : 'step', k, best: D.best, fmin, losses: Ls, gm, gf, angle: Math.acos(Math.max(-1, Math.min(1, cosv))) * 180 / Math.PI, n, B, lr: inp.lr, batch });
          return;
        }
        F.push(items, say, { phase: 'start', k, best: D.best, fmin, n, lr: inp.lr });
      });
      for (const f of F.list) f.narrowView = { x: 300, y: 0, w: 490, h: bx.y + bx.h + 80 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Stochastic gradients', text: ['L(a, b) = ½ · mean over i of (a·xᵢ + b − yᵢ)²', 'full batch: the gradient of the mean over all points', 'mini-batch: the mean over B random points', 'each pass through the data shuffles the order', 'the batch gradient is right on average: its mean is the full gradient'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which of the three paths will be smoothest?', options: ['the full batch', 'the mini-batch', 'one point at a time'], answer: 'the full batch', explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'end', pick: 'option', ask: 'With η fixed, will the one-point run settle exactly at the bottom?', options: ['no: it keeps jittering round it', 'yes, given enough steps', 'it will diverge', 'it will stop at the start'], answer: 'no: it keeps jittering round it', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'A dataset has 1000 examples and the mini-batch size is 50. How many steps make one pass through the data?', options: ['20', '50', '1000', '1'], answer: 0, why: '1000 / 50 = 20 batches per pass.' },
      { q: 'Why does a mini-batch gradient point roughly, not exactly, downhill?', options: ['it averages only a sample of the examples', 'the learning rate is too small', 'the loss changes between steps', 'it is computed in reverse'], answer: 0, why: 'A sample mean is a noisy estimate of the full mean; it is right on average.' },
      { q: 'What makes SGD attractive despite its noise?', options: ['each step is far cheaper, so many more steps fit in the same time', 'it always finds a lower loss', 'it needs no gradient', 'it never needs a learning rate'], answer: 0, why: 'One example per step costs a thousandth of a full-batch step on 1000 examples.' },
      { q: 'How can the noisy runs be made to settle at the bottom?', options: ['lower the learning rate as training goes on', 'raise the learning rate', 'shuffle less', 'use fewer epochs'], answer: 0, why: 'A decaying schedule shrinks the jitter; see the learning-rate schedules.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
