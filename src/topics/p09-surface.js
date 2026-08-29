// Part 9, the loss surface of a network small enough to draw: one hidden tanh neuron and one output weight,
// y = v · tanh(w · x), fitted to a handful of points. Its two weights make a plane, and the loss over that plane
// shows what larger networks have too: two equally good minima (flip both signs and nothing changes), a saddle at
// the origin where the gradient is zero, and flat plateaus where the neuron saturates. Gradient descent, run by the
// engine's optimiser, walks it from a chosen start while the fitted curve follows.
(function (S) {
  'use strict';
  const P = S.P09, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const XR = [-3, 3], VR = [-2.5, 2.5];
  P.SURF_DATA = {
    step: { label: 'a soft step: 1.2 · tanh(1.5x)', f: (x) => 1.2 * Math.tanh(1.5 * x) },
    line: { label: 'a straight line: 0.6x', f: (x) => 0.6 * x },
    flip: { label: 'a falling step: −tanh(2x)', f: (x) => -Math.tanh(2 * x) },
  };
  P.STARTS = {
    near: { label: 'near the saddle (0.2, 0.1)', at: [0.2, 0.1] },
    left: { label: 'lower left (−2, −0.5)', at: [-2, -0.5] },
    plateau: { label: 'on the plateau (2.6, −2)', at: [2.6, -2] },
    across: { label: 'top left (−1, 2)', at: [-1, 2] },
  };
  /** the points the network fits: 13 inputs from −2 to 2 with their targets (rounded to two decimals) */
  P.surfPoints = (kind) => { const xs = Array.from({ length: 13 }, (_, i) => -2 + (4 * i) / 12); return { xs, ys: xs.map((x) => Math.round(P.SURF_DATA[kind].f(x) * 100) / 100) }; };
  /** the loss at many (w, v) at once, in the engine: mean over the points of (v · tanh(w x) − y)² */
  P.surfLoss = function (pts, ws, vs) {
    const T = S.T, n = pts.xs.length;
    const W = T.tensor(ws.map((w) => [w])), V = T.tensor(vs.map((v) => [v])), X = T.tensor([pts.xs]), Y = T.tensor([pts.ys]);
    const out = T.mul(V, T.tanh(T.matmul(W, X)));
    return Array.from(T.mean(T.square(T.sub(out, Y)), 1).values()).map((x) => x + 0 * n);
  };
  /** the loss and its gradient at one point, by backpropagation */
  P.surfGrad = function (pts, w, v) {
    const T = S.T, wt = T.param(T.tensor(w)), vt = T.param(T.tensor(v));
    const L = T.mean(T.square(T.sub(T.mul(vt, T.tanh(T.mul(wt, T.tensor(pts.xs)))), T.tensor(pts.ys))));
    L.backward();
    return { L: L.item(), g: [wt.grad.item(), vt.grad.item()] };
  };

  S.topic({
    id: 'p09-loss-surface', family: 'p09-mlp', order: 30, title: 'The loss surface of a tiny network',
    blurb: 'A network with one hidden tanh neuron and one output weight, y = v · tanh(w · x), has just two weights, so its whole loss surface fits on the page. It already shows what big networks have: two equally good minima, a saddle at the origin where the gradient vanishes, and flat plateaus where the neuron saturates. Pick a start and watch gradient descent find its way, with the fitted curve beside it.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Points to fit', def: 'step', options: Object.keys(P.SURF_DATA).map((k) => ({ value: k, label: P.SURF_DATA[k].label })) },
      { id: 'start', kind: 'choice', label: 'Start (w, v)', def: 'near', options: Object.keys(P.STARTS).map((k) => ({ value: k, label: P.STARTS[k].label })) },
      { id: 'lr', kind: 'slider', label: 'Learning rate', def: 0.2, min: 0.02, max: 1, step: 0.02, fmt: (v) => H().fx(v, 2) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 120, min: 10, max: 400 },
    ],
    presets: [
      { label: 'Away from the saddle', values: { data: 'step', start: 'near', lr: 0.2, steps: 120 } },
      { label: 'Into the mirror minimum', values: { data: 'step', start: 'left', lr: 0.2, steps: 120 } },
      { label: 'Stuck on a plateau', values: { data: 'line', start: 'plateau', lr: 0.2, steps: 120 } },
      { label: 'A straight line', values: { data: 'line', start: 'across', lr: 0.3, steps: 200 } },
      { label: 'Steps too large', values: { data: 'flip', start: 'across', lr: 1, steps: 60 } },
    ],
    counters: [{ id: 'step', label: 'gradient steps' }],
    run(inp) {
      const T = S.T, F = H().frames(['step']), q = F.q, pts = P.surfPoints(inp.data);
      // the surface on a grid, from the engine in one batch, for its lowest point
      const N = 61, ws = [], vs = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { ws.push(XR[0] + ((XR[1] - XR[0]) * i) / (N - 1)); vs.push(VR[0] + ((VR[1] - VR[0]) * j) / (N - 1)); }
      const Ls = P.surfLoss(pts, ws, vs), k0 = Ls.indexOf(Math.min(...Ls)), gmin = [ws[k0], vs[k0]], gminL = Ls[k0];
      const mirror = P.surfLoss(pts, [-gmin[0]], [-gmin[1]])[0];
      const sad = P.surfGrad(pts, 0, 0);
      // gradient descent with the engine's optimiser
      const path = [], losses = [];
      T.float64(() => {
      const wt = T.param(T.tensor(P.STARTS[inp.start].at[0])), vt = T.param(T.tensor(P.STARTS[inp.start].at[1])), opt = new T.optim.SGD([wt, vt], { lr: inp.lr });
      path.push([wt.item(), vt.item()]);
      for (let s = 0; s < inp.steps; s++) {
        opt.zeroGrad();
        const L = T.mean(T.square(T.sub(T.mul(vt, T.tanh(T.mul(wt, T.tensor(pts.xs)))), T.tensor(pts.ys))));
        L.backward();
        losses.push(L.item());
        opt.step();
        path.push([wt.item(), vt.item()]);
      }
      });
      const end = T.float64(() => P.surfGrad(pts, path[path.length - 1][0], path[path.length - 1][1]));
      losses.push(end.L);
      // pictures
      const ax = P.ax({ prefix: 'ls', x: 60, y: 50, w: 360, h: 300, xr: XR, yr: VR, xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-2, -1, 0, 1, 2], xname: 'w (hidden weight)', yname: 'v (output weight)' });
      const plain = (w, v) => pts.xs.reduce((s, x, i) => s + (v * Math.tanh(w * x) - pts.ys[i]) ** 2, 0) / pts.xs.length;
      const surf = P.once('surf:' + inp.data, () => S.Kit.contour.surface(ax, 'ls:s', plain, { levels: 12, n: 80, depth: 2.2, min: 0 }));
      const fx = P.ax({ prefix: 'fx', x: 520, y: 50, w: 340, h: 220, xr: [-2.2, 2.2], yr: [-1.6, 1.6], xticks: [-2, -1, 0, 1, 2], yticks: [-1, 0, 1], xname: 'x', yname: 'y' });
      const fitAt = (w, v) => C().line(fx, 'fx:fit', Array.from({ length: 89 }, (_, i) => { const x = -2.2 + (4.4 * i) / 88; return [x, v * Math.tanh(w * x)]; }), 'act');
      const dataPts = pts.xs.map((x, i) => ({ key: 'fx:d' + i, kind: 'dot', x: fx.X(x), y: fx.Y(pts.ys[i]), r: 4.5, cls: 'sp c0', audit: 'overlap-ok' }));
      const marks = (o) => {
        const out = [];
        if (o.min) out.push(C().dot(ax, 'ls:m1', gmin[0], gmin[1], 'p09train', 6), C().dot(ax, 'ls:m2', -gmin[0], -gmin[1], 'p09train', 6));
        if (o.sad) out.push(C().dot(ax, 'ls:sad', 0, 0, 'p09hot', 6));
        return out;
      };
      const read = (rows) => H().lines('ls:r', 520, 330, rows, 14, 24);
      const base = [...ax.items, ...surf, ...fx.items, ...dataPts];
      F.push([...base, ...marks({ min: true }), fitAt(gmin[0], gmin[1]), ...read([['lowest loss on the grid: ' + H().fx(gminL, 4), 'math'], ['at w = ' + H().fx(gmin[0], 1) + ', v = ' + H().fx(gmin[1], 2), 'math']])],
        `Every point of the left plane is a choice of the two weights, shaded by its loss (darker is lower). The lowest point of the grid is w = ${q(gmin[0], 1)}, v = ${q(gmin[1], 2)}, with loss ${q(gminL, 4)}; its curve on the right goes through the ${q.n(pts.xs.length)} points.`, { phase: 'min', gmin, gminL, n: pts.xs.length }, { hl: { eq: [0, 1] } });
      F.push([...base, ...marks({ min: true }), fitAt(-gmin[0], -gmin[1]), ...read([['L(w, v) = L(−w, −v)', 'math'], ['mirror loss: ' + H().fx(mirror, 4), 'math']])],
        `The mirror point w = ${q(-gmin[0], 1)}, v = ${q(-gmin[1], 2)} has exactly the same loss, ${q(mirror, 4)}: tanh is odd, so flipping the sign of both weights gives the same curve. Every hidden neuron of a real network has such twins, so there is never just one best set of weights.`, { phase: 'mirror', mirror, gmin, mpt: [-gmin[0], -gmin[1]] }, { hl: { eq: [2] } });
      F.push([...base, ...marks({ min: true, sad: true }), fitAt(0, 0), ...read([['at (0, 0): loss ' + H().fx(sad.L, 4), 'math'], ['gradient (' + H().fx(sad.g[0], 3) + ', ' + H().fx(sad.g[1], 3) + ')', 'math']])],
        `Between the twins, at the origin, the network outputs zero for every input: loss ${q(sad.L, 4)}, and backpropagation gives the gradient (${q(sad.g[0], 3)}, ${q(sad.g[1], 3)}). It is flat there, yet it is not a minimum: it is a saddle, downhill towards either twin. Gradient descent started exactly on it would never move.`, { phase: 'saddle', L: sad.L, g: sad.g }, { hl: { eq: [3] } });
      const show = [0, 1, 3, 8, 20, 50].filter((s) => s < inp.steps);
      for (const s of show) {
        F.set('step', s);
        const [w, v] = path[s], L = losses[s];
        F.push([...base, ...marks({ min: true }), ...S.Kit.contour.path(ax, 'ls:gd', path.map((p) => [Math.max(XR[0], Math.min(XR[1], p[0])), Math.max(VR[0], Math.min(VR[1], p[1]))]), 'o-adam', { upto: s }), fitAt(w, v), ...read([['step ' + s + ':  w = ' + H().fx(w, 3) + ', v = ' + H().fx(v, 3), 'math'], ['loss ' + H().fx(L, 4), 'math']])],
          s === 0 ? `Gradient descent starts at w = ${q(w, 3)}, v = ${q(v, 3)} with loss ${q(L, 4)}, and each step moves both weights against the gradient, by the learning rate ${q(inp.lr, 2)} times it.`
            : `After ${P.qpl(q, s, 'step')}: w = ${q(w, 3)}, v = ${q(v, 3)}, loss ${q(L, 4)}.`, { phase: 'gd', s, w, v, L }, { hl: { eq: [4] } });
      }
      F.set('step', inp.steps);
      const [we, ve] = path[path.length - 1], gn = Math.hypot(end.g[0], end.g[1]);
      const which = end.L <= 2 * gminL + 2e-3 ? (Math.sign(we) === Math.sign(gmin[0]) ? 'the first minimum' : 'its mirror twin') : null;
      F.push([...base, ...marks({ min: true }), ...S.Kit.contour.path(ax, 'ls:gd', path.map((p) => [Math.max(XR[0], Math.min(XR[1], p[0])), Math.max(VR[0], Math.min(VR[1], p[1]))]), 'o-adam'), fitAt(we, ve), ...read([['end: loss ' + H().fx(end.L, 4), 'math'], ['gradient size ' + H().sg(gn, 2), 'math']])],
        which ? `It settles in ${which}, loss ${q(end.L, 4)} against the grid’s best ${q(gminL, 4)}, where the gradient has shrunk to ${q.sg(gn, 2)}. Which twin it reaches depends only on where it started.`
          : `After ${P.qpl(q, inp.steps, 'step')} the loss is still ${q(end.L, 4)}, far above the best ${q(gminL, 4)}, and the gradient is only ${q.sg(gn, 2)}: ` + (gn < 0.05 ? 'the neuron is saturated, the surface is a plateau, and the steps have become tiny.' : 'the steps are too large for this surface, so it keeps overshooting.'),
        { phase: 'end', L: end.L, gn, gminL, we, ve, steps: inp.steps }, { hl: { eq: [4] } });
      P.narrow(F.list, () => ({ x: 10, y: 20, w: 430, h: 370 }));
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A two-weight network', text: ['y = v · tanh(w · x)', 'L(w, v) = mean over points of (y − target)²', 'symmetry: L(w, v) = L(−w, −v)', 'at (0, 0): ∂L/∂w = ∂L/∂v = 0, a saddle', 'descent: (w, v) ← (w, v) − η ∇L'].join('\n') }] }, result: { path, losses } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'gd' && f.meta.s === 0, once: true, pick: 'option',
      ask: 'Where will gradient descent end up?',
      options: ['in one of the two minima', 'stuck on a plateau or bouncing, well above the best loss'],
      answer: (fr) => { const e = fr.find((f) => f.meta.phase === 'end').meta; return e.L <= 2 * e.gminL + 2e-3 ? 'in one of the two minima' : 'stuck on a plateau or bouncing, well above the best loss'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'end').say,
    }],
    quiz: [
      { q: 'Why does y = v · tanh(w x) have two equally good minima?', options: ['flipping the signs of w and v gives the same function', 'the data has two classes', 'the learning rate is too high', 'tanh has two zeros'], answer: 0, why: 'tanh(−z) = −tanh(z), so (−v)·tanh(−wx) = v·tanh(wx).' },
      { q: 'At (w, v) = (0, 0) the gradient is zero. That point is…', options: ['a saddle: downhill in some directions', 'the global minimum', 'a local maximum', 'outside the surface'], answer: 0, why: 'The loss falls towards either twin minimum, so it is not a minimum itself.' },
      { q: 'Far out along w, the surface is almost flat because…', options: ['tanh saturates, so changing w barely changes the output', 'the loss is zero there', 'v is zero', 'the data ends'], answer: 0, why: 'For large |w x| the neuron outputs nearly ±1 whatever w is: a plateau with tiny gradients.' },
      { q: 'Starting gradient descent exactly at the origin would…', options: ['never move, since the gradient is exactly zero', 'go to the nearest minimum', 'diverge', 'oscillate'], answer: 0, why: 'This is one reason weights are initialised randomly, not at zero.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
