// Part 3, the closed form against gradient descent. The mean squared error of a line is a bowl over the plane of
// (w, b). The closed form (QR) lands on its floor at once; gradient descent starts at (0, 0) and walks downhill,
// epoch by epoch (ML.LinearRegression with solver 'gd' and its trace). Raw inputs make the bowl a long narrow
// valley, so a safe step is tiny and the walk zig-zags; scaled inputs make it round.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n;
  const CK = () => S.Kit.contour, C = () => S.Kit.curve;

  /** which epochs get a frame: every one while few, then spreading out (always the last) */
  const pickEpochs = (E) => { const out = new Set(); for (let e = 0; e <= Math.min(E, 8); e++) out.add(e); let e = 8; while (e < E) { e = Math.ceil(e * 1.45); if (e < E) out.add(e); } out.add(E); return [...out].sort((a, b) => a - b); };

  S.topic({
    id: 'gd-closed', family: 'p03-gd', order: 10, title: 'Closed form against gradient descent',
    blurb: 'The loss L(w, b) = (1/2n) Σ (yᵢ − w·xᵢ − b)² is a bowl over the plane of the two parameters. Solving the normal equations jumps straight to its lowest point. Gradient descent starts at (0, 0) and repeatedly steps against the gradient, w ← w − η ∂L/∂w and b ← b − η ∂L/∂b. Too small a step η crawls, too large a step overshoots and can blow up, and the shape of the bowl, set by the scale of x, decides which is which.',
    inputs: [
      { id: 'scale', kind: 'choice', label: 'input x', def: 'raw', options: [{ value: 'raw', label: 'raw (0 to 10)' }, { value: 'std', label: 'standardised' }] },
      { id: 'lr', kind: 'slider', label: 'learning rate η', def: -1.6, min: -3, max: 0.4, step: 0.05, log: true, fmt: (v) => H.n(Math.pow(10, v), 2) },
      { id: 'epochs', kind: 'number', label: 'epochs', def: 60, min: 5, max: 400 },
      { id: 'seed', kind: 'number', label: 'seed', def: 4, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Raw x, a careful step: a slow ravine', values: { scale: 'raw', lr: -2, epochs: 120 } },
      { label: 'Raw x, a bolder step: zig-zag', values: { scale: 'raw', lr: Math.log10(0.045), epochs: 40 } },
      { label: 'Raw x, too big: it blows up', values: { scale: 'raw', lr: Math.log10(0.062), epochs: 30 } },
      { label: 'Scaled x, η = 0.1: a round bowl', values: { scale: 'std', lr: -1, epochs: 60 } },
      { label: 'Scaled x, η = 1: one stride', values: { scale: 'std', lr: 0, epochs: 10 } },
      { label: 'Scaled x, η = 1.9: bouncing in', values: { scale: 'std', lr: Math.log10(1.9), epochs: 40 } },
      { label: 'Scaled x, η = 2.1: bouncing out', values: { scale: 'std', lr: Math.log10(2.1), epochs: 16 } },
    ],
    counters: [{ id: 'epoch', label: 'epoch' }],
    run(inp) {
      const raw = H.lineData(20, inp.seed, 0.6, 2.2, 0.8);
      const xs0 = raw.X.map((r) => r[0]), y = raw.y, n = y.length;
      const mu = ML().mean(xs0), sd = Math.sqrt(ML().colVars(raw.X)[0]);
      const xs = inp.scale === 'std' ? xs0.map((v) => (v - mu) / sd) : xs0, X = xs.map((v) => [v]);
      const lr = Math.pow(10, inp.lr), E = inp.epochs;
      const cf = ML().LinearRegression().fit(X, y), wS = cf.coef[0], bS = cf.intercept;
      const gd = ML().LinearRegression({ solver: 'gd', lr, epochs: E, traceEvery: 1 }).fit(X, y);
      const steps = gd.trace.filter((s) => s.kind === 'gd');
      const loss = (w, b) => { let s = 0; for (let i = 0; i < n; i++) s += (y[i] - w * xs[i] - b) ** 2; return s / (2 * n); };
      const Ls = loss(wS, bS), L0 = loss(0, 0);
      // the curvature of the bowl: eigenvalues of the Hessian [[mean x², mean x], [mean x, 1]]
      const mxx = ML().mean(xs.map((v) => v * v)), mx = ML().mean(xs), tr = mxx + 1, det = mxx - mx * mx, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const lamMax = tr / 2 + disc, lamMin = tr / 2 - disc, lrMax = 2 / lamMax;
      const F = new S.Frames(['epoch']);

      // the plane of (w, b): around the floor and the start
      // the plane of (w, b): the start, the floor and every point of the walk that is not running away
      const tame = steps.filter((s) => isFinite(s.loss) && s.loss <= L0 * 30);
      const wr = H.span([0, wS, ...tame.map((s) => s.w[0])], 0.12), br = H.span([0, bS, ...tame.map((s) => s.b)], 0.12);
      const ax = H.axes({ prefix: 'gs', x: 46, y: 40, w: 400, h: 400, xr: wr, yr: br, xname: 'w (slope)', yname: 'b (intercept)', grid: false });
      const surf = CK().surface(ax, 'gs:s', loss, { levels: 10, n: 80, min: Ls });
      const star = { key: 'gs:star', kind: 'dot', x: ax.X(wS), y: ax.Y(bS), r: 7, cls: 'pl-star', audit: 'overlap-ok', layer: 'items', z: 4 };
      const clampPt = (s) => [Math.max(wr[0], Math.min(wr[1], s.w[0])), Math.max(br[0], Math.min(br[1], s.b))];
      const inside = (s) => isFinite(s.w[0]) && isFinite(s.b) && s.w[0] >= wr[0] && s.w[0] <= wr[1] && s.b >= br[0] && s.b <= br[1];
      // the loss against epochs, on a log scale
      const lg = (v) => Math.log10(Math.max(v, 1e-12));
      const lossVals = steps.map((s) => s.loss).filter(isFinite);
      const lyLo = Math.floor(lg(Math.min(Ls, ...lossVals)) * 2) / 2, lyHi = Math.min(Math.ceil(lg(Math.max(...lossVals.slice(0, 1), ...lossVals)) * 2) / 2, lg(L0) + 3);
      const lax = H.axes({ prefix: 'gl', x: 530, y: 50, w: 220, h: 170, xr: [0, E], yr: [lyLo, Math.max(lyHi, lyLo + 1)], xticks: C().ticks(0, E, 3), yticks: 1, fmtY: (v) => (Number.isInteger(v) ? '10' + (v < 0 ? '⁻' : '') + H.sup(Math.abs(v)) : ''), xname: 'epoch', yname: 'loss (log scale)' });
      const floorLine = { key: 'gl:floor', kind: 'path', d: `M${lax.X(0)},${lax.Y(lg(Ls))} H${lax.X(E)}`, cls: 'guide', audit: 'ignore', layer: 'back', z: 3 };
      const head = () => [...surf, ...ax.items, H.title('gs:t', 46, 14, 'The loss over (w, b)'), ...lax.items, floorLine, H.title('gl:t', 500, 20, 'Loss against epochs')];
      // outcome over the whole run
      const last = steps[steps.length - 1], lastLoss = last.loss, blew = !isFinite(lastLoss) || lastLoss > L0 * 4;
      const gapEnd = isFinite(lastLoss) ? Math.hypot(last.w[0] - wS, last.b - bS) : Infinity;
      let flips = 0;
      for (let k = 2; k < steps.length; k++) if (isFinite(steps[k].w[0]) && (steps[k].w[0] - wS) * (steps[k - 1].w[0] - wS) < 0) flips++;
      const reached = !blew && gapEnd < 0.02 * (Math.abs(wS) + Math.abs(bS) + 1);
      const outcome = blew ? 'it blows up' : !reached ? 'it is still on the way when the epochs run out' : flips >= 3 ? 'it zig-zags in' : 'it slides in smoothly';
      const meta0 = { wS, bS, Ls, L0, lr, lrMax, lamMax, lamMin, E, mu, sd, cond: lamMax / lamMin, xLo: 0, xHi: 10 };
      const eqX = 500, eqY = 290;
      F.add({ items: [...head(), star, ...H.eqs('gs:eq', eqX, eqY, [['closed form (QR):', 'good'], ['w* = ' + n$(wS) + ',  b* = ' + n$(bS), 'good'], ['L* = ' + n$(Ls), 'good'], ['curvatures ' + n$(lamMax) + ' and ' + n$(lamMin), 'muted'], ['stable while η < ' + n$(lrMax), 'muted']])], say: 'The shading is the loss over every slope w and intercept b' + (inp.scale === 'std' ? ' for the standardised x' : ' for the raw x, which runs from 0 to 10') + '. The closed form solves the normal equations in one go: w* = ' + n$(wS) + ', b* = ' + n$(bS) + ', loss ' + n$(Ls) + ' (green). The bowl curves ' + n$(lamMax / lamMin) + ' times more steeply one way than the other, so steps of η = ' + n$(lr) + ' stay stable only while η < 2 / ' + n$(lamMax) + ' = ' + n$(lrMax) + '.', meta: Object.assign({ phase: 'closed', outcome }, meta0) });
      const show = pickEpochs(E);
      let lastIn = 0;
      for (const e of show) {
        const st = steps[e];
        if (!st) continue;
        if (inside(st)) lastIn = e;
        const path = steps.slice(0, e + 1).filter((s, k) => k <= lastIn + 1).map(clampPt);
        const ok = isFinite(st.loss);
        const items = [...head(), star, ...CK().path(ax, 'gs:path', path, 'o-gd', { dots: e <= 30 ? 1 : 0 })];
        const curve = steps.slice(0, e + 1).map((s) => [s.epoch, isFinite(s.loss) && lg(s.loss) <= lax.o.yr[1] ? lg(s.loss) : NaN]);
        items.push(C().line(lax, 'gl:line', curve, 'act'));
        if (ok && lg(st.loss) <= lax.o.yr[1]) items.push(C().dot(lax, 'gl:dot', e, lg(st.loss), ''));
        // the negative gradient at the current point, as an arrow
        const g = st.grad.concat([0]), gb = ok ? (() => { let s = 0; for (let i = 0; i < n; i++) s += st.w[0] * xs[i] + st.b - y[i]; return s / n; })() : NaN;
        g[1] = gb;
        if (ok && inside(st) && Math.hypot(g[0], g[1]) > 1e-9 && e < E) {
          const p0 = [ax.X(st.w[0]), ax.Y(st.b)], step = [-lr * g[0], -lr * g[1]], p1 = [ax.X(st.w[0] + step[0]), ax.Y(st.b + step[1])];
          const L = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
          if (L > 6) items.push(H.arrow('gs:g', p0[0], p0[1], p1[0], p1[1], 'act', { head: 9 }));
        }
        const lines = ok
          ? [['epoch ' + e, 'act'], ['w = ' + n$(st.w[0]) + ',  b = ' + n$(st.b), ''], ['∂L/∂w = ' + n$(g[0]) + ',  ∂L/∂b = ' + n$(g[1]), 'muted'], ['loss ' + n$(st.loss), 'act'], ['closed form ' + n$(Ls), 'good']]
          : [['epoch ' + e, 'act'], ['the numbers overflowed', 'hot']];
        items.push(...H.eqs('gs:eq', eqX, eqY, lines));
        F.set('epoch', e);
        const dist = ok ? Math.hypot(st.w[0] - wS, st.b - bS) : Infinity;
        let say;
        if (!ok) say = 'Epoch ' + e + ': the steps have grown without limit and the numbers overflow. Gradient descent has diverged.';
        else if (e === 0) say = 'Gradient descent starts at w = 0, b = 0, where the loss is ' + n$(st.loss) + '. The gradient there is (' + n$(g[0]) + ', ' + n$(g[1]) + '); the first step moves by η times minus that.';
        else if (!inside(st)) say = 'Epoch ' + e + ': the point has left the picture, loss ' + n$(st.loss) + ': each step overshoots the valley by more than the last.';
        else say = 'Epoch ' + e + ': w = ' + n$(st.w[0]) + ', b = ' + n$(st.b) + ', loss ' + n$(st.loss) + ' (the floor is ' + n$(Ls) + '); the point is ' + n$(dist) + ' from the closed form.';
        F.add({ items, say, meta: Object.assign({ phase: 'step', e, w: st.w[0], b: st.b, loss: st.loss, gw: g[0], gb: g[1], dist }, meta0) });
      }
      const sumSay = blew ? 'With η = ' + n$(lr) + ', above the limit ' + n$(lrMax) + ', every step overshoots further than the one before: gradient descent diverges, while the closed form had the answer from the start.'
        : reached ? 'After ' + E + ' epochs gradient descent is within ' + n$(gapEnd) + ' of the closed form (loss ' + n$(lastLoss) + ' against ' + n$(Ls) + '); ' + outcome + '.'
          : 'After ' + E + ' epochs the loss is ' + n$(lastLoss) + ', still above the floor ' + n$(Ls) + ', and the point is ' + n$(gapEnd) + ' away: along the flat floor of the valley the gradient is tiny, so progress there is slow.';
      F.add({ items: F.last().items.filter((it) => it.key !== 'gs:g'), say: sumSay, meta: Object.assign({ phase: 'end', gapEnd, lastLoss, outcome }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Two ways to the floor', text: ['L(w, b) = (1/2n) Σ (yᵢ − w·xᵢ − b)²', '∂L/∂w = (1/n) Σ (ŷᵢ − yᵢ) xᵢ', '∂L/∂b = (1/n) Σ (ŷᵢ − yᵢ)', 'step: θ ← θ − η ∇L', 'stable while η < 2 / λmax', 'closed form: XᵀX θ = Xᵀy'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'closed', pick: 'option', ask: 'What will gradient descent do with this learning rate?', options: ['it slides in smoothly', 'it zig-zags in', 'it is still on the way when the epochs run out', 'it blows up'], answer: (fr) => fr[0].meta.outcome, explain: (fr) => 'Compare η = ' + n$(fr[0].meta.lr) + ' with the limit ' + n$(fr[0].meta.lrMax) + ', and the epochs with how stretched the bowl is.' },
    ],
    quiz: [
      { q: 'Why do raw inputs from 0 to 10 make gradient descent slow?', options: ['the bowl becomes a long narrow valley: the safe step is set by the steep direction and crawls along the flat one', 'the gradient is zero', 'the closed form fails', 'the loss is not convex'], answer: 0, why: 'w and b are tangled when x is far from 0; the Hessian has very different eigenvalues.' },
      { q: 'For this loss, gradient descent is stable when η is below…', options: ['2 divided by the largest curvature', 'the largest curvature', '1 always', '0.01 always'], answer: 0, why: 'Along each eigen-direction the error is multiplied by (1 − η λ) per step.' },
      { q: 'What does standardising x do to the bowl?', options: ['makes it round, so one step size suits every direction', 'moves its floor to 0', 'makes it flat', 'nothing'], answer: 0, why: 'With mean 0 and variance 1 the Hessian is the identity.' },
      { q: 'When would you prefer gradient descent to the closed form?', options: ['with very many features, where solving the normal equations is costly', 'never', 'with two features', 'when the data are noisy'], answer: 0, why: 'Each epoch costs O(n·d); a solve costs O(n·d² + d³).' },
      { q: 'With η = 1 on standardised x, how many steps does it take?', options: ['about one', 'hundreds', 'it diverges', 'exactly 10'], answer: 0, why: 'With a round bowl of curvature 1, a step of 1/λ lands on the floor.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
