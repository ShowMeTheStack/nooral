// Part 8, better optimisers (1): momentum and Nesterov, which keep a velocity and so gather speed along a ravine
// floor while their zigzags cancel; and the adaptive methods AdaGrad, RMSProp and Adam, which divide each
// coordinate's step by the size of that coordinate's own gradients, shown with their accumulators.
(function (S) {
  'use strict';
  const P = S.P08, H = () => S.Kit.ml;
  const boxFor = (surf, x, y, w, hmax) => { const r = (surf.yr[1] - surf.yr[0]) / (surf.xr[1] - surf.xr[0]); const h = Math.min(hmax, w * r); return { x, y, w: h / r, h }; };
  const pickSteps = (N, first) => [...new Set([...Array(first + 1).keys()].concat([8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 70, 80]).filter((k) => k <= N).concat([N]))];

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'opt-momentum', family: 'p08-optimisers', order: 10, title: 'Momentum and Nesterov',
    blurb: 'Momentum keeps a velocity: each step adds the new gradient to β times the old velocity and moves by the learning rate times the velocity. Along a ravine floor the gradients agree, so the velocity grows; across it they alternate in sign and cancel. Nesterov’s version takes its step from where the velocity is about to carry it, so it brakes earlier before a wall.',
    inputs: [
      { id: 'surf', kind: 'choice', label: 'Surface', def: 'ravine', options: P.surfOptions(['ravine', 'tilted', 'rosen']) },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 0.1, min: 0.001, max: 0.5, step: 0.001, log: true, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'beta', kind: 'slider', label: 'Momentum β', def: 0.8, min: 0, max: 0.98, step: 0.01, fmt: (v) => String(+v.toFixed(2)) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 40, min: 5, max: 80 },
    ],
    presets: [
      { label: 'Ravine, β = 0.8', values: { surf: 'ravine', lr: 0.1, beta: 0.8, steps: 40 } },
      { label: 'Ravine, β = 0.95: overshoots', values: { surf: 'ravine', lr: 0.1, beta: 0.95, steps: 60 } },
      { label: 'Tilted valley', values: { surf: 'tilted', lr: 0.1, beta: 0.7, steps: 30 } },
      { label: 'Banana valley', values: { surf: 'rosen', lr: 0.004, beta: 0.9, steps: 80 } },
      { label: 'β = 0: plain descent', values: { surf: 'ravine', lr: 0.1, beta: 0, steps: 30 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF[inp.surf], N = inp.steps, F = H().frames(['step']), q = F.q, lr = inp.lr, b = inp.beta;
      const runs = [
        Object.assign({ id: 'gd', name: 'gradient descent', cls: 'o-gd' }, P.run(surf, 'sgd', { lr }, N)),
        Object.assign({ id: 'mom', name: 'momentum', cls: 'o-mom' }, P.run(surf, 'sgd', { lr, momentum: b }, N)),
        Object.assign({ id: 'nes', name: 'Nesterov', cls: 'o-nest' }, P.run(surf, 'sgd', { lr, momentum: b, nesterov: true }, N)),
      ];
      const bx = boxFor(surf, 40, 30, 430, 430), ax = P.axes('mo', surf, bx, { xticks: 1, yticks: 1 });
      let lo = Infinity, hi = 0;
      const off = Math.max(0, -P.gridMin(surf));
      for (const r of runs) for (const v of r.loss) if (Number.isFinite(v) && v + off > 0 && v < 1e6) { lo = Math.min(lo, v + off); hi = Math.max(hi, v + off); }
      const lax = P.logAxes('ml', { x: 545, y: 50, w: 225, h: 150 }, N, Math.max(1e-10, lo), hi);
      const at = pickSteps(N, 4);
      at.forEach((k) => {
        F.set('step', k);
        const items = [...ax.items, ...P.contour(ax, 'mo:c', surf), P.start(ax, 'mo:s', surf.start), ...lax.items];
        for (const r of runs) items.push(...P.path(ax, 'mo:' + r.id, r.pts, r.cls + (r.id === 'gd' ? ' thin' : ''), Math.min(k, r.pts.length - 1), { r: 5 }), P.logLine(lax, 'ml:' + r.id, r.loss.map((v) => v + off), r.cls, k));
        const L = (r) => r.loss[Math.min(k, r.loss.length - 1)];
        items.push(...P.legend('lg', 545, 262, runs.map((r) => ({ text: r.name + ': L = ' + H().sg(L(r), 3), cls: r.cls }))));
        // the velocity of momentum after this step: where it would carry the point next, before any new gradient
        const m = runs[1], vel = k > 0 && k <= m.state.length ? m.state[k - 1].v : null;
        if (vel && !P.off(ax, m.pts, k)) {
          const p = m.pts[k], a = [ax.X(p[0]), ax.Y(p[1])], e = [ax.X(p[0] - lr * b * vel[0]), ax.Y(p[1] - lr * b * vel[1])];
          const it = P.arrow('mo:v', a, e, 'vel', { head: 9 });
          if (it) items.push(it);
        }
        items.push(...H().lines('rd', 545, 360, [['η = ' + H().sg(lr, 3) + ', β = ' + H().sg(b, 3), 'strong'], ['blue arrow: the coast −ηβv', 'legend']], 13, 21));
        let say;
        if (k === 0) say = `Three runs from the same start with η = ${q.sg(lr, 3)}: plain gradient descent, momentum and Nesterov, both with β = ${q.sg(b, 3)}. The velocity starts as the first gradient; after that, v ← βv + ∇L and θ ← θ − ηv.`;
        else if (k <= 4 && vel) say = `Step ${q.n(k)}: momentum’s velocity is (${q.sg(vel[0], 3)}, ${q.sg(vel[1], 3)}). The blue arrow is where it would coast on its own, −ηβv. Losses: plain ${q.sg(L(runs[0]), 3)}, momentum ${q.sg(L(runs[1]), 3)}, Nesterov ${q.sg(L(runs[2]), 3)}.`;
        else {
          const best = runs.slice().sort((x, y) => L(x) - L(y))[0];
          say = `After ${q.n(k)} steps: plain ${q.sg(L(runs[0]), 3)}, momentum ${q.sg(L(runs[1]), 3)}, Nesterov ${q.sg(L(runs[2]), 3)}; ${best.name} is lowest.`;
          if (vel) say += ` Momentum’s velocity is now (${q.sg(vel[0], 3)}, ${q.sg(vel[1], 3)}).`;
          if (k === N) say += b === 0 ? ` With β = ${q.sg(b, 3)} there is no memory, and all three are the same walk.` : runs.some((r) => r.diverged) ? ' Too much speed for this surface: a run has flown off.' : ' Memory pays where the gradients agree from step to step, and costs a few overshoots where they swing.';
        }
        F.push(items, say, { phase: k === 0 ? 'start' : k === N ? 'end' : 'step', k, lr, beta: b, vel, losses: runs.map(L) });
      });
      // on a phone the whole picture, so the legend that names the colours stays in view
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Momentum', text: [surf.formula, 'momentum: v ← βv + ∇L,  θ ← θ − ηv', 'Nesterov: v ← βv + ∇L,  θ ← θ − η(∇L + βv)', 'the first velocity is the first gradient', 'agreeing gradients add up: up to 1/(1 − β) times one step'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which run will have the lowest loss at the end?', options: ['gradient descent', 'momentum', 'Nesterov'], answer: (fr) => ['gradient descent', 'momentum', 'Nesterov'][fr[fr.length - 1].meta.losses.indexOf(Math.min(...fr[fr.length - 1].meta.losses))], explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'With β = 0.9 and a constant gradient g, momentum’s velocity grows towards…', options: ['10g', 'g', '0.9g', '0.1g'], answer: 0, why: 'v = g + 0.9g + 0.81g + … = g/(1 − 0.9).' },
      { q: 'Why does momentum help in a ravine?', options: ['the alternating cross-ravine gradients cancel while the along-floor ones add up', 'it lowers the learning rate', 'it ignores the gradient', 'it finds a different minimum'], answer: 0, why: 'Averaging over steps keeps the consistent part of the gradient and damps the part that flips sign.' },
      { q: 'What does Nesterov change?', options: ['it uses the gradient together with where the velocity is about to carry it', 'it removes the velocity', 'it doubles the learning rate', 'it uses second derivatives'], answer: 0, why: 'Looking ahead lets it slow down before it overshoots.' },
      { q: 'What can too large a β cause?', options: ['big overshoots and slow, swinging convergence', 'the gradient to vanish', 'a smaller step', 'nothing at all'], answer: 0, why: 'Too much memory keeps the point moving long after the gradient has turned.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  const ADA = [
    { id: 'adagrad', name: 'AdaGrad', cls: 'o-ada', acc: 'sumSq', accName: 'sum of g²' },
    { id: 'rmsprop', name: 'RMSProp', cls: 'o-rms', acc: 'sqAvg', accName: 'average of g²' },
    { id: 'adam', name: 'Adam', cls: 'o-adam', acc: 'v', accName: 'average of g² (v)' },
  ];
  S.topic({
    id: 'opt-adaptive', family: 'p08-optimisers', order: 20, title: 'AdaGrad, RMSProp and Adam: a step size for each coordinate',
    blurb: 'In a ravine one coordinate’s gradients are large and the other’s small, and one learning rate cannot suit both. The adaptive methods keep, per coordinate, a running measure of the squared gradients and divide that coordinate’s step by its square root. AdaGrad sums every g² it has seen, so its steps only shrink; RMSProp keeps a decaying average; Adam adds momentum and corrects both averages for starting at zero.',
    inputs: [
      { id: 'method', kind: 'choice', label: 'Look inside', def: 'adam', options: ADA.map((a) => ({ value: a.id, label: a.name })) },
      { id: 'surf', kind: 'choice', label: 'Surface', def: 'ravine', options: P.surfOptions(['ravine', 'tilted', 'rosen', 'plateau']) },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 0.3, min: 0.01, max: 1, step: 0.01, log: true, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 40, min: 5, max: 80 },
    ],
    presets: [
      { label: 'Adam in a ravine', values: { method: 'adam', surf: 'ravine', lr: 0.3, steps: 40 } },
      { label: 'AdaGrad: steps that shrink', values: { method: 'adagrad', surf: 'ravine', lr: 0.3, steps: 60 } },
      { label: 'RMSProp in a ravine', values: { method: 'rmsprop', surf: 'ravine', lr: 0.1, steps: 40 } },
      { label: 'Off a plateau', values: { method: 'adam', surf: 'plateau', lr: 0.2, steps: 40 } },
      { label: 'Banana valley', values: { method: 'rmsprop', surf: 'rosen', lr: 0.05, steps: 80 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF[inp.surf], N = inp.steps, F = H().frames(['step']), q = F.q, lr = inp.lr;
      const gdLr = Math.min(lr, 1.8 / P.eig2(surf.h(...surf.start))[0]);
      const runs = [Object.assign({ id: 'gd', name: 'gradient descent', cls: 'o-gd', lr: gdLr }, P.run(surf, 'sgd', { lr: gdLr }, N))];
      for (const a of ADA) runs.push(Object.assign({}, a, { lr }, P.run(surf, a.id, { lr }, N)));
      const me = runs.find((r) => r.id === inp.method), meta0 = ADA.find((a) => a.id === inp.method);
      const bx = boxFor(surf, 40, 30, 420, 420), ax = P.axes('ad', surf, bx, { xticks: 1, yticks: 1 });
      const at = pickSteps(N, 4);
      // the bars: per coordinate, the gradient's size, the accumulator and the step size it allows
      const BX = 530;
      const bars = (k) => {
        const items = [H().head('bh', BX, 44, meta0.name + ' inside, after step ' + k)];
        if (!k) return items;
        const st = me.state[k - 1], g = me.grads[k - 1], acc = st[meta0.acc], t = k;
        const eff = [0, 1].map((j) => { const d = me.pts[k][j] - me.pts[k - 1][j]; return Math.abs(g[j]) > 1e-15 ? Math.abs(d / g[j]) : 0; });
        const rows = [['|g|', g.map(Math.abs)], ['√' + (meta0.id === 'adagrad' ? 'sum' : 'avg') + ' g²', acc.map((v, j) => Math.sqrt(meta0.id === 'adam' ? v / (1 - Math.pow(0.999, t)) : v))], ['step / |g|', eff]];
        rows.forEach(([name, vals], r) => {
          const y0 = 76 + r * 64, mx = Math.max(...vals, 1e-12);
          items.push(H().text('bn' + r, BX, y0 + 12, name, 'legend', 13));
          vals.forEach((v, j) => {
            const w = Math.max(2, (130 * v) / mx), y = y0 + 20 + j * 20;
            items.push(H().text('bc' + r + j, BX + 4, y + 12, j ? 'v' : 'u', 'axname', 12));
            items.push({ key: 'bb' + r + j, kind: 'cell', x: BX + 22, y, w, h: 14, rx: 2, text: '', cls: 'p8-bar ' + (j ? 'v' : 'u'), audit: 'ignore' });
            items.push(H().text('bv' + r + j, BX + 28 + w, y + 12, H().sg(v, 3), 'legend', 12));
          });
        });
        return items;
      };
      at.forEach((k) => {
        F.set('step', k);
        const items = [...ax.items, ...P.contour(ax, 'ad:c', surf), P.start(ax, 'ad:s', surf.start)];
        for (const r of runs) items.push(...P.path(ax, 'ad:' + r.id, r.pts, r.cls + (r.id === inp.method ? '' : ' thin'), Math.min(k, r.pts.length - 1), { r: r.id === inp.method ? 6 : 4 }));
        const L = (r) => r.loss[Math.min(k, r.loss.length - 1)];
        items.push(...bars(k));
        items.push(...P.legend('lg', BX, 300, runs.map((r) => ({ text: r.name + ': L = ' + H().sg(L(r), 3), cls: r.cls }))));
        let say, meta = { phase: k === 0 ? 'start' : k === N ? 'end' : 'step', k, lr, gdLr, losses: runs.map(L) };
        if (k === 0) say = `${me.name} with η = ${q.sg(lr, 3)}, beside the other two adaptive methods and plain gradient descent (${gdLr < lr ? `η = ${q.sg(gdLr, 3)}, as large as this surface allows it` : `the same η = ${q.sg(gdLr, 3)}`}). Watch the bars: for each coordinate, the gradient, the root of the running g² it is divided by, and the step that results per unit of gradient.`;
        else {
          const st = me.state[k - 1], g = me.grads[k - 1], d = [me.pts[k][0] - me.pts[k - 1][0], me.pts[k][1] - me.pts[k - 1][1]];
          Object.assign(meta, { g, d, acc: st[meta0.acc], m: st.m });
          if (k === 1) {
            const gr = Math.max(Math.abs(g[0]), Math.abs(g[1])) / Math.max(1e-300, Math.min(Math.abs(g[0]), Math.abs(g[1])));
            Object.assign(meta, { gr });
            say = `Step 1: the gradient is (${q.sg(g[0], 3)}, ${q.sg(g[1], 3)}): one coordinate’s is ${q.sg(gr, 3)} times the other’s. Yet ${me.name}’s step is (${q.sg(d[0], 3)}, ${q.sg(d[1], 3)}): divided by its own size, each coordinate moves about η. The flat coordinate is no longer starved. RMSProp’s first step is ${q.sg(P.dist(runs[2].pts[1], runs[2].pts[0]) / P.dist(runs[3].pts[1], runs[3].pts[0]), 3)} times as long as Adam’s: its average of g² starts at zero and nothing corrects for that.`;
          Object.assign(meta, { ratio: P.dist(runs[2].pts[1], runs[2].pts[0]) / P.dist(runs[3].pts[1], runs[3].pts[0]) });
          } else if (k < N) say = `Step ${q.n(k)}: gradient (${q.sg(g[0], 3)}, ${q.sg(g[1], 3)}), step (${q.sg(d[0], 3)}, ${q.sg(d[1], 3)}). Losses: gradient descent ${q.sg(L(runs[0]), 3)}, AdaGrad ${q.sg(L(runs[1]), 3)}, RMSProp ${q.sg(L(runs[2]), 3)}, Adam ${q.sg(L(runs[3]), 3)}.`;
          else say = `After ${q.n(k)} steps: gradient descent ${q.sg(L(runs[0]), 3)}, AdaGrad ${q.sg(L(runs[1]), 3)}, RMSProp ${q.sg(L(runs[2]), 3)}, Adam ${q.sg(L(runs[3]), 3)}. ` + (inp.method === 'adagrad' ? `AdaGrad’s sums only grow (now ${q.sg(st.sumSq[0], 3)} and ${q.sg(st.sumSq[1], 3)}), so its steps keep shrinking even where the ground is flat.` : inp.method === 'rmsprop' ? `RMSProp’s averages forget old gradients (now ${q.sg(st.sqAvg[0], 3)} and ${q.sg(st.sqAvg[1], 3)}), so its steps stay lively; near the bottom they rattle at about η.` : `Adam’s first moment m = (${q.sg(st.m[0], 3)}, ${q.sg(st.m[1], 3)}) is its momentum; dividing by √v makes the step about η wherever the gradients are steady.`);
        }
        F.push(items, say, meta);
      });
      // on a phone the whole picture, so the legend that names the colours stays in view
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Adaptive steps (each coordinate on its own)', text: ['AdaGrad: h ← h + g²,  θ ← θ − η g / (√h + ε)', 'RMSProp: s ← ρs + (1 − ρ)g²,  θ ← θ − η g / (√s + ε), ρ = 0.99', 'Adam: m ← β₁m + (1 − β₁)g,  v ← β₂v + (1 − β₂)g²', '  m̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ),  θ ← θ − η m̂/(√v̂ + ε)', '  β₁ = 0.9, β₂ = 0.999'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'On the first step, how big will the adaptive step be in each coordinate?', options: ['about η in both, whatever the gradient', 'η times the gradient', 'zero in the flat coordinate', 'twice η'], answer: 'about η in both, whatever the gradient', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'AdaGrad’s sum of squared gradients in one coordinate is 16, counting the current gradient, which is 2. With η = 0.5, how long is the step in that coordinate (ignore ε)?', options: ['0.25', '1', '0.5', '0.125'], answer: 0, why: 'η·g/√h = 0.5 × 2 / 4 = 0.25.' },
      { q: 'Why can AdaGrad slow to a crawl on a long run?', options: ['its sum of squared gradients only grows, so its steps only shrink', 'it forgets its gradients', 'it has no learning rate', 'it uses momentum'], answer: 0, why: 'RMSProp fixes this with a decaying average.' },
      { q: 'What does Adam’s bias correction fix?', options: ['the moving averages start at zero and are too small in the first steps', 'the gradient sign', 'a learning rate that is too large', 'overfitting'], answer: 0, why: 'Dividing by 1 − βᵗ scales the early averages up to what they estimate.' },
      { q: 'Why do adaptive methods help in a ravine?', options: ['each coordinate’s step is divided by the size of its own gradients', 'they take the same step in every direction always', 'they compute the Hessian', 'they lower the loss directly'], answer: 0, why: 'The steep coordinate is slowed and the flat one sped up.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
