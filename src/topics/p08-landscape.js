// Part 8, the shape of the landscape: convexity by the chord test and the second derivative, with descent from
// two starts; local minima and their basins on a bumpy surface, found by starting descent from many points; and
// a plateau, where the gradient is so small that plain descent hardly moves.
(function (S) {
  'use strict';
  const P = S.P08, H = () => S.Kit.ml;

  // ---- one-dimensional losses ------------------------------------------------------------------------------------
  P.F1 = {
    square: { label: 'a parabola', formula: 'L(x) = ½(x − 0.5)²', f: (x) => 0.5 * (x - 0.5) ** 2, d: (x) => x - 0.5, d2: () => 1 },
    logcosh: { label: 'log cosh, tilted', formula: 'L(x) = log cosh(x) + 0.3x', f: (x) => Math.log(Math.cosh(x)) + 0.3 * x, d: (x) => Math.tanh(x) + 0.3, d2: (x) => 1 / Math.cosh(x) ** 2 },
    double: { label: 'a double well', formula: 'L(x) = x⁴/4 − x² + 0.3x', f: (x) => x ** 4 / 4 - x * x + 0.3 * x, d: (x) => x ** 3 - 2 * x + 0.3, d2: (x) => 3 * x * x - 2 },
    wavy: { label: 'a wavy bowl', formula: 'L(x) = 0.15x² + 0.5 sin 3x', f: (x) => 0.15 * x * x + 0.5 * Math.sin(3 * x), d: (x) => 0.3 * x + 1.5 * Math.cos(3 * x), d2: (x) => 0.3 - 4.5 * Math.sin(3 * x) },
  };
  const CHORDS = [[-2.6, 2.4], [-2, 0.4], [-1.2, 1.5], [0.3, 2.5]];
  /** the largest amount by which the curve rises above its chord from a to b, and where (on a fine grid) */
  P.chordGap = function (f, a, b) {
    let best = -Infinity, at = a;
    for (let i = 1; i < 2000; i++) { const t = i / 2000, x = a + (b - a) * t, g = f(x) - (f(a) + (f(b) - f(a)) * t); if (g > best) { best = g; at = x; } }
    return { gap: best, at };
  };
  /** a 1-D loss as a surface in (u, v), flat in v, so the engine's optimisers can walk it */
  P.as2d = (F1) => ({ f: (u) => F1.f(u), g: (u) => [F1.d(u), 0], h: (u) => [[F1.d2(u), 0], [0, 0]], xr: [-3, 3], yr: [-1, 1] });

  S.topic({
    id: 'opt-convex', family: 'p08-landscape', order: 10, title: 'Convex or not: the chord test',
    blurb: 'A loss is convex when every chord between two points of its graph lies on or above the graph, or, for a smooth loss, when its second derivative is never negative. Then any point where the slope is zero is the lowest, and descent from anywhere ends there. A loss that fails the test can trap descent in a dip that is not the deepest.',
    inputs: [{ id: 'fn', kind: 'choice', label: 'Loss', def: 'double', options: Object.keys(P.F1).map((k) => ({ value: k, label: P.F1[k].label })) }],
    presets: [
      { label: 'A parabola', values: { fn: 'square' } },
      { label: 'log cosh: convex but not a parabola', values: { fn: 'logcosh' } },
      { label: 'A double well', values: { fn: 'double' } },
      { label: 'A wavy bowl', values: { fn: 'wavy' } },
    ],
    counters: [{ id: 'chords', label: 'chords tried' }],
    run(inp) {
      const L = P.F1[inp.fn], F = H().frames(['chords']), q = F.q;
      const xs = [...Array(241).keys()].map((i) => -3 + i / 40), ys = xs.map(L.f);
      const ylo = Math.min(...ys), yhi = Math.max(...ys), pad = (yhi - ylo) * 0.08;
      const ax = S.Kit.curve.axes({ prefix: 'cv', x: 50, y: 40, w: 430, h: 270, xr: [-3, 3], yr: [ylo - pad, yhi + pad], xticks: 1, yticks: S.Kit.curve.ticks(ylo, yhi, 4), xname: 'x', yname: 'L(x)' });
      const d2s = xs.map(L.d2), dlo = Math.min(0, ...d2s), dhi = Math.max(0.5, ...d2s);
      const dax = S.Kit.curve.axes({ prefix: 'dv', x: 50, y: 370, w: 430, h: 90, xr: [-3, 3], yr: [dlo - 0.1, dhi + 0.1], xticks: 1, yticks: [0], xname: 'x', yname: 'L″(x), the curvature' });
      const curve = () => [...ax.items, S.Kit.curve.line(ax, 'cv:f', xs.map((x, i) => [x, ys[i]]), '')];
      const RX = 540;
      const chordItems = (k) => {
        const [a, b] = CHORDS[k], g = P.chordGap(L.f, a, b), items = [];
        items.push({ key: 'cv:ch', kind: 'path', d: `M${ax.X(a)},${ax.Y(L.f(a))} L${ax.X(b)},${ax.Y(L.f(b))}`, cls: 'p8-chord ' + (g.gap > 1e-9 ? 'bad' : 'good'), pathAudit: 'edge-x', layer: 'lines', box: H().boxOf([[ax.X(a), ax.Y(L.f(a))], [ax.X(b), ax.Y(L.f(b))]]) });
        items.push(S.Kit.curve.dot(ax, 'cv:a', a, L.f(a), '', 5), S.Kit.curve.dot(ax, 'cv:b', b, L.f(b), '', 5));
        if (g.gap > 1e-9) items.push(S.Kit.curve.dot(ax, 'cv:at', g.at, L.f(g.at), 'c1', 5));
        return { items, g, a, b };
      };
      let bad = 0;
      F.push([...curve(), ...H().lines('rd', RX, 60, [[L.label, 'strong'], [L.formula, 'legend']], 13, 22)], `The loss: ${L.label}, whose formula is in the definition. Is it convex? Draw chords: straight lines between two points of the graph. In a convex loss every chord lies on or above the graph between its ends.`, { phase: 'start' });
      CHORDS.forEach((_, k) => {
        F.count('chords');
        const c = chordItems(k);
        if (c.g.gap > 1e-9) bad++;
        const say = c.g.gap > 1e-9
          ? `The chord from x = ${q(c.a, 1)} to x = ${q(c.b, 1)} dips below the graph: at x = ${q(c.g.at)} the curve is ${q.sg(c.g.gap, 3)} above the chord (orange dot). One such chord is enough: this loss is not convex.`
          : `The chord from x = ${q(c.a, 1)} to x = ${q(c.b, 1)} stays on or above the graph all the way: nowhere between its ends does the curve rise above it.`;
        F.push([...curve(), ...c.items, ...H().lines('rd', RX, 60, [[L.label, 'strong'], [L.formula, 'legend'], ['chords failing: ' + bad + ' of ' + (k + 1), bad ? 'bad' : 'good']], 13, 22)], say, { phase: 'chord', k, a: c.a, b: c.b, gap: c.g.gap, at: c.g.at, bad });
      });
      // the second derivative
      const mn = Math.min(...d2s), mi = d2s.indexOf(mn);
      const d2Items = () => [...dax.items, S.Kit.curve.hline(dax, 'dv:0', 0, ''), S.Kit.curve.line(dax, 'dv:f', xs.map((x, i) => [x, d2s[i]]), mn < 0 ? 'o-rms' : 'o-adam')];
      F.push([...curve(), ...d2Items(), ...H().lines('rd', RX, 60, [[L.label, 'strong'], [L.formula, 'legend'], ['least curvature ' + H().sg(mn, 3), mn < 0 ? 'bad' : 'good']], 13, 22)],
        mn < -1e-12 ? `The test by curvature: L″ dips below zero, as low as ${q.sg(mn, 3)} at x = ${q(xs[mi])}. Where the curvature is negative the graph bends down, like a hilltop, and chords across it fall under the graph.` : `The test by curvature: L″ is never negative; its least value on the plot is ${q.sg(mn, 3)}. The graph bends up everywhere, so every chord lies above it: convex.`, { phase: 'd2', min: mn, at: xs[mi] });
      // descent from both ends
      const surf = P.as2d(L), starts = [-2.8, 2.8], lr = 0.6 / Math.max(...d2s.map(Math.abs));
      const runs = starts.map((s0) => P.run(surf, 'sgd', { lr }, 300, { start: [s0, 0] }));
      const ends = runs.map((r) => r.pts[r.pts.length - 1][0]), fe = ends.map(L.f);
      const runItems = [...curve(), ...d2Items()];
      runs.forEach((r, j) => { const pts = r.pts.filter((_, t) => t % 6 === 0 || t === r.pts.length - 1); pts.forEach((p, t) => runItems.push({ key: 'cv:r' + j + '_' + t, kind: 'dot', x: ax.X(p[0]), y: ax.Y(L.f(p[0])), r: t === pts.length - 1 ? 6 : 2.6, cls: (t === pts.length - 1 ? 'opthead ' : 'optstep ') + (j ? 'o-rms' : 'o-mom'), audit: 'overlap-ok' })); });
      const same = Math.abs(ends[0] - ends[1]) < 1e-3;
      // the lowest point of all: the best grid point, polished by Newton's method
      let gx = xs[ys.indexOf(Math.min(...ys))];
      for (let it = 0; it < 50; it++) gx -= L.d(gx) / L.d2(gx);
      const found = ends.findIndex((e) => Math.abs(e - gx) < 1e-3);
      runItems.push(...H().lines('rd', RX, 60, [[L.label, 'strong'], ['from the left: x = ' + H().fx(ends[0]), 'legend'], ['from the right: x = ' + H().fx(ends[1]), 'legend']], 13, 22));
      F.push(runItems, `Gradient descent with η = ${q.sg(lr, 3)} for ${q.n(300)} steps from x = ${q(starts[0], 1)} and from x = ${q(starts[1], 1)}. ` + (same ? `Both end at x = ${q(ends[0])}, loss ${q.sg(fe[0], 3)}: ${mn < -1e-12 ? 'here they happen to agree.' : 'on a convex loss every way down leads to the one bottom.'}` : `They stop in different dips, at x = ${q(ends[0])} (loss ${q.sg(fe[0], 3)}) and x = ${q(ends[1])} (loss ${q.sg(fe[1], 3)}). The slope is zero at both. The lowest point of all is at x = ${q(gx)} (loss ${q.sg(L.f(gx), 3)}): ${found < 0 ? 'neither run found it' : 'only the run from the ' + (found ? 'right' : 'left') + ' reached it'}. A local minimum looks like the global one from inside it.`), { phase: 'runs', lr, ends, fe, same, starts, steps: 300, gx, gf: L.f(gx) });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 500, h: 480 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Convexity', text: [L.formula, 'convex: L(ta + (1 − t)b) ≤ tL(a) + (1 − t)L(b) for all a, b and t in [0, 1]', 'for a smooth L: L″(x) ≥ 0 everywhere (in many dimensions: the Hessian has no negative eigenvalue)', 'convex: every point with zero slope is a global minimum'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will some chord fall below the graph?', options: ['yes: the loss is not convex', 'no: the loss is convex'], answer: (fr) => (fr.find((f) => f.meta.phase === 'd2').meta.min < -1e-12 ? 'yes: the loss is not convex' : 'no: the loss is convex'), explain: (fr) => fr.find((f) => f.meta.phase === 'd2').say },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'runs', pick: 'option', ask: 'Descent starts from both ends. Will both runs end at the same point?', options: ['yes', 'no'], answer: (fr, i) => (fr[i + 1].meta.same ? 'yes' : 'no'), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'Which of these is convex?', options: ['L(x) = x²', 'L(x) = −x²', 'L(x) = sin x', 'L(x) = x³'], answer: 0, why: 'Its second derivative is 2, never negative.' },
      { q: 'On a convex loss, gradient descent with a small enough step…', options: ['reaches the global minimum from any start', 'can stop in a local minimum', 'never converges', 'needs momentum to converge'], answer: 0, why: 'A convex loss has no local minimum that is not global.' },
      { q: 'L(x) = x⁴/4 − x². What is L″(0)?', options: ['−2', '0', '2', '1'], answer: 0, why: 'L″ = 3x² − 2, which is −2 at 0: the graph bends down there.' },
      { q: 'Are the losses of neural networks convex in their weights?', options: ['no, almost never', 'yes, always', 'only with ReLU', 'only for one layer'], answer: 0, why: 'Swapping two hidden units gives the same loss, so there are many equal minima with higher ground between: not convex. Linear and logistic regression are convex.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'opt-local', family: 'p08-landscape', order: 20, title: 'Local minima and their basins',
    blurb: 'On a bumpy loss, gradient descent ends in whichever pit it starts above: every pit has a basin, the set of starts that drain into it. Starting from many points shows the basins, and keeping the best end is the simplest defence against a poor pit. Wider basins catch more starts, whether or not their pit is the deepest.',
    inputs: [
      { id: 'k', kind: 'number', label: 'Starts per side', def: 4, min: 2, max: 7 },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 0.3, min: 0.05, max: 0.8, step: 0.01, fmt: (v) => String(+v.toPrecision(2)) },
    ],
    presets: [
      { label: '16 starts', values: { k: 4, lr: 0.3 } },
      { label: '4 starts', values: { k: 2, lr: 0.3 } },
      { label: '49 starts', values: { k: 7, lr: 0.3 } },
      { label: 'A large rate', values: { k: 5, lr: 0.75 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF.bumpy, F = H().frames(['step']), q = F.q, N = 120, lr = inp.lr, K = inp.k;
      // the pits: every well's bottom, found by Newton's method from the well's centre
      const pits = [];
      for (const c of [[-1.3, -1.2], [1.6, 1.1], [-1.4, 1.7], [1.5, -1.6]]) { const m = P.newtonMin(surf, c); if (!pits.some((p) => P.dist(p, m) < 1e-3)) pits.push(m); }
      pits.sort((a, b) => surf.f(...a) - surf.f(...b));
      const pitOf = (p) => { let b = -1, bd = 0.35; pits.forEach((m, j) => { const d = P.dist(p, m); if (d < bd) { bd = d; b = j; } }); return b; };
      const starts = [];
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) starts.push([-2.6 + (5.2 * (i + 0.5)) / K, -2.6 + (5.2 * (j + 0.5)) / K]);
      const runs = starts.map((s0) => P.run(surf, 'sgd', { lr }, N, { start: s0 }));
      const ends = runs.map((r) => r.pts[r.pts.length - 1]), basin = ends.map(pitOf);
      const bx = { x: 40, y: 30, w: 430, h: 430 }, ax = P.axes('lm', surf, bx, { xticks: 1, yticks: 1 });
      // the basins: descent from the centre of every cell of a grid, coloured by the pit it ends in
      const map = S.Kit.scatter.field(ax, 'lm:b', (u, v) => { const r = P.run(surf, 'sgd', { lr }, 90, { start: [u, v] }), j = pitOf(r.pts[r.pts.length - 1]); return [Math.max(0, j), j < 0 ? 0 : 0.8]; }, { n: 30, scheme: 'classes', smooth: false });
      const counts = pits.map((_, j) => basin.filter((b) => b === j).length), share = map.vals.map((v, i) => (map.vals2[i] > 0 ? v : -1));
      const area = pits.map((_, j) => share.filter((v) => v === j).length / share.length);
      const contour = P.contour(ax, 'lm:c', surf, { levels: 12 });
      const pitDots = pits.map((m, j) => ({ key: 'lm:pit' + j, kind: 'dot', x: ax.X(m[0]), y: ax.Y(m[1]), r: 7, cls: 'p8-pit c' + j, audit: 'overlap-ok', layer: 'top' }));
      const RX = 530, table = (shown) => {
        const items = [H().head('tb', RX, 44, 'The pits, deepest first')];
        pits.forEach((m, j) => {
          const y = 76 + j * 26;
          items.push({ key: 'tb:s' + j, kind: 'dot', x: RX + 7, y: y - 4.5, r: 6, cls: 'swatch c' + j });
          items.push(H().text('tb:t' + j, RX + 20, y, 'L = ' + H().sg(surf.f(...m), 3) + (shown ? ':  ' + counts[j] + ' of ' + starts.length + ' starts' : ''), 'legend', 13));
        });
        return items;
      };
      F.push([...ax.items, ...contour, ...pitDots, ...starts.map((s0, i) => P.start(ax, 'lm:s' + i, s0)), ...table(false)], `A bumpy loss with ${q.n(pits.length)} pits, the deepest at (${q(pits[0][0])}, ${q(pits[0][1])}) with loss ${q.sg(surf.f(...pits[0]), 3)}. Gradient descent with η = ${q.sg(lr, 3)} sets off from each of ${q.n(starts.length)} starts on a grid.`, { phase: 'start', pits, npits: pits.length, lr, n: starts.length, fp: pits.map((m) => surf.f(...m)) });
      for (const k of [5, 15, 40, N]) {
        F.set('step', k);
        const items = [...ax.items, ...contour, ...pitDots];
        runs.forEach((r, i) => items.push(...P.path(ax, 'lm:r' + i, r.pts, 'o-gd thin', k, { r: 4 })));
        items.push(...table(k === N));
        const at = runs.map((r) => pitOf(r.pts[Math.min(k, r.pts.length - 1)])), settled = at.filter((b) => b >= 0).length;
        F.push(items, k < N ? `After ${q.n(k)} steps, ${q.n(settled)} of the ${q.n(starts.length)} runs are already within reach of a pit’s bottom.` : `After ${q.n(N)} steps every run has settled. ${H().list(counts.map((c, j) => q.n(c) + ' in the pit with loss ' + q.sg(surf.f(...pits[j]), 3)))}. Only ${q.n(counts[0])} found the deepest pit; the best of all the runs has loss ${q.sg(Math.min(...ends.map((e) => surf.f(...e))), 3)}.`, { phase: k === N ? 'end' : 'step', k, n: starts.length, settled, counts, best: Math.min(...ends.map((e) => surf.f(...e))), fp: pits.map((m) => surf.f(...m)) });
      }
      const items = [...ax.items, ...contour, Object.assign({}, map, { z: 1 }), ...pitDots];
      runs.forEach((r, i) => items.push(...P.path(ax, 'lm:r' + i, r.pts, 'o-gd thin', N, { r: 4 })));
      items.push(...table(true), ...H().lines('ar', RX, 200, [['share of the map', 'strong']].concat(pits.map((m, j) => ['pit ' + (j + 1) + ': ' + Math.round(area[j] * 100) + '%', j === 3 ? 'p8c3' : 'c' + j])), 13, 22));
      F.push(items, `The colours are the basins: every place coloured like a pit drains into it. The deepest pit’s basin covers ${q.pct(area[0])} of the map${area.indexOf(Math.max(...area)) === 0 ? ', the largest share' : `, while the largest basin, ${q.pct(Math.max(...area))}, belongs to a shallower pit`}. A start lands in the deepest pit only if it begins inside that basin.`, { phase: 'basins', area });
      // on a phone the whole picture, so the legend that names the colours stays in view
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Local minima', text: [surf.formula, 'local minimum: lower than everything near it', 'global minimum: lower than everything', 'basin: the starts from which descent reaches a given minimum', 'many starts, keep the best: random restarts'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will most of the runs end in the deepest pit?', options: ['yes', 'no'], answer: (fr) => { const e = fr.find((f) => f.meta.phase === 'end').meta; return e.counts[0] * 2 > e.counts.reduce((a, b) => a + b, 0) ? 'yes' : 'no'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'end').say },
    ],
    quiz: [
      { q: 'What decides which local minimum gradient descent ends in?', options: ['the basin its start lies in (and the step size)', 'the depth of the minima', 'nothing: it always finds the deepest', 'the number of minima'], answer: 0, why: 'Descent only ever goes downhill from where it is.' },
      { q: 'Ten random starts each find the deepest pit with chance 0.2, independently. What is the chance that none does?', options: ['about 0.11', '0.2', '0.8', 'about 0.5'], answer: 0, why: '0.8¹⁰ ≈ 0.107.' },
      { q: 'At a local minimum that is not global, the gradient is…', options: ['zero', 'large', 'pointing to the global minimum', 'undefined'], answer: 0, why: 'From inside its basin the point looks exactly like a bottom.' },
      { q: 'Why do momentum and noisy steps sometimes find better minima?', options: ['they can carry the point over a low ridge into another basin', 'they compute the global minimum', 'they make the loss convex', 'they never stop'], answer: 0, why: 'Speed and noise let a run cross shallow barriers that plain descent cannot.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'opt-plateau', family: 'p08-landscape', order: 30, title: 'Plateaus: when the gradient all but vanishes',
    blurb: 'Far from the pit this loss is nearly flat: its gradient is tiny, so plain gradient descent, whose step is η times the gradient, barely moves for many steps, then rushes down once the ground tilts. Momentum builds speed even on a gentle slope, and Adam, dividing by the gradient’s own size, takes full-size steps from the start.',
    inputs: [
      { id: 'u0', kind: 'slider', label: 'Start u', def: 3.6, min: 2, max: 3.9, step: 0.1, fmt: (v) => v.toFixed(1) },
      { id: 'lr', kind: 'slider', label: 'Learning rate η', def: 1, min: 0.1, max: 1.5, step: 0.05, fmt: (v) => String(+v.toPrecision(2)) },
      { id: 'steps', kind: 'number', label: 'Steps', def: 120, min: 30, max: 300 },
    ],
    presets: [
      { label: 'Far out on the plateau', values: { u0: 3.6, lr: 1, steps: 120 } },
      { label: 'Nearer the edge', values: { u0: 2.4, lr: 1, steps: 60 } },
      { label: 'Very far: a long wait', values: { u0: 3.9, lr: 1, steps: 300 } },
    ],
    counters: [{ id: 'step', label: 'steps' }],
    run(inp) {
      const surf = P.SURF.plateau, N = inp.steps, F = H().frames(['step']), q = F.q, lr = inp.lr, start = [inp.u0, 0.9];
      const runs = [
        Object.assign({ id: 'gd', name: 'gradient descent', cls: 'o-gd' }, P.run(surf, 'sgd', { lr }, N, { start })),
        Object.assign({ id: 'mom', name: 'momentum (β 0.9)', cls: 'o-mom' }, P.run(surf, 'sgd', { lr, momentum: 0.9 }, N, { start })),
        Object.assign({ id: 'adam', name: 'Adam (η 0.1)', cls: 'o-adam' }, P.run(surf, 'adam', { lr: 0.1 }, N, { start })),
      ];
      for (const r of runs) { r.gn = r.grads.map((g) => Math.hypot(g[0], g[1])); r.off = r.loss.findIndex((v) => v < 0.5); }
      const bx = { x: 40, y: 30, w: 430, h: 430 * (5.2 / 8) }, ax = P.axes('pl', surf, bx, { xticks: 1, yticks: 1 });
      const gall = runs.flatMap((r) => r.gn).filter((v) => v > 0);
      const gax = P.logAxes('pg', { x: 545, y: 50, w: 225, h: 130 }, N, Math.max(1e-10, Math.min(...gall)), Math.max(...gall), { yname: 'gradient length (log)' });
      const lax = S.Kit.curve.axes({ prefix: 'pc', x: 545, y: 260, w: 225, h: 110, xr: [0, N], yr: [0, 1.05], xticks: S.Kit.curve.ticks(0, N, 4), yticks: [0, 0.5, 1], xname: 'step', yname: 'loss' });
      const AT = [...new Set([0, 5, 10, 20, 35, 50, 70, 90, 120, 160, 200, 250, 300].filter((k) => k <= N).concat([N]))];
      AT.forEach((k) => {
        F.set('step', k);
        const items = [...ax.items, ...P.contour(ax, 'pl:c', surf), P.minDot(ax, 'pl:m', [0, 0]), P.start(ax, 'pl:s', start), ...gax.items, ...lax.items];
        for (const r of runs) {
          items.push(...P.path(ax, 'pl:' + r.id, r.pts, r.cls, k, { r: 5 }));
          if (k) items.push(P.logLine(gax, 'pg:' + r.id, r.gn, r.cls, k - 1));
          items.push(S.Kit.curve.line(lax, 'pc:' + r.id, r.loss.slice(0, k + 1).map((v, t) => [t, v]), r.cls));
        }
        items.push(...P.legend('lg', 40, bx.y + bx.h + 64, runs.map((r) => ({ text: r.name, cls: r.cls })), { dir: 'row', size: 12 }));
        const L = (r) => r.loss[k], D = (r) => P.dist(r.pts[k], start);
        let say, meta = { phase: k === 0 ? 'start' : k === N ? 'end' : 'step', k, lr, start, losses: runs.map(L), moved: runs.map(D), off: runs.map((r) => r.off) };
        if (k === 0) { const g = surf.g(...start); say = `Start at (${q(start[0], 1)}, ${q(start[1], 1)}): the loss is ${q.sg(surf.f(...start), 4)}, almost its ceiling of 1, and the gradient is only ${q.sg(Math.hypot(...g), 3)} long. With η = ${q.sg(lr, 3)}, a plain step would be that small times η.`; Object.assign(meta, { g, gl: Math.hypot(...g), f0: surf.f(...start) }); }
        else if (k < N) say = `Step ${q.n(k)}: plain descent has moved ${q.sg(D(runs[0]), 3)} from the start, momentum ${q.sg(D(runs[1]), 3)}, Adam ${q.sg(D(runs[2]), 3)}. Losses ${q.sg(L(runs[0]), 3)}, ${q.sg(L(runs[1]), 3)} and ${q.sg(L(runs[2]), 3)}.`;
        else say = runs.map((r) => r.name.split(' (')[0] + (r.off >= 0 ? ' got below loss 0.5 at step ' + q.n(r.off) : ' never got below loss 0.5')).join('; ').replace(/^./, (c) => c.toUpperCase()) + `. On a plateau the size of the gradient, not the distance to the pit, sets the pace of plain descent.`;
        F.push(items, say, meta);
      });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 490, h: bx.y + bx.h + 80 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A plateau', text: [surf.formula, 'far out, |∇L| shrinks like exp(−r²/2): almost flat', 'plain descent: step = η|∇L|, tiny', 'momentum: speed builds up to 1/(1 − β) times a step', 'Adam: step ≈ η whatever the gradient’s size'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which run will leave the plateau first?', options: ['gradient descent', 'momentum', 'Adam'], answer: (fr) => { const o = fr[fr.length - 1].meta.off.map((v) => (v < 0 ? Infinity : v)); return ['gradient descent', 'momentum', 'Adam'][o.indexOf(Math.min(...o))]; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'On a plateau the gradient is 0.001 and η = 0.1. How long is a plain gradient-descent step?', options: ['0.0001', '0.1', '0.001', '0.01'], answer: 0, why: 'η × |∇L| = 0.1 × 0.001.' },
      { q: 'Why does Adam cross a plateau quickly?', options: ['it divides the gradient by its own recent size, so the step stays near η', 'it uses a larger learning rate automatically', 'it skips the plateau', 'it computes second derivatives'], answer: 0, why: 'A small gradient over a small running size is still about one.' },
      { q: 'A training loss stays flat for many epochs, then suddenly drops. What may have happened?', options: ['the parameters were crossing a plateau', 'the data changed', 'the learning rate rose by itself', 'the model overfit'], answer: 0, why: 'Flat stretches in the loss curve often mean flat ground in the loss surface.' },
      { q: 'Which can cause plateaus in a neural network?', options: ['saturated activations, such as sigmoids far from zero', 'a convex loss', 'a small dataset', 'shuffling the data'], answer: 0, why: 'Where a sigmoid is flat, the gradients through it are nearly zero.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
