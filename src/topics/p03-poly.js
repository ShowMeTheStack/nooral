// Part 3, polynomial regression and overfitting. Powers of x become extra features, and the same least-squares
// solver fits a curve. Degree by degree the curve bends more: the training error can only fall, but the error on
// fresh points from the same curve falls, bottoms out and then climbs as the curve starts chasing the noise.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML, M = () => S.Metrics;
  const n$ = H.n, C = () => S.Kit.curve;
  const SHAPES = [{ value: 'sine', label: 'a sine wave' }, { value: 'poly', label: 'a cubic' }, { value: 'abs', label: 'a V shape' }, { value: 'step', label: 'steps' }, { value: 'linear', label: 'a straight line' }];

  /** fit one degree: features u, u², …, u^d of u = x / 3 (so the powers stay between −1 and 1) */
  function fitDegree(X, y, d) {
    const pf = ML().PolynomialFeatures({ degree: d, includeBias: false }), U = X.map((r) => [r[0] / 3]);
    const m = ML().LinearRegression().fit(pf.fitTransform(U), y);
    const predict = (xs) => m.predict(pf.transform(xs.map((v) => [v / 3])));
    return { m, pf, predict, sse: m.trace.find((s) => s.kind === 'solve').sse };
  }
  H.polyFit = fitDegree;

  S.topic({
    id: 'poly-overfit', family: 'p03-poly', order: 10, title: 'Polynomial regression and overfitting',
    blurb: 'Give a straight-line method the features x, x², x³, … and it fits curves: the model is still linear in its weights, so least squares solves it exactly. Each extra degree lets the curve bend once more. The error on the training points can only go down, because a higher degree can always copy a lower one. The error on fresh points tells the truth: it falls while the curve learns the shape, then rises once it starts threading the noise.',
    inputs: [
      { id: 'shape', kind: 'choice', label: 'true curve', def: 'sine', options: SHAPES },
      { id: 'n', kind: 'number', label: 'training points', def: 12, min: 4, max: 40 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.25, min: 0, max: 0.8, step: 0.05 },
      { id: 'maxDeg', kind: 'number', label: 'highest degree', def: 11, min: 1, max: 14 },
      { id: 'seed', kind: 'number', label: 'seed', def: 2, min: 1, max: 9999 },
    ],
    check: (inp) => (inp.maxDeg > inp.n - 1 ? 'highest degree: at most ' + (inp.n - 1) + ' with ' + inp.n + ' points (a degree-d curve has d + 1 numbers to fit)' : null),
    presets: [
      { label: 'Twelve points on a sine wave', values: { shape: 'sine', n: 12, noise: 0.25, maxDeg: 11, seed: 2 } },
      { label: 'More data tames high degrees', values: { shape: 'sine', n: 40, noise: 0.25, maxDeg: 14, seed: 2 } },
      { label: 'No noise', values: { shape: 'sine', n: 12, noise: 0, maxDeg: 11, seed: 2 } },
      { label: 'A cubic: degree 3 is the truth', values: { shape: 'poly', n: 15, noise: 0.3, maxDeg: 12, seed: 6 } },
      { label: 'Steps: no polynomial is right', values: { shape: 'step', n: 20, noise: 0.15, maxDeg: 14, seed: 3 } },
      { label: 'A straight line, heavy noise', values: { shape: 'linear', n: 10, noise: 0.6, maxDeg: 9, seed: 9 } },
    ],
    counters: [{ id: 'deg', label: 'degree' }],
    run(inp) {
      const tr = S.Data.curve({ n: inp.n, shape: inp.shape, noise: inp.noise, seed: inp.seed });
      const te = S.Data.curve({ n: 60, shape: inp.shape, noise: inp.noise, seed: inp.seed + 7919 });
      const D = inp.maxDeg, rows = [];
      for (let d = 1; d <= D; d++) {
        const f = fitDegree(tr.X, tr.y, d);
        const trMse = M().mse(tr.y, f.predict(tr.X.map((r) => r[0]))).value, teMse = M().mse(te.y, f.predict(te.X.map((r) => r[0]))).value;
        rows.push({ d, f, trMse, teMse, maxCoef: Math.max(...f.m.coef.map(Math.abs)) });
      }
      const best = rows.reduce((a, b) => (b.teMse < a.teMse ? b : a));
      const F = new S.Frames(['deg']);
      const yAll = tr.y.concat(te.y), yr = H.nice(yAll, 0.25);
      const ax = H.axes({ prefix: 'pf', x: 40, y: 40, w: 400, h: 330, xr: [-3, 3], yr, xticks: 1, yticks: yr[1] - yr[0] > 5 ? 1 : 0.5, xname: 'x', yname: 'y' });
      const lg = (v) => Math.log10(Math.max(v, 1e-6));
      const eAll = rows.flatMap((r) => [r.trMse, r.teMse]), ey = [Math.floor(lg(Math.min(...eAll))), Math.ceil(lg(Math.max(...eAll)))];
      if (ey[1] - ey[0] < 1) ey[1] = ey[0] + 1;
      const eax = H.axes({ prefix: 'pe', x: 530, y: 40, w: 220, h: 200, xr: [0.5, D + 0.5], yr: ey, xticks: C().ticks(1, D, Math.min(D, 6)).filter((v) => Number.isInteger(v)), yticks: ey[1] - ey[0] > 5 ? 2 : 1, fmtY: (v) => '10' + (v < 0 ? '⁻' : '') + H.sup(Math.abs(v)), xname: 'degree', yname: 'mean squared error' });
      const truth = tr.truth, grid = [];
      for (let k = 0; k <= 160; k++) grid.push(-3 + (6 * k) / 160);
      const truthLine = C().line(ax, 'pf:truth', grid.map((x) => [x, truth(x)]), 'faint');
      const pts = () => [...S.Kit.scatter.points(ax, 'pf:te', te.X.map((r, i) => [r[0], te.y[i]]), te.y.map(() => 0), { r: 3.2, cls: () => 'hollow' }), ...S.Kit.scatter.points(ax, 'pf:tr', tr.X.map((r, i) => [r[0], tr.y[i]]), tr.y.map(() => 1), { r: 5 })];
      const legend = C().legend('pf:lg', 40, 432, [{ text: 'training', cls: 'c1', dot: true }, { text: 'fresh test points', cls: 'c0', dot: true }, { text: 'true curve', cls: 'faint' }, { text: 'fit', cls: 'act' }]);
      const errLegend = C().legend('pe:lg', 530, 300, [{ text: 'training error', cls: 'c1' }, { text: 'test error', cls: 'c0' }], { dir: 'col' });
      const head = () => [...ax.items, H.title('pf:t', 40, 14, 'The fit'), truthLine, ...eax.items, H.title('pe:t', 500, 14, 'Error against degree'), ...legend, ...errLegend];
      const meta0 = { n: inp.n, nTest: 60, bestD: best.d, bestTe: best.teMse, D };
      F.add({ items: [...head(), ...pts()], say: inp.n + ' training points (orange) lie on ' + SHAPES.find((s) => s.value === inp.shape).label + ' with noise; 60 fresh points from the same curve (hollow) are kept aside for testing. We fit polynomials of degree 1 to ' + D + ' by least squares on x, x², and so on.', meta: Object.assign({ phase: 'data' }, meta0) });
      rows.forEach((r, k) => {
        const fitPts = r.f.predict(grid).map((v, i) => [grid[i], v >= yr[0] && v <= yr[1] ? v : NaN]);
        const items = [...head(), C().line(ax, 'pf:fit', fitPts, 'act'), ...pts()];
        const upto = rows.slice(0, k + 1);
        items.push(C().line(eax, 'pe:tr', upto.map((q) => [q.d, lg(q.trMse)]), 'c1'), C().line(eax, 'pe:te', upto.map((q) => [q.d, lg(q.teMse)]), 'c0'));
        items.push(Object.assign(C().dot(eax, 'pe:dtr', r.d, lg(r.trMse), 'c1'), { point: false, audit: 'overlap-ok' }), Object.assign(C().dot(eax, 'pe:dte', r.d, lg(r.teMse), 'c0'), { point: false, audit: 'overlap-ok' }));
        items.push(...H.eqs('pf:eq', 530, 390, [['degree ' + r.d + ': ' + (r.d + 1) + ' weights', 'act'], ['largest |weight| ' + n$(r.maxCoef), 'muted']]));
        F.set('deg', r.d);
        const prev = rows[k - 1];
        let say = 'Degree ' + r.d + ': training error ' + n$(r.trMse) + ', test error ' + n$(r.teMse) + '.';
        if (r.d === 1) say = 'Degree 1 is a straight line: training error ' + n$(r.trMse) + ', test error ' + n$(r.teMse) + '. It cannot follow the curve at all: it underfits.';
        else if (prev && r.teMse > prev.teMse * 1.15 && r.d > best.d) say += ' The test error climbs while the training error keeps falling: the curve is bending to pass through the noise. The largest weight is now ' + n$(r.maxCoef) + '.';
        else if (prev && r.teMse < prev.teMse) say += ' Both errors fall: the extra bend helps.';
        if (r.trMse < 1e-6) say += ' The curve passes exactly through every training point.';
        F.add({ items, say, meta: Object.assign({ phase: 'deg', d: r.d, tr: r.trMse, te: r.teMse, maxCoef: r.maxCoef, prevTe: prev ? prev.teMse : null }, meta0) });
      });
      const bf = best.f.predict(grid).map((v, i) => [grid[i], v >= yr[0] && v <= yr[1] ? v : NaN]), lastR = rows[rows.length - 1];
      const fin = [...head(), C().line(ax, 'pf:fit', bf, 'act'), ...pts(), C().line(eax, 'pe:tr', rows.map((q) => [q.d, lg(q.trMse)]), 'c1'), C().line(eax, 'pe:te', rows.map((q) => [q.d, lg(q.teMse)]), 'c0'), C().dot(eax, 'pe:dte', best.d, lg(best.teMse), 'c0'), C().vline(eax, 'pe:best', best.d, 'act'), ...H.eqs('pf:eq', 530, 390, [['best on fresh points: degree ' + best.d, 'good'], ['test error ' + n$(best.teMse), 'good']])];
      F.add({ items: fin, say: 'The test error is lowest at degree ' + best.d + ' (' + n$(best.teMse) + '), drawn again here. Degree ' + D + ' has the smallest training error, ' + n$(lastR.trMse) + ', but a test error of ' + n$(lastR.teMse) + '. Choose the degree by the error on data the fit did not see.', meta: Object.assign({ phase: 'best', lastTr: lastR.trMse, lastTe: lastR.teMse }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Polynomial regression', text: ['ŷ = w₀ + w₁x + w₂x² + … + w_d x^d', 'features (x, x², …, x^d): still linear in w', 'training error never rises with d', 'test error: falls, then rises (overfitting)', 'd + 1 weights need more than d + 1 points'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'Which degree will do best on the fresh test points?', options: (fr) => { const D = fr[0].meta.D, b = fr[0].meta.bestD; return [...new Set([b, 1, D, Math.max(1, Math.min(D, b + 3)), Math.max(1, b - 2)])].sort((a, c) => a - c).map(String); }, answer: (fr) => String(fr[0].meta.bestD), explain: (fr) => 'Degree ' + fr[0].meta.bestD + ': test error ' + n$(fr[0].meta.bestTe) + '.' },
      { at: (f, i, fr) => f.meta.phase === 'deg' && fr[i + 1] && fr[i + 1].meta.phase === 'deg' && f.meta.d === Math.max(2, Math.floor(f.meta.D / 2)), pick: 'option', ask: 'Adding one more degree, will the training error go up or down?', options: ['down (or stay)', 'up'], answer: 'down (or stay)', explain: 'The higher degree can always copy the lower one by setting its new weight to 0, so its best fit is at least as good on the training points.' },
    ],
    quiz: [
      { q: 'Why is polynomial regression still "linear regression"?', options: ['the prediction is linear in the weights, with powers of x as features', 'the curve is a straight line', 'it uses one feature', 'it is not'], answer: 0, why: 'Least squares only needs linearity in the unknowns.' },
      { q: 'As the degree grows, the training error…', options: ['never goes up', 'always goes up', 'falls then rises', 'stays constant'], answer: 0, why: 'Each model contains the one before.' },
      { q: 'Overfitting shows as…', options: ['a low training error and a much higher test error', 'both errors high', 'both errors low', 'a high training error and a low test error'], answer: 0, why: 'The model has learned the noise of its training points.' },
      { q: 'Which of these reduces overfitting of a degree-12 fit?', options: ['more training data', 'a higher degree', 'less data', 'removing the test set'], answer: 0, why: 'More points pin the curve down between them; a penalty on the weights (ridge) also helps.' },
      { q: 'With 8 points, a polynomial of degree 7…', options: ['can pass exactly through all of them', 'cannot be fitted', 'is always the best model', 'is a straight line'], answer: 0, why: '8 weights, 8 equations: an exact fit (when the x values differ).' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
