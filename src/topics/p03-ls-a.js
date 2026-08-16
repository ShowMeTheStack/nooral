// Part 3, least squares in the plane: points, a first flat guess with its residuals drawn as squares, the normal
// equations with their numbers, and the best line from the engine (ML.LinearRegression, solved by QR). Nudging the
// slope either way makes the squares grow, which is what "least" means.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n;

  S.topic({
    id: 'ls-line', family: 'p03-ls', order: 10, title: 'The least-squares line',
    blurb: 'A line ŷ = w·x + b misses each point by a residual r = y − ŷ. Least squares picks the w and b that make the sum of the squared residuals, drawn here as real squares, as small as it can be. Setting the two derivatives to zero gives the normal equations, two linear equations in w and b; their solution is the line, and it always passes through the mean point (x̄, ȳ).',
    inputs: [
      { id: 'n', kind: 'number', label: 'points', def: 10, min: 2, max: 30 },
      { id: 'slope', kind: 'slider', label: 'true slope', def: 0.6, min: -1.5, max: 1.5, step: 0.1 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.9, min: 0, max: 3, step: 0.1 },
      { id: 'outlier', kind: 'choice', label: 'outlier', def: 'no', options: [{ value: 'no', label: 'none' }, { value: 'yes', label: 'one far below' }] },
      { id: 'seed', kind: 'number', label: 'seed', def: 3, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Ten points near a rising line', values: { n: 10, slope: 0.6, noise: 0.9, outlier: 'no', seed: 3 } },
      { label: 'Very noisy', values: { n: 16, slope: 0.4, noise: 2.4, outlier: 'no', seed: 11 } },
      { label: 'One outlier pulls the line', values: { n: 10, slope: 0.6, noise: 0.5, outlier: 'yes', seed: 3 } },
      { label: 'Two points: an exact fit', values: { n: 2, slope: 0.8, noise: 0.5, outlier: 'no', seed: 5 } },
      { label: 'No trend at all', values: { n: 14, slope: 0, noise: 1.2, outlier: 'no', seed: 8 } },
      { label: 'Falling, 30 points', values: { n: 30, slope: -0.9, noise: 1, outlier: 'no', seed: 21 } },
    ],
    counters: [{ id: 'sse', label: 'sum of squares' }],
    run(inp) {
      const data = H.lineData(inp.n, inp.seed, inp.slope, 2.2 + Math.max(0, -inp.slope) * 9, inp.noise, inp.outlier === 'yes');
      const X = data.X, y = data.y, n = X.length, xs = X.map((r) => r[0]);
      const m = ML().LinearRegression().fit(X, y), tr = m.trace;
      const design = tr.find((s) => s.kind === 'design'), qr = tr.find((s) => s.kind === 'qr'), solve = tr.find((s) => s.kind === 'solve');
      const w = solve.coef[0], b = solve.intercept, xbar = design.means[0], ybar = design.ymean;
      const sst = y.reduce((s, v) => s + (v - ybar) ** 2, 0), sse = solve.sse, r2v = sst > 0 ? 1 - sse / sst : 1;
      const Sx = ML().sum(xs), Sy = ML().sum(y), Sxx = xs.reduce((s, v) => s + v * v, 0), Sxy = xs.reduce((s, v, i) => s + v * y[i], 0);
      const sxx = xs.reduce((s, v) => s + (v - xbar) ** 2, 0), sxy = xs.reduce((s, v, i) => s + (v - xbar) * (y[i] - ybar), 0);
      const F = new S.Frames(['sse']);

      // the plot (each square is drawn square on the screen, so the areas compare fairly whatever the axes' scales)
      const xr = [0, 10], ylo = Math.min(...y, b, w * 10 + b) - 0.4, yhi = Math.max(...y, b, w * 10 + b) + 0.4;
      const yr = H.nice([ylo, yhi], 0), px = 40, py = 30, unit = 38, ph = 430;
      const ax = H.axes({ prefix: 'ls', x: px, y: py, w: unit * 10, h: ph, xr, yr, xticks: 2, yticks: (yr[1] - yr[0]) > 14 ? 4 : (yr[1] - yr[0]) > 7 ? 2 : 1, xname: 'x', yname: 'y', grid: true });
      const right = px + unit * 10 + 40;
      const pts = (cls) => S.Kit.scatter.points(ax, 'ls:p', X.map((r, i) => [r[0], y[i]]), X.map(() => 0), { r: 5, cls });
      const lineItem = (key, ww, bb, cls) => H.seg(key, ax.X(0), ax.Y(bb), ax.X(10), ax.Y(ww * 10 + bb), 'pl-cut ' + (cls || ''));
      const resid = (ww, bb, show) => {
        const items = [];
        let tot = 0;
        X.forEach((r, i) => {
          const yh = ww * r[0] + bb, res = y[i] - yh, x0 = ax.X(r[0]), y0 = ax.Y(yh), y1 = ax.Y(y[i]), s = Math.abs(y1 - y0);
          tot += res * res;
          if (s > 0.6) items.push(H.seg('ls:r' + i, x0, y0, x0, y1, 'pl-seg res ' + (res >= 0 ? 'pos' : 'neg')));
          if (show && s > 0.6) items.push({ key: 'ls:sq' + i, kind: 'path', d: `M${H.r2(x0)},${H.r2(y0)} h${H.r2(s)} V${H.r2(y1)} h${H.r2(-s)} Z`, cls: 'pl-sq', audit: 'ignore', layer: 'back', z: 5 });
        });
        return { items, tot };
      };
      const panel = (lines) => H.eqs('ls:eq', right, 60, lines, 15, 30);
      const head = () => [...ax.items, H.title('ls:t2', right, 24, 'The numbers')];
      const meta = { n, xbar, ybar, w, b, sse, sst, r2: r2v, Sx, Sy, Sxx, Sxy, sxx, sxy, R: Math.abs(qr.R[0][0]) };

      F.add({ items: [...head(), ...pts()], say: n + ' points (x, y). We want the line ŷ = w·x + b that makes the sum of the squared vertical misses as small as possible.', meta: Object.assign({ phase: 'data' }, meta) });
      // a flat first guess
      const g = resid(0, ybar, true);
      F.set('sse', +g.tot.toFixed(3));
      F.add({ items: [...head(), lineItem('ls:line', 0, ybar, 'soft'), ...g.items, ...pts(), ...panel([['flat guess: ŷ = ȳ = ' + n$(ybar), 'muted'], ['Σ (y − ȳ)² = ' + n$(sst), 'hot']])], say: 'A first guess with no slope at all: the flat line at the mean, ŷ = ȳ = ' + n$(ybar) + '. Each residual is drawn as a square; together their area is ' + n$(sst) + ', the total sum of squares.', meta: Object.assign({ phase: 'flat' }, meta) });
      // centring (the engine's first step)
      const cross = [H.seg('ls:mx', ax.X(xbar), ax.Y(yr[0]), ax.X(xbar), ax.Y(yr[1]), 'pl-seg dash', 'ignore'), H.seg('ls:my', ax.X(0), ax.Y(ybar), ax.X(10), ax.Y(ybar), 'pl-seg dash', 'ignore'), { key: 'ls:mean', kind: 'dot', x: ax.X(xbar), y: ax.Y(ybar), r: 7, cls: 'pl-star', audit: 'overlap-ok', layer: 'items', z: 3 }];
      F.add({ items: [...head(), ...cross, ...pts(), ...panel([['x̄ = ' + n$(xbar) + ',  ȳ = ' + n$(ybar), 'good'], ['centre: x − x̄ and y − ȳ', 'muted']])], say: 'The engine first centres the data: x̄ = ' + n$(xbar) + ' and ȳ = ' + n$(ybar) + '. Setting the derivative of the sum of squares with respect to b to zero forces the best line through this mean point (green).', meta: Object.assign({ phase: 'centre' }, meta) });
      // the normal equations
      const M1 = H.matrix('ls:A', right + 8, 150, [[n, Sx], [Sx, Sxx]], { cw: 62, sig: 4 }), M2 = H.matrix('ls:v', right + 8 + M1.w + 10, 150, [['b'], ['w']], { cw: 40 }), M3 = H.matrix('ls:c', right + 8 + M1.w + M2.w + 44, 150, [[Sy], [Sxy]], { cw: 70, sig: 4 });
      const normal = [...M1.items, ...M2.items, H.text('ls:eqs', right + 8 + M1.w + M2.w + 26, 176, '=', 'pl-mname', 18, 'middle'), ...M3.items];
      F.add({ items: [...head(), ...cross, ...pts(), ...panel([['XᵀX · (b, w) = Xᵀy', 'act']]), ...normal, ...H.eqs('ls:cen', right, 250, [['centred: Σ(x − x̄)²·w', 'muted'], ['        = Σ(x − x̄)(y − ȳ)', 'muted'], [n$(sxx, 4) + ' · w = ' + n$(sxy, 4), '']], 15, 28)], say: 'The normal equations: the matrix holds n = ' + n + ', Σx = ' + n$(Sx, 4) + ' and Σx² = ' + n$(Sxx, 4) + '; the right side Σy = ' + n$(Sy, 4) + ' and Σxy = ' + n$(Sxy, 4) + '. After centring only one equation is left: ' + n$(sxx, 4) + ' · w = ' + n$(sxy, 4) + '.', meta: Object.assign({ phase: 'normal' }, meta) });
      // the solution
      const fit = resid(w, b, true);
      F.set('sse', +sse.toFixed(3));
      F.add({ items: [...head(), lineItem('ls:line', w, b, 'act'), ...fit.items, { key: 'ls:mean', kind: 'dot', x: ax.X(xbar), y: ax.Y(ybar), r: 7, cls: 'pl-star', audit: 'overlap-ok', layer: 'items', z: 3 }, ...pts(), ...panel([['ŷ = ' + n$(w) + '·x' + H.term(b, '', false), 'act'], ['Σ r² = ' + n$(sse), 'hot'], ['R² = 1 − ' + n$(sse) + ' / ' + n$(sst) + ' = ' + n$(r2v), 'good']]), ...normal], say: 'The engine factors the centred column by QR (|R| = ‖x − x̄‖ = ' + n$(Math.abs(qr.R[0][0])) + ') and solves: w = ' + n$(w) + ', b = ' + n$(b) + '. The squares shrink from ' + n$(sst) + ' to ' + n$(sse) + ', so R² = ' + n$(r2v) + ': the line explains ' + H.pct(r2v) + ' of the squared spread of y.', meta: Object.assign({ phase: 'fit' }, meta) });
      // nudging the slope about the mean point only makes it worse
      const dw = Math.max(0.15, Math.abs(w) * 0.3), up = resid(w + dw, ybar - (w + dw) * xbar, false).tot, dn = resid(w - dw, ybar - (w - dw) * xbar, false).tot;
      F.add({ items: [...head(), lineItem('ls:up', w + dw, ybar - (w + dw) * xbar, 'soft'), lineItem('ls:dn', w - dw, ybar - (w - dw) * xbar, 'soft'), lineItem('ls:line', w, b, 'act'), ...fit.items, ...pts(), ...panel([['slope ' + n$(w + dw) + ':  Σ r² = ' + n$(up), 'muted'], ['slope ' + n$(w) + ':  Σ r² = ' + n$(sse), 'act'], ['slope ' + n$(w - dw) + ':  Σ r² = ' + n$(dn), 'muted'], ['Σ r = 0 and Σ r·x = 0', 'good']])], say: 'Turn the line about the mean point and the squares grow: slope ' + n$(w + dw) + ' gives ' + n$(up) + ' and slope ' + n$(w - dw) + ' gives ' + n$(dn) + ', both above ' + n$(sse) + '. At the best line the residuals sum to zero and are uncorrelated with x.', meta: Object.assign({ phase: 'nudge', up, dn, wUp: w + dw, wDn: w - dw }, meta) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Least squares', text: ['minimise Σ (yᵢ − w·xᵢ − b)²', 'normal equations: XᵀX θ = Xᵀy', 'w = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)²', 'b = ȳ − w·x̄', 'R² = 1 − Σ r² / Σ (y − ȳ)²'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'normal', pick: 'option', ask: 'Will the best line rise or fall from left to right?', options: ['rise', 'fall'], answer: (fr) => (fr[0].meta.w >= 0 ? 'rise' : 'fall'), explain: (fr) => 'w = ' + n$(fr[0].meta.w) + ': ' + (fr[0].meta.w >= 0 ? 'it rises.' : 'it falls.') },
      { at: (f) => f.meta.phase === 'flat', pick: 'option', ask: 'The best line must pass through one point for sure. Which?', options: ['the mean point (x̄, ȳ)', 'the origin', 'the leftmost point', 'the point with the largest y'], answer: 'the mean point (x̄, ȳ)', explain: 'The derivative with respect to b is −2 Σ r, so at the minimum Σ r = 0: ȳ = w·x̄ + b.' },
    ],
    quiz: [
      { q: 'What does least squares minimise?', options: ['the sum of squared vertical distances to the line', 'the sum of perpendicular distances', 'the largest distance', 'the number of points off the line'], answer: 0, why: 'Residuals are measured along y, the quantity being predicted, and squared.' },
      { q: 'For three points (0, 1), (1, 3), (2, 5), what is the least-squares slope?', options: ['2', '1', '3', '1.5'], answer: 0, why: 'They lie exactly on y = 2x + 1, so every residual is zero.' },
      { q: 'Why does one far outlier move the line so much?', options: ['its residual is squared, so it weighs far more than the others', 'outliers are removed first', 'least squares ignores it', 'it changes x̄ only'], answer: 0, why: 'A residual of 6 costs 36, as much as nine residuals of 2.' },
      { q: 'At the least-squares solution, the residuals…', options: ['sum to zero and are uncorrelated with x', 'are all equal', 'are all positive', 'grow with x'], answer: 0, why: 'Those are exactly the two normal equations.' },
      { q: 'R² = 0.9 means…', options: ['the line removes 90% of the squared spread around ȳ', 'the slope is 0.9', '90% of points are on the line', 'the error is 10 units'], answer: 0, why: 'R² = 1 − SSE / SST.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
