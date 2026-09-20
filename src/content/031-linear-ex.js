// Part 3, linear models: the exercises. Every number in a text, an answer or a hint is worked out here by the engines
// (S.ML, S.Data) when the file loads, on small fixed data, so the text and the checked answer cannot drift apart.
(function (S) {
  'use strict';
  const C = S.Content, ML = S.ML, WF = S.WF;
  if (!C || !ML || !WF || C.exercises.has('li-x-ls-normal')) return;
  const r3 = (v) => Math.round(v * 1000) / 1000, r2 = (v) => Math.round(v * 100) / 100, r4 = (v) => Math.round(v * 10000) / 10000;
  const f = (v) => String(r3(v)).replace(/^-/, '−');
  const pipe = (data, model, mp, train, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };
  const sol = (cells) => Object.fromEntries(Object.entries(cells).map(([k, v]) => [k, String(v)]));
  const mse = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length;

  // =========================================================================================================
  // Chapter 1: least squares and gradient descent
  // =========================================================================================================
  const NX = [1, 2, 4, 5], NY = [2, 3, 7, 8];
  const nfit = ML.LinearRegression().fit(NX.map((v) => [v]), NY), nw = r3(nfit.coef[0]), nb = r3(nfit.intercept);
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const NS = { n: NX.length, sx: sum(NX), sxx: sum(NX.map((v) => v * v)), sy: sum(NY), sxy: sum(NX.map((v, i) => v * NY[i])) };
  const ncells = { 'Σx|value': NS.sx, 'Σx²|value': NS.sxx, 'Σy|value': NS.sy, 'Σxy|value': NS.sxy, 'slope w|value': nw, 'intercept b|value': nb };
  const pts = NX.map((v, i) => '(' + v + ', ' + NY[i] + ')').join(', ');
  C.exercise({ id: 'li-x-ls-normal', chapter: 'li-ls', title: 'The normal equations of four points', level: 2, kind: 'table',
    text: ['Four points: ' + pts + '. Fill in the sums of the normal equations, then solve them for the least-squares line ŷ = w·x + b.', { math: 'n·b + Σx·w = Σy\nΣx·b + Σx²·w = Σxy' }],
    rows: ['Σx', 'Σx²', 'Σy', 'Σxy', 'slope w', 'intercept b'], cols: ['value'], cells: ncells, tolerance: 0.005, corner: 'quantity',
    hints: ['n = 4. Add up the x values, their squares, the y values and the products x·y.', 'The equations are ' + NS.n + 'b + ' + NS.sx + 'w = ' + NS.sy + ' and ' + NS.sx + 'b + ' + NS.sxx + 'w = ' + NS.sxy + '. Eliminate b: multiply the first by ' + NS.sx / NS.n + ' and subtract.', 'Σx = ' + NS.sx + ', Σx² = ' + NS.sxx + ', Σy = ' + NS.sy + ', Σxy = ' + NS.sxy + '. Then (' + NS.sxx + ' − ' + NS.sx * NS.sx / NS.n + ')·w = ' + NS.sxy + ' − ' + NS.sx * NS.sy / NS.n + ', so w = ' + nw + ', and b = (' + NS.sy + ' − ' + NS.sx + '·' + nw + ') / ' + NS.n + ' = ' + nb + '.'],
    solution: sol(ncells), wrong: [sol(Object.assign({}, ncells, { 'slope w|value': r3(NS.sxy / NS.sxx), 'intercept b|value': 0 })), sol(Object.assign({}, ncells, { 'Σx²|value': NS.sx * NS.sx }))] });

  C.exercise({ id: 'li-x-ls-residuals', chapter: 'li-ls', title: 'What the best line guarantees', level: 1, kind: 'choice',
    text: ['A least-squares line with an intercept has been fitted to twenty points. Which statement is always true of its residuals yᵢ − ŷᵢ?'],
    options: ['They add up to zero', 'Exactly half of them are positive', 'The largest one is as small as any line could make it', 'They all have the same size'], answer: 0,
    whyNot: { 1: 'The signs can be split unevenly: a few large positive residuals can balance many small negative ones.', 2: 'That is a different criterion (the smallest worst miss). Least squares minimises the sum of squares, and may leave one large miss.', 3: 'Only if every point is equally far from the line, which noise rarely allows.' },
    hints: ['Think of the derivative of Σ (yᵢ − w·xᵢ − b)² with respect to b.', 'Setting it to zero gives −2 Σ (yᵢ − w·xᵢ − b) = 0.', 'The first normal equation says Σ residuals = 0: they always add up to zero when the line has an intercept.'],
    solution: 0, wrong: [1, 2] });

  const npred = nfit.predict(NX.map((v) => [v])), ym = sum(NY) / NY.length;
  const sse = sum(NY.map((v, i) => (v - npred[i]) ** 2)), sst = sum(NY.map((v) => (v - ym) ** 2)), R2 = 1 - sse / sst;
  C.exercise({ id: 'li-x-ls-r2', chapter: 'li-ls', title: 'R² of the line', level: 1, kind: 'number',
    text: ['The least-squares line through ' + pts + ' is ŷ = ' + nw + '·x + ' + nb + '. What is its R²? Give three decimals.'],
    answer: r3(R2), tolerance: 0.0015, whyNot: { [String(r3(sse / sst))]: 'That is the share left unexplained, SSE / SST; R² is one minus it.', [String(r2(sse))]: 'That is the sum of squared residuals itself.' },
    hints: ['R² = 1 − SSE / SST, with SST = Σ (yᵢ − ȳ)².', 'The predictions are ' + npred.map(r3).join(', ') + ' and ȳ = ' + ym + '.', 'SSE = ' + r3(sse) + ' and SST = ' + r3(sst) + ', so R² = 1 − ' + r3(sse) + ' / ' + r3(sst) + ' = ' + r3(R2) + '.'],
    solution: String(r3(R2)), wrong: [String(r3(sse / sst)), String(r3(sse))] });

  const PX = [0, 1, 2], PY = [2, 1, 4], pf = ML.LinearRegression().fit(PX.map((v) => [v]), PY), pp = pf.predict(PX.map((v) => [v]));
  const rr = sum(PY.map((v, i) => (v - pp[i]) ** 2)), pym = sum(PY) / 3, flat = sum(PY.map((v) => (v - pym) ** 2));
  C.exercise({ id: 'li-x-ls-project', chapter: 'li-ls', title: 'The length of the residual', level: 2, kind: 'number',
    text: ['Three observations y = (' + PY.join(', ') + ') at x = (' + PX.join(', ') + '). Project y onto the plane spanned by 1 = (1, 1, 1) and x. What is ‖r‖², the squared length of the residual? Give three decimals.', { topic: 'ls-projection', input: { x: PX.map(String), y: PY.map(String) }, frame: 2, caption: 'The plane of all predictions b·1 + w·x, and the arrow y.' }],
    answer: r3(rr), tolerance: 0.002, whyNot: { [String(r3(flat))]: 'That is the squared distance to the flat guess ȳ·1, not to the nearest point of the plane.', [String(r3(Math.sqrt(rr)))]: 'That is the length ‖r‖ itself; square it.' },
    hints: ['The projection is the least-squares fit: find b and w first.', 'x̄ = 1 and ȳ = ' + r3(pym) + '; w = Σ (x − x̄)(y − ȳ) / Σ (x − x̄)² = ' + r3(pf.coef[0]) + ' and b = ' + r3(pf.intercept) + '.', 'ŷ = (' + pp.map(r3).join(', ') + '), r = (' + PY.map((v, i) => r3(v - pp[i])).join(', ') + '), and ‖r‖² = ' + r3(rr) + '.'],
    solution: String(r3(rr)), wrong: [String(r3(flat)), String(r3(Math.sqrt(rr)))] });

  const GX = [1, 2, 3], GY = [2, 3, 5], G = ML.LinearRegression({ solver: 'gd', lr: 0.1, epochs: 1 }).fit(GX.map((v) => [v]), GY);
  const g0 = G.trace.find((s) => s.kind === 'gd' && s.epoch === 0), gw = g0.grad[0], gb = -sum(GY) / GY.length;
  const gcells = { 'loss L|value': r3(g0.loss), '∂L/∂w|value': r3(gw), '∂L/∂b|value': r3(gb), 'new w|value': r3(G.coef[0]), 'new b|value': r3(G.intercept) };
  C.exercise({ id: 'li-x-gd-step', chapter: 'li-ls', title: 'One step of gradient descent', level: 2, kind: 'table',
    text: ['Three points (1, 2), (2, 3), (3, 5). The loss is L = (1/2n) Σ (yᵢ − w·xᵢ − b)² with n = 3. Start at w = 0, b = 0 and take one step of gradient descent with η = 0.1. Fill in the table to three decimals.', { math: '∂L/∂w = (1/n) Σ (ŷᵢ − yᵢ) xᵢ\n∂L/∂b = (1/n) Σ (ŷᵢ − yᵢ)' }],
    rows: ['loss L', '∂L/∂w', '∂L/∂b', 'new w', 'new b'], cols: ['value'], cells: gcells, tolerance: 0.002, corner: 'at w = 0, b = 0',
    hints: ['At w = 0, b = 0 every prediction is 0, so each ŷᵢ − yᵢ is just −yᵢ.', 'L = (4 + 9 + 25) / 6. ∂L/∂w = (1/3)(−2·1 − 3·2 − 5·3).', 'L = ' + r3(g0.loss) + ', ∂L/∂w = ' + r3(gw) + ', ∂L/∂b = ' + r3(gb) + '; the step gives w = 0 − 0.1·(' + r3(gw) + ') = ' + r3(G.coef[0]) + ' and b = ' + r3(G.intercept) + '.'],
    solution: sol(gcells), wrong: [sol(Object.assign({}, gcells, { 'new w|value': r3(-G.coef[0]), 'new b|value': r3(-G.intercept) })), sol(Object.assign({}, gcells, { '∂L/∂w|value': r3(gw * 3), 'new w|value': r3(-0.1 * gw * 3) }))] });

  const gdRef = (X, y, w, b, lr) => {
    const n = X.length, d = w.length, g = new Array(d).fill(0); let gbb = 0;
    for (let i = 0; i < n; i++) { const r = X[i].reduce((s, v, j) => s + v * w[j], b) - y[i]; gbb += r; for (let j = 0; j < d; j++) g[j] += r * X[i][j]; }
    return w.map((v, j) => v - (lr * g[j]) / n).concat([b - (lr * gbb) / n]);
  };
  C.exercise({ id: 'li-x-gd-code', chapter: 'li-ls', title: 'Write a gradient-descent step', level: 2, kind: 'code',
    text: ['Write `gdStep(X, y, w, b, lr)` for linear regression with the loss L = (1/2n) Σ (yᵢ − w·xᵢ − b)². X is an array of rows, w an array of weights (one per column), b a number. Take one step and return an array: the new weights followed by the new intercept, like `[w0, w1, b]`.'],
    fn: 'gdStep', params: ['X', 'y', 'w', 'b', 'lr'], tolerance: 1e-9,
    start: 'function gdStep(X, y, w, b, lr) {\n  // the residuals, the gradient, then the step\n  \n}',
    ref: gdRef, cases: [[[[1], [2], [3]], [2, 3, 5], [0], 0, 0.1], [[[1, 0], [0, 1], [1, 1]], [1, 2, 2], [0.5, 0.5], 0, 0.5], [[[2, 1], [1, -1], [0, 3], [1, 1]], [3, 0, 4, 2], [1, -1], 0.5, 0.05], [[[4, -2, 1]], [7], [0.1, 0.2, 0.3], -1, 0.02]],
    hints: ['For each row, the residual is r = w·x + b − y; the gradient for wⱼ is the mean of r·xⱼ, and for b the mean of r.', 'Accumulate the sums over the rows, divide by n, and subtract lr times each from its parameter.', 'function gdStep(X, y, w, b, lr) {\n  const n = X.length, g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < n; i++) {\n    const r = X[i].reduce((s, v, j) => s + v * w[j], b) - y[i];\n    gb += r;\n    for (let j = 0; j < w.length; j++) g[j] += r * X[i][j];\n  }\n  return w.map((v, j) => v - lr * g[j] / n).concat([b - lr * gb / n]);\n}'],
    solution: 'function gdStep(X, y, w, b, lr) {\n  const n = X.length, g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < n; i++) {\n    const r = X[i].reduce((s, v, j) => s + v * w[j], b) - y[i];\n    gb += r;\n    for (let j = 0; j < w.length; j++) g[j] += r * X[i][j];\n  }\n  return w.map((v, j) => v - lr * g[j] / n).concat([b - lr * gb / n]);\n}',
    wrong: ['function gdStep(X, y, w, b, lr) {\n  const g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < X.length; i++) { const r = X[i].reduce((s, v, j) => s + v * w[j], b) - y[i]; gb += r; for (let j = 0; j < w.length; j++) g[j] += r * X[i][j]; }\n  return w.map((v, j) => v - lr * g[j]).concat([b - lr * gb]);\n}',
      'function gdStep(X, y, w, b, lr) {\n  const n = X.length, g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < n; i++) { const r = X[i].reduce((s, v, j) => s + v * w[j], b) - y[i]; gb += r; for (let j = 0; j < w.length; j++) g[j] += r * X[i][j]; }\n  return w.map((v, j) => v + lr * g[j] / n).concat([b + lr * gb / n]);\n}'] });

  // a line through ten noisy points on standardised x: the loss epoch by epoch at η = 0.1
  const cr = ML.rng(3), cx = ML.range(10).map((i) => i + 1), cy = cx.map((v) => 2 * v + 1 + 0.8 * cr.normal());
  const cm = sum(cx) / 10, cs = Math.sqrt(sum(cx.map((v) => (v - cm) ** 2)) / 10), cxs = cx.map((v) => [(v - cm) / cs]);
  const ch = ML.LinearRegression({ solver: 'gd', lr: 0.1, epochs: 30 }).fit(cxs, cy).history, gdCurve = ch.map((v, e) => [e, r3(v)]);
  C.exercise({ id: 'li-x-gd-curve', chapter: 'li-ls', title: 'The loss, epoch by epoch', level: 2, kind: 'predict-curve',
    text: ['Gradient descent fits a line to ten points with a standardised input, from w = 0, b = 0, with η = 0.1. Its first three losses are drawn. Sketch the loss for all 30 epochs.'],
    plot: { xr: [0, 30], yr: [0, Math.ceil(gdCurve[0][1] + 5)], xname: 'epoch', yname: 'loss', given: gdCurve.slice(0, 3), note: 'Drag from epoch 0 to epoch 30.' },
    answer: gdCurve, tolerance: Math.max(2, r2(gdCurve[0][1] / 15)), checks: ['falling'],
    hints: ['With a standardised input the bowl is round: each step removes the same share of what is left.', 'Here that share is η = 0.1 of the distance per step, so the excess loss shrinks by (1 − 0.1)² ≈ 0.81 each epoch.', 'It falls quickly at first and flattens: ' + gdCurve.slice(0, 6).map((p) => p[1]).join(', ') + ', … and ' + gdCurve[30][1] + ' at epoch 30.'],
    solution: gdCurve, wrong: [gdCurve.map(([e]) => [e, r3(gdCurve[0][1] * (1 - e / 30))]), gdCurve.map(([e]) => [e, gdCurve[0][1]])] });

  const LX = [1, 2, 3, 4], LY = [2, 3, 5, 6], mxx = sum(LX.map((v) => v * v)) / 4, mx = sum(LX) / 4;
  const ltr = mxx + 1, ldet = mxx - mx * mx, lmax = ltr / 2 + Math.sqrt(ltr * ltr / 4 - ldet), etaMax = 2 / lmax;
  const lossAfter = (lr) => ML.LinearRegression({ solver: 'gd', lr, epochs: 100, traceEvery: 1000 }).fit(LX.map((v) => [v]), LY).history[100];
  const below = r3(etaMax - 0.01), above = r3(etaMax + 0.01), lb = lossAfter(below), la = lossAfter(above);
  C.exercise({ id: 'li-x-gd-lrmax', chapter: 'li-ls', title: 'The largest safe step', level: 3, kind: 'number',
    text: ['Gradient descent on L(w, b) = (1/2n) Σ (yᵢ − w·xᵢ − b)² with raw inputs x = (' + LX.join(', ') + '). Its curvature is the matrix H = [[mean x², mean x], [mean x, 1]], whatever the y values. Above what learning rate η does gradient descent diverge? Give three decimals.'],
    answer: r3(etaMax), tolerance: 0.002, whyNot: { [String(r3(2 / mxx))]: 'That uses only the w direction; the intercept couples to it and makes the largest curvature bigger.', [String(r3(1 / lmax))]: 'The limit is 2 divided by the largest curvature, not 1 divided by it.' },
    hints: ['A step is stable in every direction only while η < 2 / λmax, where λmax is the largest eigenvalue of H.', 'mean x² = ' + mxx + ' and mean x = ' + mx + '. For a 2 × 2 matrix, λmax = tr/2 + √(tr²/4 − det).', 'tr = ' + ltr + ', det = ' + ldet + ', λmax = ' + r3(lmax) + ', so η < 2 / ' + r3(lmax) + ' = ' + r3(etaMax) + '. The engine agrees: after 100 epochs η = ' + below + ' reaches loss ' + r3(lb) + ', while η = ' + above + ' has blown up to ' + (la > 1e6 ? 'more than a million' : r3(la)) + '.'],
    solution: String(r3(etaMax)), wrong: [String(r3(2 / mxx)), String(r3(1 / lmax))] });

  // polynomials of degree 1 to 9 on twelve noisy points of a sine wave, scored on 200 fresh points
  const PT = S.Data.load('curve', { n: 12, seed: 2, shape: 'sine', noise: 0.25 }), PE = S.Data.load('curve', { n: 200, seed: 9, shape: 'sine', noise: 0.25 });
  const polyErr = ML.range(9).map((k) => { const d = k + 1, P = ML.PolynomialFeatures({ degree: d, includeBias: false }).fit(PT.X), m = ML.LinearRegression().fit(P.transform(PT.X), PT.y); return [d, mse(PT.y, m.predict(P.transform(PT.X))), mse(PE.y, m.predict(P.transform(PE.X)))]; });
  const bestDeg = polyErr.reduce((a, b) => (b[2] < a[2] ? b : a))[0];
  C.exercise({ id: 'li-x-poly-pick', chapter: 'li-ls', title: 'Choose the degree', level: 1, kind: 'choice',
    text: ['Polynomials of degree 1 to 9 were fitted by least squares to twelve noisy points on a wave, and scored on the training points and on 200 fresh points from the same wave.', { table: [['degree', 'training error', 'test error']].concat(polyErr.map(([d, a, b]) => [d, r4(a), r3(b)])), caption: 'Mean squared errors.' }, 'Which degree would you use?'],
    options: ['1', String(bestDeg), '8', '9'], answer: 1,
    whyNot: { 0: 'A straight line misses the wave: both of its errors are high.', 2: 'Its training error is tiny, but its test error is enormous: it has learned the noise.', 3: 'The lowest training error, and the worst test error. The training error always prefers the most flexible model.' },
    hints: ['Which column tells you how the model does on points it has not seen?', 'Look for the smallest test error.', 'Degree ' + bestDeg + ' has the smallest test error, ' + r3(polyErr[bestDeg - 1][2]) + '.'],
    solution: 1, wrong: [3, 0] });
  const polyCurve = polyErr.map(([d, a]) => [d, r4(a)]);
  C.exercise({ id: 'li-x-poly-curve', chapter: 'li-ls', title: 'The training error as the degree grows', level: 2, kind: 'predict-curve',
    text: ['The same twelve points. Sketch the training error (mean squared error on the twelve points themselves) for polynomial degrees 1 to 9. Degrees 1 and 2 are drawn.'],
    plot: { xr: [1, 9], yr: [0, 0.1], xname: 'degree', yname: 'training error', given: polyCurve.slice(0, 2) },
    answer: polyCurve, tolerance: 0.012, checks: ['falling'],
    hints: ['Adding a column can never make the least-squares fit worse on its own points: the old fit is still available.', 'The big drop comes when the curve first bends twice, at degree 3.', 'It falls from ' + polyCurve[0][1] + ' to ' + polyCurve[2][1] + ' at degree 3 and on to ' + polyCurve[8][1] + ' at degree 9.'],
    solution: polyCurve, wrong: [polyErr.map(([d, , b]) => [d, Math.min(0.1, r4(b))]), polyCurve.map(([d]) => [d, polyCurve[0][1]])] });

  // =========================================================================================================
  // Chapter 2: ridge, lasso and elastic net
  // =========================================================================================================
  C.exercise({ id: 'li-x-reg-which', chapter: 'li-reg', title: 'Which penalty gives zeros', level: 1, kind: 'choice',
    text: ['A model has 40 features and you suspect most of them are useless. You want a fitted model whose useless weights are exactly 0, so you can drop those features. Which penalty do you add to the squared error?'],
    options: ['λ Σ wⱼ² (ridge)', 'λ Σ |wⱼ| (lasso)', 'Neither: penalties only shrink weights, never to exactly 0', 'A larger learning rate'], answer: 1,
    whyNot: { 0: 'Ridge shrinks every weight smoothly toward 0 but never onto it, at any finite λ.', 2: 'The lasso’s soft threshold has a flat stretch at exactly 0.', 3: 'The learning rate changes how training walks, not where the minimum is.' },
    hints: ['Picture the budget each penalty allows: a disc or a diamond.', 'The error rings often first touch a diamond at a corner, and corners lie on the axes.', 'The lasso, λ Σ |wⱼ|: its diamond’s corners set weights exactly to 0.'],
    solution: 1, wrong: [0, 2] });

  const RX = [[1], [2], [3]], RY = [2, 3, 7], ra = 4, rw = ML.Ridge({ alpha: ra, fitIntercept: false }).fit(RX, RY).coef[0], ow = ML.LinearRegression({ fitIntercept: false }).fit(RX, RY).coef[0];
  C.exercise({ id: 'li-x-reg-ridge1d', chapter: 'li-reg', title: 'Ridge with one weight', level: 2, kind: 'number',
    text: ['Points (1, 2), (2, 3), (3, 7) and a line through the origin, ŷ = w·x. Ridge minimises Σ (yᵢ − w·xᵢ)² + α w² with α = ' + ra + '. What is w? Give three decimals.'],
    answer: r3(rw), tolerance: 0.002, whyNot: { [String(r3(ow))]: 'That is plain least squares, α = 0. The penalty adds α to the denominator.', [String(r3(29 / (14 - ra)))]: 'The penalty makes the denominator larger, Σ x² + α, not smaller.' },
    hints: ['Differentiate with respect to w and set it to zero: −2 Σ xᵢ(yᵢ − w xᵢ) + 2αw = 0.', 'So w = Σ xᵢyᵢ / (Σ xᵢ² + α).', 'Σ xy = 29 and Σ x² = 14, so w = 29 / (14 + ' + ra + ') = ' + r3(rw) + ', smaller than the least-squares ' + r3(ow) + '.'],
    solution: String(r3(rw)), wrong: [String(r3(ow)), String(r3(29 / (14 - ra)))] });

  const SX = [[1], [2], [-1], [2]], SY = [3, 2, -1, 4], sa = 1.5, sw = ML.cdSolve(SX, SY, sa, 1, { maxIter: 1 }).w[0];
  const rho = sum(SX.map((r, i) => r[0] * SY[i])), sxx = sum(SX.map((r) => r[0] * r[0]));
  C.exercise({ id: 'li-x-reg-soft', chapter: 'li-reg', title: 'One coordinate-descent update', level: 2, kind: 'number',
    text: ['One feature x = (1, 2, −1, 2), targets y = (3, 2, −1, 4), no intercept. The lasso minimises (1/2n) Σ (yᵢ − w·xᵢ)² + λ|w| with n = 4 and λ = ' + sa + '. Starting from w = 0, coordinate descent sets w = S(ρ, nλ) / Σ xᵢ², with ρ = Σ xᵢ yᵢ and the soft threshold S(ρ, t) = sign(ρ)·max(|ρ| − t, 0). What is w after the update?'],
    answer: r3(sw), tolerance: 0.002, whyNot: { [String(r3(rho / sxx))]: 'That is least squares, with no threshold: ρ must first move toward zero by nλ.', [String(r3((rho - sa) / sxx))]: 'The threshold is nλ = ' + 4 * sa + ', not λ.' },
    hints: ['Compute ρ = Σ xᵢ yᵢ and Σ xᵢ².', 'ρ = ' + rho + ' and Σ xᵢ² = ' + sxx + '; the threshold is nλ = ' + 4 * sa + '.', 'S(' + rho + ', ' + 4 * sa + ') = ' + (rho - 4 * sa) + ', so w = ' + (rho - 4 * sa) + ' / ' + sxx + ' = ' + r3(sw) + '. With λ = 5 the threshold 20 would exceed ρ and give exactly 0.'],
    solution: String(r3(sw)), wrong: [String(r3(rho / sxx)), String(r3((rho - sa) / sxx))] });

  const soft = (r, t) => Math.sign(r) * Math.max(Math.abs(r) - t, 0);
  C.exercise({ id: 'li-x-reg-softcode', chapter: 'li-reg', title: 'Write the soft threshold', level: 2, kind: 'code',
    text: ['Write `softThreshold(rho, t)`: move rho toward zero by t (t ≥ 0), and return 0 if it would cross zero. This is the step that gives the lasso its exact zeros.'],
    fn: 'softThreshold', params: ['rho', 't'], tolerance: 1e-12, start: 'function softThreshold(rho, t) {\n  \n}',
    ref: soft, cases: [[16, 6], [-16, 6], [3, 5], [-3, 5], [5, 5], [0, 2], [2.5, 0]],
    hints: ['Three cases: rho > t, rho < −t, and everything in between.', 'Above t subtract t; below −t add t; in between the answer is 0.', 'function softThreshold(rho, t) {\n  return Math.sign(rho) * Math.max(Math.abs(rho) - t, 0);\n}'],
    solution: 'function softThreshold(rho, t) {\n  return Math.sign(rho) * Math.max(Math.abs(rho) - t, 0);\n}',
    wrong: ['function softThreshold(rho, t) {\n  return Math.abs(rho) > t ? rho : 0;\n}', 'function softThreshold(rho, t) {\n  return Math.max(rho - t, 0);\n}'] });

  const cdRef = (X, y, w, j, lam) => {
    const n = X.length; let rhoj = 0, sq = 0;
    for (let i = 0; i < n; i++) { let others = 0; for (let k = 0; k < w.length; k++) if (k !== j) others += X[i][k] * w[k]; rhoj += X[i][j] * (y[i] - others); sq += X[i][j] * X[i][j]; }
    return soft(rhoj, n * lam) / sq;
  };
  // the update, swept to convergence, must land on the engine's lasso (checked below in the hints)
  const CDX = [[1, 2, 0.5], [2, -1, 1], [0, 1, -1], [3, 1, 2], [1, 0, 0]], CDY = [4, 1, 0, 7, 1], cdl = 0.3;
  let cw = [0, 0, 0]; for (let s = 0; s < 200; s++) for (let j = 0; j < 3; j++) { cw = cw.slice(); cw[j] = cdRef(CDX, CDY, cw, j, cdl); }
  const lassoW = ML.Lasso({ alpha: cdl, fitIntercept: false }).fit(CDX, CDY).coef;
  C.exercise({ id: 'li-x-reg-cdcode', chapter: 'li-reg', title: 'Write a lasso coordinate update', level: 3, kind: 'code',
    text: ['The lasso without intercept minimises (1/2n) Σ (yᵢ − Σₖ xᵢₖwₖ)² + λ Σ |wₖ|. Write `lassoUpdate(X, y, w, j, lam)`, which returns the best new value of w[j] with every other weight held fixed:', { math: 'ρⱼ = Σᵢ xᵢⱼ (yᵢ − Σₖ≠ⱼ xᵢₖ wₖ)\nwⱼ = S(ρⱼ, n·λ) / Σᵢ xᵢⱼ²' }, 'S is the soft threshold; write it inside your function or beside it. Swept over j = 0, 1, 2, … until nothing changes, this update is the whole lasso solver.'],
    fn: 'lassoUpdate', params: ['X', 'y', 'w', 'j', 'lam'], tolerance: 1e-9, start: 'function lassoUpdate(X, y, w, j, lam) {\n  \n}',
    ref: cdRef, cases: [[SX, SY, [0], 0, sa], [SX, SY, [0], 0, 5], [CDX, CDY, [0, 0, 0], 0, cdl], [CDX, CDY, [1, 0.5, 0], 1, cdl], [CDX, CDY, [1.2, 0.3, 0.8], 2, cdl], [CDX, CDY, [0.5, 0.5, 0.5], 1, 2]],
    hints: ['Leave out feature j’s own contribution when you form what is left of y for each row.', 'Then ρⱼ is the sum of xᵢⱼ times that remainder; shrink it toward 0 by n·lam, and divide by Σ xᵢⱼ².', 'function lassoUpdate(X, y, w, j, lam) {\n  const n = X.length; let rho = 0, sq = 0;\n  for (let i = 0; i < n; i++) {\n    let others = 0;\n    for (let k = 0; k < w.length; k++) if (k !== j) others += X[i][k] * w[k];\n    rho += X[i][j] * (y[i] - others);\n    sq += X[i][j] * X[i][j];\n  }\n  return Math.sign(rho) * Math.max(Math.abs(rho) - n * lam, 0) / sq;\n}\nSwept 200 times over the three features of the third test, it reaches w = (' + cw.map(r3).join(', ') + '), the engine’s lasso gives (' + lassoW.map(r3).join(', ') + ').'],
    solution: 'function lassoUpdate(X, y, w, j, lam) {\n  const n = X.length; let rho = 0, sq = 0;\n  for (let i = 0; i < n; i++) {\n    let others = 0;\n    for (let k = 0; k < w.length; k++) if (k !== j) others += X[i][k] * w[k];\n    rho += X[i][j] * (y[i] - others);\n    sq += X[i][j] * X[i][j];\n  }\n  return Math.sign(rho) * Math.max(Math.abs(rho) - n * lam, 0) / sq;\n}',
    wrong: ['function lassoUpdate(X, y, w, j, lam) {\n  const n = X.length; let rho = 0, sq = 0;\n  for (let i = 0; i < n; i++) { let all = 0; for (let k = 0; k < w.length; k++) all += X[i][k] * w[k]; rho += X[i][j] * (y[i] - all); sq += X[i][j] * X[i][j]; }\n  return Math.sign(rho) * Math.max(Math.abs(rho) - n * lam, 0) / sq;\n}',
      'function lassoUpdate(X, y, w, j, lam) {\n  let rho = 0, sq = 0;\n  for (let i = 0; i < X.length; i++) { let others = 0; for (let k = 0; k < w.length; k++) if (k !== j) others += X[i][k] * w[k]; rho += X[i][j] * (y[i] - others); sq += X[i][j] * X[i][j]; }\n  return Math.sign(rho) * Math.max(Math.abs(rho) - lam, 0) / sq;\n}'] });

  C.exercise({ id: 'li-x-reg-path', chapter: 'li-reg', title: 'Read the lasso path', level: 2, kind: 'set',
    text: ['The lasso path on 60 rows with 8 standardised features, from the strongest penalty down. Step the figure forward to λ = 0.212.', { topic: 'reg-paths', frame: 0, caption: 'Step forward with the arrow keys or the play controls.' }, 'Which features have non-zero weights at λ = 0.212? Write them as a set, like {x1, x4}.'],
    answer: '{x1, x2, x3, x5, x8}', caseless: true,
    hints: ['Each step weakens the penalty; a feature joins when its weight leaves zero.', 'x1 joins first, at λ = 1.62; x2 and x3 together at λ = 1.08.', 'At λ = 0.212, x5 and x8 have just joined: {x1, x2, x3, x5, x8}.'],
    solution: '{x1, x2, x3, x5, x8}', wrong: ['{x1, x2, x3}', '{x1, x2, x3, x4, x5, x6, x7, x8}'] });

  C.exercise({ id: 'li-x-reg-twins', chapter: 'li-reg', title: 'Two near copies', level: 1, kind: 'choice',
    text: ['A shop’s sales data has two columns that are near copies: the floor area in square metres, and the same area estimated from the number of shelves. Both carry real information about sales. What does the lasso tend to do with them?'],
    options: ['Keep one and set the other to exactly 0', 'Give both the same weight', 'Set both to 0', 'Give both large weights of opposite signs'], answer: 0,
    whyNot: { 1: 'That is closer to what ridge, or the elastic net, does: the L2 part spreads the weight over twins.', 2: 'They are useful, so the lasso keeps at least one of them unless λ is very large.', 3: 'That is what plain least squares can do with near-copies; the penalty prevents it.' },
    hints: ['The lasso charges |w₁| + |w₂|. Does it cost more to split the weight between twins or give it all to one?', 'For the fit the twins are interchangeable, and for the penalty a split costs the same as giving it all to one: so a tiny edge in fit decides.', 'It keeps whichever twin fits slightly better and sets the other to exactly 0; the elastic net moves them together.'],
    solution: 0, wrong: [1, 2] });

  // ridge on degree-8 polynomial features (standardised) of twenty points (S.Data 'curve', sine, noise 0.3, seed 2), scored
  // on 200 fresh points (seed 9) at α = 10^e for e = −4, −3.5, …, 2. Printed by the engines (ML.PolynomialFeatures,
  // ML.Ridge) in a scratch script and kept here, because thirteen ridge fits cost more than a page load should.
  const lamCurve = [[-4, 0.252], [-3.5, 0.235], [-3, 0.202], [-2.5, 0.161], [-2, 0.136], [-1.5, 0.129], [-1, 0.13], [-0.5, 0.135], [0, 0.137], [0.5, 0.143], [1, 0.2], [1.5, 0.305], [2, 0.41]];
  const lamBest = lamCurve.reduce((a, b) => (b[1] < a[1] ? b : a));
  C.exercise({ id: 'li-x-reg-lambda', chapter: 'li-reg', title: 'The test error as the penalty grows', level: 3, kind: 'predict-curve',
    text: ['Ridge regression on degree-8 polynomial features (standardised) of twenty noisy points on a wave, scored on 200 fresh points. The penalty α runs from 10⁻⁴ to 10², drawn on a log scale. Sketch the test error. Its values at α = 10⁻⁴ and 10⁻³·⁵ are drawn.'],
    plot: { xr: [-4, 2], yr: [0, 0.5], xname: 'log₁₀ α', yname: 'test error', given: lamCurve.slice(0, 2) },
    answer: lamCurve, tolerance: 0.04, checks: ['valley'],
    hints: ['At the left the penalty is almost nothing: the degree-8 curve overfits. At the right it is so strong that the curve is nearly flat.', 'Somewhere in between the penalty removes the wiggles but keeps the wave: the error has a lowest point.', 'It falls from ' + lamCurve[0][1] + ' to its lowest, ' + lamBest[1] + ' at log₁₀ α = ' + lamBest[0] + ', then climbs to ' + lamCurve[12][1] + ' at α = 100.'],
    solution: lamCurve, wrong: [lamCurve.map(([e, v]) => [e, r3(lamCurve[0][1] - (lamCurve[0][1] - 0.05) * (e + 4) / 6)]), lamCurve.map(([e]) => [e, 0.4])] });

  const wave = { data: { set: 'curve', n: 30, seed: 5, opts: { shape: 'sine', noise: 0.25 } }, hidden: { seed: 70, n: 400 }, metric: 'rmse', target: 0.3 };
  C.exercise(Object.assign({ id: 'li-x-reg-wf', chapter: 'li-reg', title: 'Tame a degree-10 polynomial', level: 3, kind: 'workflow', task: 'regression',
    text: ['Thirty noisy points on a wave, and a rule: the polynomial features must have degree 10. Plain least squares on them swings wildly between the points. Build a workflow that reaches RMSE ≤ 0.3 on hidden points from the same wave.'],
    limits: { blocks: ['split', 'poly', 'scale', 'linreg', 'ridge', 'lasso', 'train', 'evaluate'], params: { poly: { degree: [10, 10] } } },
    hints: ['Look at the Train block of Polynomial features (degree 10) → Linear regression: the curve passes near the points and shoots off between them.', 'A penalty holds the weights back; it treats them fairly only if the polynomial columns are on one scale, so put a Scale block after the polynomial one.', 'Polynomial features (degree 10) → Scale (standard) → Ridge with α = 0.01: RMSE about 0.26.'],
    solution: pipe(wave.data, 'ridge', { alpha: 0.01 }, {}, [['poly', { degree: 10 }], ['scale', {}]]),
    wrong: [pipe(wave.data, 'linreg', {}, {}, [['poly', { degree: 10 }]]), pipe(wave.data, 'ridge', { alpha: 1 }, {}, [['poly', { degree: 10 }], ['scale', {}]])] }, wave));

  // =========================================================================================================
  // Chapter 3: linear classifiers
  // =========================================================================================================
  const sz = 0.8 * 5 - 3, sg = ML.sigmoid(sz);
  C.exercise({ id: 'li-x-cl-sigmoid', chapter: 'li-class', title: 'A probability from a score', level: 1, kind: 'number',
    text: ['A logistic model gives z = 0.8·x − 3. What probability p = σ(z) = 1 / (1 + e^−z) does it give at x = 5? Give three decimals.'],
    answer: r3(sg), tolerance: 0.0015, whyNot: { [String(r3(1 - sg))]: 'That is σ(−z), the probability of the other class.', [String(r3(sz))]: 'That is the score z itself; it still has to go through the sigmoid.' },
    hints: ['First the score: z = 0.8·5 − 3.', 'z = ' + sz + ', so p = 1 / (1 + e^−1).', 'e^−1 = ' + r3(Math.exp(-1)) + ', so p = 1 / ' + r3(1 + Math.exp(-1)) + ' = ' + r3(sg) + '.'],
    solution: String(r3(sg)), wrong: [String(r3(1 - sg)), String(r3(sz))] });

  const LLp = [0.9, 0.2, 0.6], LLy = [1, 1, 0], lls = LLp.map((p, i) => -Math.log(LLy[i] ? p : 1 - p)), llm = S.Metrics.logLoss(LLy, LLp.map((p) => [1 - p, p])).value;
  C.exercise({ id: 'li-x-cl-logloss', chapter: 'li-class', title: 'The log loss of three students', level: 1, kind: 'number',
    text: ['A model gives three students P(pass) = ' + LLp.join(', ') + '. The first two passed and the third failed. What is the mean log loss? Give three decimals.'],
    answer: r3(llm), tolerance: 0.0015, whyNot: { [String(r3((-Math.log(0.9) - Math.log(0.2) - Math.log(0.6)) / 3))]: 'The third student failed: the loss uses the probability of failing, 1 − 0.6.', [String(r3(sum(lls)))]: 'That is the total over the three students; the mean divides it by 3.' },
    hints: ['Each student pays −ln of the probability the model gave to what happened.', 'The third failed, so it pays −ln (1 − 0.6) = −ln 0.4.', 'The losses are ' + lls.map(r3).join(', ') + '; their mean is ' + r3(llm) + '. The second student, a pass given only 0.2, costs the most.'],
    solution: String(r3(llm)), wrong: [String(r3((-Math.log(0.9) - Math.log(0.2) - Math.log(0.6)) / 3)), String(r3(sum(lls)))] });

  const QX = [[1], [2], [3]], QY = [0, 1, 1], Q = ML.LogisticRegression({ solver: 'gd', penalty: 'none', lr: 1, epochs: 1 }).fit(QX, QY);
  const qw = r3(Q.coef[0][0]), qb = r3(Q.intercept[0]), qgw = r3(-qw), qgb = r3(-qb);
  const qcells = { 'p (every student)|value': 0.5, '∂L/∂w|value': qgw, '∂L/∂b|value': qgb, 'new w|value': qw, 'new b|value': qb };
  C.exercise({ id: 'li-x-cl-grad', chapter: 'li-class', title: 'One step of logistic regression', level: 2, kind: 'table',
    text: ['Three students studied 1, 2 and 3 hours; the first failed (y = 0) and the other two passed (y = 1). Logistic regression starts at w = 0, b = 0 and takes one gradient step on the mean log loss with η = 1. Fill in the table to three decimals.', { math: '∂L/∂w = (1/n) Σ (pᵢ − yᵢ) xᵢ\n∂L/∂b = (1/n) Σ (pᵢ − yᵢ)' }],
    rows: ['p (every student)', '∂L/∂w', '∂L/∂b', 'new w', 'new b'], cols: ['value'], cells: qcells, tolerance: 0.002, corner: 'at w = 0, b = 0',
    hints: ['With w = 0 and b = 0, every score is 0 and every p is σ(0).', 'The errors p − y are 0.5, −0.5 and −0.5; weight them by the hours for ∂L/∂w.', 'p = 0.5; ∂L/∂w = (0.5·1 − 0.5·2 − 0.5·3) / 3 = ' + qgw + ' and ∂L/∂b = ' + qgb + '; the step gives w = ' + qw + ' and b = ' + qb + '.'],
    solution: sol(qcells), wrong: [sol(Object.assign({}, qcells, { 'new w|value': qgw, 'new b|value': qgb })), sol(Object.assign({}, qcells, { '∂L/∂w|value': r3(qgw * 3), 'new w|value': r3(qw * 3) }))] });

  const lgRef = (X, y, w, b, lr) => {
    const n = X.length, g = new Array(w.length).fill(0); let gbb = 0;
    for (let i = 0; i < n; i++) { const e = ML.sigmoid(X[i].reduce((s, v, j) => s + v * w[j], b)) - y[i]; gbb += e; for (let j = 0; j < w.length; j++) g[j] += e * X[i][j]; }
    return w.map((v, j) => v - (lr * g[j]) / n).concat([b - (lr * gbb) / n]);
  };
  const lgCheck = lgRef(QX, QY, [0], 0, 1);
  C.exercise({ id: 'li-x-cl-code', chapter: 'li-class', title: 'Write a logistic-regression step', level: 3, kind: 'code',
    text: ['Write `logisticStep(X, y, w, b, lr)`: one gradient-descent step on the mean log loss of logistic regression. X is an array of rows, y holds 0s and 1s, w an array of weights. Return the new weights followed by the new intercept, like `[w0, w1, b]`.'],
    fn: 'logisticStep', params: ['X', 'y', 'w', 'b', 'lr'], tolerance: 1e-9, start: 'function logisticStep(X, y, w, b, lr) {\n  \n}',
    ref: lgRef, cases: [[QX, QY, [0], 0, 1], [[[1, 2], [2, 0], [-1, 1], [0, -2]], [1, 1, 0, 0], [0.2, -0.1], 0.3, 0.5], [[[3, 1], [1, 1], [2, -1]], [0, 1, 1], [1, 1], -1, 0.1], [[[0.5]], [1], [2], -2, 2]],
    hints: ['For each row: the score z = w·x + b, the probability p = 1 / (1 + e^−z), and the error p − y.', 'The gradient for wⱼ is the mean of (p − y)·xⱼ, for b the mean of (p − y). Then subtract lr times each.', 'function logisticStep(X, y, w, b, lr) {\n  const n = X.length, g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < n; i++) {\n    const z = X[i].reduce((s, v, j) => s + v * w[j], b);\n    const e = 1 / (1 + Math.exp(-z)) - y[i];\n    gb += e;\n    for (let j = 0; j < w.length; j++) g[j] += e * X[i][j];\n  }\n  return w.map((v, j) => v - lr * g[j] / n).concat([b - lr * gb / n]);\n}\nOn the first test it returns [' + lgCheck.map(r3).join(', ') + '], as the engine’s first step does.'],
    solution: 'function logisticStep(X, y, w, b, lr) {\n  const n = X.length, g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < n; i++) {\n    const z = X[i].reduce((s, v, j) => s + v * w[j], b);\n    const e = 1 / (1 + Math.exp(-z)) - y[i];\n    gb += e;\n    for (let j = 0; j < w.length; j++) g[j] += e * X[i][j];\n  }\n  return w.map((v, j) => v - lr * g[j] / n).concat([b - lr * gb / n]);\n}',
    wrong: ['function logisticStep(X, y, w, b, lr) {\n  const n = X.length, g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < n; i++) { const z = X[i].reduce((s, v, j) => s + v * w[j], b); const e = z - y[i]; gb += e; for (let j = 0; j < w.length; j++) g[j] += e * X[i][j]; }\n  return w.map((v, j) => v - lr * g[j] / n).concat([b - lr * gb / n]);\n}',
      'function logisticStep(X, y, w, b, lr) {\n  const n = X.length, g = w.map(() => 0); let gb = 0;\n  for (let i = 0; i < n; i++) { const z = X[i].reduce((s, v, j) => s + v * w[j], b); const e = 1 / (1 + Math.exp(-z)) - y[i]; gb += e; for (let j = 0; j < w.length; j++) g[j] += e * X[i][j]; }\n  return w.map((v, j) => v + lr * g[j] / n).concat([b + lr * gb / n]);\n}'] });

  // fourteen students: hours, and a pass drawn from the model's own probability
  const hr = ML.rng(12), HW = 0.4, HB = -3.4, hrs = ML.range(14).map(() => r2(1 + 14 * hr())), hy = hrs.map((h) => (hr() < ML.sigmoid(HW * h + HB) ? 1 : 0));
  const half = -HB / HW, sigLine = ML.range(33).map((k) => { const h = k * 0.5; return [h, r3(ML.sigmoid(HW * h + HB))]; });
  C.exercise({ id: 'li-x-cl-half', chapter: 'li-class', title: 'Where the model is undecided', level: 2, kind: 'point',
    text: ['A logistic model of exam results gives P(pass) = σ(' + HW + '·hours − ' + -HB + '). Its curve is drawn faint over fourteen students (1 = passed). Click the number of hours where the model is exactly undecided, P(pass) = 0.5.'],
    plot: { points: hrs.map((h, i) => [h, hy[i] ? 1 : 0]), classes: hy, xr: [0, 16], yr: [-0.1, 1.1], xname: 'hours', yname: 'pass', lines: [{ pts: sigLine }, { y: 0.5 }] }, mode: 'x',
    answer: r3(half), tolerance: 0.4,
    hints: ['σ(z) = 0.5 exactly when z = 0.', 'Solve ' + HW + '·hours − ' + -HB + ' = 0.', 'hours = ' + -HB + ' / ' + HW + ' = ' + r3(half) + '. Left of it the model says fail, right of it pass.'],
    solution: { x: half }, wrong: [{ x: 5 }, { x: 12 }] });

  const SM = [2, 1, -1], smp = ML.softmax(SM), smcells = { 'class A (score 2)|probability': r3(smp[0]), 'class B (score 1)|probability': r3(smp[1]), 'class C (score −1)|probability': r3(smp[2]) };
  C.exercise({ id: 'li-x-cl-softmax', chapter: 'li-class', title: 'Softmax of three scores', level: 2, kind: 'table',
    text: ['A softmax model gives a point the scores 2, 1 and −1 for classes A, B and C. Turn them into probabilities, pₖ = e^zₖ / Σⱼ e^zⱼ, to three decimals.'],
    rows: ['class A (score 2)', 'class B (score 1)', 'class C (score −1)'], cols: ['probability'], cells: smcells, tolerance: 0.002, corner: 'class',
    hints: ['Exponentiate each score: e², e¹ and e^−1.', 'e² = ' + r3(Math.E ** 2) + ', e = ' + r3(Math.E) + ', e^−1 = ' + r3(Math.exp(-1)) + '; their sum is ' + r3(Math.E ** 2 + Math.E + Math.exp(-1)) + '.', 'Divide each by the sum: ' + smp.map(r3).join(', ') + '. They add up to 1.'],
    solution: sol(smcells), wrong: [sol({ 'class A (score 2)|probability': r3(2 / 2), 'class B (score 1)|probability': r3(1 / 2), 'class C (score −1)|probability': 0 }), sol({ 'class A (score 2)|probability': r3(2 / 3), 'class B (score 1)|probability': r3(1 / 3), 'class C (score −1)|probability': 0 })] });

  const sh = ML.softmax([101, 102, 103]);
  C.exercise({ id: 'li-x-cl-shift', chapter: 'li-class', title: 'A hundred more for every class', level: 1, kind: 'choice',
    text: ['A softmax model gives scores 1, 2 and 3 to three classes. A bug adds 100 to all three scores, giving 101, 102 and 103. What happens to the three probabilities?'],
    options: ['They do not change', 'The largest becomes almost 1', 'They all become almost equal', 'They no longer add up to 1'], answer: 0,
    whyNot: { 1: 'The gaps between the scores are what decide the probabilities, and the gaps are unchanged.', 2: 'Equal probabilities need equal scores; the scores still differ by 1 and 2.', 3: 'Softmax divides by the sum, so the probabilities always add up to 1.' },
    hints: ['e^(z + 100) = e^100 · e^z.', 'The factor e^100 appears in every term above and below the fraction.', 'It cancels: the probabilities stay ' + sh.map(r3).join(', ') + '. Only differences between scores matter.'],
    solution: 0, wrong: [1, 2] });

  const PCX = [[1, 2], [3, -1]], PCY = [1, -1], PC = ML.Perceptron({ epochs: 1 }).fit(PCX, PCY);
  const pcells = { 'after point 1|w₁': 1, 'after point 1|w₂': 2, 'after point 1|b': 1, 'after point 2|w₁': PC.W[0][0], 'after point 2|w₂': PC.W[0][1], 'after point 2|b': PC.b[0] };
  C.exercise({ id: 'li-x-cl-perc', chapter: 'li-class', title: 'Two perceptron updates', level: 2, kind: 'table',
    text: ['A perceptron starts at w = (0, 0), b = 0 with η = 1 and visits two points in order: point 1 at (1, 2) with label +1, then point 2 at (3, −1) with label −1. A point is a mistake when t·(w·x + b) ≤ 0, and a mistake updates w ← w + t·x, b ← b + t. Fill in the weights after each point.'],
    rows: ['after point 1', 'after point 2'], cols: ['w₁', 'w₂', 'b'], cells: pcells, corner: 'weights',
    hints: ['Point 1 scores 0 with the zero weights: 0 ≤ 0 counts as a mistake.', 'After point 1, w = (1, 2) and b = 1. Point 2 then scores 3 − 2 + 1 = 2, but its label is −1.', 'Point 2 is a mistake too: w = (1, 2) − (3, −1) = (' + PC.W[0].join(', ') + ') and b = 1 − 1 = ' + PC.b[0] + '.'],
    solution: sol(pcells), wrong: [sol(Object.assign({}, pcells, { 'after point 2|w₁': 1, 'after point 2|w₂': 2, 'after point 2|b': 1 })), sol(Object.assign({}, pcells, { 'after point 2|w₁': 4, 'after point 2|w₂': 1, 'after point 2|b': 2 }))] });

  C.exercise({ id: 'li-x-cl-overlap', chapter: 'li-class', title: 'A perceptron on overlapping classes', level: 1, kind: 'choice',
    text: ['The two classes overlap: no straight line puts every point on its own side. What does the perceptron do, pass after pass?'],
    options: ['It keeps making updates and never settles', 'It stops at the line with the fewest mistakes', 'It stops after one pass', 'It finds the widest gap'], answer: 0,
    whyNot: { 1: 'Nothing in the rule remembers the best line so far: every mistake moves the line again.', 2: 'It stops only after a pass with no mistakes, which never comes here.', 3: 'Even on separable data the perceptron does not look for the widest gap; that is a support vector machine.' },
    hints: ['When does the perceptron stop updating?', 'Only after a pass in which every point is on its own side.', 'With overlapping classes some point is always wrong, so it keeps updating: in the topic, 17 updates and still 8 points wrong.'],
    solution: 0, wrong: [1, 3] });

  const exam = { data: { set: 'exam', n: 120, seed: 4 }, hidden: { seed: 44, n: 400 }, metric: 'accuracy', target: 0.78 };
  C.exercise(Object.assign({ id: 'li-x-cl-wf', chapter: 'li-class', title: 'Pass or fail, with a straight boundary', level: 2, kind: 'workflow', task: 'classification',
    text: ['Students’ hours of study, hours of sleep and practice papers, and whether they passed. Build a workflow with a linear classifier that reaches accuracy 0.78 on hidden students.'],
    limits: { blocks: ['split', 'scale', 'logistic', 'perceptron', 'train', 'evaluate'] },
    hints: ['Both allowed models draw a straight boundary; they differ in how they place it.', 'The perceptron stops at any separating compromise and is thrown about by overlapping points; logistic regression weighs every point by its log loss.', 'Data → Split → Logistic regression → Train → Evaluate: accuracy about 0.81.'],
    solution: pipe(exam.data, 'logistic'), wrong: [pipe(exam.data, 'perceptron'), pipe(exam.data, 'perceptron', {}, {}, [['scale', {}]])] }, exam));

  C.exercise({ id: 'li-x-cl-choose', chapter: 'li-class', title: 'When a straight boundary is enough', level: 2, kind: 'choice',
    text: ['For which of these datasets is plain logistic regression, with no extra features, a sensible first model? Pick all that apply.'],
    options: ['Pass or fail against hours of study, where more study steadily helps', 'Two classes in opposite corners of a square (XOR)', 'Loan defaults, whose risk rises steadily with debt and falls with income', 'Points inside a ring against points outside it'], answer: [0, 2],
    whyNot: { 1: 'No straight line does better than about 75% on XOR, by cutting off one corner, and logistic regression itself stays near 50%.', 3: 'A ring needs a closed boundary. Squared features such as x₁² + x₂² would make it linear again.' },
    hints: ['Logistic regression’s boundary is a straight line, and its probability changes steadily across it.', 'Ask of each dataset: does one direction in feature space steadily raise the chance of class 1?', 'Hours of study and the loans fit a straight boundary; XOR and the ring do not.'],
    solution: [0, 2], wrong: [[0, 1, 2, 3], [0]] });
})(globalThis.Nooral = globalThis.Nooral || {});
