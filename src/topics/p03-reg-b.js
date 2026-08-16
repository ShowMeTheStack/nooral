// Part 3, why lasso gives zeros: the geometry. With two weights, the squared error is a bowl whose level curves are
// ellipses around the least-squares point. A penalty is the same as a budget on the size of the weights: a disc
// for ridge, a diamond for lasso. Shrink the budget and the best point is where the smallest ellipse touches the
// shape. A disc is touched anywhere; a diamond is usually touched at a corner, where one weight is exactly zero.
// The points are ML.Ridge and ML.Lasso fits, with λ found by bisection so each fit spends exactly the budget.
(function (S) {
  'use strict';
  const H = S.PL, ML = () => S.ML;
  const n$ = H.n;

  /** two standardised features with correlation about rho, y = w1 x1 + w2 x2 + noise (centred) */
  H.twoFeat = (rho, w1, w2, seed, n) => {
    const r = S.Data.rng(seed), X = [], y = [];
    for (let i = 0; i < (n || 50); i++) { const a = r.normal(0, 1), b = rho * a + Math.sqrt(1 - rho * rho) * r.normal(0, 1); X.push([a, b]); y.push(w1 * a + w2 * b + r.normal(0, 0.8)); }
    const st = H.standardise(X), ym = ML().mean(y);
    return { X: st.X, y: y.map((v) => v - ym) };
  };
  /** the fit whose weights have the given size: bisection on log λ (size(λ) falls as λ grows) */
  H.fitBudget = (make, size, target, lo, hi) => {
    let a = Math.log(lo), b = Math.log(hi), m = null;
    for (let it = 0; it < 46; it++) { const c = (a + b) / 2; m = make(Math.exp(c)); if (size(m.coef) > target) a = c; else b = c; }
    return { model: m, alpha: Math.exp((a + b) / 2) };
  };

  S.topic({
    id: 'reg-geometry', family: 'p03-reg', order: 20, title: 'Why lasso gives zeros: L1 and L2 balls',
    blurb: 'Penalising the weights is the same as giving them a budget. Ridge allows only weights inside a disc, w₁² + w₂² ≤ t²; lasso only inside a diamond, |w₁| + |w₂| ≤ t. The squared error has elliptical level curves around the least-squares point. As the budget shrinks, the answer is the point where the lowest ellipse just touches the shape. A disc has no corners, so the touching point rarely lies on an axis. A diamond sticks out at its corners, on the axes, so the ellipse usually touches there first, and one weight is exactly zero.',
    inputs: [
      { id: 'w1', kind: 'slider', label: 'true w₁', def: 2, min: -3, max: 3, step: 0.1 },
      { id: 'w2', kind: 'slider', label: 'true w₂', def: 0.7, min: -3, max: 3, step: 0.1 },
      { id: 'rho', kind: 'slider', label: 'correlation of x₁ and x₂', def: 0.5, min: -0.85, max: 0.85, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 1, min: 1, max: 9999 },
    ],
    check: (inp) => (Math.abs(inp.w1) + Math.abs(inp.w2) < 0.3 ? 'the true weights are too small to see; make one of them larger' : null),
    presets: [
      { label: 'One strong, one weak weight', values: { w1: 2, w2: 0.7, rho: 0.5, seed: 1 } },
      { label: 'Equal weights: both survive longer', values: { w1: 1.5, w2: 1.5, rho: 0, seed: 2 } },
      { label: 'Correlated features', values: { w1: 1.8, w2: 1, rho: 0.8, seed: 3 } },
      { label: 'Opposite signs', values: { w1: 2, w2: -1, rho: -0.4, seed: 4 } },
      { label: 'Only w₂ matters', values: { w1: 0, w2: 2.2, rho: 0.3, seed: 5 } },
    ],
    counters: [{ id: 'zeros', label: 'lasso zeros' }],
    run(inp) {
      const { X, y } = H.twoFeat(inp.rho, inp.w1, inp.w2, inp.seed, 50), n = X.length;
      const ols = ML().LinearRegression({ fitIntercept: false }).fit(X, y).coef;
      const A = ML().la.gram(X).map((r) => r.map((v) => v / n)), Ainv = ML().la.inv(A);
      const L = (w) => { let s = 0; for (let i = 0; i < n; i++) s += (y[i] - w[0] * X[i][0] - w[1] * X[i][1]) ** 2; return s / (2 * n); };
      const Lmin = L(ols), l1 = (w) => Math.abs(w[0]) + Math.abs(w[1]), l2 = (w) => Math.hypot(w[0], w[1]);
      const fracs = [1.25, 1, 0.85, 0.7, 0.55, 0.42, 0.3, 0.2, 0.1];
      const amaxL = Math.max(...[0, 1].map((j) => Math.abs(X.reduce((s, r, i) => s + r[j] * y[i], 0)))) / n;
      const steps = fracs.map((f) => {
        if (f >= 1) return { f, wr: ols.slice(), wl: ols.slice(), ar: 0, al: 0, t2: l2(ols) * f, t1: l1(ols) * f };
        const t2 = l2(ols) * f, t1 = l1(ols) * f;
        const R = H.fitBudget((a) => ML().Ridge({ alpha: a, fitIntercept: false, trace: false }).fit(X, y), l2, t2, 1e-6, 1e7);
        const La = H.fitBudget((a) => ML().Lasso({ alpha: a, fitIntercept: false, tol: 1e-11, trace: false }).fit(X, y), l1, t1, 1e-7, amaxL * 1.01);
        return { f, wr: R.model.coef.slice(), wl: La.model.coef.slice(), ar: R.alpha, al: La.alpha, t2, t1 };
      });
      const lim = Math.max(1.2, ...ols.map((v) => Math.abs(v) * 1.35)), rng = [-lim, lim];
      const panel = (key, x0, title) => H.axes({ prefix: key, x: x0, y: 50, w: 330, h: 330, xr: rng, yr: rng, xticks: H.tickList(-lim, lim, lim > 3 ? 2 : lim > 1.6 ? 1 : 0.5), yticks: H.tickList(-lim, lim, lim > 3 ? 2 : lim > 1.6 ? 1 : 0.5), xname: 'w₁', yname: 'w₂', grid: true });
      const axR = panel('gr', 46, 'ridge'), axL = panel('gl', 430, 'lasso');
      // level curves of the loss: ellipses (w − ŵ)ᵀ A (w − ŵ) = 2c around ŵ
      const levels = [0.08, 0.3, 0.7, 1.3, 2.2].map((c) => c * Lmin + 0.05);
      const ellipse = (ax, key, w, cls) => { const c = L(w) - Lmin; return c > 1e-9 ? H.ellipsePath(ax, key, ols, Ainv, Math.sqrt(2 * c), cls, true) : null; };
      const contours = (ax, P) => levels.map((c, k) => H.ellipsePath(ax, P + ':lv' + k, ols, Ainv, Math.sqrt(2 * c), 'pl-contour', true));
      const disc = (ax, key, t) => { const pts = []; for (let i = 0; i <= 72; i++) { const u = (2 * Math.PI * i) / 72; pts.push([ax.X(t * Math.cos(u)), ax.Y(t * Math.sin(u))]); } return { key, kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ',' + p[1]).join(' ') + ' Z', cls: 'pl-ball l2', audit: 'ignore', layer: 'back', z: 6 }; };
      const diamond = (ax, key, t) => ({ key, kind: 'path', d: `M${ax.X(t)},${ax.Y(0)} L${ax.X(0)},${ax.Y(t)} L${ax.X(-t)},${ax.Y(0)} L${ax.X(0)},${ax.Y(-t)} Z`, cls: 'pl-ball', audit: 'ignore', layer: 'back', z: 6 });
      const olsDot = (ax, key) => ({ key, kind: 'dot', x: ax.X(ols[0]), y: ax.Y(ols[1]), r: 6, cls: 'pl-star', audit: 'overlap-ok', layer: 'items', z: 4 });
      const sol = (ax, key, w, cls) => ({ key, kind: 'dot', x: ax.X(w[0]), y: ax.Y(w[1]), r: 6.5, cls: 'pl-probe ' + (cls || ''), audit: 'overlap-ok', layer: 'items', z: 5 });
      const trail = (ax, key, ws) => (ws.length > 1 ? { key, kind: 'path', d: ws.map((w, i) => (i ? 'L' : 'M') + ax.X(w[0]) + ',' + ax.Y(w[1])).join(' '), cls: 'pl-trail', pathAudit: 'edge-x', layer: 'lines' } : null);
      const heads = [H.title('gr:t', 46, 18, 'Ridge: a disc'), H.title('gl:t', 430, 18, 'Lasso: a diamond')];
      const F = new S.Frames(['zeros']);
      const fmtW = (w) => '(' + n$(w[0]) + ', ' + n$(w[1]) + ')';
      const meta0 = { ols, rho: inp.rho, Lmin };
      F.add({ items: [...heads, ...axR.items, ...axL.items, ...contours(axR, 'gr'), ...contours(axL, 'gl'), olsDot(axR, 'gr:ols'), olsDot(axL, 'gl:ols')], say: 'The least-squares weights are ŵ = ' + fmtW(ols) + ' (green), where the squared error is lowest, ' + n$(Lmin) + '. Around it the error rises in elliptical rings, tilted because x₁ and x₂ are correlated. Both panels show the same rings.', meta: Object.assign({ phase: 'ols' }, meta0) });
      steps.forEach((st, k) => {
        const zl = st.wl.filter((v) => v === 0).length;
        F.set('zeros', zl);
        const items = [...heads, ...axR.items, ...axL.items, ...contours(axR, 'gr'), ...contours(axL, 'gl'), disc(axR, 'gr:ball', st.t2), diamond(axL, 'gl:ball', st.t1), olsDot(axR, 'gr:ols'), olsDot(axL, 'gl:ols')];
        const er = ellipse(axR, 'gr:touch', st.wr, 'pl-contour touch'), el = ellipse(axL, 'gl:touch', st.wl, 'pl-contour touch');
        if (er) items.push(er);
        if (el) items.push(el);
        const tr1 = trail(axR, 'gr:trail', steps.slice(0, k + 1).map((q) => q.wr)), tr2 = trail(axL, 'gl:trail', steps.slice(0, k + 1).map((q) => q.wl));
        if (tr1) items.push(tr1);
        if (tr2) items.push(tr2);
        items.push(sol(axR, 'gr:w', st.wr, ''), sol(axL, 'gl:w', st.wl, zl ? 'act' : ''));
        items.push(...H.eqs('gr:eq', 46, 440, [['t = ' + n$(st.t2) + ',  λ = ' + n$(st.ar), 'muted'], ['w = ' + fmtW(st.wr), '']]), ...H.eqs('gl:eq', 430, 440, [['t = ' + n$(st.t1) + ',  λ = ' + n$(st.al), 'muted'], ['w = ' + fmtW(st.wl), zl ? 'act' : '']]));
        let say;
        if (st.f > 1) say = 'A generous budget: the disc (radius ' + n$(st.t2) + ') and the diamond (|w₁| + |w₂| ≤ ' + n$(st.t1) + ') both contain ŵ, so neither penalty changes anything: λ = 0.';
        else if (st.f === 1) say = 'The budget now equals the size of ŵ itself: ŵ sits on the edge of both shapes. Any smaller budget forces the weights to change.';
        else {
          say = 'Budget ' + Math.round(st.f * 100) + '% of ŵ\'s size. Ridge (λ = ' + n$(st.ar) + '): the lowest ring touches the disc at ' + fmtW(st.wr) + '. Lasso (λ = ' + n$(st.al) + '): it touches the diamond at ' + fmtW(st.wl);
          say += zl ? (zl === 2 ? ', both weights zero.' : ', a corner: w' + (st.wl[0] === 0 ? '₁' : '₂') + ' is exactly 0.') : ', on an edge: both weights still non-zero.';
        }
        F.add({ items, say, meta: Object.assign({ phase: 'budget', k, f: st.f, pct: st.f, t1: st.t1, t2: st.t2, wr: st.wr, wl: st.wl, ar: st.ar, al: st.al, zeros: zl }, meta0) });
      });
      const firstZero = steps.find((s) => s.wl.some((v) => v === 0) && s.f < 1);
      F.add({ items: F.last().items, say: firstZero ? 'Lasso first put a weight at exactly 0 at ' + Math.round(firstZero.f * 100) + '% of the budget, when the ellipse reached a corner of the diamond. Ridge shrank both weights all the way down and never zeroed either.' : 'Here the ellipses met the diamond on an edge all the way down: with these weights neither is dropped early. Try one strong and one weak weight.', meta: Object.assign({ phase: 'end', firstPct: firstZero ? firstZero.f : null }, meta0) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Penalty or budget', text: ['minimise L(w) + λ P(w)', '  ⇔  minimise L(w) with P(w) ≤ t', 'ridge: P = w₁² + w₂²  (a disc)', 'lasso: P = |w₁| + |w₂|  (a diamond)', 'answer: where the lowest ellipse of L', '  touches the shape'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'budget' && f.meta.f === 1, pick: 'option', ask: 'As the budget shrinks, which panel will first put a weight at exactly zero?', options: ['lasso (the diamond)', 'ridge (the disc)', 'both at once', 'neither'], answer: (fr) => (fr.some((f) => f.meta.phase === 'budget' && f.meta.f < 1 && f.meta.wl.some((v) => v === 0)) ? 'lasso (the diamond)' : 'neither'), explain: 'The diamond\'s corners lie on the axes; the disc has no corners.' },
    ],
    quiz: [
      { q: 'Where are the corners of the L1 ball |w₁| + |w₂| ≤ t?', options: ['on the axes, where one weight is zero', 'on the diagonals', 'nowhere: it is round', 'at the least-squares point'], answer: 0, why: '(t, 0), (0, t), (−t, 0), (0, −t).' },
      { q: 'Why do the level curves of the squared error look like ellipses?', options: ['the loss is a quadratic function of the weights', 'the data are Gaussian', 'the penalty makes them so', 'they are circles in disguise'], answer: 0, why: 'L(w) = L(ŵ) + ½(w − ŵ)ᵀ(XᵀX/n)(w − ŵ).' },
      { q: 'A bigger λ corresponds to…', options: ['a smaller budget t', 'a bigger budget', 'the same budget', 'no budget'], answer: 0, why: 'The stronger the penalty, the smaller the weights it allows.' },
      { q: 'If the budget already contains the least-squares point, the penalised answer is…', options: ['the least-squares point', 'zero', 'on the edge of the shape', 'undefined'], answer: 0, why: 'The constraint is not active.' },
      { q: 'Why does ridge almost never give an exact zero?', options: ['a disc has a smooth edge, so the touching point is almost never exactly on an axis', 'ridge is not convex', 'ridge has no penalty on small weights', 'it always does'], answer: 0, why: 'Only a special tilt of the ellipse would touch exactly at an axis.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
