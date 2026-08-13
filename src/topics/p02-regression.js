// Part 2, regression errors: predicted against actual with the line of perfect predictions, the residuals against
// the prediction and their histogram, MAE, MSE, RMSE and R² worked from the residuals, and what the residual plot
// shows that the numbers hide: a curve the model missed, or one wild row that the squared errors magnify.
(function (S) {
  'use strict';
  const H = S.P01, C = () => S.Kit.curve, DT = () => S.Kit.dt, M = () => S.Metrics, ML = () => S.ML;
  const CASES = {
    rents: { label: 'Rents: a linear model that fits', set: 'rents', cols: (r) => r.slice() },
    curve: { label: 'A line where the truth bends', set: 'bend', cols: (r) => [r[0]] },
    curve2: { label: 'The same, with x² added', set: 'bend', cols: (r) => [r[0], r[0] * r[0]] },
    typo: { label: 'Rents with one mistyped test rent', set: 'rents', cols: (r) => r.slice(), typo: true },
  };
  /** a curved trend: y = e^(x / 1.5) plus noise, x uniform in [0, 3] */
  const bend = () => S.P02.once('bend', () => { const r = H.rng(21), X = [], y = []; for (let i = 0; i < 300; i++) { const x = +r.range(0, 3).toFixed(3); X.push([x]); y.push(+(Math.exp(x / 1.5) + r.normal(0, 0.6)).toFixed(3)); } return { X, y, target: { name: 'y', type: 'number' } }; });
  S.P02.regCase = (key) => {
    const c = CASES[key], ds = c.set === 'bend' ? bend() : H.data(c.set), X = ds.X.map(c.cols), y = ds.y.slice(), ml = ML();
    const sp = ml.trainTestSplit(X.length, { testSize: 0.3, seed: 6 }), tr = sp.train.slice().sort((a, b) => a - b), te = sp.test.slice().sort((a, b) => a - b);
    const m = ml.LinearRegression().fit(tr.map((i) => X[i]), tr.map((i) => y[i]));
    const yt = te.map((i) => y[i]);
    if (c.typo) yt[5] = yt[5] * 10;
    return { yt, yp: m.predict(te.map((i) => X[i])), target: H.short(ds.target), unit: ds.target.unit || '', title: c.label, nTrain: tr.length };
  };

  S.topic({
    id: 'eval-regression', family: 'p02-regression', order: 10, title: 'Regression metrics and residual plots',
    blurb: 'For a number, the error of one prediction is its residual: actual minus predicted. MAE averages their sizes, in the units of the target. MSE averages their squares, so big misses count far more; RMSE takes its square root to return to the units. R² compares the squared error with that of always predicting the mean: 1 is perfect, 0 is no better than the mean. Plot the residuals against the predictions too: a model that fits leaves a shapeless band around zero, and a pattern means it missed something.',
    inputs: [{ id: 'case', kind: 'choice', label: 'Case', def: 'rents', options: Object.entries(CASES).map(([value, c]) => ({ value, label: c.label })) }],
    presets: Object.entries(CASES).map(([value, c]) => ({ label: c.label, values: { case: value } })),
    counters: [{ id: 'n', label: 'test rows' }],
    run(inp) {
      const R = S.P02.regCase(inp.case), yt = R.yt, yp = R.yp, n = yt.length, m = M();
      const res = m.residuals(yt, yp), r = res.value, mae = m.mae(yt, yp).value, mse = m.mse(yt, yp).value, rmse = m.rmse(yt, yp).value, r2o = m.r2(yt, yp);
      const lo = Math.min(...yt, ...yp), hi = Math.max(...yt, ...yp), [a0, a1] = H.span(lo, hi, 0.04);
      const ax = C().axes({ prefix: 'ax', x: 56, y: 50, w: 300, h: 300, xr: [a0, a1], yr: [a0, a1], xname: 'predicted', yname: 'actual ' + R.target, tickSize: 12 });
      const rmax = Math.max(...r.map(Math.abs)) * 1.08, rr = C().ticks(-rmax, rmax, 4);
      const bx = C().axes({ prefix: 'bx', x: 440, y: 50, w: 330, h: 170, xr: [a0, a1], yr: [-rmax, rmax], yticks: rr, xname: 'predicted', yname: 'residual', tickSize: 12 });
      const ideal = { key: 'id', kind: 'path', d: `M${ax.X(a0)},${ax.Y(a0)} L${ax.X(a1)},${ax.Y(a1)}`, cls: 'diag', audit: 'ignore', layer: 'back', z: 3 };
      const zero = { key: 'z0', kind: 'path', d: `M${bx.o.x},${bx.Y(0)} H${bx.o.x + bx.o.w}`, cls: 'diag', audit: 'ignore', layer: 'back', z: 3 };
      const big = res.largest;
      const pts = yt.map((v, i) => ({ key: 'p' + i, kind: 'dot', x: ax.X(yp[i]), y: ax.Y(v), r: 3, cls: 'sp c0' + (i === big ? ' wrong' : ''), audit: 'overlap-ok' }));
      const sticks = yt.map((v, i) => ({ key: 's' + i, kind: 'path', d: `M${ax.X(yp[i])},${ax.Y(yp[i])} V${ax.Y(v)}`, cls: 'resid', audit: 'ignore', layer: 'lines' }));
      const rpts = r.map((v, i) => ({ key: 'q' + i, kind: 'dot', x: bx.X(yp[i]), y: bx.Y(v), r: 3, cls: 'sp ' + (v >= 0 ? 'c0' : 'c1') + (i === big ? ' wrong' : ''), audit: 'overlap-ok' }));
      const F = new S.Frames(['n']), view = H.box(-14, -8, 790, 470);
      const title = H.title('tt', 0, 6, R.title + ': ' + n + ' test rows, linear regression');
      const lines = (ls) => ls.map((l, i) => H.text('ln' + i, 440, 280 + i * 26, l[0], l[1] || 'strong', 14));
      const u = R.unit ? ' ' + R.unit : '';
      F.set('n', n);
      F.add({ items: [title, ...ax.items, ideal, ...pts], say: `Each test row at its prediction (across) and its true ${R.target} (up). Perfect predictions would sit on the dashed line.`, meta: { phase: 'scatter', n }, view });
      F.add({ items: [title, ...ax.items, ideal, ...sticks, ...pts, ...bx.items, zero, ...rpts], say: `The residual of a row is actual − predicted: its vertical distance from the line, drawn again on the right against the prediction. The largest is ${H.g(r[big], 4)}${u} (actual ${H.g(yt[big], 4)}, predicted ${H.g(yp[big], 4)}).`, meta: { phase: 'resid', big: r[big], yt: yt[big], yp: yp[big] }, view });
      const sAbs = H.sum(r.map(Math.abs)), sSq = H.sum(r.map((v) => v * v));
      F.add({ items: [title, ...ax.items, ideal, ...sticks, ...pts, ...bx.items, zero, ...rpts, ...lines([['MAE = Σ|r| / n = ' + H.g(sAbs, 4) + ' / ' + n + ' = ' + H.g(mae, 4), 'strong'], ['MSE = Σr² / n = ' + H.g(mse, 4), 'strong'], ['RMSE = √MSE = ' + H.g(rmse, 4) + u, 'act']])], say: `MAE = ${H.g(sAbs, 4)} / ${n} = ${H.g(mae, 4)}${u}: the typical size of a miss. MSE = ${H.g(sSq, 4)} / ${n} = ${H.g(mse, 4)}, in squared units, and RMSE = √MSE = ${H.g(rmse, 4)}${u}. RMSE is at least MAE, and the more uneven the misses, the bigger the gap.`, meta: { phase: 'errors', mae, mse, rmse, sAbs, sSq }, view });
      F.add({ items: [title, ...ax.items, ideal, ...sticks, ...pts, ...bx.items, zero, ...rpts, ...lines([['SS_res = ' + H.g(r2o.ssRes, 4), 'strong'], ['SS_tot = ' + H.g(r2o.ssTot, 4) + ' (around the mean)', 'strong'], ['R² = 1 − SS_res / SS_tot = ' + H.f(r2o.value, 3), 'act']])], say: `R² compares the model with always predicting the mean of the test targets: SS_res = ${H.g(r2o.ssRes, 4)} against SS_tot = ${H.g(r2o.ssTot, 4)}, so R² = 1 − ${H.g(r2o.ssRes, 4)} / ${H.g(r2o.ssTot, 4)} = ${H.f(r2o.value, 3)}.`, meta: { phase: 'r2', r2: r2o.value, ssRes: r2o.ssRes, ssTot: r2o.ssTot }, view });
      // the pattern: mean residual over the lowest, middle and highest thirds of the predictions
      const ord = yp.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map((q) => q[1]), third = Math.floor(n / 3);
      const groups = [ord.slice(0, third), ord.slice(third, n - third), ord.slice(n - third)], gm = groups.map((g) => H.mean(g.map((i) => r[i])));
      const hist = H.histogram(r, 16, -rmax, rmax), hx = C().axes({ prefix: 'hx', x: 440, y: 290, w: 330, h: 90, xr: [-rmax, rmax], yr: [0, Math.max(...hist.counts) * 1.1], xticks: rr, yticks: 0, xname: 'residual', grid: false, tickSize: 12 });
      let say;
      if (inp.case === 'typo') {
        const keep = r.map((_, i) => i).filter((i) => i !== big), mae2 = m.mae(keep.map((i) => yt[i]), keep.map((i) => yp[i])).value, rmse2 = m.rmse(keep.map((i) => yt[i]), keep.map((i) => yp[i])).value;
        say = `One mistyped rent (${H.g(yt[big], 4)}) sits far from the band. Without it MAE would be ${H.g(mae2, 4)} instead of ${H.g(mae, 4)}, but RMSE ${H.g(rmse2, 4)} instead of ${H.g(rmse, 4)}: squaring lets a single row dominate.`;
        F.add({ items: [title, ...ax.items, ideal, ...sticks, ...pts, ...bx.items, zero, ...rpts, ...hx.items, ...DT().hist(hx, 'rh', hist.counts, hist.edges, 'soft')], say, meta: { phase: 'pattern', gm, mae2, rmse2 }, view });
      } else {
        const spread = Math.max(...gm) - Math.min(...gm), shape = spread > 0.5 * rmse ? 'bend' : 'flat';
        say = `Mean residual over the lowest, middle and highest thirds of the predictions: ${gm.map((v) => H.g(v, 3)).join(', ')}.` + (shape === 'bend' ? ` They differ by ${H.g(spread, 3)}, more than half the RMSE: the residuals move with the prediction, so the model missed a pattern.` : ` They differ by only ${H.g(spread, 3)}, under half the RMSE: no pattern, only scatter.`);
        F.add({ items: [title, ...ax.items, ideal, ...sticks, ...pts, ...bx.items, zero, ...rpts, ...groups.map((g, k) => ({ key: 'gm' + k, kind: 'path', d: `M${bx.X(yp[g[0]])},${bx.Y(gm[k])} H${bx.X(yp[g[g.length - 1]])}`, cls: 'series hot', audit: 'ignore', layer: 'top' })), ...hx.items, ...DT().hist(hx, 'rh', hist.counts, hist.edges, 'soft')], say, meta: { phase: 'pattern', gm, shape, spread }, view });
      }
      const ctx = { n, mae, rmse, big: yt[big] };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Regression errors', text: ['residual r = actual − predicted', 'MAE = Σ|r| / n', 'MSE = Σr² / n,  RMSE = √MSE', 'R² = 1 − Σr² / Σ(y − ȳ)²', 'R² < 0: worse than predicting the mean', 'residual plot: look for shape, not just size'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'resid', pick: 'option', ask: 'Which will be larger: RMSE or MAE?', options: ['RMSE', 'MAE', 'equal'], answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'errors').meta; return m.rmse > m.mae + 1e-9 ? 'RMSE' : m.mae > m.rmse + 1e-9 ? 'MAE' : 'equal'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'errors').say },
    ],
    quiz: [
      { q: 'Residuals 1, −1, 2, −2. What is the MAE?', options: ['1.5', '0', '2.5', '1.58'], answer: 0, why: '(1 + 1 + 2 + 2) / 4.' },
      { q: 'Same residuals. RMSE is…', options: ['√2.5 ≈ 1.58', '1.5', '2.5', '0'], answer: 0, why: 'MSE = (1 + 1 + 4 + 4) / 4 = 2.5.' },
      { q: 'R² = 0 means the model…', options: ['does no better than always predicting the mean', 'is perfect', 'is always wrong by 0', 'cannot be evaluated'], answer: 0, why: 'Its squared error equals the spread around the mean.' },
      { q: 'Residuals form a U shape against the predictions. What does it suggest?', options: ['a curve the model did not capture', 'a perfect model', 'too much data', 'nothing'], answer: 0, why: 'Systematic patterns are signal left on the table.' },
      { q: 'One huge error appears. Which metric jumps most?', options: ['MSE (and RMSE)', 'MAE', 'neither', 'they jump equally'], answer: 0, why: 'Squaring magnifies large residuals.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
