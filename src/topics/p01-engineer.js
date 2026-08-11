// Part 1, feature engineering: polynomial features (one column x becomes x, x², …, x^d, so a linear model can fit
// a curve, with training and test error against the degree), and interaction features (the product x1·x2 or the
// squares let a linear classifier draw a boundary it never could on x1 and x2 alone).
(function (S) {
  'use strict';
  const H = S.P01, C = () => S.Kit.curve, SC = () => S.Kit.scatter, DT = () => S.Kit.dt, ML = () => S.ML;
  const sup = (k) => String(k).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]).join('');
  const pname = (k) => (k === 1 ? 'x' : 'x' + sup(k));

  const CURVES = {
    poly: { label: 'A cubic with noise', make: () => S.Data.curve({ n: 40, shape: 'poly', noise: 0.25, seed: 9 }) },
    sine: { label: 'A sine with noise', make: () => S.Data.curve({ n: 40, shape: 'sine', noise: 0.2, seed: 4 }) },
    abs: { label: 'A V shape', make: () => S.Data.curve({ n: 40, shape: 'abs', noise: 0.2, seed: 2 }) },
  };
  S.topic({
    id: 'data-poly', family: 'p01-engineer', order: 10, title: 'Polynomial features',
    blurb: 'A linear model is linear in its columns, not in the world. Give it the columns x, x², x³, … and it fits a polynomial: a curve, by the same least squares as a line. The degree is a dial. Too low and the curve cannot bend enough; too high and it bends to follow every noisy point, so the error on new rows climbs while the training error keeps falling.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'poly', options: Object.entries(CURVES).map(([value, c]) => ({ value, label: c.label })) },
      { id: 'degree', kind: 'slider', label: 'Degree', def: 3, min: 1, max: 12, step: 1 },
    ],
    presets: [
      { label: 'Cubic data, degree 3', values: { data: 'poly', degree: 3 } },
      { label: 'Cubic data, a line', values: { data: 'poly', degree: 1 } },
      { label: 'Degree 12: wiggles', values: { data: 'poly', degree: 12 } },
      { label: 'Sine, degree 5', values: { data: 'sine', degree: 5 } },
      { label: 'V shape, degree 2', values: { data: 'abs', degree: 2 } },
    ],
    counters: [{ id: 'cols', label: 'columns' }],
    run(inp) {
      const c = CURVES[inp.data].make(), x = c.X.map((r) => r[0]), y = c.y, n = x.length, ml = ML(), deg = inp.degree;
      const sp = ml.trainTestSplit(n, { testSize: 0.3, seed: 2 }), tr = sp.train, te = sp.test;
      const fitDeg = (d) => { const pf = ml.PolynomialFeatures({ degree: d, includeBias: false }).fit([[0]]); const m = ml.LinearRegression().fit(pf.transform(tr.map((i) => [x[i]])), tr.map((i) => y[i])); const p = (v) => m.predict(pf.transform([[v]]))[0]; const mse = (idx) => H.mean(idx.map((i) => (y[i] - p(x[i])) ** 2)); return { m, p, tr: mse(tr), te: mse(te) }; };
      const fits = Array.from({ length: 12 }, (_, k) => fitDeg(k + 1)), cur = fits[deg - 1], one = fits[0];
      const F = new S.Frames(['cols']), view = H.box(-14, -8, 790, 470);
      const [x0, x1] = [-3.1, 3.1], [y0, y1] = H.span(Math.min(...y), Math.max(...y), 0.15);
      const ax = C().axes({ prefix: 'ax', x: 50, y: 50, w: 400, h: 300, xr: [x0, x1], yr: [y0, y1], xname: 'x', yname: 'y', tickSize: 12 });
      const trS = new Set(tr), dots = x.map((v, i) => ({ key: 'p' + i, kind: 'dot', x: ax.X(v), y: ax.Y(y[i]), r: 3.6, cls: trS.has(i) ? 'sp c0' : 'sp c1 test', audit: 'overlap-ok' }));
      const curve = (fit, key, cls) => C().line(ax, key, Array.from({ length: 121 }, (_, k) => { const v = x0 + ((x1 - x0) * k) / 120; return [v, fit.p(v)]; }), cls);
      const title = H.title('tt', 0, 6, CURVES[inp.data].label + ': ' + tr.length + ' training points (blue), ' + te.length + ' test (orange)');
      const side = (lines, y0) => lines.map((l, k) => H.text('sd' + k, 480, (y0 || 70) + k * 25, l[0], l[1] || 'strong', 14));
      F.set('cols', 1);
      F.add({ items: [title, ...ax.items, curve(one, 'fit', 'bad'), ...dots, ...side([['one column: x', 'strong'], ['y ≈ ' + H.g(one.m.intercept, 3) + (one.m.coef[0] < 0 ? ' − ' : ' + ') + H.g(Math.abs(one.m.coef[0]), 3) + 'x', 'bad'], ['training MSE ' + H.g(one.tr, 3), 'strong'], ['test MSE ' + H.g(one.te, 3), 'strong']])], say: `With the single column x, least squares can only draw a line: y ≈ ${H.g(one.m.intercept, 3)} ${one.m.coef[0] < 0 ? '−' : '+'} ${H.g(Math.abs(one.m.coef[0]), 3)}x. Training MSE ${H.g(one.tr, 3)}, test MSE ${H.g(one.te, 3)}.`, meta: { phase: 'line', tr: one.tr, te: one.te, b: one.m.intercept, w: one.m.coef[0] }, view });
      // the expanded table for a few rows
      const showI = tr.slice(0, 6), dd = Math.min(deg, 4);
      const cols = [{ name: 'x', w: 56, cls: 'num' }].concat(Array.from({ length: dd - 1 }, (_, k) => ({ name: pname(k + 2), w: 56, cls: 'new' })), deg > 4 ? [{ name: '…' + pname(deg), w: 56, cls: 'new' }] : []);
      const rows = showI.map((i) => [H.g(x[i], 3)].concat(Array.from({ length: dd - 1 }, (_, k) => H.g(x[i] ** (k + 2), 3)), deg > 4 ? [H.g(x[i] ** deg, 3)] : []));
      const tb = DT().table({ prefix: 'tb', x: 480, y: 70, cols, rows, rowH: 26, size: 12, gap: 2 });
      F.set('cols', deg);
      F.add({ items: [title, ...ax.items, curve(one, 'fit', 'bad'), ...dots, ...tb.items, H.text('tbn', 480, 70 + tb.h + 26, 'one column became ' + deg, 'act', 14)], say: `Polynomial features turn the one column x into ${deg}: ${Array.from({ length: Math.min(deg, 4) }, (_, k) => pname(k + 1)).join(', ')}${deg > 4 ? ', …, ' + pname(deg) : ''}. For x = ${H.g(x[showI[0]], 3)} the row becomes ${Array.from({ length: Math.min(deg, 3) }, (_, k) => H.g(x[showI[0]] ** (k + 1), 3)).join(', ')}${deg > 3 ? ' and so on' : ''}. The model stays linear in these columns.`, meta: { phase: 'expand', deg, x0: x[showI[0]], powers: [1, 2, 3].map((k) => x[showI[0]] ** k) }, view });
      const coefs = cur.m.coef.slice(0, 4).map((w, k) => H.g(w, 3) + '·' + pname(k + 1));
      F.add({ items: [title, ...ax.items, curve(cur, 'fit', 'good'), ...dots, ...side([['degree ' + deg + ': ' + deg + ' columns', 'good'], ['training MSE ' + H.g(cur.tr, 3), 'strong'], ['test MSE ' + H.g(cur.te, 3), 'strong'], ['largest |weight| ' + H.g(Math.max(...cur.m.coef.map(Math.abs)), 3), 'tick']])], say: `Least squares on the ${deg} columns gives y ≈ ${(H.g(cur.m.intercept, 3) + ' + ' + coefs.join(' + ')).replace(/\+ −/g, '− ')}${deg > 4 ? ' + …' : ''}. Training MSE ${H.g(cur.tr, 3)}, test MSE ${H.g(cur.te, 3)}.` + (deg >= 9 ? ' The curve swings between the points to pass close to each training point, and misses the test points.' : ''), meta: { phase: 'fit', tr: cur.tr, te: cur.te, coef: cur.m.coef, b: cur.m.intercept }, view });
      const best = fits.reduce((b, f, k) => (f.te < fits[b].te ? k : b), 0);
      const top = Math.max(...fits.map((f) => Math.max(f.tr, f.te)));
      const eax = C().axes({ prefix: 'ex', x: 520, y: 70, w: 240, h: 200, xr: [0.5, 12.5], yr: [0, Math.min(top, one.te * 2.2) * 1.05], xticks: [1, 3, 6, 9, 12], yticks: C().ticks(0, Math.min(top, one.te * 2.2), 3), xname: 'degree', yname: 'MSE', tickSize: 12 });
      F.add({ items: [title, ...ax.items, curve(cur, 'fit', 'good'), ...dots, ...eax.items, C().line(eax, 'etr', fits.map((f, k) => [k + 1, f.tr]), 'c0'), C().line(eax, 'ete', fits.map((f, k) => [k + 1, f.te]), 'c1'), C().dot(eax, 'edot', deg, Math.min(cur.te, eax.o.yr[1]), 'c1'), ...C().legend('lg', 520, 330, [{ text: 'training', cls: 'c0' }, { text: 'test', cls: 'c1' }]), H.text('bst', 520, 360, 'lowest test MSE at degree ' + (best + 1), 'act', 14)], say: `Every degree from 1 to 12: the training error ${fits.every((f, k) => !k || f.tr <= fits[k - 1].tr + 1e-9) ? 'only falls' : 'falls overall'} (${H.g(fits[0].tr, 3)} at degree 1, ${H.g(fits[11].tr, 3)} at degree 12), but the test error is lowest at degree ${best + 1} (${H.g(fits[best].te, 3)}) and is ${H.g(fits[11].te, 3)} at degree 12.`, meta: { phase: 'degrees', trs: fits.map((f) => f.tr), tes: fits.map((f) => f.te), best: best + 1 }, view });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Polynomial features', text: ['x → (x, x², …, x^d)', 'y ≈ b + w₁x + w₂x² + … + w_d x^d', 'linear in the w, so least squares still works', 'two inputs, degree 2: x₁, x₂, x₁², x₁x₂, x₂²', 'high degree: low training error, high test error'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'fit', pick: 'option', ask: 'Across degrees 1 to 12, where will the test error be lowest?', options: ['at a low or middle degree', 'at degree 12', 'at degree 1'], answer: (fr) => { const b = fr.find((f) => f.meta.phase === 'degrees').meta.best; return b === 12 ? 'at degree 12' : b === 1 ? 'at degree 1' : 'at a low or middle degree'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'degrees').say },
    ],
    quiz: [
      { q: 'With degree-3 polynomial features, one input x becomes how many columns (without the constant)?', options: ['3', '1', '4', '9'], answer: 0, why: 'x, x² and x³.' },
      { q: 'Is a model y = b + w₁x + w₂x² linear?', options: ['linear in its weights, so least squares fits it directly', 'no, it cannot be fitted', 'only if w₂ = 0', 'only for positive x'], answer: 0, why: 'The curve bends in x, but the prediction is a weighted sum of fixed columns.' },
      { q: 'As the degree grows, the training error…', options: ['never increases', 'always increases', 'stays the same', 'becomes the test error'], answer: 0, why: 'A higher degree can do anything a lower one can (set the extra weights to 0), so it fits the training rows at least as well.' },
      { q: 'Two inputs with all degree-2 terms give which columns?', options: ['x₁, x₂, x₁², x₁x₂, x₂²', 'x₁², x₂²', 'x₁x₂ only', 'x₁, x₂'], answer: 0, why: 'Every product of at most two inputs.' },
      { q: 'x ranges up to 3 and the degree is 12. Why scale x first in practice?', options: ['x¹² reaches about 530 000, which makes the numbers badly conditioned', 'it is required by the formula', 'to remove noise', 'it changes the degree'], answer: 0, why: 'Columns of wildly different size make least squares and gradient descent unstable.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Interactions
  // ---------------------------------------------------------------------------------------------------------------
  const SETS = {
    xor: { label: 'XOR: opposite corners', make: () => S.Data.xor({ n: 200, noise: 0.04, seed: 3 }), r: [-1.15, 1.15] },
    circles: { label: 'Circles: a ring round a disc', make: () => S.Data.circles({ n: 200, noise: 0.08, seed: 3 }), r: [-1.35, 1.35] },
  };
  const FEATS = {
    plain: { label: 'x1, x2', cols: (p) => [p[0], p[1]], names: ['x1', 'x2'] },
    prod: { label: 'x1, x2, x1·x2', cols: (p) => [p[0], p[1], p[0] * p[1]], names: ['x1', 'x2', 'x1·x2'] },
    sq: { label: 'x1, x2, x1², x2²', cols: (p) => [p[0], p[1], p[0] * p[0], p[1] * p[1]], names: ['x1', 'x2', 'x1²', 'x2²'] },
    all2: { label: 'all of degree 2', cols: (p) => [p[0], p[1], p[0] * p[0], p[0] * p[1], p[1] * p[1]], names: ['x1', 'x2', 'x1²', 'x1·x2', 'x2²'] },
  };
  S.topic({
    id: 'data-interactions', family: 'p01-engineer', order: 20, title: 'Interaction features',
    blurb: 'Logistic regression draws a straight boundary through the columns it is given. On the XOR pattern no straight line in (x1, x2) works: the class depends on whether x1 and x2 have the same sign, which is the sign of their product. Add the column x1·x2 and the same model separates the classes; on circles the squares x1² and x2² do it. The boundary is straight in the new columns and curved back in the original plane.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'xor', options: Object.entries(SETS).map(([value, s]) => ({ value, label: s.label })) },
      { id: 'feats', kind: 'choice', label: 'Extra columns', def: 'prod', options: Object.entries(FEATS).filter(([k]) => k !== 'plain').map(([value, f]) => ({ value, label: f.label })) },
    ],
    presets: [
      { label: 'XOR with x1·x2', values: { data: 'xor', feats: 'prod' } },
      { label: 'XOR with squares (wrong tool)', values: { data: 'xor', feats: 'sq' } },
      { label: 'Circles with squares', values: { data: 'circles', feats: 'sq' } },
      { label: 'Circles, all degree 2', values: { data: 'circles', feats: 'all2' } },
    ],
    counters: [{ id: 'cols', label: 'columns' }],
    run(inp) {
      const D = SETS[inp.data].make(), X = D.X, y = D.y, ml = ML(), n = X.length;
      const fitWith = (key) => { const f = FEATS[key], m = ml.LogisticRegression({ C: 100 }).fit(X.map(f.cols), y); const prob = (u, v) => m.predictProba([f.cols([u, v])])[0][1]; return { f, m, prob, acc: ml.accuracy(y, m.predict(X.map(f.cols))) }; };
      const base = fitWith('plain'), ext = fitWith(inp.feats);
      const [r0, r1] = SETS[inp.data].r;
      const ax = C().axes({ prefix: 'ax', x: 50, y: 50, w: 340, h: 340, xr: [r0, r1], yr: [r0, r1], xname: 'x1', yname: 'x2', tickSize: 12 });
      const F = new S.Frames(['cols']), view = H.box(-14, -8, 790, 470);
      const pts = (fit) => X.map((p, i) => ({ key: 'p' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 3.4, cls: 'sp c' + y[i] + (fit && ((fit.prob(p[0], p[1]) >= 0.5 ? 1 : 0) !== y[i]) ? ' wrong' : ''), audit: 'overlap-ok' }));
      const field = (fit) => [SC().field(ax, 'fld', fit.prob, { n: 70 }), SC().level(ax, 'lvl', SC().grid(ax, fit.prob, 60), 0.5, '')];
      const title = H.title('tt', 0, 6, SETS[inp.data].label + ', ' + n + ' points, logistic regression');
      const wbars = (fit, yy) => [H.title('wt', 430, yy - 18, 'weights')].concat(DT().hbars({ prefix: 'wb', x: 430, y: yy, w: 340, rowH: 26, vals: fit.m.coef[0], names: fit.f.names, max: Math.max(1, ...fit.m.coef[0].map(Math.abs)), fmt: (v) => H.g(v, 3), nameW: 64, valW: 70, cls: (k) => (fit.m.coef[0][k] >= 0 ? 'pos' : 'neg') }));
      const side = (lines) => lines.map((l, k) => H.text('sd' + k, 430, 70 + k * 25, l[0], l[1] || 'strong', 14));
      F.set('cols', 2);
      F.add({ items: [title, ...ax.items, ...pts(null)], say: `${n} points in two classes. ` + (inp.data === 'xor' ? 'Blue sits where x1 and x2 have the same sign, orange where they differ.' : 'Orange fills a disc and blue a ring around it.') + ' Can a straight line separate them?', meta: { phase: 'data' }, view });
      F.add({ items: [title, ...ax.items, ...field(base), ...pts(base), ...side([['columns: x1, x2', 'strong'], ['training accuracy ' + H.pct(base.acc), 'bad']]), ...wbars(base, 170)], say: `Logistic regression on x1 and x2 alone draws one straight boundary, and no straight boundary works here: it gets ${H.pct(base.acc)} of the points right` + (base.acc < 0.7 ? ', not far above the 50% of guessing.' : '.') + ` Both weights stay small (${H.list(base.m.coef[0].map((w) => H.g(w, 3)))}).`, meta: { phase: 'plain', acc: base.acc, coef: base.m.coef[0] }, view });
      const nm = ext.f.names.slice(2);
      F.set('cols', ext.f.names.length);
      const p0 = X[0], vals = ext.f.cols(p0).slice(2);
      F.add({ items: [title, ...ax.items, ...pts(null), ...side([['new columns: ' + nm.join(', '), 'act'], ['point 0: x1 = ' + H.g(p0[0], 3) + ', x2 = ' + H.g(p0[1], 3), 'strong'], [nm.map((s, k) => s + ' = ' + H.g(vals[k], 3)).join(',  '), 'act']])], say: `Add ${H.plural(nm.length, 'column')} made from the old ones: ${nm.join(', ')}. For point 0 (x1 = ${H.g(p0[0], 3)}, x2 = ${H.g(p0[1], 3)}) ${nm.length === 1 ? 'it is' : 'they are'} ${nm.map((s, k) => s + ' = ' + H.g(vals[k], 3)).join(' and ')}.` + (inp.feats === 'prod' && inp.data === 'xor' ? ' Its sign is exactly the class rule.' : ''), meta: { phase: 'new', vals }, view });
      const big = ext.m.coef[0].reduce((b, w, k) => (Math.abs(w) > Math.abs(ext.m.coef[0][b]) ? k : b), 0);
      F.add({ items: [title, ...ax.items, ...field(ext), ...pts(ext), ...side([['columns: ' + ext.f.names.join(', '), 'strong'], ['training accuracy ' + H.pct(ext.acc), ext.acc > 0.9 ? 'good' : 'bad']]), ...wbars(ext, 170)], say: `The same model on ${ext.f.names.length} columns: the boundary is straight in those columns and curved back in the plane. It gets ${H.pct(ext.acc)} right, against ${H.pct(base.acc)} before. The largest weight is on ${ext.f.names[big]} (${H.g(ext.m.coef[0][big], 3)}).` + (inp.data === 'xor' && inp.feats === 'sq' ? ' Squares cannot tell same-sign from opposite-sign corners, so they barely help here.' : ''), meta: { phase: 'ext', acc: ext.acc, coef: ext.m.coef[0], big }, view });
      const ctx = { n, base: base.acc, p0, chance: 0.5 };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Interactions', text: ['p = σ(b + w₁x₁ + w₂x₂ + w₃x₁x₂)', 'w₃x₁x₂: the effect of x₁ depends on x₂', 'x₁² + x₂² = r²: distance from the centre', 'a straight boundary in new columns', '= a curved one in the old'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'new', pick: 'option', ask: 'With these columns, will training accuracy rise above 90%?', options: ['yes', 'no'], answer: (fr) => (fr.find((f) => f.meta.phase === 'ext').meta.acc > 0.9 ? 'yes' : 'no'), explain: (fr) => fr.find((f) => f.meta.phase === 'ext').say },
    ],
    quiz: [
      { q: 'On XOR data, which single extra column lets a linear classifier succeed?', options: ['x1·x2', 'x1²', 'x1 + x2', 'x2²'], answer: 0, why: 'The class is the sign of the product.' },
      { q: 'Which extra columns separate a disc from a ring around it?', options: ['x1² and x2²', 'x1·x2', 'x1 − x2', 'none can'], answer: 0, why: 'Their sum is the squared distance from the centre, and a threshold on it is a circle.' },
      { q: 'A weight on x1·x2 means…', options: ['the effect of x1 depends on the value of x2', 'x1 and x2 are equal', 'x1 is ignored', 'the model is no longer fitted by the same method'], answer: 0, why: 'The slope in x1 is w₁ + w₃x₂.' },
      { q: 'The boundary of logistic regression on (x1, x2, x1², x2²) in the (x1, x2) plane is…', options: ['a conic: a circle, ellipse or other curve', 'always a straight line', 'always a circle', 'a set of points'], answer: 0, why: 'b + w·features = 0 is a quadratic equation in x1 and x2.' },
      { q: 'With 10 inputs, all pairwise products add how many columns?', options: ['45', '10', '100', '20'], answer: 0, why: '10 × 9 / 2 = 45 distinct pairs.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
