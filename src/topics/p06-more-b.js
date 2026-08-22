// Part 6, SVR and the one-class SVM. SVR fits a curve with a tube of half-width ε round it: points inside the tube
// cost nothing, points outside pay by how far they stick out, and only those on or outside the tube become
// support vectors. The one-class SVM draws a border round the bulk of the data; ν caps the share of points left
// outside and is at most the share of support vectors.
(function (S) {
  'use strict';
  const P = S.P06, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const EPS = [0.05, 0.1, 0.2, 0.35, 0.6];

  S.topic({
    id: 'svm-svr', family: 'p06-more', order: 20, title: 'SVR and its ε-tube',
    blurb: 'Support vector regression turns the margin inside out. It looks for a function f that is as flat as it can be while every point lies within ε of it, inside a tube of half-width ε; points outside pay C for every unit they stick out. A point strictly inside the tube has no say at all, so a wider tube leaves fewer support vectors and a smoother, lazier fit.',
    inputs: [
      { id: 'shape', kind: 'choice', label: 'Curve', def: 'sine', options: [{ value: 'sine', label: 'a sine' }, { value: 'poly', label: 'a cubic' }, { value: 'step', label: 'steps' }, { value: 'abs', label: 'a V' }] },
      { id: 'noise', kind: 'slider', label: 'Noise', def: 0.15, min: 0, max: 0.4, step: 0.01 },
      { id: 'gamma', kind: 'slider', label: 'γ (RBF)', def: 0.5, min: 0.05, max: 5, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'C', kind: 'slider', label: 'C', def: 10, min: 0.1, max: 100, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'eps', kind: 'slider', label: 'Your ε', def: 0.25, min: 0.01, max: 1, step: 0.01 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'A noisy sine', values: { shape: 'sine', noise: 0.15, gamma: 0.5, C: 10, eps: 0.25, seed: 1 } },
      { label: 'A clean cubic', values: { shape: 'poly', noise: 0.05, gamma: 0.3, C: 30, eps: 0.1, seed: 2 } },
      { label: 'Steps, narrow γ', values: { shape: 'step', noise: 0.1, gamma: 3, C: 10, eps: 0.15, seed: 3 } },
      { label: 'A V, small C', values: { shape: 'abs', noise: 0.2, gamma: 0.5, C: 0.5, eps: 0.3, seed: 4 } },
    ],
    counters: [{ id: 'sv', label: 'support vectors' }],
    run(inp) {
      const d = S.Data.curve({ n: 40, shape: inp.shape, noise: inp.noise, seed: inp.seed }), X = d.X, yv = d.y, n = X.length;
      const g = +H().sg(inp.gamma, 2), Cv = +H().sg(inp.C, 2), eu = +inp.eps.toFixed(2), list = EPS.concat(EPS.includes(eu) ? [] : [eu]);
      const fits = list.map((e) => { const m = new (ML().SVR)({ kernel: 'rbf', gamma: g, C: Cv, epsilon: e, tol: 1e-4 }).fit(X, yv); P.record('svr', X, yv, m.params, m); const p = m.predict(X); return { m, e, p, sv: m.support, out: p.filter((v, i) => Math.abs(yv[i] - v) > e + 1e-6).length, mse: ML().mse(yv, p), truth: ML().mse(X.map((x) => d.truth(x[0])), p) }; });
      const F = H().frames(['sv']), q = F.q;
      const ylo = Math.min(...yv) - 0.6, yhi = Math.max(...yv) + 0.6;
      const ax = C().axes({ prefix: 'sr', x: 40, y: 24, w: 430, h: 360, xr: [-3.2, 3.2], yr: [ylo, yhi], xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: C().ticks(ylo, yhi, 5), xname: 'x', yname: 'y' });
      const xs = Array.from({ length: 121 }, (_, k) => -3.2 + (6.4 * k) / 120);
      const curveOf = (m) => m.predict(xs.map((x) => [x]));
      const tube = (fit) => {
        const c = curveOf(fit.m), up = xs.map((x, k) => [ax.X(x), ax.Y(Math.min(yhi, c[k] + fit.e))]), dn = xs.map((x, k) => [ax.X(x), ax.Y(Math.max(ylo, c[k] - fit.e))]).reverse();
        const items = [{ key: 'sr:tube', kind: 'path', d: up.concat(dn).map((p, k) => (k ? 'L' : 'M') + p[0] + ',' + p[1]).join(' ') + ' Z', cls: 'mltube', audit: 'ignore', layer: 'back', z: 1 }];
        items.push(C().line(ax, 'sr:f', xs.map((x, k) => [x, c[k]]), 'act'), C().line(ax, 'sr:u', xs.map((x, k) => [x, c[k] + fit.e]), 'faint'), C().line(ax, 'sr:d', xs.map((x, k) => [x, c[k] - fit.e]), 'faint'));
        return items;
      };
      const pts = (fit) => {
        const items = X.map((x, i) => ({ key: 'sr:p' + i, kind: 'dot', x: ax.X(x[0]), y: ax.Y(yv[i]), r: 4.4, cls: 'sp c0', audit: 'overlap-ok', layer: 'items' }));
        if (fit) for (const i of fit.sv) items.push(S.Kit.scatter.ring(ax, 'sr:r' + i, X[i][0], yv[i], 'act', 8));
        return items;
      };
      // the number of support vectors against ε
      const sx = C().axes({ prefix: 'sv', x: 560, y: 60, w: 220, h: 150, xr: [0, 0.65], yr: [0, n], xticks: [0, 0.2, 0.4, 0.6], yticks: C().ticks(0, n, 4), xname: 'ε', yname: 'support vectors' });
      const side = (k, cur) => {
        const items = [...sx.items], seen = fits.slice(0, k + 1).filter((f) => EPS.includes(f.e));
        if (seen.length > 1) items.push(C().line(sx, 'sv:l', seen.map((f) => [f.e, f.sv.length]), 'act'));
        seen.forEach((f, j) => items.push(C().dot(sx, 'sv:d' + j, f.e, f.sv.length, '', 3.5)));
        if (cur && cur.e <= 0.65) items.push(C().dot(sx, 'sv:cur', cur.e, cur.sv.length, '', 6.5));
        return items;
      };
      const truth = C().line(ax, 'sr:truth', xs.map((x) => [x, d.truth(x)]), 'faint');
      F.push([...ax.items, truth, ...pts(null), ...side(-1)], `${q.n(n)} noisy points along a curve (the faint line). SVR with the RBF kernel, γ = ${q.sg(g, 2)} and C = ${q.sg(Cv, 2)}, will fit them with tubes of different half-widths ε.`, { phase: 'data', n, gamma: g, C: Cv });
      EPS.forEach((e, k) => {
        const fit = fits[k];
        F.set('sv', fit.sv.length);
        let say = `ε = ${q.sg(e, 2)}: ${q.n(fit.sv.length)} of the ${q.n(n)} points are support vectors (ringed), all on or outside the tube; ${q.n(fit.out)} lie strictly outside and pay for it. Mean squared error against the true curve ${q.sg(fit.truth, 2)}.`;
        if (k === 0) say += ' A thin tube: almost every point sticks out, so almost every point is a support vector.';
        if (k === EPS.length - 1) say += ' A fat tube swallows most of the noise, and the few points left decide the curve on their own.';
        F.push([...ax.items, truth, ...tube(fit), ...pts(fit), ...side(k, fit)], say, { phase: 'eps', eps: e, nsv: fit.sv.length, out: fit.out, truth: fit.truth, mse: fit.mse, n, k });
      });
      const fu = fits[list.indexOf(eu)];
      F.set('sv', fu.sv.length);
      F.push([...ax.items, truth, ...tube(fu), ...pts(fu), ...side(EPS.length - 1, fu)], `Your ε = ${q.sg(eu, 2)}: ${q.n(fu.sv.length)} support vectors, ${q.n(fu.out)} points outside the tube, error against the true curve ${q.sg(fu.truth, 2)}. The loss is max(0, |y − f(x)| − ε): flat inside the tube, a straight slope outside, like a hinge on each side.`, { phase: 'mine', eps: eu, nsv: fu.sv.length, out: fu.out, truth: fu.truth });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 430 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'ε-insensitive regression', text: ['f(x) = Σ (αᵢ − αᵢ*) K(xᵢ, x) + b', 'loss: max(0, |y − f(x)| − ε)', 'minimise ½‖w‖² + C·Σ (ξᵢ + ξᵢ*)', 'inside the tube: α = α* = 0, no say', 'support vectors: on or outside the tube'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'eps' && f.meta.k === 1, pick: 'option', ask: 'Double ε. Will there be more support vectors or fewer?', options: ['fewer or the same', 'more'], answer: (fr, i) => (fr[i + 1].meta.nsv <= fr[i].meta.nsv ? 'fewer or the same' : 'more'), explain: (fr, i) => fr[i + 1].say },
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'What does a point strictly inside the tube add to the loss?', options: ['nothing', 'its distance to the curve', 'ε', 'C'], answer: 'nothing', explain: 'The ε-insensitive loss is zero within ε of the curve.' },
    ],
    quiz: [
      { q: 'With ε = 0.2, what does a point 0.5 above the curve lose?', options: ['0.3', '0.5', '0.7', '0'], answer: 0, why: 'max(0, 0.5 − 0.2) = 0.3.' },
      { q: 'Which points are the support vectors of an SVR?', options: ['those on or outside the tube', 'those inside the tube', 'every point', 'those nearest the curve'], answer: 0, why: 'Inside the tube both multipliers are zero.' },
      { q: 'What happens to the fit as ε grows very large?', options: ['the tube holds every point and f flattens towards a constant', 'it fits every point exactly', 'the number of support vectors grows', 'C stops mattering only for small ε'], answer: 0, why: 'With every point inside, nothing pulls f away from the flattest function.' },
      { q: 'What does C trade off in SVR?', options: ['flatness of f against points sticking out of the tube', 'the width of the tube against γ', 'training time against memory', 'the number of features'], answer: 0, why: 'The objective is ½‖w‖² plus C times the total excess outside the tube.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // The one-class SVM
  // ---------------------------------------------------------------------------------------------------------------
  const NUS = [0.02, 0.05, 0.1, 0.2, 0.4];
  /** normal data and a few strays: our two blobs plus scattered points, or pump readings (standardised) */
  const ocData = (inp) => {
    if (inp.set === 'pumps') {
      // complete rows only: the first 185 normal readings and the first 15 the plant logged as faulty
      const d = S.Data.load('pumps'), ok = d.X.map((r, i) => i).filter((i) => Number.isFinite(d.X[i][0]) && Number.isFinite(d.X[i][1]));
      const pick = ok.filter((i) => d.y[i] === 0).slice(0, 185).concat(ok.filter((i) => d.y[i] === 1).slice(0, 15)).sort((a, b) => a - b);
      const lab = pick.map((i) => d.y[i]), Xr = pick.map((i) => [d.X[i][0], d.X[i][1]]);
      const m = ML().colMeans(Xr), sd = ML().colVars(Xr).map(Math.sqrt);
      return { X: Xr.map((r) => r.map((v, j) => +((v - m[j]) / sd[j]).toFixed(4))), truth: lab, names: ['temperature (standardised)', 'vibration (standardised)'] };
    }
    const b = S.Data.blobs({ n: 90, centers: [[-1, -0.3], [1.1, 0.7]], std: 0.45, seed: inp.seed }), r = S.Data.rng(inp.seed + 500), X = b.X.slice(), truth = X.map(() => 0);
    // strays on a wide ring round the groups, well away from both
    for (let k = 0; k < 10; k++) { const a = r.range(0, 2 * Math.PI), rad = r.range(2.6, 3.4); X.push([+(0.05 + rad * Math.cos(a)).toFixed(4), +(0.2 + rad * Math.sin(a)).toFixed(4)]); truth.push(1); }
    return { X, truth, names: ['x₁', 'x₂'] };
  };
  S.topic({
    id: 'svm-one-class', family: 'p06-more', order: 30, title: 'The one-class SVM: a border round normal data',
    blurb: 'With no labels at all, the one-class SVM maps the points with a kernel and pushes a boundary as tight round them as it can, leaving out a share of points as strays. ν (nu) sets that share: at most a fraction ν of the training points fall outside, and at least a fraction ν are support vectors. New points outside the border are flagged as anomalies.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'blobs', options: [{ value: 'blobs', label: 'two groups and some strays' }, { value: 'pumps', label: 'pump readings (temperature, vibration)' }] },
      { id: 'gamma', kind: 'slider', label: 'γ (RBF)', def: 0.5, min: 0.05, max: 5, log: true, fmt: (v) => H().sg(v, 2) },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Two groups, γ = 0.5', values: { set: 'blobs', gamma: 0.5, seed: 1 } },
      { label: 'Two groups, γ = 3', values: { set: 'blobs', gamma: 3, seed: 2 } },
      { label: 'Pump sensor readings', values: { set: 'pumps', gamma: 0.5, seed: 1 } },
    ],
    counters: [{ id: 'out', label: 'outside' }, { id: 'sv', label: 'support vectors' }],
    run(inp) {
      const d = ocData(inp), X = d.X, n = X.length, g = +H().sg(inp.gamma, 2), F = H().frames(['out', 'sv']), q = F.q;
      // strictly outside: beyond the border by more than the solver's rounding (points on it are free support vectors)
      const OUT = -1e-4;
      const fits = NUS.map((nu) => { const m = new (ML().OneClassSVM)({ kernel: 'rbf', gamma: g, nu, tol: 1e-6 }).fit(X); P.record('ocsvm', X, null, m.params, m); const dv = m.decisionFunction(X); return { m, nu, dv, out: dv.filter((v) => v < OUT).length, nsv: m.support.length, caught: dv.filter((v, i) => v < OUT && d.truth[i] === 1).length }; });
      const ax = P.axes('oc', X, { x: 40, y: 24, w: 400, h: 400 }, { pad: 0.4, xname: d.names[0].startsWith('x') ? 'x₁' : 'temperature', yname: d.names[1].startsWith('x') ? 'x₂' : 'vibration' });
      const SC = S.Kit.scatter, strays = d.truth.filter((v) => v === 1).length;
      const draw = (fit) => {
        const items = [...ax.items];
        if (fit) {
          const f = (u, v) => fit.m.decisionFunction([[u, v]])[0], gr = SC.grid(ax, f, 70);
          items.push(SC.field(ax, 'oc:f', (u, v) => (f(u, v) >= 0 ? 0.25 : 0.75), { n: 70, alpha: 0.35 }), SC.level(ax, 'oc:b', gr, 0, ''));
        }
        X.forEach((x, i) => items.push({ key: 'oc:p' + i, kind: 'dot', x: ax.X(x[0]), y: ax.Y(x[1]), r: 4.2, cls: 'sp ' + (fit && fit.dv[i] < OUT ? 'c1' : 'c0'), audit: 'overlap-ok', layer: 'items' }));
        if (fit) for (const i of fit.m.support) items.push(SC.ring(ax, 'oc:r' + i, X[i][0], X[i][1], fit.dv[i] < OUT ? '' : 'act', 8));
        return items;
      };
      const tbl = (k) => {
        const items = [H().head('oc:t', 510, 50, 'ν against what it bounds')];
        const cols = [510, 570, 660], rows = [['ν', 'outside', 'SVs']].concat(fits.slice(0, k + 1).map((f) => [H().fx(f.nu, 2), H().pct(f.out / n), H().pct(f.nsv / n)]));
        rows.forEach((r, j) => r.forEach((t, c) => items.push(H().text('oc:c' + j + '_' + c, cols[c], 84 + j * 24, t, j ? 'legend' : 'axname', 13))));
        return items;
      };
      F.push([...draw(null), ...tbl(-1)], `${q.n(n)} points with no labels at all${inp.set === 'pumps' ? '' : `; ${q.n(strays)} of them are strays scattered at random`}. The one-class SVM with the RBF kernel, γ = ${q.sg(g, 2)}, will draw a border round the bulk of them for several values of ν.`, { phase: 'data', n, strays, gamma: g });
      NUS.forEach((nu, k) => {
        const f = fits[k];
        F.set('out', f.out).set('sv', f.nsv);
        F.push([...draw(f), ...tbl(k)], `ν = ${q.sg(nu, 2)}: ${q.n(f.out)} points, ${q.pct(f.out / n)}, fall outside the border (orange), and ${q.n(f.nsv)}, ${q.pct(f.nsv / n)}, are support vectors (ringed: those on the border in colour): outside share ≤ ν ≤ support-vector share, as the theory promises.${inp.set === 'pumps' ? ` Of the flagged readings, ${q.n(f.caught)} are pumps the plant logged as faulty.` : ` ${q.n(f.caught)} of the ${q.n(strays)} strays are flagged.`}`, { phase: 'nu', nu, out: f.out, nsv: f.nsv, outShare: f.out / n, svShare: f.nsv / n, caught: f.caught, strays, n, k });
      });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 450, h: 470 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The one-class SVM', text: ['f(x) = Σ αᵢ K(xᵢ, x) − ρ; inside where f ≥ 0', '0 ≤ αᵢ ≤ 1/(νn), Σ αᵢ = 1', 'share outside ≤ ν ≤ share of support vectors', 'small ν: a loose border; large ν: a tight one'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'nu' && f.meta.k === 2, pick: 'option', ask: (fr, i) => 'At ν = ' + H().sg(fr[i + 1].meta.nu, 2) + ', at most what share of the points can fall outside the border?', options: (fr, i) => [H().pct(fr[i + 1].meta.nu), H().pct(fr[i + 1].meta.nu / 2), H().pct(Math.min(1, fr[i + 1].meta.nu * 2)), '100%'], answer: (fr, i) => H().pct(fr[i + 1].meta.nu), explain: (fr, i) => fr[i + 1].say },
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'Raising ν makes the border…', options: ['tighter, leaving more points outside', 'looser, leaving fewer outside', 'unchanged'], answer: 'tighter, leaving more points outside', explain: 'ν is a bound on the share left outside: allow more and the border can shrink.' },
    ],
    quiz: [
      { q: 'With ν = 0.1 and 200 training points, at most how many fall outside the border?', options: ['20', '10', '2', '100'], answer: 0, why: 'At most a fraction ν: 0.1·200 = 20.' },
      { q: 'And at least how many are support vectors?', options: ['20', '0', '2', '200'], answer: 0, why: 'At least a fraction ν of the points are support vectors.' },
      { q: 'What does the one-class SVM need in its training data?', options: ['no labels: mostly normal points', 'labelled anomalies', 'two balanced classes', 'a target value per point'], answer: 0, why: 'It learns the region where the data live.' },
      { q: 'What does a large γ do to the border?', options: ['wraps it tightly round small groups of points', 'makes it a straight line', 'removes every support vector', 'nothing'], answer: 0, why: 'Narrow kernel bumps let the border follow each cluster closely, and can overfit.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
