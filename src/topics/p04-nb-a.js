// Part 4, Gaussian naive Bayes: one bell per class and feature (drawn along the axes), axis-aligned class ellipses
// (independence within a class means no tilt), the posterior map, and a probe point scored term by term.
(function (S) {
  'use strict';
  const H = S.PL, P = S.PL4, ML = () => S.ML;
  const n$ = H.n, lp = P.lp;
  const SETS = [{ value: 'flowers', label: 'meadow wildflowers' }, { value: 'iris', label: 'iris (petals)' }, { value: 'blobs', label: 'three blobs' }, { value: 'tilted', label: 'two tilted clouds' }, { value: 'moons', label: 'two moons' }];
  const dataOf = (inp) => {
    if (inp.set !== 'tilted') return P.data(inp.set, inp.n, inp.seed, 0.5);
    const g = H.gauss2(inp.n, inp.seed, [3.6, 3.4], [6.2, 6.6], [1.3, 1.3, 0.85], [1.3, 1.3, 0.85]);
    return { X: g.X, y: g.y, names: ['blue', 'orange'], xname: 'x₁', yname: 'x₂' };
  };

  S.topic({
    id: 'nb-gauss', family: 'p04-nb', order: 10, title: 'Gaussian naive Bayes',
    blurb: 'For each class, fit a normal bell to every feature on its own: a mean and a variance per feature, and the class’s share of the data as its prior. To classify a point, add the log prior and the log height of each bell at the point’s features, and compare the classes. Treating the features as independent within a class makes every class ellipse axis-aligned: it cannot tilt, even when the data do.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'flowers', options: SETS },
      { id: 'n', kind: 'number', label: 'points', def: 90, min: 6, max: 300 },
      { id: 'seed', kind: 'number', label: 'seed', def: 2, min: 1, max: 9999 },
      { id: 'qx', kind: 'slider', label: 'probe across', def: 0.55, min: 0.02, max: 0.98, step: 0.01 },
      { id: 'qy', kind: 'slider', label: 'probe up', def: 0.42, min: 0.02, max: 0.98, step: 0.01 },
    ],
    presets: [
      { label: 'Wildflowers', values: { set: 'flowers', n: 90, seed: 2, qx: 0.55, qy: 0.42 } },
      { label: 'Iris petals', values: { set: 'iris', n: 150, seed: 1, qx: 0.62, qy: 0.6 } },
      { label: 'Three blobs', values: { set: 'blobs', n: 60, seed: 3, qx: 0.5, qy: 0.5 } },
      { label: 'Tilted clouds: the assumption fails', values: { set: 'tilted', n: 120, seed: 5, qx: 0.48, qy: 0.52 } },
      { label: 'Two moons', values: { set: 'moons', n: 100, seed: 2, qx: 0.5, qy: 0.5 } },
    ],
    check: (inp) => { const d = dataOf(inp), K = Math.max(...d.y) + 1; for (let k = 0; k < K; k++) if (d.y.filter((v) => v === k).length < 2) return 'points: each class needs at least 2 points'; return null; },
    counters: [{ id: 'cls', label: 'classes fitted' }],
    run(inp) {
      const d = dataOf(inp), X = d.X, y = d.y;
      const m = ML().GaussianNB().fit(X, y), tr = m.trace.filter((s) => s.kind === 'class'), K = m.classes.length;
      const side = 360, B = { x: 44, y: 106, side };
      const xr = H.nice(X.map((r) => r[0]), 0.08), yr = H.nice(X.map((r) => r[1]), 0.08), rng = { xr, yr };
      const ax = H.axes({ prefix: 'gn', x: B.x, y: B.y, w: side, h: side, xr, yr, xname: d.xname, grid: true });
      const q = [rng.xr[0] + inp.qx * (rng.xr[1] - rng.xr[0]), rng.yr[0] + inp.qy * (rng.yr[1] - rng.yr[0])].map((v) => S.Data.round(v, 2));
      const R = 524, F = new S.Frames(['cls']);
      const pdf = (v, mu, s2) => Math.exp(-0.5 * (v - mu) ** 2 / s2) / Math.sqrt(2 * Math.PI * s2);
      // the bells: along the top for x₁, along the right for x₂, each scaled to its own strip
      const peaks = [0, 1].map((j) => Math.max(...m.var.map((v) => 1 / Math.sqrt(2 * Math.PI * v[j]))));
      const bells = (upto) => {
        const it = [H.text('gn:yn', B.x - 30, B.y - 84, '↑ ' + d.yname, 'axname', 13), H.seg('gn:bx', B.x, B.y - 8, B.x + side, B.y - 8, 'pl-seg thin', 'ignore'), H.seg('gn:by', B.x + side + 8, B.y, B.x + side + 8, B.y + side, 'pl-seg thin', 'ignore')];
        for (let k = 0; k <= upto && k < K; k++) {
          const top = [], rt = [];
          for (let i = 0; i <= 80; i++) {
            const u = rng.xr[0] + (i / 80) * (rng.xr[1] - rng.xr[0]), v = rng.yr[0] + (i / 80) * (rng.yr[1] - rng.yr[0]);
            top.push([ax.X(u), B.y - 8 - 58 * (pdf(u, m.theta[k][0], m.var[k][0]) / peaks[0])]);
            rt.push([B.x + side + 8 + 58 * (pdf(v, m.theta[k][1], m.var[k][1]) / peaks[1]), ax.Y(v)]);
          }
          for (const [key, pts] of [['t', top], ['r', rt]]) it.push({ key: 'gn:bell' + key + k, kind: 'path', d: pts.map((p, i) => (i ? 'L' : 'M') + H.r2(p[0]) + ',' + H.r2(p[1])).join(' '), cls: 'series thin c' + k, pathAudit: 'edge-x', layer: 'lines' });
        }
        return it;
      };
      const ells = (upto, soft) => {
        const it = [];
        for (let k = 0; k <= upto && k < K; k++) {
          const C = [[m.var[k][0], 0], [0, m.var[k][1]]];
          it.push(H.ellipsePath(ax, 'gn:e1' + k, m.theta[k], C, 1, 'c' + k + ' fill'), H.ellipsePath(ax, 'gn:e2' + k, m.theta[k], C, 2, 'c' + k + ' soft'));
          it.push({ key: 'gn:mu' + k, kind: 'dot', x: ax.X(m.theta[k][0]), y: ax.Y(m.theta[k][1]), r: 6, cls: 'mark c' + k, audit: 'overlap-ok', layer: 'items', z: 4 });
        }
        if (soft) it.forEach((x) => { x.cls += ' dim'; });
        return it;
      };
      const pts = () => S.Kit.scatter.points(ax, 'gn:p', X, y, { r: 3.8, cls: () => 'faded' });
      const legend = H.classLegend('gn:lg', R, 40, d.names.slice(0, K), { dir: 'col' });
      const name = (k) => d.names[k];
      F.add({ items: [...ax.items, ...pts(), ...bells(-1), ...legend], say: X.length + ' points in ' + H.plural(K, 'class', 'classes') + '. Gaussian naive Bayes will summarise each class by a prior and, for each feature separately, a mean and a variance.', meta: { phase: 'data', n: X.length, K } });
      tr.forEach((st, k) => {
        F.set('cls', k + 1);
        F.add({ items: [...ax.items, ...pts(), ...bells(k), ...ells(k), ...legend, ...P.eqs('gn:eq', R, 40 + K * 22 + 40, [[name(k) + ': prior ' + n$(st.prior), 'c' + k], ['mean (' + n$(st.mean[0]) + ', ' + n$(st.mean[1]) + ')', ''], ['variance (' + n$(st.var[0]) + ', ' + n$(st.var[1]) + ')', ''], ['no covariance: the ellipse', 'muted'], ['cannot tilt', 'muted']], 14, 25)], say: 'Class ' + name(k) + ': ' + H.pct(st.prior) + ' of the points (prior ' + n$(st.prior) + '), feature means ' + n$(st.mean[0]) + ' and ' + n$(st.mean[1]) + ', variances ' + n$(st.var[0]) + ' and ' + n$(st.var[1]) + '. The bells along the edges are its two normal curves; the ellipses mark 1 and 2 standard deviations.', meta: { phase: 'class', k, prior: st.prior, mean: st.mean, var: st.var } });
      });
      const acc = ML().accuracy(y, m.predict(X));
      const field = P.regions(ax, 'gn:f', (u, v) => { const p = m.predictProba([[u, v]])[0], c = ML().argmax(p); return [c, Math.min(1, (p[c] - 1 / K) * K / (K - 1) + 0.15)]; }, { n: 64 });
      F.add({ items: [...ax.items, field, ...pts(), ...bells(K), ...ells(K, true), ...legend, ...P.eqs('gn:eq', R, 40 + K * 22 + 40, [['training accuracy ' + H.pct(acc), 'good'], ['boundaries: quadratic curves', 'muted'], ['(the variances differ)', 'muted']], 14, 25)], say: 'Comparing the classes everywhere gives this map: each place takes the class with the largest prior times bell heights. The boundaries curve because the classes have different variances. On the training points the map is right ' + H.pct(acc) + ' of the time.', meta: { phase: 'map', acc } });
      const ex = m.explain(q), joint = m.jointLogLikelihood([q])[0], post = m.predictProba([q])[0], win = ML().argmax(post);
      const fmtT = (v) => lp(v), lines = [];
      ex.forEach((e, k) => { lines.push([name(k) + ':  P = ' + n$(post[k]), k === win ? 'act' : 'c' + k]); lines.push([fmtT(e.prior) + ' ' + (e.terms[0] < 0 ? '− ' : '+ ') + fmtT(Math.abs(e.terms[0])) + ' ' + (e.terms[1] < 0 ? '− ' : '+ ') + fmtT(Math.abs(e.terms[1])) + ' = ' + fmtT(joint[k]), 'muted']); });
      const probe = { key: 'gn:q', kind: 'dot', x: ax.X(q[0]), y: ax.Y(q[1]), r: 7, cls: 'pl-probe act', audit: 'overlap-ok', layer: 'items', z: 6 };
      const guides = [H.seg('gn:gx', ax.X(q[0]), B.y - 8, ax.X(q[0]), ax.Y(q[1]), 'pl-seg dash thin', 'ignore'), H.seg('gn:gy', ax.X(q[0]), ax.Y(q[1]), B.x + side + 8, ax.Y(q[1]), 'pl-seg dash thin', 'ignore')];
      F.add({ items: [...ax.items, field, ...pts(), ...bells(K), ...ells(K, true), ...guides, probe, ...legend, H.text('gn:hd', R, 40 + K * 22 + 40, 'log prior + log bell₁ + log bell₂', 'tick', 12), ...P.eqs('gn:eq', R, 40 + K * 22 + 66, lines, 13, 23)], say: 'A probe at (' + n$(q[0]) + ', ' + n$(q[1]) + '). For each class add the log prior and the log heights of its two bells there: ' + ex.map((e, k) => name(k) + ' ' + fmtT(joint[k])).join(', ') + '. Turning these into probabilities (exponentiate and normalise) gives ' + name(win) + ' with ' + n$(post[win]) + '.', meta: { phase: 'probe', q, joint, post, prior: ex.map((e) => e.prior), terms: ex.map((e) => e.terms), names: d.names.slice(0, K), winName: name(win) } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Gaussian naive Bayes', text: ['P(c | x) ∝ P(c) · Πⱼ N(xⱼ; μ_cj, σ²_cj)', 'log score = log P(c)', '  + Σⱼ [−½ log(2πσ²_cj) − (xⱼ − μ_cj)² / (2σ²_cj)]', 'μ_cj, σ²_cj: mean and variance of feature j in class c', 'independence ⇒ axis-aligned ellipses'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'map', pick: 'option', ask: 'Which class will the probe point (the ring, next) get?', options: (fr) => fr[fr.length - 1].meta.names, answer: (fr) => fr[fr.length - 1].meta.winName, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'What does "naive" assume?', options: ['features are independent within each class', 'all classes are equally likely', 'features are normally distributed overall', 'there are two classes'], answer: 0, why: 'The class-conditional density is a product over features.' },
      { q: 'Why are the class ellipses never tilted?', options: ['there is no covariance term between features', 'the data are standardised', 'the priors are equal', 'tilted ellipses are not Gaussian'], answer: 0, why: 'A diagonal covariance matrix has axis-aligned ellipses.' },
      { q: 'Why add log probabilities instead of multiplying probabilities?', options: ['products of many small numbers underflow; sums of logs do not', 'logs are more accurate for large numbers', 'it changes which class wins', 'to make them positive'], answer: 0, why: 'The logarithm keeps the order and turns products into sums.' },
      { q: 'With equal priors and equal variances in both classes, the boundary is…', options: ['a straight line', 'a circle', 'a parabola always', 'undefined'], answer: 0, why: 'The squared terms cancel, leaving a linear rule.' },
      { q: 'How many numbers does Gaussian NB fit for 3 classes and 4 features?', options: ['3 priors + 12 means + 12 variances', '3 priors + 3 covariance matrices', '12', '4'], answer: 0, why: 'One mean and one variance per class and feature.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
