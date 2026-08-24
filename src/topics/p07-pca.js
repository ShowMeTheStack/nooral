// Part 7, principal components: in two dimensions, the direction of greatest spread found among all directions,
// the eigenvectors of the covariance drawn from the mean, and the points projected onto the first; in many
// dimensions, the variance each component explains and what the first ones are made of; and digit images rebuilt
// from more and more components.
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const memo = new Map(), once = (k, f) => { if (!memo.has(k)) memo.set(k, f()); return memo.get(k); };
  const SETS2 = {
    tilted: { label: 'one tilted cloud', make: () => { const d = S.Data.blobs({ n: 60, centers: [[0, 0]], std: 1, seed: 4 }); return { X: d.X.map((x) => [+(1.5 * x[0] + 2).toFixed(4), +(0.8 * x[0] + 0.75 * x[1] + 1).toFixed(4)]), names: ['x₁', 'x₂'] }; } },
    iris: { label: 'iris: petal length and width (cm)', make: () => P.data('iris') },
    penguins: { label: 'penguins: bill length and depth (standardised)', make: () => P.data('penguins') },
    stretched: { label: 'three stretched groups', make: () => P.data('stretched') },
  };
  P.SETS2 = SETS2;

  S.topic({
    id: 'pca-project', family: 'p07-pca', order: 10, title: 'PCA: the direction of greatest spread',
    blurb: 'Centre the data on its mean. Now look along every direction and measure how spread out the points’ shadows are on it. The direction where the shadows spread most is the first principal component; the second is the most spread direction at right angles to it. They are the eigenvectors of the covariance matrix, and the spreads are its eigenvalues. Projecting onto the first component keeps as much of the variance as one number can.',
    inputs: [{ id: 'set', kind: 'choice', label: 'Data', def: 'tilted', options: Object.keys(SETS2).map((k) => ({ value: k, label: SETS2[k].label })) }],
    presets: [
      { label: 'A tilted cloud', values: { set: 'tilted' } },
      { label: 'Iris petals', values: { set: 'iris' } },
      { label: 'Penguin bills: weakly related', values: { set: 'penguins' } },
      { label: 'Three stretched groups', values: { set: 'stretched' } },
    ],
    counters: [{ id: 'dirs', label: 'directions tried' }],
    run(inp) {
      const d = SETS2[inp.set].make(), X = d.X, n = X.length, F = H().frames(['dirs']), q = F.q;
      const pca = ML().PCA({ trace: false }).fit(X), mu = pca.mean, [u1, u2] = pca.components, [l1, l2] = pca.explainedVariance, ratio = pca.explainedVarianceRatio;
      const rec = pca.inverseTransform(pca.transform(X).map((z) => [z[0], 0]));
      const ext = [-1, 1].flatMap((s) => [[mu[0] + s * 2.4 * Math.sqrt(l1) * u1[0], mu[1] + s * 2.4 * Math.sqrt(l1) * u1[1]]]);
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('pc', X, box, { extra: ext, xname: d.names[0], yname: d.names[1] });
      const at = (p) => [ax.X(p[0]), ax.Y(p[1])];
      const lineAlong = (key, u, cls, len) => { const L = len || 2.4 * Math.sqrt(l1); return { key, kind: 'path', d: `M${ax.X(mu[0] - L * u[0])},${ax.Y(mu[1] - L * u[1])} L${ax.X(mu[0] + L * u[0])},${ax.Y(mu[1] + L * u[1])}`, cls, pathAudit: 'edge-x', layer: 'lines' }; };
      const meanMark = { key: 'pc:mu', kind: 'state', x: ax.X(mu[0]), y: ax.Y(mu[1]), w: 14, h: 14, rx: 7, text: '', cls: 'centre', audit: 'overlap-ok', layer: 'items', z: 6 };
      const pts = (Y, o) => P.points(ax, 'pc:p', Y, null, Object.assign({ r: 4.4 }, o || {})).map((it) => Object.assign(it, { cls: it.cls.replace('cn', 'c0') }));
      const sideText = (rows) => H().lines('pc:t', 520, 70, rows, 13, 24);
      const C2 = ML().la.cov(X, 1);
      F.push([...ax.items, ...pts(X), meanMark, ...sideText([['Covariance matrix', 'strong'], [`[ ${H().fx(C2[0][0], 3)}   ${H().fx(C2[0][1], 3)} ]`, 'legend'], [`[ ${H().fx(C2[1][0], 3)}   ${H().fx(C2[1][1], 3)} ]`, 'legend']])],
        `${P.qpl(q, n, 'point')}, with their mean (${q(mu[0], 2)}, ${q(mu[1], 2)}) marked. Their covariance matrix has variances ${q(C2[0][0], 3)} and ${q(C2[1][1], 3)} on its diagonal and covariance ${q(C2[0][1], 3)} off it. PCA asks: along which direction do the points spread the most?`, { phase: 'data', mu, cov: C2, n });
      const th1 = Math.atan2(u1[1], u1[0]);
      [90, 55, 25].forEach((off) => {
        F.count('dirs');
        const th = th1 + (off * Math.PI) / 180, u = [Math.cos(th), Math.sin(th)];
        const z = X.map((x) => (x[0] - mu[0]) * u[0] + (x[1] - mu[1]) * u[1]), v = z.reduce((s, t) => s + t * t, 0) / (n - 1);
        const feet = z.map((t) => [mu[0] + t * u[0], mu[1] + t * u[1]]);
        const items = [...ax.items, lineAlong('pc:dir', u, 'p7line'), ...pts(X, { opacity: () => 0.55 }), ...feet.map((p, i) => ({ key: 'pc:f' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 3, cls: 'sp c1', audit: 'overlap-ok', layer: 'items' })), meanMark];
        items.push(...sideText([['A direction ' + off + '° from the best', 'strong'], ['variance of the shadows: ' + H().fx(v, 3), 'legend']]));
        F.push(items, `Try the direction at ${q.n(off)}° from the one PCA will choose. The points’ shadows on it (orange) have variance ${q(v, 3)}.` + (off === 25 ? ' Closer to the best direction, the shadows spread further.' : ''), { phase: 'dir', off, v, th });
      });
      const arrowTo = (u, l) => [mu[0] + 2 * Math.sqrt(l) * u[0], mu[1] + 2 * Math.sqrt(l) * u[1]];
      const pcItems = [...ax.items, lineAlong('pc:dir', u1, 'p7line'), ...pts(X), meanMark, { key: 'pc:a1', kind: 'path', d: `M${at(mu).join(',')} L${at(arrowTo(u1, l1)).join(',')}`, cls: 'p7axis', pathAudit: 'edge-x', layer: 'lines', z: 8 }, { key: 'pc:a2', kind: 'path', d: `M${at(mu).join(',')} L${at(arrowTo(u2, l2)).join(',')}`, cls: 'p7axis second', pathAudit: 'edge-x', layer: 'lines', z: 8 }];
      pcItems.push(...sideText([['PC1 = (' + H().fx(u1[0], 3) + ', ' + H().fx(u1[1], 3) + ')', 'strong'], ['eigenvalue λ₁ = ' + H().fx(l1, 3) + ' (' + H().pct(ratio[0]) + ')', 'legend'], ['PC2 = (' + H().fx(u2[0], 3) + ', ' + H().fx(u2[1], 3) + ')', 'strong'], ['eigenvalue λ₂ = ' + H().fx(l2, 3) + ' (' + H().pct(ratio[1]) + ')', 'legend'], ['arrows: two standard deviations', 'axname']]));
      F.push(pcItems, `The best direction, PC1 = (${q(u1[0], 3)}, ${q(u1[1], 3)}), is the covariance matrix’s top eigenvector; the shadows on it have variance λ₁ = ${q(l1, 3)}, more than any direction tried. PC2 = (${q(u2[0], 3)}, ${q(u2[1], 3)}) is at right angles, with λ₂ = ${q(l2, 3)}. Together they hold all the variance: ${q.pct(ratio[0])} and ${q.pct(ratio[1])}.`, { phase: 'pcs', u1, u2, l1, l2, ratio });
      const items = [...ax.items, lineAlong('pc:dir', u1, 'p7line'), ...X.map((x, i) => ({ key: 'pc:r' + i, kind: 'path', d: `M${at(x).join(',')} L${at(rec[i]).join(',')}`, cls: 'p7proj', audit: 'ignore', layer: 'back', z: 5 })), ...X.map((x, i) => ({ key: 'pc:o' + i, kind: 'dot', x: ax.X(x[0]), y: ax.Y(x[1]), r: 3, cls: 'sp cn', opacity: 0.5, audit: 'overlap-ok', layer: 'items' })), ...pts(rec), meanMark];
      const err = X.reduce((s, x, i) => s + ML().sqdist(x, rec[i]), 0);
      items.push(...sideText([['Keep PC1 only', 'strong'], ['each point becomes one number', 'legend'], ['squared error: ' + H().fx(err, 3), 'legend'], ['kept: ' + H().pct(ratio[0]) + ' of the variance', 'legend']]));
      F.push(items, `Keep only PC1: each point slides along PC2 onto the line and is described by one number, its position along PC1. The total squared error is ${q(err, 3)}, which is exactly (n − 1)·λ₂ = ${q.n(n - 1)} × ${q(l2, 3)}: what PCA drops is the variance along the components it leaves out.`, { phase: 'project', err, l2, n1: n - 1 });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Principal components', text: ['centre: x̃ = x − μ', 'covariance: C = X̃ᵀX̃ / (n − 1)', 'C uⱼ = λⱼ uⱼ, λ₁ ≥ λ₂ ≥ …', 'variance along a unit u: uᵀCu, largest at u₁', 'project: zⱼ = uⱼ·x̃; rebuild: μ + Σ zⱼuⱼ', 'error dropping uⱼ: (n − 1)·λⱼ'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'dir' && fr[i + 1].meta.phase === 'pcs', pick: 'option', ask: 'Will the variance along PC1 be larger than along every direction tried so far?', options: ['yes: PC1 has the largest variance of all directions', 'no: one of these was larger', 'they will all be equal'], answer: 'yes: PC1 has the largest variance of all directions', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'What are the principal components of centred data?', options: ['the eigenvectors of its covariance matrix', 'the class means', 'the original axes', 'the points furthest from the mean'], answer: 0, why: 'The variance along a unit vector u is uᵀCu, which is largest at the top eigenvector.' },
      { q: 'Eigenvalues 6 and 2: what share of the variance does PC1 explain?', options: ['75%', '67%', '50%', '33%'], answer: 0, why: '6 / (6 + 2).' },
      { q: 'Why centre the data first?', options: ['otherwise the first direction points at the mean, not along the spread', 'to make the eigenvalues positive', 'to save time', 'it makes no difference'], answer: 0, why: 'Without centring, the largest “spread” is the offset of the whole cloud from the origin.' },
      { q: 'Should features be standardised before PCA?', options: ['yes, when they are in different units or scales', 'never', 'always, even in the same units', 'only for two features'], answer: 0, why: 'A feature measured in large numbers would otherwise dominate the variance and the first component.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const HIGH = {
    wine: { label: 'wine: 13 chemical measurements (standardised)', make: () => { const d = S.Data.load('wine'); return { X: P.std(d.X), y: d.y, feat: d.features.map((f) => f.name.replace(/_/g, ' ')), cls: ['cultivar 1', 'cultivar 2', 'cultivar 3'] }; } },
    iris: { label: 'iris: 4 measurements in cm', make: () => { const d = S.Data.load('iris'); return { X: d.X, y: d.y, feat: d.features.map((f) => f.name.replace(/_/g, ' ')), cls: ['setosa', 'versicolor', 'virginica'] }; } },
    penguins: { label: 'penguins: 4 measurements (standardised)', make: () => { const d = S.Data.load('penguins'), rows = d.X.map((x, i) => [x[1], x[2], x[3], x[4], d.y[i]]).filter((r) => r.slice(0, 4).every((v) => v !== null && isFinite(v))); return { X: P.std(rows.map((r) => r.slice(0, 4))), y: rows.map((r) => r[4]), feat: ['bill length', 'bill depth', 'flipper length', 'body mass'], cls: ['species 1', 'species 2', 'species 3'] }; } },
  };
  S.topic({
    id: 'pca-variance', family: 'p07-pca', order: 20, title: 'Explained variance in many dimensions',
    blurb: 'With many features, PCA sorts the directions by how much variance they carry. The scree plot shows each component’s share; the running total tells how many components keep, say, 90% of the variance. Each component is a mix of the original features, its loadings, and the first two give a flat map of the data.',
    inputs: [{ id: 'set', kind: 'choice', label: 'Data', def: 'wine', options: Object.keys(HIGH).map((k) => ({ value: k, label: HIGH[k].label })) }],
    presets: [{ label: 'Wine, 13 features', values: { set: 'wine' } }, { label: 'Iris, 4 features', values: { set: 'iris' } }, { label: 'Penguins, 4 features', values: { set: 'penguins' } }],
    counters: [{ id: 'k', label: 'components shown' }],
    run(inp) {
      const d = once('hi:' + inp.set, () => HIGH[inp.set].make()), X = d.X, dim = X[0].length, F = H().frames(['k']), q = F.q;
      const pca = once('hp:' + inp.set, () => ML().PCA({ trace: false }).fit(X)), r = pca.explainedVarianceRatio, cum = r.map((_, i) => r.slice(0, i + 1).reduce((a, b) => a + b, 0));
      const k90 = cum.findIndex((c) => c >= 0.9) + 1;
      const SA = { x: 70, y: 40, w: 680, h: 300 };
      const sax = C().axes(Object.assign({ prefix: 'sc', xr: [0.4, dim + 0.6], yr: [0, 1], xticks: [...Array(dim).keys()].map((i) => i + 1), yticks: [0, 0.25, 0.5, 0.75, 1], fmtY: (v) => H().pct(v), xname: 'component', yname: 'share of the variance' }, SA));
      const bw = Math.min(40, (SA.w / dim) * 0.6);
      const scree = (upto) => { const it = [...sax.items]; r.forEach((v, i) => it.push({ key: 'sc:b' + i, kind: 'cell', x: sax.X(i + 1) - bw / 2, y: sax.Y(v), w: bw, h: Math.max(1, sax.Y(0) - sax.Y(v)), rx: 2, text: '', cls: 'bar ' + (i < upto ? 'c0' : 'soft'), audit: 'ignore' })); it.push(C().line(sax, 'sc:cum', cum.map((c, i) => [i + 1, c]), 'c1'), C().hline(sax, 'sc:90', 0.9, 'act')); it.push(...C().legend('sc:lg', SA.x + 20, SA.y + SA.h + 62, [{ text: 'each component', cls: 'c0', dot: true }, { text: 'running total', cls: 'c1' }, { text: '90%', cls: 'act' }])); return it; };
      F.set('k', k90);
      F.push(scree(k90), `PCA of ${P.qpl(q, X.length, 'row')} with ${q.n(dim)} features. PC1 carries ${q.pct(r[0])} of the variance, PC2 ${q.pct(r[1])}` + (dim > 2 ? `, PC3 ${q.pct(r[2])}` : '') + `. The running total passes 90% at ${P.qpl(q, k90, 'component')}: ${q.n(dim - k90)} of the ${q.n(dim)} directions together hold under a tenth of the spread.`, { phase: 'scree', r, cum, k90, dim, n: X.length, rest: dim - k90 });
      const loadings = (j) => {
        const u = pca.components[j], it = [H().head('ld:h', 230, 30, 'What PC' + (j + 1) + ' is made of (its loadings)')], rowH = Math.min(28, 380 / dim), mx = Math.max(...u.map(Math.abs));
        u.forEach((w, f) => {
          const y = 56 + f * rowH, len = (200 * Math.abs(w)) / mx;
          it.push(H().text('ld:n' + f, 220, y + rowH * 0.6, d.feat[f], 'legend', 12, 'end'), { key: 'ld:b' + f, kind: 'cell', x: w >= 0 ? 440 : 440 - len, y: y + 3, w: Math.max(1.5, len), h: rowH - 7, rx: 2, text: '', cls: 'bar ' + (w >= 0 ? 'pos' : 'neg'), audit: 'ignore' }, H().text('ld:v' + f, w >= 0 ? 446 + len : 434 - len, y + rowH * 0.6, H().fx(w, 2), 'legend', 12, w >= 0 ? 'start' : 'end'));
        });
        it.push({ key: 'ld:z', kind: 'path', d: `M440,50 V${56 + dim * rowH}`, cls: 'axis', audit: 'ignore', layer: 'back' });
        return it;
      };
      for (const j of [0, 1]) {
        const u = pca.components[j], top = u.map((w, f) => [Math.abs(w), f]).sort((a, b) => b[0] - a[0]).slice(0, 2).map(([, f]) => f);
        F.push(loadings(j), `PC${q.n(j + 1)} is a weighted sum of the ${P.qpl(q, dim, 'feature')}. Its largest weights are on ${d.feat[top[0]]} (${q(u[top[0]], 2)}) and ${d.feat[top[1]]} (${q(u[top[1]], 2)}); a negative weight means that feature pulls the other way.`, { phase: 'load', j, pcn: j + 1, u, top, dim });
      }
      const Z = pca.transform(X).map((z) => [z[0], z[1]]), box = { x: 60, y: 20, w: 420, h: 420 }, ax = P.axes('pz', Z, box, { xname: 'PC1', yname: 'PC2' });
      const items = [...ax.items, ...P.points(ax, 'pz:p', Z, d.y, { r: 4 })];
      d.cls.forEach((c, k) => items.push(...P.row('pz:l' + k, 530, 80 + k * 24, k, c)));
      items.push(H().text('pz:n', 530, 170, 'coloured by the known class,', 'axname', 12), H().text('pz:n2', 530, 188, 'which PCA never saw', 'axname', 12));
      F.push(items, `Every row placed by its first two component scores: ${q.pct(cum[1])} of the variance in one flat picture. The colours are the known classes, which PCA never used, yet they fall into separate regions.`, { phase: 'map', cum2: cum[1] });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Explained variance', text: ['share of PCj = λⱼ / Σ λ', 'running total: (λ₁ + … + λₖ) / Σ λ', 'loadings of PCj: the entries of uⱼ', 'score of a row on PCj: uⱼ·(x − μ)'].join('\n') }] } };
    },
    predict: [{ at: (f) => f.meta.phase === 'scree', pick: 'option', ask: 'How many components does the running total need to reach 90%?', options: (fr) => { const d = fr[0].meta.dim; return [...Array(Math.min(d, 8)).keys()].map((i) => String(i + 1)); }, answer: (fr) => String(fr[0].meta.k90), explain: (fr) => fr[0].say }],
    quiz: [
      { q: 'Shares 50%, 30%, 15%, 5%. How many components keep 90%?', options: ['3', '2', '4', '1'], answer: 0, why: '50 + 30 = 80 is short; adding 15 gives 95.' },
      { q: 'A loading of −0.6 for a feature on PC1 means…', options: ['rows high in that feature score low on PC1', 'the feature is ignored', 'the feature has negative values', 'PC1 explains 60% less'], answer: 0, why: 'The score is a weighted sum; a negative weight pulls it down.' },
      { q: 'Do the shares of all the components add up to 100%?', options: ['yes', 'no, they add to the number of features', 'only after standardising', 'only for two features'], answer: 0, why: 'The eigenvalues add up to the total variance (the trace of the covariance).' },
      { q: 'Does PCA use the class labels?', options: ['no: it only looks at the spread of the features', 'yes, to choose the directions', 'only for the second component', 'only when the classes are balanced'], answer: 0, why: 'It is unsupervised; that the classes separate on the map is a finding, not a guarantee.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  const SZ = 10, KS = [1, 2, 5, 10, 20, 40];
  const digitsPCA = () => once('digits', () => { const d = S.Data.digits({ n: 150, size: SZ, seed: 3 }), X = d.X.map((r) => Array.from(r)); return { X, y: d.y, pca: ML().PCA({ solver: 'eigen', trace: false }).fit(X) }; });
  S.topic({
    id: 'pca-recon', family: 'p07-pca', order: 30, title: 'Rebuilding images from a few components',
    blurb: 'A 10 × 10 image is a point in 100 dimensions. PCA of 150 drawn digits finds the directions in which digit images vary most: the mean image plus a few component images, each added in some amount, rebuild a digit. The more components, the closer the rebuild, and the error left is exactly the variance of the components left out.',
    inputs: [{ id: 'pick', kind: 'number', label: 'First image shown (0 to 145)', def: 0, min: 0, max: 145 }],
    presets: [{ label: 'Images 0 to 3', values: { pick: 0 } }, { label: 'Images 20 to 23', values: { pick: 20 } }, { label: 'Images 101 to 104', values: { pick: 101 } }],
    counters: [{ id: 'k', label: 'components used' }],
    run(inp) {
      const { X, y, pca } = digitsPCA(), F = H().frames(['k']), q = F.q, dim = X[0].length, ids = [0, 1, 2, 3].map((t) => inp.pick + t);
      const img = (key, x, yy, vals, scheme, max, size) => ({ key, kind: 'raster', x, y: yy, w: size || 70, h: size || 70, nx: SZ, ny: SZ, vals, scheme, max, smooth: false, cls: 'p7img', layer: 'items' });
      const cum = pca.explainedVarianceRatio.map((_, i) => pca.explainedVarianceRatio.slice(0, i + 1).reduce((a, b) => a + b, 0));
      const head = (t) => H().head('rc:h', 40, 30, t);
      const comps = pca.components.slice(0, 6), cmax = Math.max(...comps.flat().map(Math.abs));
      const items0 = [head('The mean image and the first six components'), img('rc:mean', 40, 60, pca.mean, 'ink', 1, 96), H().text('rc:mt', 88, 176, 'mean', 'legend', 12, 'middle')];
      comps.forEach((c, j) => { items0.push(img('rc:c' + j, 170 + j * 100, 60, c, 'div', cmax, 86), H().text('rc:ct' + j, 213 + j * 100, 166, 'PC' + (j + 1) + ': ' + H().pct(pca.explainedVarianceRatio[j]), 'legend', 12, 'middle')); });
      items0.push(H().text('rc:note', 170, 196, 'blue: add ink, vermilion: remove it, when the component’s score is positive', 'axname', 12));
      F.push(items0, `${P.qpl(q, X.length, 'image')} of ${q.n(dim)} pixels each. Their mean is a blurred digit; the first six components are patterns of ink to add or remove. PC1 alone explains ${q.pct(pca.explainedVarianceRatio[0])} of the pixel variance, and the first six ${q.pct(cum[5])}.`, { phase: 'comps', n: X.length, dim, r: pca.explainedVarianceRatio, cum6: cum[5] });
      const Z = pca.transform(ids.map((i) => X[i]));
      const errs = [];
      KS.forEach((k) => {
        F.set('k', k);
        const R = pca.inverseTransform(Z.map((z) => z.map((v, j) => (j < k ? v : 0))));
        const e = ids.map((i, t) => ML().sqdist(X[i], R[t]));
        errs.push(e.reduce((a, b) => a + b, 0) / e.length);
        const it = [head('Four images rebuilt from ' + H().pl(k, 'component'))];
        ids.forEach((i, t) => {
          const x0 = 40 + t * 185;
          it.push(img('rc:o' + t, x0, 60, X[i], 'ink', 1, 76), img('rc:r' + t, x0 + 86, 60, R[t].map((v) => Math.max(0, Math.min(1, v))), 'ink', 1, 76));
          it.push(H().text('rc:ol' + t, x0 + 38, 156, 'digit ' + y[i], 'legend', 12, 'middle'), H().text('rc:rl' + t, x0 + 124, 156, 'error ' + H().fx(e[t], 2), 'legend', 12, 'middle'));
        });
        const EA = { x: 100, y: 220, w: 560, h: 170 }, emax = errs[0] * 1.1;
        const eax = C().axes(Object.assign({ prefix: 're', xr: [0, 42], yr: [0, emax], xticks: [0, 10, 20, 30, 40], yticks: C().ticks(0, emax, 5).filter((v) => v <= emax), xname: 'components kept', yname: 'mean squared error of the four rebuilds' }, EA));
        it.push(...eax.items, C().line(eax, 're:l', KS.slice(0, errs.length).map((kk, t) => [kk, errs[t]]), 'act'), C().dot(eax, 're:d', k, errs[errs.length - 1], '', 5));
        F.push(it, `With ${P.qpl(q, k, 'component')} (left: the image, right: the rebuild) the four images have mean squared error ${q(errs[errs.length - 1], 3)}; across all the images, ${q.pct(cum[k - 1])} of the pixel variance is kept.` + (k === 1 ? ' One number per image gives only a blur.' : k >= 20 ? ` The digits are now easy to read, from ${q.n(k)} numbers instead of ${q.n(dim)}.` : ''), { phase: 'rebuild', k, err: errs[errs.length - 1], errs: e, kept: cum[k - 1], ids, dim });
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Rebuilding', text: ['scores: zⱼ = uⱼ·(x − μ)', 'rebuild with k: x̂ = μ + z₁u₁ + … + zₖuₖ', 'mean error over all images:', '  (n − 1)/n · (λₖ₊₁ + … + λ_d)'].join('\n') }] } };
    },
    predict: [{ at: (f) => f.meta.phase === 'rebuild' && f.meta.k === 2, pick: 'option', ask: 'Going from 2 to 5 components, will the error of every one of the four images fall?', options: ['yes: adding components can only bring each rebuild closer', 'no: some may get worse'], answer: 'yes: adding components can only bring each rebuild closer', explain: (fr, i) => fr[i + 1].say }],
    quiz: [
      { q: 'How many numbers describe a rebuilt image with k components?', options: ['k scores (plus the shared mean and components)', 'all its pixels', 'k²', 'one'], answer: 0, why: 'Each image keeps only its k scores; the mean and the components are shared by all.' },
      { q: 'Why can a component image have negative values?', options: ['it is a direction: added with a negative score, or removing ink where it is negative', 'pixels can be negative', 'it is a mistake', 'only the mean can'], answer: 0, why: 'Components describe changes from the mean, not images themselves.' },
      { q: 'With all 100 components, the rebuild error is…', options: ['zero', 'the variance of PC1', 'one pixel', 'the mean image'], answer: 0, why: 'All directions together span every image exactly.' },
      { q: 'Adding the next component lowers the total squared error by…', options: ['(n − 1) times its eigenvalue, over the data PCA was fitted on', 'a fixed amount', 'nothing after the first', 'half each time'], answer: 0, why: 'The error is the variance of the components left out.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
