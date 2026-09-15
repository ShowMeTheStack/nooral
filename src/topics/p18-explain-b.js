// Part 18, explaining one prediction: SHAP-style additive contributions drawn as a waterfall from the average output
// to this row's output (exact Shapley values over every coalition of features, Tree SHAP for a single tree, Kernel
// SHAP as a check), and a LIME-style local surrogate: samples round one point weighted by closeness, and the straight
// line that best copies the model there.
(function (S) {
  'use strict';
  const P = S.P18, H = () => S.Kit.ml, C = () => S.Kit.curve, E = () => S.Explain;
  const choose = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return Math.round(r); };

  P.SHAPCASES = {
    loans: { label: 'loans: chance of default (logistic regression)', unit: 'chance of default' },
    rents: { label: 'rents: a random-forest regressor', unit: 'rent' },
    tree: { label: 'rents: one tree, by Tree SHAP', unit: 'rent' },
  };
  P.shapModel = (id) => P.once('shm:' + id, () => {
    if (id === 'loans') {
      const d = S.Data.load('loans', { n: 400 }), X = d.X.map((r) => r.slice(0, 5)), sp = S.Data.split(d.y, 0.3, 1), Xtr = sp.train.map((i) => X[i]), sc = P.scaler(Xtr);
      const m = new S.ML.LogisticRegression({}).fit(sc.apply(Xtr), sp.train.map((i) => d.y[i]));
      return { names: ['income', 'debt ratio', 'years', 'missed', 'amount'], Xtr, Xte: sp.test.map((i) => X[i]), f: (Z) => m.predictProba(sc.apply(Z)).map((r) => r[1]), d: 3 };
    }
    const d = S.Data.load('rents'), sp = S.Data.split(d.y, 0.3, 1, false), Xtr = sp.train.map((i) => d.X[i]), ytr = sp.train.map((i) => d.y[i]);
    const names = ['area', 'rooms', 'floor', 'distance', 'balcony'];
    if (id === 'tree') { const t = new S.ML.DecisionTreeRegressor({ maxDepth: 4 }).fit(Xtr, ytr); return { names, Xtr, Xte: sp.test.map((i) => d.X[i]), f: (Z) => t.predict(Z), tree: E().treeFromNodes(t.tree), d: 0 }; }
    const m = new S.ML.RandomForestRegressor({ nEstimators: 30, maxDepth: 8, seed: 1 }).fit(Xtr, ytr);
    return { names, Xtr, Xte: sp.test.map((i) => d.X[i]), f: (Z) => m.predict(Z), d: 0 };
  });
  /** the mean marginal contribution of feature i, by coalition size, read from the values of every coalition */
  P.bySize = (values, i, M) => Array.from({ length: M }, (_, k) => {
    let s = 0, c = 0;
    for (let mask = 0; mask < 1 << M; mask++) { if ((mask >>> i) & 1) continue; let pc = 0; for (let t = mask; t; t >>>= 1) pc += t & 1; if (pc !== k) continue; s += values[mask | (1 << i)] - values[mask]; c++; }
    return { k, count: c, mean: s / c };
  });

  S.topic({
    id: 'shap-waterfall', family: 'p18-explain', order: 30, title: 'SHAP-style contributions: one prediction, explained',
    blurb: 'Why did the model say this for this row? Start from what it says on average, then let the features in one at a time; the change each one makes depends on which others are already in, so average that change over every order. These averages are Shapley values, and they add up exactly: the average output plus every feature’s contribution is this row’s output. A waterfall chart shows the walk from one to the other.',
    inputs: [
      { id: 'case', kind: 'choice', label: 'Model', def: 'loans', options: Object.keys(P.SHAPCASES).map((k) => ({ value: k, label: P.SHAPCASES[k].label })) },
      { id: 'row', kind: 'number', label: 'Test row to explain', def: 3, min: 0, max: 19 },
      { id: 'bg', kind: 'choice', label: 'Background rows (the average)', def: '30', options: ['10', '30', '60'].map((v) => ({ value: v, label: v })) },
    ],
    presets: [
      { label: 'A loan applicant', values: { case: 'loans', row: 3, bg: '30' } },
      { label: 'Another applicant', values: { case: 'loans', row: 11, bg: '30' } },
      { label: 'A flat’s rent (forest)', values: { case: 'rents', row: 0, bg: '30' } },
      { label: 'One tree, Tree SHAP', values: { case: 'tree', row: 0, bg: '30' } },
    ],
    counters: [{ id: 'added', label: 'contributions added' }],
    run(inp) {
      const Mo = P.shapModel(inp.case), x = Mo.Xte[inp.row], M = x.length, bgN = +inp.bg, bg = Mo.Xtr.slice(0, bgN), F = H().frames(['added']), q = F.q, dd = Mo.d;
      const tree = inp.case === 'tree';
      const ex = P.once(['shx', inp.case, inp.row, bgN].join(':'), () => E().shapExact(Mo.f, x, bg, { names: Mo.names }));
      const R = tree ? E().treeShap(Mo.tree, x, { names: Mo.names }) : ex, phi = R.phi, base = R.base, fx = R.fx;
      const fv = (v) => H().fx(v, dd), qv = (v) => q(v, dd);
      const order = phi.map((_, i) => i).sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
      const lab = (i) => Mo.names[i] + ' = ' + H().sg(x[i], 4);
      // the waterfall axes
      const cum = [base]; order.forEach((i) => cum.push(cum[cum.length - 1] + phi[i]));
      const all = cum.concat([fx, base]), lo = Math.min(...all), hi = Math.max(...all), pad = 0.12 * (hi - lo || 1);
      const nRow = M + 2, ax = C().axes({ prefix: 'sh:ax', x: 230, y: 70, w: 420, h: nRow * 34, xr: [lo - pad, hi + pad], yr: [0, nRow], yticks: 0, xname: P.SHAPCASES[inp.case].unit, tickSize: 12 });
      const rowY = (k) => ax.Y(nRow - k - 0.5);
      const fall = (upto) => {
        const it = [...ax.items, C().vline(ax, 'sh:base', base, ''), H().text('sh:n0', 220, rowY(0) + 4.5, 'average output', 'legend', 13, 'end'), { key: 'sh:b0', kind: 'dot', x: ax.X(base), y: rowY(0), r: 5, cls: 'mark p18mean', point: true }, H().text('sh:v0', ax.o.x + ax.o.w + 70, rowY(0) + 4.5, fv(base), 'mono', 12, 'end')];
        order.forEach((i, k) => {
          it.push(H().text('sh:n' + (k + 1), 220, rowY(k + 1) + 4.5, lab(i), k < upto ? 'strong' : 'legend', 13, 'end'));
          if (k >= upto) return;
          const a = cum[k], b = cum[k + 1], x1 = Math.min(ax.X(a), ax.X(b)), w = Math.max(1, Math.abs(ax.X(b) - ax.X(a)));
          it.push({ key: 'sh:b' + (k + 1), kind: 'cell', x: P.r2(x1), y: P.r2(rowY(k + 1) - 10), w: P.r2(w), h: 20, rx: 2, text: '', cls: 'bar ' + (phi[i] >= 0 ? 'pos' : 'neg'), audit: 'ignore' });
          const t = (phi[i] >= 0 ? '+' : '−') + H().fx(Math.abs(phi[i]), dd);
          it.push(H().text('sh:v' + (k + 1), ax.o.x + ax.o.w + 70, rowY(k + 1) + 4.5, t, 'mono ' + (phi[i] >= 0 ? 'pos' : 'neg'), 12, 'end'));
          if (k + 1 < M) it.push({ key: 'sh:c' + (k + 1), kind: 'path', d: `M${ax.X(b)},${P.r2(rowY(k + 1) + 10)} V${P.r2(rowY(k + 2) - 10)}`, cls: 'p18join', audit: 'ignore', layer: 'lines' });
        });
        if (upto >= M) it.push(H().text('sh:nf', 220, rowY(M + 1) + 4.5, 'this row’s output', 'strong', 13, 'end'), { key: 'sh:bf', kind: 'dot', x: ax.X(fx), y: rowY(M + 1), r: 6, cls: 'mark p18pick', point: true }, H().text('sh:vf', ax.o.x + ax.o.w + 70, rowY(M + 1) + 4.5, fv(fx), 'strong mono', 12, 'end'));
        return it;
      };
      F.push([...fall(0), H().head('sh:h', 20, 34, 'Test row ' + inp.row + ': from the average to this answer')],
        `Over ${P.qpl(q, bgN, 'background row')} the model gives ${qv(base)} on average; for test row ${q.n(inp.row)} it gives ${qv(fx)}. The ${P.qpl(q, M, 'feature')} must share the difference, ${qv(fx - base)}, between them.` + (tree ? ' For a single tree the average is taken over the training rows that reach each branch, which is what Tree SHAP uses.' : ''), { phase: 'row', base, fx, diff: fx - base, bg: bgN, M, row: inp.row });
      // one feature's coalitions
      const top = order[0], sizes = P.bySize(ex.values, top, M), phiTop = ex.phi[top];
      const tab = S.Kit.dt.table({ prefix: 'sh:tb', x: 60, y: 90, cols: [{ name: 'others already in', w: 150 }, { name: 'coalitions', w: 100 }, { name: 'mean change', w: 130 }, { name: 'weight', w: 80 }], rows: sizes.map((s) => [String(s.k), String(s.count), (s.mean >= 0 ? '+' : '−') + H().fx(Math.abs(s.mean), dd + 1), '1/' + M]), rowH: 30, size: 13 });
      F.push([...tab.items, H().head('sh:h', 20, 34, Mo.names[top] + ': its change, averaged over coalitions'), ...H().lines('sh:l', 60, 130 + M * 32 + 20, [['φ = (1/' + M + ') × sum of the mean changes = ' + (phiTop >= 0 ? '+' : '−') + H().fx(Math.abs(phiTop), dd + 1), 'math'], ['a coalition’s value: the average output with those', 'legend'], ['features fixed to this row and the rest from background rows', 'legend']], 13, 22)],
        `Take ${Mo.names[top]}. Add it to a coalition of the other features (those fixed at this row’s values, the rest taken from the background rows) and see how the average output changes. With no others in, adding it changes the output by ${q(sizes[0].mean, dd + 1)}; with all ${q.n(M - 1)} others in, by ${q(sizes[M - 1].mean, dd + 1)}. Shapley weights give every coalition size the same share, so its contribution is the plain mean of the ${P.qpl(q, M, 'row')}: ${q(phiTop, dd + 1)}` + (tree ? ` (Tree SHAP, which averages over the tree’s own training rows, gives ${q(phi[top], dd + 1)}).` : '.'), { phase: 'coal', top, sizes: sizes.map((s) => s.mean), counts: sizes.map((s) => s.count), first: sizes[0].mean, last: sizes[M - 1].mean, phi: phiTop, treePhi: phi[top], M });
      order.forEach((i, k) => {
        F.count('added');
        F.push([...fall(k + 1), H().head('sh:h', 20, 34, 'Test row ' + inp.row + ': adding the contributions')],
          `${Mo.names[i][0].toUpperCase() + Mo.names[i].slice(1)} (${q.sg(x[i], 4)}) ${phi[i] >= 0 ? 'pushes the output up' : 'pulls the output down'} by ${q(Math.abs(phi[i]), dd)}, to ${qv(cum[k + 1])}.`, { phase: 'add', i, phi: Math.abs(phi[i]), signed: phi[i], to: cum[k + 1], x: x[i] });
      });
      const sum = base + phi.reduce((a, b) => a + b, 0);
      let check;
      if (tree) check = `The contributions add up: ${qv(base)} plus their sum is ${qv(sum)}, the tree’s answer ${qv(fx)}. Tree SHAP got them by walking each root-to-leaf path once, without trying every coalition.`;
      else {
        const ks = P.once(['shk', inp.case, inp.row, bgN].join(':'), () => E().kernelShap(Mo.f, x, bg, { nSamples: 64 }));
        const gap = Math.max(...ks.phi.map((v, j) => Math.abs(v - phi[j])));
        check = `The contributions add up: ${qv(base)} plus their sum is ${qv(sum)}, the model’s answer. Kernel SHAP, a weighted regression over coalitions, finds the same values to within ${q.sg(gap, 2)}.`;
        F.push(fall(M).concat([H().head('sh:h', 20, 34, 'Test row ' + inp.row + ': the full walk')]), check, { phase: 'check', base, sum, fx, gap, phi });
      }
      if (tree) F.push(fall(M).concat([H().head('sh:h', 20, 34, 'Test row ' + inp.row + ': the full walk')]), check, { phase: 'check', base, sum, fx, phi });
      for (const f of F.list) f.narrowView = null;
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Shapley values', text: ['v(S) = mean over background b of f(x on S, b elsewhere)', 'φᵢ = Σ_S |S|!(M−|S|−1)!/M! · (v(S ∪ {i}) − v(S))', '   = mean over sizes of the mean change', 'v(∅) + Σᵢ φᵢ = v(all) = f(x)', 'Kernel SHAP: weighted regression on coalitions'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'row', pick: 'option', once: true,
      ask: 'Which feature will move this prediction the most?',
      options: (fr, i, inp) => P.shapModel(inp.case).names,
      answer: (fr, i, inp) => P.shapModel(inp.case).names[fr.find((f) => f.meta.phase === 'coal').meta.top],
      explain: (fr) => fr.find((f) => f.meta.phase === 'add').say,
    }],
    quiz: [
      { q: 'Average output 0.20, contributions +0.15, −0.05 and +0.10. The row’s output is…', options: ['0.40', '0.20', '0.50', '0.30'], answer: 0, why: '0.20 + 0.15 − 0.05 + 0.10.' },
      { q: 'Why average a feature’s change over coalitions?', options: ['its effect depends on which other features are already known', 'to make it faster', 'to remove noise in the data', 'because features are independent'], answer: 0, why: 'With interactions, the order of adding features changes each one’s effect.' },
      { q: 'With 5 features, how many coalitions must an exact computation evaluate?', options: ['32', '5', '10', '120'], answer: 0, why: '2⁵ subsets of the features.' },
      { q: 'A SHAP value of +0.08 for income means…', options: ['for this row, income moved the output up by 0.08 from the average', 'income is important for every row', 'raising income by 1 adds 0.08', 'income causes default'], answer: 0, why: 'It is a local, additive share of this one prediction.' },
    ],
  });

  // ---- LIME --------------------------------------------------------------------------------------------------------
  P.LIMESETS = {
    moons: { label: 'two moons, an RBF SVM', make: () => S.Data.moons({ n: 200, noise: 0.2, seed: 1 }), points: { edge: [0.5, 0.3], bend: [1.6, -0.15], deep: [-0.6, 0.6] } },
    circles: { label: 'a ring round a disc, an RBF SVM', make: () => S.Data.circles({ n: 200, noise: 0.1, seed: 1 }), points: { edge: [0.7, 0.2], bend: [0, -0.72], deep: [0.05, 0.1] } },
  };
  const PTS = { edge: 'near the boundary', bend: 'where the boundary bends', deep: 'deep inside one class' };
  P.limeModel = (set) => P.once('lm:' + set, () => { const d = P.LIMESETS[set].make(), m = new S.ML.SVC({ C: 3 }).fit(d.X, d.y); return { X: d.X, y: d.y, f: (Z) => m.decisionFunction(Z) }; });

  S.topic({
    id: 'lime-local', family: 'p18-explain', order: 40, title: 'LIME-style local surrogates',
    blurb: 'A curved model is hard to read as a whole, but near one point almost any model looks straight. Scatter samples round the point, ask the model about each, weight the samples by how close they are, and fit a weighted straight line (a plane, in more dimensions). Its slopes say which features push the answer which way here, and its R² says how faithful the straight copy is. The kernel width decides what “near” means: too wide and the line is copying the curve far away.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data and model', def: 'moons', options: Object.keys(P.LIMESETS).map((k) => ({ value: k, label: P.LIMESETS[k].label })) },
      { id: 'point', kind: 'choice', label: 'Point to explain', def: 'edge', options: Object.keys(PTS).map((k) => ({ value: k, label: PTS[k] })) },
      { id: 'width', kind: 'slider', label: 'Kernel width', def: 0.5, min: 0.2, max: 3, step: 0.1, fmt: (v) => v.toFixed(1) },
      { id: 'samples', kind: 'choice', label: 'Samples', def: '200', options: ['100', '200', '400'].map((v) => ({ value: v, label: v })) },
    ],
    presets: [
      { label: 'Near the boundary', values: { set: 'moons', point: 'edge', width: 0.5, samples: '200' } },
      { label: 'A bend, narrow kernel', values: { set: 'moons', point: 'bend', width: 0.3, samples: '200' } },
      { label: 'A bend, wide kernel', values: { set: 'moons', point: 'bend', width: 3, samples: '200' } },
      { label: 'Ring and disc', values: { set: 'circles', point: 'edge', width: 0.5, samples: '200' } },
    ],
    counters: [{ id: 'asked', label: 'model calls' }],
    run(inp) {
      const Mo = P.limeModel(inp.set), x = P.LIMESETS[inp.set].points[inp.point], F = H().frames(['asked']), q = F.q, n = +inp.samples;
      const L = P.once(['lime', inp.set, inp.point, inp.width, n].join(':'), () => E().lime(Mo.f, x, Mo.X, { nSamples: n, seed: 1, kernelWidth: inp.width }));
      const box = { x: 40, y: 50, w: 420, h: 340 }, ax = P.axes('lm:ax', Mo.X, box, { pad: 0.1 }), fmax = 2;
      const field = S.Kit.scatter.field(ax, 'lm:f', (u, v) => 1 / (1 + Math.exp(-fmax * Mo.f([[u, v]])[0])), { n: 60, scheme: 'prob' });
      const bound = H().level(ax, 'lm:b', (u, v) => Mo.f([[u, v]])[0], 0, '');
      const data = Mo.X.map((p, i) => ({ key: 'lm:d' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 2.4, cls: 'sp c' + Mo.y[i] + ' faded', audit: 'overlap-ok', layer: 'items' }));
      const me = { key: 'lm:x', kind: 'dot', x: ax.X(x[0]), y: ax.Y(x[1]), r: 7, cls: 'mark p18pick', point: true, z: 6 };
      const inBox = (p) => ax.inside(p[0], p[1]);
      const samp = (sized) => L.samples.map((z, i) => (i && inBox(z) ? { key: 'lm:s' + i, kind: 'dot', x: ax.X(z[0]), y: ax.Y(z[1]), r: sized ? P.r2(1 + 5 * L.weights[i]) : 2.2, cls: 'sp ' + (L.predictions[i] >= 0 ? 'c1' : 'c0') + (sized && L.weights[i] < 0.05 ? ' faded' : ''), audit: 'overlap-ok', layer: 'items', z: 3 } : null)).filter(Boolean);
      const side = (rows) => H().lines('lm:r', 490, 80, rows, 13, 24);
      const fx = L.predictions[0];
      F.push([field, bound, ...ax.items, ...data, me, ...side([['The model', 'strong'], ['an RBF SVM; shading: its score', 'legend'], ['blue below 0, orange above', 'legend'], ['solid line: score 0, the boundary', 'legend'], ['green dot: the point to explain', 'legend']])],
        `The SVM’s score is shaded, with its boundary where the score is zero. The point to explain, (${q(x[0], 2)}, ${q(x[1], 2)}), gets a score of ${q(fx, 3)}. Read as a whole the boundary is curved; LIME looks only nearby.`, { phase: 'model', x, fx });
      F.set('asked', n);
      F.push([field, bound, ...ax.items, me, ...samp(false), ...side([['Samples round the point', 'strong'], [n + ' draws, each feature with', 'legend'], ['its own spread in the data', 'legend'], ['colour: the model’s answer', 'legend']])],
        `Draw ${P.qpl(q, n, 'sample')} round the point, each feature jittered by its own spread in the data, and ask the model about every one. Most land far from the point, where the model may behave quite differently.`, { phase: 'samples', n });
      const near = L.weights.filter((w) => w > 0.5).length;
      F.push([field, bound, ...ax.items, me, ...samp(true), ...side([['Weighted by closeness', 'strong'], ['kernel width ' + H().fx(inp.width, 1) + ' (in spreads)', 'legend'], [near + ' samples weigh over one half', 'legend']])],
        `Weight each sample by exp(−distance² / width²), with distances measured in spreads and a kernel width of ${q(inp.width, 1)}: dots are drawn by weight. ${P.qpl(q, near, 'sample')} of ${q.n(n)} weigh more than one half; the far ones hardly count.`, { phase: 'weights', width: inp.width, near, n });
      const g = (u, v) => L.intercept + L.coef[0] * (u - x[0]) + L.coef[1] * (v - x[1]);
      const sur = H().level(ax, 'lm:g', g, 0, 'p18sur');
      const arrowLen = 60, nrm = Math.hypot(L.coef[0] * (ax.X(1) - ax.X(0)), L.coef[1] * (ax.Y(0) - ax.Y(1))) || 1;
      const dx = (L.coef[0] * (ax.X(1) - ax.X(0)) / nrm) * arrowLen, dy = -(L.coef[1] * (ax.Y(0) - ax.Y(1)) / nrm) * arrowLen;
      F.push([field, bound, sur, ...ax.items, me, ...samp(true), H().arrow('lm:ar', [ax.X(x[0]), ax.Y(x[1])], [ax.X(x[0]) + dx, ax.Y(x[1]) + dy], 'p18arrow'), ...side([['The local straight copy', 'strong'], ['dashed: where it says 0', 'legend'], ['slope x₁ ' + H().fx(L.coef[0], 2) + ', x₂ ' + H().fx(L.coef[1], 2), 'math'], ['weighted R² ' + H().fx(L.score, 2), 'math'], ['arrow: the way the score rises', 'legend']])],
        `A weighted least-squares fit gives the local surrogate: score ≈ ${q(L.intercept, 3)}${q.t(L.coef[0], 2)} × Δx₁${q.t(L.coef[1], 2)} × Δx₂ (dashed where it is zero). Here, raising x₁ ${L.coef[0] >= 0 ? 'raises' : 'lowers'} the score and raising x₂ ${L.coef[1] >= 0 ? 'raises' : 'lowers'} it. Its weighted R² is ${q(L.score, 2)}` + (L.score > 0.9 ? ': near this point the model really is almost straight.' : L.score > 0.6 ? ': a fair copy, though the boundary curves within the kernel.' : ': a poor copy; the kernel reaches round the curve, so the slopes should not be trusted.'), { phase: 'fit', intercept: L.intercept, coef: L.coef, score: L.score });
      for (const f of F.list) f.narrowView = { x: 0, y: 20, w: 480, h: 400 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'A local surrogate', text: ['zₖ = x + noise (each feature by its spread)', 'wₖ = exp(−‖(zₖ − x)/σ‖² / width²)', 'minimise Σₖ wₖ (f(zₖ) − b − cᵀ(zₖ − x))² + ridge', 'slopes c: the local effect of each feature', 'weighted R²: how faithful the copy is'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'weights', pick: 'option', once: true,
      ask: 'Will the straight copy be faithful here (weighted R² above 0.9)?',
      options: ['yes', 'no'],
      answer: (fr) => (fr[fr.length - 1].meta.score > 0.9 ? 'yes' : 'no'),
      explain: (fr) => fr[fr.length - 1].say,
    }],
    quiz: [
      { q: 'What does a LIME-style explanation describe?', options: ['how the model behaves near one point', 'the whole model', 'the training data', 'the most important feature overall'], answer: 0, why: 'It is a local, weighted straight-line copy.' },
      { q: 'Making the kernel very wide tends to…', options: ['copy the model far from the point, lowering faithfulness', 'make the copy exact', 'remove all samples', 'change the model'], answer: 0, why: 'Far samples, where the model curves, pull on the line.' },
      { q: 'A low weighted R² tells you…', options: ['the straight copy does not match the model nearby', 'the model is wrong', 'the point is misclassified', 'the features are unimportant'], answer: 0, why: 'Its slopes should then not be trusted.' },
      { q: 'Why are the samples weighted at all?', options: ['so that nearby samples decide the fit', 'to speed up the model', 'to balance the classes', 'they are not'], answer: 0, why: 'The goal is a copy that is good near the point.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
