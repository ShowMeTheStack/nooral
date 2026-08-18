// Part 4, nearest neighbours: the Voronoi cells of 1-NN (every place takes the class of the training point it is
// nearest to), and how k smooths the boundary (the class map for growing k, training and test error against k).
(function (S) {
  'use strict';
  const H = S.PL, P = S.PL4, ML = () => S.ML;
  const n$ = H.n;
  const setInput = { id: 'set', kind: 'choice', label: 'data', def: 'blobs', options: S.PL4.SETS };

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'knn-voronoi', family: 'p04-knn', order: 10, title: 'One nearest neighbour and its Voronoi cells',
    blurb: 'The 1-nearest-neighbour rule gives every place in the plane the class of the training point closest to it. The places closest to one training point form its Voronoi cell, a convex polygon bounded by the perpendicular bisectors to its neighbours. The cells tile the plane, and the class boundary is made of the walls between cells of different classes.',
    inputs: [
      setInput,
      { id: 'n', kind: 'number', label: 'points', def: 16, min: 3, max: 40 },
      { id: 'noise', kind: 'slider', label: 'spread', def: 0.5, min: 0, max: 1, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 4, min: 1, max: 9999 },
      { id: 'qx', kind: 'slider', label: 'query across', def: 0.55, min: 0.02, max: 0.98, step: 0.01 },
      { id: 'qy', kind: 'slider', label: 'query up', def: 0.45, min: 0.02, max: 0.98, step: 0.01 },
    ],
    presets: [
      { label: 'Three blobs, 16 points', values: { set: 'blobs', n: 16, noise: 0.5, seed: 4, qx: 0.55, qy: 0.45 } },
      { label: 'Two moons, 30 points', values: { set: 'moons', n: 30, noise: 0.4, seed: 2, qx: 0.5, qy: 0.5 } },
      { label: 'Ring and disc', values: { set: 'circles', n: 24, noise: 0.3, seed: 6, qx: 0.3, qy: 0.62 } },
      { label: 'Wildflowers (real measurements)', values: { set: 'flowers', n: 30, noise: 0.5, seed: 3, qx: 0.62, qy: 0.55 } },
      { label: 'Just three points', values: { set: 'blobs', n: 3, noise: 0.5, seed: 9, qx: 0.5, qy: 0.5 } },
      { label: 'Overlapping blobs', values: { set: 'blobs', n: 30, noise: 1, seed: 12, qx: 0.45, qy: 0.5 } },
    ],
    counters: [{ id: 'dist', label: 'distances measured' }],
    run(inp) {
      const d = P.data(inp.set, inp.n, inp.seed, inp.noise), X = d.X, y = d.y, n = X.length;
      const rng = P.square(X), ax = P.plot('vo', rng, [d.xname, d.yname]);
      const q = [rng.xr[0] + inp.qx * (rng.xr[1] - rng.xr[0]), rng.yr[0] + inp.qy * (rng.yr[1] - rng.yr[0])].map((v) => S.Data.round(v, 2));
      const m = ML().KNeighborsClassifier({ k: 1 }).fit(X, y), ex = m.explain(q), tr = m.trace;
      const nearest = ex.idx[0], cls = m.predict([q])[0], kq = m.classes.indexOf(cls);
      const trainAcc = ML().accuracy(y, m.predict(X));
      const cells = P.voronoi(X, rng.xr, rng.yr);
      const area = (poly) => { let s = 0; for (let i = 0; i < poly.length; i++) { const a = poly[i], b = poly[(i + 1) % poly.length]; s += a[0] * b[1] - b[0] * a[1]; } return Math.abs(s) / 2; };
      const total = (rng.xr[1] - rng.xr[0]) * (rng.yr[1] - rng.yr[0]), areas = cells.map(area), big = ML().argmax(areas);
      // walls: edges of a cell that lie on the bisector with another point; a wall between classes is the boundary
      const walls = [];
      cells.forEach((poly, i) => poly.forEach((p, e) => {
        const qq = poly[(e + 1) % poly.length];
        if (Math.hypot(qq[0] - p[0], qq[1] - p[1]) < 1e-9) return;
        for (let j = i + 1; j < n; j++) {
          const on = (u) => Math.abs(Math.hypot(u[0] - X[i][0], u[1] - X[i][1]) - Math.hypot(u[0] - X[j][0], u[1] - X[j][1])) < 1e-7 * (1 + rng.xr[1] - rng.xr[0]);
          if (on(p) && on(qq)) walls.push({ i, j, a: p, b: qq, cross: y[i] !== y[j] });
        }
      }));
      const nCross = walls.filter((w) => w.cross).length;
      const F = new S.Frames(['dist']);
      const legend = H.classLegend('vo:lg', 520, 52, d.names.slice(0, Math.max(...y) + 1));
      const right = 520;
      const pts = (o) => S.Kit.scatter.points(ax, 'vo:p', X, y, Object.assign({ r: 5.5 }, o || {}));
      const cellItems = (hl) => cells.map((poly, i) => ({ key: 'vo:c' + i, kind: 'path', d: P.polyPath(ax, poly), cls: 'p4-cell c' + m.classes.indexOf(y[i]) + (i === hl ? ' act' : ''), audit: 'ignore', layer: 'back', z: i === hl ? 3 : 2 }));
      const wallPath = (which, cls) => { const ws = walls.filter(which); return ws.length ? [{ key: 'vo:w' + cls, kind: 'path', d: ws.map((w) => `M${H.r2(ax.X(w.a[0]))},${H.r2(ax.Y(w.a[1]))} L${H.r2(ax.X(w.b[0]))},${H.r2(ax.Y(w.b[1]))}`).join(' '), cls: 'p4-wall ' + cls, audit: 'ignore', layer: 'back', z: 4 }] : []; };
      const panel = (lines) => P.eqs('vo:eq', right, 104, lines, 14, 26);
      const meta = { names: d.names.slice(0, m.classes.length), predName: d.names[kq], n, trainAcc, nearest, dNear: ex.dist[0], classes: m.classes.length, walls: walls.length, nCross, bigArea: areas[big] / total, big, q };
      const probe = (cls2) => ({ key: 'vo:q', kind: 'dot', x: ax.X(q[0]), y: ax.Y(q[1]), r: 7, cls: 'pl-probe ' + (cls2 || ''), audit: 'overlap-ok', layer: 'items', z: 5 });

      F.add({ items: [...ax.items, legend, ...pts()].flat(), say: n + ' training points in ' + H.plural(m.classes.length, 'class', 'classes') + '. ' + tr[0].say, meta: Object.assign({ phase: 'data' }, meta) });
      F.add({ items: [...ax.items, legend, ...cellItems(-1), ...wallPath(() => true, 'all'), ...pts(), ...panel([['cell of xᵢ: the places', 'muted'], ['nearer to xᵢ than to', 'muted'], ['any other point', 'muted'], [n + ' cells, ' + walls.length + ' walls', '']])].flat(), say: 'Each point owns its Voronoi cell: the places nearer to it than to any other point. The ' + n + ' cells tile the plot and meet along ' + walls.length + ' walls, each on the perpendicular bisector of two points. The largest cell covers ' + H.pct(areas[big] / total) + ' of the plot.', meta: Object.assign({ phase: 'cells' }, meta) });
      F.add({ items: [...ax.items, legend, ...cellItems(-1), ...wallPath((w) => !w.cross, 'soft'), ...wallPath((w) => w.cross, 'cross'), ...pts(), ...panel([[nCross + ' of ' + walls.length + ' walls part classes', 'act'], ['they make the 1-NN boundary', 'act'], ['training accuracy ' + H.pct(trainAcc), 'good']])].flat(), say: nCross + ' of the ' + walls.length + ' walls separate cells of different classes: together they are the decision boundary, a chain of straight pieces. Every training point sits in its own cell, so it predicts itself: training accuracy ' + H.pct(trainAcc) + '.', meta: Object.assign({ phase: 'boundary' }, meta) });
      F.set('dist', n);
      const order = X.map((p, i) => [Math.hypot(p[0] - q[0], p[1] - q[1]), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const second = order.length > 1 ? order[1] : null;
      const rays = X.map((p, i) => H.seg('vo:r' + i, ax.X(q[0]), ax.Y(q[1]), ax.X(p[0]), ax.Y(p[1]), 'pl-seg thin' + (i === nearest ? ' act' : ''), 'ignore'));
      F.add({ items: [...ax.items, legend, ...cellItems(-1), ...wallPath((w) => !w.cross, 'soft'), ...wallPath((w) => w.cross, 'cross'), ...rays, ...pts(), probe(), ...panel([['query (' + n$(q[0]) + ', ' + n$(q[1]) + ')', ''], [n + ' distances measured', 'muted'], ['nearest: ' + n$(ex.dist[0]), 'act']].concat(second ? [['next: ' + n$(second[0]), 'muted']] : []))].flat(), say: 'A query at (' + n$(q[0]) + ', ' + n$(q[1]) + '). Brute force measures the distance to all ' + n + ' training points; the nearest is ' + n$(ex.dist[0]) + ' away' + (second ? ', the next ' + n$(second[0]) + '.' : '.'), meta: Object.assign({ phase: 'query', second: second ? second[0] : 0 }, meta) });
      F.add({ items: [...ax.items, legend, ...cellItems(nearest), ...wallPath((w) => !w.cross, 'soft'), ...wallPath((w) => w.cross, 'cross'), H.seg('vo:r' + nearest, ax.X(q[0]), ax.Y(q[1]), ax.X(X[nearest][0]), ax.Y(X[nearest][1]), 'pl-seg act', 'ignore'), ...pts({ cls: (i) => (i === nearest ? 'act' : '') }), probe('act'), ...panel([['nearest point: ' + d.names[kq], 'c' + kq], ['predicted class: ' + d.names[kq], 'act'], ['the query lies in its cell', 'muted']])].flat(), say: 'The query lies inside the cell of its nearest point, which is ' + d.names[kq] + ', so 1-NN predicts ' + d.names[kq] + '. Asking which cell a place falls in and asking for its nearest neighbour are the same question.', meta: Object.assign({ phase: 'answer' }, meta) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: '1-nearest neighbour', text: ['ŷ(x) = y of argminᵢ ‖x − xᵢ‖', 'cell(xᵢ) = {x : ‖x − xᵢ‖ ≤ ‖x − xⱼ‖ for all j}', 'wall between i and j: on the', '  perpendicular bisector of xᵢ xⱼ', 'training error is 0 (no duplicate points)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'boundary', pick: 'option', ask: 'A query point is coming (the white ring). Which class will 1-NN give it?', options: (fr) => fr[0].meta.names, answer: (fr) => fr[0].meta.predName, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'What shape is a Voronoi cell (in the Euclidean plane)?', options: ['a convex polygon, possibly open to infinity', 'always a circle', 'always a square', 'any shape at all'], answer: 0, why: 'It is the intersection of half-planes, one per other point, and an intersection of half-planes is convex.' },
      { q: 'The wall between the cells of points a and b lies on…', options: ['the perpendicular bisector of a and b', 'the line through a and b', 'a circle around a', 'the x-axis'], answer: 0, why: 'Those are exactly the places equally far from a and b.' },
      { q: 'What is the training error of 1-NN when no two training points coincide?', options: ['0', '50%', 'it depends on k', 'the same as the test error'], answer: 0, why: 'Each training point is its own nearest neighbour.' },
      { q: 'Adding one new training point changes…', options: ['only the cells next to it', 'every cell', 'no cell', 'only cells of its own class'], answer: 0, why: 'It takes a piece from each neighbouring cell; distant cells keep all their walls.' },
      { q: 'Which part of the cell walls forms the decision boundary?', options: ['the walls between cells of different classes', 'all the walls', 'the walls of the largest cell', 'none of them'], answer: 0, why: 'Crossing a wall between two cells of the same class does not change the prediction.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'knn-k', family: 'p04-knn', order: 20, title: 'k against smoothness',
    blurb: 'With k neighbours voting, a small k follows every point (a ragged boundary, no training error, high variance) and a large k averages over a wide neighbourhood (a smooth boundary that may miss real structure). The test error is usually lowest in between. Here the class map is redrawn for growing k, with the training and test errors plotted beside it.',
    inputs: [
      Object.assign({}, setInput, { def: 'moons' }),
      { id: 'n', kind: 'number', label: 'training points', def: 80, min: 20, max: 200 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.6, min: 0, max: 1, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 2, min: 1, max: 9999 },
      { id: 'ks', kind: 'list', label: 'k values', def: ['1', '3', '7', '15', '31', '61'], min: 2, max: 9, check: (v, inp) => { for (const s of v) { const k = +s; if (!Number.isInteger(k) || k < 1) return 'k values: whole numbers from 1'; if (k > inp.n) return 'k values: at most the number of training points (' + inp.n + ')'; } return null; } },
    ],
    presets: [
      { label: 'Noisy moons', values: { set: 'moons', n: 80, noise: 0.6, seed: 2, ks: ['1', '3', '7', '15', '31', '61'] } },
      { label: 'Ring and disc', values: { set: 'circles', n: 100, noise: 0.5, seed: 3, ks: ['1', '5', '11', '25', '51', '99'] } },
      { label: 'Clean moons: small k is fine', values: { set: 'moons', n: 80, noise: 0.1, seed: 5, ks: ['1', '3', '9', '27', '79'] } },
      { label: 'Three blobs', values: { set: 'blobs', n: 90, noise: 0.8, seed: 7, ks: ['1', '5', '15', '45', '89'] } },
      { label: 'Wildflowers', values: { set: 'flowers', n: 90, noise: 0.5, seed: 4, ks: ['1', '3', '9', '27', '81'] } },
    ],
    counters: [{ id: 'k', label: 'k' }],
    run(inp) {
      const d = P.data(inp.set, inp.n, inp.seed, inp.noise), X = d.X, y = d.y, n = X.length;
      const te = P.data(inp.set, inp.n, inp.seed + 1000, inp.noise);
      const ks = [...new Set(inp.ks.map(Number))].sort((a, b) => a - b), K = Math.max(...y) + 1;
      const rng = P.square(X.concat(te.X)), ax = P.plot('kk', rng, [d.xname, d.yname], { x: 44, y: 26, side: 400 });
      const kmax = Math.max(...ks), lx = { x: 530, y: 70, w: 220, h: 200 };
      const models = ks.map((k) => ML().KNeighborsClassifier({ k }).fit(X, y));
      const errs = models.map((m) => [1 - S.Metrics.accuracy(y, m.predict(X)).value, 1 - S.Metrics.accuracy(te.y, m.predict(te.X)).value]);
      const emax = Math.max(0.1, ...errs.flat()) * 1.15, est = emax > 0.4 ? 0.2 : emax > 0.2 ? 0.1 : 0.05, etop = Math.ceil(emax / est) * est;
      const cax = H.axes({ prefix: 'ke', x: lx.x, y: lx.y, w: lx.w, h: lx.h, xr: [0, Math.log10(Math.max(10, kmax * 1.2))], yr: [0, etop], xticks: [0, 1, 2].filter((v) => v <= Math.log10(Math.max(10, kmax * 1.2))), yticks: est, fmtX: (v) => String(Math.round(Math.pow(10, v))), fmtY: (v) => Math.round(v * 100) + '%', xname: 'k (log scale)', yname: 'error', grid: true });
      const F = new S.Frames(['k']);
      const rows = [];
      const legend = H.classLegend('kk:lg', 44, 470, d.names.slice(0, K));
      const elegend = S.Kit.curve.legend('ke:lg', lx.x, lx.y + lx.h + 52, [{ text: 'training', cls: 'faint' }, { text: 'test', cls: 'act' }]);
      ks.forEach((k, s) => {
        const m = models[s], trErr = errs[s][0], teErr = errs[s][1];
        rows.push({ k, trErr, teErr });
        F.set('k', k);
        const field = P.regions(ax, 'kk:f', (u, v) => { const p = m.predictProba([[u, v]])[0], c = ML().argmax(p); return [c, Math.min(1, (p[c] - 1 / K) * K / (K - 1) * 0.8 + 0.25)]; }, { n: 64 });
        const L = (key, sel, cls) => S.Kit.curve.line(cax, key, rows.map((r) => [Math.log10(r.k), sel(r)]), cls);
        const dots = rows.map((r, i) => S.Kit.curve.dot(cax, 'ke:d' + i, Math.log10(r.k), r.teErr, i === s ? '' : 'muted', i === s ? 5 : 3.5));
        const best = rows.reduce((a, b) => (b.teErr < a.teErr ? b : a));
        const say = (s === 0 ? 'k = ' + k + ': each place takes the vote of its ' + (k === 1 ? 'single nearest point' : k + ' nearest points') + '. ' : 'k = ' + k + ': ') +
          'Training error ' + H.pct(trErr) + ', test error ' + H.pct(teErr) + ' on ' + te.X.length + ' fresh points.' +
          (k === 1 ? ' The map wraps every stray point in its own island.' : s === ks.length - 1 ? ' The smoothest map so far; the lowest test error was at k = ' + best.k + ' (' + H.pct(best.teErr) + ').' : k >= n / 2 ? ' The vote now spans much of the data, and the map drifts toward the majority class.' : ' The islands melt away as the vote widens.');
        F.add({ items: [...ax.items, field, ...S.Kit.scatter.points(ax, 'kk:p', X, y, { r: 4.2 }), ...legend, ...cax.items, L('ke:tr', (r) => r.trErr, 'faint'), L('ke:te', (r) => r.teErr, 'act'), ...dots, ...elegend, H.text('ke:now', lx.x, lx.y + lx.h + 84, 'k = ' + k + ':  train ' + H.pct(trErr) + ',  test ' + H.pct(teErr), 'strong', 14)].flat(), say, meta: { phase: 'k', k, trErr, teErr, nTest: te.X.length, bestK: best.k, bestErr: best.teErr, n } });
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'k nearest neighbours', text: ['N_k(x) = the k training points nearest x', 'p(c | x) = |{i ∈ N_k(x) : yᵢ = c}| / k', 'ŷ = argmax_c p(c | x)', 'small k: low bias, high variance', 'large k: smooth, high bias'].join('\n') }] } };
    },
    predict: [
      { at: (f, i) => i === 0, pick: 'option', ask: 'As k grows, will the training error go up or down?', options: ['mostly up', 'mostly down', 'stay at zero'], answer: 'mostly up', explain: 'With k = 1 every point votes for itself; more neighbours bring in votes from the other class.' },
      { at: (f, i, fr) => i === fr.length - 2, pick: 'option', ask: 'At the largest k, will the test error be lower than at the best k so far?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.teErr < fr[fr.length - 2].meta.bestErr ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Which k gives zero training error (no duplicate points)?', options: ['k = 1', 'k = n', 'the k with the lowest test error', 'any odd k'], answer: 0, why: 'Each point is its own nearest neighbour.' },
      { q: 'With k = n (all training points), what does k-NN predict everywhere?', options: ['the majority class of the training set', 'the class of the nearest point', 'a random class', 'nothing'], answer: 0, why: 'Every vote counts every point, wherever the query is.' },
      { q: 'Why choose an odd k for two classes?', options: ['so a uniform vote cannot tie', 'odd k is faster', 'even k overfits', 'it makes the boundary straight'], answer: 0, why: 'With two classes an even number of votes can split evenly.' },
      { q: 'How should k be chosen?', options: ['by validation or cross-validation error', 'as small as possible', 'by training error', 'always 5'], answer: 0, why: 'Training error always favours k = 1; held-out error shows the trade-off.' },
      { q: 'Larger k makes the model…', options: ['smoother, with more bias and less variance', 'wigglier', 'faster to train', 'more sensitive to single points'], answer: 0, why: 'It averages over more points, so single points matter less.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
