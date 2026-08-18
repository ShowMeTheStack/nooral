// Part 4, nearest neighbours: what "near" means (the balls of the Euclidean, Manhattan and Chebyshev distances and
// the neighbours each one picks), and how the neighbours' votes are weighted (one vote each, or 1 / distance).
(function (S) {
  'use strict';
  const H = S.PL, P = S.PL4, ML = () => S.ML;
  const n$ = H.n;
  const setInput = { id: 'set', kind: 'choice', label: 'data', def: 'blobs', options: S.PL4.SETS };
  const METRICS = [
    { id: 'euclidean', name: 'Euclidean', p: 2, cls: 'm2', formula: '‖x − z‖₂ = √(Σ (xⱼ − zⱼ)²)' },
    { id: 'manhattan', name: 'Manhattan', p: 1, cls: 'm1', formula: '‖x − z‖₁ = Σ |xⱼ − zⱼ|' },
    { id: 'chebyshev', name: 'Chebyshev', p: Infinity, cls: 'minf', formula: '‖x − z‖∞ = maxⱼ |xⱼ − zⱼ|' },
  ];
  /** the ball of radius r around c in a metric, as a path on the axes */
  const ball = (ax, key, c, r, mt, cls) => {
    let pts;
    if (mt.p === 2) pts = Array.from({ length: 73 }, (_, i) => { const a = (2 * Math.PI * i) / 72; return [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]; });
    else if (mt.p === 1) pts = [[c[0] + r, c[1]], [c[0], c[1] + r], [c[0] - r, c[1]], [c[0], c[1] - r]];
    else pts = [[c[0] + r, c[1] + r], [c[0] - r, c[1] + r], [c[0] - r, c[1] - r], [c[0] + r, c[1] - r]];
    const [x0, x1] = ax.o.xr, [y0, y1] = ax.o.yr, cl = (p) => [Math.max(x0, Math.min(x1, p[0])), Math.max(y0, Math.min(y1, p[1]))];
    return { key, kind: 'path', d: pts.map(cl).map((p, i) => (i ? 'L' : 'M') + H.r2(ax.X(p[0])) + ',' + H.r2(ax.Y(p[1]))).join(' ') + ' Z', cls: 'p4-ball ' + mt.cls + ' ' + (cls || ''), audit: 'ignore', layer: 'back', z: 4 };
  };
  const queryAt = (rng, inp) => [rng.xr[0] + inp.qx * (rng.xr[1] - rng.xr[0]), rng.yr[0] + inp.qy * (rng.yr[1] - rng.yr[0])].map((v) => S.Data.round(v, 2));
  const probe = (ax, q, cls) => ({ key: 'q', kind: 'dot', x: ax.X(q[0]), y: ax.Y(q[1]), r: 6.5, cls: 'pl-probe ' + (cls || ''), audit: 'overlap-ok', layer: 'items', z: 6 });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'knn-metrics', family: 'p04-knn', order: 30, title: 'Distance metrics: which points are near?',
    blurb: 'k-NN needs a distance. The Euclidean distance measures straight lines, the Manhattan distance adds the steps along each axis, and the Chebyshev distance keeps only the largest step. Their balls (all the places within a distance r) are a circle, a diamond and a square, so the same query can collect different neighbours, and a different vote.',
    inputs: [
      setInput,
      { id: 'n', kind: 'number', label: 'points', def: 40, min: 5, max: 120 },
      { id: 'noise', kind: 'slider', label: 'spread', def: 0.7, min: 0, max: 1, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 5, min: 1, max: 9999 },
      { id: 'k', kind: 'number', label: 'k', def: 6, min: 1, max: 20, check: (v, inp) => (v > inp.n ? 'k: at most the number of points' : null) },
      { id: 'qx', kind: 'slider', label: 'query across', def: 0.48, min: 0.02, max: 0.98, step: 0.01 },
      { id: 'qy', kind: 'slider', label: 'query up', def: 0.5, min: 0.02, max: 0.98, step: 0.01 },
    ],
    presets: [
      { label: 'Blobs, k = 6', values: { set: 'blobs', n: 40, noise: 0.7, seed: 5, k: 6, qx: 0.48, qy: 0.5 } },
      { label: 'Moons, k = 9', values: { set: 'moons', n: 80, noise: 0.5, seed: 3, k: 9, qx: 0.5, qy: 0.52 } },
      { label: 'Wildflowers, k = 5', values: { set: 'flowers', n: 60, noise: 0.5, seed: 2, k: 5, qx: 0.6, qy: 0.55 } },
      { label: 'Iris petals, k = 7', values: { set: 'iris', n: 90, noise: 0.5, seed: 1, k: 7, qx: 0.62, qy: 0.6 } },
      { label: 'k = 1', values: { set: 'blobs', n: 30, noise: 0.9, seed: 8, k: 1, qx: 0.4, qy: 0.42 } },
    ],
    counters: [{ id: 'm', label: 'metrics tried' }],
    run(inp) {
      const d = P.data(inp.set, inp.n, inp.seed, inp.noise), X = d.X, y = d.y, k = inp.k;
      const rng = P.square(X), ax = P.plot('mt', rng, [d.xname, d.yname]), q = queryAt(rng, inp);
      const res = METRICS.map((mt) => {
        const m = ML().KNeighborsClassifier({ k, metric: mt.id }).fit(X, y), ex = m.explain(q), p = m.predictProba([q])[0];
        return { mt, ex, idx: ex.idx, r: ex.dist[ex.dist.length - 1], votes: p.map((v) => Math.round(v * k)), cls: m.classes[ML().argmax(p)], classes: m.classes };
      });
      const F = new S.Frames(['m']), R = 520;
      const name = (c) => d.names[res[0].classes.indexOf(c)];
      // the unit balls, side by side in the panel
      const inset = () => {
        const it = [H.title('mt:ut', R, 300, 'balls of radius 1')], cx = R + 66, cy = 390, s = 56;
        it.push(H.seg('mt:ux', cx - s - 22, cy, cx + s + 22, cy, 'pl-seg thin', 'ignore'), H.seg('mt:uy', cx, cy - s - 22, cx, cy + s + 22, 'pl-seg thin', 'ignore'));
        const fake = { X: (u) => cx + u * s, Y: (v) => cy - v * s, o: { xr: [-2, 2], yr: [-2, 2] } };
        METRICS.forEach((mt, i) => it.push(Object.assign(ball(fake, 'mt:ub' + i, [0, 0], 1, mt, 'soft'), { layer: 'lines' })));
        S.Kit.curve.legend('mt:ul', R + 150, 370, METRICS.map((mt) => ({ text: mt.name, cls: 'p4' + mt.cls })), { dir: 'col' }).forEach((x) => it.push(x));
        return it;
      };
      const rows = (upto) => res.slice(0, upto + 1).map((o, i) => [o.mt.name + ': r = ' + n$(o.r) + ' → ' + name(o.cls), (i === upto ? 'act' : 'muted')]);
      const base = () => [...ax.items, ...H.classLegend('mt:lg', R, 40, d.names.slice(0, res[0].classes.length))];
      F.add({ items: [...base(), ...S.Kit.scatter.points(ax, 'mt:p', X, y, { r: 4.5 }), probe(ax, q), ...inset()], say: 'A query at (' + n$(q[0]) + ', ' + n$(q[1]) + ') and ' + X.length + ' training points. Its ' + H.plural(k, 'nearest neighbour') + ' depend on how distance is measured. The three balls of radius 1 on the right are a circle, a diamond and a square.', meta: { phase: 'start', q, k, n: X.length } });
      res.forEach((o, s) => {
        F.set('m', s + 1);
        const items = [...base()];
        res.slice(0, s).forEach((p, j) => items.push(ball(ax, 'mt:b' + j, q, p.r, p.mt, 'soft')));
        items.push(ball(ax, 'mt:b' + s, q, o.r, o.mt));
        items.push(...S.Kit.scatter.points(ax, 'mt:p', X, y, { r: 4.5, cls: (i) => (o.idx.includes(i) ? 'big' : '') }), probe(ax, q));
        items.push(...P.eqs('mt:eq', R, 92, [[o.mt.formula, 'act'], ...rows(s)], 14, 26), ...inset());
        const prev = s ? res[s - 1] : null, same = prev ? o.idx.filter((i) => prev.idx.includes(i)).length : 0;
        const tally = o.votes.map((v, c) => [v, c]).filter((p) => p[0] > 0).map((p) => p[0] + ' ' + d.names[p[1]]).join(', ');
        F.add({ items, say: o.mt.name + ': the ' + H.plural(k, 'nearest point') + ' all lie within ' + n$(o.r) + ' (the ' + (o.mt.p === 2 ? 'circle' : o.mt.p === 1 ? 'diamond' : 'square') + '). Their votes: ' + tally + ', so the query is ' + name(o.cls) + '.' + (prev ? ' It shares ' + same + ' of its ' + k + ' neighbours with the ' + prev.mt.name + ' set.' : ''), meta: { phase: 'metric', s, r: o.r, votes: o.votes, same, k } });
      });
      const all = res.map((o) => new Set(o.idx)), common = [...all[0]].filter((i) => all[1].has(i) && all[2].has(i)).length;
      const changed = new Set(res.map((o) => o.cls)).size > 1;
      const items = [...base(), ...res.map((o, j) => ball(ax, 'mt:b' + j, q, o.r, o.mt, j === 0 ? '' : 'soft')), ...S.Kit.scatter.points(ax, 'mt:p', X, y, { r: 4.5, cls: (i) => (all.every((a) => a.has(i)) ? 'big' : all.some((a) => a.has(i)) ? 'act' : '') }), probe(ax, q), ...P.eqs('mt:eq', R, 92, [['in all three sets: ' + common + ' of ' + k, 'good'], ...rows(2).map((r) => [r[0], ''])], 14, 26), ...inset()];
      F.add({ items, say: common + ' of the ' + k + ' neighbours are picked by all three distances; the others (ringed in colour) depend on the metric. ' + (changed ? 'Here the change is enough to change the vote.' : 'Here every metric still gives ' + name(res[0].cls) + '.') + ' Scaling the features changes the balls too, so k-NN usually standardises them first.', meta: { phase: 'compare', common, k } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Minkowski distances', text: ['‖x − z‖ₚ = (Σ |xⱼ − zⱼ|ᵖ)^(1/p)', 'p = 1: Manhattan (a diamond ball)', 'p = 2: Euclidean (a circle)', 'p → ∞: Chebyshev, max |xⱼ − zⱼ| (a square)', '‖·‖∞ ≤ ‖·‖₂ ≤ ‖·‖₁'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'metric' && f.meta.s === 0, pick: 'option', ask: 'Will the Manhattan distance pick exactly the same neighbours?', options: ['yes, all the same', 'no, some change'], answer: (fr) => (fr.find((f) => f.meta.phase === 'metric' && f.meta.s === 1).meta.same === fr[0].meta.k ? 'yes, all the same' : 'no, some change'), explain: (fr) => fr.find((f) => f.meta.phase === 'metric' && f.meta.s === 1).say },
    ],
    quiz: [
      { q: 'What is the Manhattan distance between (1, 2) and (4, 6)?', options: ['7', '5', '4', '25'], answer: 0, why: '|4 − 1| + |6 − 2| = 3 + 4.' },
      { q: 'And the Chebyshev distance between the same points?', options: ['4', '3', '5', '7'], answer: 0, why: 'The larger of the two steps, 3 and 4.' },
      { q: 'Which shape is the set of points within Manhattan distance 1 of the origin?', options: ['a diamond (a square turned 45°)', 'a circle', 'an axis-aligned square', 'a line'], answer: 0, why: '|x| + |y| ≤ 1 has corners at (±1, 0) and (0, ±1).' },
      { q: 'Why standardise features before k-NN?', options: ['a feature with a large range would dominate every distance', 'it makes training faster', 'k-NN needs a mean of zero', 'it removes outliers'], answer: 0, why: 'Distances add up the differences in every feature, measured in that feature’s units.' },
      { q: 'For the same two points, which is never larger than the Euclidean distance?', options: ['Chebyshev', 'Manhattan', 'neither', 'both'], answer: 0, why: 'The largest step alone is at most the straight-line length, which is at most the sum of the steps.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'knn-weights', family: 'p04-knn', order: 40, title: 'Weighting the votes',
    blurb: 'With uniform weights each of the k neighbours casts one vote. With distance weights a neighbour’s vote is 1 / distance, so a point right next to the query outweighs several far ones. The class map changes most where the classes meet, and a training point (at distance 0) always gets its own label back.',
    inputs: [
      setInput,
      { id: 'n', kind: 'number', label: 'points', def: 40, min: 5, max: 120 },
      { id: 'noise', kind: 'slider', label: 'spread', def: 0.8, min: 0, max: 1, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 6, min: 1, max: 9999 },
      { id: 'k', kind: 'number', label: 'k', def: 7, min: 1, max: 15, check: (v, inp) => (v > inp.n ? 'k: at most the number of points' : null) },
      { id: 'qx', kind: 'slider', label: 'query across', def: 0.5, min: 0.02, max: 0.98, step: 0.01 },
      { id: 'qy', kind: 'slider', label: 'query up', def: 0.45, min: 0.02, max: 0.98, step: 0.01 },
    ],
    presets: [
      { label: 'Blobs, k = 7', values: { set: 'blobs', n: 40, noise: 0.8, seed: 6, k: 7, qx: 0.5, qy: 0.45 } },
      { label: 'Moons, k = 9', values: { set: 'moons', n: 60, noise: 0.6, seed: 4, k: 9, qx: 0.47, qy: 0.5 } },
      { label: 'Wildflowers, k = 11', values: { set: 'flowers', n: 60, noise: 0.5, seed: 3, k: 11, qx: 0.66, qy: 0.62 } },
      { label: 'Ring and disc, k = 5', values: { set: 'circles', n: 50, noise: 0.6, seed: 2, k: 5, qx: 0.3, qy: 0.5 } },
    ],
    counters: [{ id: 'votes', label: 'votes counted' }],
    run(inp) {
      const d = P.data(inp.set, inp.n, inp.seed, inp.noise), X = d.X, y = d.y, k = inp.k;
      const rng = P.square(X), ax = P.plot('wt', rng, [d.xname, d.yname]), q = queryAt(rng, inp);
      const mu = ML().KNeighborsClassifier({ k }).fit(X, y), md = ML().KNeighborsClassifier({ k, weights: 'distance' }).fit(X, y);
      const eu = mu.explain(q), ed = md.explain(q), K = mu.classes.length, ci = (lab) => mu.classes.indexOf(lab);
      const pu = mu.predictProba([q])[0], pd = md.predictProba([q])[0];
      const totU = new Array(K).fill(0), totD = new Array(K).fill(0);
      eu.idx.forEach((i, t) => { totU[ci(y[i])] += eu.weights[t]; totD[ci(y[i])] += ed.weights[t]; });
      const cu = ML().argmax(pu), cd = ML().argmax(pd);
      const accD = ML().accuracy(y, md.predict(X)), accU = ML().accuracy(y, mu.predict(X));
      const F = new S.Frames(['votes']), R = 520;
      const field = (m, key) => P.regions(ax, key, (u, v) => { const p = m.predictProba([[u, v]])[0], c = ML().argmax(p); return [c, Math.min(1, (p[c] - 1 / K) * 0.9 + 0.3)]; }, { n: 60 });
      const rays = () => eu.idx.map((i) => H.seg('wt:r' + i, ax.X(q[0]), ax.Y(q[1]), ax.X(X[i][0]), ax.Y(X[i][1]), 'pl-seg thin c' + ci(y[i]), 'ignore'));
      const pts = () => S.Kit.scatter.points(ax, 'wt:p', X, y, { r: 4.5, cls: (i) => (eu.idx.includes(i) ? 'big' : '') });
      // the table of neighbours: class, distance, weight
      const rowH = Math.min(24, 250 / k);
      const table = (showW) => {
        const it = [H.text('wt:h0', R, 92, '#', 'tick', 12), H.text('wt:h1', R + 70, 92, 'distance', 'tick', 12, 'end'), H.text('wt:h2', R + 160, 92, showW ? 'weight 1/d' : 'weight', 'tick', 12, 'end')];
        eu.idx.forEach((i, t) => {
          const yy = 92 + (t + 1) * rowH;
          it.push({ key: 'wt:dot' + t, kind: 'dot', x: R + 22, y: yy - 4, r: 5, cls: 'swatch c' + ci(y[i]) });
          it.push(H.text('wt:n' + t, R, yy, String(t + 1), 'tick', 12));
          it.push(H.text('wt:d' + t, R + 70, yy, n$(eu.dist[t]), 'p4-num', 13, 'end'));
          it.push(H.text('wt:w' + t, R + 160, yy, showW ? n$(ed.weights[t]) : '1', 'p4-num c' + ci(y[i]), 13, 'end'));
        });
        return it;
      };
      const bars = (tot, win, label) => {
        const y0 = 92 + (k + 1) * rowH + 26, max = Math.max(...tot), it = [H.text('wt:bt', R, y0, label, 'title', 13)];
        tot.forEach((v, c) => {
          const yy = y0 + 14 + c * 24, w = max > 0 ? (v / max) * 150 : 0;
          it.push(H.text('wt:bn' + c, R + 60, yy + 13, d.names[c], 'pl-lab c' + c, 13, 'end'));
          if (w > 0.5) it.push({ key: 'wt:b' + c, kind: 'cell', x: R + 68, y: yy, w: H.r2(w), h: 16, rx: 2, text: '', cls: 'pl-pbar c' + c, audit: 'ignore' });
          it.push(H.text('wt:bv' + c, R + 74 + w, yy + 13, n$(v) + (c === win ? '  wins' : ''), c === win ? 'pl-lab act' : 'tick', 12));
        });
        return it;
      };
      const base = () => [...ax.items, ...H.classLegend('wt:lg', R, 40, d.names.slice(0, K))];
      F.add({ items: [...base(), ...rays(), ...pts(), probe(ax, q), ...table(false)], say: 'The ' + H.plural(k, 'nearest neighbour') + ' of the query (' + n$(q[0]) + ', ' + n$(q[1]) + ') lie between ' + n$(eu.dist[0]) + ' and ' + n$(eu.dist[k - 1]) + ' away.', meta: { phase: 'neighbours', q, k, d0: eu.dist[0], dk: eu.dist[k - 1] } });
      F.set('votes', k);
      F.add({ items: [...base(), field(mu, 'wt:f'), ...rays(), ...pts(), probe(ax, q), ...table(false), ...bars(totU, cu, 'one vote each')], say: 'Uniform weights: one vote each, ' + totU.map((v, c) => [v, c]).filter((p) => p[0]).map((p) => p[0] + ' for ' + d.names[p[1]]).join(' and ') + '. The query is ' + d.names[cu] + ' with probability ' + n$(pu[cu]) + '. The shading is this rule everywhere; on the training points it is right ' + H.pct(accU) + ' of the time.', meta: { phase: 'uniform', totU, p: pu[cu], accU } });
      const top = ML().argmax(ed.weights);
      F.add({ items: [...base(), field(md, 'wt:f'), ...rays(), ...pts(), probe(ax, q), ...table(true), ...bars(totD, cd, 'votes weighted by 1/d')], say: 'Distance weights: the nearest neighbour, ' + n$(ed.dist[top]) + ' away, now weighs ' + n$(ed.weights[top]) + ' and the farthest only ' + n$(ed.weights[k - 1]) + '. The weighted votes give ' + totD.map((v, c) => [v, c]).filter((p) => p[0]).map((p) => n$(p[0]) + ' for ' + d.names[p[1]]).join(' and ') + ', so the query is ' + d.names[cd] + ' (probability ' + n$(pd[cd]) + ')' + (cd !== cu ? ': the weighting changed the answer.' : ', as before.'), meta: { phase: 'distance', totD, p: pd[cd], w0: ed.weights[top], d0: ed.dist[top], wk: ed.weights[k - 1] } });
      F.add({ items: [...base(), field(md, 'wt:f'), ...pts(), probe(ax, q), ...table(true), ...bars(totD, cd, 'votes weighted by 1/d')], say: 'Each training point is at distance 0 from itself and then takes the whole vote, so with distance weights the training accuracy is ' + H.pct(accD) + ' against ' + H.pct(accU) + ' with uniform weights: like k = 1 it can no longer show overfitting on the training set.', meta: { phase: 'train', accD, accU } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Weighted votes', text: ['uniform: wᵢ = 1', 'distance: wᵢ = 1 / d(x, xᵢ)', '  (a neighbour at d = 0 takes the whole vote)', 'p(c | x) = Σ_{i: yᵢ = c} wᵢ / Σ wᵢ'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'uniform', pick: 'option', ask: 'With votes weighted by 1 / distance, will the answer change?', options: ['yes', 'no'], answer: (fr) => (/changed the answer/.test(fr.find((f) => f.meta.phase === 'distance').say) ? 'yes' : 'no'), explain: (fr) => fr.find((f) => f.meta.phase === 'distance').say },
    ],
    quiz: [
      { q: 'Neighbours at distances 1, 2 and 4 vote A, B, B. With 1/d weights, who wins?', options: ['A (1 against 0.75)', 'B (2 votes against 1)', 'a tie', 'it depends on k'], answer: 0, why: 'A gets 1/1 = 1, B gets 1/2 + 1/4 = 0.75.' },
      { q: 'What happens with distance weights when the query is exactly a training point?', options: ['that point decides alone', 'its vote is ignored', 'all votes are equal', 'the prediction is undefined'], answer: 0, why: 'Its weight 1/0 is treated as all of the vote.' },
      { q: 'Distance weighting mostly changes predictions…', options: ['near the boundary between classes', 'deep inside one class', 'nowhere', 'only far from the data'], answer: 0, why: 'Inside a class all neighbours agree whatever their weights.' },
      { q: 'With distance weights, the choice of k matters…', options: ['less, because far neighbours count little', 'more', 'not at all', 'only for k = 1'], answer: 0, why: 'Adding a far neighbour adds a small weight.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
