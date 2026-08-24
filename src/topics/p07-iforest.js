// Part 7, anomaly detection with an isolation forest: one random tree's cuts closing in on an odd point and on an
// ordinary one (the odd one is alone after a few cuts), the path lengths averaged over the forest and turned into
// scores; then the score over the whole plane, the points flagged at a contamination level, and a real set of
// pump readings with known faults.
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const r2 = (v) => Math.round(v * 100) / 100;
  P.OUT = {
    cloud: { label: 'one cloud and a few odd points', make: (seed) => { const d = S.Data.blobs({ n: 54, centers: [[0, 0]], std: 0.9, seed }), r = S.Data.rng(seed + 3), X = d.X.slice(); for (let i = 0; i < 6; i++) { const a = r.range(0, 2 * Math.PI), rad = r.range(3, 4.2); X.push([+(rad * Math.cos(a)).toFixed(4), +(rad * Math.sin(a)).toFixed(4)]); } return { X, truth: X.map((_, i) => (i >= 54 ? 1 : 0)), names: ['x₁', 'x₂'] }; } },
    two: { label: 'two clouds and a few odd points', make: (seed) => { const d = S.Data.blobs({ n: 54, centers: [[-2, 0], [2, 0.5]], std: 0.6, seed }), r = S.Data.rng(seed + 5), X = d.X.slice(); for (let i = 0; i < 6; i++) X.push([+r.range(-4, 4).toFixed(4), +r.range(2, 3.5).toFixed(4) * (i % 2 ? 1 : -1)]); return { X, truth: X.map((_, i) => (i >= 54 ? 1 : 0)), names: ['x₁', 'x₂'] }; } },
    pumps: { label: 'pumps: temperature and vibration (300 minutes)', make: () => { const d = S.Data.load('pumps'), X = [], truth = []; d.X.forEach((x, i) => { if (X.length < 300 && isFinite(x[0]) && isFinite(x[1]) && x[0] !== null && x[1] !== null) { X.push([x[0], x[1]]); truth.push(d.y[i]); } }); return { X, truth, names: ['temperature (°C)', 'vibration'], real: true }; } },
  };
  /** the path of x down a tree: the nodes passed, with the box each cut leaves and the sample rows still there */
  P.isoPath = function (tree, X, x, box0) {
    const steps = [];
    let node = tree, box = box0.map((r) => r.slice()), rows = X.map((_, i) => i), depth = 0;
    while (node.left) {
      const j = node.feature, t = node.threshold, goLeft = x[j] < t, nb = box.map((r) => r.slice());
      if (goLeft) nb[j][1] = t; else nb[j][0] = t;
      const keep = rows.filter((i) => (X[i][j] < t) === goLeft);
      steps.push({ feature: j, threshold: t, boxBefore: box, box: nb, before: rows.length, after: keep.length, depth: depth + 1 });
      box = nb; rows = keep; node = goLeft ? node.left : node.right; depth++;
    }
    return { steps, size: node.size, depth, h: depth + ML().isoC(node.size) };
  };

  S.topic({
    id: 'iforest-cuts', family: 'p07-anomaly', order: 10, title: 'Isolation forest: odd points are easy to cut off',
    blurb: 'Pick a feature at random and cut it at a random value between its smallest and largest; repeat in each part until every point is alone (or a depth limit is reached). An odd point, far from the rest, tends to be alone after a few cuts; a point in a crowd needs many. Average the number of cuts over many random trees, compare with what a random point would need, and you have an anomaly score that needs no distances and no labels.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'cloud', options: [{ value: 'cloud', label: P.OUT.cloud.label }, { value: 'two', label: P.OUT.two.label }] },
      { id: 'seed', kind: 'number', label: 'Seed (the forest)', def: 1, min: 1, max: 999 },
    ],
    presets: [{ label: 'One cloud', values: { set: 'cloud', seed: 1 } }, { label: 'Another forest', values: { set: 'cloud', seed: 7 } }, { label: 'Two clouds', values: { set: 'two', seed: 2 } }],
    counters: [{ id: 'cuts', label: 'cuts' }],
    run(inp) {
      const d = P.OUT[inp.set].make(2), X = d.X, n = X.length, F = H().frames(['cuts']), q = F.q;
      const forest = new (ML().IsolationForest)({ nEstimators: 100, seed: inp.seed }).fit(X), tree = forest.trees[0];
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('if', X, box, { pad: 0.04 });
      const lo = [0, 1].map((j) => Math.min(...X.map((x) => x[j]))), hi = [0, 1].map((j) => Math.max(...X.map((x) => x[j]))), box0 = [[lo[0], hi[0]], [lo[1], hi[1]]];
      const H2 = forest.pathLengths(X), odd = H2.indexOf(Math.min(...H2));
      const mean = ML().colMeans(X), normal = X.map((x, i) => [ML().sqdist(x, mean), i]).sort((a, b) => a[0] - b[0])[0][1];
      const cell = (key, b, cls) => ({ key, kind: 'path', d: `M${ax.X(b[0][0])},${ax.Y(b[1][0])} H${ax.X(b[0][1])} V${ax.Y(b[1][1])} H${ax.X(b[0][0])} Z`, cls, audit: 'ignore', layer: 'back', z: 3 });
      const cutLine = (key, st, cls) => ({ key, kind: 'path', d: st.feature === 0 ? `M${ax.X(st.threshold)},${ax.Y(st.boxBefore[1][0])} V${ax.Y(st.boxBefore[1][1])}` : `M${ax.X(st.boxBefore[0][0])},${ax.Y(st.threshold)} H${ax.X(st.boxBefore[0][1])}`, cls, audit: 'ignore', layer: 'lines', z: 4 });
      const pts = (hot) => P.points(ax, 'if:p', X, X.map((_, i) => (i === hot ? 1 : -1)), { r: (i) => (i === hot ? 6.5 : 4.2) });
      const c = ML().isoC(forest.maxSamples), score = (h) => Math.pow(2, -h / c);
      F.push([...ax.items, ...pts(-1), S.Kit.scatter.ring(ax, 'if:ro', X[odd][0], X[odd][1], 'act', 11), S.Kit.scatter.ring(ax, 'if:rn', X[normal][0], X[normal][1], 'hot', 11), H().head('if:h', 520, 60, 'Two points to isolate'), H().text('if:h1', 520, 86, 'pink ring: an odd point', 'legend', 13), H().text('if:h2', 520, 110, 'yellow ring: one in the crowd', 'legend', 13)],
        `${P.qpl(q, n, 'point')}. Follow two of them through one random tree: point ${q.n(odd)} (pink ring), off on its own, and point ${q.n(normal)} (yellow ring), in the middle of the crowd. Each tree is grown on a sample of ${q.n(forest.maxSamples)} points and stops at depth ${q.n(Math.ceil(Math.log2(Math.max(forest.maxSamples, 2))))}.`, { phase: 'start', odd, normal, n, m: forest.maxSamples, maxDepth: Math.ceil(Math.log2(Math.max(forest.maxSamples, 2))) });
      const runPath = (who, label, cls) => {
        const p = P.isoPath(tree, X, X[who], box0);
        F.set('cuts', 0);
        p.steps.forEach((st, k) => {
          F.count('cuts');
          const items = [...ax.items, cell('if:cell', st.box, 'p7box'), ...p.steps.slice(0, k).map((s2, u) => cutLine('if:old' + u, s2, 'p7split old')), cutLine('if:cut', st, 'p7split'), ...pts(who)];
          items.push(H().head('if:h', 520, 60, label + ': cut ' + (k + 1)), H().text('if:h1', 520, 86, (st.feature === 0 ? d.names[0] : d.names[1]) + ' at ' + H().fx(st.threshold, 2), 'legend', 13), H().text('if:h2', 520, 110, st.after + ' of ' + st.before + ' points on its side', 'legend', 13));
          const last = k === p.steps.length - 1;
          F.push(items, `Cut ${q.n(k + 1)} for point ${q.n(who)}: ${st.feature === 0 ? 'x₁' : 'x₂'} at ${q(st.threshold, 2)}, a random value within the shaded cell’s range. The point’s side keeps ${P.qpl(q, st.after, 'point')} of ${q.n(st.before)}.` + (last ? (p.size <= 1 ? ` It is alone: isolated after ${P.qpl(q, p.depth, 'cut')}.` : ` The depth limit is reached with ${P.qpl(q, p.size, 'point')} still together, so ${q(ML().isoC(p.size), 2)} is added for the cuts they would still need: path length ${q(p.h, 2)}.`) : ''), { phase: cls, who, k, cut: k + 1, threshold: st.threshold, feature: st.feature, before: st.before, after: st.after, depth: p.depth, size: p.size, h: p.h, extra: ML().isoC(p.size) });
        });
        return p;
      };
      const po = runPath(odd, 'Odd point', 'odd'), pn = runPath(normal, 'Crowded point', 'normal');
      const hO = H2[odd], hN = H2[normal], sO = score(hO), sN = score(hN);
      const items = [...ax.items, ...pts(-1), S.Kit.scatter.ring(ax, 'if:ro', X[odd][0], X[odd][1], 'act', 11), S.Kit.scatter.ring(ax, 'if:rn', X[normal][0], X[normal][1], 'hot', 11)];
      items.push(H().head('if:h', 520, 60, 'Over all 100 trees'), ...H().lines('if:l', 520, 90, [['odd point: mean path ' + H().fx(hO, 2) + ', score ' + H().fx(sO, 2), 'legend'], ['crowded point: mean path ' + H().fx(hN, 2) + ', score ' + H().fx(sN, 2), 'legend'], ['a random point needs c = ' + H().fx(c, 2), 'legend'], ['score = 2^(−path / c)', 'axname']], 13, 24));
      F.push(items, `Over all ${q.n(forest.trees.length)} trees, point ${q.n(odd)} needs ${q(hO, 2)} cuts on average and point ${q.n(normal)} needs ${q(hN, 2)}. A search in a random tree of ${q.n(forest.maxSamples)} points takes c = ${q(c, 2)} on average, so the scores 2^(−path / c) are ${q(sO, 3)} and ${q(sN, 3)}: close to one is an anomaly, around one half or below is ordinary.`, { phase: 'scores', hO, hN, sO, sN, c, trees: forest.trees.length, m: forest.maxSamples, odd, normal, pathOdd: po.h, pathNormal: pn.h });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Isolation forest', text: ['a tree: random feature, random cut in its range,', '  until each point is alone or the depth limit', 'h(x) = cuts to reach x’s leaf + c(points left there)', 'c(m) = 2(ln(m − 1) + 0.5772) − 2(m − 1)/m', 'score(x) = 2^(−mean h(x) / c(sample size))'].join('\n') }] } };
    },
    predict: [{ at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which point will this tree cut off in fewer cuts?', options: ['the odd point (pink)', 'the crowded point (yellow)', 'both the same'], answer: (fr) => { const o = fr.filter((f) => f.meta.phase === 'odd'), nn = fr.filter((f) => f.meta.phase === 'normal'); const a = o[o.length - 1].meta.h, b = nn[nn.length - 1].meta.h; return a < b ? 'the odd point (pink)' : a > b ? 'the crowded point (yellow)' : 'both the same'; }, explain: (fr) => fr[fr.length - 1].say }],
    quiz: [
      { q: 'Why are anomalies isolated in fewer cuts?', options: ['a random cut is likely to fall between a lone point and the rest', 'they have larger values', 'the trees are built around them', 'they are cut first by design'], answer: 0, why: 'A point far from others leaves a wide gap that a uniform random cut often hits.' },
      { q: 'Mean path 2 when c = 8. The score 2^(−2/8) is about…', options: ['0.84: likely an anomaly', '0.25', '0.5: ordinary', '1.19'], answer: 0, why: '2^(−0.25) ≈ 0.84, close to 1.' },
      { q: 'Does an isolation forest need labelled anomalies?', options: ['no', 'yes, to choose the cuts', 'yes, for the depth limit', 'only for the score'], answer: 0, why: 'The trees are random; the score comes from path lengths alone.' },
      { q: 'Why grow each tree on a small sample?', options: ['crowds are thinner, so anomalies stand out, and it is fast', 'to use less colour', 'it changes nothing', 'to make every path the same'], answer: 0, why: 'Sub-sampling reduces swamping and masking by dense clusters, and trees stay shallow.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'iforest-scores', family: 'p07-anomaly', order: 20, title: 'Anomaly scores over the plane',
    blurb: 'The average path length can be computed for any place, not only the data points, so the forest paints an anomaly score over the whole plane: low in the crowds, high far from them. Choose a contamination, the share of points you expect to be anomalies, and the forest flags that many of the highest scores. On pump readings with known faults we can see what it catches.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'pumps', options: Object.keys(P.OUT).map((k) => ({ value: k, label: P.OUT[k].label })) },
      { id: 'cont', kind: 'slider', label: 'Contamination', def: 0.05, min: 0.01, max: 0.2, step: 0.01, fmt: (v) => Math.round(v * 100) + '%' },
      { id: 'trees', kind: 'number', label: 'Trees', def: 100, min: 1, max: 200 },
    ],
    presets: [
      { label: 'Pumps, 5%', values: { set: 'pumps', cont: 0.05, trees: 100 } },
      { label: 'Pumps, 10%', values: { set: 'pumps', cont: 0.1, trees: 100 } },
      { label: 'Two clouds, 10%', values: { set: 'two', cont: 0.1, trees: 100 } },
      { label: 'Only 3 trees: a noisy score', values: { set: 'cloud', cont: 0.1, trees: 3 } },
    ],
    counters: [{ id: 'flagged', label: 'flagged' }],
    run(inp) {
      const d = P.OUT[inp.set].make(2), X = d.X, n = X.length, F = H().frames(['flagged']), q = F.q;
      const forest = new (ML().IsolationForest)({ nEstimators: inp.trees, seed: 1, contamination: inp.cont }).fit(X);
      const s = forest.scoreSamples(X).map((v) => -v), thr = -forest.offset, flagged = s.map((v) => v > thr);
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('is', X, box, { equal: !d.real, pad: 0.12, xname: d.names[0], yname: d.names[1] });
      ax.items = ax.items.filter((it) => !/:grid$/.test(it.key));
      // score every place the pictures need in one batch: list them with a dry run, then look them up
      const want = [], rec = (u, v) => { want.push([u, v]); return 0; };
      S.Kit.scatter.field(ax, 'dry', rec, { n: 40, scheme: 'seq' }); S.Kit.scatter.grid(ax, rec, 30);
      const got = new Map(), vals = forest.scoreSamples(want);
      want.forEach((p, i) => got.set(p[0] + ',' + p[1], -vals[i]));
      const at = (u, v) => got.get(u + ',' + v);
      const field = S.Kit.scatter.field(ax, 'is:f', (u, v) => at(u, v) - 0.35, { n: 40, scheme: 'seq', max: 0.4, smooth: true });
      const g = S.Kit.scatter.grid(ax, at, 30);
      const level = S.Kit.scatter.level(ax, 'is:lv', g, thr, 'margin');
      const legend = (rows) => [H().text('is:bh', 520, 280, 'score', 'axname', 12), ...S.Kit.heat.bar({ prefix: 'is:bar', x: 520, y: 290, w: 220, h: 12, scheme: 'seq', max: 0.4, min: 0, fmt: (v) => H().fx(v + 0.35, 2) }), ...H().lines('is:l', 520, 60, rows, 13, 24)].filter(Boolean);
      const top = s.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
      F.push([field, ...ax.items, ...P.points(ax, 'is:p', X, null, { r: 4 }), ...legend([['Score over the plane', 'strong'], ['from ' + inp.trees + (inp.trees === 1 ? ' tree' : ' trees'), 'legend']])],
        `The score 2^(−mean path / c) of every place in the plane, from ${P.qpl(q, inp.trees, 'tree')}: dark where a point would be cut off quickly. The crowds sit in pale valleys. The highest-scoring point, ${q.n(top[0][1])}, scores ${q(top[0][0], 3)}.` + (inp.trees < 10 ? ' With so few trees the map is blocky and noisy: every boundary is one random cut.' : ''), { phase: 'field', top: top[0][1], topScore: top[0][0], trees: inp.trees });
      const nf = flagged.filter(Boolean).length;
      F.set('flagged', nf);
      const items = [field, level, ...ax.items, ...P.points(ax, 'is:p', X, flagged.map((f) => (f ? 1 : -1)), { r: (i) => (flagged[i] ? 5.5 : 4) }), ...legend([['Contamination ' + H().pct(inp.cont), 'strong'], ['threshold ' + H().fx(thr, 3) + ' (dashed)', 'legend'], [nf + ' points flagged (orange)', 'legend']])];
      F.push(items, `With contamination ${q.pct(inp.cont)}, the threshold is set at the score that ${q.pct(inp.cont)} of the points exceed: ${q(thr, 3)}, the dashed line. ${P.qpl(q, nf, 'point')} above it ${nf === 1 ? 'is' : 'are'} flagged.`, { phase: 'flag', thr, nf, cont: inp.cont });
      const tp = flagged.filter((f, i) => f && d.truth[i]).length, pos = d.truth.filter(Boolean).length;
      const it2 = [field, level, ...ax.items, ...P.points(ax, 'is:p', X, X.map((_, i) => (flagged[i] ? (d.truth[i] ? 2 : 1) : d.truth[i] ? 4 : -1)), { r: (i) => (flagged[i] || d.truth[i] ? 5.5 : 4) })];
      it2.push(...legend([[d.real ? 'Against the known faults' : 'Against the planted odd points', 'strong']]));
      it2.push(...P.row('is:k0', 520, 96, 2, 'flagged and truly odd'), ...P.row('is:k1', 520, 120, 1, 'flagged, but ordinary'), ...P.row('is:k2', 520, 144, 4, 'missed'));
      F.push(it2, `${d.real ? 'The pump readings come with the faults marked' : 'The odd points were planted, so we know them'}: ${q.n(pos)} in all. Of the ${q.n(nf)} flagged, ${q.n(tp)} are truly odd (green), so the precision is ${q.pct(nf ? tp / nf : 0)} and ${q.pct(pos ? tp / pos : 0)} of the odd points are caught.` + (d.real ? ' Pressure-drop faults barely change temperature or vibration, so these two readings cannot reveal them.' : ''), { phase: 'truth', tp, pos, nf, prec: nf ? tp / nf : 0, rec: pos ? tp / pos : 0 });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Scores and flags', text: ['score(x) = 2^(−E[h(x)] / c(m))', 'near 1: anomaly; about 0.5 or below: normal', 'contamination q: flag the q·n highest scores', 'precision = flagged and odd / flagged', 'recall = flagged and odd / odd'].join('\n') }] } };
    },
    predict: [{ at: (f) => f.meta.phase === 'field', pick: 'option', ask: 'Raising the contamination will flag…', options: ['more points', 'fewer points', 'the same points'], answer: 'more points', explain: (fr, i) => fr[i + 1].say }],
    quiz: [
      { q: 'With 400 points and contamination 5%, about how many are flagged?', options: ['20', '5', '40', '80'], answer: 0, why: '0.05 × 400.' },
      { q: 'Why can an isolation forest score places where there are no points?', options: ['a path down each tree can be followed for any x', 'it interpolates between points', 'it stores a grid', 'it cannot'], answer: 0, why: 'Each tree is a set of cuts; any x lands in some leaf.' },
      { q: 'Why does a forest with very few trees give a blocky map?', options: ['each tree’s cuts are random, and few trees average out little of that', 'the depth limit is lower', 'the score is not defined', 'contamination is ignored'], answer: 0, why: 'The average over many trees smooths the random cut positions.' },
      { q: 'Some faults change neither of the two features plotted. The forest…', options: ['cannot find them from these features', 'finds them anyway', 'finds them with more trees', 'flags all points'], answer: 0, why: 'Scores depend only on the features given; an anomaly invisible in them is missed.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
