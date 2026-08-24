// Part 7, DBSCAN: core points (at least minPts points within ε, counting themselves), clusters grown from core
// to core through their ε-neighbourhoods, border points reached but not core, and noise; then how ε and minPts
// change the answer, with the k-distance curve that suggests an ε.
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;
  const r2 = (v) => Math.round(v * 100) / 100;
  /** circles of radius rad (data units) round some points, as one path */
  P.circles = (ax, key, pts, rad, cls) => {
    const R = r2(ax.X(rad) - ax.X(0));
    return { key, kind: 'path', d: pts.map((p) => `M${r2(ax.X(p[0]) - R)},${ax.Y(p[1])} a${R},${R} 0 1,0 ${2 * R},0 a${R},${R} 0 1,0 ${-2 * R},0`).join(' ') || 'M0,0', cls, audit: 'ignore', layer: 'back', z: 3 };
  };
  const kindCls = (kinds, labels) => (i) => (kinds[i] === 'noise' ? 'cross' : kinds[i] === 'border' ? 'hollow' : '');
  const labOf = (m) => m.labels.map((l) => (l < 0 ? -1 : l % 6));
  /** the distance from each point to its k-th nearest point, itself counted first */
  P.kdist = (X, k) => X.map((a) => X.map((b) => Math.sqrt(ML().sqdist(a, b))).sort((u, v) => u - v)[Math.min(k, X.length) - 1]);

  S.topic({
    id: 'dbscan-core', family: 'p07-dbscan', order: 10, title: 'DBSCAN: core, border and noise',
    blurb: 'DBSCAN finds clusters as crowded regions. A point is a core point if at least minPts points (itself included) lie within distance ε of it. Clusters grow from core points: everything within ε of a core point joins its cluster, and the growth carries on through every core point it reaches. Points reached but not core are border points; points no core point reaches are noise. No k is needed, and clusters can take any shape.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'noisy', options: P.setOptions(['moons', 'noisy', 'circles', 'three', 'uneven']) },
      { id: 'eps', kind: 'slider', label: 'ε (radius)', def: 0.6, min: 0.1, max: 1.5, step: 0.05 },
      { id: 'minPts', kind: 'number', label: 'minPts', def: 5, min: 2, max: 20 },
    ],
    presets: [
      { label: 'Two moons', values: { set: 'moons', eps: 0.5, minPts: 5 } },
      { label: 'Groups in noise', values: { set: 'noisy', eps: 0.6, minPts: 5 } },
      { label: 'A ring round a disc', values: { set: 'circles', eps: 0.7, minPts: 4 } },
      { label: 'ε too small: mostly noise', values: { set: 'moons', eps: 0.25, minPts: 5 } },
      { label: 'ε too big: one cluster', values: { set: 'noisy', eps: 1.2, minPts: 5 } },
    ],
    counters: [{ id: 'core', label: 'core points' }, { id: 'clusters', label: 'clusters' }],
    run(inp) {
      const d = P.data(inp.set, {}), X = d.X, n = X.length, F = H().frames(['core', 'clusters']), q = F.q, eps = inp.eps, mp = inp.minPts;
      const m = new (ML().DBSCAN)({ eps, minSamples: mp }).fit(X), core = m.core, kinds = m.kinds;
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('db', X, box);
      const nb = (i) => X.map((_, j) => j).filter((j) => Math.sqrt(ML().sqdist(X[i], X[j])) <= eps);
      // the example: a border point if there is one, else a noise point, else a core point
      const ex = [kinds.indexOf('border'), kinds.indexOf('noise'), 0].find((i) => i >= 0), exN = nb(ex);
      const cores = X.map((_, i) => i).filter((i) => core[i]), nCore = cores.length;
      const legendAt = (it, y) => it.push(H().head('db:h', 520, y, 'ε = ' + H().fx(eps, 2) + ', minPts = ' + mp));
      const it0 = [...ax.items, ...P.points(ax, 'db:p', X, X.map((_, j) => (exN.includes(j) && j !== ex ? 0 : -1)), { r: (j) => (j === ex ? 6.5 : 4.4) }), P.circles(ax, 'db:ex', [X[ex]], eps, 'p7eps')];
      legendAt(it0, 60);
      F.push(it0, `Point ${q.n(ex)} is drawn larger, with its circle of radius ε = ${q(eps, 2)}. The points inside it, coloured blue, are its neighbours. To be a core point it needs at least minPts = ${q.n(mp)} points in the circle, itself included.`, { phase: 'example', ex, eps, minPts: mp });
      F.set('core', nCore);
      const it1 = [...ax.items, P.circles(ax, 'db:cc', cores.map((i) => X[i]), eps, 'p7eps soft'), ...P.points(ax, 'db:p', X, X.map((_, j) => (core[j] ? 0 : -1)), { r: (j) => (j === ex ? 6.5 : 4.4), cls: (j) => (core[j] ? '' : 'hollow') }), P.circles(ax, 'db:ex', [X[ex]], eps, 'p7eps')];
      legendAt(it1, 60);
      it1.push(...P.row('db:l0', 520, 100, 0, 'core: ≥ ' + mp + ' within ε'), H().text('db:l1', 540, 124, 'not core', 'legend', 13), { key: 'db:l1d', kind: 'dot', x: 526, y: 119.5, r: 5, cls: 'sp cn hollow', audit: 'overlap-ok' });
      F.push(it1, `Point ${q.n(ex)} has ${P.qpl(q, exN.length, 'point')} within ε, so it ${exN.length >= mp ? 'is' : 'is not'} a core point. Counting every point the same way, ${q.n(nCore)} of the ${q.n(n)} are core points (filled); their ε-circles are drawn faintly. Only core points can grow a cluster.`, { phase: 'core', ex, count: exN.length, nCore, n });
      const steps = m.trace.filter((s) => s.kind === 'cluster'), show = steps.slice(0, 6);
      show.forEach((st, c) => {
        F.set('clusters', c + 1);
        const lab = st.labels.map((l) => (l < 0 ? -1 : l % 6)), size = st.labels.filter((l) => l === st.cluster).length;
        const items = [...ax.items, P.circles(ax, 'db:cc', cores.filter((i) => st.labels[i] === st.cluster).map((i) => X[i]), eps, 'p7eps soft'), ...P.points(ax, 'db:p', X, lab, { cls: (j) => (st.labels[j] >= 0 && !core[j] ? 'hollow' : '') }), S.Kit.scatter.ring(ax, 'db:seed', X[st.seed][0], X[st.seed][1], 'act', 10)];
        legendAt(items, 60);
        const more = c === show.length - 1 && steps.length > show.length ? ` The remaining ${P.qpl(q, steps.length - show.length, 'cluster')} grow the same way.` : '';
        F.push(items, `Cluster ${q.n(c + 1)} starts at core point ${q.n(st.seed)} (ringed): it takes every point within ε, then the points within ε of each core point it reached, and so on. It ends with ${P.qpl(q, size, 'point')}.` + more, { phase: 'grow', c, c1: c + 1, seed: st.seed, size, rest: steps.length - show.length });
      });
      F.set('clusters', m.nClusters);
      const nB = kinds.filter((k) => k === 'border').length, nN = kinds.filter((k) => k === 'noise').length;
      const items = [...ax.items, ...P.points(ax, 'db:p', X, labOf(m), { cls: kindCls(kinds) })];
      legendAt(items, 60);
      items.push({ key: 'db:k0', kind: 'dot', x: 526, y: 95.5, r: 5, cls: 'sp c0', audit: 'overlap-ok' }, H().text('db:k0t', 540, 100, 'core', 'legend', 13), { key: 'db:k1', kind: 'dot', x: 526, y: 119.5, r: 5, cls: 'sp c0 hollow', audit: 'overlap-ok' }, H().text('db:k1t', 540, 124, 'border: reached, not core', 'legend', 13), { key: 'db:k2', kind: 'dot', x: 526, y: 143.5, r: 5, cls: 'sp cn cross', audit: 'overlap-ok' }, H().text('db:k2t', 540, 148, 'noise: no core point reaches it', 'legend', 13));
      items.push(H().text('db:s', 520, 190, H().pl(m.nClusters, 'cluster') + ', ' + nCore + ' core, ' + nB + ' border, ' + nN + ' noise', 'strong', 13));
      F.push(items, m.nClusters === 0 ? `No point has ${q.n(mp)} points within ε = ${q(eps, 2)}, so there are no core points and all ${P.qpl(q, n, 'point')} are noise. Try a larger ε or a smaller minPts.` : `The result: ${P.qpl(q, m.nClusters, 'cluster')}, with ${q.n(nCore)} core points, ${q.n(nB)} border points (hollow: inside some core point’s circle, but not crowded enough to be core) and ${P.qpl(q, nN, 'noise point')} (grey), which belong to no cluster.`, { phase: 'done', clusters: m.nClusters, nCore, nB, nN, n, minPts: mp, eps });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'DBSCAN', text: ['N(x) = the points within ε of x (x included)', 'core: |N(x)| ≥ minPts', 'a cluster: core points joined by chains of', '  core points each within ε of the next,', '  plus every point within ε of them', 'border: in a cluster, not core;  noise: in none'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'example', pick: 'option', ask: 'Is the large point a core point?', options: ['yes', 'no'], answer: (fr, i) => (fr[i + 1].meta.count >= fr[i].meta.minPts ? 'yes' : 'no'), explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'done' && fr[i + 1].meta.clusters > 0, pick: 'option', ask: 'How many clusters will DBSCAN find in all?', options: (fr) => { const k = fr[fr.length - 1].meta.clusters; return [...new Set([1, 2, 3, 4, 5, k])].sort((a, b) => a - b).map(String); }, answer: (fr) => String(fr[fr.length - 1].meta.clusters), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'With minPts = 4, a point has 3 other points within ε. Is it a core point?', options: ['yes: it counts itself, so 4 points', 'no: it needs 4 others', 'only if it is a border point', 'it depends on k'], answer: 0, why: 'The point itself is in its own neighbourhood, so 1 + 3 = 4 ≥ minPts.' },
      { q: 'What is a border point?', options: ['a non-core point within ε of a core point', 'a core point at the edge of the plot', 'a point with no neighbours', 'a point in two clusters at once'], answer: 0, why: 'It is reached by a cluster but cannot extend it.' },
      { q: 'Why can DBSCAN find the two moons when k-means cannot?', options: ['clusters grow along dense chains, whatever their shape', 'it uses more iterations', 'it knows k = 2', 'it uses squared distances'], answer: 0, why: 'Density-connected regions can curve; k-means regions are convex.' },
      { q: 'What happens if ε is much too large?', options: ['everything merges into one cluster', 'everything becomes noise', 'there is one cluster per point', 'nothing changes'], answer: 0, why: 'Every point becomes core and reaches the others.' },
      { q: 'Does DBSCAN need the number of clusters in advance?', options: ['no, it needs ε and minPts', 'yes, like k-means', 'only for noise', 'yes, as minPts'], answer: 0, why: 'The number of clusters is an outcome of the density settings.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'dbscan-params', family: 'p07-dbscan', order: 20, title: 'Choosing ε and minPts',
    blurb: 'DBSCAN’s answer depends on its two settings. A small ε or a large minPts asks for a high density, so sparse parts become noise; a large ε or a small minPts lets clusters spill into one another. A k-distance curve helps: sort the points by the distance to their minPts-th nearest point. Most points are in crowded places and have a small distance; noise points have a large one. The bend in the curve is a natural ε.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'noisy', options: P.setOptions(['noisy', 'moons', 'circles', 'three', 'uneven']) },
      { id: 'minPts', kind: 'number', label: 'minPts', def: 5, min: 2, max: 20 },
    ],
    presets: [
      { label: 'Groups in noise', values: { set: 'noisy', minPts: 5 } },
      { label: 'Two moons', values: { set: 'moons', minPts: 5 } },
      { label: 'Uneven densities', values: { set: 'uneven', minPts: 6 } },
      { label: 'Ring and disc, minPts 3', values: { set: 'circles', minPts: 3 } },
    ],
    counters: [{ id: 'runs', label: 'settings tried' }],
    run(inp) {
      const d = P.data(inp.set, {}), X = d.X, n = X.length, mp = inp.minPts, F = H().frames(['runs']), q = F.q;
      const kd = P.kdist(X, mp).slice().sort((a, b) => a - b), knee = P.elbowKnee(kd), epsS = kd[knee];
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('dp', X, box);
      const KA = { x: 560, y: 50, w: 210, h: 150 }, kmax = kd[n - 1];
      const kax = C().axes(Object.assign({ prefix: 'kd', xr: [0, n], yr: [0, kmax * 1.05], xticks: C().ticks(0, n, 3), yticks: C().ticks(0, kmax, 3).filter((v) => v <= kmax * 1.05), xname: 'points, sorted', yname: 'distance to the ' + mp + P.ord(mp) + ' nearest' }, KA));
      const epsList = [0.5, 0.75, 1, 1.4, 2].map((f) => +(f * epsS).toFixed(3));
      const runs = epsList.map((e) => new (ML().DBSCAN)({ eps: e, minSamples: mp }).fit(X));
      const CA = { x: 560, y: 315, w: 210, h: 115 }, cmax = Math.max(4, ...runs.map((r) => r.nClusters));
      const cax = C().axes(Object.assign({ prefix: 'dc', xr: [epsList[0] * 0.9, epsList[4] * 1.05], yr: [0, 1], xticks: epsList.filter((_, i) => i % 2 === 0), yticks: [0, 0.5, 1], fmtX: (v) => H().fx(v, 2), fmtY: (v) => H().pct(v), xname: 'ε', yname: 'share of points that are noise' }, CA));
      const kItems = (hl) => { const it = [...kax.items, C().line(kax, 'kd:l', kd.map((v, i) => [i + 0.5, v]), 'act')]; it.push(C().dot(kax, 'kd:k', knee + 0.5, kd[knee], '', 5)); if (hl !== undefined) it.push(C().hline(kax, 'kd:e', hl, 'act')); return it; };
      F.push([...ax.items, ...P.points(ax, 'dp:p', X, null), ...kItems()], `For every point, the distance to its ${q.n(mp)}${P.ord(mp)} nearest point (itself counted first), sorted. The curve is low and flat for points in crowds and climbs steeply for isolated ones. It bends at about ${q(epsS, 3)}: a natural ε for minPts = ${q.n(mp)}.`, { phase: 'kdist', epsS, knee, minPts: mp });
      runs.forEach((m, t) => {
        F.count('runs');
        const noise = m.labels.filter((l) => l < 0).length;
        const items = [...ax.items, ...P.points(ax, 'dp:p', X, labOf(m), { cls: kindCls(m.kinds) }), ...kItems(epsList[t]), ...cax.items];
        const pts = runs.slice(0, t + 1).map((r, u) => [epsList[u], r.labels.filter((l) => l < 0).length / n]);
        items.push(C().line(cax, 'dc:l', pts, 'act'), C().dot(cax, 'dc:d', epsList[t], noise / n, '', 5));
        items.push(H().text('dp:r', 545, 268, `ε = ${H().fx(epsList[t], 3)}: ${H().pl(m.nClusters, 'cluster')}, ${noise} noise`, 'strong', 13));
        const rel = [0.5, 0.75, 1, 1.4, 2][t];
        F.push(items, `ε = ${q(epsList[t], 3)}${rel === 1 ? ', at the bend' : ''}: ${P.qpl(q, m.nClusters, 'cluster')} and ${P.qpl(q, noise, 'noise point')} (${q.pct(noise / n)}).` + (rel < 1 && noise / n > 0.3 ? ' Too strict: whole stretches of a cluster are called noise.' : rel > 1.2 && m.nClusters < runs[2].nClusters ? ' Too loose: clusters run together.' : ''), { phase: 'eps', eps: epsList[t], clusters: m.nClusters, noise, n, share: noise / n });
      });
      // minPts against the same ε
      const mps = [...new Set([2, mp, Math.min(20, mp * 3)])].sort((a, b) => a - b);
      const base = new (ML().DBSCAN)({ eps: epsS, minSamples: mp }).fit(X), baseNoise = base.labels.filter((l) => l < 0).length;
      mps.forEach((p2) => {
        F.count('runs');
        const m = new (ML().DBSCAN)({ eps: epsS, minSamples: p2 }).fit(X), noise = m.labels.filter((l) => l < 0).length;
        const items = [...ax.items, ...P.points(ax, 'dp:p', X, labOf(m), { cls: kindCls(m.kinds) }), ...kItems(epsS)];
        items.push(H().text('dp:r', 545, 268, `minPts = ${p2}: ${H().pl(m.nClusters, 'cluster')}, ${noise} noise`, 'strong', 13));
        F.push(items, `Keep ε = ${q(epsS, 3)} and set minPts = ${q.n(p2)}: ${P.qpl(q, m.nClusters, 'cluster')} and ${P.qpl(q, noise, 'noise point')}.` + (p2 < mp ? (m.nClusters > base.nClusters ? ' A small minPts makes almost any point core, so stray points form tiny clusters.' : noise < baseNoise ? ` A small minPts makes almost any point core, so fewer points are left as noise (${q.n(baseNoise)} at minPts = ${q.n(mp)}).` : '') : p2 > mp ? (noise > baseNoise ? ' A large minPts demands a crowd, so the thinner edges turn into noise.' : '') : ''), { phase: 'minpts', minPts: p2, eps: epsS, clusters: m.nClusters, noise, baseNoise, baseClusters: base.nClusters, baseMinPts: mp });
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Settings', text: ['k-distance: distance to the k-th nearest point', '  (k = minPts, the point itself first)', 'choose ε near the bend of the sorted curve', 'smaller ε, larger minPts: denser clusters,', '  more noise'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'eps' && fr[i + 1] && fr[i + 1].meta.phase === 'eps', once: true, pick: 'option', ask: 'With a larger ε, will there be more noise or less?', options: ['less', 'more', 'the same'], answer: (fr, i) => (fr[i + 1].meta.noise < fr[i].meta.noise ? 'less' : fr[i + 1].meta.noise > fr[i].meta.noise ? 'more' : 'the same'), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'As ε grows with minPts fixed, the number of noise points…', options: ['can only stay the same or fall', 'can only rise', 'always stays the same', 'goes up and down'], answer: 0, why: 'A bigger circle holds at least as many points, so core points stay core and reach at least as far.' },
      { q: 'Why does the k-distance curve bend?', options: ['crowded points have close k-th neighbours; isolated ones do not', 'k-means stops there', 'the points are sorted by class', 'it always bends at the middle'], answer: 0, why: 'The steep end belongs to sparse points, which make good noise.' },
      { q: 'What is the weakness of a single ε?', options: ['clusters of very different densities cannot all be found', 'it is slow', 'it needs k', 'it only finds round clusters'], answer: 0, why: 'An ε right for a dense cluster calls a sparse one noise, and one right for the sparse cluster merges dense ones.' },
      { q: 'A common rule of thumb for minPts in two dimensions?', options: ['about 4 or 5 (twice the dimension, or so)', '1', 'the number of points', 'equal to ε'], answer: 0, why: 'Small enough to keep thin clusters, large enough to ignore lone pairs of points.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
