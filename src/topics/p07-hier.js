// Part 7, hierarchical clustering: merge by merge, the two closest clusters joined (the pair that sets the linkage
// distance drawn in the plane, the dendrogram growing beside it), then the tree cut at a height; and the four
// linkages compared on the same points, each with its own tree and its own clusters.
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, ML = () => S.ML, DG = () => S.Kit.dendro;
  const LINKS = [{ value: 'single', label: 'single (nearest pair)' }, { value: 'complete', label: 'complete (farthest pair)' }, { value: 'average', label: 'average (mean of all pairs)' }, { value: 'ward', label: 'Ward (least growth in variance)' }];
  const linkName = (l) => (l === 'ward' ? 'Ward' : l);

  /** the members of every cluster after merge t (ids n + t) */
  const members = (merges, n) => { const m = new Map(); for (let i = 0; i < n; i++) m.set(i, [i]); merges.forEach((g, t) => m.set(n + t, m.get(g.a).concat(m.get(g.b)))); return m; };
  const centroid = (X, ids) => [0, 1].map((j) => ids.reduce((s, i) => s + X[i][j], 0) / ids.length);
  /** the two places that stand for a merge in the plane: the nearest pair, the farthest pair, or the centroids */
  P.mergePair = function (X, A, B, link) {
    if (link === 'average' || link === 'ward') return [centroid(X, A), centroid(X, B)];
    let best = null, bd = link === 'single' ? Infinity : -Infinity;
    for (const i of A) for (const j of B) { const d = ML().sqdist(X[i], X[j]); if (link === 'single' ? d < bd : d > bd) { bd = d; best = [X[i], X[j]]; } }
    return best;
  };
  const colourOf = (L, labels, n, cutH) => (id) => (L.h(id) <= cutH ? 'c' + (labels[L.leaves(id)[0]] % 6) : '');

  S.topic({
    id: 'hc-build', family: 'p07-hier', order: 10, title: 'Agglomerative clustering and the dendrogram',
    blurb: 'Start with every point as its own cluster. Again and again, merge the two clusters that are closest, until one is left. The dendrogram records the story: each join stands at the distance at which the merge happened. Cut it with a horizontal line and the branches below the cut are the clusters; a higher cut gives fewer, bigger clusters.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'three', options: P.setOptions(['three', 'bridge', 'stretched', 'moons']) },
      { id: 'n', kind: 'number', label: 'Points', def: 18, min: 6, max: 30 },
      { id: 'link', kind: 'choice', label: 'Linkage', def: 'average', options: LINKS },
      { id: 'cut', kind: 'slider', label: 'Cut height (share of the tallest join)', def: 0.5, min: 0.02, max: 0.98, step: 0.01, fmt: (v) => Math.round(v * 100) + '%' },
    ],
    presets: [
      { label: 'Three groups, average linkage', values: { set: 'three', n: 18, link: 'average', cut: 0.5 } },
      { label: 'Single linkage follows the bridge', values: { set: 'bridge', n: 24, link: 'single', cut: 0.6 } },
      { label: 'Complete linkage on the bridge', values: { set: 'bridge', n: 24, link: 'complete', cut: 0.6 } },
      { label: 'Ward, cut low', values: { set: 'three', n: 24, link: 'ward', cut: 0.15 } },
      { label: 'Moons, single linkage', values: { set: 'moons', n: 30, link: 'single', cut: 0.7 } },
    ],
    counters: [{ id: 'merges', label: 'merges' }, { id: 'clusters', label: 'clusters' }],
    run(inp) {
      const d = P.data(inp.set, { n: inp.n, seed: 2 }), X = d.X, n = X.length, F = H().frames(['merges', 'clusters']), q = F.q;
      const merges = ML().linkage(X, inp.link), L = DG().layout(merges, n), mem = members(merges, n);
      const top = merges[n - 2].dist, hmax = top * 1.06, cutH = inp.cut * top, labels = DG().cutClusters(merges, n, cutH), kCut = new Set(labels).size;
      const box = { x: 40, y: 20, w: 390, h: 400 }, ax = P.axes('hc', X, box, { fit: true });
      const DB = { x: 495, y: 60, w: 215, h: 330 };
      const seg = (key, p, cls) => ({ key, kind: 'path', d: `M${ax.X(p[0][0])},${ax.Y(p[0][1])} L${ax.X(p[1][0])},${ax.Y(p[1][1])}`, cls, pathAudit: 'edge-x', layer: 'lines' });
      const pairs = merges.map((m) => P.mergePair(X, mem.get(m.a), mem.get(m.b), inp.link));
      F.set('clusters', n);
      F.push([...ax.items, ...P.points(ax, 'hc:p', X, null), ...DG().draw(L, { prefix: 'dg', ...DB, hmax, upto: -1, yname: linkName(inp.link) + ' distance' })],
        `${P.qpl(q, n, 'point')}, each a cluster of its own. With ${inp.link === 'ward' ? 'Ward' : inp.link} linkage, the distance between two clusters is ${inp.link === 'single' ? 'the distance between their nearest pair of points' : inp.link === 'complete' ? 'the distance between their farthest pair of points' : inp.link === 'average' ? 'the mean distance over every pair of their points' : 'how much merging them would add to the spread within clusters'}. At every step the two closest clusters merge.`, { phase: 'start', n });
      merges.forEach((m, t) => {
        F.count('merges'); F.set('clusters', n - t - 1);
        const A = mem.get(m.a), B = mem.get(m.b), lab = X.map((_, i) => (A.includes(i) ? 0 : B.includes(i) ? 1 : -1));
        const items = [...ax.items];
        for (let u = 0; u < t; u++) items.push(seg('hc:s' + u, pairs[u], 'p7nb'));
        items.push(seg('hc:s' + t, pairs[t], 'p7pair'));
        items.push(...P.points(ax, 'hc:p', X, lab));
        items.push(...DG().draw(L, { prefix: 'dg', ...DB, hmax, upto: t, hot: t, yname: linkName(inp.link) + ' distance', leafCls: (i) => (A.includes(i) ? 'c0' : B.includes(i) ? 'c1' : 'cn') }));
        const say = (A.length === 1 && B.length === 1 ? `Merge ${q.n(t + 1)}: two single points, blue and orange, join at distance ${q(m.dist, 3)}.` : `Merge ${q.n(t + 1)}: a cluster of ${P.qpl(q, A.length, 'point')} (blue) and one of ${q.n(B.length)} (orange), at distance ${q(m.dist, 3)}.`) + ` ${P.qpl(q, n - t - 1, 'cluster')} ${n - t - 1 === 1 ? 'is' : 'are'} left.` + (t > 0 && m.dist < merges[t - 1].dist - 1e-12 ? '' : t === n - 2 ? ' Everything is now one cluster: the top of the tree.' : '');
        F.push(items, say, { phase: 'merge', t, a: m.a, b: m.b, sa: A.length, sb: B.length, dist: m.dist, left: n - t - 1, merge: t + 1 });
      });
      // the cut
      const items = [...ax.items];
      merges.forEach((m, u) => items.push(seg('hc:s' + u, pairs[u], 'p7nb')));
      items.push(...P.points(ax, 'hc:p', X, labels.map((l) => l % 6)));
      items.push(...DG().draw(L, { prefix: 'dg', ...DB, hmax, colour: colourOf(L, labels, n, cutH), cut: cutH, cutLabel: 'cut at ' + H().fx(cutH, 2), yname: linkName(inp.link) + ' distance', leafCls: (i) => 'c' + (labels[i] % 6) }));
      F.set('clusters', kCut);
      const sizes = P.sizes(labels, kCut).sort((a, b) => b - a);
      F.push(items, `Cut the tree at height ${q(cutH, 3)} (${q.pct(inp.cut)} of the tallest join). It crosses ${P.qpl(q, kCut, 'branch', 'branches')}, so the points fall into ${P.qpl(q, kCut, 'cluster')}` + (kCut > 1 ? ` of ${H().list(sizes.map((v) => q.n(v)))} points.` : '.') + ' Slide the cut to get more or fewer clusters without clustering again.', { phase: 'cut', cutH, k: kCut, sizes, frac: inp.cut });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Linkages', text: ['single: min over pairs d(a, b)', 'complete: max over pairs d(a, b)', 'average: mean over pairs d(a, b)', 'Ward: √(2·|A||B| / (|A| + |B|)) · ‖μA − μB‖', 'merge heights never fall for these four', 'cut at height h: clusters = branches below h'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'merge' && f.meta.t === 2 && fr[i + 1].meta.phase === 'merge', pick: 'option', ask: 'Can the next merge happen lower in the tree than this one?', options: ['no: merge heights never fall', 'yes, often', 'only with single linkage'], answer: 'no: merge heights never fall', explain: (fr, i) => fr[i + 1].say },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'cut', pick: 'option', ask: (fr) => 'How many clusters will a cut at ' + Math.round(fr[fr.length - 1].meta.frac * 100) + '% of the tallest join leave?', options: (fr) => { const k = fr[fr.length - 1].meta.k; return [...new Set([1, 2, 3, 4, 5, 6, k])].sort((a, b) => a - b).map(String); }, answer: (fr) => String(fr[fr.length - 1].meta.k), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Clusters {(0, 0), (1, 0)} and {(4, 0), (6, 0)}. What is their single-linkage distance?', options: ['3', '6', '4.5', '1'], answer: 0, why: 'The nearest pair is (1, 0) and (4, 0).' },
      { q: 'And their complete-linkage distance?', options: ['6', '3', '4.5', '5'], answer: 0, why: 'The farthest pair is (0, 0) and (6, 0).' },
      { q: 'How many merges does agglomerative clustering of n points make?', options: ['n − 1', 'n', 'n / 2', 'log n'], answer: 0, why: 'Each merge removes one cluster, from n down to 1.' },
      { q: 'What does the height of a join in the dendrogram show?', options: ['the linkage distance at which the two clusters merged', 'the number of points merged', 'the order of the merge only', 'the size of the bigger cluster'], answer: 0, why: 'That is why a horizontal cut at height h separates exactly the clusters that are further apart than h.' },
      { q: 'Why can single linkage join two clearly separate groups?', options: ['a chain of close points between them is enough', 'it uses the farthest pair', 'it averages all distances', 'it needs k in advance'], answer: 0, why: 'Only the nearest pair counts, so a thin bridge links everything along it: chaining.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'hc-linkage', family: 'p07-hier', order: 20, title: 'Four linkages, four trees',
    blurb: 'The same points give different trees depending on how the distance between two clusters is measured. Single linkage follows chains of close points; complete linkage keeps clusters compact; average linkage sits between; Ward merges whatever adds the least spread, like k-means. Each tree is cut to the same number of clusters.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'bridge', options: P.setOptions(['bridge', 'three', 'stretched', 'moons', 'uneven']) },
      { id: 'n', kind: 'number', label: 'Points', def: 30, min: 10, max: 40 },
      { id: 'k', kind: 'number', label: 'Clusters (where each tree is cut)', def: 2, min: 2, max: 6 },
    ],
    presets: [
      { label: 'A bridge: single linkage chains', values: { set: 'bridge', n: 30, k: 2 } },
      { label: 'Three round groups', values: { set: 'three', n: 30, k: 3 } },
      { label: 'Two moons', values: { set: 'moons', n: 40, k: 2 } },
      { label: 'Uneven groups', values: { set: 'uneven', n: 36, k: 3 } },
    ],
    counters: [{ id: 'trees', label: 'trees drawn' }],
    run(inp) {
      const d = P.data(inp.set, { n: inp.n, seed: 2 }), X = d.X, n = X.length, k = inp.k, F = H().frames(['trees']), q = F.q;
      const box = { x: 40, y: 20, w: 390, h: 400 }, ax = P.axes('hl', X, box, { fit: true }), DB = { x: 495, y: 60, w: 215, h: 330 };
      const res = LINKS.map((l) => {
        const merges = ML().linkage(X, l.value), L = DG().layout(merges, n), lo = merges[n - k - 1] ? merges[n - k - 1].dist : 0, hi = merges[n - k].dist, cutH = (lo + hi) / 2, labels = DG().cutClusters(merges, n, cutH);
        return { link: l.value, merges, L, cutH, labels, sizes: P.sizes(labels, k).slice().sort((a, b) => b - a), top: merges[n - 2].dist, gap: hi - lo };
      });
      F.push([...ax.items, ...P.points(ax, 'hl:p', X, null), H().head('hl:h0', 500, 80, 'Four ways to measure'), H().text('hl:h00', 500, 102, 'the distance between two clusters', 'axname', 12), ...H().lines('hl:d', 500, 136, LINKS.map((l) => [l.label, 'legend']), 13, 26)], `${P.qpl(q, n, 'point')}. Each linkage builds its own tree from them, and each tree is cut where it leaves ${P.qpl(q, k, 'cluster')}.`, { phase: 'start', n, k });
      res.forEach((r, j) => {
        F.count('trees');
        const items = [...ax.items, ...P.points(ax, 'hl:p', X, r.labels.map((l) => l % 6))];
        items.push(...DG().draw(r.L, { prefix: 'dg', ...DB, hmax: r.top * 1.06, colour: colourOf(r.L, r.labels, n, r.cutH), cut: r.cutH, cutLabel: k + ' clusters', yname: linkName(r.link) + ' distance', leafCls: (i) => 'c' + (r.labels[i] % 6) }));
        const big = r.sizes[0] / n;
        let note = '';
        if (r.link === 'single' && big > 0.8) note = ' One cluster swallows nearly everything: single linkage chains through close neighbours and leaves stragglers as tiny clusters.';
        else if (r.link === 'ward') note = ' Ward prefers clusters of similar spread, much as k-means does.';
        else if (r.link === 'complete') note = ' Complete linkage keeps every cluster’s diameter small, so it splits long groups.';
        F.push(items, `${r.link === 'ward' ? 'Ward' : r.link[0].toUpperCase() + r.link.slice(1)} linkage: the top join is at ${q(r.top, 3)}, and the cut for ${P.qpl(q, k, 'cluster')} gives sizes ${H().list(r.sizes.map((v) => q.n(v)))}.` + note, { phase: 'tree', link: r.link, top: r.top, cutH: r.cutH, sizes: r.sizes, k, big });
      });
      // agreement between the linkages: the share of pairs of points that each puts together or apart alike
      const agree = (a, b) => { let s = 0, t = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { t++; if ((a[i] === a[j]) === (b[i] === b[j])) s++; } return s / t; };
      const sw = agree(res[0].labels, res[3].labels), cw = agree(res[1].labels, res[3].labels), aw = agree(res[2].labels, res[3].labels);
      const items = [...ax.items, ...P.points(ax, 'hl:p', X, res[3].labels.map((l) => l % 6))];
      const tx = 500;
      items.push(H().head('hl:h', tx, 70, 'Agreement with Ward'), H().text('hl:h2', tx, 92, 'share of pairs put together or apart alike', 'axname', 12));
      [['single', sw], ['complete', cw], ['average', aw]].forEach(([nm, v], r) => {
        const y = 120 + r * 34;
        items.push(H().text('hl:n' + r, tx, y + 14, nm, 'legend', 13), { key: 'hl:b' + r, kind: 'cell', x: tx + 80, y: y + 2, w: Math.max(2, 150 * v), h: 16, rx: 2, text: '', cls: 'bar', audit: 'ignore' }, H().text('hl:v' + r, tx + 86 + 150 * v, y + 14, H().pct(v), 'legend', 12));
      });
      F.push(items, `How alike are the four answers? Count the pairs of points that two clusterings both put together, or both keep apart. Against Ward, single linkage agrees on ${q.pct(sw)} of pairs, complete on ${q.pct(cw)} and average on ${q.pct(aw)}. The points shown are coloured by Ward.`, { phase: 'agree', sw, cw, aw });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Cluster distances', text: ['single: d(A, B) = min d(a, b)', 'complete: d(A, B) = max d(a, b)', 'average: d(A, B) = mean d(a, b)', 'Ward: the rise in the total within-cluster', '  sum of squares if A and B merge'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which linkage is most likely to put almost every point in one cluster?', options: ['single', 'complete', 'average', 'Ward'], answer: (fr) => { const t = fr.filter((f) => f.meta.phase === 'tree'); const b = t.reduce((m, f) => (f.meta.big > m.meta.big ? f : m), t[0]); return b.meta.link === 'ward' ? 'Ward' : b.meta.link; }, explain: (fr) => fr.find((f) => f.meta.phase === 'tree').say },
    ],
    quiz: [
      { q: 'Which linkage suffers most from chaining?', options: ['single', 'complete', 'average', 'Ward'], answer: 0, why: 'Only the nearest pair counts, so a line of close points glues groups together.' },
      { q: 'Which linkage behaves most like k-means?', options: ['Ward', 'single', 'complete', 'none of them'], answer: 0, why: 'Ward merges the pair that least increases the within-cluster sum of squares, the quantity k-means minimises.' },
      { q: 'Complete linkage tends to give clusters that are…', options: ['compact, with small diameters', 'long and thin', 'of very different sizes', 'always two'], answer: 0, why: 'A merge is judged by its farthest pair, so a wide cluster is expensive.' },
      { q: 'To get k clusters from a tree, cut it…', options: ['between the heights of the (n − k)-th and (n − k + 1)-th merges', 'at height k', 'at the median merge height', 'at the root'], answer: 0, why: 'After n − k merges exactly k clusters remain; the next merge would make k − 1.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
