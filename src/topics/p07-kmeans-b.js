// Part 7, choosing k: the elbow of the inertia curve (k-means for k = 1, 2, 3 … with the curve growing beside it),
// and the silhouette, which scores each point by how much closer it is to its own cluster than to the next one.
(function (S) {
  'use strict';
  const P = S.P07, H = () => S.Kit.ml, C = () => S.Kit.curve, ML = () => S.ML;

  /** the elbow: the k whose point lies furthest below the chord from the first to the last (both axes scaled 0..1) */
  P.elbowK = function (I) {
    const n = I.length, lo = Math.min(...I), hi = Math.max(...I);
    let best = 1, bd = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = i / (n - 1), y = (I[i] - lo) / (hi - lo || 1), chord = 1 - x, d = chord - y;
      if (d > bd) { bd = d; best = i + 1; }
    }
    return best;
  };
  const fitK = (X, k) => new (ML().KMeans)({ k, seed: 0, nInit: 4, trace: false }).fit(X);

  S.topic({
    id: 'km-elbow', family: 'p07-kmeans', order: 30, title: 'Choosing k: the elbow',
    blurb: 'More centres always lower the inertia: with one centre per point it would be zero. So the inertia alone cannot choose k. Plot it against k instead: it falls steeply while each new centre splits a real group, then flattens once new centres only cut groups in two. The bend, the elbow, is a good guess for k.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'four', options: P.setOptions(['three', 'four', 'touching', 'uneven', 'moons', 'iris', 'penguins']) },
      { id: 'kmax', kind: 'number', label: 'Largest k', def: 8, min: 4, max: 10 },
    ],
    presets: [
      { label: 'Four groups: a clear elbow', values: { set: 'four', kmax: 8 } },
      { label: 'Three groups', values: { set: 'three', kmax: 8 } },
      { label: 'Groups that touch: a soft elbow', values: { set: 'touching', kmax: 8 } },
      { label: 'Two moons: no real elbow', values: { set: 'moons', kmax: 8 } },
      { label: 'Iris petals', values: { set: 'iris', kmax: 8 } },
    ],
    counters: [{ id: 'k', label: 'k tried' }],
    run(inp) {
      const d = P.data(inp.set, {}), X = d.X, F = H().frames(['k']), q = F.q, K = inp.kmax;
      const fits = [];
      for (let k = 1; k <= K; k++) fits.push(fitK(X, k));
      const I = fits.map((m) => m.inertia), elbow = P.elbowK(I);
      const box = { x: 40, y: 20, w: 430, h: 430 }, ax = P.axes('eb', X, box, { xname: d.names[0], yname: d.names[1] });
      const CA = { x: 560, y: 50, w: 210, h: 220 };
      const cax = C().axes(Object.assign({ prefix: 'ec', xr: [0.5, K + 0.5], yr: [0, I[0] * 1.05], xticks: [...Array(K).keys()].map((i) => i + 1).filter((k) => K <= 8 || k % 2), yticks: C().ticks(0, I[0], 4).filter((v) => v <= I[0] * 1.05), xname: 'k', yname: 'inertia' }, CA));
      const draw = (k, showElbow) => {
        const m = fits[k - 1], items = [...ax.items, ...P.points(ax, 'eb:p', X, m.labels.map((l) => l % 6), { r: 4.2 })];
        m.centers.forEach((c, j) => items.push(P.centre(ax, 'eb:c' + j, c, j % 6, 16)));
        items.push(...cax.items, C().line(cax, 'ec:l', I.slice(0, k).map((v, i) => [i + 1, v]), 'act'));
        I.slice(0, k).forEach((v, i) => items.push(C().dot(cax, 'ec:d' + i, i + 1, v, i + 1 === k ? '' : 'muted', i + 1 === k ? 5 : 3.5)));
        if (showElbow) { items.push(C().vline(cax, 'ec:el', elbow, 'act')); items.push(H().text('ec:elt', cax.X(elbow) + 8, CA.y + 16, 'elbow at k = ' + elbow, 'strong', 13)); }
        items.push(H().text('eb:k', 545, 330, 'k = ' + k, 'strong', 15), H().text('eb:i', 545, 354, 'inertia ' + H().fx(I[k - 1], 1), 'legend', 13));
        if (k > 1) items.push(H().text('eb:dr', 545, 378, 'down ' + H().pct((I[k - 2] - I[k - 1]) / I[k - 2]) + ' from k = ' + (k - 1), 'legend', 13));
        return items;
      };
      for (let k = 1; k <= K; k++) {
        F.set('k', k);
        const drop = k > 1 ? (I[k - 2] - I[k - 1]) / I[k - 2] : null;
        let say;
        if (k === 1) say = `With one centre, at the mean of all ${P.qpl(q, X.length, 'point')}, the inertia is ${q(I[0], 1)}: the total spread of the data.`;
        else say = `k = ${q.n(k)}: the best of four k-means runs has inertia ${q(I[k - 1], 1)}, down ${q.pct(drop)} from k = ${q.n(k - 1)}.` + (drop > 0.4 ? ' A big drop: the new centre found a group of its own.' : drop < 0.2 ? ' A small drop: the new centre mostly splits a group that was already well served.' : '');
        F.push(draw(k, false), say, { phase: 'k', k, kprev: k - 1, n: X.length, inertia: I[k - 1], prev: k > 1 ? I[k - 2] : null, drop, I: I.slice(), centers: fits[k - 1].centers });
      }
      const before = elbow > 1 ? (I[elbow - 2] - I[elbow - 1]) / I[elbow - 2] : null, after = elbow < K ? (I[elbow - 1] - I[elbow]) / I[elbow - 1] : null;
      F.push(draw(elbow, true), `The curve bends at k = ${q.n(elbow)}, the point furthest below the straight line from k = ${q.n(1)} to k = ${q.n(K)}.` + (before !== null && after !== null ? ` Going to ${q.n(elbow)} centres cut the inertia by ${q.pct(before)}; one more centre would cut only ${q.pct(after)}.` : '') + (inp.set === 'moons' ? ' Here the bend is weak: the two moons are not round groups, so no k suits k-means well.' : ''), { phase: 'elbow', elbow, before, after, K, I: I.slice(), k: elbow, first: 1 });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 480, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The elbow', text: ['I(k) = the lowest inertia found with k centres', 'I(k) never rises as k grows; I(n) = 0', 'elbow: where the drop I(k−1) − I(k) becomes small', 'here: the k furthest below the chord from I(1) to I(kmax)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'k' && f.meta.k === 2, pick: 'option', ask: 'At which k do you expect the curve to bend?', options: (fr) => { const K = fr[fr.length - 1].meta.K; return [...Array(K - 1).keys()].map((i) => String(i + 2)); }, answer: (fr) => String(fr[fr.length - 1].meta.elbow), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Why not just pick the k with the lowest inertia?', options: ['the inertia keeps falling as k grows, down to 0 at one centre per point', 'the lowest inertia is always at k = 2', 'inertia cannot be computed for large k', 'it would be too slow'], answer: 0, why: 'More centres can never fit worse, so the lowest inertia always means the largest k.' },
      { q: 'Inertia falls 1000 → 400 → 150 → 130 → 115 for k = 1 to 5. Where is the elbow?', options: ['k = 3', 'k = 2', 'k = 4', 'k = 5'], answer: 0, why: 'The drops are 600, 250, 20, 15: after k = 3 each new centre gains little.' },
      { q: 'When is the elbow hard to see?', options: ['when the groups overlap or are not round', 'when the groups are far apart', 'when k is small', 'never'], answer: 0, why: 'Without clear round groups the curve falls smoothly, with no sharp bend.' },
      { q: 'What does the inertia at k = 1 equal?', options: ['the sum of squared distances to the overall mean', 'zero', 'the number of points', 'the largest distance between two points'], answer: 0, why: 'The single best centre is the mean of all the points.' },
    ],
  });

  // -----------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'km-silhouette', family: 'p07-kmeans', order: 40, title: 'The silhouette: how well each point fits',
    blurb: 'For one point, let a be its mean distance to the other points of its own cluster, and b its mean distance to the points of the nearest other cluster. Its silhouette is s = (b − a) / max(a, b): near 1 when it sits snugly in its cluster, near 0 on a border, negative when it would rather be elsewhere. The mean silhouette scores a whole clustering, and can be compared across k.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Data', def: 'three', options: P.setOptions(['three', 'four', 'touching', 'uneven', 'iris', 'penguins']) },
      { id: 'k', kind: 'number', label: 'k (clusters)', def: 3, min: 2, max: 6 },
    ],
    presets: [
      { label: 'Three groups, k = 3', values: { set: 'three', k: 3 } },
      { label: 'Three groups, k = 5 (too many)', values: { set: 'three', k: 5 } },
      { label: 'Four groups, k = 2 (too few)', values: { set: 'four', k: 2 } },
      { label: 'Touching groups', values: { set: 'touching', k: 3 } },
      { label: 'Penguins, k = 3', values: { set: 'penguins', k: 3 } },
    ],
    counters: [{ id: 'k', label: 'k' }],
    run(inp) {
      const d = P.data(inp.set, {}), X = d.X, n = X.length, k = inp.k, F = H().frames(['k']), q = F.q;
      F.set('k', k);
      const m = fitK(X, k), lab = m.labels, sil = ML().silhouette(X, lab), s = sil.samples;
      const box = { x: 40, y: 20, w: 400, h: 430 }, ax = P.axes('si', X, box, { xname: d.names[0], yname: d.names[1] });
      // the silhouette plot: one thin bar per point, grouped by cluster, longest first
      const order = [];
      for (let j = 0; j < k; j++) order.push(...s.map((v, i) => [v, i]).filter(([, i]) => lab[i] === j).sort((a, b) => b[0] - a[0]).map(([, i]) => i));
      const SP = { x: 560, y: 40, w: 200, h: 390 }, sMin = Math.min(-0.2, ...s);
      const sax = C().axes(Object.assign({ prefix: 'sx', xr: [sMin, 1], yr: [0, 1], xticks: [0, 0.5, 1].concat(sMin < -0.3 ? [-0.5] : []), yticks: 0, xname: 'silhouette', grid: false, frame: false }, SP));
      const gap = 6, bh = (SP.h - gap * (k - 1)) / n;
      const barY = (r) => { const i = order[r]; return SP.y + r * bh + lab[i] * gap; };
      const plot = (hot) => {
        const it = [...sax.items, H().text('sx:ttl', SP.x, SP.y - 30, 'silhouette of each point', 'axname', 13), { key: 'sx:zero', kind: 'path', d: `M${sax.X(0)},${SP.y} V${SP.y + SP.h}`, cls: 'axis', audit: 'ignore', layer: 'back' }];
        order.forEach((i, r) => { const x0 = sax.X(0), x1 = sax.X(s[i]); it.push({ key: 'sx:b' + i, kind: 'cell', x: Math.min(x0, x1), y: barY(r), w: Math.max(0.5, Math.abs(x1 - x0)), h: Math.max(0.6, bh - 0.4), rx: 0, text: '', cls: 'bar ' + (i === hot ? 'hotbar' : 'c' + lab[i]), audit: 'ignore' }); });
        return it;
      };
      const pts = (hot) => P.points(ax, 'si:p', X, lab, { r: (i) => (i === hot ? 6.5 : 4.2), cls: (i) => (i === hot ? 'cross' : '') });
      F.push([...ax.items, ...pts(-1), ...m.centers.map((c, j) => P.centre(ax, 'si:c' + j, c, j, 16))], `k-means with k = ${q.n(k)} splits the ${P.qpl(q, n, 'point')} into clusters of ${H().list(P.sizes(lab, k).map((v) => q.n(v)))}. How well does each point fit its cluster?`, { phase: 'clusters', k, n, sizes: P.sizes(lab, k), centers: m.centers });
      // one point worked through: a typical one, and the worst
      const dist = (i, j) => Math.sqrt(ML().sqdist(X[i], X[j]));
      const ab = (i) => { const means = []; for (let c = 0; c < k; c++) { const js = X.map((_, j) => j).filter((j) => j !== i && lab[j] === c); means.push(js.length ? js.reduce((t, j) => t + dist(i, j), 0) / js.length : null); } const a = means[lab[i]], others = means.map((v, c) => [v, c]).filter(([v, c]) => c !== lab[i] && v !== null).sort((u, w) => u[0] - w[0]); return { a, b: others[0][0], nb: others[0][1] }; };
      const typical = order.filter((i) => lab[i] === 0)[Math.floor(order.filter((i) => lab[i] === 0).length / 2)], worst = s.indexOf(Math.min(...s));
      for (const [i, which] of [[typical, 'typical'], [worst, 'worst']]) {
        const r = ab(i), items = [...ax.items, ...pts(i)];
        X.forEach((x, j) => { if (j !== i && (lab[j] === lab[i] || lab[j] === r.nb)) items.push({ key: 'si:ln' + j, kind: 'path', d: `M${ax.X(X[i][0])},${ax.Y(X[i][1])} L${ax.X(x[0])},${ax.Y(x[1])}`, cls: 'p7nb', audit: 'ignore', layer: 'back', z: 3 }); });
        items.push(...plot(i));
        const say = which === 'typical'
          ? `Take point ${q.n(i)}, in the ${P.NAMES[lab[i]]} cluster. Its mean distance to the rest of its cluster is a = ${q(r.a, 3)}; to the nearest other cluster, ${P.NAMES[r.nb]}, it is b = ${q(r.b, 3)}. So s = (b − a) / max(a, b) = ${q(s[i], 3)}: it fits well.`
          : `The worst point is ${q.n(i)} (${P.NAMES[lab[i]]}): a = ${q(r.a, 3)}, b = ${q(r.b, 3)} to ${P.NAMES[r.nb]}, so s = ${q(s[i], 3)}.` + (s[i] < 0 ? ' It is negative: on average this point is closer to another cluster than to its own.' : ' Even the worst point is closer to its own cluster on average.');
        F.push(items, say, { phase: which, i, a: r.a, b: r.b, s: s[i], nb: r.nb });
      }
      const means = [];
      for (let c = 0; c < k; c++) { const v = s.filter((_, i) => lab[i] === c); means.push(v.reduce((a, b) => a + b, 0) / v.length); }
      const items = [...ax.items, ...pts(-1), ...plot(-1), { key: 'sx:mean', kind: 'path', d: `M${sax.X(sil.mean)},${SP.y - 6} V${SP.y + SP.h + 4}`, cls: 'guide act', audit: 'ignore', layer: 'lines' }, H().text('sx:mt', sax.X(sil.mean) + 4, SP.y - 10, 'mean ' + H().fx(sil.mean, 2), 'strong', 12)];
      const negs = s.filter((v) => v < 0).length, worstC = means.indexOf(Math.min(...means));
      F.push(items, `Every point’s silhouette, as bars sorted within each cluster. The mean silhouette is ${q(sil.mean, 3)}; the weakest cluster is ${P.NAMES[worstC]}, with mean ${q(means[worstC], 3)}, and ` + (negs ? `${P.qpl(q, negs, 'point')} ${negs === 1 ? 'has' : 'have'} a negative silhouette.` : 'no point has a negative silhouette.'), { phase: 'plot', mean: sil.mean, means, negs, worstC });
      // the mean silhouette for every k
      const ks = [2, 3, 4, 5, 6], fk = ks.map((kk) => (kk === k ? m : fitK(X, kk))), scores = ks.map((kk, t) => (kk === k ? sil.mean : ML().silhouette(X, fk[t].labels).mean)), bestK = ks[scores.indexOf(Math.max(...scores))];
      const KA = { x: 560, y: 60, w: 200, h: 200 };
      const kax = C().axes(Object.assign({ prefix: 'sk', xr: [1.5, 6.5], yr: [0, 1], xticks: ks, yticks: [0, 0.5, 1], xname: 'k', yname: 'mean silhouette' }, KA));
      const it2 = [...ax.items, ...pts(-1), ...kax.items, C().line(kax, 'sk:l', ks.map((kk, t) => [kk, Math.max(0, scores[t])]), 'act')];
      ks.forEach((kk, t) => it2.push(C().dot(kax, 'sk:d' + kk, kk, Math.max(0, scores[t]), kk === k ? '' : 'muted', kk === k ? 5.5 : 4)));
      it2.push(H().text('sk:b', 545, 330, 'best k here: ' + bestK, 'strong', 14), H().text('sk:k', 545, 354, 'this run: k = ' + k, 'legend', 13));
      F.push(it2, `The same score for k = ${q.n(ks[0])} to ${q.n(ks[ks.length - 1])}: the mean silhouette is highest, ${q(Math.max(...scores), 3)}, at k = ${q.n(bestK)}.` + (bestK === k ? ' That is the k chosen here.' : ` With k = ${q.n(k)} it is ${q(sil.mean, 3)}.`), { phase: 'ks', scores, bestK, k, mean: sil.mean, ks, centersByK: fk.map((f) => f.centers) });
      for (const f of F.list) f.narrowView = { x: 0, y: 0, w: 450, h: 475 };
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The silhouette', text: ['a(i) = mean distance from i to its own cluster', 'b(i) = smallest mean distance from i to another cluster', 's(i) = (b − a) / max(a, b), between −1 and 1', 'score of a clustering: the mean of s(i)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'typical', pick: 'option', ask: 'Will the worst point’s silhouette be negative?', options: ['yes', 'no'], answer: (fr, i) => (fr[i + 1].meta.s < 0 ? 'yes' : 'no'), explain: (fr, i) => fr[i + 1].say },
      { at: (f) => f.meta.phase === 'plot', pick: 'option', ask: 'For which k will the mean silhouette be highest?', options: ['2', '3', '4', '5', '6'], answer: (fr, i) => String(fr[i + 1].meta.bestK), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'A point has a = 1 and b = 4. What is its silhouette?', options: ['0.75', '0.25', '3', '−0.75'], answer: 0, why: '(4 − 1) / max(1, 4) = 3 / 4.' },
      { q: 'What does a negative silhouette mean?', options: ['the point is on average closer to another cluster than its own', 'the point is an outlier far from everything', 'the cluster is very tight', 'k is too small'], answer: 0, why: 'b < a: the nearest other cluster is closer on average.' },
      { q: 'Unlike the inertia, the mean silhouette…', options: ['can fall as k grows, so it can pick k by itself', 'always rises with k', 'needs the true labels', 'only works for two clusters'], answer: 0, why: 'Splitting a real group makes b small for its points, lowering their silhouettes.' },
      { q: 'What is a point’s silhouette when it is alone in its cluster?', options: ['0, by convention', '1', '−1', 'undefined, so the point is removed'], answer: 0, why: 'With no other points in its cluster a is not defined, and s is set to 0.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
