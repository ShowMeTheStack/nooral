// Part 7, unsupervised learning: the exercises. Every number is worked out here by the engines (S.ML's k-means,
// silhouette, linkage, DBSCAN, mixtures, PCA and isolation forests, and the topics' own runs) on small fixed inputs,
// so the text, the answers and the hints always agree.
(function (S) {
  'use strict';
  const C = S.Content, ML = S.ML, WF = S.WF;
  if (!C || !ML || !WF || C.exercises.has('un-x-assign')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  const meta = (id, inp) => { const t = S.byId(id); if (!t) return []; const r = t.run(S.inputOf(t, Object.assign(S.inputOf(t), inp || {}))); return (r.frames || r).map((f) => f.meta || {}); };
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };
  const ex = (x) => C.exercise(x);
  const fv = (p) => '(' + p.map((v) => r3(v)).join(', ') + ')';

  // =========================================================================================================
  // Chapter 1: k-means
  // =========================================================================================================
  const KX = [[1, 1], [3, 2], [2, 4], [5, 5], [7, 3], [4, 4]], KC = [[2, 2], [6, 4]], KN = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'];
  const km1 = new ML.KMeans({ k: 2, init: KC, maxIter: 1 }).fit(KX);
  const kAssign = km1.trace.find((s) => s.kind === 'assign').labels, kNew = km1.trace.find((s) => s.kind === 'update').centers;
  const kText = 'Six customers of a delivery firm, on a map in kilometres: ' + KX.map((p, i) => KN[i] + ' ' + fv(p)).join(', ') + '. Two depots stand at A = (2, 2) and B = (6, 4).';
  const acells = {};
  KN.forEach((n, i) => { acells[n + '|nearest depot'] = kAssign[i] ? 'B' : 'A'; });
  ex({ id: 'un-x-assign', chapter: 'un-kmeans', title: 'Assign the customers', level: 1, kind: 'table',
    text: [kText, 'The assign step of k-means: write A or B for the depot nearest each customer.'],
    rows: KN, cols: ['nearest depot'], corner: 'customer', cells: acells, cellKind: 'text',
    hints: ['Compare squared distances; the square roots do not change which is smaller.', 'P6 (4, 4): to A it is 2² + 2² = 8, to B 2² + 0² = 4, so B.', KN.map((n) => n + ' ' + acells[n + '|nearest depot']).join(', ') + '.'],
    solution: acells, wrong: [Object.assign({}, acells, { 'P6|nearest depot': 'A' }), Object.assign({}, acells, { 'P3|nearest depot': 'B', 'P6|nearest depot': 'A' })] });

  const aPts = KX.filter((_, i) => kAssign[i] === 0);
  ex({ id: 'un-x-update', chapter: 'un-kmeans', title: 'Move the depot', level: 2, kind: 'point',
    text: [kText, 'After the assign step, depot A serves ' + KN.filter((_, i) => kAssign[i] === 0).join(', ') + '. The update step moves A to the mean of its customers. Click where it goes.'],
    plot: { points: KX, classes: kAssign, labels: KN, xr: [0, 8], yr: [0, 6], xname: 'east (km)', yname: 'north (km)', centroids: KC }, mode: 'xy',
    answer: kNew[0], tolerance: 0.25, whyNot: 'The new place is the average of the customers’ x and of their y.',
    hints: ['Depot A serves ' + aPts.map(fv).join(', ') + '.', 'The mean x is (1 + 3 + 2) / 3 = 2; the mean y is (1 + 2 + 4) / 3.', 'A moves to ' + fv(kNew[0]) + '. (B moves to ' + fv(kNew[1]) + ', the mean of its three customers.)'],
    solution: { x: kNew[0][0], y: kNew[0][1] }, wrong: [{ x: 2, y: 2 }, { x: 3, y: 2.6 }] });

  const inert = (cs) => KX.reduce((s, x, i) => s + ML.sqdist(x, cs[kAssign[i]]), 0), in0 = inert(KC), in1 = inert(kNew);
  ex({ id: 'un-x-inertia', chapter: 'un-kmeans', title: 'The inertia after one round', level: 2, kind: 'number',
    text: [kText, 'After one assign and one update, A is at ' + fv(kNew[0]) + ' and B at ' + fv(kNew[1]) + ', each still serving the same three customers. What is the inertia now, the sum of every customer’s squared distance to its depot? Give two decimals.'],
    answer: r2(in1), tolerance: 0.005, whyNot: { [String(r2(in0))]: 'That is with the depots at their old places; it was ' + r2(in0) + ' before the move.', '*': 'Add up the squared distances, not the distances.' },
    hints: ['For A: (1 − 2)² + (1 − 2.333)², and so on for its three customers.', 'A’s customers add up to ' + r2(aPts.reduce((s, x) => s + ML.sqdist(x, kNew[0]), 0)) + ' and B’s to ' + r2(KX.filter((_, i) => kAssign[i] === 1).reduce((s, x) => s + ML.sqdist(x, kNew[1]), 0)) + '.', 'The inertia is ' + r2(in1) + ', down from ' + r2(in0) + ' before the update: moving a centre to the mean can only lower it.'],
    solution: String(r2(in1)), wrong: [String(r2(in0)), String(r2(Math.sqrt(in1)))] });

  const stepRef = (X, Cs) => new ML.KMeans({ k: Cs.length, init: Cs, maxIter: 1, trace: false }).fit(X).centers;
  ex({ id: 'un-x-code-kmeans', chapter: 'un-kmeans', title: 'Write one k-means iteration', level: 3, kind: 'code',
    text: ['Write `kmeansStep(X, C)`: X is an array of points, C an array of k centres (points of the same length). Assign every point to its nearest centre (on a tie, the one that comes first in C), then return the new centres: each the mean of its points. A centre with no points stays where it was.'],
    fn: 'kmeansStep', params: ['X', 'C'], start: 'function kmeansStep(X, C) {\n  \n}', tolerance: 1e-9, ref: stepRef,
    cases: [[KX, KC], [[[0], [1], [5], [6]], [[0], [2]]], [[[0, 0], [0, 1], [10, 10]], [[0, 0], [5, 5], [100, 100]]], [[[1, 1], [2, 2], [3, 3]], [[2, 2], [2, 2]]], [[[0, 0, 0], [1, 1, 1], [9, 9, 9], [8, 8, 8]], [[1, 0, 0], [8, 9, 9]]]],
    hints: ['First the labels: for each point, the index of the centre at the smallest squared distance (use < so a tie keeps the first).', 'Then sum the points of each label and count them; a label with a count of 0 keeps its old centre.', 'function kmeansStep(X, C) {\n  const d = X[0].length, sum = C.map(() => new Array(d).fill(0)), n = C.map(() => 0);\n  for (const x of X) {\n    let b = 0, bd = Infinity;\n    C.forEach((c, k) => { let s = 0; for (let j = 0; j < d; j++) s += (x[j] - c[j]) ** 2; if (s < bd) { bd = s; b = k; } });\n    n[b]++; for (let j = 0; j < d; j++) sum[b][j] += x[j];\n  }\n  return C.map((c, k) => (n[k] ? sum[k].map((s) => s / n[k]) : c.slice()));\n}'],
    solution: 'function kmeansStep(X, C) {\n  const d = X[0].length, sum = C.map(() => new Array(d).fill(0)), n = C.map(() => 0);\n  for (const x of X) {\n    let b = 0, bd = Infinity;\n    C.forEach((c, k) => { let s = 0; for (let j = 0; j < d; j++) s += (x[j] - c[j]) ** 2; if (s < bd) { bd = s; b = k; } });\n    n[b]++; for (let j = 0; j < d; j++) sum[b][j] += x[j];\n  }\n  return C.map((c, k) => (n[k] ? sum[k].map((s) => s / n[k]) : c.slice()));\n}',
    wrong: ['function kmeansStep(X, C) { const d = X[0].length, sum = C.map(() => new Array(d).fill(0)), n = C.map(() => 0); for (const x of X) { let b = 0, bd = Infinity; C.forEach((c, k) => { let s = 0; for (let j = 0; j < d; j++) s += (x[j] - c[j]) ** 2; if (s < bd) { bd = s; b = k; } }); n[b]++; for (let j = 0; j < d; j++) sum[b][j] += x[j]; } return C.map((c, k) => sum[k].map((s) => s / Math.max(1, n[k]))); }',
      'function kmeansStep(X, C) { const d = X[0].length, sum = C.map(() => new Array(d).fill(0)), n = C.map(() => 0); for (const x of X) { let b = 0, bd = Infinity; C.forEach((c, k) => { let s = 0; for (let j = 0; j < d; j++) s += (x[j] - c[j]) ** 2; if (s <= bd) { bd = s; b = k; } }); n[b]++; for (let j = 0; j < d; j++) sum[b][j] += x[j]; } return C.map((c, k) => (n[k] ? sum[k].map((s) => s / n[k]) : c.slice())); }'] });

  const PP = [0, 1, 3, 6], d2 = PP.map((v) => ML.sqdist([v], [0])), ptot = ML.sum(d2), pFar = d2[3] / ptot;
  ex({ id: 'un-x-kpp-prob', chapter: 'un-kmeans', title: 'The chance of the next centre', level: 2, kind: 'number',
    text: ['Four points on a line, at ' + PP.join(', ') + '. k-means++ has chosen its first centre at 0. With what probability is the point at 6 picked as the second centre? Give three decimals.'],
    answer: r3(pFar), tolerance: 0.0015, whyNot: { '0.25': 'k-means++ does not pick uniformly: far points are more likely.', [String(r3(6 / 10))]: 'Use the squared distance D², not the distance.', [String(r3(36 / 45))]: 'Include every point in the total, even the one at distance 1.' },
    hints: ['Each point’s weight is D², its squared distance to the nearest centre so far.', 'The weights are ' + d2.join(', ') + ', adding up to ' + ptot + '.', 'P(6) = 36 / ' + ptot + ' = ' + r3(pFar) + '. The point at 0 has weight 0: it can never be picked twice.'], solution: String(r3(pFar)), wrong: ['0.25', String(r3(6 / 10))] });

  const el = meta('km-elbow', { set: 'three', kmax: 8 }), elI = (el[0] && el[0].I) || [], elK = (el.find((m) => m.phase === 'elbow') || {}).elbow || 3;
  ex({ id: 'un-x-elbow', chapter: 'un-kmeans', title: 'Find the elbow', level: 1, kind: 'point',
    text: ['k-means is run on 90 points for k = 1 to 8, keeping the best of four runs each time. Click the k where the curve of inertia bends: the elbow.'],
    plot: { points: elI.map((v, i) => [i + 1, r2(v)]), labels: elI.map((v, i) => 'k = ' + (i + 1)), xr: [0, 9], yr: [0, Math.ceil(elI[0] / 100) * 100 + 50], xname: 'k', yname: 'inertia', lines: [{ pts: elI.map((v, i) => [i + 1, v]) }] }, mode: 'pick',
    answer: elK - 1, whyNot: { [elK - 2]: 'The drop after this k is still large: one more centre still finds a group of its own.', [elK]: 'The fall had already slowed one step earlier.', 7: 'The inertia is always lowest at the largest k; that is not the elbow.' },
    hints: ['Look at how much each extra centre lowers the inertia.', 'From k = ' + (elK - 1) + ' to ' + elK + ' the inertia falls from ' + r2(elI[elK - 2]) + ' to ' + r2(elI[elK - 1]) + '; from ' + elK + ' to ' + (elK + 1) + ' only to ' + r2(elI[elK]) + '.', 'The elbow is at k = ' + elK + ': after it, each extra centre buys little.'],
    solution: { i: elK - 1 }, wrong: [{ i: elK - 2 }, { i: 7 }] });

  const SX = [1, 2, 4, 8, 9], sil = ML.silhouette(SX.map((v) => [v]), [0, 0, 0, 1, 1]), s4 = sil.samples[2];
  ex({ id: 'un-x-silhouette', chapter: 'un-kmeans', title: 'A silhouette by hand', level: 2, kind: 'number',
    text: ['Five houses along a road, at kilometres ' + SX.join(', ') + '. A clustering puts the first three in cluster A and the last two in cluster B. What is the silhouette of the house at 4? Give three decimals.'],
    answer: r3(s4), tolerance: 0.0015, whyNot: { [String(r3((4.5 - 2.5) / 2.5))]: 'Divide by the larger of a and b, here b.', [String(r3(2.5 / 4.5))]: 'The silhouette is (b − a) / max(a, b), not a / b.', [String(r3((4 - 2.5) / 4))]: 'b is the mean distance to every house of B: to 8 and to 9.' },
    hints: ['a is the mean distance to the other houses of its own cluster; b the mean distance to the houses of the nearest other cluster.', 'a = (3 + 2) / 2 = 2.5 and b = (4 + 5) / 2 = 4.5.', 's = (4.5 − 2.5) / 4.5 = ' + r3(s4) + '. It is the worst fit of the five: the mean over all of them is ' + r3(sil.mean) + '.'], solution: String(r3(s4)), wrong: [String(r3((4.5 - 2.5) / 2.5)), String(r3(2.5 / 4.5))] });

  ex({ id: 'un-x-sil-choice', chapter: 'un-kmeans', title: 'A negative silhouette', level: 1, kind: 'choice',
    text: ['After clustering, one point has silhouette −0.3. What does that say?'],
    options: ['On average it is closer to the points of another cluster than to those of its own: it may be in the wrong cluster', 'It is an outlier far from every cluster', 'It fits its cluster very well', 'The silhouette is broken: it cannot be negative'], answer: 0,
    whyNot: { 1: 'A far outlier can still be closer to its own cluster than to any other; the silhouette compares the two.', 2: 'A good fit has a silhouette near +1.', 3: 's = (b − a) / max(a, b) is negative whenever b < a.' },
    hints: ['s = (b − a) / max(a, b).', 'For s to be negative, b must be smaller than a.', 'b < a means the nearest other cluster is closer, on average, than the point’s own: it probably belongs there.'], solution: 0, wrong: [1, 3] });

  // =========================================================================================================
  // Chapter 2: clusters of other shapes
  // =========================================================================================================
  const LP = [0, 1, 2.4, 3.7, 5.2], LN = ['A', 'B', 'C', 'D', 'E'];
  const members = (merges, n) => { const m = LN.slice(0, n).map((x) => [x]); merges.forEach((g) => m.push(m[g.a].concat(m[g.b]))); return (t) => [m[merges[t].a], m[merges[t].b]]; };
  const setStr = (pair) => '{' + pair.map((g) => '{' + g.slice().sort().join(', ') + '}').join(', ') + '}';
  const sgl = ML.linkage(LP.map((v) => [v]), 'single'), cpl = ML.linkage(LP.map((v) => [v]), 'complete');
  const s3 = members(sgl, 5)(2), c3 = members(cpl, 5)(2);
  const lText = 'Five weather stations along a valley road, at kilometres A = 0, B = 1, C = 2.4, D = 3.7 and E = 5.2. Agglomerative clustering first merges {A, B} (distance 1), then {C, D} (distance 1.3).';
  ex({ id: 'un-x-single-next', chapter: 'un-shapes', title: 'The next merge, single linkage', level: 2, kind: 'set',
    text: [lText, 'With single linkage (the distance between clusters is their closest pair), which two clusters merge third? Write them as a set of two sets, like {{A, B}, {E}}.'],
    answer: setStr(s3), caseless: true, placeholder: 'like {{A, B}, {E}}', whyNot: 'Compare the closest pair between each two clusters still apart: {A, B}, {C, D} and {E}.',
    hints: ['Three clusters are left: {A, B}, {C, D} and {E}.', 'Single linkage: {A, B} to {C, D} is the gap B–C = 1.4; {C, D} to {E} is D–E = 1.5; {A, B} to {E} is 4.2.', 'The smallest is 1.4, so ' + setStr(s3) + ' merge, at distance ' + r3(sgl[2].dist) + '.'],
    solution: setStr(s3), wrong: [setStr(c3), '{{A, B}, {E}}'] });
  ex({ id: 'un-x-complete-next', chapter: 'un-shapes', title: 'The next merge, complete linkage', level: 2, kind: 'set',
    text: [lText, 'With complete linkage (the distance between clusters is their farthest pair), which two clusters merge third?'],
    answer: setStr(c3), caseless: true, placeholder: 'like {{A, B}, {E}}', whyNot: 'Under complete linkage two clusters are as far apart as their farthest pair.',
    hints: ['Three clusters are left: {A, B}, {C, D} and {E}.', 'Complete linkage: {A, B} to {C, D} is A–D = 3.7; {C, D} to {E} is C–E = 2.8; {A, B} to {E} is 5.2.', 'The smallest is 2.8, so ' + setStr(c3) + ' merge, at distance ' + r3(cpl[2].dist) + '. Single linkage chose differently: it only looks at the closest pair.'],
    solution: setStr(c3), wrong: [setStr(s3), '{{A, B}, {E}}'] });
  const dAD = Math.abs(LP[0] - LP[3]);
  ex({ id: 'un-x-link-dist', chapter: 'un-shapes', title: 'A distance between clusters', level: 1, kind: 'number',
    text: [lText, 'Under complete linkage, how far apart are the clusters {A, B} and {C, D}?'],
    answer: r3(dAD), tolerance: 0.0005, whyNot: { '1.4': 'That is the closest pair, B–C: single linkage. Complete takes the farthest pair.', '2.55': 'That is the distance between the clusters’ midpoints, not between their farthest points.' },
    hints: ['Complete linkage takes the largest distance between a point of one cluster and a point of the other.', 'The pairs are A–C 2.4, A–D 3.7, B–C 1.4 and B–D 2.7.', 'The largest is A–D = ' + r3(dAD) + '.'], solution: String(r3(dAD)), wrong: ['1.4', '2.55'] });

  const TX = [0, 0.5, 1, 1.5, 2.6, 3, 3.4, 5], db = new ML.DBSCAN({ eps: 0.6, minSamples: 3 }).fit(TX.map((v) => [v]));
  const trows = TX.map((v) => 'taxi at ' + v + ' km'), dcells = {};
  trows.forEach((r, i) => { dcells[r + '|kind'] = db.kinds[i]; });
  ex({ id: 'un-x-dbscan-labels', chapter: 'un-shapes', title: 'Core, border or noise', level: 3, kind: 'table',
    text: ['Eight taxis wait along a straight road, at kilometres ' + TX.join(', ') + '. DBSCAN runs with ε = 0.6 and minPts = 3 (a point counts itself). Label each taxi core, border or noise.'],
    rows: trows, cols: ['kind'], corner: 'taxi', cells: dcells, cellKind: 'text',
    hints: ['Count, for each taxi, the taxis within 0.6 km of it, itself included: core needs 3.', 'The taxi at 0.5 has 0, 0.5 and 1 within reach: core. The taxi at 0 has only 0 and 0.5: not core, but it is within ε of the core taxi at 0.5, so border.', trows.map((r) => r.replace('taxi at ', '') + ': ' + dcells[r + '|kind']).join('; ') + '. That makes ' + db.nClusters + ' clusters and one noise point.'],
    solution: dcells, wrong: [Object.assign({}, dcells, { 'taxi at 0 km|kind': 'noise' }), Object.assign({}, dcells, { 'taxi at 2.6 km|kind': 'noise', 'taxi at 1.5 km|kind': 'core' })] });

  ex({ id: 'un-x-shape-choice', chapter: 'un-shapes', title: 'A clustering for odd shapes', level: 1, kind: 'choice',
    text: ['GPS fixes of waiting taxis form a long curved line along a river bank and a ring round a square, with a few scattered fixes from passing cabs. The number of ranks is not known. Which method suits best?'],
    options: ['DBSCAN', 'k-means with k = 2', 'A Gaussian mixture with 2 round components', 'Ward linkage cut into 2 clusters'], answer: 0,
    whyNot: { 1: 'k-means draws straight borders halfway between centres and makes round clusters; it would cut the curve and the ring into pieces, and it gives every stray a cluster.', 2: 'Round components cannot follow a curve or a ring, and every point is given to some component.', 3: 'Ward, like k-means, prefers compact round clusters of similar spread.' },
    hints: ['Three needs: any shape, no k given in advance, strays left out.', 'Which method grows clusters through crowded neighbourhoods and calls the isolated points noise?', 'DBSCAN: it needs ε and minPts but no k, follows any shape, and labels the strays noise.'], solution: 0, wrong: [1, 2] });

  ex({ id: 'un-x-eps-effect', chapter: 'un-shapes', title: 'Too large an ε', level: 2, kind: 'choice',
    text: ['DBSCAN with minPts = 5 finds three clusters and some noise with ε = 0.76. What happens if ε is raised to 1.5?'],
    options: ['Clusters run together, here into one, and almost nothing is noise', 'More clusters appear, and more noise', 'Nothing changes: ε only sets the speed', 'Every point becomes noise'], answer: 0,
    whyNot: { 1: 'That is what a too small ε does: whole stretches turn into noise.', 2: 'ε decides who counts as a neighbour, so it changes the clusters.', 3: 'A larger ε gives every point more neighbours, not fewer.' },
    hints: ['A larger ε puts more points inside each circle.', 'More points become core, and core points reach across the gaps between groups.', 'In the lesson ε = 1.517 gave 1 cluster and 1 noise point: the groups ran together.'], solution: 0, wrong: [1, 3] });

  const g2 = new ML.GaussianMixture({ k: 2, covarianceType: 'full' });
  Object.assign(g2, { means: [[0], [3]], weights: [0.4, 0.6], covariances: [[[1]], [[4]]] });
  const resp = g2.predictProba([[1.5]])[0][0];
  const npdf = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v), na = npdf(1.5, 0, 1), nb = npdf(1.5, 3, 4);
  ex({ id: 'un-x-em-resp', chapter: 'un-shapes', title: 'One responsibility', level: 3, kind: 'number',
    text: ['A mixture of two Gaussians on a line: component 1 has mean 0, variance 1 and weight 0.4; component 2 has mean 3, variance 4 and weight 0.6. In the E-step, what responsibility does component 1 take for the point x = 1.5? Give three decimals.'],
    answer: r3(resp), tolerance: 0.0015, whyNot: { [String(r3(na / (na + nb)))]: 'Multiply each density by its component’s weight first.', [String(r3(Math.exp(-1.125) * 0.4 / (Math.exp(-1.125) * 0.4 + Math.exp(-0.28125) * 0.6)))]: 'The densities have different variances, so their factors 1/√(2πσ²) differ too: keep them.', '0.5': 'x = 1.5 is halfway between the means, but the components differ in width and weight.' },
    hints: ['r₁ = π₁N(x | μ₁, σ₁²) / (π₁N(x | μ₁, σ₁²) + π₂N(x | μ₂, σ₂²)), with N(x | μ, σ²) = exp(−(x − μ)²/(2σ²)) / √(2πσ²).', 'N(1.5 | 0, 1) = ' + r4(na) + ' and N(1.5 | 3, 4) = ' + r4(nb) + '. Times the weights: ' + r4(0.4 * na) + ' and ' + r4(0.6 * nb) + '.', 'r₁ = ' + r4(0.4 * na) + ' / (' + r4(0.4 * na) + ' + ' + r4(0.6 * nb) + ') = ' + r3(resp) + '. The wide second component claims most of the point, though both means are 1.5 away.'],
    solution: String(r3(resp)), wrong: [String(r3(na / (na + nb))), '0.5'] });

  const EX = [0, 1, 2, 6], ER = [0.9, 0.8, 0.6, 0.1], emu = ML.sum(EX.map((x, i) => x * ER[i])) / ML.sum(ER);
  ex({ id: 'un-x-em-mean', chapter: 'un-shapes', title: 'One M-step', level: 3, kind: 'number',
    text: ['Four points at ' + EX.join(', ') + '. After an E-step, component 1 takes responsibilities ' + ER.join(', ') + ' for them. In the M-step, where does component 1’s mean move? Give three decimals.'],
    answer: r3(emu), tolerance: 0.0015, whyNot: { '2.25': 'That is the plain mean of the points; weight each by its responsibility.', [String(r3(ML.sum(EX.map((x, i) => x * ER[i]))))]: 'Divide by the total responsibility, ' + r2(ML.sum(ER)) + '.', [String(r3(ML.sum(EX.map((x, i) => x * ER[i])) / 4))]: 'Divide by the total responsibility, not by the number of points.' },
    hints: ['The new mean is the responsibility-weighted mean: Σ rᵢxᵢ / Σ rᵢ.', 'Σ rᵢxᵢ = 0 + 0.8 + 1.2 + 0.6 = 2.6, and Σ rᵢ = 2.4.', 'μ₁ = 2.6 / 2.4 = ' + r3(emu) + '. The far point at 6 barely pulls it: its responsibility is 0.1.'], solution: String(r3(emu)), wrong: ['2.25', String(r3(ML.sum(EX.map((x, i) => x * ER[i]))))] });

  const LL = -150.2, NP = 11, NN = 120, bic = -2 * LL + NP * Math.log(NN);
  ex({ id: 'un-x-bic', chapter: 'un-shapes', title: 'A BIC by hand', level: 2, kind: 'number',
    text: ['A mixture of two Gaussians in two dimensions with full covariances has 11 parameters. Fitted to 120 points, its log-likelihood is ' + LL + '. What is its BIC, −2 · log-likelihood + parameters · ln n? Give one decimal.'],
    answer: Math.round(bic * 10) / 10, tolerance: 0.06, whyNot: { [String(Math.round((-2 * LL + 2 * NP) * 10) / 10)]: 'That charges 2 per parameter: it is the AIC. BIC charges ln n each.', [String(Math.round((2 * LL + NP * Math.log(NN)) * 10) / 10)]: 'Mind the sign: −2 times a negative log-likelihood is positive.' },
    hints: ['−2 · (−150.2) = 300.4.', 'ln 120 = ' + r3(Math.log(NN)) + ', so the charge is 11 × ' + r3(Math.log(NN)) + ' = ' + r2(NP * Math.log(NN)) + '.', 'BIC = 300.4 + ' + r2(NP * Math.log(NN)) + ' = ' + Math.round(bic * 10) / 10 + '. A model with more components must raise the log-likelihood by enough to pay for its extra parameters.'],
    solution: String(Math.round(bic * 10) / 10), wrong: [String(Math.round((-2 * LL + 2 * NP) * 10) / 10), String(Math.round((2 * LL + NP * Math.log(NN)) * 10) / 10)] });

  ex({ id: 'un-x-gmm-choice', chapter: 'un-shapes', title: 'Long, tilted groups', level: 2, kind: 'choice',
    text: ['Three groups of points are long, tilted ellipses that nearly touch, and the analyst wants to know how sure the method is about the points where they meet. Which method suits best?'],
    options: ['A Gaussian mixture with full covariances', 'k-means', 'A Gaussian mixture with spherical covariances', 'Single linkage'], answer: 0,
    whyNot: { 1: 'k-means makes round clusters and gives only hard labels: no measure of how sure it is.', 2: 'Spherical components are circles: they cannot stretch along a tilted group. In the lesson BIC preferred full covariances, 495.5 against 546.3.', 3: 'Single linkage would chain the groups together where they nearly touch, and gives hard labels.' },
    hints: ['Two needs: elliptical, tilted shapes, and a degree of belief for each point.', 'Which method gives responsibilities that add up to 1 for every point?', 'A Gaussian mixture; with full covariances each component can be any tilted ellipse.'], solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 3: fewer dimensions
  // =========================================================================================================
  const CV = [[3, 1], [1, 3]], eg = ML.la.symEig(CV), ratio1 = eg.values[0] / ML.sum(eg.values);
  ex({ id: 'un-x-pca-ratio', chapter: 'un-reduce', title: 'An explained-variance ratio', level: 1, kind: 'number',
    text: ['Two measurements have covariance matrix [[3, 1], [1, 3]]: variances 3 and 3, covariance 1. Its eigenvalues are ' + r3(eg.values[0]) + ' and ' + r3(eg.values[1]) + '. What share of the variance does the first principal component explain? Give three decimals.'],
    answer: r3(ratio1), tolerance: 0.0015, whyNot: { '0.5': 'Each feature holds half the variance, but PC1 holds more than either feature alone.', '0.75': 'Divide by the total of the eigenvalues, 4 + 2, not by the first variance plus one.', '4': 'That is PC1’s variance; the ratio divides it by the total.' },
    hints: ['The explained variance ratio of PC k is λ_k / Σ λ.', 'The total variance is 4 + 2 = 6, the same as 3 + 3 on the diagonal.', '4 / 6 = ' + r3(ratio1) + '. PC1 = (0.707, 0.707): the two measurements together.'], solution: String(r3(ratio1)), wrong: ['0.5', '4'] });

  const PM = [2, 1], PV = [0.8, 0.6], PXp = [5, 3], score = PV[0] * (PXp[0] - PM[0]) + PV[1] * (PXp[1] - PM[1]);
  ex({ id: 'un-x-pca-score', chapter: 'un-reduce', title: 'A point’s score on PC1', level: 2, kind: 'number',
    text: ['PCA on a table of two features finds the mean (2, 1) and PC1 = (0.8, 0.6). What is the score of the point (5, 3) on PC1, the one number that describes it when only PC1 is kept?'],
    answer: r3(score), tolerance: 0.0005, whyNot: { '5.8': 'Centre the point first: subtract the mean.', '6.2': 'Subtract the mean (2, 1) before taking the dot product.' },
    hints: ['Centre the point: subtract the mean.', '(5, 3) − (2, 1) = (3, 2).', 'The score is the dot product with PC1: 0.8 · 3 + 0.6 · 2 = ' + r3(score) + '.'], solution: String(r3(score)), wrong: ['5.8', '6.2'] });

  const topEig = (A) => { const e = ML.la.symEig(A); return ML.signNormalise([e.vectors.map((r) => r[0])])[0]; };
  ex({ id: 'un-x-code-power', chapter: 'un-reduce', title: 'Power iteration for PC1', level: 3, kind: 'code',
    text: ['Write `firstComponent(S)`: S is a symmetric covariance matrix (an array of rows) whose largest eigenvalue is well clear of the others. Find its top eigenvector by power iteration: start from a vector of ones, then repeatedly multiply by S and scale to length 1. Return the unit vector, with the sign chosen so that its entry of largest size is positive.'],
    fn: 'firstComponent', params: ['S'], start: 'function firstComponent(S) {\n  \n}', tolerance: 1e-6, ref: topEig,
    cases: [[[[3, 1], [1, 3]]], [[[1.989, 1.313], [1.313, 1.33]]], [[[4, 0], [0, 1]]], [[[2, -1, 0], [-1, 2, -1], [0, -1, 2]]], [[[5, 2, 1], [2, 4, 1], [1, 1, 1]]], [[[1, 0.9], [0.9, 1]]]],
    hints: ['One step: w = S·v (each entry the dot product of a row of S with v), then v = w / ‖w‖.', 'A few hundred steps are plenty when the top eigenvalue stands clear. At the end, find the entry of largest absolute value and flip the vector if it is negative.', 'function firstComponent(S) {\n  const n = S.length;\n  let v = new Array(n).fill(1);\n  for (let it = 0; it < 1000; it++) {\n    const w = S.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));\n    const len = Math.hypot(...w);\n    v = w.map((x) => x / len);\n  }\n  let b = 0;\n  v.forEach((x, j) => { if (Math.abs(x) > Math.abs(v[b])) b = j; });\n  return v[b] < 0 ? v.map((x) => -x) : v;\n}'],
    solution: 'function firstComponent(S) {\n  const n = S.length;\n  let v = new Array(n).fill(1);\n  for (let it = 0; it < 1000; it++) {\n    const w = S.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));\n    const len = Math.hypot(...w);\n    v = w.map((x) => x / len);\n  }\n  let b = 0;\n  v.forEach((x, j) => { if (Math.abs(x) > Math.abs(v[b])) b = j; });\n  return v[b] < 0 ? v.map((x) => -x) : v;\n}',
    wrong: ['function firstComponent(S) { const n = S.length; let v = new Array(n).fill(1); for (let it = 0; it < 1000; it++) { v = S.map((row) => row.reduce((s, x, j) => s + x * v[j], 0)); const m = Math.max(...v.map(Math.abs)); v = v.map((x) => x / m); } return v; }',
      'function firstComponent(S) { const n = S.length; let v = new Array(n).fill(1); for (let it = 0; it < 3; it++) { const w = S.map((row) => row.reduce((s, x, j) => s + x * v[j], 0)); const len = Math.hypot(...w); v = w.map((x) => x / len); } return v; }'] });

  const pv = meta('pca-variance', { set: 'wine' }).find((m) => m.phase === 'scree') || { r: [], cum: [] };
  const need90 = pv.cum.findIndex((c) => c >= 0.9) + 1;
  ex({ id: 'un-x-pca-k', chapter: 'un-reduce', title: 'Components for 90%', level: 2, kind: 'number',
    text: ['PCA of 178 wines with 13 standardised features gives these explained variance ratios, in order:', { table: [['PC'].concat(pv.r.map((_, i) => String(i + 1))), ['ratio'].concat(pv.r.map((v) => String(r3(v))))] }, 'How many components must be kept to explain at least 90% of the variance?'],
    answer: need90, tolerance: 1e-9, whyNot: { '1': 'PC1 explains only ' + Math.round(100 * pv.r[0]) + '%.', [String(need90 - 1)]: 'The running total at that point is ' + r3(pv.cum[need90 - 2]) + ': just short of 0.9.', '13': 'All 13 explain everything, but far fewer pass 90%.' },
    hints: ['Add the ratios one by one: the running total.', 'After 5 components the total is ' + r3(pv.cum[4]) + '; after 7 it is ' + r3(pv.cum[6]) + '.', 'The total first reaches 0.9 at ' + need90 + ' components (' + r3(pv.cum[need90 - 1]) + ').'], solution: String(need90), wrong: [String(need90 - 1), '13'] });

  const cum = pv.cum.map((c, i) => [i + 1, r3(c)]);
  ex({ id: 'un-x-cumvar-curve', chapter: 'un-reduce', title: 'The running total of variance', level: 2, kind: 'predict-curve',
    text: ['For the same 178 wines, sketch the running total of explained variance against the number of components kept, from 1 to 13.'],
    plot: { xr: [0, 14], yr: [0, 1.05], xname: 'components kept', yname: 'variance explained', given: cum.slice(0, 1), note: 'Drag from 1 to 13 components. The faint point is PC1 alone.' },
    answer: cum, tolerance: 0.05, checks: ['rising'],
    hints: ['It starts at PC1’s share and must end at 1, with all 13.', 'Each component adds less than the one before, so the curve rises steeply and then flattens.', 'It goes ' + cum.filter((_, i) => [0, 1, 2, 4, 7, 12].includes(i)).map(([k, v]) => k + ': ' + v).join(', ') + '.'],
    solution: cum, wrong: [cum.map(([k]) => [k, r3(k / 13)]), cum.map(([k]) => [k, 0.9])] });

  const flw = { data: { set: 'wildflowers', n: 300, seed: 5 }, hidden: { seed: 55, n: 300 }, metric: 'accuracy', target: 0.95, task: 'classification' };
  C.exercise(Object.assign({ id: 'un-x-wf-pca', chapter: 'un-reduce', title: 'Two numbers per flower', level: 2, kind: 'workflow',
    text: ['The botany club wants to file every wildflower under two numbers instead of four measurements. Build a workflow that reduces the table to at most 2 principal components and names the species of 300 hidden flowers by k-nearest neighbours with accuracy at least 0.95.'],
    limits: { blocks: ['split', 'scale', 'pca', 'knn', 'train', 'evaluate'], models: ['knn'], params: { pca: { n: [1, 2] } } },
    hints: ['Put a PCA block between the split and Train, wiring the test rows through it too, and open it: which feature dominates PC1?', 'PCA follows variance, and the stem heights are by far the largest numbers: PC1 is mostly stem height. Scale first.', 'Split → Scale (standard) → PCA (2 components) → 5-NN → Train → Evaluate: accuracy 0.97, against 0.897 without the Scale block.'],
    solution: pipe(flw.data, 'knn', { k: 5 }, [['scale', { how: 'standard' }], ['pca', { n: 2 }]]), wrong: [pipe(flw.data, 'knn', { k: 5 }, [['pca', { n: 2 }]]), pipe(flw.data, 'knn', { k: 5 }, [['scale', { how: 'standard' }], ['pca', { n: 1 }]])] }, flw));

  ex({ id: 'un-x-map-choice', chapter: 'un-reduce', title: 'Reading a t-SNE map', level: 1, kind: 'choice',
    text: ['A t-SNE map of customer records shows three clusters: A is large and spread out, B and C are small, and B sits much closer to A than to C. Which conclusion is safe?'],
    options: ['Records that sit together on the map are probably similar in the data', 'A is the most varied group of customers', 'B is more like A than like C', 'The groups have three different sizes in the data'], answer: 0,
    whyNot: { 1: 't-SNE adapts its scale to each neighbourhood: a cluster’s size on the map says little about its spread in the data.', 2: 'Gaps between clusters on a t-SNE map mean little.', 3: 'Area on the map is not a count of records; count the points.' },
    hints: ['What does t-SNE try to keep?', 'It matches neighbour probabilities: who is near whom.', 'Only local closeness is kept: points together on the map were neighbours in the data. Sizes and gaps are not to be read.'], solution: 0, wrong: [1, 2] });

  ex({ id: 'un-x-perplexity', chapter: 'un-reduce', title: 'A tiny perplexity', level: 2, kind: 'choice',
    text: ['The same 48 points are mapped by t-SNE with perplexity 12 and with perplexity 2. With 2, the trustworthiness drops from 0.984 to 0.849 and the groups break into fragments. Why?'],
    options: ['Each point then attends to only about 2 neighbours, so small accidental clumps look as strong as the real groups', 'Perplexity 2 means two dimensions, too few for the map', 'The optimisation ran for fewer iterations', 'A small perplexity exaggerates the attractions'], answer: 0,
    whyNot: { 1: 'The map is 2-D either way; perplexity is about the number of neighbours.', 2: 'Both runs had the same 300 iterations.', 3: 'The early exaggeration is a separate factor, the same in both runs.' },
    hints: ['Perplexity is roughly the number of effective neighbours each point has.', 'With 2, each point cares about its one or two closest points only.', 'Tiny neighbourhoods keep pairs and triples together but lose the larger groups, so the groups fragment.'], solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 4: odd ones out
  // =========================================================================================================
  const IH = 6, IN = 60, ic = ML.isoC(IN), iscore = Math.pow(2, -IH / ic);
  ex({ id: 'un-x-iforest-score', chapter: 'un-anomaly', title: 'An anomaly score', level: 2, kind: 'number',
    text: ['An isolation forest is grown on samples of 60 points, for which the average path length of a search is c(60) = ' + r3(ic) + '. A reading needs ' + IH + ' cuts on average to be isolated. What is its score 2^(−E[h] / c)? Give three decimals.'],
    answer: r3(iscore), tolerance: 0.0015, whyNot: { [String(r3(IH / ic))]: 'That is the exponent with its sign dropped; the score is 2 to the power of minus it.', [String(r3(Math.exp(-IH / ic)))]: 'The base is 2, not e.' },
    hints: ['The exponent is −6 / ' + r3(ic) + ' = ' + r3(-IH / ic) + '.', '2^' + r3(-IH / ic) + ' = e^(' + r3(-IH / ic) + ' · ln 2).', 'The score is ' + r3(iscore) + ': a little above one half, so only mildly unusual. A point needing 4.2 cuts scored 0.673 in the lesson.'], solution: String(r3(iscore)), wrong: [String(r3(IH / ic)), String(r3(Math.exp(-IH / ic)))] });

  ex({ id: 'un-x-iforest-choice', chapter: 'un-anomaly', title: 'Why the odd ones fall first', level: 1, kind: 'choice',
    text: ['Why does a random cut isolate an odd point after fewer cuts, on average, than a point in a crowd?'],
    options: ['A random cut value lands in the wide empty gap around the odd point far more often than between two crowded points', 'The forest is trained on labelled anomalies', 'Odd points have larger feature values', 'The cuts are aimed at the odd points'], answer: 0,
    whyNot: { 1: 'No labels are used: the forest is unsupervised.', 2: 'An odd point can have small values; what matters is that few points are near it.', 3: 'Every cut is random: a random feature and a random value in its range.' },
    hints: ['The cut value is drawn at random within the cell’s range.', 'How likely is a random value to fall between the odd point and the rest, compared with between two crowded neighbours?', 'The gap round an odd point is wide, so a random cut lands there often and isolates it; crowded points are separated by tiny gaps that random cuts rarely hit.'], solution: 0, wrong: [1, 3] });
})(globalThis.Nooral = globalThis.Nooral || {});
