// Part 4, nearest neighbours and probabilistic models: the exercises. Every number in a text, an answer or a hint is
// worked out here by the engines (S.ML: k-NN, the k-d tree, naive Bayes, LDA; S.Data; the workflow runner for the
// workflow tasks) from the small datasets written out below, so nothing is typed from memory.
(function (S) {
  'use strict';
  const C = S.Content, ML = S.ML, WF = S.WF;
  if (!C || !ML || !WF || C.exercises.has('nn-x-nearest-pick')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const n2 = (v) => String(r2(v)), n3 = (v) => String(r3(v));
  const m3 = (v) => n3(v).replace('-', '−'), sg = (v) => (v < 0 ? '− ' + Math.abs(v) : '+ ' + v);
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };
  const fit = (Model, p, X, y) => new ML[Model](p || {}).fit(X, y);

  // =========================================================================================================
  // Chapter 1: k nearest neighbours
  // =========================================================================================================
  // ---- the nearest point under Manhattan distance ------------------------------------------------------------
  const NP = [[5, 4.5], [1.2, 3], [1, 0.5], [6, 1], [3, 5.8]], NC = [1, 0, 0, 1, 1], NQ = [3.5, 3];
  const nd1 = NP.map((p) => ML.dist(p, NQ, 1)), nd2 = NP.map((p) => ML.dist(p, NQ, 2));
  const nM = ML.argmin(nd1), nE = ML.argmin(nd2);
  C.exercise({ id: 'nn-x-nearest-pick', chapter: 'nn-knn', title: 'The nearest point, block by block', level: 2, kind: 'point',
    text: ['A 1-NN model measures distance the Manhattan way: the sum of the gaps along each axis, as a taxi drives through a grid of streets. The query is the cross at (' + NQ.join(', ') + '). Click the training point it copies.'],
    plot: { points: NP, classes: NC, xr: [0, 7], yr: [0, 7], xname: 'x₁', yname: 'x₂', centroids: [NQ], labels: NP.map((p) => 'the point at (' + p.join(', ') + ')') }, mode: 'pick',
    answer: nM, whyNot: { [nE]: 'In a straight line it is the nearest (' + n2(nd2[nE]) + '), but Manhattan distance adds the gaps: ' + n2(nd1[nE]) + '.' },
    hints: ['Manhattan distance: |x₁ − 3.5| + |x₂ − 3|.', 'The point up and to the right is ' + n2(nd2[nE]) + ' away in a straight line, but ' + n2(nd1[nE]) + ' by Manhattan.', 'The point at (' + NP[nM].join(', ') + ') is ' + n2(nd1[nM]) + ' away by Manhattan, the least of the five: ' + nd1.map(n2).join(', ') + '.'],
    solution: { i: nM }, wrong: [{ i: nE }, { i: 4 }] });

  // ---- a vote of 3 and of 5 ----------------------------------------------------------------------------------------
  const VD = [0.6, 0.9, 1.1, 1.4, 1.5, 1.9, 2.2], VC = ['orange', 'blue', 'blue', 'orange', 'orange', 'orange', 'blue'];
  const vote = (k, w) => fit('KNeighborsClassifier', { k, weights: w || 'uniform' }, VD.map((d) => [d]), VC).predict([[0]])[0];
  const v3 = vote(3), v5 = vote(5), vopts = ['blue, then blue', 'blue, then orange', 'orange, then blue', 'orange, then orange'];
  C.exercise({ id: 'nn-x-vote', chapter: 'nn-knn', title: 'Three voters, then five', level: 1, kind: 'choice',
    text: ['The seven training points nearest a new example, nearest first:', { table: [['Neighbour', '1', '2', '3', '4', '5', '6', '7'], ['Distance', ...VD.map(String)], ['Class', ...VC]] }, 'What does k-NN predict with k = 3, and then with k = 5?'],
    options: vopts, answer: vopts.indexOf(v3 + ', then ' + v5),
    whyNot: { 3: 'With k = 3 only the first three vote, and two of them are blue.', 0: 'With k = 5 the fourth and fifth neighbours join, both orange.', 2: 'The nearest neighbour alone is orange, but k = 3 asks three.' },
    hints: ['Take the first k rows of the table and count the classes.', 'k = 3: orange, blue, blue.', 'k = 3 gives ' + v3 + ' (2 of 3); k = 5 gives ' + v5 + ' (3 of 5).'], solution: vopts.indexOf(v3 + ', then ' + v5), wrong: [3, 0] });

  // ---- three distances --------------------------------------------------------------------------------------------
  const MQ = [1, 2], MP = [[4, 6], [3, -1], [-2, 2.5]];
  const MD = MP.map((p) => [2, 1, Infinity].map((q) => r2(ML.dist(p, MQ, q)))), MR = MP.map((p, i) => 'p' + (i + 1) + ' = (' + p.join(', ') + ')'), MCo = ['Euclidean', 'Manhattan', 'Chebyshev'];
  const mcells = (f) => Object.fromEntries(MR.flatMap((r, i) => MCo.map((c, j) => [r + '|' + c, f(i, j)])));
  C.exercise({ id: 'nn-x-dist-matrix', chapter: 'nn-knn', title: 'One gap, three distances', level: 1, kind: 'table',
    text: ['The query is q = (1, 2). Fill in the distance from q to each point under each metric, to two decimal places.'],
    rows: MR, cols: MCo, corner: 'point', cells: mcells((i, j) => MD[i][j]), cellKind: 'number', tolerance: 0.011,
    hints: ['For each point first write the two gaps |Δx| and |Δy|.', 'p1: the gaps are 3 and 4, so Euclidean √(9 + 16) = 5, Manhattan 3 + 4 = 7, Chebyshev max(3, 4) = 4.', 'Row by row: ' + MD.map((r, i) => 'p' + (i + 1) + ': ' + r.join(', ')).join('; ') + '.'],
    solution: mcells((i, j) => String(MD[i][j])), wrong: [mcells((i, j) => String(MD[i][[1, 0, 2][j]])), mcells((i, j) => String(j === 0 ? r2(ML.dist(MP[i], MQ, 2) ** 2) : MD[i][j]))] });

  // ---- leave-one-out 1-NN as a confusion matrix ----------------------------------------------------------------
  const LX = [0.5, 1.1, 1.8, 2.3, 2.9, 3.4, 4.2, 4.6, 5.3, 6.1], LY = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1];
  const loo = LX.map((_, i) => fit('KNeighborsClassifier', { k: 1 }, LX.filter((_, j) => j !== i).map((v) => [v]), LY.filter((_, j) => j !== i)).predict([[LX[i]]])[0]);
  const LCM = [[0, 0], [0, 0]]; LY.forEach((t, i) => LCM[t][loo[i]]++);
  C.exercise({ id: 'nn-x-loo-matrix', chapter: 'nn-knn', title: 'Leave one out', level: 2, kind: 'matrix', diag: true,
    text: ['1-NN on one feature. Each of the ten training points is held out in turn and predicted from its nearest neighbour among the other nine:', { table: [['Point', ...LX.map((_, i) => String(i + 1))], ['x', ...LX.map(String)], ['Class', ...LY.map(String)]] }, 'Fill in the confusion matrix of these ten held-out predictions.'],
    rows: ['class 0', 'class 1'], cols: ['said 0', 'said 1'], corner: 'true ↓  said →', answer: LCM, total: 10,
    hints: ['For each point, find the nearest of the other nine and copy its class.', 'Point 3 (x = 1.8) is nearest to point 4 (x = 2.3), class 0: a class-1 point predicted 0.', 'The held-out predictions are ' + loo.join(', ') + '. Row by row: ' + LCM.map((r) => r.join(', ')).join('; ') + '.'],
    solution: LCM.map((r) => r.map(String)), wrong: [[[String(LY.filter((v) => v === 0).length), '0'], ['0', String(LY.filter((v) => v === 1).length)]], LCM.map((r) => r.slice().reverse().map(String))] });

  // ---- the three nearest under Manhattan ---------------------------------------------------------------------------
  const SN = ['A', 'B', 'C', 'D', 'E'], SP = [[3, 0], [2, 2], [0, -2.6], [-1.5, 1.6], [2.5, -1.2]];
  const near3 = (q) => SP.map((p, i) => [ML.dist(p, [0, 0], q), i]).sort((a, b) => a[0] - b[0]).slice(0, 3).map((x) => SN[x[1]]).sort();
  const sM = near3(1), sE = near3(2);
  C.exercise({ id: 'nn-x-metric-set', chapter: 'nn-knn', title: 'Three nearest by Manhattan', level: 2, kind: 'set',
    text: ['The query is the origin (0, 0). Five training points: ' + SP.map((p, i) => SN[i] + ' = (' + p.join(', ') + ')').join(', ') + '.', 'Which three are the nearest under Manhattan distance? Write them as a set, like {A, B, C}.'],
    answer: '{' + sM.join(', ') + '}', caseless: true, placeholder: 'like {A, B, C}',
    hints: ['Manhattan distance from the origin is |x| + |y|.', 'The distances: ' + SP.map((p, i) => SN[i] + ' ' + n2(ML.dist(p, [0, 0], 1))).join(', ') + '.', 'The three smallest are ' + sM.join(', ') + '. (Straight-line distance would choose ' + sE.join(', ') + ' instead.)'],
    solution: '{' + sM.join(', ') + '}', wrong: ['{' + sE.join(', ') + '}', '{A, B, C}'] });

  // ---- which k overfits ----------------------------------------------------------------------------------------
  C.exercise({ id: 'nn-x-k-overfit', chapter: 'nn-knn', title: 'Which k overfits', level: 1, kind: 'choice',
    text: ['k-NN is trained on 80 noisy points from two interleaved moons and scored on fresh points. Which choice of k is the most likely to overfit: a boundary with small islands around single stray points, and a training error far below its test error?'],
    options: ['k = 1', 'k = 7', 'k = 31', 'k = 79, almost every point'], answer: 0,
    whyNot: { 1: 'Seven voters already outvote a single stray point.', 2: 'Thirty-one voters smooth the map: this errs towards underfitting.', 3: 'Nearly every point votes everywhere: the map is almost one colour, which is underfitting.' },
    hints: ['Overfitting means following the noise of the training points.', 'With k = 1 each training point predicts itself, so the training error is 0.', 'k = 1: every stray point gets an island of its own, the training error is 0 and the test error is higher.'], solution: 0, wrong: [2, 3] });

  // ---- test error as k grows ------------------------------------------------------------------------------------
  // k-NN trained on 120 moons (noise 0.35, seed 3) and scored on 300 fresh ones (seed 31), for each k: printed by the
  // engines (S.Data, ML.KNeighborsClassifier) in a scratch script and kept here, as eleven fits cost more than a page
  // load should.
  const kErr = [[1, 0.147], [3, 0.127], [5, 0.13], [9, 0.133], [15, 0.13], [25, 0.13], [41, 0.153], [61, 0.203], [81, 0.24], [101, 0.25], [119, 0.297]];
  const kBest = kErr.reduce((a, b) => (b[1] < a[1] ? b : a));
  C.exercise({ id: 'nn-x-k-curve', chapter: 'nn-knn', title: 'Test error as k grows', level: 2, kind: 'predict-curve',
    text: ['k-NN stores 120 noisy moon points and is scored on 300 fresh ones. The errors at k = 1 and k = 3 are drawn. Sketch the test error as k grows to 119, nearly every stored point.'],
    plot: { xr: [0, 120], yr: [0, 0.5], xname: 'k', yname: 'test error', given: kErr.slice(0, 2), note: 'Drag from k = 1 to k = 119.' },
    answer: kErr, tolerance: 0.04, checks: ['valley'],
    hints: ['Small k follows the noise; very large k ignores the shape of the moons.', 'The error dips to its lowest, ' + kBest[1] + ', at k = ' + kBest[0] + ', and stays near it while k is small against the 120 points.', 'Past k = 25 it climbs, as the vote starts to cut across the moons: ' + kErr.map((p) => 'k = ' + p[0] + ': ' + p[1]).join(', ') + '.'],
    solution: kErr, wrong: [kErr.map(([k]) => [k, kErr[0][1]]), kErr.map(([k]) => [k, r3(kErr[0][1] * (1 - k / 130))])] });

  // ---- weighted votes ---------------------------------------------------------------------------------------------
  const WD = [0.5, 1, 1.25, 2, 2.5], WC = ['blue', 'orange', 'orange', 'blue', 'orange'];
  const wm = (w) => fit('KNeighborsClassifier', { k: 5, weights: w }, WD.map((d) => [d]), WC);
  const pU = wm('uniform').predictProba([[0]])[0], pD = wm('distance').predictProba([[0]])[0];
  const io = wm('uniform').classes.indexOf('orange'), wU = wm('uniform').predict([[0]])[0], wDp = wm('distance').predict([[0]])[0];
  const wTable = { table: [['Neighbour', '1', '2', '3', '4', '5'], ['Distance', ...WD.map(String)], ['Class', ...WC]] };
  C.exercise({ id: 'nn-x-weighted', chapter: 'nn-knn', title: 'A vote weighted by distance', level: 2, kind: 'number',
    text: ['The five nearest neighbours of a query:', wTable, 'With distance weights each neighbour counts 1 / distance. What probability does 5-NN give to orange? Three decimal places.'],
    answer: r3(pD[io]), tolerance: 0.0015, whyNot: { [n3(pU[io])]: 'That is the uniform vote, 3 of 5. Weight each neighbour by 1 / distance first.', [n3(1 - pD[io])]: 'That is the probability of blue.' },
    hints: ['The weights are 1/0.5, 1/1, 1/1.25, 1/2 and 1/2.5.', 'Blue: 2 + 0.5 = 2.5. Orange: 1 + 0.8 + 0.4 = 2.2.', 'P(orange) = 2.2 / (2.5 + 2.2) = ' + n3(pD[io]) + '.'], solution: n3(pD[io]), wrong: [n3(pU[io]), n3(1 - pD[io])] });
  const fopts = ['uniform: ' + wU + '; distance: ' + wDp, 'uniform: ' + wDp + '; distance: ' + wU, 'both ' + wU, 'both ' + wDp];
  C.exercise({ id: 'nn-x-weighted-flip', chapter: 'nn-knn', title: 'When weighting changes the answer', level: 2, kind: 'choice',
    text: ['The same five neighbours:', wTable, 'What does 5-NN predict with uniform weights, and with distance weights?'],
    options: fopts, answer: 0,
    whyNot: { 1: 'Uniform weights simply count: three orange against two blue.', 2: 'With distance weights the blue neighbour at 0.5 counts 2 on its own.', 3: 'Uniform weights give one vote each: orange has three.' },
    hints: ['Uniform: count the classes.', 'Distance: blue gets 1/0.5 + 1/2 = 2.5, orange 1 + 0.8 + 0.4 = 2.2.', 'Uniform says ' + wU + ' (3 to 2), distance says ' + wDp + ' (2.5 to 2.2): the very close blue neighbour outweighs two far orange ones.'], solution: 0, wrong: [1, 2] });
  C.exercise({ id: 'nn-x-train-distance', chapter: 'nn-knn', title: 'A suspicious training score', level: 2, kind: 'choice',
    text: ['A colleague trains 15-NN with distance weights on 500 rows and reports 100% accuracy on those same rows. What explains it?'],
    options: ['The data are perfectly separable', 'Each row is at distance 0 from itself and takes the whole vote', 'Fifteen neighbours always fit the training set', 'Distance weights remove all noise'], answer: 1,
    whyNot: { 0: 'Noisy data give 100% too: the score says nothing about the data.', 2: 'With uniform weights 15-NN usually misses some training rows.', 3: 'Weights change who is heard, not the labels.' },
    hints: ['Ask what the nearest neighbour of a training row is.', 'It is the row itself, at distance 0, and a weight of 1/0 swamps every other vote.', 'Each row is at distance 0 from itself and takes the whole vote, so the training accuracy is 100% whatever k is. Score it on held-out rows.'], solution: 1, wrong: [0, 2] });

  // ---- units ------------------------------------------------------------------------------------------------------
  const UA = [42, 0.2], UB = [45, 0.65], ud = ML.dist(UA, UB, 2), share = 9 / ud ** 2, sdI = 12, sdD = 0.15;
  const sShare = (3 / sdI) ** 2 / ((3 / sdI) ** 2 + (0.45 / sdD) ** 2);
  C.exercise({ id: 'nn-x-scale-share', chapter: 'nn-knn', title: 'Which feature decides the distance', level: 2, kind: 'number',
    text: ['Two loan applicants: A has income 42 (thousand) and debt ratio 0.20; B has income 45 and debt ratio 0.65. k-NN measures the Euclidean distance between them on the raw numbers.', 'What share of the squared distance comes from income? Three decimal places.'],
    answer: r3(share), tolerance: 0.0015, whyNot: { [n3(1 - share)]: 'That is the share of the debt ratio.', [n3(sShare)]: 'That is the share after standardising; the question is about the raw numbers.' },
    hints: ['The squared distance is (45 − 42)² + (0.65 − 0.20)².', 'That is 9 + 0.2025 = 9.2025.', '9 / 9.2025 = ' + n3(share) + ': income decides almost alone. Standardised (income spread ' + sdI + ', debt spread ' + sdD + '), the gaps become 0.25 and 3 and income’s share falls to ' + n3(sShare) + '.'],
    solution: n3(share), wrong: [n3(1 - share), n3(sShare)] });

  // ---- code -------------------------------------------------------------------------------------------------------
  C.exercise({ id: 'nn-x-code-dist', chapter: 'nn-knn', title: 'Write the distance', level: 1, kind: 'code',
    text: ['Write `distance(a, b, p)` for two points given as arrays of numbers: Manhattan when p is 1, Euclidean when p is 2, and Chebyshev (the largest gap) when p is `Infinity`.'],
    fn: 'distance', params: ['a', 'b', 'p'], start: 'function distance(a, b, p) {\n  \n}', tolerance: 1e-9,
    ref: (a, b, p) => ML.dist(a, b, p), cases: [[[0, 0], [3, 4], 2], [[0, 0], [3, 4], 1], [[0, 0], [3, 4], Infinity], [[1, -2, 5], [4, 2, 5], 2], [[1.5, 2], [-1, 0.5], 1], [[2, 7, -1], [5, 1, 0], Infinity]],
    hints: ['Loop over the coordinates and work with |a[i] − b[i]|.', 'p = 1: add the gaps. p = 2: add the squares and take the square root. Infinity: keep the largest gap.', 'function distance(a, b, p) {\n  let s = 0;\n  for (let i = 0; i < a.length; i++) {\n    const g = Math.abs(a[i] - b[i]);\n    if (p === Infinity) s = Math.max(s, g); else s += g ** p;\n  }\n  return p === Infinity ? s : s ** (1 / p);\n}'],
    solution: 'function distance(a, b, p) {\n  let s = 0;\n  for (let i = 0; i < a.length; i++) {\n    const g = Math.abs(a[i] - b[i]);\n    if (p === Infinity) s = Math.max(s, g); else s += g ** p;\n  }\n  return p === Infinity ? s : s ** (1 / p);\n}',
    wrong: ['function distance(a, b, p) { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s); }', 'function distance(a, b, p) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) ** p; return s; }'] });
  const rq = ML.rng(404), kCase = (n, k) => { const X = Array.from({ length: n }, () => [r2(rq() * 10), r2(rq() * 10)]); const y = X.map((p) => (p[0] + p[1] + (rq() - 0.5) * 4 > 10 ? 1 : 0)); return [X, y, [r2(rq() * 10), r2(rq() * 10)], k]; };
  const knnCases = [[[[0, 0], [1, 0], [0, 1.2], [5, 5], [6, 5]], [1, 0, 0, 1, 1], [0.1, 0.1], 3], [[[2, 2], [2.5, 3], [3, 2.4], [1.6, 2.9], [8, 8], [9, 7]], [0, 1, 1, 1, 0, 0], [2.1, 2.2], 5], kCase(10, 1), kCase(12, 3), kCase(15, 5), kCase(20, 3), kCase(25, 7), kCase(9, 9), kCase(16, 1), kCase(18, 5), kCase(30, 3), kCase(22, 1)];
  C.exercise({ id: 'nn-x-code-knn', chapter: 'nn-knn', title: 'Write k-NN prediction', level: 2, kind: 'code',
    text: ['Write `knnPredict(X, y, q, k)`: X is an array of training points ([x₁, x₂]), y their labels (0 or 1), q a query point and k an odd number. Return the majority label among the k training points nearest to q in Euclidean distance.'],
    fn: 'knnPredict', params: ['X', 'y', 'q', 'k'], start: 'function knnPredict(X, y, q, k) {\n  \n}',
    ref: (X, y, q, k) => fit('KNeighborsClassifier', { k }, X, y).predict([q])[0], cases: knnCases,
    hints: ['Pair each training index with its distance to q, then sort by distance.', 'Keep the first k, count how many have label 1, and compare with k / 2.', 'function knnPredict(X, y, q, k) {\n  const d = X.map((p, i) => [Math.hypot(p[0] - q[0], p[1] - q[1]), i]).sort((a, b) => a[0] - b[0]);\n  let ones = 0;\n  for (let t = 0; t < k; t++) ones += y[d[t][1]];\n  return ones > k / 2 ? 1 : 0;\n}'],
    solution: 'function knnPredict(X, y, q, k) {\n  const d = X.map((p, i) => [Math.hypot(p[0] - q[0], p[1] - q[1]), i]).sort((a, b) => a[0] - b[0]);\n  let ones = 0;\n  for (let t = 0; t < k; t++) ones += y[d[t][1]];\n  return ones > k / 2 ? 1 : 0;\n}',
    wrong: ['function knnPredict(X, y, q, k) { const d = X.map((p, i) => [Math.hypot(p[0] - q[0], p[1] - q[1]), i]).sort((a, b) => a[0] - b[0]); return y[d[0][1]]; }', 'function knnPredict(X, y, q, k) { const d = X.map((p, i) => [Math.hypot(p[0] - q[0], p[1] - q[1]), i]).sort((a, b) => b[0] - a[0]); let ones = 0; for (let t = 0; t < k; t++) ones += y[d[t][1]]; return ones > k / 2 ? 1 : 0; }'] });

  // ---- workflow: scale before k-NN ---------------------------------------------------------------------------------
  const loans = { data: { set: 'loans', n: 500, seed: 7 }, hidden: { seed: 71, n: 600 }, metric: 'f1', target: 0.45, task: 'classification' };
  C.exercise(Object.assign({ id: 'nn-x-wf-loans', chapter: 'nn-knn', title: 'Put the loans on one scale', level: 2, kind: 'workflow',
    text: ['Incomes run in the tens of thousands, debt ratios below one and missed payments in single figures. Build a workflow with k-nearest neighbours that catches the defaults with F1 at least 0.45 on 600 hidden applications.'],
    limits: { blocks: ['split', 'scale', 'knn', 'train', 'evaluate'], models: ['knn'], params: { knn: { k: [1, 25] } } },
    hints: ['Open the Evaluate block: how many defaults does plain 5-NN catch?', 'Put a Scale block (standard) between the split and Train, and wire the test rows through it too.', 'Split → Scale (standard) → Train with 5-NN → Evaluate: F1 0.5 on the hidden rows, against 0.29 without scaling.'],
    solution: pipe(loans.data, 'knn', { k: 5 }, [['scale', { how: 'standard' }]]), wrong: [pipe(loans.data, 'knn', { k: 5 }), pipe(loans.data, 'knn', { k: 15 }, [['scale', { how: 'standard' }]])] }, loans));

  // =========================================================================================================
  // Chapter 2: fast search and many dimensions
  // =========================================================================================================
  const TP = [[1, 2], [2.5, 4], [3, 1], [4.2, 3.5], [5, 2.2], [6.5, 4.4], [8, 1.5]], TT = ML.KDTree(TP, { leafSize: 3 }).root;
  C.exercise({ id: 'nn-x-kd-split', chapter: 'nn-search', title: 'Where the k-d tree cuts first', level: 1, kind: 'point',
    text: ['A k-d tree is built over these seven points. Its first cut goes across the wider side of the box around them, at the median: the value of the middle point when the seven are sorted along that side. Click where the first cut goes.'],
    plot: { points: TP, classes: TP.map(() => 0), xr: [0, 9], yr: [0, 5], xname: 'x₁', yname: 'x₂' }, mode: TT.dim === 0 ? 'x' : 'y',
    answer: TT.split, tolerance: 0.3, whyNot: 'Sort the seven values along the wider side and take the fourth.',
    hints: ['The box runs from 1 to 8 along x₁ and from 1 to 4.4 along x₂: which side is wider?', 'Along x₁ the sorted values are ' + TP.map((p) => p[0]).sort((a, b) => a - b).join(', ') + '.', 'The fourth of seven is ' + TT.split + ': the cut is the vertical line x₁ = ' + TT.split + '.'],
    solution: { x: TT.split }, wrong: [{ x: 3 }, { x: 5 }] });

  const BQ = [6, 3], BEST = 1.5, BX = { A: [[0, 4], [0, 5]], B: [[6.5, 9], [4, 6]], C: [[7, 9], [0, 2.5]], D: [[4, 5], [4.5, 6]], E: [[2, 5.5], [0, 1]] };
  const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));
  const bd = Object.fromEntries(Object.entries(BX).map(([k, [xr, yr]]) => [k, ML.dist(BQ, [clamp(BQ[0], xr), clamp(BQ[1], yr)], 2)]));
  const skip = Object.keys(bd).filter((k) => bd[k] > BEST), axisSkip = Object.keys(BX).filter((k) => Math.max(BQ[0] - clamp(BQ[0], BX[k][0]), clamp(BQ[0], BX[k][0]) - BQ[0], Math.abs(BQ[1] - clamp(BQ[1], BX[k][1]))) > BEST);
  C.exercise({ id: 'nn-x-kd-skip', chapter: 'nn-search', title: 'Boxes the search can skip', level: 3, kind: 'set',
    text: ['A nearest-neighbour search for the query (6, 3) has already found a point ' + BEST + ' away. Five boxes of the k-d tree are still unopened:', { table: [['Box', 'x₁ from', 'x₁ to', 'x₂ from', 'x₂ to'], ...Object.entries(BX).map(([k, [xr, yr]]) => [k, xr[0], xr[1], yr[0], yr[1]])] }, 'The distance from the query to a box is the distance to the nearest place in it. Which boxes can be skipped without measuring a single point inside? Write a set, like {A, B}.'],
    answer: '{' + skip.join(', ') + '}', caseless: true, placeholder: 'like {A, B}',
    hints: ['Clamp the query into each box: the nearest place of the box. Then measure the straight-line distance to it.', 'Box D: the nearest place is (5, 4.5), so the gaps are 1 and 1.5 and the distance is ' + n3(bd.D) + '.', 'The distances: ' + Object.keys(bd).map((k) => k + ' ' + n3(bd[k])).join(', ') + '. Those beyond ' + BEST + ' are skipped: {' + skip.join(', ') + '}.'],
    solution: '{' + skip.join(', ') + '}', wrong: ['{' + axisSkip.join(', ') + '}', '{B, C}'] });

  C.exercise({ id: 'nn-x-curse-ratio', chapter: 'nn-search', title: 'Nearest against farthest', level: 1, kind: 'choice',
    text: ['300 points are scattered at random in a cube of 1000 dimensions, and a query is dropped among them. What is the distance to the nearest point divided by the distance to the farthest, roughly?'],
    options: ['0.00003', '0.03', '0.35', '0.89'], answer: 3,
    whyNot: { 0: 'That is the ratio in one dimension, where some point is almost on top of the query.', 1: 'That is about the ratio in two dimensions.', 2: 'That is about the ratio in ten dimensions.' },
    hints: ['As dimensions are added, every distance gathers more random gaps, and the totals look alike.', 'In 10 dimensions the ratio is already 0.35.', 'In 1000 dimensions the nearest is 12 away and the farthest 13.5: a ratio of 0.886, about 0.89.'], solution: 3, wrong: [0, 2] });

  // the nearest over the farthest distance, the mean over five queries, each among its own 200 points uniform in the
  // unit cube (ML.rng(2026), ML.dist): printed by the engines in a scratch script and kept here, as the 11 000
  // distances in up to 100 dimensions cost more than a page load should.
  const cRatio = [[1, 0.003], [2, 0.024], [3, 0.085], [5, 0.132], [8, 0.3], [12, 0.406], [20, 0.48], [30, 0.549], [50, 0.653], [75, 0.689], [100, 0.749]];
  C.exercise({ id: 'nn-x-curse-curve', chapter: 'nn-search', title: 'How nearness fades', level: 3, kind: 'predict-curve',
    text: ['200 random points in a unit cube and a random query. For each number of dimensions from 1 to 100, divide the nearest distance by the farthest (averaged over five such draws). The values for one and two dimensions are drawn. Sketch the rest.'],
    plot: { xr: [0, 100], yr: [0, 1], xname: 'dimensions', yname: 'nearest / farthest', given: cRatio.slice(0, 2), note: 'Drag from 1 to 100 dimensions.' },
    answer: cRatio, tolerance: 0.08, checks: ['rising'],
    hints: ['In one dimension some point is nearly on top of the query, so the ratio is tiny.', 'By 20 dimensions the ratio is already ' + cRatio[6][1] + '.', 'It climbs quickly at first, then more slowly towards 1: ' + cRatio.map((p) => p[0] + ': ' + p[1]).join(', ') + '.'],
    solution: cRatio, wrong: [cRatio.map(([d]) => [d, 0.05]), cRatio.map(([d]) => [d, r3(1 - d / 110)])] });

  // =========================================================================================================
  // Chapter 3: naive Bayes
  // =========================================================================================================
  const VOC = ['win', 'free', 'meeting', 'lunch', 'today', 'prize'];
  const SPAM = [[2, 1, 0, 0, 1, 1], [1, 2, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]];
  const HAM = [[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]];
  const nbX = SPAM.concat(HAM), nby = SPAM.map(() => 'spam').concat(HAM.map(() => 'not spam'));
  const nb = (alpha) => fit('MultinomialNB', { alpha }, nbX, nby), NB1 = nb(1), iS = NB1.classes.indexOf('spam'), iH = 1 - iS;
  const cS = NB1.featureCount[iS], cH = NB1.featureCount[iH], tS = ML.sum(cS), tH = ML.sum(cH);
  const lp = (m, k, j) => m.featureLogProb[k][j], w = (j) => VOC.indexOf(j);
  const countTable = { table: [['Word', ...VOC], ['In spam (' + SPAM.length + ' messages)', ...cS], ['In the others (' + HAM.length + ' messages)', ...cH]] };
  const pFree = Math.exp(lp(NB1, iS, w('free')));
  C.exercise({ id: 'nn-x-nb-word-prob', chapter: 'nn-bayes', title: 'One smoothed word probability', level: 1, kind: 'number',
    text: ['A tiny filter has learned from ten messages with a vocabulary of six words. Its word counts:', countTable, 'With add-one smoothing (α = 1), what is P(free | spam)? Three decimal places.'],
    answer: r3(pFree), tolerance: 0.0015, whyNot: { [n3(cS[w('free')] / tS)]: 'That is the unsmoothed share. Add α to the count and α times the vocabulary size to the total.', '*': 'Add 1 to the count of "free" and 1 for each of the 6 words of the vocabulary to the spam total.' },
    hints: ['P(word | spam) = (count + α) / (all spam words + α · vocabulary size).', 'Spam has ' + tS + ' words in all, and "free" ' + cS[w('free')] + ' of them; the vocabulary has 6 words.', '(' + cS[w('free')] + ' + 1) / (' + tS + ' + 6) = ' + n3(pFree) + '.'], solution: n3(pFree), wrong: [n3(cS[w('free')] / tS), n3((cS[w('free')] + 1) / (tS + 1))] });

  const MSG = [0, 1, 0, 1, 1, 0], pSpam = NB1.predictProba([MSG])[0][iS];
  const lo = (j) => lp(NB1, iS, j) - lp(NB1, iH, j), prior = Math.log(SPAM.length / HAM.length);
  C.exercise({ id: 'nn-x-nb-posterior', chapter: 'nn-bayes', title: 'Is “free lunch today” spam?', level: 2, kind: 'number',
    text: ['The same filter, with add-one smoothing:', countTable, 'A new message reads "free lunch today". What probability of spam does multinomial naive Bayes give it? Three decimal places.'],
    answer: r3(pSpam), tolerance: 0.002, whyNot: { [n3(1 - pSpam)]: 'That is the probability that it is not spam.', '*': 'Add the prior log-odds and the three word terms, then take the sigmoid.' },
    hints: ['Start from the prior log-odds ln(' + SPAM.length + ' / ' + HAM.length + ') = ' + n3(prior) + ', then add ln P(word | spam) − ln P(word | not spam) for each word.', 'The words add ' + ['free', 'lunch', 'today'].map((x) => x + ' ' + n3(lo(w(x)))).join(', ') + '.', 'The total is ' + n3(prior + lo(1) + lo(3) + lo(4)) + ', and 1 / (1 + e^−total) = ' + n3(pSpam) + '.'],
    solution: n3(pSpam), wrong: [n3(1 - pSpam), '0.5'] });

  const LW = ['free', 'lunch', 'today'], lcells = Object.fromEntries(LW.map((x) => [x + '|adds to the log-odds', r2(lo(w(x)))]));
  C.exercise({ id: 'nn-x-nb-logodds', chapter: 'nn-bayes', title: 'What each word adds', level: 2, kind: 'table',
    text: ['The same filter, with add-one smoothing:', countTable, 'For each word, fill in ln P(word | spam) − ln P(word | not spam), what one occurrence adds to the log-odds of spam. Two decimal places.'],
    rows: LW, cols: ['adds to the log-odds'], cells: lcells, tolerance: 0.011, corner: 'word',
    hints: ['Spam has ' + tS + ' words, the others ' + tH + '; with α = 1 each total grows by 6.', '"free": ln(' + (cS[1] + 1) + '/' + (tS + 6) + ') − ln(' + (cH[1] + 1) + '/' + (tH + 6) + ') = ' + n2(lo(1)) + '.', LW.map((x) => x + ': ' + n2(lo(w(x)))).join('; ') + '. Negative numbers push away from spam.'],
    solution: Object.fromEntries(Object.entries(lcells).map(([k, v]) => [k, String(v)])), wrong: [Object.fromEntries(LW.map((x) => [x + '|adds to the log-odds', String(r2(-lo(w(x))))]))] });

  C.exercise({ id: 'nn-x-nb-zero', chapter: 'nn-bayes', title: 'The word that vetoes', level: 1, kind: 'choice',
    text: ['The same counts, but now with no smoothing at all (α = 0). The message "win a free lunch" arrives. "lunch" never appeared in spam. What probability of spam does the filter give?', countTable],
    options: ['Exactly 0, whatever the other words say', 'About 0.5, since the evidence is mixed', 'Close to 1, since "win" and "free" are spam words', 'It cannot be computed'], answer: 0,
    whyNot: { 1: 'Products do not average: one factor of 0 makes the whole product 0.', 2: 'However strong "win" and "free" are, they are multiplied by P(lunch | spam) = 0.', 3: 'It can: the spam score is 0 and the other class’s is not.' },
    hints: ['Naive Bayes multiplies the word probabilities.', 'Unsmoothed, P(lunch | spam) = 0 / ' + tS + ' = 0.', 'One factor of 0 makes P(message | spam) = 0, so P(spam) is exactly 0: one unseen word vetoes the class. Smoothing (α > 0) prevents it.'], solution: 0, wrong: [2, 1] });

  C.exercise({ id: 'nn-x-nb-order', chapter: 'nn-bayes', title: 'Shuffled words', level: 1, kind: 'choice',
    text: ['A multinomial naive Bayes filter gives "free lunch today" a probability of spam of ' + n3(pSpam) + '. What does it give "today lunch free"?'],
    options: [n3(pSpam) + ', the same', 'Something lower, as the spam word comes last', 'Something higher', 'It depends on the smoothing'], answer: 0,
    whyNot: { 1: 'The model only counts words; it never looks at their order.', 2: 'The counts are the same, so the sum of the log terms is the same.', 3: 'Smoothing changes the word probabilities, but both orders use the same ones.' },
    hints: ['What does the model keep of a message?', 'Only how often each word appears: a bag of words.', 'The counts are identical, so the probability is the same: ' + n3(pSpam) + '.'], solution: 0, wrong: [1, 3] });

  const AL = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5], AMSG = [0, 0, 1, 0, 1, 1];
  const aCurve = AL.map((a) => [a, r3(nb(a).predictProba([AMSG])[0][iS])]);
  C.exercise({ id: 'nn-x-nb-alpha-curve', chapter: 'nn-bayes', title: 'Smoothing turned up', level: 3, kind: 'predict-curve',
    text: ['The same counts:', countTable, 'The message "meeting today prize" has one word ("prize") seen only in spam and one ("meeting") seen only in the other messages. Sketch its probability of spam as the smoothing α grows from 0.01 to 5. The values at α = 0.01 and 0.05 are drawn.'],
    plot: { xr: [0, 5], yr: [0, 1], xname: 'α', yname: 'P(spam)', given: aCurve.slice(0, 2), note: 'Drag from α = 0.01 to α = 5.' },
    answer: aCurve, tolerance: 0.06, checks: ['rising'],
    hints: ['With tiny α, "prize" nearly vetoes not-spam and "meeting" nearly vetoes spam: the two cancel, and "today", commoner in the other messages, tips it towards not spam.', 'As α grows, every word probability is pulled towards 1/6 and the words count for less, so P(spam) heads towards what the prior alone says, ' + n3(SPAM.length / nbX.length) + '.', 'The curve: ' + aCurve.map((p) => p[0] + ': ' + p[1]).join(', ') + '.'],
    solution: aCurve, wrong: [aCurve.map(([a]) => [a, 1]), aCurve.map(([a]) => [a, 0.05])] });

  const nbRef = (sp, hm, nS, nH, alpha, msg) => { const m = fit('MultinomialNB', { alpha }, [sp, hm], ['s', 'h']); m.classLogPrior = [Math.log(nH / (nS + nH)), Math.log(nS / (nS + nH))]; const J = m.jointLogLikelihood([msg])[0], a = m.classes.indexOf('s'); return J[a] - J[1 - a]; };
  C.exercise({ id: 'nn-x-code-nb', chapter: 'nn-bayes', title: 'Write the spam log-odds', level: 3, kind: 'code',
    text: ['Write `logOdds(spamCounts, hamCounts, nSpam, nHam, alpha, msg)`. The first two are arrays of word counts over the whole vocabulary for each class, nSpam and nHam the numbers of training messages, alpha the smoothing and msg the word counts of a new message. Return ln P(spam | msg) − ln P(not spam | msg) under multinomial naive Bayes.'],
    fn: 'logOdds', params: ['spamCounts', 'hamCounts', 'nSpam', 'nHam', 'alpha', 'msg'], start: 'function logOdds(spamCounts, hamCounts, nSpam, nHam, alpha, msg) {\n  \n}', tolerance: 1e-6,
    ref: nbRef, cases: [[cS, cH, 4, 6, 1, MSG], [cS, cH, 4, 6, 1, AMSG], [cS, cH, 4, 6, 0.5, [1, 1, 0, 0, 0, 2]], [[3, 0, 5, 2], [1, 4, 0, 6], 10, 30, 2, [0, 2, 1, 0]], [[0, 0, 1], [5, 5, 5], 1, 9, 1, [1, 1, 1]]],
    hints: ['Start from ln(nSpam / nHam).', 'For word j: P = (count + alpha) / (total + alpha · vocabulary size), for each class; add msg[j] times the difference of the logs.', 'function logOdds(sp, hm, nS, nH, alpha, msg) {\n  const V = sp.length, tS = sp.reduce((a, b) => a + b, 0), tH = hm.reduce((a, b) => a + b, 0);\n  let s = Math.log(nS / nH);\n  for (let j = 0; j < V; j++) s += msg[j] * (Math.log((sp[j] + alpha) / (tS + alpha * V)) - Math.log((hm[j] + alpha) / (tH + alpha * V)));\n  return s;\n}'],
    solution: 'function logOdds(sp, hm, nS, nH, alpha, msg) {\n  const V = sp.length, tS = sp.reduce((a, b) => a + b, 0), tH = hm.reduce((a, b) => a + b, 0);\n  let s = Math.log(nS / nH);\n  for (let j = 0; j < V; j++) s += msg[j] * (Math.log((sp[j] + alpha) / (tS + alpha * V)) - Math.log((hm[j] + alpha) / (tH + alpha * V)));\n  return s;\n}',
    wrong: ['function logOdds(sp, hm, nS, nH, alpha, msg) { const V = sp.length, tS = sp.reduce((a, b) => a + b, 0), tH = hm.reduce((a, b) => a + b, 0); let s = 0; for (let j = 0; j < V; j++) s += msg[j] * (Math.log((sp[j] + alpha) / (tS + alpha * V)) - Math.log((hm[j] + alpha) / (tH + alpha * V))); return s; }',
      'function logOdds(sp, hm, nS, nH, alpha, msg) { const V = sp.length, tS = sp.reduce((a, b) => a + b, 0), tH = hm.reduce((a, b) => a + b, 0); let s = Math.log(nS / nH); for (let j = 0; j < V; j++) s += msg[j] * (Math.log((sp[j] + alpha) / (tS + alpha)) - Math.log((hm[j] + alpha) / (tH + alpha))); return s; }'] });

  // ---- Gaussian naive Bayes ----------------------------------------------------------------------------------
  const GA = [1.8, 2.2, 2.0, 2.4, 1.6], GB = [2.4, 3.6, 3.0, 2.7, 3.3, 3.1, 2.9], GX = 2.3;
  const G = fit('GaussianNB', {}, GA.concat(GB).map((v) => [v]), GA.map(() => 'ashbell').concat(GB.map(() => 'copperleaf')));
  const gA = G.classes.indexOf('ashbell'), gE = G.explain([GX]), llA = gE[gA].terms[0], post = G.predictProba([[GX]])[0][gA];
  C.exercise({ id: 'nn-x-gnb-loglik', chapter: 'nn-bayes', title: 'The height of one bell', level: 2, kind: 'number',
    text: ['Five ashbell flowers have petal lengths ' + GA.join(', ') + ' cm. Gaussian naive Bayes fits them a mean and a variance (divided by n, as the model does). What is the log of the bell’s height, ln N(x; mean, variance), at a petal length of x = ' + GX + '? Two decimal places.'],
    answer: r2(llA), tolerance: 0.011, whyNot: { [n2(Math.exp(llA))]: 'That is the height itself; take its logarithm.', '*': 'Use the variance divided by n, and both terms: −½ ln(2πσ²) and −(x − μ)² / (2σ²).' },
    hints: ['ln N(x; μ, σ²) = −½ ln(2π σ²) − (x − μ)² / (2σ²).', 'The mean is ' + n3(G.theta[gA][0]) + ' and the variance ' + n3(G.var[gA][0]) + '.', '−½ ln(2π · ' + n3(G.var[gA][0]) + ') − (' + GX + ' − ' + n3(G.theta[gA][0]) + ')² / (2 · ' + n3(G.var[gA][0]) + ') = ' + n2(llA) + '.'],
    solution: n2(llA), wrong: [n2(Math.exp(llA)), n2(llA - Math.log(GA.length / 12))] });
  C.exercise({ id: 'nn-x-gnb-posterior', chapter: 'nn-bayes', title: 'Ashbell or copperleaf', level: 3, kind: 'number',
    text: ['Gaussian naive Bayes on one feature, petal length. Five ashbells measure ' + GA.join(', ') + ' cm; seven copperleafs measure ' + GB.join(', ') + ' cm. The priors are the shares of the twelve flowers. What probability does the model give to ashbell for a petal of ' + GX + ' cm? Three decimal places.'],
    answer: r3(post), tolerance: 0.002, whyNot: { [n3(1 - post)]: 'That is the probability of copperleaf.', '*': 'Each class has its own mean and variance, and its prior is its share of the twelve flowers.' },
    hints: ['For each class: log prior + ln N(x; its mean, its variance).', 'Ashbell: ln(5/12) ' + sg(r3(gE[gA].terms[0])) + ' = ' + m3(gE[gA].prior + gE[gA].terms[0]) + '. Copperleaf: ln(7/12) ' + sg(r3(gE[1 - gA].terms[0])) + ' = ' + m3(gE[1 - gA].prior + gE[1 - gA].terms[0]) + '.', 'Exponentiate and normalise: P(ashbell) = ' + n3(post) + '.'],
    solution: n3(post), wrong: [n3(1 - post), '0.5'] });
  C.exercise({ id: 'nn-x-gnb-tilted', chapter: 'nn-bayes', title: 'Tilted clouds', level: 2, kind: 'choice',
    text: ['In each of two classes, the two features rise together: the clouds are long ellipses tilted at 45°. What can Gaussian naive Bayes not represent here, and which model can?'],
    options: ['The correlation inside each class; QDA (or LDA) can', 'The class means; k-NN can', 'The priors; logistic regression can', 'Nothing: naive Bayes fits any ellipse'], answer: 0,
    whyNot: { 1: 'Naive Bayes estimates every class mean exactly.', 2: 'The priors are simply the class shares, which it uses.', 3: 'Its ellipses always line up with the axes: one variance per feature, no covariance.' },
    hints: ['"Naive" means the features are taken as independent within a class.', 'Independence means no covariance: each class’s ellipse is aligned with the axes.', 'It cannot represent the correlation inside each class. QDA fits a full covariance per class (LDA one shared), so it can tilt its ellipses.'], solution: 0, wrong: [3, 1] });

  // =========================================================================================================
  // Chapter 4: discriminant analysis
  // =========================================================================================================
  C.exercise({ id: 'nn-x-lda-qda', chapter: 'nn-da', title: 'LDA or QDA', level: 2, kind: 'choice',
    text: ['Two classes, 2000 labelled rows. One class is a round cloud; the other is long and narrow, tilted across the first. Which model fits this best?'],
    options: ['QDA: each class gets its own covariance, so the boundary can curve around the narrow class', 'LDA: one shared covariance is always safer', 'Gaussian naive Bayes: it fits a tilted ellipse per class', 'None: Gaussian models need equal shapes'], answer: 0,
    whyNot: { 1: 'With 2000 rows QDA can afford its extra numbers, and one shared shape would average a round and a narrow cloud.', 2: 'Naive Bayes cannot tilt its ellipses.', 3: 'Only LDA assumes equal shapes; QDA does not.' },
    hints: ['The two classes have different shapes.', 'LDA forces one shared covariance; QDA estimates one per class, which costs data, and there is plenty.', 'QDA: a covariance per class and a curved boundary, affordable with 2000 rows. With a dozen rows, LDA would be the safer choice.'], solution: 0, wrong: [1, 2] });

  const qd = 4, qK = 3, qda = qK * qd + qK * (qd * (qd + 1)) / 2, lda = qK * qd + (qd * (qd + 1)) / 2;
  C.exercise({ id: 'nn-x-qda-count', chapter: 'nn-da', title: 'How many numbers QDA estimates', level: 2, kind: 'number',
    text: ['QDA on a table with ' + qd + ' features and ' + qK + ' classes. Counting each class’s mean and each class’s covariance matrix (a symmetric matrix: count each distinct entry once), how many numbers does it estimate? Leave out the priors.'],
    answer: qda, tolerance: 0, whyNot: { [String(lda)]: 'That is LDA, with one covariance shared by all the classes.', [String(qK * qd + qK * qd * qd)]: 'A symmetric 4 × 4 matrix has only 10 distinct entries, not 16.' },
    hints: ['Means: one number per feature per class.', 'A symmetric d × d covariance has d(d + 1) / 2 distinct entries: 10 for d = 4.', qK + ' × ' + qd + ' means + ' + qK + ' × 10 covariance entries = ' + qda + '. LDA, with one shared covariance, needs ' + lda + '.'], solution: String(qda), wrong: [String(lda), String(qK * qd + qK * qd * qd)] });

  const LD = S.Data.load('blobs', { n: 80, seed: 6, centers: [[0, 0], [2.5, 1]], std: 1 }), LDm = fit('LDA', {}, LD.X, LD.y);
  const lw = LDm.coef[1].map((v, j) => r2(v - LDm.coef[0][j])), lb = r2(LDm.intercept[1] - LDm.intercept[0]), LP = [1.5, -0.5];
  const lscore = r2(lw[0] * LP[0] + lw[1] * LP[1] + lb), lprob = ML.sigmoid(lscore);
  C.exercise({ id: 'nn-x-lda-score', chapter: 'nn-da', title: 'Reading LDA’s log-odds', level: 3, kind: 'number',
    text: ['LDA fitted to two classes gives the log-odds of class 1 against class 0 as a straight-line function of the two features:', { math: 'log-odds(x) = ' + lw[0] + '·x₁ ' + sg(lw[1]) + '·x₂ ' + sg(lb) }, 'For the point x = (' + LP.join(', ') + '), what is the probability of class 1? Three decimal places.'],
    answer: r3(lprob), tolerance: 0.002, whyNot: { [n3(lscore)]: 'That is the log-odds. Turn it into a probability with 1 / (1 + e^−z).', [n3(1 - lprob)]: 'That is the probability of class 0.' },
    hints: ['First the log-odds: multiply and add.', 'z = ' + lw[0] + ' · ' + LP[0] + ' ' + sg(lw[1]) + ' · (−' + Math.abs(LP[1]) + ') ' + sg(lb) + ' = ' + m3(lscore) + '.', '1 / (1 + e^' + m3(-lscore) + ') = ' + n3(lprob) + '.'], solution: n3(lprob), wrong: [n3(lscore), n3(1 - lprob)] });

  const flw = { data: { set: 'wildflowers', n: 90, seed: 2 }, hidden: { seed: 21, n: 150 }, metric: 'accuracy', target: 0.95, task: 'classification' };
  C.exercise(Object.assign({ id: 'nn-x-wf-flowers', chapter: 'nn-da', title: 'Wildflowers without scaling', level: 2, kind: 'workflow',
    text: ['The wildflower table measures petals in centimetres, stems in tens of centimetres and leaves by count. There is no Scale block this time. Build a workflow that names the species of 150 hidden flowers with accuracy at least 0.95.'],
    limits: { blocks: ['split', 'knn', 'naivebayes', 'lda', 'qda', 'train', 'evaluate'] },
    hints: ['k-NN measures distances, so the stem heights drown the petals without scaling.', 'The Gaussian models fit a mean and a spread per feature and class, in each feature’s own units, so units do not matter to them.', 'Split → Train with naive Bayes (Gaussian) → Evaluate: accuracy 0.967 on the hidden flowers. (5-NN reaches only 0.86.)'],
    solution: pipe(flw.data, 'naivebayes', {}), wrong: [pipe(flw.data, 'knn', { k: 5 }), pipe(flw.data, 'knn', { k: 15 })] }, flw));
})(globalThis.Nooral = globalThis.Nooral || {});
