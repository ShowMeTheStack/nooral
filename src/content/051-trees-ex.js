// Part 5, trees and ensembles: the exercises. Every number is worked out here by the engines (S.ML's trees and
// boosters, S.Metrics, S.Data) on small fixed inputs, so the text, the answers and the hints always agree.
(function (S) {
  'use strict';
  const C = S.Content, ML = S.ML, M = S.Metrics, WF = S.WF;
  if (!C || !ML || !WF || C.exercises.has('tr-x-gini-node')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  const col = (xs) => xs.map((v) => [v]);
  // the impurity of a node holding these labels (classes 0..K−1), from the tree engine itself
  const imp = (y, crit, K) => ML.growTree(y.map(() => [0]), y, { criterion: crit || 'gini', nClasses: K || 2, maxDepth: 0 }).nodes[0].impurity;
  const labelsOf = (counts) => counts.flatMap((c, k) => Array(c).fill(k));
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };
  const ex = (x) => C.exercise(x);

  // =========================================================================================================
  // Chapter 1: growing a tree
  // =========================================================================================================
  const node73 = labelsOf([3, 7]), g73 = imp(node73), h73 = imp(node73, 'entropy');
  ex({ id: 'tr-x-gini-node', chapter: 'tr-grow', title: 'The Gini impurity of a node', level: 1, kind: 'number',
    text: ['A leaf of a loan tree holds 10 applicants: 7 repaid and 3 defaulted. What is its Gini impurity, 1 − Σ p²? Give three decimals.'],
    answer: r3(g73), tolerance: 0.0005, whyNot: { '0.3': 'That is the share of defaults, not the impurity.', '0.58': 'That is p₁² + p₂²; the impurity is 1 minus that.', '0.21': 'That is p₁·p₂; the impurity for two classes is twice that.' },
    hints: ['The shares are 0.7 and 0.3.', 'Square them and add: 0.49 + 0.09 = 0.58.', 'G = 1 − 0.58 = ' + r3(g73) + '.'], solution: String(r3(g73)), wrong: ['0.58', '0.3', '0.21'] });
  ex({ id: 'tr-x-entropy-node', chapter: 'tr-grow', title: 'The entropy of the same node', level: 1, kind: 'number',
    text: ['The same leaf: 7 repaid and 3 defaulted. What is its entropy, −Σ p log₂ p, in bits? Give three decimals.'],
    answer: r3(h73), tolerance: 0.0015, whyNot: { '0.611': 'That is with natural logarithms; entropy for trees is in bits, log base 2.', '0.42': 'That is the Gini impurity.', '*': 'Add −p log₂ p for both shares, 0.7 and 0.3.' },
    hints: ['log₂ p = ln p / ln 2.', '−0.7 · log₂ 0.7 = 0.360 and −0.3 · log₂ 0.3 = 0.521.', 'H = 0.360 + 0.521 = ' + r3(h73) + ' bits.'], solution: String(r3(h73)), wrong: ['0.611', '0.42', '1'] });
  ex({ id: 'tr-x-gini-entropy', chapter: 'tr-grow', title: 'Gini against entropy', level: 1, kind: 'choice',
    text: ['Which statement about the two impurity measures, for two classes, is true?'],
    options: ['Both are 0 for a pure node; Gini is at most 0.5 and entropy at most 1 bit, both at an even mix', 'Gini is largest for a pure node', 'Entropy can be negative when one class dominates', 'They rank every pair of splits in opposite orders'], answer: 0,
    whyNot: { 1: 'A pure node has nothing mixed: its impurity is 0, the smallest possible.', 2: 'Each term −p log₂ p is at least 0 for p between 0 and 1.', 3: 'They almost always agree; on the moons both picked the same first cut.' },
    hints: ['Try a pure node, p = (1, 0), in both formulas.', 'Then try the even mix, p = (0.5, 0.5).', 'Pure: both 0. Even mix: Gini 1 − 0.25 − 0.25 = 0.5, entropy 1 bit. The first statement is true.'], solution: 0, wrong: [1, 3] });

  const HRS = [1, 2.5, 3, 4.5, 5, 6.5, 7, 8.5, 9, 10.5], PASS = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1];
  const stump = new ML.DecisionTreeClassifier({ maxDepth: 1 }).fit(col(HRS), PASS);
  const cand = stump.trace.find((s) => s.kind === 'candidates').candidates[0], best = stump.tree.nodes[0].threshold;
  const hrsText = 'Ten students, by hours of study: ' + HRS.map((h, i) => h + ' h (' + (PASS[i] ? 'pass' : 'fail') + ')').join(', ') + '.';
  const sideG = (t) => { const L = PASS.filter((_, i) => HRS[i] <= t), R = PASS.filter((_, i) => HRS[i] > t); return { L, R, gl: imp(L), gr: imp(R), w: (L.length * imp(L) + R.length * imp(R)) / PASS.length }; };
  const s375 = sideG(3.75), H1 = (k, one, many) => k + ' ' + (k === 1 ? one : many);
  ex({ id: 'tr-x-split-table', chapter: 'tr-grow', title: 'Score one split', level: 2, kind: 'table',
    text: [hrsText, 'Split them at 3.75 h: the students at or below go left. Fill in each side’s count and Gini impurity, and the weighted Gini of the split (three decimals).'],
    rows: ['left (≤ 3.75 h)', 'right (> 3.75 h)', 'the split'], cols: ['students', 'Gini'], corner: 'side',
    cells: { 'left (≤ 3.75 h)|students': s375.L.length, 'left (≤ 3.75 h)|Gini': r3(s375.gl), 'right (> 3.75 h)|students': s375.R.length, 'right (> 3.75 h)|Gini': r3(s375.gr), 'the split|Gini': r3(s375.w) }, tolerance: 0.0015,
    hints: ['Left: the three students under 3.75 h all failed.', 'Right: 7 students, 1 fail and 6 passes, so G = 1 − (1/7)² − (6/7)² = ' + r3(s375.gr) + '.', 'Weighted: (3 · 0 + 7 · ' + r3(s375.gr) + ') / 10 = ' + r3(s375.w) + '.'],
    solution: { 'left (≤ 3.75 h)|students': '3', 'left (≤ 3.75 h)|Gini': '0', 'right (> 3.75 h)|students': '7', 'right (> 3.75 h)|Gini': String(r3(s375.gr)), 'the split|Gini': String(r3(s375.w)) },
    wrong: [{ 'left (≤ 3.75 h)|students': '3', 'left (≤ 3.75 h)|Gini': '0', 'right (> 3.75 h)|students': '7', 'right (> 3.75 h)|Gini': String(r3(s375.gr)), 'the split|Gini': String(r3((s375.gl + s375.gr) / 2)) }] });
  ex({ id: 'tr-x-best-threshold', chapter: 'tr-grow', title: 'Where the stump cuts', level: 2, kind: 'point',
    text: [hrsText, 'A tree of depth 1 tries every threshold halfway between neighbouring hours and keeps the one with the lowest weighted Gini. Click where it cuts.'],
    plot: { points: HRS.map((v, i) => [v, 0.45 + 0.15 * (i % 2)]), classes: PASS, xr: [0, 11.5], yr: [0, 1], xname: 'hours of study', yname: '' }, mode: 'x',
    answer: best, tolerance: 0.3, whyNot: 'Score the candidates around the lone pass at 4.5 h and the lone fail at 5 h.',
    hints: ['The candidates are 1.75, 2.75, 3.75, …, 9.75.', 'At 3.75 the weighted Gini is ' + r3(sideG(3.75).w) + '; at 5.75 it is ' + r3(sideG(5.75).w) + '.', 'The lowest is at ' + best + ' h: left ' + sideG(best).L.length + ' students (' + H1(sideG(best).L.filter((v) => v === 1).length, 'pass', 'passes') + '), right ' + sideG(best).R.length + ' students (' + H1(sideG(best).R.filter((v) => v === 1).length, 'pass', 'passes') + '), weighted Gini ' + r3(sideG(best).w) + '.'],
    solution: { x: best }, wrong: [{ x: 3.75 }, { x: 4.75 }] });
  const pick = [2.75, 3.75, 4.75, 5.75, 6.75], cg = {};
  pick.forEach((t) => { cg[t + ' h|weighted Gini'] = r3(cand.scores[cand.thresholds.indexOf(t)]); });
  ex({ id: 'tr-x-candidates', chapter: 'tr-grow', title: 'Every candidate scored', level: 2, kind: 'table',
    text: [hrsText, 'Fill in the weighted Gini of the split at each of these thresholds (three decimals).'],
    rows: pick.map((t) => t + ' h'), cols: ['weighted Gini'], corner: 'threshold', cells: cg, tolerance: 0.0015,
    hints: ['For each threshold count the passes and fails on each side, take each side’s Gini, and weight by its size.', 'At 2.75 h: left 2 fails (Gini 0); right 8 students, 2 fails and 6 passes (Gini 0.375); weighted 0.8 · 0.375 = ' + cg['2.75 h|weighted Gini'] + '.', pick.map((t) => t + ' h: ' + cg[t + ' h|weighted Gini']).join('; ') + '. The lowest is at ' + best + ' h.'],
    solution: Object.fromEntries(Object.entries(cg).map(([k, v]) => [k, String(v)])) });
  const pred = stump.predict(col(HRS)), cmS = M.confusion(PASS, pred, { labels: [0, 1] });
  ex({ id: 'tr-x-stump-matrix', chapter: 'tr-grow', title: 'What the stump gets wrong', level: 1, kind: 'matrix', diag: true,
    text: [hrsText, 'The stump says pass above ' + best + ' h and fail at or below it. Fill in its confusion matrix on these ten students: rows the truth, columns what the stump said.'],
    rows: ['fail', 'pass'], cols: ['said fail', 'said pass'], corner: 'true ↓  said →', answer: cmS.counts, total: 10,
    hints: ['Below ' + best + ' h are the students with 1, 2.5, 3 and 4.5 hours.', 'One of them, at 4.5 h, passed: the stump calls them a fail.', 'Rows: fail ' + cmS.counts[0].join(', ') + '; pass ' + cmS.counts[1].join(', ') + '.'],
    solution: cmS.counts.map((r) => r.map(String)), wrong: [cmS.counts.map((r) => r.slice().reverse().map(String))] });
  ex({ id: 'tr-x-xor-choice', chapter: 'tr-grow', title: 'Why XOR troubles a greedy tree', level: 2, kind: 'choice',
    text: ['Points in the four corners of a square: top-left and bottom-right are one class, the other two corners the other. A tree of depth 2 could separate them perfectly. Why does CART, choosing the best cut at each node, often fail to find that tree?'],
    options: ['Every single axis cut leaves both classes evenly mixed on both sides, so the first cut is chosen by noise and may not be the useful middle one', 'Trees cannot make two cuts on the same feature', 'Gini cannot be computed for four corners', 'The perceptron rule forbids it'], answer: 0,
    whyNot: { 1: 'A tree can cut the same feature as often as it likes.', 2: 'Gini is defined for any node.', 3: 'That rule belongs to another model.' },
    hints: ['Cut down the middle: what is on each side?', 'Each side holds one corner of each class: still 50/50, no gain at all.', 'No single cut lowers the impurity much, so the first choice is decided by small accidents of the sample; only the right later cut would pay off, and greedy search never looks ahead.'], solution: 0, wrong: [1, 2] });
  const giniRef = (counts) => (ML.sum(counts) ? imp(labelsOf(counts), 'gini', counts.length) : 0);
  ex({ id: 'tr-x-code-gini', chapter: 'tr-grow', title: 'Write the Gini impurity', level: 2, kind: 'code',
    text: ['Write `gini(counts)`: given the number of points of each class in a node, such as [7, 3] or [2, 3, 5], return 1 − Σ p², where p is each count divided by the total. An empty node, all zeros, has impurity 0.'],
    fn: 'gini', params: ['counts'], start: 'function gini(counts) {\n  \n}', tolerance: 1e-9, ref: giniRef, cases: [[[5, 5]], [[7, 3]], [[10, 0]], [[1, 21]], [[2, 3, 5]], [[0, 4, 4]], [[0, 0]]],
    hints: ['Add up the counts first to get n.', 'Then subtract (c / n)² from 1 for each count c; return 0 when n is 0.', 'function gini(counts) { const n = counts.reduce((a, b) => a + b, 0); if (!n) return 0; return 1 - counts.reduce((s, c) => s + (c / n) ** 2, 0); }'],
    solution: 'function gini(counts) { const n = counts.reduce((a, b) => a + b, 0); if (!n) return 0; return 1 - counts.reduce((s, c) => s + (c / n) ** 2, 0); }',
    wrong: ['function gini(counts) { const n = counts.reduce((a, b) => a + b, 0); return counts.reduce((s, c) => s + (c / n) ** 2, 0); }', 'function gini(counts) { const n = counts.reduce((a, b) => a + b, 0); if (!n) return 0; return 1 - counts.reduce((s, c) => s + c / n, 0); }'] });
  const splitRef = (xs, ys) => new ML.DecisionTreeClassifier({ maxDepth: 1 }).fit(col(xs), ys).tree.nodes[0].threshold;
  ex({ id: 'tr-x-code-split', chapter: 'tr-grow', title: 'Write the best split', level: 3, kind: 'code',
    text: ['Write `bestSplit(xs, ys)`: xs are the values of one feature (not sorted), ys the classes 0 or 1. Try every threshold halfway between two neighbouring distinct values, score it by the weighted Gini of the two sides (x ≤ t goes left), and return the threshold with the lowest score. If two tie, return the smaller threshold. The cases always hold both classes.'],
    fn: 'bestSplit', params: ['xs', 'ys'], start: 'function bestSplit(xs, ys) {\n  \n}', tolerance: 1e-9, ref: splitRef,
    cases: [[HRS, PASS], [[3, 1, 2, 8, 9, 7], [0, 0, 0, 1, 1, 1]], [[5, 1, 4, 2, 3], [1, 0, 1, 0, 0]], [[0.5, 1.2, 1.9, 2.4, 3.3, 3.9, 4.4, 5.1], [0, 0, 0, 1, 0, 1, 1, 1]], [[2, 2, 3, 3, 6, 7], [0, 1, 0, 0, 1, 1]], [[1, 2, 3, 4, 5, 6, 7], [1, 1, 0, 0, 0, 1, 1]]],
    hints: ['Sort the pairs by x, then walk the gaps between neighbours, skipping equal values.', 'For a gap, count the classes on each side; the score is (nL·G(L) + nR·G(R)) / n. Keep a candidate only when its score is strictly lower than the best so far.', 'function bestSplit(xs, ys) {\n  const p = xs.map((x, i) => [x, ys[i]]).sort((a, b) => a[0] - b[0]), n = p.length;\n  const g = (a, b) => { const m = a + b; return m ? 1 - (a / m) ** 2 - (b / m) ** 2 : 0; };\n  const tot = [0, 0]; p.forEach(([, y]) => tot[y]++);\n  const L = [0, 0]; let best = Infinity, t = null;\n  for (let i = 1; i < n; i++) {\n    L[p[i - 1][1]]++;\n    if (p[i][0] === p[i - 1][0]) continue;\n    const nl = i, nr = n - i, s = (nl * g(L[0], L[1]) + nr * g(tot[0] - L[0], tot[1] - L[1])) / n;\n    if (s < best) { best = s; t = (p[i - 1][0] + p[i][0]) / 2; }\n  }\n  return t;\n}'],
    solution: 'function bestSplit(xs, ys) {\n  const p = xs.map((x, i) => [x, ys[i]]).sort((a, b) => a[0] - b[0]), n = p.length;\n  const g = (a, b) => { const m = a + b; return m ? 1 - (a / m) ** 2 - (b / m) ** 2 : 0; };\n  const tot = [0, 0]; p.forEach(([, y]) => tot[y]++);\n  const L = [0, 0]; let best = Infinity, t = null;\n  for (let i = 1; i < n; i++) {\n    L[p[i - 1][1]]++;\n    if (p[i][0] === p[i - 1][0]) continue;\n    const nl = i, nr = n - i, s = (nl * g(L[0], L[1]) + nr * g(tot[0] - L[0], tot[1] - L[1])) / n;\n    if (s < best) { best = s; t = (p[i - 1][0] + p[i][0]) / 2; }\n  }\n  return t;\n}',
    wrong: ['function bestSplit(xs, ys) { const p = xs.map((x, i) => [x, ys[i]]).sort((a, b) => a[0] - b[0]); return (p[0][0] + p[p.length - 1][0]) / 2; }',
      'function bestSplit(xs, ys) {\n  const n = xs.length; const g = (a, b) => { const m = a + b; return m ? 1 - (a / m) ** 2 - (b / m) ** 2 : 0; };\n  let best = Infinity, t = null;\n  for (let i = 1; i < n; i++) { const th = (xs[i - 1] + xs[i]) / 2; let a = 0, b = 0, c = 0, d = 0; xs.forEach((x, k) => { if (x <= th) { if (ys[k]) b++; else a++; } else { if (ys[k]) d++; else c++; } }); const s = ((a + b) * g(a, b) + (c + d) * g(c, d)) / n; if (s < best) { best = s; t = th; } }\n  return t;\n}'] });

  const RX = [1, 2, 3, 4, 5, 6], RY = [3, 4, 3.5, 8, 9, 8.5];
  const sse = (ys) => ys.length * ML.growTree(ys.map(() => [0]), ys, { criterion: 'mse', maxDepth: 0 }).nodes[0].impurity;
  const sse25 = sse(RY.slice(0, 2)) + sse(RY.slice(2)), sse35 = sse(RY.slice(0, 3)) + sse(RY.slice(3));
  const rtext = 'Six plots of land with x = ' + RX.join(', ') + ' (tonnes of compost) and yields y = ' + RY.join(', ') + '.';
  ex({ id: 'tr-x-sse-split', chapter: 'tr-grow', title: 'A regression split by squared error', level: 2, kind: 'number',
    text: [rtext, 'A regression tree considers the cut x ≤ 2.5. Each side predicts the mean of its y. What is the total sum of squared errors of the two sides? Give two decimals.'],
    answer: r2(sse25), tolerance: 0.005, whyNot: { [String(r2(sse35))]: 'That is for the cut at 3.5; this one is at 2.5.', '*': 'Take each side’s mean, then add up the squared gaps of all six points to their own side’s mean.' },
    hints: ['Left: y = 3 and 4, mean 3.5. Right: 3.5, 8, 9 and 8.5, mean 7.25.', 'Left SSE = 0.25 + 0.25 = 0.5. Right SSE = 3.75² + 0.75² + 1.75² + 1.25².', 'Right SSE = 19.25, so the total is ' + r2(sse25) + '. The cut at 3.5 would leave only ' + r2(sse35) + ', which is why the tree picks it.'], solution: String(r2(sse25)), wrong: [String(r2(sse35)), '9.875'] });
  const rstump = new ML.DecisionTreeRegressor({ maxDepth: 1 }).fit(col(RX), RY), rp = rstump.predict([[5.2]])[0], rp9 = rstump.predict([[40]])[0];
  ex({ id: 'tr-x-reg-predict', chapter: 'tr-grow', title: 'What a regression stump predicts', level: 1, kind: 'number',
    text: [rtext, 'A regression tree of depth 1 cuts at x ≤ ' + rstump.tree.nodes[0].threshold + '. What does it predict for a plot with x = 5.2?'],
    answer: rp, tolerance: 0.005, whyNot: { [String(r2(ML.mean(RY)))]: 'That is the mean of all six; the stump uses the mean of the side x = 5.2 falls on.', '9': 'That is one point’s yield; a leaf predicts the mean of its points.' },
    hints: ['5.2 is above the threshold, so it goes right.', 'The right leaf holds the yields 8, 9 and 8.5.', 'Their mean is ' + rp + '. (It would say ' + rp9 + ' for x = 40 too: a tree cannot extrapolate.)'], solution: String(rp), wrong: [String(r2(ML.mean(RY))), '9'] });

  // =========================================================================================================
  // Chapter 2: how big a tree should be
  // =========================================================================================================
  const moons = S.Data.load('moons', { n: 120, noise: 0.3, seed: 7 });
  const depthCurve = [1, 2, 3, 4, 5, 6, 7, 8, 10].map((d) => [d, r3(ML.accuracy(moons.y, new ML.DecisionTreeClassifier({ maxDepth: d }).fit(moons.X, moons.y).predict(moons.X)))]);
  ex({ id: 'tr-x-curve-depth', chapter: 'tr-size', title: 'Training accuracy as the tree deepens', level: 2, kind: 'predict-curve',
    text: ['A decision tree learns 120 noisy moon points and is scored on the same points. Sketch its training accuracy as the maximum depth grows from 1 to 10.'],
    plot: { xr: [0, 11], yr: [0.7, 1.02], xname: 'max depth', yname: 'training accuracy', given: depthCurve.slice(0, 1), note: 'Drag from depth 1 to depth 10. The faint point is depth 1.' },
    answer: depthCurve, tolerance: 0.03, checks: ['rising'],
    hints: ['Every extra level can only split leaves further, so the training score never falls.', 'Once every leaf is pure, nothing more can be gained.', 'It climbs from ' + depthCurve[0][1] + ' at depth 1 to ' + depthCurve[4][1] + ' at depth 5 and reaches 1 at depth ' + depthCurve.find((p) => p[1] === 1)[0] + ', then stays flat.'],
    solution: depthCurve, wrong: [depthCurve.map(([d]) => [d, 0.84]), depthCurve.map(([d, v]) => [d, Math.max(0.7, 1.84 - v)])] });
  ex({ id: 'tr-x-overfit-choice', chapter: 'tr-size', title: 'Which tree to keep', level: 1, kind: 'choice',
    text: ['Trees of growing depth on 80 noisy moon points, scored on their training points and on 300 fresh points (from the topic “Depth and the smallest leaf”):', { table: [['Max depth', 'Training', 'Fresh points'], ['1', '84%', '73%'], ['2', '91%', '82%'], ['5', '98%', '82%'], ['12', '100%', '82%']] }, 'Which depth would you keep?'],
    options: ['12: it gets every training point right', '1: the simplest', '2: the smallest tree that reaches the best fresh score', 'It cannot be decided from this table'], answer: 2,
    whyNot: { 0: 'The training score always prefers the deepest tree; the fresh points show the extra depth bought nothing.', 1: 'Depth 1 underfits: 73% on fresh points.', 3: 'The fresh-point column is exactly what decides it.' },
    hints: ['Look only at the fresh-point column.', 'Depths 2, 5 and 12 all reach 82%.', 'Among equals keep the simplest: depth 2.'], solution: 2, wrong: [0, 1] });
  ex({ id: 'tr-x-minleaf', chapter: 'tr-size', title: 'Raising the minimum leaf size', level: 1, kind: 'choice',
    text: ['A tree is grown with min leaf 1 and then again with min leaf 10, on the same data and with no depth limit. What happens to the second tree?'],
    options: ['It has fewer leaves, and its training accuracy is the same or lower', 'It has more leaves', 'It always scores better on fresh data', 'Nothing changes: min leaf only matters for regression'], answer: 0,
    whyNot: { 1: 'A split is only allowed if both children keep at least 10 points, so fewer splits happen.', 2: 'Often, but not always: if the data are clean, small leaves may be right.', 3: 'Min leaf limits classification trees too.' },
    hints: ['A split is refused if either child would hold fewer than 10 points.', 'Fewer splits means fewer leaves, so the tree cannot wrap single noisy points.', 'Fewer leaves; training accuracy the same or lower; fresh accuracy often better but not guaranteed.'], solution: 0, wrong: [1, 2] });
  const Rt = 0.12, RTt = 0.03, nl = 3, gccp = (Rt - RTt) / (nl - 1);
  ex({ id: 'tr-x-ccp-g', chapter: 'tr-size', title: 'The price of a subtree', level: 2, kind: 'number',
    text: ['An internal node t of a grown tree has a subtree with 3 leaves whose weighted impurities add up to R(T_t) = ' + RTt + '. As a single leaf, t would have R(t) = ' + Rt + '. What is its g(t), the α at which pruning collapses it? Give three decimals.'],
    answer: r3(gccp), tolerance: 0.0005, whyNot: { '0.03': 'Divide by the number of leaves saved, 3 − 1 = 2, not by 3.', '0.09': 'That is the impurity saved; divide it by the leaves saved.' },
    hints: ['g(t) = (R(t) − R(T_t)) / (leaves − 1).', 'R(t) − R(T_t) = ' + Rt + ' − ' + RTt + ' = 0.09.', '0.09 / 2 = ' + r3(gccp) + '.'], solution: String(r3(gccp)), wrong: ['0.09', '0.03'] });
  ex({ id: 'tr-x-ccp-which', chapter: 'tr-size', title: 'The weakest link', level: 2, kind: 'choice',
    text: ['Three internal nodes of a grown tree:', { table: [['Node', 'R(t) as a leaf', 'R(T_t) of its subtree', 'leaves in the subtree'], ['A', '0.20', '0.08', '4'], ['B', '0.05', '0.02', '2'], ['C', '0.15', '0.03', '3']] }, 'As α grows from 0, which node’s subtree is collapsed first?'],
    options: ['A', 'B', 'C', 'All three at once'], answer: 1,
    whyNot: { 0: 'g(A) = 0.12 / 3 = 0.04: the subtree saves the most impurity, but it also costs three extra leaves.', 2: 'g(C) = 0.12 / 2 = 0.06, the largest.', 3: 'Each has its own g; the smallest goes first.' },
    hints: ['Work out g = (R(t) − R(T_t)) / (leaves − 1) for each.', 'g(A) = 0.04, g(C) = 0.06.', 'g(B) = (0.05 − 0.02) / 1 = 0.03, the smallest: B goes first, once α passes 0.03.'], solution: 1, wrong: [0, 2] });

  // =========================================================================================================
  // Chapter 3: bagging and random forests
  // =========================================================================================================
  const left60 = Math.pow(1 - 1 / 60, 60);
  ex({ id: 'tr-x-oob-share', chapter: 'tr-forest', title: 'How many rows a bootstrap leaves out', level: 2, kind: 'number',
    text: ['A bootstrap sample draws 60 rows, one at a time with replacement, from 60 training rows. What is the chance that one particular row is never drawn? Give three decimals.'],
    answer: r3(left60), tolerance: 0.0015, whyNot: { '0.368': 'That is 1/e, the limit for very many rows; with 60 rows it is a little less.', '0.632': 'That is the chance it is drawn at least once.', '0.983': 'That is the chance it is missed by a single draw.' },
    hints: ['One draw misses the row with probability 59/60.', 'The 60 draws are independent, so multiply: (59/60)⁶⁰.', '(59/60)⁶⁰ = ' + r3(left60) + ': about ' + Math.round(60 * left60) + ' of the 60 rows are out of bag for each tree.'], solution: String(r3(left60)), wrong: ['0.368', '0.632', '0.983'] });
  const bdraw = (() => { const r = ML.rng(17); return Array.from({ length: 8 }, () => r.int(8) + 1); })();
  const oobRows = [1, 2, 3, 4, 5, 6, 7, 8].filter((i) => !bdraw.includes(i));
  ex({ id: 'tr-x-bootstrap-set', chapter: 'tr-forest', title: 'The out-of-bag rows', level: 1, kind: 'set',
    text: ['Eight training rows, numbered 1 to 8. A tree’s bootstrap sample drew the rows ' + bdraw.join(', ') + '. Which rows are out of bag for this tree? Write them as a set, like {2, 7}.'],
    answer: '{' + oobRows.join(', ') + '}',
    hints: ['Out of bag means never drawn.', 'Cross off every number that appears in the draw, however many times.', 'The draw holds ' + [...new Set(bdraw)].sort((a, b) => a - b).join(', ') + ', so ' + oobRows.join(' and ') + ' were never drawn: {' + oobRows.join(', ') + '}.'],
    solution: '{' + oobRows.join(', ') + '}', wrong: ['{' + [...new Set(bdraw.filter((v, i) => bdraw.indexOf(v) !== i))].join(', ') + '}', '{' + oobRows[0] + '}'] });
  // out-of-bag votes of five trees on six rows: the trees that left each row out, and what they said
  const OOB = [[1, [1, 1, 0]], [0, [0]], [1, [0, 0, 1]], [0, [1, 0, 0, 0]], [1, []], [0, [1, 1]]];
  const voted = OOB.filter(([, v]) => v.length), rightN = voted.filter(([t, v]) => (ML.sum(v) * 2 > v.length ? 1 : 0) === t).length, oobAcc = rightN / voted.length;
  ex({ id: 'tr-x-oob-vote', chapter: 'tr-forest', title: 'An out-of-bag score by hand', level: 3, kind: 'number',
    text: ['A forest of five trees. For each of six training rows: its true class, and the votes of the trees that left it out of their samples.', { table: [['Row', 'True class', 'Out-of-bag votes']].concat(OOB.map(([t, v], i) => [String(i + 1), String(t), v.length ? v.join(', ') : 'none'])) }, 'Each row takes the majority of its out-of-bag votes. What is the out-of-bag accuracy? Give three decimals.'],
    answer: r3(oobAcc), tolerance: 0.0005, whyNot: { [String(r3(rightN / OOB.length))]: 'A row with no out-of-bag vote cannot be scored: leave it out of the count.', '*': 'Take the majority of each row’s votes, compare it with the true class, and divide by the number of rows that had votes.' },
    hints: ['Row 5 has no out-of-bag votes, so only five rows are scored.', 'Majorities: row 1 says 1, row 2 says 0, row 3 says 0, row 4 says 0, row 6 says 1.', rightN + ' of the ' + voted.length + ' scored rows are right: ' + r3(oobAcc) + '.'], solution: String(r3(oobAcc)), wrong: [String(r3(rightN / OOB.length)), '0.8'] });
  ex({ id: 'tr-x-maxfeatures', chapter: 'tr-forest', title: 'Why only a few features per split', level: 2, kind: 'choice',
    text: ['A random forest offers each split only √d of the d features, drawn at random. What is that for?'],
    options: ['It makes the trees less alike, so their errors are less correlated and the vote improves', 'It makes each tree more accurate', 'It removes the need for bootstrap samples', 'It lets the forest extrapolate beyond the training range'], answer: 0,
    whyNot: { 1: 'Each tree usually gets slightly worse: sometimes the best feature is not on offer.', 2: 'Forests use both bootstrap samples and feature sampling.', 3: 'A forest averages trees, and no tree extrapolates.' },
    hints: ['Think of one very strong feature and plain bagging.', 'Every bagged tree then starts with the same question, so they make similar mistakes, and a vote of similar voters gains little.', 'Hiding features at random forces variety: on the small loans, 6 different features start a tree instead of 4, and the forest beats the bagged trees, 75% to 72%.'], solution: 0, wrong: [1, 3] });
  ex({ id: 'tr-x-importance', chapter: 'tr-forest', title: 'Which importance a noise column fools', level: 2, kind: 'choice',
    text: ['A forest predicts exam passes. A column of pure noise was added. Its impurity importance is 0.134 of the total; shuffling it on fresh rows changes the accuracy by −4.1 points (it went up a little). What should you conclude?'],
    options: ['The noise column does not matter: impurity importance credits any column the trees happened to split on in training', 'The noise column matters a lot: 13% of the impurity drop is a lot', 'Permutation importance is broken: it cannot be negative', 'Both say the same thing'], answer: 0,
    whyNot: { 1: 'Deep trees find splits in any column with many distinct values; those splits fit training noise and say nothing about fresh rows.', 2: 'A negative drop just means that shuffling happened to help a little, by chance: the model does not use that column for anything real.', 3: 'One credits it with 13% and the other with nothing.' },
    hints: ['Impurity importance is measured on the training data, during the growth of the trees.', 'Permutation importance asks how much the score on fresh rows depends on the column.', 'Trust permutation here: the noise column carries no information, and its impurity share is an artefact of splitting on training noise.'], solution: 0, wrong: [1, 2] });
  const base = 0.85, shuf = [0.78, 0.8, 0.77, 0.79, 0.81], drop = base - ML.mean(shuf);
  ex({ id: 'tr-x-perm-drop', chapter: 'tr-forest', title: 'A permutation importance', level: 1, kind: 'number',
    text: ['A forest scores ' + base + ' accuracy on its test rows. The column rain is shuffled five times, and the accuracy becomes ' + shuf.join(', ') + '. What is the permutation importance of rain, the mean drop in accuracy? Give two decimals.'],
    answer: r2(drop), tolerance: 0.005, whyNot: { [String(r2(ML.mean(shuf)))]: 'That is the mean accuracy after shuffling; the importance is how far it fell from ' + base + '.', '*': 'Subtract the mean of the five shuffled scores from ' + base + '.' },
    hints: ['Average the five shuffled scores.', 'Their mean is ' + r2(ML.mean(shuf)) + '.', base + ' − ' + r2(ML.mean(shuf)) + ' = ' + r2(drop) + '.'], solution: String(r2(drop)), wrong: [String(r2(ML.mean(shuf))), '0.08'] });
  const moonsW = { data: { set: 'moons', n: 200, seed: 9, opts: { noise: 0.35 } }, hidden: { seed: 91, n: 400 }, metric: 'accuracy', target: 0.84 };
  ex(Object.assign({ id: 'tr-x-wf-moons', chapter: 'tr-forest', title: 'Hold a noisy tree back', level: 3, kind: 'workflow',
    text: ['Two moons with heavy noise. A tree grown until its leaves are pure wraps every noisy point. Build a workflow with a decision tree that reaches accuracy 0.84 on hidden points.'],
    limits: { blocks: ['split', 'tree', 'train', 'evaluate'], models: ['tree'] },
    hints: ['Grow the tree without limits first and compare its training and test scores.', 'Limit the depth (or raise the minimum leaf size) and watch the test score; depth 3 is a little too shallow here.', 'Data → Split → Decision tree with max depth 5 → Train → Evaluate: about 0.85.'],
    solution: pipe(moonsW.data, 'tree', { maxDepth: 5 }), wrong: [pipe(moonsW.data, 'tree', { maxDepth: 0 }), pipe(moonsW.data, 'tree', { maxDepth: 3 })] }, moonsW));

  // =========================================================================================================
  // Chapter 4: boosting
  // =========================================================================================================
  const AX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], AY = [0, 0, 0, 1, 0, 1, 1, 1, 0, 1];
  const ada = new ML.AdaBoostClassifier({ nEstimators: 2 }).fit(col(AX), AY);
  const eps = ada.errors[0], alpha = ada.alphas[0], wNew = ada.trace[1].weights, wrong1 = ada.trace[0].wrong;
  const iw = wrong1.indexOf(true), ir = wrong1.indexOf(false);
  const adaText = 'Ten points x = 1 to 10 with classes ' + AY.join(', ') + ', each starting with weight 0.1. The first stump says class 0 for x ≤ ' + ada.trees[0].nodes[0].threshold + ' and class 1 above.';
  ex({ id: 'tr-x-ada-alpha', chapter: 'tr-boost', title: 'The say of the first stump', level: 2, kind: 'number',
    text: [adaText, 'AdaBoost gives the stump the say α = ln((1 − ε) / ε), where ε is the weight of the points it gets wrong. What is α? Give three decimals.'],
    answer: r3(alpha), tolerance: 0.0015, whyNot: { [String(r3(alpha / 2))]: 'That is with a factor ½ in front, as some versions write it; this one uses ln((1 − ε)/ε) itself.', [String(r3(eps))]: 'That is ε, the weighted error.' },
    hints: ['Which points does the stump get wrong? Look above ' + ada.trees[0].nodes[0].threshold + ' for class 0, and below for class 1.', 'It misses x = ' + AX.filter((_, i) => wrong1[i]).join(' and ') + ', so ε = ' + r3(eps) + '.', 'α = ln(' + r3(1 - eps) + ' / ' + r3(eps) + ') = ln 4 = ' + r3(alpha) + '.'], solution: String(r3(alpha)), wrong: [String(r3(alpha / 2)), String(r3(eps))] });
  ex({ id: 'tr-x-ada-weights', chapter: 'tr-boost', title: 'The weights for round two', level: 3, kind: 'table',
    text: [adaText, 'Its weighted error is ε = ' + r3(eps) + ' and its say α = ' + r3(alpha) + '. AdaBoost multiplies the weight of each misclassified point by e^α, then rescales all ten weights so they add up to 1. Fill in the new weights (four decimals).'],
    rows: ['a point it got right', 'a point it got wrong', 'all ten together'], cols: ['new weight'], corner: 'point',
    cells: { 'a point it got right|new weight': r4(wNew[ir]), 'a point it got wrong|new weight': r4(wNew[iw]), 'all ten together|new weight': 1 }, tolerance: 0.0006,
    hints: ['e^α = (1 − ε)/ε = 4, so each wrong point goes from 0.1 to 0.4.', 'Before rescaling the weights add up to 8 · 0.1 + 2 · 0.4 = 1.6.', 'Divide by 1.6: right points ' + r4(wNew[ir]) + ', wrong points ' + r4(wNew[iw]) + '; the total is 1 again. The two mistakes now carry half of all the weight.'],
    solution: { 'a point it got right|new weight': String(r4(wNew[ir])), 'a point it got wrong|new weight': String(r4(wNew[iw])), 'all ten together|new weight': '1' },
    wrong: [{ 'a point it got right|new weight': '0.1', 'a point it got wrong|new weight': '0.4', 'all ten together|new weight': '1.6' }] });
  const reweight = (w, wrong) => { const W = ML.sum(w), e = ML.sum(w.filter((_, i) => wrong[i])) / W, a = Math.log((1 - e) / e); const v = w.map((x, i) => (wrong[i] ? x * Math.exp(a) : x)), s = ML.sum(v); return v.map((x) => x / s); };
  ex({ id: 'tr-x-code-ada', chapter: 'tr-boost', title: 'Write the AdaBoost reweighting', level: 3, kind: 'code',
    text: ['Write `reweight(w, wrong)`: w are the current weights (they need not add up to 1), wrong[i] is true when the learner misclassified point i. Compute ε = (weight of the wrong points) / (total weight) and α = ln((1 − ε)/ε), multiply each wrong point’s weight by e^α, and return the weights rescaled to add up to 1. The cases always have 0 < ε < 0.5.'],
    fn: 'reweight', params: ['w', 'wrong'], start: 'function reweight(w, wrong) {\n  \n}', tolerance: 1e-9, ref: reweight,
    cases: [[Array(10).fill(0.1), wrong1], [[0.25, 0.25, 0.25, 0.25], [false, false, false, true]], [[1, 2, 3, 4, 10], [true, false, false, false, false]], [wNew, ada.trace[1].wrong], [[0.5, 0.2, 0.2, 0.1], [false, true, false, true]]],
    hints: ['Sum all the weights, and the weights of the wrong points; ε is their ratio.', 'e^α is simply (1 − ε)/ε.', 'function reweight(w, wrong) {\n  const W = w.reduce((a, b) => a + b, 0);\n  const e = w.reduce((s, x, i) => s + (wrong[i] ? x : 0), 0) / W;\n  const f = (1 - e) / e;\n  const v = w.map((x, i) => (wrong[i] ? x * f : x));\n  const s = v.reduce((a, b) => a + b, 0);\n  return v.map((x) => x / s);\n}'],
    solution: 'function reweight(w, wrong) {\n  const W = w.reduce((a, b) => a + b, 0);\n  const e = w.reduce((s, x, i) => s + (wrong[i] ? x : 0), 0) / W;\n  const f = (1 - e) / e;\n  const v = w.map((x, i) => (wrong[i] ? x * f : x));\n  const s = v.reduce((a, b) => a + b, 0);\n  return v.map((x) => x / s);\n}',
    wrong: ['function reweight(w, wrong) {\n  const W = w.reduce((a, b) => a + b, 0);\n  const e = w.reduce((s, x, i) => s + (wrong[i] ? x : 0), 0) / W;\n  const f = (1 - e) / e;\n  return w.map((x, i) => (wrong[i] ? x * f : x));\n}', 'function reweight(w, wrong) {\n  const W = w.reduce((a, b) => a + b, 0);\n  const e = w.reduce((s, x, i) => s + (wrong[i] ? x : 0), 0) / W;\n  const a = Math.log((1 - e) / e);\n  const v = w.map((x, i) => (wrong[i] ? x * a : x));\n  const s = v.reduce((a, b) => a + b, 0);\n  return v.map((x) => x / s);\n}'] });

  const GY = [2, 3.5, 4, 7, 8.5, 9];
  const gb = new ML.GradientBoostingRegressor({ nEstimators: 1, learningRate: 0.5, maxDepth: 1 }).fit(col(RX), GY), F0 = gb.init[0], gp = gb.predict(col(RX));
  const gcells = {}; RX.forEach((x, i) => { gcells['x = ' + x + '|residual'] = r3(GY[i] - F0); gcells['x = ' + x + '|after stage 1'] = r3(gp[i]); });
  ex({ id: 'tr-x-gb-stage', chapter: 'tr-boost', title: 'One gradient-boosting stage by hand', level: 3, kind: 'table',
    text: ['Six points x = ' + RX.join(', ') + ' with y = ' + GY.join(', ') + '. Gradient boosting starts from F₀ = the mean of y. Stage 1 fits a stump to the residuals y − F₀; the stump cuts at x ≤ ' + gb.stages[0][0].nodes[0].threshold + ', and each side predicts the mean residual of its points. The stump is added at rate ν = 0.5. Fill in each residual and the model’s prediction after stage 1 (three decimals).'],
    rows: RX.map((x) => 'x = ' + x), cols: ['residual', 'after stage 1'], corner: 'point', cells: gcells, tolerance: 0.006,
    hints: ['F₀ = ' + r3(F0) + ', so the residuals are y − ' + r3(F0) + '.', 'The left three residuals average ' + r3(ML.mean(GY.slice(0, 3)) - F0) + ' and the right three ' + r3(ML.mean(GY.slice(3)) - F0) + '.', 'Prediction = ' + r3(F0) + ' + 0.5 × (leaf mean): ' + r3(gp[0]) + ' on the left and ' + r3(gp[5]) + ' on the right.'],
    solution: Object.fromEntries(Object.entries(gcells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.fromEntries(Object.entries(gcells).map(([k, v], j) => [k, String(k.endsWith('stage 1') ? r3(F0 + 2 * (v - F0)) : v)]))] });
  ex({ id: 'tr-x-shrink-choice', chapter: 'tr-boost', title: 'Big steps or small ones', level: 2, kind: 'choice',
    text: ['Gradient boosting with depth-3 trees on a noisy sine, at three learning rates. After 60 stages:', { table: [['Rate ν', 'Training error', 'Test error'], ['1', '0', '0.285'], ['0.3', '0.000358', '0.272'], ['0.1', '0.0212', '0.195']] }, 'What does the table show?'],
    options: ['The smallest rate fits the training points least closely and generalises best', 'The largest rate is best: it reaches zero training error', 'The rate makes no difference after enough stages', 'Smaller rates always need fewer stages'], answer: 0,
    whyNot: { 1: 'Zero training error on noisy data means the noise has been learned; its test error is the worst.', 2: 'The test errors differ a lot: 0.285 against 0.195.', 3: 'The opposite: small steps need more of them.' },
    hints: ['Compare the two columns for each rate.', 'ν = 1 fits the training points exactly; ν = 0.1 does not.', 'ν = 0.1 keeps some training error (0.0212) and has the lowest test error (0.195): shrinkage regularises.'], solution: 0, wrong: [1, 2] });
  const curveW = { data: { set: 'curve', n: 120, seed: 4 }, hidden: { seed: 41, n: 300 }, metric: 'r2', target: 0.95 };
  ex(Object.assign({ id: 'tr-x-wf-curve', chapter: 'tr-boost', title: 'Boost a wiggly curve', level: 3, kind: 'workflow',
    text: ['A target that wiggles along one input. A straight line misses its bends and a single tree draws coarse steps. Build a workflow that reaches R² 0.95 on hidden points.'],
    limits: { blocks: ['split', 'linreg', 'tree', 'gboost', 'train', 'evaluate'], models: ['linreg', 'tree', 'gboost'], params: { gboost: { nEstimators: [1, 100] } } },
    hints: ['Try linear regression and a tree first: neither reaches 0.95.', 'Gradient boosting adds many small trees; with too few stages at rate 0.1 it has not got far yet.', 'Gradient boosting with its defaults (60 stages, rate 0.1, depth 3): R² about 0.956.'],
    solution: pipe(curveW.data, 'gboost', {}), wrong: [pipe(curveW.data, 'gboost', { nEstimators: 5 }), pipe(curveW.data, 'tree', { maxDepth: 4 })] }, curveW));
  const circW = { data: { set: 'circles', n: 300, seed: 3, opts: { noise: 0.08 } }, hidden: { seed: 33, n: 400 }, metric: 'accuracy', target: 0.95 };
  ex(Object.assign({ id: 'tr-x-wf-circles', chapter: 'tr-boost', title: 'A ring with straight cuts', level: 2, kind: 'workflow',
    text: ['A disc inside a ring. A tree can only cut along the axes, so it must box the disc in. Build a workflow with a decision tree no deeper than 4 that reaches accuracy 0.95 on hidden points.'],
    limits: { blocks: ['split', 'tree', 'train', 'evaluate'], models: ['tree'], params: { tree: { maxDepth: [1, 4] } } },
    hints: ['Four straight cuts make a box; each cut needs a level of the tree, more or less.', 'Depth 3 is not quite enough to close the box on every side.', 'Decision tree with max depth 4: about 0.975.'],
    solution: pipe(circW.data, 'tree', { maxDepth: 4 }), wrong: [pipe(circW.data, 'tree', { maxDepth: 3 }), pipe(circW.data, 'tree', { maxDepth: 2 })] }, circW));
  ex({ id: 'tr-x-which-model', chapter: 'tr-boost', title: 'The right tree model for the job', level: 2, kind: 'choice',
    text: ['A clinic wants a model that nurses can check by hand at the bedside: five yes-or-no questions at most, each on one measurement. Accuracy matters, but every prediction must be explainable in a sentence. Which model fits?'],
    options: ['A single decision tree of depth at most 5', 'A random forest of 500 trees', 'Gradient boosting with 300 stages', 'AdaBoost with 50 stumps'], answer: 0,
    whyNot: { 1: 'Its vote of 500 trees cannot be followed at the bedside.', 2: 'A sum of 300 trees is accurate but cannot be read as a few questions.', 3: 'Even 50 weighted stumps are too many to add up by hand.' },
    hints: ['What can a person follow without a computer?', 'Each ensemble sums or votes over many trees.', 'A single shallow tree is a checklist: at most five questions down one path. Use the ensembles when accuracy matters more than reading the model.'], solution: 0, wrong: [1, 2] });
})(globalThis.Nooral = globalThis.Nooral || {});
