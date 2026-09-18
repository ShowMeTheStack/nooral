// Part 2, splitting and evaluation: the exercises. Every number in a text, an answer or a hint is worked out here by
// the engines (S.Metrics for every score and curve, S.ML for splits, folds and the validation curve, S.Data) from the
// small sets of predictions written out below, or by the workflow runner for the workflow task.
(function (S) {
  'use strict';
  const C = S.Content, ML = S.ML, M = S.Metrics, WF = S.WF;
  if (!C || !ML || !M || !WF || C.exercises.has('ev-x-split-time')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const n2 = (v) => String(r2(v)), n3 = (v) => String(r3(v));
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };
  /** labels and predictions made from the four counts of a binary confusion matrix (1 = positive) */
  const fromCounts = (tp, fn, fp, tn) => {
    const y = [], p = [];
    const add = (k, a, b) => { for (let i = 0; i < k; i++) { y.push(a); p.push(b); } };
    add(tp, 1, 1); add(fn, 1, 0); add(fp, 0, 1); add(tn, 0, 0);
    return { y, p };
  };
  /** labels and predictions made from a k × k matrix of counts */
  const fromMatrix = (labels, counts) => { const y = [], p = []; counts.forEach((r, i) => r.forEach((c, j) => { for (let k = 0; k < c; k++) { y.push(labels[i]); p.push(labels[j]); } })); return { y, p }; };
  const cmTable = (rows, cols, counts) => ({ table: [['true \\ predicted', ...cols], ...rows.map((r, i) => [r, ...counts[i].map(String)])] });

  // =========================================================================================================
  // Chapter 1: rows the model never saw
  // =========================================================================================================
  const topts = ['Train on the first two years, validate on the next six months, test on the last six months', 'Shuffle the days, then cut 60/20/20', 'Stratify by weekday, then cut at random', 'Use 5-fold cross-validation with shuffled folds'];
  C.exercise({ id: 'ev-x-split-time', chapter: 'ev-split', title: 'A split for time-ordered data', level: 1, kind: 'choice',
    text: ['A bakery has three years of daily sales and wants a model that predicts tomorrow’s sales. How should the days be split to get an honest estimate of how the model will do?'],
    options: topts, answer: 0,
    whyNot: { 1: 'Shuffled, the model trains on days after the ones it is tested on, and next to each test day sit its near-identical neighbours: the score is far too kind.', 2: 'Stratifying keeps weekday shares even, but still mixes the future into training.', 3: 'Shuffled folds have the same problem as a shuffled split: they test on the past with knowledge of the future.' },
    hints: ['The model will be used on days that come after all the days it learned from.', 'Neighbouring days are very alike: a test day surrounded by training days is easy.', 'Cut by time: train on the earliest stretch, validate on the next, test on the latest. That is how the model will be used.'],
    solution: 0, wrong: [1, 3] });

  const sy = Array.from({ length: 300 }, (_, i) => (i < 30 ? 1 : 0));
  const s1 = ML.trainTestSplit(300, { testSize: 0.2, stratify: sy, seed: 1 }), restY = s1.train.map((i) => sy[i]);
  const s2 = ML.trainTestSplit(restY.length, { testSize: 0.25, stratify: restY, seed: 1 });
  const cnt = (idx, ys, v) => idx.filter((i) => ys[i] === v).length;
  const sc = { 'training|faults': cnt(s2.train, restY, 1), 'training|normal': cnt(s2.train, restY, 0), 'validation|faults': cnt(s2.test, restY, 1), 'validation|normal': cnt(s2.test, restY, 0), 'test|faults': cnt(s1.test, sy, 1), 'test|normal': cnt(s1.test, sy, 0) };
  C.exercise({ id: 'ev-x-strat-counts', chapter: 'ev-split', title: 'A stratified 60/20/20 split', level: 2, kind: 'table',
    text: ['300 pump readings, 30 of them faults. They are split, stratified, into 60% training, 20% validation and 20% test. How many faults and normal readings land in each part?'],
    rows: ['training', 'validation', 'test'], cols: ['faults', 'normal'], corner: 'part', cells: sc,
    hints: ['Stratified: every part keeps the shares of the whole, 10% faults.', 'The parts hold 180, 60 and 60 rows.', 'Faults: ' + sc['training|faults'] + ', ' + sc['validation|faults'] + ' and ' + sc['test|faults'] + '; normal: ' + sc['training|normal'] + ', ' + sc['validation|normal'] + ' and ' + sc['test|normal'] + '.'],
    solution: Object.fromEntries(Object.entries(sc).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(sc).map(([k, v]) => [k, String(v)])), { 'test|faults': '3', 'test|normal': '57' })] });

  const popts = ['Optimistic: the test set has been used to choose, so it is no longer unseen', 'Honest: each model was scored on rows it never trained on', 'Pessimistic: twenty tries add noise', 'Honest, as long as the test set is large'];
  C.exercise({ id: 'ev-x-test-peek', chapter: 'ev-split', title: 'Peeking at the test set', level: 2, kind: 'choice',
    text: ['An analyst trains twenty versions of a model with different settings, scores each on the test set, and reports the best test score as the expected performance. How will that number compare with performance on new data?'],
    options: popts, answer: 0,
    whyNot: { 1: 'Each model is scored on unseen rows, but the choice of the best one was made by looking at them: the winner is partly the luckiest.', 2: 'Taking the best of twenty favours lucky scores, so the bias is upward.', 3: 'A larger test set shrinks the effect but does not remove it.' },
    hints: ['Which set was used to make a choice?', 'The best of twenty noisy scores is above the typical one, even if every model is equally good.', 'Optimistic. Choose among the twenty on a validation set or by cross-validation, then score the one winner once on the test set.'],
    solution: 0, wrong: [1, 2] });

  const kf = ML.kFold(23, 5), kcells = {};
  kf.forEach((f, i) => { kcells['fold ' + (i + 1) + '|test rows'] = f.test.length; kcells['fold ' + (i + 1) + '|training rows'] = f.train.length; });
  C.exercise({ id: 'ev-x-fold-sizes', chapter: 'ev-split', title: 'Five folds of 23 rows', level: 1, kind: 'table',
    text: ['23 rows are cut into 5 folds for cross-validation; as usual, the first folds take one extra row when the rows do not divide evenly. For each fold, how many rows are tested and how many trained on?'],
    rows: kf.map((_, i) => 'fold ' + (i + 1)), cols: ['test rows', 'training rows'], corner: 'fold', cells: kcells,
    hints: ['23 = 5 × 4 + 3: four rows per fold, and 3 left over.', 'The first 3 folds take one extra row each.', 'Test rows: ' + kf.map((f) => f.test.length).join(', ') + '; training rows: ' + kf.map((f) => f.train.length).join(', ') + '. Every row is tested exactly once.'],
    solution: Object.fromEntries(Object.entries(kcells).map(([k, v]) => [k, String(v)])) });

  const CVS = [0.82, 0.9, 0.86, 0.78, 0.84], cvm = ML.mean(CVS);
  C.exercise({ id: 'ev-x-cv-mean', chapter: 'ev-split', title: 'Five scores, one estimate', level: 1, kind: 'number',
    text: ['5-fold cross-validation gives the accuracies ' + CVS.join(', ') + '. What is the cross-validated accuracy? Two decimal places.'],
    answer: r2(cvm), tolerance: 0.005,
    whyNot: { 0.9: 'That is the best fold. The estimate is the mean of all five.', 0.86: 'That is the middle fold in the list, not the mean.', '*': 'Add the five scores and divide by 5.' },
    hints: ['Every row was tested once, by a model that never saw it; each fold gives one score.', 'Sum: ' + n2(CVS.reduce((a, b) => a + b, 0)) + '.', n2(CVS.reduce((a, b) => a + b, 0)) + ' / 5 = ' + n2(cvm) + ', with the folds ranging from ' + Math.min(...CVS) + ' to ' + Math.max(...CVS) + ': report the spread too.'],
    solution: n2(cvm), wrong: ['0.9', '0.86'] });

  // =========================================================================================================
  // Chapter 2: counting the mistakes
  // =========================================================================================================
  const CT = ['fault', 'normal', 'normal', 'fault', 'normal', 'normal', 'fault', 'normal', 'normal', 'normal', 'fault', 'normal'];
  const CP = ['fault', 'normal', 'fault', 'normal', 'normal', 'normal', 'fault', 'normal', 'fault', 'normal', 'fault', 'normal'];
  const CL = ['fault', 'normal'], cm1 = M.confusion(CT, CP, { labels: CL }).counts;
  C.exercise({ id: 'ev-x-cm-fill', chapter: 'ev-confusion', title: 'Count a confusion matrix', level: 1, kind: 'matrix', diag: true,
    text: ['Twelve pump minutes, the truth and the model’s prediction for each:', { table: [['minute', ...CT.map((_, i) => String(i + 1))], ['truth', ...CT], ['predicted', ...CP]] }, 'Fill in the confusion matrix: rows are the true classes, columns the predicted ones, with fault as the positive class.'],
    rows: CL, cols: CL, corner: 'true \\ predicted', answer: cm1, tolerance: 0, total: 12,
    hints: ['Go minute by minute and add one to the cell (truth, prediction).', 'Fault predicted as fault: minutes 1, 7 and 11. Fault predicted as normal: minute 4.', 'TP = ' + cm1[0][0] + ', FN = ' + cm1[0][1] + ', FP = ' + cm1[1][0] + ', TN = ' + cm1[1][1] + ': the false positives are minutes 3 and 9.'],
    solution: cm1.map((r) => r.map(String)), wrong: [[cm1[0], cm1[1]].map((r, i) => (i === 0 ? [r[0], r[1]] : [r[0] - 1, r[1] + 1]).map(String)), [[cm1[0][0], cm1[1][0]], [cm1[0][1], cm1[1][1]]].map((r) => r.map(String))] });

  const cmSol = 'function confusion(yTrue, yPred, labels) {\n  const m = labels.map(() => labels.map(() => 0));\n  for (let k = 0; k < yTrue.length; k++) {\n    const i = labels.indexOf(yTrue[k]), j = labels.indexOf(yPred[k]);\n    if (i >= 0 && j >= 0) m[i][j]++;\n  }\n  return m;\n}';
  C.exercise({ id: 'ev-x-code-cm', chapter: 'ev-confusion', title: 'Write the confusion matrix', level: 1, kind: 'code',
    text: ['Write `confusion(yTrue, yPred, labels)`: return a square array of counts where row i, column j counts the examples whose true label is `labels[i]` and whose predicted label is `labels[j]`. Labels can be numbers or strings; ignore any example whose label is not in `labels`.'],
    fn: 'confusion', params: ['yTrue', 'yPred', 'labels'], start: 'function confusion(yTrue, yPred, labels) {\n  \n}',
    ref: (t, p, l) => M.confusion(t, p, { labels: l }).counts,
    cases: [[[1, 0, 1, 1, 0], [1, 1, 0, 1, 0], [1, 0]], [['a', 'b', 'c', 'a'], ['a', 'c', 'c', 'b'], ['a', 'b', 'c']], [[0, 0, 0], [0, 0, 0], [0, 1]], [[2, 1, 2, 9], [2, 2, 1, 2], [1, 2]]],
    hints: ['Start from a k × k array of zeros, k = labels.length.', 'For each example, find the row from its true label and the column from its prediction with labels.indexOf, and add one.', cmSol],
    solution: cmSol,
    wrong: ['function confusion(yTrue, yPred, labels) {\n  const m = labels.map(() => labels.map(() => 0));\n  for (let k = 0; k < yTrue.length; k++) m[labels.indexOf(yPred[k])][labels.indexOf(yTrue[k])]++;\n  return m;\n}', 'function confusion(yTrue, yPred, labels) {\n  const m = labels.map(() => labels.map(() => 0));\n  for (let k = 0; k < yTrue.length; k++) { const i = labels.indexOf(yTrue[k]), j = labels.indexOf(yPred[k]); m[i][j]++; }\n  return m;\n}'] });

  const L5 = fromCounts(27, 32, 17, 164), prf5 = M.precisionRecallF1(L5.y, L5.p);
  const scc = { 'accuracy|score': r3(M.accuracy(L5.y, L5.p).value), 'precision|score': r3(prf5.precision), 'recall|score': r3(prf5.recall), 'specificity|score': r3(M.specificity(L5.y, L5.p).value), 'F1|score': r3(prf5.f1) };
  C.exercise({ id: 'ev-x-scores', chapter: 'ev-confusion', title: 'Five scores from four counts', level: 1, kind: 'table',
    text: ['A loan model at threshold 0.5, with default as the positive class:', cmTable(['default', 'repaid'], ['default', 'repaid'], [[27, 32], [17, 164]]), 'Work out the five scores, to three decimal places.'],
    rows: ['accuracy', 'precision', 'recall', 'specificity', 'F1'], cols: ['score'], corner: '', cells: scc, tolerance: 0.0015,
    hints: ['TP = 27, FN = 32, FP = 17, TN = 164, 240 in all.', 'Accuracy = (27 + 164) / 240; precision = 27 / (27 + 17); recall = 27 / (27 + 32); specificity = 164 / (164 + 17).', 'Accuracy ' + scc['accuracy|score'] + ', precision ' + scc['precision|score'] + ', recall ' + scc['recall|score'] + ', specificity ' + scc['specificity|score'] + ', F1 = 2PR / (P + R) = ' + scc['F1|score'] + '.'],
    solution: Object.fromEntries(Object.entries(scc).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(scc).map(([k, v]) => [k, String(v)])), { 'precision|score': String(scc['recall|score']), 'recall|score': String(scc['precision|score']) })] });

  const F8 = fromCounts(8, 8, 2, 30), prf8 = M.precisionRecallF1(F8.y, F8.p);
  C.exercise({ id: 'ev-x-f1', chapter: 'ev-confusion', title: 'The harmonic mean', level: 1, kind: 'number',
    text: ['A spam filter has precision ' + prf8.precision + ' and recall ' + prf8.recall + '. What is its F1? Three decimal places.'],
    answer: r3(prf8.f1), tolerance: 0.0015,
    whyNot: { [n3((prf8.precision + prf8.recall) / 2)]: 'That is the plain average. F1 is the harmonic mean, 2PR / (P + R), which stays nearer the smaller of the two.', [n3(prf8.precision * prf8.recall)]: 'That is the product; divide 2PR by P + R.' },
    hints: ['F1 = 2 · P · R / (P + R).', '2 × ' + prf8.precision + ' × ' + prf8.recall + ' = ' + n2(2 * prf8.precision * prf8.recall) + ', and P + R = ' + n2(prf8.precision + prf8.recall) + '.', n2(2 * prf8.precision * prf8.recall) + ' / ' + n2(prf8.precision + prf8.recall) + ' = ' + n3(prf8.f1) + ', below the plain average ' + n3((prf8.precision + prf8.recall) / 2) + '.'],
    solution: n3(prf8.f1), wrong: [n3((prf8.precision + prf8.recall) / 2), n3(prf8.precision * prf8.recall)] });

  const prfSol = 'function prf(yTrue, yPred) {\n  let tp = 0, fp = 0, fn = 0;\n  for (let i = 0; i < yTrue.length; i++) {\n    if (yPred[i] === 1 && yTrue[i] === 1) tp++;\n    else if (yPred[i] === 1) fp++;\n    else if (yTrue[i] === 1) fn++;\n  }\n  const p = tp + fp ? tp / (tp + fp) : 0, r = tp + fn ? tp / (tp + fn) : 0;\n  return [p, r, p + r ? (2 * p * r) / (p + r) : 0];\n}';
  C.exercise({ id: 'ev-x-code-prf', chapter: 'ev-confusion', title: 'Write precision, recall and F1', level: 2, kind: 'code',
    text: ['Write `prf(yTrue, yPred)` for labels 0 and 1, with 1 the positive class. Return `[precision, recall, f1]`. When a ratio would divide by zero (nothing flagged, or no positives), return 0 for it, as the usual libraries do.'],
    fn: 'prf', params: ['yTrue', 'yPred'], start: 'function prf(yTrue, yPred) {\n  \n}', tolerance: 1e-9,
    ref: (t, p) => { const r = M.precisionRecallF1(t, p, { average: 'binary', posLabel: 1 }); return [r.precision, r.recall, r.f1]; },
    cases: [[[1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0]], [[1, 1, 0, 0], [0, 0, 0, 0]], [[0, 0, 1], [1, 1, 1]], [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]], [[0, 1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1, 0, 0]]],
    hints: ['Count TP, FP and FN in one pass.', 'precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2PR / (P + R); guard every division.', prfSol],
    solution: prfSol,
    wrong: ['function prf(yTrue, yPred) {\n  let tp = 0, fp = 0, fn = 0;\n  for (let i = 0; i < yTrue.length; i++) { if (yPred[i] === 1 && yTrue[i] === 1) tp++; else if (yPred[i] === 1) fp++; else if (yTrue[i] === 1) fn++; }\n  const p = tp / (tp + fp), r = tp / (tp + fn);\n  return [p, r, (2 * p * r) / (p + r)];\n}', 'function prf(yTrue, yPred) {\n  let tp = 0, fp = 0, fn = 0;\n  for (let i = 0; i < yTrue.length; i++) { if (yPred[i] === 1 && yTrue[i] === 1) tp++; else if (yPred[i] === 1) fp++; else if (yTrue[i] === 1) fn++; }\n  const p = tp + fn ? tp / (tp + fn) : 0, r = tp + fp ? tp / (tp + fp) : 0;\n  return [p, r, p + r ? (2 * p * r) / (p + r) : 0];\n}'] });

  const ropts = ['F1 (or average precision) for the fault class', 'Accuracy', 'Specificity', 'Micro-averaged F1 over both classes'];
  C.exercise({ id: 'ev-x-rare-metric', chapter: 'ev-confusion', title: 'A metric for rare faults', level: 1, kind: 'choice',
    text: ['One minute in two hundred is a fault. The plant wants a monitor that catches faults without drowning the night engineer in false alarms. Which single number should it compare monitors by?'],
    options: ropts, answer: 0,
    whyNot: { 1: 'A monitor that never raises an alarm is 99.5% accurate.', 2: 'Specificity only counts the normal minutes left alone: the do-nothing monitor scores 1.', 3: 'Over both classes, micro-averaged F1 equals accuracy, and the normal minutes swamp it.' },
    hints: ['Which numbers does a monitor that never alarms score well on?', 'The score must look at the fault class: how many faults are caught (recall) and how many alarms are real (precision).', 'F1 on the fault class combines both, and average precision does the same over every threshold. Accuracy, specificity and micro F1 are all near perfect for a monitor that does nothing.'],
    solution: 0, wrong: [1, 3] });

  const B4 = fromCounts(14, 6, 19, 361), ba = M.balancedAccuracy(B4.y, B4.p).value, acc4 = M.accuracy(B4.y, B4.p).value, mcc4 = M.mcc(B4.y, B4.p).value;
  const bTable = cmTable(['fault', 'normal'], ['fault', 'normal'], [[14, 6], [19, 361]]);
  C.exercise({ id: 'ev-x-balanced', chapter: 'ev-confusion', title: 'Balanced accuracy', level: 2, kind: 'number',
    text: ['A pump monitor on 400 minutes:', bTable, 'What is its balanced accuracy? Three decimal places.'],
    answer: r3(ba), tolerance: 0.0015,
    whyNot: { [n3(acc4)]: 'That is the plain accuracy, (14 + 361) / 400. Balanced accuracy averages the recall of each class.', [n3(14 / 20)]: 'That is the recall of the fault class alone; average it with the normal class’s.' },
    hints: ['Balanced accuracy is the mean of the per-class recalls.', 'Fault: 14 / 20. Normal: 361 / 380.', '(' + 14 / 20 + ' + ' + 361 / 380 + ') / 2 = ' + n3(ba) + ', against a plain accuracy of ' + n3(acc4) + '.'],
    solution: n3(ba), wrong: [n3(acc4), n3(14 / 20)] });

  C.exercise({ id: 'ev-x-mcc', chapter: 'ev-confusion', title: 'The Matthews correlation', level: 2, kind: 'number',
    text: ['The same pump monitor:', bTable, 'What is its Matthews correlation coefficient, (TP·TN − FP·FN) / √((TP + FP)(TP + FN)(TN + FP)(TN + FN))? Three decimal places.'],
    answer: r3(mcc4), tolerance: 0.0015,
    whyNot: { [n3(acc4)]: 'That is the accuracy.', [n3(ba)]: 'That is the balanced accuracy.', '*': 'Put TP = 14, FN = 6, FP = 19, TN = 361 into the formula.' },
    hints: ['TP·TN − FP·FN = 14 × 361 − 19 × 6 = ' + (14 * 361 - 19 * 6) + '.', 'The four sums are TP + FP = 33, TP + FN = 20, TN + FP = 380 and TN + FN = 367.', (14 * 361 - 19 * 6) + ' / √(33 × 20 × 380 × 367) = ' + n3(mcc4) + ': a modest score, where the accuracy, ' + n3(acc4) + ', looks excellent.'],
    solution: n3(mcc4), wrong: [n3(acc4), n3(ba)] });

  const ML3 = ['ashbell', 'copperleaf', 'moonwort'];
  const MT = ['ashbell', 'moonwort', 'copperleaf', 'ashbell', 'moonwort', 'copperleaf', 'copperleaf', 'moonwort', 'ashbell', 'copperleaf', 'moonwort', 'moonwort', 'ashbell', 'copperleaf', 'moonwort'];
  const MP = ['ashbell', 'copperleaf', 'copperleaf', 'ashbell', 'moonwort', 'moonwort', 'copperleaf', 'moonwort', 'ashbell', 'copperleaf', 'copperleaf', 'moonwort', 'copperleaf', 'copperleaf', 'moonwort'];
  const cm3 = M.confusion(MT, MP, { labels: ML3 }).counts;
  C.exercise({ id: 'ev-x-cm-multi', chapter: 'ev-confusion', title: 'Three classes, fifteen flowers', level: 2, kind: 'matrix', diag: true,
    text: ['Fifteen wildflowers, their species and the model’s guess:', { table: [['flower', ...MT.map((_, i) => String(i + 1))], ['truth', ...MT.map((s) => s.slice(0, 4))], ['predicted', ...MP.map((s) => s.slice(0, 4))]] }, 'Fill in the 3 × 3 confusion matrix (rows: the truth; columns: the prediction). The table shortens the names: ashb, copp, moon.'],
    rows: ML3, cols: ML3, corner: 'true \\ predicted', answer: cm3, tolerance: 0, total: 15,
    hints: ['One flower, one cell: row of its true species, column of the guess.', 'The diagonal counts the right guesses: ' + cm3.map((r, i) => r[i]).reduce((a, b) => a + b, 0) + ' of 15.', 'Rows: ' + cm3.map((r, i) => ML3[i] + ' ' + r.join(', ')).join('; ') + '.'],
    solution: cm3.map((r) => r.map(String)), wrong: [cm3[0].map((_, j) => cm3.map((r) => String(r[j])))] });

  const RN = [[18, 2, 0], [1, 14, 5], [0, 6, 14]], rnD = fromMatrix(ML3, RN), rnM = M.confusion(rnD.y, rnD.p, { labels: ML3, normalize: 'true' }).matrix.map((r) => r.map(r2));
  const cnM = M.confusion(rnD.y, rnD.p, { labels: ML3, normalize: 'pred' }).matrix.map((r) => r.map(r2));
  C.exercise({ id: 'ev-x-row-norm', chapter: 'ev-confusion', title: 'Normalise by row', level: 2, kind: 'matrix', diag: true,
    text: ['A confusion matrix of wildflowers:', cmTable(ML3, ML3, RN), 'Normalise it by row, so each row shows where the flowers of one true species went. Two decimal places.'],
    rows: ML3, cols: ML3, corner: 'true \\ predicted', answer: rnM, tolerance: 0.006, rowsSumTo: 1,
    hints: ['Divide every cell by its row’s total.', 'The row totals are ' + RN.map((r) => r.reduce((a, b) => a + b, 0)).join(', ') + '.', 'Rows: ' + rnM.map((r) => r.join(', ')).join('; ') + '. The diagonal is each species’ recall.'],
    solution: rnM.map((r) => r.map(String)), wrong: [cnM.map((r) => r.map(String))] });

  const RN2 = [[18, 2, 0], [1, 14, 5], [0, 1, 3]], rn2 = fromMatrix(ML3, RN2);
  const mac2 = M.precisionRecallF1(rn2.y, rn2.p, { average: 'macro' }).f1, mic2 = M.precisionRecallF1(rn2.y, rn2.p, { average: 'micro' }).f1, wtd2 = M.precisionRecallF1(rn2.y, rn2.p, { average: 'weighted' }).f1, per2 = M.precisionRecallF1(rn2.y, rn2.p, { average: null }).f1;
  const mmc = { 'macro|F1': r3(mac2), 'weighted|F1': r3(wtd2), 'micro|F1': r3(mic2) };
  C.exercise({ id: 'ev-x-macro-micro', chapter: 'ev-confusion', title: 'Macro, weighted and micro F1', level: 3, kind: 'table',
    text: ['A model on 44 wildflowers, with moonwort rare:', cmTable(ML3, ML3, RN2), 'Work out the macro, weighted and micro averages of F1. Three decimal places.'],
    rows: ['macro', 'weighted', 'micro'], cols: ['F1'], corner: '', cells: mmc, tolerance: 0.0015,
    hints: ['First each class’s F1, one class against the rest: TP on the diagonal, FN the rest of its row, FP the rest of its column.', 'The class F1s are ' + per2.map(n3).join(', ') + ' (ashbell, copperleaf, moonwort), with ' + RN2.map((r) => r.reduce((a, b) => a + b, 0)).join(', ') + ' flowers.', 'Macro: their plain mean, ' + n3(mac2) + '. Weighted by class size: ' + n3(wtd2) + '. Micro pools the counts and equals the accuracy, ' + n3(mic2) + '. The rare moonwort pulls only the macro average down.'],
    solution: Object.fromEntries(Object.entries(mmc).map(([k, v]) => [k, String(v)])),
    wrong: [{ 'macro|F1': n3(mic2), 'weighted|F1': n3(wtd2), 'micro|F1': n3(mac2) }] });

  // =========================================================================================================
  // Chapter 3: thresholds and probabilities
  // =========================================================================================================
  const RY = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0], RS = [0.92, 0.81, 0.74, 0.66, 0.58, 0.47, 0.41, 0.33, 0.21, 0.12, 0.08, 0.05], rt = 0.5, ra = M.atThreshold(RY, RS, rt, 1);
  C.exercise({ id: 'ev-x-roc-point', chapter: 'ev-curves', title: 'One point on the ROC curve', level: 2, kind: 'point',
    text: ['Twelve applicants, their model scores and whether they defaulted (1) or repaid (0):', { table: [['score', ...RS.map(String)], ['defaulted', ...RY.map(String)]] }, 'At threshold ' + rt + ' (flag a score of at least ' + rt + '), where is the model on the ROC curve? Click the point (FPR, TPR).'],
    plot: { xr: [0, 1], yr: [0, 1], xname: 'false-positive rate', yname: 'true-positive rate', lines: [{ pts: [[0, 0], [1, 1]] }] }, mode: 'xy',
    answer: [ra.fpr, ra.recall], tolerance: 0.06,
    whyNot: 'TPR is the share of defaulters flagged; FPR the share of repayers flagged.',
    hints: ['Flagged at ' + rt + ': the scores ' + RS.filter((s) => s >= rt).join(', ') + '.', 'Among them ' + ra.tp + ' defaulted and ' + ra.fp + ' repaid. There are ' + RY.filter((v) => v).length + ' defaulters and ' + RY.filter((v) => !v).length + ' repayers in all.', 'TPR = ' + ra.tp + ' / ' + (ra.tp + ra.fn) + ' = ' + n2(ra.recall) + ', FPR = ' + ra.fp + ' / ' + (ra.fp + ra.tn) + ' = ' + n2(ra.fpr) + ': the point (' + n2(ra.fpr) + ', ' + n2(ra.recall) + ').'],
    solution: { x: ra.fpr, y: ra.recall }, wrong: [{ x: ra.recall, y: ra.fpr }, { x: ra.fp / RS.filter((s) => s >= rt).length, y: ra.precision }] });

  const AP = [0.9, 0.6, 0.4], AN = [0.7, 0.4, 0.3, 0.1], aauc = M.rocAuc([...AP.map(() => 1), ...AN.map(() => 0)], [...AP, ...AN]).value;
  const wins = AP.reduce((s, p) => s + AN.filter((n) => p > n).length, 0), ties = AP.reduce((s, p) => s + AN.filter((n) => p === n).length, 0), pairs = AP.length * AN.length;
  C.exercise({ id: 'ev-x-auc-pairs', chapter: 'ev-curves', title: 'AUC by counting pairs', level: 2, kind: 'number',
    text: ['Three positives scored ' + AP.join(', ') + ' and four negatives scored ' + AN.join(', ') + '. Compute the AUC as the share of positive–negative pairs in which the positive scores higher, counting a tie as half. Three decimal places.'],
    answer: r3(aauc), tolerance: 0.0015,
    whyNot: { [n3(wins / pairs)]: 'That counts the tie as a loss; a tie counts half.', [n3((wins + ties) / pairs)]: 'That counts the tie as a win; a tie counts half.' },
    hints: ['There are ' + AP.length + ' × ' + AN.length + ' = ' + pairs + ' pairs.', 'Positive 0.9 beats all 4; 0.6 beats 3; 0.4 beats 2 and ties with the negative 0.4.', '(' + wins + ' + ' + ties + ' × 0.5) / ' + pairs + ' = ' + n3(aauc) + ', the same as the trapezoid area under the ROC curve.'],
    solution: n3(aauc), wrong: [n3(wins / pairs), n3((wins + ties) / pairs)] });

  const rocRef = (y, s) => { const r = M.roc(y, s, { dropIntermediate: false }); return r.fpr.map((x, i) => [x, r.tpr[i]]); };
  const rocSol = 'function rocPoints(y, scores) {\n  const P = y.filter((v) => v === 1).length, N = y.length - P;\n  const ts = [...new Set(scores)].sort((a, b) => b - a);\n  const pts = [[0, 0]];\n  for (const t of ts) {\n    let tp = 0, fp = 0;\n    for (let i = 0; i < y.length; i++) if (scores[i] >= t) { if (y[i] === 1) tp++; else fp++; }\n    pts.push([fp / N, tp / P]);\n  }\n  return pts;\n}';
  C.exercise({ id: 'ev-x-code-roc', chapter: 'ev-curves', title: 'Write the ROC points', level: 3, kind: 'code',
    text: ['Write `rocPoints(y, scores)` for labels 0 and 1 (both present). Return the ROC curve as a list of [FPR, TPR] points: first [0, 0], then one point for each distinct score taken as the threshold, from the highest to the lowest, where a score at least the threshold is flagged. Equal scores give a single point.'],
    fn: 'rocPoints', params: ['y', 'scores'], start: 'function rocPoints(y, scores) {\n  \n}', tolerance: 1e-9,
    ref: rocRef, cases: [[[1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1]], [[0, 1, 1, 0, 1], [0.2, 0.7, 0.7, 0.5, 0.1]], [[1, 1, 0, 0, 0, 1], [0.6, 0.6, 0.6, 0.3, 0.2, 0.9]], [[0, 1], [0.8, 0.3]]],
    hints: ['Count the positives P and negatives N once. The distinct scores, sorted from high to low, are the thresholds.', 'At each threshold count TP and FP among the flagged, and push [FP / N, TP / P].', rocSol],
    solution: rocSol,
    wrong: ['function rocPoints(y, scores) {\n  const P = y.filter((v) => v === 1).length, N = y.length - P;\n  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);\n  const pts = [[0, 0]];\n  let tp = 0, fp = 0;\n  for (const i of order) { if (y[i] === 1) tp++; else fp++; pts.push([fp / N, tp / P]); }\n  return pts;\n}', 'function rocPoints(y, scores) {\n  const P = y.filter((v) => v === 1).length, N = y.length - P;\n  const ts = [...new Set(scores)].sort((a, b) => b - a);\n  const pts = [[0, 0]];\n  for (const t of ts) {\n    let tp = 0, fp = 0;\n    for (let i = 0; i < y.length; i++) if (scores[i] >= t) { if (y[i] === 1) tp++; else fp++; }\n    pts.push([tp / P, fp / N]);\n  }\n  return pts;\n}'] });

  const AY = [1, 0, 1, 1, 0, 0], AS = [0.95, 0.85, 0.7, 0.6, 0.4, 0.2], apv = M.averagePrecision(AY, AS).value, aucv = M.rocAuc(AY, AS).value;
  const apAll = ML.mean(AY.map((_, k) => AY.slice(0, k + 1).filter((v) => v).length / (k + 1)));
  C.exercise({ id: 'ev-x-ap', chapter: 'ev-curves', title: 'Average precision by hand', level: 3, kind: 'number',
    text: ['Six examples, ranked by score from the highest down: ' + AY.map((v) => (v ? 'positive' : 'negative')).join(', ') + '. Average precision adds, at each threshold, the gain in recall times the precision reached there: AP = Σ (Rₙ − Rₙ₋₁)·Pₙ. What is the AP? Three decimal places.'],
    answer: r3(apv), tolerance: 0.0015,
    whyNot: { [n3(aucv)]: 'That is the ROC AUC of the same ranking. AP uses precision, not the false-positive rate.', [n3(apAll)]: 'That averages the precision at every position, negatives too. Only the steps where recall grows, the positives, count.' },
    hints: ['Recall only grows at a positive: there are 3, so each adds 1/3.', 'The positives are at positions 1, 3 and 4; the precision there is 1/1, 2/3 and 3/4.', 'AP = (1/3)(1) + (1/3)(2/3) + (1/3)(3/4) = ' + n3(apv) + '.'],
    solution: n3(apv), wrong: [n3(aucv), n3(apAll)] });

  const pvopts = ['The precision–recall curve: its floor is the share of positives, and it shows the false alarms per catch', 'The ROC curve: it does not depend on the class balance', 'Either: they always tell the same story', 'Neither: only accuracy matters'];
  C.exercise({ id: 'ev-x-roc-or-pr', chapter: 'ev-curves', title: 'Which curve for rare positives', level: 2, kind: 'choice',
    text: ['Fraudulent payments are 1 in 1000. Two models have nearly the same ROC curve, close to the top-left corner. The bank must decide how many alerts its team can check. Which curve shows the difference that matters?'],
    options: pvopts, answer: 0,
    whyNot: { 1: 'That is exactly why it hides the problem: with 999 honest payments per fraud, an FPR of 1% still means ten false alarms per fraud.', 2: 'With rare positives they can tell very different stories.', 3: 'A model that never alerts is 99.9% accurate.' },
    hints: ['What does a false-positive rate of 0.01 mean when negatives outnumber positives 999 to 1?', 'About ten false alarms for every fraud caught: the ROC curve still looks excellent, but precision is below 0.1.', 'The precision–recall curve: precision counts the alerts that are real, and its floor, 0.001, is the share of fraud.'],
    solution: 0, wrong: [1, 2] });

  const TY = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0], TS = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1], TT = [0.8, 0.6, 0.5, 0.4];
  const tat = TT.map((t) => M.atThreshold(TY, TS, t, 1)), tgt = 0.8;
  const ok = tat.map((a) => a.recall >= tgt - 1e-12), tbest = tat.reduce((b, a, i) => (ok[i] && (b < 0 || a.precision > tat[b].precision) ? i : b), -1);
  const thopts = TT.map((t) => 'threshold ' + t);
  C.exercise({ id: 'ev-x-threshold-recall', chapter: 'ev-curves', title: 'Meet the recall, keep the precision', level: 2, kind: 'choice',
    text: ['Twelve pump minutes, ranked by fault score (1 = a real fault):', { table: [['score', ...TS.map(String)], ['fault', ...TY.map(String)]] }, 'The plant insists on catching at least ' + tgt * 100 + '% of faults (recall ≥ ' + tgt + '). Among these thresholds, which meets that with the best precision? A score at least the threshold raises an alarm.'],
    options: thopts, answer: tbest,
    whyNot: Object.fromEntries(tat.map((a, i) => [i, ok[i] ? 'It meets the recall (' + n2(a.recall) + '), but its precision is only ' + n2(a.precision) + '.' : 'Its precision is ' + n2(a.precision) + ', but it catches only ' + a.tp + ' of 5 faults: recall ' + n2(a.recall) + '.']).filter(([i]) => +i !== tbest)),
    hints: ['For each threshold, count the alarms and the faults among them; there are 5 faults in all.', tat.map((a, i) => TT[i] + ': ' + a.tp + ' of ' + (a.tp + a.fp) + ' alarms are faults, recall ' + n2(a.recall)).join('; ') + '.', 'Recall ≥ ' + tgt + ' needs 4 faults caught. Threshold ' + TT[tbest] + ' is the highest that does, with precision ' + n2(tat[tbest].precision) + '; the lower ones add more false alarms than faults.'],
    solution: tbest, wrong: TT.map((_, i) => i).filter((i) => i !== tbest).slice(0, 2) });

  const thrRef = (y, s, minRecall) => {
    const c = M.thresholdCounts(y, s, 1), P = c.tps[c.tps.length - 1];
    let best = null;
    c.thresholds.forEach((t, k) => { const rec = c.tps[k] / P, prec = c.tps[k] / (c.tps[k] + c.fps[k]); if (rec >= minRecall - 1e-12 && (!best || prec > best.prec + 1e-12)) best = { t, prec }; });
    return best ? best.t : null;
  };
  const thrSol = 'function bestThreshold(y, scores, minRecall) {\n  const P = y.filter((v) => v === 1).length;\n  const ts = [...new Set(scores)].sort((a, b) => b - a);\n  let best = null, bestP = -1;\n  for (const t of ts) {\n    let tp = 0, fp = 0;\n    for (let i = 0; i < y.length; i++) if (scores[i] >= t) { if (y[i] === 1) tp++; else fp++; }\n    const recall = tp / P, precision = tp / (tp + fp);\n    if (recall >= minRecall && precision > bestP) { best = t; bestP = precision; }\n  }\n  return best;\n}';
  C.exercise({ id: 'ev-x-code-threshold', chapter: 'ev-curves', title: 'Write a threshold picker', level: 3, kind: 'code',
    text: ['Write `bestThreshold(y, scores, minRecall)` for labels 0 and 1. Try each distinct score as the threshold (a score at least the threshold is flagged). Among the thresholds whose recall is at least `minRecall`, return the one with the highest precision; if two tie, return the higher threshold. If none qualifies, return `null`.'],
    fn: 'bestThreshold', params: ['y', 'scores', 'minRecall'], start: 'function bestThreshold(y, scores, minRecall) {\n  \n}', decimalArgs: [2],
    ref: thrRef, cases: [[TY, TS, 0.8], [[1, 0, 0, 1, 1, 0, 1], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], 0.5], [TY, TS, 0.5], [TY, TS, 1], [[1, 0, 1, 0, 1], [0.9, 0.8, 0.8, 0.3, 0.2], 0.6], [[0, 1, 0, 1], [0.9, 0.7, 0.6, 0.6], 0.5], [[1, 1, 0], [0.4, 0.4, 0.4], 0.5]],
    hints: ['Loop over the distinct scores from the highest down, counting TP and FP among the flagged at each.', 'Keep a threshold only when its recall reaches the target and its precision beats the best so far; going from high to low, a tie keeps the earlier, higher one.', thrSol],
    solution: thrSol,
    wrong: ['function bestThreshold(y, scores, minRecall) {\n  const P = y.filter((v) => v === 1).length;\n  const ts = [...new Set(scores)].sort((a, b) => a - b);\n  let best = null;\n  for (const t of ts) {\n    let tp = 0;\n    for (let i = 0; i < y.length; i++) if (scores[i] >= t && y[i] === 1) tp++;\n    if (tp / P >= minRecall) best = t;\n  }\n  return best;\n}', 'function bestThreshold(y, scores, minRecall) {\n  const P = y.filter((v) => v === 1).length;\n  const ts = [...new Set(scores)].sort((a, b) => b - a);\n  for (const t of ts) {\n    let tp = 0;\n    for (let i = 0; i < y.length; i++) if (scores[i] >= t && y[i] === 1) tp++;\n    if (tp / P >= minRecall) return t;\n  }\n  return null;\n}'] });

  const LY = [1, 0, 1, 0], LP = [0.9, 0.2, 0.6, 0.7], ll = M.logLoss(LY, LP).value, llBad = ML.mean(LP.map((p) => -Math.log(p)));
  C.exercise({ id: 'ev-x-logloss', chapter: 'ev-curves', title: 'Log loss of four predictions', level: 2, kind: 'number',
    text: ['Four applicants. The model’s probability of default and what happened:', { table: [['applicant', '1', '2', '3', '4'], ['P(default)', ...LP.map(String)], ['defaulted', ...LY.map((v) => (v ? 'yes' : 'no'))]] }, 'What is the log loss, the mean of −ln(probability given to what really happened)? Three decimal places.'],
    answer: r3(ll), tolerance: 0.0015,
    whyNot: { [n3(llBad)]: 'That takes −ln of P(default) for every applicant. For those who repaid, use the probability of repaying, 1 − P(default).', '*': 'Take −ln of the probability each applicant was given for what really happened, then the mean of the four.' },
    hints: ['For a defaulter use p; for a repayer use 1 − p.', 'The probabilities given to what happened: ' + LP.map((p, i) => n2(LY[i] ? p : 1 - p)).join(', ') + '.', 'Penalties: ' + LP.map((p, i) => n3(-Math.log(LY[i] ? p : 1 - p))).join(', ') + '; their mean is ' + n3(ll) + '. Applicant 4, given 0.3 for what happened, pays most.'],
    solution: n3(ll), wrong: [n3(llBad), n3(-Math.log(0.5))] });

  const EP = [0.05, 0.15, 0.3, 0.4, 0.55, 0.7, 0.85, 0.95], EY = [0, 0, 1, 1, 0, 1, 1, 1], ece = M.calibration(EY, EP, { nBins: 2 });
  C.exercise({ id: 'ev-x-ece', chapter: 'ev-curves', title: 'Expected calibration error', level: 3, kind: 'number',
    text: ['Eight predictions of P(yes) and what happened:', { table: [['P(yes)', ...EP.map(String)], ['yes?', ...EY.map(String)]] }, 'Put them in 2 bins of equal width, [0, 0.5] and (0.5, 1]. The expected calibration error is the average, over the bins weighted by how many predictions each holds, of |mean prediction − share that came true|. Three decimal places.'],
    answer: r3(ece.ece), tolerance: 0.0015,
    whyNot: { 0: 'The two bins do not land exactly on the diagonal: compare each bin’s mean prediction with its share of yes.', '*': 'Work bin by bin: the mean prediction, the share of yes, the gap, then weight by 4 of 8.' },
    hints: ['Low bin: ' + EP.slice(0, 4).join(', ') + '; high bin: ' + EP.slice(4).join(', ') + '.', 'Low bin: mean ' + n3(ece.bins[0].meanProb) + ', share of yes ' + n3(ece.bins[0].fraction) + '. High bin: mean ' + n3(ece.bins[1].meanProb) + ', share ' + n3(ece.bins[1].fraction) + '.', 'Gaps ' + n3(Math.abs(ece.bins[0].fraction - ece.bins[0].meanProb)) + ' and ' + n3(Math.abs(ece.bins[1].fraction - ece.bins[1].meanProb)) + ', each weighted 4/8: ECE = ' + n3(ece.ece) + '. The low bin’s probabilities are too ' + (ece.bins[0].fraction > ece.bins[0].meanProb ? 'low' : 'high') + ': a share of ' + n3(ece.bins[0].fraction) + ' of its cases came true at a mean prediction of ' + n3(ece.bins[0].meanProb) + '.'],
    solution: n3(ece.ece), wrong: ['0', n3(Math.abs(ece.bins[0].fraction - ece.bins[0].meanProb))] });

  const lropts = ['Model B: it is certain, and certain mistakes cost without limit', 'Model A: it never commits', 'They tie: the accuracy is the same', 'It cannot be told without the threshold'];
  C.exercise({ id: 'ev-x-logloss-rank', chapter: 'ev-curves', title: 'Same accuracy, different log loss', level: 1, kind: 'choice',
    text: ['Two models are each right on 90 of 100 test rows. Model A says 0.7 for the class it picks; model B says 0.999. Which has the higher (worse) log loss?'],
    options: lropts, answer: 0,
    whyNot: { 1: 'A pays −ln 0.3 = 1.2 for each mistake, a bounded cost; B pays far more.', 2: 'Log loss reads the probabilities, not just the decisions.', 3: 'Log loss uses the probabilities directly; no threshold is involved.' },
    hints: ['Each mistake pays −ln of the probability given to the true class.', 'A gives the true class 0.3 on a mistake, B gives it 0.001.', 'B: −ln 0.001 = 6.9 per mistake against 1.2 for A, which outweighs B’s smaller cost on its 90 right answers. Confident mistakes are the most expensive thing log loss knows.'],
    solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 4: errors, curves and searches
  // =========================================================================================================
  const GY = [12, 15, 9, 20, 14], GP = [11, 17, 10, 16, 14];
  const gcc = { 'MAE|error': r3(M.mae(GY, GP).value), 'MSE|error': r3(M.mse(GY, GP).value), 'RMSE|error': r3(M.rmse(GY, GP).value), 'R²|error': r3(M.r2(GY, GP).value) };
  C.exercise({ id: 'ev-x-reg-metrics', chapter: 'ev-tuning', title: 'Four errors from five residuals', level: 2, kind: 'table',
    text: ['Five bus trips, the minutes late and the model’s prediction:', { table: [['trip', '1', '2', '3', '4', '5'], ['actual', ...GY.map(String)], ['predicted', ...GP.map(String)]] }, 'Work out MAE, MSE, RMSE and R². Three decimal places.'],
    rows: ['MAE', 'MSE', 'RMSE', 'R²'], cols: ['error'], corner: '', cells: gcc, tolerance: 0.0015,
    hints: ['The residuals, actual − predicted, are ' + GY.map((v, i) => v - GP[i]).join(', ') + '.', 'MAE averages their sizes, MSE their squares; RMSE is √MSE. R² = 1 − Σ residual² / Σ (actual − mean)², and the mean is ' + ML.mean(GY) + '.', 'MAE ' + gcc['MAE|error'] + ', MSE ' + gcc['MSE|error'] + ', RMSE ' + gcc['RMSE|error'] + ', R² = 1 − ' + GY.reduce((s, v, i) => s + (v - GP[i]) ** 2, 0) + ' / ' + GY.reduce((s, v) => s + (v - ML.mean(GY)) ** 2, 0) + ' = ' + gcc['R²|error'] + '.'],
    solution: Object.fromEntries(Object.entries(gcc).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(gcc).map(([k, v]) => [k, String(v)])), { 'MAE|error': String(ML.mean(GY.map((v, i) => v - GP[i]))) })] });

  const rpopts = ['The model misses a curve: add a squared term, or use a model that can bend', 'The noise is too large: nothing can be done', 'The model is fine: residuals always look like this', 'Too few test rows'];
  C.exercise({ id: 'ev-x-residual-pattern', chapter: 'ev-tuning', title: 'A residual plot with a shape', level: 2, kind: 'choice',
    text: ['A straight line predicts a target. Plotted against the predictions, the residuals are positive at both ends and negative in the middle: a U shape. What does it say?'],
    options: rpopts, answer: 0,
    whyNot: { 1: 'Noise scatters residuals with no shape. A U is a pattern the model could have learned.', 2: 'A good fit leaves a shapeless band around zero.', 3: 'The U is systematic: more rows would show it more clearly, not remove it.' },
    hints: ['If the model had captured everything learnable, what would remain?', 'Only noise, with no shape. A U means the truth bends where the line does not.', 'The line is too low at both ends and too high in the middle: the truth curves. Add x² (or bin, or use a tree) and the residuals flatten into a band, as on the curved data where the mean residuals per third went from 0.283, −0.358, 0.195 to −0.147, 0.0537, 0.0663.'],
    solution: 0, wrong: [1, 2] });

  const dgopts = ['High bias: add flexibility or better features; more rows will not help', 'High variance: get more rows', 'It is at the noise floor: nothing to gain', 'The validation set is too small'];
  C.exercise({ id: 'ev-x-diagnose', chapter: 'ev-tuning', title: 'Read a learning curve', level: 2, kind: 'choice',
    text: ['A model’s learning curve ends, at 240 training rows, with a training error of 0.264 and a validation error of 0.269. The noise in the data puts a floor at 0.09. What is the diagnosis?'],
    options: dgopts, answer: 0,
    whyNot: { 1: 'The gap between training and validation is only 0.005: there is no variance problem for more rows to fix.', 2: 'Both errors are three times the floor of 0.09.', 3: 'Training and validation agree closely: the estimate is not the problem.' },
    hints: ['Compare the two errors with each other, then with the floor.', 'They are close together (gap 0.005) and both far above 0.09.', 'High bias: the model is too simple for the pattern. A more flexible model or new features will help; more rows will not.'],
    solution: 0, wrong: [1, 2] });

  const vd = S.Data.curve({ n: 200, shape: 'sine', noise: 0.3, seed: 3 }), vperm = ML.rng(7).perm(vd.X.length), vX = vperm.map((i) => vd.X[i]), vy = vperm.map((i) => vd.y[i]);
  const VDEP = [1, 2, 3, 4, 5, 6, 8, 10, 12], vc = ML.validationCurve(ML.DecisionTreeRegressor({}), vX, vy, 'maxDepth', VDEP, { cv: 5, scoring: 'neg_mse' });
  const vErr = VDEP.map((d, i) => [d, r3(-ML.mean(vc.test[i]))]), vBest = vErr.reduce((a, b) => (b[1] < a[1] ? b : a));
  const vTr = VDEP.map((d, i) => [d, r3(-ML.mean(vc.train[i]))]);
  C.exercise({ id: 'ev-x-val-curve', chapter: 'ev-tuning', title: 'Validation error against depth', level: 2, kind: 'predict-curve',
    text: ['A regression tree learns 200 noisy points along a wave, and each depth from 1 to 12 is cross-validated with 5 folds. The validation error at depth 1 is drawn. Sketch the validation MSE as the depth grows.'],
    plot: { xr: [1, 12], yr: [0, 0.4], xname: 'max depth', yname: 'validation MSE', given: vErr.slice(0, 1), note: 'Drag from depth 1 to depth 12.' },
    answer: vErr, tolerance: 0.03, checks: ['valley'],
    hints: ['A stump is too coarse for a wave; a very deep tree gives every noisy point its own leaf.', 'The validation error falls to its lowest, ' + vBest[1] + ', at depth ' + vBest[0] + '.', 'Then it rises again: ' + vErr.map((p) => 'depth ' + p[0] + ': ' + p[1]).join(', ') + '. The training error only falls, to ' + vTr[vTr.length - 1][1] + ' at depth 12.'],
    solution: vErr, wrong: [vTr, vErr.map(([d]) => [d, vErr[0][1]])] });

  C.exercise({ id: 'ev-x-grid-count', chapter: 'ev-tuning', title: 'The cost of a grid', level: 1, kind: 'number',
    text: ['A grid search tries 5 values of C, 4 values of γ and 3 kernels, and cross-validates each combination with 5 folds. How many models does it fit?'],
    answer: 5 * 4 * 3 * 5,
    whyNot: { [String(5 * 4 * 3)]: 'That is the number of combinations; each is fitted once per fold.', [String(5 + 4 + 3)]: 'The values multiply: every C with every γ with every kernel.', '*': 'Multiply the number of values of each setting, then the folds.' },
    hints: ['Every value of each setting is tried with every value of the others.', '5 × 4 × 3 = 60 combinations.', '60 combinations × 5 folds = ' + 5 * 4 * 3 * 5 + ' fits, before the final refit on all the training rows. Random search fixes the number of draws instead.'],
    solution: String(5 * 4 * 3 * 5), wrong: [String(5 * 4 * 3), String(5 + 4 + 3)] });

  const lo = { data: { set: 'loans', n: 500, seed: 9 }, hidden: { seed: 99, n: 800 }, metric: 'auc', target: 0.76 };
  C.exercise(Object.assign({ id: 'ev-x-wf-loans', chapter: 'ev-tuning', title: 'Rank the applicants with k-NN', level: 3, kind: 'workflow',
    text: ['The lender wants applicants ranked by risk, so its staff can review them from the top down. Ranking is judged by ROC AUC on 800 hidden applications. Only k-nearest neighbours is allowed, with k up to 100. Reach an AUC of at least 0.76.'],
    limits: { blocks: ['split', 'encode', 'scale', 'knn', 'train', 'evaluate'], models: ['knn'], params: { knn: { k: [1, 100] } } },
    hints: ['Distances on raw incomes and ratios are decided by the income alone: scale first.', 'k-NN’s probability is the share of the k neighbours that defaulted, so with k = 5 it takes only 6 values and many applicants tie. A larger k ranks more finely. Try a few values, as a search would.', 'Split → Scale (standard) → k-NN with k = 75 → Train → Evaluate: AUC 0.77 on the hidden applications, against 0.638 unscaled and 0.696 with k = 5.'],
    solution: pipe(lo.data, 'knn', { k: 75 }, [['scale', { how: 'standard' }]]),
    wrong: [pipe(lo.data, 'knn', { k: 75 }), pipe(lo.data, 'knn', { k: 5 }, [['encode', {}], ['scale', { how: 'standard' }]])] }, lo));
})(globalThis.Nooral = globalThis.Nooral || {});
