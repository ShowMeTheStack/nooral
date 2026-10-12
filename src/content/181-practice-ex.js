// Part 18, practice and interpretability: the exercises. Every number in a text, an answer or a hint is worked out
// here by the engines (S.Metrics for scores, S.Explain for partial dependence, Shapley values and the least-squares
// solve, S.WF and S.Challenges for the workflow task) from the small cases written out below, or quoted from a topic.
(function (S) {
  'use strict';
  const C = S.Content, M = S.Metrics, E = S.Explain, WF = S.WF;
  if (!C || !M || !E || !WF || C.exercises.has('pr-x-acc-paradox')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const n2 = (v) => String(r2(v)), n3 = (v) => String(r3(v));
  const mn = (v) => String(v).replace(/^-/, '−');
  const strCells = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));
  const fromCounts = (tp, fn, fp, tn) => { const y = [], p = []; const add = (k, a, b) => { for (let i = 0; i < k; i++) { y.push(a); p.push(b); } }; add(tp, 1, 1); add(fn, 1, 0); add(fp, 0, 1); add(tn, 0, 0); return { y, p }; };
  const cmTable = (labels, c) => ({ table: [['true \\ predicted', ...labels], [labels[0], String(c[0]), String(c[1])], [labels[1], String(c[2]), String(c[3])]] });
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // =========================================================================================================
  // Chapter 1: when one class is rare
  // =========================================================================================================
  const B0 = fromCounts(0, 12, 0, 388), acc0 = M.accuracy(B0.y, B0.p).value;
  C.exercise({ id: 'pr-x-acc-paradox', chapter: 'pr-imbalance', title: 'The accuracy of doing nothing', level: 1, kind: 'number',
    text: ['A test set of 400 card payments holds 12 frauds. A model that never flags anything: what is its accuracy? Three decimal places.'],
    answer: r3(acc0), tolerance: 0.0015,
    whyNot: { 0: 'It is wrong only on the 12 frauds; every other payment is called correctly.', [n3(12 / 400)]: 'That is the share of frauds, the rows it gets wrong.', 0.5: 'Accuracy counts rows, not classes.' },
    hints: ['It calls every payment honest.', 'It is right on the 388 honest payments and wrong on the 12 frauds.', '388 / 400 = ' + n3(acc0) + ', and it catches no fraud at all: recall 0.'],
    solution: n3(acc0), wrong: ['0', n3(12 / 400)] });

  const mopts = ['F1 of the rare class, or average precision', 'Accuracy', 'Specificity (recall of the common class)', 'The share of rows predicted common'];
  C.exercise({ id: 'pr-x-metric-imb', chapter: 'pr-imbalance', title: 'The right metric for rare faults', level: 1, kind: 'choice',
    text: ['One reading in fifty comes from a failing bearing. Engineers want an alarm that catches failures without crying wolf so often that it is ignored. Which number should candidate models be compared by?'],
    options: mopts, answer: 0,
    whyNot: { 1: 'An alarm that never sounds is 98% accurate.', 2: 'Specificity is perfect for an alarm that never sounds.', 3: 'That rewards doing nothing.' },
    hints: ['Which numbers does the silent alarm score well on?', 'The score must look at the rare class: failures caught (recall) and alarms that are real (precision).', 'F1 of the failure class combines both at one threshold; average precision does it over every threshold.'],
    solution: 0, wrong: [1, 2] });

  const IC = [18, 7, 30, 445], IS = fromCounts(...IC), prf = M.precisionRecallF1(IS.y, IS.p), ba = M.balancedAccuracy(IS.y, IS.p).value, ac = M.accuracy(IS.y, IS.p).value;
  const isCells = { 'accuracy|score': r3(ac), 'precision|score': r3(prf.precision), 'recall|score': r3(prf.recall), 'F1|score': r3(prf.f1), 'balanced accuracy|score': r3(ba) };
  C.exercise({ id: 'pr-x-imb-scores', chapter: 'pr-imbalance', title: 'Five scores for a rare class', level: 2, kind: 'table',
    text: ['A fault detector on 500 test readings, with fault as the positive class:', cmTable(['fault', 'normal'], IC), 'Work out the five scores, to three decimal places.'],
    rows: ['accuracy', 'precision', 'recall', 'F1', 'balanced accuracy'], cols: ['score'], corner: '', cells: isCells, tolerance: 0.0015,
    hints: ['TP = 18, FN = 7, FP = 30, TN = 445.', 'Precision = 18 / 48, recall = 18 / 25; balanced accuracy averages the recall of faults and of normal readings (445 / 475).', 'Accuracy ' + n3(ac) + ', precision ' + n3(prf.precision) + ', recall ' + n3(prf.recall) + ', F1 ' + n3(prf.f1) + ', balanced accuracy ' + n3(ba) + '. Accuracy looks fine; precision says most alarms are false.'],
    solution: strCells(Object.fromEntries(Object.entries(isCells).map(([k, v]) => [k, n3(v)]))),
    wrong: [strCells(Object.assign(Object.fromEntries(Object.entries(isCells).map(([k, v]) => [k, n3(v)])), { 'precision|score': n3(prf.recall), 'recall|score': n3(prf.precision) }))] });

  const nW = 500, nR = 40, wR = nW / (2 * nR), wC = nW / (2 * (nW - nR));
  const cwCells = { 'rare|weight': r3(wR), 'common|weight': r3(wC) };
  C.exercise({ id: 'pr-x-class-weights', chapter: 'pr-imbalance', title: 'Balanced class weights', level: 2, kind: 'table',
    text: ['A training set of 500 rows has 40 rare rows and 460 common ones. Balanced class weights give each row of class c the weight n / (k · n_c). What are the two weights? Three decimal places.'],
    rows: ['rare', 'common'], cols: ['weight'], corner: 'class', cells: cwCells, tolerance: 0.0015,
    hints: ['n = 500 rows, k = 2 classes.', 'Rare: 500 / (2 × 40). Common: 500 / (2 × 460).', 'Rare ' + n3(wR) + ', common ' + n3(wC) + '. Check: 40 × ' + n3(wR) + ' = 250 and 460 × ' + n3(wC) + ' = 250: each class now weighs half.'],
    solution: strCells({ 'rare|weight': n3(wR), 'common|weight': n3(wC) }),
    wrong: [strCells({ 'rare|weight': n3(nW / nR), 'common|weight': n3(nW / (nW - nR)) }), strCells({ 'rare|weight': n3(nR / nW), 'common|weight': n3((nW - nR) / nW) })] });

  const cwSol = 'function classWeights(y) {\n  const count = {};\n  for (const c of y) count[c] = (count[c] || 0) + 1;\n  const k = Object.keys(count).length;\n  return y.map((c) => y.length / (k * count[c]));\n}';
  const cwRef = (y) => { const cnt = new Map(); y.forEach((c) => cnt.set(c, (cnt.get(c) || 0) + 1)); return y.map((c) => y.length / (cnt.size * cnt.get(c))); };
  C.exercise({ id: 'pr-x-code-weights', chapter: 'pr-imbalance', title: 'Write balanced class weights', level: 1, kind: 'code',
    text: ['Write `classWeights(y)`: y lists each row’s class (numbers or strings, any number of classes). Return one weight per row, n / (k · n_c), where n is the number of rows, k the number of classes and n_c the number of rows of that row’s class.'],
    fn: 'classWeights', params: ['y'], start: 'function classWeights(y) {\n  \n}', tolerance: 1e-9,
    ref: cwRef,
    cases: [[[0, 0, 0, 1]], [['a', 'b', 'c', 'a', 'a', 'b']], [[1, 1, 1, 1]], [[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]]],
    hints: ['Count each class first.', 'k is the number of different classes; then map each row to n / (k · count of its class).', cwSol],
    solution: cwSol,
    wrong: ['function classWeights(y) {\n  const count = {};\n  for (const c of y) count[c] = (count[c] || 0) + 1;\n  return y.map((c) => y.length / count[c]);\n}', 'function classWeights(y) {\n  const count = {};\n  for (const c of y) count[c] = (count[c] || 0) + 1;\n  return y.map((c) => y.length / (2 * count[c]));\n}'] });

  const oopts = ['Split, then oversample the rare rows of the training part only', 'Oversample the whole table, then split', 'Oversample the test part too, so it is balanced', 'Split, then oversample both parts the same way'];
  C.exercise({ id: 'pr-x-resample-order', chapter: 'pr-imbalance', title: 'When to oversample', level: 2, kind: 'choice',
    text: ['A team wants to oversample the rare class by copying its rows. In which order should the steps go?'],
    options: oopts, answer: 0,
    whyNot: { 1: 'Copies of one rare row then land on both sides of the split: the model is tested on rows it memorised, and the score is too kind.', 2: 'The test set must look like real data, rare class and all, to say how the model will do.', 3: 'Resampling the test part changes what the score measures.' },
    hints: ['Which part must look like the data the model will meet?', 'Copies of the same row must never end up in both parts.', 'Split first, keep the test part as it is, and oversample the training part only.'],
    solution: 0, wrong: [1, 2] });

  const kopts = ['It moved the threshold more than it changed the ranking of the rows', 'Class weights never help', 'Average precision ignores the rare class', 'The weighted model is broken'];
  C.exercise({ id: 'pr-x-weights-ranking', chapter: 'pr-imbalance', title: 'Recall up, average precision flat', level: 2, kind: 'choice',
    text: [{ topic: 'imbalanced-data', frame: 4, caption: 'The toy set: the baseline, the plain logistic model and the weighted one.' }, 'Class weights took recall from 0.20 to 1.00, but average precision barely moved (0.63 to 0.62). Why?'],
    options: kopts, answer: 0,
    whyNot: { 1: 'They lifted recall and balanced accuracy a great deal.', 2: 'Average precision is all about the rare class: it scores how well the model ranks rare rows above common ones.', 3: 'It finds every rare test row.' },
    hints: ['Average precision looks at every threshold; recall at 50% looks at one.', 'Weights mostly shift the scores of all rows the same way.', 'The weights moved where the 50% line falls, not which rows score higher than which: same ranking, different threshold.'],
    solution: 0, wrong: [1, 2] });

  const wfb = { data: { set: 'loans', n: 500, seed: 4 }, hidden: { seed: 77, n: 1000 }, metric: 'balanced', target: 0.64 };
  C.exercise(Object.assign({ id: 'pr-x-wf-balanced', chapter: 'pr-imbalance', title: 'Balanced accuracy on the loans', level: 3, kind: 'workflow',
    text: ['About a quarter of the loan applicants default. Build a workflow whose model reaches a balanced accuracy of at least 0.64 on 1,000 hidden applicants: the mean of the recall of defaults and the recall of repayers, so a model that calls everyone “repaid” scores 0.5.'],
    task: 'classification',
    limits: { blocks: ['split', 'encode', 'scale', 'impute', 'logistic', 'knn', 'naivebayes', 'tree', 'lda', 'train', 'evaluate'], models: ['logistic', 'knn', 'naivebayes', 'tree', 'lda'], maxBlocks: 8 },
    hints: ['Try the usual logistic regression first and read its balanced accuracy in Evaluate.', 'Models fitted for overall accuracy lean towards “repaid”. Gaussian naive Bayes weighs each class’s own spread, and its probabilities sit less close to the majority.', 'Split → Encode → Scale → Gaussian naive Bayes → Train → Evaluate: balanced accuracy 0.652 on the hidden applicants, against 0.622 for logistic regression and 0.584 for 5-NN.'],
    solution: pipe(wfb.data, 'naivebayes', {}, [['encode', {}], ['scale', { how: 'standard' }]]),
    wrong: [pipe(wfb.data, 'logistic', {}, [['encode', {}], ['scale', { how: 'standard' }]]), pipe(wfb.data, 'knn', { k: 5 }, [['encode', {}], ['scale', { how: 'standard' }]])] }, wfb));

  // =========================================================================================================
  // Chapter 2: choosing a model
  // =========================================================================================================
  const FS = [0.86, 0.97, 0.94, 0.94, 0.86], fm = FS.reduce((a, v) => a + v, 0) / FS.length, fsd = Math.sqrt(FS.reduce((a, v) => a + (v - fm) ** 2, 0) / FS.length);
  const cvCells = { 'mean|value': r3(fm), 'spread (standard deviation)|value': r3(fsd), 'standard error|value': r3(fsd / Math.sqrt(5)) };
  C.exercise({ id: 'pr-x-cv-stats', chapter: 'pr-select', title: 'Mean, spread and standard error', level: 1, kind: 'table',
    text: ['5-fold cross-validation of one model gives the accuracies ' + FS.join(', ') + '. Work out the mean, the spread (the standard deviation of the five, dividing by 5) and the standard error, spread / √5. Three decimal places.'],
    rows: ['mean', 'spread (standard deviation)', 'standard error'], cols: ['value'], corner: '', cells: cvCells, tolerance: 0.0015,
    hints: ['Mean = the sum / 5 = ' + n3(FS.reduce((a, v) => a + v, 0)) + ' / 5.', 'Deviations from the mean: ' + FS.map((v) => mn(n3(v - fm))).join(', ') + '; square them, average, take the root.', 'Mean ' + n3(fm) + ', spread ' + n3(fsd) + ', standard error ' + n3(fsd / Math.sqrt(5)) + '.'],
    solution: strCells(Object.fromEntries(Object.entries(cvCells).map(([k, v]) => [k, n3(v)]))),
    wrong: [strCells({ 'mean|value': n3(fm), 'spread (standard deviation)|value': n3(fsd), 'standard error|value': n3(fsd) })] });

  const CAND = [['logistic regression', [0.84, 0.88, 0.86, 0.83, 0.87], 1], ['k-NN', [0.88, 0.92, 0.91, 0.87, 0.91], 2], ['a random forest', [0.87, 0.95, 0.92, 0.88, 0.93], 4], ['an RBF SVM', [0.89, 0.92, 0.9, 0.88, 0.92], 3]];
  const cs = CAND.map(([nm, f, cx]) => { const m = f.reduce((a, v) => a + v, 0) / f.length, sd = Math.sqrt(f.reduce((a, v) => a + (v - m) ** 2, 0) / f.length); return { nm, f, cx, m, se: sd / Math.sqrt(f.length) }; });
  const best = cs.reduce((a, c) => (c.m > a.m ? c : a)), line = best.m - best.se;
  const within = cs.filter((c) => c.m >= line - 1e-12), pick = within.reduce((a, c) => (c.cx < a.cx ? c : a));
  const cvTab = { table: [['model', 'fold scores', 'mean'], ...cs.map((c) => [c.nm, c.f.join(', '), n3(c.m)])] };
  C.exercise({ id: 'pr-x-one-se', chapter: 'pr-select', title: 'The one-standard-error line', level: 3, kind: 'number',
    text: ['Four candidates, 5-fold cross-validated on the same folds:', cvTab, 'The one-standard-error rule draws a line at the best mean minus that model’s standard error (its spread, dividing by 5, over √5). Where is the line? Three decimal places.'],
    answer: r3(line), tolerance: 0.0015,
    whyNot: { [n3(best.m)]: 'That is the best mean itself; subtract its standard error.', [n3(best.m - best.se * Math.sqrt(5))]: 'That subtracts the spread; the standard error divides it by √5.', '*': 'Find the best mean, its spread, divide by √5 and subtract.' },
    hints: ['The best mean is ' + best.nm + '’s, ' + n3(best.m) + '.', 'Its spread is ' + n3(best.se * Math.sqrt(5)) + ', so its standard error is ' + n3(best.se) + '.', n3(best.m) + ' − ' + n3(best.se) + ' = ' + n3(line) + '. Any model with a mean above it is within the noise of the best.'],
    solution: n3(line), wrong: [n3(best.m), n3(best.m - best.se * Math.sqrt(5))] });

  const popts = cs.map((c) => c.nm.replace(/^./, (x) => x.toUpperCase()));
  C.exercise({ id: 'pr-x-one-se-pick', chapter: 'pr-select', title: 'Pick by the one-standard-error rule', level: 2, kind: 'choice',
    text: ['The same four candidates:', cvTab, 'The line is at ' + n3(line) + '. Count them from the simplest: logistic regression, k-NN, the RBF SVM, the random forest. Which model does the rule choose?'],
    options: popts, answer: cs.indexOf(pick),
    whyNot: Object.fromEntries(cs.map((c, i) => [i, c.m < line ? 'Its mean, ' + n3(c.m) + ', is below the line.' : c === best ? 'It has the best mean, but a simpler model is within one standard error of it.' : 'It clears the line, but a simpler model does too.']).filter(([i]) => +i !== cs.indexOf(pick))),
    hints: ['Which means clear ' + n3(line) + '?', within.map((c) => c.nm + ' (' + n3(c.m) + ')').join(', ') + ' clear it.', 'The simplest of those is ' + pick.nm + '.'],
    solution: cs.indexOf(pick), wrong: [cs.indexOf(best), 0] });

  const topts = ['Keep 0.900 as the reported score, and stop choosing', 'Try the forest on the test set too, and report the better one', 'Tune k on the test set until it beats 0.928', 'Report the cross-validation mean, 0.928, instead'];
  C.exercise({ id: 'pr-x-test-reuse', chapter: 'pr-select', title: 'The test score came in lower', level: 1, kind: 'choice',
    text: ['k-NN was chosen by cross-validation at 0.928 and then scored 0.900 on the locked test set. A colleague suggests the forest, which scores 0.933 on the same test rows. What should be reported?'],
    options: topts, answer: 0,
    whyNot: { 1: 'Choosing between models on the test set makes its score optimistic: it is then a validation set.', 2: 'Tuning on the test set uses it up; its score no longer predicts new data.', 3: 'The test score is the honest estimate; the cross-validation mean was used to choose.' },
    hints: ['What was the test set for?', 'Any choice made by looking at it makes its score biased upwards.', 'Report 0.900. The choice was made before the test; changing it now would turn the test set into one more validation set.'],
    solution: 0, wrong: [1, 3] });

  const nopts = ['A model that wins on one kind of data can lose badly on another', 'The random forest is always best', 'With enough data every model scores the same', 'The simplest model always wins'];
  C.exercise({ id: 'pr-x-nfl', chapter: 'pr-select', title: 'No free lunch, in a sentence', level: 2, kind: 'choice',
    text: [{ topic: 'no-free-lunch', frame: 7, caption: 'Six models on six data sets.' }, 'Which statement does this table support?'],
    options: nopts, answer: 0,
    whyNot: { 1: '4 different models win at least once; the forest is not first everywhere.', 2: 'Logistic regression stays near guessing on XOR however much data it gets: its shape is wrong.', 3: 'The straight line wins only on the noisy line.' },
    hints: ['How many different models win somewhere?', 'Logistic regression: 0.81 on the noisy line, 0.47 on XOR.', 'Each model fits some shapes and not others, so the winner depends on the data.'],
    solution: 0, wrong: [1, 3] });

  const RK = [['moons', [0.85, 0.91, 0.9]], ['ring', [0.45, 0.97, 0.95]], ['line', [0.81, 0.77, 0.79]], ['loans', [0.79, 0.75, 0.77]]], RN = ['logistic', 'k-NN', 'tree'];
  const ranks = RK.map(([, s]) => s.map((v) => 1 + s.filter((w) => w > v).length)), avgK = ranks.reduce((a, r) => a + r[1], 0) / RK.length;
  C.exercise({ id: 'pr-x-rank', chapter: 'pr-select', title: 'An average rank', level: 2, kind: 'number',
    text: ['Three models scored on four data sets:', { table: [['set', ...RN], ...RK.map(([n, s]) => [n, ...s.map(String)])] }, 'On each set the best model gets rank 1, the next 2, the worst 3. What is k-NN’s average rank? Two decimal places.'],
    answer: r2(avgK), tolerance: 0.005,
    whyNot: { [n2(RK.reduce((a, [, s]) => a + s[1], 0) / RK.length)]: 'That is its average score; rank it on each set first.', '*': 'Rank k-NN on each set (1 is best), then average the four ranks.' },
    hints: ['Rank within each row.', 'k-NN’s ranks: ' + ranks.map((r) => r[1]).join(', ') + '.', '(' + ranks.map((r) => r[1]).join(' + ') + ') / 4 = ' + n2(avgK) + '. Good on average, yet last on two of the four sets.'],
    solution: n2(avgK), wrong: [n2(RK.reduce((a, [, s]) => a + s[1], 0) / RK.length)] });

  // =========================================================================================================
  // Chapter 3: explaining a model
  // =========================================================================================================
  const PD = [0.71, 0.69, 0.74, 0.7, 0.72], base0 = 0.82, imp = PD.reduce((a, v) => a + (base0 - v), 0) / PD.length;
  C.exercise({ id: 'pr-x-perm-drop', chapter: 'pr-explain', title: 'One feature’s importance', level: 1, kind: 'number',
    text: ['A model scores accuracy ' + base0 + ' on its test rows. Shuffling the column “hours of use” five times gives accuracies ' + PD.join(', ') + '. What is the permutation importance of the column? Three decimal places.'],
    answer: r3(imp), tolerance: 0.0015,
    whyNot: { [n3(PD.reduce((a, v) => a + v, 0) / 5)]: 'That is the mean shuffled score; the importance is how far it fell from ' + base0 + '.', [n3(base0 - Math.min(...PD))]: 'That is the largest single drop; average all five.' },
    hints: ['Importance = intact score − shuffled score, averaged over the shuffles.', 'Drops: ' + PD.map((v) => n3(base0 - v)).join(', ') + '.', 'Their mean is ' + n3(imp) + '.'],
    solution: n3(imp), wrong: [n3(PD.reduce((a, v) => a + v, 0) / 5), n3(base0 - Math.min(...PD))] });

  const ropts = ['Height and arm span carry the same information, so shuffling either leaves the other to fall back on', 'Height does not affect the outcome', 'The model is broken', 'Permutation importance is random'];
  C.exercise({ id: 'pr-x-perm-read', chapter: 'pr-explain', title: 'Two features that share a story', level: 2, kind: 'choice',
    text: ['A model predicts shirt size from height, arm span (almost the same as height) and age. Permutation importance gives height 0.02, arm span 0.03 and age 0.01. Dropping both height and arm span, though, ruins the model. How can both look unimportant?'],
    options: ropts, answer: 0,
    whyNot: { 1: 'Without both, the model is ruined: height matters a great deal.', 2: 'It is working as trained; the measure is what misleads.', 3: 'The shuffles vary, but not by that much: the effect is systematic.' },
    hints: ['What does the model still have when height alone is shuffled?', 'In the topic, a noisy copy of income took income’s importance from 0.069 to 0.005.', 'Correlated features share their importance: each looks small because the other covers for it. Shuffle them together, or group them, to see their joint importance.'],
    solution: 0, wrong: [1, 3] });

  const permModel = (rows) => rows.map((r) => (r[0] + 0.5 * r[1] > 1 ? 1 : 0));
  const permRef = (X, y, perms) => {
    const acc = (rows) => { const p = permModel(rows); return p.filter((v, i) => v === y[i]).length / y.length; };
    const a0 = acc(X);
    return X[0].map((_, j) => perms.reduce((s, pm) => s + (a0 - acc(X.map((r, i) => { const z = r.slice(); z[j] = X[pm[i]][j]; return z; }))), 0) / perms.length);
  };
  const PX = [[0.2, 1.8], [1.2, 0.1], [0.9, 0.4], [0.3, 0.2], [1.5, 1.0], [0.1, 0.5]], PY = [1, 1, 1, 0, 1, 0];
  const permSol = 'function permImportance(X, y, perms) {\n  const acc = (rows) => { const p = model(rows); let c = 0; for (let i = 0; i < y.length; i++) if (p[i] === y[i]) c++; return c / y.length; };\n  const base = acc(X);\n  return X[0].map((_, j) => {\n    let total = 0;\n    for (const pm of perms) {\n      const rows = X.map((r, i) => { const z = r.slice(); z[j] = X[pm[i]][j]; return z; });\n      total += base - acc(rows);\n    }\n    return total / perms.length;\n  });\n}';
  C.exercise({ id: 'pr-x-code-perm', chapter: 'pr-explain', title: 'Write permutation importance', level: 2, kind: 'code',
    text: ['A model is given (shown above the editor): `model(rows)` returns a 0 or 1 for each row. Write `permImportance(X, y, perms)`: for each column j, and for each permutation pm in perms (a list of row indices), replace column j of every row i by X[pm[i]][j], score the accuracy against y, and record the intact accuracy minus that score. Return each column’s mean drop.'],
    fn: 'permImportance', params: ['X', 'y', 'perms'], start: 'function permImportance(X, y, perms) {\n  \n}', tolerance: 1e-9,
    prelude: 'function model(rows) {\n  return rows.map((r) => (r[0] + 0.5 * r[1] > 1 ? 1 : 0));\n}',
    ref: permRef,
    cases: [[PX, PY, [[1, 2, 3, 4, 5, 0]]], [PX, PY, [[5, 4, 3, 2, 1, 0], [2, 0, 1, 5, 3, 4]]], [PX, [1, 1, 1, 1, 1, 1], [[3, 0, 5, 1, 4, 2]]], [[[0.5, 1.2], [0.7, 0.2]], [1, 0], [[0, 1], [1, 0]]]],
    hints: ['Score the intact data once: accuracy = the share of model(X) equal to y.', 'For column j and permutation pm, copy each row and set z[j] = X[pm[i]][j]; never change X itself. Average the drops over the permutations.', permSol],
    solution: permSol,
    wrong: ['function permImportance(X, y, perms) {\n  const acc = (rows) => { const p = model(rows); let c = 0; for (let i = 0; i < y.length; i++) if (p[i] === y[i]) c++; return c / y.length; };\n  const base = acc(X);\n  return X[0].map((_, j) => {\n    let total = 0;\n    for (const pm of perms) { for (let i = 0; i < X.length; i++) X[i][j] = X[pm[i]][j]; total += base - acc(X); }\n    return total / perms.length;\n  });\n}',
      'function permImportance(X, y, perms) {\n  const acc = (rows) => { const p = model(rows); let c = 0; for (let i = 0; i < y.length; i++) if (p[i] === y[i]) c++; return c / y.length; };\n  return X[0].map((_, j) => {\n    let total = 0;\n    for (const pm of perms) total += acc(X.map((r, i) => { const z = r.slice(); z[j] = X[pm[i]][j]; return z; }));\n    return total / perms.length;\n  });\n}'] });

  const pdModel = (rows) => rows.map((r) => 2 * r[0] + r[0] * r[1] + r[1]), PDX = [[1, 0], [0, 2], [2, 1], [1, -1]];
  const pdp = E.partialDependence(pdModel, PDX, 0, { grid: [0, 1, 2] });
  C.exercise({ id: 'pr-x-pdp-avg', chapter: 'pr-explain', title: 'Partial dependence from ICE values', level: 2, kind: 'table',
    text: ['Four rows’ ICE curves for feature x₁: each row’s prediction with x₁ set to 0, 1 and 2 and its other features kept:', { table: [['row', 'x₁ = 0', 'x₁ = 1', 'x₁ = 2'], ...pdp.ice.map((r, i) => [String(i + 1), ...r.map(mn)])] }, 'What is the partial dependence at each value? Three decimal places where needed.'],
    rows: ['partial dependence'], cols: ['x₁ = 0', 'x₁ = 1', 'x₁ = 2'], corner: '', cells: Object.fromEntries(pdp.grid.map((g, k) => ['partial dependence|x₁ = ' + g, r3(pdp.average[k])])), tolerance: 0.0015,
    hints: ['Partial dependence is the average of the ICE curves at each grid value.', 'At x₁ = 0: (' + pdp.ice.map((r) => mn(r[0])).join(' + ') + ') / 4.', pdp.average.map((v, k) => 'x₁ = ' + pdp.grid[k] + ': ' + mn(n3(v))).join('; ') + '. The curves are not parallel (row 4 is flat): an interaction with the other feature.'],
    solution: strCells(Object.fromEntries(pdp.grid.map((g, k) => ['partial dependence|x₁ = ' + g, n3(pdp.average[k])]))),
    wrong: [strCells(Object.fromEntries(pdp.grid.map((g, k) => ['partial dependence|x₁ = ' + g, n3(pdp.ice[0][k])])))] });

  C.exercise({ id: 'pr-x-pdp-read', chapter: 'pr-explain', title: 'Read a partial dependence plot', level: 2, kind: 'number',
    text: [{ topic: 'partial-dependence', frame: 3, input: { case: 'distance', rows: '30' }, caption: 'Rent against distance to the centre: 30 ICE curves and their average.' }, 'By how much does the model’s rent fall, on average, between the nearest grid value (0.491 km) and the farthest (11 km)? Read the partial dependence: highest 989, lowest 645.'],
    answer: 989 - 645, tolerance: 0.5,
    whyNot: { 1030: 'That is the widest gap between ICE curves at one distance, a spread between flats, not the average effect of distance.', '*': 'Subtract the lowest value of the thick curve from its highest.' },
    hints: ['The thick curve is the partial dependence: the average of the thin ICE curves.', 'Its highest point is 989, at 0.491 km; its lowest 645, at 11 km.', '989 − 645 = 344 a month, on average over the flats; the ICE curves are shifted copies, so the average tells the story here.'],
    solution: '344', wrong: ['1030'] });

  const iopts = ['x₁ interacts with another feature: it raises y for some rows and lowers it for others', 'x₁ does not matter', 'The model is overfitting', 'The grid is too coarse'];
  C.exercise({ id: 'pr-x-ice-read', chapter: 'pr-explain', title: 'Flat average, busy curves', level: 2, kind: 'choice',
    text: [{ topic: 'partial-dependence', frame: 3, input: { case: 'twist', rows: '30' }, caption: 'ICE curves and their average for x₁.' }, 'The partial dependence on x₁ rises only 0.393, but the ICE curves are up to 2.63 apart and slope both ways. What should be concluded?'],
    options: iopts, answer: 0,
    whyNot: { 1: 'Each row’s curve moves a lot: x₁ matters, in opposite directions for different rows.', 2: 'Overfitting would make the curves jagged, not slope in two directions.', 3: 'A finer grid would show the same two families of curves.' },
    hints: ['Look at the thin curves, not only the thick one.', 'Rising curves and falling curves cancel in the average.', 'An interaction: x₁ raises y when x₂ is above zero and lowers it otherwise. A flat partial dependence must not be read as “no effect” until the ICE curves are checked.'],
    solution: 0, wrong: [1, 2] });

  const SB = 0.31, SPHI = [0.12, -0.05, 0.08, -0.02], SFX = SB + SPHI.reduce((a, v) => a + v, 0);
  C.exercise({ id: 'pr-x-shap-sum', chapter: 'pr-explain', title: 'Where the waterfall ends', level: 1, kind: 'number',
    text: ['A model’s average output over the background rows is ' + SB + '. For one applicant the contributions are ' + SPHI.map(mn).join(', ') + '. What does the model output for this applicant? Two decimal places.'],
    answer: r2(SFX), tolerance: 0.005,
    whyNot: { [n2(SPHI.reduce((a, v) => a + v, 0))]: 'That is the sum of the contributions alone; start from the base value.', [n2(SB + SPHI.reduce((a, v) => a + Math.abs(v), 0))]: 'Contributions that pull down must be subtracted.' },
    hints: ['Contributions always add up: base + Σ φ = the prediction.', 'Σ φ = ' + mn(n2(SPHI.reduce((a, v) => a + v, 0))) + '.', SB + ' + ' + n2(SPHI.reduce((a, v) => a + v, 0)) + ' = ' + n2(SFX) + '.'],
    solution: n2(SFX), wrong: [n2(SPHI.reduce((a, v) => a + v, 0)), n2(SB + SPHI.reduce((a, v) => a + Math.abs(v), 0))] });

  const MB = 860, MF = 966, MK = [224, -126, 15, 0], miss = MF - MB - MK.reduce((a, v) => a + v, 0);
  C.exercise({ id: 'pr-x-shap-missing', chapter: 'pr-explain', title: 'The missing contribution', level: 2, kind: 'number',
    text: ['For one flat the rent model’s base value is ' + MB + ' and its answer ' + MF + '. Area contributes +224, distance −126, rooms +15 and floor 0. What is the contribution of the fifth feature, balcony?'],
    // the topic's waterfall, from unrounded contributions, shows the balcony as one less: accept both
    answer: miss - 0.5, tolerance: 0.5,
    whyNot: { [MF - MB]: 'That is the whole gap; the other four take most of it.', [-miss]: 'Check the sign: the other four add up to more than the gap.' },
    hints: ['base + all contributions = the answer.', 'The gap is ' + MF + ' − ' + MB + ' = ' + (MF - MB) + '; the four known ones add up to ' + MK.reduce((a, v) => a + v, 0) + '.', (MF - MB) + ' − ' + MK.reduce((a, v) => a + v, 0) + ' = ' + mn(miss) + ': this flat has no balcony, which pulls its rent below the average, as the topic’s waterfall shows (balcony ' + mn(miss - 1) + ' there, worked from the unrounded contributions; either answer is accepted).'],
    solution: String(miss), wrong: [String(MF - MB), String(-miss)] });

  const sopts = ['Distance', 'Area', 'Rooms', 'Floor'];
  C.exercise({ id: 'pr-x-shap-read', chapter: 'pr-explain', title: 'Read a waterfall', level: 2, kind: 'choice',
    text: [{ topic: 'shap-waterfall', frame: 7, input: { case: 'rents', row: 0, bg: '30' }, caption: 'The waterfall for one flat’s rent.' }, 'Which feature pulls this flat’s rent down the most, below the average flat?'],
    options: sopts, answer: 0,
    whyNot: { 1: 'Area pushes the rent up, by 224: it is the largest bar, but upward.', 2: 'Rooms push it up, by 15.', 3: 'Floor contributes 0.' },
    hints: ['Bars that pull down point the other way from those that push up.', 'Two bars pull down: distance and balcony.', 'Distance (9.3 km from the centre) pulls it down by 126; the missing balcony by only 8.'],
    solution: 0, wrong: [1, 2] });

  // two features: v(∅), v({1}), v({2}), v({1, 2}) as a lookup model (bit j of the row = feature j present)
  const V2 = [0.2, 0.5, 0.3, 0.8], look = (vals) => (rows) => rows.map((r) => vals[r.reduce((m, b, j) => m | ((b ? 1 : 0) << j), 0)]);
  const sh2 = E.shapExact(look(V2), [1, 1], [[0, 0]]).phi;
  C.exercise({ id: 'pr-x-shap-2', chapter: 'pr-explain', title: 'Shapley values for two features', level: 3, kind: 'table',
    text: ['A model with two features. The average output with no feature known is v(∅) = ' + V2[0] + '; with only feature 1 known, v({1}) = ' + V2[1] + '; with only feature 2, v({2}) = ' + V2[2] + '; with both, v({1, 2}) = ' + V2[3] + '. Work out the Shapley value of each feature. Three decimal places.'],
    rows: ['feature 1', 'feature 2'], cols: ['φ'], corner: '', cells: { 'feature 1|φ': r3(sh2[0]), 'feature 2|φ': r3(sh2[1]) }, tolerance: 0.0015,
    hints: ['With two features each one is added either first (to ∅) or second (to the other), each with weight one half.', 'Feature 1: ½ · (v({1}) − v(∅)) + ½ · (v({1, 2}) − v({2})) = ½ · ' + n3(V2[1] - V2[0]) + ' + ½ · ' + n3(V2[3] - V2[2]) + '.', 'φ₁ = ' + n3(sh2[0]) + ', φ₂ = ' + n3(sh2[1]) + '. They add up to v({1, 2}) − v(∅) = ' + n3(V2[3] - V2[0]) + '.'],
    solution: strCells({ 'feature 1|φ': n3(sh2[0]), 'feature 2|φ': n3(sh2[1]) }),
    wrong: [strCells({ 'feature 1|φ': n3(V2[1] - V2[0]), 'feature 2|φ': n3(V2[2] - V2[0]) }), strCells({ 'feature 1|φ': n3(V2[3] - V2[2]), 'feature 2|φ': n3(V2[3] - V2[1]) })] });

  const shSol = 'function shapley3(v) {\n  const fact = [1, 1, 2, 6];\n  const bits = (m) => (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1);\n  const phi = [0, 0, 0];\n  for (let i = 0; i < 3; i++)\n    for (let m = 0; m < 8; m++) {\n      if (m & (1 << i)) continue;\n      const s = bits(m), w = (fact[s] * fact[3 - s - 1]) / fact[3];\n      phi[i] += w * (v[m | (1 << i)] - v[m]);\n    }\n  return phi;\n}';
  C.exercise({ id: 'pr-x-code-shap3', chapter: 'pr-explain', title: 'Write exact Shapley values for three features', level: 3, kind: 'code',
    text: ['Write `shapley3(v)`: v is a list of 8 numbers, the value of every coalition of three features, indexed by a bit mask (bit i set when feature i is present: v[0] is no feature, v[7] all three, v[5] features 0 and 2). Return `[φ0, φ1, φ2]`, where φᵢ sums, over the masks S without i, |S|! (3 − |S| − 1)! / 3! · (v[S with i] − v[S]).'],
    fn: 'shapley3', params: ['v'], start: 'function shapley3(v) {\n  \n}', tolerance: 1e-9,
    ref: (v) => E.shapExact(look(v), [1, 1, 1], [[0, 0, 0]]).phi,
    cases: [[[0.2, 0.5, 0.3, 0.8, 0.2, 0.5, 0.3, 0.8]], [[0, 1, 2, 3, 4, 5, 6, 7]], [[0.1, 0.4, 0.2, 0.9, 0.3, 0.6, 0.5, 1.4]], [[5, 5, 5, 5, 5, 5, 5, 5]], [[0, 0, 0, 0, 0, 0, 0, 1]]],
    hints: ['Loop over the three features and the 8 masks; skip masks that already contain the feature.', 'The weight depends only on |S|, the number of bits set: 1/3 for |S| = 0, 1/6 for |S| = 1, 1/3 for |S| = 2. The three φ must add up to v[7] − v[0].', shSol],
    solution: shSol,
    wrong: ['function shapley3(v) {\n  const phi = [0, 0, 0];\n  for (let i = 0; i < 3; i++) for (let m = 0; m < 8; m++) { if (m & (1 << i)) continue; phi[i] += (v[m | (1 << i)] - v[m]) / 4; }\n  return phi;\n}', 'function shapley3(v) {\n  return [0, 1, 2].map((i) => v[1 << i] - v[0]);\n}'] });

  const LX = [-0.4, -0.2, 0, 0.2, 0.4], LYf = (x) => 0.5 - 1.5 * x + 2 * x * x + 3 * x * x * x, LY = LX.map((x) => r3(LYf(x))), LW = [0.2, 0.6, 1, 0.6, 0.2];
  const A = [[0, 0], [0, 0]], bb = [0, 0];
  LX.forEach((x, k) => { const r = [1, x]; for (let i = 0; i < 2; i++) { bb[i] += LW[k] * r[i] * LY[k]; for (let j = 0; j < 2; j++) A[i][j] += LW[k] * r[i] * r[j]; } });
  const [lInt, lSlope] = E.solve(A, bb), plain = (() => { const mx = LX.reduce((a, v) => a + v, 0) / 5, my = LY.reduce((a, v) => a + v, 0) / 5; return LX.reduce((a, x, k) => a + (x - mx) * (LY[k] - my), 0) / LX.reduce((a, x) => a + (x - mx) ** 2, 0); })();
  C.exercise({ id: 'pr-x-lime-slope', chapter: 'pr-explain', title: 'A local surrogate’s slope', level: 3, kind: 'number',
    text: ['LIME explains one point of a model with one input. It samples five offsets Δx around the point, asks the model, and weights each sample by closeness:', { table: [['Δx', ...LX.map(mn)], ['model’s score', ...LY.map(mn)], ['weight', ...LW.map(String)]] }, 'Fit score ≈ a + b · Δx by weighted least squares. What is the slope b? Three decimal places.'],
    answer: r3(lSlope), tolerance: 0.0015,
    whyNot: { [n3(plain)]: 'That is the unweighted slope: every sample counted alike. Here the far samples weigh 0.2 and the point itself 1.', '*': 'The weighted mean of Δx is 0 here, so b = Σ w·Δx·score / Σ w·Δx².' },
    hints: ['Weighted least squares: b = Σ w (Δx − x̄)(score − ȳ) / Σ w (Δx − x̄)², with weighted means x̄ and ȳ.', 'Here x̄ = 0, so b = Σ w·Δx·score / Σ w·Δx². Σ w·Δx² = ' + n3(LX.reduce((a, x, k) => a + LW[k] * x * x, 0)) + '.', 'Σ w·Δx·score = ' + mn(n3(LX.reduce((a, x, k) => a + LW[k] * x * LY[k], 0))) + ', so b = ' + mn(n3(lSlope)) + ' (and a = ' + n3(lInt) + '): near this point the score falls by about ' + n2(-lSlope) + ' per unit of x.'],
    solution: n3(lSlope), wrong: [n3(plain)] });

  const wopts = ['Narrower: a wide kernel averages the model over a curved region, which the low R² shows', 'Wider: more samples count', 'More samples at the same width', 'Nothing: the slopes are fine whatever R² says'];
  C.exercise({ id: 'pr-x-lime-width', chapter: 'pr-explain', title: 'A surrogate that does not fit', level: 2, kind: 'choice',
    text: ['At the bend of the moons, a LIME surrogate with kernel width 3 has weighted R² 0.22; with width 0.3 at the same point, 0.93. A colleague reports the slopes of the width-3 fit. What should the kernel be?'],
    options: wopts, answer: 0,
    whyNot: { 1: 'The width-3 kernel already counts far samples, round the curve: that is why the line fits so badly.', 2: 'More samples estimate the same poor fit more precisely.', 3: 'R² 0.22 says the line explains little of what the model does there: its slopes are not a description of the model.' },
    hints: ['What does R² of a surrogate measure?', 'How well the straight line copies the model over the weighted samples.', 'Narrower: near the point the model is nearly straight (R² 0.93 at width 0.3). Check R² and try more than one width before trusting slopes.'],
    solution: 0, wrong: [1, 3] });

  // =========================================================================================================
  // Chapter 4: a real project, start to finish
  // =========================================================================================================
  const STEPS = ['Write down the question and what each mistake costs', 'Look at the data: gaps, class shares, protected columns', 'Split off a test set, stratified', 'Fill gaps and scale, fitted on the training rows only', 'Score a do-nothing baseline, after preparing the data', 'Train a simple model and cross-validate it', 'Read its errors, overall and by group', 'Explain what the model leans on', 'Watch the inputs once it is in use'];
  const SHUF = [4, 0, 7, 2, 8, 5, 1, 3, 6], ordCells = Object.fromEntries(SHUF.map((i) => [STEPS[i] + '|step', i + 1]));
  C.exercise({ id: 'pr-x-checklist-order', chapter: 'pr-project', title: 'Put the checklist in order', level: 2, kind: 'table',
    text: ['The steps of a project, shuffled. Number them 1 to 9 in the order they should be done.'],
    rows: SHUF.map((i) => STEPS[i]), cols: ['step'], corner: 'what', cells: ordCells,
    hints: ['Nothing is fitted before the question, the metric and the split.', 'The baseline comes before any model; explaining and watching come last.', SHUF.map((i) => i + 1).join(', ') + ' down the list: question, look, split, prepare on the training rows, baseline, simple model, errors, explanation, monitoring.'],
    solution: strCells(ordCells),
    wrong: [strCells(Object.fromEntries(SHUF.map((i) => [STEPS[i] + '|step', i === 2 ? 4 : i === 3 ? 3 : i + 1])))] });

  const lopts = ['The test rows helped choose the medians, so the test score is optimistic: split first, then fit the imputer on the training rows', 'Nothing: a median is robust', 'The medians should come from the test rows', 'Medians leak only when scaling is used too'];
  C.exercise({ id: 'pr-x-leak-impute', chapter: 'pr-project', title: 'Filling gaps before the split', level: 2, kind: 'choice',
    text: ['An analyst fills every missing income with the median income of all 600 rows, then splits off 150 test rows and trains. What is wrong?'],
    options: lopts, answer: 0,
    whyNot: { 1: 'Robust or not, the median was computed with the test rows, so they shaped the training data.', 2: 'That leaks the test rows directly into training.', 3: 'Any statistic fitted on all rows leaks, whatever follows.' },
    hints: ['Which rows did the median see?', 'Anything learned from the data, a median, a mean, a scaling, is part of the model.', 'Split first; fit the imputer on the training rows (in the topic, income’s median 38.2) and apply it to the test rows unchanged.'],
    solution: 0, wrong: [1, 2] });

  const BL = fromCounts(0, 37, 0, 113), bacc = M.accuracy(BL.y, BL.p).value;
  C.exercise({ id: 'pr-x-baseline', chapter: 'pr-project', title: 'The baseline to beat', level: 1, kind: 'number',
    text: ['The 150 locked test applications hold 37 defaults. What accuracy does the rule “always predict repaid” score? Three decimal places.'],
    answer: r3(bacc), tolerance: 0.0015,
    whyNot: { [n3(37 / 150)]: 'That is the share of defaults, the rows the rule gets wrong.', 0.75: 'Use the test rows’ own counts: 113 of 150.' },
    hints: ['The rule is right on every applicant who repaid.', '150 − 37 = 113 repaid.', '113 / 150 = ' + n3(bacc) + ': any model must beat this, and on the defaults it finds.'],
    solution: n3(bacc), wrong: [n3(37 / 150), '0.75'] });

  const qopts = ['Investigate the gap between the groups and the missed defaults before relying on the model', 'Deploy it: it beats the baseline', 'Add the group column to fix the gap', 'Retrain until the test accuracy is higher'];
  C.exercise({ id: 'pr-x-read-project', chapter: 'pr-project', title: 'Ready to use?', level: 2, kind: 'choice',
    text: ['The loan model of the checklist: accuracy 0.807 against a baseline of 0.753, cross-validated at 0.798 ± 0.032. It misses 22 defaults and raises 7 false alarms, flags 10% of group A and 22% of group B, and has an equalised-odds gap of 0.14. What comes next?'],
    options: qopts, answer: 0,
    whyNot: { 1: 'Beating the baseline is necessary, not sufficient: the errors and the groups have not been looked at.', 2: 'The gap arose without the column (through income); adding it does not by itself narrow it.', 3: 'Tuning on the test set uses it up.' },
    hints: ['What did the last steps of the checklist reveal?', 'Missed defaults are the costly kind, and the groups are treated differently.', 'Read the missed defaults one by one, decide with the people affected which gap matters, and mitigate (part 17) before anyone relies on it.'],
    solution: 0, wrong: [1, 2] });
})(globalThis.Nooral = globalThis.Nooral || {});
