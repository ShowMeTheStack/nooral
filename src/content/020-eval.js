// Part 2, splitting and evaluation: the chapters and the lessons. The exercises are in 021-eval-ex.js; the model
// challenge, the tour, the course and the pages the diagram checker looks at are in 022-eval-ch.js. Each lesson opens
// with a question, runs the topic that shows the idea at the frame that makes the point, asks for a prediction, sums up
// and leads to its exercises. Every value quoted here is one the topics compute from their own inputs with the engines.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('ev-split')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'ev-split', part: 'evaluation', n: 1, title: 'Rows the model never saw', blurb: 'Training, validation and test sets, stratified cuts for rare classes, time order, and k-fold cross-validation drawn fold by fold.' });
  C.chapter({ id: 'ev-confusion', part: 'evaluation', n: 2, title: 'Counting the mistakes', blurb: 'The confusion matrix and every score read off it: accuracy, precision, recall, F1, specificity, balanced accuracy, MCC, and averages over many classes.' });
  C.chapter({ id: 'ev-curves', part: 'evaluation', n: 3, title: 'Thresholds and probabilities', blurb: 'ROC and precision–recall curves as the threshold moves, AUC as a ranking, and whether a model’s probabilities can be trusted.' });
  C.chapter({ id: 'ev-tuning', part: 'evaluation', n: 4, title: 'Errors, curves and searches', blurb: 'Regression errors and residual plots, bias and variance read off learning and validation curves, and searching for good settings.' });

  // =========================================================================================================
  // Chapter 1: rows the model never saw
  // =========================================================================================================
  C.lesson({
    id: 'ev-split', chapter: 'ev-split', title: 'Training, validation and test', minutes: 7,
    summary: 'A model is judged on rows it never saw. Shuffle, then cut: a training set to fit on, a validation set to choose settings with, and a test set used once at the end. A stratified cut keeps a rare class’s share in every part. Data in time order is cut by time instead.',
    blocks: [
      'A model that has seen an example can simply remember it. So how do we find out how it will do on the examples that matter, the ones that have not happened yet?',
      { topic: 'eval-split', frame: 2, caption: '200 pump readings cut into 120 training, 40 validation and 40 test rows. Settings are chosen on validation; the test rows wait, locked.' },
      { predict: { q: 'Faults are 10 of the 200 rows (5%). After a plain shuffle, how many faults can the 40 test rows be expected to hold?', options: ['Exactly 2', 'About 2, but it could easily be 0, 1 or 4', 'At least 5'], answer: 1, explain: '5% of 40 is 2 on average, but a plain shuffle lets it drift. With this seed the test rows hold only 1 fault: one caught or missed fault swings the recall from 0 to 100%.' } },
      { topic: 'eval-split', frame: 3, caption: 'A plain shuffle: training gets 7 faults (5.8%), validation 2 and test only 1 (2.5%).' },
      { topic: 'eval-split', input: { strat: 'yes' }, frame: 3, caption: 'Stratified: each class is shuffled and cut separately, so training gets 6 faults, validation 2 and test 2, all 5%.' },
      { h: 'When the rows have a time order' },
      'Sales by day, sensor readings by minute, loans by month of issue: the model will be used on the future, so it must be tested on a later period than it was trained on. A shuffle would let it learn from next week to predict yesterday, and neighbouring days are so alike that the test score would be far too kind. Cut by time: train on the past, validate on the next stretch, test on the stretch after that.',
      { warn: 'Use the test set once. Every time a choice is made by looking at the test score, a little of the test set leaks into the model, and the final number becomes optimistic.' },
      { h: 'Recap' },
      { list: ['Training fits, validation chooses, test reports once.', 'Stratify so rare classes keep their share in every part.', 'Time-ordered data: train on the past, test on the future.'] },
      X('ev-x-split-time'), X('ev-x-strat-counts'), X('ev-x-test-peek'),
    ],
  });

  C.lesson({
    id: 'ev-kfold', chapter: 'ev-split', title: 'k-fold cross-validation', minutes: 7,
    summary: 'One split gives one score, and another split would give another. k-fold cross-validation cuts the rows into k folds and fits k models, each tested on the fold it did not see. Every row is tested once, and the k scores give an average and a spread. Shuffle or stratify first.',
    blocks: [
      'One split said 93%. Would another split have said the same?',
      { topic: 'eval-kfold', frame: 1, caption: 'Fold 1 of 5: train on 120 wildflowers, test on the other 30. Accuracy 93.3%.' },
      { topic: 'eval-kfold', frame: 3, caption: 'Fold 3 of 5: the same model on other rows scores 83.3%.' },
      { topic: 'eval-kfold', frame: 6, caption: 'Every row was tested exactly once. The 5 scores average 90.7% with a standard deviation of 4.4 points; one split alone could have reported anything from 83.3% to 96.7%.' },
      { predict: { q: 'Iris is stored sorted by species, 50 of each. Cut it into 3 consecutive folds without shuffling. What does each fold score?', options: ['About 95%, as usual', 'About 67%', '0%'], answer: 2, explain: 'Each test fold holds one whole species, and the training folds hold none of it: the model cannot predict a class it never saw. Every fold scores 0%.' } },
      { topic: 'eval-kfold', input: { data: 'iris', k: 3, how: 'plain' }, frame: 1, caption: 'Sorted rows, no shuffle: the first test fold is all setosa, and the training folds hold none. Accuracy 0%.' },
      { topic: 'eval-kfold', input: { data: 'iris', k: 3, how: 'strat' }, frame: 4, caption: 'The same rows, stratified: 96%, 98% and 96%, a mean of 96.7%.' },
      { workflow: 'cv-wildflowers', live: true, height: 380, caption: 'Cross-validation on the canvas: five folds of wildflowers. Change k in the Cross-validation block and watch the spread of the scores.' },
      { h: 'Recap' },
      { list: ['k folds, k fits, every row tested once.', 'Report the mean and the spread of the k scores.', 'Shuffle or stratify first; for time-ordered rows, use folds that move forward in time.'] },
      X('ev-x-fold-sizes'), X('ev-x-cv-mean'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: counting the mistakes
  // =========================================================================================================
  C.lesson({
    id: 'ev-confusion', chapter: 'ev-confusion', title: 'The confusion matrix', minutes: 7,
    summary: 'Rows are the true classes, columns the predicted ones. For two classes the four cells are the true positives, the false negatives (misses), the false positives (false alarms) and the true negatives. Every score in this chapter is a ratio of these four counts.',
    blocks: [
      'A model is right 84% of the time. Which 16% does it get wrong: the cases it should have caught, or the ones it should have left alone?',
      { topic: 'eval-confusion', frame: 1, caption: '64 test points counted by true class (row) and prediction (column): TP = 25, FN = 7, FP = 3, TN = 29. The diagonal, 54 of 64, is right.' },
      { topic: 'eval-confusion', frame: 2, caption: 'Opening FN: the 7 orange points the model missed, whose scores fell below the threshold 0.5.' },
      { def: 'positive', text: 'the class we are looking for: a fault, a default, a disease. "Positive" says nothing about good or bad; it only names the class that is flagged.' },
      { predict: { q: 'On small loans, lower the threshold from 0.5 to 0.25. What happens to the false negatives (missed defaults) and the false positives (false alarms)?', options: ['Both fall', 'Misses fall, false alarms rise', 'Misses rise, false alarms fall'], answer: 1, explain: 'More applicants are flagged. Misses fall from 32 to 12, and false alarms rise from 17 to 59.' } },
      { topic: 'eval-confusion', input: { src: 'loans', t: 0.25 }, frame: 1, inputs: true, caption: 'Threshold 0.25: TP = 47, FN = 12, FP = 59, TN = 122. Slide the threshold and watch the four cells trade against each other.' },
      { topic: 'eval-confusion', frame: 6, caption: 'Every score is a ratio of the cells: accuracy 0.844, precision 0.893, recall 0.781 and specificity 0.906.' },
      { math: 'accuracy    = (TP + TN) / all\nprecision   = TP / (TP + FP)     of the flagged, how many were positive\nrecall      = TP / (TP + FN)     of the positives, how many were flagged\nspecificity = TN / (TN + FP)     of the negatives, how many were left alone' },
      { h: 'Recap' },
      { list: ['Rows: the truth. Columns: the prediction.', 'FN are misses, FP are false alarms.', 'Moving the threshold trades one for the other.'] },
      X('ev-x-cm-fill'), X('ev-x-code-cm'),
    ],
  });

  C.lesson({
    id: 'ev-scores', chapter: 'ev-confusion', title: 'Precision, recall and F1', minutes: 8,
    summary: 'Lowering the threshold flags more examples: recall rises and precision usually falls. F1, the harmonic mean of the two, is high only when both are. The right balance depends on what a miss costs against what a false alarm costs.',
    blocks: [
      'A lender can flag every applicant as risky and catch every default, or flag nobody and raise no false alarm. Where between the two should it stand?',
      { topic: 'eval-prf', frame: 0, caption: 'At threshold 0.5: 44 applicants flagged, 27 of them real defaults. Precision 27 / 44 = 0.614, recall 27 / 59 = 0.458.' },
      { topic: 'eval-prf', frame: 1, caption: 'At 0.9 only 1 applicant is flagged, and rightly: precision 1, recall 0.017.' },
      { topic: 'eval-prf', frame: 5, caption: 'At 0.1, 170 are flagged: recall 0.949, but precision only 0.329.' },
      { math: 'F1 = 2 · P · R / (P + R)' },
      { predict: { q: 'Precision 1 and recall 0.017. Their plain average is about 0.51. What is F1?', options: ['About 0.51', 'About 0.03', 'About 1'], answer: 1, explain: 'F1 = 2 × 1 × 0.017 / 1.017 = 0.033. The harmonic mean stays near the smaller of the two: a model that flags almost nobody cannot hide behind a perfect precision.' } },
      { topic: 'eval-prf', frame: 6, caption: 'Over thresholds from 0.05 to 0.95, F1 is highest at 0.39: precision 0.574, recall 0.661, F1 0.614.' },
      { tip: 'F1 weighs precision and recall equally. If a missed default costs ten times a false alarm, choose the threshold by that cost instead, or use a score that weighs recall more.' },
      { h: 'Recap' },
      { list: ['Precision: how many flags were right. Recall: how many positives were caught.', 'Lower threshold: recall up, precision usually down.', 'F1 is high only when both are high.'] },
      X('ev-x-scores'), X('ev-x-f1'), X('ev-x-code-prf'),
    ],
  });

  C.lesson({
    id: 'ev-imbalance', chapter: 'ev-confusion', title: 'When one class is rare', minutes: 8,
    summary: 'When 95% of the rows are normal, always saying "normal" is 95% accurate and useless. Balanced accuracy averages the recall of each class, so the do-nothing model scores 50%. The Matthews correlation uses the whole matrix and stays near 0 for guessing, however imbalanced the classes.',
    blocks: [
      'A pump monitor that never raises an alarm is right 95% of the time. Is it a good monitor?',
      { topic: 'eval-balanced', frame: 0, caption: 'Always "normal": right on all 380 normal rows, so accuracy 0.95, yet none of the 20 faults found. Its class recalls are 1 and 0; balanced accuracy is 0.5.' },
      { math: 'balanced accuracy = mean of the recalls of the classes\nMCC = (TP·TN − FP·FN) / √((TP + FP)(TP + FN)(TN + FP)(TN + FN))' },
      { predict: { q: 'Logistic regression catches 19 of the 20 faults and all 380 normal rows. What is its balanced accuracy?', options: ['0.998, like its accuracy', '0.975', '0.95'], answer: 1, explain: 'The recalls are 19/20 = 0.95 and 380/380 = 1, and their mean is 0.975. Accuracy, 0.998, hides the missed fault among 380 easy rows.' } },
      { topic: 'eval-balanced', frame: 3, caption: 'Accuracy ranks the three models 0.998, 0.995 and 0.95, all close. Balanced accuracy ranks them 0.975, 0.95 and 0.5: the do-nothing model is exposed.' },
      { topic: 'eval-averages', frame: 2, caption: 'Four pump states, and no vibration sensor: the 7 vibration faults are never found. Their precision, recall and F1 are all 0.' },
      { topic: 'eval-averages', frame: 4, caption: 'Macro F1, the plain mean over the 4 classes, is 0.748; weighted by class size it is 0.974. The weighted average lets the big class hide the small one.' },
      { topic: 'eval-averages', frame: 6, caption: 'The Matthews correlation over the whole matrix is 0.803: well below the accuracy of 0.983.' },
      { h: 'Recap' },
      { list: ['On imbalanced data, accuracy flatters a model that ignores the rare class.', 'Balanced accuracy and macro averages give every class an equal say.', 'MCC is near 0 for guessing and 1 only for a perfect model.'] },
      X('ev-x-rare-metric'), X('ev-x-balanced'), X('ev-x-mcc'),
    ],
  });

  C.lesson({
    id: 'ev-multi', chapter: 'ev-confusion', title: 'Many classes', minutes: 7,
    summary: 'With k classes the matrix is k × k. Dividing each row by its total shows where each true class went (the diagonal is recall); dividing each column shows how often each prediction is right (the diagonal is precision). Macro averaging takes the plain mean of the per-class scores; micro averaging pools the counts and equals accuracy.',
    blocks: [
      'Three grape cultivars, and a model that is right 69% of the time. Which cultivar is it bad at, and what does it mistake it for?',
      { topic: 'eval-confusion-multi', input: { data: 'wine', model: 'knn' }, frame: 0, caption: 'Unscaled 5-NN on wine: 50 of 72 test rows on the diagonal. The largest mistake: 9 cultivar 3 rows predicted as cultivar 2.' },
      { topic: 'eval-confusion-multi', input: { data: 'wine', model: 'knn' }, frame: 1, caption: 'Normalised by row: each true class’s share that went where. The diagonal is recall: cultivar 1 96%, cultivar 2 72% and cultivar 3 only 32%.' },
      { predict: { q: 'Now normalise by column instead. What does the diagonal show?', options: ['Recall again', 'Precision: when the model says a class, how often it is right', 'The share of each class in the data'], answer: 1, explain: 'A column holds every row predicted as one class; the diagonal cell is the share of them that really were. Here: cultivar 1 79%, cultivar 2 68%, cultivar 3 50%.' } },
      { topic: 'eval-confusion-multi', input: { data: 'wine', model: 'knn' }, frame: 2, caption: 'Normalised by column: each prediction’s reliability. When the model says cultivar 3 it is right only half the time.' },
      { h: 'Averaging over classes' },
      { math: 'macro    = plain mean of the per-class scores\nweighted = mean weighted by each class’s size\nmicro    = pool TP, FP and FN over the classes, then compute once' },
      { topic: 'eval-averages', input: { data: 'flowers' }, frame: 4, caption: 'Micro averaging pools the counts: TP = 59, FP = 1, FN = 1, so precision = recall = F1 = 0.983, exactly the accuracy. Every mistake is one false positive for one class and one false negative for another.' },
      { h: 'Recap' },
      { list: ['Row-normalised: recall on the diagonal. Column-normalised: precision.', 'Macro: every class counts the same. Micro: every example counts the same, and equals accuracy.', 'Look at the largest off-diagonal cell to see which classes are confused.'] },
      X('ev-x-cm-multi'), X('ev-x-row-norm'), X('ev-x-macro-micro'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: thresholds and probabilities
  // =========================================================================================================
  C.lesson({
    id: 'ev-roc', chapter: 'ev-curves', title: 'The ROC curve and AUC', minutes: 8,
    summary: 'Sweep the threshold from the top score down and plot the true-positive rate against the false-positive rate: the ROC curve, from (0, 0) to (1, 1). A guessing model walks the diagonal. The area under the curve equals the share of positive–negative pairs the model orders correctly.',
    blocks: [
      'A model gives every applicant a score. Each threshold gives a different confusion matrix. Can one picture show them all?',
      { topic: 'eval-roc', frame: 0, caption: 'Threshold 0.4 flags 37 of the 59 defaults (TPR 0.627) and 27 of the 181 repaid loans (FPR 0.149): one point on the curve.' },
      { math: 'TPR = TP / (TP + FN)   (recall)\nFPR = FP / (FP + TN)   (1 − specificity)' },
      { topic: 'eval-roc', frame: 1, caption: 'Above every score nothing is flagged, so the curve starts at the corner (0, 0).' },
      { topic: 'eval-roc', frame: 2, caption: 'Threshold 0.8: 6 defaults flagged and no repaid loan, TPR 0.102 and FPR 0. The curve climbs straight up.' },
      { topic: 'eval-roc', frame: 4, caption: 'Lowering the threshold, each default passed moves the curve up and each repaid loan moves it right.' },
      { predict: { q: 'A model that gives random scores. Where does its ROC curve run?', options: ['Along the diagonal from (0, 0) to (1, 1)', 'Along the left and top edges', 'Along the bottom and right edges'], answer: 0, explain: 'With random scores a lower threshold flags positives and negatives at the same rate, so TPR and FPR grow together: the diagonal, with area 0.5.' } },
      { topic: 'eval-roc', frame: 7, caption: 'Below every score everything is flagged: (1, 1). The area under the curve is AUC = 0.837.' },
      { h: 'AUC as a ranking' },
      { topic: 'eval-auc-rank', frame: 0, caption: '10 defaults (rows) and 10 repaid loans (columns): 100 pairs. In each pair, did the default get the higher score?' },
      { topic: 'eval-auc-rank', frame: 4, caption: '85 of the 100 pairs are ordered correctly, and the area under the ROC curve of the same 20 loans is 0.85: the same number.' },
      { tip: 'AUC judges the ranking, not any one threshold: two models with the same AUC can need very different thresholds.' },
      { h: 'Recap' },
      { list: ['One point per threshold: (FPR, TPR).', 'Random: the diagonal, AUC 0.5. Perfect: up the left edge, AUC 1.', 'AUC = the chance a random positive scores above a random negative (ties count half).'] },
      X('ev-x-roc-point'), X('ev-x-auc-pairs'), X('ev-x-code-roc'),
    ],
  });

  C.lesson({
    id: 'ev-pr', chapter: 'ev-curves', title: 'Precision–recall curves', minutes: 7,
    summary: 'The precision–recall curve plots precision against recall for every threshold. Its floor is the share of positives, so it shows how many false alarms come with each catch when positives are rare, which the ROC curve hides. Average precision sums it as rectangles.',
    blocks: [
      '16 faults among 320 pump readings. The ROC curve looks respectable. How many false alarms will the night engineer actually see?',
      { predict: { q: 'Before looking at the model: if the pumps were flagged at random, what precision would that give, at any recall?', options: ['About 0.5', 'About 0.05, the share of faults', 'About 0'], answer: 1, explain: 'A random flag is right as often as a random reading is a fault: 16 of 320, 0.05. That is the floor of the precision–recall plot, and it moves with the class balance, unlike the ROC diagonal.' } },
      { topic: 'eval-pr', frame: 0, caption: 'Flagging at random gives precision 0.05 at any recall: the share of faults, drawn as the dashed floor.' },
      { topic: 'eval-pr', frame: 1, caption: 'At recall 0.625 the precision is 0.323: about two false alarms for every fault caught.' },
      { math: 'average precision = Σ (Rₙ − Rₙ₋₁) · Pₙ   over the thresholds, from high to low' },
      { topic: 'eval-pr', frame: 4, caption: 'Average precision 0.633, measured above a floor of 0.05; the ROC AUC of the same scores is 0.73, measured above 0.5 (the small curve).' },
      { topic: 'eval-pr', input: { src: 'loans' }, frame: 4, caption: 'Loans, a quarter of them defaults: average precision 0.667 above a floor of 0.246, ROC AUC 0.837.' },
      { h: 'Choosing a threshold for a recall target' },
      'A plant may require that at least 80% of faults are caught. Among the thresholds that meet it, the best is the one with the highest precision, which is usually the highest such threshold: the curve is read at the target recall, not at its corner.',
      { workflow: 'evaluate-loans', live: true, height: 420, caption: 'The ROC and PR curves on the canvas, with a threshold to move. Open the ROC block and slide the threshold: the point moves on both curves and the confusion matrix is counted again.' },
      { h: 'Recap' },
      { list: ['The PR curve’s floor is the share of positives.', 'With rare positives, read precision, not FPR: it counts the false alarms per catch.', 'Average precision sums the curve as rectangles.'] },
      X('ev-x-ap'), X('ev-x-roc-or-pr'), X('ev-x-threshold-recall'), X('ev-x-code-threshold'),
    ],
  });

  C.lesson({
    id: 'ev-calibration', chapter: 'ev-curves', title: 'Probabilities you can trust', minutes: 8,
    summary: 'When a model says 0.8, is the answer yes about 80% of the time? A reliability diagram bins the predictions and plots the mean prediction against the share that came true; the expected calibration error is the size-weighted gap. Log loss scores each probability by −ln of what it gave the true class.',
    blocks: [
      'Two models flag the same applicants. One says 0.55 for each, the other 0.99. Does it matter?',
      { topic: 'eval-calibration', input: { model: 'logreg' }, frame: 1, caption: 'Logistic regression on loans: the lowest bin predicts 0.095 on average, and 0.065 of its 123 applicants really defaulted.' },
      { topic: 'eval-calibration', input: { model: 'logreg' }, frame: 2, caption: 'All 5 bins: close to the diagonal. Weighted by bin size, the gaps average 0.038: the expected calibration error.' },
      { predict: { q: 'A tree grown to the end gives every leaf a pure class, so it only ever says 0 or 1. Where will its points lie?', options: ['On the diagonal', 'Far from it: what it calls certain often is not', 'Above the diagonal at every bin'], answer: 1, explain: 'Of the applicants it calls certain defaults (1), only 0.427 default. Its calibration error is 0.292.' } },
      { topic: 'eval-calibration', input: { model: 'tree' }, frame: 2, caption: 'The deep tree: only two bins are used, and in the top one only 0.427 really default. Expected calibration error 0.292.' },
      { h: 'Log loss' },
      { topic: 'eval-logloss', frame: 0, caption: 'Each example pays −ln p, where p is the probability given to its true class: 0 at p = 1, 0.693 at 0.5, 2.303 at 0.1.' },
      { topic: 'eval-logloss', frame: 3, caption: 'The log loss is the mean penalty: 0.409 for logistic regression, against 0.558 for always predicting the share of defaults.' },
      { topic: 'eval-logloss', frame: 4, caption: 'The deep tree’s log loss is 10.513: a few confident mistakes cost without limit. Accuracy alone cannot see this.' },
      { h: 'Recap' },
      { list: ['Calibrated: among predictions of 0.8, about 80% come true.', 'Expected calibration error: the size-weighted gap to the diagonal.', 'Log loss punishes confident mistakes hardest; a model can rank well and still be badly calibrated.'] },
      X('ev-x-logloss'), X('ev-x-ece'), X('ev-x-logloss-rank'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: errors, curves and searches
  // =========================================================================================================
  C.lesson({
    id: 'ev-regression', chapter: 'ev-tuning', title: 'Errors of a number', minutes: 7,
    summary: 'A residual is actual minus predicted. MAE averages their sizes, MSE their squares, RMSE returns to the target’s units, and R² compares the squared error with always predicting the mean. A residual plot with a pattern means the model missed something; one huge residual can dominate RMSE.',
    blocks: [
      'A rent model is off by 59 a month on average. Is that good? And are all its misses alike?',
      { topic: 'eval-regression', frame: 1, caption: 'The residual of a flat is actual − predicted: its vertical distance from the dashed line. The largest is −217.4 a month.' },
      { math: 'MAE  = mean |yᵢ − ŷᵢ|\nMSE  = mean (yᵢ − ŷᵢ)²        RMSE = √MSE\nR²   = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)²' },
      { topic: 'eval-regression', frame: 2, caption: 'MAE 59.14 a month; RMSE 71.57. RMSE is at least MAE, and the gap grows with uneven misses.' },
      { topic: 'eval-regression', frame: 3, caption: 'R² = 1 − 384168 / 5801843 = 0.934: the model removes 93% of the squared error of always guessing the mean.' },
      { predict: { q: 'One test rent is mistyped with an extra digit: 12670 a month. Which error grows more?', options: ['MAE', 'RMSE', 'Both grow by the same factor'], answer: 1, explain: 'Squaring lets one row dominate: MAE goes from 59.4 to 211.2, RMSE from 71.9 to 1323.' } },
      { topic: 'eval-regression', input: { case: 'typo' }, frame: 4, caption: 'With the mistyped rent: MAE 211.2 instead of 59.4, RMSE 1323 instead of 71.9.' },
      { topic: 'eval-regression', input: { case: 'curve' }, frame: 4, caption: 'A straight line where the truth bends: the mean residual is 0.283, −0.358 and 0.195 over the low, middle and high predictions. A pattern: the model missed the bend.' },
      { topic: 'eval-regression', input: { case: 'curve2' }, frame: 4, caption: 'The same data with x² added: −0.147, 0.0537, 0.0663. No pattern, only scatter.' },
      { h: 'Recap' },
      { list: ['MAE: typical miss. RMSE: punishes big misses. R²: the share of variance explained.', 'Plot residuals against predictions: a band is good, a shape is a missed pattern.', 'One wild row can dominate RMSE; check it before trusting the number.'] },
      X('ev-x-reg-metrics'), X('ev-x-residual-pattern'),
    ],
  });

  C.lesson({
    id: 'ev-bias-variance', chapter: 'ev-tuning', title: 'Bias, variance and the curves that show them', minutes: 8,
    summary: 'A learning curve measures the training and validation error as the training set grows: both high and close together means high bias, a large gap means high variance. A validation curve sweeps a model’s flexibility: choose the setting where the validation error is lowest, never where the training error is.',
    blocks: [
      'A model is disappointing. Will more data help, or does it need a different model?',
      { topic: 'eval-learning-curve', frame: 4, caption: 'A straight line on a curved pattern: at 240 rows, training error 0.264 and validation 0.269, both well above the noise floor of 0.09. High bias: more rows will not help.' },
      { topic: 'eval-learning-curve', input: { model: 'deep' }, frame: 4, caption: 'A tree grown to the end: training error 0, validation 0.189. High variance: it memorised the noise.' },
      { predict: { q: 'For the deep tree, what should help most?', options: ['A more flexible model', 'More rows, or a simpler model', 'Nothing: 0.189 is the floor'], answer: 1, explain: 'A large gap between training and validation error is variance. More rows narrow the gap; a depth limit removes it. A depth-4 tree ends at 0.078 and 0.111, near the noise floor.' } },
      { topic: 'eval-learning-curve', input: { model: 'tree4' }, frame: 4, caption: 'A depth-4 tree: training 0.078, validation 0.111. Both near the floor with a small gap.' },
      { h: 'Validation curves' },
      { topic: 'eval-validation-curve', frame: 4, caption: 'Tree depth on a noisy wave: the validation error is lowest at depth 4 (0.111). At depth 14 the training error is 0.001 but validation 0.19.' },
      { workflow: 'learning-curve-tree', title: 'A learning curve on the canvas', text: 'A tree on two moons at every depth, trained on more and more rows.' },
      { h: 'Recap' },
      { list: ['High bias: both errors high and close. Add flexibility or features.', 'High variance: a big gap. Add data, regularise, or simplify.', 'Choose a setting by validation error, never training error.'] },
      X('ev-x-diagnose'), X('ev-x-val-curve'),
    ],
  });

  C.lesson({
    id: 'ev-search', chapter: 'ev-tuning', title: 'Searching for good settings', minutes: 6,
    summary: 'Grid search cross-validates every pair from fixed lists and draws the scores as a heat map. Random search samples the same ranges; for the same number of fits it tries many more distinct values of each setting. Either way the search uses validation folds, and the test set stays locked.',
    blocks: [
      'An RBF support vector machine has two knobs, C and γ, each worth trying over four orders of magnitude. How do we find a good pair without trying everything?',
      { topic: 'eval-search', frame: 3, caption: 'Grid search: 5 values of C by 5 of γ, each cross-validated with 5 folds, 125 fits. The best is C = 1, γ = 1 with 0.875; very large γ memorises points and scores 0.683.' },
      { predict: { q: 'Random search gets 12 draws, 60 fits instead of 125. How many different values of C does it try?', options: ['5, as the grid', '12', '3'], answer: 1, explain: 'Every draw picks a fresh C and a fresh γ, so 12 draws try 12 values of each, where the grid tried 5. When only one knob matters much, that finer coverage pays.' } },
      { topic: 'eval-search', frame: 5, caption: 'After 12 random draws the best is C = 11.3, γ = 0.841 with 0.875: as good as the full grid, with half the fits.' },
      { warn: 'The best cross-validated score of a search is a little optimistic: it was chosen as the best of many. Report the final model on the untouched test set.' },
      { h: 'Recap' },
      { list: ['Grid: every combination of fixed lists; cost multiplies with each knob.', 'Random: samples the ranges, more distinct values per knob for the same cost.', 'Search on validation folds; test once at the end.'] },
      X('ev-x-grid-count'), X('ev-x-wf-loans'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
