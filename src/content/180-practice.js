// Part 18, practice and interpretability: the chapters and the lessons. The exercises are in 181-practice-ex.js; the
// model challenge, the tour, the course and the pages the diagram checker looks at are in 182-practice-ch.js. Each
// lesson opens with a question, runs the topic that shows the idea at a frame chosen for the point being made, asks
// for a prediction, sums up, and leads to its exercises. Every value quoted here was printed by the topics themselves
// (S.ML, S.Metrics and S.Explain on the topics' own data).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('pr-baseline')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'pr-imbalance', part: 'practice', n: 1, title: 'When one class is rare', blurb: 'The do-nothing baseline that scores 95%, class weights and resampling that move the boundary, and the metrics that tell the models apart.' });
  C.chapter({ id: 'pr-select', part: 'practice', n: 2, title: 'Choosing a model', blurb: 'Cross-validation scores with their spread, the one-standard-error rule, a test set used once, and why no model wins everywhere.' });
  C.chapter({ id: 'pr-explain', part: 'practice', n: 3, title: 'Explaining a model', blurb: 'Permutation importance for the model as a whole, partial dependence and ICE curves for one feature, SHAP-style contributions and LIME-style surrogates for one prediction.' });
  C.chapter({ id: 'pr-project', part: 'practice', n: 4, title: 'A real project, start to finish', blurb: 'A checklist: the question and its costs, a look at the data, the split before anything else, a baseline, a simple model, its errors and its groups, an explanation, and watching it once in use.' });

  // =========================================================================================================
  // Chapter 1: when one class is rare
  // =========================================================================================================
  C.lesson({
    id: 'pr-baseline', chapter: 'pr-imbalance', title: 'The model that does nothing and scores 95%', minutes: 6,
    summary: 'When 5% of the rows are rare, a model that always says “common” is 95% accurate and useless. Every real model must be compared with that baseline, on metrics that look at the rare class: its recall, precision and F1, balanced accuracy, or average precision.',
    blocks: [
      'A fraud model is 95.6% accurate. Is that good? It depends entirely on how common fraud is. If it is 1 transaction in 20, a model that never flags anything scores 95%.',
      { topic: 'imbalanced-data', frame: 0, input: { set: 'toy', fix: 'none', seed: 1 }, caption: '300 rows, of which 15 (5%) are rare. A stratified split keeps that share: 209 rows to train on, 91 to test, with only 5 rare test rows.' },
      { predict: { q: 'A model says “common” for every test row. What is its accuracy?', options: ['About 0.5', '0.945', '0'], answer: 1, explain: '86 of the 91 test rows are common: 86 / 91 = 0.945, and it finds none of the rare ones.' } },
      { topic: 'imbalanced-data', frame: 1, input: { set: 'toy', fix: 'none', seed: 1 }, caption: 'The laziest model: accuracy 0.945, recall of the rare class 0.' },
      { topic: 'imbalanced-data', frame: 2, input: { set: 'toy', fix: 'none', seed: 1 }, caption: 'A logistic model trained as usual: accuracy 0.956, barely above the baseline. It catches 1 rare test point of 5 (recall 0.20): missing a few rare rows costs little in the average loss.' },
      { table: [['Metric', 'Do-nothing model', 'What it looks at'], ['accuracy', 'high', 'all rows; swamped by the common class'], ['recall of the rare class', '0', 'how many rare rows are found'], ['precision of the rare class', 'undefined (nothing flagged)', 'how many flags are right'], ['F1, balanced accuracy, average precision', 'low', 'the rare class, or both classes equally']], caption: 'Choose a metric the do-nothing model cannot game.' },
      { h: 'Recap' },
      { list: ['With a rare class, accuracy rewards doing nothing.', 'Always score the do-nothing baseline first.', 'Compare models on the rare class: recall, precision, F1, balanced accuracy, average precision.'] },
      X('pr-x-acc-paradox'), X('pr-x-metric-imb'), X('pr-x-imb-scores'),
    ],
  });

  C.lesson({
    id: 'pr-weights', chapter: 'pr-imbalance', title: 'Class weights and resampling', minutes: 8,
    summary: 'Class weights multiply each row’s loss so that both classes weigh the same in total: a row of class c gets n / (k · n_c). Oversampling copies rare rows; undersampling drops common ones. All three push the boundary out towards the common class, buying recall with precision.',
    blocks: [
      'The usual loss is an average over rows, so 199 common rows outvote 10 rare ones. What if each rare row counted for more?',
      { math: 'weight of a row of class c = n / (k · n_c)\nn rows, k classes, n_c rows of class c: every class then adds up to n / k' },
      { topic: 'imbalanced-data', frame: 3, caption: 'Class weights: each rare row counts 10.45 and each common row 0.53. The 50% line (solid) moves out past the old one (dashed): recall rises to 1.00, precision is 0.24 and accuracy 0.824.' },
      { predict: { q: 'With the weights, accuracy fell from 0.956 to 0.824. Is the weighted model worse?', options: ['Yes: accuracy fell', 'Not for finding rare rows: recall went from 0.20 to 1.00', 'They are the same model'], answer: 1, explain: 'It flags more rows, so it makes more false alarms, but it finds every rare test row. Which matters more is the project’s question.' } },
      { topic: 'imbalanced-data', frame: 3, input: { set: 'toy', fix: 'over', seed: 1 }, caption: 'Oversampling copies rare training rows until both classes count 199: recall 1.00, precision 0.21, accuracy 0.791.' },
      { topic: 'imbalanced-data', frame: 3, input: { set: 'toy', fix: 'under', seed: 1 }, caption: 'Undersampling keeps only 10 common rows: 20 rows remain. Recall 1.00, precision 0.23, accuracy 0.813, from far less data.' },
      { topic: 'imbalanced-data', frame: 4, caption: 'Side by side: balanced accuracy goes from 0.50 for the baseline to 0.91 with weights. Average precision is 0.63 without the fix and 0.62 with it: the fix mostly moves the threshold, not the ranking.' },
      { warn: 'Resample only the training rows, after the split. Copies of a rare row on both sides of the split let the model be tested on rows it has memorised.' },
      { h: 'Recap' },
      { list: ['Class weights n / (k · n_c) make every class weigh the same in the loss.', 'Oversampling and undersampling do the same with rows; undersampling throws data away.', 'They move the threshold: recall up, precision down. Ranking metrics barely move.'] },
      X('pr-x-class-weights'), X('pr-x-code-weights'), X('pr-x-resample-order'), X('pr-x-weights-ranking'),
    ],
  });

  C.lesson({
    id: 'pr-imb-real', chapter: 'pr-imbalance', title: 'Rare faults in real readings', minutes: 6,
    summary: 'On pump readings with 6% faults, class weights lift recall from 0.29 to 0.57 at the cost of precision. Judged by balanced accuracy the gain is clear; judged by accuracy it looks like a loss. The metric must be chosen before the models are compared.',
    blocks: [
      'The toy set was clean. Real faults are rarer, noisier and partly hidden among normal readings. Does the same fix still help?',
      { topic: 'imbalanced-data', frame: 0, input: { set: 'pumps', fix: 'weights', seed: 1 }, caption: '400 pump readings, 24 (6%) faults: 7 fault rows in the test set.' },
      { topic: 'imbalanced-data', frame: 2, input: { set: 'pumps', fix: 'weights', seed: 1 }, caption: 'Trained as usual: accuracy 0.958, but it catches only 2 of 7 faults (recall 0.29).' },
      { predict: { q: 'Class weights make each fault row count 8.24. Will recall reach 1.00 as on the toy set?', options: ['Yes', 'It rises, but not to 1.00', 'It falls'], answer: 1, explain: 'It rises to 0.57: some faults look just like normal readings to a straight boundary.' } },
      { topic: 'imbalanced-data', frame: 3, input: { set: 'pumps', fix: 'weights', seed: 1 }, caption: 'With class weights: recall 0.57, precision 0.20, accuracy 0.842.' },
      { topic: 'imbalanced-data', frame: 4, input: { set: 'pumps', fix: 'weights', seed: 1 }, caption: 'Balanced accuracy goes from 0.50 to 0.71; average precision stays at 0.49. A better model, not just a different threshold, is needed for more.' },
      { challenge: 'pr-ch-faults' },
      { h: 'Recap' },
      { list: ['Choose the metric before comparing models.', 'Weights help a linear model find more of the rare class, at the cost of more false alarms.', 'When the classes overlap, a more flexible model raises the ranking itself.'] },
      X('pr-x-wf-balanced'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: choosing a model
  // =========================================================================================================
  C.lesson({
    id: 'pr-cv-select', chapter: 'pr-select', title: 'Choosing a model by cross-validation', minutes: 8,
    summary: 'Lock a test set away. Score every candidate by k-fold cross-validation on the rest, as a mean and a spread. The best mean is noisy too: the one-standard-error rule picks the simplest model within one standard error (spread / √k) of the best. Then score that one model on the test set, once.',
    blocks: [
      'Six models, one data set. Pick the one with the best score, surely? But scores from a few hundred rows are noisy. Is a winner by 0.01 really better?',
      { topic: 'model-selection', frame: 0, caption: '240 rows: 60 locked away as the test set, the other 180 dealt into 5 folds of about 36 rows, keeping the class shares.' },
      { topic: 'model-selection', frame: 4, caption: 'K-NN: fold scores 0.89, 1.00, 0.97, 0.92, 0.86. Mean 0.928, spread 0.052.' },
      { topic: 'model-selection', frame: 6, caption: 'The random forest: mean 0.939, spread 0.041, the best mean.' },
      { math: 'standard error = spread / √k\none-standard-error rule: among the models whose mean is at least (best mean − its standard error), take the simplest' },
      { predict: { q: 'The forest’s standard error is 0.018. Is k-NN, at 0.928, within one standard error of 0.939?', options: ['Yes: 0.939 − 0.018 = 0.921, and 0.928 is above it', 'No: it is lower', 'Only if k = 10'], answer: 0, explain: 'Anything above 0.921 is within the noise of the best.' } },
      { topic: 'model-selection', frame: 7, caption: 'Anything above 0.921 (dashed line) is within the noise of the best. The simplest model that clears it is k-NN, at 0.928.' },
      { topic: 'model-selection', frame: 8, caption: 'Retrained on all 180 rows, k-NN scores 0.900 on the 60 test rows. This number is reported, not used to choose again.' },
      { h: 'Recap' },
      { list: ['Cross-validate every candidate; report mean and spread.', 'Within one standard error of the best, prefer the simplest.', 'The test set is used once, at the end, to report.'] },
      X('pr-x-cv-stats'), X('pr-x-one-se'), X('pr-x-one-se-pick'), X('pr-x-test-reuse'),
    ],
  });

  C.lesson({
    id: 'pr-nfl', chapter: 'pr-select', title: 'No free lunch', minutes: 6,
    summary: 'No model is best on every problem: each assumes something about the shape of the answer, and wins where that assumption holds. A straight boundary is unbeatable on a noisy line and hopeless on XOR. That is why candidates are compared on the problem at hand.',
    blocks: [
      'Is there one model to use for everything? If there were, this course would be much shorter.',
      { topic: 'no-free-lunch', frame: 0, caption: 'Six data sets, five pictures in the plane and the loans table, and six models scored by 3-fold cross-validation on each.' },
      { topic: 'no-free-lunch', frame: 1, caption: 'The noisy line: logistic regression is best, 0.81; k-NN worst, 0.77.' },
      { predict: { q: 'On XOR, four blobs in a checker pattern, how will logistic regression do?', options: ['Best again', 'About as well as guessing', 'Middling'], answer: 1, explain: 'No straight line separates XOR: it scores 0.47.' } },
      { topic: 'no-free-lunch', frame: 4, caption: 'XOR: the tree is best with 0.93, logistic regression worst with 0.47.' },
      { topic: 'no-free-lunch', frame: 7, caption: 'Over the six sets, 4 different models win at least once. The best average rank, 2.83, belongs to k-NN, yet it is not first everywhere.' },
      { h: 'Recap' },
      { list: ['Every model assumes a shape for the answer.', 'It wins where the assumption fits and loses where it does not.', 'So compare candidates on your own data, honestly.'] },
      X('pr-x-nfl'), X('pr-x-rank'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: explaining a model
  // =========================================================================================================
  C.lesson({
    id: 'pr-perm', chapter: 'pr-explain', title: 'Permutation importance', minutes: 7,
    summary: 'Shuffle one column of the test set and score the model again: the drop in score is that feature’s importance to this model. Repeat the shuffle a few times and report the mean and spread. Correlated features share their importance, and importance says what the model uses, not what causes the outcome.',
    blocks: [
      'Which inputs does a loan model lean on? We cannot read a forest of trees, but we can break one input at a time and see how much worse it gets.',
      { topic: 'permutation-importance', frame: 0, caption: 'A random forest on 420 rows of the loans data: AUC 0.753 on the 180 test rows, with every column intact.' },
      { topic: 'permutation-importance', frame: 1, caption: 'Shuffle the debt ratio column: each row now carries another row’s value. AUC falls to 0.675, a drop of 0.078.' },
      { topic: 'permutation-importance', frame: 2, caption: 'Over 5 shuffles the drops run from 0.059 to 0.116: importance 0.080 (the mean), spread 0.020.' },
      { topic: 'permutation-importance', frame: 3, caption: 'All 5 features: debt ratio 0.080, then income 0.069; years 0.000.' },
      { predict: { q: 'A noisy copy of income is added as a sixth column. What happens to income’s importance?', options: ['It stays at 0.069', 'It falls: the model can fall back on the copy', 'It doubles'], answer: 1, explain: 'Shuffling income leaves the copy to fall back on: income scores 0.005 and the copy 0.018.' } },
      { topic: 'permutation-importance', frame: 4, input: { set: 'copy', reps: '5', seed: 1 }, caption: 'With a noisy copy beside it, income scores 0.005 and the copy 0.018. Correlated features each look less important than the information they share.' },
      { workflow: 'explain-harvest', title: 'Explaining a forest on the canvas', text: 'The Permutation importance block does the same for any trained model: open it after Train to see every column’s drop, with its spread.' },
      { h: 'Recap' },
      { list: ['Importance = score with the column intact − score with it shuffled, averaged over shuffles.', 'Correlated features split their importance.', 'It describes the model, not the world.'] },
      X('pr-x-perm-drop'), X('pr-x-perm-read'), X('pr-x-code-perm'),
    ],
  });

  C.lesson({
    id: 'pr-pdp', chapter: 'pr-explain', title: 'Partial dependence and ICE curves', minutes: 7,
    summary: 'An ICE curve sets one feature of one row to each value on a grid, keeps the row’s other features, and plots the model’s answers. Partial dependence is the average of the ICE curves. When the curves are shifted copies, the average tells the story; when they cross, it can hide it.',
    blocks: [
      'How does the predicted rent change with floor area? We cannot change a flat’s area, but we can ask the model what it would say if we did.',
      { topic: 'partial-dependence', frame: 1, caption: 'Row 0 has area 85, where the model says 761. Setting area to 20 values from 37.1 to 115 traces its ICE curve, from 294 to 1100.' },
      { topic: 'partial-dependence', frame: 2, caption: 'One curve per row: at one value of area the curves are up to 433 apart, but mostly shifted copies of one shape.' },
      { topic: 'partial-dependence', frame: 3, caption: 'Their average is the partial dependence (thick): lowest 498 at area = 37.1, highest 1293 at 115. The model’s answer moves by 795 on average over this range.' },
      { predict: { q: 'In a case where x₁ raises y when x₂ > 0 and lowers it otherwise, what will the partial dependence on x₁ look like?', options: ['Steeply rising', 'Nearly flat', 'Steeply falling'], answer: 1, explain: 'The rising and falling ICE curves cancel in the average: a rise of only 0.393.' } },
      { topic: 'partial-dependence', frame: 3, input: { case: 'twist', rows: '30' }, caption: 'The partial dependence rises only 0.393, nearly flat, although each row’s curve moves a lot, up for some rows and down for others. Look at the ICE curves before trusting a flat average.' },
      { h: 'Recap' },
      { list: ['ICE: one row, one feature varied, the others kept.', 'Partial dependence: the average of the ICE curves.', 'Crossing ICE curves mean an interaction, which the average hides.'] },
      X('pr-x-pdp-avg'), X('pr-x-pdp-read'), X('pr-x-ice-read'),
    ],
  });

  C.lesson({
    id: 'pr-shap', chapter: 'pr-explain', title: 'SHAP-style contributions for one prediction', minutes: 9,
    summary: 'Shapley values share out the gap between one prediction and the average prediction among the features. A feature’s share is its average added effect over all coalitions of the others, with every coalition size weighted equally. The shares always add up exactly: base value + Σ contributions = the prediction.',
    blocks: [
      'An applicant is refused and asks why. “The model said so” is no answer. Which features pushed this applicant’s score, and by how much?',
      { topic: 'shap-waterfall', frame: 0, caption: 'Over 30 background rows the model gives 0.189 on average; for test row 3 it gives 0.063. The 5 features must share the difference, −0.126.' },
      { topic: 'shap-waterfall', frame: 1, caption: 'Years: added to a coalition of the others, it changes the output by −0.0819 with no others in and by −0.0546 with all 4. Averaged over coalition sizes, its contribution is −0.0697.' },
      { math: 'φᵢ = Σ over coalitions S without i of  |S|! (M − |S| − 1)! / M!  ·  (v(S ∪ {i}) − v(S))\nv(S) = the average output with the features in S fixed at this row’s values' },
      { topic: 'shap-waterfall', frame: 2, caption: 'The waterfall starts at the base, 0.189: years (10.7) pulls the output down by 0.070, to 0.120.' },
      { predict: { q: 'Base 0.189 and contributions −0.070, −0.040, −0.032, +0.012, +0.004. Where does the waterfall end?', options: ['0.063', '0.189', '0.126'], answer: 0, explain: '0.189 − 0.070 − 0.040 − 0.032 + 0.012 + 0.004 = 0.063, the model’s answer for this row.' } },
      { topic: 'shap-waterfall', frame: 7, caption: 'The contributions add up: 0.189 plus their sum is 0.063, the model’s answer. Kernel SHAP, a weighted regression over coalitions, agrees to within 5.5×10⁻¹⁴.' },
      { topic: 'shap-waterfall', frame: 7, input: { case: 'rents', row: 0, bg: '30' }, caption: 'A flat’s rent: from the average 860, area (97) adds 224 and distance (9.3) takes 126; the bars end at 966.' },
      { warn: 'Contributions explain the model’s answer relative to the background rows chosen. A different background gives a different base and different shares, and none of it says what would happen if the applicant changed.' },
      { h: 'Recap' },
      { list: ['Contributions share out prediction − base among the features.', 'Each is an average added effect over coalitions of the others.', 'They add up exactly: base + Σ φ = prediction.'] },
      X('pr-x-shap-sum'), X('pr-x-shap-missing'), X('pr-x-shap-read'), X('pr-x-shap-2'), X('pr-x-code-shap3'),
    ],
  });

  C.lesson({
    id: 'pr-lime', chapter: 'pr-explain', title: 'LIME-style local surrogates', minutes: 7,
    summary: 'Near one point, even a curved model is nearly straight. LIME samples around the point, asks the model about each sample, weights them by closeness, and fits a weighted straight line. Its slopes explain the model near that point, if its weighted R² says the line fits.',
    blocks: [
      'A curved boundary has no single slope. But a curve looks straight through a magnifying glass. What does the model do near this one point?',
      { topic: 'lime-local', frame: 0, caption: 'An SVM’s score, shaded, with its curved boundary. The point to explain, (0.50, 0.30), gets a score of −0.096.' },
      { topic: 'lime-local', frame: 1, caption: '200 samples round the point, each feature jittered by its own spread, and the model asked about each.' },
      { topic: 'lime-local', frame: 2, caption: 'Each sample is weighted by exp(−distance² / width²) with width 0.5: 12 of the 200 weigh more than one half; the far ones hardly count.' },
      { topic: 'lime-local', frame: 3, caption: 'The weighted fit: score ≈ −0.127 − 2.14 × Δx₁ − 3.72 × Δx₂, with weighted R² 0.91: near this point the model really is almost straight.' },
      { predict: { q: 'At the bend of the moons, with a kernel width of 3 instead of 0.3, will the surrogate fit better or worse?', options: ['Better: more samples count', 'Worse: it reaches round the curve', 'The same'], answer: 1, explain: 'R² falls from 0.93 to 0.22: a wide kernel averages over a curved region.' } },
      { topic: 'lime-local', frame: 3, input: { set: 'moons', point: 'bend', width: 3, samples: '200' }, caption: 'Width 3 at the bend: score ≈ 1.132 + 0.27 × Δx₁ − 0.88 × Δx₂, R² 0.22. A poor copy; its slopes should not be trusted.' },
      { h: 'Recap' },
      { list: ['Sample near the point, weight by closeness, fit a weighted line.', 'The slopes say how the model responds near that point.', 'Check the weighted R², and try more than one kernel width.'] },
      X('pr-x-lime-slope'), X('pr-x-lime-width'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: a real project, start to finish
  // =========================================================================================================
  C.lesson({
    id: 'pr-checklist-a', chapter: 'pr-project', title: 'Before the first model', minutes: 7,
    summary: 'Write down the question and what each kind of mistake costs, which decides the metric. Look at the data: gaps, class shares, protected columns. Split first, then fit every preparation step on the training rows alone. Score the do-nothing baseline.',
    blocks: [
      'A lender asks for “a model of who defaults”. Before any model is trained, four decisions shape everything that follows. What are they?',
      { topic: 'project-checklist', frame: 0, caption: 'The question: will this applicant default? 600 past applications, 25% defaulted, and a missed default costs more than a refused good applicant. So recall of defaults matters, not accuracy alone.' },
      { topic: 'project-checklist', frame: 1, caption: 'Look before modelling: 90 cells are missing, over 86 rows; the worst column, years, lacks 21. The group column stays out of the model but is kept to check fairness.' },
      { predict: { q: 'Missing incomes are to be filled with the median. Which rows should the median come from?', options: ['All 600 rows', 'The training rows only', 'The test rows only'], answer: 1, explain: 'A median taken over the test rows lets them leak into training: split first, then fill.' } },
      { topic: 'project-checklist', frame: 2, caption: 'Split first, stratified: 450 to train and 150 locked away. Then fill the gaps with medians from the training rows (income 38.2).' },
      { topic: 'project-checklist', frame: 3, caption: 'The simplest rule, always “repaid”, scores 0.753 accuracy and finds no defaults. Every model is judged against it.' },
      { h: 'Recap' },
      { list: ['The question and the cost of each mistake choose the metric.', 'Look at gaps, shares and protected columns first.', 'Split before fitting anything; then score the baseline.'] },
      X('pr-x-checklist-order'), X('pr-x-leak-impute'), X('pr-x-baseline'),
    ],
  });

  C.lesson({
    id: 'pr-checklist-b', chapter: 'pr-project', title: 'From the first model to the first month in use', minutes: 8,
    summary: 'Start with a simple, readable model and compare it with the baseline. Check the test score against cross-validation, read the errors one by one and by group, explain what the model leans on, and once it is in use, watch what comes in.',
    blocks: [
      'The baseline is set. Now a model, and with it the questions a careful team asks before anyone relies on it.',
      { topic: 'project-checklist', frame: 4, caption: 'Logistic regression on five scaled columns: accuracy 0.807 against the baseline’s 0.753; AUC 0.799; it catches 41% of defaults with precision 0.68.' },
      { topic: 'project-checklist', frame: 5, caption: 'Is the test score luck? 5-fold cross-validation on the training rows: mean 0.798, spread 0.032. The test accuracy 0.807 sits within two spreads.' },
      { topic: 'project-checklist', frame: 6, caption: 'The confusion matrix: 22 defaults missed and 7 false alarms. Reading those rows one by one is how problems get found.' },
      { predict: { q: 'The group column was never an input. Can the model still treat the groups differently?', options: ['No', 'Yes: income differs between them', 'Only if it overfits'], answer: 1, explain: 'It flags 10% of group A and 22% of group B, and its equalised-odds gap is 0.14.' } },
      { topic: 'project-checklist', frame: 7, caption: 'By group: it flags 10% of group A and 22% of group B, and finds 35% and 47% of their defaults; the equalised-odds gap is 0.14.' },
      { topic: 'project-checklist', frame: 8, caption: 'Shuffling income costs 0.089 of AUC, the most of any column, then debt ratio at 0.064: something a loan officer can question.' },
      { topic: 'project-checklist', frame: 9, caption: 'In use: the model flags 15% of the test rows and 15% of a fresh batch; a batch with incomes a fifth lower gets 23%, the signal to check the inputs and retrain.' },
      { h: 'Recap' },
      { list: ['Simple model first; compare with the baseline and with cross-validation.', 'Read the errors, overall and by group.', 'Explain it, then watch it in use.'] },
      X('pr-x-read-project'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
