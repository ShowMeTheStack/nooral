// Part 18, practice and interpretability: the model challenge, the guided tour, the course, and the pages the diagram
// checker looks at. The challenge's rows come from S.Data with a fixed seed, and its hidden rows from another seed of
// the same generator, so a score means the same on every visit. The target and the marks were set from the engines'
// own scores of the candidate workflows.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('pr-ch-faults')) return;
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // ---- rare faults in patchy readings: F1 on the fault class -------------------------------------------------------------
  const faults = { data: { set: 'pumps', n: 400, seed: 5, opts: { missing: 0.25 } }, hidden: { seed: 81, n: 2000 }, metric: 'f1', metricName: 'F1 on faults', target: 0.8, marks: { silver: 0.815, gold: 0.82 } };
  const imp = [['impute', { strategy: 'median' }]], impScale = [['impute', { strategy: 'median' }], ['scale', { how: 'standard' }]];
  C.challenge(Object.assign({ id: 'pr-ch-faults', n: 181, title: 'Rare faults, patchy sensors', level: 3,
    story: 'A small water works has 400 minutes of logged pump readings, and only 20 of them are faults. Its sensors are old: a quarter of all readings were never recorded. The engineers want an alarm that catches faults without crying wolf, so the plant will judge it by F1 on the fault class, over 2,000 minutes it has kept back, with the same gaps.\n\nA model that never raises the alarm is 95% accurate and worthless here. Some models are pulled towards “normal” by the 19 normal minutes for every fault; others lose the fault in the gaps. Reach an F1 of 0.80 on the hidden minutes.',
    task: 'classification',
    limits: { blocks: ['split', 'clean', 'impute', 'scale', 'resample', 'logistic', 'knn', 'naivebayes', 'tree', 'train', 'evaluate'], models: ['logistic', 'knn', 'naivebayes', 'tree'], params: { tree: { maxDepth: [1, 6] } }, maxBlocks: 8 },
    hints: ['Fill the gaps first (Impute), and read F1 and the fault row of the confusion matrix in Evaluate, not accuracy.', '5-NN lets four normal neighbours outvote a fault (F1 about 0.58); logistic regression draws one straight boundary and stops near 0.78. Balanced class weights (on the model block) or oversampling (Resample) catch more faults but raise far more false alarms: logistic regression with balanced weights falls to 0.46, and 0.71 with its threshold raised to 0.7. A shallow tree asks one question per sensor and can catch each kind of fault on its own reading.', 'Split → Impute (median) → Decision tree with max depth 5 and at least 3 rows per leaf → Train → Evaluate: F1 0.826 on the hidden minutes, a gold mark. Depth 4 alone gives 0.817.'],
    solution: pipe(faults.data, 'tree', { maxDepth: 5, minSamplesLeaf: 3 }, imp),
    wrong: [pipe(faults.data, 'knn', { k: 5 }, impScale), pipe(faults.data, 'logistic', {}, impScale), pipe(faults.data, 'tree', { maxDepth: 2 }, imp)] }, faults));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'pr-tour', title: 'Tour: practice and interpretability', blurb: 'Rare classes, choosing a model, four ways to explain one, and a project checklist, in eight steps.',
    steps: [
      { route: 'part=practice', target: '#main', text: 'Part 18 of the catalogue: practice and interpretability. The topics run real models on our own data sets and explain them with the engine’s own tools.' },
      { route: 'topic=imbalanced-data', target: '#inputs', text: 'Choose a fix here, class weights, oversampling or undersampling, and watch the boundary move and recall trade against precision.' },
      { route: 'topic=model-selection', target: '#stageWrap', text: 'Six models cross-validated fold by fold, and the one-standard-error rule picking the simplest close to the best.' },
      { route: 'topic=permutation-importance', target: '#stageWrap', text: 'Shuffle one column and watch the score fall: what the model leans on.' },
      { target: '#btnPlay', text: 'Play steps through the shuffles; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=shap-waterfall', target: '#stageWrap', text: 'One prediction explained: each feature’s contribution, adding up exactly from the average to the answer.' },
      { route: 'topic=project-checklist', target: '#stageWrap', text: 'A whole project in ten steps, from the question to watching the model in use.' },
      { route: 'learn', target: '[data-chapter="pr-imbalance"]', text: 'The lessons and exercises of this part start here: class weights, the right metric, a waterfall to read, Shapley values and permutation importance in code, and a challenge with rare faults.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'pr-course', title: 'Practice and interpretability', blurb: 'All eight topics of part 18, from a rare class to a project checklist, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'pr-baseline', note: 'The model that does nothing.' },
      { topic: 'imbalanced-data', note: 'Weights and resampling.' },
      { exercise: 'pr-x-class-weights', note: 'Weights by hand.' },
      { exercise: 'pr-x-metric-imb', note: 'The right metric.' },
      { lesson: 'pr-cv-select', note: 'Scores with their noise.' },
      { topic: 'model-selection', note: 'The one-standard-error rule.' },
      { exercise: 'pr-x-one-se', note: 'Draw the line.' },
      { topic: 'no-free-lunch', note: 'No model wins everywhere.' },
      { topic: 'permutation-importance', note: 'Shuffle a column.' },
      { exercise: 'pr-x-code-perm', note: 'Write it.' },
      { topic: 'partial-dependence', note: 'ICE curves and their average.' },
      { exercise: 'pr-x-pdp-read', note: 'Read a PDP.' },
      { topic: 'shap-waterfall', note: 'Contributions that add up.' },
      { exercise: 'pr-x-shap-sum', note: 'Where the waterfall ends.' },
      { exercise: 'pr-x-code-shap3', note: 'Exact Shapley values in code.' },
      { topic: 'lime-local', note: 'A straight line near one point.' },
      { exercise: 'pr-x-lime-slope', note: 'A surrogate’s slope.' },
      { lesson: 'pr-checklist-a', note: 'Before the first model.' },
      { topic: 'project-checklist', note: 'Start to finish.' },
      { exercise: 'pr-x-checklist-order', note: 'Put it in order.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson pr-baseline')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'practice').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['pr-x-imb-scores', 'pr-x-class-weights', 'pr-x-wf-balanced', 'pr-x-one-se', 'pr-x-code-perm', 'pr-x-pdp-avg', 'pr-x-pdp-read', 'pr-x-shap-read', 'pr-x-code-shap3', 'pr-x-lime-slope', 'pr-x-checklist-order'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course pr-course', hash: 'course=pr-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge pr-ch-faults', hash: 'challenge/pr-ch-faults', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
