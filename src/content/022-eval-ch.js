// Part 2, splitting and evaluation: the model challenge, the guided tour, the course, and the pages the diagram checker
// looks at. The challenge’s rows come from S.Data with a fixed seed, and its hidden rows from another seed of the same
// generator, so a score means the same on every visit. The target and the marks were set from the engines’ own scores.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('ev-ch-odds')) return;
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // ---- honest odds for a lender: log loss ---------------------------------------------------------------------------
  const odds = { data: { set: 'loans', n: 500, seed: 6 }, hidden: { seed: 66, n: 800 }, metric: 'logloss', target: 0.48, marks: { silver: 0.475, gold: 0.472 } };
  C.challenge(Object.assign({ id: 'ev-ch-odds', n: 21, title: 'Odds the lender can price', level: 2,
    story: 'A lender of small loans does not simply accept or refuse: it sets each applicant’s interest rate from the chance of default. A model that says 0.9 when the truth is nearer 0.4 overcharges good customers and drives them away; one that says 0.01 for a risky applicant loses money on every such loan.\n\nSo the lender scores models by log loss on 800 applications it has kept back: every applicant costs −ln of the probability the model gave to what really happened. A model that is right but overconfident pays heavily for each of its confident mistakes. Bring the log loss down to 0.48.',
    task: 'classification',
    limits: { blocks: ['split', 'encode', 'scale', 'tree', 'knn', 'naivebayes', 'logistic', 'train', 'evaluate'], models: ['tree', 'knn', 'naivebayes', 'logistic'], params: { knn: { k: [1, 60] } }, maxBlocks: 8 },
    hints: ['Open Evaluate and the Calibration view of each model you try: which ones say 0 or 1 far too often?', 'A deep tree is certain and often wrong; 5-NN can only say 0, 0.2, …, 1. Naive Bayes is close but not quite there. A model that is fitted by minimising log loss itself is a natural choice.', 'Split → Encode → Scale (standard) → Logistic regression → Train → Evaluate: log loss 0.472 on the hidden applications, a gold mark. (25-NN gets 0.503 and naive Bayes 0.496.)'],
    solution: pipe(odds.data, 'logistic', {}, [['encode', {}], ['scale', { how: 'standard' }]]),
    wrong: [pipe(odds.data, 'knn', { k: 25 }, [['encode', {}], ['scale', { how: 'standard' }]]), pipe(odds.data, 'tree', { maxDepth: 3 }), pipe(odds.data, 'naivebayes', {})] }, odds));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'ev-tour', title: 'Tour: splitting and evaluation', blurb: 'Splits, folds, the confusion matrix, a threshold to move, calibration and searches, in eight steps.',
    steps: [
      { route: 'part=evaluation', target: '#main', text: 'Part 2 of the catalogue: splitting and evaluation. Every score here is counted from real predictions, and the counts are shown.' },
      { route: 'topic=eval-split', target: '#stageWrap', text: 'Training, validation and test rows, and what a stratified cut does for a rare class.' },
      { route: 'topic=eval-kfold', target: '#stageWrap', text: 'k-fold cross-validation drawn fold by fold: every row is tested once, and the spread of the scores shows how much one split could mislead.' },
      { target: '#btnPlay', text: 'Play steps through the folds; the arrow keys go one step at a time.' },
      { route: 'topic=eval-confusion', target: '#inputs', text: 'Move the threshold here: the four cells of the confusion matrix trade against each other, and every score is read off them.' },
      { route: 'topic=eval-roc', target: '#stageWrap', text: 'The ROC curve, one point per threshold, and its area as a ranking of pairs.' },
      { route: 'topic=eval-calibration', target: '#stageWrap', text: 'Can the probabilities be trusted? Bins of predictions against what came true.' },
      { route: 'learn', target: '[data-chapter="ev-split"]', text: 'The lessons and exercises of this part start here: confusion matrices to count, scores and curves by hand, code to write, and a workflow that must rank applicants on hidden rows.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'ev-course', title: 'Splitting and evaluation', blurb: 'All sixteen topics of part 2, from a train/test split to a hyperparameter search, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'ev-split', note: 'Rows the model never saw.' },
      { topic: 'eval-split', note: 'Three parts, and a stratified cut.' },
      { exercise: 'ev-x-split-time', note: 'Time order.' },
      { topic: 'eval-kfold', note: 'Every row tested once.' },
      { exercise: 'ev-x-fold-sizes', note: 'Folds of uneven size.' },
      { topic: 'eval-confusion', note: 'Four cells to click.' },
      { exercise: 'ev-x-cm-fill', note: 'Count one yourself.' },
      { topic: 'eval-prf', note: 'Precision against recall.' },
      { exercise: 'ev-x-scores', note: 'Five scores from four counts.' },
      { topic: 'eval-confusion-multi', note: 'Many classes, normalised.' },
      { topic: 'eval-balanced', note: 'When one class is rare.' },
      { topic: 'eval-averages', note: 'Macro, micro, weighted, MCC.' },
      { exercise: 'ev-x-macro-micro', note: 'Three averages by hand.' },
      { lesson: 'ev-roc', note: 'A curve of thresholds.' },
      { topic: 'eval-roc', note: 'The ROC sweep.' },
      { topic: 'eval-auc-rank', note: 'AUC as pairs.' },
      { exercise: 'ev-x-auc-pairs', note: 'Count the pairs.' },
      { topic: 'eval-pr', note: 'Precision–recall for rare positives.' },
      { exercise: 'ev-x-threshold-recall', note: 'A threshold for a recall target.' },
      { topic: 'eval-calibration', note: 'Reliability diagrams.' },
      { topic: 'eval-logloss', note: 'Paying for confidence.' },
      { exercise: 'ev-x-logloss', note: 'Log loss by hand.' },
      { topic: 'eval-regression', note: 'Errors of a number.' },
      { exercise: 'ev-x-reg-metrics', note: 'Four errors from five residuals.' },
      { topic: 'eval-learning-curve', note: 'Bias or variance?' },
      { topic: 'eval-validation-curve', note: 'One knob swept.' },
      { topic: 'eval-search', note: 'Grid against random.' },
      { exercise: 'ev-x-wf-loans', note: 'A workflow scored on hidden rows.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson ev-split')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'evaluation').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['ev-x-strat-counts', 'ev-x-cm-fill', 'ev-x-scores', 'ev-x-cm-multi', 'ev-x-row-norm', 'ev-x-roc-point', 'ev-x-code-roc', 'ev-x-threshold-recall', 'ev-x-ece', 'ev-x-val-curve', 'ev-x-wf-loans'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course ev-course', hash: 'course=ev-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge ev-ch-odds', hash: 'challenge/ev-ch-odds', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
