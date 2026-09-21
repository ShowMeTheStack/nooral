// Part 4, nearest neighbours and probabilistic models: the model challenges, the guided tour, the course, and the
// pages the diagram checker looks at. Each challenge's rows come from a story set of S.Data with a fixed seed; the
// hidden rows use another seed. The scores quoted in hints are what the reference workflows reach on the hidden rows.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('nn-ch-pumps')) return;
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return () => WF.build({ blocks, wires });
  };

  // ---- 1. pump faults with k-NN ---------------------------------------------------------------------------------
  const pumps = { data: { set: 'pumps', n: 400, seed: 3 }, hidden: { seed: 31, n: 600 }, metric: 'f1', metricName: 'F1', target: 0.9, marks: { silver: 0.93, gold: 0.945 } };
  C.challenge(Object.assign({ id: 'nn-ch-pumps', n: 41, title: 'The pump that hums wrong', level: 2,
    story: 'A water works logs four readings from its pumps every minute: temperature in degrees, vibration in millimetres per second, pressure in bar and motor current in amperes. A few readings each week come from a pump about to fail, and the night engineer wants a warning that compares each new reading with the ones already labelled.\n\nThe sensors drop out now and then, so some cells are empty. The engineer trusts nearest neighbours because she can always look up which past readings a warning came from: only k-NN is allowed. Faults are rare, so the score is F1 on the faults.',
    task: 'classification',
    limits: { blocks: ['split', 'clean', 'impute', 'scale', 'knn', 'train', 'evaluate'], models: ['knn'], params: { knn: { k: [1, 31] } } },
    hints: ['Training stops at the first empty cell: fill the gaps before the model sees the rows.', 'Temperature runs in the fifties while vibration stays near two: without scaling, k-NN hardly sees the vibration. A small k keeps the rare faults from being outvoted.', 'Split → Impute (mean) → Scale (standard) → Train with 3-NN → Evaluate: F1 about 0.93 on the hidden readings. Unscaled, 5-NN reaches only 0.42.'],
    solution: pipe(pumps.data, 'knn', { k: 3 }, [['impute', {}], ['scale', { how: 'standard' }]])(),
    wrong: [pipe(pumps.data, 'knn', { k: 5 }, [['impute', {}]])(), pipe(pumps.data, 'knn', { k: 5 }, [['impute', {}], ['scale', { how: 'standard' }]])()] }, pumps));

  // ---- 2. exam results with a probabilistic model ---------------------------------------------------------------
  const exam = { data: { set: 'exam', n: 160, seed: 4 }, hidden: { seed: 44, n: 500 }, metric: 'accuracy', metricName: 'Accuracy', target: 0.83, marks: { silver: 0.832, gold: 0.834 } };
  C.challenge(Object.assign({ id: 'nn-ch-exam', n: 42, title: 'Who will pass the exam', level: 2,
    story: 'A volunteer tutoring club meets twice a week before the end-of-term exam. Each student who sat last term’s exam told the club three things: hours of study, hours of sleep the night before, and practice papers finished, and the club wrote down who passed. Next term it wants to see, from the same three answers, which students look likely to fail, so a tutor can sit with them first.\n\nThe club’s coordinator wants a model that explains itself as a picture of each group, so only the probabilistic classifiers are allowed: naive Bayes, LDA and QDA. Reach accuracy 0.83 on 500 students the model has never seen.',
    task: 'classification',
    limits: { blocks: ['split', 'naivebayes', 'lda', 'qda', 'train', 'evaluate'], models: ['naivebayes', 'lda', 'qda'] },
    hints: ['Try all three models and compare their boundaries in the Evaluate block.', 'Nothing says that students who pass and students who fail spread out the same way over the three answers. A model that gives each class its own covariance can follow different shapes; one that shares a covariance cannot.', 'Split → Train with QDA (no regularisation) → Evaluate: accuracy 0.834 on the hidden students. LDA reaches 0.82 and naive Bayes 0.808.'],
    solution: pipe(exam.data, 'qda', {})(), wrong: [pipe(exam.data, 'lda', {})(), pipe(exam.data, 'naivebayes', {})()] }, exam));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'nn-tour', title: 'Tour: nearest neighbours and naive Bayes', blurb: 'Voronoi cells, k against smoothness, the k-d tree, the curse of dimensionality and a spam filter, in eight steps.',
    steps: [
      { route: 'part=neighbours', target: '#main', text: 'Part 4 of the catalogue: models that remember their examples, and models that describe each class with a probability distribution.' },
      { route: 'topic=knn-voronoi', target: '#stageWrap', text: 'One nearest neighbour cuts the plane into Voronoi cells. The walls between cells of different classes are the decision boundary.' },
      { target: '#btnPlay', text: 'Play steps through the run; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=knn-k', target: '#inputs', text: 'Change the list of k values here. Small k draws islands around stray points; large k washes out the moons.' },
      { route: 'topic=kd-search', target: '#stageWrap', text: 'A k-d tree search: every box farther away than the best neighbour so far is skipped with all its points.' },
      { route: 'topic=curse-dim', target: '#stageWrap', text: 'In a thousand dimensions the nearest of 300 random points is almost as far as the farthest. Nearness fades.' },
      { route: 'topic=nb-spam', target: '#inputs', text: 'Type any message: each word adds its own log-odds of spam, and the order of the words never matters.' },
      { route: 'learn', target: '[data-chapter="nn-knn"]', text: 'The lessons and exercises of this part start here: votes by hand, distances under three metrics, a spam filter in code, and workflows that must reach a score on hidden rows.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'nn-course', title: 'Nearest neighbours and probabilistic models', blurb: 'All nine topics of part 4, from Voronoi cells to smoothing, then LDA and QDA, with lessons and exercises on the way.',
    steps: [
      { lesson: 'nn-voronoi', note: 'Training is storing; the work waits for the question.' },
      { topic: 'knn-voronoi', note: 'Cells, walls and the boundary.' },
      { exercise: 'nn-x-nearest-pick', note: 'Nearest by Manhattan distance.' },
      { topic: 'knn-k', note: 'Islands against smoothness.' },
      { exercise: 'nn-x-k-curve', note: 'Sketch the test error as k grows.' },
      { topic: 'knn-metrics', note: 'Circle, diamond and square.' },
      { exercise: 'nn-x-dist-matrix', note: 'Three distances by hand.' },
      { topic: 'knn-weights', note: 'Votes weighted by closeness.' },
      { exercise: 'nn-x-weighted', note: 'A weighted vote by hand.' },
      { exercise: 'nn-x-code-knn', note: 'Write k-NN.' },
      { exercise: 'nn-x-wf-loans', note: 'Scale before you measure.' },
      { topic: 'kd-search', note: 'Skipping whole boxes.' },
      { exercise: 'nn-x-kd-skip', note: 'Which boxes are out of reach.' },
      { topic: 'curse-dim', note: 'When everything is equally far.' },
      { lesson: 'nn-gnb', note: 'Bayes’ rule with independent features.' },
      { topic: 'nb-gauss', note: 'A bell per feature and class.' },
      { exercise: 'nn-x-gnb-posterior', note: 'A posterior by hand.' },
      { topic: 'nb-spam', note: 'A spam filter, word by word.' },
      { exercise: 'nn-x-nb-posterior', note: 'From counts to a probability.' },
      { topic: 'nb-smoothing', note: 'The word never seen.' },
      { exercise: 'nn-x-code-nb', note: 'Write the log-odds.' },
      { lesson: 'nn-lda-qda', note: 'A full Gaussian per class.' },
      { topic: 'lda-qda', note: 'One shared shape or one each.' },
      { exercise: 'nn-x-wf-flowers', note: 'Models that do not mind units.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson nn-voronoi')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'neighbours').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['nn-x-nearest-pick', 'nn-x-vote', 'nn-x-dist-matrix', 'nn-x-loo-matrix', 'nn-x-k-curve', 'nn-x-kd-split', 'nn-x-kd-skip', 'nn-x-nb-logodds', 'nn-x-code-nb', 'nn-x-lda-score', 'nn-x-wf-loans'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course nn-course', hash: 'course=nn-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    for (const id of ['nn-ch-pumps', 'nn-ch-exam']) S.Checks.push({ group: 'learn', name: 'nn challenge ' + id, hash: 'challenge/' + id, ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
