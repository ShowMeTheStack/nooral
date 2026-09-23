// Part 5, trees and ensembles: the model challenges, the guided tour, the course, and the pages the diagram checker
// looks at. Each challenge's rows come from S.Data with a fixed seed, and its hidden rows from another seed of the
// same generator, so a score means the same on every visit. The targets were set from the engines' own scores.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('tr-ch-pumps')) return;
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // ---- 1. a pump-fault checklist: a tree no deeper than 4 -------------------------------------------------------
  const pumps = { data: { set: 'pumps', n: 800, seed: 4 }, hidden: { seed: 44, n: 1000 }, metric: 'f1', metricName: 'F1', target: 0.85, marks: { silver: 0.88, gold: 0.92 } };
  C.challenge(Object.assign({ id: 'tr-ch-pumps', n: 51, title: 'A fault checklist for the night shift', level: 2,
    story: 'The water works runs its pumps through the night with one engineer on duty. When a pump starts to fail, by overheating, shaking or losing pressure, the engineer has minutes to notice. The plant wants a checklist printed on a card by the control panel: at most four questions about the sensor readings, each a simple “is this reading above that value?”, ending in “normal” or “fault”.\n\nThat is a decision tree no deeper than 4. Faults are only one minute in twenty, so the plant scores the card by F1 on the faults, on a thousand minutes of readings it has kept back. Some readings were lost when a sensor dropped out, and a tree cannot split on a blank.',
    task: 'classification',
    limits: { blocks: ['split', 'clean', 'impute', 'tree', 'train', 'evaluate'], models: ['tree'], params: { tree: { maxDepth: [1, 4] } }, maxBlocks: 7 },
    hints: ['Run the plainest workflow first: the Train block will tell you what it cannot handle.', 'Fill the missing readings (Impute, on the training rows and the test rows), then a tree of depth 4 with Gini reaches about 0.86. For more, try the other impurity measure.', 'Split → Impute (mean) → Decision tree with entropy and max depth 4 → Train → Evaluate: F1 about 0.92 on the hidden minutes, a gold mark.'],
    solution: pipe(pumps.data, 'tree', { maxDepth: 4, criterion: 'entropy' }, [['impute', { strategy: 'mean' }]]),
    wrong: [pipe(pumps.data, 'tree', { maxDepth: 4 }), pipe(pumps.data, 'tree', { maxDepth: 2 }, [['impute', { strategy: 'mean' }]])] }, pumps));

  // ---- 2. a harvest forecast: ensembles of trees ------------------------------------------------------------------
  const harvest = { data: { set: 'harvest', n: 300, seed: 7 }, hidden: { seed: 71, n: 500 }, metric: 'rmse', metricName: 'RMSE', target: 0.38, marks: { silver: 0.37, gold: 0.365 } };
  C.challenge(Object.assign({ id: 'tr-ch-harvest', n: 52, title: 'Forecast the valley harvest', level: 3,
    story: 'A co-operative of wheat farmers along a river valley pools its grain and sells it ahead of the harvest. To sell well it needs each plot’s yield in tonnes per hectare from what is known in late summer: the season’s rain, the soil, the fertiliser spread and the hours of sun. Rain helps up to a point and then floods the roots, and each soil starts from its own level, so the shape is not a straight line.\n\nOnly tree models are on the table. Bring the root mean squared error on the co-operative’s held-back plots down to 0.38 tonnes per hectare.',
    task: 'regression',
    limits: { blocks: ['split', 'encode', 'scale', 'tree', 'forest', 'gboost', 'train', 'evaluate'], models: ['tree', 'forest', 'gboost'], params: { gboost: { nEstimators: [1, 120] }, forest: { nEstimators: [1, 100] } }, maxBlocks: 8 },
    hints: ['A single tree draws coarse steps over rain: look at its residuals. Turn the soil into numbers with Encode first.', 'A forest averages deep trees and gets to about 0.41; gradient boosting adds small trees one by one, and a full learning rate of 1 overshoots.', 'Encode → Gradient boosting with 100 stages, rate 0.1 and depth 2 → Train → Evaluate: RMSE about 0.361 on the hidden plots.'],
    solution: pipe(harvest.data, 'gboost', { nEstimators: 100, learningRate: 0.1, maxDepth: 2 }, [['encode', {}]]),
    wrong: [pipe(harvest.data, 'forest', {}, [['encode', {}]]), pipe(harvest.data, 'gboost', { nEstimators: 60, learningRate: 1, maxDepth: 3 }, [['encode', {}]])] }, harvest));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'tr-tour', title: 'Tour: trees and ensembles', blurb: 'A tree growing split by split, pruning, forests and boosting, in eight steps.',
    steps: [
      { route: 'part=trees', target: '#main', text: 'Part 5 of the catalogue: trees and ensembles. Each card is a topic that runs the engine’s own trees, forests and boosters on real values.' },
      { route: 'topic=cart-grow', target: '#stageWrap', text: 'A tree growing split by split: at every node every threshold of every feature is scored by its weighted Gini, and the lowest wins.' },
      { target: '#btnPlay', text: 'Play steps through the growth; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=ccp-prune', target: '#stageWrap', text: 'Pruning grows the whole tree and cuts back the weakest link first: the subtree that saves the least impurity per leaf.' },
      { route: 'topic=rf-oob', target: '#inputs', text: 'A random forest votes on the rows each tree never saw: change the number of trees here and watch the out-of-bag score settle.' },
      { route: 'topic=adaboost', target: '#stageWrap', text: 'AdaBoost: the dots grow when a stump gets them wrong, so the next stump aims at them.' },
      { route: 'lesson=tr-gboost', target: '#lessonView', text: 'Lessons run the topics at the frames that matter, ask for a prediction, and end with exercises: here, gradient boosting fitting its residuals.' },
      { route: 'learn', target: '[data-chapter="tr-grow"]', text: 'The lessons and exercises of this part start here: compute a Gini by hand, find the best threshold, reweight AdaBoost’s points, and build workflows that must reach a score on hidden rows.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'tr-course', title: 'Trees and ensembles', blurb: 'All twelve topics of part 5, from one split to five ensembles side by side, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'tr-impurity', note: 'Gini and entropy: how mixed a node is.' },
      { topic: 'cart-grow', note: 'Every threshold weighed, then the best cut.' },
      { exercise: 'tr-x-candidates', note: 'Score the candidates yourself.' },
      { topic: 'reg-tree', note: 'A mean in every leaf.' },
      { exercise: 'tr-x-sse-split', note: 'A regression split by squared error.' },
      { topic: 'tree-size', note: 'Depth against fresh data.' },
      { topic: 'ccp-prune', note: 'The weakest link goes first.' },
      { exercise: 'tr-x-ccp-g', note: 'The price of a subtree.' },
      { lesson: 'tr-bagging', note: 'Why a vote of deep trees helps.' },
      { topic: 'bagging', note: 'Bootstrap samples and their vote.' },
      { topic: 'rf-features', note: 'A few features per split.' },
      { topic: 'rf-oob', note: 'A free test set.' },
      { exercise: 'tr-x-oob-vote', note: 'An out-of-bag score by hand.' },
      { topic: 'rf-importance', note: 'Two importances that disagree.' },
      { topic: 'adaboost', note: 'Weights grow on mistakes.' },
      { exercise: 'tr-x-ada-weights', note: 'One reweighting by hand.' },
      { topic: 'gb-residuals', note: 'Each tree fits what is left.' },
      { topic: 'gb-shrinkage', note: 'Small steps generalise.' },
      { exercise: 'tr-x-gb-stage', note: 'One stage by hand.' },
      { topic: 'ensemble-compare', note: 'Five tree models on the same points.' },
      { lesson: 'tr-compare', note: 'Choosing among them.' },
      { exercise: 'tr-x-wf-circles', note: 'A workflow scored on hidden points.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson tr-impurity')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'trees').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['tr-x-split-table', 'tr-x-best-threshold', 'tr-x-stump-matrix', 'tr-x-code-split', 'tr-x-curve-depth', 'tr-x-bootstrap-set', 'tr-x-oob-vote', 'tr-x-gb-stage', 'tr-x-wf-circles', 'tr-x-overfit-choice'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course tr-course', hash: 'course=tr-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    for (const id of ['tr-ch-pumps', 'tr-ch-harvest']) S.Checks.push({ group: 'learn', name: 'challenge ' + id, hash: 'challenge/' + id, ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
