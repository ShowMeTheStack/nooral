// Part 6, support vector machines: the model challenges, the guided tour, the course, and the pages the diagram
// checker looks at. Each challenge's rows come from S.Data with a fixed seed, and its hidden rows from another seed of
// the same generator, so a score means the same on every visit. The targets were set from the engines' own scores.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('sv-ch-plots')) return;
  const pipe = (data, model, mp, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // ---- 1. two crops on a chequered field: an RBF SVM with C and γ held in bounds -----------------------------------
  const plots = { data: { set: 'checkerboard', n: 400, seed: 3 }, hidden: { seed: 33, n: 800 }, metric: 'accuracy', target: 0.9, marks: { silver: 0.91, gold: 0.915 } };
  C.challenge(Object.assign({ id: 'sv-ch-plots', n: 61, title: 'The chequered trial field', level: 2,
    story: 'An agricultural college runs a trial field cut into sixteen square plots, planted alternately with beans and barley like a chessboard. A survey drone flies over it and logs its position for every photograph, but the crop labels of the latest flight were lost.\n\nThe college wants a model that names the crop from the drone’s two map coordinates alone. Its statistics tutor allows a support vector machine and nothing else, and insists on modest settings: C no larger than 10 and γ between 0.1 and 10, so that the model can be explained in one lecture. The score is the accuracy on 800 photographs kept back.',
    task: 'classification',
    limits: { blocks: ['split', 'scale', 'svm', 'train', 'evaluate'], models: ['svm'], params: { svm: { C: [0.001, 10], gamma: [0.1, 10] } }, maxBlocks: 6 },
    hints: ['Start with the RBF kernel and look at the boundary in the Train block: with the automatic γ the bumps are far too wide for sixteen small squares.', 'Each square is one unit wide, so the bumps should reach about a unit or less: γ of 1 or more. Then raise C until the corners of the squares are drawn sharply.', 'Split → SVM (RBF, C = 3, γ = 2) → Train → Evaluate: accuracy 0.916 on the hidden photographs, a gold mark.'],
    solution: pipe(plots.data, 'svm', { kernel: 'rbf', C: 3, gamma: 2 }),
    wrong: [pipe(plots.data, 'svm', { kernel: 'rbf', C: 1 }), pipe(plots.data, 'svm', { kernel: 'rbf', C: 1, gamma: 10 })] }, plots));

  // ---- 2. the valley harvest by support vector regression ------------------------------------------------------------
  const harvest = { data: { set: 'harvest', n: 300, seed: 7 }, hidden: { seed: 71, n: 500 }, metric: 'rmse', target: 0.37, marks: { silver: 0.36, gold: 0.356 } };
  C.challenge(Object.assign({ id: 'sv-ch-harvest', n: 62, title: 'A smooth forecast for the insurer', level: 3,
    story: 'The valley’s crop insurer prices its policies from a forecast of each plot’s wheat yield: the season’s rain, the soil, the fertiliser and the hours of sun. Its actuaries distrust forecasts that jump in steps, as trees do, and ask for a smooth curve.\n\nThey settle on support vector regression. Errors smaller than a tenth of a tonne do not matter to a premium, and a few freak plots should not drag the curve about. Bring the root mean squared error on the insurer’s 500 held-back plots down to 0.37 tonnes per hectare.',
    task: 'regression',
    limits: { blocks: ['split', 'encode', 'scale', 'svr', 'train', 'evaluate'], models: ['svr'], params: { svr: { C: [0.01, 100], epsilon: [0, 1] } }, maxBlocks: 7 },
    hints: ['The soil is a category: Encode it before the SVR. Then look at the residuals of the plain SVR.', 'The RBF kernel measures distances, and rain is in hundreds of millimetres while the soil code is 0 or 1. Scale the features, then try wider bumps (a smaller γ).', 'Split → Encode → Scale (standard) → SVR (RBF, C = 10, ε = 0.1, γ = 0.05) → Train → Evaluate: RMSE 0.354 on the hidden plots, a gold mark.'],
    solution: pipe(harvest.data, 'svr', { kernel: 'rbf', C: 10, epsilon: 0.1, gamma: 0.05 }, [['encode', {}], ['scale', { how: 'standard' }]]),
    wrong: [pipe(harvest.data, 'svr', {}, [['encode', {}]]), pipe(harvest.data, 'svr', { C: 100, epsilon: 0.1 }, [['encode', {}], ['scale', { how: 'standard' }]])] }, harvest));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'sv-tour', title: 'Tour: support vector machines', blurb: 'The widest street, the price C, kernels, the solver and its multipliers, in eight steps.',
    steps: [
      { route: 'part=svm', target: '#main', text: 'Part 6 of the catalogue: support vector machines. Each card is a topic that runs the engine’s own SMO solver on real values.' },
      { route: 'topic=svm-max-margin', target: '#stageWrap', text: 'Many lines split two groups; the SVM keeps the one with the widest empty street. The points on its edges, ringed, are the support vectors.' },
      { target: '#btnPlay', text: 'Play steps through the story; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=svm-soft-c', target: '#inputs', text: 'When classes overlap, C is the price of letting points into the street. Change it here and watch the street narrow or widen.' },
      { route: 'topic=svm-kernel-lift', target: '#stageWrap', text: 'A ring round a disc, lifted into 3-D: a flat plane separates the layers, and meets the map in a circle. That is what a kernel does without lifting.' },
      { route: 'topic=svm-smo', target: '#stageWrap', text: 'SMO solves for the multipliers α two at a time: the pair that breaks the conditions most moves first.' },
      { route: 'lesson=sv-rbf', target: '#lessonView', text: 'Lessons run the topics at the frames that matter, ask for a prediction, and end with exercises: here, γ against overfitting.' },
      { route: 'learn', target: '[data-chapter="sv-margin"]', text: 'The lessons and exercises of this part start here: a margin from w, the hinge loss, kernel values by hand, the RBF kernel in code, and workflows scored on hidden rows.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'sv-course', title: 'Support vector machines', blurb: 'All eleven topics of part 6, from the widest street to the one-class border, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'sv-street', note: 'Which line would you trust?' },
      { topic: 'svm-max-margin', note: 'The widest street and its support vectors.' },
      { exercise: 'sv-x-width', note: 'A width from w.' },
      { topic: 'svm-soft-c', note: 'The price C of slack.' },
      { topic: 'svm-hinge', note: 'The hinge loss.' },
      { exercise: 'sv-x-hinge-table', note: 'The hinge of four points.' },
      { lesson: 'sv-lift', note: 'A straight cut in a higher space.' },
      { topic: 'svm-kernel-lift', note: 'The lift into 3-D.' },
      { topic: 'svm-poly', note: 'The polynomial kernel.' },
      { topic: 'svm-rbf', note: 'The RBF kernel and γ.' },
      { exercise: 'sv-x-rbf-value', note: 'A kernel value by hand.' },
      { exercise: 'sv-x-code-rbf', note: 'The RBF kernel in code.' },
      { topic: 'svm-dual', note: 'w as a sum over support vectors.' },
      { exercise: 'sv-x-w-from-alpha', note: 'Build w from the multipliers.' },
      { topic: 'svm-smo', note: 'SMO, two at a time.' },
      { topic: 'svm-multiclass', note: 'One-vs-rest and one-vs-one.' },
      { exercise: 'sv-x-ovo-vote', note: 'Count the votes.' },
      { topic: 'svm-svr', note: 'The ε-tube.' },
      { topic: 'svm-one-class', note: 'A border round normal data.' },
      { lesson: 'sv-oneclass', note: 'What ν promises.' },
      { exercise: 'sv-x-wf-flowers', note: 'A workflow scored on hidden flowers.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson sv-street')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'svm').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['sv-x-sv-set', 'sv-x-hinge-table', 'sv-x-c-curve', 'sv-x-kernel-table', 'sv-x-ring-choice', 'sv-x-gamma-curve', 'sv-x-code-rbf', 'sv-x-wf-ring', 'sv-x-ovo-vote', 'sv-x-w-from-alpha'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course sv-course', hash: 'course=sv-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    for (const id of ['sv-ch-plots', 'sv-ch-harvest']) S.Checks.push({ group: 'learn', name: 'challenge ' + id, hash: 'challenge/' + id, ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
