// Part 3, linear models: the model challenges, the guided tour, the course, and the pages the diagram checker looks at.
// Each challenge's data comes from S.Data with a fixed seed; the hidden rows are the same set with another seed. The
// targets were set by running the reference workflow and the wrong ones through the engines (the content checker runs
// them again every time).
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('li-ch-harvest')) return;
  const pipe = (data, model, mp, train, prep) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] };
    const wires = [['data.out', 'split.in'], ['m.model', 'tr.model'], ['tr.model', 'ev.model']];
    let a = 'split.train', b = 'split.test';
    (prep || []).forEach(([t, p], i) => { blocks['p' + i] = [t, p || {}]; wires.push([a, 'p' + i + '.in'], [b, 'p' + i + '.test']); a = 'p' + i + '.out'; b = 'p' + i + '.test'; });
    wires.push([a, 'tr.data'], [b, 'ev.data']);
    return WF.build({ blocks, wires });
  };

  // ---- 1. a ring in the lawn: logistic regression needs squared features ----------------------------------------
  const ring = { data: { set: 'circles', n: 200, seed: 6, opts: { noise: 0.15 } }, hidden: { seed: 66, n: 400 }, metric: 'accuracy', metricName: 'Accuracy', target: 0.91, marks: { silver: 0.925, gold: 0.935 } };
  C.challenge(Object.assign({ id: 'li-ch-ring', n: 31, title: 'The flower bed in the lawn', level: 1, task: 'classification',
    story: 'A town park has a round flower bed in the middle of a lawn, and a new robot mower that must never drive into it. While a gardener steered it around the park, the mower logged its position every few seconds as bed or lawn. The bed is a disc and the lawn a ring around it, and the edge is blurred where the gardener cut corners.\n\nThe park’s rules allow only a linear classifier, whose weights the gardener can read: logistic regression or the perceptron. You may give it polynomial features up to degree 2.',
    limits: { blocks: ['split', 'scale', 'poly', 'logistic', 'perceptron', 'train', 'evaluate'], models: ['logistic', 'perceptron'], params: { poly: { degree: [1, 2] } } },
    hints: ['Train logistic regression on the raw positions and open the Train block: the boundary is a straight line, and a straight line cannot fence in a disc.', 'A circle is x₁² + x₂² = r²: a straight line in the squared features. Put a Polynomial features block of degree 2 before the model, and wire the test rows through it too.', 'Polynomial features (degree 2) → Logistic regression with C = 10: accuracy about 0.935 on the hidden points.'],
    solution: pipe(ring.data, 'logistic', { C: 10 }, {}, [['poly', { degree: 2 }]]),
    wrong: [pipe(ring.data, 'logistic'), pipe(ring.data, 'perceptron', {}, {}, [['poly', { degree: 2 }]])] }, ring));

  // ---- 2. the valley harvest: ridge regression on curved, encoded features ----------------------------------------
  const harvest = { data: { set: 'harvest', n: 60, seed: 3 }, hidden: { seed: 91, n: 400 }, metric: 'rmse', metricName: 'RMSE', target: 0.355, marks: { silver: 0.345, gold: 0.34 } };
  C.challenge(Object.assign({ id: 'li-ch-harvest', n: 32, title: 'A forecast for the valley farms', level: 2, task: 'regression',
    story: 'A farming cooperative along a river valley wants to tell its members, before harvest, roughly how much wheat each plot will give. It has records of only sixty plots from past seasons: the rain, the soil, the fertiliser spread and the hours of sun, with the yield in tonnes per hectare. Too little rain is bad for wheat, and so is too much.\n\nThe cooperative’s agronomist wants a model she can read weight by weight, so only linear models are allowed: linear regression, ridge or the lasso, with any preparation you like and polynomial features up to degree 2. Sixty rows are few for that many weights.',
    limits: { blocks: ['split', 'encode', 'scale', 'poly', 'linreg', 'ridge', 'lasso', 'train', 'evaluate'], models: ['linreg', 'ridge', 'lasso'], params: { poly: { degree: [1, 2] } } },
    hints: ['A straight line in the rain cannot rise and then fall: open the Residuals of a plain linear regression and look at the rain.', 'Encode the soil, then let the model square the rain with Polynomial features of degree 2. That gives 27 weights for 45 training rows, and some columns repeat others (a one-hot column squared is itself). A penalty keeps the weights in check, but it is fair only if the columns share a scale: ridge without Scale stops at 0.360. Plain linear regression on the same columns does almost as well, 0.339, because with columns that repeat each other it takes the shortest weights that fit, a penalty of its own.', 'Encode → Scale (standard) → Polynomial features (degree 2) → Ridge with α = 1: RMSE about 0.338 t/ha on the hidden plots.'],
    solution: pipe(harvest.data, 'ridge', { alpha: 1 }, {}, [['encode', {}], ['scale', {}], ['poly', { degree: 2 }]]),
    wrong: [pipe(harvest.data, 'linreg'), pipe(harvest.data, 'ridge', { alpha: 1 }, {}, [['encode', {}], ['poly', { degree: 2 }]]), pipe(harvest.data, 'ridge', { alpha: 1 }, {}, [['encode', {}], ['scale', {}]])] }, harvest));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'li-tour', title: 'Tour: linear models', blurb: 'Least squares, gradient descent, ridge and lasso, and the linear classifiers, in eight steps.',
    steps: [
      { route: 'part=linear', target: '#main', text: 'Part 3 of the catalogue: linear models. Each card is a topic that runs on real numbers, from the least-squares line to the perceptron.' },
      { route: 'topic=ls-line', target: '#stageWrap', text: 'The least-squares line: every residual drawn as a square, and the line that makes their total area smallest.' },
      { target: '#btnPlay', text: 'Play steps through the run; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=gd-closed', target: '#inputs', text: 'Gradient descent on the same loss bowl. Change the learning rate here, or switch to scaled inputs, and watch the walk crawl, zig-zag or blow up.' },
      { route: 'topic=reg-geometry', target: '#stageWrap', text: 'Why the lasso gives exact zeros: the error’s rings touch the diamond at a corner, where one weight is 0. The ridge disc has no corners.' },
      { route: 'topic=logit-boundary', target: '#stageWrap', text: 'Logistic regression: the boundary turns and moves epoch by epoch, and the loss curve beside it falls.' },
      { route: 'lesson=li-sigmoid', target: '#lessonView', text: 'Lessons run the topics at chosen moments, ask for a prediction, and end with exercises: a gradient step by hand, a log loss, a step of code.' },
      { route: 'learn', target: '[data-chapter="li-ls"]', text: 'The lessons and exercises of this part start here. Two model challenges wait on the Challenges page: fence in a flower bed with a linear classifier, and forecast a harvest with ridge regression.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'li-course', title: 'Linear models', blurb: 'All twelve topics of part 3, from the least-squares line to LDA and QDA, with lessons and exercises on the way.',
    steps: [
      { topic: 'ls-line', note: 'Residuals as squares, and the line that makes them smallest.' },
      { lesson: 'li-line', note: 'The normal equations.' },
      { exercise: 'li-x-ls-normal', note: 'Solve them for four points.' },
      { topic: 'ls-projection', note: 'The same fit as a right angle in 3-D.' },
      { topic: 'gd-closed', note: 'Gradient descent on the loss bowl.' },
      { lesson: 'li-descent', note: 'Step size and scaling.' },
      { exercise: 'li-x-gd-step', note: 'One step by hand.' },
      { exercise: 'li-x-gd-code', note: 'The same step in code.' },
      { topic: 'poly-overfit', note: 'Degree against the test error.' },
      { exercise: 'li-x-poly-pick', note: 'Choose the degree.' },
      { topic: 'reg-paths', note: 'Ridge and lasso as λ slides.' },
      { topic: 'reg-geometry', note: 'Disc and diamond.' },
      { lesson: 'li-corners', note: 'Why the lasso gives zeros.' },
      { exercise: 'li-x-reg-softcode', note: 'The soft threshold.' },
      { topic: 'elastic-net', note: 'Twin features.' },
      { exercise: 'li-x-reg-wf', note: 'Tame a degree-10 polynomial on the canvas.' },
      { topic: 'logit-sigmoid', note: 'The sigmoid and log loss.' },
      { exercise: 'li-x-cl-grad', note: 'A logistic step by hand.' },
      { topic: 'logit-boundary', note: 'A boundary forming.' },
      { topic: 'softmax', note: 'Many classes.' },
      { exercise: 'li-x-cl-softmax', note: 'Three scores into probabilities.' },
      { topic: 'perceptron', note: 'Updates only on mistakes.' },
      { exercise: 'li-x-cl-perc', note: 'Two updates by hand.' },
      { topic: 'lda-qda', note: 'Gaussian classes: a straight boundary from shared shapes, a curved one from separate shapes.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson li-line')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'linear').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['li-x-ls-normal', 'li-x-gd-code', 'li-x-gd-curve', 'li-x-poly-pick', 'li-x-reg-path', 'li-x-reg-wf', 'li-x-cl-half', 'li-x-cl-perc', 'li-x-cl-choose'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course li-course', hash: 'course=li-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    for (const id of ['li-ch-ring', 'li-ch-harvest']) S.Checks.push({ group: 'learn', name: 'challenge ' + id, hash: 'challenge/' + id, ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
