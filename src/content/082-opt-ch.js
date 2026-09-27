// Part 8, optimisation: the model challenge, the guided tour, the course, and the pages the diagram checker looks at.
// The challenge's rows come from S.Data with a fixed seed, and its hidden rows from another seed of the same generator,
// so a score means the same on every visit. The target was set from the engines' own scores.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('op-ch-moons')) return;
  const pipe = (data, model, mp, train) => WF.build({
    blocks: { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']],
  });

  // ---- 1. a budget of twenty epochs: choose the optimiser and its rate -----------------------------------------------
  const moons = { data: { set: 'moons', n: 300, seed: 2 }, hidden: { seed: 22, n: 600 }, metric: 'accuracy', target: 0.97, marks: { silver: 0.99, gold: 0.995 } };
  C.challenge(Object.assign({ id: 'op-ch-moons', n: 81, title: 'Twenty epochs on the shared machine', level: 2,
    story: 'A school’s computing club shares one old machine, booked in short slots. Each member’s network may train for at most twenty epochs before the next booking. This week’s task: two interlocking crescents of points, the moons, to be told apart by a small network of two hidden layers.\n\nThe network is fixed; what is yours to choose is how it learns: the optimiser and its learning rate. With the defaults the run ends long before the network has found the shape. Reach an accuracy of 0.97 on 600 points the club keeps back.',
    task: 'classification',
    limits: { blocks: ['split', 'mlp', 'train', 'evaluate'], models: ['mlp'], epochs: 20, maxBlocks: 5 },
    hints: ['Train with the defaults and open the Train block: the loss is still falling at the last epoch. The run is too slow, not the network too small.', 'Try each optimiser at a few rates, a factor of three apart. Plain SGD needs a much larger rate than Adam; too large a rate and the loss jumps about.', 'Split → MLP (8, 8, tanh) → Train (Adam, rate 0.1, 20 epochs) → Evaluate: accuracy 0.997 on the hidden points, a gold mark. RMSProp at 0.03 and momentum at 0.3 do as well.'],
    solution: pipe(moons.data, 'mlp', { hidden: [8, 8], act: 'tanh' }, { epochs: 20, optimizer: 'adam', lr: 0.1 }),
    wrong: [pipe(moons.data, 'mlp', { hidden: [8, 8], act: 'tanh' }, { epochs: 20, optimizer: 'adam', lr: 0.01 }), pipe(moons.data, 'mlp', { hidden: [8, 8], act: 'tanh' }, { epochs: 20, optimizer: 'sgd', lr: 0.3 })] }, moons));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'op-tour', title: 'Tour: optimisation', blurb: 'Loss surfaces, gradient descent, momentum and Adam, schedules and local minima, in eight steps.',
    steps: [
      { route: 'part=optimisation', target: '#main', text: 'Part 8 of the catalogue: optimisation. Each topic runs the engine’s own optimisers, the ones that train every network here, on surfaces you can see whole.' },
      { route: 'topic=opt-surface', target: '#stageWrap', text: 'A loss as a landscape over two parameters, from the side and from above, with the downhill arrows.' },
      { target: '#btnPlay', text: 'Play steps through the story; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=opt-lr', target: '#inputs', text: 'The learning rate: set your own here and see whether it creeps, races or explodes against the limit 2/λmax.' },
      { route: 'topic=opt-race', target: '#stageWrap', text: 'Six optimisers, each with its best rate, raced on the same surface. Try the saddle and the plateau too.' },
      { route: 'topic=opt-local', target: '#stageWrap', text: 'On a bumpy loss the start decides the pit: every pit has a basin of starts that drain into it.' },
      { route: 'lesson=op-adaptive', target: '#lessonView', text: 'Lessons run the topics at the frames that matter, ask for a prediction, and end with exercises: here, a step size for each coordinate.' },
      { route: 'learn', target: '[data-chapter="op-descent"]', text: 'The lessons and exercises of this part start here: a descent step by hand, momentum and Adam steps, schedule values, Adam in code, and a network to train in fifteen epochs.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'op-course', title: 'Optimisation', blurb: 'All twelve topics of part 8, from a loss surface to plateaus, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'op-surface', note: 'A loss is a landscape.' },
      { topic: 'opt-surface', note: 'Surfaces from the side and from above.' },
      { topic: 'opt-gd', note: 'One step at a time.' },
      { exercise: 'op-x-gd-step', note: 'A step by hand.' },
      { topic: 'opt-lr', note: 'Too small, just right, too large.' },
      { exercise: 'op-x-lr-limit', note: 'The largest safe step.' },
      { topic: 'opt-sgd', note: 'Noisy steps that cost less.' },
      { lesson: 'op-momentum', note: 'Remembering the way.' },
      { topic: 'opt-momentum', note: 'Momentum and Nesterov.' },
      { exercise: 'op-x-momentum-table', note: 'Two momentum steps.' },
      { topic: 'opt-adaptive', note: 'AdaGrad, RMSProp and Adam.' },
      { exercise: 'op-x-adam-step', note: 'Adam’s first step.' },
      { exercise: 'op-x-code-adam', note: 'Adam in code.' },
      { topic: 'opt-race', note: 'Six optimisers raced.' },
      { topic: 'opt-schedules', note: 'Lowering the rate.' },
      { exercise: 'op-x-cosine', note: 'A cosine rate.' },
      { topic: 'opt-warmup', note: 'Starting small.' },
      { topic: 'opt-convex', note: 'The chord test.' },
      { topic: 'opt-local', note: 'Basins of local minima.' },
      { topic: 'opt-plateau', note: 'Plateaus.' },
      { lesson: 'op-local', note: 'Where the walk ends.' },
      { exercise: 'op-x-wf-momentum', note: 'A network in fifteen epochs.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson op-surface')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'optimisation').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['op-x-grad', 'op-x-gd-step', 'op-x-gd-curve', 'op-x-code-gd', 'op-x-momentum-table', 'op-x-adam-step', 'op-x-adam-two', 'op-x-code-adam', 'op-x-sched-curve', 'op-x-convex-set', 'op-x-wf-momentum'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course op-course', hash: 'course=op-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge op-ch-moons', hash: 'challenge/op-ch-moons', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
