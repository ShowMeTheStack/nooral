// Part 9, neural networks: the model challenge, the guided tour, the course, and the pages the diagram checker looks
// at. The challenge's rows come from S.Data with a fixed seed, and its hidden rows from another seed of the same
// generator, so a score means the same on every visit. The target was set from the engines' own scores.
//
// The challenges of parts 9 and 10 limit the size of a network with the shared limits of S.Challenges:
// limits.neurons (hidden neurons in all) and limits.weights with limits.shape (weights and biases in a network).
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('ne-ch-spiral')) return;

  const pipe = (data, model, mp, train) => WF.build({
    blocks: { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']],
  });

  // ---- 1. two spiral arms with at most twelve neurons ---------------------------------------------------------------
  const sp = { data: { set: 'spirals', n: 300, seed: 6, opts: { noise: 0.05 } }, hidden: { seed: 66, n: 400 }, metric: 'accuracy', target: 0.93, marks: { silver: 0.95, gold: 0.97 } };
  C.challenge(Object.assign({ id: 'ne-ch-spiral', n: 91, title: 'Twelve neurons for two spirals', level: 3,
    story: 'A garden-lamp maker wants its lamps to tell the two arms of a spiral path apart, so that each arm can glow its own colour as a visitor walks it. The chip inside each lamp has room for a network of twelve hidden neurons, no more, and the lamps learn on the bench in a short run.\n\nTwo arms winding round the centre are a hard shape for so few neurons. Spend the twelve well: how many layers, how many neurons in each, which activation. Reach an accuracy of 0.93 on 400 points of the path that the lamps never trained on.',
    task: 'classification',
    limits: { blocks: ['split', 'mlp', 'train', 'evaluate'], models: ['mlp'], epochs: 200, neurons: 12, maxBlocks: 5 },
    hints: ['Try the playground first: two spirals with one layer of twelve neurons, then with two layers of six. Watch each neuron’s square.', 'One hidden layer can only add up regions with straight edges; a second layer makes curved shapes out of them. Twelve neurons go further as two layers of six than as one of twelve.', 'MLP with hidden layers 6, 6 and tanh; Train for 200 epochs with Adam at 0.03: accuracy 0.953 on the hidden points, a silver mark. One layer of 12 gets 0.748.'],
    solution: pipe(sp.data, 'mlp', { hidden: [6, 6], act: 'tanh' }, { epochs: 200, lr: 0.03 }),
    wrong: [pipe(sp.data, 'mlp', { hidden: [12], act: 'tanh' }, { epochs: 200, lr: 0.03 }), pipe(sp.data, 'mlp', { hidden: [16, 16], act: 'relu' }, { epochs: 200, lr: 0.02 })] }, sp));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'ne-tour', title: 'Tour: neural networks', blurb: 'One neuron, activations, the forward and backward passes, initialisation, regularisation and the playground, in eight steps.',
    steps: [
      { route: 'part=networks', target: '#main', text: 'Part 9 of the catalogue: neural networks. Every topic runs the engine’s own layers and autodiff, the ones that train every network here.' },
      { route: 'topic=p09-neuron', target: '#stageWrap', text: 'One neuron: the weighted sum, the activation and its slope, and the line it draws across the plane.' },
      { target: '#btnPlay', text: 'Play steps through the story; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=p09-backprop', target: '#stageWrap', text: 'Backpropagation on the computational graph: every value forward, every gradient backward, then checked by finite differences.' },
      { route: 'topic=p09-init', target: '#inputs', text: 'Initialisation: pick the rule and the activation here and watch the signal fade, saturate or keep its size through six layers.' },
      { route: 'topic=p09-overfit', target: '#stageWrap', text: 'A large network memorising thirty points; the next topics are the cures: dropout, weight decay, early stopping, normalisation.' },
      { route: 'playground', target: '#pgView', text: 'The playground: add layers and neurons, train, and see every neuron’s region over the plane.' },
      { route: 'learn', target: '[data-chapter="ne-units"]', text: 'The lessons and exercises of this part start here: a neuron and a gradient by hand, backpropagation through a tiny graph, a dense layer’s backward pass in code, and a network to rescue from memorising.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'ne-course', title: 'Neural networks', blurb: 'All twelve topics of part 9, from one neuron to the playground, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'ne-neuron', note: 'A weighted sum and a squash.' },
      { topic: 'p09-neuron', note: 'One neuron.' },
      { exercise: 'ne-x-neuron-out', note: 'A neuron’s output by hand.' },
      { topic: 'p09-activations', note: 'Activations and their slopes.' },
      { exercise: 'ne-x-sig-slope', note: 'A slope by hand.' },
      { topic: 'p09-forward', note: 'A value on every wire.' },
      { exercise: 'ne-x-forward', note: 'A forward pass by hand.' },
      { lesson: 'ne-backprop', note: 'Gradients on the graph.' },
      { topic: 'p09-backprop', note: 'Backpropagation, node by node.' },
      { exercise: 'ne-x-bp-graph', note: 'Every gradient of a tiny graph.' },
      { exercise: 'ne-x-code-dense', note: 'A dense layer’s backward pass in code.' },
      { topic: 'p09-loss-surface', note: 'Twins and a saddle.' },
      { topic: 'p09-init', note: 'Why the first weights matter.' },
      { exercise: 'ne-x-init-choice', note: 'A rule for ReLU.' },
      { topic: 'p09-overfit', note: 'Memorising thirty points.' },
      { topic: 'p09-dropout', note: 'Neurons switched off.' },
      { exercise: 'ne-x-dropout-test', note: 'Dropout at test time.' },
      { topic: 'p09-weight-decay', note: 'Keeping the weights small.' },
      { topic: 'p09-early-stop', note: 'Quit while ahead.' },
      { exercise: 'ne-x-early', note: 'When it stops.' },
      { topic: 'p09-norm', note: 'Batch and layer normalisation.' },
      { exercise: 'ne-x-norm-choice', note: 'Normalising tiny batches.' },
      { topic: 'p09-playground', note: 'Every neuron’s region.' },
      { exercise: 'ne-x-wf-overfit', note: 'Rescue a memorising network.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson ne-neuron')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'networks').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['ne-x-neuron-relu', 'ne-x-slopes-table', 'ne-x-code-act', 'ne-x-forward', 'ne-x-bp-graph', 'ne-x-code-dense', 'ne-x-init-curve', 'ne-x-overfit-curve', 'ne-x-early', 'ne-x-wf-overfit', 'ne-x-bn'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course ne-course', hash: 'course=ne-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge ne-ch-spiral', hash: 'challenge/ne-ch-spiral', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
