// Part 10, convolutional networks: the model challenge, the guided tour, the course, and the pages the diagram checker
// looks at. The challenge's digits come from S.Data with a fixed seed, and its hidden digits from another seed of the
// same generator, so a score means the same on every visit. The target was set from the engines' own scores. The size
// limit (limits.weights) is one of the shared limits of S.Challenges; the canvas reads these digits at 28 × 28.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('cn-ch-digits')) return;
  const pipe = (data, model, mp, train) => WF.build({
    blocks: { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']],
  });

  // ---- 1. digits with a small network ------------------------------------------------------------------------------
  const dg = { data: { set: 'digits', n: 600, seed: 3 }, hidden: { seed: 33, n: 400 }, metric: 'accuracy', target: 0.97, marks: { silver: 0.98, gold: 0.99 } };
  C.challenge(Object.assign({ id: 'cn-ch-digits', n: 101, title: 'Five thousand numbers to read a digit', level: 3,
    story: 'A ferry company on a chain of small islands lets passengers write their cabin number by hand on a paper slip, and a reader at the gangway sorts the slips. The reader’s chip has room for a network of at most 5,000 weights and biases, and it learns in the harbour office in a short run of at most 20 epochs.\n\nThe usual small convolutional network has more than twice that. Shrink it until it fits, then make the few weights it keeps work harder. Reach an accuracy of 0.97 on 400 digits written by passengers the reader has never seen.',
    task: 'classification',
    limits: { blocks: ['split', 'cnn', 'train', 'evaluate'], models: ['cnn'], epochs: 20, maxBlocks: 5, weights: 5000, shape: { size: 28, classes: 10 } },
    hints: ['Start from the CNN block’s defaults: it is over the limit. Most of its weights are in the dense layer after the convolutions, not in the filters; the limits line under the canvas says how many it has.', 'Fewer filters in the second convolution make the list the dense layer reads shorter, and fewer dense neurons shrink it again: 4 and 8 filters with 16 dense neurons have 4,298 weights. Then train for all 20 epochs.', 'CNN with 4 and 8 filters of 5 × 5, 16 dense neurons and augmentation on; Train for 20 epochs at 0.005: accuracy 0.983, a silver mark. Without augmentation it gets 0.978.'],
    solution: pipe(dg.data, 'cnn', { f1: 4, f2: 8, kernel: 5, hidden: 16, augment: true }, { epochs: 20, lr: 0.005 }),
    wrong: [pipe(dg.data, 'cnn', {}, { epochs: 20, lr: 0.003 }), pipe(dg.data, 'cnn', { f1: 4, f2: 8, kernel: 5, hidden: 16 }, { epochs: 6, lr: 0.003 })] }, dg));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'cn-tour', title: 'Tour: convolutional networks', blurb: 'The sliding kernel, padding and stride, pooling, a trained network map by map, receptive fields, augmentation and Grad-CAM, in eight steps.',
    steps: [
      { route: 'part=cnn', target: '#main', text: 'Part 10 of the catalogue: convolutional networks. Every topic runs the engine’s own convolution and pooling, and the networks were trained here on generated digits and shapes.' },
      { route: 'topic=conv-slide', target: '#stageWrap', text: 'A kernel sliding over a picture: every place, every product, and the map they make.' },
      { target: '#btnPlay', text: 'Play steps through the story; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=conv-geometry', target: '#inputs', text: 'Padding, stride and dilation: set your own and see the output size and which pixels each place reads.' },
      { route: 'topic=cnn-flow', target: '#stageWrap', text: 'One digit flowing through a trained network, map by map, to its scores.' },
      { route: 'topic=cnn-augment', target: '#stageWrap', text: 'Forty training images, with and without augmentation: 78% against 97% on new digits.' },
      { route: 'topic=cnn-gradcam', target: '#stageWrap', text: 'Grad-CAM: where in the picture the network found its evidence.' },
      { route: 'learn', target: '[data-chapter="cn-conv"]', text: 'The lessons and exercises of this part start here: a convolution and a pooling by hand, output sizes, weight counts, receptive fields, convolution and pooling in code, and Grad-CAM on two maps.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'cn-course', title: 'Convolutional networks', blurb: 'All twelve topics of part 10, from a sliding kernel to Grad-CAM, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'cn-slide', note: 'A kernel sliding over a picture.' },
      { topic: 'conv-slide', note: 'Every product, every place.' },
      { exercise: 'cn-x-conv-cell', note: 'One output cell by hand.' },
      { topic: 'conv-geometry', note: 'Padding, stride and dilation.' },
      { exercise: 'cn-x-size-table', note: 'Output sizes.' },
      { topic: 'conv-pool', note: 'Pooling.' },
      { exercise: 'cn-x-pool-map', note: 'Max pooling by hand.' },
      { topic: 'conv-channels', note: 'One filter over many channels.' },
      { exercise: 'cn-x-params-conv', note: 'Weights of a convolution layer.' },
      { exercise: 'cn-x-code-conv', note: 'A convolution in code.' },
      { lesson: 'cn-flow', note: 'An image flowing through.' },
      { topic: 'cnn-flow', note: 'Feature maps, layer by layer.' },
      { topic: 'cnn-filters', note: 'Learned filters as pictures.' },
      { topic: 'cnn-field', note: 'How far a unit sees.' },
      { exercise: 'cn-x-field-table', note: 'Field and jump, layer by layer.' },
      { topic: 'cnn-train', note: 'Training, with the confusion matrix.' },
      { exercise: 'cn-x-cm-recall', note: 'Reading the matrix.' },
      { topic: 'cnn-augment', note: 'More images from the same ones.' },
      { exercise: 'cn-x-wf-few', note: 'Ninety digits are not many.' },
      { topic: 'cnn-residual', note: 'A shortcut for the gradient.' },
      { topic: 'cnn-saliency', note: 'Which pixels move the score.' },
      { topic: 'cnn-gradcam', note: 'Where the evidence lies.' },
      { exercise: 'cn-x-gradcam', note: 'Grad-CAM on two maps.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson cn-slide')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'cnn').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['cn-x-conv-cell', 'cn-x-conv-map', 'cn-x-conv-relu', 'cn-x-code-conv', 'cn-x-size-table', 'cn-x-pool-grad', 'cn-x-channels', 'cn-x-params-table', 'cn-x-field-table', 'cn-x-cm-recall', 'cn-x-wf-few', 'cn-x-gradcam'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course cn-course', hash: 'course=cn-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge cn-ch-digits', hash: 'challenge/cn-ch-digits', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
