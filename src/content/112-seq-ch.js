// Part 11, sequences and embeddings: the model challenge, the guided tour, the course, and the pages the diagram
// checker looks at. The challenge trains on 300 of the reviews with a turn (S.Data 'opinions', a fixed pool) and is
// scored on the 180 reviews the pool holds out, so a score means the same on every visit. The target and the marks
// were set from the engines' own scores of the workflows below.
(function (S) {
  'use strict';
  const C = S.Content, WF = S.WF;
  if (!C || !WF || C.challenges.has('sq-ch-turns')) return;
  const pipe = (data, model, mp, train, vec) => {
    const blocks = { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] };
    const wires = [['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']];
    if (vec) { blocks.vec = ['vectorise', vec]; wires.unshift(['data.out', 'vec.in'], ['vec.out', 'split.in']); } else wires.unshift(['data.out', 'split.in']);
    return WF.build({ blocks, wires });
  };

  // ---- 1. reviews with a turn ----------------------------------------------------------------------------------
  // the pool holds out 180 of its 600 reviews; hidden.n names them all (n: 0 would score the same 180).
  // Scores on them: word counts + logistic 0.694, simple RNN 0.75, GRU at 0.01 0.811, GRU at 0.02 0.828
  const tw = { data: { set: 'opinions', n: 300, seed: 5 }, hidden: { seed: 55, n: 180 }, metric: 'accuracy', target: 0.8, marks: { silver: 0.82, gold: 0.84 } };
  C.challenge(Object.assign({ id: 'sq-ch-turns', n: 111, title: 'The verdict after the “but”', level: 3,
    story: 'A small repair café asks visitors to leave a line about their visit, and the volunteers want the unhappy ones flagged so that someone calls back. The lines are full of turns: “friendly people, but my lamp still flickers”, “a long wait, but they saved my radio”. A filter that counts kind and unkind words gets about seven in ten right, and the volunteers want better.\n\nYou have 300 labelled lines. Build a model that reaches an accuracy of 0.8 on 180 lines it has never seen. Word counts are allowed, but they cannot see where the turn falls.',
    task: 'classification',
    limits: { blocks: ['split', 'vectorise', 'logistic', 'naivebayes', 'rnn', 'train', 'evaluate'], models: ['logistic', 'naivebayes', 'rnn'], epochs: 30, maxBlocks: 6 },
    hints: ['Start with Vectorise text and logistic regression: it reaches 0.694. Open the evaluate block and read some of the mistakes: most have a “but” in them.', 'A Recurrent network block reads the words in order. A simple RNN reaches 0.75 here; the GRU cell, with a gate to hold what came before the turn, does better.', 'Recurrent network with the GRU cell, Train for 20 epochs at a learning rate of 0.02: accuracy 0.828, a silver mark. At 0.01 it gets 0.811.'],
    solution: pipe(tw.data, 'rnn', { cell: 'GRU' }, { epochs: 20, lr: 0.02 }),
    wrong: [pipe(tw.data, 'logistic', {}, {}, { weight: 'count' }), pipe(tw.data, 'rnn', { cell: 'RNN' }, { epochs: 10 })] }, tw));

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'sq-tour', title: 'Tour: sequences and embeddings', blurb: 'One-hot vectors, word2vec and its map, a recurrent network unrolled, fading gradients, LSTM and GRU gates, a character model and an encoder–decoder, in eight steps.',
    steps: [
      { route: 'part=sequences', target: '#main', text: 'Part 11 of the catalogue: sequences and embeddings. word2vec is trained here on sentences of our own, and the recurrent networks were trained on bit streams, letters and number words.' },
      { route: 'topic=word2vec-train', target: '#stageWrap', text: 'word2vec learning from neighbouring words: the map of words, epoch by epoch, as the kinds pull apart.' },
      { target: '#btnPlay', text: 'Play steps through the story; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=word2vec-map', target: '#inputs', text: 'Pick an analogy and a word: its nearest neighbours, and the parallelogram of king, queen, man and woman.' },
      { route: 'topic=rnn-unrolled', target: '#stageWrap', text: 'A recurrent cell unrolled: the same weights at every step, and the hidden state carrying the count along.' },
      { route: 'topic=rnn-gradients', target: '#stageWrap', text: 'The gradient going back through time, shrinking or growing by the same factor every step.' },
      { route: 'topic=lstm-gates', target: '#stageWrap', text: 'An LSTM’s gates opening and closing step by step, and the memory slot that keeps the count.' },
      { route: 'learn', target: '[data-chapter="sq-embed"]', text: 'The lessons and exercises of this part start here: one-hot and embedding lookups, word2vec’s loss, recurrent steps and gradients by hand, LSTM and GRU cells, perplexity, temperature, and a recurrent model on the canvas.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'sq-course', title: 'Sequences and embeddings', blurb: 'All nine topics of part 11, from one-hot vectors to an encoder–decoder, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'sq-onehot', note: 'One-hot vectors and embedding tables.' },
      { topic: 'one-hot-embeddings', note: 'A lookup is a product.' },
      { exercise: 'sq-x-lookup', note: 'Two words through the table.' },
      { topic: 'word2vec-train', note: 'Learning from neighbouring words.' },
      { exercise: 'sq-x-sgns-loss', note: 'The loss of one pair.' },
      { topic: 'word2vec-map', note: 'Neighbours and analogies.' },
      { exercise: 'sq-x-analogy', note: 'Reading an analogy.' },
      { lesson: 'sq-unroll', note: 'A recurrent network, unrolled.' },
      { topic: 'rnn-unrolled', note: 'One cell at every step.' },
      { exercise: 'sq-x-rnn-step', note: 'One step by hand.' },
      { topic: 'rnn-gradients', note: 'Fading and exploding gradients.' },
      { exercise: 'sq-x-tanh-factor', note: 'The gradient through three steps.' },
      { topic: 'lstm-gates', note: 'LSTM gates, step by step.' },
      { exercise: 'sq-x-lstm-cell', note: 'One LSTM step by hand.' },
      { topic: 'gru-gates', note: 'GRU: keep or rewrite.' },
      { exercise: 'sq-x-gru-update', note: 'A GRU update.' },
      { topic: 'next-char', note: 'Predicting the next character.' },
      { exercise: 'sq-x-perplexity', note: 'Loss and perplexity.' },
      { exercise: 'sq-x-temp', note: 'Two temperatures.' },
      { topic: 'seq2seq', note: 'Encode, then decode.' },
      { exercise: 'sq-x-s2s-curve', note: 'Longer and longer inputs.' },
      { lesson: 'sq-order', note: 'When word order decides.' },
      { exercise: 'sq-x-wf-maxlen', note: 'Read to the end.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson sq-onehot')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'sequences').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['sq-x-lookup', 'sq-x-pairs', 'sq-x-w2v-curve', 'sq-x-analogy', 'sq-x-rnn-step', 'sq-x-code-rnn', 'sq-x-grad-curve', 'sq-x-tanh-factor', 'sq-x-lstm-cell', 'sq-x-code-gru', 'sq-x-perplexity', 'sq-x-temp', 'sq-x-s2s-curve', 'sq-x-wf-maxlen'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course sq-course', hash: 'course=sq-course', ready: "document.querySelector('#courseView')", sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'challenge sq-ch-turns', hash: 'challenge/sq-ch-turns', ready: "document.querySelector('#challengeView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
