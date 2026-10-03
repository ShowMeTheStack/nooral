// Part 12, transformers: the guided tour, the course, and the pages the diagram checker looks at. There is no model
// challenge in this part: the canvas's transformer classifier, trained from scratch on a few hundred short texts, does
// no better than counting words (0.71 against 0.69 on the reviews with a turn, where a GRU reaches 0.83), so a target
// would reward luck rather than understanding. The part's own trained transformers live in its topics instead.
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.courses.has('tf-course')) return;

  // ---- the guided tour ---------------------------------------------------------------------------------------------
  C.tour({
    id: 'tf-tour', title: 'Tour: transformers', blurb: 'Tokens and positions, attention matrix by matrix, masks, heads, the block, and a tiny transformer learning to reverse digits, in eight steps.',
    steps: [
      { route: 'part=transformers', target: '#main', text: 'Part 12 of the catalogue: transformers. The small attention examples are drawn from seeds; the tiny transformers were trained here to reverse, copy and sort five digits.' },
      { route: 'topic=attention-sdpa', target: '#stageWrap', text: 'Scaled dot-product attention with every matrix shown: queries, keys and values, the scores, the division by √dₖ, the softmax, and the weighted sum.' },
      { target: '#btnPlay', text: 'Play steps through the story; the arrow keys go one step at a time, forwards or back.' },
      { route: 'topic=attention-sdpa', target: '#inputs', text: 'Change the words, the size dₖ or the seed: every number on the stage is worked out again.' },
      { route: 'topic=attention-masks', target: '#stageWrap', text: 'Padding and causal masks: scores set to −∞ before the softmax, so their weights are exactly zero.' },
      { route: 'topic=attention-multihead', target: '#stageWrap', text: 'The trained model’s four heads side by side, and what happens when each is switched off.' },
      { route: 'topic=transformer-train', target: '#stageWrap', text: 'The reversing transformer training: the loss falling, the pointing head forming, and the exact answers jumping from none to all.' },
      { route: 'learn', target: '[data-chapter="tf-read"]', text: 'The lessons and exercises of this part start here: attention worked out by hand on tiny matrices, masks, layer normalisation, parameter counts, and the cost of generating with and without a cache.' },
    ],
  });

  // ---- a course through every topic of the part --------------------------------------------------------------------
  C.course({
    id: 'tf-course', title: 'Transformers', blurb: 'All twelve topics of part 12, from tokens to the key–value cache, with the lessons and a few exercises on the way.',
    steps: [
      { lesson: 'tf-tokens', note: 'Tokens and positions.' },
      { topic: 'transformer-tokens', note: 'A token is a row of a learned table.' },
      { topic: 'transformer-positions', note: 'Sinusoids and learned rows.' },
      { exercise: 'tf-x-sinusoid', note: 'A position vector by hand.' },
      { lesson: 'tf-sdpa', note: 'Attention, matrix by matrix.' },
      { topic: 'attention-sdpa', note: 'Every matrix on show.' },
      { exercise: 'tf-x-scores', note: 'The scaled scores.' },
      { exercise: 'tf-x-weights', note: 'From scores to weights.' },
      { topic: 'attention-scale', note: 'Why divide by √dₖ.' },
      { exercise: 'tf-x-scale-curve', note: 'Without the division.' },
      { topic: 'attention-masks', note: 'Causal and padding masks.' },
      { exercise: 'tf-x-causal-weights', note: 'Weights under a causal mask.' },
      { topic: 'attention-multihead', note: 'What each head does.' },
      { exercise: 'tf-x-head-ablate', note: 'Switching heads off.' },
      { lesson: 'tf-resnorm', note: 'Residuals and layer normalisation.' },
      { topic: 'transformer-ffn', note: 'The feed-forward layer.' },
      { topic: 'transformer-residual-norm', note: 'The residual stream.' },
      { exercise: 'tf-x-layernorm', note: 'Layer normalisation by hand.' },
      { topic: 'transformer-block', note: 'The whole block, traced.' },
      { exercise: 'tf-x-block-params', note: 'Counting a block’s weights.' },
      { topic: 'encoder-decoder', note: 'Encoder against decoder.' },
      { lesson: 'tf-train', note: 'Training a small transformer.' },
      { topic: 'transformer-train', note: 'The pointing head forms.' },
      { exercise: 'tf-x-exact-curve', note: 'Exactly right answers.' },
      { topic: 'kv-cache-attention', note: 'The key–value cache.' },
      { exercise: 'tf-x-kv-rows', note: 'Work with and without the cache.' },
    ],
  });

  // ---- what the diagram checker looks at on these pages -------------------------------------------------------------
  if (S.Checks && !S.Checks.some((c) => c.name === 'lesson tf-tokens')) {
    const lessonReady = "document.querySelector('#lessonView')", exReady = "document.querySelector('#exerciseView')";
    for (const id of C.chapters.filter((c) => c.part === 'transformers').flatMap((c) => c.lessons)) S.Checks.push({ group: 'learn', name: 'lesson ' + id, hash: 'lesson=' + id, ready: lessonReady, sel: '#main' });
    for (const id of ['tf-x-sinusoid', 'tf-x-scores', 'tf-x-weights', 'tf-x-one-query', 'tf-x-code-attn', 'tf-x-scale-curve', 'tf-x-causal-weights', 'tf-x-pad-row', 'tf-x-both-masks', 'tf-x-head-ablate', 'tf-x-layernorm', 'tf-x-block-order', 'tf-x-train-curve', 'tf-x-exact-curve', 'tf-x-kv-rows'])
      S.Checks.push({ group: 'learn', name: 'exercise ' + id, hash: 'exercise=' + id, ready: exReady, sel: '#main' });
    S.Checks.push({ group: 'learn', name: 'course tf-course', hash: 'course=tf-course', ready: "document.querySelector('#courseView')", sel: '#main' });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
