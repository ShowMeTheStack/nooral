// Part 12, transformers: the chapters and the lessons. The exercises are in 121-tf-ex.js; the guided tour, the course
// and the pages the diagram checker looks at are in 122-tf-ch.js. Each lesson opens with a question, runs the topic
// that shows the idea at a frame chosen for the point being made, asks for a prediction, sums up and leads to its
// exercises. Every value quoted here was printed by the topics themselves: the small attention examples drawn from
// their seeds, and the tiny transformers trained on reversing, copying and sorting digits (p12-0-data.js).
(function (S) {
  'use strict';
  const C = S.Content;
  if (!C || C.lessons.has('tf-tokens')) return;
  const X = (id) => ({ exercise: id });

  C.chapter({ id: 'tf-read', part: 'transformers', n: 1, title: 'Tokens, positions and attention', blurb: 'Tokens become rows of a learned table, positions are added because attention cannot see order, and scaled dot-product attention lets every position gather from every other, every matrix shown.' });
  C.chapter({ id: 'tf-mask', part: 'transformers', n: 2, title: 'Masks and heads', blurb: 'Causal and padding masks decide who may look at whom; several heads ask different questions of the same sequence, and switching one off shows what it was for.' });
  C.chapter({ id: 'tf-block', part: 'transformers', n: 3, title: 'Inside the block', blurb: 'The feed-forward layer, the residual stream and layer normalisation, the whole block traced with real numbers, and how the same block makes an encoder or a decoder.' });
  C.chapter({ id: 'tf-work', part: 'transformers', n: 4, title: 'A tiny transformer at work', blurb: 'A transformer of a few thousand weights learning to reverse digits, its pointing head forming, and the key–value cache that makes each new token cheap.' });

  // =========================================================================================================
  // Chapter 1: tokens, positions and attention
  // =========================================================================================================
  C.lesson({
    id: 'tf-tokens', chapter: 'tf-read', title: 'Tokens and positions', minutes: 9,
    summary: 'A transformer reads token ids. Each id picks a row of a learned embedding matrix, which is the same as multiplying a one-hot vector by it. Attention on its own treats the rows as a set, blind to their order, so a position vector is added to every row: fixed sinusoids, or one learned row per position.',
    blocks: [
      'The tiny transformer of this part reads “2 7 4 0 5 =” and must answer “5 0 4 7 2”. It never sees a digit, only numbers. Which numbers, and how does it know that the 7 came second?',
      { topic: 'transformer-tokens', frame: 0, caption: 'Five digits, the separator “=” and the answer so far: 10 tokens, each an id from a vocabulary of 9 (the digits 0 to 7 and “=”).' },
      { topic: 'transformer-tokens', frame: 1, caption: 'The embedding matrix holds one row of 16 learned numbers per token. Position 1 holds “7”, so its vector is row 7, starting (0.79, −0.08, −0.04, −0.29).' },
      { math: 'x = E[i]  (row i of the embedding matrix E)\n  = one-hot(i) · E' },
      { topic: 'transformer-tokens', frame: 3, caption: 'Stacked, the rows make a 10 × 16 input matrix. “7” sits at positions 1 and 9 and gets the same row both times: nothing yet says where a token stands.' },
      { predict: { q: 'The model’s first attention layer is run on 2 7 4 0 5, then on the same digits shuffled, with no positions added. What happens to the output for “7”?', options: ['It is exactly the same in both', 'It changes a little', 'It changes completely'], answer: 0, explain: 'Attention compares every row with every other in the same way. Shuffling the rows only shuffles the outputs: “7” gets the same output wherever it stands.' } },
      { topic: 'transformer-positions', frame: 0, caption: 'The two output rows for “7”, in order and shuffled: exactly the same. Without positions the model cannot tell “2 7” from “7 2”.' },
      { topic: 'transformer-positions', frame: 1, caption: 'The fixed encoding: 40 positions × 32 dimensions of sines and cosines. The first columns turn every 6.28 positions, the last pair every 35300.' },
      { math: 'PE(p, 2i)   = sin(p / 10000^(2i/d))\nPE(p, 2i+1) = cos(p / 10000^(2i/d))\ninput row   = E[token] + PE(p)' },
      { topic: 'transformer-positions', frame: 3, caption: 'The dot product of two positions depends only on their distance: PE(2) · PE(6) = PE(14) · PE(18) = 11.662, both 4 apart. It peaks at d/2 = 16 for a position with itself.' },
      { topic: 'transformer-positions', frame: 4, caption: 'The trained model learned its own 11 position rows instead. Neighbours have a mean cosine of 0.13, positions five or more apart −0.06. There is no row for a 12th position.' },
      { h: 'Recap' },
      { list: ['A token is an id; its vector is a row of a learned embedding matrix, a product with a one-hot vector.', 'Attention alone is blind to order, so a position vector is added to every token’s row.', 'Sinusoids are fixed and go on for ever; learned rows fit the task but stop at the length they were trained on.'] },
      X('tf-x-emb-size'), X('tf-x-sinusoid'), X('tf-x-order'),
    ],
  });

  C.lesson({
    id: 'tf-sdpa', chapter: 'tf-read', title: 'Scaled dot-product attention, matrix by matrix', minutes: 10,
    summary: 'Each word becomes a query, a key and a value through three learned matrices. Every query is compared with every key by a dot product; the scores are divided by √dₖ; a softmax turns each row into weights that add up to 1; and each word’s output is the weighted sum of the values. All of it is a few matrix products.',
    blocks: [
      'In “rain fell on roofs”, which words should “fell” listen to, and how can a network decide that from numbers alone?',
      { topic: 'attention-sdpa', frame: 0, caption: 'Four words, each a row of 4 numbers (in a real model, its embedding plus its position). We follow “fell”: (−1.4, −0.1, 1.8, −0.3).' },
      { topic: 'attention-sdpa', frame: 1, caption: 'Q = X Wq: every word gets a query of dₖ = 3 numbers. The first number of “fell”’s query is its row times the first column of Wq: 1.40 + 0.03 + 0.54 + 0.39 = 2.36.' },
      { def: 'query, key, value', text: 'The query says what a position is looking for, the key what it offers to others, the value what it hands over to whoever attends to it. All three come from the same row through three learned matrices.' },
      { topic: 'attention-sdpa', frame: 4, caption: 'Every query meets every key. The row of “fell” scores (0.83, −5.86, 1.81, −1.12); the largest is for “on”.' },
      { topic: 'attention-sdpa', frame: 5, caption: 'Divided by √3 = 1.732: (0.48, −3.38, 1.04, −0.65).' },
      { predict: { q: 'Which word will get the largest weight in the row of “fell” after the softmax?', options: ['“on”, which had the largest score', '“fell” itself', 'all four get 0.25', '“fell” shares it equally between “rain” and “roofs”'], answer: 0, explain: 'The softmax keeps the order of the scores. “on” gets 0.57, “rain” 0.32, “roofs” 0.10 and “fell” 0.01.' } },
      { topic: 'attention-sdpa', frame: 6, caption: 'Exponentiate each scaled score and divide by their total, 5.01: the weights (0.32, 0.01, 0.57, 0.10) add up to 1. The arcs below draw the row.' },
      { topic: 'attention-sdpa', frame: 7, caption: 'The output of “fell” is the weights times the values: 0.22 + 0.02 + 0.29 − 0.03 = 0.50 for its first number, a blend led by the value of “on”.' },
      { math: 'Q = X Wq    K = X Wk    V = X Wv\nA = softmax(Q Kᵀ / √dₖ)   (row by row)\noutput = A V' },
      { topic: 'attention-sdpa', frame: 6, input: { sentence: 'mila', dk: '2', focus: 3, seed: 7 }, caption: 'Other numbers, another picture: in “Mila fed her cat” with dₖ = 2, “cat” scores 5.75 against “Mila” and gives it 0.99 of its weight. A softmax of widely spread scores picks almost one word.' },
      { h: 'Recap' },
      { list: ['Three learned matrices turn each row into a query, a key and a value.', 'Scores are dot products of queries with keys, divided by √dₖ; a softmax makes each row into weights adding up to 1.', 'Each output row is the weighted sum of the values; all positions are done at once by matrix products.'] },
      X('tf-x-scores'), X('tf-x-weights'), X('tf-x-output'), X('tf-x-one-query'), X('tf-x-code-attn'),
    ],
  });

  C.lesson({
    id: 'tf-scale', chapter: 'tf-read', title: 'Why divide by √dₖ?', minutes: 7,
    summary: 'A dot product adds up dₖ products, so the scores of random queries and keys spread out like √dₖ. Fed to a softmax, widely spread scores give one key nearly all the weight, and the rate at which the weights can change, p(1 − p), nearly vanishes. Dividing by √dₖ brings the spread back to about 1 at every size.',
    blocks: [
      'Real models use queries and keys of 64 or 128 numbers, not 3. Does that change how attention behaves, if nothing else does?',
      { topic: 'attention-scale', frame: 1, caption: 'dₖ = 2: random queries against 8 keys. The raw scores spread 1.26, close to √2; the largest weight in a row is 0.37 unscaled, 0.30 scaled.' },
      { predict: { q: 'At dₖ = 256, what will the largest weight of a row be on average, without the division?', options: ['about 0.94: one key takes almost everything', 'about 0.37, as at dₖ = 2', 'about 0.125: perfectly even'], answer: 0, explain: 'The raw scores spread like √256 = 16, so the biggest one dwarfs the rest after exponentiating.' } },
      { topic: 'attention-scale', frame: 8, caption: 'dₖ = 256: spread 15.76, close to √256 = 16. Unscaled, the largest weight is 0.94 and 81% of rows give one key more than 90%. Scaled, it stays 0.36.' },
      { topic: 'attention-scale', frame: 9, caption: 'One query at dₖ = 128: its raw scores run from −3.6 to 9.0. Unscaled, key 1 takes 0.856 of the weight; scaled, 0.231.' },
      { math: '∂pᵢ/∂sᵢ = pᵢ (1 − pᵢ)    near 0 when pᵢ is near 0 or 1' },
      { topic: 'attention-scale', frame: 10, caption: 'The rate at which weights can learn, summed over a row: 0.086 unscaled at dₖ = 256, 0.766 scaled. Saturated attention barely learns.' },
      { h: 'Recap' },
      { list: ['The spread of dot products grows like √dₖ.', 'Wide scores make the softmax nearly one-hot, and its gradient pᵢ(1 − pᵢ) nearly zero.', 'Dividing by √dₖ keeps the scores near spread 1, so attention stays soft and trainable at any size.'] },
      X('tf-x-scale-factor'), X('tf-x-scale-curve'), X('tf-x-shift'),
    ],
  });

  // =========================================================================================================
  // Chapter 2: masks and heads
  // =========================================================================================================
  C.lesson({
    id: 'tf-masks', chapter: 'tf-mask', title: 'Causal and padding masks', minutes: 8,
    summary: 'A mask sets chosen scores to −∞ (in practice −10⁹) before the softmax, so their weights become exactly 0. A padding mask hides the filler tokens that make sentences in a batch the same length. A causal mask hides every later position, so a model can be trained to predict every next token in one pass without seeing the answers.',
    blocks: [
      'Two sentences of different lengths must go through the model together, and a language model must not peek at the word it is about to predict. How can attention be told what not to look at?',
      { topic: 'attention-masks', frame: 0, caption: 'A batch is one rectangle: sentence B, 3 words, gets 2 padding tokens of zeros to match sentence A’s 5.' },
      { topic: 'attention-masks', frame: 1, caption: 'Without a mask the pads take part like words: the row of “birds” gives them 0.14 of its weight, drawn from nothing.' },
      { topic: 'attention-masks', frame: 2, caption: 'The padding mask: every score in a pad’s column becomes −10⁹, 10 scores here. After the softmax those weights are exactly 0; “birds” spreads (0.08, 0.90, 0.02) over the real words.' },
      { predict: { q: 'With a causal mask on 5 positions, how many of the 25 scores are hidden?', options: ['10', '5', '15', '20'], answer: 0, explain: 'Everything above the diagonal: 4 + 3 + 2 + 1 + 0 = 10. The diagonal stays, so each position can attend to itself.' } },
      { topic: 'attention-masks', frame: 3, caption: 'The causal mask: 10 of 25 scores hidden. The first word attends only to itself (weight 1.00); the last spreads over all 5.' },
      { topic: 'attention-masks', frame: 4, caption: 'Both masks at once: a score survives only if it is not in the future and not on a pad. 6 of B’s 9 scores among real words survive.' },
      { topic: 'attention-masks', frame: 5, caption: 'Give the last word, “trees”, different numbers. With the causal mask the 4 earlier outputs do not move at all; without it they move by up to 1.121.' },
      { tip: 'That is why one pass over a sentence trains every next-token prediction at once: position i’s output depends only on positions 0 to i, so its target, word i + 1, stays hidden from it.' },
      { h: 'Recap' },
      { list: ['A mask sets scores to −∞ before the softmax; their weights become exactly 0 and the rest still add up to 1.', 'Padding masks hide filler columns; causal masks hide everything above the diagonal.', 'Masks combine: a score must pass both.'] },
      X('tf-x-causal-count'), X('tf-x-causal-weights'), X('tf-x-pad-row'), X('tf-x-both-masks'), X('tf-x-code-causal'),
    ],
  });

  C.lesson({
    id: 'tf-heads', chapter: 'tf-mask', title: 'Multi-head attention: what each head does', minutes: 8,
    summary: 'A layer cuts its queries, keys and values into several narrower heads, each scoring with its own slice, so each head can learn its own pattern for the same cost as one wide head. Their outputs are joined and mixed by one more matrix, Wo. Looking at the weights, and switching heads off one at a time, shows which head does the work.',
    blocks: [
      'To reverse five digits, the answer just after “=” must copy the last input digit, the next one the digit before it, and so on. Does every head of the trained model do that, or only some?',
      { topic: 'attention-multihead', frame: 0, caption: 'The model is 16 numbers wide; each layer cuts its queries, keys and values into 2 heads of 8.' },
      { math: 'headₕ = softmax(Qₕ Kₕᵀ / √(d / heads)) Vₕ\noutput = [head₁ … headₙ] Wo' },
      { topic: 'attention-multihead', frame: 1, caption: 'Layer 0, head 0: answer position 6 needs the digit at position 3, but its largest weight, 0.36, goes to position 2. It hits the needed digit 2 times in 5.' },
      { topic: 'attention-multihead', frame: 2, caption: 'Layer 0, head 1: 0.96 on position 3, exactly the digit needed. It points at the right digit for all 5 answer positions.' },
      { predict: { q: 'Switch off layer 0, head 1, and keep the other three heads. How many answer digits will the model still get right?', options: ['far fewer: well under half', 'about the same', 'none at all'], answer: 0, explain: 'It falls from 100% to 39% of the digits on 150 new sequences; switching off any other head leaves it at 100%.' } },
      { topic: 'attention-multihead', frame: 6, caption: 'Ablation: without layer 0, head 1 the model gets 39% of the digits right; without each of the others, still 100%.' },
      { topic: 'attention-multihead', frame: 6, input: { task: 'sort', digits: '62517', row: 5 }, caption: 'The sort model shares the work more: without its pointing head it falls from 99% to 55%, and the other heads matter too (82%, 91% and 97% without each).' },
      { h: 'Recap' },
      { list: ['Several heads split the width: each scores with its own slice and can learn its own pattern.', 'Their outputs are joined and mixed by Wo.', 'To find what a head is for, look at its weights, and switch it off and measure.'] },
      X('tf-x-head-size'), X('tf-x-head-ablate'),
    ],
  });

  // =========================================================================================================
  // Chapter 3: inside the block
  // =========================================================================================================
  C.lesson({
    id: 'tf-ffn', chapter: 'tf-block', title: 'The feed-forward layer', minutes: 6,
    summary: 'After attention has mixed information between positions, each position goes on alone through a small two-layer network: widen (here from 16 to 32), a GELU, and narrow back. The same weights serve every position, and no position sees another here. In large models these layers hold most of the weights.',
    blocks: [
      'Attention moves information between positions, but every output is only a weighted average of values. Where does the model do anything more than average?',
      { topic: 'transformer-ffn', frame: 0, caption: 'Position 6 (“5”) reaches block 0’s feed-forward layer as 16 numbers, already mixed by attention and normalised.' },
      { topic: 'transformer-ffn', frame: 1, caption: 'Widen to 32 hidden numbers: 13 come out negative. The largest is unit 3 at 2.67, the most negative unit 29 at −1.08.' },
      { predict: { q: 'What does GELU do to unit 29’s value of −1.08?', options: ['squeezes it towards 0, to about −0.15', 'makes it exactly 0, like ReLU', 'leaves it at −1.08', 'flips it to +1.08'], answer: 0, explain: 'GELU(x) = x · Φ(x): for −1.08 that is about −0.151. It dips only a little below 0 and never below about −0.17.' } },
      { topic: 'transformer-ffn', frame: 2, caption: 'GELU keeps 2.67 at 2.66 and squeezes −1.08 to −0.151: only units pushed well above 0 pass much on.' },
      { math: 'FFN(x) = W₂ · GELU(W₁ x + b₁) + b₂     (16 → 32 → 16, the same at every position)' },
      { topic: 'transformer-ffn', frame: 4, caption: 'All 10 positions at once, each lighting its own units. This layer has 1072 weights against attention’s 1088.' },
      { h: 'Recap' },
      { list: ['The feed-forward layer works on each position alone: widen, GELU, narrow.', 'It is where the model computes new features from what attention gathered.', 'Its weights, (d·ff + ff) + (ff·d + d), are often the larger part of a block.'] },
      X('tf-x-ffn-params'), X('tf-x-gelu'),
    ],
  });

  C.lesson({
    id: 'tf-resnorm', chapter: 'tf-block', title: 'Residual connections and layer normalisation', minutes: 7,
    summary: 'Each sublayer adds its result to a running vector, the residual stream, instead of replacing it, so there is always a direct path back to the input and deep stacks train easily. Before each sublayer, layer normalisation makes a copy with mean 0 and spread 1 across its own numbers, then scales and shifts it by learned γ and β.',
    blocks: [
      'Stack two blocks, or a hundred: each one changes the vectors. How does the model keep them from drifting to huge or tiny values, and how does the gradient get back to the first layer?',
      { topic: 'transformer-residual-norm', frame: 0, caption: 'Position 6 enters block 0 with mean 0.122 and spread 0.363 over its 16 numbers.' },
      { math: 'LN(x) = γ ⊙ (x − mean(x)) / √(var(x) + ε) + β      ε = 10⁻⁵' },
      { topic: 'transformer-residual-norm', frame: 1, caption: 'Subtract the mean, divide by the spread: mean 0.000 and spread 1.000, whatever came in.' },
      { topic: 'transformer-residual-norm', frame: 2, caption: 'Then the learned γ and β: dimension 4 becomes −0.631 × 1.337 − 0.001 = −0.845.' },
      { predict: { q: 'Through the two blocks, will the length of the stream at position 6 grow, shrink or stay the same?', options: ['grow', 'stay the same, because of layer normalisation', 'shrink'], answer: 0, explain: 'The norm acts on the copies fed to the sublayers, not on the stream. The stream only has things added to it: its length goes 1.53, 4.53, 6.91, 7.01 and 10.03.' } },
      { topic: 'transformer-residual-norm', frame: 3, caption: 'The stream from the embedding to the last block: each sublayer adds; the biggest addition, 7.91, comes from block 1’s feed-forward layer.' },
      { topic: 'transformer-residual-norm', frame: 4, caption: 'x + f(x) has derivative 1 + f′(x): a direct path of 1 back to the embedding, through any number of blocks.' },
      { h: 'Recap' },
      { list: ['The residual stream is added to, never replaced; its gradient always has a path of 1.', 'Layer normalisation works across one position’s numbers: mean 0, spread 1, then γ and β.', 'Pre-norm: the norm sits inside each branch, so the stream itself is free to grow.'] },
      X('tf-x-layernorm'), X('tf-x-residual'), X('tf-x-code-layernorm'),
    ],
  });

  C.lesson({
    id: 'tf-block', chapter: 'tf-block', title: 'The whole block, traced', minutes: 8,
    summary: 'A pre-norm transformer block is two steps: x + attention(LN₁(x)), then x + FFN(LN₂(x)). It maps a matrix of positions to another of the same shape, so blocks stack. After the last one, a final norm and a product with the embedding matrix give a score for every token.',
    blocks: [
      'We have seen each part alone. Put together, in which order do they run, and what comes out at the end?',
      { topic: 'transformer-block', frame: 0, caption: 'Block 0 takes the whole 10 × 16 matrix; we follow position 7 (“0”), length 1.66.' },
      { topic: 'transformer-block', frame: 2, caption: 'Attention on the normalised copy: head 0 gives position 2 a weight of 0.65, head 1 gives it 0.83. Joined and mixed by Wo: a vector of length 4.76.' },
      { topic: 'transformer-block', frame: 3, caption: 'Added to the stream: whatever attention gathered is now part of position 7.' },
      { math: 'x ← x + MHA(LN₁(x))\nx ← x + FFN(LN₂(x))' },
      { topic: 'transformer-block', frame: 6, caption: 'After the feed-forward layer, the block’s output at position 7 has length 6.27. The block holds 2224 weights and keeps the 10 × 16 shape.' },
      { predict: { q: 'Position 7 holds “0” in 2 7 4 0 5 = 5 0 …; the answer so far is “5 0”. What should the model predict next?', options: ['4', '0', '7', '5'], answer: 0, explain: 'Reversed, 2 7 4 0 5 is 5 0 4 7 2: after “5 0” comes 4. The model gives it 99%.' } },
      { topic: 'transformer-block', frame: 7, caption: 'A final norm and a product with the same embedding matrix used at the input give 9 scores; the softmax gives “4” 99%.' },
      { h: 'Recap' },
      { list: ['A block: normalise, attend, add; normalise, feed forward, add.', 'Same shape in and out, so blocks stack.', 'The output head reuses the embedding matrix: a score per token, then a softmax.'] },
      X('tf-x-block-order'), X('tf-x-block-params'),
    ],
  });

  C.lesson({
    id: 'tf-encdec', chapter: 'tf-block', title: 'Encoder against decoder', minutes: 7,
    summary: 'The same block is an encoder when every position may look both ways, and a decoder when a causal mask lets each look only back. Encoders suit understanding a whole input; decoders generate one token at a time. An encoder–decoder model has both, joined by cross-attention: the decoder’s queries read the encoder’s keys and values.',
    blocks: [
      'Classify a review, write the next word, translate a sentence. Do these three jobs want the same kind of transformer?',
      { topic: 'encoder-decoder', frame: 0, caption: 'The trained model is a decoder: every weight to the right of the diagonal is zero. Position 0 reads only itself, position 9 all 10.' },
      { topic: 'encoder-decoder', frame: 1, caption: 'Remove the mask and the same weights act as an encoder layer: position 0 now puts 0.99 of its weight on later positions. (The model was trained with the mask; this is only for comparison.)' },
      { predict: { q: 'Change only the last token and run both again. In the decoder, which outputs move?', options: ['only the last position’s', 'every position’s', 'none'], answer: 0, explain: 'No earlier position can see the last one, so positions 0 to 8 do not move at all. In the encoder every position moves, position 0 by 2.376.' } },
      { topic: 'encoder-decoder', frame: 2, caption: 'The change, measured: decoder positions 0 to 8 stay put, every encoder position moves.' },
      { topic: 'encoder-decoder', frame: 3, caption: 'Cross-attention: the decoder’s queries meet the encoder’s keys and values in a 5 × 5 weight matrix whose rows add up to 1.' },
      { table: [['', 'looks', 'good at'], ['encoder', 'both ways', 'understanding a whole input: classifying, tagging'], ['decoder', 'back only (causal)', 'generating, one token at a time'], ['encoder–decoder', 'encoder both ways, decoder back plus cross-attention', 'turning one sequence into another: translating, summarising']], caption: 'Three uses of the same block.' },
      { h: 'Recap' },
      { list: ['Encoder: no causal mask; every output depends on the whole input.', 'Decoder: causal mask; output i depends only on positions 0 to i.', 'Cross-attention: queries from the decoder, keys and values from the encoder.'] },
      X('tf-x-arch'), X('tf-x-cross-shape'),
    ],
  });

  // =========================================================================================================
  // Chapter 4: a tiny transformer at work
  // =========================================================================================================
  C.lesson({
    id: 'tf-train', chapter: 'tf-work', title: 'Training a small transformer', minutes: 9,
    summary: 'A decoder of 2 blocks with 2 heads, 16 numbers wide, learns to reverse five digits from examples alone. Only the answer tokens are scored. The loss falls slowly at first, then the pointing head forms and the share of exactly right answers jumps from almost none to all.',
    blocks: [
      'Nobody tells the model “look at the digit four places back”. Can it find that rule just by lowering its loss?',
      { topic: 'transformer-train', frame: 0, caption: 'Random weights: the model answers 7, 7, 7, 7, 7 for the probe and gets 0 of 100 new sequences exactly right. Its attention is a blur.' },
      { topic: 'transformer-train', frame: 3, caption: 'Step 75: loss 1.29. The probe comes out 5, 0, 2, 7, 2, four digits of five; 1 of 100 new sequences is exactly right.' },
      { predict: { q: 'At step 100 the loss is 0.558. How many of 100 new sequences will be exactly right at step 150?', options: ['all 100', 'about 45', 'about 10'], answer: 0, explain: '41 at step 100, then 100 at step 150, when the loss is 0.0551. Once the pointing head is sharp, every answer digit is right.' } },
      { topic: 'transformer-train', frame: 4, caption: 'Step 100: loss 0.558, the probe all right, 41 of 100 new sequences exact; layer 0, head 1 points at the needed digit for all 5 answers.' },
      { topic: 'transformer-train', frame: 5, caption: 'Step 150: loss 0.0551, 100 of 100 exact.' },
      { topic: 'transformer-train', frame: 9, input: { task: 'sort' }, caption: 'Sorting is harder: after 500 steps the model sorts 99 of 100 new sequences, since each answer digit depends on comparing all the inputs.' },
      { workflow: 'transformer-spam', title: 'A transformer on the canvas', text: 'The same blocks classify short messages on the workflow canvas: open the model after training to see each head’s weights between the words.' },
      { warn: 'A model this small learns a toy rule from thousands of examples. On a few hundred real sentences it often does no better than counting words: attention has to be learned, and learning it takes data.' },
      { h: 'Recap' },
      { list: ['Cross-entropy on the answer tokens only; the input digits are random and cannot be predicted.', 'The loss falls, then a head learns to point, and exact answers jump.', 'Nothing in the code says “reverse”: the pattern was found by gradient descent.'] },
      X('tf-x-train-curve'), X('tf-x-exact-curve'), X('tf-x-start-loss'),
    ],
  });

  C.lesson({
    id: 'tf-kv', chapter: 'tf-work', title: 'The key–value cache', minutes: 7,
    summary: 'When a decoder generates, each new token needs the keys and values of every token before it, and those never change once computed. Keeping them, a key–value cache, means only the newest token runs through the layers at each step. Only the newest token runs through the layers; its attention still reads every cached key and value, but nothing is recomputed. The price is memory.',
    blocks: [
      'To write its fifth answer digit, must the model run the whole sequence through its blocks again?',
      { topic: 'kv-cache-attention', frame: 0, caption: 'The prefill: the 6 prompt tokens run once. Layer 0, head 1 keeps a 6 × 8 K and V; the first answer digit is “5”.' },
      { topic: 'kv-cache-attention', frame: 1, caption: '“5” is fed back. Only its own row is new: one query, one key, one value per layer. Its query scores against all 7 keys and puts 0.96 on position 3.' },
      { predict: { q: 'Why are queries not cached?', options: ['each query is used once, by its own position', 'queries are too large', 'queries change at every step'], answer: 0, explain: 'A query belongs to the position asking. Later tokens ask with their own queries, but they read every earlier key and value.' } },
      { topic: 'kv-cache-attention', frame: 5, caption: 'The answer 5, 0, 4, 7, 2, with logits identical to a full recomputation. With the cache each layer computed 10 rows of Q, K and V; without it, 40.' },
      { math: 'without a cache: n + (n + 1) + … + (n + k − 1) rows per layer\nwith a cache:    n + (k − 1) rows per layer' },
      { h: 'Recap' },
      { list: ['Earlier keys and values do not change, so they are kept.', 'Each new step computes one row and reads the rest from memory.', 'Memory grows with layers × tokens × width × 2 (keys and values).'] },
      X('tf-x-kv-rows'), X('tf-x-kv-memory'), X('tf-x-kv-what'),
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
