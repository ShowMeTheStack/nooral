// Part 12, transformers: the exercises of the last two chapters (the feed-forward layer, residuals and layer
// normalisation, the block, encoder and decoder, training, the key–value cache). Hand computations are worked out by the
// tensor engine when the file loads (T.layerNorm, T.gelu, the layers' own parameter counts); the training curves are
// the shipped run of the reversing transformer (S.P12.run), the same one its topic draws.
(function (S) {
  'use strict';
  const C = S.Content, T = S.T, P = S.P12;
  if (!C || !T || !P || C.exercises.has('tf-x-ffn-params')) return;
  const nn = T.nn;
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const sign = (v) => String(v).replace(/^-/, '−');
  const f64 = (fn) => T.float64(fn);
  const ex = (x) => C.exercise(x);
  const str = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));

  // =========================================================================================================
  // Chapter 3: inside the block
  // =========================================================================================================
  const FF = f64(() => new nn.Dense(16, 32, { rng: 1 }).numParams() + new nn.Dense(32, 16, { rng: 1 }).numParams());
  ex({ id: 'tf-x-ffn-params', chapter: 'tf-block', title: 'The weights of a feed-forward layer', level: 2, kind: 'number',
    text: ['A feed-forward layer widens 16 numbers to 32, applies GELU, and narrows back to 16. Both layers have biases. How many learned numbers does it hold?'],
    answer: FF, tolerance: 0,
    whyNot: { '1024': 'That leaves out the biases: 32 in the first layer and 16 in the second.', '512': 'There are two layers, 16 × 32 and 32 × 16.', '1056': 'The second layer has biases too: 16 of them.', '1040': 'The first layer has 32 biases, one per hidden unit.' },
    hints: ['First layer: a 16 × 32 matrix plus one bias per hidden unit.', '16 · 32 + 32 = 544. Second layer: 32 · 16 + 16.', '544 + 528 = ' + FF + ', a little less than attention’s 1088 in the same block.'],
    solution: String(FF), wrong: ['1024', '512'] });

  const G = f64(() => r3(T.gelu(T.tensor([-1])).item()));
  ex({ id: 'tf-x-gelu', chapter: 'tf-block', title: 'GELU of a negative number', level: 2, kind: 'number',
    text: ['The feed-forward layers here use GELU in its tanh form. What is GELU(−1)? Three decimals.', { math: 'GELU(x) = 0.5 · x · (1 + tanh(0.7979 · (x + 0.044715 · x³)))' }],
    answer: G, tolerance: 0.002,
    whyNot: { '0': 'That is ReLU. GELU lets a little of a negative number through.', '-1': 'GELU squeezes negative numbers towards 0.', '-0.5': 'Remember the tanh term: 1 + tanh(…) is far below 1 here.', '0.159': 'Mind the sign: the input is negative, so the output is too.' },
    hints: ['Inside the brackets: −1 + 0.044715 · (−1)³ = −1.044715.', 'tanh(0.7979 · −1.044715) = tanh(−0.8336) = −0.6824.', '0.5 · (−1) · (1 − 0.6824) = ' + sign(G) + '. ReLU would give 0; GELU lets a small negative value through.'],
    solution: String(G), wrong: ['0', '-1'] });

  const LX = [1, 3, 3, 9], LN = f64(() => T.layerNorm(T.tensor([LX])).toArray()[0].map(r3));
  const lcells = { 'LN(x)|x₁': LN[0], 'LN(x)|x₂': LN[1], 'LN(x)|x₃': LN[2], 'LN(x)|x₄': LN[3] };
  ex({ id: 'tf-x-layernorm', chapter: 'tf-block', title: 'Layer normalisation by hand', level: 2, kind: 'table',
    text: ['A position’s vector is x = (1, 3, 3, 9). Normalise it as layer normalisation does, with γ = 1, β = 0 and ε small enough to ignore. The variance divides by the number of entries, 4. Three decimals.', { math: 'LN(x)ᵢ = (xᵢ − mean) / √(variance + ε)' }],
    rows: ['LN(x)'], cols: ['x₁', 'x₂', 'x₃', 'x₄'], corner: '', cells: lcells, tolerance: 0.003,
    hints: ['The mean is (1 + 3 + 3 + 9) / 4 = 4.', 'Deviations −3, −1, −1, 5; their squares average (9 + 1 + 1 + 25) / 4 = 9, so the spread is 3.', 'Divide each deviation by 3: (' + LN.map(sign).join(', ') + '). Mean 0, spread 1, whatever the scale of x was.'],
    solution: str(lcells), wrong: [str({ 'LN(x)|x₁': -3, 'LN(x)|x₂': -1, 'LN(x)|x₃': -1, 'LN(x)|x₄': 5 }), str({ 'LN(x)|x₁': r3(-3 / 9), 'LN(x)|x₂': r3(-1 / 9), 'LN(x)|x₃': r3(-1 / 9), 'LN(x)|x₄': r3(5 / 9) })] });

  ex({ id: 'tf-x-residual', chapter: 'tf-block', title: 'Why add instead of replace?', level: 1, kind: 'choice',
    text: ['A block computes x + f(x) rather than f(x). What does this give a deep stack of blocks?'],
    options: ['A direct path for the gradient: the derivative of x + f(x) is 1 + f′(x), so it never has to pass only through f', 'Fewer weights to learn', 'Outputs that always stay between 0 and 1', 'It lets each position see the others'], answer: 0,
    whyNot: { 1: 'The addition has no weights; the count is unchanged.', 2: 'Nothing bounds the stream: it grows as sublayers add to it.', 3: 'Mixing between positions is attention’s job.' },
    hints: ['Think about the gradient flowing back through many blocks.', 'd(x + f(x))/dx = 1 + f′(x).', 'The 1 is a direct path back to the embedding through any number of blocks, so a deep stack trains about as easily as a shallow one.'],
    solution: 0, wrong: [1, 2] });

  const lnRef = (x, g, b, eps) => f64(() => T.layerNorm(T.tensor([x]), T.tensor(g), T.tensor(b), eps).toArray()[0]);
  const CODE_LN = 'function layerNorm(x, gamma, beta, eps) {\n  const n = x.length;\n  const mean = x.reduce((a, b) => a + b, 0) / n;\n  const v = x.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;\n  const s = Math.sqrt(v + eps);\n  return x.map((xi, i) => gamma[i] * (xi - mean) / s + beta[i]);\n}';
  ex({ id: 'tf-x-code-layernorm', chapter: 'tf-block', title: 'Write layer normalisation', level: 2, kind: 'code',
    text: ['Write `layerNorm(x, gamma, beta, eps)` for one vector x: subtract its mean, divide by √(variance + eps) with the variance dividing by the length, then multiply entry i by gamma[i] and add beta[i].'],
    fn: 'layerNorm', params: ['x', 'gamma', 'beta', 'eps'], start: 'function layerNorm(x, gamma, beta, eps) {\n  \n}', tolerance: 1e-6, ref: lnRef,
    cases: [[[1, 3, 3, 9], [1, 1, 1, 1], [0, 0, 0, 0], 1e-5], [[2, 4, 6, 8], [2, 1, 0.5, 1], [0.1, 0, -0.1, 0], 1e-5], [[5, 5, 5], [1, 1, 1], [0, 0, 0], 1e-5], [[0.1, -0.2, 0.3, 0, 0.05], [1.2, 0.8, 1, 1, 1.1], [0, 0.5, 0, -0.5, 0], 0.01]],
    hints: ['Two passes over x: the mean, then the mean of the squared deviations.', 'Put eps inside the square root, so a vector of equal numbers gives zeros rather than a division by zero.', CODE_LN],
    solution: CODE_LN,
    wrong: [CODE_LN.replace('/ n;\n  const s', '/ (n - 1);\n  const s'), CODE_LN.replace('gamma[i] * (xi - mean) / s + beta[i]', '(xi - mean) / s')] });

  ex({ id: 'tf-x-block-order', chapter: 'tf-block', title: 'The order inside a block', level: 2, kind: 'choice',
    text: ['The blocks of this part are pre-norm blocks. Which order is theirs?'],
    options: ['x ← x + Attention(LN₁(x)), then x ← x + FFN(LN₂(x))', 'x ← LN₁(x + Attention(x)), then x ← LN₂(x + FFN(x))', 'x ← FFN(Attention(x)), then x ← LN(x)', 'x ← x + FFN(LN₁(x)), then x ← x + Attention(LN₂(x))'], answer: 0,
    whyNot: { 1: 'That is the post-norm order: the norm sits on the stream itself, after each addition.', 2: 'There are no residual additions there, and a single norm.', 3: 'Attention comes first: positions gather from each other, then each is processed alone.' },
    hints: ['In a pre-norm block the norm is applied to the copy a sublayer reads, not to the stream.', 'The stream x is only ever added to.', 'x ← x + Attention(LN₁(x)), then x ← x + FFN(LN₂(x)): attend, then feed forward, each on a normalised copy, each added back.'],
    solution: 0, wrong: [1, 3] });

  const BP = f64(() => new nn.TransformerBlock(16, 2, { ff: 32, rng: 1 }).numParams());
  ex({ id: 'tf-x-block-params', chapter: 'tf-block', title: 'Counting a block’s weights', level: 3, kind: 'number',
    text: ['A transformer block 16 numbers wide, with 2 heads and a feed-forward layer 32 wide. Attention has four 16 × 16 matrices (Wq, Wk, Wv, Wo), each with a bias; the feed-forward layer has two layers with biases; each of the two layer norms has a γ and a β of 16. How many learned numbers in all?'],
    answer: BP, tolerance: 0,
    whyNot: { '2160': 'You left out the two layer norms: 2 × (16 + 16).', '2096': 'The four attention matrices each have a bias of 16.', '1088': 'That is attention alone; add the feed-forward layer and the norms.', '2752': 'Two heads share the width: still four 16 × 16 matrices, not eight.' },
    hints: ['Attention: 4 × (16 · 16 + 16).', 'Attention 1088, feed-forward 1072, norms 2 × 32 = 64.', '1088 + 1072 + 64 = ' + BP + '. The number of heads does not change the count: the heads split the same matrices.'],
    solution: String(BP), wrong: ['2160', '1088'] });

  ex({ id: 'tf-x-arch', chapter: 'tf-block', title: 'Which design for which job?', level: 1, kind: 'choice',
    text: ['A team must turn sentences in one language into another: read the whole input, then write the output word by word. Which design was made for exactly this?'],
    options: ['An encoder–decoder with cross-attention', 'An encoder alone', 'A single attention layer with no feed-forward layers', 'A decoder with no causal mask'], answer: 0,
    whyNot: { 1: 'An encoder understands its input but has no way to generate a sequence token by token.', 2: 'One layer of attention alone cannot compute much.', 3: 'Without the causal mask a decoder could see the words it is meant to predict, so it cannot be trained to generate.' },
    hints: ['The job has two halves: understanding all of the input, and generating the output.', 'Understanding wants positions that look both ways; generating wants a causal mask.', 'An encoder–decoder: the encoder reads the input both ways, and the decoder generates causally while its cross-attention reads the encoder’s keys and values. A decoder-only model given the input as a prompt can also do it.'],
    solution: 0, wrong: [1, 3] });

  ex({ id: 'tf-x-cross-shape', chapter: 'tf-block', title: 'The size of cross-attention', level: 2, kind: 'number',
    text: ['An encoder–decoder model reads an input of 12 tokens and has generated 7 tokens so far. In one head of the decoder’s cross-attention, how many weights does the weight matrix hold?'],
    answer: 84, tolerance: 0,
    whyNot: { '144': 'That is the encoder’s own 12 × 12 self-attention. Cross-attention pairs decoder queries with encoder keys.', '49': 'That is the decoder’s own self-attention. Its keys here come from the encoder.', '19': 'Every query meets every key: multiply.' },
    hints: ['The queries come from the decoder, the keys from the encoder.', 'One row per decoder token, one column per encoder token.', '7 × 12 = 84 weights; each of the 7 rows adds up to 1.'],
    solution: '84', wrong: ['144', '49'] });

  // =========================================================================================================
  // Chapter 4: a tiny transformer at work
  // =========================================================================================================
  const R = P.run('reverse'), CK = R.ckpts.filter((c) => c.step > 0);
  const LC = CK.map((c) => [c.step, r3(c.loss)]), EC = R.ckpts.map((c) => [c.step, c.exact]);
  ex({ id: 'tf-x-train-curve', chapter: 'tf-work', title: 'The loss of the reversing transformer', level: 2, kind: 'predict-curve',
    text: ['The tiny transformer of this part (2 blocks, 2 heads, 16 wide) trains on reversing five digits, in batches of 32. At step 25 its loss is ' + LC[0][1] + '. Sketch the loss up to step 400.'],
    plot: { xr: [0, 410], yr: [0, 2.2], xname: 'training step', yname: 'loss', given: LC.slice(0, 1), note: 'Drag from step 25 to step 400. The faint point is step 25.' },
    answer: LC, tolerance: 0.2, checks: ['falling'],
    hints: ['For a while the model only learns which digits are common; then a head learns to point.', 'Step 75: ' + LC[2][1] + '; step 100: ' + LC[3][1] + '.', 'It falls steadily to step 100, then drops almost to 0: ' + LC.map(([s, v]) => 'step ' + s + ': ' + v).join(', ') + '.'],
    solution: LC, wrong: [LC.map(([s]) => [s, r3(LC[0][1] * (1 - s / 820))]), LC.map(([s]) => [s, LC[0][1]])] });

  ex({ id: 'tf-x-exact-curve', chapter: 'tf-work', title: 'Exactly right answers, step by step', level: 2, kind: 'predict-curve',
    text: ['The same training run. At each checkpoint the model answers 100 new sequences; an answer counts only if all five digits are right. Before training and at step 25 it gets none. Sketch the number exactly right up to step 400.'],
    plot: { xr: [0, 410], yr: [0, 105], xname: 'training step', yname: 'exactly right (of 100)', given: EC.slice(0, 2), note: 'Drag from step 0 to step 400. The faint points are steps 0 and 25.' },
    answer: EC, tolerance: 12, checks: ['rising'],
    hints: ['An answer needs five digits right at once, so the count stays low until every digit is likely to be right.', 'Step 75 still has only ' + EC[3][1] + '; step 100 has ' + EC[4][1] + '.', 'A sudden jump: ' + EC.map(([s, v]) => 'step ' + s + ': ' + v).join(', ') + '. The loss fell gradually, but the exact count needs all five pointers working together.'],
    solution: EC, wrong: [EC.map(([s]) => [s, Math.round(s / 4)]), EC.map(([s]) => [s, s < 300 ? 0 : 100])] });

  const L9 = r3(Math.log(9));
  ex({ id: 'tf-x-start-loss', chapter: 'tf-work', title: 'The loss of a blind guess', level: 1, kind: 'number',
    text: ['The vocabulary has 9 tokens. A model that spreads its chances evenly over all 9 for every answer token has what cross-entropy loss (natural log)? Three decimals.'],
    answer: L9, tolerance: 0.002,
    whyNot: { '3.17': 'That is log base 2; the loss here uses the natural log.', '0.111': 'That is the chance of each token, 1/9; the loss is −ln of it.', '9': 'The loss is the negative log of the chance given to the right token.', '0.954': 'That is log base 10.' },
    hints: ['Cross-entropy for one token is −ln(chance of the right token).', 'Each token gets 1/9, so the loss is −ln(1/9) = ln 9.', 'ln 9 = ' + L9 + '. The run starts a little above it, at ' + r3(R.loss[0]) + ', because random weights are not perfectly even.'],
    solution: String(L9), wrong: ['3.17', '0.111'] });

  // the cost of generating k tokens after a prompt of n, rows of Q, K and V per layer
  const KVN = 12, KVK = 4, WITHOUT = Array.from({ length: KVK }, (_, i) => KVN + i).reduce((a, b) => a + b, 0), WITH = KVN + KVK - 1;
  const kcells = { 'without the cache|rows per layer': WITHOUT, 'with the cache|rows per layer': WITH };
  ex({ id: 'tf-x-kv-rows', chapter: 'tf-work', title: 'Work with and without the cache', level: 3, kind: 'table',
    text: ['A decoder is given a prompt of 12 tokens and generates 4 tokens. Each generated token except the last is fed back in. Count the rows of queries, keys and values each layer computes in all: without a cache every pass runs the whole sequence; with a cache the prompt runs once and then only each new token.', 'The [key–value cache topic](#topic=kv-cache-attention) counts the same way: a prompt of 6 and 5 answer digits cost 10 rows with the cache, 40 without.'],
    rows: ['without the cache', 'with the cache'], cols: ['rows per layer'], corner: '', cells: kcells,
    hints: ['Without a cache, the passes see 12, then 13, then 14, then 15 tokens.', 'With a cache: 12 rows for the prompt, then one row for each of the 3 tokens fed back.', 'Without: 12 + 13 + 14 + 15 = ' + WITHOUT + '. With: 12 + 3 = ' + WITH + '. The gap grows with the square of the length.'],
    solution: str(kcells), wrong: [str({ 'without the cache|rows per layer': 12 * 4, 'with the cache|rows per layer': 16 }), str({ 'without the cache|rows per layer': WITHOUT, 'with the cache|rows per layer': 16 })] });

  const MEM = (2 * 24 * 2000 * 1024) / 1e6;
  ex({ id: 'tf-x-kv-memory', chapter: 'tf-work', title: 'How big does the cache get?', level: 3, kind: 'number',
    text: ['A decoder has 24 layers and is 1024 numbers wide (its heads together). It has read 2000 tokens. How many numbers does its key–value cache hold, in millions?'],
    answer: MEM, tolerance: 0.01,
    whyNot: { '49.152': 'That is the keys alone; the values are kept too.', '2.048': 'That is one layer’s keys; every layer keeps its own keys and values.', '4.096': 'That is one layer; there are 24.', '196.608': 'Count numbers, not bytes.' },
    hints: ['Every layer keeps a key and a value of 1024 numbers for every token.', 'Per token: 24 layers × 2 × 1024 = 49152 numbers.', '49152 × 2000 = ' + (2 * 24 * 2000 * 1024) + ', about ' + MEM + ' million numbers. At 2 bytes each, about 197 MB, for a single text.'],
    solution: String(MEM), wrong: ['49.152', '4.096'] });

  ex({ id: 'tf-x-kv-what', chapter: 'tf-work', title: 'What goes in the cache?', level: 1, kind: 'choice',
    text: ['When a decoder generates text with a key–value cache, what does it keep from earlier tokens?'],
    options: ['Their keys and values in every layer', 'Their queries in every layer', 'Only their final output probabilities', 'Their embeddings only'], answer: 0,
    whyNot: { 1: 'A query is used once, by its own position; later tokens ask with their own queries.', 2: 'The next token needs to attend to earlier tokens in every layer, which needs their keys and values.', 3: 'Every layer has its own keys and values, computed from that layer’s input.' },
    hints: ['A new token attends to every earlier token. What does it need from them?', 'Its query is scored against their keys, and their values are averaged.', 'Keys and values, in every layer: they never change once computed, so they are kept and only the new token’s row is computed.'],
    solution: 0, wrong: [1, 2] });
})(globalThis.Nooral = globalThis.Nooral || {});
