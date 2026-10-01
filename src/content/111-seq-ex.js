// Part 11, sequences and embeddings: the exercises. The small numbers are worked out here by the engines when the file
// loads (the tensor engine's recurrent cells, sigmoid, tanh, softmax, autodiff and gradient clipping, S.T, in double
// precision), so the text, the answers and the hints agree. The word2vec numbers come from the part's own trained
// vectors (S.P11.w2v, seed 1); they cost half a second to train, so they are written out here and were checked by a
// scratch script against that run. The encoder–decoder's accuracy by length is read from the shipped job.
(function (S) {
  'use strict';
  const C = S.Content, T = S.T, WF = S.WF;
  if (!C || !T || !WF || C.exercises.has('sq-x-onehot-dist')) return;
  const nn = T.nn;
  const r3 = (v) => Math.round(v * 1000) / 1000, r4 = (v) => Math.round(v * 10000) / 10000;
  const f64 = (fn) => T.float64(fn);
  const sign = (v) => String(v).replace(/^-/, '−');
  const vals = (t) => Array.from(t.values());
  const ex = (x) => C.exercise(x);
  const pipe = (data, model, mp, train) => WF.build({
    blocks: { data: WF.challengeBlock(data), split: ['split', { test: 0.25 }], m: [model, mp || {}], tr: ['train', train || {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']],
  });
  const str = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));

  // =========================================================================================================
  // Chapter 1: words as vectors
  // =========================================================================================================
  const NV = 88, DIST = f64(() => { const a = T.oneHot(T.tensor([52]), NV), b = T.oneHot(T.tensor([31]), NV); return Math.sqrt(T.sum(T.square(T.sub(a, b))).item()); });
  ex({ id: 'sq-x-onehot-dist', chapter: 'sq-embed', title: 'How far apart are two one-hot words?', level: 1, kind: 'number',
    text: ['A vocabulary of 88 words, each coded as a one-hot vector of 88 numbers. What is the straight-line (Euclidean) distance between the one-hot vectors of “king” and “queen”? Give it to three decimals.'],
    answer: r3(DIST), tolerance: 0.002, whyNot: { '2': 'That is the squared distance; take its square root.', '0': 'The vectors differ in two places, so they are not the same point.', '1': 'They differ by 1 in two positions, not one.' },
    hints: ['Subtract one vector from the other: what is left?', 'The difference has a +1 at king’s position, a −1 at queen’s, and zeros elsewhere: its squared length is 1 + 1.', '√2 ≈ ' + r3(DIST) + '. Every pair of different words is exactly this far apart, which is why one-hot vectors say nothing about meaning.'],
    solution: String(r3(DIST)), wrong: ['2', '1'] });

  const TS = f64(() => new nn.Embedding(5000, 32, { rng: 1 }).numParams());
  ex({ id: 'sq-x-table-size', chapter: 'sq-embed', title: 'The size of an embedding table', level: 1, kind: 'number',
    text: ['A model keeps a vocabulary of 5,000 words and embeds each in 32 numbers. How many learned numbers does its embedding table hold?'],
    answer: TS, tolerance: 0, whyNot: { '5032': 'The table has a row for every word, and every row has 32 numbers: multiply.', '25000000': 'That is a 5000 × 5000 table, the size one-hot rows would make; the embedding rows are only 32 long.', '32': 'That is one row; there is a row per word.' },
    hints: ['The table has one row per word.', 'Each row holds 32 numbers.', '5000 × 32 = ' + TS + '. As one-hot vectors the same words would each need 5000 numbers.'],
    solution: String(TS), wrong: ['5032', '25000000'] });

  const EW = ['the', 'goat', 'ate', 'bread', 'hill'], E = [[0.2, -0.1, 0.4], [0.5, 0.3, -0.2], [-0.3, 0.8, 0.1], [0.1, 0, 0.6], [0.7, -0.5, -0.4]];
  const LK = f64(() => { const x = T.add(T.oneHot(T.tensor([1]), 5), T.oneHot(T.tensor([3]), 5)); return vals(T.matmul(x, T.tensor(E))).map(r4); });
  ex({ id: 'sq-x-lookup', chapter: 'sq-embed', title: 'Two words through the table', level: 2, kind: 'matrix',
    text: ['A vocabulary of five words, numbered 0 to 4, and its embedding table E (one row per word):', { table: [['word', 'd₁', 'd₂', 'd₃']].concat(EW.map((w, i) => [i + ': ' + w].concat(E[i].map((v) => sign(v))))) }, 'x is the sum of the one-hot vectors of “goat” and “bread”: x = [0, 1, 0, 1, 0]. Work out the product x · E.'],
    rows: ['x · E'], cols: ['d₁', 'd₂', 'd₃'], corner: '', answer: [LK], tolerance: 0.001, note: 'three numbers: the product of a 1 × 5 row and the 5 × 3 table',
    hints: ['Each number of the product is Σᵢ xᵢ · E[i][j]; only the rows where x is 1 count.', 'x picks rows 1 (goat) and 3 (bread), and adds them.', 'Row 1 + row 3: [0.5 + 0.1, 0.3 + 0, −0.2 + 0.6] = [' + LK.join(', ') + ']. A product with one-hot vectors is a lookup; with several, a sum of looked-up rows.'],
    solution: [LK.map(String)], wrong: [[E[1].map(String)], [E[1].map((v, j) => String(r4(v * E[3][j])))]] });

  const CA = [1, 2, 0, -1], CB = [2, 1, 1, 0];
  const COS = f64(() => { const a = T.tensor(CA), b = T.tensor(CB); return T.sum(T.mul(a, b)).item() / Math.sqrt(T.sum(T.square(a)).item() * T.sum(T.square(b)).item()); });
  ex({ id: 'sq-x-cosine', chapter: 'sq-embed', title: 'The cosine of two rows', level: 2, kind: 'number',
    text: ['Two embedding rows: a = [1, 2, 0, −1] and b = [2, 1, 1, 0]. What is their cosine similarity, a · b / (|a| |b|)? Give it to three decimals.'],
    answer: r3(COS), tolerance: 0.002, whyNot: { '4': 'That is the dot product a · b; divide it by the two lengths.', '1': 'Only rows pointing exactly the same way have cosine 1; these two do not.', '0.816': 'Check the lengths: |a|² = 1 + 4 + 0 + 1 and |b|² = 4 + 1 + 1 + 0.' },
    hints: ['First the dot product: multiply position by position and add.', 'a · b = 2 + 2 + 0 + 0 = 4. |a| = √6 and |b| = √6.', '4 / (√6 · √6) = 4 / 6 ≈ ' + r3(COS) + '. Cosine ignores the lengths and compares the directions only.'],
    solution: String(r3(COS)), wrong: ['4', '1'] });

  const SENT = 'the goat ate bread in the barn', toks = SENT.split(' '), at = toks.indexOf('bread');
  const ctx = [...new Set(toks.filter((_, i) => i !== at && Math.abs(i - at) <= 2))];
  ex({ id: 'sq-x-pairs', chapter: 'sq-embed', title: 'Which words are context?', level: 2, kind: 'set',
    text: ['word2vec reads the sentence “' + SENT + '” with a window of 2: each centre word is paired with every word up to 2 places before or after it. With “bread” as the centre, which words are its context words? Write them as a set, like {a, b}.'],
    answer: '{' + ctx.join(', ') + '}', caseless: true, placeholder: 'like {a, b}',
    hints: ['Count positions from “bread”: one and two places to the left, one and two to the right.', 'Two to the left: “goat”, “ate”. Two to the right: “in”, “the”.', '{' + ctx.join(', ') + '}: four pairs, (bread, goat), (bread, ate), (bread, in) and (bread, the). “barn” is three places away, outside the window.'],
    solution: '{' + ctx.join(', ') + '}', wrong: ['{ate, in}', '{the, goat, ate, in, barn}'] });

  const sg = (pos, negs) => f64(() => { const lp = (z) => T.log(T.sigmoid(T.tensor([z]))).item(); return -lp(pos) - negs.reduce((s, z) => s + lp(-z), 0); });
  const SG = r3(sg(2, [0, -1])), SGbad = r3(sg(2, [0, 1]));
  ex({ id: 'sq-x-sgns-loss', chapter: 'sq-embed', title: 'The loss of one pair', level: 3, kind: 'number',
    text: ['A real pair (centre c, context o) has score u_o · v_c = 2. Two negatives drawn at random have scores u₁ · v_c = 0 and u₂ · v_c = −1. What is the pair’s negative-sampling loss? Use natural logarithms and give three decimals.', { math: 'loss = −ln σ(u_o · v_c) − Σₖ ln σ(−u_k · v_c)        σ(z) = 1 / (1 + e^−z)' }],
    answer: SG, tolerance: 0.003, whyNot: { [String(SGbad)]: 'The negatives enter with a minus sign inside σ: σ(−u_k · v_c), so a score of −1 gives σ(1).', '0.127': 'That is only the real pair’s term; add the two negatives’ terms.' },
    hints: ['Three terms: one for the real pair, one for each negative.', 'Real pair: −ln σ(2) = 0.127. Negative 1: −ln σ(−0) = ln 2 = 0.693. Negative 2: −ln σ(−(−1)) = −ln σ(1).', '0.127 + 0.693 + 0.313 = ' + SG + '. The second negative already scores low (−1), so it costs little.'],
    solution: String(SG), wrong: [String(SGbad), '0.127'] });

  // word2vec on the part's corpus (S.P11.w2v(), the defaults: d = 16, window 2, 5 negatives, seed 1): the mean loss
  // per pair after each epoch, as the topic word2vec-train prints it
  const WL = [[1, 2.755], [2, 2.13], [3, 1.912], [4, 1.837], [5, 1.789], [6, 1.755], [7, 1.727], [8, 1.724]];
  ex({ id: 'sq-x-w2v-curve', chapter: 'sq-embed', title: 'The loss of word2vec, epoch by epoch', level: 2, kind: 'predict-curve',
    text: ['word2vec trains on the part’s corpus (5238 words, 16 numbers per word, a window of 2, 5 negatives). After the first epoch the mean loss per pair is 2.755. Sketch the mean loss after each epoch up to epoch 8.'],
    plot: { xr: [0.5, 8.5], yr: [1, 3], xname: 'epoch', yname: 'mean loss per pair', given: WL.slice(0, 1), note: 'Drag from epoch 1 to epoch 8. The faint point is epoch 1.' },
    answer: WL, tolerance: 0.15, checks: ['falling'],
    hints: ['The vectors start random; the first epochs fix the most.', 'Epoch 2 is already at 2.13; the falls get smaller and smaller after that.', 'It drops fast and then flattens: ' + WL.map(([e, v]) => 'epoch ' + e + ': ' + v).join(', ') + '. It cannot reach 0: the negatives are random words, and some of them do fit the context.'],
    solution: WL, wrong: [WL.map(([e]) => [e, r3(2.755 - (e - 1) * 0.15)]), WL.map(([e]) => [e, 2.755])] });

  // the cosines of man + boy − girl with some words, from S.P11.w2v() (seed 1): girl : boy = woman : ?
  const AN = [['woman', 0.985], ['man', 0.976], ['king', 0.948], ['queen', 0.912], ['boy', 0.803], ['girl', 0.725], ['bread', 0.045]];
  ex({ id: 'sq-x-analogy', chapter: 'sq-embed', title: 'Reading an analogy', level: 2, kind: 'choice',
    text: ['“girl is to boy as woman is to what?” The trained vectors give the query woman + boy − girl. Its cosines with some words of the vocabulary:', { table: [['word', 'cosine with the query']].concat(AN.map(([w, c]) => [w, String(c)])) }, 'Which word is the answer?'],
    options: ['woman', 'man', 'king', 'boy'], answer: 1,
    whyNot: { 0: '“woman” is one of the three words of the question: an analogy leaves them out, or the answer would nearly always be one of them.', 2: '“king” is close, 0.948, but “man” is closer.', 3: '“boy” is one of the words given.' },
    hints: ['The query is built from three words of the question. Should one of them count as the answer?', 'Leave out woman, boy and girl, then take the highest cosine.', '“man”, at 0.976. “woman” scores higher still (0.985), because the query is built on it, and that is why the given words are left out.'],
    solution: 1, wrong: [0, 2] });

  // =========================================================================================================
  // Chapter 2: recurrent networks
  // =========================================================================================================
  /** a plain engine RNN cell with set weights: Wx [I][H], Wh [H][H], b [H] */
  const rnnCell = (Wx, Wh, b) => { const c = new nn.RNN(Wx.length, b.length, { rng: 1 }); c.Wx.data.set(Wx.flat()); c.Wh.data.set(Wh.flat()); c.b.data.set(b); return c; };
  const rnnRun = (xs, Wx, Wh, b) => f64(() => T.noGrad(() => { const c = rnnCell(Wx, Wh, b), out = c.forward(T.tensor([xs])); return c.steps.map((s) => vals(s.h)); }));
  const RX = [[0.5, -1]], RH = [[1, 0.5], [-0.5, 1]], RB = [0, 0.2], H0 = [0.4, -0.2];
  const RPRE = f64(() => vals(T.add(T.add(T.matmul(T.tensor([[1]]), T.tensor(RX)), T.matmul(T.tensor([H0]), T.tensor(RH))), T.tensor(RB))).map(r3));
  const RSTEP = f64(() => T.noGrad(() => { const c = rnnCell(RX, RH, RB); c.forward(T.tensor([[[1]]]), T.tensor([H0])); return vals(c.steps[0].h).map(r3); }));
  const rcells = { 'before tanh|h₁': RPRE[0], 'before tanh|h₂': RPRE[1], 'h after tanh|h₁': RSTEP[0], 'h after tanh|h₂': RSTEP[1] };
  ex({ id: 'sq-x-rnn-step', chapter: 'sq-rnn', title: 'One step of a recurrent cell', level: 2, kind: 'table',
    text: ['A recurrent cell with 2 hidden numbers reads one input number. Its weights, with h as a row vector:', { math: 'W_x = [0.5, −1]\nW_h = [[1, 0.5], [−0.5, 1]]      (row 1, then row 2)\nb = [0, 0.2]\nh_t = tanh(x_t · W_x + h_{t−1} · W_h + b)' }, 'The previous state is h = [0.4, −0.2] and the input is x = 1. Fill in the sum before tanh and the new state, to three decimals.'],
    rows: ['before tanh', 'h after tanh'], cols: ['h₁', 'h₂'], corner: '', cells: rcells, tolerance: 0.003,
    hints: ['h · W_h multiplies the row [0.4, −0.2] by the matrix: the first number is 0.4 · 1 + (−0.2) · (−0.5).', 'x · W_x = [0.5, −1]; h · W_h = [0.5, 0]; b = [0, 0.2]. Add them.', 'Before tanh: [' + RPRE.map(sign).join(', ') + ']; after: [' + RSTEP.map(sign).join(', ') + '].'],
    solution: str(rcells), wrong: [str(Object.assign({}, rcells, { 'h after tanh|h₁': RPRE[0], 'h after tanh|h₂': RPRE[1] })), str({ 'before tanh|h₁': 0.9, 'before tanh|h₂': -0.6, 'h after tanh|h₁': r3(Math.tanh(0.9)), 'h after tanh|h₂': r3(Math.tanh(-0.6)) })] });

  const RP = f64(() => new nn.RNN(10, 20, { rng: 1 }).numParams());
  ex({ id: 'sq-x-rnn-params', chapter: 'sq-rnn', title: 'Weights of a recurrent cell', level: 1, kind: 'number',
    text: ['A plain recurrent cell reads inputs of 10 numbers and keeps a hidden state of 20. How many weights and biases does the cell have: W_x, W_h and b (no output layer)?'],
    answer: RP, tolerance: 0, whyNot: { '600': 'Add the bias: one per hidden number.', '220': 'W_h is 20 × 20: every hidden number reads every other.', '420': 'W_x is 10 × 20 = 200, not 0.' },
    hints: ['W_x maps 10 inputs to 20 hidden numbers; W_h maps 20 hidden numbers to 20.', 'W_x: 10 × 20 = 200. W_h: 20 × 20 = 400. b: 20.', '200 + 400 + 20 = ' + RP + ', the same for a sequence of 3 steps or 3000.'],
    solution: String(RP), wrong: ['600', '220'] });

  ex({ id: 'sq-x-share-choice', chapter: 'sq-rnn', title: 'Fifty steps, how many weights?', level: 1, kind: 'choice',
    text: ['A recurrent network reads a sentence of 50 words. Unrolled, it is 50 layers deep. Which is true of its state-to-state weights W_h?'],
    options: ['There is one W_h, used at all 50 steps; its gradient adds up a part from every step', 'There are 50 copies, one per step, each trained on its own', 'Only the last step’s W_h is trained', 'W_h is fixed; only W_x learns'], answer: 0,
    whyNot: { 1: 'Separate weights per step could not read a sentence longer than the ones seen in training.', 2: 'The loss depends on W_h through every step, so every step contributes to its gradient.', 3: 'W_h is learned like every other weight.' },
    hints: ['What makes a recurrent network able to read sequences of any length?', 'The same cell is applied at every step.', 'One W_h, shared: backpropagation through time adds up its gradient over all 50 steps.'],
    solution: 0, wrong: [1, 2] });

  const CODE_RNN = 'function rnnRun(xs, Wx, Wh, b) {\n  const H = b.length;\n  let h = new Array(H).fill(0);\n  const out = [];\n  for (const x of xs) {\n    const next = [];\n    for (let j = 0; j < H; j++) {\n      let s = b[j];\n      for (let i = 0; i < x.length; i++) s += x[i] * Wx[i][j];\n      for (let k = 0; k < H; k++) s += h[k] * Wh[k][j];\n      next.push(Math.tanh(s));\n    }\n    h = next;\n    out.push(h);\n  }\n  return out;\n}';
  ex({ id: 'sq-x-code-rnn', chapter: 'sq-rnn', title: 'Write a recurrent network’s forward pass', level: 3, kind: 'code',
    text: ['Write `rnnRun(xs, Wx, Wh, b)`. xs is a list of input vectors (each of I numbers), Wx is I × H, Wh is H × H, b has H numbers. Start from h = all zeros and, for each input in turn, set h = tanh(x · Wx + h · Wh + b), with x and h as row vectors. Return the list of states, one per input.'],
    fn: 'rnnRun', params: ['xs', 'Wx', 'Wh', 'b'], start: 'function rnnRun(xs, Wx, Wh, b) {\n  \n}', tolerance: 1e-6, ref: rnnRun,
    cases: [[[[1]], RX, RH, RB], [[[1], [0], [1]], RX, RH, RB], [[[1, 0], [0, 1], [1, 1]], [[0.3, -0.2, 0.1], [0.5, 0.4, -0.6]], [[0.2, -0.1, 0.3], [0.1, 0.4, 0], [-0.3, 0.2, 0.5]], [0.1, 0, -0.1]], [[[2], [-1], [0.5], [3]], [[1]], [[0.9]], [0]]],
    hints: ['Keep h between steps; the new h depends on the old one.', 'For hidden number j: b[j] + Σᵢ x[i] · Wx[i][j] + Σₖ h[k] · Wh[k][j], then tanh.', CODE_RNN],
    solution: CODE_RNN,
    wrong: ['function rnnRun(xs, Wx, Wh, b) { const H = b.length; let h = new Array(H).fill(0); const out = []; for (const x of xs) { const next = []; for (let j = 0; j < H; j++) { let s = b[j]; for (let i = 0; i < x.length; i++) s += x[i] * Wx[i][j]; next.push(Math.tanh(s)); } h = next; out.push(h); } return out; }',
      'function rnnRun(xs, Wx, Wh, b) { const H = b.length; let h = new Array(H).fill(0); const out = []; for (const x of xs) { const next = []; for (let j = 0; j < H; j++) { let s = b[j]; for (let i = 0; i < x.length; i++) s += x[i] * Wx[i][j]; for (let k = 0; k < H; k++) s += h[k] * Wh[j][k]; next.push(Math.tanh(s)); } h = next; out.push(h); } return out; }'] });

  // the gradient going back through a linear cell whose W_h is a rotation by 30° scaled by s
  const rot = (s) => { const c = Math.cos(Math.PI / 6), n = Math.sin(Math.PI / 6); return [[s * c, s * n], [-s * n, s * c]]; };
  const backSize = (s, g, k) => f64(() => { const h0 = T.param(T.tensor([[1, -0.5]])); let h = h0; for (let i = 0; i < k; i++) h = T.matmul(h, T.tensor(rot(s))); T.sum(T.mul(h, T.tensor([g]))).backward(); return Math.hypot(...vals(h0.grad)); });
  const GF = r3(backSize(0.9, [3, 4], 10));
  ex({ id: 'sq-x-grad-factor', chapter: 'sq-rnn', title: 'Ten steps back', level: 2, kind: 'number',
    text: ['A linear recurrent cell (no tanh) has W_h = a rotation by 30° scaled by 0.9. The gradient arriving at the last state is [3, 4], of size 5. Each step back multiplies it by W_hᵀ. What is its size 10 steps back? Give three decimals.'],
    answer: GF, tolerance: 0.002, whyNot: { '0.5': 'The scale multiplies once per step: 0.9 ten times, not 0.1 · 5.', '4.5': 'That is one step back.', '5': 'A rotation keeps the size, but the scale 0.9 does not.' },
    hints: ['A rotation does not change a vector’s size; the scale does.', 'Each step multiplies the size by exactly 0.9.', '5 · 0.9¹⁰ = 5 · 0.3487 = ' + GF + '. After 50 steps it would be 0.026: fading like sᵏ.'],
    solution: String(GF), wrong: ['4.5', '0.5'] });

  const GC = Array.from({ length: 31 }, (_, k) => [k, r3(Math.log10(backSize(1.2, [2, 2], k)))]);
  ex({ id: 'sq-x-grad-curve', chapter: 'sq-rnn', title: 'An exploding gradient on a log scale', level: 2, kind: 'predict-curve',
    text: ['A linear cell whose W_h is a rotation scaled by 1.2. The gradient at the last state is [2, 2], of size 2.83 (log₁₀ 2.83 = ' + GC[0][1] + '). Sketch log₁₀ of its size as it goes back 30 steps.'],
    plot: { xr: [0, 30], yr: [0, 3.2], xname: 'steps back', yname: 'log₁₀ of the gradient’s size', given: GC.slice(0, 1), note: 'Drag from 0 to 30 steps back. The faint point is the last state.' },
    answer: GC, tolerance: 0.2, checks: ['rising'],
    hints: ['Each step multiplies the size by exactly 1.2. What does multiplying do on a log scale?', 'It adds log₁₀ 1.2 = 0.079 per step: a straight line.', 'A straight line from ' + GC[0][1] + ' to ' + GC[30][1] + ' at 30 steps back (a size of about ' + Math.round(10 ** GC[30][1]) + ').'],
    solution: GC, wrong: [GC.map(([k]) => [k, r3(Math.log10(2.83 * (1 + 0.2 * k)))]), GC.map(([k]) => [k, r3(Math.log10(2.83 * 1.2 ** k) > 2 ? 2 : Math.log10(2.83 * 1.2 ** k))]).map(([k, v]) => [k, k > 15 ? r3(GC[15][1]) : v])] });

  // the factor dh₃/dh₀ of a scalar tanh cell, by autodiff, and each step's own factor
  const TW = 1.5, TXS = [1, -1, 0.5], TH0 = 0.5;
  const TF = f64(() => {
    const h0 = T.param(T.tensor([TH0])), hs = [];
    let h = h0;
    for (const x of TXS) { h = T.tanh(T.add(T.mul(TW, h), x)); hs.push(h); }
    h.backward();
    const H = hs.map((t) => t.item());
    return { H: H.map(r3), f: H.map((v) => r3(TW * (1 - v * v))), all: r4(h0.grad.item()) };
  });
  const tcells = { 'all three|factor': TF.all };
  TF.H.forEach((v, k) => { tcells['step ' + (k + 1) + '|h'] = v; tcells['step ' + (k + 1) + '|factor'] = TF.f[k]; });
  ex({ id: 'sq-x-tanh-factor', chapter: 'sq-rnn', title: 'The gradient through three tanh steps', level: 3, kind: 'table',
    text: ['A recurrent cell with a single number of state: h_t = tanh(1.5 · h_{t−1} + x_t). It starts from h₀ = 0.5 and reads x = 1, −1, 0.5. Fill in each state, each step’s factor ∂h_t/∂h_{t−1} = 1.5 · (1 − h_t²), and their product ∂h₃/∂h₀. Three decimals for the steps, four for the product.'],
    rows: ['step 1', 'step 2', 'step 3', 'all three'], cols: ['h', 'factor'], corner: 'step', cells: tcells, tolerance: 0.003,
    hints: ['Step 1: tanh(1.5 · 0.5 + 1) = tanh(1.75).', 'The slope of tanh at its output h is 1 − h²; the chain rule multiplies it by the weight 1.5. A state near ±1 gives a factor near 0.', 'h = ' + TF.H.map(sign).join(', ') + '; factors ' + TF.f.join(', ') + '; product ' + TF.all + '. Step 1’s state is near 1, so its factor is small, and that one step shrinks the whole product.'],
    solution: str(tcells), wrong: [str(Object.assign({}, tcells, { 'all three|factor': r4(TW ** 3) })), str(Object.assign({}, tcells, { 'step 1|factor': r3(1 - TF.H[0] * TF.H[0]), 'step 2|factor': r3(1 - TF.H[1] * TF.H[1]), 'step 3|factor': r3(1 - TF.H[2] * TF.H[2]) }))] });

  const CL = f64(() => { const p = T.param(T.tensor([0, 0, 0])); p.grad = T.tensor([3, -4, 12]); const before = T.clipGradNorm([p], 5); return { before, g: vals(p.grad).map(r3) }; });
  ex({ id: 'sq-x-clip', chapter: 'sq-rnn', title: 'Clipping a gradient', level: 2, kind: 'number',
    text: ['A gradient is [3, −4, 12]. Training clips gradients by their norm with a limit of 5: when the size passes 5, the whole vector is scaled down to size 5. What does the third number, 12, become? Three decimals.'],
    answer: CL.g[2], tolerance: 0.002, whyNot: { '5': 'That clips each number on its own (clipping by value). Clipping by norm scales the whole vector and keeps its direction.', '12': 'The size is ' + r3(CL.before) + ', above the limit, so it is scaled.', '2.4': 'Scale by 5 divided by the size, 13, not by 5.' },
    hints: ['First the size: √(3² + 4² + 12²).', 'The size is ' + r3(CL.before) + ', so every number is multiplied by 5 / ' + r3(CL.before) + '.', '12 · 5 / 13 = ' + CL.g[2] + '. The clipped gradient is [' + CL.g.map(sign).join(', ') + ']: the same direction, size 5.'],
    solution: String(CL.g[2]), wrong: ['5', '12'] });

  ex({ id: 'sq-x-cure-choice', chapter: 'sq-rnn', title: 'Losses that blow up', level: 1, kind: 'choice',
    text: ['A recurrent network trains well for a while, then its loss suddenly jumps to a huge value and becomes NaN. Logging shows the gradient’s size spiking to millions just before. What is the first thing to try?'],
    options: ['Clip the gradient’s norm', 'Raise the learning rate', 'Make the sequences longer', 'Remove the tanh'], answer: 0,
    whyNot: { 1: 'A larger step multiplies the damage of an exploding gradient.', 2: 'Longer sequences multiply more factors: more explosion, not less.', 3: 'Without tanh nothing bounds the state at all.' },
    hints: ['The gradient is exploding: its size grows like sᵏ with s above one.', 'One cure caps the size of each update without changing its direction.', 'Clip the gradient’s norm: whenever it passes a limit, scale it back to that limit.'],
    solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 3: gated cells
  // =========================================================================================================
  const LF = [0.9, 0.1], LI = [0.2, 0.95], LG = [0.5, -0.8], LO = [0.8, 0.6], LC = [2, 1];
  const LS = f64(() => { const c = T.add(T.mul(T.tensor(LF), T.tensor(LC)), T.mul(T.tensor(LI), T.tensor(LG))), h = T.mul(T.tensor(LO), T.tanh(c)); return { c: vals(c).map(r3), h: vals(h).map(r3) }; });
  const lcells = { 'slot 1|c': LS.c[0], 'slot 1|h': LS.h[0], 'slot 2|c': LS.c[1], 'slot 2|h': LS.h[1] };
  ex({ id: 'sq-x-lstm-cell', chapter: 'sq-gates', title: 'One LSTM step by hand', level: 2, kind: 'table',
    text: ['An LSTM with two memory slots. At this step its gates and candidate are already worked out:', { table: [['slot', 'forget f', 'input i', 'candidate g', 'output o', 'old memory c']].concat([0, 1].map((k) => ['slot ' + (k + 1), String(LF[k]), String(LI[k]), sign(LG[k]), String(LO[k]), String(LC[k])])) }, { math: 'c = f · c_old + i · g        h = o · tanh(c)' }, 'Fill in the new memory c and the output h of each slot, to three decimals.'],
    rows: ['slot 1', 'slot 2'], cols: ['c', 'h'], corner: '', cells: lcells, tolerance: 0.003,
    hints: ['Slot 1: keep 0.9 of the old 2 and add 0.2 of the candidate 0.5.', 'Slot 1: c = 1.8 + 0.1 = 1.9. Slot 2: c = 0.1 · 1 + 0.95 · (−0.8).', 'Slot 1: c = ' + LS.c[0] + ', h = 0.8 · tanh(1.9) = ' + LS.h[0] + '. Slot 2: c = ' + sign(LS.c[1]) + ', h = ' + sign(LS.h[1]) + '. Slot 1 keeps its memory; slot 2 forgets it and writes the candidate.'],
    solution: str(lcells), wrong: [str(Object.assign({}, lcells, { 'slot 1|h': r3(0.8 * 1.9), 'slot 2|h': r3(0.6 * LS.c[1]) })), str({ 'slot 1|c': r3(0.9 * 0.5 + 0.2 * 2), 'slot 1|h': LS.h[0], 'slot 2|c': LS.c[1], 'slot 2|h': LS.h[1] })] });

  ex({ id: 'sq-x-lstm-keep', chapter: 'sq-gates', title: 'Keeping a memory untouched', level: 1, kind: 'choice',
    text: ['In one memory slot of an LSTM, which gate values keep the memory c exactly as it was, whatever the input is?'],
    options: ['forget gate 1, input gate 0', 'forget gate 0, input gate 1', 'forget gate 1, input gate 1', 'output gate 0'], answer: 0,
    whyNot: { 1: 'That throws the old memory away and writes the candidate instead.', 2: 'That keeps the old memory and adds the candidate to it.', 3: 'The output gate hides the memory from the output; it does not change c.' },
    hints: ['c = f · c_old + i · g.', 'To keep c_old whole, f must be 1; to add nothing, i must be 0.', 'Forget gate 1 and input gate 0: c = 1 · c_old + 0 · g = c_old.'],
    solution: 0, wrong: [1, 3] });

  const PF = [0.9, 0.95, 1, 0.8, 0.99, 0.9];
  const PATH = f64(() => { const c0 = T.param(T.tensor([0.5])); let c = c0; PF.forEach((f) => { c = T.add(T.mul(f, c), 0.3); }); c.backward(); return r4(c0.grad.item()); });
  ex({ id: 'sq-x-lstm-path', chapter: 'sq-gates', title: 'The gradient along the memory', level: 2, kind: 'number',
    text: ['Along an LSTM’s memory, c_t = f_t · c_{t−1} + i_t · g_t. Take the gates and the candidates as given (do not differentiate through them). Over six steps the forget gate of one slot is 0.9, 0.95, 1, 0.8, 0.99 and 0.9. What is ∂c₆/∂c₀? Four decimals.'],
    answer: PATH, tolerance: 0.0006, whyNot: { '0.9': 'That is one step; the chain rule multiplies all six factors.', '5.54': 'The factors multiply, they do not add.' },
    hints: ['∂c_t/∂c_{t−1} = f_t: the i · g term does not depend on c_{t−1}.', 'Chain the six steps: multiply the six forget gates.', '0.9 · 0.95 · 1 · 0.8 · 0.99 · 0.9 = ' + PATH + '. No weight matrix and no tanh slope on this path: with gates near 1, the gradient survives.'],
    solution: String(PATH), wrong: ['0.9', '5.54'] });

  const GU = f64(() => { const z = T.tensor([0.8]), h = T.tensor([0.6]), n = T.tensor([-0.5]); return r3(T.add(T.mul(T.sub(1, z), n), T.mul(z, h)).item()); });
  const GUbad = r3(0.2 * 0.6 + 0.8 * -0.5);
  ex({ id: 'sq-x-gru-update', chapter: 'sq-gates', title: 'A GRU keeps or rewrites', level: 2, kind: 'number',
    text: ['In one number of a GRU’s state, the old value is 0.6, the candidate is −0.5 and the update gate is z = 0.8. With h = z · h_old + (1 − z) · n, what is the new value? Three decimals.'],
    answer: GU, tolerance: 0.002, whyNot: { [String(GUbad)]: 'That gives the 0.8 to the candidate; here z is the share of the old value that is kept.', '0.48': 'Add the candidate’s share too: (1 − z) · n.' },
    hints: ['z is the share kept from the old value; 1 − z goes to the candidate.', '0.8 · 0.6 = 0.48; 0.2 · (−0.5) = −0.1.', '0.48 − 0.1 = ' + GU + '. The state moved a fifth of the way towards the candidate.'],
    solution: String(GU), wrong: [String(GUbad), '0.48'] });

  ex({ id: 'sq-x-gru-vs-lstm', chapter: 'sq-gates', title: 'GRU against LSTM', level: 2, kind: 'choice',
    text: ['Which of these are true of a GRU compared with an LSTM of the same state size? Pick all that are true.'],
    options: ['It has no separate memory: its state is the memory', 'It has two gates instead of three', 'It has fewer weights: three blocks of weights against four', 'It cannot keep a value for more than a few steps'], answer: [0, 1, 2],
    whyNot: { 3: 'Its update gate near 1 holds a value for as long as needed: the topic holds a letter over eight blanks.' },
    hints: ['Count the gates of each, and what each keeps between steps.', 'LSTM: forget, input and output gates plus a candidate, and a memory c beside h. GRU: reset and update gates plus a candidate.', 'The first three are true. The fourth is not: with z near 1 a GRU holds its state for long stretches.'],
    solution: [0, 1, 2], wrong: [[0, 1, 2, 3], [1, 2]] });

  const gruRef = (x, h, p) => f64(() => T.noGrad(() => {
    const g = new nn.GRU(1, 1, { rng: 1 });
    g.Wx.data.set([p.wr, p.wz, p.wn]); g.Wh.data.set([p.ur, p.uz, p.un]); g.b.data.set([p.br, p.bz, p.bn]); g.bh.data.set([0, 0, 0]);
    g.forward(T.tensor([[[x]]]), T.tensor([[h]]));
    return g.steps[0].h.item();
  }));
  const GP = [{ wr: 0.5, ur: -0.3, br: 0.1, wz: 1.2, uz: 0.4, bz: -0.5, wn: 0.8, un: -1.1, bn: 0.2 }, { wr: -1, ur: 2, br: 0, wz: -2, uz: 0.5, bz: 1, wn: 1.5, un: 0.7, bn: -0.3 }, { wr: 0, ur: 0, br: 0, wz: 0, uz: 0, bz: 0, wn: 1, un: 1, bn: 0 }];
  const CODE_GRU = 'function gruStep(x, h, p) {\n  const sig = (v) => 1 / (1 + Math.exp(-v));\n  const r = sig(p.wr * x + p.ur * h + p.br);\n  const z = sig(p.wz * x + p.uz * h + p.bz);\n  const n = Math.tanh(p.wn * x + p.bn + r * (p.un * h));\n  return z * h + (1 - z) * n;\n}';
  ex({ id: 'sq-x-code-gru', chapter: 'sq-gates', title: 'Write a GRU step', level: 3, kind: 'code',
    text: ['Write `gruStep(x, h, p)` for a GRU with a single number of state: x is the input, h the old state, and p holds the weights {wr, ur, br, wz, uz, bz, wn, un, bn}. Return the new state.', { math: 'r = σ(wr·x + ur·h + br)\nz = σ(wz·x + uz·h + bz)\nn = tanh(wn·x + bn + r · (un·h))\nh_new = z·h + (1 − z)·n' }],
    fn: 'gruStep', params: ['x', 'h', 'p'], start: 'function gruStep(x, h, p) {\n  \n}', tolerance: 1e-6, ref: gruRef,
    cases: [[1, 0.5, GP[0]], [-0.5, -0.2, GP[0]], [2, 0.9, GP[1]], [0, 0, GP[1]], [1, -1, GP[2]], [0.3, 0.7, GP[2]]],
    hints: ['σ(v) = 1 / (1 + e^−v); work out r and z first.', 'The reset gate multiplies only the old state’s part inside the candidate: r · (un·h).', CODE_GRU],
    solution: CODE_GRU,
    wrong: ['function gruStep(x, h, p) { const sig = (v) => 1 / (1 + Math.exp(-v)); const r = sig(p.wr * x + p.ur * h + p.br); const z = sig(p.wz * x + p.uz * h + p.bz); const n = Math.tanh(p.wn * x + p.bn + r * (p.un * h)); return (1 - z) * h + z * n; }',
      'function gruStep(x, h, p) { const sig = (v) => 1 / (1 + Math.exp(-v)); const z = sig(p.wz * x + p.uz * h + p.bz); const n = Math.tanh(p.wn * x + p.bn + p.un * h); return z * h + (1 - z) * n; }'] });

  // =========================================================================================================
  // Chapter 4: sequence models at work
  // =========================================================================================================
  const CHL = f64(() => r3(-T.log(T.tensor([0.64])).item()));
  ex({ id: 'sq-x-char-loss', chapter: 'sq-seq', title: 'The loss of one character', level: 1, kind: 'number',
    text: ['After “t” the character model gives “h” a probability of 0.64, and “h” is indeed the next character. What is the loss at this step, −ln p? Three decimals.'],
    answer: CHL, tolerance: 0.002, whyNot: { '0.36': 'That is 1 − p; the loss is −ln p.', '0.194': 'That uses log base 10; the loss uses the natural logarithm.', '0.64': 'That is the probability itself.' },
    hints: ['The loss is the negative natural logarithm of the probability given to what really came.', 'ln 0.64 is negative, so −ln 0.64 is positive.', '−ln 0.64 = ' + CHL + '. A sure right guess (p = 1) would cost 0; p = 0.04 costs 3.22.'],
    solution: String(CHL), wrong: ['0.36', '0.194'] });

  const PP = [0.5, 0.25, 0.8, 0.1];
  const PX = f64(() => { const m = T.sum(T.neg(T.log(T.tensor(PP)))).item() / PP.length; return { m: r3(m), p: r3(Math.exp(m)) }; });
  const pcells = { 'mean loss|value': PX.m, 'perplexity|value': PX.p };
  ex({ id: 'sq-x-perplexity', chapter: 'sq-seq', title: 'Loss and perplexity of four characters', level: 2, kind: 'table',
    text: ['Over four characters of a test text, the model gave the real next characters probabilities 0.5, 0.25, 0.8 and 0.1. Fill in the mean loss (natural logarithm) and the perplexity, e^(mean loss), to three decimals.'],
    rows: ['mean loss', 'perplexity'], cols: ['value'], corner: '', cells: pcells, tolerance: 0.003,
    hints: ['The four losses are −ln 0.5, −ln 0.25, −ln 0.8 and −ln 0.1.', 'They are 0.693, 1.386, 0.223 and 2.303.', 'Mean ' + PX.m + '; perplexity e^' + PX.m + ' = ' + PX.p + ': as unsure as choosing among about three equally likely characters.'],
    solution: str(pcells), wrong: [str({ 'mean loss|value': PX.m, 'perplexity|value': r3(1 / ((0.5 + 0.25 + 0.8 + 0.1) / 4)) }), str({ 'mean loss|value': r3(4 * PX.m), 'perplexity|value': PX.p })] });

  const LOG3 = [2, 1, 0];
  const soft = (z, t) => f64(() => vals(T.softmax(T.tensor(z.map((v) => v / t)))));
  const TM = [soft(LOG3, 1).map(r3), soft(LOG3, 0.5).map(r3)];
  ex({ id: 'sq-x-temp', chapter: 'sq-seq', title: 'Probabilities at two temperatures', level: 2, kind: 'matrix',
    text: ['A model’s scores for three possible next characters are a: 2, b: 1, c: 0. Work out the probabilities at temperature 1 and at temperature 0.5, p_k = exp(z_k / T) / Σⱼ exp(z_j / T). Two decimals are enough.'],
    rows: ['T = 1', 'T = 0.5'], cols: ['a', 'b', 'c'], corner: 'temperature', answer: TM, tolerance: 0.01, rowsSumTo: 1,
    hints: ['At T = 0.5 the scores are first divided by 0.5: 4, 2 and 0.', 'T = 1: e² = 7.39, e¹ = 2.72, e⁰ = 1; divide each by their sum, 11.11.', 'T = 1: ' + TM[0].join(', ') + '. T = 0.5: ' + TM[1].join(', ') + '. The colder temperature gives the favourite more.'],
    solution: TM.map((r) => r.map(String)), wrong: [[TM[0].map(String), soft(LOG3, 2).map((v) => String(r3(v)))], [TM[0].map(String), TM[0].map(String)]] });

  ex({ id: 'sq-x-temp-choice', chapter: 'sq-seq', title: 'Hot or cold', level: 1, kind: 'choice',
    text: ['A character model writes a story by sampling. Which change makes its text more varied, at the risk of misspelt words?'],
    options: ['Raise the temperature', 'Lower the temperature towards 0', 'Always take the likeliest character', 'Lower the temperature to 1 from 1.5'], answer: 0,
    whyNot: { 1: 'A low temperature sharpens the probabilities: the text gets safer and more repetitive.', 2: 'That is greedy writing: no variety at all.', 3: 'That also sharpens the choice.' },
    hints: ['Temperature divides the scores before the softmax.', 'Dividing by a larger number brings the scores closer together, so the probabilities flatten.', 'Raise the temperature: unlikely characters get drawn more often, for more variety and more mistakes.'],
    solution: 0, wrong: [1, 2] });

  const CODE_T = 'function withTemperature(z, t) {\n  const s = z.map((v) => v / t);\n  const m = Math.max(...s);\n  const e = s.map((v) => Math.exp(v - m));\n  const sum = e.reduce((a, b) => a + b, 0);\n  return e.map((v) => v / sum);\n}';
  ex({ id: 'sq-x-code-temp', chapter: 'sq-seq', title: 'Write softmax with a temperature', level: 3, kind: 'code',
    text: ['Write `withTemperature(z, t)`: the probabilities exp(z_k / t) / Σⱼ exp(z_j / t) for a list of scores z and a temperature t > 0. It must not overflow for scores near 1000: subtract the largest scaled score before exponentiating.'],
    fn: 'withTemperature', params: ['z', 't'], start: 'function withTemperature(z, t) {\n  \n}', tolerance: 1e-9, ref: (z, t) => soft(z, t),
    cases: [[[2, 1, 0], 1], [[2, 1, 0], 0.5], [[2, 1, 0], 2], [[0, 0, 0, 0], 0.7], [[1000, 999, 800], 1], [[750, 740, 0], 0.9], [[-1, 3, 0.5], 0.1]],
    hints: ['Divide every score by t first.', 'Take the largest of the divided scores away from each before Math.exp, so the largest exponent is 0.', CODE_T],
    solution: CODE_T,
    wrong: ['function withTemperature(z, t) { const e = z.map((v) => Math.exp(v / t)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); }',
      'function withTemperature(z, t) { const s = z.map((v) => v * t); const m = Math.max(...s); const e = s.map((v) => Math.exp(v - m)); const sum = e.reduce((a, b) => a + b, 0); return e.map((v) => v / sum); }'] });

  const LENS = ((S.P11D && S.P11D.seq2seq && S.P11D.seq2seq.lengths) || []).map((x) => [x.L, x.exact]);
  ex({ id: 'sq-x-s2s-curve', chapter: 'sq-seq', title: 'Longer and longer inputs', level: 2, kind: 'predict-curve',
    text: ['The encoder–decoder of the lesson was trained on inputs of one to 5 number words. On 100 new inputs of each length, it gets every one of 1, 2 and 3 words exactly right. Sketch the share exactly right as the input grows to 8 words.'],
    plot: { xr: [0.5, 8.5], yr: [0, 1.05], xname: 'input length (words)', yname: 'share exactly right', given: LENS.slice(0, 3), note: 'Drag from 1 word to 8. The faint points are the lengths already known.' },
    answer: LENS, tolerance: 0.1, checks: ['falling'],
    hints: ['Within the lengths it trained on it does well; the longest trained inputs are the hardest of those.', 'At 5 words, the longest it saw, it is right ' + Math.round(LENS[4][1] * 100) + '% of the time.', 'It stays near 1 up to 4 words, then drops: ' + LENS.map(([L, v]) => L + ': ' + v).join(', ') + '. One vector of 32 numbers cannot hold an input longer than it was trained on.'],
    solution: LENS, wrong: [LENS.map(([L]) => [L, 1]), LENS.map(([L]) => [L, r3(1 - (L - 1) / 7)])] });

  ex({ id: 'sq-x-bottleneck-choice', chapter: 'sq-seq', title: 'Why long inputs fail', level: 1, kind: 'choice',
    text: ['The encoder–decoder writes 4-word inputs right 97% of the time and 8-word inputs never. What limits it, and what is the usual fix?'],
    options: ['Everything passes through one fixed context vector; attention lets the decoder look back at every encoder state', 'The vocabulary is too small; add more digits', 'The decoder writes too fast; slow it down', 'Eight is an unlucky length; nothing can be done'], answer: 0,
    whyNot: { 1: 'The same ten digits are enough for any length.', 2: 'The decoder writes one token per step whatever happens.', 3: 'Models with attention copy long inputs well.' },
    hints: ['What does the decoder receive from the encoder?', 'Only the encoder’s last state, 32 numbers, whatever the input’s length.', 'The fixed context vector is a bottleneck. Attention lets the decoder read every encoder state at every step.'],
    solution: 0, wrong: [1, 3] });

  ex({ id: 'sq-x-model-choice', chapter: 'sq-seq', title: 'Counts or order?', level: 1, kind: 'choice',
    text: ['A shop collects reviews like “lovely to look at, but it stopped working in a week” and “slow to arrive, but worth every penny”. Which model can tell such reviews apart best?'],
    options: ['A recurrent network reading the words in order', 'Logistic regression on word counts', 'Naive Bayes on word counts', 'k nearest neighbours on word counts'], answer: 0,
    whyNot: { 1: 'Word counts lose the order: “good, but bad” and “bad, but good” give the same row.', 2: 'Naive Bayes also reads counts only.', 3: 'The neighbours are found in the same counts, without order.' },
    hints: ['Both reviews mix good and bad words. What decides each verdict?', 'The part after “but” decides: order matters.', 'A recurrent network: it reads in order, so it can weigh what comes after the turn.'],
    solution: 0, wrong: [1, 2] });

  // reviews with a turn: a simple RNN reading 24 words reaches 0.844 on the held-out reviews; reading only the first
  // 8 words it gets 0.622, and 12 words 0.672 (S.Challenges.evaluate on these workflows)
  const ml = { data: { set: 'opinions', n: 0, seed: 3 }, hidden: { seed: 31, n: 180 }, metric: 'accuracy', target: 0.8 };
  ex(Object.assign({ id: 'sq-x-wf-maxlen', chapter: 'sq-seq', title: 'Read to the end', level: 3, kind: 'workflow',
    text: ['Reviews with a turn: the verdict often comes after a “but”. The starter trains a simple RNN that reads only the first 8 words of each review. Change it so that it classifies the held-out reviews with accuracy at least 0.8.'],
    limits: { blocks: ['split', 'rnn', 'train', 'evaluate'], models: ['rnn'], epochs: 30, maxBlocks: 5 },
    starter: () => pipe(ml.data, 'rnn', { cell: 'RNN', maxLen: 8 }, { epochs: 10 }),
    hints: ['Run the starter and open the RNN block: how many words of each review does it read?', 'Most turns come after the eighth word. The block’s “longest text read” sets how many words it sees.', 'RNN with the longest text read at 24 words, Train 10 epochs: accuracy 0.844. At 8 words it gets 0.622, at 12 words 0.672.'],
    solution: pipe(ml.data, 'rnn', { cell: 'RNN', maxLen: 24 }, { epochs: 10 }),
    wrong: [pipe(ml.data, 'rnn', { cell: 'RNN', maxLen: 8 }, { epochs: 10 }), pipe(ml.data, 'rnn', { cell: 'RNN', maxLen: 12 }, { epochs: 10 })] }, ml));
})(globalThis.Nooral = globalThis.Nooral || {});
