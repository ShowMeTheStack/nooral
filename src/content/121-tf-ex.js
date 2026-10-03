// Part 12, transformers: the exercises of the first two chapters (tokens, positions, attention, scaling, masks and
// heads); the rest are in 121-tf-ex2.js. The small matrices are worked out here by the tensor engine when the file loads
// (S.T.attention with and without its masks, the softmax, nn.sinusoid, in double precision), so the text, the answers
// and the hints agree. Figures quoted from the trained tiny transformers were printed by the topics themselves; the
// curve of the largest weight is measured by the scaling topic's own function, S.P12.scaleStats.
(function (S) {
  'use strict';
  const C = S.Content, T = S.T, P = S.P12;
  if (!C || !T || !P || C.exercises.has('tf-x-emb-size')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const sign = (v) => String(v).replace(/^-/, '−');
  const f64 = (fn) => T.float64(fn);
  const arr = (t) => t.toArray();
  const ex = (x) => C.exercise(x);
  const mtab = (rows, M, head) => ({ table: [head].concat(M.map((r, i) => [rows[i]].concat(r.map(sign)))) });
  const vec = (v) => '(' + v.map(sign).join(', ') + ')';
  const strM = (M) => M.map((r) => r.map(String));

  // =========================================================================================================
  // Chapter 1: tokens, positions and attention
  // =========================================================================================================
  const NET = P.net('reverse'), NE = NET.emb.W.size, NP = NET.pos.P.size;
  ex({ id: 'tf-x-emb-size', chapter: 'tf-read', title: 'Two tables of learned numbers', level: 1, kind: 'number',
    text: ['The tiny transformer of this part has a vocabulary of 9 tokens and is 16 numbers wide. It learns an embedding table (one row per token) and a position table (one row per position, for sequences of up to 11 tokens). How many learned numbers do the two tables hold together?'],
    answer: NE + NP, tolerance: 0,
    whyNot: { [String(NE)]: 'That is only the embedding table; the position table has a row of 16 for each of 11 positions too.', [String(NP)]: 'That is only the position table.', '25': 'Each table has a row per token or position, and every row holds 16 numbers.', [String(9 * 11 * 16)]: 'The tables are separate: 9 rows and 11 rows, not 9 × 11.' },
    hints: ['Each table has one row of 16 numbers per entry.', 'Embedding: 9 × 16. Positions: 11 × 16.', '9 × 16 = ' + NE + ' and 11 × 16 = ' + NP + ': ' + (NE + NP) + ' in all.'],
    solution: String(NE + NP), wrong: [String(NE), '25'] });

  const PE = f64(() => arr(T.nn.sinusoid(3, 4))[2].map(r3));
  const pcells = { 'p = 2|dim 0': PE[0], 'p = 2|dim 1': PE[1], 'p = 2|dim 2': PE[2], 'p = 2|dim 3': PE[3] };
  const str = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));
  ex({ id: 'tf-x-sinusoid', chapter: 'tf-read', title: 'A sinusoidal position vector', level: 2, kind: 'table',
    text: ['Work out the sinusoidal position vector of position p = 2 for a width of d = 4, to three decimals. Angles are in radians.', { math: 'PE(p, 2i)   = sin(p / 10000^(2i/d))\nPE(p, 2i+1) = cos(p / 10000^(2i/d))' }],
    rows: ['p = 2'], cols: ['dim 0', 'dim 1', 'dim 2', 'dim 3'], corner: '', cells: pcells, tolerance: 0.002,
    hints: ['Dimensions 0 and 1 use i = 0, so the angle is p / 10000⁰ = 2.', 'Dimensions 2 and 3 use i = 1: the angle is 2 / 10000^(2/4) = 2 / 100 = 0.02.', 'sin 2 = ' + sign(PE[0]) + ', cos 2 = ' + sign(PE[1]) + ', sin 0.02 = ' + sign(PE[2]) + ', cos 0.02 = ' + sign(PE[3]) + '. The first pair turns fast with p, the second barely moves.'],
    solution: str(pcells), wrong: [str({ 'p = 2|dim 0': PE[0], 'p = 2|dim 1': PE[1], 'p = 2|dim 2': PE[0], 'p = 2|dim 3': PE[1] }), str({ 'p = 2|dim 0': r3(Math.sin(2 * Math.PI / 180)), 'p = 2|dim 1': r3(Math.cos(2 * Math.PI / 180)), 'p = 2|dim 2': PE[2], 'p = 2|dim 3': PE[3] })] });

  ex({ id: 'tf-x-order', chapter: 'tf-read', title: 'Shuffled words, no positions', level: 1, kind: 'choice',
    text: ['An attention layer with no position vectors reads “dog bites man”, then “man bites dog”. What is true of its outputs?'],
    options: ['Each word gets the same output in both; the outputs are only listed in another order', 'Every output changes, because the sentence means something else', 'The layer refuses to run on a reordered sentence', 'Only the first word’s output changes'], answer: 0,
    whyNot: { 1: 'Nothing in the computation depends on where a row stands: the scores of “dog” with each word are the same in both.', 2: 'Attention runs on any set of rows.', 3: 'No position is special without position vectors.' },
    hints: ['Each output is a weighted sum of values, with weights from query–key scores.', 'Moving a row moves its query, key and value with it; no score changes.', 'Without positions attention treats its input as a set: shuffle the rows and the outputs are shuffled the same way. That is why position vectors are added.'],
    solution: 0, wrong: [1, 3] });

  // the running example: three positions, dₖ = 4
  const Q = [[1, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0]], K = [[1, 1, 0, 0], [0, 0, 2, 1], [1, 0, 0, 1]], V = [[1, 0], [0, 2], [3, 1]];
  const W3 = ['a', 'b', 'c'];
  const AT = f64(() => { const a = T.attention(T.tensor(Q), T.tensor(K), T.tensor(V)); return { raw: arr(T.matmul(T.tensor(Q), T.transpose(T.tensor(K)))), sc: arr(a.attn.scores), w: arr(a.attn.weights), out: arr(a) }; });
  const SC = AT.sc.map((r) => r.map(r3)), WT = AT.w.map((r) => r.map(r2));
  const qkText = ['Three positions a, b and c have these queries and keys (dₖ = 4):', mtab(W3, Q, ['', 'q₁', 'q₂', 'q₃', 'q₄']), mtab(W3, K, ['', 'k₁', 'k₂', 'k₃', 'k₄'])];
  ex({ id: 'tf-x-scores', chapter: 'tf-read', title: 'The scaled scores', level: 1, kind: 'matrix',
    text: qkText.concat(['Fill in the scaled scores qᵢ · kⱼ / √dₖ: row i is the query of position i, column j the key of position j.']),
    rows: W3, cols: W3, corner: 'query \\ key', answer: SC, tolerance: 0.005, note: 'Scaled scores: each dot product divided by √4 = 2.',
    hints: ['A dot product multiplies matching numbers and adds: q_a · k_a = 1·1 + 0·1 + 1·0 + 0·0.', 'The raw scores Q Kᵀ are ' + AT.raw.map((r) => '[' + r.join(', ') + ']').join(', ') + ', row by row. Now divide by √4.', 'Divide each raw score by 2: ' + SC.map((r) => '[' + r.join(', ') + ']').join(', ') + '.'],
    solution: strM(SC), wrong: [strM(AT.raw), strM(AT.raw.map((r) => r.map((v) => v / 4)))] });

  ex({ id: 'tf-x-weights', chapter: 'tf-read', title: 'From scores to weights', level: 2, kind: 'matrix',
    text: ['The scaled scores of three positions (rows: queries, columns: keys):', mtab(W3, SC, ['', 'a', 'b', 'c']), 'Turn each row into attention weights with a softmax: eˢ divided by the row’s total. Two decimals.'],
    rows: W3, cols: W3, corner: 'query \\ key', answer: WT, tolerance: 0.01, rowsSumTo: 1,
    hints: ['Row a: e^0.5 = 1.649, e^1 = 2.718, e^0.5 = 1.649.', 'Their total is 6.016, so row a is (1.649, 2.718, 1.649) / 6.016.', 'Row a: (' + WT[0].join(', ') + '); row b: (' + WT[1].join(', ') + '); row c: (' + WT[2].join(', ') + '). Each row adds up to 1, and a larger score always gets a larger weight.'],
    solution: strM(WT), wrong: [strM(SC.map((r) => { const s = r.reduce((a, b) => a + b, 0); return r.map((v) => r2(v / s)); })), strM(SC.map((r) => r.map((v) => r2(Math.exp(v)))))] });

  const AW = [[0.5, 0.25, 0.25], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]];
  const AO = f64(() => arr(T.matmul(T.tensor(AW), T.tensor(V))).map((r) => r.map(r3)));
  ex({ id: 'tf-x-output', chapter: 'tf-read', title: 'The weighted sum of the values', level: 1, kind: 'matrix',
    text: ['Three positions have these attention weights (rows add up to 1) and values:', mtab(W3, AW, ['weights', 'a', 'b', 'c']), mtab(W3, V, ['value', 'v₁', 'v₂']), 'Fill in the output of every position: its weights times the values, A V.'],
    rows: W3, cols: ['out₁', 'out₂'], corner: 'position', answer: AO, tolerance: 0.005,
    hints: ['The output of a is 0.5 · v_a + 0.25 · v_b + 0.25 · v_c.', 'First number of a: 0.5 · 1 + 0.25 · 0 + 0.25 · 3 = 1.25.', 'a: (' + AO[0].join(', ') + '); b: (' + AO[1].join(', ') + '); c: (' + AO[2].join(', ') + '). c, which keeps 0.8 for the value of c, ends up close to (3, 1).'],
    solution: strM(AO), wrong: [strM(V.map((r) => r.map(String))), strM(AW.map((r) => [r2(r[0] * V[0][0] + r[1] * V[0][1]), r2(r[0] * V[1][0] + r[1] * V[1][1])]))] });

  const q1 = [2, 0, 1, 1], K1 = [[1, 0, 2, 0], [0, 1, 0, 1], [2, 1, 1, 1]], V1 = [[2, 0], [0, 4], [1, 1]];
  const OQ = f64(() => { const a = T.attention(T.tensor([q1]), T.tensor(K1), T.tensor(V1)); return { raw: arr(T.matmul(T.tensor([q1]), T.transpose(T.tensor(K1))))[0], w: arr(a.attn.weights)[0], out: arr(a)[0] }; });
  const OUT = OQ.out.map(r2);
  ex({ id: 'tf-x-one-query', chapter: 'tf-read', title: 'One query, start to finish', level: 3, kind: 'matrix',
    text: ['One query q = ' + vec(q1) + ' attends to three keys and their values:', mtab(['1', '2', '3'], K1, ['key', 'k₁', 'k₂', 'k₃', 'k₄']), mtab(['1', '2', '3'], V1, ['value', 'v₁', 'v₂']), 'Work out the output of scaled dot-product attention for this query, to two decimals.'],
    rows: ['output'], cols: ['out₁', 'out₂'], corner: '', answer: [OUT], tolerance: 0.01, note: 'Two numbers: the weighted sum of the three values.',
    hints: ['Scores first: q · k₁ = 2·1 + 0·0 + 1·2 + 1·0 = ' + OQ.raw[0] + '. Then divide all three by √4 = 2.', 'Raw scores (' + OQ.raw.join(', ') + '), scaled (' + OQ.raw.map((v) => v / 2).join(', ') + '); softmax weights (' + OQ.w.map((v) => r3(v)).join(', ') + ').', 'Output = ' + OQ.w.map((v, i) => r3(v) + ' · v' + '₁₂₃'[i]).join(' + ') + ' = (' + OUT.join(', ') + '). Key ' + (P.argmax(OQ.raw) + 1) + ', with the largest score, pulls the output towards its value ' + vec(V1[P.argmax(OQ.raw)]) + '.'],
    solution: [OUT.map(String)], wrong: [[f64(() => arr(T.attention(T.tensor([q1]), T.tensor(K1), T.tensor(V1), { scale: 1 }))[0]).map((v) => String(r2(v)))], [V1[2].map(String)]] });

  // the reference for the attention code exercise
  const attendRef = (Qm, Km, Vm) => f64(() => arr(T.attention(T.tensor(Qm), T.tensor(Km), T.tensor(Vm))));
  const CODE_ATT = 'function attend(Q, K, V) {\n  const dk = Q[0].length;\n  return Q.map((q) => {\n    const s = K.map((k) => k.reduce((t, kj, j) => t + kj * q[j], 0) / Math.sqrt(dk));\n    const m = Math.max(...s);\n    const e = s.map((v) => Math.exp(v - m));\n    const z = e.reduce((a, b) => a + b, 0);\n    const w = e.map((v) => v / z);\n    return V[0].map((_, c) => w.reduce((t, wi, i) => t + wi * V[i][c], 0));\n  });\n}';
  ex({ id: 'tf-x-code-attn', chapter: 'tf-read', title: 'Write scaled dot-product attention', level: 3, kind: 'code',
    text: ['Write `attend(Q, K, V)`. Q is a list of queries and K a list of keys, all of dₖ numbers; V holds one value (a list of numbers) per key. Return one output per query: softmax(q · kⱼ / √dₖ over j) times the values. Subtract the row’s largest score before exponentiating, so large scores do not overflow.'],
    fn: 'attend', params: ['Q', 'K', 'V'], start: 'function attend(Q, K, V) {\n  \n}', tolerance: 1e-6, ref: attendRef,
    cases: [[Q, K, V], [[q1], K1, V1], [[[1, 0]], [[1, 0], [0, 1]], [[1], [0]]], [[[30, 40, 0, 10]], [[20, 10, 0, 5], [0, 30, 10, 0]], [[1, 2, 3], [4, 5, 6]]], [[[0.5, -1], [2, 0.3]], [[1, 1], [-1, 0.5], [0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]],
    hints: ['For each query: the scores against every key, then divide by √dₖ, where dₖ is the query’s length.', 'Softmax: subtract the largest score, take Math.exp, divide by the total. Then output column c = Σᵢ wᵢ · V[i][c].', CODE_ATT],
    solution: CODE_ATT,
    wrong: [CODE_ATT.replace(' / Math.sqrt(dk)', ''), CODE_ATT.replace('const m = Math.max(...s);', 'const m = 0;').replace('return V[0].map((_, c) => w.reduce((t, wi, i) => t + wi * V[i][c], 0));', 'return V[0].map((_, c) => s.reduce((t, wi, i) => t + wi * V[i][c], 0));')] });

  ex({ id: 'tf-x-scale-factor', chapter: 'tf-read', title: 'Dividing by √dₖ', level: 1, kind: 'number',
    text: ['A model uses queries and keys of dₖ = 64 numbers. One query and key have the dot product 12. What score goes into the softmax?'],
    answer: 1.5, tolerance: 0.001,
    whyNot: { '12': 'Every score is divided by √dₖ first.', '0.1875': 'That divides by dₖ = 64; the division is by its square root.', '96': 'Divide, do not multiply.' },
    hints: ['The score is q · k / √dₖ.', '√64 = 8.', '12 / 8 = 1.5.'],
    solution: '1.5', wrong: ['12', '0.1875'] });

  const DIMS = P.SCALE_DIMS, SCL = DIMS.map((d) => [Math.log2(d), r3(P.scaleStats(1, d, 8, 100).un.max)]);
  ex({ id: 'tf-x-scale-curve', chapter: 'tf-read', title: 'Without the division, as dₖ grows', level: 2, kind: 'predict-curve',
    text: ['Random queries and keys (numbers drawn with mean 0 and spread 1), each query against 8 keys, and no division by √dₖ. At dₖ = 2 the largest weight of a row is ' + SCL[0][1] + ' on average (0.125 would be perfectly even). Sketch that average as dₖ doubles up to 256; the x axis is log₂ dₖ.'],
    plot: { xr: [0.5, 8.5], yr: [0, 1], xname: 'log₂ dₖ', yname: 'mean largest weight', given: SCL.slice(0, 1), note: 'Drag from log₂ dₖ = 1 to 8. The faint point is dₖ = 2.' },
    answer: SCL, tolerance: 0.08, checks: ['rising'],
    hints: ['The raw scores spread like √dₖ. What does a softmax do with widely spread numbers?', 'The biggest score wins more and more of the row: at dₖ = 16 it already takes ' + SCL[3][1] + '.', 'It climbs towards 1: ' + SCL.map(([x, y]) => 'dₖ = ' + Math.pow(2, x) + ': ' + y).join(', ') + '. Scaled, it would stay between 0.30 and 0.40 at every size.'],
    solution: SCL, wrong: [SCL.map(([x]) => [x, SCL[0][1]]), SCL.map(([x]) => [x, r3(Math.max(0.125, SCL[0][1] - 0.03 * (x - 1)))])] });

  ex({ id: 'tf-x-shift', chapter: 'tf-read', title: 'The same number added to every score', level: 1, kind: 'choice',
    text: ['A row of scores (1, 3, 2) gives some weights after the softmax. Now 10 is added to every score of the row: (11, 13, 12). What happens to the weights?'],
    options: ['Nothing: they stay exactly the same', 'They all grow', 'They become nearly equal', 'The largest one takes all the weight'], answer: 0,
    whyNot: { 1: 'Weights always add up to 1, so they cannot all grow.', 2: 'The gaps between the scores are unchanged, and the softmax only sees the gaps.', 3: 'Only a wider spread sharpens the softmax; a shift does not.' },
    hints: ['e^(s + 10) = e^s · e^10.', 'Every term of the row, and their total, gains the same factor e^10.', 'The factor cancels in eˢ / Σ eˢ: the weights stay exactly the same. That is also why subtracting the largest score before exponentiating is safe.'],
    solution: 0, wrong: [1, 3] });

  // =========================================================================================================
  // Chapter 2: masks and heads
  // =========================================================================================================
  ex({ id: 'tf-x-causal-count', chapter: 'tf-mask', title: 'How many scores the causal mask hides', level: 1, kind: 'number',
    text: ['A sequence of 8 tokens has an 8 × 8 matrix of scores. How many of them does the causal mask hide?'],
    answer: 28, tolerance: 0,
    whyNot: { '36': 'That counts the diagonal too; each position may attend to itself.', '64': 'Only the scores above the diagonal are hidden.', '32': 'Count the cells above the diagonal: 7 + 6 + … + 1.', '56': 'That counts both sides of the diagonal; the past stays visible.' },
    hints: ['Row i may attend to positions 0 to i: it loses the 7 − i later ones.', 'Row 0 loses 7, row 1 loses 6, …, row 7 loses 0.', '7 + 6 + 5 + 4 + 3 + 2 + 1 = 28 = 8 · 7 / 2, out of 64.'],
    solution: '28', wrong: ['36', '64'] });

  const CW = f64(() => arr(T.attention(T.tensor(Q), T.tensor(K), T.tensor(V), { causal: true }).attn.weights)).map((r) => r.map(r2));
  ex({ id: 'tf-x-causal-weights', chapter: 'tf-mask', title: 'Weights under a causal mask', level: 2, kind: 'matrix',
    text: ['The same three positions a, b, c, in this order, with these scaled scores:', mtab(W3, SC, ['', 'a', 'b', 'c']), 'Apply a causal mask, then the softmax. Fill in the weights, to two decimals (write 0 where a weight is masked).'],
    rows: W3, cols: W3, corner: 'query \\ key', answer: CW, tolerance: 0.01, rowsSumTo: 1,
    hints: ['The mask sets every score above the diagonal to −∞; e^−∞ = 0.', 'Row a keeps only its own score: weight 1. Row b keeps (1, 0.5): e¹ / (e¹ + e^0.5).', 'a: (' + CW[0].join(', ') + '); b: (' + CW[1].join(', ') + '); c: (' + CW[2].join(', ') + '). The last row sees everything and is unchanged.'],
    solution: strM(CW), wrong: [strM(WT), strM(WT.map((r, i) => r.map((v, j) => (j > i ? 0 : v))))] });

  const PROW = [1, 2, 0.5, 3], PW = f64(() => arr(T.softmax(T.tensor([PROW.slice(0, 3).concat([-1e9])]), -1))[0].map(r2)), PU = f64(() => arr(T.softmax(T.tensor([PROW]), -1))[0].map(r2));
  ex({ id: 'tf-x-pad-row', chapter: 'tf-mask', title: 'A padded row', level: 2, kind: 'matrix',
    text: ['A sentence of three words is padded to length 4 with one padding token at the end. One word’s scaled scores against the four positions are ' + vec(PROW) + '; the last is the pad’s. With the padding mask, what are its weights? Two decimals.'],
    rows: ['weights'], cols: ['word 1', 'word 2', 'word 3', 'pad'], corner: '', answer: [PW], tolerance: 0.01, rowsSumTo: 1,
    hints: ['The mask replaces the pad’s score by −10⁹ before the softmax.', 'The pad’s weight is e^−10⁹ / total = 0; the other three share the whole row: e¹, e², e^0.5 over their own total.', 'e¹ = 2.718, e² = 7.389, e^0.5 = 1.649, total 11.756: (' + PW.join(', ') + '). Unmasked, the pad would have taken ' + PU[3] + '.'],
    solution: [PW.map(String)], wrong: [[PU.map(String)], [PU.slice(0, 3).concat([0]).map(String)]] });

  // left padding, as the canvas's transformer pads: two pads, then three words
  const BM = f64(() => { const m = arr(T.mul(T.causalMask(5), T.tensor([[0, 0, 1, 1, 1]]))); const out = []; for (let i = 2; i < 5; i++) for (let j = 0; j < 5; j++) if (m[i][j]) out.push('(' + i + ', ' + j + ')'); return '{' + out.join(', ') + '}'; });
  ex({ id: 'tf-x-both-masks', chapter: 'tf-mask', title: 'Padding first, then a causal mask', level: 2, kind: 'set',
    text: ['A batch pads short texts at the front. One text of three words becomes 5 positions: pad, pad, “tea”, “is”, “hot” (positions 0 to 4). Both a padding mask and a causal mask are applied. For the rows of the three real words (queries 2, 3 and 4), which (query, key) pairs keep their score? Write them as a set of pairs, like {(2, 0), (3, 1)}.'],
    answer: BM, placeholder: '{(query, key), …}',
    hints: ['Causal: query i may see keys 0 to i. Padding: keys 0 and 1 are never seen.', 'Query 2 (“tea”) may see keys 0, 1 and 2 by the causal mask, but 0 and 1 are pads.', BM + ': each real word sees the real words up to itself, never the pads in front.'],
    solution: BM, wrong: ['{(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)}', '{(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)}'] });

  const causalRef = (Qm, Km, Vm) => f64(() => arr(T.attention(T.tensor(Qm), T.tensor(Km), T.tensor(Vm), { causal: true })));
  const CODE_CAUSAL = 'function causalAttend(Q, K, V) {\n  const dk = Q[0].length;\n  return Q.map((q, i) => {\n    const s = K.slice(0, i + 1).map((k) => k.reduce((t, kj, j) => t + kj * q[j], 0) / Math.sqrt(dk));\n    const m = Math.max(...s);\n    const e = s.map((v) => Math.exp(v - m));\n    const z = e.reduce((a, b) => a + b, 0);\n    return V[0].map((_, c) => e.reduce((t, ei, j) => t + (ei / z) * V[j][c], 0));\n  });\n}';
  ex({ id: 'tf-x-code-causal', chapter: 'tf-mask', title: 'Write causal attention', level: 3, kind: 'code',
    text: ['Write `causalAttend(Q, K, V)`: scaled dot-product attention in which query i may attend only to keys 0 to i (Q, K and V have one row per position, in order). Return one output row per query.'],
    fn: 'causalAttend', params: ['Q', 'K', 'V'], start: 'function causalAttend(Q, K, V) {\n  \n}', tolerance: 1e-6, ref: causalRef,
    cases: [[Q, K, V], [[[1, 0]], [[1, 0]], [[5, 6]]], [[[0.5, -1], [2, 0.3], [1, 1]], [[1, 1], [-1, 0.5], [0, 2]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], [[[20, 0], [0, 20], [10, 10], [1, -1]], [[1, 0], [0, 1], [1, 1], [-1, 1]], [[1], [2], [3], [4]]]],
    hints: ['Row i only scores keys 0 to i: either skip the later keys or set their scores to −Infinity before the softmax.', 'The first row attends only to itself, so its output is V[0]. Remember the √dₖ.', CODE_CAUSAL],
    solution: CODE_CAUSAL,
    wrong: [CODE_CAUSAL.replace('K.slice(0, i + 1)', 'K'), CODE_CAUSAL.replace('K.slice(0, i + 1)', 'K.slice(0, i)').replace('const m = Math.max(...s);', 'const m = s.length ? Math.max(...s) : 0;')] });

  ex({ id: 'tf-x-head-size', chapter: 'tf-mask', title: 'How wide is each head?', level: 1, kind: 'number',
    text: ['A layer is 512 numbers wide and uses 8 heads. How many numbers does each head’s query have?'],
    answer: 64, tolerance: 0,
    whyNot: { '512': 'The width is shared out among the heads, not repeated for each.', '4096': 'That would make the heads 8 times wider than the layer.', '8': 'That is the number of heads.' },
    hints: ['The queries of the whole layer are 512 numbers, cut into equal slices.', 'One slice per head: 512 / 8.', '64 numbers per head, so each head divides its scores by √64 = 8. Eight heads of 64 cost about as much as one head of 512.'],
    solution: '64', wrong: ['512', '4096'] });

  ex({ id: 'tf-x-head-ablate', chapter: 'tf-mask', title: 'Switching heads off', level: 2, kind: 'choice',
    text: ['The tiny reversing transformer has 2 layers of 2 heads. On 150 new sequences it gets 100% of the answer digits right. Switched off one at a time, the heads give these scores:', { table: [['head switched off', 'answer digits right'], ['layer 0, head 0', '100%'], ['layer 0, head 1', '39%'], ['layer 1, head 0', '100%'], ['layer 1, head 1', '100%']] }, 'What is the best reading of this?'],
    options: ['Layer 0, head 1 does the pointing the task needs; the others are not needed for it', 'The three other heads are broken and should be retrained', 'Layer 0, head 1 is the only head with any weights', 'The model would do better without layer 0, head 1'], answer: 0,
    whyNot: { 1: 'They do no harm; the task simply does not need them once the pointing head exists.', 2: 'Every head has its own weights; switching off is what tested their use.', 3: 'Without it the score falls from 100% to 39%.' },
    hints: ['Which switch hurts the score?', 'Only one: the score falls from 100% to 39% without it.', 'Layer 0, head 1 carries the task: in its weights each answer position points at the input digit it must copy. The others can be switched off at no cost, here.'],
    solution: 0, wrong: [1, 3] });
})(globalThis.Nooral = globalThis.Nooral || {});
