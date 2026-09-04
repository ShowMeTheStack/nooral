// Part 12, scaled dot-product attention with every matrix on show: a few words as rows of numbers, their queries,
// keys and values, the score of every query against every key, the division by √dₖ, the softmax of each row and the
// weighted sum of the values. The numbers come from S.T (matmul, attention); the claims check redoes them by hand.
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml;
  const SENTENCES = {
    rain: { label: 'rain fell on roofs', words: ['rain', 'fell', 'on', 'roofs'] },
    mila: { label: 'Mila fed her cat', words: ['Mila', 'fed', 'her', 'cat'] },
    clocks: { label: 'old clocks tick very slowly', words: ['old', 'clocks', 'tick', 'very', 'slowly'] },
    two: { label: 'two words: kites fly', words: ['kites', 'fly'] },
  };
  /** the inputs of a run: X (one row per word, four features) and the three weight matrices, rounded to tenths */
  P.attnInputs = (inp) => {
    const words = SENTENCES[inp.sentence].words, g = S.T.rng(inp.seed), n = words.length, dk = +inp.dk;
    return { words, n, dk, X: P.randMat(g, n, 4, 0.8), Wq: P.randMat(g, 4, dk, 0.8), Wk: P.randMat(g, 4, dk, 0.8), Wv: P.randMat(g, 4, dk, 0.8) };
  };
  P.SENTENCES = SENTENCES;

  S.topic({
    id: 'attention-sdpa', family: 'p12-attn', order: 30, title: 'Scaled dot-product attention, matrix by matrix',
    blurb: 'Attention lets each position of a sequence gather information from every other position. Each word is a row of numbers. Three weight matrices turn it into a query (what it is looking for), a key (what it offers) and a value (what it hands over). Every query is compared with every key by a dot product, the scores are divided by √dₖ, a softmax turns each row into weights that add up to 1, and each word’s output is the weighted sum of the values. Every matrix is shown with its numbers.',
    inputs: [
      { id: 'sentence', kind: 'choice', label: 'Words', def: 'rain', options: Object.keys(SENTENCES).map((k) => ({ value: k, label: SENTENCES[k].label })) },
      { id: 'dk', kind: 'choice', label: 'Query and key size dₖ', def: '3', options: [{ value: '2', label: '2' }, { value: '3', label: '3' }, { value: '4', label: '4' }] },
      { id: 'focus', kind: 'number', label: 'Word to follow (from 0)', def: 1, min: 0, max: 4, check: (v, inp) => (v >= SENTENCES[inp.sentence].words.length ? 'Word to follow: these words are numbered 0 to ' + (SENTENCES[inp.sentence].words.length - 1) : null) },
      { id: 'seed', kind: 'number', label: 'Seed for the numbers', def: 4, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Four words, dₖ = 3', values: { sentence: 'rain', dk: '3', focus: 1, seed: 4 } },
      { label: 'Five words', values: { sentence: 'clocks', dk: '3', focus: 2, seed: 11 } },
      { label: 'dₖ = 2', values: { sentence: 'mila', dk: '2', focus: 3, seed: 7 } },
      { label: 'dₖ = 4', values: { sentence: 'mila', dk: '4', focus: 0, seed: 21 } },
      { label: 'Two words', values: { sentence: 'two', dk: '2', focus: 0, seed: 3 } },
    ],
    counters: [{ id: 'mats', label: 'matrices computed' }],
    run(inp) {
      const T = S.T, F = H().frames(['mats']), q = F.q, I = P.attnInputs(inp), { words, n, dk } = I, i = inp.focus;
      const t = (M) => T.tensor(M), arr = (x) => x.toArray();
      const Q = arr(T.matmul(t(I.X), t(I.Wq))), K = arr(T.matmul(t(I.X), t(I.Wk))), V = arr(T.matmul(t(I.X), t(I.Wv)));
      const raw = arr(T.matmul(t(Q), T.transpose(t(K))));
      const att = T.attention(t(Q), t(K), t(V)), Sc = arr(att.attn.scores), A = arr(att.attn.weights), O = arr(att);
      const root = Math.sqrt(dk), feat = ['f₁', 'f₂', 'f₃', 'f₄'], kcol = Array.from({ length: dk }, (_, j) => String(j + 1));
      const top = 52, bot = 252, cw = dk === 4 ? 38 : 42, cwn = 46, cs = cw + 2;
      // the top band: X × W = result; later Q, K and V side by side
      const xw = (W, R, name, wname, j) => [
        ...P.mat('ax', 60, top, I.X, { rows: words, cols: feat, cw, title: 'X: the words', cls: (r) => (r === i ? 'act' : '') }).items,
        P.sign('ax:x', 60 + 4 * cs + 16, top + 14 + n * 13, '×'),
        ...P.mat('aw', 60 + 4 * cs + 74, top, W, { rows: feat, cols: kcol, cw, title: wname, cls: (r, c) => (c === j ? 'act' : '') }).items,
        P.sign('ax:e', 60 + 4 * cs + 74 + dk * cs + 18, top + 14 + n * 13, '='),
        ...P.mat('ar', 60 + 4 * cs + 74 + dk * cs + 86, top, R, { rows: words, cols: kcol, cw, title: name, cls: (r, c) => (r === i && c === j ? 'act' : r === i ? 'sel' : '') }).items,
      ];
      const qkv = (lit) => [
        ...P.mat('tq', 60, top, Q, { rows: words, cols: kcol, cw, title: 'Q (queries)', cls: (r) => (lit === 'q' && r === i ? 'act' : '') }).items,
        ...P.mat('tk', 60 + dk * cs + 74, top, K, { rows: words, cols: kcol, cw, title: 'K (keys)', }).items,
        ...P.mat('tv', 60 + 2 * (dk * cs + 74), top, V, { rows: words, cols: kcol, cw, title: 'V (values)', }).items,
      ];
      const sqMax = Math.max(...Sc.flat().map(Math.abs), ...raw.flat().map(Math.abs));
      const square = (key, M, title, scheme, max, hit) => P.mat(key, 60, bot, M, { rows: words, cols: words, cw: cwn, scheme, max, title, cls: (r, c) => (r === i && c === hit ? 'act' : r === i ? 'sel' : '') }).items;
      const panel = (k) => H().lines('ap', 60 + n * 48 + 40, bot + 34, [['the row of “' + words[i] + '”', 'strong'], ['q·k     ' + P.vec(raw[i]), 'math'], ['÷ √' + dk + '    ' + P.vec(Sc[i]), 'math'], ['softmax ' + P.vec(A[i]), 'math']].slice(0, k), 13, 26);
      const sum = (a, b) => a.map((x, k) => x * b[k]);
      const terms = (prods) => prods.map((p, k) => q.t(p, 2, k === 0)).join('');
      const strip = (row) => {
        const st = P.strip('as', 60, 450, words, { sub: row ? (k) => H().fx(row[k], 2) : null, cls: (k) => (k === i ? 'p12ask' : '') });
        const items = st.items.slice();
        if (row) {
          const cx = (k) => st.boxes[k].x + st.boxes[k].w / 2, y0 = st.boxes[0].y - 3;
          row.forEach((a, k) => {
            if (k === i || a < 0.02) return;
            const x1 = cx(i) + (k > i ? 7 : -7), x2 = cx(k), hh = 16 + Math.abs(k - i) * 9;
            items.push({ key: 'as:a' + k, kind: 'path', d: `M${P.r2(x1)},${y0} C${P.r2(x1)},${y0 - hh} ${P.r2(x2)},${y0 - hh} ${P.r2(x2)},${y0}`, cls: 'p12arc', paint: { sw: P.r2(1 + 8 * a) }, pathAudit: 'edge-x', layer: 'lines', box: { x: Math.min(x1, x2), y: y0 - hh, w: Math.abs(x2 - x1), h: hh } });
          });
        }
        return items;
      };
      F.push([...P.mat('ax', 60, top, I.X, { rows: words, cols: feat, cw, title: 'X: the words', cls: (r) => (r === i ? 'act' : '') }).items, ...strip(null)],
        `Each of the ${P.qpl(q, n, 'word')} is a row of ${q.n(4)} numbers (in a real model, its embedding plus its position). Follow “${words[i]}”, row ${q.n(i)}: ${P.qvec(q, I.X[i], 1)}.`, { phase: 'x', width: 4, i, n, x: I.X[i], X: I.X, Wq: I.Wq, Wk: I.Wk, Wv: I.Wv, dk });
      F.count('mats');
      F.push([...xw(I.Wq, Q, 'Q = X Wq', 'Wq', 0), ...strip(null)],
        `Multiplying by Wq gives every word a query of ${q.n(dk)} numbers: what it is looking for. The first number of “${words[i]}”’s query is the row ${P.qvec(q, I.X[i], 1)} times the first column of Wq: ${terms(sum(I.X[i], I.Wq.map((r) => r[0])))} = ${q(Q[i][0], 2)}. Its whole query is ${P.qvec(q, Q[i])}.`, { phase: 'q', i, x: I.X[i], prods: sum(I.X[i], I.Wq.map((r) => r[0])), q0: Q[i][0], qi: Q[i] });
      F.count('mats');
      F.push([...xw(I.Wk, K, 'K = X Wk', 'Wk', -1), ...strip(null)],
        `Wk gives every word a key, what it offers to anyone searching: “${words[i]}”’s key is ${P.qvec(q, K[i])}. Queries and keys must have the same size, here ${q.n(dk)}, so that they can be compared.`, { phase: 'k', i, ki: K[i], dk });
      F.count('mats');
      F.push([...xw(I.Wv, V, 'V = X Wv', 'Wv', -1), ...strip(null)],
        `Wv gives every word a value: what it hands over to whoever attends to it. “${words[i]}”’s value is ${P.qvec(q, V[i])}. All three matrices are learned in training; the same three are used for every position.`, { phase: 'v', i, vi: V[i] });
      F.count('mats');
      const best = P.argmax(raw[i]);
      F.push([...qkv('q'), ...square('am', raw, 'scores Q Kᵀ', 'div', sqMax, best), ...panel(2), ...strip(null)],
        `Every query meets every key: the score of row i, column j is the dot product qᵢ · kⱼ. In the row of “${words[i]}” the scores are ${P.qvec(q, raw[i])}. The largest, ${q(raw[i][best], 2)}, is for “${words[best]}”: ${terms(sum(Q[i], K[best]))} = ${q(raw[i][best], 2)}.`, { phase: 'scores', i, row: raw[i], best, prods: sum(Q[i], K[best]), qi: Q[i], kb: K[best] });
      F.count('mats');
      F.push([...qkv('k'), ...square('am', Sc, 'scaled: Q Kᵀ ÷ √' + dk, 'div', sqMax, best), ...panel(3), ...strip(null)],
        `Each score is divided by √dₖ = √${q.n(dk)} = ${q(root, 3)}, so the row of “${words[i]}” becomes ${P.qvec(q, Sc[i])}. Without this, scores from longer vectors would be larger and the softmax would pick a single word too sharply (see [[attention-scale]]).`, { phase: 'scaled', i, row: Sc[i], dk, root });
      F.count('mats');
      const ex = Sc[i].map((s) => Math.exp(s)), z = ex.reduce((a, b) => a + b, 0), win = P.argmax(A[i]);
      F.push([...qkv('v'), ...square('am', A, 'weights: softmax of each row', 'seq', 1, win), ...panel(4), ...strip(A[i])],
        `The softmax turns each row into weights: eˢ divided by the row’s total. For “${words[i]}”: e to the scaled scores gives ${P.qvec(q, ex)}, which add up to ${q(z, 2)}, so the weights are ${P.qvec(q, A[i])}. They add up to 1. The most goes to “${words[win]}”, ${q(A[i][win], 2)}; the arcs show the row.`, { phase: 'softmax', i, ex, z, w: A[i], win });
      F.count('mats');
      F.push([...P.mat('am', 60, bot, A, { rows: words, cols: words, cw: cwn, scheme: 'seq', max: 1, title: 'weights A', cls: (r) => (r === i ? 'act' : '') }).items,
        P.sign('bo:x', 60 + n * 48 + 16, bot + 14 + n * 13, '×'),
        ...P.mat('bv', 60 + n * 48 + 40, bot, V, { rows: false, cols: kcol, cw, title: 'V' }).items,
        P.sign('bo:e', 60 + n * 48 + 40 + dk * cs + 16, bot + 14 + n * 13, '='),
        ...P.mat('bz', 60 + n * 48 + 40 + dk * cs + 46, bot, O, { rows: false, cols: kcol, cw, title: 'output A V', cls: (r, c) => (r === i && c === 0 ? 'act' : r === i ? 'sel' : '') }).items,
        ...qkv(''), ...strip(A[i])],
        `The output of “${words[i]}” is its weights times the values: a blend of every word’s value, mostly the value of “${words[win]}”. Its first number is ${terms(A[i].map((a, k) => a * V[k][0]))} = ${q(O[i][0], 2)}; the whole row is ${P.qvec(q, O[i])}.`, { phase: 'out', i, prods: A[i].map((a, k) => a * V[k][0]), o: O[i] });
      const self = A.map((r, k) => r[k]), most = P.argmax(self);
      F.push([...P.mat('am', 60, bot, A, { rows: words, cols: words, cw: cwn, scheme: 'seq', max: 1, title: 'weights A' }).items, ...qkv(''), ...strip(A[i])],
        `Every word did the same at once: attention is a few matrix products, ${q.n(n)} × ${q.n(n)} weights for ${P.qpl(q, n, 'word')}. The word that keeps the most for itself is “${words[most]}”, ${q(self[most], 2)}. Nothing here knows the order of the words: shuffle the rows and the outputs are shuffled the same way, which is why positions are added first.`, { phase: 'end', n, self, most });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Scaled dot-product attention', text: ['Q = X Wq,  K = X Wk,  V = X Wv', 'A = softmax(Q Kᵀ / √dₖ)  (each row)', 'output = A V', 'each row of A adds up to 1'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'scaled', once: true, pick: 'option',
      ask: (fr, i, inp) => 'Which word will “' + SENTENCES[inp.sentence].words[inp.focus] + '” give the largest weight?',
      options: (fr, i, inp) => SENTENCES[inp.sentence].words.slice(),
      answer: (fr, i, inp) => SENTENCES[inp.sentence].words[fr.find((f) => f.meta.phase === 'softmax').meta.win],
      explain: 'The softmax keeps the order of the scores: the largest scaled score always gets the largest weight.',
    }],
    quiz: [
      { q: 'A query of size dₖ = 4 meets a key. The score is…', options: ['their dot product, divided by 2', 'their dot product, divided by 4', 'the distance between them', 'the product of their lengths'], answer: 0, why: '√4 = 2.' },
      { q: 'After the softmax, each row of the weight matrix…', options: ['adds up to 1', 'adds up to the number of words', 'has exactly one 1', 'can hold negative numbers'], answer: 0, why: 'Softmax makes positive weights that sum to one, row by row.' },
      { q: 'With 6 words, the score matrix Q Kᵀ has how many numbers?', options: ['36', '6', '12', 'dₖ × 6'], answer: 0, why: 'One score for every pair of query and key: 6 × 6.' },
      { q: 'Which matrix decides what a word hands over to the words that attend to it?', options: ['Wv, through the values', 'Wq', 'Wk', 'the softmax'], answer: 0, why: 'The output is a weighted sum of values; queries and keys only set the weights.' },
      { q: 'Adding the same number to every score in a row changes the weights…', options: ['not at all', 'by making them larger', 'by making them all equal', 'only for the first word'], answer: 0, why: 'The softmax cancels a shift that is shared by the whole row.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
