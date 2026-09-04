// Part 12, masks in attention: a batch of two sentences of different lengths, the shorter padded; the padding mask
// that stops anyone attending to the pads; the causal mask that stops a position looking ahead; both together; and
// the proof that with the causal mask, changing a later word leaves every earlier output exactly as it was.
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml;
  const A_WORDS = ['we', 'planted', 'three', 'small', 'trees', 'today'], B_WORDS = ['birds', 'sang', 'at', 'dawn', 'again'];
  /** the inputs: rows for both sentences (pads are zero rows), the weights, and a replacement for A's last row */
  P.maskInputs = (inp) => {
    const g = S.T.rng(inp.seed), n = inp.lenA, m = inp.lenB;
    const XA = P.randMat(g, n, 4, 0.8), XB = P.randMat(g, m, 4, 0.8).concat(Array.from({ length: n - m }, () => [0, 0, 0, 0]));
    return { n, m, XA, XB, Wq: P.randMat(g, 4, 3, 0.8), Wk: P.randMat(g, 4, 3, 0.8), Wv: P.randMat(g, 4, 2, 0.8), last: P.randMat(g, 1, 4, 0.8)[0], wordsA: A_WORDS.slice(0, n), wordsB: B_WORDS.slice(0, m).concat(Array.from({ length: n - m }, () => '<pad>')) };
  };
  P.MASK_WORDS = { A: A_WORDS, B: B_WORDS };

  S.topic({
    id: 'attention-masks', family: 'p12-attn', order: 50, title: 'Causal and padding masks',
    blurb: 'A mask decides who may attend to whom. Before the softmax, a forbidden score is replaced by a huge negative number, so its weight comes out as exactly 0 and the rest of the row shares the whole weight. A padding mask hides the filler tokens that make short sentences as long as the longest in a batch. A causal mask hides every later position, so a model that predicts the next word cannot peek at it.',
    inputs: [
      { id: 'lenA', kind: 'number', label: 'Length of sentence A', def: 5, min: 3, max: 6 },
      { id: 'lenB', kind: 'number', label: 'Length of sentence B', def: 3, min: 1, max: 5, check: (v, inp) => (v >= inp.lenA ? 'Length of sentence B: shorter than sentence A, so it needs padding' : null) },
      { id: 'seed', kind: 'number', label: 'Seed for the numbers', def: 2, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Five words and three', values: { lenA: 5, lenB: 3, seed: 2 } },
      { label: 'Six words and two', values: { lenA: 6, lenB: 2, seed: 9 } },
      { label: 'Four words and three', values: { lenA: 4, lenB: 3, seed: 14 } },
      { label: 'One real word', values: { lenA: 4, lenB: 1, seed: 5 } },
    ],
    counters: [{ id: 'masked', label: 'scores masked' }],
    run(inp) {
      const T = S.T, F = H().frames(['masked']), q = F.q, I = P.maskInputs(inp), n = I.n, m = I.m;
      const proj = (X, W) => T.matmul(T.tensor(X), T.tensor(W));
      const att = (X, o) => { const r = T.attention(proj(X, I.Wq), proj(X, I.Wk), proj(X, I.Wv), o || {}); return { s: r.attn.scores.toArray(), w: r.attn.weights.toArray(), o: r.toArray() }; };
      const padMask = T.tensor([I.wordsB.map((w, j) => (j < m ? 1 : 0))]);
      const bNo = att(I.XB), bPad = att(I.XB, { mask: padMask }), aNo = att(I.XA), aCau = att(I.XA, { causal: true }), bBoth = att(I.XB, { mask: padMask, causal: true });
      const lo = -1e8, cw = 48, y0 = 176;
      const stripA = (lit) => [H().text('sa:h', 60, 34, 'sentence A', 'strong', 13), ...P.strip('sa', 160, 18, I.wordsA, { size: 13, cls: (j) => (lit === 'A' ? 'p12ask' : '') }).items];
      const stripB = (lit) => [H().text('sb:h', 60, 86, 'sentence B', 'strong', 13), ...P.strip('sb', 160, 70, I.wordsB, { size: 13, cls: (j) => (j >= m ? 'p12pad' : lit === 'B' ? 'p12ask' : '') }).items];
      const fmtS = (v) => (v < lo ? '−∞' : H().fx(v, 2));
      const sMax = Math.max(...[bNo, aNo].flatMap((r) => r.s.flat()).map(Math.abs));
      const scoresMat = (S2, words, title) => P.mat('ms', 70, y0, S2.map((r) => r.map((v) => (v < lo ? 0 : v))), { rows: words, cols: words, cw, ch: 26, max: sMax, title, fmt: null, cls: (i, j) => (S2[i][j] < lo ? 'p12mask' : '') }).items.map((it) => { const mm = /^ms:c(\d+)_(\d+)$/.exec(it.key); if (mm) it.text = fmtS(S2[+mm[1]][+mm[2]]); return it; });
      const weightsMat = (W, words, title, rowLit) => P.mat('mw', 430, y0, W, { rows: words, cols: words, cw, ch: 26, scheme: 'seq', max: 1, title, cls: (i, j) => (W[i][j] < 1e-12 && words[j] === '<pad>' ? 'p12mask' : i === rowLit ? 'act' : '') }).items;
      const nMasked = (S2) => S2.flat().filter((v) => v < lo).length;
      const pads = n - m, padShare = bNo.w[0].slice(m).reduce((a, b) => a + b, 0);
      F.push([...stripA(''), ...stripB('')], `A batch is one rectangle of numbers, so both sentences must have the same length. Sentence B has ${P.qpl(q, m, 'word')} and A has ${q.n(n)}, so B gets ${P.qpl(q, pads, 'padding token')} (<pad>, all zeros) at the end.`, { phase: 'batch', n, m, pads });
      F.push([...stripA(''), ...stripB('B'), ...scoresMat(bNo.s, I.wordsB, 'B: scaled scores, no mask'), ...weightsMat(bNo.w, I.wordsB, 'B: weights, no mask', 0)],
        `Without a mask the pads take part like real words. Row “${I.wordsB[0]}” gives the ${P.qpl(q, pads, 'pad')} ${q(padShare, 2)} of its weight in total: information drawn from nothing, and different for every amount of padding.`, { phase: 'nomask', padShare, pads });
      F.set('masked', nMasked(bPad.s));
      const row0 = bPad.w[0];
      F.push([...stripA(''), ...stripB('B'), ...scoresMat(bPad.s, I.wordsB, 'B: scores with the padding mask'), ...weightsMat(bPad.w, I.wordsB, 'B: weights, pads hidden', 0)],
        `The padding mask replaces every score in a pad’s column by −10⁹ (shown as −∞), ${P.qpl(q, nMasked(bPad.s), 'score')} here. After the softmax those weights are exactly 0, and row “${I.wordsB[0]}” shares all of its weight among the ${P.qpl(q, m, 'real word')}: ${P.qvec(q, row0.slice(0, m))}. The rows of the pads themselves are computed but thrown away later.`, { phase: 'pad', masked: nMasked(bPad.s), row: row0.slice(0, m), m, padW: row0.slice(m) });
      F.set('masked', nMasked(aCau.s));
      const last = n - 1;
      F.push([...stripA('A'), ...stripB(''), ...scoresMat(aCau.s, I.wordsA, 'A: scores with the causal mask'), ...weightsMat(aCau.w, I.wordsA, 'A: weights, no looking ahead', last)],
        `The causal mask hides everything above the diagonal: position i may attend only to positions 0 to i, ${P.qpl(q, nMasked(aCau.s), 'score')} masked out of ${q.n(n * n)}. The first word can only attend to itself, with weight ${q(aCau.w[0][0], 2)}; the last spreads over all ${q.n(n)}: ${P.qvec(q, aCau.w[last])}.`, { phase: 'causal', masked: nMasked(aCau.s), total: n * n, first: aCau.w[0][0], lastRow: aCau.w[last], n });
      F.set('masked', nMasked(bBoth.s));
      F.push([...stripA(''), ...stripB('B'), ...scoresMat(bBoth.s, I.wordsB, 'B: causal and padding masks'), ...weightsMat(bBoth.w, I.wordsB, 'B: weights, both masks', m - 1)],
        `The two masks multiply: a score survives only if it is neither in the future nor on a pad. For sentence B that leaves ${q.n(m * (m + 1) / 2)} of the ${q.n(m * m)} scores among real words, and the last real word “${I.wordsB[m - 1]}” spreads its weight as ${P.qvec(q, bBoth.w[m - 1].slice(0, m))}.`, { phase: 'both', kept: (m * (m + 1)) / 2, real: m * m, row: bBoth.w[m - 1].slice(0, m), masked: nMasked(bBoth.s) });
      // changing the last word of A: earlier outputs with and without the causal mask
      const XA2 = I.XA.slice(0, last).concat([I.last]), aCau2 = att(XA2, { causal: true }), aNo2 = att(XA2);
      const dmax = (o1, o2) => Math.max(...o1.slice(0, last).flatMap((r, i) => r.map((v, j) => Math.abs(v - o2[i][j]))));
      const dC = dmax(aCau.o, aCau2.o), dN = dmax(aNo.o, aNo2.o);
      F.push([...stripA('A'), ...stripB(''), ...scoresMat(aCau2.s, I.wordsA, 'A with a new last word, causal'), ...weightsMat(aCau2.w, I.wordsA, 'weights', last)],
        `Why a language model needs the causal mask: give “${I.wordsA[last]}” different numbers and recompute. With the mask, the outputs of the ${P.qpl(q, last, 'earlier word')} ${dC ? 'change by at most ' + q.sg(dC, 2) + ', mere rounding' : 'do not change at all'}: they cannot see it. Without the mask they change by up to ${q(dN, 3)}. So one pass can train the prediction at every position at once without any of them seeing its answer.`, { phase: 'peek', earlier: last, dC, dN });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Masked attention', text: ['scores = Q Kᵀ / √dₖ', 'where mask = 0:  score = −10⁹', 'weights = softmax(scores)  → 0 there', 'causal:  mask[i][j] = 1 if j ≤ i', 'padding: mask[i][j] = 0 if j is a pad'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'pad', once: true, pick: 'option',
      ask: 'With the causal mask, what weight will the first word give to itself?',
      options: ['exactly 1', 'about one half', 'it depends on the scores'],
      answer: 'exactly 1',
      explain: 'Its row has a single unmasked score, so the softmax gives it all the weight.',
    }],
    quiz: [
      { q: 'Why is a masked score set to a huge negative number rather than to 0?', options: ['so that its softmax weight becomes 0', 'so that it becomes the largest', 'to save memory', 'because 0 is not allowed'], answer: 0, why: 'e to a huge negative number is 0; a score of 0 would still get weight.' },
      { q: 'With a causal mask over 8 positions, how many scores are kept?', options: ['36', '64', '8', '28'], answer: 0, why: '1 + 2 + … + 8 = 36.' },
      { q: 'Which model needs the causal mask?', options: ['a decoder that predicts the next token', 'an encoder that classifies a whole sentence', 'neither', 'only a model with padding'], answer: 0, why: 'The next token must not be visible while it is being predicted.' },
      { q: 'A batch holds sentences of lengths 7, 4 and 2. How many pad tokens are added in all?', options: ['8', '3', '13', '0'], answer: 0, why: '0 + 3 + 5 = 8, to make every sentence 7 long.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
