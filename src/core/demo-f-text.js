// Text for the sample scenes (S.Toy): byte-pair merges, a character bigram model trained by gradient descent and
// sampled, and scaled dot-product attention with two heads whose weights are written out by hand so each head
// has a clear job.
(function (S) {
  'use strict';
  const T = (S.Toy = S.Toy || {});

  // ---- byte-pair encoding ---------------------------------------------------------------------------------------
  /**
   * Learn merges on a text, word by word (a word keeps its leading space as ·): {steps: [{tokens (the whole text
   * as tokens), pair, count, merged, vocab}]}; steps[0] is the text as characters.
   */
  /** how often each pair of neighbouring tokens occurs (never across a word boundary or punctuation) */
  T.bpePairs = function (toks) {
    const counts = new Map();
    for (let i = 1; i < toks.length; i++) {
      if (toks[i] === ' ' || toks[i].startsWith(' ') || /^[.,]/.test(toks[i]) || /[.,]$/.test(toks[i - 1])) continue;
      const k = toks[i - 1] + '\u0000' + toks[i];
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    return counts;
  };
  T.bpe = function (text, merges) {
    let toks = [...text];
    const vocab = [...new Set(toks)].sort();
    const steps = [{ tokens: toks.slice(), pair: null, count: 0, vocab: vocab.slice() }];
    for (let m = 0; m < merges; m++) {
      const counts = T.bpePairs(toks);
      let best = null, bc = 1;
      for (const [k, c] of counts) if (c > bc || (c === bc && best !== null && k < best)) { best = k; bc = c; }
      if (!best) break;
      const [a, b] = best.split('\u0000'), out = [];
      for (let i = 0; i < toks.length; i++) { if (i + 1 < toks.length && toks[i] === a && toks[i + 1] === b) { out.push(a + b); i++; } else out.push(toks[i]); }
      toks = out;
      vocab.push(a + b);
      steps.push({ tokens: toks.slice(), pair: [a, b], count: bc, merged: a + b, vocab: vocab.slice() });
    }
    return { steps };
  };

  // ---- a character bigram model ---------------------------------------------------------------------------------
  /**
   * Logits L[a][b] for "b follows a", trained by full-batch gradient descent on the mean cross-entropy:
   * {chars, snaps: [{step, loss, L}], sample(L, n, seed, first)}
   */
  T.bigram = function (text, o) {
    o = o || {};
    const chars = [...new Set(text)].sort(), V = chars.length, ix = new Map(chars.map((c, i) => [c, i]));
    const N = Array.from({ length: V }, () => new Array(V).fill(0));
    for (let i = 1; i < text.length; i++) N[ix.get(text[i - 1])][ix.get(text[i])]++;
    const rows = N.map((r) => r.reduce((a, b) => a + b, 0)), tot = rows.reduce((a, b) => a + b, 0);
    const L = Array.from({ length: V }, () => new Array(V).fill(0));
    const keep = o.keep || [0, 10, 40, 150, 600], snaps = [], lr = o.lr || 40;
    const softmax = (row) => { const m = Math.max(...row), e = row.map((z) => Math.exp(z - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
    for (let step = 0; step <= Math.max(...keep); step++) {
      let loss = 0;
      const P = L.map(softmax);
      for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) if (N[a][b]) loss -= N[a][b] * Math.log(P[a][b]);
      loss /= tot;
      if (keep.includes(step)) snaps.push({ step, loss, L: L.map((r) => r.slice()) });
      for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) L[a][b] -= (lr * (rows[a] * P[a][b] - N[a][b])) / tot;
    }
    const sample = (Lm, n, seed, first) => {
      const r = T.rng(seed || 1);
      let c = ix.get(first || ' '), out = '';
      for (let i = 0; i < n; i++) {
        const p = softmax(Lm[c]);
        let u = r(), k = 0;
        while (k < V - 1 && u > p[k]) { u -= p[k]; k++; }
        out += chars[k]; c = k;
      }
      return out;
    };
    return { chars, snaps, sample, probs: (Lm) => Lm.map(softmax), counts: N };
  };

  // ---- attention --------------------------------------------------------------------------------------------------
  // Each word is described by four features (is it a thing, a describing word, an action, a word that points
  // back like "its") and its position as an angle (cos, sin of 0.5 radians a word). A query whose position part
  // is turned by one word forward matches the key of the next word; turned two words back, the word two places
  // before. So each head's pattern has a reason, and every number can be followed by hand.
  T.attnSentence = ['the', 'small', 'robot', 'lifted', 'its', 'red', 'cup'];
  T.attnFeatures = ['thing', 'describes', 'action', 'points back', 'cos', 'sin'];
  const FEAT = { the: [0, 0, 0, 0], small: [0, 1, 0, 0], robot: [1, 0, 0, 0], lifted: [0, 0, 1, 0], its: [0, 0, 0, 1], red: [0, 1, 0, 0], cup: [1, 0, 0, 0] };
  const W = 0.5, r3 = (v) => Math.round(v * 100) / 100;
  /** query matrix: a word with feature f asks (weight 3) for things, and its position is turned by k words */
  const query = (f, k) => { const c = r3(2 * Math.cos(k * W)), s = r3(2 * Math.sin(k * W)), M = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, c, s], [0, -s, c]]; M[f][0] = 3; return M; };
  const KEY = [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]];
  const VAL = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
  T.attnValueNames = ['thing', 'describes', 'action'];
  T.attnHeads = [
    { name: 'head 1: which thing does a describing word describe?', short: 'describes → next thing', Wq: query(1, 1), Wk: KEY, Wv: VAL },
    { name: 'head 2: what does a pointing word point back to?', short: 'its → the thing two back', Wq: query(3, -2), Wk: KEY, Wv: VAL },
  ];
  T.attention = function (words, head) {
    const X = words.map((w, i) => FEAT[w].concat([r3(Math.cos(i * W)), r3(Math.sin(i * W))]));
    const mul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
    const Q = mul(X, head.Wq), K = mul(X, head.Wk), V = mul(X, head.Wv), dk = Q[0].length;
    const scores = Q.map((q) => K.map((k) => q.reduce((s, v, i) => s + v * k[i], 0) / Math.sqrt(dk)));
    const A = scores.map((row) => { const m = Math.max(...row), e = row.map((z) => Math.exp(z - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); });
    const out = mul(A, V);
    return { X, Q, K, V, scores, A, out };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
