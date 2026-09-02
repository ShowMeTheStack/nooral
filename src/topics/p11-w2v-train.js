// Part 11, word2vec trained here: skip-gram pairs read off a sentence of the corpus, one pair scored with its
// negatives before and after training, then the map of the words epoch by epoch (each epoch's vectors seen along
// the final model's two main directions) while words of a kind draw together.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11;
  const MAP = { x: 0, y: 40, w: 470, h: 420 };
  const sigm = (x) => 1 / (1 + Math.exp(-x));

  /** mean cosine of pairs of words of the same kind, and of different kinds */
  const together = (rows, idx, words) => {
    let sIn = 0, nIn = 0, sOut = 0, nOut = 0;
    for (let a = 0; a < words.length; a++) for (let b = a + 1; b < words.length; b++) {
      const c = S.DL.cosine(rows[idx[a]], rows[idx[b]]);
      if (P.catOf(words[a]) === P.catOf(words[b])) { sIn += c; nIn++; } else { sOut += c; nOut++; }
    }
    return { within: sIn / nIn, between: sOut / nOut };
  };

  S.topic({
    id: 'word2vec-train', family: 'p11-embed', order: 20, title: 'word2vec: learning vectors from neighbouring words',
    blurb: 'word2vec slides a window along the text. Each word in the middle is paired with the words around it, and training pushes the two vectors of a real pair to agree (a high score) while a few random words drawn as negatives are pushed to disagree. Nothing tells it what a word means; words that keep the same company simply end up pointing the same way. Watch the map of the words sort itself out, epoch by epoch.',
    inputs: [
      { id: 'dim', kind: 'choice', label: 'Vector size d', def: '16', options: [{ value: '4', label: '4' }, { value: '8', label: '8' }, { value: '16', label: '16' }, { value: '24', label: '24' }] },
      { id: 'window', kind: 'number', label: 'Window (words each side)', def: 2, min: 1, max: 3 },
      { id: 'negatives', kind: 'number', label: 'Negatives per pair', def: 5, min: 1, max: 10 },
      { id: 'epochs', kind: 'number', label: 'Epochs', def: 8, min: 1, max: 10 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
      { id: 'probe', kind: 'choice', label: 'Word to follow', def: 'cat', options: ['cat', 'king', 'bread', 'red', 'three', 'river', 'boat', 'tea'].map((w) => ({ value: w, label: w })) },
    ],
    presets: [
      { label: 'Default run', values: { dim: '16', window: 2, negatives: 5, epochs: 8, seed: 1, probe: 'cat' } },
      { label: 'Tiny vectors (d = 4)', values: { dim: '4', window: 2, negatives: 5, epochs: 8, seed: 1, probe: 'king' } },
      { label: 'Wide window', values: { dim: '16', window: 3, negatives: 5, epochs: 6, seed: 2, probe: 'bread' } },
      { label: 'One epoch only', values: { dim: '16', window: 2, negatives: 5, epochs: 1, seed: 1, probe: 'three' } },
      { label: 'One negative', values: { dim: '16', window: 2, negatives: 1, epochs: 8, seed: 3, probe: 'red' } },
    ],
    counters: [{ id: 'epoch', label: 'epochs trained' }],
    run(inp) {
      const F = H().frames(['epoch']), q = F.q, d = +inp.dim, K0 = inp.negatives;
      const run = P.w2v({ dim: d, window: inp.window, negatives: K0, epochs: inp.epochs, seed: inp.seed }), w = run.w, V = w.vocab.length;
      // a sentence of the corpus with the probe word in it, the probe in the middle of the window
      const sentences = run.text.split('. ').map((x) => x.replace(/\.$/, '').split(' '));
      const content = new Set(P.mapWords()), winOf = (t) => { const c = t.indexOf(inp.probe), out = []; for (let k = c - inp.window; k <= c + inp.window; k++) if (k >= 0 && k < t.length && k !== c && w.index.has(t[k])) out.push(k); return out; };
      // a sentence where a word of some kind falls in the window, so the pair followed is a telling one
      const sent = sentences.find((t) => t.includes(inp.probe) && winOf(t).some((k) => content.has(t[k]))) || sentences.find((t) => t.includes(inp.probe));
      const c = sent.indexOf(inp.probe), win = winOf(sent);
      const strip = (lit) => S.Kit.tokens.strip({ prefix: 'wv:s', x: 0, y: 30, width: 790, tokens: sent, size: 14, gap: 6, cls: (i) => (i === c ? 'new' : lit && win.includes(i) ? 'p11ctx' : '') }).items;
      const panel = (rows, x, y) => H().lines('wv:p', x === undefined ? 520 : x, y || 120, rows, 13, 22);
      const pairs = win.map((k) => sent[k]).sort((a, b) => (content.has(b) ? 1 : 0) - (content.has(a) ? 1 : 0));
      F.push([...strip(true), ...panel([['skip-gram pairs (centre, context)', 'strong'], ...pairs.map((p) => ['(' + inp.probe + ', ' + p + ')', 'mono'])], 0, 120)],
        `word2vec reads the corpus (${P.qpl(q, w.ids.length, 'word')}, ${P.qpl(q, V, 'distinct word')} kept). At each position it pairs the centre word with the words up to ${P.qpl(q, inp.window, 'place')} either side. Here “${inp.probe}” is the centre, which gives ${P.qpl(q, pairs.length, 'pair')} like (${inp.probe}, ${pairs[0]}).`, { phase: 'pairs', nWords: w.ids.length, V, window: inp.window, nPairs: pairs.length });
      // one pair and K negatives, scored with the starting vectors and with the trained ones
      const ci = w.index.get(inp.probe), oi = w.index.get(pairs[0]), g = S.T.rng(1000 + inp.seed), negs = [];
      while (negs.length < K0) { const k = w.table[g.int(w.table.length)]; if (k !== oi && k !== ci && !negs.includes(k)) negs.push(k); }
      const rowOf = (A, i) => Float64Array.from(A.slice(i * d, (i + 1) * d));
      const score = (Wm, Cm) => {
        const v = rowOf(Wm, ci), u = rowOf(Cm, oi), nv = negs.map((k) => rowOf(Cm, k)), r = S.DL.w2vPair(v, u, nv);
        const dotp = (a, b) => a.reduce((s, x, j) => s + x * b[j], 0);
        return { loss: r.loss, pos: sigm(dotp(u, v)), neg: nv.map((u2) => sigm(dotp(u2, v))) };
      };
      const before = score(run.W0, new Float32Array(V * d)), after = score(w.W, w.C);
      const barRows = (sc) => [{ label: pairs[0], v: sc.pos, cls: 'good', value: H().fx(sc.pos, 2) }].concat(negs.map((k, j) => ({ label: w.vocab[k], v: sc.neg[j], cls: 'hot', value: H().fx(sc.neg[j], 2) })));
      const barsAt = (sc, title) => [H().text('wv:bt', 0, 120, title, 'strong', 13), ...K().bars({ prefix: 'wv:b', x: 0, y: 132, w: 470, rows: barRows(sc), max: 1, labelW: 90, rowH: 26 })];
      const lossLine = (sc, when) => panel([['score = σ(u_context · v_centre)', 'math'], ['real pair: push towards 1', 'legend'], ['negatives: push towards 0', 'legend'], ['loss ' + when + ' = ' + H().fx(sc.loss, 3), 'strong']]);
      F.push([...strip(false), ...barsAt(before, 'before training: (' + inp.probe + ', ' + pairs[0] + ') and ' + K0 + ' negatives'), ...lossLine(before, 'now')],
        `Each word has two vectors: v when it is the centre, u when it is the context. Training scores the pair (${inp.probe}, ${pairs[0]}) as σ(u·v), and also ${P.qpl(q, K0, 'random word')} drawn as negatives. The context vectors start at zero, so every score is ${q(before.pos, 2)} and the loss is ${q(before.loss, 3)}.`, { phase: 'pair0', loss: before.loss, pos: before.pos, neg: before.neg, negs, K: K0 });
      // epochs on the map
      const words = P.mapWords(), idx = words.map((x) => w.index.get(x)), last = run.epochs[run.epochs.length - 1];
      const proj = S.DL.project2d(idx.map((i) => last.rows[i])), comps = proj.components;
      const pointsAt = (rows) => { const R = idx.map((i) => rows[i]), m = R[0].map((_, j) => R.reduce((s, r) => s + r[j], 0) / R.length); return R.map((r) => [0, 1].map((k) => r.reduce((s, x, j) => s + (x - m[j]) * comps[k][j], 0))); };
      const allPts = run.epochs.map((e) => pointsAt(e.rows)), flat = allPts.flat();
      const range = { u0: Math.min(...flat.map((p) => p[0])), u1: Math.max(...flat.map((p) => p[0])), v0: Math.min(...flat.map((p) => p[1])), v1: Math.max(...flat.map((p) => p[1])) };
      const losses = run.epochs.slice(1).map((e) => e.loss), lmax = Math.max(...losses) * 1.1, lmin = Math.min(...losses) * 0.9;
      const ax = S.Kit.curve.axes({ prefix: 'wv:ax', x: 560, y: 260, w: 220, h: 150, xr: [0, Math.max(2, inp.epochs)], yr: [Math.max(0, lmin), lmax], xticks: inp.epochs > 5 ? 2 : 1, yticks: 3, xname: 'epoch', yname: 'mean loss per pair', grid: false });
      const curve = (e) => [...ax.items, S.Kit.curve.line(ax, 'wv:lc', losses.slice(0, e).map((l, k) => [k + 1, l]), 'p11a'), ...(e ? [S.Kit.curve.dot(ax, 'wv:ld', e, losses[e - 1], 'p11a')] : [])];
      run.epochs.forEach((ep, e) => {
        if (e > 0) F.count('epoch');
        const m = P.wordMap(words, allPts[e], { prefix: 'wv:m', box: MAP, range, act: inp.probe, first: [inp.probe] }), tg = together(ep.rows, idx, words);
        const items = [...m.items, ...P.kindLegend('wv:lg', 520, 70), ...curve(e)];
        if (e === 0) F.push(items, `Before training the vectors are small random numbers, so the words lie in a heap. The map shows every epoch along the two main directions of the final vectors. Words of the same kind have a mean cosine of ${q(tg.within, 2)}, words of different kinds ${q(tg.between, 2)}.`, { phase: 'epoch', epoch: 0, within: tg.within, between: tg.between });
        else F.push(items, `Epoch ${q.n(e)}: the mean loss per pair is ${q(ep.loss, 3)}. Words of the same kind now have a mean cosine of ${q(tg.within, 2)}, words of different kinds ${q(tg.between, 2)}` + (e === 1 ? ': the kinds are already pulling apart.' : e === inp.epochs ? ', and the kinds sit in clumps of their own colour.' : '.'), { phase: 'epoch', epoch: e, loss: ep.loss, within: tg.within, between: tg.between });
      });
      // the same pair, trained
      F.push([...strip(false), ...barsAt(after, 'after training: the same pair and negatives'), ...lossLine(after, 'now')],
        `After ${P.qpl(q, inp.epochs, 'epoch')} the real pair scores ${q(after.pos, 2)} and the negatives ${H().list(after.neg.map((v) => q(v, 2)))}; the loss of this pair fell from ${q(before.loss, 3)} to ${q(after.loss, 3)}.`, { phase: 'pair1', loss: after.loss, loss0: before.loss, pos: after.pos, neg: after.neg });
      const nb = w.nearest(inp.probe, 5), nbW = nb.map((x) => x.label);
      const m = P.wordMap(words, allPts[allPts.length - 1], { prefix: 'wv:m', box: MAP, range, act: inp.probe, lit: new Set(nbW), faded: true, first: [inp.probe].concat(nbW) });
      const lines = nbW.filter((x) => words.includes(x)).map((x) => { const a = allPts[allPts.length - 1][words.indexOf(inp.probe)], b = allPts[allPts.length - 1][words.indexOf(x)]; return { key: 'wv:nl:' + x, kind: 'path', d: `M${m.X(a[0])},${m.Y(a[1])} L${m.X(b[0])},${m.Y(b[1])}`, cls: 'p11nb', audit: 'ignore', layer: 'lines' }; });
      F.push([...m.items, ...lines, ...P.kindLegend('wv:lg', 520, 70), ...K().bars({ prefix: 'wv:nb', x: 520, y: 260, w: 270, rows: nb.map((x) => ({ label: x.label, v: Math.max(0, x.score), value: H().fx(x.score, 2) })), max: 1, labelW: 70, rowH: 24 }), H().text('wv:nt', 520, 248, 'nearest to “' + inp.probe + '” (cosine)', 'strong', 13)],
        `The nearest words to “${inp.probe}” by cosine are ${H().list(nb.map((x) => x.label + ' (' + q(x.score, 2) + ')'))}. The model never saw a definition: these words simply turned up in the same kinds of places.`, { phase: 'nearest', nb: nb.map((x) => ({ label: x.label, score: x.score })) });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Skip-gram with negative sampling', text: ['for a centre word c and a context word o:', 'loss = −log σ(u_o · v_c) − Σ_k log σ(−u_k · v_c)', 'k: negatives drawn ∝ count^0.75', 'σ(x) = 1 / (1 + e^−x)', 'the learning rate falls linearly to 1%'].join('\n') }] } };
    },
    predict: [
      {
        at: (f) => f.meta.phase === 'pairs', once: true, pick: 'option',
        ask: 'Before training, the context vectors are all zero. What score σ(u·v) does every pair get?',
        options: ['0.5', '0', '1', 'a different one for each pair'],
        answer: (fr) => H().fx(fr.find((f) => f.meta.phase === 'pair0').meta.pos, 1),
        explain: 'σ(0) = 0.5 for every pair, real or negative.',
      },
      {
        at: (f) => f.meta.phase === 'epoch' && f.meta.epoch === 0, once: true, pick: 'option',
        ask: 'After training, will words of the same kind have a higher or lower mean cosine than now?',
        options: ['higher', 'lower', 'about the same'],
        answer: (fr) => { const e = fr.filter((f) => f.meta.phase === 'epoch'), a = e[0].meta.within, b = e[e.length - 1].meta.within; return b > a + 0.05 ? 'higher' : b < a - 0.05 ? 'lower' : 'about the same'; },
        explain: (fr) => { const e = fr.filter((f) => f.meta.phase === 'epoch'); return e[e.length - 1].say; },
      },
    ],
    quiz: [
      { q: 'In skip-gram, what does the model predict?', options: ['the words near a centre word', 'the next sentence', 'the part of speech', 'the word’s spelling'], answer: 0, why: 'Each (centre, context) pair is a training example.' },
      { q: 'What are negative samples for?', options: ['to push apart a word and random words it was not seen with', 'to remove rare words', 'to make the loss negative', 'to shorten the vectors'], answer: 0, why: 'Without them every score could simply go to 1.' },
      { q: 'A wider window tends to make neighbours…', options: ['share topic more than grammar', 'share spelling', 'random', 'identical'], answer: 0, why: 'A wide window counts loosely related words as company.' },
      { q: 'Why can two words that never appear together still get similar vectors?', options: ['they appear next to the same other words', 'they share letters', 'the model is told they are synonyms', 'they are next to each other in the vocabulary'], answer: 0, why: '“tea” and “milk” both follow “cups of”, so both learn to score well with it.' },
      { q: 'The map is drawn in 2-D, but the vectors have 16 numbers. So…', options: ['two words can look close on the map yet be far apart in 16-D', 'the map is exact', 'the extra numbers are zero', 'cosines are computed on the map'], answer: 0, why: 'The map keeps only the two directions with the most spread.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
