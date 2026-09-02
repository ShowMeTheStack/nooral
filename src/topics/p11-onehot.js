// Part 11, one-hot vectors against embeddings: a handful of words as one-hot rows (every pair equally far apart),
// then as rows of a table word2vec learned on the corpus, where looking a word up is the same as multiplying its
// one-hot row by the table, and where words used alike have similar rows.
(function (S) {
  'use strict';
  const P = S.P11, H = () => S.Kit.ml, K = () => S.Kit.p11;
  const WX = 96, TOPY = 26;

  S.topic({
    id: 'one-hot-embeddings', family: 'p11-embed', order: 10, title: 'One-hot vectors against embeddings',
    blurb: 'A network cannot read a word, only numbers. The simplest code gives each word its own position in a long vector of zeros (one-hot), but then every word is exactly as far from every other. An embedding gives each word a short row of learned numbers instead, and words used in the same way end up with similar rows. Looking a word up in the table is the same as multiplying its one-hot vector by it.',
    inputs: [
      { id: 'words', kind: 'list', label: 'Words (3 to 8)', def: ['king', 'queen', 'cat', 'dog', 'bread', 'rice', 'red', 'blue'], min: 3, max: 8, check: (v) => P.checkWords(v) },
      { id: 'dim', kind: 'choice', label: 'Embedding size d', def: '16', options: [{ value: '4', label: '4 numbers' }, { value: '8', label: '8 numbers' }, { value: '16', label: '16 numbers' }] },
    ],
    presets: [
      { label: 'Pairs of a kind', values: { words: ['king', 'queen', 'cat', 'dog', 'bread', 'rice', 'red', 'blue'], dim: '16' } },
      { label: 'People', values: { words: ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl'], dim: '16' } },
      { label: 'Food and drink', values: { words: ['bread', 'cheese', 'soup', 'tea', 'milk', 'water'], dim: '16' } },
      { label: 'Mixed, d = 4', values: { words: ['horse', 'goat', 'three', 'five', 'hill', 'river'], dim: '4' } },
    ],
    counters: [{ id: 'nums', label: 'numbers per word' }],
    run(inp) {
      const F = H().frames(['nums']), q = F.q, words = inp.words, n = words.length, d = +inp.dim;
      const run = P.w2v({ dim: d }), w = run.w, V = w.vocab.length, ids = words.map((x) => w.index.get(x));
      const rows = run.epochs[run.epochs.length - 1].rows, E = ids.map((i) => rows[i]);
      const strip = (lit) => S.Kit.tokens.strip({ prefix: 'oh:w', x: 0, y: TOPY, width: 790, tokens: words, size: 14, gap: 16, sub: (i) => 'id ' + ids[i], cls: (i) => (i === lit ? 'new' : '') }).items;
      const panel = (rows2, y) => H().lines('oh:p', 560, y || 150, rows2, 13, 22);
      const cw = n > 6 ? 36 : 42;
      // one-hot rows, shown at the columns of these words only
      const oneHot = words.map((_, i) => words.map((__, j) => (i === j ? 1 : 0)));
      const ohGrid = (lit, title) => P.grid('oh:m', WX, 150, oneHot, { cw, scheme: 'seq', max: 1, d: 0, rows: words, cols: ids.map(String), title, cls: (i) => (i === lit ? 'lit' : '') });
      F.set('nums', 0);
      F.push([...strip(-1), ...panel([[`vocabulary: ${V} words`, 'strong'], ['each word gets an id: its row', 'legend'], ['in every table that follows', 'legend']])],
        `word2vec kept ${P.qpl(q, V, 'word')} that appear at least twice in the corpus, and numbered them in order of how often they appear. Here are ${P.qpl(q, n, 'word')} and their ids.`, { phase: 'vocab', V, ids, n });
      F.set('nums', V);
      F.push([...strip(0), ...ohGrid(0, 'one-hot rows (only these ' + n + ' of ' + V + ' columns)'), ...panel([['one-hot: length ' + V, 'strong'], ['a single 1, at the word’s id', 'legend'], [V - 1 + ' zeros everywhere else', 'legend']])],
        `A one-hot vector has one position per vocabulary word: ${P.qpl(q, V, 'number')}, all zero except a 1 at the word’s own id. “${words[0]}” is the vector with its 1 at position ${q.n(ids[0])}.`, { phase: 'onehot', V, id0: ids[0] });
      // every pair of one-hot vectors: dot product 0, distance √2
      const dist = Math.sqrt(oneHot[0].reduce((s, v, j) => s + (v - oneHot[1][j]) ** 2, 0)), dots = oneHot.map((a) => oneHot.map((b) => a.reduce((s, v, j) => s + v * b[j], 0)));
      F.push([...strip(-1), ...P.grid('oh:m', WX, 150, dots, { cw, scheme: 'seq', max: 1, d: 0, rows: words, cols: words.map((x) => (cw > 40 ? x.slice(0, 5) : x.slice(0, 4))), title: 'dot products of the one-hot rows' }), ...panel([['every pair: dot product 0', 'strong'], ['distance √2 ≈ ' + H().fx(dist, 2) + ' for all', 'legend'], ['“' + words[0] + '” is as far from “' + words[1] + '”', 'legend'], ['as from any other word', 'legend']])],
        `Compare the one-hot vectors pair by pair. Two different words share no position, so every dot product is ${q.n(0)} and every distance is the same, ${q(dist, 2)}. One-hot vectors say only that words differ, never how much.`, { phase: 'onesim', dist, zero: 0, offMax: Math.max(...dots.flatMap((r, i) => r.filter((_, j) => j !== i))) });
      // the learned table
      F.set('nums', d);
      const mx = Math.max(...E.flat().map(Math.abs));
      const tcw = d > 8 ? 22 : 36, table = (lit) => P.grid('oh:e', WX, 150, E, { cw: tcw, ch: cw, scheme: 'div', max: mx, fmt: null, rows: words, cols: d > 8 ? null : Array.from({ length: d }, (_, j) => String(j + 1)), title: 'embedding rows: ' + d + ' learned numbers each', cls: (i) => (i === lit ? 'lit' : '') });
      F.push([...strip(-1), ...table(-1), ...panel([[`table E: ${V} rows × ${d}`, 'strong'], ['blue above zero, red below', 'legend'], ['rows of a kind look alike', 'legend']], 150)],
        `An embedding replaces each one-hot vector by a short row of ${P.qpl(q, d, 'number')}. word2vec learned the table from the corpus (with the average row, which every word shares, taken away). The rows of “${words[0]}” and “${words[1]}” already look alike.`, { phase: 'table', d, V });
      const row0 = E[0], show = Math.min(4, d);
      F.push([...strip(0), ...table(0), ...panel([['one-hot(' + words[0] + ') · E', 'math'], ['= row ' + ids[0] + ' of E', 'math'], ['= ' + P.vec(row0.slice(0, show)) + (d > show ? ' …' : ''), 'math'], ['a lookup, no arithmetic', 'legend']], 150)],
        `Multiplying the one-hot vector of “${words[0]}” by the table adds up ${P.qpl(q, V, 'row')}, each times 0 except row ${q.n(ids[0])} times 1, so the product is that row: ${P.qvec(q, row0.slice(0, show))}${d > show ? ' and so on' : ''}. An embedding layer is exactly this product, done as a lookup.`, { phase: 'lookup', id0: ids[0], row0: row0.slice(0, show), V });
      // cosine similarity of the learned rows
      const C = E.map((a) => E.map((b) => S.DL.cosine(a, b)));
      let best = [0, 1], worst = [0, 1];
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { if (C[i][j] > C[best[0]][best[1]]) best = [i, j]; if (C[i][j] < C[worst[0]][worst[1]]) worst = [i, j]; }
      const near0 = C[0].map((v, j) => (j === 0 ? -Infinity : v)), nb0 = P.argmax(near0);
      F.push([...strip(-1), ...P.grid('oh:m', WX, 150, C, { cw: 48, ch: cw, scheme: 'div', max: 1, d: 2, rows: words, cols: words.map((x) => x.slice(0, 6)), title: 'cosine similarity of the rows', cls: (i, j) => ((i === best[0] && j === best[1]) || (i === best[1] && j === best[0]) ? 'lit' : '') }), ...panel([['most alike: ' + words[best[0]] + ', ' + words[best[1]], 'strong'], ['cosine ' + H().fx(C[best[0]][best[1]], 2), 'legend'], ['least: ' + words[worst[0]] + ', ' + words[worst[1]], 'strong'], ['cosine ' + H().fx(C[worst[0]][worst[1]], 2), 'legend']])],
        `Now similarity means something. The cosine of two rows (1 for the same direction, 0 for unrelated, −1 for opposite) is highest for “${words[best[0]]}” and “${words[best[1]]}”, ${q(C[best[0]][best[1]], 2)}, and lowest for “${words[worst[0]]}” and “${words[worst[1]]}”, ${q(C[worst[0]][worst[1]], 2)}. The closest word to “${words[0]}” here is “${words[nb0]}”.`, { phase: 'cos', best, worst, cBest: C[best[0]][best[1]], cWorst: C[worst[0]][worst[1]], nb0, C });
      F.push([...strip(-1), ...panel([['one-hot', 'strong'], [V + ' numbers per word, fixed', 'legend'], ['no notion of “alike”', 'legend'], ['embedding', 'strong'], [d + ' numbers per word, learned', 'legend'], [V + ' × ' + d + ' = ' + V * d + ' numbers in E', 'legend']], 70), ...P.grid('oh:e', WX, 150, E, { cw: tcw, ch: cw, scheme: 'div', max: mx, fmt: null, rows: words, cols: d > 8 ? null : Array.from({ length: d }, (_, j) => String(j + 1)), title: 'embedding rows: ' + d + ' learned numbers each' })],
        `So a word costs ${P.qpl(q, V, 'number')} as a one-hot vector and ${P.qpl(q, d, 'number')} as an embedding, and the whole table holds ${q.n(V * d)} learned numbers. With a real vocabulary of tens of thousands of words the saving is huge, and the rows carry meaning that the next layers can use.`, { phase: 'size', V, d, total: V * d });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'One-hot and embedding', text: ['one-hot(w): length V, 1 at id(w)', 'embedding: E is V × d', 'e(w) = one-hot(w) · E = row id(w) of E', 'cos(a, b) = a·b / (|a| |b|)'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'onesim', once: true, pick: 'option',
      ask: (fr, i, inp) => 'In the learned table, which word will be closest to “' + inp.words[0] + '”?',
      options: (fr, i, inp) => inp.words.slice(1),
      answer: (fr, i, inp) => inp.words[fr.find((f) => f.meta.phase === 'cos').meta.nb0],
      explain: (fr) => fr.find((f) => f.meta.phase === 'cos').say,
    }],
    quiz: [
      { q: 'With a vocabulary of 50,000 words, how long is a one-hot vector?', options: ['50,000 numbers', 'the embedding size', 'one number', 'the length of the sentence'], answer: 0, why: 'One position per vocabulary word.' },
      { q: 'What is the dot product of the one-hot vectors of two different words?', options: ['0', '1', '−1', 'it depends on the words'], answer: 0, why: 'Their single 1s sit in different places.' },
      { q: 'Multiplying a one-hot row vector by an embedding table E gives…', options: ['the row of E for that word', 'the sum of all rows', 'a single number', 'the column of E for that word'], answer: 0, why: 'Every other row is multiplied by 0.' },
      { q: 'Why do “cat” and “dog” get similar embedding rows?', options: ['they appear among the same neighbouring words', 'their spellings are alike', 'someone set them by hand', 'they have neighbouring ids'], answer: 0, why: 'word2vec learns from the company words keep.' },
      { q: 'A vocabulary of 1,000 words with 16-number embeddings has a table of…', options: ['16,000 numbers', '1,016 numbers', '1,000 numbers', '256 numbers'], answer: 0, why: 'V × d = 1,000 × 16.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
