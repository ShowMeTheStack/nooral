// Part 12, how a transformer reads its input: tokens become ids, ids pick rows of a learned embedding matrix, and
// (because attention by itself cannot tell positions apart) a position vector is added to each row: fixed sinusoids
// or learned rows. The embeddings shown are those of the tiny trained transformer of [[transformer-train]].
(function (S) {
  'use strict';
  const P = S.P12, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const TASK_OPTS = [{ value: 'reverse', label: 'reverse the digits' }, { value: 'copy', label: 'copy the digits' }, { value: 'sort', label: 'sort the digits' }];
  P.TASK_OPTS = TASK_OPTS;
  const VOC = ['0', '1', '2', '3', '4', '5', '6', '7', '='];

  S.topic({
    id: 'transformer-tokens', family: 'p12-attn', order: 10, title: 'Tokens and embeddings',
    blurb: 'A transformer never sees letters or digits, only token ids, and each id is turned into a vector by looking up its row in an embedding matrix. That lookup is the same as multiplying a one-hot vector by the matrix. The rows are learned: here they belong to a tiny transformer that learned to reverse, copy or sort five digits, with a vocabulary of nine tokens (the digits 0 to 7 and “=”).',
    inputs: [
      { id: 'task', kind: 'choice', label: 'Trained model', def: 'reverse', options: TASK_OPTS },
      { id: 'digits', kind: 'string', label: 'Five digits (0 to 7)', def: '27405', alphabet: '01234567', min: 5, max: 5 },
      { id: 'pick', kind: 'number', label: 'Position to follow', def: 1, min: 0, max: 9 },
    ],
    presets: [
      { label: 'Reverse 2 7 4 0 5', values: { task: 'reverse', digits: '27405', pick: 1 } },
      { label: 'A repeated digit', values: { task: 'reverse', digits: '31337', pick: 1 } },
      { label: 'Sort model', values: { task: 'sort', digits: '62517', pick: 5 } },
      { label: 'Copy model', values: { task: 'copy', digits: '04471', pick: 3 } },
    ],
    counters: [{ id: 'rows', label: 'rows looked up' }],
    run(inp) {
      const F = H().frames(['rows']), q = F.q, net = P.net(inp.task), x = [...inp.digits].map(Number), ids = P.seqOf(inp.task, x), n = ids.length, k = inp.pick, t = ids[k];
      const E = net.emb.W.toArray(), d = E[0].length, toks = ids.map(P.tok), max = Math.max(...E.flat().map(Math.abs));
      const strip = (lit) => [H().text('tk:l1', 98, 60, 'token', 'axname', 13, 'end'), H().text('tk:l2', 98, 84, 'id', 'axname', 13, 'end'), ...P.strip('tk', 110, 40, toks, { gap: 10, sub: (i) => String(ids[i]), cls: (i) => (lit && i === k ? 'p12ask' : lit && ids[i] === t ? 'p12src' : '') }).items];
      const emat = (row) => P.mat('te', 90, 150, E, { rows: VOC, cols: Array.from({ length: d }, (_, j) => (j % 4 === 0 ? String(j) : '')), cw: 26, ch: 20, gap: 2, fmt: null, max, title: 'embedding matrix E: ' + VOC.length + ' tokens × ' + d + ' numbers', rowCls: (i) => (i === row ? 'strong' : ''), cls: (i) => (i === row ? 'act' : '') }).items;
      F.push([...strip(false)], `The input is the ${P.qpl(q, x.length, 'digit')}, the separator “=” and the answer so far: ${P.qpl(q, n, 'token')}. Each token is just an id, from a vocabulary of ${q.n(VOC.length)}.`, { phase: 'ids', n, vocab: VOC.length, digits: x.length });
      F.count('rows');
      F.push([...strip(true), ...emat(t)], `Each id owns one row of the embedding matrix, ${q.n(VOC.length)} rows of ${q.n(d)} numbers (colour: blue positive, red negative). Position ${q.n(k)} holds “${P.qtok(q, t)}”, id ${q.n(t)}, so its vector is row ${q.n(t)}, starting ${P.qvec(q, E[t].slice(0, 4))}.`, { phase: 'row', k, t, tok: t, row: E[t].slice(0, 4), d, vocab: VOC.length });
      const one = VOC.map((_, i) => (i === t ? 1 : 0));
      F.push([...strip(true), ...emat(t), ...P.mat('to', 90, 410, [one], { rows: ['one-hot'], cols: VOC, cw: 26, ch: 22, scheme: 'seq', max: 1, fmt: (v) => String(v), title: 'one-hot × E picks row ' + t, cls: (i, j) => (j === t ? 'act' : '') }).items],
        `The lookup is a matrix product in disguise: a one-hot vector, zero everywhere but at place ${q.n(t)}, times E gives exactly row ${q.n(t)}. A one-hot vector has ${q.n(VOC.length)} numbers and says nothing about which tokens are alike; the embedding has ${q.n(d)} and is trained, so tokens used alike can end up close.`, { phase: 'onehot', t, vocab: VOC.length, d });
      const same = ids.map((v, i) => (v === t ? i : -1)).filter((i) => i >= 0);
      F.set('rows', n);
      const X = ids.map((v) => E[v]);
      F.push([...strip(true), ...P.mat('tx', 90, 150, X, { rows: toks.map((s, i) => i + ': ' + s), cols: Array.from({ length: d }, (_, j) => (j % 4 === 0 ? String(j) : '')), cw: 26, ch: 20, fmt: null, max, title: 'the sequence as a matrix: ' + n + ' × ' + d, cls: (i) => (same.includes(i) ? 'act' : '') }).items],
        `Stacking the rows gives the input matrix, ${q.n(n)} × ${q.n(d)}: one row per position.` + (same.length > 1 ? ` “${P.qtok(q, t)}” appears at ${P.qpl(q, same.length, 'position')} (${H().list(same.map((i) => q.n(i)))}) and gets the same row each time: nothing yet says where a token stands.` : ` A token that appeared twice would get the same row twice: nothing yet says where a token stands.`), { phase: 'stack', n, d, same, count: same.length, t });
      const D = Array.from({ length: 8 }, (_, a) => Array.from({ length: 8 }, (_, b) => P.cos(E[a], E[b])));
      let ba = 0, bb = 1;
      for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) if (D[a][b] > D[ba][bb]) { ba = a; bb = b; }
      F.push([...strip(false), ...P.mat('ts', 90, 150, D, { rows: VOC.slice(0, 8), cols: VOC.slice(0, 8), cw: 40, ch: 26, max: 1, title: 'cosine similarity of the digit rows', cls: (i, j) => ((i === ba && j === bb) || (i === bb && j === ba) ? 'act' : '') }).items],
        `How alike are the learned rows? The cosine similarity of two rows is largest, one, when they point the same way. The closest pair of digits is ${q.n(ba)} and ${q.n(bb)}, at ${q(D[ba][bb], 2)}. Nothing told the model that digits are numbers; for this task it only needs every digit to be recognisable.`, { phase: 'similar', a: ba, b: bb, cos: D[ba][bb] });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Embedding lookup', text: ['token → id i', 'x = E[i]  (row i of E)', '  = one-hot(i) · E', 'E has one row per token in the vocabulary'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'onehot', once: true, pick: 'option',
      ask: 'The whole sequence is stacked into one matrix. How many numbers does it hold?',
      options: () => ['160', '10', '16', '144'],
      answer: '160',
      explain: 'Ten positions, each a row of sixteen numbers.',
    }],
    quiz: [
      { q: 'Multiplying a one-hot vector for id 3 by the embedding matrix gives…', options: ['row 3 of the matrix', 'column 3', 'the sum of all rows', 'a single number'], answer: 0, why: 'Only the 1 at place 3 survives the product.' },
      { q: 'A vocabulary of 50,000 tokens with vectors of 768 numbers needs an embedding matrix of…', options: ['50,000 × 768 numbers', '768 numbers', '50,000 numbers', '768 × 768 numbers'], answer: 0, why: 'One row of 768 numbers for every token.' },
      { q: 'Before positions are added, two copies of the same token in a sentence get…', options: ['the same vector', 'different vectors', 'no vector', 'vectors of different lengths'], answer: 0, why: 'The lookup depends only on the id.' },
      { q: 'Why are learned embeddings better than one-hot vectors as input?', options: ['they are short and similar tokens can get similar vectors', 'they are always exact', 'they need no training', 'they remember the position'], answer: 0, why: 'One-hot vectors are long and every pair of them is equally far apart.' },
    ],
  });

  S.topic({
    id: 'transformer-positions', family: 'p12-attn', order: 20, title: 'Positional encodings: sinusoids and learned rows',
    blurb: 'Attention compares every position with every other in the same way, so on its own it cannot tell a sequence from a shuffled one. Transformers therefore add a position vector to each token’s embedding. One classic choice is fixed: sines and cosines whose wavelengths grow from one dimension to the next, so every position gets a unique pattern and nearby positions look alike. The other is learned: one trained row per position, like a second embedding table.',
    inputs: [
      { id: 'd', kind: 'choice', label: 'Width d of the sinusoids', def: '32', options: [{ value: '16', label: '16' }, { value: '32', label: '32' }, { value: '64', label: '64' }] },
      { id: 'len', kind: 'number', label: 'Positions shown', def: 40, min: 12, max: 60 },
      { id: 'off', kind: 'number', label: 'Offset between two positions', def: 4, min: 1, max: 10 },
      { id: 'task', kind: 'choice', label: 'Learned rows of', def: 'reverse', options: TASK_OPTS },
    ],
    presets: [
      { label: 'd = 32, 40 positions', values: { d: '32', len: 40, off: 4, task: 'reverse' } },
      { label: 'Narrow, d = 16', values: { d: '16', len: 24, off: 2, task: 'copy' } },
      { label: 'Wide, d = 64', values: { d: '64', len: 60, off: 8, task: 'sort' } },
    ],
    counters: [{ id: 'pos', label: 'positions encoded' }],
    run(inp) {
      const T = S.T, F = H().frames(['pos']), q = F.q, d = +inp.d, N = inp.len, off = inp.off;
      // 1. attention alone does not see order: the trained model's first-layer attention, without positions or mask
      const net = P.net(inp.task), blk = net.blocks[0], ids = [2, 7, 4, 0, 5], perm = [3, 0, 4, 1, 2];
      const outOf = (seq) => T.noGrad(() => { const e = T.embedding(net.emb.W, seq), h = blk.ln1.forward(T.reshape(e, [1, seq.length, -1])); return blk.attn.forward(h, { causal: false }).toArray()[0]; });
      const o1 = outOf(ids), o2 = outOf(perm.map((p) => ids[p]));
      let diff = 0; perm.forEach((p, i) => o2[i].forEach((v, j) => { diff = Math.max(diff, Math.abs(v - o1[p][j])); }));
      const st = (toks, y, key) => P.strip(key, 110, y, toks, { size: 15, gap: 8, cls: (i) => (toks[i] === '7' ? 'p12ask' : '') }).items;
      const i7 = perm.findIndex((p) => ids[p] === 7), omax = Math.max(...o1.flat().map(Math.abs));
      const outRow = (key, y, v, title) => P.mat(key, 110, y, [v], { rows: false, cols: false, cw: 30, ch: 26, fmt: null, max: omax, title }).items;
      F.push([H().text('pa:h', 110, 52, 'the digits in order', 'strong', 13), ...st(ids.map(P.tok), 64, 'pa'), H().text('pb:h', 110, 142, 'the same digits shuffled', 'strong', 13), ...st(perm.map((p) => P.tok(ids[p])), 154, 'pb'),
        ...outRow('po1', 266, o1[1], 'attention output for “7”, in order (16 numbers)'), ...outRow('po2', 346, o2[i7], 'attention output for “7”, shuffled')],
        `Without positions, attention is blind to order. The trained model’s first attention layer, run on ${P.qpl(q, ids.length, 'digit')} and on the same digits shuffled, gives each digit the same output in both${diff ? ': the largest difference is ' + q.sg(diff, 2) + ', no more than rounding' : ', exactly'}. “${P.qtok(q, 7)}” has no idea whether it came first or second.`, { phase: 'blind', n: ids.length, diff, seven: 7 });
      // 2. the sinusoids
      const PE = T.nn.sinusoid(N, d).toArray();
      const rx = 110, ry = 64, rw = 360, rh = 300;
      const ras = { key: 'ps:r', kind: 'raster', x: rx, y: ry, w: rw, h: rh, nx: d, ny: N, vals: PE.flat(), scheme: 'div', max: 1 };
      const rasLab = [H().text('ps:t', rx, ry - 14, 'sinusoidal encoding: ' + N + ' positions × ' + d + ' dimensions', 'strong', 13), H().text('ps:x', rx + rw, ry + rh + 20, 'dimension →', 'axname', 13, 'end'), H().text('ps:y0', rx - 8, ry + 12, '0', 'tick', 12, 'end'), H().text('ps:y1', rx - 8, ry + rh, String(N - 1), 'tick', 12, 'end'), H().text('ps:yn', rx - 8, ry + rh / 2, 'position', 'axname', 13, 'end')];
      F.set('pos', N);
      F.push([ras, ...rasLab], `The fixed encoding fills a ${q.n(N)} × ${q.n(d)} table: dimension 2i holds sin(p / 10000^(2i/d)) and the next one the cosine of the same. The first columns flicker fast (a wavelength of ${q(2 * Math.PI, 2)} positions); the last pair barely changes, wavelength ${q.sg(2 * Math.PI * Math.pow(10000, (d - 2) / d), 3)}. Each row is a different pattern.`, { phase: 'table', N, d, w0: 2 * Math.PI, wl: 2 * Math.PI * Math.pow(10000, (d - 2) / d) });
      const A = C().axes({ prefix: 'pw', x: 530, y: 64, w: 260, h: 140, xr: [0, N - 1], yr: [-1, 1], yticks: [-1, 0, 1], xname: 'position', yname: 'value' });
      const dims = [8, 24, 80].map((wl) => Math.min(d - 2, 2 * Math.round((d * Math.log(wl / (2 * Math.PI))) / Math.log(10000) / 2)));
      const waves = dims.map((j, k) => C().line(A, 'pw:l' + k, PE.map((r, p) => [p, r[j]]), ['p12a', 'p12b', 'p12c'][k]));
      const leg = C().legend('pw:g', 530, 264, dims.map((j, k) => ({ text: 'dim ' + j, cls: ['p12a', 'p12b', 'p12c'][k] })), { size: 12 });
      F.push([ras, ...rasLab, ...A.items, ...waves, ...leg], `Read down one column and it is a wave over the positions: dimension ${q.n(dims[0])} turns fast, dimension ${q.n(dims[2])} much more slowly. Together the columns act like the hands of a clock, each turning at its own speed, so no two positions show the same time.`, { phase: 'waves', dims });
      // 3. offsets: the dot product of p and p + k depends only on k
      const dotp = (a, b) => P.dot(PE[a], PE[b]), p1 = 2, p2 = Math.min(N - 1 - off, 2 + 3 * off);
      const sim = Array.from({ length: N }, (_, p) => dotp(Math.floor(N / 2), p));
      const B = C().axes({ prefix: 'po', x: 530, y: 300, w: 260, h: 120, xr: [0, N - 1], yr: [Math.floor(Math.min(...sim)), Math.ceil(d / 2)], yticks: [0, d / 2], xname: 'position p', yname: 'PE(' + Math.floor(N / 2) + ') · PE(p)' });
      F.push([ras, ...rasLab, ...A.items, ...waves, ...leg, ...B.items, C().line(B, 'po:l', sim.map((v, p) => [p, v]), 'p12a'), C().vline(B, 'po:v', Math.floor(N / 2))],
        `Because sin and cos of a shifted angle are fixed mixtures of the unshifted ones, the dot product of two positions depends only on how far apart they are: PE(${q.n(p1)}) · PE(${q.n(p1 + off)}) = ${q(dotp(p1, p1 + off), 3)} and PE(${q.n(p2)}) · PE(${q.n(p2 + off)}) = ${q(dotp(p2, p2 + off), 3)}, both ${q.n(off)} apart. Below, it peaks at ${q(d / 2, 0)} for a position with itself and falls away with distance.`, { phase: 'offset', p1, p2, q1: p1 + off, q2: p2 + off, off, a: dotp(p1, p1 + off), b: dotp(p2, p2 + off), peak: d / 2 });
      // 4. learned rows
      const Pl = net.pos.P.toArray(), m = Pl.length, cosM = Pl.map((a) => Pl.map((b) => P.cos(a, b)));
      let near = 0, far = 0, cn = 0, cf = 0;
      for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) { if (Math.abs(a - b) === 1) { near += cosM[a][b]; cn++; } else if (Math.abs(a - b) >= 5) { far += cosM[a][b]; cf++; } }
      const names = Array.from({ length: m }, (_, i) => String(i));
      F.push([...P.mat('pl', 130, 70, cosM, { rows: names, cols: names, cw: 40, ch: 30, fmt: (v) => H().fx(v, 1), max: 1, size: 12, title: 'learned rows: cosine similarity of position vectors' }).items, H().text('pl:n', 130, 440, 'the ' + inp.task + ' model: ' + m + ' positions × ' + Pl[0].length + ' numbers, trained', 'p12hint', 12)],
        `The trained transformer learned its own ${P.qpl(q, m, 'position row')} instead. Neighbouring positions have mean cosine similarity ${q(near / cn, 2)}, positions five or more apart ${q(far / cf, 2)}. Learned rows can fit the task exactly, but there is no row for a position beyond the ${q.n(m)} it was trained on, while the sinusoids go on for ever.`, { phase: 'learned', m, near: near / cn, far: far / cf, five: 5 });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Sinusoidal encoding', text: ['PE(p, 2i)   = sin(p / 10000^(2i/d))', 'PE(p, 2i+1) = cos(p / 10000^(2i/d))', 'input row = embedding(token) + PE(p)', 'learned: PE(p) = row p of a trained table'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'waves', once: true, pick: 'option',
      ask: 'Will PE(2) · PE(2 + k) be larger, smaller or the same as the dot product of two other positions k apart?',
      options: ['the same', 'larger', 'smaller'],
      answer: 'the same',
      explain: 'Each sine–cosine pair contributes cos(ω·k), which depends only on the distance k.',
    }],
    quiz: [
      { q: 'Without position information, a transformer reading “dog bites man” and “man bites dog”…', options: ['sees the same set of vectors, only reordered', 'sees completely different vectors', 'cannot run', 'reads them backwards'], answer: 0, why: 'Attention treats its inputs as a set; order must be added.' },
      { q: 'In the sinusoidal encoding, the first dimensions…', options: ['change fastest from position to position', 'never change', 'are all zero', 'are learned'], answer: 0, why: 'Their wavelength is about 6.3 positions; later ones are far longer.' },
      { q: 'A learned position table trained with 512 positions meets a 600-token input. It…', options: ['has no row for positions 512 and beyond', 'extends itself automatically', 'repeats rows from the start, by design', 'ignores the extra tokens without loss'], answer: 0, why: 'There are only as many learned rows as were trained.' },
      { q: 'How is the position vector combined with the token embedding in these models?', options: ['added to it', 'multiplied into it', 'used instead of it', 'appended as one extra number'], answer: 0, why: 'Both have the same width and are summed.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
