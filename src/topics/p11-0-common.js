// Part 11, sequences and embeddings: the families, the original corpus the topics train on (sentences made from
// templates of our own about kings and goats, bread and boats, so that words used alike end up alike), the trained
// models shipped in p11-0-data.js rebuilt as engine networks, a small encoder–decoder made of engine layers, and
// the pieces of picture the topics share.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'sequences', order, title, blurb });
  fam('p11-embed', 10, 'Words as vectors', 'From one-hot vectors, where every word is equally far from every other, to embeddings learned from the company words keep: word2vec trained here on text of our own, its map and its nearest neighbours.');
  fam('p11-rnn', 20, 'Recurrent networks', 'One cell applied at every step with the same weights, a hidden state carried along, gradients that fade or blow up on the way back, and the gated cells (LSTM and GRU) that fix it.');
  fam('p11-seq', 30, 'Sequence models at work', 'A character model predicting the next letter, and an encoder–decoder squeezing a whole input into one vector before writing the output.');

  const P = (S.P11 = {});
  const H = () => S.Kit.ml;
  const cache = new Map();
  P.once = (k, f) => { if (!cache.has(k)) cache.set(k, f()); return cache.get(k); };

  // ---- the corpus ------------------------------------------------------------------------------------------------
  P.CATS = {
    people: ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl'],
    animals: ['cat', 'dog', 'horse', 'goat', 'sheep', 'rabbit'],
    food: ['bread', 'rice', 'soup', 'apples', 'cheese', 'plums', 'tea', 'milk', 'water', 'juice'],
    colours: ['red', 'blue', 'green', 'yellow', 'white'],
    things: ['coat', 'hat', 'cart', 'door', 'boat'],
    places: ['village', 'market', 'farm', 'river', 'hill'],
    numbers: ['two', 'three', 'four', 'five', 'six'],
  };
  P.catOf = (w) => Object.keys(P.CATS).find((k) => P.CATS[k].includes(w)) || null;
  /**
   * n sentences from templates, chosen by a seeded generator. People come in pairs (he, she), royal or not, old or
   * young, so their vectors can carry those three differences; animals, food, colours, things, places and numbers
   * each fill their own slots, so words of a kind share their neighbours.
   */
  P.corpus = (n, seed) => P.once('corpus:' + n + ':' + seed, () => {
    const r = S.T.rng(seed), pick = (a) => a[r.int(a.length)], C = P.CATS;
    const food = C.food.slice(0, 6), drink = C.food.slice(6);
    const people = [['king', 'queen', 1, 'old'], ['prince', 'princess', 1, 'young'], ['man', 'woman', 0, 'old'], ['boy', 'girl', 0, 'young']];
    const out = [];
    for (let i = 0; i < n; i++) {
      const k = r.int(10), g = r.int(2), he = g ? 'she' : 'he', his = g ? 'her' : 'his';
      const Pp = pick(people), p = Pp[g], age = Pp[3], home = Pp[2] ? 'palace' : 'cottage';
      if (k === 0) out.push(`the ${age} ${p} left the ${home} and ${he} wore ${his} ${Pp[2] ? 'crown' : 'hat'}`);
      else if (k === 1) out.push(`the ${p} walked to the ${pick(C.places)} and ${he} carried ${his} ${pick(C.things)}`);
      else if (k === 2) out.push(`the ${age} ${p} ate ${pick(food)} and drank ${pick(drink)} in the ${home}`);
      else if (k === 3) out.push(`${he} is the ${age} ${p} of the ${Pp[2] ? 'land' : 'village'}`);
      else if (k === 4) out.push(`the ${pick(C.animals)} ran across the ${pick(C.places)} and the ${pick(C.animals)} chased it`);
      else if (k === 5) out.push(`a ${pick(C.colours)} ${pick(C.things)} and a ${pick(C.colours)} ${pick(C.things)} stood by the ${pick(C.places)}`);
      else if (k === 6) out.push(`we bought ${pick(C.numbers)} loaves of ${pick(food)} and ${pick(C.numbers)} cups of ${pick(drink)}`);
      else if (k === 7) out.push(`the ${pick(C.animals)} slept in the barn and the farmer fed the ${pick(C.animals)}`);
      else if (k === 8) out.push(`the ${p} painted ${his} ${pick(C.things)} ${pick(C.colours)}`);
      else out.push(Pp[2] ? `the ${p} ruled the land and ${he} gave orders from the palace` : `the ${p} worked on the farm and ${he} fed the ${pick(C.animals)}`);
    }
    return out.join('. ') + '.';
  });
  /** the text the character model of part 11 was trained on */
  P.charText = () => P.corpus(300, 7);

  /** the words word2vec keeps from the default corpus (seen at least twice), by count, without training */
  P.vocab = () => P.once('vocab', () => {
    const c = new Map();
    for (const w of S.DL.words(P.corpus(P.W2V_DEF.sentences, 1))) c.set(w, (c.get(w) || 0) + 1);
    return [...c.entries()].filter(([, n]) => n >= 2).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map((x) => x[0]);
  });
  P.checkWords = (v) => { const V = P.vocab(), bad = (v || []).find((w) => !V.includes(w)); if (bad !== undefined) return 'Words: "' + bad + '" is not in the vocabulary (try ' + P.CATS.animals.slice(0, 3).join(', ') + ' or ' + P.CATS.food.slice(0, 2).join(', ') + ')'; if (new Set(v).size !== v.length) return 'Words: each word once'; return null; };

  // ---- word2vec, trained here --------------------------------------------------------------------------------------
  P.W2V_DEF = { sentences: 500, seed: 1, dim: 16, window: 2, negatives: 5, epochs: 8, lr: 0.05 };
  /**
   * word2vec on the corpus, one epoch at a time: the vectors after every epoch are kept (as plain rows with the mean
   * removed, the way the engine compares them), with the mean loss of the epoch.
   */
  P.w2v = (o) => {
    o = Object.assign({}, P.W2V_DEF, o || {});
    return P.once('w2v:' + JSON.stringify(o), () => {
      const text = P.corpus(o.sentences, 1);
      const w = new S.DL.Word2Vec(text, { dim: o.dim, window: o.window, negatives: o.negatives, epochs: o.epochs, minCount: 2, lr: o.lr, subsample: 0, seed: o.seed });
      const snap = () => w.rows().map((r) => Array.from(r));
      const W0 = Array.from(w.W);
      const epochs = [{ epoch: 0, loss: null, rows: snap() }];
      while (w.epoch < w.o.epochs) {
        const e = w.epoch;
        while (w.epoch === e) w.step(Math.max(1, w.cur.length - w.pos));
        epochs.push({ epoch: w.epoch, loss: w.losses[w.losses.length - 1], rows: snap() });
      }
      return { w, epochs, text, o, W0 };
    });
  };

  /** the words drawn on the maps: every word of a kind that the vocabulary holds */
  P.mapWords = () => Object.values(P.CATS).flat().filter((x) => P.vocab().includes(x));
  /**
   * a map of words: points (in map units) inside a box, a dot per word in its kind's colour and as many labels as
   * fit without touching (the words in first are labelled before the rest). o: {prefix, box, act, lit: Set, faded,
   * first: [words], range: {u0, u1, v0, v1}} -> {items, X(u), Y(v)}
   */
  P.wordMap = (words, pts, o) => {
    o = o || {};
    const B = o.box, pad = 26, us = pts.map((p) => p[0]), vs = pts.map((p) => p[1]);
    const R = o.range || { u0: Math.min(...us), u1: Math.max(...us), v0: Math.min(...vs), v1: Math.max(...vs) };
    const s = Math.min((B.w - 2 * pad) / (R.u1 - R.u0 || 1), (B.h - 2 * pad) / (R.v1 - R.v0 || 1));
    const ox = B.x + pad + (B.w - 2 * pad - s * (R.u1 - R.u0)) / 2, oy = B.y + pad + (B.h - 2 * pad - s * (R.v1 - R.v0)) / 2;
    const X = (u) => P.r2(ox + (u - R.u0) * s), Y = (v) => P.r2(oy + (R.v1 - v) * s);
    const items = [{ key: (o.prefix || 'wm') + ':box', kind: 'region', x: B.x, y: B.y, w: B.w, h: B.h, rx: 8, cls: 'p11mapbox', audit: 'group' }];
    const lit = o.lit || new Set(), cls = (w) => (P.catOf(w) || 'other') + (w === o.act ? ' act' : '') + (o.faded && w !== o.act && !lit.has(w) ? ' faded' : '');
    words.forEach((w, i) => items.push({ key: (o.prefix || 'wm') + ':d:' + w, kind: 'dot', x: X(pts[i][0]), y: Y(pts[i][1]), r: w === o.act || lit.has(w) ? 6 : 4.5, cls: 'p11w ' + cls(w), audit: 'overlap-ok', layer: 'items', z: 3 }));
    const order = words.map((w, i) => i).sort((a, b) => ((o.first || []).includes(words[b]) ? 1 : 0) - ((o.first || []).includes(words[a]) ? 1 : 0) || (words[a] === o.act ? -1 : words[b] === o.act ? 1 : 0) || (lit.has(words[b]) ? 1 : 0) - (lit.has(words[a]) ? 1 : 0) || a - b);
    const spots = S.Kit.p11.place(order.map((i) => ({ x: X(pts[i][0]), y: Y(pts[i][1]), text: words[i], size: 12, r: 5, must: words[i] === o.act || (o.first || []).includes(words[i]) })), { bounds: { x: B.x + 2, y: B.y + 2, w: B.w - 4, h: B.h - 4 }, avoid: o.avoid });
    order.forEach((i, k) => { const sp = spots[k]; if (sp) items.push({ key: (o.prefix || 'wm') + ':l:' + words[i], kind: 'text', x: sp.x, y: sp.y, text: words[i], anchor: sp.anchor, size: 12, cls: 'p11lab ' + cls(words[i]), layer: 'labels' }); });
    return { items, X, Y, R };
  };
  /** the legend of word kinds, in a column */
  P.kindLegend = (prefix, x, y) => S.Kit.curve.legend(prefix, x, y, Object.keys(P.CATS).map((k) => ({ text: k, cls: k, dot: true })), { dir: 'col', size: 12 });

  // ---- shipped models ------------------------------------------------------------------------------------------------
  const f32 = (s) => { const b = S.Data.fromBase64(s); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); };
  P.f32 = f32;
  /** a state dict from a shipped job: names, shapes and one base64 run of float32 */
  P.stateOf = (job) => {
    const all = f32(job.weights.b64), st = {};
    let at = 0;
    job.weights.names.forEach((n, i) => { const shape = job.weights.shapes[i], size = shape.reduce((a, b) => a * b, 1); st[n] = { shape, data: Array.from(all.subarray(at, at + size)) }; at += size; });
    return st;
  };
  /** the trained network of a shipped job (built once) */
  P.net = (name) => P.once('net:' + name, () => {
    const job = S.P11D[name];
    const net = job.kind === 'seq2seq' ? P.seq2seq(job.config) : S.DL.build(job.kind, job.config);
    net.loadState(P.stateOf(job));
    net.eval();
    return net;
  });
  P.job = (name) => S.P11D[name];

  /**
   * A small encoder–decoder made of engine layers: the encoder GRU reads the source tokens and its last hidden state
   * is the only thing handed on; the decoder GRU starts from it and writes the target one token at a time.
   * config {vin, vout, e, h, seed};  forward(src [B, n], tgtIn [B, m]) -> logits [B, m, vout]
   */
  P.seq2seq = (c) => {
    const T = S.T, nn = T.nn, g = T.rng(c.seed || 1);
    class Seq2Seq extends nn.Module {
      constructor() {
        super();
        this.embIn = new nn.Embedding(c.vin, c.e, { rng: g, std: 0.3 });
        this.enc = new nn.GRU(c.e, c.h, { rng: g });
        this.embOut = new nn.Embedding(c.vout, c.e, { rng: g, std: 0.3 });
        this.dec = new nn.GRU(c.e, c.h, { rng: g });
        this.head = new nn.Dense(c.h, c.vout, { rng: g });
        this.config = c;
      }
      encode(src) { return this.enc.forward(this.embIn.forward(src)); }
      forward(src, tgtIn) { const e = this.encode(src); return this.head.forward(this.dec.forward(this.embOut.forward(tgtIn), e.h)); }
      /** greedy decoding of one source: every step's hidden state and probabilities */
      translate(src, max) {
        return T.noGrad(() => {
          const e = this.encode(T.tensor([src])), encH = this.enc.steps.map((s) => Array.from(s.h.values()));
          let h = e.h, tok = P.S2S.START;
          const steps = [];
          for (let k = 0; k < max; k++) {
            const out = this.dec.forward(this.embOut.forward(T.tensor([[tok]])), h);
            h = out.h;
            const logits = Array.from(this.head.forward(h).values()), mx = Math.max(...logits), ex = logits.map((v) => Math.exp(v - mx)), z = ex.reduce((a, b) => a + b, 0), p = ex.map((v) => v / z);
            const best = p.indexOf(Math.max(...p));
            steps.push({ input: tok, h: Array.from(h.values()), probs: p, id: best });
            tok = best;
            if (best === P.S2S.END) break;
          }
          return { context: Array.from(e.h.values()), encH, steps, out: steps.map((s) => s.id).filter((i) => i !== P.S2S.END) };
        });
      }
    }
    return new Seq2Seq();
  };
  // number words in, digits out: source ids 0..9 are the words zero..nine; target ids 0..9 the digits, then the
  // start and end marks
  // (trained on one to MAXLEN words: tests/ref/p11_runs.mjs, JOBS.seq2seq.maxLen)
  P.S2S = { MAXLEN: 5, WORDS: ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'], START: 10, END: 11, vin: 10, vout: 12 };
  P.S2S.show = (id) => (id === P.S2S.START ? '<s>' : id === P.S2S.END ? '</s>' : String(id));

  // ---- small helpers ---------------------------------------------------------------------------------------------------
  P.r2 = (v) => Math.round(v * 100) / 100;
  P.vec = (v, d) => '[' + Array.from(v).map((x) => H().fx(x, d === undefined ? 2 : d)).join(', ') + ']';
  P.qvec = (q, v, d) => '[' + Array.from(v).map((x) => q(x, d === undefined ? 2 : d)).join(', ') + ']';
  P.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');
  P.softmax = (row) => { const mx = Math.max(...row), ex = row.map((v) => Math.exp(v - mx)), z = ex.reduce((a, b) => a + b, 0); return ex.map((v) => v / z); };
  P.cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return s / (Math.sqrt(na) * Math.sqrt(nb) || 1); };
  /** a matrix as heat cells with its row and column names (fmt null: no numbers in the cells) */
  P.grid = (prefix, x, y, M, o) => {
    o = o || {};
    return S.Kit.heat.grid({ prefix, x, y, cw: o.cw || 34, ch: o.ch || o.cw || 34, vals: M, scheme: o.scheme || 'seq', max: o.max || 1, fmt: o.fmt === null ? null : o.fmt || ((v) => H().fx(v, o.d === undefined ? 2 : o.d)), rows: o.rows || null, cols: o.cols || null, size: o.size || 12, labelSize: o.labelSize || 12, gap: o.gap === undefined ? 2 : o.gap, cls: o.cls, title: o.title, titleCls: 'strong', titleSize: 13, colsAt: o.colsAt }).items;
  };
  P.argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
})(globalThis.Nooral = globalThis.Nooral || {});
