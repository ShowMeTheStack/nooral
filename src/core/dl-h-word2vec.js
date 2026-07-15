// word2vec: skip-gram with negative sampling on the original corpus. Each word has an input vector (its embedding)
// and an output vector; for a centre word c and a word o seen near it, training raises σ(u_o·v_c) and lowers
// σ(u_k·v_c) for k negatives drawn from the unigram counts raised to 3/4.
//   const w = new DL.Word2Vec(text?, {dim: 16, window: 4, negatives: 5, minCount: 3, epochs: 15, lr: 0.05, subsample: 1e-3, seed})
//      (text defaults to the prose corpus and the labelled sentences; frequent words are subsampled each epoch)
//   w.step(positions = 500) -> {loss, epoch, done}   w.fit() -> w   w.losses (mean loss per epoch)
//   w.vector(word)  w.nearest(word, k)  w.analogy(a, b, c, k)  ("a is to b as c is to ?": b − a + c)
//   w.project(words?) -> {words, points, explained}   w.snapshot(n) the n most frequent words in 2-D, for pictures
//   DL.w2vPair(v, u, negs) -> {loss, gv, gu, gneg}   the loss of one pair and its gradients (checked in the tests)
//   job 'word2vec' (DL.train(params, opts, onMessage, 'word2vec')) streams the loss per chunk of positions
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});
  const fmt = (x) => S.T.fmt(x);
  const sig = (x) => (x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
  const dot = (a, ai, b, bi, d) => { let s = 0; for (let j = 0; j < d; j++) s += a[ai + j] * b[bi + j]; return s; };

  /** one positive pair (v centre, u context) and negative output vectors negs: loss and gradients */
  DL.w2vPair = (v, u, negs) => {
    const d = v.length, sp = sig(dot(u, 0, v, 0, d)), gv = new Float64Array(d), gu = new Float64Array(d);
    let loss = -Math.log(sp);
    for (let j = 0; j < d; j++) { gv[j] += (sp - 1) * u[j]; gu[j] = (sp - 1) * v[j]; }
    const gneg = negs.map((n) => {
      const sn = sig(dot(n, 0, v, 0, d)), g = new Float64Array(d);
      loss -= Math.log(1 - sn);
      for (let j = 0; j < d; j++) { gv[j] += sn * n[j]; g[j] = sn * v[j]; }
      return g;
    });
    return { loss, gv, gu, gneg };
  };

  class Word2Vec {
    constructor(text, o = {}) {
      this.o = Object.assign({ dim: 16, window: 4, negatives: 5, minCount: 3, epochs: 15, lr: 0.05, subsample: 1e-3, seed: 1 }, o);
      const words = DL.words(text || DL.Word2Vec.corpus()), cnt = new Map();
      for (const w of words) cnt.set(w, (cnt.get(w) || 0) + 1);
      this.vocab = [...cnt.entries()].filter(([, n]) => n >= this.o.minCount).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map((x) => x[0]);
      this.counts = this.vocab.map((w) => cnt.get(w));
      this.index = new Map(this.vocab.map((w, i) => [w, i]));
      this.ids = Int32Array.from(words.filter((w) => this.index.has(w)).map((w) => this.index.get(w)));
      const V = this.vocab.length, d = this.o.dim;
      this.rng = S.T.rng(this.o.seed);
      this.W = new Float32Array(V * d).map(() => (this.rng.next() - 0.5) / d);
      this.C = new Float32Array(V * d);
      // the negative-sampling table: word i fills a share of it proportional to count^0.75
      const pw = this.counts.map((c) => Math.pow(c, 0.75)), z = pw.reduce((s, x) => s + x, 0), size = Math.max(1000, V * 20);
      this.table = new Int32Array(size);
      let at = 0;
      pw.forEach((p, i) => { const m = Math.round((p / z) * size); for (let k = 0; k < m && at < size; k++) this.table[at++] = i; });
      while (at < size) this.table[at++] = 0;
      this.pos = 0; this.epoch = 0; this.losses = []; this.lossSum = 0; this.lossN = 0; this.snapshots = [];
      this.total = this.o.epochs * this.ids.length;
      // frequent words are dropped at random each epoch (kept with probability (√(f/t) + 1)·t/f), so "the" and "a"
      // do not crowd out the rarer words around them
      const tot = this.ids.length, t = this.o.subsample;
      this.keep = this.counts.map((c) => (t ? Math.min(1, (Math.sqrt(c / tot / t) + 1) * (t * tot) / c) : 1));
      this.cur = this.sampleEpoch();
      this.trace = [DL.step('start', 'The corpus has ' + words.length + ' words; ' + V + ' appear at least ' + this.o.minCount + ' times and get a ' + d + '-number vector each.', { words: words.length, vocab: V })];
    }
    /** train on the next n positions of the corpus; the learning rate falls linearly to 1% */
    sampleEpoch() { return this.o.subsample ? Int32Array.from(Array.from(this.ids).filter((i) => this.rng.next() < this.keep[i])) : this.ids; }
    step(n = 500) {
      const { window: win, negatives: K, dim: d } = this.o, ids = this.cur, N = ids.length, N0 = this.ids.length, W = this.W, C = this.C, g = this.rng;
      const gv = new Float32Array(d);
      let loss = 0, pairs = 0;
      for (let s = 0; s < n && this.epoch < this.o.epochs; s++) {
        const done = this.epoch * N0 + (this.pos * N0) / N, lr = this.o.lr * Math.max(0.01, 1 - done / this.total);
        const c = ids[this.pos], b = 1 + g.int(win), vc = c * d;
        for (let k = this.pos - b; k <= this.pos + b; k++) {
          if (k < 0 || k >= N || k === this.pos) continue;
          gv.fill(0);
          for (let m = 0; m <= K; m++) {
            const o = m === 0 ? ids[k] : this.table[g.int(this.table.length)];
            if (m > 0 && o === ids[k]) continue;
            const label = m === 0 ? 1 : 0, uo = o * d, sc = sig(dot(C, uo, W, vc, d)), e = sc - label;
            const l = -Math.log(label ? Math.max(sc, 1e-12) : Math.max(1 - sc, 1e-12));
            loss += l; this.lossSum += l;
            for (let j = 0; j < d; j++) { gv[j] += e * C[uo + j]; C[uo + j] -= lr * e * W[vc + j]; }
          }
          for (let j = 0; j < d; j++) W[vc + j] -= lr * gv[j];
          pairs++; this.lossN++;
        }
        if (++this.pos >= N) { this.endEpoch(); }
      }
      return { loss: pairs ? loss / pairs : 0, epoch: this.epoch, done: this.epoch >= this.o.epochs };
    }
    endEpoch() {
      const mean = this.lossN ? this.lossSum / this.lossN : 0;
      this.losses.push(mean);
      this.trace.push(DL.step('epoch', 'Epoch ' + (this.epoch + 1) + ': mean loss per pair ' + fmt(mean) + ' (a pair with its negatives).', { epoch: this.epoch + 1, loss: mean }));
      this.pos = 0; this.epoch++; this.lossSum = 0; this.lossN = 0;
      if (this.epoch < this.o.epochs) this.cur = this.sampleEpoch();
      if (this.o.snapshotWords) this.snapshots.push(Object.assign({ epoch: this.epoch }, this.snapshot(this.o.snapshotWords)));
    }
    fit() { while (this.epoch < this.o.epochs) this.step(5000); return this; }
    vector(w) { const i = typeof w === 'number' ? w : this.index.get(w), d = this.o.dim; if (i === undefined) return null; return this.W.subarray(i * d, (i + 1) * d); }
    /** the vectors with their mean removed: the shared direction every word drifts along says nothing about meaning */
    rows() {
      const V = this.vocab.length, d = this.o.dim, mean = new Float32Array(d), out = new Float32Array(V * d);
      for (let i = 0; i < V * d; i++) mean[i % d] += this.W[i] / V;
      for (let i = 0; i < V * d; i++) out[i] = this.W[i] - mean[i % d];
      return this.vocab.map((_, i) => out.subarray(i * d, (i + 1) * d));
    }
    nearest(w, k = 5) {
      const v = this.vector(w);
      if (!v) return [];
      return DL.nearest(this.rows(), typeof w === 'number' ? w : this.index.get(w), k, { labels: this.vocab });
    }
    /** a : b :: c : ?   the words nearest to b − a + c, leaving out the three given */
    analogy(a, b, c, k = 3) {
      if (![a, b, c].every((w) => this.index.has(w))) return { answers: [], say: 'One of the words is not in the vocabulary.' };
      const rows = this.rows(), [va, vb, vc] = [a, b, c].map((w) => rows[this.index.get(w)]);
      const q = vb.map((x, j) => x - va[j] + vc[j]);
      const answers = DL.nearest(rows, q, k, { labels: this.vocab, exclude: [a, b, c].map((w) => this.index.get(w)) });
      return { answers, say: a + ' is to ' + b + ' as ' + c + ' is to ' + (answers[0] ? answers[0].label + ' (cosine ' + fmt(answers[0].score) + ')' : '?') + '.' };
    }
    project(words) {
      const list = (words || this.vocab).filter((w) => this.index.has(w));
      const rows = this.rows(), p = DL.project2d(list.map((w) => rows[this.index.get(w)]));
      return { words: list, points: p.points, explained: p.explained };
    }
    snapshot(n = 40) { return this.project(this.vocab.slice(0, n)); }
  }
  DL.Word2Vec = Word2Vec;
  /** the default text: the prose and dialogue corpus and the labelled sentences, all written for Nooral */
  Word2Vec.corpus = () => [S.Data.proseText()].concat(Object.values(S.Data.textSets || {}).flatMap((rows) => rows.map((r) => r[0]))).join('\n');

  DL.jobs = DL.jobs || {};
  DL.jobs.word2vec = (p) => {
    const w = new Word2Vec(p.text, p);
    return {
      model: null,
      step: () => { const r = w.step(p.chunk || 400); return { loss: r.loss, metrics: { epoch: r.epoch }, stop: r.done ? 'trained ' + w.o.epochs + ' epochs' : undefined }; },
      tensors: () => ({ embeddings: S.T.make(w.W, [w.vocab.length, w.o.dim]) }),
      checkpoint: () => ({ vocab: w.vocab, W: Array.from(w.W) }),
    };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
