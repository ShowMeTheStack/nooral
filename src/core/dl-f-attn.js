// Attention models, shared pieces: a transformer block's forward pass with an optional key/value cache, the
// sampling rules for generation, and a transformer encoder classifier.
//   DL.blockForward(block, x, {mask, causal, cache}) for an nn.TransformerBlock; cache {k, v} grows by x's steps
//   DL.sample(logits row, {method: 'greedy' | 'sample', temperature, topK, topP}, rng)
//      -> {id, probs (after the rules), top: [{id, p}], kept, say}
//   DL.build('encoder', {vocab, d, heads, layers, maxLen, classes, ff, dropout, seed}) mean-pools the non-padding
//      positions (token 0 is padding, masked out of attention); named outputs embed, block0, block0.attn, ..., pooled, logits
//   tasks: 'topics' (sentence topics), 'majority' {len, n} (which token is most frequent) for the encoder
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});
  const fmt = (x) => S.T.fmt(x);

  DL.blockForward = (blk, x, o = {}) => {
    const T = S.T, a = blk.attn, [B, n] = x.shape, train = blk.training;
    const h = blk.ln1.forward(x);
    let k = a.split(a.Wk.forward(h), B, n), v = a.split(a.Wv.forward(h), B, n);
    const q = a.split(a.Wq.forward(h), B, n);
    if (o.cache) {
      if (o.cache.k) { k = T.concat([o.cache.k, k], 2); v = T.concat([o.cache.v, v], 2); }
      o.cache.k = k; o.cache.v = v;
    }
    const att = T.attention(q, k, v, { mask: o.mask, causal: o.causal, dropout: a.p, rng: a.rng, training: train });
    a.weights = att.attn.weights; a.scores = att.attn.scores;
    let y = a.Wo.forward(T.reshape(T.permute(att, [0, 2, 1, 3]), [B, n, a.d]));
    if (blk.p) y = T.dropout(y, blk.p, blk.rng, train);
    x = T.add(x, y);
    let f = blk.ff2.forward(T[blk.act](blk.ff1.forward(blk.ln2.forward(x))));
    if (blk.p) f = T.dropout(f, blk.p, blk.rng, train);
    return T.add(x, f);
  };

  /** pick the next token from a row of logits */
  DL.sample = (row, o = {}, rng) => {
    rng = S.T.asRng(rng === undefined ? 1 : rng);
    const V = row.length, method = o.method || (o.temperature === 0 ? 'greedy' : 'sample'), temp = o.temperature || 1;
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (row[i] > mx) mx = row[i];
    const p = new Float64Array(V);
    let z = 0;
    for (let i = 0; i < V; i++) { p[i] = Math.exp((row[i] - mx) / (method === 'greedy' ? 1 : temp)); z += p[i]; }
    for (let i = 0; i < V; i++) p[i] /= z;
    const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
    let keep = V;
    if (method !== 'greedy') {
      if (o.topK) keep = Math.min(keep, Math.max(1, o.topK));
      if (o.topP) { let c = 0, m = 0; while (m < keep) { c += p[order[m]]; m++; if (c >= o.topP) break; } keep = m; }
    }
    const kept = new Float64Array(V);
    let kz = 0;
    for (let j = 0; j < keep; j++) { kept[order[j]] = p[order[j]]; kz += p[order[j]]; }
    for (let i = 0; i < V; i++) kept[i] /= kz;
    let id = order[0];
    if (method !== 'greedy') {
      let r = rng.next(), c = 0;
      for (let j = 0; j < keep; j++) { c += kept[order[j]]; if (r < c) { id = order[j]; break; } id = order[j]; }
    }
    const top = order.slice(0, 5).map((i) => ({ id: i, p: kept[i] }));
    const rule = method === 'greedy' ? 'greedy: the most likely token' : 'sampled at temperature ' + fmt(temp) + (o.topK ? ', top-' + o.topK : '') + (o.topP ? ', top-p ' + fmt(o.topP) : '');
    return { id, probs: Array.from(kept), top, kept: keep, say: 'Token ' + id + ' (p = ' + fmt(kept[id]) + '), ' + rule + '; ' + keep + ' of ' + V + ' tokens could be picked.' };
  };

  DL.register('encoder', (c) => {
    const T = S.T, nn = T.nn, Net = DL.makeNet(), g = T.rng(c.seed || 1), d = c.d || 32, L = c.layers || 2;
    class Encoder extends Net {
      constructor() {
        super();
        this.emb = new nn.Embedding(c.vocab, d, { rng: g, std: 0.3 });
        this.pos = new nn.PositionalEmbedding(c.maxLen || 64, d, { rng: g });
        this.blocks = Array.from({ length: L }, () => new nn.TransformerBlock(d, c.heads || 2, { rng: g, ff: c.ff || 2 * d, dropout: c.dropout || 0 }));
        this.lnf = new nn.LayerNorm(d);
        this.head = new nn.Dense(d, c.classes || c.outputs || 2, { rng: g });
        this.snapshotNames = ['block0.attn', 'pooled', 'logits'];
        this.tokenizer = c.tokenizer ? DL.tokenizer(c.tokenizer) : null;
      }
      forward(ids) {
        this.begin();
        const [B, n] = ids.shape, iv = ids.values(), m = T.zeros([B, 1, 1, n]), cnt = T.zeros([B, 1]);
        for (let b = 0; b < B; b++) for (let j = 0; j < n; j++) if (iv[b * n + j] !== 0) { m.data[b * n + j] = 1; cnt.data[b]++; }
        for (let b = 0; b < B; b++) if (!cnt.data[b]) { m.data[b * n] = 1; cnt.data[b] = 1; }
        let h = this.tap('embed', this.pos.forward(this.emb.forward(ids)));
        this.blocks.forEach((blk, i) => { h = this.tap('block' + i, DL.blockForward(blk, h, { mask: m })); this.tap('block' + i + '.attn', blk.attn.weights); });
        h = this.lnf.forward(h);
        const pooled = this.tap('pooled', T.div(T.sum(T.mul(h, T.reshape(m, [B, n, 1])), 1), cnt));
        return this.tap('logits', this.head.forward(pooled));
      }
      extras() { return { embeddings: this.emb.W }; }
    }
    return new Encoder();
  });

  /** word sequences, right-padded with 0, for the encoder */
  const wordTask = (texts, y, classes, o) => {
    const tok = DL.wordTokenizer(texts), seqs = texts.map((t) => tok.encode(t)), L = Math.max(...seqs.map((s) => s.length));
    const ex = seqs.map((s, i) => ({ ids: s.concat(new Array(L - s.length).fill(0)), y: y[i], text: texts[i] }));
    const nTrain = Math.round(ex.length * (o.trainFrac || 0.75));
    const batchOf = (list) => ({ x: S.T.tensor(list.map((e) => e.ids)), y: list.map((e) => e.y) });
    const val = DL.split(ex.slice(nTrain), batchOf);
    return { kind: 'encoder', train: DL.split(ex.slice(0, nTrain), batchOf), val, probe: val.get([0]), tokenizer: tok, classes,
      model: { vocab: tok.size, d: 32, heads: 2, layers: 1, maxLen: L, classes: classes.length, tokenizer: tok.toJSON() }, fit: { batch: 16, lr: 3e-3 } };
  };
  DL.tasks.register('topics', (o) => { const d = S.Data.load('topics', { seed: o.seed || 1 }); return wordTask(d.texts, d.y, d.classes, o); });

  /** which of the tokens 1..k occurs most often in a sequence (no ties) */
  DL.tasks.register('majority', (o) => {
    const len = o.len || 9, k = o.k || 3, g = S.T.rng(o.seed || 1);
    const make = (n) => Array.from({ length: n }, () => {
      for (;;) {
        const ids = Array.from({ length: len }, () => 1 + g.int(k)), cnt = new Array(k + 1).fill(0);
        ids.forEach((t) => cnt[t]++);
        const best = Math.max(...cnt), arg = cnt.indexOf(best);
        if (cnt.filter((v) => v === best).length === 1) return { ids, y: arg - 1 };
      }
    });
    const batchOf = (list) => ({ x: S.T.tensor(list.map((e) => e.ids)), y: list.map((e) => e.y) });
    const val = DL.split(make(o.val || 200), batchOf);
    return { kind: 'encoder', train: DL.split(make(o.n || 1000), batchOf), val, probe: val.get([0]), classes: Array.from({ length: k }, (_, i) => 'token ' + (i + 1)),
      model: { vocab: k + 1, d: 24, heads: 2, layers: 1, maxLen: len, classes: k }, fit: { batch: 32, lr: 3e-3 } };
  });
})(globalThis.Nooral = globalThis.Nooral || {});
