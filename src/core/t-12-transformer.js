// Multi-head attention and the pre-LN transformer block.
//   new nn.MultiHeadAttention(d, heads, {causal, dropout, rng})   forward(x [B, T, d], {mask, context}) -> [B, T, d]
//      after a forward, layer.weights is the attention [B, heads, T, T] and layer.scores the scaled scores
//   new nn.TransformerBlock(d, heads, {ff = 4d, causal, dropout, act: 'gelu', rng})
//      x + attn(ln1(x)), then x + ff(ln2(x))
//   new nn.PositionalEmbedding(maxLen, d, {rng})   forward(x [B, T, d]) adds a learned vector per position
//   nn.sinusoid(maxLen, d) -> [maxLen, d] fixed sinusoidal position codes
(function (S) {
  'use strict';
  const T = S.T, nn = T.nn;
  const rngOf = (o) => T.asRng(o && o.rng !== undefined ? o.rng : 1);

  nn.MultiHeadAttention = class MultiHeadAttention extends nn.Module {
    constructor(d, heads, o = {}) {
      super();
      if (d % heads) throw new Error('model width ' + d + ' does not split into ' + heads + ' heads');
      const g = rngOf(o);
      this.d = d; this.heads = heads; this.causal = !!o.causal; this.p = o.dropout || 0; this.rng = g.fork();
      this.Wq = new nn.Dense(d, d, { rng: g }); this.Wk = new nn.Dense(d, d, { rng: g });
      this.Wv = new nn.Dense(d, d, { rng: g }); this.Wo = new nn.Dense(d, d, { rng: g });
    }
    split(t, B, n) { return T.permute(T.reshape(t, [B, n, this.heads, this.d / this.heads]), [0, 2, 1, 3]); }
    forward(x, opts = {}) {
      const [B, n] = x.shape, ctx = opts.context || x, m = ctx.shape[1];
      const q = this.split(this.Wq.forward(x), B, n), k = this.split(this.Wk.forward(ctx), B, m), v = this.split(this.Wv.forward(ctx), B, m);
      const a = T.attention(q, k, v, { mask: opts.mask, causal: opts.causal === undefined ? this.causal : opts.causal, dropout: this.p, rng: this.rng, training: this.training });
      this.weights = a.attn.weights; this.scores = a.attn.scores; this.q = q; this.k = k; this.v = v;
      const merged = T.reshape(T.permute(a, [0, 2, 1, 3]), [B, n, this.d]);
      return this.Wo.forward(merged);
    }
  };
  nn.TransformerBlock = class TransformerBlock extends nn.Module {
    constructor(d, heads, o = {}) {
      super();
      const g = rngOf(o), ff = o.ff || 4 * d;
      this.ln1 = new nn.LayerNorm(d); this.ln2 = new nn.LayerNorm(d);
      this.attn = new nn.MultiHeadAttention(d, heads, { causal: o.causal, dropout: o.dropout, rng: g });
      this.ff1 = new nn.Dense(d, ff, { rng: g }); this.ff2 = new nn.Dense(ff, d, { rng: g });
      this.act = o.act || 'gelu'; this.p = o.dropout || 0; this.rng = g.fork();
    }
    forward(x, opts = {}) {
      let a = this.attn.forward(this.ln1.forward(x), opts);
      if (this.p) a = T.dropout(a, this.p, this.rng, this.training);
      x = T.add(x, a);
      let f = this.ff2.forward(T[this.act](this.ff1.forward(this.ln2.forward(x))));
      if (this.p) f = T.dropout(f, this.p, this.rng, this.training);
      return T.add(x, f);
    }
  };
  nn.PositionalEmbedding = class PositionalEmbedding extends nn.Module {
    constructor(maxLen, d, o = {}) { super(); this.P = this.param('P', T.normal([maxLen, d], 0, o.std || 0.1, rngOf(o))); }
    forward(x) { return T.add(x, T.slice(this.P, [0, x.shape[1]], null)); }
  };
  nn.sinusoid = (maxLen, d) => {
    const t = T.zeros([maxLen, d]);
    for (let p = 0; p < maxLen; p++) for (let i = 0; i < d; i++) {
      const w = p / Math.pow(10000, (2 * Math.floor(i / 2)) / d);
      t.data[p * d + i] = i % 2 ? Math.cos(w) : Math.sin(w);
    }
    return t;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
