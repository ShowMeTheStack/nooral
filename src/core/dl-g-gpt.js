// A GPT: a decoder-only transformer with causal attention, learned positions and the output layer tied to the
// token embedding, trained to predict the next token of original text (characters or BPE tokens). The same model,
// made tiny, learns the toy sequence tasks (copy, reverse, sort).
//   DL.build('gpt', {vocab, d, heads, layers, context, ff = 4d, dropout, tie = true, initStd, seed, tokenizer})
//      forward(ids [B, T], {cache}) -> logits [B, T, V]; named outputs embed, block0, block0.attn, ..., logits
//   net.newCache() -> {layers: [{k, v}], pos}: keys and values kept per layer, so each new token costs one step
//   net.generate(prompt ids, n, sampling, rng, {cache = true}) -> {ids, steps, text}   net.complete(text, n, ...)
//   tasks: 'text' {tokenizer: 'char' | 'bpe', merges, context, stride, corpus, preset} on the prose and dialogue corpus;
//          DL.gptPresets {webgpu, cpu}: the text lab's sizes (preset 'auto' in DL.train picks by what the worker has);
//          'copy' / 'reverse' / 'sort' {L, digits, n}: digits, a separator, then the answer (only it is scored);
//          a toy task has check(net, n, seed) -> {exact, n, say}: exact matches by greedy decoding
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});
  const fmt = (x) => S.T.fmt(x);

  DL.register('gpt', (c) => {
    const T = S.T, nn = T.nn, Net = DL.makeNet(), g = T.rng(c.seed || 1), d = c.d || 64, L = c.layers || 2, ctx = c.context || 64;
    class GPT extends Net {
      constructor() {
        super();
        this.emb = new nn.Embedding(c.vocab, d, { rng: g, std: c.initStd || 0.05 });
        this.pos = new nn.PositionalEmbedding(ctx, d, { rng: g, std: 0.02 });
        this.blocks = Array.from({ length: L }, () => new nn.TransformerBlock(d, c.heads || 2, { rng: g, ff: c.ff || 4 * d, dropout: c.dropout || 0, causal: true }));
        this.lnf = new nn.LayerNorm(d);
        if (c.tie === false) this.head = new nn.Dense(d, c.vocab, { rng: g, bias: false });
        this.isLM = true;
        this.context = ctx;
        this.tokenizer = c.tokenizer ? DL.tokenizer(c.tokenizer) : null;
        this.snapshotNames = ['block0.attn', 'block' + (L - 1) + '.attn', 'embeddings'];
      }
      newCache() { return { layers: this.blocks.map(() => ({ k: null, v: null })), pos: 0 }; }
      forward(ids, o = {}) {
        this.begin();
        const n = ids.shape[1], start = o.cache ? o.cache.pos : 0;
        if (start + n > ctx) throw new Error('the context holds ' + ctx + ' tokens; got ' + (start + n));
        let h = this.tap('embed', T.add(this.emb.forward(ids), T.slice(this.pos.P, [start, start + n], null)));
        this.blocks.forEach((blk, i) => {
          h = this.tap('block' + i, DL.blockForward(blk, h, { causal: true, cache: o.cache && o.cache.layers[i] }));
          this.tap('block' + i + '.attn', blk.attn.weights);
        });
        if (o.cache) o.cache.pos += n;
        h = this.lnf.forward(h);
        return this.tap('logits', this.head ? this.head.forward(h) : T.matmul(h, T.transpose(this.emb.W)));
      }
      extras() { return { embeddings: this.emb.W }; }
      /** continue prompt by n tokens; with the cache only the newest token passes through the blocks */
      generate(prompt, n, sampling = {}, rng, o = {}) {
        rng = T.asRng(rng === undefined ? 1 : rng);
        const useCache = o.cache !== false, ids = prompt.slice(), steps = [], was = this.training;
        this.eval();
        T.noGrad(() => {
          let cache = null, z = null;
          const prefill = () => { cache = useCache ? this.newCache() : null; const w = ids.slice(-(ctx - 1)); z = this.forward(T.tensor([w]), { cache }); };
          prefill();
          for (let k = 0; k < n; k++) {
            const row = T.slice(z, 0, z.shape[1] - 1, null).values(), s = DL.sample(row, sampling, rng);
            if (this.tokenizer) s.token = this.tokenizer.vocab[s.id];
            ids.push(s.id); steps.push(s);
            if (k + 1 === n) break;
            if (!useCache || cache.pos >= ctx) prefill();
            else z = this.forward(T.tensor([[s.id]]), { cache });
          }
        });
        this.setMode(was);
        return { ids, steps, text: this.tokenizer ? this.tokenizer.decode(ids) : null };
      }
      complete(text, n, sampling, rng, o) { return this.generate(this.tokenizer.encode(text), n, sampling, rng, o); }
    }
    return new GPT();
  });

  /**
   * sizes of the GPT for the text lab, measured on the long corpus (characters): 'webgpu' trains in about a minute
   * and a quarter on a laptop GPU, 'cpu' in about five minutes of JavaScript; both reach a validation loss near
   * 1.4–1.5 nats per character, where nearly nine sampled words in ten are real words. Longer runs overfit.
   */
  DL.gptPresets = {
    webgpu: { model: { d: 192, heads: 6, layers: 4 }, context: 128, fit: { batch: 16, lr: 2e-3, warmup: 100, weightDecay: 0.1, steps: 1200 } },
    cpu: { model: { d: 96, heads: 4, layers: 3 }, context: 64, fit: { batch: 16, lr: 3e-3, warmup: 100, weightDecay: 0.1, steps: 2000 } },
  };
  /** next-token prediction on the corpus (the long one; corpus: 'short' for the first 17K characters): windows of
   *  context tokens, the last tenth of the text validates; preset 'webgpu' | 'cpu' picks a size from DL.gptPresets */
  DL.tasks.register('text', (o) => {
    const text = o.text || (o.corpus === 'short' ? S.Data.proseText() : S.Data.corpusText()), kind = o.tokenizer || 'char';
    const pre = DL.gptPresets[o.preset] || null;
    const tok = kind === 'bpe' ? DL.bpe.train(text, o.merges || 300).tokenizer : DL.charTokenizer(text);
    const ids = tok.encode(text), n = o.context || (pre ? pre.context : 64), stride = o.stride || Math.max(1, n >> 2), cut = Math.floor(ids.length * 0.9);
    const win = (a, b) => { const r = []; for (let i = a; i + n + 1 <= b; i += stride) r.push(i); return r; };
    const batchOf = (list) => ({ x: S.T.tensor(list.map((i) => ids.slice(i, i + n))), y: list.map((i) => ids.slice(i + 1, i + n + 1)) });
    const val = DL.split(win(cut, ids.length), batchOf);
    return { kind: 'gpt', train: DL.split(win(0, cut), batchOf), val, probe: val.get([0]), tokenizer: tok, ids,
      model: Object.assign({ vocab: tok.size, d: 192, heads: 6, layers: 4 }, pre && pre.model, { context: n, tokenizer: tok.toJSON() }),
      fit: Object.assign({ batch: 16, lr: 2e-3, optimizer: 'adamw', weightDecay: 0.01, clip: 1, schedule: 'cosine', warmup: 50, steps: 2000 }, pre && pre.fit, o.steps ? { steps: o.steps } : null), preset: o.preset || null };
  });

  /** toy sequence tasks: digits x, a separator, then f(x); inputs are the whole sequence but the last token */
  const toy = (fn, what) => (o) => {
    const L = o.L || 5, D = o.digits || 10, SEP = D, g = S.T.rng(o.seed || 1);
    const one = () => { const x = Array.from({ length: L }, () => g.int(D)); return x.concat([SEP], fn(x)); };
    const batchOf = (list) => ({ x: S.T.tensor(list.map((s) => s.slice(0, -1))), y: list.map((s) => s.slice(1).map((v, k) => (k >= L ? v : -1))) });
    const val = DL.split(Array.from({ length: o.val || 200 }, one), batchOf);
    const task = { kind: 'gpt', train: DL.split(Array.from({ length: o.n || 8000 }, one), batchOf), val, probe: val.get([0]),
      model: { vocab: D + 1, d: 32, heads: 2, layers: 2, ff: 64, context: 2 * L + 1, initStd: 0.3 },
      fit: { batch: 32, lr: 3e-3, optimizer: 'adamw', weightDecay: 0.01, clip: 1, schedule: 'cosine', warmup: 30, steps: o.steps || 600 } };
    task.check = (net, n = 200, seed = 99) => {
      const gt = S.T.rng(seed);
      let exact = 0;
      const examples = [];
      for (let i = 0; i < n; i++) {
        const x = Array.from({ length: L }, () => gt.int(D)), want = fn(x);
        const got = net.generate(x.concat([SEP]), L, { method: 'greedy' }).ids.slice(L + 1);
        if (got.join() === want.join()) exact++;
        if (i < 5) examples.push({ x, want, got });
      }
      return { exact, n, examples, say: 'The model ' + what + ' ' + exact + ' of ' + n + ' new sequences exactly (' + fmt((100 * exact) / n) + '%).' };
    };
    return task;
  };
  DL.tasks.register('copy', toy((x) => x.slice(), 'copies'));
  DL.tasks.register('reverse', toy((x) => x.slice().reverse(), 'reverses'));
  DL.tasks.register('sort', toy((x) => x.slice().sort((a, b) => a - b), 'sorts'));
})(globalThis.Nooral = globalThis.Nooral || {});
