// The text lab's engines (S.TL), free of the page so the same file runs in the lab's Web Worker and in node tests:
//   TL.corpus() -> the original stories (S.Data 'stories'), the training text
//   TL.bpeTrain(text, n) -> {base, start, steps: [{pair, token, count, tokens, vocab, top: [[a, b, count]..]}]}
//      byte-pair merges one by one, each with the most frequent pairs it was chosen from (ties: the pair that sorts
//      first, as S.DL.bpe.train does, so both give the same merges)
//   TL.tokenizer(base, merges) -> {pieces(text)}            the text cut with the first merges
//   TL.cut(probs, {temperature, topK, topP, greedy}) -> {probs, kept, order}   what a sampling rule leaves of a
//      next-token distribution: temperature first, then top-k, then top-p, renormalised
//   TL.modelSize(name) -> the settings of a size: 'webgpu' and 'cpu' are S.DL.gptPresets, 'small' a quick one
//   TL.Engine: S.DL's GPT on its 'text' task with its trainer (on the graphics card where WebGPU starts), trained on
//      characters: init(cfg), step() -> loss, await useGPU(), await stepAsync(), evaluate(), next(ids) -> probs,
//      generate(ids, n, rule, seed), attention(ids, layer) -> [heads][T][T]
//   TL.Bigram: the counts of character pairs, a model to sample from before the transformer has learned anything
//   TL.serve(self): the worker side of TL.Engine
(function (S) {
  'use strict';
  const TL = (S.TL = S.TL || {});

  TL.corpus = function () {
    try { const d = S.Data.load('stories'); return d.text || (d.texts || []).join('\n\n'); } catch (e) { return 'The lab needs the stories dataset of S.Data.'; }
  };
  const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+/g;
  TL.pretokenize = (text) => (S.DL && S.DL.bpe && S.DL.bpe.pretokenize ? S.DL.bpe.pretokenize(text) : text.match(PRE) || []);

  TL.bpeTrain = function (text, n, topN) {
    topN = topN || 8;
    const freq = new Map();
    for (const ch of TL.pretokenize(text)) freq.set(ch, (freq.get(ch) || 0) + 1);
    const base = [...new Set(text)].sort();
    const words = [...freq.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([w, c]) => ({ sym: Array.from(w), n: c }));
    let tokens = words.reduce((s, w) => s + w.n * w.sym.length, 0);
    const start = tokens, steps = [];
    for (let m = 0; m < n; m++) {
      const pc = new Map();
      for (const w of words) for (let i = 0; i + 1 < w.sym.length; i++) { const k = w.sym[i] + '\u0000' + w.sym[i + 1]; pc.set(k, (pc.get(k) || 0) + w.n); }
      const ranked = [...pc.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
      if (!ranked.length || ranked[0][1] < 2) break;
      const [bestK, bestN] = ranked[0], [a, b] = bestK.split('\u0000'), tok = a + b;
      let saved = 0;
      for (const w of words) {
        if (w.sym.length < 2) continue;
        const out = [];
        for (let i = 0; i < w.sym.length; i++) { if (i + 1 < w.sym.length && w.sym[i] === a && w.sym[i + 1] === b) { out.push(tok); i++; saved += w.n; } else out.push(w.sym[i]); }
        w.sym = out;
      }
      tokens -= saved;
      steps.push({ pair: [a, b], token: tok, count: bestN, tokens, vocab: base.length + m + 1, top: ranked.slice(0, topN).map(([k, c]) => k.split('\u0000').concat([c])) });
    }
    return { base, start, steps, words: words.length };
  };
  /** a text cut into pieces with the first k merges, lowest-ranked merge first in each chunk */
  TL.tokenizer = function (base, merges) {
    const rank = new Map(merges.map(([a, b], i) => [a + '\u0000' + b, i]));
    const split = (chunk) => {
      let sym = Array.from(chunk);
      for (;;) {
        let best = Infinity, at = -1;
        for (let i = 0; i + 1 < sym.length; i++) { const r = rank.get(sym[i] + '\u0000' + sym[i + 1]); if (r !== undefined && r < best) { best = r; at = i; } }
        if (at < 0) return sym;
        const [a, b] = merges[best], out = [];
        for (let i = 0; i < sym.length; i++) { if (i + 1 < sym.length && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i++; } else out.push(sym[i]); }
        sym = out;
      }
    };
    return { pieces: (text) => TL.pretokenize(text).flatMap(split) };
  };

  // ---- sampling rules ---------------------------------------------------------------------------------------------
  TL.cut = function (probs, o) {
    o = o || {};
    const V = probs.length, order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b);
    let p;
    if (o.greedy) { p = new Array(V).fill(0); p[order[0]] = 1; return { probs: p, kept: p.map((x) => x > 0), order }; }
    const t = Math.max(1e-3, o.temperature === undefined ? 1 : o.temperature);
    // a temperature divides the logits: p^(1/t), renormalised
    const lg = probs.map((x) => Math.log(Math.max(x, 1e-30)) / t), m = Math.max(...lg);
    p = lg.map((z) => Math.exp(z - m));
    let s = p.reduce((a, b) => a + b, 0);
    p = p.map((x) => x / s);
    const ord = Array.from({ length: V }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
    const kept = new Array(V).fill(true);
    if (o.topK && o.topK > 0 && o.topK < V) ord.slice(o.topK).forEach((i) => { kept[i] = false; });
    if (o.topP !== undefined && o.topP < 1) {
      // the smallest set of the most likely tokens whose probabilities (after top-k) add to at least p
      const live = ord.filter((i) => kept[i]), tot = live.reduce((a, i) => a + p[i], 0);
      let acc = 0, done = false;
      for (const i of live) { if (done) kept[i] = false; acc += p[i] / tot; if (acc >= o.topP - 1e-12) done = true; }
    }
    s = p.reduce((a, x, i) => a + (kept[i] ? x : 0), 0);
    p = p.map((x, i) => (kept[i] ? x / s : 0));
    return { probs: p, kept, order: ord };
  };
  TL.pick = function (probs, r) { let u = r(), acc = 0; for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u < acc) return i; } let last = probs.length - 1; while (last > 0 && !probs[last]) last--; return last; };

  // ---- the character vocabulary and a bigram model ------------------------------------------------------------
  TL.chars = (text) => {
    const vocab = [...new Set(text)].sort(), index = new Map(vocab.map((c, i) => [c, i]));
    return { vocab, size: vocab.length, encode: (s) => Array.from(s, (c) => (index.has(c) ? index.get(c) : index.get(' ') || 0)), decode: (ids) => ids.map((i) => vocab[i] || '').join('') };
  };
  TL.Bigram = function (text) {
    const tok = TL.chars(text), V = tok.size, ids = tok.encode(text);
    const N = Array.from({ length: V }, () => new Float64Array(V).fill(0.1));
    for (let i = 1; i < ids.length; i++) N[ids[i - 1]][ids[i]]++;
    return { tok, next(ctx) { const row = N[ctx.length ? ctx[ctx.length - 1] : 0], s = row.reduce((a, b) => a + b, 0); return Array.from(row, (x) => x / s); } };
  };

  // ---- the transformer ----------------------------------------------------------------------------------------------
  // The sizes are S.DL.gptPresets, measured on the long corpus: 'webgpu' (about 1.8M parameters, about a minute on a
  // laptop's graphics card) and 'cpu' (about five minutes of JavaScript), both to a validation loss near 1.4–1.5 nats
  // a character. 'auto' takes the WebGPU size where WebGPU starts in the worker, else the CPU size (S.DL.pickPreset,
  // the rule S.DL.train uses for taskOpts.preset 'auto'). 'small' is a quick model for trying things out.
  TL.SIZE_NAMES = ['auto', 'webgpu', 'cpu', 'small'];
  /** the settings of one named size: {name, d, layers, heads, ctx, batch, lr, warmup, weightDecay, steps, seed} */
  TL.modelSize = function (name) {
    const P = S.DL && S.DL.gptPresets && S.DL.gptPresets[name];
    if (P) return { name, d: P.model.d, layers: P.model.layers, heads: P.model.heads, ctx: P.context, batch: P.fit.batch, lr: P.fit.lr, warmup: P.fit.warmup || 0, weightDecay: P.fit.weightDecay || 0, steps: P.fit.steps || 2000, seed: 1 };
    return { name: 'small', d: 32, layers: 2, heads: 4, ctx: 32, batch: 16, lr: 0.008, warmup: 30, weightDecay: 0.01, steps: 1500, seed: 1 };
  };
  TL.DEFAULTS = { size: 'auto', seed: 1 };
  /** the text it learns from: the long corpus the presets were measured on, else the stories */
  TL.trainText = () => { try { if (S.Data && S.Data.corpusText) return S.Data.corpusText(); } catch (e) { /* fall back */ } return TL.corpus(); };

  /**
   * The engine: S.DL's GPT, its text task and its trainer (S.DL.setup), so the lab trains exactly what the presets
   * describe. init(cfg, text) builds it on the CPU (cfg: {size} or explicit {d, layers, heads, ctx, batch, lr, ...});
   * await useGPU() moves training to the graphics card; then await stepAsync() / evaluateAsync() / sync().
   */
  class Engine {
    init(cfg, text) {
      cfg = Object.assign({}, cfg || {});
      // explicit sizes (no size name) make a custom model; 'auto' outside the worker's choice is the CPU size
      const named = cfg.d === undefined ? TL.modelSize(!cfg.size || cfg.size === 'auto' ? 'cpu' : cfg.size) : { name: 'custom' };
      this.cfg = cfg = Object.assign({ heads: 2, batch: 16, lr: 3e-3, warmup: 50, weightDecay: 0.01, steps: 2000, seed: 1 }, named, cfg, { name: named.name });
      this.text = text || TL.trainText();
      const pre = S.DL.gptPresets[cfg.name] ? cfg.name : null;
      const built = S.DL.setup({
        task: 'text', seed: cfg.seed,
        taskOpts: { text: this.text, preset: pre, context: cfg.ctx },
        config: { d: cfg.d, layers: cfg.layers, heads: cfg.heads },
        train: { batch: cfg.batch, lr: cfg.lr, warmup: cfg.warmup, weightDecay: cfg.weightDecay, steps: cfg.steps, evalMax: 32 },
      });
      this.model = built.net; this.trainer = built.trainer; this.task = built.task;
      this.tok = built.task.tokenizer;
      this.backend = 'cpu'; this.why = '';
      this.steps = 0;
      this.history = [];
      return this.info();
    }
    info() {
      const c = this.cfg;
      return { vocab: this.tok.vocab, params: this.model.numParams(), chars: this.text.length, size: c.name, backend: this.backend, why: this.why || null,
        cfg: { d: c.d, layers: c.layers, heads: c.heads, ctx: c.ctx, batch: c.batch, steps: c.steps } };
    }
    /** train on the graphics card from now on; false (with why) where it does not start */
    async useGPU() {
      try { await this.trainer.useGPU(); this.backend = 'webgpu'; this.trainer.o.evalMax = 128; return true; } catch (e) { this.why = String(e.message || e); return false; }
    }
    record(r) { this.steps = this.trainer.stepNo; this.history.push({ step: this.steps, loss: r.loss }); return r.loss; }
    step() { return this.record(this.trainer.step()); }
    async stepAsync() { return this.record(this.backend === 'webgpu' ? await this.trainer.stepAsync() : this.trainer.step()); }
    /** the validation loss on the first windows of the held-out tenth (the same windows every time) */
    evaluate() { return this.trainer.evaluate('val', this.trainer.o.evalMax).loss; }
    async evaluateAsync() { return (this.backend === 'webgpu' ? await this.trainer.evaluateAsync('val', this.trainer.o.evalMax) : this.trainer.evaluate('val', this.trainer.o.evalMax)).loss; }
    /** bring the graphics card's weights back to the CPU copy the questions below read */
    async sync() { if (this.backend === 'webgpu') await this.trainer.syncAsync(); }
    softmax(row) { const m = Math.max(...row), e = row.map((z) => Math.exp(z - m)), s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); }
    /** the probabilities of the next character after ids (the last ctx of them) */
    next(ids) {
      const T = S.T, V = this.tok.size, w = ids.slice(-this.cfg.ctx);
      if (!w.length) w.push(this.tok.encode(' ')[0]);
      return T.noGrad(() => { const v = this.model.forward(T.tensor([w])).values(), off = (w.length - 1) * V; return this.softmax(Array.from(v.slice(off, off + V))); });
    }
    /** n more characters under a sampling rule, one at a time, with the model's key-value cache */
    generate(ids, n, rule, seed) {
      const T = S.T, net = this.model, r = S.ML.rng(seed || 1), out = ids.slice(), ctx = this.cfg.ctx, V = this.tok.size;
      if (!out.length) out.push(this.tok.encode(' ')[0]);
      const was = net.training;
      net.eval();
      T.noGrad(() => {
        let cache = null, z = null;
        const prefill = () => { cache = net.newCache(); z = net.forward(T.tensor([out.slice(-(ctx - 1))]), { cache }); };
        prefill();
        for (let k = 0; k < n; k++) {
          const v = z.values(), off = (z.shape[1] - 1) * V;
          out.push(TL.pick(TL.cut(this.softmax(Array.from(v.slice(off, off + V))), rule).probs, r));
          if (k + 1 === n) break;
          if (cache.pos >= ctx) prefill(); else z = net.forward(T.tensor([[out[out.length - 1]]]), { cache });
        }
      });
      net.setMode(was);
      return out.slice(ids.length || 1);
    }
    /** the attention of every head of one layer on ids: [heads][T][T] */
    attention(ids, layer) {
      const T = S.T, w = ids.slice(-this.cfg.ctx);
      T.noGrad(() => this.model.forward(T.tensor([w])));
      const a = this.model.blocks[Math.min(layer, this.model.blocks.length - 1)].attn.weights, [, H, n] = a.shape, v = a.values();
      return Array.from({ length: H }, (_, hh) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => v[(hh * n + i) * n + j])));
    }
  }
  TL.Engine = Engine;

  /**
   * the worker side: {cmd: 'init', cfg} | {cmd: 'run', every} | {cmd: 'pause'} | {cmd: 'steps', n} | {cmd: 'ask', id,
   * what: 'next' | 'generate' | 'attention', args}. It posts {type: 'ready', info} | {type: 'progress', step, loss, val}
   * | {type: 'sample', step, text} | {type: 'state', running} | {type: 'answer', id, value} | {type: 'error', message}.
   * Everything runs one after another (a chain of promises), so a question never lands in the middle of a GPU step.
   */
  TL.serve = function (self) {
    let E = null, running = false, budget = 0, timer = null, every = 150, prompt = 'The ', gen = 0, chain = Promise.resolve();
    const post = (m) => self.postMessage(m);
    const serial = (fn) => (chain = chain.then(fn, fn));
    const report = async () => {
      const l = E.history[E.history.length - 1].loss;
      if (E.steps % 10 === 0 || E.steps < 10) post({ type: 'progress', step: E.steps, loss: l, val: E.steps % 50 === 0 ? await E.evaluateAsync() : undefined });
      if (E.steps % every === 0) { await E.sync(); post({ type: 'sample', step: E.steps, text: E.tok.decode(E.generate(E.tok.encode(prompt), 120, { temperature: 0.8 }, E.steps)), prompt }); }
    };
    const loop = (g) => serial(async () => {
      timer = null;
      if (g !== gen || !E || (!running && budget <= 0)) return;
      const t0 = Date.now();
      try {
        while (Date.now() - t0 < 60 && (running || budget > 0)) { await E.stepAsync(); if (budget > 0) budget--; await report(); }
      } catch (e) { running = false; budget = 0; post({ type: 'error', message: String(e.message || e) }); return; }
      if (!running && budget <= 0) { post({ type: 'state', running: false, step: E.steps }); return; }
      kick();
    });
    const kick = () => { if (!timer) { const g = gen; timer = setTimeout(() => loop(g), 0); } };
    const init = (m) => serial(async () => {
      const cfg = Object.assign({}, TL.DEFAULTS, m.cfg || {});
      let gpu = false, why = '';
      if (cfg.size === 'auto' || cfg.size === 'webgpu') {
        const pick = await S.DL.pickPreset('auto');
        gpu = pick.gpu; why = pick.why;
        if (cfg.size === 'auto') cfg.size = pick.preset;
      }
      E = new Engine();
      E.init(cfg, m.text);
      if (gpu && !(await E.useGPU())) why = E.why;
      E.why = why;
      post({ type: 'ready', info: Object.assign(E.info(), { auto: m.cfg && m.cfg.size === 'auto' }), val: await E.evaluateAsync() });
    });
    const answer = (m) => serial(async () => {
      try {
        if (!E) throw new Error('the model is not built yet');
        await E.sync();
        const a = m.args;
        post({ type: 'answer', id: m.id, value: m.what === 'next' ? E.next(a.ids) : m.what === 'generate' ? E.generate(a.ids, a.n, a.rule, a.seed) : m.what === 'attention' ? E.attention(a.ids, a.layer) : null });
      } catch (e) { post({ type: 'error', message: String(e.message || e), id: m.id }); }
    });
    self.onmessage = (ev) => {
      const m = ev.data;
      try {
        if (m.cmd === 'init') { running = false; budget = 0; gen++; clearTimeout(timer); timer = null; if (m.every) every = m.every; if (m.prompt) prompt = m.prompt; init(m).catch((e) => post({ type: 'error', message: String(e.message || e) })); }
        else if (m.cmd === 'run') { running = true; post({ type: 'state', running: true }); kick(); }
        else if (m.cmd === 'pause') { running = false; budget = 0; post({ type: 'state', running: false, step: E ? E.steps : 0 }); }
        else if (m.cmd === 'steps') { budget += m.n || 1; kick(); }
        else if (m.cmd === 'ask') answer(m);
      } catch (e) { post({ type: 'error', message: String(e.message || e), id: m.id }); }
    };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
