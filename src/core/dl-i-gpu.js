// Training a GPT on the GPU. The same model (its parameters are copied once to the GPU) runs its forward pass, the
// backward pass and AdamW there, with the graph recorded by S.T.gpu exactly as the CPU records it; the CPU model is
// the reference, and the copies come back only when asked (a snapshot, a checkpoint, an evaluation on the CPU).
//   s = await DL.gpuSession(net, {decoupled})   (throws where WebGPU or the model is not supported)
//     await s.step(batch, {lr, t, betas, eps, weightDecay, clip}) -> {loss, correct, count, gradNorm}
//     await s.evaluate(batch) -> {loss, correct, count}     s.outputs: the named GPU tensors of the last forward
//     await s.read(name) a named output (embed, block0, block0.attn, ..., logits) as a CPU tensor
//     await s.grads() -> {path: CPU tensor} the last gradients    await s.sync({state}) copy back into net (and opt)
//     s.upload() copy the CPU parameters up again (after loadState)
//   trainer.useGPU() / await trainer.stepAsync() / await trainer.evaluateAsync(split, max) / await trainer.syncAsync()
//   await net.trainStepAsync(batch, {lr, weightDecay, clip}) one AdamW step of a GPT on the GPU (in dl-a-base.js)
//   DL.gpuSupports(net, trainer) -> '' when the GPU path can train it, else the reason
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});

  DL.gpuSupports = (net, trainer) => {
    if (!net || net.kind !== 'gpt') return 'only the GPT trains on the GPU';
    if (net.config && net.config.dropout) return 'dropout runs on the CPU only';
    if (trainer && !/^adamw?$/i.test(trainer.o.optimizer)) return 'the GPU path has Adam and AdamW';
    return '';
  };

  /** the GPT forward pass on GPU tensors, the same steps as the GPT's forward and DL.blockForward */
  function forward(net, P, ids, taps) {
    const G = S.T.gpu, B = ids.length, n = ids[0].length, d = net.config.d || 64, H = net.config.heads || 2, dh = d / H, rows = B * n;
    if (n > net.context) throw new Error('the context holds ' + net.context + ' tokens; got ' + n);
    const lin = (x, name) => (P[name + '.b'] ? G.add(G.matmul(x, P[name + '.W']), P[name + '.b']) : G.matmul(x, P[name + '.W']));
    const heads = (x) => G.reshape(G.permute(G.reshape(x, [B, n, H, dh]), [0, 2, 1, 3]), [B * H, n, dh]);
    let h = G.add(G.embedding(P['emb.W'], ids.flat(), [rows]), P['pos.P'], n * d);
    taps.embed = h;
    net.blocks.forEach((blk, i) => {
      const p = 'blocks.' + i + '.';
      const a = G.layerNorm(h, P[p + 'ln1.gamma'], P[p + 'ln1.beta'], blk.ln1.eps);
      const q = heads(lin(a, p + 'attn.Wq')), k = heads(lin(a, p + 'attn.Wk')), v = heads(lin(a, p + 'attn.Wv'));
      const w = G.softmaxRows(G.matmul(q, k, { tb: 1 }), { causal: true, scale: 1 / Math.sqrt(dh) });
      const att = G.reshape(G.permute(G.reshape(G.matmul(w, v), [B, H, n, dh]), [0, 2, 1, 3]), [rows, d]);
      h = G.add(h, lin(att, p + 'attn.Wo'));
      const f = lin(G.gelu(lin(G.layerNorm(h, P[p + 'ln2.gamma'], P[p + 'ln2.beta'], blk.ln2.eps), p + 'ff1')), p + 'ff2');
      h = G.add(h, f);
      taps['block' + i] = h; taps['block' + i + '.attn'] = G.reshape(w, [B, H, n, n]);
    });
    h = G.layerNorm(h, P['lnf.gamma'], P['lnf.beta'], net.lnf.eps);
    const logits = P['head.W'] ? G.matmul(h, P['head.W']) : G.matmul(h, P['emb.W'], { tb: 1 });
    taps.logits = logits;
    return logits;
  }

  DL.gpuSession = async (net, o = {}) => {
    const G = S.T.gpu, why = DL.gpuSupports(net);
    if (why) throw new Error(why);
    if (!G || !(await G.init())) throw new Error('WebGPU is not available here' + (G && G.why ? ': ' + G.why : ''));
    const named = net.namedParameters(), P = {}, list = [];
    for (const [name, t] of named) { const g = G.param(t); g.name = name; P[name] = g; list.push(g); }
    const adam = new G.Adam(list, { decoupled: o.decoupled !== false });
    const shapeTaps = (taps, B, n) => {
      const d = net.config.d || 64, out = {};
      for (const [k, t] of Object.entries(taps)) out[k] = k === 'logits' ? G.reshape(t, [B, n, t.shape[1]]) : /^block\d+$|^embed$/.test(k) ? G.reshape(t, [B, n, d]) : t;
      return out;
    };
    const s = {
      net, P, params: list, adam, outputs: {}, steps: 0,
      async step(batch, so = {}) {
        G.endStep();
        for (const p of list) p.grad = null;
        const ids = batch.x.toArray(), B = ids.length, n = ids[0].length, taps = {};
        const logits = forward(net, P, ids, taps), loss = G.crossEntropy(logits, batch.y.flat(Infinity));
        G.backward(loss);
        const norm = adam.step({ lr: so.lr, t: so.t, betas: so.betas, eps: so.eps, weightDecay: so.weightDecay, clip: so.clip });
        s.outputs = shapeTaps(taps, B, n); s.loss = loss; s.steps++;
        const [l, c, g] = await G.readMany([loss, loss.correct, norm]);
        return { loss: l.data[0], correct: c.data[0], count: loss.count, gradNorm: g.data[0] };
      },
      async evaluate(batch) {
        G.endStep();
        return G.noGrad(async () => {
          const ids = batch.x.toArray(), taps = {}, logits = forward(net, P, ids, taps), loss = G.crossEntropy(logits, batch.y.flat(Infinity));
          s.outputs = shapeTaps(taps, ids.length, ids[0].length);
          const [l, c] = await G.readMany([loss, loss.correct]);
          return { loss: l.data[0], correct: c.data[0], count: loss.count };
        });
      },
      async read(name) { const t = s.outputs[name]; if (!t) throw new Error('no GPU output called ' + name + '; there are ' + Object.keys(s.outputs).join(', ')); return G.read(t); },
      async grads() { const have = list.filter((p) => p.grad), r = await G.readMany(have.map((p) => p.grad)); return Object.fromEntries(have.map((p, i) => [p.name, r[i]])); },
      /** copy the parameters (and with state, an optimiser's moments) back to the CPU model */
      async sync(so = {}) {
        const r = await G.readMany(list);
        list.forEach((p, i) => p.cpu.data.set(r[i].data));
        if (so.state && so.opt) {
          const m = await G.readMany(list.map((p, i) => new G.GT(adam.m[i], p.shape))), v = await G.readMany(list.map((p, i) => new G.GT(adam.v[i], p.shape)));
          list.forEach((p, i) => { const st = so.opt.st(p.cpu); st.m = new p.cpu.data.constructor(m[i].data); st.v = new p.cpu.data.constructor(v[i].data); st.t = so.opt.t; });
        }
      },
      upload() { for (const p of list) G.write(p, p.cpu); },
      free() { G.endStep(); adam.free(); for (const p of list) p.buf.destroy(); },
    };
    return s;
  };

  const Trainer = DL.Trainer;
  /** move this trainer's model to the GPU (the optimiser's moments go along) */
  Trainer.prototype.useGPU = async function () {
    if (this.gpu) return this.gpu;
    const why = DL.gpuSupports(this.net, this);
    if (why) throw new Error(why);
    const s = await DL.gpuSession(this.net, { decoupled: !!this.opt.opts.decoupled });
    s.params.forEach((p, i) => { const st = this.opt.state.get(p.cpu); if (st && st.m) s.adam.setState(i, st.m, st.v); });
    this.gpu = s; this.backend = 'webgpu';
    return s;
  };
  Trainer.prototype.stepAsync = async function () {
    const t0 = performance.now(), net = this.net, opt = this.opt, o = this.o;
    if (!this.gpu) await this.useGPU();
    const batch = this.task.train.get(this.nextIndices(), this.rng, o.augment);
    net.train();
    opt.t++;
    if (opt.schedule) opt.lr = opt.schedule(opt.t - 1);
    const r = await this.gpu.step(batch, { lr: opt.lr, t: opt.t, betas: opt.opts.betas, eps: opt.opts.eps, weightDecay: opt.opts.weightDecay, clip: o.clip });
    this.stepNo++; this.stale = true;
    const loss = r.loss, metrics = { accuracy: r.count ? r.correct / r.count : undefined };
    if (net.isLM) metrics.perplexity = Math.exp(loss);
    if (o.clip) metrics.gradNorm = r.gradNorm;
    this.history.push({ step: this.stepNo, loss, accuracy: metrics.accuracy });
    if (o.evalEvery && this.stepNo % o.evalEvery === 0 && this.task.val) {
      const v = await this.evaluateAsync('val', o.evalMax);
      Object.assign(metrics, { valLoss: v.loss, valAccuracy: v.accuracy, valPerplexity: v.perplexity });
    }
    if (o.snapshotEvery && this.stepNo % o.snapshotEvery === 0) { await this.syncAsync(); this.snapshot(o.snapshot); }
    if (o.checkpointEvery && this.stepNo % o.checkpointEvery === 0) { await this.syncAsync(); this.checkpoint(); }
    this.ms += performance.now() - t0;
    return { loss, metrics, lr: opt.lr, backend: 'webgpu' };
  };
  /** the same numbers as evaluate(), computed on the GPU */
  Trainer.prototype.evaluateAsync = async function (split = 'val', max = 512) {
    if (!this.gpu) return this.evaluate(split, max);
    const s = this.task[split], net = this.net;
    if (!s || !s.n) return null;
    const n = Math.min(max, s.n), bs = Math.max(1, Math.min(this.o.batch * 4, n));
    let lossSum = 0, w = 0, correct = 0, count = 0;
    for (let a = 0; a < n; a += bs) {
      const idx = Array.from({ length: Math.min(bs, n - a) }, (_, i) => a + i), r = await this.gpu.evaluate(s.get(idx, S.T.rng(7))), k = r.count || idx.length;
      lossSum += r.loss * k; w += k; correct += r.correct; count += r.count;
    }
    const loss = lossSum / w, out = { step: this.stepNo, split, loss, accuracy: count ? correct / count : undefined };
    if (net.isLM) out.perplexity = Math.exp(loss);
    out.say = 'After ' + this.stepNo + ' steps the ' + split + ' loss is ' + S.T.fmt(loss) + (out.accuracy !== undefined ? ' and accuracy ' + S.T.fmt(out.accuracy) : '') + (out.perplexity ? ', perplexity ' + S.T.fmt(out.perplexity) : '') + '.';
    if (split === 'val') this.valHistory.push(out);
    return out;
  };
  /** bring the GPU copies back into the CPU model (and the optimiser's moments with state) */
  Trainer.prototype.syncAsync = async function (so = {}) {
    if (!this.gpu || (!this.stale && !so.state)) return;
    await this.gpu.sync({ state: so.state, opt: this.opt });
    this.stale = false;
  };
  const restore = Trainer.prototype.restore;
  Trainer.prototype.restore = function (c) { restore.call(this, c); if (this.gpu) this.gpu.upload(); };
  /** fit() with GPU steps: until steps or ms run out, or until(metrics, trainer) */
  Trainer.prototype.fitAsync = async function (o = {}) {
    const t0 = performance.now(), steps = o.steps || Infinity, ms = o.ms || Infinity;
    let r = null;
    for (let i = 0; i < steps && performance.now() - t0 < ms; i++) {
      r = this.gpu ? await this.stepAsync() : this.step();
      if (o.onProgress) o.onProgress(r, this);
      if (o.until && o.until(r, this)) break;
    }
    return r;
  };

})(globalThis.Nooral = globalThis.Nooral || {});
