// The training loop for every DL model: mini-batches from a task, an optimiser with an optional warm-up and cosine
// schedule, gradient clipping, training and validation loss, accuracy and perplexity, checkpoints, and snapshots
// of chosen internals (feature maps, filters, attention per head, gate activations, embeddings) every N steps.
//   A task gives {train: split, val: split, probe?, model: {config the model needs}, fit: {trainer options}, info}; a split is
//   {n, get(indices, rng, augment) -> batch}.   DL.tasks.register(name, (opts) => task)   DL.task(name, opts)
//   new DL.Trainer(net, task, {batch, lr, optimizer, weightDecay, schedule: 'cosine', warmup, steps, clip, seed,
//       evalEvery, evalMax, snapshotEvery, snapshot: [names], checkpointEvery, keepCheckpoints, augment,
//       fast (the float32 SIMD matrix kernel; default for models of 100K parameters and more)})
//   trainer.step() -> {loss, metrics, lr}   trainer.evaluate(split, max)   trainer.fit({steps, ms, until, onProgress})
//   trainer.history / valHistory / snapshots / checkpoints   trainer.snapshot(names)   trainer.tensors()
//   DL.setup({kind, config, task, taskOpts, train}) -> {net, task, trainer}    (plain parameters: works in a worker)
//   DL.train(params, opts, onMessage, job = 'dl')   the same in a Web Worker (S.T.train), streaming progress messages;
//     a GPT trains on WebGPU there when the worker has it (params.backend 'auto' | 'webgpu' | 'cpu'; see dl-i-gpu.js);
//   DL.jobs = {name: factory} are registered with S.T.jobs (DL.ensureJobs)
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});
  const fmt = (x) => S.T.fmt(x);

  const taskMakers = new Map();
  DL.tasks = { register: (name, make) => taskMakers.set(name, make), names: () => [...taskMakers.keys()] };
  DL.task = (name, opts = {}) => {
    const make = taskMakers.get(name);
    if (!make) throw new Error('no task called ' + name);
    const t = make(opts);
    t.name = name;
    return t;
  };
  /** a split over a list of examples; batchOf(list of examples, rng, augment) builds the tensors */
  DL.split = (examples, batchOf) => ({ n: examples.length, examples, get: (idx, rng, aug) => batchOf(idx.map((i) => examples[i]), rng, aug) });

  class Trainer {
    constructor(net, task, o = {}) {
      const T = S.T;
      this.net = net; this.task = task;
      this.o = Object.assign({ batch: 32, lr: 1e-3, optimizer: 'adam', weightDecay: 0, clip: 0, seed: 1, evalEvery: 0, evalMax: 512, snapshotEvery: 0, checkpointEvery: 0, keepCheckpoints: 3 }, o);
      this.rng = T.rng(this.o.seed);
      let schedule = typeof this.o.schedule === 'function' ? this.o.schedule : null;
      if (this.o.schedule === 'cosine') {
        const total = this.o.steps || 1000, warm = this.o.warmup || 0, lr = this.o.lr;
        const cos = T.sched.cosine({ lr, min: lr / 10, total: Math.max(1, total - warm) });
        schedule = warm ? T.sched.warmup({ steps: warm, then: cos }) : cos;
      }
      this.opt = T.optim.make(this.o.optimizer, net.parameters(), { lr: this.o.lr, weightDecay: this.o.weightDecay, schedule });
      this.stepNo = 0; this.epoch = 0; this.order = []; this.at = 0;
      this.history = []; this.valHistory = []; this.snapshots = []; this.checkpoints = [];
      this.ms = 0;
      this.probe = task.probe || (task.val && task.val.n ? task.val.get([0], T.rng(1)) : null);
      // big models multiply with the float32 SIMD kernel (S.T.fast); small ones keep the exact kernel
      this.fast = this.o.fast === undefined ? net.numParams() >= 100000 : !!this.o.fast;
    }
    /** run fn with this trainer's kernel choice */
    kernels(fn) { return this.fast ? S.T.fast(fn) : fn(); }
    nextIndices() {
      const n = this.task.train.n, bs = Math.min(this.o.batch, n);
      if (this.at + bs > this.order.length) { this.order = this.rng.permutation(n); this.at = 0; if (this.stepNo) this.epoch++; }
      const idx = this.order.slice(this.at, this.at + bs);
      this.at += bs;
      return idx;
    }
    step() { return this.kernels(() => this.stepNow()); }
    stepNow() {
      const T = S.T, t0 = performance.now(), net = this.net;
      const batch = this.task.train.get(this.nextIndices(), this.rng, this.o.augment);
      net.train();
      this.opt.zeroGrad();
      const r = net.lossOn(batch);
      r.loss.backward();
      const gnorm = this.o.clip ? T.clipGradNorm(net.parameters(), this.o.clip) : undefined;
      const { lr } = this.opt.step();
      this.stepNo++;
      const loss = r.loss.item(), metrics = { accuracy: r.count ? r.correct / r.count : undefined };
      if (net.isLM) metrics.perplexity = Math.exp(loss);
      if (gnorm !== undefined) metrics.gradNorm = gnorm;
      this.history.push({ step: this.stepNo, loss, accuracy: metrics.accuracy });
      const o = this.o;
      if (o.evalEvery && this.stepNo % o.evalEvery === 0 && this.task.val) {
        const v = this.evaluate('val', o.evalMax);
        Object.assign(metrics, { valLoss: v.loss, valAccuracy: v.accuracy, valPerplexity: v.perplexity });
      }
      if (o.snapshotEvery && this.stepNo % o.snapshotEvery === 0) this.snapshot(o.snapshot);
      if (o.checkpointEvery && this.stepNo % o.checkpointEvery === 0) this.checkpoint();
      this.ms += performance.now() - t0;
      return { loss, metrics, lr };
    }
    /** mean loss, accuracy (and perplexity for language models) over up to max examples of a split */
    evaluate(split = 'val', max = 512) { return this.kernels(() => this.evaluateNow(split, max)); }
    evaluateNow(split, max) {
      const T = S.T, s = this.task[split], net = this.net, was = net.training;
      if (!s || !s.n) return null;
      const n = Math.min(max, s.n), bs = Math.max(1, Math.min(this.o.batch * 4, n));
      let lossSum = 0, w = 0, correct = 0, count = 0;
      net.eval();
      T.noGrad(() => {
        for (let a = 0; a < n; a += bs) {
          const idx = Array.from({ length: Math.min(bs, n - a) }, (_, i) => a + i), b = s.get(idx, T.rng(7));
          const r = net.lossOn(b), k = r.count || idx.length;
          lossSum += r.loss.item() * k; w += k; correct += r.correct || 0; count += r.count || 0;
        }
      });
      net.setMode(was);
      const loss = lossSum / w, out = { step: this.stepNo, split, loss, accuracy: count ? correct / count : undefined };
      if (net.isLM) out.perplexity = Math.exp(loss);
      out.say = 'After ' + this.stepNo + ' steps the ' + split + ' loss is ' + fmt(loss) + (out.accuracy !== undefined ? ' and accuracy ' + fmt(out.accuracy) : '') + (out.perplexity ? ', perplexity ' + fmt(out.perplexity) : '') + '.';
      if (split === 'val') this.valHistory.push(out);
      return out;
    }
    /** every tensor a picture might want: parameters, the probe's named outputs, and the model's extras */
    tensors() {
      const net = this.net, all = Object.fromEntries(net.namedParameters());
      if (this.probe) { net.predict(this.probe.x); Object.assign(all, net.outputs); }
      if (net.extras) Object.assign(all, net.extras());
      return all;
    }
    snapshot(names) {
      const all = this.tensors(), pick = names && names.length ? names : this.net.snapshotNames || Object.keys(all), tensors = {};
      for (const n of pick) if (all[n]) tensors[n] = DL.pack(all[n]);
      const snap = { step: this.stepNo, tensors };
      this.snapshots.push(snap);
      return snap;
    }
    checkpoint() {
      const c = { step: this.stepNo, model: DL.save(this.net), rng: this.rng.state() };
      this.checkpoints.push(c);
      while (this.checkpoints.length > this.o.keepCheckpoints) this.checkpoints.shift();
      return c;
    }
    restore(c) { this.net.loadState((c.model || c).state); if (c.rng) this.rng.setState(c.rng); if (c.step) this.stepNo = c.step; }
    /** train until steps or ms run out, or until(metrics, trainer) is true; returns the last step's result */
    fit(o = {}) {
      const t0 = performance.now(), steps = o.steps || Infinity, ms = o.ms || Infinity;
      let r = null;
      for (let i = 0; i < steps && performance.now() - t0 < ms; i++) {
        r = this.step();
        if (o.onProgress) o.onProgress(r, this);
        if (o.until && o.until(r, this)) break;
      }
      return r;
    }
  }
  DL.Trainer = Trainer;

  /** build the net, the task and the trainer from plain parameters */
  DL.setup = (p = {}) => {
    const task = DL.task(p.task, Object.assign({ seed: p.seed || 1 }, p.taskOpts));
    const config = Object.assign({ seed: p.seed || 1 }, task.model, p.config);
    const net = DL.build(p.kind || task.kind, config);
    const trainer = new Trainer(net, task, Object.assign({ seed: p.seed || 1 }, task.fit, p.train));
    return { net, task, trainer };
  };
  /** training jobs for S.T.jobs (so a Web Worker can build them from plain parameters): name -> factory */
  DL.jobs = DL.jobs || {};
  // params.backend: 'auto' (the default: WebGPU where it works and the model can use it), 'webgpu' or 'cpu';
  // taskOpts.preset 'auto' (text): the GPU size where WebGPU starts, else the CPU size (decided at the first step)
  /** the text preset 'auto' resolves to: 'webgpu' where WebGPU starts (and backend is not 'cpu'), else 'cpu' */
  DL.pickPreset = async (backend = 'auto') => {
    const W = S.T.backends && S.T.backends.webgpu;
    if (backend === 'cpu') return { preset: 'cpu', gpu: false, why: 'the CPU was chosen' };
    if (!W || !W.supported()) return { preset: 'cpu', gpu: false, why: (W && W.why) || 'no WebGPU in this browser or context' };
    const ok = await S.T.gpu.init().catch(() => false);
    return { preset: ok ? 'webgpu' : 'cpu', gpu: !!ok, why: ok ? '' : S.T.gpu.why || 'WebGPU did not start' };
  };
  DL.jobs.dl = (p) => {
    const want = p.backend || 'auto', W = S.T.backends && S.T.backends.webgpu, maybe = want !== 'cpu' && W && W.supported();
    const auto = p.taskOpts && p.taskOpts.preset === 'auto', withPreset = (k) => Object.assign({}, p, { taskOpts: Object.assign({}, p.taskOpts, { preset: k }) });
    let built = null, mode = null;
    const build = (q) => { built = DL.setup(q); return built; };
    const cpuJob = (b) => ({
      get model() { return b.net; },
      step: () => Object.assign(b.trainer.step(), { backend: 'cpu' }),
      tensors: () => b.trainer.tensors(),
      checkpoint: () => DL.save(b.net),
      restore: (s) => b.net.loadState(s.state || s),
    });
    if (!maybe) return cpuJob(build(auto ? withPreset('cpu') : p));
    if (!auto) { build(p); if (DL.gpuSupports(built.net, built.trainer)) return cpuJob(built); }
    const job = {
      get model() { return built && built.net; },
      async step() {
        if (!mode) {
          const ok = (await DL.pickPreset(want)).gpu;
          if (auto) { build(withPreset(ok ? 'webgpu' : 'cpu')); if (job.pending) built.net.loadState(job.pending.state || job.pending); }
          if (ok) { try { await built.trainer.useGPU(); mode = 'webgpu'; } catch (e) { if (want === 'webgpu') throw e; mode = 'cpu'; job.why = String(e.message || e); } }
          else if (want === 'webgpu') throw new Error('WebGPU is not available here: ' + S.T.gpu.why);
          else { mode = 'cpu'; job.why = S.T.gpu.why; }
        }
        const r = mode === 'webgpu' ? await built.trainer.stepAsync() : Object.assign(built.trainer.step(), { backend: 'cpu' });
        if (built.trainer.stepNo === 1) r.info = { backend: mode, preset: built.task.preset || null, params: built.net.numParams(), config: built.net.config, why: job.why || null };
        return r;
      },
      sync: () => (mode === 'webgpu' ? built.trainer.syncAsync() : null),
      tensors: () => built.trainer.tensors(),
      checkpoint: () => DL.save(built.net),
      restore: (s) => { if (!built) { job.pending = s; return; } built.net.loadState(s.state || s); if (built.trainer.gpu) built.trainer.gpu.upload(); },
    };
    return job;
  };
  DL.ensureJobs = () => {
    const J = S.T && S.T.jobs;
    if (!J) return false;
    for (const [name, make] of Object.entries(DL.jobs)) if (!J.has(name)) J.register(name, make);
    return true;
  };
  // the job registry is made by a later file: register as soon as it exists. In a worker every file loads in one
  // script, so the microtask runs before any message arrives; on a page the timer retries until the engine is in.
  const retry = () => { if (!DL.ensureJobs()) setTimeout(retry, 0); };
  (typeof queueMicrotask === 'function' ? queueMicrotask : (f) => Promise.resolve().then(f))(retry);
  /** train in a Web Worker where one can be made (else on the page in slices); messages as in S.T.train */
  DL.train = (params, opts = {}, onMessage, job = 'dl') => { DL.ensureJobs(); return S.T.train(job, params, opts, onMessage); };
})(globalThis.Nooral = globalThis.Nooral || {});
