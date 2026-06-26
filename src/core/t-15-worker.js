// Training loops that stream. A job is made by a registered factory from plain parameters (so it can be built
// inside a Web Worker); a Runner drives it a few steps at a time and posts messages:
//   {type: 'progress', step, loss, metrics, lr, ms}   every step (or every opts.progressEvery)
//   {type: 'snapshot', step, tensors: {name: {shape, data}}}   every opts.snapshotEvery steps (opts.snapshot = names)
//   {type: 'checkpoint', step, state}   every opts.checkpointEvery steps;  {type: 'state', state: 'running' | ...}
//   {type: 'done', step, reason}   {type: 'error', message}
//   S.T.jobs.register(name, (params) => job)     job: {step() -> {loss, metrics?, lr?, backend?} or a promise of it,
//                                    tensors?() -> {name: tensor}, sync?() (a promise: bring tensors back before
//                                    a snapshot or checkpoint), checkpoint?() -> JSON, restore?(state), model?}
//   new S.T.Runner(job, opts, post)  .start() .pause() .resume() .step(n) .stop() .snapshot(names?) .checkpoint()
//   S.T.train(name, params, opts, onMessage) -> {mode: 'worker' | 'page', pause, resume, step(n), stop, snapshot, checkpoint}
//   S.T.worker.serve(self)   the worker side; S.T.worker.source() the code a worker is made from
(function (S) {
  'use strict';
  const T = S.T;
  const registry = new Map();
  T.jobs = {
    register(name, factory) { registry.set(name, factory); },
    has: (name) => registry.has(name),
    names: () => [...registry.keys()],
    make(name, params) { const f = registry.get(name); if (!f) throw new Error('no training job called ' + name); return f(params || {}); },
  };
  const pack = (t) => ({ shape: t.shape.slice(), data: Float32Array.from(t.values()) });
  const later = typeof setImmediate === 'function' ? (f) => setImmediate(f) : (f) => setTimeout(f, 0);

  class Runner {
    constructor(job, opts = {}, post = () => {}) {
      this.job = job; this.opts = opts; this.post = post;
      this.stepNo = 0; this.state = 'idle'; this.history = [];
      this.maxSteps = opts.maxSteps || Infinity; this.chunkMs = opts.chunkMs || 30;
    }
    setState(s) { this.state = s; this.post({ type: 'state', state: s, step: this.stepNo }); }
    /** one step; a job whose step returns a promise (a GPU job) gives a promise of the same answer */
    one() {
      const t0 = performance.now(), r = this.job.step(this.stepNo);
      if (r && typeof r.then === 'function') return r.then((x) => this.after(x || {}, t0));
      return this.after(r || {}, t0);
    }
    after(r, t0) {
      this.stepNo++;
      const msg = { type: 'progress', step: this.stepNo, loss: r.loss, metrics: r.metrics || null, lr: r.lr, ms: performance.now() - t0 };
      if (r.backend) msg.backend = r.backend;
      if (r.info) msg.info = r.info;
      this.history.push({ step: this.stepNo, loss: r.loss });
      const o = this.opts;
      if (!o.progressEvery || this.stepNo % o.progressEvery === 0) this.post(msg);
      if (o.snapshotEvery && this.stepNo % o.snapshotEvery === 0) this.snapshot(o.snapshot);
      if (o.checkpointEvery && this.stepNo % o.checkpointEvery === 0) this.checkpoint();
      if (!(r.loss === undefined || isFinite(r.loss))) { this.finish('the loss became ' + r.loss); return false; }
      if (r.stop) { this.finish(r.stop); return false; }
      if (this.stepNo >= this.maxSteps) { this.finish('reached ' + this.maxSteps + ' steps'); return false; }
      return true;
    }
    fail(e) { this.state = 'done'; this.busy = false; this.post({ type: 'error', message: String((e && e.message) || e), step: this.stepNo }); }
    finish(reason) { this.state = 'done'; this.post({ type: 'done', step: this.stepNo, reason }); }
    /** steps while `more()` holds, then done(); asynchronous steps run one after another, never two at once */
    run(more, done) {
      if (this.busy) return;
      const go = () => {
        try {
          while (more()) {
            const k = this.one();
            if (k && typeof k.then === 'function') { this.busy = true; k.then((ok) => { this.busy = false; if (ok) go(); }, (e) => this.fail(e)); return; }
            if (!k) return;
          }
        } catch (e) { this.fail(e); return; }
        done();
      };
      go();
    }
    loop() {
      if (this.state !== 'running') return;
      const t0 = performance.now();
      this.run(() => this.state === 'running' && performance.now() - t0 < this.chunkMs, () => { if (this.state === 'running') later(() => this.loop()); });
    }
    start() { if (this.state === 'done') return; this.setState('running'); this.loop(); }
    pause() { if (this.state === 'running') this.setState('paused'); }
    resume() { if (this.state === 'paused' || this.state === 'idle') this.start(); }
    /** n steps now, then paused (for stepping through training) */
    step(n = 1) {
      if (this.state === 'running') this.setState('paused');
      let i = 0;
      this.run(() => i++ < n && this.state !== 'done', () => { if (this.state === 'running') this.loop(); else if (this.state !== 'done') this.setState('paused'); });
    }
    stop() { if (this.state !== 'done') this.finish('stopped'); }
    /** a job that trains elsewhere (on the GPU) brings its tensors back first: then the message is posted later */
    snapshot(names) {
      const sy = this.job.sync && this.job.sync();
      if (sy && typeof sy.then === 'function') return sy.then(() => this.snapshotNow(names));
      return this.snapshotNow(names);
    }
    snapshotNow(names) {
      const all = this.job.tensors ? this.job.tensors() : this.job.model ? Object.fromEntries(this.job.model.namedState()) : {};
      const pick = names && names.length ? names : Object.keys(all), tensors = {};
      for (const n of pick) if (all[n]) tensors[n] = pack(all[n]);
      const msg = { type: 'snapshot', step: this.stepNo, tensors };
      this.post(msg);
      return msg;
    }
    checkpoint() {
      const sy = this.job.sync && this.job.sync();
      if (sy && typeof sy.then === 'function') return sy.then(() => this.checkpointNow());
      return this.checkpointNow();
    }
    checkpointNow() {
      const state = this.job.checkpoint ? this.job.checkpoint() : this.job.model ? this.job.model.stateDict() : null;
      const msg = { type: 'checkpoint', step: this.stepNo, state };
      this.post(msg);
      return msg;
    }
    restore(state) { if (this.job.restore) this.job.restore(state); else if (this.job.model) this.job.model.loadState(state); }
  }
  T.Runner = Runner;

  const Wk = (T.worker = {});
  /** the worker side: start / pause / resume / step / stop / snapshot / checkpoint / restore */
  Wk.serve = (self) => {
    let runner = null;
    self.onmessage = (e) => {
      const m = e.data || {};
      try {
        if (m.cmd === 'start') {
          const job = T.jobs.make(m.job, m.params);
          runner = new Runner(job, m.opts || {}, (x) => self.postMessage(x));
          if (m.restore) runner.restore(m.restore);
          if (m.opts && m.opts.paused) runner.setState('paused'); else runner.start();
        } else if (runner && typeof runner[m.cmd] === 'function' && ['pause', 'resume', 'step', 'stop', 'snapshot', 'checkpoint', 'restore'].includes(m.cmd)) runner[m.cmd](m.arg);
      } catch (err) { self.postMessage({ type: 'error', message: String((err && err.message) || err) }); }
    };
  };
  /** the code for a worker: every core file, cut from the built page or imported by address on the dev page */
  Wk.source = () => {
    if (typeof document === 'undefined') return null;
    const tags = [...document.querySelectorAll('script[src]')].filter((s) => /(^|\/)src\/core\/[^/]+\.js$/.test(s.getAttribute('src')));
    if (tags.length) return 'self.Nooral = {};\nimportScripts(' + tags.map((t) => JSON.stringify(new URL(t.getAttribute('src'), location.href).href)).join(', ') + ');\nNooral.T.worker.serve(self);\n';
    const text = [...document.scripts].map((s) => s.textContent || '').find((t) => t.includes('// ==== core/t-15-worker.js ===='));
    if (!text) return null;
    // each file starts with a line "// ==== core/name.js ====" (only at the start of a line)
    const heads = [...text.matchAll(/(^|\n)\/\/ ==== ([^\n]+) ====(?=\n)/g)].map((x) => ({ at: x.index + x[1].length, core: x[2].startsWith('core/') }));
    const parts = heads.map((h, i) => (h.core ? text.slice(h.at, i + 1 < heads.length ? heads[i + 1].at : text.length) : null)).filter(Boolean);
    return 'self.Nooral = {};\n' + parts.join('\n') + '\nNooral.T.worker.serve(self);\n';
  };
  let url = null;
  Wk.url = () => {
    if (url) return url;
    if (typeof Worker === 'undefined' || typeof Blob === 'undefined') return null;
    const code = Wk.source();
    if (!code) return null;
    url = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
    return url;
  };
  /** train a registered job in a worker when one can be made, else on the page in short slices */
  T.train = (name, params, opts = {}, onMessage = () => {}) => {
    const u = opts.forcePage ? null : Wk.url();
    if (u) {
      let w = null;
      try { w = new Worker(u); } catch (e) { w = null; }
      if (w) {
        let finished = false;
        w.onmessage = (e) => { const d = e.data; if (d.type === 'done' || d.type === 'error') finished = true; onMessage(d); };
        w.onerror = (e) => { e.preventDefault(); if (!finished) onMessage({ type: 'error', message: 'worker failed: ' + (e.message || 'unknown') }); };
        w.postMessage({ cmd: 'start', job: name, params, opts: Object.assign({}, opts, { forcePage: undefined }), restore: opts.restore });
        const send = (cmd) => (arg) => w.postMessage({ cmd, arg });
        return { mode: 'worker', worker: w, pause: send('pause'), resume: send('resume'), step: send('step'), snapshot: send('snapshot'), checkpoint: send('checkpoint'),
          stop() { w.postMessage({ cmd: 'stop' }); setTimeout(() => w.terminate(), 50); } };
      }
    }
    const runner = new Runner(T.jobs.make(name, params), opts, (m) => later(() => onMessage(m)));
    if (opts.restore) runner.restore(opts.restore);
    if (opts.paused) runner.setState('paused'); else runner.start();
    return { mode: 'page', runner, pause: () => runner.pause(), resume: () => runner.resume(), step: (n) => runner.step(n), stop: () => runner.stop(), snapshot: (n) => runner.snapshot(n), checkpoint: () => runner.checkpoint() };
  };

  // ---- a general job: an MLP classifier on a table of numbers ----
  // params: {X: [[...]], y: [class ids], sizes: [hidden...], act, optimizer, lr, batch, seed, weightDecay}
  T.jobs.register('mlp', (p) => {
    const g = T.rng(p.seed || 1), X = T.tensor(p.X), y = Float64Array.from(p.y), k = p.classes || Math.max(...p.y) + 1;
    const model = new T.nn.MLP([X.shape[1]].concat(p.sizes || [16], [k]), { rng: g, act: p.act || 'relu' });
    const opt = T.optim.make(p.optimizer || 'adam', model.parameters(), { lr: p.lr || 0.01, weightDecay: p.weightDecay || 0 });
    const n = X.shape[0], bs = Math.min(p.batch || n, n);
    let order = g.permutation(n), at = 0;
    return {
      model,
      step() {
        if (at + bs > n) { order = g.permutation(n); at = 0; }
        const idx = order.slice(at, at + bs); at += bs;
        const xb = T.take(X, idx, 0), yb = idx.map((i) => y[i]);
        opt.zeroGrad();
        const logits = model.forward(xb), loss = T.crossEntropy(logits, yb);
        loss.backward(); opt.step();
        return { loss: loss.item(), metrics: { accuracy: T.accuracy(logits, yb) }, lr: opt.lr };
      },
      tensors() { const o = Object.fromEntries(model.namedState()); return o; },
    };
  });
})(globalThis.Nooral = globalThis.Nooral || {});
