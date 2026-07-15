// Deep-learning models on the tensor engine (S.T), shared pieces. Every model is a DL.Net: a module whose forward
// pass names its intermediate outputs (taps), so pictures, Grad-CAM and saliency can take a layer by name.
//   DL.register(kind, (config) => net)   DL.build(kind, config) -> net (net.kind, net.config)   DL.kinds()
//   net.forward(x, opts) leaves net.outputs = {name: tensor}; net.tapNames(); net.hooks = {name: (t) => t'}
//   net.lossOn(batch) -> {loss, logits, correct, count}   net.isLM (report perplexity)
//   await net.trainStepAsync(batch, {lr, weightDecay, clip}) an AdamW step on the GPU (GPT)   await net.syncFromGPU()
//   DL.save(net) -> JSON {format, kind, config, state}    DL.load(json) -> net (same predictions)
//   DL.paramCounts(net) -> {total, layers: [{name, shape, size}]}
//   DL.nearest(E, query, k, {metric, labels, exclude}) nearest rows of an embedding matrix; DL.neighbours(net, token, k)
//   DL.project2d(E) -> {points, components, explained}   DL.attentionMaps(net, b) -> [{layer, head, weights}]
//   DL.gradCam(net, x, cls, layer)  DL.saliency(net, x, cls)  DL.camNet(net, layer) (for S.Explain.gradCamT)
//   DL.gradCheck(make, opts) -> {worst, checked, ok}  every parameter of a small float64 net against finite differences
(function (S) {
  'use strict';
  const DL = (S.DL = S.DL || {});
  const fmt = (x) => S.T.fmt(x);
  DL.step = (kind, say, data) => Object.assign({ kind, say }, data || {});

  const builders = new Map();
  DL.register = (kind, make) => builders.set(kind, make);
  DL.kinds = () => [...builders.keys()];
  DL.build = (kind, config = {}) => {
    const make = builders.get(kind);
    if (!make) throw new Error('no model kind called ' + kind);
    const net = make(Object.assign({}, config));
    net.kind = kind;
    net.config = Object.assign({}, config, net.config || {});
    return net;
  };

  /** the base of every model: named intermediate outputs, hooks, and a loss on a batch */
  DL.makeNet = () => {
    const nn = S.T.nn;
    if (DL.Net) return DL.Net;
    DL.Net = class Net extends nn.Module {
      /** record t under name (and let a hook replace it); returns the tensor to carry on with */
      tap(name, t) {
        if (this.hooks && this.hooks[name]) { const r = this.hooks[name](t); if (r) t = r; }
        if (this.outputs) this.outputs[name] = t;
        return t;
      }
      begin() { this.outputs = {}; }
      tapNames() { return Object.keys(this.outputs || {}); }
      /** forward without recording a graph, in evaluation mode */
      predict(x, opts) {
        const was = this.training;
        this.eval();
        try { return S.T.noGrad(() => this.forward(x, opts)); } finally { this.setMode(was); }
      }
      /** one AdamW step on the GPU (a GPT; see dl-i-gpu.js): the model keeps its own GPU session and step count */
      async trainStepAsync(batch, o = {}) {
        if (!this.gpuSession) { this.gpuSession = await DL.gpuSession(this, { decoupled: o.optimizer !== 'adam' }); this.gpuT = 0; }
        this.gpuT++;
        return this.gpuSession.step(batch, { lr: o.lr || 1e-3, t: this.gpuT, betas: o.betas, eps: o.eps, weightDecay: o.weightDecay === undefined ? 0.01 : o.weightDecay, clip: o.clip || 0 });
      }
      /** copy the parameters trained on the GPU back into this model */
      async syncFromGPU() { if (this.gpuSession) await this.gpuSession.sync(); return this; }
      /** classification loss on {x, y}: cross-entropy, with the number right */
      lossOn(b) {
        const logits = this.forward(b.x), loss = S.T.crossEntropy(logits, b.y, { ignoreIndex: -1 });
        return Object.assign({ loss, logits }, DL.countRight(logits, b.y));
      }
    };
    return DL.Net;
  };

  /** predictions right and counted, ignoring targets of −1 */
  DL.countRight = (logits, y) => {
    const am = S.T.argmax(logits, -1).values(), yv = Array.isArray(y) ? y.flat(Infinity) : Array.from(y.values ? y.values() : y);
    let correct = 0, count = 0;
    for (let i = 0; i < am.length; i++) { if (yv[i] === -1) continue; count++; if (am[i] === yv[i]) correct++; }
    return { correct, count };
  };

  // ---- saving ----
  DL.save = (net) => ({ format: 'nooral-dl/1', kind: net.kind, config: JSON.parse(JSON.stringify(net.config || {})), params: net.numParams(), state: net.stateDict() });
  DL.load = (obj) => {
    if (typeof obj === 'string') obj = JSON.parse(obj);
    if (!obj || obj.format !== 'nooral-dl/1') throw new Error('not a saved Nooral model');
    return DL.build(obj.kind, obj.config).loadState(obj.state);
  };
  DL.paramCounts = (net) => {
    const layers = net.namedParameters().map(([name, p]) => ({ name, shape: p.shape.slice(), size: p.size }));
    return { total: layers.reduce((s, l) => s + l.size, 0), layers };
  };
  DL.pack = (t) => ({ shape: t.shape.slice(), data: Float32Array.from(t.values()) });

  // ---- embeddings ----
  const rowsOf = (E) => {
    if (E instanceof S.T.Tensor) { const [n, d] = E.shape, v = E.values(); return Array.from({ length: n }, (_, i) => v.subarray(i * d, (i + 1) * d)); }
    return E;
  };
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = (a) => Math.sqrt(dot(a, a)) || 1;
  DL.cosine = (a, b) => dot(a, b) / (norm(a) * norm(b));
  /**
   * nearest(E, query, k): the k rows of E closest to query (a row index or a vector), by cosine similarity
   * (metric 'cosine', higher is nearer) or Euclidean distance ('euclid', lower is nearer).
   */
  DL.nearest = (E, query, k = 5, o = {}) => {
    const rows = rowsOf(E), q = typeof query === 'number' ? rows[query] : query, metric = o.metric || 'cosine';
    const skip = new Set(o.exclude || (typeof query === 'number' ? [query] : []));
    const scored = [];
    rows.forEach((r, i) => {
      if (skip.has(i)) return;
      let s;
      if (metric === 'cosine') s = DL.cosine(q, r);
      else { s = 0; for (let j = 0; j < r.length; j++) s += (r[j] - q[j]) ** 2; s = Math.sqrt(s); }
      scored.push({ index: i, score: s, label: o.labels ? o.labels[i] : String(i) });
    });
    scored.sort((a, b) => (metric === 'cosine' ? b.score - a.score : a.score - b.score) || a.index - b.index);
    const top = scored.slice(0, k), qn = typeof query === 'number' && o.labels ? '"' + o.labels[query] + '"' : 'the query';
    top.say = 'Nearest to ' + qn + ': ' + top.map((t) => t.label + ' (' + fmt(t.score) + ')').join(', ') + '.';
    return top;
  };
  /** the nearest tokens to a token (its string or id) in a model's embedding matrix, labelled by its tokenizer */
  DL.neighbours = (net, token, k = 5) => {
    const E = net.extras().embeddings, tok = net.tokenizer, id = typeof token === 'number' ? token : tok && tok.index.get(token);
    if (id === undefined) throw new Error('no token ' + JSON.stringify(token) + ' in the vocabulary');
    return DL.nearest(E, id, k, { labels: tok ? tok.vocab.map((v) => JSON.stringify(v).slice(1, -1)) : null });
  };
  /** the two main directions of the rows (principal components by power iteration) -> [[x, y], ...] */
  DL.project2d = (E, o = {}) => {
    const rows = rowsOf(E).map((r) => Array.from(r)), n = rows.length, d = rows[0].length;
    const mean = new Array(d).fill(0);
    for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j] / n;
    const X = rows.map((r) => r.map((v, j) => v - mean[j]));
    const C = Array.from({ length: d }, () => new Array(d).fill(0));
    for (const r of X) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += (r[i] * r[j]) / n;
    const comps = [], vars = [];
    const g = S.T.rng(o.seed || 3);
    for (let c = 0; c < 2; c++) {
      let v = Array.from({ length: d }, () => g.normal());
      for (let it = 0; it < 200; it++) {
        let w = C.map((row) => dot(row, v));
        for (const u of comps) { const p = dot(w, u); w = w.map((x, j) => x - p * u[j]); }
        const nv = norm(w); v = w.map((x) => x / nv);
      }
      if (v[v.findIndex((x) => Math.abs(x) > 1e-9)] < 0) v = v.map((x) => -x);
      comps.push(v); vars.push(dot(C.map((row) => dot(row, v)), v));
    }
    const total = C.reduce((s, r, i) => s + r[i], 0) || 1;
    return { points: X.map((r) => [dot(r, comps[0]), dot(r, comps[1])]), components: comps, explained: vars.map((v) => v / total) };
  };

  // ---- attention retrieval ----
  /** the attention weights of the last forward pass: [{layer, head, weights: [[...]]}] for batch item b */
  DL.attentionMaps = (net, b = 0) => {
    const out = [];
    for (const name of net.tapNames()) {
      const m = /^block(\d+)\.attn$/.exec(name);
      if (!m) continue;
      const w = net.outputs[name], [, H, Tq, Tk] = w.shape, v = w.values();
      for (let h = 0; h < H; h++) {
        const base = ((b * H + h) * Tq) * Tk;
        out.push({ layer: +m[1], head: h, weights: Array.from({ length: Tq }, (_, i) => Array.from(v.subarray(base + i * Tk, base + (i + 1) * Tk))) });
      }
    }
    return out;
  };

  // ---- interpretability ----
  const asInput = (net, x) => {
    if (x instanceof S.T.Tensor) return x;
    const c = net.config || {}, ch = c.channels || 1, s = c.size || Math.round(Math.sqrt(x.length / ch));
    return S.T.tensor(Array.from(x), [1, ch, s, s]);
  };
  /** score of class cls in the first row of logits, as a scalar in the graph */
  const scoreOf = (logits, cls) => { const m = new Array(logits.size).fill(0); m[cls] = 1; return S.T.sum(S.T.mul(logits, S.T.tensor(m, logits.shape))); };
  const top1 = (logits) => S.T.argmax(logits, -1).values()[0];
  /** Grad-CAM on a named convolutional output of a CNN (default its last convolution after ReLU) */
  DL.gradCam = (net, x, cls, layer) => {
    const was = net.training; net.eval();
    try {
      const logits = S.T.enableGrad(() => net.forward(asInput(net, x)));
      layer = layer || net.camLayer || 'relu2';
      if (cls == null) cls = top1(logits);
      const act = net.outputs[layer];
      if (!act) throw new Error('no output called ' + layer + '; the layers are ' + net.tapNames().join(', '));
      scoreOf(logits, cls).backward();
      const [, C, H, W] = act.shape, a = act.values(), g = act.grad ? act.grad.values() : new Float32Array(act.size);
      const grid = (v) => Array.from({ length: C }, (_, c) => Array.from({ length: H }, (_, i) => Array.from({ length: W }, (_, j) => v[(c * H + i) * W + j])));
      const r = S.Explain.gradCam(grid(a), grid(g));
      r.layer = layer; r.cls = cls;
      return r;
    } finally { net.setMode(was); }
  };
  /** an adapter for S.Explain.gradCamT(net, x, shape, cls): hands the named layer over through keep() */
  DL.camNet = (net, layer) => (input, keep) => {
    const name = layer || net.camLayer || 'relu2';
    net.hooks = Object.assign({}, net.hooks, { [name]: keep });
    try { net.eval(); return net.forward(input); } finally { delete net.hooks[name]; }
  };
  /** saliency: |∂ score / ∂ pixel|, scaled to [0, 1], with the input's shape */
  DL.saliency = (net, x, cls) => {
    const was = net.training; net.eval();
    try {
      const x0 = asInput(net, x), inp = S.T.param(S.T.tensor(Float32Array.from(x0.values()), x0.shape.slice()));
      const logits = S.T.enableGrad(() => net.forward(inp));
      if (cls == null) cls = top1(logits);
      scoreOf(logits, cls).backward();
      const g = inp.grad.values(), map = Array.from(g, Math.abs), mx = Math.max(...map) || 1;
      const peak = map.indexOf(mx);
      return { cls, shape: inp.shape.slice(), map: map.map((v) => v / mx), raw: Array.from(g),
        trace: [DL.step('saliency', 'The score for class ' + cls + ' is most sensitive to pixel ' + peak + ', where a small change moves it by ' + fmt(mx) + ' per unit.', { peak, max: mx })] };
    } finally { net.setMode(was); }
  };

  /**
   * gradCheck(make): make() runs inside float64 storage and returns {net, loss: () => scalar tensor}. Every
   * parameter (up to opts.per elements each) is compared with central differences; relative error per element.
   */
  DL.gradCheck = (make, o = {}) => S.T.float64(() => {
    const { net, loss } = make(), eps = o.eps || 1e-6, per = o.per || 6, g = S.T.rng(o.seed || 5);
    net.zeroGrad();
    loss().backward();
    let worst = 0, checked = 0, where = '';
    for (const [name, p] of net.namedParameters()) {
      const picks = p.size <= per ? [...Array(p.size).keys()] : Array.from({ length: per }, () => g.int(p.size));
      for (const i of picks) {
        const old = p.data[i];
        p.data[i] = old + eps; const lp = S.T.noGrad(loss).item();
        p.data[i] = old - eps; const lm = S.T.noGrad(loss).item();
        p.data[i] = old;
        const num = (lp - lm) / (2 * eps), ana = p.grad ? p.grad.data[i] : 0;
        const err = Math.abs(num - ana) / Math.max(1e-3, Math.abs(num) + Math.abs(ana));
        checked++;
        if (err > worst) { worst = err; where = name + '[' + i + '] analytic ' + ana + ' numeric ' + num; }
      }
    }
    return { worst, checked, where, ok: worst < (o.tol || 1e-4) };
  });
})(globalThis.Nooral = globalThis.Nooral || {});
