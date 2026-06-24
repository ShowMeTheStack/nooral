// A small module framework. A Module holds parameters (tensors made with this.param), buffers (state that is saved
// but not trained, like batch-norm running statistics) and sub-modules, found by walking its own fields.
//   m.forward(x) / m.call(x)   m.parameters()   m.namedParameters() -> [[path, tensor]]   m.train() / m.eval()
//   m.stateDict() -> {path: {shape, data: [...]}} (JSON)   m.loadState(obj)
//   m.toTyped() -> {names, shapes, buffer: Float32Array}   m.fromTyped(t)   m.numParams()
// Layers (S.T.nn): Dense, Conv2d, MaxPool, AvgPool, Flatten, Dropout, BatchNorm, LayerNorm, Embedding,
//   Activation('relu' | 'tanh' | 'sigmoid' | 'gelu' | 'leakyRelu'), Sequential. RNNs and attention are in later files.
// Every layer takes {rng} (a seed or generator) for its initial weights, so the same seed gives the same model.
(function (S) {
  'use strict';
  const T = S.T;
  const nn = (T.nn = {});

  class Module {
    constructor() { this.training = true; }
    param(name, t) { t.requiresGrad = true; t.isParam = true; t.name = name; return t; }
    buffer(name, t) { t.isBuffer = true; t.name = name; return t; }
    children() { return Object.entries(this).filter(([k, v]) => k !== 'parent' && (v instanceof Module || (Array.isArray(v) && v.some((x) => x instanceof Module)))); }
    /** visit every parameter and buffer with its path (which also becomes its name in graph pictures) */
    walk(fn0, prefix = '') {
      const fn = (n, t) => { t.name = n; fn0(n, t); };
      for (const [k, v] of Object.entries(this)) {
        if (k === 'parent') continue;
        if (v instanceof T.Tensor && (v.isParam || v.isBuffer)) fn(prefix + k, v);
        else if (v instanceof Module) v.walk(fn0, prefix + k + '.');
        else if (Array.isArray(v)) v.forEach((x, i) => { if (x instanceof Module) x.walk(fn0, prefix + k + '.' + i + '.'); else if (x instanceof T.Tensor && (x.isParam || x.isBuffer)) fn(prefix + k + '.' + i, x); });
      }
    }
    namedParameters() { const r = []; this.walk((n, t) => { if (t.isParam) r.push([n, t]); }); return r; }
    parameters() { return this.namedParameters().map((x) => x[1]); }
    namedState() { const r = []; this.walk((n, t) => r.push([n, t])); return r; }
    numParams() { return this.parameters().reduce((s, p) => s + p.size, 0); }
    setMode(on) { this.training = on; for (const [, v] of this.children()) (Array.isArray(v) ? v : [v]).forEach((m) => m instanceof Module && m.setMode(on)); return this; }
    train() { return this.setMode(true); }
    eval() { return this.setMode(false); }
    zeroGrad() { for (const p of this.parameters()) p.grad = null; }
    call(...a) { return this.forward(...a); }
    stateDict() { const o = {}; for (const [n, t] of this.namedState()) o[n] = { shape: t.shape.slice(), data: Array.from(t.values()) }; return o; }
    loadState(o) {
      for (const [n, t] of this.namedState()) {
        const e = o[n];
        if (!e) throw new Error('loadState: no entry for ' + n);
        if (!T.sameShape(e.shape, t.shape)) throw new Error('loadState: ' + n + ' has shape [' + e.shape + '], expected [' + t.shape + ']');
        t.data.set(e.data);
      }
      return this;
    }
    toTyped() {
      const st = this.namedState(), total = st.reduce((s, [, t]) => s + t.size, 0), buffer = new Float32Array(total);
      let at = 0;
      for (const [, t] of st) { buffer.set(t.values(), at); at += t.size; }
      return { names: st.map((x) => x[0]), shapes: st.map((x) => x[1].shape.slice()), buffer };
    }
    fromTyped(tp) {
      let at = 0;
      const st = this.namedState();
      st.forEach(([n, t], i) => {
        if (tp.names[i] !== n || !T.sameShape(tp.shapes[i], t.shape)) throw new Error('fromTyped: layout differs at ' + n);
        t.data.set(tp.buffer.subarray(at, at + t.size)); at += t.size;
      });
      return this;
    }
    /** a short description of the layers, for a picture or a log */
    describe(prefix = '') {
      const own = Object.entries(this).filter(([, v]) => v instanceof T.Tensor && v.isParam).map(([k, v]) => k + '[' + v.shape + ']');
      const lines = [prefix + this.constructor.name + (own.length ? ' ' + own.join(' ') : '')];
      for (const [k, v] of this.children()) (Array.isArray(v) ? v : [v]).forEach((m, i) => m instanceof Module && lines.push(...m.describe(prefix + '  ').map((l, j) => (j ? l : l.replace(/^\s*/, prefix + '  ' + k + (Array.isArray(v) ? '.' + i : '') + ': ')))));
      return lines;
    }
  }
  nn.Module = Module;
  const rngOf = (o) => T.asRng(o && o.rng !== undefined ? o.rng : 1);

  nn.Sequential = class Sequential extends Module {
    constructor(...layers) { super(); this.layers = layers.flat(); }
    forward(x) { this.outputs = []; for (const l of this.layers) { x = l.forward(x); this.outputs.push(x); } return x; }
  };
  /** y = x W + b with W [in, out]; init 'he' (default for relu nets) or 'xavier' */
  nn.Dense = class Dense extends Module {
    constructor(inF, outF, o = {}) {
      super();
      const g = rngOf(o);
      this.inF = inF; this.outF = outF;
      this.W = this.param('W', T.init(o.init || 'xavier', [inF, outF], g));
      if (o.bias !== false) this.b = this.param('b', T.zeros([outF]));
    }
    forward(x) { return T.linear(x, this.W, this.b); }
  };
  nn.Conv2d = class Conv2d extends Module {
    constructor(cin, cout, k, o = {}) {
      super();
      const kk = Array.isArray(k) ? k : [k, k], groups = o.groups || 1, g = rngOf(o);
      this.opts = { stride: o.stride || 1, padding: o.padding || 0, dilation: o.dilation || 1, groups };
      this.W = this.param('W', T.init(o.init || 'he', [cout, cin / groups, kk[0], kk[1]], g));
      if (o.bias !== false) this.b = this.param('b', T.zeros([cout]));
    }
    forward(x) { return T.conv2d(x, this.W, this.b, this.opts); }
  };
  nn.MaxPool = class MaxPool extends Module { constructor(k, stride, padding = 0) { super(); this.k = k; this.stride = stride; this.padding = padding; } forward(x) { return T.maxPool2d(x, this.k, this.stride, this.padding); } };
  nn.AvgPool = class AvgPool extends Module { constructor(k, stride, padding = 0) { super(); this.k = k; this.stride = stride; this.padding = padding; } forward(x) { return T.avgPool2d(x, this.k, this.stride, this.padding); } };
  nn.Flatten = class Flatten extends Module { forward(x) { return T.flatten(x, 1); } };
  nn.Dropout = class Dropout extends Module {
    constructor(p = 0.5, o = {}) { super(); this.p = p; this.rng = rngOf(o); }
    forward(x) { return T.dropout(x, this.p, this.rng, this.training); }
  };
  nn.BatchNorm = class BatchNorm extends Module {
    constructor(ch, o = {}) {
      super();
      this.gamma = this.param('gamma', T.ones([ch])); this.beta = this.param('beta', T.zeros([ch]));
      this.runningMean = this.buffer('runningMean', T.zeros([ch])); this.runningVar = this.buffer('runningVar', T.ones([ch]));
      this.momentum = o.momentum === undefined ? 0.1 : o.momentum; this.eps = o.eps || 1e-5;
    }
    forward(x) { return T.batchNorm(x, this.gamma, this.beta, { training: this.training, running: { mean: this.runningMean, var: this.runningVar }, momentum: this.momentum, eps: this.eps }); }
  };
  nn.LayerNorm = class LayerNorm extends Module {
    constructor(d, o = {}) { super(); this.gamma = this.param('gamma', T.ones([d])); this.beta = this.param('beta', T.zeros([d])); this.eps = o.eps || 1e-5; }
    forward(x) { return T.layerNorm(x, this.gamma, this.beta, this.eps); }
  };
  nn.Embedding = class Embedding extends Module {
    constructor(n, d, o = {}) { super(); this.W = this.param('W', T.normal([n, d], 0, o.std || 1, rngOf(o))); }
    forward(ids) { return T.embedding(this.W, ids); }
  };
  nn.Activation = class Activation extends Module {
    constructor(kind = 'relu', arg) { super(); this.kind = kind; this.arg = arg; }
    forward(x) { return this.arg === undefined ? T[this.kind](x) : T[this.kind](x, this.arg); }
  };
  for (const k of ['relu', 'tanh', 'sigmoid', 'gelu', 'leakyRelu']) nn[k[0].toUpperCase() + k.slice(1)] = class extends nn.Activation { constructor(a) { super(k, a); } };
  /** a multilayer perceptron: sizes [in, h1, ..., out], hidden activation `act` */
  nn.MLP = class MLP extends nn.Sequential {
    constructor(sizes, o = {}) {
      const g = rngOf(o), layers = [];
      for (let i = 0; i + 1 < sizes.length; i++) {
        layers.push(new nn.Dense(sizes[i], sizes[i + 1], { rng: g, init: o.init || ((o.act || 'relu') === 'relu' ? 'he' : 'xavier') }));
        if (i + 2 < sizes.length) layers.push(new nn.Activation(o.act || 'relu'));
      }
      super(...layers);
    }
  };
})(globalThis.Nooral = globalThis.Nooral || {});
