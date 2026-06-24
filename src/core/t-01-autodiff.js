// Reverse-mode autodiff over a recorded graph. Every op that has an input needing a gradient records a node on its
// output: {op, parents, backward, info}. t.backward() walks the graph in reverse topological order and leaves a
// gradient on every node (leaves and intermediates alike), so the whole graph can be drawn with values and grads.
//   S.T.param(t, name)   a leaf that needs a gradient          t.named('h1')  a name for the pictures
//   S.T.noGrad(fn)       run fn without recording               t.detach()     the same values, cut from the graph
//   t.backward(g?)       g defaults to ones (a scalar loss)     t.zeroGrad()
//   S.T.graph(root)      -> {nodes: [{id, name, op, shape, value, grad, parents}], edges: [[from, to]]}
//   S.T.traceForward(root) / traceBackward(root) -> steps {kind, say, ...}
// Ops are defined with S.T.def(name, fn), which also puts them on tensors as methods: a.add(b) is S.T.add(a, b).
(function (S) {
  'use strict';
  const T = S.T;
  let enabled = true;
  T.gradEnabled = () => enabled;
  T.noGrad = (fn) => { const old = enabled; enabled = false; try { return fn(); } finally { enabled = old; } };
  T.enableGrad = (fn) => { const old = enabled; enabled = true; try { return fn(); } finally { enabled = old; } };

  /** numbers and arrays become tensors (of the current float type) */
  T.as = (x, like) => {
    if (x instanceof T.Tensor) return x;
    if (typeof x === 'number') { const F = like ? like.data.constructor : T.F; return T.make(new F([x]), []); }
    return T.tensor(x);
  };
  T.def = (name, fn) => {
    T[name] = fn;
    T.Tensor.prototype[name] = function (...a) { return fn(this, ...a); };
    return fn;
  };

  /** record a node: backward(g) returns one gradient (a tensor or null) per parent */
  T.record = (out, op, parents, backward, info) => {
    if (!enabled) return out;
    let need = false;
    for (const p of parents) if (p && p.requiresGrad) { need = true; break; }
    if (!need) return out;
    out.requiresGrad = true;
    out.ctx = { op, parents, backward, info: info || null };
    return out;
  };

  T.param = (t, name) => { const p = t instanceof T.Tensor ? t : T.tensor(t); p.requiresGrad = true; p.isParam = true; if (name) p.name = name; return p; };
  const P = T.Tensor.prototype;
  P.named = function (n) { this.name = n; return this; };
  P.requireGrad = function (on = true) { this.requiresGrad = on; return this; };
  P.detach = function () { const t = T.make(this.data, this.shape, this.strides, this.offset); t.name = this.name; return t; };
  P.zeroGrad = function () { this.grad = null; return this; };
  P.clone = function () { return T.make(new this.data.constructor(this.values()), this.shape); };

  /** nodes reachable from root, parents before children */
  T.topo = (root) => {
    const order = [], seen = new Set(), stack = [[root, 0]];
    seen.add(root);
    while (stack.length) {
      const top = stack[stack.length - 1], t = top[0], ps = t.ctx ? t.ctx.parents : [];
      if (top[1] < ps.length) {
        const p = ps[top[1]++];
        if (p && !seen.has(p)) { seen.add(p); stack.push([p, 0]); }
      } else { stack.pop(); order.push(t); }
    }
    return order;
  };

  const accumulate = (t, g) => {
    // a sparse gradient (from a slice) adds itself into the parent's gradient: no full-size temporary per use
    if (g.addTo) { if (!t.grad) t.grad = T.make(new t.data.constructor(t.size), t.shape); g.addTo(t.grad); return; }
    if (!T.sameShape(g.shape, t.shape)) {
      if (g.size !== t.size) throw new Error('gradient of shape [' + g.shape + '] for a tensor of shape [' + t.shape + '] in ' + (t.ctx ? t.ctx.op : 'a leaf'));
      g = T.make(g.values(), t.shape);
    }
    // gradients are always private copies, so accumulating in place never touches another node's gradient
    if (!t.grad) { t.grad = T.make(new t.data.constructor(g.values()), t.shape); return; }
    const a = t.grad.data, b = g.values();
    for (let i = 0; i < a.length; i++) a[i] += b[i];
  };

  /** backpropagate from this tensor; opts.keepGraph keeps the recorded nodes (they are kept by default) */
  P.backward = function (g) {
    if (!this.requiresGrad) throw new Error('backward() on a tensor that does not need a gradient');
    const seed = g === undefined ? T.onesLike(this) : T.as(g, this);
    if (g === undefined && this.size !== 1) throw new Error('backward() without a gradient needs a scalar, not shape [' + this.shape + ']');
    const order = T.topo(this);
    // a fresh pass: intermediate gradients from an earlier pass are cleared, leaf gradients accumulate
    for (const t of order) if (t.ctx) t.grad = null;
    accumulate(this, seed);
    T.noGrad(() => {
      for (let i = order.length - 1; i >= 0; i--) {
        const t = order[i];
        if (!t.ctx || !t.grad) continue;
        const gs = t.ctx.backward(t.grad);
        const ps = t.ctx.parents;
        for (let k = 0; k < ps.length; k++) if (ps[k] && ps[k].requiresGrad && gs[k]) accumulate(ps[k], gs[k]);
      }
    });
    return this;
  };

  // ---- inspection, for the computational-graph pictures ----
  const summary = (t) => {
    if (!t) return null;
    const v = t.values();
    let mn = Infinity, mx = -Infinity, s = 0, s2 = 0;
    for (let i = 0; i < v.length; i++) { const x = v[i]; if (x < mn) mn = x; if (x > mx) mx = x; s += x; s2 += x * x; }
    const n = v.length || 1;
    return { min: mn, max: mx, mean: s / n, norm: Math.sqrt(s2), values: v.length <= 64 ? Array.from(v) : null };
  };
  T.summary = summary;
  const label = (t) => t.name || (t.ctx ? t.ctx.op + '#' + t.id : (t.isParam ? 'param#' : 'input#') + t.id);
  T.label = label;
  T.graph = (root) => {
    const order = T.topo(root), nodes = [], edges = [];
    const ids = new Set(order.map((t) => t.id));
    for (const t of order) {
      const parents = t.ctx ? t.ctx.parents.filter((p) => p && ids.has(p.id)).map((p) => p.id) : [];
      for (const p of parents) edges.push([p, t.id]);
      nodes.push({ id: t.id, name: label(t), op: t.ctx ? t.ctx.op : t.isParam ? 'param' : 'input', shape: t.shape.slice(), leaf: !t.ctx, param: !!t.isParam, value: summary(t), grad: summary(t.grad), parents, info: t.ctx && t.ctx.info, tensor: t });
    }
    return { nodes, edges, find(name) { return nodes.find((n) => n.name === name); } };
  };
  T.traceForward = (root) => T.topo(root).map((t) => {
    const v = summary(t), sh = '[' + t.shape.join('×') + ']';
    if (!t.ctx) return T.step('leaf', `${label(t)} is a ${t.isParam ? 'parameter' : 'input'} of shape ${sh}, values from ${T.fmt(v.min)} to ${T.fmt(v.max)}.`, { id: t.id, name: label(t), shape: t.shape.slice() });
    return T.step('op', `${label(t)} = ${t.ctx.op}(${t.ctx.parents.filter(Boolean).map(label).join(', ')}) has shape ${sh} and mean ${T.fmt(v.mean)}.`, { id: t.id, name: label(t), op: t.ctx.op, shape: t.shape.slice(), parents: t.ctx.parents.filter(Boolean).map((p) => p.id) });
  });
  T.traceBackward = (root) => T.topo(root).reverse().filter((t) => t.grad).map((t) => {
    const g = summary(t.grad);
    return T.step('grad', `The gradient of the loss with respect to ${label(t)} has norm ${T.fmt(g.norm)} (largest entry ${T.fmt(Math.max(Math.abs(g.min), Math.abs(g.max)))}).`, { id: t.id, name: label(t), norm: g.norm });
  });
})(globalThis.Nooral = globalThis.Nooral || {});
