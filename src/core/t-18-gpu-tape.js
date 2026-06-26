// Autodiff on the GPU. A GPU tensor (S.T.gpu.GT) is a buffer and a shape; the ops below record a graph exactly as
// the CPU tensors do (ctx: op, parents, backward), and backward() walks it, leaving .grad on every node. Nothing is
// read back unless asked: await gpu.read(t) gives a CPU tensor of any value or gradient, at any time.
//   await S.T.gpu.init() -> true where WebGPU works            gpu.ready()
//   gpu.param(cpuTensor) a persistent copy that needs a gradient    gpu.upload(values, shape) a step tensor
//   ops: matmul(a, b, {ta, tb})  reshape  add(a, b, period?)  gelu  layerNorm(x, γ, β, eps)  permute(x, [4 axes])
//        softmaxRows(s, {Tq, Tk, causal, scale})  embedding(W, ids)  crossEntropy(logits [N, V], targets) -> scalar
//        (with .count, and .correct: a GT holding how many rows have the target as their top class)
//   gpu.backward(loss, seed?)   gpu.graph(loss) the recorded nodes   gpu.endStep() (the step's buffers go back to the pool)   await gpu.read(t) / readMany([t])
//   new gpu.Adam(params, {decoupled}) .step({lr, t, betas, eps, weightDecay, clip}) -> the gradient norm, as a GT
// Every dispatch of a step is recorded into one command encoder; its sizes go into 256-byte slots of one uniform
// buffer written just before the submit.
(function (S) {
  'use strict';
  const T = S.T, KS = () => T.gpuKernels;
  const G = (T.gpu = T.gpu || {});
  const SLOTS = 4096;
  let dev = null, pipes = null, ubo = null, uboData = null, slots = 0, enc = null, pass = null, live = [], pool = new Map(), one = null;
  let grad = true;

  G.ready = () => !!(dev && T.backends.webgpu.device === dev);
  G.init = async () => {
    if (G.ready()) return true;
    const W = T.backends && T.backends.webgpu;
    if (!W || !(await W.init())) { G.why = (W && W.why) || 'no WebGPU backend'; return false; }
    dev = W.device; pipes = new Map(); pool = new Map(); live = []; slots = 0; enc = pass = null;
    ubo = dev.createBuffer({ size: SLOTS * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    uboData = new ArrayBuffer(SLOTS * 256);
    one = G.persistent(1); dev.queue.writeBuffer(one, 0, new Float32Array([1]));
    G.dispatches = 0;
    return true;
  };
  G.noGrad = async (fn) => { const old = grad; grad = false; try { return await fn(); } finally { grad = old; } };

  const pipe = (name) => {
    let p = pipes.get(name);
    if (!p) { p = dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: KS()[name] }), entryPoint: 'main' } }); pipes.set(name, p); }
    return p;
  };
  const F32 = (x) => ({ f: x });
  /** record one dispatch: storage buffers, then the uniform values (numbers are u32, F32(x) are f32) */
  function dispatch(name, bufs, u, groups) {
    if (slots >= SLOTS) G.flush();
    const off = slots++ * 256, dv = new DataView(uboData, off, 256);
    for (let i = 0; i < u.length; i++) { const x = u[i]; if (typeof x === 'number') dv.setUint32(4 * i, x >>> 0, true); else dv.setFloat32(4 * i, x.f, true); }
    const p = pipe(name), entries = bufs.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    entries.push({ binding: bufs.length, resource: { buffer: ubo, offset: off, size: 256 } });
    if (!pass) { enc = dev.createCommandEncoder(); pass = enc.beginComputePass(); }
    pass.setPipeline(p);
    pass.setBindGroup(0, dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries }));
    pass.dispatchWorkgroups(groups[0], groups[1] || 1, groups[2] || 1);
    G.dispatches++;
  }
  G.flush = () => {
    if (!pass) return;
    pass.end();
    dev.queue.writeBuffer(ubo, 0, uboData, 0, slots * 256);
    dev.queue.submit([enc.finish()]);
    pass = enc = null; slots = 0;
  };
  const flat = (n) => { const g = Math.ceil(n / 256); return g <= 65535 ? [Math.max(1, g), 1, 1] : [65535, Math.ceil(g / 65535), 1]; };
  const rowsGrid = (r) => (r <= 65535 ? [r, 1, 1] : [65535, Math.ceil(r / 65535), 1]);

  const USE = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const bytesFor = (n) => Math.max(256, Math.ceil((4 * n) / 256) * 256);
  /** a buffer for this step (back to the pool at endStep) */
  function alloc(n) {
    const bytes = bytesFor(n), list = pool.get(bytes), b = list && list.length ? list.pop() : dev.createBuffer({ size: bytes, usage: USE() });
    b.nBytes = bytes; live.push(b);
    return b;
  }
  G.persistent = (n) => dev.createBuffer({ size: bytesFor(n), usage: USE() });
  G.endStep = () => {
    G.flush();
    for (const b of live) { let l = pool.get(b.nBytes); if (!l) pool.set(b.nBytes, (l = [])); l.push(b); }
    live = [];
  };
  G.poolBytes = () => { let s = 0; for (const [k, l] of pool) s += k * l.length; return s; };

  let ID = 0;
  class GT {
    constructor(buf, shape) { this.id = ++ID; this.buf = buf; this.shape = shape.slice(); this.size = T.prod(shape); this.grad = null; this.ctx = null; this.requiresGrad = false; this.name = null; }
    get ndim() { return this.shape.length; }
    named(n) { this.name = n; return this; }
    toString() { return 'GPU tensor' + (this.name ? ' ' + this.name : '') + ' [' + this.shape.join(', ') + ']'; }
  }
  G.GT = GT;
  const make = (shape) => new GT(alloc(Math.max(1, T.prod(shape))), shape);
  const record = (out, op, parents, backward) => {
    if (!grad || !parents.some((p) => p && p.requiresGrad)) return out;
    out.requiresGrad = true; out.ctx = { op, parents, backward };
    return out;
  };
  G.upload = (values, shape) => {
    const f = values instanceof Float32Array ? values : Float32Array.from(values), t = make(shape || [f.length]);
    dev.queue.writeBuffer(t.buf, 0, f.buffer, f.byteOffset, f.byteLength);
    return t;
  };
  /** a persistent GPU copy of a parameter (its gradient is recorded) */
  G.param = (p) => {
    const t = new GT(G.persistent(p.size), p.shape);
    G.write(t, p);
    t.requiresGrad = true; t.isParam = true; t.name = p.name; t.cpu = p;
    return t;
  };
  G.write = (t, cpu) => { const v = cpu.values(), f = v instanceof Float32Array ? v : Float32Array.from(v); dev.queue.writeBuffer(t.buf, 0, f.buffer, f.byteOffset, f.byteLength); };

  // ---- the ops ----
  const last = (t, k = 1) => t.shape[t.shape.length - k];
  G.reshape = (x, shape) => {
    if (T.prod(shape) !== x.size) throw new Error('gpu reshape: [' + x.shape + '] into [' + shape + ']');
    const out = new GT(x.buf, shape);
    return record(out, 'reshape', [x], (g) => [G.reshape(g, x.shape)]);
  };
  G.matmul = (a, b, o = {}) => {
    const ta = o.ta ? 1 : 0, tb = o.tb ? 1 : 0;
    const M = ta ? last(a) : last(a, 2), Ka = ta ? last(a, 2) : last(a), N = tb ? last(b, 2) : last(b), Kb = tb ? last(b) : last(b, 2);
    if (Ka !== Kb) throw new Error('gpu matmul: inner sizes ' + Ka + ' and ' + Kb + ' differ');
    const nb = T.prod(a.shape.slice(0, -2)), bb = T.prod(b.shape.slice(0, -2));
    if (bb !== nb) throw new Error('gpu matmul: batches ' + nb + ' and ' + bb + ' differ (flatten a shared weight first)');
    const out = make(a.shape.slice(0, -2).concat([M, N]));
    dispatch('mm', [a.buf, b.buf, out.buf], [M, Ka, N, nb > 1 ? M * Ka : 0, nb > 1 ? Ka * N : 0, M * N, ta, tb, 0, 0, 0, 0], [Math.ceil(N / 64), Math.ceil(M / 64), nb]);
    return record(out, 'matmul', [a, b], (g) => {
      let da = null, db = null;
      if (!ta && !tb) { if (a.requiresGrad) da = G.matmul(g, b, { tb: 1 }); if (b.requiresGrad) db = G.matmul(a, g, { ta: 1 }); }
      else if (tb && !ta) { if (a.requiresGrad) da = G.matmul(g, b); if (b.requiresGrad) db = G.matmul(g, a, { ta: 1 }); }
      else if (ta && !tb) { if (a.requiresGrad) da = G.matmul(b, g, { tb: 1 }); if (b.requiresGrad) db = G.matmul(a, g); }
      else throw new Error('gpu matmul: the backward of Aᵀ Bᵀ is not needed here');
      return [da, db];
    });
  };
  const ew = (op, a, b, p, c = 0, c2 = 0) => { const out = make(a.shape); dispatch('ew', [a.buf, b.buf, out.buf], [a.size, p, op, 0, F32(c), F32(c2), F32(0), F32(0)], flat(a.size)); return out; };
  const colsum = (g, rows, p, outN, shape) => { const out = make(shape); dispatch('colsum', [g.buf, out.buf], [rows, p, outN, 0], flat(outN)); return out; };
  G.addRaw = (a, b) => ew(0, a, b, b.size);
  /** a + b where b (or its first `period` values) repeats along a; b's gradient sums the repeats */
  G.add = (a, b, period) => {
    const p = period || b.size;
    if (a.size % p) throw new Error('gpu add: ' + p + ' does not divide ' + a.size);
    const out = ew(0, a, b, p);
    return record(out, 'add', [a, b], (g) => [a.requiresGrad ? g : null, b.requiresGrad ? (p === b.size && p === a.size ? g : colsum(g, a.size / p, p, b.size, b.shape)) : null]);
  };
  G.scale = (a, c) => { const out = ew(6, a, a, a.size, c); return record(out, 'scale', [a], (g) => [G.scale(g, c)]); };
  G.gelu = (x) => {
    const out = ew(4, x, x, x.size);
    return record(out, 'gelu', [x], (g) => [ew(5, x, g, x.size)]);
  };
  G.layerNorm = (x, gm, bt, eps = 1e-5) => {
    const D = last(x), rows = x.size / D, out = make(x.shape), st = new GT(alloc(2 * rows), [rows, 2]);
    dispatch('lnf', [x.buf, gm.buf, bt.buf, out.buf, st.buf], [rows, D, F32(eps), 0], rowsGrid(rows));
    out.stats = st;
    return record(out, 'layerNorm', [x, gm, bt], (g) => {
      const dx = make(x.shape), dg = make(gm.shape), db = make(bt.shape);
      dispatch('lnb', [x.buf, gm.buf, g.buf, st.buf, dx.buf], [rows, D, F32(eps), 0], rowsGrid(rows));
      dispatch('lnp', [x.buf, g.buf, st.buf, dg.buf, db.buf], [rows, D, F32(eps), 0], [Math.ceil(D / 64), 1, 1]);
      return [dx, dg, db];
    });
  };
  G.permute = (x, perm) => {
    if (x.ndim !== 4) throw new Error('gpu permute works on four axes');
    const st = T.stridesOf(x.shape), shape = perm.map((i) => x.shape[i]), out = make(shape);
    dispatch('perm', [x.buf, out.buf], [x.size, ...shape, ...perm.map((i) => st[i]), 0, 0, 0], flat(x.size));
    const inv = new Array(4); perm.forEach((d, i) => { inv[d] = i; });
    return record(out, 'permute', [x], (g) => [G.permute(g, inv)]);
  };
  /** softmax over the last axis of scores [.., Tq, Tk] times scale, keys after the query masked out when causal */
  G.softmaxRows = (s, o) => {
    const Tk = last(s), Tq = o.Tq || last(s, 2), rows = s.size / Tk, out = make(s.shape), u = [rows, Tq, Tk, o.causal ? 1 : 0, F32(o.scale || 1), F32(0), F32(0), F32(0)];
    dispatch('smf', [s.buf, out.buf], u, rowsGrid(rows));
    return record(out, 'softmax', [s], (g) => { const dx = make(s.shape); dispatch('smb', [out.buf, g.buf, dx.buf], u, rowsGrid(rows)); return [dx]; });
  };
  /** rows of W picked by integer ids (a plain array); the gradient adds each row's uses, in token order */
  G.embedding = (W, ids, shape) => {
    const n = ids.length, d = last(W), V = W.shape[0], idt = G.upload(Float32Array.from(ids), [n]), out = make((shape || [n]).concat([d]));
    dispatch('gather', [idt.buf, W.buf, out.buf], [n, d, 0, 0], flat(n * d));
    return record(out, 'embedding', [W], (g) => {
      const starts = new Uint32Array(V + 1), pos = new Uint32Array(Math.max(1, n));
      for (const v of ids) starts[v + 1]++;
      for (let v = 0; v < V; v++) starts[v + 1] += starts[v];
      const at = starts.slice(0, V);
      for (let t = 0; t < n; t++) pos[at[ids[t]]++] = t;
      const sb = new GT(alloc(V + 1), [V + 1]), pb = new GT(alloc(n), [n]), dw = make(W.shape);
      dev.queue.writeBuffer(sb.buf, 0, starts); dev.queue.writeBuffer(pb.buf, 0, pos);
      dispatch('scatter', [sb.buf, pb.buf, g.buf, dw.buf], [V, d, 0, 0], flat(V * d));
      return [dw];
    });
  };
  /** mean cross-entropy of logits [N, V] against class ids (−1 is ignored), as a scalar GT */
  G.crossEntropy = (z, targets) => {
    const V = last(z), N = z.size / V, tv = Float32Array.from(targets);
    let count = 0; for (let i = 0; i < N; i++) if (tv[i] >= 0) count++;
    const inv = 1 / Math.max(1, count), tg = G.upload(tv, [N]), rowLoss = make([N]), dz = make(z.shape), cor = make([N]), loss = make([]), right = make([]);
    dispatch('ce', [z.buf, tg.buf, rowLoss.buf, dz.buf, cor.buf], [N, V, F32(inv), 0], rowsGrid(N));
    dispatch('reduce', [rowLoss.buf, loss.buf], [N, 0, 0, 0, F32(inv), F32(0), F32(0), F32(0)], [1, 1, 1]);
    dispatch('reduce', [cor.buf, right.buf], [N, 0, 0, 0, F32(1), F32(0), F32(0), F32(0)], [1, 1, 1]);
    loss.count = count; loss.correct = right;
    return record(loss, 'crossEntropy', [z], (g) => [ew(2, dz, g, 1)]);
  };

  // ---- backward ----
  const topo = (root) => {
    const order = [], seen = new Set([root]), stack = [[root, 0]];
    while (stack.length) {
      const top = stack[stack.length - 1], t = top[0], ps = t.ctx ? t.ctx.parents : [];
      if (top[1] < ps.length) { const p = ps[top[1]++]; if (p && !seen.has(p)) { seen.add(p); stack.push([p, 0]); } } else { stack.pop(); order.push(t); }
    }
    return order;
  };
  /** backpropagate from root (a scalar, or any tensor with seed: its upstream gradient as a GT) */
  G.backward = (root, seed) => {
    const order = topo(root);
    for (const t of order) if (t.ctx) t.grad = null;
    root.grad = seed || new GT(one, root.shape);
    const old = grad; grad = false;
    try {
      for (let i = order.length - 1; i >= 0; i--) {
        const t = order[i];
        if (!t.ctx || !t.grad) continue;
        const gs = t.ctx.backward(t.grad), ps = t.ctx.parents;
        for (let k = 0; k < ps.length; k++) {
          const p = ps[k], gk = gs[k];
          if (!p || !p.requiresGrad || !gk) continue;
          const gg = gk.size === p.size && !T.sameShape(gk.shape, p.shape) ? new GT(gk.buf, p.shape) : gk;
          p.grad = p.grad ? G.addRaw(p.grad, gg) : gg;
        }
      }
    } finally { grad = old; }
    return order;
  };

  /** the recorded graph under root, parents first: [{id, name, op, shape, parents, hasGrad}] (values: await gpu.read) */
  G.graph = (root) => topo(root).map((t) => ({ id: t.id, name: t.name, op: t.ctx ? t.ctx.op : t.isParam ? 'param' : 'input', shape: t.shape.slice(), parents: t.ctx ? t.ctx.parents.filter(Boolean).map((p) => p.id) : [], hasGrad: !!t.grad, tensor: t }));

  // ---- reading back ----
  G.readMany = async (ts) => {
    G.flush();
    const total = ts.reduce((s, t) => s + t.size, 0), rb = dev.createBuffer({ size: Math.max(16, Math.ceil((4 * total) / 16) * 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = dev.createCommandEncoder();
    let at = 0;
    for (const t of ts) { if (t.size) e.copyBufferToBuffer(t.buf, 0, rb, 4 * at, Math.ceil(t.size) * 4); at += t.size; }
    dev.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    at = 0;
    return ts.map((t) => { const v = all.slice(at, at + t.size); at += t.size; return T.make(v, t.shape); });
  };
  G.read = async (t) => (await G.readMany([t]))[0];

  // ---- the optimiser: Adam / AdamW with the global norm clip, all on the GPU ----
  G.Adam = class Adam {
    constructor(params, o = {}) {
      this.params = params; this.decoupled = !!o.decoupled;
      this.m = params.map((p) => G.persistent(p.size)); this.v = params.map((p) => G.persistent(p.size));
      for (let i = 0; i < params.length; i++) { const z = new Float32Array(params[i].size); dev.queue.writeBuffer(this.m[i], 0, z); dev.queue.writeBuffer(this.v[i], 0, z); }
      this.sq = G.persistent(params.length); this.norm = G.persistent(4);
    }
    /** set the moments (from a CPU optimiser's state, so a run can move between the CPU and the GPU) */
    setState(i, m, v) { if (m) dev.queue.writeBuffer(this.m[i], 0, Float32Array.from(m)); if (v) dev.queue.writeBuffer(this.v[i], 0, Float32Array.from(v)); }
    /** the gradient norm is computed on the GPU (clipping needs it there); returns it as a GT of one value */
    step(o) {
      const [b1, b2] = o.betas || [0.9, 0.999], c1 = 1 - Math.pow(b1, o.t), c2 = 1 - Math.pow(b2, o.t), clip = o.clip || 0;
      const ps = this.params;
      ps.forEach((p, i) => dispatch('reduce', [p.grad ? p.grad.buf : one, this.sq], [p.grad ? p.size : 0, 1, i, 0, F32(1), F32(0), F32(0), F32(0)], [1, 1, 1]));
      dispatch('reduce', [this.sq, this.norm], [ps.length, 0, 0, 1, F32(1), F32(0), F32(0), F32(0)], [1, 1, 1]);
      ps.forEach((p, i) => {
        if (!p.grad) return;
        dispatch('adam', [p.buf, p.grad.buf, this.m[i], this.v[i], this.norm], [p.size, this.decoupled ? 1 : 0, clip ? 1 : 0, 0, F32(o.lr), F32(b1), F32(b2), F32(o.eps || 1e-8), F32(o.weightDecay || 0), F32(c1), F32(c2), F32(clip)], flat(p.size));
      });
      return new GT(this.norm, []);
    }
    free() { for (const b of this.m.concat(this.v, [this.sq, this.norm])) b.destroy(); }
  };
  G.F32 = F32;
})(globalThis.Nooral = globalThis.Nooral || {});
