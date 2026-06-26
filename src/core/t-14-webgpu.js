// The WebGPU backend: WGSL compute shaders for matmul (64×64 tiles, 4×4 outputs per thread), conv2d (im2col then
// matmul, or a direct loop), elementwise ops with broadcasting, a row softmax and attention (scores, mask, softmax and
// the weighted sum, all kept on the GPU). Tensors cross to the GPU as GPU tensors {buf, shape}; only results come back.
//   S.T.backends.webgpu: the backend interface of t-13 (init() resolves false where WebGPU is missing)
//   be.gpu: upload(t), read(g), matmul(ga, gb, {transB}), unary, binary, softmax, scaleMask, conv2d, attention, free(g)
(function (S) {
  'use strict';
  const T = S.T, B = T.backends;
  const nav = () => (typeof navigator !== 'undefined' ? navigator : null);
  const ST = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  const MATMUL = `
struct D { M: u32, K: u32, N: u32, sa: u32, sb: u32, sc: u32, aBase: u32, bBase: u32, cBase: u32, tb: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Bm: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: D;
const TS = 64u; const TK = 16u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let r0 = w.y * TS; let c0 = w.x * TS; let bt = w.z;
  let aOff = d.aBase + bt * d.sa; let bOff = d.bBase + bt * d.sb; let cOff = d.cBase + bt * d.sc;
  let tid = l.y * 16u + l.x;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < d.K; k0 += TK) {
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u; let r = e / TK; let k = e % TK; let gr = r0 + r; let gk = k0 + k;
      var v = 0.0;
      if (gr < d.M && gk < d.K) { v = A[aOff + gr * d.K + gk]; }
      As[r * TK + k] = v;
    }
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u; let k = e / TS; let c = e % TS; let gk = k0 + k; let gc = c0 + c;
      var v = 0.0;
      if (gk < d.K && gc < d.N) {
        if (d.tb == 1u) { v = Bm[bOff + gc * d.K + gk]; } else { v = Bm[bOff + gk * d.N + gc]; }
      }
      Bs[k * TS + c] = v;
    }
    workgroupBarrier();
    for (var k = 0u; k < TK; k++) {
      var a: array<f32, 4>; var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[(l.y * 4u + i) * TK + k]; b[i] = Bs[k * TS + l.x + i * 16u]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += a[i] * b[j]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let r = r0 + l.y * 4u + i;
    if (r >= d.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let c = c0 + l.x + j * 16u;
      if (c < d.N) { C[cOff + r * d.N + c] = acc[i * 4u + j]; }
    }
  }
}`;

  const UN = { relu: 'max(x, 0.0)', sigmoid: '1.0 / (1.0 + exp(-x))', tanh: 'tanh(clamp(x, -40.0, 40.0))', exp: 'exp(x)', log: 'log(x)', sqrt: 'sqrt(x)', abs: 'abs(x)', neg: '-x', square: 'x * x',
    gelu: '0.5 * x * (1.0 + tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), -40.0, 40.0)))' };
  const BI = { add: 'a + b', sub: 'a - b', mul: 'a * b', div: 'a / b', pow: 'pow(a, b)', maximum: 'max(a, b)', minimum: 'min(a, b)' };
  const unarySrc = (expr) => `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> n: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= n.x) { return; }
  let x = X[i];
  Y[i] = ${expr};
}`;
  // binary with broadcasting over up to 6 axes: meta = [n, ndim, shape[6], aStride[6], bStride[6]]
  const binarySrc = (expr) => `
@group(0) @binding(0) var<storage, read> Am: array<f32>;
@group(0) @binding(1) var<storage, read> Bm: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<storage, read> m: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 256u;
  if (i >= m[0]) { return; }
  var rem = i; var ia = 0u; var ib = 0u;
  for (var d = 0u; d < m[1]; d++) {
    let ax = m[1] - 1u - d;
    let s = m[2u + ax];
    let k = rem % s; rem = rem / s;
    ia += k * m[8u + ax]; ib += k * m[14u + ax];
  }
  let a = Am[ia]; let b = Bm[ib];
  Y[i] = ${expr};
}`;
  const SOFTMAX = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> n: vec4<u32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let row = w.x + w.y * 65535u;
  if (row >= n.x) { return; }
  let L = n.y; let base = row * L;
  var m = -3.4e38;
  for (var j = l.x; j < L; j += 64u) { m = max(m, X[base + j]); }
  red[l.x] = m; workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) { if (l.x < s) { red[l.x] = max(red[l.x], red[l.x + s]); } workgroupBarrier(); }
  m = red[0]; workgroupBarrier();
  var t = 0.0;
  for (var j = l.x; j < L; j += 64u) { t += exp(X[base + j] - m); }
  red[l.x] = t; workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) { if (l.x < s) { red[l.x] = red[l.x] + red[l.x + s]; } workgroupBarrier(); }
  t = red[0];
  for (var j = l.x; j < L; j += 64u) { Y[base + j] = exp(X[base + j] - m) / t; }
}`;
  // x * scale, and -1e9 where the mask (period n.y elements, repeated) is zero; n.z = 1 when there is a mask
  const SCALEMASK = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Mk: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> n: vec4<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * 65535u * 256u;
  let total = u32(n.x);
  if (i >= total) { return; }
  var v = X[i] * n.w;
  if (n.z > 0.5 && Mk[i % u32(n.y)] == 0.0) { v = -1e9; }
  Y[i] = v;
}`;
  // meta: N C H W Cg kh kw Ho Wo sh sw ph pw dh dw (cols for all channels: [N, C·kh·kw, Ho·Wo])
  const IM2COL = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Cols: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let N = m[0]; let C = m[1]; let H = m[2]; let W = m[3]; let kh = m[5]; let kw = m[6]; let Ho = m[7]; let Wo = m[8];
  let L = Ho * Wo; let K = C * kh * kw;
  let i = g.x + g.y * 65535u * 256u;
  if (i >= N * K * L) { return; }
  let l = i % L; let row = (i / L) % K; let n = i / (L * K);
  let j = row % kw; let ii = (row / kw) % kh; let c = row / (kw * kh);
  let oy = l / Wo; let ox = l % Wo;
  let y = i32(oy * m[9] + ii * m[13]) - i32(m[11]);
  let x = i32(ox * m[10] + j * m[14]) - i32(m[12]);
  var v = 0.0;
  if (y >= 0 && y < i32(H) && x >= 0 && x < i32(W)) { v = X[((n * C + c) * H + u32(y)) * W + u32(x)]; }
  Cols[i] = v;
}`;
  // meta: N C H W O Cg kh kw Ho Wo sh sw ph pw dh dw groups hasBias
  const DIRECT = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> Bs: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> m: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let N = m[0]; let C = m[1]; let H = m[2]; let W = m[3]; let O = m[4]; let Cg = m[5]; let kh = m[6]; let kw = m[7]; let Ho = m[8]; let Wo = m[9];
  let i = g.x + g.y * 65535u * 256u;
  if (i >= N * O * Ho * Wo) { return; }
  let ox = i % Wo; let oy = (i / Wo) % Ho; let o = (i / (Wo * Ho)) % O; let n = i / (Wo * Ho * O);
  let Og = O / m[16]; let c0 = (o / Og) * Cg;
  var acc = 0.0;
  if (m[17] == 1u) { acc = Bs[o]; }
  for (var c = 0u; c < Cg; c++) {
    for (var a = 0u; a < kh; a++) {
      let y = i32(oy * m[10] + a * m[14]) - i32(m[12]);
      if (y < 0 || y >= i32(H)) { continue; }
      for (var b = 0u; b < kw; b++) {
        let x = i32(ox * m[11] + b * m[15]) - i32(m[13]);
        if (x < 0 || x >= i32(W)) { continue; }
        acc += X[((n * C + c0 + c) * H + u32(y)) * W + u32(x)] * Wt[((o * Cg + c) * kh + a) * kw + b];
      }
    }
  }
  Y[i] = acc;
}`;

  const W = (B.webgpu = {
    name: 'webgpu',
    info: '',
    device: null,
    why: '',
    supported: () => !!(nav() && nav().gpu),
    async init() {
      if (W.device) return true;
      if (!W.supported()) { W.why = 'navigator.gpu is missing (no WebGPU in this browser or context)'; return false; }
      const adapter = await nav().gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) { W.why = 'no WebGPU adapter'; return false; }
      W.device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
      const inf = adapter.info || {};
      W.info = 'WebGPU on ' + ([inf.vendor, inf.architecture, inf.description].filter(Boolean).join(' ') || 'an adapter');
      W.device.lost.then((e) => { W.device = null; W.why = 'device lost: ' + (e && e.message); if (T.compute === W) T.compute = B.cpu; });
      W.pipes = new Map();
      return true;
    },
  });
  const pipe = (key, code) => {
    let p = W.pipes.get(key);
    if (!p) { p = W.device.createComputePipeline({ layout: 'auto', compute: { module: W.device.createShaderModule({ code }), entryPoint: 'main' } }); W.pipes.set(key, p); }
    return p;
  };
  const buffer = (bytes, usage) => W.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
  const meta = (arr, Kind = Uint32Array, uniform = false) => {
    const a = new Kind(Math.max(4, Math.ceil(arr.length / 4) * 4)); a.set(arr);
    const b = buffer(a.byteLength, (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST);
    W.device.queue.writeBuffer(b, 0, a);
    return b;
  };
  const tmp = [];
  /** run a pipeline over `threads` (split across y when more than 65535 groups are needed) */
  function run(p, bufs, groups) {
    const bind = W.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const enc = W.device.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(p); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...groups);
    pass.end();
    W.device.queue.submit([enc.finish()]);
    while (tmp.length) tmp.pop().destroy();
  }
  const flat = (n, per = 256) => { const g = Math.ceil(n / per); return g <= 65535 ? [g, 1, 1] : [65535, Math.ceil(g / 65535), 1]; };

  const G = (W.gpu = {});
  const gt = (buf, shape) => ({ buf, shape: shape.slice(), size: T.prod(shape), gpu: true });
  G.empty = (shape) => gt(buffer(4 * Math.max(1, T.prod(shape)), ST()), shape);
  G.upload = (t) => {
    if (t && t.gpu) return t;
    const v = t.values(), f = v instanceof Float32Array ? v : new Float32Array(v), g = G.empty(t.shape);
    W.device.queue.writeBuffer(g.buf, 0, f.buffer, f.byteOffset, f.byteLength);
    g.owned = true;
    return g;
  };
  G.read = async (g) => {
    const bytes = 4 * g.size, rb = W.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = W.device.createCommandEncoder();
    enc.copyBufferToBuffer(g.buf, 0, rb, 0, Math.max(4, bytes));
    W.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0, bytes));
    rb.unmap(); rb.destroy();
    return T.make(T.F === Float64Array ? new Float64Array(out) : out, g.shape);
  };
  G.free = (...gs) => { for (const g of gs) if (g && g.buf) g.buf.destroy(); };
  G.done = () => W.device.queue.onSubmittedWorkDone();

  /** C = A @ B (or A @ Bᵀ), batched along a leading axis; base offsets pick a block (for grouped convolution) */
  G.matmul = (a, b, o = {}) => {
    const M = o.M || a.shape[a.shape.length - 2], K = o.K || a.shape[a.shape.length - 1];
    const N = o.N || (o.transB ? b.shape[b.shape.length - 2] : b.shape[b.shape.length - 1]);
    const ba = o.M ? 1 : T.prod(a.shape.slice(0, -2)), bb = o.M ? 1 : T.prod(b.shape.slice(0, -2)), nb = o.batch || Math.max(ba, bb);
    const out = o.out || G.empty((a.shape.length > 2 ? a.shape.slice(0, -2) : b.shape.slice(0, -2)).concat([M, N]));
    const sa = o.sa !== undefined ? o.sa : ba === 1 ? 0 : M * K, sb = o.sb !== undefined ? o.sb : bb === 1 ? 0 : K * N, sc = o.sc !== undefined ? o.sc : M * N;
    const d = meta([M, K, N, sa, sb, sc, o.aBase || 0, o.bBase || 0, o.cBase || 0, o.transB ? 1 : 0, 0, 0], Uint32Array, true);
    tmp.push(d);
    run(pipe('matmul', MATMUL), [a.buf, b.buf, out.buf, d], [Math.ceil(N / 64), Math.ceil(M / 64), nb]);
    return out;
  };
  G.unary = (op, x) => {
    const y = G.empty(x.shape), n = meta([x.size, 0, 0, 0], Uint32Array, true);
    tmp.push(n);
    run(pipe('u:' + op, unarySrc(UN[op])), [x.buf, y.buf, n], flat(x.size));
    return y;
  };
  G.binary = (op, a, b) => {
    const shape = T.broadcastShapes(a.shape, b.shape);
    if (shape.length > 6) throw new Error('the GPU broadcasts over at most 6 axes');
    const sa = T.broadcastStrides(a.shape, T.stridesOf(a.shape), shape), sb = T.broadcastStrides(b.shape, T.stridesOf(b.shape), shape);
    const pad = (s, f) => s.concat(new Array(6 - s.length).fill(f));
    const y = G.empty(shape), m = meta([T.prod(shape), shape.length].concat(pad(shape, 1), pad(sa, 0), pad(sb, 0)));
    tmp.push(m);
    run(pipe('b:' + op, binarySrc(BI[op])), [a.buf, b.buf, y.buf, m], flat(T.prod(shape)));
    return y;
  };
  G.softmax = (x) => {
    const L = x.shape[x.shape.length - 1], rows = x.size / L, y = G.empty(x.shape), n = meta([rows, L, 0, 0], Uint32Array, true);
    tmp.push(n);
    run(pipe('softmax', SOFTMAX), [x.buf, y.buf, n], rows <= 65535 ? [rows, 1, 1] : [65535, Math.ceil(rows / 65535), 1]);
    return y;
  };
  G.scaleMask = (x, scale, mask) => {
    const y = G.empty(x.shape), mk = mask ? G.upload(mask) : G.empty([1]);
    const n = meta([x.size, mask ? mask.size : 1, mask ? 1 : 0, scale], Float32Array, true);
    tmp.push(n, mk.buf);
    run(pipe('scaleMask', SCALEMASK), [x.buf, mk.buf, y.buf, n], flat(x.size));
    return y;
  };
  const pair = (v, d) => (v === undefined ? [d, d] : Array.isArray(v) ? v : [v, v]);
  G.conv2d = (x, w, b, opts = {}) => {
    const [N, C, H, Wd] = x.shape, [O, Cg, kh, kw] = w.shape, groups = opts.groups || 1;
    const s = pair(opts.stride, 1), p = pair(opts.padding, 0), dl = pair(opts.dilation, 1), [Ho, Wo] = T.convOut(H, Wd, kh, kw, opts), L = Ho * Wo;
    if (opts.mode === 'direct') {
      const y = G.empty([N, O, Ho, Wo]), bb = b || G.empty([1]);
      const m = meta([N, C, H, Wd, O, Cg, kh, kw, Ho, Wo, s[0], s[1], p[0], p[1], dl[0], dl[1], groups, b ? 1 : 0]);
      tmp.push(m); if (!b) tmp.push(bb.buf);
      run(pipe('direct', DIRECT), [x.buf, w.buf, bb.buf, y.buf, m], flat(N * O * L));
      return y;
    }
    const K = C * kh * kw, Kg = Cg * kh * kw, Og = O / groups, cols = G.empty([N, K, L]);
    const m = meta([N, C, H, Wd, Cg, kh, kw, Ho, Wo, s[0], s[1], p[0], p[1], dl[0], dl[1]]);
    tmp.push(m);
    run(pipe('im2col', IM2COL), [x.buf, cols.buf, m], flat(N * K * L));
    let y = G.empty([N, O, Ho, Wo]);
    for (let gi = 0; gi < groups; gi++) {
      G.matmul(w, cols, { M: Og, K: Kg, N: L, batch: N, sa: 0, sb: K * L, sc: O * L, aBase: gi * Og * Kg, bBase: gi * Kg * L, cBase: gi * Og * L, out: y });
    }
    G.free(cols);
    if (b) { const yb = G.binary('add', y, gt(b.buf, [1, O, 1, 1])); G.free(y); y = yb; }
    return y;
  };
  G.attention = (q, k, v, opts = {}) => {
    const d = q.shape[q.shape.length - 1], Tq = q.shape[q.shape.length - 2], Tk = k.shape[k.shape.length - 2];
    let mask = opts.mask ? T.as(opts.mask) : null;
    if (opts.causal) mask = mask ? T.noGrad(() => T.mul(mask, T.causalMask(Tq, Tk))) : T.causalMask(Tq, Tk);
    if (mask && mask.size !== Tq * Tk) mask = T.noGrad(() => T.broadcastTo(mask, [Tq, Tk])); // per-batch masks: the CPU path
    const sc = G.matmul(q, k, { transB: true }), sm = G.scaleMask(sc, opts.scale || 1 / Math.sqrt(d), mask), wts = G.softmax(sm), out = G.matmul(wts, v);
    G.free(sc, sm);
    return { out, weights: wts };
  };

  // ---- the backend interface: plain tensors in, plain tensors out ----
  const wrap = async (inputs, fn) => {
    const gs = inputs.map((t) => (t ? G.upload(t) : null));
    let res;
    try { res = fn(...gs); return await G.read(res); } finally { G.free(...gs.filter((g, i) => g && !(inputs[i] && inputs[i].gpu))); if (res) G.free(res); }
  };
  W.matmul = (a, b) => {
    if (a.ndim < 2 || b.ndim < 2) return B.cpu.matmul(a, b);
    const ba = a.shape.slice(0, -2), bb = b.shape.slice(0, -2);
    if (bb.length === 0 && ba.length) { // [..., M, K] @ [K, N]: all the rows in one product
      const M = T.prod(a.shape.slice(0, -1));
      return wrap([a, b], (x, y) => G.matmul(x, y, { M, K: a.shape[a.ndim - 1], N: b.shape[1], batch: 1, out: G.empty(a.shape.slice(0, -1).concat([b.shape[1]])) }));
    }
    if (T.prod(ba) !== T.prod(bb) && T.prod(ba) !== 1 && T.prod(bb) !== 1) return B.cpu.matmul(a, b);
    return wrap([a, b], (x, y) => G.matmul(x, y));
  };
  W.conv2d = (x, w, b, opts) => wrap([x, w, b], (gx, gw, gb) => G.conv2d(gx, gw, gb, opts));
  W.unary = (op, x) => { if (!UN[op]) throw new Error('unknown unary op ' + op); return wrap([x], (g) => G.unary(op, g)); };
  W.binary = (op, a, b) => { if (!BI[op]) throw new Error('unknown binary op ' + op); return wrap([T.as(a), T.as(b)], (x, y) => G.binary(op, x, y)); };
  W.softmax = (x) => wrap([x], (g) => G.softmax(g));
  W.attention = async (q, k, v, opts = {}) => {
    const gq = G.upload(q), gk = G.upload(k), gv = G.upload(v), r = G.attention(gq, gk, gv, opts);
    try { const out = await G.read(r.out); out.weights = await G.read(r.weights); return out; } finally { G.free(gq, gk, gv, r.out, r.weights); }
  };
})(globalThis.Nooral = globalThis.Nooral || {});
