// Compute backends. The tensors and their autodiff graph run on the CPU (synchronous, every value inspectable); a
// backend speeds up the heavy forward kernels: matmul, conv2d, elementwise ops, softmax and attention. Training on
// the GPU, with its own recorded graph and values read back on request, is S.T.gpu (t-17, t-18) and DL.gpuSession.
// Every backend has the same asynchronous interface on plain tensors, and must agree with the CPU within a tolerance.
//   await S.T.backends.select({prefer: 'webgpu' | 'cpu'}) -> the chosen backend (falls back to the CPU)
//   S.T.compute                     the backend in use (the CPU until select() picks another)
//   be.name, be.info                'cpu' | 'webgpu', a description of the device
//   await be.matmul(a, b)  conv2d(x, w, b, {stride, padding, dilation, groups, mode: 'im2col' | 'direct'})
//   await be.unary(op, x)  binary(op, a, b)  softmax(x)  attention(q, k, v, {mask, causal, scale})
//   S.T.backends.compare(be, cases?) -> [{op, maxErr, ok}]   checks a backend against the CPU
(function (S) {
  'use strict';
  const T = S.T;
  const B = (T.backends = T.backends || {});
  const UNARY = ['relu', 'sigmoid', 'tanh', 'gelu', 'exp', 'log', 'sqrt', 'abs', 'neg', 'square'];
  const BINARY = ['add', 'sub', 'mul', 'div', 'pow', 'maximum', 'minimum'];
  B.UNARY = UNARY;
  B.BINARY = BINARY;
  const cpuOnly = (fn) => T.noGrad(fn);

  B.cpu = {
    name: 'cpu',
    info: 'JavaScript on the main thread or a worker (the reference)',
    async init() { return true; },
    async matmul(a, b) { return cpuOnly(() => T.matmul(a, b)); },
    async conv2d(x, w, b, opts) { return cpuOnly(() => T.conv2d(x, w, b, opts)); },
    async unary(op, x) { if (!UNARY.includes(op)) throw new Error('unknown unary op ' + op); return cpuOnly(() => T[op](x)); },
    async binary(op, a, b) { if (!BINARY.includes(op)) throw new Error('unknown binary op ' + op); return cpuOnly(() => T[op](a, b)); },
    async softmax(x) { return cpuOnly(() => T.softmax(x, -1)); },
    async attention(q, k, v, opts) { return cpuOnly(() => { const o = T.attention(q, k, v, opts); const r = o.detach(); r.weights = o.attn.weights.detach(); return r; }); },
  };
  T.compute = B.cpu;

  B.available = () => ['cpu'].concat(B.webgpu && B.webgpu.supported() ? ['webgpu'] : []);
  B.select = async (opts = {}) => {
    const order = opts.prefer === 'cpu' ? ['cpu'] : ['webgpu', 'cpu'];
    B.log = [];
    for (const name of order) {
      const be = B[name];
      if (!be) { B.log.push(name + ': not built in'); continue; }
      try {
        if (await be.init()) { T.compute = be; B.log.push(name + ': ready (' + be.info + ')'); return be; }
        B.log.push(name + ': ' + (be.why || 'not available here'));
      } catch (e) { B.log.push(name + ': failed, ' + (e.message || e)); }
    }
    T.compute = B.cpu;
    return B.cpu;
  };

  const maxErr = (a, b) => {
    const x = a.values(), y = b.values();
    if (x.length !== y.length) return Infinity;
    let m = 0;
    for (let i = 0; i < x.length; i++) { const d = Math.abs(x[i] - y[i]) / Math.max(1, Math.abs(y[i])); if (!(d <= m)) m = d; }
    return m;
  };
  B.maxErr = maxErr;
  /** the standard comparison set: each case is [label, () => [cpuPromise, bePromise]] */
  B.compare = async (be, opts = {}) => {
    const g = T.rng(opts.seed || 7), r = (s) => T.normal(s, 0, 1, g), tol = opts.tol || 1e-4, out = [];
    const cases = [
      ['matmul 37×53 @ 53×29', (k) => k.matmul(A1, B1)],
      ['matmul batched [3,17,40] @ [3,40,9]', (k) => k.matmul(A2, B2)],
      ['matmul broadcast [4,8,16] @ [16,12]', (k) => k.matmul(A3, B3)],
      ['conv2d im2col 3×3 pad 1', (k) => k.conv2d(X1, W1, b1, { padding: 1, mode: 'im2col' })],
      ['conv2d direct 3×3 pad 1', (k) => k.conv2d(X1, W1, b1, { padding: 1, mode: 'direct' })],
      ['conv2d stride 2, dilation 2', (k) => k.conv2d(X1, W1, b1, { stride: 2, dilation: 2, padding: 2 })],
      ['conv2d groups 2', (k) => k.conv2d(X2, W2, null, { padding: 1, groups: 2 })],
      ['conv2d direct groups 2, stride 2', (k) => k.conv2d(X2, W2, null, { padding: 1, groups: 2, stride: 2, mode: 'direct' })],
      ['softmax [6, 33]', (k) => k.softmax(A1.slice(null, [0, 33]).contiguous())],
      ['attention [2,3,7,8] causal', (k) => k.attention(Q, K, V, { causal: true })],
      ['attention with a padding mask', (k) => k.attention(Q, K, V, { mask: M })],
    ];
    const A1 = r([37, 53]), B1 = r([53, 29]), A2 = r([3, 17, 40]), B2 = r([3, 40, 9]), A3 = r([4, 8, 16]), B3 = r([16, 12]);
    const X1 = r([2, 3, 11, 9]), W1 = r([5, 3, 3, 3]), b1 = r([5]), X2 = r([2, 4, 8, 8]), W2 = r([6, 2, 3, 3]);
    const Q = r([2, 3, 7, 8]), K = r([2, 3, 7, 8]), V = r([2, 3, 7, 8]), M = T.ones([7, 7]);
    for (let j = 5; j < 7; j++) for (let i = 0; i < 7; i++) M.data[i * 7 + j] = 0;
    for (const op of UNARY) { const x = op === 'log' || op === 'sqrt' ? T.uniform([5, 41], 0.1, 3, g) : r([5, 41]); cases.push(['unary ' + op, (k) => k.unary(op, x)]); }
    for (const op of BINARY) {
      const a = op === 'pow' ? T.uniform([4, 6, 5], 0.2, 2, g) : r([4, 6, 5]), b = op === 'div' ? T.uniform([6, 1], 0.5, 2, g) : r([6, 1]);
      cases.push(['binary ' + op + ' broadcast [4,6,5]·[6,1]', (k) => k.binary(op, a, b)]);
    }
    for (const [label, run] of cases) {
      try {
        const [c, d] = [await run(B.cpu), await run(be)];
        const e = maxErr(d, c);
        out.push({ op: label, maxErr: e, ok: e <= tol && T.sameShape(c.shape, d.shape), shape: d.shape.slice() });
      } catch (e) { out.push({ op: label, maxErr: Infinity, ok: false, error: String(e.message || e) }); }
    }
    return out;
  };

  /** time a function: runs it until `ms` have passed (at least `min` times), returns ms per run */
  B.time = async (fn, opts = {}) => {
    await fn();
    const t0 = performance.now(), budget = opts.ms || 300, min = opts.min || 3;
    let n = 0;
    while (n < min || performance.now() - t0 < budget) { await fn(); n++; }
    return (performance.now() - t0) / n;
  };
  /** matmul and conv speed of a backend: GFLOP/s for square products and a small CNN layer */
  B.bench = async (be, opts = {}) => {
    const g = T.rng(3), rows = [];
    for (const n of opts.sizes || [128, 256, 512]) {
      const a = T.normal([n, n], 0, 1, g), b = T.normal([n, n], 0, 1, g);
      const ms = await B.time(() => be.matmul(a, b), opts);
      rows.push({ op: 'matmul ' + n + '×' + n, ms, gflops: (2 * n * n * n) / ms / 1e6 });
    }
    const conv = opts.conv || [[8, 16, 32, 32, 32, 3]];
    for (const [N, C, H, W, O, k] of conv) {
      const x = T.normal([N, C, H, W], 0, 1, g), w = T.normal([O, C, k, k], 0, 1, g);
      for (const mode of be.name === 'cpu' ? ['im2col'] : ['im2col', 'direct']) {
        const ms = await B.time(() => be.conv2d(x, w, null, { padding: (k - 1) / 2, mode }), opts);
        rows.push({ op: `conv2d ${mode} [${N},${C},${H},${W}]→${O} ${k}×${k}`, ms, gflops: (2 * N * O * H * W * C * k * k) / ms / 1e6 });
      }
    }
    return rows;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
