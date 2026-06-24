// Matrix products. The kernels work on row-major typed arrays; for float32 data a WebAssembly SIMD kernel
// (t-05-wasm.js) packs both sides into panels and runs 4×8 register blocks, about five times the plain JS kernel.
// Float64 data (gradient checks) and browsers without WebAssembly SIMD use the JS kernel below.
//   matmul(a, b)   [..., M, K] @ [..., K, N] with broadcast batch axes; a 1-D side is treated as a row / column
//   bmm(a, b)      the same, for [B, M, K] @ [B, K, N]
//   linear(x, W, b?)  x @ W + b, with W of shape [in, out]
//   S.T.kernels.gemm(A, B, M, K, N, out?, aOff, bOff, oOff, ta, tb) -> out = op(A) @ op(B), where ta means A is
//      stored transposed ([K, M]) and tb means B is stored transposed ([N, K]); offsets are in elements
(function (S) {
  'use strict';
  const T = S.T;
  const F = (t) => t.data.constructor;

  // two rows of A and four of B at a time, accumulated in float64 rows, in column blocks of 256
  let r0 = new Float64Array(256), r1 = new Float64Array(256);
  function gemmRows(A, B, M, K, N, out, aOff, bOff, oOff) {
    if (r0.length < N) { r0 = new Float64Array(N); r1 = new Float64Array(N); }
    const BS = 256;
    let i = 0;
    for (; i + 1 < M; i += 2) {
      const a0 = aOff + i * K, a1 = a0 + K, o0 = oOff + i * N, o1 = o0 + N;
      for (let j0 = 0; j0 < N; j0 += BS) {
        const j1 = j0 + BS < N ? j0 + BS : N;
        for (let j = j0; j < j1; j++) { r0[j] = 0; r1[j] = 0; }
        let k = 0;
        for (; k + 3 < K; k += 4) {
          const p0 = A[a0 + k], p1 = A[a0 + k + 1], p2 = A[a0 + k + 2], p3 = A[a0 + k + 3];
          const q0 = A[a1 + k], q1 = A[a1 + k + 1], q2 = A[a1 + k + 2], q3 = A[a1 + k + 3];
          const b0 = bOff + k * N, b1 = b0 + N, b2 = b1 + N, b3 = b2 + N;
          for (let j = j0; j < j1; j++) {
            const x0 = B[b0 + j], x1 = B[b1 + j], x2 = B[b2 + j], x3 = B[b3 + j];
            r0[j] += p0 * x0 + p1 * x1 + p2 * x2 + p3 * x3;
            r1[j] += q0 * x0 + q1 * x1 + q2 * x2 + q3 * x3;
          }
        }
        for (; k < K; k++) {
          const p = A[a0 + k], q = A[a1 + k], b0 = bOff + k * N;
          for (let j = j0; j < j1; j++) { const x = B[b0 + j]; r0[j] += p * x; r1[j] += q * x; }
        }
        for (let j = j0; j < j1; j++) { out[o0 + j] = r0[j]; out[o1 + j] = r1[j]; }
      }
    }
    for (; i < M; i++) {
      const a0 = aOff + i * K, o0 = oOff + i * N;
      for (let j = 0; j < N; j++) r0[j] = 0;
      for (let k = 0; k < K; k++) { const p = A[a0 + k], b0 = bOff + k * N; for (let j = 0; j < N; j++) r0[j] += p * B[b0 + j]; }
      for (let j = 0; j < N; j++) out[o0 + j] = r0[j];
    }
    return out;
  }
  /** a transposed copy of the rows × cols block at off */
  const transposed = (X, off, rows, cols) => {
    const o = new X.constructor(rows * cols);
    for (let i = 0; i < rows; i++) { const b = off + i * cols; for (let j = 0; j < cols; j++) o[j * rows + i] = X[b + j]; }
    return o;
  };
  function gemmJS(A, B, M, K, N, out, aOff = 0, bOff = 0, oOff = 0, ta = false, tb = false) {
    out = out || new (A.constructor)(M * N);
    if (ta) { A = transposed(A, aOff, K, M); aOff = 0; }
    if (tb) { B = transposed(B, bOff, N, K); bOff = 0; }
    return gemmRows(A, B, M, K, N, out, aOff, bOff, oOff);
  }
  T.kernels = T.kernels || {};
  T.kernels.gemmCPU = gemmJS;
  T.kernels.gemmJS = gemmJS;
  T.kernels.gemm = gemmJS;

  const contig = (t) => (t.isContiguous() ? t.data : t.values());
  const cast = (v, C) => (v.constructor === C ? v : new C(v));
  /**
   * op(a) @ op(b) for tensors whose last two axes are the matrices (ta / tb: that side's last two axes are swapped
   * in storage); batch axes broadcast. Returns the product as a new contiguous tensor.
   */
  function product(a, b, ta = false, tb = false) {
    const an = a.ndim, bn = b.ndim;
    const M = ta ? a.shape[an - 1] : a.shape[an - 2], K = ta ? a.shape[an - 2] : a.shape[an - 1];
    const Kb = tb ? b.shape[bn - 1] : b.shape[bn - 2], N = tb ? b.shape[bn - 2] : b.shape[bn - 1];
    if (Kb !== K) throw new Error('matmul: [' + a.shape + '] @ [' + b.shape + ']: inner sizes ' + K + ' and ' + Kb + ' differ');
    const C = a.data instanceof Float64Array || b.data instanceof Float64Array ? Float64Array : F(a), gemm = T.kernels.gemm;
    if (bn === 2 && !ta) {
      // [..., M, K] @ [K, N]: one product of all the rows
      const rows = a.size / K, out = new C(rows * N);
      gemm(cast(contig(a), C), cast(contig(b), C), rows, K, N, out, 0, 0, 0, false, tb);
      return T.make(out, a.shape.slice(0, -1).concat([N]));
    }
    const batch = T.broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2)), nb = T.prod(batch);
    const sameA = T.sameShape(a.shape.slice(0, -2), batch), sameB = T.sameShape(b.shape.slice(0, -2), batch);
    const A = cast(sameA ? contig(a) : T.expandValues(a, batch.concat(a.shape.slice(-2))), C);
    const B = cast(sameB ? contig(b) : T.expandValues(b, batch.concat(b.shape.slice(-2))), C);
    const out = new C(nb * M * N);
    for (let i = 0; i < nb; i++) gemm(A, B, M, K, N, out, i * M * K, i * K * N, i * M * N, ta, tb);
    return T.make(out, batch.concat([M, N]));
  }
  T.product = product;

  T.def('matmul', (a, b) => {
    a = T.as(a); b = T.as(b);
    const a1 = a.ndim === 1, b1 = b.ndim === 1;
    const A = a1 ? T.make(a.data, [1, a.size], [a.size * (a.strides[0] || 1), a.strides[0] || 1], a.offset) : a;
    const B = b1 ? T.make(b.data, [b.size, 1], [b.strides[0] || 1, 1], b.offset) : b;
    let out = product(A, B);
    const full = out.shape.slice();
    if (a1 || b1) {
      const s = out.shape.slice();
      if (b1) s.pop();
      if (a1) s.splice(s.length - (b1 ? 1 : 2), 1);
      out = T.make(out.data, s);
    }
    return T.record(out, 'matmul', [a, b], (g) => {
      const G = T.make(g.isContiguous() ? g.data : g.values(), full);
      let ga = null, gb = null;
      // dA = G Bᵀ, dB = Aᵀ G: the kernels read the transposes in place
      if (a.requiresGrad) ga = T.make(T.sumTo(product(G, B, false, true), A.shape).data, a.shape);
      if (b.requiresGrad) {
        if (B.ndim === 2) {
          const K = A.shape[A.ndim - 1], N = B.shape[1], rows = A.size / K;
          gb = product(T.make(contig(A), [rows, K]), T.make(G.data, [rows, N]), true, false);
        } else gb = T.sumTo(product(A, G, true, false), B.shape);
        gb = T.make(gb.data, b.shape);
      }
      return [ga, gb];
    }, { a: a.shape.slice(), b: b.shape.slice() });
  });
  T.bmm = T.matmul;
  T.linear = (x, W, b) => (b ? T.add(T.matmul(x, W), b) : T.matmul(x, W));
})(globalThis.Nooral = globalThis.Nooral || {});
