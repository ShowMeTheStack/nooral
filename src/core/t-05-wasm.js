// WebAssembly SIMD matrix kernels, assembled here from their instructions (no build step, no binary blob). Both
// pack their inputs into panels first, which also reads transposed storage for free.
//   exact  the default: float64 sums in 2-lane registers, in exactly the order of the JS kernel (rows in pairs, k in
//          groups of four summed first, the rest one by one; an odd last row one by one), so every result is the
//          same to the last bit as the JS kernel's, two to three times faster. B is packed in panels of 4 columns.
//   fast   float32 sums in 4-lane registers, a 4 × 8 block of C held in eight registers while k runs; A is packed in
//          panels of 4 rows and B of 8 columns, so a B panel stays in the first-level cache. Four to six times the JS
//          kernel; the sums round differently (about 1e-6 relative), still the same on every run.
//   S.T.kernels.gemm picks: float32 or float64 arrays go to exact, or to fast inside S.T.fast(fn) when all are float32.
//   S.T.kernels.wasm is false where WebAssembly SIMD does not validate (the JS kernel is used).
(function (S) {
  'use strict';
  const T = S.T, K_ = T.kernels;
  K_.mode = 'exact';
  /** run fn with the fast float32 kernel for float32 products (results differ from the exact kernel in rounding) */
  T.fast = (fn) => { const old = K_.mode; K_.mode = 'fast'; try { return fn(); } finally { K_.mode = old; } };
  if (typeof WebAssembly === 'undefined') { K_.wasm = false; return; }

  const uleb = (n) => { const o = []; do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; o.push(b); } while (n); return o; };
  const sleb = (n) => { const o = []; for (;;) { const b = n & 0x7f; n >>= 7; if ((n === 0 && !(b & 0x40)) || (n === -1 && b & 0x40)) { o.push(b); return o; } o.push(b | 0x80); } };
  const str = (s) => [...uleb(s.length), ...[...s].map((c) => c.charCodeAt(0))];
  const sec = (id, body) => [id, ...uleb(body.length), ...body];
  const vec = (items) => [...uleb(items.length), ...items.flat()];
  const I32 = 0x7f, V128 = 0x7b;
  const get = (i) => [0x20, ...uleb(i)], set = (i) => [0x21, ...uleb(i)], tee = (i) => [0x22, ...uleb(i)], i32 = (n) => [0x41, ...sleb(n)];
  const ADD = [0x6a], SUB = [0x6b], MUL = [0x6c], AND = [0x71], EQ = [0x46], LTU = [0x49], GTU = [0x4b], LEU = [0x4d], GEU = [0x4f];
  const simd = (op) => [0xfd, ...uleb(op)];
  const vload = (off) => [...simd(0), 4, ...uleb(off)], vstore = (off) => [...simd(11), 4, ...uleb(off)];
  const fload = (off) => [0x2a, 2, ...uleb(off)], dload = (off) => [0x2b, 3, ...uleb(off)];
  const FSPLAT = simd(19), FADD = simd(228), FMUL = simd(230), DSPLAT = simd(20), DADD = simd(240), DMUL = simd(242), VZERO = [...simd(12), ...new Array(16).fill(0)];
  const loop = (b) => [0x03, 0x40, ...b, 0x0b], block = (b) => [0x02, 0x40, ...b, 0x0b], IF = (b) => [0x04, 0x40, ...b, 0x0b];
  const brIf = (d) => [0x0d, d], br = (d) => [0x0c, d];

  // ---- fast(pa, pb, pc, Mp, K, Np): C [Mp × Np] (float32) = Ap · Bp ----
  // locals: 6 jb, 7 ib, 8 x, 9 y, 10 k, 11 o (i32); 12..19 the accumulators, 20 b0, 21 b1, 22 a (v128)
  const fast = (() => {
    const jb = 6, ib = 7, x = 8, y = 9, k = 10, o = 11, acc = 12, b0 = 20, b1 = 21, a = 22;
    const row = (r) => [...get(x), ...fload(4 * r), ...FSPLAT, ...tee(a), ...get(b0), ...FMUL, ...get(acc + 2 * r), ...FADD, ...set(acc + 2 * r),
      ...get(a), ...get(b1), ...FMUL, ...get(acc + 2 * r + 1), ...FADD, ...set(acc + 2 * r + 1)];
    const blk = [
      ...[0, 1, 2, 3, 4, 5, 6, 7].flatMap((i) => [...VZERO, ...set(acc + i)]),
      ...get(0), ...get(ib), ...get(4), ...MUL, ...i32(4), ...MUL, ...ADD, ...set(x),
      ...get(1), ...get(jb), ...get(4), ...MUL, ...i32(4), ...MUL, ...ADD, ...set(y),
      ...get(4), ...set(k),
      ...loop([...get(y), ...vload(0), ...set(b0), ...get(y), ...vload(16), ...set(b1), ...row(0), ...row(1), ...row(2), ...row(3),
        ...get(x), ...i32(16), ...ADD, ...set(x), ...get(y), ...i32(32), ...ADD, ...set(y), ...get(k), ...i32(1), ...SUB, ...tee(k), ...brIf(0)]),
      ...get(2), ...get(ib), ...get(5), ...MUL, ...get(jb), ...ADD, ...i32(4), ...MUL, ...ADD, ...set(o),
      ...[0, 1, 2, 3].flatMap((r) => [...get(o), ...get(acc + 2 * r), ...vstore(0), ...get(o), ...get(acc + 2 * r + 1), ...vstore(16), ...get(o), ...get(5), ...i32(4), ...MUL, ...ADD, ...set(o)]),
    ];
    const body = [...i32(0), ...set(jb), ...loop([...i32(0), ...set(ib),
      ...loop([...blk, ...get(ib), ...i32(4), ...ADD, ...tee(ib), ...get(3), ...LTU, ...brIf(0)]),
      ...get(jb), ...i32(8), ...ADD, ...tee(jb), ...get(5), ...LTU, ...brIf(0)])];
    return [...vec([[...uleb(6), I32], [...uleb(11), V128]]), ...body, 0x0b];
  })();

  // ---- exact(pa, pb, pc, M, K, Np): C [M × Np] (float64) = A [M × K] (float64) · Bp (float64, panels of 4 columns) ----
  // locals: 6 jp, 7 i, 8..11 row pointers, 12 y, 13 k, 14 o, 15 K4 (K rounded down to fours), 16 Mg (rows summed in
  // groups) (i32); 17..24 two accumulators per row, 25..32 B at four k (v128)
  const exact = (() => {
    const jp = 6, i = 7, X = 8, y = 12, k = 13, o = 14, K4 = 15, Mg = 16, acc = 17, bv = 25;
    const ld = (r, kk) => [...get(X + r), ...dload(8 * kk), ...DSPLAT];
    // acc += ((a0 b0 + a1 b1) + a2 b2) + a3 b3, as the JS kernel adds
    const groupRow = (r) => [0, 1].flatMap((h) => [...ld(r, 0), ...get(bv + h), ...DMUL, ...ld(r, 1), ...get(bv + 2 + h), ...DMUL, ...DADD,
      ...ld(r, 2), ...get(bv + 4 + h), ...DMUL, ...DADD, ...ld(r, 3), ...get(bv + 6 + h), ...DMUL, ...DADD, ...get(acc + 2 * r + h), ...DADD, ...set(acc + 2 * r + h)]);
    const loadB = (kk) => [...get(y), ...vload(32 * kk), ...set(bv + 2 * kk), ...get(y), ...vload(32 * kk + 16), ...set(bv + 2 * kk + 1)];
    const adv = (rows, by) => [...rows.flatMap((r) => [...get(X + r), ...i32(8 * by), ...ADD, ...set(X + r)]), ...get(y), ...i32(32 * by), ...ADD, ...set(y)];
    const single = (rows) => [...get(y), ...vload(0), ...set(bv), ...get(y), ...vload(16), ...set(bv + 1),
      ...rows.flatMap((r) => [0, 1].flatMap((h) => [...ld(r, 0), ...get(bv + h), ...DMUL, ...get(acc + 2 * r + h), ...DADD, ...set(acc + 2 * r + h)])),
      ...adv(rows, 1), ...get(k), ...i32(1), ...ADD, ...set(k)];
    const store = (rows) => rows.flatMap((r) => [
      ...get(2), ...get(i), ...i32(r), ...ADD, ...get(5), ...MUL, ...get(jp), ...ADD, ...i32(8), ...MUL, ...ADD, ...set(o),
      ...get(o), ...get(acc + 2 * r), ...vstore(0), ...get(o), ...get(acc + 2 * r + 1), ...vstore(16)]);
    const setup = (rows) => [
      ...rows.flatMap((r) => [...VZERO, ...set(acc + 2 * r), ...VZERO, ...set(acc + 2 * r + 1),
        ...get(0), ...get(i), ...i32(r), ...ADD, ...get(4), ...MUL, ...i32(8), ...MUL, ...ADD, ...set(X + r)]),
      ...get(1), ...get(jp), ...get(4), ...MUL, ...i32(8), ...MUL, ...ADD, ...set(y), ...i32(0), ...set(k)];
    const rest = (rows) => block([...loop([...get(k), ...get(4), ...GEU, ...brIf(1), ...single(rows), ...br(0)])]);
    const grouped = (rows) => [...setup(rows),
      ...block([...get(K4), ...i32(0), ...EQ, ...brIf(0), ...loop([...loadB(0), ...loadB(1), ...loadB(2), ...loadB(3), ...rows.flatMap(groupRow), ...adv(rows, 4),
        ...get(k), ...i32(4), ...ADD, ...tee(k), ...get(K4), ...LTU, ...brIf(0)])]),
      ...rest(rows), ...store(rows)];
    const body = [
      ...get(4), ...i32(-4), ...AND, ...set(K4), ...get(3), ...get(3), ...i32(1), ...AND, ...SUB, ...set(Mg), ...i32(0), ...set(jp),
      ...loop([
        ...i32(0), ...set(i),
        ...block([...loop([...get(i), ...i32(4), ...ADD, ...get(Mg), ...GTU, ...brIf(1), ...grouped([0, 1, 2, 3]), ...get(i), ...i32(4), ...ADD, ...set(i), ...br(0)])]),
        ...get(i), ...i32(2), ...ADD, ...get(Mg), ...LEU, ...IF([...grouped([0, 1]), ...get(i), ...i32(2), ...ADD, ...set(i)]),
        ...get(Mg), ...get(3), ...LTU, ...IF([...get(Mg), ...set(i), ...setup([0]), ...rest([0]), ...store([0])]),
        ...get(jp), ...i32(4), ...ADD, ...tee(jp), ...get(5), ...LTU, ...brIf(0)]),
    ];
    return [...vec([[...uleb(11), I32], [...uleb(16), V128]]), ...body, 0x0b];
  })();

  const bytes = new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...sec(1, vec([[0x60, ...vec(new Array(6).fill([I32])), 0]])), ...sec(3, vec([[0], [0]])), ...sec(5, vec([[0, ...uleb(1)]])),
    ...sec(7, vec([[...str('mem'), 2, 0], [...str('fast'), 0, 0], [...str('exact'), 0, 1]])),
    ...sec(10, vec([[...uleb(fast.length), ...fast], [...uleb(exact.length), ...exact]]))]);

  let inst = null;
  try { if (WebAssembly.validate(bytes)) inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {}); } catch (e) { inst = null; }
  K_.wasm = !!inst;
  if (!inst) return;
  const mem = inst.exports.mem, gemmJS = K_.gemmJS;
  const room = (bytes) => { if (mem.buffer.byteLength < bytes) mem.grow(Math.ceil((bytes - mem.buffer.byteLength) / 65536)); };

  function runFast(A, B, M, K, N, out, aOff, bOff, oOff, ta, tb) {
    const Mp = (M + 3) & ~3, Np = (N + 7) & ~7, pb = Mp * K, pc = pb + Np * K;
    room(4 * (pc + Mp * Np));
    const W = new Float32Array(mem.buffer);
    for (let i0 = 0; i0 < Mp; i0 += 4) {
      for (let r = 0; r < 4; r++) {
        const i = i0 + r, base = i0 * K + r;
        if (i >= M) { for (let q = 0; q < K; q++) W[base + 4 * q] = 0; continue; }
        if (ta) { for (let q = 0, s = aOff + i; q < K; q++, s += M) W[base + 4 * q] = A[s]; }
        else { for (let q = 0, s = aOff + i * K; q < K; q++, s++) W[base + 4 * q] = A[s]; }
      }
    }
    for (let j0 = 0; j0 < Np; j0 += 8) {
      for (let c = 0; c < 8; c++) {
        const j = j0 + c, base = pb + j0 * K + c;
        if (j >= N) { for (let q = 0; q < K; q++) W[base + 8 * q] = 0; continue; }
        if (tb) { for (let q = 0, s = bOff + j * K; q < K; q++, s++) W[base + 8 * q] = B[s]; }
        else { for (let q = 0, s = bOff + j; q < K; q++, s += N) W[base + 8 * q] = B[s]; }
      }
    }
    inst.exports.fast(0, 4 * pb, 4 * pc, Mp, K, Np);
    if (Np === N) out.set(W.subarray(pc, pc + M * N), oOff);
    else for (let i = 0; i < M; i++) out.set(W.subarray(pc + i * Np, pc + i * Np + N), oOff + i * N);
    return out;
  }
  function runExact(A, B, M, K, N, out, aOff, bOff, oOff, ta, tb) {
    const Np = (N + 3) & ~3, pb = M * K, pc = pb + Np * K;
    room(8 * (pc + M * Np));
    const W = new Float64Array(mem.buffer);
    if (ta) { for (let i = 0; i < M; i++) for (let q = 0, s = aOff + i, d = i * K; q < K; q++, s += M) W[d + q] = A[s]; }
    else for (let q = 0; q < M * K; q++) W[q] = A[aOff + q];
    for (let j0 = 0; j0 < Np; j0 += 4) {
      for (let c = 0; c < 4; c++) {
        const j = j0 + c, base = pb + j0 * K + c;
        if (j >= N) { for (let q = 0; q < K; q++) W[base + 4 * q] = 0; continue; }
        if (tb) { for (let q = 0, s = bOff + j * K; q < K; q++, s++) W[base + 4 * q] = B[s]; }
        else { for (let q = 0, s = bOff + j; q < K; q++, s += N) W[base + 4 * q] = B[s]; }
      }
    }
    inst.exports.exact(0, 8 * pb, 8 * pc, M, K, Np);
    for (let i = 0; i < M; i++) { const s = pc + i * Np, d = oOff + i * N; for (let j = 0; j < N; j++) out[d + j] = W[s + j]; }
    return out;
  }
  const f32 = (x) => x instanceof Float32Array, f64 = (x) => x instanceof Float64Array, real = (x) => f32(x) || f64(x);
  /** out = op(A) @ op(B) (see t-05-matmul.js) */
  function gemm(A, B, M, K, N, out, aOff = 0, bOff = 0, oOff = 0, ta = false, tb = false) {
    out = out || new (A.constructor)(M * N);
    if (!real(A) || !real(B) || !real(out) || M * K * N < 256 || !K) return gemmJS(A, B, M, K, N, out, aOff, bOff, oOff, ta, tb);
    if (K_.mode === 'fast' && f32(A) && f32(B) && f32(out)) return runFast(A, B, M, K, N, out, aOff, bOff, oOff, ta, tb);
    return runExact(A, B, M, K, N, out, aOff, bOff, oOff, ta, tb);
  }
  K_.gemmFast = (A, B, M, K, N, out, aOff = 0, bOff = 0, oOff = 0, ta = false, tb = false) => runFast(A, B, M, K, N, out || new Float32Array(M * N), aOff, bOff, oOff, ta, tb);
  K_.gemmExact = (A, B, M, K, N, out, aOff = 0, bOff = 0, oOff = 0, ta = false, tb = false) => runExact(A, B, M, K, N, out || new (A.constructor)(M * N), aOff, bOff, oOff, ta, tb);
  K_.gemmWasm = gemm;
  K_.gemm = gemm;
})(globalThis.Nooral = globalThis.Nooral || {});
