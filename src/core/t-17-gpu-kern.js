// WGSL kernels for training on the GPU: every forward op of a transformer and its backward pass, the loss and the
// optimiser, so a whole training step runs without the tensors leaving the GPU (t-18-gpu-tape.js drives them).
// Each kernel reads its sizes from a 256-byte uniform slot. All reductions run in a fixed order (no atomics), so a
// step gives the same numbers every time.
//   mm         C = op(A) · op(B) (op = as stored, or transposed), batched with strides, 64×64 tiles, 4×4 per thread
//   ew         y = a ⊕ b[i mod p]: add, sub, mul, c·a + c2·b, GELU, GELU's backward (a = x, b = upstream), c·a
//   colsum     out[j] = Σ_r g[r·p + j]  (the gradient of a bias or of position rows; zero past p)
//   lnf / lnb / lnp   layer norm forward (keeps mean and 1/std per row), its dx, and its dγ, dβ
//   smf / smb  softmax over each row of attention scores, scaled and causal, and its backward
//   perm       a permutation of up to four axes          gather / scatter   embedding rows and their gradient
//   ce         cross-entropy of a row of logits: the row's loss, whether its top class is the target, and the
//              gradient (softmax − one-hot) / count
//   reduce     Σ x or Σ x² of a buffer (times a factor, optionally the square root) into one slot
//   adam       Adam / AdamW with the global gradient-norm clip read from a buffer
(function (S) {
  'use strict';
  const T = S.T;
  const K = (T.gpuKernels = {});
  const IDX = 'let i = g.x + g.y * 16776960u;'; // 65535 groups of 256 per row of the grid

  K.mm = `
struct D { M: u32, K: u32, N: u32, sa: u32, sb: u32, sc: u32, ta: u32, tb: u32, aOff: u32, bOff: u32, cOff: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> Bm: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> d: D;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let r0 = w.y * 64u; let c0 = w.x * 64u;
  let aOff = d.aOff + w.z * d.sa; let bOff = d.bOff + w.z * d.sb; let cOff = d.cOff + w.z * d.sc;
  let tid = l.y * 16u + l.x;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < d.K; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      var r = e / 16u; var k = e % 16u;
      if (d.ta == 1u) { r = e % 64u; k = e / 64u; }
      let gr = r0 + r; let gk = k0 + k;
      var v = 0.0;
      if (gr < d.M && gk < d.K) { if (d.ta == 1u) { v = A[aOff + gk * d.M + gr]; } else { v = A[aOff + gr * d.K + gk]; } }
      As[r * 16u + k] = v;
    }
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      var k = e / 64u; var c = e % 64u;
      if (d.tb == 1u) { k = e % 16u; c = e / 16u; }
      let gk = k0 + k; let gc = c0 + c;
      var v = 0.0;
      if (gk < d.K && gc < d.N) { if (d.tb == 1u) { v = Bm[bOff + gc * d.K + gk]; } else { v = Bm[bOff + gk * d.N + gc]; } }
      Bs[k * 64u + c] = v;
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      var a: array<f32, 4>; var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[(l.y * 4u + i) * 16u + k]; b[i] = Bs[k * 64u + l.x + i * 16u]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += a[i] * b[j]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let r = r0 + l.y * 4u + i;
    if (r >= d.M) { continue; }
    for (var j = 0u; j < 4u; j++) { let c = c0 + l.x + j * 16u; if (c < d.N) { C[cOff + r * d.N + c] = acc[i * 4u + j]; } }
  }
}`;

  K.ew = `
struct U { n: u32, p: u32, op: u32, pad: u32, c: f32, c2: f32, q0: f32, q1: f32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Bv: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
const GC = 0.7978845608028654;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${IDX}
  if (i >= u.n) { return; }
  let a = X[i]; let b = Bv[i % u.p];
  var y = 0.0;
  switch u.op {
    case 0u: { y = a + b; }
    case 1u: { y = a - b; }
    case 2u: { y = a * b; }
    case 3u: { y = u.c * a + u.c2 * b; }
    case 4u: { let t = tanh(clamp(GC * (a + 0.044715 * a * a * a), -20.0, 20.0)); y = 0.5 * a * (1.0 + t); }
    case 5u: { let t = tanh(clamp(GC * (a + 0.044715 * a * a * a), -20.0, 20.0));
      y = b * (0.5 * (1.0 + t) + 0.5 * a * (1.0 - t * t) * GC * (1.0 + 3.0 * 0.044715 * a * a)); }
    default: { y = u.c * a; }
  }
  Y[i] = y;
}`;

  K.colsum = `
struct U { rows: u32, p: u32, outN: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${IDX}
  if (i >= u.outN) { return; }
  if (i >= u.p) { O[i] = 0.0; return; }
  var s = 0.0;
  for (var r = 0u; r < u.rows; r++) { s += G[r * u.p + i]; }
  O[i] = s;
}`;

  // one workgroup of 64 per row; red is the shared scratch for the row sums
  const ROW = `
var<workgroup> red: array<f32, 64>;
fn total(l: u32, x: f32) -> f32 {
  red[l] = x; workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) { if (l < s) { red[l] = red[l] + red[l + s]; } workgroupBarrier(); }
  let t = red[0]; workgroupBarrier();
  return t;
}
fn biggest(l: u32, x: f32) -> f32 {
  red[l] = x; workgroupBarrier();
  for (var s = 32u; s > 0u; s = s / 2u) { if (l < s) { red[l] = max(red[l], red[l + s]); } workgroupBarrier(); }
  let t = red[0]; workgroupBarrier();
  return t;
}`;
  const ROWID = 'let row = w.x + w.y * 65535u; if (row >= u.rows) { return; } let t = l.x;';

  K.lnf = `
struct U { rows: u32, D: u32, eps: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gm: array<f32>;
@group(0) @binding(2) var<storage, read> Bt: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> St: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
${ROW}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  ${ROWID}
  let base = row * u.D; let D = f32(u.D);
  var s = 0.0;
  for (var j = t; j < u.D; j += 64u) { s += X[base + j]; }
  let m = total(t, s) / D;
  var q = 0.0;
  for (var j = t; j < u.D; j += 64u) { let z = X[base + j] - m; q += z * z; }
  let rs = 1.0 / sqrt(total(t, q) / D + u.eps);
  for (var j = t; j < u.D; j += 64u) { Y[base + j] = (X[base + j] - m) * rs * Gm[j] + Bt[j]; }
  if (t == 0u) { St[2u * row] = m; St[2u * row + 1u] = rs; }
}`;
  K.lnb = `
struct U { rows: u32, D: u32, eps: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gm: array<f32>;
@group(0) @binding(2) var<storage, read> Gr: array<f32>;
@group(0) @binding(3) var<storage, read> St: array<f32>;
@group(0) @binding(4) var<storage, read_write> DX: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
${ROW}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  ${ROWID}
  let base = row * u.D; let D = f32(u.D); let m = St[2u * row]; let rs = St[2u * row + 1u];
  var a = 0.0; var b = 0.0;
  for (var j = t; j < u.D; j += 64u) { let dh = Gr[base + j] * Gm[j]; a += dh; b += dh * (X[base + j] - m) * rs; }
  let s1 = total(t, a); let s2 = total(t, b);
  for (var j = t; j < u.D; j += 64u) {
    let dh = Gr[base + j] * Gm[j]; let xh = (X[base + j] - m) * rs;
    DX[base + j] = (rs / D) * (D * dh - s1 - xh * s2);
  }
}`;
  K.lnp = `
struct U { rows: u32, D: u32, eps: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gr: array<f32>;
@group(0) @binding(2) var<storage, read> St: array<f32>;
@group(0) @binding(3) var<storage, read_write> DG: array<f32>;
@group(0) @binding(4) var<storage, read_write> DB: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let j = g.x;
  if (j >= u.D) { return; }
  var a = 0.0; var b = 0.0;
  for (var r = 0u; r < u.rows; r++) { let gg = Gr[r * u.D + j]; a += gg * (X[r * u.D + j] - St[2u * r]) * St[2u * r + 1u]; b += gg; }
  DG[j] = a; DB[j] = b;
}`;

  // rows of attention scores [batch·heads·Tq, Tk]; with causal, query i sees keys j ≤ i + Tk − Tq
  K.smf = `
struct U { rows: u32, Tq: u32, Tk: u32, causal: u32, scale: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> P: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
${ROW}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  ${ROWID}
  let base = row * u.Tk;
  var lim = u.Tk;
  if (u.causal == 1u) { lim = min(u.Tk, row % u.Tq + u.Tk - u.Tq + 1u); }
  var m = -3.4e38;
  for (var j = t; j < lim; j += 64u) { m = max(m, X[base + j] * u.scale); }
  m = biggest(t, m);
  var s = 0.0;
  for (var j = t; j < lim; j += 64u) { s += exp(X[base + j] * u.scale - m); }
  let q = 1.0 / total(t, s);
  for (var j = t; j < u.Tk; j += 64u) { if (j < lim) { P[base + j] = exp(X[base + j] * u.scale - m) * q; } else { P[base + j] = 0.0; } }
}`;
  K.smb = `
struct U { rows: u32, Tq: u32, Tk: u32, causal: u32, scale: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<storage, read> P: array<f32>;
@group(0) @binding(1) var<storage, read> Gr: array<f32>;
@group(0) @binding(2) var<storage, read_write> DX: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
${ROW}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  ${ROWID}
  let base = row * u.Tk;
  var a = 0.0;
  for (var j = t; j < u.Tk; j += 64u) { a += Gr[base + j] * P[base + j]; }
  let s = total(t, a);
  for (var j = t; j < u.Tk; j += 64u) { DX[base + j] = u.scale * P[base + j] * (Gr[base + j] - s); }
}`;

  K.perm = `
struct U { n: u32, o0: u32, o1: u32, o2: u32, o3: u32, s0: u32, s1: u32, s2: u32, s3: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${IDX}
  if (i >= u.n) { return; }
  var r = i;
  let d = r % u.o3; r = r / u.o3;
  let c = r % u.o2; r = r / u.o2;
  let b = r % u.o1; let a = r / u.o1;
  Y[i] = X[a * u.s0 + b * u.s1 + c * u.s2 + d * u.s3];
}`;

  K.gather = `
struct U { n: u32, d: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> Ids: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${IDX}
  if (i >= u.n * u.d) { return; }
  let t = i / u.d; let j = i % u.d;
  Y[i] = W[u32(Ids[t]) * u.d + j];
}`;
  // the gradient of an embedding: row v sums the upstream rows of the tokens that are v (listed per row, in order)
  K.scatter = `
struct U { V: u32, d: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> Starts: array<u32>;
@group(0) @binding(1) var<storage, read> Pos: array<u32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> DW: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${IDX}
  if (i >= u.V * u.d) { return; }
  let v = i / u.d; let j = i % u.d;
  var s = 0.0;
  for (var k = Starts[v]; k < Starts[v + 1u]; k++) { s += G[Pos[k] * u.d + j]; }
  DW[i] = s;
}`;

  K.ce = `
struct U { rows: u32, V: u32, inv: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> Z: array<f32>;
@group(0) @binding(1) var<storage, read> Tg: array<f32>;
@group(0) @binding(2) var<storage, read_write> L: array<f32>;
@group(0) @binding(3) var<storage, read_write> DZ: array<f32>;
@group(0) @binding(4) var<storage, read_write> Cor: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
${ROW}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  ${ROWID}
  let base = row * u.V; let y = Tg[row];
  var m = -3.4e38;
  for (var j = t; j < u.V; j += 64u) { m = max(m, Z[base + j]); }
  m = biggest(t, m);
  var first = -3.4e38;
  for (var j = t; j < u.V; j += 64u) { if (Z[base + j] == m) { first = max(first, -f32(j)); } }
  let top = -biggest(t, first);
  var s = 0.0;
  for (var j = t; j < u.V; j += 64u) { s += exp(Z[base + j] - m); }
  let lse = m + log(total(t, s));
  let keep = y >= 0.0;
  for (var j = t; j < u.V; j += 64u) {
    var d = 0.0;
    if (keep) { d = exp(Z[base + j] - lse); if (f32(j) == y) { d -= 1.0; } d *= u.inv; }
    DZ[base + j] = d;
  }
  if (t == 0u) {
    if (keep) { L[row] = lse - Z[base + u32(y)]; } else { L[row] = 0.0; }
    if (keep && top == y) { Cor[row] = 1.0; } else { Cor[row] = 0.0; }
  }
}`;

  K.reduce = `
struct U { n: u32, sq: u32, slot: u32, root: u32, c: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var s = 0.0;
  for (var j = l.x; j < u.n; j += 256u) { let x = X[j]; if (u.sq == 1u) { s += x * x; } else { s += x; } }
  red[l.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k = k / 2u) { if (l.x < k) { red[l.x] = red[l.x] + red[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { var t = red[0] * u.c; if (u.root == 1u) { t = sqrt(t); } O[u.slot] = t; }
}`;

  K.adam = `
struct U { n: u32, decoupled: u32, clipOn: u32, pad: u32, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, c1: f32, c2: f32, clip: f32 };
@group(0) @binding(0) var<storage, read_write> W: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> Mm: array<f32>;
@group(0) @binding(3) var<storage, read_write> Vv: array<f32>;
@group(0) @binding(4) var<storage, read> Nrm: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${IDX}
  if (i >= u.n) { return; }
  var gi = G[i]; var w = W[i];
  if (u.clipOn == 1u) { let nr = Nrm[0]; if (nr > u.clip && nr > 0.0) { gi = gi * (u.clip / nr); } }
  if (u.decoupled == 1u) { w = w - u.lr * u.wd * w; } else { gi = gi + u.wd * w; }
  let m = u.b1 * Mm[i] + (1.0 - u.b1) * gi;
  let v = u.b2 * Vv[i] + (1.0 - u.b2) * gi * gi;
  Mm[i] = m; Vv[i] = v;
  W[i] = w - (u.lr * (m / u.c1)) / (sqrt(v / u.c2) + u.eps);
}`;
})(globalThis.Nooral = globalThis.Nooral || {});
