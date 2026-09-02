// Hand-written versions of part 11's models for tests/claims-p11.mjs: plain loops in double precision over the
// weights decoded straight from the base64 in src/topics/p11-0-data.js (not through the topic's decoder), so the
// claims check compares the engine's numbers with an independent working-out.
export const sig = (x) => 1 / (1 + Math.exp(-x));
export const softmax = (row) => { const m = Math.max(...row), e = row.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
export const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const cos = (a, b) => dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));

/** {name: matrix or vector} from a shipped job's weights, decoded with Buffer */
export function weights(job) {
  const buf = Buffer.from(job.weights.b64, 'base64'), all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4), out = {};
  let at = 0;
  job.weights.names.forEach((n, i) => {
    const sh = job.weights.shapes[i], size = sh.reduce((a, b) => a * b, 1), v = Array.from(all.subarray(at, at + size));
    at += size;
    out[n] = sh.length === 2 ? Array.from({ length: sh[0] }, (_, r) => v.slice(r * sh[1], (r + 1) * sh[1])) : v;
  });
  out.size = at;
  return out;
}
/** row vector x (length I) times matrix W [I][O] */
export const vm = (x, W) => W[0].map((_, j) => x.reduce((s, v, i) => s + v * W[i][j], 0));
const add = (...vs) => vs[0].map((_, j) => vs.reduce((s, v) => s + v[j], 0));

/** one step of each cell, in the engine's gate order */
export function rnnStep(w, p, x, h) { return add(vm(x, w[p + 'Wx']), vm(h, w[p + 'Wh']), w[p + 'b']).map(Math.tanh); }
export function lstmStep(w, p, x, h, c) {
  const z = add(vm(x, w[p + 'Wx']), vm(h, w[p + 'Wh']), w[p + 'b']), H = h.length;
  const i = z.slice(0, H).map(sig), f = z.slice(H, 2 * H).map(sig), g = z.slice(2 * H, 3 * H).map(Math.tanh), o = z.slice(3 * H).map(sig);
  const c2 = c.map((cv, k) => f[k] * cv + i[k] * g[k]), h2 = c2.map((cv, k) => o[k] * Math.tanh(cv));
  return { i, f, g, o, c: c2, h: h2 };
}
export function gruStep(w, p, x, h) {
  const H = h.length, xw = add(vm(x, w[p + 'Wx']), w[p + 'b']), hh = add(vm(h, w[p + 'Wh']), w[p + 'bh']);
  const r = xw.slice(0, H).map((v, k) => sig(v + hh[k])), z = xw.slice(H, 2 * H).map((v, k) => sig(v + hh[H + k]));
  const n = xw.slice(2 * H).map((v, k) => Math.tanh(v + r[k] * hh[2 * H + k]));
  return { r, z, n, h: n.map((nv, k) => (1 - z[k]) * nv + z[k] * h[k]) };
}
export const head = (w, h, p = 'head.') => add(vm(h, w[p + 'W']), w[p + 'b']);

/** a whole stream through a recurrent 'rnn' model: per step the cell's values and the output probabilities */
export function runRecurrent(w, cell, inputs, embed) {
  const H = w['cell.Wh'].length;
  let h = new Array(H).fill(0), c = new Array(H).fill(0);
  return inputs.map((x) => {
    const xv = embed ? w['emb.W'][x] : [x];
    let s;
    if (cell === 'rnn') { h = rnnStep(w, 'cell.', xv, h); s = { h }; }
    else if (cell === 'lstm') { s = lstmStep(w, 'cell.', xv, h, c); h = s.h; c = s.c; }
    else { s = gruStep(w, 'cell.', xv, h); h = s.h; }
    return Object.assign({ p: softmax(head(w, h)) }, s);
  });
}

/** the encoder–decoder, greedy: the context, every decoder step's state and probabilities */
export function translate(w, src, max, START, END) {
  const H = w['enc.Wh'].length;
  let h = new Array(H).fill(0);
  for (const x of src) h = gruStep(w, 'enc.', w['embIn.W'][x], h).h;
  const context = h.slice(), steps = [];
  let tok = START;
  for (let k = 0; k < max; k++) {
    h = gruStep(w, 'dec.', w['embOut.W'][tok], h).h;
    const p = softmax(head(w, h)), id = argmax(p);
    steps.push({ input: tok, p, id, h: h.slice() });
    tok = id;
    if (id === END) break;
  }
  return { context, steps, out: steps.map((s) => s.id).filter((i) => i !== END) };
}

/** eigenvalues of a symmetric matrix by Jacobi rotations */
export function eigSym(A0) {
  const n = A0.length, A = A0.map((r) => r.slice());
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
    }
  }
  return A.map((r, i) => r[i]).sort((a, b) => b - a);
}
