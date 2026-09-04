// Part 12, transformers: the narration of every topic read back against an independent working-out. Attention on
// the tiny matrices by plain loops; softmax, layer normalisation and GELU by their formulas; the trained models run
// again by a forward pass written here from their weights (no engine ops); the KV cache against a full recompute;
// and the shipped training runs replayed from scratch for their first steps (the same seed must give the same numbers).
// Usage: node tests/claims-p12.mjs [topic id prefix ...] [--full] (--full replays whole training runs: about a minute)
import { S, run, phase } from './claims-lib-p06-08.mjs';
import { RUNS, setupOf, probeInput, probeAttn } from './ref/p12t_runs.mjs';

const P = S.P12;
const FULL = process.argv.includes('--full');
// formula constants the narration may write as they are
const ALLOWED = [/towards 0\b/g, /about −0\.17/g, /a direct path of 1\b/g, /1 \+ f′\(x\)/g, /positions 0 to i\b/g, /exactly 0\b/g, /10000\^\(2i\/d\)/g, /dimension 2i\b/g, /p\(1 − p\)/g, /near 0 when p is near 0 or 1/g, /add up to 1\b/g, /adds up to 1\b/g, /\b1 \/ /g, /√\(?dₖ\)?/g, /ε = 10⁻⁵/g, /\bmean 0\b/g, /\bspread 1\b/g, /−10⁹/g];
const near = (c, a, b, msg, tol) => c.ok(Math.abs(a - b) <= (tol || 1e-6) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const vnear = (c, a, b, msg, tol) => c.ok(a && b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= (tol || 1e-6) * Math.max(1, Math.abs(b[i]))), msg + ': got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b));
const mnear = (c, A, B, msg, tol) => c.ok(A && B && A.length === B.length && A.every((r, i) => r.length === B[i].length && r.every((v, j) => Math.abs(v - B[i][j]) <= (tol || 1e-6) * Math.max(1, Math.abs(B[i][j])))), msg);
const f32 = 2e-5;

// ---- plain arithmetic, nothing from the engine ---------------------------------------------------------------------
const mul = (A, B) => A.map((r) => B[0].map((_, j) => { let s = 0; for (let k = 0; k < r.length; k++) s += r[k] * B[k][j]; return s; }));
const tr = (M) => M[0].map((_, j) => M.map((r) => r[j]));
const softmax = (row) => { const m = Math.max(...row), e = row.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
const layerNorm = (v, g, b) => { const n = v.length, m = v.reduce((a, x) => a + x, 0) / n, va = v.reduce((a, x) => a + (x - m) ** 2, 0) / n, r = 1 / Math.sqrt(va + 1e-5); return v.map((x, i) => (x - m) * r * g[i] + b[i]); };
const argmax = (v) => v.reduce((bi, x, i) => (x > v[bi] ? i : bi), 0);
const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
/** the weights of a shipped model as plain nested arrays, by name */
const weightsOf = (name) => {
  const r = S.P12D.runs[name], w = Float32Array.from(Buffer.from(r.weights, 'base64').buffer.slice(0)), out = {};
  const buf = Buffer.from(r.weights, 'base64'), f = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  let at = 0;
  for (const [n, sh] of r.names) { const size = sh.reduce((a, b) => a * b, 1), v = Array.from(f.subarray(at, at + size)); at += size; out[n] = sh.length === 2 ? Array.from({ length: sh[0] }, (_, i) => v.slice(i * sh[1], (i + 1) * sh[1])) : v; }
  void w;
  return { W: out, model: r.model };
};
/** a forward pass of the tiny GPT written out by hand: every intermediate, as P.trace gives them */
const handForward = (name, ids, o = {}) => {
  const { W, model } = weightsOf(name), n = ids.length, d = model.d, Hh = model.heads, dk = d / Hh;
  const lin = (X, p) => mul(X, W[p + '.W']).map((r) => r.map((v, j) => v + (W[p + '.b'] ? W[p + '.b'][j] : 0)));
  let x = ids.map((t, i) => W['emb.W'][t].map((v, j) => v + W['pos.P'][i][j]));
  const out = { x0: x, blocks: [] };
  for (let l = 0; l < model.layers; l++) {
    const p = 'blocks.' + l + '.', b = { inp: x };
    const l1 = x.map((r) => layerNorm(r, W[p + 'ln1.gamma'], W[p + 'ln1.beta']));
    const Q = lin(l1, p + 'attn.Wq'), K = lin(l1, p + 'attn.Wk'), V = lin(l1, p + 'attn.Wv');
    b.ln1 = l1; b.q = []; b.k = []; b.v = []; b.w = []; b.scores = [];
    const joined = Array.from({ length: n }, () => new Array(d).fill(0));
    for (let h = 0; h < Hh; h++) {
      const cut = (M) => M.map((r) => r.slice(h * dk, (h + 1) * dk)), q = cut(Q), k = cut(K), v = cut(V);
      const sc = q.map((qi, i) => k.map((kj, j) => (o.causal === false || j <= i ? qi.reduce((s, a, t) => s + a * kj[t], 0) / Math.sqrt(dk) : -1e9)));
      const wts = sc.map(softmax);
      b.q.push(q); b.k.push(k); b.v.push(v); b.scores.push(sc); b.w.push(wts);
      const ov = mul(wts, v);
      if (o.ablate && o.ablate.layer === l && o.ablate.head === h) continue;
      for (let i = 0; i < n; i++) for (let t = 0; t < dk; t++) joined[i][h * dk + t] = ov[i][t];
    }
    b.heads = joined;
    b.att = lin(joined, p + 'attn.Wo');
    x = x.map((r, i) => r.map((v, j) => v + b.att[i][j])); b.mid = x;
    b.ln2 = x.map((r) => layerNorm(r, W[p + 'ln2.gamma'], W[p + 'ln2.beta']));
    b.pre = lin(b.ln2, p + 'ff1'); b.act = b.pre.map((r) => r.map(gelu)); b.ff = lin(b.act, p + 'ff2');
    x = x.map((r, i) => r.map((v, j) => v + b.ff[i][j])); b.out = x;
    out.blocks.push(b);
  }
  out.lnf = x.map((r) => layerNorm(r, W['lnf.gamma'], W['lnf.beta']));
  out.logits = mul(out.lnf, tr(W['emb.W']));
  out.probs = out.logits.map(softmax);
  out.W = W; out.model = model;
  return out;
};
export { handForward, weightsOf };
/** one attention layer by hand: rows X (already normalised), weights under prefix p, with or without the causal mask */
const attnLayer = (W, p, X, Hh, causal) => {
  const d = X[0].length, dk = d / Hh, n = X.length;
  const lin = (M, q) => mul(M, W[q + '.W']).map((r) => r.map((v, j) => v + W[q + '.b'][j]));
  const Q = lin(X, p + 'attn.Wq'), K = lin(X, p + 'attn.Wk'), V = lin(X, p + 'attn.Wv'), joined = Array.from({ length: n }, () => new Array(d).fill(0)), ws = [];
  for (let h = 0; h < Hh; h++) {
    const cut = (M) => M.map((r) => r.slice(h * dk, (h + 1) * dk)), q = cut(Q), k = cut(K), v = cut(V);
    const w = q.map((qi, i) => softmax(k.map((kj, j) => (!causal || j <= i ? qi.reduce((s, a, t) => s + a * kj[t], 0) / Math.sqrt(dk) : -1e9))));
    ws.push(w);
    const ov = mul(w, v);
    for (let i = 0; i < n; i++) for (let t = 0; t < dk; t++) joined[i][h * dk + t] = ov[i][t];
  }
  return { out: lin(joined, p + 'attn.Wo'), w: ws, Q, K, V };
};
const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) / (norm(a) * norm(b));

const checks = {
  'attention-sdpa': {
    random: (r) => { const sentence = ['rain', 'mila', 'clocks', 'two'][Math.floor(r() * 4)], n = P.SENTENCES[sentence].words.length; return { sentence, dk: String(2 + Math.floor(r() * 3)), focus: Math.floor(r() * n), seed: 1 + Math.floor(r() * 900) }; },
    randomRuns: 10,
    check(c) {
      const m0 = phase(c.frames, 'x')[0].meta, i = c.input.focus, dk = +c.input.dk;
      const Q = mul(m0.X, m0.Wq), K = mul(m0.X, m0.Wk), V = mul(m0.X, m0.Wv), raw = mul(Q, tr(K)), sc = raw.map((r) => r.map((v) => v / Math.sqrt(dk))), A = sc.map(softmax), O = mul(A, V);
      vnear(c, m0.x, m0.X[i], 'the followed row');
      const mq = phase(c.frames, 'q')[0].meta;
      vnear(c, mq.qi, Q[i], 'query row', f32); near(c, mq.q0, Q[i][0], 'first query number', f32);
      vnear(c, mq.prods, m0.X[i].map((x, k) => x * m0.Wq[k][0]), 'products of the first query number', f32);
      vnear(c, phase(c.frames, 'k')[0].meta.ki, K[i], 'key row', f32);
      vnear(c, phase(c.frames, 'v')[0].meta.vi, V[i], 'value row', f32);
      const ms = phase(c.frames, 'scores')[0].meta;
      vnear(c, ms.row, raw[i], 'raw scores', f32); c.eq(ms.best, argmax(raw[i]), 'largest score');
      vnear(c, ms.prods, Q[i].map((v, t) => v * K[ms.best][t]), 'products of the best score', f32);
      const mc = phase(c.frames, 'scaled')[0].meta;
      vnear(c, mc.row, sc[i], 'scaled scores', f32); near(c, mc.root, Math.sqrt(dk), '√dₖ');
      const mx = phase(c.frames, 'softmax')[0].meta;
      vnear(c, mx.ex, sc[i].map(Math.exp), 'exponentials', 1e-4); near(c, mx.z, sc[i].map(Math.exp).reduce((a, b) => a + b, 0), 'their total', 1e-4);
      vnear(c, mx.w, A[i], 'weights', f32); c.eq(mx.win, argmax(A[i]), 'largest weight');
      const mo = phase(c.frames, 'out')[0].meta;
      vnear(c, mo.o, O[i], 'output row', f32); vnear(c, mo.prods, A[i].map((a, k) => a * V[k][0]), 'output products', f32);
      const me = phase(c.frames, 'end')[0].meta;
      vnear(c, me.self, A.map((r, k) => r[k]), 'self weights', f32); c.eq(me.most, argmax(A.map((r, k) => r[k])), 'most self weight');
    },
  },
  'attention-scale': {
    check(c) {
      const n = +c.input.n, m = phase(c.frames, 'start')[0].meta.m;
      const stat = (d) => {
        const { q, k } = P.scaleDraw(c.input.seed, d, n, m), raw = [], wu = [], ws = [];
        for (let r = 0; r < m; r++) {
          const row = []; for (let j = 0; j < n; j++) { let s = 0; for (let t = 0; t < d; t++) s += q[r * d + t] * k[(r * n + j) * d + t]; row.push(s); }
          raw.push(row); wu.push(softmax(row)); ws.push(softmax(row.map((v) => v / Math.sqrt(d))));
        }
        const all = raw.flat(), mean = all.reduce((a, b) => a + b, 0) / all.length, sd = Math.sqrt(all.reduce((a, b) => a + (b - mean) ** 2, 0) / all.length);
        const st = (W) => ({ max: W.reduce((a, r) => a + Math.max(...r), 0) / m, grad: W.reduce((a, r) => a + r.reduce((s, p) => s + p * (1 - p), 0), 0) / m, sharp: W.filter((r) => Math.max(...r) > 0.9).length / m });
        return { sd, un: st(wu), sc: st(ws), raw, wu, ws };
      };
      const memo = {};
      const get = (d) => (memo[d] = memo[d] || stat(d));
      phase(c.frames, 'size').forEach((f) => {
        const s = get(f.meta.d);
        near(c, f.meta.sd, s.sd, 'score spread at ' + f.meta.d, 1e-4); near(c, f.meta.un, s.un.max, 'largest weight unscaled at ' + f.meta.d, 1e-4);
        near(c, f.meta.sc, s.sc.max, 'largest weight scaled at ' + f.meta.d, 1e-4); near(c, f.meta.sharp, s.un.sharp, 'sharp rows at ' + f.meta.d, 1e-9);
      });
      const e = phase(c.frames, 'example')[0].meta, s = get(e.d);
      vnear(c, e.raw, s.raw[0], 'example scores', 1e-4); vnear(c, e.wu, s.wu[0], 'example unscaled', 1e-4); vnear(c, e.ws, s.ws[0], 'example scaled', 1e-4); c.eq(e.bu, argmax(s.wu[0]), 'example winner');
      const g = phase(c.frames, 'grad')[0].meta, sg = get(g.d);
      near(c, g.gu, sg.un.grad, 'gradient size unscaled', 1e-4); near(c, g.gs, sg.sc.grad, 'gradient size scaled', 1e-4);
    },
  },
  'transformer-tokens': {
    random: (r) => ({ task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)], digits: Array.from({ length: 5 }, () => Math.floor(r() * 8)).join(''), pick: Math.floor(r() * 10) }),
    check(c) {
      const { W } = weightsOf(c.input.task), x = [...c.input.digits].map(Number), ans = c.input.task === 'reverse' ? x.slice().reverse() : c.input.task === 'sort' ? x.slice().sort((a, b) => a - b) : x.slice();
      const ids = x.concat([8], ans).slice(0, -1), k = c.input.pick, t = ids[k];
      c.eq(phase(c.frames, 'ids')[0].meta.n, 10, 'sequence length');
      const r = phase(c.frames, 'row')[0].meta;
      c.eq(r.t, t, 'token id'); vnear(c, r.row, W['emb.W'][t].slice(0, 4), 'embedding row', 1e-7);
      c.eq(phase(c.frames, 'stack')[0].meta.same, ids.map((v, i) => (v === t ? i : -1)).filter((i) => i >= 0), 'positions holding the token');
      const s = phase(c.frames, 'similar')[0].meta;
      let ba = 0, bb = 1; for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) if (cosine(W['emb.W'][a], W['emb.W'][b]) > cosine(W['emb.W'][ba], W['emb.W'][bb])) { ba = a; bb = b; }
      c.eq([s.a, s.b], [ba, bb], 'closest pair'); near(c, s.cos, cosine(W['emb.W'][ba], W['emb.W'][bb]), 'their cosine', 1e-5);
    },
  },
  'transformer-positions': {
    random: (r) => ({ d: ['16', '32', '64'][Math.floor(r() * 3)], len: 12 + Math.floor(r() * 49), off: 1 + Math.floor(r() * 10), task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)] }),
    check(c) {
      const d = +c.input.d, N = c.input.len, off = c.input.off, { W, model } = weightsOf(c.input.task);
      const pe = (p) => Array.from({ length: d }, (_, i) => { const w = p / Math.pow(10000, (2 * Math.floor(i / 2)) / d); return i % 2 ? Math.cos(w) : Math.sin(w); });
      const ids = [2, 7, 4, 0, 5], perm = [3, 0, 4, 1, 2];
      const out = (seq) => attnLayer(W, 'blocks.0.', seq.map((t) => layerNorm(W['emb.W'][t], W['blocks.0.ln1.gamma'], W['blocks.0.ln1.beta'])), model.heads, false).out;
      const o1 = out(ids), o2 = out(perm.map((p) => ids[p]));
      let diff = 0; perm.forEach((p, i) => o2[i].forEach((v, j) => { diff = Math.max(diff, Math.abs(v - o1[p][j])); }));
      const b = phase(c.frames, 'blind')[0].meta;
      c.ok(diff < 1e-5 && b.diff < 1e-5, 'shuffling changes nothing but rounding: ' + b.diff + ' / ' + diff);
      const t = phase(c.frames, 'table')[0].meta;
      near(c, t.w0, 2 * Math.PI, 'fastest wavelength'); near(c, t.wl, 2 * Math.PI * Math.pow(10000, (d - 2) / d), 'slowest wavelength');
      const o = phase(c.frames, 'offset')[0].meta, dp = (a, bb) => pe(a).reduce((s, v, i) => s + v * pe(bb)[i], 0);
      near(c, o.a, dp(o.p1, o.p1 + off), 'first offset product', 1e-5); near(c, o.b, dp(o.p2, o.p2 + off), 'second offset product', 1e-5);
      c.ok(Math.abs(o.a - o.b) < 1e-4, 'the two products agree'); near(c, o.peak, dp(3, 3), 'a position with itself', 1e-6); c.ok(o.p2 + off < N, 'positions inside the table');
      const L = phase(c.frames, 'learned')[0].meta, Pl = W['pos.P'], m = Pl.length;
      let nr = 0, fr = 0, cn = 0, cf = 0; for (let a = 0; a < m; a++) for (let bb = 0; bb < m; bb++) { const cs = cosine(Pl[a], Pl[bb]); if (Math.abs(a - bb) === 1) { nr += cs; cn++; } else if (Math.abs(a - bb) >= 5) { fr += cs; cf++; } }
      near(c, L.near, nr / cn, 'neighbour similarity', 1e-5); near(c, L.far, fr / cf, 'far similarity', 1e-5); c.eq(L.m, m, 'learned rows');
    },
  },
  'attention-masks': {
    random: (r) => { const lenA = 3 + Math.floor(r() * 4); return { lenA, lenB: 1 + Math.floor(r() * (lenA - 1)), seed: 1 + Math.floor(r() * 900) }; },
    randomRuns: 10,
    check(c) {
      const I = P.maskInputs(c.input), n = I.n, m = I.m;
      const att = (X, causal, pad) => {
        const Q = mul(X, I.Wq), K = mul(X, I.Wk), V = mul(X, I.Wv);
        const sc = Q.map((qi, i) => K.map((kj, j) => ((causal && j > i) || (pad && j >= m) ? -1e9 : qi.reduce((s, a, t) => s + a * kj[t], 0) / Math.sqrt(3))));
        const w = sc.map(softmax);
        return { sc, w, o: mul(w, V) };
      };
      const bNo = att(I.XB, false, false), bPad = att(I.XB, false, true), aCau = att(I.XA, true, false), bBoth = att(I.XB, true, true), aNo = att(I.XA, false, false);
      const b = phase(c.frames, 'batch')[0].meta; c.eq([b.n, b.m, b.pads], [n, m, n - m], 'lengths and pads');
      near(c, phase(c.frames, 'nomask')[0].meta.padShare, bNo.w[0].slice(m).reduce((a, x) => a + x, 0), 'weight on pads without the mask', 1e-5);
      const p = phase(c.frames, 'pad')[0].meta;
      c.eq(p.masked, n * (n - m), 'padding scores masked'); vnear(c, p.row, bPad.w[0].slice(0, m), 'row weights with padding mask', 1e-5); c.ok(p.padW.every((v) => v === 0), 'pads get exactly 0');
      near(c, p.row.reduce((a, x) => a + x, 0), 1, 'real weights add to 1', 1e-5);
      const ca = phase(c.frames, 'causal')[0].meta;
      c.eq(ca.masked, (n * (n - 1)) / 2, 'causal scores masked'); near(c, ca.first, 1, 'first word on itself', 1e-6); vnear(c, ca.lastRow, aCau.w[n - 1], 'last row', 1e-5);
      const bo = phase(c.frames, 'both')[0].meta;
      vnear(c, bo.row, bBoth.w[m - 1].slice(0, m), 'both masks row', 1e-5);
      let kept = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (j <= i && j < m && i < m) kept++;
      c.eq(bo.kept, kept, 'scores kept among real words'); c.eq(bo.masked, n * n - bBoth.sc.flat().filter((v) => v > -1e8).length, 'masked with both');
      const XA2 = I.XA.slice(0, n - 1).concat([I.last]), aCau2 = att(XA2, true, false), aNo2 = att(XA2, false, false);
      const dmax = (o1, o2) => Math.max(...o1.slice(0, n - 1).flatMap((r, i) => r.map((v, j) => Math.abs(v - o2[i][j]))));
      const pk = phase(c.frames, 'peek')[0].meta;
      c.ok(pk.dC < 1e-6 && dmax(aCau.o, aCau2.o) < 1e-12, 'causal outputs unchanged'); near(c, pk.dN, dmax(aNo.o, aNo2.o), 'change without the mask', 1e-5);
    },
  },
  'attention-multihead': {
    random: (r) => ({ task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)], digits: Array.from({ length: 5 }, () => Math.floor(r() * 8)).join(''), row: 5 + Math.floor(r() * 5) }),
    randomRuns: 4,
    check(c) {
      const task = c.input.task, x = [...c.input.digits].map(Number), ans = P.answer(task, x), full = x.concat([8], ans), ids = full.slice(0, -1), hf = handForward(task, ids), r = c.input.row;
      const src = (i) => full.slice(0, 5).indexOf(full[i + 1]);
      const sp = phase(c.frames, 'split')[0].meta; c.eq([sp.d, sp.Hh, sp.dk], [hf.model.d, hf.model.heads, hf.model.d / hf.model.heads], 'width and heads');
      phase(c.frames, 'head').forEach((f) => {
        const w = hf.blocks[f.meta.l].w[f.meta.h];
        c.eq(f.meta.src, src(r), 'needed position'); c.eq(f.meta.j, argmax(w[r]), 'most weight'); near(c, f.meta.wj, w[r][f.meta.j], 'its weight', 1e-4);
        let k = 0; for (let i = 5; i < 10; i++) if (argmax(w[i]) === src(i)) k++;
        c.eq(f.meta.hits, k, 'pointing count L' + f.meta.l + 'H' + f.meta.h);
      });
      const a = phase(c.frames, 'ablate')[0].meta, set = P.probeSet(task, a.count, 7);
      const acc = (off) => { let right = 0, tot = 0; for (const s of set) { const f = handForward(task, s.ids, { ablate: off }); for (let k = 0; k < 5; k++) { tot++; if (argmax(f.logits[5 + k]) === s.want[k]) right++; } } return right / tot; };
      near(c, a.base, acc(null), 'full accuracy', 1e-9);
      [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(([l, h], k) => near(c, a.drops[k], acc({ layer: l, head: h }), 'without L' + l + 'H' + h, 1e-9));
    },
  },
  'transformer-ffn': {
    random: (r) => ({ task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)], digits: Array.from({ length: 5 }, () => Math.floor(r() * 8)).join(''), pos: Math.floor(r() * 10), layer: r() < 0.5 ? '0' : '1' }),
    check(c) {
      const task = c.input.task, x = [...c.input.digits].map(Number), ids = x.concat([8], P.answer(task, x)).slice(0, -1), hf = handForward(task, ids), b = hf.blocks[+c.input.layer], k = c.input.pos, W = hf.W, p = 'blocks.' + c.input.layer + '.';
      const pre = b.pre[k], act = b.act[k];
      near(c, phase(c.frames, 'in')[0].meta.maxIn, Math.max(...b.ln2[k]), 'largest input', 1e-4);
      const w = phase(c.frames, 'widen')[0].meta;
      c.eq(w.neg, pre.filter((v) => v < 0).length, 'negative hidden units'); c.eq(w.big, argmax(pre), 'largest unit'); near(c, w.pb, Math.max(...pre), 'its value', 1e-4); near(c, w.pl, Math.min(...pre), 'most negative', 1e-4);
      const g = phase(c.frames, 'gelu')[0].meta;
      near(c, g.ab, gelu(g.pb), 'GELU of the largest', 1e-5); near(c, g.al, gelu(g.pl), 'GELU of the most negative', 1e-5);
      const nr = phase(c.frames, 'narrow')[0].meta;
      near(c, nr.outNorm, norm(b.ff[k]), 'output length', 1e-4); near(c, nr.midNorm, norm(b.mid[k]), 'stream length', 1e-4);
      const a = phase(c.frames, 'all')[0].meta, active = b.act.map((r) => r.filter((v) => v > 0.1).length);
      c.eq(a.act, Math.max(...active), 'most active units'); c.eq(a.most, active.indexOf(Math.max(...active)), 'most active position');
      c.eq(a.pff, W[p + 'ff1.W'].length * W[p + 'ff1.W'][0].length + W[p + 'ff1.b'].length + W[p + 'ff2.W'].length * W[p + 'ff2.W'][0].length + W[p + 'ff2.b'].length, 'feed-forward weights');
      c.eq(a.pat, 4 * (16 * 16 + 16), 'attention weights');
      void act;
    },
  },
  'transformer-residual-norm': {
    random: (r) => ({ task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)], digits: Array.from({ length: 5 }, () => Math.floor(r() * 8)).join(''), pos: Math.floor(r() * 10), layer: r() < 0.5 ? '0' : '1' }),
    check(c) {
      const task = c.input.task, x = [...c.input.digits].map(Number), ids = x.concat([8], P.answer(task, x)).slice(0, -1), hf = handForward(task, ids), l = +c.input.layer, b = hf.blocks[l], k = c.input.pos, W = hf.W, p = 'blocks.' + l + '.';
      const v = b.inp[k], d = v.length, mean = v.reduce((a, t) => a + t, 0) / d, va = v.reduce((a, t) => a + (t - mean) ** 2, 0) / d;
      const m0 = phase(c.frames, 'in')[0].meta; near(c, m0.mean, mean, 'mean', 1e-5); near(c, m0.sd, Math.sqrt(va), 'spread', 1e-5);
      const z = v.map((t) => (t - mean) / Math.sqrt(va + 1e-5)), zm = z.reduce((a, t) => a + t, 0) / d;
      const mn = phase(c.frames, 'norm')[0].meta; near(c, mn.zm, zm, 'normalised mean', 1e-5); near(c, mn.zs, Math.sqrt(z.reduce((a, t) => a + (t - zm) ** 2, 0) / d), 'normalised spread', 1e-5);
      const af = phase(c.frames, 'affine')[0].meta, g = W[p + 'ln1.gamma'], be = W[p + 'ln1.beta'];
      c.eq(af.gi, argmax(g), 'largest γ'); near(c, af.z, z[af.gi], 'normalised value', 1e-5); near(c, af.y, layerNorm(v, g, be)[af.gi], 'after scale and shift', 1e-5); near(c, af.y, af.z * af.gam + af.beta, 'the sum as said', 1e-5);
      const st = phase(c.frames, 'stream')[0].meta, norms = [hf.x0[k]].concat(hf.blocks.flatMap((bb) => [bb.mid[k], bb.out[k]])).map(norm), deltas = hf.blocks.flatMap((bb) => [norm(bb.att[k]), norm(bb.ff[k])]);
      vnear(c, st.norms, norms, 'stream lengths', 1e-4); vnear(c, st.deltas, deltas, 'added lengths', 1e-4); c.eq(st.dmax, argmax(deltas), 'biggest addition');
      near(c, phase(c.frames, 'why')[0].meta.grow, norms[norms.length - 1] / norms[0], 'growth', 1e-4);
    },
  },
  'transformer-block': {
    random: (r) => ({ task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)], digits: Array.from({ length: 5 }, () => Math.floor(r() * 8)).join(''), pos: Math.floor(r() * 10), layer: r() < 0.5 ? '0' : '1' }),
    check(c) {
      const task = c.input.task, x = [...c.input.digits].map(Number), full = x.concat([8], P.answer(task, x)), ids = full.slice(0, -1), hf = handForward(task, ids), l = +c.input.layer, b = hf.blocks[l], k = c.input.pos;
      near(c, phase(c.frames, 'in')[0].meta.len, norm(b.inp[k]), 'input length', 1e-4);
      near(c, phase(c.frames, 'ln1')[0].meta.len, norm(b.ln1[k]), 'after norm 1', 1e-4);
      const a = phase(c.frames, 'attn')[0].meta;
      c.eq([a.j0, a.j1], [argmax(b.w[0][k]), argmax(b.w[1][k])], 'most attended'); near(c, a.a0, Math.max(...b.w[0][k]), 'head 0 weight', 1e-4); near(c, a.a1, Math.max(...b.w[1][k]), 'head 1 weight', 1e-4); near(c, a.len, norm(b.att[k]), 'attention output length', 1e-4);
      const a1 = phase(c.frames, 'add1')[0].meta; near(c, a1.c, norm(b.mid[k]), 'after the first addition', 1e-4);
      near(c, phase(c.frames, 'ln2')[0].meta.len, norm(b.ln2[k]), 'after norm 2', 1e-4);
      near(c, phase(c.frames, 'ffn')[0].meta.len, norm(b.ff[k]), 'feed-forward length', 1e-4);
      const a2 = phase(c.frames, 'add2')[0].meta; near(c, a2.len, norm(b.out[k]), 'block output length', 1e-4); c.ok(a2.diff < 1e-5, 'the engine block agrees');
      const { W } = weightsOf(task), pp = 'blocks.' + l + '.'; let cnt = 0; for (const key of Object.keys(W)) if (key.startsWith(pp)) cnt += Array.isArray(W[key][0]) ? W[key].length * W[key][0].length : W[key].length;
      c.eq(a2.params, cnt, 'weights in the block');
      const o = phase(c.frames, 'output')[0].meta; c.eq(o.best, argmax(hf.probs[k]), 'predicted token'); near(c, o.p, Math.max(...hf.probs[k]), 'its chance', 1e-4); c.eq(o.want, full[k + 1] === undefined ? -1 : full[k + 1], 'the right token');
    },
  },
  'encoder-decoder': {
    random: (r) => ({ task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)], digits: Array.from({ length: 5 }, () => Math.floor(r() * 8)).join(''), head: r() < 0.5 ? '0' : '1', change: Math.floor(r() * 8) }),
    check(c) {
      const task = c.input.task, x = [...c.input.digits].map(Number), ids = x.concat([8], P.answer(task, x)).slice(0, -1), n = ids.length, hd = +c.input.head, { W, model } = weightsOf(task), g = W['blocks.0.ln1.gamma'], be = W['blocks.0.ln1.beta'];
      const rowsOf = (seq, from) => seq.map((t, i) => layerNorm(W['emb.W'][t].map((v, j) => v + W['pos.P'][from + i][j]), g, be));
      const dec = attnLayer(W, 'blocks.0.', rowsOf(ids, 0), model.heads, true), enc = attnLayer(W, 'blocks.0.', rowsOf(ids, 0), model.heads, false);
      const fut = enc.w[hd][0].slice(1).reduce((a, b) => a + b, 0);
      near(c, phase(c.frames, 'enc')[0].meta.fut, fut, 'weight on later positions', 1e-4);
      const ch = phase(c.frames, 'change')[0].meta, ids2 = ids.slice(0, n - 1).concat([ch.tok]);
      c.ok(ch.tok !== ids[n - 1], 'the last token really changes');
      const dec2 = attnLayer(W, 'blocks.0.', rowsOf(ids2, 0), model.heads, true), enc2 = attnLayer(W, 'blocks.0.', rowsOf(ids2, 0), model.heads, false);
      const mv = (a, b) => a.out.map((r, i) => Math.sqrt(r.reduce((s, v, j) => s + (v - b.out[i][j]) ** 2, 0)));
      c.ok(ch.maxD < 1e-5 && Math.max(...mv(dec, dec2).slice(0, n - 1)) < 1e-9, 'decoder outputs do not move'); near(c, ch.e0, mv(enc, enc2)[0], 'encoder position 0 moves', 1e-4);
      const cr = phase(c.frames, 'cross')[0].meta; c.eq([cr.ne, cr.nd], [5, n - 5], 'cross-attention shape');
    },
  },
  'transformer-train': {
    check(c) {
      const task = c.input.task, R = S.P12D.runs[task], Rd = P.run(task), probe = R.probe, full = probe.slice(0, 5).concat([8], P.answer(task, probe.slice(0, 5)));
      const src = (i) => full.slice(0, 5).indexOf(full[i + 1]);
      const fr = phase(c.frames, 'ckpt');
      c.eq(fr.length, R.ckpts.length, 'one frame per checkpoint');
      fr.forEach((f) => {
        const k = R.ckpts[f.meta.idx], maps = Rd.ckpts[f.meta.idx].maps;
        c.eq([f.meta.step, f.meta.exact, f.meta.out], [k.step, k.exact, k.out], 'checkpoint ' + k.step);
        if (k.step) near(c, f.meta.loss, R.loss[k.step - 1], 'loss at ' + k.step, 1e-4);
        c.eq(f.meta.right, k.out.filter((v, i) => v === full[6 + i]).length, 'probe digits right at ' + k.step);
        const hs = maps.map((m) => { let h = 0; for (let i = 5; i < 10; i++) if (argmax(m.weights[i]) === src(i)) h++; return h; });
        c.eq(f.meta.hits, hs, 'pointing counts at ' + k.step); c.eq(f.meta.top, hs.indexOf(Math.max(...hs)), 'sharpest head at ' + k.step);
      });
      const e = phase(c.frames, 'end')[0].meta, last = R.ckpts[R.ckpts.length - 1], half = R.ckpts.find((k) => k.exact >= 50);
      c.eq([e.step, e.exact, e.half, e.seen], [last.step, last.exact, half ? half.step : -1, Math.round((last.step * 32) / 1000) * 1000], 'the end');
      // the final weights, run by hand: the probe's attention and answer, and the score on the same 100 new sequences
      const hf = handForward(task, probe), lastMaps = Rd.ckpts[R.ckpts.length - 1].maps;
      lastMaps.forEach((m) => mnear(c, m.weights, hf.blocks[m.layer].w[m.head], 'final attention L' + m.layer + 'H' + m.head + ' from the weights', 0.0025));
      const greedy = (prefix) => { const seq = prefix.slice(); for (let t = 0; t < 5; t++) { const f = handForward(task, seq); seq.push(argmax(f.logits[seq.length - 1])); } return seq.slice(prefix.length); };
      c.eq(greedy(probe.slice(0, 5).concat([8])), last.out, 'final probe answer by hand');
      const gt = S.T.rng(99); let exact = 0;
      for (let i = 0; i < 100; i++) { const x = Array.from({ length: 5 }, () => gt.int(8)); if (greedy(x.concat([8])).join() === P.answer(task, x).join()) exact++; }
      c.eq(exact, last.exact, 'final exact count by hand');
      // the run itself, replayed from scratch with the engine for its first checkpoints (all of them with --full)
      const Rr = RUNS[task], { net, trainer, task: tk } = setupOf(S, Rr), until = FULL ? Rr.steps : Rr.keep[1], ids = probeInput(S, Rr);
      const checkAt = (step) => {
        const k = R.ckpts.find((x) => x.step === step); if (!k) return;
        const maps = probeAttn(S, net, ids), want = Rd.ckpts[R.ckpts.indexOf(k)].maps;
        maps.forEach((m, j) => mnear(c, m.weights, want[j].weights, 'replayed attention at ' + step + ' L' + m.layer + 'H' + m.head, 0.0025));
        c.eq(tk.check(net, 100, 99).exact, k.exact, 'replayed exact count at ' + step);
        c.eq(net.generate(probe.slice(0, 5).concat([8]), 5, { method: 'greedy' }).ids.slice(6), k.out, 'replayed probe answer at ' + step);
      };
      checkAt(0);
      for (let st = 1; st <= until; st++) { trainer.step(); near(c, trainer.history[st - 1].loss, R.loss[st - 1], 'replayed loss at step ' + st, 2e-3); checkAt(st); }
    },
  },
  'kv-cache-attention': {
    random: (r) => ({ task: ['reverse', 'copy', 'sort'][Math.floor(r() * 3)], digits: Array.from({ length: 5 }, () => Math.floor(r() * 8)).join(''), head: r() < 0.5 ? '0' : '1' }),
    check(c) {
      const task = c.input.task, x = [...c.input.digits].map(Number), hd = +c.input.head, seq = x.concat([8]);
      for (let t = 0; t < 5; t++) { const f = handForward(task, seq); seq.push(argmax(f.logits[seq.length - 1])); }
      const ans = seq.slice(6), hf = handForward(task, seq.slice(0, 10));
      c.eq(phase(c.frames, 'prefill')[0].meta.id, ans[0], 'first answer digit');
      phase(c.frames, 'step').forEach((f) => {
        c.eq([f.meta.id, f.meta.next], [ans[f.meta.s], ans[f.meta.s + 1]], 'tokens at step ' + f.meta.s);
        const row = hf.blocks[0].w[hd][f.meta.t];
        c.eq(f.meta.j, argmax(row), 'most attended at step ' + f.meta.s); near(c, f.meta.wj, Math.max(...row), 'its weight (cached row = full row)', 1e-4);
      });
      const e = phase(c.frames, 'end')[0].meta;
      c.eq(e.ans, ans, 'the answer'); c.eq(e.right, ans.filter((v, i) => v === P.answer(task, x)[i]).length, 'digits right');
      c.ok(e.worst < 1e-4, 'cache and full recompute agree: ' + e.worst);
      c.eq([e.total, e.totalNo], [6 + 4, 6 + 7 + 8 + 9 + 10], 'rows computed with and without the cache');
    },
  },
};

run('claims-p12', /^p12-/, checks, ALLOWED);
