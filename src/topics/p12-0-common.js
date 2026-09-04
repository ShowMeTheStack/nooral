// Part 12, transformers: the families, the trained tiny transformers (rebuilt from the weights kept in
// p12-0-data.js), a forward pass that keeps every intermediate vector of every block, and the pieces of picture the
// topics share (a matrix as heat cells with names, a token row, a curve).
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'transformers', order, title, blurb });
  fam('p12-attn', 10, 'Tokens, positions and attention', 'How a transformer reads: tokens become vectors, positions are added, and scaled dot-product attention lets every position gather from the others, with masks deciding who may look at whom and several heads asking different questions.');
  fam('p12-block', 20, 'Inside the transformer block', 'The feed-forward layer, the residual stream and layer normalisation, the whole block traced with real numbers, and how the same block makes an encoder or a decoder.');
  fam('p12-train', 30, 'A tiny transformer at work', 'A transformer of under five thousand numbers learns to reverse, copy and sort digits, its attention patterns forming as it trains, and the key–value cache that makes generating each new token cheap.');

  const P = (S.P12 = {});
  const H = () => S.Kit.ml;
  const cache = new Map();
  P.once = (k, f) => { if (!cache.has(k)) cache.set(k, f()); return cache.get(k); };
  P.SEP = 8;
  P.tok = (id) => (id === P.SEP ? '=' : String(id));
  /** a token for the narration, a digit quoted through the frame quoter */
  P.qtok = (q, id) => (id === P.SEP ? '=' : q.n(id));
  P.TASKS = { reverse: 'reverse', copy: 'copy', sort: 'sort' };
  P.answer = (task, x) => (task === 'reverse' ? x.slice().reverse() : task === 'sort' ? x.slice().sort((a, b) => a - b) : x.slice());
  P.digitsCheck = (v) => (/^[0-7]{5}$/.test(v) ? null : 'five digits from 0 to 7');

  const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/', LUT = new Int16Array(128).fill(-1);
  for (let i = 0; i < 64; i++) LUT[B64.charCodeAt(i)] = i;
  /** base64 to bytes, written out so it runs anywhere (page, worker, node) */
  const bytes = (b64) => {
    const n = b64.length, pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0, out = new Uint8Array((n / 4) * 3 - pad);
    let o = 0;
    for (let i = 0; i < n; i += 4) {
      const v = (LUT[b64.charCodeAt(i)] << 18) | (LUT[b64.charCodeAt(i + 1)] << 12) | ((LUT[b64.charCodeAt(i + 2)] & 63) << 6) | (LUT[b64.charCodeAt(i + 3)] & 63);
      if (o < out.length) out[o++] = (v >> 16) & 255; if (o < out.length) out[o++] = (v >> 8) & 255; if (o < out.length) out[o++] = v & 255;
    }
    return out;
  };
  P.f32 = (b64) => { const u = bytes(b64); return new Float32Array(u.buffer, u.byteOffset, u.byteLength / 4); };
  /** a shipped run with its checkpoints' attention decoded: ckpt.maps = [{layer, head, weights}] */
  P.run = (name) => P.once('run:' + name, () => {
    const r = S.P12D.runs[name], n = r.probe.length, L = r.model.layers, Hh = r.model.heads;
    const ckpts = r.ckpts.map((c) => { const u = bytes(c.attn), maps = []; for (let l = 0; l < L; l++) for (let h = 0; h < Hh; h++) { const base = (l * Hh + h) * n * n; maps.push({ layer: l, head: h, weights: Array.from({ length: n }, (_, i) => Array.from(u.subarray(base + i * n, base + (i + 1) * n), (v) => v / 255)) }); } return Object.assign({}, c, { maps }); });
    return Object.assign({}, r, { ckpts });
  });
  /** the trained model of a run, rebuilt from its float32 weights */
  P.net = (name) => P.once('net:' + name, () => {
    const r = S.P12D.runs[name], net = S.DL.build('gpt', r.model), w = P.f32(r.weights);
    let at = 0;
    for (const [, p] of net.namedParameters()) { p.data.set(w.subarray(at, at + p.size)); at += p.size; }
    net.eval();
    return net;
  });
  /** the full input a model reads for digits x on a task: x, the separator, the answer but its last digit */
  P.seqOf = (task, x) => x.concat([P.SEP], P.answer(task, x)).slice(0, -1);
  const rows = (t) => { const a = t.toArray(); return a; };

  /**
   * every intermediate of a forward pass, as plain arrays for one sequence:
   * {tok, pos, x0, blocks: [{inp, ln1, q, k, v ([head][t][dk]), scores, w ([head][t][t]), heads ([t][d] joined), att, mid, ln2, pre, act, ff, out}], lnf, logits, probs}
   */
  P.trace = (net, ids) => S.T.noGrad(() => {
    const T = S.T, n = ids.length, x = T.tensor([ids]);
    const tok = T.embedding(net.emb.W, ids), pos = T.slice(net.pos.P, [0, n], null);
    let h = T.add(net.emb.forward(x), T.slice(net.pos.P, [0, n], null));
    const out = { ids: ids.slice(), tok: rows(tok), pos: rows(pos), x0: rows(h)[0], blocks: [] };
    for (const blk of net.blocks) {
      const a = blk.attn, b = { inp: rows(h)[0] }, l1 = blk.ln1.forward(h);
      b.ln1 = rows(l1)[0];
      const y = a.forward(l1, { causal: true });
      b.q = rows(a.q)[0]; b.k = rows(a.k)[0]; b.v = rows(a.v)[0]; b.scores = rows(a.scores)[0]; b.w = rows(a.weights)[0];
      b.att = rows(y)[0];
      h = T.add(h, y); b.mid = rows(h)[0];
      const l2 = blk.ln2.forward(h), pre = blk.ff1.forward(l2), act = T[blk.act](pre), f = blk.ff2.forward(act);
      b.ln2 = rows(l2)[0]; b.pre = rows(pre)[0]; b.act = rows(act)[0]; b.ff = rows(f)[0];
      h = T.add(h, f); b.out = rows(h)[0];
      out.blocks.push(b);
    }
    const lf = net.lnf.forward(h), lg = T.matmul(lf, T.transpose(net.emb.W)), pr = T.softmax(lg, -1);
    out.lnf = rows(lf)[0]; out.logits = rows(lg)[0]; out.probs = rows(pr)[0];
    return out;
  });
  P.argmax = (v) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);
  P.norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  P.dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  P.cos = (a, b) => P.dot(a, b) / (P.norm(a) * P.norm(b));
  P.r2 = (v) => Math.round(v * 100) / 100;
  /** a vector quoted through the frame quoter: [0.25, −0.10] */
  P.qvec = (q, v, d) => '(' + v.map((x) => q(x, d === undefined ? 2 : d)).join(', ') + ')';
  P.vec = (v, d) => '(' + v.map((x) => H().fx(x, d === undefined ? 2 : d)).join(', ') + ')';

  /** a matrix as heat cells: o {rows, cols, cw, ch, scheme, max, d, title, cls(i, j), size, fmt} -> {items, w, h, cell} */
  P.mat = function (prefix, x, y, M, o) {
    o = o || {};
    const m = M[0].length, max = o.max || Math.max(1e-9, ...M.flat().map(Math.abs));
    return S.Kit.heat.grid({ prefix, x, y, cw: o.cw || 40, ch: o.ch || 24, vals: M, scheme: o.scheme || 'div', max, fmt: o.fmt === undefined ? (v) => H().fx(v, o.d === undefined ? 2 : o.d) : o.fmt, rows: o.rows === false ? null : o.rows || M.map((_, i) => String(i)), cols: o.cols === false ? null : o.cols || Array.from({ length: m }, (_, j) => String(j)), size: o.size || 12, labelSize: o.labelSize || 12, cls: o.cls, title: o.title, titleCls: o.titleCls || 'strong', titleSize: o.titleSize || 13, gap: o.gap === undefined ? 2 : o.gap, rowCls: o.rowCls, colCls: o.colCls, colsAt: o.colsAt });
  };
  /** an operator sign between matrices */
  P.sign = (key, x, y, s) => H().text(key, x, y, s, 'p12op', 20, 'middle');
  /** a row of tokens, centred on cx when given */
  P.strip = function (prefix, x, y, toks, o) {
    o = o || {};
    const st = S.Kit.tokens.strip({ prefix, x, y, width: o.width || 760, tokens: toks, size: o.size || 14, gap: o.gap === undefined ? 6 : o.gap, minW: o.minW || 30, sub: o.sub, cls: o.cls });
    return st;
  };
  /** the softmax of a row (plain formula, for narration and pictures) */
  P.softmax = (row) => { const m = Math.max(...row.filter((v) => v > -1e8)), e = row.map((v) => (v < -1e8 ? 0 : Math.exp(v - m))), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
  P.transpose = (M) => M[0].map((_, j) => M.map((r) => r[j]));
  P.matmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
  /** a seeded matrix with entries rounded to one decimal, so hand arithmetic is easy to follow */
  P.randMat = (rng, n, m, sd) => Array.from({ length: n }, () => Array.from({ length: m }, () => Math.round(rng.normal() * (sd || 1) * 10) / 10));
  P.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');
  /** the input position a head should read for each answer row: where the next answer digit sits among the digits */
  P.sourceOf = (seq, L, i) => { const next = seq[i + 1] === undefined ? null : seq[i + 1]; return next === null ? -1 : seq.slice(0, L).indexOf(next); };
})(globalThis.Nooral = globalThis.Nooral || {});
