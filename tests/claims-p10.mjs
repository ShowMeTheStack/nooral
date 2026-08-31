// Part 10, convolutional networks: the narration of every topic read back against an independent working-out.
// Convolutions, pooling and channel sums by hand-written loops; output sizes and receptive fields by their formulas;
// the trained networks run forward (and backward, for Grad-CAM) by a plain float64 implementation from the stored
// weights, decoded here with Buffer; saliency checked by finite differences on the class score; the residual stacks
// by a hand-written forward and backward pass; the stored training runs re-read, and their first steps re-run (the
// same seed must give the same numbers).
// Usage: node tests/claims-p10.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';
import { JOBS, setupOf } from './ref/p10_runs.mjs';

const P = S.P10;
// formula constants the narration may write as they are
const ALLOWED = [/\b2 · /g, /− 1\)/g, /\) \+ 1\b/g, /⌋ \+ 1\b/g, /\b3 × 3\b/g, /\b5 × 5\b/g, /\b2 × 2\b/g, /\(2 − 1\)/g, /\b1 \+ f′/g, /\b0…1\b/g, /\b1 is ink, 0 is paper\b/g, /\b(Six|six|Twelve|twelve) /g, /\b8 × 8\b/g, /\b6 × 6\b/g, /\b7 × 7\b/g];
const near = (c, a, b, msg, tol) => c.ok(Math.abs(a - b) <= (tol || 1e-6) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const f32 = (s) => { const b = Buffer.from(s, 'base64'); return Array.from(new Float32Array(b.buffer, b.byteOffset, b.length >> 2)); };
const u8 = (s) => Array.from(Buffer.from(s, 'base64'), (v) => v / 255);

// ---- a plain CNN in float64 -------------------------------------------------------------------------------------
/** x [C][H][W], w [O][C][k][k] flat, b [O]: a convolution with padding p, by loops */
const conv = (x, C, Hh, Wd, w, O, k, b, p, s = 1, d = 1) => {
  const Ho = Math.floor((Hh + 2 * p - d * (k - 1) - 1) / s) + 1, Wo = Math.floor((Wd + 2 * p - d * (k - 1) - 1) / s) + 1, out = new Array(O * Ho * Wo).fill(0);
  for (let o = 0; o < O; o++) for (let i = 0; i < Ho; i++) for (let j = 0; j < Wo; j++) {
    let t = b ? b[o] : 0;
    for (let c = 0; c < C; c++) for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) {
      const y = i * s - p + u * d, z = j * s - p + v * d;
      if (y >= 0 && y < Hh && z >= 0 && z < Wd) t += x[(c * Hh + y) * Wd + z] * w[((o * C + c) * k + u) * k + v];
    }
    out[(o * Ho + i) * Wo + j] = t;
  }
  return { v: out, Ho, Wo };
};
const relu = (v) => v.map((x) => Math.max(0, x));
const pool = (x, C, n, k, s, kind) => {
  const m = Math.floor((n - k) / s) + 1, out = [], arg = [];
  for (let c = 0; c < C; c++) for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
    let best = -Infinity, at = -1, sum = 0;
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const id = (c * n + i * s + a) * n + j * s + b, v = x[id]; sum += v; if (v > best) { best = v; at = id; } }
    out.push(kind === 'avg' ? sum / (k * k) : best); arg.push(at);
  }
  return { v: out, m, arg };
};
const nets = {};
const weights = (run) => { if (!nets[run]) { const J = S.P10D[run]; nets[run] = Object.fromEntries(Object.entries(J.weights).map(([n, w]) => [n, f32(w.d)])); } return nets[run]; };
/** the trained network's forward pass by hand: every stage, and the logits */
const forward = (run, px) => {
  const W = weights(run), K = S.P10D[run].classes.length;
  const c1 = conv(Array.from(px), 1, 16, 16, W['conv1.W'], 6, 5, W['conv1.b'], 2).v, r1 = relu(c1), p1 = pool(r1, 6, 16, 2, 2);
  const c2 = conv(p1.v, 6, 8, 8, W['conv2.W'], 12, 5, W['conv2.b'], 0).v, r2 = relu(c2), p2 = pool(r2, 12, 4, 2, 2);
  const fc0 = Array.from({ length: 32 }, (_, j) => W['fc.0.b'][j] + p2.v.reduce((s, x, i) => s + x * W['fc.0.W'][i * 32 + j], 0)), a0 = relu(fc0);
  const logits = Array.from({ length: K }, (_, j) => W['fc.1.b'][j] + a0.reduce((s, x, i) => s + x * W['fc.1.W'][i * K + j], 0));
  const m = Math.max(...logits), e = logits.map((z) => Math.exp(z - m)), t = e.reduce((a, b) => a + b, 0);
  return { c1, r1, p1, c2, r2, p2, fc0, a0, logits, probs: e.map((z) => z / t) };
};
/** ∂ logit c / ∂ relu2, by hand */
const backToRelu2 = (run, f, c) => {
  const W = weights(run), K = S.P10D[run].classes.length;
  const ga0 = Array.from({ length: 32 }, (_, i) => W['fc.1.W'][i * K + c]), gfc0 = ga0.map((g, i) => (f.fc0[i] > 0 ? g : 0));
  const gp2 = Array.from({ length: 48 }, (_, i) => gfc0.reduce((s, g, j) => s + g * W['fc.0.W'][i * 32 + j], 0));
  const g = new Array(192).fill(0);
  f.p2.arg.forEach((at, i) => { g[at] += gp2[i]; });
  return g;
};
const pick = (run, c, k) => P.pick(run, c, k);
const imageInput = (r) => { const set = r() < 0.7 ? 'digits' : 'shapes'; return { set, cls: Math.floor(r() * (set === 'digits' ? 10 : 4)), k: Math.floor(r() * 21) }; };
const replayed = {};

const checks = {
  'conv-slide': {
    random: (r) => ({ picture: Object.keys(P.PICTURES)[Math.floor(r() * 5)], kernel: Object.keys(P.KERNELS)[Math.floor(r() * 5)], bias: ['0', '-1', '1'][Math.floor(r() * 3)] }),
    check(c) {
      const pic = P.PICTURES[c.input.picture].px, K = P.KERNELS[c.input.kernel].w, b = +c.input.bias, n = pic.length, m = n - 2;
      const out = [];
      for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) { let s = b; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) s += pic[i + u][j + v] * K[u][v]; out.push(s); }
      const ws = phase(c.frames, 'window');
      c.eq(ws.length, m * m, 'one frame per place');
      ws.forEach((f) => {
        near(c, f.meta.value, out[f.meta.oy * m + f.meta.ox], 'output at ' + f.meta.oy + ',' + f.meta.ox, 1e-5);
        const terms = [];
        for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) { const x = pic[f.meta.oy + u][f.meta.ox + v], w = K[u][v]; if (x && w) terms.push([x, w]); }
        c.ok(terms.length === f.meta.terms.length && terms.every((t, i) => Math.abs(t[0] - f.meta.terms[i][0]) < 1e-6 && Math.abs(t[1] - f.meta.terms[i][1]) < 1e-6), 'the products listed at ' + f.meta.oy + ',' + f.meta.ox);
      });
      const e = phase(c.frames, 'end')[0].meta;
      near(c, e.max, Math.max(...out), 'largest', 1e-5); near(c, e.min, Math.min(...out), 'smallest', 1e-5);
      c.eq(e.neg, out.filter((v) => v < -1e-6).length, 'negatives');
    },
  },
  'conv-geometry': {
    random: (r) => ({ n: 3 + Math.floor(r() * 6), k: 1 + Math.floor(r() * 5), p: Math.floor(r() * 3), s: 1 + Math.floor(r() * 3), d: 1 + Math.floor(r() * 3) }),
    randomRuns: 14,
    check(c) {
      const { n, k, p, s, d } = c.input, Ho = Math.floor((n + 2 * p - d * (k - 1) - 1) / s) + 1, st = phase(c.frames, 'start')[0].meta;
      c.eq([st.Ho, st.Wo], [Ho, Ho], 'output size by the formula');
      const uses = new Array(n * n).fill(0);
      for (let i = 0; i < Ho; i++) for (let j = 0; j < Ho; j++) for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const y = i * s - p + a * d, x = j * s - p + b * d; if (y >= 0 && y < n && x >= 0 && x < n) uses[y * n + x]++; }
      phase(c.frames, 'place').forEach((f) => {
        let pad = 0;
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const y = f.meta.oi * s - p + a * d, x = f.meta.oj * s - p + b * d; if (y < 0 || x < 0 || y >= n || x >= n) pad++; }
        c.eq(f.meta.padTaps, pad, 'taps on padding at ' + f.meta.oi + ',' + f.meta.oj);
      });
      const e = phase(c.frames, 'end')[0].meta;
      c.eq(e.uses, uses, 'how often each pixel is read');
      c.eq([e.most, e.corner, e.zero], [Math.max(...uses), uses[0], uses.filter((u) => !u).length], 'busiest, corner, never read');
    },
  },
  'conv-pool': {
    random: (r) => ({ map: r() < 0.5 ? 'toy' : 'digit', kind: r() < 0.5 ? 'max' : 'avg', size: 2 + Math.floor(r() * 2), stride: 1 + Math.floor(r() * 3) }),
    check(c) {
      let M;
      if (c.input.map === 'toy') M = P.POOLMAPS.toy.make().M;
      else {
        // the trained network's first layer on the first validation 7, by hand: the busiest channel, its middle 8 × 8
        const f = forward('digits', pick('digits', 7, 0).px), tot = [0, 1, 2, 3, 4, 5].map((ch) => f.r1.slice(ch * 256, ch * 256 + 256).reduce((a, b) => a + b, 0)), ch = tot.indexOf(Math.max(...tot));
        M = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => f.r1[ch * 256 + (i + 4) * 16 + j + 4]));
      }
      const n = M.length, k = c.input.size, s = c.input.stride, out = pool(M.flat(), 1, n, k, s, c.input.kind);
      c.eq(phase(c.frames, 'start')[0].meta.m, out.m, 'pooled size');
      phase(c.frames, 'window').forEach((f) => {
        near(c, f.meta.val, out.v[f.meta.i * out.m + f.meta.j], 'window ' + f.meta.i + ',' + f.meta.j, 1e-4);
        c.eq([f.meta.r0, f.meta.c0], [f.meta.i * s, f.meta.j * s], 'window position');
      });
      const sh = M.map((r) => [0].concat(r.slice(0, n - 1))), out2 = pool(sh.flat(), 1, n, k, s, c.input.kind);
      c.eq(phase(c.frames, 'shift')[0].meta.same, out.v.filter((v, i) => Math.abs(v - out2.v[i]) < 1e-6).length, 'unchanged after the shift');
    },
  },
  'conv-channels': {
    random: (r) => ({ picture: Object.keys(P.CHANNELS.PICS)[Math.floor(r() * 3)], filter: Object.keys(P.CHANNELS.FILTERS)[Math.floor(r() * 4)] }),
    check(c) {
      const X = P.CHANNELS.PICS[c.input.picture].ch, Fl = P.CHANNELS.FILTERS[c.input.filter];
      const part = (ch, i, j) => { let t = 0; for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) t += X[ch][i + u][j + v] * Fl.w[ch][u][v]; return t; };
      phase(c.frames, 'channel').forEach((f) => near(c, f.meta.val, part(f.meta.c, 0, 0), 'channel ' + f.meta.c + ' at the first place', 1e-5));
      const pl = phase(c.frames, 'sum').map((f) => Object.assign({ i: 0, j: 0 }, f.meta)).concat(phase(c.frames, 'place').map((f) => f.meta));
      c.eq(pl.length, 9, 'nine places');
      pl.forEach((m) => {
        [0, 1, 2].forEach((ch) => near(c, m.parts[ch], part(ch, m.i, m.j), 'channel ' + ch + ' at ' + m.i + ',' + m.j, 1e-5));
        near(c, m.val, part(0, m.i, m.j) + part(1, m.i, m.j) + part(2, m.i, m.j) + Fl.b, 'total at ' + m.i + ',' + m.j, 1e-5);
      });
      c.eq(phase(c.frames, 'end')[0].meta.params, 16 * (27 + 1), 'a layer of 16 filters');
    },
  },
  'cnn-flow': {
    random: imageInput, randomRuns: 6,
    check(c) {
      const run = c.input.set, im = pick(run, c.input.cls, c.input.k), f = forward(run, im.px), m = (f2) => phase(c.frames, f2)[0].meta;
      near(c, m('conv1').max, Math.max(...f.c1), 'conv1 largest', 1e-4); near(c, m('conv1').min, Math.min(...f.c1), 'conv1 smallest', 1e-4);
      c.eq(m('relu1').zeroed, f.c1.filter((v) => v <= 0).length, 'zeroed by ReLU 1');
      c.eq(m('pool1').n, f.p1.v.length, 'pooled numbers');
      const s2 = Array.from({ length: 12 }, (_, ch) => f.r2.slice(ch * 16, ch * 16 + 16).reduce((a, b) => a + b, 0));
      c.eq(m('relu2').zeroed, f.c2.filter((v) => v <= 0).length, 'zeroed by ReLU 2');
      c.eq(m('relu2').top, s2.indexOf(Math.max(...s2)) + 1, 'busiest map');
      c.eq(m('relu2').dead, s2.filter((v) => v === 0).length, 'dark maps');
      c.eq(m('dense').on, f.a0.filter((v) => v > 0).length, 'active units');
      near(c, m('dense').max, Math.max(...f.a0), 'strongest unit', 1e-4);
      const best = f.probs.indexOf(Math.max(...f.probs));
      c.eq(m('out').best, best, 'the chosen class');
      near(c, m('out').pBest, f.probs[best], 'its chance', 1e-4);
      c.eq(m('conv2').field, 14, 'the field of a conv2 unit: 5, then +1 by the pool, then +4·2');
    },
  },
  'cnn-filters': {
    check(c) {
      const J = S.P10D[c.input.set], W0 = f32(J.snaps[0].conv1), nrm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      phase(c.frames, 'snap').forEach((f) => { const W = f32(J.snaps.find((s) => s.step === f.meta.step).conv1); near(c, f.meta.change, nrm(W.map((v, i) => v - W0[i])) / nrm(W0), 'change at step ' + f.meta.step, 1e-5); });
      const fin = weights(c.input.set)['conv1.W'];
      c.eq(f32(J.snaps[J.snaps.length - 1].conv1).map((v, i) => Math.abs(v - fin[i]) < 1e-12).every(Boolean), true, 'the last snapshot is the final weights');
      const im = pick(c.input.set, c.input.cls, 0), maps = conv(Array.from(im.px), 1, 16, 16, fin, 6, 5, null, 2).v;
      phase(c.frames, 'filter').forEach((f) => {
        const w = fin.slice((f.meta.f - 1) * 25, f.meta.f * 25);
        near(c, f.meta.pos, w.filter((v) => v > 0).reduce((a, b) => a + b, 0), 'positive sum', 1e-5);
        const mp = maps.slice((f.meta.f - 1) * 256, f.meta.f * 256), at = mp.indexOf(Math.max(...mp));
        c.eq([f.meta.row, f.meta.col], [Math.floor(at / 16) + 1, (at % 16) + 1], 'strongest place of filter ' + f.meta.f);
        near(c, f.meta.max, mp[at], 'its value', 1e-4);
      });
      c.eq(phase(c.frames, 'layer2')[0].meta.weights, 1800, 'second-layer weights');
    },
  },
  'cnn-field': {
    check(c) {
      let r = 1, j = 1;
      phase(c.frames, 'layer').forEach((f) => {
        const m = f.meta;
        r += (m.k - 1) * m.d * j; j *= m.s;
        c.eq([m.r, m.j], [r, j], 'the rule at layer ' + m.l);
        c.eq([m.lo, m.hi], [[Math.max(0, m.a), Math.max(0, m.a)], [Math.min(m.N - 1, m.b), Math.min(m.N - 1, m.b)]], 'the non-zero gradient spans the interval the rule maps back, at layer ' + m.l);
        c.eq(m.b - m.a + 1, m.r, 'that interval is the field’s width');
        const w = m.hi[1] - m.lo[1] + 1;
        if (!m.clipped) c.eq(w, m.r, 'the gradient’s field equals the rule’s at layer ' + m.l);
      });
    },
  },
  'cnn-train': {
    check(c) {
      const run = c.input.set, J = S.P10D[run], y = [...J.y].map(Number);
      phase(c.frames, 'snap').forEach((f) => {
        const sn = J.snaps.find((s) => s.step === f.meta.step), right = [...sn.preds].filter((p, i) => +p === y[i]).length;
        c.eq(f.meta.right, right, 'right at step ' + f.meta.step);
        near(c, f.meta.valLoss, J.val.find((v) => v.step === f.meta.step).valLoss, 'validation loss', 1e-12);
      });
      // the final predictions agree with the stored weights run by hand
      const fin = J.snaps[J.snaps.length - 1].preds, imgs = u8(J.images);
      let agree = 0;
      for (let i = 0; i < y.length; i += 15) { const f = forward(run, imgs.slice(i * 256, i * 256 + 256)); if (f.logits.indexOf(Math.max(...f.logits)) === +fin[i]) agree++; }
      c.eq(agree, Math.ceil(y.length / 15), 'stored predictions match the stored weights');
      // the first steps again, from scratch: the same seed gives the same losses
      if (!replayed[run]) { const { trainer } = setupOf(S, JOBS[run]); replayed[run] = [trainer.step().loss, trainer.step().loss, trainer.step().loss]; }
      replayed[run].forEach((l, i) => near(c, l, J.hist[i][0], 'replayed loss at step ' + (i + 1), 1e-4));
    },
  },
  'cnn-augment': {
    random: (r) => ({ cls: Math.floor(r() * 10), shift: Math.round(r() * 8) / 2, rotate: Math.floor(r() * 46), scale: Math.floor(r() * 31), seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const im = pick('digits', c.input.cls, 0), g = S.T.rng(c.input.seed), o = { shift: c.input.shift, rotate: (c.input.rotate * Math.PI) / 180, scale: c.input.scale / 100 };
      const copies = Array.from({ length: 8 }, () => S.DL.augmentImage(im.px, 16, 16, g, o));
      const ctr = (px) => { let s = 0, x = 0, y = 0; px.forEach((v, i) => { s += v; x += v * (i % 16); y += v * Math.floor(i / 16); }); return [x / s, y / s]; };
      const c0 = ctr(im.px), mv = copies.map((p) => { const q = ctr(p); return Math.hypot(q[0] - c0[0], q[1] - c0[1]); });
      const cp = phase(c.frames, 'copies')[0].meta;
      if (c.input.shift || c.input.rotate || c.input.scale) { c.eq(cp.far, mv.indexOf(Math.max(...mv)) + 1, 'copy moved most'); near(c, cp.move, Math.max(...mv), 'how far', 1e-6); }
      else near(c, cp.maxDiff, 0, 'no change without augmentation', 1e-9);
      if (!c.input.shift && !c.input.rotate && !c.input.scale) c.ok(copies.every((p) => p.every((v, i) => Math.abs(v - im.px[i]) < 1e-6)), 'copies equal the image');
      const off = S.P10D.augOff, on = S.P10D.augOn;
      phase(c.frames, 'curve').forEach((f) => { const a = off.val.find((v) => v.step === f.meta.step), b = on.val.find((v) => v.step === f.meta.step); c.eq([f.meta.offVal, f.meta.onVal], [a.valAcc, b.valAcc], 'stored accuracies at ' + f.meta.step); });
      c.eq([off.nTrain, on.nTrain], [40, 40], 'both runs train on 40 images');
      if (!replayed.aug) { const { trainer } = setupOf(S, JOBS.augOn); replayed.aug = [trainer.step().loss, trainer.step().loss]; }
      replayed.aug.forEach((l, i) => near(c, l, on.hist[i][0], 'augmented run replayed at step ' + (i + 1), 1e-4));
    },
  },
  'cnn-residual': {
    random: (r) => ({ depth: 2 + Math.floor(r() * 25), init: ['small', 'he', 'large'][Math.floor(r() * 3)], seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const L = c.input.depth, sc = { small: 0.5, he: 1, large: 1.5 }[c.input.init];
      for (const residual of [false, true]) {
        const st = P.stack({ depth: L, scale: sc, seed: c.input.seed, residual });
        // the same stack by hand, in float64: forward, then backward through every layer
        const hs = [st.x.slice()], pre = [];
        for (let l = 0; l < L; l++) { const z = conv(hs[l], 4, 8, 8, st.W[l], 4, 3, null, 1).v; pre.push(z); const f = relu(z); hs.push(residual ? hs[l].map((v, i) => v + f[i] / Math.sqrt(L)) : f); }
        const gs = new Array(L + 1);
        gs[L] = st.r.slice();
        for (let l = L - 1; l >= 0; l--) {
          const gf = gs[l + 1].map((g, i) => (pre[l][i] > 0 ? g * (residual ? 1 / Math.sqrt(L) : 1) : 0));
          // the convolution's input gradient: the kernel flipped, by loops
          const gx = new Array(256).fill(0);
          for (let o = 0; o < 4; o++) for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { const g = gf[(o * 8 + i) * 8 + j]; if (!g) continue; for (let ch = 0; ch < 4; ch++) for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) { const y = i - 1 + u, x = j - 1 + v; if (y >= 0 && y < 8 && x >= 0 && x < 8) gx[(ch * 8 + y) * 8 + x] += g * st.W[l][((o * 4 + ch) * 3 + u) * 3 + v]; } }
          gs[l] = residual ? gx.map((v, i) => v + gs[l + 1][i]) : gx;
        }
        const rms = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);
        const tag = residual ? 'residual' : 'plain';
        st.grad.forEach((g, l) => c.ok(Math.abs(g - rms(gs[l])) <= 2e-3 * rms(gs[l]) + 1e-12, tag + ' gradient at layer ' + l + ' (relative): ' + g + ' vs ' + rms(gs[l])));
        st.act.forEach((a, l) => c.ok(Math.abs(a - rms(hs[l])) <= 2e-3 * rms(hs[l]) + 1e-12, tag + ' activation at layer ' + l));
      }
      const e = phase(c.frames, 'end')[0].meta, rb = phase(c.frames, 'resB')[0].meta, pb = phase(c.frames, 'plainB')[0].meta;
      near(c, e.gain, rb.g0 / pb.g0, 'gain', 1e-9);
    },
  },
  'cnn-saliency': {
    random: (r) => Object.assign(imageInput(r), { target: ['pred', 'second', 'true'][Math.floor(r() * 3)], blank: 4 + Math.floor(r() * 20) }),
    randomRuns: 5,
    check(c) {
      const run = c.input.set, im = pick(run, c.input.cls, c.input.k), st = phase(c.frames, 'start')[0].meta, cls = st.c, f = forward(run, im.px);
      near(c, st.s0, f.logits[cls], 'the score', 1e-4);
      const sal = S.DL.saliency(S.P10.net(run), im.px, cls), raw = sal.raw;
      // finite differences on the score (float64, by hand) for the strongest pixels and a spread of others. Max pooling
      // over ReLU maps has exact ties in blank areas (every value there is the bias), where the score has a corner and
      // no derivative: a pixel is checked only where the two one-sided differences agree.
      const score = (px) => forward(run, px).logits[cls], e = 1e-6, s0 = score(Array.from(im.px));
      const order = raw.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
      let checked = 0;
      for (const i of order.slice(0, 6).concat([0, 40, 100, 136, 200, 255])) {
        const a = Array.from(im.px), b = a.slice(); a[i] += e; b[i] -= e;
        const up = (score(a) - s0) / e, dn = (s0 - score(b)) / e;
        if (Math.abs(up - dn) > 1e-4 * Math.max(1, Math.abs(up))) continue;
        checked++;
        c.ok(Math.abs((up + dn) / 2 - raw[i]) <= 1e-3 * Math.max(1, Math.abs(raw[i])), 'saliency at pixel ' + i + ': engine ' + raw[i] + ', finite difference ' + (up + dn) / 2);
      }
      c.ok(checked >= 3, 'at least three pixels checked by finite differences (' + checked + ')');
      const rw = phase(c.frames, 'raw')[0].meta;
      near(c, rw.max, Math.max(...raw.map(Math.abs)), 'largest gradient', 1e-6);
      c.eq(rw.pos, raw.filter((v) => v > 0).length, 'pixels pushing the score up');
      const t = phase(c.frames, 'test')[0].meta, ink = Array.from(im.px.keys()).filter((i) => im.px[i] > 0.2), n = Math.min(c.input.blank, ink.length);
      const marked = ink.slice().sort((a, b) => raw[b] - raw[a]).slice(0, n), A = Array.from(im.px);
      for (const i of marked) A[i] = 0;
      near(c, t.sA, forward(run, A).logits[cls], 'score with the marked pixels blanked', 1e-4);
    },
  },
  'cnn-gradcam': {
    random: (r) => Object.assign(imageInput(r), { target: ['pred', 'second', 'true'][Math.floor(r() * 3)] }),
    randomRuns: 5,
    check(c) {
      const run = c.input.set, im = pick(run, c.input.cls, c.input.k), cls = phase(c.frames, 'start')[0].meta.c, f = forward(run, im.px);
      const g = backToRelu2(run, f, cls), alpha = Array.from({ length: 12 }, (_, k) => g.slice(k * 16, k * 16 + 16).reduce((a, b) => a + b, 0) / 16);
      const raw = Array.from({ length: 16 }, (_, i) => Math.max(0, alpha.reduce((s, a, k) => s + a * f.r2[k * 16 + i], 0)));
      const w = phase(c.frames, 'weights')[0].meta, best = alpha.indexOf(Math.max(...alpha));
      c.eq(w.best, best + 1, 'the map that counts most');
      near(c, w.wBest, alpha[best], 'its weight', 1e-4);
      c.eq(w.negs, alpha.filter((v) => v < 0).length, 'negative weights');
      const s = phase(c.frames, 'sum')[0].meta, pk = raw.indexOf(Math.max(...raw));
      near(c, s.max, raw[pk], 'largest cell', 1e-4);
      c.eq([s.row, s.col], [Math.floor(pk / 4) + 1, (pk % 4) + 1], 'where it is');
      c.eq(s.zero, raw.filter((v) => v === 0).length, 'cells cut to zero');
      // the share of the stretched map that falls on ink, by hand (each cell covers a 4 × 4 block)
      const mx = Math.max(...raw);
      if (mx > 0) { let on = 0, tot = 0; for (let i = 0; i < 256; i++) { const v = raw[Math.floor(i / 64) * 4 + Math.floor((i % 16) / 4)] / mx; on += v * im.px[i]; tot += v; } near(c, phase(c.frames, 'overlay')[0].meta.share, on / tot, 'share on ink', 1e-4); }
    },
  },
};

run('claims-p10', /^p10-/, checks, ALLOWED);
