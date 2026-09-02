// Part 11, sequences and embeddings: the narration of every topic read back against an independent working-out.
// word2vec is trained again from scratch with the engine (the same seed must give the same vectors), and every
// cosine, neighbour, analogy, projection share and pair loss is recomputed with plain loops from the raw vectors.
// The recurrent models' weights are decoded straight from the base64 (tests/claims-p11-hand.mjs) and run by hand:
// every hidden state, gate, probability, count, loss and greedy output quoted must match. The gradients through
// time are recomputed by a hand-written backward recursion. Finally each shipped training run is replayed with the
// engine (whole, or its first steps for the long ones) and must match what src/topics/p11-0-data.js keeps.
// Usage: node tests/claims-p11.mjs [topic id prefix ...] [--no-replay]
import { S, run, phase } from './claims-lib-p06-08.mjs';
import { harness } from './load.mjs';
import * as M from './claims-p11-hand.mjs';
import { JOBS, runJob } from './ref/p11_runs.mjs';

const P = S.P11, D = S.P11D;
const ALLOWED = [/word2vec/g, /\ba 1\b/g, /\bits 1\b/g, /times [01]\b/g, /\(1 for the same direction, 0 for unrelated, −1 for opposite\)/g, /e\^/g, /between 0 \(shut\) and 1 \(open\)/g, /\bon a 0\b/g];
const near = (c, a, b, msg, tol) => c.ok(Math.abs(a - b) <= (tol || 1e-6) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const vnear = (c, a, b, msg, tol) => c.ok(a && b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= (tol || 1e-6) * Math.max(1, Math.abs(b[i]))), msg + ': got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b));
const f32 = 2e-5, hand = 2e-4;

// ---- word2vec, trained again ---------------------------------------------------------------------------------------
const replays = new Map();
function w2v(o) {
  o = Object.assign({}, P.W2V_DEF, o);
  const key = JSON.stringify(o);
  if (replays.has(key)) return replays.get(key);
  const text = P.corpus(o.sentences, 1), w = new S.DL.Word2Vec(text, { dim: o.dim, window: o.window, negatives: o.negatives, epochs: o.epochs, minCount: 2, lr: o.lr, subsample: 0, seed: o.seed });
  const W0 = Array.from(w.W), epochs = [{ W: W0, loss: null }];
  while (w.epoch < o.epochs) { const e = w.epoch; while (w.epoch === e) w.step(Math.max(1, w.cur.length - w.pos)); epochs.push({ W: Array.from(w.W), loss: w.losses[w.losses.length - 1] }); }
  // the vocabulary counted again by hand: words seen at least twice, by count, then alphabetically
  const cnt = new Map();
  for (const x of text.toLowerCase().match(/[a-z]+/g)) cnt.set(x, (cnt.get(x) || 0) + 1);
  const vocab = [...cnt.entries()].filter(([, n]) => n >= 2).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map((x) => x[0]);
  const d = o.dim, V = vocab.length;
  // rows with the mean taken away, from the raw vectors
  const rowsOf = (W) => { const mean = new Array(d).fill(0); for (let i = 0; i < V; i++) for (let j = 0; j < d; j++) mean[j] += W[i * d + j] / V; return vocab.map((_, i) => Array.from({ length: d }, (__, j) => W[i * d + j] - mean[j])); };
  const r = { o, text, vocab, index: new Map(vocab.map((x, i) => [x, i])), W0, epochs, rows: epochs.map((e) => rowsOf(e.W)), C: Array.from(w.C), d, V, engine: w, words: w.ids.length };
  r.final = r.rows[r.rows.length - 1];
  r.nearest = (x, k, rows) => { rows = rows || r.final; const q = rows[r.index.get(x)]; return vocab.map((y, i) => ({ label: y, score: M.cos(q, rows[i]) })).filter((z) => z.label !== x).sort((a, b) => b.score - a.score).slice(0, k); };
  replays.set(key, r);
  return r;
}
const kinds = Object.keys({ people: 1, animals: 1, food: 1, colours: 1, things: 1, places: 1, numbers: 1 });

const checks = {
  'one-hot-embeddings': {
    random: (rnd) => { const pool = P.mapWords(), k = 3 + Math.floor(rnd() * 6), out = []; while (out.length < k) { const x = pool[Math.floor(rnd() * pool.length)]; if (!out.includes(x)) out.push(x); } return { words: out, dim: ['4', '8', '16'][Math.floor(rnd() * 3)] }; },
    randomRuns: 4,
    check(c) {
      const r = w2v({ dim: +c.input.dim }), words = c.input.words, ids = words.map((x) => r.index.get(x));
      c.eq(r.V, r.engine.vocab.length, 'the hand-counted vocabulary has the engine’s size');
      const v = phase(c.frames, 'vocab')[0].meta;
      c.eq(v.V, r.V, 'vocabulary size'); c.eq(v.ids, ids, 'ids from the hand-counted order');
      near(c, phase(c.frames, 'onesim')[0].meta.dist, Math.SQRT2, 'one-hot distance');
      c.eq(phase(c.frames, 'onesim')[0].meta.offMax, 0, 'every off-diagonal dot product is zero');
      const lk = phase(c.frames, 'lookup')[0].meta, oneHot = r.vocab.map((_, i) => (i === ids[0] ? 1 : 0));
      // the product written out: one-hot row times the table
      const prod = r.final[0].map((_, j) => oneHot.reduce((s, h, i) => s + h * r.final[i][j], 0));
      vnear(c, lk.row0, prod.slice(0, lk.row0.length), 'one-hot · E', f32);
      const cs = phase(c.frames, 'cos')[0].meta, E = ids.map((i) => r.final[i]);
      const Cm = E.map((a) => E.map((b) => M.cos(a, b)));
      let best = [0, 1], worst = [0, 1];
      for (let i = 0; i < E.length; i++) for (let j = i + 1; j < E.length; j++) { if (Cm[i][j] > Cm[best[0]][best[1]]) best = [i, j]; if (Cm[i][j] < Cm[worst[0]][worst[1]]) worst = [i, j]; }
      c.eq(cs.best, best, 'most alike pair'); c.eq(cs.worst, worst, 'least alike pair');
      near(c, cs.cBest, Cm[best[0]][best[1]], 'best cosine', f32); near(c, cs.cWorst, Cm[worst[0]][worst[1]], 'worst cosine', f32);
      c.eq(cs.nb0, M.argmax(Cm[0].map((x, j) => (j ? x : -9))), 'closest to the first word');
      c.eq(phase(c.frames, 'size')[0].meta.total, r.V * r.d, 'numbers in the table');
    },
  },
  'word2vec-train': {
    random: (rnd) => ({ dim: ['4', '8', '16'][Math.floor(rnd() * 3)], window: 1 + Math.floor(rnd() * 3), negatives: 1 + Math.floor(rnd() * 8), epochs: 1 + Math.floor(rnd() * 6), seed: 1 + Math.floor(rnd() * 50), probe: ['cat', 'king', 'bread', 'red', 'three', 'river', 'boat', 'tea'][Math.floor(rnd() * 8)] }),
    randomRuns: 3,
    check(c) {
      const inp = c.input, r = w2v({ dim: +inp.dim, window: inp.window, negatives: inp.negatives, epochs: inp.epochs, seed: inp.seed }), d = r.d, K = inp.negatives;
      const p0 = phase(c.frames, 'pairs')[0].meta;
      c.eq([p0.nWords, p0.V], [r.words, r.V], 'corpus and vocabulary sizes');
      // the pair at the start: every context vector is zero, so every score is 1/2 and the loss (K + 1) ln 2
      const a = phase(c.frames, 'pair0')[0].meta;
      near(c, a.pos, 0.5, 'starting score'); near(c, a.loss, (K + 1) * Math.LN2, 'starting loss');
      // the same pair, trained: the scores and loss by formula from the raw vectors
      const b = phase(c.frames, 'pair1')[0].meta, W = r.epochs[r.epochs.length - 1].W, row = (A, i) => A.slice(i * d, (i + 1) * d);
      const ci = r.index.get(inp.probe), sent = r.text.split('. ').map((x) => x.replace(/\.$/, '').split(' '));
      const v = row(W, ci), negs = a.negs, pos = phase(c.frames, 'pairs')[0].say.match(/like \(([a-z]+), ([a-z]+)\)/), oi = r.index.get(pos[2]);
      c.ok(sent.some((t) => { const k = t.indexOf(inp.probe); return k >= 0 && t.slice(Math.max(0, k - inp.window), k + inp.window + 1).includes(pos[2]); }), 'the context word is within the window of the probe');
      const sp = M.sig(M.dot(row(r.C, oi), v)), sn = negs.map((k) => M.sig(M.dot(row(r.C, k), v)));
      near(c, b.pos, sp, 'trained score', f32); vnear(c, b.neg, sn, 'trained negative scores', f32);
      near(c, b.loss, -Math.log(sp) - sn.reduce((s, x) => s + Math.log(1 - x), 0), 'trained pair loss', f32);
      // epoch by epoch: the loss and how close words of a kind are
      const words = P.mapWords(), idx = words.map((x) => r.index.get(x)), kindOf = (x) => kinds.find((k) => P.CATS[k].includes(x));
      phase(c.frames, 'epoch').forEach((f) => {
        const rows = r.rows[f.meta.epoch];
        if (f.meta.epoch) near(c, f.meta.loss, r.epochs[f.meta.epoch].loss, 'loss of epoch ' + f.meta.epoch, 1e-9);
        let sIn = 0, nIn = 0, sOut = 0, nOut = 0;
        for (let i = 0; i < words.length; i++) for (let j = i + 1; j < words.length; j++) { const x = M.cos(rows[idx[i]], rows[idx[j]]); if (kindOf(words[i]) === kindOf(words[j])) { sIn += x; nIn++; } else { sOut += x; nOut++; } }
        near(c, f.meta.within, sIn / nIn, 'within-kind cosine at epoch ' + f.meta.epoch, f32); near(c, f.meta.between, sOut / nOut, 'between-kind cosine at epoch ' + f.meta.epoch, f32);
      });
      const nb = phase(c.frames, 'nearest')[0].meta.nb, mine = r.nearest(inp.probe, 5);
      c.eq(nb.map((x) => x.label), mine.map((x) => x.label), 'nearest words'); vnear(c, nb.map((x) => x.score), mine.map((x) => x.score), 'their cosines', f32);
    },
  },
  'word2vec-map': {
    check(c) {
      const inp = c.input, r = w2v({ seed: inp.seed });
      const mp = phase(c.frames, 'map')[0].meta, words = phase(c.frames, 'map')[0].items.filter((it) => it.key.startsWith('wm:m:d:')).map((it) => it.key.slice(7));
      // the share of the spread on two axes: the top two eigenvalues of the covariance of these rows over the total
      const R = words.map((x) => r.final[r.index.get(x)]), mean = R[0].map((_, j) => R.reduce((s, x) => s + x[j], 0) / R.length);
      const X = R.map((x) => x.map((v, j) => v - mean[j])), Cv = mean.map((_, i) => mean.map((__, j) => X.reduce((s, x) => s + x[i] * x[j], 0) / X.length)), ev = M.eigSym(Cv), tot = ev.reduce((a, b) => a + b, 0);
      near(c, mp.share, (ev[0] + ev[1]) / tot, 'share of spread on the map', 1e-4);
      const pr = phase(c.frames, 'probe')[0].meta, mine = r.nearest(inp.probe, inp.k);
      c.eq(pr.others, r.V - 1, 'words compared');
      c.eq(pr.nb.map((x) => x.label), mine.map((x) => x.label), 'neighbours'); vnear(c, pr.nb.map((x) => x.score), mine.map((x) => x.score), 'neighbour cosines', f32);
      const an = phase(c.frames, 'answer')[0].meta, [a, b, cc] = phase(c.frames, 'arrow')[0].meta.abc, Rw = (x) => r.final[r.index.get(x)];
      const qv = Rw(b).map((v, j) => v - Rw(a)[j] + Rw(cc)[j]);
      const ranked = r.vocab.filter((x) => ![a, b, cc].includes(x)).map((x) => ({ label: x, score: M.cos(qv, Rw(x)) })).sort((u, v) => v.score - u.score).slice(0, 3);
      c.eq(an.answers.map((x) => x.label), ranked.map((x) => x.label), 'analogy answers'); vnear(c, an.answers.map((x) => x.score), ranked.map((x) => x.score), 'analogy cosines', f32);
      c.eq(an.won, ranked[0].label === an.want, 'whether the expected word won'); c.eq(an.rank, ranked.findIndex((x) => x.label === an.want), 'rank of the expected word');
    },
  },
  'rnn-unrolled': {
    random: (rnd) => ({ bits: Array.from({ length: 3 + Math.floor(rnd() * 6) }, () => (rnd() < 0.5 ? '1' : '0')).join('') }),
    check(c) {
      const job = D.rnnCount, w = M.weights(job), bits = [...c.input.bits].map(Number), st = M.runRecurrent(w, 'rnn', bits, false);
      const wf = phase(c.frames, 'weights')[0].meta;
      c.eq(wf.nPar, w.size, 'parameter count'); c.eq(wf.acc, job.evals[job.evals.length - 1].accuracy, 'held-out accuracy'); c.eq(wf.steps, JOBS.rnnCount.steps, 'training steps');
      let hp = [0, 0, 0, 0];
      phase(c.frames, 'step').forEach((f) => {
        const s = st[f.meta.t - 1];
        vnear(c, f.meta.h, s.h, 'h at step ' + f.meta.t, hand);
        vnear(c, f.meta.hw, M.vm(hp, w['cell.Wh']), 'W_h part at step ' + f.meta.t, hand);
        vnear(c, f.meta.pre, s.h.map(Math.atanh), 'sum before tanh at step ' + f.meta.t, 1e-3);
        c.eq(f.meta.count, M.argmax(s.p), 'count at step ' + f.meta.t); near(c, f.meta.pk, Math.max(...s.p), 'its probability', hand);
        hp = s.h;
      });
      const truth = bits.map((_, t) => bits.slice(0, t + 1).reduce((a, b) => a + b, 0));
      c.eq(phase(c.frames, 'done')[0].meta.right, st.filter((s, t) => M.argmax(s.p) === truth[t]).length, 'right counts');
    },
  },
  'rnn-gradients': {
    random: (rnd) => ({ scale: Math.round((0.5 + rnd()) * 20) / 20, act: rnd() < 0.5 ? 'tanh' : 'linear', steps: 10 + Math.floor(rnd() * 31), seed: 1 + Math.floor(rnd() * 99) }),
    check(c) {
      // the same random cell, then the gradient by the backward recursion written out
      const gradNorms = (o) => {
        const g = S.T.rng(o.seed), n = 8, Q = [];
        while (Q.length < n) { let v = Array.from({ length: n }, () => g.normal()); for (const u of Q) { const p = M.dot(u, v); v = v.map((x, j) => x - p * u[j]); } const L = Math.hypot(...v); if (L > 1e-6) Q.push(v.map((x) => x / L)); }
        const Wh = Q.map((r) => r.map((v) => Math.fround(v * o.scale))), Wx = Array.from({ length: n }, () => Math.fround(0.5 * g.normal())), xs = Array.from({ length: o.steps }, () => g.normal());
        let h = new Array(n).fill(0); const hs = [];
        for (const x of xs) { const pre = M.vm(h, Wh).map((v, j) => v + Wx[j] * x); h = o.act === 'tanh' ? pre.map(Math.tanh) : pre; hs.push(h); }
        let delta = new Array(n).fill(1); const out = new Array(o.steps);
        for (let t = o.steps - 1; t >= 0; t--) { out[t] = Math.hypot(...delta); const dz = o.act === 'tanh' ? delta.map((v, j) => v * (1 - hs[t][j] ** 2)) : delta; delta = Wh.map((row) => M.dot(row, dz)); }
        return out;
      };
      const inp = c.input, gn = gradNorms(inp), n = inp.steps;
      phase(c.frames, 'back').forEach((f) => {
        near(c, f.meta.g, gn[n - 1 - f.meta.k], 'gradient ' + f.meta.k + ' steps back', 1e-3);
        if (f.meta.k) near(c, f.meta.ratio, Math.pow(gn[n - 1 - f.meta.k] / gn[n - 1], 1 / f.meta.k), 'mean factor per step', 1e-3);
      });
      const cmp = phase(c.frames, 'compare')[0].meta;
      vnear(c, cmp.ends, cmp.scales.map((s) => gradNorms(Object.assign({}, inp, { scale: s }))[0]), 'first-step gradients for the three stretches', 1e-3);
    },
  },
  'lstm-gates': {
    random: (rnd) => ({ bits: Array.from({ length: 3 + Math.floor(rnd() * 6) }, () => (rnd() < 0.5 ? '1' : '0')).join('') }),
    check(c) {
      const job = D.lstmCount, w = M.weights(job), bits = [...c.input.bits].map(Number), st = M.runRecurrent(w, 'lstm', bits, false);
      const u = phase(c.frames, 'step')[0].meta.unit, range = [0, 1, 2, 3].map((k) => Math.max(...st.map((s) => s.c[k])) - Math.min(0, ...st.map((s) => s.c[k])));
      c.eq(u, M.argmax(range), 'the counter slot is the one that moves most');
      phase(c.frames, 'step').forEach((f) => {
        const s = st[f.meta.t - 1], cp = f.meta.t > 1 ? st[f.meta.t - 2].c[u] : 0;
        vnear(c, [f.meta.f, f.meta.i, f.meta.g, f.meta.cPrev, f.meta.c], [s.f[u], s.i[u], s.g[u], cp, s.c[u]], 'gates and memory at step ' + f.meta.t, hand);
        near(c, f.meta.c, f.meta.f * f.meta.cPrev + f.meta.i * f.meta.g, 'c = f·c_prev + i·g at step ' + f.meta.t, 1e-5);
        c.eq(f.meta.count, M.argmax(s.p), 'count at step ' + f.meta.t);
      });
      const dn = phase(c.frames, 'done')[0].meta, per = st.map((s, t) => s.c[u] - (t ? st[t - 1].c[u] : 0)), mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      near(c, dn.up1, mean(per.filter((_, t) => bits[t])), 'mean rise on a 1', hand);
      if (dn.up0 !== null) near(c, dn.up0, mean(per.filter((_, t) => !bits[t])), 'mean change on a 0', hand);
    },
  },
  'gru-gates': {
    random: (rnd) => ({ stream: Array.from({ length: 4 + Math.floor(rnd() * 7) }, () => '-ABC'[rnd() < 0.6 ? 0 : 1 + Math.floor(rnd() * 3)]).join('') }),
    check(c) {
      const w = M.weights(D.gruLast), ids = [...c.input.stream].map((ch) => '-ABC'.indexOf(ch)), st = M.runRecurrent(w, 'gru', ids, true);
      phase(c.frames, 'step').forEach((f) => {
        const s = st[f.meta.t - 1], u = f.meta.unit, hp = f.meta.t > 1 ? st[f.meta.t - 2].h[u] : 0;
        vnear(c, [f.meta.z, f.meta.n, f.meta.hPrev, f.meta.h], [s.z[u], s.n[u], hp, s.h[u]], 'gate and state at step ' + f.meta.t, hand);
        near(c, f.meta.h, (1 - f.meta.z) * f.meta.n + f.meta.z * f.meta.hPrev, 'h = (1 − z)·n + z·h_prev at step ' + f.meta.t, 1e-5);
        c.eq(f.meta.last, M.argmax(s.p), 'reported letter at step ' + f.meta.t);
      });
      const dn = phase(c.frames, 'done')[0].meta, zs = st.map((s) => s.z[dn.unit]), mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
      if (dn.zL !== null && dn.zB !== null && ids.some((x) => x) && ids.some((x) => !x)) { near(c, dn.zL, mean(zs.filter((_, t) => ids[t])), 'mean z at letters', hand); near(c, dn.zB, mean(zs.filter((_, t) => !ids[t])), 'mean z at blanks', hand); }
      let last = 0; const truth = ids.map((k) => (k ? (last = k) : last));
      c.eq(dn.right, st.filter((s, t) => M.argmax(s.p) === truth[t]).length, 'right answers');
    },
  },
  'next-char': {
    check(c) {
      const job = D.charLM, w = M.weights(job), vocab = job.config.tokenizer.vocab, ids = [...c.input.prompt].map((ch) => vocab.indexOf(ch)), st = M.runRecurrent(w, 'lstm', ids, true);
      const lr = phase(c.frames, 'learned')[0].meta;
      c.eq(lr.V, vocab.length, 'vocabulary'); c.eq(lr.nText, P.charText().length, 'training text length');
      let sum = 0, cnt = 0;
      phase(c.frames, 'read').forEach((f) => {
        const p = st[f.meta.t].p;
        c.eq(f.meta.guess, M.argmax(p), 'likeliest after ' + (f.meta.t + 1) + ' characters'); near(c, f.meta.pGuess, Math.max(...p), 'its probability', hand);
        if (f.meta.real !== undefined) { near(c, f.meta.pReal, p[ids[f.meta.t + 1]], 'probability of the real next character', hand); near(c, f.meta.loss, -Math.log(p[ids[f.meta.t + 1]]), 'its loss', hand); sum += -Math.log(p[ids[f.meta.t + 1]]); cnt++; }
      });
      const ls = phase(c.frames, 'loss')[0].meta;
      near(c, ls.mean, sum / cnt, 'mean loss', hand); near(c, ls.ppl, Math.exp(sum / cnt), 'perplexity', hand);
      // greedy and sampled continuations, by hand: the state carried on, the argmax (or a draw by the same rule) fed back
      const cont = (n, pick) => {
        const H = w['cell.Wh'].length; let h = new Array(H).fill(0), cc = new Array(H).fill(0), out = '', p;
        for (const x of ids) { const s = M.lstmStep(w, 'cell.', w['emb.W'][x], h, cc); h = s.h; cc = s.c; p = M.softmax(M.head(w, h)); }
        const chosen = [];
        for (let k = 0; k < n; k++) { const id = pick(p); chosen.push(p[id]); out += vocab[id]; const s = M.lstmStep(w, 'cell.', w['emb.W'][id], h, cc); h = s.h; cc = s.c; p = M.softmax(M.head(w, h)); }
        return { out, chosen };
      };
      c.eq(phase(c.frames, 'greedy')[0].meta.gen, cont(24, M.argmax).out, 'greedy continuation');
      const sm = phase(c.frames, 'sample')[0].meta, rng = S.T.rng(c.input.seed);
      const draw = (p) => { const logits = p.map((x) => Math.log(x) / c.input.temp), q = M.softmax(logits), order = q.map((_, i) => i).sort((a, b) => q[b] - q[a] || a - b), r = rng.next(); let acc = 0; for (const i of order) { acc += q[i]; if (r < acc) return i; } return order[order.length - 1]; };
      const tempP = []; const smp = cont(24, (p) => { const id = draw(p); const q = M.softmax(p.map((x) => Math.log(x) / c.input.temp)); tempP.push(q[id]); return id; });
      c.eq(sm.gen, smp.out, 'sampled continuation');
      near(c, sm.avgP, tempP.reduce((a, b) => a + b, 0) / tempP.length, 'mean probability of the drawn characters', 1e-3);
    },
  },
  'seq2seq': {
    random: (rnd) => ({ digits: Array.from({ length: 1 + Math.floor(rnd() * 8) }, () => String(Math.floor(rnd() * 10))).join('') }),
    check(c) {
      const job = D.seq2seq, w = M.weights(job), src = [...c.input.digits].map(Number), C = P.S2S, r = M.translate(w, src, src.length + 2, C.START, C.END);
      c.eq(phase(c.frames, 'task')[0].meta.maxTrained, JOBS.seq2seq.maxLen, 'longest training input');
      phase(c.frames, 'dec').forEach((f) => {
        const s = r.steps[f.meta.t - 1], order = s.p.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
        c.eq(f.meta.id, s.id, 'token at decoder step ' + f.meta.t); near(c, f.meta.p, order[0][0], 'its probability', hand);
        c.eq(f.meta.second, order[1][1], 'runner-up at step ' + f.meta.t); near(c, f.meta.p2, order[1][0], 'its probability', hand);
        c.eq(f.meta.right, s.id === (f.meta.t <= src.length ? src[f.meta.t - 1] : C.END), 'right or wrong at step ' + f.meta.t);
      });
      c.eq(phase(c.frames, 'dec').length, r.steps.length, 'decoder steps');
      const ln = phase(c.frames, 'lengths')[0].meta;
      c.eq(ln.ok, r.out.join() === src.join(), 'whether the whole output is right');
      c.eq(ln.exact, job.lengths.find((x) => x.L === src.length).exact, 'accuracy at this length');
    },
  },
};

const noReplay = process.argv.includes('--no-replay');
process.argv = process.argv.filter((a) => a !== '--no-replay');
run('claims-p11', /^p11-/, checks, ALLOWED);

// ---- the shipped runs, replayed with the engine ------------------------------------------------------------------
if (!noReplay && process.argv.slice(2).length === 0) {
  const T = harness('replays-p11');
  // the length scores of the encoder–decoder, recomputed by hand on the same test inputs
  { const w = M.weights(D.seq2seq), C = P.S2S;
    for (const x of D.seq2seq.lengths) { const g = S.T.rng(91); let right = 0; for (let i = 0; i < 100; i++) { const d = Array.from({ length: x.L }, () => g.int(10)); if (M.translate(w, d, x.L + 2, C.START, C.END).out.join() === d.join()) right++; } T.ok(Math.abs(right / 100 - x.exact) < 1e-9, 'seq2seq exact matches at length ' + x.L + ': ' + right + '% by hand, ' + x.exact * 100 + '% kept'); } }
  const part = { rnnCount: null, lstmCount: null, gruLast: null, charLM: 60, seq2seq: 200 };
  for (const name of Object.keys(JOBS)) {
    const steps = part[name], kept = D[name], again = runJob(S, name, steps || undefined);
    const k = steps || kept.steps, bad = again.loss.slice(0, k).findIndex((v, i) => Math.abs(v - kept.loss[i]) > 1e-4 * Math.max(1, Math.abs(kept.loss[i])));
    T.ok(bad < 0, name + ': the loss at step ' + (bad + 1) + ' is ' + again.loss[bad] + ' replayed, ' + kept.loss[bad] + ' kept');
    if (!steps) {
      // the engine may sum in a different order after a speed-up, so the replay must agree closely, not bit for bit
      const a = M.weights(again), b = M.weights(kept), flat = (w) => kept.weights.names.flatMap((n) => [].concat(w[n]).flat());
      const fa = flat(a), fb = flat(b), worst = Math.max(...fa.map((v, i) => Math.abs(v - fb[i]) / Math.max(1, Math.abs(fb[i]))));
      T.ok(worst < 1e-3, name + ': the final weights replayed differ from the ones kept by up to ' + worst);
      const ev = again.evals.every((e, i) => Object.keys(e).every((k) => (Array.isArray(e[k]) ? e[k].every((v, j) => Math.abs(v - kept.evals[i][k][j]) < 1e-3) : Math.abs(e[k] - kept.evals[i][k]) <= 1e-3 * Math.max(1, Math.abs(kept.evals[i][k])))));
      T.ok(ev, name + ': the validation scores replayed: ' + JSON.stringify(again.evals).slice(0, 200));
    }
  }
  T.done();
}
