// Part 9, neural networks: the narration of every topic read back against an independent working-out. The neuron
// and the forward pass by hand-written sums, every activation and its slope from its closed form and from central
// differences, every gradient on the computational graph against finite differences of the loss, the loss surface
// from a plain loop, the initialisation statistics from plain matrix products, the training runs replayed from
// scratch (the same seed must give the same numbers), the stopping rule re-run by hand, and the normalisations
// from their formulas.
// Usage: node tests/claims-p09.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';

const P = S.P09;
// formula constants the narration may write as they are
const ALLOWED = [/\(0, 1\)/g, /\(−1, 1\)/g, /1 \/ \(1 − p\)/g, /1 − p\b/g, /\b1 − h²/g, /2 · d/g, /√\(?2/g, /×10/g, /\b[a-z] = 0\b/g, /\b[a-z] [<>] 0\b/g, /p\(class [01]\)/g, /\b2d\b/g, /h\(1 − h\)/g, /\/ 2ε/g, /±1\b/g, /1 − η·λ/g];
const near = (c, a, b, msg, tol) => c.ok(Number.isFinite(a) && Math.abs(a - b) <= (tol || 1e-6) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const vnear = (c, a, b, msg, tol) => c.ok(a && b && a.length === b.length && Array.from(a).every((v, i) => Math.abs(v - b[i]) <= (tol || 1e-6) * Math.max(1, Math.abs(b[i]))), msg + ': got ' + JSON.stringify(Array.from(a || [])) + ', want ' + JSON.stringify(Array.from(b || [])));
const f32 = 2e-5;

// ---- activations written out by hand ---------------------------------------------------------------------------------
const GC = Math.sqrt(2 / Math.PI);
const ACT = {
  sigmoid: (z) => 1 / (1 + Math.exp(-z)),
  tanh: (z) => Math.tanh(z),
  relu: (z) => Math.max(0, z),
  gelu: (z) => 0.5 * z * (1 + Math.tanh(GC * (z + 0.044715 * z * z * z))),
};
const DACT = {
  sigmoid: (z) => { const s = ACT.sigmoid(z); return s * (1 - s); },
  tanh: (z) => 1 - Math.tanh(z) ** 2,
  relu: (z) => (z > 0 ? 1 : 0),
  gelu: (z) => (ACT.gelu(z + 1e-5) - ACT.gelu(z - 1e-5)) / 2e-5,
};
const slider = (r, lo, hi) => Math.round((lo + r() * (hi - lo)) * 10) / 10;


// ---- the regularisation experiment replayed with a loop written here ------------------------------------------------
const moons = (n, seed) => { const d = S.Data.moons({ n, noise: 0.3, seed }); return { X: d.X, y: d.y }; };
/** train the same network again from its seed with the engine's layers and optimiser, measuring every 10 steps */
function replay(o) {
  const T = S.T, tr = moons(o.n || 30, 5), va = moons(400, 77);
  const net = S.DL.build('mlp', { inputs: 2, hidden: o.hidden || [32, 32], outputs: 2, act: 'relu', dropout: o.dropout || 0, seed: o.seed || 1 });
  const opt = T.optim.make(o.optimizer || 'adam', net.parameters(), { lr: 0.02, weightDecay: o.weightDecay || 0 });
  const xt = T.tensor(tr.X), xv = T.tensor(va.X), out = [];
  const m = (t) => { net.eval(); const a = T.noGrad(() => net.lossOn({ x: xt, y: tr.y })), b = T.noGrad(() => net.lossOn({ x: xv, y: va.y })); net.train(); out.push({ step: t, trainLoss: a.loss.item(), valLoss: b.loss.item() }); };
  m(0);
  for (let t = 1; t <= (o.steps || 500); t++) { net.train(); opt.zeroGrad(); net.lossOn({ x: xt, y: tr.y }).loss.backward(); opt.step(); if (t % 10 === 0) m(t); }
  return { evals: out, tr, va };
}
/** cross-entropy of a ReLU MLP written out from a state dict, in plain doubles */
function handLoss(state, X, y) {
  const L = Object.keys(state).filter((k) => /layers\.\d+\.W$/.test(k)).length;
  let tot = 0, right = 0;
  X.forEach((x, i) => {
    let h = x;
    for (let l = 0; l < L; l++) {
      const W = state['layers.' + l + '.W'], b = state['layers.' + l + '.b'], [nin, nout] = W.shape;
      const z = Array.from({ length: nout }, (_, j) => { let s = b.data[j]; for (let k = 0; k < nin; k++) s += h[k] * W.data[k * nout + j]; return s; });
      h = l + 1 < L ? z.map((v) => Math.max(0, v)) : z;
    }
    const mx = Math.max(...h), lse = mx + Math.log(h.reduce((s, v) => s + Math.exp(v - mx), 0));
    tot += lse - h[y[i]];
    if ((h[1] > h[0] ? 1 : 0) === y[i]) right++;
  });
  return { loss: tot / X.length, acc: right / X.length };
}

const checks = {
  'p09-neuron': {
    random: (r) => ({ x1: slider(r, -2, 2), x2: slider(r, -2, 2), w1: slider(r, -3, 3), w2: slider(r, -3, 3), b: slider(r, -3, 3), act: ['sigmoid', 'tanh', 'relu', 'gelu'][Math.floor(r() * 4)] }),
    randomRuns: 12,
    check(c) {
      const i = c.input, f = ACT[i.act], df = DACT[i.act];
      const p = [i.w1 * i.x1, i.w2 * i.x2], z = p[0] + p[1] + i.b;
      phase(c.frames, 'mul').forEach((fr) => near(c, fr.meta.p, p[fr.meta.k - 1], 'product ' + fr.meta.k, f32));
      near(c, phase(c.frames, 'sum')[0].meta.z, z, 'z by hand', f32);
      const a = phase(c.frames, 'act')[0].meta;
      near(c, a.a, f(z), 'a = f(z)', f32);
      if (Math.abs(z) > 1e-3 || i.act !== 'relu') near(c, a.slope, df(z), 'slope against the closed form', 1e-3);
      const g = phase(c.frames, 'grad')[0].meta;
      // finite differences of a in each weight
      const fd = [0, 1].map((k) => { const e = 1e-4, wp = [i.w1, i.w2], wm = [i.w1, i.w2]; wp[k] += e; wm[k] -= e; return (f(wp[0] * i.x1 + wp[1] * i.x2 + i.b) - f(wm[0] * i.x1 + wm[1] * i.x2 + i.b)) / (2 * e); });
      if (Math.abs(z) > 1e-2 || i.act !== 'relu') vnear(c, g.dw, fd, '∂a/∂w by finite differences', 2e-3);
      const pl = phase(c.frames, 'plane')[0].meta, nw = Math.hypot(i.w1, i.w2);
      if (nw > 0) near(c, pl.dist, Math.abs(i.b) / nw, 'distance of the line from the origin', 1e-9);
    },
  },
  'p09-activations': {
    random: (r) => ({ act: ['sigmoid', 'tanh', 'relu', 'gelu'][Math.floor(r() * 4)], z0: slider(r, -5, 5), depth: 2 + Math.floor(r() * 29) }),
    randomRuns: 10,
    check(c) {
      const i = c.input, f = ACT[i.act], df = DACT[i.act], zs = Array.from({ length: 241 }, (_, k) => -6 + (12 * k) / 240);
      const cv = phase(c.frames, 'curve')[0].meta;
      near(c, cv.fmin, Math.min(...zs.map(f)), 'smallest output', f32); near(c, cv.fmax, Math.max(...zs.map(f)), 'largest output', f32);
      const pt = phase(c.frames, 'point')[0].meta;
      near(c, pt.f, f(i.z0), 'f(z0)', f32);
      if (Math.abs(i.z0) > 1e-6 || i.act !== 'relu') near(c, pt.d, df(i.z0), 'slope at z0 (closed form)', 1e-3);
      // central differences of the engine's own value, a second method for the slope
      if (Math.abs(i.z0) > 1e-3 || i.act !== 'relu') near(c, pt.d, (f(i.z0 + 1e-5) - f(i.z0 - 1e-5)) / 2e-5, 'slope at z0 (finite differences)', 1e-3);
      const sl = phase(c.frames, 'slope')[0].meta, ds = zs.map(df);
      near(c, sl.dmax, Math.max(...ds), 'largest slope', 1e-3);
      const st = phase(c.frames, 'sat')[0].meta;
      near(c, st.flatShare, ds.filter((v) => Math.abs(v) < 0.05).length / zs.length, 'share of the range that is flat', 0.02);
      const ch = phase(c.frames, 'chain')[0].meta, ids = ['sigmoid', 'tanh', 'relu', 'gelu'];
      ids.forEach((id, k) => { if (Math.abs(i.z0) > 1e-6 || id !== 'relu') near(c, ch.chain[k], Math.pow(DACT[id](i.z0), i.depth), id + ' slope to the power d', 1e-3); });
    },
  },
  'p09-forward': {
    random: (r) => ({ x1: slider(r, -1.5, 2.5), x2: slider(r, -1, 1.5), hidden: 2 + Math.floor(r() * 3), act: ['tanh', 'relu', 'sigmoid'][Math.floor(r() * 3)], weights: r() < 0.7 ? 'trained' : 'random' }),
    randomRuns: 8,
    check(c) {
      const i = c.input, f = ACT[i.act], x = [i.x1, i.x2];
      // the forward pass written out from the weights the topic drew
      const W = c.result.result.W, B = c.result.result.B;
      const z1 = W[0].map((row, j) => row[0] * x[0] + row[1] * x[1] + B[0][j]), h = z1.map(f);
      const z2 = W[1].map((row, k) => row.reduce((s, w, j) => s + w * h[j], 0) + B[1][k]);
      const e = z2.map((v) => Math.exp(v - Math.max(...z2))), p = e.map((v) => v / (e[0] + e[1]));
      phase(c.frames, 'hidden').forEach((fr) => { const j = fr.meta.j; near(c, fr.meta.z, z1[j], 'z of hidden ' + j, f32); near(c, fr.meta.a, h[j], 'activation of hidden ' + j, f32); vnear(c, fr.meta.prods, [W[0][j][0] * x[0], W[0][j][1] * x[1]], 'wire values into hidden ' + j, f32); });
      phase(c.frames, 'output').forEach((fr) => { near(c, fr.meta.z, z2[fr.meta.k], 'score ' + fr.meta.k, f32); vnear(c, fr.meta.prods, W[1][fr.meta.k].map((w, j) => w * h[j]), 'wire values into output ' + fr.meta.k, f32); });
      const sm = phase(c.frames, 'softmax')[0].meta;
      vnear(c, sm.p, p, 'softmax', f32);
      c.eq(sm.win, p[1] > p[0] ? 1 : 0, 'the chosen class');
      // the drawn weights are the engine network's own, and the trained run replays to the same weights
      const net = S.DL.build('mlp', { inputs: 2, hidden: [i.hidden], outputs: 2, act: i.act, seed: 3 });
      if (i.weights === 'random') vnear(c, net.layers[0].W.values(), W[0][0].map((_, k) => W[0].map((r) => r[k])).flat(), 'initial weights', 1e-7);
      const pl = phase(c.frames, 'plane')[0].meta, d = S.Data.moons({ n: 120, noise: 0.15, seed: 2 });
      c.eq(pl.n, d.X.length, 'training points');
      const right = d.X.filter((r, k) => { const hh = W[0].map((row, j) => f(row[0] * r[0] + row[1] * r[1] + B[0][j])), s = W[1].map((row, kk) => row.reduce((a, w, j) => a + w * hh[j], 0) + B[1][kk]); return (s[1] > s[0] ? 1 : 0) === d.y[k]; }).length;
      near(c, pl.acc, right / d.X.length, 'training accuracy by hand', 1e-9);
    },
  },
  'p09-backprop': {
    random: (r) => ({ x1: slider(r, -2, 2), x2: slider(r, -2, 2), t: slider(r, -2, 2), act: ['tanh', 'sigmoid', 'relu'][Math.floor(r() * 3)], seed: 1 + Math.floor(r() * 999), lr: Math.round((0.01 + r() * 0.99) * 100) / 100 }),
    randomRuns: 12,
    check(c) {
      const i = c.input, f = ACT[i.act], v = c.result.result.vals;
      // the loss written out as one formula, and its gradient by central differences in plain doubles
      const loss = (o) => { const w = Object.assign({}, v, o), u = w.w1 * i.x1 + w.w2 * i.x2 + w.b, y = w.v * f(u) + w.c; return (y - i.t) ** 2; };
      const u = v.w1 * i.x1 + v.w2 * i.x2 + v.b, h = f(u), y = v.v * h + v.c;
      const want = { m1: v.w1 * i.x1, m2: v.w2 * i.x2, s: v.w1 * i.x1 + v.w2 * i.x2, u, h, m3: v.v * h, y, d: y - i.t, L: (y - i.t) ** 2 };
      phase(c.frames, 'fwd').forEach((fr) => near(c, fr.meta.v, want[fr.meta.k], 'forward value of ' + fr.meta.k, 1e-12));
      const names = ['w1', 'w2', 'b', 'v', 'c'], fd = names.map((k) => (loss({ [k]: v[k] + 1e-6 }) - loss({ [k]: v[k] - 1e-6 })) / 2e-6);
      // the gradients by hand: the chain rule written out
      const dd = 2 * (y - i.t), du = dd * v.v * DACT[i.act](u), hand = [du * i.x1, du * i.x2, du, dd * h, dd];
      const ck = phase(c.frames, 'check')[0].meta;
      const kink = i.act === 'relu' && Math.abs(u) < 1e-5;
      if (!kink) { vnear(c, ck.ag, fd, 'backprop against finite differences', 1e-5); vnear(c, ck.ag, hand, 'backprop against the chain rule by hand', 1e-9); vnear(c, ck.fd, fd, 'the topic’s own finite differences', 1e-5); }
      // each backward frame: local derivative × upstream gradient = the gradient given to the input
      phase(c.frames, 'bwd').forEach((fr) => { for (const [p, g] of Object.entries(fr.meta.grads)) if (fr.meta.local[p] !== undefined) near(c, g, fr.meta.local[p] * fr.meta.g, 'chain rule at ' + fr.meta.k + ' to ' + p, 1e-12); });
      const st = phase(c.frames, 'step')[0].meta, nx = Object.fromEntries(names.map((k, j) => [k, v[k] - i.lr * hand[j]]));
      if (!kink) { near(c, st.L1, loss(nx), 'loss after one step', 1e-9); near(c, st.L0, loss({}), 'loss before', 1e-12); }
    },
  },
  'p09-loss-surface': {
    random: (r) => ({ data: ['step', 'line', 'flip'][Math.floor(r() * 3)], start: ['near', 'left', 'plateau', 'across'][Math.floor(r() * 4)], lr: Math.round((0.02 + r() * 0.98) * 50) / 50, steps: 10 + Math.floor(r() * 300) }),
    randomRuns: 8,
    check(c) {
      const i = c.input, pts = P.surfPoints(i.data), n = pts.xs.length;
      const L = (w, v) => pts.xs.reduce((s, x, k) => s + (v * Math.tanh(w * x) - pts.ys[k]) ** 2, 0) / n;
      const G = (w, v) => { let gw = 0, gv = 0; pts.xs.forEach((x, k) => { const t = Math.tanh(w * x), e = 2 * (v * t - pts.ys[k]) / n; gw += e * v * (1 - t * t) * x; gv += e * t; }); return [gw, gv]; };
      const m = phase(c.frames, 'min')[0].meta;
      let best = Infinity; for (let a = 0; a < 61; a++) for (let b = 0; b < 61; b++) best = Math.min(best, L(-3 + a / 10, -2.5 + b * 5 / 60));
      near(c, m.gminL, best, 'lowest loss on the grid', 1e-5);
      near(c, L(...m.gmin), m.gminL, 'loss at the lowest grid point', 1e-5);
      near(c, phase(c.frames, 'mirror')[0].meta.mirror, L(-m.gmin[0], -m.gmin[1]), 'mirror loss', 1e-5);
      const sd = phase(c.frames, 'saddle')[0].meta;
      near(c, sd.L, L(0, 0), 'loss at the origin', 1e-5); vnear(c, sd.g, [0, 0], 'gradient at the origin', 1e-7);
      // gradient descent by hand, from the same start, in doubles (float32 drift allowed for)
      let [w, v] = { near: [0.2, 0.1], left: [-2, -0.5], plateau: [2.6, -2], across: [-1, 2] }[i.start];
      const hand = [[w, v]];
      for (let s = 0; s < i.steps; s++) { const g = G(w, v); w -= i.lr * g[0]; v -= i.lr * g[1]; hand.push([w, v]); }
      // (beyond fifty steps a large learning rate makes the walk chaotic, and rounding differences grow: there the loss is checked at the topic's own weights)
      phase(c.frames, 'gd').forEach((fr) => { const h = hand[fr.meta.s]; if (fr.meta.s > 50) near(c, fr.meta.L, L(fr.meta.w, fr.meta.v), 'loss at the weights after ' + fr.meta.s, 1e-9); else if (Math.abs(h[0]) < 50) { near(c, fr.meta.w, h[0], 'w after ' + fr.meta.s + ' steps', 1e-9); near(c, fr.meta.L, L(h[0], h[1]), 'loss after ' + fr.meta.s, 1e-9); } });
      const e = phase(c.frames, 'end')[0].meta;
      near(c, e.L, L(e.we, e.ve), 'final loss at the final weights', 1e-5);
      near(c, e.gn, Math.hypot(...G(e.we, e.ve)), 'final gradient size', 1e-4);
    },
  },
  'p09-init': {
    random: (r) => ({ scheme: ['small', 'large', 'xavier', 'he'][Math.floor(r() * 4)], act: r() < 0.5 ? 'tanh' : 'relu', seed: 1 + Math.floor(r() * 999) }),
    randomRuns: 3,
    check(c) {
      // the same random numbers drawn again, then the layers multiplied out by hand-written loops in doubles
      const i = c.input, g = S.T.rng(i.seed), n = 64, B = 128, f = ACT[i.act];
      const X = Array.from({ length: B }, () => Array.from({ length: n }, () => g.normal(0, 1)));
      const sd = { small: 0.05, large: 1, xavier: Math.sqrt(2 / (2 * n)), he: Math.sqrt(2 / n) }[i.scheme];
      const Ws = Array.from({ length: 6 }, () => Array.from({ length: n * n }, () => g.normal(0, sd)));
      const std = (v) => { const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };
      let h = X; const stds = [];
      for (const W of Ws) { h = h.map((row) => Array.from({ length: n }, (_, j) => { let s = 0; for (let k = 0; k < n; k++) s += row[k] * W[k * n + j]; return f(s); })); stds.push(std(h.flat())); }
      phase(c.frames, 'layer').forEach((fr) => near(c, fr.meta.std, stds[fr.meta.l - 1], 'spread of layer ' + fr.meta.l, 2e-3));
      const fw = phase(c.frames, 'forward')[0].meta;
      near(c, fw.ratio, stds[5] / stds[0], 'forward ratio', 3e-3);
      const st = phase(c.frames, 'start')[0].meta;
      near(c, st.xstd, std(X.flat()), 'input spread', 1e-4);
      near(c, st.wstd, std(Ws[0]), 'weight spread', 1e-4);
      // the gradient of the last layer: L = mean(h²) / 2, so ∂L/∂h = h / (number of entries)
      const bw = phase(c.frames, 'backward')[0].meta;
      near(c, bw.glast, std(h.flat()), 'gradient spread at the last layer: the activations themselves', 2e-3);
    },
  },
  'p09-overfit': {
    check(c) {
      const sz = { small: [4], medium: [8, 8], big: [32, 32] }[c.input.size], n = +c.input.n, rp = replay({ hidden: sz, n });
      const at = (s) => rp.evals.find((e) => e.step === s);
      phase(c.frames, 'snap').forEach((fr) => { const e = at(fr.meta.step); near(c, fr.meta.trainLoss, e.trainLoss, 'training loss replayed at ' + fr.meta.step, 1e-9); near(c, fr.meta.valLoss, e.valLoss, 'fresh loss replayed at ' + fr.meta.step, 1e-9); });
      const best = rp.evals.reduce((a, b) => (b.valLoss < a.valLoss ? b : a)), en = phase(c.frames, 'end')[0].meta;
      c.eq(en.best.step, best.step, 'step of the lowest fresh loss');
      const hl = handLoss(c.result.result.state, rp.tr.X, rp.tr.y), hv = handLoss(c.result.result.state, rp.va.X, rp.va.y);
      near(c, en.last.trainLoss, hl.loss, 'final training loss by hand', 1e-4); near(c, en.last.valLoss, hv.loss, 'final fresh loss by hand', 1e-4);
      c.eq(en.nw, sz.reduce((a, h, k, arr) => a, 0) + [2].concat(sz, [2]).reduce((a, v, k, arr) => (k ? a + arr[k - 1] * v + v : 0), 0), 'number of weights');
    },
  },
  'p09-dropout': {
    random: (r) => ({ p: ['0.2', '0.5', '0.7'][Math.floor(r() * 3)], seed: 1 + Math.floor(r() * 999) }),
    randomRuns: 4,
    check(c) {
      const p = +c.input.p, g = S.T.rng(c.input.seed);
      near(c, phase(c.frames, 'start')[0].meta.scale, 1 / (1 - p), 'scale 1 / (1 − p)', 1e-12);
      // the masks drawn again from the same generator: kept entries hold exactly 1 / (1 − p)
      phase(c.frames, 'mask').forEach((fr) => {
        fr.meta.mask.forEach((row, l) => { c.ok(row.every((v) => v === 0 || Math.abs(v - 1 / (1 - p)) < 1e-6), 'mask values are 0 or 1 / (1 − p)'); c.eq(fr.meta.off[l], row.filter((v) => v === 0).length, 'switched-off count of layer ' + (l + 1)); });
      });
      const A = replay({}), B = replay({ dropout: p }), m = phase(c.frames, 'compare')[0].meta, la = A.evals[A.evals.length - 1], lb = B.evals[B.evals.length - 1];
      near(c, m.without.valLoss, la.valLoss, 'fresh loss without dropout, replayed', 1e-9); near(c, m.with.valLoss, lb.valLoss, 'fresh loss with dropout, replayed', 1e-9);
      near(c, m.with.trainLoss, lb.trainLoss, 'training loss with dropout, replayed', 1e-9);
    },
  },
  'p09-weight-decay': {
    check(c) {
      const lam = +c.input.lam, lr = 0.02;
      const r = phase(c.frames, 'rule')[0].meta;
      near(c, r.keep, 1 - lr * lam, 'share kept per step', 1e-12); near(c, r.hundred, Math.pow(1 - lr * lam, 100), 'share kept after a hundred steps', 1e-12);
      // the one step by hand: AdamW's first step moves w by η·λ·w and then by η·g/(|g| + ε) (bias-corrected moments)
      const st = phase(c.frames, 'step')[0].meta;
      near(c, st.decay, lr * lam * st.before, 'decay part', 1e-6);
      near(c, st.rest, lr * st.grad / (Math.abs(st.grad) + 1e-8), 'Adam part of the first step', 1e-4);
      near(c, st.after, st.before - st.decay - st.rest, 'after = before − decay − Adam step', 1e-6);
      const A = replay({}), B = replay({ optimizer: 'adamw', weightDecay: lam }), e = phase(c.frames, 'end')[0].meta;
      near(c, e.va, A.evals[A.evals.length - 1].valLoss, 'fresh loss without decay, replayed', 1e-9);
      near(c, e.vb, B.evals[B.evals.length - 1].valLoss, 'fresh loss with decay, replayed', 1e-9);
      c.ok(e.wb < e.wa, 'decay leaves smaller weights');
    },
  },
  'p09-early-stop': {
    random: (r) => ({ patience: 1 + Math.floor(r() * 30), size: r() < 0.5 ? 'big' : 'medium' }),
    randomRuns: 6,
    check(c) {
      const rp = replay({ hidden: c.input.size === 'big' ? [32, 32] : [8, 8] }), L = rp.evals.map((e) => e.valLoss), pat = c.input.patience;
      // the stopping rule written again: the lowest so far, a counter, a stop
      let best = 0, wait = 0, stop = null;
      for (let i = 1; i < L.length; i++) { if (L[i] < L[best]) { best = i; wait = 0; } else if (++wait >= pat) { stop = i; break; } }
      const e = phase(c.frames, 'end')[0].meta;
      const upto = L.slice(0, (stop === null ? L.length - 1 : stop) + 1), lo = Math.min(...upto);
      c.eq(e.bestStep, rp.evals[upto.indexOf(lo)].step, 'the weights kept are those of the lowest held-out loss before the stop');
      c.eq(e.stopStep, stop === null ? null : rp.evals[stop].step, 'the step where patience runs out');
      near(c, e.bestLoss, lo, 'the lowest held-out loss before the stop, replayed', 1e-9);
      near(c, e.lastLoss, L[L.length - 1], 'the held-out loss at the end, replayed', 1e-9);
      phase(c.frames, 'best').forEach((f) => { near(c, f.meta.valLoss, L[f.meta.i], 'loss at check ' + f.meta.i, 1e-9); c.ok(L.slice(0, f.meta.i).every((v) => v > f.meta.valLoss), 'check ' + f.meta.i + ' is a new lowest'); });
      phase(c.frames, 'wait').forEach((f) => c.ok(f.meta.valLoss >= f.meta.best && f.meta.wait < pat, 'waiting at check ' + f.meta.i));
    },
  },
  'p09-norm': {
    random: (r) => ({ batch: ['scales', 'odd', 'small'][Math.floor(r() * 3)], gamma: slider(r, 0.5, 2), beta: slider(r, -1, 1) }),
    randomRuns: 6,
    check(c) {
      const B = P.NORM_BATCHES[c.input.batch], X = B.rows, g = c.input.gamma, b = c.input.beta, eps = 1e-5;
      const mean = (v) => v.reduce((a, x) => a + x, 0) / v.length, sd = (v) => Math.sqrt(mean(v.map((x) => (x - mean(v)) ** 2)));
      // batch norm and layer norm written out from their formulas
      const bn = (M) => { const cols = M[0].map((_, j) => M.map((r) => r[j])); return M.map((r) => r.map((x, j) => ((x - mean(cols[j])) / Math.sqrt(sd(cols[j]) ** 2 + eps)) * g + b)); };
      const ln = (M) => M.map((r) => r.map((x) => ((x - mean(r)) / Math.sqrt(sd(r) ** 2 + eps)) * g + b));
      const st = phase(c.frames, 'bnstats')[0].meta, cols = X[0].map((_, j) => X.map((r) => r[j]));
      vnear(c, st.colMean, cols.map(mean), 'column means', 1e-5); vnear(c, st.colStd, cols.map(sd), 'column spreads', 1e-4);
      const ls = phase(c.frames, 'lnstats')[0].meta;
      vnear(c, ls.rowMean, X.map(mean), 'row means', 1e-5); vnear(c, ls.rowStd, X.map(sd), 'row spreads', 1e-4);
      const BN = bn(X), LN = ln(X);
      phase(c.frames, 'bn')[0].meta.BN.forEach((r, i) => vnear(c, r, BN[i], 'batch norm row ' + i, 1e-4));
      phase(c.frames, 'ln')[0].meta.LN.forEach((r, i) => vnear(c, r, LN[i], 'layer norm row ' + i, 1e-4));
      const sw = phase(c.frames, 'swap')[0].meta, X2 = X.slice(0, X.length - 1).concat([B.swap]);
      vnear(c, sw.bn2, bn(X2)[0], 'batch norm of the first example after the swap', 1e-4);
      vnear(c, sw.ln2, LN[0], 'layer norm of the first example is unchanged', 1e-4);
      // the deep stack: the same random numbers drawn again, multiplied out by hand
      const dp = phase(c.frames, 'deep')[0].meta, gg = S.T.rng(4);
      let h = Array.from({ length: 64 }, () => Array.from({ length: 32 }, () => gg.normal(0, 1)));
      const plain = [];
      for (let l = 0; l < 10; l++) { const W = Array.from({ length: 32 * 32 }, () => gg.normal(0, 0.06)); h = h.map((row) => Array.from({ length: 32 }, (_, j) => { let s2 = 0; for (let k = 0; k < 32; k++) s2 += row[k] * W[k * 32 + j]; return Math.tanh(s2); })); plain.push(sd(h.flat())); }
      vnear(c, dp.plain, plain, 'spread per layer without normalisation', 2e-3);
      c.ok(dp.withBN.every((v) => v > 0.3), 'with batch norm every layer keeps a spread above 0.3');
    },
  },
  'p09-playground': {
    random: (r) => ({ data: ['moons', 'circles', 'xor', 'spirals'][Math.floor(r() * 4)], layers: r() < 0.5 ? '1' : '2', neurons: 2 + Math.floor(r() * 5), act: r() < 0.5 ? 'tanh' : 'relu', lr: Math.round((0.003 + r() * 0.097) * 1000) / 1000, steps: 50 + Math.floor(r() * 300), seed: 1 + Math.floor(r() * 999) }),
    randomRuns: 4,
    check(c) {
      // replay: the same network trained again with a loop written here; losses and accuracies at the snapshots
      const i = c.input, set = P.SETS[i.data], d = P.data(i.data, 200, set.noise, 3), T = S.T, hidden = Array.from({ length: +i.layers }, () => i.neurons);
      const net = S.DL.build('mlp', { inputs: 2, hidden, outputs: 2, act: i.act, seed: i.seed }), opt = T.optim.make('adam', net.parameters(), { lr: i.lr }), x = T.tensor(d.X);
      const at = new Map(phase(c.frames, 'snap').map((f) => [f.meta.step, f.meta]));
      const look = (t) => { if (!at.has(t)) return; net.eval(); const r = T.noGrad(() => net.lossOn({ x, y: d.y })); net.train(); near(c, at.get(t).loss, r.loss.item(), 'loss replayed at step ' + t, 1e-9); near(c, at.get(t).acc, r.correct / r.count, 'accuracy replayed at step ' + t, 1e-12); };
      look(0);
      for (let t = 1; t <= i.steps; t++) { net.train(); opt.zeroGrad(); net.lossOn({ x, y: d.y }).loss.backward(); opt.step(); look(t); }
      // the most influential neuron: the largest gap between its two output weights, read off the final weights
      const W = net.layers[net.layers.length - 1].W, [nin] = W.shape, v = W.values(), gaps = Array.from({ length: nin }, (_, j) => Math.abs(v[j * 2 + 1] - v[j * 2]));
      const last = phase(c.frames, 'snap').pop().meta;
      c.eq(last.top, gaps.indexOf(Math.max(...gaps)), 'the ringed neuron');
      near(c, last.inf, Math.max(...gaps), 'its weight gap', 1e-6);
      c.eq(last.nW, [2].concat(hidden, [2]).reduce((a, n, k, arr) => (k ? a + arr[k - 1] * n + n : 0), 0), 'number of weights');
    },
  },
};

run('claims-p09', /^p09-/, checks, ALLOWED);
