// Part 14, generative models: the narration of every topic read back against an independent working-out (see
// tests/claims-p14-ref.mjs): the stored runs decoded with Node's own base64, the trained VAE and denoiser evaluated
// by hand-written matrix products, the reparameterisation and the KL term by formula, the diffusion schedule by
// formula, coverage, distances and correlations from their definitions. The stored runs are also checked to be what
// the engine makes: the first stretch of each training run is repeated and compared with what is kept.
// Usage: node tests/claims-p14.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';
import * as R from './claims-p14-ref.mjs';
import { JOBS } from './ref/p14_runs.mjs';

const ALLOWED = [/16×16/g, /N\(0, I\)/g, /2σ/g, /− 1 − log/g, /∂z\/∂μ = 1/g, /1\/√α/g, /UCB1/g];
const D = S.P14D, G = S.Gen;
const vecNear = (c, a, b, tol, msg) => c.ok(a && b && a.length === b.length && a.every((v, i) => R.rel(v, b[i], tol)), msg + ': ' + JSON.stringify(a).slice(0, 120) + ' against ' + JSON.stringify(b).slice(0, 120));
const near = (c, a, b, tol, msg) => c.ok(R.rel(a, b, tol), msg + ': ' + a + ' against ' + b);
const once = new Map(), memo = (k, f) => { if (!once.has(k)) once.set(k, f()); return once.get(k); };

// ---- the stored runs are the engine's: repeat the start of each and compare ------------------------------------
const only = process.argv.slice(2);
if (!only.length || only.some((o) => 'gen-'.startsWith(o) || o.startsWith('gen-'))) {
  const bad = [];
  for (const k of [2, 4, 8]) {
    const J = JOBS['ae' + k], tr = G.autoencoder(J.o); tr.run(25);
    const s = tr.snapshots.find((x) => x.step === 25), kept = D['ae' + k].snaps.find((x) => x.step === 25), mine = R.u8(kept.recon), eng = s.recon.flatMap((a) => Array.from(a));
    if (!eng.every((v, i) => Math.abs(v - mine[i]) <= 0.5 / 255 + 1e-6)) bad.push('ae' + k + ' reconstructions at step 25');
    if (!R.rel(s.loss, kept.loss, 1e-5)) bad.push('ae' + k + ' loss at step 25');
  }
  for (const b of [0, 1, 4, 16]) {
    const J = JOBS['vae' + b], tr = G.vae(J.o); tr.run(50);
    const e = tr.elbo(), kept = D['vae' + b].snaps.find((x) => x.step === 50), map = R.i16(kept.map), s = tr.snapshots.find((x) => x.step === 50);
    if (!R.rel(e.recon, kept.recon, 1e-5) || !R.rel(e.kl, kept.kl, 1e-4)) bad.push('vae' + b + ' losses at step 50');
    if (!s.map.flatMap((m) => m.z).every((v, i) => Math.abs(v - map[i]) <= kept.map.s)) bad.push('vae' + b + ' latent map at step 50');
  }
  for (const id of ['ganGood', 'ganCollapse']) {
    const tr = G.gan2d(JOBS[id].o); tr.run(100);
    const s = tr.snapshots.find((x) => x.step === 100), kept = D[id].snaps.find((x) => x.step === 100), sm = R.i16(kept.samples);
    if (!s.samples.flat().every((v, i) => Math.abs(v - sm[i]) <= kept.samples.s)) bad.push(id + ' samples at step 100');
    if (s.coverage.covered !== kept.covered) bad.push(id + ' coverage at step 100');
  }
  for (const id of ['ganImg', 'ganImgPlain']) {
    const tr = G.ganImages(JOBS[id].o); tr.run(100);
    const s = tr.snapshots.find((x) => x.step === 100), kept = D[id].snaps.find((x) => x.step === 100), mine = R.u8(kept.images), eng = s.images.flatMap((a) => Array.from(a));
    if (!eng.every((v, i) => Math.abs(v - mine[i]) <= 0.5 / 255 + 1e-6)) bad.push(id + ' images at step 100');
  }
  {
    const tr = G.diffusion2d(JOBS.diff2d.o); tr.run(250);
    const s = tr.snapshots.find((x) => x.step === 250), kept = D.diff2d.snaps.find((x) => x.step === 250), sm = R.i16(kept.samples);
    if (!s.samples.flat().every((v, i) => Math.abs(v - sm[i]) <= kept.samples.s)) bad.push('diff2d samples at step 250');
  }
  console.log('  the stored runs repeat: ' + (bad.length ? 'FAIL ' + bad.join('; ') : 'autoencoders, VAEs, the four GANs and the 2-D diffusion model agree'));
  if (bad.length) process.exitCode = 1;
}

// ---- shared independent pieces ------------------------------------------------------------------------------------
const vae = () => memo('vae', () => R.vaeNet(D.vae1.weights));
const vaeTest = () => memo('vaeTest', () => G.images('digits', { n: 128, size: 16, seed: 1001, classes: JOBS.vae1.o.classes }));
const vaeMeans = () => memo('vaeMeans', () => { const t = vaeTest(); return R.classMeans(t.X, t.y, JOBS.vae1.o.classes); });
const mapMu = () => memo('mapMu', () => { const t = vaeTest(), v = vae(); return [...Array(t.n).keys()].map((i) => ({ z: v.enc(Array.from(t.X.subarray(i * 256, (i + 1) * 256))).mu, y: t.y[i] })); });
const centres = () => memo('centres', () => Object.fromEntries(JOBS.vae1.o.classes.map((d) => { const ms = mapMu().filter((m) => m.y === d); return [d, [ms.reduce((s, m) => s + m.z[0], 0) / ms.length, ms.reduce((s, m) => s + m.z[1], 0) / ms.length]]; })));
const ring = () => memo('ring', () => G.dist2d('ring', { n: 2048, seed: 1 }));
// the digit reader from its stored weights by hand: 256 → 32 (ReLU) → 10, softmax
const reader = () => memo('reader', () => { const [W1, b1, W2, b2] = D.judge.weights.map(R.i16), h = b1.length; return (im) => { const z = R.dense(R.relu(R.dense(im, W1, b1, h)), W2, b2, 10), m = Math.max(...z), e = Array.from(z, (v) => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0); return e.map((v) => v / t); }; });
const readAll = (imgs) => {
  const counts = new Array(10).fill(0), sureCounts = new Array(10).fill(0), digit = [];
  for (const im of imgs) { const p = reader()(Array.from(im)); let b = 0; for (let k = 1; k < 10; k++) if (p[k] > p[b]) b = k; digit.push(b); counts[b]++; if (p[b] >= 0.9) sureCounts[b]++; }
  const n = imgs.length, H = counts.reduce((h, c) => (c ? h - (c / n) * Math.log(c / n) : h), 0);
  return { digit, counts, sureCounts, sure: sureCounts.reduce((a, b) => a + b, 0), effective: Math.exp(H) };
};
const ganFinal = (id) => memo('final' + id, () => R.chunks(R.u8(D[id].final64), 256));
const ganRun = (id) => memo(id, () => ({ hist: D[id].hist, snaps: D[id].snaps.map((s) => ({ step: s.step, samples: R.pairs(R.i16(s.samples)) })) }));

const checks = {
  'gen-ae': {
    check(c) {
      const k = +c.input.k, J = D['ae' + k], test = G.images('digits', { n: 128, size: 16, seed: 1001 }), i = c.input.img - 1, orig = (j) => Array.from(test.X.subarray(j * 256, (j + 1) * 256));
      for (const f of phase(c.frames, 'snap')) {
        const s = J.snaps.find((x) => x.step === f.meta.step), recon = R.chunks(R.u8(s.recon), 256);
        near(c, f.meta.trainLoss, J.loss[f.meta.step / 5 - 1], 1e-9, 'the training loss at step ' + f.meta.step);
        near(c, f.meta.loss, s.loss, 1e-9, 'the test loss at step ' + f.meta.step);
        vecNear(c, f.meta.code, s.codes.slice(i * k, i * k + k), 1e-6, 'the code');
        near(c, f.meta.mae, R.mae(recon[i], orig(i)), 1e-5, 'the mean pixel error');
        c.eq(f.meta.digit, test.y[i], 'the digit');
      }
      const s0 = J.snaps[0], m0 = R.u8(s0.recon).reduce((a, b) => a + b, 0) / (8 * 256);
      near(c, phase(c.frames, 'start')[0].meta.mean, m0, 1e-5, 'the mean pixel before training');
      const end = phase(c.frames, 'end')[0].meta, last = R.chunks(R.u8(J.snaps[J.snaps.length - 1].recon), 256), errs = last.map((im, j) => R.mae(im, orig(j)));
      vecNear(c, end.testLoss, [2, 4, 8].map((kk) => D['ae' + kk].testLoss), 1e-9, 'the three test losses');
      c.eq(end.worst, errs.indexOf(Math.max(...errs)) + 1, 'the hardest digit');
    },
  },
  'gen-vae-map': {
    random: (r) => { const o = ['0', '1', '3', '7', 'centre', 'far']; const a = o[Math.floor(r() * 6)]; let b = o[Math.floor(r() * 6)]; if (b === a) b = a === '0' ? '7' : '0'; return { from: a, to: b, n: 3 + Math.floor(r() * 7) }; },
    check(c) {
      const v = vae(), m0 = phase(c.frames, 'map')[0].meta, mu = mapMu(), ce = centres(), digits = JOBS.vae1.o.classes;
      vecNear(c, m0.counts, digits.map((d) => mu.filter((m) => m.y === d).length), 1e-12, 'the digits on the map');
      const place = (p) => (p === 'far' ? [2.6, -2.6] : p === 'centre' ? [0, 0] : ce[+p]), a = place(c.input.from), b = place(c.input.to), n = c.input.n;
      for (const f of phase(c.frames, 'walk')) {
        const j = f.meta.j - 1, z = [a[0] + ((b[0] - a[0]) * j) / (n - 1), a[1] + ((b[1] - a[1]) * j) / (n - 1)], im = v.dec(z);
        vecNear(c, f.meta.z, z, 1e-4, 'the point on the walk');
        near(c, f.meta.r, Math.hypot(z[0], z[1]), 1e-4, 'its distance from the centre');
        near(c, f.meta.ink, R.ink(im), 2e-4, 'the ink of the decoded image');
        const nc = digits.map((d) => Math.hypot(z[0] - ce[d][0], z[1] - ce[d][1])), lk = R.likeBest(im, vaeMeans());
        const order = digits.map((d, k) => [d, nc[k]]).sort((x, y) => x[1] - y[1]);
        if (f.meta.tie) c.ok(order[1][1] - order[0][1] < 2e-3 && [order[0][0], order[1][0]].sort().join() === [f.meta.near, f.meta.near2].sort().join(), 'a tie between two regions');
        else c.eq(f.meta.near, order[0][0], 'the nearest region');
        near(c, f.meta.nearDist, order[0][1], 1e-4, 'the distance to it');
        c.eq(f.meta.looks, lk.digit, 'what it looks like'); near(c, f.meta.corr, lk.corr, 1e-3, 'the correlation');
      }
    },
  },
  'gen-reparam': {
    random: (r) => ({ img: 1 + Math.floor(r() * 128), seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const v = vae(), t = vaeTest(), i = c.input.img - 1, x = Array.from(t.X.subarray(i * 256, (i + 1) * 256)), e = v.enc(x), m = c.frames[0].meta;
      vecNear(c, m.mu, e.mu, 1e-4, 'μ by hand'); vecNear(c, m.lv, e.lv, 1e-4, 'log σ² by hand');
      vecNear(c, m.sd, m.lv.map((l) => Math.exp(l / 2)), 1e-6, 'σ = exp(½ log σ²)');
      vecNear(c, m.z, m.mu.map((u, k) => u + m.sd[k] * m.eps[k]), 1e-6, 'z = μ + σ ε');
      vecNear(c, m.kl, m.mu.map((u, k) => 0.5 * (u * u + Math.exp(m.lv[k]) - 1 - m.lv[k])), 1e-5, 'the KL per coordinate');
      near(c, m.klSum, m.kl[0] + m.kl[1], 1e-9, 'the KL in all');
      near(c, m.rec, R.bce(v.logits(m.z), x), 1e-3, 'the reconstruction loss by hand');
      near(c, m.elbo, -(m.rec + m.klSum), 1e-5, 'the ELBO');
      c.eq(m.digit, t.y[i], 'the digit');
      const many = phase(c.frames, 'many')[0].meta, imgs = many.more.map((z) => v.dec(z));
      vecNear(c, many.inks, imgs.map(R.ink), 2e-4, 'the inks of the six draws');
      c.eq(many.same, imgs.filter((im) => R.likeBest(im, vaeMeans()).digit === m.digit).length, 'how many look like the digit');
      for (const [k, z] of many.more.entries()) c.ok(Math.abs(z[0] - m.mu[0]) < 6 * m.sd[0] + 1e-9 && Math.abs(z[1] - m.mu[1]) < 6 * m.sd[1] + 1e-9, 'draw ' + k + ' lies near μ');
    },
  },
  'gen-kl': {
    check(c) {
      const b = +c.input.beta, J = D['vae' + b];
      for (const f of c.frames.filter((x) => x.meta.phase === 'snap' || x.meta.phase === 'start')) {
        const s = J.snaps.find((x) => x.step === f.meta.step), z = R.pairs(R.i16(s.map));
        near(c, f.meta.recon, s.recon, 1e-12, 'the reconstruction'); near(c, f.meta.kl, s.kl, 1e-12, 'the KL');
        near(c, f.meta.loss, s.recon + b * s.kl, 1e-9, 'the weighted loss');
        near(c, f.meta.spread, z.reduce((a, p) => a + Math.hypot(p[0], p[1]), 0) / z.length, 1e-6, 'the spread of the codes');
      }
      const end = phase(c.frames, 'end')[0].meta;
      vecNear(c, end.rec, [0, 1, 4, 16].map((x) => D['vae' + x].final.recon), 1e-12, 'the final reconstructions');
      vecNear(c, end.kl, [0, 1, 4, 16].map((x) => D['vae' + x].final.kl), 1e-12, 'the final KLs');
      c.ok(end.kl[3] < end.kl[1] && end.kl[1] < end.kl[0], 'a heavier KL weight leaves a smaller KL');
    },
  },
  'gen-gan-duel': {
    check(c) {
      const id = c.input.run === 'good' ? 'ganGood' : 'ganCollapse', g = ganRun(id), data = ring();
      for (const f of c.frames.filter((x) => x.meta.phase === 'snap')) {
        const k = f.meta.step / 10 - 1;
        for (const key of ['dReal', 'dFake', 'lossD', 'lossG']) near(c, f.meta[key], g.hist[key][k], 1e-12, key + ' at step ' + f.meta.step);
        const cov = R.coverage(g.snaps.find((s) => s.step === f.meta.step).samples, data.modes, 3 * data.sd);
        c.eq(f.meta.covered, cov.covered, 'the clusters covered at step ' + f.meta.step); near(c, f.meta.effective, cov.effective, 1e-3, 'the effective number of modes');
      }
    },
  },
  'gen-mode-collapse': {
    random: (r) => ({ r: 2 + 0.5 * Math.floor(r() * 9) }),
    check(c) {
      const A = ganRun('ganGood'), B = ganRun('ganCollapse'), data = ring(), rad = c.input.r * data.sd;
      for (const f of c.frames.filter((x) => x.meta.phase !== 'end')) {
        const ca = R.coverage(A.snaps.find((s) => s.step === f.meta.step).samples, data.modes, rad), cb = R.coverage(B.snaps.find((s) => s.step === f.meta.step).samples, data.modes, rad);
        vecNear(c, f.meta.a, ca.counts, 1e-12, 'the balanced run’s counts'); vecNear(c, f.meta.b, cb.counts, 1e-12, 'the other run’s counts');
        c.eq([f.meta.coveredA, f.meta.coveredB], [ca.covered, cb.covered], 'the clusters covered');
        near(c, f.meta.effA, ca.effective, 1e-3, 'effective modes (balanced)'); near(c, f.meta.effB, cb.effective, 1e-3, 'effective modes (other)');
      }
    },
  },
  'gen-gan-digits': {
    check(c) {
      const id = c.input.run === 'plain' ? 'ganImgPlain' : 'ganImg', other = id === 'ganImg' ? 'ganImgPlain' : 'ganImg', J = D[id], j = c.input.pick - 1;
      for (const f of c.frames.filter((x) => x.meta.phase === 'start' || x.meta.phase === 'snap')) {
        const s = J.snaps.find((x) => x.step === f.meta.step), imgs = R.chunks(R.u8(s.images), 256), rd = readAll(imgs);
        near(c, f.meta.mf, s.dFake.reduce((a, b) => a + b, 0) / 16, 1e-9, 'D’s mean on the fakes'); near(c, f.meta.mr, s.dReal.reduce((a, b) => a + b, 0) / 16, 1e-9, 'on the real digits');
        near(c, f.meta.ink, R.ink(imgs[j]), 1e-6, 'the ink of the followed image'); near(c, f.meta.d, s.dFake[j], 1e-12, 'D on it');
        c.eq(f.meta.fooled, s.dFake.filter((v) => v >= 0.5).length, 'the fakes that pass');
        vecNear(c, f.meta.counts, rd.counts, 1e-12, 'what the reader takes the 16 fakes for'); c.eq(f.meta.sure, rd.sure, 'how many it is sure of');
      }
      const t = phase(c.frames, 'tally')[0].meta, rd = readAll(ganFinal(id)), set = G.images('digits', { n: 512, size: 16, seed: 1, classes: J.o.classes }), rr = readAll([...Array(64).keys()].map((i) => set.X.subarray(i * 256, (i + 1) * 256)));
      vecNear(c, t.counts, rd.counts, 1e-12, 'the 64 samples read'); vecNear(c, t.sureCounts, rd.sureCounts, 1e-12, 'read with p ≥ 0.9'); near(c, t.eff, rd.effective, 1e-9, 'their effective number of digits');
      vecNear(c, t.realCounts, rr.counts, 1e-12, 'the 64 real digits read'); near(c, t.effReal, rr.effective, 1e-9, 'theirs');
      const test = G.images('digits', { n: 1000, size: 16, seed: 78 }), hit = readAll([...Array(test.n).keys()].map((i) => test.X.subarray(i * 256, (i + 1) * 256))).digit.filter((d, i) => d === test.y[i]).length;
      near(c, t.acc, hit / test.n, 1e-12, 'the reader’s accuracy on 1000 new digits');
      const e = phase(c.frames, 'end')[0].meta, ro = readAll(ganFinal(other));
      vecNear(c, e.otherCounts, ro.counts, 1e-12, 'the other run’s samples read'); near(c, e.otherEff, ro.effective, 1e-9, 'its effective number of digits');
      c.eq(e.kinds, J.o.classes.filter((d) => rd.counts[d] > 0).length, 'how many of the trained digits it draws');
    },
  },
  'gen-diff-forward': {
    random: (r) => ({ kind: r() < 0.5 ? 'linear' : 'cosine', T: 20 + Math.floor(r() * 181), data: ['moons', 'ring', 'spiral', 'circle'][Math.floor(r() * 4)], shape: ['circle', 'square', 'triangle', 'star'][Math.floor(r() * 4)] }),
    check(c) {
      const sc = R.schedule(c.input.T, c.input.kind), m0 = c.frames[0].meta, g = S.T.rng(31), data = G.dist2d(c.input.data, { n: 2048, seed: 1 }), p0 = data.X[0], e0 = [g.normal(), g.normal()];
      near(c, m0.b1, sc.beta[1], 1e-9, 'β at the first step'); near(c, m0.bT, sc.beta[c.input.T], 1e-9, 'β at the last');
      for (const f of c.frames.filter((x) => x.meta.t > 0)) {
        const t = f.meta.t, ab = sc.ab[t];
        near(c, f.meta.ab, ab, 1e-9, 'ᾱ at ' + t); near(c, f.meta.beta, sc.beta[t], 1e-9, 'β at ' + t);
        near(c, f.meta.sg, Math.sqrt(ab), 1e-9, 'the signal share'); near(c, f.meta.nz, Math.sqrt(1 - ab), 1e-9, 'the noise share');
        vecNear(c, f.meta.p, [Math.sqrt(ab) * p0[0] + Math.sqrt(1 - ab) * e0[0], Math.sqrt(ab) * p0[1] + Math.sqrt(1 - ab) * e0[1]], 1e-9, 'the noised point');
      }
    },
  },
  'gen-diff-denoise': {
    random: (r) => ({ ex: 1 + Math.floor(r() * 50), t: 1 + Math.floor(r() * 100) }),
    check(c) {
      const net = R.denoiser(D.diff2d.weights, 100), sc = R.schedule(100, 'linear'), t = c.input.t, ab = sc.ab[t];
      const pm = phase(c.frames, 'predict')[0].meta, x0 = pm.x0, eps = pm.eps, xt = [Math.sqrt(ab) * x0[0] + Math.sqrt(1 - ab) * eps[0], Math.sqrt(ab) * x0[1] + Math.sqrt(1 - ab) * eps[1]];
      vecNear(c, phase(c.frames, 'noised')[0].meta.xt, xt, 1e-9, 'x_t'); near(c, phase(c.frames, 'noised')[0].meta.ab, ab, 1e-9, 'ᾱ');
      const eh = net(xt, t);
      vecNear(c, pm.eh, eh, 2e-4, 'the predicted noise by hand');
      near(c, pm.err, (eps[0] - pm.eh[0]) ** 2 + (eps[1] - pm.eh[1]) ** 2, 1e-9, 'the squared error');
      vecNear(c, pm.x0h, [(xt[0] - Math.sqrt(1 - ab) * pm.eh[0]) / Math.sqrt(ab), (xt[1] - Math.sqrt(1 - ab) * pm.eh[1]) / Math.sqrt(ab)], 1e-6, 'x̂₀');
      const X = G.dist2d('moons', { n: 2048, seed: 1 }).X, grid = []; for (let a = 0; a < 11; a++) for (let b = 0; b < 11; b++) grid.push([-2.7 + (5.4 * b) / 10, 2.7 - (5.4 * a) / 10]);
      for (const f of phase(c.frames, 'field')) {
        const a2 = sc.ab[f.meta.t], guess = grid.map((p) => { const e = net(p, f.meta.t); return [(p[0] - Math.sqrt(1 - a2) * e[0]) / Math.sqrt(a2), (p[1] - Math.sqrt(1 - a2) * e[1]) / Math.sqrt(a2)]; });
        near(c, f.meta.before, R.chamfer(grid, X.slice(0, 600)), 1e-9, 'the grid’s distance to the data'); near(c, f.meta.after, R.chamfer(guess, X.slice(0, 600)), 2e-3, 'the guesses’ distance');
      }
      const tr = phase(c.frames, 'train')[0].meta, L = D.diff2d.loss;
      near(c, tr.first, L[0], 1e-12, 'the first loss'); near(c, tr.tail, L.slice(-10).reduce((a, b) => a + b, 0) / 10, 1e-9, 'the last losses');
    },
  },
  'gen-diff-sample': {
    random: (r) => ({ n: 50 + Math.floor(r() * 200), seed: 1 + Math.floor(r() * 9000) }),
    randomRuns: 3,
    check(c) {
      const sc = R.schedule(100, 'linear'), X = G.dist2d('moons', { n: 2048, seed: 1 }).X;
      for (const f of c.frames.filter((x) => x.meta.at)) {
        const t = f.meta.at;
        near(c, f.meta.k1, 1 / Math.sqrt(1 - sc.beta[t]), 1e-9, '1/√α at ' + t); near(c, f.meta.k2, sc.beta[t] / Math.sqrt(1 - sc.ab[t]), 1e-9, 'the share of ε̂ removed');
        near(c, f.meta.sigma, t > 1 ? Math.sqrt((sc.beta[t] * (1 - sc.ab[t - 1])) / (1 - sc.ab[t])) : 0, 1e-9, 'σ');
        const pts = f.items.filter((it) => /^ds:s\d/.test(it.key)).length; c.ok(pts > 0, 'samples drawn');
      }
      const end = phase(c.frames, 'end')[0].meta, g = S.T.rng(c.input.seed), noise = Array.from({ length: c.input.n }, () => [g.normal(), g.normal()]);
      near(c, end.swdNoise, R.swd(noise, X.slice(0, c.input.n)), 1e-9, 'the distance of plain noise');
      c.ok(end.swdNow < end.swdNoise, 'the samples are closer to the data than noise');
      vecNear(c, end.swds, D.diff2d.snaps.map((s) => s.swd), 1e-12, 'the distances during training');
    },
  },
  'gen-diff-images': {
    check(c) {
      const J = D.diffImg, sc = R.schedule(J.o.steps, J.o.kind || 'cosine'), j = c.input.pick - 1;
      for (const f of c.frames.filter((x) => x.meta.phase !== 'end')) {
        const fr = J.frames.find((x) => x.t === f.meta.t), im = R.chunks(R.u8(fr.images), 256)[j], m = im.reduce((a, b) => a + b, 0) / 256, sd = Math.sqrt(im.reduce((a, b) => a + (b - m) ** 2, 0) / 256);
        near(c, f.meta.ab, sc.ab[f.meta.t], 1e-9, 'ᾱ'); near(c, f.meta.mean, m, 1e-6, 'the mean brightness'); near(c, f.meta.sd, sd, 1e-6, 'the spread');
      }
      const end = phase(c.frames, 'end')[0].meta, last = R.chunks(R.u8(J.frames[J.frames.length - 1].images), 256), set = G.images('shapes', { n: 64, size: 16, seed: 1, classes: [0] });
      const area = (px) => px.filter((v) => v > 0.5).length, circ = [...Array(set.n).keys()].map((i) => area(Array.from(set.X.subarray(i * 256, (i + 1) * 256)))), lo = Math.min(...circ), hi = Math.max(...circ), areas = last.map(area);
      vecNear(c, end.areas, areas, 1e-12, 'the dark areas of the samples'); c.eq([end.lo, end.hi], [lo, hi], 'the range of the training circles');
      c.eq(end.good, areas.filter((a) => a >= lo && a <= hi).length, 'the samples in that range');
      c.eq([end.small, end.big, end.blank], [areas.filter((a) => a < lo).length, areas.filter((a) => a > hi).length, areas.filter((a) => a < 5).length], 'the samples too small, too large and blank');
      c.eq([end.nTrain, end.temp], [J.o.n || 256, J.sample && J.sample.temp !== 1 ? J.sample.temp : null], 'the training circles and the temperature');
      if (J.final64) { const more = R.chunks(R.u8(J.final64), 256).map(area); c.eq([end.n64, end.good64], [more.length, more.filter((a) => a >= lo && a <= hi).length], 'the 64 more samples in that range'); }
    },
  },
  'gen-compare': {
    random: (r) => ({ seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const v = vae(), g = S.T.rng(c.input.seed), zs = [...Array(8).keys()].map(() => [g.normal(), g.normal()]), cls = zs.map((z) => R.likeBest(v.dec(z), vaeMeans()).digit);
      vecNear(c, phase(c.frames, 'vae')[0].meta.tally, JOBS.vae1.o.classes.map((d) => cls.filter((x) => x === d).length), 1e-12, 'what the VAE’s draws look like');
      const gm = phase(c.frames, 'gan')[0].meta;
      vecNear(c, gm.counts, readAll(ganFinal('ganImg')).counts, 1e-12, 'what the GAN’s samples are read as'); vecNear(c, gm.plainCounts, readAll(ganFinal('ganImgPlain')).counts, 1e-12, 'the plain GAN’s');
      c.eq(phase(c.frames, 'diff')[0].meta.T, JOBS.diffImg.o.steps, 'the diffusion steps');
    },
  },
};

run('claims-p14', /^p14-/, checks, ALLOWED);
