// Generative models: the diffusion schedule, gradient checks of the VAE loss (with the reparameterisation and the
// KL term), the upsampling used by the convolutional decoder and the diffusion denoiser; convergence (the VAE's
// ELBO improves, the GAN covers the ring's modes on the good preset and collapses on the other, diffusion samples
// land near the data); traces; determinism.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T, G = S.Gen;
const t = harness('gen');
// budgets are in CPU time, so a busy machine does not fail them
const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
const timed = (name, budgetMs, fn) => {
  const t0 = cpu(), r = fn(), ms = cpu() - t0;
  console.log(`  ${name.padEnd(34)} ${r}  in ${(ms / 1000).toFixed(2)} s`);
  t.ok(ms < budgetMs, name + ' took ' + ms.toFixed(0) + ' ms (budget ' + budgetMs + ')');
};
/** central differences on up to k elements of each parameter, in float64: lossOf() must rebuild the graph */
function gradcheck(name, params, lossOf, k = 12, tol = 1e-5) {
  for (const p of params) p.grad = null;
  lossOf().backward();
  const g = T.rng(3);
  params.forEach((p, pi) => {
    let bad = null;
    for (let c = 0; c < Math.min(k, p.size); c++) {
      const i = p.size <= k ? c : g.int(p.size), old = p.data[i], eps = 1e-6;
      p.data[i] = old + eps; const lp = T.noGrad(lossOf).item();
      p.data[i] = old - eps; const lm = T.noGrad(lossOf).item();
      p.data[i] = old;
      const num = (lp - lm) / (2 * eps), ana = p.grad ? p.grad.data[i] : 0, err = Math.abs(num - ana) / Math.max(1, Math.abs(num), Math.abs(ana));
      if (!(err <= tol) && !bad) bad = 'param ' + pi + ' [' + p.shape + '] element ' + i + ': analytic ' + ana + ', numeric ' + num;
    }
    t.ok(!bad, name + ': ' + bad);
  });
}

// ---- the noise schedule ----
for (const kind of ['linear', 'cosine']) {
  const s = G.schedule({ steps: 100, kind });
  let mono = true; for (let i = 1; i <= 100; i++) if (!(s.alphaBar[i] < s.alphaBar[i - 1])) mono = false;
  t.ok(mono, kind + ' schedule: ᾱ falls at every step');
  t.ok(s.alphaBar[100] < 0.01 && s.alphaBar[1] > 0.95, kind + ' schedule: ᾱ from ' + s.alphaBar[1] + ' to ' + s.alphaBar[100]);
  t.ok(s.trace.length === 3 && /ᾱ/.test(s.trace[1].say), kind + ' schedule trace');
}
{ // x_t = √ᾱ x0 + √(1 − ᾱ) ε: the mean shrinks and the variance tends to 1
  const s = G.schedule({ steps: 50 }), g = T.rng(4), xs = Array.from({ length: 4000 }, () => G.noise([2], 50, s, [g.normal()])[0]);
  const m = xs.reduce((a, b) => a + b, 0) / xs.length, v = xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length;
  t.ok(Math.abs(m - 2 * Math.sqrt(s.alphaBar[50])) < 0.05 && Math.abs(v - (1 - s.alphaBar[50])) < 0.06, 'forward noising mean ' + m + ' and variance ' + v);
}

// ---- gradient checks (float64) ----
T.float64(() => {
  const x = T.param(T.normal([2, 3, 2, 3], 0, 1, 5)), R = T.normal([2, 3, 4, 6], 0, 1, 6);
  gradcheck('up2', [x], () => T.sum(T.mul(G.up2(x), R)));
  const v = G.vae({ seed: 2, n: 16, nTest: 8, hidden: 8, batch: 4 }), xb = G.batchOf(v.set, [0, 1, 2, 3]);
  gradcheck('VAE loss (reconstruction + KL through z = μ + σε)', v.net.parameters(), () => v.pass(xb, T.rng(9)).loss, 8);
  gradcheck('VAE KL term alone', v.net.enc.parameters(), () => v.pass(xb, T.rng(9)).kl, 8);
  const c = G.autoencoder({ kind: 'conv', seed: 3, n: 8, nTest: 4, latent: 3 }), xc = G.batchOf(c.set, [0, 1]);
  // biases start at zero, which puts the blank background exactly on ReLU's kink; move them off it first
  { const g = T.rng(12); for (const p of c.net.parameters()) if (p.ndim === 1) for (let i = 0; i < p.size; i++) p.data[i] = g.normal(0, 0.1); }
  gradcheck('conv autoencoder loss', c.net.parameters(), () => T.sum(T.bceWithLogits(c.net.decode(c.net.encode(xc)), xc, { reduction: 'none' })), 6);
  const d = G.diffusion2d({ seed: 4, hidden: 8, n: 64 }), xt = T.normal([5, 2], 0, 1, 7), eps = T.normal([5, 2], 0, 1, 8);
  gradcheck('diffusion denoiser loss', d.net.parameters(), () => T.mseLoss(d.net.forward(xt, [1, 20, 40, 70, 100]), eps), 8);
  const gn = G.gan2d({ seed: 5, hidden: 8 }), z = T.normal([6, 4], 0, 1, 9);
  gradcheck('GAN generator loss through the discriminator', gn.G.parameters(), () => T.bceWithLogits(gn.D.forward(gn.G.forward(z)), T.ones([6, 1])), 8);
});

// ---- convergence ----
timed('autoencoder (MLP) reconstructs', 15000, () => {
  const a = G.autoencoder({ kind: 'mlp', seed: 1 }), l0 = a.testLoss();
  a.run(300);
  const l1 = a.testLoss();
  t.ok(l1 < 0.4 * l0, 'MLP autoencoder test loss ' + l0 + ' -> ' + l1);
  t.ok(a.snapshots.length >= 6 && a.snapshots[0].recon.length === 8, 'autoencoder snapshots with reconstructions');
  return 'test loss ' + l0.toFixed(1) + ' -> ' + l1.toFixed(1);
});
timed('autoencoder (conv) learns', 20000, () => {
  const a = G.autoencoder({ kind: 'conv', seed: 1, snapEvery: 1000 }), l0 = a.testLoss();
  a.run(120);
  const l1 = a.testLoss();
  t.ok(l1 < 0.6 * l0, 'conv autoencoder test loss ' + l0 + ' -> ' + l1);
  return 'test loss ' + l0.toFixed(1) + ' -> ' + l1.toFixed(1);
});
timed('VAE ELBO improves', 20000, () => {
  const v = G.vae({ seed: 1 }), e0 = v.elbo();
  v.run(600);
  const e1 = v.elbo();
  t.ok(e1.elbo > e0.elbo + 80, 'VAE held-out ELBO ' + e0.elbo + ' -> ' + e1.elbo);
  t.ok(e1.kl > 0.5, 'VAE uses its latent space (KL ' + e1.kl + ')');
  const last = v.snapshots[v.snapshots.length - 1];
  t.ok(last.map.length === 128 && last.map[0].z.length === 2 && last.grid.images.length === 81, 'VAE snapshot: a latent map and a 9×9 decoded grid');
  const ex = v.explain(0);
  t.ok(ex.length === 7 && ex.every((s) => /\d/.test(s.say)), 'VAE explain: seven steps with numbers');
  const z = ex[3].z, mu = ex[0].mu, sd = ex[1].sigma, e = ex[2].eps;
  t.ok(z.every((zz, k) => Math.abs(zz - (mu[k] + sd[k] * e[k])) < 1e-5), 'z = μ + σ ε in the trace');
  return 'ELBO ' + e0.elbo.toFixed(1) + ' -> ' + e1.elbo.toFixed(1);
});
timed('GAN covers the ring (good preset)', 45000, () => {
  const a = G.gan2d({ preset: 'good', seed: 1, snapEvery: 250 });
  a.run(1000);
  const c = a.coverage();
  t.ok(c.covered === 8 && c.effective > 7.5 && c.quality > 0.5, 'good GAN: ' + c.say);
  const s = a.snapshots[a.snapshots.length - 1];
  t.ok(s.samples.length === 256 && s.field.d.length === 625 && s.field.d.every((p) => p >= 0 && p <= 1), 'GAN snapshot: samples and the discriminator field');
  t.ok(a.explain().length === 5, 'GAN explain');
  return c.covered + '/8 modes, ' + Math.round(100 * c.quality) + '% near a mode';
});
timed('GAN collapses (collapse preset)', 20000, () => {
  const a = G.gan2d({ preset: 'collapse', seed: 1, snapEvery: 1000 });
  a.run(1000);
  const c = a.coverage();
  t.ok(c.effective <= 3 && c.covered <= 3, 'collapse preset: ' + c.say);
  return 'samples spread over the equivalent of ' + c.effective.toFixed(1) + ' of 8 modes';
});
timed('GAN on 16×16 digits runs', 15000, () => {
  const a = G.ganImages({ seed: 1, snapEvery: 100 });
  a.run(200);
  const s = a.snapshots[a.snapshots.length - 1], h = a.history;
  t.ok(s.images.length === 16 && s.images[0].length === 256 && s.images[0].every((v) => v >= 0 && v <= 1), 'image GAN snapshot: 16 generated images in [0, 1]');
  t.ok(s.dFake.length === 16 && s.dReal.length === 16 && h.every((r) => Number.isFinite(r.lossD) && Number.isFinite(r.lossG)), 'the discriminator\'s view of fakes and reals, finite losses');
  const ink = (im) => im.reduce((x, y) => x + y, 0) / im.length, real = a.set.X.slice(0, 256 * 16).reduce((x, y) => x + y, 0) / (256 * 16);
  const fake = s.images.reduce((x, im) => x + ink(im), 0) / 16;
  t.ok(Math.abs(fake - real) < 0.1, 'generated images carry about as much ink as real ones (' + fake.toFixed(3) + ' against ' + real.toFixed(3) + ')');
  return 'mean ink ' + fake.toFixed(3) + ' (real ' + real.toFixed(3) + ')';
});
timed('steadied GAN keeps all three digits', 240000, () => {
  // a digit reader first (the same kind the topics use), then the steadied duel for 1000 steps
  const jn = G.judgeNet({ hidden: 32, n: 2000, seed: 5 }); jn.train(600);
  const acc = jn.accuracy(G.images('digits', { n: 300, size: 16, seed: 78 }));
  t.ok(acc > 0.95, 'the digit reader reads ' + acc + ' of new digits correctly');
  const a = G.ganImages({ preset: 'steady', classes: [0, 1, 7], seed: 1, snapEvery: 1000 });
  a.run(1000);
  const rd = jn.read(a.samples(64));
  t.ok([0, 1, 7].every((d) => rd.counts[d] >= 8), 'steadied GAN: 64 samples read as ' + rd.counts.join(' '));
  return 'reader ' + acc.toFixed(3) + '; samples read as 0/1/7: ' + [0, 1, 7].map((d) => rd.counts[d]).join('/');
});
{
  // the spectral-norm cap: after enough power-iteration steps the largest singular value is at most the cap
  const L = new T.nn.Dense(40, 30, { rng: T.rng(3) }), W = L.W.data;
  for (let k = 0; k < W.length; k++) W[k] *= 6;
  const st = new Map(); let est = 0;
  for (let k = 0; k < 40; k++) est = G.specNorm([L], st, 1)[0];
  // the true largest singular value by a long power iteration of WᵀW
  let v = new Float64Array(40).fill(1);
  for (let it = 0; it < 500; it++) { const u = new Float64Array(30); for (let i = 0; i < 40; i++) for (let j = 0; j < 30; j++) u[j] += W[i * 30 + j] * v[i]; const v2 = new Float64Array(40); for (let i = 0; i < 40; i++) for (let j = 0; j < 30; j++) v2[i] += W[i * 30 + j] * u[j]; const nv = Math.hypot(...v2); v = v2.map((x) => x / nv); }
  const u = new Float64Array(30); for (let i = 0; i < 40; i++) for (let j = 0; j < 30; j++) u[j] += W[i * 30 + j] * v[i];
  const sigma = Math.hypot(...u);
  t.ok(sigma <= 1.01 && sigma > 0.9 && est > 0.99, 'spectral-norm cap: the largest singular value is ' + sigma.toFixed(4) + ' (estimate ' + est.toFixed(4) + ')');
  // the steadied discriminator sees the batch: the same first image is judged differently in another batch
  const a = G.ganImages({ preset: 'steady', seed: 2, snapEvery: 1000 }), x = G.batchOf(a.set, [0, 1, 2, 3]), y = G.batchOf(a.set, [0, 0, 0, 0]);
  const d1 = T.noGrad(() => a.D.forward(x).values()[0]), d2 = T.noGrad(() => a.D.forward(y).values()[0]);
  t.ok(Math.abs(d1 - d2) > 1e-6, 'the steadied discriminator sees the batch’s spread (' + d1 + ' against ' + d2 + ')');
}
timed('diffusion (2-D) samples near the data', 40000, () => {
  const d = G.diffusion2d({ dist: 'moons', seed: 1, snapEvery: 1000 });
  d.run(1500);
  const q = d.quality();
  t.ok(q.chamfer < 0.5 * q.noiseChamfer, 'diffusion samples: mean distance to the data ' + q.chamfer + ' against ' + q.noiseChamfer + ' for plain noise');
  t.ok(q.swd < 0.6 * q.noiseSwd, 'diffusion samples: sliced Wasserstein ' + q.swd + ' against ' + q.noiseSwd);
  const s = d.sample({ n: 64, frames: 5 });
  t.ok(s.frames.length === 6 && s.frames[0].t === 100 && s.frames[5].t === 0 && s.trace.length >= 5, 'reverse chain frames ' + s.frames.map((f) => f.t));
  const f = d.forward(64);
  t.ok(f.frames.length === 11 && f.frames[10].noise > 0.99, 'forward process frames');
  return 'distance ' + q.chamfer.toFixed(3) + ' (noise ' + q.noiseChamfer.toFixed(3) + ')';
});
timed('diffusion (16×16 images) learns', 30000, () => {
  const d = G.diffusionImages({ seed: 1, snapEvery: 1000 });
  d.run(60);
  const h = d.history.map((r) => r.loss), first = h.slice(0, 10).reduce((a, b) => a + b) / 10, last = h.slice(-10).reduce((a, b) => a + b) / 10;
  t.ok(last < 0.8 * first, 'image denoiser loss ' + first + ' -> ' + last);
  const s = d.sample({ n: 2, frames: 3 });
  t.ok(s.final.length === 2 && s.final[0].length === 256 && s.frames.length === 4, 'image sampling frames');
  // with a moving average of the weights, sampling uses the average and leaves the trained weights as they were
  const e = G.diffusionImages({ seed: 1, snapEvery: 1000, ema: 0.9 });
  e.run(5);
  const before = G.copyParams(e.net);
  e.sample({ n: 1, frames: 1 });
  t.ok(G.copyParams(e.net).every((p, i) => p.every((v, k) => v === before[i][k])), 'sampling with averaged weights restores the trained ones');
  return 'loss ' + first.toFixed(3) + ' -> ' + last.toFixed(3);
});

// ---- determinism and jobs ----
const hist = (make, n) => { const a = make(); a.run(n); return a.history.map((r) => r.loss); };
t.eq(hist(() => G.vae({ seed: 3, snapEvery: 1000 }), 15), hist(() => G.vae({ seed: 3, snapEvery: 1000 }), 15), 'VAE is deterministic');
t.eq(hist(() => G.gan2d({ seed: 3, snapEvery: 1000 }), 15), hist(() => G.gan2d({ seed: 3, snapEvery: 1000 }), 15), 'GAN is deterministic');
t.eq(hist(() => G.diffusion2d({ seed: 3, snapEvery: 1000 }), 15), hist(() => G.diffusion2d({ seed: 3, snapEvery: 1000 }), 15), 'diffusion is deterministic');
t.ok(JSON.stringify(hist(() => G.vae({ seed: 3, snapEvery: 1000 }), 5)) !== JSON.stringify(hist(() => G.vae({ seed: 4, snapEvery: 1000 }), 5)), 'another seed gives another run');
for (const j of ['autoencoder', 'vae', 'gan2d', 'ganImages', 'diffusion2d', 'diffusionImages']) t.ok(T.jobs.has(j), 'job ' + j + ' is registered');
{ const job = T.jobs.make('gan2d', { seed: 2, snapEvery: 1000 }), r = job.step(); t.ok(Number.isFinite(r.loss) && r.metrics.dReal > 0, 'a GAN job steps'); }
t.done();
