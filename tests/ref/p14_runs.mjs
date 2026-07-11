// The training runs behind the topics of part 14 (generative models). Training these models takes from seconds to
// minutes in this engine, far longer than a topic may take to draw, so they are run here once, on fixed seeds, and
// what the topics show is kept in src/topics/p14-0-data.js: training curves, snapshots (reconstructions, latent
// maps, samples, the discriminator's field) and, where the topic runs the model live, its final weights.
//   node tests/ref/p14_runs.mjs            every job, four at a time, then the data file
//   node tests/ref/p14_runs.mjs job <name> one job, into the system's temporary folder (nooral-p14-runs/<name>.json)
//   node tests/ref/p14_runs.mjs combine    the data file from the job files
//   node tests/ref/p14_runs.mjs combine --reuse   the same, taking a job with no job file from the data file as it
//                                          is now, when its settings are unchanged
// tests/claims-p14.mjs re-runs the start of each job and checks that it matches what is kept.
import { load } from '../load.mjs';
import { writeFileSync, readFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url)), OUT = join(tmpdir(), 'nooral-p14-runs'), DATA = join(HERE, '..', '..', 'src', 'topics', 'p14-0-data.js');

// ---- the jobs: their settings are exported so the claims check re-runs them the same way -----------------------
export const VAE_CLASSES = [0, 1, 3, 7];
export const JOBS = {
  ae2: { kind: 'ae', o: { latent: 2, n: 512, hidden: 64, snapEvery: 25, show: 8, seed: 1 }, steps: 600, keep: [0, 25, 50, 100, 200, 400, 600] },
  ae4: { kind: 'ae', o: { latent: 4, n: 512, hidden: 64, snapEvery: 25, show: 8, seed: 1 }, steps: 600, keep: [0, 25, 50, 100, 200, 400, 600] },
  ae8: { kind: 'ae', o: { latent: 8, n: 512, hidden: 64, snapEvery: 25, show: 8, seed: 1 }, steps: 600, keep: [0, 25, 50, 100, 200, 400, 600] },
  vae0: { kind: 'vae', o: { beta: 0, n: 512, hidden: 64, classes: VAE_CLASSES, snapEvery: 50, seed: 1, gridEvery: false }, steps: 1500, keep: [0, 50, 150, 400, 800, 1500] },
  vae1: { kind: 'vae', o: { beta: 1, n: 512, hidden: 64, classes: VAE_CLASSES, snapEvery: 50, seed: 1, gridEvery: false }, steps: 1500, keep: [0, 50, 150, 400, 800, 1500], weights: true },
  vae4: { kind: 'vae', o: { beta: 4, n: 512, hidden: 64, classes: VAE_CLASSES, snapEvery: 50, seed: 1, gridEvery: false }, steps: 1500, keep: [0, 50, 150, 400, 800, 1500] },
  vae16: { kind: 'vae', o: { beta: 16, n: 512, hidden: 64, classes: VAE_CLASSES, snapEvery: 50, seed: 1, gridEvery: false }, steps: 1500, keep: [0, 50, 150, 400, 800, 1500] },
  ganGood: { kind: 'gan', o: { preset: 'good', dist: 'ring', snapEvery: 100, seed: 1 }, steps: 1500, keep: [0, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500] },
  ganCollapse: { kind: 'gan', o: { preset: 'collapse', lrG: 4e-3, lrD: 4e-4, dist: 'ring', snapEvery: 100, seed: 1 }, steps: 1500, keep: [0, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500] },
  ganImg: { kind: 'ganImg', o: { preset: 'steady', classes: [0, 1, 7], snapEvery: 100, show: 16, seed: 1 }, steps: 4000, keep: [0, 100, 300, 600, 1000, 2000, 3000, 4000] },
  ganImgPlain: { kind: 'ganImg', o: { preset: 'plain', classes: [0, 1, 7], snapEvery: 100, show: 16, seed: 1 }, steps: 1500, keep: [0, 100, 200, 400, 700, 1000, 1500] },
  judge: { kind: 'judge', o: { hidden: 32, n: 4000, seed: 5 }, steps: 3000, test: { n: 1000, seed: 78 } },
  diff2d: { kind: 'diff2d', o: { dist: 'moons', steps: 100, snapEvery: 250, seed: 1 }, steps: 3000, keep: [0, 250, 500, 1000, 2000, 3000] },
  diffImg: { kind: 'diffImg', o: { data: 'shapes', classes: [0], n: 2048, steps: 60, snapEvery: 100000, seed: 1 }, steps: 1000, sample: { n: 8, seed: 7, frames: 10, temp: 0.7 }, more: { n: 64, seed: 8, frames: 1, temp: 0.7 } },
};
export const GRID_K = 7, GRID_RANGE = 2.5, GAN_SAMPLE_SEED = 4242;

// ---- compact text for numbers -----------------------------------------------------------------------------------
const b64 = (bytes) => Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
/** values in [0, 1] as bytes */
export const u8 = (vals) => b64(Uint8Array.from(vals, (v) => Math.max(0, Math.min(255, Math.round(v * 255)))));
/** any values as 16-bit integers with one scale: {s, d} */
export const i16 = (vals) => { const m = Math.max(1e-12, ...Array.from(vals, Math.abs)), s = m / 32767; return { s, d: b64(Int16Array.from(vals, (v) => Math.round(v / s))) }; };
const sig = (v, n = 6) => (Number.isFinite(v) ? +(+v).toPrecision(n) : v);
const every = (arr, k) => arr.filter((_, i) => (i + 1) % k === 0);

function runJob(name) {
  const S = load(['core']), G = S.Gen, T = S.T, J = JOBS[name], out = { name, kind: J.kind, o: J.o, steps: J.steps };
  const t0 = Date.now();
  if (J.kind === 'ae') {
    const tr = G.autoencoder(J.o), show = [...Array(J.o.show).keys()], X = G.batchOf(tr.test, show);
    out.snaps = [];
    const codes = () => Array.from(tr.encode(X).values(), (v) => sig(v, 5)), codes0 = codes();
    const snap = (c) => { const s = tr.snapshots[tr.snapshots.length - 1]; out.snaps.push({ step: s.step, loss: sig(s.loss), recon: u8(s.recon.flatMap((a) => Array.from(a))), codes: c }); };
    for (let k = 1; k <= J.steps; k++) { tr.step(); if (k === 1) snap(codes0); if (J.keep.includes(k)) snap(codes()); }
    out.loss = every(tr.history.map((h) => sig(h.loss, 5)), 5);
    out.testLoss = sig(tr.testLoss());
  } else if (J.kind === 'vae') {
    const tr = G.vae(J.o), e0 = tr.elbo();
    out.snaps = [];
    for (let k = 1; k <= J.steps; k++) {
      tr.step();
      const s = tr.snapshots[tr.snapshots.length - 1];
      if ((k === 1 || J.keep.includes(k)) && !out.snaps.some((x) => x.step === s.step)) out.snaps.push(Object.assign({ step: s.step }, (({ elbo, recon, kl }) => ({ elbo: sig(elbo), recon: sig(recon), kl: sig(kl) }))(s.step ? tr.elbo() : e0), { map: i16(s.map.flatMap((m) => m.z)), y: s.map.map((m) => m.y).join('') }));
    }
    const h = tr.history;
    out.hist = { elbo: every(h.map((r) => sig(r.elbo, 5)), 10), recon: every(h.map((r) => sig(r.recon, 5)), 10), kl: every(h.map((r) => sig(r.kl, 5)), 10) };
    out.final = (() => { const e = tr.elbo(); return { elbo: sig(e.elbo), recon: sig(e.recon), kl: sig(e.kl) }; })();
    const g = tr.decodeGrid(GRID_K, GRID_RANGE);
    out.grid = u8(g.images.flatMap((a) => Array.from(a)));
    if (J.weights) out.weights = G.copyParams(tr.net).map(i16);
  } else if (J.kind === 'gan') {
    const tr = G.gan2d(J.o);
    out.snaps = [];
    for (let k = 1; k <= J.steps; k++) {
      tr.step();
      const s = tr.snapshots[tr.snapshots.length - 1];
      if ((k === 1 || J.keep.includes(k)) && !out.snaps.some((x) => x.step === s.step)) out.snaps.push({ step: s.step, samples: i16(s.samples.flat()), field: u8(s.field.d), k: s.field.k, swd: sig(s.swd), covered: s.coverage.covered, effective: sig(s.coverage.effective) });
    }
    const h = tr.history;
    out.hist = { lossD: every(h.map((r) => sig(r.lossD, 4)), 10), lossG: every(h.map((r) => sig(r.lossG, 4)), 10), dReal: every(h.map((r) => sig(r.dReal, 4)), 10), dFake: every(h.map((r) => sig(r.dFake, 4)), 10) };
  } else if (J.kind === 'ganImg') {
    const tr = G.ganImages(J.o);
    out.snaps = [];
    for (let k = 1; k <= J.steps; k++) {
      tr.step();
      const s = tr.snapshots[tr.snapshots.length - 1];
      if ((k === 1 || J.keep.includes(k)) && !out.snaps.some((x) => x.step === s.step)) out.snaps.push({ step: s.step, images: u8(s.images.flatMap((a) => Array.from(a))), dFake: s.dFake.map((v) => sig(v, 5)), dReal: s.dReal.map((v) => sig(v, 5)) });
    }
    const h = tr.history;
    out.hist = { lossD: every(h.map((r) => sig(r.lossD, 4)), 10), lossG: every(h.map((r) => sig(r.lossG, 4)), 10), dReal: every(h.map((r) => sig(r.dReal, 4)), 10), dFake: every(h.map((r) => sig(r.dFake, 4)), 10) };
    // 64 samples from fresh noise of their own seed, for counting what the generator makes
    out.sampleSeed = GAN_SAMPLE_SEED;
    out.final64 = u8(tr.samples(64, GAN_SAMPLE_SEED).flatMap((a) => Array.from(a)));
  } else if (J.kind === 'judge') {
    const jn = G.judgeNet(J.o);
    out.loss = sig(jn.train(J.steps));
    out.accuracy = sig(jn.accuracy(G.images('digits', { n: J.test.n, size: 16, seed: J.test.seed })));
    out.weights = G.copyParams(jn.net).map(i16);
  } else if (J.kind === 'diff2d') {
    const tr = G.diffusion2d(J.o);
    out.snaps = [];
    for (let k = 1; k <= J.steps; k++) {
      tr.step();
      const s = tr.snapshots[tr.snapshots.length - 1];
      if ((k === 1 || J.keep.includes(k)) && !out.snaps.some((x) => x.step === s.step)) out.snaps.push({ step: s.step, samples: i16(s.samples.flat()), swd: sig(s.swd) });
    }
    out.loss = every(tr.history.map((r) => sig(r.loss, 4)), 10);
    out.weights = G.copyParams(tr.net).map(i16);
  } else if (J.kind === 'diffImg') {
    const tr = G.diffusionImages(J.o);
    for (let k = 1; k <= J.steps; k++) tr.step();
    out.loss = every(tr.history.map((r) => sig(r.loss, 4)), 10);
    out.sample = J.sample;
    const s = tr.sample(J.sample);
    out.frames = s.frames.map((f) => ({ t: f.t, images: u8(f.images.flatMap((a) => Array.from(a))) }));
    out.final = u8(s.final.flatMap((a) => Array.from(a)));
    // 64 more samples from noise of another seed, for an honest count of how many come out clean
    if (J.more) out.final64 = u8(tr.sample(J.more).final.flatMap((a) => Array.from(a)));
  }
  out.seconds = Math.round((Date.now() - t0) / 1000);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, name + '.json'), JSON.stringify(out));
  console.log(name + ' done in ' + out.seconds + ' s');
}

/** the runs kept in the data file now (to reuse a job that was not re-run, when its settings are unchanged) */
function kept() {
  if (!existsSync(DATA)) return {};
  const box = { Nooral: {} };
  new Function('globalThis', readFileSync(DATA, 'utf8'))(box);
  return box.Nooral.P14D || {};
}

function combine(reuse) {
  const all = {}, old = reuse ? kept() : {};
  for (const name of Object.keys(JOBS)) {
    const f = join(OUT, name + '.json');
    if (!existsSync(f) && reuse && old[name] && JSON.stringify(old[name].o) === JSON.stringify(JOBS[name].o) && old[name].steps === JOBS[name].steps) { all[name] = old[name]; console.log('kept ' + name + ' from the data file'); continue; }
    if (!existsSync(f)) throw new Error('missing job ' + name + ': run node tests/ref/p14_runs.mjs job ' + name);
    const j = JSON.parse(readFileSync(f, 'utf8'));
    delete j.seconds;
    all[name] = j;
  }
  const text = '// Made by tests/ref/p14_runs.mjs from training runs of the engines in src/core (S.Gen) on fixed seeds: do not\n'
    + '// edit by hand. Numbers are kept as written by the engine; images and weights as base64 bytes (S.P14.u8, S.P14.i16).\n'
    + '(function (S) {\n  \'use strict\';\n  S.P14D = ' + JSON.stringify(all) + ';\n})(globalThis.Nooral = globalThis.Nooral || {});\n';
  writeFileSync(DATA, text);
  console.log('wrote ' + DATA + ' (' + Math.round(text.length / 1024) + ' KB)');
}

const [cmd, arg] = process.argv.slice(2);
if (process.argv[1] && process.argv[1].endsWith('p14_runs.mjs')) {
  if (cmd === 'job') runJob(arg);
  else if (cmd === 'combine') combine(arg === '--reuse');
  else if (!cmd) {
    const names = Object.keys(JOBS), me = fileURLToPath(import.meta.url);
    let next = 0;
    const go = () => new Promise((done) => {
      const one = () => { if (next >= names.length) return done(); const n = names[next++]; spawn(process.execPath, [me, 'job', n], { stdio: 'inherit' }).on('exit', one); };
      one();
    });
    await Promise.all([go(), go(), go(), go()]);
    combine();
  }
}
