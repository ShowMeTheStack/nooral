// The training runs behind the topics of part 10 (convolutional networks). A small CNN takes tens of seconds to train
// in this engine, far longer than a topic may take to draw, so the runs are made here once, on fixed seeds, and what
// the topics show is kept in src/topics/p10-0-data.js: the training curves, the validation predictions at a few
// steps (for the confusion matrix), the filters at those steps, a set of validation images, and the final weights
// (as exact float32), from which the topics rebuild the trained network and run it live.
//   node tests/ref/p10_runs.mjs            every job, then the data file
//   node tests/ref/p10_runs.mjs job <name> one job, into tests/ref/p10/<name>.json
//   node tests/ref/p10_runs.mjs combine    the data file from the job files
// tests/claims-p10.mjs re-runs the first steps of each job and checks that they match what is kept.
import { load } from '../load.mjs';
import { writeFileSync, readFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url)), OUT = join(HERE, 'p10'), DATA = join(HERE, '..', '..', 'src', 'topics', 'p10-0-data.js');
const CNN = { filters: [6, 12], kernel: 5, hidden: [32] };
// settings are exported so the claims check rebuilds each run the same way
export const JOBS = {
  digits: { task: 'digits', taskOpts: { n: 2000, val: 300, size: 16, seed: 1 }, config: CNN, train: { seed: 1 }, steps: 600, evalEvery: 25, keep: [0, 25, 100, 600], weights: true, images: true },
  shapes: { task: 'shapes', taskOpts: { n: 1200, val: 300, size: 16, seed: 2 }, config: CNN, train: { seed: 2 }, steps: 400, evalEvery: 25, keep: [0, 25, 100, 400], weights: true, images: true },
  augOff: { task: 'digits', taskOpts: { n: 40, val: 300, size: 16, seed: 1 }, config: CNN, train: { seed: 3 }, steps: 600, evalEvery: 25, keep: [] },
  augOn: { task: 'digits', taskOpts: { n: 40, val: 300, size: 16, seed: 1, augment: { shift: 2, rotate: 0.3, scale: 0.1 } }, config: CNN, train: { seed: 3 }, steps: 600, evalEvery: 25, keep: [] },
};
export const setupOf = (S, J) => S.DL.setup({ kind: 'cnn', task: J.task, taskOpts: J.taskOpts, config: J.config, seed: J.taskOpts.seed, train: J.train });

const b64 = (bytes) => Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
const u8 = (vals) => b64(Uint8Array.from(vals, (v) => Math.max(0, Math.min(255, Math.round(v * 255)))));
const f32 = (vals) => b64(Float32Array.from(vals));
const sig = (v, n = 6) => +(+v).toPrecision(n);

function runJob(name) {
  const S = load(['core']), J = JOBS[name], { net, task, trainer } = setupOf(S, J), T = S.T;
  const out = { name, steps: J.steps, task: J.task, classes: task.classes, nTrain: task.train.n, nVal: task.val.n, augment: J.taskOpts.augment || null, hist: [], val: [], snaps: [] };
  const valPreds = () => {
    const n = task.val.n, b = task.val.get([...Array(n).keys()], T.rng(7)), lg = net.predict(b.x);
    return Array.from(T.argmax(lg, -1).values()).join('');
  };
  const evalBoth = () => { const v = trainer.evaluate('val', 300), t = trainer.evaluate('train', 300); return { step: trainer.stepNo, valLoss: sig(v.loss), valAcc: sig(v.accuracy), trainLoss: sig(t.loss), trainAcc: sig(t.accuracy) }; };
  const snap = () => { const p = Object.fromEntries(net.namedParameters()); out.snaps.push({ step: trainer.stepNo, preds: valPreds(), conv1: f32(p['conv1.W'].values()) }); };
  out.val.push(evalBoth());
  if (J.keep.includes(0)) snap();
  const t0 = Date.now();
  for (let k = 1; k <= J.steps; k++) {
    const r = trainer.step();
    out.hist.push([sig(r.loss, 5), r.metrics.accuracy]);
    if (k % J.evalEvery === 0) out.val.push(evalBoth());
    if (J.keep.includes(k)) snap();
  }
  out.ms = Date.now() - t0;
  if (J.weights) out.weights = Object.fromEntries(net.namedParameters().map(([n, p]) => [n, { shape: p.shape.slice(), d: f32(p.values()) }]));
  if (J.images) {
    const b = task.val.get([...Array(task.val.n).keys()], T.rng(7));
    out.images = u8(b.x.values()); out.y = b.y.join('');
  }
  if (J.taskOpts.augment) {
    // eight augmented copies of the first training image, as the trainer would draw them (rng seed 11)
    const e = task.train.examples[0], g = T.rng(11);
    out.augShow = { y: e.y, src: u8(e.px), copies: Array.from({ length: 8 }, () => u8(S.DL.augmentImage(e.px, 16, 16, g, J.taskOpts.augment))) };
  }
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, name + '.json'), JSON.stringify(out));
  console.log(name + ': ' + out.ms + ' ms, final val accuracy ' + out.val[out.val.length - 1].valAcc);
}

function combine() {
  const all = {};
  for (const name of Object.keys(JOBS)) all[name] = JSON.parse(readFileSync(join(OUT, name + '.json'), 'utf8'));
  writeFileSync(DATA, '// Made by tests/ref/p10_runs.mjs from training runs of the engines in src/core (S.DL) on fixed seeds: do not\n// edit by hand. Weights are exact float32 and images bytes, both in base64 (S.P10.f32, S.P10.u8).\n(function (S) {\n  \'use strict\';\n  S.P10D = ' + JSON.stringify(all) + ';\n})(globalThis.Nooral = globalThis.Nooral || {});\n');
  console.log('wrote ' + DATA);
}

const [cmd, arg] = process.argv.slice(2);
if (import.meta.url === 'file://' + process.argv[1]) {
  if (cmd === 'job') runJob(arg);
  else if (cmd === 'combine') combine();
  else {
    await Promise.all(Object.keys(JOBS).map((name) => new Promise((res, rej) => { const p = spawn(process.execPath, [fileURLToPath(import.meta.url), 'job', name], { stdio: 'inherit' }); p.on('exit', (c) => (c ? rej(new Error(name + ' failed')) : res())); })));
    combine();
  }
}
