// The training runs behind the topics of part 12 (transformers). A tiny decoder-only transformer (S.DL 'gpt') learns
// to copy, reverse or sort five digits. Training one takes about twenty seconds in this engine, far longer than a
// topic may take to draw, so each run is made here once, on a fixed seed, and what the topics show is kept in
// src/topics/p12-0-data.js:
//   - the training loss at every step;
//   - at a few checkpoints, the loss, the exact-match score on 100 new sequences (greedy decoding) and the attention
//     of every layer and head on one fixed probe sequence (bytes, 0..255 for weights 0..1), and the model's own
//     greedy answer for the probe;
//   - the final weights as exact float32 (base64), from which the topics rebuild the trained model and run it live.
//   node tests/ref/p12t_runs.mjs            every run, then the data file
//   node tests/ref/p12t_runs.mjs run <name> one run, into tests/ref/p12t/<name>.json
//   node tests/ref/p12t_runs.mjs combine    the data file from the run files
// tests/claims-p12.mjs re-runs the first steps of each run and checks that they match what is kept.
import { load } from '../load.mjs';
import { writeFileSync, readFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url)), OUT = join(HERE, 'p12t'), DATA = join(HERE, '..', '..', 'src', 'topics', 'p12-0-data.js');
const MODEL = { d: 16, heads: 2, layers: 2, ff: 32 };
export const PROBE = [2, 7, 4, 0, 5];
export const RUNS = {
  reverse: { task: 'reverse', steps: 400, keep: [0, 25, 50, 75, 100, 150, 200, 300, 400] },
  copy: { task: 'copy', steps: 300, keep: [0, 25, 50, 75, 100, 150, 200, 300] },
  sort: { task: 'sort', steps: 500, keep: [0, 25, 50, 75, 100, 150, 200, 300, 500] },
};
export const taskOpts = (R) => ({ L: 5, digits: 8, n: 4000, steps: R.steps });
export const setupOf = (S, R) => S.DL.setup({ task: R.task, seed: 1, taskOpts: taskOpts(R), config: MODEL, train: { batch: 32 } });
/** the probe as the model reads it: the digits, the separator and the answer but its last token */
export const probeInput = (S, R) => { const fn = { reverse: (x) => x.slice().reverse(), copy: (x) => x.slice(), sort: (x) => x.slice().sort((a, b) => a - b) }[R.task]; return PROBE.concat([8], fn(PROBE)).slice(0, -1); };
/** the attention of every layer and head on the probe: [{layer, head, weights}] */
export const probeAttn = (S, net, ids) => { net.predict(S.T.tensor([ids])); return S.DL.attentionMaps(net, 0); };

const b64 = (bytes) => Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
const sig = (v, n = 6) => +(+v).toPrecision(n);

function runOne(name) {
  const S = load(['core']), R = RUNS[name], { net, trainer, task } = setupOf(S, R), ids = probeInput(S, R), ckpts = [];
  const keep = (step) => {
    const maps = probeAttn(S, net, ids), ex = task.check(net, 100, 99), out = net.generate(PROBE.concat([8]), 5, { method: 'greedy' }).ids.slice(6);
    ckpts.push({ step, out, loss: step ? sig(trainer.history[step - 1].loss) : null, exact: ex.exact, n: ex.n, attn: b64(Uint8Array.from(maps.flatMap((m) => m.weights.flat()), (v) => Math.round(v * 255))) });
  };
  const t0 = Date.now();
  if (R.keep.includes(0)) keep(0);
  for (let s = 1; s <= R.steps; s++) { trainer.step(); if (R.keep.includes(s)) keep(s); }
  const params = net.namedParameters();
  const out = { name, task: R.task, model: Object.assign({ vocab: 9, context: 11, initStd: 0.3, seed: 1 }, MODEL), taskOpts: taskOpts(R), steps: R.steps, probe: ids, loss: trainer.history.map((h) => sig(h.loss, 5)), ckpts,
    names: params.map(([n, p]) => [n, p.shape]), weights: b64(Float32Array.from(params.flatMap(([, p]) => Array.from(p.values())))) };
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, name + '.json'), JSON.stringify(out));
  console.log(name + ': ' + R.steps + ' steps, final loss ' + out.loss.at(-1) + ', exact ' + ckpts.at(-1).exact + '/100, ' + ((Date.now() - t0) / 1000).toFixed(0) + ' s');
}

function combine() {
  const runs = {};
  for (const n of Object.keys(RUNS)) runs[n] = JSON.parse(readFileSync(join(OUT, n + '.json'), 'utf8'));
  const head = '// Made by tests/ref/p12t_runs.mjs from training runs of S.DL (a tiny transformer on copy, reverse and sort) on\n// fixed seeds: do not edit by hand. Losses as the engine wrote them; attention as bytes (0..255 for 0..1) and the\n// final weights as float32, both base64.\n';
  writeFileSync(DATA, head + '(function (S) {\n  \'use strict\';\n  S.P12D = ' + JSON.stringify({ runs }) + ';\n})(globalThis.Nooral = globalThis.Nooral || {});\n');
  console.log('wrote ' + DATA + ' (' + readFileSync(DATA).length + ' bytes)');
}

const [cmd, arg] = process.argv.slice(2);
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (cmd === 'run') runOne(arg);
  else if (cmd === 'combine') combine();
  else {
    await Promise.all(Object.keys(RUNS).map((n) => new Promise((res, rej) => { const p = spawn(process.execPath, [fileURLToPath(import.meta.url), 'run', n], { stdio: 'inherit' }); p.on('exit', (c) => (c ? rej(new Error(n + ' failed')) : res())); })));
    combine();
  }
}
