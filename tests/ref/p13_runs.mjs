// The training runs behind the topics of part 13 (language models). A tiny GPT takes minutes to train in this
// engine, far longer than a topic may take to draw, so the runs are made here once, on fixed seeds, from the text in
// src/topics/p13-0-text.js, and what the topics show is kept in src/topics/p13-0-data.js:
//   gpt    the character GPT (d 32, 2 layers, 4 heads, context 32): the loss of every step, the validation loss
//          every 50 steps, a sample at a few checkpoints, and the final weights as exact float32 (base64), from which
//          the topics rebuild the model and run it live (next-token bars, sampling, the KV cache, the limits)
//   scale  five widths of the same model trained for the same steps: the validation loss against the parameters
//   icl    a GPT trained on sequences of key–value pairs whose meaning changes from sequence to sequence (keys a–f,
//          values 0–9): it can only answer a repeated key by reading the earlier pair in its context
//   node tests/ref/p13_runs.mjs            every job (in parallel), then the data file
//   node tests/ref/p13_runs.mjs job <name> one job, into tests/ref/p13/<name>.json
//   node tests/ref/p13_runs.mjs combine    the data file from the job files
// tests/claims-p13.mjs re-runs the first steps of each run and checks that they match what is kept.
import { load } from '../load.mjs';
import { writeFileSync, readFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import vm from 'node:vm';

const HERE = dirname(fileURLToPath(import.meta.url)), OUT = join(HERE, 'p13'), SRC = join(HERE, '..', '..', 'src');
const DATA = join(SRC, 'topics', 'p13-0-data.js');

export const GPT = { d: 32, heads: 4, layers: 2, context: 32, dropout: 0.1 };
export const GPT_TRAIN = { batch: 16, lr: 3e-3, warmup: 30, seed: 1 };
export const GPT_STEPS = 1600;
export const CHECKPOINTS = [0, 50, 100, 200, 400, 800, 1600];
export const PROMPT = 'the sailor ';
export const SAMPLE = { n: 64, temperature: 0.8, seed: 7 };
export const SCALE = { widths: [8, 16, 24, 32, 48], heads: 2, layers: 2, steps: 400, evalEvery: 50 };
export const ICL = { keys: 6, pairs: 8, n: 6000, val: 300, d: 32, heads: 2, layers: 2, ff: 64, steps: 2500, seed: 5 };

/** the engines and the part's text, loaded as the page loads them */
export function engines() {
  const S = load(['core']);
  vm.runInThisContext(readFileSync(join(SRC, 'topics', 'p13-0-text.js'), 'utf8'), { filename: 'p13-0-text.js' });
  return S;
}
export const gptSetup = (S, config, train, steps) => S.DL.setup({ task: 'text', taskOpts: { text: S.P13T.harbour, context: GPT.context, stride: 4, steps }, config, train: Object.assign({}, GPT_TRAIN, train, { steps }), seed: 1 });
/** the in-context task: digits 0–9 are values, 10 + k is key k; the target is scored at the value places only */
export function iclTask(S) {
  const T = S.T, g = T.rng(ICL.seed), K = ICL.keys, P = ICL.pairs;
  const one = () => { const map = Array.from({ length: K }, () => g.int(10)), s = []; for (let i = 0; i < P; i++) { const k = g.int(K); s.push(10 + k, map[k]); } return s; };
  const batchOf = (list) => ({ x: T.tensor(list.map((s) => s.slice(0, -1))), y: list.map((s) => s.slice(1).map((v, k) => (k % 2 === 0 ? v : -1))) });
  const train = Array.from({ length: ICL.n }, one), val = Array.from({ length: ICL.val }, one);
  return { train: S.DL.split(train, batchOf), val: S.DL.split(val, batchOf), examples: val };
}
export const iclConfig = { vocab: 10 + ICL.keys, d: ICL.d, heads: ICL.heads, layers: ICL.layers, context: 2 * ICL.pairs, ff: ICL.ff, initStd: 0.3, seed: 1 };
export const iclTrain = { batch: 32, lr: 3e-3, optimizer: 'adamw', weightDecay: 0.01, clip: 1, schedule: 'cosine', warmup: 30, steps: ICL.steps, seed: 1 };

const b64 = (arr) => Buffer.from(Float32Array.from(arr).buffer).toString('base64');
const sig = (v, n = 6) => +(+v).toPrecision(n);
const weights = (net) => Object.fromEntries(net.namedParameters().map(([n, p]) => [n, { shape: p.shape.slice(), d: b64(p.values()) }]));

function runGpt() {
  const S = engines(), { net, trainer, task } = gptSetup(S, GPT, {}, GPT_STEPS);
  const out = { config: Object.assign({ vocab: task.tokenizer.size }, GPT), vocab: task.tokenizer.vocab, params: net.numParams(), steps: GPT_STEPS, hist: [], val: [], samples: [], nTrain: task.train.n, nVal: task.val.n };
  const sample = () => out.samples.push({ step: trainer.stepNo, text: net.complete(PROMPT, SAMPLE.n, { temperature: SAMPLE.temperature }, SAMPLE.seed).text });
  const ev = () => { const v = trainer.evaluate('val', 10000); out.val.push([trainer.stepNo, sig(v.loss)]); };
  ev(); sample();
  for (let k = 1; k <= GPT_STEPS; k++) {
    const r = trainer.step();
    out.hist.push(sig(r.loss, 5));
    if (k % 50 === 0) ev();
    if (CHECKPOINTS.includes(k)) sample();
  }
  out.weights = weights(net);
  return out;
}
function runScale() {
  const S = engines(), out = { runs: [] };
  for (const d of SCALE.widths) {
    const { net, trainer } = gptSetup(S, { d, heads: SCALE.heads, layers: SCALE.layers, context: GPT.context, dropout: 0.1 }, {}, SCALE.steps);
    const run = { d, params: net.numParams(), val: [], hist: [] };
    for (let k = 1; k <= SCALE.steps; k++) {
      const r = trainer.step();
      run.hist.push(sig(r.loss, 5));
      if (k % SCALE.evalEvery === 0) run.val.push([k, sig(trainer.evaluate('val', 10000).loss)]);
    }
    out.runs.push(run);
    console.log('scale d=' + d + ': ' + run.val[run.val.length - 1][1]);
  }
  return out;
}
function runIcl() {
  const S = engines(), task = iclTask(S), net = S.DL.build('gpt', iclConfig), trainer = new S.DL.Trainer(net, task, iclTrain);
  const out = { config: iclConfig, params: net.numParams(), steps: ICL.steps, hist: [], val: [] };
  for (let k = 1; k <= ICL.steps; k++) {
    const r = trainer.step();
    out.hist.push(sig(r.loss, 5));
    if (k % 100 === 0) out.val.push([k, sig(trainer.evaluate('val', 10000).loss)]);
  }
  out.weights = weights(net);
  return out;
}
const RUNS = { gpt: runGpt, scale: runScale, icl: runIcl };

function combine() {
  const all = {};
  for (const name of Object.keys(RUNS)) all[name] = JSON.parse(readFileSync(join(OUT, name + '.json'), 'utf8'));
  const head = '// Made by tests/ref/p13_runs.mjs from training runs of the engines in src/core (S.DL) on the text of p13-0-text.js,\n// on fixed seeds: do not edit by hand. Losses as written by the engine; weights exact float32 in base64 (S.P13.f32).\n';
  writeFileSync(DATA, head + '// This file: the character GPT. p13-0-data2.js: the scaling runs and the in-context model.\n(function (S) {\n  \'use strict\';\n  S.P13D = Object.assign(S.P13D || {}, ' + JSON.stringify({ gpt: all.gpt }) + ');\n})(globalThis.Nooral = globalThis.Nooral || {});\n');
  writeFileSync(DATA.replace('data.js', 'data2.js'), head + '(function (S) {\n  \'use strict\';\n  S.P13D = Object.assign(S.P13D || {}, ' + JSON.stringify({ scale: all.scale, icl: all.icl }) + ');\n})(globalThis.Nooral = globalThis.Nooral || {});\n');
  console.log('wrote ' + DATA + ' and data2.js');
}

const [cmd, arg] = process.argv.slice(2);
if (import.meta.url === 'file://' + process.argv[1]) {
  mkdirSync(OUT, { recursive: true });
  if (cmd === 'job') { const t0 = Date.now(), r = RUNS[arg](); r.ms = Date.now() - t0; writeFileSync(join(OUT, arg + '.json'), JSON.stringify(r)); console.log(arg + ': ' + r.ms + ' ms'); }
  else if (cmd === 'combine') combine();
  else {
    await Promise.all(Object.keys(RUNS).map((name) => new Promise((res, rej) => { const p = spawn(process.execPath, [fileURLToPath(import.meta.url), 'job', name], { stdio: 'inherit' }); p.on('exit', (c) => (c ? rej(new Error(name + ' failed')) : res())); })));
    combine();
  }
}
