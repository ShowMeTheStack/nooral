// The training runs behind the topics of part 11 (sequences and embeddings). Recurrent networks take seconds to
// minutes to train in this engine, longer than a topic may take to draw, so they are trained here once, on fixed
// seeds, and src/topics/p11-0-data.js keeps what the topics show: the loss history, validation scores, samples at a
// few steps (for the character model) and the final weights as exact float32 in base64, from which the topics
// rebuild each network and run it live. word2vec is quick enough to train inside its topics, so it is not here.
//   node tests/ref/p11_runs.mjs            every job, then the data file
//   node tests/ref/p11_runs.mjs <job>      one job (the data file keeps the others as they were)
// tests/claims-p11.mjs replays every job (whole or its first steps) and checks it matches what is kept.
import { load } from '../load.mjs';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const HERE = dirname(fileURLToPath(import.meta.url)), DATA = join(HERE, '..', '..', 'src', 'topics', 'p11-0-data.js');

/** the engines plus part 11's common file (the corpus and the encoder–decoder) */
export function engines() {
  const S = load(['core', 'stage']);
  vm.runInThisContext(readFileSync(join(HERE, '..', '..', 'src', 'topics', 'p11-0-common.js'), 'utf8'), { filename: 'p11-0-common.js' });
  return S;
}

// ---- tasks of our own --------------------------------------------------------------------------------------------
/** the most recent symbol: a stream of blanks (0) and symbols A, B, C (1..3); the answer at each step is the last
 *  symbol seen so far (0 while there has been none) */
export function lastSymbolTask(S, o) {
  const len = o.len || 10, g = S.T.rng(o.seed || 1), p = o.p || 0.35;
  const make = (n) => Array.from({ length: n }, () => {
    let last = 0;
    const x = [], y = [];
    for (let k = 0; k < len; k++) { const s = g.next() < p ? 1 + g.int(3) : 0; if (s) last = s; x.push(s); y.push(last); }
    return { x, y };
  });
  const batchOf = (list) => ({ x: S.T.tensor(list.map((e) => e.x)), y: list.map((e) => e.y) });
  const val = S.DL.split(make(o.val || 200), batchOf);
  return { kind: 'rnn', train: S.DL.split(make(o.n || 512), batchOf), val, probe: val.get([0]),
    model: { cell: 'gru', vocab: 4, embed: 4, hidden: 4, outputs: 4 }, fit: { batch: 32, lr: 2e-2 } };
}

/** number words to digits, for the encoder–decoder: each batch has one length, drawn from 1..maxLen */
export function wordsToDigits(S, seed) {
  const g = S.T.rng(seed);
  return (B, L) => {
    const src = [], tin = [], tout = [];
    for (let b = 0; b < B; b++) {
      const d = Array.from({ length: L }, () => g.int(10));
      src.push(d.slice()); tin.push([S.P11.S2S.START].concat(d)); tout.push(d.concat([S.P11.S2S.END]));
    }
    return { src, tin, tout };
  };
}

export const JOBS = {
  rnnCount: { kind: 'rnn', setup: { task: 'counting', taskOpts: { len: 8, n: 512, seed: 1 }, config: { cell: 'rnn', hidden: 4 }, train: { lr: 2e-2, seed: 1 } }, steps: 800, evalEvery: 100 },
  lstmCount: { kind: 'rnn', setup: { task: 'counting', taskOpts: { len: 8, n: 512, seed: 1 }, config: { cell: 'lstm', hidden: 4 }, train: { lr: 2e-2, seed: 1 } }, steps: 800, evalEvery: 100 },
  gruLast: { kind: 'rnn', own: 'last', taskOpts: { len: 10, n: 512, seed: 9, p: 0.25 }, train: { lr: 2e-2, seed: 9 }, steps: 600, evalEvery: 100 },
  charLM: { kind: 'rnn', setup: { task: 'chars', taskOpts: { len: 32, stride: 3, seed: 1 }, config: { embed: 12, hidden: 48 }, train: { lr: 8e-3, seed: 1, batch: 16, clip: 1 } }, steps: 1500, evalEvery: 250, samples: [0, 100, 400, 1500], prompt: 'the ' },
  seq2seq: { kind: 'seq2seq', config: { vin: 10, vout: 12, e: 12, h: 32, seed: 2 }, steps: 1500, batch: 32, maxLen: 5, lr: 1e-2, seed: 2, evalEvery: 250 },
};

/** a trainer for a job (a DL.Trainer, or the encoder–decoder's own small loop with the same step()/history) */
export function trainerFor(S, name) {
  const J = JOBS[name];
  if (J.kind === 'seq2seq') {
    const T = S.T, net = S.P11.seq2seq(J.config), opt = T.optim.make('adam', net.parameters(), { lr: J.lr }), data = wordsToDigits(S, J.seed), g = T.rng(J.seed + 100);
    const tr = { net, history: [], stepNo: 0 };
    tr.step = () => {
      const L = 1 + g.int(J.maxLen), b = data(J.batch, L);
      net.train();
      opt.zeroGrad();
      const logits = net.forward(T.tensor(b.src), T.tensor(b.tin)), loss = T.crossEntropy(logits, b.tout);
      loss.backward();
      T.clipGradNorm(net.parameters(), 1);
      opt.step();
      tr.stepNo++;
      tr.history.push({ step: tr.stepNo, loss: loss.item() });
    };
    /** exact matches of greedy decoding on fresh inputs of each length */
    tr.exact = (L, n, seed) => {
      const gg = T.rng(seed);
      let right = 0;
      for (let i = 0; i < n; i++) { const d = Array.from({ length: L }, () => gg.int(10)); if (net.translate(d, L + 2).out.join() === d.join()) right++; }
      return right / n;
    };
    return tr;
  }
  let net, trainer, task;
  if (J.own === 'last') {
    task = lastSymbolTask(S, J.taskOpts);
    net = S.DL.build('rnn', Object.assign({ seed: J.taskOpts.seed }, task.model));
    trainer = new S.DL.Trainer(net, task, Object.assign({}, task.fit, J.train));
  } else {
    const setup = JSON.parse(JSON.stringify(J.setup));
    if (setup.task === 'chars') setup.taskOpts.text = S.P11.charText();
    ({ net, trainer, task } = S.DL.setup(setup));
  }
  return trainer;
}

const sig = (v, n = 7) => +(+v).toPrecision(n);
const b64 = (arr) => Buffer.from(Float32Array.from(arr).buffer).toString('base64');

export function runJob(S, name, steps) {
  const J = JOBS[name], tr = trainerFor(S, name), net = tr.net, out = { name, kind: J.kind, steps: steps || J.steps, evals: [], samples: [] };
  const total = steps || J.steps;
  const sample = () => {
    const r0 = net.generate(net.tokenizer.encode(J.prompt), 60, { method: 'greedy' }), r1 = net.generate(net.tokenizer.encode(J.prompt), 60, { temperature: 0.8 }, 5);
    out.samples.push({ step: tr.stepNo, greedy: r0.text, sampled: r1.text });
  };
  const evaluate = () => {
    if (J.kind === 'seq2seq') out.evals.push({ step: tr.stepNo, exact: [1, 2, 3, 4, 5].map((L) => sig(tr.exact(L, 50, 77), 4)) });
    else { const v = tr.evaluate('val', 200); out.evals.push({ step: tr.stepNo, loss: sig(v.loss), accuracy: v.accuracy === undefined ? null : sig(v.accuracy) }); }
  };
  if (J.samples && J.samples.includes(0)) sample();
  for (let i = 0; i < total; i++) {
    tr.step();
    if (J.evalEvery && tr.stepNo % J.evalEvery === 0) evaluate();
    if (J.samples && J.samples.includes(tr.stepNo)) sample();
  }
  out.loss = tr.history.map((h) => sig(h.loss, 5));
  if (J.kind === 'seq2seq') {
    out.config = J.config;
    out.lengths = Array.from({ length: 8 }, (_, k) => ({ L: k + 1, exact: sig(tr.exact(k + 1, 100, 91), 4) }));
  } else out.config = JSON.parse(JSON.stringify(net.config));
  const named = net.namedParameters();
  out.weights = { names: named.map((x) => x[0]), shapes: named.map((x) => x[1].shape.slice()), b64: b64(named.flatMap((x) => Array.from(x[1].values()))) };
  return out;
}

if (process.argv[1] && process.argv[1].endsWith('p11_runs.mjs')) {
  const S = engines(), only = process.argv.slice(2);
  let data = {};
  if (existsSync(DATA)) { const m = /S\.P11D = (\{[\s\S]*\});\n/.exec(readFileSync(DATA, 'utf8')); if (m) data = JSON.parse(m[1]); }
  for (const name of Object.keys(JOBS)) {
    if (only.length && !only.includes(name)) continue;
    const t0 = Date.now();
    data[name] = runJob(S, name);
    const e = data[name].evals[data[name].evals.length - 1];
    console.log(name + ': ' + ((Date.now() - t0) / 1000).toFixed(1) + ' s, last eval ' + JSON.stringify(e));
  }
  const head = '// Made by tests/ref/p11_runs.mjs from training runs of the engines in src/core on fixed seeds: do not edit by hand.\n// Losses and scores as the engine gave them (to 5-7 significant digits); weights as exact float32 in base64 (S.P11.stateOf).\n';
  writeFileSync(DATA, head + '(function (S) {\n  \'use strict\';\n  S.P11D = ' + JSON.stringify(data) + ';\n})(globalThis.Nooral = globalThis.Nooral || {});\n');
  console.log('wrote ' + DATA + ' (' + Math.round(readFileSync(DATA).length / 1024) + ' KB)');
}
