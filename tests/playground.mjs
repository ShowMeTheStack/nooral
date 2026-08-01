// The playground's network: deterministic for a seed, and it learns (node tests/playground.mjs).
import { files, harness } from './load.mjs';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
// the core engines and this page's own engine file only, so another page's file being written does not stop the test
const ctx = vm.createContext({ console, performance, setTimeout, clearTimeout, setInterval, clearInterval, queueMicrotask, structuredClone, TextEncoder, TextDecoder, URL });
for (const f of files(['core']).concat(['app/playground-a-model.js'])) vm.runInContext(readFileSync(new URL('../src/' + f, import.meta.url), 'utf8'), ctx, { filename: f });
const S = ctx.Nooral;
const PG = S.PG, t = harness('playground');
const t0 = Date.now();
for (const data of ['circles', 'xor']) {
  const lab = PG.make({ data, layers: [4, 2], act: 'tanh', lr: 0.03, seed: 2, noise: 0 });
  const L0 = lab.losses();
  for (let e = 0; e < 60; e++) lab.epochStep();
  const L1 = lab.losses();
  t.ok(L1.train < L0.train * 0.6, data + ': the training loss falls (' + L0.train.toFixed(3) + ' → ' + L1.train.toFixed(3) + ')');
  t.ok(L1.testAcc > 0.85, data + ': test accuracy ' + L1.testAcc.toFixed(3));
  t.eq(lab.epoch, 60, data + ': epochs counted');
}
// the same seed gives the same network
const a = PG.make({ data: 'moons', seed: 5 }), b = PG.make({ data: 'moons', seed: 5 });
for (let i = 0; i < 40; i++) { a.step(); b.step(); }
t.eq(a.weights(), b.weights(), 'deterministic for a seed');
const c = PG.make({ data: 'moons', seed: 6 });
t.ok(JSON.stringify(c.weights()) !== JSON.stringify(a.weights()), 'another seed, another network');
// the plain-number forward pass agrees with the tensor engine
const w = a.weights(), T = S.T;
const p = a.forward(0.3, -1.2).pop()[0];
let h = T.tensor([[0.3, -1.2]]);
a.layers.forEach((l, i) => { h = l.forward(h); if (i + 1 < a.layers.length) h = T.tanh(h); });
t.ok(Math.abs(p - 1 / (1 + Math.exp(-h.item()))) < 1e-5, 'drawing forward = engine forward');
t.eq(w.W.map((m) => [m.length, m[0].length]), [[4, 2], [2, 4], [1, 2]], 'weight shapes out × in');
// regularisation pulls the weights in
const big = PG.make({ data: 'xor', seed: 3, layers: [6], reg: 'none' }), l2 = PG.make({ data: 'xor', seed: 3, layers: [6], reg: 'l2', regRate: 0.1 });
for (let e = 0; e < 30; e++) { big.epochStep(); l2.epochStep(); }
const norm = (lab) => lab.weights().W.flat(2).reduce((s, x) => s + x * x, 0);
t.ok(norm(l2) < norm(big), 'L2 keeps the weights smaller (' + norm(l2).toFixed(2) + ' < ' + norm(big).toFixed(2) + ')');
// features: x1² and x2² alone make circles easy for a single neuron
const sq = PG.make({ data: 'circles', features: ['x1sq', 'x2sq'], layers: [], seed: 1, noise: 0, lr: 0.1 });
for (let e = 0; e < 60; e++) sq.epochStep();
t.ok(sq.losses().testAcc > 0.9, 'squares solve circles with no hidden layer: ' + sq.losses().testAcc.toFixed(3));
t.ok(Date.now() - t0 < 60000, 'within a minute (' + (Date.now() - t0) + ' ms)');
t.done();
