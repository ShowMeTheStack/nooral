// The training runs behind the deep reinforcement-learning topics of part 15: REINFORCE on cart-pole and a small
// DQN on a grid world. They take seconds to minutes here, so they are run once on fixed seeds and the topics keep
// what they show in src/topics/p15-0-data.js: the policy's weights at chosen updates (the topic replays the pole
// with them live), the training curves, the DQN's Q-table as it learns and a replayed minibatch with its targets.
//   node tests/ref/p15_runs.mjs
// tests/claims-p15.mjs re-runs the start of each job and checks that it matches what is kept.
import { load } from '../load.mjs';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url)), DATA = join(HERE, '..', '..', 'src', 'topics', 'p15-0-data.js');
export const JOBS = {
  reinforce: { o: { seed: 1, snapEvery: 1000 }, updates: 40, keep: [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40] },
  dqn: { o: { env: 'grid', world: 'pits', steps: 2000, seed: 1, snapEvery: 100 }, keep: [0, 100, 200, 300, 400, 600, 800, 1000, 1300, 1600, 2000], batchAt: [300, 1000, 2000] },
};
const sig = (v, n = 6) => (Number.isFinite(v) ? +(+v).toPrecision(n) : v);

export function runAll(S) {
  const RL = S.RL, G = S.Gen, out = {};
  {
    const J = JOBS.reinforce, tr = RL.reinforce(J.o), snaps = [{ update: 0, policy: G.copyParams(tr.policy).map((a) => Array.from(a, (v) => sig(v, 7))) }];
    for (let u = 1; u <= J.updates; u++) { tr.step(); if (J.keep.includes(u)) snaps.push({ update: u, episodes: tr.episodes(), policy: G.copyParams(tr.policy).map((a) => Array.from(a, (v) => sig(v, 7))) }); }
    out.reinforce = { o: J.o, updates: J.updates, snaps, meanLen: tr.history.map((h) => sig(h.meanLen)), loss: tr.history.map((h) => sig(h.loss, 5)), episodes: tr.history.map((h) => h.episode) };
  }
  {
    const J = JOBS.dqn, tr = RL.dqn(J.o), snaps = [], batches = [];
    const q = () => tr.qTable().map((r) => r.map((v) => sig(v, 5)));
    snaps.push({ step: 0, Q: q(), eps: 1, episodes: 0 });
    const hist = [];
    for (let k = 1; k <= J.o.steps; k++) {
      const r = tr.step();
      if (k % 10 === 0) hist.push({ step: k, loss: sig(r.loss, 4), eps: sig(r.eps, 4), episodes: r.episodes, syncs: r.syncs });
      if (J.keep.includes(k)) snaps.push({ step: k, Q: q(), eps: sig(r.eps), episodes: r.episodes, syncs: r.syncs, buffer: tr.buffer.length });
      if (J.batchAt.includes(k)) {
        // six stored transitions, as a minibatch would draw them, with the online and target networks' numbers
        const g = S.T.rng(k), rows = [];
        for (let i = 0; i < 6; i++) {
          const [x, a, rw, nx, d] = tr.buffer[g.int(tr.buffer.length)], qs = Array.from(RL.fastForward(tr.net, x)), qt = Array.from(RL.fastForward(tr.target, nx));
          rows.push({ s: x.indexOf(1), a, r: sig(rw), s2: nx.indexOf(1), done: d, q: qs.map((v) => sig(v, 7)), qt: qt.map((v) => sig(v, 7)) });
        }
        batches.push({ step: k, rows, syncs: r.syncs });
      }
    }
    out.dqn = { o: Object.assign({}, J.o), gamma: 0.95, targetEvery: 100, snaps, hist, batches, returns: tr.returns.map((x) => ({ t: x.t, ret: sig(x.ret, 5), len: x.len })) };
  }
  return out;
}

if (process.argv[1] && process.argv[1].endsWith('p15_runs.mjs')) {
  const S = load(['core']), t0 = Date.now(), all = runAll(S);
  const text = '// Made by tests/ref/p15_runs.mjs from training runs of the engines in src/core (S.RL) on fixed seeds: do not\n'
    + '// edit by hand.\n(function (S) {\n  \'use strict\';\n  S.P15D = ' + JSON.stringify(all) + ';\n})(globalThis.Nooral = globalThis.Nooral || {});\n';
  writeFileSync(DATA, text);
  console.log('wrote ' + DATA + ' (' + Math.round(text.length / 1024) + ' KB) in ' + Math.round((Date.now() - t0) / 1000) + ' s');
}
