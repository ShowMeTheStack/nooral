// Deep reinforcement learning: REINFORCE with a learned baseline on cart-pole, and a small DQN with experience
// replay and a target network (on a grid world with one-hot states, or on cart-pole).
//   S.RL.reinforce({episodes: 600, perUpdate: 4, gamma: 0.99, hidden: 32, lr: 0.01, lrV: 0.01, seed, snapEvery: 10})
//      -> trainer: step() runs one update (perUpdate episodes), run(n), history [{update, episode, meanLen, loss}],
//         snapshots [{update, meanLen, frames (one greedy episode's states), probs}], evaluate({episodes, greedy}),
//         explain() -> one update spelled out (returns, baseline, advantages, the loss)
//   S.RL.dqn({env: 'grid' | 'cartpole', world: 'pits', steps: 3000, hidden: 64, lr: 1e-3, gamma: 0.95, batch: 32,
//             buffer: 5000, targetEvery: 100, epsStart: 1, epsEnd: 0.05, epsSteps, seed})
//      -> trainer: step() is one environment step plus one learning step; snapshots {Q (grid: per state), eps}
//   S.RL.fastForward(seq, x) evaluates a Dense / activation stack on one input without recording a graph
(function (S) {
  'use strict';
  const RL = S.RL, fmt = (x) => RL.fmt(x);

  RL.fastForward = (seq, x) => {
    let a = x;
    for (const l of seq.layers) {
      if (l.W) {
        const W = l.W.values(), b = l.b ? l.b.values() : null, nI = l.inF, nO = l.outF, out = new Float64Array(nO);
        for (let j = 0; j < nO; j++) out[j] = b ? b[j] : 0;
        for (let i = 0; i < nI; i++) { const v = a[i]; if (v) for (let j = 0; j < nO; j++) out[j] += v * W[i * nO + j]; }
        a = out;
      } else if (l.kind === 'tanh') a = a.map(Math.tanh);
      else if (l.kind === 'relu') a = a.map((v) => (v > 0 ? v : 0));
      else if (l.kind === 'leakyRelu') a = a.map((v) => (v > 0 ? v : (l.arg || 0.01) * v));
      else if (l.kind === 'sigmoid') a = a.map((v) => 1 / (1 + Math.exp(-v)));
    }
    return a;
  };
  const SCALE = [2.4, 3, 0.21, 3];
  const norm = (s) => s.map((v, i) => v / SCALE[i]);
  const softmax2 = (z) => { const m = Math.max(z[0], z[1]), a = Math.exp(z[0] - m), b = Math.exp(z[1] - m); return [a / (a + b), b / (a + b)]; };

  RL.reinforce = (o0 = {}) => {
    const T = S.T, nn = T.nn, o = Object.assign({ perUpdate: 4, gamma: 0.99, hidden: 32, lr: 0.01, lrV: 0.01, maxSteps: 500, snapEvery: 10 }, o0);
    const g = T.rng(o.seed || 1), rng = T.rng((o.seed || 1) * 13 + 5), env = RL.cartPole({ maxSteps: o.maxSteps, seed: (o.seed || 1) * 7 + 1 });
    const pi = new nn.Sequential(new nn.Dense(4, o.hidden, { rng: g }), new nn.Activation('tanh'), new nn.Dense(o.hidden, 2, { rng: g }));
    const vf = new nn.Sequential(new nn.Dense(4, o.hidden, { rng: g }), new nn.Activation('tanh'), new nn.Dense(o.hidden, 1, { rng: g }));
    const optP = new T.optim.Adam(pi.parameters(), { lr: o.lr }), optV = new T.optim.Adam(vf.parameters(), { lr: o.lrV });
    const probs = (s) => softmax2(RL.fastForward(pi, norm(s)));
    const episode = (greedy, r = rng) => {
      const states = [], actions = [], rewards = [];
      let s = env.reset(r);
      for (;;) {
        const p = probs(s), a = greedy ? (p[1] > p[0] ? 1 : 0) : r.next() < p[1] ? 1 : 0;
        states.push(s); actions.push(a);
        const out = env.step(a); rewards.push(out.r); s = out.s;
        if (out.done) break;
      }
      return { states, actions, rewards, len: rewards.length };
    };
    let episodes = 0, last = null;
    const update = () => {
      const eps = Array.from({ length: o.perUpdate }, () => episode(false)), S0 = [], A = [], R = [];
      for (const e of eps) { let G = 0; const ret = new Array(e.len); for (let t = e.len - 1; t >= 0; t--) { G = e.rewards[t] + o.gamma * G; ret[t] = G; } e.states.forEach((s, t) => { S0.push(norm(s)); A.push(e.actions[t]); R.push(ret[t]); }); }
      episodes += eps.length;
      const X = T.tensor(S0), N = S0.length, Rs = R.map((v) => v / 100);
      const base = T.noGrad(() => Array.from(vf.forward(X).values()));
      let adv = Rs.map((v, i) => v - base[i]);
      const m = adv.reduce((s, v) => s + v, 0) / N, sd = Math.sqrt(adv.reduce((s, v) => s + (v - m) ** 2, 0) / N) + 1e-8;
      adv = adv.map((v) => (v - m) / sd);
      optP.zeroGrad();
      const lp = T.logSoftmax(pi.forward(X), -1), picked = T.reshape(T.gather(lp, 1, T.tensor(A.map((a) => [a]))), [N]);
      const loss = T.neg(T.mean(T.mul(picked, T.tensor(adv))));
      loss.backward(); optP.step();
      optV.zeroGrad();
      const lv = T.mseLoss(T.reshape(vf.forward(X), [N]), T.tensor(Rs));
      lv.backward(); optV.step();
      const meanLen = eps.reduce((s, e) => s + e.len, 0) / eps.length;
      last = { eps, R, base, adv, loss: loss.item(), lossV: lv.item() };
      return { loss: loss.item(), lossV: lv.item(), meanLen, episode: episodes };
    };
    const evaluate = (eo = {}) => { const r = T.rng(eo.seed || 4242), n = eo.episodes || 20, lens = []; for (let i = 0; i < n; i++) lens.push(episode(eo.greedy !== false, r).len); return { mean: lens.reduce((s, v) => s + v, 0) / n, lens }; };
    const tr = S.Gen.trainer({ step: update, snapEvery: o.snapEvery, snapshot: () => { const e = episode(true, T.rng(99)); return { meanLen: tr && tr.history.length ? tr.history[tr.history.length - 1].meanLen : 0, frames: e.states, greedyLen: e.len, probs: e.states.slice(0, 50).map(probs) }; } });
    return Object.assign(tr, {
      model: pi, policy: pi, value: vf, env, evaluate, probs, episodes: () => episodes,
      explain() {
        if (!last) update();
        const e = last.eps[0], t0 = 0, k = Math.min(e.len - 1, 3);
        return [
          T.step('episode', 'The first of ' + last.eps.length + ' episodes lasted ' + e.len + ' steps, earning +1 per step.', { len: e.len }),
          T.step('return', 'The discounted return from step 0 is G = Σ γ^t r_t = ' + fmt(last.R[t0]) + ' (γ = ' + fmt(o.gamma) + '); from step ' + k + ' it is ' + fmt(last.R[k]) + '.', { G: last.R.slice(0, e.len) }),
          T.step('baseline', 'The value network predicts ' + fmt(last.base[t0] * 100) + ' for the first state; subtracting it leaves only how much better than expected the episode went.', { baseline: last.base[t0] * 100 }),
          T.step('advantage', 'After centring and scaling, the advantage of the first action is ' + fmt(last.adv[t0]) + ': ' + (last.adv[t0] > 0 ? 'its probability is pushed up' : 'its probability is pushed down') + '.', { adv: last.adv[t0] }),
          T.step('loss', 'The policy loss −mean(log π(a|s) · A) is ' + fmt(last.loss) + ' and the baseline\'s squared error is ' + fmt(last.lossV) + '.', { loss: last.loss }),
        ];
      },
    });
  };

  RL.dqn = (o0 = {}) => {
    const T = S.T, nn = T.nn, isGrid = (o0.env || 'grid') === 'grid';
    const o = Object.assign({ steps: 3000, hidden: 64, lr: 1e-3, gamma: isGrid ? 0.95 : 0.99, batch: 32, buffer: 5000, targetEvery: 100, epsStart: 1, epsEnd: 0.05, learnStart: 100, maxSteps: isGrid ? 100 : 500, snapEvery: 250 }, o0);
    o.epsSteps = o.epsSteps || Math.round(o.steps * 0.6);
    const g = T.rng(o.seed || 1), rng = T.rng((o.seed || 1) * 17 + 3);
    const env = isGrid ? RL.grid({ world: o.world || 'pits' }) : RL.cartPole({ maxSteps: o.maxSteps, seed: o.seed || 1 });
    const dim = isGrid ? env.n : 4, nA = env.nA, feat = (s) => (isGrid ? env.features(s) : norm(s));
    const make = () => new nn.Sequential(new nn.Dense(dim, o.hidden, { rng: g, init: 'he' }), new nn.Activation('relu'), new nn.Dense(o.hidden, nA, { rng: g }));
    const net = make(), target = make(), opt = new T.optim.Adam(net.parameters(), { lr: o.lr });
    S.Gen.setParams(target, S.Gen.copyParams(net));
    const buf = [];
    let bi = 0, s = isGrid ? env.reset() : env.reset(rng), epRet = 0, epLen = 0, t = 0, syncs = 0;
    const returns = [];
    const epsAt = () => Math.max(o.epsEnd, o.epsStart - ((o.epsStart - o.epsEnd) * t) / o.epsSteps);
    const qOf = (x) => RL.fastForward(net, feat(x));
    const step = () => {
      const eps = epsAt(), a = rng.next() < eps ? rng.int(nA) : RL.argmax(Array.from(qOf(s)), rng);
      let s2, r, done, cut = false;
      if (isGrid) { const out = env.sample(s, a, rng); s2 = out.s2; r = out.r; done = out.done; } else { const out = env.step(a); s2 = out.s; r = out.fell ? -1 : 0.1; done = out.fell; cut = !out.fell && out.done; }
      epRet += isGrid ? r : 1; epLen++; t++;
      const tr0 = [feat(s), a, r, feat(s2), done ? 1 : 0];
      if (buf.length < o.buffer) buf.push(tr0); else buf[bi] = tr0;
      bi = (bi + 1) % o.buffer;
      s = s2;
      if (done || cut || epLen >= o.maxSteps) { returns.push({ t, ret: epRet, len: epLen }); s = isGrid ? env.reset() : env.reset(rng); epRet = 0; epLen = 0; }
      let loss = 0;
      if (buf.length >= o.learnStart) {
        const idx = Array.from({ length: o.batch }, () => rng.int(buf.length)), B = idx.map((i) => buf[i]);
        // the bootstrap target uses the frozen target network: y = r + γ max_a' Q_target(s', a')
        const y = B.map(([, , rr, ns, d]) => rr + (d ? 0 : o.gamma * Math.max(...RL.fastForward(target, ns))));
        opt.zeroGrad();
        const q = T.reshape(T.gather(net.forward(T.tensor(B.map((b) => b[0]))), 1, T.tensor(B.map((b) => [b[1]]))), [o.batch]);
        const l = T.huberLoss(q, T.tensor(y));
        l.backward(); opt.step(); loss = l.item();
      }
      if (t % o.targetEvery === 0) { S.Gen.setParams(target, S.Gen.copyParams(net)); syncs++; }
      const lastR = returns.length ? returns[returns.length - 1] : null;
      return { loss, eps, episodes: returns.length, lastReturn: lastR ? lastR.ret : 0, syncs };
    };
    const qTable = () => (isGrid ? Array.from({ length: env.n }, (_, st) => Array.from(qOf(st))) : null);
    const tr = S.Gen.trainer({ step, snapshot: () => ({ Q: qTable(), eps: epsAt(), episodes: returns.length }), snapEvery: o.snapEvery });
    return Object.assign(tr, {
      model: net, net, target, env, returns, qTable, buffer: buf,
      policy: () => (isGrid ? RL.greedy(qTable()) : null),
      greedyPath: () => (isGrid ? RL.greedyPath(env, RL.greedy(qTable()), o.maxSteps) : null),
      evaluate(n = 10) { if (isGrid) return RL.greedyPath(env, RL.greedy(qTable()), o.maxSteps); const r2 = T.rng(4242), lens = []; for (let i = 0; i < n; i++) { let st = env.reset(r2), len = 0; for (;;) { const out = env.step(RL.argmax(Array.from(qOf(st)), r2)); st = out.s; len++; if (out.done) break; } lens.push(len); } s = env.reset(rng); return { mean: lens.reduce((a, b) => a + b, 0) / n, lens }; },
      explain() {
        if (buf.length < 1) step();
        const [x, a, r, nx, d] = buf[0], q = RL.fastForward(net, x), qt = RL.fastForward(target, nx), y = r + (d ? 0 : o.gamma * Math.max(...qt));
        return [
          T.step('replay', 'A stored transition is drawn from the replay buffer of ' + buf.length + ': action ' + (isGrid ? RL.ACTIONS[a] : a ? 'right' : 'left') + ', reward ' + fmt(r) + (d ? ', and the episode ended' : '') + '.', { a, r }),
          T.step('target', 'The target network (copied every ' + o.targetEvery + ' steps) gives y = r + γ max Q_target(s′) = ' + fmt(r) + ' + ' + fmt(o.gamma) + ' × ' + fmt(d ? 0 : Math.max(...qt)) + ' = ' + fmt(y) + '.', { y }),
          T.step('td', 'The online network says Q(s, a) = ' + fmt(q[a]) + '; the Huber loss pulls it towards ' + fmt(y) + ' (a TD error of ' + fmt(y - q[a]) + ').', { q: q[a], err: y - q[a] }),
        ];
      },
    });
  };

  S.lateJob('reinforce', S.Gen.asJob((p) => RL.reinforce(p)));
  S.lateJob('dqn', S.Gen.asJob((p) => RL.dqn(p)));
})(globalThis.Nooral = globalThis.Nooral || {});
