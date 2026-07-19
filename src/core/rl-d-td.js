// Model-free control on grid worlds with a Q-table: Monte Carlo control, SARSA and Q-learning, all ε-greedy.
//   S.RL.tabular(env, {algo: 'q' | 'sarsa' | 'mc', episodes: 300, alpha: 0.5, gamma: 0.95, eps: 0.1, epsMin, epsDecay,
//                 seed, maxSteps: 200, snapEvery: 10, traceEpisode: 0, traceSteps: 40})
//      -> {Q, policy, returns[], lengths[], snapshots [{episode, Q, policy, ret}], trace (the updates of one episode),
//          visits[s]}
//   S.RL.greedyPath(env, policy, maxSteps) -> {path: [s], reached, ret}
// Q-learning backs up the best next action (off-policy); SARSA the action it will actually take (on-policy);
// Monte Carlo waits for the end of the episode and averages the returns that followed each first visit.
(function (S) {
  'use strict';
  const RL = S.RL, fmt = (x) => RL.fmt(x);
  const NAMES = { q: 'Q-learning', sarsa: 'SARSA', mc: 'Monte Carlo control' };

  RL.greedyPath = (env, policy, maxSteps = 200) => {
    let s = env.reset(), ret = 0, reached = false;
    const path = [s];
    for (let t = 0; t < maxSteps; t++) {
      const o = env.P(s, policy[s]).reduce((b, x) => (x.p > b.p ? x : b));
      ret += o.r; s = o.s2; path.push(s);
      if (o.done) { reached = !!env.goal[s]; break; }
    }
    return { path, reached, ret, steps: path.length - 1 };
  };

  RL.tabular = (env, o = {}) => {
    const T = S.T, algo = o.algo || 'q', episodes = o.episodes || 300, alpha = o.alpha == null ? 0.5 : o.alpha, gamma = o.gamma == null ? env.gamma || 0.95 : o.gamma;
    const rng = T.rng(o.seed || 1), maxSteps = o.maxSteps || 200, snapEvery = o.snapEvery || 10, traceEp = o.traceEpisode == null ? 0 : o.traceEpisode, traceMax = o.traceSteps || 40;
    const Q = Array.from({ length: env.n }, () => new Array(env.nA).fill(o.init || 0)), N = Array.from({ length: env.n }, () => new Array(env.nA).fill(0));
    const visits = new Array(env.n).fill(0), returns = [], lengths = [], snapshots = [], trace = [];
    let eps = o.eps == null ? 0.1 : o.eps;
    const act = (s) => (rng.next() < eps ? rng.int(env.nA) : RL.argmax(Q[s], rng));
    const cellOf = (s) => '(' + env.cell(s).join(', ') + ')';
    const snap = (ep, ret) => snapshots.push({ episode: ep, Q: Q.map((q) => q.slice()), policy: RL.greedy(Q), ret, eps });
    snap(0, 0);
    for (let ep = 1; ep <= episodes; ep++) {
      const tracing = ep - 1 === traceEp;
      let s = env.reset(), a = act(s), ret = 0, t = 0;
      const hist = [];
      for (; t < maxSteps; t++) {
        visits[s]++;
        const out = env.sample(s, a, rng), s2 = out.s2;
        ret += out.r;
        if (algo === 'mc') { hist.push([s, a, out.r]); s = s2; if (out.done) { t++; break; } a = act(s); continue; }
        const a2 = out.done ? 0 : act(s2);
        const nextV = out.done ? 0 : algo === 'q' ? Math.max(...Q[s2]) : Q[s2][a2];
        const target = out.r + gamma * nextV, old = Q[s][a];
        Q[s][a] = old + alpha * (target - old);
        if (tracing && trace.length < traceMax)
          trace.push(T.step('update', 'From ' + cellOf(s) + ' go ' + RL.ACTIONS[a] + ', reward ' + fmt(out.r) + ', land in ' + cellOf(s2) + (out.fell ? ' (a fall: back to the start)' : '') + '. Target = r + γ·' + (out.done ? '0 (the episode ends)' : algo === 'q' ? 'max Q(s′) = ' + fmt(out.r) + ' + ' + fmt(gamma) + '·' + fmt(nextV) : 'Q(s′, ' + RL.ACTIONS[a2] + ') = ' + fmt(out.r) + ' + ' + fmt(gamma) + '·' + fmt(nextV)) + ' = ' + fmt(target) + '; Q moves from ' + fmt(old) + ' to ' + fmt(Q[s][a]) + ' (α = ' + fmt(alpha) + ').', { s, a, r: out.r, s2, target, old, now: Q[s][a] }));
        s = s2; a = a2;
        if (out.done) { t++; break; }
      }
      if (algo === 'mc') {
        // first-visit returns, computed backwards
        let G = 0;
        const first = new Map(); hist.forEach(([hs, ha], i) => { const k = hs * 8 + ha; if (!first.has(k)) first.set(k, i); });
        for (let i = hist.length - 1; i >= 0; i--) {
          const [hs, ha, hr] = hist[i]; G = hr + gamma * G;
          if (first.get(hs * 8 + ha) !== i) continue;
          N[hs][ha]++; const old = Q[hs][ha]; Q[hs][ha] = old + (G - old) / N[hs][ha];
          if (tracing && trace.length < traceMax) trace.push(T.step('update', 'Return after the first visit to ' + cellOf(hs) + ' going ' + RL.ACTIONS[ha] + ' is G = ' + fmt(G) + '; the average of ' + N[hs][ha] + ' such returns moves Q from ' + fmt(old) + ' to ' + fmt(Q[hs][ha]) + '.', { s: hs, a: ha, G, old, now: Q[hs][ha] }));
        }
      }
      returns.push(ret); lengths.push(t);
      if (o.epsDecay) eps = Math.max(o.epsMin || 0.01, eps * o.epsDecay);
      if (ep % snapEvery === 0 || ep === episodes) snap(ep, ret);
    }
    const policy = RL.greedy(Q), path = RL.greedyPath(env, policy, maxSteps);
    trace.push(T.step('done', NAMES[algo] + ' after ' + episodes + ' episodes: the greedy path from the start ' + (path.reached ? 'reaches the goal in ' + path.steps + ' steps' : 'does not reach the goal') + ' with return ' + fmt(path.ret) + '.', { policy, path: path.path }));
    return { algo, name: NAMES[algo], Q, policy, returns, lengths, snapshots, trace, visits, path };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
