// Reinforcement learning: bandits and their regret, value iteration against policy iteration, Q-learning reaching
// the optimal policy, SARSA and Monte Carlo control, cart-pole physics, REINFORCE balancing the pole, a DQN
// solving a small grid; gradient checks of the policy-gradient and DQN losses; determinism.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T, RL = S.RL;
const t = harness('rl');
const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
const timed = (name, budgetMs, fn) => {
  const t0 = cpu(), r = fn(), ms = cpu() - t0;
  console.log(`  ${name.padEnd(34)} ${r}  in ${(ms / 1000).toFixed(2)} s`);
  t.ok(ms < budgetMs, name + ' took ' + ms.toFixed(0) + ' ms (budget ' + budgetMs + ')');
};

// ---- bandits ----
{
  const g = T.rng(2), xs = Array.from({ length: 4000 }, () => RL.sampleBeta(2, 5, g)), m = xs.reduce((a, b) => a + b) / xs.length;
  t.ok(Math.abs(m - 2 / 7) < 0.01 && xs.every((x) => x > 0 && x < 1), 'Beta(2, 5) draws have mean ' + m);
  const b = RL.bandit({ seed: 3 });
  t.ok(b.means.length === 10 && b.means[b.best] === Math.max(...b.means), 'bandit best arm');
  const r = RL.runBandit(b, { algo: 'ucb', steps: 300 });
  t.ok(r.regret.every((v, i) => i === 0 || v >= r.regret[i - 1] - 1e-12), 'cumulative regret never falls');
  t.ok(r.trace.length === 20 && r.trace.slice(0, 10).every((s) => /not been tried/.test(s.say)), 'UCB1 tries every arm once first');
  t.ok(r.counts.reduce((a, c) => a + c) === 300, 'pull counts add up');
}
timed('bandit regret', 8000, () => {
  const out = [];
  for (const kind of ['bernoulli', 'gaussian']) {
    const rows = RL.compareBandits({ k: 10, kind }, null, { steps: 1000, runs: 30 }), reg = Object.fromEntries(rows.map((r) => [r.algo, r.regret[999]]));
    t.ok(reg.thompson < reg.greedy && reg.ucb < reg.greedy && reg.epsilon < reg.greedy, kind + ': every exploring method beats greedy ' + JSON.stringify(reg));
    t.ok(reg.thompson < reg.epsilon, kind + ': Thompson sampling has less regret than ε-greedy');
    out.push(kind + ' ' + Object.entries(reg).map(([k, v]) => k + ' ' + v.toFixed(0)).join(', '));
  }
  return out.join('; ');
});

// ---- dynamic programming ----
const worlds = { pits: RL.grid({ world: 'pits' }), rooms: RL.grid({ world: 'rooms' }), cliff: RL.cliff(), slippery: RL.grid({ world: 'pits', slip: 0.2 }), maze: RL.grid({ world: S.Data.gridWorld({ name: 'maze', seed: 4 }) }) };
for (const [name, env] of Object.entries(worlds)) {
  const vi = RL.valueIteration(env), pi = RL.policyIteration(env);
  const dv = Math.max(...env.states.map((s) => Math.abs(vi.V[s] - pi.V[s])));
  t.ok(dv < 1e-4, name + ': value iteration and policy iteration agree on V (largest gap ' + dv + ')');
  t.ok(env.states.every((s) => RL.optimalActions(vi.Q, s, 1e-4).includes(pi.policy[s])), name + ': policy iteration\'s actions are optimal for value iteration\'s Q');
  // V is a fixed point of the Bellman optimality backup
  const res = Math.max(...env.states.map((s) => Math.abs(Math.max(...vi.Q[s]) - vi.V[s])));
  t.ok(res < 1e-4, name + ': Bellman residual ' + res);
  t.ok(vi.sweeps.length > 1 && vi.trace.length === vi.sweeps.length + 2 && pi.iterations[pi.iterations.length - 1].changed === 0, name + ': traced sweeps and iterations');
  if (!env.slip) { const p = RL.greedyPath(env, vi.policy); t.ok(p.reached && p.steps === S.Data.gridDistance(env.world), name + ': the optimal path is a shortest path (' + p.steps + ')'); }
}

// ---- model-free control ----
timed('Q-learning reaches the optimal policy', 3000, () => {
  // Q-learning is off-policy: it learns the greedy policy's values while exploring a lot (ε = 0.5)
  const env = worlds.pits, vi = RL.valueIteration(env), out = [];
  for (const seed of [1, 2, 3]) {
    const q = RL.tabular(env, { algo: 'q', episodes: 500, eps: 0.5, alpha: 0.8, seed });
    const wrong = env.states.filter((s) => !RL.optimalActions(vi.Q, s, 1e-3).includes(q.policy[s]));
    t.ok(!wrong.length, 'Q-learning (seed ' + seed + '): greedy actions optimal in every state (wrong in ' + wrong.map((s) => env.cell(s).join(',')).join(' ') + ')');
    const dq = Math.max(...env.states.map((s) => Math.abs(Math.max(...q.Q[s]) - vi.V[s])));
    t.ok(dq < 0.05, 'Q-learning (seed ' + seed + '): max Q close to V* (largest gap ' + dq + ')');
    if (seed === 1) {
      t.ok(q.snapshots.length === 51 && q.snapshots[0].Q.every((r) => r.every((v) => v === 0)), 'Q-table snapshots, starting from zeros');
      t.ok(q.trace.length > 5 && /Target/.test(q.trace[0].say), 'Q-learning trace of one episode');
    }
    out.push(dq.toFixed(4));
  }
  return 'all ' + env.states.length + ' states optimal for 3 seeds; gaps ' + out.join(', ');
});
{
  const env = worlds.cliff, ql = RL.tabular(env, { algo: 'q', episodes: 500, seed: 1, maxSteps: 1000 }), sa = RL.tabular(env, { algo: 'sarsa', episodes: 500, seed: 1, maxSteps: 1000 });
  const edge = (p) => p.path.filter((s) => env.cell(s)[0] === 2).length;
  t.ok(ql.path.reached && ql.path.steps === 13 && edge(ql.path) === 12, 'cliff: Q-learning takes the shortest path, right along the edge (' + ql.path.steps + ' steps)');
  t.ok(sa.path.reached && sa.path.steps > 13 && edge(sa.path) < 3, 'cliff: SARSA takes a safer path away from the edge (' + sa.path.steps + ' steps)');
  const avg = (r) => r.returns.slice(-100).reduce((a, b) => a + b) / 100;
  t.ok(avg(sa) > avg(ql), 'cliff: while still exploring, SARSA earns more per episode (' + avg(sa) + ' against ' + avg(ql) + ')');
  const mc = RL.tabular(worlds.pits, { algo: 'mc', episodes: 500, seed: 1 });
  t.ok(mc.path.reached, 'Monte Carlo control reaches the goal in ' + mc.path.steps + ' steps');
  t.ok(mc.trace.some((s) => /first visit/.test(s.say)), 'Monte Carlo trace');
}

// ---- cart-pole physics ----
{
  const env = RL.cartPole({ seed: 1 });
  env.state = [0, 0, 0.1, 0];
  const d = env.dynamics(env.state, 10), c = Math.cos(0.1), s = Math.sin(0.1), temp = 10 / 1.1, thAcc = (9.8 * s - c * temp) / (0.5 * (4 / 3 - (0.1 * c * c) / 1.1)), xAcc = temp - (0.05 * thAcc * c) / 1.1;
  t.ok(Math.abs(d.thAcc - thAcc) < 1e-12 && Math.abs(d.xAcc - xAcc) < 1e-12, 'cart-pole accelerations by the standard equations');
  const o = env.step(1);
  t.ok(Math.abs(o.s[2] - 0.1) < 1e-12 && Math.abs(o.s[3] - 0.02 * thAcc) < 1e-12 && Math.abs(o.s[1] - 0.02 * xAcc) < 1e-12, 'one Euler step');
  env.reset(); let n = 0; for (; n < 500; n++) if (env.step(1).done) break;
  t.ok(n > 5 && n < 60, 'pushing one way all the time drops the pole after ' + n + ' steps');
  t.ok(/rad\/s²/.test(env.explain(0).say), 'physics explanation');
}

// ---- gradient checks ----
T.float64(() => {
  const check = (name, params, lossOf) => {
    for (const p of params) p.grad = null;
    lossOf().backward();
    let bad = null;
    params.forEach((p, pi) => { for (let i = 0; i < Math.min(p.size, 10); i++) { const o = p.data[i]; p.data[i] = o + 1e-6; const a = T.noGrad(lossOf).item(); p.data[i] = o - 1e-6; const b = T.noGrad(lossOf).item(); p.data[i] = o; const num = (a - b) / 2e-6, ana = p.grad ? p.grad.data[i] : 0; if (Math.abs(num - ana) / Math.max(1, Math.abs(num)) > 1e-6 && !bad) bad = 'param ' + pi + ' element ' + i + ': ' + ana + ' vs ' + num; } });
    t.ok(!bad, name + ': ' + bad);
  };
  const r = RL.reinforce({ seed: 2 }), X = T.normal([6, 4], 0, 1, 3), A = [0, 1, 1, 0, 1, 0], adv = T.tensor([0.5, -1, 2, 0.1, -0.3, 1]);
  check('REINFORCE loss −mean(log π(a|s) · A)', r.policy.parameters(), () => T.neg(T.mean(T.mul(T.reshape(T.gather(T.logSoftmax(r.policy.forward(X), -1), 1, T.tensor(A.map((a) => [a]))), [6]), adv))));
  const d = RL.dqn({ seed: 2 }), Xs = T.tensor([0, 3, 7].map((s) => d.env.features(s)));
  check('DQN Huber loss on Q(s, a)', d.net.parameters(), () => T.huberLoss(T.reshape(T.gather(d.net.forward(Xs), 1, T.tensor([[0], [2], [1]])), [3]), T.tensor([0.5, -2, 0.1])));
});
{ // the fast forward used for acting agrees with the recorded one
  const r = RL.reinforce({ seed: 2 }), x = [0.1, -0.2, 0.05, 0.3], a = RL.fastForward(r.policy, x), b = T.noGrad(() => r.policy.forward(T.tensor([x]))).values();
  t.ok(Math.abs(a[0] - b[0]) < 1e-5 && Math.abs(a[1] - b[1]) < 1e-5, 'fastForward matches forward');
}

// ---- deep RL ----
timed('REINFORCE balances cart-pole', 30000, () => {
  const r = RL.reinforce({ seed: 1 });
  r.run(50);
  const greedy = r.evaluate(), sampled = r.evaluate({ greedy: false });
  t.ok(greedy.mean >= 150, 'REINFORCE: greedy policy balances for ' + greedy.mean + ' steps on average');
  t.ok(sampled.mean >= 150, 'REINFORCE: sampled policy balances for ' + sampled.mean + ' steps on average');
  t.ok(r.snapshots.length >= 5 && r.snapshots[1].frames.length > 1 && r.snapshots[1].frames[0].length === 4, 'REINFORCE snapshots carry an episode to draw');
  t.ok(r.explain().length === 5, 'REINFORCE explain');
  return 'greedy ' + greedy.mean.toFixed(0) + ', sampled ' + sampled.mean.toFixed(0) + ' steps after ' + r.episodes() + ' episodes';
});
timed('DQN solves a small grid', 15000, () => {
  const d = RL.dqn({ seed: 1 });
  d.run(3000);
  const p = d.greedyPath();
  t.ok(p.reached && p.steps === S.Data.gridDistance(d.env.world), 'DQN: greedy path reaches the goal in ' + p.steps + ' steps');
  t.ok(d.history[d.history.length - 1].syncs === 30, 'target network copied every 100 steps');
  t.ok(d.snapshots.length === 13 && d.snapshots[1].Q.length === d.env.n, 'DQN snapshots of Q per state');
  t.ok(d.explain().length === 3, 'DQN explain');
  return 'path ' + p.steps + ' steps, ' + d.returns.length + ' episodes';
});

// ---- determinism ----
t.eq(RL.tabular(worlds.rooms, { algo: 'q', episodes: 50, seed: 5 }).Q, RL.tabular(worlds.rooms, { algo: 'q', episodes: 50, seed: 5 }).Q, 'Q-learning is deterministic');
t.eq(RL.runBandit(RL.bandit({ seed: 2 }), { algo: 'thompson', steps: 200, seed: 3 }).choices, RL.runBandit(RL.bandit({ seed: 2 }), { algo: 'thompson', steps: 200, seed: 3 }).choices, 'Thompson sampling is deterministic');
{ const h = () => { const r = RL.reinforce({ seed: 4, snapEvery: 1000 }); r.run(3); return r.history.map((x) => x.loss); }; t.eq(h(), h(), 'REINFORCE is deterministic'); }
{ const h = () => { const d = RL.dqn({ seed: 4 }); d.run(300); return d.history.map((x) => x.loss); }; t.eq(h(), h(), 'DQN is deterministic'); }
t.ok(T.jobs.has('reinforce') && T.jobs.has('dqn'), 'RL jobs are registered');
t.done();
