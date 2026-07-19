// Multi-armed bandits: ε-greedy, UCB1 and Thompson sampling, with regret.
//   S.RL.runBandit(bandit, {algo: 'epsilon' | 'greedy' | 'ucb' | 'thompson', eps: 0.1, c: 1, steps: 1000, seed,
//                  traceSteps: 20, snapEvery: 50})
//      -> {choices, rewards, regret (cumulative expected regret), optimal (0/1 per step), counts, estimates,
//          snapshots [{t, counts, estimates}], trace (the first traceSteps steps, each one sentence)}
//   S.RL.compareBandits(banditOpts, algos [{algo, eps, c, label}], {steps, runs, seed})
//      -> [{label, meanReward[t], regret[t], optimal[t]}] averaged over runs (a fresh bandit per run)
//   S.RL.sampleBeta(a, b, rng)  Beta draws (from two Gamma draws by squeeze-and-reject sampling)
(function (S) {
  'use strict';
  const RL = S.RL, fmt = (x) => RL.fmt(x);

  const gamma = (k, rng) => {
    if (k < 1) return gamma(k + 1, rng) * Math.pow(rng.next() || 1e-12, 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = rng.normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v; const u = rng.next();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  };
  RL.sampleBeta = (a, b, rng) => { const x = gamma(a, rng), y = gamma(b, rng); return x / (x + y); };
  const argmax = (v, rng) => { let m = -Infinity, ties = []; v.forEach((x, i) => { if (x > m + 1e-12) { m = x; ties = [i]; } else if (Math.abs(x - m) <= 1e-12) ties.push(i); }); return ties.length > 1 ? ties[rng.int(ties.length)] : ties[0]; };
  RL.argmax = argmax;

  RL.runBandit = (bandit, o = {}) => {
    const T = S.T, algo = o.algo || 'epsilon', steps = o.steps || 1000, rng = T.rng(o.seed || 1), k = bandit.k;
    const eps = algo === 'greedy' ? 0 : o.eps == null ? 0.1 : o.eps, c = o.c == null ? 1 : o.c;
    const counts = new Array(k).fill(0), sums = new Array(k).fill(0), est = new Array(k).fill(o.init || 0), wins = new Array(k).fill(0);
    const choices = [], rewards = [], regret = [], optimal = [], snapshots = [], trace = [];
    let reg = 0;
    for (let t = 1; t <= steps; t++) {
      let a, why, extra = {};
      if (algo === 'ucb') {
        const untried = counts.findIndex((n) => n === 0);
        if (untried >= 0) { a = untried; why = 'arm ' + a + ' has not been tried yet, so UCB1 tries it first'; }
        else {
          const ucb = est.map((q, i) => q + c * Math.sqrt((2 * Math.log(t)) / counts[i]));
          a = argmax(ucb, rng); extra.ucb = ucb;
          why = 'UCB1 picks arm ' + a + ': its estimate ' + fmt(est[a]) + ' plus bonus ' + fmt(ucb[a] - est[a]) + ' = ' + fmt(ucb[a]) + ' is the highest';
        }
      } else if (algo === 'thompson') {
        const draws = bandit.kind === 'bernoulli' ? wins.map((s, i) => RL.sampleBeta(1 + s, 1 + counts[i] - s, rng))
          : sums.map((s, i) => rng.normal(s / (counts[i] + 1), bandit.sd / Math.sqrt(counts[i] + 1)));
        a = argmax(draws, rng); extra.draws = draws;
        why = 'Thompson sampling draws a plausible mean for each arm from its posterior; arm ' + a + ' drew the highest, ' + fmt(draws[a]);
      } else if (rng.next() < eps) { a = rng.int(k); why = 'with probability ε = ' + fmt(eps) + ' it explores, and picks arm ' + a + ' at random'; }
      else { a = argmax(est, rng); why = 'it exploits arm ' + a + ', whose estimate ' + fmt(est[a]) + ' is the highest'; }
      const r = bandit.pull(a, rng);
      counts[a]++; sums[a] += r; if (r > 0.5) wins[a] += bandit.kind === 'bernoulli' ? 1 : 0;
      const old = est[a]; est[a] = old + (r - old) / counts[a];
      reg += bandit.bestMean - bandit.means[a];
      choices.push(a); rewards.push(r); regret.push(reg); optimal.push(a === bandit.best ? 1 : 0);
      if (t <= (o.traceSteps == null ? 20 : o.traceSteps)) trace.push(T.step('pull', 'Step ' + t + ': ' + why + '. Reward ' + fmt(r) + ', so its estimate moves from ' + fmt(old) + ' to ' + fmt(est[a]) + ' (' + counts[a] + ' pulls). Regret so far ' + fmt(reg) + '.', Object.assign({ t, a, r, estimates: est.slice(), counts: counts.slice() }, extra)));
      if (t % (o.snapEvery || 50) === 0 || t === steps) snapshots.push({ t, counts: counts.slice(), estimates: est.slice() });
    }
    return { algo, choices, rewards, regret, optimal, counts, estimates: est, snapshots, trace, total: rewards.reduce((s, x) => s + x, 0) };
  };

  RL.compareBandits = (bo = {}, algos, o = {}) => {
    algos = algos || [{ algo: 'greedy', label: 'greedy' }, { algo: 'epsilon', eps: 0.1, label: 'ε-greedy (0.1)' }, { algo: 'ucb', c: 1, label: 'UCB1' }, { algo: 'thompson', label: 'Thompson' }];
    const steps = o.steps || 1000, runs = o.runs || 50;
    return algos.map((al, ai) => {
      const mr = new Float64Array(steps), rg = new Float64Array(steps), op = new Float64Array(steps);
      for (let run = 0; run < runs; run++) {
        const b = RL.bandit(Object.assign({}, bo, { seed: (o.seed || 1) * 1000 + run + 1 }));
        const res = RL.runBandit(b, Object.assign({}, al, { steps, seed: (o.seed || 1) * 7777 + run * 31 + ai, traceSteps: 0, snapEvery: steps }));
        for (let t = 0; t < steps; t++) { mr[t] += res.rewards[t] / runs; rg[t] += res.regret[t] / runs; op[t] += res.optimal[t] / runs; }
      }
      return { label: al.label || al.algo, algo: al.algo, meanReward: Array.from(mr), regret: Array.from(rg), optimal: Array.from(op),
        say: (al.label || al.algo) + ': average regret after ' + steps + ' steps is ' + fmt(rg[steps - 1]) + ', and the best arm is chosen ' + Math.round(100 * op[steps - 1]) + '% of the time at the end.' };
    });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
