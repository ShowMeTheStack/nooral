// Dynamic programming on a known model: value iteration and policy iteration, with every sweep recorded.
//   S.RL.qFromV(env, V, gamma) -> Q[s][a] = Σ p (r + γ V(s'))  (V(s') is 0 after a terminal transition)
//   S.RL.backup(env, V, s, gamma) -> {q, best, say}: the Bellman backup of one state, spelled out
//   S.RL.valueIteration(env, {gamma: 0.95, theta: 1e-6, maxSweeps: 500})
//      -> {V, Q, policy, sweeps: [{k, V, delta, policy}], trace}
//   S.RL.policyIteration(env, {gamma, theta, maxIter: 50, init: 0})
//      -> {V, Q, policy, iterations: [{k, evalSweeps, V, policy, changed}], trace}
//   S.RL.optimalActions(Q, s, tol) -> the actions within tol of the best
(function (S) {
  'use strict';
  const RL = S.RL, fmt = (x) => RL.fmt(x);

  RL.qFromV = (env, V, gamma) => Array.from({ length: env.n }, (_, s) => Array.from({ length: env.nA }, (_, a) => env.P(s, a).reduce((q, o) => q + o.p * (o.r + (o.done ? 0 : gamma * V[o.s2])), 0)));
  RL.greedy = (Q) => Q.map((q) => { let b = 0; q.forEach((v, a) => { if (v > q[b] + 1e-12) b = a; }); return b; });
  RL.optimalActions = (Q, s, tol = 1e-6) => { const m = Math.max(...Q[s]); return Q[s].map((v, a) => (v >= m - tol ? a : -1)).filter((a) => a >= 0); };

  RL.backup = (env, V, s, gamma = 0.95) => {
    const q = Array.from({ length: env.nA }, (_, a) => env.P(s, a).reduce((t, o) => t + o.p * (o.r + (o.done ? 0 : gamma * V[o.s2])), 0));
    const best = RL.argmax(q, { int: () => 0 }), [i, j] = env.cell(s);
    const parts = q.map((v, a) => RL.ACTIONS[a] + ' ' + fmt(v));
    return { q, best, say: 'Cell (' + i + ', ' + j + '): r + γ·V(next) for each action is ' + parts.join(', ') + '; the best is ' + RL.ACTIONS[best] + ', so V becomes ' + fmt(q[best]) + '.' };
  };

  RL.valueIteration = (env, o = {}) => {
    const T = S.T, gamma = o.gamma == null ? env.gamma || 0.95 : o.gamma, theta = o.theta || 1e-6, maxSweeps = o.maxSweeps || 500;
    let V = new Array(env.n).fill(0);
    const sweeps = [], trace = [T.step('init', 'Start with V = 0 in every cell; each sweep replaces V(s) by the best one-step look-ahead max_a Σ p (r + γ V(s′)), with γ = ' + fmt(gamma) + '.', { V: V.slice() })];
    for (let k = 1; k <= maxSweeps; k++) {
      const nV = V.slice();
      let delta = 0, where = env.start;
      for (const s of env.states) {
        const b = RL.backup(env, V, s, gamma), v = b.q[b.best], d = Math.abs(v - V[s]);
        nV[s] = v; if (d > delta) { delta = d; where = s; }
      }
      const show = RL.backup(env, V, where, gamma);
      V = nV;
      const policy = RL.greedy(RL.qFromV(env, V, gamma));
      sweeps.push({ k, V: V.slice(), delta, policy });
      trace.push(T.step('sweep', 'Sweep ' + k + ': the largest change is ' + fmt(delta) + ' at cell (' + env.cell(where).join(', ') + '). ' + show.say, { k, delta, V: V.slice(), policy, cell: where }));
      if (delta < theta) break;
    }
    const Q = RL.qFromV(env, V, gamma), policy = RL.greedy(Q);
    trace.push(T.step('done', 'Converged after ' + sweeps.length + ' sweeps (largest change below ' + fmt(theta) + '); V at the start is ' + fmt(V[env.start]) + ' and the greedy arrows form the optimal policy.', { V: V.slice(), policy }));
    return { V, Q, policy, sweeps, trace, gamma };
  };

  /** iterative policy evaluation: V^π to tolerance theta, in place (Gauss–Seidel) */
  RL.evaluatePolicy = (env, policy, gamma, theta = 1e-6, V0, maxSweeps = 10000) => {
    const V = V0 ? V0.slice() : new Array(env.n).fill(0);
    let sweeps = 0;
    for (; sweeps < maxSweeps; sweeps++) {
      let delta = 0;
      for (const s of env.states) {
        const v = env.P(s, policy[s]).reduce((t, o) => t + o.p * (o.r + (o.done ? 0 : gamma * V[o.s2])), 0);
        delta = Math.max(delta, Math.abs(v - V[s])); V[s] = v;
      }
      if (delta < theta) { sweeps++; break; }
    }
    return { V, sweeps };
  };

  RL.policyIteration = (env, o = {}) => {
    const T = S.T, gamma = o.gamma == null ? env.gamma || 0.95 : o.gamma, theta = o.theta || 1e-8, maxIter = o.maxIter || 50;
    let policy = new Array(env.n).fill(o.init || 0), V = new Array(env.n).fill(0);
    const iterations = [], trace = [T.step('init', 'Start with the policy "always ' + RL.ACTIONS[o.init || 0] + '" everywhere.', { policy: policy.slice() })];
    for (let k = 1; k <= maxIter; k++) {
      const ev = RL.evaluatePolicy(env, policy, gamma, theta, V);
      V = ev.V;
      const Q = RL.qFromV(env, V, gamma);
      const next = policy.slice();
      let changed = 0;
      for (const s of env.states) {
        // keep the current action when it is still among the best, so the loop cannot cycle between ties
        const best = RL.optimalActions(Q, s, 1e-9);
        if (!best.includes(policy[s])) { next[s] = best[0]; changed++; }
      }
      iterations.push({ k, evalSweeps: ev.sweeps, V: V.slice(), policy: next.slice(), changed });
      trace.push(T.step('iteration', 'Iteration ' + k + ': evaluating the policy took ' + ev.sweeps + ' sweeps (V at the start = ' + fmt(V[env.start]) + '); acting greedily on those values changes the action in ' + changed + ' cell' + (changed === 1 ? '' : 's') + '.', { k, V: V.slice(), policy: next.slice(), changed }));
      policy = next;
      if (!changed) break;
    }
    const Q = RL.qFromV(env, V, gamma);
    trace.push(T.step('done', 'Stable after ' + iterations.length + ' iterations: no cell wants a different action, so the policy is optimal.', { V: V.slice(), policy: policy.slice() }));
    return { V, Q, policy, iterations, trace, gamma };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
