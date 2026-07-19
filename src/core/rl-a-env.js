// Reinforcement-learning environments (S.RL): multi-armed bandits, grid worlds (from S.Data.gridWorld) with a full
// transition model for dynamic programming, cliff walking, and cart-pole physics.
//   S.RL.bandit({kind: 'bernoulli' | 'gaussian', k: 10, seed, means}) -> {k, means, best, pull(a, rng)}
//   S.RL.grid({world: 'pits' | map object, step: −0.04, goal: 1, pit: −1, slip: 0, pitMode: 'end' | 'restart'})
//      -> {n, w, h, nA: 4, start, P(s, a) -> [{p, s2, r, done}], reset(), sample(s, a, rng) -> {s2, r, done},
//          terminal[s], wall[s], states (the non-wall, non-terminal cells), cell(s) -> [row, col], arrows}
//   S.RL.cliff() -> cliff walking on a 4×12 map: −1 per step and −100 (back to the start) for a fall, γ = 0.99
//   S.RL.parseMap(rows, name) -> a map object from rows of text
// env.gamma is the discount the algorithms use unless told otherwise (0.95 by default).
//   S.RL.cartPole({maxSteps: 500, seed}) -> {reset(), step(action 0 | 1) -> {s, r, done}, state, t, physics}
// Actions in a grid: 0 north, 1 east, 2 south, 3 west.
(function (S) {
  'use strict';
  const RL = (S.RL = S.RL || {});
  RL.fmt = (x) => S.T.fmt(x);
  RL.ACTIONS = ['north', 'east', 'south', 'west'];
  RL.ARROWS = ['↑', '→', '↓', '←'];
  const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];

  RL.bandit = (o = {}) => {
    const k = o.k || (o.means ? o.means.length : 10), kind = o.kind || 'bernoulli', g = S.T.rng(o.seed || 1);
    const means = o.means ? o.means.slice() : Array.from({ length: k }, () => (kind === 'bernoulli' ? 0.1 + 0.8 * g.next() : g.normal(0, 1)));
    let best = 0; means.forEach((m, i) => { if (m > means[best]) best = i; });
    const sd = o.sd || 1;
    return {
      kind, k, means, best, bestMean: means[best], sd,
      pull: (a, rng) => (kind === 'bernoulli' ? (rng.next() < means[a] ? 1 : 0) : rng.normal(means[a], sd)),
      say: 'A ' + k + '-armed ' + kind + ' bandit; the best arm is ' + best + ' with mean reward ' + RL.fmt(means[best]) + '.',
    };
  };

  RL.grid = (o = {}) => {
    const wd = typeof o.world === 'object' && o.world ? o.world : S.Data.gridWorld({ name: o.world || 'pits', seed: o.seed, w: o.w, h: o.h });
    const w = wd.w, h = wd.h, n = w * h, ch = (s) => wd.rows[Math.floor(s / w)][s % w];
    const rw = { step: o.step == null ? -0.04 : o.step, goal: o.goal == null ? 1 : o.goal, pit: o.pit == null ? -1 : o.pit };
    const slip = o.slip || 0, pitMode = o.pitMode || 'end', start = wd.start[0] * w + wd.start[1];
    const wall = Array.from({ length: n }, (_, s) => ch(s) === '#'), goal = Array.from({ length: n }, (_, s) => ch(s) === 'G'), pit = Array.from({ length: n }, (_, s) => ch(s) === 'X');
    const terminal = Array.from({ length: n }, (_, s) => goal[s] || (pit[s] && pitMode === 'end'));
    const move = (s, a) => { const i = Math.floor(s / w) + MOVES[a][0], j = (s % w) + MOVES[a][1]; if (i < 0 || j < 0 || i >= h || j >= w || wall[i * w + j]) return s; return i * w + j; };
    const land = (s2) => (goal[s2] ? { s2, r: rw.goal, done: true } : pit[s2] ? (pitMode === 'end' ? { s2, r: rw.pit, done: true } : { s2: start, r: rw.pit, done: false, fell: true }) : { s2, r: rw.step, done: false });
    const P = (s, a) => {
      if (terminal[s] || wall[s]) return [{ p: 1, s2: s, r: 0, done: true }];
      const outs = slip ? [[a, 1 - slip], [(a + 1) % 4, slip / 2], [(a + 3) % 4, slip / 2]] : [[a, 1]], m = new Map();
      for (const [b, p] of outs) { const l = land(move(s, b)), key = l.s2 + '|' + l.r; if (m.has(key)) m.get(key).p += p; else m.set(key, Object.assign({ p }, l)); }
      return [...m.values()];
    };
    const env = {
      name: wd.name, world: wd, w, h, n, nA: 4, gamma: o.gamma || 0.95, start, rewards: rw, slip, pitMode, wall, goal, pit, terminal,
      states: [...Array(n).keys()].filter((s) => !wall[s] && !terminal[s]),
      cell: (s) => [Math.floor(s / w), s % w], P, move,
      reset: () => start,
      sample(s, a, rng) { if (!slip) return land(move(s, a)); const u = rng.next(), b = u < 1 - slip ? a : u < 1 - slip / 2 ? (a + 1) % 4 : (a + 3) % 4; return land(move(s, b)); },
      /** a one-hot (or row, column) encoding of a state, for function approximation */
      features(s, kind = 'onehot') { if (kind === 'coords') { const [i, j] = env.cell(s); return [i / Math.max(1, h - 1), j / Math.max(1, w - 1)]; } const v = new Array(n).fill(0); v[s] = 1; return v; },
      arrows: (policy) => wd.rows.map((row, i) => [...row].map((c, j) => { const s = i * w + j; return wall[s] ? '#' : goal[s] ? 'G' : pit[s] && pitMode === 'end' ? 'X' : RL.ARROWS[policy[s]]; }).join('')),
      say: 'A ' + h + '×' + w + ' grid world (' + wd.name + '): ' + RL.fmt(rw.step) + ' per step, ' + RL.fmt(rw.goal) + ' at the goal, ' + RL.fmt(rw.pit) + ' in a pit' + (slip ? ', and the agent slips sideways with probability ' + RL.fmt(slip) : '') + '.',
    };
    return env;
  };
  /** a map from rows of text ('.' floor, '#' wall, 'S' start, 'G' goal, 'X' pit), in the form S.Data.gridWorld gives */
  RL.parseMap = (rows, name = 'custom') => {
    const find = (ch) => { const out = []; rows.forEach((row, i) => [...row].forEach((c, j) => { if (c === ch) out.push([i, j]); })); return out; };
    return { rows, h: rows.length, w: rows[0].length, start: find('S')[0], goals: find('G'), pits: find('X'), walls: find('#'), name };
  };
  // cliff walking: start and goal at the two ends of the bottom row, with the cliff between them
  RL.CLIFF = ['............', '............', '............', 'SXXXXXXXXXXG'];
  RL.cliff = (o = {}) => RL.grid(Object.assign({ world: RL.parseMap(RL.CLIFF, 'cliff walking'), step: -1, goal: 0, pit: -100, pitMode: 'restart', gamma: 0.99 }, o));

  /** cart-pole: a pole hinged on a cart that is pushed left or right with a fixed force (Euler steps of 0.02 s) */
  RL.cartPole = (o = {}) => {
    const ph = Object.assign({ gravity: 9.8, massCart: 1, massPole: 0.1, length: 0.5, force: 10, tau: 0.02, xLimit: 2.4, thetaLimit: (12 * Math.PI) / 180 }, o.physics || {});
    const total = ph.massCart + ph.massPole, pml = ph.massPole * ph.length, rng = S.T.rng(o.seed || 1), maxSteps = o.maxSteps || 500;
    const env = {
      physics: ph, maxSteps, state: [0, 0, 0, 0], t: 0, done: false, nA: 2, dim: 4,
      reset(g) { const r = g || rng; env.state = [0, 0, 0, 0].map(() => -0.05 + 0.1 * r.next()); env.t = 0; env.done = false; return env.state.slice(); },
      /** the accelerations for a push f; returned for pictures as well as used by step */
      dynamics(s, f) {
        const [, , th, thd] = s, c = Math.cos(th), sn = Math.sin(th);
        const temp = (f + pml * thd * thd * sn) / total;
        const thAcc = (ph.gravity * sn - c * temp) / (ph.length * (4 / 3 - (ph.massPole * c * c) / total));
        const xAcc = temp - (pml * thAcc * c) / total;
        return { xAcc, thAcc };
      },
      step(a) {
        const f = a === 1 ? ph.force : -ph.force, s = env.state, { xAcc, thAcc } = env.dynamics(s, f);
        env.state = [s[0] + ph.tau * s[1], s[1] + ph.tau * xAcc, s[2] + ph.tau * s[3], s[3] + ph.tau * thAcc];
        env.t++;
        const fell = Math.abs(env.state[0]) > ph.xLimit || Math.abs(env.state[2]) > ph.thetaLimit;
        env.done = fell || env.t >= maxSteps;
        return { s: env.state.slice(), r: 1, done: env.done, fell, xAcc, thAcc };
      },
      explain(a) {
        const s = env.state, f = a === 1 ? ph.force : -ph.force, d = env.dynamics(s, f);
        return S.T.step('physics', 'Pushing ' + (a ? 'right' : 'left') + ' with ' + RL.fmt(f) + ' N at angle ' + RL.fmt((s[2] * 180) / Math.PI) + '° gives the pole an angular acceleration of ' + RL.fmt(d.thAcc) + ' rad/s² and the cart ' + RL.fmt(d.xAcc) + ' m/s².', d);
      },
    };
    env.reset();
    return env;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
