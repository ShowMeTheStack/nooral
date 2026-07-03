// Optimisers racing on a two-parameter loss surface, and Q-learning in a grid world, for the sample scenes (S.Toy).
(function (S) {
  'use strict';
  const T = (S.Toy = S.Toy || {});

  /** a curved valley: steep walls, a gently sloping floor that bends, its lowest point at (1, 1) */
  T.valley = {
    f: (u, v) => 0.5 * (1 - u) ** 2 + 4 * (v - u * u) ** 2,
    grad: (u, v) => [-(1 - u) - 16 * u * (v - u * u), 8 * (v - u * u)],
    min: [1, 1],
  };
  /** the paths of plain gradient descent, momentum and Adam from one start: {gd, mom, adam}: [[u, v]..] */
  T.race = function (surf, start, o) {
    o = o || {};
    const n = o.steps || 150, out = {};
    { // plain gradient descent
      let p = start.slice(); const path = [p.slice()], lr = o.lrGd || 0.02;
      for (let k = 0; k < n; k++) { const g = surf.grad(p[0], p[1]); p = [p[0] - lr * g[0], p[1] - lr * g[1]]; path.push(p.slice()); }
      out.gd = path;
    }
    { // momentum: a velocity that remembers past gradients
      let p = start.slice(), vel = [0, 0]; const path = [p.slice()], lr = o.lrMom || 0.02, beta = o.beta || 0.9;
      for (let k = 0; k < n; k++) { const g = surf.grad(p[0], p[1]); vel = [beta * vel[0] - lr * g[0], beta * vel[1] - lr * g[1]]; p = [p[0] + vel[0], p[1] + vel[1]]; path.push(p.slice()); }
      out.mom = path;
    }
    { // Adam: each parameter's step scaled by its own running size of gradient
      let p = start.slice(), m = [0, 0], s = [0, 0]; const path = [p.slice()], lr = o.lrAdam || 0.08, b1 = 0.9, b2 = 0.999;
      for (let k = 1; k <= n; k++) {
        const g = surf.grad(p[0], p[1]);
        m = [b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1]];
        s = [b2 * s[0] + (1 - b2) * g[0] ** 2, b2 * s[1] + (1 - b2) * g[1] ** 2];
        const mh = [m[0] / (1 - b1 ** k), m[1] / (1 - b1 ** k)], sh = [s[0] / (1 - b2 ** k), s[1] / (1 - b2 ** k)];
        p = [p[0] - (lr * mh[0]) / (Math.sqrt(sh[0]) + 1e-8), p[1] - (lr * mh[1]) / (Math.sqrt(sh[1]) + 1e-8)];
        path.push(p.slice());
      }
      out.adam = path;
    }
    return out;
  };

  // ---- a grid world and Q-learning ------------------------------------------------------------------------------
  /** rows of the map: S start, G goal (+1), X pit (−1), # wall, . open */
  T.world = { map: ['...G', '.#.X', '....', 'S...'], step: -0.04 };
  T.worldAt = (w, r, c) => w.map[r][c];
  /**
   * Q-learning with ε-greedy exploration (seeded): Q[r][c] = [north, east, south, west]. Snapshots after the
   * episodes asked: {episode, Q, steps (moves in that episode), ret (its return), path (the last episode's cells)}.
   */
  T.qlearn = function (w, o) {
    o = o || {};
    const R = w.map.length, Cn = w.map[0].length, r = T.rng(o.seed || 21), alpha = o.alpha || 0.5, gamma = o.gamma || 0.9;
    const Q = Array.from({ length: R }, () => Array.from({ length: Cn }, () => [0, 0, 0, 0]));
    const D = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    let start = [0, 0];
    w.map.forEach((row, i) => { const j = row.indexOf('S'); if (j >= 0) start = [i, j]; });
    const keep = o.keep || [1, 3, 10, 30, 100, 300], snaps = [], maxEp = Math.max(...keep), rets = [];
    const best = (q) => { let b = 0; for (let d = 1; d < 4; d++) if (q[d] > q[b] + 1e-12) b = d; return b; };
    for (let ep = 1; ep <= maxEp; ep++) {
      let s = start.slice(), ret = 0, steps = 0;
      const eps = Math.max(0.05, (o.eps || 0.5) * Math.pow(0.985, ep));
      const path = [s.slice()];
      while (steps < 60) {
        const q = Q[s[0]][s[1]];
        const a = r() < eps ? r.int(4) : best(q);
        // the move slips sideways one time in ten
        const slip = r(), act = slip < 0.05 ? (a + 1) % 4 : slip < 0.1 ? (a + 3) % 4 : a;
        let ns = [s[0] + D[act][0], s[1] + D[act][1]];
        if (ns[0] < 0 || ns[0] >= R || ns[1] < 0 || ns[1] >= Cn || w.map[ns[0]][ns[1]] === '#') ns = s.slice();
        const cell = w.map[ns[0]][ns[1]], done = cell === 'G' || cell === 'X';
        const rew = cell === 'G' ? 1 : cell === 'X' ? -1 : w.step;
        const target = rew + (done ? 0 : gamma * Math.max(...Q[ns[0]][ns[1]]));
        q[a] += alpha * (target - q[a]);
        ret += rew; steps++; s = ns; path.push(s.slice());
        if (done) break;
      }
      rets.push(ret);
      if (keep.includes(ep)) snaps.push({ episode: ep, Q: Q.map((row) => row.map((v) => v.slice())), steps, ret, path, eps });
    }
    return { snaps, best, start, rets };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
