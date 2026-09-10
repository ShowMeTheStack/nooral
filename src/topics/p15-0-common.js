// Part 15, reinforcement learning: the families, the grid worlds the topics use, and the pictures they share: a
// board of cells coloured by their value with the policy's arrows, the same board split into four action values
// (S.Kit.grid), a path walked across it, bandit arms, and a cart with its pole.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'rl', order, title, blurb });
  fam('p15-bandits', 10, 'Bandits', 'The simplest choice under uncertainty: several slot machines with unknown payouts. Exploring against exploiting with ε-greedy, upper confidence bounds and Thompson sampling, and the regret each one runs up.');
  fam('p15-planning', 20, 'Markov decision processes and planning', 'States, actions, transitions, rewards and discounting; and, when the model is known, value iteration and policy iteration on a grid world, with the values and arrows drawn.');
  fam('p15-tabular', 30, 'Learning from experience', 'When the model is unknown the agent must learn from its own episodes: Monte Carlo control, Q-learning against SARSA on a cliff, and how much to explore.');
  fam('p15-deep', 40, 'Deep reinforcement learning', 'A network in place of the table: a policy gradient balancing a pole on a cart, and a deep Q-network with a replay buffer and a target network.');

  const P = (S.P15 = {}), H = () => S.Kit.ml;
  const memo = new Map();
  P.once = (key, f) => { if (!memo.has(key)) memo.set(key, f()); return memo.get(key); };
  P.D = () => S.P15D;
  P.ACT = ['north', 'east', 'south', 'west'];
  const MOVE = [[-1, 0], [0, 1], [1, 0], [0, -1]];

  /** the worlds: name → options for S.RL.grid (the cliff has its own builder) */
  P.WORLDS = { pits: 'a field of pits (4×5)', rooms: 'two rooms (7×10)', cliff: 'cliff walking (4×12)' };
  P.env = (world, o) => (world === 'cliff' ? S.RL.cliff(o || {}) : S.RL.grid(Object.assign({ world }, o || {})));
  P.kind = (env, s) => (env.wall[s] ? 'wall' : env.goal[s] ? 'goal' : env.pit[s] ? 'pit' : s === env.start ? 'start' : '');
  P.cellName = (env, s) => { const [i, j] = env.cell(s); return '(' + i + ', ' + j + ')'; };
  /** frames whose narration may name cells: F.cell(env, s) quotes the row and column and keeps them in the meta */
  P.frames = (counters) => {
    const F = H().frames(counters), cells = [], push = F.push;
    F.cell = (env, s) => { const [i, j] = env.cell(s); cells.push(i, j); return '(' + F.q.n(i) + ', ' + F.q.n(j) + ')'; };
    F.push = (items, say, meta, extra) => push(items, say, Object.assign({}, meta || {}, { cells: cells.splice(0) }), extra);
    return F;
  };

  /** a board coloured by V with each cell's value and the policy's arrow; o: {prefix, env, x, y, cell, V, vmax,
   *  policy, values: true, hot: [states], dimOff: true, names, fmt} */
  P.board = (o) => {
    const env = o.env, c = o.cell, pre = o.prefix || 'bd', items = [], vmax = o.vmax || Math.max(1e-9, ...(o.V || [0]).map(Math.abs));
    for (let s = 0; s < env.n; s++) {
      const [i, j] = env.cell(s), x = o.x + j * c, y = o.y + i * c, k = P.kind(env, s), hot = o.hot && o.hot.includes(s);
      const open = !env.wall[s] && !env.terminal[s];
      let paint, text = '';
      if (k === 'goal') text = o.names && o.names.goal !== undefined ? o.names.goal : '+' + H().sg(env.rewards.goal, 2);
      else if (k === 'pit') text = env.pitMode === 'end' ? H().sg(env.rewards.pit, 2) : '';
      let vt = null;
      if (open && o.V) { const col = S.Kit.heat.colour(o.V[s], 'div', vmax); paint = { fill: col.fill, text: col.text }; if (o.values !== false) vt = { text: (o.fmt || ((v) => (Math.abs(v) >= 9.995 ? H().sg(v, 3) : H().fx(v, 2))))(o.V[s]), colour: col.text }; }
      items.push({ key: pre + ':c' + s, kind: 'cell', x: H().r2(x), y: H().r2(y), w: c, h: c, rx: 0, text, size: o.size || (c >= 60 ? 13 : 12), cls: 'gcell p15cell ' + k + (env.pit[s] && env.pitMode !== 'end' ? ' pit' : '') + (hot ? ' hotcell' : ''), paint, layer: 'back', z: 1, audit: 'ignore' });
      if (vt) items.push({ key: pre + ':v' + s, kind: 'text', x: H().r2(x + c / 2), y: H().r2(y + (o.policy ? c * 0.38 : c / 2 + 4)), text: vt.text, anchor: 'middle', size: o.size || (c >= 60 ? 13 : 12), cls: 'p15val', paint: { text: vt.colour } });
      if (open && o.policy && o.policy[s] >= 0 && !(o.skip && o.skip.includes(s))) {
        const L = c * 0.2, dx = MOVE[o.policy[s]][1], dy = MOVE[o.policy[s]][0], cx = x + c / 2, cy = y + c / 2 + (o.V && o.values !== false ? c * 0.2 : 0);
        items.push({ key: pre + ':a' + s, kind: 'wire', pts: [[H().r2(cx - dx * L), H().r2(cy - dy * L)], [H().r2(cx + dx * L), H().r2(cy + dy * L)]], head: true, cls: 'greedy p15arrow' + (o.changed && o.changed.includes(s) ? ' changed' : '') });
      }
      if (k === 'start' && o.startLabel !== false) items.push({ key: pre + ':s', kind: 'text', x: H().r2(x + 4), y: H().r2(o.V ? y + c - 5 : y + 13), text: 'S', size: 12, cls: 'gstart' });
    }
    items.push({ key: pre + ':frame', kind: 'path', d: `M${o.x},${o.y} h${env.w * c} v${env.h * c} h${-env.w * c} Z`, cls: 'gframe', audit: 'ignore', layer: 'top' });
    return items;
  };
  /** a path of states drawn through the cells' centres */
  P.path = (key, env, path, x, y, c, cls) => {
    const pts = path.map((s) => { const [i, j] = env.cell(s); return [H().r2(x + j * c + c / 2), H().r2(y + i * c + c / 2)]; });
    return { key, kind: 'path', d: pts.map((p, k) => (k ? 'L' : 'M') + p[0] + ',' + p[1]).join(' '), cls: 'p15path ' + (cls || ''), pathAudit: 'edge-x', layer: 'heads', box: H().boxOf(pts) };
  };
  /** the four action values of every open cell (S.Kit.grid), with the greedy arrow */
  P.qboard = (pre, env, x, y, c, Q, o) => {
    o = o || {};
    const qmax = Math.max(1e-9, ...Q.flatMap((q, s) => (env.states.includes(s) ? q.map(Math.abs) : [])));
    // along a walked path the path shows the way: those cells' arrows and the start's corner name are left out
    const drop = new Set((o.path || []).map((s) => { const [i, j] = env.cell(s); return pre + ':a' + i + '_' + j; }));
    drop.add(pre + ':s');
    const items = S.Kit.grid.world({
      prefix: pre, x, y, cell: c, rows: env.h, cols: env.w, kinds: (r, cc) => P.kind(env, r * env.w + cc), qmax: o.qmax || qmax, values: o.values,
      q: (r, cc) => { const s = r * env.w + cc; return env.states.includes(s) ? Q[s] : null; },
      best: (r, cc) => { const s = r * env.w + cc; if (!env.states.includes(s)) return -1; const q = Q[s]; if (q.every((v) => v === q[0])) return -1; return q.indexOf(Math.max(...q)); },
      names: { goal: o.goal || '+' + H().sg(env.rewards.goal, 2), pit: env.pitMode === 'end' ? H().sg(env.rewards.pit, 2) : ' ', start: 'start' }, hot: o.hot, agent: o.agent, fmt: o.fmt,
    }).filter((it) => !drop.has(it.key)).map((it) => (it.kind === 'cell' && /gcell\s*($|hotcell)/.test(it.cls) ? Object.assign(it, { audit: 'ignore' }) : it));
    { const [i, j] = env.cell(env.start); items.push({ key: pre + ':S', kind: 'text', x: H().r2(x + j * c + 3), y: H().r2(y + i * c + c - 4), text: 'S', size: 12, cls: 'gstart' }); }
    return items;
  };
  /** a moving average over k values */
  P.smooth = (a, k) => a.map((_, i) => { const lo = Math.max(0, i - k + 1); let s = 0; for (let j = lo; j <= i; j++) s += a[j]; return s / (i - lo + 1); });
  P.mean = (a) => a.reduce((s, v) => s + v, 0) / Math.max(1, a.length);
})(globalThis.Nooral = globalThis.Nooral || {});
