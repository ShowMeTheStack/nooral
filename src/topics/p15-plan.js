// Part 15, Markov decision processes and planning with a known model: the parts of an MDP on a slippery grid
// (S.RL.grid: its transition model, a sampled episode and its discounted return), then value iteration sweep by sweep
// and policy iteration step by step (S.RL.valueIteration, S.RL.policyIteration). All live.
(function (S) {
  'use strict';
  const P = S.P15, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const MOVE = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  const cellOf = (c) => (c >= 60 ? 13 : 12);
  /** the size of a cell so a world fits about 520 units wide and 330 high */
  P.cellSize = (env) => Math.min(84, Math.floor(520 / env.w), Math.floor(330 / env.h));
  const WORLD_OPTS = Object.entries(P.WORLDS).map(([value, label]) => ({ value, label }));

  S.topic({
    id: 'rl-mdp', family: 'p15-planning', order: 10, title: 'A Markov decision process',
    blurb: 'An MDP is the frame of reinforcement learning: states the agent can be in, actions it can take, a transition model saying where each action may lead and how likely, a reward for each move, and a discount γ that makes rewards sooner worth more than rewards later. Here: a 4×5 grid with walls, two pits and a goal, on slippery ground where an action can slide sideways.',
    inputs: [
      { id: 'slip', kind: 'slider', label: 'Chance of slipping sideways', def: 0.2, min: 0, max: 0.4, step: 0.05, fmt: (v) => String(v) },
      { id: 'step', kind: 'slider', label: 'Reward per step', def: -0.04, min: -0.3, max: 0, step: 0.01, fmt: (v) => String(v) },
      { id: 'gamma', kind: 'slider', label: 'Discount γ', def: 0.9, min: 0.5, max: 0.99, step: 0.01, fmt: (v) => String(v) },
      { id: 'act', kind: 'choice', label: 'Action to look at', def: '1', options: P.ACT.map((a, i) => ({ value: String(i), label: a })) },
      { id: 'seed', kind: 'number', label: 'Seed of the random walk', def: 4, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Slippery ground', values: { slip: 0.2, step: -0.04, gamma: 0.9, act: '1', seed: 4 } },
      { label: 'Firm ground', values: { slip: 0, step: -0.04, gamma: 0.9, act: '1', seed: 4 } },
      { label: 'Very slippery, costly steps', values: { slip: 0.4, step: -0.2, gamma: 0.9, act: '2', seed: 9 } },
      { label: 'Short-sighted (γ = 0.5)', values: { slip: 0.2, step: -0.04, gamma: 0.5, act: '1', seed: 4 } },
    ],
    counters: [{ id: 'moves', label: 'moves' }],
    run(inp) {
      const env = P.env('pits', { slip: inp.slip, step: inp.step, gamma: inp.gamma }), F = P.frames(['moves']), q = F.q, c = 80, X0 = 40, Y0 = 50, a = +inp.act;
      const pick = 12, [pi, pj] = env.cell(pick), cx = X0 + pj * c + c / 2, cy = Y0 + pi * c + c / 2;
      const board = (o) => P.board(Object.assign({ prefix: 'md', env, x: X0, y: Y0, cell: c }, o || {}));
      const panel = (rows) => H().lines('md:p', 470, 70, rows, 13, 24);
      const nOpen = env.states.length, pits = env.pit.filter(Boolean).length;
      F.push([...board(), ...panel([['states: the open cells', 'legend'], ['actions: north, east, south, west', 'legend'], ['the goal and the pits end an episode', 'legend']])], `The states are the ${q.n(nOpen)} open cells; the goal and the ${q.n(pits)} pits end an episode, and walls cannot be entered. In every state there are ${q.n(4)} actions: north, east, south and west.`, { phase: 'states', nOpen, pits, four: 4 });
      const acts = [0, 1, 2, 3].map((b) => H().arrow('md:act' + b, [cx + MOVE[b][1] * 12, cy + MOVE[b][0] * 12], [cx + MOVE[b][1] * 34, cy + MOVE[b][0] * 34], b === a ? 'p15push' : 'muted'));
      F.push([...board({ hot: [pick] }), ...acts, ...panel([['the agent is in cell ' + P.cellName(env, pick), 'legend'], ['it chooses ' + P.ACT[a], 'legend']])], `Take the agent in cell ${F.cell(env, pick)}. It chooses ${P.ACT[a]}. Whether it gets there is up to the ground: with probability ${q(inp.slip, 2)} it slides to one side or the other instead.`, { phase: 'actions', slip: inp.slip, s: pick, a });
      const outs = env.P(pick, a), fan = outs.map((o, k) => { const [i, j] = env.cell(o.s2), tx = X0 + j * c + c / 2, ty = Y0 + i * c + c / 2, same = o.s2 === pick; return [same ? { key: 'md:f' + k, kind: 'path', d: `M${cx - 14},${cy - 18} a14,14 0 1,1 28,0`, cls: 'p15fan', layer: 'heads', box: { x: cx - 14, y: cy - 32, w: 28, h: 14 } } : H().arrow('md:f' + k, [cx + (tx - cx) * 0.2, cy + (ty - cy) * 0.2], [cx + (tx - cx) * 0.78, cy + (ty - cy) * 0.78], 'p15push'), H().text('md:fp' + k, same ? cx : tx, same ? cy - 36 : ty + 5, H().fx(o.p, 2), 'p15p', 13, 'middle')]; }).flat();
      const desc = outs.map((o) => q(o.p, 2) + ' to ' + (o.s2 === pick ? 'stay put (it bumps into a wall)' : F.cell(env, o.s2)) + ' with reward ' + q(o.r, 2));
      F.push([...board({ hot: [pick] }), ...fan, ...panel([['P(s′ | s, a): where the action leads', 'legend'], ['the chances add up to 1', 'legend']])], `The transition model P(s′ | s, ${P.ACT[a]}) for this cell: ${H().list(desc)}. Only the current cell and action matter, not how the agent got here: that is the Markov property.`, { phase: 'model', outs: outs.map((o) => [o.p, o.s2, o.r]) });
      F.push([...board(), ...panel([['reward per step: ' + H().sg(env.rewards.step, 2), 'legend'], ['reaching the goal: ' + H().sg(env.rewards.goal, 2), 'legend'], ['falling in a pit: ' + H().sg(env.rewards.pit, 2), 'legend'], ['discount γ = ' + H().sg(inp.gamma, 2), 'legend']])], `Rewards: ${q(env.rewards.step, 2)} for every move, ${q(env.rewards.goal, 0)} for reaching the goal and ${q(env.rewards.pit, 0)} for a pit. A small cost per step makes shorter routes better; the discount γ = ${q(inp.gamma, 2)} makes a reward k steps away worth γᵏ of itself now.`, { phase: 'rewards', step: env.rewards.step, goal: env.rewards.goal, pit: env.rewards.pit, gamma: inp.gamma });
      // a random walk from the start
      const g = S.T.rng(inp.seed), path = [env.start], rs = [];
      let s = env.start;
      for (let t = 0; t < 40; t++) { const o = env.sample(s, g.int(4), g); rs.push(o.r); s = o.s2; path.push(s); if (o.done) break; }
      const G = rs.reduce((acc, r, k) => acc + Math.pow(inp.gamma, k) * r, 0), end = env.goal[s] ? 'ending in the goal' : env.pit[s] ? 'ending in a pit' : 'without reaching the goal or a pit';
      F.set('moves', rs.length);
      F.push([...board(), P.path('md:walk', env, path, X0, Y0, c), ...panel([['an episode: actions chosen at random', 'legend'], [rs.length + (rs.length === 1 ? ' move, ' : ' moves, ') + end, 'legend']])], `An episode with actions chosen at random: ${q.n(rs.length)} ${rs.length === 1 ? 'move' : 'moves'} from the start, ${end}. The rewards were ${q(env.rewards.step, 2)} per move` + (env.goal[s] || env.pit[s] ? `, then ${q(rs[rs.length - 1], 0)} at the end.` : '.'), { phase: 'walk', moves: rs.length, rewards: rs, step: env.rewards.step, last: rs[rs.length - 1] });
      const terms = rs.slice(0, 3).map((r, k) => q.t(r, 2, k === 0) + (k ? ' × ' + q(Math.pow(inp.gamma, k), 3) : '')), Gs = [0.5, 0.9, 0.99].map((gg) => rs.reduce((acc, r, k) => acc + Math.pow(gg, k) * r, 0));
      F.push([...board(), P.path('md:walk', env, path, X0, Y0, c), ...panel([['G = r₀ + γ r₁ + γ² r₂ + …', 'legend'], ['the return of this episode', 'legend']])], `Its return G = r₀ + γ r₁ + γ² r₂ + … = ${terms.join('')}${rs.length > 3 ? ' + …' : ''} = ${q.sg(G, 3)}. With γ = ${q(0.5, 1)}, ${q(0.9, 1)} or ${q(0.99, 2)} the same rewards would be worth ${q.sg(Gs[0], 3)}, ${q.sg(Gs[1], 3)} and ${q.sg(Gs[2], 3)}: the smaller γ, the less the distant end matters.`, { phase: 'return', G, Gs, gammas: [0.5, 0.9, 0.99], pw: [1, inp.gamma, inp.gamma ** 2], rewards: rs });
      const east = new Array(env.n).fill(1), ev = S.RL.evaluatePolicy(env, east, inp.gamma, 1e-9), vi = S.RL.valueIteration(env, { gamma: inp.gamma });
      F.push([...board({ V: ev.V, policy: east }), ...panel([['V^π(s): the expected return from s', 'legend'], ['here π = always go east', 'legend']])], `A policy says what to do in each state. The value V^π(s) is the expected return from s when following π. For “always go east”, V at the start is ${q.sg(ev.V[env.start], 3)}: averaged over every way the slips can go.`, { phase: 'policy', vEast: ev.V[env.start] });
      F.push([...board({ V: vi.V, policy: vi.policy }), ...panel([['the best policy π* and its values V*', 'legend'], ['found by planning: the next topics', 'legend']])], `The best policy does better: V* at the start is ${q.sg(vi.V[env.start], 3)}. Finding π* when the model is known is planning: [[rl-value-iteration]] and [[rl-policy-iteration]] do it.`, { phase: 'best', vStar: vi.V[env.start] });
      for (const f of F.list) f.narrowView = { x: 20, y: 30, w: 440, h: 360 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The parts of an MDP', text: ['states S, actions A', 'P(s′ | s, a): the transition model', 'R(s, a, s′): the reward', 'γ: the discount, 0 ≤ γ < 1', 'G = Σ γᵏ r_k,  V^π(s) = E[G | start in s, follow π]'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'policy', pick: 'option', ask: 'Will the best policy’s value at the start be higher or lower than “always east”?', options: ['higher', 'lower', 'the same'], answer: (fr) => { const a = fr.find((f) => f.meta.phase === 'policy').meta.vEast, b = fr[fr.length - 1].meta.vStar; return Math.abs(a - b) < 1e-9 ? 'the same' : b > a ? 'higher' : 'lower'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Rewards −1, −1, +10 with γ = 0.9. What is the return?', options: ['−1 − 0.9 + 8.1 = 6.2', '8', '10', '6.1'], answer: 0, why: 'G = −1 + 0.9 × (−1) + 0.81 × 10.' },
      { q: 'What does the Markov property say?', options: ['what happens next depends only on the current state and action', 'rewards are always positive', 'the agent remembers everything', 'transitions are deterministic'], answer: 0, why: 'The state must carry everything the future depends on.' },
      { q: 'Why use a discount γ < 1?', options: ['it values sooner rewards more and keeps infinite sums finite', 'it makes rewards negative', 'it speeds up the agent', 'it removes randomness'], answer: 0, why: 'With γ < 1 the return of an endless episode stays bounded.' },
      { q: 'With slip 0.2, an action goes the intended way with probability…', options: ['0.8', '0.2', '0.1', '1'], answer: 0, why: 'It slides to each side with 0.1, so straight on with 0.8.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  const planInputs = [
    { id: 'world', kind: 'choice', label: 'World', def: 'pits', options: WORLD_OPTS },
    { id: 'gamma', kind: 'slider', label: 'Discount γ', def: 0.95, min: 0.5, max: 0.99, step: 0.01, fmt: (v) => String(v) },
    { id: 'slip', kind: 'slider', label: 'Chance of slipping', def: 0, min: 0, max: 0.4, step: 0.05, fmt: (v) => String(v) },
  ];
  const planEnv = (inp) => P.env(inp.world, inp.world === 'cliff' ? { slip: inp.slip, gamma: inp.gamma } : { slip: inp.slip, gamma: inp.gamma });

  S.topic({
    id: 'rl-value-iteration', family: 'p15-planning', order: 20, title: 'Value iteration',
    blurb: 'When the model is known, the best values can be computed. Value iteration starts with V = 0 everywhere and sweeps over the states, replacing each V(s) by the best one-step look-ahead: max over actions of Σ P(s′ | s, a) (r + γ V(s′)). News of the goal spreads one cell per sweep; when nothing changes any more, the arrows pointing to the best neighbour are the optimal policy.',
    inputs: planInputs,
    presets: [
      { label: 'Pits, firm ground', values: { world: 'pits', gamma: 0.95, slip: 0 } },
      { label: 'Pits, slippery', values: { world: 'pits', gamma: 0.95, slip: 0.2 } },
      { label: 'Two rooms', values: { world: 'rooms', gamma: 0.95, slip: 0 } },
      { label: 'The cliff', values: { world: 'cliff', gamma: 0.99, slip: 0 } },
    ],
    counters: [{ id: 'sweep', label: 'sweeps' }],
    run(inp) {
      const env = planEnv(inp), res = S.RL.valueIteration(env, { gamma: inp.gamma }), F = P.frames(['sweep']), q = F.q, c = P.cellSize(env), K = res.sweeps.length;
      const vmax = Math.max(1e-9, ...res.V.map(Math.abs)), bw = env.w * c, ax0 = 60 + bw + 50;
      const ld = res.sweeps.map((s) => Math.log10(Math.max(1e-12, s.delta))), lo = Math.floor(Math.min(...ld)), hi = Math.ceil(Math.max(...ld));
      const ax = C().axes({ prefix: 'vd', x: ax0, y: 60, w: 220, h: 160, xr: [0, Math.max(2, K)], yr: [lo, hi], xticks: C().ticks(0, K, 4), yticks: C().ticks(lo, hi, 3).filter((v) => Number.isInteger(v)), fmtY: (v) => '10' + (v < 0 ? '⁻' : '') + String(Math.abs(v)).split('').map((d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]).join(''), xname: 'sweep', yname: 'largest change' });
      const show = [...new Set([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 80, 120, 200, 300, K].filter((k) => k <= K))];
      F.push([...P.board({ prefix: 'vi', env, x: 60, y: 50, cell: c, V: new Array(env.n).fill(0), vmax, size: cellOf(c) }), ...ax.items], `Start with V = ${q.n(0)} in every cell. Each sweep replaces V(s) by max over actions of Σ P(s′ | s, a) (r + γ V(s′)), with γ = ${q(inp.gamma, 2)}; the values at the goal and the pits stay ${q.n(0)}, since an episode ends there.`, { phase: 'init', gamma: inp.gamma, zero: 0 });
      for (const k of show) {
        F.set('sweep', k);
        const sw = res.sweeps[k - 1], prev = k > 1 ? res.sweeps[k - 2].V : new Array(env.n).fill(0), where = sw.cell !== undefined ? sw.cell : res.trace[k].cell;
        const b = S.RL.backup(env, prev, where, inp.gamma), qs = b.q;
        const items = [...P.board({ prefix: 'vi', env, x: 60, y: 50, cell: c, V: sw.V, vmax, policy: sw.policy, hot: [where], size: cellOf(c) }), ...ax.items, C().line(ax, 'vd:l', ld.slice(0, k).map((v, j) => [j + 1, v]), 'p15m'), C().vline(ax, 'vd:now', k, 'act')];
        if (sw.delta === 0) { F.push(items, `Sweep ${q.n(k)}: no value changes at all. On firm ground every route is certain, so the values are already exact.`, { phase: 'sweep', k, delta: 0, cell: where, q: qs, best: b.best, V: sw.V }); continue; }
        F.push(items, `Sweep ${q.n(k)}: the largest change, ${q.sg(sw.delta, 3)}, is at cell ${F.cell(env, where)}. Its look-ahead values were ${H().list(qs.map((v, i) => P.ACT[i] + ' ' + q.sg(v, 3)))}; the best becomes its new V, ${q.sg(qs[b.best], 3)}.`, { phase: 'sweep', k, delta: sw.delta, cell: where, q: qs, best: b.best, V: sw.V });
      }
      const start = res.V[env.start], path = S.RL.greedyPath(env, res.policy, 200);
      F.push([...P.board({ prefix: 'vi', env, x: 60, y: 50, cell: c, V: res.V, vmax, policy: res.policy, size: cellOf(c), hot: path.reached ? path.path.slice(1, -1) : [] }), ...ax.items, C().line(ax, 'vd:l', ld.map((v, j) => [j + 1, v]), 'p15m')], `Converged after ${q.n(K)} sweeps: the largest change fell below ${q.sg(1e-6, 1)}. V at the start is ${q.sg(start, 3)}, and following the arrows from the start ` + (path.reached ? `reaches the goal in ${q.n(path.steps)} moves.` : 'does not reach the goal.'), { phase: 'done', K, theta: 1e-6, start, steps: path.steps, reached: path.reached, V: res.V, policy: res.policy });
      for (const f of F.list) f.narrowView = { x: 40, y: 30, w: env.w * c + 40, h: env.h * c + 40 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'The Bellman optimality update', text: ['V(s) ← max over a of Σ P(s′ | s, a) [r + γ V(s′)]', 'γ = ' + inp.gamma + ',  stop when the largest change < 10⁻⁶', 'π*(s) = the a that achieves the max'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'init', pick: 'option', ask: 'After the first sweep, which cells will have a nonzero value?', options: ['only cells next to the goal or a pit (and any step cost everywhere)', 'every cell equally', 'only the start'], answer: () => 'only cells next to the goal or a pit (and any step cost everywhere)', explain: (fr) => fr[1].say },
    ],
    quiz: [
      { q: 'A cell’s neighbours have V = 0.5 (north) and 0.8 (east); r = −0.04 and γ = 0.9, no slipping. What does one update give?', options: ['−0.04 + 0.9 × 0.8 = 0.68', '0.8', '0.72', '0.41'], answer: 0, why: 'Take the best action: east.' },
      { q: 'Why does value spread out from the goal one cell per sweep?', options: ['each update looks only one step ahead', 'the goal moves', 'γ is small', 'the walls block it'], answer: 0, why: 'A cell k steps from the goal first feels it on sweep k.' },
      { q: 'What is the optimal policy once V* is known?', options: ['in each state, the action with the best one-step look-ahead', 'the action taken most often', 'random', 'always towards the start'], answer: 0, why: 'Acting greedily with respect to V* is optimal.' },
      { q: 'Why can value iteration stop before V is exact?', options: ['the error shrinks by a factor γ each sweep, so a small change means V is close', 'it runs out of states', 'the policy is always random', 'the rewards are zero'], answer: 0, why: 'The Bellman update is a contraction.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'rl-policy-iteration', family: 'p15-planning', order: 30, title: 'Policy iteration',
    blurb: 'Policy iteration alternates two steps. Evaluation: work out V^π for the current policy exactly (here by repeated sweeps until nothing changes). Improvement: in every state switch to the action that looks best given V^π. A policy can only get better this way, and there are finitely many, so it stops, usually after very few rounds, at the optimal policy.',
    inputs: planInputs.concat([{ id: 'init', kind: 'choice', label: 'Starting policy', def: '0', options: P.ACT.map((a, i) => ({ value: String(i), label: 'always ' + a })) }]),
    presets: [
      { label: 'Pits, start “always north”', values: { world: 'pits', gamma: 0.95, slip: 0, init: '0' } },
      { label: 'Pits, slippery, start “always west”', values: { world: 'pits', gamma: 0.95, slip: 0.2, init: '3' } },
      { label: 'Two rooms', values: { world: 'rooms', gamma: 0.95, slip: 0, init: '1' } },
      { label: 'The cliff', values: { world: 'cliff', gamma: 0.99, slip: 0, init: '0' } },
    ],
    counters: [{ id: 'round', label: 'rounds' }],
    run(inp) {
      const env = planEnv(inp), init = +inp.init, res = S.RL.policyIteration(env, { gamma: inp.gamma, init }), F = P.frames(['round']), q = F.q, c = P.cellSize(env);
      const vmax = Math.max(1e-9, ...res.iterations.flatMap((it) => it.V.map(Math.abs))), p0 = new Array(env.n).fill(init);
      const bx = 60 + env.w * c + 50, side = (rows) => H().lines('pi:s', bx, 70, rows, 13, 24);
      const at = (V) => V[env.start];
      F.push([...P.board({ prefix: 'pi', env, x: 60, y: 50, cell: c, policy: p0, size: cellOf(c) }), ...side([['π₀: always ' + P.ACT[init], 'legend']])], `Start with the policy “always ${P.ACT[init]}” in every cell.`, { phase: 'init', init });
      let pol = p0;
      res.iterations.forEach((it) => {
        F.set('round', it.k);
        F.push([...P.board({ prefix: 'pi', env, x: 60, y: 50, cell: c, V: it.V, vmax, policy: pol, size: cellOf(c) }), ...side([['round ' + it.k + ': evaluate π', 'legend'], [it.evalSweeps + ' sweeps', 'legend']])], `Round ${q.n(it.k)}, evaluation: following the current arrows, V^π comes out after ${q.n(it.evalSweeps)} sweeps; at the start it is ${q.sg(at(it.V), 3)}.`, { phase: 'eval', k: it.k, sweeps: it.evalSweeps, vStart: at(it.V), V: it.V, policy: pol });
        const changed = env.states.filter((s) => it.policy[s] !== pol[s]);
        F.push([...P.board({ prefix: 'pi', env, x: 60, y: 50, cell: c, V: it.V, vmax, policy: it.policy, changed, size: cellOf(c) }), ...side([['round ' + it.k + ': improve π', 'legend'], [changed.length + ' arrows change', 'legend']])], changed.length ? `Improvement: acting greedily on those values changes the arrow in ${q.n(changed.length)} cell${changed.length === 1 ? '' : 's'} (lit).` : `Improvement: no cell wants a different action, so the policy is stable, and therefore optimal.`, { phase: 'improve', k: it.k, changed: changed.length, cells: changed, V: it.V, policy: it.policy, before: pol });
        pol = it.policy;
      });
      const vi = S.RL.valueIteration(env, { gamma: inp.gamma }), last = res.iterations[res.iterations.length - 1];
      F.push(F.last().items, `Policy iteration needed ${q.n(res.iterations.length)} rounds. Its value at the start, ${q.sg(at(last.V), 4)}, matches value iteration’s ${q.sg(at(vi.V), 4)}: both find the same optimum.`, { phase: 'done', rounds: res.iterations.length, vPI: at(last.V), vVI: at(vi.V) });
      for (const f of F.list) f.narrowView = { x: 40, y: 30, w: env.w * c + 40, h: env.h * c + 40 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Policy iteration', text: ['evaluate: V^π(s) = Σ P(s′ | s, π(s)) [r + γ V^π(s′)]', 'improve: π(s) ← argmax over a of Σ P(s′ | s, a) [r + γ V^π(s′)]', 'repeat until π does not change'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'init', pick: 'option', ask: 'How many rounds of evaluate-and-improve will it take?', options: ['1–3', '4–6', '7 or more'], answer: (fr) => { const n = fr[fr.length - 1].meta.rounds; return n <= 3 ? '1–3' : n <= 6 ? '4–6' : '7 or more'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'What does the evaluation step compute?', options: ['the value of every state under the current, fixed policy', 'the optimal values directly', 'a new policy', 'the rewards'], answer: 0, why: 'It answers: how good is this policy?' },
      { q: 'Why must policy iteration stop?', options: ['each round gives a policy at least as good, and there are finitely many policies', 'the values become zero', 'γ shrinks', 'it runs out of memory'], answer: 0, why: 'No policy repeats unless nothing changes.' },
      { q: 'Policy iteration usually needs fewer rounds than value iteration needs sweeps. Why is it not always faster?', options: ['each round’s evaluation is itself many sweeps', 'it never converges', 'its improvement step is random', 'it needs more memory per state'], answer: 0, why: 'The cost moves into the evaluation.' },
      { q: 'In the improvement step a state’s action stays the same when…', options: ['it is already among the best actions under the current values', 'its value is negative', 'it is next to a wall', 'never'], answer: 0, why: 'Keeping ties prevents endless switching between equal actions.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
