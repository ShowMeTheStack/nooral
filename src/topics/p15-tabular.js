// Part 15, learning from experience with a Q-table (S.RL.tabular, live): Monte Carlo control averaging whole
// returns, Q-learning against SARSA on the cliff, and how the amount of exploration changes what an agent learns.
(function (S) {
  'use strict';
  const P = S.P15, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const eps2 = (v) => String(v);

  S.topic({
    id: 'rl-mc', family: 'p15-tabular', order: 10, title: 'Monte Carlo control',
    blurb: 'Without a model, the agent learns from what happens to it. Monte Carlo control plays a whole episode, ε-greedily, then walks back through it adding up the discounted return that followed the first visit to each state and action. Q(s, a) is the average of those returns, and the policy is greedy with respect to Q. Each cell below is split into four triangles, one per action, coloured by Q.',
    inputs: [
      { id: 'world', kind: 'choice', label: 'World', def: 'pits', options: [{ value: 'pits', label: P.WORLDS.pits }, { value: 'rooms', label: P.WORLDS.rooms }] },
      { id: 'eps', kind: 'slider', label: 'ε (exploration)', def: 0.2, min: 0.05, max: 0.5, step: 0.05, fmt: eps2 },
      { id: 'episodes', kind: 'number', label: 'Episodes', def: 300, min: 50, max: 1500 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 2, min: 1, max: 999 },
    ],
    presets: [
      { label: 'Pits, 300 episodes', values: { world: 'pits', eps: 0.2, episodes: 300, seed: 2 } },
      { label: 'Little exploration', values: { world: 'pits', eps: 0.05, episodes: 300, seed: 2 } },
      { label: 'Two rooms, 1000 episodes', values: { world: 'rooms', eps: 0.2, episodes: 1000, seed: 3 } },
    ],
    counters: [{ id: 'ep', label: 'episodes' }],
    run(inp) {
      const env = P.env(inp.world), gamma = 0.95, res = S.RL.tabular(env, { algo: 'mc', episodes: inp.episodes, eps: inp.eps, gamma, seed: inp.seed, snapEvery: 10, traceEpisode: 0, traceSteps: 60 });
      const F = P.frames(['ep']), q = F.q, c = P.cellSize(env), bx = 60 + env.w * c + 50, sm = P.smooth(res.returns, 20);
      const lo = Math.min(...sm), hi = Math.max(...sm), ax = C().axes({ prefix: 'mr', x: bx, y: 60, w: 230, h: 150, xr: [0, inp.episodes], yr: [Math.min(lo, -1), Math.max(hi, 1)], xticks: C().ticks(0, inp.episodes, 3), yticks: C().ticks(Math.min(lo, -1), Math.max(hi, 1), 3), xname: 'episode', yname: 'return (average of 20)' });
      const qmax = Math.max(1e-9, ...res.Q.flatMap((qq, s) => (env.states.includes(s) ? qq.map(Math.abs) : [])));
      const board = (Q, o) => P.qboard('mq', env, 60, 50, c, Q, Object.assign({ qmax }, o || {}));
      // the first episode's updates, replayed from the trace onto an empty table
      const ups = res.trace.filter((t) => t.kind === 'update'), Q0 = Array.from({ length: env.n }, () => [0, 0, 0, 0]);
      F.push([...board(Q0), ...ax.items], `All ${q.n(4)} action values start at ${q.n(0)} in every cell. The first episode is played ε-greedily with ε = ${q(inp.eps, 2)}: with ties everywhere it is a random walk. It lasted ${q.n(res.lengths[0])} moves and returned ${q.sg(res.returns[0], 3)}.`, { phase: 'start', four: 4, zero: 0, eps: inp.eps, len: res.lengths[0], ret: res.returns[0] });
      const showUps = ups.slice(0, 3);
      showUps.forEach((u, k) => {
        Q0[u.s][u.a] = u.now;
        F.push([...board(Q0.map((r) => r.slice()), { hot: env.cell(u.s) }), ...ax.items], (k === 0 ? 'Then it walks back from the end. ' : '') + `The return after the first visit to ${F.cell(env, u.s)} going ${P.ACT[u.a]} is G = ${q.sg(u.G, 3)}; as the average of ${q.n(1)} such return so far, Q moves from ${q.sg(u.old, 3)} to ${q.sg(u.now, 3)}.`, { phase: 'update', s: u.s, a: u.a, G: u.G, old: u.old, now: u.now, one: 1 });
      });
      for (const u of ups.slice(3)) Q0[u.s][u.a] = u.now;
      const set = [10, 20, 50, 100, 200, 300, 500, 1000, 1500].filter((e) => e < inp.episodes).concat([inp.episodes]);
      for (const e of set) {
        const sn = res.snapshots.find((x) => x.episode === e);
        if (!sn) continue;
        F.set('ep', e);
        const path = S.RL.greedyPath(env, sn.policy, 60), avg = sm[e - 1];
        const items = [...board(sn.Q, path.reached ? { path: path.path } : {}), ...ax.items, C().line(ax, 'mr:l', sm.slice(0, e).map((v, j) => [j + 1, v]), 'p15m'), C().vline(ax, 'mr:now', e, 'act')];
        if (path.reached) items.push(P.path('mq:path', env, path.path, 60, 50, c));
        F.push(items, `After ${q.n(e)} episodes the average return over the last ${q.n(Math.min(20, e))} is ${q.sg(avg, 3)}. Following the greedy arrows from the start ` + (path.reached ? `reaches the goal in ${q.n(path.steps)} moves.` : 'does not reach the goal yet.'), { phase: e === inp.episodes ? 'end' : 'snap', e, avg, window: Math.min(20, e), reached: path.reached, steps: path.steps, policy: sn.policy, returns: res.returns.slice(0, e) });
      }
      for (const f of F.list) f.narrowView = { x: 40, y: 30, w: env.w * c + 40, h: env.h * c + 40 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'First-visit Monte Carlo control', text: ['play an episode with an ε-greedy policy, ε = ' + inp.eps, 'walking back: G ← r + γ G,  γ = ' + gamma, 'first visit of (s, a): N ← N + 1,  Q ← Q + (G − Q) / N', 'policy: greedy in Q, with ε of random moves'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will the greedy path reach the goal by the end?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.reached ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Returns after visits to (s, a) were −0.2, 0.6 and 0.5. What is Q(s, a)?', options: ['0.3', '0.5', '0.9', '0.6'], answer: 0, why: 'Their average: 0.9 / 3.' },
      { q: 'What must Monte Carlo wait for before it can update?', options: ['the end of the episode, to know the return', 'the next step only', 'a model of the world', 'nothing'], answer: 0, why: 'The return adds up every reward to the end.' },
      { q: 'Why play ε-greedily instead of purely greedily?', options: ['so every action keeps being tried and its average stays honest', 'to finish episodes faster', 'to save memory', 'it makes Q exact'], answer: 0, why: 'An action never tried keeps its initial value for ever.' },
      { q: 'Walking back, r = −0.04 and the return after it is 0.9 with γ = 0.95. What is G here?', options: ['−0.04 + 0.95 × 0.9 = 0.815', '0.86', '0.9', '0.855'], answer: 0, why: 'G ← r + γ G.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'rl-cliff', family: 'p15-tabular', order: 20, title: 'Q-learning against SARSA on the cliff',
    blurb: 'Both learn after every step from a target r + γ · (the value of the next move). Q-learning uses the best next action, whatever it will really do: it learns the values of the greedy policy. SARSA uses the action it actually takes next, ε-random moves included: it learns the values of the policy it follows. On a path along a cliff, where a random step means a fall of −100, that difference shows in the routes they choose.',
    inputs: [
      { id: 'eps', kind: 'slider', label: 'ε (exploration)', def: 0.1, min: 0.02, max: 0.3, step: 0.01, fmt: eps2 },
      { id: 'alpha', kind: 'slider', label: 'Learning rate α', def: 0.2, min: 0.05, max: 1, step: 0.05, fmt: eps2 },
      { id: 'episodes', kind: 'number', label: 'Episodes', def: 500, min: 100, max: 1000 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'ε = 0.1', values: { eps: 0.1, alpha: 0.2, episodes: 500, seed: 1 } },
      { label: 'More exploration, ε = 0.25', values: { eps: 0.25, alpha: 0.2, episodes: 500, seed: 1 } },
      { label: 'Hardly any, ε = 0.02', values: { eps: 0.02, alpha: 0.2, episodes: 500, seed: 1 } },
      { label: 'A large step, α = 0.5', values: { eps: 0.1, alpha: 0.5, episodes: 500, seed: 1 } },
    ],
    counters: [{ id: 'ep', label: 'episodes' }],
    run(inp) {
      const env = P.env('cliff'), o = { episodes: inp.episodes, eps: inp.eps, alpha: inp.alpha, seed: inp.seed, snapEvery: 10, traceEpisode: 0, traceSteps: 4, maxSteps: 400 };
      const A = S.RL.tabular(env, Object.assign({ algo: 'q' }, o)), B = S.RL.tabular(env, Object.assign({ algo: 'sarsa' }, o)), F = P.frames(['ep']), q = F.q, c = 38, g = env.gamma;
      const sa = P.smooth(A.returns, 20), sb = P.smooth(B.returns, 20), lo = Math.max(-150, Math.min(...sa, ...sb)), ax = C().axes({ prefix: 'cr', x: 560, y: 70, w: 220, h: 230, xr: [0, inp.episodes], yr: [lo, 0], xticks: C().ticks(0, inp.episodes, 3), yticks: C().ticks(lo, 0, 4), xname: 'episode', yname: 'return (average of 20)' });
      const lg = C().legend('lg', 560, 360, [{ text: 'Q-learning', cls: 'p15q' }, { text: 'SARSA', cls: 'p15s' }], { dir: 'col', size: 13 });
      const boards = (QA, QB, pa, pb) => [H().text('h:q', 40, 40, 'Q-learning', 'strong', 14), ...P.qboard('cq', env, 40, 50, c, QA, { goal: 'G', path: pa && pa.reached ? pa.path : null }), H().text('h:s', 40, 240, 'SARSA', 'strong', 14), ...P.qboard('cs', env, 40, 250, c, QB, { goal: 'G', path: pb && pb.reached ? pb.path : null })].concat(pa && pa.reached ? [P.path('cq:p', env, pa.path, 40, 50, c, 'q')] : [], pb && pb.reached ? [P.path('cs:p', env, pb.path, 40, 250, c, 's')] : []);
      const zero = Array.from({ length: env.n }, () => [0, 0, 0, 0]);
      F.push([...boards(zero, zero), ...ax.items, ...lg], `The cliff: start at the bottom left, the goal at the bottom right, and the cliff between them. Every move costs ${q.sg(-1, 3)}; stepping off the cliff costs ${q.sg(-100, 3)} and sends the agent back to the start. Both agents explore with ε = ${q(inp.eps, 2)} and learn with α = ${q(inp.alpha, 2)}.`, { phase: 'start', step: -1, fall: -100, eps: inp.eps, alpha: inp.alpha });
      const ua = A.trace.find((t) => t.kind === 'update'), ub = B.trace.find((t) => t.kind === 'update');
      F.push([...boards(zero, zero), ...ax.items, ...lg], `Their first updates. Q-learning: from ${F.cell(env, ua.s)} going ${P.ACT[ua.a]}, reward ${q.sg(ua.r, 3)}, target r + γ · max Q(s′) = ${q.sg(ua.target, 3)}, so Q moves from ${q.sg(ua.old, 3)} to ${q.sg(ua.now, 3)}. SARSA: from ${F.cell(env, ub.s)} going ${P.ACT[ub.a]}, reward ${q.sg(ub.r, 3)}, target r + γ · Q(s′, next action) = ${q.sg(ub.target, 3)}, so Q moves to ${q.sg(ub.now, 3)}.`, { phase: 'update', a: [ua.r, ua.target, ua.old, ua.now], b: [ub.r, ub.target, ub.old, ub.now], gamma: g, alpha: inp.alpha });
      const set = [10, 20, 50, 100, 200, 300, 500, 750, 1000].filter((e) => e < inp.episodes).concat([inp.episodes]);
      for (const e of set) {
        const qa = A.snapshots.find((x) => x.episode === e), qb = B.snapshots.find((x) => x.episode === e);
        if (!qa || !qb) continue;
        F.set('ep', e);
        const pa = S.RL.greedyPath(env, qa.policy, 100), pb = S.RL.greedyPath(env, qb.policy, 100);
        const items = [...boards(qa.Q, qb.Q, pa, pb), ...ax.items, ...lg, C().line(ax, 'cr:q', sa.slice(0, e).map((v, j) => [j + 1, v]), 'p15q'), C().line(ax, 'cr:s', sb.slice(0, e).map((v, j) => [j + 1, v]), 'p15s'), C().vline(ax, 'cr:now', e, 'act')];
        const desc = (p) => (p.reached ? `${q.n(p.steps)} moves` : 'no path yet');
        F.push(items, `After ${q.n(e)} episodes, the greedy routes: Q-learning ${desc(pa)}, SARSA ${desc(pb)}. Over the last ${q.n(Math.min(20, e))} episodes, while exploring, Q-learning averaged ${q.sg(sa[e - 1], 3)} per episode and SARSA ${q.sg(sb[e - 1], 3)}.`, { phase: e === inp.episodes ? 'end' : 'snap', e, window: Math.min(20, e), stepsA: pa.reached ? pa.steps : -1, stepsB: pb.reached ? pb.steps : -1, avgA: sa[e - 1], avgB: sb[e - 1], polA: qa.policy, polB: qb.policy, retA: A.returns.slice(0, e), retB: B.returns.slice(0, e) });
      }
      const m = F.last().meta;
      let verdict;
      if (m.stepsA > 0 && m.stepsB > 0 && m.stepsA < m.stepsB) verdict = `Q-learning learns the shortest route, right along the edge (${q.n(m.stepsA)} moves), the best when nothing random happens. SARSA learns the values of the policy it really follows, random exploring steps included, and keeps a safer distance (${q.n(m.stepsB)} moves).`;
      else if (m.stepsA > 0 && m.stepsB > 0) verdict = `This time both greedy routes have ${q.n(m.stepsA)} and ${q.n(m.stepsB)} moves: with so little exploration, SARSA’s values hardly count the risk of a random step.`;
      else verdict = `Q-learning’s greedy route ` + (m.stepsA > 0 ? `has ${q.n(m.stepsA)} moves` : 'does not reach the goal') + `, while SARSA’s greedy arrows ` + (m.stepsB > 0 ? `take ${q.n(m.stepsB)} moves` : 'still loop near the start: with a large step α its values are noisy, and two actions there are almost tied') + '.';
      F.push(F.last().items, verdict + ` While still exploring, SARSA’s episodes average ${q.sg(m.avgB, 3)} against Q-learning’s ${q.sg(m.avgA, 3)}: ` + (m.avgB > m.avgA ? 'the safer path pays while random steps go on.' : 'here the two are close.'), Object.assign({}, m, { phase: 'verdict' }));
      for (const f of F.list) f.narrowView = { x: 20, y: 20, w: 490, h: 420 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Two targets', text: ['Q-learning: Q(s, a) ← Q + α [r + γ max over a′ of Q(s′, a′) − Q]', 'SARSA: Q(s, a) ← Q + α [r + γ Q(s′, a′) − Q],  a′ the action it will take', 'γ = ' + g + ',  α = ' + inp.alpha + ',  ε = ' + inp.eps].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Which agent will learn the route closer to the cliff edge?', options: ['Q-learning', 'SARSA', 'the same route'], answer: (fr) => { const m = fr[fr.length - 1].meta; return m.stepsA === m.stepsB ? 'the same route' : m.stepsA < m.stepsB ? 'Q-learning' : 'SARSA'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'r = −1, γ = 1, next state’s action values are −5, −3, −8, −9, and the agent will actually take the −8 action. What are the two targets?', options: ['Q-learning −4, SARSA −9', 'Q-learning −9, SARSA −4', 'both −4', 'both −9'], answer: 0, why: 'Q-learning: −1 + max = −1 − 3; SARSA: −1 + (−8).' },
      { q: 'Which one is “off-policy”?', options: ['Q-learning: it learns the greedy policy while following an ε-greedy one', 'SARSA', 'both', 'neither'], answer: 0, why: 'Its target does not depend on the action actually taken next.' },
      { q: 'Why does SARSA keep away from the cliff edge?', options: ['its values include the chance that an exploring step falls off', 'it cannot see the cliff', 'its learning rate is lower', 'it uses Monte Carlo returns'], answer: 0, why: 'Next to the edge, the ε-greedy policy it evaluates sometimes falls.' },
      { q: 'If ε shrank to 0 over training, what would SARSA’s route become?', options: ['the shortest route, like Q-learning’s', 'stay the safe route', 'no route', 'a random route'], answer: 0, why: 'With no random steps, the safe detour is no longer worth its cost.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'rl-explore', family: 'p15-tabular', order: 30, title: 'Exploration and ε',
    blurb: 'An agent learns only about what it tries. With no exploration it repeats the first route that worked, or never finds one; with too much, it keeps wandering after it knows better. Here Q-learning learns to cross two rooms, and the colour of each cell is how often it was visited. Decaying ε, starting high and shrinking every episode, explores early and exploits later.',
    inputs: [
      { id: 'eps', kind: 'slider', label: 'ε at the start', def: 0.3, min: 0, max: 1, step: 0.05, fmt: eps2 },
      { id: 'decay', kind: 'choice', label: 'Decay', def: 'none', options: [{ value: 'none', label: 'keep ε fixed' }, { value: '0.98', label: 'ε × 0.98 every episode' }, { value: '0.95', label: 'ε × 0.95 every episode' }] },
      { id: 'episodes', kind: 'number', label: 'Episodes', def: 300, min: 50, max: 800 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 1, min: 1, max: 999 },
    ],
    presets: [
      { label: 'ε = 0.3 fixed', values: { eps: 0.3, decay: 'none', episodes: 300, seed: 1 } },
      { label: 'No exploration', values: { eps: 0, decay: 'none', episodes: 300, seed: 1 } },
      { label: 'Wander: ε = 0.8', values: { eps: 0.8, decay: 'none', episodes: 300, seed: 1 } },
      { label: 'Start at 1, decay', values: { eps: 1, decay: '0.98', episodes: 300, seed: 1 } },
    ],
    counters: [{ id: 'ep', label: 'episodes' }],
    run(inp) {
      const env = P.env('rooms'), F = P.frames(['ep']), q = F.q, c = 44, opts = (e) => ({ algo: 'q', episodes: e, eps: inp.eps, epsDecay: inp.decay === 'none' ? 0 : +inp.decay, epsMin: 0.01, alpha: 0.5, seed: inp.seed, snapEvery: 10, maxSteps: 200 });
      const full = S.RL.tabular(env, opts(inp.episodes)), sm = P.smooth(full.returns, 20);
      const lo = Math.min(...sm), hi = Math.max(...sm, 1), bx = 60 + env.w * c + 40;
      const ax = C().axes({ prefix: 'xr', x: bx, y: 60, w: 210, h: 140, xr: [0, inp.episodes], yr: [Math.min(lo, -1), hi], xticks: C().ticks(0, inp.episodes, 3), yticks: C().ticks(Math.min(lo, -1), hi, 3), xname: 'episode', yname: 'return (average of 20)' });
      const ae = C().axes({ prefix: 'xe', x: bx, y: 270, w: 210, h: 90, xr: [0, inp.episodes], yr: [0, 1], xticks: C().ticks(0, inp.episodes, 3), yticks: [0, 0.5, 1], xname: 'episode', yname: 'ε' });
      const epsAt = (e) => (inp.decay === 'none' ? inp.eps : Math.max(0.01, inp.eps * Math.pow(+inp.decay, e)));
      const epsCurve = [...Array(inp.episodes + 1).keys()].map((e) => [e, epsAt(e)]);
      const set = [10, 25, 50, 100, 200, 300, 500, 800].filter((e) => e < inp.episodes).concat([inp.episodes]);
      for (const e of set) {
        const r = e === inp.episodes ? full : S.RL.tabular(env, opts(e)), vmax = Math.max(1, ...r.visits), lv = r.visits.map((v) => Math.log(1 + v) / Math.log(1 + vmax));
        F.set('ep', e);
        const seen = env.states.filter((s) => r.visits[s] > 0).length, path = r.path;
        const items = [...P.board({ prefix: 'xb', env, x: 60, y: 50, cell: c, V: lv.map((v, s) => (env.wall[s] ? 0 : v)), vmax: 1, values: false, startLabel: !path.reached }), ...ax.items, ...ae.items, C().line(ax, 'xr:l', sm.slice(0, e).map((v, j) => [j + 1, v]), 'p15m'), C().vline(ax, 'xr:now', e, 'act'), C().line(ae, 'xe:l', epsCurve, 'p15eps'), C().vline(ae, 'xe:now', e, 'act')];
        if (path.reached) items.push(P.path('xb:p', env, path.path, 60, 50, c));
        items.push(H().text('xb:h', 60, 36, 'Visits (darker: more often)', 'strong', 14));
        F.push(items, `After ${q.n(e)} episodes (ε now ${q(epsAt(e), 2)}): ${q.n(seen)} of the ${q.n(env.states.length)} open cells have been visited, ${q.n(r.visits.reduce((a, b) => a + b, 0))} visits in all. The average return over the last ${q.n(Math.min(20, e))} episodes is ${q.sg(sm[e - 1], 3)}, and the greedy path ` + (path.reached ? `reaches the goal in ${q.n(path.steps)} moves.` : 'does not reach the goal.'), { phase: e === inp.episodes ? 'end' : 'snap', e, eps: epsAt(e), seen, open: env.states.length, visits: r.visits.reduce((a, b) => a + b, 0), lengths: r.lengths, avg: sm[e - 1], window: Math.min(20, e), reached: path.reached, steps: path.steps, returns: full.returns.slice(0, e) });
      }
      for (const f of F.list) f.narrowView = { x: 40, y: 20, w: env.w * c + 40, h: env.h * c + 50 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'ε-greedy', text: ['with probability ε: a random action', 'otherwise: argmax over a of Q(s, a)', inp.decay === 'none' ? 'ε fixed at ' + inp.eps : 'after each episode: ε ← max(0.01, ' + inp.decay + ' ε)', 'Q-learning, α = 0.5, γ = ' + env.gamma].join('\n') }] } };
    },
    predict: [
      { at: (f, i) => i === 0, pick: 'option', ask: 'Will the greedy path reach the goal at the end?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.reached ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'With ε = 0 and all Q-values starting at 0, what does a Q-learning agent do at first?', options: ['breaks ties at random, then sticks with whatever first looks best', 'explores everything evenly', 'goes straight to the goal', 'nothing'], answer: 0, why: 'Greedy choice with no exploration can lock in early.' },
      { q: 'Why can a large fixed ε hurt the return even late in training?', options: ['the agent keeps taking random moves after it has learned better', 'it forgets Q', 'it learns slower forever', 'it stops visiting cells'], answer: 0, why: 'The return measured while acting includes the cost of exploring.' },
      { q: 'ε = 1 multiplied by 0.98 each episode. About what is it after 100 episodes?', options: ['0.13', '0.98', '0.5', '0.02'], answer: 0, why: '0.98¹⁰⁰ ≈ 0.133.' },
      { q: 'Another way to explore, besides random moves?', options: ['optimistic initial values, which make untried actions look good', 'a smaller γ', 'a larger α', 'fewer episodes'], answer: 0, why: 'High starting Q-values lure a greedy agent to try everything once.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
