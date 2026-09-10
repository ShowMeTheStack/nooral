// Part 15, deep reinforcement learning: REINFORCE balancing a pole (the policy's weights kept at chosen updates of the
// engine's training run; the pole is replayed live with them in the engine's cart-pole), and a small DQN with a
// replay buffer and a target network on a grid (its Q-table and replayed minibatches kept from the engine's run).
(function (S) {
  'use strict';
  const P = S.P15, H = () => S.Kit.ml, C = () => S.Kit.curve;

  /** a trainer built as in the stored run, so its environment and networks are the same; weights set per snapshot */
  P.rf = () => P.once('rf', () => S.RL.reinforce(P.D().reinforce.o));
  /** the greedy episode of a policy from a start drawn with seed (as the engine's own snapshots draw it) */
  P.greedyEpisode = (weights, seed) => {
    const tr = P.rf(), env = tr.env;
    S.Gen.setParams(tr.policy, weights.map((w) => Float32Array.from(w)));
    const r = S.T.rng(seed), states = [], actions = [], probs = [];
    let s = env.reset(r), fell = false;
    for (;;) {
      const p = tr.probs(s), a = p[1] > p[0] ? 1 : 0;
      states.push(s); actions.push(a); probs.push(p);
      const out = env.step(a); s = out.s;
      if (out.done) { fell = !!out.fell; states.push(s); break; }
    }
    return { states, actions, probs, len: actions.length, fell, max: env.maxSteps };
  };
  /** the cart and its pole at state st, drawn on a track of half-width xl metres */
  P.cart = (pre, st, x0, y0, w, xl, o) => {
    o = o || {};
    const sc = w / (2 * xl), cx = x0 + w / 2 + st[0] * sc, L = 120, th = st[2], items = [];
    if (o.ghost) items.push({ key: pre + ':cart', kind: 'cell', x: H().r2(cx - 26), y: y0 - 26, w: 52, h: 22, rx: 4, text: '', cls: 'p15cart ghost', layer: 'lines', audit: 'ignore' });
    if (!o.ghost) {
      items.push({ key: pre + ':track', kind: 'path', d: `M${x0},${y0} h${w}`, cls: 'p15track', audit: 'ignore', layer: 'back', box: { x: x0, y: y0, w, h: 1 } });
      for (const sx of [-1, 1]) items.push({ key: pre + ':lim' + sx, kind: 'path', d: `M${H().r2(x0 + w / 2 + sx * xl * sc)},${y0 - 90} V${y0 + 8}`, cls: 'p15limit', audit: 'ignore', layer: 'back', box: { x: x0 + w / 2 + sx * xl * sc, y: y0 - 90, w: 1, h: 98 } });
      items.push({ key: pre + ':cart', kind: 'cell', x: H().r2(cx - 26), y: y0 - 26, w: 52, h: 22, rx: 4, text: '', cls: 'p15cart', layer: 'items' });
    }
    const hx = cx, hy = y0 - 26, tx = hx + L * Math.sin(th), ty = hy - L * Math.cos(th);
    items.push({ key: pre + ':pole', kind: 'path', d: `M${H().r2(hx)},${H().r2(hy)} L${H().r2(tx)},${H().r2(ty)}`, cls: 'p15pole' + (o.ghost ? ' ghost' : '') + (o.fell ? ' fell' : ''), layer: o.ghost ? 'lines' : 'heads', audit: 'ignore', box: H().boxOf([[hx, hy], [tx, ty]]) });
    if (!o.ghost) items.push({ key: pre + ':hinge', kind: 'dot', x: H().r2(hx), y: H().r2(hy), r: 4.5, cls: 'p15hinge', layer: 'heads', audit: 'overlap-ok' });
    return items;
  };
  const deg = (r) => (r * 180) / Math.PI;

  S.topic({
    id: 'rl-reinforce', family: 'p15-deep', order: 10, title: 'REINFORCE: a policy that balances a pole',
    blurb: 'A pole stands on a cart that can be pushed left or right; the episode ends when the pole leans past 12° or the cart leaves the track, and every step it stays up earns 1. A small network reads the four numbers of the state (position, velocity, angle, angular velocity) and gives the probability of each push. REINFORCE plays some episodes, then raises the log-probability of each action in proportion to how much better than expected the rest of its episode went.',
    inputs: [{ id: 'seed', kind: 'number', label: 'Seed of the starting state', def: 99, min: 1, max: 999 }],
    presets: [{ label: 'The usual start', values: { seed: 99 } }, { label: 'Another start', values: { seed: 7 } }, { label: 'A third start', values: { seed: 314 } }],
    counters: [{ id: 'upd', label: 'updates' }],
    run(inp) {
      const D = P.D().reinforce, tr = P.rf(), env = tr.env, ph = env.physics, F = P.frames(['upd']), q = F.q, xl = ph.xLimit, lim = deg(ph.thetaLimit);
      const ml = C().axes({ prefix: 'rm', x: 520, y: 60, w: 260, h: 120, xr: [0, D.updates], yr: [0, env.maxSteps], xticks: C().ticks(0, D.updates, 4), yticks: [0, 250, 500], xname: 'update', yname: 'episode length while training' });
      const at = C().axes({ prefix: 'ra', x: 60, y: 360, w: 400, h: 110, xr: [0, env.maxSteps], yr: [-lim, lim], xticks: [0, 100, 200, 300, 400, 500], yticks: [-12, 0, 12], xname: 'step of the episode', yname: 'pole angle (°)' });
      const ax = C().axes({ prefix: 'rx', x: 520, y: 260, w: 260, h: 110, xr: [0, env.maxSteps], yr: [-xl, xl], xticks: [0, 250, 500], yticks: [-2.4, 0, 2.4], xname: 'step', yname: 'cart position (m)' });
      const cartAt = (ep, k, fell) => P.cart('cp', ep.states[Math.min(k, ep.states.length - 1)], 60, 290, 400, xl, { fell });
      const ghosts = (ep) => [0.2, 0.4, 0.6, 0.8].map((f, i) => P.cart('cg' + i, ep.states[Math.floor(f * (ep.states.length - 1))], 60, 290, 400, xl, { ghost: true })).flat();
      const s0 = P.greedyEpisode(D.snaps[0].policy, inp.seed), st0 = s0.states[0], d = env.dynamics(st0, ph.force);
      F.push([...P.cart('cp', st0, 60, 290, 400, xl), H().arrow('push', [80, 312], [140, 312], 'p15push'), H().text('push:t', 150, 316, 'push ± ' + ph.force + ' N', 'axname', 13), ...ml.items, ...at.items, ...ax.items], `The start: the cart at ${q(st0[0], 3)} m and the pole at ${q(deg(st0[2]), 2)}°. Each step the policy pushes with ${q.n(ph.force)} N one way or the other; pushing right here would tip the pole at ${q.sg(d.thAcc, 3)} rad/s². An episode ends at ${q(lim, 0)}°, at ${q(xl, 1)} m from the centre, or after ${q.n(env.maxSteps)} steps.`, { phase: 'start', st0, x: st0[0], th: deg(st0[2]), force: ph.force, thAcc: d.thAcc, lim, xl, max: env.maxSteps });
      // the first update, spelled out by the engine on a fresh copy of the trainer
      const fresh = S.RL.reinforce(D.o), ex = Object.fromEntries(fresh.explain().map((s) => [s.kind, s]));
      const L0 = ex.episode.len, G0 = ex.return.G[0];
      F.push([...P.cart('cp', st0, 60, 290, 400, xl), ...ml.items, ...at.items, ...ax.items, ...H().lines('ex', 520, 436, [['returns G, minus a learned baseline,', 'legend'], ['scaled to unit spread: the advantage', 'legend']], 13, 20)], `The first update plays ${q.n(D.o.perUpdate || 4)} episodes with the untrained policy. The first lasted ${q.n(L0)} steps, so with γ = ${q(0.99, 2)} its return from the start is ${q.sg(G0, 4)}. The baseline network predicted ${q.sg(ex.baseline.baseline, 3)}; after centring and scaling, that first action’s advantage is ${q.sg(ex.advantage.adv, 3)}, so its probability is pushed ${ex.advantage.adv > 0 ? 'up' : 'down'}. The policy loss is ${q.sg(ex.loss.loss, 3)}.`, { phase: 'update', episodes: D.o.perUpdate || 4, L0, G0, gamma: 0.99, base: ex.baseline.baseline, adv: ex.advantage.adv, loss: ex.loss.loss });
      for (const sn of D.snaps) {
        F.set('upd', sn.update);
        const ep = P.greedyEpisode(sn.policy, inp.seed), ths = ep.states.map((s) => deg(s[2])), xs = ep.states.map((s) => s[0]), mx = Math.max(...ths.map(Math.abs));
        const items = [...ghosts(ep), ...cartAt(ep, ep.states.length - 1, ep.fell), ...ml.items, ...at.items, ...ax.items, C().line(at, 'ra:l', ths.map((v, k) => [k, v]), 'p15th'), C().line(ax, 'rx:l', xs.map((v, k) => [k, v]), 'p15x')];
        if (sn.update) items.push(C().line(ml, 'rm:l', D.meanLen.slice(0, sn.update).map((v, k) => [k + 1, v]), 'p15len'), C().vline(ml, 'rm:now', sn.update, 'act'));
        const p0 = ep.probs[0];
        const say = (sn.update ? `After ${q.n(sn.update)} updates (${q.n(sn.episodes)} episodes; the last training batch lasted ${q.sg(D.meanLen[sn.update - 1], 3)} steps on average)` : 'Before any update')
          + `, the greedy policy (always the likelier push) ` + (ep.fell ? `keeps the pole up for ${q.n(ep.len)} steps before it falls` : `balances the pole for all ${q.n(ep.len)} steps`) + `; the largest lean was ${q(mx, 1)}°. At the start it pushes right with probability ${q(p0[1], 2)}.`;
        F.push(items, say, { phase: sn.update === D.updates ? 'end' : 'snap', update: sn.update, episodes: sn.episodes, mean: sn.update ? D.meanLen[sn.update - 1] : null, len: ep.len, fell: ep.fell, maxTh: mx, p: p0[1] });
      }
      for (const f of F.list) f.narrowView = { x: 30, y: 150, w: 460, h: 390 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'REINFORCE with a baseline', text: ['π(a | s): 4 → 32 (tanh) → 2 (softmax)', 'G_t = Σ γᵏ r_{t+k},  γ = 0.99', 'A_t = (G_t − b(s_t)), centred and scaled', 'loss = − mean of log π(a_t | s_t) · A_t', 'the baseline b(s) is fitted to G by squared error'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'update', pick: 'option', ask: 'After how many updates will the greedy policy first keep the pole up for the whole episode?', options: ['10 or fewer', '11–25', 'more than 25', 'never in this run'], answer: (fr) => { const f = fr.find((x) => (x.meta.phase === 'snap' || x.meta.phase === 'end') && x.meta.fell === false); if (!f) return 'never in this run'; const u = f.meta.update; return u <= 10 ? '10 or fewer' : u <= 25 ? '11–25' : 'more than 25'; }, explain: () => 'Follow the episode lengths frame by frame: the greedy policy holds the pole once its pushes counter each lean in time.' },
    ],
    quiz: [
      { q: 'Every step earns 1 and γ = 0.99. What is the return of a 3-step episode from its start?', options: ['1 + 0.99 + 0.9801 = 2.9701', '3', '2.97', '0.9801'], answer: 0, why: 'G = Σ γᵏ over the three rewards.' },
      { q: 'Why subtract a baseline from the return?', options: ['it removes what was expected anyway, so the gradient is less noisy; the expected gradient is unchanged', 'to make returns negative', 'to speed up the cart', 'it changes what is optimal'], answer: 0, why: 'Only how much better or worse than usual an action did should move it.' },
      { q: 'The advantage of an action is negative. What does the update do?', options: ['makes that action less likely in that state', 'makes it more likely', 'removes it', 'nothing'], answer: 0, why: 'The loss is −log π · A, so a negative A lowers log π.' },
      { q: 'Why is REINFORCE called a Monte Carlo method?', options: ['it waits for whole episodes and uses their actual returns', 'it uses a model of the physics', 'it samples the learning rate', 'it uses a Q-table'], answer: 0, why: 'Like Monte Carlo control, it learns from complete returns.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'rl-dqn', family: 'p15-deep', order: 20, title: 'A small deep Q-network with replay',
    blurb: 'A deep Q-network replaces the Q-table by a network: here the state goes in as 20 numbers (a 1 for the agent’s cell, 0 elsewhere) and the four action values come out. Two tricks make it learn steadily. Experience replay stores every transition and trains on random minibatches of old ones, so consecutive, similar steps do not dominate; a target network, a copy refreshed every 100 steps, provides the bootstrap target r + γ max Q_target(s′), so the target does not move with every update.',
    inputs: [{ id: 'batch', kind: 'choice', label: 'Minibatch to inspect', def: '1000', options: [{ value: '300', label: 'at step 300 (early)' }, { value: '1000', label: 'at step 1000' }, { value: '2000', label: 'at step 2000 (the end)' }] }],
    presets: [{ label: 'A minibatch at step 1000', values: { batch: '1000' } }, { label: 'An early minibatch', values: { batch: '300' } }, { label: 'The last minibatch', values: { batch: '2000' } }],
    counters: [{ id: 'step', label: 'environment steps' }],
    run(inp) {
      const D = P.D().dqn, env = P.env('pits'), F = P.frames(['step']), q = F.q, c = 70, g = D.gamma, N = D.o.steps, epsSteps = Math.round(N * 0.6);
      const al = C().axes({ prefix: 'dl', x: 460, y: 60, w: 300, h: 70, xr: [0, N], yr: [0, 1], xticks: C().ticks(0, N, 4), yticks: [0, 1], yname: 'ε' });
      const rlo = Math.min(-1, ...D.returns.map((r) => r.ret)), ar = C().axes({ prefix: 'dr', x: 460, y: 190, w: 300, h: 110, xr: [0, N], yr: [rlo, 1], xticks: C().ticks(0, N, 4), yticks: C().ticks(rlo, 1, 3), xname: 'environment step', yname: 'return of each episode' });
      const curves = (t) => [...ar.items, C().line(al, 'dl:e', D.hist.filter((h) => h.step <= t).map((h) => [h.step, h.eps]), 'p15eps'), C().vline(al, 'dl:now', t, 'act'), C().vline(ar, 'dr:now', t, 'act'), ...D.returns.filter((r) => r.t <= t).map((r, k) => ({ key: 'dr:p' + k, kind: 'dot', x: ar.X(r.t), y: ar.Y(r.ret), r: 2.6, cls: 'p15ret', audit: 'overlap-ok', layer: 'items' }))];
      const board = (Q, o) => P.qboard('dq', env, 40, 60, c, Q, o || {});
      const zeroQ = D.snaps[0].Q;
      F.push([...board(zeroQ), ...al.items, ...ar.items, ...H().lines('net', 40, 372, [['one-hot state (20) → 64 ReLU → 4 Q-values', 'legend'], ['acts ε-greedily; ε falls from 1 to 0.05', 'legend'], ['replay buffer: up to 5000 transitions', 'legend'], ['target network: copied every 100 steps', 'legend']], 13, 24)], `The untrained network already gives every state ${q.n(4)} action values, small and arbitrary. The agent acts ε-greedily, with ε falling from ${q.n(1)} to ${q(0.05, 2)} over the first ${q.n(epsSteps)} steps; every transition (s, a, r, s′) goes into the replay buffer.`, { phase: 'start', four: 4, one: 1, epsEnd: 0.05, epsSteps });
      const B = D.batches.find((b) => String(b.step) === inp.batch), snapAt = (t) => D.snaps.find((s) => s.step === t);
      const table = (b) => {
        const rows = b.rows.map((r) => { const mq = r.done ? 0 : Math.max(...r.qt), y = r.r + g * mq; return [P.cellName(env, r.s), P.ACT[r.a], H().sg(r.r, 2), r.done ? 'end' : P.cellName(env, r.s2), H().sg(y, 3), H().sg(r.q[r.a], 3), H().sg(y - r.q[r.a], 3)]; });
        return S.Kit.dt.table({ prefix: 'tb', x: 40, y: 360, cols: [{ name: 's', w: 66 }, { name: 'a', w: 64 }, { name: 'r', w: 60 }, { name: 's′', w: 66 }, { name: 'target y', w: 86 }, { name: 'Q(s, a)', w: 86 }, { name: 'y − Q', w: 80 }], rows, rowH: 24, size: 12 }).items;
      };
      const showSteps = [100, 300, 600, 1000, 1600, 2000];
      for (const t of showSteps) {
        const sn = snapAt(t); if (!sn) continue;
        F.set('step', t);
        const path = S.RL.greedyPath(env, S.RL.greedy(sn.Q), 30), items = [...board(sn.Q, path.reached ? { path: path.path } : {}), ...al.items, ...curves(t)];
        if (path.reached) items.push(P.path('dq:p', env, path.path, 40, 60, c));
        F.push(items, `Step ${q.n(t)}: ε = ${q(sn.eps, 2)}, ${q.n(sn.episodes)} episodes have ended, the buffer holds ${q.n(sn.buffer)} transitions and the target network has been refreshed ${q.n(sn.syncs)} ${sn.syncs === 1 ? 'time' : 'times'}. ` + (path.reached ? `The greedy arrows now lead from the start to the goal in ${q.n(path.steps)} moves.` : 'The greedy arrows do not yet lead to the goal.'), { phase: 'snap', t, eps: sn.eps, episodes: sn.episodes, buffer: sn.buffer, syncs: sn.syncs, reached: path.reached, steps: path.steps, Q: sn.Q });
        if (B && t === B.step) {
          const r0 = B.rows[0], mq = r0.done ? 0 : Math.max(...r0.qt), y0 = r0.r + g * mq;
          F.push([...items, ...table(B)], `A minibatch replayed at step ${q.n(B.step)}: ${q.n(B.rows.length)} stored transitions drawn at random. For the first, from ${F.cell(env, r0.s)} going ${P.ACT[r0.a]} with reward ${q.sg(r0.r, 2)}, the target network gives ` + (r0.done ? `nothing to add, since the episode ended there, so y = ${q.sg(y0, 3)}` : `max Q_target(s′) = ${q.sg(mq, 3)}, so y = ${q.sg(r0.r, 2)} + ${q(g, 2)} × ${q.sg(mq, 3)} = ${q.sg(y0, 3)}`) + `. The online network says Q(s, a) = ${q.sg(r0.q[r0.a], 3)}; the Huber loss pulls it towards y.`, { phase: 'batch', t: B.step, rows: B.rows.length, r: r0.r, mq, y: y0, qsa: r0.q[r0.a], gamma: g });
        }
      }
      const last = D.snaps[D.snaps.length - 1], path = S.RL.greedyPath(env, S.RL.greedy(last.Q), 30), lastRet = D.returns.slice(-10), avg = P.mean(lastRet.map((r) => r.ret));
      F.push([...board(last.Q, path.reached ? { path: path.path } : {}), ...al.items, ...curves(N)].concat(path.reached ? [P.path('dq:p', env, path.path, 40, 60, c)] : []), `After ${q.n(N)} steps, ${q.n(D.returns.length)} episodes. The last ${q.n(lastRet.length)} returned ${q.sg(avg, 3)} on average, and the greedy path ` + (path.reached ? `reaches the goal in ${q.n(path.steps)} moves with return ${q.sg(path.ret, 3)}.` : 'does not reach the goal.'), { phase: 'end', N, episodes: D.returns.length, n: lastRet.length, avg, reached: path.reached, steps: path.steps, ret: path.ret, rets: lastRet.map((r) => r.ret) });
      for (const f of F.list) if (f.meta.phase !== 'batch') f.narrowView = { x: 20, y: 40, w: 390, h: 320 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'DQN', text: ['y = r + γ · max over a′ of Q_target(s′, a′)   (y = r at the end)', 'loss = Huber(Q(s, a) − y) over a random minibatch of 32', 'Q_target ← Q every 100 steps', 'γ = ' + g + ', ε from 1 to 0.05 over ' + epsSteps + ' steps'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'By which step will the greedy arrows first lead to the goal?', options: ['300 or earlier', '600 or 1000', '1600 or 2000', 'not at all'], answer: (fr) => { const f = fr.find((x) => x.meta.phase === 'snap' && x.meta.reached); if (!f) return 'not at all'; const t = f.meta.t; return t <= 300 ? '300 or earlier' : t <= 1000 ? '600 or 1000' : '1600 or 2000'; }, explain: () => 'Follow the arrows snapshot by snapshot: the route appears once the goal’s value has spread back to the start.' },
    ],
    quiz: [
      { q: 'r = −0.04, γ = 0.95 and the target network’s values at s′ are 0.2, 0.6, 0.1 and 0.4. What is y?', options: ['−0.04 + 0.95 × 0.6 = 0.53', '0.6', '0.56', '0.34'], answer: 0, why: 'Take the largest target value at s′.' },
      { q: 'Why train on random minibatches from a replay buffer?', options: ['consecutive steps are alike; mixing old transitions makes the batches more independent and reuses data', 'it makes the network smaller', 'it removes the need for rewards', 'to act faster'], answer: 0, why: 'Correlated batches make gradient steps lurch in one direction.' },
      { q: 'What would go wrong without a target network?', options: ['each update moves the very target it is chasing, which can oscillate or diverge', 'nothing at all', 'the buffer fills up', 'ε would not fall'], answer: 0, why: 'The frozen copy holds the target still for a while.' },
      { q: 'Why feed the state as one-hot numbers on this grid?', options: ['the cell number itself has no meaning as a size; one-hot treats cells as separate', 'it is required by ReLU', 'to save memory', 'it makes the network linear'], answer: 0, why: 'With one-hot inputs the network can give each cell its own values.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
