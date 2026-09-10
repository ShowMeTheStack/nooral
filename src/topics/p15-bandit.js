// Part 15, bandits: one run of ε-greedy, UCB1 or Thompson sampling pull by pull (S.RL.runBandit), and the regret of
// all of them averaged over many bandits (S.RL.compareBandits). Everything runs live.
(function (S) {
  'use strict';
  const P = S.P15, H = () => S.Kit.ml, C = () => S.Kit.curve;
  const ALGOS = [{ value: 'epsilon', label: 'ε-greedy' }, { value: 'greedy', label: 'greedy (ε = 0)' }, { value: 'ucb', label: 'UCB1' }, { value: 'thompson', label: 'Thompson sampling' }];
  const NAME = { epsilon: 'ε-greedy', greedy: 'greedy', ucb: 'UCB1', thompson: 'Thompson sampling' };

  /** the arms: a bar for each estimate, a dashed mark at each true mean, the pull counts under them */
  P.arms = (pre, x, y, w, h, bandit, est, counts, o) => {
    o = o || {};
    const k = bandit.k, cw = w / k, items = [], ax = C().axes({ prefix: pre, x, y, w, h, xr: [0, k], yr: [0, 1], xticks: 0, yticks: [0, 0.5, 1], yname: 'payout probability', grid: false });
    items.push(...ax.items);
    for (let a = 0; a < k; a++) {
      const cx = x + a * cw, bw = cw * 0.56, bx = cx + (cw - bw) / 2, e = Math.max(0, Math.min(1, est[a])), lit = o.pick === a;
      if (e > 0.003) items.push({ key: pre + ':b' + a, kind: 'cell', x: H().r2(bx), y: ax.Y(e), w: H().r2(bw), h: H().r2(ax.Y(0) - ax.Y(e)), rx: 2, text: '', cls: 'p15est' + (lit ? ' act' : ''), audit: 'ignore' });
      items.push({ key: pre + ':m' + a, kind: 'path', d: `M${H().r2(cx + cw * 0.1)},${ax.Y(bandit.means[a])} h${H().r2(cw * 0.8)}`, cls: 'p15true' + (a === bandit.best ? ' best' : ''), audit: 'ignore', layer: 'lines' });
      if (o.ucb) { const u = Math.min(1.25, o.ucb[a]); items.push({ key: pre + ':u' + a, kind: 'path', d: `M${H().r2(cx + cw / 2)},${ax.Y(e)} V${ax.Y(Math.min(1, u))}`, cls: 'p15ucb', audit: 'ignore', layer: 'lines' }, { key: pre + ':uc' + a, kind: 'path', d: `M${H().r2(cx + cw / 2 - 6)},${ax.Y(Math.min(1, u))} h12`, cls: 'p15ucb', audit: 'ignore', layer: 'lines' }); }
      if (o.draws) items.push({ key: pre + ':d' + a, kind: 'dot', x: H().r2(cx + cw / 2), y: ax.Y(Math.max(0, Math.min(1, o.draws[a]))), r: 5, cls: 'p15draw' + (lit ? ' act' : ''), layer: 'heads', audit: 'overlap-ok' });
      items.push(H().text(pre + ':n' + a, cx + cw / 2, y + h + 18, 'arm ' + a, 'axname' + (lit ? ' strong' : ''), 13, 'middle'), H().text(pre + ':c' + a, cx + cw / 2, y + h + 36, counts[a] + ' pulls', 'tick', 12, 'middle'));
    }
    return items;
  };

  S.topic({
    id: 'rl-bandit', family: 'p15-bandits', order: 10, title: 'A multi-armed bandit, pull by pull',
    blurb: 'Several slot machines (arms) pay out 1 with different, unknown probabilities. Every pull earns a reward and teaches the agent a little about one arm. Pull only the arm that looks best and you may never learn that another is better; explore too much and you waste pulls. Each strategy here keeps an estimate per arm (the bars); the dashed marks are the true probabilities, which the agent never sees.',
    inputs: [
      { id: 'algo', kind: 'choice', label: 'Strategy', def: 'epsilon', options: ALGOS },
      { id: 'eps', kind: 'slider', label: 'ε (ε-greedy)', def: 0.1, min: 0.01, max: 0.5, step: 0.01, fmt: (v) => String(v) },
      { id: 'c', kind: 'slider', label: 'c (UCB1)', def: 1, min: 0.2, max: 3, step: 0.1, fmt: (v) => String(v) },
      { id: 'k', kind: 'number', label: 'Arms', def: 5, min: 3, max: 8 },
      { id: 'steps', kind: 'number', label: 'Pulls', def: 500, min: 50, max: 2000 },
      { id: 'seed', kind: 'number', label: 'Seed', def: 3, min: 1, max: 999 },
    ],
    presets: [
      { label: 'ε-greedy, ε = 0.1', values: { algo: 'epsilon', eps: 0.1, k: 5, steps: 500, seed: 3 } },
      { label: 'Greedy gets stuck', values: { algo: 'greedy', k: 5, steps: 500, seed: 3 } },
      { label: 'UCB1', values: { algo: 'ucb', c: 1, k: 5, steps: 500, seed: 3 } },
      { label: 'Thompson sampling', values: { algo: 'thompson', k: 5, steps: 500, seed: 3 } },
    ],
    counters: [{ id: 't', label: 'pulls' }],
    run(inp) {
      const RL = S.RL, b = RL.bandit({ k: inp.k, seed: inp.seed }), res = RL.runBandit(b, { algo: inp.algo, eps: inp.eps, c: inp.c, steps: inp.steps, seed: inp.seed * 7 + 1, traceSteps: 10, snapEvery: 25 });
      const F = P.frames(['t']), q = F.q, N = inp.steps, name = NAME[inp.algo];
      const rmax = Math.max(1e-6, res.regret[N - 1]), ar = C().axes({ prefix: 'br', x: 520, y: 60, w: 260, h: 120, xr: [0, N], yr: [0, rmax * 1.05], xticks: P.ticks(N), yticks: C().ticks(0, rmax, 3), yname: 'regret so far' });
      const opt = res.optimal.reduce((acc, v, i) => { acc.push(((acc[i - 1] || 0) * i + v) / (i + 1)); return acc; }, []);
      const ao = C().axes({ prefix: 'bo', x: 520, y: 250, w: 260, h: 110, xr: [0, N], yr: [0, 1], xticks: P.ticks(N), yticks: [0, 0.5, 1], xname: 'pull', yname: 'share of pulls on the best arm' });
      const curves = (t) => (t ? [C().line(ar, 'br:l', res.regret.slice(0, t).map((v, i) => [i + 1, v]), 'p15reg'), C().line(ao, 'bo:l', opt.slice(0, t).map((v, i) => [i + 1, v]), 'p15opt'), C().vline(ar, 'br:now', t, 'act'), C().vline(ao, 'bo:now', t, 'act')] : []);
      const hist = (t) => ({ choices: res.choices.slice(0, t), rewards: res.rewards.slice(0, t) });
      const base = [...ar.items, ...ao.items];
      F.push([...P.arms('ba', 60, 60, 380, 260, b, new Array(b.k).fill(0), new Array(b.k).fill(0)), ...base], `${q.n(b.k)} arms pay out with probabilities ${H().list(b.means.map((m) => q(m, 2)))}: the best is arm ${q.n(b.best)}. ${name} starts knowing none of this, with every estimate at ${q.n(0)}.`, { phase: 'start', means: b.means, best: b.best, k: b.k, zero: 0 });
      let prevEst = new Array(b.k).fill(0);
      for (const st of res.trace) {
        const d = st.data || st, t = d.t, a = d.a, r = d.r, est = d.estimates, cnt = d.counts, before = prevEst;
        F.set('t', t);
        const head = `Pull ${q.n(t)}: `;
        let why;
        if (inp.algo === 'ucb') why = d.ucb ? `the UCB rule picks arm ${q.n(a)}: its estimate ${q(before[a], 2)} plus a bonus of ${q(d.ucb[a] - before[a], 2)} gives ${q(d.ucb[a], 2)}, the highest upper bound` : `arm ${q.n(a)} has not been tried yet, so it is tried first`;
        else if (inp.algo === 'thompson') why = `each arm draws a plausible payout from its Beta posterior; arm ${q.n(a)} drew the highest, ${q(d.draws[a], 2)}`;
        else { const m = Math.max(...before), top = before.map((v, i) => (Math.abs(v - m) < 1e-12 ? i : -1)).filter((i) => i >= 0); why = !top.includes(a) ? `it explores: arm ${q.n(a)} at random` : top.length > 1 ? `${q.n(top.length)} arms tie for the best estimate, ${q(m, 2)}, and it picks arm ${q.n(a)} among them` : `it exploits arm ${q.n(a)}, whose estimate ${q(m, 2)} is the highest`; }
        const say = `${head}${why}. Reward ${q.n(r)}, so arm ${q.n(a)}’s estimate moves from ${q(before[a], 2)} to ${q(est[a], 2)} after ${q.n(cnt[a])} pull${cnt[a] === 1 ? '' : 's'}. Regret so far: ${q(res.regret[t - 1], 2)}.`;
        F.push([...P.arms('ba', 60, 60, 380, 260, b, est, cnt, { pick: a, ucb: d.ucb, draws: d.draws }), ...base, ...curves(t)], say, Object.assign({ phase: 'pull', t, a, r, before, est, cnt, regret: res.regret[t - 1], ucb: d.ucb || null, bonus: d.ucb ? d.ucb[a] - before[a] : null, draws: d.draws || null, means: b.means }, hist(t)));
        prevEst = est;
      }
      for (const sn of res.snapshots.filter((x) => [50, 100, 250, 500, 1000, 2000].includes(x.t) || x.t === N)) {
        if (sn.t <= res.trace.length) continue;
        F.set('t', sn.t);
        const most = sn.counts.indexOf(Math.max(...sn.counts));
        F.push([...P.arms('ba', 60, 60, 380, 260, b, sn.estimates, sn.counts, { pick: most }), ...base, ...curves(sn.t)], `After ${q.n(sn.t)} pulls the estimates are ${H().list(sn.estimates.map((e) => q(e, 2)))}. Arm ${q.n(most)} has had the most pulls, ${q.n(sn.counts[most])}; the best arm has had ${q(opt[sn.t - 1] * 100, 0)}% of all pulls so far, and the regret is ${q(res.regret[sn.t - 1], 1)}.` + (sn.t === N ? (most === b.best ? ' It has found the best arm.' : ' It has settled on the wrong arm.') : ''), Object.assign({ phase: sn.t === N ? 'end' : 'snap', t: sn.t, est: sn.estimates, cnt: sn.counts, most, best: b.best, share: opt[sn.t - 1] * 100, regret: res.regret[sn.t - 1], means: b.means }, hist(sn.t)));
      }
      for (const f of F.list) f.narrowView = { x: 10, y: 30, w: 450, h: 380 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: name, text: [inp.algo === 'ucb' ? 'pick argmax  Q(a) + c √(2 ln t / N(a)),  c = ' + inp.c : inp.algo === 'thompson' ? 'draw θ(a) ~ Beta(1 + wins, 1 + losses); pick argmax θ(a)' : inp.algo === 'greedy' ? 'pick argmax Q(a)' : 'with probability ε = ' + inp.eps + ' a random arm, else argmax Q(a)', 'Q(a) ← Q(a) + (r − Q(a)) / N(a)', 'regret = Σ (best mean − mean of the arm pulled)'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'Will this strategy end with the best arm pulled most often?', options: ['yes', 'no'], answer: (fr) => { const m = fr[fr.length - 1].meta; return m.most === m.best ? 'yes' : 'no'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'An arm paid 1, 0, 1 and 1 in four pulls. What is its estimate?', options: ['0.75', '0.5', '1', '3'], answer: 0, why: 'The average reward: 3/4.' },
      { q: 'Why can pure greedy get stuck on a poor arm?', options: ['once another arm’s estimate falls below it, that arm is never tried again to correct it', 'it explores too much', 'its estimates are wrong by design', 'it cannot count'], answer: 0, why: 'Unlucky early rewards can bury the best arm for ever.' },
      { q: 'UCB1’s bonus √(2 ln t / N(a)) is largest for…', options: ['arms pulled rarely', 'arms pulled often', 'the best arm', 'arms that paid 0'], answer: 0, why: 'Few pulls mean a wide uncertainty, so a high optimistic bound.' },
      { q: 'What does Thompson sampling draw for each arm?', options: ['a plausible payout from its posterior, then it pulls the arm with the highest draw', 'a random arm', 'the arm’s true mean', 'the UCB bonus'], answer: 0, why: 'Arms are chosen in proportion to the probability that they are the best.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  P.ticks = (n) => C().ticks(0, n, 4);
  const LINES = [{ algo: 'greedy', cls: 'p15a0' }, { algo: 'epsilon', cls: 'p15a1' }, { algo: 'ucb', cls: 'p15a2' }, { algo: 'thompson', cls: 'p15a3' }];

  S.topic({
    id: 'rl-regret', family: 'p15-bandits', order: 20, title: 'Regret: greedy, ε-greedy, UCB and Thompson',
    blurb: 'Regret adds up what each pull lost against always pulling the best arm. A strategy that never explores can run up regret in a straight line for ever; a good one learns, and its regret curve bends over. Here the four strategies play many fresh bandits each, and the curves are their averages.',
    inputs: [
      { id: 'kind', kind: 'choice', label: 'Arms', def: 'bernoulli', options: [{ value: 'bernoulli', label: 'pay 0 or 1 (Bernoulli)' }, { value: 'gaussian', label: 'pay a normal amount (Gaussian)' }] },
      { id: 'k', kind: 'number', label: 'Arms per bandit', def: 10, min: 2, max: 20 },
      { id: 'eps', kind: 'slider', label: 'ε for ε-greedy', def: 0.1, min: 0.01, max: 0.5, step: 0.01, fmt: (v) => String(v) },
      { id: 'c', kind: 'slider', label: 'c for UCB1', def: 1, min: 0.2, max: 3, step: 0.1, fmt: (v) => String(v) },
      { id: 'runs', kind: 'number', label: 'Bandits averaged', def: 30, min: 5, max: 60 },
      { id: 'steps', kind: 'number', label: 'Pulls', def: 500, min: 100, max: 1000 },
    ],
    presets: [
      { label: '10 arms, 0 or 1', values: { kind: 'bernoulli', k: 10, eps: 0.1, c: 1, runs: 30, steps: 500 } },
      { label: 'Normal payouts', values: { kind: 'gaussian', k: 10, eps: 0.1, c: 1, runs: 30, steps: 500 } },
      { label: 'Too much exploring', values: { kind: 'bernoulli', k: 10, eps: 0.4, c: 2.5, runs: 30, steps: 500 } },
      { label: 'Two arms', values: { kind: 'bernoulli', k: 2, eps: 0.1, c: 1, runs: 40, steps: 300 } },
    ],
    counters: [{ id: 't', label: 'pulls' }],
    run(inp) {
      const algos = [{ algo: 'greedy', label: 'greedy' }, { algo: 'epsilon', eps: inp.eps, label: 'ε-greedy' }, { algo: 'ucb', c: inp.c, label: 'UCB1' }, { algo: 'thompson', label: 'Thompson' }];
      const res = S.RL.compareBandits({ kind: inp.kind, k: inp.k }, algos, { steps: inp.steps, runs: inp.runs, seed: 1 }), N = inp.steps, F = P.frames(['t']), q = F.q;
      const rmax = Math.max(...res.map((r) => r.regret[N - 1])), ar = C().axes({ prefix: 'rg', x: 60, y: 50, w: 400, h: 300, xr: [0, N], yr: [0, rmax * 1.05], xticks: P.ticks(N), yticks: C().ticks(0, rmax, 4), xname: 'pull', yname: 'average regret so far' });
      const ao = C().axes({ prefix: 'ro', x: 540, y: 50, w: 240, h: 300, xr: [0, N], yr: [0, 1], xticks: P.ticks(N), yticks: [0, 0.25, 0.5, 0.75, 1], xname: 'pull', yname: 'pulls of the best arm' });
      const sm = res.map((r) => P.smooth(r.optimal, 10));
      const lg = C().legend('lg', 60, 410, res.map((r, i) => ({ text: r.label, cls: LINES[i].cls })), { dir: 'row', size: 13 });
      const ts = [10, 25, 50, 100, 200, 350, 500, 750, 1000].filter((t) => t < N).concat([N]);
      for (const t of ts) {
        F.set('t', t);
        const items = [...ar.items, ...ao.items, ...lg, C().vline(ar, 'rg:now', t, 'act'), C().vline(ao, 'ro:now', t, 'act')];
        res.forEach((r, i) => { items.push(C().line(ar, 'rg:' + i, r.regret.slice(0, t).map((v, j) => [j + 1, v]), LINES[i].cls), C().line(ao, 'ro:' + i, sm[i].slice(0, t).map((v, j) => [j + 1, v]), LINES[i].cls)); });
        const reg = res.map((r) => r.regret[t - 1]), op = res.map((r) => r.optimal[t - 1]);
        const best = reg.indexOf(Math.min(...reg));
        const say = t === N
          ? `After ${q.n(N)} pulls, averaged over ${q.n(inp.runs)} bandits: regret ${H().list(res.map((r, i) => r.label + ' ' + q(reg[i], 1)))}. ${res[best].label} lost least. ` + (reg[0] > Math.min(reg[1], reg[2], reg[3]) ? 'Greedy’s curve stays a straight line: when it locks onto a poor arm it never finds out.' : 'Here even greedy did well: with few arms an early mistake is rare.')
          : `Pull ${q.n(t)}: average regret ${H().list(res.map((r, i) => r.label + ' ' + q(reg[i], 1)))}; the best arm was pulled at this step in ${H().list(op.map((v) => q.pct(v)))} of the bandits.`;
        F.push(items, say, { phase: t === N ? 'end' : 'step', t, reg, op, runs: inp.runs, N });
      }
      for (const f of F.list) f.narrowView = { x: 10, y: 20, w: 470, h: 420 }; return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Regret', text: ['regret(T) = Σ over t ≤ T of (μ* − μ(a_t))', 'greedy: can grow in a straight line', 'ε-greedy with fixed ε: grows like ε T', 'UCB1, Thompson: grow like log T'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.t === 10, pick: 'option', ask: 'Which strategy will have the least regret at the end?', options: ['greedy', 'ε-greedy', 'UCB1', 'Thompson'], answer: (fr) => { const r = fr[fr.length - 1].meta.reg; return ['greedy', 'ε-greedy', 'UCB1', 'Thompson'][r.indexOf(Math.min(...r))]; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Best arm mean 0.8; you pulled arms with means 0.8, 0.5 and 0.6. What is the regret?', options: ['0.5', '0.2', '1.9', '0.3'], answer: 0, why: '0 + 0.3 + 0.2.' },
      { q: 'Why does ε-greedy with a fixed ε have regret that keeps growing in a straight line?', options: ['a share ε of pulls stays random for ever, and most random arms are worse', 'it never exploits', 'its estimates diverge', 'it ignores rewards'], answer: 0, why: 'The exploration never stops, so its cost never stops.' },
      { q: 'Logarithmic regret means…', options: ['the extra loss per pull shrinks as the agent learns', 'regret falls over time', 'regret is zero', 'regret doubles each pull'], answer: 0, why: 'The curve still rises, but ever more slowly.' },
      { q: 'Why average over many bandits?', options: ['one run is mostly luck; the average shows what a strategy tends to do', 'to make it faster', 'because regret is random by definition', 'to remove the best arm'], answer: 0, why: 'Strategies are compared by their expected regret.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
