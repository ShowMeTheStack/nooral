// Part 15, reinforcement learning: the exercises. Hand computations are worked out here by the engines (S.RL: grid
// worlds, Bellman backups and value iteration) or by the update rules they use (rl-b-bandit.js, rl-d-td.js); the regret
// figures quote the topic rl-regret, whose numbers a scratch run printed.
(function (S) {
  'use strict';
  const C = S.Content, RL = S.RL;
  if (!C || !RL || C.exercises.has('re-x-estimates')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const mn = (v) => String(v).replace(/^-/, '−');
  const n2 = (v) => mn(r2(v).toFixed(2)), n3 = (v) => mn(r3(v).toFixed(3));
  const k2 = (v) => String(r2(v)), k3 = (v) => String(r3(v));
  const strs = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));

  // =========================================================================================================
  // Chapter 1: bandits
  // =========================================================================================================
  const PULLS = [['A', 1], ['B', 0], ['A', 0], ['C', 1], ['A', 1], ['C', 1], ['B', 1], ['C', 0]];
  const est = {}; for (const arm of ['A', 'B', 'C']) { const rs = PULLS.filter((p) => p[0] === arm).map((p) => p[1]); est[arm] = { n: rs.length, q: rs.reduce((a, b) => a + b, 0) / rs.length }; }
  const estCells = {}; for (const arm of ['A', 'B', 'C']) { estCells['arm ' + arm + '|pulls'] = est[arm].n; estCells['arm ' + arm + '|estimate'] = r2(est[arm].q); }
  C.exercise({ id: 're-x-estimates', chapter: 're-bandits', title: 'Estimates after a few pulls', level: 1, kind: 'table',
    text: ['A bandit with three arms. The pulls so far, in order, with the reward each paid:', { table: [['pull', ...PULLS.map((_, i) => String(i + 1))], ['arm', ...PULLS.map((p) => p[0])], ['reward', ...PULLS.map((p) => String(p[1]))]] }, 'For each arm, how many times has it been pulled, and what is its estimate Q, the mean of its rewards? Two decimal places.'],
    rows: ['arm A', 'arm B', 'arm C'], cols: ['pulls', 'estimate'], corner: '', cells: estCells, tolerance: 0.006,
    hints: ['Group the pulls by arm.', 'Arm A was pulled at 1, 3 and 5, paying 1, 0 and 1.', 'A: 3 pulls, 2/3 = 0.67; B: 2 pulls, 1/2 = 0.50; C: 3 pulls, 2/3 = 0.67.'],
    solution: strs(estCells),
    wrong: [Object.assign(strs(estCells), { 'arm A|estimate': '0.25', 'arm B|estimate': '0.13', 'arm C|estimate': '0.25' })] });

  const IQ = 0.6, IN = 4, IR = 1, IQ2 = IQ + (IR - IQ) / (IN + 1);
  C.exercise({ id: 're-x-incremental', chapter: 're-bandits', title: 'Update a running mean', level: 1, kind: 'number',
    text: ['An arm has been pulled 4 times and its estimate is Q = 0.6. The fifth pull pays 1. With Q ← Q + (r − Q) / N, where N counts the new pull, what is the new estimate?'],
    answer: r3(IQ2), tolerance: 0.001,
    whyNot: { 0.7: 'That divides by 4; N counts the new pull, so it is 5.', 0.8: 'That averages the old estimate and the new reward as if they weighed the same; the old one stands for 4 pulls.', 1: 'One reward does not replace four.', '*': 'N = 5: Q + (1 − Q) / 5.' },
    hints: ['After the fifth pull N = 5.', 'r − Q = 1 − 0.6 = 0.4.', '0.6 + 0.4 / 5 = ' + k3(IQ2) + ', the mean of the five rewards.'],
    solution: k3(IQ2), wrong: ['0.7', '0.8'] });

  const EPS = 0.1, EK = 5, EP = 1 - EPS + EPS / EK;
  C.exercise({ id: 're-x-eps-prob', chapter: 're-bandits', title: 'How often is the best-looking arm pulled?', level: 2, kind: 'number',
    text: ['An ε-greedy agent with ε = 0.1 plays a bandit with 5 arms. With probability ε it picks an arm uniformly at random (possibly the best-looking one); otherwise it picks the arm with the highest estimate. On one pull, what is the probability that it pulls the arm with the highest estimate (no ties)?'],
    answer: r3(EP), tolerance: 0.001,
    whyNot: { 0.9: 'A random pick can also land on the best-looking arm: add ε / 5.', 0.1: 'That is the chance of exploring at all.', 0.02: 'That is only the chance of reaching it by a random pick.', '*': 'Exploiting, or exploring and landing on it by chance.' },
    hints: ['Two ways to pull it: exploit, or explore and happen to pick it.', 'Exploit: 1 − ε = 0.9. Explore and pick it: ε × 1/5 = 0.02.', '0.9 + 0.02 = ' + k3(EP) + '.'],
    solution: k3(EP), wrong: ['0.9', '0.1'] });

  const gsOpts = ['The other estimates stay at 0, below arm 0’s, so greedy never pulls them again', 'Arm 0 is in fact the best arm', 'Greedy picks an arm at random after a loss', 'Its estimates are wrong because it divides by the wrong N'];
  C.exercise({ id: 're-x-greedy-stuck', chapter: 're-bandits', title: 'Why greedy gets stuck', level: 1, kind: 'choice',
    text: ['In [A multi-armed bandit, pull by pull](#topic=rl-bandit) with the preset “Greedy gets stuck”, arm 0 (paying 0.68) pays 1 on pull 4, and greedy then pulls it 497 times out of 500, never trying arm 4 (paying 0.71). Why?'],
    options: gsOpts, answer: 0,
    whyNot: { 1: 'Arm 4 pays 0.71 on average, more than arm 0’s 0.68.', 2: 'Greedy always picks the highest estimate; it only breaks ties at random.', 3: 'The estimates are correct running means; the trouble is which arms get pulled at all.' },
    hints: ['Look at the other estimates: they are still 0.', 'Arm 0’s estimate stays well above 0 even after a few losses.', 'An arm that is never pulled never changes its estimate, so greedy never learns that arm 4 is better. Exploration fixes this.'],
    solution: 0, wrong: [1, 2] });

  const RMU = [0.7, 0.5, 0.2], RCNT = [5, 3, 2], REG = RCNT.reduce((s, n, i) => s + n * (RMU[0] - RMU[i]), 0);
  C.exercise({ id: 're-x-regret', chapter: 're-bandits', title: 'Regret after ten pulls', level: 1, kind: 'number',
    text: ['Three arms pay out on average 0.7, 0.5 and 0.2. In 10 pulls an agent pulled the first arm 5 times, the second 3 times and the third twice. What is its regret, Σ (best mean − mean of the arm pulled), over the 10 pulls?'],
    answer: r2(REG), tolerance: 0.001,
    whyNot: { 7: 'That is the reward the best arm would have earned; regret is what was lost against it.', 5.4: 'That is the expected reward earned; subtract it from 7.', 0.7: 'Each pull of a worse arm adds the gap, once per pull.', '*': 'Each pull of arm 2 loses 0.2; each pull of arm 3 loses 0.5.' },
    hints: ['Pulls of the best arm lose nothing.', 'Arm 2 loses 0.7 − 0.5 = 0.2 per pull, arm 3 loses 0.5 per pull.', '3 × 0.2 + 2 × 0.5 = ' + k2(REG) + '.'],
    solution: k2(REG), wrong: ['7', '5.4'] });

  const UT = 20, UC = 1, UQ = [0.6, 0.5, 0.8], UN = [10, 4, 6], UB = UN.map((n) => UC * Math.sqrt((2 * Math.log(UT)) / n)), US = UQ.map((q, i) => q + UB[i]);
  const ubCells = {}; UQ.forEach((_, i) => { ubCells['arm ' + i + '|bonus'] = r3(UB[i]); ubCells['arm ' + i + '|score'] = r3(US[i]); });
  const ubWrong = {}; UQ.forEach((q, i) => { const b = Math.sqrt(Math.log(UT) / UN[i]); ubWrong['arm ' + i + '|bonus'] = n3(b); ubWrong['arm ' + i + '|score'] = n3(q + b); });
  const upick = US.indexOf(Math.max(...US));
  C.exercise({ id: 're-x-ucb', chapter: 're-bandits', title: 'UCB scores', level: 2, kind: 'table',
    text: ['UCB1 with c = 1 is about to make pull t = 20 (natural logarithms). Three arms:', { table: [['arm', '0', '1', '2'], ['estimate Q', ...UQ.map(String)], ['pulls N', ...UN.map(String)]] }, 'Compute each arm’s bonus c · √(2 ln t / N) and its score Q + bonus. Three decimal places.'],
    rows: ['arm 0', 'arm 1', 'arm 2'], cols: ['bonus', 'score'], corner: '', cells: ubCells, tolerance: 0.002,
    hints: ['2 ln 20 = ' + n3(2 * Math.log(UT)) + '.', 'Arm 0: √(' + n3(2 * Math.log(UT)) + ' / 10) = ' + n3(UB[0]) + '. Fewer pulls, larger bonus.', 'Bonuses ' + UB.map(n3).join(', ') + '; scores ' + US.map(n3).join(', ') + '. UCB1 pulls arm ' + upick + ': the highest score, though arm 1 has the lowest estimate and the largest bonus.'],
    solution: strs(ubCells), wrong: [ubWrong] });

  const unOpts = ['Arm 3: an arm never pulled has an unlimited bonus, so UCB1 tries it first', 'Arm 0: it has the highest estimate', 'Arm 1: it has the most pulls', 'Any arm at random'];
  C.exercise({ id: 're-x-ucb-untried', chapter: 're-bandits', title: 'An arm never pulled', level: 1, kind: 'choice',
    text: ['A UCB1 agent has pulled arms 0, 1 and 2 several times each, with estimates 0.8, 0.4 and 0.6, but has never pulled arm 3. Which arm does it pull next?'],
    options: unOpts, answer: 0,
    whyNot: { 1: 'The estimate is only half the score: arm 3’s bonus, with N = 0, is unbounded.', 2: 'Many pulls mean a small bonus, not a large one.', 3: 'UCB1 is deterministic: it pulls the highest score.' },
    hints: ['The bonus has N(a) under the square root, in the denominator.', 'With N = 0 the bonus is unbounded.', 'Arm 3: UCB1 tries every arm once before comparing scores, as [A multi-armed bandit, pull by pull](#topic=rl-bandit) shows on its first pulls.'],
    solution: 0, wrong: [1, 2] });

  const TW = 3, TL = 1, TMEAN = (1 + TW) / (2 + TW + TL);
  C.exercise({ id: 're-x-thompson', chapter: 're-bandits', title: 'Thompson sampling’s belief about an arm', level: 2, kind: 'number',
    text: ['Thompson sampling on 0-or-1 rewards keeps, for each arm, a Beta(1 + wins, 1 + losses) distribution of its payout chance. An arm has paid 1 three times and 0 once. What is the mean of its Beta distribution, (1 + wins) / (2 + wins + losses)? Three decimal places.'],
    answer: r3(TMEAN), tolerance: 0.001,
    whyNot: { 0.75: 'That is the raw win rate 3/4; the Beta(1, 1) start adds one win and one loss.', 0.5: 'That is the start, before any pulls.', '*': 'Beta(4, 2): mean 4 / 6.' },
    hints: ['Wins 3, losses 1: Beta(1 + 3, 1 + 1) = Beta(4, 2).', 'The mean of Beta(a, b) is a / (a + b).', '4 / 6 = ' + k3(TMEAN) + '. Each pull it draws one value from Beta(4, 2), sometimes high, sometimes low, so the arm still gets tried.'],
    solution: k3(TMEAN), wrong: ['0.75', '0.5'] });

  const RCX = [10, 25, 50, 100, 200, 350, 500], RCT = [3.0, 5.8, 9.6, 14.3, 20.5, 26.7, 30.0], RCG = [2.5, 6.0, 11.7, 23.3, 46.4, 81.0, 115.6];
  const rcPts = RCX.map((x, i) => [x, RCT[i]]);
  C.exercise({ id: 're-x-regret-curve', chapter: 're-bandits', title: 'Sketch Thompson sampling’s regret', level: 2, kind: 'predict-curve',
    text: ['In [Regret: greedy, ε-greedy, UCB and Thompson](#topic=rl-regret), averaged over 30 bandits of 10 arms, greedy play’s regret grows in a nearly straight line, to 115.6 after 500 pulls. Thompson sampling has ' + RCT[0] + ' after 10 pulls. Sketch Thompson sampling’s average regret up to 500 pulls.'],
    plot: { xr: [0, 500], yr: [0, 120], xname: 'pulls', yname: 'average regret', given: [rcPts[0]], note: 'Thompson sampling, averaged over 30 bandits' },
    answer: rcPts, tolerance: 6, checks: ['rising'],
    hints: ['Regret never goes down: each pull adds a gap of 0 or more.', 'As Thompson sampling grows sure of the best arm it pulls the others less, so the curve bends over.', 'The topic’s values: ' + rcPts.map(([x, v]) => x + ': ' + v.toFixed(1)).join(', ') + '. It ends at about a quarter of greedy’s.'],
    solution: rcPts, wrong: [RCX.map((x, i) => [x, RCG[i]]), RCX.map((x) => [x, RCT[0]])] });

  // =========================================================================================================
  // Chapter 2: planning with a known model
  // =========================================================================================================
  const RW = [-1, -1, -1, 10], RG = 0.9, RET = RW.reduce((s, r, k) => s + r * RG ** k, 0);
  C.exercise({ id: 're-x-return', chapter: 're-plan', title: 'A discounted return', level: 1, kind: 'number',
    text: ['An episode gives the rewards −1, −1, −1 and then 10, and ends. With γ = 0.9, what is the return from the start, G = r₀ + γ r₁ + γ² r₂ + γ³ r₃? Two decimal places.'],
    answer: r2(RET), tolerance: 0.006,
    whyNot: { 7: 'That adds the rewards without discounting.', [k2(-1 - 0.9 - 0.81 + 10 * 0.81)]: 'The last reward comes 3 steps later: γ³ = 0.729, not γ².', [k2(RW.reduce((s, r, k) => s + r * RG ** (k + 1), 0))]: 'The first reward is not discounted: γ⁰ = 1.', '*': '−1 − 0.9 − 0.81 + 0.729 × 10.' },
    hints: ['γ⁰ = 1, γ¹ = 0.9, γ² = 0.81, γ³ = 0.729.', '−1 − 0.9 − 0.81 = −2.71, and 0.729 × 10 = 7.29.', '−2.71 + 7.29 = ' + k2(RET) + '.'],
    solution: k2(RET), wrong: ['7', k2(-1 - 0.9 - 0.81 + 10 * 0.81)] });

  const drSol = 'function discountedReturn(rewards, gamma) {\n  let G = 0;\n  for (let k = rewards.length - 1; k >= 0; k--) G = rewards[k] + gamma * G;\n  return G;\n}';
  C.exercise({ id: 're-x-code-return', chapter: 're-plan', title: 'Write the discounted return', level: 1, kind: 'code',
    text: ['Write `discountedReturn(rewards, gamma)`: the rewards of an episode in order, and the discount γ. Return G = r₀ + γ r₁ + γ² r₂ + …, the return from the start. An empty episode returns 0.'],
    fn: 'discountedReturn', params: ['rewards', 'gamma'], start: 'function discountedReturn(rewards, gamma) {\n  \n}', tolerance: 1e-9,
    ref: (rw, g) => rw.reduce((s, r, k) => s + r * g ** k, 0),
    cases: [[RW, RG], [[], 0.9], [[5], 0.5], [[1, 1, 1, 1], 1], [[-0.04, -0.04, -0.04, 1], 0.95], [[0, 0, 0, 0, 100], 0.5], [Array.from({ length: 40 }, (_, i) => (i % 3) - 1), 0.99], [[2, -3], 0]],
    hints: ['Either sum r_k · γᵏ, or walk backwards with G = r + γ G.', 'Walking backwards needs no powers: start with G = 0 and go from the last reward to the first.', drSol],
    solution: drSol,
    wrong: ['function discountedReturn(rewards, gamma) {\n  return rewards.reduce((s, r) => s + r, 0);\n}', 'function discountedReturn(rewards, gamma) {\n  let G = 0;\n  for (let k = 0; k < rewards.length; k++) G = rewards[k] + gamma * G;\n  return G;\n}', 'function discountedReturn(rewards, gamma) {\n  return rewards.reduce((s, r, k) => s + r * Math.pow(gamma, k + 1), 0);\n}'] });

  const mkOpts = ['The chance of each next state depends only on the current state and action, not on how the agent got there', 'The agent always ends in the goal', 'Rewards are the same in every state', 'The agent remembers its whole history'];
  C.exercise({ id: 're-x-markov', chapter: 're-plan', title: 'The Markov property', level: 1, kind: 'choice',
    text: ['What does it mean that a decision process is Markov?'],
    options: mkOpts, answer: 0,
    whyNot: { 1: 'Episodes can end in a pit, or not at all; that has nothing to do with the Markov property.', 2: 'Rewards may differ everywhere: a pit, a goal, a step.', 3: 'The point is the reverse: the current state says everything needed, so history can be forgotten.' },
    hints: ['It is about what the future depends on.', 'P(s′ | s, a): only s and a appear.', 'The next state’s chances depend only on the current state and action. That is what lets a value be attached to a state alone.'],
    solution: 0, wrong: [1, 3] });

  const gmOpts = ['The γ = 0.5 agent: rewards far away are worth little to it, so a nearer, smaller reward wins', 'The γ = 0.99 agent', 'Both pick the same: γ only scales all values', 'Neither: γ changes speed, not choices'];
  const gNear = 1 * 0.5 ** 1, gFar = 5 * 0.5 ** 5, gNear2 = 1 * 0.99 ** 1, gFar2 = 5 * 0.99 ** 5;
  C.exercise({ id: 're-x-gamma', chapter: 're-plan', title: 'Short-sighted or patient', level: 2, kind: 'choice',
    text: ['From a cell, one route reaches a reward of 1 in 2 moves and another reaches a reward of 5 in 6 moves; nothing else is earned on the way. The first reward is discounted by γ¹ and the second by γ⁵. One agent uses γ = 0.5, another γ = 0.99. Which agent prefers the near, small reward?'],
    options: gmOpts, answer: 0,
    whyNot: { 1: 'With γ = 0.99: 1 × 0.99 = ' + n3(gNear2) + ' against 5 × 0.99⁵ = ' + n3(gFar2) + '. It waits for the 5.', 2: 'The two rewards are discounted by different powers of γ, so their order can change.', 3: 'The values decide the actions, and γ changes the values.' },
    hints: ['Compare γ¹ × 1 with γ⁵ × 5 for each agent.', 'γ = 0.5: 0.5 against 5 × 0.03125 = ' + n3(gFar) + '.', 'γ = 0.5 values the near reward at ' + n3(gNear) + ' and the far one at ' + n3(gFar) + ': it takes the near one. γ = 0.99 values them at ' + n3(gNear2) + ' and ' + n3(gFar2) + ': it takes the far one.'],
    solution: 0, wrong: [1, 2] });

  // a Bellman backup with two actions and slipping outcomes
  const BG = 0.9, BR = -0.04, BA = [{ name: 'east', outs: [[0.8, 0.5], [0.1, 0.2], [0.1, 0.3]] }, { name: 'south', outs: [[0.8, 0.4], [0.1, 0.5], [0.1, 0.3]] }];
  const BQ = BA.map((a) => a.outs.reduce((s, [p, v]) => s + p * (BR + BG * v), 0)), BV = Math.max(...BQ);
  C.exercise({ id: 're-x-bellman', chapter: 're-plan', title: 'One Bellman backup', level: 3, kind: 'number',
    text: ['A cell with two sensible actions on slippery ground, γ = 0.9 and a reward of −0.04 for every move. The current values of the cells each action can lead to, with their chances:', { table: [['action', 'outcome 1', 'outcome 2', 'outcome 3'], ...BA.map((a) => [a.name, ...a.outs.map(([p, v]) => 'p ' + p + ', V ' + v)])] }, 'What is the cell’s new value after one Bellman backup, max over the actions of Σ p (r + γ V(s′))? Three decimal places.'],
    answer: r3(BV), tolerance: 0.001,
    whyNot: { [k3(Math.min(...BQ))]: 'That is the worse action; the backup takes the best.', [k3(BA[0].outs.reduce((s, [p, v]) => s + p * BG * v, 0))]: 'The reward of −0.04 is earned on every move: add it.', [k3(BR + BG * 0.5)]: 'That ignores the slips: the 0.8 outcome is only one of three.', '*': 'Compute Σ p (r + γ V) for each action, then take the larger.' },
    hints: ['For each action, add p × (−0.04 + 0.9 × V) over its three outcomes.', 'East: 0.8 × (−0.04 + 0.45) + 0.1 × (−0.04 + 0.18) + 0.1 × (−0.04 + 0.27) = ' + n3(BQ[0]) + '. South: ' + n3(BQ[1]) + '.', 'The larger is east, ' + n3(BV) + ': the new value of the cell, and east is its best action.'],
    solution: k3(BV), wrong: [k3(Math.min(...BQ)), k3(BA[0].outs.reduce((s, [p, v]) => s + p * BG * v, 0))] });

  const VMAP = ['S..', '.XG'], venv = RL.grid({ world: RL.parseMap(VMAP), gamma: 0.9 }), vvi = RL.valueIteration(venv, { gamma: 0.9, theta: 1e-300, maxSweeps: 2 });
  const VCELLS = [[0, 0], [0, 1], [0, 2], [1, 0]], vsw = (k) => vvi.sweeps[k - 1].V;
  const viCells = {}; for (const k of [1, 2]) for (const [i, j] of VCELLS) viCells['sweep ' + k + '|(' + i + ', ' + j + ')'] = r3(vsw(k)[i * 3 + j]);
  const viWrong = strs(viCells); viWrong['sweep 2|(0, 0)'] = '0.734';
  C.exercise({ id: 're-x-vi-sweep', chapter: 're-plan', title: 'Two sweeps of value iteration', level: 3, kind: 'table',
    text: ['A grid of 2 rows and 3 columns, written (row, column):', { table: [['', 'column 0', 'column 1', 'column 2'], ['row 0', 'S (start)', '.', '.'], ['row 1', '.', 'X (pit)', 'G (goal)']] }, 'Moves are certain; a move into the edge stays put. Every move earns −0.04, except that entering the goal earns 1 and entering the pit −1, and both end the episode (their values stay 0). With γ = 0.9 and every V starting at 0, run two sweeps of value iteration. Each sweep uses only the values from the sweep before.'],
    rows: ['sweep 1', 'sweep 2'], cols: VCELLS.map(([i, j]) => '(' + i + ', ' + j + ')'), corner: 'cell', cells: viCells, tolerance: 0.002,
    hints: ['Sweep 1, with every V = 0: a cell next to the goal can earn 1 by entering it; any other move earns −0.04.', 'Sweep 1: (0, 2) gets 1 (south into the goal); the rest −0.04. Sweep 2: (0, 1) east to (0, 2) gives −0.04 + 0.9 × 1 = 0.86.', 'Sweep 1: ' + VCELLS.map(([i, j]) => n3(vsw(1)[i * 3 + j])).join(', ') + '. Sweep 2: ' + VCELLS.map(([i, j]) => n3(vsw(2)[i * 3 + j])).join(', ') + '. (0, 0) only learns of the goal in sweep 3, using sweep 2’s 0.86 at (0, 1).'],
    solution: strs(viCells), wrong: [viWrong] });

  const viRef = (rows, gamma, sweeps) => {
    const env = RL.grid({ world: RL.parseMap(rows), gamma }), r = RL.valueIteration(env, { gamma, theta: 1e-300, maxSweeps: sweeps });
    return rows.map((row, i) => [...row].map((_, j) => r.V[i * env.w + j]));
  };
  const viSol = [
    'function valueIteration(rows, gamma, sweeps) {',
    '  const h = rows.length, w = rows[0].length, moves = [[-1, 0], [0, 1], [1, 0], [0, -1]];',
    '  const at = (i, j) => rows[i][j];',
    '  let V = rows.map((r) => [...r].map(() => 0));',
    '  for (let k = 0; k < sweeps; k++) {',
    '    const nV = V.map((r) => r.slice());',
    '    for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {',
    '      if (\'#GX\'.includes(at(i, j))) continue;',
    '      let best = -Infinity;',
    '      for (const [di, dj] of moves) {',
    '        let ni = i + di, nj = j + dj;',
    '        if (ni < 0 || nj < 0 || ni >= h || nj >= w || at(ni, nj) === \'#\') { ni = i; nj = j; }',
    '        const c = at(ni, nj);',
    '        const q = c === \'G\' ? 1 : c === \'X\' ? -1 : -0.04 + gamma * V[ni][nj];',
    '        best = Math.max(best, q);',
    '      }',
    '      nV[i][j] = best;',
    '    }',
    '    V = nV;',
    '  }',
    '  return V;',
    '}'].join('\n');
  const M1 = ['S...', '.#.X', '...G'], M2 = ['S.#G', '..#.', '....'];
  C.exercise({ id: 're-x-code-vi', chapter: 're-plan', title: 'Write value iteration on a grid', level: 3, kind: 'code',
    text: ['Write `valueIteration(rows, gamma, sweeps)`. The grid comes as strings, one per row: `.` floor, `S` the start (also floor), `#` a wall, `G` the goal, `X` a pit. Four actions move north, east, south or west, with certainty; a move into a wall or off the grid stays put. A move earns −0.04, except that entering `G` earns 1 and entering `X` earns −1, and both end the episode, so their values stay 0 (walls too). Start with every V at 0 and run `sweeps` sweeps (at least 1), each using only the values of the sweep before: V(s) ← max over actions of r + γ V(s′), with V(s′) left out when the move ends the episode. Return V as an array of rows of numbers, one per cell.'],
    fn: 'valueIteration', params: ['rows', 'gamma', 'sweeps'], start: 'function valueIteration(rows, gamma, sweeps) {\n  \n}', tolerance: 1e-6,
    ref: viRef,
    cases: [[VMAP, 0.9, 1], [VMAP, 0.9, 2], [VMAP, 0.9, 5], [['SG'], 0.5, 1], [M1, 0.95, 3], [M1, 0.95, 40], [M2, 0.9, 10], [['S..X', '....', '#..G'], 0.8, 25]],
    hints: ['Keep two tables: the old V that every backup reads, and a new one that the sweep writes.', 'For each floor cell and each action, find where the move lands (stay put at a wall or edge). Entering G gives 1, entering X gives −1, both with nothing after; any other landing gives −0.04 + γ V(landing). Take the best of the four.', viSol],
    solution: viSol,
    wrong: [viSol.replace('const nV = V.map((r) => r.slice());', 'const nV = V;'), viSol.replace("c === 'G' ? 1 : c === 'X' ? -1 : -0.04 + gamma * V[ni][nj]", "c === 'G' ? 1 + gamma * V[ni][nj] : c === 'X' ? -1 : gamma * V[ni][nj]"), viSol.replace('best = Math.max(best, q);', 'best = best === -Infinity ? q : (best + q) / 2;')] });

  const PE = { p: 0.5, rGoal: 1, rStay: -0.04, g: 0.9 }, PEV = (PE.p * PE.rGoal + (1 - PE.p) * PE.rStay) / (1 - (1 - PE.p) * PE.g);
  C.exercise({ id: 're-x-policy-eval', chapter: 're-plan', title: 'Evaluate a policy that may stay put', level: 3, kind: 'number',
    text: ['A policy in one cell always pushes towards the goal next door. Half the time it gets there, earning 1 and ending the episode; the other half it slips and stays in the same cell, earning −0.04. With γ = 0.9, the cell’s value under this policy satisfies V = 0.5 × 1 + 0.5 × (−0.04 + 0.9 V). What is V? Three decimal places.'],
    answer: r3(PEV), tolerance: 0.001,
    whyNot: { 0.48: 'That is one step’s expected reward with V(s′) = 0; but staying leads back to this same cell, worth V.', [k3(0.5 + 0.5 * (-0.04 + 0.9 * 0.48))]: 'That is two rounds of the backup; solve the equation for its fixed point.', 1: 'Half the time it slips, paying −0.04 and losing time.', '*': 'Gather the V terms on one side: V (1 − 0.45) = 0.48.' },
    hints: ['V appears on both sides: staying puts the agent back in the same cell.', 'V = 0.5 − 0.02 + 0.45 V, so 0.55 V = 0.48.', 'V = 0.48 / 0.55 = ' + k3(PEV) + '. Policy evaluation by sweeps would creep towards this same number.'],
    solution: k3(PEV), wrong: ['0.48', k3(0.5 + 0.5 * (-0.04 + 0.9 * 0.48))] });

  const pvOpts = ['Both reach the same optimal values; policy iteration takes few rounds but each evaluates a policy fully, value iteration takes many cheap sweeps', 'Policy iteration finds a better policy than value iteration', 'Value iteration needs the model, policy iteration does not', 'Policy iteration never stops on a slippery grid'];
  C.exercise({ id: 're-x-pi-vs-vi', chapter: 're-plan', title: 'Policy iteration against value iteration', level: 2, kind: 'choice',
    text: ['On the pits world, [Policy iteration](#topic=rl-policy-iteration) needs 8 rounds and ends with V at the start = 0.5232; [Value iteration](#topic=rl-value-iteration) needs 9 sweeps and ends with 0.5232 too. Which statement is right?'],
    options: pvOpts, answer: 0,
    whyNot: { 1: 'Both end with 0.5232 at the start: the same optimum.', 2: 'Both are planning methods: both use P(s′ | s, a).', 3: 'It stops as soon as no state wants to change its action, slippery or not.' },
    hints: ['Compare their end values.', 'A round of policy iteration contains a whole evaluation, which can take hundreds of sweeps (298 in the first round here).', 'Both find V*; they divide the work differently.'],
    solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 3: learning from experience
  // =========================================================================================================
  const MR = [-0.04, -0.04, -0.04, 1], MG = 0.9, MRET = []; { let g = 0; for (let k = MR.length - 1; k >= 0; k--) { g = MR[k] + MG * g; MRET[k] = g; } }
  const mcCells = Object.fromEntries(MRET.map((g, k) => ['t = ' + k + '|G_t', r3(g)]));
  const mcWrong = {}; MR.forEach((_, k) => { mcWrong['t = ' + k + '|G_t'] = k3(MR.slice(k).reduce((s, r) => s + r, 0)); });
  C.exercise({ id: 're-x-mc-return', chapter: 're-learn', title: 'Returns, walking back from the end', level: 2, kind: 'table',
    text: ['An episode of four steps earns −0.04, −0.04, −0.04 and then 1 at the goal. With γ = 0.9, Monte Carlo walks back from the end with G_t = r_t + γ G_{t+1}. Write the return G_t after each step t. Three decimal places.'],
    rows: MR.map((_, k) => 't = ' + k), cols: ['G_t'], corner: '', cells: mcCells, tolerance: 0.002,
    hints: ['Start at the end: G₃ = 1, since nothing follows.', 'G₂ = −0.04 + 0.9 × 1 = 0.86.', 'G = ' + MRET.map(n3).join(', ') + ' for t = 0 to 3. Each G_t is a sample that Monte Carlo averages into Q of that step’s state and action.'],
    solution: strs(mcCells), wrong: [mcWrong] });

  const QQ = 0.2, QR = -0.04, QG = 0.9, QA = 0.5, QN = [0.5, 0.7, 0.1, 0.3], QNA = 3;
  const qT = QR + QG * Math.max(...QN), sT = QR + QG * QN[QNA], qNew = QQ + QA * (qT - QQ), sNew = QQ + QA * (sT - QQ);
  C.exercise({ id: 're-x-q-update', chapter: 're-learn', title: 'One Q-learning update', level: 1, kind: 'number',
    text: ['Q-learning with α = 0.5 and γ = 0.9. The agent has Q(s, east) = 0.2. It goes east, earns −0.04 and lands in s′, whose action values are north 0.5, east 0.7, south 0.1 and west 0.3. What is Q(s, east) after the update Q ← Q + α (r + γ max Q(s′, ·) − Q)? Three decimal places.'],
    answer: r3(qNew), tolerance: 0.001,
    whyNot: { [k3(qT)]: 'That is the target; Q moves only a fraction α of the way to it.', [k3(QQ + QA * (QR + Math.max(...QN) - QQ))]: 'Discount the next value: γ × 0.7.', [k3(QQ + QA * (QR + QG * (QN.reduce((a, b) => a + b, 0) / 4) - QQ))]: 'Q-learning uses the best next value, not the average.', '*': 'Target −0.04 + 0.9 × 0.7, then move half way.' },
    hints: ['The best action value in s′ is 0.7.', 'Target: −0.04 + 0.9 × 0.7 = ' + k3(qT) + '. The error is ' + k3(qT) + ' − 0.2 = ' + k3(qT - QQ) + '.', 'Q = 0.2 + 0.5 × ' + k3(qT - QQ) + ' = ' + k3(qNew) + '.'],
    solution: k3(qNew), wrong: [k3(qT), k3(QQ + QA * (QR + Math.max(...QN) - QQ))] });

  const tgCells = { 'Q-learning|target': r3(qT), 'Q-learning|new Q(s, east)': r3(qNew), 'SARSA|target': r3(sT), 'SARSA|new Q(s, east)': r3(sNew) };
  C.exercise({ id: 're-x-targets', chapter: 're-learn', title: 'SARSA’s target against Q-learning’s', level: 2, kind: 'table',
    text: ['The same step as before: Q(s, east) = 0.2, reward −0.04, α = 0.5, γ = 0.9, and in s′ the action values are north 0.5, east 0.7, south 0.1 and west 0.3. Exploring, the agent has already chosen its next action in s′: west. Fill in each method’s target and the new Q(s, east). Three decimal places.'],
    rows: ['Q-learning', 'SARSA'], cols: ['target', 'new Q(s, east)'], corner: '', cells: tgCells, tolerance: 0.002,
    hints: ['Q-learning uses the best next value, whatever the agent will do; SARSA uses the value of the action it will really take.', 'Q-learning: −0.04 + 0.9 × 0.7 = ' + k3(qT) + '. SARSA: −0.04 + 0.9 × 0.3 = ' + k3(sT) + '.', 'New values: Q-learning ' + k3(qNew) + ', SARSA 0.2 + 0.5 × (' + k3(sT) + ' − 0.2) = ' + k3(sNew) + '. SARSA counts the cost of its own exploring.'],
    solution: strs(tgCells), wrong: [strs(Object.assign({}, tgCells, { 'SARSA|target': r3(qT), 'SARSA|new Q(s, east)': r3(qNew) }))] });

  const cpOpts = ['Q-learning learns the edge route, since its targets assume the best next action; SARSA learns a safer route, since its targets include its own random steps', 'SARSA learns the edge route, Q-learning the safer one', 'Both learn the edge route; SARSA just learns it more slowly', 'Neither reaches the goal while ε > 0'];
  C.exercise({ id: 're-x-cliff-paths', chapter: 're-learn', title: 'Two routes along the cliff', level: 2, kind: 'choice',
    text: ['On [the cliff](#topic=rl-cliff), with ε = 0.1, Q-learning’s greedy route takes 13 moves and SARSA’s 17, yet while exploring SARSA’s episodes average −29.6 and Q-learning’s −51.3. Which explanation is right?'],
    options: cpOpts, answer: 0,
    whyNot: { 1: 'The 13-move route, the shortest, runs along the edge: it is Q-learning’s.', 2: 'SARSA’s greedy route stays 17 moves: it keeps away from the edge.', 3: 'Both greedy routes reach the goal; exploring only spoils some episodes.' },
    hints: ['Which target pretends the agent will act greedily next?', 'Q-learning uses max Q(s′, ·); SARSA uses Q(s′, a′) for the exploring action a′, which sometimes steps off the edge.', 'Q-learning learns the shortest, edge route but falls while exploring; SARSA learns a route that is safe for an agent that sometimes steps at random.'],
    solution: 0, wrong: [1, 2] });

  const qSol = 'function qUpdate(Q, s, a, r, s2, done, alpha, gamma) {\n  const next = done ? 0 : Math.max(...Q[s2]);\n  const target = r + gamma * next;\n  return Q[s][a] + alpha * (target - Q[s][a]);\n}';
  const QT1 = [[0.2, 0.1, 0, -0.3], [0.5, 0.7, 0.1, 0.3], [0, 0, 0, 0]];
  C.exercise({ id: 're-x-code-q', chapter: 're-learn', title: 'Write the Q-learning update', level: 2, kind: 'code',
    text: ['Write `qUpdate(Q, s, a, r, s2, done, alpha, gamma)`. Q is a table, Q[state][action]. The agent took action a in state s, earned r and landed in s2; done is true when that ended the episode. Return the new value of Q[s][a] under Q-learning: the target is r + γ max over a′ of Q[s2][a′], or just r when done; move Q[s][a] a fraction α of the way to it.'],
    fn: 'qUpdate', params: ['Q', 's', 'a', 'r', 's2', 'done', 'alpha', 'gamma'], start: 'function qUpdate(Q, s, a, r, s2, done, alpha, gamma) {\n  \n}', tolerance: 1e-9,
    ref: (Q, s, a, r, s2, done, al, g) => Q[s][a] + al * (r + (done ? 0 : g * Math.max(...Q[s2])) - Q[s][a]),
    cases: [[QT1, 0, 1, -0.04, 1, false, 0.5, 0.9], [QT1, 0, 1, 1, 2, true, 0.5, 0.9], [QT1, 1, 3, -1, 0, false, 0.1, 0.99], [[[-5, -6], [-2, -1]], 0, 0, -1, 1, false, 1, 1], [[[-5, -6], [-2, -1]], 0, 0, -1, 1, true, 1, 1], [[[0, 0], [-3, -4]], 1, 1, -1, 1, false, 0.2, 0.9]],
    hints: ['The target looks at the next state’s best value, unless the episode ended.', 'Mind tables whose values are all negative: start a maximum from the first value (Math.max(...row)), not from 0.', qSol],
    solution: qSol,
    wrong: ['function qUpdate(Q, s, a, r, s2, done, alpha, gamma) {\n  const target = r + gamma * Math.max(...Q[s2]);\n  return Q[s][a] + alpha * (target - Q[s][a]);\n}', 'function qUpdate(Q, s, a, r, s2, done, alpha, gamma) {\n  const next = done ? 0 : Math.max(0, ...Q[s2]);\n  return Q[s][a] + alpha * (r + gamma * next - Q[s][a]);\n}', 'function qUpdate(Q, s, a, r, s2, done, alpha, gamma) {\n  const next = done ? 0 : Math.max(...Q[s2]);\n  return r + gamma * next;\n}'] });

  const ED = 0.98, EE = 1, EN = 50, EV = EE * ED ** EN;
  C.exercise({ id: 're-x-eps-decay', chapter: 're-learn', title: 'ε after fifty episodes', level: 2, kind: 'number',
    text: ['An agent starts with ε = 1 and multiplies ε by 0.98 after every episode. What is ε after 50 episodes? Three decimal places.'],
    answer: r3(EV), tolerance: 0.002,
    whyNot: { 0.02: 'That is 1 − 0.98: one step of decay, not fifty.', [k3(1 - 50 * 0.02)]: 'That subtracts 0.02 fifty times; the decay multiplies, and multiplying never reaches 0.', '*': '0.98 to the power 50.' },
    hints: ['Each episode multiplies ε by 0.98.', 'After 50 episodes: 1 × 0.98⁵⁰.', '0.98⁵⁰ = ' + k3(EV) + ', as [Exploration and ε](#topic=rl-explore) shows after 50 episodes (0.36).'],
    solution: k3(EV), wrong: ['0.02', k3(1 - 50 * 0.02)] });

  const xoOpts = ['Start ε high and let it decay: explore early, exploit later', 'Keep ε = 0 from the start: exploring only wastes reward', 'Keep ε = 0.8 for ever so that nothing is missed', 'Raise ε as training goes on'];
  C.exercise({ id: 're-x-explore-choice', chapter: 're-learn', title: 'Choose an exploration plan', level: 2, kind: 'choice',
    text: ['Q-learning crosses two rooms. After 300 episodes the last 20 average 0.118 with ε fixed at 0.3, and 0.466 with ε starting at 1 and decaying by 0.98 per episode. Which plan is sound in general?'],
    options: xoOpts, answer: 0,
    whyNot: { 1: 'With no exploration an agent may never find a better route than its first; it worked here only because ties were broken at random.', 2: 'A high fixed ε keeps paying for random moves after the route is known.', 3: 'That explores most when the agent knows most.' },
    hints: ['Early on the agent knows nothing; later it knows a good route.', 'Exploration is most useful early and most costly late.', 'Decay ε: 0.466 against 0.118 here.'],
    solution: 0, wrong: [1, 2] });

  // =========================================================================================================
  // Chapter 4: policies and networks
  // =========================================================================================================
  const PG = [['right', 2.0], ['right', -1.5], ['left', 0.8], ['left', -0.6]];
  const pgCells = Object.fromEntries(PG.map(([a, A], i) => ['step ' + (i + 1) + '|P(right)', (a === 'right') === (A > 0) ? 'up' : 'down']));
  C.exercise({ id: 're-x-pg-sign', chapter: 're-deep', title: 'Which way does REINFORCE push?', level: 1, kind: 'table',
    text: ['A cart-pole policy has two actions, left and right. In one update REINFORCE looks at four steps, each with the action taken and its advantage A = G − b:', { table: [['step', '1', '2', '3', '4'], ['action', ...PG.map((p) => p[0])], ['advantage A', ...PG.map((p) => mn(p[1]))]] }, 'Taken on its own, does each step’s term A · ∇ log π(action) push the probability of pushing right up or down? Write up or down.'],
    rows: PG.map((_, i) => 'step ' + (i + 1)), cols: ['P(right)'], corner: '', cells: pgCells, cellKind: 'text',
    hints: ['A positive advantage makes the action taken likelier; a negative one makes it less likely.', 'With two actions, making left likelier makes right less likely.', 'Step 1: right, A > 0: up. Step 2: right, A < 0: down. Step 3: left, A > 0: left up, so right down. Step 4: left, A < 0: left down, so right up.'],
    solution: pgCells, wrong: [Object.fromEntries(PG.map(([, A], i) => ['step ' + (i + 1) + '|P(right)', A > 0 ? 'up' : 'down']))] });

  const PT = 0.4, PA = 1.5, PLR = 0.1, PP = 1 / (1 + Math.exp(-PT)), PTN = PT + PLR * PA * (1 - PP);
  C.exercise({ id: 're-x-pg-step', chapter: 're-deep', title: 'One REINFORCE step by hand', level: 3, kind: 'number',
    text: ['A one-parameter policy pushes right with probability π(right) = σ(θ) = 1 / (1 + e^−θ), and left otherwise. Now θ = 0.4. The agent pushed right, and that step’s advantage is A = 1.5. For this policy ∂ log π(right) / ∂θ = 1 − σ(θ). With learning rate 0.1, what is θ after the update θ ← θ + lr · A · ∂ log π(right) / ∂θ? Four decimal places.'],
    answer: Math.round(PTN * 1e4) / 1e4, tolerance: 0.0002,
    whyNot: { [String(Math.round((PT + PLR * PA * PP) * 1e4) / 1e4)]: 'That uses σ(θ); the gradient of log π(right) is 1 − σ(θ).', [String(Math.round((PT + PLR * PA) * 1e4) / 1e4)]: 'The gradient of log π(right) is not 1: it is 1 − σ(θ).', [String(Math.round((PT - PLR * PA * (1 - PP)) * 1e4) / 1e4)]: 'A positive advantage makes the action taken likelier: add, do not subtract.', '*': 'σ(0.4) first, then 1 − σ, then θ + 0.1 × 1.5 × (1 − σ).' },
    hints: ['σ(0.4) = 1 / (1 + e^−0.4) = ' + PP.toFixed(4) + '.', '1 − σ(0.4) = ' + (1 - PP).toFixed(4) + '; times A = 1.5 and lr = 0.1 that is ' + (PLR * PA * (1 - PP)).toFixed(4) + '.', 'θ = 0.4 + ' + (PLR * PA * (1 - PP)).toFixed(4) + ' = ' + PTN.toFixed(4) + ', so π(right) rises from ' + PP.toFixed(3) + ' to ' + (1 / (1 + Math.exp(-PTN))).toFixed(3) + '.'],
    solution: PTN.toFixed(4), wrong: [(PT + PLR * PA * PP).toFixed(4), (PT + PLR * PA).toFixed(4)] });

  const blOpts = ['It makes the updates less noisy, without changing their direction on average', 'It makes every advantage positive', 'It turns REINFORCE into Q-learning', 'It replaces the need for returns'];
  C.exercise({ id: 're-x-baseline', chapter: 're-deep', title: 'Why subtract a baseline?', level: 2, kind: 'choice',
    text: ['In cart-pole every step earns +1, so every return G is positive. Without a baseline REINFORCE would make every action taken likelier, the good ones only a little more than the bad. [REINFORCE: a policy that balances a pole](#topic=rl-reinforce) subtracts a learned baseline b(s) and uses A = G − b(s). What does the baseline do?'],
    options: blOpts, answer: 0,
    whyNot: { 1: 'The reverse: with a baseline, worse-than-expected actions get negative advantages, so they are pushed down.', 2: 'It is still a policy gradient: it changes action probabilities, not a table of values.', 3: 'The advantage is built from the return: A = G − b.' },
    hints: ['Subtracting a number that does not depend on the action leaves the expected gradient the same.', 'But it centres the advantages: some positive, some negative.', 'Actions are judged against what was expected from that state; the updates are far less noisy, so learning is faster.'],
    solution: 0, wrong: [1, 3] });

  const DQ = [0.31, 0.47, 0.42, 0.12], DR = -0.04, DGm = 0.95, DY = DR + DGm * Math.max(...DQ);
  C.exercise({ id: 're-x-dqn-target', chapter: 're-deep', title: 'A deep Q-network’s target', level: 1, kind: 'number',
    text: ['A transition drawn from the replay buffer: reward −0.04, and the episode goes on. For the next state the target network gives the action values 0.31, 0.47, 0.42 and 0.12. With γ = 0.95, what is the target y = r + γ max Q_target(s′, ·)? Four decimal places.'],
    answer: Math.round(DY * 1e4) / 1e4, tolerance: 0.0002,
    whyNot: { 0.47: 'Add the reward and discount the next value: −0.04 + 0.95 × 0.47.', [String(Math.round((DR + DGm * DQ.reduce((a, b) => a + b, 0) / 4) * 1e4) / 1e4)]: 'The target uses the best next value, not the average.', [String(Math.round((DR + Math.max(...DQ)) * 1e4) / 1e4)]: 'Discount the next value by γ = 0.95.', '*': '−0.04 + 0.95 × 0.47.' },
    hints: ['The best next value is 0.47.', '0.95 × 0.47 = 0.4465.', 'y = −0.04 + 0.4465 = ' + DY.toFixed(4) + '. The online network’s Q(s, a) is pulled towards it.'],
    solution: DY.toFixed(4), wrong: ['0.47', String(Math.round((DR + Math.max(...DQ)) * 1e4) / 1e4)] });

  C.exercise({ id: 're-x-dqn-terminal', chapter: 're-deep', title: 'The target when the episode ends', level: 2, kind: 'number',
    text: ['Another transition from the buffer: the agent stepped into the goal, earning 1, and the episode ended. The target network happens to give that cell the values 0.2, 0.6, 0.1 and 0.3. With γ = 0.95, what is the target y?'],
    answer: 1, tolerance: 0.0001,
    whyNot: { 1.57: 'After the end there is nothing more to earn: the next state’s values are not used.', 0.6: 'The reward 1 is earned; only the future part is dropped.', '*': 'y = r when the episode ends.' },
    hints: ['y = r + γ max Q_target(s′, ·) holds only while the episode goes on.', 'At the end, the value of what follows is 0 by definition.', 'y = r = 1. Using the network’s values there would teach it a future that does not exist.'],
    solution: '1', wrong: ['1.57', '0.6'] });

  const dwOpts = ['A replay buffer: drawing old transitions at random breaks the correlation between consecutive steps and reuses each one many times', 'A target network: a copy refreshed only now and then keeps the targets from moving with every update', 'A replay buffer lets the agent skip exploration', 'A target network makes the reward larger', 'Both are needed only because the network is small'];
  C.exercise({ id: 're-x-dqn-why', chapter: 're-deep', title: 'Why replay and a target network?', level: 2, kind: 'choice',
    text: ['[A small deep Q-network with replay](#topic=rl-dqn) keeps a replay buffer and a target network. Which statements say what they are for? Choose all that do.'],
    options: dwOpts, answer: [0, 1],
    whyNot: { 2: 'The agent still explores (ε falls from 1 to 0.05); replay only changes which transitions it learns from.', 3: 'Rewards come from the world; the target network only supplies the next-state values.', 4: 'They matter more, not less, for large networks: any network’s update moves the values of other states.' },
    hints: ['Consecutive steps of one episode are very alike.', 'With one network, the target r + γ max Q(s′, ·) moves every time the network is updated.', 'Replay decorrelates and reuses experience; a frozen target network holds the targets still between refreshes.'],
    solution: [0, 1], wrong: [[0], [0, 1, 2]] });
})(globalThis.Nooral = globalThis.Nooral || {});
