// Part 15, reinforcement learning: the narration of every topic read back against an independent working-out. The
// grid worlds are rebuilt here from their maps (moves, slips, rewards), and value iteration, policy evaluation (by
// solving the linear equations exactly), returns, TD targets and greedy walks are written out by hand; the bandits'
// estimates and regret are recomputed from the pulls; the pole is re-simulated with hand-written physics and the
// stored policy's weights. The stored training runs are checked to be what the engine makes.
// Usage: node tests/claims-p15.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';
import { JOBS, runAll } from './ref/p15_runs.mjs';

const ALLOWED = [/UCB1/g];
const D = S.P15D, RL = S.RL;
const rel = (a, b, t) => Math.abs(a - b) <= t * Math.max(1, Math.abs(a), Math.abs(b));
const near = (c, a, b, tol, msg) => c.ok(rel(a, b, tol), msg + ': ' + a + ' against ' + b);
const vecNear = (c, a, b, tol, msg) => c.ok(a && b && a.length === b.length && a.every((v, i) => rel(v, b[i], tol)), msg + ': ' + JSON.stringify(a).slice(0, 100) + ' against ' + JSON.stringify(b).slice(0, 100));
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

// ---- the stored runs are the engine's -----------------------------------------------------------------------------
{
  const only = process.argv.slice(2);
  if (!only.length || only.some((o) => o.startsWith('rl-r') || o.startsWith('rl-d') || o === 'rl-')) {
    const again = runAll(load15()), bad = [];
    if (JSON.stringify(again.reinforce.snaps) !== JSON.stringify(D.reinforce.snaps)) bad.push('the policy snapshots');
    if (JSON.stringify(again.reinforce.meanLen) !== JSON.stringify(D.reinforce.meanLen)) bad.push('the episode lengths');
    if (JSON.stringify(again.dqn.snaps) !== JSON.stringify(D.dqn.snaps)) bad.push('the Q-tables');
    if (JSON.stringify(again.dqn.batches) !== JSON.stringify(D.dqn.batches)) bad.push('the replayed minibatches');
    console.log('  the stored runs repeat: ' + (bad.length ? 'FAIL ' + bad.join('; ') : 'REINFORCE and the DQN agree exactly'));
    if (bad.length) process.exitCode = 1;
  }
}
function load15() { return S; }

// ---- a grid world rebuilt from its map ------------------------------------------------------------------------------
const MOVE = [[-1, 0], [0, 1], [1, 0], [0, -1]];
function world(env) {
  const rows = env.world.rows, h = rows.length, w = rows[0].length, n = w * h, ch = (s) => rows[Math.floor(s / w)][s % w], rw = env.rewards, slip = env.slip, end = env.pitMode === 'end';
  const start = rows.findIndex((r) => r.includes('S')) * w + rows.find((r) => r.includes('S')).indexOf('S');
  const terminal = (s) => ch(s) === 'G' || (ch(s) === 'X' && end);
  const go = (s, a) => { const i = Math.floor(s / w) + MOVE[a][0], j = (s % w) + MOVE[a][1]; return i < 0 || j < 0 || i >= h || j >= w || rows[i][j] === '#' ? s : i * w + j; };
  const land = (s2) => (ch(s2) === 'G' ? { s2, r: rw.goal, done: true } : ch(s2) === 'X' ? (end ? { s2, r: rw.pit, done: true } : { s2: start, r: rw.pit, done: false }) : { s2, r: rw.step, done: false });
  const T = (s, a) => { const outs = slip ? [[a, 1 - slip], [(a + 1) % 4, slip / 2], [(a + 3) % 4, slip / 2]] : [[a, 1]]; return outs.map(([b, p]) => Object.assign({ p }, land(go(s, b)))); };
  const open = [...Array(n).keys()].filter((s) => ch(s) !== '#' && !terminal(s));
  const qv = (V, s, a, g) => T(s, a).reduce((t, o) => t + o.p * (o.r + (o.done ? 0 : g * V[o.s2])), 0);
  return { n, w, h, start, open, T, qv, terminal, ch };
}
/** V^π exactly: (I − γ P_π) V = r_π over the open states, by Gaussian elimination */
function solvePolicy(W, pol, g) {
  const idx = new Map(W.open.map((s, k) => [s, k])), m = W.open.length, A = Array.from({ length: m }, () => new Array(m + 1).fill(0));
  W.open.forEach((s, k) => { A[k][k] += 1; for (const o of W.T(s, pol[s])) { A[k][m] += o.p * o.r; if (!o.done && idx.has(o.s2)) A[k][idx.get(o.s2)] -= g * o.p; } });
  for (let col = 0; col < m; col++) { let p = col; for (let r = col + 1; r < m; r++) if (Math.abs(A[r][col]) > Math.abs(A[p][col])) p = r; [A[col], A[p]] = [A[p], A[col]]; for (let r = 0; r < m; r++) if (r !== col) { const f = A[r][col] / A[col][col]; if (f) for (let k = col; k <= m; k++) A[r][k] -= f * A[col][k]; } }
  const V = new Array(W.n).fill(0); W.open.forEach((s, k) => { V[s] = A[k][m] / A[k][k]; });
  return V;
}
function valueIter(W, g, theta = 1e-6) { let V = new Array(W.n).fill(0); const sweeps = []; for (let k = 0; k < 2000; k++) { const nV = V.slice(); let d = 0, at = W.start; for (const s of W.open) { const v = Math.max(...[0, 1, 2, 3].map((a) => W.qv(V, s, a, g))); if (Math.abs(v - V[s]) > d) { d = Math.abs(v - V[s]); at = s; } nV[s] = v; } sweeps.push({ V: nV, delta: d, at, prev: V }); V = nV; if (d < theta) break; } return { V, sweeps }; }
const argmax = (q) => { let b = 0; q.forEach((v, a) => { if (v > q[b] + 1e-12) b = a; }); return b; };
/** the most likely next cell, step by step, following a policy (as a greedy walk does) */
function walk(W, pol, max) { let s = W.start, ret = 0; const path = [s]; for (let t = 0; t < max; t++) { const o = W.T(s, pol[s]).reduce((b, x) => (x.p > b.p ? x : b)); ret += o.r; s = o.s2; path.push(s); if (o.done) return { path, steps: path.length - 1, ret, reached: W.ch(s) === 'G' }; } return { path, steps: path.length - 1, ret, reached: false }; }

const checks = {
  'rl-bandit': {
    random: (r) => ({ algo: ['epsilon', 'greedy', 'ucb', 'thompson'][Math.floor(r() * 4)], eps: +(0.01 + 0.49 * r()).toFixed(2), c: +(0.2 + 2.8 * r()).toFixed(1), k: 3 + Math.floor(r() * 6), steps: 50 + Math.floor(r() * 1950), seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const m0 = c.frames[0].meta, best = m0.means.indexOf(Math.max(...m0.means));
      c.eq(m0.best, best, 'the best arm');
      for (const f of c.frames.slice(1)) {
        const m = f.meta, k = m0.k, cnt = new Array(k).fill(0), sum = new Array(k).fill(0);
        m.choices.forEach((a, i) => { cnt[a]++; sum[a] += m.rewards[i]; });
        vecNear(c, m.est, sum.map((s, a) => (cnt[a] ? s / cnt[a] : 0)), 1e-9, 'the estimates are the averages of the rewards');
        vecNear(c, m.cnt, cnt, 1e-12, 'the pull counts');
        near(c, m.regret, m.choices.reduce((s, a) => s + (m0.means[best] - m0.means[a]), 0), 1e-9, 'the regret');
        if (m.phase === 'pull') {
          const t = m.t, pc = new Array(k).fill(0), ps = new Array(k).fill(0);
          m.choices.slice(0, t - 1).forEach((a, i) => { pc[a]++; ps[a] += m.rewards[i]; });
          vecNear(c, m.before, ps.map((s, a) => (pc[a] ? s / pc[a] : 0)), 1e-9, 'the estimates before the pull');
          if (m.ucb) { vecNear(c, m.ucb, m.before.map((e, a) => e + c.input.c * Math.sqrt((2 * Math.log(t)) / pc[a])), 1e-9, 'the upper bounds'); c.ok(m.ucb[m.a] >= Math.max(...m.ucb) - 1e-9, 'UCB picks a highest bound'); }
          if (m.draws) c.eq(m.a, m.draws.indexOf(Math.max(...m.draws)), 'Thompson pulls the highest draw');
        } else {
          near(c, m.share, (100 * m.choices.filter((a) => a === best).length) / m.t, 1e-9, 'the share of pulls on the best arm');
          c.eq(m.most, m.cnt.indexOf(Math.max(...m.cnt)), 'the arm pulled most');
        }
      }
    },
  },
  'rl-regret': {
    randomRuns: 2,
    random: (r) => ({ kind: r() < 0.5 ? 'bernoulli' : 'gaussian', k: 2 + Math.floor(r() * 10), eps: +(0.05 + 0.3 * r()).toFixed(2), c: +(0.5 + 2 * r()).toFixed(1), runs: 5 + Math.floor(r() * 10), steps: 100 + Math.floor(r() * 300) }),
    check(c) {
      const inp = c.input, algos = [{ algo: 'greedy' }, { algo: 'epsilon', eps: inp.eps }, { algo: 'ucb', c: inp.c }, { algo: 'thompson' }];
      const avg = algos.map((al, ai) => { const rg = new Float64Array(inp.steps), op = new Float64Array(inp.steps); for (let run = 0; run < inp.runs; run++) { const b = RL.bandit({ kind: inp.kind, k: inp.k, seed: 1000 + run + 1 }), r = RL.runBandit(b, Object.assign({}, al, { steps: inp.steps, seed: 7777 + run * 31 + ai, traceSteps: 0 })); let reg = 0; r.choices.forEach((a, t) => { reg += b.bestMean - b.means[a]; rg[t] += reg / inp.runs; op[t] += (a === b.best ? 1 : 0) / inp.runs; }); } return { rg, op }; });
      for (const f of c.frames) {
        vecNear(c, f.meta.reg, avg.map((x) => x.rg[f.meta.t - 1]), 1e-9, 'the average regrets at ' + f.meta.t);
        vecNear(c, f.meta.op, avg.map((x) => x.op[f.meta.t - 1]), 1e-9, 'the shares on the best arm at ' + f.meta.t);
      }
    },
  },
  'rl-mdp': {
    random: (r) => ({ slip: 0.05 * Math.floor(r() * 9), step: -0.01 * Math.floor(r() * 31), gamma: +(0.5 + 0.49 * r()).toFixed(2), act: String(Math.floor(r() * 4)), seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const env = S.P15.env('pits', { slip: c.input.slip, step: c.input.step, gamma: c.input.gamma }), W = world(env), g = c.input.gamma;
      const md = phase(c.frames, 'model')[0].meta, mine = W.T(12, +c.input.act), merged = new Map();
      for (const o of mine) { const k = o.s2 + '|' + o.r; merged.set(k, (merged.get(k) || 0) + o.p); }
      c.eq(md.outs.length, merged.size, 'the outcomes of the action');
      for (const [p, s2, r] of md.outs) near(c, p, merged.get(s2 + '|' + r) || -1, 1e-12, 'the chance of reaching ' + s2);
      near(c, md.outs.reduce((s, o) => s + o[0], 0), 1, 1e-12, 'the chances add up to 1');
      const ret = phase(c.frames, 'return')[0].meta;
      near(c, ret.G, ret.rewards.reduce((s, r, k) => s + Math.pow(g, k) * r, 0), 1e-12, 'the return');
      vecNear(c, ret.Gs, [0.5, 0.9, 0.99].map((gg) => ret.rewards.reduce((s, r, k) => s + Math.pow(gg, k) * r, 0)), 1e-12, 'the returns at other discounts');
      near(c, phase(c.frames, 'policy')[0].meta.vEast, solvePolicy(W, new Array(W.n).fill(1), g)[W.start], 1e-6, 'V of “always east”, solved exactly');
      near(c, phase(c.frames, 'best')[0].meta.vStar, valueIter(W, g, 1e-10).V[W.start], 1e-5, 'V* at the start');
    },
  },
  'rl-value-iteration': {
    random: (r) => ({ world: ['pits', 'rooms', 'cliff'][Math.floor(r() * 3)], gamma: +(0.5 + 0.49 * r()).toFixed(2), slip: 0.05 * Math.floor(r() * 9) }),
    check(c) {
      const env = S.P15.env(c.input.world, { slip: c.input.slip, gamma: c.input.gamma }), W = world(env), g = c.input.gamma, mine = valueIter(W, g);
      for (const f of phase(c.frames, 'sweep')) {
        const sw = mine.sweeps[f.meta.k - 1];
        near(c, f.meta.delta, sw.delta, 1e-9, 'the largest change in sweep ' + f.meta.k);
        // the cell named has the largest change (equal changes may be told apart by rounding)
        c.ok(Math.abs(Math.abs(sw.V[f.meta.cell] - sw.prev[f.meta.cell]) - sw.delta) < 1e-9, 'the cell named has the largest change');
        vecNear(c, f.meta.q, [0, 1, 2, 3].map((a) => W.qv(sw.prev, f.meta.cell, a, g)), 1e-9, 'the look-ahead values');
        c.ok(W.open.every((s) => rel(f.meta.V[s], sw.V[s], 1e-9)), 'the values after sweep ' + f.meta.k);
      }
      const d = phase(c.frames, 'done')[0].meta;
      c.eq(d.K, mine.sweeps.length, 'the sweeps to converge'); near(c, d.start, mine.V[W.start], 1e-9, 'V at the start');
      const pol = new Array(W.n).fill(0); for (const s of W.open) pol[s] = argmax([0, 1, 2, 3].map((a) => W.qv(mine.V, s, a, g)));
      const p = walk(W, pol, 200); c.eq([d.reached, d.reached ? d.steps : -1], [p.reached, p.reached ? p.steps : -1], 'the walk along the arrows');
      c.ok(W.open.every((s) => Math.abs(mine.V[s] - Math.max(...[0, 1, 2, 3].map((a) => W.qv(mine.V, s, a, g)))) < 1e-5), 'V satisfies the Bellman optimality equation');
    },
  },
  'rl-policy-iteration': {
    random: (r) => ({ world: ['pits', 'rooms', 'cliff'][Math.floor(r() * 3)], gamma: +(0.5 + 0.49 * r()).toFixed(2), slip: 0.05 * Math.floor(r() * 9), init: String(Math.floor(r() * 4)) }),
    check(c) {
      const env = S.P15.env(c.input.world, { slip: c.input.slip, gamma: c.input.gamma }), W = world(env), g = c.input.gamma;
      for (const f of phase(c.frames, 'eval')) { const V = solvePolicy(W, f.meta.policy, g); c.ok(W.open.every((s) => Math.abs(f.meta.V[s] - V[s]) < 1e-5), 'V^π of round ' + f.meta.k + ' against the exact solution'); near(c, f.meta.vStart, V[W.start], 1e-5, 'V^π at the start'); }
      for (const f of phase(c.frames, 'improve')) {
        const V = solvePolicy(W, f.meta.before, g);
        for (const s of W.open) { const q = [0, 1, 2, 3].map((a) => W.qv(V, s, a, g)), m = Math.max(...q); c.ok(q[f.meta.policy[s]] >= m - 1e-6, 'the improved action is among the best at ' + s); }
        c.eq(f.meta.changed, W.open.filter((s) => f.meta.policy[s] !== f.meta.before[s]).length, 'the arrows that changed');
      }
      const d = phase(c.frames, 'done')[0].meta; near(c, d.vPI, d.vVI, 1e-4, 'both methods agree'); near(c, d.vVI, valueIter(W, g).V[W.start], 1e-9, 'value iteration by hand');
    },
  },
  'rl-mc': {
    random: (r) => ({ world: r() < 0.6 ? 'pits' : 'rooms', eps: 0.05 * (1 + Math.floor(r() * 10)), episodes: 50 + Math.floor(r() * 400), seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const env = S.P15.env(c.input.world), W = world(env);
      for (const f of phase(c.frames, 'update')) near(c, f.meta.now, f.meta.old + (f.meta.G - f.meta.old) / 1, 1e-12, 'the first return becomes the average');
      for (const f of c.frames.filter((x) => x.meta.phase === 'snap' || x.meta.phase === 'end')) {
        const w = Math.min(20, f.meta.e); near(c, f.meta.avg, mean(f.meta.returns.slice(-w)), 1e-9, 'the average return');
        const p = walk(W, f.meta.policy, 60); c.eq(f.meta.reached, p.reached, 'the greedy walk reaches the goal or not'); if (p.reached) c.eq(f.meta.steps, p.steps, 'its length');
      }
    },
  },
  'rl-cliff': {
    random: (r) => ({ eps: +(0.02 + 0.28 * r()).toFixed(2), alpha: +(0.1 + 0.9 * r()).toFixed(2), episodes: 100 + Math.floor(r() * 400), seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const env = S.P15.env('cliff'), W = world(env), u = phase(c.frames, 'update')[0].meta, al = c.input.alpha;
      for (const x of [u.a, u.b]) { const [r, target, old, now] = x; near(c, target, r, 1e-12, 'the first target: every Q is still 0'); near(c, now, old + al * (target - old), 1e-12, 'Q moves by α of the error'); }
      for (const f of c.frames.filter((x) => x.meta.phase === 'snap' || x.meta.phase === 'end')) {
        const pa = walk(W, f.meta.polA, 100), pb = walk(W, f.meta.polB, 100);
        c.eq([f.meta.stepsA, f.meta.stepsB], [pa.reached ? pa.steps : -1, pb.reached ? pb.steps : -1], 'the greedy routes');
        near(c, f.meta.avgA, mean(f.meta.retA.slice(-f.meta.window)), 1e-9, 'Q-learning’s average'); near(c, f.meta.avgB, mean(f.meta.retB.slice(-f.meta.window)), 1e-9, 'SARSA’s average');
      }
    },
  },
  'rl-explore': {
    random: (r) => ({ eps: 0.05 * Math.floor(r() * 21), decay: ['none', '0.98', '0.95'][Math.floor(r() * 3)], episodes: 50 + Math.floor(r() * 300), seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      for (const f of c.frames) {
        const m = f.meta;
        c.eq(m.visits, m.lengths.reduce((a, b) => a + b, 0), 'every move is one visit');
        near(c, m.avg, mean(m.returns.slice(-m.window)), 1e-9, 'the average return');
        near(c, m.eps, c.input.decay === 'none' ? c.input.eps : Math.max(0.01, c.input.eps * Math.pow(+c.input.decay, m.e)), 1e-9, 'ε after ' + m.e + ' episodes');
      }
    },
  },
  'rl-reinforce': {
    random: (r) => ({ seed: 1 + Math.floor(r() * 999) }),
    check(c) {
      const ph = { g: 9.8, mc: 1, mp: 0.1, l: 0.5, f: 10, tau: 0.02, xl: 2.4, th: (12 * Math.PI) / 180 }, tot = ph.mc + ph.mp, pml = ph.mp * ph.l;
      const acc = (s, f) => { const c2 = Math.cos(s[2]), sn = Math.sin(s[2]), tmp = (f + pml * s[3] * s[3] * sn) / tot, thA = (ph.g * sn - c2 * tmp) / (ph.l * (4 / 3 - (ph.mp * c2 * c2) / tot)); return { thA, xA: tmp - (pml * thA * c2) / tot }; };
      const st = phase(c.frames, 'start')[0].meta;
      near(c, st.thAcc, acc(st.st0, 10).thA, 1e-9, 'the angular acceleration of a push');
      const up = phase(c.frames, 'update')[0].meta;
      near(c, up.G0, (1 - Math.pow(0.99, up.L0)) / (1 - 0.99), 1e-9, 'the return of the first episode');
      for (const f of c.frames.filter((x) => x.meta.phase === 'snap' || x.meta.phase === 'end')) {
        const sn = D.reinforce.snaps.find((x) => x.update === f.meta.update), [W1, b1, W2, b2] = sn.policy, h = b1.length;
        const pol = (s) => { const x = [s[0] / 2.4, s[1] / 3, s[2] / 0.21, s[3] / 3], hid = b1.map((b, j) => Math.tanh(b + x.reduce((t, v, i) => t + v * W1[i * h + j], 0))), z = b2.map((b, j) => b + hid.reduce((t, v, i) => t + v * W2[i * 2 + j], 0)), m = Math.max(...z), e = z.map((v) => Math.exp(v - m)); return e[1] / (e[0] + e[1]); };
        const g = S.T.rng(c.input.seed); let s = [0, 0, 0, 0].map(() => -0.05 + 0.1 * g.next()), t = 0, mx = Math.abs(s[2]), fell = false;
        const p0 = pol(s);
        for (;;) { const a = pol(s) > 0.5 ? 1 : 0, { thA, xA } = acc(s, a ? ph.f : -ph.f); s = [s[0] + ph.tau * s[1], s[1] + ph.tau * xA, s[2] + ph.tau * s[3], s[3] + ph.tau * thA]; t++; mx = Math.max(mx, Math.abs(s[2])); fell = Math.abs(s[0]) > ph.xl || Math.abs(s[2]) > ph.th; if (fell || t >= 500) break; }
        c.eq([f.meta.len, f.meta.fell], [t, fell], 'the greedy episode after ' + f.meta.update + ' updates, simulated by hand');
        near(c, f.meta.maxTh, (mx * 180) / Math.PI, 1e-6, 'the largest lean'); near(c, f.meta.p, p0, 1e-5, 'the first push’s probability');
        if (f.meta.update) near(c, f.meta.mean, D.reinforce.meanLen[f.meta.update - 1], 1e-12, 'the training lengths');
      }
    },
  },
  'rl-dqn': {
    check(c) {
      const env = S.P15.env('pits'), W = world(env), g = D.dqn.gamma;
      for (const f of phase(c.frames, 'snap')) {
        const sn = D.dqn.snaps.find((x) => x.step === f.meta.t), pol = sn.Q.map((q) => argmax(q)), p = walk(W, pol, 30);
        near(c, f.meta.eps, Math.max(0.05, 1 - (0.95 * (f.meta.t - 1)) / 1200), 1e-5, 'ε at step ' + f.meta.t);
        c.eq(f.meta.reached, p.reached, 'the greedy arrows reach the goal or not'); if (p.reached) c.eq(f.meta.steps, p.steps, 'the length of that path');
      }
      for (const f of phase(c.frames, 'batch')) {
        const b = D.dqn.batches.find((x) => x.step === f.meta.t), r0 = b.rows[0], mq = r0.done ? 0 : Math.max(...r0.qt);
        near(c, f.meta.mq, mq, 1e-12, 'the target network’s best value'); near(c, f.meta.y, r0.r + g * mq, 1e-12, 'the target y'); near(c, f.meta.qsa, r0.q[r0.a], 1e-12, 'Q(s, a)');
        for (const r of b.rows) c.ok(r.done ? W.terminal(r.s2) || W.ch(r.s2) === 'G' || W.ch(r.s2) === 'X' : true, 'an ended transition lands on the goal or a pit');
      }
      const end = phase(c.frames, 'end')[0].meta; near(c, end.avg, mean(end.rets), 1e-9, 'the average of the last returns');
    },
  },
};

run('claims-p15', /^p15-/, checks, ALLOWED);
