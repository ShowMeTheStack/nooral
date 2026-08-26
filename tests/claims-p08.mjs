// Part 8, optimisation: the narration of every topic read back against an independent working-out. Every run is
// replayed with hand-written update rules (tests/claims-p08-replay.mjs), every gradient and Hessian the topics use is
// checked by finite differences, schedules against their formulas, eigenvalues from the characteristic polynomial.
// Usage: node tests/claims-p08.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';
import { replay, SCHED, fdGrad, fdHess, eigs, rel, dist } from './claims-p08-replay.mjs';

const P = S.P08;
// formula constants the narration may write as they are
const ALLOWED = [/1 − ηλmax/g, /2\/λmax/g, /loss 0\.5/g, /ceiling of 1\b/g, /Step 1\b/g, /about 10 times/g];
const RT = 1e-9;
const surfF = (s) => (p) => s.f(p[0], p[1]);
const surfG = (s) => (p) => s.g(p[0], p[1]);
const vecNear = (c, a, b, msg, tol) => c.ok(a && b && a.length === b.length && a.every((v, i) => rel(v, b[i], tol || RT) || Math.abs(v - b[i]) < (tol || RT)), msg + ': got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b));
const near = (c, a, b, msg, tol) => c.ok(rel(a, b, tol || RT), msg + ': got ' + a + ', want ' + b);
const ownMin = (s, n = 120) => { let m = Infinity; for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) m = Math.min(m, s.f(s.xr[0] + ((s.xr[1] - s.xr[0]) * i) / n, s.yr[0] + ((s.yr[1] - s.yr[0]) * j) / n)); return m; };

// ---- once: every surface's gradient and Hessian against finite differences --------------------------------------
{
  const T = { ok: 0, bad: [] };
  for (const [id, s] of Object.entries(P.SURF)) for (let k = 0; k < 25; k++) {
    const p = [s.xr[0] + ((s.xr[1] - s.xr[0]) * ((k * 7) % 25 + 0.5)) / 25, s.yr[0] + ((s.yr[1] - s.yr[0]) * ((k * 11) % 25 + 0.5)) / 25];
    const g = s.g(...p), fg = fdGrad(surfF(s), p), h = s.h(...p), fh = fdHess(surfG(s), p);
    if (g.every((v, i) => Math.abs(v - fg[i]) < 1e-5 * Math.max(1, Math.abs(v)))) T.ok++; else T.bad.push(id + ' gradient at ' + p + ': ' + g + ' vs ' + fg);
    if (h.every((r, i) => r.every((v, j) => Math.abs(v - fh[i][j]) < 1e-4 * Math.max(1, Math.abs(v))))) T.ok++; else T.bad.push(id + ' Hessian at ' + p);
  }
  for (const [id, L] of Object.entries(P.F1)) for (let x = -2.9; x < 3; x += 0.37) {
    const d = (L.f(x + 1e-6) - L.f(x - 1e-6)) / 2e-6, d2 = (L.d(x + 1e-6) - L.d(x - 1e-6)) / 2e-6;
    if (Math.abs(d - L.d(x)) < 1e-5 && Math.abs(d2 - L.d2(x)) < 1e-4) T.ok++; else T.bad.push(id + ' derivatives at ' + x);
  }
  console.log('  gradients and Hessians by finite differences: ' + T.ok + ' agree' + (T.bad.length ? '; FAIL ' + T.bad.slice(0, 5).join('; ') : ''));
  if (T.bad.length) process.exitCode = 1;
}

const checks = {
  'opt-surface': {
    check(c) {
      const s = P.SURF[c.input.surf], f0 = c.frames[0].meta;
      near(c, f0.lo, ownMin(s), 'the lowest value over the box');
      let hi = -Infinity;
      for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) hi = Math.max(hi, s.f(s.xr[0] + ((s.xr[1] - s.xr[0]) * i) / 40, s.yr[0] + ((s.yr[1] - s.yr[0]) * j) / 40));
      near(c, f0.hi, hi, 'the highest value over the box');
      const ar = phase(c.frames, 'arrows')[0].meta;
      let gm = 0;
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { const g = fdGrad(surfF(s), [s.xr[0] + ((i + 0.5) * (s.xr[1] - s.xr[0])) / 5, s.yr[0] + ((j + 0.5) * (s.yr[1] - s.yr[0])) / 5]); gm = Math.max(gm, Math.hypot(...g)); }
      near(c, ar.gmax, gm, 'the steepest arrow', 1e-5);
      const pt = phase(c.frames, 'point')[0].meta, g = fdGrad(surfF(s), pt.p);
      near(c, pt.f, s.f(...pt.p), 'the loss at the point');
      vecNear(c, pt.g, g, 'the gradient at the point', 1e-5);
      near(c, pt.gl, Math.hypot(...g), 'its length', 1e-5);
      const mn = phase(c.frames, 'min')[0].meta;
      c.ok(Math.hypot(...fdGrad(surfF(s), mn.m)) < 1e-6, 'the gradient vanishes at the lowest point');
      c.ok(mn.f <= ownMin(s) + 1e-9, 'nothing on the grid is lower than the lowest point');
      vecNear(c, mn.ev, eigs(fdHess(surfG(s), mn.m)), 'the curvatures at the lowest point', 1e-5);
      if (mn.ratio) near(c, mn.ratio, mn.ev[0] / mn.ev[1], 'their ratio');
      if (mn.sp) { c.ok(Math.hypot(...fdGrad(surfF(s), mn.sp)) < 1e-6, 'the saddle is flat'); const e = eigs(fdHess(surfG(s), mn.sp)); vecNear(c, mn.se, e, 'the saddle’s curvatures', 1e-5); c.ok(e[0] > 0 && e[1] < 0, 'one up, one down'); }
    },
  },
  'opt-gd': {
    random: (r) => ({ surf: ['bowl', 'ravine', 'tilted', 'rosen', 'saddle'][Math.floor(r() * 5)], lr: +(0.001 * Math.pow(2000, r())).toFixed(3), steps: 3 + Math.floor(r() * 50) }),
    check(c) {
      const s = P.SURF[c.input.surf], R = replay('sgd', { lr: c.input.lr }, surfF(s), surfG(s), s.start, c.input.steps);
      for (const f of c.frames) {
        const m = f.meta, k = m.k;
        if (m.phase === 'start') { vecNear(c, m.p, R.pts[0], 'the start'); near(c, m.loss, R.loss[0], 'the first loss'); continue; }
        near(c, m.loss, R.loss[k], 'the loss after step ' + k);
        if (m.phase !== 'step') continue;
        vecNear(c, m.p, R.pts[k], 'the point after step ' + k);
        vecNear(c, m.g, R.grads[k - 1], 'the gradient of step ' + k);
        const d = [R.pts[k][0] - R.pts[k - 1][0], R.pts[k][1] - R.pts[k - 1][1]];
        vecNear(c, m.d, d, 'the step, η times the gradient', 1e-7);
        vecNear(c, d, R.grads[k - 1].map((v) => -c.input.lr * v), 'the step is −η∇L', 1e-7);
        near(c, m.step, Math.hypot(...d), 'the step’s length', 1e-7);
        near(c, m.prev, R.loss[k - 1], 'the loss before');
      }
    },
  },
  'opt-lr': {
    random: (r) => ({ surf: ['ravine', 'tilted', 'bowl'][Math.floor(r() * 3)], lr: +(0.01 * Math.pow(250, r())).toFixed(2), steps: 5 + Math.floor(r() * 50) }),
    check(c) {
      const s = P.SURF[c.input.surf], H0 = fdHess(surfG(s), [0.3, -0.2]), [lmax, lmin] = eigs(H0), m0 = c.frames[0].meta;
      near(c, m0.lmax, lmax, 'λmax', 1e-6); near(c, m0.lmin, lmin, 'λmin', 1e-6); near(c, m0.limit, 2 / lmax, 'the limit', 1e-6);
      vecNear(c, m0.lrs, [0.15 / lmax, 2 / (lmax + lmin), 2.12 / lmax, c.input.lr], 'the four rates', 1e-6);
      const R = m0.lrs.map((lr) => replay('sgd', { lr }, surfF(s), surfG(s), s.start, c.input.steps));
      for (const f of c.frames) {
        const k = f.meta.k;
        vecNear(c, f.meta.losses, R.map((r) => r.loss[Math.min(k, r.loss.length - 1)]), 'the losses at step ' + k, 1e-7);
        vecNear(c, f.meta.f1, m0.lrs.map((lr) => 1 - lr * lmax), 'the factors 1 − ηλmax', 1e-6);
        vecNear(c, f.meta.fac, [Math.abs(1 - m0.lrs[1] * lmax), Math.abs(1 - m0.lrs[1] * lmin)], 'the balanced factors', 1e-6);
        near(c, f.meta.fac[0], f.meta.fac[1], 'the balanced rate shrinks both directions alike', 1e-6);
      }
      const end = phase(c.frames, 'end')[0].meta;
      c.ok(end.losses[2] > R[2].loss[0], 'the rate past the limit ends higher than it started');
      c.ok(end.losses[1] <= Math.min(end.losses[0], end.losses[2]), 'the balanced rate beats the other two');
    },
  },
  'opt-sgd': {
    random: (r) => ({ B: 2 + Math.floor(r() * 19), lr: +(0.02 + 0.5 * r()).toFixed(2), steps: 10 + Math.floor(r() * 60), seed: 1 + Math.floor(r() * 300) }),
    check(c) {
      const n = 40, D = P.lsq(n, c.input.seed), X = D.X, Y = D.Y, N = c.input.steps, B = c.input.B;
      // the least-squares fit from the normal equations, solved here
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      X.forEach((x, i) => { sx += x; sy += Y[i]; sxx += x * x; sxy += x * Y[i]; });
      const a = (n * sxy - sx * sy) / (n * sxx - sx * sx), b = (sy - a * sx) / n;
      const f = (p) => X.reduce((s, x, i) => s + (p[0] * x + p[1] - Y[i]) ** 2, 0) / (2 * n);
      const gOf = (p, idx) => { let ga = 0, gb = 0; for (const i of idx) { const res = p[0] * x(i) + p[1] - Y[i]; ga += res * x(i); gb += res; } return [ga / idx.length, gb / idx.length]; };
      const x = (i) => X[i];
      const all = [...Array(n).keys()];
      const m0 = phase(c.frames, 'start')[0].meta;
      vecNear(c, m0.best, [a, b], 'the least-squares fit', 1e-8);
      near(c, m0.fmin, f([a, b]), 'its loss', 1e-8);
      for (const id of ['mini', 'sgd']) {
        const bs = P.batches(n, id === 'mini' ? B : 1, N, c.input.seed * 7 + (id === 'mini' ? 1 : 2));
        // each pass through the data uses every point once
        const flat = bs.flat().slice(0, Math.floor((bs.flat().length) / n) * n);
        for (let p = 0; p < flat.length; p += n) c.ok(new Set(flat.slice(p, p + n)).size === n, 'a pass through the data is a permutation');
        if (id === 'mini') c.mini = bs; else c.one = bs;
      }
      const start = D.surf.start;
      const R = [replay('sgd', { lr: c.input.lr }, f, (p) => gOf(p, all), start, N), replay('sgd', { lr: c.input.lr }, f, (p, k) => gOf(p, c.mini[k]), start, N), replay('sgd', { lr: c.input.lr }, f, (p, k) => gOf(p, c.one[k]), start, N)];
      for (const fr of c.frames) {
        const m = fr.meta, k = m.k;
        if (m.phase === 'start') continue;
        vecNear(c, m.losses, R.map((r) => r.loss[k]), 'the three losses at step ' + k, 1e-7);
        vecNear(c, m.gm, R[1].grads[k - 1], 'the mini-batch gradient', 1e-7);
        vecNear(c, m.gf, gOf(R[1].pts[k - 1], all), 'the full gradient at the same point', 1e-7);
        vecNear(c, m.gf, fdGrad(f, R[1].pts[k - 1]), 'the full gradient by finite differences', 1e-5);
        const cos = (m.gm[0] * m.gf[0] + m.gm[1] * m.gf[1]) / (Math.hypot(...m.gm) * Math.hypot(...m.gf));
        near(c, m.angle, (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI, 'the angle between them', 1e-7);
        c.eq(m.batch, c.mini[k - 1], 'the batch drawn');
      }
    },
  },
  'opt-momentum': {
    random: (r) => ({ surf: ['ravine', 'tilted', 'rosen'][Math.floor(r() * 3)], lr: +(0.001 * Math.pow(400, r())).toFixed(3), beta: +(0.98 * r()).toFixed(2), steps: 5 + Math.floor(r() * 70) }),
    check(c) {
      const s = P.SURF[c.input.surf], { lr, beta, steps } = c.input;
      const R = [replay('sgd', { lr }, surfF(s), surfG(s), s.start, steps), replay('sgd', { lr, momentum: beta }, surfF(s), surfG(s), s.start, steps), replay('sgd', { lr, momentum: beta, nesterov: true }, surfF(s), surfG(s), s.start, steps)];
      for (const f of c.frames) {
        const k = f.meta.k;
        vecNear(c, f.meta.losses, R.map((r) => r.loss[Math.min(k, r.loss.length - 1)]), 'the losses at step ' + k, 1e-7);
        if (f.meta.vel) vecNear(c, f.meta.vel, R[1].state[k - 1].v, 'momentum’s velocity after step ' + k, 1e-7);
      }
    },
  },
  'opt-adaptive': {
    random: (r) => ({ method: ['adagrad', 'rmsprop', 'adam'][Math.floor(r() * 3)], surf: ['ravine', 'tilted', 'rosen', 'plateau'][Math.floor(r() * 4)], lr: +(0.01 * Math.pow(100, r())).toFixed(2), steps: 5 + Math.floor(r() * 70) }),
    check(c) {
      const s = P.SURF[c.input.surf], { lr, steps, method } = c.input, m0 = c.frames[0].meta;
      near(c, m0.gdLr, Math.min(lr, 1.8 / eigs(fdHess(surfG(s), s.start))[0]), 'the rate given to plain descent', 1e-5);
      const R = [replay('sgd', { lr: m0.gdLr }, surfF(s), surfG(s), s.start, steps)].concat(['adagrad', 'rmsprop', 'adam'].map((id) => replay(id, { lr }, surfF(s), surfG(s), s.start, steps)));
      const me = R[1 + ['adagrad', 'rmsprop', 'adam'].indexOf(method)], acc = { adagrad: 'sumSq', rmsprop: 'sqAvg', adam: 'v' }[method];
      for (const f of c.frames) {
        const m = f.meta, k = m.k;
        vecNear(c, m.losses, R.map((r) => r.loss[Math.min(k, r.loss.length - 1)]), 'the losses at step ' + k, 1e-7);
        if (!k) continue;
        vecNear(c, m.g, me.grads[k - 1], 'the gradient', 1e-7);
        vecNear(c, m.d, [me.pts[k][0] - me.pts[k - 1][0], me.pts[k][1] - me.pts[k - 1][1]], 'the step', 1e-7);
        vecNear(c, m.acc, me.state[k - 1][acc], 'the accumulator of g²', 1e-7);
        if (method === 'adam') vecNear(c, m.m, me.state[k - 1].m, 'Adam’s first moment', 1e-7);
        if (m.ratio !== undefined) near(c, m.ratio, dist(R[2].pts[1], R[2].pts[0]) / dist(R[3].pts[1], R[3].pts[0]), 'RMSProp’s first step against Adam’s', 1e-7);
        if (k === 1 && method === 'adam') vecNear(c, m.d.map(Math.abs), [lr, lr], 'Adam’s first step is η in each coordinate', 1e-4);
      }
    },
  },
  'opt-race': {
    extra: [{ surf: 'ravine', steps: 40 }],
    check(c) {
      const s = P.SURF[c.input.surf], N = c.input.steps, m0 = c.frames[0].meta, f = surfF(s), g = surfG(s), start = m0.start;
      c.ok(Math.hypot(...fdGrad(f, m0.m)) < 1e-6 && m0.Lstar <= ownMin(s) + 1e-9, 'the lowest point is flat and lowest');
      // every optimiser's rate is the best of the grid by the mean log loss, with strays and divergences ruled out
      const opts = { sgd: {}, momentum: { momentum: 0.9 }, nesterov: { momentum: 0.9, nesterov: true }, adagrad: {}, rmsprop: {}, adam: {} };
      const names = ['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam'];
      const R = names.map((id, j) => {
        let best = null;
        for (const lr of [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]) {
          const r = replay(id === 'momentum' || id === 'nesterov' ? 'sgd' : id, Object.assign({ lr }, opts[id]), f, g, start, N);
          const wild = r.pts.some((p) => p[0] < s.xr[0] - 1 || p[0] > s.xr[1] + 1 || p[1] < s.yr[0] - 1 || p[1] > s.yr[1] + 1);
          const sc = r.diverged || wild ? Infinity : r.loss.reduce((a, v) => a + Math.log10(Math.max(1e-12, v - m0.Lstar)), 0) / r.loss.length;
          if (!best || sc < best.sc) best = { lr, sc, r };
        }
        c.ok(best.lr === m0.lrs[j], id + ': the best rate of the grid is ' + best.lr + ', the topic says ' + m0.lrs[j]);
        return best.r;
      });
      for (const fr of c.frames) {
        const k = fr.meta.k, L = R.map((r) => r.loss[Math.min(k, r.loss.length - 1)]);
        vecNear(c, fr.meta.losses, L, 'the six losses at step ' + k, 1e-7);
        if (fr.meta.phase === 'end') {
          vecNear(c, fr.meta.gaps, L.map((v) => v - m0.Lstar), 'the gaps to the lowest', 1e-6);
          c.eq(fr.meta.nfar, R.filter((r) => dist(r.pts[r.pts.length - 1], m0.m) > 0.3).length, 'the runs that ended elsewhere');
        }
      }
    },
  },
  'opt-schedules': {
    random: (r) => ({ focus: ['constant', 'step', 'exp', 'cosine'][Math.floor(r() * 4)], lr: +(0.05 + 0.55 * r()).toFixed(2), steps: 15 + Math.floor(r() * 100), seed: 1 + Math.floor(r() * 500) }),
    check(c) {
      const { lr, steps: N, seed } = c.input, D = P.lsq(40, seed), order = P.batches(40, 1, N, seed * 11 + 5);
      const f = (p) => D.X.reduce((s, x, i) => s + (p[0] * x + p[1] - D.Y[i]) ** 2, 0) / 80;
      const g = (p, k) => { const i = order[k][0], res = p[0] * D.X[i] + p[1] - D.Y[i]; return [res * D.X[i], res]; };
      const sch = [SCHED.constant(lr), SCHED.step(lr, Math.ceil(N / 3), 0.3), SCHED.exp(lr, Math.pow(0.02, 1 / N)), SCHED.cosine(lr, N)];
      const R = sch.map((fn) => replay('sgd', { lr }, f, g, D.surf.start, N, fn));
      near(c, c.frames[0].meta.Ls, f(D.best), 'the best fit’s loss', 1e-8);
      for (const fr of c.frames) {
        const k = fr.meta.k;
        vecNear(c, fr.meta.lrs, sch.map((fn) => fn(Math.max(0, k - 1))), 'the rates at step ' + k, 1e-9);
        vecNear(c, fr.meta.losses, R.map((r) => r.loss[k]), 'the losses at step ' + k, 1e-7);
        if (fr.meta.tail) {
          const a = Math.max(1, Math.floor(k * 0.8));
          vecNear(c, fr.meta.tail, R.map((r) => Math.max(...r.pts.slice(a, k + 1).map((p) => dist(p, D.best)))), 'how far each strays at the end', 1e-7);
        }
      }
    },
  },
  'opt-warmup': {
    random: (r) => { const steps = 10 + Math.floor(r() * 100); return { surf: ['rosen', 'ravine', 'tilted'][Math.floor(r() * 3)], lr: +(0.05 + 1.4 * r()).toFixed(2), warm: 2 + Math.floor(r() * Math.min(38, steps - 3)), steps }; },
    check(c) {
      const s = P.SURF[c.input.surf], { lr, warm: W, steps: N } = c.input;
      const R = [replay('adam', { lr }, surfF(s), surfG(s), s.start, N), replay('adam', { lr }, surfF(s), surfG(s), s.start, N, SCHED.warmup(lr, W))];
      for (const fr of c.frames) {
        const k = fr.meta.k;
        vecNear(c, fr.meta.losses, R.map((r) => r.loss[k]), 'the losses at step ' + k, 1e-7);
        vecNear(c, fr.meta.rates, [lr, SCHED.warmup(lr, W)(Math.max(0, k - 1))], 'the rates', 1e-9);
        vecNear(c, fr.meta.maxes, R.map((r) => Math.max(...r.loss.slice(0, k + 1))), 'the worst losses so far', 1e-7);
        if (fr.meta.lens) vecNear(c, fr.meta.lens, R.map((r) => dist(r.pts[k], r.pts[k - 1])), 'the step lengths', 1e-7);
      }
    },
  },
  'opt-convex': {
    check(c) {
      const L = P.F1[c.input.fn];
      for (const fr of phase(c.frames, 'chord')) {
        const { a, b } = fr.meta;
        let gap = -Infinity;
        for (let i = 1; i < 4000; i++) { const t = i / 4000, x = a + (b - a) * t; gap = Math.max(gap, L.f(x) - (L.f(a) + (L.f(b) - L.f(a)) * t)); }
        if (gap > 1e-9) c.ok(Math.abs(fr.meta.gap - gap) < 1e-4 * Math.max(1, Math.abs(gap)), 'the chord’s largest gap: ' + fr.meta.gap + ' vs ' + gap);
        c.ok((fr.meta.gap > 1e-9) === (gap > 1e-9), 'whether the chord fails');
        near(c, L.f(fr.meta.at) - (L.f(a) + ((L.f(b) - L.f(a)) * (fr.meta.at - a)) / (b - a)), fr.meta.gap, 'the gap at its place', 1e-9);
      }
      const d2 = phase(c.frames, 'd2')[0].meta, fd2 = (x) => (L.f(x + 1e-4) - 2 * L.f(x) + L.f(x - 1e-4)) / 1e-8;
      let mn = Infinity;
      for (let i = 0; i <= 240; i++) mn = Math.min(mn, fd2(-3 + i / 40));
      c.ok(Math.abs(d2.min - mn) < 1e-4 * Math.max(1, Math.abs(mn)), 'the least curvature: ' + d2.min + ' vs ' + mn);
      near(c, fd2(d2.at), d2.min, 'where it is least', 1e-4);
      const rr = phase(c.frames, 'runs')[0].meta, f1 = (p) => L.f(p[0]), g1 = (p) => [L.d(p[0]), 0];
      const ends = rr.starts.map((s0) => { const r = replay('sgd', { lr: rr.lr }, f1, g1, [s0, 0], 300); return r.pts[r.pts.length - 1][0]; });
      vecNear(c, rr.ends, ends, 'where the two descents end', 1e-7);
      vecNear(c, rr.fe, ends.map(L.f), 'their losses', 1e-7);
      let gx = 0, gv = Infinity;
      for (let i = 0; i <= 60000; i++) { const x = -3 + i / 10000; if (L.f(x) < gv) { gv = L.f(x); gx = x; } }
      if (rr.gx !== undefined) c.ok(Math.abs(rr.gx - gx) < 1e-3, 'the lowest point of all: ' + rr.gx + ' vs ' + gx);
      c.eq(rr.same, Math.abs(ends[0] - ends[1]) < 1e-3, 'whether the two ends agree');
    },
  },
  'opt-local': {
    check(c) {
      const s = P.SURF.bumpy, f = surfF(s), g = surfG(s), m0 = c.frames[0].meta, K = c.input.k, lr = c.input.lr;
      for (const p of m0.pits) c.ok(Math.hypot(...fdGrad(f, p)) < 1e-6 && eigs(fdHess(g, p)).every((e) => e > 0), 'each pit is a true minimum');
      vecNear(c, m0.fp, m0.pits.map(f), 'the pits’ losses');
      c.ok(m0.fp.every((v, i) => !i || v >= m0.fp[i - 1]), 'deepest first');
      c.ok(m0.fp[0] <= ownMin(s) + 1e-9, 'the deepest pit is the lowest point');
      const pitOf = (p) => { let b = -1, bd = 0.35; m0.pits.forEach((m, j) => { const d = dist(p, m); if (d < bd) { bd = d; b = j; } }); return b; };
      const starts = [];
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) starts.push([-2.6 + (5.2 * (i + 0.5)) / K, -2.6 + (5.2 * (j + 0.5)) / K]);
      const R = starts.map((s0) => replay('sgd', { lr }, f, g, s0, 120));
      for (const fr of c.frames) {
        const m = fr.meta;
        if (m.phase === 'step' || m.phase === 'end') c.eq(m.settled, R.filter((r) => pitOf(r.pts[m.k]) >= 0).length, 'the runs settled by step ' + m.k);
        if (m.phase === 'end') {
          c.eq(m.counts, m0.pits.map((_, j) => R.filter((r) => pitOf(r.pts[120]) === j).length), 'the runs in each pit');
          near(c, m.best, Math.min(...R.map((r) => r.loss[120])), 'the best end', 1e-7);
        }
        if (m.phase === 'basins') {
          const n = 30, cnt = m0.pits.map(() => 0);
          for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const r = replay('sgd', { lr }, f, g, [-3 + (6 * (i + 0.5)) / n, 3 - (6 * (j + 0.5)) / n], 90), b = pitOf(r.pts[90]); if (b >= 0) cnt[b]++; }
          vecNear(c, m.area, cnt.map((v) => v / (n * n)), 'each basin’s share of the map', 1e-9);
        }
      }
    },
  },
  'opt-plateau': {
    random: (r) => ({ u0: +(2 + 1.9 * r()).toFixed(1), lr: +(0.1 + 1.4 * r()).toFixed(2), steps: 30 + Math.floor(r() * 200) }),
    check(c) {
      const s = P.SURF.plateau, { u0, lr, steps: N } = c.input, start = [u0, 0.9], f = surfF(s), g = surfG(s);
      const R = [replay('sgd', { lr }, f, g, start, N), replay('sgd', { lr, momentum: 0.9 }, f, g, start, N), replay('adam', { lr: 0.1 }, f, g, start, N)];
      for (const fr of c.frames) {
        const m = fr.meta, k = m.k;
        vecNear(c, m.losses, R.map((r) => r.loss[k]), 'the losses at step ' + k, 1e-7);
        vecNear(c, m.moved, R.map((r) => dist(r.pts[k], start)), 'the distances moved', 1e-7);
        c.eq(m.off, R.map((r) => r.loss.findIndex((v) => v < 0.5)), 'the step each gets below 0.5');
        if (m.phase === 'start') { vecNear(c, m.g, fdGrad(f, start), 'the gradient at the start', 1e-5); near(c, m.f0, f(start), 'the loss at the start'); }
      }
    },
  },
};
run('claims-p08', /^p08-/, checks, ALLOWED);
