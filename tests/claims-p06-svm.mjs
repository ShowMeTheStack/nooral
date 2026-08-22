// The per-topic claims of part 6. Each check recomputes what the frames claim by a method of its own: the widest
// street by scanning every direction, slacks, margins and objectives from the weights, kernel values by building
// the lifted features, every SMO step by solving its two-variable problem afresh, and the machines themselves
// fitted again by scikit-learn (tests/ref/p06/sk.json, made by tests/ref/gen_p06.mjs).
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { phase } from './claims-lib-p06-08.mjs';

const REF = join(dirname(fileURLToPath(import.meta.url)), 'ref', 'p06', 'sk.json');
const SK = existsSync(REF) ? JSON.parse(readFileSync(REF, 'utf8')) : {};
const dot = (a, b) => a.reduce((s, v, j) => s + v * b[j], 0);
const sum = (a) => a.reduce((s, v) => s + v, 0);
const acc = (y, p) => y.filter((v, i) => v === p[i]).length / y.length;
const maxDiff = (a, b) => Math.max(...a.map((v, i) => (Array.isArray(v) ? maxDiff(v, b[i]) : Math.abs(v - b[i]))));

/** the widest street that splits classes 0 and 1, found by scanning directions (no SVM involved): full width */
export function widestStreet(X, y) {
  const m = (th) => { const n = [Math.cos(th), Math.sin(th)]; let a = -Infinity, b = Infinity; X.forEach((x, i) => { const s = n[0] * x[0] + n[1] * x[1]; if (y[i] === 0) a = Math.max(a, s); else b = Math.min(b, s); }); return b - a; };
  let best = -Infinity, bt = 0;
  for (let k = 0; k < 7200; k++) { const th = (2 * Math.PI * k) / 7200, v = m(th); if (v > best) { best = v; bt = th; } }
  let lo = bt - 0.002, hi = bt + 0.002;
  for (let it = 0; it < 80; it++) { const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3; if (m(a) < m(b)) lo = a; else hi = b; }
  return Math.max(best, m((lo + hi) / 2));
}
/** kernels written out on their own */
const kern = (p, gamma) => (p.kernel === 'linear' ? (a, b) => dot(a, b) : p.kernel === 'poly' ? (a, b) => (gamma * dot(a, b) + p.coef0) ** p.degree : (a, b) => Math.exp(-gamma * a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0)));

export function svmChecks(S) {
  const P = S.P06;
  /** the reference fits of a run, in the order the topic fitted them, and a check of every one against the engine */
  const refs = (c) => {
    const fits = (c.result.result && c.result.result.fits) || [];
    return fits.map((f, k) => {
      const r = SK[c.t.id + '|' + JSON.stringify(c.input) + '|' + k];
      if (!r) return { f, r: null };
      const scale = Math.max(1, ...((r.dec || r.pred || [1]).flat().map(Math.abs)));
      if (f.kind === 'svc' && f.dec && r.dec) c.ok(maxDiff(f.dec, r.dec) < 2e-3 * scale, 'fit ' + k + ': decision values agree with the reference (largest gap ' + maxDiff(f.dec, r.dec).toExponential(2) + ')');
      if (f.pred && r.pred) c.ok(maxDiff(f.pred, r.pred) < (f.kind === 'svr' ? 2e-3 * scale : 0.5) || acc(f.pred, r.pred) >= 1 - 2 / f.pred.length, 'fit ' + k + ': predictions agree with the reference');
      if (f.decE && r.dec_e) c.ok(maxDiff(f.decE, r.dec_e) < 2e-3 * scale, 'fit ' + k + ': decision values at the extra points agree');
      if (f.kind === 'ocsvm') c.ok(maxDiff(f.dec, r.dec) < 2e-3 * scale, 'fit ' + k + ': one-class decision values agree');
      return { f, r };
    });
  };
  const hasRef = (c) => Object.keys(SK).some((k) => k.startsWith(c.t.id + '|' + JSON.stringify(c.input) + '|'));
  /** support sets may differ only by points that lie (within the solvers' rounding) exactly on a margin */
  const sameSupport = (c, a, b, onEdge, what) => { const A = new Set(a), B = new Set(b), diff = [...A].filter((i) => !B.has(i)).concat([...B].filter((i) => !A.has(i))); c.ok(diff.every(onEdge), what + ': support vectors agree with the reference (differing: ' + diff.join(', ') + ')'); };

  return {
    'svm-max-margin': {
      random: (r) => ({ set: r() < 0.5 ? 'apart' : 'close', n: 8 + Math.floor(r() * 40), seed: 1 + Math.floor(r() * 200) }),
      randomRuns: 8,
      check(c) {
        const d = P.data(c.input.set, { n: c.input.n, seed: c.input.seed }), X = d.X, y = d.y;
        refs(c);
        for (const f of phase(c.frames, 'cand')) {
          const n = [Math.cos(f.meta.th), Math.sin(f.meta.th)], s = X.map((x) => dot(n, x));
          const a = Math.max(...s.filter((_, i) => y[i] === 0)), b = Math.min(...s.filter((_, i) => y[i] === 1));
          c.near(f.meta.width, b - a, 1e-9, 'a candidate street’s width');
          c.ok(f.meta.width < widestStreet(X, y) + 1e-9, 'a candidate is no wider than the widest street');
        }
        const sv = phase(c.frames, 'svm')[0], eq = phase(c.frames, 'eq')[0];
        c.near(sv.meta.width, widestStreet(X, y), 2e-4, 'the SVM street is the widest of all directions');
        const w = eq.meta.w, b = eq.meta.b;
        c.near(eq.meta.width, 2 / Math.hypot(w[0], w[1]), 1e-12, 'width = 2/‖w‖');
        const yf = X.map((x, i) => (y[i] ? 1 : -1) * (dot(w, x) + b));
        c.ok(Math.min(...yf) > 1 - 1e-3, 'every point is on or outside its margin');
        const onMargin = yf.map((v, i) => [v, i]).filter(([v]) => Math.abs(v - 1) < 1e-3).map(([, i]) => i);
        c.ok(sv.meta.sv.every((i) => onMargin.includes(i)), 'every support vector lies on a margin');
        c.eq(sv.meta.nB + sv.meta.nO, sv.meta.sv.length, 'blue and orange support vectors add up');
        c.eq(sv.meta.nB, sv.meta.sv.filter((i) => y[i] === 0).length, 'blue support vectors');
        const df = phase(c.frames, 'drop-free')[0], ds = phase(c.frames, 'drop-sv')[0];
        const without = (k) => [X.filter((_, i) => i !== k), y.filter((_, i) => i !== k)];
        c.near(df.meta.width, widestStreet(...without(df.meta.drop)), 2e-4, 'width after deleting a point off the street');
        c.near(df.meta.width, sv.meta.width, 2e-4, 'deleting a point off the street leaves the width');
        c.near(ds.meta.width, widestStreet(...without(ds.meta.drop)), 2e-4, 'width after deleting a support vector');
        c.ok(ds.meta.width >= sv.meta.width - 1e-6, 'deleting a support vector can only widen the street');
      },
    },

    'svm-soft-c': {
      random: (r) => ({ set: ['overlap', 'line', 'iris'][Math.floor(r() * 3)], seed: 1 + Math.floor(r() * 300), C: +(0.01 * Math.pow(1e4, r())).toPrecision(2) }),
      check(c) {
        const R = refs(c), X = R[0].f.X, y = R[0].f.y, ys = y.map((v) => (v ? 1 : -1));
        let prevW = Infinity;
        for (const f of phase(c.frames, 'sweep').concat(phase(c.frames, 'mine'))) {
          const { w, b, C } = f.meta, yf = X.map((x, i) => ys[i] * (dot(w, x) + b)), xi = yf.map((v) => Math.max(0, 1 - v));
          c.near(f.meta.width, 2 / Math.hypot(...w), 1e-12, 'C = ' + C + ': width = 2/‖w‖');
          c.near(f.meta.xi, sum(xi), 1e-9, 'C = ' + C + ': total slack');
          if (f.meta.inside !== undefined) {
            c.eq(f.meta.inside, yf.filter((v) => v < 1 - 1e-3 && v >= 0).length, 'C = ' + C + ': points inside the street');
            c.eq(f.meta.wrong, yf.filter((v) => v < 0).length, 'C = ' + C + ': points on the wrong side');
            c.eq(f.meta.free + f.meta.bound, f.meta.nsv, 'C = ' + C + ': free and capped support vectors');
            c.ok(f.meta.bound >= xi.filter((v) => v > 1e-3).length, 'C = ' + C + ': every point with slack is capped');
            if (f.meta.phase === 'sweep') { c.ok(f.meta.width <= prevW * (1 + 1e-4), 'the street narrows as C grows'); prevW = f.meta.width; }
          }
          const e = R.find((x) => x.f.params.C === C);
          if (e && e.r) {
            c.ok(Math.hypot(w[0] - e.r.w[0], w[1] - e.r.w[1]) + Math.abs(b - e.r.b) < 5e-3 * Math.max(1, Math.hypot(...e.r.w)), 'C = ' + C + ': w and b agree with the reference');
            c.ok(Math.abs(f.meta.acc - acc(y, e.r.pred)) <= 1 / y.length + 1e-12, 'C = ' + C + ': training accuracy agrees with the reference');
          }
        }
      },
    },

    'svm-hinge': {
      random: (r) => ({ set: ['overlap', 'line'][Math.floor(r() * 2)], seed: 1 + Math.floor(r() * 300), C: +(0.01 * Math.pow(1e4, r())).toPrecision(2) }),
      check(c) {
        const R = refs(c), e = R[0], X = e.f.X, y = e.f.y, ys = y.map((v) => (v ? 1 : -1)), C = e.f.params.C;
        const tot = phase(c.frames, 'total')[0], zero = phase(c.frames, 'zero')[0], small = phase(c.frames, 'small')[0], big = phase(c.frames, 'big')[0];
        if (!e.r) { c.ok(hasRef(c) === false, 'reference present'); return; }
        const m = X.map((x, i) => ys[i] * (dot(e.r.w, x) + e.r.b)), loss = m.map((v) => Math.max(0, 1 - v)), near = (v) => Math.abs(v - 1) < 5e-3 || Math.abs(v) < 5e-3;
        const amb = m.filter(near).length;
        c.ok(Math.abs(zero.meta.count - m.filter((v) => v >= 1).length) <= amb, 'points with zero loss');
        c.ok(Math.abs(small.meta.count - m.filter((v) => v < 1 && v >= 0).length) <= amb, 'points inside the street');
        c.ok(Math.abs(big.meta.count - m.filter((v) => v < 0).length) <= amb, 'points on the wrong side');
        c.eq(zero.meta.count + small.meta.count + big.meta.count, X.length, 'every point is in one group');
        const primal = 0.5 * dot(e.r.w, e.r.w) + C * sum(loss);
        c.near(tot.meta.primal, primal, 2e-3, 'the primal objective, from the reference weights');
        c.near(tot.meta.hinge, sum(loss), 5e-3, 'the total hinge loss');
        c.near(tot.meta.dual, tot.meta.primal, 1e-3, 'strong duality: the dual reaches the primal');
        c.near(tot.meta.primal, tot.meta.half + C * tot.meta.hinge, 1e-12, 'the objective adds up');
      },
    },

    'svm-kernel-lift': {
      random: (r) => ({ n: 20 + Math.floor(r() * 80), noise: Math.round(r() * 20) / 100, seed: 1 + Math.floor(r() * 300) }),
      check(c) {
        const R = refs(c), X = R[0].f.X, y = R[0].f.y, Z = R[1].f.X;
        Z.forEach((z, i) => c.ok(Math.abs(z[2] - X[i][0] ** 2 - X[i][1] ** 2) < 1e-12, 'the lift is x₁² + x₂²'));
        const lift = phase(c.frames, 'lift')[0];
        c.eq(lift.meta.zin, Math.max(...Z.filter((_, i) => y[i] === 1).map((p) => p[2])), 'the highest point of the disc');
        c.eq(lift.meta.zout, Math.min(...Z.filter((_, i) => y[i] === 0).map((p) => p[2])), 'the lowest point of the ring');
        const ci = phase(c.frames, 'circle')[0], [w1, w2, w3] = ci.meta.w, b = ci.meta.b;
        // the circle from completing the square, written out again
        const cx = -w1 / (2 * w3), cy = -w2 / (2 * w3), r2 = cx * cx + cy * cy - b / w3;
        c.near(ci.meta.cx, cx, 1e-12, 'circle centre x'); c.near(ci.meta.cy, cy, 1e-12, 'circle centre y'); c.near(ci.meta.rad, Math.sqrt(r2), 1e-12, 'circle radius');
        const onC = Math.atan2(1, 1), p = [cx + Math.sqrt(r2) * Math.cos(onC), cy + Math.sqrt(r2) * Math.sin(onC)];
        c.ok(Math.abs(w3 * (p[0] ** 2 + p[1] ** 2) + w1 * p[0] + w2 * p[1] + b) < 1e-9, 'a point of the circle lies on the plane');
        c.near(ci.meta.acc, acc(y, X.map((x) => (w3 * (x[0] ** 2 + x[1] ** 2) + w1 * x[0] + w2 * x[1] + b > 0 ? 1 : 0))), 1e-12, 'accuracy of the circle');
        if (R[1].r) c.near(ci.meta.acc, acc(y, R[1].r.pred), 1 / y.length + 1e-9, 'accuracy agrees with the reference');
        const k = phase(c.frames, 'kernel')[0], phi = (x) => [x[0], x[1], x[0] ** 2 + x[1] ** 2];
        c.near(k.meta.viaPhi, dot(phi(k.meta.a), phi(k.meta.c)), 1e-12, 'the lifted dot product');
        c.near(k.meta.viaK, dot(k.meta.a, k.meta.c) + dot(k.meta.a, k.meta.a) * dot(k.meta.c, k.meta.c), 1e-12, 'the kernel');
        c.near(k.meta.viaK, k.meta.viaPhi, 1e-9, 'the kernel equals the lifted dot product');
      },
    },

    'svm-poly': {
      random: (r) => ({ set: ['xor', 'moons', 'circles'][Math.floor(r() * 3)], C: +(0.1 * Math.pow(1000, r())).toPrecision(2), seed: 1 + Math.floor(r() * 300) }),
      randomRuns: 3,
      check(c) {
        const R = refs(c);
        for (const f of phase(c.frames, 'deg')) {
          const e = R.find((x) => x.f.params.degree === f.meta.deg);
          c.eq(f.meta.feat, [3, 6, 10, 15, 21][f.meta.deg - 1], 'degree ' + f.meta.deg + ': the number of monomials');
          if (!e.r) continue;
          const y = e.f.y, ys = y.map((v) => (v ? 1 : -1));
          c.ok(Math.abs(f.meta.acc - acc(y, e.r.pred)) <= 1 / y.length + 1e-12, 'degree ' + f.meta.deg + ': accuracy agrees with the reference');
          sameSupport(c, e.f.support, e.r.support, (i) => Math.abs(ys[i] * e.r.dec[i] - 1) < 1e-2, 'degree ' + f.meta.deg);
        }
        const k = phase(c.frames, 'check')[0], a = k.meta.a, b = k.meta.c, s2 = Math.SQRT2;
        const phi = (x) => [1, s2 * x[0], s2 * x[1], x[0] ** 2, x[1] ** 2, s2 * x[0] * x[1]];
        c.near(k.meta.ex, dot(phi(a), phi(b)), 1e-12, 'the six features multiplied');
        c.near(k.meta.kv, (dot(a, b) + 1) ** 2, 1e-12, 'the kernel');
        c.near(k.meta.kv, k.meta.ex, 1e-9, 'the kernel equals the feature dot product');
      },
    },

    'svm-rbf': {
      random: (r) => ({ set: ['moons', 'circles', 'xor'][Math.floor(r() * 3)], C: +(0.1 * Math.pow(1000, r())).toPrecision(2), seed: 1 + Math.floor(r() * 300) }),
      randomRuns: 3,
      check(c) {
        const R = refs(c);
        for (const f of phase(c.frames, 'gamma')) {
          const e = R.find((x) => x.f.params.gamma === f.meta.gamma);
          c.near(f.meta.reach, 1 / Math.sqrt(f.meta.gamma), 1e-12, 'the reach 1/√γ');
          if (!e.r) continue;
          const yte = c.result.result.test;
          c.ok(Math.abs(f.meta.acc - acc(e.f.y, e.r.pred)) <= 1 / e.f.y.length + 1e-12, 'γ = ' + f.meta.gamma + ': training accuracy agrees with the reference');
          if (yte) c.ok(Math.abs(f.meta.test - acc(yte, e.r.dec_e.map((v) => (v > 0 ? 1 : 0)))) <= 1 / yte.length + 1e-12, 'γ = ' + f.meta.gamma + ': held-back accuracy agrees with the reference');
          const ys = e.f.y.map((v) => (v ? 1 : -1));
          sameSupport(c, e.f.support, e.r.support, (i) => Math.abs(ys[i] * e.r.dec[i] - 1) < 1e-2, 'γ = ' + f.meta.gamma);
        }
        const g = phase(c.frames, 'gamma'), best = phase(c.frames, 'best')[0];
        c.eq(best.meta.test, Math.max(...g.map((f) => f.meta.test)), 'the best held-back accuracy');
        c.eq(best.meta.gamma, g.find((f) => f.meta.test === best.meta.test).meta.gamma, 'the γ chosen is the first with the best score');
      },
    },

    'svm-dual': {
      random: (r) => ({ n: 6 + Math.floor(r() * 34), spread: 0.3 + Math.round(r() * 14) * 0.05, C: +(0.05 * Math.pow(1000, r())).toPrecision(2), seed: 1 + Math.floor(r() * 300) }),
      check(c) {
        const R = refs(c), e = R[0], X = e.f.X, y = e.f.y, C = e.f.params.C;
        let sx = 0, sy = 0;
        for (const f of phase(c.frames, 'add')) {
          const i = f.meta.i, t = [f.meta.alpha * f.meta.y * X[i][0], f.meta.alpha * f.meta.y * X[i][1]];
          c.eq(f.meta.y, y[i] ? 1 : -1, 'the label of point ' + i);
          c.near(f.meta.term[0], t[0], 1e-12, 'the term αyx'); c.near(f.meta.term[1], t[1], 1e-12, 'the term αyx');
          sx += t[0]; sy += t[1];
          c.near(f.meta.sum[0], sx, 1e-9, 'the running sum'); c.near(f.meta.sum[1], sy, 1e-9, 'the running sum');
          if (e.r) { const ra = (e.r.alpha.find((p) => p[0] === i) || [i, 0])[1]; c.ok(Math.abs(ra - f.meta.alpha) < 1e-3 * Math.max(1, C), 'α of point ' + i + ' agrees with the reference'); }
        }
        const done = phase(c.frames, 'done')[0];
        c.near(done.meta.w[0], sx, 1e-6, 'the sum ends at w'); c.near(done.meta.w[1], sy, 1e-6, 'the sum ends at w');
        c.near(done.meta.pos, done.meta.neg, 1e-6, 'Σ αᵢyᵢ = 0');
        c.near(done.meta.dual, done.meta.sumAlpha - 0.5 * dot(done.meta.w, done.meta.w), 1e-6, 'the dual objective Σα − ½‖w‖²');
        if (e.r) c.ok(Math.hypot(done.meta.w[0] - e.r.w[0], done.meta.w[1] - e.r.w[1]) < 1e-3 * Math.max(1, Math.hypot(...e.r.w)), 'w agrees with the reference');
      },
    },

    'svm-smo': {
      random: (r) => ({ n: 6 + Math.floor(r() * 14), spread: 0.3 + Math.round(r() * 14) * 0.05, kernel: r() < 0.5 ? 'linear' : 'rbf', C: +(0.1 * Math.pow(200, r())).toPrecision(2), seed: 1 + Math.floor(r() * 300) }),
      check(c) {
        const R = refs(c), e = R[0], X = e.f.X, n = X.length, C = e.f.params.C, K = kern(e.f.params, 1);
        const yy = e.f.y.map((v) => (v === 0 ? 1 : -1)), Km = X.map((a) => X.map((b) => K(a, b)));
        const dual = (a) => { let s = sum(a); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s -= 0.5 * a[i] * a[j] * yy[i] * yy[j] * Km[i][j]; return s; };
        let prev = 0;
        for (const f of phase(c.frames, 'step')) {
          const { i, j, oi, oj, ai, aj } = f.meta;
          c.near(yy[i] * (ai - oi) + yy[j] * (aj - oj), 0, 1e-9, 'step ' + f.meta.k + ': Σ αᵢyᵢ is unchanged');
          for (const v of [ai, aj]) c.ok(v >= -1e-12 && v <= C + 1e-12, 'step ' + f.meta.k + ': α stays in [0, C]');
          // the best point on the pair's line, worked out afresh: move αj by d and αi by −s·d (s = yᵢyⱼ), keep Σ αy
          const a0 = f.meta.prev.slice(), Gi = (t) => { let s = -1; for (let u = 0; u < n; u++) s += a0[u] * yy[t] * yy[u] * Km[t][u]; return s; };
          const s = yy[i] * yy[j], eta = Km[i][i] + Km[j][j] - 2 * Km[i][j];
          // D is a parabola in d: three samples give its slope and curvature exactly
          const D1 = (d) => { const a = a0.slice(); a[j] = oj + d; a[i] = oi - s * d; return dual(a); };
          const curv = D1(1) - 2 * D1(0) + D1(-1), slope = (D1(1) - D1(-1)) / 2, dStar = curv < -1e-15 ? -slope / curv : 0;
          const clamp = (v0, sign) => { const l = sign > 0 ? -v0 : v0 - C, h = sign > 0 ? C - v0 : v0; return [l, h]; };
          const [l1, h1] = clamp(oj, 1), [l2, h2] = s > 0 ? clamp(oi, -1) : clamp(oi, 1);
          const lo = Math.max(l1, l2), hi = Math.min(h1, h2);
          const dBest = Math.max(lo, Math.min(hi, dStar));
          if (eta > 1e-9) c.ok(Math.abs(aj - (oj + dBest)) < 1e-6 * Math.max(1, C), 'step ' + f.meta.k + ': the pair moves to the best clipped point on its line');
          c.near(f.meta.dual, dual(f.meta.alpha), 1e-9, 'step ' + f.meta.k + ': the dual objective');
          c.ok(f.meta.dual >= prev - 1e-9, 'step ' + f.meta.k + ': the dual objective never falls');
          prev = f.meta.dual;
          // the first of the pair is the most violating multiplier that can still move up (ties to the later index)
          const G = a0.map((_, t) => Gi(t)), up = (t) => (yy[t] === 1 ? a0[t] < C : a0[t] > 0);
          let best = -Infinity, bi = -1;
          for (let t = 0; t < n; t++) if (up(t) && -yy[t] * G[t] >= best - 1e-12) { best = Math.max(best, -yy[t] * G[t]); bi = t; }
          c.ok(Math.abs(-yy[i] * G[i] - best) < 1e-9, 'step ' + f.meta.k + ': the first multiplier is the most violating (' + i + ' vs ' + bi + ')');
        }
        const done = phase(c.frames, 'done')[0];
        c.near(done.meta.dual, dual(done.meta.alpha), 1e-9, 'the final dual objective');
        c.eq(done.meta.nsv, done.meta.alpha.filter((v) => v > 0).length, 'the multipliers above zero');
        if (e.r) for (let t = 0; t < n; t++) { const ra = (e.r.alpha.find((p) => p[0] === t) || [t, 0])[1]; c.ok(Math.abs(ra - done.meta.alpha[t]) < 2e-2 * Math.max(1, C), 'final α of point ' + t + ' agrees with the reference (' + ra + ' vs ' + done.meta.alpha[t] + ')'); }
      },
    },

    'svm-multiclass': {
      random: (r) => ({ set: ['apart', 'close'][Math.floor(r() * 2)], kernel: r() < 0.5 ? 'linear' : 'rbf', C: +(0.1 * Math.pow(1000, r())).toPrecision(2), seed: 1 + Math.floor(r() * 300) }),
      randomRuns: 3,
      check(c) {
        const R = refs(c), ovr = R[0], ovo = R[1], y = ovr.f.y;
        for (const f of phase(c.frames, 'ovr')) {
          if (ovr.r) c.ok(Math.abs(f.meta.nsv - ovr.r.nsv[f.meta.t]) <= Math.max(1, 0.03 * ovr.r.nsv[f.meta.t]), 'one-vs-rest machine ' + f.meta.no + ': support vectors agree with the reference');
          if (ovr.r) c.ok(Math.abs(f.meta.own - y.filter((v, i) => v === f.meta.cls && ovr.r.dec[i][f.meta.cls] > 0).length) <= 1, 'machine ' + f.meta.no + ': its own class on its positive side');
          c.eq(f.meta.count, y.filter((v) => v === f.meta.cls).length, 'the size of the class');
        }
        for (const f of phase(c.frames, 'ovo')) {
          c.eq(f.meta.used, y.filter((v) => v === f.meta.a || v === f.meta.b).length, 'one-vs-one machine ' + f.meta.no + ': the points it trains on');
          if (ovo.r) c.ok(Math.abs(f.meta.nsv - ovo.r.pair_nsv[f.meta.t]) <= Math.max(1, 0.03 * ovo.r.pair_nsv[f.meta.t]), 'one-vs-one machine ' + f.meta.no + ': support vectors agree with the reference');
        }
        const ra = phase(c.frames, 'ovr-all')[0], oa = phase(c.frames, 'ovo-all')[0];
        if (ovr.r) { c.ok(maxDiff(ra.meta.vals, ovr.r.dec_e[0]) < 2e-3 * Math.max(1, ...ovr.r.dec_e[0].map(Math.abs)), 'the three one-vs-rest values at the square'); c.near(ra.meta.acc, acc(y, ovr.r.pred), 1 / y.length + 1e-9, 'one-vs-rest accuracy'); }
        c.eq(ra.meta.win, ra.meta.vals.indexOf(Math.max(...ra.meta.vals)), 'one-vs-rest picks the largest value');
        if (ovo.r) {
          const votes = [0, 0, 0], pairs = [[0, 1], [0, 2], [1, 2]];
          ovo.r.dec_e[0].forEach((v, t) => votes[v > 0 ? pairs[t][0] : pairs[t][1]]++);
          c.eq(oa.meta.votes, votes, 'the votes at the square, from the reference machines');
          c.near(oa.meta.acc, acc(y, ovo.r.pred), 1 / y.length + 1e-9, 'one-vs-one accuracy');
        }
        c.eq(oa.meta.votes.reduce((a, b) => a + b, 0), 3, 'three machines vote');
      },
    },

    'svm-svr': {
      random: (r) => ({ shape: ['sine', 'poly', 'step', 'abs'][Math.floor(r() * 4)], noise: Math.round(r() * 40) / 100, gamma: +(0.05 * Math.pow(100, r())).toPrecision(2), C: +(0.1 * Math.pow(1000, r())).toPrecision(2), eps: Math.round(1 + r() * 99) / 100, seed: 1 + Math.floor(r() * 300) }),
      randomRuns: 4,
      check(c) {
        const R = refs(c), d = S.Data.curve({ n: 40, shape: c.input.shape, noise: c.input.noise, seed: c.input.seed });
        for (const f of phase(c.frames, 'eps').concat(phase(c.frames, 'mine'))) {
          const e = R.find((x) => x.f.params.epsilon === f.meta.eps);
          if (!e.r) continue;
          const res = d.y.map((v, i) => Math.abs(v - e.r.pred[i])), edge = (i) => Math.abs(res[i] - f.meta.eps) < 2e-3;
          sameSupport(c, e.f.support, e.r.support, edge, 'ε = ' + f.meta.eps);
          c.ok(Math.abs(f.meta.out - res.filter((v) => v > f.meta.eps + 1e-6).length) <= d.y.filter((_, i) => edge(i)).length, 'ε = ' + f.meta.eps + ': points outside the tube');
          const mse = d.X.reduce((s, x, i) => s + (d.truth(x[0]) - e.r.pred[i]) ** 2, 0) / d.X.length;
          c.near(f.meta.truth, mse, 2e-2, 'ε = ' + f.meta.eps + ': error against the true curve');
          c.ok(f.meta.out <= f.meta.nsv, 'every point outside the tube is a support vector');
        }
        const eps = phase(c.frames, 'eps');
        for (let k = 1; k < eps.length; k++) c.ok(eps[k].meta.nsv <= eps[k - 1].meta.nsv + 1, 'a wider tube keeps no more support vectors');
      },
    },

    'svm-one-class': {
      random: (r) => ({ set: r() < 0.7 ? 'blobs' : 'pumps', gamma: +(0.05 * Math.pow(100, r())).toPrecision(2), seed: 1 + Math.floor(r() * 300) }),
      randomRuns: 3,
      check(c) {
        const R = refs(c);
        for (const f of phase(c.frames, 'nu')) {
          const e = R.find((x) => x.f.params.nu === f.meta.nu), n = f.meta.n;
          c.ok(f.meta.out / n <= f.meta.nu + 1e-9, 'ν = ' + f.meta.nu + ': the share outside is at most ν');
          c.ok(f.meta.nu <= f.meta.nsv / n + 1e-9, 'ν = ' + f.meta.nu + ': ν is at most the share of support vectors');
          c.near(f.meta.outShare, f.meta.out / n, 1e-12, 'the share outside'); c.near(f.meta.svShare, f.meta.nsv / n, 1e-12, 'the share of support vectors');
          if (!e.r) continue;
          const scale = Math.max(...e.r.dec.map(Math.abs)), edge = (i) => Math.abs(e.r.dec[i]) < 2e-3 * Math.max(1, scale);
          c.ok(Math.abs(f.meta.out - e.r.dec.filter((v) => v < -1e-4).length) <= e.r.dec.filter((_, i) => edge(i)).length, 'ν = ' + f.meta.nu + ': points outside agree with the reference');
          sameSupport(c, e.f.support, e.r.support, edge, 'ν = ' + f.meta.nu);
        }
      },
    },
  };
}
