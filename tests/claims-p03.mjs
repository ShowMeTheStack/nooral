// Part 3, linear models: every preset and random inputs of each topic; every number in the narration must be in the
// frame's values (tests/claims-p03-lib.mjs), and those values are worked out again here by an independent method:
// closed forms and normal equations solved by hand, finite differences, scikit-learn references where the data match.
// Usage: node tests/claims-p03.mjs [topic id ...]   Prints "claims-p03 N passed, M failed".
import { setup, runPart } from './claims-p03-lib.mjs';

const { S, t, rnd } = setup('claims-p03');
const only = process.argv.slice(2);
const pick = (r, a) => a[Math.floor(r() * a.length)];
const between = (r, a, b, step) => Math.round((a + (b - a) * r()) / step) * step;

// ---- independent pieces ----
/** simple linear regression by its textbook formulas */
function slr(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  const w = sxy / sxx, b = my - w * mx;
  const sse = ys.reduce((s, y, i) => s + (y - w * xs[i] - b) ** 2, 0), sst = ys.reduce((s, y) => s + (y - my) ** 2, 0);
  return { w, b, mx, my, sxx, sxy, sse, sst };
}


/** Gauss–Jordan solve of A x = b (small, dense), written here apart from the engines */
function gsolve(A, b) {
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const sigm = (z) => 1 / (1 + Math.exp(-z));
/** standardise columns as the topics do (population standard deviation) */
function stdz(X) {
  const d = X[0].length, m = [], sd = [];
  for (let j = 0; j < d; j++) { const c = X.map((r) => r[j]); m.push(mean(c)); sd.push(Math.sqrt(mean(c.map((v) => (v - m[j]) ** 2))) || 1); }
  return { X: X.map((r) => r.map((v, j) => (v - m[j]) / sd[j])), m, sd };
}
/** the KKT conditions of elastic net (1/2n)||y − Xw||² + a(r|w|₁ + (1 − r)/2 ||w||²) on centred data */
function kkt(X, y, w, a, r, tol) {
  const n = X.length, d = X[0].length, ym = mean(y), xm = [...Array(d).keys()].map((j) => mean(X.map((q) => q[j])));
  const res = X.map((q, i) => y[i] - ym - q.reduce((s, v, j) => s + (v - xm[j]) * w[j], 0));
  for (let j = 0; j < d; j++) {
    const g = X.reduce((s, q, i) => s + (q[j] - xm[j]) * res[i], 0) / n - a * (1 - r) * w[j];
    if (w[j] !== 0 && Math.abs(g - a * r * Math.sign(w[j])) > tol * Math.max(1, a)) return 'feature ' + j + ': gradient ' + g + ' against ' + a * r * Math.sign(w[j]);
    if (w[j] === 0 && Math.abs(g) > a * r + tol * Math.max(1, a)) return 'zero feature ' + j + ': |gradient| ' + Math.abs(g) + ' above ' + a * r;
  }
  return null;
}

const mods = {
  'ls-line': {
    random: (r) => ({ n: 2 + Math.floor(r() * 29), slope: between(r, -1.5, 1.5, 0.1), noise: between(r, 0, 3, 0.1), outlier: pick(r, ['no', 'yes']), seed: 1 + Math.floor(r() * 999) }),
    check({ r, fail, pass, near, input }) {
      const d = S.PL.lineData(input.n, input.seed, input.slope, 2.2 + Math.max(0, -input.slope) * 9, input.noise, input.outlier === 'yes');
      const xs = d.X.map((q) => q[0]), ref = slr(xs, d.y), m = r.frames[0].meta;
      for (const k of ['w', 'sse', 'sst', 'sxx', 'sxy']) (near(m[k], ref[k], 1e-7) ? pass : fail)(k + ' = ' + m[k] + ', by the formula ' + ref[k]);
      (near(m.b, ref.b, 1e-7) ? pass : fail)('b');
      (near(m.xbar, ref.mx, 1e-9) && near(m.ybar, ref.my, 1e-9) ? pass : fail)('means');
      (near(m.Sxx, xs.reduce((s, v) => s + v * v, 0)) && near(m.Sxy, xs.reduce((s, v, i) => s + v * d.y[i], 0)) ? pass : fail)('sums');
      const nudge = r.frames.find((f) => f.meta.phase === 'nudge').meta;
      const sseAt = (w) => d.y.reduce((s, y, i) => s + (y - w * xs[i] - (ref.my - w * ref.mx)) ** 2, 0);
      (near(nudge.up, sseAt(nudge.wUp)) && near(nudge.dn, sseAt(nudge.wDn)) ? pass : fail)('nudged sums of squares');
      (nudge.up >= ref.sse - 1e-9 && nudge.dn >= ref.sse - 1e-9 ? pass : fail)('the best line is the least');
      const say = r.frames.find((f) => f.meta.phase === 'normal').say;
      (say.includes('n = ' + xs.length) ? pass : fail)('n quoted');
    },
  },
  'ls-projection': {
    random: (r) => ({ x: [0, 1, 2].map(() => String(between(r, -4, 4, 0.5))), y: [0, 1, 2].map(() => String(between(r, -4, 4, 0.5))), az: between(r, -60, 60, 5), el: between(r, 10, 50, 1) }),
    check({ r, fail, pass, near, input }) {
      const xs = input.x.map((v) => +String(v).replace('−', '-')), ys = input.y.map((v) => +String(v).replace('−', '-')), ref = slr(xs, ys), m = r.frames[0].meta;
      (near(m.w, ref.w, 1e-8) && near(m.b, ref.b, 1e-8) ? pass : fail)('w, b');
      const yh = xs.map((v) => ref.w * v + ref.b), res = ys.map((v, i) => v - yh[i]);
      (res.every((v, i) => near(v, m.r[i], 1e-8)) ? pass : fail)('residual vector');
      (Math.abs(res.reduce((a, b) => a + b)) < 1e-9 && Math.abs(res.reduce((a, v, i) => a + v * xs[i], 0)) < 1e-8 ? pass : fail)('the residual is perpendicular to 1 and x');
      (near(m.dist ** 2, ref.sse, 1e-8) ? pass : fail)('‖r‖² = SSE');
      (near(m.dFlat ** 2, ref.sst, 1e-8) && m.dist <= m.dFlat + 1e-12 ? pass : fail)('the flat guess is farther');
      (Math.abs(m.r1) < 1e-9 && Math.abs(m.rx) < 1e-8 ? pass : fail)('quoted dot products are zero');
    },
  },
  'gd-closed': {
    random: (r) => ({ scale: pick(r, ['raw', 'std']), lr: between(r, -3, 0.4, 0.05), epochs: 5 + Math.floor(r() * 120), seed: 1 + Math.floor(r() * 99) }),
    check({ r, fail, pass, near, input }) {
      const d = S.PL.lineData(20, input.seed, 0.6, 2.2, 0.8), x0 = d.X.map((q) => q[0]), mu = mean(x0), sd = Math.sqrt(mean(x0.map((v) => (v - mu) ** 2)));
      const xs = input.scale === 'std' ? x0.map((v) => (v - mu) / sd) : x0, y = d.y, n = y.length, lr = Math.pow(10, input.lr), ref = slr(xs, y);
      const m0 = r.frames[0].meta;
      (near(m0.wS, ref.w, 1e-8) && near(m0.bS, ref.b, 1e-8) ? pass : fail)('closed form');
      const L = (w, b) => y.reduce((s, v, i) => s + (v - w * xs[i] - b) ** 2, 0) / (2 * n);
      (near(m0.Ls, L(ref.w, ref.b), 1e-9) ? pass : fail)('floor loss');
      const mxx = mean(xs.map((v) => v * v)), mx = mean(xs), lam = (mxx + 1) / 2 + Math.sqrt(((mxx - 1) / 2) ** 2 + mx * mx);
      (near(m0.lrMax, 2 / lam, 1e-9) ? pass : fail)('stability limit');
      // gradient descent replayed here
      let w = 0, b = 0;
      const hist = [];
      for (let e = 0; e <= input.epochs; e++) {
        let gw = 0, gb = 0;
        for (let i = 0; i < n; i++) { const res = w * xs[i] + b - y[i]; gw += res * xs[i]; gb += res; }
        hist.push({ w, b, gw: gw / n, gb: gb / n });
        w -= (lr * gw) / n; b -= (lr * gb) / n;
      }
      for (const f of r.frames.filter((q) => q.meta.phase === 'step')) {
        const h = hist[f.meta.e];
        if (!isFinite(f.meta.loss)) { (!isFinite(h.w) || Math.abs(h.w) > 1e6 ? pass : fail)('overflow at ' + f.meta.e); continue; }
        (near(f.meta.w, h.w, 1e-7) && near(f.meta.b, h.b, 1e-7) ? pass : fail)('epoch ' + f.meta.e + ' weights');
        (near(f.meta.loss, L(h.w, h.b), 1e-7) ? pass : fail)('epoch ' + f.meta.e + ' loss');
        const eps = 1e-6 * Math.max(1, Math.abs(h.w)), fd = (L(h.w + eps, h.b) - L(h.w - eps, h.b)) / (2 * eps), fdb = (L(h.w, h.b + eps) - L(h.w, h.b - eps)) / (2 * eps);
        (Math.abs(fd - f.meta.gw) <= 1e-4 * Math.max(1, Math.abs(fd)) && Math.abs(fdb - f.meta.gb) <= 1e-4 * Math.max(1, Math.abs(fdb)) ? pass : fail)('epoch ' + f.meta.e + ' gradient against finite differences');
      }
    },
  },
  'poly-overfit': {
    random: (r) => { const n = 6 + Math.floor(r() * 30); return { shape: pick(r, ['sine', 'poly', 'abs', 'step', 'linear']), n, noise: between(r, 0, 0.8, 0.05), maxDeg: 1 + Math.floor(r() * Math.min(13, n - 1)), seed: 1 + Math.floor(r() * 999) }; },
    check({ r, fail, pass, near, input }) {
      const tr = S.Data.curve({ n: input.n, shape: input.shape, noise: input.noise, seed: input.seed }), te = S.Data.curve({ n: 60, shape: input.shape, noise: input.noise, seed: input.seed + 7919 });
      const feats = (x, d) => [1, ...[...Array(d).keys()].map((k) => Math.pow(x / 3, k + 1))];
      let prevTr = Infinity;
      for (const f of r.frames.filter((q) => q.meta.phase === 'deg')) {
        const d = f.meta.d;
        (f.meta.tr <= prevTr * (1 + 1e-6) + 1e-10 ? pass : fail)('training error rises at degree ' + d);
        prevTr = f.meta.tr;
        if (d > 7 || d >= input.n - 1) continue;
        const A = tr.X.map((q) => feats(q[0], d)), G = A[0].map((_, a) => A[0].map((__, c) => A.reduce((s, row) => s + row[a] * row[c], 0))), g = A[0].map((_, a) => A.reduce((s, row, i) => s + row[a] * tr.y[i], 0));
        const w = gsolve(G, g), pr = (x) => feats(x, d).reduce((s, v, j) => s + v * w[j], 0);
        const mtr = mean(tr.X.map((q, i) => (tr.y[i] - pr(q[0])) ** 2)), mte = mean(te.X.map((q, i) => (te.y[i] - pr(q[0])) ** 2));
        (near(f.meta.tr, mtr, 1e-5) || Math.abs(f.meta.tr - mtr) < 1e-9 ? pass : fail)('degree ' + d + ' training MSE ' + f.meta.tr + ' against ' + mtr);
        (near(f.meta.te, mte, 1e-5) ? pass : fail)('degree ' + d + ' test MSE ' + f.meta.te + ' against ' + mte);
      }
      const b = r.frames[0].meta, all = r.frames.filter((q) => q.meta.phase === 'deg').map((q) => q.meta.te);
      (near(b.bestTe, Math.min(...all), 1e-12) ? pass : fail)('the best degree');
    },
  },
  'reg-paths': {
    random: (r) => ({ data: pick(r, ['sparse', 'rents', 'pair']), method: pick(r, ['lasso', 'ridge']), steps: 6 + Math.floor(r() * 20), seed: 1 + Math.floor(r() * 99) }),
    check({ r, fail, pass, near, input }) {
      const raw = S.PL.regData(input.data, input.seed), X = stdz(raw.X).X, y = raw.y, n = X.length, d = X[0].length, ym = mean(y);
      const G = X[0].map((_, a) => X[0].map((__, c) => X.reduce((s, q) => s + q[a] * q[c], 0))), g = X[0].map((_, a) => X.reduce((s, q, i) => s + q[a] * (y[i] - ym), 0));
      const ols = gsolve(G, g);
      ((r.frames[0].meta.ols).every((v, j) => near(v, ols[j], 1e-6)) ? pass : fail)('least squares');
      for (const f of r.frames.filter((q) => q.meta.phase === 'path')) {
        const w = f.meta.w, a = f.meta.alpha;
        if (input.method === 'lasso') { const e = kkt(X, y, w, a, 1, 1e-6); (e ? fail : pass)('lasso at ' + a + ': ' + e); }
        else { const lhs = G.map((row, i) => row.reduce((s, v, j) => s + v * w[j], 0) + a * w[i]); (lhs.every((v, i) => near(v, g[i], 1e-7)) ? pass : fail)('ridge normal equations at ' + a); }
        const mse = mean(X.map((q, i) => (y[i] - ym - q.reduce((s, v, j) => s + v * w[j], 0)) ** 2));
        (near(f.meta.mse, mse, 1e-8) ? pass : fail)('training MSE');
        (f.meta.nz === w.filter((v) => v !== 0).length ? pass : fail)('non-zero count');
      }
    },
  },
  'reg-geometry': {
    random: (r) => ({ w1: between(r, -3, 3, 0.1), w2: between(r, -3, 3, 0.1), rho: between(r, -0.85, 0.85, 0.05), seed: 1 + Math.floor(r() * 99) }),
    check({ r, fail, pass, near, input }) {
      const { X, y } = S.PL.twoFeat(input.rho, input.w1, input.w2, input.seed, 50), n = X.length;
      const G = [[0, 0], [0, 0]], g = [0, 0];
      X.forEach((q, i) => { for (let a = 0; a < 2; a++) { g[a] += q[a] * y[i]; for (let c = 0; c < 2; c++) G[a][c] += q[a] * q[c]; } });
      const ols = gsolve(G, g), m0 = r.frames[0].meta;
      (near(m0.ols[0], ols[0], 1e-8) && near(m0.ols[1], ols[1], 1e-8) ? pass : fail)('least squares');
      for (const f of r.frames.filter((q) => q.meta.phase === 'budget' && q.meta.f < 1)) {
        const { wl, wr, al, ar, t1, t2 } = f.meta;
        (near(Math.abs(wl[0]) + Math.abs(wl[1]), t1, 1e-6) ? pass : fail)('lasso spends its budget');
        (near(Math.hypot(wr[0], wr[1]), t2, 1e-6) ? pass : fail)('ridge spends its budget');
        // lasso KKT without an intercept
        const res = X.map((q, i) => y[i] - q[0] * wl[0] - q[1] * wl[1]);
        let ok = true;
        for (let j = 0; j < 2; j++) { const gr = X.reduce((s, q, i) => s + q[j] * res[i], 0) / n; if (wl[j] !== 0 ? Math.abs(gr - al * Math.sign(wl[j])) > 1e-6 : Math.abs(gr) > al + 1e-6) ok = false; }
        (ok ? pass : fail)('lasso optimality at ' + al);
        const lhs = [G[0][0] * wr[0] + G[0][1] * wr[1] + ar * wr[0], G[1][0] * wr[0] + G[1][1] * wr[1] + ar * wr[1]];
        (near(lhs[0], g[0], 1e-6) && near(lhs[1], g[1], 1e-6) ? pass : fail)('ridge optimality at ' + ar);
        ((f.meta.zeros === wl.filter((v) => v === 0).length) ? pass : fail)('zero count');
      }
    },
  },
  'elastic-net': {
    random: (r) => ({ r: between(r, 0.1, 0.9, 0.05), steps: 6 + Math.floor(r() * 20), seed: 1 + Math.floor(r() * 99) }),
    check({ r, fail, pass, input }) {
      const raw = S.PL.regData('pair', input.seed), X = stdz(raw.X).X, y = raw.y;
      for (const f of r.frames.filter((q) => q.meta.phase === 'path')) {
        const e1 = kkt(X, y, f.meta.wl, f.meta.al, 1, 1e-6), e2 = kkt(X, y, f.meta.we, f.meta.ae, input.r, 1e-6);
        (e1 ? fail : pass)('lasso optimality: ' + e1);
        (e2 ? fail : pass)('elastic net optimality: ' + e2);
        (f.meta.nzl === f.meta.wl.filter((v) => v !== 0).length && f.meta.nze === f.meta.we.filter((v) => v !== 0).length ? pass : fail)('non-zero counts');
      }
    },
  },
  'logit-sigmoid': {
    random: (r) => ({ data: pick(r, ['exam', 'overlap', 'apart']), n: 8 + Math.floor(r() * 80), lr: between(r, 0.05, 3, 0.05), epochs: 5 + Math.floor(r() * 200), seed: 1 + Math.floor(r() * 999) }),
    check({ r, fail, pass, near, input }) {
      const D = S.PL.oneD(input.data, input.n, input.seed), x = D.x, y = D.y, n = x.length, mu = mean(x), sd = Math.sqrt(mean(x.map((v) => (v - mu) ** 2))) || 1, u = x.map((v) => (v - mu) / sd);
      const L = (w, b) => mean(u.map((q, i) => { const p = sigm(w * q + b); return -Math.log(y[i] ? p : 1 - p); }));
      let w = 0, b = 0;
      const hist = [];
      for (let e = 0; e <= input.epochs; e++) {
        let gw = 0, gb = 0;
        u.forEach((q, i) => { const er = sigm(w * q + b) - y[i]; gw += er * q; gb += er; });
        hist.push({ w, b, gw: gw / n, gb: gb / n });
        w -= input.lr * gw / n; b -= input.lr * gb / n;
      }
      for (const f of r.frames.filter((q) => q.meta.phase === 'step')) {
        const h = hist[f.meta.e];
        (near(f.meta.wStd, h.w, 1e-7) && near(f.meta.bStd, h.b, 1e-7) ? pass : fail)('epoch ' + f.meta.e + ' weights');
        (near(f.meta.loss, L(h.w, h.b), 1e-7) ? pass : fail)('epoch ' + f.meta.e + ' loss');
        const eps = 1e-6, fd = (L(h.w + eps, h.b) - L(h.w - eps, h.b)) / (2 * eps), fdb = (L(h.w, h.b + eps) - L(h.w, h.b - eps)) / (2 * eps);
        (Math.abs(fd - f.meta.gw) < 1e-5 && Math.abs(fdb - f.meta.gb) < 1e-5 ? pass : fail)('epoch ' + f.meta.e + ' gradient against finite differences');
        // the same line in hours
        (near(f.meta.w, h.w / sd, 1e-7) && near(f.meta.b, h.b - h.w * mu / sd, 1e-7) ? pass : fail)('hours conversion');
        const acc = mean(u.map((q, i) => ((sigm(h.w * q + h.b) >= 0.5 ? 1 : 0) === y[i] ? 1 : 0)));
        (near(f.meta.acc, acc, 1e-12) ? pass : fail)('accuracy');
      }
    },
  },
  'logit-boundary': {
    random: (r) => ({ data: pick(r, ['blobs', 'linear', 'exam', 'moons', 'xor']), n: 10 + Math.floor(r() * 120), lr: between(r, 0.05, 4, 0.05), epochs: 5 + Math.floor(r() * 200), seed: 1 + Math.floor(r() * 999) }),
    check({ r, fail, pass, near, input }) {
      const D = S.PL.plane2(input.data, input.n, input.seed), st = stdz(D.X), U = st.X, y = D.y, n = U.length;
      const L = (t) => mean(U.map((q, i) => { const p = sigm(t[0] * q[0] + t[1] * q[1] + t[2]); return -Math.log(Math.max(1e-300, y[i] ? p : 1 - p)); }));
      let t = [0, 0, 0];
      const hist = [];
      for (let e = 0; e <= input.epochs; e++) {
        const g = [0, 0, 0];
        U.forEach((q, i) => { const er = sigm(t[0] * q[0] + t[1] * q[1] + t[2]) - y[i]; g[0] += er * q[0]; g[1] += er * q[1]; g[2] += er; });
        hist.push({ t: t.slice(), g: g.map((v) => v / n) });
        t = t.map((v, k) => v - input.lr * g[k] / n);
      }
      for (const f of r.frames.filter((q) => q.meta.phase === 'step')) {
        const h = hist[f.meta.e];
        (f.meta.w.every((v, k) => near(v, h.t[k], 1e-7)) ? pass : fail)('epoch ' + f.meta.e + ' weights');
        (near(f.meta.loss, L(h.t), 1e-7) ? pass : fail)('epoch ' + f.meta.e + ' loss');
        (f.meta.g.every((v, k) => near(v, h.g[k], 1e-7)) ? pass : fail)('epoch ' + f.meta.e + ' gradient');
        const wrong = U.filter((q, i) => ((sigm(h.t[0] * q[0] + h.t[1] * q[1] + h.t[2]) >= 0.5 ? 1 : 0) !== y[i])).length;
        (f.meta.wrong === wrong ? pass : fail)('epoch ' + f.meta.e + ' misclassified ' + f.meta.wrong + ' against ' + wrong);
      }
    },
  },
  softmax: {
    random: (r) => ({ data: pick(r, ['three', 'four', 'flowers']), n: 15 + Math.floor(r() * 120), lr: between(r, 0.05, 3, 0.05), epochs: 5 + Math.floor(r() * 150), seed: 1 + Math.floor(r() * 999) }),
    check({ r, fail, pass, near, input }) {
      const D = S.PL.multi(input.data, input.n, input.seed), st = stdz(D.X), U = st.X, y = D.y, n = U.length, K = D.classes.length;
      const sm = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b); return e.map((v) => v / s); };
      let W = [...Array(K)].map(() => [0, 0, 0]);
      const hist = [];
      for (let e = 0; e <= input.epochs; e++) {
        const G = [...Array(K)].map(() => [0, 0, 0]);
        let loss = 0;
        U.forEach((q, i) => { const p = sm(W.map((w) => w[0] * q[0] + w[1] * q[1] + w[2])); loss -= Math.log(p[y[i]]); for (let k = 0; k < K; k++) { const er = p[k] - (y[i] === k ? 1 : 0); G[k][0] += er * q[0]; G[k][1] += er * q[1]; G[k][2] += er; } });
        hist.push({ W: W.map((w) => w.slice()), loss: loss / n });
        W = W.map((w, k) => w.map((v, j) => v - input.lr * G[k][j] / n));
      }
      (near(r.frames[0].meta.L0, Math.log(K), 1e-12) ? pass : fail)('starting loss ln K');
      const probe = r.frames[0].meta.probe, pu = [(probe[0] - st.m[0]) / st.sd[0], (probe[1] - st.m[1]) / st.sd[1]];
      for (const f of r.frames.filter((q) => q.meta.phase === 'step')) {
        const h = hist[f.meta.e];
        (near(f.meta.loss, h.loss, 1e-7) ? pass : fail)('epoch ' + f.meta.e + ' loss');
        const z = h.W.map((w) => w[0] * pu[0] + w[1] * pu[1] + w[2]), p = sm(z);
        (f.meta.zP.every((v, k) => near(v, z[k], 1e-6)) && f.meta.pP.every((v, k) => near(v, p[k], 1e-6)) ? pass : fail)('probe scores and probabilities');
        (near(f.meta.sumP, 1, 1e-12) ? pass : fail)('probabilities add to 1');
        const acc = mean(U.map((q, i) => { const zz = h.W.map((w) => w[0] * q[0] + w[1] * q[1] + w[2]); return zz.indexOf(Math.max(...zz)) === y[i] ? 1 : 0; }));
        (near(f.meta.acc, acc, 1e-12) ? pass : fail)('accuracy');
      }
    },
  },
  perceptron: {
    random: (r) => ({ data: pick(r, ['easy', 'narrow', 'overlap']), n: 4 + Math.floor(r() * 50), eta: between(r, 0.1, 2, 0.1), epochs: 1 + Math.floor(r() * 20), seed: 1 + Math.floor(r() * 999) }),
    check({ r, fail, pass, near, input }) {
      const { X, y } = S.PL.percData(input.data, input.n, input.seed), t = y.map((v) => (v ? 1 : -1));
      let w = [0, 0], b = 0, ups = [];
      for (let e = 0; e < input.epochs; e++) {
        let mist = 0;
        X.forEach((x, i) => { const z = w[0] * x[0] + w[1] * x[1] + b; if (t[i] * z <= 0) { mist++; ups.push({ i, z, old: w.slice(), oldB: b }); w = [w[0] + input.eta * t[i] * x[0], w[1] + input.eta * t[i] * x[1]]; b += input.eta * t[i]; ups[ups.length - 1].w = w.slice(); ups[ups.length - 1].b = b; } });
        if (!mist) break;
      }
      for (const f of r.frames.filter((q) => q.meta.phase === 'update')) {
        const u = ups[f.meta.upd - 1];
        (u && u.i === f.meta.i && near(u.z, f.meta.z, 1e-9) && near(u.w[0], f.meta.w[0], 1e-9) && near(u.w[1], f.meta.w[1], 1e-9) && near(u.b, f.meta.b, 1e-9) && near(u.old[0], f.meta.old[0], 1e-9) ? pass : fail)('update ' + f.meta.upd);
      }
      const end = r.frames[r.frames.length - 1].meta;
      (end.upd === ups.length ? pass : fail)('number of updates ' + end.upd + ' against ' + ups.length);
      const wrong = X.filter((x, i) => t[i] * (w[0] * x[0] + w[1] * x[1] + b) <= 0).length;
      (end.wrong === wrong ? pass : fail)('points left wrong');
    },
  },
  'lda-qda': {
    random: (r) => ({ data: pick(r, ['same', 'diff', 'inside', 'flowers']), n: 12 + Math.floor(r() * 200), seed: 1 + Math.floor(r() * 999) }),
    check({ r, fail, pass, near, input }) {
      const D = S.PL.daData(input.data, input.n, input.seed), X = D.X, y = D.y, K = D.classes.length, n = X.length;
      const groups = [...Array(K)].map((_, k) => X.filter((q, i) => y[i] === k)), means = groups.map((g) => [mean(g.map((q) => q[0])), mean(g.map((q) => q[1]))]);
      const cov = (g, m) => { const C = [[0, 0], [0, 0]]; g.forEach((q) => { for (let a = 0; a < 2; a++) for (let c = 0; c < 2; c++) C[a][c] += (q[a] - m[a]) * (q[c] - m[c]); }); return C; };
      const mm = r.frames.find((f) => f.meta.phase === 'means').meta.means;
      (mm.every((m, k) => near(m[0], means[k][0], 1e-9) && near(m[1], means[k][1], 1e-9)) ? pass : fail)('class means');
      const Sw = groups.reduce((A, g, k) => { const C = cov(g, means[k]); return A.map((row, a) => row.map((v, c) => v + C[a][c])); }, [[0, 0], [0, 0]]).map((row) => row.map((v) => v / n));
      const sh = r.frames.find((f) => f.meta.phase === 'shared').meta.cov;
      (sh.every((row, a) => row.every((v, c) => near(v, Sw[a][c], 1e-9))) ? pass : fail)('pooled covariance');
      const Ck = groups.map((g, k) => cov(g, means[k]).map((row) => row.map((v) => v / g.length)));
      for (const f of r.frames.filter((q) => q.meta.phase === 'qcov')) (f.meta.cov.every((row, a) => row.every((v, c) => near(v, Ck[f.meta.k][a][c], 1e-9))) ? pass : fail)('class covariance ' + f.meta.k);
      // classify by the Gaussian scores written out here
      const inv = (C) => { const d = C[0][0] * C[1][1] - C[0][1] * C[1][0]; return { I: [[C[1][1] / d, -C[0][1] / d], [-C[1][0] / d, C[0][0] / d]], ld: Math.log(d) }; };
      const pri = groups.map((g) => g.length / n);
      const score = (q, k, C) => { const { I, ld } = inv(C), dx = [q[0] - means[k][0], q[1] - means[k][1]]; return Math.log(pri[k]) - 0.5 * ld - 0.5 * (dx[0] * (I[0][0] * dx[0] + I[0][1] * dx[1]) + dx[1] * (I[1][0] * dx[0] + I[1][1] * dx[1])); };
      const accOf = (A, b, which) => mean(A.map((q, i) => { const s = [...Array(K).keys()].map((k) => score(q, k, which === 'l' ? Sw : Ck[k])); return s.indexOf(Math.max(...s)) === b[i] ? 1 : 0; }));
      const bm = r.frames.find((f) => f.meta.phase === 'bound').meta, tm = r.frames.find((f) => f.meta.phase === 'test').meta;
      (near(bm.accL, accOf(X, y, 'l'), 1e-12) && near(bm.accQ, accOf(X, y, 'q'), 1e-12) ? pass : fail)('training accuracies');
      const te = S.PL.daData(input.data, 400, input.seed + 7777);
      (near(tm.tL, accOf(te.X, te.y, 'l'), 1e-12) && near(tm.tQ, accOf(te.X, te.y, 'q'), 1e-12) ? pass : fail)('test accuracies');
    },
  },
};

const res = runPart(S, t, rnd, /^p03-/, mods, only);
t.ok(res.runs > 0, 'some runs');
t.done();
