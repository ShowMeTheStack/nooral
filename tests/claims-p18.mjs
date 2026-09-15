// Part 18, practice and interpretability: the narration of every topic read back against an independent working-out.
// Metrics from confusion counts written out by hand (and average precision by its own sweep), cross-validation by a
// hand-written fold loop, permutation importance by a hand-written shuffle-and-score loop, partial dependence by direct
// averaging, Shapley values from their definition (every ordering of the features), LIME by its own weighted
// least-squares solve, and the project's steps recomputed from the raw table.
// Usage: node tests/claims-p18.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';

const P = S.P18;
const ALLOWED = [/50%/g, /2\.0/g];
const near = (c, a, b, msg, tol) => c.ok(Math.abs(a - b) <= (tol || 1e-9) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const vnear = (c, a, b, msg, tol) => c.ok(a && b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= (tol || 1e-9) * Math.max(1, Math.abs(b[i]))), msg + ': got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b));
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };

/** binary metrics from the four counts, and average precision by sweeping the distinct scores from the top */
function metrics(y, pred, score) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  y.forEach((t, i) => { if (pred[i] && t) tp++; else if (pred[i]) fp++; else if (t) fn++; else tn++; });
  const rec = tp + fn ? tp / (tp + fn) : 0, prec = tp + fp ? tp / (tp + fp) : 0;
  let ap = null;
  if (score) {
    const P1 = y.filter((v) => v).length, ths = [...new Set(score)].sort((a, b) => b - a);
    let prevR = 0; ap = 0;
    for (const t of ths) { const sel = y.map((v, i) => [v, score[i]]).filter(([, s]) => s >= t), tpp = sel.filter(([v]) => v).length, r = tpp / P1, p = tpp / sel.length; ap += (r - prevR) * p; prevR = r; }
  }
  return { cm: [[tn, fp], [fn, tp]], acc: (tp + tn) / y.length, recall: rec, precision: prec, f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0, bal: ((tn + fp ? tn / (tn + fp) : 0) + rec) / 2, ap };
}
/** column scaling by hand */
// (sums of v / n, as the topics accumulate them, so a tree's thresholds see bit-identical inputs)
const scale = (Xtr) => { const d = Xtr[0].length, n = Xtr.length, m = new Array(d).fill(0), s = new Array(d).fill(0); for (const r of Xtr) for (let j = 0; j < d; j++) m[j] += r[j] / n; for (const r of Xtr) for (let j = 0; j < d; j++) s[j] += (r[j] - m[j]) ** 2 / n; const q = s.map((v) => Math.sqrt(v) || 1); return (Z) => Z.map((r) => r.map((v, j) => (v - m[j]) / q[j])); };
const foldScores = (mk, X, y, k) => S.ML.stratifiedKFold(y, k).map((fd) => {
  const tr = fd.train, te = fd.test, f = scale(tr.map((i) => X[i])), m = mk().fit(f(tr.map((i) => X[i])), tr.map((i) => y[i])), p = m.predict(f(te.map((i) => X[i])));
  return te.filter((i, t) => p[t] === y[i]).length / te.length;
});
/** solve A b = c by Gaussian elimination */
const solve = (A, c) => { const n = c.length, M = A.map((r, i) => r.concat([c[i]])); for (let i = 0; i < n; i++) { let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r; [M[i], M[p]] = [M[p], M[i]]; for (let r = 0; r < n; r++) if (r !== i) { const q = M[r][i] / M[i][i]; for (let k = i; k <= n; k++) M[r][k] -= q * M[i][k]; } } return M.map((r, i) => r[n] / r[i]); };

const checks = {
  'imbalanced-data': {
    random: (r) => ({ set: r() < 0.5 ? 'toy' : 'pumps', fix: ['none', 'over', 'under', 'weights'][Math.floor(r() * 4)], seed: 1 + Math.floor(r() * 300) }),
    check(c) {
      const base = P.imbRun(c.input.set, 'none', 1), d = base.d, nR = d.y.filter((v) => v).length;
      const D0 = phase(c.frames, 'data')[0].meta;
      c.eq([D0.n, D0.nR, D0.nTr + D0.nTe, D0.nTeR], [d.y.length, nR, d.y.length, base.yte.filter((v) => v).length], 'row counts');
      const b = phase(c.frames, 'baseline')[0].meta;
      near(c, b.acc, base.yte.filter((v) => !v).length / base.yte.length, 'baseline accuracy');
      const mp = metrics(base.yte, base.pred, base.score), pl = phase(c.frames, 'plain')[0].meta;
      near(c, pl.acc, mp.acc, 'plain accuracy'); near(c, pl.recall, mp.recall, 'plain recall'); c.eq(pl.tp, mp.cm[1][1], 'rare points found');
      c.ok(base.pred.every((p, i) => p === (base.score[i] >= 0.5 ? 1 : 0)), 'predictions are the 50% cut of the scores');
      const fx = phase(c.frames, 'fix')[0];
      if (c.input.fix !== 'none') {
        const r = P.imbRun(c.input.set, c.input.fix, c.input.seed), w = r.w, rare = r.ytr.map((v, i) => (v ? w[i] : 0)).reduce((a, x) => a + x, 0), com = r.ytr.map((v, i) => (v ? 0 : w[i])).reduce((a, x) => a + x, 0);
        const nr = r.ytr.filter((v) => v).length, nc = r.ytr.length - nr;
        if (c.input.fix === 'over') { c.ok(w.every((x) => Number.isInteger(x) && x >= 1), 'copies are whole'); near(c, rare, nc, 'rare copies match the common count'); near(c, fx.meta.copies, rare + com, 'rows after copying'); }
        if (c.input.fix === 'under') { c.eq(r.ytr.filter((v, i) => !v && w[i] === 1).length, nr, 'common rows kept'); c.eq(fx.meta.kept, 2 * nr, 'rows left'); }
        if (c.input.fix === 'weights') { near(c, rare, r.ytr.length / 2, 'rare weight total'); near(c, com, r.ytr.length / 2, 'common weight total'); near(c, fx.meta.wR, r.ytr.length / (2 * nr), 'rare weight'); }
        const mf = metrics(r.yte, r.pred, r.score);
        near(c, fx.meta.recall, mf.recall, 'recall after the fix'); near(c, fx.meta.precision, mf.precision, 'precision after the fix'); near(c, fx.meta.acc, mf.acc, 'accuracy after the fix');
        const cp = phase(c.frames, 'compare')[0].meta;
        near(c, cp.balL, mf.bal, 'balanced accuracy'); near(c, cp.apL, mf.ap, 'average precision after the fix', 1e-9);
      } else c.ok(!fx, 'no fix frame');
      const cp = phase(c.frames, 'compare')[0].meta;
      near(c, cp.balB, 0.5, 'baseline balanced accuracy'); near(c, cp.apP, mp.ap, 'average precision without the fix');
    },
  },
  'model-selection': {
    random: (r) => ({ set: ['moons', 'circles', 'xor', 'loans'][Math.floor(r() * 4)], k: ['3', '5', '10'][Math.floor(r() * 3)], seed: 1 + Math.floor(r() * 200) }),
    randomRuns: 4,
    check(c) {
      const k = +c.input.k, R = P.selectRun(c.input.set, k, c.input.seed);
      const d = P.toy(c.input.set, 240, 1), sp = S.Data.split(d.y, 0.25, c.input.seed), Xd = sp.train.map((i) => d.X[i]), yd = sp.train.map((i) => d.y[i]);
      const f0 = phase(c.frames, 'folds')[0].meta;
      c.eq([f0.nDev, f0.nTest, f0.total], [yd.length, sp.test.length, d.y.length], 'row counts');
      const rows = P.MODELS.map((m) => foldScores(m.make, Xd, yd, k));
      phase(c.frames, 'model').forEach((f) => { vnear(c, f.meta.scores, rows[f.meta.i], 'fold scores of ' + P.MODELS[f.meta.i].id); near(c, f.meta.mean, mean(rows[f.meta.i]), 'mean'); near(c, f.meta.sd, sd(rows[f.meta.i]), 'spread'); });
      const means = rows.map(mean), best = means.indexOf(Math.max(...means)), se = sd(rows[best]) / Math.sqrt(k), pick = means.findIndex((m) => m >= means[best] - se - 1e-12);
      const ch = phase(c.frames, 'choose')[0].meta;
      c.eq([ch.best, ch.pick], [best, pick], 'best and one-SE choice'); near(c, ch.se, se, 'standard error'); near(c, ch.bar, means[best] - se, 'the bar');
      const t = phase(c.frames, 'test')[0].meta, f = scale(Xd), m = P.MODELS[pick].make().fit(f(Xd), yd), p = m.predict(f(sp.test.map((i) => d.X[i])));
      near(c, t.testPick, sp.test.filter((i, j) => p[j] === d.y[i]).length / sp.test.length, 'test accuracy of the choice');
    },
  },
  'no-free-lunch': {
    check(c) {
      const R = P.nflRun(+c.input.n, +c.input.k, c.input.seed);
      phase(c.frames, 'set').forEach((f) => {
        const d = R[f.meta.i].d, means = P.MODELS.map((m) => mean(foldScores(m.make, d.X, d.y, +c.input.k)));
        vnear(c, f.meta.means, means, 'cross-validated means of ' + R[f.meta.i].id);
        c.eq([f.meta.best, f.meta.worst], [means.indexOf(Math.max(...means)), means.indexOf(Math.min(...means))], 'best and worst');
      });
      const e = phase(c.frames, 'end')[0].meta, wins = R.map((r) => r.means.indexOf(Math.max(...r.means)));
      c.eq(e.distinct, new Set(wins).size, 'distinct winners');
      const ranks = P.MODELS.map((_, j) => mean(R.map((r) => 1 + r.means.filter((v) => v > r.means[j]).length)));
      near(c, e.rank, Math.min(...ranks), 'best average rank');
      near(c, e.linXor, R[3].means[1], 'logistic on XOR');
    },
  },
  'permutation-importance': {
    random: (r) => ({ set: ['loans', 'copy', 'rents'][Math.floor(r() * 3)], reps: ['3', '5'][Math.floor(r() * 2)], seed: 1 + Math.floor(r() * 300) }),
    randomRuns: 3,
    check(c) {
      const T = P.tableRun(c.input.set), reps = +c.input.reps, rnd = S.Data.rng(c.input.seed), X = T.Xte, y = T.yte, n = X.length, d = X[0].length;
      const base = T.score(y, T.predict(X)), drops = [];
      for (let j = 0; j < d; j++) { drops.push([]); for (let r = 0; r < reps; r++) { const col = X.map((x) => x[j]); for (let i = n - 1; i > 0; i--) { const k = Math.floor(rnd() * (i + 1)); const t = col[i]; col[i] = col[k]; col[k] = t; } drops[j].push(base - T.score(y, T.predict(X.map((x, i) => { const z = x.slice(); z[j] = col[i]; return z; })))); } }
      near(c, phase(c.frames, 'base')[0].meta.baseline, base, 'baseline score');
      const sh = phase(c.frames, 'shuffle')[0].meta;
      near(c, sh.drop, drops[sh.j][0], 'first shuffle drop'); near(c, sh.after, base - drops[sh.j][0], 'score after the shuffle');
      const rp = phase(c.frames, 'repeats')[0].meta;
      near(c, rp.mean, mean(drops[rp.j]), 'mean drop'); near(c, rp.std, sd(drops[rp.j]), 'spread of drops'); near(c, rp.min, Math.min(...drops[rp.j]), 'smallest drop');
      const all = phase(c.frames, 'all')[0].meta;
      vnear(c, all.mean, drops.map(mean), 'importances');
      c.eq(all.order[0], drops.map(mean).indexOf(Math.max(...drops.map(mean))), 'the top feature');
    },
  },
  'partial-dependence': {
    extra: [{ case: 'rain', rows: '15' }, { case: 'twist', rows: '50' }],
    check(c) {
      const k = P.PDCASES[c.input.case], D = P.pdModel(k.set), j = k.j, X = D.X.slice(0, +c.input.rows);
      const one = phase(c.frames, 'one')[0].meta, pd = phase(c.frames, 'pdp')[0].meta;
      const sweep = (x, v) => { const z = x.slice(); z[j] = v; return D.predict([z])[0]; };
      near(c, one.a, sweep(X[0], one.g0), 'row 0 at the lowest grid value'); near(c, one.b, sweep(X[0], one.g1), 'row 0 at the highest grid value');
      near(c, one.curPred, D.predict([X[0]])[0], 'row 0 prediction');
      const avg = (v) => mean(X.map((x) => sweep(x, v)));
      near(c, pd.lo, avg(pd.glo), 'partial dependence at its low point'); near(c, pd.hi, avg(pd.ghi), 'partial dependence at its high point');
      near(c, pd.rise, pd.hi - pd.lo, 'rise');
    },
  },
  'shap-waterfall': {
    random: (r) => ({ case: ['loans', 'rents', 'tree'][Math.floor(r() * 3)], row: Math.floor(r() * 20), bg: ['10', '30'][Math.floor(r() * 2)] }),
    randomRuns: 4,
    check(c) {
      const Mo = P.shapModel(c.input.case), x = Mo.Xte[c.input.row], bg = Mo.Xtr.slice(0, +c.input.bg), M = x.length;
      const v = (mask) => mean(Mo.f(bg.map((b) => b.map((bv, j) => ((mask >>> j) & 1 ? x[j] : bv)))));
      const val = []; for (let m = 0; m < 1 << M; m++) val.push(v(m));
      // every ordering of the features: the mean change each one makes when it joins
      const perms = []; const permute = (a, k) => { if (k === a.length) { perms.push(a.slice()); return; } for (let i = k; i < a.length; i++) { [a[k], a[i]] = [a[i], a[k]]; permute(a, k + 1); [a[k], a[i]] = [a[i], a[k]]; } };
      permute([...Array(M).keys()], 0);
      const phi = new Array(M).fill(0);
      for (const p of perms) { let mask = 0; for (const i of p) { phi[i] += (val[mask | (1 << i)] - val[mask]) / perms.length; mask |= 1 << i; } }
      const co = phase(c.frames, 'coal')[0].meta;
      near(c, co.phi, phi[co.top], 'Shapley value of the top feature', 1e-6);
      co.sizes.forEach((m, k) => { const ms = []; for (let s = 0; s < 1 << M; s++) { if ((s >>> co.top) & 1) continue; let pc = 0; for (let t = s; t; t >>>= 1) pc += t & 1; if (pc === k) ms.push(val[s | (1 << co.top)] - val[s]); } near(c, m, mean(ms), 'mean change with ' + k + ' others in', 1e-6); c.eq(co.counts[k], ms.length, 'coalitions of size ' + k); });
      const row = phase(c.frames, 'row')[0].meta, chk = phase(c.frames, 'check')[0].meta;
      near(c, chk.sum, chk.base + chk.phi.reduce((a, b) => a + b, 0), 'contributions add up');
      near(c, chk.sum, row.fx, 'the sum is the output', 1e-6);
      if (c.input.case === 'tree') {
        near(c, row.fx, Mo.f([x])[0], 'the tree’s answer');
        const N = Mo.tree.nodes, ex = (j) => (N[j].left < 0 ? N[j].value : (N[N[j].left].cover * ex(N[j].left) + N[N[j].right].cover * ex(N[j].right)) / N[j].cover);
        near(c, row.base, ex(0), 'the tree’s expected value');
      } else {
        near(c, row.base, val[0], 'average output'); near(c, row.fx, val[(1 << M) - 1], 'this row’s output');
        vnear(c, chk.phi, phi, 'every Shapley value', 1e-6);
        c.ok(chk.gap < 1e-6, 'Kernel SHAP agrees');
      }
      let cum = row.base;
      phase(c.frames, 'add').forEach((f) => { cum += f.meta.signed; near(c, f.meta.to, cum, 'running total'); });
    },
  },
  'lime-local': {
    random: (r) => ({ set: r() < 0.5 ? 'moons' : 'circles', point: ['edge', 'bend', 'deep'][Math.floor(r() * 3)], width: Math.round((0.2 + r() * 2.8) * 10) / 10, samples: ['100', '200'][Math.floor(r() * 2)] }),
    randomRuns: 5,
    check(c) {
      const Mo = P.limeModel(c.input.set), x = P.LIMESETS[c.input.set].points[c.input.point], n = +c.input.samples;
      const L = S.Explain.lime(Mo.f, x, Mo.X, { nSamples: n, seed: 1, kernelWidth: c.input.width });
      const sdv = [0, 1].map((j) => sd(Mo.X.map((r) => r[j]))), Z = L.samples, y = Mo.f(Z);
      const w = Z.map((z) => Math.exp(-(((z[0] - x[0]) / sdv[0]) ** 2 + ((z[1] - x[1]) / sdv[1]) ** 2) / (c.input.width * c.input.width)));
      vnear(c, L.weights, w, 'weights from the kernel', 1e-9);
      // weighted ridge on scaled offsets (intercept free), solved by hand
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bb = [0, 0, 0];
      Z.forEach((z, k) => { const r = [1, (z[0] - x[0]) / sdv[0], (z[1] - x[1]) / sdv[1]]; for (let i = 0; i < 3; i++) { bb[i] += w[k] * r[i] * y[k]; for (let j = 0; j < 3; j++) A[i][j] += w[k] * r[i] * r[j]; } });
      A[1][1] += 1e-3; A[2][2] += 1e-3;
      const beta = solve(A, bb), coef = [beta[1] / sdv[0], beta[2] / sdv[1]];
      const fit = phase(c.frames, 'fit')[0].meta;
      near(c, fit.intercept, beta[0], 'intercept', 1e-6); vnear(c, fit.coef, coef, 'slopes', 1e-6);
      const g = Z.map((z) => beta[0] + coef[0] * (z[0] - x[0]) + coef[1] * (z[1] - x[1])), W = w.reduce((a, b) => a + b, 0), ym = y.reduce((s, v, k) => s + w[k] * v, 0) / W;
      near(c, fit.score, 1 - y.reduce((s, v, k) => s + w[k] * (v - g[k]) ** 2, 0) / y.reduce((s, v, k) => s + w[k] * (v - ym) ** 2, 0), 'weighted R²', 1e-6);
      c.eq(phase(c.frames, 'weights')[0].meta.near, w.filter((v) => v > 0.5).length, 'samples over one half');
      near(c, phase(c.frames, 'model')[0].meta.fx, Mo.f([x])[0], 'score at the point');
    },
  },
  'project-checklist': {
    extra: [{ missing: '0.1', seed: 7 }],
    check(c) {
      const R = P.projectRun(+c.input.missing, c.input.seed), d = S.Data.load('loans', { n: 600, seed: c.input.seed, missing: +c.input.missing });
      const au = phase(c.frames, 'audit')[0].meta;
      c.eq(au.cells, d.X.reduce((s, r) => s + r.slice(0, 5).filter((v) => Number.isNaN(v)).length, 0), 'missing cells');
      const sp = S.Data.split(d.y, 0.25, 1), med = [0, 1, 2, 3, 4].map((j) => { const v = sp.train.map((i) => d.X[i][j]).filter((u) => !Number.isNaN(u)).sort((a, b) => a - b); const m = v.length; return m % 2 ? v[(m - 1) / 2] : (v[m / 2 - 1] + v[m / 2]) / 2; });
      vnear(c, phase(c.frames, 'split')[0].meta.med, med, 'training medians');
      const fill = (r) => r.slice(0, 5).map((v, j) => (Number.isNaN(v) ? med[j] : v)), Xtr = sp.train.map((i) => fill(d.X[i])), ytr = sp.train.map((i) => d.y[i]), Xte = sp.test.map((i) => fill(d.X[i])), yte = sp.test.map((i) => d.y[i]);
      near(c, phase(c.frames, 'baseline')[0].meta.base, yte.filter((v) => !v).length / yte.length, 'baseline accuracy');
      const f = scale(Xtr), m = new S.ML.LogisticRegression({}).fit(f(Xtr), ytr), pred = m.predict(f(Xte)), mt = metrics(yte, pred, null);
      const mo = phase(c.frames, 'model')[0].meta;
      near(c, mo.acc, mt.acc, 'model accuracy'); near(c, mo.recall, mt.recall, 'recall'); near(c, mo.precision, mt.precision, 'precision');
      const er = phase(c.frames, 'errors')[0].meta;
      c.eq([er.fn, er.fp], [mt.cm[1][0], mt.cm[0][1]], 'missed defaults and false alarms');
      const va = phase(c.frames, 'validate')[0].meta, cv = foldScores(() => new S.ML.LogisticRegression({}), Xtr, ytr, 5);
      vnear(c, va.cv, cv, 'fold accuracies'); near(c, va.mean, mean(cv), 'cv mean'); near(c, va.sd, sd(cv), 'cv spread');
      const fa = phase(c.frames, 'fair')[0].meta, g = sp.test.map((i) => d.X[i][5]);
      const rate = (G) => { const idx = g.map((v, i) => (v === G ? i : -1)).filter((i) => i >= 0); return [idx.filter((i) => pred[i]).length / idx.length, idx.filter((i) => pred[i] && yte[i]).length / idx.filter((i) => yte[i]).length]; };
      vnear(c, [fa.selA, fa.tprA], rate(0), 'group A rates'); vnear(c, [fa.selB, fa.tprB], rate(1), 'group B rates');
      near(c, phase(c.frames, 'monitor')[0].meta.shareTest, pred.filter((v) => v).length / pred.length, 'share flagged on the test rows');
    },
  },
};

run('claims-p18', /^p18-/, checks, ALLOWED);
