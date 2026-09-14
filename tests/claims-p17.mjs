// Part 17, robustness, fairness and privacy: the narration of every topic read back against an independent working-
// out. Input gradients by finite differences, sign steps and flips from fresh replays of the training, smoothing votes
// from the engine's own vote, drift scores with a second decile routine, group confusion tables counted by hand from
// a replayed model, reweighing weights from their formula, the first DP-SGD step rebuilt from the same random draws
// with gradients in closed form and the accountant written out again, membership advantages by brute force, and the
// neighbour vote by sorting distances.
// Usage: node tests/claims-p17.mjs [topic id prefix ...]
import { S, run, phase } from './claims-lib-p06-08.mjs';

const P = S.P17, R = S.Rob, T = S.T;
const ALLOWED = [];
const near = (c, a, b, msg, tol) => c.ok(Math.abs(a - b) <= (tol || 1e-9) * Math.max(1, Math.abs(b)), msg + ': got ' + a + ', want ' + b);
const vnear = (c, a, b, msg, tol) => c.ok(a && b && a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= (tol || 1e-9) * Math.max(1, Math.abs(b[i]))), msg + ': got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b));
const sig = (z) => 1 / (1 + Math.exp(-z));
const lin = (m, x) => sig(m.b + x.reduce((s, v, j) => s + v * m.w[j], 0));
const fresh = (seed, eps) => { const c = R.classifier({ data: 'moons', seed, hidden: 16, steps: 300, n: 200, nTest: 200, adv: eps ? { eps } : undefined }); c.run(300); return c; };
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

const checks = {
  'robust-brittle': {
    random: (r) => ({ which: ['near', 'typical', 'far'][Math.floor(r() * 3)], eps: Math.round((0.02 + r() * 0.58) * 100) / 100, seed: 1 + Math.floor(r() * 50) }),
    randomRuns: 4,
    check(c) {
      const clf = fresh(c.input.seed), d = clf.data, st = phase(c.frames, 'start')[0].meta;
      near(c, st.acc, clf.accuracy(), 'test accuracy replayed', 1e-12);
      const pr = clf.probs(d.Xte), ok = d.Xte.map((_, i) => i).filter((i) => (pr[i][1] > pr[i][0] ? 1 : 0) === d.yte[i]).map((i) => pr[i][d.yte[i]]).sort((a, b) => a - b);
      const want = c.input.which === 'near' ? ok[0] : c.input.which === 'far' ? ok[ok.length - 1] : ok[Math.floor(ok.length / 2)];
      near(c, pr[st.i][d.yte[st.i]], want, 'the chosen point’s confidence', 1e-12);
      const x = d.Xte[st.i], y = d.yte[st.i], loss = (z) => T.noGrad(() => T.crossEntropy(clf.model.forward(T.tensor([z])), [y]).item());
      const fd = x.map((_, j) => { const a = x.slice(), b = x.slice(); a[j] += 1e-3; b[j] -= 1e-3; return (loss(a) - loss(b)) / 2e-3; });
      const g = phase(c.frames, 'grad')[0].meta;
      vnear(c, g.g, fd, 'input gradient against finite differences', 2e-2);
      near(c, g.gl, Math.hypot(...g.g), 'gradient length');
      const xa = phase(c.frames, 'step')[0].meta.xa;
      vnear(c, xa, x.map((v, j) => v + c.input.eps * Math.sign(g.g[j])), 'the sign step', 1e-12);
      const pa = clf.probs([xa])[0], af = phase(c.frames, 'after')[0].meta;
      c.eq(af.after, pa[1] > pa[0] ? 1 : 0, 'class after the step');
      c.eq(af.flipped, af.after !== st.before, 'flipped or not');
      const sm = phase(c.frames, 'smallest')[0].meta;
      if (sm.epsStar !== null) {
        const at = (e) => clf.predict([x.map((v, j) => v + e * Math.sign(g.g[j]))])[0];
        c.ok(at(sm.epsStar) !== st.before && at(Math.max(0, sm.epsStar - 1e-6)) === st.before, 'the smallest flipping step brackets the flip');
      }
      near(c, sm.span, Math.max(...d.Xtr.map((r) => r[0])) - Math.min(...d.Xtr.map((r) => r[0])), 'span of the data');
    },
  },
  'robust-defend': {
    random: (r) => ({ eps: [0.1, 0.15, 0.2, 0.25, 0.3][Math.floor(r() * 5)], sigma: Math.round((0.1 + r() * 0.5) * 20) / 20, seed: 1 + Math.floor(r() * 20) }),
    randomRuns: 3,
    check(c) {
      const plain = fresh(c.input.seed), adv = fresh(c.input.seed, c.input.eps), d = plain.data, e = c.input.eps;
      near(c, phase(c.frames, 'plain')[0].meta.accP, plain.accuracy(), 'plain accuracy replayed', 1e-12);
      const curve = (m) => R.robustCurve(m, [e], { n: 200 })[0].accuracy;
      near(c, phase(c.frames, 'plainAtt')[0].meta.attP, curve(plain), 'plain accuracy under nudges (robust curve)', 1e-12);
      near(c, phase(c.frames, 'adv')[0].meta.accA, adv.accuracy(), 'adversarial accuracy replayed', 1e-12);
      near(c, phase(c.frames, 'advAtt')[0].meta.attA, curve(adv), 'adversarial accuracy under nudges', 1e-12);
      const sm = phase(c.frames, 'smooth')[0].meta;
      if (sm.xp) {
        const v = R.smoothPredict(plain, sm.xp, { sigma: c.input.sigma, samples: 32 });
        c.eq(sm.votes, v.votes, 'smoothing votes'); c.eq(sm.win, v.label, 'smoothed answer');
        c.ok(plain.predict([sm.xp])[0] !== sm.y, 'the nudged point fooled the plain model');
      }
      const tb = phase(c.frames, 'table')[0].meta, df = R.defences(plain, e, { sigma: c.input.sigma, samples: 32, n: 200 });
      vnear(c, tb.rows.flatMap((r) => [r.clean, r.attacked]), df.rows.flatMap((r) => [r.clean, r.attacked]), 'defence table', 1e-12);
    },
  },
  'shift-kinds': {
    random: (r) => ({ kind: ['covariate', 'label', 'concept'][Math.floor(r() * 3)], amount: Math.round(r() * 20) / 20, seed: 1 + Math.floor(r() * 99) }),
    check(c) {
      const base = R.shift({ kind: c.input.kind, amount: 0, seed: c.input.seed }), m = R.logreg(base.train.X, base.train.y, { iters: 300 });
      const acc = (s) => s.X.filter((x, i) => (lin(m, x) >= 0.5 ? 1 : 0) === s.y[i]).length / s.y.length;
      near(c, phase(c.frames, 'train')[0].meta.accTr, acc(base.train), 'training accuracy');
      const s = R.shift({ kind: c.input.kind, amount: c.input.amount, seed: c.input.seed }).test, t = phase(c.frames, 'test')[0].meta;
      near(c, t.mx, mean(s.X.map((x) => x[0])), 'test mean of x₁'); near(c, t.share, mean(s.y), 'test share of class 1');
      near(c, t.turn, c.input.kind === 'concept' ? Math.PI * c.input.amount : 0, 'phase of the rule');
      near(c, phase(c.frames, 'acc')[0].meta.accTe, acc(s), 'accuracy after the shift');
      const dr = phase(c.frames, 'kinds')[0].meta.drops;
      R.shiftKinds.forEach((k, j) => {
        const b = R.shift({ kind: k, amount: 0, seed: c.input.seed }), mk = R.logreg(b.train.X, b.train.y, { iters: 300 });
        const a = [0, 0.25, 0.5, 0.75, 1].map((am) => { const z = R.shift({ kind: k, amount: am, seed: c.input.seed }).test; return z.X.filter((x, i) => (lin(mk, x) >= 0.5 ? 1 : 0) === z.y[i]).length / z.y.length; });
        near(c, dr[j], a[0] - Math.min(...a), 'worst drop for ' + k);
      });
    },
  },
  'shift-monitor': {
    random: (r) => ({ scen: ['covariate', 'concept', 'label', 'none'][Math.floor(r() * 4)], speed: [0.04, 0.08, 0.12, 0.16, 0.2][Math.floor(r() * 5)], thr: [0.1, 0.15, 0.2, 0.25, 0.3][Math.floor(r() * 5)], seed: 1 + Math.floor(r() * 50) }),
    check(c) {
      const kind = c.input.scen === 'none' ? 'covariate' : c.input.scen, base = R.shift({ kind, amount: 0, seed: c.input.seed }), m = R.logreg(base.train.X, base.train.y, { iters: 300 });
      // deciles a second way: the value below which k tenths of the sorted column lie
      const cuts = [0, 1].map((j) => { const v = base.train.X.map((x) => x[j]).sort((a, b) => a - b); return [1, 2, 3, 4, 5, 6, 7, 8, 9].map((k) => v[Math.floor((k * v.length) / 10)]); });
      const share = (vals, e) => { const cnt = new Array(10).fill(0); vals.forEach((x) => { cnt[e.filter((t) => x >= t).length]++; }); return cnt.map((k) => Math.max(1e-4, k / vals.length)); };
      const ref = [0, 1].map((j) => share(base.train.X.map((x) => x[j]), cuts[j]));
      let alarms = 0, first = null;
      phase(c.frames, 'batch').forEach((f) => {
        const b = f.meta.b, amount = c.input.scen === 'none' ? 0 : Math.min(1, c.input.speed * b), s = R.shift({ kind, amount, n: 150, seed: c.input.seed * 100 + b }).test;
        const psi = [0, 1].map((j) => { const now = share(s.X.map((x) => x[j]), cuts[j]); return now.reduce((t, v, k) => t + (v - ref[j][k]) * Math.log(v / ref[j][k]), 0); });
        vnear(c, f.meta.psi, psi, 'drift scores of batch ' + b, 1e-9);
        c.eq(f.meta.alarm, Math.max(...psi) > c.input.thr, 'alarm at batch ' + b);
        if (Math.max(...psi) > c.input.thr) { alarms++; if (first === null) first = b; }
        near(c, f.meta.acc, s.X.filter((x, i) => (lin(m, x) >= 0.5 ? 1 : 0) === s.y[i]).length / s.y.length, 'accuracy of batch ' + b);
      });
      const e = phase(c.frames, 'end')[0].meta;
      c.eq([e.alarms, e.first], [alarms, first], 'alarms and the first');
      near(c, e.worstAcc, Math.min(...phase(c.frames, 'batch').map((f) => f.meta.acc)), 'lowest accuracy');
    },
  },
  'fair-metrics': {
    random: (r) => ({ useGroup: r() < 0.5 ? 'yes' : 'no', seed: 1 + Math.floor(r() * 30) }),
    randomRuns: 4,
    check(c) {
      const L = R.loansData(c.input.seed), ytr = L.ytr.map((v) => 1 - v), yte = L.yte.map((v) => 1 - v);
      const counts = (useG) => {
        const add = (X, G) => (useG ? X.map((x, i) => x.concat([G[i] ? 1 : -1])) : X), m = R.logreg(add(L.Xtr, L.gtr), ytr, { iters: 400 }), pred = add(L.Xte, L.gte).map((x) => (lin(m, x) >= 0.5 ? 1 : 0));
        return [0, 1].map((g) => { const o = { tp: 0, fp: 0, fn: 0, tn: 0, n: 0 }; pred.forEach((p, i) => { if (L.gte[i] !== g) return; o.n++; if (p && yte[i]) o.tp++; else if (p) o.fp++; else if (yte[i]) o.fn++; else o.tn++; }); return o; });
      };
      const rates = (G) => ({ sel: G.map((o) => (o.tp + o.fp) / o.n), tpr: G.map((o) => o.tp / (o.tp + o.fn)), fpr: G.map((o) => o.fp / (o.fp + o.tn)) });
      const G = counts(c.input.useGroup === 'yes'), rt = rates(G);
      phase(c.frames, 'group').forEach((f) => c.eq([f.meta.tp, f.meta.fn, f.meta.fp, f.meta.tn], [G[f.meta.g].tp, G[f.meta.g].fn, G[f.meta.g].fp, G[f.meta.g].tn], 'confusion table of group ' + f.meta.g));
      const b = phase(c.frames, 'base')[0].meta;
      vnear(c, b.base, [0, 1].map((g) => { const idx = L.gte.map((x, i) => (x === g ? i : -1)).filter((i) => i >= 0); return idx.filter((i) => yte[i]).length / idx.length; }), 'repaid shares');
      near(c, phase(c.frames, 'parity')[0].meta.dp, Math.abs(rt.sel[0] - rt.sel[1]), 'parity difference');
      near(c, phase(c.frames, 'odds')[0].meta.eo, Math.max(Math.abs(rt.tpr[0] - rt.tpr[1]), Math.abs(rt.fpr[0] - rt.fpr[1])), 'equalised-odds difference');
      const o = rates(counts(c.input.useGroup !== 'yes')), h = phase(c.frames, 'hide')[0].meta;
      near(c, h.dpO, Math.abs(o.sel[0] - o.sel[1]), 'parity difference with the other setting');
    },
  },
  'fair-mitigate': {
    random: (r) => ({ target: r() < 0.5 ? 'parity' : 'opportunity', seed: 1 + Math.floor(r() * 30) }),
    randomRuns: 4,
    check(c) {
      const L = R.loansData(c.input.seed), ytr = L.ytr.map((v) => 1 - v), yte = L.yte.map((v) => 1 - v), n = ytr.length;
      const w = phase(c.frames, 'weights')[0].meta, want = [];
      for (const g of [0, 1]) for (const y of [0, 1]) { const ng = L.gtr.filter((v) => v === g).length, ny = ytr.filter((v) => v === y).length, nb = ytr.filter((v, i) => v === y && L.gtr[i] === g).length; want.push((ng / n) * (ny / n) / (nb / n)); }
      vnear(c, w.weights, want, 'reweighing weights', 1e-12);
      const th = phase(c.frames, 'threshold')[0].meta, m = R.logreg(L.Xtr.map((x, i) => x.concat([L.gtr[i] ? 1 : -1])), ytr, { iters: 400 });
      const pred = L.Xte.map((x, i) => (lin(m, x.concat([L.gte[i] ? 1 : -1])) >= th.thr[L.gte[i]] ? 1 : 0));
      const rate = (g, f) => { const idx = pred.map((_, i) => i).filter((i) => L.gte[i] === g && f(i)); return idx.filter((i) => pred[i]).length / idx.length; };
      const sel = [0, 1].map((g) => rate(g, () => true)), tpr = [0, 1].map((g) => rate(g, (i) => yte[i] === 1)), fpr = [0, 1].map((g) => rate(g, (i) => yte[i] === 0));
      near(c, th.dp, Math.abs(sel[0] - sel[1]), 'parity difference with cut-offs');
      near(c, th.eo, Math.max(Math.abs(tpr[0] - tpr[1]), Math.abs(fpr[0] - fpr[1])), 'equalised-odds difference with cut-offs');
      near(c, th.acc, pred.filter((p, i) => p === yte[i]).length / yte.length, 'accuracy with cut-offs');
      const cm = phase(c.frames, 'compare')[0].meta;
      near(c, cm.bestDp, Math.min(...cm.dps)); near(c, cm.bestEo, Math.min(...cm.eos));
    },
  },
  'dp-sgd': {
    random: (r) => ({ sigma: Math.round(r() * 16) / 2, clip: 0.5 + Math.round(r() * 10) / 4, seed: 1 + Math.floor(r() * 99) }),
    check(c) {
      const L = R.loansData(1), X = L.Xtr, y = L.ytr, n = X.length, g = T.rng(c.input.seed), B = 16, C = c.input.clip;
      const idx = Array.from({ length: B }, () => g.int(n));
      // at the first step the weights are zero, so p = ½ and each gradient is (½ − y)·[x, 1]
      const G = idx.map((i) => X[i].map((v) => (0.5 - y[i]) * v).concat([0.5 - y[i]])), norms = G.map((r) => Math.hypot(...r));
      const gm = phase(c.frames, 'grads')[0].meta;
      near(c, gm.min, Math.min(...norms), 'shortest gradient', 1e-9); near(c, gm.max, Math.max(...norms), 'longest gradient', 1e-9);
      c.eq(phase(c.frames, 'clip')[0].meta.clipped, norms.filter((v) => v > C).length, 'gradients clipped');
      const sum = new Array(6).fill(0); G.forEach((r, k) => { const f = Math.min(1, C / norms[k]); r.forEach((v, j) => { sum[j] += v * f; }); });
      const noise = sum.map(() => g.normal(0, c.input.sigma * C)), nm = phase(c.frames, 'noise')[0].meta;
      near(c, nm.sumLen, Math.hypot(...sum), 'length of the clipped sum', 1e-9); near(c, nm.noiseLen, Math.hypot(...noise), 'length of the noise', 1e-9);
      const tm = phase(c.frames, 'train')[0].meta;
      near(c, tm.upd, Math.hypot(...sum.map((v, j) => (v + noise[j]) / B)), 'length of the step', 1e-9);
      c.eq(tm.steps, Math.round((3 * n) / B), 'steps in three epochs');
      // the accountant written out again: per step min(α / 2σ², 2q²α / σ²), plus log(1/δ) / (α − 1), best α
      const epsOf = (s) => { if (!s) return null; let best = Infinity; for (const a of [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 256]) best = Math.min(best, tm.steps * Math.min(a / (2 * s * s), (2 * (B / n) ** 2 * a) / (s * s)) + Math.log(1e5) / (a - 1)); return best; };
      if (c.input.sigma) near(c, tm.eps, epsOf(c.input.sigma), 'approximate ε', 1e-9); else c.ok(tm.eps === null, 'no ε without noise');
      const r = R.dpsgd({ sigma: c.input.sigma, clip: C, seed: c.input.seed, data: L });
      near(c, tm.acc, r.accuracy, 'accuracy replayed', 1e-12);
      const tr = phase(c.frames, 'trade')[0].meta, accN = mean([1, 2, 3].map((seed) => R.dpsgd({ sigma: 8, clip: C, seed, data: L }).accuracy));
      near(c, tr.accN, accN, 'accuracy at σ = 8, mean of three', 1e-12); near(c, tr.epsN, epsOf(8), 'ε at σ = 8', 1e-9);
    },
  },
  'membership-inference': {
    random: (r) => ({ model: ['deep', 'depth4', 'leaf20', 'logistic', 'dp'][Math.floor(r() * 5)], seed: 1 + Math.floor(r() * 20) }),
    randomRuns: 4,
    check(c) {
      const L = R.loansData(c.input.seed), k = c.input.model;
      let pTr, pTe;
      if (k === 'logistic' || k === 'dp') { const m = k === 'dp' ? R.dpsgd({ sigma: 1, data: L, seed: 1 }) : R.logreg(L.Xtr, L.ytr, {}); pTr = L.Xtr.map((x) => lin(m, x)); pTe = L.Xte.map((x) => lin(m, x)); }
      else { const t = new S.ML.DecisionTreeClassifier(k === 'deep' ? {} : k === 'depth4' ? { maxDepth: 4 } : { minSamplesLeaf: 20 }).fit(L.Xtr, L.ytr); pTr = t.predictProba(L.Xtr).map((p) => p[1]); pTe = t.predictProba(L.Xte).map((p) => p[1]); }
      const mem = pTr.map((p, i) => (L.ytr[i] ? p : 1 - p)), non = pTe.map((p, i) => (L.yte[i] ? p : 1 - p));
      const h = phase(c.frames, 'hist')[0].meta;
      near(c, h.mMem, mean(mem), 'mean confidence of members'); near(c, h.mNon, mean(non), 'mean confidence of unseen rows');
      // brute force over every cut between distinct confidences
      let best = 0; for (const t of mem.concat(non)) best = Math.max(best, mem.filter((v) => v >= t).length / mem.length - non.filter((v) => v >= t).length / non.length);
      const gm = phase(c.frames, 'guess')[0].meta;
      near(c, gm.adv, best, 'membership advantage');
      near(c, gm.tpr - gm.fpr, gm.adv, 'advantage is TPR − FPR');
      const f = phase(c.frames, 'fit')[0].meta;
      near(c, f.accTr, pTr.filter((p, i) => (p >= 0.5 ? 1 : 0) === L.ytr[i]).length / L.ytr.length, 'training accuracy');
      near(c, f.accTe, pTe.filter((p, i) => (p >= 0.5 ? 1 : 0) === L.yte[i]).length / L.yte.length, 'test accuracy');
    },
  },
  'data-poisoning': {
    random: (r) => ({ share: Math.round(r() * 6) / 20, k: 3 + Math.floor(r() * 13), seed: 1 + Math.floor(r() * 50) }),
    check(c) {
      const D = P.lineData(c.input.seed), n = D.Xtr.length;
      const acc = (m) => D.Xte.filter((x, i) => (lin(m, x) >= 0.5 ? 1 : 0) === D.yte[i]).length / D.yte.length;
      near(c, phase(c.frames, 'clean')[0].meta.acc0, acc(R.logreg(D.Xtr, D.ytr, {})), 'clean accuracy');
      const cand = D.Xtr.map((x, i) => i).filter((i) => D.ytr[i] === 0 && D.Xtr[i][1] > 0), order = S.Data.rng(c.input.seed + 4).shuffle(cand.slice());
      const flip = new Set(order.slice(0, Math.min(cand.length, Math.round(c.input.share * n)))), yp = D.ytr.map((v, i) => (flip.has(i) ? 1 : v));
      const po = phase(c.frames, 'poison')[0].meta;
      c.eq(po.flipped, flip.size, 'labels flipped');
      near(c, po.acc1, acc(R.logreg(D.Xtr, yp, {})), 'poisoned accuracy');
      // the vote by sorting distances: the k nearest, the point itself included
      const keep = D.Xtr.map((x, i) => { const nb = D.Xtr.map((z, j) => [Math.hypot(z[0] - x[0], z[1] - x[1]), j]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, c.input.k); return nb.filter(([, j]) => yp[j] === yp[i]).length / c.input.k >= 0.5; });
      const v = phase(c.frames, 'vote')[0].meta, dropped = keep.map((k, i) => (k ? -1 : i)).filter((i) => i >= 0);
      c.eq(v.dropped, dropped.length, 'points set aside'); c.eq(v.caught, dropped.filter((i) => flip.has(i)).length, 'flipped labels caught');
      near(c, phase(c.frames, 'retrain')[0].meta.acc2, acc(R.logreg(D.Xtr.filter((_, i) => keep[i]), yp.filter((_, i) => keep[i]), {})), 'accuracy after cleaning');
    },
  },
};

run('claims-p17', /^p17-/, checks, ALLOWED);
