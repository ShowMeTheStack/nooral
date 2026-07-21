// Robustness, fairness and privacy: input gradients and per-example gradients against finite differences, the
// brittleness of a toy classifier (FGSM-style), adversarial training and input defences helping, distribution shift
// lowering accuracy, DP-SGD's privacy–accuracy trade-off and its approximate accountant, fairness mitigations
// narrowing the gaps measured by S.Metrics.fairness, and determinism.
import { load, harness } from './load.mjs';
const S = load(['core']), T = S.T, R = S.Rob;
const t = harness('rob');
const cpu = () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
const timed = (name, budgetMs, fn) => {
  const t0 = cpu(), r = fn(), ms = cpu() - t0;
  console.log(`  ${name.padEnd(34)} ${r}  in ${(ms / 1000).toFixed(2)} s`);
  t.ok(ms < budgetMs, name + ' took ' + ms.toFixed(0) + ' ms (budget ' + budgetMs + ')');
};

// ---- gradient checks ----
T.float64(() => {
  const c = R.classifier({ data: 'moons', seed: 2 });
  c.run(50);
  for (const [k, x] of [[0, [0.3, -0.2]], [1, [1.1, 0.4]], [2, [-0.5, 0.9]]]) {
    const y = k % 2, { grad } = R.inputGradient(c, x, y), loss = (z) => T.noGrad(() => T.crossEntropy(c.model.forward(T.tensor([z])), [y]).item());
    const num = x.map((_, j) => { const a = x.slice(), b = x.slice(); a[j] += 1e-6; b[j] -= 1e-6; return (loss(a) - loss(b)) / 2e-6; });
    t.ok(num.every((v, j) => Math.abs(v - grad[j]) < 1e-6 * Math.max(1, Math.abs(v))), 'input gradient ' + grad + ' against ' + num);
  }
  const X = [[0.2, 0.4], [1, -0.3], [-0.6, 0.1]], y = [1, 0, 1], batch = R.fgsmBatch(c, X, y, 0.1);
  t.ok(batch.every((xa, i) => xa.every((v, j) => Math.abs(v - R.fgsm(c, X[i], y[i], 0.1).x[j]) < 1e-9)), 'batched FGSM equals the one-at-a-time version');
  // per-example logistic gradients in closed form against autodiff
  const w = [0.3, -0.7, 0.2], b = 0.1, Xe = [[1, 2, -1], [0.5, -0.3, 0.8]], ye = [1, 0], G = R.perExampleGrads(w, b, Xe, ye);
  Xe.forEach((x, i) => {
    const wt = T.param(T.tensor(w)), bt = T.param(T.tensor([b]));
    T.bceWithLogits(T.add(T.sum(T.mul(wt, T.tensor(x))), bt), T.tensor([ye[i]])).backward();
    const auto = Array.from(wt.grad.values()).concat([bt.grad.values()[0]]);
    t.ok(auto.every((v, j) => Math.abs(v - G[i][j]) < 1e-9), 'per-example gradient ' + i + ' in closed form matches autodiff');
  });
});

// ---- brittleness and defences ----
timed('toy classifier is brittle (2-D)', 15000, () => {
  const c = R.classifier({ data: 'moons', seed: 1 }); c.run(800);
  const acc = c.accuracy();
  t.ok(acc >= 0.95, 'moons classifier accuracy ' + acc);
  let flips = 0, smaller = 0;
  for (let i = 0; i < 20; i++) {
    const x = c.data.Xte[i], y = c.data.yte[i];
    if (c.predict([x])[0] !== y) continue;
    const f = R.smallestFlip(c, x, y);
    if (f && f.after !== f.before) { flips++; if (f.eps < 0.5) smaller++; }
  }
  t.ok(flips >= 15 && smaller >= 12, 'a nudge of under 0.5 flips ' + smaller + ' of ' + flips + ' correctly classified points');
  const f = R.fgsm(c, c.data.Xte[0], c.data.yte[0], 0.15);
  t.ok(f.trace.length === 4 && f.delta.every((d) => Math.abs(Math.abs(d) - 0.15) < 1e-12), 'FGSM: every coordinate moves by exactly ε');
  const curve = R.robustCurve(c, [0, 0.05, 0.1, 0.2, 0.3]);
  t.ok(curve.every((r, i) => i === 0 || r.accuracy <= curve[i - 1].accuracy) && curve[4].accuracy < curve[0].accuracy - 0.3, 'accuracy falls as ε grows: ' + curve.map((r) => r.accuracy));
  const a = R.classifier({ data: 'moons', seed: 1, adv: { eps: 0.15 } }); a.run(800);
  const ca = R.robustCurve(a, [0, 0.1, 0.2]);
  t.ok(ca[0].accuracy >= 0.95 && ca[1].accuracy > curve[2].accuracy && ca[2].accuracy > curve[3].accuracy, 'adversarial training: robust accuracy ' + ca.map((r) => r.accuracy) + ' against ' + [curve[0], curve[2], curve[3]].map((r) => r.accuracy));
  return 'clean ' + acc.toFixed(3) + ', at ε 0.2: ' + curve[3].accuracy.toFixed(2) + ' plain, ' + ca[2].accuracy.toFixed(2) + ' adversarially trained';
});
timed('toy classifier is brittle (digits)', 40000, () => {
  const c = R.classifier({ data: 'digits', seed: 1 }); c.run(600);
  const acc = c.accuracy();
  t.ok(acc >= 0.9, 'digit classifier accuracy ' + acc);
  const curve = R.robustCurve(c, [0, 0.05, 0.1]);
  t.ok(curve[2].accuracy < 0.5, 'pixel changes of at most 0.1 drop accuracy to ' + curve[2].accuracy);
  const x = c.data.Xte[0], fa = R.fgsm(c, x, c.data.yte[0], 0.1);
  t.ok(fa.x.every((v) => v >= 0 && v <= 1), 'perturbed pixels stay in [0, 1]');
  const a = R.classifier({ data: 'digits', seed: 1, adv: { eps: 0.1 } }); a.run(600);
  const ca = R.robustCurve(a, [0, 0.1]);
  t.ok(ca[0].accuracy >= 0.9 && ca[1].accuracy > curve[2].accuracy + 0.3, 'adversarial training keeps ' + ca[1].accuracy + ' at ε = 0.1 (plain: ' + curve[2].accuracy + ')');
  const d = R.defences(c, 0.1, { sigma: 0.3 }), sq = d.rows.find((r) => /squeezing/.test(r.defence)), none = d.rows[0];
  t.ok(sq.attacked > none.attacked + 0.3 && sq.clean > none.clean - 0.05, 'squeezing to 1 bit: attacked accuracy ' + sq.attacked + ' against ' + none.attacked);
  t.ok(d.rows.length === 3 && d.trace.length === 3, 'defence table and trace');
  return 'clean ' + acc.toFixed(2) + '; at ε 0.1: plain ' + curve[2].accuracy.toFixed(2) + ', adversarially trained ' + ca[1].accuracy.toFixed(2) + ', squeezed ' + sq.attacked.toFixed(2);
});
{
  const c = R.classifier({ data: 'moons', seed: 1 }); c.run(400);
  const x = c.data.Xte[3], s = R.smoothPredict(c, x, { sigma: 0.2, samples: 50 });
  t.ok(s.votes.reduce((a, b) => a + b) === 50 && s.label === s.votes.indexOf(Math.max(...s.votes)), 'smoothing takes a majority vote over noisy copies');
}

// ---- distribution shift ----
timed('distribution shift', 10000, () => {
  const s = R.shift({ kind: 'label', amount: 1 }), share = s.test.y.filter((v) => v).length / s.test.y.length;
  t.ok(Math.abs(share - 0.95) < 0.01 && s.train.y.filter((v) => v).length === 200, 'label shift: class 1 share ' + share + ' in test, exactly half in training');
  const c0 = R.shift({ kind: 'covariate', amount: 1 }), mean = (X) => X.reduce((a, x) => a + x[0], 0) / X.length;
  t.ok(Math.abs(mean(c0.train.X) + 1.3) < 0.1 && Math.abs(mean(c0.test.X) - 1.3) < 0.1, 'covariate shift moves the inputs');
  const cov = R.shiftCurve({ kind: 'covariate' }), con = R.shiftCurve({ kind: 'concept' });
  t.ok(cov.rows[0].accuracy > 0.9 && cov.drop > 0.15, 'covariate shift: accuracy ' + cov.rows.map((r) => r.accuracy.toFixed(2)));
  t.ok(con.rows[0].accuracy > 0.9 && con.drop > 0.3, 'concept shift: accuracy ' + con.rows.map((r) => r.accuracy.toFixed(2)));
  t.ok(cov.trace.length === 6, 'shift trace');
  return 'drops: covariate ' + cov.drop.toFixed(2) + ', concept ' + con.drop.toFixed(2);
});

// ---- differential privacy ----
{
  const e = (sigma, steps, q = 0.04) => R.dpEpsilon({ q, sigma, steps, delta: 1e-5 }).eps;
  t.ok(e(0, 100) === Infinity, 'no noise, no guarantee');
  t.ok(e(1, 100) > e(2, 100) && e(2, 100) > e(4, 100), 'ε falls as noise grows');
  t.ok(e(1, 400) > e(1, 100), 'ε grows with the number of steps');
  t.ok(e(1, 100, 0.1) > e(1, 100, 0.01), 'ε grows with the sampling rate');
  t.ok(R.dpEpsilon({ q: 0.05, sigma: 1, steps: 100 }).approximate === true && /approximate|not a certified/.test(R.dpEpsilon({ q: 0.05, sigma: 1, steps: 100 }).say), 'the accountant says it is approximate');
}
timed('DP-SGD trade-off', 5000, () => {
  const d = R.dpsgd({ sigma: 1, seed: 1 });
  t.ok(d.trace.length === 5 && /clipped|scaled down/.test(d.trace[1].say), 'DP-SGD step trace: gradients, clipping, noise, update, result');
  const norms = d.trace[0].norms, clipped = d.trace[1].clipped;
  t.ok(clipped === norms.filter((v) => v > 1).length, 'the clip count matches the gradient lengths');
  const tr = R.dpTradeoff();
  const rows = tr.rows;
  t.ok(rows.every((r, i) => i === 0 || r.eps < rows[i - 1].eps), 'ε falls along the curve: ' + rows.map((r) => r.eps.toFixed(2)));
  t.ok(rows[0].accuracy > 0.75 && rows[rows.length - 1].accuracy < rows[0].accuracy - 0.1, 'accuracy pays for privacy: ' + rows.map((r) => r.accuracy.toFixed(3)));
  return rows.map((r) => 'σ ' + r.sigma + ': ε ' + (isFinite(r.eps) ? r.eps.toFixed(2) : '∞') + ', acc ' + r.accuracy.toFixed(2)).join('; ');
});

// ---- fairness ----
{
  const y = [1, 1, 0, 0, 1, 0, 0, 0], g = [0, 0, 0, 0, 1, 1, 1, 1], { weights, table } = R.reweigh(y, g);
  // after reweighing, the weighted share of label 1 is the same in both groups
  const share = (grp) => { let a = 0, b = 0; y.forEach((v, i) => { if (g[i] === grp) { b += weights[i]; if (v) a += weights[i]; } }); return a / b; };
  t.ok(Math.abs(share(0) - share(1)) < 1e-12 && Math.abs(share(0) - 3 / 8) < 1e-12, 'reweighing makes the label independent of the group');
  t.ok(table.length === 4 && Math.abs(table.find((r) => r.group === 0 && r.label === 1).weight - 0.75) < 1e-12, 'weight P(g)P(y)/P(g, y) = 0.5·0.375/0.25');
  const rows = R.fairCompare({ target: 'parity' }), [none, rew, thr] = rows;
  t.ok(none.parity > 0.05, 'the plain model approves the groups at different rates (gap ' + none.parity + ')');
  t.ok(rew.parity < none.parity && thr.parity < none.parity, 'both mitigations narrow the parity gap: ' + rows.map((r) => r.parity.toFixed(3)));
  t.ok(rows.every((r) => r.accuracy > 0.7), 'accuracy stays reasonable: ' + rows.map((r) => r.accuracy.toFixed(3)));
  const opp = R.fairCompare({ target: 'opportunity' });
  t.ok(opp[2].result.metrics.tprDifference < opp[0].result.metrics.tprDifference, 'per-group thresholds for equal opportunity narrow the TPR gap');
  const m = S.Metrics.fairness([1, 0, 1, 1], [1, 0, 0, 1], ['A', 'A', 'B', 'B']);
  const r = R.fairness({ mitigation: 'none' });
  t.ok(r.trace.some((s) => /Demographic parity/.test(s.say)) && r.metrics.groups.length === 2 && m.groups.length === 2, 'fairness uses S.Metrics.fairness and its trace');
}

// ---- determinism ----
{
  const h = (seed) => { const c = R.classifier({ data: 'moons', seed, adv: { eps: 0.1 } }); c.run(30); return c.history.map((x) => x.loss); };
  t.eq(h(2), h(2), 'adversarial training is deterministic');
  t.ok(JSON.stringify(h(2)) !== JSON.stringify(h(3)), 'another seed gives another run');
  t.eq(R.dpsgd({ seed: 4 }).w, R.dpsgd({ seed: 4 }).w, 'DP-SGD is deterministic for a seed');
  t.eq(R.shift({ kind: 'concept', seed: 3 }).test, R.shift({ kind: 'concept', seed: 3 }).test, 'the shift generator is deterministic');
}
t.done();
