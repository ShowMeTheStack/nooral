// Metrics and explanations against scikit-learn reference outputs (tests/ref/metrics/*.json, made by
// tests/ref/gen_metrics.py), hand-worked cases, and exact Shapley values against brute force.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, harness, rng } from './load.mjs';

const S = load(['core']), M = S.Metrics, E = S.Explain;
const t = harness('metrics');
const here = dirname(fileURLToPath(import.meta.url));
const ref = (n) => JSON.parse(readFileSync(join(here, 'ref', 'metrics', n + '.json'), 'utf8'));
const num = (v) => (v === 'nan' ? NaN : v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v);
const close = (a, b, tol = 1e-9) => {
  if (Array.isArray(b)) return Array.isArray(a) && a.length === b.length && b.every((v, i) => close(a[i], v, tol));
  a = num(a); b = num(b);
  if (Number.isNaN(b)) return Number.isNaN(a);
  if (!Number.isFinite(b)) return a === b;
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
};
const near = (a, b, msg, tol) => t.ok(close(a, b, tol), msg + ': got ' + JSON.stringify(a) + ', want ' + JSON.stringify(b));

// classification
ref('classification').forEach((c, i) => {
  const cm = M.confusion(c.yTrue, c.yPred);
  t.eq(cm.labels, c.labels, 'labels ' + i);
  t.eq(cm.counts, c.counts, 'confusion ' + i);
  if (c.normTrue) {
    near(M.confusion(c.yTrue, c.yPred, { normalize: 'true' }).matrix, c.normTrue, 'norm true ' + i);
    near(M.confusion(c.yTrue, c.yPred, { normalize: 'pred' }).matrix, c.normPred, 'norm pred ' + i);
    near(M.confusion(c.yTrue, c.yPred, { normalize: 'all' }).matrix, c.normAll, 'norm all ' + i);
  }
  t.ok(cm.cells.every((r, a) => r.every((cell, b) => cell.length === cm.counts[a][b] && cell.every((k) => c.yTrue[k] === cm.labels[a] && c.yPred[k] === cm.labels[b]))), 'cells ' + i);
  near(M.accuracy(c.yTrue, c.yPred).value, c.accuracy, 'accuracy ' + i);
  near(M.balancedAccuracy(c.yTrue, c.yPred).value, c.balanced, 'balanced ' + i);
  near(M.mcc(c.yTrue, c.yPred).value, c.mcc, 'mcc ' + i);
  near(M.kappa(c.yTrue, c.yPred).value, c.kappa, 'kappa ' + i);
  for (const avg of ['macro', 'micro', 'weighted']) {
    const r = M.precisionRecallF1(c.yTrue, c.yPred, { average: avg });
    near([r.precision, r.recall, r.f1], c[avg], avg + ' ' + i);
  }
  if (c.perClass) {
    const r = M.precisionRecallF1(c.yTrue, c.yPred, { average: null });
    near([r.precision, r.recall, r.f1, r.support], c.perClass, 'per class ' + i);
  }
  if (c.binary) {
    const r = M.precisionRecallF1(c.yTrue, c.yPred, { average: 'binary', posLabel: 1 });
    near([r.precision, r.recall, r.f1], c.binary, 'binary ' + i);
    near(M.specificity(c.yTrue, c.yPred, { posLabel: 1 }).value, c.specificity, 'specificity ' + i);
  }
  t.ok(M.mcc(c.yTrue, c.yPred).trace.every((s) => s.kind && typeof s.say === 'string'), 'trace ' + i);
});

// curves
ref('curves').forEach((c, i) => {
  const r = M.roc(c.y, c.s);
  near(r.fpr, c.roc.fpr, 'roc fpr ' + i); near(r.tpr, c.roc.tpr, 'roc tpr ' + i); near(r.thresholds, c.roc.thresholds, 'roc thresholds ' + i);
  const r2 = M.roc(c.y, c.s, { dropIntermediate: false });
  near(r2.fpr, c.rocAll.fpr, 'roc all fpr ' + i); near(r2.thresholds, c.rocAll.thresholds, 'roc all th ' + i);
  near(r.auc, c.auc, 'auc ' + i);
  const p = M.prCurve(c.y, c.s);
  near(p.precision, c.pr.precision, 'pr precision ' + i); near(p.recall, c.pr.recall, 'pr recall ' + i); near(p.thresholds, c.pr.thresholds, 'pr thresholds ' + i);
  near(p.ap, c.ap, 'ap ' + i);
  if (c.logloss !== undefined) near(M.logLoss(c.y, c.s).value, c.logloss, 'logloss ' + i);
  near(M.brier(c.y, c.s, { posLabel: 1 }).value, c.brier, 'brier ' + i);
  const u = M.calibration(c.y, c.s, { nBins: c.nBins }), q = M.calibration(c.y, c.s, { nBins: c.nBins, strategy: 'quantile' });
  near([u.probTrue, u.probPred], c.calUniform, 'calibration uniform ' + i);
  near([q.probTrue, q.probPred], c.calQuantile, 'calibration quantile ' + i);
});
ref('multiclass').forEach((c, i) => {
  near(M.logLoss(c.y, c.P).value, c.logloss, 'multi logloss ' + i);
  near(M.brier(c.y, c.P).value, c.brier, 'multi brier ' + i);
  near(M.rocAucOvr(c.y, c.P).value, c.aucMacro, 'ovr macro ' + i);
  near(M.rocAucOvr(c.y, c.P, { average: 'weighted' }).value, c.aucWeighted, 'ovr weighted ' + i);
  near([1, 2, 3].map((k) => M.topK(c.y, c.P, k).value), c.topk, 'top-k ' + i);
});

// regression
ref('regression').forEach((c, i) => {
  near(M.mae(c.y, c.yh).value, c.mae, 'mae ' + i); near(M.mse(c.y, c.yh).value, c.mse, 'mse ' + i);
  near(M.rmse(c.y, c.yh).value, c.rmse, 'rmse ' + i); near(M.mape(c.y, c.yh).value, c.mape, 'mape ' + i, 1e-7);
  if (c.r2 !== undefined) { near(M.r2(c.y, c.yh).value, c.r2, 'r2 ' + i); near(M.explainedVariance(c.y, c.yh).value, c.ev, 'ev ' + i); }
});

// clustering
ref('clustering').forEach((c, i) => {
  const s = M.silhouette(c.X, c.labels);
  near(s.value, c.silhouette, 'silhouette ' + i); near(s.perExample, c.silhouetteSamples, 'silhouette samples ' + i);
  near(M.daviesBouldin(c.X, c.labels).value, c.db, 'davies-bouldin ' + i);
  near(M.adjustedRand(c.labels, c.other).value, c.ari, 'ari ' + i);
  near(M.nmi(c.labels, c.other).value, c.nmi, 'nmi ' + i);
});

// fairness
ref('fairness').forEach((c, i) => {
  const r = M.fairness(c.y, c.yp, c.g);
  near([r.demographicParity, r.tprDifference, r.fprDifference], [c.dp, c.dt, c.df], 'fairness ' + i);
  t.ok(r.groups.every((g) => M.sum(g.confusion.counts.flat()) === g.n), 'group confusion ' + i);
});

// hand-worked cases and edge cases
{
  const r = M.precisionRecallF1([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]);
  near([r.precision, r.recall, r.f1], [2 / 3, 2 / 3, 2 / 3], 'binary by hand');
  t.ok(/TP = 2, FP = 1, FN = 1/.test(r.trace[0].say), 'trace has counts: ' + r.trace[0].say);
  const e = M.precisionRecallF1([0, 0, 0], [0, 0, 0]);
  near([e.precision, e.recall, e.f1], [0, 0, 0], 'no positives at all');
  t.ok(Number.isNaN(M.roc([1, 1, 1], [0.2, 0.5, 0.9]).auc), 'auc undefined with one class');
  t.ok(Number.isNaN(M.kappa([2, 2], [2, 2]).value), 'kappa undefined with one label');
  near(M.mcc([1, 1, 1], [1, 1, 1]).value, 0, 'mcc one class');
  near(M.roc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]).auc, 0.5, 'all tied auc');
  near(M.roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).auc, 0.75, 'auc by hand');
  const th = M.atThreshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.35);
  t.eq([th.tp, th.fp, th.fn, th.tn], [2, 1, 0, 1], 'threshold counts');
  near(M.perplexity([0.25, 0.25, 0.25, 0.25]).value, 4, 'perplexity uniform');
  near(M.perplexity([0.5, 0.125]).value, 4, 'perplexity mixed');
  const ng = M.ngramPrecision('the the the the', 'the cat is on the mat', 1);
  t.eq([ng.match, ng.total], [2, 4], 'clipped unigram precision');
  near(M.bleuLike('the cat is on the mat', 'the cat is on the mat').value, 1, 'bleu identical');
  t.ok(M.bleuLike('the cat', 'the cat is on the mat', 2).brevity < 0.2, 'bleu brevity penalty');
  near(M.topK([0, 1, 0], [0.2, 0.7, 0.6], 1).value, 2 / 3, 'binary top-1');
  near(M.topK([2], [[0.5, 0.5, 0.0]], 2, { labels: [0, 1, 2] }).value, 0, 'top-2 with ties');
  near(M.topK([0], [[0.5, 0.5, 0.0]], 1, { labels: [0, 1, 2] }).value, 0, 'ties: later class ranks first');
}

// explanations
if (E) {
  const r = rng(7);
  const shapCase = (M_, nb) => {
    const w = Array.from({ length: M_ }, () => r() * 2 - 1), pair = r() * 2;
    const model = (X) => X.map((x) => w.reduce((s, wi, j) => s + wi * x[j], 0) + pair * x[0] * x[1 % M_] + Math.sin(x[M_ - 1]));
    const bg = Array.from({ length: nb }, () => Array.from({ length: M_ }, () => r() * 2 - 1));
    const x = Array.from({ length: M_ }, () => r() * 2 - 1);
    return { model, bg, x };
  };
  // brute force: the Shapley formula over every ordering's coalition weights
  const brute = (model, bg, x) => {
    const Mf = x.length, v = (mask) => M.mean(model(bg.map((b) => b.map((bv, j) => ((mask >> j) & 1 ? x[j] : bv)))));
    const fact = (k) => (k <= 1 ? 1 : k * fact(k - 1));
    return x.map((_, i) => {
      let phi = 0;
      for (let m = 0; m < 1 << Mf; m++) {
        if ((m >> i) & 1) continue;
        const s = popc(m);
        phi += (fact(s) * fact(Mf - s - 1)) / fact(Mf) * (v(m | (1 << i)) - v(m));
      }
      return phi;
    });
  };
  const popc = (m) => { let c = 0; while (m) { c += m & 1; m >>= 1; } return c; };
  for (const Mf of [1, 2, 3, 5, 7]) {
    const { model, bg, x } = shapCase(Mf, 6);
    const ex = E.shapExact(model, x, bg), bf = brute(model, bg, x);
    near(ex.phi, bf, 'exact shapley ' + Mf, 1e-9);
    near(M.sum(ex.phi) + ex.base, model([x])[0], 'efficiency ' + Mf, 1e-9);
    const ks = E.kernelShap(model, x, bg, { nSamples: 4000, seed: 3 });
    near(ks.phi, bf, 'kernel shap (all coalitions) ' + Mf, 1e-6);
  }
  {
    const { model, bg, x } = shapCase(10, 4);
    const ex = E.shapExact(model, x, bg), ks = E.kernelShap(model, x, bg, { nSamples: 600, seed: 5 });
    const err = Math.max(...ex.phi.map((p, i) => Math.abs(p - ks.phi[i]))), scale = Math.max(...ex.phi.map(Math.abs));
    t.ok(err < 0.05 * scale + 1e-3, 'sampled kernel shap near exact (10 features): max error ' + err.toFixed(4) + ' of ' + scale.toFixed(3));
    near(M.sum(ks.phi) + ks.base, model([x])[0], 'kernel shap efficiency', 1e-9);
    t.ok(E.shapExact.length >= 0 && (() => { try { E.shapExact(model, [...x, 0], bg.map((b) => [...b, 0])); return false; } catch (e) { return true; } })(), 'exact refuses more than 10 features');
  }
  // tree SHAP (path-dependent) against brute force over the tree's cover-weighted expectation
  const ex = ref('explain');
  const tree = E.treeFromNodes(ex.nodes);
  near(ex.X.map((x) => E.treePredict(tree, x)), ex.pred, 'tree predictions');
  const condExp = (x, mask) => {
    const go = (k) => { const nd = ex.nodes[k]; if (nd.left < 0) return nd.value; if ((mask >> nd.feature) & 1) return go(x[nd.feature] <= nd.threshold ? nd.left : nd.right); const L = ex.nodes[nd.left], R = ex.nodes[nd.right]; return (L.cover * go(nd.left) + R.cover * go(nd.right)) / nd.cover; };
    return go(0);
  };
  for (const k of [0, 5, 17, 60]) {
    const x = ex.X[k], ts = E.treeShap(tree, x);
    const fact = (q) => (q <= 1 ? 1 : q * fact(q - 1)), F = 3;
    const bf = x.map((_, i) => { let p = 0; for (let m = 0; m < 1 << F; m++) { if ((m >> i) & 1) continue; const s = popc(m); p += (fact(s) * fact(F - s - 1)) / fact(F) * (condExp(x, m | (1 << i)) - condExp(x, m)); } return p; });
    near(ts.phi, bf, 'tree shap ' + k, 1e-9);
    near(M.sum(ts.phi) + ts.base, ex.pred[k], 'tree shap efficiency ' + k, 1e-9);
  }
  // Tree SHAP on a tree grown by the classical engine (regression and class probability)
  if (S.ML && S.ML.growTree) {
    for (const cls of [false, true]) {
      const yv = ex.X.map((x) => (cls ? (x[0] + x[2] > 1 ? 1 : 0) : x[0] * x[1] + x[2]));
      const grown = S.ML.growTree(ex.X, yv, cls ? { criterion: 'gini', maxDepth: 4 } : { criterion: 'squared_error', maxDepth: 4 });
      const tr = E.treeFromNodes(grown, { output: 1 });
      const x = ex.X[11], ts = E.treeShap(tr, x);
      near(M.sum(ts.phi) + ts.base, E.treePredict(tr, x), 'tree shap on a grown ' + (cls ? 'classifier' : 'regressor'), 1e-9);
    }
  }
  // partial dependence and ICE against scikit-learn's brute method
  const f = (X) => X.map((x) => E.treePredict(tree, x));
  for (const p of ex.pd) {
    const pd = E.partialDependence(f, ex.X, p.feature, { gridSize: 11 });
    near(pd.grid, p.grid, 'pd grid ' + p.feature, 1e-9);
    near(pd.average, p.average, 'pd average ' + p.feature, 1e-9);
    near(pd.ice.slice(0, 5), p.ice, 'ice ' + p.feature, 1e-9);
  }
  // permutation importance: an unused feature scores 0 and a used one scores high
  {
    const X = Array.from({ length: 200 }, () => [r() * 2 - 1, r() * 2 - 1, r() * 2 - 1]), y = X.map((x) => (x[0] + 0.2 * x[1] > 0 ? 1 : 0));
    const model = (Z) => Z.map((x) => (x[0] + 0.2 * x[1] > 0 ? 1 : 0));
    const pi = E.permutationImportance(model, X, y, { nRepeats: 5, seed: 2 });
    t.ok(pi.mean[2] === 0 && pi.mean[0] > 0.3 && pi.mean[0] > pi.mean[1] && pi.mean[1] > 0, 'permutation importance order: ' + pi.mean.map((v) => v.toFixed(3)));
    const pi2 = E.permutationImportance(model, X, y, { nRepeats: 5, seed: 2 });
    t.eq(pi2.mean, pi.mean, 'permutation importance deterministic');
  }
  // LIME recovers the slopes of a linear model exactly
  {
    const X = Array.from({ length: 100 }, () => [r() * 4, r() - 0.5, r() * 10]);
    const model = (Z) => Z.map((x) => 2 * x[0] - 3 * x[1] + 0.5 * x[2] + 1);
    const lm = E.lime(model, X[3], X, { nSamples: 300, seed: 4, ridge: 0 });
    near(lm.coef, [2, -3, 0.5], 'lime slopes', 1e-6);
    near(lm.score, 1, 'lime fit', 1e-9);
  }
  // Grad-CAM from given activations and gradients; saliency by finite differences
  {
    const A = [[[1, 0], [0, 2]], [[0, 1], [1, 0]]], G = [[[0.5, 0.5], [0.5, 0.5]], [[-1, -1], [-1, -1]]];
    const cam = E.gradCam(A, G);
    near(cam.weights, [0.5, -1], 'grad-cam weights');
    near(cam.raw, [[0.5, 0], [0, 1]], 'grad-cam map');
    const sal = E.saliency((x) => 3 * x[0] * x[0] - x[1], [1, 2]);
    near(sal.grad, [6, -1], 'saliency by finite differences', 1e-5);
    if (S.T && S.T.mul) {
      const T = S.T;
      near(E.saliencyT((z) => T.mul(z, z), [1, -2, 3], [3], 1).grad, [0, -4, 0], 'saliency through S.T');
      const g = E.gradCamT((inp, keep) => { const a = keep(T.mul(inp, T.tensor([2], [1]))); return T.reshape(T.sum(T.reshape(a, [2, 4]), 1), [2]); }, [1, 0, 0, 2, 0, 1, 1, 0], [1, 2, 2, 2], 0);
      near(g.weights, [1, 0], 'grad-cam through S.T: weights');
      near(g.map, [[0.5, 0], [0, 1]], 'grad-cam through S.T: map');
    }
  }
}

// every trace sentence is filled in with real values
{
  const yt = [0, 1, 1, 0, 1, 2, 2, 0], yp = [0, 1, 0, 0, 2, 2, 1, 0], s = [0.1, 0.9, 0.4, 0.3, 0.8, 0.7, 0.6, 0.2], yb = yt.map((v) => (v ? 1 : 0));
  const X = [[0, 0], [0, 1], [5, 5], [5, 6], [9, 0], [9, 1], [0, 2], [5, 4]];
  const runs = [M.confusion(yt, yp, { normalize: 'true' }), M.accuracy(yt, yp), M.balancedAccuracy(yt, yp), M.precisionRecallF1(yt, yp, { average: 'macro' }), M.precisionRecallF1(yt, yp, { average: 'weighted' }),
    M.precisionRecallF1(yt, yp, { average: 'micro' }), M.specificity(yt, yp, { average: 'macro' }), M.mcc(yb, yp.map((v) => (v ? 1 : 0))), M.kappa(yt, yp), M.roc(yb, s), M.prCurve(yb, s), M.logLoss(yb, s), M.brier(yb, s),
    M.calibration(yb, s, { nBins: 3 }), M.calibration(yb, s, { nBins: 3, strategy: 'quantile' }), M.mae(s, yb), M.rmse(s, yb), M.r2(s, yb), M.explainedVariance(s, yb), M.mape(s, yb), M.residuals(s, yb),
    M.silhouette(X, yt), M.daviesBouldin(X, yt), M.adjustedRand(yt, yp), M.nmi(yt, yp), M.topK(yt, yt.map(() => [0.2, 0.5, 0.3]), 2), M.perplexity(s), M.bleuLike('a cat sat on the mat', 'the cat sat on the mat', 2),
    M.fairness(yb, yp.map((v) => (v ? 1 : 0)), ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b']), M.rocAucOvr(yt, yt.map((_, i) => [s[i] / 2, s[i] / 2, 1 - s[i]]))];
  if (E) {
    const model = (Z) => Z.map((z) => z[0] * 2 + z[1]);
    runs.push(E.shapExact(model, [1, 2], X), E.kernelShap(model, [1, 2], X), E.permutationImportance(model, X, X.map((z) => z[0] * 2 + z[1] + 0.5)), E.partialDependence(model, X, 0), E.lime(model, X[2], X), E.gradCam([[[1]]], [[[1]]]));
  }
  runs.forEach((r, i) => t.ok(r.trace.length > 0 && r.trace.every((st) => st.kind && typeof st.say === 'string' && st.say.length > 10 && !/NaN|undefined|null|\[object/.test(st.say)), 'trace ' + i + ': ' + JSON.stringify(r.trace.map((st) => st.say))));
}
t.done();
