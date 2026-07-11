// Classical algorithms against scikit-learn references (tests/ref/classic/*.json from gen_classic.py),
// plus property tests. Deterministic algorithms are compared value by value; stochastic ones either share
// the reference's randomness (initial centres, bootstrap draws) or are compared by behaviour.
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, harness } from './load.mjs';

const S = load(['core']), ML = S.ML, t = harness('classic');
const dir = join(dirname(fileURLToPath(import.meta.url)), 'ref', 'classic');
const ref = (n) => JSON.parse(readFileSync(join(dir, n + '.json'), 'utf8'));
const D = ref('data');
const XY = (k) => [D[k].X, D[k].y];

// ---- comparison helpers ----
const flat = (a) => (Array.isArray(a) ? a.flatMap(flat) : [a]);
function close(got, want, tol, msg) {
  const g = flat(got), w = flat(want);
  if (g.length !== w.length) return t.ok(false, msg + ': length ' + g.length + ' vs ' + w.length);
  let worst = 0, at = -1;
  for (let i = 0; i < g.length; i++) {
    const e = Math.abs(g[i] - w[i]) / Math.max(1, Math.abs(w[i]));
    if (!(e <= worst)) { worst = e; at = i; }
  }
  t.ok(worst <= tol, msg + ': max rel error ' + worst.toExponential(2) + ' at ' + at + ' (got ' + g[at] + ', want ' + w[at] + ')');
}
const same = (got, want, msg) => t.eq(flat(got), flat(want), msg);
// labels equal up to a renaming of the clusters (noise -1 must stay -1)
function sameUpToRelabel(a, b, msg) {
  const m = new Map(), r = new Map();
  let ok = a.length === b.length;
  for (let i = 0; ok && i < a.length; i++) {
    if ((a[i] === -1) !== (b[i] === -1)) ok = false;
    else if (m.has(a[i]) ? m.get(a[i]) !== b[i] : r.has(b[i])) ok = false;
    else { m.set(a[i], b[i]); r.set(b[i], a[i]); }
  }
  t.ok(ok, msg);
}
const time = (f) => { const t0 = performance.now(); const v = f(); return [v, performance.now() - t0]; };

// ================= linear models =================
{
  const L = ref('linear'), [X, y] = XY('reg');
  let m = new ML.LinearRegression().fit(X, y);
  close(m.coef, L.ols.coef, 1e-10, 'OLS coef (QR)'); close(m.intercept, L.ols.intercept, 1e-10, 'OLS intercept');
  t.ok(m.trace.some((s) => s.kind === 'qr') && m.trace.every((s) => typeof s.say === 'string'), 'OLS trace has the QR step');
  const g = new ML.LinearRegression({ solver: 'gd', lr: 0.1, epochs: 3000, traceEvery: 100 }).fit(X, y);
  close(g.coef, L.ols.coef, 1e-6, 'gradient descent reaches the closed form');
  t.ok(g.history.every((v, i) => i === 0 || v <= g.history[i - 1] + 1e-12), 'gradient descent loss never rises');
  m = new ML.Ridge({ alpha: L.ridge.alpha }).fit(X, y);
  close(m.coef, L.ridge.coef, 1e-10, 'ridge coef'); close(m.intercept, L.ridge.intercept, 1e-10, 'ridge intercept');
  m = new ML.Lasso({ alpha: L.lasso.alpha }).fit(X, y);
  close(m.coef, L.lasso.coef, 1e-8, 'lasso coef'); close(m.intercept, L.lasso.intercept, 1e-8, 'lasso intercept');
  same(m.coef.map((v) => v === 0), L.lasso.coef.map((v) => v === 0), 'lasso zero pattern');
  m = new ML.ElasticNet({ alpha: L.enet.alpha, l1Ratio: L.enet.l1Ratio }).fit(X, y);
  close(m.coef, L.enet.coef, 1e-8, 'elastic net coef');
  const P = ML.enetPath(X, y, { nAlphas: 30 });
  close(P.alphas, L.path.alphas, 1e-10, 'lasso path alphas'); close(P.coefs, L.path.coefs, 1e-7, 'lasso path coefs');
  const pf = ML.PolynomialFeatures({ degree: 3 }).fit(L.poly.X);
  close(pf.transform(L.poly.X), L.poly.out, 1e-12, 'polynomial features');
  t.eq(pf.names.length, L.poly.names.length, 'polynomial feature count');
  const [Xw, yw] = XY('wave'), pw = ML.PolynomialFeatures({ degree: 5, includeBias: false }).fitTransform(Xw);
  m = new ML.LinearRegression().fit(pw, yw);
  close(m.coef, L.polyreg.coef, 1e-7, 'degree-5 polynomial regression');
  // logistic
  const [X2, y2] = XY('cls2');
  m = new ML.LogisticRegression({ C: L.logit2.C }).fit(X2, y2);
  close(m.coef, L.logit2.coef, 1e-6, 'binary logistic coef'); close(m.predictProba(X2), L.logit2.proba, 1e-7, 'binary logistic proba');
  const [Xi, yi] = XY('iris');
  m = new ML.LogisticRegression({ C: 1 }).fit(Xi, yi);
  close(m.coef, L.logit3.coef, 1e-5, 'multinomial logistic coef (iris)'); close(m.intercept, L.logit3.intercept, 1e-5, 'multinomial intercepts');
  close(m.predictProba(Xi), L.logit3.proba, 1e-6, 'multinomial proba (iris)');
  const [Xs, ys] = XY('wine');
  close(new ML.LogisticRegression({ C: L.logitWine.C }).fit(Xs, ys).predictProba(Xs), L.logitWine.proba, 1e-6, 'multinomial proba (wine)');
  const gd = new ML.LogisticRegression({ C: L.logit2.C, solver: 'gd', lr: 0.5, epochs: 4000 }).fit(X2, y2);
  close(gd.coef, L.logit2.coef, 1e-3, 'logistic gradient descent approaches Newton');
  t.ok(gd.trace.length > 10 && gd.trace[0].kind === 'gd', 'logistic GD trace');
  m = new ML.Perceptron({ epochs: L.perceptron.epochs }).fit(X2, y2);
  close(m.coef, L.perceptron.coef, 1e-12, 'perceptron weights'); close(m.intercept, L.perceptron.intercept, 1e-12, 'perceptron bias');
  m = new ML.Perceptron({ epochs: L.perceptron3.epochs, eta: L.perceptron3.eta }).fit(Xi, yi);
  close(m.coef, L.perceptron3.coef, 1e-12, 'one-vs-rest perceptron weights (iris)');
}

// ================= discriminant analysis and naive Bayes =================
{
  const G = ref('generative');
  for (const key of ['cls3', 'iris']) {
    const [X, y] = XY(key), R = G[key];
    close(new ML.LDA().fit(X, y).predictProba(X), R.lda_svd, 1e-8, key + ' LDA proba (SVD reference)');
    const l = new ML.LDA().fit(X, y);
    close(l.predictProba(X), R.lda_lsqr, 1e-8, key + ' LDA proba (least-squares reference)');
    close(l.coef, R.lda_coef, 1e-8, key + ' LDA coef');
    close(l.explainedVarianceRatio, R.lda_evr, 1e-8, key + ' LDA explained variance ratio');
    close(new ML.QDA().fit(X, y).predictProba(X), R.qda, 1e-8, key + ' QDA proba');
    close(new ML.QDA({ regParam: 0.1 }).fit(X, y).predictProba(X), R.qda_reg, 1e-8, key + ' QDA proba, reg 0.1');
    const g = new ML.GaussianNB().fit(X, y);
    close(g.var, R.gnb_var, 1e-10, key + ' Gaussian NB variances'); close(g.predictProba(X), R.gnb, 1e-8, key + ' Gaussian NB proba');
  }
  const [Xc, yc] = XY('counts');
  for (const k of ['mnb_1', 'mnb_0.1']) {
    const m = new ML.MultinomialNB({ alpha: G[k].alpha }).fit(Xc, yc);
    close(m.featureLogProb, G[k].logp, 1e-12, k + ' feature log-probabilities'); close(m.predictProba(Xc), G[k].proba, 1e-10, k + ' proba');
  }
}

// ================= k nearest neighbours =================
{
  const K = ref('knn'), [X, y] = XY('cls3');
  const conv = { e5u: { k: 5 }, e7d: { k: 7, weights: 'distance' }, m4u: { k: 4, metric: 'manhattan' }, p3d: { k: 6, p: 3, weights: 'distance' }, cheb: { k: 5, metric: 'chebyshev' } };
  for (const [name, params] of Object.entries(conv)) for (const algorithm of ['brute', 'kd_tree']) {
    const m = new ML.KNeighborsClassifier(Object.assign({ algorithm, leafSize: 4 }, params)).fit(X, y), nb = m.kneighbors(K.Q);
    same(nb.map((r) => r.idx), K[name].idx, name + ' ' + algorithm + ' neighbour indices');
    close(nb.map((r) => r.dist), K[name].dist, 1e-12, name + ' ' + algorithm + ' distances');
    close(m.predictProba(K.Q), K[name].proba, 1e-12, name + ' ' + algorithm + ' proba');
  }
  const kd = new ML.KNeighborsClassifier({ algorithm: 'kd_tree', leafSize: 4 }).fit(X, y).explain(K.Q[0]);
  t.ok(kd.steps.some((s) => s.kind === 'prune'), 'k-d tree search prunes boxes');
  const [Xw, yw] = XY('wave');
  close(new ML.KNeighborsRegressor({ k: 6, weights: 'distance' }).fit(Xw, yw).predict(K.reg.Q), K.reg.pred, 1e-12, 'k-NN regression');
}

// ================= trees =================
const sameTree = (m, R, msg, tol = 1e-10) => {
  const a = m.arrays();
  same(a.childrenLeft, R.left, msg + ' left children'); same(a.childrenRight, R.right, msg + ' right children');
  same(a.feature, R.feature, msg + ' split features'); close(a.threshold, R.threshold, tol, msg + ' thresholds');
  close(a.impurity, R.impurity, tol, msg + ' impurities'); same(a.nNodeSamples, R.n, msg + ' node sizes'); close(a.value, R.value, tol, msg + ' values');
};
{
  const R = ref('trees'), [X, y] = XY('tree3');
  for (const crit of ['gini', 'entropy']) for (const depth of [3, null]) {
    const m = new ML.DecisionTreeClassifier({ criterion: crit, maxDepth: depth, float32: true, seed: R.seed0 }).fit(X, y), r = R[crit + '_' + (depth == null ? 'None' : depth)];
    sameTree(m, r.tree, crit + ' depth ' + depth);
    close(m.featureImportances, r.imp, 1e-10, crit + ' depth ' + depth + ' importances');
  }
  sameTree(new ML.DecisionTreeClassifier({ minSamplesLeaf: 5, float32: true, seed: R.seed0 }).fit(X, y), R.msl5.tree, 'min_samples_leaf 5');
  const path = new ML.DecisionTreeClassifier({ float32: true, seed: R.seed0 }).costComplexityPath(X, y);
  close(path.alphas, R.ccp_path.alphas, 1e-10, 'pruning path alphas'); close(path.impurities, R.ccp_path.impurities, 1e-10, 'pruning path impurities');
  sameTree(new ML.DecisionTreeClassifier({ ccpAlpha: R.ccp.alpha, float32: true, seed: R.seed0 }).fit(X, y), R.ccp.tree, 'pruned tree');
  const [Xr, yr] = XY('treg');
  const m = new ML.DecisionTreeRegressor({ maxDepth: 4, float32: true, seed: R.seed0 }).fit(Xr, yr);
  sameTree(m, R.reg4.tree, 'regression tree'); close(m.featureImportances, R.reg4.imp, 1e-10, 'regression importances');
  sameTree(new ML.DecisionTreeRegressor({ minSamplesLeaf: 4, float32: true, seed: R.seed0 }).fit(Xr, yr), R.reg_msl4.tree, 'regression tree, min leaf 4');
  const rp = new ML.DecisionTreeRegressor({ maxDepth: 6, float32: true, seed: R.seed0 }).costComplexityPath(Xr, yr);
  close(rp.alphas, R.reg_ccp_path.alphas, 1e-9, 'regression pruning path');
  const [Xi, yi] = XY('iris'), ti = new ML.DecisionTreeClassifier({ seed: R.seed0, float32: true }).fit(Xi, yi);
  t.ok(ti.depth === R.iris.depth && ti.nLeaves === R.iris.leaves && ti.score(Xi, yi) === R.iris.acc, 'iris tree: same depth, leaves and accuracy');
  const tr = new ML.DecisionTreeClassifier({ maxDepth: 2 }).fit(X, y).trace;
  t.ok(tr.filter((s) => s.kind === 'candidates').every((s) => s.candidates.every((c) => c.thresholds.length === c.scores.length)), 'every candidate split carries its impurity');
}

// ================= ensembles =================
{
  const E = ref('ensemble'), [X, y] = XY('tree3'), [Xr, yr] = XY('treg');
  const every5 = (A) => A.filter((_, i) => i % 5 === 0);
  // forests replay the reference's bootstrap draws and tree seeds, so they match exactly
  let rf = new ML.RandomForestClassifier({ nEstimators: 15, oobScore: true, bootstrapIndices: E.rf.boot, treeSeeds: E.rf.split, float32: true }).fit(X, y);
  close(rf.predictProba(every5(X)), E.rf.proba, 1e-12, 'random forest proba (shared bootstrap)');
  close(rf.oobScore, E.rf.oob, 1e-12, 'random forest out-of-bag score'); close(rf.oobDecision, E.rf.oob_dec, 1e-12, 'random forest OOB decision');
  close(rf.featureImportances, E.rf.imp, 1e-10, 'random forest impurity importance');
  const rr = new ML.RandomForestRegressor({ nEstimators: 10, maxDepth: 5, oobScore: true, bootstrapIndices: E.rfr.boot, treeSeeds: E.rfr.split, float32: true }).fit(Xr, yr);
  close(rr.predict(every5(Xr)), E.rfr.pred, 1e-10, 'regression forest predictions'); close(rr.oobScore, E.rfr.oob, 1e-10, 'regression forest OOB R2');
  close(rr.featureImportances, E.rfr.imp, 1e-10, 'regression forest importances');
  // with our own randomness: the same behaviour statistically
  const ours = [0, 1, 2, 3, 4].map((s) => new ML.RandomForestClassifier({ nEstimators: 60, oobScore: true, seed: s }).fit(X, y).oobScore);
  const mo = ML.mean(ours), mr = ML.mean(E.rf_stats.oob);
  t.ok(Math.abs(mo - mr) < 0.03, 'own-seed forests: mean OOB score over 5 seeds within 0.03 of the reference (' + mo.toFixed(3) + ' vs ' + mr.toFixed(3) + ')');
  t.eq(new ML.RandomForestClassifier({ nEstimators: 5, seed: 4 }).fit(X, y).predictProba(X), new ML.RandomForestClassifier({ nEstimators: 5, seed: 4 }).fit(X, y).predictProba(X), 'forest is deterministic for a seed');
  for (const [key, data] of [['ada2', 'cls2'], ['ada3', 'tree3']]) {
    const [Xa, ya] = XY(data), a = new ML.AdaBoostClassifier({ nEstimators: 12, treeSeeds: E[key].split, float32: true }).fit(Xa, ya);
    close(a.alphas, E[key].alphas, 1e-10, key + ' AdaBoost learner weights'); close(a.errors, E[key].errors, 1e-10, key + ' AdaBoost errors');
    close(a.decisionFunction(Xa), E[key].dec, 1e-10, key + ' AdaBoost decision function'); same(a.predict(Xa), E[key].pred, key + ' AdaBoost predictions');
  }
  for (const key of ['gbr', 'gbr_sub']) {
    const g = new ML.GradientBoostingRegressor({ nEstimators: 25, learningRate: 0.2, maxDepth: 3, subsample: E[key].sub, masks: E[key].masks, treeSeeds: E[key].split, float32: true }).fit(Xr, yr);
    close(g.predict(Xr), E[key].pred, 1e-9, key + ' gradient boosting predictions');
    const st = g.stagedPredict(Xr);
    close([st[0], st[9], st[24]], E[key].staged, 1e-9, key + ' staged predictions');
  }
  const [X2, y2] = XY('cls2');
  let g = new ML.GradientBoostingClassifier({ nEstimators: 20, learningRate: 0.3, maxDepth: 2, treeSeeds: E.gbc2.split, float32: true }).fit(X2, y2);
  close(g.decisionFunction(X2), E.gbc2.dec, 1e-9, 'binary gradient boosting decision'); close(g.predictProba(X2), E.gbc2.proba, 1e-9, 'binary gradient boosting proba');
  g = new ML.GradientBoostingClassifier({ nEstimators: 15, learningRate: 0.2, maxDepth: 2, subsample: 0.8, masks: E.gbc3.masks, treeSeeds: E.gbc3.split, float32: true }).fit(X, y);
  close(g.decisionFunction(X), E.gbc3.dec, 1e-9, 'multiclass gradient boosting decision (subsample 0.8)'); close(g.predictProba(X), E.gbc3.proba, 1e-9, 'multiclass gradient boosting proba');
  t.ok(g.trainLoss.every((v, i) => i === 0 || v < g.trainLoss[i - 1] + 0.05) && g.trace.length === 15, 'gradient boosting stage trace');
  // isolation forest: random by nature, compared by behaviour
  const iso = new ML.IsolationForest({ nEstimators: 200, seed: 1 }).fit(E.iso.X), sc = iso.scoreSamples(E.iso.X);
  const corr = (a, b) => { const ma = ML.mean(a), mb = ML.mean(b); let s = 0, sa = 0, sb = 0; a.forEach((v, i) => { s += (v - ma) * (b[i] - mb); sa += (v - ma) ** 2; sb += (b[i] - mb) ** 2; }); return s / Math.sqrt(sa * sb); };
  const r = corr(sc, E.iso.score);
  t.ok(r > 0.9, 'isolation forest scores correlate with the reference (r = ' + r.toFixed(3) + ')');
  const worst = (s) => s.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).slice(0, 4).map((p) => p[1]).sort((a, b) => a - b);
  same(worst(sc), worst(E.iso.score), 'isolation forest: the four planted outliers score lowest in both');
  t.ok(Math.abs(ML.isoC(256) - 10.244770920116851) < 1e-9, 'average path length c(256)');
}

// ================= support vector machines =================
{
  const V = ref('svm'), conv = (p) => Object.assign({}, p, { tol: 1e-8 });
  for (const key of ['lin', 'poly', 'rbf', 'rbf_scale', 'sig', 'iris_rbf', 'iris_lin']) {
    const R = V[key], [X, y] = XY(R.data), m = new ML.SVC(conv(R.params)).fit(X, y);
    close(m.gammaValue, R.gamma, 1e-12, key + ' gamma');
    if (R.dec) close(m.decisionFunction(X), R.dec, 1e-5, key + ' SVC decision values');
    else { close(m.decisionFunction(X, 'ovo'), R.ovo, 1e-5, key + ' one-vs-one decision values'); close(m.decisionFunction(X), R.ovr_shape, 1e-5, key + ' one-vs-rest shaped decision'); }
    same(m.support, R.support, key + ' support vector indices'); same(m.nSupport, R.n_support, key + ' support vectors per class');
    same(m.predict(X), R.pred, key + ' SVC predictions');
  }
  const [Xi, yi] = XY('iris'), ovr = new ML.SVC({ kernel: 'rbf', gamma: 0.3, C: 2, tol: 1e-8, multiClass: 'ovr' }).fit(Xi, yi);
  close(ovr.decisionFunction(Xi), V.iris_ovr.dec, 1e-5, 'one-vs-rest trained SVC decision values'); same(ovr.predict(Xi), V.iris_ovr.pred, 'one-vs-rest predictions');
  const [Xw, yw] = XY('wave');
  for (const key of ['svr_rbf', 'svr_lin']) {
    const m = new ML.SVR(conv(V[key].params)).fit(Xw, yw);
    close(m.predict(Xw), V[key].pred, 1e-5, key + ' SVR predictions'); same(m.support, V[key].support, key + ' SVR support vectors');
    close(m.coef, V[key].dual, 1e-4, key + ' SVR dual coefficients'); close(m.intercept, V[key].intercept, 1e-5, key + ' SVR intercept');
  }
  const [Xb] = XY('blobs'), oc = new ML.OneClassSVM({ kernel: 'rbf', gamma: 0.5, nu: 0.1, tol: 1e-8 }).fit(Xb);
  close(oc.decisionFunction(Xb), V.ocsvm.dec, 1e-5, 'one-class SVM decision values'); same(oc.support, V.ocsvm.support, 'one-class SVM support vectors');
  // points on the boundary have decision ~0 in both, so compare labels only where the reference is clear
  const clear = V.ocsvm.dec.map((v, i) => (Math.abs(v) > 1e-6 ? i : -1)).filter((i) => i >= 0), ocp = oc.predict(Xb);
  same(clear.map((i) => ocp[i]), clear.map((i) => V.ocsvm.pred[i]), 'one-class SVM inliers and outliers (off the boundary)');
  const lm = new ML.SVC({ kernel: 'linear', C: 0.5 }).fit(...XY('cls2'));
  t.ok(lm.trace.filter((s) => s.kind === 'pair').length > 3 && lm.trace.every((s) => s.say), 'SMO traces pair by pair');
  // property: KKT holds at the solution; margins of free support vectors are 1
  const m0 = lm.machines[0], [X2, y2] = XY('cls2'), dv = lm.machineValues(X2).map((v) => v[0]);
  const free = m0.sv.filter((i, t2) => Math.abs(m0.coef[t2]) < 0.5 - 1e-6);
  t.ok(free.every((i) => Math.abs(Math.abs(dv[i]) - 1) < 1e-2), 'free support vectors sit on the margin');
}

// ================= unsupervised =================
{
  const U = ref('unsupervised'), [Xb, yb] = XY('blobs'), [Xi, yi] = XY('iris');
  for (const [key, X] of [['km_blobs', Xb], ['km_iris', Xi], ['km_iris_bad', Xi]]) {
    const R = U[key], m = new ML.KMeans({ k: 3, init: R.init }).fit(X);
    same(m.labels, R.labels, key + ' k-means labels (shared initial centres)'); close(m.inertia, R.inertia, 1e-10, key + ' inertia');
    close(m.centers, R.centers, 1e-10, key + ' centres'); t.eq(m.nIter, R.n_iter, key + ' iterations');
    const sil = ML.silhouette(X, m.labels);
    close(sil.mean, R.sil, 1e-10, key + ' silhouette'); close(sil.samples, R.sil_samples, 1e-10, key + ' silhouette per sample');
  }
  const kmpp = [0, 1, 2, 3, 4].map((s) => new ML.KMeans({ k: 3, seed: s, nInit: 3 }).fit(Xi).inertia);
  t.ok(Math.min(...kmpp) <= U.km_best.inertia * (1 + 1e-9) && ML.mean(kmpp) < U.km_best.inertia * 1.05, 'k-means++ restarts reach the reference best inertia');
  const km = new ML.KMeans({ k: 3, seed: 2 }).fit(Xb);
  t.ok(km.trace.filter((s) => s.kind === 'seed').length === 3 && km.trace.some((s) => s.kind === 'assign'), 'k-means trace: seeding and Lloyd steps');
  const [Xm] = XY('moons');
  for (const [key, X] of [['db_moons', Xm], ['db_moons2', Xm], ['db_blobs', Xb]]) {
    const R = U[key], m = new ML.DBSCAN({ eps: R.eps, minSamples: R.min_samples }).fit(X);
    same(m.labels, R.labels, key + ' DBSCAN labels'); same(m.coreIndices, R.core, key + ' DBSCAN core points');
  }
  for (const link of ['single', 'complete', 'average', 'ward']) {
    const R = U['agg_' + link], m = new ML.AgglomerativeClustering({ nClusters: 3, linkage: link }).fit(U.agg_X);
    same(m.merges.map((g) => [g.a, g.b]), R.children.map((c) => c.slice().sort((a, b) => a - b)), link + ' linkage merge order');
    close(m.merges.map((g) => g.dist), R.distances, 1e-10, link + ' linkage merge distances');
    sameUpToRelabel(m.labels, R.labels, link + ' linkage labels');
  }
  for (const ct of ['full', 'diag', 'spherical']) {
    const R = U['gmm_' + ct], g = new ML.GaussianMixture({ k: 3, covarianceType: ct, meansInit: R.means0, weightsInit: [1 / 3, 1 / 3, 1 / 3], covariancesInit: R.cov0, tol: 1e-6, maxIter: 500 }).fit(Xb);
    close(g.means, R.means, 1e-7, ct + ' GMM means'); close(g.weights, R.weights, 1e-7, ct + ' GMM weights'); close(g.covariances, R.cov, 1e-7, ct + ' GMM covariances');
    close(g.lowerBound, R.lower, 1e-9, ct + ' GMM log-likelihood'); close(g.bic(Xb), R.bic, 1e-9, ct + ' GMM BIC'); t.eq(g.nIter, R.n_iter, ct + ' GMM iterations');
    same(g.labels, R.labels, ct + ' GMM labels');
    t.ok(g.history.every((v, i) => i === 0 || v >= g.history[i - 1] - 1e-9), ct + ' EM never lowers the likelihood');
  }
  const [Xw] = XY('wine');
  for (const [key, X] of [['pca_iris', Xi], ['pca_wine', Xw]]) {
    const R = U[key], p = ML.PCA().fit(X);
    close(p.components, R.components, 1e-8, key + ' components (sign-normalised)'); close(p.explainedVariance, R.var, 1e-9, key + ' explained variance');
    close(p.explainedVarianceRatio, R.ratio, 1e-9, key + ' variance ratio'); close(p.singularValues, R.sv, 1e-9, key + ' singular values');
    const q = ML.PCA({ solver: 'eigen' }).fit(X);
    close(q.components, R.components, 1e-7, key + ' components by eigen-decomposition');
  }
  const p2 = ML.PCA({ nComponents: 2 }).fit(Xi);
  close(p2.inverseTransform(p2.transform(Xi.slice(0, 10))), U.pca_iris.recon, 1e-9, 'PCA reconstruction from 2 components');
  t.eq(ML.PCA({ nComponents: 0.95 }).fit(Xw).nComponents, U.pca_iris.n95, 'PCA components for 95% variance');
  const pf = ML.PCA().fit(Xi);
  close(pf.inverseTransform(pf.transform(Xi)), Xi, 1e-9, 'PCA with all components reconstructs exactly');
  // t-SNE: the joint probabilities match exactly; the optimised map is compared by its KL and its neighbourhoods
  const TS = U.tsne, ts = ML.TSNE({ perplexity: 10, method: 'exact', init: TS.init, maxIter: 500 });
  const [Y, ms] = time(() => ts.fitTransform(TS.X));
  close(ts.P.map((r, i) => r.filter((_, j) => j > i)).flat(), TS.P, 1e-6, 't-SNE joint probabilities P');
  close(ts.kl, TS.kl, 0.02, 't-SNE final KL divergence (within 2%)');
  const tw = ML.trustworthiness(TS.X, Y, 5);
  t.ok(Math.abs(tw - TS.trust) < 0.02, 't-SNE trustworthiness ' + tw.toFixed(3) + ' vs reference ' + TS.trust.toFixed(3));
  close(ML.trustworthiness(TS.X, TS.Y, 5), TS.trust, 1e-3, 'trustworthiness metric itself (iris has duplicate rows, so rank ties)');
  t.ok(ts.snapshots.length > 10 && ts.snapshots[0].Y.length === TS.X.length, 't-SNE snapshots for the animation');
  const bh = ML.TSNE({ perplexity: 20, method: 'barnes_hut', seed: 0 }), [Yb, msb] = time(() => bh.fitTransform(U.tsne_bh.X)), twb = ML.trustworthiness(U.tsne_bh.X, Yb, 5);
  t.ok(twb > U.tsne_bh.trust - 0.03, 'Barnes-Hut t-SNE trustworthiness ' + twb.toFixed(3) + ' vs reference ' + U.tsne_bh.trust.toFixed(3));
  t.ok(Math.abs(bh.kl - U.tsne_bh.kl) / U.tsne_bh.kl < 0.15, 'Barnes-Hut KL ' + bh.kl.toFixed(3) + ' vs reference ' + U.tsne_bh.kl.toFixed(3));
  t.ok(msb < 20000, 'Barnes-Hut t-SNE on ' + U.tsne_bh.X.length + ' points in ' + Math.round(msb) + ' ms');
  // UMAP-style layout: behaviour only
  const um = ML.UMAP({ seed: 1 }), Yu = um.fitTransform(Xb), twu = ML.trustworthiness(Xb, Yu, 5);
  t.ok(twu > 0.9, 'UMAP-style layout keeps neighbourhoods (trustworthiness ' + twu.toFixed(3) + ')');
  const agree = (a, b) => { const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]; return Math.max(...perms.map((q) => a.filter((v, i) => q[v] === b[i]).length / a.length)); };
  const ag = agree(new ML.KMeans({ k: 3, seed: 0, nInit: 5 }).fit(Yu).labels, yb);
  t.ok(ag > 0.95, 'UMAP-style layout keeps the three blobs apart (' + (100 * ag).toFixed(1) + '% agree with the true blobs)');
  const ab = ML.umapAB(0.1, 1);
  t.ok(Math.abs(ab.a - 1.577) < 0.02 && Math.abs(ab.b - 0.895) < 0.02, 'UMAP curve parameters for min_dist 0.1: a = ' + ab.a.toFixed(3) + ', b = ' + ab.b.toFixed(3));
}

// ================= model selection =================
{
  const M = ref('selection'), [Xi, yi] = XY('iris'), [Xw, yw] = XY('wine'), [Xr, yr] = XY('reg');
  const sameFolds = (a, b, msg) => { same(a.map((f) => f.train), b.map((f) => f.train), msg + ' train indices'); same(a.map((f) => f.test), b.map((f) => f.test), msg + ' test indices'); };
  sameFolds(ML.kFold(50, 4), M.kfold, 'KFold(4)'); sameFolds(ML.kFold(52, 7), M.kfold7, 'KFold(7) on 52 rows');
  sameFolds(ML.stratifiedKFold(yw, 5), M.skf_wine, 'StratifiedKFold(5) on wine'); sameFolds(ML.stratifiedKFold(M.skf_odd.y, 3), M.skf_odd.folds, 'StratifiedKFold(3), uneven classes');
  const sp = ML.trainTestSplit(yw.length, { testSize: 0.3, stratify: yw, seed: 3 });
  same(sp.testCounts, M.tts.test_counts, 'stratified split: test rows per class'); same(sp.trainCounts, M.tts.train_counts, 'stratified split: train rows per class');
  t.ok(new Set([...sp.train, ...sp.test]).size === yw.length && sp.train.length + sp.test.length === yw.length, 'split covers every row once');
  close(ML.crossValScore(new ML.LogisticRegression({ C: 1 }), Xi, yi), M.cv_logit, 1e-12, 'cross_val_score, logistic regression on iris');
  close(ML.crossValScore(new ML.DecisionTreeClassifier({ seed: M.seed0, float32: true }), Xw, yw), M.cv_tree, 1e-12, 'cross_val_score, tree on wine');
  close(ML.crossValScore(new ML.Ridge({ alpha: 1 }), Xr, yr), M.cv_ridge, 1e-9, 'cross_val_score, ridge (R2, KFold)');
  const gs = ML.gridSearch(new ML.KNeighborsClassifier(), { k: [1, 3, 5, 7, 9, 11], weights: ['uniform', 'distance'] }, Xw, yw);
  close(gs.results.map((r) => r.mean), M.grid.mean, 1e-12, 'grid search mean scores'); close(gs.results.map((r) => r.std), M.grid.std, 1e-12, 'grid search std');
  t.eq(gs.bestIndex, M.grid.best, 'grid search best candidate'); t.ok(gs.bestModel && gs.bestModel.fitted, 'grid search refits the best');
  const rs = ML.randomSearch(new ML.KNeighborsClassifier(), { k: [1, 3, 5, 7, 9, 11, 13], weights: ['uniform', 'distance'] }, Xw, yw, { nIter: 8, seed: 2 });
  t.ok(rs.results.length === 8 && rs.bestScore >= Math.max(...M.grid.mean) - 0.05, 'random search finds a good candidate');
  const lc = ML.learningCurve(new ML.DecisionTreeClassifier({ seed: M.seed0, float32: true, maxDepth: 3 }), Xw, yw);
  same(lc.sizes, M.lc.sizes, 'learning curve sizes'); close(lc.train, M.lc.train, 1e-12, 'learning curve training scores'); close(lc.test, M.lc.test, 1e-12, 'learning curve validation scores');
  const vc = ML.validationCurve(new ML.DecisionTreeClassifier({ seed: M.seed0, float32: true }), Xw, yw, 'maxDepth', [1, 2, 3, 4, 6]);
  close(vc.train, M.vc.train, 1e-12, 'validation curve training scores'); close(vc.test, M.vc.test, 1e-12, 'validation curve validation scores');
  // permutation importance: an unused feature scores ~0, the strongest one most
  const [X, y] = XY('tree3'), rf = new ML.RandomForestClassifier({ nEstimators: 30, seed: 1 }).fit(X, y);
  const pi = ML.permutationImportance(rf, X, y, { nRepeats: 5, seed: 1 });
  t.ok(pi.mean.every((v) => v > 0.02), 'permutation importance: every feature of tree3 matters (' + ML.fmtv(pi.mean) + ')');
  // on held-out rows a pure-noise column ranks last
  const g5 = ML.rng(5), Xn = X.map((r) => r.concat([g5()])), hs = ML.trainTestSplit(X.length, { testSize: 0.5, seed: 2 });
  const rn = new ML.RandomForestClassifier({ nEstimators: 50, seed: 2 }).fit(ML.take(Xn, hs.train), ML.take(y, hs.train)), pn = ML.permutationImportance(rn, ML.take(Xn, hs.test), ML.take(y, hs.test), { nRepeats: 20, seed: 2 });
  t.ok(pn.mean[3] < Math.min(...pn.mean.slice(0, 3)) && Math.abs(pn.mean[3]) < 0.03, 'permutation importance on held-out rows: the noise column ranks last (' + ML.fmtv(pn.mean) + ')');
}

// ================= properties =================
{
  const [X, y] = XY('cls2'), [Xr, yr] = XY('reg');
  // determinism: the same seed gives the same model
  for (const [name, mk] of [['k-means', () => new ML.KMeans({ k: 3, seed: 5 }).fit(X).labels], ['gradient boosting', () => new ML.GradientBoostingClassifier({ nEstimators: 10, subsample: 0.7, seed: 3 }).fit(X, y).predictProba(X)], ['isolation forest', () => new ML.IsolationForest({ nEstimators: 20, seed: 9 }).fit(X).scoreSamples(X)], ['t-SNE', () => ML.TSNE({ perplexity: 8, maxIter: 300, init: 'random', seed: 4 }).fitTransform(X.slice(0, 30))], ['UMAP-style', () => ML.UMAP({ seed: 4, nEpochs: 50 }).fitTransform(X)]])
    t.eq(mk(), mk(), name + ' is deterministic for a seed');
  // ridge shrinks: the coefficient norm falls as alpha grows
  const norms = [0.01, 1, 10, 100, 1000].map((a) => ML.la.norm(new ML.Ridge({ alpha: a }).fit(Xr, yr).coef));
  t.ok(norms.every((v, i) => i === 0 || v < norms[i - 1]), 'ridge coefficients shrink as alpha grows');
  // lasso path: more non-zero coefficients as alpha falls, none at alpha_max
  const P = ML.enetPath(Xr, yr, { nAlphas: 20 });
  t.ok(P.coefs[0].every((v) => v === 0) && P.coefs[19].filter((v) => v !== 0).length >= 3, 'lasso path starts empty and fills in');
  // probabilities are probabilities
  for (const m of [new ML.LogisticRegression(), new ML.GaussianNB(), new ML.LDA(), new ML.QDA(), new ML.KNeighborsClassifier(), new ML.DecisionTreeClassifier({ maxDepth: 3 }), new ML.RandomForestClassifier({ nEstimators: 10 }), new ML.GradientBoostingClassifier({ nEstimators: 10 })]) {
    const pr = m.fit(X, y).predictProba(X);
    t.ok(pr.every((r) => Math.abs(ML.sum(r) - 1) < 1e-9 && r.every((v) => v >= 0 && v <= 1)), m.name + ' probabilities sum to 1');
  }
  // pruning: larger ccp alpha never adds leaves
  const [X3, y3] = XY('tree3'), leaves = [0, 0.002, 0.005, 0.01, 0.03].map((a) => new ML.DecisionTreeClassifier({ ccpAlpha: a }).fit(X3, y3).nLeaves);
  t.ok(leaves.every((v, i) => i === 0 || v <= leaves[i - 1]), 'cost-complexity pruning: leaves ' + leaves.join(' >= '));
  // every trace step has a kind and a plain sentence
  const steps = [new ML.LinearRegression().fit(Xr, yr), new ML.Lasso({ alpha: 0.1 }).fit(Xr, yr), new ML.DecisionTreeClassifier({ maxDepth: 2 }).fit(X, y), new ML.AdaBoostClassifier({ nEstimators: 3 }).fit(X, y), new ML.SVC().fit(X, y), new ML.GaussianMixture({ k: 2 }).fit(X), new ML.DBSCAN({ eps: 0.5 }).fit(X), new ML.AgglomerativeClustering().fit(X.slice(0, 12))].flatMap((m) => m.trace);
  t.ok(steps.length > 50 && steps.every((s) => typeof s.kind === 'string' && typeof s.say === 'string' && s.say.length > 10 && /\.$/.test(s.say)), 'trace steps carry kind and a sentence');
  t.ok(new ML.SVC({ trace: false }).fit(X, y).trace.length === 0, 'tracing can be switched off');
  // speed budgets on demo-sized data
  const big = ML.range(600).map((i) => { const r = ML.rng(i); return [r.normal() + (i % 2) * 2, r.normal()]; }), by = big.map((_, i) => i % 2);
  const [, tSvc] = time(() => new ML.SVC({ trace: false }).fit(big, by)), [, tRf] = time(() => new ML.RandomForestClassifier({ nEstimators: 50, trace: false }).fit(big, by)), [, tKd] = time(() => new ML.KNeighborsClassifier({ algorithm: 'kd_tree' }).fit(big, by).predict(big));
  t.ok(tSvc < 3000 && tRf < 5000 && tKd < 2000, 'speed on 600 rows: SVC ' + Math.round(tSvc) + ' ms, forest ' + Math.round(tRf) + ' ms, k-d tree k-NN ' + Math.round(tKd) + ' ms');
}

// engine edge cases, against scikit-learn 1.x (values printed by a scratch run of ~/.nooral-venv)
{
  const near = (a, b, tol) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < tol);
  // more columns than rows: the least-norm solution (numpy's lstsq), not a crash
  const X = [[1, 2, 3, 4, 5], [2, 0, 1, 3, 1], [0, 1, 4, 2, 2]], y = [1, 2, 0.5];
  const lr = new ML.LinearRegression({ trace: false }).fit(X, y);
  t.ok(near(lr.coef, [0.19635627530364372, -0.076923076923077, -0.28744939271255066, 0.11943319838056679, -0.048582995951416984], 1e-9) && Math.abs(lr.intercept - 1.58502024291498) < 1e-9, 'LinearRegression with 5 columns and 3 rows: least-norm coefficients ' + lr.coef.map((v) => v.toFixed(4)));
  t.ok(near(lr.predict(X), y, 1e-9), 'the least-norm fit passes through every row');
  // multinomial naive Bayes with alpha = 0: clamped to 1e-10 as scikit-learn does (force_alpha=False)
  const nb = new ML.MultinomialNB({ alpha: 0, trace: false }).fit([[2, 0, 1], [0, 3, 0], [1, 1, 0], [0, 0, 2]], [0, 1, 0, 1]);
  const pr = nb.predictProba([[1, 1, 1], [0, 2, 1]]);
  t.ok(pr.flat().every(Number.isFinite) && near(pr[0], [0.9999999998, 1.9999999992999974e-10], 1e-12) && near(pr[1], [0.0526315789565097, 0.9473684210434903], 1e-9), 'MultinomialNB alpha = 0: ' + JSON.stringify(pr));
  t.ok(near(nb.featureLogProb[1], [-24.63528884243456, -0.5108256237926572, -0.916290731884155], 1e-6), 'MultinomialNB alpha = 0: the log probabilities of an unseen word are finite');
  // max features: a whole number is a count, a number below 1 a share
  const R = ML.resolveMaxFeatures;
  t.eq([R(1, 5), R(2, 5), R(5, 5), R(0.5, 5), R(0.1, 5), R('1.0', 5), R('0.4', 5), R('3', 5), R(null, 5), R('sqrt', 16)], [1, 2, null, 2, 1, null, 2, 3, null, 4], 'maxFeatures follows the count-or-share rule');
  const rf = new ML.RandomForestClassifier({ nEstimators: 5, maxFeatures: 1, seed: 3, trace: false }).fit(D.moons ? D.moons.X : [[0, 0], [1, 1], [0, 1], [1, 0]], D.moons ? D.moons.y : [0, 1, 1, 0]);
  t.ok(rf.trees.length === 5, 'a forest with maxFeatures 1 grows');
  // class weights, against scikit-learn (tests/ref/classic/classweight.json: 40 rows of class 0, 8 of class 1)
  const cwr = ref('classweight');
  const lb = new ML.LogisticRegression({ C: 1, classWeight: 'balanced', trace: false }).fit(cwr.X, cwr.y);
  t.ok(near(lb.coef[0], cwr.lr[0], 1e-6) && Math.abs(lb.intercept[0] - cwr.lr[1]) < 1e-6, 'logistic, balanced class weights: ' + lb.coef[0].map((v) => v.toFixed(5)) + ' ' + lb.intercept[0].toFixed(5));
  const l3 = new ML.LogisticRegression({ C: 1, classWeight: { 0: 1, 1: 3 }, trace: false }).fit(cwr.X, cwr.y);
  t.ok(near(l3.coef[0], cwr.lr2[0], 1e-6) && Math.abs(l3.intercept[0] - cwr.lr2[1]) < 1e-6, 'logistic, custom class weights');
  const tb = new ML.DecisionTreeClassifier({ maxDepth: 2, classWeight: 'balanced', trace: false }).fit(cwr.X, cwr.y);
  t.ok(tb.nLeaves === cwr.dtl && tb.predictProba(cwr.X).every((r, i) => near(r, cwr.dt[i], 1e-9)), 'tree, balanced class weights: the same leaves and probabilities');
  const sb = new ML.SVC({ C: 1, kernel: 'linear', classWeight: 'balanced', trace: false }).fit(cwr.X, cwr.y);
  const dfs = sb.decisionFunction(cwr.X);
  t.ok(near(dfs, cwr.sv, 2e-3), 'SVC, balanced class weights: decision values within 2e-3 (largest gap ' + Math.max(...dfs.map((v, i) => Math.abs(v - cwr.sv[i]))).toExponential(2) + ')');
  const rf0 = new ML.RandomForestClassifier({ nEstimators: 30, maxDepth: 2, seed: 2, trace: false }).fit(cwr.X, cwr.y), rfb = new ML.RandomForestClassifier({ nEstimators: 30, maxDepth: 2, seed: 2, classWeight: 'balanced', trace: false }).fit(cwr.X, cwr.y);
  const hits = (m) => m.predict(cwr.X).filter((v, i) => v === 1 && cwr.y[i] === 1).length;
  t.ok(hits(rfb) > hits(rf0), 'forest, balanced class weights: more of the rare class found (' + hits(rf0) + ' → ' + hits(rfb) + ' of 8)');
  // SVC probabilities (Platt scaling on held-out decision values), against scikit-learn's probability=True
  const pr2 = ref('platt');
  const ab = ML.plattFit(pr2.f, pr2.y.map((v) => v === 1));
  t.ok(Math.abs(ab.A - pr2.a) < 1e-6 && Math.abs(ab.B - pr2.b) < 1e-6, 'Platt sigmoid on fixed decision values: A ' + ab.A.toFixed(4) + ' B ' + ab.B.toFixed(4) + ' (scikit-learn ' + pr2.a.toFixed(4) + ', ' + pr2.b.toFixed(4) + ')');
  const sp2 = new ML.SVC({ C: 1, kernel: 'rbf', gamma: 0.5, probability: true, trace: false }).fit(pr2.X, pr2.Y).predictProba(pr2.X);
  const gap2 = Math.max(...sp2.map((r, i) => Math.abs(r[1] - pr2.p[i][1])));
  t.ok(gap2 < 0.05 && sp2.every((r) => Math.abs(r[0] + r[1] - 1) < 1e-12), 'SVC probabilities, two classes: within ' + gap2.toFixed(3) + ' of scikit-learn (other folds)');
  const sp3 = new ML.SVC({ C: 1, kernel: 'rbf', gamma: 0.3, probability: true, trace: false }).fit(pr2.X3, pr2.Y3).predictProba(pr2.X3);
  const gap3 = Math.max(...sp3.flatMap((r, i) => r.map((q, k) => Math.abs(q - pr2.p3[i][k]))));
  t.ok(gap3 < 0.06 && sp3.every((r) => Math.abs(r.reduce((a, b) => a + b, 0) - 1) < 1e-9), 'SVC probabilities, three classes (pairwise coupling): within ' + gap3.toFixed(3) + ' of scikit-learn');
  t.ok(typeof new ML.SVC({ trace: false }).fit(pr2.X, pr2.Y).predictProba !== 'function', 'no probabilities unless asked for');
}

t.done();
