// The model families of the workflow canvas: every sample trains and scores sensibly, networks are deterministic for a
// seed and their trainers step by batch and by epoch, and every view is small enough to cross from the worker.
import { load, harness } from './load.mjs';
const S = load(['core', 'app']);
const WF = S.WF, t = harness('wf-models');

const want = { 'tree-moons': 0.85, 'svm-circles': 0.9, 'forest-moons': 0.85, 'forest-loans': 0.7, 'boost-moons': 0.85, 'ada-xor': 0.85, 'knn-blobs': 0.85, 'nb-moons': 0.75, 'qda-blobs': 0.85, 'perceptron-line': 0.75,
  'mlp-spirals': 0.8, 'cnn-digits': 0.6, 'lstm-spam': 0.75, 'transformer-spam': 0.7, 'regression-rents': 0.8, 'lasso-rents': 0.8, 'svr-curve': 0.8, 'boost-curve': 0.8 };
const quick = process.argv.includes('--quick');
for (const [id, min] of Object.entries(want)) {
  if (quick && /cnn/.test(id)) continue;
  const doc = WF.sample(id);
  t.ok(doc, 'sample ' + id);
  if (!doc) continue;
  const t0 = Date.now(), s = WF.runSync(doc), ms = Date.now() - t0;
  for (const b of doc.blocks) t.eq(s.res[b.id].state, 'done', id + ': ' + b.id + ' ran ' + (s.res[b.id].error || ''));
  const p = s.output('evaluate1', 'pred');
  if (!p) continue;
  const score = WF.score(p.task === 'regression' ? 'r2' : 'accuracy', p);
  t.ok(score >= min, id + ': test score ' + score.toFixed(3) + ' >= ' + min);
  const size = JSON.stringify(s.res.train1.view).length;
  t.ok(size < 400000, id + ': the Train view is ' + Math.round(size / 1024) + ' KB');
  t.ok(ms < 30000, id + ': trained in ' + ms + ' ms');
  const v = s.res.train1.view;
  t.ok(v.internals && !v.internals.error, id + ': internals ' + (v.internals && v.internals.error || ''));
  if (doc.blocks.some((b) => ['moons', 'circles', 'spirals', 'blobs', 'xor', 'linear'].includes(b.params.set))) t.ok(v.plane && v.plane.vals.length === (v.plane.n + 1) ** 2, id + ': the plane is sampled');
}

// a network is the same network for the same seed, and stepping by batch and epoch counts right
const mk = () => { const d = WF.sample('mlp-spirals'); WF.byId(d, 'train1').params.epochs = 4; return d; };
const a = WF.runSync(mk()), b = WF.runSync(mk());
t.eq(JSON.stringify(a.res.train1.view.curve), JSON.stringify(b.res.train1.view.curve), 'the same seed gives the same training');
const d2 = mk(); WF.byId(d2, 'train1').params.seed = 7;
t.ok(JSON.stringify(WF.runSync(d2).res.train1.view.curve) !== JSON.stringify(a.res.train1.view.curve), 'another seed gives another run');

const sess = new WF.Session({ autoTrain: false }).load(mk());
await sess.run();
t.eq(sess.res.train1.state, 'paused', 'paused at the start');
await sess.control('train1', 'batch');
const r1 = sess.res.train1.result;
t.eq([r1.epoch, r1.batch, r1.steps], [0, 1, 1], 'one batch');
await sess.control('train1', 'epoch');
const r2 = sess.res.train1.result;
t.eq([r2.epoch, r2.batch], [1, 0], 'to the end of the epoch');
t.eq(r2.curve.length, 1, 'one point on the curves');
t.ok(r2.curve[0].valLoss !== null, 'with a validation loss');
await sess.control('train1', 'play');
t.eq(sess.res.train1.result.epoch, 4, 'play runs to the last epoch');
t.eq(sess.res.evaluate1.state, 'done', 'and the evaluation follows');
const live = sess.trainers.train1.trainer.snapshot();
t.ok(live.live && live.internals && live.plane, 'a live snapshot carries the insides and the plane');

// the Train block answers questions about one row
const k = WF.runSync(WF.sample('knn-blobs'));
const nb = k.call('train1', 'neighbours', { x: [0, 0] });
t.eq(nb.all.length, 7, 'k-NN: seven neighbours');
t.ok(k.call('train1', 'predictRow', { x: [0, 0] }).proba.length === 3, 'predictRow gives the probabilities');

// more features than rows: linear regression falls back to the smallest-weight answer; QDA says what is wrong
const wide = WF.build({ blocks: { data1: ['data', { set: 'rents', n: 40 }], poly: ['poly', { degree: 3 }], m: ['linreg', {}], train1: ['train', {}] }, wires: [['data1.out', 'poly.in'], ['m.model', 'train1.model'], ['poly.out', 'train1.data']] });
t.ok(!!WF.def('poly'), 'the polynomial block is there');
const ws = WF.runSync(wide);
t.eq(ws.res.train1.state, 'done', 'linear regression on wide data: ' + (ws.res.train1.error || ''));
t.ok(ws.res.train1.view && ws.res.train1.view.internals && ws.res.train1.view.internals.leastNorm, 'and it used the smallest-weight answer');
const fam = WF.families.linreg, g = S.ML.rng(3), X = Array.from({ length: 12 }, () => Array.from({ length: 20 }, () => g())), y = X.map((r) => r[0] + g());
const est = fam.make({ solver: 'qr' }, { X, y, features: X[0].map((_, j) => ({ name: 'f' + j })) });
est.fit(X, y);
t.ok(est.leastNorm && est.predict(X).every((v, i) => Math.abs(v - y[i]) < 1e-3), 'least-norm fit interpolates 12 rows with 20 features');
const q = WF.families.qda.make({ regParam: 0 }, { X });
let msg = '';
try { q.fit(X, X.map((_, i) => i % 2)); } catch (e) { msg = e.message; }
t.ok(/too few to estimate its own covariance/.test(msg), 'QDA on wide data says why: ' + msg);
t.done();
