// The evaluation, comparison, inference and explanation blocks of the workflow canvas (node tests/wf-eval.mjs).
import { load, harness } from './load.mjs';
const S = load(['core', 'app']);
const WF = S.WF, M = S.Metrics, t = harness('wf-eval');
const size = (v) => JSON.stringify(v).length;
const run = (id) => { const d = WF.sample(id); const s = WF.runSync(d); return { d, s }; };
const ok = (s, d, ids) => { for (const id of ids) { const r = s.res[id]; t.eq(r && r.state, 'done', d.name + ': ' + id + ' ran' + (r && r.error ? ': ' + r.error : '')); if (r && r.view) t.ok(size(r.view) < 400000, id + ' view is small (' + size(r.view) + ' bytes)'); } };

// evaluate, roc, calibration on loans
{
  const { d, s } = run('evaluate-loans');
  ok(s, d, ['evaluate1', 'roc1', 'calibration1']);
  const ev = s.res.evaluate1.view, pred = s.output('evaluate1', 'pred');
  // every example in a cell has the cell's truth and prediction, and is the row it says
  let good = true;
  ev.cells.forEach((row, i) => row.forEach((list, j) => list.forEach((e) => { if (e.y !== i || e.pred !== j || pred.rows[e.k] !== e.row || pred.y[e.k] !== i) good = false; })));
  t.ok(good, 'confusion cell examples are the right rows');
  t.eq(ev.cellTotals.flat().reduce((a, b) => a + b, 0), pred.y.length, 'cell totals add up');
  t.ok(Math.abs(ev.mcc - M.mcc(pred.y, pred.pred).value) < 1e-12, 'MCC');
  const rv = s.res.roc1.view;
  for (const th of [0.2, 0.5, 0.73]) {
    const a = WF.EV.at(rv, 1, th), b = M.atThreshold(pred.y, pred.proba.map((r) => r[1]), th, 1);
    // the view keeps probabilities to four places, so compare with the rounded scores
    const c = M.atThreshold(pred.y, pred.proba.map((r) => Math.round(r[1] * 1e4) / 1e4), th, 1);
    t.eq([a.tp, a.fp, a.fn, a.tn], [c.tp, c.fp, c.fn, c.tn], 'ROC view matrix at ' + th);
    t.ok(Math.abs(a.tp - b.tp) + Math.abs(a.fp - b.fp) <= 2, 'and agrees with the exact scores at ' + th);
  }
  const cv = EVcurves(rv);
  t.ok(Math.abs(cv.roc.auc - M.roc(pred.y, pred.proba.map((r) => r[1])).auc) < 1e-3, 'AUC from the view');
  const cal = s.res.calibration1.view;
  t.eq(cal.bins.reduce((a, b) => a + b.count, 0), pred.y.length, 'calibration bins hold every row');
}
function EVcurves(v) { return WF.EV.curves(v, v.cls); }

// compare three models, with boundaries
{
  const { d, s } = run('compare-three');
  ok(s, d, ['compare1']);
  const v = s.res.compare1.view;
  t.eq(v.items.length, 3, 'three models compared');
  t.ok(v.items.every((it) => it.boundary && it.boundary.vals.length === 48 * 48), 'each has a boundary');
  t.ok(v.metrics.includes('accuracy'), 'accuracy compared');
}
// predict on a table and on the 2-D plane
{
  const { d, s } = run('predict-exam');
  ok(s, d, ['predict1']);
  const v = s.res.predict1.view;
  t.ok(v.features.length >= 3 && v.features.every((f) => f.type === 'category' || Number.isFinite(f.min)), 'raw fields with ranges');
  const ans = s.call('predict1', 'predict', { rows: [v.features.map((f) => f.start)] });
  t.ok(ans.length === 1 && ans[0].proba && Math.abs(ans[0].proba.reduce((a, b) => a + b, 0) - 1) < 1e-3, 'a typed row gets probabilities');
  t.ok(ans[0].why, 'and a reason');
  const m2 = run('predict-moons');
  ok(m2.s, m2.d, ['predict1']);
  t.ok(m2.s.res.predict1.view.plane && m2.s.res.predict1.view.plane.vals.length === 44 * 44, 'the moons plane');
}
// explanations
{
  const { d, s } = run('explain-harvest');
  ok(s, d, ['importance1', 'pdp1', 'shap1', 'lime1']);
  const sh = s.res.shap1.view;
  t.ok(Math.abs(sh.base + sh.phi.reduce((a, b) => a + b, 0) - sh.fx) < 1e-2, 'SHAP values add up to the prediction (' + sh.method + ')');
  t.eq(s.res.importance1.view.mean.length, s.res.importance1.view.features.length, 'importance per feature');
  t.ok(s.res.pdp1.view.grid.length > 3, 'pdp grid');
}
// residuals and the learning curve
{
  const { d, s } = run('regression-residuals');
  ok(s, d, ['evaluate1', 'residuals1', 'learningcurve1']);
  const lc = s.res.learningcurve1.view;
  t.ok(lc.sizes.length === 7 && lc.sizes[0] < lc.sizes[6], 'growing sizes');
  const l2 = run('learning-curve-tree');
  ok(l2.s, l2.d, ['learningcurve1']);
  t.ok(l2.s.res.learningcurve1.view.sweep && l2.s.res.learningcurve1.view.sweep.pts.length >= 5, 'the validation curve over a setting');
}
// errors say what to do
{
  const doc = WF.build({ blocks: { data1: ['data', { set: 'moons' }], m1: ['logistic', {}], train1: ['train', {}], ev1: ['evaluate', {}], res1: ['residuals', {}] }, wires: [['data1.out', 'train1.data'], ['m1.model', 'train1.model'], ['train1.model', 'ev1.model'], ['data1.out', 'ev1.data'], ['ev1.pred', 'res1.pred']] });
  const s = WF.runSync(doc);
  t.ok(/number to predict/.test(s.res.res1.error || ''), 'residuals refuse classes: ' + s.res.res1.error);
}
// image models: saliency (when a network block is in this build)
const imgOk = (() => { try { const v = WF.loadSet('digits', { n: 4 }); return v.X[0] && v.X[0].length > 1; } catch (e) { return false; } })();
if ((WF.def('cnn') || WF.def('mlp')) && imgOk) {
  const { d, s } = run('predict-digits');
  ok(s, d, ['predict1', 'saliency1']);
  const v = s.res.saliency1.view;
  t.ok(v.items.length === 4 && v.items[0].sal.length === v.w * v.h, 'a saliency map per image');
  if (WF.byId(d, 'm1').type === 'cnn') t.ok(v.items.every((it) => it.cam && it.how === 'backpropagation'), 'Grad-CAM and exact saliency for a CNN');
}
t.done();
