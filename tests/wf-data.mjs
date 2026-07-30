// The data, prepare, split and unsupervised blocks of the workflow canvas (node tests/wf-data.mjs): each runs on a set
// that suits it, its pipe step does to raw rows what it did to the table, views stay small, and cross-validation
// gives the same scores for the same seed.
import { load, harness } from './load.mjs';
const S = load(['core', 'app']);
const WF = S.WF, t = harness('wf-data');
const size = (v) => JSON.stringify(v).length;
const close = (a, b) => a.length === b.length && a.every((r, i) => r.every((x, j) => (Number.isNaN(x) && Number.isNaN(b[i][j])) || Math.abs(x - b[i][j]) < 1e-9));

function run(spec) {
  const doc = WF.build(spec);
  const s = WF.runSync(doc);
  for (const b of doc.blocks) t.ok(s.res[b.id].state === 'done', b.id + ' (' + b.type + ') ran: ' + (s.res[b.id].error || s.res[b.id].state));
  for (const b of doc.blocks) if (s.res[b.id].view) t.ok(size(s.res[b.id].view) < 300000, b.id + ' view is small (' + size(s.res[b.id].view) + ')');
  return s;
}
/** a block's pipe, run on the raw rows of the data it came from, gives its table */
function pipeAgrees(s, src, id, port) {
  const raw = s.output(src, 'out'), out = s.output(id, port || 'out');
  const rows = out.rows.map((r) => raw.X[raw.rows.indexOf(r)]);
  t.ok(close(WF.applyPipe(out.pipe, rows), out.X), id + ': the pipe redoes the table on raw rows');
}

// ---- data, csv, split -----------------------------------------------------------------------------------------
{
  const s = run({ blocks: { d: ['data', { set: 'loans', n: 300 }], sp: ['split', { test: 0.3 }] }, wires: [['d.out', 'sp.in']] });
  const v = s.res.d.view;
  t.ok(v.columns.length === 6 && v.rows.length === 12 && v.counts.length === 2, 'the data view has columns, rows and class counts');
  t.ok(v.points && v.points.P.length === 300, 'a wide table is drawn on its principal components');
  t.eq([s.output('sp', 'train').part, s.output('sp', 'test').part], ['train', 'test'], 'the split marks its sides');
  t.ok(s.res.sp.view.points.test.filter(Boolean).length === 90, 'the split view marks the test rows');
}
{
  const csv = 'height,weight,colour,label\n1.2,30,red,a\n1.5,,blue,b\n1.1,28,red,a\n1.7,50,green,b\nNA,41,blue,b\n1.3,33,red,a\n';
  const doc = WF.build({ blocks: { c: ['csv', { csv, name: 'tiny' }] } });
  const s = WF.runSync(doc);
  const v = s.output('c', 'out');
  t.eq([v.X.length, v.features.length, v.task, v.classes], [6, 3, 'classification', ['a', 'b']], 'a CSV becomes a table');
  t.ok(Number.isNaN(v.X[1][1]) && Number.isNaN(v.X[4][0]), 'missing cells are NaN');
  t.eq(v.features[2].type, 'category', 'a word column is a category');
  const doc2 = WF.build({ blocks: { c: ['csv', { csv, target: 'weight' }] } });
  const v2 = WF.runSync(doc2).output('c', 'out');
  t.eq([v2.task, v2.y.length], ['regression', 5], 'a number target is regression, and rows without it are left out');
}

// ---- prepare --------------------------------------------------------------------------------------------------------
{
  const s = run({ blocks: { d: ['data', { set: 'pumps', n: 400, seed: 3 }], sp: ['split', {}], cl: ['clean', { mode: 'cols', maxMissing: 0.5 }], im: ['impute', { strategy: 'median', indicator: true }], sc: ['scale', { how: 'robust' }], sel: ['select', { k: 2, score: 'mi' }] },
    wires: [['d.out', 'sp.in'], ['sp.train', 'cl.in'], ['sp.test', 'cl.test'], ['cl.out', 'im.in'], ['cl.test', 'im.test'], ['im.out', 'sc.in'], ['im.test', 'sc.test'], ['sc.out', 'sel.in'], ['sc.test', 'sel.test']] });
  const raw = s.output('d', 'out');
  t.ok(WF.hasMissing(raw), 'pumps has missing cells');
  t.ok(!WF.hasMissing(s.output('im', 'out')) && !WF.hasMissing(s.output('im', 'test')), 'impute fills both sides');
  t.eq(s.output('sel', 'out').features.length, 2, 'select keeps k columns');
  t.eq(s.output('sel', 'test').features.length, 2, 'and the test side has the same');
  t.ok(s.res.sc.view.leak.ok, 'fitting on the training side is not leakage');
  pipeAgrees(s, 'd', 'sel');
  pipeAgrees(s, 'd', 'sel', 'test');
  const m = s.output('sc', 'out'), med = m.X.map((r) => r[0]).sort((a, b) => a - b)[Math.floor(m.X.length / 2)];
  t.ok(Math.abs(med) < 0.05, 'robust scaling centres the median: ' + med);
}
{
  const s = run({ blocks: { d: ['data', { set: 'loans', n: 200 }], en: ['encode', {}], sc: ['scale', {}], po: ['poly', { degree: 2 }], pc: ['pca', { n: 3 }] }, wires: [['d.out', 'en.in'], ['en.out', 'sc.in'], ['sc.out', 'po.in'], ['po.out', 'pc.in']] });
  t.eq(s.output('en', 'out').features.length, 7, 'one-hot turns the group column into two');
  t.ok(!s.res.sc.view.leak.ok && /leak/.test(s.res.sc.view.leak.note), 'scaling before a split is called leakage');
  const sc = s.output('sc', 'out'), col = sc.X.map((r) => r[0]), mean = col.reduce((a, b) => a + b, 0) / col.length;
  t.ok(Math.abs(mean) < 1e-9, 'standard scaling centres');
  t.eq(s.output('po', 'out').features.length, 35, 'degree 2 on 7 columns makes 35 features');
  t.eq(s.output('pc', 'out').features.map((f) => f.name), ['PC1', 'PC2', 'PC3'], 'PCA keeps 3 components');
  t.ok(s.res.pc.view.info.ratio[0] >= s.res.pc.view.info.ratio[1], 'variance ratios fall');
  pipeAgrees(s, 'd', 'pc');
}
{
  const s = run({ blocks: { d: ['data', { set: 'spam' }], sp: ['split', {}], ve: ['vectorise', { weight: 'tfidf', vocab: 50 }] }, wires: [['d.out', 'sp.in'], ['sp.train', 've.in'], ['sp.test', 've.test']] });
  const v = s.output('ve', 'out');
  t.eq([v.kind, v.features.length], ['table', 50], 'texts become a table of 50 words');
  const norm = Math.sqrt(v.X[0].reduce((a, b) => a + b * b, 0));
  t.ok(Math.abs(norm - 1) < 1e-9 || norm === 0, 'TF-IDF rows have length 1');
  const te = s.output('ve', 'test');
  t.ok(close(WF.applyPipe(te.pipe, te.texts), te.X), 'the pipe turns raw texts into the same rows');
  // a model learns from the words
  const s2 = run({ blocks: { d: ['data', { set: 'spam' }], sp: ['split', { test: 0.3 }], ve: ['vectorise', {}], m: ['logistic', {}], tr: ['train', {}], ev: ['evaluate', {}] },
    wires: [['d.out', 'sp.in'], ['sp.train', 've.in'], ['sp.test', 've.test'], ['m.model', 'tr.model'], ['ve.out', 'tr.data'], ['tr.model', 'ev.model'], ['ve.test', 'ev.data']] });
  t.ok(WF.score('accuracy', s2.output('ev', 'pred')) > 0.6, 'spam is learned from word counts: ' + WF.score('accuracy', s2.output('ev', 'pred')));
}
{
  const s = run({ blocks: { d: ['data', { set: 'pumps', n: 300 }], cl: ['clean', { mode: 'rows' }], cp: ['clean', { mode: 'clip' }] }, wires: [['d.out', 'cl.in'], ['d.out', 'cp.in']] });
  const raw = s.output('d', 'out'), out = s.output('cl', 'out');
  t.ok(out.y.length < raw.y.length && !WF.hasMissing(out), 'dropping rows removes every gap');
  t.ok(s.res.cp.view.info.clipped.some((c) => c > 0), 'clipping catches some outliers');
}

// ---- cross-validation -----------------------------------------------------------------------------------------------
{
  const spec = { blocks: { d: ['data', { set: 'wildflowers' }], m: ['logistic', {}], cv: ['cv', { k: 5, seed: 4 }] }, wires: [['d.out', 'cv.data'], ['m.model', 'cv.model']] };
  const a = run(spec), b = run(spec);
  t.eq(a.res.cv.view.perFold.map((f) => f.score), b.res.cv.view.perFold.map((f) => f.score), 'the same seed gives the same fold scores');
  t.eq(a.res.cv.view.perFold.length, 5, 'five folds');
  const p = a.output('cv', 'pred');
  t.ok(p.pred.every((x) => x !== undefined) && p.folds.every((f) => f >= 0), 'every row is predicted once');
  t.ok(a.res.cv.view.mean > 0.8, 'wildflowers are told apart: ' + a.res.cv.view.mean);
}

// ---- unsupervised -----------------------------------------------------------------------------------------------------
{
  const s = run({ blocks: { d: ['data', { set: 'blobs', n: 240 }], km: ['kmeans', { k: 3 }], db: ['dbscan', { eps: 0.9, minPts: 5 }], hc: ['hierarchical', { k: 3 }], gm: ['gmm', { k: 3 }] },
    wires: [['d.out', 'km.in'], ['d.out', 'db.in'], ['d.out', 'hc.in'], ['d.out', 'gm.in']] });
  for (const id of ['km', 'hc', 'gm']) t.ok(s.res[id].view.ari > 0.8, id + ' finds the blobs: ARI ' + s.res[id].view.ari);
  t.ok(s.res.km.view.steps.length >= 2 && s.res.km.view.elbow.length === 8, 'k-means keeps its steps and the elbow');
  t.ok(s.res.db.view.clusters >= 2, 'DBSCAN finds clusters: ' + s.res.db.view.clusters);
  t.ok(s.res.gm.view.em.length >= 2 && s.res.gm.view.clouds.length === 3, 'the mixture keeps its EM rounds');
  for (const id of ['km', 'db', 'hc', 'gm']) {
    const o = s.output(id, 'out'), raw = s.output('d', 'out');
    t.ok(close(WF.applyPipe(o.pipe, raw.X), o.X), id + ': new rows are placed in the cluster they were given');
  }
}
{
  const s = run({ blocks: { d: ['data', { set: 'wildflowers' }], ts: ['tsne', { iters: 300 }], um: ['umap', { epochs: 60 }] }, wires: [['d.out', 'ts.in'], ['d.out', 'um.in']] });
  t.eq(s.output('ts', 'out').features.length, 2, 't-SNE gives two columns');
  t.ok(s.res.ts.view.snaps.length > 3 && s.res.um.view.snaps.length > 3, 'the maps keep snapshots');
  let threw = '';
  try { WF.applyPipe(s.output('um', 'out').pipe, [[1, 2, 3, 4]]); } catch (e) { threw = e.message; }
  t.ok(/cannot place new rows/.test(threw), 'a map refuses new rows');
}
t.done();
