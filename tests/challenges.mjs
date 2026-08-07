// Challenges and the machine-learning exercise kinds: scoring on a hidden set, the limits, medals, and the checkers of
// matrix, point, predict-curve and workflow (node tests/challenges.mjs).
import { load, harness } from './load.mjs';
const S = load(['core', 'stage', 'topics', 'app', 'content']);
const WF = S.WF, C = S.Challenges, K = S.Check, t = harness('challenges');

const x = S.Content.challenges.get('ev-ch-odds');
t.ok(!!x, 'the loans challenge is registered');
t.ok(![...S.Content.challenges.values(), ...S.Content.exercises.values()].some((c) => c.sample), 'no placeholder challenges or exercises');
const sol = WF.fromJSON(x.solution);
const s = WF.runSync(sol);
const r = WF.scoreHidden(s, { data: x.data, hidden: x.hidden, metric: x.metric });
t.ok(r.score > 0 && r.score < 1, 'a hidden score: ' + r.score);
t.eq(r.n, x.hidden.n, 'scored on the hidden rows');
t.eq(r.block, WF.modelToScore(sol), 'the trained model feeding Evaluate is scored');
// the hidden rows are not the training rows
const pub = WF.loadSet(x.data.set, Object.assign({ n: x.data.n, seed: x.data.seed })), hid = WF.hiddenSet(x.data, x.hidden);
t.ok(JSON.stringify(pub.X[0]) !== JSON.stringify(hid.X[0]), 'the hidden set differs from the public one');
// fixed pools (classic sets, digits, texts): the hidden rows are held out from every public draw
for (const set of ['iris', 'wine', 'spam', 'penguins']) {
  t.ok(WF.isPool(set), set + ' is a fixed pool');
  // the row's place in the pool (a few classic rows repeat the same measurements, so the values would not do)
  const key = (v, i) => v.rows[i];
  const H = WF.hiddenSet({ set }, { seed: 5, n: 40 }), hk = new Set(H.y.map((_, i) => key(H, i)));
  let shared = 0;
  for (const seed of [1, 2, 3]) { const P = WF.publicSet({ set, seed, n: 0 }); for (let i = 0; i < P.y.length; i++) if (hk.has(key(P, i))) shared++; }
  t.eq(shared, 0, set + ': no hidden row is a public row');
  t.ok(H.y.length === 40 || H.y.length === WF.poolParts({ set }).hold.length, set + ': the hidden rows asked for');
}
t.ok(!WF.isPool('moons') && !WF.isPool('loans') && !WF.isPool('digits'), 'generated sets draw fresh rows');
// a decision threshold on the model block: Evaluate and the hidden-set score both use it
{
  const mk = (mp) => WF.build({ blocks: { data: WF.challengeBlock(x.data), split: ['split', { test: 0.25 }], m: ['logistic', mp], tr: ['train', {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['m.model', 'tr.model'], ['split.train', 'tr.data'], ['tr.model', 'ev.model'], ['split.test', 'ev.data']] });
  const at = (mp, metric) => WF.scoreHidden(WF.runSync(mk(mp)), { data: x.data, hidden: x.hidden, metric }).score;
  const r5 = at({}, 'recall'), r2 = at({ threshold: 0.2 }, 'recall'), p5 = at({}, 'precision'), p2 = at({ threshold: 0.2 }, 'precision');
  t.ok(r2 > r5 && p2 < p5, 'threshold 0.2 raises hidden recall (' + r5.toFixed(3) + ' → ' + r2.toFixed(3) + ') and lowers precision (' + p5.toFixed(3) + ' → ' + p2.toFixed(3) + ')');
  const s2 = WF.runSync(mk({ threshold: 0.2 }));
  const ev = s2.output('ev', 'pred'), want = ev.proba.map((q) => (q[1] >= 0.2 ? 1 : 0));
  t.eq(ev.pred, want, 'Evaluate predicts class 1 at probability 0.2 and above');
  // class weights reach the engine: balanced weights find more of the rare class
  t.ok(at({ classWeight: 'balanced' }, 'recall') > r5, 'balanced class weights raise hidden recall');
  t.ok(at({ classWeight: 'custom', classWeights: '1, 4' }, 'recall') > r5, 'custom class weights raise hidden recall');
}
// resampling: the training rows only; the test rows pass through
{
  const doc = WF.build({ blocks: { data: ['data', { set: 'pumps', n: 400, seed: 5 }], split: ['split', { test: 0.25 }], rs: ['resample', { how: 'over' }], m: ['logistic', {}], tr: ['train', {}], ev: ['evaluate', {}] },
    wires: [['data.out', 'split.in'], ['split.train', 'rs.in'], ['split.test', 'rs.test'], ['m.model', 'tr.model'], ['rs.out', 'tr.data'], ['tr.model', 'ev.model'], ['rs.test', 'ev.data']] });
  const ss = WF.runSync(doc), tr = ss.output('split', 'train'), te = ss.output('split', 'test'), o = ss.output('rs', 'out'), ot = ss.output('rs', 'test');
  const cnt = (v) => v.classes.map((_, k) => v.y.filter((c) => c === k).length);
  t.ok(cnt(o)[0] === cnt(o)[1] && cnt(o)[1] > cnt(tr)[1] && cnt(o)[0] === cnt(tr)[0], 'oversampling evens the training classes: ' + cnt(tr) + ' → ' + cnt(o));
  t.ok(ot === te, 'the test rows pass through Resample unchanged');
  const u = WF.resample(tr.y, 2, 'under', 1, 3);
  t.ok(u.after[0] === u.after[1] && u.after[1] === cnt(tr)[1], 'undersampling keeps the rare class whole and cuts the common one: ' + u.after);
}
// clusterings scored on hidden rows: ARI against the classes, silhouette of the groups
{
  const data = { set: 'blobs', n: 300, seed: 4 }, hidden = { seed: 44, n: 300 };
  const mk = (k) => WF.build({ blocks: { data: WF.challengeBlock(data), sc: ['scale', { how: 'standard' }], km: ['kmeans', { k, elbow: false }] }, wires: [['data.out', 'sc.in'], ['sc.out', 'km.in']] });
  const s3 = WF.runSync(mk(3)), s6 = WF.runSync(mk(6));
  const a3 = WF.scoreHidden(s3, { data, hidden, metric: 'ari' }), a6 = WF.scoreHidden(s6, { data, hidden, metric: 'ari' });
  t.ok(a3.score > 0.8 && a3.score > a6.score && a3.n === 300 && a3.block === 'km', 'ARI on hidden rows: k = 3 ' + a3.score.toFixed(3) + ' beats k = 6 ' + a6.score.toFixed(3));
  const sl = WF.scoreHidden(s3, { data, hidden, metric: 'silhouette' });
  t.ok(sl.score > 0.3 && sl.score <= 1, 'silhouette on hidden rows: ' + sl.score.toFixed(3));
  // the same groups by hand: the hidden rows scaled with the training statistics, then their nearest centre
  const km = s3.res.km.result.km, H = WF.hiddenSet(data, hidden), sc = s3.output('sc', 'out').pipe;
  const lab = km.predict(WF.applyPipe(sc, H.X));
  t.ok(Math.abs(S.Metrics.adjustedRand(H.y, lab).value - a3.score) < 1e-12, 'the hidden groups are the nearest centres of the scaled hidden rows');
  t.ok(WF.scoreHidden(WF.runSync(WF.build({ blocks: { data: WF.challengeBlock(data) }, wires: [] })), { data, hidden, metric: 'ari' }).problems.length > 0, 'no clustering block: a problem, not a score');
}
// a constraint: the parity gap on the hidden rows, and a broken one means the target is not reached
{
  const y = S.Content.challenges.get('rb-ch-fair-loans');
  const good = C.evaluate(y.solution, y), bad = C.evaluate(y.wrong[0], y);
  t.ok(good.reached && good.constraints[0].ok && good.constraints[0].value <= 0.06, 'the fair solution keeps the parity gap: ' + good.constraints[0].value.toFixed(3));
  t.ok(!bad.reached && bad.score >= y.target && !bad.constraints[0].ok && bad.medal === null, 'a higher F1 with a broken constraint is refused: F1 ' + bad.score.toFixed(3) + ', gap ' + bad.constraints[0].value.toFixed(3));
  const pr = bad.constraints[0].rates;
  t.ok(Math.abs(Math.abs(pr[0].rate - pr[1].rate) - bad.constraints[0].value) < 1e-12, 'the gap is the difference of the two groups’ flag rates');
}
// the same as a function the worker answers
t.eq(WF.workerFns.scoreHidden(s, { data: x.data, hidden: x.hidden, metric: x.metric }).score, r.score, 'the worker function gives the same score');

// limits
const bad = WF.clone(sol); bad.blocks.push({ id: 'k', type: 'knn', params: WF.defaults('knn'), x: 0, y: 0 });
t.eq(C.limitsProblems(bad, x).length, 0, 'knn is allowed here');
const bad2 = WF.clone(sol); bad2.blocks.push({ id: 'g', type: 'gboost', params: WF.defaults('gboost'), x: 0, y: 0 });
t.ok(C.limitsProblems(bad2, x).some((p) => /not one of the blocks/.test(p)), 'a block outside the list is refused');
const bad3 = WF.clone(sol); bad3.blocks.find((b) => b.type === 'challenge-data').params.seed = 99;
t.ok(C.limitsProblems(bad3, x).some((p) => /changed/.test(p)), 'the challenge data cannot be changed');
const big = WF.clone(sol); for (let i = 0; i < 4; i++) big.blocks.push({ id: 'e' + i, type: 'evaluate', params: {}, x: 0, y: 0 });
t.ok(C.limitsProblems(big, Object.assign({}, x, { limits: Object.assign({}, x.limits, { maxBlocks: 7 }) })).some((p) => /blocks; the limit/.test(p)), 'too many blocks');
const sp = S.Content.challenges.get('op-ch-moons');
const long = WF.fromJSON(sp.solution); long.blocks.find((b) => b.type === 'train').params.epochs = 50;
t.ok(C.limitsProblems(long, sp).some((p) => /epochs/.test(p)), 'too many epochs');
t.ok(C.limitSentences(sp).some((l) => /20 epochs/.test(l)), 'the limits in words');
// the size of a network: hidden neurons in all, each hidden layer, and weights and biases
{
  const ne = S.Content.challenges.get('ne-ch-spiral'), d = WF.fromJSON(ne.solution), m = d.blocks.find((b) => b.type === 'mlp');
  t.eq(C.limitsProblems(d, ne), [], 'the twelve-neuron solution keeps to the limits');
  t.ok(C.limitSentences(ne).some((l) => /12 hidden neurons/.test(l)), 'the neuron limit in words');
  m.params.hidden = [10, 10];
  t.ok(C.limitsProblems(d, ne).some((p) => /20 hidden neurons; the limit is 12/.test(p)), 'two layers of ten break a limit of twelve neurons');
  const per = Object.assign({}, ne, { limits: { params: { mlp: { hidden: [1, 8] } } } });
  t.eq(C.limitsProblems(d, per).filter((p) => /layer \d of the hidden layers is 10, above the limit of 8/.test(p)).length, 2, 'a cap on the hidden layers checks every layer of a deep MLP');
  m.params.hidden = [6, 6];
  t.eq(C.limitsProblems(d, per).filter((p) => /hidden layers/.test(p)).length, 0, 'layers within the cap pass');
  const cn = S.Content.challenges.get('cn-ch-digits'), cd = WF.fromJSON(cn.solution);
  t.eq(C.limitsProblems(cd, cn), [], 'the small CNN keeps under 5,000 weights');
  cd.blocks.find((b) => b.type === 'cnn').params.hidden = 64;
  t.ok(C.limitsProblems(cd, cn).some((p) => /weights and biases; the limit is 5,000/.test(p)), 'a wider dense layer breaks the weight limit');
  t.ok(C.limitSentences(cn).some((l) => /5,000 weights and biases/.test(l)), 'the weight limit in words');
}
// hidden.n: 0 scores everything held out, and the limits say how many that is
t.ok(/on 180 hidden rows/.test(C.limitSentences({ data: { set: 'opinions', n: 300, seed: 5 }, hidden: { seed: 55, n: 0 } }).pop()), 'n: 0 on a pool: the held-out count, not the public one');
t.eq(C.hiddenCount({ data: { set: 'opinions', n: 300, seed: 5 }, hidden: { seed: 55, n: 0 } }), WF.hiddenSet({ set: 'opinions', n: 300, seed: 5 }, { seed: 55, n: 0 }).y.length, 'the count is the hidden set’s');
// medals and directions
const ring = S.Content.challenges.get('li-ch-ring');
t.eq(C.medal(ring, ring.target), 'bronze', 'bronze at the target');
t.eq(C.medal(ring, ring.marks.gold), 'gold', 'gold');
t.eq(C.medal(ring, ring.target - 0.05), null, 'no medal below the target');
const harvest = S.Content.challenges.get('li-ch-harvest');
t.ok(C.reaches(harvest, harvest.target - 0.01) && !C.reaches(harvest, harvest.target + 0.05), 'an error is to be at most the target');
t.eq(C.medal(harvest, harvest.marks.gold - 0.001), 'gold', 'a lower error earns more');
// evaluate puts it together
const ev = C.evaluate(x.solution, x);
t.ok(ev.ok && ev.medal, 'the solution reaches the target');
t.ok(!C.evaluate(x.wrong[0], x).ok, 'a wrong workflow does not');
const lint = C.lint(x);
t.eq(lint.problems, [], 'the challenge passes its own check');
// a workflow with no Train block
const lone = C.starter(x);
const r0 = C.evaluate(lone, x);
t.ok(r0.score === null && /no Train block/.test(r0.problems.join(' ')), 'nothing to score: ' + r0.problems.join(' '));

// exercise kinds
const mx = { kind: 'matrix', rows: ['a', 'b'], cols: ['a', 'b'], answer: [[3, 1], [2, 4]], total: 10 };
t.ok(K.answer(mx, [['3', '1'], ['2', '4']]).ok, 'matrix right');
const m2 = K.answer(mx, [['3', '1'], ['4', '2']]);
t.ok(!m2.ok && m2.cells['1,0'] === false && m2.cells['0,0'] === true, 'matrix marks cells');
t.ok(/still empty/.test(K.answer(mx, [['3', ''], ['2', '4']]).details.join(' ')), 'matrix empty cells');
const at = { kind: 'matrix', rows: ['q'], cols: ['k1', 'k2'], answer: [[0.25, 0.75]], tolerance: 0.01, rowsSumTo: 1 };
t.ok(/add up to 1/.test(K.answer(at, [['0.5', '0.7']]).details.join(' ')), 'attention rows must add up');
t.ok(K.answer(at, [['0.255', '0.745']]).ok, 'within the tolerance');
const pt = { kind: 'point', mode: 'xy', plot: { xr: [0, 4], yr: [0, 4] }, answer: [1, 2], tolerance: 0.2 };
t.ok(K.answer(pt, { x: 1.1, y: 2.05 }).ok, 'a click near enough');
t.ok(/too high/.test(K.answer(pt, { x: 1, y: 3 }).message), 'says which way: ' + K.answer(pt, { x: 1, y: 3 }).message);
t.ok(!K.answer(pt, null).ok, 'no click yet');
t.ok(K.answer({ kind: 'point', mode: 'x', plot: { xr: [0, 4], yr: [0, 1] }, answer: 2.5, tolerance: 0.1 }, { x: 2.45 }).ok, 'a threshold');
t.ok(K.answer({ kind: 'point', mode: 'pick', plot: { xr: [0, 4], yr: [0, 4], points: [[1, 1], [2, 2]] }, answer: 1 }, { i: 1 }).ok, 'a pick');
const pc = { kind: 'predict-curve', plot: { xr: [0, 10], yr: [0, 10] }, answer: [[0, 0], [10, 10]], tolerance: 0.5, checks: ['rising'] };
t.ok(K.answer(pc, [[0, 0.2], [5, 5.1], [10, 9.8]]).ok, 'a close sketch');
const far = K.answer(pc, [[0, 3], [5, 8], [10, 13]]);
t.ok(!far.ok && /too high/.test(far.details.join(' ')), 'too high is said: ' + far.details.join(' '));
t.ok(/whole range/.test(K.answer(pc, [[0, 0], [3, 3]]).message), 'a short sketch');
t.ok(/rise/.test(K.answer(pc, [[0, 10], [10, 0]]).message), 'the shape check');
const wx = S.Content.exercises.get('tr-x-wf-moons');
t.ok(K.answer(wx, wx.solution).ok, 'a workflow exercise solution passes');
t.ok(/short of the target/.test(K.answer(wx, wx.wrong[0]).message), 'a workflow that misses says so');
t.ok(/Build a workflow/.test(K.answer(wx, null).message), 'an empty workflow');
for (const k of ['matrix', 'point', 'predict-curve', 'workflow']) t.ok(K.kindNames.includes(k), k + ' is documented');
t.done();
