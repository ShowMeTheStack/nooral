// Challenges and workflow exercises, the part that runs where the workflow runs (the worker, node): the locked data
// block that gives the public training rows of a challenge, and scoring a trained workflow on a hidden test set made
// from the same set with a seed of its own. Nothing here touches the page.
//   'challenge-data' block: params {set, opts, seed, n}: the rows the reader may learn from
//   WF.scoreHidden(session, {data: {set, opts, seed, n}, hidden: {seed, n}, metric}) -> {score, metric, n, model, problems}
//   WF.modelToScore(doc, session) -> the id of the Train block whose model is scored
(function (S) {
  'use strict';
  const WF = S.WF;

  WF.block('challenge-data', {
    cat: 'Data', label: 'Challenge data', icon: 'trophy', order: 90, locked: true,
    blurb: 'The training rows of this task. The rows the score is taken on are kept apart and never shown.',
    outputs: (p) => [{ id: 'out', label: 'data', type: WF.kindOfSet(p.set) }],
    params: [
      { id: 'set', label: 'Dataset', kind: 'text', def: 'moons', show: () => false },
      { id: 'opts', label: 'Options', kind: 'text', def: null, show: () => false },
      { id: 'n', label: 'Rows', kind: 'int', def: 0, show: () => false },
      { id: 'seed', label: 'Seed', kind: 'int', def: 1, show: () => false },
    ],
    run(ctx) {
      const p = ctx.params;
      return { out: { out: WF.publicSet({ set: p.set, opts: p.opts, n: p.n, seed: p.seed }) } };
    },
    view(r) {
      const v = r.out.out, doc = S.Data && S.Data.doc(v.source.set);
      return { kind: v.kind, name: v.name, n: v.y.length, task: v.task, classes: v.classes, counts: WF.counts(v), target: v.target,
        doc: doc ? { title: doc.title, story: doc.story, licence: doc.licence } : null,
        head: v.kind === 'table' ? { features: v.features, rows: v.X.slice(0, 12), y: v.y.slice(0, 12) } : v.kind === 'images' ? { images: v.X.slice(0, 24).map((px) => Array.from(px, (x) => Math.round(x * 15))), y: v.y.slice(0, 24), w: v.w, h: v.h } : { texts: (v.texts || []).slice(0, 12), y: v.y.slice(0, 12) },
        points: v.kind === 'table' && v.features.length === 2 ? { X: v.X, y: v.y } : null };
    },
    summary: (r) => WF.describe(r.out.out),
  });

  /** a challenge-data block made from a challenge's data spec */
  WF.challengeBlock = (data) => ['challenge-data', { set: data.set, opts: data.opts || null, n: data.n || 0, seed: data.seed || 1 }];

  /** the Train block whose model is scored: the one feeding an Evaluate block (the last such), else the last Train */
  WF.modelToScore = function (doc) {
    const order = WF.order(doc);
    const trains = order.filter((id) => WF.byId(doc, id).type === 'train');
    const fed = order.filter((id) => WF.byId(doc, id).type === 'evaluate').map((id) => (WF.feeds(doc, id).model || [])[0]).filter((id) => id && trains.includes(id));
    return fed.length ? fed[fed.length - 1] : trains.length ? trains[trains.length - 1] : null;
  };

  // Two kinds of set. A generated set (the 2-D sets, the tables with stories) draws fresh rows for every seed: the
  // hidden rows are simply drawn with a seed of their own. A fixed pool (the classic sets, the digits, the texts) has
  // the same rows whatever the seed: it is split once, by the set alone, into public rows (70%) and held-out rows
  // (30%); the public rows of every challenge on it come from the first part, its hidden rows from the second, so a
  // hidden row is never a training row.
  const HOLD = 0.3, pools = new Map();
  const rowKey = (v, i) => (v.texts ? v.texts[i] : JSON.stringify(Array.from(v.X[i], (x) => (typeof x === 'number' ? +x.toPrecision(10) : x)))) + '|' + v.y[i];
  /** is this set a fixed pool of rows (the same rows, maybe reordered, for every seed)? */
  WF.isPool = function (set, opts) {
    const k = set + JSON.stringify(opts || {});
    if (!pools.has(k)) {
      const a = WF.loadSet(set, Object.assign({}, opts || {}, { seed: 1 })), b = WF.loadSet(set, Object.assign({}, opts || {}, { seed: 2 }));
      const ka = a.y.map((_, i) => rowKey(a, i)).sort(), kb = b.y.map((_, i) => rowKey(b, i)).sort();
      pools.set(k, ka.length === kb.length && ka.every((x, i) => x === kb[i]));
    }
    return pools.get(k);
  };
  /** a fixed pool split into its public and held-out parts: {all, pub: [row indexes], hold: [row indexes]} */
  WF.poolParts = function (data) {
    const all = WF.loadSet(data.set, Object.assign({}, data.opts || {}, { seed: 1 }));
    const sp = S.Data.split(all.y, data.holdout || HOLD, 20261, all.task === 'classification');
    return { all, pub: sp.train, hold: sp.test };
  };
  /** n of the indexes, drawn by a seed (all of them when n is 0 or at least their number) */
  const take = (idx, n, seed) => { if (!n || n >= idx.length) return idx.slice(); const r = S.Data.rng(seed), a = idx.slice(); r.shuffle(a); return a.slice(0, n).sort((x, y) => x - y); };
  /** the rows a challenge's reader may learn from */
  WF.publicSet = function (data) {
    if (!WF.isPool(data.set, data.opts)) return WF.loadSet(data.set, Object.assign({}, data.opts || {}, { n: data.n, seed: data.seed }));
    const P = WF.poolParts(data);
    return WF.subset(P.all, take(P.pub, data.n, data.seed), P.all.name);
  };
  /** the hidden test set: fresh rows of a generated set (another seed), or the held-out rows of a fixed pool */
  WF.hiddenSet = function (data, hidden) {
    if (!WF.isPool(data.set, data.opts)) {
      // a set whose shape is drawn from the seed too (the centres of blobs) keeps the public rows' shape: only the
      // rows are fresh
      const o = Object.assign({}, data.opts || {});
      if (!o.centers) { const pub = S.Data.load(data.set, Object.assign({}, o, { n: data.n || undefined, seed: data.seed || 1 })); if (pub.centers) o.centers = pub.centers; }
      return WF.loadSet(data.set, Object.assign(o, { n: hidden.n || data.n || 0, seed: hidden.seed }));
    }
    const P = WF.poolParts(data);
    return WF.subset(P.all, take(P.hold, hidden.n, hidden.seed), P.all.name + ' (held out)');
  };

  WF.CLUSTERERS = ['kmeans', 'dbscan', 'hierarchical', 'gmm'];
  /** the clustering block whose groups are scored: the last one in the workflow */
  WF.clusterToScore = function (doc) { const ids = WF.order(doc).filter((id) => WF.CLUSTERERS.includes(WF.byId(doc, id).type)); return ids.length ? ids[ids.length - 1] : null; };
  /**
   * a clustering scored on hidden rows: each hidden row is placed the way the block places a new row (the steps
   * before it, then its nearest centre, core point or component), and the groups are compared with the classes
   * (adjusted Rand index) or with each other (silhouette, on the rows as the block saw them; noise set aside)
   */
  WF.scoreClusters = function (session, args) {
    const { data, hidden, metric } = args, doc = session.doc;
    const id = WF.clusterToScore(doc);
    if (!id) return { score: null, metric, problems: ['there is no clustering block to score: wire the challenge data into k-means, DBSCAN, hierarchical clustering or a Gaussian mixture'] };
    const r = session.res[id];
    if (!r || r.state === 'error') return { score: null, metric, problems: ['the clustering block could not run' + (r && r.error ? ': ' + r.error : '')] };
    const v = session.output(id, 'out');
    if (!v || !v.pipe || !v.pipe.length) return { score: null, metric, problems: ['the clustering block has no groups yet'] };
    const H = WF.hiddenSet(data, hidden);
    let Y;
    try { Y = WF.applyPipe(v.pipe, H.X); } catch (e) { return { score: null, metric, problems: ['the hidden rows could not be placed in groups: ' + (e.message || e)] }; }
    const cats = v.features[v.features.length - 1].categories || [];
    const labels = Y.map((row) => { const c = cats[row[row.length - 1]]; return c === undefined || c === 'noise' ? -1 : +c; });
    const P = { task: 'clustering', y: H.y, labels, X: Y.map((row) => row.slice(0, -1)), classes: H.classes };
    let score = null;
    const problems = [];
    try { score = WF.score(metric, P); } catch (e) { problems.push(e.message || String(e)); }
    return { score, metric, n: H.y.length, model: WF.label(WF.byId(doc, id)), block: id, problems, groups: new Set(labels.filter((l) => l >= 0)).size, noise: labels.filter((l) => l < 0).length };
  };

  WF.scoreHidden = function (session, args) {
    const { data, hidden, metric } = args;
    if (WF.METRICS[metric] && WF.METRICS[metric].task === 'clustering') return WF.withConstraints(session, args, WF.scoreClusters(session, args));
    return WF.withConstraints(session, args, scoreModel(session, args));
  };
  const scoreModel = function (session, args) {
    const { data, hidden, metric } = args;
    const doc = session.doc, problems = [];
    const id = WF.modelToScore(doc);
    if (!id) return { score: null, metric, problems: ['there is no Train block to score: wire a model and the challenge data into Train'] };
    const r = session.res[id];
    if (!r || r.state === 'error') return { score: null, metric, problems: ['the Train block could not run' + (r && r.error ? ': ' + r.error : '')] };
    const model = session.output(id, 'model');
    if (!model || !model.fitted) return { score: null, metric, problems: ['the Train block has no trained model yet' + (r && r.state === 'waiting' ? ': it ' + (r.words || 'waits for its inputs') : '')] };
    const H = WF.hiddenSet(data, hidden);
    if (H.kind !== model.inKind && !(H.kind === 'text' && model.inKind === 'table')) problems.push('the model learned from ' + model.inKind + ', but the task is about ' + H.kind);
    const raw = H.kind === 'text' ? H.texts : H.X;
    let pred, proba = null;
    try {
      pred = Array.from(model.predictRaw(raw));
      if (model.probaRaw) { const p = model.probaRaw(raw); proba = p ? p.map((x) => Array.from(x)) : null; }
    } catch (e) { return { score: null, metric, problems: problems.concat(['the model could not be run on new rows: ' + (e.message || e)]) }; }
    const P = { task: H.task, y: H.y, pred, proba, classes: H.classes };
    let score = null;
    try { score = WF.score(metric, P); } catch (e) { problems.push(e.message || String(e)); }
    const out = { score, metric, n: H.y.length, model: model.label, block: id, problems };
    // kept for the constraints, never sent back
    Object.defineProperty(out, 'hiddenRows', { value: { H, pred }, enumerable: false });
    return out;
  };

  // ---- constraints: a second condition the scored model must meet (a fairness gap on the hidden rows) ----------------
  WF.CONSTRAINTS = {
    parity: { label: 'demographic parity difference', say: 'the gap between the groups’ rates of positive predictions' },
    odds: { label: 'equalised odds difference', say: 'the larger of the groups’ gaps in true and false positive rates' },
  };
  /** a constraint in words: "demographic parity difference between the groups ≤ 0.1" */
  WF.constraintText = (c) => (WF.CONSTRAINTS[c.kind] || { label: c.kind }).label + (c.group === 'group' ? ' between the groups' : ' between the ' + c.group + ' groups') + ' ≤ ' + c.max;
  /** the constraints of a challenge measured on the scored model's hidden predictions: res.constraints [{kind, value, max, ok}] */
  WF.withConstraints = function (session, args, res) {
    const list = args.constraints || [];
    if (!list.length) return res;
    const rows = res.hiddenRows;
    res.constraints = list.map((c) => {
      const o = { kind: c.kind, group: c.group, max: c.max, label: WF.constraintText(c), value: null, ok: false };
      if (!rows) return o;
      const H = rows.H, j = H.features.findIndex((f) => f.name === c.group);
      if (j < 0) { res.problems.push('the hidden rows have no column ' + c.group); return o; }
      const g = H.X.map((r) => r[j]);
      try {
        const F = S.Metrics.fairness(H.y, rows.pred, g, { posLabel: c.pos === undefined ? 1 : c.pos });
        o.value = c.kind === 'odds' ? F.equalizedOdds : F.demographicParity;
        o.ok = o.value <= c.max + 1e-12;
        o.rates = F.groups.map((q) => ({ group: (H.features[j].categories || [])[q.group] || String(q.group), rate: q.selectionRate, tpr: q.tpr, fpr: q.fpr }));
      } catch (e) { res.problems.push('the constraint could not be measured: ' + (e.message || e)); }
      return o;
    });
    return res;
  };
  WF.workerFns.scoreHidden = (session, args) => WF.scoreHidden(session, args);
})(globalThis.Nooral = globalThis.Nooral || {});
