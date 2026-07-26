// The core blocks of a workflow, the ones every pipeline needs: a built-in dataset, a train/test split, a model
// (logistic regression; the other families are in wf-b-models*.js), Train and Evaluate. They set the pattern the
// other block files follow: params described for the dialog, run or train, a view for the viewer, a summary.
(function (S) {
  'use strict';
  const WF = S.WF;
  const SKIP = new Set(['graph', 'grid']);
  /** the datasets a Data block offers, in groups: [[id, label]] */
  WF.datasetOptions = function () {
    const D = S.Data;
    if (!D || !D.list) return [['moons', 'Two moons']];
    const order = ['toy', 'tabular', 'classic', 'image', 'text'], name = { toy: '2-D', tabular: 'table', classic: 'classic', image: 'images', text: 'text' };
    return D.list().filter((d) => !SKIP.has(d.kind) && d.id !== 'stories').sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind)).map((d) => [d.id, d.title + ' (' + (name[d.kind] || d.kind) + ')']);
  };
  WF.kindOfSet = (id) => { const d = S.Data && S.Data.doc && S.Data.doc(id); return !d ? 'table' : d.kind === 'image' ? 'images' : d.kind === 'text' ? 'text' : 'table'; };
  /** load a built-in set as the value on a wire (n: 0 for the set's own size) */
  WF.loadSet = function (id, opts) {
    const o = Object.assign({}, opts || {});
    for (const k of Object.keys(o)) if (o[k] === 0 || o[k] === '' || o[k] === null || o[k] === undefined) delete o[k];
    return WF.fromDataset(S.Data.load(id, o), { set: id, opts: o });
  };

  WF.block('data', {
    cat: 'Data', label: 'Dataset', icon: 'drive', order: 10,
    blurb: 'A built-in dataset: 2-D sets for seeing boundaries, tables with stories of their own, classic open sets, small images and short texts.',
    outputs: (p) => [{ id: 'out', label: 'data', type: WF.kindOfSet(p.set) }],
    params: [
      { id: 'set', label: 'Dataset', kind: 'choice', def: 'moons', options: () => WF.datasetOptions() },
      { id: 'n', label: 'Rows (0: the set’s own size)', kind: 'int', def: 0, min: 0, max: 20000 },
      { id: 'noise', label: 'Noise (2-D sets)', kind: 'number', def: 0, min: 0, max: 2, step: 0.01, show: (p) => S.Data && S.Data.doc(p.set) && S.Data.doc(p.set).kind === 'toy', help: '0 keeps the set’s own noise' },
      { id: 'size', label: 'Image size (image sets)', kind: 'choice', def: 28, options: [[16, '16 × 16'], [28, '28 × 28']], show: (p) => S.Data && S.Data.doc(p.set) && S.Data.doc(p.set).kind === 'image' },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const p = ctx.params, img = WF.kindOfSet(p.set) === 'images';
      return { out: { out: WF.loadSet(p.set, Object.assign({ n: p.n, noise: p.noise, seed: p.seed }, img ? { size: +p.size || 28 } : {})) } };
    },
    view(r) {
      const v = r.out.out, D = S.Data, doc = D && D.doc(v.source.set);
      const head = v.kind === 'table' ? { features: v.features, rows: v.X.slice(0, 12), y: v.y.slice(0, 12) } : v.kind === 'text' ? { texts: v.texts.slice(0, 12), y: v.y.slice(0, 12) } : { images: v.X.slice(0, 24).map((px) => Array.from(px, (x) => Math.round(x * 15))), y: v.y.slice(0, 24), w: v.w, h: v.h };
      return { kind: v.kind, name: v.name, doc: doc ? { title: doc.title, story: doc.story, rule: doc.rule, licence: doc.licence, source: doc.source } : null, n: v.y.length, task: v.task, classes: v.classes, counts: WF.counts(v), target: v.target, head,
        // 2-D sets are drawn as a scatter: every point
        points: v.kind === 'table' && v.features.length === 2 ? { X: v.X, y: v.y } : null };
    },
    summary: (r) => WF.describe(r.out.out),
  });

  WF.block('split', {
    cat: 'Split', label: 'Train/test split', icon: 'shuffle', order: 10,
    blurb: 'Holds back part of the rows for testing: a model is judged on rows it never saw. Stratified keeps each class in the same share on both sides.',
    inputs: [{ id: 'in', label: 'data', type: 'data' }],
    outputs: [{ id: 'train', label: 'train', type: 'same:in' }, { id: 'test', label: 'test', type: 'same:in' }],
    params: [
      { id: 'test', label: 'Test share', kind: 'number', def: 0.25, min: 0.05, max: 0.9, step: 0.05 },
      { id: 'stratify', label: 'Stratified (classification)', kind: 'bool', def: true },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const v = ctx.in.in, p = ctx.params;
      const sp = S.Data.split(v.y, p.test, p.seed, !!p.stratify && v.task === 'classification');
      return { out: { train: WF.subset(v, sp.train, v.name + ' (train)'), test: WF.subset(v, sp.test, v.name + ' (test)') }, split: sp };
    },
    view(r) { const tr = r.out.train, te = r.out.test; return { n: tr.y.length + te.y.length, train: tr.y.length, test: te.y.length, task: tr.task, classes: tr.classes, countsTrain: WF.counts(tr), countsTest: WF.counts(te), testRows: r.split.test }; },
    summary: (r) => r.out.train.y.length + ' train · ' + r.out.test.y.length + ' test',
  });

  // ---- models: a Models block only describes the model; Train fits it -----------------------------------------------
  /**
   * WF.model(type, {label, blurb, family, task: 'classification' | 'regression' | 'both', params, make(params, data, rng)
   *   -> {fit(X, y), predict(X), predictProba?(X)} | iterative: true and a trainer in wf-b-train, internals(est, data)})
   * registers a Models block whose output is the model's description, and remembers how to fit it.
   */
  WF.families = WF.families || {};
  // every classifier gets a decision threshold (two classes, a model with probabilities: class 1 when its
  // probability is at least this); these also get class weights, which their engines take as scikit-learn does
  WF.CLASS_WEIGHTED = new Set(['logistic', 'tree', 'forest', 'svm']);
  const SHARED = {
    classWeight: { id: 'classWeight', label: 'Class weights (classification)', kind: 'choice', def: 'none', options: [['none', 'none: every row counts the same'], ['balanced', 'balanced: rare classes count more'], ['custom', 'custom: a weight for each class']], help: 'balanced gives class k the weight n / (classes × rows of k)' },
    classWeights: { id: 'classWeights', label: 'Custom weights, one per class in order', kind: 'text', def: '1, 1', show: (p) => p.classWeight === 'custom', help: 'for example 1, 5: a row of the second class counts five times' },
    threshold: { id: 'threshold', label: 'Decision threshold (two classes)', kind: 'number', def: 0.5, min: 0.01, max: 0.99, step: 0.01, help: 'class 1 when its probability is at least this; the hidden-set score uses it too' },
  };
  WF.model = function (type, m) {
    WF.families[type] = m;
    const params = (m.params || []).slice();
    if (m.task !== 'regression' && WF.CLASS_WEIGHTED.has(type) && !params.some((q) => q.id === 'classWeight')) params.push(SHARED.classWeight, SHARED.classWeights);
    if (m.task !== 'regression' && !params.some((q) => q.id === 'threshold')) params.push(SHARED.threshold);
    return WF.block(type, {
      cat: 'Models', label: m.label, blurb: m.blurb, icon: m.icon || 'kit', order: m.order || 50,
      outputs: [{ id: 'model', label: 'model', type: 'model' }],
      params,
      run(ctx) { return { out: { model: { kind: 'model', fitted: false, family: type, label: m.label, params: ctx.params, task: m.task, iterative: !!m.iterative, takes: m.takes || ['table'] } } }; },
      view(r) { return { family: type, label: m.label, params: r.out.model.params, untrained: true }; },
      summary: () => 'untrained',
    });
  };
  WF.model('logistic', {
    label: 'Logistic regression', task: 'classification', order: 10, icon: 'curve',
    blurb: 'A straight boundary (in its features) and a probability from the sigmoid of the score; softmax for more than two classes. C is how little the weights are held back.',
    params: [
      { id: 'C', label: 'C (inverse regularisation)', kind: 'number', def: 1, min: 0.001, max: 1000, log: true },
      { id: 'solver', label: 'Solver', kind: 'choice', def: 'newton', options: [['newton', 'Newton (exact)'], ['gd', 'Gradient descent (traced)']] },
    ],
    make: (p) => new S.ML.LogisticRegression({ C: p.C, solver: p.solver, maxIter: 200 }),
    internals(est, data) {
      const W = est.coef || est.W || null, b = est.intercept !== undefined ? est.intercept : est.b;
      return { coef: W ? (Array.isArray(W[0]) || ArrayBuffer.isView(W[0]) ? W.map((r) => Array.from(r)) : [Array.from(W)]) : null, intercept: b === undefined ? null : [].concat(Array.isArray(b) || ArrayBuffer.isView(b) ? Array.from(b) : b), features: data.features.map((f) => f.name) };
    },
  });

  // ---- train -----------------------------------------------------------------------------------------------------
  /**
   * Fits a model on data. A classical model is fitted in one step; an iterative one (a network) gets a trainer from
   * WF.trainers[family](ctx, spec, data) -> {epochs, epoch, batch, step(), done(), current(), progress(), snapshot()}.
   */
  WF.trainers = WF.trainers || {};
  WF.block('train', {
    cat: 'Train', label: 'Train', icon: 'run', order: 10,
    blurb: 'Fits the model to the training rows. Networks learn epoch by epoch: play, pause, or step one epoch or one batch, and watch the curves.',
    inputs: [{ id: 'model', label: 'model', type: 'model' }, { id: 'data', label: 'data', type: 'data' }],
    outputs: [{ id: 'model', label: 'trained model', type: 'model' }],
    params: [
      { id: 'epochs', label: 'Epochs (networks)', kind: 'int', def: 30, min: 1, max: 2000 },
      { id: 'batch', label: 'Batch size (networks)', kind: 'int', def: 32, min: 1, max: 4096 },
      { id: 'optimizer', label: 'Optimiser (networks)', kind: 'choice', def: 'adam', options: [['sgd', 'SGD'], ['momentum', 'Momentum'], ['rmsprop', 'RMSProp'], ['adam', 'Adam']] },
      { id: 'lr', label: 'Learning rate (networks)', kind: 'number', def: 0.01, min: 0.00001, max: 5, log: true },
      { id: 'valShare', label: 'Validation share (networks)', kind: 'number', def: 0.2, min: 0, max: 0.5, step: 0.05 },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    train(ctx) {
      const spec = ctx.in.model, data = ctx.in.data;
      if (!spec || spec.kind !== 'model') throw new Error('wire a model from the Models blocks into Train');
      if (spec.fitted) throw new Error('this model is trained already: wire the Models block itself into Train');
      if (!data.task) throw new Error('this data has no target to learn');
      if (spec.task && spec.task !== 'both' && spec.task !== data.task) throw new Error(spec.label + ' is for ' + spec.task + ', but this data asks for ' + data.task);
      if (spec.takes && !spec.takes.includes(data.kind)) throw new Error(spec.label + ' learns from ' + spec.takes.join(' or ') + ', not ' + data.kind + (data.kind === 'text' ? ': turn the texts into numbers with a Vectorise block, or use a sequence model' : ''));
      if (WF.hasMissing(data)) throw new Error('some cells are missing: clean or impute them before training');
      const fam = WF.families[spec.family];
      if (spec.iterative) {
        const make = WF.trainers[spec.family];
        if (!make) throw new Error('the trainer for ' + spec.label + ' is not in this build yet');
        return WF.thresholded(make(ctx, spec, data, fam), spec);
      }
      return WF.thresholded(WF.oneStep(() => {
        const t0 = Date.now();
        const est = fam.make(spec.params, data, ctx.rng, ctx);
        const cw = WF.classWeightOf(spec, data);
        if (cw) est.params.classWeight = cw;
        est.fit(data.X, data.y);
        const proba = typeof est.predictProba === 'function' && data.task === 'classification' ? (X) => est.predictProba(X) : null;
        const model = WF.fitted({ family: spec.family, label: spec.label, params: spec.params, est, predict: (X) => est.predict(X), proba, ms: Date.now() - t0 }, data);
        model.internals = fam.internals ? fam.internals(est, data) : null;
        return model;
      }, data), spec);
    },
    view(r, ctx) {
      const m = r.out.model, fam = WF.families[m.family];
      const train = WF.predict(m, ctx.in.data);
      const v = { family: m.family, label: m.label, params: m.params, threshold: m.threshold, task: m.task, classes: m.classes, features: m.features.map((f) => f.name), internals: m.internals || null, curve: r.curve || null, trainScores: WF.scores(train), ms: m.ms, epoch: r.epoch, epochs: r.epochs };
      if (fam && fam.view) Object.assign(v, fam.view(m, ctx.in.data, ctx));
      return v;
    },
    summary(r, v) { const s = v && v.trainScores; const k = s && (s.accuracy !== undefined ? 'accuracy' : 'r2'); return (r.epochs ? 'epoch ' + r.epoch + ' · ' : '') + (k ? 'train ' + WF.METRICS[k].label + ' ' + WF.fmt(s[k], 3) : 'trained') + (r.out && r.out.model && r.out.model.threshold !== undefined ? ' · threshold ' + r.out.model.threshold : ''); },
  });
  /** the class weights a model's params ask for, as its engine takes them (null for none, or for a regression) */
  WF.classWeightOf = function (spec, data) {
    const p = spec.params || {};
    if (data.task !== 'classification' || !WF.CLASS_WEIGHTED.has(spec.family) || !p.classWeight || p.classWeight === 'none') return null;
    if (p.classWeight === 'balanced') return 'balanced';
    const w = String(p.classWeights || '').split(/[,;\s]+/).filter(Boolean).map(Number);
    if (w.length !== data.classes.length || !w.every((v) => Number.isFinite(v) && v > 0)) throw new Error('custom class weights: give ' + data.classes.length + ' positive numbers, one for each class (' + data.classes.join(', ') + ')');
    return w;
  };
  /** class 1 when its probability is at least t, for a classifier of two classes with probabilities */
  WF.withThreshold = function (m, t) {
    if (!m || !m.fitted || m.task !== 'classification' || !m.proba || (m.classes || []).length !== 2) return m;
    t = +t;
    if (!Number.isFinite(t) || Math.abs(t - 0.5) < 1e-12) return m;
    if (!m._predict0) m._predict0 = m.predict;
    m.threshold = t;
    m.predict = (X) => m.proba(X).map((r) => (r[1] >= t ? 1 : 0));
    return m;
  };
  /** a trainer whose models predict with the model block's threshold */
  WF.thresholded = function (trainer, spec) {
    const t = spec.params && spec.params.threshold;
    if (t === undefined || t === null || Math.abs(+t - 0.5) < 1e-12) return trainer;
    const cur = trainer.current.bind(trainer);
    trainer.current = () => { const o = cur(); if (o && o.out && o.out.model) WF.withThreshold(o.out.model, t); return o; };
    return trainer;
  };
  /** a trainer that does all its work in one step (a classical model's fit) */
  WF.oneStep = function (fit, data) {
    let model = null;
    return { epochs: 0, epoch: 0, batch: 0, step() { model = fit(); this.epoch = 1; return {}; }, done: () => !!model, progress: () => (model ? 1 : 0), current: () => ({ out: { model } }), snapshot: null, data };
  };

  // ---- evaluate ------------------------------------------------------------------------------------------------
  WF.block('evaluate', {
    cat: 'Evaluate', label: 'Evaluate', icon: 'check', order: 10,
    blurb: 'Runs a trained model on rows it did not learn from and compares what it says with the truth: the confusion matrix (click a cell to see its examples) or, for numbers, the errors.',
    inputs: [{ id: 'model', label: 'trained model', type: 'model' }, { id: 'data', label: 'test data', type: 'data' }],
    outputs: [{ id: 'pred', label: 'predictions', type: 'predictions' }],
    params: [{ id: 'normalize', label: 'Show', kind: 'choice', def: 'counts', options: [['counts', 'counts'], ['true', 'shares of each true class'], ['pred', 'shares of each prediction']] }],
    run(ctx) { return { out: { pred: WF.predict(ctx.in.model, ctx.in.data) } }; },
    view(r, ctx) {
      const p = r.out.pred, v = { task: p.task, classes: p.classes, scores: WF.scores(p), n: p.y.length, model: p.model, data: p.data, normalize: ctx.params.normalize };
      if (p.task === 'classification') {
        const cm = S.Metrics.confusion(p.y, p.pred, { labels: p.classes.map((_, i) => i) });
        v.counts = cm.counts;
        // a few examples of every cell, to show when the cell is clicked
        v.cells = cm.cells.map((row) => row.map((ks) => ks.slice(0, 24).map((k) => WF.example(p, k))));
        v.cellTotals = cm.cells.map((row) => row.map((ks) => ks.length));
      } else {
        v.points = p.y.map((y, i) => [y, p.pred[i]]);
      }
      return v;
    },
    summary(r) { const s = WF.scores(r.out.pred); return s.accuracy !== undefined ? 'accuracy ' + WF.fmt(s.accuracy, 3) : 'R² ' + WF.fmt(s.r2, 3); },
  });
  /** one example of predictions as plain JSON: its row (or image, or text), truth, prediction and probabilities */
  WF.example = function (p, k) {
    const e = { k, row: p.rows[k], y: p.y[k], pred: p.pred[k] };
    if (p.proba) e.proba = p.proba[k].map((x) => +x.toFixed(4));
    if (p.X) e.x = Array.from(p.X[k], (x) => +(+x).toPrecision(5));
    if (p.images) e.img = Array.from(p.images[k], (x) => Math.round(x * 15));
    if (p.texts) e.text = p.texts[k];
    return e;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
