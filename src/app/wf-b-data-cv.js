// Cross-validation: the rows dealt into k folds, a fresh copy of the model trained on k − 1 of them and tested on the
// one left out, k times, so every row is predicted once by a model that never saw it. The out-of-fold predictions
// go on to Evaluate-type blocks; the view shows the folds and the score of each.
(function (S) {
  'use strict';
  const WF = S.WF, U = () => WF.dutil;

  /** fit a model description on a table, the way Train would (networks with at most maxEpochs epochs) */
  WF.fitQuick = function (spec, data, seed, opts) {
    opts = opts || {};
    const fam = WF.families[spec.family];
    if (!fam) throw new Error('there is no model family ' + spec.family);
    if (spec.task && spec.task !== 'both' && spec.task !== data.task) throw new Error(spec.label + ' is for ' + spec.task + ', but this data asks for ' + data.task);
    if (spec.iterative) {
      const make = WF.trainers[spec.family];
      if (!make) throw new Error('the trainer for ' + spec.label + ' is not in this build yet');
      const params = Object.assign(WF.defaults('train'), opts.train || {}, { seed });
      params.epochs = Math.min(params.epochs, opts.maxEpochs || 30);
      const ctx = { id: 'cv', params, in: { model: spec, data }, seed, rng: S.ML.rng(seed), progress() {} };
      const tr = make(ctx, spec, data, fam);
      let guard = 0;
      while (!tr.done() && guard++ < 1e7) tr.step();
      return tr.current().out.model;
    }
    const est = fam.make(spec.params, data, S.ML.rng(seed), { seed });
    est.fit(data.X, data.y);
    const proba = typeof est.predictProba === 'function' && data.task === 'classification' ? (X) => est.predictProba(X) : null;
    return WF.fitted({ family: spec.family, label: spec.label, params: spec.params, est, predict: (X) => est.predict(X), proba }, data);
  };

  WF.block('cv', {
    cat: 'Split', label: 'Cross-validation', icon: 'refresh', order: 20,
    blurb: 'Deals the rows into k folds and trains k copies of the model, each tested on the fold it did not see. Every row gets one honest prediction, and the spread of the k scores shows how much a single split could mislead.',
    inputs: [{ id: 'model', label: 'model', type: 'model' }, { id: 'data', label: 'data', type: 'data' }],
    outputs: [{ id: 'pred', label: 'out-of-fold predictions', type: 'predictions' }],
    params: [
      { id: 'k', label: 'Folds (k)', kind: 'int', def: 5, min: 2, max: 20 },
      { id: 'stratify', label: 'Stratified (classification)', kind: 'bool', def: true },
      { id: 'shuffle', label: 'Shuffle before dealing', kind: 'bool', def: true },
      { id: 'maxEpochs', label: 'Epochs per fold (networks)', kind: 'int', def: 20, min: 1, max: 500 },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const spec = ctx.in.model, v = ctx.in.data, p = ctx.params, ML = S.ML;
      if (!spec || spec.kind !== 'model') throw new Error('wire a model from the Models blocks into Cross-validation');
      if (spec.fitted) throw new Error('cross-validation trains its own copies: wire the Models block itself, not a trained model');
      if (!v.task) throw new Error('this data has no target to learn');
      if (WF.hasMissing(v)) throw new Error('some cells are missing: clean or impute them first');
      const n = v.y.length, k = Math.max(2, Math.min(p.k, n));
      const folds = p.stratify && v.task === 'classification' ? ML.stratifiedKFold(v.y, k, { shuffle: !!p.shuffle, seed: p.seed }) : ML.kFold(n, k, { shuffle: !!p.shuffle, seed: p.seed });
      const pred = new Array(n), proba = new Array(n), fold = new Array(n).fill(-1), scores = [];
      let hasProba = v.task === 'classification';
      folds.forEach((f, i) => {
        ctx.progress(i / k, 'fold ' + (i + 1) + ' of ' + k);
        const tr = WF.subset(v, f.train), te = WF.subset(v, f.test);
        const m = WF.fitQuick(spec, tr, p.seed + i, { maxEpochs: p.maxEpochs });
        const pr = WF.predict(m, te);
        pr.y.forEach((_, t) => { const row = f.test[t]; pred[row] = pr.pred[t]; fold[row] = i; if (pr.proba) proba[row] = pr.proba[t]; else hasProba = false; });
        const sc = WF.scores(pr);
        scores.push({ fold: i, train: f.train.length, test: f.test.length, scores: sc });
      });
      const out = { kind: 'predictions', task: v.task, y: v.y.slice(), pred, proba: hasProba ? proba : null, classes: v.classes, rows: v.rows.slice(), X: v.kind === 'table' ? v.X : null, features: v.features,
        images: v.kind === 'images' ? v.X : null, w: v.w, h: v.h, texts: v.texts || null, model: spec.label + ' (' + k + '-fold)', data: v.name, folds: fold };
      return { out: { pred: out }, folds, scores, k };
    },
    view(r, ctx) {
      const u = U(), v = ctx.in.data, main = r.out.pred.task === 'regression' ? 'r2' : 'accuracy';
      const vals = r.scores.map((s) => s.scores[main]), m = vals.reduce((a, b) => a + b, 0) / vals.length, sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, vals.length - 1));
      const keep = u.sample(v.y.length, 400), fold = r.out.pred.folds;
      const o = { k: r.k, n: v.y.length, metric: main, metricLabel: WF.METRICS[main].label, perFold: r.scores.map((s) => ({ fold: s.fold, train: s.train, test: s.test, score: u.r(s.scores[main], 4), all: Object.fromEntries(Object.entries(s.scores).map(([a, b]) => [a, u.r(b, 4)])) })),
        mean: u.r(m, 4), sd: u.r(sd, 4), overall: Object.fromEntries(Object.entries(WF.scores(r.out.pred)).map(([a, b]) => [a, u.r(b, 4)])), model: ctx.in.model.label, task: v.task, classes: v.classes,
        rows: keep.map((i) => [fold[i], v.y[i]]), counts: r.folds.map((f) => (v.task === 'classification' ? v.classes.map((_, c) => f.test.filter((i) => v.y[i] === c).length) : [f.test.length])) };
      if (v.kind === 'table' && v.features.length <= 40) { const pl = u.plane(keep.map((i) => v.X[i]), v.features.map((f) => f.name)); o.points = { P: u.round2(pl.P), y: keep.map((i) => v.y[i]), fold: keep.map((i) => fold[i]), names: pl.names, how: pl.how }; }
      return o;
    },
    summary(r) { const main = r.out.pred.task === 'regression' ? 'r2' : 'accuracy', vals = r.scores.map((s) => s.scores[main]), m = vals.reduce((a, b) => a + b, 0) / vals.length; return r.k + ' folds · ' + WF.METRICS[main].label + ' ' + WF.fmt(m, 3); },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
