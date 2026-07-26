// The evaluation blocks after Evaluate: ROC and PR curves with a threshold, calibration, residuals for numbers, and
// learning and validation curves. Each keeps in its view only what its picture needs (the scores, rounded), so the
// view crosses from the worker quickly; the threshold sums (WF.EV) are shared with the page, which redraws the
// confusion matrix at any threshold without asking the worker.
(function (S) {
  'use strict';
  const WF = S.WF, M = () => S.Metrics;
  const EV = (WF.EV = WF.EV || {});
  const r4 = (x) => Math.round(x * 1e4) / 1e4;
  const MAXROWS = 3000;

  /** the rows a view keeps of predictions: all of them up to a cap (a seeded sample beyond it) */
  EV.keep = function (n, seed) {
    if (n <= MAXROWS) return Array.from({ length: n }, (_, i) => i);
    const r = S.ML.rng(seed || 7), idx = r.perm(n).slice(0, MAXROWS);
    return idx.sort((a, b) => a - b);
  };
  /** what a curve view carries: truth, the class probabilities (rounded) and the class names */
  EV.probView = function (p) {
    if (p.task !== 'classification') throw new Error('these curves are for classification; for numbers use Residuals');
    if (!p.proba) throw new Error(p.model + ' gives labels only, no probabilities, so there is no threshold to move');
    const idx = EV.keep(p.y.length);
    return { classes: p.classes, y: idx.map((i) => p.y[i]), proba: idx.map((i) => p.proba[i].map(r4)), n: p.y.length, kept: idx.length, model: p.model, data: p.data };
  };
  /** class k against the rest: {y: 0/1, score} */
  EV.binary = (v, k) => ({ y: v.y.map((c) => (c === k ? 1 : 0)), score: v.proba.map((r) => r[k]) });
  /** the ROC and PR curves of class k */
  EV.curves = function (v, k) {
    const b = EV.binary(v, k), roc = M().roc(b.y, b.score, { posLabel: 1 }), pr = M().prCurve(b.y, b.score, { posLabel: 1 });
    return { roc: { fpr: roc.fpr, tpr: roc.tpr, thresholds: roc.thresholds, auc: roc.auc, P: roc.P, N: roc.N }, pr: { precision: pr.precision, recall: pr.recall, thresholds: pr.thresholds, ap: pr.ap } };
  };
  /** the confusion counts of class k against the rest when "positive" means score ≥ t */
  EV.at = function (v, k, t) { const b = EV.binary(v, k); return M().atThreshold(b.y, b.score, t, 1); };
  /** the default positive class: 1 of two, else the first */
  EV.posClass = (classes) => (classes && classes.length === 2 ? 1 : 0);

  WF.block('roc', {
    cat: 'Evaluate', label: 'ROC and PR curves', icon: 'curve', order: 20,
    blurb: 'How the true-positive rate and the precision trade against the false alarms as the threshold moves. Slide the threshold: the point moves along both curves and the confusion matrix beside them is counted again.',
    inputs: [{ id: 'pred', label: 'predictions', type: 'predictions' }],
    outputs: [],
    params: [
      { id: 'threshold', label: 'Threshold', kind: 'number', def: 0.5, min: 0, max: 1, step: 0.01 },
      { id: 'cls', label: 'Positive class (index; −1: the usual one)', kind: 'int', def: -1, min: -1, max: 50 },
    ],
    run(ctx) { return { pred: ctx.in.pred }; },
    view(r, ctx) {
      const v = EV.probView(r.pred), k = ctx.params.cls >= 0 && ctx.params.cls < v.classes.length ? ctx.params.cls : EV.posClass(v.classes);
      const aucs = v.classes.map((_, c) => { try { return EV.curves(v, c).roc.auc; } catch (e) { return NaN; } });
      return Object.assign(v, { cls: k, threshold: ctx.params.threshold, aucs, ap: EV.curves(v, k).pr.ap });
    },
    summary: (r, v) => (v ? 'AUC ' + WF.fmt(v.aucs[v.cls], 3) + ' · AP ' + WF.fmt(v.ap, 3) : ''),
  });

  WF.block('calibration', {
    cat: 'Evaluate', label: 'Calibration', icon: 'curve', order: 30,
    blurb: 'Does a predicted 70% come true 70% of the time? The scores are put in bins and each bin’s mean prediction is set against how often it was really the class.',
    inputs: [{ id: 'pred', label: 'predictions', type: 'predictions' }],
    outputs: [],
    params: [
      { id: 'bins', label: 'Bins', kind: 'int', def: 10, min: 2, max: 30 },
      { id: 'strategy', label: 'Bins by', kind: 'choice', def: 'uniform', options: [['uniform', 'equal width'], ['quantile', 'equal counts']] },
      { id: 'cls', label: 'Class (index; −1: the usual one)', kind: 'int', def: -1, min: -1, max: 50 },
    ],
    run(ctx) { return { pred: ctx.in.pred }; },
    view(r, ctx) {
      const p = r.pred, v = EV.probView(p), k = ctx.params.cls >= 0 && ctx.params.cls < v.classes.length ? ctx.params.cls : EV.posClass(v.classes);
      const b = EV.binary(v, k), cal = M().calibration(b.y, b.score, { nBins: ctx.params.bins, strategy: ctx.params.strategy, posLabel: 1 });
      // the histogram of scores always in 20 even bins, split by the true class
      const hist = [new Array(20).fill(0), new Array(20).fill(0)];
      b.score.forEach((s, i) => { hist[b.y[i]][Math.min(19, Math.floor(s * 20))]++; });
      return { classes: v.classes, cls: k, n: v.n, model: v.model, bins: cal.bins.map((x) => ({ lo: x.lo, hi: x.hi, count: x.count, positives: x.positives, meanProb: x.meanProb, fraction: x.fraction })),
        ece: cal.ece, brier: M().brier(b.y, b.score, { posLabel: 1 }).value, logloss: M().logLoss(p.y, p.proba, { labels: p.classes.map((_, i) => i) }).value, hist, strategy: ctx.params.strategy };
    },
    summary: (r, v) => (v ? 'Brier ' + WF.fmt(v.brier, 3) + ' · ECE ' + WF.fmt(v.ece, 3) : ''),
  });

  WF.block('residuals', {
    cat: 'Evaluate', label: 'Residuals', icon: 'curve', order: 40,
    blurb: 'For a number to predict: each error (truth − prediction) against the prediction and against each feature, and how the errors spread. A good model leaves a shapeless band around zero.',
    inputs: [{ id: 'pred', label: 'predictions', type: 'predictions' }],
    outputs: [],
    params: [{ id: 'feature', label: 'Feature to plot against (index)', kind: 'int', def: 0, min: 0, max: 500 }],
    run(ctx) {
      if (ctx.in.pred.task !== 'regression') throw new Error('residuals are for a number to predict; this model predicts classes');
      return { pred: ctx.in.pred };
    },
    view(r, ctx) {
      const p = r.pred, idx = EV.keep(p.y.length), j = Math.min(ctx.params.feature, Math.max(0, (p.features || []).length - 1));
      const res = M().residuals(p.y, p.pred);
      return { n: p.y.length, model: p.model, data: p.data, target: p.target || null, y: idx.map((i) => r4(p.y[i])), pred: idx.map((i) => r4(p.pred[i])),
        features: (p.features || []).map((f) => f.name), feature: j, fx: p.X ? idx.map((i) => r4(p.X[i][j])) : null,
        mae: M().mae(p.y, p.pred).value, rmse: M().rmse(p.y, p.pred).value, r2: M().r2(p.y, p.pred).value, mean: res.mean !== undefined ? res.mean : M().mean(res.residuals || p.y.map((y, i) => y - p.pred[i])) };
    },
    summary: (r, v) => (v ? 'RMSE ' + WF.fmt(v.rmse, 3) + ' · R² ' + WF.fmt(v.r2, 3) : ''),
  });

  // ---- learning and validation curves ----------------------------------------------------------------------------
  /** the numeric parameters of a model family that a validation curve can sweep */
  EV.sweepable = (family) => ((WF.def(family) && WF.def(family).params) || []).filter((p) => (p.kind === 'number' || p.kind === 'int') && p.min !== undefined && p.max !== undefined && p.id !== 'seed');
  EV.sweepValues = function (p, n) {
    const lo = p.kind === 'int' ? Math.max(1, p.min) : p.min, hi = p.max, out = [];
    if (p.log || (lo > 0 && hi / lo >= 100)) { const a = Math.log10(Math.max(lo, 1e-6)), b = Math.log10(hi); for (let i = 0; i < n; i++) out.push(+Math.pow(10, a + ((b - a) * i) / (n - 1)).toPrecision(3)); }
    else for (let i = 0; i < n; i++) out.push(+(lo + ((hi - lo) * i) / (n - 1)).toPrecision(4));
    const vals = p.kind === 'int' ? [...new Set(out.map((v) => Math.max(lo, Math.round(v))))] : out;
    return vals;
  };
  /** fit a fresh model of the spec on rows tr and score it on rows te and on its own rows */
  EV.fitScore = function (spec, data, tr, te, metric, over, rng) {
    const fam = WF.families[spec.family], sub = WF.subset(data, tr), test = WF.subset(data, te);
    const est = fam.make(Object.assign({}, spec.params, over || {}), sub, rng);
    est.fit(sub.X, sub.y);
    const model = WF.fitted({ family: spec.family, label: spec.label, params: spec.params, est, predict: (X) => est.predict(X), proba: typeof est.predictProba === 'function' && data.task === 'classification' ? (X) => est.predictProba(X) : null }, sub);
    return { train: WF.score(metric, WF.predict(model, sub)), val: WF.score(metric, WF.predict(model, test)) };
  };
  const meanSd = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length; return [m, Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length)]; };
  WF.block('learningcurve', {
    cat: 'Evaluate', label: 'Learning curve', icon: 'curve', order: 50,
    blurb: 'Trains the model again and again: on more and more rows (the learning curve) and over a range of one setting (the validation curve). Where the two scores sit and how far apart tells underfitting from overfitting.',
    inputs: [{ id: 'model', label: 'model (untrained)', type: 'model' }, { id: 'data', label: 'data', type: 'data' }],
    outputs: [],
    params: [
      { id: 'steps', label: 'Training sizes', kind: 'int', def: 8, min: 3, max: 15 },
      { id: 'repeats', label: 'Repeats (for the bands)', kind: 'int', def: 3, min: 1, max: 8 },
      { id: 'val', label: 'Validation share', kind: 'number', def: 0.3, min: 0.1, max: 0.5, step: 0.05 },
      { id: 'sweep', label: 'Setting to sweep (empty: none)', kind: 'text', def: '' },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const spec = ctx.in.model, data = ctx.in.data, p = ctx.params;
      if (!spec || spec.kind !== 'model' || spec.fitted) throw new Error('wire a Models block (an untrained model) into the learning curve');
      if (spec.iterative) throw new Error(spec.label + ' learns epoch by epoch: the Train block draws its own curves');
      if (!data.task) throw new Error('this data has no target to learn');
      if (WF.hasMissing(data)) throw new Error('some cells are missing: clean or impute them first');
      const metric = data.task === 'classification' ? 'accuracy' : 'r2', n = data.y.length;
      const sizes = [], learn = [];
      for (let rep = 0; rep < p.repeats; rep++) {
        const sp = S.Data.split(data.y, p.val, p.seed + rep, data.task === 'classification');
        const order = S.ML.rng(p.seed * 31 + rep).shuffle(sp.train.slice());
        for (let s = 0; s < p.steps; s++) {
          const frac = 0.1 + (0.9 * s) / (p.steps - 1), m = Math.max(data.task === 'classification' ? data.classes.length * 2 : 3, Math.round(order.length * frac));
          let tr = order.slice(0, m);
          if (data.task === 'classification' && new Set(tr.map((i) => data.y[i])).size < 2) tr = order.slice(0, Math.min(order.length, m + 10));
          let sc;
          try { sc = EV.fitScore(spec, data, tr, sp.test, metric, null, S.ML.rng(p.seed + s)); } catch (e) { sc = { train: NaN, val: NaN }; }
          if (!rep) { sizes.push(tr.length); learn.push({ train: [], val: [] }); }
          learn[s].train.push(sc.train); learn[s].val.push(sc.val);
          ctx.progress && ctx.progress((rep * p.steps + s + 1) / (p.repeats * p.steps), 'size ' + tr.length);
        }
      }
      let sweep = null;
      const sp = EV.sweepable(spec.family).find((x) => x.id === String(p.sweep).trim()) || null;
      if (sp) {
        const values = EV.sweepValues(sp, 9), split = S.Data.split(data.y, p.val, p.seed, data.task === 'classification'), pts = [];
        for (const v of values) { let sc; try { sc = EV.fitScore(spec, data, split.train, split.test, metric, { [sp.id]: v }, S.ML.rng(p.seed)); } catch (e) { sc = { train: NaN, val: NaN }; } pts.push({ v, train: sc.train, val: sc.val }); }
        sweep = { param: sp.id, label: sp.label, log: !!(sp.log || (sp.min > 0 && sp.max / sp.min >= 100)), pts };
      }
      return { metric, n, sizes, learn: learn.map((l) => ({ train: meanSd(l.train.filter(Number.isFinite)), val: meanSd(l.val.filter(Number.isFinite)) })), sweep, label: spec.label, sweepable: EV.sweepable(spec.family).map((x) => [x.id, x.label]) };
    },
    view(r) {
      const last = r.learn[r.learn.length - 1], gap = last.train[0] - last.val[0];
      const say = gap > 0.06 ? 'over' : last.val[0] < (r.metric === 'accuracy' ? 0.75 : 0.5) ? 'under' : 'fit';
      return { metric: r.metric, metricLabel: WF.METRICS[r.metric].label, sizes: r.sizes, learn: r.learn, sweep: r.sweep, label: r.label, sweepable: r.sweepable, gap, verdict: say, final: last.val[0] };
    },
    summary: (r, v) => (v ? 'validation ' + WF.METRICS[v.metric].label + ' ' + WF.fmt(v.final, 3) : ''),
  });

  // ---- Evaluate gets the scores worked out from its counts, per class -----------------------------------------------
  const ev = WF.def('evaluate');
  if (ev && !ev.__more) {
    const base = ev.view;
    ev.__more = true;
    ev.view = function (r, ctx) {
      const v = base.call(this, r, ctx), p = r.out.pred;
      if (p.task === 'classification') {
        const per = v.classes.map((_, k) => {
          const tp = v.counts[k][k], fp = v.counts.reduce((s, row, i) => s + (i === k ? 0 : row[k]), 0), fn = v.counts[k].reduce((s, c, j) => s + (j === k ? 0 : c), 0);
          const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0;
          return { tp, fp, fn, precision: prec, recall: rec, f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0, support: tp + fn };
        });
        v.perClass = per;
        v.mcc = M().mcc(p.y, p.pred).value;
        v.features = (p.features || []).map((f) => f.name);
        v.kindOfRows = p.images ? 'images' : p.texts ? 'text' : 'table';
        v.w = p.w; v.h = p.h;
      } else {
        const idx = EV.keep(p.y.length);
        v.points = idx.map((i) => [r4(p.y[i]), r4(p.pred[i])]);
        v.mae = M().mae(p.y, p.pred).value; v.rmse = M().rmse(p.y, p.pred).value; v.r2 = M().r2(p.y, p.pred).value;
      }
      return v;
    };
  }
})(globalThis.Nooral = globalThis.Nooral || {});
