// What flows along the wires of a workflow, and the helpers every block shares (no page here: workers and node use it).
//   table        {kind: 'table', name, X: [[..]], y: [..], features: [{name, type: 'number' | 'category', categories}],
//                 target: {name, type}, task: 'classification' | 'regression' | null, classes: [names] | null,
//                 rows: [index in the source], pipe: [steps], raw: {features, target}, source: {set, opts} | null}
//                 y holds class ids 0..k−1 for classification. pipe is what was done to the source's rows to get X:
//                 [{name, say, apply(X) -> X}], so a trained model can take raw rows (a hidden test set, a row typed in).
//   images       the same with kind: 'images', X a list of pixel rows (Float32Array, w·h each, 0..1), w, h
//   text         the same with kind: 'text', texts: [..] beside X (empty until a block turns the texts into numbers)
//   model        {kind: 'model', fitted: false, family, label, params, task?, iterative} from a Models block, or after
//                Train: {kind: 'model', fitted: true, family, label, params, task, classes, features, pipe, inKind, est,
//                predict(X), proba(X) | null, predictRaw(X), internals}
//   predictions  {kind: 'predictions', task, y, pred, proba, classes, rows, X, features, images, texts, w, h, model, data}
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});

  /** classification or regression, from a target */
  WF.taskOf = (target, y) => (!target ? null : target.type === 'category' || target.type === 'class' ? 'classification' : target.type === 'number' ? 'regression' : y && y.every((v) => Number.isInteger(v)) && new Set(y).size <= 20 ? 'classification' : 'regression');

  /** a dataset of S.Data (or one read from a file) as the value on a wire */
  WF.fromDataset = function (ds, source) {
    const kind = ds.images || ds.doc && ds.doc.kind === 'image' ? 'images' : ds.texts || ds.doc && ds.doc.kind === 'text' ? 'text' : 'table';
    const target = ds.target || { name: 'label', type: 'category' };
    const task = WF.taskOf(target, ds.y);
    const classes = task === 'classification' ? (ds.classes || target.categories || [...new Set(ds.y)].sort((a, b) => a - b).map(String)).map(String) : null;
    const n = (ds.X || ds.images || ds.texts || []).length;
    const v = { kind, name: (ds.doc && ds.doc.title) || ds.name || ds.id || 'data', y: Array.from(ds.y || []), target, task, classes, rows: Array.from({ length: n }, (_, i) => i), pipe: [], source: source || null };
    if (kind === 'images') {
      const n0 = ds.n || (ds.y || []).length;
      const imgs = ds.get ? Array.from({ length: n0 }, (_, i) => ds.get(i)) : Array.isArray(ds.images) ? ds.images : ds.X;
      Object.assign(v, { X: imgs.map((p) => (p instanceof Float32Array ? p : Float32Array.from(p))), w: ds.w, h: ds.h, features: [] });
    } else if (kind === 'text') Object.assign(v, { texts: ds.texts.slice(), X: ds.X || [], features: ds.features || [] });
    else Object.assign(v, { X: ds.X.map((r) => Array.from(r)), features: (ds.features || []).map((f) => (typeof f === 'string' ? { name: f, type: 'number' } : Object.assign({}, f))) });
    v.raw = { features: v.features.map((f) => Object.assign({}, f)), target, classes };
    return v;
  };
  /** the rows idx of a table (or image or text set), keeping everything else */
  WF.subset = function (v, idx, name) {
    const o = Object.assign({}, v, { X: idx.map((i) => v.X[i]), y: idx.map((i) => v.y[i]), rows: idx.map((i) => v.rows[i]) });
    if (v.texts) o.texts = idx.map((i) => v.texts[i]);
    if (name) o.name = name;
    return o;
  };
  /** a table with new columns, one more step on its pipe (the step remembers how to do the same to new rows) */
  WF.transformed = function (v, X, features, step, extra) { return Object.assign({}, v, { X, features, pipe: v.pipe.concat([step]) }, extra || {}); };
  /** raw rows through a pipe */
  WF.applyPipe = (pipe, X) => pipe.reduce((acc, st) => st.apply(acc), X);
  /** the numbers of a table row, as a plain array */
  WF.row = (v, i) => Array.from(v.X[i]);
  WF.isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  WF.hasMissing = (v) => v.kind === 'table' && v.X.some((r) => r.some((x) => !WF.isNum(x)));
  WF.fmt = (x, d) => (S.ML && S.ML.fmt ? S.ML.fmt(x, d) : String(+(+x).toPrecision(d || 3)));
  /** the count of each class, in class order */
  WF.counts = (v) => { const c = (v.classes || []).map(() => 0); for (const y of v.y) if (c[y] !== undefined) c[y]++; return c; };
  /** a short description of what is on a wire */
  WF.describe = function (v) {
    if (!v) return '';
    if (v.kind === 'model') return v.fitted ? 'a trained ' + v.label : 'an untrained ' + v.label;
    if (v.kind === 'predictions') return v.y.length + ' predictions';
    const n = v.y.length, what = v.kind === 'images' ? 'images' : v.kind === 'text' ? 'texts' : 'rows';
    return n + ' ' + what + (v.kind === 'table' ? ' × ' + v.features.length + ' features' : '') + (v.task === 'classification' ? ' · ' + v.classes.length + ' classes' : v.task === 'regression' ? ' · a number to predict' : '');
  };

  /**
   * a trained model as the value on a wire. o: {family, label, params, est (anything), predict(X) -> labels | numbers,
   * proba(X) -> rows of class probabilities | null, internals}; data is the table it learned from
   */
  WF.fitted = function (o, data) {
    const m = Object.assign({ kind: 'model', fitted: true, task: data.task, classes: data.classes, features: data.features, pipe: data.pipe, inKind: data.kind, w: data.w, h: data.h, trainedOn: data.y.length }, o);
    if (!m.proba) m.proba = null;
    m.predictRaw = (X) => m.predict(WF.applyPipe(m.pipe, X));
    m.probaRaw = (X) => (m.proba ? m.proba(WF.applyPipe(m.pipe, X)) : null);
    return m;
  };
  /** a trained model's predictions on a table: the predictions value */
  WF.predict = function (model, data) {
    if (!model || !model.fitted) throw new Error('the model has not been trained: wire it through a Train block first');
    if (data.kind !== model.inKind && !(data.kind === 'text' && model.inKind === 'table')) throw new Error('the model learned from ' + model.inKind + ', but this is ' + data.kind);
    // steps the model's rows went through after the split and these rows did not: do the same to them
    const dp = data.pipe || [], mp = model.pipe || [];
    if (mp.length > dp.length && dp.every((st, i) => st === mp[i])) {
      const rest = mp.slice(dp.length);
      data = Object.assign({}, data, { X: WF.applyPipe(rest, data.kind === 'text' && data.texts && !(data.X && data.X.length) ? data.texts : data.X), features: model.features, pipe: mp, kind: model.inKind });
    }
    if (data.kind === 'table' && data.features.length !== model.features.length) throw new Error('the model learned from ' + model.features.length + ' features, but this table has ' + data.features.length + ': prepare it the same way');
    const rows = model.inKind === 'text' && data.texts ? data.texts : data.X;
    const pred = Array.from(model.predict(rows)), proba = model.proba ? model.proba(rows).map((r) => Array.from(r)) : null;
    return { kind: 'predictions', task: model.task, y: data.y.slice(), pred, proba, classes: data.classes, rows: data.rows.slice(), X: data.kind === 'table' ? data.X : null, features: data.features,
      images: data.kind === 'images' ? data.X : null, w: data.w, h: data.h, texts: data.texts || null, model: model.label, data: data.name };
  };

  // ---- scores --------------------------------------------------------------------------------------------------
  WF.METRICS = {
    accuracy: { label: 'accuracy', task: 'classification', higher: true },
    f1: { label: 'F1', task: 'classification', higher: true },
    precision: { label: 'precision', task: 'classification', higher: true },
    recall: { label: 'recall', task: 'classification', higher: true },
    balanced: { label: 'balanced accuracy', task: 'classification', higher: true },
    auc: { label: 'ROC AUC', task: 'classification', higher: true },
    logloss: { label: 'log loss', task: 'classification', higher: false },
    r2: { label: 'R²', task: 'regression', higher: true },
    mae: { label: 'MAE', task: 'regression', higher: false },
    rmse: { label: 'RMSE', task: 'regression', higher: false },
    mse: { label: 'MSE', task: 'regression', higher: false },
    ari: { label: 'adjusted Rand index', task: 'clustering', higher: true },
    silhouette: { label: 'silhouette', task: 'clustering', higher: true },
  };
  /** a score of predictions: F1, precision and recall are binary (class 1 positive) for two classes, macro for more */
  WF.score = function (metric, p) {
    const M = S.Metrics, k = (p.classes || []).length;
    const avg = k > 2 ? { average: 'macro' } : { average: 'binary', posLabel: 1 };
    switch (metric) {
      case 'accuracy': return M.accuracy(p.y, p.pred).value;
      case 'balanced': return M.balancedAccuracy(p.y, p.pred).value;
      case 'f1': return M.precisionRecallF1(p.y, p.pred, avg).f1;
      case 'precision': return M.precisionRecallF1(p.y, p.pred, avg).precision;
      case 'recall': return M.precisionRecallF1(p.y, p.pred, avg).recall;
      case 'auc': if (!p.proba) throw new Error('ROC AUC needs probabilities, and this model gives none'); return k > 2 ? M.rocAucOvr(p.y, p.proba).value : M.rocAuc(p.y, p.proba.map((r) => r[1])).value;
      case 'logloss': if (!p.proba) throw new Error('log loss needs probabilities, and this model gives none'); return M.logLoss(p.y, p.proba, { labels: p.classes.map((_, i) => i) }).value;
      case 'r2': return M.r2(p.y, p.pred).value;
      case 'mae': return M.mae(p.y, p.pred).value;
      case 'rmse': return M.rmse(p.y, p.pred).value;
      case 'mse': return M.mse(p.y, p.pred).value;
      // clusterings: p.labels the group of each row (-1 for noise), p.y the classes, p.X the rows as clustered
      case 'ari': if (!p.labels) throw new Error('the adjusted Rand index compares groups with classes: score a clustering block'); return M.adjustedRand(p.y, p.labels).value;
      case 'silhouette': {
        if (!p.labels || !p.X) throw new Error('the silhouette scores a clustering: score a clustering block');
        const ok = p.labels.map((l, i) => (l >= 0 ? i : -1)).filter((i) => i >= 0);
        if (new Set(ok.map((i) => p.labels[i])).size < 2) throw new Error('the silhouette needs at least two groups (noise aside)');
        return M.silhouette(ok.map((i) => p.X[i]), ok.map((i) => p.labels[i])).value;
      }
      default: throw new Error('unknown metric ' + metric);
    }
  };
  /** the usual scores of predictions, for a block's summary and the Compare table */
  WF.scores = function (p) {
    const names = p.task === 'regression' ? ['r2', 'mae', 'rmse'] : ['accuracy', 'f1', 'precision', 'recall'].concat(p.proba ? ['auc'] : []);
    const out = {};
    for (const m of names) { try { out[m] = WF.score(m, p); } catch (e) { /* not for this one */ } }
    return out;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
