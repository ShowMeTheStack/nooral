// Predict and the first explanations. Predict is an inference panel: the student types a row (or clicks a point, draws
// a digit, writes a sentence) and the trained model answers with its probabilities and the reason in brief. Permutation
// importance and partial dependence explain a model as a whole, in the features it learned from.
(function (S) {
  'use strict';
  const WF = S.WF;
  const EX = (WF.EX = WF.EX || {});
  const r4 = (x) => Math.round(x * 1e4) / 1e4;

  /** the kind of input a model takes from a person: 'table', 'images' or 'text' */
  EX.rawKind = (model, data) => (model.inKind === 'images' ? 'images' : model.inKind === 'text' || (data && data.texts) ? 'text' : 'table');
  /**
   * the raw rows behind a table (before its pipe), with the raw features: {features, X, y}. When the table came
   * through steps, the built-in set it came from is loaded again and its rows picked out.
   */
  EX.rawOf = function (model, data) {
    if (data && data.kind === 'table' && !data.pipe.length) return { features: data.features, X: data.X, y: data.y };
    if (data && data.source && data.source.set) {
      try { const src = WF.loadSet(data.source.set, data.source.opts); if (src.kind === 'table') return { features: src.features, X: data.rows.map((i) => src.X[i]), y: data.rows.map((i) => src.y[i]) }; } catch (e) { /* not reloadable */ }
    }
    if (data && data.raw && data.raw.features) return { features: data.raw.features, X: null, y: data.y };
    if (!model.pipe || !model.pipe.length) return { features: model.features, X: null, y: null };
    return { features: null, X: null, y: null };
  };
  const range = (col) => { const s = col.filter(Number.isFinite).sort((a, b) => a - b); if (!s.length) return [0, 1, 0]; const q = (t) => s[Math.min(s.length - 1, Math.floor(t * (s.length - 1)))]; return [s[0], s[s.length - 1], q(0.5)]; };
  const niceStep = (lo, hi) => { const span = hi - lo || 1, p = Math.pow(10, Math.floor(Math.log10(span / 100))); return +p.toPrecision(1); };
  /** a model as a function of rows that gives one number each: the probability of class k, or the number predicted */
  EX.fnOf = function (model, k) {
    if (model.task === 'regression') return (X) => Array.from(model.predict(X));
    if (model.proba) return (X) => model.proba(X).map((r) => r[k]);
    return (X) => Array.from(model.predict(X), (c) => (c === k ? 1 : 0));
  };
  EX.classOf = (model, k) => (model.task !== 'classification' ? 0 : k >= 0 && k < model.classes.length ? k : model.classes.length === 2 ? 1 : 0);
  /** the table a model-explaining block works on: the model's own features */
  EX.need = function (model, data) {
    if (!model || !model.fitted) throw new Error('wire a trained model (from a Train block) into this block');
    if (!data) throw new Error('wire the data to explain on');
    if (data.kind !== 'table' || model.inKind !== 'table') throw new Error('this explanation works on tables of numbers; for images use Saliency');
    if (data.features.length !== model.features.length) throw new Error('the model learned from ' + model.features.length + ' features, this table has ' + data.features.length + ': prepare it the same way');
    if (WF.hasMissing(data)) throw new Error('some cells are missing: clean or impute them first');
  };

  // ---- predict ----------------------------------------------------------------------------------------------------
  WF.block('predict', {
    cat: 'Predict', label: 'Predict', icon: 'target', order: 10,
    blurb: 'Ask the trained model about a case of your own: type the values (or click a point, draw a digit, write a sentence) and see what it says, how sure it is, and why in brief.',
    inputs: [{ id: 'model', label: 'trained model', type: 'model' }, { id: 'data', label: 'data (for the ranges)', type: 'data', optional: true }],
    outputs: [],
    params: [],
    run(ctx) {
      const model = ctx.in.model, data = ctx.in.data || null;
      if (!model || !model.fitted) throw new Error('wire a trained model (from a Train block) into Predict');
      const kind = EX.rawKind(model, data), raw = kind === 'table' ? EX.rawOf(model, data) : { features: null, X: null, y: null };
      if (kind === 'table' && !raw.features) throw new Error('the model learned from prepared rows: wire the data into Predict too, so it knows the raw columns');
      const sample = raw.X ? raw.X.slice(0, 1500) : null;
      return { model, kind, raw: { features: raw.features, X: sample, y: raw.y ? raw.y.slice(0, 1500) : null }, data };
    },
    view(r) {
      const m = r.model, v = { kind: r.kind, task: m.task, classes: m.classes, label: m.label, w: m.w, h: m.h };
      if (r.kind === 'table') {
        v.features = r.raw.features.map((f, j) => {
          const o = { name: f.name, type: f.type, unit: f.unit || '', categories: f.categories || null };
          if (r.raw.X && f.type !== 'category') { const col = r.raw.X.map((x) => x[j]), [lo, hi, med] = range(col), whole = col.every((x) => !Number.isFinite(x) || Number.isInteger(x)); Object.assign(o, { min: lo, max: hi, start: med, step: whole ? 1 : niceStep(lo, hi) }); }
          else o.start = r.raw.X ? r.raw.X[0][j] : 0;
          return o;
        });
        const num = v.features.filter((f) => f.type !== 'category');
        if (v.features.length === 2 && num.length === 2 && r.raw.X) {
          // the 2-D plane in raw units: what the model says everywhere, and the rows it saw
          const n = 44, [x0, x1] = [v.features[0].min, v.features[0].max], [y0, y1] = [v.features[1].min, v.features[1].max];
          const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.06 || 1, xr = [x0 - px, x1 + px], yr = [y0 - py, y1 + py], pts = [];
          for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) pts.push([xr[0] + ((i + 0.5) / n) * (xr[1] - xr[0]), yr[1] - ((j + 0.5) / n) * (yr[1] - yr[0])]);
          const plane = { xr, yr, nx: n, ny: n };
          if (m.task === 'classification') { const P = m.proba ? m.probaRaw(pts) : null, L = P ? null : m.predictRaw(pts); plane.vals = pts.map((_, i) => (P ? P[i].indexOf(Math.max(...P[i])) : L[i])); plane.conf = pts.map((_, i) => (P ? r4(Math.max(...P[i])) : 1)); }
          else plane.vals = Array.from(m.predictRaw(pts)).map(r4);
          v.plane = plane;
          v.points = { X: r.raw.X.slice(0, 300).map((x) => x.map(r4)), y: r.raw.y ? r.raw.y.slice(0, 300) : null };
        }
      }
      return v;
    },
    methods: {
      /** {rows: [[raw values]]} | {images: [[pixels 0..1]]} | {texts: ['...']} -> [{pred, label, proba, why}] */
      predict(r, args) {
        const m = r.model, items = args.rows || args.images || args.texts || [];
        if (!items.length) return [];
        const input = args.images ? items.map((px) => Float32Array.from(px)) : items;
        const pred = Array.from(m.predictRaw(input)), P = m.proba ? m.probaRaw(input) : null;
        return items.map((x, i) => {
          const o = { pred: m.task === 'classification' ? pred[i] : r4(pred[i]), label: m.task === 'classification' ? m.classes[pred[i]] : String(r4(pred[i])), proba: P ? Array.from(P[i], r4) : null };
          if (r.kind === 'table') o.why = EX.why(r, x, o);
          return o;
        });
      },
    },
    summary: (r) => 'ask ' + r.model.label,
  });
  /** why, in brief: the contributions of a linear model on raw rows, else the nearest rows the model saw */
  EX.why = function (r, x, o) {
    const m = r.model, I = m.internals;
    if (I && I.coef && (!m.pipe || !m.pipe.length) && I.coef[0] && I.coef[0].length === x.length) {
      const k = m.task === 'classification' && I.coef.length > 1 ? o.pred : 0, w = I.coef[k], b = I.intercept ? I.intercept[k] || 0 : 0;
      return { kind: 'linear', parts: x.map((v, j) => ({ name: r.raw.features[j].name, value: v, weight: r4(w[j]), part: r4(w[j] * v) })), intercept: r4(b), cls: m.task === 'classification' && I.coef.length === 1 ? 1 : k };
    }
    if (!r.raw.X || !r.raw.X.length) return null;
    const F = r.raw.features, sd = F.map((_, j) => { const c = r.raw.X.map((z) => z[j]), mu = c.reduce((a, b) => a + b, 0) / c.length; return Math.sqrt(c.reduce((a, b) => a + (b - mu) ** 2, 0) / c.length) || 1; });
    const d = r.raw.X.map((z, i) => [i, z.reduce((s, v, j) => s + (F[j].type === 'category' ? (v === x[j] ? 0 : 1) : ((v - x[j]) / sd[j]) ** 2), 0)]).sort((a, b) => a[1] - b[1]).slice(0, 3);
    return { kind: 'near', rows: d.map(([i, dist]) => ({ x: r.raw.X[i].map(r4), y: r.raw.y ? r.raw.y[i] : null, dist: r4(Math.sqrt(dist)) })) };
  };

  // ---- permutation importance ------------------------------------------------------------------------------------------
  WF.block('importance', {
    cat: 'Explain', label: 'Permutation importance', icon: 'probe', order: 10,
    blurb: 'Shuffle one column and see how far the score falls: a column the model relies on costs a lot, one it ignores costs nothing. Repeats give the spread.',
    inputs: [{ id: 'model', label: 'trained model', type: 'model' }, { id: 'data', label: 'data (test rows)', type: 'data' }],
    outputs: [],
    params: [{ id: 'repeats', label: 'Repeats', kind: 'int', def: 5, min: 1, max: 30 }, { id: 'seed', label: 'Seed', kind: 'seed', def: 1 }],
    run(ctx) {
      const m = ctx.in.model, d = ctx.in.data;
      EX.need(m, d);
      const metric = m.task === 'classification' ? 'accuracy' : 'r2';
      const score = (y, p) => WF.score(metric, { y, pred: p, classes: m.classes, task: m.task });
      const res = S.Explain.permutationImportance({ predict: (X) => Array.from(m.predict(X)) }, d.X, d.y, { nRepeats: ctx.params.repeats, seed: ctx.params.seed, score, names: d.features.map((f) => f.name) });
      return { res, metric, features: d.features.map((f) => f.name), label: m.label };
    },
    view(r) {
      const x = r.res;
      return { features: r.features, metric: WF.METRICS[r.metric].label, baseline: x.baseline, mean: x.mean, std: x.std, order: x.order, drops: x.drops.map((a) => a.map(r4)), label: r.label, say: x.trace.map((t) => t.say).slice(0, 4) };
    },
    summary: (r) => { const j = r.res.order[0]; return 'top: ' + r.features[j]; },
  });

  // ---- partial dependence ------------------------------------------------------------------------------------------------
  WF.block('pdp', {
    cat: 'Explain', label: 'Partial dependence', icon: 'curve', order: 20,
    blurb: 'Set one feature to each value in turn in every row and average what the model says: the average curve is partial dependence, each row’s own curve (ICE) shows when the effect differs from row to row.',
    inputs: [{ id: 'model', label: 'trained model', type: 'model' }, { id: 'data', label: 'data', type: 'data' }],
    outputs: [],
    params: [
      { id: 'feature', label: 'Feature (index)', kind: 'int', def: 0, min: 0, max: 500 },
      { id: 'cls', label: 'Class (index; −1: the usual one)', kind: 'int', def: -1, min: -1, max: 50 },
      { id: 'rows', label: 'Rows to use', kind: 'int', def: 120, min: 10, max: 600 },
    ],
    run(ctx) {
      const m = ctx.in.model, d = ctx.in.data, p = ctx.params;
      EX.need(m, d);
      const j = Math.min(p.feature, d.features.length - 1), k = EX.classOf(m, p.cls);
      const X = d.X.slice(0, p.rows);
      const res = S.Explain.partialDependence(EX.fnOf(m, k), X, j, { gridSize: 24, names: d.features.map((f) => f.name) });
      return { res, j, k, features: d.features.map((f) => f.name), task: m.task, classes: m.classes, label: m.label, col: X.map((x) => x[j]) };
    },
    view(r) {
      const x = r.res, step = Math.max(1, Math.ceil(x.ice.length / 50));
      return { features: r.features, feature: r.j, cls: r.k, task: r.task, classes: r.classes, label: r.label, grid: x.grid.map(r4), average: x.average.map(r4), ice: x.ice.filter((_, i) => i % step === 0).map((row) => row.map(r4)), rug: r.col.map(r4).slice(0, 300), say: x.trace.map((t) => t.say) };
    },
    summary: (r) => 'on ' + r.features[r.j],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
