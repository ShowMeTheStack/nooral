// The numbers the model pictures are drawn from, worked out where the model lives (the workflow's worker): what a
// trained model says at every place of the plane when the data has two features (a probability, a class, an SVM's
// decision value), its curve when it has one, the boundaries of a forest's first trees and of boosting stage by
// stage, a k-NN neighbourhood, and the ridge and lasso paths. Also the questions a viewer may ask the Train block
// later: what the model says for one row, and which training rows are a point's nearest neighbours.
(function (S) {
  'use strict';
  const WF = S.WF, ML = () => S.ML;
  const { r4, arr } = WF.mr;
  const MV = (WF.mv = WF.mv || {});
  const N = 56; // the plane is sampled at (N + 1) × (N + 1) corners

  /** up to max rows, spread evenly, as [indexes] */
  MV.sample = (n, max) => { if (n <= max) return Array.from({ length: n }, (_, i) => i); const out = []; for (let k = 0; k < max; k++) out.push(Math.floor((k * n) / max)); return out; };
  /** the range of column j, widened by a margin */
  const range = (X, j, pad) => { let lo = Infinity, hi = -Infinity; for (const r of X) { if (r[j] < lo) lo = r[j]; if (r[j] > hi) hi = r[j]; } const m = (hi - lo || 1) * (pad === undefined ? 0.08 : pad); return [r4(lo - m), r4(hi + m)]; };
  MV.range = range;
  /** the corners of the plane, rows from the top: [[u, v]..] */
  MV.corners = (xr, yr, n) => { const pts = []; for (let j = 0; j <= n; j++) { const v = yr[1] - ((yr[1] - yr[0]) * j) / n; for (let i = 0; i <= n; i++) pts.push([xr[0] + ((xr[1] - xr[0]) * i) / n, v]); } return pts; };
  /** the points of the data for a plot (at most 400) */
  MV.points = (data) => { const idx = MV.sample(data.X.length, 400); return { idx, X: idx.map((i) => arr(data.X[i])), y: idx.map((i) => (typeof data.y[i] === 'number' ? r4(data.y[i]) : data.y[i])) }; };

  /**
   * what a model says over the plane: {xr, yr, n, scheme: 'prob' | 'classes' | 'value', vals, vals2, score, levels, points}
   * score (when given) is the function whose level lines are the boundary: p − 0.5 for two classes, f for an SVM
   */
  MV.plane = function (m, data, opts) {
    opts = opts || {};
    if (data.kind !== 'table' || data.features.length !== 2) return null;
    const xr = range(data.X, 0), yr = range(data.X, 1), n = opts.n || N, pts = MV.corners(xr, yr, n);
    const out = { xr, yr, n, names: data.features.map((f) => f.name), points: MV.points(data) };
    if (m.task === 'classification') {
      const K = (m.classes || []).length;
      const P = m.proba ? m.proba(pts) : null;
      if (P && K === 2) { out.scheme = 'prob'; out.vals = P.map((p) => r4(p[1])); out.score = out.vals.map((p) => r4(p - 0.5)); out.levels = [0]; }
      else {
        const pred = m.predict(pts);
        out.scheme = 'classes';
        out.vals = Array.from(pred, (c) => +c);
        out.vals2 = P ? P.map((p) => r4(Math.max(...p))) : out.vals.map(() => 1);
        if (K === 2) { out.score = out.vals.map((c) => (c ? 0.5 : -0.5)); out.levels = [0]; }
      }
    } else {
      out.scheme = 'value';
      out.vals = Array.from(m.predict(pts), r4);
      let lo = Infinity, hi = -Infinity; for (const v of data.y) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      out.vr = [r4(lo), r4(hi)];
    }
    return out;
  };
  /** a regression model with one feature: its curve over the data's range */
  MV.line = function (m, data) {
    if (data.kind !== 'table' || data.features.length !== 1 || m.task !== 'regression') return null;
    const xr = range(data.X, 0, 0.04), xs = Array.from({ length: 161 }, (_, i) => xr[0] + ((xr[1] - xr[0]) * i) / 160);
    const ys = Array.from(m.predict(xs.map((x) => [x])), r4);
    let lo = Infinity, hi = -Infinity; for (const v of data.y.concat(ys)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo || 1) * 0.08;
    return { xr, yr: [r4(lo - pad), r4(hi + pad)], xs: xs.map(r4), ys, name: data.features[0].name, target: data.target ? data.target.name : 'y', points: MV.points(data) };
  };
  /** the usual picture: the plane or the curve */
  MV.basic = (m, data) => ({ plane: MV.plane(m, data), line: MV.line(m, data) });

  // ---- what each family adds ------------------------------------------------------------------------------------
  const fam = (id) => WF.families[id] || {};
  const leafValue = (T, x) => { const id = ML().treeApply(T, x); const node = T.nodes[id]; return node.value; };
  for (const id of ['linreg', 'naivebayes', 'lda', 'qda', 'tree', 'adaboost', 'logistic']) fam(id).view = (m, data) => MV.basic(m, data);

  fam('perceptron').view = function (m, data) {
    const v = MV.basic(m, data);
    // the boundary from the score itself, so it is a straight line and not the edge of a coloured grid
    if (v.plane && m.classes.length === 2 && m.est.decisionFunction) { const f = m.est.decisionFunction(MV.corners(v.plane.xr, v.plane.yr, v.plane.n)); v.plane.score = Array.from(f, (x) => r4(Array.isArray(x) || ArrayBuffer.isView(x) ? x[0] : x)); v.plane.levels = [0]; }
    return v;
  };
  fam('svm').view = function (m, data) {
    const v = MV.basic(m, data);
    if (v.plane && m.est.machines && m.est.machines.length === 1) {
      const pts = MV.corners(v.plane.xr, v.plane.yr, v.plane.n);
      v.plane.score = Array.from(m.est.decisionFunction(pts), (f) => r4(Array.isArray(f) || ArrayBuffer.isView(f) ? f[0] : f));
      v.plane.levels = [-1, 0, 1];
      v.plane.f = true;
      const pos = new Map(v.plane.points.idx.map((i, k) => [i, k]));
      v.plane.sv = (m.internals.alpha || []).filter(([i]) => pos.has(i)).map(([i, a]) => [pos.get(i), a]);
    }
    return v;
  };
  fam('svr').view = function (m, data) {
    const v = MV.basic(m, data);
    if (v.line) { v.line.eps = m.internals.epsilon; const sv = new Set(m.internals.support); v.line.sv = v.line.points.idx.map((i, k) => (sv.has(i) ? k : -1)).filter((k) => k >= 0); }
    return v;
  };
  fam('knn').view = function (m, data) {
    const v = MV.basic(m, data);
    if (v.plane) {
      // one example query: the training point nearest the middle of the plane, nudged off it
      const q = [r4((v.plane.xr[0] + v.plane.xr[1]) / 2), r4((v.plane.yr[0] + v.plane.yr[1]) / 2)];
      v.plane.query = MV.neighbours(m, data, q);
    }
    return v;
  };
  fam('forest').view = function (m, data) {
    const v = MV.basic(m, data);
    if (v.plane && m.task === 'classification' && m.classes.length === 2) {
      const n = 28, pts = MV.corners(v.plane.xr, v.plane.yr, n);
      v.trees = m.est.trees.slice(0, 3).map((T) => ({ n, vals: pts.map((x) => { const val = leafValue(T, x); return r4(Array.isArray(val) || ArrayBuffer.isView(val) ? val[1] / (val[0] + val[1] || 1) : val); }) }));
    }
    return v;
  };
  fam('gboost').view = function (m, data) {
    const v = MV.basic(m, data), est = m.est, nS = est.stages.length;
    const at = [...new Set([1, 2, 5, 15, nS].filter((k) => k <= nS))];
    const rawAt = (X, k) => X.map((x) => { let r = est.init[0]; for (let s = 0; s < k; s++) { const val = leafValue(est.stages[s][0], x); r += est.params.learningRate * (Array.isArray(val) || ArrayBuffer.isView(val) ? val[0] : val); } return r; });
    if (est.K === 1 && v.plane) {
      const n = 28, pts = MV.corners(v.plane.xr, v.plane.yr, n);
      v.stages = at.map((k) => ({ k, n, vals: rawAt(pts, k).map((r) => r4(m.task === 'classification' ? 1 / (1 + Math.exp(-r)) : r)) }));
    }
    if (est.K === 1 && v.line) v.stages = at.map((k) => ({ k, ys: rawAt(v.line.xs.map((x) => [x]), k).map(r4) }));
    return v;
  };
  /** the regularisation path: every coefficient as α runs from strong to weak */
  const path = (kind) => function (m, data) {
    const v = MV.basic(m, data);
    try {
      if (kind === 'lasso') { const p = ML().enetPath(data.X, data.y, { l1Ratio: 1, nAlphas: 40 }); v.path = { alphas: arr(p.alphas), coefs: p.coefs.map(arr), at: r4(m.params.alpha) }; }
      else {
        const a0 = Math.max(1e-4, m.params.alpha), alphas = Array.from({ length: 40 }, (_, k) => a0 * Math.pow(10, 4 - (8 * k) / 39));
        v.path = { alphas: alphas.map(r4), coefs: alphas.map((a) => { const e = ML().create('Ridge', { alpha: a }); e.fit(data.X, data.y); return arr(e.coef); }), at: r4(m.params.alpha) };
      }
    } catch (e) { v.path = null; }
    return v;
  };
  fam('ridge').view = path('ridge');
  fam('lasso').view = path('lasso');

  /** the k training rows nearest x: {x, idx (in the plotted points), all: [row indexes], dist, pred} */
  MV.neighbours = function (m, data, x) {
    const k = Math.min(m.est.params.k, data.X.length);
    const d = data.X.map((r, i) => [i, Math.hypot(...r.map((v, j) => v - x[j]))]).sort((a, b) => a[1] - b[1]).slice(0, k);
    return { x: arr(x), all: d.map((e) => e[0]), dist: d.map((e) => r4(e[1])), y: d.map((e) => data.y[e[0]]), pred: m.predict([x])[0], proba: m.proba ? arr(m.proba([x])[0]) : null };
  };

  // ---- questions for the Train block --------------------------------------------------------------------------
  const train = WF.def('train');
  if (train) {
    train.methods = Object.assign({}, train.methods, {
      /** what the trained model says for one row of the features it learned from: {pred, proba} */
      predictRow(r, a) { const m = r.out.model; return { pred: m.predict([a.x])[0], proba: m.proba ? arr(m.proba([a.x])[0]) : null }; },
      /** k-NN: the neighbours of a point the reader clicked */
      neighbours(r, a, ctx) { const m = r.out.model; if (m.family !== 'knn') throw new Error('only k-NN has neighbours'); return MV.neighbours(m, ctx.in.data, a.x); },
    });
  }
})(globalThis.Nooral = globalThis.Nooral || {});
