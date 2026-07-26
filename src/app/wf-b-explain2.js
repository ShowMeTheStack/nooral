// Explanations of single cases: SHAP-style additive contributions (exact for a few features, Tree SHAP for a single
// tree, Kernel SHAP otherwise) for one row and summed up over many, a LIME-style local straight-line surrogate, and
// for image models saliency (and Grad-CAM where the model can give its feature maps).
(function (S) {
  'use strict';
  const WF = S.WF, EX = WF.EX;
  const r4 = (x) => Math.round(x * 1e4) / 1e4;

  /** Shapley values of row x: {phi, base, fx, method} */
  EX.shap = function (m, k, x, background, budget) {
    const E = S.Explain, f = EX.fnOf(m, k), d = x.length;
    const tree = m.est && m.est.tree && m.est.tree.nodes && !m.est.trees ? m.est.tree : null;
    if (tree && m.task === 'classification' && m.proba) { const r = E.treeShap(E.treeFromNodes(tree, { output: k }), x); return { phi: r.phi, base: r.base, fx: r.fx, method: 'Tree SHAP' }; }
    if (tree && m.task === 'regression') { const r = E.treeShap(E.treeFromNodes(tree), x); return { phi: r.phi, base: r.base, fx: r.fx, method: 'Tree SHAP' }; }
    if (d <= 8) { const r = E.shapExact(f, x, background); return { phi: r.phi, base: r.base, fx: r.fx, method: 'exact Shapley values' }; }
    const r = E.kernelShap(f, x, background, { nSamples: budget || 512, seed: 1 });
    return { phi: r.phi, base: r.base, fx: r.fx, method: 'Kernel SHAP' };
  };
  const pick = (n, k, seed) => { if (n <= k) return Array.from({ length: n }, (_, i) => i); return S.ML.rng(seed || 3).perm(n).slice(0, k).sort((a, b) => a - b); };

  WF.block('shap', {
    cat: 'Explain', label: 'SHAP values', icon: 'probe', order: 30,
    blurb: 'Splits one prediction into a share for each feature, so the shares add up from the average prediction to this one; then the same over many rows, to see which features push which way.',
    inputs: [{ id: 'model', label: 'trained model', type: 'model' }, { id: 'data', label: 'data', type: 'data' }],
    outputs: [],
    params: [
      { id: 'row', label: 'Row to explain (index)', kind: 'int', def: 0, min: 0, max: 100000 },
      { id: 'cls', label: 'Class (index; −1: the usual one)', kind: 'int', def: -1, min: -1, max: 50 },
      { id: 'rows', label: 'Rows in the summary', kind: 'int', def: 30, min: 5, max: 80 },
    ],
    run(ctx) {
      const m = ctx.in.model, d = ctx.in.data, p = ctx.params;
      EX.need(m, d);
      const k = EX.classOf(m, p.cls), i = Math.min(p.row, d.y.length - 1);
      const bg = pick(d.y.length, 16, 5).map((j) => d.X[j]);
      const one = EX.shap(m, k, d.X[i], bg, 1024);
      const many = pick(d.y.length, p.rows, 9).map((j, t) => { ctx.progress && ctx.progress(t / p.rows, 'row ' + j); return { j, phi: EX.shap(m, k, d.X[j], bg, 256).phi }; });
      return { m, d, k, i, one, many };
    },
    view(r) {
      const d = r.d, F = d.features.map((f) => f.name), nF = F.length;
      const meanAbs = F.map((_, j) => r.many.reduce((s, x) => s + Math.abs(x.phi[j]), 0) / r.many.length);
      const lo = F.map((_, j) => Math.min(...d.X.map((x) => x[j]))), hi = F.map((_, j) => Math.max(...d.X.map((x) => x[j])));
      return { features: F, row: r.i, x: Array.from(d.X[r.i], r4), y: d.y[r.i], task: r.m.task, classes: r.m.classes, cls: r.k, label: r.m.label, method: r.one.method,
        phi: r.one.phi.map(r4), base: r4(r.one.base), fx: r4(r.one.fx), meanAbs: meanAbs.map(r4),
        // each summary point: the SHAP value and where the feature's value sits in its range (0 low, 1 high)
        summary: r.many.map((x) => ({ row: x.j, phi: x.phi.map(r4), pos: Array.from({ length: nF }, (_, j) => r4(hi[j] > lo[j] ? (d.X[x.j][j] - lo[j]) / (hi[j] - lo[j]) : 0.5)) })), n: d.y.length };
    },
    summary: (r) => 'row ' + r.i + ' · ' + r.one.method,
  });

  WF.block('lime', {
    cat: 'Explain', label: 'LIME', icon: 'probe', order: 40,
    blurb: 'Near one row the model is fitted by a straight line: points are drawn around the row, the model is asked about each, and nearer points count more. The line’s slopes say what matters here, and its fit says how far to trust them.',
    inputs: [{ id: 'model', label: 'trained model', type: 'model' }, { id: 'data', label: 'data', type: 'data' }],
    outputs: [],
    params: [
      { id: 'row', label: 'Row to explain (index)', kind: 'int', def: 0, min: 0, max: 100000 },
      { id: 'cls', label: 'Class (index; −1: the usual one)', kind: 'int', def: -1, min: -1, max: 50 },
      { id: 'samples', label: 'Samples', kind: 'int', def: 500, min: 50, max: 5000 },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const m = ctx.in.model, d = ctx.in.data, p = ctx.params;
      EX.need(m, d);
      const k = EX.classOf(m, p.cls), i = Math.min(p.row, d.y.length - 1);
      const res = S.Explain.lime(EX.fnOf(m, k), d.X[i], d.X, { nSamples: p.samples, seed: p.seed, names: d.features.map((f) => f.name) });
      const sd = d.features.map((_, j) => { const c = d.X.map((x) => x[j]), mu = c.reduce((a, b) => a + b, 0) / c.length; return Math.sqrt(c.reduce((a, b) => a + (b - mu) ** 2, 0) / c.length) || 1; });
      return { m, d, k, i, res, sd };
    },
    view(r) {
      const x = r.res;
      return { features: r.d.features.map((f) => f.name), row: r.i, x: Array.from(r.d.X[r.i], r4), y: r.d.y[r.i], task: r.m.task, classes: r.m.classes, cls: r.k, label: r.m.label,
        coef: x.coef.map(r4), scaled: x.coef.map((c, j) => r4(c * r.sd[j])), sd: r.sd.map(r4), intercept: r4(x.intercept), score: r4(x.score), fx: r4(x.predictions[0]), samples: x.samples.length, say: x.trace.map((t) => t.say).slice(0, 3) };
    },
    summary: (r) => 'fit R² ' + WF.fmt(r.res.score, 3),
  });

  WF.block('saliency', {
    cat: 'Explain', label: 'Saliency and Grad-CAM', icon: 'eye', order: 50,
    blurb: 'For an image model: which pixels would change its answer most (saliency), and, for a convolutional model, which regions its last feature maps found the evidence in (Grad-CAM).',
    inputs: [{ id: 'model', label: 'trained image model', type: 'model' }, { id: 'data', label: 'images', type: 'images' }],
    outputs: [],
    params: [{ id: 'first', label: 'First image (index)', kind: 'int', def: 0, min: 0, max: 100000 }, { id: 'count', label: 'Images', kind: 'int', def: 4, min: 1, max: 8 }],
    run(ctx) {
      const m = ctx.in.model, d = ctx.in.data, p = ctx.params;
      if (!m || !m.fitted) throw new Error('wire a trained model into Saliency');
      if (m.inKind !== 'images') throw new Error('saliency is for image models; for tables use SHAP or LIME');
      const out = [];
      for (let t = 0; t < p.count && p.first + t < d.y.length; t++) {
        const i = p.first + t, x = d.X[i], pr = m.proba ? Array.from(m.proba([x])[0]) : null, c = pr ? pr.indexOf(Math.max(...pr)) : m.predict([x])[0];
        let sal = null, cam = null, how = 'central differences';
        if (m.explainImage) { const e = m.explainImage(x, c) || {}; sal = e.saliency || null; cam = e.gradcam || null; how = e.how || 'backpropagation'; }
        else if (m.net && m.net.kind === 'cnn' && S.DL && S.DL.saliency) {
          // a convolutional network: the exact gradient by backpropagation, and Grad-CAM on its last convolution
          try { sal = S.DL.saliency(m.net, x, c).map; how = 'backpropagation'; } catch (e) { sal = null; }
          try { cam = S.DL.gradCam(m.net, x, c).map; } catch (e) { cam = null; }
        }
        if (!sal) {
          // the gradient by central differences: two questions to the model per pixel, asked as one batch
          const h = 0.05, rows = [];
          for (let j = 0; j < x.length; j++) { const a = Float32Array.from(x), b = Float32Array.from(x); a[j] = Math.min(1, a[j] + h); b[j] = Math.max(0, b[j] - h); rows.push(a, b); }
          const f = EX.fnOf(m, c)(rows), g = x.map((_, j) => (f[2 * j] - f[2 * j + 1]) / (2 * h)), mx = Math.max(...g.map(Math.abs)) || 1;
          sal = g.map((v) => Math.abs(v) / mx);
        }
        out.push({ i, y: d.y[i], pred: c, proba: pr, img: Array.from(x, (v) => Math.round(v * 15)), sal: Array.from(sal, (v) => Math.round(v * 100) / 100), cam: cam ? cam.map((row) => Array.from(row, (v) => Math.round(v * 100) / 100)) : null, how });
        ctx.progress && ctx.progress((t + 1) / p.count, 'image ' + i);
      }
      return { out, w: d.w, h: d.h, classes: m.classes, label: m.label };
    },
    view: (r) => ({ items: r.out, w: r.w, h: r.h, classes: r.classes, label: r.label }),
    summary: (r) => r.out.length + ' images',
  });
})(globalThis.Nooral = globalThis.Nooral || {});
