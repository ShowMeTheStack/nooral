// Compare: up to four models side by side on the same rows. Each input takes predictions (from an Evaluate block) or a
// trained model (then the block makes the predictions itself on the data wired in, and for 2-D data it also asks the
// model about every place in the plane, so the boundaries can be drawn side by side).
(function (S) {
  'use strict';
  const WF = S.WF;
  const r3 = (x) => Math.round(x * 1000) / 1000;
  const SLOTS = ['a', 'b', 'c', 'd'];
  /** what a 2-D model says over the plane of its data: {xr, yr, nx, ny, vals (class), conf} */
  WF.EV = WF.EV || {};
  WF.EV.boundary = function (model, data, n) {
    if (!model || !model.fitted || data.kind !== 'table' || data.features.length !== 2 || model.features.length !== 2) return null;
    n = n || 48;
    const xs = data.X.map((r) => r[0]), ys = data.X.map((r) => r[1]);
    const pad = (a, b) => { const d = (b - a) * 0.08 || 1; return [a - d, b + d]; };
    const xr = pad(Math.min(...xs), Math.max(...xs)), yr = pad(Math.min(...ys), Math.max(...ys));
    const pts = [];
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) pts.push([xr[0] + ((i + 0.5) / n) * (xr[1] - xr[0]), yr[1] - ((j + 0.5) / n) * (yr[1] - yr[0])]);
    const out = { xr, yr, nx: n, ny: n };
    if (data.task === 'classification') {
      const P = model.proba ? model.proba(pts) : null, L = P ? null : model.predict(pts);
      out.vals = pts.map((_, i) => (P ? P[i].indexOf(Math.max(...P[i])) : L[i]));
      out.conf = pts.map((_, i) => (P ? r3(Math.max(...P[i])) : 1));
      if (P && data.classes.length === 2) out.p1 = pts.map((_, i) => r3(P[i][1]));
    } else out.vals = model.predict(pts).map(r3);
    return out;
  };
  WF.block('compare', {
    cat: 'Compare', label: 'Compare', icon: 'compare', order: 10,
    blurb: 'Up to four models on the same rows: their scores side by side with the best of each marked, their confusion matrices, and on 2-D data their boundaries. Wire in predictions, or trained models and the test data.',
    inputs: [
      { id: 'a', label: 'first', type: ['predictions', 'model'] }, { id: 'b', label: 'second', type: ['predictions', 'model'], optional: true },
      { id: 'c', label: 'third', type: ['predictions', 'model'], optional: true }, { id: 'd', label: 'fourth', type: ['predictions', 'model'], optional: true },
      { id: 'data', label: 'test data (for models)', type: 'data', optional: true },
    ],
    outputs: [],
    params: [],
    run(ctx) {
      const out = [];
      for (const k of SLOTS) {
        const v = ctx.in[k];
        if (!v) continue;
        if (v.kind === 'model') {
          if (!ctx.in.data) throw new Error('the ' + k + ' input is a model: wire the test data into Compare too');
          const t0 = Date.now(), pred = WF.predict(v, ctx.in.data);
          out.push({ slot: k, pred, boundary: WF.EV.boundary(v, ctx.in.data), ms: Date.now() - t0 });
        } else out.push({ slot: k, pred: v });
      }
      if (!out.length) throw new Error('wire at least one model or set of predictions into Compare');
      const task = out[0].pred.task;
      if (out.some((o) => o.pred.task !== task)) throw new Error('these models do not answer the same kind of question');
      if (out.some((o) => o.pred.y.length !== out[0].pred.y.length)) throw new Error('these predictions are for different rows: compare models on the same test rows');
      return { list: out, data: ctx.in.data || null };
    },
    view(r) {
      const L = r.list, p0 = L[0].pred, cls = p0.task === 'classification';
      const items = L.map((o) => ({ slot: o.slot, model: o.pred.model, scores: WF.scores(o.pred), counts: cls ? S.Metrics.confusion(o.pred.y, o.pred.pred, { labels: p0.classes.map((_, i) => i) }).counts : null, boundary: o.boundary || null }));
      const names = Object.keys(items[0].scores).filter((m) => items.every((it) => it.scores[m] !== undefined));
      const best = {};
      for (const m of names) { const hi = WF.METRICS[m].higher; best[m] = items.reduce((b, it, i) => ((hi ? it.scores[m] > items[b].scores[m] : it.scores[m] < items[b].scores[m]) ? i : b), 0); }
      const d = r.data, pts = d && d.kind === 'table' && d.features.length === 2 ? { X: d.X.slice(0, 400).map((x) => x.map(r3)), y: d.y.slice(0, 400), names: d.features.map((f) => f.name) } : null;
      return { task: p0.task, classes: p0.classes, n: p0.y.length, items, metrics: names, best, points: pts };
    },
    summary(r, v) {
      if (!v) return '';
      const m = v.metrics.includes('accuracy') ? 'accuracy' : v.metrics.includes('r2') ? 'r2' : v.metrics[0];
      return m ? 'best ' + WF.METRICS[m].label + ': ' + v.items[v.best[m]].model : v.items.length + ' models';
    },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
