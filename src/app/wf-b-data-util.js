// Shared pieces of the data, prepare and unsupervised blocks (DOM-free: they run in the workflow's worker and in node):
// column statistics and small histograms for the viewers, a 2-D projection for pictures of wide tables, a sample of
// rows that keeps a view small, and the frame of a Prepare block (fit on the rows wired into 'in', then do the same
// to the rows wired into the optional 'test' input, and remember it as one more step of the table's pipe).
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const U = (WF.dutil = WF.dutil || {});
  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  U.isNum = isNum;
  U.r = (x, d) => (isNum(x) ? +(+x).toPrecision(d || 4) : null);
  U.col = (X, j) => X.map((r) => r[j]);
  U.mean = (a) => { let s = 0, n = 0; for (const v of a) if (isNum(v)) { s += v; n++; } return n ? s / n : NaN; };
  U.quantile = (sorted, q) => { if (!sorted.length) return NaN; const p = (sorted.length - 1) * q, i = Math.floor(p), f = p - i; return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i]; };
  U.mode = (a) => { const m = new Map(); let best = NaN, bn = 0; for (const v of a) if (isNum(v)) { const c = (m.get(v) || 0) + 1; m.set(v, c); if (c > bn || (c === bn && v < best)) { bn = c; best = v; } } return best; };

  /** a column's summary: count, missing, mean, sd, min, quartiles, max, and a histogram of up to bins bars */
  U.stats = function (X, j, f, bins) {
    const all = U.col(X, j), xs = all.filter(isNum).sort((a, b) => a - b), n = xs.length;
    const o = { name: f.name, type: f.type, n, missing: all.length - n };
    if (!n) return o;
    if (f.type === 'category' && f.categories) {
      const c = f.categories.map(() => 0);
      for (const v of xs) if (c[v] !== undefined) c[v]++;
      return Object.assign(o, { categories: f.categories.slice(0, 12), counts: c.slice(0, 12), more: Math.max(0, f.categories.length - 12) });
    }
    const m = xs.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, n - 1));
    const lo = xs[0], hi = xs[n - 1], k = bins || 12, h = new Array(k).fill(0), w = (hi - lo) / k || 1;
    for (const v of xs) h[Math.min(k - 1, Math.floor((v - lo) / w))]++;
    return Object.assign(o, { mean: U.r(m), sd: U.r(sd), min: U.r(lo), q1: U.r(U.quantile(xs, 0.25)), median: U.r(U.quantile(xs, 0.5)), q3: U.r(U.quantile(xs, 0.75)), max: U.r(hi), hist: h, lo: U.r(lo), hi: U.r(hi) });
  };
  /** the summaries of the first `max` columns of a table */
  U.columns = (v, max) => v.features.slice(0, max || 10).map((f, j) => U.stats(v.X, j, f));

  /** at most n row indexes, spread evenly (the same ones every time) */
  U.sample = function (len, n) { if (len <= n) return Array.from({ length: len }, (_, i) => i); const out = []; for (let i = 0; i < n; i++) out.push(Math.floor((i * len) / n)); return out; };

  /** 2-D coordinates of rows for a picture: the two columns themselves, or the first two principal components */
  U.plane = function (X, names) {
    const d = X[0] ? X[0].length : 0;
    if (d === 2) return { P: X.map((r) => [r[0], r[1]]), names: names.slice(0, 2), how: 'the two features' };
    if (d === 1) return { P: X.map((r) => [r[0], 0]), names: [names[0], ''], how: 'the one feature' };
    // on standardised columns, so a feature in large units does not decide the picture alone
    const ok = X.filter((r) => r.every(isNum)), m = new Array(d).fill(0), sd = new Array(d).fill(0);
    for (const r of ok) r.forEach((x, j) => (m[j] += x / ok.length));
    for (const r of ok) r.forEach((x, j) => (sd[j] += (x - m[j]) ** 2 / ok.length));
    const z = (r) => r.map((x, j) => (isNum(x) ? (x - m[j]) / (Math.sqrt(sd[j]) || 1) : 0));
    const pca = S.ML.PCA({ nComponents: 2, trace: false }).fit(ok.map(z));
    const proj = { transform: (Y) => pca.transform(Y.map(z)), components: pca.components.map((c) => c.map((w, j) => w / (Math.sqrt(sd[j]) || 1))) };
    return { P: proj.transform(X), names: ['PC1', 'PC2'], how: 'the first two principal components of the ' + d + ' standardised features', pca: proj };
  };
  U.round2 = (P) => P.map((p) => [U.r(p[0], 4), U.r(p[1], 4)]);

  /** the roles of the rows a block was fitted on, for the leakage note */
  U.leak = function (v, what) {
    if (v.part === 'train') return { ok: true, note: 'Fitted on the ' + v.y.length + ' training rows only, then the same ' + what + ' was applied to the test rows: nothing about the test rows leaked into training.' };
    if (v.part === 'test') return { ok: false, note: 'Fitted on test rows: the model would be judged with knowledge of what it is judged on. Fit on the training rows and wire the test rows into the second input.' };
    return { ok: false, note: 'Fitted on all ' + v.y.length + ' rows, before any split: the ' + what + ' already knows the rows that will later be used for testing, so their statistics leak into training. To avoid it, split first, wire the training rows into this block and the test rows into its second input.' };
  };

  /**
   * a Prepare block: def.fit(v, params, ctx) -> {X, features, step: {name, say, apply(X)}, keep?: [row indexes],
   * info: {...for the view}}; the block fits on 'in' and does the same (step.apply, or keepTest) to 'test'
   */
  WF.prepBlock = function (type, def) {
    return WF.block(type, {
      cat: 'Prepare', label: def.label, blurb: def.blurb, icon: def.icon || 'tidy', order: def.order || 50,
      inputs: [{ id: 'in', label: 'fit on', type: def.takes || 'table' }, { id: 'test', label: 'apply to', type: def.takes || 'table', optional: true }],
      outputs: [{ id: 'out', label: 'out', type: def.gives || 'table' }, { id: 'test', label: 'applied', type: def.gives || 'table' }],
      params: def.params || [],
      run(ctx) {
        const v = ctx.in.in, p = ctx.params;
        if (def.takes !== 'text' && v.kind !== 'table') throw new Error(def.label + ' works on tables, not ' + v.kind);
        const r = def.fit(v, p, ctx);
        let out = WF.transformed(v, r.X, r.features, r.step, r.extra);
        if (r.keep) out = WF.subset(out, r.keep);
        const res = { out: { out }, info: r.info || {}, before: v, keep: r.keep || null };
        const t = ctx.in.test;
        if (t) {
          let tx = t.kind === 'text' && def.takes === 'text' ? r.step.apply(t.texts) : r.step.apply(t.X);
          let tv = WF.transformed(t, tx, r.features, r.step, r.extra);
          if (def.keepTest) { const k = def.keepTest(t, p); if (k) tv = WF.subset(tv, k); }
          res.out.test = tv;
        }
        return res;
      },
      view(r, ctx) {
        const v = r.before, o = r.out.out;
        const base = { kind: type, label: def.label, n: v.y.length, nOut: o.y.length, test: r.out.test ? r.out.test.y.length : 0, leak: def.leakWhat ? U.leak(v, def.leakWhat) : null,
          featuresBefore: v.features ? v.features.map((f) => f.name) : [], featuresAfter: o.features.map((f) => f.name), info: r.info, task: v.task, classes: v.classes };
        if (def.view) Object.assign(base, def.view(r, ctx));
        return base;
      },
      summary(r) { return def.summary ? def.summary(r) : r.out.out.features.length + ' features' + (r.keep ? ' · ' + r.out.out.y.length + ' rows' : ''); },
    });
  };
  /** before/after summaries of the columns a step changed (the same names on both sides where they exist) */
  U.beforeAfter = function (r, max) {
    const v = r.before, o = r.out.out, names = o.features.slice(0, max || 6).map((f) => f.name);
    return names.map((nm) => {
      const jb = v.features.findIndex((f) => f.name === nm), ja = o.features.findIndex((f) => f.name === nm);
      return { name: nm, before: jb >= 0 ? U.stats(v.X, jb, v.features[jb]) : null, after: U.stats(o.X, ja, o.features[ja]) };
    });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
