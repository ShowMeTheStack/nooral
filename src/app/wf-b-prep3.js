// The Resample block: random oversampling (copies of rows of the smaller classes) or undersampling (a random part of
// the larger classes), done to the training rows only. The rows wired into its second input (the test rows) pass
// through untouched: a test set with copied or missing rows would no longer say how the model does on the world.
(function (S) {
  'use strict';
  const WF = S.WF;
  WF.block('resample', {
    cat: 'Prepare', label: 'Resample', icon: 'shuffle', order: 80,
    blurb: 'Evens out the classes of the training rows: copy rows of the rare classes (oversampling) or keep only part of the common ones (undersampling). The test rows pass through unchanged.',
    inputs: [{ id: 'in', label: 'training rows', type: 'data' }, { id: 'test', label: 'test rows', type: 'data', optional: true }],
    outputs: [{ id: 'out', label: 'resampled', type: 'same:in' }, { id: 'test', label: 'test (unchanged)', type: 'same:in' }],
    params: [
      { id: 'how', label: 'How', kind: 'choice', def: 'over', options: [['over', 'oversample: copy rows of the smaller classes'], ['under', 'undersample: drop rows of the larger classes']] },
      { id: 'ratio', label: 'Each class at least this share of the largest (over) / at most this many times the smallest (under)', kind: 'number', def: 1, min: 0.1, max: 10, step: 0.1 },
      { id: 'seed', label: 'Seed', kind: 'seed', def: 1 },
    ],
    run(ctx) {
      const v = ctx.in.in, p = ctx.params;
      if (v.task !== 'classification') throw new Error('resampling evens out classes: this data has no classes (it is a ' + (v.task || 'table without a target') + ')');
      const r = WF.resample(v.y, v.classes.length, p.how, +p.ratio || 1, p.seed);
      const out = WF.subset(v, r.idx, v.name);
      const o = { out };
      if (ctx.in.test) o.test = ctx.in.test;
      return { out: o, before: r.before, after: r.after, how: p.how };
    },
    view(r) { return { how: r.how, classes: r.out.out.classes, before: r.before, after: r.after, test: r.out.test ? r.out.test.y.length : 0 }; },
    summary(r) { return (r.how === 'over' ? 'oversampled ' : 'undersampled ') + r.before.reduce((a, b) => a + b, 0) + ' → ' + r.after.reduce((a, b) => a + b, 0) + ' rows'; },
  });
  /**
   * the rows a resampling keeps, with repeats: over brings every class up to ratio × the largest (copies drawn at
   * random, every original row kept); under brings every class down to ratio × the smallest (a random part kept)
   */
  WF.resample = function (y, K, how, ratio, seed) {
    const rng = S.Data.rng(seed || 1), groups = Array.from({ length: K }, () => []);
    y.forEach((c, i) => groups[c].push(i));
    const before = groups.map((g) => g.length), present = before.filter((n) => n > 0);
    const idx = [];
    if (how === 'under') {
      const cap = Math.max(1, Math.round(Math.min(...present) * Math.max(1, ratio)));
      for (const g of groups) { const a = g.slice(); rng.shuffle(a); idx.push(...a.slice(0, Math.min(cap, a.length))); }
    } else {
      const want = Math.round(Math.max(...present) * Math.min(1, ratio));
      for (const g of groups) { idx.push(...g); for (let k = g.length; g.length && k < want; k++) idx.push(g[Math.floor(rng() * g.length)]); }
    }
    idx.sort((a, b) => a - b);
    const after = new Array(K).fill(0);
    for (const i of idx) after[y[i]]++;
    return { idx, before, after };
  };
})(globalThis.Nooral = globalThis.Nooral || {});
