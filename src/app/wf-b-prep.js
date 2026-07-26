// The Prepare blocks, part one: Clean (missing cells and outliers), Impute, Encode (categories as numbers) and Scale.
// Each is fitted on the rows wired into its first input and remembered as a step of the table's pipe, so the same
// numbers are used on the test rows (its second input), on a hidden test set and on a row typed in to predict.
(function (S) {
  'use strict';
  const WF = S.WF, U = () => WF.dutil;
  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  const numCols = (v) => v.features.map((f, j) => (f.type !== 'category' ? j : -1)).filter((j) => j >= 0);

  // ---- clean ------------------------------------------------------------------------------------------------------
  WF.prepBlock('clean', {
    label: 'Clean', icon: 'clear', order: 10, leakWhat: 'cleaning',
    blurb: 'Deals with gaps and wild values: drop the rows that have a missing cell, drop the columns that are mostly missing, or clip outliers to the whiskers of a box plot.',
    params: [
      { id: 'mode', label: 'What to do', kind: 'choice', def: 'rows', options: [['rows', 'drop rows with a missing cell'], ['cols', 'drop columns with many missing cells'], ['clip', 'clip outliers (1.5 × IQR)']] },
      { id: 'maxMissing', label: 'Drop a column when more than this share is missing', kind: 'number', def: 0.3, min: 0, max: 1, step: 0.05, show: (p) => p.mode === 'cols' },
    ],
    keepTest: (t, p) => (p.mode === 'rows' ? t.X.map((r, i) => (r.every(isNum) ? i : -1)).filter((i) => i >= 0) : null),
    fit(v, p) {
      const u = U();
      if (p.mode === 'cols') {
        const share = v.features.map((_, j) => v.X.filter((r) => !isNum(r[j])).length / Math.max(1, v.X.length));
        const keepCols = share.map((s, j) => (s <= p.maxMissing ? j : -1)).filter((j) => j >= 0);
        const pick = (X) => X.map((r) => keepCols.map((j) => r[j]));
        const dropped = v.features.filter((_, j) => !keepCols.includes(j)).map((f, k) => ({ name: f.name, share: u.r(share[v.features.indexOf(f)], 3) }));
        return { X: pick(v.X), features: keepCols.map((j) => v.features[j]), step: { name: 'clean', say: 'keep ' + keepCols.length + ' of ' + v.features.length + ' columns', apply: pick }, info: { mode: 'cols', dropped, share: share.map((s) => u.r(s, 3)) } };
      }
      if (p.mode === 'clip') {
        const lim = v.features.map((f, j) => {
          if (f.type === 'category') return null;
          const xs = u.col(v.X, j).filter(isNum).sort((a, b) => a - b);
          if (!xs.length) return null;
          const q1 = u.quantile(xs, 0.25), q3 = u.quantile(xs, 0.75), iqr = q3 - q1;
          return [q1 - 1.5 * iqr, q3 + 1.5 * iqr];
        });
        const clip = (X) => X.map((r) => r.map((x, j) => (lim[j] && isNum(x) ? Math.min(lim[j][1], Math.max(lim[j][0], x)) : x)));
        const clipped = v.features.map((f, j) => (lim[j] ? v.X.filter((r) => isNum(r[j]) && (r[j] < lim[j][0] || r[j] > lim[j][1])).length : 0));
        return { X: clip(v.X), features: v.features, step: { name: 'clean', say: 'clip outliers to the box-plot whiskers', apply: clip }, info: { mode: 'clip', limits: lim.map((l) => (l ? [u.r(l[0]), u.r(l[1])] : null)), clipped } };
      }
      // drop rows: new rows cannot be dropped at prediction time, so their gaps are filled with the training means
      const keep = v.X.map((r, i) => (r.every(isNum) ? i : -1)).filter((i) => i >= 0);
      if (!keep.length) throw new Error('every row has a missing cell: drop the emptiest columns first, or impute');
      const means = v.features.map((f, j) => (f.type === 'category' ? u.mode(keep.map((i) => v.X[i][j])) : u.mean(keep.map((i) => v.X[i][j]))));
      const fill = (X) => X.map((r) => r.map((x, j) => (isNum(x) ? x : means[j])));
      return { X: v.X, features: v.features, keep, step: { name: 'clean', say: 'drop rows with gaps (new rows: fill gaps with the training means)', apply: fill }, info: { mode: 'rows', dropped: v.X.length - keep.length, missingPerColumn: v.features.map((_, j) => v.X.filter((r) => !isNum(r[j])).length) } };
    },
    view(r) { return { columns: U().beforeAfter(r, 6) }; },
    summary(r) { const i = r.info; return i.mode === 'rows' ? 'dropped ' + i.dropped + ' rows' : i.mode === 'cols' ? 'dropped ' + i.dropped.length + ' columns' : 'clipped ' + i.clipped.reduce((a, b) => a + b, 0) + ' cells'; },
  });

  // ---- impute -----------------------------------------------------------------------------------------------------
  WF.prepBlock('impute', {
    label: 'Impute', icon: 'edit', order: 20, leakWhat: 'filling',
    blurb: 'Fills each missing cell with a value learned from its column: the mean, the median or the most frequent value. Optionally adds a column saying which cells were missing, since a gap can itself carry information.',
    params: [
      { id: 'strategy', label: 'Fill with', kind: 'choice', def: 'mean', options: [['mean', 'the mean'], ['median', 'the median'], ['mode', 'the most frequent value']] },
      { id: 'indicator', label: 'Add “was missing” columns', kind: 'bool', def: false },
    ],
    fit(v, p) {
      const u = U();
      const fills = v.features.map((f, j) => {
        const xs = u.col(v.X, j).filter(isNum).sort((a, b) => a - b);
        if (!xs.length) return 0;
        if (f.type === 'category' || p.strategy === 'mode') return u.mode(xs);
        return p.strategy === 'median' ? u.quantile(xs, 0.5) : xs.reduce((a, b) => a + b, 0) / xs.length;
      });
      const gaps = v.features.map((_, j) => v.X.filter((r) => !isNum(r[j])).length);
      const flag = p.indicator ? gaps.map((g, j) => (g ? j : -1)).filter((j) => j >= 0) : [];
      const apply = (X) => X.map((r) => r.map((x, j) => (isNum(x) ? x : fills[j])).concat(flag.map((j) => (isNum(r[j]) ? 0 : 1))));
      const features = v.features.concat(flag.map((j) => ({ name: v.features[j].name + ' was missing', type: 'number' })));
      return { X: apply(v.X), features, step: { name: 'impute', say: 'fill gaps with the ' + p.strategy, apply }, info: { fills: fills.map((x) => u.r(x)), gaps, strategy: p.strategy, flagged: flag.map((j) => v.features[j].name) } };
    },
    view(r) { return { columns: U().beforeAfter(r, 6) }; },
    summary(r) { return 'filled ' + r.info.gaps.reduce((a, b) => a + b, 0) + ' cells'; },
  });

  // ---- encode -----------------------------------------------------------------------------------------------------
  WF.prepBlock('encode', {
    label: 'Encode', icon: 'convert', order: 30, leakWhat: 'encoding',
    blurb: 'Turns category columns into numbers a model can use: one-hot (a 0/1 column per category, no order implied) or ordinal (the category’s position, which does imply an order).',
    params: [
      { id: 'how', label: 'Encoding', kind: 'choice', def: 'onehot', options: [['onehot', 'one-hot'], ['ordinal', 'ordinal']] },
      { id: 'dropFirst', label: 'Drop the first category’s column (one-hot)', kind: 'bool', def: false, show: (p) => p.how === 'onehot' },
    ],
    fit(v, p) {
      const cats = v.features.map((f, j) => (f.type === 'category' ? j : -1)).filter((j) => j >= 0);
      if (p.how === 'ordinal' || !cats.length) {
        const features = v.features.map((f) => (f.type === 'category' ? Object.assign({}, f, { type: 'number', ordinal: f.categories }) : f));
        return { X: v.X.map((r) => r.slice()), features, step: { name: 'encode', say: 'categories as their positions', apply: (X) => X.map((r) => r.slice()) }, info: { how: 'ordinal', columns: cats.map((j) => ({ name: v.features[j].name, categories: (v.features[j].categories || []).slice(0, 12) })) } };
      }
      const plan = [], features = [];
      v.features.forEach((f, j) => {
        if (f.type !== 'category') { plan.push([j, null]); features.push(f); return; }
        const cs = f.categories || [...new Set(v.X.map((r) => r[j]).filter(isNum))].sort((a, b) => a - b).map(String);
        cs.forEach((c, k) => { if (p.dropFirst && k === 0) return; plan.push([j, k]); features.push({ name: f.name + '=' + c, type: 'number', from: f.name }); });
      });
      const apply = (X) => X.map((r) => plan.map(([j, k]) => (k === null ? r[j] : isNum(r[j]) ? (r[j] === k ? 1 : 0) : NaN)));
      return { X: apply(v.X), features, step: { name: 'encode', say: 'one-hot categories', apply }, info: { how: 'onehot', columns: cats.map((j) => ({ name: v.features[j].name, categories: (v.features[j].categories || []).slice(0, 12) })), width: features.length } };
    },
    view(r) {
      const v = r.before, o = r.out.out, k = U().sample(v.y.length, 6);
      return { sampleBefore: k.map((i) => v.X[i].map((x) => U().r(x, 4))), sampleAfter: k.map((i) => o.X[i].map((x) => U().r(x, 4))).map((row) => row.slice(0, 16)), namesAfter: o.features.slice(0, 16).map((f) => f.name), catsBefore: v.features.map((f) => (f.type === 'category' ? (f.categories || []).slice(0, 12) : null)) };
    },
    summary(r) { return r.before.features.length + ' → ' + r.out.out.features.length + ' columns'; },
  });

  // ---- scale ------------------------------------------------------------------------------------------------------
  WF.prepBlock('scale', {
    label: 'Scale', icon: 'fit', order: 40, leakWhat: 'scaling',
    blurb: 'Puts the number columns on a common scale, so a feature measured in thousands does not drown one measured in fractions: standard (mean 0, standard deviation 1), min–max (0 to 1) or robust (median 0, interquartile range 1).',
    params: [{ id: 'how', label: 'Scaling', kind: 'choice', def: 'standard', options: [['standard', 'standard: (x − mean) / sd'], ['minmax', 'min–max: (x − min) / (max − min)'], ['robust', 'robust: (x − median) / IQR']] }],
    fit(v, p) {
      const u = U(), cols = numCols(v);
      const par = v.features.map((f, j) => {
        if (!cols.includes(j)) return null;
        const xs = u.col(v.X, j).filter(isNum).sort((a, b) => a - b);
        if (!xs.length) return [0, 1];
        if (p.how === 'minmax') return [xs[0], xs[xs.length - 1] - xs[0] || 1];
        if (p.how === 'robust') return [u.quantile(xs, 0.5), u.quantile(xs, 0.75) - u.quantile(xs, 0.25) || 1];
        const m = xs.reduce((a, b) => a + b, 0) / xs.length, sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
        return [m, sd || 1];
      });
      const apply = (X) => X.map((r) => r.map((x, j) => (par[j] && isNum(x) ? (x - par[j][0]) / par[j][1] : x)));
      return { X: apply(v.X), features: v.features, step: { name: 'scale', say: p.how + ' scaling', apply }, info: { how: p.how, params: par.map((q, j) => (q ? { name: v.features[j].name, shift: u.r(q[0]), by: u.r(q[1]) } : null)).filter(Boolean) } };
    },
    view(r) { return { columns: U().beforeAfter(r, 6) }; },
    summary(r) { return r.info.how + ' · ' + r.info.params.length + ' columns'; },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
