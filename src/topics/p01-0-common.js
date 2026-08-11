// Part 1, data and features: the families, and the helpers its topics share. The statistics and the preparation
// steps here (scalers, imputers, encoders, bins, polynomial terms, univariate scores) follow the usual library
// definitions, so the claims check can compare them with reference outputs; every number a narration quotes is
// computed by these or by the engines in src/core on our own datasets.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'data', order, title, blurb });
  fam('p01-tables', 10, 'Tables and features', 'A dataset as a table: one row per example, one column per feature, the label to predict, and the types a column can have.');
  fam('p01-describe', 20, 'Describing data', 'Summaries of one column (centre, spread, quantiles, the histogram), of pairs of columns (the scatter matrix) and correlation.');
  fam('p01-clean', 30, 'Missing values and outliers', 'Gaps in a table and what to fill them with, and values far from the rest: how to find them and what to do.');
  fam('p01-transform', 40, 'Scaling, encoding and binning', 'Putting columns on comparable scales, turning categories into numbers, and cutting numbers into bins.');
  fam('p01-engineer', 50, 'Feature engineering and selection', 'New columns made from old ones (powers and products) and keeping only the columns that help.');
  fam('p01-leakage', 60, 'Leakage', 'When information from the test rows sneaks into training: the score looks wonderful and means nothing.');

  const H = (S.P01 = {});
  // ---- words and numbers ----
  /** a number for a sentence: d decimals at most, trailing zeros dropped, a true minus sign */
  H.f = (v, d) => {
    if (v === Infinity) return '∞';
    if (v === -Infinity) return '−∞';
    if (!Number.isFinite(v)) return '?';
    const s = (+v).toFixed(d === undefined ? 2 : d).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
    return (s === '-0' ? '0' : s).replace(/^-/, '−');
  };
  /** a number with about n significant digits (for values of any size) */
  H.g = (v, n) => {
    if (!Number.isFinite(v)) return H.f(v);
    const a = Math.abs(v);
    if (a >= 1000) return H.f(v, 0);
    if (a === 0) return '0';
    const d = Math.max(0, (n || 3) - 1 - Math.floor(Math.log10(a)));
    return H.f(v, Math.min(d, 6));
  };
  H.pct = (v, d) => H.f(100 * v, d === undefined ? 1 : d) + '%';
  H.plural = (k, word, many) => k + ' ' + (Number(k) === 1 ? word : many || word + 's');
  H.list = (xs) => { xs = xs.map(String); return xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]; };
  H.an = (w) => (/^[aeiou]/i.test(String(w)) && !/^(one|use|uni|eu)/i.test(String(w)) ? 'an ' : 'a ') + w;
  H.cap = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);
  /** a free text item */
  H.text = (key, x, y, text, cls, size, anchor) => { const it = { key, kind: 'text', x, y, text, cls: cls === undefined ? 'strong' : cls }; if (size) it.size = size; if (anchor) it.anchor = anchor; return it; };
  H.title = (key, x, y, text) => ({ key, kind: 'text', x, y, text, cls: 'title' });
  H.box = (x, y, w, h) => ({ x, y, w, h });

  // ---- statistics (population forms, as the scalers use) ----
  H.isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  H.sum = (a) => { let s = 0; for (const v of a) s += v; return s; };
  H.mean = (a) => H.sum(a) / a.length;
  H.variance = (a, ddof) => { const m = H.mean(a); let s = 0; for (const v of a) s += (v - m) * (v - m); return s / (a.length - (ddof || 0)); };
  H.sd = (a, ddof) => Math.sqrt(H.variance(a, ddof));
  H.sorted = (a) => a.slice().sort((x, y) => x - y);
  /** the linear-interpolation quantile of sorted values (the default of the common numeric libraries) */
  H.quantile = (s, q) => { if (!s.length) return NaN; const h = (s.length - 1) * q, lo = Math.floor(h), hi = Math.min(lo + 1, s.length - 1); return s[lo] + (h - lo) * (s[hi] - s[lo]); };
  H.median = (a) => H.quantile(H.sorted(a), 0.5);
  /** five numbers and the box-plot whiskers (the furthest values within 1.5 IQR of the box) */
  H.five = (a, k) => {
    const s = H.sorted(a), q1 = H.quantile(s, 0.25), med = H.quantile(s, 0.5), q3 = H.quantile(s, 0.75), iqr = q3 - q1, kk = k === undefined ? 1.5 : k;
    const loF = q1 - kk * iqr, hiF = q3 + kk * iqr, inside = s.filter((v) => v >= loF && v <= hiF);
    return { min: s[0], max: s[s.length - 1], q1, med, q3, iqr, loF, hiF, lo: inside.length ? inside[0] : q1, hi: inside.length ? inside[inside.length - 1] : q3, n: s.length };
  };
  /** equal-width histogram edges over [min, max] (n bins) and the counts; the last bin includes its right edge */
  H.histogram = (a, n, lo, hi) => {
    lo = lo === undefined ? Math.min(...a) : lo; hi = hi === undefined ? Math.max(...a) : hi;
    if (hi === lo) { lo -= 0.5; hi += 0.5; }
    const edges = Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n), counts = new Array(n).fill(0);
    for (const v of a) { if (v < lo || v > hi) continue; let b = Math.floor(((v - lo) / (hi - lo)) * n); if (b >= n) b = n - 1; counts[b]++; }
    return { edges, counts };
  };
  /** Pearson's correlation of two columns, with the sums it is made of */
  H.pearson = (x, y) => {
    const mx = H.mean(x), my = H.mean(y);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < x.length; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
    return { r: sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0, mx, my, sxy, sxx, syy };
  };
  /** ranks with ties given their average rank (1-based) */
  H.ranks = (a) => {
    const idx = a.map((_, i) => i).sort((i, j) => a[i] - a[j]), r = new Array(a.length);
    for (let k = 0; k < idx.length;) { let e = k; while (e + 1 < idx.length && a[idx[e + 1]] === a[idx[k]]) e++; const avg = (k + e) / 2 + 1; for (let t = k; t <= e; t++) r[idx[t]] = avg; k = e + 1; }
    return r;
  };
  H.spearman = (x, y) => H.pearson(H.ranks(x), H.ranks(y)).r;

  // ---- datasets ----
  const cache = new Map();
  /** a dataset from S.Data, built once per page for the same options */
  H.data = (id, opt) => { const k = id + JSON.stringify(opt || {}); if (!cache.has(k)) cache.set(k, S.Data.load(id, opt || {})); return cache.get(k); };
  H.col = (ds, j) => ds.X.map((r) => r[j]);
  /** the name of a feature with its unit: "rain (mm)" */
  H.fname = (f) => f.name.replace(/_/g, ' ') + (f.unit ? ' (' + f.unit + ')' : '');
  H.short = (f) => f.name.replace(/_/g, ' ');
  /** a class name of a dataset */
  H.cname = (ds, c) => (ds.classes ? ds.classes[c] : ds.target && ds.target.categories ? ds.target.categories[c] : String(c));

  // ---- preparation steps, fitted on some rows and applied to any ----
  /** fit a scaler on the columns of X: kind 'standard' | 'minmax' | 'robust'; returns {centre, scale, apply(X)} */
  H.scaler = (X, kind) => {
    const d = X[0].length, centre = [], scale = [];
    for (let j = 0; j < d; j++) {
      const c = X.map((r) => r[j]);
      if (kind === 'minmax') { const lo = Math.min(...c), hi = Math.max(...c); centre.push(lo); scale.push(hi - lo || 1); }
      else if (kind === 'robust') { const s = H.sorted(c), q1 = H.quantile(s, 0.25), q3 = H.quantile(s, 0.75); centre.push(H.quantile(s, 0.5)); scale.push(q3 - q1 || 1); }
      else if (kind === 'none') { centre.push(0); scale.push(1); }
      else { centre.push(H.mean(c)); scale.push(H.sd(c) || 1); }
    }
    return { kind, centre, scale, apply: (Y) => Y.map((r) => r.map((v, j) => (v - centre[j]) / scale[j])) };
  };
  H.SCALERS = [
    { value: 'none', label: 'no scaling' }, { value: 'standard', label: 'standard (z-score)' },
    { value: 'minmax', label: 'min-max to [0, 1]' }, { value: 'robust', label: 'robust (median, IQR)' },
  ];
  H.scalerWords = { none: 'no scaling', standard: 'standard scaling', minmax: 'min-max scaling', robust: 'robust scaling' };

  /** a seeded random generator from S.Data */
  H.rng = (seed) => S.Data.rng(seed);
})(globalThis.Nooral = globalThis.Nooral || {});
