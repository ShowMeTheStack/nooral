// The Prepare blocks, part two: Select (keep columns by hand or the k that tell most about the target), Polynomial
// features, PCA, and Vectorise (texts to a table of word counts or TF-IDF weights).
(function (S) {
  'use strict';
  const WF = S.WF, U = () => WF.dutil;
  const isNum = (x) => typeof x === 'number' && Number.isFinite(x);
  const pick = (cols) => (X) => X.map((r) => cols.map((j) => r[j]));

  /** how much a column tells about the target: mutual information (binned), |correlation| or plain variance */
  function relevance(v, j, how) {
    const u = U(), xs = v.X.map((r) => r[j]), ok = xs.map((x, i) => (isNum(x) && isNum(v.y[i]) ? i : -1)).filter((i) => i >= 0);
    const x = ok.map((i) => xs[i]), y = ok.map((i) => v.y[i]), n = x.length;
    if (n < 3) return 0;
    const mx = x.reduce((a, b) => a + b, 0) / n;
    if (how === 'variance') return x.reduce((a, b) => a + (b - mx) ** 2, 0) / n;
    if (how === 'corr') {
      if (v.task === 'classification' && (v.classes || []).length > 2) {
        // the correlation ratio: the share of x's variance explained by the class means
        const tot = x.reduce((a, b) => a + (b - mx) ** 2, 0) || 1, by = new Map();
        x.forEach((xi, i) => { const g = by.get(y[i]) || [0, 0]; g[0] += xi; g[1]++; by.set(y[i], g); });
        let between = 0; for (const [s, c] of by.values()) between += c * (s / c - mx) ** 2;
        return Math.sqrt(between / tot);
      }
      const my = y.reduce((a, b) => a + b, 0) / n;
      let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
      return sxx && syy ? Math.abs(sxy / Math.sqrt(sxx * syy)) : 0;
    }
    // mutual information between x (in 8 equal-count bins) and y (classes, or 8 bins of a number)
    const bin = (vals, k) => { const s = vals.slice().sort((a, b) => a - b), cuts = []; for (let q = 1; q < k; q++) cuts.push(u.quantile(s, q / k)); return vals.map((t) => cuts.filter((c) => t > c).length); };
    const bx = new Set(x).size <= 8 ? x.map((t) => [...new Set(x)].sort((a, b) => a - b).indexOf(t)) : bin(x, 8);
    const by = v.task === 'classification' ? y : bin(y, 8);
    const joint = new Map(), px = new Map(), py = new Map();
    for (let i = 0; i < n; i++) { const k = bx[i] + ',' + by[i]; joint.set(k, (joint.get(k) || 0) + 1); px.set(bx[i], (px.get(bx[i]) || 0) + 1); py.set(by[i], (py.get(by[i]) || 0) + 1); }
    let mi = 0;
    for (const [k, c] of joint) { const [a, b] = k.split(',').map(Number); mi += (c / n) * Math.log((c * n) / (px.get(a) * py.get(b))); }
    return Math.max(0, mi);
  }

  WF.prepBlock('select', {
    label: 'Select features', icon: 'check', order: 50, leakWhat: 'choice of columns',
    blurb: 'Keeps some columns and drops the rest: the ones you name, or the k columns that tell most about the target (mutual information, correlation) or vary most.',
    params: [
      { id: 'mode', label: 'Keep', kind: 'choice', def: 'kbest', options: [['kbest', 'the k best columns'], ['columns', 'the columns I name']] },
      { id: 'score', label: 'Best by', kind: 'choice', def: 'mi', options: [['mi', 'mutual information with the target'], ['corr', 'correlation with the target'], ['variance', 'variance']], show: (p) => p.mode === 'kbest' },
      { id: 'k', label: 'k', kind: 'int', def: 3, min: 1, max: 500, show: (p) => p.mode === 'kbest' },
      { id: 'columns', label: 'Columns to keep', kind: 'columns', def: [], show: (p) => p.mode === 'columns' },
    ],
    fit(v, p) {
      const u = U();
      const scores = v.features.map((_, j) => relevance(v, j, p.mode === 'kbest' ? p.score : 'mi'));
      let cols;
      if (p.mode === 'columns') {
        const want = [].concat(p.columns || []).map(String);
        cols = v.features.map((f, j) => (want.includes(f.name) ? j : -1)).filter((j) => j >= 0);
        if (!cols.length) throw new Error('name at least one column to keep: double-click the block');
      } else cols = scores.map((s, j) => [s, j]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, Math.max(1, Math.min(p.k, v.features.length))).map((x) => x[1]).sort((a, b) => a - b);
      return { X: pick(cols)(v.X), features: cols.map((j) => v.features[j]), step: { name: 'select', say: 'keep ' + cols.map((j) => v.features[j].name).join(', '), apply: pick(cols) }, info: { scores: scores.map((s) => u.r(s, 3)), kept: cols, by: p.mode === 'kbest' ? p.score : 'mi' } };
    },
    view(r) { return { names: r.before.features.map((f) => f.name) }; },
    summary(r) { return 'kept ' + r.info.kept.length + ' of ' + r.before.features.length; },
  });

  WF.prepBlock('poly', {
    label: 'Polynomial features', icon: 'curve', order: 60,
    blurb: 'Adds the squares, cubes and products of the columns, so a straight-line model can draw curves. Degree 2 on two columns gives x₁, x₂, x₁², x₁x₂ and x₂²; the count grows fast with the degree.',
    params: [
      { id: 'degree', label: 'Degree', kind: 'int', def: 2, min: 2, max: 5 },
      { id: 'interactionOnly', label: 'Only products of different columns', kind: 'bool', def: false },
    ],
    fit(v, p) {
      const d = v.features.length;
      const pf = S.ML.PolynomialFeatures({ degree: p.degree, includeBias: false, interactionOnly: !!p.interactionOnly }).fit(v.X.filter((r) => r.every(isNum)).slice(0, 1).concat([new Array(d).fill(0)]));
      if (pf.combos.length > 400) throw new Error('degree ' + p.degree + ' on ' + d + ' columns makes ' + pf.combos.length + ' features: too many to learn from here. Lower the degree or select fewer columns first');
      const sup = '⁰¹²³⁴⁵⁶⁷⁸⁹';
      const nm = (c) => { const m = new Map(); c.forEach((j) => m.set(j, (m.get(j) || 0) + 1)); return [...m].map(([j, k]) => v.features[j].name + (k > 1 ? String(k).split('').map((ch) => sup[+ch]).join('') : '')).join('·'); };
      const features = pf.combos.map((c) => ({ name: nm(c), type: 'number', degree: c.length }));
      const apply = (X) => pf.transform(X);
      return { X: apply(v.X), features, step: { name: 'poly', say: 'polynomial features of degree ' + p.degree, apply }, info: { degree: p.degree, count: features.length, combos: pf.combos.length } };
    },
    view(r) { const u = U(), o = r.out.out, k = u.sample(o.y.length, 5); return { names: o.features.slice(0, 40).map((f) => f.name), more: Math.max(0, o.features.length - 40), sample: k.map((i) => o.X[i].slice(0, 10).map((x) => u.r(x, 3))) }; },
    summary(r) { return r.before.features.length + ' → ' + r.out.out.features.length + ' features'; },
  });

  WF.prepBlock('pca', {
    label: 'PCA', icon: 'target', order: 70, leakWhat: 'projection',
    blurb: 'Principal component analysis: rotates the data onto the directions in which it varies most and keeps the first few. Scale first, or the column with the largest units decides everything.',
    params: [
      { id: 'n', label: 'Components', kind: 'int', def: 2, min: 1, max: 100 },
      { id: 'whiten', label: 'Whiten (each component to variance 1)', kind: 'bool', def: false },
    ],
    fit(v, p) {
      const u = U();
      if (v.X.some((r) => !r.every(isNum))) throw new Error('PCA needs every cell filled: clean or impute first');
      const k = Math.min(p.n, v.features.length, v.X.length);
      const pca = S.ML.PCA({ nComponents: k, whiten: !!p.whiten, trace: false }).fit(v.X);
      const apply = (X) => pca.transform(X);
      const full = S.ML.PCA({ trace: false }).fit(v.X);
      return { X: apply(v.X), features: Array.from({ length: k }, (_, i) => ({ name: 'PC' + (i + 1), type: 'number' })), step: { name: 'pca', say: k + ' principal components', apply },
        info: { k, ratio: full.explainedVarianceRatio.slice(0, 20).map((x) => u.r(x, 4)), components: pca.components.slice(0, 6).map((c) => c.slice(0, 20).map((x) => u.r(x, 3))), names: v.features.slice(0, 20).map((f) => f.name), kept: u.r(pca.explainedVarianceRatio.reduce((a, b) => a + b, 0), 4) } };
    },
    view(r) {
      const u = U(), o = r.out.out, keep = u.sample(o.y.length, 900);
      return { points: o.features.length >= 2 ? { P: keep.map((i) => [u.r(o.X[i][0]), u.r(o.X[i][1])]), y: keep.map((i) => o.y[i]) } : null };
    },
    summary(r) { return r.info.k + ' components · ' + Math.round(100 * r.info.kept) + '% of the variance'; },
  });

  // ---- texts to numbers ---------------------------------------------------------------------------------------------
  const STOP = new Set('a an the and or but of to in on at for with is are was were be been it its this that these those i you he she we they my your our their me him her them as by from so not no do does did have has had will would can could just than then there here'.split(' '));
  const tokens = (t, p) => { let w = String(t).toLowerCase().match(/[a-z0-9']+/g) || []; if (p.stop) w = w.filter((x) => !STOP.has(x)); if (p.ngrams === 2) w = w.concat(w.slice(1).map((x, i) => w[i] + ' ' + x)); return w; };
  WF.prepBlock('vectorise', {
    label: 'Vectorise text', icon: 'terminal', order: 80, takes: 'text', gives: 'table', leakWhat: 'vocabulary',
    blurb: 'Turns each text into a row of numbers: one column per word of the vocabulary, holding how often the word appears (counts), whether it does (0/1), or TF-IDF, which weighs down words that appear everywhere.',
    params: [
      { id: 'weight', label: 'Each cell holds', kind: 'choice', def: 'count', options: [['count', 'the count'], ['binary', '0 or 1'], ['tfidf', 'TF-IDF']] },
      { id: 'vocab', label: 'Vocabulary size', kind: 'int', def: 200, min: 5, max: 3000 },
      { id: 'ngrams', label: 'Words', kind: 'choice', def: 1, options: [[1, 'single words'], [2, 'single words and pairs']] },
      { id: 'stop', label: 'Leave out common words (the, and, of…)', kind: 'bool', def: true },
    ],
    fit(v, p) {
      const u = U();
      if (v.kind !== 'text') throw new Error('Vectorise works on texts, not ' + v.kind);
      const docs = v.texts.map((t) => tokens(t, p)), df = new Map(), tf = new Map();
      for (const d of docs) { for (const w of new Set(d)) df.set(w, (df.get(w) || 0) + 1); for (const w of d) tf.set(w, (tf.get(w) || 0) + 1); }
      const vocab = [...tf.keys()].sort((a, b) => tf.get(b) - tf.get(a) || (a < b ? -1 : 1)).slice(0, p.vocab);
      const at = new Map(vocab.map((w, j) => [w, j])), N = docs.length;
      const idf = vocab.map((w) => Math.log((1 + N) / (1 + df.get(w))) + 1);
      const apply = (texts) => texts.map((t) => {
        const row = new Array(vocab.length).fill(0);
        for (const w of tokens(t, p)) { const j = at.get(w); if (j !== undefined) row[j] += 1; }
        if (p.weight === 'binary') return row.map((c) => (c ? 1 : 0));
        if (p.weight === 'tfidf') { const r = row.map((c, j) => c * idf[j]), nr = Math.sqrt(r.reduce((a, b) => a + b * b, 0)) || 1; return r.map((x) => x / nr); }
        return row;
      });
      const features = vocab.map((w) => ({ name: w, type: 'number' }));
      return { X: apply(v.texts), features, step: { name: 'vectorise', say: 'a vocabulary of ' + vocab.length + ' words', apply }, extra: { kind: 'table' },
        info: { vocab: vocab.slice(0, 60), counts: vocab.slice(0, 60).map((w) => tf.get(w)), df: vocab.slice(0, 60).map((w) => df.get(w)), idf: idf.slice(0, 60).map((x) => u.r(x, 3)), size: vocab.length, weight: p.weight } };
    },
    view(r) {
      const u = U(), o = r.out.out, k = u.sample(o.y.length, 6), cols = r.info.vocab.slice(0, 12);
      return { texts: k.map((i) => o.texts[i]), rowsY: k.map((i) => o.y[i]), cols, cells: k.map((i) => cols.map((_, j) => u.r(o.X[i][j], 3))) };
    },
    summary(r) { return r.info.size + ' words'; },
  });
})(globalThis.Nooral = globalThis.Nooral || {});
