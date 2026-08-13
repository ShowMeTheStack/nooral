// Part 2, splitting and evaluation: the families, and what its topics share: binary tasks scored by a fitted model
// on held-out rows (the scores every threshold, curve and calibration topic works on), and the confusion matrix as a
// drawing whose four (or k × k) cells name their counts.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'evaluation', order, title, blurb });
  fam('p02-split', 10, 'Splitting the data', 'Training, validation and test rows, keeping class shares with stratification, and k-fold cross-validation fold by fold.');
  fam('p02-confusion', 20, 'The confusion matrix', 'Where the predictions went: the four outcomes of a binary classifier, many classes, normalised rows and columns, and the scores read from them.');
  fam('p02-threshold', 30, 'Thresholds and curves', 'A score becomes a decision at a threshold: precision against recall, ROC and its area, the precision–recall curve.');
  fam('p02-prob', 40, 'Probabilities', 'Are the probabilities honest? Reliability diagrams, calibration error, and the log loss that punishes confident mistakes.');
  fam('p02-regression', 50, 'Regression errors', 'MAE, MSE, RMSE and R², and the residual plots that show what a single number hides.');
  fam('p02-tuning', 60, 'Complexity and tuning', 'Bias and variance in learning curves and validation curves, and hyperparameter search drawn as a heat map.');

  const H = S.P01, P = (S.P02 = {});
  const cache = new Map();
  const once = (k, f) => { if (!cache.has(k)) cache.set(k, f()); return cache.get(k); };
  P.once = once;

  /**
   * A binary task scored on held-out rows: { key, title, posName, negName, y (1 = positive), score (P(positive)),
   * Xte (rows, when two-dimensional), nTrain, model } from a logistic regression fitted on the training rows.
   */
  P.SOURCES = {
    loans: { label: 'Loans: will it default?', pos: 'default', neg: 'repaid' },
    moons: { label: 'Two moons (a straight boundary)', pos: 'orange', neg: 'blue' },
    pumps: { label: 'Pumps, two sensors: rare faults (5%)', pos: 'fault', neg: 'normal' },
    cancer: { label: 'Breast cancer: malignant?', pos: 'malignant', neg: 'benign' },
    exam: { label: 'Exam: pass?', pos: 'pass', neg: 'fail' },
  };
  P.binary = (key) => once('bin:' + key, () => {
    const ML = S.ML, src = P.SOURCES[key];
    let X, y, model = 'logistic regression';
    if (key === 'loans') { const d = H.data('loans'); X = d.X.map((r) => r.slice(0, 5)); y = d.y; }
    else if (key === 'moons') { const d = S.Data.moons({ n: 160, noise: 0.25, seed: 4 }); X = d.X; y = d.y; }
    else if (key === 'pumps') { const d = H.data('pumps', { missing: 0, n: 800 }); X = d.X.map((r) => [r[0], r[3]]); y = d.y; }
    else if (key === 'cancer') { const d = H.data('breastCancer'); X = d.X.map((r) => r.slice(0, 10)); y = d.y.map((v) => 1 - v); }
    else { const d = H.data('exam', { n: 200 }); X = d.X; y = d.y; }
    const sp = ML.trainTestSplit(X.length, { testSize: 0.4, stratify: y, seed: 3 });
    const tr = sp.train.slice().sort((a, b) => a - b), te = sp.test.slice().sort((a, b) => a - b);
    const sc = H.scaler(tr.map((i) => X[i]), 'standard'), Z = sc.apply(X);
    const m = ML.LogisticRegression({ C: 1 }).fit(tr.map((i) => Z[i]), tr.map((i) => y[i]));
    const score = m.predictProba(te.map((i) => Z[i])).map((p) => p[1]);
    return { key, title: src.label, posName: src.pos, negName: src.neg, y: te.map((i) => y[i]), score, Xte: key === 'moons' ? te.map((i) => X[i]) : null, Xall: X, yall: y, train: tr, test: te, nTrain: tr.length, model, fit: m, scaler: sc };
  });
  P.sourceOptions = (keys) => (keys || Object.keys(P.SOURCES)).map((value) => ({ value, label: P.SOURCES[value].label }));

  /** counts at a threshold (positive means score ≥ t), straight from the metrics engine */
  P.at = (B, t) => S.Metrics.atThreshold(B.y, B.score, t, 1);

  /**
   * a binary confusion matrix: rows true (positive first), columns predicted (positive first), each cell its count
   * and its name; o: {prefix, x, y, cell, c: {tp, fp, fn, tn}, open: 'tp'|'fp'|'fn'|'tn', posName, negName}
   */
  P.cm2 = (o) => {
    const p = o.prefix || 'cm', s = o.cell || 84, g = 4, items = [], c = o.c, n = c.tp + c.fp + c.fn + c.tn || 1;
    const cells = [['tp', 0, 0, 'TP'], ['fn', 0, 1, 'FN'], ['fp', 1, 0, 'FP'], ['tn', 1, 1, 'TN']];
    for (const [k, i, j, nm] of cells) {
      const v = c[k], good = k === 'tp' || k === 'tn', t = Math.min(1, v / n * 2.2);
      const fill = `color-mix(in srgb, var(${good ? '--good' : '--bad'}) ${Math.round(12 + 60 * t)}%, var(--node))`;
      // o.info(k): what a click on the cell opens (its examples), so every frame's matrix can be explored
      const info = o.info ? o.info(k) : null;
      items.push(Object.assign({ key: p + ':' + k, kind: 'cell', x: o.x + j * (s + g), y: o.y + i * (s + g), w: s, h: s, rx: 6, lines: [String(v), nm], size: 15, cls: 'cm ' + (o.open === k ? 'open' : '') + (o.pick || info ? ' pickable' : ''), paint: { fill, text: t > 0.62 ? 'var(--st-bg)' : 'var(--text)' } }, info ? { info, pick: true } : {}));
    }
    const ls = 13;
    items.push(H.text(p + ':pt', o.x + s + g / 2, o.y - 30, 'predicted', 'tick', 12, 'middle'));
    items.push(H.text(p + ':p1', o.x + s / 2, o.y - 10, o.posName, 'hmname', ls, 'middle'), H.text(p + ':p0', o.x + s + g + s / 2, o.y - 10, o.negName, 'hmname', ls, 'middle'));
    items.push(H.text(p + ':t1', o.x - 8, o.y + s / 2 + 5, o.posName, 'hmname', ls, 'end'), H.text(p + ':t0', o.x - 8, o.y + s + g + s / 2 + 5, o.negName, 'hmname', ls, 'end'));
    items.push(H.text(p + ':tt', o.x - 8, o.y - 10, 'true ↓', 'tick', 12, 'end'));
    return items;
  };

  /** a strip of scores: one dot per example along a 0..1 axis, positives above the line and negatives below */
  P.strip = (ax, key, B, o) => {
    o = o || {};
    const items = [];
    B.score.forEach((s, i) => {
      const pos = B.y[i] === 1, j = H.jit(i);
      const yy = pos ? ax.o.y + ax.o.h * 0.3 + j * ax.o.h * 0.16 : ax.o.y + ax.o.h * 0.72 + j * ax.o.h * 0.16;
      const above = s >= (o.t === undefined ? 2 : o.t);
      const wrong = o.t !== undefined && above !== pos;
      items.push({ key: key + i, kind: 'dot', x: ax.X(s), y: +yy.toFixed(2), r: B.y.length > 300 ? 2.4 : 3.2, cls: 'sp c' + (pos ? 1 : 0) + (wrong && o.mark ? ' wrong' : ''), audit: 'overlap-ok' });
    });
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
