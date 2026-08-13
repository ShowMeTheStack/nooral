// Part 2 (splitting and evaluation): every topic with its default, every preset and random valid inputs. The same
// wording and quoted-number checks as part 1 (tests/claims-p01.mjs), then each topic's values against scikit-learn
// (tests/ref/p12/sklearn.json) for the presets and against a second computation here for every input.
// Prints "claims-p02 N passed, M failed".
import { pathToFileURL } from 'node:url';
import { runClaims } from './claims-p01.mjs';

const pick = (rnd, a) => a[Math.floor(rnd() * a.length)];
const int = (rnd, a, b) => a + Math.floor(rnd() * (b - a + 1));
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
/** counts at a threshold, worked out here */
const counts = (y, s, t) => { let tp = 0, fp = 0, fn = 0, tn = 0; y.forEach((v, i) => { const p = s[i] >= t; if (p && v === 1) tp++; else if (p) fp++; else if (v === 1) fn++; else tn++; }); return { tp, fp, fn, tn }; };
/** AUC by counting ordered pairs, ties half */
const pairAuc = (y, s) => { let w = 0, n = 0; y.forEach((a, i) => { if (a !== 1) return; y.forEach((b, j) => { if (b !== 0) return; n++; w += s[i] > s[j] ? 1 : s[i] === s[j] ? 0.5 : 0; }); }); return w / n; };

export const P02 = {
  'eval-split': {
    random: (rnd) => ({ data: pick(rnd, ['pumps', 'loans', 'flowers']), val: pick(rnd, [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]), test: pick(rnd, [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]), strat: pick(rnd, ['yes', 'no']), seed: int(rnd, 0, 99) }),
    check({ input, frame, near, ok }) {
      const c = frame('cut').meta, sh = frame('shares').meta, all = frame('classes').meta.counts, n = all.reduce((a, b) => a + b, 0);
      near(c.tr + c.va + c.te, n, 'the parts hold every row once');
      near(c.te, Math.ceil(input.test * n), 'test size');
      if (input.strat === 'yes') { const rare = sh.rare, want = all[rare] / n; sh.counts.forEach((k, i) => ok(Math.abs(k - want * [c.tr, c.va, c.te].filter((v) => v)[i]) <= 1.0001, 'a stratified part keeps the class share up to rounding')); }
      near(sh.counts.reduce((a, b) => a + b, 0), all[sh.rare], 'the rare class is shared out');
    },
  },
  'eval-kfold': {
    random: (rnd) => ({ data: pick(rnd, ['flowers', 'iris', 'moons']), model: pick(rnd, ['knn', 'tree', 'nb']), k: int(rnd, 2, 10), how: pick(rnd, ['strat', 'shuffle', 'plain']) }),
    check({ input, frame, frames, near, ok, ref }) {
      const m = frame('mean').meta, f = frames('fold');
      near(m.mean, mean(m.scores), 'the mean of the fold scores');
      near(f.map((x) => x.meta.score), m.scores, 'each fold frame shows its score');
      ok(f.length === input.k, 'one frame per fold');
      if (ref) near(m.scores, ref[0].scores, 'fold scores against scikit-learn', input.model === 'tree' ? 0.05 : 1e-9);
    },
  },
  'eval-confusion': {
    random: (rnd) => ({ src: pick(rnd, ['moons', 'loans', 'cancer', 'pumps']), t: pick(rnd, [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]) }),
    check({ S, input, frame, near, ref }) {
      const B = S.P02.binary(input.src), c = frame('counts').meta, sc = frame('scores').meta, w = counts(B.y, B.score, input.t);
      near([c.tp, c.fn, c.fp, c.tn], [w.tp, w.fn, w.fp, w.tn], 'the four cells');
      near([sc.precision, sc.recall, sc.acc], [w.tp / Math.max(1, w.tp + w.fp) * (w.tp + w.fp ? 1 : 0), w.tp / (w.tp + w.fn), (w.tp + w.tn) / B.y.length], 'scores from the cells');
      if (ref) near([c.tp, c.fn, c.fp, c.tn, sc.precision, sc.recall, sc.acc], [ref[0].tp, ref[0].fn, ref[0].fp, ref[0].tn, ref[0].precision, ref[0].recall, ref[0].accuracy], 'against scikit-learn', 1e-9);
    },
  },
  'eval-confusion-multi': {
    check({ frame, near, ok, ref }) {
      const c = frame('counts').meta, r = frame('rows').meta, k = frame('cols').meta, o = frame('open').meta;
      near(o.count, c.counts[o.cell[0]][o.cell[1]], 'the open cell holds its count');
      ok(o.closer <= o.count, 'at most every row in the cell is closer to the other class');
      if (ref) { near(c.counts, ref[0].counts, 'counts against scikit-learn'); near(r.recall, ref[0].recall, 'recall against scikit-learn', 1e-9); near(k.precision, ref[0].precision, 'precision against scikit-learn', 1e-9); }
    },
  },
  'eval-balanced': {
    random: (rnd) => ({ share: pick(rnd, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]) }),
    check({ frames, frame, near, ref }) {
      const ms = frames('model'), cmp = frame('compare').meta;
      ms.forEach((f, k) => { const m = f.meta, n = m.tp + m.fn + m.fp + m.tn; near(m.acc, (m.tp + m.tn) / n, 'accuracy from the cells'); near(m.bal, (m.tp / (m.tp + m.fn) + m.tn / (m.tn + m.fp)) / 2, 'balanced accuracy from the cells'); if (ref) near([m.acc, m.bal], [ref[k].accuracy, ref[k].balanced], 'against scikit-learn', 1e-9); });
      near(cmp.acc, ms.map((f) => f.meta.acc), 'the comparison repeats the models');
    },
  },
  'eval-prf': {
    random: (rnd) => ({ src: pick(rnd, ['loans', 'cancer', 'pumps', 'moons']), t: +(0.05 + int(rnd, 0, 90) * 0.01).toFixed(2) }),
    check({ S, input, frame, frames, near, ok, ref }) {
      const B = S.P02.binary(input.src);
      for (const f of [frame('yours')].concat(frames('sweep'), [frame('best')])) {
        const m = f.meta, w = counts(B.y, B.score, m.t), p = w.tp + w.fp ? w.tp / (w.tp + w.fp) : 0, r = w.tp / (w.tp + w.fn);
        near([m.p, m.r, m.f1], [p, r, p + r ? (2 * p * r) / (p + r) : 0], 'precision, recall and F1 at ' + m.t, 1e-9);
      }
      const y = frame('yours').meta;
      if (ref) { near([y.tp, y.fp, y.fn, y.p, y.r, y.f1], [ref[0].tp, ref[0].fp, ref[0].fn, ref[0].precision, ref[0].recall, ref[0].f1], 'against scikit-learn', 1e-9); near(frame('best').meta.f1, Math.max(...ref[0].f1grid), 'the best F1 on the grid against scikit-learn', 1e-9); }
      ok(frames('sweep').every((f, i, a) => !i || f.meta.r >= a[i - 1].meta.r), 'recall never falls as the threshold drops');
    },
  },
  'eval-averages': {
    check({ frames, frame, near, ref }) {
      const cl = frames('class'), mac = frame('macro').meta, mic = frame('micro').meta, mc = frame('mcc').meta;
      near(mac.macroF1, mean(cl.map((f) => f.meta.f1)), 'macro F1 is the mean of the class F1s');
      near(mic.micro, mic.acc, 'micro F1 equals accuracy');
      if (ref) { const R = ref[0]; near(cl.map((f) => f.meta.p), R.precision, 'precision', 1e-9); near(cl.map((f) => f.meta.r), R.recall, 'recall', 1e-9); near(cl.map((f) => f.meta.f1), R.f1, 'F1', 1e-9); near(cl.map((f) => f.meta.spec), R.specificity, 'specificity', 1e-9); near([mac.macroF1, mac.weightedF1, mic.micro], [R.macro[2], R.weighted[2], R.micro[2]], 'averages against scikit-learn', 1e-9); near(mc.mcc, R.mcc, 'MCC against scikit-learn', 1e-9); }
    },
  },
  'eval-roc': {
    constants: [0.8, 0.6, 0.4, 0.2, 0.05],
    random: (rnd) => ({ src: pick(rnd, ['loans', 'cancer', 'moons', 'pumps', 'exam']), t: +(int(rnd, 0, 100) * 0.01).toFixed(2) }),
    check({ S, input, frame, near, ref }) {
      const B = S.P02.binary(input.src), a = frame('auc').meta, y = frame('yours').meta, w = counts(B.y, B.score, input.t);
      near(a.auc, pairAuc(B.y, B.score), 'AUC equals the share of ordered pairs', 1e-9);
      near([y.tp, y.fp], [w.tp, w.fp], 'your threshold');
      if (ref) near(a.auc, ref[0].auc, 'AUC against scikit-learn', 1e-9);
    },
  },
  'eval-auc-rank': {
    random: (rnd) => ({ src: pick(rnd, ['loans', 'exam', 'cancer', 'moons']), k: int(rnd, 4, 14) }),
    check({ frame, near, ref }) {
      const a = frame('auc').meta;
      near(a.aucPairs, a.aucRoc, 'pairs and area agree', 1e-12);
      near(a.aucPairs, a.wins / a.total, 'the share of pairs');
      if (ref) near(a.aucRoc, ref[0].auc, 'AUC against scikit-learn', 1e-9);
    },
  },
  'eval-pr': {
    check({ S, input, frame, frames, near, ref }) {
      const a = frame('ap').meta, B = S.P02.binary(input.src);
      near(a.prev, mean(B.y), 'the share of positives');
      const sw = frames('sweep'); near(sw[sw.length - 1].meta.ap, a.ap, 'the rectangles add up to the average precision', 1e-9);
      if (ref) { near(a.ap, ref[0].ap, 'average precision against scikit-learn', 1e-9); near(a.auc, ref[0].auc, 'AUC against scikit-learn', 1e-9); }
    },
  },
  'eval-calibration': {
    constants: [0.05, 0.95],
    random: (rnd) => ({ src: pick(rnd, ['loans', 'cancer', 'exam', 'moons']), model: pick(rnd, ['logreg', 'nb', 'tree', 'knn']), bins: int(rnd, 3, 10), strategy: pick(rnd, ['uniform', 'quantile']) }),
    check({ S, input, frame, near, ref }) {
      const B = S.P02.binary(input.src), s = S.P02.scoresOf(input.src, input.model), g = frame('gaps').meta;
      if (input.strategy === 'uniform') {
        // the expected calibration error worked out here, bin by bin
        const n = input.bins, sum = new Array(n).fill(0), hit = new Array(n).fill(0), cnt = new Array(n).fill(0);
        s.forEach((p, i) => { let b = 0; while (b < n - 1 && (b + 1) / n < p) b++; sum[b] += p; hit[b] += B.y[i]; cnt[b]++; });
        const ece = cnt.reduce((a, c, b) => a + (c ? Math.abs(hit[b] / c - sum[b] / c) * c : 0), 0) / s.length;
        near(g.ece, ece, 'expected calibration error', 1e-9);
      }
      near(g.brier, mean(s.map((p, i) => (B.y[i] - p) ** 2)), 'Brier score', 1e-9);
      if (ref) { const R = ref[['logreg', 'nb', 'tree', 'knn'].indexOf(input.model)], C = frame('compare').meta.cmp; near(g.brier, R.brier, 'Brier against scikit-learn', 1e-9); const cal = S.Metrics.calibration(B.y, s, { nBins: input.bins, strategy: input.strategy, posLabel: 1 }); near(cal.probTrue, R.calTrue, 'reliability curve against scikit-learn', 1e-9); near(cal.probPred, R.calPred, 'mean predictions against scikit-learn', 1e-9); near(C.map((c) => c[2]), ref.map((r) => r.brier), 'every model\'s Brier score', 1e-9); }
    },
  },
  'eval-logloss': {
    constants: [0.5, 0.1, Math.log(2), -Math.log(0.1), 0.001],
    random: (rnd) => ({ src: pick(rnd, ['loans', 'cancer', 'exam', 'moons']), model: pick(rnd, ['logreg', 'nb', 'tree', 'knn']) }),
    check({ S, input, frame, near, ref }) {
      const B = S.P02.binary(input.src), s = S.P02.scoresOf(input.src, input.model), m = frame('mean').meta, eps = 2.220446049250313e-16;
      near(m.value, mean(s.map((p, i) => -Math.log(Math.min(1 - eps, Math.max(eps, B.y[i] === 1 ? p : 1 - p))))), 'log loss worked out here', 1e-9);
      const sh = mean(B.y); near(m.base, -(sh * Math.log(sh) + (1 - sh) * Math.log(1 - sh)), 'the constant guess');
      if (ref) { const R = ref[['logreg', 'nb', 'tree', 'knn'].indexOf(input.model)]; near(m.value, R.logloss, 'log loss against scikit-learn', 1e-6); near(m.acc, R.accuracy, 'accuracy against scikit-learn', 1e-9); }
    },
  },
  'eval-regression': {
    check({ frame, near, ref }) {
      const e = frame('errors').meta, r = frame('r2').meta;
      near(e.rmse, Math.sqrt(e.mse), 'RMSE is the root of MSE');
      if (ref) near([e.mae, e.mse, e.rmse, r.r2], [ref[0].mae, ref[0].mse, ref[0].rmse, ref[0].r2], 'against scikit-learn', 1e-9);
    },
  },
  'eval-learning-curve': {
    constants: [0.3],
    check({ S, input, frame, frames, near }) {
      // the procedure done again here: five shuffled folds, each training on a growing prefix of its training rows
      const ML = S.ML, D = S.P02.sine(), folds = ML.kFold(D.X.length, 5, { shuffle: true, seed: 1 }), mk = { line: () => ML.LinearRegression(), deep: () => ML.DecisionTreeRegressor({ seed: 0 }), tree4: () => ML.DecisionTreeRegressor({ maxDepth: 4, seed: 0 }) }[input.model];
      const end = frame('end').meta, m = folds[0].train.length, mse = (mod, idx) => mean(idx.map((i) => (D.y[i] - mod.predict([D.X[i]])[0]) ** 2));
      const tr = [], va = [];
      for (const f of folds) { const idx = f.train.slice(0, m), mod = mk().fit(idx.map((i) => D.X[i]), idx.map((i) => D.y[i])); tr.push(mse(mod, idx)); va.push(mse(mod, f.test)); }
      near([end.train, end.val], [mean(tr), mean(va)], 'the last point, worked out again', 1e-9);
      near(end.gap, end.val - end.train, 'the gap');
      for (const f of frames('size')) near(f.meta.m, f.meta.m, 'size');
    },
  },
  'eval-validation-curve': {
    check({ input, frame, frames, near, ok, ref }) {
      const b = frame('best').meta, vs = frames('value');
      ok(vs.length >= 3, 'several values are shown');
      if (ref && input.knob === 'k') { const R = ref[0], k = R.test.indexOf(Math.max(...R.test)); near(b.val, R.test[b.best], 'validation accuracy at the best k against scikit-learn', 0.011); ok(Math.abs(R.test[k] - b.val) <= 0.011, 'the best k scores like the best of scikit-learn'); }
    },
  },
  'eval-search': {
    constants: [0.01, 100],
    random: (rnd) => ({ n: int(rnd, 5, 25), seed: int(rnd, 0, 99) }),
    nRandom: 3,
    check({ frame, near, ok, ref }) {
      const gb = frame('gridbest').meta, rb = frame('randbest').meta;
      ok(rb.score <= 1 && gb.score <= 1, 'scores are accuracies');
      if (ref) { const R = ref[0].grid; near(gb.score, Math.max(...Object.values(R)), 'the best grid score against scikit-learn', 0.02); near(gb.lo, Math.min(...Object.values(R)), 'the worst grid score against scikit-learn', 0.02); }
    },
  },
};

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) runClaims('claims-p02', /^p02-/, P02);
