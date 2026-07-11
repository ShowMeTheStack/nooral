// The reference jobs for the topics of parts 1 and 2: for a topic and an input, the raw numbers (columns, splits,
// scores) that tests/ref/gen_p12.py hands to scikit-learn, numpy and scipy, so tests/claims-p01.mjs and
// tests/claims-p02.mjs can compare every quoted number with an independent computation.
// node tests/ref/p12_jobs.mjs writes tests/ref/p12/inputs.json; then ~/.nooral-venv/bin/python tests/ref/gen_p12.py.
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const nan = (v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v);
const clean = (a) => JSON.parse(JSON.stringify(a, (k, v) => nan(v)));

/** the jobs of one run: a list of {kind, ...data}; the claims checks read the results in the same order */
export function jobsFor(S, id, inp) {
  const H = S.P01, P = S.P02, ML = S.ML, jobs = [];
  const take = (a, idx) => idx.map((i) => a[i]);
  switch (id) {
    case 'data-describe': {
      const ds = H.data(inp.set), j = H.numCols(ds)[inp.col], v = H.col(ds, j).filter(H.isNum);
      jobs.push({ kind: 'describe', x: v, bins: inp.bins });
      break;
    }
    case 'data-correlation': {
      // the pair is rebuilt the way the topic builds it
      const PAIRS = { petal: ['iris', 2, 3], rent: ['rents', 3, -1], flowers: ['wildflowers', 2, 3], rain: ['harvest', 0, -1], wine: ['wine', 6, 5] };
      if (inp.pair === 'outlier') {
        const r = H.rng(11), x = [], y = [];
        for (let i = 0; i < 20; i++) { const u = r.range(0, 10); x.push(+u.toFixed(2)); y.push(+(5 - 0.3 * u + r.normal(0, 0.6)).toFixed(2)); }
        x.push(30); y.push(14);
        jobs.push({ kind: 'pearson', x, y }, { kind: 'pearson', x: x.slice(0, -1), y: y.slice(0, -1) });
      } else {
        const [set, a, b] = PAIRS[inp.pair], ds = H.data(set), col = (j) => (j < 0 ? ds.y : H.col(ds, j)), xs = col(a), ys = col(b), keep = xs.map((_, i) => H.isNum(xs[i]) && H.isNum(ys[i]));
        jobs.push({ kind: 'pearson', x: xs.filter((_, i) => keep[i]), y: ys.filter((_, i) => keep[i]) });
        const cols = H.numCols(ds).slice(0, 6), ok = ds.X.map((r) => cols.every((j) => H.isNum(r[j])));
        jobs.push({ kind: 'corrmat', X: ds.X.filter((_, i) => ok[i]).map((r) => cols.map((j) => r[j])) });
      }
      break;
    }
    case 'data-scatter-matrix': {
      const ds = H.data(inp.set), cols = H.numCols(ds).slice(inp.from, inp.from + inp.k), keep = ds.X.map((r) => cols.every((j) => H.isNum(r[j])));
      jobs.push({ kind: 'eta', X: ds.X.filter((_, i) => keep[i]).map((r) => cols.map((j) => r[j])), y: ds.y.filter((_, i) => keep[i]) });
      break;
    }
    case 'data-missing': {
      const GAP = { penguins: ['penguins', {}], pumps: ['pumps', { missing: 0.04, n: 400 }], harvest: ['harvest', { missing: 0.1, n: 200 }], rents: ['rents', { missing: 0.2, n: 150 }] };
      const [set, opt] = GAP[inp.set], ds = H.data(set, opt);
      jobs.push({ kind: 'impute', X: ds.X, cat: ds.features.map((f) => f.type === 'category') });
      break;
    }
    case 'data-outliers': {
      const OUT = { pumps: ['pumps', { missing: 0 }, 0], typo: ['wildflowers', {}, 2, true], years: ['loans', {}, 2], rain: ['buses', {}, 2] };
      const [set, opt, c, typo] = OUT[inp.data], v = H.col(H.data(set, opt), c).slice();
      if (typo) v[7] = +(v[7] * 100).toFixed(1);
      jobs.push({ kind: 'describe', x: v, bins: 10 });
      break;
    }
    case 'data-scaling': {
      const SETS = { wine: ['wine', 12, 10], loans: ['loans', 0, 1], outlier: ['wine', 12, 10, true] };
      const [set, a, b, out] = SETS[inp.data], ds = H.data(set), X = ds.X.map((r) => [r[a], r[b]]), y = ds.y;
      if (out) X[3] = [X[3][0] * 6, X[3][1]];
      const sp = ML.trainTestSplit(X.length, { testSize: 0.3, stratify: y, seed: 1 });
      jobs.push({ kind: 'scaling', Xtr: take(X, sp.train), ytr: take(y, sp.train), Xte: take(X, sp.test), yte: take(y, sp.test), k: inp.k, scaler: inp.scaler });
      break;
    }
    case 'data-encoding': {
      const ENC = { soil: ['harvest', 1, null], day: ['buses', 1, { n: 350 }], rooms: ['rents', 1, null, true] };
      const [set, j, opt, numeric] = ENC[inp.data], ds = H.data(set, opt || undefined), vals = H.col(ds, j);
      const cats = numeric ? [...new Set(vals)].sort((p, q) => p - q).map(String) : ds.features[j].categories;
      const ci = vals.map((v) => (numeric ? cats.indexOf(String(v)) : v));
      jobs.push({ kind: 'encoding', ci, y: ds.y, code: H.ordinalCodes(cats, inp.order), k: cats.length });
      break;
    }
    case 'data-binning': {
      const d = inp.data === 'rain' ? (() => { const q = H.data('harvest', { n: 150 }); return { X: q.X.map((r) => [r[0]]), y: q.y }; })() : S.Data.curve(inp.data === 'sine' ? { n: 120, shape: 'sine', noise: 0.25, seed: 5 } : { n: 120, shape: 'step', noise: 0.2, seed: 6 });
      const x = d.X.map((r) => r[0]), sp = ML.trainTestSplit(x.length, { testSize: 0.3, seed: 4 });
      jobs.push({ kind: 'binning', xtr: take(x, sp.train), ytr: take(d.y, sp.train), xte: take(x, sp.test), yte: take(d.y, sp.test), n: inp.n, strategy: inp.how });
      break;
    }
    case 'data-poly': {
      const C = { poly: { n: 40, shape: 'poly', noise: 0.25, seed: 9 }, sine: { n: 40, shape: 'sine', noise: 0.2, seed: 4 }, abs: { n: 40, shape: 'abs', noise: 0.2, seed: 2 } }[inp.data];
      const d = S.Data.curve(C), x = d.X.map((r) => r[0]), sp = ML.trainTestSplit(x.length, { testSize: 0.3, seed: 2 });
      jobs.push({ kind: 'poly', xtr: take(x, sp.train), ytr: take(d.y, sp.train), xte: take(x, sp.test), yte: take(d.y, sp.test) });
      break;
    }
    case 'data-interactions': {
      const d = inp.data === 'xor' ? S.Data.xor({ n: 200, noise: 0.04, seed: 3 }) : S.Data.circles({ n: 200, noise: 0.08, seed: 3 });
      const F = { plain: (p) => [p[0], p[1]], prod: (p) => [p[0], p[1], p[0] * p[1]], sq: (p) => [p[0], p[1], p[0] * p[0], p[1] * p[1]], all2: (p) => [p[0], p[1], p[0] * p[0], p[0] * p[1], p[1] * p[1]] };
      jobs.push({ kind: 'logreg', X: d.X.map(F.plain), y: d.y, C: 100 }, { kind: 'logreg', X: d.X.map(F[inp.feats]), y: d.y, C: 100 });
      break;
    }
    case 'data-select': {
      const r = H.rng(77);
      let X, y, reg;
      if (inp.data === 'rents') { const d = H.data('rents', { n: 200 }); X = d.X.map((q) => q.slice()); y = d.y; reg = true; } else { const d = H.data('breastCancer'); X = d.X.map((q) => q.slice(0, 10)); y = d.y; reg = false; }
      for (let k = 0; k < inp.noise; k++) X.forEach((q) => q.push(+r.normal(0, 1).toFixed(3)));
      jobs.push({ kind: 'select', X, y, reg, alpha: 8 });
      break;
    }
    case 'eval-confusion': case 'eval-prf': case 'eval-roc': case 'eval-pr': {
      const B = P.binary(inp.src);
      jobs.push({ kind: 'binary', y: B.y, s: B.score, t: inp.t === undefined ? 0.5 : inp.t });
      break;
    }
    case 'eval-auc-rank': {
      const B = P.binary(inp.src), k = inp.k;
      const pick = (cls) => { const idx = B.y.map((v, i) => (v === cls ? i : -1)).filter((i) => i >= 0).sort((a, b) => B.score[b] - B.score[a] || a - b); const step = idx.length / k; return Array.from({ length: k }, (_, j) => idx[Math.floor(j * step + step / 2)]); };
      const pos = pick(1), neg = pick(0);
      jobs.push({ kind: 'binary', y: pos.concat(neg).map((i) => B.y[i]), s: pos.concat(neg).map((i) => B.score[i]), t: 0.5 });
      break;
    }
    case 'eval-calibration': case 'eval-logloss': {
      const B = P.binary(inp.src);
      for (const m of ['logreg', 'nb', 'tree', 'knn']) jobs.push({ kind: 'binary', y: B.y, s: P.scoresOf(inp.src, m), t: 0.5, nBins: inp.bins || 5, strategy: inp.strategy || 'uniform', model: m });
      break;
    }
    case 'eval-confusion-multi': {
      const R = P.multi(inp.data, inp.model);
      jobs.push({ kind: 'multi', yt: R.yt, yp: R.yp, labels: R.classes.map((_, k) => k) });
      break;
    }
    case 'eval-balanced': {
      const R = P.imbalanced(inp.share);
      for (const [, yp] of R.models) jobs.push({ kind: 'multi', yt: R.yt, yp, labels: [0, 1] });
      break;
    }
    case 'eval-averages': {
      const R = P.AVG[inp.data].make();
      jobs.push({ kind: 'multi', yt: R.yt, yp: R.yp, labels: R.classes.map((_, k) => k) });
      break;
    }
    case 'eval-regression': {
      const R = P.regCase(inp.case);
      jobs.push({ kind: 'regression', y: R.yt, p: R.yp });
      break;
    }
    case 'eval-kfold': {
      const SETS = { flowers: () => H.data('wildflowers'), iris: () => H.data('iris'), moons: () => S.Data.moons({ n: 100, noise: 0.25, seed: 7 }) };
      const d = SETS[inp.data](), folds = inp.how === 'strat' ? ML.stratifiedKFold(d.y, inp.k, { shuffle: true, seed: 1 }) : ML.kFold(d.y.length, inp.k, { shuffle: inp.how === 'shuffle', seed: 1 });
      jobs.push({ kind: 'cv', X: d.X, y: d.y, folds: folds.map((f) => f.test), model: inp.model });
      break;
    }
    case 'eval-search': {
      const d = P.searchData();
      jobs.push({ kind: 'svcgrid', X: d.X, y: d.y, folds: ML.stratifiedKFold(d.y, 5).map((f) => f.test), C: [0.01, 0.1, 1, 10, 100], gamma: [0.01, 0.1, 1, 10, 100] });
      break;
    }
    case 'eval-validation-curve': {
      if (inp.knob === 'k') { const d = P.moonsVC(); jobs.push({ kind: 'vcknn', X: d.X, y: d.y, folds: ML.stratifiedKFold(d.y, 5, { shuffle: true, seed: 1 }).map((f) => f.test), values: [1, 3, 5, 9, 15, 25, 45, 75, 119] }); }
      break;
    }
  }
  return clean(jobs);
}

// the inputs every claims run uses: the default and every preset
export function presetInputs(S, t) { return [{}].concat((t.presets || []).map((p) => p.values)).map((v) => S.inputOf(t, Object.assign(S.inputOf(t), v))); }
export const refKey = (id, inp) => id + '|' + JSON.stringify(inp);

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { load } = await import('../load.mjs');
  const S = load(['core', 'stage', 'topics']);
  const out = {};
  for (const t of S.topics) {
    if (!/^p0[12]-/.test(t.family)) continue;
    for (const inp of presetInputs(S, t)) { const j = jobsFor(S, t.id, inp); if (j.length) out[refKey(t.id, inp)] = j; }
  }
  const dir = join(dirname(fileURLToPath(import.meta.url)), 'p12');
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'inputs.json'), JSON.stringify(out));
  console.log(Object.keys(out).length + ' cases written');
}
