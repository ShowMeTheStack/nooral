// Part 1, encoding and binning. A category column turned into ordinal codes (one number, which forces an order and
// equal steps on a linear model) or into one-hot columns (one 0/1 column per category, so each gets its own effect),
// with the linear fit of each drawn over the data; and a number cut into bins (equal widths or equal counts) whose
// means make a step-shaped fit beside the straight line.
(function (S) {
  'use strict';
  const H = S.P01, C = () => S.Kit.curve, DT = () => S.Kit.dt, ML = () => S.ML;
  const r2 = (v) => Math.round(v * 100) / 100;

  // ---- encoders ----
  /** the codes of an ordinal encoding: category index → code, in the order given ('alpha' sorts the names) */
  H.ordinalCodes = (cats, order) => { const idx = cats.map((_, i) => i); if (order === 'alpha') idx.sort((a, b) => (cats[a] < cats[b] ? -1 : cats[a] > cats[b] ? 1 : 0)); const code = new Array(cats.length); idx.forEach((c, k) => { code[c] = k; }); return code; };
  H.oneHot = (v, k, dropFirst) => { const r = new Array(k).fill(0); r[v] = 1; return dropFirst ? r.slice(1) : r; };

  const ENC = {
    soil: { set: 'harvest', j: 1, label: 'Harvest: soil → yield' },
    day: { set: 'buses', j: 1, label: 'Buses: day → delay', opt: { n: 350 } },
    rooms: { set: 'rents', j: 1, label: 'Rents: rooms (ordered) → rent', numeric: true },
  };
  S.topic({
    id: 'data-encoding', family: 'p01-transform', order: 20, title: 'One-hot and ordinal encoding',
    blurb: 'Models need numbers, so category names must be encoded. Ordinal encoding gives each category one code (0, 1, 2, …): compact, but a linear model then believes the categories lie on a line, in that order, equally spaced. One-hot encoding gives each category its own 0/1 column, so each can have its own effect, at the cost of one column per category. Ordinal codes suit categories with a real order (sizes, grades); one-hot suits names.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Column', def: 'soil', options: Object.entries(ENC).map(([value, e]) => ({ value, label: e.label })) },
      { id: 'order', kind: 'choice', label: 'Ordinal codes', def: 'alpha', options: [{ value: 'alpha', label: 'in alphabetical order' }, { value: 'given', label: 'in the natural order' }] },
    ],
    presets: [
      { label: 'Soil, codes alphabetical', values: { data: 'soil', order: 'alpha' } },
      { label: 'Soil, codes in a lucky order', values: { data: 'soil', order: 'given' } },
      { label: 'Days of the week', values: { data: 'day', order: 'given' } },
      { label: 'Rooms: a real order', values: { data: 'rooms', order: 'given' } },
    ],
    counters: [{ id: 'cols', label: 'columns used' }],
    run(inp) {
      const e = ENC[inp.data], ds = H.data(e.set, e.opt), f = ds.features[e.j], y = ds.y;
      const vals = H.col(ds, e.j);
      const cats = e.numeric ? [...new Set(vals)].sort((a, b) => a - b).map(String) : f.categories;
      const catOf = (v) => (e.numeric ? cats.indexOf(String(v)) : v), ci = vals.map(catOf), k = cats.length;
      const code = H.ordinalCodes(cats, inp.order), codeOf = (i) => code[ci[i]];
      const order = cats.map((_, c) => c).sort((a, b) => code[a] - code[b]); // categories in code order
      const tname = H.short(ds.target), name = H.short(f);
      const ml = ML();
      const ord = ml.LinearRegression().fit(ci.map((c) => [code[c]]), y);
      const predOrd = cats.map((_, c) => ord.predict([[code[c]]])[0]);
      const oh = ml.LinearRegression().fit(ci.map((c) => H.oneHot(c, k, true)), y);
      const predOh = cats.map((_, c) => oh.predict([H.oneHot(c, k, true)])[0]);
      const means = cats.map((_, c) => H.mean(y.filter((_, i) => ci[i] === c))), counts = cats.map((_, c) => ci.filter((x) => x === c).length);
      const mse = (p) => H.mean(y.map((v, i) => (v - p[ci[i]]) ** 2));
      const mOrd = mse(predOrd), mOh = mse(predOh), mNone = H.mean(y.map((v) => (v - H.mean(y)) ** 2));
      const F = new S.Frames(['cols']), view = H.box(-14, -8, 790, 470);
      const yt = [Math.min(...y), Math.max(...y)], [y0, y1] = H.span(yt[0], yt[1], 0.05);
      const ohW = 84 + 64 + 6 + order.reduce((a, c) => a + (k > 5 ? 26 : Math.max(44, S.textWidth(cats[c], 12, { bold: true }) + 12)) + 2, 0);
      const axX = Math.max(330, ohW + 56);
      const ax = C().axes({ prefix: 'ax', x: axX, y: 50, w: 770 - axX, h: 330, xr: [-0.5, k - 0.5], yr: [y0, y1], xticks: order.map((_, p) => p), fmtX: (p) => (k > 5 ? cats[order[Math.round(p)]].slice(0, 3) : cats[order[Math.round(p)]]), xname: name + ' (in code order)', yname: tname, tickSize: 12 });
      const dots = y.map((v, i) => ({ key: 'p' + i, kind: 'dot', x: r2(ax.X(code[ci[i]] + H.jit(i) * 0.28)), y: ax.Y(v), r: 2.6, cls: 'pt0', audit: 'overlap-ok' }));
      const showI = ds.X.map((_, i) => i).slice(0, 8);
      const tableRows = (mode) => showI.map((i) => [cats[ci[i]]].concat(mode === 'ord' ? [String(codeOf(i))] : mode === 'oh' ? H.oneHot(ci[i], k).map(String) : [], [H.g(y[i], 3)]));
      const tcols = (mode) => [{ name, w: 84, cls: 'cat' }].concat(mode === 'ord' ? [{ name: 'code', w: 56, cls: 'num' }] : mode === 'oh' ? order.map((c) => ({ name: k > 5 ? cats[c].slice(0, 3) : cats[c], w: k > 5 ? 26 : Math.max(44, S.textWidth(cats[c], 12, { bold: true }) + 12), cls: 'new', size: 12 })) : [], [{ name: tname, w: 64, cls: 'lab' }]);
      const table = (mode) => { const cols = tcols(mode); const rows = tableRows(mode).map((r) => (mode === 'oh' ? [r[0]].concat(order.map((c) => r[1 + c]), [r[r.length - 1]]) : r)); return DT().table({ prefix: 'tb', x: 0, y: 50, cols, rows, rowH: 28, size: 13, gap: 2, cls: (r, c) => (mode === 'oh' && c >= 1 && c <= k && rows[r][c] === '1' ? 'one' : '') }).items; };
      const title = H.title('tt', 0, 6, ds.doc.title + ': ' + name + ' and ' + tname);
      const fitOrd = C().line(ax, 'ford', [[-0.5, ord.intercept + ord.coef[0] * -0.5], [k - 0.5, ord.intercept + ord.coef[0] * (k - 0.5)]], 'bad');
      const fitOh = order.map((c, p) => C().line(ax, 'foh' + c, [[p - 0.36, predOh[c]], [p + 0.36, predOh[c]]], 'good'));
      const note = (lines) => lines.map((l, i) => H.text('nt' + i, 0, 340 + i * 24, l[0], l[1] || 'strong', 14));
      F.set('cols', 0);
      F.add({ items: [title, ...table('raw'), ...ax.items, ...dots], say: `"${name}" is a category with ${k} values: ${H.list(cats.map((c, i) => c + ' (' + counts[i] + ' rows)'))}. Each dot is one row's ${tname}, grouped by its ${name}.`, meta: { phase: 'raw', k, counts, means }, view });
      F.set('cols', 1);
      F.add({ items: [title, ...table('ord'), ...ax.items, ...dots, ...note([['ordinal: ' + order.map((c) => cats[c] + ' → ' + code[c]).slice(0, 4).join(', ') + (k > 4 ? ', …' : ''), 'act']])], say: `Ordinal encoding replaces each name by one code: ${H.list(order.map((c) => cats[c] + ' ' + code[c]))}` + (inp.order === 'alpha' ? ', in alphabetical order, as an encoder that knows nothing about the data would.' : ', in the natural order.') + ' One column of numbers.', meta: { phase: 'ordinal', code }, view });
      F.add({ items: [title, ...table('ord'), ...ax.items, ...dots, fitOrd, ...note([['a line in the code: ' + tname + ' ≈ ' + H.g(ord.intercept, 3) + (ord.coef[0] < 0 ? ' − ' : ' + ') + H.g(Math.abs(ord.coef[0]), 3) + ' × code', 'bad'], ['mean squared error ' + H.g(mOrd, 3), 'strong']])], say: `A linear model on the code can only draw a straight line through the categories: ${tname} ≈ ${H.g(ord.intercept, 3)} ${ord.coef[0] < 0 ? '−' : '+'} ${H.g(Math.abs(ord.coef[0]), 3)} × code. It must change by the same step from each category to the next, in code order. Mean squared error ${H.g(mOrd, 3)}.`, meta: { phase: 'ordfit', coef: ord.coef[0], intercept: ord.intercept, mse: mOrd, pred: predOrd }, view });
      F.set('cols', k);
      F.add({ items: [title, ...table('oh'), ...ax.items, ...dots, fitOrd, ...note([['one-hot: ' + k + ' columns, exactly one 1 per row', 'good'], ['for a linear model one is dropped:', 'tick'], ['the others say how far from ' + cats[0], 'tick']])], say: `One-hot encoding makes ${k} columns of 0 and 1, one per category, with a single 1 in each row. They always add up to 1, the same as the intercept, so a linear model drops one (here ${cats[0]}) and measures the others against it.`, meta: { phase: 'onehot', k }, view });
      F.add({ items: [title, ...table('oh'), ...ax.items, ...dots, fitOrd, ...fitOh, ...note([['one-hot fit: each category its own level', 'good'], ['MSE one-hot ' + H.g(mOh, 3) + ', ordinal ' + H.g(mOrd, 3), 'strong'], ['predicting the overall mean: ' + H.g(mNone, 3), 'tick']])], say: `With one-hot columns the model gives each category its own level, which is just that category's mean: ${H.list(order.map((c) => cats[c] + ' ' + H.g(predOh[c], 3)))}. Mean squared error ${H.g(mOh, 3)} against ${H.g(mOrd, 3)} for the ordinal line` + (mOrd - mOh < 0.03 * mOh ? ': here the ordinal line fits nearly as well, with one column instead of ' + (k - 1) + '.' : '.'), meta: { phase: 'ohfit', pred: predOh, mse: mOh, mseOrd: mOrd, mseNone: mNone, k1: k - 1 }, view });
      F.list.forEach((f) => { f.meta.ctx = { names: cats, n: y.length }; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Encodings', text: ['ordinal: category → 0, 1, 2, … (one column)', '  a linear model: one slope, equal steps', 'one-hot: one 0/1 column per category', '  k columns sum to 1, so drop one with an intercept', '  a linear model: each category its own mean', 'trees can use ordinal codes; they split anywhere'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'onehot', pick: 'option', ask: 'Which encoding will fit these rows with the smaller error?', options: ['one-hot', 'ordinal', 'they tie'], answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'ohfit').meta; return m.mse < m.mseOrd - 1e-9 ? 'one-hot' : m.mse > m.mseOrd + 1e-9 ? 'ordinal' : 'they tie'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'ohfit').say },
    ],
    quiz: [
      { q: 'Colour ∈ {red, green, blue} is ordinal-encoded as 0, 1, 2 for a linear model. What does the model assume?', options: ['green lies exactly halfway between red and blue', 'nothing at all', 'the colours are independent', 'blue is the most common'], answer: 0, why: 'One slope times the code makes equal steps in code order.' },
      { q: 'How many one-hot columns does a category with 7 values need (no column dropped)?', options: ['7', '1', '3', '6'], answer: 0, why: 'One column per value.' },
      { q: 'Why drop one one-hot column for a linear model with an intercept?', options: ['the columns add up to 1, duplicating the intercept', 'to save memory only', 'the first category is unimportant', 'it improves accuracy a lot'], answer: 0, why: 'Exact collinearity makes the coefficients not unique.' },
      { q: 'T-shirt sizes S, M, L, XL are best encoded as…', options: ['ordinal codes in size order', 'one-hot only', 'alphabetical codes', 'they cannot be encoded'], answer: 0, why: 'They have a real order, so one ordered column keeps it compactly.' },
      { q: 'A column has 5000 distinct postcodes. One-hot encoding it…', options: ['makes 5000 mostly-zero columns', 'makes one column', 'is impossible', 'removes the information'], answer: 0, why: 'High-cardinality categories make one-hot very wide; grouping or other encodings help.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Binning
  // ---------------------------------------------------------------------------------------------------------------
  /** bin edges: 'uniform' (equal widths over [min, max]) or 'quantile' (equal counts) */
  H.binEdges = (x, n, how) => {
    const lo = Math.min(...x), hi = Math.max(...x);
    if (how === 'quantile') { const s = H.sorted(x); return Array.from({ length: n + 1 }, (_, i) => H.quantile(s, i / n)); }
    return Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
  };
  /** which bin a value falls in: the inner edges decide, values beyond the ends join the end bins */
  H.binOf = (v, edges) => { let b = 0; while (b < edges.length - 2 && v >= edges[b + 1]) b++; return b; };
  const BINS = {
    sine: { label: 'A noisy sine curve', make: () => S.Data.curve({ n: 120, shape: 'sine', noise: 0.25, seed: 5 }), xn: 'x', yn: 'y' },
    rain: { label: 'Harvest: rain → yield', make: () => { const d = H.data('harvest', { n: 150 }); return { X: d.X.map((r) => [r[0]]), y: d.y }; }, xn: 'rain (mm)', yn: 'yield (t/ha)' },
    step: { label: 'A step function', make: () => S.Data.curve({ n: 120, shape: 'step', noise: 0.2, seed: 6 }), xn: 'x', yn: 'y' },
  };
  S.topic({
    id: 'data-binning', family: 'p01-transform', order: 30, title: 'Binning',
    blurb: 'Binning cuts a number into ranges and replaces it by the range it falls in. Equal-width bins split the axis evenly, so some may hold few rows; quantile bins hold equal numbers of rows, so their widths vary. One-hot encode the bins and a linear model fits a separate level in each: a step-shaped curve that can follow bends a straight line cannot. More bins follow the data more closely, and its noise too.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'sine', options: Object.entries(BINS).map(([value, b]) => ({ value, label: b.label })) },
      { id: 'how', kind: 'choice', label: 'Bins', def: 'uniform', options: [{ value: 'uniform', label: 'equal widths' }, { value: 'quantile', label: 'equal counts (quantiles)' }] },
      { id: 'n', kind: 'slider', label: 'Number of bins', def: 6, min: 2, max: 24, step: 1 },
    ],
    presets: [
      { label: 'Sine, 6 equal-width bins', values: { data: 'sine', how: 'uniform', n: 6 } },
      { label: 'Sine, 6 quantile bins', values: { data: 'sine', how: 'quantile', n: 6 } },
      { label: 'Many bins: 24', values: { data: 'sine', how: 'quantile', n: 24 } },
      { label: 'Rain: best in the middle', values: { data: 'rain', how: 'quantile', n: 5 } },
      { label: 'A step, 3 bins', values: { data: 'step', how: 'uniform', n: 3 } },
    ],
    counters: [{ id: 'bins', label: 'bins' }],
    run(inp) {
      const B = BINS[inp.data], d = B.make(), x = d.X.map((r) => r[0]), y = d.y, n = x.length, ml = ML();
      const sp = ml.trainTestSplit(n, { testSize: 0.3, seed: 4 }), tr = sp.train, te = sp.test;
      const xtr = tr.map((i) => x[i]), ytr = tr.map((i) => y[i]);
      const edges = H.binEdges(xtr, inp.n, inp.how), nb = inp.n;
      const binTr = xtr.map((v) => H.binOf(v, edges)), counts = Array.from({ length: nb }, (_, b) => binTr.filter((q) => q === b).length);
      const means = counts.map((c, b) => (c ? H.mean(ytr.filter((_, i) => binTr[i] === b)) : H.mean(ytr)));
      const line = ml.LinearRegression().fit(xtr.map((v) => [v]), ytr);
      const pl = (v) => line.intercept + line.coef[0] * v, pb = (v) => means[H.binOf(v, edges)];
      const mse = (idx, p) => H.mean(idx.map((i) => (y[i] - p(x[i])) ** 2));
      const res = { lineTr: mse(tr, pl), lineTe: mse(te, pl), binTr: mse(tr, pb), binTe: mse(te, pb) };
      const F = new S.Frames(['bins']), view = H.box(-14, -8, 790, 470);
      const [x0, x1] = H.span(Math.min(...x), Math.max(...x), 0.02), [y0, y1] = H.span(Math.min(...y), Math.max(...y), 0.06);
      const ax = C().axes({ prefix: 'ax', x: 56, y: 50, w: 480, h: 300, xr: [x0, x1], yr: [y0, y1], xname: B.xn, yname: B.yn, tickSize: 12 });
      const dots = (cls) => x.map((v, i) => ({ key: 'p' + i, kind: 'dot', x: ax.X(v), y: ax.Y(y[i]), r: 3, cls: cls(i), audit: 'overlap-ok' }));
      const trSet = new Set(tr), base = (i) => (trSet.has(i) ? 'pt0' : 'sp c1 test');
      const edgeLines = () => edges.slice(1, -1).map((e, k) => ({ key: 'e' + k, kind: 'path', d: `M${ax.X(e)},${ax.o.y} V${ax.o.y + ax.o.h}`, cls: 'guide act', audit: 'ignore', layer: 'back', z: 3 }));
      const countLabels = () => (nb <= 12 ? counts.map((c, b) => H.text('cn' + b, r2((ax.X(Math.max(x0, edges[b])) + ax.X(Math.min(x1, edges[b + 1]))) / 2), ax.o.y + 14, String(c), 'tick', 12, 'middle')) : []);
      const side = (lines) => lines.map((l, k) => H.text('sd' + k, 560, 70 + k * 26, l[0], l[1] || 'strong', 14));
      const title = H.title('tt', 0, 6, B.label + ': ' + tr.length + ' training rows, ' + te.length + ' test rows (orange)');
      const lineIt = C().line(ax, 'fl', [[x0, pl(x0)], [x1, pl(x1)]], 'bad');
      const steps = () => { const pts = []; for (let b = 0; b < nb; b++) { const a = Math.max(x0, b === 0 ? x0 : edges[b]), z = Math.min(x1, b === nb - 1 ? x1 : edges[b + 1]); pts.push([a, means[b]], [z, means[b]]); } return C().line(ax, 'fb', pts, 'good'); };
      F.add({ items: [title, ...ax.items, ...dots(base), lineIt, ...side([['straight line on x', 'bad'], ['training MSE ' + H.g(res.lineTr, 3), 'strong'], ['test MSE ' + H.g(res.lineTe, 3), 'strong']])], say: `${n} points; a straight line fitted to the ${tr.length} training rows misses the bends: training MSE ${H.g(res.lineTr, 3)}, test MSE ${H.g(res.lineTe, 3)}.`, meta: { phase: 'line', lineTr: res.lineTr, lineTe: res.lineTe }, view });
      F.set('bins', nb);
      const widths = edges.slice(1).map((e, b) => e - edges[b]);
      F.add({ items: [title, ...ax.items, ...edgeLines(), ...countLabels(), ...dots(base), ...side([[nb + ' ' + (inp.how === 'quantile' ? 'quantile' : 'equal-width') + ' bins', 'act'], ['training rows per bin:', 'tick'], [nb <= 10 ? counts.join(', ') : 'from ' + Math.min(...counts) + ' to ' + Math.max(...counts), 'mono']])], say: inp.how === 'quantile' ? `Quantile edges from the training rows put about ${H.g(tr.length / nb, 2)} rows in each of the ${nb} bins (${counts.join(', ')}); the bins are as narrow as ${H.g(Math.min(...widths), 3)} and as wide as ${H.g(Math.max(...widths), 3)}.` : `Equal-width edges cut the training range into ${nb} bins of width ${H.g(widths[0], 3)}; they hold ${counts.join(', ')} rows, from ${Math.min(...counts)} to ${Math.max(...counts)}.`, meta: { phase: 'edges', edges, counts }, view });
      F.add({ items: [title, ...ax.items, ...edgeLines(), ...dots(base), steps(), lineIt, ...side([['one level per bin (its mean)', 'good'], ['training MSE ' + H.g(res.binTr, 3), 'strong'], ['test MSE ' + H.g(res.binTe, 3), 'strong'], ['line: ' + H.g(res.lineTr, 3) + ' and ' + H.g(res.lineTe, 3), 'bad']])], say: `One-hot encode the bin and a linear model fits one level per bin, the mean of its training rows (${H.list(means.slice(0, 6).map((m) => H.g(m, 3)))}${nb > 6 ? ', …' : ''}). The steps follow the curve: training MSE ${H.g(res.binTr, 3)} and test MSE ${H.g(res.binTe, 3)}, against ${H.g(res.lineTr, 3)} and ${H.g(res.lineTe, 3)} for the line.` + (res.binTe > res.binTr * 1.5 && nb >= 12 ? ' With this many bins each level rests on few rows, so the test error is well above the training error.' : ''), meta: { phase: 'steps', means, binTr: res.binTr, binTe: res.binTe }, view });
      const ctx = { n, ntr: tr.length, perBin: tr.length / nb, widths: [widths[0], Math.min(...widths), Math.max(...widths)], line: [res.lineTr, res.lineTe], counts: [Math.min(...counts), Math.max(...counts)] };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Bins', text: ['equal width: edges min + (max − min)·i/n', 'quantile: edges at the i/n quantiles', 'bin → one-hot → one level per bin', 'that level = the mean of the bin\'s rows', 'edges and levels come from training rows'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'edges', pick: 'option', ask: 'Will the binned steps have a lower test error than the straight line?', options: ['yes', 'no'], answer: (fr) => { const m = fr.find((f) => f.meta.phase === 'steps').meta, l = fr[0].meta; return m.binTe < l.lineTe ? 'yes' : 'no'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'steps').say },
    ],
    quiz: [
      { q: 'Values 1 to 100, cut into 4 equal-width bins. The edges are…', options: ['1, 25.75, 50.5, 75.25, 100', '1, 25, 50, 75, 100 exactly', 'the quartiles of the data', '0, 1, 2, 3, 4'], answer: 0, why: 'Width (100 − 1)/4 = 24.75 from the minimum.' },
      { q: 'Most values are small and a few are huge. Which bins avoid near-empty bins?', options: ['quantile bins', 'equal-width bins', 'either', 'neither'], answer: 0, why: 'Quantile edges put the same number of rows in each bin.' },
      { q: 'A linear model on one-hot bins predicts, inside each bin…', options: ['the mean of that bin\'s training rows', 'a straight line', 'zero', 'the overall mean'], answer: 0, why: 'Each bin\'s indicator gets its own coefficient, which least squares sets to the bin mean.' },
      { q: 'With very many bins, the binned fit…', options: ['follows the noise: low training error, higher test error', 'always improves on test data', 'becomes a straight line', 'ignores the data'], answer: 0, why: 'Each level rests on few rows.' },
      { q: 'What is lost by binning a number?', options: ['the differences between values inside a bin', 'nothing', 'the order of the bins', 'the label'], answer: 0, why: 'Every value in a bin looks the same to the model.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
