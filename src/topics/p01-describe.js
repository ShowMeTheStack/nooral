// Part 1, describing data: one column as a strip of dots, a box plot and a histogram with its centre and spread
// marked; and two columns as a scatter with Pearson's r built from the centred products, beside the matrix of every
// pair's r. (The scatter matrix is in p01-describe-b.js.)
(function (S) {
  'use strict';
  const H = S.P01, C = () => S.Kit.curve, DT = () => S.Kit.dt, HM = () => S.Kit.heat;
  const STATSETS = [
    { value: 'harvest', label: 'Valley harvest' }, { value: 'buses', label: 'City bus delays' }, { value: 'loans', label: 'Small loans' },
    { value: 'rents', label: 'Flat rents' }, { value: 'pumps', label: 'Pump sensors' }, { value: 'iris', label: 'Iris' },
    { value: 'penguins', label: 'Palmer penguins' }, { value: 'wine', label: 'Wine' },
  ];
  /** the numeric columns of a dataset */
  H.numCols = (ds) => ds.features.map((f, j) => (f.type !== 'category' ? j : -1)).filter((j) => j >= 0);
  /** a jitter in [−1, 1] that depends only on the index */
  H.jit = (i) => { const s = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return 2 * (s - Math.floor(s)) - 1; };
  /** a round axis range around [lo, hi] */
  H.span = (lo, hi, pad) => { const p = (hi - lo || 1) * (pad === undefined ? 0.04 : pad); const t = C().ticks(lo - p, hi + p, 6), st = t.length > 1 ? t[1] - t[0] : 1; return [Math.min(t[0], lo - p) , Math.max(t[t.length - 1], hi + p), st]; };

  // ---------------------------------------------------------------------------------------------------------------
  // One column: centre, spread and shape
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'data-describe', family: 'p01-describe', order: 10, title: 'Describing one column',
    blurb: 'Before modelling, look at each column. Sort its values and the quartiles fall out: the median halves them and Q1 and Q3 hold the middle half, drawn as a box. The mean is their balance point and the standard deviation their typical distance from it; when a long tail pulls the mean away from the median the column is skewed. A histogram counts the values in equal bins and shows the shape at a glance.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Dataset', def: 'buses', options: STATSETS },
      { id: 'col', kind: 'number', label: 'Numeric column (0 = first)', def: 3, min: 0, max: 12, check: (v, inp) => { const k = H.numCols(H.data(inp.set)).length; return v >= k ? 'Numeric column: this dataset has ' + k + ' (0 to ' + (k - 1) + ')' : null; } },
      { id: 'bins', kind: 'slider', label: 'Bins', def: 12, min: 4, max: 30, step: 1 },
    ],
    presets: [
      { label: 'Bus passengers: flat', values: { set: 'buses', col: 3, bins: 12 } },
      { label: 'Loan years: skewed right', values: { set: 'loans', col: 2, bins: 15 } },
      { label: 'Rain on bus trips: mostly 0', values: { set: 'buses', col: 1, bins: 10 } },
      { label: 'Iris petal length: two humps', values: { set: 'iris', col: 2, bins: 20 } },
      { label: 'Penguin body mass, with gaps', values: { set: 'penguins', col: 3, bins: 14 } },
      { label: 'Pump temperature: a few faults', values: { set: 'pumps', col: 0, bins: 24 } },
    ],
    counters: [{ id: 'n', label: 'values' }, { id: 'bins', label: 'bins' }],
    run(inp) {
      const ds = H.data(inp.set), j = H.numCols(ds)[inp.col], f = ds.features[j], all = H.col(ds, j), v = all.filter(H.isNum), miss = all.length - v.length;
      const name = H.short(f), unit = f.unit ? ' ' + f.unit : '';
      const st = H.five(v), mean = H.mean(v), sd = H.sd(v), hist = H.histogram(v, inp.bins);
      const [x0, x1] = H.span(st.min, st.max, 0.03);
      const big = hist.counts.indexOf(Math.max(...hist.counts));
      const yMax = C().ticks(0, Math.max(...hist.counts) * 1.1, 4);
      const L = { x: 64, w: 690 };
      const ax = C().axes({ prefix: 'hx', x: L.x, y: 50, w: L.w, h: 178, xr: [x0, x1], yr: [0, yMax[yMax.length - 1]], xticks: 0, yticks: yMax, yname: 'count', tickSize: 12 });
      const bx = C().axes({ prefix: 'bx', x: L.x, y: 250, w: L.w, h: 150, xr: [x0, x1], yr: [0, 1], xticks: undefined, yticks: 0, xname: name + unit, grid: false, tickSize: 12 });
      const stripY = (i) => 345 + H.jit(i) * 30;
      const dots = (cls) => v.map((x, i) => ({ key: 'pt' + i, kind: 'dot', x: bx.X(x), y: Math.round(stripY(i) * 100) / 100, r: v.length > 400 ? 2 : 2.8, cls: (typeof cls === 'function' ? cls(x) : cls) || 'pt0', audit: 'overlap-ok' }));
      const F = new S.Frames(['n', 'bins']);
      const view = H.box(-14, -8, 790, 470);
      const boxItems = () => DT().box(bx, 'bp', st, 280, 30, '');
      const lbl = (key, xv, text, cls, y) => ({ key, kind: 'text', x: bx.X(xv), y: y || 256, text, anchor: 'middle', size: 12, cls: cls || 'tick' });
      const qLabels = () => [lbl('lq1', st.q1, 'Q1 ' + H.g(st.q1)), lbl('lmd', st.med, 'median ' + H.g(st.med), 'act'), lbl('lq3', st.q3, 'Q3 ' + H.g(st.q3))];
      const spread = (st.q3 - st.q1) / (x1 - x0);
      // the three quartile names collide when the box is narrow: then only the median is named over it
      const qNames = () => (spread < 0.24 ? [lbl('lmd', st.med, 'median ' + H.g(st.med) + ' (Q1 ' + H.g(st.q1) + ', Q3 ' + H.g(st.q3) + ')', 'act')] : qLabels());
      const meanItems = () => [{ key: 'mean', kind: 'path', d: `M${bx.X(mean)},50 V410`, cls: 'mean', audit: 'ignore', layer: 'lines' }, { key: 'sdb', kind: 'path', d: `M${bx.X(Math.max(x0, mean - sd))},${388} H${bx.X(Math.min(x1, mean + sd))}`, cls: 'series hot', audit: 'ignore' }];
      F.set('n', v.length);
      F.add({ items: [...bx.items, ...dots(), H.title('tt', 0, 6, ds.doc.title + ': ' + name)], say: `The column "${name}" holds ${v.length} values` + (miss ? ` (${H.plural(miss, 'cell is', 'cells are')} missing and left out)` : '') + `, each a dot along the axis. They run from ${H.g(st.min)} to ${H.g(st.max)}${unit}.`, meta: { phase: 'dots', n: v.length, min: st.min, max: st.max, miss }, view });
      F.add({ items: [...bx.items, ...dots((x) => (x < st.q1 ? 'pt0' : x > st.q3 ? 'pt0' : 'sp c0')), ...boxItems(), ...qNames(), H.title('tt', 0, 6, ds.doc.title + ': ' + name)], say: `Sorted, the middle value is the median, ${H.g(st.med)}. A quarter of the values lie below Q1 = ${H.g(st.q1)} and a quarter above Q3 = ${H.g(st.q3)}: the box holds the middle half, and its width IQR = ${H.g(st.iqr)}.`, meta: { phase: 'box', q1: st.q1, med: st.med, q3: st.q3, iqr: st.iqr }, view });
      const skew = Math.abs(mean - st.med) < 0.03 * (st.max - st.min) ? 'equal' : mean > st.med ? 'above' : 'below';
      const skewSay = skew === 'equal' ? 'They nearly agree, so the column is roughly symmetric.' : skew === 'above' ? 'The mean sits above the median: a long tail of large values pulls it up (skewed right).' : 'The mean sits below the median: a tail of small values pulls it down (skewed left).';
      F.add({ items: [...bx.items, ...dots(), ...boxItems(), ...qNames(), ...meanItems(), lbl('lmean', mean, 'mean ' + H.g(mean), 'hot', 42), H.title('tt', 0, 6, ds.doc.title + ': ' + name)], say: `The mean is ${H.g(mean)} (dashed) and the standard deviation ${H.g(sd)} (the bar spans mean ± one sd). ${skewSay}`, meta: { phase: 'mean', mean, sd, skew }, view });
      F.set('bins', inp.bins);
      const binW = hist.edges[1] - hist.edges[0];
      F.add({ items: [...ax.items, ...bx.items, ...DT().hist(ax, 'hb', hist.counts, hist.edges, (k) => (k === big ? 'hot' : '')), ...dots(), ...boxItems(), ...meanItems(), H.title('tt', 0, 6, ds.doc.title + ': ' + name)], say: `The histogram cuts the range into ${inp.bins} bins of width ${H.g(binW)} and counts the values in each. The fullest bin, ${H.g(hist.edges[big])} to ${H.g(hist.edges[big + 1])}, holds ${hist.counts[big]} of them.`, meta: { phase: 'hist', counts: hist.counts, edges: hist.edges, big, binW }, view });
      const out = v.filter((x) => x < st.loF || x > st.hiF).length;
      F.add({ items: [...ax.items, ...bx.items, ...DT().hist(ax, 'hb', hist.counts, hist.edges, ''), ...dots((x) => (x < st.loF || x > st.hiF ? 'outl' : 'pt0')), ...boxItems(), ...meanItems(), H.title('tt', 0, 6, ds.doc.title + ': ' + name)], say: `The whiskers reach the furthest values within 1.5 × IQR of the box (from ${H.g(st.lo)} to ${H.g(st.hi)}). ` + (out ? `${H.plural(out, 'value lies', 'values lie')} beyond them (red): candidates for a closer look, not automatically errors.` : 'No value lies beyond them.'), meta: { phase: 'whiskers', lo: st.lo, hi: st.hi, out }, view });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Summaries', text: ['mean x̄ = Σ xᵢ / n', 'sd = √(Σ (xᵢ − x̄)² / n)', 'median: the middle sorted value', 'Q1, Q3: the 25% and 75% points', 'IQR = Q3 − Q1', 'whiskers: last values within 1.5·IQR of the box'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'box', pick: 'option', ask: 'The next frame marks the mean. Where will it be, compared with the median?', options: ['above the median', 'below the median', 'about equal'], answer: (fr) => { const s = fr.find((f) => f.meta.phase === 'mean').meta.skew; return s === 'equal' ? 'about equal' : s + ' the median'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'mean').say },
    ],
    quiz: [
      { q: 'Ten salaries: nine near 30 000 and one of 1 000 000. Which describes a typical salary better?', options: ['the median', 'the mean', 'the standard deviation', 'the maximum'], answer: 0, why: 'The one huge value drags the mean far up; the median ignores how far it is.' },
      { q: 'What does the box of a box plot contain?', options: ['the middle half of the values, from Q1 to Q3', 'the values within one sd of the mean', 'all but the outliers', 'the most common value'], answer: 0, why: 'A quarter lie below the box and a quarter above it.' },
      { q: 'A column is skewed right (a long tail of large values). The mean is…', options: ['above the median', 'below the median', 'equal to the median', 'always zero'], answer: 0, why: 'The tail pulls the balance point towards it.' },
      { q: 'Doubling the number of histogram bins…', options: ['shows finer detail but noisier counts', 'changes the data', 'changes the median', 'always reveals outliers'], answer: 0, why: 'Narrower bins hold fewer values each, so the shape gets ragged.' },
      { q: 'A value just beyond the whisker is…', options: ['worth a look, but not necessarily wrong', 'certainly a typing error', 'always removed', 'the maximum of the box'], answer: 0, why: 'The 1.5 × IQR rule flags unusual values; whether they are errors needs the story of the data.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Correlation
  // ---------------------------------------------------------------------------------------------------------------
  const PAIRS = {
    petal: { set: 'iris', a: 2, b: 3, label: 'Iris: petal length and width' },
    rent: { set: 'rents', a: 3, b: -1, label: 'Rents: distance and rent' },
    flowers: { set: 'wildflowers', a: 2, b: 3, label: 'Wildflowers: stem height and leaves' },
    rain: { set: 'harvest', a: 0, b: -1, label: 'Harvest: rain and yield (a curve)' },
    outlier: { gen: 'outlier', label: 'Twenty points and one outlier' },
    wine: { set: 'wine', a: 6, b: 5, label: 'Wine: flavanoids and phenols' },
  };
  function pairData(key) {
    const p = PAIRS[key];
    if (p.gen) {
      const r = H.rng(11), x = [], y = [];
      for (let i = 0; i < 20; i++) { const u = r.range(0, 10); x.push(+u.toFixed(2)); y.push(+(5 - 0.3 * u + r.normal(0, 0.6)).toFixed(2)); }
      x.push(30); y.push(14);
      return { x: x.map(Number), y, xn: 'x', yn: 'y', title: 'Generated: 20 points and one outlier', matrix: null };
    }
    const ds = H.data(p.set), col = (j) => (j < 0 ? ds.y : H.col(ds, j)), xs = col(p.a), ys = col(p.b), keep = xs.map((_, i) => H.isNum(xs[i]) && H.isNum(ys[i]));
    const nm = (j) => (j < 0 ? ds.target.name : H.short(ds.features[j]));
    const cols = H.numCols(ds).slice(0, 6), names = cols.map((j) => H.short(ds.features[j]));
    return { x: xs.filter((_, i) => keep[i]), y: ys.filter((_, i) => keep[i]), xn: nm(p.a), yn: nm(p.b), title: ds.doc.title, ds, cols, names, a: p.a, b: p.b };
  }
  S.topic({
    id: 'data-correlation', family: 'p01-describe', order: 30, title: 'Correlation',
    blurb: 'Pearson\'s r measures how well a straight line describes two columns together. Centre both columns on their means: a point in the upper-right or lower-left quarter adds a positive product, one in the other two quarters a negative product. r is the sum of those products, divided so that it lies between −1 and 1. It misses curved patterns and one far point can swing it; Spearman\'s r, the same formula on ranks, is steadier.',
    inputs: [{ id: 'pair', kind: 'choice', label: 'Pair', def: 'petal', options: Object.entries(PAIRS).map(([value, p]) => ({ value, label: p.label })) }],
    presets: Object.keys(PAIRS).map((k) => ({ label: PAIRS[k].label, values: { pair: k } })),
    counters: [{ id: 'n', label: 'points' }],
    run(inp) {
      const P = pairData(inp.pair), n = P.x.length, pr = H.pearson(P.x, P.y), sp = H.spearman(P.x, P.y), F = new S.Frames(['n']);
      const [x0, x1] = H.span(Math.min(...P.x), Math.max(...P.x), 0.05), [y0, y1] = H.span(Math.min(...P.y), Math.max(...P.y), 0.06);
      const ax = C().axes({ prefix: 'cx', x: 58, y: 52, w: 400, h: 344, xr: [x0, x1], yr: [y0, y1], xname: P.xn, yname: P.yn, tickSize: 12 });
      const view = H.box(-14, -8, 790, 470);
      const prod = P.x.map((x, i) => (x - pr.mx) * (P.y[i] - pr.my));
      const pos = prod.filter((v) => v > 0).length, neg = prod.filter((v) => v < 0).length;
      const sPos = H.sum(prod.filter((v) => v > 0)), sNeg = H.sum(prod.filter((v) => v < 0));
      const dots = (cls) => P.x.map((x, i) => ({ key: 'p' + i, kind: 'dot', x: ax.X(x), y: ax.Y(P.y[i]), r: n > 200 ? 2.6 : 3.6, cls: 'sp ' + (cls ? cls(i) : 'cn'), audit: 'overlap-ok' }));
      const cross = [{ key: 'mx', kind: 'path', d: `M${ax.X(pr.mx)},${ax.o.y} V${ax.o.y + ax.o.h}`, cls: 'guide act', audit: 'ignore', layer: 'back', z: 3 }, { key: 'my', kind: 'path', d: `M${ax.o.x},${ax.Y(pr.my)} H${ax.o.x + ax.o.w}`, cls: 'guide act', audit: 'ignore', layer: 'back', z: 3 }];
      const side = (lines, key) => lines.map((l, k) => H.text(key + k, 488, 70 + k * 30, l[0], l[1] || 'strong', 14));
      F.set('n', n);
      F.add({ items: [...ax.items, ...dots(), H.title('tt', 0, 6, P.title)], say: `${n} points: ${P.xn} across, ${P.yn} up. Does one rise with the other, fall with it, or neither?`, meta: { phase: 'scatter', n }, view });
      const quad = (i) => (prod[i] > 0 ? 'c0' : prod[i] < 0 ? 'c1' : 'cn');
      F.add({ items: [...ax.items, ...cross, ...dots(quad), H.title('tt', 0, 6, P.title), ...side([['x̄ = ' + H.g(pr.mx) + ',  ȳ = ' + H.g(pr.my), 'act'], ['blue: (x − x̄)(y − ȳ) > 0', 'c0'], ['orange: (x − x̄)(y − ȳ) < 0', 'c1']], 's')], say: `The dashed lines cross at the means, x̄ = ${H.g(pr.mx)} and ȳ = ${H.g(pr.my)}. ${pos} points (blue) lie in the two quarters where both are above or both below their means, ${neg} (orange) in the other two.`, meta: { phase: 'quarters', mx: pr.mx, my: pr.my, pos, neg }, view });
      F.add({ items: [...ax.items, ...cross, ...dots(quad), H.title('tt', 0, 6, P.title), ...side([['Σ positive products = ' + H.g(sPos), 'c0'], ['Σ negative products = ' + H.g(sNeg), 'c1'], ['Σ (x − x̄)(y − ȳ) = ' + H.g(pr.sxy), 'strong']], 's')], say: `Adding the products: the blue points give ${H.g(sPos)}, the orange ones ${H.g(sNeg)}, so the sum is ${H.g(pr.sxy)}. Its sign says which way the cloud leans; its size still depends on the units.`, meta: { phase: 'sum', sPos, sNeg, sxy: pr.sxy }, view });
      const word = Math.abs(pr.r) > 0.8 ? 'strong' : Math.abs(pr.r) > 0.5 ? 'moderate' : Math.abs(pr.r) > 0.2 ? 'weak' : 'almost no';
      const rLines = [['r = Σ(x − x̄)(y − ȳ)', 'math'], ['     / √(Σ(x − x̄)² · Σ(y − ȳ)²)', 'math'], ['= ' + H.g(pr.sxy) + ' / √(' + H.g(pr.sxx) + ' × ' + H.g(pr.syy) + ')', 'math'], ['r = ' + H.f(pr.r, 3), 'act'], ['Spearman (ranks): ' + H.f(sp, 3), 'strong']];
      F.add({ items: [...ax.items, ...cross, ...dots(quad), H.title('tt', 0, 6, P.title), ...side(rLines, 's')], say: `Dividing by √(${H.g(pr.sxx)} × ${H.g(pr.syy)}) removes the units: r = ${H.f(pr.r, 3)}, ${word} ${pr.r >= 0 ? 'positive' : 'negative'} linear association${word === 'almost no' ? '' : ''}. On ranks instead of values, Spearman's r is ${H.f(sp, 3)}.` + (inp.pair === 'rain' ? ' Yield rises and then falls with rain: a strong pattern that a straight-line measure barely sees.' : inp.pair === 'outlier' ? ' The one far point at the right drags r a long way; the ranks hardly notice it.' : ''), meta: { phase: 'r', r: pr.r, spearman: sp, sxx: pr.sxx, syy: pr.syy }, view });
      if (inp.pair === 'outlier') {
        const xr = P.x.slice(0, -1), yr = P.y.slice(0, -1), r2 = H.pearson(xr, yr).r;
        F.add({ items: [...ax.items, ...dots((i) => (i === n - 1 ? 'c1 wrong' : 'c0')), H.title('tt', 0, 6, P.title), ...side([['with the outlier: r = ' + H.f(pr.r, 3), 'strong'], ['without it: r = ' + H.f(r2, 3), 'act']], 's')], say: `Without the far point, r is ${H.f(r2, 3)} instead of ${H.f(pr.r, 3)}: one point out of ${n} turned a clear negative relation into ${pr.r > 0 ? 'a positive one' : 'a weaker one'}. Always look at the scatter, not just the number.`, meta: { phase: 'without', r2, r: pr.r, n }, view });
      } else {
        const cols = P.cols, X = cols.map((j) => H.col(P.ds, j)), ok = P.ds.X.map((row) => cols.every((j) => H.isNum(row[j])));
        const R = cols.map((_, a) => cols.map((_, b) => H.pearson(X[a].filter((_, i) => ok[i]), X[b].filter((_, i) => ok[i])).r));
        const ia = cols.indexOf(P.a), ib = cols.indexOf(P.b), k = cols.length, cw = Math.min(44, 220 / k);
        const g = HM().grid({ prefix: 'cm', x: 520, y: 90, cw, ch: cw, vals: R, scheme: 'div', max: 1, fmt: k > 5 ? (v) => H.f(v, 1).replace(/^0\./, '.').replace(/^−0\./, '−.') : (v) => H.f(v, 2).replace(/^0\./, '.').replace(/^−0\./, '−.'), rows: cols.map((_, a) => String(a + 1)), cols: cols.map((_, a) => String(a + 1)), size: 12, cls: (a, b) => ((a === ia && b === ib) || (a === ib && b === ia) ? 'sel' : '') });
        const names = P.names.map((nm, a) => H.text('nm' + a, 500, 90 + k * (cw + 2) + 24 + a * 19, (a + 1) + ' = ' + nm, 'tick', 12));
        let best = [0, 1, 0];
        for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) if (Math.abs(R[a][b]) > Math.abs(best[2])) best = [a, b, R[a][b]];
        F.add({ items: [...ax.items, ...cross, ...dots(quad), H.title('tt', 0, 6, P.title), H.title('ct', 500, 50, 'r for every pair'), ...g.items, ...names], say: `The same r for every pair of the first ${k} numeric columns of ${P.title}: blue is positive, vermilion negative, and the diagonal is 1 (a column with itself). The strongest pair is ${P.names[best[0]]} with ${P.names[best[1]]}, r = ${H.f(best[2], 3)}.`, meta: { phase: 'matrix', R, best }, view });
      }
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Pearson and Spearman', text: ['r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Σ(xᵢ − x̄)² Σ(yᵢ − ȳ)²)', '−1 ≤ r ≤ 1; ±1 only on a straight line', 'r ≈ 0: no straight-line trend (a curve may hide)', 'Spearman: r of the ranks; follows any rising trend'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'scatter', pick: 'option', ask: 'Will r be positive, negative, or close to 0?', options: ['clearly positive', 'clearly negative', 'close to 0'], answer: (fr) => { const r = fr.find((f) => f.meta.phase === 'r').meta.r; return r > 0.2 ? 'clearly positive' : r < -0.2 ? 'clearly negative' : 'close to 0'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'r').say },
    ],
    quiz: [
      { q: 'r = −0.9 between distance and rent means…', options: ['flats further out tend to be much cheaper, close to a straight line', 'distance causes rent', 'rent is 90% of distance', 'there is no relation'], answer: 0, why: 'A strong negative linear association; correlation alone says nothing about cause.' },
      { q: 'Points lie on a perfect U shape. Pearson\'s r is…', options: ['near 0', 'near 1', 'near −1', 'undefined'], answer: 0, why: 'The left half falls and the right half rises; the products cancel.' },
      { q: 'Why divide Σ(x − x̄)(y − ȳ) by √(Σ(x − x̄)² Σ(y − ȳ)²)?', options: ['to remove the units, so r lies in [−1, 1]', 'to make it positive', 'to count the points', 'to centre the data'], answer: 0, why: 'Without it, measuring in centimetres instead of metres would change the number.' },
      { q: 'One far outlier is added. Which measure changes least?', options: ['Spearman\'s r (on ranks)', 'Pearson\'s r', 'the mean of x', 'Σ(x − x̄)(y − ȳ)'], answer: 0, why: 'The outlier only takes the top rank, however far away it is.' },
      { q: 'Two columns have r = 0.98. For a linear model this means…', options: ['they carry nearly the same information', 'both must be dropped', 'the model will be perfect', 'one causes the other'], answer: 0, why: 'Highly correlated features are nearly redundant; keeping both can make coefficients unstable.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
