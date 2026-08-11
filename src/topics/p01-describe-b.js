// Part 1, the scatter matrix: every pair of columns as a small scatter coloured by class, with each column's own
// histogram (one per class) on the diagonal; the pairs are visited one by one with their correlation, and the
// column that keeps the classes furthest apart is named by its share of variance between the classes (η²).
(function (S) {
  'use strict';
  const H = S.P01, DT = () => S.Kit.dt;
  const SETS = [
    { value: 'iris', label: 'Iris' }, { value: 'penguins', label: 'Palmer penguins' }, { value: 'wildflowers', label: 'Meadow wildflowers' },
    { value: 'wine', label: 'Wine' }, { value: 'exam', label: 'Exam results' }, { value: 'loans', label: 'Small loans' },
  ];
  /** η²: the share of a column's variance that lies between the class means */
  H.eta2 = (x, y) => {
    const m = H.mean(x), groups = new Map();
    x.forEach((v, i) => { if (!groups.has(y[i])) groups.set(y[i], []); groups.get(y[i]).push(v); });
    let between = 0, total = 0;
    for (const g of groups.values()) between += g.length * (H.mean(g) - m) ** 2;
    for (const v of x) total += (v - m) ** 2;
    return total ? between / total : 0;
  };

  S.topic({
    id: 'data-scatter-matrix', family: 'p01-describe', order: 20, title: 'The scatter matrix',
    blurb: 'With a handful of columns, draw every pair. Panel (row i, column j) plots column j across and column i up, coloured by class; the diagonal, where a column would meet itself, shows that column\'s histogram for each class instead. One picture shows which columns separate the classes, which pairs move together, and where the classes overlap.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Dataset', def: 'iris', options: SETS },
      { id: 'k', kind: 'number', label: 'Columns', def: 4, min: 2, max: 4 },
      { id: 'from', kind: 'number', label: 'From numeric column', def: 0, min: 0, max: 12, check: (v, inp) => { const n = H.numCols(H.data(inp.set)).length; return v + inp.k > n ? 'From numeric column: at most ' + (n - inp.k) + ' for ' + inp.k + ' columns of this dataset' : null; } },
    ],
    presets: [
      { label: 'Iris: petals separate the species', values: { set: 'iris', k: 4, from: 0 } },
      { label: 'Penguins: bills and flippers', values: { set: 'penguins', k: 4, from: 0 } },
      { label: 'Wildflowers: two species overlap', values: { set: 'wildflowers', k: 4, from: 0 } },
      { label: 'Wine: phenols and flavanoids', values: { set: 'wine', k: 3, from: 5 } },
      { label: 'Exam: study and sleep', values: { set: 'exam', k: 3, from: 0 } },
    ],
    counters: [{ id: 'panels', label: 'panels drawn' }],
    run(inp) {
      const ds = H.data(inp.set), cols = H.numCols(ds).slice(inp.from, inp.from + inp.k), k = cols.length;
      const keep = ds.X.map((r) => cols.every((j) => H.isNum(r[j])));
      const rows = ds.X.filter((_, i) => keep[i]), y = ds.y.filter((_, i) => keep[i]), n = rows.length, dropped = ds.X.length - n;
      const X = cols.map((j) => rows.map((r) => r[j])), names = cols.map((j) => H.short(ds.features[j]));
      const classes = ds.classes || ds.target.categories, nc = classes.length;
      const rng = X.map((c) => { const lo = Math.min(...c), hi = Math.max(...c), p = (hi - lo || 1) * 0.06; return [lo - p, hi + p]; });
      const G = { x: 8, y: 34, size: 424 }, gap = 8, s = (G.size - (k - 1) * gap) / k;
      const px = (a) => G.x + a * (s + gap), py = (b) => G.y + b * (s + gap);
      const eta = X.map((c) => H.eta2(c, y)), best = eta.indexOf(Math.max(...eta));
      const R = X.map((a) => X.map((b) => H.pearson(a, b).r));
      const all = [];
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) all.push([a, b]);
      const F = new S.Frames(['panels']), view = H.box(-14, -8, 790, 500), r = n > 200 ? 1.8 : 2.4;
      const frameBox = (a, b, cls) => ({ key: 'fb' + a + '_' + b, kind: 'path', d: `M${px(b)},${py(a)} h${s} v${s} h${-s} Z`, cls: 'axis ' + (cls || ''), audit: 'ignore', layer: 'back' });
      const pairItems = (a, b) => {
        const it = [], [u0, u1] = rng[b], [v0, v1] = rng[a];
        for (let i = 0; i < n; i++) it.push({ key: 'pt' + a + '_' + b + '_' + i, kind: 'dot', x: +(px(b) + ((X[b][i] - u0) / (u1 - u0)) * s).toFixed(2), y: +(py(a) + s - ((X[a][i] - v0) / (v1 - v0)) * s).toFixed(2), r, cls: 'sp c' + y[i], audit: 'overlap-ok' });
        return it;
      };
      const diagItems = (a) => {
        const [u0, u1] = rng[a], bins = Math.max(8, Math.min(16, Math.round(s / 12)));
        const hs = classes.map((_, c) => H.histogram(X[a].filter((_, i) => y[i] === c), bins, u0, u1));
        const top = Math.max(1, ...hs.flatMap((h) => h.counts));
        const ax = { X: (u) => +(px(a) + ((u - u0) / (u1 - u0)) * s).toFixed(2), Y: (v) => +(py(a) + s - (v / top) * (s - 22)).toFixed(2), o: { yr: [0, top] } };
        const it = [];
        hs.forEach((h, c) => it.push(...DT().hist(ax, 'h' + a + '_' + c, h.counts, h.edges, 'c' + c)));
        it.push(H.text('dn' + a, px(a) + 6, py(a) + 16, names[a], 'hbname', 13));
        return it;
      };
      const legend = classes.map((c, i) => [{ key: 'lgd' + i, kind: 'dot', x: 470, y: 64 + i * 24, r: 6, cls: 'swatch c' + i }, H.text('lgt' + i, 484, 69 + i * 24, c, 'legend', 14)]).flat();
      const header = [H.title('tt', 0, 6, ds.doc.title + ': ' + k + ' columns, ' + n + ' rows')].concat(legend);
      const boxes = () => all.map(([a, b]) => frameBox(a, b, a === b ? 'diag' : ''));
      const names2 = () => names.map((nm, a) => H.text('cn' + a, px(a) + s / 2, G.y + G.size + 20, nm, 'axname', 13, 'middle'));
      const side = (lines) => lines.map((l, i) => H.text('sd' + i, 470, 64 + nc * 24 + 24 + i * 26, l[0], l[1] || 'strong', 14));
            F.add({ items: header.concat(boxes(), names2(), side([['row i, column j:', 'tick'], ['column j across, column i up', 'strong'], ['diagonal: one column alone', 'strong']])), say: `${k} columns make a ${k} × ${k} grid of panels. Panel (row i, column j) will plot column j across and column i up for all ${n} rows` + (dropped ? ` (${dropped} rows with a missing value are left out)` : '') + '; the diagonal holds each column on its own.', meta: { phase: 'grid', n, k, dropped, names }, view });
      let diag = [];
      for (let a = 0; a < k; a++) diag.push(...diagItems(a));
      F.set('panels', k);
      F.add({ items: header.concat(boxes(), diag, names2(), side(eta.map((e, a) => [names[a] + ': η² = ' + H.f(e, 3), a === best ? 'act' : 'strong']))), say: `The diagonal shows each column's histogram, one colour per class. Where the colours sit apart the column separates the classes: the share of its variance that lies between the class means (η²) is ${H.list(eta.map((e, a) => names[a] + ' ' + H.f(e, 2)))}.`, meta: { phase: 'diagonal', eta, best }, view });
      let done = diag.slice();
      for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) {
        done = done.concat(pairItems(a, b), pairItems(b, a));
        F.count('panels', 2);
        const lit = [Object.assign(frameBox(a, b), { key: 'lit1', cls: 'guide act' }), Object.assign(frameBox(b, a), { key: 'lit2', cls: 'guide act' })];
        F.add({ items: header.concat(boxes(), done, lit, names2(), side([[names[b] + ' and ' + names[a], 'act'], ['r = ' + H.f(R[a][b], 3), 'strong'], ['the mirror panel swaps the axes', 'tick']])), say: `${H.cap(names[b])} against ${names[a]}, drawn twice (each way round): r = ${H.f(R[a][b], 3)}. ` + (Math.abs(R[a][b]) > 0.7 ? 'The two columns move closely together.' : Math.abs(R[a][b]) < 0.25 ? 'Little straight-line relation between them.' : 'A loose relation.'), meta: { phase: 'pair', a, b, r: R[a][b] }, view });
      }
      let bp = [0, 1];
      for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) if (Math.abs(R[a][b]) > Math.abs(R[bp[0]][bp[1]])) bp = [a, b];
      F.add({ items: header.concat(boxes(), done, names2(), side([['separates best: ' + names[best], 'act'], ['η² = ' + H.f(eta[best], 3), 'strong'], ['most related: ' + names[bp[0]] + ', ' + names[bp[1]], 'strong'], ['r = ' + H.f(R[bp[0]][bp[1]], 3), 'strong']])), say: `Reading the whole matrix: ${names[best]} separates the classes best on its own (η² = ${H.f(eta[best], 3)}), and the most related pair is ${names[bp[0]]} with ${names[bp[1]]} (r = ${H.f(R[bp[0]][bp[1]], 3)}). Strongly related columns carry overlapping information.`, meta: { phase: 'summary', best, eta: eta[best], pair: bp, r: R[bp[0]][bp[1]] }, view });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Reading it', text: ['panel (i, j): column j across, column i up', 'panel (j, i): the same pair, mirrored', 'diagonal: one histogram per class', 'η² = Σ nₖ(x̄ₖ − x̄)² / Σ(xᵢ − x̄)²', 'η² near 1: classes far apart on that column'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'grid', pick: 'option', ask: 'Which column will separate the classes best on its own?', options: (fr) => fr[0].meta.names, answer: (fr) => fr[0].meta.names[fr[1].meta.best], explain: (fr) => fr[1].say },
    ],
    quiz: [
      { q: 'What does the panel in row 2, column 3 show?', options: ['column 3 across, column 2 up', 'column 2 across, column 3 up', 'column 2 alone', 'the correlation of 2 and 3 only'], answer: 0, why: 'Columns of the grid set the horizontal axis, rows the vertical one.' },
      { q: 'Why is the diagonal not a scatter?', options: ['a column against itself is just a straight line', 'there is no data there', 'to save space', 'it shows the labels'], answer: 0, why: 'Every point would sit on y = x, so a histogram is more useful.' },
      { q: 'Panels (1, 2) and (2, 1)…', options: ['show the same pair with the axes swapped', 'show different rows', 'are always identical pictures', 'show the training and test sets'], answer: 0, why: 'The matrix is symmetric up to a reflection of each panel.' },
      { q: 'On the diagonal the class histograms barely overlap. That column…', options: ['separates the classes well on its own', 'is useless', 'has missing values', 'is correlated with every other column'], answer: 0, why: 'Knowing the value almost tells you the class.' },
      { q: 'With 10 columns, how many panels does a scatter matrix have?', options: ['100', '10', '45', '20'], answer: 0, why: '10 × 10, of which 45 distinct pairs appear twice and 10 are on the diagonal.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
