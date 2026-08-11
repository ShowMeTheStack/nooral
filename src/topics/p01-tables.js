// Part 1, tables: a dataset drawn as a table whose header colours say each column's type (number, category, the
// label), walked through row, feature, label and type; and an image set, where every pixel becomes a column.
(function (S) {
  'use strict';
  const H = S.P01, DT = () => S.Kit.dt;
  const SETS = [
    { value: 'harvest', label: 'Valley harvest (regression)' }, { value: 'buses', label: 'City bus delays (regression)' },
    { value: 'loans', label: 'Small loans (two classes)' }, { value: 'rents', label: 'Flat rents (regression)' },
    { value: 'penguins', label: 'Palmer penguins (three classes)' }, { value: 'iris', label: 'Iris (three classes)' },
    { value: 'wine', label: 'Wine (13 features)' }, { value: 'digits', label: 'Drawn digits (images)' },
  ];
  const MAXCOLS = 5;
  const cellText = (f, v) => (!H.isNum(v) ? NaN : f.type === 'category' ? (f.categories ? f.categories[v] : String(v)) : String(v));
  const colW = (name, vals) => Math.max(80, Math.ceil(Math.max(S.textWidth(name, 14, { bold: true }), ...vals.map((v) => S.textWidth(String(v), 14, { mono: true })))) + 18);

  function tableRun(inp) {
    const ds = H.data(inp.set), n = Math.min(inp.rows, ds.X.length), F = new S.Frames(['rows', 'cols']);
    const feats = ds.features, d = feats.length, shownF = feats.slice(0, MAXCOLS).map((f, j) => j), more = d - shownF.length;
    const tgt = ds.target, isClass = tgt.type === 'category', tName = tgt.name.replace(/_/g, ' ');
    const label = (v) => (isClass ? H.cname(ds, v) : String(v));
    const rows = ds.X.slice(0, n).map((r, i) => [String(i)].concat(shownF.map((j) => cellText(feats[j], r[j])), more ? ['…'] : [], [label(ds.y[i])]));
    const cols = [{ name: '#', w: 44, cls: 'idx' }].concat(shownF.map((j) => ({ name: H.short(feats[j]), w: colW(H.short(feats[j]), rows.map((r) => r[1 + shownF.indexOf(j)])), cls: feats[j].type === 'category' ? 'cat' : 'num' })), more ? [{ name: '+' + more, w: 46, cls: 'idx' }] : [], [{ name: tName, w: colW(tName, rows.map((r) => r[r.length - 1])), cls: 'lab' }]);
    const tcol = cols.length - 1;
    // per-column facts, from the whole table
    const facts = feats.map((f, j) => {
      const c = H.col(ds, j), ok = c.filter(H.isNum), miss = c.length - ok.length;
      if (f.type === 'category') { const k = (f.categories || []).length || new Set(ok).size; return { j, type: 'category', k, counts: (f.categories || []).map((_, v) => ok.filter((x) => x === v).length), miss }; }
      return { j, type: 'number', min: Math.min(...ok), max: Math.max(...ok), mean: H.mean(ok), miss };
    });
    const nNum = facts.filter((x) => x.type === 'number').length, nCat = d - nNum;
    const classCounts = isClass ? (ds.classes || tgt.categories).map((_, c) => ds.y.filter((v) => v === c).length) : null;
    const ys = isClass ? null : ds.y;
    const top = 34;
    const draw = (lit) => {
      const t = DT().table({ prefix: 'tb', x: 0, y: top, cols, rows, rowH: 32, size: 14, cls: (i, j) => (lit.row === i ? 'act' : lit.col === j ? 'act' : '') });
      const items = [H.title('tt', 0, 10, ds.doc.title + ': the first ' + n + ' of ' + ds.X.length + ' rows')].concat(t.items);
      // the types under the table, as a legend
      const ly = top + t.h + 30;
      items.push({ key: 'lg:num', kind: 'cell', x: 0, y: ly, w: 16, h: 16, rx: 3, text: '', cls: 'dthead num' }, H.text('lg:numt', 24, ly + 13, 'number (' + nNum + ')', 'legend', 13));
      items.push({ key: 'lg:cat', kind: 'cell', x: 150, y: ly, w: 16, h: 16, rx: 3, text: '', cls: 'dthead cat' }, H.text('lg:catt', 174, ly + 13, 'category (' + nCat + ')', 'legend', 13));
      items.push({ key: 'lg:lab', kind: 'cell', x: 310, y: ly, w: 16, h: 16, rx: 3, text: '', cls: 'dthead lab' }, H.text('lg:labt', 334, ly + 13, 'label: ' + (isClass ? 'a class' : 'a number'), 'legend', 13));
      if (lit.note) items.push(H.text('note', 0, ly + 46, lit.note, lit.noteCls || 'strong', 15));
      return { items, t };
    };
    const view = H.box(-14, -8, 790, 470);
    const add = (lit, say, meta) => { const { items } = draw(lit); return F.add({ items, say, meta, view }); };
    F.set('rows', n).set('cols', d);
    add({}, `${ds.doc.title}: ${ds.X.length} rows and ${d} feature columns, plus the label "${tName}". Each row is one example; each column is one feature, measured the same way for every row.`, { phase: 'table', nRows: ds.X.length, d });
    const r0 = ds.X[0], parts = feats.slice(0, 4).map((f, j) => H.short(f) + ' ' + (f.type === 'category' ? cellText(f, r0[j]) : H.isNum(r0[j]) ? r0[j] + (f.unit ? ' ' + f.unit : '') : 'missing'));
    add({ row: 0 }, `Row 0 is one example: ${H.list(parts)}${d > 4 ? ' and ' + (d - 4) + ' more' : ''}. Its label is ${label(ds.y[0])}.`, { phase: 'row', y0: ds.y[0], row: r0.slice(0, 4), more: d - 4, names: feats.map((f) => (f.categories ? f.categories : [])) });
    const nf = facts.find((x) => x.type === 'number'), jn = nf.j;
    add({ col: 1 + shownF.indexOf(jn) }, `The column "${H.short(feats[jn])}" is one feature: over all ${ds.X.length} rows it runs from ${H.g(nf.min)} to ${H.g(nf.max)}${feats[jn].unit ? ' ' + feats[jn].unit : ''}, with mean ${H.g(nf.mean)}.` + (nf.miss ? ` ${H.plural(nf.miss, 'cell is', 'cells are')} missing (—).` : ''), { phase: 'feature', j: jn, min: nf.min, max: nf.max, mean: nf.mean, miss: nf.miss, nRows: ds.X.length });
    const cf = facts.find((x) => x.type === 'category');
    if (cf && shownF.includes(cf.j)) {
      const f = feats[cf.j];
      add({ col: 1 + shownF.indexOf(cf.j) }, `"${H.short(f)}" is a category: it takes ${cf.k} values (${H.list(f.categories.map((c, v) => c + ' ' + cf.counts[v]))} rows). They are stored as the codes 0 to ${cf.k - 1}, but the codes are names, not amounts: ${f.categories[cf.k - 1]} is not "${cf.k - 1} times" ${f.categories[1]}.`, { phase: 'category', j: cf.j, k: cf.k, kmax: cf.k - 1, counts: cf.counts, names: f.categories });
    }
    const ysum = isClass ? null : { min: Math.min(...ys), max: Math.max(...ys), mean: H.mean(ys) };
    const tsay = isClass
      ? `The label "${tName}" is a class: ${H.list((ds.classes || tgt.categories).map((c, k) => c + ' (' + classCounts[k] + ')'))}. Predicting a class is classification.`
      : `The label "${tName}" is a number from ${H.g(ysum.min)} to ${H.g(ysum.max)} (mean ${H.g(ysum.mean)}). Predicting a number is regression.`;
    add({ col: tcol, note: isClass ? 'classification: predict one of ' + classCounts.length + ' classes' : 'regression: predict a number', noteCls: 'act' }, tsay, { phase: 'label', isClass, classCounts, ysum, y0: ds.y[0], names: isClass ? (ds.classes || tgt.categories) : [] });
    const missAll = facts.reduce((a, x) => a + x.miss, 0);
    add({ note: `${ds.X.length} × ${d} = ${ds.X.length * d} feature cells` + (missAll ? `, ${missAll} of them missing` : ', none missing') }, `To a model the table is a matrix X of ${ds.X.length} rows by ${d} columns and a vector y of ${ds.X.length} labels: ${nNum} numeric column${nNum === 1 ? '' : 's'}` + (nCat ? ` and ${nCat} category column${nCat === 1 ? '' : 's'} that must be encoded as numbers first.` : ', ready to use as they are.'), { phase: 'matrix', nNum, nCat, missAll, cells: ds.X.length * d, nRows: ds.X.length, d });
    return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'A dataset', text: ['row = one example (a plot, a trip, a flower)', 'column = one feature, the same for every row', 'label y = what we want to predict', 'number: amounts you may add and compare', 'category: names from a fixed list', 'y a class → classification; y a number → regression'].join('\n') }] } };
  }

  function imageRun(inp) {
    const ds = H.data('digits', { n: 60 }), F = new S.Frames(['rows', 'cols']), w = ds.w, h = ds.h, n = Math.min(inp.rows, 8);
    const img = (i) => Array.from(ds.X[i]);
    const view = H.box(-14, -8, 790, 470);
    const px = 9, ix = 0, iy = 40, x0 = img(0);
    const raster = (key, i, x, y, s) => ({ key, kind: 'raster', x, y, w: w * s, h: h * s, nx: w, ny: h, vals: img(i), scheme: 'ink', smooth: false });
    const on = x0.filter((v) => v > 0).length;
    F.set('rows', 1).set('cols', w * h);
    const base = [H.title('tt', 0, 14, 'One drawn digit: ' + w + ' × ' + h + ' pixels'), raster('im0', 0, ix, iy, px)];
    F.add({ items: base.concat([H.text('lab0', ix + w * px / 2, iy + h * px + 24, 'label: ' + ds.y[0], 'act', 15, 'middle')]), say: `An image is a grid of numbers: this digit ${ds.y[0]} is ${w} × ${h} pixels, each a darkness from 0 (paper) to 1 (ink). ${on} of the ${w * h} pixels have some ink.`, meta: { phase: 'image', on, n: w * h, y0: ds.y[0], w, h }, view });
    // the pixel row: a strip of the first and last pixels of the flattened image
    const rowY = iy + h * px + 60, firstRow = Math.floor(h / 2);
    const strip = (i, y, lit) => {
      const it = [], vals = img(i), show = Array.from({ length: 10 }, (_, k) => firstRow * w + 9 + k);
      it.push({ key: 'st' + i + ':nm', kind: 'text', x: 0, y: y + 19, text: 'image ' + i, size: 13, cls: 'hbname' });
      show.forEach((p, k) => it.push({ key: 'st' + i + ':' + k, kind: 'cell', x: 80 + k * 50, y, w: 46, h: 26, rx: 3, text: H.f(vals[p], 2), size: 12, cls: 'dt num ' + (lit && vals[p] > 0 ? 'act' : '') }));
      it.push({ key: 'st' + i + ':y', kind: 'cell', x: 80 + 10 * 50 + 16, y, w: 54, h: 26, rx: 3, text: String(ds.y[i]), size: 13, cls: 'dt lab' });
      return it;
    };
    const pIdx = firstRow * w + 9;
    const head = [H.text('ph', 80, rowY - 10, 'pixels ' + pIdx + ' to ' + (pIdx + 9) + ' of ' + (w * h) + ' (row ' + firstRow + ' of the image)', 'tick', 13)];
    F.add({ items: base.concat([H.text('lab0', ix + w * px / 2, iy + h * px + 24, 'label: ' + ds.y[0], 'act', 15, 'middle'), { key: 'rowband', kind: 'region', x: ix - 2, y: iy + firstRow * px - 2, w: w * px + 4, h: px + 4, rx: 2, cls: 'lens', audit: 'overlap-ok' }], head, strip(0, rowY, true), [H.text('bct', 290, 60, 'all ' + (w * h) + ' pixels of image 0 in one row, wrapped every 196', 'tick', 13), ...[0, 1, 2, 3].map((k) => ({ key: 'bc' + k, kind: 'raster', x: 290, y: 74 + k * 34, w: 480, h: 24, nx: 196, ny: 1, vals: img(0).slice(k * 196, (k + 1) * 196), scheme: 'ink', smooth: false }))]), say: `Read the pixels row by row and the image becomes one row of ${w * h} numbers: pixel ${firstRow * w} is the start of row ${firstRow}. Each pixel is now a feature column, and the digit is the label.`, meta: { phase: 'flatten', n: w * h, firstRow, start: firstRow * w }, view });
    const rows = [];
    for (let i = 0; i < n; i++) rows.push(...strip(i, 40 + i * 34, false));
    const thumbs = [];
    for (let i = 0; i < n; i++) thumbs.push(raster('th' + i, i, 690, 40 + i * 34 - 1, 1));
    F.set('rows', n);
    F.add({ items: [H.title('tt', 0, 14, 'A table of images: one row each, ' + (w * h) + ' pixel columns')].concat(rows, thumbs), say: `A set of ${n} images is a table of ${n} rows by ${w * h} numeric columns. Neighbouring pixels are strongly related, which is why the convolutional networks of part 10 look at the grid instead of a flat row.`, meta: { phase: 'table', rows: n, n: w * h }, view });
    return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'An image as a row', text: [w + ' × ' + h + ' pixels = ' + (w * h) + ' features', 'pixel (r, c) → column r·' + w + ' + c', 'value: 0 paper … 1 ink', 'label: the digit drawn'].join('\n') }] } };
  }

  S.topic({
    id: 'data-table', family: 'p01-tables', order: 10, title: 'Tables, rows, features and labels',
    blurb: 'Every dataset in this course is a table: one row per example, one column per feature, and a label column holding what we want to predict. A column is a number (an amount you can compare and add) or a category (a name from a short list). The label decides the task: a class means classification, a number means regression. Images fit the same mould once each pixel becomes a column.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'Dataset', def: 'harvest', options: SETS },
      { id: 'rows', kind: 'number', label: 'Rows shown', def: 6, min: 2, max: 8 },
    ],
    presets: [
      { label: 'Harvest: regression', values: { set: 'harvest' } },
      { label: 'Penguins: categories and gaps', values: { set: 'penguins' } },
      { label: 'Loans: two classes', values: { set: 'loans' } },
      { label: 'Wine: many columns', values: { set: 'wine', rows: 5 } },
      { label: 'Digits: pixels as columns', values: { set: 'digits', rows: 8 } },
    ],
    counters: [{ id: 'rows', label: 'rows shown' }, { id: 'cols', label: 'feature columns' }],
    run(inp) { return inp.set === 'digits' ? imageRun(inp) : tableRun(inp); },
    predict: [
      { at: (f) => f.meta && f.meta.phase === 'feature', pick: 'option', ask: 'The next frame looks at the label column. Is this a classification task or a regression task?', options: ['classification', 'regression'], answer: (fr) => (fr.find((f) => f.meta.phase === 'label').meta.isClass ? 'classification' : 'regression'), explain: (fr) => fr.find((f) => f.meta.phase === 'label').say },
      { at: (f) => f.meta && f.meta.phase === 'image', pick: 'option', ask: 'How many feature columns does one image become?', options: (fr) => { const n = fr[0].meta.n; return [String(n), String(Math.sqrt(n)), String(2 * Math.sqrt(n)), '1']; }, answer: (fr) => String(fr[0].meta.n), explain: (fr) => fr[1].say },
    ],
    quiz: [
      { q: 'A table has 300 rows, 4 feature columns and a label. How many examples does it hold?', options: ['300', '4', '1200', '5'], answer: 0, why: 'Each row is one example; the columns describe it.' },
      { q: 'Soil is stored as 0 (loam), 1 (clay), 2 (sand). What kind of column is it?', options: ['a category: the codes are only names', 'a number: sand is twice clay', 'the label', 'a missing value'], answer: 0, why: 'The codes have no order or size; they must be encoded before a model treats them as numbers.' },
      { q: 'The label is the delay of a bus in minutes. Which task is it?', options: ['regression', 'classification', 'clustering', 'none: delays are not labels'], answer: 0, why: 'The label is a number, so we predict a number.' },
      { q: 'A 28 × 28 grey image becomes how many feature columns?', options: ['784', '56', '28', '1'], answer: 0, why: 'One column per pixel: 28 × 28 = 784.' },
      { q: 'Which of these is a feature rather than a label in the loans table?', options: ['income', 'default (repaid or defaulted)', 'neither', 'both'], answer: 0, why: 'Income is known when someone applies; default is what the lender wants to predict.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
