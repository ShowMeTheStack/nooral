// Part 2, the confusion matrix. Binary: the test examples (a plane with the model's probability shaded, or a strip
// of scores) beside the four cells, each cell opened in turn with its examples ringed and listed; then the scores
// read from the cells. Many classes: the k × k counts, normalised by row (recall) and by column (precision), and the
// worst confusion opened to show the rows inside it.
(function (S) {
  'use strict';
  const H = S.P01, P = S.P02, C = () => S.Kit.curve, SC = () => S.Kit.scatter, HM = () => S.Kit.heat, DT = () => S.Kit.dt, M = () => S.Metrics;
  const CELLS = { tp: ['true positives', 'positives the model caught'], fn: ['false negatives', 'positives the model missed'], fp: ['false positives', 'negatives the model flagged'], tn: ['true negatives', 'negatives correctly left alone'] };

  S.topic({
    id: 'eval-confusion', family: 'p02-confusion', order: 10, title: 'The confusion matrix',
    blurb: 'Accuracy is one number; the confusion matrix shows where it came from. Rows are the true classes, columns the predicted ones. For two classes the four cells are the true positives (caught), false negatives (missed), false positives (false alarms) and true negatives. Open a cell to see its examples: the misses and false alarms are where a model is worth studying, and every score in the next topics is read off these four counts.',
    inputs: [
      { id: 'src', kind: 'choice', label: 'Task', def: 'moons', options: P.sourceOptions(['moons', 'loans', 'cancer', 'pumps']) },
      { id: 't', kind: 'slider', label: 'Threshold', def: 0.5, min: 0.05, max: 0.95, step: 0.05, fmt: (v) => (+v).toFixed(2) },
    ],
    presets: [
      { label: 'Two moons at 0.5', values: { src: 'moons', t: 0.5 } },
      { label: 'Moons, a cautious threshold 0.8', values: { src: 'moons', t: 0.8 } },
      { label: 'Loans at 0.5', values: { src: 'loans', t: 0.5 } },
      { label: 'Loans at 0.25: catch more defaults', values: { src: 'loans', t: 0.25 } },
      { label: 'Cancer at 0.5', values: { src: 'cancer', t: 0.5 } },
    ],
    counters: [{ id: 'n', label: 'test examples' }],
    run(inp) {
      const B = P.binary(inp.src), t = inp.t, c = P.at(B, t), n = B.y.length;
      const kind = B.score.map((s, i) => (s >= t ? (B.y[i] === 1 ? 'tp' : 'fp') : B.y[i] === 1 ? 'fn' : 'tn'));
      const F = new S.Frames(['n']), view = H.box(-14, -8, 790, 470);
      const plane = !!B.Xte;
      let ax, base;
      if (plane) {
        const xs = B.Xte.map((p) => p[0]), ys = B.Xte.map((p) => p[1]);
        ax = C().axes({ prefix: 'ax', x: 40, y: 50, w: 380, h: 300, xr: H.span(Math.min(...xs), Math.max(...xs), 0.04).slice(0, 2), yr: H.span(Math.min(...ys), Math.max(...ys), 0.06).slice(0, 2), xname: 'x1', yname: 'x2', tickSize: 12 });
        const prob = (u, v) => B.fit.predictProba(B.scaler.apply([[u, v]]))[0][1];
        base = [...ax.items, SC().field(ax, 'fld', prob, { n: 60 }), SC().level(ax, 'lvl', SC().grid(ax, (u, v) => prob(u, v) - t, 50), 0, '')];
      } else {
        ax = C().axes({ prefix: 'ax', x: 40, y: 50, w: 380, h: 300, xr: [0, 1], yr: [0, 1], yticks: 0, xname: 'score: P(' + B.posName + ')', grid: false, tickSize: 12 });
        base = [...ax.items, { key: 'thr', kind: 'path', d: `M${ax.X(t)},${ax.o.y} V${ax.o.y + ax.o.h}`, cls: 'thr', audit: 'ignore', layer: 'lines' }, H.text('lp', 46, 72, B.posName, 'c1', 13), H.text('ln', 46, 340, B.negName, 'c0', 13)];
      }
      const dots = (open) => (plane ? B.Xte.map((p, i) => ({ key: 'p' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: open && kind[i] === open ? 5 : 3.4, cls: 'sp c' + B.y[i] + (open && kind[i] === open ? ' wrong' : open ? ' faded' : ''), audit: 'overlap-ok' })) : P.strip(ax, 'p', B).map((d, i) => Object.assign(d, { r: open && kind[i] === open ? 4.6 : d.r, cls: d.cls + (open && kind[i] === open ? ' wrong' : open ? ' faded' : '') })));
      const title = H.title('tt', 0, 6, B.title + ': ' + n + ' test examples, threshold ' + H.f(t, 2));
      const cmAt = { x: 560, y: 110, cell: 86 };
      // a click on a cell lists its examples (the ones nearest the threshold first: the hardest calls)
      const cellInfo = (k) => {
        const idx = B.score.map((s, i) => [s, i]).filter(([, i]) => kind[i] === k).sort((a, b) => Math.abs(a[0] - t) - Math.abs(b[0] - t));
        return { title: k.toUpperCase() + ': ' + c[k] + ' ' + CELLS[k][0], text: idx.length ? 'Their scores, nearest the threshold ' + H.f(t, 2) + ' first (the closest calls).' : 'No example falls in this cell.', cols: idx.length ? ['example', 'score', 'true class'] : null, rows: idx.slice(0, 12).map(([s, i]) => [i, H.f(s, 3), B.y[i] === 1 ? B.posName : B.negName]) };
      };
      const cm = (open, pick) => P.cm2({ prefix: 'cm', x: cmAt.x, y: cmAt.y, cell: cmAt.cell, c, open, posName: B.posName, negName: B.negName, pick, info: cellInfo });
      const list = (k) => {
        const idx = B.score.map((s, i) => [s, i]).filter(([, i]) => kind[i] === k).sort((a, b) => (k === 'tp' || k === 'fp' ? b[0] - a[0] : a[0] - b[0])).slice(0, 4);
        if (!idx.length) return [H.text('ex0', 470, 330, 'no examples in this cell', 'tick', 13)];
        return [H.text('exh', 470, 318, (k === 'fn' || k === 'tn' ? 'lowest' : 'highest') + ' scores in this cell:', 'tick', 13)].concat(idx.map(([s, i], r) => H.text('ex' + r, 470, 342 + r * 22, 'example ' + i + ': score ' + H.f(s, 3), 'mono', 13)));
      };
      F.set('n', n);
      F.add({ items: [title, ...base, ...dots(null)], say: plane ? `${n} test points the model never saw; the shading is its probability of ${B.posName}, and the line is where that probability equals the threshold ${H.f(t, 2)}. Points on the ${B.posName} side are predicted ${B.posName}.` : `${n} test examples, each placed at the model's score (its probability of ${B.posName}); ${B.posName} examples above, ${B.negName} below. Scores at or above the threshold ${H.f(t, 2)} (the line) are predicted ${B.posName}.`, meta: { phase: 'data', n, t }, view });
      F.add({ items: [title, ...base, ...dots(null), ...cm(null, false)], say: `Counting each test example by its true class (row) and its prediction (column): TP = ${c.tp}, FN = ${c.fn}, FP = ${c.fp}, TN = ${c.tn}. The diagonal, ${c.tp + c.tn} of ${n}, is right.`, meta: { phase: 'counts', tp: c.tp, fn: c.fn, fp: c.fp, tn: c.tn }, view });
      for (const k of ['fn', 'fp', 'tp', 'tn']) {
        F.add({ items: [title, ...base, ...dots(k), ...cm(k, false), ...list(k)], say: `Opening ${k.toUpperCase()}: the ${c[k]} ${CELLS[k][0]}, ${CELLS[k][1]} (ringed).` + (k === 'fn' && c.fn ? ` They are ${B.posName} examples whose scores fell below ${H.f(t, 2)}.` : k === 'fp' && c.fp ? ` They are ${B.negName} examples that scored ${H.f(t, 2)} or more.` : ''), meta: { phase: 'open', cell: k, count: c[k] }, view });
      }
      const acc = (c.tp + c.tn) / n;
      const lines = [['accuracy = (TP + TN) / n = ' + (c.tp + c.tn) + ' / ' + n + ' = ' + H.f(acc, 3), 'strong'], ['precision = TP / (TP + FP) = ' + c.tp + ' / ' + (c.tp + c.fp) + ' = ' + H.f(c.precision, 3), 'strong'], ['recall = TP / (TP + FN) = ' + c.tp + ' / ' + (c.tp + c.fn) + ' = ' + H.f(c.recall, 3), 'strong'], ['specificity = TN / (TN + FP) = ' + c.tn + ' / ' + (c.tn + c.fp) + ' = ' + H.f(M().div(c.tn, c.tn + c.fp), 3), 'strong']];
      F.add({ items: [title, ...base, ...dots(null), ...cm(null, false), ...lines.map((l, i) => H.text('m' + i, 440, 330 + i * 26, l[0], l[1], 13))], say: `Every score is a ratio of these cells: accuracy ${H.f(acc, 3)}, precision ${H.f(c.precision, 3)} (of the flagged, how many were ${B.posName}), recall ${H.f(c.recall, 3)} (of the ${B.posName} examples, how many were caught) and specificity ${H.f(M().div(c.tn, c.tn + c.fp), 3)}.`, meta: { phase: 'scores', acc, precision: c.precision, recall: c.recall, specificity: M().div(c.tn, c.tn + c.fp) }, view });
      const ctx = { n, right: c.tp + c.tn, t };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Four cells', text: ['rows: the truth; columns: the prediction', 'TP: positive, predicted positive', 'FN: positive, predicted negative (a miss)', 'FP: negative, predicted positive (a false alarm)', 'TN: negative, predicted negative', 'TP + FN + FP + TN = n'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'data', pick: 'option', ask: 'Of the two kinds of mistake, which will be more common at this threshold?', options: ['false negatives (misses)', 'false positives (false alarms)', 'equally common'], answer: (fr) => { const m = fr[1].meta; return m.fn > m.fp ? 'false negatives (misses)' : m.fp > m.fn ? 'false positives (false alarms)' : 'equally common'; }, explain: (fr) => fr[1].say },
      { at: (f) => f.meta.phase === 'counts', pick: 'item', ask: 'Click the cell that holds the positives the model missed.', answer: 'cm:fn', explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'A test for a disease misses a sick patient. Which cell is that?', options: ['false negative', 'false positive', 'true negative', 'true positive'], answer: 0, why: 'The truth is positive (sick) and the prediction negative.' },
      { q: 'TP = 30, FN = 10, FP = 20, TN = 140. What is the recall?', options: ['0.75', '0.6', '0.85', '0.3'], answer: 0, why: '30 / (30 + 10).' },
      { q: 'Same counts. What is the precision?', options: ['0.6', '0.75', '0.875', '0.15'], answer: 0, why: '30 / (30 + 20).' },
      { q: 'Raising the threshold usually moves examples from…', options: ['TP to FN and FP to TN', 'FN to TP', 'TN to FP', 'nowhere: the counts are fixed'], answer: 0, why: 'Fewer examples are predicted positive.' },
      { q: 'Which sum equals the number of truly positive examples?', options: ['TP + FN', 'TP + FP', 'TP + TN', 'FP + FN'], answer: 0, why: 'The first row of the matrix.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Many classes, normalised
  // ---------------------------------------------------------------------------------------------------------------
  const MSETS = {
    flowers: { label: 'Wildflowers (3 species)', load: () => H.data('wildflowers') },
    wine: { label: 'Wine (3 cultivars)', load: () => H.data('wine') },
    penguins: { label: 'Penguins (3 species)', load: () => H.data('penguins') },
  };
  const MMODELS = { nb: { label: 'Gaussian naive Bayes', make: () => S.ML.GaussianNB() }, tree: { label: 'a depth-2 tree', make: () => S.ML.DecisionTreeClassifier({ maxDepth: 2, seed: 0 }) }, knn: { label: '5-NN (unscaled)', make: () => S.ML.KNeighborsClassifier({ k: 5 }) } };
  P.multi = (data, model) => P.once('multi:' + data + ':' + model, () => {
    const ds = MSETS[data].load(), keep = ds.X.map((r) => r.every(H.isNum));
    const X = ds.X.filter((_, i) => keep[i]), y = ds.y.filter((_, i) => keep[i]);
    const sp = S.ML.trainTestSplit(X.length, { testSize: 0.4, stratify: y, seed: 4 }), te = sp.test.slice().sort((a, b) => a - b);
    const m = MMODELS[model].make().fit(sp.train.map((i) => X[i]), sp.train.map((i) => y[i]));
    return { ds, X, y, te, yt: te.map((i) => y[i]), yp: m.predict(te.map((i) => X[i])), classes: ds.classes };
  });
  S.topic({
    id: 'eval-confusion-multi', family: 'p02-confusion', order: 20, title: 'Many classes, normalised',
    blurb: 'With k classes the matrix is k × k: row i, column j counts the examples of class i predicted as class j. Raw counts favour the big classes, so normalise. Dividing each row by its total gives the share of each true class that went where (the diagonal is each class\'s recall). Dividing each column by its total tells you, when the model says class j, how often it is right (the diagonal is precision). Opening the largest off-diagonal cell shows which examples get confused.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'flowers', options: Object.entries(MSETS).map(([value, s]) => ({ value, label: s.label })) },
      { id: 'model', kind: 'choice', label: 'Model', def: 'nb', options: Object.entries(MMODELS).map(([value, m]) => ({ value, label: m.label })) },
    ],
    presets: [
      { label: 'Wildflowers, naive Bayes', values: { data: 'flowers', model: 'nb' } },
      { label: 'Wildflowers, a depth-2 tree', values: { data: 'flowers', model: 'tree' } },
      { label: 'Wine, unscaled 5-NN', values: { data: 'wine', model: 'knn' } },
      { label: 'Penguins, naive Bayes', values: { data: 'penguins', model: 'nb' } },
    ],
    counters: [{ id: 'n', label: 'test examples' }],
    run(inp) {
      const R = P.multi(inp.data, inp.model), K = R.classes.length, labels = R.classes.map((_, k) => k);
      const cm = M().confusion(R.yt, R.yp, { labels }), byRow = M().confusion(R.yt, R.yp, { labels, normalize: 'true' }).matrix, byCol = M().confusion(R.yt, R.yp, { labels, normalize: 'pred' }).matrix;
      const n = cm.n, F = new S.Frames(['n']), view = H.box(-14, -8, 790, 470);
      let worst = [0, 1];
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) if (i !== j && cm.counts[i][j] > cm.counts[worst[0]][worst[1]]) worst = [i, j];
      const names = R.classes.map((c) => c.replace('cultivar ', 'c'));
      const grid = (vals, fmt, max, open) => HM().grid({ prefix: 'hm', x: 120, y: 90, cw: 92, ch: 70, vals, scheme: 'seq', max, fmt, rows: names, cols: names, size: 16, cls: (i, j) => (open && i === open[0] && j === open[1] ? 'sel' : '') }).items;
      const axisNames = [H.text('pa', 120 + (K * 94) / 2, 50, 'predicted →', 'tick', 13, 'middle'), H.text('ta', 110, 70, 'true ↓', 'tick', 13, 'end')];
      const title = H.title('tt', 0, 6, MSETS[inp.data].label + ', ' + MMODELS[inp.model].label + ': ' + n + ' test rows');
      const side = (lines) => lines.map((l, i) => H.text('sd' + i, 450, 110 + i * 26, l[0], l[1] || 'strong', 14));
      const pct = (v) => H.f(v * 100, 0) + '%';
      const correct = labels.reduce((a, k) => a + cm.counts[k][k], 0);
      F.set('n', n);
      F.add({ items: [title, ...axisNames, ...grid(cm.counts, (v) => String(v), Math.max(...cm.rowTotals)), ...side([['counts', 'act'], ['row totals: ' + cm.rowTotals.join(', '), 'strong'], ['column totals: ' + cm.colTotals.join(', '), 'strong'], ['correct (diagonal): ' + correct + ' of ' + n, 'good']])], say: `Counts: row i, column j holds the test rows of true class i that were predicted as class j. ${correct} of ${n} lie on the diagonal; the largest mistake is ${H.plural(cm.counts[worst[0]][worst[1]], R.classes[worst[0]] + ' row')} predicted as ${R.classes[worst[1]]}.`, meta: { phase: 'counts', counts: cm.counts, worst, correct }, view });
      const rec = labels.map((k) => byRow[k][k]);
      F.add({ items: [title, ...axisNames, ...grid(byRow, pct, 1), ...side([['each row divided by its total', 'act'], ...labels.map((k) => ['recall of ' + R.classes[k] + ': ' + pct(rec[k]), rec[k] < 0.8 ? 'bad' : 'strong'])])], say: `Normalised by row: each row now adds up to 100%, so it shows where the members of one true class went. The diagonal is each class's recall: ${H.list(labels.map((k) => R.classes[k] + ' ' + pct(rec[k])))}.`, meta: { phase: 'rows', recall: rec }, view });
      const prec = labels.map((k) => byCol[k][k]);
      F.add({ items: [title, ...axisNames, ...grid(byCol, pct, 1), ...side([['each column divided by its total', 'act'], ...labels.map((k) => ['precision of ' + R.classes[k] + ': ' + pct(prec[k]), prec[k] < 0.8 ? 'bad' : 'strong'])])], say: `Normalised by column: each column adds up to 100%. When the model says ${R.classes[0]}, it is right ${pct(prec[0])} of the time; the diagonal is each class's precision: ${H.list(labels.map((k) => R.classes[k] + ' ' + pct(prec[k])))}.`, meta: { phase: 'cols', precision: prec }, view });
      const inCell = cm.cells[worst[0]][worst[1]].slice(0, 5), fe = R.ds.features.filter((f) => f.type !== 'category').slice(0, 3), fj = fe.map((f) => R.ds.features.indexOf(f));
      const rows = inCell.map((k) => [String(R.te[k])].concat(fj.map((j) => H.g(R.X[R.te[k]][j], 3))));
      const means = labels.map((c) => fj.map((j) => H.mean(R.X.filter((_, i) => R.y[i] === c).map((r) => r[j]))));
      const sds = fj.map((j) => H.sd(R.X.map((r) => r[j]))), dz = (row, mu) => Math.sqrt(H.sum(fj.map((j, q) => ((row[j] - mu[q]) / sds[q]) ** 2)));
      const closer = cm.cells[worst[0]][worst[1]].filter((k) => dz(R.X[R.te[k]], means[worst[1]]) < dz(R.X[R.te[k]], means[worst[0]])).length;
      const tb = rows.length ? DT().table({ prefix: 'ex', x: 440, y: 110, cols: [{ name: 'row', w: 50, cls: 'idx' }].concat(fe.map((f) => ({ name: H.short(f), w: 92, cls: 'num', size: 12 }))), rows, rowH: 26, size: 13 }).items : [];
      F.add({ items: [title, ...axisNames, ...grid(cm.counts, (v) => String(v), Math.max(...cm.rowTotals), worst), H.title('ext', 450, 90, 'inside the open cell'), ...tb, H.text('mt', 450, 300, 'class means of these columns:', 'tick', 13), ...[worst[0], worst[1]].map((c, r) => H.text('mm' + r, 450, 324 + r * 22, R.classes[c] + ': ' + means[c].map((v) => H.g(v, 3)).join(', '), 'mono', 13))], say: `Opening the largest mistake: ${H.plural(cm.counts[worst[0]][worst[1]], R.classes[worst[0]] + ' row')} predicted as ${R.classes[worst[1]]}. Measured in standard deviations over these ${fj.length} columns, ${closer} of ${cm.cells[worst[0]][worst[1]].length} ${cm.cells[worst[0]][worst[1]].length === 1 ? 'lies' : 'lie'} nearer the ${R.classes[worst[1]]} mean than ${cm.cells[worst[0]][worst[1]].length === 1 ? 'its' : 'their'} own class's mean` + (closer * 2 > cm.cells[worst[0]][worst[1]].length ? (cm.cells[worst[0]][worst[1]].length === 1 ? ': it looks like the other class.' : ': they look like the other class.') : (cm.cells[worst[0]][worst[1]].length === 1 ? ': it still sits nearer its own class; the model misplaces it anyway.' : ': most still sit nearer their own class; the model misplaces them anyway.')), meta: { phase: 'open', cell: worst, count: cm.counts[worst[0]][worst[1]], means: [means[worst[0]], means[worst[1]]], closer }, view });
      const ctx = { n, names: R.classes, whole: 1 };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Reading k × k', text: ['C[i][j]: true class i, predicted j', 'diagonal: correct', 'row-normalised: C[i][j] / Σⱼ C[i][j]', '  diagonal = recall of class i', 'column-normalised: C[i][j] / Σᵢ C[i][j]', '  diagonal = precision of class j'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'counts', pick: 'item', ask: 'Click the cell with the largest mistake (the biggest count off the diagonal).', answer: (fr) => 'hm:c' + fr[0].meta.worst[0] + '_' + fr[0].meta.worst[1], explain: (fr) => fr[0].say },
    ],
    quiz: [
      { q: 'In a row-normalised matrix, what does each row add up to?', options: ['100%', 'the number of classes', 'the accuracy', 'the column total'], answer: 0, why: 'Each row is divided by its own total.' },
      { q: 'The diagonal of the row-normalised matrix is…', options: ['each class\'s recall', 'each class\'s precision', 'the accuracy', 'the F1 scores'], answer: 0, why: 'Correct members of the class over all members of the class.' },
      { q: 'The diagonal of the column-normalised matrix is…', options: ['each class\'s precision', 'each class\'s recall', 'the class sizes', 'always 1'], answer: 0, why: 'Correct predictions of j over all predictions of j.' },
      { q: 'A class with 5 test rows has 4 right. Another with 500 has 400 right. Their row-normalised diagonals are…', options: ['both 80%', '4 and 400', '80% and 0.8%', 'impossible to say'], answer: 0, why: 'Normalising removes the difference in class size.' },
      { q: 'Off-diagonal cells cluster between two particular classes. This suggests…', options: ['those two classes overlap in the features', 'the model is perfect', 'the data has no labels', 'the other classes are missing'], answer: 0, why: 'Confusions happen where classes look alike.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
