// Part 1, scaling: two columns in very different units, drawn first on their own axes and then to one common scale
// (which is how a distance sees them), a test point's nearest neighbours before and after scaling, and the k-NN
// accuracy on held-out rows under each scaler.
(function (S) {
  'use strict';
  const H = S.P01, C = () => S.Kit.curve, ML = () => S.ML, DT = () => S.Kit.dt;
  const SETS = {
    wine: { set: 'wine', a: 12, b: 10, label: 'Wine: proline and hue' },
    loans: { set: 'loans', a: 0, b: 1, label: 'Loans: income and debt ratio' },
    outlier: { set: 'wine', a: 12, b: 10, outlier: true, label: 'Wine with one wild proline value' },
  };
  function scaleData(key) {
    const s = SETS[key], ds = H.data(s.set), X = ds.X.map((r) => [r[s.a], r[s.b]]), y = ds.y.slice();
    if (s.outlier) X[3] = [X[3][0] * 6, X[3][1]];
    return { X, y, names: [H.short(ds.features[s.a]), H.short(ds.features[s.b])], units: [ds.features[s.a].unit, ds.features[s.b].unit], title: ds.doc.title, classes: ds.classes };
  }
  /** a pair of axes whose units are equally long across and up, centred on the data */
  function equalAxes(prefix, X, box, names) {
    const xs = X.map((p) => p[0]), ys = X.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const k = Math.min(box.w / ((Math.max(...xs) - Math.min(...xs)) * 1.08 || 1), box.h / ((Math.max(...ys) - Math.min(...ys)) * 1.08 || 1));
    const xr = [cx - box.w / 2 / k, cx + box.w / 2 / k], yr = [cy - box.h / 2 / k, cy + box.h / 2 / k];
    const t = C().ticks(xr[0], xr[1], 5), ty = C().ticks(yr[0], yr[1], 4);
    return C().axes(Object.assign({ prefix, xr, yr, xticks: t, yticks: ty.length > 7 ? ty.filter((_, i) => i % 2 === 0) : ty, xname: names[0], yname: names[1], tickSize: 12 }, box));
  }

  S.topic({
    id: 'data-scaling', family: 'p01-transform', order: 10, title: 'Scaling, and why k-NN needs it',
    blurb: 'A distance adds up differences in every column, in whatever units the columns happen to have. When one column runs into the thousands and another stays below 2, the first decides every distance and the second is ignored. Standard scaling subtracts each column\'s mean and divides by its standard deviation; min-max scaling maps each column onto [0, 1]; robust scaling uses the median and the interquartile range, so one wild value cannot squash the rest. The numbers are always learned from the training rows.',
    inputs: [
      { id: 'data', kind: 'choice', label: 'Data', def: 'wine', options: Object.entries(SETS).map(([value, s]) => ({ value, label: s.label })) },
      { id: 'scaler', kind: 'choice', label: 'Scaler', def: 'standard', options: H.SCALERS.filter((o) => o.value !== 'none') },
      { id: 'k', kind: 'number', label: 'k (neighbours)', def: 5, min: 1, max: 15 },
      { id: 'q', kind: 'number', label: 'Test point (index among the test rows)', def: 13, min: 0, max: 40 },
    ],
    presets: [
      { label: 'Wine, standard scaling', values: { data: 'wine', scaler: 'standard' } },
      { label: 'Wine, min-max', values: { data: 'wine', scaler: 'minmax' } },
      { label: 'One wild value: min-max squashes', values: { data: 'outlier', scaler: 'minmax', q: 13 } },
      { label: 'One wild value: robust copes', values: { data: 'outlier', scaler: 'robust', q: 13 } },
      { label: 'Loans: income swamps debt', values: { data: 'loans', scaler: 'standard', k: 9, q: 21 } },
    ],
    counters: [{ id: 'k', label: 'neighbours' }],
    run(inp) {
      const D = scaleData(inp.data), X = D.X, y = D.y, n = X.length, ml = ML();
      const sp = ml.trainTestSplit(n, { testSize: 0.3, stratify: y, seed: 1 });
      const tr = sp.train, te = sp.test, q = te[Math.min(inp.q, te.length - 1)];
      const Xtr = tr.map((i) => X[i]), ytr = tr.map((i) => y[i]);
      const sc = H.scaler(Xtr, inp.scaler), Z = sc.apply(X);
      const knnAcc = (A) => { const m = ml.KNeighborsClassifier({ k: inp.k }).fit(tr.map((i) => A[i]), ytr); return ml.accuracy(te.map((i) => y[i]), m.predict(te.map((i) => A[i]))); };
      const neighbours = (A) => tr.map((i) => [ml.dist(A[i], A[q]), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, inp.k).map(([, i]) => i);
      const vote = (nb) => { const c = new Map(); for (const i of nb) c.set(y[i], (c.get(y[i]) || 0) + 1); let best = null; for (const [cl, v] of c) if (best === null || v > c.get(best) || (v === c.get(best) && cl < best)) best = cl; return { cls: best, count: c.get(best) }; };
      const nbRaw = neighbours(X), nbSc = neighbours(Z), vRaw = vote(nbRaw), vSc = vote(nbSc);
      const accRaw = knnAcc(X), accs = ['standard', 'minmax', 'robust'].map((s) => [s, knnAcc(H.scaler(Xtr, s).apply(X))]);
      const accSc = accs.find((a) => a[0] === inp.scaler)[1];
      const cn = (c) => (D.classes ? D.classes[c] : String(c));
      const F = new S.Frames(['k']), view = H.box(-14, -8, 790, 470);
      const box = { x: 70, y: 52, w: 430, h: 340 };
      const title = H.title('tt', 0, 6, D.title + ': ' + D.names[0] + ' and ' + D.names[1]);
      const pts = (ax, A, ring) => A.map((p, i) => ({ key: 'p' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: i === q ? 6 : 3.4, cls: 'sp c' + y[i] + (te.includes(i) ? ' test' : '') + (ring && ring.includes(i) ? ' wrong' : ''), audit: 'overlap-ok' }));
      const spokes = (ax, A, nb) => nb.map((i, k) => ({ key: 'nb' + k, kind: 'path', d: `M${ax.X(A[q][0])},${ax.Y(A[q][1])} L${ax.X(A[i][0])},${ax.Y(A[i][1])}`, cls: 'guide act', audit: 'ignore', layer: 'lines' }));
      const legend = (D.classes || ['0', '1']).map((c, i) => [{ key: 'lg' + i, kind: 'dot', x: 540, y: 64 + i * 24, r: 6, cls: 'swatch c' + i }, H.text('lgt' + i, 554, 69 + i * 24, c, 'legend', 14)]).flat().concat([{ key: 'lgq', kind: 'dot', x: 540, y: 64 + (D.classes || [0, 1]).length * 24, r: 6, cls: 'sp test c0' }, H.text('lgqt', 554, 69 + (D.classes || [0, 1]).length * 24, 'hollow: a test row', 'legend', 14)]);
      const side = (lines) => lines.map((l, k) => H.text('sd' + k, 540, 170 + k * 26, l[0], l[1] || 'strong', 14));
      const r0 = (j) => [Math.min(...X.map((p) => p[j])), Math.max(...X.map((p) => p[j]))];
      const [ax0, ax1] = r0(0), [ay0, ay1] = r0(1);
      const own = C().axes(Object.assign({ prefix: 'ow', xr: H.span(ax0, ax1, 0.04).slice(0, 2), yr: H.span(ay0, ay1, 0.05).slice(0, 2), xname: D.names[0], yname: D.names[1], tickSize: 12 }, box));
      F.set('k', inp.k);
      F.add({ items: [title, ...own.items, ...pts(own, X), ...legend, ...side([[D.names[0] + ': ' + H.g(ax0) + ' to ' + H.g(ax1), 'strong'], [D.names[1] + ': ' + H.g(ay0) + ' to ' + H.g(ay1), 'strong'], ['each axis stretched to fit', 'tick']])], say: `${n} rows, ${tr.length} for training and ${te.length} held out for testing (hollow). On its own axis each column looks well spread: ${D.names[0]} runs from ${H.g(ax0)} to ${H.g(ax1)}, ${D.names[1]} from ${H.g(ay0)} to ${H.g(ay1)}.`, meta: { phase: 'own', ntr: tr.length, nte: te.length, rx: [ax0, ax1], ry: [ay0, ay1] }, view });
      const eq = equalAxes('eq', X, box, D.names);
      F.add({ items: [title, ...eq.items, ...spokes(eq, X, nbRaw), ...pts(eq, X, nbRaw), ...legend, ...side([['one unit = one unit, both ways', 'act'], ['raw vote: ' + cn(vRaw.cls) + ' (' + vRaw.count + ' of ' + inp.k + ')', vRaw.cls === y[q] ? 'good' : 'bad'], ['the test point is ' + cn(y[q]), 'strong']])], say: `A distance treats one unit of ${D.names[0]} the same as one unit of ${D.names[1]}. Drawn that way the cloud is a thin line: the ${inp.k} nearest neighbours of the big test point are simply the rows with the closest ${D.names[0]}, and they vote ${cn(vRaw.cls)} (${vRaw.count} of ${inp.k}) for a point that is really ${cn(y[q])}` + (vRaw.cls === y[q] ? ': right here, but for the wrong reason.' : '.'), meta: { phase: 'raw', vote: vRaw.cls, count: vRaw.count, truth: y[q] }, view });
      const fmtc = (j) => (inp.scaler === 'minmax' ? `(x − ${H.g(sc.centre[j])}) / ${H.g(sc.scale[j])}` : inp.scaler === 'robust' ? `(x − ${H.g(sc.centre[j])}) / ${H.g(sc.scale[j])}` : `(x − ${H.g(sc.centre[j])}) / ${H.g(sc.scale[j])}`);
      const what = inp.scaler === 'minmax' ? ['minimum', 'range (max − min)'] : inp.scaler === 'robust' ? ['median', 'IQR'] : ['mean', 'sd'];
      F.add({ items: [title, ...eq.items, ...pts(eq, X), ...legend, ...side([[H.cap(H.scalerWords[inp.scaler]) + ', fitted on', 'act'], ['the ' + tr.length + ' training rows:', 'act'], [D.names[0] + ' → ' + fmtc(0), 'math'], [D.names[1] + ' → ' + fmtc(1), 'math']])], say: `${H.cap(H.scalerWords[inp.scaler])} learns two numbers per column from the ${tr.length} training rows only, the ${what[0]} and the ${what[1]}: ${D.names[0]} ${H.g(sc.centre[0])} and ${H.g(sc.scale[0])}, ${D.names[1]} ${H.g(sc.centre[1])} and ${H.g(sc.scale[1])}. Every row, training or test, is then transformed with these same numbers.`, meta: { phase: 'fit', centre: sc.centre, scale: sc.scale }, view });
      const eqz = equalAxes('ez', Z, box, [D.names[0] + ' (scaled)', D.names[1] + ' (scaled)']);
      const zr = (j) => [Math.min(...Z.map((p) => p[j])), Math.max(...Z.map((p) => p[j]))];
      const changed = nbSc.filter((i) => !nbRaw.includes(i)).length;
      F.add({ items: [title, ...eqz.items, ...spokes(eqz, Z, nbSc), ...pts(eqz, Z, nbSc), ...legend, ...side([['scaled: ' + D.names[0] + ' ' + H.g(zr(0)[0]) + ' to ' + H.g(zr(0)[1]), 'strong'], [D.names[1] + ' ' + H.g(zr(1)[0]) + ' to ' + H.g(zr(1)[1]), 'strong'], ['scaled vote: ' + cn(vSc.cls) + ' (' + vSc.count + ' of ' + inp.k + ')', vSc.cls === y[q] ? 'good' : 'bad']])], say: `After ${H.scalerWords[inp.scaler]} both columns span comparable ranges (${H.g(zr(0)[0])} to ${H.g(zr(0)[1])} and ${H.g(zr(1)[0])} to ${H.g(zr(1)[1])}), so both count. ${changed} of the ${inp.k} neighbours change, and they now vote ${cn(vSc.cls)} (${vSc.count} of ${inp.k})` + (vSc.cls === y[q] ? (vRaw.cls === y[q] ? '.' : ', the right class.') : ', still wrong for this point.') + (inp.data === 'outlier' && inp.scaler === 'minmax' ? ' The one wild value took the whole range of proline, so the rest of the column is squeezed into a narrow band.' : ''), meta: { phase: 'scaled', vote: vSc.cls, count: vSc.count, changed, zx: zr(0), zy: zr(1) }, view });
      const bars = [['none', accRaw]].concat(accs);
      F.add({ items: [title, ...eqz.items, ...pts(eqz, Z), ...DT().hbars({ prefix: 'ac', x: 520, y: 150, w: 250, rowH: 30, vals: bars.map((b) => b[1]), names: bars.map((b) => (b[0] === 'none' ? 'raw' : b[0])), max: 1, fmt: (v) => H.pct(v), nameW: 82, valW: 56, cls: (i) => (bars[i][0] === inp.scaler ? 'act' : i === 0 ? 'bad' : 'soft') }), H.title('act', 520, 130, inp.k + '-NN test accuracy')], say: `On the ${te.length} held-out rows, ${inp.k}-NN is right ${H.pct(accRaw)} of the time on the raw columns and ${H.pct(accSc)} after ${H.scalerWords[inp.scaler]}` + `; the three scalers give ${H.list(accs.map(([s, a]) => H.pct(a) + ' (' + s + ')'))}.`, meta: { phase: 'acc', accRaw, accSc, accs: accs.map((a) => a[1]) }, view });
      const ctx = { n, ntr: tr.length, nte: te.length, names: D.classes || [] };
      F.list.forEach((f) => { f.meta.ctx = ctx; });
      return { frames: F.list, def: { sections: [{ id: 'd', kind: 'math', title: 'Three scalers', text: ['standard: z = (x − mean) / sd', 'min-max: z = (x − min) / (max − min)', 'robust: z = (x − median) / IQR', 'fitted per column on the training rows', 'distance-based models need it (k-NN, SVM, k-means);', 'trees do not care'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'fit', pick: 'option', ask: 'Will k-NN be more accurate after scaling than on the raw columns?', options: ['more accurate', 'less accurate', 'exactly the same'], answer: (fr) => { const a = fr.find((f) => f.meta.phase === 'acc').meta; return a.accSc > a.accRaw ? 'more accurate' : a.accSc < a.accRaw ? 'less accurate' : 'exactly the same'; }, explain: (fr) => fr.find((f) => f.meta.phase === 'acc').say },
    ],
    quiz: [
      { q: 'Column A ranges over 0–2000 and column B over 0–1. Without scaling, a Euclidean distance is decided by…', options: ['A almost entirely', 'B almost entirely', 'both equally', 'neither'], answer: 0, why: 'A difference of 300 in A dwarfs any possible difference in B.' },
      { q: 'Standard scaling of the value 70, when the training mean is 50 and the sd 10, gives…', options: ['2', '7', '0.7', '20'], answer: 0, why: '(70 − 50) / 10 = 2.' },
      { q: 'Why fit the scaler on the training rows only?', options: ['the test rows must not influence anything learned', 'test rows cannot be scaled', 'it is required by the formula', 'to make the test rows bigger'], answer: 0, why: 'Their mean and sd are information about the test set; using it is a small leak.' },
      { q: 'One value is 10 times larger than the rest. Which scaler keeps the other values spread out?', options: ['robust (median and IQR)', 'min-max', 'neither', 'both equally'], answer: 0, why: 'Min-max divides by the whole range, which the wild value makes huge; the IQR ignores it.' },
      { q: 'Which model is unaffected by scaling a column?', options: ['a decision tree', 'k-NN', 'an RBF-kernel SVM', 'k-means'], answer: 0, why: 'A tree splits on thresholds of one column at a time; stretching the column moves the threshold with it.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
