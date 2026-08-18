// Part 4, nearest neighbours and probabilistic models: the families, and the helpers its topics share (the small
// labelled sets in the plane, a square scatter plot, Voronoi cells by clipping, the bag of words of the messages).
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'neighbours', order, title, blurb });
  fam('p04-knn', 10, 'Nearest neighbours', 'Predict what the closest training examples say: Voronoi cells, the choice of k, the distance, the weights of the votes.');
  fam('p04-kd', 20, 'Searching fast', 'A k-d tree cuts the plane into boxes so a query can skip most of the points, and in many dimensions every point is about as far as every other.');
  fam('p04-nb', 30, 'Naive Bayes', 'Bayes’ rule with features treated as independent within a class: Gaussian bells for numbers, word counts for messages, and smoothing for words never seen.');

  const H = S.PL, ML = () => S.ML;
  const P = (S.PL4 = S.PL4 || {});

  // ---- labelled sets in the plane ----------------------------------------------------------------------------
  P.SETS = [
    { value: 'blobs', label: 'three blobs' },
    { value: 'moons', label: 'two moons' },
    { value: 'circles', label: 'ring and disc' },
    { value: 'flowers', label: 'meadow wildflowers' },
    { value: 'iris', label: 'iris (petals)' },
  ];
  /** a labelled set in the plane: {X, y, names, xname, yname} (moons and circles are scaled up to about 0..10) */
  P.data = (name, n, seed, noise) => H.once('p4d:' + [name, n, seed, noise].join(':'), () => {
    const D = S.Data, R = (v) => D.round(v, 2);
    if (name === 'blobs') { const d = D.blobs({ n, seed, std: 0.4 + noise * 1.4, centers: [[2.6, 2.8], [7.2, 3.6], [4.6, 7.4]] }); return { X: d.X.map((r) => r.map(R)), y: d.y, names: ['blue', 'orange', 'green'], xname: 'x₁', yname: 'x₂' }; }
    if (name === 'moons') { const d = D.moons({ n, seed, noise: noise * 0.35 }); return { X: d.X.map((r) => [R(2.4 + r[0] * 2.6), R(3.6 + r[1] * 3.6)]), y: d.y, names: ['blue', 'orange'], xname: 'x₁', yname: 'x₂' }; }
    if (name === 'circles') { const d = D.circles({ n, seed, noise: noise * 0.2 }); return { X: d.X.map((r) => [R(5 + r[0] * 3.8), R(5 + r[1] * 3.8)]), y: d.y, names: ['ring', 'disc'], xname: 'x₁', yname: 'x₂' }; }
    if (name === 'flowers') {
      const d = D.load('wildflowers', { n, seed });
      return { X: d.X.map((r) => [r[0], r[1]]), y: d.y, names: d.classes, xname: 'petal length (cm)', yname: 'petal width (cm)' };
    }
    const d = D.load('iris', {}), rnd = ML().rng(seed), idx = rnd.perm(d.X.length).slice(0, Math.min(n, d.X.length)).sort((a, b) => a - b);
    return { X: idx.map((i) => [d.X[i][2], d.X[i][3]]), y: idx.map((i) => d.y[i]), names: ['setosa', 'versicolor', 'virginica'], xname: 'petal length (cm)', yname: 'petal width (cm)' };
  });
  /** a square plot range covering points (and extra points), equal in both directions */
  P.square = (pts, pad) => {
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    let [x0, x1] = H.span(xs, pad === undefined ? 0.05 : pad), [y0, y1] = H.span(ys, pad === undefined ? 0.05 : pad);
    const s = Math.max(x1 - x0, y1 - y0), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const t = S.Kit.curve.ticks(cx - s / 2, cx + s / 2, 5), st = t.length > 1 ? t[1] - t[0] : 1;
    x0 = Math.floor((cx - s / 2) / st) * st; y0 = Math.floor((cy - s / 2) / st) * st;
    const side = Math.max(Math.ceil((cx + s / 2) / st) * st - x0, Math.ceil((cy + s / 2) / st) * st - y0);
    return { xr: [x0, x0 + side], yr: [y0, y0 + side], step: st };
  };
  /** square axes at the left of the stage: box {x, y, side} */
  P.plot = (prefix, rng, names, box) => {
    box = box || { x: 44, y: 26, side: 430 };
    return H.axes({ prefix, x: box.x, y: box.y, w: box.side, h: box.side, xr: rng.xr, yr: rng.yr, xticks: rng.step, yticks: rng.step, xname: names[0], yname: names[1], grid: true });
  };
  P.parse = (v) => +v;
  /** formula lines (monospace: the camera measures them as such) */
  P.eqs = (...a) => H.eqs(...a).map((x) => Object.assign(x, { mono: true }));

  // ---- Voronoi cells: the box clipped by every bisector -----------------------------------------------------------
  const clip = (poly, a, c) => { // keep a·p <= c
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i], q = poly[(i + 1) % poly.length], fp = a[0] * p[0] + a[1] * p[1] - c, fq = a[0] * q[0] + a[1] * q[1] - c;
      if (fp <= 0) out.push(p);
      if ((fp < 0 && fq > 0) || (fp > 0 && fq < 0)) { const t = fp / (fp - fq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
    }
    return out;
  };
  /** the Voronoi cell of every point inside the rectangle xr × yr, as a polygon in data units */
  P.voronoi = (X, xr, yr) => X.map((p, i) => {
    let poly = [[xr[0], yr[0]], [xr[1], yr[0]], [xr[1], yr[1]], [xr[0], yr[1]]];
    for (let j = 0; j < X.length && poly.length; j++) {
      if (j === i) continue;
      const q = X[j];
      if (q[0] === p[0] && q[1] === p[1]) { if (j < i) poly = []; continue; }
      poly = clip(poly, [2 * (q[0] - p[0]), 2 * (q[1] - p[1])], q[0] * q[0] + q[1] * q[1] - p[0] * p[0] - p[1] * p[1]);
    }
    return poly;
  });
  P.polyPath = (ax, poly) => (poly.length ? poly.map((p, i) => (i ? 'L' : 'M') + H.r2(ax.X(p[0])) + ',' + H.r2(ax.Y(p[1]))).join(' ') + ' Z' : 'M0,0');

  /** a class-coloured region raster from a predict-one function */
  P.regions = (ax, key, f, o) => S.Kit.scatter.field(ax, key, f, Object.assign({ n: 90, scheme: 'classes', smooth: false, alpha: 0.32 }, o || {}));

  // ---- the messages as a bag of words -------------------------------------------------------------------------
  P.words = (text) => (String(text).toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g) || []);
  /** the vocabulary (sorted) and count rows of the labelled messages, leaving out any whose text is in skip */
  P.bag = (rows, skip) => H.once('p4bag:' + (skip || ''), () => {
    const use = rows.filter((r) => r[0] !== skip), vocab = [...new Set(use.flatMap((r) => P.words(r[0])))].sort(), at = new Map(vocab.map((w, j) => [w, j]));
    const X = use.map((r) => { const c = new Array(vocab.length).fill(0); for (const w of P.words(r[0])) c[at.get(w)]++; return c; });
    return { vocab, at, X, y: use.map((r) => r[1]), texts: use.map((r) => r[0]) };
  });
  P.vector = (bag, text) => { const c = new Array(bag.vocab.length).fill(0); for (const w of P.words(text)) if (bag.at.has(w)) c[bag.at.get(w)]++; return c; };
  /** a log-probability for narration: natural log, 2 decimals */
  P.lp = (v) => H.fix(v, 2);
})(globalThis.Nooral = globalThis.Nooral || {});
