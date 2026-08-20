// Part 5, trees and ensembles: the families, and the helpers its topics share: two-feature data sets, a tree from
// the engine (ML.growTree's nodes) drawn as a diagram with fixed places, the regions and cut lines of a tree over a
// scatter, and small boundary panels for ensembles.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'trees', order, title, blurb });
  fam('p05-cart', 10, 'Growing a decision tree', 'CART tries every threshold on every feature, keeps the cut that leaves the purest halves, and repeats on each half.');
  fam('p05-size', 20, 'How big a tree', 'Depth, the smallest leaf allowed and cost-complexity pruning: the knobs that stop a tree from memorising its training data.');
  fam('p05-forest', 30, 'Bagging and random forests', 'Many trees on bootstrap samples, each split choosing among a random few features, voting together; the rows a tree never saw grade it for free.');
  fam('p05-boost', 40, 'Boosting', 'Small trees added one after another, each aimed at what the ones before got wrong: AdaBoost reweights the mistakes, gradient boosting fits the residuals.');
  fam('p05-compare', 50, 'Ensembles side by side', 'One tree against bagging, a forest, AdaBoost and gradient boosting on the same points.');

  const H = S.PL, ML = () => S.ML;
  const T5 = (S.PL5 = {});
  const r2 = (v) => Math.round(v * 100) / 100;

  // ---- data -------------------------------------------------------------------------------------------------
  T5.SETS = [
    { value: 'moons', label: 'two moons' }, { value: 'circles', label: 'circles' }, { value: 'xor', label: 'XOR' },
    { value: 'blobs', label: 'three blobs' }, { value: 'flowers', label: 'wildflowers (petals)' }, { value: 'exam', label: 'exam (study, sleep)' },
  ];
  /** a two-feature classification set: {X, y, names, classes, xr, yr} */
  T5.data2 = (set, n, seed, noise) => H.once('p5d:' + [set, n, seed, noise].join(','), () => {
    const D = S.Data;
    let ds, names = ['x1', 'x2'], classes, short;
    if (set === 'moons') ds = D.moons({ n, noise: noise == null ? 0.25 : noise, seed });
    else if (set === 'circles') ds = D.circles({ n, noise: noise == null ? 0.12 : noise, seed, factor: 0.5 });
    else if (set === 'xor') ds = D.xor({ n, noise: noise == null ? 0.05 : noise, seed });
    else if (set === 'blobs') ds = D.blobs({ n, std: noise == null ? 1.1 : 0.6 + 3 * noise, seed, centers: [[-2, -1], [2, -1.5], [0, 2.2]] });
    else if (set === 'flowers') { const t = D.load('wildflowers', { n, seed }); ds = { X: t.X.map((r) => [r[0], r[1]]), y: t.y }; names = ['petal length', 'petal width']; short = ['length', 'width']; classes = t.classes; }
    else if (set === 'exam') { const t = D.load('exam', { n, seed }); ds = { X: t.X.map((r) => [r[0], r[1]]), y: t.y }; names = ['study', 'sleep']; classes = ['fail', 'pass']; }
    else throw new Error('no set ' + set);
    const X = ds.X.map((r) => r.slice()), y = ds.y.slice(), K = Math.max(...y) + 1;
    return { X, y, names, short: short || names, classes: classes || [...Array(K).keys()].map((k) => 'class ' + k), K, xr: H.nice(X.map((r) => r[0]), 0.04), yr: H.nice(X.map((r) => r[1]), 0.04) };
  });
  /** a table with several features: {X, y, names, classes, K} (short names fit a cell) */
  const SHORT = { loans: ['income', 'debt', 'years', 'missed', 'amount', 'group'], wildflowers: ['length', 'width', 'stem', 'leaves'], exam: ['study', 'sleep', 'papers'] };
  T5.TABLES = [{ value: 'wildflowers', label: 'wildflowers (4 features)' }, { value: 'exam', label: 'exam (3 features)' }, { value: 'loans', label: 'small loans (6 features)' }];
  T5.table = (id, n, seed) => H.once('p5t:' + [id, n, seed].join(','), () => {
    const t = S.Data.load(id, { n, seed });
    return { X: t.X.map((r) => r.slice()), y: t.y.slice(), names: SHORT[id].slice(), classes: t.classes.slice(), K: t.classes.length };
  });
  /** a train/test split of a set (the test rows drawn with a different seed) */
  T5.split2 = (set, n, seed, noise) => ({ train: T5.data2(set, n, seed, noise), test: T5.data2(set, n, seed + 1000, noise) });
  /** the majority class names in plain words */
  T5.clsName = (d, k) => d.classes[k];

  // ---- a tree from the engine ---------------------------------------------------------------------------------
  /** the depth-first ids of a tree's nodes under (and including) id */
  T5.subtree = (T, id) => { const out = [], go = (i) => { out.push(i); const n = T.nodes[i]; if (n.left >= 0) { go(n.left); go(n.right); } }; go(id); return out; };
  /** the training rows that reach a node */
  T5.rowsAt = (T, X, id) => X.map((x, i) => i).filter((i) => ML().treePath(T, X[i]).includes(id));
  /** the leaf a point reaches when only the nodes in `open` (a Set, or null for all) have been split */
  T5.leafOf = (T, x, open) => { let n = T.nodes[0]; while (n.left >= 0 && (!open || open.has(n.id))) n = T.nodes[x[n.feature] <= n.threshold ? n.left : n.right]; return n; };
  /** the class a node predicts (classification trees) */
  T5.major = (node) => ML().argmax(node.value);
  /**
   * the tree diagram, laid out once on the whole tree so nodes keep their places while it grows:
   * {items(open, o) -> stage items}; open: Set of split nodes shown open (null: all); o.act: id lit, o.cls(node) extra classes
   */
  T5.diagram = (T, o) => {
    const P = o.prefix || 'td', N = T.nodes;
    const leaves = (i) => (N[i].left < 0 ? 1 : leaves(N[i].left) + leaves(N[i].right));
    const depth = Math.max(...N.map((n) => n.depth)), nl = leaves(0);
    const gap = o.gap === undefined ? 6 : o.gap, bw = Math.min(o.maxW || 96, (o.w - gap * (nl - 1)) / nl), bh = o.boxH || 44;
    const levelH = Math.max(bh + 12, Math.min(o.maxLevel || 64, depth ? (o.h - bh) / depth : 64));
    const make = (i) => ({ id: i, kids: N[i].left >= 0 ? [make(N[i].left), make(N[i].right)] : [] });
    const root = make(0);
    S.Kit.tree.layout(root, { x: 0, y: o.y + bh / 2, levelH, boxW: bw, gapX: gap });
    const b = S.Kit.tree.bounds(root, bw, bh), dx = o.x + o.w / 2 - (b.x + b.w / 2);
    const pos = {};
    const walk = (n) => { pos[n.id] = { x: r2(n.x + dx), y: n.y }; n.kids.forEach(walk); };
    walk(root);
    const fs = o.size || (bw < 56 ? 11 : 12);
    return {
      bw, bh, pos, levelH,
      items(open, oo) {
        oo = oo || {};
        const vis = (i) => {
          const n = N[i], isOpen = n.left >= 0 && (!open || open.has(i));
          return { id: i, x: pos[i].x, y: pos[i].y, lines: o.label(n, isOpen, bw), cls: (o.cls ? o.cls(n, isOpen) : '') + (oo.act === i ? ' act' : '') + (oo.cls ? ' ' + (oo.cls(n, isOpen) || '') : ''), bar: o.bar === false ? null : o.bar ? o.bar(n) : null, kids: isOpen ? [vis(n.left), vis(n.right)] : [], edge: '' };
        };
        const r = vis(0);
        if (o.edges) { const lab = (n) => { if (n.kids.length) { n.kids[0].edge = o.edges ? o.edges[0] : '≤'; n.kids[1].edge = o.edges ? o.edges[1] : '>'; n.kids.forEach(lab); } }; lab(r); }
        return S.Kit.tree.draw(r, { prefix: P, boxW: bw, boxH: bh, size: fs, edgeSize: 12 });
      },
    };
  };
  /** a dot on a curve plot that may sit on another (two series meeting) */
  T5.dot = (ax, key, u, v, cls) => Object.assign(S.Kit.curve.dot(ax, key, u, v, cls), { point: false, audit: 'overlap-ok' });
  /** the height a tree diagram needs for its depth */
  T5.treeHeight = (T, bh, gap) => Math.max(...T.nodes.map((x) => x.depth)) * (bh + (gap || 14)) + bh;
  /** the boxes of the regions a two-feature tree makes, for its nodes in `open` */
  T5.cuts = (ax, T, open, xr, yr, cls) => {
    let d = '';
    const walk = (i, bx, by) => {
      const n = T.nodes[i];
      if (n.left < 0 || (open && !open.has(i))) return;
      const t = n.threshold;
      if (n.feature === 0) { d += `M${ax.X(t)},${ax.Y(Math.min(by[1], yr[1]))} V${ax.Y(Math.max(by[0], yr[0]))} `; walk(n.left, [bx[0], t], by); walk(n.right, [t, bx[1]], by); }
      else { d += `M${ax.X(Math.max(bx[0], xr[0]))},${ax.Y(t)} H${ax.X(Math.min(bx[1], xr[1]))} `; walk(n.left, bx, [by[0], t]); walk(n.right, bx, [t, by[1]]); }
    };
    walk(0, [-1e9, 1e9], [-1e9, 1e9]);
    return { key: (cls && cls.key) || 'cuts', kind: 'path', d: d.trim() || 'M0,0', cls: 'pl-cut ' + ((cls && cls.cls) || ''), pathAudit: 'edge-x', layer: 'lines' };
  };
  /** the box in data coordinates of one node (for lighting the region being split) */
  T5.nodeBox = (T, id, xr, yr) => {
    const N = T.nodes, box = [xr.slice(), yr.slice()];
    let n = N[id];
    while (n.parent >= 0) {
      const p = N[n.parent], b = box[p.feature];
      if (p.left === n.id) b[1] = Math.min(b[1], p.threshold); else b[0] = Math.max(b[0], p.threshold);
      n = p;
    }
    return box;
  };
  T5.boxPath = (ax, key, box, cls) => {
    const x0 = ax.X(box[0][0]), x1 = ax.X(box[0][1]), y0 = ax.Y(box[1][1]), y1 = ax.Y(box[1][0]);
    return { key, kind: 'path', d: `M${x0},${y0} H${x1} V${y1} H${x0} Z`, cls: 'pl-box ' + (cls || ''), audit: 'ignore', layer: 'back', z: 6 };
  };
  /** the shaded classes of any classifier f(u, v) -> [class, confidence 0..1] over a plot */
  T5.field = (ax, key, f, n) => S.Kit.scatter.field(ax, key, f, { n: n || 70, scheme: 'classes', smooth: false });
  /** the shaded regions of a tree, its nodes in `open` split */
  T5.treeField = (ax, key, T, open, n) => T5.field(ax, key, (u, v) => { const nd = T5.leafOf(T, [u, v], open); const k = T5.major(nd); return [k, Math.max(0.25, nd.value[k])]; }, n);
  /** accuracy of predictions */
  T5.acc = (y, p) => { let c = 0; y.forEach((v, i) => { if (v === p[i]) c++; }); return c / y.length; };
  /** a small boundary panel for a model: frame, field, points, a title above */
  T5.panel = (key, box, d, f, title, o) => {
    o = o || {};
    const ax = H.axes({ prefix: key, x: box.x, y: box.y, w: box.w, h: box.h, xr: d.xr, yr: d.yr, xticks: 0, yticks: 0, grid: false, frame: 'box' });
    const items = [...ax.items, T5.field(ax, key + ':f', f, o.n || 44), ...S.Kit.scatter.points(ax, key + ':p', d.X, d.y, { r: o.r || 2.6, cls: o.pcls })];
    if (title) items.push(H.text(key + ':t', box.x, box.y - 8, title, o.tcls || 'pl-lab', 13));
    return { items, ax };
  };
  /** the impurity of a vector of class counts */
  T5.impurity = (crit, counts) => {
    const W = counts.reduce((a, b) => a + b, 0);
    if (!W) return 0;
    if (crit === 'entropy') return -counts.reduce((s, c) => (c > 0 ? s + (c / W) * Math.log2(c / W) : s), 0);
    return 1 - counts.reduce((s, c) => s + (c / W) ** 2, 0);
  };
})(globalThis.Nooral = globalThis.Nooral || {});
