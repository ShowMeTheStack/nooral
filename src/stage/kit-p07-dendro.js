// Dendrograms on the stage: the merge list of agglomerative clustering (merges[t] = {a, b, dist, size}, the new
// cluster's id n + t) drawn as a tree whose leaves are the points, in an order where no two links cross, and whose
// joins stand at the merge distance. Drawn up to any merge, with the newest link lit and every link coloured by the
// cluster it falls in once the tree is cut.
//   const L = S.Kit.dendro.layout(merges, n)  -> {order: [leaf..], x(id) in 0..1 slots, h(id)}
//   S.Kit.dendro.draw(L, {prefix, x, y, w, h, hmax, upto, cut, colour(id) -> class, hot: merge index}) -> items
//   S.Kit.dendro.cutClusters(merges, n, height) -> labels (0..k−1, numbered by first leaf in the drawn order)
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const D = (K.dendro = {});
  const r2 = (v) => Math.round(v * 100) / 100;

  D.layout = function (merges, n) {
    const kids = (id) => (id < n ? null : [merges[id - n].a, merges[id - n].b]);
    const order = [];
    const walk = (id) => { const k = kids(id); if (!k) { order.push(id); return; } walk(k[0]); walk(k[1]); };
    walk(n + merges.length - 1);
    const pos = new Map();
    order.forEach((leaf, i) => pos.set(leaf, i));
    const xs = new Map(), hs = new Map();
    for (let i = 0; i < n; i++) { xs.set(i, pos.get(i)); hs.set(i, 0); }
    merges.forEach((m, t) => { xs.set(n + t, (xs.get(m.a) + xs.get(m.b)) / 2); hs.set(n + t, m.dist); });
    return { order, n, merges, x: (id) => xs.get(id), h: (id) => hs.get(id), leaves: (id) => { const out = []; const w = (j) => { const k = kids(j); if (!k) out.push(j); else { w(k[0]); w(k[1]); } }; w(id); return out; } };
  };

  /** the clusters below a cut height: labels per point, numbered in the order their first leaf is drawn */
  D.cutClusters = function (merges, n, height) {
    const parent = [...Array(2 * n - 1).keys()], find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    merges.forEach((m, t) => { if (m.dist <= height) { parent[find(m.a)] = n + t; parent[find(m.b)] = n + t; } });
    const L = D.layout(merges, n), ids = new Map(), labels = new Array(n);
    for (const leaf of L.order) { const r = find(leaf); if (!ids.has(r)) ids.set(r, ids.size); labels[leaf] = ids.get(r); }
    return labels;
  };

  D.draw = function (L, o) {
    const P = o.prefix || 'dg', n = L.n, upto = o.upto === undefined ? L.merges.length - 1 : o.upto, items = [];
    const slot = o.w / n, X = (id) => r2(o.x + (L.x(id) + 0.5) * slot), Y = (h) => r2(o.y + o.h - (h / o.hmax) * o.h);
    for (let t = 0; t <= upto; t++) {
      const m = L.merges[t], top = Y(m.dist);
      const cls = 'p7link ' + (t === o.hot ? 'new' : o.colour ? o.colour(n + t) || '' : '');
      items.push({ key: P + ':m' + t, kind: 'path', d: `M${X(m.a)},${Y(L.h(m.a))} V${top} H${X(m.b)} V${Y(L.h(m.b))}`, cls, pathAudit: 'edge-x', layer: 'lines', z: t === o.hot ? 9 : 1 });
    }
    // the leaves, as small dots on the base line
    for (let i = 0; i < n; i++) items.push({ key: P + ':l' + i, kind: 'dot', x: X(i), y: Y(0), r: Math.min(4, slot * 0.32), cls: 'sp ' + (o.leafCls ? o.leafCls(i) : 'cn'), audit: 'overlap-ok', layer: 'items' });
    // the height axis on the left
    const ticks = S.Kit.curve.ticks(0, o.hmax, 4).filter((v) => v <= o.hmax + 1e-9);
    items.push({ key: P + ':ax', kind: 'path', d: `M${r2(o.x - 8)},${r2(o.y)} V${r2(o.y + o.h)}`, cls: 'axis', audit: 'ignore', layer: 'back' });
    for (const v of ticks) items.push({ key: P + ':t' + v, kind: 'text', x: r2(o.x - 13), y: r2(Y(v) + 4), text: S.Kit.curve.fmt(v), anchor: 'end', size: 12, cls: 'tick' });
    if (o.yname) items.push({ key: P + ':yn', kind: 'text', x: r2(o.x - 8), y: r2(o.y - 12), text: o.yname, size: 13, cls: 'axname' });
    if (o.cut !== undefined && o.cut !== null) {
      const cy = Y(Math.min(o.cut, o.hmax));
      items.push({ key: P + ':cut', kind: 'path', d: `M${r2(o.x - 8)},${cy} H${r2(o.x + o.w + 4)}`, cls: 'p7cut', audit: 'ignore', layer: 'top' });
      if (o.cutLabel) items.push({ key: P + ':cutl', kind: 'text', x: r2(o.x + o.w + 10), y: r2(cy + 4), text: o.cutLabel, anchor: 'start', size: 12, cls: 'p7cut' });
    }
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
