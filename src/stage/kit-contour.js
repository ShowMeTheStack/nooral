// Loss surfaces on the stage: a function of two parameters drawn as a shaded raster with level lines (spaced
// evenly in the logarithm of the height, so the valley floor shows as well as the walls), and the paths
// optimisers take across it, step by step.
//   S.Kit.contour.surface(ax, key, f(u, v), {levels: n, n, log, min}) -> items: the shading and the level lines
//     (the lines are background: optimiser paths cross them)
//   S.Kit.contour.path(ax, key, pts [[u, v]..], cls, {upto, dots}) -> the path so far and a dot at its head
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const CT = (K.contour = {});
  const r2 = (v) => Math.round(v * 100) / 100;

  CT.surface = function (ax, key, f, o) {
    o = o || {};
    const SC = S.Kit.scatter, n = o.n || 90, min = o.min === undefined ? 0 : o.min;
    const lg = (v) => Math.log(1 + Math.max(0, v - min));
    const g = SC.grid(ax, (u, v) => (o.log === false ? f(u, v) : lg(f(u, v))), n);
    let hi = -Infinity, lo = Infinity;
    for (const v of g.vals) { hi = Math.max(hi, v); lo = Math.min(lo, v); }
    const items = [SC.field(ax, key + ':sh', (u, v) => 1 - ((o.log === false ? f(u, v) : lg(f(u, v))) - lo) / (hi - lo || 1), { n: Math.round(n * 0.8), scheme: 'seq', max: o.depth || 1.8, smooth: true })];
    items[0].cls = 'surface';
    const L = o.levels || 9;
    let d = '';
    for (let k = 1; k <= L; k++) {
      const lv = lo + ((hi - lo) * Math.pow(k / (L + 1), 1.6));
      for (const line of SC.trace(g, lv)) if (line.length > 2) d += line.map((p, i) => (i ? 'L' : 'M') + r2(p[0]) + ',' + r2(p[1])).join(' ') + ' ';
    }
    items.push({ key: key + ':lv', kind: 'path', d: d.trim() || 'M0,0', cls: 'isoline', audit: 'ignore', layer: 'back', z: 4 });
    return items;
  };
  CT.path = function (ax, key, pts, cls, o) {
    o = o || {};
    const n = Math.max(1, Math.min(pts.length, o.upto === undefined ? pts.length : o.upto + 1)), use = pts.slice(0, n), items = [];
    if (use.length > 1) items.push({ key: key + ':p', kind: 'path', d: use.map((p, i) => (i ? 'L' : 'M') + ax.X(p[0]) + ',' + ax.Y(p[1])).join(' '), cls: 'optpath ' + (cls || ''), pathAudit: 'edge-x', layer: 'lines' });
    if (o.dots) use.forEach((p, i) => { if (i && i < use.length - 1 && i % o.dots === 0) items.push({ key: key + ':d' + i, kind: 'dot', x: ax.X(p[0]), y: ax.Y(p[1]), r: 2.4, cls: 'optstep ' + (cls || ''), audit: 'ignore' }); });
    const h = use[use.length - 1];
    items.push({ key: key + ':h', kind: 'dot', x: ax.X(h[0]), y: ax.Y(h[1]), r: 6, cls: 'opthead ' + (cls || ''), audit: 'overlap-ok', layer: 'items' });
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
