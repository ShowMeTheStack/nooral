// Network diagrams on the stage: layers of neurons joined by their weights. A weight's sign is its colour (blue
// positive, vermilion negative) and its size is the line's thickness; a neuron's activation fills it. Dense layers
// cross one another by nature, so the weights are drawn as edges that may cross (edge-x), while still keeping
// clear of the other neurons and of labels.
//   const L = S.Kit.net.layout({x, y, w, h, sizes: [2, 5, 2], r, gapY})  -> {pos: [[{x, y}..]..], r}
//   S.Kit.net.draw(L, {prefix, weights: [W1 (out × in), ..], acts: [[..]..], names: [[..]..], wmax, amax,
//                      nodeCls(l, i), edgeCls(l, i, j), minW, labels: {side: 'left' | 'right', layer}})
//     -> items; neuron (l, i) has the key prefix:n<l>_<i>
//   S.Kit.net.weightPaint(w, wmax, wide) -> {stroke, sw} for one weight
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const N = (K.net = {});
  const r2 = (v) => Math.round(v * 100) / 100;

  N.layout = function (o) {
    const n = o.sizes.length, r = o.r || 16, pos = [];
    o.sizes.forEach((m, l) => {
      const x = n === 1 ? o.x + o.w / 2 : o.x + (l * o.w) / (n - 1);
      const gap = o.gapY || Math.min(4.2 * r, (o.h - 2 * r) / Math.max(1, Math.max(...o.sizes) - 1));
      const top = o.y + o.h / 2 - ((m - 1) * gap) / 2;
      pos.push(Array.from({ length: m }, (_, i) => ({ x: r2(x), y: r2(top + i * gap) })));
    });
    return { pos, r, sizes: o.sizes };
  };
  N.weightPaint = function (w, wmax, wide) {
    const t = Math.min(1, Math.abs(w) / (wmax || 1));
    return { stroke: `color-mix(in srgb, var(${w >= 0 ? '--pos' : '--neg'}) ${Math.round(35 + 65 * t)}%, var(--st-bg))`, sw: r2(0.8 + t * (wide || 6)) };
  };
  N.draw = function (L, o) {
    const P = o.prefix || 'net', items = [], r = L.r, wmax = o.wmax || 1;
    (o.weights || []).forEach((W, l) => {
      if (!W) return;
      W.forEach((row, i) => row.forEach((w, j) => {
        const a = L.pos[l][j], b = L.pos[l + 1][i];
        if (o.minW && Math.abs(w) < o.minW) return;
        const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
        const p = N.weightPaint(w, wmax, o.wide);
        items.push({ key: `${P}:w${l}_${i}_${j}`, kind: 'path', d: `M${r2(a.x + ux * (r + 1))},${r2(a.y + uy * (r + 1))} L${r2(b.x - ux * (r + 1))},${r2(b.y - uy * (r + 1))}`, cls: 'weight ' + (o.edgeCls ? o.edgeCls(l, i, j) || '' : ''), pathAudit: 'edge-x', paint: p, layer: 'lines', z: Math.round(Math.abs(w) * 100) });
      }));
    });
    L.pos.forEach((col, l) => col.forEach((p, i) => {
      const a = o.acts && o.acts[l] ? o.acts[l][i] : null, amax = o.amax || 1;
      const paint = a === null || a === undefined ? null : { fill: `color-mix(in srgb, var(${a >= 0 ? '--pos' : '--neg'}) ${Math.round(Math.min(1, Math.abs(a) / amax) * 75)}%, var(--node))`, text: Math.abs(a) / amax > 0.6 ? 'var(--st-bg)' : 'var(--text)' };
      const it = { key: `${P}:n${l}_${i}`, kind: 'state', x: p.x, y: p.y, r, text: o.names && o.names[l] ? o.names[l][i] || '' : '', size: o.nameSize || 13, cls: 'neuron ' + (o.nodeCls ? o.nodeCls(l, i) || '' : '') };
      if (paint) it.paint = paint;
      items.push(it);
    }));
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
