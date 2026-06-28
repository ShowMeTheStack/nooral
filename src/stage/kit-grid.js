// Grid worlds on the stage: a board of cells (walls, a goal, a pit, the start), each open cell split into four
// triangles coloured by the value of moving north, east, south or west from it, the best move drawn as an arrow,
// and an agent that walks.
//   S.Kit.grid.world({prefix, x, y, cell, rows, cols, kinds(r, c) -> 'wall' | 'goal' | 'pit' | 'start' | '',
//                     q(r, c) -> [n, e, s, w] | null, qmax, best(r, c) -> 0..3 | -1, values: true (numbers in
//                     the triangles), names: {goal, pit, start}, agent: [r, c], hot: [r, c]}) -> items
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const GW = (K.grid = {});
  const r2 = (v) => Math.round(v * 100) / 100;
  // the four moves, and where each one's triangle points from the cell's centre
  GW.DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  GW.NAMES = ['north', 'east', 'south', 'west'];

  GW.world = function (o) {
    const P = o.prefix || 'gw', c = o.cell || 80, items = [], qmax = o.qmax || 1;
    const fmt = o.fmt || ((v) => (Math.abs(v) < 0.005 ? '0' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(2).replace(/^0\./, '.')));
    for (let r = 0; r < o.rows; r++) for (let col = 0; col < o.cols; col++) {
      const x = o.x + col * c, y = o.y + r * c, kind = o.kinds(r, col) || '', id = r + '_' + col;
      const q = kind === 'wall' || kind === 'goal' || kind === 'pit' ? null : o.q ? o.q(r, col) : null;
      const hot = o.hot && o.hot[0] === r && o.hot[1] === col;
      items.push({ key: P + ':c' + id, kind: 'cell', x: r2(x), y: r2(y), w: c, h: c, rx: 0, text: kind === 'goal' ? (o.names && o.names.goal) || '+1' : kind === 'pit' ? (o.names && o.names.pit) || '−1' : '', size: 16, cls: 'gcell ' + kind + (hot ? ' hotcell' : ''), audit: q ? 'ignore' : undefined, layer: 'back', z: 1 });
      if (q) {
        const cx = x + c / 2, cy = y + c / 2, corners = [[x, y], [x + c, y], [x + c, y + c], [x, y + c]];
        for (let d = 0; d < 4; d++) {
          const a = corners[d], b = corners[(d + 1) % 4], v = q[d], t = Math.max(-1, Math.min(1, v / qmax));
          const fill = `color-mix(in srgb, var(${t >= 0 ? '--pos' : '--neg'}) ${Math.round(Math.abs(t) * 70)}%, var(--div-mid))`;
          items.push({ key: P + ':t' + id + '_' + d, kind: 'path', d: `M${r2(cx)},${r2(cy)} L${r2(a[0])},${r2(a[1])} L${r2(b[0])},${r2(b[1])} Z`, cls: 'qtri', paint: { fill, stroke: 'var(--st-bg)', sw: 1 }, audit: 'ignore', layer: 'back', z: 2 });
          if (o.values) {
            const tx = cx + GW.DIRS[d][1] * c * 0.335, ty = cy + GW.DIRS[d][0] * c * 0.33 + 4;
            items.push({ key: P + ':v' + id + '_' + d, kind: 'text', x: r2(tx), y: r2(ty), text: fmt(v), anchor: 'middle', size: o.valueSize || 11, cls: 'qval', paint: { text: Math.abs(t) > 0.62 ? 'var(--st-bg)' : 'var(--text)' } });
          }
        }
        const b = o.best ? o.best(r, col) : -1;
        if (b >= 0) {
          const L = c * (o.values ? 0.13 : 0.26), dx = GW.DIRS[b][1], dy = GW.DIRS[b][0];
          items.push({ key: P + ':a' + id, kind: 'wire', pts: [[r2(cx - dx * L), r2(cy - dy * L)], [r2(cx + dx * L), r2(cy + dy * L)]], head: true, cls: 'greedy' });
        }
      }
      // the start is named in its corner, or under the board when the corner holds a value
      if (kind === 'start' && !o.agent) {
        if (!o.values) items.push({ key: P + ':s', kind: 'text', x: r2(x + 6), y: r2(y + 16), text: (o.names && o.names.start) || 'start', size: 12, cls: 'gstart' });
        else if (r === o.rows - 1) items.push({ key: P + ':s', kind: 'text', x: r2(x + c / 2), y: r2(y + c + 18), text: '↑ ' + ((o.names && o.names.start) || 'start'), anchor: 'middle', size: 12, cls: 'gstart' });
      }
    }
    items.push({ key: P + ':frame', kind: 'path', d: `M${o.x},${o.y} h${o.cols * c} v${o.rows * c} h${-o.cols * c} Z`, cls: 'gframe', audit: 'ignore', layer: 'top' });
    if (o.agent) items.push({ key: P + ':agent', kind: 'token', x: r2(o.x + o.agent[1] * c + c / 2), y: r2(o.y + o.agent[0] * c + c / 2), r: c * 0.16, text: '', cls: 'agent', audit: 'overlap-ok' });
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
