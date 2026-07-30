// The canvas's router: a wire never runs through or under a block. The tidy-up lays wires out clear of every block;
// a wire that would cross a block as drawn (a block dragged into its way, a lane the tidy-up left that a block now
// covers) is routed around the blocks instead: in straight runs with rounded corners, along the channels between the
// blocks, as short as it can be with as few turns as it can have (a shortest path over the grid of the blocks' edges).
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const PAD = 12, STUB = 16, TURN = 40;

  const inside = (p, r) => p.x > r.x + 1e-6 && p.x < r.x + r.w - 1e-6 && p.y > r.y + 1e-6 && p.y < r.y + r.h - 1e-6;
  /** does an axis-parallel run from p to q cross the inside of r? */
  const runHits = (p, q, r) => {
    if (p.y === q.y) return p.y > r.y && p.y < r.y + r.h && Math.max(p.x, q.x) > r.x && Math.min(p.x, q.x) < r.x + r.w;
    return p.x > r.x && p.x < r.x + r.w && Math.max(p.y, q.y) > r.y && Math.min(p.y, q.y) < r.y + r.h;
  };
  /** the boxes every wire keeps clear of: each block, padded */
  WF.obstacles = (doc, pad) => doc.blocks.map((b) => { const g = WF.geom(b, doc), d = pad === undefined ? PAD : pad; return { id: b.id, x: b.x - d, y: b.y - d, w: g.w + 2 * d, h: g.h + 2 * d }; });

  /**
   * a route from output port a to input port b around the boxes: the corner points between them (marked c: 1, so the
   * wire is drawn in straight runs with rounded corners), or null when there is none
   */
  WF.route = function (doc, a, b, boxes) {
    boxes = boxes || WF.obstacles(doc);
    const s = { x: a.x + STUB, y: a.y }, t = { x: b.x - STUB, y: b.y };
    const uniq = (v) => [...new Set(v.map((x) => Math.round(x * 2) / 2))].sort((p, q) => p - q);
    let xs = uniq([s.x, t.x].concat(...boxes.map((r) => [r.x, r.x + r.w])));
    let ys = uniq([s.y, t.y].concat(...boxes.map((r) => [r.y, r.y + r.h])));
    // the middles of the channels between blocks, so a wire runs down the middle of a gap rather than along an edge
    const mids = (v) => v.slice(1).map((x, i) => (x + v[i]) / 2).filter((m, i) => v[i + 1] - v[i] > 6);
    if (boxes.length <= 40) { xs = uniq(xs.concat(mids(xs))); ys = uniq(ys.concat(mids(ys))); }
    // room around the whole drawing, for a wire that must go round everything
    const lo = (v) => v[0] - 3 * PAD, hi = (v) => v[v.length - 1] + 3 * PAD;
    xs = [lo(xs)].concat(xs, [hi(xs)]); ys = [lo(ys)].concat(ys, [hi(ys)]);
    const nx = xs.length, ny = ys.length, N = nx * ny;
    const pt = (k) => ({ x: xs[k % nx], y: ys[Math.floor(k / nx)] });
    const free = new Uint8Array(N);
    for (let k = 0; k < N; k++) { const p = pt(k); free[k] = boxes.some((r) => inside(p, r)) ? 0 : 1; }
    const si = xs.indexOf(Math.round(s.x * 2) / 2) + nx * ys.indexOf(Math.round(s.y * 2) / 2);
    const ti = xs.indexOf(Math.round(t.x * 2) / 2) + nx * ys.indexOf(Math.round(t.y * 2) / 2);
    free[si] = 1; free[ti] = 1;
    // Dijkstra over (grid point, the direction it was entered in): length, and a cost for every turn
    const DX = [1, 0, -1, 0], DY = [0, 1, 0, -1];
    const dist = new Float64Array(N * 4).fill(Infinity), prev = new Int32Array(N * 4).fill(-1);
    const heap = [];
    const push = (d, st) => { heap.push([d, st]); let i = heap.length - 1; while (i > 0) { const j = (i - 1) >> 1; if (heap[j][0] <= heap[i][0]) break; [heap[i], heap[j]] = [heap[j], heap[i]]; i = j; } };
    const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[i], heap[m]] = [heap[m], heap[i]]; i = m; } } return top; };
    dist[si * 4] = 0; push(0, si * 4);
    let end = -1;
    while (heap.length) {
      const [d, st] = pop();
      if (d > dist[st]) continue;
      const k = st >> 2, dir = st & 3;
      // the wire enters its port from the left: arriving at the stub going right costs nothing more, else one turn
      if (k === ti) { end = st; if (dir === 0) break; }
      const ix = k % nx, iy = Math.floor(k / nx), p = pt(k);
      for (let nd = 0; nd < 4; nd++) {
        if (nd === (dir + 2) % 4) continue;
        const jx = ix + DX[nd], jy = iy + DY[nd];
        if (jx < 0 || jy < 0 || jx >= nx || jy >= ny) continue;
        const j = jx + nx * jy;
        if (!free[j]) continue;
        const q = pt(j);
        if (boxes.some((r) => runHits(p, q, r))) continue;
        const c = d + Math.abs(q.x - p.x) + Math.abs(q.y - p.y) + (nd !== dir ? TURN : 0), ns = j * 4 + nd;
        if (c < dist[ns]) { dist[ns] = c; prev[ns] = st; push(c, ns); }
      }
    }
    if (end < 0) return null;
    // the best way into the stub, turns counted
    let best = -1;
    for (let dd = 0; dd < 4; dd++) { const st = ti * 4 + dd; if (dist[st] < Infinity && (best < 0 || dist[st] + (dd ? TURN : 0) < dist[best] + (best & 3 ? TURN : 0))) best = st; }
    const path = [];
    for (let st = best; st >= 0; st = prev[st]) path.unshift(pt(st >> 2));
    // keep the corners only
    const all = [a].concat(path, [b]), out = [];
    for (let i = 1; i + 1 < all.length; i++) {
      const p = all[i - 1], q = all[i], n = all[i + 1];
      const straight = (Math.abs(p.x - q.x) < 1e-6 && Math.abs(q.x - n.x) < 1e-6) || (Math.abs(p.y - q.y) < 1e-6 && Math.abs(q.y - n.y) < 1e-6);
      if (!straight && (Math.abs(q.x - p.x) > 1e-6 || Math.abs(q.y - p.y) > 1e-6)) out.push({ x: q.x, y: q.y, c: 1 });
    }
    return out.length ? out : null;
  };

  /** does the wire through these points cross a block's inside (its own ends included)? */
  WF.pathHits = function (pts, boxes) {
    const seg = (p, q, r) => {
      let t0 = 0, t1 = 1;
      const dx = q.x - p.x, dy = q.y - p.y;
      for (const [pp, qq] of [[-dx, p.x - r.x], [dx, r.x + r.w - p.x], [-dy, p.y - r.y], [dy, r.y + r.h - p.y]]) {
        if (Math.abs(pp) < 1e-12) { if (qq <= 0) return false; continue; }
        const u = qq / pp;
        if (pp < 0) { if (u > t1) return false; if (u > t0) t0 = u; } else { if (u < t0) return false; if (u < t1) t1 = u; }
      }
      return t1 - t0 > 1e-6;
    };
    for (let i = 1; i < pts.length; i++) for (const r of boxes) if (seg(pts[i - 1], pts[i], r)) return true;
    return false;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
