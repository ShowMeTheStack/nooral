// Graph views for part 16 (graph neural networks): a seeded force layout that keeps every edge clear of the nodes
// it does not join, and the drawing of a graph as numbered nodes joined by links (arrows for a directed graph).
//   S.Kit.graph.layout(g, {x, y, w, h, r, seed, iters, groups}) -> [{x, y}] one place per node, inside the box
//     (groups: a group per node; nodes of a group start together, so communities come out as clumps)
//   S.Kit.graph.draw(g, pos, {prefix, r, text(i), nodeCls(i), edgeCls(a, b), paint(i), accept(i), directed, size}) -> items;
//     (accept(i): a second ring round node i, as for a labelled node)
//     node i has the key prefix:n<i>, the link between a and b (a < b unless directed) prefix:e<a>_<b>
//   S.Kit.graph.edgeKey(prefix, a, b, directed) -> that key
//   S.Kit.graph.fit(pos, box, r) -> the places scaled into a new box (the same shape kept)
(function (S) {
  'use strict';
  const K = (S.Kit = S.Kit || {});
  const G = (K.graph = {});
  const r2 = (v) => Math.round(v * 100) / 100;

  /** distance from point p to the segment a–b, and the foot of the perpendicular */
  const segDist = (p, a, b) => {
    const dx = b.x - a.x, dy = b.y - a.y, L = dx * dx + dy * dy || 1, t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / L));
    const fx = a.x + t * dx, fy = a.y + t * dy;
    return { d: Math.hypot(p.x - fx, p.y - fy), fx, fy, t };
  };

  G.fit = function (pos, box, r) {
    const xs = pos.map((p) => p.x), ys = pos.map((p) => p.y), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const iw = box.w - 2 * r, ih = box.h - 2 * r, s = Math.min(iw / (x1 - x0 || 1), ih / (y1 - y0 || 1));
    const ox = box.x + r + (iw - s * (x1 - x0)) / 2, oy = box.y + r + (ih - s * (y1 - y0)) / 2;
    return pos.map((p) => ({ x: r2(ox + (p.x - x0) * s), y: r2(oy + (p.y - y0) * s) }));
  };

  G.layout = function (g, o) {
    o = o || {};
    const n = g.n, rnd = S.Data.rng(o.seed || 7), r = o.r || 15;
    const und = [];
    const seen = new Set();
    for (const [a, b] of g.edges) { if (a === b) continue; const k = Math.min(a, b) + ',' + Math.max(a, b); if (!seen.has(k)) { seen.add(k); und.push([Math.min(a, b), Math.max(a, b)]); } }
    // start: on a circle, or each group round its own centre
    let P;
    if (o.groups) {
      const gs = [...new Set(o.groups)], m = gs.length;
      P = o.groups.map((c, i) => { const k = gs.indexOf(c), a = (2 * Math.PI * k) / m, b = 2 * Math.PI * rnd(); return { x: 3 * Math.cos(a) + Math.cos(b) * rnd(), y: 3 * Math.sin(a) + Math.sin(b) * rnd() }; });
    } else P = Array.from({ length: n }, (_, i) => ({ x: Math.cos((2 * Math.PI * i) / n) + 0.05 * rnd(), y: Math.sin((2 * Math.PI * i) / n) + 0.05 * rnd() }));
    const k = Math.sqrt(9 / Math.max(1, n)) * 1.2, iters = o.iters || 400;
    let temp = 0.6;
    for (let it = 0; it < iters; it++) {
      const D = P.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, d = Math.hypot(dx, dy);
        if (d < 1e-6) { dx = 1e-3 * (i - j); dy = 1e-3; d = Math.hypot(dx, dy); }
        const f = (k * k) / d;
        D[i].x += (dx / d) * f; D[i].y += (dy / d) * f; D[j].x -= (dx / d) * f; D[j].y -= (dy / d) * f;
      }
      for (const [a, b] of und) {
        const dx = P[a].x - P[b].x, dy = P[a].y - P[b].y, d = Math.hypot(dx, dy) || 1e-6, f = (d * d) / k;
        D[a].x -= (dx / d) * f; D[a].y -= (dy / d) * f; D[b].x += (dx / d) * f; D[b].y += (dy / d) * f;
      }
      // a gentle pull to the middle keeps loose parts from drifting away
      for (let i = 0; i < n; i++) { D[i].x -= 0.05 * P[i].x; D[i].y -= 0.05 * P[i].y; }
      for (let i = 0; i < n; i++) { const L = Math.hypot(D[i].x, D[i].y) || 1, s = Math.min(L, temp); P[i].x += (D[i].x / L) * s; P[i].y += (D[i].y / L) * s; }
      temp = Math.max(0.01, temp * 0.985);
    }
    const box = { x: o.x || 0, y: o.y || 0, w: o.w || 400, h: o.h || 300 };
    let Q = G.fit(P, box, r + 4);
    // repair in stage units: nodes apart, and no link passing over a node it does not join
    const gap = o.gap || Math.max(2 * r + 10, 0.55 * Math.sqrt((box.w * box.h) / Math.max(1, n))), clear = r + 7;
    for (let round = 0; round < 80; round++) {
      let moved = false;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const dx = Q[j].x - Q[i].x, dy = Q[j].y - Q[i].y, d = Math.hypot(dx, dy) || 1e-3;
        if (d < gap) { const s = (gap - d) / 2 + 0.5, ux = dx / d, uy = dy / d; Q[i].x -= ux * s; Q[i].y -= uy * s; Q[j].x += ux * s; Q[j].y += uy * s; moved = true; }
      }
      for (const [a, b] of und) for (let c = 0; c < n; c++) {
        if (c === a || c === b) continue;
        const s = segDist(Q[c], Q[a], Q[b]);
        if (s.d < clear && s.t > 0.02 && s.t < 0.98) {
          let ux = Q[c].x - s.fx, uy = Q[c].y - s.fy, L = Math.hypot(ux, uy);
          if (L < 1e-6) { ux = -(Q[b].y - Q[a].y); uy = Q[b].x - Q[a].x; L = Math.hypot(ux, uy) || 1; }
          const m = clear - s.d + 1;
          Q[c].x += (ux / L) * m; Q[c].y += (uy / L) * m;
          Q[a].x -= (ux / L) * m * 0.3; Q[a].y -= (uy / L) * m * 0.3; Q[b].x -= (ux / L) * m * 0.3; Q[b].y -= (uy / L) * m * 0.3;
          moved = true;
        }
      }
      if (!moved) break;
      Q = G.fit(Q, box, r + 4);
    }
    return Q.map((p) => ({ x: r2(p.x), y: r2(p.y) }));
  };

  G.edgeKey = (prefix, a, b, directed) => prefix + ':e' + (directed ? a + '_' + b : Math.min(a, b) + '_' + Math.max(a, b));

  G.draw = function (g, pos, o) {
    o = o || {};
    const P = o.prefix || 'g', r = o.r || 15, items = [], directed = !!o.directed, done = new Set();
    for (const [a, b] of g.edges) {
      if (a === b) continue;
      const key = G.edgeKey(P, a, b, directed);
      if (done.has(key)) continue;
      done.add(key);
      const p = pos[a], q = pos[b], dx = q.x - p.x, dy = q.y - p.y, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
      const ra = o.accept && o.accept(a) ? r + 5 : r, rb = o.accept && o.accept(b) ? r + 5 : r, end = directed ? rb + 2 : rb;
      items.push({ key, kind: 'path', d: `M${r2(p.x + ux * ra)},${r2(p.y + uy * ra)} L${r2(q.x - ux * end)},${r2(q.y - uy * end)}`, cls: 'glink ' + (directed ? 'dir ' : '') + (o.edgeCls ? o.edgeCls(a, b) || '' : ''), pathAudit: 'edge-x', layer: 'lines', z: o.edgeZ ? o.edgeZ(a, b) || 0 : 0, box: { x: Math.min(p.x, q.x), y: Math.min(p.y, q.y), w: Math.abs(dx), h: Math.abs(dy) } });
      if (directed) {
        // an arrowhead at the cited end, drawn as a small filled triangle
        const tx = q.x - ux * end, ty = q.y - uy * end, bx = tx - ux * 9, by = ty - uy * 9;
        items.push({ key: key + ':h', kind: 'path', d: `M${r2(tx)},${r2(ty)} L${r2(bx - uy * 4.5)},${r2(by + ux * 4.5)} L${r2(bx + uy * 4.5)},${r2(by - ux * 4.5)} Z`, cls: 'ghead ' + (o.edgeCls ? o.edgeCls(a, b) || '' : ''), audit: 'ignore', layer: 'lines', z: 1 });
      }
    }
    pos.forEach((p, i) => {
      const it = { key: P + ':n' + i, kind: 'state', x: p.x, y: p.y, r, text: o.text ? o.text(i) : String(i), size: o.size || (r < 13 ? 11 : 13), cls: 'gnode ' + (o.nodeCls ? o.nodeCls(i) || '' : '') };
      if (o.accept && o.accept(i)) it.accept = true;
      const paint = o.paint ? o.paint(i) : null;
      if (paint) it.paint = paint;
      items.push(it);
    });
    return items;
  };
})(globalThis.Nooral = globalThis.Nooral || {});
