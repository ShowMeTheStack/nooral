// The geometry of the canvas and its tidy layout. A block is a box with its inputs on the left and its outputs on the
// right; a wire is a curve from an output to an input, through the points the layout left for it when it passes a
// column of blocks. Tidy-up lays the workflow out left to right in layers (each block one layer after what feeds it),
// orders every layer to remove crossings (barycentre sweeps and swaps, counted port by port), then places the blocks
// so wires run as straight as they can (isotonic regression of the wanted heights). No page here: node tests it.
(function (S) {
  'use strict';
  const WF = (S.WF = S.WF || {});
  const G0 = (WF.G = { W: 200, GRID: 12, GAPX: 84, GAPY: 30, LANE: 14, HEAD: 50, PORT: 22, ROWGAP: 72, MAXL: 4 });
  // a compact canvas (inside a lesson or a challenge): narrower blocks closer together, the ports named by their colour
  // and their tooltips, so a whole pipeline fits a column of text at full size
  const GC = (WF.GC = { W: 172, GRID: 12, GAPX: 46, GAPY: 24, LANE: 12, HEAD: 46, PORT: 18, ROWGAP: 60, MAXL: 4, compact: true });
  /** the sizes a document is drawn at */
  WF.dims = (doc) => (doc && doc._compact ? GC : G0);
  /** draw a document compact (or not): a mark that does not travel with it when it is saved */
  WF.setCompact = (doc, on) => { Object.defineProperty(doc, '_compact', { value: !!on, enumerable: false, configurable: true, writable: true }); return doc; };

  /** a block's box and ports: {w, h, ins: [{id, label, type, x, y}], outs: [...]}, ports relative to the block */
  WF.geom = function (b, doc) {
    const G = WF.dims(doc);
    const ports = WF.ports(b);
    const n = Math.max(1, ports.inputs.length, ports.outputs.length);
    const h = G.HEAD + n * G.PORT + 8;
    const at = (i, k) => G.HEAD + (i + 0.5) * ((n * G.PORT) / k) + 2;
    const ins = ports.inputs.map((p, i) => ({ id: p.id, label: p.label || p.id, type: p.type, optional: p.optional, x: 0, y: at(i, ports.inputs.length) }));
    const outs = ports.outputs.map((p, i) => ({ id: p.id, label: p.label || p.id, type: doc ? WF.outType(doc, b.id, p.id) || p.type : p.type, x: G.W, y: at(i, ports.outputs.length) }));
    return { w: G.W, h, ins, outs };
  };
  /** the page position of a port: {x, y} */
  WF.portAt = function (doc, id, port, side) {
    const b = WF.byId(doc, id);
    if (!b) return null;
    const g = WF.geom(b, doc), p = (side === 'in' ? g.ins : g.outs).find((q) => q.id === port);
    return p ? { x: b.x + p.x, y: b.y + p.y } : null;
  };
  /** a wire's path: SVG d, and the points along it (for hit tests and crossing counts) */
  /** a wire drafted in straight runs with rounded corners (a wire folded from one row of blocks to the next) */
  const cornerPath = (pts) => {
    let d = 'M' + r1(pts[0].x) + ',' + r1(pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i - 1], q = pts[i], n = pts[i + 1];
      if (!n) { d += ' L' + r1(q.x) + ',' + r1(q.y); break; }
      const r = Math.min(10, Math.hypot(q.x - p.x, q.y - p.y) / 2, Math.hypot(n.x - q.x, n.y - q.y) / 2);
      const u1 = { x: (q.x - p.x) / (Math.hypot(q.x - p.x, q.y - p.y) || 1), y: (q.y - p.y) / (Math.hypot(q.x - p.x, q.y - p.y) || 1) };
      const u2 = { x: (n.x - q.x) / (Math.hypot(n.x - q.x, n.y - q.y) || 1), y: (n.y - q.y) / (Math.hypot(n.x - q.x, n.y - q.y) || 1) };
      d += ' L' + r1(q.x - u1.x * r) + ',' + r1(q.y - u1.y * r) + ' Q' + r1(q.x) + ',' + r1(q.y) + ' ' + r1(q.x + u2.x * r) + ',' + r1(q.y + u2.y * r);
    }
    return d;
  };
  WF.wirePath = function (a, b, via) {
    if (via && via.length && via[0].c) return cornerPath([a].concat(via, [b]));
    const pts = [a].concat(via || [], [b]);
    let d = 'M' + r1(a.x) + ',' + r1(a.y);
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i - 1], q = pts[i];
      if (Math.abs(q.y - p.y) < 0.5 && q.x < p.x) { d += ' L' + r1(q.x) + ',' + r1(q.y); continue; }
      if (q.x >= p.x + 8) { const k = Math.max(26, (q.x - p.x) * 0.5); d += ' C' + r1(p.x + k) + ',' + r1(p.y) + ' ' + r1(q.x - k) + ',' + r1(q.y) + ' ' + r1(q.x) + ',' + r1(q.y); }
      else {
        // backwards: out to the right, round below, back in from the left
        const yy = Math.max(p.y, q.y) + 60;
        d += ' C' + r1(p.x + 70) + ',' + r1(p.y) + ' ' + r1(p.x + 70) + ',' + r1(yy) + ' ' + r1(p.x) + ',' + r1(yy) + ' L' + r1(q.x) + ',' + r1(yy) + ' C' + r1(q.x - 70) + ',' + r1(yy) + ' ' + r1(q.x - 70) + ',' + r1(q.y) + ' ' + r1(q.x) + ',' + r1(q.y);
      }
    }
    return d;
  };
  const r1 = (v) => Math.round(v * 10) / 10;
  /** points along a wire, for geometry: the same curves, sampled */
  WF.wirePoints = function (a, b, via, n) {
    if (via && via.length && via[0].c) return [a].concat(via.map((p) => ({ x: p.x, y: p.y })), [b]);
    const pts = [a].concat(via || [], [b]), out = [a];
    n = n || 16;
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i - 1], q = pts[i];
      if (Math.abs(q.y - p.y) < 0.5 && q.x < p.x) { out.push(q); continue; }
      if (q.x >= p.x + 8) {
        const k = Math.max(26, (q.x - p.x) * 0.5), c1 = [p.x + k, p.y], c2 = [q.x - k, q.y];
        for (let s = 1; s <= n; s++) { const t = s / n, u = 1 - t; out.push({ x: u * u * u * p.x + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * q.x, y: u * u * u * p.y + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * q.y }); }
      } else { const yy = Math.max(p.y, q.y) + 60; out.push({ x: p.x + 50, y: p.y }, { x: p.x + 50, y: yy }, { x: q.x - 50, y: yy }, { x: q.x - 50, y: q.y }, q); }
    }
    return out;
  };
  /** every wire of a document with its ends and path */
  const routed = new WeakMap();
  WF.wireGeom = function (doc) {
    // a wire that would run through or under a block as drawn is routed around the blocks (wf-ui-layout-route.js);
    // the routes are kept until a block or a wire changes
    let near = null, far = null;
    const key = doc.blocks.map((b) => b.id + ':' + b.x + ',' + b.y + ',' + WF.geom(b, doc).h).join(';') + '|' + doc.wires.map((w) => w.id + (w.via ? JSON.stringify(w.via) : '')).join(';') + '|' + !!doc._compact;
    const memo = routed.get(doc);
    if (memo && memo.key === key) return memo.out.map((g) => Object.assign({}, g));
    const out = doc.wires.map((w) => {
      const a = WF.portAt(doc, w.from[0], w.from[1], 'out'), b = WF.portAt(doc, w.to[0], w.to[1], 'in');
      if (!a || !b) return null;
      let via = w.via || null;
      if (WF.route && WF.pathHits) {
        near = near || WF.obstacles(doc, -3);
        if (WF.pathHits(WF.wirePoints(a, b, via, 32), near)) { far = far || WF.obstacles(doc); via = WF.route(doc, a, b, far) || via; }
      }
      return { w, a, b, via };
    }).filter(Boolean);
    routed.set(doc, { key, out });
    return out.map((g) => Object.assign({}, g));
  };
  const segX = (p1, p2, p3, p4) => {
    const d = (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x);
    if (Math.abs(d) < 1e-9) return false;
    const t = ((p3.x - p1.x) * (p4.y - p3.y) - (p3.y - p1.y) * (p4.x - p3.x)) / d, u = ((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / d;
    return t > 1e-6 && t < 1 - 1e-6 && u > 1e-6 && u < 1 - 1e-6;
  };
  /** how many pairs of wires cross as drawn (wires that share an end do not count where they meet) */
  WF.crossings = function (doc) {
    const ws = WF.wireGeom(doc).map((g) => Object.assign(g, { pts: WF.wirePoints(g.a, g.b, g.via, 24) }));
    let n = 0;
    for (let i = 0; i < ws.length; i++) for (let j = i + 1; j < ws.length; j++) {
      const A = ws[i], B = ws[j];
      if ((A.w.from[0] === B.w.from[0] && A.w.from[1] === B.w.from[1])) continue;
      let hit = false;
      for (let a = 1; a < A.pts.length && !hit; a++) for (let b = 1; b < B.pts.length && !hit; b++) if (segX(A.pts[a - 1], A.pts[a], B.pts[b - 1], B.pts[b])) hit = true;
      if (hit) n++;
    }
    return n;
  };

  /** does the segment p-q pass through the inside of box r? (Liang-Barsky clipping) */
  const segInBox = (p, q, r) => {
    let t0 = 0, t1 = 1;
    const dx = q.x - p.x, dy = q.y - p.y;
    for (const [pp, qq] of [[-dx, p.x - r.x], [dx, r.x + r.w - p.x], [-dy, p.y - r.y], [dy, r.y + r.h - p.y]]) {
      if (Math.abs(pp) < 1e-12) { if (qq <= 0) return false; continue; }
      const t = qq / pp;
      if (pp < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; }
    }
    return t1 - t0 > 1e-6;
  };
  /**
   * the wires that run through or under a block as drawn: [{wire, block}]. A wire may touch the blocks at its ends at
   * their ports, but never cross the inside of any block, its own ends included (margin: px kept clear inside the box)
   */
  WF.wireHits = function (doc, margin) {
    const m = margin === undefined ? 3 : margin, out = [];
    const boxes = doc.blocks.map((b) => { const g = WF.geom(b, doc); return { id: b.id, r: { x: b.x + m, y: b.y + m, w: g.w - 2 * m, h: g.h - 2 * m } }; });
    for (const g of WF.wireGeom(doc)) {
      const pts = WF.wirePoints(g.a, g.b, g.via, 32);
      for (const bx of boxes) {
        for (let i = 1; i < pts.length; i++) if (segInBox(pts[i - 1], pts[i], bx.r)) { out.push({ wire: g.w.id, block: bx.id }); break; }
      }
    }
    return out;
  };

  // ---- tidy-up ------------------------------------------------------------------------------------------------
  const layoutOne = function (doc) {
    const G = WF.dims(doc);
    if (!doc.blocks.length) return doc;
    const ids = doc.blocks.map((b) => b.id), geo = {};
    for (const b of doc.blocks) geo[b.id] = WF.geom(b, doc);
    const wires = doc.wires.filter((w) => geo[w.from[0]] && geo[w.to[0]]);
    // layers: longest path from the sources; a source sits just before the first block it feeds
    let order;
    try { order = WF.order(doc); } catch (e) { order = ids.slice(); }
    const L = {};
    for (const id of order) { const ins = wires.filter((w) => w.to[0] === id); L[id] = ins.length ? Math.max(...ins.map((w) => (L[w.from[0]] || 0) + 1)) : 0; }
    for (let pass = 0; pass < 3; pass++) for (const id of order.slice().reverse()) {
      if (wires.some((w) => w.to[0] === id)) continue;
      const outs = wires.filter((w) => w.from[0] === id);
      if (outs.length) L[id] = Math.max(0, Math.min(...outs.map((w) => L[w.to[0]])) - 1);
    }
    // nodes: blocks and the dummies of wires that span several layers
    const nodes = {};
    for (const id of ids) nodes[id] = { id, layer: L[id], h: geo[id].h, dummy: false };
    const edges = []; // {from: node, fp: port offset, to: node, tp}
    const frac = (g, side, port) => { const list = side === 'out' ? g.outs : g.ins; const i = list.findIndex((p) => p.id === port); return list.length ? (i + 0.5) / list.length - 0.5 : 0; };
    const chains = [];
    wires.forEach((w, wi) => {
      const a = w.from[0], b = w.to[0];
      let prev = a, pf = frac(geo[a], 'out', w.from[1]) * 0.8;
      const chain = [];
      for (let l = L[a] + 1; l < L[b]; l++) { const d = '~' + wi + '.' + l; nodes[d] = { id: d, layer: l, h: G.LANE, dummy: true }; edges.push({ from: prev, fp: pf, to: d, tp: 0 }); chain.push(d); prev = d; pf = 0; }
      edges.push({ from: prev, fp: pf, to: b, tp: frac(geo[b], 'in', w.to[1]) * 0.8 });
      chains.push({ w, chain });
    });
    const nL = Math.max(...Object.values(nodes).map((n) => n.layer)) + 1;
    const layers = Array.from({ length: nL }, () => []);
    // a first order: depth-first from the sources, following the ports top to bottom
    const seen = new Set(), visit = (id) => { if (seen.has(id)) return; seen.add(id); layers[nodes[id].layer].push(id); edges.filter((e) => e.from === id).sort((x, y) => x.fp - y.fp).forEach((e) => visit(e.to)); };
    for (const id of order) if (!edges.some((e) => e.to === id)) visit(id);
    for (const id of Object.keys(nodes)) visit(id);
    const pos = {};
    const setPos = () => layers.forEach((ly) => ly.forEach((id, i) => { pos[id] = i; }));
    setPos();
    const byLayer = Array.from({ length: nL }, (_, l) => edges.filter((e) => nodes[e.from].layer === l));
    /** crossings between layer l and the next */
    const crossL = (l) => {
      let c = 0;
      const es = l >= 0 && l < nL ? byLayer[l] : [];
      for (let i = 0; i < es.length; i++) for (let j = i + 1; j < es.length; j++) {
        const u = pos[es[i].from] + es[i].fp - pos[es[j].from] - es[j].fp, v = pos[es[i].to] + es[i].tp - pos[es[j].to] - es[j].tp;
        if (Math.abs(u) > 1e-9 && Math.abs(v) > 1e-9 && u * v < 0) c++;
      }
      return c;
    };
    const cross = () => { let c = 0; for (let l = 0; l + 1 < nL; l++) c += crossL(l); return c; };
    let best = cross(), bestLayers = layers.map((x) => x.slice());
    const bary = (id, dir) => {
      const es = dir === 'down' ? edges.filter((e) => e.to === id).map((e) => pos[e.from] + e.fp - e.tp) : edges.filter((e) => e.from === id).map((e) => pos[e.to] + e.tp - e.fp);
      return es.length ? es.reduce((a, b) => a + b, 0) / es.length : pos[id];
    };
    for (let it = 0; it < 12 && best > 0; it++) {
      const dir = it % 2 ? 'up' : 'down';
      const seq = dir === 'down' ? [...Array(nL).keys()].slice(1) : [...Array(nL).keys()].reverse().slice(1);
      for (const l of seq) { const bc = Object.fromEntries(layers[l].map((id) => [id, bary(id, dir)])); layers[l].sort((a, b) => bc[a] - bc[b] || pos[a] - pos[b]); layers[l].forEach((id, i) => { pos[id] = i; }); }
      // swaps of neighbours where they help
      let better = true;
      while (better) {
        better = false;
        layers.forEach((ly, l) => {
          for (let i = 0; i + 1 < ly.length; i++) {
            const c0 = crossL(l - 1) + crossL(l);
            [ly[i], ly[i + 1]] = [ly[i + 1], ly[i]]; pos[ly[i]] = i; pos[ly[i + 1]] = i + 1;
            if (crossL(l - 1) + crossL(l) < c0) better = true; else { [ly[i], ly[i + 1]] = [ly[i + 1], ly[i]]; pos[ly[i]] = i; pos[ly[i + 1]] = i + 1; }
          }
        });
      }
      const c = cross();
      if (c < best) { best = c; bestLayers = layers.map((x) => x.slice()); }
    }
    layers.splice(0, nL, ...bestLayers);
    setPos();
    // small enough to try every order of every layer: take the true minimum
    if (best > 0) {
      const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
      const total = layers.reduce((a, ly) => a * fact(ly.length), 1);
      if (total <= 60000) {
        const perms = (a) => (a.length <= 1 ? [a.slice()] : a.flatMap((x, i) => perms(a.slice(0, i).concat(a.slice(i + 1))).map((r) => [x].concat(r))));
        const all = layers.map((ly) => perms(ly));
        const pick = new Array(nL).fill(0);
        // layer by layer, with what the layers so far already cross: a branch that cannot beat the best is cut
        const tryAll = (l, sofar) => {
          if (best === 0 || sofar >= best) return;
          if (l === nL) { best = sofar; bestLayers = layers.map((x) => x.slice()); return; }
          for (const p of all[l]) { layers[l] = p; p.forEach((id, i) => { pos[id] = i; }); tryAll(l + 1, sofar + (l ? crossL(l - 1) : 0)); }
        };
        void pick;
        tryAll(0, 0);
        layers.splice(0, nL, ...bestLayers.map((x) => x.slice()));
        setPos();
      }
    }
    // heights: each layer stacked in its order, then pulled towards the ports it joins
    const top = {};
    const portY = (id, f) => (nodes[id].dummy ? top[id] + nodes[id].h / 2 : top[id] + G.HEAD + (f + 0.5) * (nodes[id].h - G.HEAD - 8) + 2);
    const gapOf = (a, b) => (nodes[a].dummy || nodes[b].dummy ? G.GAPY * 0.6 : G.GAPY);
    layers.forEach((ly) => { let y = 0; ly.forEach((id, i) => { if (i) y += gapOf(ly[i - 1], id); top[id] = y; y += nodes[id].h; }); });
    const fracOf = (e, end) => (end === 'from' ? e.fp / 0.8 : e.tp / 0.8);
    const place = (l, mode) => {
      const ly = layers[l];
      const want = ly.map((id) => {
        const ys = [];
        if (mode !== 'up') for (const e of edges) if (e.to === id && nodes[e.from].layer < l) ys.push(portY(e.from, fracOf(e, 'from')) - (portY(id, fracOf(e, 'to')) - top[id]));
        if (mode !== 'down') for (const e of edges) if (e.from === id && nodes[e.to].layer > l) ys.push(portY(e.to, fracOf(e, 'to')) - (portY(id, fracOf(e, 'from')) - top[id]));
        return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : top[id];
      });
      // isotonic regression: top[i] >= top[i-1] + h[i-1] + gap, as close as can be to what each wants
      const off = []; let acc = 0;
      ly.forEach((id, i) => { if (i) acc += nodes[ly[i - 1]].h + gapOf(ly[i - 1], id); off.push(acc); });
      const blocks = [];
      want.forEach((w, i) => { blocks.push({ v: w - off[i], n: 1 }); while (blocks.length > 1 && blocks[blocks.length - 2].v > blocks[blocks.length - 1].v) { const b = blocks.pop(), a = blocks.pop(); blocks.push({ v: (a.v * a.n + b.v * b.n) / (a.n + b.n), n: a.n + b.n }); } });
      let i = 0;
      for (const b of blocks) for (let k = 0; k < b.n; k++, i++) top[ly[i]] = b.v + off[i];
    };
    for (let it = 0; it < 10; it++) {
      if (it % 2 === 0) for (let l = 1; l < nL; l++) place(l, it < 6 ? 'down' : 'both');
      else for (let l = nL - 2; l >= 0; l--) place(l, it < 6 ? 'up' : 'both');
    }
    const minY = Math.min(...Object.values(top));
    const snap = (v) => Math.round(v / G.GRID) * G.GRID;
    for (const b of doc.blocks) { b.x = nodes[b.id].layer * (G.W + G.GAPX); b.y = snap(top[b.id] - minY); }
    for (const { w, chain } of chains) {
      if (!chain.length) { delete w.via; continue; }
      // a wire that passes a column of blocks runs flat across it, in the lane the layout kept for it
      w.via = chain.flatMap((d) => { const x0 = nodes[d].layer * (G.W + G.GAPX), y = Math.round(top[d] - minY + G.LANE / 2); return [{ x: x0 - 4, y }, { x: x0 + G.W + 4, y }]; });
    }
    WF.tidy.last = { order: best, layers: layers.map((x) => x.slice()) };
    // a via made before the blocks were snapped may sit a little off its wire: line it up with the ends where flat
    for (const { w } of chains) if (w.via) { const a = WF.portAt(doc, w.from[0], w.from[1], 'out'), b = WF.portAt(doc, w.to[0], w.to[1], 'in'); if (a && b && Math.abs(a.y - b.y) < 2) w.via.forEach((p) => { p.y = a.y; }); }
    return doc;
  };
  /** the layers of a workflow: longest path from the sources, a source just before the first block it feeds */
  const layering = function (doc) {
    const wires = doc.wires;
    let order;
    try { order = WF.order(doc); } catch (e) { order = doc.blocks.map((b) => b.id); }
    const L = {};
    for (const id of order) { const ins = wires.filter((w) => w.to[0] === id); L[id] = ins.length ? Math.max(...ins.map((w) => (L[w.from[0]] || 0) + 1)) : 0; }
    for (let pass = 0; pass < 3; pass++) for (const id of order.slice().reverse()) {
      if (wires.some((w) => w.to[0] === id)) continue;
      const outs = wires.filter((w) => w.from[0] === id);
      if (outs.length) L[id] = Math.max(0, Math.min(...outs.map((w) => L[w.to[0]])) - 1);
    }
    return L;
  };
  const sub = (doc, ids) => WF.setCompact(Object.assign({}, doc, { blocks: doc.blocks.filter((b) => ids.has(b.id)), wires: doc.wires.filter((w) => ids.has(w.from[0]) && ids.has(w.to[0])) }), doc._compact);
  /**
   * tidy-up. A workflow longer than maxLayers columns (a chain of preparation steps) is folded into rows where a single
   * wire joins one row to the next: that wire runs down the gap between the rows and back to the left, crossing nothing.
   */
  WF.tidy = function (doc, opts) {
    const G = WF.dims(doc);
    const max = Math.max(2, (opts && opts.maxLayers) || G.MAXL);
    if (!doc.blocks.length) return doc;
    const L = layering(doc), nL = Math.max(...Object.values(L)) + 1;
    if (nL <= max) { layoutOne(doc); return doc; }
    // the best place to fold: as far right as the row allows, where at most three wires cross, all of them leaving the
    // row's last column and entering the next row's first
    const tryCut = (b) => {
      const A = new Set(doc.blocks.filter((x) => L[x.id] < b).map((x) => x.id));
      // a source whose every wire leads past the fold goes with what it feeds
      for (const id of [...A]) { const outs = doc.wires.filter((w) => w.from[0] === id); if (!doc.wires.some((w) => w.to[0] === id) && outs.length && outs.every((w) => !A.has(w.to[0]))) A.delete(id); }
      const across = doc.wires.filter((w) => A.has(w.from[0]) !== A.has(w.to[0]));
      if (!across.length || across.length > 4 || !A.size || A.size === doc.blocks.length || !across.every((w) => A.has(w.from[0]))) return null;
      return { A, across };
    };
    const snapshot = () => doc.blocks.map((x) => [x, x.x, x.y]).concat([]);
    const before = snapshot(), vias = doc.wires.map((w) => [w, w.via]);
    let cand = null;
    for (let b = max; b >= 2; b--) {
      const cut = tryCut(b);
      if (!cut) continue;
      const Bids = new Set(doc.blocks.filter((x) => !cut.A.has(x.id)).map((x) => x.id));
      layoutOne(sub(doc, cut.A));
      const oa = WF.tidy.last.order;
      WF.tidy(sub(doc, Bids), opts);
      const ob = WF.tidy.last.order;
      let bottom = 0;
      for (const x of doc.blocks) if (cut.A.has(x.id)) bottom = Math.max(bottom, x.y + WF.geom(x, doc).h);
      for (const w of doc.wires) if (cut.A.has(w.from[0]) && cut.A.has(w.to[0])) for (const p of w.via || []) bottom = Math.max(bottom, p.y + 8);
      const k = cut.across.length, gap = G.ROWGAP + (k - 1) * 12;
      const top = Math.ceil((bottom + gap) / G.GRID) * G.GRID;
      for (const x of doc.blocks) if (Bids.has(x.id)) x.y += top;
      for (const w of doc.wires) if (Bids.has(w.from[0]) && Bids.has(w.to[0]) && w.via) for (const p of w.via) p.y += top;
      // the folded wires: out to the right, down to a lane of their own, back to the left, down into their port; the
      // highest port takes the outermost run, so the wires nest and cross nothing
      const ends = cut.across.map((w) => ({ w, a: WF.portAt(doc, w.from[0], w.from[1], 'out'), b: WF.portAt(doc, w.to[0], w.to[1], 'in') }));
      const ax = Math.max(...ends.map((e) => e.a.x)), bx = Math.min(...ends.map((e) => e.b.x));
      const byA = ends.slice().sort((p, q) => p.a.y - q.a.y), byB = ends.slice().sort((p, q) => p.b.y - q.b.y);
      // wires that leave the same port run as one bundle (one run down, one lane) until each turns down to its port
      const src = (e) => e.w.from.join('.');
      const srcs = [...new Set(byA.map(src))];
      const nested = byA.every((e, i) => byB[i] === e) || srcs.length === 1 || byB.every((e, i) => !i || srcs.indexOf(src(byB[i - 1])) <= srcs.indexOf(src(e)));
      if (ends.every((e) => e.a.x === ax && e.b.x === bx) && nested) {
        const ks = srcs.length, lane0 = Math.round(bottom + (top - bottom - (ks - 1) * 12) / 2);
        byB.forEach((e, i) => {
          const si = srcs.indexOf(src(e)), o = ks - 1 - si, xa = ax + 22 + o * 12, y = lane0 + o * 12, xb = bx - 22 - i * 12;
          e.w.via = [{ x: xa, y: e.a.y, c: 1 }, { x: xa, y, c: 1 }, { x: xb, y, c: 1 }, { x: xb, y: e.b.y, c: 1 }];
        });
        WF.tidy.last = { order: oa + ob, folded: true };
        return doc;
      }
      // the ends do not line up: each wire still runs down the gap after its own column, along a lane of its own
      // between the rows, and down the gap before its port's column (no block stands in a gap); the order of the
      // lanes with the fewest crossings is kept as a candidate, and the fold with the fewest crossings wins below
      const order = generic(doc, ends, srcs, bottom, top);
      if (order !== null && (!cand || order + oa + ob < cand.c)) cand = { b, c: order + oa + ob, added: order, pos: doc.blocks.map((x) => [x, x.x, x.y]), vias: doc.wires.map((w) => [w, w.via ? JSON.parse(JSON.stringify(w.via)) : null]) };
      for (const [x, px, py] of before) { x.x = px; x.y = py; }
      for (const [w, v] of vias) { if (v) w.via = v; else delete w.via; }
    }
    // a fold that adds a crossing is worth it only in a compact canvas, where an unfolded workflow would be tiny
    if (cand && (!cand.added || doc._compact)) {
      for (const [x, px, py] of cand.pos) { x.x = px; x.y = py; }
      for (const [w, v] of cand.vias) { if (v) w.via = v; else delete w.via; }
      WF.tidy.last = { order: WF.crossings(doc), folded: true };
      return doc;
    }
    layoutOne(doc);
    return doc;
  };
  /** lay the folded wires of a fold whose ends do not line up; returns the crossings it adds (or null) */
  const generic = function (doc, ends, srcs, bottom, top) {
    if (srcs.length > 5) return null;
    const perms = (a) => (a.length <= 1 ? [a.slice()] : a.flatMap((x, i) => perms(a.slice(0, i).concat(a.slice(i + 1))).map((r) => [x].concat(r))));
    const src = (e) => e.w.from.join('.');
    const base = WF.crossings(doc);
    // the drops into one column: the highest port nearest its column
    const drop = new Map();
    const byCol = new Map();
    for (const e of ends) { if (!byCol.has(e.b.x)) byCol.set(e.b.x, []); byCol.get(e.b.x).push(e); }
    // runs that a fold inside the lower row already took are left to it
    const mine = new Set(ends.map((e) => e.w));
    const used = new Set();
    for (const w of doc.wires) if (!mine.has(w) && w.via) for (const q of w.via) if (q.c) used.add(Math.round(q.x));
    const free = (x0, step) => { let x = x0; while (used.has(Math.round(x))) x += step; used.add(Math.round(x)); return x; };
    for (const list of byCol.values()) { let x = list[0].b.x - 22; list.sort((p, q) => p.b.y - q.b.y).forEach((e) => { x = free(x, -12); drop.set(e, x); x -= 12; }); }
    // the runs down from each source port: the lowest port of a column nearest it
    const rise = new Map(), firstOf = new Map();
    for (const e of ends) if (!firstOf.has(src(e))) firstOf.set(src(e), e);
    const cols = new Map();
    for (const e of firstOf.values()) { if (!cols.has(e.a.x)) cols.set(e.a.x, []); cols.get(e.a.x).push(e); }
    for (const list of cols.values()) { let x = list[0].a.x + 22; list.sort((p, q) => q.a.y - p.a.y).forEach((e) => { x = free(x, 12); rise.set(src(e), x); x += 12; }); }
    const ks = srcs.length, lane0 = Math.round(bottom + (top - bottom - (ks - 1) * 12) / 2);
    let best = null;
    for (const p of perms(srcs)) {
      for (const e of ends) {
        const xa = rise.get(src(e)), y = lane0 + p.indexOf(src(e)) * 12, xb = drop.get(e);
        e.w.via = [{ x: xa, y: e.a.y, c: 1 }, { x: xa, y, c: 1 }, { x: xb, y, c: 1 }, { x: xb, y: e.b.y, c: 1 }];
      }
      const c = WF.crossings(doc) - base;
      if (!best || c < best.c) best = { c, vias: ends.map((e) => e.w.via) };
      if (!c) break;
    }
    ends.forEach((e, i) => { e.w.via = best.vias[i]; });
    return best.c;
  };
  /** do the blocks still sit where nobody placed them (all at 0, 0)? then they want a tidy-up */
  WF.unplaced = (doc) => doc.blocks.length > 1 && doc.blocks.every((b) => !b.x && !b.y);
})(globalThis.Nooral = globalThis.Nooral || {});
