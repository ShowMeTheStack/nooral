// Part 16, graph neural networks: the families, the small graphs the topics use (with two numbers on every node),
// their layouts (computed once), and the pieces of picture they share.
(function (S) {
  'use strict';
  const fam = (id, order, title, blurb) => S.family({ id, part: 'graphs', order, title, blurb });
  fam('p16-graphs', 10, 'Graphs and message passing', 'Nodes and links written as an adjacency matrix, then the one idea every graph network is built on: each node gathers its neighbours’ numbers and mixes them into its own.');
  fam('p16-gcn', 20, 'Graph convolutional networks', 'The GCN layer as matrices (normalised adjacency, features, weights), node classification from a handful of labels, and why stacking many layers makes every node look alike.');
  fam('p16-readout', 30, 'Whole-graph tasks', 'Pooling the nodes of a graph into one vector, so that graphs of any size can be classified.');

  const P = (S.P16 = {});
  const H = () => S.Kit.ml;
  // a hand-made friendship graph: two triangles joined by a short path; two numbers per person (hours of sport and
  // of music a week, in tens of hours)
  const FRIENDS = { n: 7, edges: [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6]], X: [[0.9, 0.1], [0.8, 0.3], [0.6, 0.2], [0.4, 0.5], [0.2, 0.7], [0.1, 0.9], [0.3, 0.8]], names: ['sport', 'music'], pos: [[20, 50], [20, 310], [110, 180], [190, 215], [265, 120], [215, 20], [345, 45]] };
  /** a graph of the family set, the smallest of its kind (a ring, a star, two clusters) */
  const family = (k) => () => {
    const set = S.GNN.graphSet({ n: 90, seed: 1 }).graphs.map((g, i) => [g, i]).filter(([g]) => g.y === k).sort((a, b) => a[0].n - b[0].n || a[1] - b[1]);
    const g = set[0][0];
    return { n: g.n, edges: g.edges, X: g.X, names: ['one', 'degree / 4'] };
  };
  const two = (X) => X.map((r) => [r[0], r[1]]);
  P.GRAPHS = {
    friends: { label: 'friends: two triangles and a path (7 people)', make: () => FRIENDS },
    ring: { label: 'a ring', make: family(0) },
    star: { label: 'a star', make: family(1) },
    clusters: { label: 'two clusters and a bridge', make: family(2) },
    communities: { label: 'three clubs (12 people)', make: () => { const g = S.Data.communities({ n: 12, k: 3, pIn: 0.6, pOut: 0.06, seed: 15 }); return { n: g.n, edges: g.edges, X: two(g.X).map((r) => r.map((v) => +v.toFixed(1))), y: g.y, names: ['f0', 'f1'] }; } },
    citations: { label: 'ten papers citing older ones', make: () => { const g = S.Data.citations({ n: 10, seed: 1 }); return { n: g.n, edges: g.edges, directed: true, X: g.X.map((r) => [r.slice(0, 10).reduce((a, b) => a + b, 0), r.slice(10, 20).reduce((a, b) => a + b, 0)]), y: g.y, names: ['learning words', 'biology words'] }; } },
  };
  const cache = new Map();
  P.once = (k, f) => { if (!cache.has(k)) cache.set(k, f()); return cache.get(k); };
  P.graph = (id) => P.once('g:' + id, () => Object.assign({ id }, P.GRAPHS[id].make()));
  P.setOptions = (ids) => ids.map((id) => ({ value: id, label: P.GRAPHS[id].label }));
  /** the layout of a graph in a box (computed once per graph and box) */
  P.layout = (g, box, r) => P.once('l:' + g.id + ':' + JSON.stringify(box) + ':' + r, () => g.pos ? S.Kit.graph.fit(g.pos.map((p) => ({ x: p[0], y: p[1] })), box, r + 4) : S.Kit.graph.layout(g, Object.assign({ r, seed: 3, groups: Array.isArray(g.y) && g.id !== 'citations' ? g.y : null }, box)));
  /** neighbour lists, both ways */
  P.nbrs = (g) => { const nb = Array.from({ length: g.n }, () => new Set()); for (const [a, b] of g.edges) if (a !== b) { nb[a].add(b); nb[b].add(a); } return nb.map((s) => [...s].sort((a, b) => a - b)); };
  /** the nodes within k hops of v (v included) */
  P.within = (g, v, k) => { const nb = P.nbrs(g), seen = new Set([v]); let front = [v]; for (let s = 0; s < k; s++) { const nx = []; for (const u of front) for (const w of nb[u]) if (!seen.has(w)) { seen.add(w); nx.push(w); } front = nx; } return [...seen].sort((a, b) => a - b); };
  /** a vector as text: [0.25, −0.1] */
  P.vec = (v, d) => '[' + v.map((x) => H().fx(x, d === undefined ? 2 : d)).join(', ') + ']';
  /** a vector quoted through the frame quoter, so every number is checked */
  P.qvec = (q, v, d) => '[' + v.map((x) => q(x, d === undefined ? 2 : d)).join(', ') + ']';
  /** "node 3" / "nodes 1, 4 and 6", the numbers quoted */
  P.qnodes = (q, list) => (list.length === 1 ? 'node ' + q.n(list[0]) : 'nodes ' + H().list(list.map((i) => q.n(i))));
  /** a small table of numbers: one row per node, a header of names */
  P.table = function (prefix, x, y, rows, o) {
    o = o || {};
    const cols = [{ name: o.first || 'node', w: o.firstW || 52, cls: 'p16id' }].concat((o.names || []).map((nm, j) => ({ name: nm, w: o.colW || 66, cls: o.colCls ? o.colCls(j) || '' : '' })));
    return S.Kit.dt.table({ prefix, x, y, cols, rows: rows.map((r, i) => [o.label ? o.label(i) : String(i)].concat(r.map((v) => (typeof v === 'number' ? H().fx(v, o.d === undefined ? 2 : o.d) : v)))), rowH: o.rowH || 26, headH: o.headH || 28, size: o.size || 13, cls: o.cls });
  };
  /** a matrix drawn as heat cells with its row and column numbers */
  P.matrix = function (prefix, x, y, M, o) {
    o = o || {};
    const n = M.length, m = M[0].length, cw = o.cw || 30, ch = o.ch || cw;
    return S.Kit.heat.grid({ prefix, x, y, cw, ch, vals: M, scheme: o.scheme || 'seq', max: o.max || 1, fmt: o.fmt === undefined ? (v) => H().fx(v, o.d === undefined ? 2 : o.d) : o.fmt, rows: o.rows === false ? null : o.rows || M.map((_, i) => String(i)), cols: o.cols === false ? null : o.cols || Array.from({ length: m }, (_, j) => String(j)), size: o.size, labelSize: o.labelSize || 12, cls: o.cls, title: o.title, titleCls: o.titleCls, titleSize: o.titleSize, gap: o.gap === undefined ? 2 : o.gap, rowCls: o.rowCls, colCls: o.colCls, colsAt: o.colsAt });
  };
  P.r2 = (v) => Math.round(v * 100) / 100;
})(globalThis.Nooral = globalThis.Nooral || {});
// the graph drawn with its lit parts: act (the node in focus), lit (a set of nodes), litEdge(a, b), cls(i) extra
(function (S) {
  'use strict';
  const P = S.P16;
  P.draw = function (g, pos, o) {
    o = o || {};
    const lit = o.lit || new Set(), r = o.r || 16;
    return S.Kit.graph.draw(g, pos, {
      prefix: o.prefix || 'g', r, directed: o.directed === undefined ? !!g.directed : o.directed,
      nodeCls: (i) => [i === o.act ? 'act' : lit.has(i) ? 'lit' : '', o.dimOthers && i !== o.act && !lit.has(i) ? 'faded' : '', o.cls ? o.cls(i) || '' : ''].join(' '),
      edgeCls: (a, b) => (o.litEdge && o.litEdge(a, b) ? 'lit' : o.dimOthers ? 'faded' : ''),
      edgeZ: (a, b) => (o.litEdge && o.litEdge(a, b) ? 2 : 0),
      paint: o.paint, text: o.text, size: o.size, accept: o.accept,
    });
  };
})(globalThis.Nooral = globalThis.Nooral || {});
(function (S) {
  'use strict';
  /** a quoted count with its noun: "1 node", "3 nodes" */
  S.P16.qpl = (q, k, w, many) => q.n(k) + ' ' + (k === 1 ? w : many || w + 's');
})(globalThis.Nooral = globalThis.Nooral || {});
