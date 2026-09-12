// Part 16, graphs and their adjacency matrix: each node's row filled in from its links, the mirror symmetry of an
// undirected graph, two-step walks counted by A², and the self loops a GCN adds.
(function (S) {
  'use strict';
  const P = S.P16, H = () => S.Kit.ml;
  const BOX = { x: 0, y: 34, w: 340, h: 340 };

  /** adjacency cells: rows filled so far, cells lit, the diagonal of self loops */
  function adjCells(prefix, x, y, A, o) {
    const n = A.length, cw = o.cw, items = [], filled = o.filled === undefined ? n : o.filled;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = A[i][j], wait = i >= filled && !(o.rowsDone && o.rowsDone.has(i));
      const cls = ['p16a', wait ? 'wait' : v ? (i === j && o.loops ? 'loop' : 'one') : '', o.lit && o.lit(i, j) ? 'lit' : ''].join(' ');
      items.push({ key: prefix + ':c' + i + '_' + j, kind: 'cell', x: x + j * (cw + 2), y: y + i * (cw + 2), w: cw, h: cw, rx: 3, text: wait ? '' : o.show ? o.show(i, j, v) : String(v), size: 12, cls });
    }
    for (let i = 0; i < n; i++) {
      items.push(H().text(prefix + ':r' + i, x - 7, y + i * (cw + 2) + cw / 2 + 4.5, String(i), 'tick' + (o.rowLit && o.rowLit(i) ? ' hot' : ''), 12, 'end'));
      items.push(H().text(prefix + ':k' + i, x + i * (cw + 2) + cw / 2, y - 8, String(i), 'tick' + (o.colLit && o.colLit(i) ? ' hot' : ''), 12, 'middle'));
    }
    return items;
  }

  S.topic({
    id: 'graph-adjacency', family: 'p16-graphs', order: 10, title: 'Graphs and the adjacency matrix',
    blurb: 'A graph is a set of nodes and the links between them: friends, papers that cite each other, atoms in a molecule. To compute with it we write it as a square table, the adjacency matrix: row i has a 1 in column j when i is linked to j. A node’s degree is the number of 1s in its row; for links that go both ways the table is its own mirror image; and squaring it counts the two-step walks, which is what two rounds of message passing will reach.',
    inputs: [
      { id: 'graph', kind: 'choice', label: 'Graph', def: 'friends', options: P.setOptions(['friends', 'ring', 'star', 'clusters', 'communities', 'citations']) },
      { id: 'loops', kind: 'choice', label: 'Self loops', def: 'no', options: [{ value: 'no', label: 'no: A' }, { value: 'yes', label: 'yes: A + I, as a GCN uses' }] },
    ],
    presets: [
      { label: 'Friends', values: { graph: 'friends', loops: 'no' } },
      { label: 'A star (one busy hub)', values: { graph: 'star', loops: 'no' } },
      { label: 'Three clubs, with self loops', values: { graph: 'communities', loops: 'yes' } },
      { label: 'Citations (one-way links)', values: { graph: 'citations', loops: 'no' } },
    ],
    counters: [{ id: 'rows', label: 'rows filled' }, { id: 'ones', label: 'ones in the matrix' }],
    run(inp) {
      const g = P.graph(inp.graph), n = g.n, dir = !!g.directed, F = H().frames(['rows', 'ones']), q = F.q;
      const und = S.GNN.adjacency(g), A = dir ? Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (g.edges.some(([a, b]) => a === i && b === j) ? 1 : 0))) : und.A;
      const deg = A.map((r) => r.reduce((s, v) => s + v, 0)), links = dir ? g.edges.length : und.deg.reduce((s, v) => s + v, 0) / 2;
      const pos = P.layout(g, BOX, 16), cw = Math.min(30, Math.floor(300 / n)), mx = 404, my = 84;
      const head = (t, sub) => [H().text('ga:t', mx - 6, 20, t, 'strong', 15), H().text('ga:s', mx - 6, 42, sub, 'legend', 13)];
      const degCol = (upto, lit, dv) => (dv || deg).map((d, i) => (i < upto ? H().text('ga:d' + i, mx + n * (cw + 2) + 8, my + i * (cw + 2) + cw / 2 + 4.5, String(d), 'p16deg' + (i === lit ? ' hot' : ''), 12) : null)).filter(Boolean)
        .concat(upto ? [H().text('ga:dh', mx + n * (cw + 2) + 8, my - 8, 'deg', 'tick', 12)] : []);
      const ones = (M) => M.reduce((s, r) => s + r.reduce((a, b) => a + b, 0), 0);
      F.push([...P.draw(g, pos), ...adjCells('ga', mx, my, A, { cw, filled: 0 }), ...head('Adjacency matrix A', dir ? 'row i: the papers that i cites' : 'row i: the nodes linked to i')],
        `${P.qpl(q, n, 'node')}, so A is ${q.n(n)} × ${q.n(n)}. ${dir ? 'Each arrow is one citation' : 'Each line is one link'}, ${q.n(links)} in all. ${dir ? 'Entry (i, j) is 1 when paper i cites paper j, else 0.' : 'Entry (i, j) is 1 when nodes i and j are linked, else 0.'}`, { phase: 'start', n, links });
      let filled = 0;
      for (let i = 0; i < n; i++) {
        const nb = A[i].map((v, j) => (v ? j : -1)).filter((j) => j >= 0);
        filled = i + 1;
        F.set('rows', filled).set('ones', ones(A.slice(0, filled)));
        const items = [...P.draw(g, pos, { act: i, lit: new Set(nb), litEdge: (a, b) => (dir ? a === i && nb.includes(b) : (a === i && nb.includes(b)) || (b === i && nb.includes(a))) }), ...adjCells('ga', mx, my, A, { cw, filled, lit: (r, c) => r === i && A[r][c] === 1, rowLit: (r) => r === i }), ...degCol(filled, i), ...head('Row ' + i, nb.length ? (dir ? 'cites ' : 'linked to ') + H().list(nb.map(String)) : (dir ? 'cites no paper here' : 'no links'))];
        F.push(items, nb.length
          ? `${dir ? 'Paper' : 'Node'} ${q.n(i)} ${dir ? 'cites' : 'is linked to'} ${dir ? (nb.length === 1 ? 'paper ' : 'papers ') + H().list(nb.map((u) => q.n(u))) : P.qnodes(q, nb)}, so row ${q.n(i)} has a one in ${nb.length === 1 ? 'that column' : 'each of those columns'}: its ${dir ? 'out-degree' : 'degree'} is ${q.n(deg[i])}.`
          : `${dir ? 'Paper' : 'Node'} ${q.n(i)} ${dir ? 'cites no earlier paper in this set' : 'has no links'}, so row ${q.n(i)} is all zeros: ${dir ? 'out-degree' : 'degree'} ${q.n(deg[i])}.`, { phase: 'row', node: i, nb, deg: deg[i] });
      }
      const total = ones(A);
      if (dir) {
        const U = und.A, totU = ones(U);
        F.set('ones', totU);
        F.push([...P.draw(g, pos, { directed: false }), ...adjCells('ga', mx, my, U, { cw, lit: (r, c) => U[r][c] === 1 && A[r][c] === 0 }), ...degCol(n, -1, und.deg), ...head('Both ways: A + Aᵀ', 'outlined: added for the cited end')],
          `A citation runs one way, so this A is not symmetric: it has ${q.n(total)} ones, one per citation. A GCN lets information flow both ways, so it uses the links in both directions: the outlined cells are added, giving ${q.n(totU)} ones and degrees from ${q.n(Math.min(...und.deg))} to ${q.n(Math.max(...und.deg))}.`, { phase: 'sym', total, totU, dmin: Math.min(...und.deg), dmax: Math.max(...und.deg) });
      } else {
        F.push([...P.draw(g, pos), ...adjCells('ga', mx, my, A, { cw, lit: (r, c) => A[r][c] === 1 && r > c }), ...degCol(n, -1), ...head('A mirror image', 'entry (i, j) equals entry (j, i)')],
          `Every link is written twice, at (i, j) and at (j, i), so the matrix equals its own transpose. The ${q.n(links)} links give ${q.n(total)} ones, which is also the sum of the degrees.`, { phase: 'sym', total, links, sumDeg: deg.reduce((s, v) => s + v, 0) });
      }
      // two-step walks: A² from the engine's matrix product, on the links taken both ways
      const U = und.A, A2t = S.T.matmul(S.T.tensor(U), S.T.tensor(U)).values(), A2 = (i, j) => Math.round(A2t[i * n + j]);
      let bi = 0, bj = 1, best = -1;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (A2(i, j) > best) { best = A2(i, j); bi = i; bj = j; }
      const common = [...Array(n).keys()].filter((k) => U[bi][k] && U[bj][k]);
      F.push([...P.draw(g, pos, { directed: false, act: bi, lit: new Set([bj, ...common]), litEdge: (a, b) => common.some((k) => (a === k || b === k) && [a, b].some((e) => e === bi || e === bj)) }),
        ...adjCells('ga', mx, my, U.map((r, i) => r.map((_, j) => A2(i, j))), { cw, show: (i, j, v) => String(v), lit: (r, c) => (r === bi && c === bj) || (r === bj && c === bi) }), ...head('A² counts two-step walks', 'cell (' + bi + ', ' + bj + '): ' + H().pl(best, 'walk'))],
        `Multiply A by itself: entry (i, j) of A² is the number of walks of two steps from i to j. Nodes ${q.n(bi)} and ${q.n(bj)} share ${P.qpl(q, best, 'neighbour')} (${H().list(common.map((k) => q.n(k)))}), so A² has ${q.n(best)} there, the most of any pair. Two rounds of message passing reach exactly these two-step neighbours.`, { phase: 'square', i: bi, j: bj, count: best, common });
      if (inp.loops === 'yes') {
        const L = S.GNN.adjacency(g, { selfLoops: true });
        F.set('ones', ones(L.A));
        F.push([...P.draw(g, pos, { directed: false }), ...adjCells('ga', mx, my, L.A, { cw, loops: true, lit: (r, c) => r === c }), ...degCol(n, -1, L.deg), ...head('Self loops: A + I', 'each node also listens to itself')],
          `Adding the identity puts a one on the diagonal: each node is now its own neighbour, so when it gathers messages its own features are kept. Every degree grows by one: they now run from ${q.n(Math.min(...L.deg))} to ${q.n(Math.max(...L.deg))}, and the matrix holds ${q.n(ones(L.A))} ones.`, { phase: 'loops', dmin: Math.min(...L.deg), dmax: Math.max(...L.deg), total: ones(L.A) });
      }
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Adjacency', text: ['A[i][j] = 1 if i and j are linked, else 0', 'degree d_i = Σ_j A[i][j]  (the row sum)', 'undirected: A = Aᵀ, Σ_i d_i = 2 × links', '(A²)[i][j] = walks of two steps from i to j', 'with self loops: A + I, degrees d_i + 1'].join('\n') }] } };
    },
    predict: [{
      at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'row' && fr[i + 1].meta.node === fr.filter((x) => x.meta.phase === 'row').reduce((b, x) => (x.meta.deg > b.meta.deg ? x : b)).meta.node,
      once: true, pick: 'option',
      ask: (fr, i) => 'How many ones will row ' + fr[i + 1].meta.node + ' have?',
      options: (fr, i) => { const d = fr[i + 1].meta.deg; return [d - 1, d, d + 1, d + 2].filter((v) => v >= 0).map(String); },
      answer: (fr, i) => String(fr[i + 1].meta.deg),
      explain: (fr, i) => fr[i + 1].say,
    }],
    quiz: [
      { q: 'A graph has 5 links, none of them loops, and they go both ways. How many ones are in its adjacency matrix?', options: ['10', '5', '25', '20'], answer: 0, why: 'Each link is written at (i, j) and at (j, i).' },
      { q: 'What does the sum of row i of A give?', options: ['the degree of node i', 'the number of nodes', 'the number of two-step walks', 'the features of node i'], answer: 0, why: 'Row i has a 1 for every neighbour of i.' },
      { q: 'Entry (2, 7) of A² is 3. This means…', options: ['nodes 2 and 7 have three neighbours in common', 'node 2 has degree 3', 'nodes 2 and 7 are three links apart', 'node 7 links to itself three times'], answer: 0, why: 'A two-step walk from 2 to 7 passes through a shared neighbour.' },
      { q: 'Why does a GCN add self loops (A + I)?', options: ['so a node keeps its own features when it mixes in its neighbours’', 'to make A symmetric', 'to remove isolated nodes', 'to count the links twice'], answer: 0, why: 'Without the loop, a node’s new value would come only from its neighbours.' },
      { q: 'For a graph of citations (one-way links), A is…', options: ['not symmetric in general', 'always symmetric', 'all ones', 'the identity'], answer: 0, why: 'Paper i citing paper j does not mean j cites i.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
