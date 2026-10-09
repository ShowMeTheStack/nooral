// Part 16, graph neural networks: the exercises. Every number in a text, an answer or a hint is worked out here by the
// engines (S.GNN for adjacency, Â, message passing and readouts; S.T for products) on the small graphs written out
// below. The over-smoothing table quotes the topic over-smoothing, whose numbers a scratch run printed.
(function (S) {
  'use strict';
  const C = S.Content, N = S.GNN, T = S.T;
  if (!C || !N || !T || !S.P16 || C.exercises.has('gn-x-adj-fill')) return;
  const r2 = (v) => Math.round(v * 100) / 100, r3 = (v) => Math.round(v * 1000) / 1000;
  const n2 = (v) => String(r2(v)), n3 = (v) => String(r3(v));
  const rows = (Mt, n) => { const v = Mt.values ? Mt.values() : Mt.data; return Array.from({ length: n }, (_, i) => Array.from(v.slice(i * n, i * n + n))); };
  const ahatOf = (g) => rows(N.normAdj(g, { kind: 'sym' }).M, g.n);
  const mul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
  const nb = (g, v) => N.adjacency(g).A[v].map((x, u) => (x ? u : -1)).filter((u) => u >= 0);
  const edgeText = (g) => g.edges.map(([a, b]) => a + '–' + b).join(', ');
  const strM = (M, f) => M.map((r) => r.map((v) => (f ? f(v) : String(v))));
  const nodes = (n) => Array.from({ length: n }, (_, i) => String(i));
  const mn = (v) => String(v).replace(/^-/, '−');
  const vec = (a, f) => '[' + a.map((x) => mn(f ? f(x) : x)).join(', ') + ']';
  const hTable = (H, names) => ({ table: [['node', ...(names || H[0].map((_, j) => 'h' + '₁₂₃₄'[j]))], ...H.map((r, i) => [String(i), ...r.map(mn)])] });

  // the small graphs of these exercises
  const G5 = { n: 5, edges: [[0, 1], [0, 4], [1, 2], [1, 3], [2, 3]] };
  const G6 = { n: 6, edges: [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5]] };
  const G4 = { n: 4, edges: [[0, 1], [0, 2], [1, 2], [2, 3]] };
  const P4 = { n: 4, edges: [[0, 1], [1, 2], [2, 3]] };

  // =========================================================================================================
  // Chapter 1: graphs as tables of links
  // =========================================================================================================
  const A5 = N.adjacency(G5).A;
  C.exercise({ id: 'gn-x-adj-fill', chapter: 'gn-graphs', title: 'Write the adjacency matrix', level: 1, kind: 'matrix',
    text: ['Five people and their friendships, each one two-way: ' + edgeText(G5) + '. Fill in the adjacency matrix A: a 1 where row and column are friends, else 0 (nobody is their own friend here).'],
    rows: nodes(5), cols: nodes(5), corner: 'node', answer: A5, tolerance: 0, total: 2 * G5.edges.length,
    note: 'A 1 or a 0 in every cell.',
    hints: ['Each link a–b puts a 1 at row a, column b, and another at row b, column a.', 'Row 1: node 1 is linked to 0, 2 and 3. The diagonal stays 0.', 'Rows: ' + A5.map((r) => '[' + r.join(' ') + ']').join(', ') + '. There are ' + 2 * G5.edges.length + ' ones, twice the ' + G5.edges.length + ' links.'],
    solution: strM(A5),
    wrong: [strM(A5.map((r, i) => r.map((v, j) => (j > i ? v : 0)))), strM(A5.map((r, i) => r.map((v, j) => (i === j ? 1 : v))))] });

  const deg6 = N.adjacency(G6).deg, degCells = Object.fromEntries(deg6.map((d, i) => ['node ' + i + '|degree', d]));
  C.exercise({ id: 'gn-x-degrees', chapter: 'gn-graphs', title: 'Degrees from a list of links', level: 1, kind: 'table',
    text: ['A graph of 6 nodes has the two-way links ' + edgeText(G6) + '. What is the degree of each node?'],
    rows: nodes(6).map((i) => 'node ' + i), cols: ['degree'], corner: '', cells: degCells,
    hints: ['A node’s degree is the number of links it is part of: the ones in its row of A.', 'Node 1 appears in 0–1, 1–2 and 1–3.', 'Degrees: ' + deg6.join(', ') + '. They add up to ' + deg6.reduce((a, b) => a + b, 0) + ', twice the ' + G6.edges.length + ' links.'],
    solution: Object.fromEntries(Object.entries(degCells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(degCells).map(([k, v]) => [k, String(v)])), { 'node 5|degree': '0' })] });

  C.exercise({ id: 'gn-x-links', chapter: 'gn-graphs', title: 'Links from the ones', level: 1, kind: 'number',
    text: ['The adjacency matrix of a network of two-way links, with no self loops, holds 18 ones. How many links does the network have?'],
    answer: 9, tolerance: 0,
    whyNot: { 18: 'Each two-way link is written twice, at (i, j) and at (j, i).', 36: 'That doubles instead of halving: each link puts two ones in the matrix.', '*': 'Each link gives two ones.' },
    hints: ['A link between i and j sets A[i][j] and A[j][i].', 'So the ones are twice the links.', '18 / 2 = 9 links.'],
    solution: '9', wrong: ['18', '36'] });

  const dopts = ['A[3][1] = 1 only; before a GCN uses it, A[1][3] is set to 1 as well', 'A[1][3] = 1 and A[3][1] = 1, from the start', 'A[3][3] = 1', 'A[3][1] = 1 only, and a GCN uses it as it is'];
  C.exercise({ id: 'gn-x-directed', chapter: 'gn-graphs', title: 'A one-way link', level: 1, kind: 'choice',
    text: ['In a citation graph, paper 3 cites paper 1, with rows for the citing paper. Which entries of A are 1, and what does a GCN do with them, as in the topic [Graphs and the adjacency matrix](#topic=graph-adjacency)?'],
    options: dopts, answer: 0,
    whyNot: { 1: 'A citation runs one way: A[3][1] = 1 records it, and A[1][3] stays 0 until the links are taken both ways.', 2: 'The diagonal is for self loops; a paper citing another is an off-diagonal entry.', 3: 'A GCN usually takes links both ways, A + Aᵀ, so information flows from the cited paper to the citing one and back.' },
    hints: ['Row i lists what paper i cites.', 'Paper 3 cites paper 1: row 3, column 1.', 'A[3][1] = 1, and A[1][3] = 0 in A itself; before a GCN uses it, the links are made two-way, A + Aᵀ, which sets A[1][3] too.'],
    solution: 0, wrong: [1, 3] });

  const A6 = N.adjacency(G6).A, A6sq = mul(A6, A6), sh03 = nb(G6, 0).filter((k) => A6[3][k]);
  C.exercise({ id: 'gn-x-a2-entry', chapter: 'gn-graphs', title: 'One entry of A²', level: 1, kind: 'number',
    text: ['The graph with links ' + edgeText(G6) + '. What is entry (0, 3) of A², the number of two-step walks from node 0 to node 3?'],
    answer: A6sq[0][3], tolerance: 0,
    whyNot: { 0: 'Nodes 0 and 3 are not linked, but A² counts walks through a middle node, not direct links.', 1: 'Count every middle node linked to both 0 and 3.', '*': 'Count the neighbours that 0 and 3 share.' },
    hints: ['(A²)[0][3] = Σₖ A[0][k] · A[k][3]: one for each node k linked to both.', 'Node 0’s neighbours: ' + nb(G6, 0).join(', ') + '. Node 3’s: ' + nb(G6, 3).join(', ') + '.', 'They share ' + sh03.join(' and ') + ', so (A²)[0][3] = ' + A6sq[0][3] + ': the walks 0 → ' + sh03.join(' → 3 and 0 → ') + ' → 3.'],
    solution: String(A6sq[0][3]), wrong: ['0', '1'] });

  const AP = N.adjacency(P4).A, AP2 = mul(AP, AP);
  C.exercise({ id: 'gn-x-a2-matrix', chapter: 'gn-graphs', title: 'Square the adjacency of a path', level: 3, kind: 'matrix',
    text: ['Four stations on one line, 0–1–2–3. Fill in A², every entry the number of two-step walks between the two stations, including the diagonal.'],
    rows: nodes(4), cols: nodes(4), corner: 'node', answer: AP2, tolerance: 0, total: AP2.flat().reduce((a, b) => a + b, 0),
    note: 'Whole numbers: walks of two steps.',
    hints: ['(A²)[i][j] counts the nodes k linked to both i and j.', 'A walk from a node back to itself goes out to a neighbour and returns: the diagonal holds the degrees.', 'A² = ' + AP2.map((r) => '[' + r.join(' ') + ']').join(', ') + '. Neighbours on the line get 0: no two-step walk joins them.'],
    solution: strM(AP2), wrong: [strM(AP), strM(AP2.map((r, i) => r.map((v, j) => (i === j ? 0 : v))))] });

  const hop = (() => { const d = new Array(G6.n).fill(-1), q = [0]; d[0] = 0; while (q.length) { const v = q.shift(); for (const u of nb(G6, v)) if (d[u] < 0) { d[u] = d[v] + 1; q.push(u); } } return d; })();
  C.exercise({ id: 'gn-x-hops', chapter: 'gn-graphs', title: 'How many rounds to hear a far node?', level: 2, kind: 'number',
    text: ['The graph with links ' + edgeText(G6) + '. Node 5 holds a feature nobody else has. After how many rounds of message passing can node 0’s numbers first depend on it?'],
    answer: hop[5], tolerance: 0,
    whyNot: { [hop[5] - 1]: 'One round short: a round carries information exactly one link.', 2: 'Two rounds reach only nodes two links away.', '*': 'Find the shortest path from node 5 to node 0.' },
    hints: ['Each round carries information one link further.', 'So the answer is the length of the shortest path from 5 to 0.', 'The shortest path is 5 → 4 → 3 → 1 → 0 (or through 2): ' + hop[5] + ' links, so ' + hop[5] + ' rounds.'],
    solution: String(hop[5]), wrong: [String(hop[5] - 1), '2'] });

  const dl4 = N.adjacency(G4, { selfLoops: true }).deg, dlCells = Object.fromEntries(dl4.map((d, i) => ['node ' + i + '|d̂', d]));
  C.exercise({ id: 'gn-x-loop-deg', chapter: 'gn-graphs', title: 'Degrees with self loops', level: 1, kind: 'table',
    text: ['A triangle 0–1–2 with a tail 2–3: links ' + edgeText(G4) + '. A GCN adds a self loop to every node. What is each node’s degree d̂ with its loop?'],
    rows: nodes(4).map((i) => 'node ' + i), cols: ['d̂'], corner: '', cells: dlCells,
    hints: ['d̂ is the number of links plus one for the loop.', 'Node 2 is linked to 0, 1 and 3.', 'd̂ = ' + dl4.join(', ') + ' for nodes 0 to 3.'],
    solution: Object.fromEntries(Object.entries(dlCells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.fromEntries(Object.entries(dlCells).map(([k, v]) => [k, String(v - 1)]))] });

  const AH4 = ahatOf(G4);
  C.exercise({ id: 'gn-x-ahat-entry', chapter: 'gn-graphs', title: 'One weight of Â', level: 1, kind: 'number',
    text: ['In the same graph (' + edgeText(G4) + '), with self loops, node 2 has d̂ = ' + dl4[2] + ' and node 3 has d̂ = ' + dl4[3] + '. What is Â[2][3] = 1 / √(d̂₂ · d̂₃)? Three decimal places.'],
    answer: r3(AH4[2][3]), tolerance: 0.0015,
    whyNot: { [n3(1 / (dl4[2] * dl4[3]))]: 'That divides by the product itself; take its square root first.', [n3(1 / dl4[2])]: 'That is the mean rule, 1 / d̂₂. The GCN uses both ends: 1 / √(d̂₂ d̂₃).', '*': 'Multiply the two degrees, take the square root, and divide 1 by it.' },
    hints: ['Â[i][j] = 1 / √(d̂ᵢ d̂ⱼ) for a linked pair.', dl4[2] + ' × ' + dl4[3] + ' = ' + dl4[2] * dl4[3] + '.', '1 / √' + dl4[2] * dl4[3] + ' = ' + n3(AH4[2][3]) + '.'],
    solution: n3(AH4[2][3]), wrong: [n3(1 / (dl4[2] * dl4[3])), n3(1 / dl4[2])] });

  const AN4 = N.adjacency(G4).deg, noLoop = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => (N.adjacency(G4).A[i][j] ? 1 / Math.sqrt(AN4[i] * AN4[j]) : 0)));
  const meanM = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => (N.adjacency(G4, { selfLoops: true }).A[i][j] ? 1 / dl4[i] : 0)));
  C.exercise({ id: 'gn-x-ahat', chapter: 'gn-graphs', title: 'Build Â for a small graph', level: 3, kind: 'matrix',
    text: ['The triangle with a tail, links ' + edgeText(G4) + '. Fill in Â = D̂^−½ (A + I) D̂^−½: with self loops, entry (i, j) is 1 / √(d̂ᵢ d̂ⱼ) when i = j or i and j are linked, else 0. Two decimal places.'],
    rows: nodes(4), cols: nodes(4), corner: 'node', answer: AH4.map((r) => r.map(r2)), tolerance: 0.006,
    note: 'Numbers to two decimal places; 0 where there is no link.',
    hints: ['First the degrees with loops: d̂ = ' + dl4.join(', ') + '.', 'The diagonal is 1 / d̂ᵢ; a linked pair gets 1 / √(d̂ᵢ d̂ⱼ); nodes 0 and 3 are not linked, so that entry is 0.', 'Â = ' + AH4.map((r) => '[' + r.map(n2).join(' ') + ']').join(', ') + '.'],
    solution: strM(AH4, n2), wrong: [strM(noLoop, n2), strM(meanM, n2)] });

  const rs2 = AH4[2].reduce((a, b) => a + b, 0);
  C.exercise({ id: 'gn-x-ahat-rowsum', chapter: 'gn-graphs', title: 'A row of Â does not sum to one', level: 2, kind: 'number',
    text: ['In Â for the triangle with a tail (' + edgeText(G4) + '), what is the sum of row 2? Three decimal places.'],
    answer: r3(rs2), tolerance: 0.002,
    whyNot: { 1: 'That would be the mean rule, whose rows sum to one. The GCN’s symmetric scaling does not.', '*': 'Add the four entries of row 2: three links and the self loop.' },
    hints: ['Row 2 has an entry for itself and for each of its neighbours 0, 1 and 3.', 'Entries: ' + AH4[2].map(n3).join(', ') + '.', 'Sum: ' + n3(rs2) + '. Node 2 is busier than its neighbours 0 and 1 and its tail 3 is quieter, so the row does not come to exactly one.'],
    solution: n3(rs2), wrong: ['1'] });

  const mean4 = rows(N.normAdj(G4, { kind: 'mean' }).M, 4);
  C.exercise({ id: 'gn-x-mean-adj', chapter: 'gn-graphs', title: 'The mean rule as a matrix', level: 2, kind: 'matrix',
    text: ['The mean rule (as in GraphSAGE) averages a node’s neighbours, without a self loop. For the triangle with a tail (' + edgeText(G4) + '), fill in the matrix that does it: row i has 1 / (degree of i) in each neighbour’s column. Two decimal places.'],
    rows: nodes(4), cols: nodes(4), corner: 'node', answer: mean4.map((r) => r.map(r2)), tolerance: 0.006, rowsSumTo: 1,
    hints: ['Degrees without loops: ' + AN4.join(', ') + '.', 'Row 3 has one neighbour, node 2: it gets the whole weight, 1.', mean4.map((r) => '[' + r.map(n2).join(' ') + ']').join(', ') + '. Every row sums to one.'],
    solution: strM(mean4, n2), wrong: [strM(meanM, n2), strM(AH4, n2)] });

  // =========================================================================================================
  // Chapter 2: message passing and the GCN layer
  // =========================================================================================================
  const H4 = [[1, 0], [0, 2], [2, 1], [4, 2]], mp4 = N.messagePassing(G4, H4, { kind: 'gcn' }).out;
  const aggCells = { 'node 2|m₁': r3(mp4[2][0]), 'node 2|m₂': r3(mp4[2][1]) };
  C.exercise({ id: 'gn-x-gcn-agg', chapter: 'gn-passing', title: 'One GCN aggregation by hand', level: 2, kind: 'table',
    text: ['The triangle with a tail (' + edgeText(G4) + '), each node with two numbers:', hTable(H4), 'With the GCN rule (self loops, weights 1 / √(d̂ᵤ d̂ᵥ)), what are node 2’s new numbers m = Σᵤ Â[2][u] · hᵤ? Three decimal places.'],
    rows: ['node 2'], cols: ['m₁', 'm₂'], corner: '', cells: aggCells, tolerance: 0.002,
    hints: ['Node 2 hears from itself and from 0, 1 and 3. d̂ = ' + dl4.join(', ') + '.', 'The weights are Â[2][0] = ' + n3(AH4[2][0]) + ', Â[2][1] = ' + n3(AH4[2][1]) + ', Â[2][2] = ' + n3(AH4[2][2]) + ', Â[2][3] = ' + n3(AH4[2][3]) + '.', 'm₁ = ' + [0, 1, 2, 3].map((u) => n3(AH4[2][u]) + ' × ' + H4[u][0]).join(' + ') + ' = ' + n3(mp4[2][0]) + '; m₂ = ' + [0, 1, 2, 3].map((u) => n3(AH4[2][u]) + ' × ' + H4[u][1]).join(' + ') + ' = ' + n3(mp4[2][1]) + '.'],
    solution: { 'node 2|m₁': n3(mp4[2][0]), 'node 2|m₂': n3(mp4[2][1]) },
    wrong: [{ 'node 2|m₁': n3(mp4[2][0] - AH4[2][2] * H4[2][0]), 'node 2|m₂': n3(mp4[2][1] - AH4[2][2] * H4[2][1]) }] });

  const H5 = [[1, 0], [2, 1], [0, 3], [1, 1], [3, 2]], sum5 = N.messagePassing(G5, H5, { kind: 'sum' }).out;
  const sumCells = {}; sum5.forEach((r, i) => { sumCells['node ' + i + '|m₁'] = r[0]; sumCells['node ' + i + '|m₂'] = r[1]; });
  C.exercise({ id: 'gn-x-mp-round', chapter: 'gn-passing', title: 'A whole round of message passing', level: 2, kind: 'table',
    text: ['Five nodes with links ' + edgeText(G5) + ' and these numbers:', hTable(H5), 'One round with the sum rule: every node’s new numbers are the sum of its neighbours’ old numbers (no self loop, weight 1). Fill in every node’s m.'],
    rows: nodes(5).map((i) => 'node ' + i), cols: ['m₁', 'm₂'], corner: '', cells: sumCells,
    hints: ['Use the old numbers for every node: all nodes update at once.', 'Node 0’s neighbours are ' + nb(G5, 0).join(' and ') + ', so m = ' + nb(G5, 0).map((u) => '[' + H5[u].join(', ') + ']').join(' + ') + '.', sum5.map((r, i) => 'node ' + i + ': [' + r.join(', ') + ']').join('; ') + '.'],
    solution: Object.fromEntries(Object.entries(sumCells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.fromEntries(Object.entries(sumCells).map(([k, v]) => { const [nm, c] = k.split('|'), i = +nm.slice(5); return [k, String(v + H5[i][c === 'm₁' ? 0 : 1])]; }))] });

  const reach = (g, v, k) => { let s = new Set([v]); for (let r = 0; r < k; r++) { const t = new Set(s); for (const u of s) for (const w of nb(g, u)) t.add(w); s = t; } return [...s].sort((a, b) => a - b); };
  const rc4 = reach(G6, 4, 2), rc1 = reach(G6, 4, 1);
  C.exercise({ id: 'gn-x-receptive', chapter: 'gn-passing', title: 'Who can a node hear in two rounds?', level: 2, kind: 'set',
    text: ['The graph with links ' + edgeText(G6) + '. After two rounds of GCN message passing (with self loops), node 4’s numbers depend on which nodes’ starting numbers? Give the set, node 4 included.'],
    answer: '{' + rc4.join(', ') + '}',
    hints: ['One round: node 4 and its neighbours. Two rounds: their neighbours too.', 'After one round: {' + rc1.join(', ') + '}.', 'Add the neighbours of those: {' + rc4.join(', ') + '}. Node 0 is three links away, so it is not heard yet.'],
    solution: '{' + rc4.join(', ') + '}', wrong: ['{' + rc1.join(', ') + '}', '{' + rc4.filter((u) => u !== 4).join(', ') + '}'] });

  const gcnSol = 'function gcnAggregate(n, edges, H) {\n  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));\n  for (const [a, b] of edges) { A[a][b] = 1; A[b][a] = 1; }\n  const d = A.map((r) => r.reduce((s, v) => s + v, 0));\n  return A.map((r, i) => H[0].map((_, k) => r.reduce((s, v, j) => s + (v ? H[j][k] / Math.sqrt(d[i] * d[j]) : 0), 0)));\n}';
  C.exercise({ id: 'gn-x-code-gcn', chapter: 'gn-passing', title: 'Write the GCN aggregation', level: 2, kind: 'code',
    text: ['Write `gcnAggregate(n, edges, H)`: n nodes, a list of two-way links `[a, b]` (no self links, no repeats), and H, one row of numbers per node. Return ÂH: add a self loop to every node, let d̂ be each node’s degree counting its loop, and give node i the sum over itself and its neighbours j of H[j] / √(d̂ᵢ d̂ⱼ).'],
    fn: 'gcnAggregate', params: ['n', 'edges', 'H'], start: 'function gcnAggregate(n, edges, H) {\n  \n}', tolerance: 1e-9,
    ref: (n, edges, H) => N.messagePassing({ n, edges }, H, { kind: 'gcn' }).out,
    cases: [[4, G4.edges, H4], [5, G5.edges, H5], [3, [[0, 1]], [[1], [2], [3]]], [6, G6.edges, [[1, 0, 2], [0, 1, 0], [2, 2, 1], [0, 0, 1], [1, 3, 0], [2, 1, 1]]], [1, [], [[5, -2]]]],
    hints: ['Build A + I as an n × n table, setting both A[a][b] and A[b][a] for every link.', 'd̂ᵢ is the sum of row i. Then out[i][k] = Σⱼ (A + I)[i][j] · H[j][k] / √(d̂ᵢ d̂ⱼ). A node with no links keeps H[i] (d̂ = 1).', gcnSol],
    solution: gcnSol,
    wrong: ['function gcnAggregate(n, edges, H) {\n  const A = Array.from({ length: n }, () => new Array(n).fill(0));\n  for (const [a, b] of edges) { A[a][b] = 1; A[b][a] = 1; }\n  const d = A.map((r) => Math.max(1, r.reduce((s, v) => s + v, 0)));\n  return A.map((r, i) => H[0].map((_, k) => r.reduce((s, v, j) => s + (v ? H[j][k] / Math.sqrt(d[i] * d[j]) : 0), 0)));\n}',
      'function gcnAggregate(n, edges, H) {\n  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));\n  for (const [a, b] of edges) A[a][b] = 1;\n  const d = A.map((r) => r.reduce((s, v) => s + v, 0));\n  return A.map((r, i) => H[0].map((_, k) => r.reduce((s, v, j) => s + (v ? H[j][k] / Math.sqrt(d[i] * d[j]) : 0), 0)));\n}',
      'function gcnAggregate(n, edges, H) {\n  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));\n  for (const [a, b] of edges) { A[a][b] = 1; A[b][a] = 1; }\n  const d = A.map((r) => r.reduce((s, v) => s + v, 0));\n  return A.map((r, i) => H[0].map((_, k) => r.reduce((s, v, j) => s + (v ? H[j][k] / d[i] : 0), 0)));\n}'] });

  const H6 = [[1, 2], [3, 0], [0, 1], [2, 2], [4, 1], [1, 3]], sage = N.messagePassing(G6, H6, { kind: 'sage' }).out[3];
  const sgCells = { 'node 3|own h₁': sage[0], 'node 3|own h₂': sage[1], 'node 3|mean h₁': r3(sage[2]), 'node 3|mean h₂': r3(sage[3]) };
  C.exercise({ id: 'gn-x-sage-round', chapter: 'gn-passing', title: 'The mean rule, kept apart', level: 2, kind: 'table',
    text: ['The graph with links ' + edgeText(G6) + ' and these numbers:', hTable(H6), 'A GraphSAGE-style layer sets a node’s own numbers beside the mean of its neighbours’ numbers. Write node 3’s row: its own two numbers, then the two means. Three decimal places.'],
    rows: ['node 3'], cols: ['own h₁', 'own h₂', 'mean h₁', 'mean h₂'], corner: '', cells: sgCells, tolerance: 0.002,
    hints: ['Node 3’s neighbours are ' + nb(G6, 3).join(', ') + '; the mean leaves node 3 itself out.', 'Their numbers: ' + nb(G6, 3).map((u) => '[' + H6[u].join(', ') + ']').join(', ') + '.', 'Own [' + H6[3].join(', ') + '], mean [' + n3(sage[2]) + ', ' + n3(sage[3]) + ']: the row is [' + [sage[0], sage[1], n3(sage[2]), n3(sage[3])].join(', ') + '].'],
    solution: Object.fromEntries(Object.entries(sgCells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(sgCells).map(([k, v]) => [k, String(v)])), { 'node 3|mean h₁': n3((sage[2] * 3 + H6[3][0]) / 4), 'node 3|mean h₂': n3((sage[3] * 3 + H6[3][1]) / 4) })] });

  const sopts = ['The sum rule', 'The mean rule', 'The GCN rule', 'Any of them: they all count neighbours'];
  C.exercise({ id: 'gn-x-sum-vs-mean', chapter: 'gn-passing', title: 'Which rule can count?', level: 2, kind: 'choice',
    text: ['Every node of a social graph starts with the single number 1. After one round, a model should be able to tell how many friends each person has. Which aggregation rule keeps that information?'],
    options: sopts, answer: 0,
    whyNot: { 1: 'The mean of any number of 1s is 1: a person with one friend and a person with fifty look alike.', 2: 'The GCN’s scaling nearly cancels the count: the hub of the star in the lesson got [1.691, 0.814] from inputs near 1.', 3: 'Only a rule that adds at full weight keeps the count.' },
    hints: ['Work out one round for a node with 2 friends and one with 5, every number 1.', 'Mean: 1 and 1. Sum: 2 and 5.', 'The sum rule: its total is exactly the degree. That is why graph models that must count, or tell structures apart, use sums.'],
    solution: 0, wrong: [1, 2] });

  const W = [[0.5, -1], [1, 0.5]], Z4 = mul(H4, W), M4 = mul(AH4, Z4), R4 = M4.map((r) => r.map((v) => Math.max(0, v)));
  const lyCells = { 'Z = HW|col 1': r3(Z4[2][0]), 'Z = HW|col 2': r3(Z4[2][1]), 'ÂZ|col 1': r3(M4[2][0]), 'ÂZ|col 2': r3(M4[2][1]), 'ReLU|col 1': r3(R4[2][0]), 'ReLU|col 2': r3(R4[2][1]) };
  C.exercise({ id: 'gn-x-layer', chapter: 'gn-passing', title: 'A whole GCN layer for one node', level: 3, kind: 'table',
    text: ['The triangle with a tail (' + edgeText(G4) + ') with the numbers of the aggregation exercise:', hTable(H4), 'The layer’s weights are W = [' + vec(W[0]) + ', ' + vec(W[1]) + '] (row k of W multiplies feature k) and there is no bias. For node 2, write its row of Z = HW, its row of ÂZ, and what leaves the ReLU. Three decimal places.'],
    rows: ['Z = HW', 'ÂZ', 'ReLU'], cols: ['col 1', 'col 2'], corner: 'node 2', cells: lyCells, tolerance: 0.002,
    hints: ['Z for any node u is h₁ · ' + vec(W[0]) + ' + h₂ · ' + vec(W[1]) + '. You need Z for node 2 and for its neighbours 0, 1 and 3.', 'Z rows: ' + Z4.map((r, i) => i + ': ' + vec(r, n3)).join('; ') + '. Row 2 of Â: ' + AH4[2].map(n3).join(', ') + '.', 'Z₂ = ' + vec(Z4[2], n3) + '; (ÂZ)₂ = ' + vec(M4[2], n3) + '; after the ReLU ' + vec(R4[2], n3) + '.'],
    solution: Object.fromEntries(Object.entries(lyCells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(lyCells).map(([k, v]) => [k, String(v)])), { 'ReLU|col 1': n3(Math.abs(M4[2][0])), 'ReLU|col 2': n3(Math.abs(M4[2][1])) }), Object.assign(Object.fromEntries(Object.entries(lyCells).map(([k, v]) => [k, String(v)])), { 'ÂZ|col 1': n3(Z4[2][0]), 'ÂZ|col 2': n3(Z4[2][1]) })] });

  C.exercise({ id: 'gn-x-params', chapter: 'gn-passing', title: 'How many weights in a GCN layer?', level: 1, kind: 'number',
    text: ['A GCN layer takes 16 numbers per node to 8, with a bias, on a graph of 2,000 nodes. How many trainable numbers does the layer have?'],
    answer: 16 * 8 + 8, tolerance: 0,
    whyNot: { 128: 'That is W alone; the bias adds one number per output.', [16 * 8 * 2000]: 'W is shared by every node: the size of the graph does not enter.', [2000 * 2000]: 'Â comes from the links and is not trained.', '*': 'Count W (16 × 8) and the bias (8).' },
    hints: ['The same W multiplies every node’s row.', 'W is 16 × 8 and the bias has 8 entries; Â is fixed by the links.', '16 × 8 + 8 = ' + (16 * 8 + 8) + ', whatever the number of nodes.'],
    solution: String(16 * 8 + 8), wrong: ['128', String(16 * 8 * 2000)] });

  const layerSol = 'function gcnLayer(n, edges, H, W, b) {\n  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));\n  for (const [p, q] of edges) { A[p][q] = 1; A[q][p] = 1; }\n  const d = A.map((r) => r.reduce((s, v) => s + v, 0));\n  const Z = H.map((h) => W[0].map((_, c) => h.reduce((s, v, k) => s + v * W[k][c], 0)));\n  return A.map((r, i) => W[0].map((_, c) => Math.max(0, b[c] + r.reduce((s, v, j) => s + (v ? Z[j][c] / Math.sqrt(d[i] * d[j]) : 0), 0))));\n}';
  const layerRef = (n, edges, H, Wt, b) => { const A = ahatOf({ n, edges }); const Z = mul(H, Wt); return mul(A, Z).map((r) => r.map((v, c) => Math.max(0, v + b[c]))); };
  C.exercise({ id: 'gn-x-code-layer', chapter: 'gn-passing', title: 'Write a GCN layer', level: 3, kind: 'code',
    text: ['Write `gcnLayer(n, edges, H, W, b)`: return ReLU(Â H W + b). H has one row of f numbers per node, W is f × g (row k multiplies feature k), b has g numbers added to every node’s row, and Â is the GCN’s scaled adjacency with self loops, as in the aggregation exercise. Links are two-way, with no repeats.'],
    fn: 'gcnLayer', params: ['n', 'edges', 'H', 'W', 'b'], start: 'function gcnLayer(n, edges, H, W, b) {\n  \n}', tolerance: 1e-6,
    ref: layerRef,
    cases: [[4, G4.edges, H4, W, [0, 0]], [4, G4.edges, H4, W, [0.5, -1]], [5, G5.edges, H5, [[1, 0, -1], [0.5, 1, 0]], [0, 0.1, 0.2]], [3, [[0, 1], [1, 2]], [[1], [-2], [3]], [[2, -1]], [0, 0]]],
    hints: ['Three steps: Z = HW, then M = ÂZ, then max(0, M + b) entry by entry.', 'Â as before: A + I, d̂ the row sums, entry 1 / √(d̂ᵢ d̂ⱼ). Add b after mixing, before the ReLU.', layerSol],
    solution: layerSol,
    wrong: ['function gcnLayer(n, edges, H, W, b) {\n  const Z = H.map((h) => W[0].map((_, c) => h.reduce((s, v, k) => s + v * W[k][c], 0)));\n  return Z.map((r) => r.map((v, c) => Math.max(0, v + b[c])));\n}',
      'function gcnLayer(n, edges, H, W, b) {\n  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));\n  for (const [p, q] of edges) { A[p][q] = 1; A[q][p] = 1; }\n  const d = A.map((r) => r.reduce((s, v) => s + v, 0));\n  const Z = H.map((h) => W[0].map((_, c) => h.reduce((s, v, k) => s + v * W[k][c], 0)));\n  return A.map((r, i) => W[0].map((_, c) => b[c] + r.reduce((s, v, j) => s + (v ? Z[j][c] / Math.sqrt(d[i] * d[j]) : 0), 0)));\n}'] });

  // =========================================================================================================
  // Chapter 3: learning node labels
  // =========================================================================================================
  C.exercise({ id: 'gn-x-semi-count', chapter: 'gn-nodes', title: 'Which nodes the loss sees', level: 1, kind: 'number',
    text: ['A graph has 48 nodes in 4 classes. For node classification, 5 nodes of each class carry labels; the rest are held out. The whole graph runs through the network every epoch. On how many nodes is the training loss computed?'],
    answer: 20, tolerance: 0,
    whyNot: { 48: 'Every node is run through the network, but only the labelled ones have a label to compare with.', 28: 'That is the number of held-out nodes.', 5: 'Five per class, and there are four classes.' },
    hints: ['A loss needs a label to compare the prediction with.', 'Labelled nodes: 5 per class, 4 classes.', '5 × 4 = 20 nodes; the other 28 shape the predictions through the links but add nothing to the loss.'],
    solution: '20', wrong: ['48', '28'] });

  C.exercise({ id: 'gn-x-test-wrong', chapter: 'gn-nodes', title: 'Reading a test accuracy', level: 1, kind: 'number',
    text: [{ topic: 'node-classify', frame: 3, caption: 'The GCN on the three clubs at epoch 20.' }, 'With 3 labelled nodes per class in 30 nodes and 3 classes, the 21 other nodes are the test. At epoch 20 the test nodes are 81% right (to the nearest per cent). How many test nodes are wrong?'],
    answer: 4, tolerance: 0,
    whyNot: { 17: 'That is the number right.', 19: 'That is 19% of 100, not of 21 nodes.', 6: 'Use the 21 test nodes, not all 30.' },
    hints: ['Right: 81% of 21 is about 17.', '17 / 21 = 0.81, which rounds to 81%.', '21 − 17 = 4 wrong, as the topic shows with dark outlines.'],
    solution: '4', wrong: ['17', '6'] });

  const mopts = ['The GCN: friends mostly share a club, so averaging over them cancels the noise in each one’s own answers', 'The MLP: the links only add noise', 'The same: the features are what matter', 'The GCN, but only if every node is labelled'];
  C.exercise({ id: 'gn-x-mlp-vs-gcn', chapter: 'gn-nodes', title: 'Do the links help here?', level: 2, kind: 'choice',
    text: ['A school wants to guess which sports club each pupil belongs to, from a short survey whose answers are noisy, and from who sits with whom at lunch. Pupils mostly sit with others from their own club. 10 pupils per club have known clubs. Which model should do better on the rest: an MLP on the survey alone, or a GCN on the survey and the lunch graph?'],
    options: mopts, answer: 0,
    whyNot: { 1: 'When neighbours share a class, the links carry signal, not noise: on the three clubs the GCN reached 81% against the MLP’s 62%.', 2: 'With noisy features, the neighbours’ features add evidence the node’s own lack.', 3: 'A few labels are enough: the links carry what is learned to the unlabelled pupils.' },
    hints: ['When do a node’s neighbours tell you about its class?', 'Here neighbours mostly share a class: that is homophily.', 'The GCN. Averaging each pupil with their lunch companions cancels much of the survey’s noise; an MLP sees only the noise.'],
    solution: 0, wrong: [1, 2] });

  const cls6 = [0, 0, 0, 1, 1, 1], same = G6.edges.filter(([a, b]) => cls6[a] === cls6[b]).length, homo = same / G6.edges.length;
  C.exercise({ id: 'gn-x-homophily', chapter: 'gn-nodes', title: 'How alike are neighbours?', level: 2, kind: 'number',
    text: ['The graph with links ' + edgeText(G6) + '. Nodes 0, 1 and 2 are in class A; nodes 3, 4 and 5 in class B. What share of the links join two nodes of the same class (the edge homophily)? Three decimal places.'],
    answer: r3(homo), tolerance: 0.0015,
    whyNot: { [n3((G6.edges.length - same) / G6.edges.length)]: 'That is the share of links that cross between the classes.', '*': 'Count the links whose two ends are in the same class, and divide by all the links.' },
    hints: ['Go through the ' + G6.edges.length + ' links one by one.', 'Crossing links: ' + G6.edges.filter(([a, b]) => cls6[a] !== cls6[b]).map(([a, b]) => a + '–' + b).join(' and ') + '.', same + ' of ' + G6.edges.length + ' = ' + n3(homo) + ': most neighbours share a class, so averaging over them helps.'],
    solution: n3(homo), wrong: [n3((G6.edges.length - same) / G6.edges.length)] });

  const SM_K = [0, 1, 2, 3, 4, 6, 8, 12, 16];
  const sm = N.overSmoothing({ data: S.P16.nodeSet('communities'), depths: SM_K, train: false });
  const smPts = sm.map((r) => [r.depth, r3(r.similarity)]);
  C.exercise({ id: 'gn-x-smooth-curve', chapter: 'gn-nodes', title: 'Sketch the similarity as layers stack up', level: 2, kind: 'predict-curve',
    text: ['On the three clubs of [Over-smoothing](#topic=over-smoothing), the nodes’ features are propagated by Â again and again. Before any propagation the mean cosine similarity between the nodes is ' + n2(smPts[0][1]) + '. Sketch it from 0 to 16 propagations.'],
    plot: { xr: [0, 16], yr: [0, 1], xname: 'propagations by Â', yname: 'mean cosine similarity', given: [smPts[0]], note: 'mean similarity between every pair of nodes' },
    answer: smPts, tolerance: 0.1, checks: ['rising'],
    hints: ['Each propagation averages each node with its neighbours: nodes grow more alike.', 'One propagation takes it to ' + n2(smPts[1][1]) + '; it keeps climbing but more slowly.', 'The engine’s values: ' + smPts.map(([k, v]) => k + ': ' + n2(v)).join(', ') + '. It levels off near 1: every node points the same way.'],
    solution: smPts, wrong: [SM_K.map((k) => [k, smPts[0][1]]), SM_K.map((k) => [k, Math.max(0, smPts[0][1] + 0.9 - k * 0.06)])] });

  const dopts2 = ['1 or 2 layers', '4 layers', '8 layers', '16 layers: each node hears the whole graph'];
  C.exercise({ id: 'gn-x-depth-choice', chapter: 'gn-nodes', title: 'How many layers?', level: 2, kind: 'choice',
    text: ['On the citations graph, GCNs of several depths were trained the same way. Their test accuracies:', { table: [['layers', '1', '2', '4', '8', '16'], ['test accuracy', '96%', '93%', '81%', '81%', '44%'], ['mean similarity of nodes', '0.46', '0.63', '0.81', '0.95', '1.00']] }, 'Which depth should be used?'],
    options: dopts2, answer: 0,
    whyNot: { 1: 'At 4 layers the nodes are already 0.81 alike and accuracy has fallen to 81%.', 2: 'At 8 layers the nodes are 0.95 alike: the classes are melting together.', 3: 'At 16 layers every node points the same way (similarity 1.00) and accuracy is 44%: hearing the whole graph means hearing an average of everything.' },
    hints: ['Which row tells you how good each depth is?', 'Accuracy falls as the similarity climbs towards 1.', 'One or two layers: 96% and 93%. More layers over-smooth.'],
    solution: 0, wrong: [2, 3] });

  // =========================================================================================================
  // Chapter 4: whole graphs
  // =========================================================================================================
  const HR = [[2, -1], [0, 3], [4, 1]], ro = (k) => Array.from(N.readout(T.tensor(HR), k).values());
  const roMean = ro('mean'), roSum = ro('sum'), roMax = ro('max');
  const roCells = { 'mean|col 1': r3(roMean[0]), 'mean|col 2': r3(roMean[1]), 'sum|col 1': roSum[0], 'sum|col 2': roSum[1], 'max|col 1': roMax[0], 'max|col 2': roMax[1] };
  C.exercise({ id: 'gn-x-readout-compute', chapter: 'gn-whole', title: 'Three readouts of one graph', level: 1, kind: 'table',
    text: ['After its GCN layers, a graph of three nodes has these rows:', hTable(HR, ['col 1', 'col 2']), 'Pool them into one row each way: the mean, the sum and the maximum, column by column. Three decimal places where needed.'],
    rows: ['mean', 'sum', 'max'], cols: ['col 1', 'col 2'], corner: 'readout', cells: roCells, tolerance: 0.002,
    hints: ['Each readout works down one column at a time.', 'Column 1 holds 2, 0 and 4; column 2 holds −1, 3 and 1.', 'Mean ' + vec(roMean, n3) + '; sum ' + vec(roSum) + '; max ' + vec(roMax) + '.'],
    solution: Object.fromEntries(Object.entries(roCells).map(([k, v]) => [k, String(v)])),
    wrong: [Object.assign(Object.fromEntries(Object.entries(roCells).map(([k, v]) => [k, String(v)])), { 'max|col 1': '4', 'max|col 2': '4' })] });

  const ropts = ['The sum', 'The mean', 'The maximum', 'The first node’s row'];
  C.exercise({ id: 'gn-x-readout-choice', chapter: 'gn-whole', title: 'Choose the readout: a molecule’s weight', level: 2, kind: 'choice',
    text: ['Each node of a molecule graph is an atom, and its features include the atom’s mass. A model must predict the mass of the whole molecule, which is the total over its atoms. Molecules have 5 to 60 atoms. Which readout fits?'],
    options: ropts, answer: 0,
    whyNot: { 1: 'The mean gives the average atom: a molecule twice the size with the same kinds of atoms gets the same answer.', 2: 'The maximum keeps the heaviest atom only.', 3: 'The first row depends on how the atoms were numbered.' },
    hints: ['Does the answer grow with the number of atoms?', 'A total is a sum; the mean divides it by the size and loses the size.', 'The sum: it adds every atom’s vector, so the pooled row can carry the total mass.'],
    solution: 0, wrong: [1, 3] });

  const xopts = ['The maximum', 'The mean', 'The sum', 'Any readout gives the same answer'];
  C.exercise({ id: 'gn-x-readout-max', chapter: 'gn-whole', title: 'Choose the readout: one bad group', level: 2, kind: 'choice',
    text: ['A screening model flags a compound when any one of its atom groups is of a reactive kind, however large the rest of the molecule is. After the GCN layers, one column of each node’s row is high exactly at such a group. Which readout passes that on most directly?'],
    options: xopts, answer: 0,
    whyNot: { 1: 'The mean dilutes one high node among many: a large molecule with one reactive group looks harmless.', 2: 'The sum still grows with size: many mildly high nodes can look like one very high one.', 3: 'They keep different things: here the question is whether one node stands out.' },
    hints: ['Is the question “how much” or “is there one”?', 'One special node decides, whatever the size.', 'The maximum: the column’s largest value is high when any one group is reactive, whatever else the molecule holds.'],
    solution: 0, wrong: [1, 2] });

  const roSol = 'function readout(H, kind) {\n  return H[0].map((_, j) => {\n    const col = H.map((r) => r[j]);\n    if (kind === \'max\') return Math.max(...col);\n    const s = col.reduce((a, b) => a + b, 0);\n    return kind === \'sum\' ? s : s / col.length;\n  });\n}';
  C.exercise({ id: 'gn-x-code-readout', chapter: 'gn-whole', title: 'Write the readouts', level: 1, kind: 'code',
    text: ['Write `readout(H, kind)`: H has one row of numbers per node (at least one row); kind is `\'mean\'`, `\'sum\'` or `\'max\'`. Return one row: that pooling of each column.'],
    fn: 'readout', params: ['H', 'kind'], start: 'function readout(H, kind) {\n  \n}', tolerance: 1e-9,
    ref: (H, k) => Array.from(N.readout(T.tensor(H), k).values()),
    cases: [[HR, 'mean'], [HR, 'sum'], [HR, 'max'], [[[-3, -1, -2]], 'max'], [[[1, 2], [3, 4], [5, 6], [7, -8]], 'mean'], [[[-1], [-5]], 'max']],
    hints: ['Work column by column: col = H.map((r) => r[j]).', 'Sum it, divide by the number of rows for the mean, or take Math.max(...col). Mind columns that are all negative: start a maximum from the first value, not from 0.', roSol],
    solution: roSol,
    wrong: ['function readout(H, kind) {\n  return H[0].map((_, j) => {\n    const col = H.map((r) => r[j]);\n    if (kind === \'max\') return col.reduce((a, b) => Math.max(a, b), 0);\n    const s = col.reduce((a, b) => a + b, 0);\n    return kind === \'sum\' ? s : s / col.length;\n  });\n}',
      'function readout(H, kind) {\n  return H.map((r) => { const s = r.reduce((a, b) => a + b, 0); return kind === \'max\' ? Math.max(...r) : kind === \'sum\' ? s : s / r.length; });\n}'] });

  const popts = ['The mean of the rows', 'The sum of the rows', 'The largest value in each column', 'All the rows laid end to end, in node order', 'The row of node 0'];
  C.exercise({ id: 'gn-x-perm-choice', chapter: 'gn-whole', title: 'Which poolings ignore the numbering?', level: 2, kind: 'choice',
    text: ['Renumbering a graph’s nodes reorders the rows of its node features. Which of these ways of pooling the rows into one input for a classifier give the same result however the nodes are numbered? Choose all that do.'],
    options: popts, answer: [0, 1, 2],
    whyNot: { 3: 'Laid end to end, the rows land in different places when the order changes, and graphs of different sizes give inputs of different lengths.', 4: 'Node 0 is whichever node happened to be numbered first.' },
    hints: ['Try swapping two rows and see which results change.', 'A sum, a mean and a maximum of a list do not depend on its order.', 'The mean, the sum and the maximum. Flattening in node order and picking node 0 both change with the numbering.'],
    solution: [0, 1, 2], wrong: [[0, 1, 2, 3], [0, 1], [3, 4]] });

  // renumber the triangle with a tail: old node 3 becomes new node 0, old 0 → 1, old 1 → 2, old 2 → 3
  const perm = [1, 2, 3, 0], GP = { n: 4, edges: G4.edges.map(([a, b]) => [perm[a], perm[b]]) }, HP = []; H4.forEach((r, i) => { HP[perm[i]] = r; });
  const mpP = N.messagePassing(GP, HP, { kind: 'gcn' }).out, prCells = { 'new node 0|m₁': r3(mpP[0][0]), 'new node 0|m₂': r3(mpP[0][1]) };
  C.exercise({ id: 'gn-x-perm-rows', chapter: 'gn-whole', title: 'Renumber, and mix again', level: 3, kind: 'table',
    text: ['The triangle with a tail is drawn again with its nodes renumbered: old node 3 is now node 0, and old nodes 0, 1 and 2 are now 1, 2 and 3. The new links are ' + edgeText(GP) + ', and the features move with their nodes:', hTable(HP), 'With the GCN rule, what is new node 0’s aggregated row, m = Σᵤ Â[0][u] hᵤ in the new numbering? Three decimal places.'],
    rows: ['new node 0'], cols: ['m₁', 'm₂'], corner: '', cells: prCells, tolerance: 0.002,
    hints: ['New node 0 is the old tail: one link, to new node 3, the old node 2.', 'd̂ = 2 for it and d̂ = 4 for its neighbour: weights 1 / 2 for itself and 1 / √8 for the neighbour.', 'm = 0.5 × [' + HP[0].join(', ') + '] + ' + n3(1 / Math.sqrt(8)) + ' × [' + HP[3].join(', ') + '] = [' + mpP[0].map(n3).join(', ') + ']. It is exactly old node 3’s row, [' + mp4[3].map(n3).join(', ') + ']: renumbering only moves the rows.'],
    solution: { 'new node 0|m₁': n3(mpP[0][0]), 'new node 0|m₂': n3(mpP[0][1]) },
    wrong: [{ 'new node 0|m₁': n3(mp4[0][0]), 'new node 0|m₂': n3(mp4[0][1]) }] });
})(globalThis.Nooral = globalThis.Nooral || {});
