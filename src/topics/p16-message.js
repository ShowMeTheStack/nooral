// Part 16, message passing: one node gathering its neighbours' numbers, message by message, under three rules (the
// GCN's symmetric weights, the GraphSAGE mean, a plain sum); then every node at once, and a second round that
// reaches two links away.
(function (S) {
  'use strict';
  const P = S.P16, H = () => S.Kit.ml;
  const BOX = { x: 0, y: 34, w: 340, h: 330 };
  const RULES = {
    gcn: { label: 'GCN: weight 1 / √(d̂ᵤ d̂ᵥ), self included', short: 'GCN' },
    sage: { label: 'mean of the neighbours (GraphSAGE)', short: 'mean' },
    sum: { label: 'plain sum of the neighbours', short: 'sum' },
  };

  S.topic({
    id: 'message-passing', family: 'p16-graphs', order: 20, title: 'Message passing: a node listens to its neighbours',
    blurb: 'Every graph neural network repeats one step. Each node sends its numbers along its links; each node gathers what arrives, weighs it and adds it up, and that becomes its new numbers. Follow one node as the messages come in, see how the rule for weighing them changes the result, then let every node do it at once. A second round carries news two links away.',
    inputs: [
      { id: 'graph', kind: 'choice', label: 'Graph', def: 'friends', options: P.setOptions(['friends', 'star', 'clusters', 'communities']) },
      { id: 'node', kind: 'number', label: 'Node to follow', def: 2, min: 0, max: 15, check: (v, inp) => (v >= P.graph(inp.graph).n ? 'Node to follow: this graph has nodes 0 to ' + (P.graph(inp.graph).n - 1) : null) },
      { id: 'rule', kind: 'choice', label: 'How messages are weighed', def: 'gcn', options: Object.keys(RULES).map((k) => ({ value: k, label: RULES[k].label })) },
      { id: 'rounds', kind: 'choice', label: 'Rounds', def: '2', options: [{ value: '1', label: 'one' }, { value: '2', label: 'two' }] },
    ],
    presets: [
      { label: 'GCN rule, node 2', values: { graph: 'friends', node: 2, rule: 'gcn', rounds: '2' } },
      { label: 'Mean rule', values: { graph: 'friends', node: 2, rule: 'sage', rounds: '2' } },
      { label: 'Sum rule at a hub', values: { graph: 'star', node: 0, rule: 'sum', rounds: '1' } },
      { label: 'GCN rule at a hub', values: { graph: 'star', node: 0, rule: 'gcn', rounds: '1' } },
      { label: 'Three clubs', values: { graph: 'communities', node: 5, rule: 'gcn', rounds: '2' } },
    ],
    counters: [{ id: 'msgs', label: 'messages gathered' }],
    run(inp) {
      const g = P.graph(inp.graph), n = g.n, v = inp.node, kind = inp.rule, F = H().frames(['msgs']), q = F.q;
      const X = g.X.map((r) => r.slice(0, 2)), mp = S.GNN.messagePassing(g, X, { kind, nodes: [v] }), st = mp.trace[0].data || mp.trace[0];
      const m = mp.out.map((r) => (kind === 'sage' ? r.slice(2) : r));
      const from = st.from, w = st.weights, agg = st.aggregate;
      const nb = P.nbrs(g), dHat = S.GNN.adjacency(g, { selfLoops: kind === 'gcn' }).deg;
      const pos = P.layout(g, BOX, 16), tx = 404;
      const names = g.names.map((s) => s.length > 8 ? s.slice(0, 8) : s);
      const table = (show, lit, act) => P.table('mp:t', tx, 64, X.map((r, i) => r.concat(show(i) ? m[i] : ['', ''])), { names: ['h₁', 'h₂', 'm₁', 'm₂'], colW: 60, rowH: n > 9 ? 22 : 26, cls: (i, j) => (i === act ? 'act' : lit.has(i) && j <= 2 ? 'lit' : j > 2 && show(i) ? 'new' : '') }).items;
      const panel = (rows) => H().lines('mp:p', 0, 400, rows, 13, 22);
      const ruleLine = kind === 'gcn' ? 'weight = 1 / √(d̂ᵤ · d̂ᵥ), d̂ counting the self loop' : kind === 'sage' ? 'weight = 1 / (number of neighbours)' : 'weight = 1 for every neighbour';
      F.push([...P.draw(g, pos), ...table(() => false, new Set(), -1), ...panel([[g.names[0] + ' and ' + g.names[1] + ': the two numbers on every node', 'legend']])],
        `Every node carries two numbers, h = (h₁, h₂). In one round each node gathers the numbers of its neighbours, weighs them and adds them up into m, its new numbers. Follow node ${q.n(v)}.`, { phase: 'start', v });
      const self = kind === 'gcn';
      const others = from.filter((u) => u !== v);
      F.push([...P.draw(g, pos, { act: v, lit: new Set(others), litEdge: (a, b) => (a === v && others.includes(b)) || (b === v && others.includes(a)) }), ...table(() => false, new Set(others), v), ...panel([['Rule: ' + RULES[kind].short, 'strong'], [ruleLine, 'math']])],
        `Node ${q.n(v)} has ${P.qpl(q, nb[v].length, 'neighbour')}: ${H().list(nb[v].map((u) => q.n(u)))}.` + (self ? ` The GCN rule adds a self loop, so the node also hears itself, and it counts ${q.n(dHat[v])} with the loop. A message from u is scaled by 1 / √(d̂ᵤ · d̂ᵥ): busy senders and busy receivers are both turned down.` : kind === 'sage' ? ` The mean rule gives each of them the same weight, one over ${q.n(nb[v].length)}.` : ' The sum rule simply adds what arrives: a node with many neighbours collects a large total.'), { phase: 'focus', v, deg: nb[v].length, dHat: dHat[v], from, weights: w });
      const lines = [['Rule: ' + RULES[kind].short, 'strong']];
      const sofar = [0, 0];
      const chips = [];
      from.forEach((u, k) => {
        F.count('msgs');
        const msg = X[u].map((x) => w[k] * x);
        sofar[0] += msg[0]; sofar[1] += msg[1];
        // a chip with the weight sits on the link when there is room for it, clear of the chips already there
        const a = pos[u], b = pos[v], cx = P.r2((a.x + b.x) / 2), cy = P.r2((a.y + b.y) / 2), cw2 = 27, ch2 = 12;
        const room = u !== v && Math.hypot(b.x - a.x, b.y - a.y) - 32 > 2 * Math.hypot(cw2 * (b.x - a.x), ch2 * (b.y - a.y)) / Math.hypot(b.x - a.x, b.y - a.y) + 10;
        if (room && !chips.some((c) => Math.abs(c.x - cx) < 2 * cw2 + 6 && Math.abs(c.y - cy) < 2 * ch2 + 6)) {
          chips.push({ key: 'mp:chip' + u, kind: 'chip', x: cx, y: cy, text: '×' + H().fx(w[k], 2), size: 12, cls: 'p16chip', enter: 'fly', from: { x: a.x, y: a.y }, layer: 'top' });
        }
        lines.push(['from ' + (u === v ? 'itself' : 'node ' + u) + ': ' + H().fx(w[k], 2) + ' × ' + P.vec(X[u], 1) + ' = ' + P.vec(msg), 'math']);
        const items = [...P.draw(g, pos, { act: v, lit: new Set([u]), litEdge: (a, b) => (a === v && b === u) || (a === u && b === v) }), ...chips, ...table(() => false, new Set([u]), v), ...panel(lines)];
        const who = u === v ? 'From itself' : 'From node ' + q.n(u);
        const wsay = kind === 'gcn' ? `weight 1 / √(${q.n(dHat[u])} × ${q.n(dHat[v])}) = ${q(w[k], 3)}` : kind === 'sage' ? `the mean over ${P.qpl(q, nb[v].length, 'neighbour')} gives weight ${q(w[k], 3)}` : `weight ${q(w[k], 0)}`;
        F.push(items, `${who}: ${wsay}, so the message is ${q(w[k], 3)} × ${P.qvec(q, X[u], 1)} = ${P.qvec(q, msg, 3)}.`, { phase: 'msg', u, w: w[k], h: X[u], msg, k, dU: dHat[u], dV: dHat[v], nv: nb[v].length });
      });
      lines.push(['m' + v + ' = ' + P.vec(agg), 'strong']);
      F.push([...P.draw(g, pos, { act: v, lit: new Set(others) }), ...chips, ...table((i) => i === v, new Set(), v), ...panel(lines)],
        `Adding the ${P.qpl(q, from.length, 'message')} gives m = ${P.qvec(q, agg, 3)}: node ${q.n(v)}’s numbers were ${P.qvec(q, X[v], 1)}.` + (kind === 'sage' ? ' A GraphSAGE layer then sets this average beside the node’s own h, so it can tell itself apart from its neighbours.' : kind === 'sum' ? ' Nothing scales the total down, so it grows with the number of neighbours.' : ' Its own numbers are kept, mixed with its neighbours’.'), { phase: 'sum', v, agg, h: X[v], count: from.length });
      // every node at once
      const hub = dHat.indexOf(Math.max(...dHat)), size = m.map((r) => Math.hypot(...r)), big = size.indexOf(Math.max(...size));
      F.push([...P.draw(g, pos, { act: v }), ...table(() => true, new Set(), v), ...panel([['Every node, at the same time', 'strong'], ['each uses the old h of its neighbours', 'legend'], ['largest m: node ' + big + ', ' + P.vec(m[big]), 'math']])],
        `Every node does the same at the same moment, each from the old numbers of its neighbours, and the m column fills. The largest result is at node ${q.n(big)}, ${P.qvec(q, m[big], 2)}` + (kind === 'sum' && big === hub ? `: the busiest node, with ${P.qpl(q, nb[hub].length, 'neighbour')}, adds the most.` : '.'), { phase: 'all', big, mBig: m[big], hub, degHub: nb[hub].length });
      if (inp.rounds === '2') {
        const mp2 = S.GNN.messagePassing(g, m, { kind, nodes: [v] }), m2 = mp2.out[v].slice(kind === 'sage' ? 2 : 0), one = P.within(g, v, 1), two = P.within(g, v, 2), far = two.filter((u) => !one.includes(u));
        F.push([...P.draw(g, pos, { act: v, lit: new Set(two.filter((u) => u !== v)), dimOthers: true, cls: (i) => (far.includes(i) ? 'far' : '') }), ...table(() => true, new Set(two.filter((u) => u !== v)), v), ...panel([['Round two', 'strong'], ['node ' + v + ' now holds ' + P.vec(m2), 'math'], ['it depends on ' + H().pl(two.length, 'node') + ' (lit)', 'legend']])],
          `A second round repeats the step on the new numbers, so node ${q.n(v)} now hears from its neighbours, who already heard from theirs. Its value ${P.qvec(q, m2, 3)} depends on ${P.qpl(q, two.length, 'node')}: itself, ${P.qpl(q, one.length - 1, 'neighbour')} and ${P.qpl(q, far.length, 'node')} two links away${far.length ? ' (' + H().list(far.map((u) => q.n(u))) + ')' : ''}. With k rounds a node sees k links out.`, { phase: 'round2', v, m2, reach: two.length, near: one.length - 1, far, nFar: far.length });
      }
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'One round of message passing', text: ['m_v = Σ_{u ∈ N(v)} w_uv · h_u', 'GCN:  N(v) includes v,  w_uv = 1 / √(d̂_u d̂_v)', 'mean: w_uv = 1 / |N(v)|,  then [h_v, m_v]', 'sum:  w_uv = 1', 'k rounds: v hears from nodes up to k links away'].join('\n') }] } };
    },
    predict: [{
      at: (f) => f.meta.phase === 'focus',
      once: true, pick: 'option',
      ask: (fr, i) => 'After gathering, will node ' + fr[i].meta.v + '’s first number h₁ go up or down?',
      options: ['up', 'down', 'stay the same'],
      answer: (fr) => { const s = fr.find((f) => f.meta.phase === 'sum').meta, d = s.agg[0] - s.h[0]; return Math.abs(d) < 5e-4 ? 'stay the same' : d > 0 ? 'up' : 'down'; },
      explain: (fr) => fr.find((f) => f.meta.phase === 'sum').say,
    }],
    quiz: [
      { q: 'Under the GCN rule, node v has degree 3 and neighbour u has degree 1 (both without the self loop). The weight on u’s message is…', options: ['1 / √(2 × 4) ≈ 0.35', '1 / 3', '1', '1 / √3'], answer: 0, why: 'With self loops the degrees are 4 and 2.' },
      { q: 'Why can the plain sum rule be a problem?', options: ['busy nodes collect much larger numbers than quiet ones', 'it ignores the neighbours', 'it needs labels', 'it only works on trees'], answer: 0, why: 'The total grows with the degree, so scales differ wildly across the graph.' },
      { q: 'After three rounds of message passing, a node’s numbers depend on…', options: ['nodes up to three links away', 'only its neighbours', 'every node in the graph, always', 'only itself'], answer: 0, why: 'Each round reaches one link further.' },
      { q: 'In one round, do nodes use their neighbours’ old or new numbers?', options: ['old: every node updates at the same time', 'new, in node order', 'whichever finished first', 'random'], answer: 0, why: 'A round is one matrix product: all new values come from the old ones.' },
      { q: 'Renumbering the nodes of a graph changes each node’s message-passing result…', options: ['not at all: only who is linked to whom matters', 'completely', 'only for the first node', 'only under the mean rule'], answer: 0, why: 'Sums over neighbours do not depend on the order of the nodes.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
