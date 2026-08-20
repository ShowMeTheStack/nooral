// Part 5, growing a tree: CART on two features, node by node as the engine's trace goes (node, candidates, split
// or leaf). For the node being split, every threshold on each feature is tried and its weighted child impurity
// (Gini or entropy) is drawn as a curve; the lowest point of the two curves is the cut taken. The scatter is shaded
// by the leaves' classes and the tree grows below it, each node keeping its place.
(function (S) {
  'use strict';
  const H = S.PL, T5 = S.PL5, ML = () => S.ML;
  const n$ = H.n;

  S.topic({
    id: 'cart-grow', family: 'p05-cart', order: 10, title: 'A tree growing split by split',
    blurb: 'At each node CART tries every threshold halfway between neighbouring values of every feature. A threshold sends the points with a smaller value left and the rest right; it is scored by the impurity of the two halves, each weighted by its share of the points. The lowest score wins, and the same search runs again inside each half until a half is pure, too small, or at the depth limit.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'moons', options: T5.SETS },
      { id: 'n', kind: 'number', label: 'points', def: 60, min: 12, max: 160 },
      { id: 'depth', kind: 'number', label: 'max depth', def: 3, min: 1, max: 3 },
      { id: 'criterion', kind: 'choice', label: 'impurity', def: 'gini', options: [{ value: 'gini', label: 'Gini' }, { value: 'entropy', label: 'entropy' }] },
      { id: 'seed', kind: 'number', label: 'seed', def: 4, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Two moons, Gini', values: { set: 'moons', n: 60, depth: 3, criterion: 'gini', seed: 4 } },
      { label: 'The same with entropy', values: { set: 'moons', n: 60, depth: 3, criterion: 'entropy', seed: 4 } },
      { label: 'XOR: no single good first cut', values: { set: 'xor', n: 48, depth: 3, criterion: 'gini', seed: 2 } },
      { label: 'Three blobs', values: { set: 'blobs', n: 60, depth: 2, criterion: 'gini', seed: 3 } },
      { label: 'Wildflowers by their petals', values: { set: 'flowers', n: 90, depth: 3, criterion: 'gini', seed: 1 } },
      { label: 'Exam: study and sleep', values: { set: 'exam', n: 80, depth: 2, criterion: 'entropy', seed: 5 } },
      { label: 'A stump: depth 1', values: { set: 'circles', n: 50, depth: 1, criterion: 'gini', seed: 6 } },
    ],
    counters: [{ id: 'splits', label: 'splits' }, { id: 'leaves', label: 'leaves' }],
    run(inp) {
      const d = T5.data2(inp.set, inp.n, inp.seed), X = d.X, y = d.y, K = d.K, crit = inp.criterion, cname = crit === 'gini' ? 'Gini' : 'entropy';
      const m = ML().DecisionTreeClassifier({ criterion: crit, maxDepth: inp.depth, seed: 0 }).fit(X, y), T = m.tree, N = T.nodes, tr = m.trace;
      const F = new S.Frames(['splits', 'leaves']);
      const cnt = (nd) => nd.value.map((v) => Math.round(v * nd.w));
      const cstr = (c) => c.join(' | ');
      const xr = d.xr, yr = d.yr;
      const ax = H.axes({ prefix: 'cg', x: 44, y: 34, w: 300, h: 240, xr, yr, xname: d.names[0], yname: d.names[1], grid: false });
      const top = (crit === 'gini' ? 1 - 1 / K : Math.log2(K)) * 1.02;
      // the two curves of a node share a y-axis zoomed to their own range, so the lowest point (the best cut) shows
      const caxFor = (lo, hi) => {
        const C = S.Kit.curve, tk = C.ticks(lo, hi, 3), y0 = Math.min(lo, tk[0]), y1 = Math.max(hi, tk[tk.length - 1]);
        return [0, 1].map((f) => H.axes({ prefix: 'cc' + f, x: 430, y: f ? 192 : 52, w: 320, h: 80, xr: f ? yr : xr, yr: [y0, y1], yticks: tk, xname: 'cuts on ' + d.names[f], yname: f ? '' : 'weighted ' + (crit === 'gini' ? 'Gini' : 'entropy'), grid: false, tickSize: 12 }));
      };
      const cax0 = [0, 1].map((f) => H.axes({ prefix: 'cc' + f, x: 430, y: f ? 192 : 52, w: 320, h: 80, xr: f ? yr : xr, yr: [0, top], yticks: top > 1.2 ? 0.5 : 0.25, xname: 'cuts on ' + d.names[f], yname: f ? '' : 'weighted ' + (crit === 'gini' ? 'Gini' : 'entropy'), grid: false, tickSize: 12 }));
      const dia = T5.diagram(T, { prefix: 'cg', x: 0, y: 342, w: 760, h: T5.treeHeight(T, 44, 18), label: (n, open) => [open ? d.short[n.feature] + ' ≤ ' + n$(n.threshold) : d.classes[T5.major(n)], cstr(cnt(n))], cls: (n, open) => (open ? '' : 'leaf c' + T5.major(n)), bar: (n) => cnt(n), maxW: 104 });
      const open = new Set();
      const field = () => T5.treeField(ax, 'cg:f', T, open, 70);
      const pts = (inNode) => S.Kit.scatter.points(ax, 'cg:p', X, y, { r: 4, cls: (i) => (inNode && !inNode.has(i) ? 'faded' : '') });
      const cutsItem = () => T5.cuts(ax, T, open, xr, yr, { key: 'cg:cuts' });
      const empty = (cax) => [...(cax || cax0).flatMap((a) => a.items), H.text('cg:key', 0, 326, 'each box: its cut (points with a smaller value go left) and its class counts', 'tick', 12)];
      const curves = (step, node) => {
        if (!step) return empty();
        const all = step.candidates.flatMap((c) => c.scores), lo = Math.min(...all), hi = Math.max(...all), pad = Math.max((hi - lo) * 0.12, 0.01);
        const cax = caxFor(Math.max(0, lo - pad), hi + pad);
        const items = empty(cax);
        for (const c of step.candidates) {
          const f = c.feature, a = cax[f], pts2 = c.thresholds.map((t, k) => [t, c.scores[k]]);
          const win = node.feature === f;
          items.push(S.Kit.curve.line(a, 'cc' + f + ':c', pts2, 'thin ' + (win ? 'act' : 'muted')));
          const k = c.scores.indexOf(Math.min(...c.scores));
          items.push(T5.dot(a, 'cc' + f + ':m', c.thresholds[k], c.scores[k], win ? '' : 'muted'));
        }
        return items;
      };
      let splits = 0, leaves = 1;
      F.set('leaves', 1);
      const rowsOf = (id) => new Set(T5.rowsAt(T, X, id));
      F.add({ items: [...ax.items, field(), cutsItem(), ...pts(), ...empty(), ...dia.items(open, { act: 0 })], say: 'The root holds all ' + X.length + ' points (' + H.list(cnt(N[0]).map((c, k) => c + ' ' + d.classes[k])) + '): ' + cname + ' ' + n$(N[0].impurity) + '. As a leaf it would predict ' + d.classes[T5.major(N[0])] + '.', meta: { phase: 'root', n: X.length, counts: cnt(N[0]), imp: N[0].impurity } });
      let pend = null;
      for (const st of tr) {
        if (st.kind === 'node') { pend = null; continue; }
        const nd = N[st.node], inNode = rowsOf(nd.id), box = T5.boxPath(ax, 'cg:box', T5.nodeBox(T, nd.id, xr, yr), 'act');
        if (st.kind === 'candidates') {
          pend = st;
          const best = {}, nt = {};
          for (const c of st.candidates) { const k = c.scores.indexOf(Math.min(...c.scores)); best[c.feature] = { t: c.thresholds[k], g: c.scores[k] }; nt[c.feature] = c.thresholds.length; }
          const f = nd.feature >= 0 ? nd.feature : +Object.keys(best).sort((a, b) => best[a].g - best[b].g)[0], o = 1 - f;
          const tried = st.candidates.reduce((s, c) => s + c.thresholds.length, 0);
          const say = 'Node ' + nd.id + ': ' + nd.n + ' points (' + cstr(cnt(nd)) + '), ' + cname + ' ' + n$(nd.impurity) + '. ' + tried + ' thresholds tried. The best on ' + d.names[f] + ' is ' + d.short[f] + ' ≤ ' + n$(best[f].t) + ', weighted ' + cname + ' ' + n$(best[f].g) + (best[o] ? '; the best on ' + d.names[o] + ' only reaches ' + n$(best[o].g) + '.' : '; ' + d.names[o] + ' is constant here.');
          F.add({ items: [...ax.items, field(), box, cutsItem(), ...pts(inNode), ...curves(st, nd), ...dia.items(open, { act: nd.id })], say, meta: { phase: 'cands', node: nd.id, nd: nd.n, counts: cnt(nd), imp: nd.impurity, tried, bestF: f, best: best[f], other: best[o] || null, depth: nd.depth } });
        } else if (st.kind === 'split') {
          open.add(nd.id); splits++;
          F.set('splits', splits);
          // a binary tree with s splits has s + 1 leaves: every node drawn without children of its own
          leaves = splits + 1;
          F.set('leaves', leaves);
          const L = N[nd.left], R = N[nd.right], child = (L.w * L.impurity + R.w * R.impurity) / nd.w;
          F.add({ items: [...ax.items, field(), box, cutsItem(), ...pts(inNode), ...curves(pend, nd), ...dia.items(open, { act: nd.id })], say: 'Split node ' + nd.id + ' at ' + d.short[nd.feature] + ' ≤ ' + n$(nd.threshold) + ': ' + cstr(cnt(L)) + ' go left (' + cname + ' ' + n$(L.impurity) + ') and ' + cstr(cnt(R)) + ' go right (' + n$(R.impurity) + '). Weighted, the impurity falls from ' + n$(nd.impurity) + ' to ' + n$(child) + '.', meta: { phase: 'split', node: nd.id, L: cnt(L), R: cnt(R), il: L.impurity, ir: R.impurity, imp: nd.impurity, child, t: nd.threshold } });
        } else if (st.kind === 'leaf') {
          const why = nd.impurity <= 1e-12 ? 'it is pure' : nd.depth >= inp.depth ? 'it is at the depth limit of ' + inp.depth : 'no cut separates its points';
          F.add({ items: [...ax.items, field(), box, cutsItem(), ...pts(inNode), ...empty(), ...dia.items(open, { act: nd.id })], say: 'Node ' + nd.id + ' (' + cstr(cnt(nd)) + ') stays a leaf because ' + why + '. It predicts ' + d.classes[T5.major(nd)] + ' with probability ' + n$(Math.max(...nd.value)) + '.', meta: { phase: 'leaf', node: nd.id, counts: cnt(nd), p: Math.max(...nd.value), depth: nd.depth } });
        }
      }
      const acc = T5.acc(y, m.predict(X));
      F.add({ items: [...ax.items, field(), cutsItem(), ...pts(), ...empty(), ...dia.items(open, {})], say: 'The finished tree: ' + splits + ' splits and ' + leaves + ' leaves, depth ' + m.depth + '. It puts ' + Math.round(acc * X.length) + ' of the ' + X.length + ' training points (' + H.pct(acc) + ') on the side of their own class. Every cut is straight and along an axis, so the boundary is a staircase.', meta: { phase: 'done', splits, leaves, depth: m.depth, right: Math.round(acc * X.length), n: X.length, acc } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Impurity', text: ['Gini  G = 1 − Σ pₖ²', 'entropy  H = − Σ pₖ log₂ pₖ', 'score of a cut = (nL·I(L) + nR·I(R)) / n', 'take the cut with the lowest score', 'stop: pure, too few, or max depth'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'root', pick: 'option', ask: 'The first cut: on which feature?', options: (fr, i, inp) => T5.data2(inp.set, inp.n, inp.seed).names.slice(), answer: (fr, i, inp) => { const c = fr.find((x) => x.meta.phase === 'cands'); return c ? T5.data2(inp.set, inp.n, inp.seed).names[c.meta.bestF] : null; }, explain: (fr) => fr.find((x) => x.meta.phase === 'cands').say },
      { at: (f, i, fr) => fr[i + 1] && fr[i + 1].meta.phase === 'split' && fr[i + 1].meta.node > 0 && f.meta.phase === 'cands', once: true, pick: 'option', ask: 'After this cut, will the weighted impurity be below half of the node’s impurity?', options: ['yes', 'no'], answer: (fr, i) => (fr[i + 1].meta.child < fr[i + 1].meta.imp / 2 ? 'yes' : 'no'), explain: (fr, i) => fr[i + 1].say },
    ],
    quiz: [
      { q: 'A node holds 8 points of one class and 2 of another. Its Gini impurity is…', options: ['0.32', '0.2', '0.5', '0.8'], answer: 0, why: '1 − 0.8² − 0.2² = 1 − 0.64 − 0.04 = 0.32.' },
      { q: 'Where does CART put the thresholds it tries on a feature?', options: ['halfway between neighbouring distinct values', 'at every training value', 'at the quartiles', 'at random'], answer: 0, why: 'Any threshold between two neighbouring values splits the points the same way; the midpoint is the convention.' },
      { q: 'Why is a split scored by the weighted average of the children’s impurities?', options: ['so a tiny pure child cannot win by itself', 'to make the numbers smaller', 'because Gini needs it', 'to break ties'], answer: 0, why: 'Cutting off one point gives a pure child, but the big impure child weighs almost everything.' },
      { q: 'The entropy of a node with two classes in equal numbers is…', options: ['1 bit', '0.5', '0', '2 bits'], answer: 0, why: '−(½ log₂ ½ + ½ log₂ ½) = 1.' },
      { q: 'What shape is the boundary of a tree on two features?', options: ['a staircase of cuts parallel to the axes', 'a smooth curve', 'one straight line at any angle', 'circles'], answer: 0, why: 'Each split tests one feature against a threshold.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // A regression tree on one input
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'reg-tree', family: 'p05-cart', order: 20, title: 'A regression tree',
    blurb: 'For a number to predict, a leaf predicts the mean of its points, and a node is scored by the squared error around that mean (the variance). Each candidate cut is scored by the weighted squared error of its two halves; the best cut splits the line into two steps. The finished tree is a staircase with one flat step per leaf.',
    inputs: [
      { id: 'shape', kind: 'choice', label: 'curve', def: 'sine', options: [{ value: 'sine', label: 'sine' }, { value: 'step', label: 'steps' }, { value: 'poly', label: 'cubic' }, { value: 'abs', label: 'V shape' }, { value: 'linear', label: 'straight line' }] },
      { id: 'n', kind: 'number', label: 'points', def: 40, min: 8, max: 120 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.2, min: 0, max: 1, step: 0.05 },
      { id: 'depth', kind: 'number', label: 'max depth', def: 3, min: 1, max: 3 },
      { id: 'minLeaf', kind: 'number', label: 'min leaf', def: 1, min: 1, max: 20 },
      { id: 'seed', kind: 'number', label: 'seed', def: 2, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'A sine wave, depth 3', values: { shape: 'sine', n: 40, noise: 0.2, depth: 3, minLeaf: 1, seed: 2 } },
      { label: 'Steps: a tree’s favourite shape', values: { shape: 'step', n: 40, noise: 0.15, depth: 2, minLeaf: 1, seed: 3 } },
      { label: 'A straight line in stairs', values: { shape: 'linear', n: 40, noise: 0.1, depth: 3, minLeaf: 1, seed: 4 } },
      { label: 'Noisy, leaves of at least 6', values: { shape: 'sine', n: 60, noise: 0.5, depth: 3, minLeaf: 6, seed: 5 } },
      { label: 'One split only', values: { shape: 'abs', n: 30, noise: 0.15, depth: 1, minLeaf: 1, seed: 6 } },
    ],
    counters: [{ id: 'leaves', label: 'leaves' }],
    run(inp) {
      const d = S.Data.curve({ n: inp.n, shape: inp.shape, noise: inp.noise, seed: inp.seed }), X = d.X, y = d.y, n = X.length;
      const m = ML().DecisionTreeRegressor({ maxDepth: inp.depth, minSamplesLeaf: inp.minLeaf, seed: 0 }).fit(X, y), T = m.tree, N = T.nodes;
      const F = new S.Frames(['leaves']);
      const yr = H.nice(y, 0.06), ax = H.axes({ prefix: 'rt', x: 46, y: 30, w: 700, h: 180, xr: [-3, 3], yr, xticks: 1, xname: 'x', yname: 'y', grid: true });
      const sax = H.axes({ prefix: 'rs', x: 46, y: 250, w: 700, h: 56, xr: [-3, 3], yr: [0, 1], xticks: 0, yticks: 0, grid: false, xname: 'weighted squared error of each cut' });
      const dia = T5.diagram(T, { prefix: 'rt', x: 0, y: 352, w: 760, h: T5.treeHeight(T, 44, 18), label: (nd, open) => [open ? 'x ≤ ' + n$(nd.threshold) : 'ŷ = ' + n$(nd.value), 'n ' + nd.n], cls: (nd, open) => (open ? '' : 'leaf'), maxW: 100, bar: false });
      const open = new Set();
      const leafAt = (x) => T5.leafOf(T, [x], open);
      const fitPath = () => {
        let dd = '', prev = null;
        for (let k = 0; k <= 600; k++) {
          const x = -3 + (6 * k) / 600, v = leafAt(x).value;
          dd += (k === 0 ? 'M' : prev !== v ? 'V' + ax.Y(v) + ' H' : 'H') + (k === 0 ? ax.X(x) + ',' + ax.Y(v) : ax.X(x)) + ' ';
          prev = v;
        }
        return { key: 'rt:fit', kind: 'path', d: dd.trim(), cls: 'pl-step act', pathAudit: 'edge-x', layer: 'lines' };
      };
      const pts = (inNode) => S.Kit.scatter.points(ax, 'rt:p', X.map((r, i) => [r[0], y[i]]), X.map(() => 0), { r: 3.8, cls: (i) => (inNode && !inNode.has(i) ? 'faded' : '') });
      const span = (id) => { const b = T5.nodeBox(T, id, [-3, 3], [0, 1])[0]; return [Math.max(-3, b[0]), Math.min(3, b[1])]; };
      const spanItem = (id) => { const [a, b] = span(id); return { key: 'rt:span', kind: 'path', d: `M${ax.X(a)},${H.r2(ax.o.y)} H${ax.X(b)} V${H.r2(ax.o.y + ax.o.h)} H${ax.X(a)} Z`, cls: 'pl-box act', audit: 'ignore', layer: 'back', z: 2 }; };
      const scoreCurve = (st) => {
        if (!st) return sax.items;
        const c = st.candidates[0], hi = Math.max(...c.scores), lo = Math.min(...c.scores), sc = (v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5);
        const k = c.scores.indexOf(lo);
        return [...sax.items, S.Kit.curve.line(sax, 'rs:c', c.thresholds.map((t, i) => [t, 0.08 + 0.84 * sc(c.scores[i])]), 'thin act'), T5.dot(sax, 'rs:m', c.thresholds[k], 0.08, '')];
      };
      let leaves = 0, pend = null;
      const sse = (id) => N[id].impurity * N[id].w;
      F.add({ items: [...ax.items, fitPath(), ...pts(), ...scoreCurve(null), ...dia.items(open, { act: 0 })], say: n + ' points. As one leaf the tree predicts their mean, ' + n$(N[0].value) + ', for every x; the squared error around it averages ' + n$(N[0].impurity) + ' per point.', meta: { phase: 'root', n, mean: N[0].value, mse: N[0].impurity } });
      for (const st of m.trace) {
        if (st.kind === 'node') continue;
        const nd = N[st.node], inNode = new Set(T5.rowsAt(T, X, nd.id));
        if (st.kind === 'candidates') {
          pend = st;
          const c = st.candidates[0], k = c.scores.indexOf(Math.min(...c.scores));
          F.add({ items: [...ax.items, spanItem(nd.id), fitPath(), ...pts(inNode), ...scoreCurve(st), ...dia.items(open, { act: nd.id })], say: 'Node ' + nd.id + ': ' + nd.n + ' points with mean ' + n$(nd.value) + ' and squared error ' + n$(nd.impurity) + ' per point. Of its ' + c.thresholds.length + ' cuts, x ≤ ' + n$(c.thresholds[k]) + ' leaves the least: ' + n$(c.scores[k]) + ' per point.', meta: { phase: 'cands', node: nd.id, nd: nd.n, mean: nd.value, mse: nd.impurity, tried: c.thresholds.length, t: c.thresholds[k], g: c.scores[k] } });
        } else if (st.kind === 'split') {
          open.add(nd.id);
          const L = N[nd.left], R = N[nd.right];
          F.add({ items: [...ax.items, spanItem(nd.id), fitPath(), ...pts(inNode), ...scoreCurve(pend), ...dia.items(open, { act: nd.id })], say: 'Split at x ≤ ' + n$(nd.threshold) + ': ' + L.n + ' points on the left with mean ' + n$(L.value) + ', ' + R.n + ' on the right with mean ' + n$(R.value) + '. The sum of squared errors falls from ' + n$(sse(nd.id)) + ' to ' + n$(sse(L.id) + sse(R.id)) + '.', meta: { phase: 'split', node: nd.id, t: nd.threshold, nl: L.n, nr: R.n, ml: L.value, mr: R.value, before: sse(nd.id), after: sse(L.id) + sse(R.id) } });
        } else if (st.kind === 'leaf') {
          leaves++;
          F.set('leaves', leaves);
          const why = nd.impurity <= 1e-12 ? 'all its points have the same y' : nd.depth >= inp.depth ? 'it is at the depth limit' : nd.n < 2 * inp.minLeaf ? 'it has fewer than ' + 2 * inp.minLeaf + ' points, so no cut leaves ' + inp.minLeaf + ' on each side' : 'no cut leaves ' + inp.minLeaf + ' points on each side';
          F.add({ items: [...ax.items, spanItem(nd.id), fitPath(), ...pts(inNode), ...scoreCurve(null), ...dia.items(open, { act: nd.id })], say: 'Node ' + nd.id + ' (' + nd.n + ' points) is a leaf because ' + why + ': it predicts ' + n$(nd.value) + ' for every x in its stretch.', meta: { phase: 'leaf', node: nd.id, nd: nd.n, mean: nd.value, minLeaf2: 2 * inp.minLeaf } });
        }
      }
      const pred = m.predict(X), mse = ML().mse(y, pred), r2v = ML().r2(y, pred);
      F.add({ items: [...ax.items, fitPath(), ...pts(), ...scoreCurve(null), ...dia.items(open, {})], say: 'The finished tree is a staircase of ' + leaves + ' flat steps. Its mean squared error on these points is ' + n$(mse) + ' (R² = ' + n$(r2v) + '), down from ' + n$(N[0].impurity) + ' for the single mean.', meta: { phase: 'done', leaves, mse, r2: r2v, mse0: N[0].impurity } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Regression tree', text: ['leaf predicts the mean ȳ of its points', 'node error: (1/n) Σ (y − ȳ)²', 'cut score: (nL·MSE(L) + nR·MSE(R)) / n', 'the curve below the plot: the score', '  of every cut of the node being split'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'root' && fr.some((x) => x.meta.phase === 'split'), pick: 'option', ask: 'Where will the first cut go?', options: ['x < −1', 'between −1 and 1', 'x > 1'], answer: (fr) => { const t = fr.find((x) => x.meta.phase === 'split').meta.t; return t < -1 ? 'x < −1' : t > 1 ? 'x > 1' : 'between −1 and 1'; }, explain: (fr) => fr.find((x) => x.meta.phase === 'split').say },
    ],
    quiz: [
      { q: 'What does a leaf of a regression tree predict?', options: ['the mean target of its training points', 'the median of all targets', 'a straight line', 'the most common class'], answer: 0, why: 'The mean minimises the squared error inside the leaf.' },
      { q: 'A tree of depth 3 can have at most how many flat steps?', options: ['8', '3', '6', '9'], answer: 0, why: 'Each level can double the number of leaves: 2³ = 8.' },
      { q: 'Why does a regression tree struggle with a straight sloping line?', options: ['it can only make flat steps, so it needs many leaves to follow a slope', 'it cannot split on x', 'the squared error is undefined', 'it predicts classes'], answer: 0, why: 'A slope is approximated by a staircase; more depth, finer stairs.' },
      { q: 'Raising the minimum leaf size does what?', options: ['smooths the fit: fewer, wider steps', 'makes more leaves', 'changes the leaf means to medians', 'nothing'], answer: 0, why: 'Cuts that would leave a tiny leaf are not allowed.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
