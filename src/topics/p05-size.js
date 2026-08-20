// Part 5, how big a tree: sweeping the depth limit or the smallest leaf allowed (the boundary, the leaf count and the
// train and test accuracy as the knob turns), and cost-complexity pruning, the weakest link cut away one at a time
// along the engine's pruning path (ML.ccpPrune).
(function (S) {
  'use strict';
  const H = S.PL, T5 = S.PL5, ML = () => S.ML;
  const n$ = H.n;
  const DEPTHS = [1, 2, 3, 4, 5, 6, 8, 12], LEAVES = [40, 25, 15, 10, 6, 3, 2, 1];
  const pct1 = (v) => (v === 1 ? '100' : (100 * v).toFixed(1)) + '%';

  S.topic({
    id: 'tree-size', family: 'p05-size', order: 10, title: 'Depth and the smallest leaf',
    blurb: 'A tree left to grow keeps splitting until every leaf is pure, and then it has learned the noise as well as the pattern. Two knobs stop it earlier: a maximum depth, and a minimum number of training points in every leaf. Turn either one and watch the training accuracy climb while the accuracy on fresh test points rises, levels off and falls.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'moons', options: T5.SETS },
      { id: 'knob', kind: 'choice', label: 'turn', def: 'depth', options: [{ value: 'depth', label: 'max depth' }, { value: 'leaf', label: 'min leaf size' }] },
      { id: 'n', kind: 'number', label: 'training points', def: 80, min: 30, max: 200 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.3, min: 0.05, max: 0.6, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 7, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Noisy moons, deeper and deeper', values: { set: 'moons', knob: 'depth', n: 80, noise: 0.3, seed: 7 } },
      { label: 'Noisy moons, smaller and smaller leaves', values: { set: 'moons', knob: 'leaf', n: 80, noise: 0.3, seed: 7 } },
      { label: 'Clean circles need depth', values: { set: 'circles', knob: 'depth', n: 120, noise: 0.08, seed: 3 } },
      { label: 'Wildflowers', values: { set: 'flowers', knob: 'depth', n: 90, noise: 0.3, seed: 2 } },
      { label: 'Exam results', values: { set: 'exam', knob: 'leaf', n: 120, noise: 0.3, seed: 4 } },
    ],
    counters: [{ id: 'leaves', label: 'leaves' }],
    run(inp) {
      const tr = T5.data2(inp.set, inp.n, inp.seed, inp.noise), te = T5.data2(inp.set, 300, inp.seed + 1000, inp.noise);
      const vals = inp.knob === 'depth' ? DEPTHS : LEAVES, kname = inp.knob === 'depth' ? 'max depth' : 'min leaf';
      const F = new S.Frames(['leaves']);
      const fits = vals.map((v) => {
        const m = ML().DecisionTreeClassifier(inp.knob === 'depth' ? { maxDepth: v, seed: 0 } : { minSamplesLeaf: v, seed: 0 }).fit(tr.X, tr.y);
        return { v, m, train: T5.acc(tr.y, m.predict(tr.X)), test: T5.acc(te.y, m.predict(te.X)), leaves: m.nLeaves, depth: m.depth };
      });
      const ax = H.axes({ prefix: 'ts', x: 44, y: 34, w: 360, h: 360, xr: tr.xr, yr: tr.yr, xname: tr.names[0], yname: tr.names[1], grid: false });
      const lo = Math.max(0, Math.floor(Math.min(...fits.map((f) => f.test), ...fits.map((f) => f.train)) * 10) / 10 - 0.1);
      const cx = H.axes({ prefix: 'tc', x: 480, y: 60, w: 270, h: 220, xr: [-0.4, vals.length - 0.6], yr: [lo, 1], xticks: vals.map((_, k) => k), fmtX: (k) => String(vals[Math.round(k)]), yticks: 0.1, fmtY: (v) => Math.round(v * 100) + '%', xname: kname, yname: 'accuracy', grid: true });
      const best = fits.reduce((a, b) => (b.test > a.test + 1e-12 ? b : a));
      // the sweep runs from the simplest tree to the most complex; scores within a point of the best are a tie in practice
      const near = fits.filter((g) => g.test >= best.test - 0.01 - 1e-12), simple = near[0];
      const legend = S.Kit.curve.legend('tc:lg', 480, 330, [{ text: 'training points', cls: 'c0' }, { text: 'test points', cls: 'c1' }], { dir: 'col' });
      fits.forEach((f, k) => {
        const T = f.m.tree, items = [...ax.items, T5.treeField(ax, 'ts:f', T, null, 80), T5.cuts(ax, T, null, tr.xr, tr.yr, { key: 'ts:cuts', cls: 'soft' }), ...S.Kit.scatter.points(ax, 'ts:p', tr.X, tr.y, { r: 3.6 })];
        items.push(...cx.items, ...legend, S.Kit.curve.line(cx, 'tc:tr', fits.slice(0, k + 1).map((g, j) => [j, g.train]), 'c0'), S.Kit.curve.line(cx, 'tc:te', fits.slice(0, k + 1).map((g, j) => [j, g.test]), 'c1'));
        items.push(T5.dot(cx, 'tc:d0', k, f.train, 'c0'), T5.dot(cx, 'tc:d1', k, f.test, 'c1'));
        items.push(H.text('tc:read', 480, 410, kname + ' ' + f.v + ':  ' + f.leaves + ' leaves, depth ' + f.depth, 'pl-lab', 13));
        F.set('leaves', f.leaves);
        let say = 'With ' + kname + ' ' + f.v + ' the tree has ' + H.plural(f.leaves, 'leaf', 'leaves') + ' and depth ' + f.depth + '. It gets ' + pct1(f.train) + ' of its ' + tr.X.length + ' training points right and ' + pct1(f.test) + ' of 300 fresh test points.';
        if (k === vals.length - 1) {
          if (simple !== best) {
            const tail = fits.slice(fits.indexOf(simple)).every((g) => near.includes(g));
            say += ' The best test accuracy, ' + pct1(best.test) + ', came at ' + kname + ' ' + best.v + ', but ' + (tail ? 'every ' + kname + ' from ' + simple.v + ' on' : kname + ' ' + H.list(near.map((g) => String(g.v))) + ' all') + ' score' + (tail ? 's' : '') + ' within one point of it: about ' + Math.round(best.test * 100) + '%. The simplest of them, ' + kname + ' ' + simple.v + ', is the one to choose; beyond it the tree fits training noise without gaining on fresh points.';
          } else say += ' The best test accuracy, ' + pct1(best.test) + ', came at ' + kname + ' ' + best.v + (f.train > best.train + 1e-12 ? '; beyond it the tree fits the training noise.' : '.');
        }
        else if (k > 0 && f.train === 1 && fits[k - 1].train < 1) say += ' Every training point is now on its own side: the leaves are pure.';
        F.add({ items, say, meta: { phase: k === vals.length - 1 ? 'last' : 'step', k, v: f.v, leaves: f.leaves, depth: f.depth, train: f.train, test: f.test, ntr: tr.X.length, nte: 300, best: best.test, bestV: best.v, nearV: near.map((g) => g.v), simpleV: simple.v } });
      });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Size limits', text: ['max depth d: at most 2ᵈ leaves', 'min leaf m: every leaf keeps ≥ m points', 'small trees: high bias (underfit)', 'big trees: high variance (overfit)', 'pick the knob on held-out data'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.k === 0, pick: 'option', ask: 'As the tree is allowed to grow, its training accuracy will…', options: ['keep rising (or stay level)', 'rise then fall', 'fall'], answer: 'keep rising (or stay level)', explain: 'More splits can only fit the training points as well or better: at the end they are all right or nearly so.' },
      { at: (f) => f.meta.k === 2, pick: 'option', ask: 'Where will the test accuracy be best?', options: (fr, i, inp) => (inp.knob === 'depth' ? ['at a small or middle depth', 'at the deepest tree'] : ['at a middle leaf size', 'with leaves of one point']), answer: (fr, i, inp) => { const L = fr[fr.length - 1].meta; const atEnd = L.best <= L.test + 1e-12; return inp.knob === 'depth' ? (atEnd ? 'at the deepest tree' : 'at a small or middle depth') : (atEnd ? 'with leaves of one point' : 'at a middle leaf size'); }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'A tree of depth 1 is called…', options: ['a stump', 'a leaf', 'a forest', 'a branch'], answer: 0, why: 'One split, two leaves.' },
      { q: 'Training accuracy 100%, test accuracy 78%. The tree probably…', options: ['overfits: it is too big', 'underfits: it is too small', 'is just right', 'has a bug'], answer: 0, why: 'A large gap between train and test is the mark of variance.' },
      { q: 'Raising the minimum leaf size makes the tree…', options: ['smaller and smoother', 'deeper', 'purer at the leaves', 'slower to predict'], answer: 0, why: 'Splits that would leave a small leaf are not allowed.' },
      { q: 'How should the depth be chosen?', options: ['by accuracy on data the tree was not trained on', 'as deep as possible', 'by training accuracy', 'always 3'], answer: 0, why: 'Only held-out data shows when extra depth stops helping.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // Cost-complexity pruning
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'ccp-prune', family: 'p05-size', order: 20, title: 'Cost-complexity pruning',
    blurb: 'Grow the tree fully, then cut it back. The cost of a tree is its total leaf impurity plus α for every leaf. For each inner node t, collapsing its subtree into one leaf adds impurity R(t) − R(Tₜ) and removes |Tₜ| − 1 leaves; the α at which that trade breaks even is g(t) = (R(t) − R(Tₜ)) / (|Tₜ| − 1). The node with the smallest g, the weakest link, goes first, and so on to the root: a path of ever smaller trees, one for each range of α.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'moons', options: T5.SETS },
      { id: 'n', kind: 'number', label: 'training points', def: 40, min: 16, max: 60 },
      { id: 'noise', kind: 'slider', label: 'noise', def: 0.3, min: 0.05, max: 0.6, step: 0.05 },
      { id: 'seed', kind: 'number', label: 'seed', def: 5, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Noisy moons', values: { set: 'moons', n: 40, noise: 0.3, seed: 5 } },
      { label: 'XOR', values: { set: 'xor', n: 40, noise: 0.1, seed: 2 } },
      { label: 'Wildflowers (three classes)', values: { set: 'flowers', n: 45, noise: 0.3, seed: 3 } },
      { label: 'Exam results', values: { set: 'exam', n: 50, noise: 0.3, seed: 6 } },
    ],
    counters: [{ id: 'leaves', label: 'leaves' }],
    run(inp) {
      const tr = T5.data2(inp.set, inp.n, inp.seed, inp.noise), te = T5.data2(inp.set, 300, inp.seed + 1000, inp.noise);
      const m = ML().DecisionTreeClassifier({ seed: 0 }).fit(tr.X, tr.y), T = m.tree, N = T.nodes, W0 = N[0].w;
      const path = ML().ccpPrune(T);
      const F = new S.Frames(['leaves']);
      const cnt = (nd) => nd.value.map((v) => Math.round(v * nd.w));
      const depth = Math.max(...N.map((x) => x.depth));
      const th = T5.treeHeight(T, 24, 12), dia = T5.diagram(T, { prefix: 'cp', x: 0, y: 12, w: 760, h: th, label: (nd, open) => (open ? ['#' + nd.id] : [cnt(nd).join('|')]), cls: (nd, open) => (open ? '' : 'leaf c' + T5.major(nd)), bar: false, boxH: 24, maxW: 70, size: 12, gap: 4 }), by = 12 + th + 40;
      const ax = H.axes({ prefix: 'cs', x: 40, y: by, w: 260, h: 190, xr: tr.xr, yr: tr.yr, xticks: 0, yticks: 0, grid: false, frame: 'box' });
      const amax = path.alphas[path.alphas.length - 1] || 1;
      const sq = (a) => Math.sqrt(a / amax);
      const pa = H.axes({ prefix: 'cq', x: 400, y: by, w: 340, h: 70, xr: [0, 1], yr: [0, Math.max(...path.impurities) * 1.08 || 1], xticks: 0, yticks: 0, yname: 'total leaf impurity R(T)', grid: false });
      const accs = [];
      const qa = H.axes({ prefix: 'cr', x: 400, y: by + 110, w: 340, h: 70, xr: [0, 1], yr: [0.4, 1], xticks: 0, yticks: [0.5, 0.75, 1], fmtY: (v) => Math.round(v * 100) + '%', yname: 'test accuracy', xname: '√(α / α_max)', grid: false });
      const collapsed = new Set();
      const under = (i) => { for (let p = N[i].parent; p >= 0; p = N[p].parent) if (collapsed.has(p)) return true; return false; };
      const open = () => new Set(N.filter((x) => x.left >= 0 && !collapsed.has(x.id) && !under(x.id)).map((x) => x.id));
      const leavesOf = (i, op) => (N[i].left < 0 || !op.has(i) ? [i] : [...leavesOf(N[i].left, op), ...leavesOf(N[i].right, op)]);
      const R = (i) => (N[i].w / W0) * N[i].impurity;
      const testAcc = (op) => T5.acc(te.y, te.X.map((x) => T5.major(T5.leafOf(T, x, op))));
      const stepPts = (k) => { const pts = []; for (let j = 0; j <= k; j++) { if (j) pts.push([sq(path.alphas[j]), path.impurities[j - 1]]); pts.push([sq(path.alphas[j]), path.impurities[j]]); } return pts; };
      for (let k = 0; k < path.alphas.length; k++) {
        const op = open(), nLeaves = leavesOf(0, op).length, acc = testAcc(op);
        accs.push([sq(path.alphas[k]), acc]);
        F.set('leaves', nLeaves);
        const next = path.prunedAt[k], items = [];
        const subtree = next === undefined ? [] : T5.subtree(T, next).filter((i) => i !== next);
        items.push(...dia.items(op, { act: next, cls: (nd) => (subtree.includes(nd.id) ? 'prune' : '') }));
        items.push(...ax.items, T5.treeField(ax, 'cs:f', T, op, 60), T5.cuts(ax, T, op, tr.xr, tr.yr, { key: 'cs:cuts', cls: 'soft' }), ...S.Kit.scatter.points(ax, 'cs:p', tr.X, tr.y, { r: 3 }));
        if (next !== undefined) items.push(T5.boxPath(ax, 'cs:box', T5.nodeBox(T, next, tr.xr, tr.yr), 'prune'));
        items.push(...pa.items, S.Kit.curve.line(pa, 'cq:l', stepPts(k), 'act'), T5.dot(pa, 'cq:d', sq(path.alphas[k]), path.impurities[k], ''));
        items.push(...qa.items, S.Kit.curve.line(qa, 'cr:l', accs.slice(), 'c1'), T5.dot(qa, 'cr:d', sq(path.alphas[k]), acc, 'c1'));
        const meta = { phase: k === 0 ? 'full' : 'step', k, alpha: path.alphas[k], Rt: path.impurities[k], leaves: nLeaves, acc, nte: 300, ntr: tr.X.length };
        let say;
        if (k === 0) say = 'The full tree: ' + nLeaves + ' leaves, every one pure, so the total leaf impurity is ' + n$(path.impurities[0]) + '. On 300 test points it scores ' + H.pct(acc) + '.';
        else say = 'α = ' + n$(path.alphas[k]) + ': the tree is down to ' + H.plural(nLeaves, 'leaf', 'leaves') + ', with total leaf impurity ' + n$(path.impurities[k]) + ' and ' + H.pct(acc) + ' on the test points.';
        if (next !== undefined) {
          const lv = leavesOf(next, op), Rb = lv.reduce((s, i) => s + R(i), 0), g = (R(next) - Rb) / (lv.length - 1);
          Object.assign(meta, { node: next, Rnode: R(next), Rbranch: Rb, nl: lv.length, nl1: lv.length - 1, g, nextAlpha: path.alphas[k + 1] });
          say += ' Weakest link: node ' + next + ', whose ' + lv.length + ' leaves have impurity ' + n$(Rb) + ' against ' + n$(R(next)) + ' as one leaf, so g = (' + n$(R(next)) + ' − ' + n$(Rb) + ') / ' + (lv.length - 1) + ' = ' + n$(g) + '.';
        } else say += ' Only the root is left: no more links to cut.';
        F.add({ items, say, meta });
        if (next !== undefined) collapsed.add(next);
      }
      const bestK = accs.reduce((b, a, k) => (a[1] > accs[b][1] + 1e-12 ? k : b), 0);
      F.list[F.list.length - 1].say += ' The best test score on the path, ' + H.pct(accs[bestK][1]) + ', is at α = ' + n$(path.alphas[bestK]) + '.';
      Object.assign(F.list[F.list.length - 1].meta, { bestAcc: accs[bestK][1], bestAlpha: path.alphas[bestK] });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Cost complexity', text: ['R_α(T) = R(T) + α·|leaves(T)|', 'R(t) = (n_t / n)·impurity(t)', 'g(t) = (R(t) − R(Tₜ)) / (|Tₜ| − 1)', 'cut the node with the smallest g;', 'that g is the next α on the path'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'full' && fr.length > 2, pick: 'item', ask: 'Click the node that will be pruned first: the weakest link.', answer: (fr) => 'cp:n' + fr[0].meta.node, explain: (fr) => fr[0].say },
      { at: (f, i, fr) => f.meta.k === 1 && fr.length > 3, pick: 'option', ask: 'Will the next α be larger than this one?', options: ['larger or equal', 'smaller'], answer: 'larger or equal', explain: 'Along the path α never decreases: each pruned tree survives for a wider range of α.' },
    ],
    quiz: [
      { q: 'A node’s subtree has 4 leaves with total R 0.02; collapsed it would have R 0.11. Its g is…', options: ['0.03', '0.09', '0.0225', '0.13'], answer: 0, why: '(0.11 − 0.02) / (4 − 1) = 0.03.' },
      { q: 'Why grow the full tree first and then prune?', options: ['a split that looks useless can enable good splits below it', 'it is faster', 'pruning needs pure leaves', 'to make α larger'], answer: 0, why: 'XOR is the classic case: no single cut helps, but two do.' },
      { q: 'At α = 0 the best tree on the path is…', options: ['the full tree', 'the root alone', 'a stump', 'half the tree'], answer: 0, why: 'Leaves cost nothing, so impurity alone counts.' },
      { q: 'How is α usually chosen?', options: ['by cross-validation along the path', 'the largest α', 'α = 1', 'by training accuracy'], answer: 0, why: 'Each α on the path gives a tree; held-out data picks one.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
