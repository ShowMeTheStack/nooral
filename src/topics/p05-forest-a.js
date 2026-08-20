// Part 5, bagging and the out-of-bag score. Bagging: each tree is grown on a bootstrap sample (n rows drawn with
// replacement) and the trees vote; the small panels are the trees one by one, the large one their vote so far.
// Out of bag: the rows a tree never drew grade it, so every row collects votes only from trees that did not see it,
// and those votes give a test score for free. Both use ML.RandomForestClassifier with the draws passed in, so the
// picture shows exactly the samples the engine used.
(function (S) {
  'use strict';
  const H = S.PL, T5 = S.PL5, ML = () => S.ML;
  const n$ = H.n;
  /** the bootstrap draws and tree seeds for B trees on n rows, from one seed */
  T5.draws = (n, B, seed) => { const rng = ML().rng(seed); const draws = [], seeds = []; for (let b = 0; b < B; b++) { draws.push(ML().range(n).map(() => rng.int(n))); seeds.push(1 + rng.int(0x7ffffffe)); } return { draws, seeds }; };
  /** the averaged class probabilities of the first b trees of a forest at x */
  T5.vote = (trees, b, x) => { const K = trees[0].nodes[0].value.length, s = new Array(K).fill(0); for (let t = 0; t < b; t++) { const v = trees[t].nodes[ML().treeApply(trees[t], x)].value; for (let k = 0; k < K; k++) s[k] += v[k] / b; } return s; };

  S.topic({
    id: 'bagging', family: 'p05-forest', order: 10, title: 'Bagging: many trees on bootstrap samples',
    blurb: 'A deep tree changes a lot when its training points change a little: it has high variance. Bagging grows many deep trees, each on its own bootstrap sample (n rows drawn with replacement, so some rows come twice and about a third not at all), and lets them vote. The wobbles of the single trees average out and the voted boundary is steadier and usually more accurate than any one of them.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'moons', options: T5.SETS },
      { id: 'n', kind: 'number', label: 'training points', def: 60, min: 20, max: 150 },
      { id: 'B', kind: 'number', label: 'trees', def: 9, min: 2, max: 9 },
      { id: 'seed', kind: 'number', label: 'seed', def: 3, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Two moons, nine trees', values: { set: 'moons', n: 60, B: 9, seed: 3 } },
      { label: 'Circles', values: { set: 'circles', n: 80, B: 9, seed: 5 } },
      { label: 'Wildflowers', values: { set: 'flowers', n: 90, B: 9, seed: 2 } },
      { label: 'Exam results, few points', values: { set: 'exam', n: 40, B: 6, seed: 8 } },
      { label: 'XOR, three trees', values: { set: 'xor', n: 60, B: 3, seed: 4 } },
    ],
    counters: [{ id: 'trees', label: 'trees' }],
    run(inp) {
      const d = T5.data2(inp.set, inp.n, inp.seed), te = T5.data2(inp.set, 300, inp.seed + 1000), n = d.X.length, B = inp.B;
      const { draws, seeds } = T5.draws(n, B, inp.seed);
      const m = ML().RandomForestClassifier({ nEstimators: B, maxFeatures: null, bootstrap: true, bootstrapIndices: draws, treeSeeds: seeds }).fit(d.X, d.y);
      const F = new S.Frames(['trees']);
      const ax = H.axes({ prefix: 'bg', x: 44, y: 40, w: 330, h: 330, xr: d.xr, yr: d.yr, xname: d.names[0], yname: d.names[1], grid: false });
      const accOf = (f) => T5.acc(te.y, te.X.map((x) => ML().argmax(f(x))));
      const leafV = (T, x) => T.nodes[ML().treeApply(T, x)].value;
      const treeAcc = m.trees.map((T) => accOf((x) => leafV(T, x))), bagAcc = [];
      for (let b = 1; b <= B; b++) bagAcc.push(accOf((x) => T5.vote(m.trees, b, x)));
      const slot = (b) => ({ x: 420 + (b % 3) * 118, y: 46 + Math.floor(b / 3) * 128, w: 104, h: 96 });
      const small = (b, act) => {
        const w = m.inBag[b], idx = d.X.map((_, i) => i).filter((i) => w[i] > 0);
        const sub = { X: idx.map((i) => d.X[i]), y: idx.map((i) => d.y[i]), xr: d.xr, yr: d.yr };
        const T = m.trees[b];
        const p = T5.panel('bs' + b, slot(b), sub, (u, v) => { const val = leafV(T, [u, v]), k = ML().argmax(val); return [k, 0.8]; }, 'tree ' + (b + 1) + ' · ' + H.pct(treeAcc[b]), { r: 2.2, tcls: 'pl-lab' + (act ? ' act' : ''), pcls: (j) => (w[idx[j]] > 1 ? 'p5-twice' : '') });
        return p.items;
      };
      const big = (b) => {
        const items = [...ax.items];
        if (b > 0) items.push(T5.field(ax, 'bg:f', (u, v) => { const s = T5.vote(m.trees, b, [u, v]), k = ML().argmax(s); return [k, Math.max(0.15, (s[k] - 1 / d.K) / (1 - 1 / d.K))]; }, 80));
        items.push(...S.Kit.scatter.points(ax, 'bg:p', d.X, d.y, { r: 3.6 }));
        items.push(H.text('bg:read', 44, 410, b ? 'the vote of ' + H.plural(b, 'tree') + ': ' + H.pct(bagAcc[b - 1]) + ' on test points' : 'no trees yet', 'pl-lab', 13));
        return items;
      };
      const ghost = (from) => { const it = []; for (let b = from; b < B; b++) { const sl = slot(b); it.push({ key: 'bs' + b + ':g', kind: 'path', d: `M${sl.x},${sl.y} h${sl.w} v${sl.h} h${-sl.w} Z`, cls: 'pl-box', audit: 'ignore', layer: 'back' }, H.text('bs' + b + ':t', sl.x, sl.y - 8, 'tree ' + (b + 1), 'pl-lab muted', 13)); } return it; };
      F.add({ items: [...big(0), ...ghost(0)], say: n + ' training points. Each tree will get its own bootstrap sample of ' + n + ' rows drawn with replacement; the panels on the right will show them, and the shading here the vote of the trees grown so far.', meta: { phase: 'start', n, B } });
      let shown = [];
      for (let b = 0; b < B; b++) {
        const st = m.trace[b], distinct = st.inBag.filter((v) => v > 0).length, twice = st.inBag.filter((v) => v > 1).length;
        F.set('trees', b + 1);
        shown = shown.concat(small(b, true));
        const items = [...big(b + 1), ...ghost(b + 1), ...shown.map((it) => (it.key.startsWith('bs' + b + ':t') ? it : Object.assign({}, it, it.key.endsWith(':t') ? { cls: 'pl-lab' } : {})))];
        F.add({ items, say: 'Tree ' + (b + 1) + ' drew ' + distinct + ' different rows (' + twice + ' of them more than once, drawn larger); ' + st.oob + ' rows were never drawn. Grown to purity it has ' + st.leaves + ' leaves and scores ' + H.pct(treeAcc[b]) + ' on 300 test points; the vote of the first ' + (b + 1) + ' scores ' + H.pct(bagAcc[b]) + '.', meta: { phase: 'tree', b: b + 1, distinct, twice, oob: st.oob, leaves: st.leaves, treeAcc: treeAcc[b], bagAcc: bagAcc[b], nte: 300 } });
      }
      const mean = ML().mean(treeAcc), bestTree = Math.max(...treeAcc);
      F.add({ items: [...big(B), ...shown], say: 'On their own the ' + B + ' trees score ' + H.pct(mean) + ' on average (the best ' + H.pct(bestTree) + '). Their vote scores ' + H.pct(bagAcc[B - 1]) + ', and its boundary is smoother: the pale bands are where the trees disagree.', meta: { phase: 'done', B, mean, bestTree, bag: bagAcc[B - 1] } });
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Bagging', text: ['for b = 1 … B:', '  draw n rows with replacement', '  grow a deep tree on them', 'predict: average the trees’ votes', 'P(a row is never drawn) = (1 − 1/n)ⁿ ≈ 0.368'].join('\n') }] } };
    },
    predict: [
      { at: (f) => f.meta.phase === 'start', pick: 'option', ask: 'About what share of the rows will a bootstrap sample leave out?', options: ['about a third', 'none', 'about half', 'about a tenth'], answer: 'about a third', explain: (fr) => 'Each row is missed by all n draws with chance (1 − 1/n)ⁿ ≈ 0.37. Tree 1 left out ' + fr[1].meta.oob + ' of ' + fr[0].meta.n + '.' },
      { at: (f) => f.meta.phase === 'tree' && f.meta.b === 2, pick: 'option', ask: 'Will the vote of all the trees beat the average single tree on the test points?', options: ['yes', 'no'], answer: (fr) => (fr[fr.length - 1].meta.bag > fr[fr.length - 1].meta.mean ? 'yes' : 'no'), explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'A bootstrap sample of n rows is…', options: ['n rows drawn with replacement', 'n/2 rows without replacement', 'the rows in a random order', 'every row twice'], answer: 0, why: 'Some rows come more than once and others not at all.' },
      { q: 'Bagging mostly reduces a model’s…', options: ['variance', 'bias', 'training time', 'number of features'], answer: 0, why: 'Averaging many unstable fits steadies them; it does not make a too-simple model richer.' },
      { q: 'Why are the bagged trees grown deep?', options: ['deep trees have low bias; the vote removes their variance', 'shallow trees cannot be bagged', 'it is faster', 'to use every feature'], answer: 0, why: 'Bagging fixes variance, so start from trees with little bias.' },
      { q: 'Which model gains least from bagging?', options: ['a stable one, such as linear regression', 'a deep tree', 'a 1-nearest-neighbour rule', 'a large unpruned tree'], answer: 0, why: 'If the fit barely changes between samples, there is little variance to average away.' },
    ],
  });

  // ---------------------------------------------------------------------------------------------------------------
  // The out-of-bag score
  // ---------------------------------------------------------------------------------------------------------------
  S.topic({
    id: 'rf-oob', family: 'p05-forest', order: 30, title: 'The out-of-bag score',
    blurb: 'Every tree of a forest leaves about a third of the rows out of its bootstrap sample. For each row, collect the votes of only those trees that never saw it: that is a prediction on unseen data. Scored over all rows, it estimates the forest’s test accuracy without setting any data aside.',
    inputs: [
      { id: 'set', kind: 'choice', label: 'data', def: 'moons', options: T5.SETS },
      { id: 'n', kind: 'number', label: 'training points', def: 40, min: 16, max: 60 },
      { id: 'B', kind: 'number', label: 'trees', def: 10, min: 2, max: 12 },
      { id: 'seed', kind: 'number', label: 'seed', def: 6, min: 1, max: 9999 },
    ],
    presets: [
      { label: 'Two moons, ten trees', values: { set: 'moons', n: 40, B: 10, seed: 6 } },
      { label: 'Exam results, twelve trees', values: { set: 'exam', n: 60, B: 12, seed: 2 } },
      { label: 'Wildflowers', values: { set: 'flowers', n: 45, B: 10, seed: 3 } },
      { label: 'Only three trees: some rows get no vote', values: { set: 'circles', n: 40, B: 3, seed: 9 } },
    ],
    counters: [{ id: 'trees', label: 'trees' }, { id: 'voted', label: 'rows with a vote' }],
    run(inp) {
      const d = T5.data2(inp.set, inp.n, inp.seed), te = T5.data2(inp.set, 300, inp.seed + 1000), n = d.X.length, B = inp.B;
      const { draws, seeds } = T5.draws(n, B, inp.seed);
      const fit = (b) => ML().RandomForestClassifier({ nEstimators: b, maxFeatures: 'sqrt', bootstrap: true, oobScore: true, bootstrapIndices: draws.slice(0, b), treeSeeds: seeds.slice(0, b) }).fit(d.X, d.y);
      const m = fit(B);
      const F = new S.Frames(['trees', 'voted']);
      const order = d.X.map((_, i) => i).sort((a, b) => d.y[a] - d.y[b] || d.X[a][0] - d.X[b][0]);
      const cw = Math.min(12, 690 / n), rh = Math.min(16, 200 / B), mx = 70;
      const rowItems = (b) => {
        const w = m.inBag[b], items = [H.text('ob:l' + b, mx - 8, 40 + b * rh + rh / 2 + 4, 'tree ' + (b + 1), 'tick', 12, 'end')];
        order.forEach((i, j) => items.push({ key: 'ob:c' + b + '_' + j, kind: 'cell', x: H.r2(mx + j * cw), y: H.r2(40 + b * rh), w: H.r2(cw - 1.5), h: H.r2(rh - 2), rx: 1, text: '', cls: 'p5-bag ' + (w[i] ? 'c' + d.y[i] + (w[i] === 1 ? ' x1' : w[i] === 2 ? ' x2' : '') : 'out'), audit: 'ignore' }));
        return items;
      };
      const head = [H.text('ob:h', mx, 26, 'one row per tree, one column per training row: darker = drawn more often, dashed = out of the bag', 'tick', 12)];
      const ax = H.axes({ prefix: 'os', x: 44, y: 300, w: 300, h: 220, xr: d.xr, yr: d.yr, xname: d.names[0], grid: false });
      const cax = H.axes({ prefix: 'oc', x: 440, y: 300, w: 300, h: 200, xr: [0.5, B + 0.5], yr: [0.4, 1], xticks: B > 8 ? 2 : 1, yticks: 0.1, fmtY: (v) => Math.round(v * 100) + '%', xname: 'trees', yname: 'accuracy', grid: true });
      const legend = S.Kit.curve.legend('oc:lg', 440, 566, [{ text: 'out of bag (rows with a vote)', cls: 'c0' }, { text: 'test points', cls: 'c1' }]);
      const votes = d.X.map(() => new Array(d.K).fill(0)), nv = new Array(n).fill(0), curOob = [], curTest = [];
      let rows = [];
      F.add({ items: [...head, ...ax.items, ...S.Kit.scatter.points(ax, 'os:p', d.X, d.y, { r: 3.6, cls: () => 'hollow' }), ...cax.items, ...legend], say: n + ' training rows, no trees yet. Hollow points have no out-of-bag vote so far.', meta: { phase: 'start', n } });
      for (let b = 0; b < B; b++) {
        const w = m.inBag[b], T = m.trees[b];
        for (let i = 0; i < n; i++) if (!w[i]) { const v = T.nodes[ML().treeApply(T, d.X[i])].value; v.forEach((q, k) => (votes[i][k] += q)); nv[i]++; }
        const voted = nv.filter((v) => v > 0).length, pred = votes.map((v) => ML().argmax(v));
        const right = d.X.filter((_, i) => nv[i] > 0 && pred[i] === d.y[i]).length, oobAcc = voted ? right / voted : 0;
        const testAcc = T5.acc(te.y, te.X.map((x) => ML().argmax(T5.vote(m.trees, b + 1, x))));
        curOob.push([b + 1, oobAcc]); curTest.push([b + 1, testAcc]);
        rows = rows.concat(rowItems(b));
        const wrong = d.X.map((_, i) => i).filter((i) => nv[i] > 0 && pred[i] !== d.y[i]);
        F.set('trees', b + 1); F.set('voted', voted);
        const items = [...head, ...rows, ...ax.items, ...S.Kit.scatter.points(ax, 'os:p', d.X, d.y, { r: 3.6, cls: (i) => (nv[i] ? '' : 'hollow') }), ...wrong.map((i) => S.Kit.scatter.ring(ax, 'os:r' + i, d.X[i][0], d.X[i][1], '', 7)), ...cax.items, ...legend];
        items.push(S.Kit.curve.line(cax, 'oc:o', curOob.slice(), 'c0'), S.Kit.curve.line(cax, 'oc:t', curTest.slice(), 'c1'), T5.dot(cax, 'oc:do', b + 1, oobAcc, 'c0'), T5.dot(cax, 'oc:dt', b + 1, testAcc, 'c1'));
        const oobN = m.trace[b].oob;
        let say = 'Tree ' + (b + 1) + ' left ' + oobN + ' rows out of its sample, and votes on each of them. ' + voted + ' of ' + n + ' rows now have at least one vote; their votes are right for ' + right + ' (' + H.pct(oobAcc) + ', ringed points wrong). On 300 test points the forest so far scores ' + H.pct(testAcc) + '.';
        const meta = { phase: 'tree', b: b + 1, oob: oobN, voted, n, right, oobAcc, testAcc, nte: 300 };
        if (b === B - 1) {
          meta.unvoted = n - voted; if (voted === n) { say += ' Every row has a vote, so the out-of-bag score is ' + H.pct(m.oobScore) + '.'; meta.oobScore = m.oobScore; }
          else say += ' ' + (n - voted) + ' rows were drawn by every tree and still have no vote: more trees are needed for a full out-of-bag score.';
          meta.phase = 'last';
        }
        F.add({ items, say, meta });
      }
      return { frames: F.list, def: { sections: [{ id: 'eq', kind: 'math', title: 'Out of bag', text: ['OOB(i) = trees whose sample missed row i', 'vote(i) = average of those trees on xᵢ', 'OOB score = accuracy of vote(i) over i', 'each tree misses ≈ 36.8% of rows', 'no rows need to be held back'].join('\n') }] } };
    },
    predict: [
      { at: (f, i, fr) => f.meta.phase === 'start' && fr.length > 3, pick: 'option', ask: 'After all the trees, will the out-of-bag accuracy be within 10 points of the test accuracy?', options: ['yes', 'no'], answer: (fr) => { const L = fr[fr.length - 1].meta; return Math.abs(L.oobAcc - L.testAcc) <= 0.1 ? 'yes' : 'no'; }, explain: (fr) => fr[fr.length - 1].say },
    ],
    quiz: [
      { q: 'Which trees vote on row i for its out-of-bag prediction?', options: ['only the trees whose bootstrap sample did not contain row i', 'all the trees', 'the tree that saw it most often', 'a random half'], answer: 0, why: 'Those trees never trained on it, so their vote is an honest test.' },
      { q: 'Roughly what share of the trees vote on each row?', options: ['about 37%', 'about 63%', 'about 10%', 'all of them'], answer: 0, why: 'A row is left out of a sample with chance (1 − 1/n)ⁿ ≈ e⁻¹.' },
      { q: 'What does the out-of-bag score save?', options: ['a separate validation set', 'training time', 'memory', 'the bootstrap'], answer: 0, why: 'It is a test score computed from the training rows themselves.' },
      { q: 'With very few trees the out-of-bag score is…', options: ['noisy, and some rows may have no vote at all', 'exactly the test score', 'always 100%', 'undefined for every row'], answer: 0, why: 'Each row gets only a handful of votes, or none.' },
    ],
  });
})(globalThis.Nooral = globalThis.Nooral || {});
